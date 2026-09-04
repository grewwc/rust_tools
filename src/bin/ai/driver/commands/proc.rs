//! `/proc` command: show currently running sessions and published subagent status.
//!
//! Inspired by the Unix `/proc` filesystem: it summarizes all "live" sessions
//! in one table.
//!
//! Data sources (triple probe, falling back layer by layer):
//! 1. **PID files** (`<id>.pid` in the sessions directory): written automatically
//!    when a newer `a` starts, removed automatically on exit. Most reliable.
//! 2. **`lsof` scan**: only for `a` processes not registered via a PID file,
//!    query the `.sqlite` files they have open (`lsof -p`), rather than
//!    recursively scanning the whole sessions directory. Covers sessions started
//!    by older versions of `a` (they do not write PID files), though idle
//!    older-version sessions waiting for input may be missed.
//! 3. **`pgrep` count**: counts all processes named `a`, to hint
//!    "N `a` processes are running, but only M sessions were identified".
//!
//! Note: sessions suspended via `/bg`, `/suspend`, etc. are **not** active:
//! their processes have exited, only their state was saved for later recovery.
//! Use `/sessions list` to see all saved sessions.

use std::collections::{BTreeMap, BTreeSet};
use std::process;

use crate::ai::{driver::session_pid, history::SessionStore, types::App};

/// Merged active-session record.
struct ActiveSession {
    session_id: String,
    pid: i32,
    source: &'static str, // "pid-file" / "lsof"
}

pub fn try_handle_proc_command(app: &App, input: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let normalized = if let Some(rest) = trimmed.strip_prefix('/') {
        rest
    } else if let Some(rest) = trimmed.strip_prefix(':') {
        rest
    } else {
        return Ok(false);
    };
    let mut parts = normalized.split_whitespace();
    let Some(cmd) = parts.next() else {
        return Ok(false);
    };
    if cmd != "proc" {
        return Ok(false);
    }
    let action = parts.next().unwrap_or("list");
    if matches!(action, "help" | "h") {
        print_proc_help();
        return Ok(true);
    }
    if !matches!(action, "list" | "ls" | "") {
        println!("Unknown /proc subcommand: {action}");
        println!("Run /proc help for usage.");
        return Ok(true);
    }

    let store = SessionStore::new(app.config.history_file.as_path());
    let _ = store.ensure_root_dir();
    let current_pid = process::id() as i32;
    let sessions_root = store.sessions_root();

    // ---- Collect active sessions (triple probe) ----
    let mut by_session_pid: BTreeMap<(String, i32), ActiveSession> = BTreeMap::new();
    let mut registered_pids: BTreeSet<i32> = BTreeSet::new();

    // 1) PID files (scan the base directory and all *.sessions subdirectories)
    for (sid, pid, alive) in session_pid::scan_all_session_pids(sessions_root)? {
        if alive {
            registered_pids.insert(pid);
            insert_active_session(&mut by_session_pid, sid, pid, "pid-file");
        }
    }

    // 2) lsof fallback (query only unregistered a processes for open .sqlite
    //    files, avoiding a recursive lsof +D scan)
    let a_pids = session_pid::list_a_pids();
    let total_a = a_pids.iter().filter(|pid| **pid != current_pid).count();
    let unregistered: Vec<i32> = a_pids
        .into_iter()
        .filter(|&p| p != current_pid && !registered_pids.contains(&p))
        .collect();
    for (sid, pid) in session_pid::discover_lsof_sessions(sessions_root, &unregistered) {
        insert_active_session(&mut by_session_pid, sid, pid, "lsof");
    }

    // Exclude our own process: `a /proc` is a one-shot query, not a real active session.
    let sessions: Vec<&ActiveSession> = by_session_pid
        .values()
        .filter(|s| s.pid != current_pid)
        .collect();
    let identified = sessions.len();

    if sessions.is_empty() {
        println!("No active sessions identified.");
        if total_a > 0 {
            println!(
                "  (but {total_a} `a` process(es) detected via pgrep - possibly started with an older version)"
            );
        }
        return Ok(true);
    }

    println!("Active main agents ({identified}):");
    println!();

    // Query the tty of all active sessions in one batch: a single `ps` call
    // instead of per-process fork/exec.
    let active_pids: Vec<i32> = sessions.iter().map(|s| s.pid).collect();
    let tty_map = session_pid::tty_map_for_pids(&active_pids);

    // Read previews (summary + modified) for active sessions only. Do not use
    // list_sessions(): it opens the .sqlite of every saved session and
    // recursively counts each session's assets/checkpoints directory sizes,
    // while /proc only needs a text preview of a few active sessions.
    let previews: BTreeMap<String, (Option<String>, Option<String>)> = sessions
        .iter()
        .map(|s| {
            let (summary, modified) = store
                .read_session_preview(&s.session_id)
                .ok()
                .flatten()
                .unwrap_or((None, None));
            let modified = modified.map(|t| t.format("%Y-%m-%d %H:%M").to_string());
            (s.session_id.clone(), (summary, modified))
        })
        .collect();

    for s in &sessions {
        let tag = if *tty_map.get(&s.pid).unwrap_or(&false) {
            "interactive"
        } else {
            "background"
        };

        let (summary, modified) = previews.get(&s.session_id).cloned().unwrap_or((None, None));

        println!("  [{tag:<11}]  pid={:<8}  session={}", s.pid, s.session_id);
        if let Some(m) = &modified {
            println!("                modified: {m}");
        }
        println!(
            "                summary : {}",
            summary.as_deref().unwrap_or("-")
        );
        if s.source == "lsof" {
            println!("                source  : lsof (no pid-file, possibly older version)");
            println!(
                "                agents  : unavailable (process does not publish live status)"
            );
        } else {
            let snapshots = session_pid::read_agent_snapshots(sessions_root, &s.session_id, s.pid);
            for line in format_subagent_tree(&snapshots) {
                println!("                {line}");
            }
        }
        println!();
    }

    // Hint about unidentified processes
    if total_a > identified {
        let diff = total_a - identified;
        println!("Note: {diff} additional `a` process(es) running but session not identified");
        println!("      (likely started with an older version without pid-file support)");
        println!();
    }

    Ok(true)
}

fn insert_active_session(
    sessions: &mut BTreeMap<(String, i32), ActiveSession>,
    session_id: String,
    pid: i32,
    source: &'static str,
) {
    sessions
        .entry((session_id.clone(), pid))
        .or_insert(ActiveSession {
            session_id,
            pid,
            source,
        });
}

fn format_subagent_tree(snapshots: &[session_pid::AgentSnapshot]) -> Vec<String> {
    if snapshots.is_empty() {
        return vec!["agents  : main agent (no active subagents)".to_string()];
    }

    let mut lines = vec![format!(
        "agents  : main agent + {} subagent(s)",
        snapshots.len()
    )];
    for (index, snapshot) in snapshots.iter().enumerate() {
        let last = index + 1 == snapshots.len();
        let branch = if last { "└─" } else { "├─" };
        let continuation = if last { "   " } else { "│  " };
        let agent = compact_agent_field(&snapshot.agent_name, 32);
        let state = compact_agent_field(&snapshot.state, 18);
        lines.push(format!(
            "{branch} {agent}  [{state} · {}]",
            format_elapsed(snapshot.elapsed_secs)
        ));
        lines.push(format!(
            "{continuation}  {}",
            compact_agent_field(&snapshot.description, 92)
        ));
        if let Some(progress) = snapshot
            .progress
            .as_deref()
            .filter(|value| !value.is_empty())
        {
            lines.push(format!(
                "{continuation}  {}",
                compact_agent_field(progress, 92)
            ));
        }
    }
    lines
}

fn compact_agent_field(value: &str, limit: usize) -> String {
    let mut clean = String::with_capacity(value.len());
    let mut pending_space = false;
    for character in value.chars() {
        if character.is_control() || character.is_whitespace() {
            pending_space = !clean.is_empty();
            continue;
        }
        if pending_space {
            clean.push(' ');
            pending_space = false;
        }
        clean.push(character);
    }
    if clean.is_empty() {
        return "-".to_string();
    }
    let mut shortened = clean.chars().take(limit).collect::<String>();
    if clean.chars().nth(limit).is_some() {
        shortened.push('…');
    }
    shortened
}

fn format_elapsed(seconds: u64) -> String {
    let hours = seconds / 3_600;
    let minutes = (seconds % 3_600) / 60;
    let seconds = seconds % 60;
    if hours > 0 {
        format!("{hours}h {minutes:02}m")
    } else if minutes > 0 {
        format!("{minutes}m {seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

fn print_proc_help() {
    println!("/proc commands:");
    println!();
    println!("  /proc                     show running sessions and published subagent status");
    println!("  /proc list                same as /proc");
    println!("  /proc help                show this help message");
    println!();
    println!("Note: sessions suspended via /bg or /suspend are NOT shown here -");
    println!("      their processes have exited. Use /sessions list to see all saved sessions.");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_snapshot_tree_is_hierarchical_and_safe_for_terminal_output() {
        let snapshots = vec![
            session_pid::AgentSnapshot {
                agent_name: "researcher".to_string(),
                description: "Trace the\ncurrent implementation".to_string(),
                state: "running".to_string(),
                elapsed_secs: 75,
                progress: Some("reading\tproc.rs".to_string()),
            },
            session_pid::AgentSnapshot {
                agent_name: "reviewer".to_string(),
                description: "Check the final patch".to_string(),
                state: "waiting".to_string(),
                elapsed_secs: 3,
                progress: None,
            },
        ];

        assert_eq!(
            format_subagent_tree(&snapshots),
            vec![
                "agents  : main agent + 2 subagent(s)",
                "├─ researcher  [running · 1m 15s]",
                "│    Trace the current implementation",
                "│    reading proc.rs",
                "└─ reviewer  [waiting · 3s]",
                "     Check the final patch",
            ]
        );
    }

    #[test]
    fn agent_snapshot_sessions_keep_distinct_main_pids() {
        let mut sessions = BTreeMap::new();
        insert_active_session(&mut sessions, "shared-session".to_string(), 101, "pid-file");
        insert_active_session(&mut sessions, "shared-session".to_string(), 202, "pid-file");

        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[&("shared-session".to_string(), 101)].pid, 101);
        assert_eq!(sessions[&("shared-session".to_string(), 202)].pid, 202);
    }
}
