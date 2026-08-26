//! `/proc` command: show the currently running sessions.
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
    let mut by_sid: BTreeMap<String, ActiveSession> = BTreeMap::new();
    let mut registered_pids: BTreeSet<i32> = BTreeSet::new();

    // 1) PID files (scan the base directory and all *.sessions subdirectories)
    for (sid, pid, alive) in session_pid::scan_all_session_pids(sessions_root)? {
        if alive {
            registered_pids.insert(pid);
            by_sid.entry(sid.clone()).or_insert(ActiveSession {
                session_id: sid,
                pid,
                source: "pid-file",
            });
        }
    }

    // 2) lsof fallback (query only unregistered a processes for open .sqlite
    //    files, avoiding a recursive lsof +D scan)
    let a_pids = session_pid::list_a_pids();
    let total_a = a_pids.len().saturating_sub(1); // minus ourselves
    let unregistered: Vec<i32> = a_pids
        .into_iter()
        .filter(|&p| p != current_pid && !registered_pids.contains(&p))
        .collect();
    for (sid, pid) in session_pid::discover_lsof_sessions(sessions_root, &unregistered) {
        by_sid.entry(sid.clone()).or_insert(ActiveSession {
            session_id: sid,
            pid,
            source: "lsof",
        });
    }

    // Exclude our own process: `a /proc` is a one-shot query, not a real active session.
    let sessions: Vec<&ActiveSession> = by_sid.values().filter(|s| s.pid != current_pid).collect();
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

    println!("Active sessions ({identified}):");
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

fn print_proc_help() {
    println!("/proc commands:");
    println!();
    println!("  /proc                     show running sessions (interactive + background)");
    println!("  /proc list                same as /proc");
    println!("  /proc help                show this help message");
    println!();
    println!("Note: sessions suspended via /bg or /suspend are NOT shown here -");
    println!("      their processes have exited. Use /sessions list to see all saved sessions.");
    println!();
}
