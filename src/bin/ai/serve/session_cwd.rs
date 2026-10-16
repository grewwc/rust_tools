//! Per-session virtual working directories for serve-chat `/cd`.
//!
//! A serve daemon hosts many sessions in one process, and the process cwd is
//! shared. Each session that runs `/cd` gets an entry in `ServeState.cwds`;
//! the daemon spawns every turn child with `.current_dir()` set to that
//! entry, so the child's real cwd — and therefore `effective_cwd()`, every
//! relative path in file/shell tools, MCP children spawned by the child, and
//! the system prompt's working-directory line — is per session. Sessions
//! without an entry use the daemon's own cwd, which keeps the no-`/cd` case
//! byte-identical to before.
//!
//! The map is also persisted next to the session store (`serve-cwds.json`):
//! the daemon re-execs itself on `--serve-restart`, and without persistence
//! every `/cd` would silently revert to the daemon cwd — for the main turn
//! and, through it, for every subagent that inherits the turn's directory.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::ai::driver::commands::cwd::{CwdCommand, CwdOutcome, cd_usage_line, parse_cwd_command};
use crate::commonw::utils::expanduser;

use super::*;

/// File holding the persisted per-session working directories. It lives next
/// to the session store so a daemon restart (which re-execs into a fresh
/// process with an empty in-memory map) restores every session's `/cd`.
fn cwds_file(history_file: &Path) -> Option<PathBuf> {
    let parent = history_file.parent()?;
    if parent.as_os_str().is_empty() {
        return None;
    }
    Some(parent.join("serve-cwds.json"))
}

/// Load persisted per-session working directories. Best-effort: any IO or
/// parse failure yields an empty map, and entries whose directory no longer
/// exists are dropped, so a stale file can never break daemon startup.
pub(crate) fn load_cwds(history_file: &Path) -> HashMap<String, SessionCwd> {
    let Some(path) = cwds_file(history_file) else {
        return HashMap::new();
    };
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    if text.trim().is_empty() {
        return HashMap::new();
    }
    let map: HashMap<String, SessionCwd> = serde_json::from_str(&text).unwrap_or_default();
    map.into_iter()
        .filter(|(_, entry)| entry.current.is_dir())
        .collect()
}

/// Persist the in-memory map. Best-effort: failures are logged and never fail
/// the `/cd` command or the turn they belong to.
fn save_cwds(state: &ServeState) {
    let Some(path) = cwds_file(&state.history_file) else {
        return;
    };
    let text = match state.cwds.lock().ok().map(|guard| {
        serde_json::to_string(&*guard).unwrap_or_else(|_| "{}".to_string())
    }) {
        Some(text) => text,
        None => return,
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(err) = std::fs::write(&path, text) {
        eprintln!("[serve] warning: failed to persist session cwds: {err}");
    }
}

/// The session's working directory, or the daemon's own when the session
/// never ran `/cd` — or when its recorded directory no longer exists (stale
/// entry after the directory was moved or deleted).
pub(crate) fn session_cwd(state: &ServeState, id: &str) -> PathBuf {
    let stored = state
        .cwds
        .lock()
        .ok()
        .and_then(|guard| guard.get(id).cloned())
        .map(|entry| entry.current);
    match stored {
        Some(dir) if dir.is_dir() => dir,
        _ => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    }
}

/// Apply a `/cd`/`:cd` line to the session's virtual cwd without touching the
/// daemon process. Returns the message lines the local switch would print, so
/// the HTTP response reads identically. `None` when the line is not a cd
/// command.
pub(crate) fn run_session_cd(state: &ServeState, id: &str, command: &str) -> Option<CwdOutcome> {
    let outcome = match parse_cwd_command(command)? {
        CwdCommand::Show => show_outcome(state, id),
        CwdCommand::GoBack => go_back_outcome(state, id),
        CwdCommand::Switch(target) => switch_outcome(state, id, &target),
    };
    Some(outcome)
}

/// Forget a session's cwd entry (session deleted) and persist the removal.
pub(crate) fn drop_session_cwd(state: &ServeState, id: &str) {
    if let Ok(mut guard) = state.cwds.lock() {
        guard.remove(id);
    }
    save_cwds(state);
}

/// Carry a session's working directory across a fork: the branch starts where
/// its parent was, instead of silently reverting to the daemon cwd.
pub(crate) fn copy_session_cwd(state: &ServeState, src_id: &str, dst_id: &str) {
    let entry = state
        .cwds
        .lock()
        .ok()
        .and_then(|guard| guard.get(src_id).cloned());
    if let Some(entry) = entry {
        if let Ok(mut guard) = state.cwds.lock() {
            guard.insert(dst_id.to_string(), entry);
        }
        save_cwds(state);
    }
}

fn show_outcome(state: &ServeState, id: &str) -> CwdOutcome {
    CwdOutcome {
        stdout: format!("Working directory: {}\n", session_cwd(state, id).display()),
        stderr: String::new(),
    }
}

fn go_back_outcome(state: &ServeState, id: &str) -> CwdOutcome {
    let previous = state
        .cwds
        .lock()
        .ok()
        .and_then(|guard| guard.get(id).and_then(|entry| entry.previous.clone()));
    match previous {
        Some(dir) => apply_switch(state, id, &dir),
        None => CwdOutcome {
            stdout: "No previous working directory recorded yet.\n".to_string(),
            stderr: String::new(),
        },
    }
}

fn switch_outcome(state: &ServeState, id: &str, target: &str) -> CwdOutcome {
    let current = session_cwd(state, id);
    let expanded = expanduser(target).to_string();
    let raw = PathBuf::from(&expanded);
    // Relative targets resolve against the *session's* cwd, never the daemon's.
    let resolved = if raw.is_absolute() {
        raw
    } else {
        current.join(raw)
    };
    if !resolved.is_dir() {
        return CwdOutcome {
            stdout: String::new(),
            stderr: format!(
                "[cd] not a directory: {}\n{}",
                resolved.display(),
                cd_usage_line()
            ),
        };
    }
    apply_switch(state, id, &resolved)
}

/// Record the switch in the map. The stored path is canonicalized so the
/// printed value is the absolute, symlink-resolved directory the next turn
/// child's tools will resolve against (the property `getcwd` gives the local
/// switch).
fn apply_switch(state: &ServeState, id: &str, dir: &Path) -> CwdOutcome {
    let now = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let previous = session_cwd(state, id);
    if let Ok(mut guard) = state.cwds.lock() {
        guard.insert(
            id.to_string(),
            super::SessionCwd {
                current: now.clone(),
                previous: Some(previous.clone()),
            },
        );
    }
    save_cwds(state);
    CwdOutcome {
        stdout: format!(
            "Working directory: {} (was {})\n",
            now.display(),
            previous.display()
        ),
        stderr: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Standalone state: no session ever ran `/cd`, so lookups fall back to
    /// the daemon cwd.
    fn test_state() -> ServeState {
        ServeState {
            history_file: std::env::temp_dir().join(format!(
                "a-serve-session-cwd-test-{}.sqlite",
                std::process::id()
            )),
            token: String::new(),
            cwds: Arc::new(std::sync::Mutex::new(HashMap::new())),
            locks: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            active_turns: Arc::new(std::sync::Mutex::new(HashMap::new())),
            confirms: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("a-serve-session-cwd-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// `current_dir()` returns the physical path while `temp_dir()` keeps the
    /// `TMPDIR` spelling (`/var/...` vs `/private/var/...` on macOS), so both
    /// sides are canonicalized before comparing.
    fn cwd() -> PathBuf {
        std::fs::canonicalize(std::env::current_dir().unwrap()).unwrap()
    }

    #[test]
    fn absent_session_uses_daemon_cwd() {
        let state = test_state();
        assert_eq!(session_cwd(&state, "s1"), cwd());
    }

    #[test]
    fn switch_sets_session_cwd_and_prints_local_message() {
        let state = test_state();
        let target = temp_dir("switch");
        let outcome =
            run_session_cd(&state, "s1", &format!("/cd {}", target.display())).expect("cd command");
        assert!(outcome.stderr.is_empty());
        assert!(
            outcome
                .stdout
                .contains(&format!("(was {})\n", cwd().display()))
        );
        assert_eq!(session_cwd(&state, "s1"), target.canonicalize().unwrap());
        // Bare `/cd` prints the session cwd.
        let shown = run_session_cd(&state, "s1", "/cd").expect("cd command");
        assert_eq!(
            shown.stdout,
            format!(
                "Working directory: {}\n",
                target.canonicalize().unwrap().display()
            )
        );
    }

    #[test]
    fn go_back_returns_to_previous_dir() {
        let state = test_state();
        let first = temp_dir("back-a");
        let second = temp_dir("back-b");
        run_session_cd(&state, "s1", &format!("/cd {}", first.display())).expect("cd");
        run_session_cd(&state, "s1", &format!("/cd {}", second.display())).expect("cd");
        let back = run_session_cd(&state, "s1", "/cd -").expect("cd");
        assert_eq!(session_cwd(&state, "s1"), first.canonicalize().unwrap());
        assert!(back.stdout.contains(&format!(
            "(was {})\n",
            second.canonicalize().unwrap().display()
        )));
    }

    #[test]
    fn go_back_without_history_prints_local_message() {
        let state = test_state();
        let outcome = run_session_cd(&state, "s1", "/cd -").expect("cd");
        assert_eq!(
            outcome.stdout,
            "No previous working directory recorded yet.\n"
        );
    }

    #[test]
    fn not_a_directory_reports_error_with_usage() {
        let state = test_state();
        let outcome = run_session_cd(&state, "s1", "/cd /definitely/not/a/real/dir").expect("cd");
        assert!(outcome.stdout.is_empty());
        assert!(outcome.stderr.contains("[cd] not a directory:"));
        assert!(outcome.stderr.contains("Usage: /cd [dir]"));
        assert_eq!(session_cwd(&state, "s1"), cwd());
    }

    #[test]
    fn relative_target_resolves_against_session_cwd() {
        let state = test_state();
        let session_base = temp_dir("rel-base");
        std::fs::create_dir_all(session_base.join("sub")).unwrap();
        // Move the session to `session_base` first; the daemon cwd has no
        // `sub` entry, so a successful `/cd sub` proves the relative target
        // resolved against the session cwd, not the daemon's.
        run_session_cd(&state, "s1", &format!("/cd {}", session_base.display())).expect("cd");
        let outcome = run_session_cd(&state, "s1", "/cd sub").expect("cd");
        assert!(outcome.stderr.is_empty(), "{}", outcome.stderr);
        assert_eq!(
            session_cwd(&state, "s1"),
            session_base.join("sub").canonicalize().unwrap()
        );
    }

    #[test]
    fn non_cd_command_returns_none() {
        let state = test_state();
        assert!(run_session_cd(&state, "s1", "/model foo").is_none());
        assert!(run_session_cd(&state, "s1", "cd /tmp").is_none());
        assert!(run_session_cd(&state, "s1", "").is_none());
    }

    #[test]
    fn sessions_are_isolated() {
        let state = test_state();
        let first = temp_dir("iso-a");
        let second = temp_dir("iso-b");
        run_session_cd(&state, "s1", &format!("/cd {}", first.display())).expect("cd");
        run_session_cd(&state, "s2", &format!("/cd {}", second.display())).expect("cd");
        assert_eq!(session_cwd(&state, "s1"), first.canonicalize().unwrap());
        assert_eq!(session_cwd(&state, "s2"), second.canonicalize().unwrap());
    }

    #[test]
    fn drop_session_restores_daemon_cwd() {
        let state = test_state();
        let target = temp_dir("drop");
        run_session_cd(&state, "s1", &format!("/cd {}", target.display())).expect("cd");
        assert_ne!(session_cwd(&state, "s1"), cwd());
        drop_session_cwd(&state, "s1");
        assert_eq!(session_cwd(&state, "s1"), cwd());
    }

    /// State with an isolated history file, so persistence tests never share
    /// the `serve-cwds.json` sidecar with other tests or a live daemon.
    fn persist_state(tag: &str) -> ServeState {
        let dir = temp_dir(tag);
        ServeState {
            history_file: dir.join("history.sqlite"),
            token: String::new(),
            cwds: Arc::new(std::sync::Mutex::new(HashMap::new())),
            locks: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            active_turns: Arc::new(std::sync::Mutex::new(HashMap::new())),
            confirms: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    /// A daemon restart (fresh in-memory map + `load_cwds`) keeps the session's
    /// `/cd`: without this the next turn — and every subagent inheriting from
    /// it — silently reverts to the daemon cwd.
    #[test]
    fn persisted_cwds_survive_daemon_restart() {
        let state = persist_state("persist-restart");
        let target = temp_dir("persist-target");
        run_session_cd(&state, "s1", &format!("/cd {}", target.display())).expect("cd");
        let canonical = target.canonicalize().unwrap();
        assert_eq!(session_cwd(&state, "s1"), canonical);
        // Simulate a restart: empty map, same history file.
        let restarted = persist_state("persist-restart");
        let loaded = load_cwds(&restarted.history_file);
        *restarted.cwds.lock().unwrap() = loaded;
        assert_eq!(session_cwd(&restarted, "s1"), canonical);
    }

    /// A recorded directory that no longer exists (moved/deleted between
    /// turns) falls back to the daemon cwd instead of spawning the turn
    /// child into a missing directory.
    #[test]
    fn stale_entry_falls_back_to_daemon_cwd() {
        let state = persist_state("persist-stale");
        let missing = temp_dir("persist-stale-missing").join("gone");
        assert!(!missing.exists());
        state.cwds.lock().unwrap().insert(
            "s1".to_string(),
            SessionCwd {
                current: missing,
                previous: None,
            },
        );
        assert_eq!(session_cwd(&state, "s1"), cwd());
    }

    /// Forking carries the parent's working directory to the branch.
    #[test]
    fn fork_keeps_parent_cwd() {
        let state = persist_state("persist-fork");
        let target = temp_dir("persist-fork-target");
        run_session_cd(&state, "s1", &format!("/cd {}", target.display())).expect("cd");
        copy_session_cwd(&state, "s1", "s2");
        assert_eq!(
            session_cwd(&state, "s2"),
            target.canonicalize().unwrap()
        );
    }
}
