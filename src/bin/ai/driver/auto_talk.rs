//! Headless bot-to-bot talk loop (`--auto-talk`).
//!
//! An interactive session parked at the input prompt never drains its talk
//! inbox: the drain only runs at iteration tops and inside the pending-talk
//! branch, both of which require a running turn. So two bots can never start
//! chatting on their own. This module holds the small pure pieces of the
//! opt-in headless loop that polls the inbox instead of blocking on terminal
//! input: limit resolution, the re-entrancy guard, and the reply-directive
//! framing. The loop glue itself lives in the driver `run_loop` so every turn
//! still flows through the exact same `run_turn` pipeline (and the same
//! pending-talk question format) as interactive mode.

use std::path::Path;

pub const DEFAULT_POLL_SECS: u64 = 5;
pub const MIN_POLL_SECS: u64 = 1;
pub const MAX_POLL_SECS: u64 = 300;

pub const DEFAULT_MAX_TURNS: u32 = 30;
pub const MIN_MAX_TURNS: u32 = 1;
pub const MAX_MAX_TURNS: u32 = 1000;

pub const DEFAULT_IDLE_SECS: u64 = 180;
pub const MIN_IDLE_SECS: u64 = 10;
pub const MAX_IDLE_SECS: u64 = 3600;

/// Resolved `--auto-talk` runtime limits: poll interval while the inbox is
/// empty, hard cap on model turns (cost), and idle exit without new mail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoTalkLimits {
    pub poll_secs: u64,
    pub max_turns: u32,
    pub idle_secs: u64,
}

/// Clamp raw CLI values into their safe ranges; raw values outside the range
/// (including 0 from a failed parse fallback) can never disable the guardrails.
pub fn resolve_limits(poll_secs: u64, max_turns: u32, idle_secs: u64) -> AutoTalkLimits {
    AutoTalkLimits {
        poll_secs: poll_secs.clamp(MIN_POLL_SECS, MAX_POLL_SECS),
        max_turns: max_turns.clamp(MIN_MAX_TURNS, MAX_MAX_TURNS),
        idle_secs: idle_secs.clamp(MIN_IDLE_SECS, MAX_IDLE_SECS),
    }
}

/// Refuse to start the loop when another live process already owns the
/// session: two writers would interleave history rows in the same sqlite
/// file. `own_pid` (this process, whose marker was registered at startup) is
/// always exempt, so a normal single-owner start passes.
pub fn check_auto_talk_startup(
    sessions_root: &Path,
    session_id: &str,
    own_pid: i32,
) -> Result<(), String> {
    let live = super::session_pid::scan_all_session_pids(sessions_root)
        .map_err(|err| format!("--auto-talk: cannot scan live sessions: {err}"))?;
    if let Some((_, pid, _)) = live
        .into_iter()
        .find(|(sid, pid, alive)| *alive && sid == session_id && *pid != own_pid)
    {
        return Err(format!(
            "--auto-talk: session '{session_id}' is already owned by live process pid={pid}; stop it first"
        ));
    }
    Ok(())
}

/// Directive appended to auto-talk questions so the model answers peer mail
/// with the `talk_send` tool instead of only chatting into its own history.
pub const AUTO_TALK_DIRECTIVE: &str = "(auto-talk mode: the messages above are from peer sessions. If you have something to say, reply with the talk_send tool to the sender's session id; if there is nothing to add, answer briefly without calling any tools.)";

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn resolve_limits_passes_through_sane_values() {
        let limits = resolve_limits(5, 30, 180);
        assert_eq!(
            limits,
            AutoTalkLimits {
                poll_secs: 5,
                max_turns: 30,
                idle_secs: 180,
            }
        );
    }

    #[test]
    fn resolve_limits_clamps_out_of_range() {
        let limits = resolve_limits(0, 0, 0);
        assert_eq!(limits.poll_secs, MIN_POLL_SECS);
        assert_eq!(limits.max_turns, MIN_MAX_TURNS);
        assert_eq!(limits.idle_secs, MIN_IDLE_SECS);
        let limits = resolve_limits(u64::MAX, u32::MAX, u64::MAX);
        assert_eq!(limits.poll_secs, MAX_POLL_SECS);
        assert_eq!(limits.max_turns, MAX_MAX_TURNS);
        assert_eq!(limits.idle_secs, MAX_IDLE_SECS);
    }

    fn temp_sessions_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "auto_talk_test_{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // Pin `resolve_sessions_base` to this dir: otherwise a stray
        // `*.sqlite` / `*.sessions` entry in the temp parent can promote the
        // base to the parent and the pid files below become invisible.
        std::fs::create_dir_all(dir.join("probe.sessions")).unwrap();
        dir
    }

    #[test]
    fn startup_guard_allows_unowned_session() {
        let dir = temp_sessions_dir();
        let own = std::process::id() as i32;
        assert!(check_auto_talk_startup(&dir, "fresh-session", own).is_ok());
    }

    #[test]
    fn startup_guard_rejects_other_live_owner() {
        let dir = temp_sessions_dir();
        let other = std::process::id() as i32;
        std::fs::write(dir.join(format!("bot-session.{other}.pid")), other.to_string()).unwrap();
        let err = check_auto_talk_startup(&dir, "bot-session", other.wrapping_add(1)).unwrap_err();
        assert!(err.contains("already owned"), "unexpected error: {err}");
        assert!(err.contains("bot-session"), "unexpected error: {err}");
    }

    #[test]
    fn startup_guard_exempts_own_marker() {
        let dir = temp_sessions_dir();
        let own = std::process::id() as i32;
        std::fs::write(dir.join(format!("bot-session.{own}.pid")), own.to_string()).unwrap();
        assert!(check_auto_talk_startup(&dir, "bot-session", own).is_ok());
    }

    #[test]
    fn startup_guard_ignores_dead_pid_files() {
        let dir = temp_sessions_dir();
        let own = std::process::id() as i32;
        // A PID that cannot be alive: the liveness probe fails and the stale
        // file must neither block startup nor fail the scan.
        std::fs::write(dir.join("bot-session.2000000000.pid"), "2000000000").unwrap();
        assert!(check_auto_talk_startup(&dir, "bot-session", own).is_ok());
    }
}
