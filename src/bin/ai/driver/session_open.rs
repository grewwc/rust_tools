//! Process-wide owner of the current session's open lease.
//!
//! A terminal client keeps the lease for as long as it has the session open.
//! Serve turn children and subagents must not take it: their parent client
//! already owns the session, and their writes are covered by the turn lease.

use std::io;
use std::path::Path;
use std::sync::Mutex;

use crate::ai::history::turn_lease::SessionOpenLease;

static CURRENT: Mutex<Option<(std::path::PathBuf, SessionOpenLease)>> = Mutex::new(None);

/// Take the open lease for `history_file`, replacing the process's previous
/// session lease. On failure the previous lease stays in place.
pub(in crate::ai) fn hold_if_top_level(
    history_file: &Path,
    session_id: &str,
) -> io::Result<()> {
    if crate::ai::background::is_serve_child()
        || crate::ai::driver::runtime_ctx::current_subagent_depth() > 0
    {
        return Ok(());
    }
    let mut current = CURRENT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // Re-holding the session already owned by this process is a no-op. A fresh
    // flock on a second descriptor would otherwise make the process collide
    // with itself when `/resume` names the current session.
    if current
        .as_ref()
        .is_some_and(|(path, _)| path == history_file)
    {
        return Ok(());
    }
    let Some(next) = SessionOpenLease::acquire_for_history(history_file, session_id)? else {
        *current = None;
        return Ok(());
    };
    *current = Some((history_file.to_path_buf(), next));
    Ok(())
}