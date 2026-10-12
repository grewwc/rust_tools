//! Completion-claim persistence: the `complete` tool records the model's
//! structured "all work is done" assertion as session state, and the
//! outstanding-work gate checks it against the plan at `FinalResponse` time.
//!
//! A claim is single-use and self-validating: it snapshots the plan's
//! `updated_at_ms` when recorded, so any later `plan` / `plan_update` silently
//! invalidates it — no turn-start hook or cross-turn bookkeeping is needed. The
//! gate consumes the file on every read, so an assertion never survives into a
//! later round; the model must re-assert after doing more work.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::ai::driver::side_note::assets_dir_for_history;
use crate::ai::types::App;

pub(crate) const COMPLETION_CLAIM_FILE_NAME: &str = "completion-claim.json";

/// A structured completion assertion recorded by the `complete` tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct CompletionClaim {
    pub(crate) schema_version: u32,
    pub(crate) summary: String,
    /// `PlanState::updated_at_ms` snapshot (`None` when no plan existed).
    /// The gate treats a claim as fresh only when this still matches.
    pub(crate) plan_updated_at_ms: Option<u64>,
    pub(crate) created_at_ms: u64,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Persistent path of the session's completion claim (under the session assets root).
pub(crate) fn completion_claim_path(app: &App) -> PathBuf {
    assets_dir_for_history(&app.session_history_file).join(COMPLETION_CLAIM_FILE_NAME)
}

/// Write mutex for the claim file: `record` / `consume` must be atomic, or
/// parallel tool rounds could read a half-written claim. Only guards threads
/// inside this process; cross-process safety comes from the atomic tmp+rename
/// (last writer wins).
static CLAIM_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn record_completion_claim(
    app: &App,
    summary: &str,
    plan_updated_at_ms: Option<u64>,
) -> Result<CompletionClaim, String> {
    let _guard = CLAIM_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let claim = CompletionClaim {
        schema_version: 1,
        summary: summary.to_string(),
        plan_updated_at_ms,
        created_at_ms: now_ms(),
    };
    let path = completion_claim_path(app);
    let parent = path
        .parent()
        .ok_or_else(|| "invalid completion-claim path".to_string())?;
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("cannot create completion-claim dir: {e}"))?;
    let tmp = parent.join(format!(
        ".{}.{}.tmp",
        COMPLETION_CLAIM_FILE_NAME,
        std::process::id()
    ));
    let bytes =
        serde_json::to_vec_pretty(&claim).map_err(|e| format!("serialize completion claim: {e}"))?;
    if let Err(e) = std::fs::write(&tmp, &bytes) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("write completion claim {}: {e}", path.display()));
    }
    std::fs::rename(&tmp, &path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("persist completion claim {}: {e}", path.display())
    })?;
    Ok(claim)
}

pub(crate) fn load_completion_claim(app: &App) -> Result<Option<CompletionClaim>, String> {
    let path = completion_claim_path(app);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("read completion claim {}: {e}", path.display())),
    };
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|e| format!("completion claim {} is corrupt: {e}", path.display()))
}

/// Single-use consumption: remove the claim file if present. Best-effort — a
/// missing file is the normal "no assertion" state, and I/O failures here must
/// not block the turn (the gate then simply sees no claim).
pub(crate) fn consume_completion_claim(app: &App) {
    let _guard = CLAIM_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    let _ = std::fs::remove_file(completion_claim_path(app));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_app() -> (App, std::path::PathBuf) {
        let base = std::env::temp_dir().join(format!(
            "completion-claim-test-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let mut app = crate::ai::middleware::test_util::test_app();
        app.session_history_file = base.join("session.sqlite");
        (app, base)
    }

    #[test]
    fn claim_roundtrip_and_consume() {
        let (app, base) = test_app();
        assert!(load_completion_claim(&app).unwrap().is_none());
        let claim = record_completion_claim(&app, "all done", Some(42)).unwrap();
        assert_eq!(claim.summary, "all done");
        assert_eq!(claim.plan_updated_at_ms, Some(42));
        let loaded = load_completion_claim(&app).unwrap().unwrap();
        assert_eq!(loaded, claim);
        consume_completion_claim(&app);
        assert!(load_completion_claim(&app).unwrap().is_none());
        // Consuming twice is a silent no-op, never an error.
        consume_completion_claim(&app);
        let _ = std::fs::remove_dir_all(&base);
    }
}
