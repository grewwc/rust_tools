//! Plan state persistence: tool entry points explicitly pass `&App` (session context
//! resolution lives in the driver); the session asset path is derived from the same
//! helper as side_note (`driver::side_note::assets_dir_for_history`); atomic writes
//! (temp file + rename) plus an in-process mutex prevent concurrent read-modify-write
//! races on the state file.
//!
//! Layer responsibilities: `model` is pure data (no I/O, no rendering), `store` is
//! persistence only, `render` produces user-facing text. `update_plan_step` enriches
//! the "step not found" error with the full rendered plan so it is self-healing:
//! after context compression folds the plan-creation turns away, the model has no
//! other way to learn the real step numbers (regression: hallucinated step 6/5
//! updates in a muse-spark session that ended with a fabricated plan narrative).

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;

use crate::ai::driver::side_note::assets_dir_for_history;
use crate::ai::types::App;

use super::model::{PlanState, StepStatus, StepTransition};

pub(crate) const PLAN_STATE_FILE_NAME: &str = "plan-state.json";

/// Persistent path of the session's active plan (under the session assets root).
pub(crate) fn plan_state_path(app: &App) -> PathBuf {
    // Derived from the same session assets root as side_note / checkpoint:
    // `session_history_file` is `<sessions_root>/<id>.sqlite`; its parent dir + stem
    // give `<sessions_root>/<id>.assets` (see `assets_dir_for_history`).
    assets_dir_for_history(&app.session_history_file).join(PLAN_STATE_FILE_NAME)
}

pub(crate) fn load_plan_state(app: &App) -> Result<Option<PlanState>, String> {
    let path = plan_state_path(app);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("read plan state {}: {e}", path.display())),
    };
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|e| format!("plan state {} is corrupt: {e}", path.display()))
}

/// Monotonic sequence used to give temp files unique names (see `save_plan_state`).
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Read-modify-write mutex for plan state: the load→mutate→save sequence in
/// `record_plan` / `update_plan_step` must be atomic, or concurrent calls (parallel
/// tools in one round, background subagents) would overwrite each other's state.
/// Only guards threads inside this process; cross-process safety comes from the
/// atomic tmp+rename (last writer wins).
static PLAN_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn save_plan_state(app: &App, state: &PlanState) -> Result<(), String> {
    let path = plan_state_path(app);
    let parent = path
        .parent()
        .ok_or_else(|| "invalid plan-state path".to_string())?;
    std::fs::create_dir_all(parent).map_err(|e| format!("cannot create plan-state dir: {e}"))?;
    // Temp names are unique per process id + monotonic sequence, so concurrent
    // writers never clobber each other's temp files; rename is atomic, so the final
    // path only ever contains a complete JSON document.
    let tmp = parent.join(format!(
        ".{}.{}-{}.tmp",
        PLAN_STATE_FILE_NAME,
        std::process::id(),
        TMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let bytes =
        serde_json::to_vec_pretty(state).map_err(|e| format!("serialize plan-state: {e}"))?;
    if let Err(e) = std::fs::write(&tmp, &bytes) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("write plan state {}: {e}", path.display()));
    }
    std::fs::rename(&tmp, &path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("persist plan state {}: {e}", path.display())
    })?;
    Ok(())
}

pub(crate) fn record_plan(
    app: &App,
    summary: &str,
    raw_steps: &[Value],
) -> Result<PlanState, String> {
    let _guard = PLAN_LOCK
        .lock()
        .map_err(|_| "plan-state lock poisoned".to_string())?;
    let previous = load_plan_state(app)?;
    let state = PlanState::build(summary, raw_steps, previous.as_ref())?;
    save_plan_state(app, &state)?;
    Ok(state)
}

pub(crate) fn update_plan_step(
    app: &App,
    step: u64,
    status: StepStatus,
    note: Option<String>,
) -> Result<(PlanState, StepTransition), String> {
    let _guard = PLAN_LOCK
        .lock()
        .map_err(|_| "plan-state lock poisoned".to_string())?;
    let mut state = match load_plan_state(app)? {
        Some(state) => state,
        None => {
            return Err(
                "No active plan in this session. Call `plan` (or plan again) to create the step list first."
                    .to_string(),
            )
        }
    };
    let transition = match state.apply_update(step, status, note) {
        Ok(transition) => transition,
        // The plan-creation turns may be folded away by context compression, so a bare
        // "Step N not found" would leave the model guessing step numbers from memory.
        // Attach the full rendered plan plus a read-only persistence pointer (the
        // absolute plan-state.json path) to make the error self-healing. The pointer
        // must NOT suggest calling `plan` to "view": `plan` only creates/replaces the
        // plan and requires `steps`, so a view call would either fail or silently
        // replace the plan (regression: an earlier hint told the model to "re-view"
        // via `plan`).
        Err(msg) => {
            return Err(format!(
                "{msg}\nCurrent plan:\n{}Plan state persists at {} (plan-state.json, session assets) and survives context compression; read it with read_file if you need the current steps later. `plan` only creates or replaces the plan — use `plan_update` for status changes.\n",
                state.render(),
                plan_state_path(app).display()
            ))
        }
    };
    save_plan_state(app, &state)?;
    Ok((state, transition))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Removes the temporary session dir even when an assertion panics.
    struct TempDirGuard(std::path::PathBuf);
    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Per-process counter so tests starting in the same millisecond never share a
    /// temp dir (pid + timestamp alone collided and the tests destroyed each other's
    /// plan-state files; reproduced as flaky failures under parallel test threads).
    static TEST_SEQ: AtomicU64 = AtomicU64::new(0);

    /// Test app whose plan-state file lands in a fresh temp dir (never the shared
    /// `default.assets/` fallback), mirroring the plan_tools test setup.
    fn test_app() -> (App, TempDirGuard) {
        let base = std::env::temp_dir().join(format!(
            "plan-state-store-test-{}-{}-{}",
            std::process::id(),
            TEST_SEQ.fetch_add(1, Ordering::Relaxed),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let guard = TempDirGuard(base.clone());
        let mut app = crate::ai::middleware::test_util::test_app();
        app.session_history_file = base.join("session.sqlite");
        app.session_id = "plan-state-store-test".to_string();
        (app, guard)
    }

    fn three_step_plan(app: &App) {
        let steps = serde_json::json!([
            { "step": 1, "action": "Read", "tool": "read_file" },
            { "step": 2, "action": "Patch", "tool": "apply_patch" },
            { "step": 3, "action": "Verify", "tool": "execute_command" }
        ]);
        record_plan(app, "Demo", steps.as_array().unwrap()).unwrap();
    }

    #[test]
    fn update_missing_step_reports_full_plan_for_self_healing() {
        let (app, _guard) = test_app();
        three_step_plan(&app);
        // Give step 1 a visible terminal status so the reported plan shows suffixes.
        update_plan_step(&app, 1, StepStatus::Done, None).unwrap();

        let err = update_plan_step(&app, 9, StepStatus::Done, None).unwrap_err();
        assert!(err.contains("Step 9 not found in the active plan."), "{err}");
        assert!(err.contains("Current plan:"), "{err}");
        // Real step numbers and statuses are listed, so the model can pick one instead
        // of guessing (regression: hallucinated step 6/5 after compression).
        assert!(err.contains("Step 1. [read_file] Read (done)"), "{err}");
        assert!(err.contains("Step 2. [apply_patch] Patch"), "{err}");
        assert!(err.contains("3 step(s) planned."), "{err}");
        // Persistence pointer: the plan survives compression at plan-state.json.
        assert!(err.contains("plan-state.json"), "{err}");
        // The recovery pointer must stay read-only: `plan` only creates/replaces the
        // plan (regression: a hint telling the model to "re-view" via `plan` would
        // fail on bare calls or silently replace the plan with guessed steps).
        assert!(err.contains("read_file"), "{err}");
        assert!(!err.contains("re-view"), "{err}");

        // The failed update must not have mutated any state.
        let state = load_plan_state(&app).unwrap().unwrap();
        assert_eq!(state.done_count(), 1);
        assert_eq!(state.steps[1].status, StepStatus::Pending);
    }

    #[test]
    fn update_without_plan_asks_to_create_one() {
        let (app, _guard) = test_app();
        let err = update_plan_step(&app, 1, StepStatus::Done, None).unwrap_err();
        assert!(err.contains("No active plan"), "{err}");
        assert!(err.contains("Call `plan`"), "{err}");
    }
}