//! Tests for the outstanding-work gate: plan state, not prose, decides whether
//! a final response may land.

use super::super::*;
use super::common::*;
use crate::ai::tools::plan_state::{
    StepStatus, load_completion_claim, record_completion_claim, record_plan, update_plan_step,
};

/// Test app whose plan/claim files land in a fresh temp dir (mirrors the
/// plan-state store test setup; pid + sequence + timestamp keeps parallel
/// test threads from sharing a dir).
pub(super) fn test_app_with_session_and_tools(tools: &[&str]) -> (App, PathBuf) {
    use std::time::{SystemTime, UNIX_EPOCH};
    static TEST_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let base = std::env::temp_dir().join(format!(
        "outstanding-gate-test-{}-{}-{}",
        std::process::id(),
        TEST_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis()
    ));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();
    let mut app = test_app_with_tools(tools);
    app.session_history_file = base.join("session.sqlite");
    app.session_id = "outstanding-gate-test".to_string();
    (app, base)
}

fn test_app_with_session() -> (App, PathBuf) {
    test_app_with_session_and_tools(&[])
}

pub(super) fn two_step_plan(app: &App) {
    let steps = serde_json::json!([
        { "step": 1, "action": "Read", "tool": "read_file" },
        { "step": 2, "action": "Patch", "tool": "apply_patch" }
    ]);
    record_plan(app, "Demo", steps.as_array().unwrap()).unwrap();
}

fn last_note(messages: &[Message]) -> &str {
    messages
        .last()
        .unwrap()
        .content
        .as_str()
        .expect("gate note must be a string message")
}

#[test]
fn no_plan_file_allows_final() {
    let (app, base) = test_app_with_session();
    let mut messages = Vec::new();
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16);
    assert_eq!(action, OutstandingWorkGateAction::Allow);
    assert!(messages.is_empty());
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn no_plan_still_consumes_claim() {
    // A stale assertion must not leak into the next round that does have a plan.
    let (app, base) = test_app_with_session();
    record_completion_claim(&app, "all done", None).unwrap();
    let mut messages = Vec::new();
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16);
    assert_eq!(action, OutstandingWorkGateAction::Allow);
    assert!(
        load_completion_claim(&app).unwrap().is_none(),
        "no-plan evaluation must still consume the single-use claim"
    );
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn open_steps_reopen_once_then_warn() {
    let (app, base) = test_app_with_session();
    two_step_plan(&app);
    let mut messages = Vec::new();
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16);
    assert_eq!(action, OutstandingWorkGateAction::Reopen);
    assert!(
        last_note(&messages).contains(OUTSTANDING_WORK_REQUIRED_MARKER),
        "reopen must carry the budget marker"
    );
    assert!(last_note(&messages).contains("step 1"), "{}", last_note(&messages));
    assert!(last_note(&messages).contains("step 2"), "{}", last_note(&messages));
    // Second evaluation in the same turn sees the marker and warns instead.
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, false, 1, 16);
    assert_eq!(action, OutstandingWorkGateAction::Warn);
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn failed_step_blocks_completion() {
    let (app, base) = test_app_with_session();
    two_step_plan(&app);
    update_plan_step(&app, 1, StepStatus::Done, None).unwrap();
    update_plan_step(&app, 2, StepStatus::Failed, None).unwrap();
    let mut messages = Vec::new();
    // A failed step needs retry/redo or an explicit skip: ending here would drop it.
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16);
    assert_eq!(action, OutstandingWorkGateAction::Reopen);
    assert!(last_note(&messages).contains("step 2"), "{}", last_note(&messages));
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn settled_plan_without_claim_reopens_once_then_warns() {
    let (app, base) = test_app_with_session();
    two_step_plan(&app);
    update_plan_step(&app, 1, StepStatus::Done, None).unwrap();
    update_plan_step(&app, 2, StepStatus::Skipped, None).unwrap();
    // A settled plan is not a completion assertion: landing without `complete`
    // must reopen once (the agent-side "continue"), guiding the model to
    // assert completion structurally instead of silently ending the turn.
    let mut messages = Vec::new();
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16);
    assert_eq!(action, OutstandingWorkGateAction::Reopen);
    assert!(last_note(&messages).contains("no `complete` call asserts completion"));
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, false, 1, 16);
    assert_eq!(action, OutstandingWorkGateAction::WarnUnasserted);
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn settled_plan_with_fresh_claim_allows_final() {
    let (app, base) = test_app_with_session();
    two_step_plan(&app);
    update_plan_step(&app, 1, StepStatus::Done, None).unwrap();
    update_plan_step(&app, 2, StepStatus::Skipped, None).unwrap();
    let plan_updated = crate::ai::tools::plan_state::load_plan_state(&app)
        .unwrap()
        .unwrap()
        .updated_at_ms;
    record_completion_claim(&app, "Done", Some(plan_updated)).unwrap();
    let mut messages = Vec::new();
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16);
    assert_eq!(action, OutstandingWorkGateAction::Allow);
    assert!(messages.is_empty());
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn fresh_claim_is_consumed_but_still_reopens_on_open_work() {
    let (app, base) = test_app_with_session();
    two_step_plan(&app);
    let plan_updated = crate::ai::tools::plan_state::load_plan_state(&app)
        .unwrap()
        .unwrap()
        .updated_at_ms;
    record_completion_claim(&app, "all done", Some(plan_updated)).unwrap();
    let mut messages = Vec::new();
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16);
    assert_eq!(action, OutstandingWorkGateAction::Reopen);
    assert!(
        last_note(&messages).contains("called `complete`"),
        "a fresh claim contradicting open work must be named: {}",
        last_note(&messages)
    );
    // Single-use: the claim file is gone, so the next round cannot reuse it.
    assert!(load_completion_claim(&app).unwrap().is_none());
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn stale_claim_after_plan_update_is_treated_as_unasserted() {
    let (app, base) = test_app_with_session();
    two_step_plan(&app);
    let plan_updated = crate::ai::tools::plan_state::load_plan_state(&app)
        .unwrap()
        .unwrap()
        .updated_at_ms;
    record_completion_claim(&app, "all done", Some(plan_updated)).unwrap();
    // Any plan change bumps updated_at_ms and silently invalidates the claim.
    update_plan_step(&app, 1, StepStatus::Running, None).unwrap();
    let mut messages = Vec::new();
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16);
    assert_eq!(action, OutstandingWorkGateAction::Reopen);
    assert!(
        !last_note(&messages).contains("called `complete`"),
        "a stale claim must not be presented as a live assertion: {}",
        last_note(&messages)
    );
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn corrupt_plan_fails_open() {
    let (app, base) = test_app_with_session();
    two_step_plan(&app);
    let path = crate::ai::tools::plan_state::plan_state_path(&app);
    fs::write(&path, b"not json").unwrap();
    let mut messages = Vec::new();
    // plan_update already surfaces the corruption; the gate must not brick the turn.
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16);
    assert_eq!(action, OutstandingWorkGateAction::Allow);
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn final_gate_state_has_independent_outstanding_retry_budget() {
    let mut state = FinalGateState::default();
    assert!(state.can_reopen(false, 1, 16));
    assert!(state.can_reopen_outstanding_work(false, 1, 16));

    state.consume_retry();
    assert!(!state.can_reopen(false, 1, 16));
    assert!(state.can_reopen_outstanding_work(false, 1, 16));

    state.consume_outstanding_work_retry();
    assert!(!state.can_reopen_outstanding_work(false, 1, 16));
    assert!(!FinalGateState::default().can_reopen_outstanding_work(true, 1, 16));
    assert!(!FinalGateState::default().can_reopen_outstanding_work(false, 16, 16));
}
