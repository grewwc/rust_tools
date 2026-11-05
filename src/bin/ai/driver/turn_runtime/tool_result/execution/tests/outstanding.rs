//! Tests for the outstanding-work gate: plan state, not prose, decides whether
//! a final response may land.

use super::super::*;
use super::common::*;
use crate::ai::tools::plan_state::{StepStatus, record_plan, update_plan_step};

/// Test app whose plan files land in a fresh temp dir (mirrors the
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

fn tool_exchange(id: &str, name: &str, content: &str) -> Vec<Message> {
    vec![
        assistant_tool_call_message(test_tool_call(id, name, serde_json::json!({}))),
        tool_result_message(id, content),
    ]
}

#[test]
fn tool_assisted_turn_without_plan_allows_final() {
    let (app, base) = test_app_with_session();
    let mut messages = tool_exchange("work", "execute_command", "[command exited: exit code 0]");
    let before = messages.clone();
    let action = outstanding_work_gate_action(&app, &mut messages, false, 0, 16);
    assert_eq!(action, OutstandingWorkGateAction::Allow);
    assert_eq!(messages, before, "no plan must not inject a reopen note");
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn request_user_input_allows_final_without_plan() {
    let (app, base) = test_app_with_session();
    let mut messages = tool_exchange("ask", "request_user_input", "Input requested.");
    let action = outstanding_work_gate_action(&app, &mut messages, false, 0, 16);
    assert_eq!(action, OutstandingWorkGateAction::Allow);
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn recoverable_pre_execution_pause_uses_its_own_gate() {
    let (app, base) = test_app_with_session();
    let pause = format!("Error: {SCOPED_INSTRUCTIONS_PAUSE_PREFIX}");
    let turn_messages = tool_exchange("patch", "apply_patch", &pause);
    let mut messages = turn_messages.clone();
    let action = outstanding_work_gate_action(&app, &mut messages, false, 0, 16);
    assert_eq!(action, OutstandingWorkGateAction::Allow);
    assert_eq!(
        recoverable_pause_gate_action(&mut messages, &turn_messages, false, 0, 16),
        RecoverablePauseGateAction::Reopen
    );
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn outstanding_reopens_three_times_then_warns_without_duplicate_marker() {
    let (app, base) = test_app_with_session();
    // Unsettled plan steps keep reopening until the bounded budget is spent.
    two_step_plan(&app);
    let mut messages = Vec::new();
    let mut gate_state = FinalGateState::default();

    for attempt in 1..=OUTSTANDING_WORK_MAX_REOPENS {
        assert!(gate_state.can_reopen_outstanding_work(false, attempt, 16));
        let action =
            super::super::outstanding_work_gate_action(&app, &mut messages, false, attempt, 16);
        assert_eq!(action, OutstandingWorkGateAction::Reopen);
        assert_eq!(gate_state.consume_outstanding_work_retry(), attempt);
    }
    assert!(!gate_state.can_reopen_outstanding_work(false, 4, 16));
    let final_action = super::super::outstanding_work_gate_action(&app, &mut messages, true, 4, 16);
    assert!(
        matches!(final_action, OutstandingWorkGateAction::Warn(steps) if steps.contains("step 1") && steps.contains("step 2"))
    );

    assert_eq!(
        messages
            .iter()
            .filter(|message| {
                message
                    .content
                    .as_str()
                    .is_some_and(|text| text.starts_with(OUTSTANDING_WORK_REQUIRED_MARKER))
            })
            .count(),
        1
    );
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn open_steps_reopen_until_forced_then_warn_with_steps() {
    let (app, base) = test_app_with_session();
    two_step_plan(&app);
    let mut messages = Vec::new();
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16);
    assert_eq!(action, OutstandingWorkGateAction::Reopen);
    assert!(
        last_note(&messages).contains(OUTSTANDING_WORK_REQUIRED_MARKER),
        "reopen must carry the budget marker"
    );
    assert!(
        last_note(&messages).contains("step 1"),
        "{}",
        last_note(&messages)
    );
    assert!(
        last_note(&messages).contains("step 2"),
        "{}",
        last_note(&messages)
    );
    // Persistent gate: a second evaluation in the same turn still reopens.
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, false, 1, 16);
    assert_eq!(action, OutstandingWorkGateAction::Reopen);
    // Forced finals land annotated with the unsettled steps named.
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, true, 1, 16);
    match action {
        OutstandingWorkGateAction::Warn(steps) => {
            assert!(steps.contains("step 1"), "{steps}");
            assert!(steps.contains("step 2"), "{steps}");
        }
        other => panic!("expected Warn with steps, got {other:?}"),
    }
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, false, 16, 16);
    assert!(matches!(action, OutstandingWorkGateAction::Warn(_)));
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
    assert!(
        last_note(&messages).contains("step 2"),
        "{}",
        last_note(&messages)
    );
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn settled_plan_allows_final_directly() {
    let (app, base) = test_app_with_session();
    two_step_plan(&app);
    update_plan_step(&app, 1, StepStatus::Done, None).unwrap();
    update_plan_step(&app, 2, StepStatus::Skipped, None).unwrap();
    let mut messages = Vec::new();
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16);
    assert_eq!(action, OutstandingWorkGateAction::Allow);
    assert!(messages.is_empty());
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn reopen_is_recorded_in_the_decision_log() {
    let (app, base) = test_app_with_session();
    two_step_plan(&app);
    let mut messages = Vec::new();
    // Open steps must leave a decision-log trail; the note is projection-only.
    let action = super::super::outstanding_work_gate_action(&app, &mut messages, false, 3, 16);
    assert_eq!(action, OutstandingWorkGateAction::Reopen);
    let recent = crate::ai::driver::decision_log::get_decision_log_store()
        .recent_by_session(&app.session_id, 5);
    assert!(
        recent
            .iter()
            .any(|entry| { entry.context.contains("reason=outstanding_work_open_steps") }),
        "reopen must be observable in the decision log: {recent:?}"
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
fn final_gate_state_keeps_outstanding_reopen_independent() {
    let mut state = FinalGateState::default();
    assert!(state.can_reopen(false, 1, 16));
    assert!(state.can_reopen_outstanding_work(false, 1, 16));

    state.consume_retry();
    assert!(!state.can_reopen(false, 1, 16));
    assert!(state.can_reopen_outstanding_work(false, 1, 16));

    state.consume_outstanding_work_retry();
    // One consumed retry leaves room in the independent bounded budget.
    assert!(state.can_reopen_outstanding_work(false, 1, 16));
    assert!(!FinalGateState::default().can_reopen_outstanding_work(true, 1, 16));
    assert!(!FinalGateState::default().can_reopen_outstanding_work(false, 16, 16));
}

/// Run `f` as if inside session turn `turn_id` (the scope `run_turn` installs).
fn in_turn<R>(app: &App, turn_id: usize, f: impl FnOnce() -> R) -> R {
    crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope((app.session_id.clone(), turn_id), f)
}

#[test]
fn plan_from_earlier_turn_does_not_bind_a_new_turn() {
    let (app, base) = test_app_with_session();
    in_turn(&app, 5, || two_step_plan(&app));
    let mut messages = Vec::new();
    // Turn 6 carries an unrelated request: the turn-5 plan must not reopen it.
    let action = in_turn(&app, 6, || {
        super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16)
    });
    assert_eq!(action, OutstandingWorkGateAction::Allow);
    assert!(
        messages.is_empty(),
        "a carried plan must not inject a reopen note"
    );
    // The same turn that wrote the plan stays bound.
    let action = in_turn(&app, 5, || {
        super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16)
    });
    assert_eq!(action, OutstandingWorkGateAction::Reopen);
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn settled_plan_from_earlier_turn_allows_final() {
    let (app, base) = test_app_with_session();
    in_turn(&app, 5, || {
        two_step_plan(&app);
        update_plan_step(&app, 1, StepStatus::Done, None).unwrap();
        update_plan_step(&app, 2, StepStatus::Done, None).unwrap();
    });
    let mut messages = Vec::new();
    let action = in_turn(&app, 6, || {
        super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16)
    });
    assert_eq!(action, OutstandingWorkGateAction::Allow);
    assert!(messages.is_empty());
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn updating_carried_plan_rebinds_it_to_the_current_turn() {
    let (app, base) = test_app_with_session();
    in_turn(&app, 5, || two_step_plan(&app));
    let mut messages = Vec::new();
    let action = in_turn(&app, 6, || {
        update_plan_step(&app, 1, StepStatus::Done, None).unwrap();
        super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16)
    });
    assert_eq!(action, OutstandingWorkGateAction::Reopen);
    assert!(
        last_note(&messages).contains("step 2"),
        "{}",
        last_note(&messages)
    );
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn carried_plan_still_binds_resume_and_goal_mode_turns() {
    let (mut app, base) = test_app_with_session();
    in_turn(&app, 5, || two_step_plan(&app));
    let mut messages = Vec::new();
    let resumed = crate::ai::driver::runtime_ctx::IS_RESUME_TURN.sync_scope(true, || {
        in_turn(&app, 6, || {
            super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16)
        })
    });
    assert_eq!(resumed, OutstandingWorkGateAction::Reopen);
    app.goal_mode = Some("ship the feature".to_string());
    let goal = in_turn(&app, 7, || {
        super::super::outstanding_work_gate_action(&app, &mut messages, false, 0, 16)
    });
    assert_eq!(goal, OutstandingWorkGateAction::Reopen);
    let _ = fs::remove_dir_all(&base);
}
