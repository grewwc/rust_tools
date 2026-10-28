//! Regression coverage for the post-`complete` final-response terminator.

use super::super::*;
use super::common::*;
use super::outstanding::{test_app_with_session_and_tools, two_step_plan};
use crate::ai::mcp::{McpClient, SharedMcpClient};
use crate::ai::tools::plan_state::{
    load_completion_claim, load_plan_state, record_completion_claim,
};
use crate::ai::tools::{COMPLETE_CLAIM_RECORDED_PREFIX, FINAL_RESPONSE_END_MARKER};
use std::sync::{Arc, Mutex};

fn completed_tool_turn(app: &App) -> Vec<Message> {
    record_completion_claim(app, "work finished", None).unwrap();
    vec![
        assistant_tool_call_message(test_tool_call(
            "work",
            "read_file",
            serde_json::json!({"file_path": "src/lib.rs"}),
        )),
        tool_result_message("work", "contents"),
        assistant_tool_call_message(test_tool_call(
            "complete",
            "complete",
            serde_json::json!({"summary": "work finished"}),
        )),
        tool_result_message(
            "complete",
            &format!("{COMPLETE_CLAIM_RECORDED_PREFIX} work finished"),
        ),
    ]
}

fn run_final_response(
    app: &mut App,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    text: &str,
    iteration: usize,
) -> (TurnLoopStep, String, bool, Option<String>) {
    let shared_mcp: SharedMcpClient = Arc::new(Mutex::new(McpClient::new()));
    let mut persisted_turn_messages = 0;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut force_final_response = false;
    let mut terminal_dedupe_candidate = None;
    let mut turn_had_tool_error = false;

    let step = handle_iteration_execution(
        app,
        "inspect the file",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        IterationExecution::FinalResponse(crate::ai::types::StreamResult {
            outcome: crate::ai::types::StreamOutcome::Completed,
            assistant_text: text.to_string(),
            skip_response_drain: true,
            ..Default::default()
        }),
        messages,
        turn_messages,
        false,
        &mut persisted_turn_messages,
        &mut final_assistant_text,
        &mut final_assistant_recorded,
        &mut force_final_response,
        &mut terminal_dedupe_candidate,
        true,
        iteration,
        16,
        0,
        &mut turn_had_tool_error,
    )
    .unwrap();

    (
        step,
        final_assistant_text,
        final_assistant_recorded,
        terminal_dedupe_candidate,
    )
}

#[test]
fn missing_terminator_reopens_once_without_consuming_claim() {
    let (app, root) = test_app_with_session_and_tools(&["read_file", "complete"]);
    let turn_messages = completed_tool_turn(&app);
    let mut messages = Vec::new();
    let mut text = "Complete answer without the terminator.".to_string();

    assert_eq!(
        final_terminator_gate_action(&app, &mut messages, &turn_messages, &mut text, true,),
        FinalTerminatorGateAction::Reopen
    );
    assert!(load_completion_claim(&app).unwrap().is_some());
    assert!(current_turn_has_internal_marker(
        &messages,
        FINAL_TERMINATOR_RETRY_MARKER
    ));

    assert_eq!(
        final_terminator_gate_action(&app, &mut messages, &turn_messages, &mut text, true,),
        FinalTerminatorGateAction::Warn
    );
    assert!(load_completion_claim(&app).unwrap().is_some());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn final_without_complete_is_unchanged() {
    let (app, root) = test_app_with_session_and_tools(&[]);
    let mut messages = Vec::new();
    let mut text = "Literal protocol discussion.\n[END]".to_string();
    assert_eq!(
        final_terminator_gate_action(&app, &mut messages, &[], &mut text, true),
        FinalTerminatorGateAction::Allow
    );
    assert_eq!(text, "Literal protocol discussion.\n[END]");
    assert!(messages.is_empty());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn outstanding_work_reopens_before_the_terminator_gate() {
    let (mut app, root) = test_app_with_session_and_tools(&["read_file", "complete"]);
    two_step_plan(&app);
    let plan_updated_at_ms = load_plan_state(&app).unwrap().unwrap().updated_at_ms;
    let mut turn_messages = completed_tool_turn(&app);
    record_completion_claim(&app, "work finished", Some(plan_updated_at_ms)).unwrap();
    let mut messages = turn_messages.clone();

    let (step, _, recorded, _) = run_final_response(
        &mut app,
        &mut messages,
        &mut turn_messages,
        "Premature answer without a terminator.",
        2,
    );

    assert!(matches!(step, TurnLoopStep::Continue));
    assert!(!recorded);
    assert!(current_turn_has_internal_marker(
        &messages,
        OUTSTANDING_WORK_REQUIRED_MARKER
    ));
    assert!(!current_turn_has_internal_marker(
        &messages,
        FINAL_TERMINATOR_RETRY_MARKER
    ));
    assert!(load_completion_claim(&app).unwrap().is_none());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn terminator_retry_keeps_claim_until_replacement_is_accepted() {
    let (mut app, root) = test_app_with_session_and_tools(&["read_file", "complete"]);
    let mut turn_messages = completed_tool_turn(&app);
    let mut messages = turn_messages.clone();

    let (first_step, _, first_recorded, _) = run_final_response(
        &mut app,
        &mut messages,
        &mut turn_messages,
        "Complete answer without a terminator.",
        2,
    );
    assert!(matches!(first_step, TurnLoopStep::Continue));
    assert!(!first_recorded);
    assert!(load_completion_claim(&app).unwrap().is_some());

    let (second_step, final_text, second_recorded, terminal) = run_final_response(
        &mut app,
        &mut messages,
        &mut turn_messages,
        "Replacement answer.\n[END]",
        3,
    );
    assert!(matches!(second_step, TurnLoopStep::Break));
    assert!(second_recorded);
    assert_eq!(final_text, "Replacement answer.");
    assert_eq!(terminal.as_deref(), Some("Replacement answer."));
    assert!(load_completion_claim(&app).unwrap().is_none());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn forced_echo_stop_consumes_completion_claim() {
    let (mut app, root) = test_app_with_session_and_tools(&["read_file", "complete"]);
    let mut turn_messages = completed_tool_turn(&app);
    let mut messages = turn_messages.clone();

    let (step, final_text, recorded, terminal) = run_final_response(
        &mut app,
        &mut messages,
        &mut turn_messages,
        "self_note:completion_evidence_required\ninspect the diff first.",
        16,
    );

    assert!(matches!(step, TurnLoopStep::Break));
    assert!(!recorded);
    assert_eq!(final_text, INJECTED_CONTEXT_ECHO_STOP);
    assert!(terminal.is_none());
    assert!(load_completion_claim(&app).unwrap().is_none());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn accepted_terminator_never_reaches_history_or_terminal_body() {
    let (mut app, root) = test_app_with_session_and_tools(&["read_file", "complete"]);
    let shared_mcp: SharedMcpClient = Arc::new(Mutex::new(McpClient::new()));
    let mut turn_messages = completed_tool_turn(&app);
    let mut messages = turn_messages.clone();
    let mut persisted_turn_messages = 0;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut force_final_response = false;
    let mut terminal_dedupe_candidate = None;
    let mut turn_had_tool_error = false;

    let step = handle_iteration_execution(
        &mut app,
        "inspect the file",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        IterationExecution::FinalResponse(crate::ai::types::StreamResult {
            outcome: crate::ai::types::StreamOutcome::Completed,
            assistant_text: "Complete answer.\n[END]\n".to_string(),
            skip_response_drain: true,
            ..Default::default()
        }),
        &mut messages,
        &mut turn_messages,
        false,
        &mut persisted_turn_messages,
        &mut final_assistant_text,
        &mut final_assistant_recorded,
        &mut force_final_response,
        &mut terminal_dedupe_candidate,
        true,
        2,
        16,
        0,
        &mut turn_had_tool_error,
    )
    .unwrap();

    assert!(matches!(step, TurnLoopStep::Break));
    assert!(final_assistant_recorded);
    assert_eq!(final_assistant_text, "Complete answer.");
    assert_eq!(
        terminal_dedupe_candidate.as_deref(),
        Some("Complete answer.")
    );
    assert!(load_completion_claim(&app).unwrap().is_none());
    let recorded = turn_messages.last().unwrap();
    assert_eq!(recorded.role, "assistant");
    assert_eq!(recorded.content.as_str(), Some("Complete answer."));
    assert!(!turn_messages.iter().any(|message| {
        message.role == "assistant"
            && message
                .content
                .as_str()
                .is_some_and(|text| text.contains(FINAL_RESPONSE_END_MARKER))
    }));
    let _ = std::fs::remove_dir_all(root);
}
