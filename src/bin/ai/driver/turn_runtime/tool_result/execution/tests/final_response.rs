//! Final responses are accepted without a tool handshake or control terminator.

use super::super::*;
use super::common::*;
use super::outstanding::{test_app_with_session_and_tools, two_step_plan};
use crate::ai::mcp::{McpClient, SharedMcpClient};
use crate::ai::tools::plan_state::{StepStatus, update_plan_step};
use std::sync::{Arc, Mutex};

fn inspected_tool_turn() -> Vec<Message> {
    vec![
        assistant_tool_call_message(test_tool_call(
            "work",
            "read_file",
            serde_json::json!({"file_path": "src/lib.rs"}),
        )),
        tool_result_message("work", "contents"),
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
fn removed_completion_tool_is_not_registered_or_discoverable() {
    use crate::ai::tools::registry::common::{
        ToolDisplayRegistration, ToolHistoryPolicyRegistration, ToolRegistration,
    };

    assert!(
        inventory::iter::<ToolRegistration>
            .into_iter()
            .all(|entry| entry.spec.name != "complete")
    );
    assert!(
        inventory::iter::<ToolDisplayRegistration>
            .into_iter()
            .all(|entry| entry.name != "complete")
    );
    assert!(
        inventory::iter::<ToolHistoryPolicyRegistration>
            .into_iter()
            .all(|entry| entry.name != "complete")
    );
    assert!(crate::ai::tools::get_tool_definitions_by_names(&["complete".to_string()]).is_empty());
}

#[test]
fn plain_final_is_accepted_without_tools() {
    let (mut app, root) = test_app_with_session_and_tools(&[]);
    let mut messages = Vec::new();
    let mut turn_messages = Vec::new();
    let text = "An answer with no tool calls.\n";
    let (step, final_text, recorded, terminal) =
        run_final_response(&mut app, &mut messages, &mut turn_messages, text, 2);
    assert!(matches!(step, TurnLoopStep::Break));
    assert!(recorded);
    assert_eq!(final_text, text);
    assert_eq!(terminal.as_deref(), Some(text));
    assert_eq!(turn_messages.last().unwrap().content.as_str(), Some(text));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn final_text_is_preserved_in_history_and_terminal_body() {
    for text in [
        "Complete answer without a control marker.",
        "  Literal protocol discussion.\n[END]\n\n",
        "```text\nLiteral text\n[END]",
    ] {
        for with_plan in [false, true] {
            let (mut app, root) = test_app_with_session_and_tools(&["read_file"]);
            if with_plan {
                two_step_plan(&app);
                update_plan_step(&app, 1, StepStatus::Done, None).unwrap();
                update_plan_step(&app, 2, StepStatus::Skipped, None).unwrap();
            }
            let mut turn_messages = inspected_tool_turn();
            let mut messages = turn_messages.clone();

            let (step, final_text, recorded, terminal) =
                run_final_response(&mut app, &mut messages, &mut turn_messages, text, 2);

            assert!(matches!(step, TurnLoopStep::Break));
            assert!(recorded);
            assert_eq!(final_text, text);
            assert_eq!(terminal.as_deref(), Some(text));
            for history in [&messages, &turn_messages] {
                let recorded = history.last().unwrap();
                assert_eq!(recorded.role, "assistant");
                assert_eq!(recorded.content.as_str(), Some(text));
            }
            let _ = std::fs::remove_dir_all(root);
        }
    }
}

#[test]
fn outstanding_plan_reopens_until_steps_are_settled() {
    let (mut app, root) = test_app_with_session_and_tools(&["read_file"]);
    two_step_plan(&app);
    let mut turn_messages = inspected_tool_turn();
    let mut messages = turn_messages.clone();

    let (step, _, recorded, _) = run_final_response(
        &mut app,
        &mut messages,
        &mut turn_messages,
        "Premature answer.",
        2,
    );
    assert!(matches!(step, TurnLoopStep::Continue));
    assert!(!recorded);
    assert!(current_turn_has_internal_marker(
        &messages,
        OUTSTANDING_WORK_REQUIRED_MARKER
    ));

    update_plan_step(&app, 1, StepStatus::Done, None).unwrap();
    update_plan_step(&app, 2, StepStatus::Skipped, None).unwrap();
    let text = "Finished answer.";
    let (step, final_text, recorded, terminal) =
        run_final_response(&mut app, &mut messages, &mut turn_messages, text, 3);
    assert!(matches!(step, TurnLoopStep::Break));
    assert!(recorded);
    assert_eq!(final_text, text);
    assert_eq!(terminal.as_deref(), Some(text));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn forced_echo_stop_discards_internal_note() {
    let (mut app, root) = test_app_with_session_and_tools(&["read_file"]);
    let mut turn_messages = inspected_tool_turn();
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
    let _ = std::fs::remove_dir_all(root);
}
