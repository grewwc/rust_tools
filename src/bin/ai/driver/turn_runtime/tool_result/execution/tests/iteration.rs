//! Tests for the `iteration` cluster.

use super::super::*;
use super::common::*;

#[test]
fn tool_call_round_persists_hidden_context_checkpoint() {
    let session_root =
        std::env::temp_dir().join(format!("ai-tool-round-checkpoint-{}", uuid::Uuid::new_v4()));
    let history_file = session_root.join("history.sqlite");
    let mut app = test_app_with_tools(&["read_file"]);
    app.config.history_file = history_file.clone();
    app.session_history_file = history_file.clone();
    app.session_id = "checkpoint-test".to_string();

    let shared_mcp_client = Arc::new(std::sync::Mutex::new(McpClient::new()));
    let mut messages = Vec::new();
    let mut turn_messages = Vec::new();
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut terminal_dedupe_candidate = None;
    let mut force_final_response = false;
    let mut persisted_turn_messages = 0;
    let mut turn_had_tool_error = false;

    let step = handle_iteration_execution(
        &mut app,
        "read the file and continue",
        &mcp_snapshot(&shared_mcp_client),
        &shared_mcp_client,
        IterationExecution::ToolCall(ToolCallExecution {
            stream_result: crate::ai::types::StreamResult {
                assistant_text: "先读文件。".to_string(),
                hidden_meta: "<meta:self_note>\n<context_checkpoint>\nsummary: 已确认根因\n证据：src/lib.rs:42。\n</context_checkpoint>\n</meta:self_note>".to_string(),
                tool_calls: vec![test_tool_call(
                    "call_read",
                    "read_file",
                    serde_json::json!({ "file_path": "Cargo.toml" }),
                )],
                ..Default::default()
            },
            allowed_tool_names: rust_tools::commonw::FastSet::from_iter(["read_file".to_string()]),
        }),
        &mut messages,
        &mut turn_messages,
        false,
        &mut persisted_turn_messages,
        &mut final_assistant_text,
        &mut final_assistant_recorded,
        &mut force_final_response,
        &mut terminal_dedupe_candidate,
        false,
        1,
        16,
        0,
        &mut turn_had_tool_error,
    )
    .unwrap();

    assert!(matches!(step, TurnLoopStep::Continue));
    assert_eq!(terminal_dedupe_candidate.as_deref(), Some("先读文件。"));
    let checkpoint_marker = turn_messages
        .iter()
        .find_map(|message| {
            (message.role == ROLE_INTERNAL_NOTE)
                .then(|| message.content.as_str())
                .flatten()
                .filter(|content| content.starts_with("[context_checkpoint path="))
        })
        .expect("tool-call hidden checkpoint should be persisted");
    let marker_path = checkpoint_marker
        .strip_prefix("[context_checkpoint path=")
        .and_then(|rest| rest.split(']').next())
        .expect("marker should include checkpoint path");
    assert!(
        std::path::Path::new(marker_path).is_file(),
        "checkpoint file should exist: {marker_path}"
    );

    let _ = std::fs::remove_dir_all(session_root.join("history.sessions"));
}
