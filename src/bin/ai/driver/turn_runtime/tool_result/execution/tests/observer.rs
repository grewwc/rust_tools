//! Tests for the `observer` cluster.

use super::super::*;
use super::common::*;

#[test]
fn command_input_marks_pseudo_terminal_mode() {
    let pty = format_command_input(r#"{"command":"login --qr","pty":true,"cwd":"/tmp"}"#)
        .expect("valid command arguments");
    assert_eq!(pty, "login --qr  (cwd: /tmp)  (PTY)");

    let piped = format_command_input(r#"{"command":"git diff","pty":false}"#)
        .expect("valid command arguments");
    assert_eq!(piped, "git diff");
}

#[test]
fn full_streaming_is_limited_to_explicit_pty_execute_command() {
    let interactive = test_tool_call(
        "call_interactive",
        "execute_command",
        serde_json::json!({ "command": "example-cli auth login", "pty": true }),
    );
    assert!(execute_command_uses_pseudo_terminal(&interactive));

    let ordinary = test_tool_call(
        "call_ordinary",
        "execute_command",
        serde_json::json!({ "command": "cargo check", "pty": false }),
    );
    assert!(!execute_command_uses_pseudo_terminal(&ordinary));

    let unrelated = test_tool_call(
        "call_unrelated",
        "read_file",
        serde_json::json!({ "file_path": "Cargo.toml", "pty": true }),
    );
    assert!(!execute_command_uses_pseudo_terminal(&unrelated));
}

#[test]
fn partial_stream_with_structured_failure_never_renders_success() {
    let call = test_tool_call(
        "call_timeout",
        "execute_command",
        serde_json::json!({ "command": "sleep 30", "pty": true }),
    );
    let result = tools::RunOneResult {
        tool_result: crate::ai::types::ToolResult {
            task_report: None,
            tool_call_id: call.id.clone(),
            content: "partial output before timeout".to_string(),
        },
        ok: false,
        executed: true,
        cached: false,
    };

    assert!(streamed_tool_result_is_failure(&call, &result));
}

#[test]
fn serve_tool_dispatch_closes_thinking_before_tool_frame() {
    let app = test_app_with_tools(&[]);
    let mut observer = TerminalToolObserver::new_for_test(&app);
    let call = test_tool_call(
        "call_wait",
        "task_wait",
        serde_json::json!({"task_ids": ["task_1"]}),
    );

    let (_, frames) = crate::ai::background::capture_serve_frames(|| {
        crate::ai::background::with_serve_live_streaming_for_test(true, || {
            // A reasoning phase is open when the call dispatches.
            crate::ai::background::publish_serve_frame(
                crate::ai::background::ServeLiveKind::Thinking,
                "Wait again with a",
            );
            crate::ai::driver::tools::ToolExecutionObserver::on_tool_dispatched(
                &mut observer,
                &call,
            );
            // End-of-stream finalize may have closed the same phase first;
            // the publisher must swallow that duplicate.
            crate::ai::background::publish_serve_frame(
                crate::ai::background::ServeLiveKind::ThinkingDone,
                "",
            );
        });
    });

    let done: Vec<_> = frames
        .iter()
        .filter(|(kind, _)| *kind == crate::ai::background::ServeLiveKind::ThinkingDone)
        .collect();
    assert_eq!(done.len(), 1, "one close per thinking phase: {frames:?}");
    let done_index = frames
        .iter()
        .position(|(kind, _)| *kind == crate::ai::background::ServeLiveKind::ThinkingDone)
        .expect("dispatch closes the open phase");
    let tool_index = frames
        .iter()
        .position(|(kind, _)| *kind == crate::ai::background::ServeLiveKind::Tool)
        .expect("dispatch publishes the tool card");
    assert!(
        done_index < tool_index,
        "the thinking fold must close before the tool card: {frames:?}"
    );
    assert!(frames[tool_index].1.contains("\"phase\":\"started\""));
}
