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
        serde_json::json!({ "command": "lark-cli auth login", "pty": true }),
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
            tool_call_id: call.id.clone(),
            content: "partial output before timeout".to_string(),
        },
        ok: false,
        executed: true,
        cached: false,
    };

    assert!(streamed_tool_result_is_failure(&call, &result));
}
