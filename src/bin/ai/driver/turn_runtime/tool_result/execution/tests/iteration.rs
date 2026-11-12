//! Tests for the `iteration` cluster.

use super::super::*;
use super::common::*;

fn patch_file_preflight_round(
    app: &mut App,
    shared_mcp_client: &SharedMcpClient,
    tool_calls: Vec<ToolCall>,
) -> (TurnLoopStep, Vec<Message>) {
    let mut messages = Vec::new();
    let mut turn_messages = Vec::new();
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut terminal_dedupe_candidate = None;
    let mut force_final_response = false;
    let mut persisted_turn_messages = 0;
    let mut turn_had_tool_error = false;
    let step = handle_iteration_execution(
        app,
        "apply the prepared patch",
        &mcp_snapshot(shared_mcp_client),
        shared_mcp_client,
        IterationExecution::ToolCall(ToolCallExecution {
            stream_result: crate::ai::types::StreamResult {
                tool_calls,
                ..Default::default()
            },
            allowed_tool_names: FastSet::from_iter(["write_file".into(), "apply_patch".into()]),
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
    (step, turn_messages)
}

#[test]
fn patch_file_preflight_rejects_sources_created_or_changed_in_same_batch() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    for initially_exists in [false, true] {
        let root =
            std::env::temp_dir().join(format!("ai-patch-file-preflight-{}", uuid::Uuid::new_v4()));
        let primary = root.join("primary");
        let added = root.join("added");
        fs::create_dir_all(primary.join(".git")).unwrap();
        fs::create_dir_all(&added).unwrap();
        fs::write(added.join("AGENTS.md"), "added root rules").unwrap();
        let source = primary.join("source.patch");
        let original_target = primary.join("original.rs");
        if initially_exists {
            fs::write(
                &source,
                "*** Begin Patch\n*** Add File: original.rs\n+original\n*** End Patch\n",
            )
            .unwrap();
        }
        let target = added.join("new.rs");
        let replacement = format!(
            "*** Begin Patch\n*** Add File: {}\n+changed\n*** End Patch\n",
            target.display()
        );
        let mut app = test_app_with_tools(&["write_file", "apply_patch"]);
        app.cli.add_dirs = vec![added.canonicalize().unwrap()];
        app.config.history_file = root.join("history.sqlite");
        app.session_history_file = app.config.history_file.clone();
        let shared_mcp_client = Arc::new(std::sync::Mutex::new(McpClient::new()));
        let context = crate::ai::driver::runtime_ctx::DriverContext::new(
            app.clone(),
            shared_mcp_client.clone(),
            Arc::new(Vec::new()),
            Arc::new(Vec::new()),
        );
        let ((step, messages), retry_step) = SUBAGENT_CWD.sync_scope(primary, || {
            crate::ai::driver::runtime_ctx::DRIVER_CTX.sync_scope(context, || {
                let first = patch_file_preflight_round(
                    &mut app,
                    &shared_mcp_client,
                    vec![
                        test_tool_call(
                            "call_source",
                            "write_file",
                            serde_json::json!({"file_path": source, "content": replacement}),
                        ),
                        test_tool_call(
                            "call_patch",
                            "apply_patch",
                            serde_json::json!({
                                "patch_file": source,
                            }),
                        ),
                        test_tool_call(
                            "call_patch_duplicate",
                            "apply_patch",
                            serde_json::json!({"patch_file": source}),
                        ),
                    ],
                );
                // A later round must bind the new version after all duplicate
                // source guards have unwound, then load the added root's rules.
                let (retry_step, _) = patch_file_preflight_round(
                    &mut app,
                    &shared_mcp_client,
                    vec![test_tool_call(
                        "call_patch_retry",
                        "apply_patch",
                        serde_json::json!({"patch_file": source}),
                    )],
                );
                (first, retry_step)
            })
        });
        assert!(matches!(step, TurnLoopStep::Continue));
        assert!(matches!(
            retry_step,
            TurnLoopStep::ScopedPreflightContinue(_)
        ));
        assert_eq!(fs::read_to_string(&source).unwrap(), replacement);
        assert!(!target.exists(), "the changed patch must not be applied");
        assert!(!original_target.exists());
        let error = messages
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("call_patch"))
            .and_then(|message| message.content.as_str())
            .expect("apply_patch must return an aligned error result");
        assert!(error.contains("preflight"), "{error}");
        assert!(error.contains("retry"), "{error}");
        assert!(
            error.contains(if initially_exists {
                "changed"
            } else {
                "unavailable"
            }),
            "{error}"
        );
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn patch_file_preflight_preserves_unchanged_large_file_sources() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let root = std::env::temp_dir().join(format!(
        "ai-patch-file-preflight-large-{}",
        uuid::Uuid::new_v4()
    ));
    fs::create_dir_all(root.join(".git")).unwrap();
    let content = "x".repeat(9_000);
    let source = root.join("source.patch");
    fs::write(
        &source,
        format!("*** Begin Patch\n*** Add File: large.rs\n+{content}\n*** End Patch\n"),
    )
    .unwrap();
    let mut app = test_app_with_tools(&["apply_patch"]);
    app.config.history_file = root.join("history.sqlite");
    app.session_history_file = app.config.history_file.clone();
    let shared_mcp_client = Arc::new(std::sync::Mutex::new(McpClient::new()));
    let (step, messages) = SUBAGENT_CWD.sync_scope(root.clone(), || {
        patch_file_preflight_round(
            &mut app,
            &shared_mcp_client,
            vec![test_tool_call(
                "call_patch",
                "apply_patch",
                serde_json::json!({"patch_file": source}),
            )],
        )
    });
    assert!(matches!(step, TurnLoopStep::Continue));
    assert_eq!(fs::read_to_string(root.join("large.rs")).unwrap(), content);
    assert!(messages.iter().any(|message| {
        message.tool_call_id.as_deref() == Some("call_patch")
            && message
                .content
                .as_str()
                .is_some_and(|content| !content.starts_with("Error:"))
    }));
    fs::remove_dir_all(root).unwrap();
}

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

fn tool_round_app() -> (App, Arc<std::sync::Mutex<McpClient>>, std::path::PathBuf) {
    let session_root =
        std::env::temp_dir().join(format!("ai-serve-commit-{}", uuid::Uuid::new_v4()));
    let history_file = session_root.join("history.sqlite");
    let mut app = test_app_with_tools(&["read_file"]);
    app.config.history_file = history_file.clone();
    app.session_history_file = history_file.clone();
    app.session_id = "serve-commit-test".to_string();
    let shared_mcp_client = Arc::new(std::sync::Mutex::new(McpClient::new()));
    (app, shared_mcp_client, session_root)
}

fn run_tool_call_round(
    app: &mut App,
    shared_mcp_client: &Arc<std::sync::Mutex<McpClient>>,
) -> Result<TurnLoopStep, Box<dyn std::error::Error>> {
    let mut messages = Vec::new();
    let mut turn_messages = Vec::new();
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut terminal_dedupe_candidate = None;
    let mut force_final_response = false;
    let mut persisted_turn_messages = 0;
    let mut turn_had_tool_error = false;
    handle_iteration_execution(
        app,
        "read the file and continue",
        &mcp_snapshot(shared_mcp_client),
        shared_mcp_client,
        IterationExecution::ToolCall(ToolCallExecution {
            stream_result: crate::ai::types::StreamResult {
                assistant_text: "先读文件。".to_string(),
                hidden_meta: String::new(),
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
}

#[test]
fn serve_mode_tool_round_commits_visible_body() {
    let (mut app, shared_mcp_client, session_root) = tool_round_app();
    let (step, frames) = crate::ai::background::capture_serve_frames(|| {
        crate::ai::background::with_serve_live_streaming_for_test(true, || {
            run_tool_call_round(&mut app, &shared_mcp_client).unwrap()
        })
    });
    assert!(matches!(step, TurnLoopStep::Continue));
    let replace: Vec<_> = frames
        .iter()
        .filter(|(kind, _)| *kind == crate::ai::background::ServeLiveKind::Replace)
        .collect();
    assert!(
        replace.iter().any(|(_, text)| text.contains("先读文件。")),
        "serve mode must commit the tool round body: {frames:?}"
    );
    let _ = std::fs::remove_dir_all(session_root.join("history.sessions"));
}

#[test]
fn cli_mode_tool_round_does_not_commit_visible_body() {
    let (mut app, shared_mcp_client, session_root) = tool_round_app();
    let (step, frames) = crate::ai::background::capture_serve_frames(|| {
        run_tool_call_round(&mut app, &shared_mcp_client).unwrap()
    });
    assert!(matches!(step, TurnLoopStep::Continue));
    assert!(
        frames.is_empty(),
        "CLI mode must not publish serve frames: {frames:?}"
    );
    let _ = std::fs::remove_dir_all(session_root.join("history.sessions"));
}

#[test]
fn tool_round_serve_commit_gate_conditions() {
    crate::ai::background::with_serve_live_streaming_for_test(true, || {
        assert!(should_commit_tool_round_body_for_serve(
            &Some("body".to_string()),
            false,
            1,
            16,
            false,
        ));
        // Force-final second-violation path is excluded: finalize commits once.
        assert!(!should_commit_tool_round_body_for_serve(
            &Some("body".to_string()),
            true,
            1,
            16,
            true,
        ));
        // Force-final before the retry marker exists still commits.
        assert!(should_commit_tool_round_body_for_serve(
            &Some("body".to_string()),
            true,
            1,
            16,
            false,
        ));
        assert!(!should_commit_tool_round_body_for_serve(
            &None,
            false,
            1,
            16,
            false,
        ));
    });
    // CLI mode: never committed.
    assert!(!should_commit_tool_round_body_for_serve(
        &Some("body".to_string()),
        false,
        1,
        16,
        false,
    ));
}
