//! Tests for the `rejection` cluster.

use super::common::*;
use super::super::*;

#[test]
fn scoped_instruction_preflight_blocks_first_mutation_until_rules_are_loaded() {
    let root = std::env::temp_dir().join(format!(
        "scoped-preflight-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let target = root.join("src/feature/mod.rs");
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::create_dir_all(target.parent().unwrap()).unwrap();
    fs::write(root.join("AGENTS.md"), "root rules\n").unwrap();
    fs::write(root.join("src/feature/AGENTS.md"), "feature rules\n").unwrap();
    fs::write(&target, "// source\n").unwrap();
    let mutation = test_tool_call(
        "command",
        "execute_command",
        serde_json::json!({
            "command": format!("printf changed > {}", target.display()),
            "pty": false
        }),
    );
    let mut messages = vec![Message {
        role: "system".to_string(),
        content: Value::String("base system".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    SUBAGENT_CWD.sync_scope(root.clone(), || {
        assert!(mutation_needs_scoped_instruction_preflight(
            &messages,
            std::slice::from_ref(&mutation)
        ));
        let mut app = test_app_with_tools(&["execute_command"]);
        let shared_mcp_client = Arc::new(std::sync::Mutex::new(McpClient::new()));
        let mut turn_messages = Vec::new();
        let mut persisted_turn_messages = 0;
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut force_final_response = false;
        let mut terminal_dedupe_candidate = None;
        let mut turn_had_tool_error = false;
        let step = handle_iteration_execution(
            &mut app,
            "change the file",
            &mcp_snapshot(&shared_mcp_client),
            &shared_mcp_client,
            IterationExecution::ToolCall(ToolCallExecution {
                stream_result: crate::ai::types::StreamResult {
                    tool_calls: vec![mutation.clone()],
                    ..Default::default()
                },
                allowed_tool_names: ["execute_command".to_string()].into_iter().collect(),
            }),
            &mut messages,
            &mut turn_messages,
            true,
            &mut persisted_turn_messages,
            &mut final_assistant_text,
            &mut final_assistant_recorded,
            &mut force_final_response,
            &mut terminal_dedupe_candidate,
            false,
            1,
            1,
            0,
            &mut turn_had_tool_error,
        )
        .unwrap();
        assert!(matches!(step, TurnLoopStep::ScopedPreflightContinue(_)));
        assert!(!force_final_response);
        assert_eq!(fs::read_to_string(&target).unwrap(), "// source\n");

        let targets =
            super::super::super::super::iteration::project_instruction_target_paths_from_tool_calls(
                std::slice::from_ref(&mutation),
                false,
            );
        let docs =
            crate::ai::agents::load_scoped_project_instruction_docs_for_targets(&targets);
        let loaded = docs
            .iter()
            .map(|doc| {
                format!(
                    "<instructions path=\"{}\">\n{}\n</instructions>",
                    doc.path, doc.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        messages[0].content = Value::String(format!("base system\n{loaded}"));
        assert!(!mutation_needs_scoped_instruction_preflight(
            &messages,
            std::slice::from_ref(&mutation)
        ));
    });
    assert!(
        rejected_tool_call_message(
            "execute_command",
            ToolCallRejectionReason::ScopedInstructionsNeedReload
        )
        .contains("No file was changed")
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn duplicate_read_only_call_ids_span_intervening_tool_calls() {
    let args = serde_json::json!({ "file_path": "/tmp/demo.txt", "offset": 1 });
    let previous = test_tool_call("call_previous", TEST_REPLAY_TOOL, args.clone());
    let current = test_tool_call("call_current", TEST_REPLAY_TOOL, args);
    let messages = vec![
        assistant_tool_call_message(previous),
        tool_result_message("call_previous", "previous result"),
        assistant_tool_call_message(test_tool_call(
            "call_other",
            TEST_REPLAY_TOOL,
            serde_json::json!({ "file_path": "/tmp/other.txt" }),
        )),
        tool_result_message("call_other", "other.rs"),
    ];

    assert_eq!(
        duplicate_read_only_call_ids(&messages, &[current]),
        HashSet::from(["call_current".to_string()])
    );
}

#[test]
fn duplicate_read_only_suppression_references_previous_successful_result() {
    let args = serde_json::json!({ "file_path": "/tmp/demo.txt", "offset": 1 });
    let previous = test_tool_call("call_previous", TEST_REPLAY_TOOL, args.clone());
    let current = test_tool_call("call_current", TEST_REPLAY_TOOL, args);
    let messages = vec![
        assistant_tool_call_message(previous),
        tool_result_message("call_previous", "previous result"),
    ];

    let suppressed = duplicate_read_only_suppressions(&messages, &messages, &[current]);
    let content = suppressed
        .get("call_current")
        .expect("duplicate suppressed");
    assert!(content.contains("call_previous"));
    assert!(!content.contains("previous result"));
}

#[test]
fn compressed_read_result_is_not_used_as_duplicate_anchor() {
    let args = serde_json::json!({ "file_path": "/tmp/demo.txt" });
    let previous = test_tool_call("call_previous", TEST_REPLAY_TOOL, args.clone());
    let current = test_tool_call("call_current", TEST_REPLAY_TOOL, args);
    let turn_messages = vec![
        assistant_tool_call_message(previous.clone()),
        tool_result_message("call_previous", "canonical file contents"),
    ];
    let messages = vec![
        assistant_tool_call_message(previous),
        tool_result_message(
            "call_previous",
            "[[PRESERVED_TOOL_OVERFLOW_STUB_V1]]\nOutput preserved in file_path: /tmp/result.txt",
        ),
    ];

    assert!(
        duplicate_read_only_call_ids_with_context(&messages, &turn_messages, &[current])
            .is_empty()
    );
}

#[test]
fn suppression_result_does_not_form_an_indirect_anchor_chain() {
    let args = serde_json::json!({ "file_path": "/tmp/demo.txt" });
    let previous = test_tool_call("call_previous", TEST_REPLAY_TOOL, args.clone());
    let suppressed = test_tool_call("call_suppressed", TEST_REPLAY_TOOL, args.clone());
    let current = test_tool_call("call_current", TEST_REPLAY_TOOL, args);
    let turn_messages = vec![
        assistant_tool_call_message(previous),
        tool_result_message("call_previous", "canonical file contents"),
        assistant_tool_call_message(suppressed.clone()),
        tool_result_message(
            "call_suppressed",
            &duplicate_read_only_suppression_message(TEST_REPLAY_TOOL, "call_previous"),
        ),
    ];
    let messages = vec![
        assistant_tool_call_message(suppressed),
        tool_result_message(
            "call_suppressed",
            &duplicate_read_only_suppression_message(TEST_REPLAY_TOOL, "call_previous"),
        ),
    ];

    assert!(
        duplicate_read_only_call_ids_with_context(&messages, &turn_messages, &[current])
            .is_empty()
    );
}

#[test]
fn successful_mutation_invalidates_previous_read_only_result() {
    let args = serde_json::json!({ "file_path": "/tmp/demo.txt", "offset": 1 });
    let previous = test_tool_call("call_previous", TEST_REPLAY_TOOL, args.clone());
    let current = test_tool_call("call_current", TEST_REPLAY_TOOL, args);
    let messages = vec![
        assistant_tool_call_message(previous),
        tool_result_message("call_previous", "old file contents"),
        assistant_tool_call_message(test_tool_call(
            "call_patch",
            "apply_patch",
            serde_json::json!({ "patch": "*** Begin Patch\n*** End Patch" }),
        )),
        tool_result_message("call_patch", "Done!"),
    ];

    assert!(duplicate_read_only_call_ids(&messages, &[current]).is_empty());
}

#[test]
fn state_writes_invalidate_generic_read_replay() {
    let cases = ["shm_write", "send_ipc_message", "save_skill", "write_file"];

    for write_name in cases {
        let args = serde_json::json!({ "resource": "demo" });
        let previous = test_tool_call("call_previous", TEST_REPLAY_TOOL, args.clone());
        let current = test_tool_call("call_current", TEST_REPLAY_TOOL, args);
        let write_args = if write_name == "write_file" {
            serde_json::json!({ "file_path": "demo.txt", "content": "new", "temp": true })
        } else {
            serde_json::json!({ "value": "new" })
        };
        let messages = vec![
            assistant_tool_call_message(previous),
            tool_result_message("call_previous", "old state"),
            assistant_tool_call_message(test_tool_call("call_write", write_name, write_args)),
            tool_result_message("call_write", "Done!"),
        ];

        assert!(
            duplicate_read_only_call_ids(&messages, &[current]).is_empty(),
            "{write_name} must invalidate cached output"
        );
    }
}

#[test]
fn failed_mutation_also_invalidates_generic_read_replay() {
    let args = serde_json::json!({ "resource": "demo" });
    let previous = test_tool_call("call_previous", TEST_REPLAY_TOOL, args.clone());
    let current = test_tool_call("call_current", TEST_REPLAY_TOOL, args);
    let messages = vec![
        assistant_tool_call_message(previous),
        tool_result_message("call_previous", "old state"),
        assistant_tool_call_message(test_tool_call(
            "call_failed_write",
            "execute_command",
            serde_json::json!({ "command": "printf new > demo.txt; false" }),
        )),
        tool_result_message("call_failed_write", "Exit code: 1"),
    ];

    assert!(duplicate_read_only_call_ids(&messages, &[current]).is_empty());
}

#[test]
fn duplicate_read_only_call_ids_do_not_cross_user_boundary() {
    let args = serde_json::json!({ "file_path": "/tmp/demo.txt" });
    let previous = test_tool_call("call_previous", TEST_REPLAY_TOOL, args.clone());
    let current = test_tool_call("call_current", TEST_REPLAY_TOOL, args);
    let messages = vec![
        assistant_tool_call_message(previous),
        tool_result_message("call_previous", "previous result"),
        Message {
            role: "user".to_string(),
            content: Value::String("read it again".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    assert!(duplicate_read_only_call_ids(&messages, &[current]).is_empty());
}

#[test]
fn browser_read_after_navigation_is_not_suppressed_as_duplicate() {
    // Browser reads target the mutable external state of the “current page”: after
    // navigating to a new page, a get_text with the same name and args is a fresh read
    // of the new page and must not be mistaken for a duplicate and suppressed.
    let read_args = serde_json::json!({ "selector": "body" });
    let previous = test_tool_call("call_previous", "mcp_browser_get_text", read_args.clone());
    let current = test_tool_call("call_current", "mcp_browser_get_text", read_args);
    let messages = vec![
        assistant_tool_call_message(previous),
        tool_result_message("call_previous", "old page text"),
        assistant_tool_call_message(test_tool_call(
            "call_nav",
            "mcp_browser_navigate",
            serde_json::json!({ "url": "https://example.com/next" }),
        )),
        tool_result_message("call_nav", "navigated"),
    ];

    assert!(duplicate_read_only_call_ids(&messages, &[current]).is_empty());
}

#[test]
fn repeated_mutating_tool_request_is_not_suppressed() {
    let args = serde_json::json!({ "command": "cargo check" });
    let previous = test_tool_call("call_previous", "execute_command", args.clone());
    let current = test_tool_call("call_current", "execute_command", args);
    let messages = vec![
        assistant_tool_call_message(previous),
        tool_result_message("call_previous", "previous result"),
    ];

    assert!(duplicate_read_only_call_ids(&messages, &[current]).is_empty());
}

#[test]
fn failed_read_only_call_is_not_suppressed() {
    let args = serde_json::json!({ "file_path": "/tmp/demo.txt" });
    let previous = test_tool_call("call_previous", TEST_REPLAY_TOOL, args.clone());
    let current = test_tool_call("call_current", TEST_REPLAY_TOOL, args);
    let messages = vec![
        assistant_tool_call_message(previous),
        tool_result_message("call_previous", "Error: file temporarily unavailable"),
    ];

    assert!(duplicate_read_only_call_ids(&messages, &[current]).is_empty());
}

#[test]
fn duplicate_knowledge_search_is_suppressed_inside_mixed_tool_batch() {
    let previous = test_tool_call(
        "call_search_previous",
        "knowledge_search",
        serde_json::json!({ "query": "durable preference" }),
    );
    let messages = vec![
        assistant_tool_call_message(previous),
        tool_result_message("call_search_previous", "1. matching preference"),
    ];
    let current = vec![
        test_tool_call(
            "call_command",
            "execute_command",
            serde_json::json!({ "command": "pwd" }),
        ),
        test_tool_call(
            "call_search_retry",
            "knowledge_search",
            serde_json::json!({
                "query": "  DURABLE PREFERENCE ",
                "category": "",
                "limit": 10
            }),
        ),
    ];

    let suppressed = duplicate_knowledge_search_call_ids(&messages, &current);
    assert_eq!(suppressed, HashSet::from(["call_search_retry".to_string()]));
}

#[test]
fn knowledge_change_allows_the_same_search_again() {
    let messages = vec![
        assistant_tool_call_message(test_tool_call(
            "call_search_previous",
            "knowledge_search",
            serde_json::json!({ "query": "durable preference" }),
        )),
        tool_result_message("call_search_previous", "1. matching preference"),
        assistant_tool_call_message(test_tool_call(
            "call_save",
            "knowledge_save",
            serde_json::json!({ "content": "new durable preference" }),
        )),
        tool_result_message("call_save", "Saved to knowledge"),
    ];
    let current = test_tool_call(
        "call_search_retry",
        "knowledge_search",
        serde_json::json!({ "query": "durable preference" }),
    );

    assert!(duplicate_knowledge_search_call_ids(&messages, &[current]).is_empty());
}

#[test]
fn failed_knowledge_search_does_not_block_retry() {
    let previous = test_tool_call(
        "call_search_previous",
        "knowledge_search",
        serde_json::json!({ "query": "durable preference" }),
    );
    let messages = vec![
        assistant_tool_call_message(previous),
        tool_result_message(
            "call_search_previous",
            "Error: knowledge database unavailable",
        ),
    ];
    let current = test_tool_call(
        "call_search_retry",
        "knowledge_search",
        serde_json::json!({ "query": "durable preference" }),
    );

    assert!(duplicate_knowledge_search_call_ids(&messages, &[current]).is_empty());
}

#[test]
fn stale_patch_target_read_is_never_replay_suppressed() {
    let path = "/tmp/patch-target.rs";
    let read_args = serde_json::json!({ "file_path": path });
    let messages = vec![
        assistant_tool_call_message(test_tool_call(
            "call_first_read",
            "read_file",
            read_args.clone(),
        )),
        tool_result_message("call_first_read", "fn current() {}\n"),
        assistant_tool_call_message(test_tool_call(
            "call_failed_patch",
            "apply_patch",
            serde_json::json!({ "file_path": path, "patch": "@@\n-old\n+new" }),
        )),
        tool_result_message(
            "call_failed_patch",
            "Error: apply_patch failed: ambiguous patch: hunk context matches 2 locations.",
        ),
    ];
    let fresh_read = test_tool_call("call_fresh_read", "read_file", read_args);

    assert!(
        duplicate_read_only_call_ids(&messages, std::slice::from_ref(&fresh_read)).is_empty(),
        "read_file is externally mutable and must always execute"
    );
}

#[test]
fn mutable_disk_and_ipc_tools_are_not_replay_registered() {
    // IPC / skill-list reads target the current process's or external mutable state:
    // they must execute against the current state.
    for name in ["read_mailbox", "shm_read", "list_skills", "load_skill"] {
        let call = test_tool_call("call", name, serde_json::json!({}));
        assert!(
            read_only_tool_signature(&call).is_none(),
            "{name} must execute against current external state"
        );
    }
    // read_file and provably read-only execute_command register as same-turn reusable
    // snapshots; mutating commands rejected by read_only_tool_signature's read-only
    // gate must still be really executed.
    let read = test_tool_call("read", "read_file", serde_json::json!({ "file_path": "/tmp/a" }));
    assert!(read_only_tool_signature(&read).is_some());
    let ro_cmd = test_tool_call(
        "ro",
        "execute_command",
        serde_json::json!({ "command": "cat /tmp/a" }),
    );
    assert!(read_only_tool_signature(&ro_cmd).is_some());
    let mutating = test_tool_call(
        "mutating",
        "execute_command",
        serde_json::json!({ "command": "cargo check" }),
    );
    assert!(read_only_tool_signature(&mutating).is_none());
    // Multi-segment commands containing a cargo verification segment must also be
    // excluded: when the first substantive segment is not cargo, it must not be
    // allowed through early.
    let chained = test_tool_call(
        "chained",
        "execute_command",
        serde_json::json!({ "command": "echo hi && cargo check" }),
    );
    assert!(read_only_tool_signature(&chained).is_none());
    let stable = test_tool_call("stable", TEST_REPLAY_TOOL, serde_json::json!({}));
    assert!(read_only_tool_signature(&stable).is_some());
}

#[test]
fn duplicate_read_file_call_is_suppressed_and_invalidated_by_mutation() {
    let read_args = serde_json::json!({ "file_path": "tmp/dup-read.rs" });
    let previous = test_tool_call("call_previous", "read_file", read_args.clone());
    let current = test_tool_call("call_current", "read_file", read_args.clone());
    let messages = vec![
        assistant_tool_call_message(previous),
        tool_result_message("call_previous", "fn one() {}\n"),
    ];
    let suppressed = duplicate_read_only_call_ids(&messages, std::slice::from_ref(&current));
    assert_eq!(
        suppressed.len(),
        1,
        "identical successful read_file must be suppressed"
    );
    assert!(suppressed.contains("call_current"));

    // Normalize: `./x` and `x` (relative paths) count as the same read with
    // identical signatures.
    let current_rel = test_tool_call(
        "call_current_rel",
        "read_file",
        serde_json::json!({ "file_path": "./tmp/dup-read.rs" }),
    );
    let suppressed_rel =
        duplicate_read_only_call_ids(&messages, std::slice::from_ref(&current_rel));
    assert_eq!(
        suppressed_rel.len(),
        1,
        "`./x` must share the read_file signature of `x`"
    );

    // A successful mutation call (write_file) between two reads invalidates the old
    // snapshot: must really read.
    let messages_with_write = vec![
        assistant_tool_call_message(test_tool_call(
            "call_previous",
            "read_file",
            read_args.clone(),
        )),
        tool_result_message("call_previous", "fn one() {}\n"),
        assistant_tool_call_message(test_tool_call(
            "call_write",
            "write_file",
            serde_json::json!({ "file_path": "tmp/dup-read.rs", "content": "fn two() {}\n" }),
        )),
        tool_result_message("call_write", "wrote 12 bytes"),
    ];
    let after_write = test_tool_call("call_after_write", "read_file", read_args);
    assert!(
        duplicate_read_only_call_ids(&messages_with_write, std::slice::from_ref(&after_write))
            .is_empty(),
        "read_file after a successful mutation must execute against current state"
    );
}

#[test]
fn duplicate_read_only_tool_call_is_suppressed_without_forcing_final_response() {
    let mut app = test_app_with_tools(&[TEST_REPLAY_TOOL]);
    let shared_mcp_client = Arc::new(std::sync::Mutex::new(McpClient::new()));
    let current_call = test_tool_call(
        "call_current",
        TEST_REPLAY_TOOL,
        serde_json::json!({ "file_path": "/tmp/demo.txt" }),
    );
    let mut messages = vec![
        assistant_tool_call_message(test_tool_call(
            "call_previous",
            TEST_REPLAY_TOOL,
            serde_json::json!({ "file_path": "/tmp/demo.txt" }),
        )),
        tool_result_message("call_previous", "previous result"),
    ];
    let mut turn_messages = messages.clone();
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut terminal_dedupe_candidate = None;
    let consecutive_truncations = 0;
    let mut force_final_response = false;
    let mut persisted_turn_messages = 0;
    let mut turn_had_tool_error = false;

    let step = handle_iteration_execution(
        &mut app,
        "read the file",
        &mcp_snapshot(&shared_mcp_client),
        &shared_mcp_client,
        IterationExecution::ToolCall(ToolCallExecution {
            stream_result: crate::ai::types::StreamResult {
                tool_calls: vec![current_call],
                ..Default::default()
            },
            allowed_tool_names: rust_tools::commonw::FastSet::from_iter([
                TEST_REPLAY_TOOL.to_string()
            ]),
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
        consecutive_truncations,
        &mut turn_had_tool_error,
    )
    .unwrap();

    assert!(matches!(step, TurnLoopStep::Continue));
    assert!(!force_final_response);
    assert!(!turn_had_tool_error);
    let rejected_tool_result = messages
        .iter()
        .rev()
        .find(|message| message.role == "tool")
        .expect("rejection should append a tool result");
    assert!(
        rejected_tool_result
            .content
            .as_str()
            .unwrap_or_default()
            .contains("Duplicate read-only call")
    );
    assert!(
        rejected_tool_result
            .content
            .as_str()
            .unwrap_or_default()
            .contains("call_previous")
    );
}
