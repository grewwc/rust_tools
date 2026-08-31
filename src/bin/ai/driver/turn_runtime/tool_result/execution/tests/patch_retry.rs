//! Tests for the `patch_retry` cluster.

use super::common::*;
use super::super::*;

#[test]
fn context_mismatch_does_not_require_fresh_read() {
    let path = "/tmp/patch-target.rs";
    let messages = vec![
        assistant_tool_call_message(test_tool_call(
            "call_failed_patch",
            "apply_patch",
            serde_json::json!({ "file_path": path, "patch": "@@\n-old\n+new" }),
        )),
        tool_result_message(
            "call_failed_patch",
            "Error: apply_patch failed: context mismatch: patch hunk could not be located.\nMismatched lines (showing 1 of 1):\n  line 12: expected \"ambiguous patch: stale source text\", found \"current source text\"\nCurrent file text at this location (copy verbatim, no line-number prefix):\n<<<PATCH_TEXT\ncurrent source text\nPATCH_TEXT>>>",
        ),
    ];
    let retry = test_tool_call(
        "call_retry",
        "apply_patch",
        serde_json::json!({ "path": path, "patch": "@@\n-old\n+newer" }),
    );

    let ledger = ledger_from_messages(&messages);
    assert!(ledger.is_empty());
    assert!(!patch_retry_requires_fresh_read(&ledger, &[retry]));
}

#[test]
fn patch_retry_is_released_by_successful_read_of_same_target() {
    let path = "/tmp/patch-target.rs";
    let messages = vec![
        assistant_tool_call_message(test_tool_call(
            "call_failed_patch",
            "apply_patch",
            serde_json::json!({ "file_path": path, "patch": "@@\n-old\n+new" }),
        )),
        tool_result_message(
            "call_failed_patch",
            "Error: apply_patch failed: ambiguous patch: hunk context matches 2 locations.",
        ),
        assistant_tool_call_message(test_tool_call(
            "call_fresh_read",
            "read_file",
            serde_json::json!({ "path": path }),
        )),
        tool_result_message("call_fresh_read", "fn current() {}\n"),
    ];
    let retry = test_tool_call(
        "call_retry",
        "apply_patch",
        serde_json::json!({ "file_path": path, "patch": "@@\n-old\n+newer" }),
    );

    let ledger = ledger_from_messages(&messages);
    assert!(!patch_retry_requires_fresh_read(&ledger, &[retry]));
}

#[test]
fn patch_retry_is_not_released_by_read_of_another_target() {
    let patch_path = "/tmp/patch-target.rs";
    let messages = vec![
        assistant_tool_call_message(test_tool_call(
            "call_failed_patch",
            "apply_patch",
            serde_json::json!({
                "patch": format!(
                    "*** Begin Patch\n*** Update File: {patch_path}\n@@\n-old\n+new\n*** End Patch"
                )
            }),
        )),
        tool_result_message(
            "call_failed_patch",
            "Error: apply_patch failed: ambiguous patch: hunk context matches 2 locations.",
        ),
        assistant_tool_call_message(test_tool_call(
            "call_other_read",
            "read_file",
            serde_json::json!({ "file_path": "/tmp/another-target.rs" }),
        )),
        tool_result_message("call_other_read", "unrelated current content\n"),
    ];
    let retry = test_tool_call(
        "call_retry",
        "apply_patch",
        serde_json::json!({ "file_path": patch_path, "patch": "@@\n-old\n+newer" }),
    );

    let ledger = ledger_from_messages(&messages);
    assert!(patch_retry_requires_fresh_read(&ledger, &[retry]));
}

#[test]
fn patch_retry_multi_file_failure_blocks_only_failed_target() {
    let a = "/tmp/patch-a.rs";
    let b = "/tmp/patch-b.rs";
    let messages = vec![
        assistant_tool_call_message(test_tool_call(
            "call_failed_patch",
            "apply_patch",
            serde_json::json!({
                "patch": format!(
                    "*** Begin Patch\n*** Update File: {a}\n@@\n-old_a\n+new_a\n*** Update File: {b}\n@@\n-old_b\n+new_b\n*** End Patch"
                )
            }),
        )),
        tool_result_message(
            "call_failed_patch",
            &format!(
                "Error: apply_patch failed: failed while preparing patch for {b}: ambiguous patch: hunk context matches 2 locations."
            ),
        ),
    ];
    let retry_a = test_tool_call(
        "call_retry_a",
        "apply_patch",
        serde_json::json!({ "file_path": a, "patch": "@@\n-old_a\n+newer_a" }),
    );
    let retry_b = test_tool_call(
        "call_retry_b",
        "apply_patch",
        serde_json::json!({ "file_path": b, "patch": "@@\n-old_b\n+newer_b" }),
    );

    let ledger = ledger_from_messages(&messages);
    assert!(!patch_retry_requires_fresh_read(&ledger, &[retry_a]));
    assert!(patch_retry_requires_fresh_read(&ledger, &[retry_b]));
}

#[test]
fn patch_retry_multi_file_relative_targets_match_normalized_error_path() {
    let a = "audit-relative/patch-a.rs";
    let b = "audit-relative/patch-b.rs";
    let normalized_b = FileStore::new(PathBuf::from(b)).path().to_path_buf();
    let messages = vec![
        assistant_tool_call_message(test_tool_call(
            "call_failed_patch",
            "apply_patch",
            serde_json::json!({
                "patch": format!(
                    "*** Begin Patch\n*** Update File: {a}\n@@\n-old_a\n+new_a\n*** Update File: {b}\n@@\n-old_b\n+new_b\n*** End Patch"
                )
            }),
        )),
        tool_result_message(
            "call_failed_patch",
            &format!(
                "Error: apply_patch failed: failed while preparing patch for {}: ambiguous patch: hunk context matches 2 locations.",
                normalized_b.display()
            ),
        ),
    ];

    let ledger = ledger_from_messages(&messages);
    assert_eq!(ledger, rustc_hash::FxHashSet::from_iter([normalized_b]));
}

#[test]
fn patch_retry_target_path_may_contain_patch_text_marker() {
    let a = "/tmp/patch-a.rs";
    let b = "/tmp/patch<<<PATCH_TEXT.rs";
    let messages = vec![
        assistant_tool_call_message(test_tool_call(
            "call_failed_patch",
            "apply_patch",
            serde_json::json!({
                "patch": format!(
                    "*** Begin Patch\n*** Update File: {a}\n@@\n-old_a\n+new_a\n*** Update File: {b}\n@@\n-old_b\n+new_b\n*** End Patch"
                )
            }),
        )),
        tool_result_message(
            "call_failed_patch",
            &format!(
                "Error: apply_patch failed: failed while preparing patch for {b}: ambiguous patch: hunk context matches 2 locations.\n{}current text\nPATCH_TEXT>>>",
                crate::ai::tools::PATCH_TEXT_BLOCK_START
            ),
        ),
    ];

    let ledger = ledger_from_messages(&messages);
    assert_eq!(
        ledger,
        rustc_hash::FxHashSet::from_iter([FileStore::new(PathBuf::from(b))
            .path()
            .to_path_buf()])
    );
}

#[test]
fn patch_retry_multi_file_failure_is_released_after_failed_target_is_re_read() {
    let a = "/tmp/patch-a.rs";
    let b = "/tmp/patch-b.rs";
    let messages = vec![
        assistant_tool_call_message(test_tool_call(
            "call_failed_patch",
            "apply_patch",
            serde_json::json!({
                "patch": format!(
                    "*** Begin Patch\n*** Update File: {a}\n@@\n-old_a\n+new_a\n*** Update File: {b}\n@@\n-old_b\n+new_b\n*** End Patch"
                )
            }),
        )),
        tool_result_message(
            "call_failed_patch",
            &format!(
                "Error: apply_patch failed: failed while preparing patch for {b}: ambiguous patch: hunk context matches 2 locations."
            ),
        ),
        assistant_tool_call_message(test_tool_call(
            "call_read_a",
            "read_file",
            serde_json::json!({ "file_path": a }),
        )),
        tool_result_message("call_read_a", "fn current_a() {}\n"),
        assistant_tool_call_message(test_tool_call(
            "call_read_b",
            "read_file",
            serde_json::json!({ "path": b }),
        )),
        tool_result_message("call_read_b", "1| fn current_b() {}\n"),
    ];
    let retry = test_tool_call(
        "call_retry_b",
        "apply_patch",
        serde_json::json!({ "file_path": b, "patch": "@@\n-old_b\n+newer_b" }),
    );

    let ledger = ledger_from_messages(&messages);
    assert!(!patch_retry_requires_fresh_read(&ledger, &[retry]));
}

#[test]
fn patch_retry_without_fresh_read_is_rejected() {
    let mut app = test_app_with_tools(&["apply_patch", "read_file"]);
    let shared_mcp_client = Arc::new(std::sync::Mutex::new(McpClient::new()));
    let path = "/tmp/patch-target.rs";
    let current_call = test_tool_call(
        "call_retry",
        "apply_patch",
        serde_json::json!({ "file_path": path, "patch": "@@\n-old\n+new" }),
    );
    let mut messages = vec![
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
    // The ledger is the guard's truth source: equivalent to the state that
    // update_stale_patch_targets settled from this failure history at the end of the
    // previous handle_tool_call_round. Even if the history messages are later
    // compressed/folded, the ledger survives independently.
    app.stale_patch_targets = ledger_from_messages(&messages);
    let mut turn_messages = Vec::new();
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut terminal_dedupe_candidate = None;
    let consecutive_truncations = 0;
    let mut force_final_response = false;
    let mut persisted_turn_messages = 0;
    let mut turn_had_tool_error = false;

    let step = handle_iteration_execution(
        &mut app,
        "update the file",
        &mcp_snapshot(&shared_mcp_client),
        &shared_mcp_client,
        IterationExecution::ToolCall(ToolCallExecution {
            stream_result: crate::ai::types::StreamResult {
                tool_calls: vec![current_call],
                ..Default::default()
            },
            allowed_tool_names: rust_tools::commonw::FastSet::from_iter([
                "apply_patch".to_string(),
                "read_file".to_string(),
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
    assert!(turn_had_tool_error);
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
            .contains("apply_patch retry blocked")
    );
}

/// Core regression: after apply_patch fails with ambiguous patch, the ledger
/// remembers the stale target; even when the failed round is later fully erased
/// from `messages` by history compression (simulated as folded into an
/// internal_note stub), the guard still blocks retries on the same path from the
/// ledger. This is exactly the scenario where the old message-scanning
/// implementation failed.
#[test]
fn stale_patch_guard_survives_history_compression_via_ledger() {
    let mut ledger: rustc_hash::FxHashSet<PathBuf> = Default::default();

    // Round 1: apply_patch fails on table.rs (ambiguous patch).
    let failed_patch = test_tool_call(
        "call_patch_1",
        "apply_patch",
        serde_json::json!({ "file_path": "/tmp/proj/table.rs", "patch": "*** Update File: /tmp/proj/table.rs\n@@\n-old\n+new\n" }),
    );
    update_stale_patch_targets(
        &mut ledger,
        std::slice::from_ref(&failed_patch),
        &[tool_result(
            "call_patch_1",
            "Error: apply_patch failed: ambiguous patch: hunk context matches 2 locations.",
        )],
    );
    let normalized = FileStore::new(PathBuf::from("/tmp/proj/table.rs"))
        .path()
        .to_path_buf();
    assert!(
        ledger.contains(&normalized),
        "failed patch target must be recorded in the ledger"
    );

    // Simulate history compression: the failed round's structured messages are
    // folded and fully vanish from messages. The old implementation derived stale
    // state from messages and would miss this; the ledger is unaffected.
    let retry_patch = test_tool_call(
        "call_patch_2",
        "apply_patch",
        serde_json::json!({ "file_path": "/tmp/proj/table.rs", "patch": "*** Update File: /tmp/proj/table.rs\n@@\n-old2\n+new2\n" }),
    );
    assert!(
        patch_retry_requires_fresh_read(&ledger, std::slice::from_ref(&retry_patch)),
        "guard must block stale retry using the ledger even after the failed round was compressed out of messages"
    );
}

/// After a successful read_file re-reads the truth for the same path, the ledger
/// releases the target and the guard allows later patches. Verifies the recovery
/// chain converges normally (no permanent lockout).
#[test]
fn stale_patch_guard_clears_after_fresh_read() {
    let mut ledger: rustc_hash::FxHashSet<PathBuf> = Default::default();
    let normalized = FileStore::new(PathBuf::from("/tmp/proj/table.rs"))
        .path()
        .to_path_buf();
    ledger.insert(normalized.clone());

    // Successful read_file on the same target → the ledger releases it.
    let fresh_read = test_tool_call(
        "call_read_1",
        "read_file",
        serde_json::json!({ "file_path": "/tmp/proj/table.rs" }),
    );
    update_stale_patch_targets(
        &mut ledger,
        std::slice::from_ref(&fresh_read),
        &[tool_result("call_read_1", "   1\tfn table() {}\n")],
    );
    assert!(
        !ledger.contains(&normalized),
        "successful read_file must clear the stale target"
    );

    let retry_patch = test_tool_call(
        "call_patch_2",
        "apply_patch",
        serde_json::json!({ "file_path": "/tmp/proj/table.rs", "patch": "*** Update File: /tmp/proj/table.rs\n@@\n-a\n+b\n" }),
    );
    assert!(
        !patch_retry_requires_fresh_read(&ledger, std::slice::from_ref(&retry_patch)),
        "guard must allow the retry once the target has been freshly read"
    );
}

#[test]
fn stale_patch_ledger_tracks_delete_file_envelope_targets() {
    let mut ledger: rustc_hash::FxHashSet<PathBuf> = Default::default();
    let failed_delete = test_tool_call(
        "call_delete",
        "apply_patch",
        serde_json::json!({
            "patch": "*** Begin Patch\n*** Delete File: /tmp/proj/obsolete.rs\n*** End Patch",
        }),
    );

    update_stale_patch_targets(
        &mut ledger,
        std::slice::from_ref(&failed_delete),
        &[tool_result(
            "call_delete",
            "Error: apply_patch failed: ambiguous patch: hunk context matches 2 locations.",
        )],
    );

    let normalized = FileStore::new(PathBuf::from("/tmp/proj/obsolete.rs"))
        .path()
        .to_path_buf();
    assert!(ledger.contains(&normalized));
    assert!(patch_retry_requires_fresh_read(
        &ledger,
        std::slice::from_ref(&failed_delete)
    ));
}
