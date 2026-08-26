//! `tool_call_id`-aware unit tests for `dedup_adjacent`.
//!
//! Historical bug: adjacent tool results were deduped when their text/signature
//! matched, without comparing `tool_call_id`. Parallel tool calls (multiple
//! tool_calls issued in the same assistant turn) often return identical content,
//! so the second result was wrongly dropped, breaking the assistant
//! tool_call <-> tool result pairing. After the fix, dedup only happens when
//! `tool_call_id` also matches (a true duplicate).

use super::*;

fn tool_result(id: &str, content: &str) -> Message {
    Message {
        role: "tool".to_string(),
        content: Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: Some(id.to_string()),
        reasoning_content: None,
    }
}

/// Parallel batch: two different `tool_call_id`s returning identical content
/// must both be kept.
#[test]
fn keeps_parallel_tool_results_with_different_call_ids() {
    let mut messages = vec![tool_result("call_A", "done"), tool_result("call_B", "done")];
    dedup_adjacent(&mut messages);
    assert_eq!(
        messages.len(),
        2,
        "different tool_call_id must not be deduped"
    );
    assert_eq!(messages[0].tool_call_id.as_deref(), Some("call_A"));
    assert_eq!(messages[1].tool_call_id.as_deref(), Some("call_B"));
}

/// Same `tool_call_id` plus identical content is a true duplicate and should be
/// deduped to one.
#[test]
fn drops_genuine_duplicate_with_same_call_id() {
    let mut messages = vec![tool_result("call_A", "done"), tool_result("call_A", "done")];
    dedup_adjacent(&mut messages);
    assert_eq!(
        messages.len(),
        1,
        "same tool_call_id + same content is a real dup"
    );
    assert_eq!(messages[0].tool_call_id.as_deref(), Some("call_A"));
}

/// Different content should never be deduped (regression protection).
#[test]
fn keeps_adjacent_tool_results_with_different_content() {
    let mut messages = vec![
        tool_result("call_A", "result-1"),
        tool_result("call_B", "result-2"),
    ];
    dedup_adjacent(&mut messages);
    assert_eq!(messages.len(), 2);
}
