//! `dedup_adjacent` 的 `tool_call_id` 感知单元测试。
//!
//! 历史 bug：相邻 tool 结果若文本/签名相同即被去重，未比对 `tool_call_id`。
//! 并行工具调用（同一 assistant 轮发起多个 tool_call）经常返回相同内容，
//! 第二条结果会被误删，破坏 assistant tool_call ↔ tool result 的配对。
//! 修复后只有 `tool_call_id` 也相同（真重复）才去重。

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

/// 并行批次：两个不同 `tool_call_id` 返回相同内容，必须双双保留。
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

/// 同一 `tool_call_id` + 相同内容才是真重复，应去重为一条。
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

/// 不同内容本就不应被去重（回归保护）。
#[test]
fn keeps_adjacent_tool_results_with_different_content() {
    let mut messages = vec![
        tool_result("call_A", "result-1"),
        tool_result("call_B", "result-2"),
    ];
    dedup_adjacent(&mut messages);
    assert_eq!(messages.len(), 2);
}
