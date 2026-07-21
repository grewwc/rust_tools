//! `coalesce_accumulated_summary_notes` 的单元测试。
//!
//! 覆盖压缩 bug 的 P1 修复：历史上因 `is_summary_message` 漏认
//! `长期记忆摘要（压缩保留）` 前缀，每轮压缩重复插入「摘要 + 归档」note 对，
//! 长 session 堆积几十对。折叠逻辑需无损收敛这些堆积 note，同时绝不触碰
//! 正常历史。

use super::*;

fn note(content: &str) -> Message {
    Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn msg(role: &str, content: &str) -> Message {
    Message {
        role: role.to_string(),
        content: Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn summary_note(body: &str) -> Message {
    note(&format!("长期记忆摘要（压缩保留）:\n{body}"))
}

fn archive_note() -> Message {
    note("长期记忆归档: 见 overflow-history.md（原文已归档）")
}

/// note 合计 ≤ 2 条时不应改写（返回值须与入参逐条相等，避免无谓落盘）。
#[test]
fn coalesce_leaves_small_histories_untouched() {
    let input = vec![
        msg("system", "sys"),
        msg("user", "hi"),
        summary_note("目标 A"),
        archive_note(),
        msg("assistant", "ok"),
    ];
    let out = coalesce_accumulated_summary_notes(input.clone());
    assert_eq!(out, input, "note 合计=2 不应触发折叠");
}

/// 堆积多对「摘要 + 归档」note 应被折叠为单条合并摘要 + 单条归档指针，
/// 且所有不同的摘要正文都要保留（无损）。
#[test]
fn coalesce_merges_accumulated_notes_losslessly() {
    let input = vec![
        msg("system", "sys"),
        summary_note("目标 A"),
        archive_note(),
        summary_note("目标 B"),
        archive_note(),
        summary_note("目标 C"),
        archive_note(),
        msg("user", "继续"),
        msg("assistant", "好的"),
    ];
    let out = coalesce_accumulated_summary_notes(input);

    let summaries: Vec<&Message> = out.iter().filter(|m| is_summary_message(m)).collect();
    assert_eq!(summaries.len(), 1, "多条摘要应被折叠为一条");
    let archives: Vec<&Message> = out.iter().filter(|m| is_archive_note_message(m)).collect();
    assert_eq!(archives.len(), 1, "多条归档指针应去重为一条");

    // 三个不同目标全部保留（无损）。
    let merged = value_to_string(&summaries[0].content);
    assert!(merged.contains("目标 A"), "merged={merged}");
    assert!(merged.contains("目标 B"), "merged={merged}");
    assert!(merged.contains("目标 C"), "merged={merged}");

    // 非摘要/归档消息原样保留、顺序不变。
    assert_eq!(out.first().map(|m| m.role.as_str()), Some("system"));
    assert!(
        out.iter()
            .any(|m| m.role == "user" && value_to_string(&m.content) == "继续")
    );
    assert!(
        out.iter()
            .any(|m| m.role == "assistant" && value_to_string(&m.content) == "好的")
    );
}

/// 重复正文只保留一份，去重保序。
#[test]
fn coalesce_dedups_identical_summary_bodies() {
    let input = vec![
        summary_note("同一目标"),
        archive_note(),
        summary_note("同一目标"),
        archive_note(),
        summary_note("同一目标"),
        archive_note(),
    ];
    let out = coalesce_accumulated_summary_notes(input);
    let summaries: Vec<&Message> = out.iter().filter(|m| is_summary_message(m)).collect();
    assert_eq!(summaries.len(), 1);
    let merged = value_to_string(&summaries[0].content);
    // "同一目标" 只应出现一次（去重）。
    assert_eq!(merged.matches("同一目标").count(), 1, "merged={merged}");
}

/// 合并后的摘要 note 必须能被 `is_summary_message` 识别——否则折叠一次后
/// 下一轮又会被当成"新内容"重复处理，等于没修好防重复 guard。
#[test]
fn coalesced_summary_is_recognized_by_guard() {
    let input = vec![
        summary_note("目标 A"),
        archive_note(),
        summary_note("目标 B"),
        archive_note(),
        summary_note("目标 C"),
        archive_note(),
    ];
    let out = coalesce_accumulated_summary_notes(input);
    let summary = out
        .iter()
        .find(|m| is_summary_message(m))
        .expect("应有合并摘要");
    // 幂等性：对已折叠结果再折叠一次，note 数不再变化。
    let again = coalesce_accumulated_summary_notes(out.clone());
    assert_eq!(
        again
            .iter()
            .filter(|m| is_summary_or_archive_note(m))
            .count(),
        out.iter().filter(|m| is_summary_or_archive_note(m)).count(),
        "折叠应幂等"
    );
    assert!(is_summary_note_text(&value_to_string(&summary.content)));
}
