//! `keep_recent_user_turns_when_trimming` 的字节上限逃逸阀单元测试。
//!
//! 覆盖变更 B：保护尾窗原先仅按「user 轮数」定义、无字节上限，tool-heavy
//! agentic 会话（少 user 轮 × 每轮上百次工具调用）会让尾窗撑到 MB 级且结构上
//! 禁止收敛。加入字节上限后，尾窗过大时逐步收缩保护轮数（最少 1 轮），把更早
//! 的工具组暴露给 fold/spill 路径。正常小尾窗行为零变化。

use super::*;

fn msg(role: &str, content: &str) -> Message {
    Message {
        role: role.to_string(),
        content: Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

/// 正常小会话：多个 user 轮但尾窗很小，行为不变——
/// ≤48K → 3 轮；无 budget（budget=0）时永远走基础判定。
#[test]
fn tail_window_keeps_baseline_for_small_sessions() {
    let messages = vec![
        msg("user", "q1"),
        msg("assistant", "a1"),
        msg("user", "q2"),
        msg("assistant", "a2"),
        msg("user", "q3"),
        msg("assistant", "a3"),
        msg("user", "q4"),
        msg("assistant", "a4"),
    ];
    // 小尾窗 + 大 budget：保持 3 轮（≤48K 分支）。
    assert_eq!(keep_recent_user_turns_when_trimming(&messages, 90_000), 3);
    // budget=0 显式关闭上限：仍走基础判定。
    assert_eq!(keep_recent_user_turns_when_trimming(&messages, 0), 3);
}

/// tool-heavy：仅 2 个 user 轮，但最近这轮尾窗 billable ≫ budget。
/// 逃逸阀应把保护轮数下调到 1，让倒数第 2 轮及更早暴露给收敛路径。
#[test]
fn tail_window_shrinks_when_bytes_exceed_budget() {
    let huge = "x".repeat(60_000);
    let messages = vec![
        msg("user", "第一轮问题"),
        msg("assistant", &huge), // 第一轮的巨大工具/回复体量
        msg("user", "第二轮问题"),
        msg("assistant", &huge), // 第二轮同样巨大
    ];
    // 总量已 > 48K → 基础判定给 2；但「保留 2 轮」的尾窗（从第一个 user 起）
    // billable ≈ 120K ≫ budget(40K)，逃逸阀下调到 1。
    assert_eq!(keep_recent_user_turns_when_trimming(&messages, 40_000), 1);
}

/// 保底不变式：即便最新一轮自身就超 budget，也永不低于 1 轮
/// （最新一轮 user 及其工具组必须逐字保留，由组级保护继续兜底）。
#[test]
fn tail_window_never_drops_below_one_turn() {
    let huge = "y".repeat(200_000);
    let messages = vec![msg("user", "唯一一轮"), msg("assistant", &huge)];
    assert_eq!(keep_recent_user_turns_when_trimming(&messages, 10_000), 1);
}

/// budget 足够大能容纳按基础判定算出的尾窗时，不触发下调。
#[test]
fn tail_window_no_shrink_when_budget_accommodates() {
    let mid = "z".repeat(30_000);
    let messages = vec![
        msg("user", "q1"),
        msg("assistant", &mid),
        msg("user", "q2"),
        msg("assistant", &mid),
    ];
    // 总量 ≈ 60K > 48K → 基础判定 2；尾窗（2 user 轮 = 全量）billable ≈ 60K
    // ≤ budget(90K)，不触发下调，保持 2。
    assert_eq!(keep_recent_user_turns_when_trimming(&messages, 90_000), 2);
}
