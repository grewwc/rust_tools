//! Character-budget checks for
//! `keep_recent_user_turns_when_trimming`.
//!
//! Reduce user-turn protection when the tail exceeds the total budget, retaining
//! at least one turn so the newest user input remains protected.

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

/// Normal small session: several user turns but a small tail window, behavior
/// unchanged — ≤48K → 3 turns; no budget (budget=0) always falls back to the
/// baseline decision.
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
    // Small tail window + large budget: keep 3 turns (the ≤48K branch).
    assert_eq!(keep_recent_user_turns_when_trimming(&messages, 90_000), 3);
    // budget=0 explicitly disables the cap: still uses the baseline decision.
    assert_eq!(keep_recent_user_turns_when_trimming(&messages, 0), 3);
}

/// Tool-heavy: only 2 user turns, but the tail window's billable for the
/// latest turn ≫ budget. The escape valve should lower the protected turn
/// count to 1, exposing the second-to-last turn and earlier to the
/// convergence path.
#[test]
fn tail_window_shrinks_when_bytes_exceed_budget() {
    let huge = "x".repeat(60_000);
    let messages = vec![
        msg("user", "第一轮问题"),
        msg("assistant", &huge), // Huge tool/reply payload from the first turn
        msg("user", "第二轮问题"),
        msg("assistant", &huge), // Second turn is equally huge
    ];
    // Total already > 48K → baseline gives 2; but the "keep 2 turns" tail
    // window (from the first user on) is billable ≈ 120K ≫ budget(40K), so the
    // escape valve drops it to 1.
    assert_eq!(keep_recent_user_turns_when_trimming(&messages, 40_000), 1);
}

/// Even if the latest turn alone exceeds the budget, retain its user boundary.
/// Tool-group folding has its own retention policy.
#[test]
fn tail_window_never_drops_below_one_turn() {
    let huge = "y".repeat(200_000);
    let messages = vec![msg("user", "唯一一轮"), msg("assistant", &huge)];
    assert_eq!(keep_recent_user_turns_when_trimming(&messages, 10_000), 1);
}

/// When the budget is large enough to accommodate the tail window computed by
/// the baseline, no shrink is triggered.
#[test]
fn tail_window_no_shrink_when_budget_accommodates() {
    let mid = "z".repeat(30_000);
    let messages = vec![
        msg("user", "q1"),
        msg("assistant", &mid),
        msg("user", "q2"),
        msg("assistant", &mid),
    ];
    // Total ≈ 60K > 48K → baseline 2; the tail window (2 user turns = the
    // whole set) is billable ≈ 60K ≤ budget(90K), so no shrink, stays 2.
    assert_eq!(keep_recent_user_turns_when_trimming(&messages, 90_000), 2);
}
