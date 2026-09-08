//! Unit tests for the byte-budget escape valve of
//! `keep_recent_user_turns_when_trimming`.
//!
//! Covers change B: the protected tail window was originally defined only by
//! "user turn count" with no byte cap, so tool-heavy agentic sessions (few user
//! turns x hundreds of tool calls per turn) could grow the tail window to MB
//! scale and structurally prevent convergence. With the byte cap in place, an
//! oversized tail window gradually shrinks the number of protected turns (at
//! least 1), exposing earlier tool groups to the fold/spill path. Normal small
//! tail-window behavior is unchanged.

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

/// Floor invariant: even if the latest turn alone exceeds the budget, never go
/// below 1 turn (the latest user turn and its tool groups must be preserved
/// verbatim; group-level protection continues to provide the safety net).
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
