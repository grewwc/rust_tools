//! Differential regression tests: the batched drop (`drop_trim_candidates_batch`)
//! must produce byte-identical results to the OLD sequential drop loop
//! (`first_trim_candidate` + `Vec::remove` one message per round, keep and
//! boundary recomputed every round from the shrinking list).
//!
//! The parity is non-obvious (see `batch_drop_drift_is_load_bearing_*` below),
//! so this file pins it with deterministic shapes covering:
//! - both keep regimes (total above/below the 48K base) and the 2->3 flip
//!   mid-stretch;
//! - the byte-cap (protected tail exceeding `max_chars` forces keep down);
//! - deletion of real users below the window (the boundary-drift path);
//! - preserved user stubs, checkpoint markers, tool results, assistant
//!   tool-call messages and leading protected system messages (skip predicates).

use super::*;
use crate::ai::types::{FunctionCall, ToolCall};
use serde_json::Value;

fn msg(role: &str, content: &str) -> Message {
    Message {
        role: role.to_string(),
        content: Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn assistant_call(id: &str) -> Message {
    Message {
        role: "assistant".to_string(),
        content: Value::String("thinking".to_string()),
        tool_calls: Some(vec![ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn long_text(prefix: &str, n: usize) -> String {
    prefix.repeat(n)
}

/// Faithful replica of the OLD sequential drop loop from
/// `shrink_messages_to_fit_with_summary`: one `first_trim_candidate` +
/// `Vec::remove` per round, with the `total > max_chars` loop guard checked
/// before every round exactly like the original.
fn sequential_drop_loop(
    messages: &mut Vec<Message>,
    max_chars: usize,
    total: &mut usize,
) -> (Vec<Message>, Vec<Message>) {
    let mut dropped: Vec<Message> = Vec::new();
    let mut dropped_internal_notes: Vec<Message> = Vec::new();
    loop {
        if *total <= max_chars {
            break;
        }
        if let Some(idx) = first_trim_candidate(messages, max_chars) {
            let removed_msg = messages.remove(idx);
            *total = total.saturating_sub(message_billable_chars(&removed_msg));
            if is_internal_note_role(&removed_msg.role) {
                dropped_internal_notes.push(removed_msg);
            } else {
                dropped.push(removed_msg);
            }
        } else {
            break;
        }
    }
    (dropped, dropped_internal_notes)
}

/// Outer loop mirroring the real caller: keep calling the batched drop until it
/// reports no removals.
fn batch_drop_loop(
    messages: &mut Vec<Message>,
    max_chars: usize,
    total: &mut usize,
) -> (Vec<Message>, Vec<Message>) {
    let mut dropped: Vec<Message> = Vec::new();
    let mut dropped_internal_notes: Vec<Message> = Vec::new();
    let mut snap: Option<Vec<Message>> = None;
    loop {
        if drop_trim_candidates_batch(
            messages,
            max_chars,
            total,
            &mut snap,
            &mut dropped,
            &mut dropped_internal_notes,
        ) > 0
        {
            continue;
        }
        break;
    }
    (dropped, dropped_internal_notes)
}

/// Copy of `drop_trim_candidates_batch` with ONLY the boundary-drift line
/// removed: on deleting a below-boundary user the boundary ordinal stays put
/// (which is the "obviously correct" physical interpretation). This variant
/// exists solely to prove the drift is load-bearing: it trims MORE than the
/// sequential loop (it keeps deleting candidates below the physical boundary
/// after `user_count` drops to `keep`, which sequential protects by
/// recomputing `retained_turn_start == 0`).
fn drop_trim_candidates_batch_exact(
    messages: &mut Vec<Message>,
    max_chars: usize,
    total: &mut usize,
    messages_before_first_drop: &mut Option<Vec<Message>>,
    dropped: &mut Vec<Message>,
    dropped_internal_notes: &mut Vec<Message>,
) -> usize {
    let len = messages.len();
    if len == 0 || *total <= max_chars {
        return 0;
    }
    let chars: Vec<usize> = messages.iter().map(message_billable_chars).collect();
    let tail_chars = |keep: usize| -> usize {
        let start = retained_turn_start(messages, keep);
        chars[start..].iter().sum()
    };
    let capped_keep = |base: usize| -> usize {
        let mut keep = base;
        while keep > 1 && tail_chars(keep) > max_chars {
            keep -= 1;
        }
        keep
    };
    let keep2 = capped_keep(2);
    let keep3 = capped_keep(3);
    let user_positions: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(idx, message)| {
            (message.role == "user" && !is_runtime_synthetic_user_message(message)).then_some(idx)
        })
        .collect();
    let user_count = user_positions.len();
    let mut alive = vec![true; user_count];
    let mut alive_users = user_count;
    let mut keep_now = if *total <= KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS {
        keep3
    } else {
        keep2
    };
    let mut boundary_ptr: isize = if user_count <= keep_now {
        -1
    } else {
        (user_count - keep_now) as isize
    };
    let mut tombstones = vec![false; len];
    let mut removed = 0usize;
    let mut user_ordinal = 0usize;
    let mut idx = 0usize;
    while idx < len && is_protected_leading_system_like_message(&messages[idx]) {
        idx += 1;
    }
    let mut head_run_end = idx;
    while idx < len {
        if idx < head_run_end {
            idx = head_run_end;
            continue;
        }
        let boundary = if boundary_ptr >= 0 {
            user_positions[boundary_ptr as usize]
        } else {
            0
        };
        if idx >= boundary || *total <= max_chars {
            break;
        }
        let message = &messages[idx];
        if is_context_checkpoint_marker(message) {
            idx += 1;
            continue;
        }
        if is_preserved_user_or_image_stub(&value_to_string(&message.content)) {
            idx += 1;
            continue;
        }
        if message.role == "tool" {
            idx += 1;
            continue;
        }
        if message.role == "assistant"
            && message
                .tool_calls
                .as_ref()
                .map(|calls| !calls.is_empty())
                .unwrap_or(false)
        {
            idx += 1;
            continue;
        }
        if messages_before_first_drop.is_none() {
            *messages_before_first_drop = Some(messages.clone());
        }
        tombstones[idx] = true;
        removed += 1;
        *total = total.saturating_sub(chars[idx]);
        while user_ordinal < user_count && user_positions[user_ordinal] < idx {
            user_ordinal += 1;
        }
        if user_ordinal < user_count && user_positions[user_ordinal] == idx {
            alive[user_ordinal] = false;
            alive_users -= 1;
            user_ordinal += 1;
            // EXACT variant: no drift here. boundary_ptr stays put because the
            // physical boundary user never changes on below-boundary deletion.
        }
        let new_keep = if *total <= KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS {
            keep3
        } else {
            keep2
        };
        while keep_now < new_keep {
            keep_now += 1;
            if boundary_ptr >= 0 {
                if alive_users <= keep_now {
                    boundary_ptr = -1;
                } else {
                    boundary_ptr = prev_alive_user(&alive, boundary_ptr - 1);
                }
            }
        }
        if idx == head_run_end {
            head_run_end = idx + 1;
            while head_run_end < len
                && is_protected_leading_system_like_message(&messages[head_run_end])
            {
                head_run_end += 1;
            }
            idx = head_run_end;
        } else {
            idx += 1;
        }
        if *total <= max_chars {
            break;
        }
    }
    if removed == 0 {
        return 0;
    }
    let old = std::mem::take(messages);
    let mut kept = Vec::with_capacity(old.len() - removed);
    for (index, message) in old.into_iter().enumerate() {
        if tombstones[index] {
            if is_internal_note_role(&message.role) {
                dropped_internal_notes.push(message);
            } else {
                dropped.push(message);
            }
        } else {
            kept.push(message);
        }
    }
    *messages = kept;
    removed
}

fn batch_drop_loop_exact(
    messages: &mut Vec<Message>,
    max_chars: usize,
    total: &mut usize,
) -> (Vec<Message>, Vec<Message>) {
    let mut dropped: Vec<Message> = Vec::new();
    let mut dropped_internal_notes: Vec<Message> = Vec::new();
    let mut snap: Option<Vec<Message>> = None;
    loop {
        if drop_trim_candidates_batch_exact(
            messages,
            max_chars,
            total,
            &mut snap,
            &mut dropped,
            &mut dropped_internal_notes,
        ) > 0
        {
            continue;
        }
        break;
    }
    (dropped, dropped_internal_notes)
}

fn run_case(name: &str, messages: Vec<Message>, max_chars: usize) {
    let mut seq = messages.clone();
    let mut seq_total = messages_total_chars(&seq);
    let (seq_dropped, seq_internal) = sequential_drop_loop(&mut seq, max_chars, &mut seq_total);

    let mut bat = messages;
    let mut bat_total = messages_total_chars(&bat);
    let (bat_dropped, bat_internal) = batch_drop_loop(&mut bat, max_chars, &mut bat_total);

    assert_eq!(
        format!("{:?}", seq),
        format!("{:?}", bat),
        "{name}: final messages differ"
    );
    assert_eq!(
        format!("{:?}", seq_dropped),
        format!("{:?}", bat_dropped),
        "{name}: dropped differ"
    );
    assert_eq!(
        format!("{:?}", seq_internal),
        format!("{:?}", bat_internal),
        "{name}: dropped internal notes differ"
    );
    assert_eq!(seq_total, bat_total, "{name}: total differ");
}

/// Build a history: `head` protected system messages, then `big_block` long
/// plain assistant messages (each ~1200 chars, all candidates) so the total
/// reliably exceeds the 48K keep base and stays high through several
/// removals, then `users` real users separated by `gap` plain assistants,
/// then `tail` plain assistants.
fn build_history(
    head: usize,
    big_block: usize,
    users: usize,
    gap: usize,
    tail: usize,
) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::new();
    for i in 0..head {
        out.push(msg("system", &format!("system-{i}")));
    }
    for b in 0..big_block {
        out.push(msg("assistant", &long_text(&format!("big-{b}-"), 100)));
    }
    for u in 0..users {
        out.push(msg("user", &format!("user-{u}")));
        for g in 0..gap {
            out.push(msg("assistant", &format!("assistant-{u}-{g}")));
        }
    }
    for t in 0..tail {
        out.push(msg("assistant", &format!("tail-{t}")));
    }
    out
}

/// big_block = 60 => ~72K total (keep=2 until it crosses 48K mid-stretch);
/// big_block = 30 => ~36K total (keep=3 throughout). Covers the boundary-drift
/// shapes in both keep regimes plus the 48K 2->3 flip.
#[test]
fn batch_drop_matches_sequential_drift_shapes() {
    for big in [60usize, 30] {
        for users in [4usize, 5, 6, 8] {
            for gap in [0usize, 1, 3] {
                for head in [0usize, 1, 3] {
                    run_case(
                        &format!("drift big={big} u={users} gap={gap} head={head}"),
                        build_history(head, big, users, gap, 2),
                        1_000,
                    );
                }
            }
        }
    }
}

/// Keep=2 -> 3 crossing via a single huge early user message, plus the
/// byte-cap path (protected tail exceeding `max_chars` forces keep down to 1).
#[test]
fn batch_drop_matches_sequential_48k_cross_and_byte_cap() {
    let mut m = build_history(1, 10, 4, 2, 2);
    m.insert(1, msg("user", &long_text("x", 49_000)));
    run_case("48k-cross", m, 1_000);

    let mut m = build_history(1, 20, 3, 2, 0);
    m.push(msg("assistant", &long_text("fat-tail", 5_000)));
    run_case("byte-cap-keep1", m, 400);
}

/// Shape that actually deletes below-boundary USERS mid-stretch (keep=2,
/// total stays above 48K): six users with huge contents. This is the case
/// where the boundary drift matters - it must match sequential, and the
/// non-drifting ("exact") boundary must NOT (it keeps trimming below the
/// physical boundary after user_count drops to keep, which sequential
/// protects via retained_turn_start == 0).
#[test]
fn batch_drop_drift_is_load_bearing_for_keep2_user_deletion() {
    let mut m: Vec<Message> = vec![msg("system", "sys")];
    for u in 0..6 {
        m.push(msg(
            "user",
            &long_text(&format!("user-{u}-payload-"), 2_000),
        )); // ~22K chars
        m.push(msg("assistant", &format!("assistant-{u}")));
    }
    m.push(msg("assistant", "tail-0"));
    m.push(msg("assistant", "tail-1"));

    // Sanity: total > 48K so keep=2 throughout (even after deleting users).
    let total = messages_total_chars(&m);
    assert!(
        total > KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS,
        "total={total}"
    );
    // After deleting 4 below-boundary users the total must STILL exceed 48K.
    assert!(
        total - 4 * message_billable_chars(&msg("user", &long_text("x", 2_000)))
            > KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS
    );

    run_case("keep2-user-deletion", m.clone(), 1_000);

    // Demonstrate the drift is load-bearing: sequential == drifted batch,
    // but the exact (non-drifting) boundary diverges (it trims more).
    let mut seq = m.clone();
    let mut seq_total = messages_total_chars(&seq);
    let (_, _) = sequential_drop_loop(&mut seq, 1_000, &mut seq_total);
    let mut drifted = m.clone();
    let mut drifted_total = messages_total_chars(&drifted);
    let (_, _) = batch_drop_loop(&mut drifted, 1_000, &mut drifted_total);
    let mut exact = m.clone();
    let mut exact_total = messages_total_chars(&exact);
    let (_, _) = batch_drop_loop_exact(&mut exact, 1_000, &mut exact_total);

    assert_eq!(
        format!("{:?}", seq),
        format!("{:?}", drifted),
        "drifted batch must match sequential"
    );
    assert_ne!(
        format!("{:?}", seq),
        format!("{:?}", exact),
        "exact (non-drifting) boundary must diverge from sequential - the drift \
         reproduces sequential's protect-everything stop once user_count <= keep"
    );
}

/// Skip predicates parity: preserved user stub (counted as a real user, never
/// deletable), checkpoint marker, assistant tool-call message and tool result
/// are all skipped identically by both paths.
#[test]
fn batch_drop_matches_sequential_special_messages() {
    let mut m = build_history(1, 20, 4, 1, 1);
    m.insert(
        1,
        msg(
            "user",
            "[[PRESERVED_CONTENT_STUB_V1]]{\"kind\":\"user\",\"file_path\":\"/tmp/x\"}",
        ),
    );
    m.insert(
        2,
        msg("assistant", "[context_checkpoint summary: scratch test"),
    );
    m.insert(3, assistant_call("call-1"));
    m.insert(4, msg("tool", "tool result"));
    run_case("special-messages", m, 2_000);
}

/// Persisted reference parts render as short markers: image-only messages still
/// summarize to "[图片]", text-file/PDF references render as their file name
/// (never the raw path or content), and real user text is preserved.
#[test]
fn value_to_string_renders_reference_boundary_markers() {
    let content = serde_json::json!([
        { "type": "reference", "kind": "image", "name": "shot.png", "path": "/assets/shot.png" },
        { "type": "reference", "kind": "file", "name": "service.rs", "path": "/tmp/service.rs" },
        { "type": "reference", "kind": "audio", "name": "clip.m4a", "path": "/tmp/clip.m4a" },
        { "type": "text", "text": "帮我 review 这个文件" }
    ]);
    let rendered = value_to_string(&content);
    assert!(rendered.contains("帮我 review 这个文件"));
    assert!(rendered.contains("[Attached file: service.rs]"));
    // Unknown/future reference kinds render a marker instead of leaking raw JSON.
    assert!(rendered.contains("[audio: clip.m4a]"));
    assert!(!rendered.contains("/tmp/service.rs"));
    assert!(!rendered.contains("/tmp/clip.m4a"));
    assert!(!rendered.contains("shot.png"));

    // A message that is only images still collapses to the "[图片]" marker.
    let image_only = serde_json::json!([
        { "type": "reference", "kind": "image", "name": "shot.png", "path": "/assets/shot.png" }
    ]);
    assert_eq!(value_to_string(&image_only), "[图片]".to_string());
}
