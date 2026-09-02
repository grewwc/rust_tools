//! Mid-turn (in-request) compression: current-turn precision/lossless
//! protection and LLM-summary fallback.

use super::*;

/// Last-resort rescue for the reactive overflow path, for the case where
/// mid-turn compression can no longer make progress: its policies never
/// truncate user messages, so an oversized current user message would
/// otherwise fail the turn outright. Offloads the middle of the last **real**
/// user message to the overflow archive and replaces it with a head+tail
/// preview stub, using the same machinery as mutable assistant fields
/// ([`truncate_mutable_field`]).
///
/// Only call this after the provider actually rejected the request: the
/// pre-request soft budget must keep the current user message intact so
/// legitimate large contexts reach the provider unchanged. Returns true when
/// the message was truncated. Refuses (returns false, message untouched) when
/// the overflow archive write fails: a preview-only stub would be the only
/// surviving copy of the user's instruction and could never be read back, so
/// the caller must surface the provider error instead of retrying on an
/// unrecoverable fragment. Marker-prefixed content is always re-archived
/// through the trusted session sink before it is collapsed, so an embedded
/// user-controlled path is never treated as provenance.
pub(in crate::ai) fn truncate_last_real_user_message_to_fit(
    messages: &mut [Message],
    target_chars: usize,
    overflow_dir: Option<&Path>,
) -> bool {
    let total = messages_total_chars(messages);
    if total <= target_chars {
        return false;
    }
    let Some(index) = last_real_user_index(messages) else {
        return false;
    };
    let message = &mut messages[index];
    // Multimodal (array) content must not be flattened into a text stub —
    // that would drop image parts. Only plain string content can be offloaded.
    if message.content.as_str().is_none() {
        return false;
    }
    truncate_mutable_field(
        message,
        MutableMessageField::Content,
        total - target_chars,
        overflow_dir,
        // Required: the current user instruction must stay recoverable via
        // the archive; without an archived copy the rescue must not fire.
        FieldArchivePolicy::Required,
    )
}

pub(in crate::ai) fn current_turn_precision_tool_call_ids(messages: &[Message]) -> rustc_hash::FxHashSet<String> {
    let mut out = rustc_hash::FxHashSet::default();
    // Synthetic user messages do not form a turn boundary: otherwise precision
    // tool results from earlier turns of this round would lose protection and be
    // lossy truncated by Path C. If there is no real user at all, the whole
    // history counts as the current synthetic turn, consistent with
    // retained_turn_start's conservative boundary.
    let current_turn_start = last_real_user_index(messages).unwrap_or(0);
    for message in messages.iter().skip(current_turn_start) {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            if is_non_compressible_tool(&tool_call.function.name)
                && crate::ai::tools::tool_history_policy(&tool_call.function.name)
                    .counts_toward_precision_inline_budget()
            {
                out.insert(tool_call.id.clone());
            }
        }
    }
    out
}

/// Collect every tool call in the current turn that forbids lossy compaction. It
/// is wider than the precision inline set: aggregated results like `task_wait`
/// are not part of the precision quota, but their bodies likewise must not be
/// truncated by Path C.
pub(in crate::ai) fn current_turn_lossless_tool_call_ids(messages: &[Message]) -> rustc_hash::FxHashSet<String> {
    let mut out = rustc_hash::FxHashSet::default();
    // When there is no real user, lossless-mandatory results of the synthetic
    // turn must not be exposed to Path C.
    let current_turn_start = last_real_user_index(messages).unwrap_or(0);
    for message in messages.iter().skip(current_turn_start) {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            if !crate::ai::tools::tool_history_policy(&tool_call.function.name)
                .allows_lossy_compress()
            {
                out.insert(tool_call.id.clone());
            }
        }
    }
    out
}

/// Public proxy of [`messages_total_chars`] for callers in other ai modules
/// (e.g. mid-turn compression in `turn_runtime`) that need to check budget
/// without re-implementing the same accounting.
pub(in crate::ai) fn messages_total_chars_pub(messages: &[Message]) -> usize {
    messages_total_chars(messages)
}

pub(in crate::ai) const CONTEXT_COMPACTION_STATE_PREFIX: &str = "[runtime context state]";

pub(in crate::ai) const CONTEXT_COMPACTION_STATE: &str = "[runtime context state]\n\
- This request uses a compacted context projection and has passed the runtime budget guard.\n\
- Folded or truncated tool output is recoverable evidence; it does not mean the model context is full.\n\
- Prefer the stub's original_file_path/original_range. Read archive_file_path only when the original source is unavailable.\n\
- Report context exhaustion only when the provider returns an explicit context-length error.\n\
- Continue from the latest working checkpoint and verify uncertain details from the cited source.";

pub(in crate::ai) fn is_context_compaction_state(message: &Message) -> bool {
    message.role == ROLE_INTERNAL_NOTE
        && message
            .content
            .as_str()
            .is_some_and(|content| content.starts_with(CONTEXT_COMPACTION_STATE_PREFIX))
}

pub(in crate::ai) fn upsert_context_compaction_state(messages: &mut Vec<Message>) {
    messages.retain(|message| !is_context_compaction_state(message));
    let insert_at = last_real_user_index(messages).map_or(messages.len(), |index| index + 1);
    messages.insert(
        insert_at,
        Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(CONTEXT_COMPACTION_STATE.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    );
}

/// Mid-turn progressive compaction: reuse the first tiers of the cross-turn
/// compaction pipeline inside the iteration loop. Only "lossless/weakly-lossy"
/// operations; system messages untouched and the most recent keep_recent tool
/// messages never deleted:
///   1. dedup_repeated_tool_results — older results with the same (tool, args)
///      folded into stubs
///   2. prepare_tool_messages_structured — remote tool results trimmed by line to
///      480 chars
///   3. fold_tool_call_group_to_stub  — still over budget: fold the whole remote
///      (assistant + tool) group
/// Returns: (messages_after, before_chars, after_chars)
pub(in crate::ai) fn mid_turn_compress(
    messages: Vec<Message>,
    soft_threshold: usize,
    overflow_dir: Option<&Path>,
    cwd: Option<&Path>,
) -> (Vec<Message>, usize, usize) {
    let before = messages_total_chars(&messages);
    let messages = trim_compressed_tool_evidence_to_inline_budget(messages, overflow_dir);
    let after_evidence_trim = messages_total_chars(&messages);
    if after_evidence_trim <= soft_threshold {
        return (messages, before, after_evidence_trim);
    }
    let mut out = messages;
    // Hand the compaction state to the model explicitly, so it does not misread
    // recoverable evidence stubs as a full context. Inserted before any trimming;
    // the later budget calculation folds this fixed overhead in.
    upsert_context_compaction_state(&mut out);
    // 0. Clean up stale reasoning_content: multiple reasoning chains returned by
    //    the LLM within one turn add nothing to later decisions, but some vendors
    //    require historical reasoning to pair with tool_calls. Keep only the last
    //    assistant's reasoning_content; set the rest to None.
    keep_only_recent_reasoning_content(&mut out);
    if messages_total_chars(&out) <= soft_threshold {
        let after = messages_total_chars(&out);
        return (out, before, after);
    }
    // 1. Dedupe tool results with the same signature
    let protected_tool_call_ids = current_turn_precision_tool_call_ids(&out);
    dedup_repeated_tool_results(&mut out, &protected_tool_call_ids);
    if messages_total_chars(&out) <= soft_threshold {
        let after = messages_total_chars(&out);
        return (out, before, after);
    }
    // 2. Structured remote trimming: the middle of each tool result is folded by
    //    line down to 480 chars per entry; the most recent 6 keep full text.
    //    When overflow_dir is passed, large outputs of "non-compressible" tools
    //    like read_file/grep spill to the session file with zero compression and
    //    leave a head+tail preview stub (consistent with cross-turn compaction),
    //    freeing context without losing information — the model can re-read via
    //    the stub's file_path.
    prepare_tool_messages_structured(
        &mut out,
        480,
        KEEP_RECENT_TOOL_GROUPS,
        overflow_dir,
        cwd,
        &protected_tool_call_ids,
    );
    if messages_total_chars(&out) <= soft_threshold {
        let after = messages_total_chars(&out);
        return (out, before, after);
    }
    // 3. Still over budget: use shrink_messages_to_fit for "fold tool groups +
    //    overall backstop"
    out = shrink_messages_to_fit(
        out,
        soft_threshold,
        overflow_dir,
        cwd,
        &protected_tool_call_ids,
    );
    let after = messages_total_chars(&out);
    (out, before, after)
}

/// Minimum net decrease (chars) for an LLM summary to count as "effective
/// compaction". Below this it is considered ineffective and `was_effective`
/// returns false; the hard-budget backstop may still return a slightly smaller
/// context result. Same order of magnitude as `summary_max_chars`: if the net
/// decrease is smaller than the injected summary text itself, the compressor is
/// spinning (typical symptom: "295K shrank to 294K and stopped").
pub(in crate::ai) const MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS: usize = 4_000;

/// Path C backstop: per-message cap for head+tail truncating a single oversized
/// non-system message inside the tail window. Triggers only when progressive
/// folding still leaves the context over `hard_target` — prefer truncation over
/// letting the model 4xx.
pub(in crate::ai) const PATH_C_PER_MSG_CAP: usize = 8_000;

/// Mid-turn LLM summary backstop: called when the lossless/weakly-lossy pipeline
/// still leaves the context over threshold. Three complementary paths:
///   - Path A (cross-turn summary): if conversation remains before the most recent
///     `keep_recent_turns` user turns, call the LLM summarizer to compress that
///     span into a single `internal_note` injected before the tail window; also
///     fold older tool groups inside the tail window, so "bloat concentrated in
///     the newest turn" can still shrink.
///   - Path B+C (progressive folding): start from `keep_recent=4` (equivalent to
///     the original Path B) and shrink the protection window step by step to 2→1,
///     until compaction is effective or the context drops below `hard_target`.
///     Fixes compressor spin when "all the bloat sits inside the protected tail
///     window and early history has nothing left to fold".
///   - Path C backstop (per-message truncation): when progressive folding still
///     exceeds `hard_target`, head+tail truncate a single oversized non-system
///     message in the tail window. This is the absolute last resort.
/// All leading system / internal_note messages (agent instructions, tool lists,
/// global guidance) are always kept verbatim. Returns
/// `(messages_after, before, after, was_effective, llm_summary_inserted)`;
/// `was_effective` is true only when the net decrease is >=
/// [`MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS`]. false does not mean the returned
/// messages are unchanged; the hard-budget backstop may produce a partial decrease
/// below the effective threshold. `llm_summary_inserted` says whether Path A
/// actually ran and injected `[mid-turn-summary]`: false with `after < before`
/// means the decrease came entirely from mechanical paths (fold/truncate/spill),
/// letting the upper report distinguish "LLM summary executed" from "purely
/// mechanical compaction" and avoid false reporting.
pub(in crate::ai) async fn mid_turn_llm_summarize(
    app: &App,
    messages: Vec<Message>,
    keep_recent_turns: usize,
    summary_max_chars: usize,
    hard_target: usize,
    cwd: Option<&Path>,
) -> (Vec<Message>, usize, usize, bool, bool) {
    let before = messages_total_chars(&messages);
    let overflow_dir = crate::ai::history::SessionStore::new(app.config.history_file.as_path())
        .session_assets_dir(&app.session_id);
    let protected_tool_call_ids = current_turn_precision_tool_call_ids(&messages);
    let lossless_tool_call_ids = current_turn_lossless_tool_call_ids(&messages);
    // best tracks the smallest result so far; None means the original messages are
    // still in use.
    let mut best: Option<Vec<Message>> = None;
    let mut best_after = before;
    // Whether Path A actually ran and injected [mid-turn-summary] (see the return
    // doc).
    let mut llm_summary_inserted = false;

    // === Path A: cross-turn LLM summary ===
    // First compute the cut point as "keep the most recent keep_recent_turns user
    // turns". After upstream projection compaction, older user messages may already
    // have been replaced by internal_note summaries (role != "user"), leaving fewer
    // visible user boundaries in the projection than keep_recent_turns, so
    // retained_turn_start returns 0. That does not mean there is no compactable old
    // content — before the first user message there may still be
    // assistant(tool_calls)/tool records protected by protocol pairing (impossible
    // to delete one by one). In that case fall the cut point back to the first user
    // message position: the trailing user turns stay protected, the leading
    // system-like summary/archive markers are kept by preserved_system_end, and the
    // old conversation span between them can be reclaimed by the LLM summary.
    let mut split_at = retained_turn_start(&messages, keep_recent_turns);
    if split_at == 0 {
        if let Some(first_user) = messages.iter().position(|m| m.role == "user") {
            if first_user > 0 {
                split_at = first_user;
            }
        }
    }
    if split_at > 0 && split_at < messages.len() {
        // Keep the leading contiguous run of system-like messages (agent
        // instructions etc.) and summarize only the conversation span after them.
        // An early version dropped the messages[0] system prompt outright, which
        // made the model instantly lose its agent behavior instructions — observed
        // as "replies cut off abruptly / extremely short / off track after
        // compaction".
        let preserved_system_end = messages[..split_at]
            .iter()
            .position(|m| !is_system_like_role(&m.role))
            .unwrap_or(split_at);
        let earlier = &messages[preserved_system_end..split_at];
        // Extract context checkpoint markers from the to-be-summarized span: they
        // are the only index locating saved checkpoint bodies and must never be
        // swallowed by the summary. The regular persisted-compaction path already
        // does the same; this closes the gap here.
        let checkpoint_markers: Vec<Message> = earlier
            .iter()
            .filter(|m| is_context_checkpoint_marker(m))
            .cloned()
            .collect();
        let summary_source: Vec<Message> = earlier
            .iter()
            .filter(|m| !is_context_checkpoint_marker(m))
            .cloned()
            .collect();
        let has_dialog = earlier
            .iter()
            .any(|m| m.role == "user" || m.role == "assistant")
            || !checkpoint_markers.is_empty();
        if has_dialog {
            let summary =
                build_persisted_summary_text_with_app(app, &summary_source, summary_max_chars)
                    .await;
            if !summary.trim().is_empty() {
                let archive_file_path = overflow_dir.join(OVERFLOW_HISTORY_FILENAME);
                let tail_plan = plan_early_tool_groups(
                    &messages[split_at..],
                    MID_TURN_LLM_SUMMARY_KEEP_RECENT_TOOL_GROUPS,
                    Some(overflow_dir.as_path()),
                    &protected_tool_call_ids,
                );
                let mut out =
                    Vec::with_capacity(preserved_system_end + 2 + (messages.len() - split_at));
                // 1. Leading system / internal_notes (agent instructions etc.)
                //    kept verbatim
                out.extend_from_slice(&messages[..preserved_system_end]);
                // 2. The summary is injected as an internal_note
                //    (normalize_messages_for_request classifies it as a Summary
                //    heading and merges it into the system message)
                out.push(Message {
                    role: ROLE_INTERNAL_NOTE.to_string(),
                    content: Value::String(format!(
                        "[mid-turn-summary] 早期工具调用与对话已被 LLM 摘要：\n{summary}"
                    )),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
                insert_archive_note_if_missing(
                    &mut out,
                    build_overflow_placeholder(&archive_file_path.to_string_lossy()),
                );
                // 2b. Put the extracted context checkpoint markers back, keeping
                //     their re-readable index
                out.extend(checkpoint_markers.iter().cloned());
                // 3. Tail-window folding is planned only at first; disk writes
                //    happen uniformly after the whole Path A candidate is confirmed
                //    better than the current best.
                out.extend_from_slice(tail_plan.messages());
                let after = messages_total_chars(&out);
                // Commit the tail-window fold first and archive `earlier` only
                // after the candidate is confirmed adopted: archive appends to
                // overflow-history.md (non-idempotent), so archiving early and then
                // failing the commit would leave `earlier` on disk while the
                // context never adopted `out`, and the next compaction round would
                // archive the same messages again -> orphan accumulation. The
                // short-circuit `&&` guarantees the archive is never touched when
                // commit fails; if commit succeeds but archiving fails, `best` is
                // not updated and the context still keeps `earlier` — no data loss
                // (only an idempotently hash-named fold file remains).
                if after < best_after
                    && tail_plan.commit()
                    && archive_messages_to_overflow(earlier, Some(overflow_dir.as_path())).is_some()
                {
                    best = Some(out);
                    best_after = after;
                    llm_summary_inserted = true;
                }
                // Effective compaction meeting the target -> return directly
                if before.saturating_sub(best_after) >= MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS
                    && best_after <= hard_target
                {
                    return (best.unwrap(), before, best_after, true, true);
                }
            }
        }
    }

    // === Path B+C: progressive tool-group folding ===
    // Start from keep_recent=4 (equivalent to the original Path B) and shrink the
    // protection window step by step to 2→1 (never to 0), until compaction is
    // effective or the context drops below hard_target. Fixes the spin when "all
    // the bloat sits inside the protected tail window". Folding chains on best
    // (the Path A result or the original messages): already-folded groups became
    // stubs (internal_notes) and will not match fold_early_tool_groups again, so
    // each iteration folds only the groups the previous round kept, progressively
    // releasing the protected tail window. The window never drops to 0 (see
    // [`MIN_KEEP_RECENT_TOOL_GROUPS`]): the most recent 1 group stays verbatim,
    // and remaining excess is handled by the Path C per-message truncation
    // backstop below, avoiding stub-izing the most recent tool interaction too.
    for &keep_recent in progressive_fold_windows().iter() {
        if best_after <= hard_target {
            break;
        }
        let current = best.as_ref().unwrap_or(&messages);
        let plan = plan_early_tool_groups(
            current,
            keep_recent,
            Some(overflow_dir.as_path()),
            &protected_tool_call_ids,
        );
        if plan.folded_groups() == 0 {
            continue;
        }
        let after = messages_total_chars(plan.messages());
        if after < best_after && plan.commit() {
            let (folded, _) = plan.into_result();
            best = Some(folded);
            best_after = after;
        }
    }

    // Only truly reaching hard_target allows an early return. The old logic
    // returned as soon as the net decrease exceeded 4K, skipping the hard backstop
    // below and letting "older groups already saved a lot, but the newest group
    // alone still overflows the window" keep sending over-limit requests.
    if best_after <= hard_target {
        let was_effective = before.saturating_sub(best_after) >= MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS;
        return (
            best.unwrap_or(messages),
            before,
            best_after,
            was_effective,
            llm_summary_inserted,
        );
    }

    // === Path C backstop: budget-aware structure-preserving truncation ===
    // Keep the assistant↔tool pairing of system/user messages and the most recent
    // tool groups; compress only re-retrievable result bodies, reasoning, and
    // oversized tool arguments. Unlike the old "8K per message", this keeps
    // tightening against the total budget, so it converges even with many parallel
    // tool results. If the untrimmable system/user content itself is over budget,
    // return the smallest achievable result instead of corrupting the user's task
    // text.
    let mut result = best.unwrap_or(messages);
    // Before the Path C backstop, spill every current-turn result that forbids
    // lossy compaction with zero compression: persist the original as a
    // re-readable asset and replace it with a stub, so the immediately following
    // `emergency_cap_messages_to_fit` cannot lossy-truncate that grounding
    // evidence to 8K / ~160 chars, making the original unrecoverable.
    spill_protected_precision_to_fit(
        &mut result,
        hard_target,
        Some(overflow_dir.as_path()),
        cwd,
        &lossless_tool_call_ids,
    );
    emergency_cap_messages_to_fit(
        &mut result,
        hard_target,
        PATH_C_PER_MSG_CAP,
        Some(overflow_dir.as_path()),
        &lossless_tool_call_ids,
    );
    let after = messages_total_chars(&result);
    let savings = before.saturating_sub(after);
    (
        result,
        before,
        after,
        savings >= MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS,
        llm_summary_inserted,
    )
}
