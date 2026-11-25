//! Mid-turn (in-request) compression: current-turn precision/lossless
//! protection and LLM-summary fallback.

use super::*;
use crate::ai::history::{HistoryMessageSummarizer, is_user_input_message};
use std::sync::atomic::{AtomicBool, Ordering};

/// Shared tail policy for request-level and persisted LLM summary projections.
/// Keep the most recent real user turns verbatim before summarizing older spans.
pub(in crate::ai) const MID_TURN_LLM_SUMMARY_KEEP_RECENT_TURNS: usize = 2;
/// Match the resident increment window so persisting request-level compression
/// never reduces the summary's capacity to retain decisions and prerequisites.
pub(in crate::ai) const MID_TURN_LLM_SUMMARY_MAX_CHARS: usize = 8_000;

/// Last-resort rescue for the reactive overflow path, for the case where
/// mid-turn compression can no longer make progress: its policies never
/// truncate user messages, so an oversized current user message would
/// otherwise fail the turn outright. Offloads the middle of the last **real**
/// user message (or its file snapshots) to the overflow archive and replaces
/// those text fields with head+tail previews, using the same machinery as
/// mutable assistant fields ([`truncate_mutable_field`]).
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
    if message.content.is_array() {
        return truncate_user_file_references_to_fit(message, total - target_chars, overflow_dir);
    }
    if !message.content.is_string() {
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

/// Keep the array's provenance and question intact while rescuing file snapshots.
/// Image arrays retain their existing protection; reply parts are never truncated.
fn truncate_user_file_references_to_fit(
    message: &mut Message,
    reduce_by: usize,
    overflow_dir: Option<&Path>,
) -> bool {
    if crate::ai::request::content_has_image(&message.content) {
        return false;
    }
    let Some(overflow_dir) = overflow_dir else {
        return false;
    };
    let archive_path = overflow_dir.join(OVERFLOW_HISTORY_FILENAME);
    let keep_floor =
        (super::truncate::content_truncation_prefix(Some(&archive_path.to_string_lossy()))
            .chars()
            .count()
            + super::truncate::MIN_CONTENT_PREVIEW_CHARS)
            .max(160);
    let parts = message.content.as_array().expect("array content");
    let mut files: Vec<_> = parts
        .iter()
        .enumerate()
        .filter(|(_, part)| {
            part.get("type").and_then(Value::as_str) == Some("reference")
                && part.get("kind").and_then(Value::as_str) == Some("file")
        })
        .filter_map(|(index, part)| {
            let chars = part.get("text")?.as_str()?.chars().count();
            (chars > 160).then_some((index, chars))
        })
        .collect();
    files.sort_by_key(|(_, chars)| std::cmp::Reverse(*chars));
    let before = message_billable_chars(message);
    let target = before.saturating_sub(reduce_by);
    // Stage every cut so a failed archive write cannot publish a partial rescue.
    let mut candidate = message.clone();
    for (index, chars) in files {
        let remaining = message_billable_chars(&candidate).saturating_sub(target);
        if remaining == 0 {
            break;
        }
        let reduction = remaining.min(chars.saturating_sub(keep_floor));
        if reduction == 0 {
            continue;
        }
        if !truncate_mutable_field(
            &mut candidate,
            MutableMessageField::FileReferenceText(index),
            reduction,
            Some(overflow_dir),
            FieldArchivePolicy::Required,
        ) {
            return false;
        }
    }
    if message_billable_chars(&candidate) >= before {
        return false;
    }
    *message = candidate;
    true
}

/// Char budget for current-turn precision result protection. Traverse newest
/// results first and include the result that exhausts the budget whole, even
/// when that one result exceeds the quota.
pub(in crate::ai) const PRECISION_TAIL_MAX_CHARS: usize = 120_000;

pub(in crate::ai) fn current_turn_precision_tool_call_ids(
    messages: &[Message],
) -> rustc_hash::FxHashSet<String> {
    let mut out = rustc_hash::FxHashSet::default();
    // Synthetic user messages do not form a turn boundary. Traverse newest
    // precision results first, including the result that exhausts the budget,
    // so recent evidence stays protected without pinning the entire turn.
    let current_turn_start = last_real_user_index(messages).unwrap_or(0);
    let mut id_to_name: rustc_hash::FxHashMap<&str, &str> = rustc_hash::FxHashMap::default();
    for message in messages.iter().skip(current_turn_start) {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            id_to_name.insert(tool_call.id.as_str(), tool_call.function.name.as_str());
        }
    }
    let mut budget = PRECISION_TAIL_MAX_CHARS;
    for index in (current_turn_start..messages.len()).rev() {
        if messages[index].role != "tool" {
            continue;
        }
        let Some(id) = messages[index].tool_call_id.as_ref() else {
            continue;
        };
        let Some(name) = id_to_name.get(id.as_str()).copied() else {
            continue;
        };
        if !super::tool_overflow::is_non_compressible_tool(name)
            || !crate::ai::tools::tool_history_policy(name).counts_toward_precision_inline_budget()
        {
            continue;
        }
        let chars = messages[index]
            .content
            .as_str()
            .map_or(0, |text| text.chars().count());
        budget = budget.saturating_sub(chars);
        out.insert(id.clone());
        if budget == 0 {
            break;
        }
    }
    out
}

/// Collect every tool call in the current turn that forbids lossy compaction. It
/// is wider than the precision inline set: aggregated results like `task_wait`
/// are not part of the precision quota, but their bodies likewise must not be
/// truncated by Path C.
pub(in crate::ai) fn current_turn_lossless_tool_call_ids(
    messages: &[Message],
) -> rustc_hash::FxHashSet<String> {
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
- For exact historical output, read the stub's saved archive; only for current content, read a small relevant range of original_file_path.\n\
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

/// Compact a request projection within a turn: cap accumulated evidence notes,
/// trim unmarked reasoning, deduplicate results, prepare structured tool output,
/// then use `shrink_messages_to_fit` if still over budget.
/// Returns `(messages_after, before_chars, after_chars)`.
pub(in crate::ai) fn mid_turn_compress(
    messages: Vec<Message>,
    soft_threshold: usize,
    overflow_dir: Option<&Path>,
    cwd: Option<&Path>,
) -> (Vec<Message>, usize, usize) {
    let before = messages_total_chars(&messages);
    let messages = trim_compressed_tool_evidence_to_inline_budget(messages, overflow_dir);
    // Re-bound the memory-increment window before this loop can append another
    // record, so a long turn cannot grow the head past the cap plus the newest one.
    let messages = trim_incremental_summary_notes_to_inline_budget(
        messages,
        overflow_dir,
        MAX_INCREMENTAL_SUMMARY_INLINE_CHARS,
    );
    let after_evidence_trim = messages_total_chars(&messages);
    if after_evidence_trim <= soft_threshold {
        return (messages, before, after_evidence_trim);
    }
    let mut out = messages;
    // Hand the compaction state to the model explicitly, so it does not misread
    // recoverable evidence stubs as a full context. Inserted before any trimming;
    // the later budget calculation folds this fixed overhead in.
    upsert_context_compaction_state(&mut out);
    // Trim unmarked reasoning according to `keep_only_recent_reasoning_content`;
    // exact continuation state stays with its protocol messages.
    keep_only_recent_reasoning_content(&mut out);
    if messages_total_chars(&out) <= soft_threshold {
        let after = messages_total_chars(&out);
        return (out, before, after);
    }
    // 1. Dedupe tool results with the same signature. Only the inline precision
    //    tail belongs in this set: spilling and group folding keep the original
    //    recoverable through the archive, so results outside the tail must stay
    //    eligible even when their tool forbids *lossy* compaction. The lossy
    //    truncation pass in `shrink_messages_to_fit` re-adds the lossless ids
    //    where truncating in place would actually lose evidence.
    let protected_tool_call_ids = current_turn_precision_tool_call_ids(&out);
    dedup_repeated_tool_results(&mut out, &protected_tool_call_ids);
    if messages_total_chars(&out) <= soft_threshold {
        let after = messages_total_chars(&out);
        return (out, before, after);
    }
    // 2. Structured tool preparation uses bounded summaries while protecting
    //    the most recent KEEP_RECENT_TOOL_GROUPS tool groups.
    //    When overflow_dir is passed, large outputs of "non-compressible" tools
    //    like read_file/grep spill to the session file with zero compression and
    //    leave a head+tail preview stub (consistent with cross-turn compaction),
    //    freeing context without losing information — the model can re-read via
    //    the stub's file_path.
    //    Results below MID_TURN_SPILL_MIN_CHARS stay inline: folding them frees
    //    too little budget to pay for the risk that the model trusts the preview
    //    instead of the original.
    //    Dedup runs first: a back-reference stub must point at a still-raw
    //    canonical copy, and an already-archived copy can no longer be folded.
    dedup_repeated_tool_results(&mut out, &protected_tool_call_ids);
    prepare_tool_messages_structured(
        &mut out,
        480,
        MID_TURN_SPILL_MIN_CHARS,
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
/// compaction". Below this the outcome's `effective` flag is false;
/// the hard-budget backstop may still return a slightly smaller
/// context result. Same order of magnitude as `summary_max_chars`: if the net
/// decrease is smaller than the injected summary text itself, the compressor is
/// spinning (typical symptom: "295K shrank to 294K and stopped").
pub(in crate::ai) const MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS: usize = 4_000;

/// Initial per-field cap for emergency compaction. Further cuts may be needed
/// to meet the total `hard_target`; user/system fields remain excluded.
pub(in crate::ai) const PATH_C_PER_MSG_CAP: usize = 8_000;

/// Minimum chars for mid-turn compaction step 2 to replace a non-compressible
/// tool result (`read_file`, `execute_command`, ...) with a preview stub.
/// Folding a smaller result frees too little budget to pay for the risk that the
/// model reads the preview as the content and skips re-reading the original,
/// while genuine bloat (large file reads, big command output) is far above this
/// threshold and still folds. Step 3 (`shrink_messages_to_fit`) uses the shared
/// per-result floor, while large task evidence keeps recovery and review anchors.
pub(in crate::ai) const MID_TURN_SPILL_MIN_CHARS: usize = 6_000;

/// Cap inherited folded evidence before hard-target checks, accounting for any
/// archive pointer inserted by trimming.
fn cap_mid_turn_llm_evidence(messages: Vec<Message>, overflow_dir: &Path) -> (Vec<Message>, usize) {
    let messages = trim_compressed_tool_evidence_to_inline_budget(messages, Some(overflow_dir));
    let after = messages_total_chars(&messages);
    (messages, after)
}

/// Summary execution is independent of candidate insertion and net savings.
/// A failed model response or rejected archive plan still consumes a call.
pub(in crate::ai) struct MidTurnLlmSummaryOutcome {
    pub(in crate::ai) messages: Vec<Message>,
    pub(in crate::ai) before_chars: usize,
    pub(in crate::ai) after_chars: usize,
    pub(in crate::ai) effective: bool,
    pub(in crate::ai) summary_inserted: bool,
    pub(in crate::ai) model_called: bool,
}

struct ObservedSummarizer<'a, S: ?Sized> {
    inner: &'a S,
    model_called: AtomicBool,
}

impl<S: HistoryMessageSummarizer + ?Sized> HistoryMessageSummarizer for ObservedSummarizer<'_, S> {
    fn summarize_history_messages<'a>(
        &'a self,
        messages: &'a [Message],
        max_chars: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send + 'a>> {
        // Incremental planning has already checked for fresh, nonempty input.
        // Observe the port before awaiting it, regardless of its eventual result.
        self.model_called.store(true, Ordering::Relaxed);
        self.inner.summarize_history_messages(messages, max_chars)
    }
}

fn finalize_mid_turn_llm_summary(
    messages: Vec<Message>,
    before: usize,
    llm_summary_inserted: bool,
    model_called: bool,
) -> MidTurnLlmSummaryOutcome {
    let after = messages_total_chars(&messages);
    let was_effective = before.saturating_sub(after) >= MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS;
    MidTurnLlmSummaryOutcome {
        messages,
        before_chars: before,
        after_chars: after,
        effective: was_effective,
        summary_inserted: llm_summary_inserted,
        model_called,
    }
}

/// Retain the projection-only interface for persisted history callers. The
/// turn driver uses the explicit execution outcome below to account for calls.
pub(in crate::ai) async fn mid_turn_llm_summarize(
    app: &App,
    messages: Vec<Message>,
    keep_recent_turns: usize,
    summary_max_chars: usize,
    hard_target: usize,
    cwd: Option<&Path>,
) -> (Vec<Message>, usize, usize, bool, bool) {
    let outcome = mid_turn_llm_summarize_with_summarizer(
        app,
        app,
        messages,
        keep_recent_turns,
        summary_max_chars,
        hard_target,
        cwd,
    )
    .await;
    (
        outcome.messages,
        outcome.before_chars,
        outcome.after_chars,
        outcome.effective,
        outcome.summary_inserted,
    )
}

/// Try a source-bound summary of older turns, then progressively fold tool groups.
/// If still over budget, spill protected results before emergency field caps.
/// Protected user/system text can leave the target unmet.
/// `effective`, `summary_inserted`, and `model_called` are independent:
/// a called summarizer can yield a rejected candidate.
pub(in crate::ai) async fn mid_turn_llm_summarize_with_summarizer(
    app: &App,
    summarizer: &(impl HistoryMessageSummarizer + Sync + ?Sized),
    messages: Vec<Message>,
    keep_recent_turns: usize,
    summary_max_chars: usize,
    hard_target: usize,
    cwd: Option<&Path>,
) -> MidTurnLlmSummaryOutcome {
    let observed = ObservedSummarizer {
        inner: summarizer,
        model_called: AtomicBool::new(false),
    };
    let before = messages_total_chars(&messages);
    let overflow_dir = crate::ai::history::SessionStore::new(app.config.history_file.as_path())
        .session_assets_dir(&app.session_id);
    let protected_tool_call_ids = current_turn_precision_tool_call_ids(&messages);
    let lossless_tool_call_ids = current_turn_lossless_tool_call_ids(&messages);
    // Cap inherited evidence before any hard-target decision. Trimming can add an
    // archive pointer, so Path C must see and budget that pointer rather than add it
    // after the final hard-cap pass.
    let (messages, initial_after) = cap_mid_turn_llm_evidence(messages, overflow_dir.as_path());
    // best tracks the smallest result after the capped baseline; None means that
    // baseline is still in use.
    let mut best: Option<Vec<Message>> = None;
    let mut best_after = initial_after;
    // Whether Path A actually ran and injected a summary increment (see the return
    // doc).
    let mut llm_summary_inserted = false;
    if best_after <= hard_target {
        return finalize_mid_turn_llm_summary(messages, before, llm_summary_inserted, false);
    }

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
        if let Some(first_user) = messages
            .iter()
            .position(|m| m.role == "user" && !is_runtime_synthetic_user_message(m))
        {
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
        let dialogue = RecentDialogue::new(&messages, hard_target);
        // Extract context checkpoint markers from the to-be-summarized span: they
        // are the only index locating saved checkpoint bodies and must never be
        // swallowed by the summary. The regular persisted-compaction path already
        // does the same; this closes the gap here.
        let checkpoint_markers: Vec<Message> = earlier
            .iter()
            .filter(|m| is_context_checkpoint_marker(m))
            .cloned()
            .collect();
        // The summary source is exactly the span removed from the request.
        // Verbatim dialogue, side notes, prior summaries/pointers, and checkpoint
        // locators remain inline and must not be summarized or source-bound again.
        let mut retained_user_outside_source =
            messages[split_at..].iter().any(is_user_input_message);
        let mut summary_source = Vec::new();
        for (offset, message) in earlier.iter().enumerate() {
            let protected = dialogue.contains(preserved_system_end + offset);
            retained_user_outside_source |= protected && is_user_input_message(message);
            let retained_inline = protected
                || is_summary_or_archive_note(message)
                || is_side_note_message(message)
                || is_context_checkpoint_marker(message);
            if !retained_inline {
                summary_source.push(message.clone());
            }
        }
        let has_compactable = summary_source
            .iter()
            .any(|message| !is_system_like_role(&message.role));
        if has_compactable {
            if let Some(summary_plan) =
                incremental::plan_incremental_summary_with_prior_summarizer_and_retained_user_boundary(
                    &observed,
                    &summary_source,
                    summary_max_chars,
                    Some(overflow_dir.as_path()),
                    &messages[..preserved_system_end],
                    retained_user_outside_source,
                )
                .await
            {
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
                // 2. Preserve previous summaries, live guidance, and exact
                // dialogue in order. Insert the derived increment before the
                // retained dialogue and keep checkpoint locators recoverable.
                let mut increment_at = None;
                for (offset, message) in earlier.iter().enumerate() {
                    let protected = dialogue.contains(preserved_system_end + offset);
                    if protected {
                        increment_at.get_or_insert(out.len());
                    }
                    if protected
                        || is_summary_or_archive_note(message)
                        || is_side_note_message(message)
                    {
                        out.push(message.clone());
                    }
                }
                out.insert(increment_at.unwrap_or(out.len()), summary_plan.message().clone());
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
                // Commit only after the candidate beats the current best. Fold
                // and source writes precede the overflow append so a failed
                // earlier commit cannot append an outgoing span. If a later
                // write fails, keep `earlier` in context; committed files may remain.
                if after < best_after
                    && tail_plan.commit()
                    && summary_plan.commit()
                    && archive_messages_to_overflow(
                        &summary_source,
                        Some(overflow_dir.as_path()),
                    )
                    .is_some()
                {
                    let (out, after) = cap_mid_turn_llm_evidence(out, overflow_dir.as_path());
                    best = Some(out);
                    best_after = after;
                    llm_summary_inserted = true;
                }
                // Effective compaction meeting the target -> return directly
                if before.saturating_sub(best_after) >= MIN_EFFECTIVE_LLM_SUMMARY_SAVINGS
                    && best_after <= hard_target
                {
                    return finalize_mid_turn_llm_summary(best.unwrap(), before, true, true);
                }
            }
        }
    }

    // === Path B+C: progressive tool-group folding ===
    // Use progressive_fold_windows to narrow protection without reaching zero,
    // stopping once the context fits hard_target. This handles cases where "all
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
            let (folded, after) = cap_mid_turn_llm_evidence(folded, overflow_dir.as_path());
            best = Some(folded);
            best_after = after;
        }
    }

    // Only truly reaching hard_target allows an early return. The old logic
    // returned as soon as the net decrease exceeded 4K, skipping the hard backstop
    // below and letting "older groups already saved a lot, but the newest group
    // alone still overflows the window" keep sending over-limit requests.
    if best_after <= hard_target {
        return finalize_mid_turn_llm_summary(
            best.unwrap_or(messages),
            before,
            llm_summary_inserted,
            observed.model_called.load(Ordering::Relaxed),
        );
    }

    // === Path C backstop: budget-aware structure-preserving truncation ===
    // Keep system/user messages and assistant↔tool pairing while shortening
    // mutable fields against the total budget. When system/user content alone
    // is too large, return the smallest achievable projection.
    let mut result = best.unwrap_or(messages);
    // Spill eligible current-turn results before the emergency cap so successful
    // archives leave recoverable stubs. Failed or ineligible spills retain the
    // original fields for the hard-budget fallback.
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
    finalize_mid_turn_llm_summary(
        result,
        before,
        llm_summary_inserted,
        observed.model_called.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence_note(label: &str) -> Message {
        Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: serde_json::Value::String(format!(
                "compressed_tool_round: 1 tool calls\n{COMPRESSED_TOOL_EVIDENCE_MARKER}\n{label}:{}",
                "e".repeat(7_000)
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn llm_summary_evidence_cap_preserves_notes_when_archive_commit_fails() {
        let root = std::env::temp_dir().join(format!(
            "ai-mid-turn-llm-evidence-failure-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let not_a_directory = root.join("not-a-directory");
        std::fs::write(&not_a_directory, "file blocks archive directory creation").unwrap();
        let messages = vec![evidence_note("older"), evidence_note("newer")];
        let before = messages_total_chars(&messages);

        let (capped, after) = cap_mid_turn_llm_evidence(messages.clone(), &not_a_directory);

        let _ = std::fs::remove_dir_all(root);
        assert_eq!(capped, messages);
        assert_eq!(after, before);
    }
}
