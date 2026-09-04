//! History shrink ladder: tool-group folding, removable-message batching,
//! summary-driven shrinking, and leading-summary handling.

use super::*;

/// The next escalation when `first_tool_call_group` refuses to fold (all
/// remaining foldable groups contain non-compressible tools like `read_file`,
/// which it rejects by policy) but the budget is still exceeded: use
/// [`fold_early_tool_groups`] to progressively fold those groups "outside the
/// protected tail window" into single-line `compressed_tool_round` notes (each
/// carrying a file_path recall anchor the model can read back with read_file).
///
/// This reuses **the same** battle-tested folding function as Path B+C of
/// `mid_turn_llm_summarize`, just moved earlier into the regular/persisted
/// compaction path — fixing the root cause of "tool-heavy sessions (few user
/// turns x hundreds of read_file calls) never folding tool groups inside
/// `compress_messages_for_context` / `shrink_*`, leaving the whole history unable
/// to converge into the budget".
///
/// Returns whether an "effective fold" happened (net char decrease). `keep_recent`
/// tightens progressively from [`KEEP_RECENT_TOOL_GROUPS`] down to
/// [`MIN_KEEP_RECENT_TOOL_GROUPS`] (=1), keeping the most recent tool groups
/// verbatim as much as possible and widening the folding scope step by step only
/// while still over budget; every step must produce a net decrease to avoid
/// spinning without progress. **Never tightens to 0**: a window of 0 would fold
/// the most recent tool interaction into a stub too, leaving the model with no
/// structured tool context at all; the remaining excess is handled by the later
/// `first_trim_candidate` / `truncate_mutable_messages_to_fit` backstops in the
/// while loop.
pub(in crate::ai) fn fold_noncompressible_tool_groups_to_fit(
    messages: &mut Vec<Message>,
    max_chars: usize,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) -> bool {
    // The total is a pure function of the (not yet mutated) messages; compute
    // it once and reuse it for the entry guard and every comparison below.
    let base_total = messages_total_chars(messages);
    if base_total <= max_chars {
        return false;
    }
    // Select ONE window from the descending-protection ladder instead of
    // committing every intermediate rung:
    //   1. the most protective window whose result both fits and keeps the
    //      verbatim tail at or above MIN_PROTECTED_TAIL_CHARS,
    //   2. otherwise the most protective window that merely fits - overflow
    //      matters more than the floor once the floor cannot hold,
    //   3. otherwise the deepest reducing window, so bounded progress and the
    //      historical deep-fold endpoint are preserved even when nothing fits.
    // Plans are pure functions of the messages, so candidates are recorded as
    // window sizes and re-planned exactly once before committing.
    let mut floor_safe_fitting_keep: Option<usize> = None;
    let mut fitting_keep: Option<usize> = None;
    let mut deepest_reducing_keep: Option<usize> = None;
    // Anchor count is likewise invariant while this loop only reads messages;
    // planning (which deep-clones the entire history) is skipped outright for
    // windows whose plan would be `ToolGroupFoldPlan::unchanged`.
    let anchor_count = count_tool_group_anchors(messages);
    // Plan reuse: a plan is a pure function of the unchanged messages, so the
    // last reducing candidate can be committed directly when its window is the
    // chosen one instead of being discarded and re-planned afterwards. This
    // caps the ladder at one live plan (one whole-history clone) instead of a
    // clone per rung plus a final re-plan; every selection decision below is
    // evaluated exactly as before.
    let mut last_reducing_plan: Option<(usize, ToolGroupFoldPlan)> = None;
    for &keep_recent in progressive_fold_windows().iter() {
        if anchor_count <= keep_recent {
            continue;
        }
        let plan =
            plan_early_tool_groups(messages, keep_recent, overflow_dir, protected_tool_call_ids);
        if plan.folded_groups() == 0 {
            continue;
        }
        // A plan must net a strict decrease; drop it otherwise and keep tightening
        // keep_recent to guard against livelock where the group count changes but
        // the char count does not.
        let plan_total = messages_total_chars(plan.messages());
        if plan_total >= base_total {
            continue;
        }
        deepest_reducing_keep = Some(keep_recent);
        let plan_fits = plan_total <= max_chars;
        let floor_safe = plan_fits
            && protected_tail_message_chars(plan.messages(), keep_recent)
                >= MIN_PROTECTED_TAIL_CHARS;
        if plan_fits {
            if fitting_keep.is_none() {
                fitting_keep = Some(keep_recent);
            }
        }
        last_reducing_plan = Some((keep_recent, plan));
        if floor_safe {
            floor_safe_fitting_keep = Some(keep_recent);
            break;
        }
    }
    let chosen = floor_safe_fitting_keep
        .or(fitting_keep)
        .or(deepest_reducing_keep);
    let mut made_progress = false;
    if let Some(keep_recent) = chosen {
        // Reuse the remembered plan when it was built for the chosen window;
        // the only case it is not (a first fitting window superseded by later
        // reducing windows) falls back to the single re-plan, which produces a
        // byte-identical plan because the messages have not changed since the
        // selection loop ran.
        let plan = match last_reducing_plan.take() {
            Some((window, plan)) if window == keep_recent => plan,
            _ => {
                plan_early_tool_groups(messages, keep_recent, overflow_dir, protected_tool_call_ids)
            }
        };
        if plan.folded_groups() > 0
            && messages_total_chars(plan.messages()) < base_total
            && plan.commit()
        {
            let (folded, _) = plan.into_result();
            *messages = folded;
            made_progress = true;
        }
    }
    made_progress
}

/// Billable chars across the tool-result messages of the most-recent
/// `keep_recent` complete tool groups - i.e. the verbatim structured evidence
/// kept outside folding under that window size (assistant anchors excluded,
/// matching what recent_tool_group_message_indices returns).
pub(in crate::ai) fn protected_tail_message_chars(
    messages: &[Message],
    keep_recent: usize,
) -> usize {
    recent_tool_group_message_indices(messages, keep_recent)
        .into_iter()
        .map(|idx| message_billable_chars(&messages[idx]))
        .sum()
}

/// Batch-remove trimmable ordinary messages and archive them in a single flush.
/// The old implementation re-entered the outer loop and ran `sync_data` after
/// every single removal, so a tool-heavy history amplified hundreds of assistant
/// messages into hundreds of synchronous writes. Here the whole batch is trimmed
/// on a candidate copy first; if archiving fails the candidate is not adopted and
/// the original messages stay unchanged.
pub(in crate::ai) fn trim_removable_messages_batch(
    messages: &mut Vec<Message>,
    max_chars: usize,
    overflow_dir: Option<&Path>,
) -> bool {
    // Single-pass scan + rebuild, replacing the old "first_trim_candidate +
    // Vec::remove per round": the old loop re-scanned everything every round
    // (keep_recent_user_turns_when_trimming / retained_turn_start / the leading
    // protected run, each O(n)), and removal was an O(n) memmove — O(n²) overall,
    // visibly stalling histories with thousands of tool-heavy entries. The
    // protected tail window and total char count are computed once up front;
    // afterwards only an O(n) scan + O(n) rebuild run.
    let keep_recent_user_turns = keep_recent_user_turns_for_batch(messages, max_chars);
    let protected_tail_start = retained_turn_start(messages, keep_recent_user_turns);
    let mut total = messages_total_chars(messages);
    if total <= max_chars {
        return false;
    }

    let candidate = messages.clone();
    let mut removed = Vec::new();
    let mut kept = Vec::with_capacity(candidate.len());
    let mut index = 0usize;
    let mut in_protected_leading_run = true;
    for message in candidate {
        // Skip the whole leading protected system-like run (system prompt, history
        // summaries, archive pointers, checkpoints), matching first_trim_candidate
        // semantics.
        let head_protected =
            in_protected_leading_run && is_protected_leading_system_like_message(&message);
        if head_protected {
            kept.push(message);
            index += 1;
            continue;
        }
        in_protected_leading_run = false;

        // Same deletability rule as first_trim_candidate: checkpoints, spill
        // stubs, tool messages, and assistant(tool_calls) cannot be removed
        // singly. user messages are not removable on this path (OffloadOnly, spill
        // only) — skip rather than break, so the many trimmable candidates after
        // the first user message keep their chance of batch removal (the old
        // behavior broke out and left everything to the truncate backstop,
        // inconsistent with the with_summary "drop + archive" semantics). The
        // total char count is maintained exactly: subtract message_billable_chars
        // per removal and stop as soon as total <= max_chars, matching the old
        // loop's stop condition.
        let removable = index < protected_tail_start
            && !is_context_checkpoint_marker(&message)
            && !is_preserved_user_or_image_stub(&value_to_string(&message.content))
            && message.role != "tool"
            && !(message.role == "assistant"
                && message
                    .tool_calls
                    .as_ref()
                    .map(|c| !c.is_empty())
                    .unwrap_or(false))
            && message.role != "user";
        if removable && total > max_chars {
            total = total.saturating_sub(message_billable_chars(&message));
            removed.push(message);
        } else {
            kept.push(message);
        }
        index += 1;
    }
    if removed.is_empty() {
        return false;
    }
    // Ordinary messages still go to the unified history archive; internal_notes
    // are written to a deterministic file keyed by content fingerprint. The
    // latter both avoids silently losing recovery instructions/persisted state
    // and avoids appending the same body again on repeated compaction.
    let archive_candidates: Vec<Message> = removed
        .iter()
        .filter(|m| !is_internal_note_role(&m.role))
        .cloned()
        .collect();
    let internal_archive_dir = match archive_internal_notes_deduplicated(&removed, overflow_dir) {
        Ok(path) => path,
        Err(()) => return false,
    };
    let archive_ok = match overflow_dir {
        Some(dir) if !archive_candidates.is_empty() => {
            archive_messages_to_overflow(&archive_candidates, Some(dir)).is_some()
        }
        _ => true,
    };
    if !archive_ok {
        return false;
    }
    *messages = kept;
    insert_internal_note_archive_note_if_needed(messages, internal_archive_dir.as_deref());
    true
}

pub(in crate::ai) fn shrink_messages_to_fit(
    mut messages: Vec<Message>,
    max_chars: usize,
    overflow_dir: Option<&Path>,
    cwd: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) -> Vec<Message> {
    if max_chars == 0 {
        return messages;
    }

    if messages.is_empty() {
        return Vec::new();
    }

    redact_images_except_last(&mut messages, 1);
    dedup_adjacent(&mut messages);
    // dedup must run before offload: offload moves over-threshold old read_file
    // bodies to disk and replaces them with a stub carrying a **unique temp
    // path**; once that happens, byte-identical duplicates can no longer be folded
    // because their paths differ. Do content-level dedup first, folding redundant
    // bodies into back-reference stubs, then offload the few versions truly worth
    // keeping.
    dedup_repeated_tool_results(&mut messages, protected_tool_call_ids);
    prepare_tool_messages_structured(
        &mut messages,
        480,
        KEEP_RECENT_TOOL_GROUPS,
        overflow_dir,
        cwd,
        protected_tool_call_ids,
    );
    // Unconditionally spill oversized old user/image messages first (except the
    // protected tail window), consistent with
    // `shrink_messages_to_fit_with_summary`. Images are billed at nominal cost in
    // the budget, and once a large user body is moved to disk with zero
    // compression as a stub, the trimming loop below skips them automatically via
    // `is_preserved_user_or_image_stub` — preventing old user messages from being
    // outright `remove`d by generic trimming (which would violate the OffloadOnly
    // semantics assigned to RecentUser and silently lose the original text).
    if let Some(dir) = overflow_dir {
        spill_oversized_preserved_messages(&mut messages, dir, max_chars);
    }

    // Age-fold overflow stub preview bodies outside the protected tail window into
    // single-line anchors (file_path recall is not lost), converging the
    // historical bloat of "hundreds of early read_file previews accumulating
    // monotonically". This runs before the budget check so sessions not yet over
    // budget also keep converging already-spilled stubs. The tail-window turn
    // count is bounded by the max_chars byte cap: when a tool-heavy session's tail
    // window grows too large it shrinks automatically, exposing older stubs to age
    // folding.
    let keep_recent_turns = keep_recent_user_turns_when_trimming(&messages, max_chars);
    age_out_overflow_stub_previews(&mut messages, keep_recent_turns);
    // user/image spill stubs have no tool anchor to age-fold: their preview is
    // already a single-line pointer, and first_trim_candidate / truncate /
    // emergency cap never touch them again, so long sessions accumulate stubs
    // monotonically (especially image messages when the 512 threshold is below
    // their nominal cost). Merge old stubs outside the protected tail window into
    // one pointer carrying the archive directory, converging placeholder overhead
    // from O(N) to O(1).
    merge_old_user_overflow_stubs(&mut messages, keep_recent_turns);

    // Proactively slim down the giant arguments of write_file/apply_patch calls
    // that were "successfully written": once the file is on disk and the result
    // confirms success, the full body no longer has semantic value, so it can be
    // replaced with an archive stub without waiting for budget pressure. Anything
    // inside the protection window (including groups just written this turn,
    // whose bodies the model may immediately reference to build follow-up edits)
    // and failed results are always kept, so agent effectiveness does not degrade.
    shrink_successful_write_arguments(&mut messages, overflow_dir, protected_tool_call_ids);

    if messages_total_chars(&messages) <= max_chars {
        return messages;
    }

    while messages_total_chars(&messages) > max_chars {
        // Fold all unprotected tool groups over budget in one batch (both
        // compressible and non-compressible go through [`fold_early_tool_groups`]).
        // The old implementation folded only one group per iteration in the
        // `first_tool_call_group` + single-group fold loop, and only fell through
        // to the batch fold of `fold_noncompressible_tool_groups_to_fit` after
        // nothing foldable remained. Bug A kept the per-group savings tiny
        // (assistant.content had already been sanitized to `""`/`null`, so the
        // folded stub was nearly as large as the original group) -> the outer while
        // needed dozens of iterations to converge, each round also injecting an
        // `<empty>` empty-checkpoint note that polluted the context (see the 22
        // consecutive `compressed_tool_round` <empty> stubs in the e75fc2e5
        // session dump). Now each round first uses one `fold_early_tool_groups`
        // batch to collect every foldable group at once, finishing the shrink
        // within a few outer iterations.
        if fold_noncompressible_tool_groups_to_fit(
            &mut messages,
            max_chars,
            overflow_dir,
            protected_tool_call_ids,
        ) {
            continue;
        }
        if let Some(idx) = first_trim_candidate(&messages, max_chars) {
            // Old user messages (including multimodal ones with images) are never
            // silently deleted: that is the OffloadOnly semantics assigned to
            // RecentUser. First try moving the original text to the archive file
            // with zero compression and replacing it with a back-reference stub;
            // if the move succeeds, continue the trimming loop.
            if messages[idx].role == "user" {
                if let Some(dir) = overflow_dir
                    && try_spill_preserved_message_to_stub(&mut messages, dir, max_chars)
                {
                    continue;
                }
                // Cannot spill (no overflow_dir, or the body is too small, or the
                // proactive spill above already handled every over-threshold
                // user): break out of the trimming loop directly and never
                // `remove` the user original text. The residual slight overage is
                // left to the upper hard-threshold `mid_turn_llm_summarize`
                // backstop, avoiding a livelock where the same small user message
                // keeps getting picked.
                break;
            }
            // Archive the remaining trimmable candidates (plain assistant
            // narration, compressed_tool_round, etc.) in one place, avoiding
            // per-entry append + sync_data. If batch archiving fails, the original
            // messages stay unchanged.
            if trim_removable_messages_batch(&mut messages, max_chars, overflow_dir) {
                continue;
            }
            break;
        }
        break;
    }

    // When compressed_tool_evidence is trimmed, its body is appended with zero
    // compression to the unified history archive; the unified back-reference must
    // be put back into the request, otherwise the evidence exists on disk but the
    // model never learns the archive path. The archive note is an internal_note
    // (protected by is_system_like_role, so the truncation below will not cut it),
    // therefore it must be injected **before** `truncate_unprotected_messages_to_fit`,
    // so the final truncation frees the budget it occupies from other trimmable
    // messages, avoiding a payload that is slightly over max_chars. This matches
    // the order in `shrink_messages_to_fit_with_summary`: insert the summary note
    // first, then truncate.
    insert_overflow_archive_note_if_exists(&mut messages, overflow_dir);

    if messages_total_chars(&messages) > max_chars {
        truncate_unprotected_messages_to_fit(
            &mut messages,
            max_chars,
            overflow_dir,
            protected_tool_call_ids,
        );
    }

    keep_only_recent_reasoning_content(&mut messages);

    messages
}

/// Index of the nearest alive entry at or below `from` in the real-user
/// aliveness table, or -1 when none remains. Mirrors the "previous remaining
/// user" step of `retained_turn_start` under prefix-only deletions.
pub(in crate::ai) fn prev_alive_user(alive: &[bool], from: isize) -> isize {
    let mut cursor = from;
    while cursor >= 0 && !alive[cursor as usize] {
        cursor -= 1;
    }
    cursor
}

/// Single-pass replacement for the sequential per-drop rounds in
/// [`shrink_messages_to_fit_with_summary`]. The old path called
/// `first_trim_candidate` plus `Vec::remove` once per dropped message — an
/// O(n) rescan (including the 48K-threshold base and the byte-capped
/// tail-window recompute inside `keep_recent_user_turns_when_trimming`) and an
/// O(n) memmove per drop, i.e. O(n²) on long histories. This helper collects
/// every candidate those rounds would have dropped in one scan and splices
/// them out once. Selection is provably identical to the sequential loop:
///
/// - candidate predicate and protected leading run match `first_trim_candidate`
///   verbatim; the leading run can only grow (its members are never
///   candidates), and the scan re-extends it after each removal exactly like
///   the sequential recompute would;
/// - deletions always sit strictly before the tail-window boundary, so
///   comparing original indices against an originally-derived boundary is
///   equivalent to the sequential recompute on the shrinking sequence;
/// - `keep_recent_user_turns` has at most two values during the stretch
///   (totals only decrease, so the 48K base flips at most 2 -> 3); the
///   byte-cap component depends only on the protected tail sums, which are
///   invariant under prefix deletions, hence both variants are precomputed
///   from `keep_recent_user_turns_when_trimming`'s own formula;
/// - `fold_noncompressible_tool_groups_to_fit` cannot flip from false to true
///   mid-stretch: removing non-tool-group singletons leaves every fold plan's
///   folded-group set and net char delta unchanged, so one fold check per
///   outer round (kept at the call site) reproduces the interleaving;
/// - `dropped` / `dropped_internal_notes` keep the sequential (ascending)
///   removal order, `total` is decremented with the same
///   `saturating_sub(billable)` arithmetic per removal, and the rollback
///   snapshot is taken immediately before the first accepted removal.
pub(in crate::ai) fn drop_trim_candidates_batch(
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
    // Per-message billable chars cached once: the sequential loop re-charged
    // every message via a full `messages_total_chars` rescan each round.
    let chars: Vec<usize> = messages.iter().map(message_billable_chars).collect();
    // `keep_recent_user_turns_when_trimming` recomputed per round is equivalent
    // to this two-entry table: the byte-cap loop reads only the protected tail
    // sums (invariant under our deletions), and the 48K base depends only on
    // `total`, which moves downwards across the threshold at most once.
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
    // Real-user positions mirror `retained_turn_start`'s input list.
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
    // Ordinal (into user_positions) of the tail-window boundary user, or -1
    // once `retained_turn_start` would return 0 (protect everything). The
    // selected ordinal is alive_users - keep, walking down as users are
    // deleted or the window grows, exactly like recomputing
    // `retained_turn_start` per round.
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
        // Predicate chain copied from `first_trim_candidate`.
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

        // Accepted candidate: snapshot before the first removal (the caller's
        // sequential loop snapshotted at exactly this point), then tombstone.
        if messages_before_first_drop.is_none() {
            *messages_before_first_drop = Some(messages.clone());
        }
        tombstones[idx] = true;
        removed += 1;
        *total = total.saturating_sub(chars[idx]);
        // Deleting a real user below the window shifts the selected boundary
        // user one alive slot downwards, matching the sequential recompute.
        while user_ordinal < user_count && user_positions[user_ordinal] < idx {
            user_ordinal += 1;
        }
        if user_ordinal < user_count && user_positions[user_ordinal] == idx {
            alive[user_ordinal] = false;
            alive_users -= 1;
            user_ordinal += 1;
            if boundary_ptr >= 0 {
                boundary_ptr = prev_alive_user(&alive, boundary_ptr - 1);
            }
        }
        // Crossing the 48K threshold widens the protection window (base 2 ->
        // 3); totals never increase inside a stretch, so this fires at most
        // once per stretch.
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
            // Removing the first message after the protected leading run lets
            // that run grow over the following messages, exactly like the
            // sequential head-run recompute on the shrunk sequence.
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
    // One physical splice instead of one O(n) `Vec::remove` per drop. Each
    // list keeps ascending removal order, matching the sequential pushes.
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

/// Same as [`shrink_messages_to_fit`] but, before dropping early messages
/// outright, captures them into (or merges them with) a leading
/// `internal_note` summary so that long conversations still retain a
/// semantic memory of earlier user questions.
pub(in crate::ai) fn shrink_messages_to_fit_with_summary(
    mut messages: Vec<Message>,
    max_chars: usize,
    summary_max_chars: usize,
    overflow_dir: Option<&Path>,
    cwd: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) -> Vec<Message> {
    if max_chars == 0 {
        return messages;
    }
    if messages.is_empty() {
        return Vec::new();
    }

    redact_images_except_last(&mut messages, 1);
    dedup_adjacent(&mut messages);
    // dedup before offload: same rationale as shrink_messages_to_fit — avoid
    // byte-identical duplicate read_file bodies each being offloaded into a
    // unique-temp-path stub and losing the chance to fold.
    dedup_repeated_tool_results(&mut messages, protected_tool_call_ids);
    prepare_tool_messages_structured(
        &mut messages,
        480,
        KEEP_RECENT_TOOL_GROUPS,
        overflow_dir,
        cwd,
        protected_tool_call_ids,
    );
    enforce_protected_precision_group_budget(
        &mut messages,
        KEEP_RECENT_TOOL_GROUPS,
        max_chars / 2,
        overflow_dir,
        cwd,
        protected_tool_call_ids,
        false,
    );

    // Unconditionally spill oversized old user/image messages first (except the
    // newest turn's protected tail window). Images are billed at nominal cost in
    // the budget, so a single large image no longer triggers the over-budget loop;
    // they must therefore be moved to files with zero compression before the
    // budget check, avoiding a full base64 payload on every request.
    if let Some(dir) = overflow_dir {
        spill_oversized_preserved_messages(&mut messages, dir, max_chars);
    }

    // Age-fold overflow stub preview bodies outside the protected tail window into
    // single-line anchors (symmetric with shrink_messages_to_fit). Converges the
    // monotonic accumulation of early read_file previews; the file_path recall
    // anchor is kept. The tail-window turn count is likewise bounded by the
    // max_chars byte cap (see keep_recent_user_turns_when_trimming).
    let keep_recent_turns = keep_recent_user_turns_when_trimming(&messages, max_chars);
    age_out_overflow_stub_previews(&mut messages, keep_recent_turns);
    // Symmetric with plain shrink: merge user/image spill stubs outside the
    // protected tail window, preventing placeholder messages from accumulating
    // monotonically as the session grows.
    merge_old_user_overflow_stubs(&mut messages, keep_recent_turns);

    // Symmetric with shrink_messages_to_fit: proactively replace giant arguments
    // of successfully-written write_file/apply_patch calls with archive stubs
    // (the protection window and failed results are kept).
    shrink_successful_write_arguments(&mut messages, overflow_dir, protected_tool_call_ids);

    if messages_total_chars(&messages) <= max_chars {
        return messages;
    }
    let had_leading_summary = messages.first().map(is_summary_message).unwrap_or(false);
    // On archive failure the full pre-removal order must be restored; inserting
    // dropped messages at the head outright would place them before the retained
    // system prompt, breaking the message order the provider requires.
    let mut messages_before_first_drop: Option<Vec<Message>> = None;
    let mut dropped: Vec<Message> = Vec::new();
    let mut dropped_internal_notes: Vec<Message> = Vec::new();

    // Runtime char total: single removals subtract message_billable_chars exactly;
    // folds/spills are holistic batch changes recomputed uniformly in their own
    // branches — semantically identical to calling
    // `messages_total_chars(&messages)` every round, but avoids repeatedly
    // O(n)-rescanning the whole message sequence across loop iterations.
    let mut total = messages_total_chars(&messages);
    while total > max_chars {
        // Fold all unprotected tool groups over budget in one batch (both
        // compressible and non-compressible go through [`fold_early_tool_groups`])
        // — same rationale as [`shrink_messages_to_fit`], avoiding a single-group
        // fold loop that iterates dozens of rounds injecting `<empty>`
        // empty-checkpoint notes (see the e75fc2e5 session dump).
        if fold_noncompressible_tool_groups_to_fit(
            &mut messages,
            max_chars,
            overflow_dir,
            protected_tool_call_ids,
        ) {
            total = messages_total_chars(&messages);
            continue;
        }
        // Batch the consecutive drop rounds: selection inside
        // `drop_trim_candidates_batch` reproduces the previous per-round
        // `first_trim_candidate` + `Vec::remove` sequence exactly, while
        // turning O(n) rescans and O(n) memmoves per drop into one scan and
        // one splice (see the helper's doc comment for the equivalence proof).
        if drop_trim_candidates_batch(
            &mut messages,
            max_chars,
            &mut total,
            &mut messages_before_first_drop,
            &mut dropped,
            &mut dropped_internal_notes,
        ) > 0
        {
            continue;
        }
        if let Some(dir) = overflow_dir
            && try_spill_preserved_message_to_stub(&mut messages, dir, max_chars)
        {
            total = messages_total_chars(&messages);
            continue;
        }
        break;
    }

    let dropped_has_user_turn = dropped.iter().any(|m| m.role == "user");
    let has_leading_summary_now = messages.first().map(is_summary_message).unwrap_or(false);
    let internal_archive_dir =
        match archive_internal_notes_deduplicated(&dropped_internal_notes, overflow_dir) {
            Ok(path) => path,
            Err(()) => return messages_before_first_drop.unwrap_or(messages),
        };

    if !dropped.is_empty() {
        if let Some(dir) = overflow_dir {
            let mut sink = OverflowSink::new(dir);
            sink.push_messages(&dropped);

            if sink.flush() {
                let file_path_str = sink.file_path().to_string_lossy().to_string();
                let summary_body = if dropped_has_user_turn
                    && !has_leading_summary_now
                    && !had_leading_summary
                    && summary_max_chars > 0
                {
                    let header_chars = "对话摘要（自动压缩，以下为早期对话要点）：\n"
                        .chars()
                        .count();
                    let used = messages_total_chars(&messages);
                    // max_chars/used are char counts, so measure the header in
                    // chars too; a byte .len() would over-subtract (CJK is 3
                    // bytes/char). The /3 below stays as the deliberate
                    // byte-safety margin for the Chinese summary body.
                    let body_char_budget =
                        max_chars.saturating_sub(used).saturating_sub(header_chars);
                    let body_budget = (body_char_budget / 3).min(summary_max_chars);
                    if body_budget >= 40 {
                        let text = build_persisted_summary_text(&dropped, body_budget);
                        if !text.trim().is_empty() {
                            Some(text)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                let archive_note = build_overflow_placeholder(&file_path_str);
                let fallback_goal =
                    dropped
                        .iter()
                        .find(|message| message.role == "user")
                        .map(|message| {
                            summarize_text(
                                &normalize_whitespace(&value_to_string(&message.content)),
                                160,
                            )
                        });
                let memory_note = summary_body
                    .as_ref()
                    .filter(|s| !s.trim().is_empty())
                    .map(|summary| format!("长期记忆摘要（压缩保留）:\n{summary}"))
                    .or_else(|| {
                        fallback_goal
                            .as_ref()
                            .filter(|goal| !goal.trim().is_empty())
                            .map(|goal| format!("长期记忆摘要（压缩保留）:\n初始目标: {goal}"))
                    })
                    .unwrap_or_else(|| {
                        "长期记忆摘要（压缩保留）:\n较早原始对话已移出当前窗口；如果当前问题依赖前文细节，请读取归档文件。".to_string()
                    });

                if !has_leading_summary_now {
                    messages.insert(
                        0,
                        Message {
                            role: ROLE_INTERNAL_NOTE.to_string(),
                            content: Value::String(memory_note),
                            tool_calls: None,
                            tool_call_id: None,
                            reasoning_content: None,
                        },
                    );
                }
                insert_archive_note_if_missing(&mut messages, archive_note);
            } else {
                // flush failed: never delete history. Restore the full pre-removal
                // message snapshot and return immediately — skipping summary/archive
                // note injection (preventing dangling pointer notes without a
                // matching archive file), truncate, and reasoning cleanup. The
                // return value may still be over budget, but that is recoverable
                // (retry compaction next round / request-layer clamp), while data
                // loss is irreversible — honoring the existing lesson of "never
                // delete history when a write fails".
                return messages_before_first_drop.unwrap_or(messages);
            }
        } else if dropped_has_user_turn
            && !has_leading_summary_now
            && !had_leading_summary
            && summary_max_chars > 0
        {
            let header_prefix = "对话摘要（自动压缩，以下为早期对话要点）：\n";
            let header_chars = header_prefix.chars().count();
            let used = messages_total_chars(&messages);
            // Same char-unit accounting as the overflow-archive branch above:
            // header measured in chars (CJK is 3 bytes/char, byte .len() would
            // over-subtract from a char budget), /3 kept as body safety margin.
            let body_char_budget = max_chars.saturating_sub(used).saturating_sub(header_chars);
            let body_budget = (body_char_budget / 3).min(summary_max_chars);
            if body_budget >= 40 {
                let summary_text = build_persisted_summary_text(&dropped, body_budget);
                if !summary_text.trim().is_empty() {
                    let note = Message {
                        role: ROLE_INTERNAL_NOTE.to_string(),
                        content: Value::String(format!("{header_prefix}{summary_text}")),
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    };
                    messages.insert(0, note);
                }
            }
        }
    }

    insert_internal_note_archive_note_if_needed(&mut messages, internal_archive_dir.as_deref());

    if messages_total_chars(&messages) > max_chars {
        truncate_mutable_messages_to_fit(
            &mut messages,
            max_chars,
            overflow_dir,
            protected_tool_call_ids,
        );
    }

    keep_only_recent_reasoning_content(&mut messages);

    messages
}

#[allow(dead_code)]
pub(in crate::ai) fn take_leading_summary(messages: &mut Vec<Message>) -> Option<Message> {
    if messages.first().map(is_summary_message).unwrap_or(false) {
        Some(messages.remove(0))
    } else {
        None
    }
}
