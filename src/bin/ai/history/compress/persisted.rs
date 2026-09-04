//! Persisted-history pipeline: request-time context compaction,
//! sanitization for persisted history, reasoning-replay state, and
//! summary/archive-note coalescing.

use super::*;

/// Physical cap for a single raw tool result in the request context. Canonical
/// history is unaffected.
pub(in crate::ai) const TOOL_RESULT_RAW_HARD_CAP_CHARS: usize = 64_000;

pub(in crate::ai) fn cap_raw_tool_results_for_context(
    messages: &mut [Message],
    overflow_dir: Option<&Path>,
    cwd: Option<&Path>,
) -> usize {
    cap_oversized_tool_results_for_context(
        messages,
        TOOL_RESULT_RAW_HARD_CAP_CHARS,
        overflow_dir,
        cwd,
    )
}

/// Prefixes of all "auto-compaction summary" notes. The writer side (which
/// generates summary notes) and the recognizer side (duplicate guard, sqlite
/// resume points, request-side grouping) **must share this one list**; otherwise
/// the two sides split apart — "prefixes written are not recognized". Historically
/// `长期记忆摘要（压缩保留）` was never registered and thus bypassed the
/// duplicate guard, causing a summary note to be re-inserted every turn, the
/// context budget to creep up continuously, and the compaction pipeline to spin
/// on every turn. When adding a new summary prefix, change only this list.
///
/// Note: entries must be bare prefixes "after leading whitespace is stripped";
/// detection uniformly goes through [`is_summary_note_text`], which `trim_start`s
/// first and then checks `starts_with` one by one, so full-width/half-width
/// colons only need to be listed once each.
pub(in crate::ai) const SUMMARY_NOTE_PREFIXES: &[&str] = &[
    "对话摘要（自动压缩",
    "历史摘要（自动压缩",
    "长期记忆摘要（压缩保留）",
    "[mid-turn-summary]",
];

/// Marker of the deterministic evidence note generated when folding a tool group.
///
/// This is not an LLM-generated summary; it is evidence/checkpoint extracted
/// mechanically by the compressor from tool_call arguments and tool results. It
/// must survive secondary summarization; otherwise long tool chains degrade into
/// a tool bill of bare file_path / original_file_path entries, and the model
/// tends to re-gather evidence after compaction.
pub(in crate::ai) const COMPRESSED_TOOL_EVIDENCE_MARKER: &str = "[compressed-tool-evidence]";

/// Prefix of archive-pointer notes (back-references to overflow originals). They
/// appear paired with summary notes; the P1 folding logic relies on this to
/// recognize and dedupe piled-up archive pointers.
pub(in crate::ai) const ARCHIVE_NOTE_PREFIX: &str = "长期记忆归档";

/// Whether a piece of text is the body of an "auto-compaction summary" note
/// (prefix match, tolerant of leading whitespace). This is the **single source of
/// truth** for summary detection, shared by the guard / sqlite / request
/// normalization paths.
pub(in crate::ai) fn is_summary_note_text(text: &str) -> bool {
    let trimmed = text.trim_start();
    SUMMARY_NOTE_PREFIXES
        .iter()
        .any(|prefix| trimmed.starts_with(prefix))
}

/// Whether a piece of text is an overflow archive-pointer note.
pub(in crate::ai) fn is_archive_note_text(text: &str) -> bool {
    text.trim_start().starts_with(ARCHIVE_NOTE_PREFIX)
}

pub(in crate::ai) const PERSISTED_HISTORY_KEEP_RECENT_TURNS: usize = 160;

/// Dynamic bounds of the recent user-turn tail window protected during compaction
/// fallback (first_trim_candidate). Small contexts prefer keeping 3 turns to
/// improve multi-stage task continuity; very large contexts fall back to 2 turns
/// to control the budget.
pub(in crate::ai) const KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MIN: usize = 2;

pub(in crate::ai) const KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MAX: usize = 3;

/// Number of "recent user-turn" tail-window turns to fully exempt from
/// trimming/spilling/folding.
///
/// The base rule (pick one by total size) is unchanged: <=48K -> 3 turns,
/// otherwise 2 — zero behavior change for normal sessions.
///
/// **Byte-cap escape valve** (active when `budget > 0`): the protected tail window
/// is a "full exemption zone" and must not itself exceed the entire history
/// budget. Tool-heavy agentic sessions (few user turns x hundreds of tool calls
/// per turn) can balloon the tail window to MB scale and **structurally prevent
/// convergence** — even hundreds of tool groups inside the window are all
/// exempted. In that case shrink the protected turn count step by step, exposing
/// tool groups "from the second-to-last turn and earlier" to the fold/spill paths
/// to restore convergence. **Floor invariant: never below 1 turn** — the newest
/// user turn and its tool groups are always kept verbatim (the group-level
/// protection of `KEEP_RECENT_TOOL_GROUPS` remains the backstop).
///
/// `budget == 0` means the caller explicitly sets no cap (old behavior kept), for
/// reuse in contexts without a budget.
pub(in crate::ai) fn keep_recent_user_turns_when_trimming(
    messages: &[Message],
    budget: usize,
) -> usize {
    let mut keep = if messages_total_chars(messages) <= KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS {
        KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MAX
    } else {
        KEEP_RECENT_USER_TURNS_WHEN_TRIMMING_MIN
    };
    if budget == 0 {
        return keep;
    }
    while keep > 1 {
        let tail_start = retained_turn_start(messages, keep);
        if messages_total_chars(&messages[tail_start..]) <= budget {
            break;
        }
        keep -= 1;
    }
    keep
}

/// Batch trimming cannot recompute protection boundaries mid-execution, so a
/// low-budget target must adopt the sub-48K three-turn protection policy from the
/// start; otherwise the third-most-recent user turn may already have been deleted
/// before the total crosses 48K.
pub(in crate::ai) fn keep_recent_user_turns_for_batch(
    messages: &[Message],
    budget: usize,
) -> usize {
    let total_chars = messages_total_chars(messages);
    let mut keep = if budget > 0 && budget <= KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS {
        3
    } else if total_chars <= KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS {
        3
    } else {
        2
    };
    if budget > 0 {
        while keep > 1 {
            let tail_start = retained_turn_start(messages, keep);
            if messages_total_chars(&messages[tail_start..]) <= budget {
                break;
            }
            keep -= 1;
        }
    }
    keep
}

/// Constant accessors exposed to the rest of the crate, avoiding duplicated
/// threshold numbers in mod.rs.
pub(in crate::ai) fn persisted_history_keep_recent_turns() -> usize {
    PERSISTED_HISTORY_KEEP_RECENT_TURNS
}

/// Maximum number of self_note entries kept in the messages array. self_notes are
/// already persisted to MemoryStore (`memory_store::AgentMemoryEntry`); the copy
/// in messages is only the "redundant inline copy" the LLM saw within the same
/// turn. Over a long session with thousands of turns these inline copies bloat
/// monotonically and need sliding-window pruning.
pub(in crate::ai) const MAX_SELF_NOTES_IN_MESSAGES: usize = 8;

/// Total char cap for keeping mechanical evidence of older tool groups verbatim
/// in the model context. Older evidence is appended to overflow-history.md with
/// zero compression, and only a unified back-reference is kept in messages.
pub(in crate::ai) const MAX_COMPRESSED_TOOL_EVIDENCE_INLINE_CHARS: usize = 12_000;

pub(in crate::ai) const CONTEXT_CHECKPOINT_MARKER_PREFIX: &str = "[context_checkpoint";

pub(in crate::ai) const QUERY_MEMORY_INDEX_PREFIX: &str = "[query-memory-index-v1]";

pub(in crate::ai) fn compressed_tool_evidence_inline_chars_limit() -> usize {
    MAX_COMPRESSED_TOOL_EVIDENCE_INLINE_CHARS
}

/// Keep only the `self_note:` entries among the most recent `keep_recent`
/// internal_notes. Other internal_notes (cache hints, loop-breakers, history
/// summaries) are outside the pruning scope.
pub(in crate::ai) fn trim_self_notes_to_recent(
    messages: Vec<Message>,
    keep_recent: usize,
) -> Vec<Message> {
    let total_self_notes = messages.iter().filter(|m| is_self_note_message(m)).count();
    if total_self_notes <= keep_recent {
        return messages;
    }
    let drop_count = total_self_notes - keep_recent;
    let mut dropped = 0usize;
    messages
        .into_iter()
        .filter(|m| {
            if is_self_note_message(m) && dropped < drop_count {
                dropped += 1;
                false
            } else {
                true
            }
        })
        .collect()
}

pub(in crate::ai) fn is_self_note_message(m: &Message) -> bool {
    if m.role != ROLE_INTERNAL_NOTE {
        return false;
    }
    let s = value_to_string(&m.content);
    s.trim_start().starts_with("self_note:")
}

pub(in crate::ai) fn is_compressed_tool_evidence_note(m: &Message) -> bool {
    m.role == ROLE_INTERNAL_NOTE
        && value_to_string(&m.content)
            .trim_start()
            .contains(COMPRESSED_TOOL_EVIDENCE_MARKER)
}

pub(in crate::ai) const PERSISTED_HISTORY_SUMMARY_MAX_CHARS: usize = 8_000;

pub(in crate::ai) const INTERNAL_NOTE_OVERFLOW_DIR: &str = "internal-note-overflow";

pub(in crate::ai) const PRESERVED_TOOL_OVERFLOW_DIR: &str = "tool-overflow-compressed";

pub(in crate::ai) const PRESERVED_USER_OVERFLOW_DIR: &str = "user-overflow-preserved";

pub(in crate::ai) const PRESERVED_IMAGE_OVERFLOW_DIR: &str = "image-overflow-preserved";

pub(in crate::ai) const PRESERVED_CONTENT_STUB_PREFIX: &str = "[[PRESERVED_CONTENT_STUB_V1]]";

pub(in crate::ai) const USER_OVERFLOW_SPILL_MIN_CHARS: usize = 1_024;

pub(in crate::ai) const IMAGE_OVERFLOW_SPILL_MIN_CHARS: usize = 512;

pub(in crate::ai) fn compress_messages_for_context(
    mut messages: Vec<Message>,
    max_chars: usize,
    keep_last: usize,
    summary_max_chars: usize,
    overflow_dir: Option<PathBuf>,
    cwd: Option<&Path>,
) -> Vec<Message> {
    // The history store may still hold legacy JSON stubs. They are an internal
    // protocol of the compressor and must not be handed to the model as-is,
    // otherwise the model treats them as ordinary user text or even repeats them
    // verbatim in its final reply.
    normalize_preserved_message_stubs_for_model(&mut messages);
    if max_chars == 0 || messages.is_empty() {
        return messages;
    }

    // compressed_tool_round notes are themselves compaction products; without an
    // independent cap they accumulate one by one before the global history budget
    // triggers, forming another kind of linear context bloat.
    messages = trim_compressed_tool_evidence_to_inline_budget(messages, overflow_dir.as_deref());

    // Prune the self_note sliding cap before large-block compaction, so the
    // self_notes accumulated over thousands of turns (already written to
    // MemoryStore; the copy in messages is just a redundant backup) do not bloat
    // monotonically. MemoryStore still keeps every record.
    let messages = trim_self_notes_to_recent(messages, MAX_SELF_NOTES_IN_MESSAGES);

    // Converge duplicate summary/archive notes piled up by past duplicate-guard
    // breakage. Doing this at the request-time entry lets an old session that
    // already piled up dozens of note pairs recover on its very next request
    // instead of waiting for a flush.
    let messages = coalesce_accumulated_summary_notes(messages);

    let keep_last = keep_last.min(messages.len());
    if keep_last == 0 {
        return shrink_messages_to_fit_with_summary(
            messages,
            max_chars,
            summary_max_chars,
            overflow_dir.as_deref(),
            cwd,
            &rustc_hash::FxHashSet::default(),
        );
    }

    let split_at = retained_turn_start(&messages, keep_last);
    let (older, recent) = messages.split_at(split_at);
    if older.is_empty() {
        return shrink_messages_to_fit_with_summary(
            recent.to_vec(),
            max_chars,
            summary_max_chars,
            overflow_dir.as_deref(),
            cwd,
            &rustc_hash::FxHashSet::default(),
        );
    }

    let mut out = Vec::new();
    if summary_max_chars > 0 {
        let summary_source: Vec<Message> = older
            .iter()
            .filter(|message| !is_context_checkpoint_marker(message))
            .cloned()
            .collect();
        let summary = build_persisted_summary_text(&summary_source, summary_max_chars);
        if !summary.trim().is_empty() {
            out.push(Message {
                role: ROLE_INTERNAL_NOTE.to_string(),
                content: Value::String(format!(
                    "对话摘要（自动压缩，以下为早期对话要点）：\n{summary}"
                )),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }
    }
    out.extend(
        older
            .iter()
            .filter(|message| {
                // When the summary budget is 0 (e.g. second-round compaction in the
                // production path) no summary is rebuilt; the old summary/archive
                // notes are themselves "the compressed representation of the early
                // conversation" and must be kept like the checkpoint marker,
                // otherwise the summary prepare_turn already produced would be
                // silently dropped in the second compaction round.
                is_context_checkpoint_marker(message)
                    || (summary_max_chars == 0 && is_summary_or_archive_note(message))
            })
            .cloned(),
    );
    out.extend_from_slice(recent);
    shrink_messages_to_fit_with_summary(
        out,
        max_chars,
        summary_max_chars,
        overflow_dir.as_deref(),
        cwd,
        &rustc_hash::FxHashSet::default(),
    )
}

/// Char cap applied to "assistant narration carrying tool_calls" in the persisted
/// history.
///
/// The folder [`tool_groups::fold_tool_call_group_to_stub`] uses the visible
/// narration before this turn's tool calls as the source of
/// `assistant_checkpoint`. Except for continuation state the model protocol
/// explicitly requires replaying, full reasoning_content is never persisted and
/// must never be promoted into assistant body text; tool-call-only messages get
/// a safe operation summary rebuilt by the folder from structured tool_calls.
/// 720 chars is the same order of magnitude as the post-fold checkpoint cap.
pub(in crate::ai) const PERSISTED_TOOL_CALL_ASSISTANT_NARRATION_MAX_CHARS: usize = 720;

/// exact-replay continuation state exists only in the rebuildable context
/// projection. The payload carries the originating model, so switching models
/// never mistakes another provider's hidden state for the current model's resume
/// state.
pub(in crate::ai) fn encode_reasoning_replay_state(model: &str, reasoning: &str) -> String {
    format!(
        "{PERSISTED_REASONING_REPLAY_PREFIX}{}",
        serde_json::json!({ "model": model, "reasoning": reasoning })
    )
}

pub(in crate::ai) fn decode_reasoning_replay_for_model(
    model: &str,
    encoded: &str,
) -> Option<String> {
    let payload = encoded.strip_prefix(PERSISTED_REASONING_REPLAY_PREFIX)?;
    let payload: Value = serde_json::from_str(payload).ok()?;
    (payload.get("model")?.as_str()? == model)
        .then(|| payload.get("reasoning")?.as_str().map(str::to_owned))?
}

/// Replay prefix for encrypted reasoning under the Responses protocol. Kept
/// separate from exact-replay (`PERSISTED_REASONING_REPLAY_PREFIX`) because the
/// payload shapes differ: exact stores a plaintext reasoning string; encrypted
/// stores the reasoning output-item delivered by the provider (a JSON array with
/// `encrypted_content`). Separate prefixes prevent the request side from mixing
/// the two payload kinds and let the compaction/sanitize layers handle both with
/// the same "keep if marked" rule.
pub(in crate::ai) const PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX: &str =
    "\u{1e}aios:reasoning-encrypted-replay:v1\u{1f}";

/// Whether this reasoning payload is persisted replay continuation state that
/// must stay byte-for-byte intact: exact plaintext replay
/// (`PERSISTED_REASONING_REPLAY_PREFIX`) or encrypted Responses replay
/// (`PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX`). Compaction/truncation layers
/// use this to exclude replay state from trimming — truncating the JSON payload
/// while keeping the marker would break request-side decoding, and dropping it
/// would silently lose cross-turn continuation.
pub(in crate::ai) fn is_persisted_reasoning_replay(reasoning: &str) -> bool {
    reasoning.starts_with(PERSISTED_REASONING_REPLAY_PREFIX)
        || reasoning.starts_with(PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX)
}

/// Runtime master switch for cross-turn encrypted reasoning replay. On by
/// default; setting `AIOS_DISABLE_ENCRYPTED_REPLAY=1` short-circuits persistence
/// and request-side rebuild, for A/B experiments reproducing the "pre-fix"
/// behavior (encrypted reasoning lost across turns/resume). Experimental
/// scaffolding only; default product behavior is unchanged.
pub(in crate::ai) fn encrypted_reasoning_replay_runtime_enabled() -> bool {
    std::env::var("AIOS_DISABLE_ENCRYPTED_REPLAY")
        .map(|v| v.trim().is_empty() || v == "0")
        .unwrap_or(true)
}

/// Encode the encrypted reasoning items captured this turn, together with the
/// originating model, into a single string for persisting into
/// `reasoning_content`. Carries a model marker: when switching/falling back to
/// another model, request-side decoding drops it on model mismatch, avoiding
/// feeding model A's encrypted state to model B (the provider would 400).
pub(in crate::ai) fn encode_encrypted_reasoning_replay_state(
    model: &str,
    items: &[Value],
) -> String {
    format!(
        "{PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX}{}",
        serde_json::json!({ "model": model, "items": items })
    )
}

/// Decode encrypted reasoning items from the persisted `reasoning_content`.
/// Returns them only when the originating model inside the marker matches the
/// current request model; otherwise returns `None` (no cross-model replay).
pub(in crate::ai) fn decode_encrypted_reasoning_replay_for_model(
    model: &str,
    encoded: &str,
) -> Option<Vec<Value>> {
    let payload = encoded.strip_prefix(PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX)?;
    let payload: Value = serde_json::from_str(payload).ok()?;
    if payload.get("model")?.as_str()? != model {
        return None;
    }
    let mut items: Vec<Value> = payload.get("items")?.as_array()?.to_vec();
    // The gateway re-delivers the same reasoning resource twice — `.added`
    // (partial payload) and `.done` (full payload). The pre-fix accumulator
    // deduped by all-fields equality and failed to converge, so history may hold
    // two entries with the same id. Dedupe by id, keeping the later one (`.done`
    // is the protocol's final authoritative state); otherwise replay emits the
    // same resource id twice and modelhub returns 400 (-4003 Duplicate item found).
    dedup_reasoning_items_by_id(&mut items);
    Some(items)
}

/// Converge reasoning items by `id`: keep the later entry for the same resource.
///
/// The gateway re-delivers the same reasoning resource twice — `.added` (partial
/// payload) and `.done` (full payload); same id, different content, so
/// all-fields-equality dedup judges them unequal and leaves duplicate ids behind.
/// Replay then emits the same resource id twice and modelhub returns 400 (-4003
/// Duplicate item found). Converge by id here and keep the later entry: within a
/// stream `.done` always follows `.added` and is the protocol's final
/// authoritative state (carrying the full payload), so last-writer-wins naturally
/// picks it. Items without an `id` are never merged (keep all of them to avoid
/// wrong deletions).
pub(in crate::ai) fn dedup_reasoning_items_by_id(items: &mut Vec<Value>) {
    let mut deduped: Vec<Value> = Vec::with_capacity(items.len());
    for item in items.drain(..) {
        let id = item.get("id").cloned();
        match deduped
            .iter_mut()
            .find(|existing| id.is_some() && existing.get("id") == id.as_ref())
        {
            Some(existing) => *existing = item,
            None => deduped.push(item),
        }
    }
    *items = deduped;
}

pub(in crate::ai) fn sanitize_message_for_persisted_history_inner(
    message: &Message,
    replay_source_model: Option<&str>,
) -> Message {
    let mut sanitized = message.clone();
    if sanitized.role != "assistant" {
        return sanitized;
    }

    // The persisted history keeps only the assistant facts truly needed across
    // turns:
    // - `reasoning_content` is hidden reasoning; the persistence layer always
    //   drops it and never copies it into visible body text. When the provider
    //   needs the field shape, the request layer fills in an empty string.
    // - Assistant narration carrying tool_calls must not be emptied: otherwise
    //   the checkpoint of [`tool_groups::fold_tool_call_group_to_stub`] sees no
    //   text at all and collapses into
    //   "assistant_checkpoint: <empty; no persisted decision before these tool calls>"，
    //   leaving the model amnesic after compaction, re-gathering evidence from
    //   the same turn.
    //
    if sanitized
        .tool_calls
        .as_ref()
        .is_some_and(|tool_calls| !tool_calls.is_empty())
    {
        let narration = match &sanitized.content {
            Value::Null => String::new(),
            Value::String(text) => text.clone(),
            other => value_to_string(other),
        };
        let capped = truncate_to_chars(
            &narration,
            PERSISTED_TOOL_CALL_ASSISTANT_NARRATION_MAX_CHARS,
        );
        sanitized.content = Value::String(capped);
    }
    let has_tool_calls = sanitized
        .tool_calls
        .as_ref()
        .is_some_and(|tool_calls| !tool_calls.is_empty());
    if has_tool_calls {
        if let Some(reasoning) = sanitized.reasoning_content.as_mut() {
            if is_persisted_reasoning_replay(reasoning) {
                // Already a continuation state carrying the internal marker (exact
                // plaintext / responses encrypted); stay idempotent.
            } else if let Some(model) = replay_source_model {
                *reasoning = encode_reasoning_replay_state(model, reasoning);
            } else {
                sanitized.reasoning_content = None;
            }
        }
    } else {
        sanitized.reasoning_content = None;
    }
    sanitized
}

pub(in crate::ai) fn sanitize_message_for_persisted_history(message: &Message) -> Message {
    sanitize_message_for_persisted_history_inner(message, None)
}

/// Build the persisted projection according to the model protocol. Only models
/// that explicitly declare they need verbatim replay keep hidden reasoning for
/// tool-call assistant messages; final answers and other messages are still
/// always dropped.
pub(in crate::ai) fn sanitize_message_for_persisted_history_for_model(
    model: &str,
    message: &Message,
) -> Message {
    let replay_source_model =
        crate::ai::models::reasoning_content_replay_enabled(model).then_some(model);
    sanitize_message_for_persisted_history_inner(message, replay_source_model)
}

pub(in crate::ai) fn sanitize_persisted_history_messages(messages: Vec<Message>) -> Vec<Message> {
    let messages = coalesce_accumulated_summary_notes(messages);
    messages
        .into_iter()
        // Only reasoning carrying the internal marker is continuation state that
        // the runtime explicitly kept according to model capability; legacy
        // history, imported files, and bare reasoning from other models are still
        // dropped per the original policy.
        .map(|message| {
            let preserve = message
                .reasoning_content
                .as_deref()
                .is_some_and(|reasoning| is_persisted_reasoning_replay(reasoning));
            sanitize_message_for_persisted_history_inner(
                &message,
                preserve.then_some("already-tagged"),
            )
        })
        .collect()
}

/// Converge the multiple summary/archive notes piled up by past duplicate-guard
/// breakage.
///
/// Background: the `长期记忆摘要（压缩保留）` prefix was once not registered in
/// `is_summary_message`, so every compaction round re-inserted a "summary +
/// archive" note pair at the top; a long session could pile up dozens of pairs,
/// polluting the context budget and inflating `total_chars` until the compaction
/// pipeline spun on every turn.
///
/// Folding policy (lossless):
/// - **Summary notes**: dedupe and concatenate each note body (header stripped)
///   in original order into **one** note, put back where the first summary sat.
///   The "initial goal" each evicted round recorded is therefore fully kept.
/// - **Archive-pointer notes**: keep only one when contents are identical, keep
///   all when they differ, placed right after the merged summary — avoids losing
///   back-references to other archive files when importing/migrating sessions.
/// - All other messages are kept verbatim and in order (non-summary/archive
///   messages are never touched).
///
/// Fold only when there is more than one summary or identical archive pointers
/// exist, avoiding pointless rewriting of healthy history (when the return value
/// equals the input entry by entry, the caller's `compacted == messages` check
/// skips persisting).
pub(in crate::ai) fn coalesce_accumulated_summary_notes(messages: Vec<Message>) -> Vec<Message> {
    let summary_count = messages.iter().filter(|m| is_summary_message(m)).count();
    let mut seen_archive_texts = rustc_hash::FxHashSet::default();
    let has_duplicate_archive = messages
        .iter()
        .filter(|m| is_archive_note_message(m))
        .map(|m| value_to_string(&m.content))
        .any(|text| !seen_archive_texts.insert(text));
    if summary_count <= 1 && !has_duplicate_archive {
        return messages;
    }

    // Merge all summary bodies and dedupe archive pointers with identical
    // content; both keep their original order.
    let mut merged_bodies: Vec<String> = Vec::new();
    let mut first_summary_role: Option<String> = None;
    let mut archive_notes: Vec<Message> = Vec::new();
    let mut seen_archive_texts = rustc_hash::FxHashSet::default();
    for m in &messages {
        if is_summary_message(m) {
            if first_summary_role.is_none() {
                first_summary_role = Some(m.role.clone());
            }
            let text = value_to_string(&m.content);
            let body = automatic_summary_body(&text).unwrap_or_else(|| text.trim());
            let body = body.trim();
            if !body.is_empty() && !merged_bodies.iter().any(|b| b == body) {
                merged_bodies.push(body.to_string());
            }
        } else if is_archive_note_message(m) {
            let text = value_to_string(&m.content);
            if seen_archive_texts.insert(text) {
                archive_notes.push(m.clone());
            }
        }
    }

    let merged_summary = if merged_bodies.is_empty() {
        None
    } else {
        Some(Message {
            role: first_summary_role.unwrap_or_else(|| ROLE_INTERNAL_NOTE.to_string()),
            content: Value::String(format!(
                "长期记忆摘要（压缩保留）:\n{}",
                merged_bodies.join("\n")
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        })
    };

    // Rebuild the sequence: put the merged summary plus deduped archive pointers
    // at the position of "the first summary/archive note", drop the other
    // summary/archive notes, and keep every other message as-is.
    let mut out = Vec::with_capacity(messages.len());
    let mut inserted = false;
    for m in messages {
        if is_summary_or_archive_note(&m) {
            if !inserted {
                if let Some(summary) = merged_summary.clone() {
                    out.push(summary);
                }
                out.extend(archive_notes.iter().cloned());
                inserted = true;
            }
            // Remaining summaries and already-collected archive notes are dropped.
        } else {
            out.push(m);
        }
    }
    out
}

pub(in crate::ai) fn is_summary_or_archive_note(m: &Message) -> bool {
    is_summary_message(m) || is_archive_note_message(m)
}

pub(in crate::ai) fn is_archive_note_message(m: &Message) -> bool {
    is_system_like_role(&m.role) && is_archive_note_text(&value_to_string(&m.content))
}

pub(in crate::ai) fn compact_persisted_history(messages: Vec<Message>) -> Vec<Message> {
    let messages = sanitize_persisted_history_messages(messages);
    let user_turns = messages
        .iter()
        .filter(|message| {
            // Synthetic user messages (image followups etc.) do not form a real
            // turn boundary, avoiding premature history truncation.
            message.role == "user" && !is_runtime_synthetic_user_message(message)
        })
        .count();
    if user_turns <= MAX_HISTORY_TURNS {
        return messages;
    }

    let keep_recent_turns = PERSISTED_HISTORY_KEEP_RECENT_TURNS.min(MAX_HISTORY_TURNS - 1);
    let split_at = retained_turn_start(&messages, keep_recent_turns);
    if split_at == 0 || split_at >= messages.len() {
        return messages;
    }

    let checkpoint_markers: Vec<Message> = messages[..split_at]
        .iter()
        .filter(|message| is_context_checkpoint_marker(message))
        .cloned()
        .collect();
    let summary_source: Vec<Message> = messages[..split_at]
        .iter()
        .filter(|message| !is_context_checkpoint_marker(message))
        .cloned()
        .collect();
    let summary =
        build_persisted_summary_text(&summary_source, PERSISTED_HISTORY_SUMMARY_MAX_CHARS);
    let mut out = Vec::with_capacity(messages.len() - split_at + 1);
    if !summary.is_empty() {
        out.push(Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(format!(
                "历史摘要（自动压缩，以下为更早对话的简短语义）：\n{summary}"
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }
    out.extend(checkpoint_markers);
    out.extend_from_slice(&messages[split_at..]);
    out
}

pub(in crate::ai) async fn compact_persisted_history_with_app(
    app: &App,
    messages: Vec<Message>,
) -> Vec<Message> {
    compact_persisted_history_with_app_inner(app, messages, MAX_HISTORY_TURNS).await
}

/// Proactive compaction triggered by a task boundary (a turn ended with no
/// further tool calls, meaning the agent gave its final answer). The threshold is
/// lowered from `MAX_HISTORY_TURNS` (200) to `PERSISTED_HISTORY_KEEP_RECENT_TURNS`
/// (160), so the natural "task done" boundary triggers summarization earlier
/// instead of passively switching only when the hard cap is hit. Conversations
/// below 160 turns are still never summarized, so short sessions are unaffected.
pub(in crate::ai) async fn compact_persisted_history_at_boundary_with_app(
    app: &App,
    messages: Vec<Message>,
) -> Vec<Message> {
    compact_persisted_history_with_app_inner(app, messages, PERSISTED_HISTORY_KEEP_RECENT_TURNS)
        .await
}

async fn compact_persisted_history_with_app_inner(
    app: &App,
    messages: Vec<Message>,
    threshold_turns: usize,
) -> Vec<Message> {
    let messages = sanitize_persisted_history_messages(messages);
    let user_turns = messages
        .iter()
        .filter(|message| message.role == "user" && !is_runtime_synthetic_user_message(message))
        .count();
    if user_turns <= threshold_turns {
        return messages;
    }

    let keep_recent_turns = PERSISTED_HISTORY_KEEP_RECENT_TURNS.min(MAX_HISTORY_TURNS - 1);
    let split_at = retained_turn_start(&messages, keep_recent_turns);
    if split_at == 0 || split_at >= messages.len() {
        return messages;
    }

    let checkpoint_markers: Vec<Message> = messages[..split_at]
        .iter()
        .filter(|message| is_context_checkpoint_marker(message))
        .cloned()
        .collect();
    let summary_source: Vec<Message> = messages[..split_at]
        .iter()
        .filter(|message| !is_context_checkpoint_marker(message))
        .cloned()
        .collect();
    let summary = build_persisted_summary_text_with_app(
        app,
        &summary_source,
        PERSISTED_HISTORY_SUMMARY_MAX_CHARS,
    )
    .await;
    let mut out = Vec::with_capacity(messages.len() - split_at + 1);
    if !summary.is_empty() {
        out.push(Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(format!(
                "历史摘要（自动压缩，以下为更早对话的简短语义）：\n{summary}"
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }
    out.extend(checkpoint_markers);
    out.extend_from_slice(&messages[split_at..]);
    out
}
