//! Mutable-message truncation to fit budgets: per-field truncation,
//! overflow pointer building, and emergency caps.

use super::tool_overflow::{
    TOOL_RESULT_FOLD_MIN_CHARS, build_preserved_tool_overflow_stub_with_archive,
    task_evidence_review_lines,
};
use super::*;

fn mutable_content_can_truncate(
    message: &Message,
    content_chars: usize,
    allow_small_tool_results: bool,
) -> bool {
    allow_small_tool_results
        || message.role != "tool"
        || content_chars > TOOL_RESULT_FOLD_MIN_CHARS
        || message
            .content
            .as_str()
            .is_some_and(is_context_overflow_truncated_stub)
}

/// The last-resort hard-budget escape valve: keep the system/user and tool-call
/// pairing structure intact and only shorten rebuildable assistant/tool bodies,
/// reasoning, and oversized tool arguments. Current high-precision results are
/// protected first; if the target is still not met, truncating those results is
/// allowed. If the untrimmable system/user content itself is already over the
/// limit, return false.
/// Small-result floors apply to soft compaction only; the hard backstop must
/// also shrink accumulated small results to enforce the final budget.
pub(in crate::ai) fn truncate_mutable_messages_to_fit(
    messages: &mut Vec<Message>,
    max_chars: usize,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) -> bool {
    truncate_mutable_messages_to_fit_with_policy(
        messages,
        max_chars,
        overflow_dir,
        protected_tool_call_ids,
        true,
    )
}

/// Billing counters for one message, cached across candidate scans.
///
/// Byte lengths are only a hint: a char-shorter preview can keep the same byte
/// length because the truncation ellipsis is three bytes. A successful field
/// truncation therefore refreshes its entry explicitly; the byte check only
/// catches replacements made outside this pass. A value with no cheap byte
/// length is marked unknown and always recomputed.
struct MessageFieldCounts {
    entries: Vec<FieldCountEntry>,
}

struct FieldCountEntry {
    content_chars: usize,
    content_bytes: Option<usize>,
    content_is_dedup_stub: bool,
    content_is_preserved_overflow: bool,
    reasoning_chars: usize,
    reasoning_bytes: Option<usize>,
    argument_chars: Vec<usize>,
    argument_bytes: Vec<Option<usize>>,
}

/// Byte length of a value when it is a string; composite values have no cheap
/// length and therefore never match a cached entry.
fn value_byte_len(value: &Value) -> Option<usize> {
    match value {
        Value::String(text) => Some(text.len()),
        _ => None,
    }
}

/// Cross-checks a cached entry against a fresh count so a stale cache fails
/// loudly in tests instead of silently changing which field gets truncated.
#[cfg(all(test, debug_assertions))]
fn debug_assert_field_counts(entry: &FieldCountEntry, message: &Message) {
    let fresh = MessageFieldCounts::entry(message);
    assert_eq!(entry.content_chars, fresh.content_chars);
    assert_eq!(entry.content_is_dedup_stub, fresh.content_is_dedup_stub);
    assert_eq!(
        entry.content_is_preserved_overflow,
        fresh.content_is_preserved_overflow
    );
    assert_eq!(entry.reasoning_chars, fresh.reasoning_chars);
    assert_eq!(entry.argument_chars, fresh.argument_chars);
}

impl MessageFieldCounts {
    fn collect(messages: &[Message]) -> Self {
        Self {
            entries: messages.iter().map(Self::entry).collect(),
        }
    }

    fn entry(message: &Message) -> FieldCountEntry {
        let tool_calls = message.tool_calls.as_deref().unwrap_or_default();
        FieldCountEntry {
            content_chars: value_len_chars(&message.content),
            content_bytes: value_byte_len(&message.content),
            content_is_dedup_stub: super::dedup::is_dedup_tool_stub(&message.content),
            content_is_preserved_overflow: super::tool_overflow::is_preserved_tool_overflow_content(
                &message.content,
            ),
            reasoning_chars: message
                .reasoning_content
                .as_deref()
                .map(|reasoning| reasoning.chars().count())
                .unwrap_or_default(),
            reasoning_bytes: message.reasoning_content.as_deref().map(str::len),
            argument_chars: tool_calls
                .iter()
                .map(|call| call.function.arguments.chars().count())
                .collect(),
            argument_bytes: tool_calls
                .iter()
                .map(|call| Some(call.function.arguments.len()))
                .collect(),
        }
    }

    fn refresh(&mut self, index: usize, message: &Message) -> &FieldCountEntry {
        let stale = {
            let entry = &self.entries[index];
            let tool_calls = message.tool_calls.as_deref().unwrap_or_default();
            entry.content_bytes != value_byte_len(&message.content)
                || entry.reasoning_bytes != message.reasoning_content.as_deref().map(str::len)
                || entry.argument_chars.len() != tool_calls.len()
                || entry
                    .argument_bytes
                    .iter()
                    .zip(tool_calls)
                    .any(|(cached, call)| *cached != Some(call.function.arguments.len()))
        };
        if stale {
            self.entries[index] = Self::entry(message);
        }
        &self.entries[index]
    }

    fn insert(&mut self, index: usize, message: &Message) {
        self.entries.insert(index, Self::entry(message));
    }
}

/// Moves blocked-field entries past a freshly inserted archive note. The note
/// is inserted at index 0 or 1, so every later message shifts by one; a stale
/// entry would make the candidate scan skip the wrong field.
pub(super) fn shift_blocked_fields_after_note(
    blocked_fields: &mut rustc_hash::FxHashMap<(usize, MutableMessageField), bool>,
    inserted_at: usize,
) {
    *blocked_fields = std::mem::take(blocked_fields)
        .into_iter()
        .map(|((index, field), blocks_forever)| {
            (
                (
                    if index >= inserted_at {
                        index + 1
                    } else {
                        index
                    },
                    field,
                ),
                blocks_forever,
            )
        })
        .collect();
}

pub(in crate::ai) fn truncate_mutable_messages_to_fit_with_policy(
    messages: &mut Vec<Message>,
    max_chars: usize,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
    allow_protected_fallback: bool,
) -> bool {
    if max_chars == 0 || messages_total_chars(messages) <= max_chars {
        return true;
    }

    // The overflow asset is the single source of truth for protected evidence. When
    // the budget is short, drop the optional preview first and keep only the
    // parseable file_path pointer; the later generic head+tail truncation must
    // never touch that minimal protocol.
    minimize_overflow_stubs_for_hard_budget(messages);
    minimize_truncated_previews_for_hard_budget(messages);
    let mut total_chars = messages_total_chars(messages);
    if total_chars <= max_chars {
        return true;
    }

    let mut blocked_fields = rustc_hash::FxHashMap::default();
    // The overflow archive note is inserted at most once. Remember that it is
    // present so a long truncate loop does not rescan the whole history after
    // every successful field cut just to rediscover the same note.
    let mut overflow_archive_note_present = false;
    // A candidate scan can run once per edited field; recounting the whole
    // history each time makes a large session quadratic in its own bytes.
    let mut counts = MessageFieldCounts::collect(messages);
    for include_protected in [false, true] {
        if include_protected && !allow_protected_fallback {
            break;
        }
        while total_chars > max_chars {
            let excess = total_chars.saturating_sub(max_chars);
            let mut best: Option<(usize, MutableMessageField, usize)> = None;
            for (index, message) in messages.iter().enumerate() {
                if is_system_like_role(&message.role) || message.role == "user" {
                    continue;
                }
                let is_protected = protected_tool_context_message(message, protected_tool_call_ids);
                if is_protected && !include_protected {
                    continue;
                }
                let counts_entry = counts.refresh(index, message);
                #[cfg(all(test, debug_assertions))]
                debug_assert_field_counts(counts_entry, message);
                let content_chars = counts_entry.content_chars;
                if !message_contains_image(&message.content)
                    && !counts_entry.content_is_preserved_overflow
                    && !counts_entry.content_is_dedup_stub
                    && mutable_content_can_truncate(
                        message,
                        content_chars,
                        allow_protected_fallback,
                    )
                    && content_chars > 160
                    && !blocked_fields.contains_key(&(index, MutableMessageField::Content))
                {
                    choose_larger_mutable_field(
                        &mut best,
                        (index, MutableMessageField::Content, content_chars - 160),
                    );
                }
                if let Some(reasoning) = message.reasoning_content.as_deref()
                    && !is_persisted_reasoning_replay(reasoning)
                    && counts_entry.reasoning_chars > 160
                    && !blocked_fields.contains_key(&(index, MutableMessageField::Reasoning))
                {
                    choose_larger_mutable_field(
                        &mut best,
                        (
                            index,
                            MutableMessageField::Reasoning,
                            counts_entry.reasoning_chars - 160,
                        ),
                    );
                }
                if let Some(tool_calls) = &message.tool_calls {
                    for call_index in 0..tool_calls.len() {
                        let argument_chars = counts_entry
                            .argument_chars
                            .get(call_index)
                            .copied()
                            .unwrap_or_default();
                        let field = MutableMessageField::ToolArguments(call_index);
                        if argument_chars > 160 && !blocked_fields.contains_key(&(index, field)) {
                            choose_larger_mutable_field(
                                &mut best,
                                (index, field, argument_chars - 160),
                            );
                        }
                    }
                }
            }

            let Some((message_index, field, reducible)) = best else {
                break;
            };
            let reduce_by = excess.min(reducible).max(1);
            let message_chars_before = message_billable_chars(&messages[message_index]);
            let outcome = truncate_mutable_field_with_policy(
                &mut messages[message_index],
                field,
                reduce_by,
                overflow_dir,
                FieldArchivePolicy::BestEffort,
                allow_protected_fallback,
            );
            if outcome != FieldTruncateOutcome::Truncated {
                // A fixed marker / archive path may already be the smallest
                // form the field can reach. Skip the no-progress field and try
                // other candidates instead of re-selecting the same one.
                blocked_fields.insert((message_index, field), outcome.blocks_forever());
                continue;
            }
            // A two-char cut leaves `len()` unchanged: 4000 ASCII chars become
            // 3998 chars plus the three-byte ellipsis, still 4000 bytes. The
            // byte-only staleness check cannot see that edit, so refresh the
            // entry here instead of letting the next scan use the old count.
            counts.entries[message_index] = MessageFieldCounts::entry(&messages[message_index]);
            let archive_note_index = if !overflow_archive_note_present
                && overflow_dir.is_some_and(|dir| dir.join(OVERFLOW_HISTORY_FILENAME).is_file())
            {
                let len_before_archive_note = messages.len();
                insert_overflow_archive_note_if_exists(messages, overflow_dir);
                // Whether this call inserted the note or found an existing one,
                // the note is present from now on. The overflow path is stable
                // for the rest of the loop.
                overflow_archive_note_present = true;
                (messages.len() > len_before_archive_note).then_some(len_before_archive_note.min(1))
            } else {
                None
            };
            if let Some(note_index) = archive_note_index {
                shift_blocked_fields_after_note(&mut blocked_fields, note_index);
            }
            let shifted_message_index =
                if archive_note_index.is_some_and(|note_index| note_index <= message_index) {
                    message_index + 1
                } else {
                    message_index
                };
            // Only the edited message and a freshly inserted note can differ
            // from the fully minimized snapshot taken before the loop. Every
            // other message is already at its idempotent fixed point; running
            // the whole-history pass here made each field cut a full rescan.
            minimize_truncated_preview_for_hard_budget(&mut messages[shifted_message_index]);
            if let Some(note_index) = archive_note_index {
                minimize_truncated_preview_for_hard_budget(&mut messages[note_index]);
                counts.insert(note_index, &messages[note_index]);
            }
            let message_chars_after = message_billable_chars(&messages[shifted_message_index]);
            let previous_total = total_chars;
            total_chars -= message_chars_before;
            total_chars += message_chars_after;
            if let Some(note_index) = archive_note_index {
                total_chars += message_billable_chars(&messages[note_index]);
            }
            #[cfg(test)]
            debug_assert_eq!(total_chars, messages_total_chars(messages));
            if total_chars >= previous_total {
                // The first archive note can outweigh a minimal field cut. That is
                // note overhead, not field no-progress: only retire the field when
                // its own chars failed to shrink. Otherwise the next iteration can
                // cut it again now that the note is already present.
                if message_chars_after >= message_chars_before {
                    // No own shrink at this cut; retry after a later successful
                    // cut, matching the previous clear-on-success behavior.
                    blocked_fields.insert((shifted_message_index, field), false);
                }
            } else {
                // Inserting the archive note the first time may shift message
                // indices; re-evaluate candidates after a real shrink. Terminal
                // fields cannot shrink under any target, so they stay blocked.
                retain_terminal_blocked_fields(&mut blocked_fields);
            }
        }
    }

    total_chars <= max_chars
}

/// Soft compaction trims only unprotected fields; the current turn's
/// high-precision tool results must be left to the real hard-target backstop and
/// must not lose freshly-read precise context just because the soft threshold is
/// small.
pub(in crate::ai) fn truncate_unprotected_messages_to_fit(
    messages: &mut Vec<Message>,
    max_chars: usize,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) -> bool {
    truncate_mutable_messages_to_fit_with_policy(
        messages,
        max_chars,
        overflow_dir,
        protected_tool_call_ids,
        false,
    )
}

/// Path C first caps each trimmable field individually so one newest result cannot
/// monopolize the whole window, then keeps tightening against the total budget.
/// Neither step deletes messages or tool calls, nor rewrites exact-replay protocol
/// state, so assistant↔tool pairing and reasoning continuation state stay intact.
pub(in crate::ai) fn emergency_cap_messages_to_fit(
    messages: &mut Vec<Message>,
    max_chars: usize,
    per_field_cap: usize,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) -> bool {
    minimize_overflow_stubs_for_hard_budget(messages);
    minimize_truncated_previews_for_hard_budget(messages);
    let mut truncated_any = false;
    for message in messages.iter_mut() {
        if is_system_like_role(&message.role) || message.role == "user" {
            continue;
        }
        // Precision protection covers the grounding content (tool arguments and
        // their results), not the model's own reasoning trace. Skipping the
        // whole message let a protected assistant message keep an arbitrarily
        // large reasoning chain: no spill path owns reasoning text, so the
        // projection could never reach the hard target.
        let content_protected = protected_tool_context_message(message, protected_tool_call_ids);
        let content_chars = value_len_chars(&message.content);
        if !content_protected
            && !message_contains_image(&message.content)
            && !is_preserved_tool_overflow_content(&message.content)
            && !super::dedup::is_dedup_tool_stub(&message.content)
            && content_chars > per_field_cap
        {
            truncated_any |= truncate_mutable_field_with_policy(
                message,
                MutableMessageField::Content,
                content_chars - per_field_cap,
                overflow_dir,
                FieldArchivePolicy::BestEffort,
                true,
            )
            .truncated();
        }
        if let Some(reasoning_chars) = message
            .reasoning_content
            .as_deref()
            .filter(|reasoning| !is_persisted_reasoning_replay(reasoning))
            .map(|reasoning| reasoning.chars().count())
            && reasoning_chars > per_field_cap
        {
            truncated_any |= truncate_mutable_field(
                message,
                MutableMessageField::Reasoning,
                reasoning_chars - per_field_cap,
                overflow_dir,
                FieldArchivePolicy::BestEffort,
            );
        }
        let tool_call_count = message.tool_calls.as_ref().map(Vec::len).unwrap_or(0);
        for call_index in 0..tool_call_count {
            let call = message
                .tool_calls
                .as_ref()
                .and_then(|calls| calls.get(call_index));
            let argument_chars = call
                .map(|call| call.function.arguments.chars().count())
                .unwrap_or(0);
            // Precision protection is keyed by tool-call id and covers the
            // arguments that ground the request (patch bodies, command text),
            // not just the result message. Capping a protected call here would
            // replace it with an archive stub even when trimming reasoning had
            // already brought the request under budget.
            let arguments_protected =
                call.is_some_and(|call| protected_tool_call_ids.contains(&call.id));
            if !arguments_protected && argument_chars > per_field_cap {
                truncated_any |= truncate_mutable_field(
                    message,
                    MutableMessageField::ToolArguments(call_index),
                    argument_chars - per_field_cap,
                    overflow_dir,
                    FieldArchivePolicy::BestEffort,
                );
            }
        }
    }
    if truncated_any {
        insert_overflow_archive_note_if_exists(messages, overflow_dir);
    }
    let inner = truncate_mutable_messages_to_fit(
        messages,
        max_chars,
        overflow_dir,
        protected_tool_call_ids,
    );
    truncated_any || inner
}

/// Hard-budget de-preview: collapse already-truncated previews to minimal
/// archive pointers. The generic per-field loop sizes the pointer against a
/// computed `target` and refuses when the pointer is larger than that target,
/// even though swapping an 8K preview for a ~250-char pointer still saves
/// budget. Run this before that loop so Path C always reaches the smallest
/// recoverable form first; both replacements keep the archive path and stay
/// strictly smaller than the preview they replace.
fn minimize_truncated_previews_for_hard_budget(messages: &mut [Message]) {
    for message in messages.iter_mut() {
        minimize_truncated_preview_for_hard_budget(message);
    }
}

/// Apply the idempotent de-preview rules to one message. Callers that already
/// minimized the rest of the history can use this to update only the message
/// they just edited, instead of rescanning the whole history per field cut.
fn minimize_truncated_preview_for_hard_budget(message: &mut Message) {
    // Archive notes are system-like and therefore exempt from generic field
    // truncation. Keep only their stable recovery key at the hard backstop.
    if message.role == crate::ai::history::ROLE_INTERNAL_NOTE
        && let Value::String(text) = &message.content
        && let Some(file_path) = archive_note_file_path(text)
    {
        let shortened = build_minimal_archive_file_pointer(file_path);
        if shortened.chars().count() < text.chars().count() {
            message.content = Value::String(shortened);
        }
    }
    // Fold stubs are system-like and never enter the generic truncation loop,
    // but their two guidance lines are static boilerplate (~350 chars per
    // stub), not evidence: the full group content lives behind
    // archive_file_path. Shrink them to short budget notes so the hard
    // target stays reachable when several folded groups accumulate.
    if message.role == crate::ai::history::ROLE_INTERNAL_NOTE
        && let Value::String(text) = &message.content
        && text.contains("compressed_tool_round")
        && text.contains("context_budget_note:")
    {
        let shortened = text
                .replace(
                    "context_budget_note: earlier tool outputs were folded out of inline context because the request exceeded its tool/context budget; this does not mean the target file or command output is too large or broken.",
                    "context_budget_note: folded for budget; not a file/command limit.",
                )
                .replace(
                    "recovery_guidance: use the evidence, previews, and original_* anchors before repeating tools. For code edits, continue with smaller targeted reads and split patches instead of re-reading the same archive/log.",
                    "recovery_guidance: use evidence/archive before re-running; split large reads/patches.",
                );
        if shortened.chars().count() < text.chars().count() {
            message.content = Value::String(shortened);
        }
    }
    if is_system_like_role(&message.role) || message.role == "user" {
        return;
    }
    if let Value::String(text) = &message.content
        && is_context_overflow_truncated_stub(text)
        && let Some(pointer) = build_context_overflow_pointer(text, usize::MAX)
        && pointer.chars().count() < text.chars().count()
    {
        message.content = Value::String(pointer);
    }
    if let Some(calls) = message.tool_calls.as_mut() {
        for call in calls.iter_mut() {
            if is_context_overflow_truncated_tool_arguments(&call.function.arguments)
                && let Some(pointer) = build_context_overflow_tool_arguments_pointer(
                    &call.function.arguments,
                    usize::MAX,
                )
                && pointer.chars().count() < call.function.arguments.chars().count()
            {
                call.function.arguments = pointer;
            }
        }
    }
}

pub(in crate::ai) fn choose_larger_mutable_field(
    best: &mut Option<(usize, MutableMessageField, usize)>,
    candidate: (usize, MutableMessageField, usize),
) {
    if best
        .as_ref()
        .is_none_or(|(_, _, best_reducible)| candidate.2 > *best_reducible)
    {
        *best = Some(candidate);
    }
}

pub(in crate::ai) const CONTEXT_OVERFLOW_TRUNCATED_PREFIX: &str = "[context-overflow-truncated]";

pub(in crate::ai) const CONTEXT_OVERFLOW_UNARCHIVED_POINTER: &str = "[context-overflow-truncated] full original was not archived; inline preview omitted to meet context budget.";

/// Whether text already has the overflow-truncated marker.
pub(in crate::ai) fn is_context_overflow_truncated_stub(text: &str) -> bool {
    text.trim_start()
        .starts_with(CONTEXT_OVERFLOW_TRUNCATED_PREFIX)
}

/// Extract the `archived at: <path>` target embedded in an overflow stub, if
/// any. The canonical form puts the pointer on the first line; a legacy inline
/// form cuts the path at `;` or end of line.
pub(in crate::ai) fn embedded_archive_path(text: &str) -> Option<&str> {
    text.lines()
        .find_map(|line| line.split_once("archived at: ").map(|(_, p)| p.trim()))
        .or_else(|| {
            text.split_once("archived at: ")
                .map(|(_, rest)| rest.split([';', '\n']).next().unwrap_or(rest).trim())
        })
}

/// Fold an existing overflow-truncated stub into its minimal terminal state:
/// keep the archive path when one exists; when none exists, state plainly that
/// the full original is not readable back, never fabricate an archive pointer.
pub(in crate::ai) fn build_context_overflow_pointer(text: &str, target: usize) -> Option<String> {
    let path = embedded_archive_path(text);
    if let Some(path) = path {
        // Prefer the full pointer form when it fits the target.
        let full_pointer =
            format!("{CONTEXT_OVERFLOW_TRUNCATED_PREFIX} full original archived at: {path}\n");
        if full_pointer.chars().count() <= target {
            return Some(full_pointer);
        }
        // Otherwise keep only the single-line archive pointer.
        let minimal = format!("{CONTEXT_OVERFLOW_TRUNCATED_PREFIX} archived at: {path}");
        return (minimal.chars().count() < text.chars().count()).then_some(minimal);
    }
    (CONTEXT_OVERFLOW_UNARCHIVED_POINTER.chars().count() <= target
        && CONTEXT_OVERFLOW_UNARCHIVED_POINTER.chars().count() < text.chars().count())
    .then(|| CONTEXT_OVERFLOW_UNARCHIVED_POINTER.to_string())
}

/// Truncated tool arguments are terminal too. Re-archiving a metadata-only or
/// legacy preview stub would falsely claim the full arguments are recoverable.
pub(in crate::ai) fn is_context_overflow_truncated_tool_arguments(arguments: &str) -> bool {
    serde_json::from_str::<Value>(arguments)
        .ok()
        .is_some_and(|value| {
            value
                .get("_context_overflow_truncated")
                .and_then(Value::as_bool)
                == Some(true)
                && value.get("archive_file_path").is_some()
        })
}

pub(in crate::ai) fn build_context_overflow_tool_arguments_pointer(
    arguments: &str,
    target: usize,
) -> Option<String> {
    let value = serde_json::from_str::<Value>(arguments).ok()?;
    if value
        .get("_context_overflow_truncated")
        .and_then(Value::as_bool)
        != Some(true)
        || value.get("archive_file_path").is_none()
    {
        return None;
    }
    let archived_path = value
        .get("archive_file_path")
        .and_then(Value::as_str)
        .filter(|path| !path.is_empty());
    let mut pointer = match archived_path {
        Some(path) => serde_json::json!({
            "_context_overflow_truncated": true,
            "archive_file_path": path,
        }),
        None => serde_json::json!({
            "_context_overflow_truncated": true,
            "archive_file_path": Value::Null,
            "original_unavailable": true,
        }),
    };
    if let Some(original_chars) = value.get("original_chars") {
        pointer["original_chars"] = original_chars.clone();
    }
    let pointer = pointer.to_string();
    (pointer.chars().count() <= target && pointer.chars().count() < arguments.chars().count())
        .then_some(pointer)
}

/// Build metadata for an already-selected argument field within its target.
pub(super) fn build_context_overflow_tool_arguments_stub(
    original_chars: usize,
    archive_file_path: Option<&str>,
    target: usize,
) -> Option<String> {
    let pointer = serde_json::json!({
        "_context_overflow_truncated": true,
        "original_chars": original_chars,
        "archive_file_path": archive_file_path,
    })
    .to_string();
    let pointer_chars = pointer.chars().count();
    (pointer_chars <= target && pointer_chars < original_chars).then_some(pointer)
}

pub(in crate::ai) fn truncate_mutable_field(
    message: &mut Message,
    field: MutableMessageField,
    reduce_by: usize,
    overflow_dir: Option<&Path>,
    archive_policy: FieldArchivePolicy,
) -> bool {
    truncate_mutable_field_with_policy(
        message,
        field,
        reduce_by,
        overflow_dir,
        archive_policy,
        false,
    )
    .truncated()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FieldTruncateOutcome {
    Truncated,
    NoProgress,
    Retryable,
}

impl FieldTruncateOutcome {
    /// A target-dependent failure at the gentlest cut (`reduce_by == 1`) already
    /// used the largest allowed target; a later, gentler cut cannot succeed.
    /// Failures at more aggressive cuts may still succeed once the excess
    /// shrinks and the target grows.
    fn target_dependent(reduce_by: usize) -> Self {
        if reduce_by <= 1 {
            Self::NoProgress
        } else {
            Self::Retryable
        }
    }

    /// Archive writes can fail transiently, so keep them retryable while a sink
    /// exists. Without an overflow directory no retry can ever succeed.
    fn archive_dependent(overflow_dir: Option<&Path>) -> Self {
        if overflow_dir.is_some() {
            Self::Retryable
        } else {
            Self::NoProgress
        }
    }

    fn blocks_forever(self) -> bool {
        self == Self::NoProgress
    }

    fn truncated(self) -> bool {
        self == Self::Truncated
    }
}

/// Terminal failures stay blocked across later successful cuts; retryable
/// failures are released so a gentler target can be attempted.
pub(super) fn retain_terminal_blocked_fields(
    blocked_fields: &mut rustc_hash::FxHashMap<(usize, MutableMessageField), bool>,
) {
    blocked_fields.retain(|_, blocks_forever| *blocks_forever);
}

fn truncate_mutable_field_with_policy(
    message: &mut Message,
    field: MutableMessageField,
    reduce_by: usize,
    overflow_dir: Option<&Path>,
    archive_policy: FieldArchivePolicy,
    allow_small_tool_results: bool,
) -> FieldTruncateOutcome {
    // The archive path can be derived directly from overflow_dir (same source
    // as OverflowSink::new), so size the stub against that path first and do
    // the real archive write last: if the stub would not be strictly shorter
    // than the original, give up immediately instead of archiving first and
    // failing the size check afterwards (which would re-archive the same field
    // every compression round and grow the overflow file without bound).
    let archive_path_hint: Option<String> = overflow_dir.map(|dir| {
        dir.join(OVERFLOW_HISTORY_FILENAME)
            .to_string_lossy()
            .into_owned()
    });
    match field {
        MutableMessageField::Content => {
            if is_preserved_tool_overflow_content(&message.content)
                || !mutable_content_can_truncate(
                    message,
                    value_len_chars(&message.content),
                    allow_small_tool_results,
                )
            {
                return FieldTruncateOutcome::NoProgress;
            }
            let text = value_to_string(&message.content);
            let original_chars = text.chars().count();
            let target = original_chars.saturating_sub(reduce_by).max(160);
            let task_review_lines = if message.role == "tool" {
                task_evidence_review_lines(&text)
            } else {
                Vec::new()
            };
            if !task_review_lines.is_empty() && !is_context_overflow_truncated_stub(&text) {
                // Both archive outcomes use the same terminal stub protocol so
                // later passes cannot reassign reviews by folding a stub again.
                let build_stub = |path: Option<&str>| {
                    build_preserved_tool_overflow_stub_with_archive(
                        path.map(Path::new),
                        "unknown_tool",
                        &text,
                        &task_review_lines,
                    )
                };
                if build_stub(archive_path_hint.as_deref()).chars().count() >= original_chars {
                    return FieldTruncateOutcome::NoProgress;
                }
                let archived = archive_truncated_field_to_overflow(message, field, overflow_dir);
                if archived.is_none() && archive_policy == FieldArchivePolicy::Required {
                    return FieldTruncateOutcome::archive_dependent(overflow_dir);
                }
                message.content = Value::String(build_stub(archived.as_deref()));
                return FieldTruncateOutcome::Truncated;
            }
            // BestEffort fields may reuse an embedded archive path. Required
            // fields take the trusted re-archive path below instead.
            if is_context_overflow_truncated_stub(&text) {
                // A Required field is authoritative user input. Never trust an
                // `archived at:` path embedded in that user-controlled text:
                // it may name an unrelated existing file. First prove that the
                // trusted session sink can produce a shorter pointer, then
                // archive the entire current field and point only at the path
                // returned by that successful write.
                if archive_policy == FieldArchivePolicy::Required {
                    let pointer_for_path = |path: &str| {
                        let canonical = format!(
                            "{CONTEXT_OVERFLOW_TRUNCATED_PREFIX} full original archived at: {path}"
                        );
                        build_context_overflow_pointer(&canonical, target)
                            .filter(|pointer| pointer.chars().count() < original_chars)
                    };
                    let Some(path_hint) = archive_path_hint.as_deref() else {
                        return FieldTruncateOutcome::NoProgress;
                    };
                    if pointer_for_path(path_hint).is_none() {
                        return FieldTruncateOutcome::target_dependent(reduce_by);
                    }
                    let Some(archive_file_path) =
                        archive_truncated_field_to_overflow(message, field, overflow_dir)
                    else {
                        return FieldTruncateOutcome::archive_dependent(overflow_dir);
                    };
                    let Some(pointer) = pointer_for_path(&archive_file_path) else {
                        return FieldTruncateOutcome::target_dependent(reduce_by);
                    };
                    message.content = Value::String(pointer);
                    return FieldTruncateOutcome::Truncated;
                }
                if let Some(pointer) = build_context_overflow_pointer(&text, target) {
                    if pointer.chars().count() < original_chars {
                        message.content = Value::String(pointer);
                        return FieldTruncateOutcome::Truncated;
                    }
                }
                return FieldTruncateOutcome::target_dependent(reduce_by);
            }
            // When the preview budget is too small, the stub would be only a
            // long path with no actual content (a fake truncation): a small
            // result (e.g. a task_status poll) turned into an empty-preview
            // stub leaves the model unable to judge the real state and can
            // trap it in a "cannot confirm status → poll forever" loop. Keep
            // the original and let the hard budget bail out instead of
            // producing an information-free stub.
            const MIN_CONTENT_PREVIEW_CHARS: usize = 32;
            let build_truncated = |path: Option<&str>| -> Option<String> {
                let prefix = path
                    .map(|p| {
                        format!(
                            "[context-overflow-truncated] full original archived at: {p}\nhead+tail preview:\n"
                        )
                    })
                    .unwrap_or_else(|| {
                        "[context-overflow-truncated] head+tail preview:\n".to_string()
                    });
                let preview_budget = target.saturating_sub(prefix.chars().count());
                if preview_budget < MIN_CONTENT_PREVIEW_CHARS {
                    // The hard backstop may keep only the recovery pointer when
                    // a long archive path leaves no room for an inline preview.
                    if allow_small_tool_results {
                        return match path {
                            Some(path) => build_context_overflow_pointer(
                                &format!(
                                    "{CONTEXT_OVERFLOW_TRUNCATED_PREFIX} full original archived at: {path}"
                                ),
                                target,
                            ),
                            None => (CONTEXT_OVERFLOW_UNARCHIVED_POINTER.chars().count() <= target)
                                .then(|| CONTEXT_OVERFLOW_UNARCHIVED_POINTER.to_string()),
                        };
                    }
                    return None;
                }
                Some(format!(
                    "{prefix}{}",
                    keep_ends_by_chars(&text, preview_budget)
                ))
            };
            let Some(truncated) = build_truncated(archive_path_hint.as_deref())
                .filter(|candidate| candidate.chars().count() < original_chars)
            else {
                return FieldTruncateOutcome::target_dependent(reduce_by);
            };
            // Truncate only when the archived form keeps a meaningful preview;
            // only after the archive write actually failed may we degrade to
            // the no-path inline stub, so a good archive is never downgraded
            // to an unreadable preview.
            let archive_file_path =
                archive_truncated_field_to_overflow(message, field, overflow_dir);
            // Required policy (current user instruction): without an archived
            // copy the preview-only stub would be the last surviving version of
            // the instruction and could never be read back. Refuse and keep
            // the original so the caller can surface the error.
            if archive_file_path.is_none() && archive_policy == FieldArchivePolicy::Required {
                return FieldTruncateOutcome::archive_dependent(overflow_dir);
            }
            let truncated = build_truncated(archive_file_path.as_deref()).unwrap_or(truncated);
            message.content = Value::String(truncated);
            FieldTruncateOutcome::Truncated
        }
        MutableMessageField::Reasoning => {
            let Some(reasoning) = message.reasoning_content.as_deref() else {
                return FieldTruncateOutcome::NoProgress;
            };
            // Exact/encrypted replay payloads must remain byte-for-byte intact.
            // Keeping the marker while truncating its payload would break
            // decoding in the request layer, so callers must shrink another
            // field instead.
            if is_persisted_reasoning_replay(reasoning) {
                return FieldTruncateOutcome::NoProgress;
            }
            let original_chars = reasoning.chars().count();
            let target = original_chars.saturating_sub(reduce_by).max(160);
            // As in Content, reject a stub whose long archive path leaves no
            // meaningful preview; keep the original for the hard-budget path.
            const MIN_REASONING_PREVIEW_CHARS: usize = 32;
            let build_truncated = |path: Option<&str>| -> Option<String> {
                let prefix = path
                    .map(|p| {
                        format!("[context-overflow-truncated] full original archived at: {p}; ")
                    })
                    .unwrap_or_else(|| "[context-overflow-truncated] ".to_string());
                let preview_budget = target.saturating_sub(prefix.chars().count());
                if preview_budget < MIN_REASONING_PREVIEW_CHARS {
                    return None;
                }
                Some(format!(
                    "{prefix}{}",
                    keep_ends_by_chars(reasoning, preview_budget)
                ))
            };
            let Some(truncated) = build_truncated(archive_path_hint.as_deref())
                .filter(|candidate| candidate.chars().count() < original_chars)
            else {
                return FieldTruncateOutcome::target_dependent(reduce_by);
            };
            let archive_file_path =
                archive_truncated_field_to_overflow(message, field, overflow_dir);
            // Required policy: an unarchived stub must never replace the only
            // copy of the field; refuse so the caller keeps the original.
            if archive_file_path.is_none() && archive_policy == FieldArchivePolicy::Required {
                return FieldTruncateOutcome::archive_dependent(overflow_dir);
            }
            let truncated = build_truncated(archive_file_path.as_deref()).unwrap_or(truncated);
            message.reasoning_content = Some(truncated);
            FieldTruncateOutcome::Truncated
        }
        MutableMessageField::ToolArguments(call_index) => {
            let Some(arguments) = message
                .tool_calls
                .as_ref()
                .and_then(|calls| calls.get(call_index))
                .map(|call| call.function.arguments.clone())
            else {
                return FieldTruncateOutcome::NoProgress;
            };
            let original_chars = arguments.chars().count();
            let target = original_chars.saturating_sub(reduce_by).max(160);
            if is_context_overflow_truncated_tool_arguments(&arguments) {
                let Some(pointer) =
                    build_context_overflow_tool_arguments_pointer(&arguments, target)
                else {
                    return FieldTruncateOutcome::target_dependent(reduce_by);
                };
                let Some(call) = message
                    .tool_calls
                    .as_mut()
                    .and_then(|calls| calls.get_mut(call_index))
                else {
                    return FieldTruncateOutcome::NoProgress;
                };
                call.function.arguments = pointer;
                return FieldTruncateOutcome::Truncated;
            }
            // Preserve the established preview-based eligibility and size checks.
            // Preview bytes are used only for sizing; the published pointer below
            // contains recovery metadata instead of tool payload data.
            const MIN_TOOL_ARGS_PREVIEW_CHARS: usize = 8;
            let build_truncated = |path: Option<&str>, preview: String| {
                serde_json::json!({
                    "_context_overflow_truncated": true,
                    "original_chars": original_chars,
                    "archive_file_path": path,
                    "preview": preview,
                })
                .to_string()
            };
            let build_candidate = |path: Option<&str>| -> Option<String> {
                let fixed_chars = build_truncated(path, String::new()).chars().count();
                let mut preview_budget = target.saturating_sub(fixed_chars);
                let mut preview_text = keep_ends_by_chars(&arguments, preview_budget);
                let mut candidate = build_truncated(path, preview_text.clone());
                // JSON escaping can expand characters; tighten by the measured
                // serialized excess.
                while candidate.chars().count() > target && preview_budget > 0 {
                    let excess = candidate.chars().count() - target;
                    preview_budget = preview_budget.saturating_sub(excess.max(1));
                    preview_text = keep_ends_by_chars(&arguments, preview_budget);
                    candidate = build_truncated(path, preview_text.clone());
                }
                (preview_text.chars().count() >= MIN_TOOL_ARGS_PREVIEW_CHARS
                    && candidate.chars().count() < original_chars)
                    .then_some(candidate)
            };
            let Some(truncated) = build_candidate(archive_path_hint.as_deref()) else {
                return FieldTruncateOutcome::target_dependent(reduce_by);
            };
            let archive_file_path =
                archive_truncated_field_to_overflow(message, field, overflow_dir);
            // Required policy: an unarchived stub must never replace the only
            // copy of the field; refuse so the caller keeps the original.
            if archive_file_path.is_none() && archive_policy == FieldArchivePolicy::Required {
                return FieldTruncateOutcome::archive_dependent(overflow_dir);
            }
            // Recompute the preview against the no-path stub after the archive
            // write failed, so a dead path does not eat the preview budget.
            let truncated = build_candidate(archive_file_path.as_deref()).unwrap_or(truncated);
            let Some(call) = message
                .tool_calls
                .as_mut()
                .and_then(|calls| calls.get_mut(call_index))
            else {
                return FieldTruncateOutcome::NoProgress;
            };
            // Arguments must remain valid JSON, but must not resemble a partial
            // payload that the model can copy into a fresh call.
            let Some(pointer) = build_context_overflow_tool_arguments_stub(
                original_chars,
                archive_file_path.as_deref(),
                target,
            ) else {
                return FieldTruncateOutcome::target_dependent(reduce_by);
            };
            debug_assert!(pointer.chars().count() < truncated.chars().count());
            call.function.arguments = pointer;
            FieldTruncateOutcome::Truncated
        }
    }
}

/// Proactively slim down the giant arguments of write_file / apply_patch calls
/// that were "successfully written".
///
/// Once the file is on disk (the result message confirms success), the full
/// content/patch body has no semantic value for later turns — the model references
/// the file path, not the body — so keeping it only occupies context. This is
/// independent of budget pressure: as soon as the group has slid out of the recent
/// protection window (the model can no longer plausibly reference the just-written
/// body to construct follow-up edits), replace it with a
/// `_context_overflow_truncated` pointer stub and archive the original with zero
/// compression. Failed results, in-window results, and current-turn protected ids
/// are never touched. Recovery metadata keeps the original accessible via
/// archive_file_path; current files should be re-read for follow-up edits.
pub(in crate::ai) fn shrink_successful_write_arguments(
    messages: &mut Vec<Message>,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) {
    if messages.is_empty() {
        return;
    }
    // Protection window: the most recent KEEP_RECENT_TOOL_GROUPS tool groups that
    // already have results (including groups just written this turn, whose bodies
    // the model may immediately reference for follow-up edits) — their calls
    // always keep full arguments.
    let protected_recent_call_ids: rustc_hash::FxHashSet<String> =
        recent_tool_result_groups(messages, KEEP_RECENT_TOOL_GROUPS)
            .into_iter()
            .flatten()
            .filter_map(|idx| messages[idx].tool_call_id.clone())
            .collect();
    // tool_call_id -> result text (used to judge success/failure).
    let result_by_call_id: rustc_hash::FxHashMap<String, String> = messages
        .iter()
        .filter(|message| message.role == "tool")
        .filter_map(|message| {
            message
                .tool_call_id
                .as_deref()
                .map(|id| (id.to_string(), value_to_string(&message.content)))
        })
        .collect();

    let mut changed = false;
    for message in messages.iter_mut() {
        if message.role != "assistant" {
            continue;
        }
        let Some(tool_calls) = message.tool_calls.as_mut() else {
            continue;
        };
        // Collect candidates first, to avoid an exclusive-borrow conflict with
        // truncate_mutable_field.
        let mut candidates: Vec<(usize, usize)> = Vec::new();
        for (call_index, call) in tool_calls.iter().enumerate() {
            let name = call.function.name.as_str();
            if name != "write_file" && name != "apply_patch" {
                continue;
            }
            if protected_recent_call_ids.contains(&call.id)
                || protected_tool_call_ids.contains(&call.id)
            {
                continue;
            }
            let arguments = &call.function.arguments;
            if arguments.contains("\"_context_overflow_truncated\"") {
                continue; // already replaced; idempotent (avoids duplicate archiving/duplicate file writes)
            }
            let original_chars = arguments.chars().count();
            if original_chars <= 160 {
                continue;
            }
            let Some(result_text) = result_by_call_id.get(&call.id) else {
                continue;
            };
            if !is_successful_write_result(name, result_text) {
                continue;
            }
            candidates.push((call_index, original_chars));
        }
        for (call_index, original_chars) in candidates {
            if truncate_mutable_field(
                message,
                MutableMessageField::ToolArguments(call_index),
                original_chars.saturating_sub(240),
                overflow_dir,
                FieldArchivePolicy::BestEffort,
            ) {
                changed = true;
            }
        }
    }
    if changed {
        insert_overflow_archive_note_if_exists(messages, overflow_dir);
    }
}

/// Whether a write_file / apply_patch result succeeded. Failed results must keep
/// full arguments so the model can fix from the original text; only successful
/// results are safe to slim down.
pub(in crate::ai) fn is_successful_write_result(tool_name: &str, result_text: &str) -> bool {
    let trimmed = result_text.trim_start();
    if trimmed.starts_with("Error:") || trimmed.starts_with("Exit code:") {
        return false;
    }
    match tool_name {
        "write_file" => trimmed.starts_with("Successfully wrote to"),
        "apply_patch" => trimmed.starts_with("Successfully patched"),
        _ => false,
    }
}
