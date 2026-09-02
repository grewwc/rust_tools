//! Mutable-message truncation to fit budgets: per-field truncation,
//! overflow pointer building, and emergency caps.

use super::*;

/// The last-resort hard-budget escape valve: keep the system/user and tool-call
/// pairing structure intact and only shorten rebuildable assistant/tool bodies,
/// reasoning, and oversized tool arguments. Current high-precision results are
/// protected first; if the target is still not met, truncating those results is
/// allowed. If the untrimmable system/user content itself is already over the
/// limit, return false.
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
    if messages_total_chars(messages) <= max_chars {
        return true;
    }

    let mut blocked_fields = rustc_hash::FxHashSet::default();
    for include_protected in [false, true] {
        if include_protected && !allow_protected_fallback {
            break;
        }
        while messages_total_chars(messages) > max_chars {
            let excess = messages_total_chars(messages).saturating_sub(max_chars);
            let mut best: Option<(usize, MutableMessageField, usize)> = None;
            for (index, message) in messages.iter().enumerate() {
                if is_system_like_role(&message.role) || message.role == "user" {
                    continue;
                }
                let is_protected = protected_tool_context_message(message, protected_tool_call_ids);
                if is_protected && !include_protected {
                    continue;
                }
                let content_chars = value_len_chars(&message.content);
                if !message_contains_image(&message.content)
                    && !is_preserved_tool_overflow_content(&message.content)
                    && content_chars > 160
                    && !blocked_fields.contains(&(index, MutableMessageField::Content))
                {
                    choose_larger_mutable_field(
                        &mut best,
                        (index, MutableMessageField::Content, content_chars - 160),
                    );
                }
                if let Some(reasoning) = message.reasoning_content.as_deref()
                    && !is_persisted_reasoning_replay(reasoning)
                    && reasoning.chars().count() > 160
                    && !blocked_fields.contains(&(index, MutableMessageField::Reasoning))
                {
                    choose_larger_mutable_field(
                        &mut best,
                        (
                            index,
                            MutableMessageField::Reasoning,
                            reasoning.chars().count() - 160,
                        ),
                    );
                }
                if let Some(tool_calls) = &message.tool_calls {
                    for (call_index, call) in tool_calls.iter().enumerate() {
                        let argument_chars = call.function.arguments.chars().count();
                        let field = MutableMessageField::ToolArguments(call_index);
                        if argument_chars > 160 && !blocked_fields.contains(&(index, field)) {
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
            if !truncate_mutable_field(
                &mut messages[message_index],
                field,
                reduce_by,
                overflow_dir,
                FieldArchivePolicy::BestEffort,
            ) {
                // A fixed marker / archive path may already be the smallest
                // form the field can reach. Skip the no-progress field and try
                // other candidates instead of re-selecting the same one.
                blocked_fields.insert((message_index, field));
                continue;
            }
            insert_overflow_archive_note_if_exists(messages, overflow_dir);
            // Inserting the archive note the first time may shift message indices;
            // re-evaluate candidates after a successful shrink.
            blocked_fields.clear();
        }
    }

    messages_total_chars(messages) <= max_chars
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
    let mut truncated_any = false;
    for message in messages.iter_mut() {
        if is_system_like_role(&message.role)
            || message.role == "user"
            || protected_tool_context_message(message, protected_tool_call_ids)
        {
            continue;
        }
        let content_chars = value_len_chars(&message.content);
        if !message_contains_image(&message.content)
            && !is_preserved_tool_overflow_content(&message.content)
            && content_chars > per_field_cap
        {
            truncated_any |= truncate_mutable_field(
                message,
                MutableMessageField::Content,
                content_chars - per_field_cap,
                overflow_dir,
                FieldArchivePolicy::BestEffort,
            );
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
            let argument_chars = message
                .tool_calls
                .as_ref()
                .and_then(|calls| calls.get(call_index))
                .map(|call| call.function.arguments.chars().count())
                .unwrap_or(0);
            if argument_chars > per_field_cap {
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

/// Truncated tool arguments are terminal too. When archiving failed, only the
/// preview remains; re-archiving it would falsely claim the full arguments are
/// recoverable.
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

pub(in crate::ai) fn build_context_overflow_tool_arguments_pointer(arguments: &str, target: usize) -> Option<String> {
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
    let pointer = match archived_path {
        Some(path) => serde_json::json!({
            "_context_overflow_truncated": true,
            "archive_file_path": path,
        })
        .to_string(),
        None => serde_json::json!({
            "_context_overflow_truncated": true,
            "archive_file_path": Value::Null,
            "original_unavailable": true,
        })
        .to_string(),
    };
    (pointer.chars().count() <= target && pointer.chars().count() < arguments.chars().count())
        .then_some(pointer)
}

pub(in crate::ai) fn truncate_mutable_field(
    message: &mut Message,
    field: MutableMessageField,
    reduce_by: usize,
    overflow_dir: Option<&Path>,
    archive_policy: FieldArchivePolicy,
) -> bool {
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
            if is_preserved_tool_overflow_content(&message.content) {
                return false;
            }
            let text = value_to_string(&message.content);
            let original_chars = text.chars().count();
            let target = original_chars.saturating_sub(reduce_by).max(160);
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
                        return false;
                    };
                    if pointer_for_path(path_hint).is_none() {
                        return false;
                    }
                    let Some(archive_file_path) =
                        archive_truncated_field_to_overflow(message, field, overflow_dir)
                    else {
                        return false;
                    };
                    let Some(pointer) = pointer_for_path(&archive_file_path) else {
                        return false;
                    };
                    message.content = Value::String(pointer);
                    return true;
                }
                if let Some(pointer) = build_context_overflow_pointer(&text, target) {
                    if pointer.chars().count() < original_chars {
                        message.content = Value::String(pointer);
                        return true;
                    }
                }
                return false;
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
                return false;
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
                return false;
            }
            let truncated = build_truncated(archive_file_path.as_deref()).unwrap_or(truncated);
            message.content = Value::String(truncated);
            true
        }
        MutableMessageField::Reasoning => {
            let Some(reasoning) = message.reasoning_content.as_deref() else {
                return false;
            };
            // Exact/encrypted replay payloads must remain byte-for-byte intact.
            // Keeping the marker while truncating its payload would break
            // decoding in the request layer, so callers must shrink another
            // field instead.
            if is_persisted_reasoning_replay(reasoning) {
                return false;
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
                return false;
            };
            let archive_file_path =
                archive_truncated_field_to_overflow(message, field, overflow_dir);
            // Required policy: an unarchived stub must never replace the only
            // copy of the field; refuse so the caller keeps the original.
            if archive_file_path.is_none() && archive_policy == FieldArchivePolicy::Required {
                return false;
            }
            let truncated = build_truncated(archive_file_path.as_deref()).unwrap_or(truncated);
            message.reasoning_content = Some(truncated);
            true
        }
        MutableMessageField::ToolArguments(call_index) => {
            let Some(arguments) = message
                .tool_calls
                .as_ref()
                .and_then(|calls| calls.get(call_index))
                .map(|call| call.function.arguments.clone())
            else {
                return false;
            };
            let original_chars = arguments.chars().count();
            let target = original_chars.saturating_sub(reduce_by).max(160);
            if is_context_overflow_truncated_tool_arguments(&arguments) {
                let Some(pointer) =
                    build_context_overflow_tool_arguments_pointer(&arguments, target)
                else {
                    return false;
                };
                let Some(call) = message
                    .tool_calls
                    .as_mut()
                    .and_then(|calls| calls.get_mut(call_index))
                else {
                    return false;
                };
                call.function.arguments = pointer;
                return true;
            }
            // A fixed JSON prefix plus a long archive path can consume the
            // entire target. Reject empty or near-empty previews: they contain
            // no real argument data and can make the model replay protocol keys
            // as tool arguments. Larger previews remain useful even when the
            // path consumes much of the budget.
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
                return false;
            };
            let archive_file_path =
                archive_truncated_field_to_overflow(message, field, overflow_dir);
            // Required policy: an unarchived stub must never replace the only
            // copy of the field; refuse so the caller keeps the original.
            if archive_file_path.is_none() && archive_policy == FieldArchivePolicy::Required {
                return false;
            }
            // Recompute the preview against the no-path stub after the archive
            // write failed, so a dead path does not eat the preview budget.
            let truncated = build_candidate(archive_file_path.as_deref()).unwrap_or(truncated);
            let Some(call) = message
                .tool_calls
                .as_mut()
                .and_then(|calls| calls.get_mut(call_index))
            else {
                return false;
            };
            // Arguments must remain valid JSON; slicing the string directly
            // would make the provider reject the request.
            call.function.arguments = truncated;
            true
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
/// are never touched, so agent effectiveness does not degrade (when needed, the
/// model can still read the original back via the stub's archive_file_path, or
/// recognize the file's content outline from the preview).
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
