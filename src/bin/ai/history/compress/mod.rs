use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::ai::types::App;

use super::types::{
    MAX_HISTORY_TURNS, Message, ROLE_INTERNAL_NOTE, is_internal_note_role,
    is_runtime_synthetic_user_message, is_system_like_role, last_real_user_index,
    retained_turn_start,
};

mod dedup;
pub(crate) mod llm_prune;
mod overflow_sink;
mod text_utils;
mod tool_groups;
mod tool_overflow;

use text_utils::{keep_ends_by_chars, summarize_text, truncate_to_chars};
#[cfg(test)]
use tool_groups::{FOLDED_TOOL_GROUP_ARCHIVE_DIR, fold_early_tool_groups};
use tool_groups::{
    MID_TURN_LLM_SUMMARY_KEEP_RECENT_TOOL_GROUPS, ToolGroupFoldPlan, count_tool_group_anchors,
    first_trim_candidate, is_protected_leading_system_like_message, plan_early_tool_groups,
    recent_tool_group_message_indices, recent_tool_result_groups,
};
#[cfg(test)]
use tool_overflow::{
    FINGERPRINT_KEY_COUNT, PRESERVED_TOOL_OVERFLOW_STUB_PREFIX, build_overflow_content_preview,
    build_preserved_tool_overflow_stub, collapse_overflow_stub_to_anchor,
    extract_fingerprint_keywords, minimize_overflow_stub_to_pointer,
    normalize_internal_notes_for_summary_model, stub_fingerprint_line,
    write_preserved_tool_overflow_file_stable,
};
use tool_overflow::{
    age_out_overflow_stub_previews, build_persisted_summary_text,
    build_persisted_summary_text_with_app, cap_oversized_tool_results_for_context,
    enforce_protected_precision_group_budget, is_non_compressible_tool,
    is_preserved_tool_overflow_content, is_preserved_user_or_image_stub,
    merge_old_user_overflow_stubs, minimize_overflow_stubs_for_hard_budget,
    normalize_preserved_message_stubs_for_model, prepare_tool_messages_structured,
    spill_oversized_preserved_messages, spill_protected_precision_to_fit,
    try_spill_preserved_message_to_stub,
};

mod persisted;
pub(in crate::ai) use persisted::*;
mod shrink;
pub(in crate::ai) use shrink::*;
mod truncate;
pub(in crate::ai) use truncate::*;
mod mid_turn;
pub(in crate::ai) use mid_turn::*;
mod billable;
pub(in crate::ai) use billable::*;

/// When the context char count is at or below this threshold, prefer keeping 3
/// recent user turns.
const KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS: usize = 48_000;

/// The checkpoint body is already written to the session asset; this short
/// marker is the model's only index for relocating that body after
/// compression, so history compression must never swallow or drop it. The
/// request projection may replace many old markers with a hierarchical index
/// holding every exact path; that index is protected by the same rule, and the
/// original markers always stay in canonical history.
pub(in crate::ai) fn is_context_checkpoint_marker(m: &Message) -> bool {
    if m.role != ROLE_INTERNAL_NOTE {
        return false;
    }
    let content = value_to_string(&m.content);
    let content = content.trim_start();
    content.starts_with(CONTEXT_CHECKPOINT_MARKER_PREFIX)
        || content.starts_with(QUERY_MEMORY_INDEX_PREFIX)
}

const OVERFLOW_HISTORY_FILENAME: &str = "overflow-history.md";

pub(in crate::ai) const PERSISTED_REASONING_REPLAY_PREFIX: &str =
    "\u{1e}aios:reasoning-content-replay:v1\u{1f}";

/// Inject an archive back-reference after the leading summary; idempotent when an
/// identical back-reference already exists, so each compaction round does not
/// append another identical `internal_note` at the top of the context.
fn insert_archive_note_if_missing(messages: &mut Vec<Message>, archive_note: String) {
    let already_present = messages.iter().any(|message| {
        is_archive_note_message(message) && value_to_string(&message.content) == archive_note
    });
    if already_present {
        return;
    }

    let archive_idx = messages.len().min(1);
    messages.insert(
        archive_idx,
        Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(archive_note),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    );
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::ai) enum MutableMessageField {
    Content,
    Reasoning,
    ToolArguments(usize),
}

impl MutableMessageField {
    fn archive_label(self) -> String {
        match self {
            MutableMessageField::Content => "content".to_string(),
            MutableMessageField::Reasoning => "reasoning_content".to_string(),
            MutableMessageField::ToolArguments(index) => {
                format!("tool_calls[{index}].function.arguments")
            }
        }
    }

    fn original_text(self, message: &Message) -> Option<String> {
        match self {
            MutableMessageField::Content => Some(value_to_string(&message.content)),
            MutableMessageField::Reasoning => message.reasoning_content.clone(),
            MutableMessageField::ToolArguments(index) => message
                .tool_calls
                .as_ref()
                .and_then(|calls| calls.get(index))
                .map(|call| call.function.arguments.clone()),
        }
    }
}

/// Archive-write policy for [`truncate_mutable_field`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::ai) enum FieldArchivePolicy {
    /// On archive-write failure, still truncate to a preview-only inline stub.
    /// Acceptable for rebuildable assistant/tool fields: losing the full text
    /// costs precision, not the user's instruction.
    BestEffort,
    /// The field is the authoritative copy of a user instruction. Archive it
    /// through the trusted session sink before every truncation, including
    /// re-collapsing marker-prefixed text; never trust a path from its content.
    /// Refuse without mutation when that write fails.
    Required,
}

fn messages_total_chars(messages: &[Message]) -> usize {
    messages.iter().map(message_billable_chars).sum::<usize>()
}

fn protected_tool_result_message(
    message: &Message,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) -> bool {
    message.role == "tool"
        && message
            .tool_call_id
            .as_ref()
            .is_some_and(|id| protected_tool_call_ids.contains(id))
}

fn protected_tool_context_message(
    message: &Message,
    protected_tool_call_ids: &rustc_hash::FxHashSet<String>,
) -> bool {
    protected_tool_result_message(message, protected_tool_call_ids)
        || message.tool_calls.as_ref().is_some_and(|calls| {
            calls
                .iter()
                .any(|call| protected_tool_call_ids.contains(&call.id))
        })
}

/// "Budget char count" of a Value's content (Unicode scalar count).
/// Historically this returned byte length, overestimating the char budget ~3x for
/// Chinese/emoji content: a 36K-char soft threshold could be falsely triggered by
/// a 12K-char Chinese turn, re-running the compaction pipeline over and over. Now
/// measured uniformly by `chars().count()`, consistent with the naming of the
/// outer `cap_chars` / `max_chars` thresholds. Image parts are billed nominally
/// via [`IMAGE_BUDGET_CHARS`], keeping base64 text length from polluting the
/// budget (see that constant's doc).
fn value_len_chars(v: &Value) -> usize {
    if let Some(s) = v.as_str() {
        if is_inline_image_data_url(s) {
            return IMAGE_BUDGET_CHARS;
        }
        return s.chars().count();
    }
    if let Some(arr) = v.as_array() {
        return arr.iter().map(content_part_budget_chars).sum();
    }
    v.to_string().chars().count()
}

/// "Billed char count" of a single message entering a model request — the single
/// authoritative measure.
///
/// Historically many budget checks counted only `content`, entirely missing
/// `tool_calls[].function.arguments` (typically `apply_patch` puts a whole large
/// patch into arguments with empty content) and `reasoning_content` (long
/// thinking chains), letting large messages bypass compaction gating and making
/// TPM preflight and the max_tokens clamp underestimate input together.
///
/// This combines all three into one measure, aligned with the SQL side
/// `total_message_chars_sqlite`
/// (`length(content)+length(tool_calls)+length(reasoning_content)`), so the
/// "in-memory budget" and "persisted budget" share one measure. Images are still
/// billed nominally via [`IMAGE_BUDGET_CHARS`] (see `value_len_chars`).
pub(in crate::ai) fn message_billable_chars(m: &Message) -> usize {
    let content_chars = value_len_chars(&m.content);
    let tool_call_chars = m
        .tool_calls
        .as_ref()
        .map(|tc| {
            serde_json::to_string(tc)
                .map(|s| s.chars().count())
                .unwrap_or(0)
        })
        .unwrap_or(0);
    let reasoning_chars = m
        .reasoning_content
        .as_deref()
        .map(|s| s.chars().count())
        .unwrap_or(0);
    content_chars + tool_call_chars + reasoning_chars
}

pub(in crate::ai) fn value_to_string(v: &Value) -> String {
    if let Some(s) = v.as_str() {
        return s.to_string();
    }
    // Multimodal messages (JSON arrays): extract only the text parts and discard
    // image base64 data, avoiding feeding huge base64 content to the model or
    // showing it to the user when generating summaries/titles.
    if let Some(arr) = v.as_array() {
        let mut text_parts = Vec::new();
        let mut has_image = false;
        for item in arr {
            if let Some(obj) = item.as_object() {
                let item_type = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match item_type {
                    "text" => {
                        if let Some(t) = obj.get("text").and_then(|t| t.as_str()) {
                            let trimmed = t.trim();
                            if !trimmed.is_empty() {
                                text_parts.push(trimmed.to_string());
                            }
                        }
                    }
                    // The persisted reference form (build_reference_content)
                    // counts as an image too, so image-only messages still
                    // summarize to "[图片]" instead of leaking the file path.
                    "image_url" => has_image = true,
                    "reference" if obj.get("kind").and_then(|k| k.as_str()) == Some("image") => {
                        has_image = true;
                    }
                    // Persisted text-file/PDF references render as a marker
                    // (name only, never the raw path/content) so /history and
                    // summaries show the attachment boundary instead of leaking
                    // file contents.
                    "reference" if obj.get("kind").and_then(|k| k.as_str()) == Some("file") => {
                        let name = obj.get("name").and_then(|n| n.as_str()).unwrap_or("file");
                        text_parts.push(format!("[Attached file: {name}]"));
                    }
                    // Any other/future reference kind renders as a marker so
                    // summaries never leak raw JSON and new kinds are not
                    // silently dropped from /history or summary text.
                    "reference" => {
                        let kind = obj
                            .get("kind")
                            .and_then(|k| k.as_str())
                            .unwrap_or("reference");
                        let name = obj
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or("attachment");
                        text_parts.push(format!("[{kind}: {name}]"));
                    }
                    _ => {}
                }
            }
        }
        if text_parts.is_empty() && has_image {
            return "[图片]".to_string();
        }
        return text_parts.join(" ");
    }
    v.to_string()
}

fn normalize_whitespace(s: &str) -> String {
    let mut out = String::new();
    let mut in_ws = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !in_ws {
                out.push(' ');
                in_ws = true;
            }
        } else {
            out.push(ch);
            in_ws = false;
        }
    }
    out.trim().to_string()
}

fn is_summary_message(message: &Message) -> bool {
    if !is_system_like_role(&message.role) {
        return false;
    }
    is_summary_note_text(&value_to_string(&message.content))
}

/// Protection window over recent complete tool groups. An assistant(tool_calls)
/// batch is an indivisible evidence unit: it must never be truncated message by
/// message, or parallel reads would leave half a batch behind.
const KEEP_RECENT_TOOL_GROUPS: usize = 4;

/// Window sequence for progressive folding: tightens stepwise from
/// [`KEEP_RECENT_TOOL_GROUPS`] down to [`MIN_KEEP_RECENT_TOOL_GROUPS`], widening
/// what may be folded but never reaching 0. Both progressive paths
/// ([`fold_noncompressible_tool_groups_to_fit`] and mid-turn summarization's
/// Path B+C) share one sequence so the minimum-protection policy has a single
/// source of truth.
fn progressive_fold_windows() -> Vec<usize> {
    let mut windows = Vec::new();
    let mut keep = KEEP_RECENT_TOOL_GROUPS;
    loop {
        windows.push(keep);
        if keep <= MIN_KEEP_RECENT_TOOL_GROUPS {
            break;
        }
        keep = (keep / 2).max(MIN_KEEP_RECENT_TOOL_GROUPS);
    }
    windows
}

/// Whether message content contains a real image attachment (OpenAI Vision
/// schema). The image must exist as a multimodal `Value::Array`, and the array
/// must contain
/// `{"type":"image_url", "image_url":{...}}`。
/// The old implementation misjudged via `text.contains("data:image/")`: the agent
/// merely discussing the `data:image/png` string in plain text got the whole
/// message replaced, losing information.
fn message_contains_image(content: &Value) -> bool {
    let Some(arr) = content.as_array() else {
        return false;
    };
    arr.iter().any(|item| {
        item.get("type").and_then(|v| v.as_str()) == Some("image_url")
            || item.get("image_url").is_some()
    })
}

#[cfg(test)]
use dedup::dedup_overlapping_read_file_results;
use dedup::{dedup_adjacent, dedup_repeated_tool_results, keep_only_recent_reasoning_content};
pub(in crate::ai) use overflow_sink::compressed_tool_evidence_exceeds_inline_budget;
use overflow_sink::{
    OverflowSink, PlannedArchiveWrite, archive_internal_notes_deduplicated,
    archive_messages_to_overflow, archive_truncated_field_to_overflow, build_overflow_placeholder,
    content_sha256_hex, insert_internal_note_archive_note_if_needed,
    insert_overflow_archive_note_if_exists, trim_compressed_tool_evidence_to_inline_budget,
};

#[cfg(test)]
mod coalesce_summary_notes_tests;
#[cfg(test)]
mod dedup_adjacent_tests;
#[cfg(test)]
mod drop_trim_differential_tests;
#[cfg(test)]
mod fold_early_tool_groups_tests;
#[cfg(test)]
mod overflow_sink_dedup_tests;
#[cfg(test)]
mod overflow_stub_merge_tests;
#[cfg(test)]
mod shrink_successful_write_arguments_tests;
#[cfg(test)]
mod tail_window_tests;
#[cfg(test)]
mod tool_overflow_tests;
#[cfg(test)]
mod truncate_last_real_user_message_tests;
