//! Tool group folding.
//!
//! Fold earlier `assistant(tool_calls) + accompanying tool` groups in the message
//! sequence into a single `internal_note` stub, keeping the most recent groups verbatim.

use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;
use std::path::Path;

use crate::ai::types::ToolCall;

use super::super::types::{Message, ROLE_INTERNAL_NOTE, ROLE_SYSTEM, retained_turn_start};
use super::text_utils::truncate_to_chars;
use super::tool_overflow::{
    build_tool_overflow_recall_lines, is_non_compressible_tool, is_preserved_tool_overflow_stub,
    is_preserved_user_or_image_stub, plan_noncompressible_tool_result_for_fold,
};
use super::{
    COMPRESSED_TOOL_EVIDENCE_MARKER, PlannedArchiveWrite, content_sha256_hex,
    is_archive_note_message, is_compressed_tool_evidence_note, is_context_checkpoint_marker,
    is_summary_message, keep_recent_user_turns_when_trimming, normalize_whitespace,
    value_to_string,
};

pub(super) const FOLDED_TOOL_GROUP_ARCHIVE_DIR: &str = "folded-tool-groups";

/// Header of the folded-group archive file. The archived JSON is a verbatim
/// copy of the messages folded out of the *request projection*, not of the
/// canonical history: high-precision tool results (tools that forbid lossy
/// compression, e.g. read_file / execute_command / apply_patch / task results)
/// appear either as spill stubs (when a prior stage already spilled them,
/// carrying `original_file_path` / `archive_file_path`) or as their full text
/// (when still verbatim at fold time - protected by the recent-window until
/// this fold, or below the spill threshold); lossy-compressible results (e.g.
/// plan, tree) may have been reduced to a summary before folding. Stating this
/// in the header keeps a model reading the archive from expecting the original
/// full tool output inside it, and tells it how to retrieve current output
/// instead.
const FOLDED_GROUP_ARCHIVE_HEADER: &str = "\
# Folded tool group (request-projection copy)

This JSON is a verbatim copy of the messages folded out of the request \
projection, not the canonical history. High-precision tool results (tools that
forbid lossy compression, e.g. read_file / execute_command / apply_patch / task
results) appear here either as spill stubs carrying `original_file_path` /
`archive_file_path` (when a prior stage already spilled them to
`tool-overflow-compressed/`) or as their full text (when they were still
verbatim at fold time - protected by the recent-window until this fold, or
below the spill threshold). Lossy-compressible tool results (e.g. plan, tree)
may have been reduced to a structured summary before folding; when they were,
only the summary is archived here. Results that survived summarization
(recent-window protection or small output) are kept verbatim in this file. Read
this file for the exact archived content; re-run the tool against the same
target only when the content here lacks the detail you need.

raw_message_json:
```json
";

pub(super) struct ToolGroupFoldPlan {
    messages: Vec<Message>,
    folded_groups: usize,
    archives: Vec<PlannedArchiveWrite>,
}

impl ToolGroupFoldPlan {
    fn unchanged(messages: &[Message]) -> Self {
        Self {
            messages: messages.to_vec(),
            folded_groups: 0,
            archives: Vec::new(),
        }
    }

    pub(super) fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub(super) fn folded_groups(&self) -> usize {
        self.folded_groups
    }

    pub(super) fn commit(&self) -> bool {
        self.archives.iter().all(PlannedArchiveWrite::commit)
    }

    pub(super) fn into_result(self) -> (Vec<Message>, usize) {
        (self.messages, self.folded_groups)
    }
}

pub(super) fn first_tool_call_group(messages: &[Message]) -> Option<Vec<usize>> {
    let assistant_idx = messages.iter().position(|m| {
        if m.role != "assistant" {
            return false;
        }
        let Some(tool_calls) = &m.tool_calls else {
            return false;
        };
        if tool_calls.is_empty() {
            return false;
        }
        !tool_calls
            .iter()
            .any(|tc| is_non_compressible_tool(&tc.function.name))
    })?;
    let tool_call_ids: Vec<String> = messages[assistant_idx]
        .tool_calls
        .as_ref()
        .unwrap()
        .iter()
        .map(|tc| tc.id.clone())
        .collect();
    let mut group = vec![assistant_idx];
    // Only collect tool messages that appear consecutively right after the
    // assistant, stopping at any non-tool message (including the next
    // assistant/user/system/internal_note). This keeps stale same-id stubs left
    // over from other turns (produced by dedup_repeated_tool_results replacement)
    // from being pulled into the same group, which would fold/delete whole groups
    // across turns.
    let mut i = assistant_idx + 1;
    while i < messages.len() && messages[i].role == "tool" {
        if let Some(ref id) = messages[i].tool_call_id {
            if tool_call_ids.contains(id) {
                group.push(i);
            } else {
                // A tool message not belonging to this assistant appeared in the
                // same position (should not happen, but if it does, stop scanning
                // to avoid breaking the OpenAI pairing protocol).
                break;
            }
        } else {
            break;
        }
        i += 1;
    }
    Some(group)
}

pub(super) fn first_trim_candidate(messages: &[Message], budget: usize) -> Option<usize> {
    let keep_recent_user_turns = keep_recent_user_turns_when_trimming(messages, budget);
    let protected_tail_start = retained_turn_start(messages, keep_recent_user_turns);

    // Skip head-protected system-like messages: the real system prompt, history
    // summaries, archive pointers, and checkpoints. Do not blanket-protect all
    // internal_note messages: compressed_tool_round is also an internal_note, and
    // if one sits at the head of the persisted history it becomes untrimmable
    // noise that keeps eating context.
    // The old implementation only skipped entries prefixed with "conversation
    // summary / history summary", treating an ordinary system prompt as trimmable
    // and triggering "replies cut off abruptly after context compression".
    // Also treat the entire tail window starting from the most recent N user
    // turns as protected, so a multi-stage task's previous subgoal is not split
    // from the current one.
    let mut index = 0usize;
    while index < messages.len() && is_protected_leading_system_like_message(&messages[index]) {
        index += 1;
    }

    while index < protected_tail_start {
        let message = &messages[index];

        // The checkpoint body is already on disk; the short marker is the only
        // way back to it. It must not be trimmed by the fallback merely for
        // appearing early in the conversation.
        if is_context_checkpoint_marker(message) {
            index += 1;
            continue;
        }

        // user/image placeholder stubs spilled to session files are not deleted:
        // they are just pointers to archive files (original text stored on disk
        // with zero compression); deleting them would leave the model with no
        // leads. Whether ordinary user / image-bearing messages may be deleted is
        // decided by the caller path; this function only reports candidates:
        // - with_summary: drop + archive + summarize (returning the user
        //   authorizes deletion; the summary carries the early goal, otherwise
        //   30+ turns of plain-text dialogue would never converge into budget);
        // - plain shrink / batch trim: user can only be OffloadOnly — spill to
        //   disk first; on failure plain shrink must break (otherwise the same
        //   user would be returned forever), and batch skips that user and keeps
        //   scanning later candidates; never remove the original text.
        if is_preserved_user_or_image_stub(&value_to_string(&message.content)) {
            index += 1;
            continue;
        }

        // tool messages are never deleted alone: that could break the
        // assistant(tool_calls) ↔ tool pairing.
        if message.role == "tool" {
            index += 1;
            continue;
        }

        // Assistants carrying tool_calls are never deleted alone: preserves
        // protocol pairing consistency.
        if message.role == "assistant"
            && message
                .tool_calls
                .as_ref()
                .map(|calls| !calls.is_empty())
                .unwrap_or(false)
        {
            index += 1;
            continue;
        }

        return Some(index);
    }

    None
}

pub(super) fn is_protected_leading_system_like_message(message: &Message) -> bool {
    if message.role == ROLE_SYSTEM {
        return true;
    }
    if message.role != ROLE_INTERNAL_NOTE {
        return false;
    }
    if is_compressed_tool_evidence_note(message) {
        return false;
    }
    is_summary_message(message)
        || is_archive_note_message(message)
        || is_context_checkpoint_marker(message)
}

/// Plan folding one `(assistant tool_calls + accompanying tool results)` group
/// into a single `internal_note`. This only produces the deterministic path and
/// content to write; no file I/O is performed.
fn plan_tool_call_group_fold(
    messages: &[Message],
    group: &[usize],
    overflow_dir: Option<&Path>,
) -> Option<(Message, Vec<PlannedArchiveWrite>)> {
    if group.is_empty() {
        return None;
    }
    let assistant_idx = group[0];
    let assistant = messages.get(assistant_idx)?;
    let tool_calls = assistant.tool_calls.as_ref()?;
    if tool_calls.is_empty() {
        return None;
    }

    let mut archives = Vec::new();
    let archive_file_path = if let Some(dir) = overflow_dir {
        let group_messages = group
            .iter()
            .filter_map(|idx| messages.get(*idx).cloned())
            .collect::<Vec<_>>();
        let raw_messages = serde_json::to_string_pretty(&group_messages).ok()?;
        let content = format!("{FOLDED_GROUP_ARCHIVE_HEADER}{raw_messages}\n```\n");
        let digest = content_sha256_hex(content.as_bytes());
        let path = dir
            .join(FOLDED_TOOL_GROUP_ARCHIVE_DIR)
            .join(format!("{digest}.md"));
        archives.push(PlannedArchiveWrite::new(path.clone(), content));
        Some(path.to_string_lossy().to_string())
    } else {
        None
    };

    let mut lines = Vec::with_capacity(tool_calls.len() + 8);
    lines.push(format!(
        "compressed_tool_round: {} tool calls (folded for context budget)",
        tool_calls.len()
    ));
    lines.push(COMPRESSED_TOOL_EVIDENCE_MARKER.to_string());
    if let Some(path) = archive_file_path.as_deref() {
        lines.push(format!("- archive_file_path: {path}"));
        lines.push("- archive_scope: folded_tool_group_projection_messages".to_string());
    }

    // Checkpoints use only the user-visible assistant body. Hidden
    // reasoning_content may be unverified intermediate inference and must not be
    // promoted to persisted fact during compression; when the body is empty,
    // rebuild "completed tool activity" from the structured tool calls so the
    // model does not assume those calls were never executed.
    let assistant_content = match &assistant.content {
        Value::Null => String::new(),
        content => value_to_string(content),
    };
    let assistant_text = normalize_whitespace(assistant_content.trim());
    let checkpoint = if !assistant_text.is_empty() {
        assistant_text
    } else {
        reconstructed_tool_call_checkpoint(tool_calls)
    };
    lines.push(format!(
        "assistant_checkpoint: {}",
        truncate_to_chars(&checkpoint, 720)
    ));
    lines.push("evidence:".to_string());

    for tc in tool_calls.iter().take(8) {
        let result_text = group
            .iter()
            .skip(1)
            .find_map(|idx| {
                let m = messages.get(*idx)?;
                if m.tool_call_id.as_deref() == Some(tc.id.as_str()) {
                    Some(value_to_string(&m.content))
                } else {
                    None
                }
            })
            .unwrap_or_default();
        let recall = tool_result_recall_text(tc, &result_text, overflow_dir, &mut archives)?;
        let invocation = tool_call_invocation_recall(tc);
        let target = tool_call_target_recall(tc);
        lines.push(format!(
            "- {}{}{} => {}",
            tc.function.name, target, invocation, recall
        ));
    }
    if tool_calls.len() > 8 {
        lines.push(format!(
            "- ... ({} more tools omitted)",
            tool_calls.len() - 8
        ));
    }
    // The `reuse evidence` guidance is only truthful when the group keeps a
    // recoverable pointer (a non-compressible tool spill, or an already-spilled
    // stub). Lossy-only groups may archive either a summary (results summarized
    // before folding) or the full text (results that survived summarization -
    // recent-window protection or small output), so their guidance points at
    // `archive_file_path` first and falls back to re-running the tool only when
    // the archived text lacks the detail needed. When no archive was written at
    // all (`overflow_dir = None`), neither pointer exists and the evidence lines
    // above are the only record, so the guidance must say so instead of
    // referencing a nonexistent archive.
    let has_recoverable_result = tool_calls
        .iter()
        .any(|tool_call| is_non_compressible_tool(&tool_call.function.name))
        || group.iter().skip(1).any(|idx| {
            messages.get(*idx).is_some_and(|message| {
                is_preserved_tool_overflow_stub(&value_to_string(&message.content))
            })
        });
    if has_recoverable_result {
        lines.push("compression_decision: reuse the evidence above before repeating the same read/search/list/command action; only re-run or re-read if exact omitted text is required or the underlying target changed.".to_string());
    } else if archive_file_path.is_some() {
        lines.push("compression_decision: the archived tool results are the request-projection content at fold time - full text when the result survived lossy summarization (recent-window protection or small output), otherwise only a lossy summary. Read `archive_file_path` for the exact archived content; re-run the affected tool against the same target only when the archived text lacks the detail you need.".to_string());
    } else {
        lines.push("compression_decision: no archive was written for this fold (overflow archiving is disabled); the evidence lines above are the only remaining record of these tool results. Re-run the affected tool against the same target to get current output.".to_string());
    }

    Some((
        Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(lines.join("\n")),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        archives,
    ))
}

fn parsed_tool_args(tool_call: &ToolCall) -> Option<Value> {
    serde_json::from_str::<Value>(&tool_call.function.arguments).ok()
}

fn reconstructed_tool_call_checkpoint(tool_calls: &[ToolCall]) -> String {
    let mut calls = Vec::new();
    for tool_call in tool_calls.iter().take(4) {
        let target = tool_call_target_recall(tool_call);
        let invocation = tool_call_invocation_recall(tool_call);
        let detail = format!("{target}{invocation}");
        calls.push(format!("{}{}", tool_call.function.name, detail));
    }
    if tool_calls.len() > 4 {
        calls.push(format!("... {} more tool calls", tool_calls.len() - 4));
    }
    format!(
        "no assistant narration was persisted; reconstructed completed tool activity: {}",
        calls.join("; ")
    )
}

fn arg_string(args: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| args.get(*key).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn arg_u64(args: &Value, key: &str) -> Option<u64> {
    args.get(key)
        .and_then(|value| value.as_u64().or_else(|| value.as_str()?.parse().ok()))
}

fn tool_call_target_recall(tool_call: &ToolCall) -> String {
    let Some(args) = parsed_tool_args(tool_call) else {
        return String::new();
    };
    let mut fields = Vec::new();
    match tool_call.function.name.as_str() {
        "read_file" => {
            if let Some(path) = arg_string(&args, &["file_path", "path", "filePath"]) {
                fields.push(format!(
                    "file: {}",
                    truncate_to_chars(&normalize_whitespace(&path), 240)
                ));
            }
            if let Some(offset) = arg_u64(&args, "offset") {
                if let Some(limit) = arg_u64(&args, "limit") {
                    fields.push(format!(
                        "range: lines={}..{}",
                        offset,
                        offset.saturating_add(limit.saturating_sub(1))
                    ));
                } else {
                    fields.push(format!("range: offset={offset}"));
                }
            } else if let Some(limit) = arg_u64(&args, "limit") {
                fields.push(format!("range: first {limit} lines"));
            }
        }
        "tree" => {
            if let Some(path) = arg_string(&args, &["path"]) {
                fields.push(format!(
                    "path: {}",
                    truncate_to_chars(&normalize_whitespace(&path), 160)
                ));
            }
        }
        "write_file" | "create_file" | "edit_file" => {
            if let Some(path) = arg_string(&args, &["file_path", "path", "filePath"]) {
                fields.push(format!(
                    "file: {}",
                    truncate_to_chars(&normalize_whitespace(&path), 240)
                ));
            }
        }
        _ => {}
    }

    if fields.is_empty() {
        String::new()
    } else {
        format!(" [{}]", fields.join("; "))
    }
}

/// An `execute_command` result by itself cannot say which question it answered.
/// Tool group folding removes the original tool_call, so the command and cwd must
/// stay in the recall text; otherwise multiple indistinguishable "successful but
/// empty" git logs would degrade into records the model treats as never run and
/// starts investigating again.
fn tool_call_invocation_recall(tool_call: &ToolCall) -> String {
    if tool_call.function.name != "execute_command" {
        return String::new();
    }

    let args = serde_json::from_str::<Value>(&tool_call.function.arguments).ok();
    let command = args
        .as_ref()
        .and_then(|args| args.get("command"))
        .and_then(Value::as_str)
        .map(|command| truncate_to_chars(&normalize_whitespace(command), 720));
    let cwd = args
        .as_ref()
        .and_then(|args| args.get("cwd"))
        .and_then(Value::as_str)
        .filter(|cwd| !cwd.trim().is_empty())
        .map(|cwd| truncate_to_chars(&normalize_whitespace(cwd), 240));

    let mut fields = Vec::with_capacity(2);
    if let Some(command) = command.filter(|command| !command.is_empty()) {
        fields.push(format!("command: {command}"));
    }
    if let Some(cwd) = cwd.filter(|cwd| !cwd.is_empty()) {
        fields.push(format!("cwd: {cwd}"));
    }
    if fields.is_empty() {
        String::new()
    } else {
        format!(" [{}]", fields.join("; "))
    }
}

/// Generate the result recall text for tool group folding. High-precision results
/// must be archived before the original messages are removed; if archiving fails,
/// return `None` so the caller keeps the whole group verbatim rather than
/// demoting the only evidence to a single sentence.
fn tool_result_recall_text(
    tool_call: &ToolCall,
    result_text: &str,
    overflow_dir: Option<&Path>,
    archives: &mut Vec<PlannedArchiveWrite>,
) -> Option<String> {
    let tool_name = tool_call.function.name.as_str();
    let already_archived = result_text
        .lines()
        .map(str::trim)
        .any(|line| line.starts_with("- file_path:") || line.starts_with("file_path:"));
    if !is_non_compressible_tool(tool_name) || result_text.trim().is_empty() {
        let recall = tool_result_recall_one_liner(result_text);
        return Some(if already_archived {
            append_original_recall_lines(recall, tool_call, result_text)
        } else {
            recall
        });
    }

    let original_recall_lines =
        build_tool_overflow_recall_lines(&tool_call.function.name, &tool_call.function.arguments);
    let (preserved, archive) = plan_noncompressible_tool_result_for_fold(
        overflow_dir,
        &tool_call.id,
        tool_name,
        result_text,
        &original_recall_lines,
    )?;
    if let Some(archive) = archive {
        archives.push(archive);
    }
    let file_path_line = preserved_file_path_line(&preserved);
    let archive_path_line = preserved_archive_file_path_line(&preserved);

    if tool_name == "execute_command" {
        let full_output_hint = file_path_line
            .clone()
            .or_else(|| archive_path_line.clone())
            .unwrap_or_else(|| "完整日志已归档到会话 asset。".to_string());
        if already_archived {
            return Some(append_original_recall_lines(
                full_output_hint,
                tool_call,
                &preserved,
            ));
        }

        let recall = command_result_recall(result_text);
        let recall_lower = recall.to_ascii_lowercase();
        let has_error_signal = recall_lower.contains("error")
            || recall_lower.contains("failed")
            || recall_lower.contains("panic")
            || recall_lower.contains("blocked")
            || recall_lower.contains("aborting")
            || recall_lower.contains("could not compile");
        if has_error_signal {
            return Some(append_original_recall_lines(
                format!(
                    "{}\n  {}\n  - 命令输出包含错误信号；仅当当前诊断确实需要完整日志时，再读取 `file_path`。",
                    recall, full_output_hint
                ),
                tool_call,
                &preserved,
            ));
        }
        return Some(append_original_recall_lines(
            format!("{}\n  {}", recall, full_output_hint),
            tool_call,
            &preserved,
        ));
    }

    if tool_name == "read_file" {
        return Some(precision_grounding_tool_recall(
            tool_call,
            &preserved,
            archive_path_line,
        ));
    }

    Some(append_original_recall_lines(
        archive_path_line.unwrap_or_else(|| "完整结果已归档到会话 asset。".to_string()),
        tool_call,
        &preserved,
    ))
}

/// Tool group folding is the second compression level: the `original_*` call
/// anchors in first-level stubs must not be dropped a second time.
///
/// Rebuild them from ToolCall arguments that are still present; fall back to the
/// anchors already in the stub when the format is old or argument parsing fails.
/// That way, even when history retains only internal archive paths, the model
/// still knows what the original file, command, or search was.
fn append_original_recall_lines(
    mut recall: String,
    tool_call: &ToolCall,
    preserved: &str,
) -> String {
    for line in collect_original_recall_lines(tool_call, preserved) {
        recall.push_str("\n  ");
        recall.push_str(&line);
    }
    recall
}

fn collect_original_recall_lines(tool_call: &ToolCall, preserved: &str) -> Vec<String> {
    let mut seen = FxHashSet::default();
    let mut out = Vec::new();
    let from_call =
        build_tool_overflow_recall_lines(&tool_call.function.name, &tool_call.function.arguments);
    let from_preserved = preserved.lines().filter_map(|line| {
        let line = line.trim();
        line.starts_with("- original_").then_some(line)
    });

    for line in from_call.iter().map(String::as_str).chain(from_preserved) {
        if seen.insert(line.to_string()) {
            out.push(line.to_string());
        }
    }
    out
}

fn preserved_file_path_value(preserved: &str) -> Option<String> {
    preserved
        .lines()
        .map(str::trim)
        .find_map(|line| {
            line.strip_prefix("- file_path: ")
                .or_else(|| line.strip_prefix("file_path:"))
                .map(str::trim)
        })
        .filter(|path| !path.is_empty())
        .map(|path| truncate_to_chars(&normalize_whitespace(path), 240))
}

fn preserved_file_path_line(preserved: &str) -> Option<String> {
    preserved_file_path_value(preserved).map(|path| format!("- file_path: {path}"))
}

/// When folding read_file results, do not disguise the internal overflow archive
/// path as the ordinary `file_path` lead; the real follow-up investigation target
/// should be `original_file_path` / `original_range`.
fn preserved_archive_file_path_line(preserved: &str) -> Option<String> {
    preserved_file_path_value(preserved).map(|path| format!("- archive_file_path: {path}"))
}

/// First-level overflow stubs already carry head/tail previews; keep a few
/// previews during second-level tool-group folding too, so the history does not
/// degrade into just the two path entries `original_file_path`/`archive_file_path`,
/// leaving the model without the state of "what was seen in the file" after
/// compression.
fn preserved_preview_recall(preserved: &str, max_lines: usize) -> Option<String> {
    let mut in_preview = false;
    let mut lines = Vec::new();
    for line in preserved.lines() {
        let trimmed = line.trim();
        if !in_preview {
            if trimmed.starts_with("Preview (") {
                in_preview = true;
            }
            continue;
        }
        if trimmed.is_empty()
            || (trimmed.starts_with("... [") && trimmed.contains("omitted"))
            || trimmed.starts_with("[[")
        {
            continue;
        }
        let normalized = truncate_to_chars(&normalize_whitespace(trimmed), 180);
        if normalized.is_empty() {
            continue;
        }
        lines.push(normalized);
        if lines.len() >= max_lines {
            break;
        }
    }

    (!lines.is_empty()).then(|| format!("preview: {}", lines.join(" | ")))
}

fn precision_grounding_tool_recall(
    tool_call: &ToolCall,
    preserved: &str,
    archive_path_line: Option<String>,
) -> String {
    let mut lines = Vec::new();
    if let Some(preview) = preserved_preview_recall(preserved, 12) {
        lines.push(preview);
    }
    lines.extend(collect_original_recall_lines(tool_call, preserved));
    if let Some(archive_path_line) = archive_path_line {
        lines.push(archive_path_line);
    }

    match tool_call.function.name.as_str() {
        "read_file" => lines.push(
            "优先依据 `original_file_path` / `original_range` 和 preview 继续判断；仅当这些锚点仍不足时再读取 `archive_file_path`。".to_string(),
        ),
        _ => {}
    }

    if lines.is_empty() {
        "高精度结果已归档；仅在确需完整原文时再读取 `archive_file_path`。".to_string()
    } else {
        lines.join("\n  ")
    }
}

/// When command output is folded, keep at least the exit status, key diagnostics,
/// and tail conclusions. The full log is still archived by the caller to
/// `file_path`; the job here is only to let the model decide the next step without
/// re-running the command.
fn command_result_recall(result_text: &str) -> String {
    const MAX_SIGNALS: usize = 5;
    const MAX_CHARS: usize = 720;

    let lines = result_text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return "command produced no output".to_string();
    }

    let mut signals = Vec::with_capacity(MAX_SIGNALS + 2);
    push_command_signal(&mut signals, lines[0]);
    for line in &lines {
        let lower = line.to_ascii_lowercase();
        let diagnostic = lower.contains("error")
            || lower.contains("failed")
            || lower.contains("panic")
            || lower.contains("test result:")
            || lower.contains("failures:")
            || lower.contains("could not compile")
            || lower.contains("aborting due to");
        if diagnostic {
            push_command_signal(&mut signals, line);
            if signals.len() >= MAX_SIGNALS {
                break;
            }
        }
    }
    if signals.len() < MAX_SIGNALS {
        push_command_signal(&mut signals, lines[lines.len() - 1]);
    }

    truncate_to_chars(&signals.join(" | "), MAX_CHARS)
}

fn push_command_signal(signals: &mut Vec<String>, line: &str) {
    let line = truncate_to_chars(&normalize_whitespace(line), 220);
    if !line.is_empty() && !signals.iter().any(|existing| existing == &line) {
        signals.push(line);
    }
}

/// Within a single turn: when the LLM-summary fallback runs, keep the verbatim
/// count of recent tool groups inside the tail window (the current user turn);
/// fold earlier same-turn tool groups into one-line stubs. This fixes compressor
/// idling when all the bulk piles up inside one user turn with no cross-turn
/// boundary to summarize.
pub(super) const MID_TURN_LLM_SUMMARY_KEEP_RECENT_TOOL_GROUPS: usize = 4;

/// Return the message indices of all tool results in the most recent
/// `keep_recent_groups` complete tool groups.
///
/// One assistant(tool_calls) may issue any number of parallel calls. The
/// protection window must be computed per atomic group, not cut by tool message
/// count; otherwise a batch could end up with some results still in context and
/// others already offloaded/deduped, forcing the model to re-read the missing
/// files.
pub(super) fn recent_tool_group_message_indices(
    messages: &[Message],
    keep_recent_groups: usize,
) -> FxHashSet<usize> {
    recent_tool_result_groups(messages, keep_recent_groups)
        .into_iter()
        .flatten()
        .collect()
}

/// Return the tool result indices of the most recent `keep_recent_groups` complete
/// tool groups, preserving group boundaries.
pub(super) fn recent_tool_result_groups(
    messages: &[Message],
    keep_recent_groups: usize,
) -> Vec<Vec<usize>> {
    if keep_recent_groups == 0 {
        return Vec::new();
    }

    let mut groups = Vec::<Vec<usize>>::new();
    for (anchor, assistant) in messages.iter().enumerate() {
        if assistant.role != "assistant" {
            continue;
        }
        let Some(calls) = assistant
            .tool_calls
            .as_ref()
            .filter(|calls| !calls.is_empty())
        else {
            continue;
        };
        let call_ids: FxHashSet<&str> = calls.iter().map(|call| call.id.as_str()).collect();
        let mut result_indices = Vec::new();
        for (idx, message) in messages.iter().enumerate().skip(anchor + 1) {
            if message.role == "assistant" && message.tool_calls.is_some() {
                break;
            }
            if message.role == "tool"
                && message
                    .tool_call_id
                    .as_deref()
                    .is_some_and(|id| call_ids.contains(id))
            {
                result_indices.push(idx);
            }
        }
        if !result_indices.is_empty() {
            groups.push(result_indices);
        }
    }

    groups.into_iter().rev().take(keep_recent_groups).collect()
}

/// Generate the single-line "recall anchor" for a folded stub: prefer extracting
/// the `file_path:` pointer from spilled tool results (the model can re-run
/// read_file from it), otherwise fall back to the first non-empty result line.
/// Guarantees that folding early precision tool groups still leaves a recallable
/// lead, avoiding amnesia-style repeated searches.
fn tool_result_recall_one_liner(result_text: &str) -> String {
    if let Some(original_line) = result_text
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("- original_"))
    {
        return truncate_to_chars(&normalize_whitespace(original_line), 220);
    }
    if let Some(path_line) = result_text
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("- archive_file_path:"))
    {
        return truncate_to_chars(&normalize_whitespace(path_line), 220);
    }
    if let Some(path_line) = result_text
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("- file_path:") || line.starts_with("file_path:"))
    {
        return truncate_to_chars(&normalize_whitespace(path_line), 200);
    }
    let first_meaningful = result_text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("");
    truncate_to_chars(&normalize_whitespace(first_meaningful), 160)
}

/// Extract `file_path` (or the compatible `path`) from a tool call's JSON
/// `arguments`. `apply_patch` may also write the target inside the `patch` body's
/// `*** Update File: <path>` / `*** Add File: <path>` / `*** Delete File: <path>` /
/// `*** Replace in line: <path>` envelopes; parse those as a fallback.
fn extract_file_path_args(arguments: &str) -> Vec<String> {
    let Ok(v) = serde_json::from_str::<Value>(arguments) else {
        return Vec::new();
    };
    if let Some(p) = v.get("file_path").and_then(|x| x.as_str()) {
        return vec![p.to_string()];
    }
    if let Some(p) = v.get("path").and_then(|x| x.as_str()) {
        return vec![p.to_string()];
    }
    if let Some(patch) = v.get("patch").and_then(|x| x.as_str()) {
        return patch
            .lines()
            .filter_map(|line| {
                let trimmed = line.trim_start();
                trimmed
                    .strip_prefix("*** Update File:")
                    .or_else(|| trimmed.strip_prefix("*** Add File:"))
                    .or_else(|| trimmed.strip_prefix("*** Delete File:"))
                    .or_else(|| trimmed.strip_prefix("*** Replace in line:"))
                    .map(|rest| rest.trim().to_string())
            })
            .collect();
    }
    Vec::new()
}

/// Scan the whole message stream for the set of target file paths whose most
/// recent `apply_patch` failed with no later success on the same path.
///
/// When the folder hits a tool group that reads these paths via `read_file`, it
/// skips folding so the model can still see the original file content to construct
/// exact patch context. Rule: record, in message order, the most recent
/// `apply_patch` result per path; a result content starting with `Successfully
/// patched` counts as success, anything else as failure. Only paths whose final
/// state is "failed" are kept — a later successful patch on the same path lifts
/// the protection automatically. Once the failed `apply_patch` call itself is
/// compressed out of history, the path disappears with it, so the protection scope
/// is naturally bounded (usually 1–3 files).
fn collect_pending_patch_paths(messages: &[Message]) -> FxHashSet<String> {
    // tool_call_id → tool result text
    let mut result_by_id: FxHashMap<String, String> = FxHashMap::default();
    for m in messages {
        if m.role == "tool" {
            if let Some(id) = m.tool_call_id.clone() {
                result_by_id.insert(id, value_to_string(&m.content));
            }
        }
    }
    // path → whether the most recent apply_patch succeeded
    let mut last_state: FxHashMap<String, bool> = FxHashMap::default();
    for m in messages {
        if m.role != "assistant" {
            continue;
        }
        let Some(tcs) = m.tool_calls.as_ref() else {
            continue;
        };
        for tc in tcs {
            if tc.function.name != "apply_patch" {
                continue;
            }
            let paths = extract_file_path_args(&tc.function.arguments);
            if paths.is_empty() {
                continue;
            }
            let result = result_by_id.get(&tc.id).map(String::as_str).unwrap_or("");
            let succeeded = result.trim_start().starts_with("Successfully patched");
            for path in paths {
                last_state.insert(path, succeeded);
            }
        }
    }
    last_state
        .into_iter()
        .filter(|(_, ok)| !*ok)
        .map(|(p, _)| p)
        .collect()
}

/// Whether this tool group contains a `read_file` call whose `file_path` is
/// exactly a not-yet-successful `apply_patch` target. If so, the folder should
/// skip folding to preserve the original file content the model needs to retry
/// the patch.
fn group_reads_pending_patch_target(
    messages: &[Message],
    group: &[usize],
    pending_paths: &FxHashSet<String>,
) -> bool {
    let Some(assistant) = messages.get(group[0]) else {
        return false;
    };
    let Some(tcs) = assistant.tool_calls.as_ref() else {
        return false;
    };
    tcs.iter().any(|tc| {
        tc.function.name == "read_file"
            && extract_file_path_args(&tc.function.arguments)
                .into_iter()
                .any(|p| pending_paths.contains(&p))
    })
}

/// Fold earlier `assistant(tool_calls) + accompanying tool` groups in the message
/// sequence into a single `internal_note` stub, keeping the most recent
/// `keep_recent_groups` groups verbatim plus all non-tool-group messages
/// (user / system / internal_note / plain-text assistant).
///
/// Key invariant: folding replaces the assistant and all of its tool responses
/// **as one whole group** with a single stub, so the pairing breakage of "keeping
/// assistant.tool_calls while dropping the accompanying tool responses" can never
/// happen (the OpenAI protocol requires the two to be paired). Groups containing
/// `is_non_compressible_tool` tools are folded too, but their `file_path:` recall
/// anchor is kept inside the stub.
///
/// Returns the purely in-memory [`ToolGroupFoldPlan`]; the caller must explicitly
/// `commit` after deciding to adopt the candidate.
pub(super) fn plan_early_tool_groups(
    messages: &[Message],
    keep_recent_groups: usize,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &FxHashSet<String>,
) -> ToolGroupFoldPlan {
    // Locate the start of every assistant(tool_calls) message as a tool group anchor.
    let group_anchors: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(idx, m)| {
            let has_calls = m
                .tool_calls
                .as_ref()
                .map(|calls| !calls.is_empty())
                .unwrap_or(false);
            (m.role == "assistant" && has_calls).then_some(idx)
        })
        .collect();
    if group_anchors.len() <= keep_recent_groups {
        return ToolGroupFoldPlan::unchanged(messages);
    }
    // Keep the most recent keep_recent_groups tool groups verbatim; fold earlier
    // ones. When keep_recent_groups=0, fold all tool groups (fold_before_anchor
    // takes the end of the messages) to avoid an out-of-bounds panic on
    // group_anchors[len - 0].
    let fold_before_anchor = if keep_recent_groups == 0 {
        messages.len()
    } else {
        group_anchors[group_anchors.len() - keep_recent_groups]
    };

    // Direction two: pending-patch targeted retention. Scan the history for target
    // file paths whose most recent apply_patch failed with no later success on the
    // same path; when the folder hits a read_file group reading those paths, it
    // skips folding so the model can still see the original file content to
    // construct exact patch context, avoiding the deadlock of "content offloaded →
    // patch context mismatch → re-read → judged no progress → hard stop". The
    // protection scope is naturally bounded: once the succeeded or failed
    // apply_patch call itself is compressed out of history, the path is released.
    let pending_patch_paths = collect_pending_patch_paths(messages);

    let mut out: Vec<Message> = Vec::with_capacity(messages.len());
    let mut folded_groups = 0usize;
    let mut archives = Vec::new();
    let mut idx = 0usize;
    while idx < messages.len() {
        let message = &messages[idx];
        let has_calls = message
            .tool_calls
            .as_ref()
            .map(|calls| !calls.is_empty())
            .unwrap_or(false);
        // Only assistant(tool_calls) groups inside the fold region (earlier than
        // the recent retention window) are folded.
        if idx < fold_before_anchor && message.role == "assistant" && has_calls {
            let tool_call_ids: FxHashSet<&str> = message
                .tool_calls
                .as_ref()
                .unwrap()
                .iter()
                .map(|tc| tc.id.as_str())
                .collect();
            if tool_call_ids
                .iter()
                .any(|id| protected_tool_call_ids.contains(*id))
            {
                out.push(message.clone());
                idx += 1;
                continue;
            }
            // Collect the consecutive tool responses right after it that belong to
            // this assistant, forming the complete group.
            let mut group = vec![idx];
            let mut cursor = idx + 1;
            while cursor < messages.len() && messages[cursor].role == "tool" {
                match messages[cursor].tool_call_id.as_deref() {
                    Some(id) if tool_call_ids.contains(id) => group.push(cursor),
                    _ => break,
                }
                cursor += 1;
            }
            // read_file groups targeting pending-patch paths skip folding and are
            // kept verbatim. This must be checked before building the stub so no
            // pointless archive plan is generated for a group that would never be
            // adopted anyway.
            if group_reads_pending_patch_target(messages, &group, &pending_patch_paths) {
                for &gi in &group {
                    out.push(messages[gi].clone());
                }
                idx = cursor;
                continue;
            }
            if let Some((stub, group_archives)) =
                plan_tool_call_group_fold(messages, &group, overflow_dir)
            {
                out.push(stub);
                archives.extend(group_archives);
                folded_groups += 1;
                idx = cursor;
                continue;
            }
        }
        out.push(message.clone());
        idx += 1;
    }
    ToolGroupFoldPlan {
        messages: out,
        folded_groups,
        archives,
    }
}

/// Number of `assistant(tool_calls)` group anchors, matching the collection in
/// [`plan_early_tool_groups`]. Lets ladder callers skip planning (and the
/// whole-history clone it performs) for windows that cannot fold anything.
pub(super) fn count_tool_group_anchors(messages: &[Message]) -> usize {
    messages
        .iter()
        .filter(|m| {
            let has_calls = m
                .tool_calls
                .as_ref()
                .map(|calls| !calls.is_empty())
                .unwrap_or(false);
            m.role == "assistant" && has_calls
        })
        .count()
}

/// Convenience entry point for non-speculative calls: commit immediately after a
/// successful plan; if archiving fails, keep the original messages and never
/// produce a folded stub pointing at nonexistent evidence.
pub(super) fn fold_early_tool_groups(
    messages: &[Message],
    keep_recent_groups: usize,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &FxHashSet<String>,
) -> (Vec<Message>, usize) {
    let plan = plan_early_tool_groups(
        messages,
        keep_recent_groups,
        overflow_dir,
        protected_tool_call_ids,
    );
    if !plan.commit() {
        return (messages.to_vec(), 0);
    }
    plan.into_result()
}
