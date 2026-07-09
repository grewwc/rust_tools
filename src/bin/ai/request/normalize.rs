//! 消息归一化与工具提示过滤。
//!
//! 在发送请求前对 `messages` 做最终清理：
//! - `agent_tools_for_request`：根据模型能力组装 tools/tool_choice 字段
//! - `strip_unavailable_tool_hints_from_messages`：移除引用未注册工具的提示行
//! - `normalize_messages_for_request`：系统消息合并、内部笔记注入、token 截断
//! - `normalize_message_content_for_text_only_model`：纯文本模型降级
//! - `normalize_messages_for_model`：模型特定的消息归一化

use serde_json::Value;

use crate::ai::history::{Message, ROLE_SYSTEM, is_internal_note_role, is_system_like_role};
use crate::ai::models;
use crate::ai::types::App;

#[allow(unused_imports)]
use super::error::config_bool_is_true;
use super::types::extract_displayable_text;
#[allow(unused_imports)]
use crate::ai::config_schema::AiConfig;
#[allow(unused_imports)]
use crate::commonw::configw;

pub(super) fn agent_tools_for_request(app: &App, model: &str) -> (Option<Value>, Option<Value>) {
    if !models::tools_enabled(model) {
        return (None, None);
    }
    let Some(ctx) = app.agent_context.as_ref() else {
        return (None, None);
    };
    if ctx.tools.is_empty() {
        return (None, None);
    }
    let tools_value = serde_json::to_value(&ctx.tools).ok();
    let tool_choice = tools_value
        .as_ref()
        .map(|_| Value::String("auto".to_string()));
    (tools_value, tool_choice)
}

pub(super) fn request_tool_names_for_model(
    app: &App,
    model: &str,
) -> rust_tools::commonw::FastSet<String> {
    if !models::tools_enabled(model) {
        return Default::default();
    }
    app.agent_context
        .as_ref()
        .map(|ctx| {
            ctx.tools
                .iter()
                .map(|tool| tool.function.name.clone())
                .collect()
        })
        .unwrap_or_default()
}

pub(super) fn line_references_unavailable_registered_tool(
    line: &str,
    available_tool_names: &rust_tools::commonw::FastSet<String>,
) -> bool {
    line.split('`')
        .enumerate()
        .filter_map(|(idx, chunk)| {
            if idx % 2 == 0 {
                return None;
            }
            let candidate = chunk
                .trim()
                .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'))
                .next()
                .unwrap_or("");
            if candidate.is_empty() {
                return None;
            }
            crate::ai::tools::registry::common::is_registered_tool_name(candidate)
                .then_some(candidate)
        })
        .any(|tool_name| !available_tool_names.contains(tool_name))
}

pub(super) fn line_can_contain_unavailable_tool_hint(message: &Message, line: &str) -> bool {
    let trimmed = line.trim_start();
    if !trimmed.contains('`') {
        return false;
    }

    if message.role == ROLE_SYSTEM || is_internal_note_role(&message.role) {
        return trimmed.starts_with("- ") || trimmed.starts_with("Code-navigation correction:");
    }

    if message.role == "tool" {
        return trimmed.starts_with("Suggestion:");
    }

    false
}

pub(super) fn message_may_need_unavailable_tool_hint_filter(message: &Message, text: &str) -> bool {
    if !text.contains('`') {
        return false;
    }

    if message.role == ROLE_SYSTEM || is_internal_note_role(&message.role) {
        return text.contains("- ") || text.contains("Code-navigation correction:");
    }

    if message.role == "tool" {
        return text.contains("Suggestion:");
    }

    false
}

pub(super) fn should_drop_unavailable_tool_hint_line(
    message: &Message,
    line: &str,
    available_tool_names: &rust_tools::commonw::FastSet<String>,
) -> bool {
    if !line_can_contain_unavailable_tool_hint(message, line) {
        return false;
    }

    if !line_references_unavailable_registered_tool(line, available_tool_names) {
        return false;
    }

    let trimmed = line.trim_start();
    if message.role == ROLE_SYSTEM || is_internal_note_role(&message.role) {
        return trimmed.starts_with("- ") || trimmed.starts_with("Code-navigation correction:");
    }
    if message.role == "tool" {
        return trimmed.starts_with("Suggestion:");
    }
    false
}

pub(super) fn strip_unavailable_tool_hints_from_messages(
    messages: &mut [Message],
    available_tool_names: &rust_tools::commonw::FastSet<String>,
) {
    for message in messages.iter_mut() {
        let Some(text) = message.content.as_str() else {
            continue;
        };
        if !message_may_need_unavailable_tool_hint_filter(message, text) {
            continue;
        }

        let mut filtered = String::new();
        let mut removed_any = false;
        let mut prefix_len = 0usize;
        for raw_line in text.split_inclusive('\n') {
            let line = raw_line.strip_suffix('\n').unwrap_or(raw_line);
            if should_drop_unavailable_tool_hint_line(message, line, available_tool_names) {
                if !removed_any {
                    filtered = String::with_capacity(text.len());
                    filtered.push_str(&text[..prefix_len]);
                    removed_any = true;
                }
            } else if removed_any {
                filtered.push_str(raw_line);
            }
            prefix_len += raw_line.len();
        }

        if removed_any {
            message.content = Value::String(filtered);
        }
    }
}

pub(super) fn normalize_messages_for_request(messages: &[Message]) -> Vec<Message> {
    const MERGED_NOTES_MAX_CHARS: usize = 4_000;
    const MERGED_SINGLE_NOTE_MAX_CHARS: usize = 1_200;

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum InternalNoteKind {
        WorkingMemory,
        CodeDiscovery,
        CachedTools,
        Summary,
        SelfNote,
        Generic,
    }

    fn truncate_chars(s: &str, max_chars: usize) -> String {
        if s.chars().count() <= max_chars {
            return s.to_string();
        }
        let mut out = String::new();
        for ch in s.chars().take(max_chars.saturating_sub(1)) {
            out.push(ch);
        }
        out.push('…');
        out
    }

    fn truncate_note_text(text: &str, max_chars: usize) -> String {
        if text.chars().count() <= max_chars {
            return text.to_string();
        }

        let lines = text
            .lines()
            .map(str::trim_end)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>();
        if lines.is_empty() {
            return truncate_chars(text, max_chars);
        }

        // 结构化裁剪：优先保留头部条目 + 尾部少量最新条目，避免硬切半句。
        let mut selected = Vec::new();
        let mut used = 0usize;
        let head_budget = max_chars.saturating_mul(2).saturating_div(3);
        for line in lines.iter().take(24) {
            let line_chars = line.chars().count();
            let extra = if selected.is_empty() { 0 } else { 1 };
            if used + extra + line_chars > head_budget {
                break;
            }
            used += extra + line_chars;
            selected.push((*line).to_string());
        }

        let mut tail = Vec::new();
        for line in lines.iter().rev().take(6).rev() {
            if selected.iter().any(|existing| existing == line) {
                continue;
            }
            tail.push((*line).to_string());
        }

        let omitted = text.chars().count().saturating_sub(
            selected
                .iter()
                .map(|line| line.chars().count())
                .sum::<usize>(),
        );
        if !tail.is_empty() {
            selected.push(format!("... [truncated: {omitted} chars omitted]"));
            selected.extend(tail);
        }
        truncate_chars(&selected.join("\n"), max_chars)
    }

    fn detect_note_kind(text: &str) -> InternalNoteKind {
        let trimmed = text.trim();
        if trimmed.starts_with("Current code-inspection working memory:") {
            return InternalNoteKind::WorkingMemory;
        }
        if trimmed.starts_with("code_discovery:") {
            return InternalNoteKind::CodeDiscovery;
        }
        if trimmed.starts_with("Context note: reused cached tool results") {
            return InternalNoteKind::CachedTools;
        }
        if trimmed.starts_with("对话摘要（自动压缩")
            || trimmed.starts_with("历史摘要（自动压缩")
            || trimmed.starts_with("[mid-turn-summary]")
        {
            return InternalNoteKind::Summary;
        }
        if trimmed.starts_with("self_note:") {
            return InternalNoteKind::SelfNote;
        }
        InternalNoteKind::Generic
    }

    fn note_heading(kind: InternalNoteKind) -> &'static str {
        match kind {
            InternalNoteKind::WorkingMemory => "Working Memory",
            InternalNoteKind::CodeDiscovery => "Code Discoveries",
            InternalNoteKind::CachedTools => "Cached Tool Results",
            InternalNoteKind::Summary => "History Summary",
            InternalNoteKind::SelfNote => "Self Notes",
            InternalNoteKind::Generic => "Additional Notes",
        }
    }

    fn content_is_effectively_empty(content: &Value) -> bool {
        match content {
            Value::Null => true,
            Value::String(s) => s.trim().is_empty(),
            Value::Array(items) => items.is_empty(),
            Value::Object(map) => map.is_empty(),
            Value::Bool(_) | Value::Number(_) => false,
        }
    }

    fn sanitize_tool_call_for_request(
        tool_call: &crate::ai::types::ToolCall,
    ) -> Option<crate::ai::types::ToolCall> {
        let raw_args = tool_call.function.arguments.trim();
        let normalized_arguments = if raw_args.is_empty() {
            "{}".to_string()
        } else {
            serde_json::from_str::<Value>(raw_args).ok()?;
            raw_args.to_string()
        };

        let mut sanitized = tool_call.clone();
        sanitized.function.arguments = normalized_arguments;
        // 确保 tool_type 不为空（部分 provider 在 stream 中不返回 type）
        if sanitized.tool_type.is_empty() {
            sanitized.tool_type = "function".to_string();
        }
        Some(sanitized)
    }

    fn sanitize_tool_message_sequence(messages: Vec<Message>) -> Vec<Message> {
        fn build_unpaired_tool_evidence_note(
            reason: &str,
            tool_messages: &[Message],
        ) -> Option<Message> {
            if tool_messages.is_empty() {
                return None;
            }
            let mut lines = vec![
                "Context note: preserved unmatched tool outputs from prior rounds.".to_string(),
                format!("reason: {reason}"),
            ];
            for message in tool_messages.iter().take(8) {
                let tool_call_id = message
                    .tool_call_id
                    .as_deref()
                    .filter(|id| !id.trim().is_empty())
                    .unwrap_or("unknown");
                let text = message
                    .content
                    .as_str()
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .unwrap_or_default();
                if text.is_empty() {
                    continue;
                }
                let preview = if text.chars().count() > 240 {
                    let mut p = text.chars().take(239).collect::<String>();
                    p.push('…');
                    p
                } else {
                    text.to_string()
                };
                lines.push(format!("- tool_call_id={tool_call_id}: {preview}"));
            }
            if lines.len() <= 2 {
                return None;
            }
            Some(Message {
                role: "internal_note".to_string(),
                content: Value::String(lines.join("\n")),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            })
        }

        let mut out = Vec::with_capacity(messages.len());
        let mut idx = 0usize;

        while idx < messages.len() {
            let message = &messages[idx];
            if message.role == "tool" {
                idx += 1;
                continue;
            }

            let Some(tool_calls) = message
                .tool_calls
                .as_ref()
                .filter(|calls| !calls.is_empty())
            else {
                out.push(message.clone());
                idx += 1;
                continue;
            };

            let sanitized_tool_calls = tool_calls
                .iter()
                .filter_map(sanitize_tool_call_for_request)
                .collect::<Vec<_>>();
            let mut scan = idx + 1;

            if sanitized_tool_calls.is_empty() {
                let mut raw_tool_messages = Vec::new();
                while scan < messages.len() && messages[scan].role == "tool" {
                    raw_tool_messages.push(messages[scan].clone());
                    scan += 1;
                }
                let mut assistant_only = message.clone();
                assistant_only.tool_calls = None;
                if !content_is_effectively_empty(&assistant_only.content) {
                    out.push(assistant_only);
                }
                if let Some(note) = build_unpaired_tool_evidence_note(
                    "tool_calls dropped because arguments failed request sanitization",
                    &raw_tool_messages,
                ) {
                    out.push(note);
                }
                idx = scan.max(idx + 1);
                continue;
            }

            let expected_ids = sanitized_tool_calls
                .iter()
                .map(|tool_call| tool_call.id.as_str())
                .collect::<Vec<_>>();
            let mut matched_ids = Vec::new();
            let mut matched_tool_messages = Vec::new();
            let mut unmatched_tool_messages = Vec::new();

            while scan < messages.len() && messages[scan].role == "tool" {
                let tool_message = &messages[scan];
                if let Some(tool_call_id) = tool_message.tool_call_id.as_deref()
                    && expected_ids
                        .iter()
                        .any(|expected| *expected == tool_call_id)
                    && !matched_ids.iter().any(|seen| seen == tool_call_id)
                {
                    matched_ids.push(tool_call_id.to_string());
                    matched_tool_messages.push(tool_message.clone());
                } else {
                    unmatched_tool_messages.push(tool_message.clone());
                }
                scan += 1;
            }

            if matched_ids.is_empty() {
                let mut assistant_only = message.clone();
                assistant_only.tool_calls = None;
                if !content_is_effectively_empty(&assistant_only.content) {
                    out.push(assistant_only);
                }
                if let Some(note) = build_unpaired_tool_evidence_note(
                    "tool_call ids could not be matched with sanitized assistant tool_calls",
                    &unmatched_tool_messages,
                ) {
                    out.push(note);
                }
                idx = scan.max(idx + 1);
                continue;
            }

            let mut assistant_with_matched_calls = message.clone();
            assistant_with_matched_calls.tool_calls = Some(
                sanitized_tool_calls
                    .iter()
                    .filter(|tool_call| matched_ids.iter().any(|id| id == &tool_call.id))
                    .cloned()
                    .collect(),
            );
            out.push(assistant_with_matched_calls);
            out.extend(matched_tool_messages);
            if let Some(note) = build_unpaired_tool_evidence_note(
                "some tool outputs were unmatched and preserved as context note",
                &unmatched_tool_messages,
            ) {
                out.push(note);
            }
            idx = scan;
        }

        out
    }

    let first_system_idx = messages
        .iter()
        .position(|m| m.role == ROLE_SYSTEM || is_internal_note_role(&m.role));
    let Some(first_system_idx) = first_system_idx else {
        return sanitize_tool_message_sequence(messages.to_vec());
    };

    // Only merge system-like notes that sit BEFORE the first conversational
    // (user/assistant/tool) message — those are produced by history
    // compression at the very top and stay stable across turns. Notes that
    // arrive later (working memory / code discovery / self_note / cached
    // tool results, etc.) are kept in their original positions with role
    // rewritten to "system", so growing tail notes only invalidate the
    // suffix of the provider's prompt cache instead of the whole request.
    let first_body_idx = messages
        .iter()
        .position(|m| !is_system_like_role(&m.role))
        .unwrap_or(messages.len());

    let mut merged_notes: Vec<(usize, InternalNoteKind, String)> = Vec::new();
    for (idx, message) in messages.iter().enumerate().take(first_body_idx) {
        if idx == first_system_idx {
            continue;
        }
        let text = message
            .content
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or_default();
        if !text.is_empty() {
            merged_notes.push((
                idx,
                detect_note_kind(text),
                truncate_note_text(text, MERGED_SINGLE_NOTE_MAX_CHARS),
            ));
        }
    }
    merged_notes.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    let mut merged_first = messages[first_system_idx].clone();
    merged_first.role = ROLE_SYSTEM.to_string();
    if let Some(base) = merged_first.content.as_str() {
        let mut content = base.to_string();
        if !merged_notes.is_empty() {
            content.push_str("\n\n[Merged system notes from history/runtime]\n");
            let mut grouped = Vec::new();
            let mut current_kind: Option<InternalNoteKind> = None;
            for (_, kind, text) in merged_notes {
                if current_kind != Some(kind) {
                    grouped.push(format!("## {}", note_heading(kind)));
                    current_kind = Some(kind);
                }
                grouped.push(text);
            }
            let merged_blob = truncate_chars(&grouped.join("\n\n"), MERGED_NOTES_MAX_CHARS);
            content.push_str(&merged_blob);
        }
        merged_first.content = Value::String(content);
    }

    let mut out = Vec::with_capacity(messages.len());
    out.push(merged_first);
    for (idx, message) in messages.iter().enumerate() {
        if idx == first_system_idx {
            continue;
        }
        if idx < first_body_idx {
            // Already folded into merged_first above.
            continue;
        }
        if is_internal_note_role(&message.role) {
            // Keep the note in-place but normalize the role so the API
            // accepts it. Stable position preserves prompt-cache prefix:
            // older notes don't move when newer ones get appended.
            let mut promoted = message.clone();
            promoted.role = ROLE_SYSTEM.to_string();
            // Cap mid-stream notes to a reasonable budget to avoid bloating
            // the request with stale long notes (working memory / code
            // discoveries / self_note / cached-tool notes accumulated over
            // many tool rounds).
            if let Value::String(text) = &promoted.content {
                if text.chars().count() > MERGED_SINGLE_NOTE_MAX_CHARS {
                    promoted.content =
                        Value::String(truncate_note_text(text, MERGED_SINGLE_NOTE_MAX_CHARS));
                }
            }
            out.push(promoted);
            continue;
        }
        out.push(message.clone());
    }
    sanitize_tool_message_sequence(out)
}

pub(super) fn normalize_message_content_for_text_only_model(content: &Value) -> Value {
    const IMAGE_PLACEHOLDER: &str = "[image omitted]";

    match content {
        Value::Array(items) => {
            let mut segments = Vec::new();
            for item in items {
                if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                    if !text.trim().is_empty() {
                        segments.push(text.to_string());
                    }
                    continue;
                }

                if item.get("image_url").is_some()
                    || item.get("type").and_then(|v| v.as_str()) == Some("image_url")
                {
                    segments.push(IMAGE_PLACEHOLDER.to_string());
                    continue;
                }

                let fallback = extract_displayable_text(item);
                if !fallback.trim().is_empty() {
                    segments.push(fallback);
                }
            }

            if segments.is_empty() {
                Value::String(IMAGE_PLACEHOLDER.to_string())
            } else {
                Value::String(segments.join("\n"))
            }
        }
        _ => content.clone(),
    }
}

pub(super) fn normalize_messages_for_model(model: &str, messages: &[Message]) -> Vec<Message> {
    let mut normalized = normalize_messages_for_request(messages);
    if models::supports_image_input(model) {
        return normalized;
    }

    for message in &mut normalized {
        message.content = normalize_message_content_for_text_only_model(&message.content);
    }
    normalized
}
