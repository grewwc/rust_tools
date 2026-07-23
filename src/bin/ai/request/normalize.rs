//! 消息归一化与工具提示过滤。
//!
//! 在发送请求前对 `messages` 做最终清理：
//! - `agent_tools_for_request`：根据模型能力组装 tools/tool_choice 字段
//! - `strip_unavailable_tool_hints_from_messages`：移除引用未注册工具的提示行
//! - `normalize_messages_for_request`：系统消息合并、内部笔记注入、token 截断
//! - `normalize_message_content_for_text_only_model`：纯文本模型降级
//! - `normalize_messages_for_model`：模型特定的消息归一化

use rustc_hash::FxHashSet;
use serde_json::Value;

use crate::ai::history::{
    Message, ROLE_SYSTEM, is_internal_note_role, is_summary_note_text, is_system_like_role,
};
use crate::ai::models;
use crate::ai::types::App;

const IMAGE_PLACEHOLDER: &str = "[image omitted]";

fn flatten_multimodal_content_to_text(content: &Value) -> Option<String> {
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

            Some(if segments.is_empty() {
                IMAGE_PLACEHOLDER.to_string()
            } else {
                segments.join("\n")
            })
        }
        _ => None,
    }
}

fn normalize_system_like_content_for_request(content: &Value) -> Value {
    flatten_multimodal_content_to_text(content)
        .map(Value::String)
        .unwrap_or_else(|| content.clone())
}

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
        return trimmed.starts_with("- ");
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
        return text.contains("- ");
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
        return trimmed.starts_with("- ");
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

const MERGED_NOTES_MAX_CHARS: usize = 4_000;
const MERGED_SINGLE_NOTE_MAX_CHARS: usize = 1_200;
/// 持久化 checkpoint 全部保存在 history 中；请求只投影最近有限条，避免 marker
/// 本身随着长期会话无限增长。
const REQUEST_CONTEXT_CHECKPOINT_LIMIT: usize = 8;

/// checkpoint marker 是压缩后重新定位归档正文的短索引，写入端**始终**是
/// `role=internal_note`（见 `history::compress::is_context_checkpoint_marker`）。
/// 请求归一化会把 marker 收集后拼进一条独立 `system` 消息注入模型，因此这里
/// **必须同时校验 role**：若仅按 content 前缀识别，任何 user/tool/assistant
/// 正文只要以 `[context_checkpoint ` 开头就会被提权成 system 指令并注入——
/// 一条跨信任边界的 prompt 注入通道。合法 marker（internal_note）行为不变。
fn is_context_checkpoint_marker(message: &Message) -> bool {
    is_internal_note_role(&message.role)
        && message
            .content
            .as_str()
            .is_some_and(|text| text.trim_start().starts_with("[context_checkpoint "))
}

fn context_checkpoint_marker_key(marker: &str) -> String {
    const PREFIX: &str = "[context_checkpoint path=";
    let trimmed = marker.trim_start();
    let path = trimmed
        .strip_prefix(PREFIX)
        .and_then(|rest| rest.split(']').next())
        .map(str::trim)
        .filter(|path| !path.is_empty());
    path.unwrap_or(trimmed).to_string()
}

fn dedupe_context_checkpoint_markers_by_path(markers: Vec<String>) -> Vec<String> {
    let mut seen = FxHashSet::default();
    let mut deduped_newest_first = Vec::with_capacity(markers.len());
    for marker in markers.into_iter().rev() {
        if seen.insert(context_checkpoint_marker_key(&marker)) {
            deduped_newest_first.push(marker);
        }
    }
    deduped_newest_first.reverse();
    deduped_newest_first
}

pub(super) fn normalize_messages_for_request(messages: &[Message]) -> Vec<Message> {
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum InternalNoteKind {
        WorkingMemory,
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
        if trimmed.starts_with("Context note: reused cached tool results") {
            return InternalNoteKind::CachedTools;
        }
        if is_summary_note_text(trimmed) {
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
    ) -> crate::ai::types::ToolCall {
        let raw_args = tool_call.function.arguments.trim();
        // args 必须是合法 JSON 对象字符串才能过 provider 校验。但绝不能因为
        // args 坏了就丢弃整个 tool_call：该工具此时早已执行完并产生了真实结果，
        // 丢 tool_call 会破坏 assistant/tool 配对，导致真实 tool 结果被降级成
        // 预览 note（read_file/execute_command 结果被弱化 → 模型误判
        // 要重跑同一工具的根因之一）。坏 JSON 时修复成合法对象并原样保留原始
        // 文本，既满足 provider 校验又保住配对与结果。
        let normalized_arguments = if raw_args.is_empty() {
            "{}".to_string()
        } else if serde_json::from_str::<Value>(raw_args).is_ok() {
            raw_args.to_string()
        } else {
            serde_json::json!({ "_malformed_arguments": raw_args }).to_string()
        };

        let mut sanitized = tool_call.clone();
        sanitized.function.arguments = normalized_arguments;
        // 确保 tool_type 不为空（部分 provider 在 stream 中不返回 type）
        if sanitized.tool_type.is_empty() {
            sanitized.tool_type = "function".to_string();
        }
        sanitized
    }

    fn has_valid_function_name(tool_call: &crate::ai::types::ToolCall) -> bool {
        // `/v1/responses` 会严格校验 replay 的 function_call.name。历史记录可能来自
        // 中断的流或旧版本，不能让路径等参数值误写为函数名后导致整个请求 400。
        !tool_call.function.name.is_empty()
            && tool_call
                .function
                .name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
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
            // 这条 note 在 normalize_messages_for_request 末尾的
            // sanitize_tool_message_sequence 内部生成，此时所有 internal_note
            // 都已被改写为 system。若这里再用 internal_note，它会绕过 role
            // 归一化直接进入请求，触发 provider 400（'internal_note' 不是合法
            // role）。直接用 system：本就是给模型的 context note，语义上属于
            // system 提示，且 mid-stream system 消息是本项目已有合法模式。
            Some(Message {
                role: ROLE_SYSTEM.to_string(),
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
                .filter(|tool_call| has_valid_function_name(tool_call))
                .map(sanitize_tool_call_for_request)
                .collect::<Vec<_>>();
            let mut scan = idx + 1;

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
    // arrive later (working memory / self_note / cached tool results, etc.)
    // are kept in their original positions with role
    // rewritten to "system", so growing tail notes only invalidate the
    // suffix of the provider's prompt cache instead of the whole request.
    let first_body_idx = messages
        .iter()
        .position(|m| !is_system_like_role(&m.role))
        .unwrap_or(messages.len());

    let checkpoint_markers = messages
        .iter()
        .filter(|message| is_context_checkpoint_marker(message))
        .filter_map(|message| message.content.as_str().map(|text| text.trim().to_string()))
        .collect::<Vec<_>>();

    let mut merged_notes: Vec<(usize, InternalNoteKind, String)> = Vec::new();
    for (idx, message) in messages.iter().enumerate().take(first_body_idx) {
        let text = message
            .content
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or_default();
        if is_context_checkpoint_marker(message) {
            continue;
        }
        if idx == first_system_idx {
            continue;
        }
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
    merged_first.content = normalize_system_like_content_for_request(&merged_first.content);
    // 用**原始**消息（role 尚未被改写为 system）判定 checkpoint marker：merged_first
    // 的 role 已被上一行改成 ROLE_SYSTEM，若拿它判定会因 role 校验失败而漏掉合法
    // marker 的清空处理。
    if is_context_checkpoint_marker(&messages[first_system_idx]) {
        merged_first.content = Value::String(String::new());
    }
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

    let checkpoint_markers = dedupe_context_checkpoint_markers_by_path(checkpoint_markers)
        .into_iter()
        .rev()
        .take(REQUEST_CONTEXT_CHECKPOINT_LIMIT)
        .collect::<Vec<_>>();

    let mut out = Vec::with_capacity(messages.len() + usize::from(!checkpoint_markers.is_empty()));
    out.push(merged_first);
    if !checkpoint_markers.is_empty() {
        let mut checkpoint_context = String::from(
            "[Persistent context checkpoints: durable intermediate results you saved earlier so they survive compression. Each entry is a one-line summary followed by the file path holding the full details; read_file that path when you need the complete conclusion or evidence instead of rediscovering it.]\n",
        );
        for marker in checkpoint_markers.iter().rev() {
            checkpoint_context.push_str(marker);
            checkpoint_context.push('\n');
        }
        out.push(Message {
            role: ROLE_SYSTEM.to_string(),
            content: Value::String(checkpoint_context),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }
    for (idx, message) in messages.iter().enumerate() {
        if idx == first_system_idx {
            continue;
        }
        if is_context_checkpoint_marker(message) {
            // 所有 checkpoint 都集中投影到请求前缀并限量，不能让历史尾部的 marker
            // 随会话增长；持久化 history 保留完整记录，后续请求只带最近若干条。
            continue;
        }
        if idx < first_body_idx {
            // 普通 note 已折叠进 merged_first；checkpoint 已作为独立、无截断的
            // system 消息投影，避免被普通 note 的字符预算截断。
            continue;
        }
        if is_system_like_role(&message.role) {
            // Keep the note in-place but normalize the role so the API
            // accepts it. Stable position preserves prompt-cache prefix:
            // older notes don't move when newer ones get appended.
            let mut promoted = message.clone();
            promoted.role = ROLE_SYSTEM.to_string();
            // Cap mid-stream notes to a reasonable budget to avoid bloating
            // the request with stale long notes (working memory / self_note /
            // cached-tool notes accumulated over many tool rounds).
            if let Value::String(text) = &promoted.content {
                if text.chars().count() > MERGED_SINGLE_NOTE_MAX_CHARS {
                    promoted.content =
                        Value::String(truncate_note_text(text, MERGED_SINGLE_NOTE_MAX_CHARS));
                }
            }
            promoted.content = normalize_system_like_content_for_request(&promoted.content);
            out.push(promoted);
            continue;
        }
        out.push(message.clone());
    }
    sanitize_tool_message_sequence(out)
}

pub(super) fn normalize_message_content_for_text_only_model(content: &Value) -> Value {
    flatten_multimodal_content_to_text(content)
        .map(Value::String)
        .unwrap_or_else(|| content.clone())
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

/// 仅修改即将发送给模型的消息投影：如果一个失败调用之后出现了执行签名完全一致
/// 的成功调用，则用短标记替换旧失败正文。签名在执行时构造，包含 session、有效
/// cwd、路由、精确工具名和 canonical JSON 参数；这里不扫描自然语言猜测成败。
pub(crate) fn fold_resolved_tool_failures(
    messages: &mut [Message],
    outcomes: &[crate::ai::history::ToolExecutionOutcome],
) {
    use rustc_hash::FxHashMap;

    // 老历史可能包含 provider/fallback 跨轮复用的 ID，而旧版旁路表只保留了
    // 最后一条 outcome。对任何歧义 ID 安全退化为保留全文，绝不猜测 occurrence。
    let mut message_occurrences: FxHashMap<&str, usize> = FxHashMap::default();
    for message in messages.iter() {
        if message.role == "tool"
            && let Some(tool_call_id) = message.tool_call_id.as_deref()
        {
            *message_occurrences.entry(tool_call_id).or_default() += 1;
        }
    }
    let outcome_by_id = outcomes
        .iter()
        .filter(|outcome| message_occurrences.get(outcome.tool_call_id.as_str()) == Some(&1))
        .map(|outcome| (outcome.tool_call_id.as_str(), outcome))
        .collect::<FxHashMap<_, _>>();
    let mut later_success_by_signature: FxHashMap<&str, &str> = FxHashMap::default();

    for message in messages.iter_mut().rev() {
        if message.role != "tool" {
            continue;
        }
        let Some(tool_call_id) = message.tool_call_id.as_deref() else {
            continue;
        };
        let Some(outcome) = outcome_by_id.get(tool_call_id).copied() else {
            // 老历史没有结构化状态时保留全文；绝不回退到字符串启发式。
            continue;
        };
        if outcome.succeeded {
            later_success_by_signature
                .entry(outcome.execution_signature.as_str())
                .or_insert(tool_call_id);
            continue;
        }
        let Some(successful_tool_call_id) = later_success_by_signature
            .get(outcome.execution_signature.as_str())
            .copied()
        else {
            continue;
        };
        message.content = Value::String(format!(
            "[resolved tool failure: a later invocation with the identical tool, arguments, and execution environment succeeded; failed_tool_call_id={tool_call_id}; successful_tool_call_id={successful_tool_call_id}; original diagnostics omitted]"
        ));
    }
}

#[cfg(test)]
mod resolved_tool_failure_tests {
    use super::*;
    use crate::ai::history::ToolExecutionOutcome;

    fn tool_message(id: &str, content: &str) -> Message {
        Message {
            role: "tool".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: Some(id.to_string()),
            reasoning_content: None,
        }
    }

    fn outcome(id: &str, signature: &str, succeeded: bool) -> ToolExecutionOutcome {
        ToolExecutionOutcome {
            tool_call_id: id.to_string(),
            execution_signature: signature.to_string(),
            succeeded,
        }
    }

    #[test]
    fn resolved_tool_failures_fold_only_before_identical_success() {
        let original = vec![
            tool_message("fail-1", "Error: first diagnostic"),
            tool_message("fail-2", "Error: second diagnostic"),
            tool_message("success", "complete result"),
            tool_message("new-fail", "Error: current failure"),
            tool_message("other-fail", "Error: different environment"),
        ];
        let mut projected = original.clone();
        let outcomes = vec![
            outcome("fail-1", "same-signature", false),
            outcome("fail-2", "same-signature", false),
            outcome("success", "same-signature", true),
            outcome("new-fail", "same-signature", false),
            outcome("other-fail", "different-signature", false),
        ];

        fold_resolved_tool_failures(&mut projected, &outcomes);

        assert!(projected[0].content.as_str().unwrap().starts_with("[resolved tool failure:"));
        assert!(projected[1].content.as_str().unwrap().starts_with("[resolved tool failure:"));
        assert_eq!(projected[2].content, original[2].content);
        assert_eq!(projected[3].content, original[3].content);
        assert_eq!(projected[4].content, original[4].content);
        assert_eq!(original[0].content, Value::String("Error: first diagnostic".to_string()));
        assert_eq!(projected[0].role, "tool");
        assert_eq!(projected[0].tool_call_id.as_deref(), Some("fail-1"));
    }

    #[test]
    fn reused_legacy_tool_call_id_is_never_folded_ambiguously() {
        let original = vec![
            tool_message("reused", "successful result from an older occurrence"),
            tool_message("reused", "Error: newer occurrence failed"),
            tool_message("success", "later success"),
        ];
        let mut projected = original.clone();
        let outcomes = vec![
            outcome("reused", "same-signature", false),
            outcome("success", "same-signature", true),
        ];

        fold_resolved_tool_failures(&mut projected, &outcomes);

        assert_eq!(projected, original);
    }
}
