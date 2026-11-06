//! Anthropic Messages request encoding.

use serde_json::{Value, json};

use super::super::reasoning::{
    is_code_inspection_memory_message, is_nonempty_text_block, mark_content_cache_breakpoint,
};
use super::super::{RequestBody, types::extract_displayable_text};
use crate::ai::history::Message;

/// Anthropic Messages API body: a top-level `system` string, a `messages` array
/// of content-block messages, `tools` as `{name, description, input_schema}` and
/// a required `max_tokens`. History roles map onto Anthropic user/assistant
/// blocks; OpenAI-shaped tool calls become `tool_use` blocks and tool results
/// become `tool_result` blocks inside a user message. Extended thinking is
/// merged in from `request.thinking` (produced by the thinking dialect) and the
/// assistant thinking chain is replayed as `thinking` blocks, so vision and
/// thinking stay available on this dialect instead of being silently dropped.
pub(super) fn build_messages_request_body(request: &RequestBody<'_>) -> Value {
    let max_tokens = request.max_tokens.unwrap_or(4096);
    let mut body = serde_json::Map::new();
    body.insert("model".to_string(), Value::String(request.model.clone()));
    body.insert("stream".to_string(), Value::Bool(request.stream));
    body.insert("max_tokens".to_string(), json!(max_tokens));

    let mut system_parts: Vec<String> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    for message in request.messages {
        match message.role.as_str() {
            "system" => {
                let text = extract_displayable_text(&message.content);
                if !text.is_empty() {
                    system_parts.push(text);
                }
            }
            "user" => messages.push(json!({
                "role": "user",
                "content": messages_user_content(message),
            })),
            "assistant" => messages.push(messages_assistant_message(message)),
            "tool" => {
                let mut tool_result = serde_json::Map::new();
                tool_result.insert("type".to_string(), json!("tool_result"));
                tool_result.insert(
                    "tool_use_id".to_string(),
                    json!(message.tool_call_id.clone().unwrap_or_default()),
                );
                tool_result.insert(
                    "content".to_string(),
                    messages_content_blocks(&message.content),
                );
                // Signal failed tool executions so the model treats them as
                // errors to correct, not as ordinary text output. The flag is a
                // turn-scoped side channel (see `RequestBody.tool_errors`),
                // keyed by tool_call_id.
                let tool_use_id = message.tool_call_id.clone().unwrap_or_default();
                let failed = request
                    .tool_errors
                    .and_then(|errors| errors.get(&tool_use_id))
                    .copied()
                    .unwrap_or(false);
                if failed {
                    tool_result.insert("is_error".to_string(), json!(true));
                }
                messages.push(json!({
                    "role": "user",
                    "content": [Value::Object(tool_result)],
                }));
            }
            // Runtime-only roles (internal_note, ...) never reach the wire.
            _ => {}
        }
    }
    if !system_parts.is_empty() {
        let system = system_parts.join("\n\n");
        // Anthropic caches in tools → system → messages order, so this
        // breakpoint covers the tool schemas as well.
        let system = if request.explicit_cache_breakpoints {
            json!([{ "type": "text", "text": system, "cache_control": { "type": "ephemeral" } }])
        } else {
            Value::String(system)
        };
        body.insert("system".to_string(), system);
    }
    if request.explicit_cache_breakpoints {
        mark_messages_cache_breakpoint(&mut messages);
    }
    body.insert("messages".to_string(), Value::Array(messages));

    if let Some(tools) = messages_tools(request.tools.as_ref()) {
        body.insert("tools".to_string(), tools);
    }

    // Merge the resolved thinking controls (Anthropic extended thinking, set
    // by the thinking dialect) into the body.
    for (key, value) in &request.thinking {
        body.insert(key.clone(), value.clone());
    }

    Value::Object(body)
}

/// Rolling Anthropic cache breakpoint on the last user message (tool results
/// are user messages on this dialect), so each request writes the conversation
/// prefix the next one reads back. Any block type except empty text accepts
/// `cache_control` here (`tool_result` and `image` included).
pub(super) fn mark_messages_cache_breakpoint(messages: &mut [Value]) {
    if let Some(content) = messages
        .iter_mut()
        .rev()
        .find(|message| {
            message.get("role").and_then(Value::as_str) == Some("user")
                && !is_code_inspection_memory_message(message)
        })
        .and_then(|message| message.get_mut("content"))
    {
        mark_content_cache_breakpoint(content, |block| {
            block.get("type").and_then(Value::as_str) != Some("text")
                || is_nonempty_text_block(block)
        });
    }
}

/// User message content: plain strings pass through; structured blocks are
/// converted to Anthropic content blocks (text → `text`, images → `image`),
/// so vision-capable models receive the actual pixels instead of a degraded
/// text extraction.
fn messages_user_content(message: &Message) -> Value {
    messages_content_blocks(&message.content)
}

/// Convert a message content value to Anthropic Messages content blocks.
/// Strings pass through verbatim; block arrays are mapped part-by-part via
/// [`messages_content_block`]; anything else is reduced to its displayable text.
fn messages_content_blocks(content: &Value) -> Value {
    match content {
        Value::String(text) => Value::String(text.clone()),
        Value::Array(items) => {
            let blocks: Vec<Value> = items.iter().filter_map(messages_content_block).collect();
            if blocks.is_empty() {
                Value::String(String::new())
            } else {
                Value::Array(blocks)
            }
        }
        other => Value::String(extract_displayable_text(other)),
    }
}

/// One content part → Anthropic block, or `None` when nothing sendable remains
/// (e.g. an image without a usable URL). Empty text parts are dropped because
/// Anthropic rejects empty `text` blocks with a 400.
fn messages_content_block(item: &Value) -> Option<Value> {
    let is_image = matches!(item.get("type").and_then(Value::as_str), Some("image_url"))
        || item.get("image_url").is_some()
        || matches!(item.get("kind").and_then(Value::as_str), Some("image"));
    if is_image {
        return messages_image_block(item);
    }
    let text = extract_displayable_text(item);
    if text.is_empty() {
        return None;
    }
    Some(json!({ "type": "text", "text": text }))
}

/// OpenAI image part (`{"type": "image_url", "image_url": {"url": ...}}` or a
/// bare `{"kind": "image", "url": ...}`) → Anthropic `image` block. Base64
/// `data:` URIs become `source.type = base64`; http(s) URLs become
/// `source.type = url`; anything else is dropped.
fn messages_image_block(item: &Value) -> Option<Value> {
    let url = item
        .get("image_url")
        .and_then(|v| v.get("url").or(Some(v)))
        .and_then(Value::as_str)
        .or_else(|| item.get("url").and_then(Value::as_str))?;
    if let Some(rest) = url.strip_prefix("data:") {
        let (media_type, payload) = rest.split_once(',')?;
        let media_type = media_type.strip_suffix(";base64").unwrap_or(media_type);
        return Some(json!({
            "type": "image",
            "source": { "type": "base64", "media_type": media_type, "data": payload },
        }));
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return Some(json!({
            "type": "image",
            "source": { "type": "url", "url": url },
        }));
    }
    None
}

/// Assistant message → content blocks: text plus `tool_use` blocks for each tool
/// call (the JSON-encoded arguments become the `input` object; unparseable
/// arguments are wrapped so the wire stays valid JSON).
fn messages_assistant_message(message: &Message) -> Value {
    let mut blocks: Vec<Value> = Vec::new();
    // Replay the previous thinking chain before the visible text / tool_use
    // blocks. Anthropic requires thinking blocks to be echoed back when a tool
    // round continues the same assistant turn, so replay only on tool-call
    // rounds (the only case where a `tool_result` follows); stale reasoning on
    // plain-answer turns is not resent. The zen gateway's adaptive-thinking
    // dialect accepts plain `thinking` blocks; the `signature` field is not
    // persisted and is not required by that gateway.
    if let Some(thinking) = message
        .reasoning_content
        .as_deref()
        .filter(|text| !text.trim().is_empty())
    {
        let has_tool_calls = message
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| !tool_calls.is_empty());
        if has_tool_calls {
            blocks.push(json!({ "type": "thinking", "thinking": thinking }));
        }
    }
    let text = extract_displayable_text(&message.content);
    if !text.is_empty() {
        blocks.push(json!({"type": "text", "text": text}));
    }
    if let Some(tool_calls) = message.tool_calls.as_ref() {
        for call in tool_calls {
            let input = serde_json::from_str(&call.function.arguments)
                .unwrap_or_else(|_| json!({"arguments": call.function.arguments}));
            blocks.push(json!({
                "type": "tool_use",
                "id": call.id,
                "name": call.function.name,
                "input": input,
            }));
        }
    }
    if blocks.is_empty() {
        blocks.push(json!({"type": "text", "text": ""}));
    }
    json!({"role": "assistant", "content": blocks})
}

/// OpenAI-compatible `tools` array (function shape) → Anthropic
/// `{name, description, input_schema}`. Non-function tools (e.g. web_search) are
/// dropped; the Messages dialect keeps search off for now.
fn messages_tools(tools: Option<&Value>) -> Option<Value> {
    let converted = tools
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| {
            let function = tool.get("function").unwrap_or(tool);
            let name = function.get("name").and_then(Value::as_str)?;
            let mut converted = serde_json::Map::new();
            converted.insert("name".to_string(), Value::String(name.to_string()));
            if let Some(description) = function.get("description").and_then(Value::as_str) {
                converted.insert(
                    "description".to_string(),
                    Value::String(description.to_string()),
                );
            }
            if let Some(parameters) = function.get("parameters") {
                converted.insert("input_schema".to_string(), parameters.clone());
            }
            Some(Value::Object(converted))
        })
        .collect::<Vec<_>>();
    (!converted.is_empty()).then_some(Value::Array(converted))
}
