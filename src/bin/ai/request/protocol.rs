//! Request protocol dialect.
//!
//! The provider adapter owns "who sends and what fields look like"; endpoint-level wire
//! differences such as `/v1/chat/completions` vs `/v1/responses` are centralized in this
//! module, so protocol decisions do not scatter across `transport.rs` / `builder.rs`.

use serde_json::{Value, json};

use super::reasoning::resolve_reasoning_wire_controls;
use super::{RequestBody, types::extract_displayable_text};
use crate::ai::history::Message;
use crate::ai::models;
use crate::ai::request_protocol::RequestProtocolDialect;
use crate::ai::types::ToolCall;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai::request) struct ResponsesReasoningReplayStats {
    pub(in crate::ai::request) tool_call_groups: usize,
    pub(in crate::ai::request) replayed_groups: usize,
    pub(in crate::ai::request) missing_groups: usize,
}

impl RequestProtocolDialect {
    /// Serializes the request into wire bytes. chat-completions goes straight through `to_vec`,
    /// avoiding a `to_value` deep clone of the whole request before a second serialization (the
    /// waste grows with history/tool schemas); the responses dialect still needs to build a Value
    /// first, then serialize.
    pub(super) fn build_http_body(self, request: &RequestBody<'_>) -> Vec<u8> {
        match self {
            Self::ChatCompletions => {
                serde_json::to_vec(request).expect("chat-completions body should serialize")
            }
            Self::Responses => serde_json::to_vec(&build_responses_request_body(request))
                .expect("responses body should serialize"),
            Self::Messages => serde_json::to_vec(&build_messages_request_body(request))
                .expect("messages body should serialize"),
        }
    }
}

/// Anthropic Messages API body: a top-level `system` string, a `messages` array
/// of content-block messages, `tools` as `{name, description, input_schema}` and
/// a required `max_tokens`. History roles map onto Anthropic user/assistant
/// blocks; OpenAI-shaped tool calls become `tool_use` blocks and tool results
/// become `tool_result` blocks inside a user message. Extended thinking is
/// merged in from `request.thinking` (produced by the thinking dialect) and the
/// assistant thinking chain is replayed as `thinking` blocks, so vision and
/// thinking stay available on this dialect instead of being silently dropped.
fn build_messages_request_body(request: &RequestBody<'_>) -> Value {
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
        body.insert("system".to_string(), Value::String(system_parts.join("\n\n")));
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
                converted.insert("description".to_string(), Value::String(description.to_string()));
            }
            if let Some(parameters) = function.get("parameters") {
                converted.insert("input_schema".to_string(), parameters.clone());
            }
            Some(Value::Object(converted))
        })
        .collect::<Vec<_>>();
    (!converted.is_empty()).then_some(Value::Array(converted))
}

pub(crate) fn build_http_body_for_request(
    model: &str,
    endpoint: &str,
    request: &mut RequestBody<'_>,
) -> Vec<u8> {
    // Step 4: provider differences converge into the Adapter hook -- fired uniformly on every
    // request path before serialization.
    crate::ai::provider::adapter_for(models::model_adapter(model), endpoint).adapt_request(request);
    models::request_protocol_dialect(model, endpoint).build_http_body(request)
}

pub(crate) fn json_messages_to_request_messages(messages: &[Value]) -> Vec<Message> {
    messages
        .iter()
        .map(|message| {
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user")
                .to_string();
            let content = message
                .get("content")
                .cloned()
                .unwrap_or_else(|| Value::String(String::new()));
            let tool_calls = message
                .get("tool_calls")
                .and_then(|value| serde_json::from_value::<Vec<ToolCall>>(value.clone()).ok());
            let tool_call_id = message
                .get("tool_call_id")
                .and_then(Value::as_str)
                .map(str::to_string);
            let reasoning_content = message
                .get("reasoning_content")
                .and_then(Value::as_str)
                .map(str::to_string);
            Message {
                role,
                content,
                tool_calls,
                tool_call_id,
                reasoning_content,
            }
        })
        .collect()
}

pub(crate) fn build_http_body_for_json_messages(
    model: &str,
    endpoint: &str,
    messages: &[Value],
    stream: bool,
    reasoning_effort: Option<&str>,
    include_stream_usage: bool,
) -> Vec<u8> {
    let request_messages = json_messages_to_request_messages(messages);
    let (thinking, reasoning_effort, reasoning) =
        resolve_reasoning_wire_controls(model, endpoint, false, reasoning_effort);
    let stream_options = (stream && include_stream_usage).then(|| json!({ "include_usage": true }));
    let mut request = RequestBody {
        model: models::request_model_name(model),
        messages: &request_messages,
        stream,
        thinking,
        enable_search: None,
        tools: None,
        tool_choice: None,
        reasoning_effort,
        reasoning,
        stream_options,
        max_tokens: None,
        reasoning_items: None,
        reasoning_encrypted_replay: models::reasoning_encrypted_replay_enabled(model),
        estimated_prompt_tokens: 0,
        tool_errors: None,
    };
    build_http_body_for_request(model, endpoint, &mut request)
}

pub(crate) fn extract_response_text(v: &Value) -> Option<String> {
    if let Some(content) = extract_chat_choices_text(v) {
        return Some(content);
    }
    if let Some(text) = v.get("output_text").and_then(Value::as_str) {
        return Some(text.to_string());
    }
    if let Some(output) = v.get("output").and_then(Value::as_array) {
        let mut out = String::new();
        for item in output {
            append_responses_output_item_text(&mut out, item);
        }
        if !out.is_empty() {
            return Some(out);
        }
    }
    None
}

fn extract_chat_choices_text(v: &Value) -> Option<String> {
    let choices = v
        .get("choices")
        .or_else(|| v.get("output").and_then(|o| o.get("choices")))?;
    let msg = choices.get(0)?.get("message")?;
    let content = msg.get("content")?;
    extract_content_text(content)
}

fn extract_content_text(content: &Value) -> Option<String> {
    match content {
        Value::String(s) => Some(s.to_string()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                append_content_part_text(&mut out, part);
            }
            Some(out)
        }
        _ => None,
    }
}

fn append_responses_output_item_text(out: &mut String, item: &Value) {
    if let Some(content) = item.get("content") {
        match content {
            Value::Array(parts) => {
                for part in parts {
                    append_content_part_text(out, part);
                }
            }
            Value::String(text) => out.push_str(text),
            _ => {}
        }
        return;
    }

    let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
    if matches!(item_type, "output_text" | "text" | "refusal") {
        append_content_part_text(out, item);
    }
}

fn append_content_part_text(out: &mut String, part: &Value) {
    if let Some(text) = part
        .get("text")
        .or_else(|| part.get("output_text"))
        .or_else(|| part.get("refusal"))
        .and_then(Value::as_str)
    {
        out.push_str(text);
    }
}

fn responses_content_type_for_role(role: &str) -> &'static str {
    if role.eq_ignore_ascii_case("assistant") {
        "output_text"
    } else {
        "input_text"
    }
}

fn responses_content_item(role: &str, item: &Value) -> Value {
    let item_type = item
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    if matches!(
        item_type,
        "input_text"
            | "input_image"
            | "output_text"
            | "refusal"
            | "input_file"
            | "computer_screenshot"
            | "summary_text"
            | "tether_browsing_display"
    ) {
        return item.clone();
    }

    if let Some(url) = item
        .get("image_url")
        .and_then(|v| v.get("url").or(Some(v)))
        .and_then(|v| v.as_str())
    {
        return json!({
            "type": "input_image",
            "image_url": url,
        });
    }

    if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
        return json!({
            "type": responses_content_type_for_role(role),
            "text": text,
        });
    }

    let fallback = extract_displayable_text(item);
    if !fallback.is_empty() {
        return json!({
            "type": responses_content_type_for_role(role),
            "text": fallback,
        });
    }

    item.clone()
}

fn responses_content_items(role: &str, content: &Value) -> Vec<Value> {
    match content {
        Value::Array(items) => items
            .iter()
            .map(|item| responses_content_item(role, item))
            .collect(),
        Value::String(text) => vec![json!({
            "type": responses_content_type_for_role(role),
            "text": text,
        })],
        Value::Null => Vec::new(),
        other => {
            let fallback = extract_displayable_text(other);
            if fallback.is_empty() {
                vec![other.clone()]
            } else {
                vec![json!({
                    "type": responses_content_type_for_role(role),
                    "text": fallback,
                })]
            }
        }
    }
}

/// Filters out content items with empty text: the Responses API rejects `output_text` /
/// `input_text` whose `text` is an empty string (returns 400 invalid_value).
fn responses_item_is_empty_text(item: &Value) -> bool {
    matches!(
        item.get("type").and_then(Value::as_str),
        Some("input_text" | "output_text")
    ) && item
        .get("text")
        .and_then(Value::as_str)
        .is_none_or(|text| text.is_empty())
}

fn responses_message_content(message: &Message) -> Vec<Value> {
    // Note: do not replay `reasoning_content` as `summary_text` in message content.
    // The Responses API message content only accepts `output_text` / `refusal`; putting
    // `summary_text` in causes a 400. Reasoning summaries are a separate `reasoning` output item,
    // and replay would need the server-side original item id / encrypted_content, which we do not
    // persist -- so faithful replay is impossible; do not send them back and let this round's
    // `reasoning` request parameter fetch them anew.
    responses_content_items(&message.role, &message.content)
        .into_iter()
        .filter(|item| !responses_item_is_empty_text(item))
        .collect()
}

fn responses_message_input(message: &Message) -> Value {
    json!({
        "role": message.role,
        "content": responses_message_content(message),
    })
}

pub(in crate::ai::request) fn responses_reasoning_replay_stats(
    messages: &[Message],
    reasoning_items: Option<&rustc_hash::FxHashMap<String, Vec<Value>>>,
) -> ResponsesReasoningReplayStats {
    let mut stats = ResponsesReasoningReplayStats::default();
    for message in messages {
        let Some(first_tool_call) = message
            .tool_calls
            .as_ref()
            .filter(|calls| !calls.is_empty())
            .and_then(|calls| calls.first())
        else {
            continue;
        };
        stats.tool_call_groups += 1;
        let replayed = reasoning_items
            .and_then(|items| items.get(&first_tool_call.id))
            .is_some_and(|items| !items.is_empty());
        if replayed {
            stats.replayed_groups += 1;
        } else {
            stats.missing_groups += 1;
        }
    }
    stats
}

fn responses_input(
    messages: &[Message],
    reasoning_items: Option<&rustc_hash::FxHashMap<String, Vec<Value>>>,
) -> Vec<Value> {
    let mut input = Vec::new();
    for message in messages {
        if let Some(tool_calls) = message
            .tool_calls
            .as_ref()
            .filter(|calls| !calls.is_empty())
        {
            // A Responses API tool round is a flat sequence of output items. Replay it in the
            // order the provider streamed it: reasoning items (if this turn's side channel
            // captured them, keyed by the first tool_call id — spliced verbatim so the model keeps
            // the previous hop's reasoning context), then the narration message item the model
            // produced before dispatching the tool, then the function_call items. The narration is
            // preserved on the chat-completions wire (driver/turn_runtime/tool_result/messaging.rs
            // persists it, truncated to 800 chars), so dropping it here would make Responses lose
            // the model's intermediate conclusions across tool hops.
            if let Some(items) = reasoning_items
                .zip(tool_calls.first())
                .and_then(|(map, first)| map.get(&first.id))
            {
                input.extend(items.iter().cloned());
            }
            let narration_items = responses_message_content(message);
            if !narration_items.is_empty() {
                // The Responses API rejects empty output_text, so only emit the message item when
                // the model actually produced narration.
                input.push(json!({
                    "role": "assistant",
                    "content": narration_items,
                }));
            }
            for tool_call in tool_calls {
                input.push(json!({
                    "type": "function_call",
                    "call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }));
            }
        } else if message.role == "tool" {
            if let Some(call_id) = &message.tool_call_id {
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": message.content,
                }));
            }
        } else {
            input.push(responses_message_input(message));
        }
    }
    input
}

fn responses_tools(tools: Option<&Value>, enable_search: bool) -> Option<Value> {
    let mut converted = tools
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|tool| {
            let function = tool.get("function").unwrap_or(tool);
            json!({
                "type": "function",
                "name": function.get("name").cloned().unwrap_or(Value::Null),
                "description": function.get("description").cloned().unwrap_or(Value::Null),
                "parameters": function.get("parameters").cloned().unwrap_or(Value::Null),
            })
        })
        .collect::<Vec<_>>();
    if enable_search {
        converted.push(json!({"type": "web_search"}));
    }
    (!converted.is_empty()).then_some(Value::Array(converted))
}

pub(super) fn build_responses_request_body(request: &RequestBody<'_>) -> Value {
    let mut body = json!({
        "model": request.model,
        "input": responses_input(request.messages, request.reasoning_items),
        "stream": request.stream,
    });
    let object = body.as_object_mut().expect("responses body is an object");
    // Encrypted reasoning replay: only when explicitly requesting `reasoning.encrypted_content`
    // does the server deliver a reasoning item carrying encrypted_content in
    // `response.output_item.done`, for replay within the same turn's tool chain (see
    // responses_input). Enabled only for models that advertise this capability bit.
    if request.reasoning_encrypted_replay {
        object.insert(
            "include".to_string(),
            json!(["reasoning.encrypted_content"]),
        );
    }
    if let Some(tools) =
        responses_tools(request.tools.as_ref(), request.enable_search == Some(true))
    {
        object.insert("tools".to_string(), tools);
    }
    if let Some(tool_choice) = &request.tool_choice {
        object.insert("tool_choice".to_string(), tool_choice.clone());
    }
    if let Some(effort) = request.reasoning_effort {
        // The Responses API does not return reasoning text by default; only when explicitly
        // requesting a reasoning summary does the provider send response.reasoning_summary_text.*
        // events.
        object.insert(
            "reasoning".to_string(),
            json!({ "effort": effort, "summary": "auto" }),
        );
    } else if let Some(reasoning) = &request.reasoning {
        object.insert("reasoning".to_string(), reasoning.clone());
    }
    if let Some(max_tokens) = request.max_tokens {
        object.insert("max_output_tokens".to_string(), max_tokens.into());
    }
    body
}

#[cfg(test)]
mod messages_body_tests {
    use super::*;
    use serde_json::json;

    fn build_body(messages: &[Value], tools: Option<Value>, max_tokens: Option<u32>) -> Value {
        let messages = json_messages_to_request_messages(messages);
        let request = RequestBody {
            model: "claude-opus-5-5".to_string(),
            messages: &messages,
            stream: true,
            thinking: serde_json::Map::new(),
            enable_search: None,
            tools,
            tool_choice: None,
            reasoning_effort: None,
            reasoning: None,
            stream_options: None,
            max_tokens,
            reasoning_items: None,
            reasoning_encrypted_replay: false,
            estimated_prompt_tokens: 0,
            tool_errors: None,
        };
        build_messages_request_body(&request)
    }

    #[test]
    fn messages_body_merges_thinking_controls() {
        let messages = json_messages_to_request_messages(&[]);
        let mut thinking = serde_json::Map::new();
        thinking.insert("thinking".to_string(), json!({"type": "adaptive"}));
        thinking.insert("output_config".to_string(), json!({"effort": "xhigh"}));
        let request = RequestBody {
            model: "claude-opus-5-5".to_string(),
            messages: &messages,
            stream: true,
            thinking,
            enable_search: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            reasoning: None,
            stream_options: None,
            max_tokens: Some(8192),
            reasoning_items: None,
            reasoning_encrypted_replay: false,
            estimated_prompt_tokens: 0,
            tool_errors: None,
        };
        let body = build_messages_request_body(&request);
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["output_config"]["effort"], "xhigh");
    }

    #[test]
    fn messages_body_maps_history_roles_and_tools() {
        let body = build_body(
            &[
                json!({"role": "system", "content": "You are a helper."}),
                json!({"role": "user", "content": "search rust"}),
                json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{\"q\": \"rust\"}"},
                    }],
                }),
                json!({"role": "tool", "tool_call_id": "call_1", "content": "results here"}),
            ],
            Some(json!([{
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }])),
            Some(1024),
        );

        assert_eq!(body["model"], "claude-opus-5-5");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 1024);
        assert_eq!(body["system"], "You are a helper.");
        assert_eq!(body["tools"][0]["name"], "web_search");
        assert_eq!(body["tools"][0]["description"], "Search the web");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");

        let messages = body["messages"].as_array().expect("messages array");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "search rust");
        // Assistant: tool_use content block (empty text stays off the wire).
        assert_eq!(messages[1]["role"], "assistant");
        let assistant_blocks = messages[1]["content"].as_array().expect("assistant blocks");
        assert_eq!(assistant_blocks.len(), 1);
        assert_eq!(assistant_blocks[0]["type"], "tool_use");
        assert_eq!(assistant_blocks[0]["id"], "call_1");
        assert_eq!(assistant_blocks[0]["name"], "web_search");
        assert_eq!(assistant_blocks[0]["input"]["q"], "rust");
        // Tool result: user message with a tool_result block.
        assert_eq!(messages[2]["role"], "user");
        let tool_block = &messages[2]["content"][0];
        assert_eq!(tool_block["type"], "tool_result");
        assert_eq!(tool_block["tool_use_id"], "call_1");
        assert_eq!(tool_block["content"], "results here");
    }

    #[test]
    fn messages_body_converts_user_images_to_anthropic_image_blocks() {
        let body = build_body(
            &[json!({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                    {"type": "text", "text": "what is in these images?"},
                ],
            })],
            None,
            None,
        );
        let blocks = body["messages"][0]["content"].as_array().expect("blocks");
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0]["type"], "image");
        assert_eq!(blocks[0]["source"]["type"], "base64");
        assert_eq!(blocks[0]["source"]["media_type"], "image/png");
        assert_eq!(blocks[0]["source"]["data"], "AAAA");
        assert_eq!(blocks[1]["type"], "image");
        assert_eq!(blocks[1]["source"]["type"], "url");
        assert_eq!(blocks[1]["source"]["url"], "https://example.com/a.png");
        assert_eq!(blocks[2]["type"], "text");
        assert_eq!(blocks[2]["text"], "what is in these images?");
    }

    #[test]
    fn messages_body_replays_thinking_only_on_tool_call_rounds() {
        // Tool-call round: the previous thinking chain is echoed before the
        // tool_use block (the Anthropic-required continuation case).
        let body = build_body(
            &[json!({
                "role": "assistant",
                "content": "let me check",
                "reasoning_content": "the file is at src/lib.rs",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{\"command\": \"ls\"}"},
                }],
            })],
            None,
            None,
        );
        let blocks = body["messages"][0]["content"].as_array().expect("blocks");
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["thinking"], "the file is at src/lib.rs");
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[2]["type"], "tool_use");

        // Plain-answer round: no `tool_result` follows, so the stale thinking
        // is not resent (avoids context bloat and orphan-thinking strictness).
        let body = build_body(
            &[json!({
                "role": "assistant",
                "content": "done",
                "reasoning_content": "old chain from a previous answer",
            })],
            None,
            None,
        );
        let blocks = body["messages"][0]["content"].as_array().expect("blocks");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "text");
    }

    #[test]
    fn messages_body_thinking_replay_survives_request_normalization_for_anthropic_dialect() {
        // Real-chain proof: `normalize_reasoning_content_replay_for_model`
        // strips `reasoning_content` for models that do not declare replay
        // (reasoning.rs). The Anthropic Messages dialect opts into
        // shape-preservation via `requires_reasoning_content_echo`, so a
        // tool-call round's thinking text survives to the wire. This runs the
        // actual request-layer normalization instead of constructing the body
        // directly.
        let mut messages = json_messages_to_request_messages(&[
            json!({
                "role": "assistant",
                "content": "let me check",
                "reasoning_content": "the file is at src/lib.rs",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": "{\"command\": \"ls\"}"},
                }],
            }),
            json!({"role": "tool", "tool_call_id": "call_1", "content": "ok"}),
        ]);
        crate::ai::request::normalize_reasoning_content_replay_for_model(
            "claude-opus-5-5-opencode",
            &mut messages,
        );
        assert_eq!(
            messages[0].reasoning_content.as_deref(),
            Some("the file is at src/lib.rs"),
            "tool-call round reasoning must survive the request-layer normalization"
        );
        let request = RequestBody {
            model: "claude-opus-5-5-opencode".to_string(),
            messages: &messages,
            stream: true,
            thinking: serde_json::Map::new(),
            enable_search: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            reasoning: None,
            stream_options: None,
            max_tokens: Some(4096),
            reasoning_items: None,
            reasoning_encrypted_replay: false,
            estimated_prompt_tokens: 0,
            tool_errors: None,
        };
        let body = build_messages_request_body(&request);
        let blocks = body["messages"][0]["content"].as_array().expect("blocks");
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["thinking"], "the file is at src/lib.rs");
    }

    #[test]
    fn messages_body_marks_failed_tool_results_with_is_error() {
        let messages = json_messages_to_request_messages(&[
            json!({"role": "tool", "tool_call_id": "call_1", "content": "boom"}),
            json!({"role": "tool", "tool_call_id": "call_2", "content": "fine"}),
        ]);
        let mut tool_errors = rustc_hash::FxHashMap::default();
        tool_errors.insert("call_1".to_string(), true);
        let request = RequestBody {
            model: "claude-opus-5-5".to_string(),
            messages: &messages,
            stream: true,
            thinking: serde_json::Map::new(),
            enable_search: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            reasoning: None,
            stream_options: None,
            max_tokens: None,
            reasoning_items: None,
            reasoning_encrypted_replay: false,
            estimated_prompt_tokens: 0,
            tool_errors: Some(&tool_errors),
        };
        let body = build_messages_request_body(&request);
        let messages = body["messages"].as_array().expect("messages array");
        assert_eq!(messages[0]["content"][0]["is_error"], true);
        assert!(messages[1]["content"][0].get("is_error").is_none());
    }

    #[test]
    fn messages_body_defaults_max_tokens_and_skips_non_function_tools() {
        let body = build_body(
            &[json!({"role": "user", "content": "hi"})],
            Some(json!([
                {"type": "function", "function": {"name": "a", "parameters": {}}},
                {"type": "web_search"},
            ])),
            None,
        );
        assert_eq!(body["max_tokens"], 4096);
        let tools = body["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "a");
        assert!(body.get("system").is_none());
    }

    #[test]
    fn messages_body_wraps_unparseable_tool_arguments() {
        let body = build_body(
            &[json!({
                "role": "assistant",
                "content": "thinking out loud",
                "tool_calls": [{
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "f", "arguments": "not json"},
                }],
            })],
            None,
            Some(256),
        );
        let blocks = body["messages"][0]["content"].as_array().expect("blocks");
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "thinking out loud");
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["input"]["arguments"], "not json");
    }
}
