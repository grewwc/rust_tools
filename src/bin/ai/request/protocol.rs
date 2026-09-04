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
        }
    }
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
