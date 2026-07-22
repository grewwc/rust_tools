//! 请求协议方言。
//!
//! provider adapter 负责「谁来发、字段长什么样」；而 `/v1/chat/completions`
//! 与 `/v1/responses` 这种 endpoint 级 wire 差异，则集中在本模块处理，
//! 避免把协议判断散落在 `transport.rs` / `builder.rs`。

use serde_json::{Value, json};

use super::reasoning::resolve_reasoning_wire_controls;
use super::{RequestBody, types::extract_displayable_text};
use crate::ai::history::Message;
use crate::ai::models;
use crate::ai::request_protocol::RequestProtocolDialect;
use crate::ai::types::ToolCall;

impl RequestProtocolDialect {
    pub(super) fn build_http_body(self, request: &RequestBody<'_>) -> Value {
        match self {
            Self::ChatCompletions => {
                serde_json::to_value(request).expect("chat-completions body should serialize")
            }
            Self::Responses => build_responses_request_body(request),
        }
    }
}

pub(crate) fn build_http_body_for_request(
    model: &str,
    endpoint: &str,
    request: &RequestBody<'_>,
) -> Value {
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
) -> Value {
    let request_messages = json_messages_to_request_messages(messages);
    let (thinking, reasoning_effort, reasoning) =
        resolve_reasoning_wire_controls(model, endpoint, false, reasoning_effort);
    let stream_options = (stream && include_stream_usage).then(|| json!({ "include_usage": true }));
    let request = RequestBody {
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
    };
    build_http_body_for_request(model, endpoint, &request)
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

/// 过滤掉空文本的 content item：Responses API 会拒绝 `text` 为空串的
/// `output_text` / `input_text`（返回 400 invalid_value）。
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
    // 注意：不要把 `reasoning_content` 回放成 message content 里的 `summary_text`。
    // Responses API 的 message content 只接受 `output_text` / `refusal`，塞入
    // `summary_text` 会 400。推理 summary 属于独立的 `reasoning` output item，
    // 且回放需要服务端原始 item id / encrypted_content，我们持久化时并未保留，
    // 因此无法忠实回放——直接不回传，交由本轮 `reasoning` 请求参数重新索取。
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
            // Responses API 的工具回合是扁平 output-item 序列。若本 turn 侧信道
            // 捕获到了该回合的 reasoning items（以首个 tool_call id 为 key），
            // 先原样 splice 回去，让模型保留上一跳的推理上下文（encrypted），
            // 再续上 function_call items。narration 文本仍不单独补发。
            if let Some(items) = reasoning_items
                .zip(tool_calls.first())
                .and_then(|(map, first)| map.get(&first.id))
            {
                input.extend(items.iter().cloned());
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

fn responses_tools(tools: &Value) -> Value {
    Value::Array(
        tools
            .as_array()
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
            .collect(),
    )
}

pub(super) fn build_responses_request_body(request: &RequestBody<'_>) -> Value {
    let mut body = json!({
        "model": request.model,
        "input": responses_input(request.messages, request.reasoning_items),
        "stream": request.stream,
    });
    let object = body.as_object_mut().expect("responses body is an object");
    // 加密推理回放：显式索取 `reasoning.encrypted_content`，服务端才会在
    // `response.output_item.done` 里下发带 encrypted_content 的 reasoning item，
    // 供同 turn 工具链回放（见 responses_input）。仅对声明了该能力位的模型开启。
    if request.reasoning_encrypted_replay {
        object.insert(
            "include".to_string(),
            json!(["reasoning.encrypted_content"]),
        );
    }
    if let Some(tools) = &request.tools {
        object.insert("tools".to_string(), responses_tools(tools));
    }
    if let Some(tool_choice) = &request.tool_choice {
        object.insert("tool_choice".to_string(), tool_choice.clone());
    }
    if let Some(effort) = request.reasoning_effort {
        // Responses API 默认不会返回推理文本；显式索取 reasoning summary，
        // provider 才会发送 response.reasoning_summary_text.* 事件。
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
