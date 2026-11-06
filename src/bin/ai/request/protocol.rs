//! Request protocol dialect.
//!
//! The provider adapter owns "who sends and what fields look like"; endpoint-level wire
//! differences such as `/v1/chat/completions` vs `/v1/responses` are centralized in this
//! module, so protocol decisions do not scatter across `transport.rs` / `builder.rs`.
//! Messages and Responses encoders live in private child modules; public builders
//! and the adapter-before-encoding boundary remain here.

mod messages;
mod responses;

use serde_json::{Value, json};

use super::RequestBody;
use super::builder::RequestModelInputs;
use super::reasoning::{
    apply_prompt_cache_breakpoints, resolve_reasoning_wire_controls_from_parts,
};
use crate::ai::history::Message;
use crate::ai::models;
use crate::ai::provider::ApiProvider;
use crate::ai::request_protocol::RequestProtocolDialect;
use crate::ai::types::ToolCall;

use messages::build_messages_request_body;
#[cfg(test)]
use messages::mark_messages_cache_breakpoint;
pub(super) use responses::build_responses_request_body;
pub(in crate::ai::request) use responses::responses_reasoning_replay_stats;

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
            Self::ChatCompletions if request.explicit_cache_breakpoints => {
                // Breakpoints patch the wire copy only; the cost of the `Value`
                // round trip is paid solely by opted-in models.
                let mut body =
                    serde_json::to_value(request).expect("chat-completions body should serialize");
                if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
                    apply_prompt_cache_breakpoints(messages);
                }
                serde_json::to_vec(&body).expect("chat-completions body should serialize")
            }
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

pub(crate) fn build_http_body_for_request(
    model: &str,
    endpoint: &str,
    request: &mut RequestBody<'_>,
) -> Vec<u8> {
    build_http_body_for_request_from_parts(
        models::model_adapter(model),
        models::request_protocol_dialect(model, endpoint),
        endpoint,
        request,
    )
}

/// Registry-free core of [`build_http_body_for_request`] over the pre-resolved
/// adapter and protocol dialect.
pub(crate) fn build_http_body_for_request_from_parts(
    adapter: ApiProvider,
    dialect: RequestProtocolDialect,
    endpoint: &str,
    request: &mut RequestBody<'_>,
) -> Vec<u8> {
    // Step 4: provider differences converge into the Adapter hook -- fired uniformly on every
    // request path before serialization.
    crate::ai::provider::adapter_for(adapter, endpoint).adapt_request(request);
    dialect.build_http_body(request)
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
    build_http_body_for_json_messages_from_parts(
        &RequestModelInputs::resolve_with_endpoint(model, endpoint),
        messages,
        stream,
        reasoning_effort,
        include_stream_usage,
    )
}

/// Registry-free core of [`build_http_body_for_json_messages`] over
/// pre-resolved [`RequestModelInputs`].
pub(crate) fn build_http_body_for_json_messages_from_parts(
    inputs: &RequestModelInputs,
    messages: &[Value],
    stream: bool,
    reasoning_effort: Option<&str>,
    include_stream_usage: bool,
) -> Vec<u8> {
    let model = inputs.request_model.as_str();
    let endpoint = inputs.endpoint.as_str();
    let request_messages = json_messages_to_request_messages(messages);
    let (thinking, reasoning_effort, reasoning) = resolve_reasoning_wire_controls_from_parts(
        inputs.adapter,
        model,
        endpoint,
        inputs.effort_wire,
        false,
        reasoning_effort,
    );
    let stream_options = (stream && include_stream_usage).then(|| json!({ "include_usage": true }));
    let mut request = RequestBody {
        model: inputs.request_model.clone(),
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
        reasoning_encrypted_replay: inputs.encrypted_replay,
        estimated_prompt_tokens: 0,
        estimated_prompt_tokens_upper: 0,
        message_counts: Default::default(),
        prompt_cache_key: None,
        explicit_cache_breakpoints: false,
        tool_errors: None,
    };
    build_http_body_for_request_from_parts(inputs.adapter, inputs.dialect, endpoint, &mut request)
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
            estimated_prompt_tokens_upper: 0,
            message_counts: Default::default(),
            prompt_cache_key: None,
            explicit_cache_breakpoints: false,
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
            estimated_prompt_tokens_upper: 0,
            message_counts: Default::default(),
            prompt_cache_key: None,
            explicit_cache_breakpoints: false,
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
    fn messages_body_places_system_and_rolling_cache_breakpoints_only_when_enabled() {
        let messages = json_messages_to_request_messages(&[
            json!({"role": "system", "content": "You are a helper."}),
            json!({"role": "user", "content": "search rust"}),
            json!({"role": "assistant", "content": "done"}),
            json!({"role": "user", "content": "thanks"}),
        ]);
        let mut request = RequestBody {
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
            estimated_prompt_tokens_upper: 0,
            message_counts: Default::default(),
            prompt_cache_key: None,
            explicit_cache_breakpoints: false,
            tool_errors: None,
        };
        let plain = build_messages_request_body(&request);
        assert!(!plain.to_string().contains("cache_control"));

        request.explicit_cache_breakpoints = true;
        let body = build_messages_request_body(&request);
        assert_eq!(body["system"][0]["text"], "You are a helper.");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        let wire = body["messages"].as_array().expect("messages array");
        // Only the last user message carries the rolling breakpoint.
        assert_eq!(wire[0]["content"], "search rust");
        assert_eq!(wire[2]["content"][0]["text"], "thanks");
        assert_eq!(wire[2]["content"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn messages_cache_breakpoint_skips_empty_text_but_marks_tool_results() {
        let mut messages = vec![json!({
            "role": "user",
            "content": [
                { "type": "tool_result", "tool_use_id": "call_1", "content": "ok" },
                { "type": "text", "text": "" },
            ],
        })];
        mark_messages_cache_breakpoint(&mut messages);
        let blocks = messages[0]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["cache_control"]["type"], "ephemeral");
        assert!(blocks[1].get("cache_control").is_none());

        let mut empty = vec![json!({ "role": "user", "content": "" })];
        mark_messages_cache_breakpoint(&mut empty);
        assert_eq!(empty[0]["content"], "");
    }

    #[test]
    fn messages_cache_breakpoint_skips_the_working_memory_tail_note() {
        let mut messages = vec![
            json!({"role": "user", "content": "question"}),
            json!({"role": "assistant", "content": "calling tool"}),
            json!({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "ok"},
            ]}),
            json!({"role": "user", "content":
                "Current code-inspection working memory:\n- read_file(file=src/lib.rs) => ok"}),
        ];
        mark_messages_cache_breakpoint(&mut messages);

        // The editable note must not own the rolling breakpoint; the last stable tool result
        // keeps it so the next round can reuse the whole previous prefix.
        assert!(messages[3].get("cache_control").is_none());
        assert!(messages[3]["content"].get("cache_control").is_none());
        let blocks = messages[2]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn messages_body_keeps_the_working_memory_note_out_of_the_system_prefix() {
        let note = "Current code-inspection working memory:\n- read_file(file=src/lib.rs) => ok";
        let messages = json_messages_to_request_messages(&[
            json!({"role": "system", "content": "stable system"}),
            json!({"role": "user", "content": "question"}),
            json!({"role": "assistant", "content": "calling tool"}),
            json!({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "ok"},
            ]}),
            json!({"role": "user", "content": note}),
        ]);
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
            estimated_prompt_tokens_upper: 0,
            message_counts: Default::default(),
            prompt_cache_key: None,
            explicit_cache_breakpoints: false,
            tool_errors: None,
        };

        let body = build_messages_request_body(&request);
        assert_eq!(body["system"].as_str(), Some("stable system"));
        let wire = body["messages"].as_array().expect("messages array");
        assert_eq!(wire.last().unwrap()["role"], "user");
        assert_eq!(wire.last().unwrap()["content"], note);
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
        // Real-chain proof: the from_parts normalization strips
        // `reasoning_content` for models that do not declare replay
        // (reasoning.rs). The Anthropic Messages dialect opts into
        // shape-preservation via `requires_reasoning_content_echo`, so a
        // tool-call round's thinking text survives to the wire. Synthetic
        // (OpenCode adapter, /v1/messages endpoint) inputs pin the dialect
        // without depending on any registry key.
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
        crate::ai::request::reasoning::normalize_reasoning_content_replay_from_parts(
            "synthetic-test-tag",
            false,
            ApiProvider::OpenCode,
            "synthetic-messages-model",
            "https://example.com/v1/messages",
            &mut messages,
        );
        assert_eq!(
            messages[0].reasoning_content.as_deref(),
            Some("the file is at src/lib.rs"),
            "tool-call round reasoning must survive the request-layer normalization"
        );
        // Registry coverage: the Messages-dialect class the synthetic inputs
        // stand for must keep existing, or the pin tests a dead path.
        assert!(
            crate::ai::model_names::all()
                .iter()
                .any(|m| m.request_protocol == Some(RequestProtocolDialect::Messages)),
            "registry must keep at least one Messages-protocol entry"
        );
        let request = RequestBody {
            model: "synthetic-messages-model".to_string(),
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
            estimated_prompt_tokens_upper: 0,
            message_counts: Default::default(),
            prompt_cache_key: None,
            explicit_cache_breakpoints: false,
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
            estimated_prompt_tokens_upper: 0,
            message_counts: Default::default(),
            prompt_cache_key: None,
            explicit_cache_breakpoints: false,
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

#[cfg(test)]
mod wire_body_tests {
    use super::*;
    use crate::ai::types::ReasoningItemMap;
    use std::sync::Arc;

    #[test]
    fn http_body_builder_preserves_messages_and_responses_wire_bytes() {
        let messages = json_messages_to_request_messages(&[
            json!({"role": "system", "content": "Stable instruction."}),
            json!({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "Inspect this."},
            ]}),
            json!({
                "role": "assistant",
                "content": "Checking.",
                "reasoning_content": "First inspect the image.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "inspect", "arguments": "{\"path\":\"a.png\"}"},
                }],
            }),
            json!({"role": "tool", "tool_call_id": "call_1", "content": "failed"}),
        ]);
        let reasoning_item = json!({
            "type": "reasoning", "id": "rs_1", "encrypted_content": "ENC", "summary": [],
        });
        let mut reasoning_items = ReasoningItemMap::default();
        reasoning_items.insert("call_1".to_string(), Arc::new(vec![reasoning_item.clone()]));
        let tool_errors = [("call_1".to_string(), true)].into_iter().collect();
        let mut request = RequestBody {
            model: "synthetic-wire-model".to_string(),
            messages: &messages,
            stream: true,
            thinking: serde_json::Map::from_iter([(
                "thinking".to_string(),
                json!({"type": "adaptive"}),
            )]),
            enable_search: Some(true),
            tools: Some(json!([{"type": "function", "function": {
                "name": "inspect", "description": "Inspect an image",
                "parameters": {"type": "object"},
            }}])),
            tool_choice: Some(json!("auto")),
            reasoning_effort: Some("high"),
            reasoning: None,
            stream_options: Some(json!({"include_usage": true})),
            max_tokens: Some(1024),
            reasoning_items: Some(&reasoning_items),
            reasoning_encrypted_replay: true,
            estimated_prompt_tokens: 0,
            estimated_prompt_tokens_upper: 0,
            message_counts: Default::default(),
            prompt_cache_key: Some("session-key".to_string()),
            explicit_cache_breakpoints: true,
            tool_errors: Some(&tool_errors),
        };

        // Pin the pre-extraction bodies independently of the dialect helpers,
        // including object insertion order and tool-round content order.
        let messages_body = json!({
            "model": "synthetic-wire-model",
            "stream": true,
            "max_tokens": 1024,
            "system": [{"type": "text", "text": "Stable instruction.",
                "cache_control": {"type": "ephemeral"}}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": "AAAA",
                    }},
                    {"type": "text", "text": "Inspect this."},
                ]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "First inspect the image."},
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "call_1", "name": "inspect",
                        "input": {"path": "a.png"}},
                ]},
                {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": "call_1", "content": "failed",
                    "is_error": true, "cache_control": {"type": "ephemeral"},
                }]},
            ],
            "tools": [{"name": "inspect", "description": "Inspect an image",
                "input_schema": {"type": "object"}}],
            "thinking": {"type": "adaptive"},
        });
        let responses_body = json!({
            "model": "synthetic-wire-model",
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": "Stable instruction."}]},
                {"role": "user", "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                    {"type": "input_text", "text": "Inspect this."},
                ]},
                reasoning_item,
                {"role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]},
                {"type": "function_call", "call_id": "call_1", "name": "inspect",
                    "arguments": "{\"path\":\"a.png\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "failed"},
            ],
            "stream": true,
            "include": ["reasoning.encrypted_content"],
            "tools": [
                {"type": "function", "name": "inspect", "description": "Inspect an image",
                    "parameters": {"type": "object"}},
                {"type": "web_search"},
            ],
            "tool_choice": "auto",
            "reasoning": {"effort": "high", "summary": "auto"},
            "max_output_tokens": 1024,
            "prompt_cache_key": "session-key",
        });

        for (dialect, endpoint, expected) in [
            (
                RequestProtocolDialect::Messages,
                "https://example.com/v1/messages",
                messages_body,
            ),
            (
                RequestProtocolDialect::Responses,
                "https://example.com/v1/responses",
                responses_body,
            ),
        ] {
            let actual = build_http_body_for_request_from_parts(
                ApiProvider::Compatible,
                dialect,
                endpoint,
                &mut request,
            );
            assert_eq!(
                serde_json::from_slice::<Value>(&actual).expect("wire body should be valid JSON"),
                expected,
                "{dialect:?}",
            );
            assert_eq!(
                actual,
                serde_json::to_vec(&expected).unwrap(),
                "{dialect:?}"
            );
        }
    }
}
