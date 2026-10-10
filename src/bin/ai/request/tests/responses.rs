use super::super::protocol::build_http_body_for_json_messages;
use super::super::*;
use crate::ai::tools::os_tools::{GLOBAL_OS, init_os_tools_globals};
use crate::ai::{cli::ParsedCli, types::AppConfig};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};

#[test]
fn responses_protocol_dialect_infers_from_endpoint_when_unspecified() {
    assert_eq!(
        RequestProtocolDialect::infer_from_endpoint("https://api.example.com/v1/chat/completions"),
        RequestProtocolDialect::ChatCompletions
    );
    assert_eq!(
        RequestProtocolDialect::infer_from_endpoint("https://api.example.com/v1/responses"),
        RequestProtocolDialect::Responses
    );
}

#[test]
fn json_messages_aux_body_uses_responses_protocol_for_modelhub_models() {
    let endpoint = models::endpoint_for_model("gpt-5.5", "");
    let messages = vec![
        json!({"role": "system", "content": "Return JSON only."}),
        json!({"role": "user", "content": "classify this"}),
    ];

    let body_bytes =
        build_http_body_for_json_messages("gpt-5.5", &endpoint, &messages, false, None, false);
    let body: Value = serde_json::from_slice(&body_bytes).expect("body should be valid JSON");

    assert_eq!(body["model"], models::request_model_name("gpt-5.5"));
    assert!(body.get("messages").is_none());
    assert_eq!(body["input"][0]["role"], "system");
    assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
    assert_eq!(body["input"][0]["content"][0]["text"], "Return JSON only.");
    assert_eq!(body["input"][1]["role"], "user");
    assert_eq!(body["input"][1]["content"][0]["text"], "classify this");
}

#[test]
fn extract_response_text_reads_chat_and_responses_outputs() {
    let chat = json!({
        "choices": [{
            "message": {
                "content": [{"type": "text", "text": "chat text"}]
            }
        }]
    });
    assert_eq!(extract_response_text(&chat).as_deref(), Some("chat text"));

    let responses_shortcut = json!({
        "output_text": "shortcut text",
        "output": []
    });
    assert_eq!(
        extract_response_text(&responses_shortcut).as_deref(),
        Some("shortcut text")
    );

    let responses_output = json!({
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "hello"},
                {"type": "output_text", "text": " world"}
            ]
        }]
    });
    assert_eq!(
        extract_response_text(&responses_output).as_deref(),
        Some("hello world")
    );
}

#[test]
fn aux_stream_payload_reads_responses_text_and_usage_events() {
    let mut content = String::new();
    let mut usage = None;

    super::super::transport::apply_aux_stream_payload(
        r#"{"delta":"hello"}"#,
        Some("response.output_text.delta"),
        &mut content,
        &mut usage,
    );
    super::super::transport::apply_aux_stream_payload(
        r#"{"delta":" world"}"#,
        Some("response.output_text.delta"),
        &mut content,
        &mut usage,
    );
    super::super::transport::apply_aux_stream_payload(
        r#"{"response":{"model":"gpt-5.5","usage":{"input_tokens":11,"output_tokens":7,"total_tokens":18}}}"#,
        Some("response.completed"),
        &mut content,
        &mut usage,
    );

    assert_eq!(content, "hello world");
    let (model, usage) = usage.expect("response.completed should capture usage");
    assert_eq!(model, "gpt-5.5");
    assert_eq!(usage.prompt_tokens, 11);
    assert_eq!(usage.completion_tokens, 7);
    assert_eq!(usage.total_tokens, 18);
}

#[test]
fn responses_request_body_omits_assistant_reasoning_from_message_content() {
    // Responses message content accepts only output_text/refusal;
    // reasoning_content must not be replayed as summary_text (would 400) — keep
    // only the visible answer text.
    let messages = vec![Message {
        role: "assistant".to_string(),
        content: Value::String("final answer".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: Some("step summary".to_string()),
    }];
    let request = RequestBody {
        model: "gpt-5.5".to_string(),
        messages: &messages,
        stream: false,
        thinking: serde_json::Map::new(),
        enable_search: None,
        tools: None,
        tool_choice: None,
        reasoning_effort: Some("high"),
        reasoning: None,
        stream_options: None,
        max_tokens: None,
        reasoning_items: None,
        reasoning_encrypted_replay: false,
        estimated_prompt_tokens: 0,
    };

    let body = super::super::build_responses_request_body(&request);
    let content = body["input"][0]["content"]
        .as_array()
        .expect("assistant content should be encoded as array");
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "output_text");
    assert_eq!(content[0]["text"], "final answer");
    assert!(
        !content.iter().any(|item| item["type"] == "summary_text"),
        "message content must not contain summary_text"
    );
}

#[test]
fn responses_request_body_emits_bare_function_call_for_tool_turn() {
    // When the assistant carries only tool_calls (no visible text), emit flat
    // function_call items directly — no assistant message item, no reasoning
    // summary_text injected.
    let messages = vec![Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{\"path\":\"src/main.rs\"}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: Some("need to inspect the file first".to_string()),
    }];
    let request = RequestBody {
        model: "gpt-5.5".to_string(),
        messages: &messages,
        stream: false,
        thinking: serde_json::Map::new(),
        enable_search: None,
        tools: None,
        tool_choice: None,
        reasoning_effort: Some("high"),
        reasoning: None,
        stream_options: None,
        max_tokens: None,
        reasoning_items: None,
        reasoning_encrypted_replay: false,
        estimated_prompt_tokens: 0,
    };

    let body = super::super::build_responses_request_body(&request);
    let input = body["input"]
        .as_array()
        .expect("responses request should contain input items");
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["type"], "function_call");
    assert_eq!(input[0]["call_id"], "call_1");
    assert!(
        !input
            .iter()
            .any(|item| item["type"] == "summary_text"
                || item["content"][0]["type"] == "summary_text"),
        "tool-call turn must not replay reasoning as summary_text"
    );
}

#[test]
fn responses_request_body_drops_empty_text_content_items() {
    // Empty-string text is rejected by the Responses API (400 invalid_value) and
    // must be filtered out.
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String(String::new()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let request = RequestBody {
        model: "gpt-5.5".to_string(),
        messages: &messages,
        stream: false,
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
    };

    let body = super::super::build_responses_request_body(&request);
    let content = body["input"][0]["content"]
        .as_array()
        .expect("content should be an array");
    assert!(
        content.is_empty(),
        "empty-text content item should be filtered out, got: {content:?}"
    );
}

#[test]
fn responses_request_body_includes_encrypted_reasoning_flag_for_capable_model() {
    // Models declaring reasoning_encrypted_replay: the request must carry
    // include: ["reasoning.encrypted_content"], otherwise the server sends no
    // encrypted_content.
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let request = build_request_body(
        "gpt-5.5", &messages, false, false, None, None, None, None, None, None, None,
    );
    assert!(
        request.reasoning_encrypted_replay,
        "gpt-5.5 declares reasoning_encrypted_replay=true in model registry"
    );

    let body = super::super::build_responses_request_body(&request);
    assert_eq!(
        body["include"],
        json!(["reasoning.encrypted_content"]),
        "capable model must request encrypted reasoning include"
    );
}

#[test]
fn responses_request_body_omits_include_without_encrypted_replay() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let request = RequestBody {
        model: "gpt-5.5".to_string(),
        messages: &messages,
        stream: false,
        thinking: serde_json::Map::new(),
        enable_search: None,
        tools: None,
        tool_choice: None,
        reasoning_effort: Some("high"),
        reasoning: None,
        stream_options: None,
        max_tokens: None,
        reasoning_items: None,
        reasoning_encrypted_replay: false,
        estimated_prompt_tokens: 0,
    };

    let body = super::super::build_responses_request_body(&request);
    assert!(
        body.get("include").is_none(),
        "include must be omitted when encrypted replay is off"
    );
}

#[test]
fn responses_request_body_replays_reasoning_items_before_function_call() {
    // Side-channel hit: reasoning items keyed by the first tool_call id must be
    // spliced verbatim before the corresponding function_call, so the model keeps
    // the previous hop's reasoning context.
    let messages = vec![Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    }];
    let mut items = rustc_hash::FxHashMap::default();
    items.insert(
        "call_1".to_string(),
        vec![json!({
            "type": "reasoning",
            "id": "rs_abc",
            "encrypted_content": "ENC",
            "summary": [],
        })],
    );
    let request = RequestBody {
        model: "gpt-5.5".to_string(),
        messages: &messages,
        stream: false,
        thinking: serde_json::Map::new(),
        enable_search: None,
        tools: None,
        tool_choice: None,
        reasoning_effort: Some("high"),
        reasoning: None,
        stream_options: None,
        max_tokens: None,
        reasoning_items: Some(&items),
        reasoning_encrypted_replay: true,
        estimated_prompt_tokens: 0,
    };

    let body = super::super::build_responses_request_body(&request);
    let input = body["input"].as_array().expect("input array");
    assert_eq!(input.len(), 2, "reasoning item + function_call");
    assert_eq!(input[0]["type"], "reasoning");
    assert_eq!(input[0]["id"], "rs_abc");
    assert_eq!(input[0]["encrypted_content"], "ENC");
    assert_eq!(input[1]["type"], "function_call");
    assert_eq!(input[1]["call_id"], "call_1");

    let stats = responses_reasoning_replay_stats(&messages, Some(&items));
    assert_eq!(stats.tool_call_groups, 1);
    assert_eq!(stats.replayed_groups, 1);
    assert_eq!(stats.missing_groups, 0);
}

#[test]
fn responses_request_body_degrades_to_bare_function_call_without_reasoning_items() {
    // Side-channel miss (no encrypted_content available): degrade to flat
    // function_call, zero regression.
    let messages = vec![Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    }];
    let empty = rustc_hash::FxHashMap::default();
    let request = RequestBody {
        model: "gpt-5.5".to_string(),
        messages: &messages,
        stream: false,
        thinking: serde_json::Map::new(),
        enable_search: None,
        tools: None,
        tool_choice: None,
        reasoning_effort: Some("high"),
        reasoning: None,
        stream_options: None,
        max_tokens: None,
        reasoning_items: Some(&empty),
        reasoning_encrypted_replay: true,
        estimated_prompt_tokens: 0,
    };

    let body = super::super::build_responses_request_body(&request);
    let input = body["input"].as_array().expect("input array");
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["type"], "function_call");
    assert_eq!(input[0]["call_id"], "call_1");

    let stats = responses_reasoning_replay_stats(&messages, Some(&empty));
    assert_eq!(stats.tool_call_groups, 1);
    assert_eq!(stats.replayed_groups, 0);
    assert_eq!(stats.missing_groups, 1);
}

#[test]
fn responses_request_body_replays_tool_round_narration_before_function_call() {
    // P2 regression: assistant narration (the text the model produced before dispatching a
    // tool) was dropped on the Responses wire, while chat-completions keeps it. It must be
    // replayed as a message item ahead of the function_call items.
    let messages = vec![Message {
        role: "assistant".to_string(),
        content: Value::String("Let me read the file first.".to_string()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    }];
    let request = RequestBody {
        model: "gpt-5.5".to_string(),
        messages: &messages,
        stream: false,
        thinking: serde_json::Map::new(),
        enable_search: None,
        tools: None,
        tool_choice: None,
        reasoning_effort: Some("high"),
        reasoning: None,
        stream_options: None,
        max_tokens: None,
        reasoning_items: None,
        reasoning_encrypted_replay: false,
        estimated_prompt_tokens: 0,
    };

    let body = super::super::build_responses_request_body(&request);
    let input = body["input"].as_array().expect("input array");
    assert_eq!(input.len(), 2, "narration message + function_call");
    assert_eq!(input[0]["role"], "assistant");
    assert_eq!(input[0]["content"][0]["type"], "output_text");
    assert_eq!(
        input[0]["content"][0]["text"],
        "Let me read the file first."
    );
    assert_eq!(input[1]["type"], "function_call");
    assert_eq!(input[1]["call_id"], "call_1");
}

#[test]
fn responses_request_body_orders_reasoning_narration_function_call() {
    // A full tool round replays in the provider-streamed order: reasoning items, then the
    // narration message item, then function_call.
    let messages = vec![Message {
        role: "assistant".to_string(),
        content: Value::String("I need the current directory contents.".to_string()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "list_files".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    }];
    let mut items = rustc_hash::FxHashMap::default();
    items.insert(
        "call_1".to_string(),
        vec![json!({
            "type": "reasoning",
            "id": "rs_abc",
            "encrypted_content": "ENC",
            "summary": [],
        })],
    );
    let request = RequestBody {
        model: "gpt-5.5".to_string(),
        messages: &messages,
        stream: false,
        thinking: serde_json::Map::new(),
        enable_search: None,
        tools: None,
        tool_choice: None,
        reasoning_effort: Some("high"),
        reasoning: None,
        stream_options: None,
        max_tokens: None,
        reasoning_items: Some(&items),
        reasoning_encrypted_replay: true,
        estimated_prompt_tokens: 0,
    };

    let body = super::super::build_responses_request_body(&request);
    let input = body["input"].as_array().expect("input array");
    assert_eq!(
        input.len(),
        3,
        "reasoning + narration message + function_call"
    );
    assert_eq!(input[0]["type"], "reasoning");
    assert_eq!(input[1]["role"], "assistant");
    assert_eq!(
        input[1]["content"][0]["text"],
        "I need the current directory contents."
    );
    assert_eq!(input[2]["type"], "function_call");
}
