use std::sync::Arc;

use super::super::builder::RequestModelInputs;
use super::super::protocol::build_http_body_for_json_messages_from_parts;
use super::super::*;
use crate::ai::provider::ApiProvider;
use serde_json::Value;

#[test]
fn responses_request_body_replays_plain_and_tool_turns_in_order_and_counts_each_owner() {
    use super::super::builder::{
        build_request_body_from_parts_with_replay_owners, measure_request_prompt,
        measure_request_prompt_from_definitions,
    };
    use super::super::reasoning::{
        SplicedReplayKeys, normalize_reasoning_content_replay_for_model,
        reconstruct_encrypted_reasoning_items_for_model_recorded,
    };
    use crate::ai::history::compress::encode_encrypted_reasoning_replay_state;
    use crate::ai::types::{FunctionCall, ToolCall};
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let model = crate::ai::model_names::all()
        .iter()
        .find(|entry| {
            entry.reasoning_encrypted_replay
                && entry.request_protocol == Some(RequestProtocolDialect::Responses)
                && !crate::ai::models::reasoning_content_replay_enabled(&entry.key)
        })
        .expect("registry must contain a Responses encrypted replay model")
        .key
        .clone();
    let item = |id: &str| json!({"type":"reasoning", "id":id, "encrypted_content":"A".repeat(4_000), "summary":[]});
    let first = item("rs_wire_plain_first");
    let second = item("rs_wire_plain_second");
    let tool = item("rs_wire_tool");
    let plain = |item: &Value, empty_calls: bool| Message {
        role: "assistant".to_string(),
        content: json!("same answer"),
        tool_calls: empty_calls.then(Vec::new),
        tool_call_id: None,
        reasoning_content: Some(encode_encrypted_reasoning_replay_state(
            &model,
            None,
            &[item.clone()],
        )),
    };
    let user = |text: &str| Message {
        role: "user".to_string(),
        content: json!(text),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    };
    let mut messages = vec![
        user("first question"),
        plain(&first, false),
        user("second question"),
        plain(&second, true),
        Message {
            role: "assistant".to_string(),
            content: json!("tool narration"),
            tool_calls: Some(vec![ToolCall {
                id: "call-mixed-wire".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: Some(encode_encrypted_reasoning_replay_state(
                &model,
                None,
                &[tool.clone()],
            )),
        },
        Message {
            role: "tool".to_string(),
            content: json!("tool result"),
            tool_calls: None,
            tool_call_id: Some("call-mixed-wire".to_string()),
            reasoning_content: None,
        },
        user("following question"),
    ];
    let mut spliced = SplicedReplayKeys::default();
    let mut items = reconstruct_encrypted_reasoning_items_for_model_recorded(
        &model,
        "https://plain-wire-replay.test/v1",
        "plain-wire-replay",
        std::path::Path::new(""),
        &messages,
        &Default::default(),
        &mut spliced,
    );
    assert_ne!(spliced.message_owners()[&1], spliced.message_owners()[&3]);
    items.insert(
        "compressed-away".to_string(),
        Arc::new(vec![item("rs_unprojected")]),
    );
    normalize_reasoning_content_replay_for_model(&model, &mut messages);
    assert!(
        messages
            .iter()
            .all(|message| message.reasoning_content.is_none())
    );
    let request = build_request_body_from_parts_with_replay_owners(
        &RequestModelInputs::resolve(&model),
        &messages,
        false,
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(&items),
        Some(spliced.message_owners()),
    );
    let body = build_responses_request_body(&request);
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 11);
    assert_eq!(input[0]["content"][0]["text"], "first question");
    assert_eq!(input[1], first);
    assert_eq!(input[2]["role"], "assistant");
    assert_eq!(input[2]["content"][0]["type"], "output_text");
    assert_eq!(input[2]["content"][0]["text"], "same answer");
    assert_eq!(input[3]["content"][0]["text"], "second question");
    assert_eq!(input[4], second);
    assert_eq!(input[5]["content"][0]["text"], "same answer");
    assert_eq!(input[6], tool);
    assert_eq!(input[7]["content"][0]["text"], "tool narration");
    assert_eq!(input[8]["type"], "function_call");
    assert_eq!(input[8]["call_id"], "call-mixed-wire");
    assert_eq!(input[9]["type"], "function_call_output");
    assert_eq!(input[10]["content"][0]["text"], "following question");
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    let empty_payloads = items
        .iter()
        .map(|(owner, items)| {
            let stripped = items
                .iter()
                .cloned()
                .map(|mut item| {
                    item["encrypted_content"] = json!("");
                    item
                })
                .collect();
            (owner.clone(), Arc::new(stripped))
        })
        .collect();
    let empty_counts = measure_request_prompt(
        &messages,
        None,
        Some(&empty_payloads),
        Some(spliced.message_owners()),
    );
    // Each 4000-byte base64 payload bills ceil((3000 - 650) / 4) tokens;
    // the upper bound uses divisor 2. The unprojected owner is never billed.
    assert_eq!(
        request.estimated_prompt_tokens - empty_counts.lower_tokens(),
        3 * 588
    );
    assert_eq!(
        request.estimated_prompt_tokens_upper - empty_counts.upper_tokens(),
        3 * 1_175
    );
    let preview = measure_request_prompt_from_definitions(
        &messages,
        None,
        Some(&items),
        Some(spliced.message_owners()),
    );
    assert_eq!(request.estimated_prompt_tokens, preview.lower_tokens());
    assert_eq!(
        request.estimated_prompt_tokens_upper,
        preview.upper_tokens()
    );
    assert!(
        serde_json::to_value(&request)
            .unwrap()
            .get("reasoning_replay_owners")
            .is_none()
    );
}

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
fn json_messages_aux_body_uses_responses_protocol_for_responses_dialect() {
    // Wire behavior pinned on synthetic inputs: the Responses shape comes from
    // the pre-resolved dialect, not from any registry key.
    let inputs = RequestModelInputs {
        request_model: "synthetic-responses-model".to_string(),
        endpoint: "https://modelhub.example.com/v1".to_string(),
        adapter: ApiProvider::Compatible,
        effort_wire: None,
        dialect: RequestProtocolDialect::Responses,
        max_output: None,
        encrypted_replay: false,
        context_window: 256_000,
    };
    let messages = vec![
        json!({"role": "system", "content": "Return JSON only."}),
        json!({"role": "user", "content": "classify this"}),
    ];

    let body_bytes =
        build_http_body_for_json_messages_from_parts(&inputs, &messages, false, None, false);
    let body: Value = serde_json::from_slice(&body_bytes).expect("body should be valid JSON");

    assert_eq!(body["model"], "synthetic-responses-model");
    assert!(body.get("messages").is_none());
    assert_eq!(body["input"][0]["role"], "system");
    assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
    assert_eq!(body["input"][0]["content"][0]["text"], "Return JSON only.");
    assert_eq!(body["input"][1]["role"], "user");
    assert_eq!(body["input"][1]["content"][0]["text"], "classify this");

    // Registry coverage: the Responses class the wire pin above stands for
    // must keep existing, or the pin tests a dead path.
    assert!(
        crate::ai::model_names::all()
            .iter()
            .any(|m| m.request_protocol == Some(RequestProtocolDialect::Responses)),
        "registry must keep at least one Responses-protocol entry"
    );
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
        reasoning_replay_owners: None,
        reasoning_encrypted_replay: false,
        estimated_prompt_tokens: 0,
        estimated_prompt_tokens_upper: 0,
        message_counts: Default::default(),
        prompt_cache_key: None,
        explicit_cache_breakpoints: false,
        tool_errors: None,
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
fn responses_request_body_emits_prompt_cache_key_only_when_set() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let mut request = RequestBody {
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
        reasoning_replay_owners: None,
        reasoning_encrypted_replay: false,
        estimated_prompt_tokens: 0,
        estimated_prompt_tokens_upper: 0,
        message_counts: Default::default(),
        prompt_cache_key: None,
        explicit_cache_breakpoints: false,
        tool_errors: None,
    };
    let body = super::super::build_responses_request_body(&request);
    assert!(body.get("prompt_cache_key").is_none());
    // Never leaks into chat-completions bodies, which serialize RequestBody directly.
    let chat: Value = serde_json::to_value(&request).unwrap();
    assert!(chat.get("prompt_cache_key").is_none());

    request.prompt_cache_key = Some("session-1".to_string());
    let body = super::super::build_responses_request_body(&request);
    assert_eq!(body["prompt_cache_key"], "session-1");
    let chat: Value = serde_json::to_value(&request).unwrap();
    assert!(chat.get("prompt_cache_key").is_none());
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
        reasoning_replay_owners: None,
        reasoning_encrypted_replay: false,
        estimated_prompt_tokens: 0,
        estimated_prompt_tokens_upper: 0,
        message_counts: Default::default(),
        prompt_cache_key: None,
        explicit_cache_breakpoints: false,
        tool_errors: None,
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
        reasoning_replay_owners: None,
        reasoning_encrypted_replay: false,
        estimated_prompt_tokens: 0,
        estimated_prompt_tokens_upper: 0,
        message_counts: Default::default(),
        prompt_cache_key: None,
        explicit_cache_breakpoints: false,
        tool_errors: None,
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
    let inputs = RequestModelInputs {
        request_model: "synthetic-responses-model".to_string(),
        endpoint: "https://modelhub.example.com/v1".to_string(),
        adapter: ApiProvider::Compatible,
        effort_wire: None,
        dialect: RequestProtocolDialect::Responses,
        max_output: None,
        encrypted_replay: true,
        context_window: 256_000,
    };
    let request = super::super::builder::build_request_body_from_parts(
        &inputs, &messages, false, false, None, None, None, None, None, None, None,
    );
    assert!(
        request.reasoning_encrypted_replay,
        "inputs with encrypted_replay=true must set the request flag"
    );

    let body = super::super::build_responses_request_body(&request);
    assert_eq!(
        body["include"],
        json!(["reasoning.encrypted_content"]),
        "capable model must request encrypted reasoning include"
    );

    // Registry coverage: at least one entry must keep declaring the flag, or
    // the wire pin above tests a dead path.
    assert!(
        crate::ai::model_names::all()
            .iter()
            .any(|m| m.reasoning_encrypted_replay),
        "registry must keep at least one reasoning_encrypted_replay entry"
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
        reasoning_replay_owners: None,
        reasoning_encrypted_replay: false,
        estimated_prompt_tokens: 0,
        estimated_prompt_tokens_upper: 0,
        message_counts: Default::default(),
        prompt_cache_key: None,
        explicit_cache_breakpoints: false,
        tool_errors: None,
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
        Arc::new(vec![json!({
            "type": "reasoning",
            "id": "rs_abc",
            "encrypted_content": "ENC",
            "summary": [],
        })]),
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
        reasoning_replay_owners: None,
        reasoning_encrypted_replay: true,
        estimated_prompt_tokens: 0,
        estimated_prompt_tokens_upper: 0,
        message_counts: Default::default(),
        prompt_cache_key: None,
        explicit_cache_breakpoints: false,
        tool_errors: None,
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
        reasoning_replay_owners: None,
        reasoning_encrypted_replay: true,
        estimated_prompt_tokens: 0,
        estimated_prompt_tokens_upper: 0,
        message_counts: Default::default(),
        prompt_cache_key: None,
        explicit_cache_breakpoints: false,
        tool_errors: None,
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
        reasoning_replay_owners: None,
        reasoning_encrypted_replay: false,
        estimated_prompt_tokens: 0,
        estimated_prompt_tokens_upper: 0,
        message_counts: Default::default(),
        prompt_cache_key: None,
        explicit_cache_breakpoints: false,
        tool_errors: None,
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
        Arc::new(vec![json!({
            "type": "reasoning",
            "id": "rs_abc",
            "encrypted_content": "ENC",
            "summary": [],
        })]),
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
        reasoning_replay_owners: None,
        reasoning_encrypted_replay: true,
        estimated_prompt_tokens: 0,
        estimated_prompt_tokens_upper: 0,
        message_counts: Default::default(),
        prompt_cache_key: None,
        explicit_cache_breakpoints: false,
        tool_errors: None,
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
