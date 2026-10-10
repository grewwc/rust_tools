use super::common::{first_openai_vl_model_name, first_alibaba_vl_model_name};
use super::super::*;
use crate::ai::tools::os_tools::{GLOBAL_OS, init_os_tools_globals};
use crate::ai::{cli::ParsedCli, types::AppConfig};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};

#[test]
fn openai_image_content_uses_object_image_url_shape() {
    // The contract "images are sent in the {image_url:{url:...}} object shape"
    // can only be verified when the model registry (models/) contains an
    // OpenAi-provider model with is_vl=true.
    // When no such model exists in the real environment (e.g. the registry only
    // has a Compatible VL), this contract cannot be verified and the test is
    // simply skipped.
    let Some(model) = first_openai_vl_model_name() else {
        eprintln!(
            "[test] skipping openai_image_content_uses_object_image_url_shape: \
                 no OpenAi+VL model present in model registry"
        );
        return;
    };

    let path = std::env::temp_dir().join(format!("ai-openai-image-{}.png", uuid::Uuid::new_v4()));
    std::fs::write(&path, b"fake").unwrap();

    let value = build_content(&model, "describe", &[path.to_string_lossy().to_string()]).unwrap();

    let first = value.as_array().and_then(|items| items.first()).unwrap();
    assert_eq!(
        first.get("type").and_then(|v| v.as_str()),
        Some("image_url")
    );
    assert!(
        first
            .get("image_url")
            .and_then(|v| v.get("url"))
            .and_then(|v| v.as_str())
            .map(|s| s.starts_with("data:image/png;base64,"))
            .unwrap_or(false)
    );
}

#[test]
fn alibaba_image_content_also_uses_object_image_url_shape() {
    let Some(model) = first_alibaba_vl_model_name() else {
        eprintln!(
            "[test] skipping alibaba_image_content_also_uses_object_image_url_shape: \
                 no Alibaba+VL model present in model registry"
        );
        return;
    };

    let path = std::env::temp_dir().join(format!("ai-alibaba-image-{}.png", uuid::Uuid::new_v4()));
    std::fs::write(&path, b"fake").unwrap();

    let value = build_content(&model, "describe", &[path.to_string_lossy().to_string()]).unwrap();

    let first = value.as_array().and_then(|items| items.first()).unwrap();
    assert_eq!(
        first.get("type").and_then(|v| v.as_str()),
        Some("image_url")
    );
    assert!(
        first
            .get("image_url")
            .and_then(|v| v.get("url"))
            .and_then(|v| v.as_str())
            .map(|s| s.starts_with("data:image/png;base64,"))
            .unwrap_or(false)
    );
}

#[test]
fn responses_request_body_converts_chat_multimodal_content_to_input_items() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::Array(vec![
            serde_json::json!({
                "type": "image_url",
                "image_url": { "url": "data:image/png;base64,AAAA" }
            }),
            serde_json::json!({
                "type": "text",
                "text": "please explain"
            }),
        ]),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    let request = RequestBody {
        model: "gpt-test".to_string(),
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
    let content = body
        .get("input")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(|item| item.get("content"))
        .and_then(Value::as_array)
        .expect("responses body should contain array content");

    assert_eq!(
        content[0].get("type").and_then(Value::as_str),
        Some("input_image")
    );
    assert_eq!(
        content[0].get("image_url").and_then(Value::as_str),
        Some("data:image/png;base64,AAAA")
    );
    assert_eq!(
        content[1].get("type").and_then(Value::as_str),
        Some("input_text")
    );
    assert_eq!(
        content[1].get("text").and_then(Value::as_str),
        Some("please explain")
    );
}

#[test]
fn normalize_messages_flattens_internal_note_multimodal_content_to_text_only_system_note() {
    let messages = vec![Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::Array(vec![
            serde_json::json!({
                "type": "image_url",
                "image_url": { "url": "data:image/png;base64,AAAA" }
            }),
            serde_json::json!({
                "type": "text",
                "text": "[Process 1 Woke Up] resume now"
            }),
        ]),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    let normalized = normalize_messages_for_request(&messages);
    assert_eq!(normalized.len(), 1);
    assert_eq!(normalized[0].role, "system");
    assert_eq!(
        normalized[0].content,
        Value::String("[image omitted]\n[Process 1 Woke Up] resume now".to_string())
    );
}

#[test]
fn responses_request_body_does_not_emit_input_image_for_system_multimodal_content() {
    let messages = vec![Message {
        role: "system".to_string(),
        content: Value::Array(vec![
            serde_json::json!({
                "type": "image_url",
                "image_url": { "url": "data:image/png;base64,AAAA" }
            }),
            serde_json::json!({
                "type": "text",
                "text": "resume instructions"
            }),
        ]),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    let normalized = normalize_messages_for_request(&messages);
    let request = RequestBody {
        model: "gpt-test".to_string(),
        messages: &normalized,
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
    let content = body
        .get("input")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(|item| item.get("content"))
        .and_then(Value::as_array)
        .expect("responses body should contain array content");

    assert_eq!(content.len(), 1);
    assert_eq!(
        content[0].get("type").and_then(Value::as_str),
        Some("input_text")
    );
    assert_eq!(
        content[0].get("text").and_then(Value::as_str),
        Some("[image omitted]\nresume instructions")
    );
}

#[test]
fn normalize_messages_downgrades_image_content_for_text_only_models() {
    let Some(model) = crate::ai::model_names::all()
        .iter()
        .find(|m| !m.is_vl)
        .map(|m| m.name.clone())
    else {
        eprintln!(
            "[test] skipping normalize_messages_downgrades_image_content_for_text_only_models: no text-only model present in model registry"
        );
        return;
    };

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String("base system".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::Array(vec![
                serde_json::json!({
                    "type": "image_url",
                    "image_url": { "url": "data:image/png;base64,AAAA" }
                }),
                serde_json::json!({
                    "type": "text",
                    "text": "please explain"
                }),
            ]),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_model(&model, &messages);

    assert!(
        normalized
            .iter()
            .all(|message| !matches!(message.content, Value::Array(_)))
    );
    let content = normalized[1].content.as_str().unwrap();
    assert!(content.contains("[image omitted]"));
    assert!(content.contains("please explain"));
}

#[test]
fn normalize_messages_drops_path_like_historical_tool_call_names() {
    let messages = vec![
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![crate::ai::types::ToolCall {
                id: "call_bad_name".to_string(),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "stream/splitter.rs".to_string(),
                    arguments: r#"{"path":"stream/splitter.rs"}"#.to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "tool".to_string(),
            content: Value::String("source contents".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_bad_name".to_string()),
            reasoning_content: None,
        },
    ];

    let normalized = normalize_messages_for_request(&messages);

    assert!(normalized.iter().all(|message| {
        message.tool_calls.as_ref().is_none_or(|calls| {
            calls
                .iter()
                .all(|call| call.function.name != "stream/splitter.rs")
        })
    }));
    assert!(normalized.iter().all(|message| message.role != "tool"));
    assert!(normalized.iter().any(|message| {
        message.role == "system"
            && message
                .content
                .as_str()
                .is_some_and(|content| content.contains("source contents"))
    }));
}
