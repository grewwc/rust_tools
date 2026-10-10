use super::common::{first_openai_model_name, first_model_key_for_adapter};
use super::super::*;
use crate::ai::tools::os_tools::{GLOBAL_OS, init_os_tools_globals};
use crate::ai::{cli::ParsedCli, types::AppConfig};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};

#[test]
fn deepseek_request_body_uses_thinking_object() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    // Off: thinking={"type":"disabled"}
    let disabled = build_request_body(
        "deepseek-v4-flash-free",
        &messages,
        false,
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let disabled = serde_json::to_value(&disabled).unwrap();
    assert_eq!(
        disabled.get("thinking"),
        Some(&json!({ "type": "disabled" }))
    );
    // DeepSeek must no longer send enable_thinking (it conflicts with / is
    // ignored next to the thinking object).
    assert!(disabled.get("enable_thinking").is_none());

    // On: thinking={"type":"enabled"}
    let enabled = build_request_body(
        "deepseek-v4-flash-free",
        &messages,
        false,
        true,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let enabled = serde_json::to_value(&enabled).unwrap();
    assert_eq!(enabled.get("thinking"), Some(&json!({ "type": "enabled" })));
}

#[test]
fn non_deepseek_request_body_omits_thinking_object() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let body = build_request_body(
        "qwen3.7-plus",
        &messages,
        true,
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let value = serde_json::to_value(&body).unwrap();
    assert!(value.get("thinking").is_none());
}

#[test]
fn deepseek_tool_call_messages_echo_empty_reasoning_content() {
    let mut messages = vec![Message {
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

    normalize_reasoning_content_replay_for_model("deepseek-v4-flash-free", &mut messages);
    assert_eq!(messages[0].reasoning_content.as_deref(), Some(""));

    let value = serde_json::to_value(&messages[0]).unwrap();
    assert_eq!(
        value.get("reasoning_content").and_then(|v| v.as_str()),
        Some("")
    );
}

#[test]
fn opencode_deepseek_tool_call_messages_echo_even_without_thinking_gate() {
    let mut messages = vec![Message {
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

    // Regression: OpenCode DeepSeek always sends the `thinking` object (when
    // disabled it takes precedence over the top-level reasoning_effort); a
    // historical tool-call assistant must still get an empty reasoning_content
    // filled in, otherwise post-compression continuation requests fail with 400.
    normalize_reasoning_content_replay_for_model("deepseek-v4-flash-free-opencode", &mut messages);
    assert_eq!(messages[0].reasoning_content.as_deref(), Some(""));

    let body = build_request_body(
        "deepseek-v4-flash-free-opencode",
        &messages,
        false,
        false,
        None,
        None,
        None,
        Some("high"),
        None,
        None,
        None,
    );
    let value = serde_json::to_value(&body).unwrap();
    // Effort tier placed: the registry declares `reasoning_effort_wire:
    // "top_level"` for this entry, so the tier bypasses the dialect's
    // binary-switch omission; the disabled thinking object still takes
    // precedence for the off-switch.
    assert_eq!(value.get("reasoning_effort"), Some(&json!("high")));
    assert_eq!(
        value.pointer("/thinking/type").and_then(|v| v.as_str()),
        Some("disabled")
    );
    let echoed = value
        .get("messages")
        .and_then(|v| v.as_array())
        .and_then(|items| items.first())
        .and_then(|msg| msg.get("reasoning_content"))
        .and_then(|v| v.as_str());
    assert_eq!(echoed, Some(""));
}

#[test]
fn deepseek_non_tool_call_assistant_messages_echo_reasoning_shape() {
    // Regression: Console Go (opencode Zen) rejects multi-turn thinking-mode
    // requests whose replayed assistant final replies (no tool calls) lack the
    // `reasoning_content` field: 400 "The reasoning_content in the thinking mode
    // must be passed back". The field shape (empty string) satisfies the
    // gateway, mirroring the existing tool-call echo contract.
    let mut messages = vec![Message {
        role: "assistant".to_string(),
        content: Value::String("上一轮的最终回答".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    normalize_reasoning_content_replay_for_model("deepseek-flash-opencode", &mut messages);
    assert_eq!(messages[0].reasoning_content.as_deref(), Some(""));

    let value = serde_json::to_value(&messages[0]).unwrap();
    assert_eq!(
        value.get("reasoning_content").and_then(|v| v.as_str()),
        Some(""),
        "a missing reasoning_content must be echoed as an empty string field, not omitted"
    );
}

#[test]
fn deepseek_flagged_model_replays_own_exact_blob_and_fills_missing_shape() {
    // `deepseek-flash-opencode` declares both `reasoning_content_replay` (exact
    // blob pipeline) and the DeepSeek echo dialect. The echo must win: own-model
    // exact blobs decode back to the original provider text, while missing or
    // foreign reasoning still gets the empty field shape instead of being
    // stripped (which previously 400'd mid-turn and across turns).
    let own_blob = crate::ai::history::compress::encode_reasoning_replay_state(
        "deepseek-flash-opencode",
        "需要先读取目标文件。",
    );
    let tool_call = || Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: "call_deepseek_echo".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: Some(own_blob.clone()),
    };
    let mut own_blob_messages = vec![tool_call()];
    normalize_reasoning_content_replay_for_model("deepseek-flash-opencode", &mut own_blob_messages);
    assert_eq!(
        own_blob_messages[0].reasoning_content.as_deref(),
        Some("需要先读取目标文件。"),
        "an own exact-replay blob should decode back to the original provider text"
    );

    let mut empty_tool_call_messages = vec![Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: "call_deepseek_empty".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    }];
    normalize_reasoning_content_replay_for_model(
        "deepseek-flash-opencode",
        &mut empty_tool_call_messages,
    );
    assert_eq!(empty_tool_call_messages[0].reasoning_content.as_deref(), Some(""));

    // A GLM-origin exact blob must never leak to the DeepSeek provider: it is
    // cleared to the empty shape (cross-model semantics preserved).
    let glm_blob = crate::ai::history::compress::encode_reasoning_replay_state(
        "glm-5.3",
        "GLM 的推理文本",
    );
    let mut switched_messages = vec![Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: "call_deepseek_switched".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: Some(glm_blob),
    }];
    normalize_reasoning_content_replay_for_model(
        "deepseek-flash-opencode",
        &mut switched_messages,
    );
    assert_eq!(switched_messages[0].reasoning_content.as_deref(), Some(""));
}

#[test]
fn reasoning_content_replay_is_exact_only_for_declared_models() {
    let assistant = Message {
        role: "assistant".to_string(),
        content: json!("准备调用工具"),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: "call_glm_replay".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: Some("必须原样回放的 GLM 推理".to_string()),
    };

    let projected = crate::ai::history::compress::sanitize_message_for_persisted_history_for_model(
        "glm-5.3", &assistant,
    );
    assert_ne!(
        projected.reasoning_content, assistant.reasoning_content,
        "请求上下文投影应携带来源模型标记"
    );
    let mut glm_messages = vec![projected.clone()];
    normalize_reasoning_content_replay_for_model("glm-5.3", &mut glm_messages);
    assert_eq!(
        glm_messages[0].reasoning_content.as_deref(),
        Some("必须原样回放的 GLM 推理")
    );

    let mut volcano_glm_messages = vec![projected.clone()];
    normalize_reasoning_content_replay_for_model("gpt-5.5", &mut volcano_glm_messages);
    assert_eq!(volcano_glm_messages[0].reasoning_content, None);

    let mut untagged_glm_messages = vec![assistant.clone()];
    normalize_reasoning_content_replay_for_model("glm-5.3", &mut untagged_glm_messages);
    assert_eq!(untagged_glm_messages[0].reasoning_content, None);

    let mut deepseek_messages = vec![assistant];
    normalize_reasoning_content_replay_for_model("deepseek-v4-flash-free", &mut deepseek_messages);
    assert_eq!(
        deepseek_messages[0].reasoning_content.as_deref(),
        Some("必须原样回放的 GLM 推理"),
        "DeepSeek 的已有 reasoning 不能退化为空字符串"
    );

    let mut switched_to_deepseek = vec![projected.clone()];
    normalize_reasoning_content_replay_for_model(
        "deepseek-v4-flash-free",
        &mut switched_to_deepseek,
    );
    assert_eq!(
        switched_to_deepseek[0].reasoning_content.as_deref(),
        Some(""),
        "从 GLM 切到 DeepSeek 时不能把内部持久化标记发给 provider"
    );

    let mut gpt_messages = vec![projected];
    normalize_reasoning_content_replay_for_model("gpt-5.5", &mut gpt_messages);
    assert_eq!(gpt_messages[0].reasoning_content, None);
}

/// The encrypted reasoning-replay blob
/// (`PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX`) is protected internal
/// state: when switching to a shape-only model that requires echoing back
/// `reasoning_content` (the DeepSeek dialect), it must be cleared to an empty
/// string instead of sending the blob verbatim to the provider — matching the
/// existing GLM-exact-marker-to-DeepSeek semantics (test at lines 2007-2016
/// above). For the model the blob came from (the encrypted-replay model), the
/// else branch strips it entirely (the side-channel is rebuilt from history
/// before normalize; see transport.rs).
#[test]
fn encrypted_replay_blob_cleared_when_switching_to_shape_only_model() {
    let items = vec![serde_json::json!({
        "type": "reasoning",
        "encrypted_content": "ENC-cross-model-switch",
        "summary": []
    })];
    let blob = crate::ai::history::compress::encode_encrypted_reasoning_replay_state(
        "muse-spark-1.2-contributor",
        &items,
    );
    assert!(
        blob.starts_with(crate::ai::history::compress::PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX)
    );

    let assistant = Message {
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
        reasoning_content: Some(blob.clone()),
    };

    let mut switched_to_deepseek = vec![assistant.clone()];
    normalize_reasoning_content_replay_for_model(
        "deepseek-v4-flash-free",
        &mut switched_to_deepseek,
    );
    assert_eq!(
        switched_to_deepseek[0].reasoning_content.as_deref(),
        Some(""),
        "加密回放 blob 不能原样发给 DeepSeek provider"
    );

    // For the blob's origin model (the encrypted-replay model), the else
    // branch strips it entirely: the side-channel is rebuilt from history
    // before normalize (transport.rs), so the blob must not leak onto the wire.
    let mut origin_model_messages = vec![assistant];
    normalize_reasoning_content_replay_for_model(
        "muse-spark-1.2-contributor",
        &mut origin_model_messages,
    );
    assert_eq!(origin_model_messages[0].reasoning_content, None);
}

/// Core regression: Alibaba-provider models on the DashScope compatible-mode
/// endpoint must send `enable_thinking` per the thinking gate decision,
/// otherwise "off" is silently dropped and the model keeps reasoning. The model
/// registry changes over time, so pick an actual Alibaba model from it.
#[test]
fn dashscope_alibaba_provider_honors_enable_thinking_gate() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    let model = first_model_key_for_adapter(crate::ai::provider::ApiProvider::Alibaba)
        .expect("model registry must contain at least one Alibaba-adapter model");

    // gate off → enable_thinking:false
    let disabled = build_request_body(
        &model, &messages, false, false, None, None, None, None, None, None, None,
    );
    let disabled = serde_json::to_value(&disabled).unwrap();
    assert_eq!(
        disabled.get("enable_thinking").and_then(|v| v.as_bool()),
        Some(false),
        "{model} should emit enable_thinking:false when gate disables thinking"
    );
    // Uses enable_thinking rather than the deepseek thinking object
    assert!(disabled.get("thinking").is_none(), "{model}");

    // gate on → enable_thinking:true
    let enabled = build_request_body(
        &model, &messages, false, true, None, None, None, None, None, None, None,
    );
    let enabled = serde_json::to_value(&enabled).unwrap();
    assert_eq!(
        enabled.get("enable_thinking").and_then(|v| v.as_bool()),
        Some(true),
        "{model} should emit enable_thinking:true when gate enables thinking"
    );
}

/// Auxiliary (non-mainline) requests must explicitly disable thinking for
/// DashScope endpoint models, otherwise the default-on long reasoning chains
/// blow the aux task timeout.
#[test]
fn dashscope_aux_requests_disable_thinking_regardless_of_provider() {
    // Only DashScope (alibaba adapter) models use the enable_thinking field;
    // deepseek-v4-pro has moved to the OpenCode gateway (uses the thinking
    // object, see the assertion below), and kimi-k2.7-code is no longer in the
    // model registry (models/).
    for model in ["qwen3.7-plus", "qwen3.7-max", "deepseek-v4-flash-0731"] {
        let mut body = json!({ "model": model, "messages": [], "stream": false });
        apply_aux_thinking_fields(model, &mut body);
        assert_eq!(
            body.get("enable_thinking").and_then(|v| v.as_bool()),
            Some(false),
            "{model} aux request should disable thinking via enable_thinking:false"
        );
    }

    // OpenCode's deepseek does not rely on enable_thinking; aux turns thinking
    // off via the thinking object.
    let mut deepseek =
        json!({ "model": "deepseek-v4-flash-free", "messages": [], "stream": false });
    apply_aux_thinking_fields("deepseek-v4-flash-free", &mut deepseek);
    assert_eq!(
        deepseek.get("thinking"),
        Some(&json!({ "type": "disabled" }))
    );
    assert!(deepseek.get("enable_thinking").is_none());

    // MiniMax (mimo) on OpenCode has no reliable off switch; aux injects no
    // thinking fields at all.
    let mut mimo = json!({ "model": "mimo-v2.5-free", "messages": [], "stream": false });
    apply_aux_thinking_fields("mimo-v2.5-free", &mut mimo);
    assert!(mimo.get("thinking").is_none());
    assert!(mimo.get("enable_thinking").is_none());
}

#[test]
fn openai_request_body_omits_nonstandard_flags() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hello".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let Some(model) = first_openai_model_name() else {
        eprintln!(
            "[test] skipping openai_request_body_omits_nonstandard_flags: \
                 no OpenAi model present in model registry"
        );
        return;
    };
    let body = build_request_body(
        &model,
        &messages,
        true,
        true,
        Some(true),
        None,
        None,
        Some("high"),
        None,
        None,
        None,
    );
    let value = serde_json::to_value(&body).unwrap();

    // OpenAI-provider sends no DashScope extension fields; reasoning strength
    // goes through the top-level reasoning_effort.
    assert!(value.get("enable_thinking").is_none());
    assert!(value.get("enable_search").is_none());
    assert_eq!(
        value.get("reasoning_effort").and_then(|v| v.as_str()),
        Some("high")
    );
    assert!(value.get("reasoning").is_none());
    assert_eq!(
        value.get("model").and_then(|v| v.as_str()),
        Some(model.as_str())
    );
}

#[test]
fn alibaba_request_body_keeps_extension_flags() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hello".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let body = build_request_body(
        "qwen3.7-plus",
        &messages,
        false,
        true,
        Some(true),
        None,
        None,
        Some("high"),
        None,
        None,
        None,
    );
    let value = serde_json::to_value(&body).unwrap();

    assert_eq!(
        value.get("enable_thinking").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        value.get("enable_search").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        value.get("reasoning_effort").and_then(|v| v.as_str()),
        Some("high")
    );
    assert!(value.get("reasoning").is_none());
}
