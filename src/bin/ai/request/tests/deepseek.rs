use super::super::builder::{RequestModelInputs, build_request_body_from_parts};
use super::super::reasoning::{
    normalize_reasoning_content_replay_from_parts, resolve_reasoning_wire_controls_from_parts,
};
use super::super::*;
use crate::ai::provider::{ALIBABA_DEFAULT_ENDPOINT, OPENCODE_DEFAULT_ENDPOINT, ApiProvider};
use serde_json::Value;

/// Pre-resolved inputs keep wire assertions independent of registry renames.
/// `resolve_reasoning_wire_controls_from_parts` in `request/reasoning.rs`
/// combines provider dialect selection with explicit effort wire placement.
fn synthetic_inputs(
    request_model: &str,
    endpoint: &str,
    adapter: ApiProvider,
    effort_wire: Option<crate::ai::model_names::ReasoningEffortWire>,
) -> RequestModelInputs {
    RequestModelInputs {
        request_model: request_model.to_string(),
        endpoint: endpoint.to_string(),
        adapter,
        effort_wire,
        dialect: RequestProtocolDialect::ChatCompletions,
        max_output: None,
        encrypted_replay: false,
        context_window: 256_000,
    }
}

const TOP_LEVEL: Option<crate::ai::model_names::ReasoningEffortWire> =
    Some(crate::ai::model_names::ReasoningEffortWire::TopLevel);
const OPENAI_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";
const OPENCODE_RESPONSES_ENDPOINT: &str = "https://opencode.ai/inference/go/openai/v1/responses";

fn user_hi() -> Vec<Message> {
    vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }]
}

/// Registry-free twin of production `apply_aux_thinking_fields`: resolve the
/// dialect controls with thinking disabled and no effort tier, then merge the
/// dialect's off fields into the body. The effort-wire input cannot change the
/// off fields (dialects only consult effort when the switch is on), so
/// production's registry-declared wire and this `None` produce identical
/// bodies.
fn apply_aux_thinking_fields_from_parts(
    adapter: ApiProvider,
    request_model: &str,
    endpoint: &str,
    body: &mut Value,
) {
    let (fields, _, _) = resolve_reasoning_wire_controls_from_parts(
        adapter,
        request_model,
        endpoint,
        None,
        false,
        None,
    );
    if fields.is_empty() {
        return;
    }
    if let Some(map) = body.as_object_mut() {
        for (key, value) in fields {
            map.insert(key, value);
        }
    }
}

fn opencode_deepseek_entries_exist() {
    assert!(
        crate::ai::model_names::all().iter().any(|m| {
            m.adapter == ApiProvider::OpenCode
                && !m.name.starts_with("enc:")
                && m.name.to_ascii_lowercase().contains("deepseek")
        }),
        "registry must contain a plaintext OpenCode DeepSeek entry"
    );
}

#[test]
fn deepseek_request_body_uses_thinking_object() {
    // OpenCode dispatch selects the DeepSeek thinking object for a DeepSeek
    // request model.
    const REQUEST_MODEL: &str = "synthetic-deepseek-v4";
    let messages = user_hi();
    let inputs = synthetic_inputs(
        REQUEST_MODEL,
        OPENCODE_DEFAULT_ENDPOINT,
        ApiProvider::OpenCode,
        None,
    );

    let off = build_request_body_from_parts(
        &inputs,
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
    let off = serde_json::to_value(&off).unwrap();
    assert_eq!(
        off.get("thinking"),
        Some(&json!({ "type": "disabled" }))
    );
    assert!(off.get("enable_thinking").is_none());

    let on = build_request_body_from_parts(
        &inputs,
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
    let on = serde_json::to_value(&on).unwrap();
    assert_eq!(on.get("thinking"), Some(&json!({ "type": "enabled" })));
    assert!(on.get("enable_thinking").is_none());

    opencode_deepseek_entries_exist();
}

#[test]
fn non_deepseek_request_body_omits_thinking_object() {
    // Alibaba on DashScope uses `enable_thinking` and omits the DeepSeek
    // `thinking` object.
    let messages = user_hi();
    let inputs = synthetic_inputs(
        "synthetic-qwen",
        ALIBABA_DEFAULT_ENDPOINT,
        ApiProvider::Alibaba,
        None,
    );

    let body = build_request_body_from_parts(
        &inputs,
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
    let body = serde_json::to_value(&body).unwrap();
    assert!(body.get("thinking").is_none());

    assert!(
        crate::ai::model_names::all()
            .iter()
            .any(|m| m.adapter == ApiProvider::Alibaba),
        "registry must contain an Alibaba-adapter entry"
    );
}

#[test]
fn deepseek_tool_call_messages_echo_empty_reasoning_content() {
    // The DeepSeek thinking dialect (OpenCode Zen / official api.deepseek.com)
    // requires assistant tool-call messages to carry reasoning_content. When
    // the message carries none, normalize fills an empty string.
    const TAG: &str = "synthetic-deepseek-v4";
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

    normalize_reasoning_content_replay_from_parts(
        TAG,
        true,
        ApiProvider::OpenCode,
        TAG,
        OPENCODE_DEFAULT_ENDPOINT,
        &mut messages,
    );
    assert_eq!(messages[0].reasoning_content.as_deref(), Some(""));

    opencode_deepseek_entries_exist();
}

#[test]
fn opencode_deepseek_tool_call_messages_echo_even_without_thinking_gate() {
    // DeepSeek assistant messages retain the `reasoning_content` echo shape
    // even when the thinking gate is off. Explicit top-level effort placement
    // is tested separately from the disabled thinking object.
    const TAG: &str = "synthetic-deepseek-v4";
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

    let mut wire_messages = messages.clone();
    normalize_reasoning_content_replay_from_parts(
        TAG,
        true,
        ApiProvider::OpenCode,
        TAG,
        OPENCODE_DEFAULT_ENDPOINT,
        &mut wire_messages,
    );
    assert_eq!(wire_messages[0].reasoning_content.as_deref(), Some(""));

    let inputs = synthetic_inputs(TAG, OPENCODE_DEFAULT_ENDPOINT, ApiProvider::OpenCode, TOP_LEVEL);
    let body = build_request_body_from_parts(
        &inputs,
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
    let body = serde_json::to_value(&body).unwrap();
    assert_eq!(body.get("reasoning_effort"), Some(&json!("high")));
    let thinking = body.get("thinking");
    assert_eq!(
        thinking.and_then(|t| t.get("type")),
        Some(&json!("disabled"))
    );

    // Registry coverage: the mirrored class (OpenCode DeepSeek with exact
    // replay) must still exist.
    assert!(
        crate::ai::model_names::all().iter().any(|m| {
            m.adapter == ApiProvider::OpenCode
                && !m.name.starts_with("enc:")
                && m.name.to_ascii_lowercase().contains("deepseek")
                && models::reasoning_content_replay_enabled(&m.key)
        }),
        "registry must contain an OpenCode DeepSeek entry with exact replay"
    );
}

#[test]
fn deepseek_non_tool_call_assistant_messages_echo_reasoning_shape() {
    // DeepSeek models served via gateways that require the reasoning_content
    // field also expect the shape on plain assistant messages: normalize fills
    // "" so the wire body keeps the echo contract for the whole turn, not just
    // tool-call turns.
    const TAG: &str = "synthetic-deepseek-v4";
    let mut messages = vec![
        Message {
            role: "user".to_string(),
            content: Value::String("hi".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("done".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    normalize_reasoning_content_replay_from_parts(
        TAG,
        true,
        ApiProvider::OpenCode,
        TAG,
        OPENCODE_DEFAULT_ENDPOINT,
        &mut messages,
    );
    assert_eq!(messages[1].reasoning_content.as_deref(), Some(""));

    opencode_deepseek_entries_exist();
}

#[test]
fn deepseek_flagged_model_replays_own_exact_blob_and_fills_missing_shape() {
    // Declared exact replay decodes the model's own tagged blob. Missing or
    // foreign state becomes the required empty echo shape, keeping internal
    // markers out of the provider field.
    const TAG: &str = "synthetic-deepseek-v4";
    const FOREIGN_TAG: &str = "synthetic-glm";
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
        reasoning_content: None,
    };

    let reasoning = "需要先读取目标文件。".to_string();
    let own_blob = crate::ai::history::compress::encode_reasoning_replay_state(TAG, &reasoning);
    let mut own = vec![Message {
        reasoning_content: Some(own_blob),
        ..tool_call()
    }];
    normalize_reasoning_content_replay_from_parts(
        TAG,
        true,
        ApiProvider::OpenCode,
        TAG,
        OPENCODE_DEFAULT_ENDPOINT,
        &mut own,
    );
    // The flagged model decodes its own persisted blob back to the original
    // provider reasoning text.
    assert_eq!(own[0].reasoning_content, Some(reasoning));

    let mut missing = vec![tool_call()];
    normalize_reasoning_content_replay_from_parts(
        TAG,
        true,
        ApiProvider::OpenCode,
        TAG,
        OPENCODE_DEFAULT_ENDPOINT,
        &mut missing,
    );
    // No persisted continuation: fill the echo shape so the gateway accepts
    // the turn.
    assert_eq!(missing[0].reasoning_content.as_deref(), Some(""));

    let foreign_blob =
        crate::ai::history::compress::encode_reasoning_replay_state(FOREIGN_TAG, "foreign reasoning");
    let mut foreign = vec![Message {
        reasoning_content: Some(foreign_blob),
        ..tool_call()
    }];
    normalize_reasoning_content_replay_from_parts(
        TAG,
        true,
        ApiProvider::OpenCode,
        TAG,
        OPENCODE_DEFAULT_ENDPOINT,
        &mut foreign,
    );
    // Foreign (cross-model) blobs never leak verbatim: the echo field carries
    // an empty shape instead.
    assert_eq!(foreign[0].reasoning_content.as_deref(), Some(""));

    // Registry coverage: the mirrored class must still exist.
    assert!(
        crate::ai::model_names::all().iter().any(|m| {
            m.adapter == ApiProvider::OpenCode
                && !m.name.starts_with("enc:")
                && m.name.to_ascii_lowercase().contains("deepseek")
                && models::reasoning_content_replay_enabled(&m.key)
        }),
        "registry must contain an OpenCode DeepSeek entry with exact replay"
    );
}

#[test]
fn reasoning_content_replay_is_exact_only_for_declared_models() {
    // Synthetic tags cover declared exact replay without a DeepSeek echo,
    // a model without replay, and the DeepSeek echo dialect.
    const GLM_TAG: &str = "synthetic-glm";
    const GPT_TAG: &str = "synthetic-gpt";
    const DEEPSEEK_TAG: &str = "synthetic-deepseek-v4";

    let assistant = || Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
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

    // The exact-replay model tags its tool-call reasoning with its own source
    // tag in the persisted projection.
    let projected =
        crate::ai::history::compress::sanitize_message_for_persisted_history_inner(
            &assistant(),
            Some(GLM_TAG),
        );
    assert_ne!(projected.reasoning_content, assistant().reasoning_content);

    // The declaring model decodes its own blob back to the original text.
    let mut own = vec![projected.clone()];
    normalize_reasoning_content_replay_from_parts(
        GLM_TAG,
        true,
        ApiProvider::OpenCode,
        GLM_TAG,
        OPENCODE_DEFAULT_ENDPOINT,
        &mut own,
    );
    assert_eq!(
        own[0].reasoning_content,
        Some("必须原样回放的 GLM 推理".to_string())
    );

    // Models without the replay bit never carry hidden reasoning to the wire:
    // the blob (even tagged) is dropped.
    let mut stripped = vec![projected.clone()];
    normalize_reasoning_content_replay_from_parts(
        GPT_TAG,
        false,
        ApiProvider::OpenAi,
        GPT_TAG,
        OPENAI_ENDPOINT,
        &mut stripped,
    );
    assert_eq!(stripped[0].reasoning_content, None);

    // Untagged reasoning never replays on the exact-replay model either:
    // without a source tag the runtime cannot attribute it.
    let mut untagged = vec![assistant()];
    normalize_reasoning_content_replay_from_parts(
        GLM_TAG,
        true,
        ApiProvider::OpenCode,
        GLM_TAG,
        OPENCODE_DEFAULT_ENDPOINT,
        &mut untagged,
    );
    assert_eq!(untagged[0].reasoning_content, None);

    // A foreign exact-replay blob clears to the required empty echo shape;
    // untagged reasoning passes through. Internal markers must not reach the
    // provider's reasoning field.
    let mut switched_to_deepseek = vec![projected.clone()];
    normalize_reasoning_content_replay_from_parts(
        DEEPSEEK_TAG,
        true,
        ApiProvider::OpenCode,
        DEEPSEEK_TAG,
        OPENCODE_DEFAULT_ENDPOINT,
        &mut switched_to_deepseek,
    );
    assert_eq!(
        switched_to_deepseek[0].reasoning_content.as_deref(),
        Some("")
    );

    let mut deepseek_untagged = vec![assistant()];
    normalize_reasoning_content_replay_from_parts(
        DEEPSEEK_TAG,
        true,
        ApiProvider::OpenCode,
        DEEPSEEK_TAG,
        OPENCODE_DEFAULT_ENDPOINT,
        &mut deepseek_untagged,
    );
    assert_eq!(
        deepseek_untagged[0].reasoning_content,
        Some("必须原样回放的 GLM 推理".to_string())
    );

    // Registry coverage: the exact-replay class must still exist (the wire
    // shapes above pin what it resolves to).
    assert!(
        crate::ai::model_names::all()
            .iter()
            .any(|m| models::reasoning_content_replay_enabled(&m.key)),
        "registry must contain an entry declaring exact reasoning replay"
    );
    opencode_deepseek_entries_exist();
}

/// The encrypted reasoning-replay blob
/// (`PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX`) is protected internal
/// state: when switching to a shape-only model that requires echoing back
/// `reasoning_content` (the DeepSeek dialect), it must be cleared to an empty
/// string instead of sending the blob verbatim to the provider — matching the
/// existing GLM-exact-marker-to-DeepSeek semantics. For the model the blob came
/// from (the encrypted-replay model), the else branch strips it entirely (the
/// side-channel is rebuilt from history before normalize; see transport.rs).
#[test]
fn encrypted_replay_blob_cleared_when_switching_to_shape_only_model() {
    // The Responses origin does not need a DeepSeek `reasoning_content` echo
    // field; encrypted replay uses a separate input-item channel.
    const ORIGIN_TAG: &str = "synthetic-muse-spark";
    const DEEPSEEK_TAG: &str = "synthetic-deepseek-v4";
    let items = vec![json!({"id": "rs_1"})];
    let blob = crate::ai::history::compress::encode_encrypted_reasoning_replay_state(
        ORIGIN_TAG,
        None,
        &items,
    );
    assert!(blob.starts_with(
        crate::ai::history::compress::PERSISTED_ENCRYPTED_REASONING_REPLAY_PREFIX
    ));

    let tool_call = || Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: "call_enc_replay".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: Some(blob.clone()),
    };

    let mut switched = vec![tool_call()];
    normalize_reasoning_content_replay_from_parts(
        DEEPSEEK_TAG,
        true,
        ApiProvider::OpenCode,
        DEEPSEEK_TAG,
        OPENCODE_DEFAULT_ENDPOINT,
        &mut switched,
    );
    assert_eq!(switched[0].reasoning_content.as_deref(), Some(""));

    // The blob's own model strips the marker entirely (the side-channel is
    // rebuilt from history before normalize).
    let mut origin = vec![tool_call()];
    normalize_reasoning_content_replay_from_parts(
        ORIGIN_TAG,
        false,
        ApiProvider::OpenCode,
        ORIGIN_TAG,
        OPENCODE_RESPONSES_ENDPOINT,
        &mut origin,
    );
    assert_eq!(origin[0].reasoning_content, None);

    // Registry coverage: the encrypted-replay class must still exist.
    assert!(
        crate::ai::model_names::all()
            .iter()
            .any(|m| models::reasoning_encrypted_replay_enabled(&m.key)),
        "registry must contain an entry declaring encrypted reasoning replay"
    );
}

/// Core regression: Alibaba-provider models on the DashScope compatible-mode
/// endpoint must send `enable_thinking` per the thinking gate decision,
/// otherwise "off" is silently dropped and the model keeps reasoning.
#[test]
fn dashscope_alibaba_provider_honors_enable_thinking_gate() {
    let messages = user_hi();
    let inputs = synthetic_inputs(
        "synthetic-qwen-gate",
        ALIBABA_DEFAULT_ENDPOINT,
        ApiProvider::Alibaba,
        None,
    );

    let off = build_request_body_from_parts(
        &inputs, &messages, false, false, None, None, None, None, None, None, None,
    );
    let off = serde_json::to_value(&off).unwrap();
    assert_eq!(off.get("enable_thinking"), Some(&json!(false)));
    assert!(off.get("thinking").is_none());

    let on = build_request_body_from_parts(
        &inputs, &messages, false, true, None, None, None, None, None, None, None,
    );
    let on = serde_json::to_value(&on).unwrap();
    assert_eq!(on.get("enable_thinking"), Some(&json!(true)));
    assert!(on.get("thinking").is_none());

    assert!(
        crate::ai::model_names::all()
            .iter()
            .any(|m| m.adapter == ApiProvider::Alibaba),
        "registry must contain an Alibaba-adapter entry"
    );
}

/// Auxiliary (non-mainline) requests must explicitly disable thinking for
/// DashScope endpoint models, otherwise the default-on long reasoning chains
/// blow the aux task timeout.
#[test]
fn dashscope_aux_requests_disable_thinking_regardless_of_provider() {
    // Alibaba dispatch uses `enable_thinking` even for DeepSeek request names;
    // the model name does not override the adapter's switch dialect.
    for (request_model, endpoint) in [
        ("synthetic-qwen-max", ALIBABA_DEFAULT_ENDPOINT),
        ("synthetic-qwen-flash", ALIBABA_DEFAULT_ENDPOINT),
        ("synthetic-dashscope-deepseek", ALIBABA_DEFAULT_ENDPOINT),
    ] {
        let mut body = json!({"model": request_model});
        apply_aux_thinking_fields_from_parts(
            ApiProvider::Alibaba,
            request_model,
            endpoint,
            &mut body,
        );
        assert_eq!(
            body.get("enable_thinking"),
            Some(&json!(false)),
            "{request_model} aux request should disable thinking via enable_thinking:false"
        );
        assert!(body.get("thinking").is_none());
    }

    // The OpenCode DeepSeek dialect disables via the thinking object instead.
    let mut deepseek = json!({"model": "synthetic-deepseek-v4"});
    apply_aux_thinking_fields_from_parts(
        ApiProvider::OpenCode,
        "synthetic-deepseek-v4",
        OPENCODE_DEFAULT_ENDPOINT,
        &mut deepseek,
    );
    assert_eq!(
        deepseek.get("thinking"),
        Some(&json!({"type": "disabled"}))
    );
    assert!(deepseek.get("enable_thinking").is_none());

    // NoOffSwitchDialect members send no thinking fields because the adapter
    // declares no reliable off switch.
    let mut mimo = json!({"model": "synthetic-mimo-wire"});
    apply_aux_thinking_fields_from_parts(
        ApiProvider::OpenCode,
        "synthetic-mimo-wire",
        OPENCODE_DEFAULT_ENDPOINT,
        &mut mimo,
    );
    assert!(mimo.get("thinking").is_none());
    assert!(mimo.get("enable_thinking").is_none());

    // Registry coverage: the covered classes must still exist.
    assert!(
        crate::ai::model_names::all()
            .iter()
            .any(|m| m.adapter == ApiProvider::Alibaba),
        "registry must contain an Alibaba-adapter entry"
    );
    opencode_deepseek_entries_exist();
    assert!(
        crate::ai::model_names::all().iter().any(|m| {
            let name = m.name.to_ascii_lowercase();
            name.contains("mimo") || name.contains("minimax") || name.contains("beam")
        }),
        "registry must contain a no-off-switch-dialect member (MiniMax/beam family)"
    );
}

#[test]
fn openai_request_body_omits_nonstandard_flags() {
    let messages = user_hi();
    let inputs = synthetic_inputs("synthetic-gpt", OPENAI_ENDPOINT, ApiProvider::OpenAi, None);

    let body = build_request_body_from_parts(
        &inputs,
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

    assert!(value.get("enable_thinking").is_none());
    assert!(value.get("enable_search").is_none());
    assert_eq!(value.get("reasoning_effort"), Some(&json!("high")));
    assert!(value.get("reasoning").is_none());
    assert_eq!(
        value.get("model").and_then(|v| v.as_str()),
        Some("synthetic-gpt")
    );
}

#[test]
fn alibaba_request_body_keeps_extension_flags() {
    let messages = user_hi();
    let inputs = synthetic_inputs(
        "synthetic-qwen",
        ALIBABA_DEFAULT_ENDPOINT,
        ApiProvider::Alibaba,
        TOP_LEVEL,
    );

    let body = build_request_body_from_parts(
        &inputs,
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

    assert_eq!(value["enable_thinking"], json!(true));
    assert_eq!(value["enable_search"], json!(true));
    assert_eq!(value["reasoning_effort"], json!("high"));
    assert!(value.get("reasoning").is_none());

    // Registry coverage: the mirrored class must still exist.
    assert!(
        crate::ai::model_names::all().iter().any(|m| {
            m.adapter == ApiProvider::Alibaba
                && models::reasoning_effort_wire(&m.key)
                    == Some(crate::ai::model_names::ReasoningEffortWire::TopLevel)
        }),
        "registry must contain an Alibaba entry declaring top_level effort wire"
    );
}
