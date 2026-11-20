use super::super::builder::{
    CONTEXT_WINDOW_SAFETY_MARGIN_TOKENS, RequestModelInputs, build_request_body_from_parts,
    clamp_with_estimated_prompt_from_parts, estimate_request_prompt_tokens,
    MIN_OUTPUT_TOKENS_FLOOR,
};
use super::super::reasoning::{
    apply_thinking_force_off_effort_from_parts, model_effort_graded_from_parts,
    resolve_reasoning_wire_controls_from_parts,
};
use super::super::*;
use crate::ai::provider::{
    ALIBABA_DEFAULT_ENDPOINT, OPENCODE_DEFAULT_ENDPOINT, ApiProvider,
    reasoning_effort_reduces_thinking_for, thinking_off_capability_for,
};
use serde_json::Value;

/// Pre-resolved fixtures keep wire assertions independent of registry renames.
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

fn user_hi() -> Vec<Message> {
    vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }]
}

/// Reuse the production clamp over pre-resolved inputs to construct the expected
/// `max_tokens` fragment (`,"max_tokens":N` when the inputs declare an output cap).
fn expected_max_tokens_field_from_parts(inputs: &RequestModelInputs, messages: &[Message]) -> String {
    match inputs.max_output {
        Some(model_max) => {
            let est = estimate_request_prompt_tokens(messages, None, None);
            let clamped =
                clamp_with_estimated_prompt_from_parts(inputs.context_window, est, model_max);
            format!(r#","max_tokens":{clamped}"#)
        }
        None => String::new(),
    }
}

const TOP_LEVEL: Option<crate::ai::model_names::ReasoningEffortWire> =
    Some(crate::ai::model_names::ReasoningEffortWire::TopLevel);
const DASHSCOPE_DEEPSEEK_OFFICIAL_ENDPOINT: &str = "https://api.deepseek.com/chat/completions";
const VOLCANO_ARK_ENDPOINT: &str = "https://ark.cn-beijing.volces.com/api/v3";
const MODELHUB_ENDPOINT: &str = "https://dataagent-dev-llm.bytedance.net/v1";

#[test]
fn dashscope_and_other_adapter_request_body_wire_format_is_byte_stable() {
    let messages = user_hi();

    // Declared top-level effort placement bypasses the dialect's binary-switch
    // omission, retaining `reasoning_effort` alongside `enable_thinking` and
    // `enable_search`.
    let alibaba_inputs = RequestModelInputs {
        max_output: Some(131_072),
        context_window: 1_000_000,
        ..synthetic_inputs(
            "synthetic-qwen-wire",
            ALIBABA_DEFAULT_ENDPOINT,
            ApiProvider::Alibaba,
            TOP_LEVEL,
        )
    };
    let alibaba = build_request_body_from_parts(
        &alibaba_inputs,
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
    let alibaba_max_tokens_field = expected_max_tokens_field_from_parts(&alibaba_inputs, &messages);
    assert_eq!(
        serde_json::to_string(&alibaba).unwrap(),
        format!(
            r#"{{"model":"synthetic-qwen-wire","messages":[{{"role":"user","content":"hi"}}],"stream":false,"enable_thinking":true,"enable_search":true,"reasoning_effort":"high"{alibaba_max_tokens_field}}}"#
        )
    );

    // OpenCode non-DeepSeek: fields match the OpenAI-compatible family
    // (top-level reasoning_effort, extension fields omitted). The
    // DeepSeek-specific `thinking` field is covered by the separate
    // `deepseek_request_body_uses_thinking_object` test.
    let opencode_inputs = RequestModelInputs {
        max_output: Some(131_072),
        context_window: 262_144,
        ..synthetic_inputs(
            "synthetic-glm-wire",
            OPENCODE_DEFAULT_ENDPOINT,
            ApiProvider::OpenCode,
            None,
        )
    };
    let opencode = build_request_body_from_parts(
        &opencode_inputs,
        &messages,
        false,
        true,
        Some(true),
        None,
        None,
        Some("medium"),
        None,
        None,
        None,
    );
    let opencode_max_tokens_field =
        expected_max_tokens_field_from_parts(&opencode_inputs, &messages);
    assert_eq!(
        serde_json::to_string(&opencode).unwrap(),
        format!(
            r#"{{"model":"synthetic-glm-wire","messages":[{{"role":"user","content":"hi"}}],"stream":false,"reasoning_effort":"medium"{opencode_max_tokens_field}}}"#
        )
    );

    // Require registry entries for both adapter classes to keep these fixtures
    // relevant to production.
    assert!(
        crate::ai::model_names::all().iter().any(|model| {
            model.adapter == ApiProvider::Alibaba
                && model.reasoning_effort_wire
                    == Some(crate::ai::model_names::ReasoningEffortWire::TopLevel)
        }),
        "model registry must contain an Alibaba model declaring top_level effort wire"
    );
    assert!(
        crate::ai::model_names::all()
            .iter()
            .any(|m| m.adapter == ApiProvider::OpenCode
                && !m.name.to_ascii_lowercase().contains("deepseek")),
        "model registry must contain a non-DeepSeek OpenCode model"
    );
}

/// The clamp consumes a validated current-request count. Compression rejection
/// belongs to prompt_feedback, before any observed count reaches this function.
#[test]
fn prompt_feedback_clamp_preserves_current_count_without_truncating_overflow() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("short message after compression".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    // Synthetic window/cap (mirrors an entry declaring max_output_tokens): an
    // overflowed validated count clamps to the floor, a fresh short prompt
    // keeps the full cap.
    let context_window = 200_000;
    let model_max = 32_768;
    let overflow = usize::try_from(u64::MAX).unwrap_or(usize::MAX);
    assert_eq!(
        clamp_with_estimated_prompt_from_parts(context_window, overflow, model_max),
        MIN_OUTPUT_TOKENS_FLOOR
    );
    let est = estimate_request_prompt_tokens(&messages, None, None);
    assert_eq!(
        clamp_with_estimated_prompt_from_parts(context_window, est, model_max),
        model_max
    );

    // Registry coverage: at least one entry still declares an output cap, so
    // the clamp path above stays reachable in production.
    assert!(
        crate::ai::model_names::all()
            .iter()
            .any(|entry| models::max_output_tokens(&entry.key).is_some()),
        "registry must contain an output cap"
    );
}

#[test]
fn short_prompt_keeps_every_declared_output_budget() {
    let messages = user_hi();

    // A short prompt retains the full output cap when the window covers the
    // prompt, cap, and safety margin.
    let est = estimate_request_prompt_tokens(&messages, None, None);
    assert_eq!(
        clamp_with_estimated_prompt_from_parts(200_000, est, 32_768),
        32_768
    );

    // Registry coverage: every entry declaring an output cap must keep a window
    // large enough that the short "hi" prompt preserves its full budget —
    // iterating the registry (instead of naming keys) keeps this true after
    // registry renames.
    let mut checked = 0;
    for def in crate::ai::model_names::all() {
        let Some(model_max) = models::max_output_tokens(&def.key) else {
            continue;
        };
        let window = models::context_window_tokens(&def.key);
        assert!(
            window >= model_max as usize + CONTEXT_WINDOW_SAFETY_MARGIN_TOKENS + est,
            "{}: window {window} must cover max_output {model_max} + margin + short prompt {est}",
            def.key
        );
        checked += 1;
    }
    assert!(checked > 0, "registry must declare at least one output cap");
}

#[test]
fn build_request_body_sends_provider_model_name_for_key_handle() {
    let messages = user_hi();

    // The wire body carries the pre-resolved provider request name; registry
    // selector keys may include platform suffixes.
    let inputs = RequestModelInputs {
        context_window: 1_024_000,
        ..synthetic_inputs(
            "synthetic-deepseek-wire-v4",
            OPENCODE_DEFAULT_ENDPOINT,
            ApiProvider::OpenCode,
            TOP_LEVEL,
        )
    };
    let body = build_request_body_from_parts(
        &inputs,
        &messages,
        false,
        true,
        Some(true),
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let json = serde_json::to_value(&body).unwrap();

    assert_eq!(
        json.get("model").and_then(|v| v.as_str()),
        Some("synthetic-deepseek-wire-v4"),
    );
    assert_eq!(
        json.pointer("/thinking/type").and_then(|v| v.as_str()),
        Some("enabled")
    );

    // Require a thinking-enabled plaintext OpenCode DeepSeek entry so the
    // provider-name fixture corresponds to a registered model class.
    assert!(
        crate::ai::model_names::all().iter().any(|m| {
            m.adapter == ApiProvider::OpenCode
                && m.enable_thinking
                && !m.name.starts_with("enc:")
                && m.name.to_ascii_lowercase().contains("deepseek")
        }),
        "registry must contain a plaintext OpenCode DeepSeek entry with thinking enabled"
    );
}

#[test]
fn opencode_thinking_entries_send_thinking_object_and_declared_effort() {
    let messages = user_hi();

    // OpenCode dispatch selects the thinking object for DeepSeek request names.
    // Explicit top-level effort placement emits the tier alongside that object
    // without relying on the dialect's default effort behavior.
    let inputs = synthetic_inputs(
        "synthetic-deepseek-wire",
        OPENCODE_DEFAULT_ENDPOINT,
        ApiProvider::OpenCode,
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
    let json = serde_json::to_value(&body).unwrap();
    assert_eq!(
        json.pointer("/thinking/type").and_then(|v| v.as_str()),
        Some("enabled"),
    );
    assert_eq!(json.get("reasoning_effort"), Some(&json!("high")));

    // Registry coverage: every matching entry must still declare the top_level
    // wire placement the wire shape above depends on.
    let mut checked = 0;
    for def in crate::ai::model_names::all() {
        if def.adapter != ApiProvider::OpenCode || !def.enable_thinking || def.name.starts_with("enc:") {
            continue;
        }
        if !models::request_model_name(&def.key)
            .to_ascii_lowercase()
            .contains("deepseek")
        {
            continue;
        }
        assert_eq!(
            models::reasoning_effort_wire(&def.key),
            Some(crate::ai::model_names::ReasoningEffortWire::TopLevel),
            "{}",
            def.key
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "registry must contain an OpenCode entry with thinking enabled"
    );
}

#[test]
fn deepseek_official_entries_send_thinking_object_and_effort() {
    let messages = user_hi();

    // The official DeepSeek endpoint selects the same thinking-object dialect
    // as OpenCode DeepSeek. Explicit top-level wire placement also emits effort.
    let inputs = synthetic_inputs(
        "synthetic-deepseek-official",
        DASHSCOPE_DEEPSEEK_OFFICIAL_ENDPOINT,
        ApiProvider::Compatible,
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
    let json = serde_json::to_value(&body).unwrap();
    assert_eq!(
        json.pointer("/thinking/type").and_then(|v| v.as_str()),
        Some("enabled"),
    );
    assert_eq!(json.get("reasoning_effort"), Some(&json!("high")));
    assert!(json.get("enable_thinking").is_none());

    // Registry coverage: the official-endpoint class must still exist and
    // declare the graded wire placement.
    let mut checked = 0;
    for def in crate::ai::model_names::all() {
        let Some(endpoint) = def.endpoint.as_deref() else {
            continue;
        };
        if def.adapter != ApiProvider::Compatible
            || !def.enable_thinking
            || !endpoint
                .trim()
                .to_ascii_lowercase()
                .contains("api.deepseek.com")
        {
            continue;
        }
        assert_eq!(
            models::reasoning_effort_wire(&def.key),
            Some(crate::ai::model_names::ReasoningEffortWire::TopLevel),
            "{}",
            def.key
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "registry must contain a compatible-adapter entry on api.deepseek.com with thinking enabled"
    );
}

#[test]
fn dashscope_deepseek_uses_model_specific_reasoning_contract() {
    // Exercise Alibaba on DashScope with top-level effort and source-tagged
    // exact reasoning replay.
    const REQUEST_MODEL: &str = "synthetic-dashscope-deepseek";
    let messages = user_hi();

    let inputs = RequestModelInputs {
        max_output: Some(393_216),
        context_window: 1_000_000,
        ..synthetic_inputs(
            REQUEST_MODEL,
            ALIBABA_DEFAULT_ENDPOINT,
            ApiProvider::Alibaba,
            TOP_LEVEL,
        )
    };
    let body = build_request_body_from_parts(
        &inputs,
        &messages,
        false,
        true,
        Some(true),
        None,
        None,
        Some("max"),
        None,
        None,
        None,
    );
    let json = serde_json::to_value(&body).unwrap();
    assert_eq!(
        json.get("model").and_then(|value| value.as_str()),
        Some(REQUEST_MODEL)
    );
    assert_eq!(
        json.get("enable_thinking")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    // The declared `reasoning_effort_wire: "top_level"` lets the tier reach the
    // wire even though the dialect alone is a binary `enable_thinking` switch
    // (resolve_reasoning_wire_controls bypasses adapt_effort for
    // registry-declared models).
    assert_eq!(json.get("reasoning_effort"), Some(&json!("max")));
    assert!(json.get("reasoning").is_none());

    // Registry coverage: the DashScope DeepSeek contract class (DashScope
    // endpoint + declared top_level wire + exact replay) must still exist with
    // its documented metadata (default Max effort, graded effort, replay on,
    // no explicit prompt cache).
    let mut checked = 0;
    for def in crate::ai::model_names::all() {
        let endpoint = def.endpoint.as_deref().unwrap_or("");
        if !endpoint.contains("dashscope.aliyuncs.com")
            || models::reasoning_effort_wire(&def.key)
                != Some(crate::ai::model_names::ReasoningEffortWire::TopLevel)
            || !models::reasoning_content_replay_enabled(&def.key)
        {
            continue;
        }
        assert_eq!(
            models::default_reasoning_effort(&def.key),
            Some(crate::ai::provider::ReasoningEffort::Max),
            "{}",
            def.key
        );
        assert!(models::reasoning_effort_reduces_thinking(&def.key), "{}", def.key);
        assert!(!models::explicit_prompt_cache_enabled(&def.key), "{}", def.key);
        checked += 1;
    }
    assert!(
        checked > 0,
        "registry must contain a DashScope DeepSeek entry with top_level wire and exact replay"
    );

    // Exact replay round-trip on synthetic inputs: the persisted projection
    // tags the tool-call reasoning with the source tag, and normalize decodes
    // the own blob back to the original provider text.
    let assistant = Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![crate::ai::types::ToolCall {
            id: "call_dashscope_deepseek".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: Some("需要先读取目标文件。".to_string()),
    };
    let projected =
        crate::ai::history::compress::sanitize_message_for_persisted_history_inner(
            &assistant,
            Some(REQUEST_MODEL),
        );
    let mut projected_messages = vec![projected];
    super::super::reasoning::normalize_reasoning_content_replay_from_parts(
        REQUEST_MODEL,
        true,
        ApiProvider::Alibaba,
        REQUEST_MODEL,
        ALIBABA_DEFAULT_ENDPOINT,
        &mut projected_messages,
    );
    assert_eq!(
        projected_messages[0].reasoning_content,
        assistant.reasoning_content
    );
}

#[test]
fn dashscope_deepseek_sends_effort_and_exact_reasoning_replay() {
    const REQUEST_MODEL: &str = "synthetic-dashscope-deepseek";
    let messages = user_hi();

    let inputs = synthetic_inputs(
        REQUEST_MODEL,
        ALIBABA_DEFAULT_ENDPOINT,
        ApiProvider::Alibaba,
        TOP_LEVEL,
    );
    let body = build_request_body_from_parts(
        &inputs,
        &messages,
        false,
        true,
        None,
        None,
        None,
        Some("high"),
        None,
        None,
        None,
    );
    let json = serde_json::to_value(&body).unwrap();
    assert_eq!(
        json.get("enable_thinking")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    // Declared top-level wire placement emits effort even for a binary-switch
    // dialect.
    assert_eq!(json.get("reasoning_effort"), Some(&json!("high")));
    assert!(json.get("reasoning").is_none());

    // Registry coverage: the graded-effort + exact-replay contract class must
    // still exist on the DashScope endpoint.
    assert!(
        crate::ai::model_names::all().iter().any(|def| {
            let endpoint = def.endpoint.as_deref().unwrap_or("");
            endpoint.contains("dashscope.aliyuncs.com")
                && models::reasoning_effort_wire(&def.key)
                    == Some(crate::ai::model_names::ReasoningEffortWire::TopLevel)
                && models::reasoning_content_replay_enabled(&def.key)
                && models::reasoning_effort_reduces_thinking(&def.key)
        }),
        "registry must contain a DashScope entry with graded effort and exact replay"
    );
}

#[test]
fn dashscope_qwen_metadata_matches_documented_thinking_controls() {
    // Discover Alibaba thinking fixtures by metadata so renaming models cannot
    // invalidate coverage.
    let matching: Vec<_> = crate::ai::model_names::all()
        .iter()
        .filter(|def| {
            def.adapter == ApiProvider::Alibaba
                && models::enable_thinking(&def.key)
                && models::default_reasoning_effort(&def.key)
                    == Some(crate::ai::provider::ReasoningEffort::Max)
        })
        .map(|def| def.key.clone())
        .collect();
    assert!(
        matching.len() >= 2,
        "registry must still document at least two Alibaba thinking entries with Max default effort, found: {matching:?}"
    );
}

#[test]
fn opencode_disabled_thinking_object_with_declared_effort() {
    const REQUEST_MODEL: &str = "synthetic-deepseek-wire";
    let (thinking, top_level_reasoning_effort, nested_reasoning) =
        resolve_reasoning_wire_controls_from_parts(
            ApiProvider::OpenCode,
            REQUEST_MODEL,
            OPENCODE_DEFAULT_ENDPOINT,
            TOP_LEVEL,
            false,
            Some("high"),
        );

    // Disabling the switch does not suppress an explicitly selected top-level
    // effort field. Check both serialized controls independently.
    assert_eq!(
        thinking
            .get("thinking")
            .and_then(|v| v.get("type"))
            .and_then(|v| v.as_str()),
        Some("disabled"),
    );
    assert_eq!(top_level_reasoning_effort, Some("high"));
    assert!(nested_reasoning.is_none());

    // Registry coverage: the mirrored class must still exist.
    assert!(
        crate::ai::model_names::all().iter().any(|m| {
            m.adapter == ApiProvider::OpenCode
                && m.enable_thinking
                && !m.name.starts_with("enc:")
                && m.name.to_ascii_lowercase().contains("deepseek")
        }),
        "registry must contain a plaintext OpenCode DeepSeek entry with thinking enabled"
    );
}

#[test]
fn effort_value_adaptation_is_per_vendor_on_wire() {
    // Explicit effort wire placement selects the top-level or nested field.
    // Otherwise the provider dialect supplies the effort behavior.
    let (thinking, top_level, nested) = resolve_reasoning_wire_controls_from_parts(
        ApiProvider::Alibaba,
        "synthetic-qwen",
        ALIBABA_DEFAULT_ENDPOINT,
        TOP_LEVEL,
        true,
        Some("max"),
    );
    assert_eq!(thinking.get("enable_thinking"), Some(&json!(true)));
    assert_eq!(
        top_level,
        Some("max"),
        "the mirrored qwen entry declares top_level wire placement, so the tier passes through"
    );
    assert!(nested.is_none());

    let (thinking, top_level, nested) = resolve_reasoning_wire_controls_from_parts(
        ApiProvider::Compatible,
        "synthetic-deepseek-official",
        DASHSCOPE_DEEPSEEK_OFFICIAL_ENDPOINT,
        TOP_LEVEL,
        true,
        Some("max"),
    );
    assert_eq!(
        thinking.get("thinking").and_then(|v| v.get("type")),
        Some(&json!("enabled"))
    );
    assert_eq!(
        top_level,
        Some("max"),
        "the mirrored official-DeepSeek entry declares top_level wire placement, so the tier passes through"
    );
    assert!(nested.is_none());

    // The Volcano fixture exercises top-level effort without a thinking switch.
    let (thinking, top_level, nested) = resolve_reasoning_wire_controls_from_parts(
        ApiProvider::Compatible,
        "synthetic-deepseek-volcano",
        VOLCANO_ARK_ENDPOINT,
        TOP_LEVEL,
        true,
        Some("max"),
    );
    assert!(thinking.is_empty());
    assert_eq!(top_level, Some("max"));
    assert!(nested.is_none());

    // Registry coverage: each mirrored vendor class must still exist.
    assert!(
        crate::ai::model_names::all().iter().any(|def| {
            def.adapter == ApiProvider::Alibaba
                && def.endpoint.as_deref().unwrap_or("").contains("dashscope.aliyuncs.com")
                && models::reasoning_effort_wire(&def.key)
                    == Some(crate::ai::model_names::ReasoningEffortWire::TopLevel)
        }),
        "registry must contain an Alibaba DashScope entry with top_level wire"
    );
    assert!(
        crate::ai::model_names::all().iter().any(|def| {
            def.adapter == ApiProvider::Compatible
                && def.endpoint.as_deref().unwrap_or("").contains("api.deepseek.com")
        }),
        "registry must contain a compatible-adapter entry on api.deepseek.com"
    );
    assert!(
        crate::ai::model_names::all().iter().any(|def| {
            def.endpoint
                .as_deref()
                .unwrap_or("")
                .contains("ark.cn-beijing.volces.com")
        }),
        "registry must contain a Volcano ark entry"
    );
}

#[test]
fn model_effort_graded_reflects_registry_declared_gradation() {
    // Declared effort wire placement marks the model as graded. Without a
    // declaration, gradation follows the provider dialect.
    assert!(model_effort_graded_from_parts(
        TOP_LEVEL,
        ApiProvider::Alibaba,
        "synthetic-qwen",
        ALIBABA_DEFAULT_ENDPOINT
    ));
    assert!(model_effort_graded_from_parts(
        TOP_LEVEL,
        ApiProvider::Alibaba,
        "synthetic-dashscope-deepseek",
        ALIBABA_DEFAULT_ENDPOINT
    ));
    assert!(model_effort_graded_from_parts(
        TOP_LEVEL,
        ApiProvider::Compatible,
        "synthetic-deepseek-official",
        DASHSCOPE_DEEPSEEK_OFFICIAL_ENDPOINT
    ));
    // Effort-only dialects keep the gradation.
    assert!(model_effort_graded_from_parts(
        TOP_LEVEL,
        ApiProvider::Compatible,
        "synthetic-deepseek-volcano",
        VOLCANO_ARK_ENDPOINT
    ));

    // Registry coverage: the declared-gradation class on binary-switch
    // dialects must still exist (otherwise the override above pins dead code).
    assert!(
        crate::ai::model_names::all().iter().any(|def| {
            let endpoint = def.endpoint.as_deref().unwrap_or("");
            (endpoint.contains("dashscope.aliyuncs.com") || endpoint.contains("api.deepseek.com"))
                && models::reasoning_effort_wire(&def.key).is_some()
        }),
        "registry must contain a binary-switch-dialect entry declaring effort gradation"
    );
}

#[test]
fn responses_protocol_entries_support_reasoning_with_tools() {
    // Every entry declaring `request_protocol: responses` must keep the responses dialect and
    // accept reasoning_effort together with tools. Registry-driven, so renames cannot leave the
    // assertions running against a model that no longer resolves.
    // Encrypted endpoints are decrypted through a key file whose path derives from CONFIGW_PATH,
    // so this test must hold ENV_LOCK against tests that repoint CONFIGW_PATH.
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());

    let mut checked = 0;
    for def in crate::ai::model_names::all() {
        if def.request_protocol != Some(RequestProtocolDialect::Responses) {
            continue;
        }
        assert!(
            !models::reasoning_effort_conflicts_with_tools(&def.key),
            "{}",
            def.key
        );
        let endpoint = models::endpoint_for_model(&def.key, "");
        assert_eq!(
            models::request_protocol_dialect(&def.key, &endpoint),
            RequestProtocolDialect::Responses,
            "{}",
            def.key
        );
        // Only assert the route shape when the endpoint is readable: without the key file the
        // encrypted value stays encrypted and says nothing about the route.
        if !crate::commonw::secret::is_encrypted(&endpoint) {
            assert!(
                endpoint.ends_with("/v1/responses"),
                "{}: {endpoint}",
                def.key
            );
        }
        checked += 1;
    }
    assert!(
        checked > 0,
        "registry must declare at least one responses-protocol entry"
    );
}

/// Responses fixtures use encrypted replay with no explicit effort wire placement.
fn synthetic_responses_inputs(request_model: &str) -> RequestModelInputs {
    RequestModelInputs {
        dialect: RequestProtocolDialect::Responses,
        encrypted_replay: true,
        ..synthetic_inputs(request_model, MODELHUB_ENDPOINT, ApiProvider::Compatible, None)
    }
}

fn responses_entries_exist() {
    assert!(
        crate::ai::model_names::all()
            .iter()
            .any(|def| def.request_protocol == Some(RequestProtocolDialect::Responses)),
        "registry must declare at least one responses-protocol entry"
    );
}

#[test]
fn responses_request_body_uses_function_tools_and_nested_reasoning() {
    let messages = user_hi();
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object"}
        }
    }]);
    let inputs = synthetic_responses_inputs("synthetic-responses-model");
    let request = build_request_body_from_parts(
        &inputs,
        &messages,
        false,
        false,
        None,
        Some(tools),
        None,
        Some("high"),
        None,
        None,
        None,
    );

    let body = super::super::build_responses_request_body(&request);
    assert_eq!(body["reasoning"]["effort"], "high");
    assert_eq!(body["reasoning"]["summary"], "auto");
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["name"], "get_weather");
    assert!(body.get("messages").is_none());
    assert!(body.get("reasoning_effort").is_none());
    assert_eq!(body["input"][0]["role"], "user");
    assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
    assert_eq!(body["input"][0]["content"][0]["text"], "hi");

    responses_entries_exist();
}

#[test]
fn responses_force_off_override_emits_none_effort_on_wire() {
    // For EffortNone dialects, the truncation override replaces the reduced
    // "low" tier with "none" so Responses receives its reasoning.effort off control.
    let messages = user_hi();
    // Use production dialect dispatch for the fixture's adapter, request name,
    // and endpoint.
    let effort_reduces =
        reasoning_effort_reduces_thinking_for(ApiProvider::Compatible, "synthetic-modelhub-model", MODELHUB_ENDPOINT);
    let off_capability =
        thinking_off_capability_for(ApiProvider::Compatible, "synthetic-modelhub-model", MODELHUB_ENDPOINT);
    assert!(effort_reduces);
    assert!(matches!(
        off_capability,
        crate::ai::provider::ThinkingOffCapability::EffortNone
    ));
    let effort =
        apply_thinking_force_off_effort_from_parts(true, effort_reduces, off_capability, Some("low"));
    assert_eq!(effort, Some("none"));
    let inputs = synthetic_responses_inputs("synthetic-responses-model");
    let request = build_request_body_from_parts(
        &inputs, &messages, false, false, None, None, None, effort, None, None, None,
    );
    let body = super::super::build_responses_request_body(&request);
    assert_eq!(body["reasoning"]["effort"], "none");
    assert_eq!(body["reasoning"]["summary"], "auto");

    responses_entries_exist();
}

#[test]
fn responses_search_uses_builtin_web_search_tool() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("搜索今天的新闻".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件",
            "parameters": {"type": "object", "properties": {}}
        }
    }]);
    let inputs = synthetic_responses_inputs("synthetic-responses-model");
    let request = build_request_body_from_parts(
        &inputs,
        &messages,
        false,
        false,
        Some(true),
        Some(tools),
        None,
        None,
        None,
        None,
        None,
    );
    let body = super::super::build_responses_request_body(&request);

    assert!(body.get("enable_search").is_none());
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][1], json!({"type": "web_search"}));

    let request_without_function_tools = build_request_body_from_parts(
        &inputs,
        &messages,
        false,
        false,
        Some(true),
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let body_without_function_tools =
        super::super::build_responses_request_body(&request_without_function_tools);
    assert_eq!(
        body_without_function_tools["tools"],
        json!([{"type": "web_search"}])
    );

    responses_entries_exist();
}

#[test]
fn no_tool_request_bodies_omit_tools_and_tool_choice() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("summarize the completed work".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let inputs = synthetic_responses_inputs("synthetic-responses-model");
    let request = build_request_body_from_parts(
        &inputs, &messages, false, false, None, None, None, None, None, None, None,
    );

    let chat_body = serde_json::to_value(&request).expect("request body should serialize");
    assert!(chat_body.get("tools").is_none());
    assert!(chat_body.get("tool_choice").is_none());

    let responses_body = super::super::build_responses_request_body(&request);
    assert!(responses_body.get("tools").is_none());
    assert!(responses_body.get("tool_choice").is_none());

    responses_entries_exist();
}
