use super::common::{expected_max_tokens_field};
use super::super::builder::MIN_OUTPUT_TOKENS_FLOOR;
use super::super::*;
use crate::ai::tools::os_tools::{GLOBAL_OS, init_os_tools_globals};
use crate::ai::{cli::ParsedCli, types::AppConfig};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};

#[test]
fn dashscope_and_other_adapter_request_body_wire_format_is_byte_stable() {
    use crate::ai::provider::ApiProvider;

    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    // Alibaba models declaring top_level wire: enable_thinking/enable_search
    // and the top-level reasoning_effort are all sent — the declared wire
    // placement means the tier bypasses the dialect's binary-switch omission in
    // resolve_reasoning_wire_controls. Locate the model by its unique key
    // (consistent with the production path) to avoid ambiguous entries sharing
    // a name.
    let alibaba_model = crate::ai::model_names::all()
        .iter()
        .find(|model| {
            model.adapter == ApiProvider::Alibaba
                && model.reasoning_effort_wire
                    == Some(crate::ai::model_names::ReasoningEffortWire::TopLevel)
        })
        .map(|model| model.key.clone())
        .expect("model registry must contain an Alibaba model declaring top_level effort wire");
    let alibaba = build_request_body(
        &alibaba_model,
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
    // The wire's model field is the resolved request_model_name (the provider's
    // actual model name), which may differ from the key used for lookup.
    let alibaba_wire_model = super::super::models::request_model_name(&alibaba_model);
    // max_tokens is now clamped by the remaining context window; it is sent only
    // when the model declares max_output_tokens.
    // The expected value is derived from the same clamping function, keeping the
    // wire assertion valid as model configuration changes.
    let alibaba_max_tokens_field = expected_max_tokens_field(&alibaba_model, &messages);
    assert_eq!(
        serde_json::to_string(&alibaba).unwrap(),
        format!(
            r#"{{"model":"{alibaba_wire_model}","messages":[{{"role":"user","content":"hi"}}],"stream":false,"enable_thinking":true,"enable_search":true,"reasoning_effort":"high"{alibaba_max_tokens_field}}}"#
        )
    );

    // OpenCode non-DeepSeek: fields match the OpenAI-compatible family
    // (top-level reasoning_effort, extension fields omitted). The
    // DeepSeek-specific `thinking` field is covered by the separate
    // `deepseek_request_body_uses_thinking_object` test.
    let non_deepseek_opencode = crate::ai::model_names::all()
        .iter()
        .find(|m| {
            m.adapter == ApiProvider::OpenCode && !m.name.to_ascii_lowercase().contains("deepseek")
        })
        .map(|m| m.key.clone());
    if let Some(opencode_model) = non_deepseek_opencode {
        let opencode = build_request_body(
            &opencode_model,
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
        let opencode_wire_model = super::super::models::request_model_name(&opencode_model);
        let opencode_max_tokens_field = expected_max_tokens_field(&opencode_model, &messages);
        assert_eq!(
            serde_json::to_string(&opencode).unwrap(),
            format!(
                r#"{{"model":"{opencode_wire_model}","messages":[{{"role":"user","content":"hi"}}],"stream":false,"reasoning_effort":"medium"{opencode_max_tokens_field}}}"#
            )
        );
    }
}


/// The clamp consumes a validated current-request count. Compression rejection
/// belongs to prompt_feedback, before any observed count reaches this function.
#[test]
fn prompt_feedback_clamp_preserves_current_count_without_truncating_overflow() {
    let model = crate::ai::model_names::all()
        .iter()
        .find(|entry| models::max_output_tokens(&entry.key).is_some())
        .expect("registry must contain an output cap")
        .key
        .clone();
    let model_max = models::max_output_tokens(&model).unwrap();
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("short message after compression".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    assert_eq!(
        clamp_max_tokens_for_prompt(&model, &messages, None, model_max, Some(u64::MAX)),
        MIN_OUTPUT_TOKENS_FLOOR
    );
    assert_eq!(
        clamp_max_tokens_for_prompt(&model, &messages, None, model_max, None),
        model_max
    );
}

#[test]
fn short_prompt_keeps_every_declared_output_budget() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    // A short prompt must not squeeze a declared completion budget: whenever the entry's window
    // covers `max_output_tokens + safety margin`, the full cap has to survive. Iterating the
    // registry (instead of naming keys) keeps this true after registry renames.
    let mut checked = 0;
    for def in crate::ai::model_names::all() {
        let Some(model_max) = models::max_output_tokens(&def.key) else {
            continue;
        };
        assert_eq!(
            clamp_max_tokens_for_prompt(&def.key, &messages, None, model_max, None),
            model_max,
            "{}",
            def.key
        );
        checked += 1;
    }
    assert!(checked > 0, "registry must declare at least one output cap");
}

#[test]
fn build_request_body_sends_provider_model_name_for_key_handle() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    // The wire model name is the registry `name`, while the selector/`key` may include the
    // platform suffix. Take the entry from the registry so a rename cannot make this observe a
    // fallback path instead of the mapping it is meant to guard.
    let def = crate::ai::model_names::all()
        .iter()
        .find(|m| {
            m.adapter == crate::ai::provider::ApiProvider::OpenCode
                && m.enable_thinking
                && !m.name.starts_with("enc:")
                && m.name.to_ascii_lowercase().contains("deepseek")
        })
        .copied()
        .expect("registry must contain a plaintext OpenCode DeepSeek entry with thinking enabled");

    let body = build_request_body(
        &def.key,
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
        Some(def.name.as_str()),
        "{}",
        def.key
    );
    assert_eq!(
        json.pointer("/thinking/type").and_then(|v| v.as_str()),
        Some("enabled")
    );
}

#[test]
fn opencode_thinking_entries_send_thinking_object_and_declared_effort() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    // The OpenCode dialect only sends the thinking object for DeepSeek models
    // (thinking_dialect_for in src/bin/ai/provider/adapter/thinking.rs). Each
    // such entry with thinking enabled must keep sending it, and the top-level
    // reasoning_effort is placed as well: the registry declares
    // `reasoning_effort_wire: "top_level"` for these entries (DeepSeek v4 on
    // the OpenCode Zen gateway supports [none, low, high, max] per the official
    // API docs), so the tier bypasses the dialect's binary-switch omission in
    // resolve_reasoning_wire_controls. Entries are discovered from the registry
    // so renames cannot empty the loop.
    let mut checked = 0;
    for def in crate::ai::model_names::all() {
        if def.adapter != crate::ai::provider::ApiProvider::OpenCode
            || !def.enable_thinking
            || def.name.starts_with("enc:")
        {
            continue;
        }
        if !models::request_model_name(&def.key)
            .to_ascii_lowercase()
            .contains("deepseek")
        {
            continue;
        }
        let model = def.key.as_str();
        let body = build_request_body(
            model,
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
            "{model}"
        );
        assert_eq!(json.get("reasoning_effort"), Some(&json!("high")), "{model}");
        checked += 1;
    }
    assert!(
        checked > 0,
        "registry must contain an OpenCode entry with thinking enabled"
    );
}

#[test]
fn deepseek_official_entries_send_thinking_object_and_effort() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    // The official DeepSeek endpoint (api.deepseek.com) uses the same `thinking`
    // object dialect as the OpenCode Zen gateway (thinking_dialect_for in
    // src/bin/ai/provider/adapter/thinking.rs), per the official thinking-mode
    // docs: {"thinking": {"type": "enabled"}} together with the top-level
    // reasoning_effort — the official API documents `[none, low, high, max]`
    // (graded), and the registry entry declares `reasoning_effort_wire:
    // "top_level"`, so the tier is sent. Entries are discovered from the
    // registry so renames cannot empty the loop.
    let mut checked = 0;
    for def in crate::ai::model_names::all() {
        let Some(endpoint) = def.endpoint.as_deref() else {
            continue;
        };
        if def.adapter != crate::ai::provider::ApiProvider::Compatible
            || !def.enable_thinking
            || !endpoint
                .trim()
                .to_ascii_lowercase()
                .contains("api.deepseek.com")
        {
            continue;
        }
        let model = def.key.as_str();
        let body = build_request_body(
            model,
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
            "{model}"
        );
        assert_eq!(json.get("reasoning_effort"), Some(&json!("high")), "{model}");
        assert!(json.get("enable_thinking").is_none(), "{model}");
        checked += 1;
    }
    assert!(
        checked > 0,
        "registry must contain a compatible-adapter entry on api.deepseek.com with thinking enabled"
    );
}

#[test]
fn dashscope_deepseek_uses_model_specific_reasoning_contract() {
    const MODEL: &str = "deepseek-v4-flash-0731-alibaba";
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    let body = build_request_body(
        MODEL,
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
        Some("deepseek-v4-flash-0731")
    );
    assert_eq!(
        json.get("enable_thinking")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    // The registry declares `reasoning_effort_wire: "top_level"` for this
    // model, so the tier reaches the wire even though the dialect alone is a
    // binary `enable_thinking` switch (resolve_reasoning_wire_controls bypasses
    // adapt_effort for registry-declared models).
    assert_eq!(json.get("reasoning_effort"), Some(&json!("max")));
    assert!(json.get("reasoning").is_none());
    assert_eq!(
        models::default_reasoning_effort(MODEL),
        Some(crate::ai::provider::ReasoningEffort::Max)
    );
    assert!(models::reasoning_effort_reduces_thinking(MODEL));
    assert!(models::reasoning_content_replay_enabled(MODEL));
    assert!(!models::explicit_prompt_cache_enabled(MODEL));

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
    let projected = crate::ai::history::compress::sanitize_message_for_persisted_history_for_model(
        MODEL, &assistant,
    );
    let mut projected_messages = vec![projected];
    normalize_reasoning_content_replay_for_model(MODEL, &mut projected_messages);
    assert_eq!(
        projected_messages[0].reasoning_content,
        assistant.reasoning_content
    );
}

#[test]
fn dashscope_deepseek_sends_effort_and_exact_reasoning_replay() {
    const MODEL: &str = "deepseek-v4-flash-0731-alibaba";
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    let body = build_request_body(
        MODEL,
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
    // Effort tier sent on the wire: the registry declares a top-level
    // reasoning_effort wire placement for this model (DashScope DeepSeek v4
    // supports low/high/max per the official docs), so the tier bypasses the
    // dialect's binary-switch omission.
    assert_eq!(json.get("reasoning_effort"), Some(&json!("high")));
    assert!(json.get("reasoning").is_none());
    assert!(models::reasoning_effort_reduces_thinking(MODEL));
    assert!(models::reasoning_content_replay_enabled(MODEL));
}

#[test]
fn dashscope_qwen_metadata_matches_documented_thinking_controls() {
    assert_eq!(
        models::default_reasoning_effort("qwen3.7-plus-alibaba"),
        Some(crate::ai::provider::ReasoningEffort::High)
    );
    assert_eq!(
        models::default_reasoning_effort("qwen3.7-max-alibaba"),
        Some(crate::ai::provider::ReasoningEffort::High)
    );
    assert_eq!(
        // Registry entry renamed: models/07-qwen3.6-flash-alibaba.json is now
        // models/qwen-flash-alibaba.json (name "qwen3.8-flash", effort "max").
        models::default_reasoning_effort("qwen3.8-flash-alibaba"),
        Some(crate::ai::provider::ReasoningEffort::Max)
    );
    assert!(models::enable_thinking("qwen3.8-flash-alibaba"));
}

#[test]
fn opencode_disabled_thinking_object_with_declared_effort() {
    let endpoint = crate::ai::provider::OPENCODE_DEFAULT_ENDPOINT.to_string();
    let def = crate::ai::model_names::all()
        .iter()
        .find(|m| {
            m.adapter == crate::ai::provider::ApiProvider::OpenCode
                && m.enable_thinking
                && !m.name.starts_with("enc:")
                && m.name.to_ascii_lowercase().contains("deepseek")
        })
        .copied()
        .expect("registry must contain a plaintext OpenCode DeepSeek entry with thinking enabled");
    let (thinking, top_level_reasoning_effort, nested_reasoning) = resolve_reasoning_wire_controls(
        &def.key,
        &endpoint,
        false,
        Some("high"),
    );

    // The OpenCode DeepSeek dialect always sends thinking:{"type":"disabled"}
    // when force-off is requested (the reliable off-switch). The top-level
    // reasoning_effort is still placed because the registry declares
    // `reasoning_effort_wire: "top_level"` for this model — the official API
    // documents `[none, low, high, max]` and the disabled thinking object takes
    // precedence, so carrying the tier is harmless.
    assert_eq!(
        thinking
            .get("thinking")
            .and_then(|v| v.get("type"))
            .and_then(|v| v.as_str()),
        Some("disabled"),
        "{}",
        def.key
    );
    assert_eq!(top_level_reasoning_effort, Some("high"));
    assert!(nested_reasoning.is_none());
}

#[test]
fn effort_value_adaptation_is_per_vendor_on_wire() {
    // The same local tier must adapt per vendor on the wire: models without a
    // registry-declared wire placement follow the dialect default (binary-switch
    // dialects omit it); registry-declared models (qwen / DeepSeek v4 on
    // DashScope, official DeepSeek, OpenCode Zen) and effort-only dialects pass
    // it through. No if/else in the request layer — the registry placement or
    // the dialect owns the value.
    let dashscope_endpoint =
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";
    let (thinking, top_level, nested) = resolve_reasoning_wire_controls(
        "qwen3.7-max-alibaba",
        dashscope_endpoint,
        true,
        Some("max"),
    );
    assert_eq!(thinking.get("enable_thinking"), Some(&json!(true)));
    assert_eq!(
        top_level,
        Some("max"),
        "qwen3.7-max-alibaba declares top_level wire placement, so the tier passes through"
    );
    assert!(nested.is_none());

    let (thinking, top_level, nested) = resolve_reasoning_wire_controls(
        "deepseek-flash-official",
        "https://api.deepseek.com/chat/completions",
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
        "deepseek-flash-official declares top_level wire placement, so the tier passes through"
    );
    assert!(nested.is_none());

    // Volcano DeepSeek declares top_level wire placement like the rest of the
    // DeepSeek v4 family (effort-only NoThinkingDialect passes it through
    // unchanged either way).
    let (thinking, top_level, nested) = resolve_reasoning_wire_controls(
        "deepseek-v4-flash-volcano",
        "https://ark.cn-beijing.volces.com/api/v3",
        true,
        Some("max"),
    );
    assert!(thinking.is_empty());
    assert_eq!(top_level, Some("max"));
    assert!(nested.is_none());
}

#[test]
fn model_effort_graded_reflects_registry_declared_gradation() {
    // Registry-declared wire placement means the model is vendor-verified
    // graded (qwen / DeepSeek v4 on DashScope, official DeepSeek, OpenCode
    // Zen), so "graded" is true. Undeclared models follow the dialect default,
    // where binary-switch dialects omit the tier.
    assert!(model_effort_graded("qwen3.7-max-alibaba"));
    assert!(model_effort_graded("deepseek-v4-flash-0731-alibaba"));
    assert!(model_effort_graded("deepseek-flash-official"));
    // Effort-only dialects keep the gradation.
    assert!(model_effort_graded("deepseek-v4-flash-volcano"));
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

#[test]
fn responses_request_body_uses_function_tools_and_nested_reasoning() {
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object"}
        }
    }]);
    let request = build_request_body(
        "gpt-5.5",
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
}

#[test]
fn responses_force_off_override_emits_none_effort_on_wire() {
    // The truncation ladder's last-resort force-off fallback (thinking_disabled_override) must
    // actually disable thinking on the Responses wire. gpt-5.x uses NoThinkingDialect (no
    // thinking field), so the only lever is reasoning.effort="none"; without the override the
    // ladder's "low" effort would keep thinking on — the pre-fix no-op.
    let messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("hi".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let effort = super::super::reasoning::apply_thinking_force_off_effort(
        true,
        crate::ai::provider::ApiProvider::Compatible,
        "gpt-5.5",
        "https://dataagent-dev-llm.bytedance.net/v1",
        Some("low"),
    );
    assert_eq!(effort, Some("none"));
    let request = build_request_body(
        "gpt-5.5", &messages, false, false, None, None, None, effort, None, None, None,
    );
    let body = super::super::build_responses_request_body(&request);
    assert_eq!(body["reasoning"]["effort"], "none");
    assert_eq!(body["reasoning"]["summary"], "auto");
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
    let request = build_request_body(
        "gpt-5.5",
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

    let request_without_function_tools = build_request_body(
        "gpt-5.5",
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
    let request = build_request_body(
        "gpt-5.5", &messages, false, false, None, None, None, None, None, None, None,
    );

    let chat_body = serde_json::to_value(&request).expect("request body should serialize");
    assert!(chat_body.get("tools").is_none());
    assert!(chat_body.get("tool_choice").is_none());

    let responses_body = super::super::build_responses_request_body(&request);
    assert!(responses_body.get("tools").is_none());
    assert!(responses_body.get("tool_choice").is_none());
}
