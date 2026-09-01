use super::{
    ModelStrengthTier, SubagentTaskDifficulty, agent_model_tier, api_key_for_model,
    auto_subagent_model_for_agent, classify_subagent_task_difficulty, context_window_tokens,
    default_model, determine_model, determine_vl_model, enable_thinking, endpoint_for_model,
    endpoint_supports_anonymous_auth, initial_model, max_output_tokens,
    merge_agent_tier_with_difficulty, model_adapter, model_platform_label, model_quality_tier,
    parse_disabled_model_tokens, reasoning_in_content_enabled, request_model_name,
    request_protocol_dialect,
};
use crate::ai::agents::{AgentManifest, AgentMode, AgentModelTier};
use crate::ai::cli::ParsedCli;
use crate::ai::config_schema::AiConfig;
use crate::ai::model_names::ModelDef;
use crate::ai::provider::{
    ALIBABA_DEFAULT_ENDPOINT, ApiProvider, ModelQualityTier, OPENCODE_DEFAULT_ENDPOINT,
    OPENROUTER_ENDPOINT,
};
use crate::ai::request_protocol::RequestProtocolDialect;
use serde_json::json;

fn manifest(name: &str, description: &str, model_tier: Option<AgentModelTier>) -> AgentManifest {
    AgentManifest {
        name: name.to_string(),
        description: description.to_string(),
        mode: AgentMode::Subagent,
        model: None,
        temperature: None,
        max_steps: None,
        prompt: String::new(),
        system_prompt: None,
        tools: Vec::new(),
        tool_groups: Vec::new(),
        mcp_servers: Vec::new(),
        disable_mcp_tools: false,
        validated_claims: false,
        model_tier,
        disabled: false,
        hidden: false,
        color: None,
        source_path: None,
    }
}

#[test]
fn light_subagent_tasks_use_light_tier() {
    assert_eq!(
        classify_subagent_task_difficulty(
            "Locate task tool",
            "Find where the task tool is implemented and summarize the file."
        ),
        SubagentTaskDifficulty::Light
    );
}

#[test]
fn heavy_subagent_tasks_use_heavy_tier() {
    assert_eq!(
        classify_subagent_task_difficulty(
            "Debug end-to-end failure",
            "Investigate a failing build across multiple files, implement fixes, run tests, and summarize remaining risks."
        ),
        SubagentTaskDifficulty::Heavy
    );
}

#[test]
fn heavy_subagent_model_prefers_tool_capable_thinking_model() {
    let model = auto_subagent_model_for_agent(
        &manifest(
            "build",
            "Autonomous execution agent",
            Some(AgentModelTier::Heavy),
        ),
        "Debug end-to-end failure",
        "Investigate a failing build across multiple files, implement fixes, run tests, and summarize remaining risks.",
    );
    let def = super::model_names::find_by_identifier(&model).expect("selected model must exist");
    assert!(def.tools_default_enabled);
    assert!(def.enable_thinking);
    assert_eq!(def.quality_tier, ModelQualityTier::Flagship);
}

#[test]
fn standard_subagent_model_prefers_high_quality_tier() {
    let model = auto_subagent_model_for_agent(
        &manifest(
            "plan",
            "Read-only planning and analysis agent",
            Some(AgentModelTier::Standard),
        ),
        "Plan a refactor",
        "Review the architecture, compare approaches, and propose a refactor strategy.",
    );
    let def = super::model_names::find_by_identifier(&model).expect("selected model must exist");
    assert!(def.tools_default_enabled);
    assert!(def.enable_thinking);
    assert!(def.quality_tier >= ModelQualityTier::Strong);
}

#[test]
fn light_tier_agents_resolve_to_light_strength() {
    let agent = manifest(
        "light-agent",
        "Generic light-tier agent",
        Some(AgentModelTier::Light),
    );
    assert_eq!(agent_model_tier(&agent), ModelStrengthTier::Light);
}

#[test]
fn plan_agents_default_to_standard_tier() {
    let agent = manifest(
        "plan",
        "Read-only planning and analysis agent",
        Some(AgentModelTier::Standard),
    );
    assert_eq!(agent_model_tier(&agent), ModelStrengthTier::Standard);
}

#[test]
fn build_agents_default_to_heavy_tier() {
    let agent = manifest(
        "build",
        "Autonomous execution and debugging agent",
        Some(AgentModelTier::Heavy),
    );
    assert_eq!(agent_model_tier(&agent), ModelStrengthTier::Heavy);
}

#[test]
fn model_def_parses_request_tpm_limit_when_present() {
    let def: ModelDef = serde_json::from_value(json!({
        "key": "demo",
        "name": "demo-model",
        "adapter": "compatible",
        "quality_tier": "strong",
        "is_vl": false,
        "search_enabled": true,
        "tools_default_enabled": true,
        "enable_thinking": false,
        "request_tpm_limit": 123456
    }))
    .unwrap();
    assert_eq!(def.request_tpm_limit, Some(123456));
}

#[test]
fn model_def_defaults_request_tpm_limit_to_none() {
    let def: ModelDef = serde_json::from_value(json!({
        "key": "demo",
        "name": "demo-model",
        "adapter": "compatible",
        "quality_tier": "strong",
        "is_vl": false,
        "search_enabled": true,
        "tools_default_enabled": true,
        "enable_thinking": false
    }))
    .unwrap();
    assert_eq!(def.request_tpm_limit, None);
}

#[test]
fn model_def_parses_request_protocol_when_present() {
    let def: ModelDef = serde_json::from_value(json!({
        "key": "demo",
        "name": "demo-model",
        "adapter": "compatible",
        "quality_tier": "strong",
        "is_vl": false,
        "search_enabled": true,
        "tools_default_enabled": true,
        "enable_thinking": false,
        "request_protocol": "responses"
    }))
    .unwrap();
    assert_eq!(
        def.request_protocol,
        Some(RequestProtocolDialect::Responses)
    );
}

#[test]
fn model_def_defaults_request_protocol_to_none() {
    let def: ModelDef = serde_json::from_value(json!({
        "key": "demo",
        "name": "demo-model",
        "adapter": "compatible",
        "quality_tier": "strong",
        "is_vl": false,
        "search_enabled": true,
        "tools_default_enabled": true,
        "enable_thinking": false
    }))
    .unwrap();
    assert_eq!(def.request_protocol, None);
}

#[test]
fn request_protocol_dialect_prefers_model_declared_protocol() {
    let endpoint = endpoint_for_model("gpt-5.5", "");
    assert_eq!(
        request_protocol_dialect("gpt-5.5", &endpoint),
        RequestProtocolDialect::Responses
    );
}

#[test]
fn request_protocol_dialect_falls_back_to_endpoint_inference_for_unknown_model() {
    assert_eq!(
        request_protocol_dialect("unknown-model", "https://api.example.com/v1/responses"),
        RequestProtocolDialect::Responses
    );
    assert_eq!(
        request_protocol_dialect(
            "unknown-model",
            "https://api.example.com/v1/chat/completions"
        ),
        RequestProtocolDialect::ChatCompletions
    );
}

#[test]
fn light_tasks_downgrade_heavy_agents_to_standard_tier() {
    assert_eq!(
        merge_agent_tier_with_difficulty(ModelStrengthTier::Heavy, SubagentTaskDifficulty::Light),
        ModelStrengthTier::Standard
    );
}

#[test]
fn disabled_model_tokens_accept_names_and_keys() {
    let models = super::model_names::all();
    let model_def = models
        .first()
        .expect("model registry should contain at least one model");
    let disabled =
        parse_disabled_model_tokens(&format!(" {}, {}\nfoo ", model_def.name, model_def.key));
    assert!(disabled.contains(&model_def.name.to_ascii_lowercase()));
    assert!(disabled.contains(&model_def.key.to_ascii_lowercase()));
    assert!(disabled.contains(&"foo".to_string()));
}

/// Picks a real model name whose adapter is Alibaba as the test input;
/// so the test does not break when individual entries in the model registry (models/) change.
fn first_alibaba_model_name() -> String {
    super::model_names::all()
        .iter()
        .find(|m| m.adapter == ApiProvider::Alibaba)
        .map(|m| m.name.clone())
        .expect("model registry must contain at least one Alibaba-adapter model")
}

fn first_alibaba_vl_model_name() -> Option<String> {
    super::model_names::all()
        .iter()
        .find(|m| m.adapter == ApiProvider::Alibaba && m.is_vl)
        .map(|m| m.name.clone())
}

#[test]
fn known_model_entries_resolve_exactly_by_name() {
    let alibaba = first_alibaba_model_name();
    let alibaba_def = super::model_names::find_by_name(&alibaba).expect("model must exist");
    assert_eq!(
        determine_model(&alibaba),
        super::model_names::model_handle(alibaba_def)
    );
    if let Some(vl) = first_alibaba_vl_model_name() {
        let vl_def = super::model_names::find_by_name(&vl).expect("model must exist");
        assert_eq!(
            determine_vl_model(&vl),
            super::model_names::model_handle(vl_def)
        );
    }
}

#[test]
fn model_keys_resolve_to_model_handles() {
    // Validate the key→handle mapping in reverse against the first real entry in
    // the model registry, rather than hardcoding a specific key.
    let first = super::model_names::all()
        .first()
        .map(|m| {
            (
                m.key.clone(),
                super::model_names::model_handle(m).to_string(),
            )
        })
        .expect("model registry must contain at least one entry");
    assert_eq!(determine_model(&first.0), first.1);
}

#[test]
fn model_key_selects_duplicate_name_provider() {
    let key = "deepseek-v4-flash-opencode";
    let def = super::model_names::find_by_identifier(key)
        .expect("model registry should contain opencode deepseek-v4-flash");

    assert_eq!(def.name, "deepseek-v4-flash");
    assert_eq!(def.adapter, ApiProvider::OpenCode);
    assert_eq!(determine_model(key), key);
    assert_eq!(request_model_name(key), "deepseek-v4-flash");
    assert_eq!(model_adapter(key), ApiProvider::OpenCode);
    assert_eq!(determine_model("DEEPSEEK_V4_FLASH_OPENCODE"), key);
    assert_eq!(determine_model("deepseek-v4-flash opencode"), key);
}

#[test]
fn deepseek_v4_flash_routes_declare_full_context_budget() {
    // DeepSeek V4 Flash's 393,216 output cap needs the 1M context window; it must not
    // fall back to strong tier's conservative 200K default, or the builder would clamp max_tokens too early.
    for key in [
        "deepseek-v4-flash-opencode",
        "deepseek-v4-flash-0731-alibaba",
    ] {
        assert_eq!(context_window_tokens(key), 1_000_000, "{key}");
        assert_eq!(max_output_tokens(key), Some(393_216), "{key}");
    }
}

#[test]
fn platform_changes_model_handle_but_legacy_adapter_handle_still_resolves() {
    let volcano = "deepseek-v4-flash-volcano";
    let def = super::model_names::find_by_identifier(volcano)
        .expect("model registry should contain volcano deepseek-v4-flash");
    assert_eq!(super::model_names::model_handle(def), volcano);
    assert_eq!(determine_model("deepseek-v4-flash-compatible"), volcano);
    assert_eq!(model_platform_label(volcano), "volcano");
    // Volcengine Coding Plan's official config declares 1,024K context / 393,216 output.
    // It must be configured explicitly here and must not fall back to flagship's conservative 256K default window.
    assert_eq!(context_window_tokens(volcano), 1_024_000);
    assert_eq!(max_output_tokens(volcano), Some(393_216));
}

#[test]
fn volcano_models_keep_dedicated_reasoning_channel() {
    // Volcengine GLM/DeepSeek streams return reasoning_content separately; if the content
    // demux is also armed, the body is held back until the stream ends and would be misrouted into thinking on a response event.
    assert!(!reasoning_in_content_enabled("deepseek-v4-flash-volcano"));
    assert!(!reasoning_in_content_enabled("glm-5.2-volcano"));
}

#[test]
fn initial_model_normalizes_configured_model_key() {
    let mut cli = ParsedCli::default();
    cli.model = None;
    let model = initial_model(&cli);
    let configured = crate::commonw::configw::get_all_config()
        .get_opt(AiConfig::MODEL_DEFAULT)
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    if let Some(key) = configured
        && let Some(def) = super::model_names::find_by_identifier(&key)
    {
        assert_eq!(model, super::model_names::model_handle(def));
    }
}

#[test]
fn known_model_entries_carry_adapter_and_quality_tier() {
    let name = first_alibaba_model_name();
    let def = super::model_names::find_by_name(&name).expect("model must exist");
    assert_eq!(model_adapter(&name), def.adapter);
    assert_eq!(model_quality_tier(&name), def.quality_tier);
}

#[test]
fn endpoint_for_known_model_prefers_model_config_over_global_fallback() {
    // Any entry that explicitly declares an endpoint in the model registry should prefer its own config.
    let (name, expected) = super::model_names::all()
        .iter()
        .filter_map(|m| {
            let endpoint = m
                .endpoint
                .as_deref()
                .map(str::trim)
                .filter(|e| !e.is_empty())?;
            let resolved = super::model_names::find_by_identifier(&m.name)?;
            std::ptr::eq(resolved, *m).then(|| (m.name.clone(), endpoint.to_string()))
        })
        .next()
        .expect("model registry must contain a selectable entry with an explicit endpoint");
    let endpoint = endpoint_for_model(
        &name,
        "https://example.com/should-not-be-used/v1/chat/completions",
    );
    assert_eq!(endpoint, expected);
}

#[test]
fn deepseek_v4_flash_selector_keeps_opencode_compatibility() {
    assert_eq!(
        determine_model("deepseek-v4-flash"),
        "deepseek-v4-flash-opencode"
    );
    assert_eq!(
        determine_model("deepseek-v4-flash-0731-alibaba"),
        "deepseek-v4-flash-0731-alibaba"
    );
}

#[test]
fn endpoint_for_alibaba_model_prefers_model_config() {
    // Find a model with an Alibaba adapter that has an endpoint configured, ensuring the model config is used.
    let (name, expected) = super::model_names::all()
        .iter()
        .find_map(|m| {
            if m.adapter != ApiProvider::Alibaba {
                return None;
            }
            m.endpoint
                .as_deref()
                .map(str::trim)
                .filter(|e| !e.is_empty())
                .map(|e| (m.name.clone(), e.to_string()))
        })
        .expect("model registry must contain at least one Alibaba entry with endpoint");
    let endpoint = endpoint_for_model(&name, "");
    assert_eq!(endpoint, expected);
    assert_eq!(endpoint, ALIBABA_DEFAULT_ENDPOINT);
}

#[test]
fn alibaba_model_entries_accept_alibaba_api_key_config() {
    let name = first_alibaba_model_name();
    let key = api_key_for_model(&name, "fallback-key");
    assert!(!key.is_empty());
}

#[test]
fn openrouter_models_use_openrouter_endpoint_in_config() {
    // Any model configured with an openrouter endpoint should go through openrouter.
    let openrouter_model = super::model_names::all()
        .iter()
        .find(|m| {
            m.endpoint
                .as_deref()
                .map(|e| e.trim().eq_ignore_ascii_case(OPENROUTER_ENDPOINT))
                .unwrap_or(false)
        })
        .map(|m| m.name.clone());
    if let Some(name) = openrouter_model {
        let endpoint = endpoint_for_model(&name, "");
        assert_eq!(endpoint, OPENROUTER_ENDPOINT);
    }
}

#[test]
fn known_model_without_endpoint_uses_provider_default_before_global_fallback() {
    let model = super::model_names::all()
        .iter()
        .find(|m| m.adapter == ApiProvider::OpenCode && m.endpoint.is_none())
        .map(|m| super::model_names::model_handle(m).to_string())
        .expect("model registry must contain at least one OpenCode entry without endpoint");
    let endpoint = endpoint_for_model(&model, "https://example.com/v1/chat/completions");
    assert_eq!(endpoint, OPENCODE_DEFAULT_ENDPOINT);
}

#[test]
fn unknown_model_uses_global_fallback_endpoint() {
    let endpoint = endpoint_for_model("custom-model", "https://example.com/v1/chat/completions");
    assert_eq!(endpoint, "https://example.com/v1/chat/completions");
}

#[test]
fn localhost_endpoint_supports_anonymous_auth() {
    assert!(endpoint_supports_anonymous_auth(
        "http://127.0.0.1:11434/v1/chat/completions"
    ));
    assert!(endpoint_supports_anonymous_auth(
        "http://localhost:11434/v1/chat/completions"
    ));
    assert!(!endpoint_supports_anonymous_auth(
        "https://openrouter.ai/api/v1/chat/completions"
    ));
}

#[test]
fn default_model_prefers_high_quality_alibaba_or_compatible_model() {
    // default_model is filtered by Alibaba / Compatible adapter in choose_default_model_name,
    // then falls back to the full set and picks the highest quality_tier. The invariant is
    //  1. must be non-vl
    //  2. quality_tier must be at least as high as every candidate under the same adapter preference
    let def = super::model_names::find_by_identifier(&default_model())
        .expect("default model must exist in model registry");
    assert!(!def.is_vl, "default model should be non-VL");

    let best_non_vl_tier = super::model_names::all()
        .iter()
        .filter(|m| !m.is_vl)
        .map(|m| m.quality_tier)
        .max()
        .expect("model registry must contain at least one non-VL model");
    assert_eq!(def.quality_tier, best_non_vl_tier);
}

#[test]
fn opencode_model_entries_do_not_advertise_thinking_when_disabled() {
    // This used to hard-assert the fixed model name gpt-5.4-pro; it now targets any
    // opencode model that explicitly declares enable_thinking=false in the model registry,
    // checking that enable_thinking() matches the config. Specific registry entry changes
    // no longer break this test, while still guarding the invariant of not misreading false as true.
    let candidate = super::model_names::all()
        .iter()
        .find(|m| m.adapter == ApiProvider::OpenCode && !m.enable_thinking)
        .map(|m| m.name.clone());
    if let Some(name) = candidate {
        assert!(!enable_thinking(&name));
    }
}

#[test]
/// Regression test: the api_key_config_key field in the model registry may be encrypted (enc:...),
/// so api_key_for_model must decrypt it first before using it as the key name in configw.
/// Before the fix: the encrypted string was used directly as the key name -> not found -> fallthrough to the global api_key -> 401.
fn api_key_config_key_decrypts_before_configw_lookup() {
    use crate::ai::test_support::ENV_LOCK;
    use crate::commonw::{configw, secret};

    let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());

    // Find a model in the registry whose api_key_config_key is encrypted
    let target = super::model_names::all().iter().find_map(|m| {
        let enc = m.api_key_config_key.as_deref()?;
        if !secret::is_encrypted(enc) {
            return None;
        }
        secret::decrypt(enc)
            .ok()
            .map(|plain| (m.key.clone(), plain))
    });

    let Some((model_key, config_key_name)) = target else {
        // No encrypted api_key_config_key entries exist, so this scenario cannot be tested
        return;
    };

    // secret_path() depends on config_path(); changing CONFIGW_PATH moves the key file path.
    // Read the real key file before changing CONFIGW_PATH and copy it into a temp directory.
    let real_secret_path = configw::config_path()
        .parent()
        .map(|p| p.join(".configW.secret"))
        .filter(|p| p.exists());
    let Some(real_secret_path) = real_secret_path else {
        eprintln!("skip: secret key file not found");
        return;
    };

    // Create a temp directory and copy the key file into it (keeping it decryptable)
    let temp_dir =
        std::env::temp_dir().join(format!("rt_enc_test_{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&temp_dir).unwrap();
    let temp_secret = temp_dir.join(".configW.secret");
    std::fs::copy(&real_secret_path, &temp_secret).unwrap();

    // Write the temporary config file
    let cfg_path = temp_dir.join("configw");
    let test_value = "test-encrypted-key-resolution-value";
    std::fs::write(&cfg_path, format!("{config_key_name} = {test_value}\n")).unwrap();

    let old_cfg = std::env::var_os("CONFIGW_PATH");
    unsafe { std::env::set_var("CONFIGW_PATH", &cfg_path) };
    configw::refresh();

    // Deliberately set global_fallback to a value that can never be right, so there is no fallthrough
    let resolved = api_key_for_model(&model_key, "GLOBAL_FALLBACK_SHOULD_NOT_BE_USED");

    // Restore the environment
    match old_cfg {
        Some(value) => unsafe { std::env::set_var("CONFIGW_PATH", value) },
        None => unsafe { std::env::remove_var("CONFIGW_PATH") },
    }
    configw::refresh();
    let _ = std::fs::remove_dir_all(&temp_dir);

    assert_eq!(
        resolved, test_value,
        "api_key_config_key (encrypted) should decrypt to '{config_key_name}' \
             and resolve from configw, not fall through to global fallback"
    );
}
