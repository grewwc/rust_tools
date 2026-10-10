use super::super::*;
use crate::ai::{cli::ParsedCli, types::AppConfig};
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};

pub(crate) fn test_app() -> App {
    // Test processes have no main(); ensure the rustls ring provider is
    // installed before constructing reqwest clients (reqwest 0.13 panics at
    // Client::build() with rustls-no-provider and no installed provider).
    rust_tools::ensure_rustls_provider();
    App {
        cli: ParsedCli::default(),
        scoped_preflight_required: Vec::new(),
        hooks: Default::default(),
        config: AppConfig {
            api_key: String::new(),
            base_history_file: PathBuf::new(),
            history_file: PathBuf::new(),
            endpoint: String::new(),
            vl_default_model: String::new(),
            history_max_chars: 0,
            history_keep_last: 0,
            history_summary_max_chars: 0,
            intent_model: None,
        },
        session_id: String::new(),
        session_history_file: PathBuf::new(),
        active_persona: crate::ai::persona::default_persona(),
        client: reqwest::Client::builder().build().unwrap(),
        current_model: String::new(),
        current_agent: String::new(),
        current_agent_manifest: None,
        pending_files: None,
        forced_skills: Vec::new(),
        forced_skill_source: None,
        pending_skill_continuation: None,
        forced_question: None,
        attached_image_files: Vec::new(),
        shutdown: Arc::new(AtomicBool::new(false)),
        streaming: Arc::new(AtomicBool::new(false)),
        cancel_stream: Arc::new(AtomicBool::new(false)),
        ignore_next_prompt_interrupt: false,
        prompt_editor: None,
        agent_context: None,
        last_skill_bias: None,
        os: crate::ai::driver::new_local_kernel(),
        agent_reload_counter: None,
        observers: vec![Box::new(
            crate::ai::driver::thinking::ThinkingOrchestrator::new(),
        )],
        last_known_prompt_tokens: None,
        last_known_cached_prompt_tokens: None,
        goal_mode: None,
        last_turn_had_tool_calls: false,
        last_turn_interrupted: false,
        prune_marks: Default::default(),
        turn_reasoning_items: Default::default(),
        stale_patch_targets: Default::default(),
        tool_middlewares: Vec::new(),
        llm_middlewares: Vec::new(),
    }
}
/// Finds a real OpenAi-adapter model name as test input, avoiding hardcoded
/// model strings that would break tests when the model registry (models/)
/// changes.
pub(super) fn first_openai_model_name() -> Option<String> {
    crate::ai::model_names::all()
        .iter()
        .find(|m| m.adapter == crate::ai::provider::ApiProvider::OpenAi)
        .map(|m| m.name.clone())
}

pub(super) fn first_openai_vl_model_name() -> Option<String> {
    crate::ai::model_names::all()
        .iter()
        .find(|m| m.adapter == crate::ai::provider::ApiProvider::OpenAi && m.is_vl)
        .map(|m| m.name.clone())
}

pub(super) fn first_alibaba_vl_model_name() -> Option<String> {
    crate::ai::model_names::all()
        .iter()
        .find(|m| m.adapter == crate::ai::provider::ApiProvider::Alibaba && m.is_vl)
        .map(|m| m.name.clone())
}

/// Returns the **unique key** of the adapter's first model (not the `name`).
/// The production path locates models by key (log identifiers look like
/// `glm-5.2-opencode`), while a `name` (e.g. `glm-5.2`) may be shared by entries
/// of multiple adapters/platforms; looking up by name would hit an ambiguous
/// entry and resolve the wrong adapter dialect. Tests must use the key,
/// consistent with production.
pub(super) fn first_model_key_for_adapter(adapter: crate::ai::provider::ApiProvider) -> Option<String> {
    crate::ai::model_names::all()
        .iter()
        .find(|m| m.adapter == adapter)
        .map(|m| m.key.clone())
}
/// Reuses the production clamping logic to build the expected `max_tokens`
/// fragment for the wire assertion: `,"max_tokens":N` when the model declares
/// max_output_tokens, otherwise an empty string.
pub(super) fn expected_max_tokens_field(model: &str, messages: &[Message]) -> String {
    match super::super::models::max_output_tokens(model) {
        Some(model_max) => {
            let clamped = clamp_max_tokens_for_prompt(model, messages, None, model_max, None);
            format!(r#","max_tokens":{clamped}"#)
        }
        None => String::new(),
    }
}
pub(super) fn test_attachment_assets_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ai-reference-assets-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub(super) fn write_image_snapshot(assets_dir: &std::path::Path, name: &str, bytes: &[u8]) -> String {
    let dir = assets_dir
        .join("attachments")
        .join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path.to_string_lossy().into_owned()
}
