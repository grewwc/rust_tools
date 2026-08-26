// =============================================================================
// Shared utilities for port-middleware tests (test-only compilation)
// =============================================================================
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::ai::cli::ParsedCli;
use crate::ai::types::AgentContext;
use crate::ai::persona::default_persona;
use crate::ai::types::{App, AppConfig};

/// Builds a minimal usable App for port-middleware tests (the mock client/executor does not read App fields).
pub fn test_app() -> App {
    let model = crate::ai::model_names::all()
        .first()
        .map(|m| m.name.clone())
        .unwrap_or_default();
    App {
        cli: ParsedCli::default(),
        hooks: Default::default(),
        config: AppConfig {
            api_key: String::new(),
            base_history_file: PathBuf::new(),
            history_file: PathBuf::new(),
            endpoint: String::new(),
            vl_default_model: model.clone(),
            history_max_chars: 12000,
            history_keep_last: 8,
            history_summary_max_chars: 4000,
            intent_model: None,
        },
        session_id: String::new(),
        session_history_file: PathBuf::new(),
        active_persona: default_persona(),
        client: reqwest::Client::new(),
        current_model: model,
        current_agent: "test-agent".to_string(),
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
        agent_context: Some(AgentContext {
            tools: Vec::new(),
            mcp_servers: Default::default(),
            max_iterations: 64 * 64,
        }),
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
