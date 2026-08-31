//! Shared helpers for the `execution/` cluster test modules: app builders,
//! message factories, and the replay-tool registration hook.

use super::super::*;

pub(super) use crate::ai::{
    cli::ParsedCli,
    driver::{runtime_ctx::SUBAGENT_CWD, signal},
    types::{
        AgentContext, App, AppConfig, FunctionCall, FunctionDefinition, ToolDefinition,
        ToolResult,
    },
};
pub(super) use aios_kernel::primitives::ResourceLimit;
pub(super) use rust_tools::cw::SkipMap;
pub(super) use serde_json::Value;
pub(super) use std::fs;
pub(super) use std::path::PathBuf;
pub(super) use std::sync::{Arc, atomic::AtomicBool};
pub(super) use std::time::{Duration, Instant};

pub(super) const TEST_REPLAY_TOOL: &str = "test_stable_read";

inventory::submit!(crate::ai::tools::ToolReplayRegistration {
    name: TEST_REPLAY_TOOL,
});

/// Take a non-locking McpClient snapshot (consistent with the production orchestrator's
/// routing_snapshot pattern). Passing `shared.lock().unwrap()`'s guard directly into
/// handle_iteration_execution would keep the guard alive until the whole call statement
/// ends, while the adapter locks the same mutex again during execution → self-deadlock.
pub(super) fn mcp_snapshot(shared: &SharedMcpClient) -> McpClient {
    shared.lock().unwrap().routing_snapshot()
}

pub(super) fn test_app_with_tools(tool_names: &[&str]) -> App {
    App {
        cli: ParsedCli::default(),
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
        session_id: "test".to_string(),
        session_history_file: PathBuf::new(),
        active_persona: crate::ai::persona::default_persona(),
        client: reqwest::Client::builder().build().unwrap(),
        current_model: String::new(),
        current_agent: "build".to_string(),
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
            tools: tool_names
                .iter()
                .map(|name| ToolDefinition {
                    tool_type: "function".to_string(),
                    function: FunctionDefinition {
                        name: (*name).to_string(),
                        description: String::new(),
                        parameters: serde_json::json!({}),
                    },
                })
                .collect(),
            mcp_servers: SkipMap::default(),
            max_iterations: 16,
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
        hooks: Default::default(),
    }
}

pub(super) fn test_tool_call(id: &str, name: &str, arguments: serde_json::Value) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        },
    }
}

pub(super) fn assistant_tool_call_message(tool_call: ToolCall) -> Message {
    Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![tool_call]),
        tool_call_id: None,
        reasoning_content: None,
    }
}

pub(super) fn tool_result_message(id: &str, content: &str) -> Message {
    Message {
        role: "tool".to_string(),
        content: Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: Some(id.to_string()),
        reasoning_content: None,
    }
}

/// Replay an `assistant(tool_calls)` + `tool` message sequence into the stale-target
/// ledger in chronological order, equivalent to the accumulated effect of calling
/// [`update_stale_patch_targets`] round by round at runtime. Guard tests can keep
/// expressing scenarios as intuitive “history messages” and then assert on the gate
/// behavior derived from the ledger — covering the full fixed chain
/// (messages → ledger → guard).
pub(super) fn ledger_from_messages(messages: &[Message]) -> rustc_hash::FxHashSet<PathBuf> {
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut tool_results: Vec<crate::ai::types::ToolResult> = Vec::new();
    for message in messages {
        if let Some(calls) = &message.tool_calls {
            tool_calls.extend(calls.iter().cloned());
        }
        if message.role == "tool" {
            if let (Some(id), Some(content)) =
                (message.tool_call_id.as_deref(), message.content.as_str())
            {
                tool_results.push(tool_result(id, content));
            }
        }
    }
    let mut ledger = rustc_hash::FxHashSet::default();
    update_stale_patch_targets(&mut ledger, &tool_calls, &tool_results);
    ledger
}

pub(super) fn tool_result(id: &str, content: &str) -> crate::ai::types::ToolResult {
    crate::ai::types::ToolResult {
        tool_call_id: id.to_string(),
        content: content.to_string(),
    }
}

