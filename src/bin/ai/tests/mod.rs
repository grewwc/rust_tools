//! Unit tests for the `ai` module, split into themed submodules.
//!
//! `mod tests;` in `src/bin/ai/mod.rs` resolves to this directory. Each submodule
//! imports the `ai` module's internals via `use super::super::*` (the `tests`
//! module is a descendant of `ai`, so the ancestor's private items are visible).
//! Shared test helper functions live here and are reached from submodules via
//! `use super::*;`.

use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};

use serde_json::Value;

use super::{
    history::{Message, append_history_messages},
    types::{FunctionCall, ToolCall},
};
use super::*;

mod cli;
mod compress;
mod dedup;
mod driver;
mod files;
mod history;
mod mid_turn_compress;
mod mid_turn_llm;
mod models;
mod prompt;
mod session;
mod stream;
mod tools;

fn any_model_name() -> String {
    model_names::all()
        .first()
        .map(|m| m.name.clone())
        .expect("model registry is empty")
}

fn vl_model_name_at(index: usize) -> Option<String> {
    model_names::all()
        .iter()
        .filter(|m| m.is_vl)
        .nth(index)
        .map(|m| m.name.clone())
}

fn any_vl_model_name() -> String {
    vl_model_name_at(0).unwrap_or_else(any_model_name)
}

fn vl_model_handle_at(index: usize) -> Option<String> {
    model_names::all()
        .iter()
        .filter(|m| m.is_vl)
        .nth(index)
        .map(|m| model_names::model_handle(m))
}

fn any_vl_model_handle() -> String {
    vl_model_handle_at(0).unwrap_or_else(any_model_name)
}

fn test_app_with_cancel_stream(cancel_stream: Arc<AtomicBool>) -> types::App {
    types::App {
        cli: super::cli::ParsedCli::default(),
        hooks: Default::default(),
        config: types::AppConfig {
            api_key: String::new(),
            base_history_file: PathBuf::new(),
            history_file: PathBuf::new(),
            endpoint: String::new(),
            vl_default_model: any_vl_model_name(),
            history_max_chars: 12000,
            history_keep_last: 8,
            history_summary_max_chars: 4000,
            intent_model: None,
        },
        session_id: String::new(),
        session_history_file: PathBuf::new(),
        active_persona: persona::default_persona(),
        client: reqwest::Client::builder().build().unwrap(),
        current_model: any_model_name(),
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
        cancel_stream,
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

fn extract_stub_file_path(stub: &str) -> Option<String> {
    stub.lines()
        .find_map(|line| line.strip_prefix("归档文件: "))
        .map(str::to_string)
}

fn read_file_call_pair(id: &str, path: &str, content: &str) -> (Message, Message) {
    let assistant = Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: format!(r#"{{"filePath":"{path}"}}"#),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    };
    let tool = Message {
        role: "tool".to_string(),
        content: Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: Some(id.to_string()),
        reasoning_content: None,
    };
    (assistant, tool)
}

fn tool_call_pair(id: &str, tool_name: &str, arguments: &str, content: &str) -> (Message, Message) {
    let assistant = Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: tool_name.to_string(),
                arguments: arguments.to_string(),
            },
        }]),
        tool_call_id: None,
        reasoning_content: None,
    };
    let tool = Message {
        role: "tool".to_string(),
        content: Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: Some(id.to_string()),
        reasoning_content: None,
    };
    (assistant, tool)
}

fn structured_history_messages() -> Vec<Message> {
    vec![
        Message {
            role: "system".to_string(),
            content: Value::String("system prompt".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::Array(vec![serde_json::json!({
                "type": "text",
                "text": "hello"
            })]),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "demo".to_string(),
                    arguments: r#"{"x":1}"#.to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "tool".to_string(),
            content: Value::String("tool output".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
            reasoning_content: None,
        },
        Message {
            role: "assistant".to_string(),
            content: Value::String("done".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ]
}

/// sqlite first open / WAL initialization can transiently report SQLITE_IOERR (mapped to WouldBlock) under parallel test load;
/// it is a transient error: retry a few times with short backoff (same WouldBlock retry semantics as the production async path).
fn append_history_messages_retry_transient(
    path: &std::path::Path,
    messages: &[Message],
) -> std::io::Result<()> {
    let mut last_err: Option<std::io::Error> = None;
    for _ in 0..5 {
        match append_history_messages(path, messages) {
            Ok(()) => return Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                last_err = Some(err);
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(err) => return Err(err),
        }
    }
    Err(last_err.unwrap_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            "transient append retries exhausted",
        )
    }))
}
