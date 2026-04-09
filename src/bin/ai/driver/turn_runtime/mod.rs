mod finalize;
mod iteration;
mod prepare;
mod tool_round;

use std::{fs, path::PathBuf};

use colored::Colorize;
use serde_json::Value;

use crate::ai::{
    history::{Message, SessionStore, append_history_messages, build_context_history},
    mcp::McpClient,
    request::{self, do_request_messages},
    stream,
    types::{App, StreamOutcome, StreamResult},
};

use super::{
    drain_response, input,
    print::{print_assistant_banner, print_tool_output_block},
    reflection, tools,
};
use finalize::finalize_turn;
use iteration::{execute_turn_iteration, refresh_skill_turn_for_iteration};
use prepare::prepare_turn;
use tool_round::handle_iteration_execution;
#[cfg(test)]
use tool_round::prepare_tool_result;

const MAX_TOOL_RESULT_INLINE_CHARS: usize = 32_000;
const TOOL_OVERFLOW_PREVIEW_CHARS: usize = 800;

struct PreparedToolResult {
    content_for_model: String,
    content_for_terminal: String,
}

struct LargeToolSummary {
    body: String,
    summary: String,
    top_level_keys: Vec<String>,
    field_samples: Vec<String>,
}

struct TurnPreparation {
    skill_turn: super::skill_runtime::SkillTurnGuard,
    messages: Vec<Message>,
    turn_messages: Vec<Message>,
    persisted_turn_messages: usize,
    max_iterations: usize,
}

enum IterationExecution {
    Exit(TurnOutcome),
    RequestFailed(String),
    FinalResponse(StreamResult),
    ToolCall(StreamResult),
}

enum TurnLoopStep {
    Continue,
    Break,
    Return(TurnOutcome),
}

// #region debug-point agent-hang:reporter
#[cfg(feature = "agent-hang-debug")]
pub(in crate::ai) fn report_agent_hang_debug(
    run_id: &'static str,
    hypothesis_id: &'static str,
    location: &'static str,
    msg: &'static str,
    data: Value,
) {
    std::thread::spawn(move || {
        let mut debug_server_url = "http://127.0.0.1:7777/event".to_string();
        let mut debug_session_id = "agent-hang".to_string();
        if let Ok(env_text) = fs::read_to_string(".dbg/agent-hang.env") {
            for line in env_text.lines() {
                if let Some(value) = line.strip_prefix("DEBUG_SERVER_URL=") {
                    if !value.trim().is_empty() {
                        debug_server_url = value.trim().to_string();
                    }
                } else if let Some(value) = line.strip_prefix("DEBUG_SESSION_ID=") {
                    if !value.trim().is_empty() {
                        debug_session_id = value.trim().to_string();
                    }
                }
            }
        }
        let payload = serde_json::json!({
            "sessionId": debug_session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data,
            "ts": chrono::Utc::now().timestamp_millis(),
        });
        if let Ok(client) = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_millis(300))
            .build()
        {
            let _ = client.post(debug_server_url).json(&payload).send();
        }
    });
}

#[cfg(not(feature = "agent-hang-debug"))]
pub(in crate::ai) fn report_agent_hang_debug(
    _run_id: &'static str,
    _hypothesis_id: &'static str,
    _location: &'static str,
    _msg: &'static str,
    _data: Value,
) {
}
// #endregion

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TurnOutcome {
    Continue,
    Quit,
}

#[crate::ai::agent_hang_span(
    "pre-fix",
    "A",
    "turn_runtime::run_turn",
    "[DEBUG] run_turn started",
    "[DEBUG] run_turn finished",
    {
        "history_count": history_count,
        "question_len": question.chars().count(),
        "model": next_model.as_str(),
        "one_shot_mode": one_shot_mode,
        "should_quit": should_quit,
    },
    {
        "ok": __agent_hang_result.is_ok(),
        "outcome": __agent_hang_result
            .as_ref()
            .map(|v| format!("{:?}", v))
            .unwrap_or_else(|err| err.to_string()),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
pub(super) async fn run_turn(
    app: &mut App,
    mcp_client: &mut McpClient,
    skill_manifests: &[crate::ai::skills::SkillManifest],
    history_count: usize,
    question: String,
    next_model: String,
    one_shot_mode: bool,
    should_quit: bool,
) -> Result<TurnOutcome, Box<dyn std::error::Error>> {
    // 1. prepare 
    let TurnPreparation {
        mut skill_turn,
        mut messages,
        mut turn_messages,
        mut persisted_turn_messages,
        max_iterations,
    } = prepare_turn(
        app,
        mcp_client,
        skill_manifests,
        history_count,
        &question,
        &next_model,
    )
    .await?;

    let mut iteration = 0usize;
    let mut force_final_response = false;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    loop {
        iteration += 1;
        // 2. re-choose skill
        refresh_skill_turn_for_iteration(
            app,
            mcp_client,
            skill_manifests,
            &question,
            iteration,
            &mut skill_turn,
            &mut messages,
        )
        .await;
        // 3. execute
        let execution = execute_turn_iteration(
            app,
            &next_model,
            &mut messages,
            &turn_messages,
            one_shot_mode,
            &mut persisted_turn_messages,
            should_quit,
            force_final_response,
            iteration,
        )
        .await?;
        // 4. handle execution result 
        match handle_iteration_execution(
            app,
            mcp_client,
            execution,
            &mut messages,
            &mut turn_messages,
            one_shot_mode,
            &mut persisted_turn_messages,
            &mut final_assistant_text,
            &mut final_assistant_recorded,
            &mut force_final_response,
            iteration,
            max_iterations,
        )? {
            TurnLoopStep::Continue => {}
            TurnLoopStep::Break => break,
            TurnLoopStep::Return(outcome) => return Ok(outcome),
        }
    }
    // 5. finilization
    finalize_turn(
        app,
        &next_model,
        &question,
        &final_assistant_text,
        final_assistant_recorded,
        &mut turn_messages,
        one_shot_mode,
        &mut persisted_turn_messages,
        should_quit,
    )
    .await
}

fn persist_pending_turn_messages(
    app: &App,
    one_shot_mode: bool,
    turn_messages: &[Message],
    persisted_turn_messages: &mut usize,
) {
    if one_shot_mode || *persisted_turn_messages >= turn_messages.len() {
        return;
    }

    if let Err(err) = append_history_messages(
        &app.session_history_file,
        &turn_messages[*persisted_turn_messages..],
    ) {
        eprintln!("[Warning] Failed to save history: {}", err);
        return;
    }

    *persisted_turn_messages = turn_messages.len();
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool};

    use serde_json::Value;

    use super::*;
    use crate::ai::{
        cli::ParsedCli,
        history::{SessionStore, build_message_arr},
        types::AppConfig,
    };

    fn test_app(history_file: PathBuf) -> App {
        App {
            cli: ParsedCli::default(),
            config: AppConfig {
                api_key: String::new(),
                history_file: history_file.clone(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 12_000,
                history_keep_last: 256,
                history_summary_max_chars: 4_000,
                intent_model: None,
                intent_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("config/intent/intent_model.json"),
            },
            session_id: "test".to_string(),
            session_history_file: history_file,
            client: reqwest::Client::builder().build().unwrap(),
            current_model: String::new(),
            current_agent: "build".to_string(),
            current_agent_manifest: None,
            pending_files: None,
            pending_short_output: false,
            attached_image_files: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            streaming: Arc::new(AtomicBool::new(false)),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            ignore_next_prompt_interrupt: false,
            writer: None,
            prompt_editor: None,
            agent_context: None,
        }
    }

    fn extract_stub_path(stub: &str) -> Option<PathBuf> {
        stub.lines()
            .find_map(|line| line.strip_prefix("- file_path: "))
            .map(PathBuf::from)
    }

    #[test]
    fn persist_pending_turn_messages_only_appends_new_entries() {
        let path =
            std::env::temp_dir().join(format!("ai-turn-history-{}.sqlite", uuid::Uuid::new_v4()));
        let app = test_app(path.clone());

        let mut turn_messages = vec![Message {
            role: "user".to_string(),
            content: Value::String("hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
        }];
        let mut persisted = 0usize;

        persist_pending_turn_messages(&app, false, &turn_messages, &mut persisted);
        assert_eq!(persisted, 1);

        turn_messages.push(Message {
            role: "tool".to_string(),
            content: Value::String("tool output".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
        });
        turn_messages.push(Message {
            role: "assistant".to_string(),
            content: Value::String("done".to_string()),
            tool_calls: None,
            tool_call_id: None,
        });

        persist_pending_turn_messages(&app, false, &turn_messages, &mut persisted);
        assert_eq!(persisted, 3);

        let loaded = build_message_arr(16, &path).unwrap();
        assert_eq!(loaded, turn_messages);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn prepare_tool_result_spills_large_output_to_session_file() {
        let history_file =
            std::env::temp_dir().join(format!("ai-tool-overflow-{}.sqlite", uuid::Uuid::new_v4()));
        let app = test_app(history_file.clone());
        let store = SessionStore::new(history_file.as_path());
        store.ensure_root_dir().unwrap();
        std::fs::write(store.session_history_file(&app.session_id), b"test").unwrap();

        let content = "x".repeat(MAX_TOOL_RESULT_INLINE_CHARS + 256);
        let prepared = prepare_tool_result(&app, "mcp_big_payload", &content);

        assert!(
            prepared
                .content_for_model
                .contains("Output too large; full result saved")
        );
        let path = extract_stub_path(&prepared.content_for_model).unwrap();
        assert!(path.is_absolute());
        assert!(path.exists());
        let saved = std::fs::read_to_string(&path).unwrap();
        assert_eq!(saved, content);

        let _ = store.delete_session(&app.session_id);
        assert!(!path.exists());
    }

    #[test]
    fn prepare_tool_result_json_stub_includes_keys_and_samples() {
        let history_file = std::env::temp_dir().join(format!(
            "ai-tool-overflow-json-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file.clone());
        let store = SessionStore::new(history_file.as_path());
        store.ensure_root_dir().unwrap();
        std::fs::write(store.session_history_file(&app.session_id), b"test").unwrap();

        let payload = serde_json::json!({
            "id": 123,
            "name": "example payload",
            "items": [
                { "kind": "doc", "token": "abc", "size": 42 }
            ],
            "meta": {
                "source": "mcp",
                "ok": true
            }
        });
        let content = format!("{}{}", payload, " ".repeat(MAX_TOOL_RESULT_INLINE_CHARS));
        let prepared = prepare_tool_result(&app, "mcp_json_payload", &content);

        assert!(prepared.content_for_model.contains("- top_level_keys:"));
        assert!(prepared.content_for_model.contains("id"));
        assert!(prepared.content_for_model.contains("name"));
        assert!(prepared.content_for_model.contains("- field_samples:"));
        assert!(prepared.content_for_model.contains("items:"));
        assert!(prepared.content_for_model.contains("meta:"));

        let _ = store.delete_session(&app.session_id);
    }

    #[test]
    fn prepare_tool_result_truncates_terminal_preview_but_keeps_model_content() {
        let history_file =
            std::env::temp_dir().join(format!("ai-tool-preview-{}.sqlite", uuid::Uuid::new_v4()));
        let app = test_app(history_file.clone());

        let mut content = String::new();
        for i in 0..160usize {
            content.push_str(&format!("{}→{}\n", i, "x".repeat(120)));
        }
        assert!(content.chars().count() < MAX_TOOL_RESULT_INLINE_CHARS);

        let prepared = prepare_tool_result(&app, "read_file_lines", &content);

        eprintln!("DEBUG: content chars = {}", content.chars().count());
        eprintln!("DEBUG: content lines = {}", content.lines().count());
        eprintln!("DEBUG: terminal preview len = {}", prepared.content_for_terminal.len());
        eprintln!("DEBUG: terminal preview first 300 chars:\n{}", &prepared.content_for_terminal[..300.min(prepared.content_for_terminal.len())]);
        
        assert_eq!(prepared.content_for_model, content);
        assert!(prepared.content_for_terminal.contains("truncated for terminal preview"));
        assert!(prepared.content_for_terminal.len() < prepared.content_for_model.len());
        assert!(prepared.content_for_terminal.contains("0→"));
        assert!(prepared.content_for_terminal.contains("159→"));
    }

    #[test]
    fn read_file_lines_uses_shorter_terminal_preview_policy() {
        let history_file = std::env::temp_dir().join(format!(
            "ai-tool-preview-read-file-lines-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file);

        let mut content = String::new();
        for i in 0..90usize {
            content.push_str(&format!("{}→{}\n", i, "x".repeat(100)));
        }

        let prepared = prepare_tool_result(&app, "read_file_lines", &content);

        assert_eq!(prepared.content_for_model, content);
        assert!(prepared.content_for_terminal.contains("truncated for terminal preview"));
        assert!(prepared.content_for_terminal.contains("0→"));
        assert!(prepared.content_for_terminal.contains("89→"));
        assert!(!prepared.content_for_terminal.contains("39→"));
        assert!(prepared.content_for_terminal.len() < 3000);
    }

    #[test]
    fn web_search_uses_summary_first_terminal_preview() {
        let history_file =
            std::env::temp_dir().join(format!("ai-tool-preview-web-search-{}.sqlite", uuid::Uuid::new_v4()));
        let app = test_app(history_file);

        let mut content = String::new();
        for i in 0..40usize {
            content.push_str(&format!("result {}: title {}\n", i, "x".repeat(60)));
        }

        let prepared = prepare_tool_result(&app, "web_search", &content);

        assert_eq!(prepared.content_for_model, content);
        assert!(prepared.content_for_terminal.contains("summary-first terminal preview"));
        assert!(prepared.content_for_terminal.contains("result 0"));
        assert!(!prepared.content_for_terminal.contains("result 39"));
    }
}
