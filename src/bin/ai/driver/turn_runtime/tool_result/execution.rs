use crate::ai::{
    driver::tools::{self, ExecuteToolCallsResult},
    history::Message,
    mcp::McpClient,
    types::{App, ToolCall},
};
use serde_json::Value;
use std::collections::BTreeSet;

use super::{
    messaging::{
        append_cached_tool_results_note, append_code_inspection_working_memory,
        append_tool_result_messages, record_final_stream_response,
        record_persistent_code_discoveries,
    },
    overflow::{build_model_overflow_stub, summarize_large_tool_output, write_tool_overflow_file},
    preview::{build_terminal_preview, tail_chars},
};
use super::super::persistence::persist_pending_turn_messages;
use super::super::{
    MAX_TOOL_RESULT_INLINE_CHARS, TOOL_OVERFLOW_PREVIEW_CHARS,
    types::{IterationExecution, PreparedToolResult, TurnLoopStep},
};

const TOOL_USE_CORRECTION_PREFIX: &str = "Tool-use correction:";

fn turn_has_tool_use(turn_messages: &[Message]) -> bool {
    turn_messages.iter().any(|message| {
        message.role == "tool"
            || message
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty())
    })
}

fn count_tool_use_corrections(messages: &[Message]) -> usize {
    messages
        .iter()
        .filter(|message| message.role == "system")
        .filter_map(|message| message.content.as_str())
        .filter(|content| content.starts_with(TOOL_USE_CORRECTION_PREFIX))
        .count()
}

fn available_tool_names(app: &App) -> BTreeSet<String> {
    app.agent_context
        .as_ref()
        .map(|ctx| {
            ctx.tools
                .iter()
                .map(|tool| tool.function.name.clone())
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default()
}

fn looks_time_sensitive_request(question: &str) -> bool {
    let lower = question.to_ascii_lowercase();
    [
        "latest",
        "current",
        "today",
        "now",
        "weather",
        "news",
        "stock",
        "sports",
        "score",
        "recent",
        "release",
        "今天",
        "现在",
        "最新",
        "近期",
        "实时",
        "天气",
        "新闻",
        "股价",
        "比分",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn looks_command_request(question: &str) -> bool {
    let lower = question.to_ascii_lowercase();
    [
        "run ",
        "build",
        "test",
        "compile",
        "cargo ",
        "npm ",
        "pnpm ",
        "yarn ",
        "执行",
        "运行",
        "构建",
        "测试",
        "编译",
        "跑一下",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn looks_edit_request(question: &str) -> bool {
    let lower = question.to_ascii_lowercase();
    [
        "fix",
        "modify",
        "edit",
        "update",
        "refactor",
        "implement",
        "change",
        "修复",
        "修改",
        "改一下",
        "重构",
        "实现",
        "新增",
        "添加",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn looks_repo_or_code_request(question: &str) -> bool {
    let lower = question.to_ascii_lowercase();
    [
        ".rs",
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".java",
        ".go",
        "src/",
        "cargo.toml",
        "代码",
        "函数",
        "文件",
        "配置",
        "逻辑",
        "报错",
        "bug",
        "stack trace",
        "repo",
        "repository",
        "agent",
        "symbol",
        "class",
        "method",
        "why",
        "检查一下代码",
        "看看代码",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn tool_use_requirement_reason(question: &str, app: &App) -> Option<String> {
    let available = available_tool_names(app);
    if available.is_empty() {
        return None;
    }

    if looks_time_sensitive_request(question) && available.contains("web_search") {
        return Some(
            "This request is time-sensitive. Call `web_search` before giving a final answer."
                .to_string(),
        );
    }

    if looks_command_request(question) && available.contains("execute_command") {
        return Some(
            "This request asks you to run, build, test, or reproduce behavior. Call `execute_command` before giving a final answer."
                .to_string(),
        );
    }

    if looks_edit_request(question)
        && (available.contains("apply_patch")
            || available.contains("write_file")
            || available.contains("read_file")
            || available.contains("read_file_lines"))
    {
        return Some(
            "This request asks for code or file changes. Inspect the file with `read_file` / `read_file_lines`, then make the change with editing tools before giving a final answer."
                .to_string(),
        );
    }

    if looks_repo_or_code_request(question)
        && (available.contains("code_search")
            || available.contains("read_file")
            || available.contains("read_file_lines"))
    {
        return Some(
            "This request depends on repository code or file contents. Inspect the repo with `code_search` or file-read tools before giving a final answer."
                .to_string(),
        );
    }

    None
}

fn maybe_enqueue_tool_use_correction(
    app: &App,
    question: &str,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
) -> bool {
    if turn_has_tool_use(turn_messages) || count_tool_use_corrections(messages) >= 2 {
        return false;
    }
    let Some(reason) = tool_use_requirement_reason(question, app) else {
        return false;
    };
    let note = Message {
        role: "system".to_string(),
        content: Value::String(format!(
            "{TOOL_USE_CORRECTION_PREFIX} {reason}\nCall at least one relevant tool in your next response. Do not give a final answer yet."
        )),
        tool_calls: None,
        tool_call_id: None,
    };
    messages.push(note.clone());
    turn_messages.push(note);
    true
}

pub(in crate::ai::driver::turn_runtime) fn prepare_tool_result(
    app: &App,
    tool_name: &str,
    content: &str,
) -> PreparedToolResult {
    if content.chars().count() <= MAX_TOOL_RESULT_INLINE_CHARS {
        return PreparedToolResult {
            content_for_model: content.to_string(),
            content_for_terminal: build_terminal_preview(tool_name, content),
        };
    }

    let summary = summarize_large_tool_output(content);
    let path = write_tool_overflow_file(app, tool_name, &summary.body).ok();
    let content_for_model = build_model_overflow_stub(path.as_ref(), &summary);
    let content_for_terminal = if let Some(path) = path {
        format!(
            "{}\n[Saved full output to {}]\n",
            build_terminal_preview(
                tool_name,
                &tail_chars(&summary.body, TOOL_OVERFLOW_PREVIEW_CHARS)
            ),
            path.display()
        )
    } else {
        build_terminal_preview(tool_name, &tail_chars(&summary.body, TOOL_OVERFLOW_PREVIEW_CHARS))
    };

    PreparedToolResult {
        content_for_model,
        content_for_terminal,
    }
}

#[crate::ai::agent_hang_span(
    "pre-fix",
    "C",
    "turn_runtime::run_turn:execute_tool_calls",
    "[DEBUG] executing tool calls",
    "[DEBUG] executed tool calls",
    {
        "iteration": _iteration,
        "tool_calls": tool_calls
            .iter()
            .map(|tool| tool.function.name.clone())
            .collect::<Vec<_>>(),
    },
    {
        "iteration": _iteration,
        "tool_result_count": __agent_hang_result
            .as_ref()
            .map(|v| v.tool_results.len())
            .unwrap_or(0),
        "cached_hits": __agent_hang_result
            .as_ref()
            .map(|v| v.cached_hits.clone())
            .unwrap_or_default(),
        "ok": __agent_hang_result.is_ok(),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
fn execute_tool_calls_for_round(
    session_id: &str,
    mcp_client: &mut McpClient,
    tool_calls: &[ToolCall],
    _iteration: usize,
) -> Result<ExecuteToolCallsResult, Box<dyn std::error::Error>> {
    tools::execute_tool_calls(session_id, mcp_client, tool_calls)
}

fn handle_tool_call_round(
    app: &App,
    mcp_client: &mut McpClient,
    stream_result: &crate::ai::types::StreamResult,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    iteration: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let exec_result = execute_tool_calls_for_round(
        &app.session_id,
        mcp_client,
        &stream_result.tool_calls,
        iteration,
    )?;

    append_cached_tool_results_note(&exec_result, messages, turn_messages);
    append_tool_result_messages(
        app,
        &stream_result.assistant_text,
        &exec_result,
        messages,
        turn_messages,
    );
    append_code_inspection_working_memory(messages, turn_messages);
    record_persistent_code_discoveries(app, messages, turn_messages);

    persist_pending_turn_messages(
        app,
        one_shot_mode,
        turn_messages,
        persisted_turn_messages,
    );

    Ok(())
}

pub(in crate::ai::driver::turn_runtime) fn handle_iteration_execution(
    app: &App,
    question: &str,
    mcp_client: &mut McpClient,
    execution: IterationExecution,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    final_assistant_text: &mut String,
    final_assistant_recorded: &mut bool,
    force_final_response: &mut bool,
    iteration: usize,
    max_iterations: usize,
) -> Result<TurnLoopStep, Box<dyn std::error::Error>> {
    match execution {
        IterationExecution::Exit(outcome) => Ok(TurnLoopStep::Return(outcome)),
        IterationExecution::RequestFailed(text) => {
            *final_assistant_text = text;
            Ok(TurnLoopStep::Break)
        }
        IterationExecution::FinalResponse(stream_result) => {
            if maybe_enqueue_tool_use_correction(app, question, messages, turn_messages) {
                return Ok(TurnLoopStep::Continue);
            }
            record_final_stream_response(
                app,
                stream_result,
                messages,
                turn_messages,
                final_assistant_text,
                final_assistant_recorded,
            );
            Ok(TurnLoopStep::Break)
        }
        IterationExecution::ToolCall(stream_result) => {
            handle_tool_call_round(
                app,
                mcp_client,
                &stream_result,
                messages,
                turn_messages,
                one_shot_mode,
                persisted_turn_messages,
                iteration,
            )?;

            crate::ai::driver::input::clear_stdin_buffer();

            if iteration >= max_iterations {
                if *force_final_response {
                    *final_assistant_text = format!(
                        "Agent reached the tool iteration limit ({max_iterations}) without producing a final answer."
                    );
                    return Ok(TurnLoopStep::Break);
                }
                *force_final_response = true;
            }

            Ok(TurnLoopStep::Continue)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{
        cli::ParsedCli,
        types::{AgentContext, App, AppConfig, FunctionDefinition, ToolDefinition},
    };
    use rust_tools::commonw::FastMap;
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool};

    fn test_app_with_tools(tool_names: &[&str]) -> App {
        App {
            cli: ParsedCli::default(),
            config: AppConfig {
                api_key: String::new(),
                history_file: PathBuf::new(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 0,
                history_keep_last: 0,
                history_summary_max_chars: 0,
                intent_model: None,
                intent_model_path: PathBuf::new(),
            },
            session_id: "test".to_string(),
            session_history_file: PathBuf::new(),
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
                mcp_servers: FastMap::default(),
                max_iterations: 16,
            }),
            agent_reload_counter: None,
        }
    }

    #[test]
    fn tool_requirement_detects_time_sensitive_requests() {
        let app = test_app_with_tools(&["web_search"]);
        let reason = tool_use_requirement_reason("帮我查一下今天的天气", &app);
        assert!(reason.is_some());
        assert!(reason.unwrap().contains("web_search"));
    }

    #[test]
    fn tool_requirement_detects_code_requests() {
        let app = test_app_with_tools(&["code_search", "read_file"]);
        let reason = tool_use_requirement_reason("帮我看一下 a.rs 这个 agent 为什么会报错", &app);
        assert!(reason.is_some());
        assert!(reason.unwrap().contains("code_search"));
    }

    #[test]
    fn enqueue_tool_use_correction_only_happens_without_prior_tool_use() {
        let app = test_app_with_tools(&["code_search"]);
        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        assert!(maybe_enqueue_tool_use_correction(
            &app,
            "帮我看一下 a.rs 这个 agent",
            &mut messages,
            &mut turn_messages,
        ));
        assert_eq!(messages.len(), 1);
        assert!(messages[0]
            .content
            .as_str()
            .unwrap()
            .starts_with(TOOL_USE_CORRECTION_PREFIX));
    }
}
