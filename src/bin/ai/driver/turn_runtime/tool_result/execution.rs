use crate::ai::{
    driver::tools::{self, ExecuteToolCallsResult},
    history::Message,
    mcp::{McpClient, SharedMcpClient},
    types::{App, ToolCall},
};
use std::io::Write;

use super::{
    messaging::{
        append_cached_tool_results_note, append_code_inspection_working_memory,
        append_tool_result_messages, record_final_stream_response,
        record_persistent_code_discoveries,
    },
    overflow::{build_model_overflow_stub, summarize_large_tool_output, write_tool_overflow_file},
    preview::{build_terminal_preview, tail_chars},
};
use crate::ai::driver::print::{
    format_tool_output_prefix, print_tool_note_line, print_tool_output_block,
    sanitize_for_terminal,
};
use super::messaging::print_tool_result_preview;
use super::super::persistence::persist_pending_turn_messages;
use super::super::{
    MAX_TOOL_RESULT_INLINE_CHARS, TOOL_OVERFLOW_PREVIEW_CHARS,
    types::{IterationExecution, PreparedToolResult, TurnLoopStep},
};

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
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    tool_calls: &[ToolCall],
    observer: Option<&mut dyn tools::ToolExecutionObserver>,
    _iteration: usize,
) -> Result<ExecuteToolCallsResult, Box<dyn std::error::Error>> {
    tools::execute_tool_calls(session_id, mcp_client, shared_mcp_client, tool_calls, observer)
}

struct TerminalToolObserver<'a> {
    app: &'a App,
    active_stream_tool_call_id: Option<String>,
    pending_utf8: Vec<u8>,
    at_line_start: bool,
    streamed_any_output: bool,
}

impl<'a> TerminalToolObserver<'a> {
    fn new(app: &'a App) -> Self {
        Self {
            app,
            active_stream_tool_call_id: None,
            pending_utf8: Vec::new(),
            at_line_start: true,
            streamed_any_output: false,
        }
    }

    fn reset_stream_state(&mut self) {
        self.active_stream_tool_call_id = None;
        self.pending_utf8.clear();
        self.at_line_start = true;
        self.streamed_any_output = false;
    }

    fn push_stream_text(&mut self, text: &str) {
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        let sanitized = sanitize_for_terminal(&normalized);
        let mut rendered = String::new();
        for ch in sanitized.chars() {
            if self.at_line_start {
                rendered.push_str(&format_tool_output_prefix());
                self.at_line_start = false;
            }
            if ch == '\n' {
                rendered.push_str("\x1b[0m\n");
                self.at_line_start = true;
            } else {
                rendered.push(ch);
            }
        }
        if !rendered.is_empty() {
            print!("{rendered}");
            let _ = std::io::stdout().flush();
            self.streamed_any_output = true;
        }
    }

    fn flush_pending_utf8(&mut self) {
        if self.pending_utf8.is_empty() {
            return;
        }
        let text = String::from_utf8_lossy(&self.pending_utf8).into_owned();
        self.pending_utf8.clear();
        self.push_stream_text(&text);
    }

    fn finish_stream_output(&mut self, newline: bool) {
        self.flush_pending_utf8();
        if !self.at_line_start {
            if newline {
                print!("\x1b[0m\n");
                self.at_line_start = true;
            } else {
                print!("\x1b[0m");
            }
            let _ = std::io::stdout().flush();
        }
    }
}

impl tools::ToolExecutionObserver for TerminalToolObserver<'_> {
    fn on_tool_started(&mut self, tool_call: &ToolCall) {
        if tool_call.function.name != "execute_command" {
            return;
        }

        self.reset_stream_state();
        self.active_stream_tool_call_id = Some(tool_call.id.clone());
        print_tool_note_line("output", "streaming command output");
    }

    fn on_tool_stream(&mut self, tool_call: &ToolCall, chunk: &[u8]) {
        if self.active_stream_tool_call_id.as_deref() != Some(tool_call.id.as_str()) {
            return;
        }

        self.pending_utf8.extend_from_slice(chunk);
        loop {
            match std::str::from_utf8(&self.pending_utf8) {
                Ok(text) => {
                    let text = text.to_string();
                    self.pending_utf8.clear();
                    self.push_stream_text(&text);
                    break;
                }
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    if valid_up_to == 0 {
                        if err.error_len().is_some() {
                            self.flush_pending_utf8();
                        }
                        break;
                    }

                    let text = String::from_utf8_lossy(&self.pending_utf8[..valid_up_to]).into_owned();
                    self.pending_utf8.drain(..valid_up_to);
                    self.push_stream_text(&text);

                    if err.error_len().is_some() {
                        self.flush_pending_utf8();
                    }
                }
            }
        }
    }

    fn on_tool_finished(&mut self, tool_call: &ToolCall, run_result: &tools::RunOneResult) {
        if tool_call.function.name == "execute_command" {
            let is_failure = run_result.tool_result.content.starts_with("Exit code:");
            self.finish_stream_output(is_failure);

            let prepared = prepare_tool_result(
                self.app,
                &tool_call.function.name,
                &run_result.tool_result.content,
            );
            if !self.streamed_any_output {
                print_tool_note_line("result", "captured command output");
                print_tool_output_block(&prepared.content_for_terminal);
            } else if is_failure {
                if let Some(exit_line) = run_result.tool_result.content.lines().next() {
                    print_tool_note_line("error", exit_line);
                }
            } else {
                print_tool_note_line("result", "command completed");
            }

            self.reset_stream_state();
            return;
        }

        let prepared = prepare_tool_result(self.app, &tool_call.function.name, &run_result.tool_result.content);
        print_tool_result_preview(&tool_call.function.name, &prepared);
    }
}

fn handle_tool_call_round(
    app: &mut App,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    stream_result: &crate::ai::types::StreamResult,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    iteration: usize,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let mut observer = TerminalToolObserver::new(app);
    let exec_result = execute_tool_calls_for_round(
        &app.session_id,
        mcp_client,
        shared_mcp_client,
        &stream_result.tool_calls,
        Some(&mut observer),
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

    Ok(None)
}

pub(in crate::ai::driver::turn_runtime) fn handle_iteration_execution(
    app: &mut App,
    _question: &str,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    execution: IterationExecution,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    final_assistant_text: &mut String,
    final_assistant_recorded: &mut bool,
    force_final_response: &mut bool,
    terminal_dedupe_candidate: &mut Option<String>,
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
            *terminal_dedupe_candidate = handle_tool_call_round(
                app,
                mcp_client,
                shared_mcp_client,
                &stream_result,
                messages,
                turn_messages,
                one_shot_mode,
                persisted_turn_messages,
                iteration,
            )?;

            crate::ai::driver::input::clear_stdin_buffer();

            {
                let mut os = app.os.lock().unwrap();
                if os.consume_yield_requested() {
                    return Ok(TurnLoopStep::Return(
                        crate::ai::driver::turn_runtime::types::TurnOutcome::Continue,
                    ));
                }
            }

            if iteration >= max_iterations {
                if *force_final_response {
                    *final_assistant_text = format!(
                        "Agent reached the tool iteration limit ({max_iterations}) without producing a final answer."
                    );
                    return Ok(TurnLoopStep::Break);
                }
                *force_final_response = true;
            } else {
                // AIOS: kernel is the authoritative source for tool-call quota.
                // If kernel says we've exceeded rlimit.max_tool_calls even when
                // the stale user-space `iteration` counter hasn't tripped yet,
                // honor the kernel verdict.
                use aios_kernel::primitives::{RlimitDim, RlimitVerdict};
                let os = app.os.lock().unwrap();
                if let Some(pid) = os.current_process_id() {
                    if let RlimitVerdict::Exceeded { dimension, used, limit } =
                        os.rlimit_check(pid, &Default::default())
                    {
                        drop(os);
                        if matches!(dimension, RlimitDim::ToolCalls | RlimitDim::Turns)
                            && *force_final_response
                        {
                            *final_assistant_text = format!(
                                "Agent exceeded kernel rlimit ({:?}: used={} limit={}).",
                                dimension, used, limit
                            );
                            return Ok(TurnLoopStep::Break);
                        }
                        *force_final_response = true;
                    }
                }
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
        types::{
            AgentContext, App, AppConfig, FunctionCall, FunctionDefinition, ToolDefinition,
            ToolResult,
        },
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
                agent_route_model_path: PathBuf::new(),
                skill_match_model_path: PathBuf::new(),
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
            last_skill_bias: None,
            os: crate::ai::driver::new_local_kernel(),
            agent_reload_counter: None,
            observers: vec![Box::new(crate::ai::driver::thinking::ThinkingOrchestrator::new())],
        }
    }

    #[test]
    fn tool_call_round_no_longer_requests_terminal_dedupe() {
        let exec_result = ExecuteToolCallsResult {
            executed_tool_calls: vec![ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "execute_command".to_string(),
                    arguments: "{\"command\":\"seq 3\"}".to_string(),
                },
            }],
            tool_results: vec![ToolResult {
                tool_call_id: "call_1".to_string(),
                content: "1\n2\n3\n".to_string(),
            }],
            cached_hits: vec![false],
        };

        assert_eq!(exec_result.executed_tool_calls.len(), 1);
        assert_eq!(exec_result.tool_results.len(), 1);
    }
}
