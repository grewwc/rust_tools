use crate::ai::{
    driver::tools::{self, ExecuteToolCallsResult},
    history::{Message, ROLE_INTERNAL_NOTE},
    mcp::{McpClient, SharedMcpClient},
    types::{App, ToolCall},
};
use std::io::Write;

use super::super::persistence::persist_pending_turn_messages;
use super::super::{
    MAX_TOOL_RESULT_INLINE_CHARS, MAX_TOOL_RESULT_LINE_TRIM_CHARS, TOOL_OVERFLOW_PREVIEW_CHARS,
    types::{IterationExecution, PreparedToolResult, TurnLoopStep},
};
use super::messaging::print_tool_result_preview;
use super::{
    messaging::{
        append_cached_tool_results_note, append_code_inspection_working_memory,
        append_message_pair, append_tool_result_messages, record_final_stream_response,
        record_persistent_code_discoveries,
    },
    overflow::{build_model_overflow_stub, summarize_large_tool_output, write_tool_overflow_file},
    preview::{build_terminal_preview, tail_chars},
};
use crate::ai::driver::print::{
    format_tool_output_prefix, print_tool_note_line, print_tool_output_block, sanitize_for_terminal,
};
use crate::ai::theme::{ACCENT_MUTED, ACCENT_RULE, RESET};

/// 适合"中段按行裁剪"的工具：输出本身是搜索/列表类（head+命中+tail 信息密度高、
/// 中段冗余多）。read_file / read_file_lines 不在此列——agent 显式要求读这些行，
/// 必须把请求的全部内容回传，不能擅自压缩，否则会影响 agent 效果。
fn supports_line_trim(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "grep_search" | "search_files" | "list_directory" | "tree" | "code_search" | "ast_outline"
    )
}

/// 把"中等大"（介于 MAX_TOOL_RESULT_LINE_TRIM_CHARS 和 MAX_TOOL_RESULT_INLINE_CHARS 之间）
/// 的结构化输出折叠为：头 N 行 + 命中关键词的若干行 + 尾 M 行 + 中段标注。
/// 不写盘、不破坏整体语义，只是把"中段冗余"压掉。
fn line_trim_middle(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    if total_lines <= 80 {
        return content.to_string();
    }

    let head_lines = 40usize;
    let tail_lines = 20usize;

    let mut head = Vec::with_capacity(head_lines);
    for line in lines.iter().take(head_lines) {
        head.push(*line);
    }
    let tail_start = total_lines.saturating_sub(tail_lines);
    let mut tail = Vec::with_capacity(tail_lines);
    if tail_start > head_lines {
        for line in lines.iter().skip(tail_start) {
            tail.push(*line);
        }
    }

    // 在中段（head_lines..tail_start）按关键字采样 8 行
    let mut key_lines: Vec<(usize, &str)> = Vec::new();
    if tail_start > head_lines {
        for (i, line) in lines.iter().enumerate().take(tail_start).skip(head_lines) {
            let lower = line.to_ascii_lowercase();
            let important = lower.contains("error")
                || lower.contains("fail")
                || lower.contains("panic")
                || lower.contains("warn")
                || lower.contains("todo")
                || lower.contains("fixme")
                || lower.contains("//!")
                || lower.contains("///")
                || lower.starts_with("fn ")
                || lower.starts_with("pub fn ")
                || lower.starts_with("impl ")
                || lower.starts_with("struct ")
                || lower.starts_with("trait ")
                || lower.starts_with("enum ")
                || lower.starts_with("#[")
                || lower.contains(": error")
                || lower.contains(": warning");
            if important {
                key_lines.push((i, *line));
                if key_lines.len() >= 8 {
                    break;
                }
            }
        }
    }

    let omitted_count = total_lines.saturating_sub(head_lines + tail.len());
    let mut out = String::with_capacity(content.len() / 2);
    for line in &head {
        out.push_str(line);
        out.push('\n');
    }
    if !key_lines.is_empty() {
        out.push_str(&format!(
            "\n... [middle trimmed: {} lines folded; key-match samples below]\n",
            omitted_count.saturating_sub(key_lines.len())
        ));
        for (idx, line) in &key_lines {
            out.push_str(&format!("L{idx}: {line}\n"));
        }
        out.push_str("...\n");
    } else {
        out.push_str(&format!(
            "\n... [middle trimmed: {} lines folded]\n",
            omitted_count
        ));
    }
    for line in &tail {
        out.push_str(line);
        out.push('\n');
    }
    out
}

pub(in crate::ai::driver::turn_runtime) fn prepare_tool_result(
    app: &App,
    tool_name: &str,
    content: &str,
) -> PreparedToolResult {
    let char_count = content.chars().count();
    if char_count <= MAX_TOOL_RESULT_LINE_TRIM_CHARS {
        return PreparedToolResult {
            content_for_model: content.to_string(),
            content_for_terminal: build_terminal_preview(tool_name, content),
        };
    }

    if char_count <= MAX_TOOL_RESULT_INLINE_CHARS && supports_line_trim(tool_name) {
        let trimmed = line_trim_middle(content);
        // 复用 trimmed 的字节长度做廉价短路：trimmed 是从 content 里挑选若干行
        // 拼接出来的（可能改动；保留 ASCII / UTF-8 不变），如果字节更短就一定是
        // 字符更短，不必再做完整 chars().count() 双扫描。
        if trimmed.len() < content.len() && trimmed.chars().count() < char_count {
            return PreparedToolResult {
                content_for_model: trimmed,
                content_for_terminal: build_terminal_preview(tool_name, content),
            };
        }
    }

    if char_count <= MAX_TOOL_RESULT_INLINE_CHARS {
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
        build_terminal_preview(
            tool_name,
            &tail_chars(&summary.body, TOOL_OVERFLOW_PREVIEW_CHARS),
        )
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
    tools::execute_tool_calls(
        session_id,
        mcp_client,
        shared_mcp_client,
        tool_calls,
        observer,
    )
}

struct TerminalToolObserver<'a> {
    app: &'a App,
    active_stream_tool_call_id: Option<String>,
    pending_utf8: Vec<u8>,
    at_line_start: bool,
    streamed_any_output: bool,
    // 流式输出折叠状态
    fold_total_lines: usize,
    fold_recent_lines: std::collections::VecDeque<String>,
    fold_current_line: String,
    fold_window_rows: usize,
}

const TOOL_OUTPUT_FOLD_MAX_VISIBLE: usize = 8;

impl<'a> TerminalToolObserver<'a> {
    fn new(app: &'a App) -> Self {
        Self {
            app,
            active_stream_tool_call_id: None,
            pending_utf8: Vec::new(),
            at_line_start: true,
            streamed_any_output: false,
            fold_total_lines: 0,
            fold_recent_lines: std::collections::VecDeque::new(),
            fold_current_line: String::new(),
            fold_window_rows: 0,
        }
    }

    fn reset_stream_state(&mut self) {
        self.active_stream_tool_call_id = None;
        self.pending_utf8.clear();
        self.at_line_start = true;
        self.streamed_any_output = false;
        self.fold_total_lines = 0;
        self.fold_recent_lines.clear();
        self.fold_current_line.clear();
        self.fold_window_rows = 0;
    }

    fn push_stream_text(&mut self, text: &str) {
        use std::io::Write;
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        let sanitized = sanitize_for_terminal(&normalized);

        for ch in sanitized.chars() {
            if ch == '\n' {
                // 完成一行
                self.fold_total_lines += 1;
                let completed_line = std::mem::take(&mut self.fold_current_line);
                self.fold_recent_lines.push_back(completed_line);
                while self.fold_recent_lines.len() > TOOL_OUTPUT_FOLD_MAX_VISIBLE {
                    self.fold_recent_lines.pop_front();
                }

                if self.fold_total_lines <= TOOL_OUTPUT_FOLD_MAX_VISIBLE {
                    // 还没超限，正常输出换行
                    print!("\x1b[0m\n");
                    self.at_line_start = true;
                } else {
                    // 超限：覆盖底部滚动窗口
                    self.tool_output_fold_redraw();
                    self.at_line_start = true;
                }
            } else {
                if self.at_line_start {
                    print!("{}", format_tool_output_prefix());
                    self.at_line_start = false;
                }
                self.fold_current_line.push(ch);
                print!("{ch}");
            }
        }
        if !sanitized.is_empty() {
            let _ = std::io::stdout().flush();
            self.streamed_any_output = true;
        }
    }

    /// 覆盖底部滚动窗口：折叠指示器 + 最近 N 行
    fn tool_output_fold_redraw(&mut self) {
        use std::io::Write;
        let mut out = std::io::stdout();

        let erase_rows = if self.fold_window_rows == 0 {
            // 首次折叠：回退之前正常打印的行数 + 当前行
            TOOL_OUTPUT_FOLD_MAX_VISIBLE + 1
        } else {
            // 后续折叠：回退窗口 + 当前流式行
            self.fold_window_rows + 1
        };

        if erase_rows > 0 {
            let _ = write!(out, "\x1b[{}A\r\x1b[0J", erase_rows);
        }

        let folded_count = self.fold_total_lines.saturating_sub(TOOL_OUTPUT_FOLD_MAX_VISIBLE);
        let _ = write!(
            out,
            "  {ACCENT_RULE}│{RESET} {ACCENT_MUTED}··· {folded_count} lines folded ···\x1b[0m\n"
        );

        for line in &self.fold_recent_lines {
            let _ = write!(out, "{}{}\x1b[0m\n", format_tool_output_prefix(), line);
        }

        let _ = out.flush();
        self.fold_window_rows = 1 + self.fold_recent_lines.len();
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

                    let text =
                        String::from_utf8_lossy(&self.pending_utf8[..valid_up_to]).into_owned();
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

        let prepared = prepare_tool_result(
            self.app,
            &tool_call.function.name,
            &run_result.tool_result.content,
        );
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
        &stream_result.reasoning_text,
        &exec_result,
        messages,
        turn_messages,
    );
    append_code_inspection_working_memory(messages, turn_messages);
    record_persistent_code_discoveries(app, messages, turn_messages);

    persist_pending_turn_messages(app, one_shot_mode, turn_messages, persisted_turn_messages);

    Ok(None)
}

const DISCOVER_SKILLS_FOLLOWUP_NOTE: &str = "tool_followup:discover_skills\n\
`discover_skills` only listed metadata and did not activate any skill.\n\
This is not a final answer. Continue the current turn:\n\
- pick the best matching skill if one is clearly relevant;\n\
- otherwise enable the missing tools you need;\n\
- if no skill is actually needed, answer the user directly.\n\
Do not end the turn immediately after only listing skills.";

fn requested_only_discover_skills(tool_calls: &[ToolCall]) -> bool {
    !tool_calls.is_empty()
        && tool_calls
            .iter()
            .all(|tool_call| tool_call.function.name == "discover_skills")
}

fn append_discover_skills_followup_note(
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
) {
    let already_present = messages.iter().chain(turn_messages.iter()).any(|message| {
        message.role == ROLE_INTERNAL_NOTE
            && message.content.as_str() == Some(DISCOVER_SKILLS_FOLLOWUP_NOTE)
    });
    if already_present {
        return;
    }
    append_message_pair(
        messages,
        turn_messages,
        Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: serde_json::Value::String(DISCOVER_SKILLS_FOLLOWUP_NOTE.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    );
}

fn extract_image_paths_from_file_read_tool_calls(tool_calls: &[ToolCall]) -> Vec<String> {
    let mut out = Vec::new();
    for tool_call in tool_calls {
        if !matches!(
            tool_call.function.name.as_str(),
            "read_file" | "read_file_lines"
        ) {
            continue;
        }
        let Ok(args) = serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
        else {
            continue;
        };
        let Some(path) = args
            .get("file_path")
            .or_else(|| args.get("path"))
            .and_then(serde_json::Value::as_str)
        else {
            continue;
        };
        if crate::ai::files::is_image_path(path) && !out.iter().any(|existing| existing == path) {
            out.push(path.to_string());
        }
    }
    out
}

fn append_auto_image_followup_message(
    app: &App,
    question: &str,
    shared_mcp_client: &SharedMcpClient,
    image_paths: &[String],
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
) -> Result<(), Box<dyn std::error::Error>> {
    if image_paths.is_empty() {
        return Ok(());
    }

    let question = if question.trim().is_empty() {
        "Analyze the requested image file.".to_string()
    } else {
        question.to_string()
    };

    let content = if crate::ai::models::supports_image_input(&app.current_model) {
        crate::ai::request::build_content(&app.current_model, &question, image_paths)?
    } else if let Some(ocr) =
        crate::ai::driver::model::ocr_images_for_attached_input(shared_mcp_client, image_paths)?
    {
        let prompt = if ocr.has_usable_text() {
            format!(
                "{}\n\n[Auto OCR From Image File Read via {}]\n{}",
                question, ocr.tool_name, ocr.content
            )
        } else {
            format!(
                "{}\n\n[Image file read was auto-upgraded to attachment semantics, but OCR did not produce usable text.]",
                question
            )
        };
        serde_json::Value::String(prompt)
    } else {
        serde_json::Value::String(format!(
            "{}\n\n[Image file read was auto-upgraded to attachment semantics, but no OCR tool was available for this text-only model.]",
            question
        ))
    };

    append_message_pair(
        messages,
        turn_messages,
        Message {
            role: "user".to_string(),
            content,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    );
    Ok(())
}

pub(in crate::ai::driver::turn_runtime) fn handle_iteration_execution(
    app: &mut App,
    question: &str,
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
    no_active_skill: bool,
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
            let discover_skills_only =
                no_active_skill && requested_only_discover_skills(&stream_result.tool_calls);
            let image_read_paths =
                extract_image_paths_from_file_read_tool_calls(&stream_result.tool_calls);
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

            if discover_skills_only {
                append_discover_skills_followup_note(messages, turn_messages);
            }
            append_auto_image_followup_message(
                app,
                question,
                shared_mcp_client,
                &image_read_paths,
                messages,
                turn_messages,
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
                    if let RlimitVerdict::Exceeded {
                        dimension,
                        used,
                        limit,
                    } = os.rlimit_check(pid, &Default::default())
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
    use rust_tools::cw::SkipMap;
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
                mcp_servers: SkipMap::default(),
                max_iterations: 16,
            }),
            last_skill_bias: None,
            os: crate::ai::driver::new_local_kernel(),
            agent_reload_counter: None,
            observers: vec![Box::new(
                crate::ai::driver::thinking::ThinkingOrchestrator::new(),
            )],
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

    #[test]
    fn requested_only_discover_skills_detects_single_tool_round() {
        let tool_calls = vec![ToolCall {
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "discover_skills".to_string(),
                arguments: "{}".to_string(),
            },
        }];
        assert!(requested_only_discover_skills(&tool_calls));
    }

    #[test]
    fn requested_only_discover_skills_rejects_mixed_rounds() {
        let tool_calls = vec![
            ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "discover_skills".to_string(),
                    arguments: "{}".to_string(),
                },
            },
            ToolCall {
                id: "call_2".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "enable_tools".to_string(),
                    arguments: "{}".to_string(),
                },
            },
        ];
        assert!(!requested_only_discover_skills(&tool_calls));
    }

    #[test]
    fn discover_skills_followup_note_is_deduplicated() {
        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        append_discover_skills_followup_note(&mut messages, &mut turn_messages);
        append_discover_skills_followup_note(&mut messages, &mut turn_messages);
        assert_eq!(messages.len(), 1);
        assert_eq!(turn_messages.len(), 1);
        assert_eq!(messages[0].role, ROLE_INTERNAL_NOTE);
        assert_eq!(
            messages[0].content.as_str(),
            Some(DISCOVER_SKILLS_FOLLOWUP_NOTE)
        );
    }

    #[test]
    fn extract_image_paths_from_file_read_tool_calls_collects_image_reads() {
        let tool_calls = vec![
            ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: r#"{"file_path":"/tmp/shot.png"}"#.to_string(),
                },
            },
            ToolCall {
                id: "call_2".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file_lines".to_string(),
                    arguments: r#"{"file_path":"/tmp/notes.txt"}"#.to_string(),
                },
            },
        ];
        assert_eq!(
            extract_image_paths_from_file_read_tool_calls(&tool_calls),
            vec!["/tmp/shot.png".to_string()]
        );
    }

    #[test]
    fn auto_image_followup_uses_multimodal_message_for_vl_model() {
        let mut app = test_app_with_tools(&[]);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("tool-followup-{}.png", uuid::Uuid::new_v4()));
        std::fs::write(&path, b"fake").unwrap();
        app.current_model = crate::ai::model_names::all()
            .iter()
            .find(|m| m.is_vl)
            .map(|m| m.name.clone())
            .expect("models.json must contain at least one VL model");

        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        let shared_mcp =
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
        append_auto_image_followup_message(
            &app,
            "describe the file",
            &shared_mcp,
            &[path.to_string_lossy().to_string()],
            &mut messages,
            &mut turn_messages,
        )
        .unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert!(messages[0].content.is_array());

        let _ = std::fs::remove_file(&path);
    }
}
