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
    types::{IterationExecution, PreparedToolResult, ToolCallExecution, TurnLoopStep},
};
use super::messaging::print_tool_result_preview;
use super::{
    messaging::{
        append_cached_tool_results_note, append_message_pair, append_tool_result_messages,
        record_final_stream_response, record_tool_inspection_artifacts,
    },
    overflow::{build_model_overflow_stub, summarize_large_tool_output, write_tool_overflow_file},
    preview::{build_terminal_preview, tail_chars},
};
use crate::ai::driver::print::{
    format_tool_output_prefix, print_tool_note_line, print_tool_output_block, sanitize_for_terminal,
};
use crate::ai::theme::{ACCENT_MUTED, ACCENT_RULE, RESET};

/// 适合"中段按行裁剪"的非精确概览工具。
///
/// find_path / code_search / search_files / read_file(_lines) 的每一行都可能是
/// agent 后续判断需要引用的精确证据，不能做有损中段抽样；这些工具只允许在
/// 超过 inline 上限后 offload 到 session 文件，并在模型上下文里保留 path + stub。
fn supports_line_trim(tool_name: &str) -> bool {
    matches!(tool_name, "tree" | "ast_outline")
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
    allowed_tool_names: &rust_tools::commonw::FastSet<String>,
    observer: Option<&mut dyn tools::ToolExecutionObserver>,
    _iteration: usize,
) -> Result<ExecuteToolCallsResult, Box<dyn std::error::Error>> {
    tools::execute_tool_calls(
        session_id,
        mcp_client,
        shared_mcp_client,
        tool_calls,
        Some(allowed_tool_names),
        observer,
    )
}

/// 前台同步工具执行（尤其是 `execute_command` 的流式输出）也属于“当前 turn 的可中断
/// 输出阶段”。若这里不抬起 `app.streaming`，Ctrl+C 会被 SIGINT 处理器误判成
/// `Shutdown`，直接退出主进程，而不是取消当前工具轮次。
struct ToolExecutionStreamingGuard {
    flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl ToolExecutionStreamingGuard {
    fn new(flag: &std::sync::Arc<std::sync::atomic::AtomicBool>) -> Self {
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
        Self {
            flag: std::sync::Arc::clone(flag),
        }
    }
}

impl Drop for ToolExecutionStreamingGuard {
    fn drop(&mut self) {
        self.flag.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

struct TerminalToolObserver<'a> {
    app: &'a App,
    active_stream_tool_call_id: Option<String>,
    pending_utf8: Vec<u8>,
    at_line_start: bool,
    streamed_any_output: bool,
    // 流式输出折叠状态
    fold_total_lines: usize,
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
        }
    }

    fn reset_stream_state(&mut self) {
        self.active_stream_tool_call_id = None;
        self.pending_utf8.clear();
        self.at_line_start = true;
        self.streamed_any_output = false;
        self.fold_total_lines = 0;
    }

    fn push_stream_text(&mut self, text: &str) {
        use std::io::Write;
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        let sanitized = sanitize_for_terminal(&normalized);

        for ch in sanitized.chars() {
            if ch == '\n' {
                // 完成一行
                self.fold_total_lines += 1;

                if self.fold_total_lines <= TOOL_OUTPUT_FOLD_MAX_VISIBLE {
                    // 还没超限，正常输出换行
                    print!("\x1b[0m\n");
                    self.at_line_start = true;
                } else if self.fold_total_lines == TOOL_OUTPUT_FOLD_MAX_VISIBLE + 1 {
                    // 刚超限：结束当前行，切换到单行计数器模式
                    print!("\x1b[0m\n");
                    self.at_line_start = true;
                    // 打印计数器行（后续会用 \r\x1b[2K 原地更新）
                    let folded = self.fold_total_lines - TOOL_OUTPUT_FOLD_MAX_VISIBLE;
                    print!(
                        "  {ACCENT_RULE}│{RESET} {ACCENT_MUTED}··· {folded} lines folded (streaming) ···\x1b[0m"
                    );
                    let _ = std::io::stdout().flush();
                } else {
                    // 已在计数器模式：原地更新计数
                    let folded = self.fold_total_lines - TOOL_OUTPUT_FOLD_MAX_VISIBLE;
                    print!(
                        "\r\x1b[2K  {ACCENT_RULE}│{RESET} {ACCENT_MUTED}··· {folded} lines folded (streaming) ···\x1b[0m"
                    );
                    let _ = std::io::stdout().flush();
                    self.at_line_start = false;
                }
            } else {
                // 只在未超限时打印字符
                if self.fold_total_lines < TOOL_OUTPUT_FOLD_MAX_VISIBLE {
                    if self.at_line_start {
                        print!("{}", format_tool_output_prefix());
                        self.at_line_start = false;
                    }
                    print!("{ch}");
                }
                // 超限后不输出任何字符内容
            }
        }
        if !sanitized.is_empty() {
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
        // 如果正在计数器模式，换行结束计数器行
        if self.fold_total_lines > TOOL_OUTPUT_FOLD_MAX_VISIBLE {
            // 把 "(streaming)" 替换为最终状态
            let folded = self.fold_total_lines - TOOL_OUTPUT_FOLD_MAX_VISIBLE;
            print!(
                "\r\x1b[2K  {ACCENT_RULE}│{RESET} {ACCENT_MUTED}··· {folded} lines folded ···\x1b[0m\n"
            );
            let _ = std::io::stdout().flush();
            self.at_line_start = true;
        } else if !self.at_line_start {
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
    tool_call_execution: &ToolCallExecution,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    iteration: usize,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let mut observer = TerminalToolObserver::new(app);
    let _streaming_guard = ToolExecutionStreamingGuard::new(&app.streaming);
    let exec_result = execute_tool_calls_for_round(
        &app.session_id,
        mcp_client,
        shared_mcp_client,
        &tool_call_execution.stream_result.tool_calls,
        &tool_call_execution.allowed_tool_names,
        Some(&mut observer),
        iteration,
    )?;
    append_cached_tool_results_note(&exec_result, messages, turn_messages);
    append_tool_result_messages(
        app,
        &tool_call_execution.stream_result.assistant_text,
        &tool_call_execution.stream_result.reasoning_text,
        &exec_result,
        messages,
        turn_messages,
    );
    record_tool_inspection_artifacts(app, messages, turn_messages);

    persist_pending_turn_messages(app, one_shot_mode, turn_messages, persisted_turn_messages);

    Ok(None)
}

const DISCOVER_SKILLS_FOLLOWUP_NOTE: &str = "tool_followup:discover_skills\n\
`discover_skills` only listed metadata and did not activate any skill.\n\
This is not a final answer. Continue the current turn:\n\
- if one listed skill clearly matches the user's task, call `activate_skill` with its name to load its prompt and tools;\n\
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
            let reasoning_only_completion = stream_result.assistant_text.trim().is_empty()
                && !stream_result.reasoning_text.trim().is_empty()
                && stream_result.tool_calls.is_empty();
            if reasoning_only_completion {
                if *force_final_response {
                    *final_assistant_text =
                        "[模型只返回了思考内容，没有给出最终回答，请重试或切换模型]".to_string();
                    return Ok(TurnLoopStep::Break);
                }
                *force_final_response = true;
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
        IterationExecution::ToolCall(tool_call_execution) => {
            let discover_skills_only = no_active_skill
                && requested_only_discover_skills(&tool_call_execution.stream_result.tool_calls);
            let image_read_paths = extract_image_paths_from_file_read_tool_calls(
                &tool_call_execution.stream_result.tool_calls,
            );
            *terminal_dedupe_candidate = handle_tool_call_round(
                app,
                mcp_client,
                shared_mcp_client,
                &tool_call_execution,
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
                // 当前 usage 已经超限、或下一次 tool call 会超限，都应该切到
                // force-final，但 tool-call 配额本身不该阻断“无工具的最终回答”。
                use aios_kernel::primitives::{ResourceUsageDelta, RlimitDim, RlimitVerdict};
                let os = app.os.lock().unwrap();
                if let Some(pid) = os.current_process_id() {
                    let current_verdict = os.rlimit_check(pid, &Default::default());
                    let next_tool_verdict = os.rlimit_check(
                        pid,
                        &ResourceUsageDelta {
                            tool_calls: 1,
                            ..Default::default()
                        },
                    );
                    drop(os);
                    if let RlimitVerdict::Exceeded {
                        dimension,
                        used,
                        limit,
                    } = current_verdict
                    {
                        match dimension {
                            RlimitDim::Turns => {
                                if *force_final_response {
                                    *final_assistant_text = format!(
                                        "Agent exceeded kernel rlimit ({:?}: used={} limit={}).",
                                        dimension, used, limit
                                    );
                                    return Ok(TurnLoopStep::Break);
                                }
                                *force_final_response = true;
                            }
                            RlimitDim::ToolCalls => {
                                *force_final_response = true;
                            }
                            _ => {}
                        }
                    }
                    if matches!(
                        next_tool_verdict,
                        RlimitVerdict::Exceeded {
                            dimension: RlimitDim::ToolCalls,
                            ..
                        }
                    ) {
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
        driver::signal,
        types::{
            AgentContext, App, AppConfig, FunctionCall, FunctionDefinition, ToolDefinition,
            ToolResult,
        },
    };
    use aios_kernel::primitives::ResourceLimit;
    use rust_tools::cw::SkipMap;
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool};
    use std::time::{Duration, Instant};

    fn test_app_with_tools(tool_names: &[&str]) -> App {
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
                intent_model_path: PathBuf::new(),
                agent_route_model_path: PathBuf::new(),
                skill_match_model_path: PathBuf::new(),
            },
            session_id: "test".to_string(),
            session_history_file: PathBuf::new(),
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::builder().build().unwrap(),
            current_model: String::new(),
            current_agent: "build".to_string(),
            current_agent_manifest: None,
            pending_files: None,
            forced_skill: None,
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
    fn reasoning_only_final_response_retries_once_with_forced_final() {
        let mut app = test_app_with_tools(&["read_file"]);
        let mcp = crate::ai::mcp::McpClient::new();
        let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(mcp));
        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        let mut persisted_turn_messages = 0usize;
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut force_final_response = false;
        let mut terminal_dedupe_candidate = None;

        let step = handle_iteration_execution(
            &mut app,
            "compare two yaml files",
            &shared_mcp.lock().unwrap(),
            &shared_mcp,
            IterationExecution::FinalResponse(crate::ai::types::StreamResult {
                outcome: crate::ai::types::StreamOutcome::Completed,
                tool_calls: Vec::new(),
                assistant_text: String::new(),
                hidden_meta: String::new(),
                reasoning_text: "I should read both files first.".to_string(),
                skip_response_drain: true,
            }),
            &mut messages,
            &mut turn_messages,
            false,
            &mut persisted_turn_messages,
            &mut final_assistant_text,
            &mut final_assistant_recorded,
            &mut force_final_response,
            &mut terminal_dedupe_candidate,
            true,
            1,
            16,
        )
        .unwrap();

        assert!(matches!(step, TurnLoopStep::Continue));
        assert!(force_final_response);
        assert!(final_assistant_text.is_empty());
        assert!(!final_assistant_recorded);
        assert!(messages.is_empty());
        assert!(turn_messages.is_empty());
    }

    #[test]
    fn reasoning_only_final_response_stops_after_forced_retry() {
        let mut app = test_app_with_tools(&["read_file"]);
        let mcp = crate::ai::mcp::McpClient::new();
        let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(mcp));
        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        let mut persisted_turn_messages = 0usize;
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut force_final_response = true;
        let mut terminal_dedupe_candidate = None;

        let step = handle_iteration_execution(
            &mut app,
            "compare two yaml files",
            &shared_mcp.lock().unwrap(),
            &shared_mcp,
            IterationExecution::FinalResponse(crate::ai::types::StreamResult {
                outcome: crate::ai::types::StreamOutcome::Completed,
                tool_calls: Vec::new(),
                assistant_text: String::new(),
                hidden_meta: String::new(),
                reasoning_text: "I should read both files first.".to_string(),
                skip_response_drain: true,
            }),
            &mut messages,
            &mut turn_messages,
            false,
            &mut persisted_turn_messages,
            &mut final_assistant_text,
            &mut final_assistant_recorded,
            &mut force_final_response,
            &mut terminal_dedupe_candidate,
            true,
            2,
            16,
        )
        .unwrap();

        assert!(matches!(step, TurnLoopStep::Break));
        assert_eq!(
            final_assistant_text,
            "[模型只返回了思考内容，没有给出最终回答，请重试或切换模型]"
        );
        assert!(!final_assistant_recorded);
        assert!(messages.is_empty());
        assert!(turn_messages.is_empty());
    }

    #[test]
    fn forced_final_hallucinated_tool_call_is_rejected_without_consuming_quota() {
        let _env_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        let mut app = test_app_with_tools(&["read_file"]);
        let pid = {
            let mut os = app.os.lock().unwrap();
            let pid =
                os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
            let mut lim = ResourceLimit::unlimited();
            lim.max_tool_calls = 64;
            os.rlimit_set(pid, lim).unwrap();
            pid
        };
        crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());

        let path = std::env::temp_dir().join(format!("forced-final-{}.txt", pid));
        std::fs::write(&path, "hello").unwrap();

        let shared_mcp =
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        let mut persisted_turn_messages = 0usize;
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut force_final_response = true;
        let mut terminal_dedupe_candidate = None;

        let step = handle_iteration_execution(
            &mut app,
            "summarize findings",
            &shared_mcp.lock().unwrap(),
            &shared_mcp,
            IterationExecution::ToolCall(ToolCallExecution {
                stream_result: crate::ai::types::StreamResult {
                    outcome: crate::ai::types::StreamOutcome::ToolCall,
                    tool_calls: vec![ToolCall {
                        id: "call_1".to_string(),
                        tool_type: "function".to_string(),
                        function: FunctionCall {
                            name: "read_file".to_string(),
                            arguments: format!(r#"{{"file_path":"{}"}}"#, path.to_string_lossy()),
                        },
                    }],
                    assistant_text: String::new(),
                    hidden_meta: String::new(),
                    reasoning_text: String::new(),
                    skip_response_drain: true,
                },
                allowed_tool_names: Default::default(),
            }),
            &mut messages,
            &mut turn_messages,
            false,
            &mut persisted_turn_messages,
            &mut final_assistant_text,
            &mut final_assistant_recorded,
            &mut force_final_response,
            &mut terminal_dedupe_candidate,
            true,
            3,
            16,
        )
        .unwrap();

        assert!(matches!(step, TurnLoopStep::Continue));
        assert!(force_final_response);
        assert!(final_assistant_text.is_empty());
        assert!(!final_assistant_recorded);
        {
            let os = app.os.lock().unwrap();
            assert_eq!(os.rusage_get(pid).unwrap().tool_calls, 0);
        }
        let joined = turn_messages
            .iter()
            .map(|msg| msg.content.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("not available in this turn's tool schema"));
        assert!(!joined.contains("exceeded kernel rlimit"));

        let _ = std::fs::remove_file(&path);
        if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
            *guard = None;
        }
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

    #[test]
    fn ctrl_c_during_foreground_tool_round_cancels_without_shutdown() {
        let _env_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        signal::clear_request_interrupt();

        let app = test_app_with_tools(&["execute_command"]);
        {
            let mut os = app.os.lock().unwrap();
            let _ = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        }
        crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());

        let streaming = app.streaming.clone();
        let shutdown = app.shutdown.clone();
        let cancel_stream = app.cancel_stream.clone();

        let handle = std::thread::spawn(move || {
            let mut app = app;
            let mcp = crate::ai::mcp::McpClient::new();
            let shared_mcp =
                std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
            let mut messages = Vec::new();
            let mut turn_messages = Vec::new();
            let mut persisted_turn_messages = 0usize;
            let start = Instant::now();
            let result = handle_tool_call_round(
                &mut app,
                &mcp,
                &shared_mcp,
                &ToolCallExecution {
                    stream_result: crate::ai::types::StreamResult {
                        outcome: crate::ai::types::StreamOutcome::ToolCall,
                        tool_calls: vec![ToolCall {
                            id: "call_1".to_string(),
                            tool_type: "function".to_string(),
                            function: FunctionCall {
                                name: "execute_command".to_string(),
                                arguments: r#"{"command":"sleep 2"}"#.to_string(),
                            },
                        }],
                        assistant_text: String::new(),
                        hidden_meta: String::new(),
                        reasoning_text: String::new(),
                        skip_response_drain: true,
                    },
                    allowed_tool_names: ["execute_command".to_string()].into_iter().collect(),
                },
                &mut messages,
                &mut turn_messages,
                true,
                &mut persisted_turn_messages,
                1,
            );
            (
                result.map(|_| ()).map_err(|err| err.to_string()),
                start.elapsed(),
                app,
            )
        });

        let wait_started = Instant::now();
        while !streaming.load(std::sync::atomic::Ordering::Relaxed)
            && wait_started.elapsed() < Duration::from_secs(1)
        {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            streaming.load(std::sync::atomic::Ordering::Relaxed),
            "foreground tool round never raised streaming flag"
        );

        signal::handle_sigint(
            shutdown.as_ref(),
            streaming.as_ref(),
            cancel_stream.as_ref(),
        );

        let (result, elapsed, returned_app) = handle.join().unwrap();

        returned_app
            .cancel_stream
            .store(false, std::sync::atomic::Ordering::Relaxed);
        crate::ai::tools::registry::common::clear_tool_cancel();
        signal::clear_request_interrupt();
        if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
            *guard = None;
        }

        assert!(result.is_ok());
        assert!(
            elapsed < Duration::from_secs(1),
            "tool round did not stop promptly after Ctrl+C: {elapsed:?}"
        );
        assert!(
            !shutdown.load(std::sync::atomic::Ordering::Relaxed),
            "Ctrl+C during foreground tool round should not request shutdown"
        );
    }
}
