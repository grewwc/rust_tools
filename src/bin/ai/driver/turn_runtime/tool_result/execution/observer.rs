//! Terminal observer and execution adapter: streams tool output to
//! the terminal and adapts a tool-call round to the `ToolExecutor` port.

use super::*;

/// Foreground synchronous tool execution (especially `execute_command`'s streamed output)
/// is also part of the “interruptible output phase of the current turn”. Without raising
/// `app.streaming` here, Ctrl+C would be misjudged by the SIGINT handler as `Shutdown`,
/// exiting the main process instead of cancelling the current tool round.
pub(in crate::ai::driver::turn_runtime) struct ToolExecutionStreamingGuard {
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

pub(in crate::ai::driver::turn_runtime) struct TerminalToolObserver<'a> {
    app: &'a App,
    active_stream_tool_call_id: Option<String>,
    pending_utf8: Vec<u8>,
    render_full_pty_stream: bool,
    visual_output_probe: String,
    visual_output_line: String,
    visual_output_detected: bool,
    at_line_start: bool,
    streamed_any_output: bool,
    // Streamed-output folding state
    allow_inline_fold_updates: bool,
    fold_total_lines: usize,
    tty_fold: TtyToolOutputFoldState,
}

// A typical terminal QR code is about 30–50 lines; keeping 64 lines shows one-shot
// visual output such as QR-login in full while still bounding unbounded streamed output
// such as build logs.
impl<'a> TerminalToolObserver<'a> {
    fn new(app: &'a App) -> Self {
        Self {
            app,
            active_stream_tool_call_id: None,
            pending_utf8: Vec::new(),
            render_full_pty_stream: false,
            visual_output_probe: String::new(),
            visual_output_line: String::new(),
            visual_output_detected: false,
            at_line_start: true,
            streamed_any_output: false,
            fold_total_lines: 0,
            // In-place refresh sequences like `\r` / `CSI 2K` only suit a real TTY. IDE Chat /
            // pipe / log-capture environments do not interpret ANSI cursor control, so passing
            // them through verbatim would leak raw `[2K` sequences.
            allow_inline_fold_updates: std::io::IsTerminal::is_terminal(&std::io::stdout()),
            tty_fold: TtyToolOutputFoldState::default(),
        }
    }

    fn reset_stream_state(&mut self) {
        self.active_stream_tool_call_id = None;
        self.pending_utf8.clear();
        self.render_full_pty_stream = false;
        self.visual_output_probe.clear();
        self.visual_output_line.clear();
        self.visual_output_detected = false;
        self.at_line_start = true;
        self.streamed_any_output = false;
        self.fold_total_lines = 0;
        self.tty_fold.reset();
    }

    fn start_stream_output(&mut self, tool_call: &ToolCall) {
        if self.active_stream_tool_call_id.as_deref() == Some(tool_call.id.as_str()) {
            return;
        }
        self.reset_stream_state();
        self.active_stream_tool_call_id = Some(tool_call.id.clone());
        // `pty: true` is the caller's explicit request for interactive-terminal capability.
        // Forward this path's output in full so menus, confirmation prompts, and login
        // guides stay visible; ordinary piped commands remain silent so logs never flood
        // the terminal.
        self.render_full_pty_stream = execute_command_uses_pseudo_terminal(tool_call);
        // The streamed content is already rendered live; no extra label is needed.
    }

    fn push_stream_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.streamed_any_output = true;
        // Even when tool output is disabled, still record that a stream was received so
        // completion never falsely reports “no output”; but never bypass runtime_ctx's
        // terminal-output switch to write straight to stdout.
        if !crate::ai::driver::runtime_ctx::terminal_output_enabled() {
            return;
        }

        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        let sanitized = sanitize_for_terminal(&normalized);
        if sanitized.is_empty() {
            return;
        }

        if self.render_full_pty_stream {
            self.render_visible_stream_text(&sanitized);
            return;
        }

        if !self.visual_output_detected {
            self.visual_output_probe.push_str(&sanitized);
            if !contains_terminal_visual_grid(&self.visual_output_probe) {
                trim_visual_output_probe(&mut self.visual_output_probe);
                return;
            }

            self.visual_output_detected = true;
            let visual_output = std::mem::take(&mut self.visual_output_probe);
            self.push_visual_output_text(&visual_output);
            return;
        }

        self.push_visual_output_text(&sanitized);
    }

    /// Once a visual grid has been confirmed, only show the rows that actually form the
    /// grid; subsequent plain logs stay hidden.
    fn push_visual_output_text(&mut self, text: &str) {
        self.visual_output_line.push_str(text);
        while let Some(newline_at) = self.visual_output_line.find('\n') {
            let line = self.visual_output_line[..=newline_at].to_string();
            self.visual_output_line.drain(..=newline_at);
            if is_terminal_visual_grid_line(&line) {
                self.render_visible_stream_text(&line);
            }
        }

        // Non-newline plain logs must not pile up without bound; QR-code rows are only
        // judged once a newline arrives.
        if self.visual_output_line.len() > VISUAL_OUTPUT_PROBE_MAX_BYTES {
            self.visual_output_line.clear();
        }
    }

    fn flush_visual_output_line(&mut self) {
        if self.visual_output_line.is_empty() {
            return;
        }

        let line = std::mem::take(&mut self.visual_output_line);
        if is_terminal_visual_grid_line(&line) {
            // Append a newline so the completion status that follows does not stick to
            // the last visual-output line.
            self.render_visible_stream_text(&format!("{line}\n"));
        }
    }

    /// Render streamed text approved for display: explicit PTY output, or an identified
    /// visual grid.
    fn render_visible_stream_text(&mut self, text: &str) {
        if self.allow_inline_fold_updates {
            let _ = self.tty_fold.push_text(text);
            let _ = std::io::stdout().flush();
            return;
        }

        for ch in text.chars() {
            if ch == '\n' {
                self.fold_total_lines += 1;
                if self.fold_total_lines <= TOOL_OUTPUT_FOLD_MAX_VISIBLE {
                    print!("{RESET}\n");
                    self.at_line_start = true;
                } else if self.fold_total_lines == TOOL_OUTPUT_FOLD_MAX_VISIBLE + 1 {
                    print!("{RESET}\n");
                    self.at_line_start = true;
                    println!(
                        "  {ACCENT_RULE}│{RESET} {ACCENT_MUTED}··· streaming output folded until completion ···{RESET}"
                    );
                }
            } else if self.fold_total_lines < TOOL_OUTPUT_FOLD_MAX_VISIBLE {
                if self.at_line_start {
                    print!("{}", format_tool_output_prefix());
                    self.at_line_start = false;
                }
                print!("{ch}");
            }
        }
        let _ = std::io::stdout().flush();
    }

    fn push_stream_text_for_tool(&mut self, tool_call: &ToolCall, text: &str) {
        if text.is_empty() {
            return;
        }
        self.start_stream_output(tool_call);
        self.push_stream_text(text);
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
        self.flush_visual_output_line();
        if !crate::ai::driver::runtime_ctx::terminal_output_enabled() {
            return;
        }
        if !self.visual_output_detected && !self.render_full_pty_stream {
            return;
        }
        if self.allow_inline_fold_updates {
            let _ = self.tty_fold.finish();
            return;
        }
        if self.fold_total_lines > TOOL_OUTPUT_FOLD_MAX_VISIBLE {
            let folded = self.fold_total_lines - TOOL_OUTPUT_FOLD_MAX_VISIBLE;
            println!("  {ACCENT_RULE}│{RESET} {ACCENT_MUTED}··· {folded} lines folded ···{RESET}");
            self.at_line_start = true;
        } else if !self.at_line_start {
            if newline {
                print!("{RESET}\n");
                self.at_line_start = true;
            } else {
                print!("{RESET}");
            }
            let _ = std::io::stdout().flush();
        }
    }

    fn print_prepared_tool_result(&mut self, prepared: &PreparedToolResult) {
        // The terminal no longer prints tool output content; only the status line is kept.
        let _ = prepared;
    }

    fn print_captured_command_output(&mut self, prepared: &PreparedToolResult) {
        // The terminal no longer prints tool output content; only the status line is kept.
        let _ = prepared;
    }
}

/// Streamed output is only shown in full when `execute_command` explicitly requests a
/// PTY. A PTY is the opt-in signal for interactive CLIs (menus, confirmations, QR-login,
/// etc.); regular commands keep going through visual-grid detection so build/search logs
/// are not written to the terminal.
pub(in crate::ai::driver::turn_runtime) fn execute_command_uses_pseudo_terminal(
    tool_call: &ToolCall,
) -> bool {
    tool_call.function.name == "execute_command"
        && serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
            .ok()
            .and_then(|args| args.get("pty").and_then(serde_json::Value::as_bool))
            == Some(true)
}

/// Render the arguments of command-like tools (e.g. `execute_command`) into a single-line
/// readable command text, printed in the terminal when the tool starts. Multi-line
/// commands are folded to one line; overlong ones are truncated.
/// Returns None when parsing fails (missing `command` field or invalid JSON).
pub(in crate::ai::driver::turn_runtime) fn format_command_input(arguments: &str) -> Option<String> {
    let args: serde_json::Value = serde_json::from_str(arguments).ok()?;
    let command = args.get("command")?.as_str()?;
    // Fold newlines so a command never spans multiple terminal lines and disturbs the
    // status-line layout
    let mut line = command.replace('\n', " ⏎ ").replace('\r', "");
    const MAX_CHARS: usize = 200;
    if line.chars().count() > MAX_CHARS {
        let kept: String = line.chars().take(MAX_CHARS.saturating_sub(1)).collect();
        line = format!("{kept}…");
    }
    if let Some(cwd) = args.get("cwd").and_then(serde_json::Value::as_str) {
        if !cwd.is_empty() {
            line.push_str(&format!("  (cwd: {cwd})"));
        }
    }
    if args.get("pty").and_then(serde_json::Value::as_bool) == Some(true) {
        line.push_str("  (PTY)");
    }
    Some(line)
}

impl tools::ToolExecutionObserver for TerminalToolObserver<'_> {
    fn on_tool_started(&mut self, tool_call: &ToolCall) {
        if matches!(
            tool_call.function.name.as_str(),
            "execute_command" | "run_command" | "shell" | "bash"
        ) {
            if let Some(line) = format_command_input(&tool_call.function.arguments) {
                print_tool_command_line(&line);
            }
        }
    }

    fn on_tool_stream(&mut self, tool_call: &ToolCall, chunk: &[u8]) {
        self.pending_utf8.extend_from_slice(chunk);
        loop {
            match std::str::from_utf8(&self.pending_utf8) {
                Ok(text) => {
                    let text = text.to_string();
                    self.pending_utf8.clear();
                    self.push_stream_text_for_tool(tool_call, &text);
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
                    self.push_stream_text_for_tool(tool_call, &text);

                    if err.error_len().is_some() {
                        self.flush_pending_utf8();
                    }
                }
            }
        }
    }

    fn on_tool_finished(&mut self, tool_call: &ToolCall, run_result: &tools::RunOneResult) {
        let streamed_output = self.active_stream_tool_call_id.as_deref()
            == Some(tool_call.id.as_str())
            && self.streamed_any_output;
        if streamed_output {
            let is_failure = streamed_tool_result_is_failure(tool_call, run_result);
            self.finish_stream_output(is_failure);

            if is_failure {
                if let Some(exit_line) = run_result.tool_result.content.lines().next() {
                    print_tool_note_line("error", exit_line);
                }
            }

            self.reset_stream_state();
            return;
        }

        let prepared = prepare_recent_tool_result(
            self.app,
            &tool_call.function.name,
            &run_result.tool_result.content,
        );
        self.print_prepared_tool_result(&prepared);
    }
}

pub(in crate::ai::driver::turn_runtime) fn streamed_tool_result_is_failure(
    tool_call: &ToolCall,
    run_result: &tools::RunOneResult,
) -> bool {
    !run_result.ok
        || (tool_call.function.name == "execute_command"
            && run_result.tool_result.content.starts_with("Exit code:"))
}

/// Step 5: per-round ToolExecutor adapter that bridges the port contract to real dispatch.
///
/// Holds all the context real dispatch needs; `&McpClient` is obtained inside `execute`
/// from `SharedMcpClient`'s `routing_snapshot()` snapshot, so no lock is held across
/// dispatch (avoiding a second `lock()` on the same `Mutex` deadlocking against the MCP
/// branch in subagent `run_turn`/`tools/mod.rs`). The caller's `mcp_client` parameter is
/// likewise a `routing_snapshot()` value in production (empty servers, routed from the
/// same source as the real client through the shared `cached_server_prefixes` Arc, see
/// orchestrator.rs:1093), equivalent to the snapshot routing result; real MCP execution
/// always goes through `shared_mcp_client`.
pub(in crate::ai::driver::turn_runtime) struct RoundToolExecutorAdapter {
    pub(super) session_id: String,
    pub(super) shared_mcp_client: SharedMcpClient,
    pub(super) allowed_tool_names: FastSet<String>,
    pub(super) suppressed_read_only_results: HashMap<String, String>,
    pub(super) iteration: usize,
}

impl ToolExecutor for RoundToolExecutorAdapter {
    fn execute<'a>(
        &'a self,
        app: &'a mut App,
        tool_calls: Vec<ToolCall>,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<ToolExecOutput, Box<dyn std::error::Error + Send + Sync>>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            let mut observer = TerminalToolObserver::new(app);
            let _streaming_guard = ToolExecutionStreamingGuard::new(&app.streaming);
            // Do not hold the lock across dispatch: take a non-locking routing_snapshot for
            // routing so a temporary MutexGuard does not outlive the whole let statement.
            // Otherwise a synchronous `task` subagent running `run_turn` on another thread
            // (`mcp_client.lock()` in prepare.rs) would never acquire this lock, while the
            // parent thread blocks waiting for the subagent to return → cross-thread
            // deadlock (symptom: subagent stuck in preparing context).
            // See the mcp_snapshot test-helper comments in this file.
            let snapshot = self.shared_mcp_client.lock().unwrap().routing_snapshot();
            let result = execute_tool_calls_with_suppressed_read_only_calls(
                &self.session_id,
                &snapshot,
                &self.shared_mcp_client,
                &tool_calls,
                &self.allowed_tool_names,
                Some(&mut observer),
                self.iteration,
                &self.suppressed_read_only_results,
            )
            // Dispatch returns `Box<dyn Error>` (not Send+Sync) while the port requires
            // Send+Sync: wrap in `io::Error` to preserve the error message for string
            // display upstream.
            .map_err(|e| std::io::Error::other(format!("tool dispatch failed: {e}")))?;
            Ok(result.into_tool_exec_output())
        })
    }
}
