use super::inline_recovery::{
    collect_valid_tool_calls, ensure_tool_calls_section_open, normalize_inline_tool_call_markup,
    recover_inline_tool_calls,
};
use std::cell::RefCell;
use std::io::{self, IsTerminal, Write};
use std::time::{Duration, Instant};

use crate::ai::{
    config_schema::AiConfig,
    driver::{print::sanitize_for_terminal, runtime_ctx},
    models,
    provider::{self, ProviderAdapter},
    request::{StreamChunk, merge_reasoning_fragments},
    theme::{ACCENT_MUTED, DIM, RESET},
    types::{App, StreamOutcome, StreamResult, take_stream_cancelled},
};
use crate::commonw::configw;

use super::{
    MarkdownStreamRenderer,
    extract::{StreamTextEvent, extract_chunk_events_streaming, normalize_stream_text},
    framing, normalize,
    render::markdown::{
        clamp_line_to_terminal_row_with_reserve, live_preview_cursor_rows,
        wrap_line_to_terminal_rows_with_reserve,
    },
    splitter::{InternalToolCallStreamEvent, StreamSplitSegment},
    state::{
        StreamChunkStep, StreamContentState, StreamMarkers, StreamProcessingState,
        TerminalDedupeState, ToolCallBuilder,
    },
};

/// Maximum number of decode errors before giving up and returning partial content
const MAX_DECODE_ERRORS: usize = 3;
/// Delay in milliseconds between retry attempts on transient errors
const DECODE_ERROR_RETRY_DELAY_MS: u64 = 100;
/// Grace window after an OpenAI-compatible `finish_reason` chunk. Some backends
/// do not emit `[DONE]` or close the HTTP body, while others can still send a
/// final snapshot immediately after the finish chunk.
const FINISH_REASON_GRACE_MS: u64 = 750;

/// Idle timeout: after some content arrived, a long stretch without a new chunk means the server has silently finished.
/// Some providers send neither finish_reason nor close the connection after finishing; only this timeout catches that.
const STREAM_IDLE_TIMEOUT_SECS: u64 = 45;
/// First-chunk timeout: the request was sent but the server never sends the first byte (queued, stuck gateway, ...).
/// Longer than the idle timeout, since some models take time to cold-start or queue.
const STREAM_FIRST_CHUNK_TIMEOUT_SECS: u64 = 90;
/// Default visible-window height for `thinking` in the terminal. Only affects display, not reasoning accumulation.
/// Streaming shows the most recent N lines (default 2); when thinking ends, `finalize_fold` forces a pure-summary
/// fold (redraws with a 0-line window) so conclusions/questions restated at the tail of thinking are not shown twice
/// alongside the final answer in the terminal.
const DEFAULT_THINKING_MAX_VISIBLE_LINES: usize = 2;
/// Indentation for folded thinking/subagent bodies: header/footer use 2 spaces, body is indented one more level.
const THINKING_FOLD_BODY_INDENT: &str = "    ";
const THINKING_FOLD_BODY_INDENT_WIDTH: usize = 4;
/// Terminals usually wrap at the right edge with delayed-wrap; folded redraws always leave two extra columns so that
/// a missing terminal flag or a one-column width/char-width drift cannot trigger an implicit wrap that was not counted
/// in cursor-up, leaving residue from the old window under the `✓`.
const FOLD_REWRITE_RIGHT_MARGIN_COLS: usize = 2;
/// Shortest repeated fragment and decision count for reasoning-stream degeneration. Only reasoning is checked, so
/// legitimately repeated body the model was asked to produce (tables, code, test data) is not misjudged as degeneration.
const MIN_REASONING_REPEAT_CHARS: usize = 16;
const MAX_REASONING_REPEAT_CHARS: usize = 512;
const REASONING_REPEAT_COUNT: usize = 3;
const DEGENERATE_REPETITION_FINISH_REASON: &str = "degenerate_repetition";
/// Cap on accumulated streaming tool-call arguments (total across all tool calls in one turn).
/// Once the model opens a tool call it should close quickly; if arguments keep growing until the cap is hit (e.g.
/// an endless loop concatenating the same body in apply_patch), the output has degenerated. Existing degeneration
/// detection only covers reasoning/assistant text, not tool arguments; this total cap is a backstop against infinite waits and memory growth.
const MAX_TOOL_ARG_BYTES: usize = 1 << 20; // 1 MiB

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct StreamPayloadOutcome {
    should_stop: bool,
    meaningful_progress: bool,
}

impl StreamPayloadOutcome {
    fn stop() -> Self {
        Self {
            should_stop: true,
            meaningful_progress: false,
        }
    }

    fn stop_with_progress() -> Self {
        Self {
            should_stop: true,
            meaningful_progress: true,
        }
    }
}

fn initial_stream_processing_state(app: &App) -> StreamProcessingState {
    StreamProcessingState::with_filters(app.hooks.stream_filters().clone())
}

pub(super) async fn stream_response(
    app: &mut App,
    response: &mut reqwest::Response,
    current_history: &mut String,
    terminal_dedupe_candidate: Option<&str>,
) -> Result<StreamResult, Box<dyn std::error::Error>> {
    let mut markers = StreamMarkers::new();
    let mut state = initial_stream_processing_state(app);
    state.render.terminal_dedupe = terminal_dedupe_candidate
        .map(str::trim)
        .filter(|candidate| !candidate.is_empty())
        .map(|candidate| TerminalDedupeState {
            candidate: candidate.to_string(),
            buffered_terminal_output: String::new(),
        });
    // Reasoners with a prefilled `thinking` template inline their chain in the content channel and only close with a dangling
    // `response`, never producing reasoning_content. Arm the splitter for such models to pull leaked
    // reasoning back into reasoning, so the chain is not dumped into visible body together with the final answer.
    if models::reasoning_in_content_enabled(&app.current_model) {
        state.content.content_think_demuxer.arm();
    }
    configure_thinking_fold(&mut state);
    configure_subagent_preview_fold(app, &mut state, &mut markers);
    // A completed answer is provisional until the driver's completion/citation
    // gates accept it. Keep assistant prose transactional while preserving live
    // thinking and tool activity.
    state.render.defer_assistant_body = runtime_ctx::terminal_output_enabled();
    let adapter = provider::adapter_for(
        models::model_adapter(&app.current_model),
        &models::endpoint_for_model(&app.current_model, &app.config.endpoint),
    );

    if should_show_waiting_hint(app) {
        print_waiting_hint(&mut state)?;
    }

    let mut last_meaningful_progress_at = Instant::now();
    let has_meaningful_progress = |s: &StreamProcessingState| -> bool {
        !s.content.assistant_text.is_empty()
            || !s.content.tool_calls_map.is_empty()
            || s.content.finish_reason_seen
    };
    let mut idle_timeout_secs = None;

    while !app.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        if let Some(result) = immediate_cancel_result(app, &mut state) {
            return Ok(result);
        }

        // Use a shorter idle timeout when there is executable/visible progress; empty packets, usage-only and heartbeat
        // do not refresh this timer, so a provider pushing useless packets cannot keep the stream open forever.
        let timeout_secs = if has_meaningful_progress(&state) {
            STREAM_IDLE_TIMEOUT_SECS
        } else {
            STREAM_FIRST_CHUNK_TIMEOUT_SECS
        };
        let chunk_result = if state.content.finish_reason_seen {
            tokio::select! {
                chunk = response.chunk() => chunk,
                _ = wait_for_interrupt(app) => {
                    return Ok(cancelled_stream_result(&mut state));
                }
                _ = tokio::time::sleep(Duration::from_millis(FINISH_REASON_GRACE_MS)) => break,
            }
        } else {
            let idle_remaining = Duration::from_secs(timeout_secs)
                .saturating_sub(last_meaningful_progress_at.elapsed());
            tokio::select! {
                chunk = response.chunk() => chunk,
                _ = wait_for_interrupt(app) => {
                    return Ok(cancelled_stream_result(&mut state));
                }
                _ = tokio::time::sleep(idle_remaining) => {
                    idle_timeout_secs = Some(timeout_secs);
                    break;
                }
            }
        };

        match process_chunk_result(
            app,
            current_history,
            &markers,
            &mut state,
            adapter,
            chunk_result,
        )
        .await?
        {
            StreamChunkStep::Continue {
                meaningful_progress,
            } => {
                if meaningful_progress {
                    last_meaningful_progress_at = Instant::now();
                }
            }
            StreamChunkStep::Stop => break,
            StreamChunkStep::Return(result) => return Ok(result),
        }
    }

    if let Some(result) =
        process_pending_tail(app, current_history, &markers, &mut state, adapter).await?
    {
        return Ok(result);
    }

    if let Some(timeout_secs) = idle_timeout_secs.filter(|_| !state.content.finish_reason_seen) {
        state.content.stream_idle_timed_out = true;
        if runtime_ctx::terminal_output_enabled() {
            clear_waiting_hint(&mut state)?;
            let stdout = io::stdout();
            let mut out = stdout.lock();
            writeln!(
                out,
                "  ⚠ 响应流连续 {timeout_secs} 秒无有效进展，按流中断处理…"
            )?;
            out.flush()?;
        }
    }

    finalize_stream_response(app, current_history, &markers, state)
}

/// Whether to show a compact "waiting for model output" status hint in the terminal.
/// Applies to all TTY sessions. Written and flushed on its own line so it appears
/// immediately; once the first visible chunk arrives it is cleared with
/// \x1b[1A\r\x1b[2K, leaving no extra lines behind.
fn should_show_waiting_hint(app: &App) -> bool {
    runtime_ctx::terminal_output_enabled()
        && io::stdout().is_terminal()
        && !app.shutdown.load(std::sync::atomic::Ordering::Relaxed)
}

fn print_waiting_hint(state: &mut StreamProcessingState) -> io::Result<()> {
    if state.render.waiting_hint_active {
        return Ok(());
    }
    // Waiting hint on its own line: cleared with \x1b[1A\r\x1b[2K when the first chunk arrives.
    write_waiting_hint_line("waiting…")?;
    state.render.waiting_hint_active = true;
    state.render.waiting_hint_tool_call = false;
    Ok(())
}

fn write_waiting_hint_line(label: &str) -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "  {ACCENT_MUTED}⠋ {label}{RESET}")?;
    out.flush()
}

fn sanitize_waiting_hint_tool_name(function_name: &str) -> String {
    let sanitized = sanitize_for_terminal(function_name);
    let single_line = sanitized.split_whitespace().collect::<Vec<_>>().join(" ");
    if single_line.is_empty() {
        "tool".to_string()
    } else {
        single_line
    }
}

fn print_tool_call_waiting_hint(
    state: &mut StreamProcessingState,
    function_name: &str,
) -> io::Result<()> {
    if state.render.waiting_hint_active {
        clear_waiting_hint(state)?;
    }
    let function_name = sanitize_waiting_hint_tool_name(function_name);
    write_waiting_hint_line(&format!("receiving `{function_name}` arguments…"))?;
    state.render.waiting_hint_active = true;
    state.render.waiting_hint_tool_call = true;
    Ok(())
}

fn configure_thinking_fold(state: &mut StreamProcessingState) {
    state.render.thinking_fold.max_visible_lines = resolve_thinking_fold_max_visible_lines(
        io::stdout().is_terminal(),
        configw::get_all_config()
            .get_opt(AiConfig::OUTPUT_THINKING_MAX_VISIBLE_LINES)
            .as_deref(),
    );
    state.render.thinking_fold.rewrite_right_margin_cols = FOLD_REWRITE_RIGHT_MARGIN_COLS;
}

fn configure_subagent_preview_fold(
    app: &App,
    state: &mut StreamProcessingState,
    markers: &mut StreamMarkers,
) {
    state.render.subagent_fold.rewrite_right_margin_cols = FOLD_REWRITE_RIGHT_MARGIN_COLS;
    if !io::stdout().is_terminal() || runtime_ctx::current_subagent_depth() == 0 {
        state.render.subagent_fold.max_visible_lines = usize::MAX;
        return;
    }

    state.render.subagent_fold.max_visible_lines = resolve_thinking_fold_max_visible_lines(
        true,
        configw::get_all_config()
            .get_opt(AiConfig::OUTPUT_THINKING_MAX_VISIBLE_LINES)
            .as_deref(),
    );
    markers.enable_subagent_preview(&app.current_agent);
    if let (Some(header), Some(footer)) = (
        markers.subagent_fold_header.as_deref(),
        markers.subagent_fold_footer.as_deref(),
    ) {
        state.render.subagent_fold.set_labels(header, footer);
    }
}

fn resolve_thinking_fold_max_visible_lines(is_tty: bool, raw: Option<&str>) -> usize {
    if !is_tty {
        return usize::MAX;
    }

    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return DEFAULT_THINKING_MAX_VISIBLE_LINES;
    };

    match raw.parse::<usize>() {
        Ok(0) => usize::MAX,
        Ok(lines) => lines,
        Err(_) => DEFAULT_THINKING_MAX_VISIBLE_LINES,
    }
}

fn upgrade_waiting_hint_for_buffering(state: &mut StreamProcessingState) -> io::Result<()> {
    if !state.render.waiting_hint_active
        || state.render.waiting_hint_buffering
        || state.render.waiting_hint_tool_call
    {
        return Ok(());
    }
    // Move the cursor up, clear the line, then rewrite it so the buffering state stays visible.
    let stdout = io::stdout();
    let mut out = stdout.lock();
    write!(out, "\x1b[1A\r\x1b[2K")?;
    writeln!(out, "  {ACCENT_MUTED}⠋ buffering…{RESET}")?;
    out.flush()?;
    state.render.waiting_hint_buffering = true;
    Ok(())
}

/// Show a compact "generating…" hint while the assistant body is being withheld
/// (defer_assistant_body) and thinking is closed. The final answer is streamed but
/// not rendered until the completion/citation gates accept it, so without this the
/// terminal stays blank for the whole generation. Upgrades the initial "waiting…"
/// line in place (cursor up + clear + rewrite) and stays put until
/// `clear_waiting_hint` fires — at the next renderable chunk or at stream end.
fn show_deferred_body_buffering_hint(state: &mut StreamProcessingState) -> io::Result<()> {
    if !io::stdout().is_terminal()
        || state.render.waiting_hint_buffering
        || state.render.waiting_hint_tool_call
    {
        return Ok(());
    }
    let stdout = io::stdout();
    let mut out = stdout.lock();
    if state.render.waiting_hint_active {
        write!(out, "\x1b[1A\r\x1b[2K")?;
    }
    writeln!(out, "  {ACCENT_MUTED}⠋ generating…{RESET}")?;
    out.flush()?;
    state.render.waiting_hint_active = true;
    state.render.waiting_hint_buffering = true;
    Ok(())
}

pub(super) fn clear_waiting_hint(state: &mut StreamProcessingState) -> io::Result<()> {
    if !state.render.waiting_hint_active {
        return Ok(());
    }
    // Cursor up one line + \r + clear line: erase the standalone hint line so content prints in place.
    let stdout = io::stdout();
    let mut out = stdout.lock();
    write!(out, "\x1b[1A\r\x1b[2K")?;
    out.flush()?;
    state.render.waiting_hint_active = false;
    state.render.waiting_hint_buffering = false;
    state.render.waiting_hint_tool_call = false;
    Ok(())
}

fn immediate_cancel_result(app: &App, state: &mut StreamProcessingState) -> Option<StreamResult> {
    stream_interrupt_requested(app).then(|| cancelled_stream_result(state))
}

fn flush_inline_markup_normalizer_on_cancel(state: &mut StreamProcessingState) {
    let normalized = state.content.inline_markup_normalizer.flush();
    if normalized.is_empty() {
        return;
    }
    // Finish the protocol parsers so a partial tool call stays stripped, but discard
    // emitted tool events because an interrupted stream must never execute them.
    let (cleaned, _) = state.content.hermes_tool_call_streamer.push(&normalized);
    let (cleaned, _) = state.content.anthropic_tool_call_streamer.push(&cleaned);
    let (cleaned, _) = state.content.bare_xml_tool_call_streamer.push(&cleaned);
    let content = normalize_stream_text(cleaned);
    state.content.assistant_text.push_str(&content);
}

/// Thinking-fold cleanup on cancel/interrupt: if the fold window is still active we
/// must erase the current window and settle with a `✓`; otherwise a half-drawn
/// thinking window stays on screen, and the fresh state of the next retry draws a
/// new header below it — stacking into a "duplicate header + large blank area".
fn cancelled_stream_result(state: &mut StreamProcessingState) -> StreamResult {
    flush_inline_markup_normalizer_on_cancel(state);
    // A content-channel reasoner can keep all received text inside the demuxer until
    // it observes its response delimiter. Ctrl+C must not discard that received
    // prefix merely because the delimiter never arrived.
    let (residual_reasoning, residual_content) = state.content.content_think_demuxer.flush();
    state.content.reasoning_text.push_str(&residual_reasoning);
    state.content.assistant_text.push_str(&residual_content);

    if runtime_ctx::terminal_output_enabled() {
        let _ = clear_waiting_hint(state);
        if state.render.thinking_fold.active {
            let _ = finalize_thinking_fold(state);
        } else if state.render.subagent_fold.active {
            let _ = finalize_subagent_preview_fold(state);
        } else if state.content.thinking_open {
            print!("\x1b[0m");
            let _ = io::stdout().flush();
        }
    }
    StreamResult {
        outcome: StreamOutcome::Cancelled,
        tool_calls: Vec::new(),
        assistant_text: std::mem::take(&mut state.content.assistant_text),
        hidden_meta: String::new(),
        reasoning_text: std::mem::take(&mut state.content.reasoning_text),
        reasoning_items: Vec::new(),
        skip_response_drain: true,
        truncated_by_length: false,
        stream_error: false,
        finish_reason_value: None,
        usage_prompt_tokens: 0,
        usage_cached_prompt_tokens: 0,
        usage_completion_tokens: 0,
        usage_reasoning_tokens: 0,
    }
}

async fn process_chunk_result<T: AsRef<[u8]>>(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter: &'static dyn ProviderAdapter,
    chunk_result: Result<Option<T>, reqwest::Error>,
) -> Result<StreamChunkStep, Box<dyn std::error::Error>> {
    match chunk_result {
        Ok(Some(chunk)) => {
            framing::push_chunk(&mut state.framing, chunk.as_ref());
            state.framing.decode_error_count = 0;
            consume_pending_complete_lines(app, current_history, markers, state, adapter).await
        }
        Ok(None) => Ok(StreamChunkStep::Stop),
        Err(err) => {
            if let Some(result) = handle_stream_decode_error(app, markers, state, err).await {
                Ok(StreamChunkStep::Return(result))
            } else {
                Ok(StreamChunkStep::Continue {
                    meaningful_progress: false,
                })
            }
        }
    }
}

async fn consume_pending_complete_lines(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter: &'static dyn ProviderAdapter,
) -> Result<StreamChunkStep, Box<dyn std::error::Error>> {
    // Move the pending buffer out so line slices can borrow from it while `state`
    // remains available for mutation inside `process_stream_line()`.
    let lines = framing::take_complete_lines(&mut state.framing);
    let mut should_stop = false;
    let mut meaningful_progress = false;
    for line in lines {
        let outcome = process_stream_line(app, current_history, markers, state, adapter, &line)?;
        meaningful_progress |= outcome.meaningful_progress;
        if outcome.should_stop {
            should_stop = true;
            break;
        }
    }
    Ok(if should_stop {
        StreamChunkStep::Stop
    } else {
        StreamChunkStep::Continue {
            meaningful_progress,
        }
    })
}

async fn process_pending_tail(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter: &'static dyn ProviderAdapter,
) -> Result<Option<StreamResult>, Box<dyn std::error::Error>> {
    if state.framing.pending.is_empty() {
        // Even with pending empty, still check whether sse_event_data holds one last
        // unflushed event. Some providers don't send the final blank line (\n\n)
        // before closing the connection, which would drop the last SSE event.
        if !state.framing.sse_event_data.trim().is_empty() {
            if flush_sse_event(app, current_history, markers, state, adapter)?.should_stop {
                let final_state = std::mem::replace(state, StreamProcessingState::new());
                return Ok(Some(finalize_stream_response(
                    app,
                    current_history,
                    markers,
                    final_state,
                )?));
            }
        }
        return Ok(None);
    }

    let Some(line) = framing::take_pending_tail(&mut state.framing) else {
        return Ok(None);
    };
    if !line.is_empty() {
        let _ = process_stream_line(app, current_history, markers, state, adapter, &line)?;
    }
    if flush_sse_event(app, current_history, markers, state, adapter)?.should_stop {
        let final_state = std::mem::replace(state, StreamProcessingState::new());
        return Ok(Some(finalize_stream_response(
            app,
            current_history,
            markers,
            final_state,
        )?));
    }
    Ok(None)
}

fn finalize_stream_response(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    mut state: StreamProcessingState,
) -> Result<StreamResult, Box<dyn std::error::Error>> {
    let render_terminal = runtime_ctx::terminal_output_enabled();
    let deferred_dedupe_candidate = state
        .render
        .terminal_dedupe
        .as_ref()
        .map(|dedupe| dedupe.candidate.clone());
    if render_terminal {
        clear_waiting_hint(&mut state)?;
    }

    if render_terminal && state.content.thinking_open {
        flush_digest_filter_to_terminal(markers, &mut state, true)?;
        if state.render.thinking_fold.active {
            finalize_thinking_fold(&mut state)?;
        } else {
            write_stream_content(
                &format!("\n{}\n", markers.end_thinking_tag),
                &mut state.render.markdown,
                false,
            )?;
        }
    }

    if render_terminal && state.render.subagent_fold.active {
        finalize_subagent_preview_fold(&mut state)?;
    }

    flush_inline_markup_normalizer(app, current_history, markers, &mut state)?;

    // Flush whatever the inline `response` splitter still holds. Still capturing means `response` never arrived:
    // under withhold the whole buffer safely falls back as **content** (better to degrade to "thinking leaked into body"
    // than to lose the visible answer). Always empty for models that never arm the splitter.
    let (residual_reasoning, residual_content) = state.content.content_think_demuxer.flush();
    if !residual_reasoning.is_empty() {
        state.content.reasoning_text.push_str(&residual_reasoning);
    }
    if !residual_content.is_empty() {
        commit_visible_content(app, current_history, markers, &mut state, residual_content)?;
    }

    if render_terminal {
        // Residue must still go through the dedup/fold/style pipeline; it cannot be written straight to the terminal.
        flush_digest_filter_to_terminal(markers, &mut state, false)?;
        flush_terminal_splitter(&mut state, markers)?;
        let suppress_duplicate = final_assistant_matches_terminal_dedupe(&state);
        disable_terminal_dedupe(&mut state, suppress_duplicate)?;
        state.render.markdown.flush_pending()?;
        // Residual content committed below may have re-shown the deferred-body hint;
        // the stream is over, so clear it before the driver renders the final answer.
        clear_waiting_hint(&mut state)?;
    }

    if take_stream_cancelled(app) {
        return Ok(cancelled_stream_result(&mut state));
    }

    // AIOS: flush any pending LLM usage to kernel `/dev/llm` before returning.
    // Prefer the model echoed by the provider; fall back to what we requested.
    // Snapshot usage stats first so StreamResult truncation diagnostics can use them (take() consumes).
    let usage_snapshot = state.pending_llm_usage.as_ref().map(|(_, u)| {
        let cached = u
            .prompt_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens)
            .unwrap_or(0);
        let reasoning = u
            .completion_tokens_details
            .as_ref()
            .map(|d| d.reasoning_tokens)
            .unwrap_or(0);
        (u.prompt_tokens, cached, u.completion_tokens, reasoning)
    });
    if let Some((echoed_model, usage)) = state.pending_llm_usage.take() {
        let model_for_pricing = if echoed_model.is_empty() {
            app.current_model.clone()
        } else {
            echoed_model
        };
        let _ = crate::ai::request::charge_llm_usage_to_kernel(app, &model_for_pricing, &usage, 0);
        maybe_print_prompt_cache_metrics(&usage);
    }

    let stream_error = state.content.stream_idle_timed_out;
    let (mut tool_calls, dropped_malformed) =
        collect_valid_tool_calls(&mut state.content.tool_calls_map);
    state.content.dropped_malformed_tool_call = dropped_malformed;
    if stream_error || state.content.tool_args_cap_exceeded {
        // Idle timeout before finish_reason does not prove the tool call is complete; even if the current
        // arguments happen to be valid JSON, an operation that may still be generating must not run early. The same
        // applies when arguments were cut off by hitting the cap: the model was force-stopped, so arguments may be incomplete.
        tool_calls.clear();
    }

    // Fallback: some providers return function calls as plain content instead of going through
    // delta.tool_calls[]. When streaming parse misses, do one conservative recovery pass over the full assistant_text.
    // Same principle as idle timeout when arguments were cut off at the cap: the model was force-stopped, so any
    // suspected inline tool call in assistant_text is equally untrustworthy and is skipped, keeping the drop logic intact.
    if !stream_error && !state.content.tool_args_cap_exceeded && tool_calls.is_empty() {
        if let Some(recovered) = recover_inline_tool_calls(&state.content.assistant_text) {
            tool_calls = recovered;
            // The protocol payload is neither assistant body nor a model self_note. After successful recovery, drop it
            // outright so a no-tool handoff does not persist DSML/JSON as an internal_note.
            state.content.assistant_text.clear();
        }
    }

    let truncated_by_length = state
        .content
        .finish_reason_value
        .as_deref()
        .is_some_and(|reason| reason.eq_ignore_ascii_case("length"));
    let degenerate_repetition = state
        .content
        .finish_reason_value
        .as_deref()
        .is_some_and(|reason| reason == DEGENERATE_REPETITION_FINISH_REASON);

    let outcome = if stream_error {
        StreamOutcome::Truncated
    } else if !tool_calls.is_empty() {
        StreamOutcome::ToolCall
    } else {
        let has_text = !state.content.assistant_text.trim().is_empty();
        let has_reasoning = !state.content.reasoning_text.trim().is_empty();
        // Truncation takes priority: this turn had no valid tool calls, but some were dropped (half-cut arguments JSON).
        // Ending such an "interrupted mid-work" turn silently as Completed would make large-file write_file
        // operations vanish. Escalate to retryable Truncated so the upper layer injects a shrink hint and retries.
        //
        // Note: finish_reason=length (hitting the output cap) alone does NOT trigger Truncated, because reasoning models
        // often return finish_reason=length after reasoning tokens filled the output budget while the displayable
        // assistant_text is actually complete. With both visible text and finish_reason=length, treat it as
        // Completed to avoid pointless retry loops. Only when there is no visible output at all is length truncation
        // retried as Truncated (the model may have been cut off right as it started outputting).
        if degenerate_repetition {
            StreamOutcome::Truncated
        } else if state.content.dropped_malformed_tool_call {
            StreamOutcome::Truncated
        } else if truncated_by_length && !has_text {
            // finish_reason=length with no visible text: the model may have produced only reasoning
            // before being cut off, or produced nothing. Retry at a lower effort so budget goes to actual content.
            StreamOutcome::Truncated
        } else if has_reasoning && !has_text && !state.content.finish_reason_seen {
            // Reasoning-only early stop: the stream ended (idle timeout / early EOF, common for GLM and other
            // enable_thinking models that sit on their chain without visible content and hit the idle timeout)
            // with only thinking emitted, no visible text and **no finish_reason at all**. Ending such a
            // "cut off mid-thinking" turn silently as Completed would leave the answer empty. Escalate to
            // retryable Truncated; the upper layer downgrades / disables thinking and retries.
            //
            // Complementary to the length branch above: length is an explicit server-side truncation; here the
            // stream stopped early without ever seeing the end marker. Distinct from a normal finish_reason=stop
            // reasoning-only response — there finish_reason_seen=true, so this branch is not entered and it stays Completed.
            StreamOutcome::Truncated
        } else if !has_text && !has_reasoning {
            // Detect empty responses: no text, no tool calls, no reasoning content.
            // Usually a provider-side problem (rate limit, model error); trigger a retry.
            StreamOutcome::EmptyResponse
        } else {
            StreamOutcome::Completed
        }
    };

    if render_terminal && state.render.defer_assistant_body && outcome != StreamOutcome::Completed {
        let visible_text = crate::ai::request::strip_digest_blocks(&state.content.assistant_text);
        let duplicate = deferred_dedupe_candidate
            .as_deref()
            .is_some_and(|candidate| candidate.trim() == visible_text.trim());
        if !duplicate && !visible_text.trim().is_empty() {
            super::render_markdown_block(&visible_text)?;
        }
    }

    Ok(StreamResult {
        outcome,
        tool_calls,
        assistant_text: state.content.assistant_text,
        hidden_meta: state.content.hidden_meta,
        reasoning_text: state.content.reasoning_text,
        reasoning_items: std::mem::take(&mut state.content.reasoning_items),
        skip_response_drain: true,
        truncated_by_length,
        stream_error,
        finish_reason_value: state.content.finish_reason_value.clone(),
        usage_prompt_tokens: usage_snapshot.map(|(p, _, _, _)| p).unwrap_or(0),
        usage_cached_prompt_tokens: usage_snapshot.map(|(_, cp, _, _)| cp).unwrap_or(0),
        usage_completion_tokens: usage_snapshot.map(|(_, _, c, _)| c).unwrap_or(0),
        usage_reasoning_tokens: usage_snapshot.map(|(_, _, _, r)| r).unwrap_or(0),
    })
}

/// When `ai.prompt_cache.show_metrics` (default on) is set and this request hit the prompt
/// cache, print one line of cache-hit metrics. OpenAI / DashScope etc. cache server-side;
/// this just visualizes the `cached_tokens` they already reported.
fn maybe_print_prompt_cache_metrics(usage: &crate::ai::request::StreamUsage) {
    if !runtime_ctx::terminal_output_enabled() {
        return;
    }
    let show = crate::commonw::configw::get_all_config()
        .get(
            crate::ai::config_schema::AiConfig::PROMPT_CACHE_SHOW_METRICS,
            "true",
        )
        .trim()
        .eq_ignore_ascii_case("true");
    if !show {
        return;
    }
    let cached = usage
        .prompt_tokens_details
        .as_ref()
        .map(|d| d.cached_tokens)
        .unwrap_or(0);
    if let Some(line) = format_prompt_cache_metrics(usage.prompt_tokens, cached) {
        println!("  {ACCENT_MUTED}{line}{RESET}");
    }
}

/// Pure function: build a readable cache-hit line from prompt_tokens / cached_tokens.
/// Returns Some only when there really was a hit (cached > 0), to avoid pointless noise.
fn format_prompt_cache_metrics(prompt_tokens: u64, cached_tokens: u64) -> Option<String> {
    if cached_tokens == 0 || prompt_tokens == 0 {
        return None;
    }
    let pct = (cached_tokens as f64 / prompt_tokens as f64 * 100.0).min(100.0);
    Some(format!(
        "↳ cache · {}/{} tokens · {pct:.0}% hit",
        format_compact_token_count(cached_tokens),
        format_compact_token_count(prompt_tokens)
    ))
}

fn format_compact_token_count(tokens: u64) -> String {
    if tokens < 1_000 {
        tokens.to_string()
    } else if tokens < 1_000_000 {
        format!("{:.1}k", tokens as f64 / 1_000.0)
    } else {
        format!("{:.1}m", tokens as f64 / 1_000_000.0)
    }
}

async fn wait_for_interrupt(app: &App) {
    let _ = wait_for_interrupt_or_timeout(app, None).await;
}

fn stream_interrupt_requested(app: &App) -> bool {
    app.shutdown.load(std::sync::atomic::Ordering::Relaxed)
        || app.cancel_stream.load(std::sync::atomic::Ordering::Relaxed)
        || crate::ai::driver::signal::request_interrupt_ready()
}

async fn wait_for_interrupt_or_timeout(app: &App, delay: Option<Duration>) -> bool {
    if stream_interrupt_requested(app) {
        return true;
    }

    match delay {
        Some(delay) => {
            tokio::select! {
                _ = tokio::time::sleep(delay) => false,
                _ = crate::ai::driver::signal::wait_for_interrupt_sources(None, None, Some(app.cancel_stream.as_ref())) => true,
            }
        }
        None => {
            crate::ai::driver::signal::wait_for_interrupt_sources(
                None,
                None,
                Some(app.cancel_stream.as_ref()),
            )
            .await;
            true
        }
    }
}

async fn handle_stream_decode_error<E: std::fmt::Display>(
    app: &mut App,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    err: E,
) -> Option<StreamResult> {
    state.framing.decode_error_count += 1;
    if runtime_ctx::terminal_output_enabled() {
        let _ = clear_waiting_hint(state);
        eprintln!(
            "[Warning] 读取响应流时出错：{} (错误次数：{}/{})",
            err, state.framing.decode_error_count, MAX_DECODE_ERRORS
        );
    }

    if take_stream_cancelled(app) {
        return Some(cancelled_stream_result(state));
    }

    if state.framing.decode_error_count <= MAX_DECODE_ERRORS {
        if runtime_ctx::terminal_output_enabled() {
            eprintln!("[Warning] 尝试继续读取...");
        }
        if wait_for_interrupt_or_timeout(
            app,
            Some(Duration::from_millis(DECODE_ERROR_RETRY_DELAY_MS)),
        )
        .await
        {
            return Some(cancelled_stream_result(state));
        }
        return None;
    }

    if runtime_ctx::terminal_output_enabled() {
        eprintln!("[Error] 响应流读取失败，返回已收集的内容");
    }

    if runtime_ctx::terminal_output_enabled() {
        if state.content.thinking_open {
            let _ = flush_digest_filter_to_terminal(markers, state, true);
        }
        if state.content.thinking_open {
            let _ = write_stream_content(
                &format!("\n{}\n", markers.end_thinking_tag),
                &mut state.render.markdown,
                false,
            );
            print!("\x1b[0m");
            let _ = io::stdout().flush();
        }
        if state.render.subagent_fold.active {
            let _ = finalize_subagent_preview_fold(state);
        }
        let _ = flush_digest_filter_to_terminal(markers, state, false);
        let _ = flush_terminal_splitter(state, markers);
        let suppress_duplicate = final_assistant_matches_terminal_dedupe(state);
        let _ = disable_terminal_dedupe(state, suppress_duplicate);
        let _ = state.render.markdown.flush_pending();
    }

    let (tool_calls, dropped_malformed) =
        collect_valid_tool_calls(&mut state.content.tool_calls_map);
    // The decode-error fallback path is itself the product of a stream cut mid-way; if tool calls were also dropped,
    // mark it as truncation to trigger the upper-layer automatic retry instead of silently finishing.
    let outcome = if dropped_malformed {
        StreamOutcome::Truncated
    } else {
        StreamOutcome::Completed
    };
    let assistant_text = std::mem::take(&mut state.content.assistant_text);
    if runtime_ctx::terminal_output_enabled()
        && state.render.defer_assistant_body
        && outcome != StreamOutcome::Completed
    {
        let visible_text = crate::ai::request::strip_digest_blocks(&assistant_text);
        if !visible_text.trim().is_empty() {
            let _ = super::render_markdown_block(&visible_text);
        }
    }

    Some(StreamResult {
        outcome,
        tool_calls,
        assistant_text,
        hidden_meta: String::new(),
        reasoning_text: std::mem::take(&mut state.content.reasoning_text),
        reasoning_items: std::mem::take(&mut state.content.reasoning_items),
        skip_response_drain: true,
        truncated_by_length: false,
        // Truncation caused by a stream read failure, not by over-long model output.
        stream_error: true,
        finish_reason_value: state.content.finish_reason_value.clone(),
        usage_prompt_tokens: 0,
        usage_cached_prompt_tokens: 0,
        usage_completion_tokens: 0,
        usage_reasoning_tokens: 0,
    })
}

struct ToolCallRenderChunk {
    function_name: String,
    arguments: String,
    open_line: bool,
}

fn take_tool_call_render_chunk(
    current_printing_index: Option<usize>,
    index: usize,
    builder: &mut ToolCallBuilder,
) -> Option<ToolCallRenderChunk> {
    if builder.function_name.is_empty() {
        return None;
    }

    let start = builder.printed_arguments_len.min(builder.arguments.len());
    let arguments = builder.arguments[start..].to_string();
    builder.printed_arguments_len = builder.arguments.len();

    Some(ToolCallRenderChunk {
        function_name: builder.function_name.clone(),
        arguments,
        open_line: current_printing_index != Some(index),
    })
}

fn open_tool_call_line(
    state: &mut StreamProcessingState,
    index: usize,
    function_name: &str,
) -> io::Result<()> {
    state.render.current_printing_index = Some(index);
    if runtime_ctx::terminal_output_enabled() && io::stdout().is_terminal() {
        print_tool_call_waiting_hint(state, function_name)?;
    }
    Ok(())
}

/// The terminal does not print tool-call arguments; the streaming receive phase only shows an erasable
/// tool-name status line, while the actual execution lines are printed uniformly by the tool execution layer.
fn write_tool_call_arguments_stream(_arguments: &str) -> io::Result<()> {
    Ok(())
}

fn process_external_tool_calls_delta(
    app: &mut App,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    chunk: &StreamChunk,
    merge_mode: StreamEventMergeMode,
) -> bool {
    let Some(choice) = chunk.choices.first() else {
        return false;
    };

    let mut meaningful_progress = false;
    for stream_tool_call in &choice.delta.tool_calls {
        if stream_tool_call.id.is_empty()
            && stream_tool_call.tool_type.is_empty()
            && stream_tool_call.function.name.is_empty()
            && stream_tool_call.function.arguments.is_empty()
        {
            continue;
        }
        let index = match stream_tool_call.index {
            Some(idx) => idx,
            None => resolve_indexless_tool_call_key(&mut state.content, &stream_tool_call.id),
        };
        ensure_tool_calls_section_open(app, markers, state);

        let render_chunk = {
            let builder = state.content.tool_calls_map.entry(index).or_default();
            let before = (
                builder.id.len(),
                builder.tool_type.len(),
                builder.function_name.len(),
                builder.arguments.len(),
            );
            if !stream_tool_call.id.is_empty() {
                builder.id.clone_from(&stream_tool_call.id);
            }
            if !stream_tool_call.tool_type.is_empty() {
                builder.tool_type.clone_from(&stream_tool_call.tool_type);
            }
            if !stream_tool_call.function.name.is_empty() {
                builder
                    .function_name
                    .clone_from(&stream_tool_call.function.name);
            }
            append_tool_call_arguments(
                &mut builder.arguments,
                &stream_tool_call.function.arguments,
                merge_mode,
            );
            let after = (
                builder.id.len(),
                builder.tool_type.len(),
                builder.function_name.len(),
                builder.arguments.len(),
            );
            meaningful_progress |= after != before;
            take_tool_call_render_chunk(state.render.current_printing_index, index, builder)
        };

        if let Some(render_chunk) = render_chunk {
            if render_chunk.open_line {
                let _ = open_tool_call_line(state, index, &render_chunk.function_name);
            }
            let _ = write_tool_call_arguments_stream(&render_chunk.arguments);
        }
    }
    meaningful_progress
}

/// Incrementally resolves cumulative keys for chat-completions tool calls whose
/// `index` is missing. If everything fell onto the default key 0, parallel tool
/// calls would merge into one; instead group by id: reuse the key of an existing
/// builder with the same id, otherwise synthesize a stable key in
/// [10000, usize::MAX) from a hash of the id (real provider indexes are single
/// digits, so no collision). Parameter-continuation deltas with neither id nor
/// index attach to the most recent call without an index; if there is none, fall
/// back to the old-behavior key 0.
fn resolve_indexless_tool_call_key(state: &mut StreamContentState, id: &str) -> usize {
    if !id.is_empty() {
        for (key, builder) in state.tool_calls_map.iter() {
            if !builder.id.is_empty() && builder.id == id {
                return *key;
            }
        }
        let mut hash = 10000u64;
        for byte in id.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        let key = hash as usize;
        state.last_indexless_tool_call_key = Some(key);
        return key;
    }
    state.last_indexless_tool_call_key.unwrap_or(0)
}

fn append_tool_call_arguments(
    existing: &mut String,
    incoming: &str,
    merge_mode: StreamEventMergeMode,
) {
    if incoming.is_empty() {
        return;
    }

    match merge_mode {
        StreamEventMergeMode::Append => existing.push_str(incoming),
        StreamEventMergeMode::AppendMissingSuffix => {
            let suffix = unseen_suffix(existing, incoming);
            existing.push_str(&suffix);
        }
    }
}

/// Consume internal tool-call stream events. The return value indicates whether this batch detected a hallucinated
/// internal tool-protocol marker (`HallucinatedProtocolMarker`) — the caller then stops the stream and retries downgraded.
fn process_internal_tool_calls(
    app: &mut App,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    internal_tool_call_events: Vec<InternalToolCallStreamEvent>,
) -> (bool, bool) {
    let mut saw_hallucinated_marker = false;
    let mut meaningful_progress = false;
    for event in internal_tool_call_events {
        match event {
            InternalToolCallStreamEvent::Begin(function_name) => {
                if function_name.trim().is_empty() {
                    continue;
                }
                meaningful_progress = true;
                ensure_tool_calls_section_open(app, markers, state);

                let index = state.content.internal_tool_call_idx;
                let builder = state.content.tool_calls_map.entry(index).or_default();
                builder.id = format!("internal_{index}");
                builder.tool_type = "function".to_string();
                builder.function_name = function_name.clone();

                let _ = open_tool_call_line(state, index, &function_name);
            }
            InternalToolCallStreamEvent::Args(chunk) => {
                if chunk.is_empty() {
                    continue;
                }
                meaningful_progress = true;
                let index = state.content.internal_tool_call_idx;
                let builder = state.content.tool_calls_map.entry(index).or_default();
                if builder.function_name.is_empty() {
                    builder.id = format!("internal_{index}");
                    builder.tool_type = "function".to_string();
                }
                builder.arguments.push_str(&chunk);
                builder.printed_arguments_len = builder.arguments.len();

                let _ = write_tool_call_arguments_stream(&chunk);
            }
            InternalToolCallStreamEvent::End => {
                if state.render.current_printing_index == Some(state.content.internal_tool_call_idx)
                {
                    // The streaming phase no longer prints tool name/arguments (open_tool_call_line and
                    // write_tool_call_arguments_stream are no-ops), so only the color needs resetting here.
                    // Never use println! — it would insert a blank line between the `✓` and the following output
                    // (the external delta tool path never prints this line anyway).
                    if runtime_ctx::terminal_output_enabled() {
                        print!("\x1b[0m");
                    }
                    state.render.current_printing_index = None;
                    if runtime_ctx::terminal_output_enabled() {
                        let _ = io::stdout().flush();
                    }
                }
                state.content.internal_tool_call_idx += 1;
            }
            InternalToolCallStreamEvent::HallucinatedProtocolMarker => {
                // The streamer already strips the whole hallucinated "tool result" so nothing is shown; only the signal
                // is recorded here, and the caller stops the stream and takes the degenerate_repetition downgrade-retry path.
                saw_hallucinated_marker = true;
            }
        }
    }
    (saw_hallucinated_marker, meaningful_progress)
}

fn commit_visible_content(
    _app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    mut content: String,
) -> Result<(), Box<dyn std::error::Error>> {
    if content.is_empty() {
        return Ok(());
    }

    let render_terminal = runtime_ctx::terminal_output_enabled();
    if render_terminal {
        normalize_end_thinking_boundary(&mut content, markers, &state.render.markdown);
        if state.content.thinking_open || !state.render.defer_assistant_body {
            // Renderable content (thinking, or body with the deferral off): the
            // waiting/buffering hint is no longer needed.
            clear_waiting_hint(state)?;
        } else {
            // Deferred-body buffering: the final answer is withheld from the
            // terminal until the gates accept it. Keep a hint on its own line so
            // the terminal is never silently blank while the model generates.
            show_deferred_body_buffering_hint(state)?;
        }
    }

    // When thinking fold mode is active and an end_thinking_tag arrives, do the final fold render
    if render_terminal
        && !state.content.thinking_open
        && state.render.thinking_fold.active
        && is_standalone_stream_marker(&content, &markers.end_thinking_tag)
    {
        finalize_thinking_fold(state)?;
        // end_thinking_tag content is only a visual separator; it must not be appended to assistant_text
        let text = content.replace(&markers.end_thinking_tag, "");
        let text = text.trim_matches('\n');
        if !text.is_empty() {
            current_history.push_str(text);
            state.content.assistant_text.push_str(text);
        }
        return Ok(());
    }

    if render_terminal && (state.content.thinking_open || !state.render.defer_assistant_body) {
        // digest is extra image-understanding content meant for the model; strip it for terminal display (history/assistant_text keep the original)
        let terminal_content = state.render.digest_filter.push(&content);
        if markers.subagent_preview_enabled() {
            write_subagent_content_folded(terminal_content.as_str(), state)?;
        } else {
            maybe_write_stream_content(
                terminal_content.as_str(),
                state,
                markers,
                state.content.thinking_open,
            )?;
        }
    }
    if state.content.thinking_open {
        return Ok(());
    }

    let text = if is_standalone_stream_marker(&content, &markers.end_thinking_tag) {
        String::new()
    } else {
        content
    };
    current_history.reserve(text.len());
    state.content.assistant_text.reserve(text.len());
    current_history.push_str(&text);
    state.content.assistant_text.push_str(&text);

    Ok(())
}

pub(super) fn format_end_thinking_line(
    markers: &StreamMarkers,
    markdown: &MarkdownStreamRenderer,
) -> String {
    let mut content = format!("{}\n", markers.end_thinking_tag);
    normalize_end_thinking_boundary(&mut content, markers, markdown);
    content
}

fn normalize_end_thinking_boundary(
    content: &mut String,
    markers: &StreamMarkers,
    markdown: &MarkdownStreamRenderer,
) {
    if content.starts_with(&markers.end_thinking_tag) && markdown.has_unfinished_line() {
        content.insert(0, '\n');
    }
}

/// At stream end, flush the tail cached by `InlineMarkupNormalizer` (a complete marker whose closing was never
/// delivered, or plain text) through the same parse chain as streaming: recognized tool calls go into
/// tool_calls_map and remaining visible text is appended to assistant_text, so no half-cut marker or
/// buffered content is lost.
fn flush_inline_markup_normalizer(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
) -> Result<(), Box<dyn std::error::Error>> {
    let normalized = state.content.inline_markup_normalizer.flush();
    if normalized.is_empty() {
        return Ok(());
    }
    let (cleaned, mut tool_events) = state.content.hermes_tool_call_streamer.push(&normalized);
    let (cleaned, anthropic_events) = state.content.anthropic_tool_call_streamer.push(&cleaned);
    let (cleaned, bare_xml_events) = state.content.bare_xml_tool_call_streamer.push(&cleaned);
    tool_events.extend(anthropic_events);
    tool_events.extend(bare_xml_events);
    if !tool_events.is_empty() {
        // The stream is over during flush, so we cannot stop and retry; even if a hallucination marker were detected,
        // the streamer already stripped the whole block and cleaned contains no protocol markers, so the return value is ignored (no hallucinated body persisted).
        let _ = process_internal_tool_calls(app, markers, state, tool_events);
    }
    if !cleaned.is_empty() {
        let content = normalize_stream_text(cleaned);
        if !content.is_empty() {
            commit_visible_content(app, current_history, markers, state, content)?;
        }
    }
    Ok(())
}

fn flush_terminal_splitter(
    state: &mut StreamProcessingState,
    markers: &StreamMarkers,
) -> io::Result<()> {
    let marker_line = format!("{}\n", markers.end_thinking_tag);
    let marker_line_with_prefix = format!("\n{}\n", markers.end_thinking_tag);
    let segments = state
        .render
        .terminal_splitter
        .flush(&[marker_line_with_prefix.as_str(), marker_line.as_str()]);
    for segment in segments {
        write_stream_split_segment(segment, state)?;
    }
    Ok(())
}

fn write_stream_split_segment(
    segment: StreamSplitSegment,
    state: &mut StreamProcessingState,
) -> io::Result<()> {
    match segment {
        StreamSplitSegment::Text(text) => maybe_write_plain_stream_text(&text, state, false),
        StreamSplitSegment::Marker {
            marker_index: _,
            text,
        } => {
            let suppress_duplicate = terminal_dedupe_buffer_is_complete_match(state);
            disable_terminal_dedupe(state, suppress_duplicate)?;
            write_stream_content_to_terminal(&text, &mut state.render.markdown, false)
        }
    }
}

fn maybe_write_plain_stream_text(
    content: &str,
    state: &mut StreamProcessingState,
    dimmed: bool,
) -> io::Result<()> {
    if content.is_empty() {
        return Ok(());
    }
    if dimmed {
        return write_stream_content_to_terminal(content, &mut state.render.markdown, true);
    }

    if let Some(dedupe) = state.render.terminal_dedupe.as_mut() {
        dedupe.buffered_terminal_output.push_str(content);
        if terminal_dedupe_still_matches(state) {
            return Ok(());
        }
        disable_terminal_dedupe(state, false)?;
        return Ok(());
    }

    write_stream_content_to_terminal(content, &mut state.render.markdown, false)
}

fn terminal_dedupe_still_matches(state: &StreamProcessingState) -> bool {
    let Some(dedupe) = state.render.terminal_dedupe.as_ref() else {
        return false;
    };
    let buffered = dedupe.buffered_terminal_output.as_str();
    let candidate = dedupe.candidate.as_str();
    candidate.starts_with(buffered)
        || (buffered.starts_with(candidate) && buffered[candidate.len()..].trim().is_empty())
}

fn terminal_dedupe_buffer_is_complete_match(state: &StreamProcessingState) -> bool {
    state
        .render
        .terminal_dedupe
        .as_ref()
        .is_some_and(|dedupe| dedupe.buffered_terminal_output.trim() == dedupe.candidate.trim())
}

fn final_assistant_matches_terminal_dedupe(state: &StreamProcessingState) -> bool {
    state.render.terminal_dedupe.as_ref().is_some_and(|dedupe| {
        crate::ai::request::strip_digest_blocks(&state.content.assistant_text).trim()
            == dedupe.candidate.trim()
    })
}

fn disable_terminal_dedupe(
    state: &mut StreamProcessingState,
    suppress_buffered: bool,
) -> io::Result<()> {
    let Some(dedupe) = state.render.terminal_dedupe.take() else {
        return Ok(());
    };
    if !suppress_buffered && !dedupe.buffered_terminal_output.is_empty() {
        write_stream_content_to_terminal(
            &dedupe.buffered_terminal_output,
            &mut state.render.markdown,
            false,
        )?;
    }
    Ok(())
}

fn is_standalone_stream_marker(content: &str, marker: &str) -> bool {
    content.trim_matches('\n') == marker
}

/// Folded rendering of thinking content: maintain a rewritable window starting at the first line,
/// always showing only the most recent N lines in the terminal and folding the rest into one summary line.
fn write_thinking_content_folded(
    content: &str,
    state: &mut StreamProcessingState,
    markers: &StreamMarkers,
) -> io::Result<()> {
    if content.is_empty() {
        return Ok(());
    }
    let fold = &mut state.render.thinking_fold;

    if fold.max_visible_lines == usize::MAX {
        return write_stream_content_to_terminal(content, &mut state.render.markdown, true);
    }

    // Control lines must be exclusive markers; they must not be interleaved with body text.
    if is_standalone_stream_marker(content, &markers.thinking_tag) {
        if !fold.active {
            if state.render.markdown.has_unfinished_line() {
                write_stream_content_to_terminal("\n", &mut state.render.markdown, false)?;
            }
            fold.active = true;
        }
        return thinking_fold_redraw(fold);
    }

    if !fold.active {
        return write_stream_content_to_terminal(content, &mut state.render.markdown, true);
    }

    append_fold_content(fold, content);

    thinking_fold_redraw(fold)
}

fn write_subagent_content_folded(
    content: &str,
    state: &mut StreamProcessingState,
) -> io::Result<()> {
    if content.is_empty() {
        return Ok(());
    }

    let fold = &mut state.render.subagent_fold;
    if fold.max_visible_lines == usize::MAX {
        return write_stream_content_to_terminal(content, &mut state.render.markdown, false);
    }

    if !fold.active {
        if state.render.markdown.has_unfinished_line() {
            write_stream_content_to_terminal("\n", &mut state.render.markdown, false)?;
        }
        fold.active = true;
    }

    append_fold_content(fold, content);
    thinking_fold_redraw(fold)
}

fn append_fold_content(fold: &mut super::state::ThinkingFoldState, content: &str) {
    for ch in content.chars() {
        if ch == '\n' {
            let completed_line = std::mem::take(&mut fold.current_line);
            if fold.skip_blank_lines && completed_line.trim().is_empty() {
                continue;
            }
            fold.total_lines += 1;
            fold.recent_lines.push_back(completed_line);
            while fold.recent_lines.len() > fold.max_visible_lines {
                fold.recent_lines.pop_front();
            }
        } else {
            fold.current_line.push(ch);
        }
    }
}

/// Covers only the thinking body window (fold summary + recent visible lines); the header is not included.
///
/// The header (`○`) is printed once when folding activates and stays anchored above the body; every redraw after that only
/// erases and rewrites the body. The body shows at most `max_visible_lines` physical content lines plus one fold summary, and stays
/// within the visible viewport, so relative erases always reach it and it never scrolls out of sync into the
/// scrollback — and even if it did, no second header could be created, eliminating "orphan header stacking" at the root.
fn thinking_fold_redraw(fold: &mut super::state::ThinkingFoldState) -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    // Only erase the previous body region; the header is anchored above it and must never be touched.
    // Note: after the terminal narrows, the terminal auto-reflows old body into more physical lines at the current width;
    // so do not trust the cached window_rows — recompute how many rows the last body would take at the current width.
    let erase_rows = thinking_fold_rendered_body_rows(fold).max(fold.window_rows);
    erase_fold_body(&mut out, erase_rows)?;
    if fold.active && !fold.header_drawn {
        write_fold_header(&mut out, fold)?;
        fold.header_drawn = true;
    }

    let (body_lines, marker_lines) = thinking_fold_window_lines(fold);
    THINKING_FOLD_BODY_BUF.with(|buf| -> io::Result<()> {
        let mut buf = buf.borrow_mut();
        let (body_rows, rendered_body_lines) = render_thinking_fold_window_lines(
            &body_lines,
            marker_lines,
            fold.rewrite_right_margin_cols,
            fold.max_visible_lines,
            &mut buf,
        );
        if !buf.is_empty() {
            out.write_all(buf.as_bytes())?;
        }
        fold.window_rows = body_rows;
        fold.rendered_body_lines = rendered_body_lines;
        Ok(())
    })?;
    out.flush()?;
    Ok(())
}

/// Print the anchored fold header. Should be called only once after folding activates.
fn write_fold_header(
    out: &mut impl Write,
    fold: &super::state::ThinkingFoldState,
) -> io::Result<()> {
    write!(out, "  {ACCENT_MUTED}{}\x1b[0m\r\n", fold.header_label)
}

/// Write the final header directly when thinking ends; used for an empty fold that never wrote an in-progress header.
fn write_thinking_fold_completion_header(
    out: &mut impl Write,
    fold: &super::state::ThinkingFoldState,
    line_count: usize,
) -> io::Result<()> {
    write!(
        out,
        "  {ACCENT_MUTED}{} · {line_count} lines\x1b[0m\r\n",
        fold.footer_label,
    )
}

/// Rewrite the anchored `○ thinking` in place to the completed state instead of printing a separate `✓ thinking` below the body.
///
/// Before this, `erase_fold_body` moved the cursor back to the first body line under the header; so move up one line,
/// clear the header and rewrite it. The folded body is still drawn below the new header by the usual logic.
fn replace_thinking_fold_header(
    out: &mut impl Write,
    fold: &super::state::ThinkingFoldState,
    line_count: usize,
) -> io::Result<()> {
    write!(out, "\r\x1b[1A\r\x1b[2K")?;
    write_thinking_fold_completion_header(out, fold, line_count)
}

/// After rendering the body the cursor rests on the last physical line, not on an extra blank line; redraws therefore only need to move up
/// `rows - 1`; returning to the line start before erasing to the screen bottom covers reflow lines produced by a narrowed window.
fn erase_fold_body(out: &mut impl Write, rows: usize) -> io::Result<()> {
    if rows == 0 {
        return Ok(());
    }
    write!(out, "\r")?;
    if rows > 1 {
        write!(out, "\x1b[{}A", rows - 1)?;
    }
    // CSI 0J cannot be used: it clears from the first body line to the end of the physical screen, crossing the DECSTBM scroll
    // region and wiping out the side-note composer at the bottom. Clear only the rendered body window line by line, then restore
    // the cursor to the first body line so relative-cursor semantics of later redraws stay unchanged.
    for row in 0..rows {
        write!(out, "\r\x1b[2K")?;
        if row + 1 < rows {
            write!(out, "\x1b[1B")?;
        }
    }
    if rows > 1 {
        write!(out, "\x1b[{}A", rows - 1)?;
    }
    write!(out, "\r")
}

/// Final rendering when thinking ends: overwrite the body window and turn the anchored `○` into `✓` in place.
pub(super) fn finalize_thinking_fold(state: &mut StreamProcessingState) -> io::Result<()> {
    finalize_fold(&mut state.render.thinking_fold, true)
}

fn finalize_subagent_preview_fold(state: &mut StreamProcessingState) -> io::Result<()> {
    // Subagent preview keeps the window body at the end: the subagent's final output should stay visible in the terminal,
    // so the "forced 0-line pure summary" ending used by thinking (which would fold the key conclusions too) is not applied.
    finalize_fold(&mut state.render.subagent_fold, false)
}

fn finalize_fold(
    fold: &mut super::state::ThinkingFoldState,
    collapse_body: bool,
) -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    finalize_fold_to(&mut out, fold, collapse_body)
}

/// Fold-finalize writer implementation; extracted behind a writer so regression tests can verify terminal cursor sequences precisely.
fn finalize_fold_to(
    mut out: &mut impl Write,
    fold: &mut super::state::ThinkingFoldState,
    collapse_body: bool,
) -> io::Result<()> {
    if !fold.active {
        return Ok(());
    }

    let erase_rows = thinking_fold_rendered_body_rows(fold).max(fold.window_rows);
    erase_fold_body(&mut out, erase_rows)?;
    // The thinking completed state replaces the in-progress header instead of printing another footer below the body.
    // If the fold never actually landed, write the completed state directly to avoid a brief `○ thinking`.
    let line_count = fold
        .total_lines
        .saturating_add(usize::from(!fold.current_line.is_empty()));
    if collapse_body {
        if fold.header_drawn {
            replace_thinking_fold_header(&mut out, fold, line_count)?;
        } else {
            write_thinking_fold_completion_header(&mut out, fold, line_count)?;
            fold.header_drawn = true;
        }
    } else if !fold.header_drawn {
        // Subagent preview keeps the existing two-line header/footer layout.
        write_fold_header(&mut out, fold)?;
        fold.header_drawn = true;
    }

    // Thinking finalize folds to a pure summary (0-line window) with no body lines: the tail of thinking often restates
    // conclusions/questions, and keeping visible lines would duplicate the final answer that follows in the terminal. Subagent
    // preview does not fold — it keeps the recent visible window lines so the subagent's final output stays visible.
    let saved_max_visible_lines = fold.max_visible_lines;
    if collapse_body {
        fold.max_visible_lines = 0;
    }
    let (body_lines, marker_lines) = thinking_fold_window_lines(fold);
    let max_visible_rows = fold.max_visible_lines;
    let mut final_body_rows = 0usize;
    THINKING_FOLD_BODY_BUF.with(|buf| -> io::Result<()> {
        let mut buf = buf.borrow_mut();
        let (body_rows, rendered_body_lines) = render_thinking_fold_window_lines(
            &body_lines,
            marker_lines,
            fold.rewrite_right_margin_cols,
            max_visible_rows,
            &mut buf,
        );
        fold.max_visible_lines = saved_max_visible_lines;
        if !buf.is_empty() {
            out.write_all(buf.as_bytes())?;
        }
        fold.window_rows = body_rows;
        fold.rendered_body_lines = rendered_body_lines;
        final_body_rows = body_rows;
        Ok(())
    })?;

    if !collapse_body {
        // Subagent preview keeps its footer; thinking's scale info was already written into the in-place-replaced header.
        if final_body_rows > 0 {
            out.write_all(b"\r\n")?;
        }
        write!(
            out,
            "  {ACCENT_MUTED}{} · {line_count} lines\x1b[0m\r\n",
            fold.footer_label,
        )?;
    }
    out.flush()?;

    // Reset fold state
    fold.reset();
    Ok(())
}

fn thinking_fold_hidden_count(fold: &super::state::ThinkingFoldState) -> usize {
    let current_line = usize::from(!fold.current_line.is_empty());
    fold.total_lines
        .saturating_add(current_line)
        .saturating_sub(fold.max_visible_lines)
}

fn thinking_fold_visible_lines(fold: &super::state::ThinkingFoldState) -> Vec<&str> {
    // 0-line window = pure summary mode: even the current incomplete line is hidden, so restated conclusions cannot leak to the terminal.
    if fold.max_visible_lines == 0 {
        return Vec::new();
    }
    let current_line = usize::from(!fold.current_line.is_empty());
    let visible_completed = fold.max_visible_lines.saturating_sub(current_line);
    let completed_skip = fold.recent_lines.len().saturating_sub(visible_completed);
    let mut visible = fold
        .recent_lines
        .iter()
        .skip(completed_skip)
        .map(String::as_str)
        .collect::<Vec<_>>();
    if current_line > 0 {
        visible.push(fold.current_line.as_str());
    }
    visible
}

fn thinking_fold_rendered_body_rows(fold: &super::state::ThinkingFoldState) -> usize {
    fold.rendered_body_lines
        .iter()
        .map(|line| live_preview_cursor_rows(line))
        .sum()
}

fn thinking_fold_window_lines(fold: &super::state::ThinkingFoldState) -> (Vec<String>, usize) {
    let hidden_count = thinking_fold_hidden_count(fold);
    let visible_lines = thinking_fold_visible_lines(fold);
    if hidden_count == 0 && visible_lines.is_empty() {
        return (Vec::new(), 0);
    }

    let mut lines = Vec::with_capacity(visible_lines.len() + usize::from(hidden_count > 0));
    let marker_lines = usize::from(hidden_count > 0);
    if hidden_count > 0 {
        lines.push(format!("… {hidden_count} earlier lines"));
    }
    for line in visible_lines {
        lines.push(line.to_string());
    }
    (lines, marker_lines)
}

/// Render the **body** of the fold window (fold summary + recent visible lines), without the header.
/// The header is anchored and printed separately by `write_fold_header`. Returns the number of physical body lines; the body does not
/// end with a newline and the cursor always stays on the last line, so xterm.js does not interpret a trailing LF as extra scrolling.
fn render_thinking_fold_window(fold: &super::state::ThinkingFoldState) -> (String, usize) {
    let (lines, marker_lines) = thinking_fold_window_lines(fold);
    THINKING_FOLD_BODY_BUF.with(|buf| {
        let mut buf = buf.borrow_mut();
        let (rows, _) = render_thinking_fold_window_lines(
            &lines,
            marker_lines,
            fold.rewrite_right_margin_cols,
            fold.max_visible_lines,
            &mut buf,
        );
        (buf.clone(), rows)
    })
}

// Scratch buffer reused across thinking-fold body renders so each redraw does not
// rebuild a zero-capacity String. Purely an allocation-reuse optimization: the
// renderer clears the buffer before every rebuild, so the bytes handed to the
// terminal are identical to building a fresh String each time.
thread_local! {
    static THINKING_FOLD_BODY_BUF: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Render the **body** of the fold window (fold summary + recent visible lines), without the header.
/// The header is anchored and printed separately by `write_fold_header`. Writes the body into
/// `out` (cleared first) and returns the number of physical body lines plus the plain-text rows
/// kept for later width recomputation; the body does not end with a newline and the cursor always
/// stays on the last line, so xterm.js does not interpret a trailing LF as extra scrolling.
fn render_thinking_fold_window_lines(
    lines: &[String],
    marker_lines: usize,
    rewrite_right_margin_cols: usize,
    max_visible_rows: usize,
    out: &mut String,
) -> (usize, Vec<String>) {
    out.clear();
    if lines.is_empty() {
        return (0, Vec::new());
    }

    let reserve_cols = THINKING_FOLD_BODY_INDENT_WIDTH + rewrite_right_margin_cols;
    let marker_lines = marker_lines.min(lines.len());
    let mut wrapped_content_rows = Vec::new();
    for line in lines.iter().skip(marker_lines) {
        wrapped_content_rows.extend(wrap_line_to_terminal_rows_with_reserve(line, reserve_cols));
    }
    let hidden_wrapped_rows = wrapped_content_rows.len().saturating_sub(max_visible_rows);
    let marker = if hidden_wrapped_rows > 0 {
        // When truncated at a physical line, the first hidden content may come from a still-streaming logical line, so the
        // imprecise "earlier lines" count can no longer be reported.
        Some("… more".to_string())
    } else if marker_lines > 0 {
        Some(lines[0].clone())
    } else {
        None
    };
    let mut rows_to_render = Vec::with_capacity(
        wrapped_content_rows
            .len()
            .saturating_sub(hidden_wrapped_rows)
            .saturating_add(usize::from(marker.is_some())),
    );
    if let Some(marker) = marker {
        // The fold hint must always occupy exactly one physical line; only the body is allowed to wrap.
        rows_to_render.push((
            clamp_line_to_terminal_row_with_reserve(&marker, reserve_cols),
            true,
        ));
    }
    rows_to_render.extend(
        wrapped_content_rows
            .into_iter()
            .skip(hidden_wrapped_rows)
            .map(|line| (line, false)),
    );

    let mut rendered_lines = Vec::with_capacity(rows_to_render.len());
    // Folded body has fixed indentation. The body keeps at most max_visible_rows wrapped physical lines; if more
    // content must be hidden, the single-line fold hint does not count against the body budget. Each wrapped segment
    // occupies exactly one physical line; extra right margin for the xterm.js integrated terminal.
    let mut rows = 0usize;
    let mut first_rendered_row = true;

    for (wrapped_row, is_marker) in rows_to_render {
        if !first_rendered_row {
            out.push_str("\r\n");
        }
        first_rendered_row = false;
        let rendered_line = format!("{THINKING_FOLD_BODY_INDENT}{wrapped_row}");
        rows += 1;
        if is_marker {
            out.push_str(ACCENT_MUTED);
            out.push_str(&rendered_line);
            out.push_str("\x1b[0m");
        } else {
            out.push_str(DIM);
            out.push_str(&rendered_line);
            out.push_str(RESET);
        }
        rendered_lines.push(rendered_line);
    }

    (rows, rendered_lines)
}

fn maybe_write_stream_content(
    content: &str,
    state: &mut StreamProcessingState,
    markers: &StreamMarkers,
    dimmed: bool,
) -> io::Result<()> {
    if dimmed {
        return write_thinking_content_folded(content, state, markers);
    }

    let marker_line = format!("{}\n", markers.end_thinking_tag);
    let marker_line_with_prefix = format!("\n{}\n", markers.end_thinking_tag);
    let segments = state.render.terminal_splitter.push(
        content,
        &[marker_line_with_prefix.as_str(), marker_line.as_str()],
    );
    for segment in segments {
        write_stream_split_segment(segment, state)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;

fn process_stream_payload(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter: &'static dyn ProviderAdapter,
    event_type: Option<&str>,
    payload: &str,
) -> Result<StreamPayloadOutcome, Box<dyn std::error::Error>> {
    let (mut chunk, merge_mode, is_replayed) =
        match normalize::parse_stream_payload(adapter, payload, event_type) {
            super::state::ParsedStreamPayload::Ignore => {
                return Ok(StreamPayloadOutcome::default());
            }
            super::state::ParsedStreamPayload::Done => return Ok(StreamPayloadOutcome::stop()),
            super::state::ParsedStreamPayload::Error(msg) => {
                return Err(format!("provider stream error: {msg}").into());
            }
            super::state::ParsedStreamPayload::ReasoningItem(item) => {
                // Capture the full reasoning item (incl. encrypted_content) for same-turn tool-chain replay.
                // Produces no visible output and never enters persisted history. Models like Spark may emit multiple
                // reasoning segments (different ids) before one tool_call; all must be kept. The gateway re-sends the same
                // reasoning resource in .added (partial payload) and .done (full payload) with identical ids but
                // different content, so whole-field equality dedup would judge them unequal; converge by id and keep the
                // later one (.done always comes after .added and is the protocol's authoritative final state), otherwise the same
                // resource id appears twice during replay and modelhub returns 400 (-4003).
                state.content.reasoning_items.push(item);
                crate::ai::history::compress::dedup_reasoning_items_by_id(
                    &mut state.content.reasoning_items,
                );
                return Ok(StreamPayloadOutcome::default());
            }
            super::state::ParsedStreamPayload::Chunk(chunk) => {
                (chunk, StreamEventMergeMode::Append, false)
            }
            super::state::ParsedStreamPayload::ReplayedChunk(chunk) => {
                (chunk, StreamEventMergeMode::Append, true)
            }
            super::state::ParsedStreamPayload::SnapshotChunk(chunk) => {
                (chunk, StreamEventMergeMode::AppendMissingSuffix, false)
            }
        };

    // AIOS: capture usage block from whichever chunk carries it. OpenAI emits
    // the final `usage` on a chunk with `choices: []`, so we must pull it *before*
    // the empty-choices early return below.
    if let Some(ref usage) = chunk.usage {
        state.pending_llm_usage = Some((chunk.model.clone(), usage.clone().normalized()));
    }

    let saw_finish_reason = chunk.choices.iter().any(|choice| {
        choice
            .finish_reason
            .as_deref()
            .is_some_and(|reason| !reason.trim().is_empty())
    });
    if saw_finish_reason {
        state.content.finish_reason_seen = true;
    }

    // Record the most recent non-empty finish_reason value. `length` means the server cut output at the limit,
    // the key signal for escalating this turn to retryable Truncated.
    if let Some(reason) = chunk.choices.iter().find_map(|choice| {
        choice
            .finish_reason
            .as_deref()
            .map(str::trim)
            .filter(|reason| !reason.is_empty())
    }) {
        state.content.finish_reason_value = Some(reason.to_string());
    }

    if chunk.choices.is_empty() {
        state.content.empty_choice_chunks += 1;
        if should_show_waiting_hint(app) && state.content.empty_choice_chunks >= 3 {
            let _ = upgrade_waiting_hint_for_buffering(state);
        }
        return Ok(StreamPayloadOutcome {
            should_stop: false,
            meaningful_progress: saw_finish_reason,
        });
    }

    state.content.empty_choice_chunks = 0;

    // content_part.added / output_text.done may re-send already-present body overlapping the output_text.delta
    // increments: compute unseen suffixes from the **raw content-channel text before demux**. Deduping only by
    // assistant_text fails once demux is closed: a re-sent `reasoning`answer` prefix no longer matches the visible
    // `answer` body, and reasoning would leak into the body again.
    let mut content_channel_progress = false;
    if let Some(choice) = chunk.choices.first_mut()
        && !choice.delta.content.is_empty()
    {
        if is_replayed || matches!(merge_mode, StreamEventMergeMode::AppendMissingSuffix) {
            choice.delta.content =
                unseen_suffix(&state.content.content_replay_text, &choice.delta.content);
        }
        if !choice.delta.content.is_empty() {
            state
                .content
                .content_replay_text
                .push_str(&choice.delta.content);
            content_channel_progress = true;
        }
    }

    // Prefilled `thinking` template splitting: such reasoners write the chain into the content channel and only close with a dangling
    // `response`. While capturing, withhold — content is buffered in the splitter and not emitted to any channel
    // incrementally until `response` arrives, when the whole prefix is attributed at once to delta.reasoning_content (reusing
    // the existing thinking fold & accumulation paths) and the body stays in content. If `response` never arrives,
    // flush falls back the whole segment as content safely. Models that never arm it pass through with zero impact. Snapshots (.done)
    // were deduped above on the raw content channel, so only unseen suffixes enter the stateful splitter and nothing is
    // double-counted. content_channel_progress refreshes the idle timer, so a long withheld chain is not misjudged as
    // first-packet/idle timeout.
    if let Some(choice) = chunk.choices.first_mut()
        && !choice.delta.content.is_empty()
    {
        let (reasoning, content) = state
            .content
            .content_think_demuxer
            .push(&choice.delta.content);
        choice.delta.content = content;
        if !reasoning.is_empty() {
            // Consistent with existing semantics: reasoning_content can concatenate with prior reasoning fragments.
            choice.delta.reasoning_content =
                merge_reasoning_fragments(&choice.delta.reasoning_content, &reasoning);
        }
    }

    // reasoning_content dedup: the Responses API re-sends the same reasoning summary through multiple event paths
    // (reasoning_summary_text.{delta,done} and content_part.{added,done} carry identical summary_text). Previously only
    // SnapshotChunk (.done) got unseen-suffix dedup; Append-mode (.delta/.added) reasoning_content was not deduped, causing
    // duplicated thinking across event paths. Here both modes compute unseen suffixes, so rendering outputs only the new part.
    // When accumulating into reasoning_text, distinguish modes: Append accumulates the original text to keep degeneration
    // detection of model repetition loops (has_degenerate_reasoning_repetition depends on consecutive repeats); Snapshot
    // accumulates the deduped suffix (a snapshot re-sends already-seen text whose original was accumulated during Append).
    let original_reasoning = chunk
        .choices
        .first()
        .map(|c| c.delta.reasoning_content.clone())
        .unwrap_or_default();
    let emitted_reasoning = match merge_mode {
        StreamEventMergeMode::Append => original_reasoning.clone(),
        StreamEventMergeMode::AppendMissingSuffix => {
            unseen_suffix(&state.content.reasoning_text, &original_reasoning)
        }
    };
    let reasoning_progress = !emitted_reasoning.is_empty();

    if !original_reasoning.is_empty() {
        state.content.saw_reasoning_output = true;
        state.content.reasoning_text.push_str(&emitted_reasoning);

        // Some long tool-chain contexts make the model verbatim-repeat one sentence in thinking. Continuing to read only
        // burns the output budget and makes the terminal look stuck; escalate to retryable truncation and let the upper layer lower the reasoning tier.
        if has_degenerate_repetition(&state.content.reasoning_text) {
            state.content.finish_reason_seen = true;
            state.content.finish_reason_value =
                Some(DEGENERATE_REPETITION_FINISH_REASON.to_string());
            if runtime_ctx::terminal_output_enabled() {
                eprintln!("\n  ⚠ 检测到模型推理重复循环，停止当前响应并自动重试…");
            }
            return Ok(StreamPayloadOutcome::stop_with_progress());
        }
    }

    let recovered_inline_events =
        recover_protocol_only_inline_tool_call_snapshot(&mut chunk, merge_mode, state);

    // Incremental events keep the model's original text; snapshot events only render unseen suffixes to avoid protocol re-sends.
    if let Some(choice) = chunk.choices.first_mut() {
        choice.delta.reasoning_content = emitted_reasoning;
    }

    let external_tool_progress =
        process_external_tool_calls_delta(app, markers, state, &chunk, merge_mode);

    let (events, mut internal_tool_call_events) = extract_chunk_events_streaming(
        &chunk,
        markers.hidden_begin,
        markers.hidden_end,
        &mut state.content.thinking_open,
        &mut state.content.hidden_meta_parse,
        &mut state.content.internal_tool_call_streamer,
        &mut state.content.hermes_tool_call_streamer,
        &mut state.content.anthropic_tool_call_streamer,
        &mut state.content.bare_xml_tool_call_streamer,
        &mut state.content.inline_markup_normalizer,
    );
    internal_tool_call_events.extend(recovered_inline_events);
    let (saw_hallucinated_marker, internal_tool_progress) =
        process_internal_tool_calls(app, markers, state, internal_tool_call_events);
    let mut meaningful_progress = content_channel_progress
        || reasoning_progress
        || saw_finish_reason
        || external_tool_progress
        || internal_tool_progress;
    if saw_hallucinated_marker {
        // The model acts out "tool call → tool result" in visible body, emitting internal protocol markers that the system
        // never generates (`<function_results>` etc.). The streamer already strips the whole block; here we stop the stream and
        // reuse the degenerate_repetition downgrade-retry path so hallucinated body is never persisted to poison the next request. This is a
        // zero-false-positive signal: legitimate repeated code/wording never contains internal protocol markers, so no text-statistical threshold is needed.
        state.content.finish_reason_seen = true;
        state.content.finish_reason_value = Some(DEGENERATE_REPETITION_FINISH_REASON.to_string());
        if runtime_ctx::terminal_output_enabled() {
            eprintln!("\n  ⚠ 检测到模型伪造工具结果标记（输出退化），停止当前响应并自动重试…");
        }
        return Ok(StreamPayloadOutcome::stop_with_progress());
    }

    // Total tool-argument cap as a backstop: after the model opens a tool call it should close quickly (id/name/few args).
    // If the argument stream keeps growing until it passes MAX_TOOL_ARG_BYTES, the model is looping forever emitting arguments
    // (the incident was apply_patch arguments streaming for 20+ minutes). Such degeneration has no text-repetition signature
    // (new content every time), so repetition detection cannot catch it; only this total cap can. On hit, stop the stream and
    // reuse the degenerate_repetition downgrade-retry path instead of waiting forever for End/finish_reason.
    let tool_arg_bytes = state
        .content
        .tool_calls_map
        .iter()
        .map(|(_index, builder)| builder.arguments.len())
        .sum::<usize>();
    if tool_arg_bytes > MAX_TOOL_ARG_BYTES {
        state.content.finish_reason_seen = true;
        state.content.finish_reason_value = Some(DEGENERATE_REPETITION_FINISH_REASON.to_string());
        // The stream was cut off: the model may still be generating arguments, so even JSON that happens to be valid at
        // the cut instant must not run as a complete tool call (same principle as stream_idle_timed_out).
        state.content.tool_args_cap_exceeded = true;
        if runtime_ctx::terminal_output_enabled() {
            eprintln!(
                "\n  ⚠ 工具调用参数累积超过上限（{tool_arg_bytes} 字节 > {MAX_TOOL_ARG_BYTES}），判定输出退化，停止当前响应并自动重试…"
            );
        }
        return Ok(StreamPayloadOutcome::stop_with_progress());
    }

    if events.is_empty() {
        return Ok(StreamPayloadOutcome {
            should_stop: false,
            meaningful_progress,
        });
    }
    for event in events {
        match event {
            StreamTextEvent::AppendHiddenMeta(text) => {
                state.content.hidden_meta.push_str(&text);
            }
            StreamTextEvent::OpenThinking
            | StreamTextEvent::AppendThinking(_)
            | StreamTextEvent::CloseThinking => {
                render_thinking_event(markers, state, &event)?;
            }
            other => {
                let Some(content) = stream_text_event_to_content(
                    &other,
                    markers,
                    merge_mode,
                    &state.content.assistant_text,
                ) else {
                    continue;
                };
                if content.is_empty() {
                    continue;
                }
                // Step 6: pluggable stream filters (`state.filters`; the port lives in ports/stream.rs).
                // An empty chain passes through (zero behavior change); None = drop this chunk's visible text, keep it out of
                // assistant_text / the terminal, and out of the degeneration-repeat detection below.
                let Some(content) = state.filters.apply(content) else {
                    continue;
                };
                if content.is_empty() {
                    continue;
                }
                let assistant_len_before = state.content.assistant_text.len();
                commit_visible_content(app, current_history, markers, state, content)?;
                meaningful_progress |= state.content.assistant_text.len() > assistant_len_before;

                // Symmetric to the reasoning path: the model can also degenerate into verbatim repetition of one phrase in the
                // **visible output** (the incident was assistant content repeating one phrase verbatim until it filled the
                // output budget, producing 160k chars of junk that got persisted and poisoned the next request, triggering a provider
                // 400 InvalidParameter). Previously the degeneration guard only hung on reasoning_content, leaving visible
                // text completely unguarded. On hit, set finish_reason and stop the stream; the upper layer retries downgraded.
                if has_degenerate_repetition(&state.content.assistant_text) {
                    state.content.finish_reason_seen = true;
                    state.content.finish_reason_value =
                        Some(DEGENERATE_REPETITION_FINISH_REASON.to_string());
                    if runtime_ctx::terminal_output_enabled() {
                        eprintln!("\n  ⚠ 检测到模型输出重复循环，停止当前响应并自动重试…");
                    }
                    return Ok(StreamPayloadOutcome::stop_with_progress());
                }
            }
        }
    }

    // Keep streaming until explicit stream end ([DONE]/EOF) or the outer loop's
    // short post-finish grace window expires. Some providers can set
    // finish_reason before all visible content chunks are delivered.
    Ok(StreamPayloadOutcome {
        should_stop: false,
        meaningful_progress,
    })
}

/// Compatible gateways like OpenCode sometimes return the complete DSML tool protocol as a single content snapshot.
/// Recognize such a full wrapper before body submission and terminal rendering, so we do not only recover via a stream-end fallback.
fn recover_protocol_only_inline_tool_call_snapshot(
    chunk: &mut StreamChunk,
    merge_mode: StreamEventMergeMode,
    state: &StreamProcessingState,
) -> Vec<InternalToolCallStreamEvent> {
    let Some(choice) = chunk.choices.first_mut() else {
        return Vec::new();
    };
    if !choice.delta.tool_calls.is_empty() || choice.delta.content.trim().is_empty() {
        return Vec::new();
    }

    let normalized = normalize_inline_tool_call_markup(&choice.delta.content);
    let normalized = normalized.trim();
    if !normalized.starts_with("<tool_calls>") || !normalized.ends_with("</tool_calls>") {
        return Vec::new();
    }
    let Some(mut tool_calls) = recover_inline_tool_calls(normalized) else {
        return Vec::new();
    };

    choice.delta.content.clear();
    if matches!(merge_mode, StreamEventMergeMode::AppendMissingSuffix) {
        // A `.done` snapshot re-sends the full protocol already parsed from earlier deltas; filter only semantically equal calls,
        // so genuinely new parallel calls are not swallowed just because other tool calls already exist.
        tool_calls.retain(|tool_call| {
            !state
                .content
                .tool_calls_map
                .iter()
                .any(|(_, builder)| collected_tool_call_matches(builder, tool_call))
        });
    }

    let mut events = Vec::with_capacity(tool_calls.len().saturating_mul(3));
    for tool_call in tool_calls {
        events.push(InternalToolCallStreamEvent::Begin(tool_call.function.name));
        events.push(InternalToolCallStreamEvent::Args(
            tool_call.function.arguments,
        ));
        events.push(InternalToolCallStreamEvent::End);
    }
    events
}

fn collected_tool_call_matches(
    builder: &ToolCallBuilder,
    tool_call: &crate::ai::types::ToolCall,
) -> bool {
    if builder.function_name != tool_call.function.name {
        return false;
    }
    match (
        serde_json::from_str::<serde_json::Value>(&builder.arguments),
        serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments),
    ) {
        (Ok(existing), Ok(incoming)) => existing == incoming,
        _ => builder.arguments.trim() == tool_call.function.arguments.trim(),
    }
}

/// Detect three consecutive, exactly-identical long fragments at the tail of the text (degenerate repetition loop).
///
/// Used for both reasoning_content and visible assistant output: under long tool-chain contexts the model may
/// verbatim-repeat one sentence in the chain or body; continuing to read only drains the output budget and persists junk.
/// Compares characters rather than bytes to handle Chinese correctly; the fragment must contain enough real content —
/// letters, digits or Chinese characters — so separator lines, whitespace or Markdown punctuation are not misjudged as a degeneration loop.
fn has_degenerate_repetition(text: &str) -> bool {
    // This detector runs on every stream chunk, so only keep a tail large enough to cover the largest candidate fragment,
    // avoiding a long reasoning that degrades into repeatedly scanning the whole text as context grows.
    let mut chars = text
        .chars()
        .rev()
        .take(MAX_REASONING_REPEAT_CHARS * REASONING_REPEAT_COUNT)
        .collect::<Vec<_>>();
    chars.reverse();
    let max_pattern_len = (chars.len() / REASONING_REPEAT_COUNT).min(MAX_REASONING_REPEAT_CHARS);
    if max_pattern_len < MIN_REASONING_REPEAT_CHARS {
        return false;
    }

    for pattern_len in MIN_REASONING_REPEAT_CHARS..=max_pattern_len {
        let repeated_len = pattern_len * REASONING_REPEAT_COUNT;
        let repeated = &chars[chars.len() - repeated_len..];
        let pattern = &repeated[..pattern_len];
        if pattern.iter().filter(|ch| ch.is_alphanumeric()).count() < MIN_REASONING_REPEAT_CHARS / 2
        {
            continue;
        }
        if repeated[pattern_len..pattern_len * 2] == *pattern
            && repeated[pattern_len * 2..] == *pattern
        {
            return true;
        }
    }
    false
}

#[derive(Clone, Copy)]
enum StreamEventMergeMode {
    Append,
    AppendMissingSuffix,
}

fn render_thinking_event(
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    event: &StreamTextEvent,
) -> Result<(), Box<dyn std::error::Error>> {
    if !runtime_ctx::terminal_output_enabled() {
        return Ok(());
    }

    match event {
        StreamTextEvent::OpenThinking => {
            flush_digest_filter_to_terminal(markers, state, false)?;
            if markers.subagent_preview_enabled() {
                return Ok(());
            }
            clear_waiting_hint(state)?;
            maybe_write_stream_content(
                &format!("\n{}\n", markers.thinking_tag),
                state,
                markers,
                true,
            )?;
        }
        StreamTextEvent::AppendThinking(text) => {
            if text.is_empty() {
                return Ok(());
            }
            clear_waiting_hint(state)?;
            // digest is extra image-understanding content meant for the model; the thinking channel's terminal display strips it too
            let terminal_text = state.render.digest_filter.push(text);
            if terminal_text.is_empty() {
                return Ok(());
            }
            if markers.subagent_preview_enabled() {
                write_subagent_content_folded(terminal_text.as_str(), state)?;
            } else {
                maybe_write_stream_content(terminal_text.as_str(), state, markers, true)?;
            }
        }
        StreamTextEvent::CloseThinking => {
            flush_digest_filter_to_terminal(markers, state, true)?;
            if markers.subagent_preview_enabled() {
                return Ok(());
            }
            clear_waiting_hint(state)?;
            if state.render.thinking_fold.active {
                finalize_thinking_fold(state)?;
            } else {
                maybe_write_stream_content(
                    &format!("{}\n", markers.end_thinking_tag),
                    state,
                    markers,
                    true,
                )?;
            }
        }
        StreamTextEvent::AppendContent(_) | StreamTextEvent::AppendHiddenMeta(_) => {}
    }

    Ok(())
}

fn flush_digest_filter_to_terminal(
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    dimmed: bool,
) -> io::Result<()> {
    let residual = state.render.digest_filter.flush();
    if residual.is_empty() || !runtime_ctx::terminal_output_enabled() {
        return Ok(());
    }
    clear_waiting_hint(state)?;
    if markers.subagent_preview_enabled() {
        write_subagent_content_folded(&residual, state)
    } else {
        maybe_write_stream_content(&residual, state, markers, dimmed)
    }
}

fn stream_text_event_to_content(
    event: &StreamTextEvent,
    markers: &StreamMarkers,
    merge_mode: StreamEventMergeMode,
    assistant_text: &str,
) -> Option<String> {
    // Thinking events may only go through render_thinking_event()'s terminal display path; they must not enter
    // assistant_text/current_history. Deliberately only visible body is returned here.
    if markers.subagent_preview_enabled() {
        return match event {
            StreamTextEvent::AppendContent(text) => match merge_mode {
                StreamEventMergeMode::Append => (!text.is_empty()).then(|| text.clone()),
                StreamEventMergeMode::AppendMissingSuffix => {
                    let suffix = unseen_suffix(assistant_text, text);
                    (!suffix.is_empty()).then_some(suffix)
                }
            },
            StreamTextEvent::OpenThinking
            | StreamTextEvent::AppendThinking(_)
            | StreamTextEvent::CloseThinking
            | StreamTextEvent::AppendHiddenMeta(_) => None,
        };
    }

    match event {
        StreamTextEvent::AppendContent(text) => match merge_mode {
            StreamEventMergeMode::Append => (!text.is_empty()).then(|| text.clone()),
            StreamEventMergeMode::AppendMissingSuffix => {
                let suffix = unseen_suffix(assistant_text, text);
                (!suffix.is_empty()).then_some(suffix)
            }
        },
        StreamTextEvent::OpenThinking
        | StreamTextEvent::AppendThinking(_)
        | StreamTextEvent::CloseThinking
        | StreamTextEvent::AppendHiddenMeta(_) => None,
    }
}

fn unseen_suffix(existing: &str, incoming: &str) -> String {
    if incoming.is_empty() || existing.ends_with(incoming) {
        return String::new();
    }

    let leading_ws_len = incoming
        .char_indices()
        .find_map(|(idx, c)| (!c.is_whitespace()).then_some(idx))
        .unwrap_or(incoming.len());
    if leading_ws_len > 0 {
        let trimmed = &incoming[leading_ws_len..];
        if trimmed.is_empty() || existing.ends_with(trimmed) {
            return String::new();
        }
        if let Some(suffix) = unseen_suffix_after_visible_overlap(existing, trimmed) {
            return suffix;
        }
    }

    if let Some(suffix) = unseen_suffix_after_visible_overlap(existing, incoming) {
        return suffix;
    }

    if let Some(suffix) = unseen_suffix_whitespace_tolerant(existing, incoming) {
        return suffix;
    }
    incoming.to_string()
}

fn unseen_suffix_after_visible_overlap(existing: &str, incoming: &str) -> Option<String> {
    let boundaries = incoming
        .char_indices()
        .map(|(idx, _)| idx)
        .chain(std::iter::once(incoming.len()))
        .collect::<Vec<_>>();

    for overlap_chars in (1..boundaries.len()).rev() {
        let split_idx = boundaries[overlap_chars];
        let overlap = &incoming[..split_idx];
        // Overlaps of pure whitespace (e.g. \n) are almost always false matches — models often
        // start a new paragraph with \n, and assistant_text often ends with \n. Only overlaps
        // containing visible characters count as real repetition.
        if existing.ends_with(overlap) && overlap.chars().any(|c| !c.is_whitespace()) {
            return Some(incoming[split_idx..].to_string());
        }
    }

    None
}

/// Whitespace-tolerant suffix dedup.
///
/// Tolerates whitespace differences from old history/abnormal providers: when `assistant_text` and the final
/// `response.output_text.done` snapshot differ only in whitespace, still avoid re-appending the whole already-streamed
/// snapshot as new content.
///
/// Aligns existing and incoming character by character with "whitespace skippable" to find the incoming prefix already
/// covered by existing, and returns the remaining incoming tail (original whitespace preserved). If all of incoming's visible
/// characters are covered, returns `Some("")`; if the visible characters cannot align, returns `None`.
fn unseen_suffix_whitespace_tolerant(existing: &str, incoming: &str) -> Option<String> {
    let e: Vec<(usize, char)> = existing.char_indices().collect();
    let i: Vec<(usize, char)> = incoming.char_indices().collect();
    let (mut ei, mut ii) = (0usize, 0usize);

    // Skip leading whitespace of incoming (snapshots often start with a newline while assistant_text does not)
    while ii < i.len() && i[ii].1.is_whitespace() {
        ii += 1;
    }

    // last_matched_ii records the Vec index just after the last matched visible character in incoming,
    // used to locate the byte start of the "remaining uncovered tail" in incoming at the end.
    let mut last_matched_ii = ii;

    while ei < e.len() && ii < i.len() {
        let (ec, ic) = (e[ei].1, i[ii].1);
        if ec.is_whitespace() && ic.is_whitespace() {
            while ei < e.len() && e[ei].1.is_whitespace() {
                ei += 1;
            }
            while ii < i.len() && i[ii].1.is_whitespace() {
                ii += 1;
            }
            continue;
        }
        if ec.is_whitespace() {
            ei += 1;
            continue;
        }
        if ic.is_whitespace() {
            ii += 1;
            continue;
        }
        // Both sides are visible characters: they must be equal to count as aligned
        if ec == ic {
            last_matched_ii = ii + 1;
            ei += 1;
            ii += 1;
        } else {
            return None;
        }
    }

    // existing is exhausted; the bytes after last_matched_ii in incoming are the remaining (uncovered) tail.
    // If incoming is fully matched too, start_byte == incoming.len(), returning an empty string.
    let start_byte = i
        .get(last_matched_ii)
        .map(|(b, _)| *b)
        .unwrap_or(incoming.len());
    Some(incoming[start_byte..].to_string())
}

fn flush_sse_event(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter: &'static dyn ProviderAdapter,
) -> Result<StreamPayloadOutcome, Box<dyn std::error::Error>> {
    let Some(event) = framing::flush_sse_event(&mut state.framing) else {
        return Ok(StreamPayloadOutcome::default());
    };
    process_stream_payload(
        app,
        current_history,
        markers,
        state,
        adapter,
        event.event_type.as_deref(),
        &event.payload,
    )
}

fn process_stream_line(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter: &'static dyn ProviderAdapter,
    line: &str,
) -> Result<StreamPayloadOutcome, Box<dyn std::error::Error>> {
    if let Some(event) = framing::consume_sse_line(&mut state.framing, line) {
        return process_stream_payload(
            app,
            current_history,
            markers,
            state,
            adapter,
            event.event_type.as_deref(),
            &event.payload,
        );
    }

    Ok(StreamPayloadOutcome::default())
}

pub(super) fn write_stream_content(
    content: &str,
    markdown: &mut MarkdownStreamRenderer,
    dimmed: bool,
) -> io::Result<()> {
    if !runtime_ctx::terminal_output_enabled() {
        return Ok(());
    }
    write_stream_content_to_terminal(content, markdown, dimmed)
}

fn write_stream_content_to_terminal(
    content: &str,
    markdown: &mut MarkdownStreamRenderer,
    dimmed: bool,
) -> io::Result<()> {
    if markdown.should_render(content) {
        markdown.write_chunk(content, dimmed)?;
        io::stdout().flush()?;
    } else {
        if dimmed {
            print!("{DIM}{content}{RESET}");
        } else {
            print!("{content}");
        }
        io::stdout().flush()?;
    }
    Ok(())
}
