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
    theme::{self, RESET},
    types::{App, StreamOutcome, StreamResult, take_stream_cancelled},
};
use crate::commonw::configw;

use super::{
    MarkdownStreamRenderer,
    extract::{StreamTextEvent, extract_chunk_events_streaming, normalize_stream_text},
    framing, normalize,
    render::markdown::{
        clamp_line_to_terminal_row, clamp_line_to_terminal_row_with_reserve,
        live_preview_cursor_rows, raw_terminal_rows, wrap_line_to_terminal_rows_with_reserve,
    },
    splitter::{InternalToolCallStreamEvent, StreamSplitSegment},
    state::{
        StreamChunkStep, StreamContentState, StreamMarkers, StreamProcessingState,
        TerminalDedupeState, ToolCallBuilder,
    },
    wire_log::WireOutcomeNote,
};
mod metrics;
mod hints;
mod tool_calls;
mod fold_render;
mod driver;

use metrics::*;
use tool_calls::*;
pub(super) use hints::*;
pub(super) use fold_render::*;
pub(super) use driver::*;

// Slices under runtime/ still address their old sibling modules via `super::state` and
// `super::side_note_input` (before the split those paths resolved through `runtime`
// itself). Re-bind the sibling module names here so the moved code keeps resolving
// without per-file path edits.
use crate::ai::stream::side_note_input;
use crate::ai::stream::state;


/// Maximum number of decode errors before giving up and returning partial content
/// Delay in milliseconds between retry attempts on transient errors
/// Grace window after an OpenAI-compatible `finish_reason` chunk. Some backends
/// do not emit `[DONE]` or close the HTTP body, while others can still send a
/// final snapshot immediately after the finish chunk.

/// Silence allowance on a connection with no provider-declared work in progress.
///
/// This stays a *connection* bound: a stream that already delivered something is worth preserving through a quiet
/// stretch, but not indefinitely — after this much silence it is treated as stalled and the attempt replays
/// through the retry ladder. Healthy streams on this wire keep refreshing it (visible text, reasoning summaries,
/// tool-argument deltas, item open/close events), so a provider that really stopped sending is discovered
/// quickly. Silences that are *expected* because the provider declared work are covered by
/// `STREAM_DECLARED_ITEM_TIMEOUT_SECS` instead of by widening this bound.
///
/// A model whose gateway hides long stretches of work behind silence without opening an output item may widen
/// this default per model (`stream_silence_timeout_secs` in the model registry).
/// Silence allowance while the provider holds an open output item (`response.output_item.added` without its
/// matching `.done`).
///
/// An open item is the provider's own declaration that work is under way, and both shapes this wire produces can
/// then stay completely quiet: hidden thinking streams no content at all (measured 111s of total wire silence
/// inside one reasoning item, and ~285s before the first output item, on muse-spark-1.3 at xhigh effort), and a
/// large tool-argument payload is generated server-side before it is delivered (a 20KB `write_file` argument
/// arrives as a single late delta).
///
/// Only that declared state gets the allowance: with no item open, silence is still cut by the connection bound
/// above, so a dead connection is normally discovered in seconds rather than minutes. Inside an open item a dead
/// connection is indistinguishable from a slow provider — no bytes arrive either way — so this is the accepted
/// cost of not discarding healthy generations; `MAX_TOOL_ARG_BYTES` and the tool-argument stall bound below stay
/// as the runaway guards for tool calls.
/// Tool-call argument stall timeout: an absolute bound on how long one tool call may stay open, even if the provider
/// keeps trickling argument deltas (the silence allowance only catches total silence; a server sending one small
/// delta every few minutes refreshes the meaningful-progress timer forever, leaving the terminal stuck on
/// "receiving `tool` arguments…" — observed incident: an apply_patch call whose arguments never completed for 13+
/// minutes). Generating the arguments of a large artifact is legitimate work that can take minutes, so this is a
/// backstop against a stream that never finishes, not a normal-path timeout; `MAX_TOOL_ARG_BYTES` remains the primary
/// runaway guard. On expiry the open call is dropped and the attempt replays through the truncation path.
/// First-chunk timeout: the request was sent but the server never sends the first byte (queued, stuck gateway, ...).
/// A stream that produced nothing yet gets this shorter window: unlike a stall after work has been delivered, there
/// is nothing to preserve by waiting longer, so the retry may start earlier.
/// Default visible-window height for `thinking` in the terminal. Only affects display, not reasoning accumulation.
/// Streaming shows the most recent N lines (default 2); when thinking ends, `finalize_fold` forces a pure-summary
/// fold (redraws with a 0-line window) so conclusions/questions restated at the tail of thinking are not shown twice
/// alongside the final answer in the terminal.
/// Physical rows the live fold window reserves outside its body budget: the anchored header (1), the fold-summary
/// marker (1), and one row of headroom so the cursor never sits on the very bottom row. Bounding the body to
/// `viewport_rows - this` keeps the whole window (header + body) inside the visible viewport, so the relative-cursor
/// erase (`\x1b[nA`) in `erase_fold_body` always reaches the top body row instead of being clamped at the viewport
/// top once the window scrolls into scrollback — the root cause of stacked `… more` / `… N earlier lines` markers.
/// Indentation for folded thinking/subagent bodies: header/footer use 2 spaces, body is indented one more level.
/// Terminals usually wrap at the right edge with delayed-wrap; folded redraws always leave two extra columns so that
/// a missing terminal flag or a one-column width/char-width drift cannot trigger an implicit wrap that was not counted
/// in cursor-up, leaving residue from the old window under the `✓`.
/// Shortest repeated fragment and decision count for reasoning-stream degeneration. Only reasoning is checked, so
/// legitimately repeated body the model was asked to produce (tables, code, test data) is not misjudged as degeneration.
/// Cap on accumulated streaming tool-call arguments (total across all tool calls in one turn).
/// Once the model opens a tool call it should close quickly; if arguments keep growing until the cap is hit (e.g.
/// an endless loop concatenating the same body in apply_patch), the output has degenerated. Existing degeneration
/// detection only covers reasoning/assistant text, not tool arguments; this total cap is a backstop against infinite waits and memory growth.

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

    /// Not a stop: the payload carried no content but advanced the provider's declared work state
    /// (a reasoning item opening or closing), which refreshes the silence timer.
    fn progress() -> Self {
        Self {
            should_stop: false,
            meaningful_progress: true,
        }
    }
}

/// Silence allowance for the current stream state.
///
/// A held open output item is a state the provider declared, so it selects the declared-item allowance (reasoning
/// items stream no content while they are open, and an open `function_call` item is an argument payload being
/// generated server-side). Every other case is a connection bound: anything delivered makes the stream worth
/// preserving through a quiet stretch, while a stream that has produced nothing at all gets the shorter
/// first-chunk window because there is nothing to preserve. Empty packets, usage-only and heartbeat events do not
/// refresh the timer, so a provider pushing useless packets cannot keep the stream open forever.
///
/// A model may widen the connection bound through the registry (`stream_silence_timeout_secs`); the declared-item
/// and first-chunk allowances stay global.
fn stream_silence_timeout_secs(state: &StreamProcessingState) -> u64 {
    if state.content.open_output_items > 0 {
        return STREAM_DECLARED_ITEM_TIMEOUT_SECS;
    }
    let produced_something = !state.content.assistant_text.is_empty()
        || !state.content.tool_calls_map.is_empty()
        || state.content.finish_reason_seen;
    if produced_something {
        state
            .model_silence_timeout_secs
            .unwrap_or(STREAM_IDLE_TIMEOUT_SECS)
    } else {
        STREAM_FIRST_CHUNK_TIMEOUT_SECS
    }
}

fn initial_stream_processing_state(app: &App) -> StreamProcessingState {
    let mut state = StreamProcessingState::with_filters(app.hooks.stream_filters().clone());
    state.wire_log.arm(&app.current_model);
    state.model_silence_timeout_secs = models::stream_silence_timeout_for_model(&app.current_model);
    state
}

/// Update the tool-argument stall timer: starts it when a tool call opens, resets it once no
/// tool call is open or the provider already declared how the response ended. See
/// `STREAM_TOOL_ARGS_STALL_TIMEOUT_SECS`.

/// Whether an open tool call has been receiving arguments for at least `stall_timeout` without finishing.

/// Track an open `function_call` output item from the Responses item events.
///
/// The provider declaring a tool call in progress means that call's arguments are being generated server-side, and
/// this wire can deliver the whole payload as one late delta (a 20KB `write_file` argument) with no event in
/// between, so that silence must not be read as a stalled connection. Reasoning items carry the same open/closed
/// state in their own parse result, which also resolves an `.added` line whose item status is already terminal;
/// only chunks can come from a `function_call` item here, so the two paths cannot double count.

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
    state.wire_log.set_wire(adapter.label());

    if should_show_waiting_hint(app) {
        print_waiting_hint(&mut state)?;
    }

    let mut last_meaningful_progress_at = Instant::now();
    let mut idle_timeout_secs = None;
    let mut tool_args_open_at = None;
    let mut tool_args_stalled = false;

    while !app.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        if let Some(result) = immediate_cancel_result(app, &mut state) {
            return Ok(result);
        }

        // A tool call that stays open too long is a stalled stream even if the provider keeps
        // trickling argument deltas (each delta refreshes the idle timer as meaningful progress).
        // Bound the total time a tool call may stay open; on expiry fall through to the same
        // truncation path as the idle timeout (drops the unconfirmed tool call and retries).
        tool_args_open_at = update_tool_args_open_at(&state.content, tool_args_open_at);
        if tool_args_stream_stalled(
            tool_args_open_at,
            Instant::now(),
            Duration::from_secs(STREAM_TOOL_ARGS_STALL_TIMEOUT_SECS),
        ) {
            tool_args_stalled = true;
            idle_timeout_secs = Some(STREAM_TOOL_ARGS_STALL_TIMEOUT_SECS);
            break;
        }

        let timeout_secs = stream_silence_timeout_secs(&state);
        // A provider-declared end (`response.completed`, or a stop declared as incomplete) also ends
        // the response, so only the grace window is still waited for: a socket that stays open after
        // the provider said the response is over cannot be read as a stalled stream and discard
        // delivered tool calls.
        let chunk_result = if state.content.finish_reason_seen
            || state.content.response_completed
            || state.content.response_incomplete
        {
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

        let step = match process_chunk_result(
            app,
            current_history,
            &markers,
            &mut state,
            adapter,
            chunk_result,
        )
        .await
        {
            Ok(step) => step,
            Err(err) => {
                // Only `finalize` erases the fold's live header and body, and each attempt renders
                // through its own fresh state. Aborting on a stream error would leave a
                // `○ thinking · ...` row on screen that the next attempt's fold stacks a second
                // header below. Settle the fold before propagating the error.
                let _ = finalize_thinking_fold(&mut state);
                return Err(err.into());
            }
        };

        match step {
            StreamChunkStep::Continue {
                meaningful_progress,
            } => {
                if meaningful_progress {
                    last_meaningful_progress_at = Instant::now();
                }
            }
            StreamChunkStep::Stop => break,
            StreamChunkStep::Return(result) => {
                // `Return` leaves the loop without going through the normal tail, and only fold
                // finalize erases the fold's live header/body; settle here so an aborted attempt
                // cannot leave a stale `○ thinking` row above the next attempt's output.
                let _ = finalize_thinking_fold(&mut state);
                return Ok(result);
            }
        }
    }

    let tail = match process_pending_tail(app, current_history, &markers, &mut state, adapter).await
    {
        Ok(tail) => tail,
        Err(err) => {
            // Same as the chunk loop above: an error here aborts the attempt while the thinking
            // fold is still live on screen, leaving a header for the next attempt to stack on.
            let _ = finalize_thinking_fold(&mut state);
            return Err(err.into());
        }
    };
    if let Some(result) = tail {
        return Ok(result);
    }

    // A provider-declared end (a completion, or a declared incomplete stop) already delivered
    // everything it had, so silence after it is the socket closing rather than a stream failure, and
    // must not be reported as one. A stalled tool-argument stream is a runtime-side cut regardless
    // (`tool_args_stalled`), and stays an error so half-generated arguments are never executed.
    let provider_declared_end =
        (state.content.response_completed || state.content.response_incomplete) && !tool_args_stalled;
    let idle_timeout_secs =
        idle_timeout_secs.filter(|_| !state.content.finish_reason_seen && !provider_declared_end);
    if idle_timeout_secs.is_some() {
        state.content.stream_idle_timed_out = true;
    }

    // Live regions erase relative to the current cursor. Finish all buffered
    // output before the warning moves it, otherwise cleanup leaves an old
    // thinking header behind and can erase part of the warning instead.
    let result = finalize_stream_response(app, current_history, &markers, state);
    if let Some(timeout_secs) = idle_timeout_secs {
        if runtime_ctx::terminal_output_enabled() {
            // The advisory warning must not replace the finalization result,
            // including an error returned while flushing terminal output.
            let stdout = io::stdout();
            let mut out = stdout.lock();
            if tool_args_stalled {
                let _ = writeln!(
                    out,
                    "  ⚠ 工具调用参数流超过 {timeout_secs} 秒未完成，按流中断处理…"
                );
            } else {
                let _ = writeln!(
                    out,
                    "  ⚠ 响应流连续 {timeout_secs} 秒无有效进展，按流中断处理…"
                );
            }
            let _ = out.flush();
        }
    }

    result
}

/// Whether to show a compact "waiting for model output" status hint in the terminal.
/// Applies to all TTY sessions. Written and flushed on its own line so it appears
/// immediately; once the first visible chunk arrives it is erased by moving the cursor back up over
/// the rows the hint occupies (see `erase_waiting_hint`), leaving no extra lines behind.


/// Write the hint as exactly one live-region row and remember its plain text: the hint owns its own line
/// and is erased by moving the cursor back up over it, and a terminal that narrowed re-wraps the row that
/// is already on screen, so the erase recomputes the physical row count from the text actually written.

/// Erase the hint row(s) currently on screen and park the cursor where the hint started, so the next
/// output prints in its place. Clears `waiting_hint_line` but leaves the hint flags to the caller:
/// `clear_waiting_hint` resets them, while the in-place rewrites (`upgrade_waiting_hint_for_buffering`,
/// `show_deferred_body_buffering_hint`, `refresh_deferred_body_rate_hint`,
/// `refresh_tool_call_rate_hint`) keep the hint active.


/// Label for the tool-call receiving hint. `rate_text` is the optional live
/// argument-throughput suffix: it stays `None` until a measurable argument window
/// exists, so the row keeps its original wording in the first moments of a call.






/// Show a compact "generating…" hint while the assistant body is being withheld
/// (defer_assistant_body) and thinking is closed. The final answer is streamed but
/// not rendered until the completion/citation gates accept it, so without this the
/// terminal stays blank for the whole generation. Upgrades the initial "waiting…"
/// line in place (cursor up + clear + rewrite) and stays put until
/// `clear_waiting_hint` fires — at the next renderable chunk or at stream end.

/// Minimum interval between in-place refreshes of a live rate hint, shared by the
/// deferred-body "generating…" hint and the tool-call "receiving `X` arguments…" hint
/// (only one of them owns the row at a time). The refresh is chunk-driven (no timer in
/// the stream loop), so this bounds terminal repaints to ~2 Hz while tokens keep
/// flowing; a stalled stream keeps the last written rate.

/// Real-time output-throughput text for the deferred-body "generating…" hint.
/// While `defer_assistant_body` withholds the final answer from the terminal until
/// the completion/citation gates accept it, this in-place rewrite is the only
/// place the live rate is visible. Called on each committed output chunk and
/// throttled by `DEFERRED_HINT_RATE_REFRESH_MS`; rows whose text is unchanged are
/// not repainted, so there is no flicker. The `~` prefix marks the estimate as
/// heuristic; the exact server-reported numbers are printed at stream end by
/// `maybe_print_token_throughput_metrics`.

/// Live argument-throughput text for the tool-call receiving hint. Tool-call arguments
/// are never printed to the terminal (`write_tool_call_arguments_stream` is a no-op), so
/// this in-place rewrite is the only sign that a large payload — apply_patch,
/// execute_command, task, … — is still flowing. Throttled and gated exactly like the
/// deferred-body hint; the `~` prefix marks the count as a text-based estimate.

/// Pure formatter for a waiting hint's live rate text (`~N tok @ R tok/s`), shared by the
/// deferred-body "generating…" hint and the tool-call "receiving `X` arguments…" hint.
/// `None` until the first token of that window and until `MIN_RATE_WINDOW` has elapsed,
/// mirroring `format_token_rate`'s "—" gate so the very first chunk is not reported as a
/// misleading burst rate.


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
    state.wire_log.note_cancelled();
    flush_inline_markup_normalizer_on_cancel(state);
    // A content-channel reasoner can keep all received text inside the demuxer until
    // it observes its response delimiter. Ctrl+C must not discard that received
    // prefix merely because the delimiter never arrived.
    let (residual_reasoning, residual_content) = state.content.content_think_demuxer.flush();
    state.content.reasoning_text.push_str(&residual_reasoning);
    state.content.assistant_text.push_str(&residual_content);
    state.content.live_reasoning_tokens = state
        .content
        .live_reasoning_tokens
        .saturating_add(estimate_stream_tokens(&residual_reasoning));

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
        } else if crate::ai::background::serve_live_streaming() {
            // Truncated stream with no local fold: close the remote fold
            // instead of leaking the end marker to child stdout (the SSE
            // line pump would forward it as a plain message). Local piped
            // runs keep printing the marker, as before.
            crate::ai::background::publish_serve_frame(
                crate::ai::background::ServeLiveKind::ThinkingDone,
                "",
            );
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
        state.content.live_reasoning_tokens = state
            .content
            .live_reasoning_tokens
            .saturating_add(estimate_stream_tokens(&residual_reasoning));
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
        let result = cancelled_stream_result(&mut state);
        return Ok(result);
    }

    let metrics_finished_at = Instant::now();
    // Idle timeout means the stream died before the provider could deliver a
    // trailing usage chunk; that path already warns separately, so only surface
    // missing usage on streams that actually completed.
    let stream_error = state.content.stream_idle_timed_out;

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
        maybe_print_token_throughput_metrics(&usage, &state.content, metrics_finished_at);
    } else if !stream_error && !state.content.tool_args_cap_exceeded {
        // The request explicitly asked for stream_options.include_usage, so a
        // completed stream without any usage block means the provider dropped
        // the accounting data: surface the gap instead of silently skipping
        // kernel charging (token stats would be missing with no trace).
        crate::ai::request::emit_request_diagnostic(format_args!(
            "[Warning] 流式响应结束但未收到 usage (model={})：本次调用未计入 token 统计，/usage 将缺失此调用",
            app.current_model
        ));
    }
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
        } else if state.content.response_incomplete {
            // The provider itself declared this response incomplete (Responses wire
            // `response.incomplete`, or a terminal event whose status is not `completed`). That is
            // the server stating the model was stopped rather than finished, so the visible text is
            // a partial generation — treating it as Completed is what ends a cut-off "announced the
            // work, then stopped" turn silently. Escalate to retryable Truncated so the upper layer
            // injects the shrink hint and retries.
            //
            // The `finish_reason=length` rule below keeps its existing scope: several reasoning
            // models report `length` on top of a complete visible body, and only the provider's own
            // terminal declaration separates those from a generation that was actually cut.
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
    // Serve-streamed turns already delivered this text through the live FIFO;
    // re-printing a truncated tail would show it twice.
    if render_terminal && state.render.defer_assistant_body && outcome != StreamOutcome::Completed
        && !crate::ai::background::serve_live_streaming()
    {
        let visible_text = crate::ai::request::strip_digest_blocks(&state.content.assistant_text);
        let duplicate = deferred_dedupe_candidate
            .as_deref()
            .is_some_and(|candidate| candidate.trim() == visible_text.trim());
        if !duplicate && !visible_text.trim().is_empty() {
            super::render_markdown_block(&visible_text)?;
        }
    }

    state.wire_log.note_outcome(WireOutcomeNote {
        outcome: outcome.clone(),
        truncated_by_length,
        stream_error,
        finish_reason: state.content.finish_reason_value.as_deref(),
        dropped_malformed_tool_call: state.content.dropped_malformed_tool_call,
        tool_calls: tool_calls.len(),
        assistant_chars: state.content.assistant_text.chars().count(),
        reasoning_chars: state.content.reasoning_text.chars().count(),
        response_completed: state.content.response_completed,
        response_incomplete: state.content.response_incomplete,
        tool_args_cap_exceeded: state.content.tool_args_cap_exceeded,
        decode_errors: state.framing.decode_error_count,
        usage: usage_snapshot,
    });

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

/// Print a heuristic generation-throughput line after a stream finishes. The timing split is an
/// estimate, not server-reported: the reasoning window ends at the first visible output token, so
/// interleaved thinking after output starts (or idle gaps between phases) skews the reasoning
/// rate. `completion_tokens` includes reasoning tokens and tool-call argument tokens; the output
/// slice is the non-reasoning remainder after subtracting
/// `completion_tokens_details.reasoning_tokens`, so it can exceed what was rendered.

/// Live reasoning-rate text for the in-progress thinking-fold header. It is
/// recomputed on every fold redraw (each thinking chunk) from the approximate
/// `~`-prefixed token estimate; the exact server-reported metrics are printed
/// separately at stream end. Returns `None` until the first reasoning token
/// arrives so the header stays stable during the pre-token window.

/// Approximate token count for live throughput display (heuristic, prefixed with `~`):
/// ASCII text averages ~4 chars/token, CJK characters ~1 token each (3 UTF-8 bytes each).
/// A pure byte/4 estimate under-counts CJK by ~25%, so split by character class instead.

/// Pure formatter for the split reasoning/output throughput line.
/// A missing timing sample is rendered as an em dash instead of inventing a rate.

/// Minimum elapsed window before an instantaneous rate is reported; shorter windows
/// (e.g. the first chunk right after output starts) would print misleadingly large rates.



/// Pure function: build a readable cache-hit line from prompt_tokens / cached_tokens.
/// Returns Some only when there really was a hit (cached > 0), to avoid pointless noise.


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
        let _ = suspend_live_terminal_regions(state);
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

    let mut fold_settled = false;
    if runtime_ctx::terminal_output_enabled() {
        if state.content.thinking_open {
            let _ = flush_digest_filter_to_terminal(markers, state, true);
        }
        fold_settled = finalize_live_folds_before_diagnostic(state);
        eprintln!("[Error] 响应流读取失败，返回已收集的内容");
    }

    if runtime_ctx::terminal_output_enabled() {
        if state.content.thinking_open {
            let _ = flush_digest_filter_to_terminal(markers, state, true);
        }
        if state.content.thinking_open && !fold_settled {
            if crate::ai::background::serve_live_streaming() {
                // Same off-stdout rule as the splitter Marker arm: close the
                // remote fold instead of printing the marker where the SSE
                // line pump would forward it as a plain message.
                crate::ai::background::publish_serve_frame(
                    crate::ai::background::ServeLiveKind::ThinkingDone,
                    "",
                );
            } else {
                let _ = write_stream_content(
                    &format!("\n{}\n", markers.end_thinking_tag),
                    &mut state.render.markdown,
                    false,
                );
                print!("\x1b[0m");
                let _ = io::stdout().flush();
            }
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
    // Serve-streamed turns already delivered this text through the live FIFO;
    // re-printing a truncated tail would show it twice.
    if runtime_ctx::terminal_output_enabled()
        && state.render.defer_assistant_body
        && outcome != StreamOutcome::Completed
        && !crate::ai::background::serve_live_streaming()
    {
        let visible_text = crate::ai::request::strip_digest_blocks(&assistant_text);
        if !visible_text.trim().is_empty() {
            let _ = super::render_markdown_block(&visible_text);
        }
    }

    state.wire_log.note_stream_cut(state.framing.decode_error_count);

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




/// The terminal does not print tool-call arguments; the streaming receive phase only shows an erasable
/// tool-name status line, while the actual execution lines are printed uniformly by the tool execution layer.


/// Incrementally resolves cumulative keys for chat-completions tool calls whose
/// `index` is missing. If everything fell onto the default key 0, parallel tool
/// calls would merge into one; instead group by id: reuse the key of an existing
/// builder with the same id, otherwise synthesize a stable key in
/// [10000, usize::MAX) from a hash of the id (real provider indexes are single
/// digits, so no collision). Parameter-continuation deltas with neither id nor
/// index attach to the most recent call without an index; if there is none, fall
/// back to the old-behavior key 0.


/// Consume internal tool-call stream events. The return value indicates whether this batch detected a hallucinated
/// internal tool-protocol marker (`HallucinatedProtocolMarker`) — the caller then stops the stream and retries downgraded.

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
            refresh_deferred_body_rate_hint(state)?;
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
            // Output timing starts only when real visible text is committed; a pure
            // end_thinking_tag separator (or trimmed-away residue) must not count as output.
            state.content.mark_output_started();
            state.content.live_output_tokens = state
                .content
                .live_output_tokens
                .saturating_add(estimate_stream_tokens(text));
            current_history.push_str(text);
            state.content.assistant_text.push_str(text);
            // Live output for a real-time `/bg` handoff (best-effort, no-op
            // unless the backgrounded process opened its live-output FIFO).
            crate::ai::background::publish_live_output(text);
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
    if !text.is_empty() {
        state.content.mark_output_started();
        state.content.live_output_tokens = state
            .content
            .live_output_tokens
            .saturating_add(estimate_stream_tokens(&text));
    }
    current_history.reserve(text.len());
    state.content.assistant_text.reserve(text.len());
    current_history.push_str(&text);
    state.content.assistant_text.push_str(&text);
    // Live output for a real-time `/bg` handoff (best-effort, no-op unless the
    // backgrounded process opened its live-output FIFO).
    crate::ai::background::publish_live_output(&text);

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
            if crate::ai::background::serve_live_streaming() {
                // Serve children have no local fold: a marker reaching the
                // body path means thinking already closed when the chunk
                // arrived (typical for tool-call rounds), or a late splitter
                // flush. Keep it off child stdout — the SSE line pump would
                // forward it as a plain message and the chat client would
                // print a second close line under its folded `✓ thinking`
                // summary. The chat-side close is idempotent, so this also
                // safely covers markers the CloseThinking event reported.
                crate::ai::background::publish_serve_frame(
                    crate::ai::background::ServeLiveKind::ThinkingDone,
                    "",
                );
                return Ok(());
            }
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


// Scratch buffer reused across thinking-fold body renders so each redraw does not
// rebuild a zero-capacity String. Purely an allocation-reuse optimization: the
// renderer clears the buffer before every rebuild, so the bytes handed to the
// terminal are identical to building a fresh String each time.
thread_local! {
    static THINKING_FOLD_BODY_BUF: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Make control characters literal before wrapping a rewritable fold row. Raw tabs
/// move to terminal-defined tab stops, while CR/ESC can move the cursor independently
/// of cell widths; either invalidates the erase footprint and leaks previous frames.
/// Use fixed four-space tab indentation and visible escapes for other controls.
/// This is display-only: the fold buffers and canonical model content stay unchanged.

/// Render the **body** of the fold window (fold summary + recent visible lines), without the header.
/// The header is anchored and printed separately by `write_fold_header`. Writes the body into
/// `out` (cleared first) and returns the number of physical body lines plus the plain-text rows
/// kept for later width recomputation; the body does not end with a newline and the cursor always
/// stays on the last line, so xterm.js does not interpret a trailing LF as extra scrolling.

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
    state.wire_log.observe(event_type, payload);
    let parsed = normalize::parse_stream_payload(adapter, payload, event_type);
    note_function_call_item_lifecycle(event_type, &parsed, &mut state.content);
    let (mut chunk, merge_mode, is_replayed) =
        match parsed {
            super::state::ParsedStreamPayload::Ignore => {
                return Ok(StreamPayloadOutcome::default());
            }
            super::state::ParsedStreamPayload::Done => return Ok(StreamPayloadOutcome::stop()),
            super::state::ParsedStreamPayload::Error(msg) => {
                return Err(format!("provider stream error: {msg}").into());
            }
            super::state::ParsedStreamPayload::ReasoningItem { item, open } => {
                // Track the provider's open items: while one is open the model is thinking, and this
                // wire streams nothing for that thinking, so the silence must not be read as a stalled
                // stream (see `stream_silence_timeout_secs`).
                if open {
                    state.content.open_output_items =
                        state.content.open_output_items.saturating_add(1);
                } else {
                    state.content.open_output_items =
                        state.content.open_output_items.saturating_sub(1);
                }
                if let Some(item) = item {
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
                }
                return Ok(StreamPayloadOutcome::progress());
            }
            super::state::ParsedStreamPayload::Chunk(chunk) => {
                (chunk, StreamEventMergeMode::Append, false)
            }
            super::state::ParsedStreamPayload::ResponseTerminal { status, chunk } => {
                // The provider declared how this response ended (Responses wire). Record it before
                // the chunk is processed: a completed response ends at its own marker (`response.completed`),
                // and an incomplete one must stay retryable even when partial visible text exists.
                match status {
                    crate::ai::request::ResponseTerminalStatus::Completed => {
                        state.content.response_completed = true;
                    }
                    crate::ai::request::ResponseTerminalStatus::Incomplete => {
                        state.content.response_incomplete = true;
                    }
                }
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
        // Passthrough (initial state and after CLOSE_TAG) hands content through
        // unchanged: skip the demuxer push to avoid its full-chunk to_string copy.
        && !state.content.content_think_demuxer.is_passthrough()
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
        state.content.mark_reasoning_started();
        state.content.saw_reasoning_output = true;
        state.content.reasoning_text.push_str(&emitted_reasoning);
        state.content.live_reasoning_tokens = state
            .content
            .live_reasoning_tokens
            .saturating_add(estimate_stream_tokens(&emitted_reasoning));

        // Some long tool-chain contexts make the model verbatim-repeat one sentence in thinking. Continuing to read only
        // burns the output budget and makes the terminal look stuck; escalate to retryable truncation and let the upper layer lower the reasoning tier.
        // Gate the reasoning-tail kill on "no productive output yet": once the model is emitting visible text or tool
        // calls, a repeating reasoning tail is cosmetic (the answer is already arriving) and killing the stream would
        // discard valid content. Visible-text repetition is guarded separately below by its own detector, so a genuine
        // content loop is still caught.
        if state.content.assistant_text.trim().is_empty()
            && state.content.tool_calls_map.is_empty()
            && has_degenerate_repetition(&state.content.reasoning_text)
        {
            state.content.finish_reason_seen = true;
            state.content.finish_reason_value =
                Some(DEGENERATE_REPETITION_FINISH_REASON.to_string());
            if runtime_ctx::terminal_output_enabled() {
                let _ = finalize_live_folds_before_diagnostic(state);
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
            let _ = finalize_live_folds_before_diagnostic(state);
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
            let _ = finalize_live_folds_before_diagnostic(state);
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
                // text completely unguarded. On hit, strip the repeated junk tail so the preserved partial text is clean,
                // set finish_reason and stop the stream; the upper layer retries downgraded.
                if let Some(repeated_len) =
                    degenerate_repetition_strip_len(&state.content.assistant_text)
                {
                    // Drop the looped tail (at most MAX_REASONING_REPEAT_CHARS * REASONING_REPEAT_COUNT chars) from the
                    // partial text: without this, the junk would ride into the retry/finalize as "partial progress" and
                    // could be persisted, poisoning the next request exactly like the original incident.
                    let keep = state.content.assistant_text.chars().count() - repeated_len;
                    state.content.assistant_text =
                        state.content.assistant_text.chars().take(keep).collect();
                    state.content.finish_reason_seen = true;
                    state.content.finish_reason_value =
                        Some(DEGENERATE_REPETITION_FINISH_REASON.to_string());
                    if runtime_ctx::terminal_output_enabled() {
                        let _ = finalize_live_folds_before_diagnostic(state);
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
