use super::inline_recovery::{
    collect_valid_tool_calls, ensure_tool_calls_section_open, normalize_inline_tool_call_markup,
    recover_inline_tool_calls,
};
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
        StreamChunkStep, StreamMarkers, StreamProcessingState, TerminalDedupeState,
        ToolCallBuilder,
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

/// 空闲超时：已收到内容后长时间无新 chunk 到达，视为服务端已静默结束。
/// 部分 provider 在输出完毕后既不发送 finish_reason 也不关闭连接，只能靠此超时兜底。
const STREAM_IDLE_TIMEOUT_SECS: u64 = 45;
/// 首 chunk 超时：请求已发出但服务端迟迟不发第一个字节（排队/网关卡住等）。
/// 比 idle 超时更长，因为某些模型冷启动或排队需要时间。
const STREAM_FIRST_CHUNK_TIMEOUT_SECS: u64 = 90;
/// terminal 下 thinking 可见窗口的默认高度。只影响展示，不影响 reasoning 累积。
/// 流式过程显示最近 N 行（默认 2）；思考结束时 `finalize_fold` 会强制折叠为纯摘要
/// （临时按 0 行窗口重画），避免模型在 thinking 尾部复述的结论/问句与最终回答在
/// 终端重复显示。
const DEFAULT_THINKING_MAX_VISIBLE_LINES: usize = 2;
/// thinking / subagent 折叠正文缩进：header/footer 用 2 空格，正文再内缩一层。
const THINKING_FOLD_BODY_INDENT: &str = "    ";
const THINKING_FOLD_BODY_INDENT_WIDTH: usize = 4;
/// 终端右边界通常使用 delayed-wrap；折叠重画统一额外留两列，避免终端标识缺失、
/// 宽度或字符宽度存在一列偏差时触发未计入 cursor-up 的隐式换行，把旧窗口尾部残留
/// 在 `✓` 下方。
const FOLD_REWRITE_RIGHT_MARGIN_COLS: usize = 2;
/// 推理流连续重复的最短片段和判定次数。只检测 reasoning，避免把用户要求生成的重复
/// 正文（表格、代码、测试数据等）误判为模型退化。
const MIN_REASONING_REPEAT_CHARS: usize = 16;
const MAX_REASONING_REPEAT_CHARS: usize = 512;
const REASONING_REPEAT_COUNT: usize = 3;
const DEGENERATE_REPETITION_FINISH_REASON: &str = "degenerate_repetition";

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

pub(super) async fn stream_response(
    app: &mut App,
    response: &mut reqwest::Response,
    current_history: &mut String,
    terminal_dedupe_candidate: Option<&str>,
) -> Result<StreamResult, Box<dyn std::error::Error>> {
    let mut markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.render.terminal_dedupe = terminal_dedupe_candidate
        .map(str::trim)
        .filter(|candidate| !candidate.is_empty())
        .map(|candidate| TerminalDedupeState {
            candidate: candidate.to_string(),
            buffered_terminal_output: String::new(),
        });
    // 预填 `<think>` 模板的 reasoner 把推理链内联在 content 通道，仅以悬空
    // `</think>` 收尾、从不产生 reasoning_content。为这类模型 arm 拆分器，把泄漏的
    // 推理拆回 reasoning，避免思考链与正式答案一起落进可见正文。
    if models::reasoning_in_content_enabled(&app.current_model) {
        state.content.content_think_demuxer.arm();
    }
    configure_thinking_fold(&mut state);
    configure_subagent_preview_fold(app, &mut state, &mut markers);
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

        // 已有可执行/可见进展时用较短的 idle 超时；空包、usage-only、heartbeat
        // 不刷新该计时器，避免 provider 持续推无效包导致 stream 永不收口。
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
            eprintln!("  ⚠ 响应流连续 {timeout_secs} 秒无有效进展，按流中断处理…");
        }
    }

    finalize_stream_response(app, current_history, &markers, state)
}

/// 是否在终端显示「等待模型输出」的紧凑状态提示。
/// 对所有 TTY 会话生效。独立行写入并 flush，保证提示立即可见，
/// 收到首个可见 chunk 时用 \x1b[1A\r\x1b[2K 清掉，不残留额外行。
fn should_show_waiting_hint(app: &App) -> bool {
    runtime_ctx::terminal_output_enabled()
        && io::stdout().is_terminal()
        && !app.shutdown.load(std::sync::atomic::Ordering::Relaxed)
}

fn print_waiting_hint(state: &mut StreamProcessingState) -> io::Result<()> {
    if state.render.waiting_hint_active {
        return Ok(());
    }
    // 独立行等待提示：首个 chunk 到达时用 \x1b[1A\r\x1b[2K 清掉。
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
    // 光标上移、清行后重写独立行，保持 buffering 状态可见。
    let stdout = io::stdout();
    let mut out = stdout.lock();
    write!(out, "\x1b[1A\r\x1b[2K")?;
    writeln!(out, "  {ACCENT_MUTED}⠋ buffering…{RESET}")?;
    out.flush()?;
    state.render.waiting_hint_buffering = true;
    Ok(())
}

pub(super) fn clear_waiting_hint(state: &mut StreamProcessingState) -> io::Result<()> {
    if !state.render.waiting_hint_active {
        return Ok(());
    }
    // 光标上移一行 + \r + 清行：擦掉独立提示行，让内容在原位输出。
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

/// 取消/中断时的 thinking 折叠收尾：折叠窗口若仍活跃，必须先擦掉当前窗口并落一个
/// `✓` 收口，否则半截 thinking 窗口会被留在屏幕上，下一轮重试
/// 的 fresh state 会在其下方再画一个新 header——累积成「重复 header + 大段空白」。
fn cancelled_stream_result(state: &mut StreamProcessingState) -> StreamResult {
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
        assistant_text: String::new(),
        hidden_meta: String::new(),
        reasoning_text: String::new(),
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
        // pending 为空时，仍需检查 sse_event_data 是否有未 flush 的最后一个事件。
        // 部分 provider 在关闭连接前不发送最终空行（\n\n），导致最后一个 SSE 事件被丢弃。
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

    // 冲刷内联 `</think>` 拆分器的残留。仍处捕获态说明 `</think>` 从未到达：
    // withhold 设计下整段缓冲按 **content** 安全回退（宁可降级为「思考泄漏进正文」
    // 也不丢失可见答案）。未 arm 的模型此处恒为空。
    let (residual_reasoning, residual_content) = state.content.content_think_demuxer.flush();
    if !residual_reasoning.is_empty() {
        state.content.reasoning_text.push_str(&residual_reasoning);
    }
    if !residual_content.is_empty() {
        commit_visible_content(app, current_history, markers, &mut state, residual_content)?;
    }

    if render_terminal {
        // 残留也必须经过去重/折叠/样式管线，不能直接写终端。
        flush_digest_filter_to_terminal(markers, &mut state, false)?;
        flush_terminal_splitter(&mut state, markers)?;
        let suppress_duplicate = final_assistant_matches_terminal_dedupe(&state);
        disable_terminal_dedupe(&mut state, suppress_duplicate)?;
        state.render.markdown.flush_pending()?;
    }

    if take_stream_cancelled(app) {
        return Ok(cancelled_stream_result(&mut state));
    }

    // AIOS: flush any pending LLM usage to kernel `/dev/llm` before returning.
    // Prefer the model echoed by the provider; fall back to what we requested.
    // 先快照 usage 统计，供 StreamResult 截断诊断使用（take 会消费）。
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
    if stream_error {
        // 未收到 finish_reason 就 idle timeout，无法证明工具调用已经完整；即使当前
        // arguments 恰好是合法 JSON，也不能提前执行可能仍在生成中的操作。
        tool_calls.clear();
    }

    // Fallback：部分 provider 会把 function call 作为普通 content 返回，而不走
    // delta.tool_calls[]。流式解析未命中时，对完整 assistant_text 再做一次保守恢复。
    if !stream_error && tool_calls.is_empty() {
        if let Some(recovered) = recover_inline_tool_calls(&state.content.assistant_text) {
            tool_calls = recovered;
            // 协议载荷既不是 assistant 正文，也不是模型 self_note。恢复成功后直接
            // 丢弃，避免 no-tool handoff 把 DSML/JSON 持久化为 internal_note。
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
        // 截断优先判定：本轮无有效工具调用，但有工具调用被丢弃（arguments JSON 半截）。
        // 这类"想干活但被切断"的情况若按 Completed 静默结束，会让大文件 write_file
        // 等操作凭空消失。升级为可重试的 Truncated，由上层注入收缩提示后自动重试。
        //
        // 注意：finish_reason=length（撞输出上限）本身并不触发 Truncated，因为推理模型
        // 经常在 reasoning token 占满输出预算后返回 finish_reason=length，但可展示的
        // assistant_text 实际上已完整。若同时有可见文本和 finish_reason=length，按
        // Completed 处理，避免无意义的重试循环。只有在完全没有可见输出时，length 截断
        // 才作为 Truncated 重试（此时模型可能刚开始输出就被掐断）。
        if degenerate_repetition {
            StreamOutcome::Truncated
        } else if state.content.dropped_malformed_tool_call {
            StreamOutcome::Truncated
        } else if truncated_by_length && !has_text {
            // finish_reason=length 且没有可见文本：模型可能只产出了 reasoning
            // 就被掐断，或根本没输出。降 effort 重试，把预算让给实际内容。
            StreamOutcome::Truncated
        } else if has_reasoning && !has_text && !state.content.finish_reason_seen {
            // reasoning-only 早停：只吐了思考、没有可见文本、也**没收到任何
            // finish_reason** 就断流（idle 超时 / 提前 EOF，常见于 GLM 等
            // enable_thinking 模型憋着思考链迟迟不产出可见内容，撞上 idle 超时
            // 被掐断）。这类"思考到一半被切断"若按 Completed 静默结束，会让本轮
            // 回答凭空为空。升级为可重试 Truncated，由上层降档 / 关 thinking 后重试。
            //
            // 与上面的 length 分支互补：length 是服务端显式报截断；这里是流早停、
            // 根本没等到结束标记。区别于「正常 finish_reason=stop 的 reasoning-only
            // 响应」——那种 finish_reason_seen=true，不进本分支，仍按 Completed。
            StreamOutcome::Truncated
        } else if !has_text && !has_reasoning {
            // 检测空响应：模型没有文本、没有工具调用、没有推理内容。
            // 通常是 provider 端的问题（如限流、模型异常），触发重试。
            StreamOutcome::EmptyResponse
        } else {
            StreamOutcome::Completed
        }
    };

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

/// 当开启 `ai.prompt_cache.show_metrics`（默认开）且本次请求命中了 prompt
/// 缓存时，打印一行缓存命中指标。OpenAI / DashScope 等是服务端自动缓存，
/// 这里只是把它们已经上报的 `cached_tokens` 可视化出来。
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

/// 纯函数：根据 prompt_tokens / cached_tokens 生成可读的缓存命中行。
/// 仅当确实有缓存命中（cached > 0）时返回 Some，避免无意义噪声。
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
    // 解码错误兜底路径本身就是流被中途切断的产物；若还伴随工具调用被丢弃，
    // 明确标记为截断以触发上层自动重试，而非静默按完成收尾。
    let outcome = if dropped_malformed {
        StreamOutcome::Truncated
    } else {
        StreamOutcome::Completed
    };

    Some(StreamResult {
        outcome,
        tool_calls,
        assistant_text: std::mem::take(&mut state.content.assistant_text),
        hidden_meta: String::new(),
        reasoning_text: std::mem::take(&mut state.content.reasoning_text),
        reasoning_items: std::mem::take(&mut state.content.reasoning_items),
        skip_response_drain: true,
        truncated_by_length: false,
        // 流读取失败导致的截断，不是模型输出过长。
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

/// 终端不打印工具调用参数；流式接收阶段只显示可擦除的工具名状态提示，
/// 实际执行行仍由工具执行层统一输出。
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
        let index = stream_tool_call.index;
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

/// 消费内部工具调用流事件。返回值表示本批事件里是否检出了模型幻觉的内部工具协议
/// 标记（`HallucinatedProtocolMarker`）——调用方据此停流并降档重试。
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
                    // 流式阶段不再打印工具名/参数（open_tool_call_line 与
                    // write_tool_call_arguments_stream 均为 no-op），因此这里只需
                    // 复位颜色即可。绝不能用 println!——那会在「✓」与后续
                    // 输出之间凭空插入一行空行（外部 delta 工具路径本就不打这行）。
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
                // streamer 已把整块幻觉「工具结果」剥离不外显；这里只记录信号，
                // 由调用方停流并走 degenerate_repetition 降档重试路径。
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
        clear_waiting_hint(state)?;
    }

    // 当 thinking 折叠模式活跃且遇到 end_thinking_tag 时，做最终的折叠渲染
    if render_terminal
        && !state.content.thinking_open
        && state.render.thinking_fold.active
        && is_standalone_stream_marker(&content, &markers.end_thinking_tag)
    {
        finalize_thinking_fold(state)?;
        // end_thinking_tag 内容只用于视觉分隔，不需要追加到 assistant_text
        let text = content.replace(&markers.end_thinking_tag, "");
        let text = text.trim_matches('\n');
        if !text.is_empty() {
            current_history.push_str(text);
            state.content.assistant_text.push_str(text);
        }
        return Ok(());
    }

    if render_terminal {
        // digest 是给模型看的附加图片理解内容，终端展示时剥离（历史/assistant_text 保留原文）
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

/// 流结束时冲刷 `InlineMarkupNormalizer` 缓存的尾部（可能是一个完整但收尾没
/// 补齐的 marker，或普通文本），经与流式相同的解析链处理：识别成的 tool call
/// 进入 tool_calls_map，剩余可见文本追加到 assistant_text，避免半截 marker 或
/// 被暂存的内容丢失。
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
        // 冲刷阶段流已结束，无法再停流重试；此处即便检出幻觉标记，streamer 也已把
        // 整块剥离，cleaned 不含协议标记，直接忽略返回值即可（不落盘幻觉正文）。
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
    state.render.terminal_dedupe.as_ref().is_some_and(|dedupe| {
        dedupe.buffered_terminal_output.trim() == dedupe.candidate.trim()
    })
}

fn final_assistant_matches_terminal_dedupe(state: &StreamProcessingState) -> bool {
    state
        .render
        .terminal_dedupe
        .as_ref()
        .is_some_and(|dedupe| {
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

/// Thinking 内容的折叠渲染：从第一行开始维护一个可重写窗口，
/// 始终只在 terminal 中展示最近 N 行，超出部分折叠为一条摘要。
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

    // 控制行必须是独占 marker，本身不应与正文混写。
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

/// 只覆盖 thinking 正文窗口（折叠摘要 + 最近可见行），header 不在此列。
///
/// header（`○`）在折叠激活时打印一次并锚定在正文之上，之后每次重画都只
/// 擦除并重写正文。正文最多展示 `max_visible_lines` 条物理内容行和一条折叠摘要，恒定
/// 落在可视视口内，因此相对擦除永远够得着，不会随窗口滚入 scrollback
/// 而失步——即便失步，也无法再生出第二个 header，从根上杜绝「孤儿 header 叠加」。
fn thinking_fold_redraw(fold: &mut super::state::ThinkingFoldState) -> io::Result<()> {
    let mut out = io::stdout();
    // 只擦除上一次的正文区域；header 已锚定在其上方，绝不触碰。
    // 注意：terminal 缩窄后，旧正文会被终端按当前宽度自动 reflow 成更多物理行；
    // 这里不能只信缓存的 window_rows，而要按"上次正文在当前宽度下会占几行"重算。
    let erase_rows = thinking_fold_rendered_body_rows(fold).max(fold.window_rows);
    erase_fold_body(&mut out, erase_rows)?;
    if fold.active && !fold.header_drawn {
        write_fold_header(&mut out, fold)?;
        fold.header_drawn = true;
    }

    let (body_lines, marker_lines) = thinking_fold_window_lines(fold);
    let (body, body_rows, rendered_body_lines) = render_thinking_fold_window_lines(
        &body_lines,
        marker_lines,
        fold.rewrite_right_margin_cols,
        fold.max_visible_lines,
    );
    if !body.is_empty() {
        out.write_all(body.as_bytes())?;
    }
    out.flush()?;
    fold.window_rows = body_rows;
    fold.rendered_body_lines = rendered_body_lines;
    Ok(())
}

/// 打印锚定的折叠 header。只应在折叠激活后被调用一次。
fn write_fold_header(
    out: &mut impl Write,
    fold: &super::state::ThinkingFoldState,
) -> io::Result<()> {
    write!(out, "  {ACCENT_MUTED}{}\x1b[0m\r\n", fold.header_label)
}

/// thinking 收尾时直接落最终 header；用于尚未写过进行中 header 的空折叠。
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

/// 将锚定的 `○ thinking` 原位改写为完成态，而不在正文下另起一条 `✓ thinking`。
///
/// 调用前 `erase_fold_body` 已让光标回到 header 下方的正文首行；因此上移一行、清空
/// header 后重写即可。折叠正文仍按原逻辑绘制在新 header 下方。
fn replace_thinking_fold_header(
    out: &mut impl Write,
    fold: &super::state::ThinkingFoldState,
    line_count: usize,
) -> io::Result<()> {
    write!(out, "\r\x1b[1A\r\x1b[2K")?;
    write_thinking_fold_completion_header(out, fold, line_count)
}

/// 正文渲染后光标停在最后一条物理行，而不是额外的空白行。重画时因此只需上移
/// `rows - 1`；先回到行首再擦到屏幕底部，可覆盖窗口变窄后产生的 reflow 行。
fn erase_fold_body(out: &mut impl Write, rows: usize) -> io::Result<()> {
    if rows == 0 {
        return Ok(());
    }
    write!(out, "\r")?;
    if rows > 1 {
        write!(out, "\x1b[{}A", rows - 1)?;
    }
    write!(out, "\x1b[0J")
}

/// Thinking 结束时的最终渲染：覆盖正文窗口，并把锚定的 `○` 原位改为 `✓`。
pub(super) fn finalize_thinking_fold(state: &mut StreamProcessingState) -> io::Result<()> {
    finalize_fold(&mut state.render.thinking_fold, true)
}

fn finalize_subagent_preview_fold(state: &mut StreamProcessingState) -> io::Result<()> {
    // subagent 预览收尾保留窗口正文：子代理的最终输出应留在终端可见，
    // 不套用 thinking 的「强制 0 行纯摘要」收尾（那会把关键结论一并折叠）。
    finalize_fold(&mut state.render.subagent_fold, false)
}

fn finalize_fold(
    fold: &mut super::state::ThinkingFoldState,
    collapse_body: bool,
) -> io::Result<()> {
    let mut out = io::stdout();
    finalize_fold_to(&mut out, fold, collapse_body)
}

/// 折叠收尾写入实现；抽出 writer 以便回归测试精确验证终端光标序列。
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
    // thinking 的完成态要替换进行中 header，而不是在正文下再打印一条 footer。
    // 若折叠尚未真正落地，直接写完成态，避免短暂出现 `○ thinking`。
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
        // subagent 预览维持既有 header/footer 两行布局。
        write_fold_header(&mut out, fold)?;
        fold.header_drawn = true;
    }

    // thinking 收尾折叠为纯摘要（0 行窗口），不显示正文行：思考尾部常复述结论/问句，
    // 若保留可见行会与紧随其后的最终回答在终端重复。subagent 预览收尾不折叠，
    // 保留最近可见窗口行，让子代理最终输出对用户可见。
    let saved_max_visible_lines = fold.max_visible_lines;
    if collapse_body {
        fold.max_visible_lines = 0;
    }
    let (body_lines, marker_lines) = thinking_fold_window_lines(fold);
    let (body, body_rows, rendered_body_lines) = render_thinking_fold_window_lines(
        &body_lines,
        marker_lines,
        fold.rewrite_right_margin_cols,
        fold.max_visible_lines,
    );
    fold.max_visible_lines = saved_max_visible_lines;
    if !body.is_empty() {
        out.write_all(body.as_bytes())?;
    }
    fold.window_rows = body_rows;
    fold.rendered_body_lines = rendered_body_lines;

    if !collapse_body {
        // subagent 预览保留既有 footer；thinking 的规模信息已写入被原位替换的 header。
        if body_rows > 0 {
            out.write_all(b"\r\n")?;
        }
        write!(
            out,
            "  {ACCENT_MUTED}{} · {line_count} lines\x1b[0m\r\n",
            fold.footer_label,
        )?;
    }
    out.flush()?;

    // 重置折叠状态
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
    // 0 行窗口 = 纯摘要模式：连当前未完成行也不显示，避免结论复述泄漏到终端。
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

/// 渲染折叠窗口的**正文**（折叠摘要 + 最近可见行），不含 header。
/// header 由 `write_fold_header` 单独锚定打印。返回的行数即正文物理行数；正文末尾不
/// 输出换行，光标始终停在最后一行，避免 xterm.js 把末尾 LF 解释成额外滚屏。
fn render_thinking_fold_window(fold: &super::state::ThinkingFoldState) -> (String, usize) {
    let (lines, marker_lines) = thinking_fold_window_lines(fold);
    let (window, rows, _) = render_thinking_fold_window_lines(
        &lines,
        marker_lines,
        fold.rewrite_right_margin_cols,
        fold.max_visible_lines,
    );
    (window, rows)
}

fn render_thinking_fold_window_lines(
    lines: &[String],
    marker_lines: usize,
    rewrite_right_margin_cols: usize,
    max_visible_rows: usize,
) -> (String, usize, Vec<String>) {
    if lines.is_empty() {
        return (String::new(), 0, Vec::new());
    }

    let reserve_cols = THINKING_FOLD_BODY_INDENT_WIDTH + rewrite_right_margin_cols;
    let marker_lines = marker_lines.min(lines.len());
    let mut wrapped_content_rows = Vec::new();
    for line in lines.iter().skip(marker_lines) {
        wrapped_content_rows.extend(wrap_line_to_terminal_rows_with_reserve(line, reserve_cols));
    }
    let hidden_wrapped_rows = wrapped_content_rows.len().saturating_sub(max_visible_rows);
    let marker = if hidden_wrapped_rows > 0 {
        // 已按物理行截断时，首个被隐藏的内容可能来自一条仍在输出的逻辑行，不能再
        // 报告不精确的“earlier lines”计数。
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
        // 折叠提示必须永远只占一物理行；正文才允许逐行包裹。
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

    let mut out = String::new();
    let mut rendered_lines = Vec::with_capacity(rows_to_render.len());
    // 折叠正文固定内缩。正文最多保留 max_visible_rows 条包裹后的物理行；若还需
    // 隐藏内容，单行折叠提示不计入正文预算。每个包裹段本身恰好占一个物理行，
    // xterm.js 集成终端额外留出右边距。
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

    (out, rows, rendered_lines)
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
                // 捕获完整 reasoning item（含 encrypted_content）供同 turn 工具链回放。
                // 不产生可见输出，也不进持久化历史。
                state.content.reasoning_items.push(item);
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

    // 记录最近一个非空 finish_reason 的具体值。`length` 表示服务端因输出上限
    // 截断，是把本轮升级为可重试 Truncated 的关键信号。
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

    // content_part.added / output_text.done 可能重发已存在正文，与 output_text.delta
    // 增量重叠：用 **demux 前的原始 content 通道文本** 做未见后缀计算。仅用
    // assistant_text 去重会在 demux 已关闭后失效：重发的 `reasoning</think>answer`
    // 前缀与可见正文 `answer` 对不上，会把 reasoning 再次泄漏到正文。
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

    // 预填 `<think>` 模板拆分：这类 reasoner 把推理链写进 content 通道、仅以悬空
    // `</think>` 收尾。捕获态下 withhold--content 暂存于拆分器缓冲区、不向任何通道
    // 增量吐出，直到 `</think>` 到达才一次性把前缀归入 delta.reasoning_content（复用
    // 既有 thinking 折叠与累积路径）、正文留在 content。若 `</think>` 始终未到达，
    // flush 时整段按 content 安全回退。未 arm 的模型此处直通、零影响。快照(.done)
    // 先在上方按原始 content 通道去重，只把未见后缀送入有状态拆分器，因此不会重复
    // 计数。content_channel_progress 会刷新 idle 计时器，避免长思考被 withhold 误判为
    // 首包/空闲超时。
    if let Some(choice) = chunk.choices.first_mut()
        && !choice.delta.content.is_empty()
    {
        let (reasoning, content) = state
            .content
            .content_think_demuxer
            .push(&choice.delta.content);
        choice.delta.content = content;
        if !reasoning.is_empty() {
            // 与既有语义一致：reasoning_content 可与推理片段拼接续写。
            choice.delta.reasoning_content =
                merge_reasoning_fragments(&choice.delta.reasoning_content, &reasoning);
        }
    }

    // reasoning_content 去重：Responses API 对同一段推理摘要会通过多条事件路径重复
    // 下发（reasoning_summary_text.{delta,done} 与 content_part.{added,done} 的
    // summary_text 携带相同内容）。此前仅对 SnapshotChunk（.done）做未见后缀去重，
    // Append 模式（.delta/.added）的 reasoning_content 未去重，导致跨事件路径的
    // thinking 重复输出。这里统一对两种模式计算未见后缀，渲染时只输出新增部分。
    //
    // 累积到 reasoning_text 时区分模式：Append 累积原文以保留模型复读循环的退化检测
    // （has_degenerate_reasoning_repetition 依赖连续重复）；Snapshot 累积去重后缀
    // （快照是已见文本的重发，原文已在 Append 阶段累积过）。
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

        // 某些长工具链上下文会诱发模型在 thinking 中逐字复读同一句话。继续读取只会
        // 消耗输出预算并让终端看似卡死；升级为可重试截断，交给上层降低推理档位。
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

    // 增量事件保留模型原始文本；快照事件仅渲染未见后缀，避免协议重发。
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
        // 模型在可见正文里自编自演「工具调用→工具结果」，吐出系统从不生成的内部
        // 协议标记（`<function_results>` 等）。streamer 已把整块剥离，这里停流并复用
        // degenerate_repetition 降档重试路径，避免幻觉正文落盘毒化下一轮请求。这是
        // 零误伤信号：合法重复代码/措辞永不含内部协议标记，无需任何文本统计阈值。
        state.content.finish_reason_seen = true;
        state.content.finish_reason_value = Some(DEGENERATE_REPETITION_FINISH_REASON.to_string());
        if runtime_ctx::terminal_output_enabled() {
            eprintln!("\n  ⚠ 检测到模型伪造工具结果标记（输出退化），停止当前响应并自动重试…");
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
                let assistant_len_before = state.content.assistant_text.len();
                commit_visible_content(app, current_history, markers, state, content)?;
                meaningful_progress |= state.content.assistant_text.len() > assistant_len_before;

                // 与 reasoning 路径对称：模型也会在**可见输出**里退化成逐字复读同一
                // 短语（本次事故即 assistant content 复读「我再重新读一遍…」直到撑满
                // 输出预算，产出 16 万字符垃圾并落盘，毒化下一轮请求触发 provider
                // 400 InvalidParameter）。此前退化守卫只挂在 reasoning_content 上，
                // 可见文本完全失守。命中即置 finish_reason 并停流，交给上层降档重试。
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

/// OpenCode 等兼容网关偶尔会把完整 DSML 工具协议作为单个 content 快照返回。
/// 在正文提交和终端渲染前识别这种完整包裹，避免只能在流末 fallback 恢复。
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
        // `.done` 快照会重发此前 delta 已解析的完整协议；只过滤语义相同的调用，
        // 不能因已有其他工具调用就误吞真正新增的并行调用。
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

/// 检测文本尾部是否出现三次连续、完全相同的长片段（退化复读循环）。
///
/// 同时用于 reasoning_content 与可见 assistant 输出：模型在长工具链上下文下可能
/// 在思考链或正文里逐字复读同一句话，继续读取只会耗尽输出预算并把垃圾落盘。
/// 使用字符而非字节比较以正确处理中文；要求片段包含足够多的字母、数字或中文等实际
/// 内容，避免把分隔线、空白或 Markdown 标点误判为退化循环。
fn has_degenerate_repetition(text: &str) -> bool {
    // 每个流 chunk 都会调用该检测，因此只保留足以覆盖最大候选片段的尾部，避免长推理
    // 随上下文增长退化成反复扫描整段文本。
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
            // digest 是给模型看的附加图片理解内容，thinking 通道的终端展示同样剥离
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
    // Thinking 事件只允许走 render_thinking_event() 的终端展示路径，不能进入
    // assistant_text/current_history。这里故意只返回可见正文。
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
        // 纯空白（如 \n）的重叠几乎总是伪匹配——模型常以 \n 开始新段落，
        // 而 assistant_text 也常以 \n 结尾。只有含可见字符的重叠才视为真正的重复。
        if existing.ends_with(overlap) && overlap.chars().any(|c| !c.is_whitespace()) {
            return Some(incoming[split_idx..].to_string());
        }
    }

    None
}

/// 空白容忍的后缀去重。
///
/// 兼容旧历史/异常 provider 产生的空白差异：当 `assistant_text` 与最终
/// `response.output_text.done` 快照仅在空白上不一致时，仍避免把已流式输出的整段
/// 快照当作新内容重复追加。
///
/// 这里按"空白可跳过"的方式逐字符对齐 existing 与 incoming，找出 incoming 中已被
/// existing 覆盖的前缀，返回 incoming 剩余的（保留原始空白）尾部。若 incoming 的可见
/// 字符已全部被覆盖则返回 `Some("")`；若两者在可见字符上无法对齐则返回 `None`。
fn unseen_suffix_whitespace_tolerant(existing: &str, incoming: &str) -> Option<String> {
    let e: Vec<(usize, char)> = existing.char_indices().collect();
    let i: Vec<(usize, char)> = incoming.char_indices().collect();
    let (mut ei, mut ii) = (0usize, 0usize);

    // 跳过 incoming 开头的空白（快照常以换行开头，而 assistant_text 没有）
    while ii < i.len() && i[ii].1.is_whitespace() {
        ii += 1;
    }

    // last_matched_ii 记录 incoming 中最后一个被匹配上的可见字符之后的 Vec 下标，
    // 用于在结束时定位"剩余未覆盖尾部"在 incoming 中的字节起点。
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
        // 两侧都是可见字符：必须相等才算对齐
        if ec == ic {
            last_matched_ii = ii + 1;
            ei += 1;
            ii += 1;
        } else {
            return None;
        }
    }

    // existing 已耗尽；incoming 中 last_matched_ii 之后的字节即为剩余（未覆盖）尾部。
    // 若 incoming 也已全部匹配则 start_byte == incoming.len()，返回空串。
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
