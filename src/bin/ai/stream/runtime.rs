use super::inline_recovery::{
    collect_valid_tool_calls, ensure_tool_calls_section_open, recover_inline_tool_calls,
};
use std::io::{self, IsTerminal, Write};
use std::time::{Duration, Instant};

use crate::ai::{
    config_schema::AiConfig,
    driver::runtime_ctx,
    models,
    provider::{self, ProviderAdapter},
    request::StreamChunk,
    theme::{ACCENT_MUTED, ACCENT_RULE, DIM, RESET},
    types::{App, StreamOutcome, StreamResult, take_stream_cancelled},
};
use crate::commonw::configw;

use super::{
    MarkdownStreamRenderer,
    extract::{StreamTextEvent, extract_chunk_events_streaming},
    framing, normalize,
    render::markdown::{clamp_line_to_terminal_row, live_preview_cursor_rows},
    splitter::{InternalToolCallStreamEvent, StreamSplitSegment},
    state::{StreamChunkStep, StreamMarkers, StreamProcessingState, ToolCallBuilder},
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
const DEFAULT_THINKING_MAX_VISIBLE_LINES: usize = 4;
/// 推理流连续重复的最短片段和判定次数。只检测 reasoning，避免把用户要求生成的重复
/// 正文（表格、代码、测试数据等）误判为模型退化。
const MIN_REASONING_REPEAT_CHARS: usize = 16;
const MAX_REASONING_REPEAT_CHARS: usize = 512;
const REASONING_REPEAT_COUNT: usize = 3;
const DEGENERATE_REPETITION_FINISH_REASON: &str = "degenerate_repetition";

pub(super) async fn stream_response(
    app: &mut App,
    response: &mut reqwest::Response,
    current_history: &mut String,
    _terminal_dedupe_candidate: Option<&str>,
) -> Result<StreamResult, Box<dyn std::error::Error>> {
    let mut markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    configure_thinking_fold(&mut state);
    configure_subagent_preview_fold(app, &mut state, &mut markers);
    let adapter = provider::adapter_for(
        models::model_adapter(&app.current_model),
        &models::endpoint_for_model(&app.current_model, &app.config.endpoint),
    );

    if should_show_waiting_hint(app) {
        print_waiting_hint(&mut state)?;
    }

    let mut last_chunk_at = Instant::now();
    let has_content = |s: &StreamProcessingState| -> bool {
        !s.content.assistant_text.is_empty()
            || !s.content.reasoning_text.is_empty()
            || !s.content.tool_calls_map.is_empty()
    };

    while !app.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        if let Some(result) = immediate_cancel_result(app, &mut state) {
            return Ok(result);
        }

        // 已有内容时用较短的 idle 超时，无内容时用较长的首 chunk 超时
        let timeout_secs = if has_content(&state) {
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
            let idle_remaining =
                Duration::from_secs(timeout_secs).saturating_sub(last_chunk_at.elapsed());
            tokio::select! {
                chunk = response.chunk() => chunk,
                _ = wait_for_interrupt(app) => {
                    return Ok(cancelled_stream_result(&mut state));
                }
                _ = tokio::time::sleep(idle_remaining) => {
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
            StreamChunkStep::Continue => {
                last_chunk_at = Instant::now();
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

    finalize_stream_response(app, &markers, state)
}

/// 是否在终端显示「等待模型输出」的紧凑状态提示。
/// 对所有 TTY 会话生效。println! 保证提示立即可见，
/// 收到首个可见 chunk 时用 \x1b[1A\r\x1b[2K 清掉，不残留额外行。
fn should_show_waiting_hint(app: &App) -> bool {
    io::stdout().is_terminal() && !app.shutdown.load(std::sync::atomic::Ordering::Relaxed)
}

fn print_waiting_hint(state: &mut StreamProcessingState) -> io::Result<()> {
    if state.render.waiting_hint_active {
        return Ok(());
    }
    // 独立行等待提示：println! 保证终端立即显示，首个 chunk 到达时用 \x1b[1A\r\x1b[2K 清掉
    println!("⠋ waiting…");
    io::stdout().flush()?;
    state.render.waiting_hint_active = true;
    Ok(())
}

fn configure_thinking_fold(state: &mut StreamProcessingState) {
    state.render.thinking_fold.max_visible_lines = resolve_thinking_fold_max_visible_lines(
        io::stdout().is_terminal(),
        configw::get_all_config()
            .get_opt(AiConfig::OUTPUT_THINKING_MAX_VISIBLE_LINES)
            .as_deref(),
    );
}

fn configure_subagent_preview_fold(
    app: &App,
    state: &mut StreamProcessingState,
    markers: &mut StreamMarkers,
) {
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
    if !state.render.waiting_hint_active || state.render.waiting_hint_buffering {
        return Ok(());
    }
    // 光标上移+清行后重新 println!，保证 buffering 状态在独立行可见
    print!("\x1b[1A\r\x1b[2K");
    println!("⠋ buffering…");
    io::stdout().flush()?;
    state.render.waiting_hint_buffering = true;
    Ok(())
}

pub(super) fn clear_waiting_hint(state: &mut StreamProcessingState) -> io::Result<()> {
    if !state.render.waiting_hint_active {
        return Ok(());
    }
    // 光标上移一行 + \r + 清行：擦掉 println! 输出的独立提示行，让内容在原位输出
    print!("\x1b[1A\r\x1b[2K");
    io::stdout().flush()?;
    state.render.waiting_hint_active = false;
    state.render.waiting_hint_buffering = false;
    Ok(())
}

fn immediate_cancel_result(app: &App, state: &mut StreamProcessingState) -> Option<StreamResult> {
    stream_interrupt_requested(app).then(|| cancelled_stream_result(state))
}

/// 取消/中断时的 thinking 折叠收尾：折叠窗口若仍活跃，必须先擦掉当前窗口并落一个
/// `╰─ done thinking` 收口，否则半截 `╭─ thinking` 窗口会被留在屏幕上，下一轮重试
/// 的 fresh state 会在其下方再画一个新 header——累积成「重复 header + 大段空白」。
fn cancelled_stream_result(state: &mut StreamProcessingState) -> StreamResult {
    let _ = clear_waiting_hint(state);
    if state.render.thinking_fold.active {
        let _ = finalize_thinking_fold(state);
    } else if state.render.subagent_fold.active {
        let _ = finalize_subagent_preview_fold(state);
    } else if state.content.thinking_open {
        print!("\x1b[0m");
        let _ = io::stdout().flush();
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
                Ok(StreamChunkStep::Continue)
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
    for line in lines {
        if process_stream_line(app, current_history, markers, state, adapter, &line)? {
            should_stop = true;
            break;
        }
    }
    Ok(if should_stop {
        StreamChunkStep::Stop
    } else {
        StreamChunkStep::Continue
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
            if flush_sse_event(app, current_history, markers, state, adapter)? {
                let final_state = std::mem::replace(state, StreamProcessingState::new());
                return Ok(Some(finalize_stream_response(app, markers, final_state)?));
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
    if flush_sse_event(app, current_history, markers, state, adapter)? {
        let final_state = std::mem::replace(state, StreamProcessingState::new());
        return Ok(Some(finalize_stream_response(app, markers, final_state)?));
    }
    Ok(None)
}

fn finalize_stream_response(
    app: &mut App,
    markers: &StreamMarkers,
    mut state: StreamProcessingState,
) -> Result<StreamResult, Box<dyn std::error::Error>> {
    clear_waiting_hint(&mut state)?;

    if state.content.thinking_open {
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

    if state.render.subagent_fold.active {
        finalize_subagent_preview_fold(&mut state)?;
    }

    flush_terminal_splitter(&mut state, markers)?;

    state.render.markdown.flush_pending()?;

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

    let (mut tool_calls, dropped_malformed) =
        collect_valid_tool_calls(&mut state.content.tool_calls_map);
    state.content.dropped_malformed_tool_call = dropped_malformed;

    // Fallback：部分 provider（已知 qwen3.7-max thinking 模式）会把 function call
    // 以纯 content JSON 的形式输出而不走 delta.tool_calls[]，导致 turn 在打印完
    // 那段 JSON 后被判定为 Completed 直接结束。这里做一次保守的反向识别：仅当
    // assistant_text 整体（去掉代码围栏 / <tool_call> 标签后）就是一个含 name 和
    // arguments 的 JSON 对象/数组时，才升级为 tool_call。
    if tool_calls.is_empty() {
        if let Some(recovered) = recover_inline_tool_calls(&state.content.assistant_text) {
            tool_calls = recovered;
            // 把误打成 content 的那段 JSON 从可见输出里挪走，避免被持久化为
            // 真正的 assistant 文本——否则下一轮请求模型会看到"自己上一轮回答
            // 了一段 JSON"，进一步混乱。
            let stripped = std::mem::take(&mut state.content.assistant_text);
            if state.content.hidden_meta.is_empty() {
                state.content.hidden_meta = stripped;
            } else {
                state.content.hidden_meta.push('\n');
                state.content.hidden_meta.push_str(&stripped);
            }
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

    let outcome = if !tool_calls.is_empty() {
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
        stream_error: false,
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
        use colored::Colorize;
        println!("{}", line.dimmed());
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
        "[prompt cache] {cached_tokens}/{prompt_tokens} prompt tokens cached ({pct:.0}% hit)"
    ))
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
                _ = crate::ai::driver::signal::wait_for_interrupt_sources(None, None) => true,
            }
        }
        None => {
            crate::ai::driver::signal::wait_for_interrupt_sources(None, None).await;
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
    eprintln!(
        "[Warning] 读取响应流时出错：{} (错误次数：{}/{})",
        err, state.framing.decode_error_count, MAX_DECODE_ERRORS
    );

    if take_stream_cancelled(app) {
        return Some(cancelled_stream_result(state));
    }

    if state.framing.decode_error_count <= MAX_DECODE_ERRORS {
        eprintln!("[Warning] 尝试继续读取...");
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

    eprintln!("[Error] 响应流读取失败，返回已收集的内容");

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

    let _ = state.render.markdown.flush_pending();

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
    _function_name: &str,
) -> io::Result<()> {
    state.render.current_printing_index = Some(index);
    Ok(())
}

/// 终端不再打印工具调用参数，只保留工具名称。
fn write_tool_call_arguments_stream(_arguments: &str) -> io::Result<()> {
    Ok(())
}

fn process_external_tool_calls_delta(
    app: &mut App,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    chunk: &StreamChunk,
    merge_mode: StreamEventMergeMode,
) {
    let Some(choice) = chunk.choices.first() else {
        return;
    };

    for stream_tool_call in &choice.delta.tool_calls {
        let index = stream_tool_call.index;
        ensure_tool_calls_section_open(app, markers, state);

        let render_chunk = {
            let builder = state.content.tool_calls_map.entry(index).or_default();
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
            take_tool_call_render_chunk(state.render.current_printing_index, index, builder)
        };

        if let Some(render_chunk) = render_chunk {
            if render_chunk.open_line {
                let _ = open_tool_call_line(state, index, &render_chunk.function_name);
            }
            let _ = write_tool_call_arguments_stream(&render_chunk.arguments);
        }
    }
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

fn process_internal_tool_calls(
    app: &mut App,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    internal_tool_call_events: Vec<InternalToolCallStreamEvent>,
) {
    for event in internal_tool_call_events {
        match event {
            InternalToolCallStreamEvent::Begin(function_name) => {
                if function_name.trim().is_empty() {
                    continue;
                }
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
                    // 复位颜色即可。绝不能用 println!——那会在「done thinking」与后续
                    // 输出之间凭空插入一行空行（外部 delta 工具路径本就不打这行）。
                    print!("\x1b[0m");
                    state.render.current_printing_index = None;
                    let _ = io::stdout().flush();
                }
                state.content.internal_tool_call_idx += 1;
            }
        }
    }
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

    normalize_end_thinking_boundary(&mut content, markers, &state.render.markdown);

    clear_waiting_hint(state)?;

    // 当 thinking 折叠模式活跃且遇到 end_thinking_tag 时，做最终的折叠渲染
    if !state.content.thinking_open
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

    if markers.subagent_preview_enabled() {
        write_subagent_content_folded(content.as_str(), state)?;
    } else {
        maybe_write_stream_content(
            content.as_str(),
            state,
            markers,
            state.content.thinking_open,
        )?;
    }
    if state.content.thinking_open {
        return Ok(());
    }

    let text = if is_standalone_stream_marker(&content, &markers.end_thinking_tag) {
        String::new()
    } else {
        content
    };
    let text = text.trim_matches('\n');
    current_history.reserve(text.len());
    state.content.assistant_text.reserve(text.len());
    current_history.push_str(text);
    state.content.assistant_text.push_str(text);

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
        } => write_stream_content_to_terminal(&text, &mut state.render.markdown, false),
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

    write_stream_content_to_terminal(content, &mut state.render.markdown, false)
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
/// header（`╭─ thinking`）在折叠激活时打印一次并锚定在正文之上，之后每次重画都只
/// 擦除并重写正文。正文行数上限为 `max_visible_lines + 1`（折叠摘要），恒定落在可视
/// 视口内，因此 `\x1b[{window_rows}A` 相对擦除永远够得着，不会随窗口滚入 scrollback
/// 而失步——即便失步，也无法再生出第二个 header，从根上杜绝「孤儿 header 叠加」。
fn thinking_fold_redraw(fold: &mut super::state::ThinkingFoldState) -> io::Result<()> {
    let mut out = io::stdout();
    // 只擦除上一次的正文区域；header 已锚定在其上方，绝不触碰。
    // 注意：terminal 缩窄后，旧正文会被终端按当前宽度自动 reflow 成更多物理行；
    // 这里不能只信缓存的 window_rows，而要按"上次正文在当前宽度下会占几行"重算。
    let erase_rows = thinking_fold_rendered_body_rows(fold).max(fold.window_rows);
    if erase_rows > 0 {
        write!(out, "\x1b[{}A\r\x1b[0J", erase_rows)?;
    }
    if fold.active && !fold.header_drawn {
        write_fold_header(&mut out, fold)?;
        fold.header_drawn = true;
    }

    let (body_lines, marker_lines) = thinking_fold_window_lines(fold);
    let (body, body_rows) = render_thinking_fold_window_lines(&body_lines, marker_lines);
    if !body.is_empty() {
        out.write_all(body.as_bytes())?;
    }
    out.flush()?;
    fold.window_rows = body_rows;
    fold.rendered_body_lines = body_lines;
    Ok(())
}

/// 打印锚定的折叠 header。只应在折叠激活后被调用一次。
fn write_fold_header(
    out: &mut impl Write,
    fold: &super::state::ThinkingFoldState,
) -> io::Result<()> {
    write!(
        out,
        "{ACCENT_RULE}╭─\x1b[0m {ACCENT_MUTED}{}\x1b[0m\n",
        fold.header_label
    )
}

/// Thinking 结束时的最终渲染：覆盖正文窗口，输出最终折叠摘要 + "done thinking"。
pub(super) fn finalize_thinking_fold(state: &mut StreamProcessingState) -> io::Result<()> {
    finalize_fold(&mut state.render.thinking_fold)
}

fn finalize_subagent_preview_fold(state: &mut StreamProcessingState) -> io::Result<()> {
    finalize_fold(&mut state.render.subagent_fold)
}

fn finalize_fold(fold: &mut super::state::ThinkingFoldState) -> io::Result<()> {
    if !fold.active {
        return Ok(());
    }

    let mut out = io::stdout();
    let erase_rows = thinking_fold_rendered_body_rows(fold).max(fold.window_rows);
    if erase_rows > 0 {
        write!(out, "\x1b[{}A\r\x1b[0J", erase_rows)?;
    }
    // 若正文从未渲染过（header 尚未落地），补一个 header 以保证块结构完整。
    if !fold.header_drawn {
        write_fold_header(&mut out, fold)?;
        fold.header_drawn = true;
    }

    let (body_lines, marker_lines) = thinking_fold_window_lines(fold);
    let (body, body_rows) = render_thinking_fold_window_lines(&body_lines, marker_lines);
    if !body.is_empty() {
        out.write_all(body.as_bytes())?;
    }
    fold.window_rows = body_rows;
    fold.rendered_body_lines = body_lines;

    // "done thinking" 结尾标记
    write!(
        out,
        "{ACCENT_RULE}╰─\x1b[0m {ACCENT_MUTED}{}\x1b[0m\n",
        fold.footer_label
    )?;
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
        lines.push(clamp_line_to_terminal_row(&format!(
            "  ··· {hidden_count} lines folded ···"
        )));
    }
    for line in visible_lines {
        lines.push(clamp_line_to_terminal_row(line));
    }
    (lines, marker_lines)
}

/// 渲染折叠窗口的**正文**（折叠摘要 + 最近可见行），不含 header。
/// header 由 `write_thinking_fold_header` 单独锚定打印。返回的行数即正文物理行数，
/// 供 `\x1b[{n}A` 精确擦除；正文行数上限为 `max_visible_lines + 1`，恒在视口内。
fn render_thinking_fold_window(fold: &super::state::ThinkingFoldState) -> (String, usize) {
    let (lines, marker_lines) = thinking_fold_window_lines(fold);
    render_thinking_fold_window_lines(&lines, marker_lines)
}

fn render_thinking_fold_window_lines(lines: &[String], marker_lines: usize) -> (String, usize) {
    if lines.is_empty() {
        return (String::new(), 0);
    }
    let mut out = String::new();
    // 每条可见行都被 clamp 成「最多占一个物理行」，因此窗口物理行数恒等于逻辑行数。
    // cursor-up 擦除据此精确，不再依赖对自动折行的预测（tab/CJK/超长行/resize 免疫）。
    let rows = lines.len();

    for (idx, line) in lines.iter().enumerate() {
        if idx < marker_lines {
            out.push_str(ACCENT_MUTED);
            out.push_str(line);
            out.push_str("\x1b[0m\n");
        } else {
            out.push_str(DIM);
            out.push_str(line);
            out.push_str(RESET);
            out.push('\n');
        }
    }

    (out, rows)
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
) -> Result<bool, Box<dyn std::error::Error>> {
    let (mut chunk, merge_mode) =
        match normalize::parse_stream_payload(adapter, payload, event_type) {
            super::state::ParsedStreamPayload::Ignore => return Ok(false),
            super::state::ParsedStreamPayload::Done => return Ok(true),
            super::state::ParsedStreamPayload::Error(msg) => {
                return Err(format!("provider stream error: {msg}").into());
            }
            super::state::ParsedStreamPayload::ReasoningItem(item) => {
                // 捕获完整 reasoning item（含 encrypted_content）供同 turn 工具链回放。
                // 不产生可见输出，也不进持久化历史。
                state.content.reasoning_items.push(item);
                return Ok(false);
            }
            super::state::ParsedStreamPayload::Chunk(chunk) => {
                (chunk, StreamEventMergeMode::Append)
            }
            super::state::ParsedStreamPayload::SnapshotChunk(chunk) => {
                (chunk, StreamEventMergeMode::AppendMissingSuffix)
            }
        };

    // AIOS: capture usage block from whichever chunk carries it. OpenAI emits
    // the final `usage` on a chunk with `choices: []`, so we must pull it *before*
    // the empty-choices early return below.
    if let Some(ref usage) = chunk.usage {
        state.pending_llm_usage = Some((chunk.model.clone(), usage.clone().normalized()));
    }

    if chunk.choices.iter().any(|choice| {
        choice
            .finish_reason
            .as_deref()
            .is_some_and(|reason| !reason.trim().is_empty())
    }) {
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
        return Ok(false);
    }

    state.content.empty_choice_chunks = 0;

    // `.done` reasoning 事件会把整段推理摘要作为 SnapshotChunk 再发一次（见
    // normalize::textual_event_chunk）。可见 assistant 文本靠 stream_text_event_to_content
    // 的 unseen_suffix 去重，但 reasoning 文本在累积（下方 reasoning_text.push_str）与
    // extract_chunk_events_streaming 里都按 Append 处理，snapshot 会导致 thinking 重复
    // 输出。这里在两处消费前，把 snapshot 的 reasoning_content 收敛为未见后缀。
    if matches!(merge_mode, StreamEventMergeMode::AppendMissingSuffix) {
        if let Some(choice) = chunk.choices.first_mut() {
            if !choice.delta.reasoning_content.is_empty() {
                let suffix = unseen_suffix(
                    &state.content.reasoning_text,
                    &choice.delta.reasoning_content,
                );
                choice.delta.reasoning_content = suffix;
            }
        }
    }

    if let Some(choice) = chunk.choices.first() {
        if !choice.delta.reasoning_content.is_empty() {
            state.content.saw_reasoning_output = true;
            state
                .content
                .reasoning_text
                .push_str(&choice.delta.reasoning_content);

            // 某些长工具链上下文会诱发模型在 thinking 中逐字复读同一句话。继续读取只会
            // 消耗输出预算并让终端看似卡死；升级为可重试截断，交给上层降低推理档位。
            if has_degenerate_reasoning_repetition(&state.content.reasoning_text) {
                state.content.finish_reason_seen = true;
                state.content.finish_reason_value =
                    Some(DEGENERATE_REPETITION_FINISH_REASON.to_string());
                eprintln!("\n  ⚠ 检测到模型推理重复循环，停止当前响应并自动重试…");
                return Ok(true);
            }
        }
    }

    process_external_tool_calls_delta(app, markers, state, &chunk, merge_mode);

    let (events, internal_tool_call_events) = extract_chunk_events_streaming(
        &chunk,
        markers.hidden_begin,
        markers.hidden_end,
        &mut state.content.thinking_open,
        &mut state.content.hidden_meta_parse,
        &mut state.content.internal_tool_call_streamer,
        &mut state.content.hermes_tool_call_streamer,
        &mut state.content.anthropic_tool_call_streamer,
        &mut state.content.bare_xml_tool_call_streamer,
    );
    process_internal_tool_calls(app, markers, state, internal_tool_call_events);

    if events.is_empty() {
        return Ok(false);
    }
    for event in events {
        match event {
            StreamTextEvent::AppendHiddenMeta(text) => {
                state.content.hidden_meta.push_str(&text);
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
                commit_visible_content(app, current_history, markers, state, content)?;
            }
        }
    }

    // Keep streaming until explicit stream end ([DONE]/EOF) or the outer loop's
    // short post-finish grace window expires. Some providers can set
    // finish_reason before all visible content chunks are delivered.
    Ok(false)
}

/// 检测 reasoning 尾部是否出现三次连续、完全相同的长片段。
///
/// 使用字符而非字节比较以正确处理中文；要求片段包含足够多的字母、数字或中文等实际
/// 内容，避免把分隔线、空白或 Markdown 标点误判为退化循环。
fn has_degenerate_reasoning_repetition(text: &str) -> bool {
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

fn stream_text_event_to_content(
    event: &StreamTextEvent,
    markers: &StreamMarkers,
    merge_mode: StreamEventMergeMode,
    assistant_text: &str,
) -> Option<String> {
    if markers.subagent_preview_enabled() {
        return match event {
            StreamTextEvent::OpenThinking | StreamTextEvent::CloseThinking => None,
            StreamTextEvent::AppendThinking(text) => (!text.is_empty()).then(|| text.clone()),
            StreamTextEvent::AppendContent(text) => match merge_mode {
                StreamEventMergeMode::Append => (!text.is_empty()).then(|| text.clone()),
                StreamEventMergeMode::AppendMissingSuffix => {
                    let suffix = unseen_suffix(assistant_text, text);
                    (!suffix.is_empty()).then_some(suffix)
                }
            },
            StreamTextEvent::AppendHiddenMeta(_) => None,
        };
    }

    match event {
        StreamTextEvent::OpenThinking => Some(format!("\n{}\n", markers.thinking_tag)),
        StreamTextEvent::AppendThinking(text) => (!text.is_empty()).then(|| text.clone()),
        StreamTextEvent::AppendContent(text) => match merge_mode {
            StreamEventMergeMode::Append => (!text.is_empty()).then(|| text.clone()),
            StreamEventMergeMode::AppendMissingSuffix => {
                let suffix = unseen_suffix(assistant_text, text);
                (!suffix.is_empty()).then_some(suffix)
            }
        },
        StreamTextEvent::AppendHiddenMeta(_) => None,
        StreamTextEvent::CloseThinking => Some(format!("{}\n", markers.end_thinking_tag)),
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
/// `commit_visible_content` 会对每段内容做 `trim_matches('\n')`，纯空白 delta
/// 也不会累积进 `assistant_text`，因此 `assistant_text` 相对 `response.output_text.done`
/// 快照会丢失段间换行。此时精确的 `ends_with` 与 `unseen_suffix_after_visible_overlap`
/// 都会失败，导致整段已流式输出的快照被当作新内容重复输出。
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
) -> Result<bool, Box<dyn std::error::Error>> {
    let Some(event) = framing::flush_sse_event(&mut state.framing) else {
        return Ok(false);
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
) -> Result<bool, Box<dyn std::error::Error>> {
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

    Ok(false)
}

pub(super) fn write_stream_content(
    content: &str,
    markdown: &mut MarkdownStreamRenderer,
    dimmed: bool,
) -> io::Result<()> {
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
