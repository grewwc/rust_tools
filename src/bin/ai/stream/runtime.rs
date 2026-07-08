use std::io::{self, IsTerminal, Write};
use super::inline_recovery::{
    collect_valid_tool_calls, ensure_tool_calls_section_open, recover_inline_tool_calls,
};
use std::time::{Duration, Instant};

use crate::ai::{
    config_schema::AiConfig,
    models,
    request::StreamChunk,
    theme::{ACCENT_MUTED, ACCENT_RULE, DIM, RESET},
    types::{App, StreamOutcome, StreamResult, take_stream_cancelled},
};
use crate::commonw::configw;

use super::{
    MarkdownStreamRenderer,
    extract::{StreamTextEvent, extract_chunk_events_streaming},
    framing, normalize,
    render::markdown::clamp_line_to_terminal_row,
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

pub(super) async fn stream_response(
    app: &mut App,
    response: &mut reqwest::Response,
    current_history: &mut String,
    _terminal_dedupe_candidate: Option<&str>,
) -> Result<StreamResult, Box<dyn std::error::Error>> {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    configure_thinking_fold(&mut state);
    let adapter_kind = normalize::resolve_adapter_kind(
        models::model_provider(&app.current_model),
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
            adapter_kind,
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
        process_pending_tail(app, current_history, &markers, &mut state, adapter_kind).await?
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

fn immediate_cancel_result(
    app: &App,
    state: &mut StreamProcessingState,
) -> Option<StreamResult> {
    stream_interrupt_requested(app).then(|| cancelled_stream_result(state))
}

/// 取消/中断时的 thinking 折叠收尾：折叠窗口若仍活跃，必须先擦掉当前窗口并落一个
/// `╰─ done thinking` 收口，否则半截 `╭─ thinking` 窗口会被留在屏幕上，下一轮重试
/// 的 fresh state 会在其下方再画一个新 header——累积成「重复 header + 大段空白」。
fn cancelled_stream_result(state: &mut StreamProcessingState) -> StreamResult {
    let _ = clear_waiting_hint(state);
    if state.render.thinking_fold.active {
        let _ = finalize_thinking_fold(state);
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
        skip_response_drain: true,
        truncated_by_length: false,
    }
}

async fn process_chunk_result<T: AsRef<[u8]>>(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter_kind: normalize::StreamProviderAdapterKind,
    chunk_result: Result<Option<T>, reqwest::Error>,
) -> Result<StreamChunkStep, Box<dyn std::error::Error>> {
    match chunk_result {
        Ok(Some(chunk)) => {
            framing::push_chunk(&mut state.framing, chunk.as_ref());
            state.framing.decode_error_count = 0;
            consume_pending_complete_lines(app, current_history, markers, state, adapter_kind).await
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
    adapter_kind: normalize::StreamProviderAdapterKind,
) -> Result<StreamChunkStep, Box<dyn std::error::Error>> {
    // Move the pending buffer out so line slices can borrow from it while `state`
    // remains available for mutation inside `process_stream_line()`.
    let lines = framing::take_complete_lines(&mut state.framing);
    let mut should_stop = false;
    for line in lines {
        if process_stream_line(app, current_history, markers, state, adapter_kind, &line)? {
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
    adapter_kind: normalize::StreamProviderAdapterKind,
) -> Result<Option<StreamResult>, Box<dyn std::error::Error>> {
    if state.framing.pending.is_empty() {
        // pending 为空时，仍需检查 sse_event_data 是否有未 flush 的最后一个事件。
        // 部分 provider 在关闭连接前不发送最终空行（\n\n），导致最后一个 SSE 事件被丢弃。
        if !state.framing.sse_event_data.trim().is_empty() {
            if flush_sse_event(app, current_history, markers, state, adapter_kind)? {
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
        let _ = process_stream_line(app, current_history, markers, state, adapter_kind, &line)?;
    }
    if flush_sse_event(app, current_history, markers, state, adapter_kind)? {
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

    flush_terminal_splitter(&mut state, markers)?;

    state.render.markdown.flush_pending()?;

    if take_stream_cancelled(app) {
        return Ok(cancelled_stream_result(&mut state));
    }

    // AIOS: flush any pending LLM usage to kernel `/dev/llm` before returning.
    // Prefer the model echoed by the provider; fall back to what we requested.
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
        if state.content.dropped_malformed_tool_call {
            StreamOutcome::Truncated
        } else if truncated_by_length && !has_text {
            // finish_reason=length 且没有可见文本：模型可能只产出了 reasoning
            // 就被掐断，或根本没输出。降 effort 重试，把预算让给实际内容。
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
        skip_response_drain: true,
        truncated_by_length,
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
        skip_response_drain: true,
        truncated_by_length: false,
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

    maybe_write_stream_content(
        content.as_str(),
        state,
        markers,
        state.content.thinking_open,
    )?;
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

pub(super) fn format_end_thinking_line(markers: &StreamMarkers, markdown: &MarkdownStreamRenderer) -> String {
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

    // 逐字符处理
    for ch in content.chars() {
        if ch == '\n' {
            // 一行完成
            let completed_line = std::mem::take(&mut fold.current_line);
            // 折叠视图内跳过空行（无论开头还是段间）：模型推理常用空行分段，
            // 但紧凑折叠窗口里空行纯属噪音，会平白占一行可见预算。原文仍完整
            // 保留在 reasoning_text，不影响回传后端。
            if !completed_line.trim().is_empty() {
                fold.total_lines += 1;
                fold.recent_lines.push_back(completed_line);
                // 只保留最近 max_visible_lines 行
                while fold.recent_lines.len() > fold.max_visible_lines {
                    fold.recent_lines.pop_front();
                }
            }
        } else {
            fold.current_line.push(ch);
        }
    }

    thinking_fold_redraw(fold)
}

/// 只覆盖 thinking header 下方的固定窗口区域。
fn thinking_fold_redraw(fold: &mut super::state::ThinkingFoldState) -> io::Result<()> {
    let mut out = io::stdout();
    if fold.window_rows > 0 {
        write!(out, "\x1b[{}A\r\x1b[0J", fold.window_rows)?;
    }

    let (window, window_rows) = render_thinking_fold_window(fold);
    if !window.is_empty() {
        out.write_all(window.as_bytes())?;
        out.flush()?;
    }
    fold.window_rows = window_rows;
    Ok(())
}

/// Thinking 结束时的最终渲染：覆盖底部窗口，输出最终折叠摘要 + "done thinking"。
pub(super) fn finalize_thinking_fold(state: &mut StreamProcessingState) -> io::Result<()> {
    let fold = &mut state.render.thinking_fold;
    if !fold.active {
        return Ok(());
    }

    let mut out = io::stdout();
    if fold.window_rows > 0 {
        write!(out, "\x1b[{}A\r\x1b[0J", fold.window_rows)?;
    }

    let (window, _) = render_thinking_fold_window(fold);
    if !window.is_empty() {
        out.write_all(window.as_bytes())?;
    }

    // "done thinking" 结尾标记
    write!(
        out,
        "{ACCENT_RULE}╰─\x1b[0m {ACCENT_MUTED}done thinking\x1b[0m\n"
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

fn thinking_fold_visible_lines(
    fold: &super::state::ThinkingFoldState,
) -> Vec<&str> {
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

fn render_thinking_fold_window(
    fold: &super::state::ThinkingFoldState,
) -> (String, usize) {
    let hidden_count = thinking_fold_hidden_count(fold);
    let visible_lines = thinking_fold_visible_lines(fold);
    if !fold.active && hidden_count == 0 && visible_lines.is_empty() {
        return (String::new(), 0);
    }

    let mut out = String::new();
    // 每条可见行都被 clamp 成「最多占一个物理行」，因此窗口物理行数恒等于逻辑行数。
    // cursor-up 擦除据此精确，不再依赖对自动折行的预测（tab/CJK/超长行/resize 免疫）。
    let mut rows = 0;

    if fold.active {
        let header = format!("{ACCENT_RULE}╭─\x1b[0m {ACCENT_MUTED}thinking\x1b[0m");
        rows += 1;
        out.push_str(&header);
        out.push('\n');
    }

    if hidden_count > 0 {
        let marker = clamp_line_to_terminal_row(&format!("  ··· {hidden_count} lines folded ···"));
        rows += 1;
        out.push_str(ACCENT_MUTED);
        out.push_str(&marker);
        out.push_str("\x1b[0m\n");
    }

    for line in visible_lines {
        let clamped = clamp_line_to_terminal_row(line);
        rows += 1;
        out.push_str(DIM);
        out.push_str(&clamped);
        out.push_str(RESET);
        out.push('\n');
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
    adapter_kind: normalize::StreamProviderAdapterKind,
    event_type: Option<&str>,
    payload: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let (chunk, merge_mode) =
        match normalize::parse_stream_payload(adapter_kind, payload, event_type) {
            super::state::ParsedStreamPayload::Ignore => return Ok(false),
            super::state::ParsedStreamPayload::Done => return Ok(true),
            super::state::ParsedStreamPayload::Error(msg) => {
                return Err(format!("provider stream error: {msg}").into());
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

    if let Some(choice) = chunk.choices.first() {
        if !choice.delta.reasoning_content.is_empty() {
            state.content.saw_reasoning_output = true;
            state
                .content
                .reasoning_text
                .push_str(&choice.delta.reasoning_content);
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
            return incoming[split_idx..].to_string();
        }
    }

    incoming.to_string()
}

fn flush_sse_event(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter_kind: normalize::StreamProviderAdapterKind,
) -> Result<bool, Box<dyn std::error::Error>> {
    let Some(event) = framing::flush_sse_event(&mut state.framing) else {
        return Ok(false);
    };
    process_stream_payload(
        app,
        current_history,
        markers,
        state,
        adapter_kind,
        event.event_type.as_deref(),
        &event.payload,
    )
}

fn process_stream_line(
    app: &mut App,
    current_history: &mut String,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
    adapter_kind: normalize::StreamProviderAdapterKind,
    line: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    if let Some(event) = framing::consume_sse_line(&mut state.framing, line) {
        return process_stream_payload(
            app,
            current_history,
            markers,
            state,
            adapter_kind,
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
