use std::io::{self, IsTerminal, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::ai::{
    driver::print::format_section_header,
    models,
    provider::ApiProvider,
    request::StreamChunk,
    theme::{ACCENT_MUTED, ACCENT_RULE, DIM, RESET},
    types::{App, StreamOutcome, StreamResult, ToolCall, take_stream_cancelled},
};

use super::{
    MarkdownStreamRenderer,
    extract::{StreamTextEvent, extract_chunk_events_streaming, strip_ansi_codes},
    framing, normalize,
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

pub(super) async fn stream_response(
    app: &mut App,
    response: &mut reqwest::Response,
    current_history: &mut String,
    _terminal_dedupe_candidate: Option<&str>,
) -> Result<StreamResult, Box<dyn std::error::Error>> {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let adapter_kind = normalize::resolve_adapter_kind(
        models::model_provider(&app.current_model),
        &models::endpoint_for_model(&app.current_model, &app.config.endpoint),
    );

    if should_show_opencode_waiting_hint(app) {
        print_waiting_hint(&mut state)?;
    }

    let mut last_chunk_at = Instant::now();
    let has_content = |s: &StreamProcessingState| -> bool {
        !s.content.assistant_text.is_empty()
            || !s.content.reasoning_text.is_empty()
            || !s.content.tool_calls_map.is_empty()
    };

    while !app.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
        if let Some(result) = immediate_cancel_result(app, state.content.thinking_open) {
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
                    return Ok(cancelled_stream_result(state.content.thinking_open));
                }
                _ = tokio::time::sleep(Duration::from_millis(FINISH_REASON_GRACE_MS)) => break,
            }
        } else {
            let idle_remaining = Duration::from_secs(timeout_secs).saturating_sub(last_chunk_at.elapsed());
            tokio::select! {
                chunk = response.chunk() => chunk,
                _ = wait_for_interrupt(app) => {
                    return Ok(cancelled_stream_result(state.content.thinking_open));
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

fn should_show_opencode_waiting_hint(app: &App) -> bool {
    io::stdout().is_terminal()
        && models::model_provider(&app.current_model) == ApiProvider::OpenCode
}

fn print_waiting_hint(state: &mut StreamProcessingState) -> io::Result<()> {
    if state.render.waiting_hint_active {
        return Ok(());
    }
    println!(
        "{}",
        format_section_header(
            "stream",
            Some("waiting for first visible chunk from provider...")
        )
    );
    io::stdout().flush()?;
    state.render.waiting_hint_active = true;
    Ok(())
}

fn upgrade_waiting_hint_for_buffering(state: &mut StreamProcessingState) -> io::Result<()> {
    if !state.render.waiting_hint_active || state.render.waiting_hint_buffering {
        return Ok(());
    }
    print!("\x1b[1A\r\x1b[2K");
    println!(
        "{}",
        format_section_header(
            "stream",
            Some("provider is alive but buffering visible output; text may arrive in one block...")
        )
    );
    io::stdout().flush()?;
    state.render.waiting_hint_buffering = true;
    Ok(())
}

fn clear_waiting_hint(state: &mut StreamProcessingState) -> io::Result<()> {
    if !state.render.waiting_hint_active {
        return Ok(());
    }
    print!("\x1b[1A\r\x1b[2K");
    io::stdout().flush()?;
    state.render.waiting_hint_active = false;
    state.render.waiting_hint_buffering = false;
    Ok(())
}

fn immediate_cancel_result(app: &App, thinking_open: bool) -> Option<StreamResult> {
    stream_interrupt_requested(app).then(|| cancelled_stream_result(thinking_open))
}

fn cancelled_stream_result(thinking_open: bool) -> StreamResult {
    if thinking_open {
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
                app.writer.as_ref(),
                &mut state.render.markdown,
                false,
            )?;
        }
    }

    flush_terminal_splitter(&mut state, markers)?;

    state.render.markdown.flush_pending()?;

    if state.render.current_printing_index.is_some() {
        println!("\x1b[0m)");
    }

    if take_stream_cancelled(app) {
        return Ok(cancelled_stream_result(false));
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

    let mut tool_calls = collect_valid_tool_calls(&mut state.content.tool_calls_map);

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

    let outcome = if !tool_calls.is_empty() {
        StreamOutcome::ToolCall
    } else {
        StreamOutcome::Completed
    };

    Ok(StreamResult {
        outcome,
        tool_calls,
        assistant_text: state.content.assistant_text,
        hidden_meta: state.content.hidden_meta,
        reasoning_text: state.content.reasoning_text,
        skip_response_drain: true,
    })
}

/// 当开启 `ai.prompt_cache.show_metrics`（默认开）且本次请求命中了 prompt
/// 缓存时，打印一行缓存命中指标。OpenAI / DashScope 等是服务端自动缓存，
/// 这里只是把它们已经上报的 `cached_tokens` 可视化出来。
fn maybe_print_prompt_cache_metrics(usage: &crate::ai::request::StreamUsage) {
    let show = crate::commonw::configw::get_all_config()
        .get(crate::ai::config_schema::AiConfig::PROMPT_CACHE_SHOW_METRICS, "true")
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

/// 尝试把整段 assistant 文本反向识别为一个/多个 tool_call。仅在文本经过
/// 围栏剥离后能完整解析成 JSON 对象（含 `name` + `arguments`）或这种对象
/// 的数组时才会成功，避免误伤普通文本回答。
fn recover_inline_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    // 优先尝试 Hermes / Qwen 风格的 XML tool call：
    //   <tool_call><function=read_file>{"path":"/x"}</function></tool_call>
    // 或 parameter 标签形式：
    //   <function=read_file><parameter=path>/x</parameter></function>
    // 这种形态不是合法 JSON，旧实现会解析失败 → 原样打印 markup 且不执行工具，
    // turn 直接 dead-end。`<function=` 是极强的信号，普通散文不会出现，安全。
    if text.contains("<function=") {
        if let Some(calls) = recover_hermes_xml_tool_calls(text) {
            return Some(calls);
        }
    }

    let stripped = strip_inline_tool_call_wrappers(text);
    let trimmed = stripped.trim();
    if trimmed.is_empty() {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(trimmed).ok()?;
    let raw_calls: Vec<&serde_json::Value> = match &value {
        serde_json::Value::Object(_) => vec![&value],
        serde_json::Value::Array(items) if !items.is_empty() => items.iter().collect(),
        _ => return None,
    };

    let mut out = Vec::with_capacity(raw_calls.len());
    for (idx, raw) in raw_calls.into_iter().enumerate() {
        let obj = raw.as_object()?;
        // 兼容 OpenAI 风格 {"function": {"name", "arguments"}, "id"} 与
        // 简化风格 {"name", "arguments"}。
        let (name, arguments_value, id) = if let Some(func) = obj.get("function") {
            let func_obj = func.as_object()?;
            let name = func_obj.get("name")?.as_str()?.to_string();
            let args = func_obj.get("arguments").cloned().unwrap_or_else(|| {
                serde_json::Value::Object(serde_json::Map::new())
            });
            let id = obj
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            (name, args, id)
        } else {
            let name = obj.get("name")?.as_str()?.to_string();
            let args = obj.get("arguments").cloned().unwrap_or_else(|| {
                serde_json::Value::Object(serde_json::Map::new())
            });
            let id = obj
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            (name, args, id)
        };
        if name.trim().is_empty() {
            return None;
        }
        let arguments = match arguments_value {
            serde_json::Value::String(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    "{}".to_string()
                } else {
                    // 校验内层字符串确实是 JSON，避免把任意字符串当 args 透传。
                    serde_json::from_str::<serde_json::Value>(trimmed).ok()?;
                    trimmed.to_string()
                }
            }
            other => other.to_string(),
        };
        out.push(ToolCall {
            id: id.unwrap_or_else(|| format!("inline_{idx}")),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall { name, arguments },
        });
    }
    if out.is_empty() { None } else { Some(out) }
}

/// 解析 Hermes / Qwen 风格的 XML tool call。支持：
///   - 多个 `<function=NAME> ... </function>` 块（并行工具调用）
///   - body 为 JSON：`<function=read_file>{"path":"/x"}</function>`
///   - body 为 parameter 标签：`<function=read_file><parameter=path>/x</parameter></function>`
///   - 外层可有可无 `<tool_call>...</tool_call>` 包裹
/// 任意一个 `<function=...>` 块解析成功即返回；全部失败返回 None。
fn recover_hermes_xml_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let mut out: Vec<ToolCall> = Vec::new();
    let mut rest = text;
    let mut idx = 0usize;
    while let Some(open_rel) = rest.find("<function=") {
        let after_open = &rest[open_rel + "<function=".len()..];
        // 函数名到第一个 '>' 为止。
        let Some(name_end) = after_open.find('>') else {
            break;
        };
        let name = after_open[..name_end].trim().to_string();
        let body_start = name_end + 1;
        // body 到配套 </function> 为止；缺失闭合标签时取剩余全部。
        let body_region = &after_open[body_start..];
        let (body, consumed_to) = match body_region.find("</function>") {
            Some(close_rel) => (
                &body_region[..close_rel],
                body_start + close_rel + "</function>".len(),
            ),
            None => (body_region, body_region.len() + body_start),
        };
        if !name.is_empty() {
            if let Some(arguments) = parse_hermes_function_body(body) {
                out.push(ToolCall {
                    id: format!("inline_xml_{idx}"),
                    tool_type: "function".to_string(),
                    function: crate::ai::types::FunctionCall {
                        name,
                        arguments,
                    },
                });
                idx += 1;
            }
        }
        // 前进到本块结束之后，继续扫描后续并行块。
        let advance = open_rel + "<function=".len() + consumed_to;
        if advance >= rest.len() {
            break;
        }
        rest = &rest[advance..];
    }
    if out.is_empty() { None } else { Some(out) }
}

/// 把单个 `<function=...>` 的 body 解析为 JSON arguments 字符串。
/// body 既可能直接是 JSON 对象，也可能是若干 `<parameter=key>value</parameter>`。
pub(super) fn parse_hermes_function_body(body: &str) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        // 无参数工具调用（如 `<function=list_dir></function>`）合法，返回空对象。
        return Some("{}".to_string());
    }
    // 形态 1：body 本身就是 JSON 对象。
    if trimmed.starts_with('{') {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if value.is_object() {
                return Some(value.to_string());
            }
        }
    }
    // 形态 2：<parameter=key>value</parameter> 标签集合。
    if trimmed.contains("<parameter=") {
        let mut map = serde_json::Map::new();
        let mut rest = trimmed;
        while let Some(open_rel) = rest.find("<parameter=") {
            let after_open = &rest[open_rel + "<parameter=".len()..];
            let Some(key_end) = after_open.find('>') else {
                break;
            };
            let key = after_open[..key_end].trim().to_string();
            let value_region = &after_open[key_end + 1..];
            let Some(close_rel) = value_region.find("</parameter>") else {
                break;
            };
            let raw_value = value_region[..close_rel].trim();
            // 尝试把值解析成 JSON 标量/结构（数字、bool、对象、数组）；否则当字符串。
            let value = serde_json::from_str::<serde_json::Value>(raw_value)
                .unwrap_or_else(|_| serde_json::Value::String(raw_value.to_string()));
            if !key.is_empty() {
                map.insert(key, value);
            }
            rest = &value_region[close_rel + "</parameter>".len()..];
        }
        if !map.is_empty() {
            return Some(serde_json::Value::Object(map).to_string());
        }
    }
    None
}

/// 剥掉模型常见的包裹形态：```json ... ```、``` ... ```、
/// `<tool_call> ... </tool_call>`、`<|tool_call_begin|> ... <|tool_call_end|>`。
/// 仅当输入整体被这些包裹时才剥离一层；否则原样返回。
fn strip_inline_tool_call_wrappers(text: &str) -> String {
    let mut s = text.trim().to_string();
    // markdown fenced code block
    if let Some(rest) = s.strip_prefix("```") {
        if let Some(end) = rest.rfind("```") {
            let inner = &rest[..end];
            // 去掉首行可能的语言标签（json / JSON）
            let inner_trimmed = inner.trim_start();
            let inner_no_lang = inner_trimmed
                .strip_prefix("json")
                .or_else(|| inner_trimmed.strip_prefix("JSON"))
                .unwrap_or(inner_trimmed);
            s = inner_no_lang.trim().to_string();
        }
    }
    // <tool_call>...</tool_call>
    if let Some(rest) = s.strip_prefix("<tool_call>") {
        if let Some(end) = rest.rfind("</tool_call>") {
            s = rest[..end].trim().to_string();
        }
    }
    // <|tool_call_begin|>...<|tool_call_end|>
    if let Some(rest) = s.strip_prefix("<|tool_call_begin|>") {
        if let Some(end) = rest.rfind("<|tool_call_end|>") {
            s = rest[..end].trim().to_string();
        }
    }
    s
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
        return Some(cancelled_stream_result(state.content.thinking_open));
    }

    if state.framing.decode_error_count <= MAX_DECODE_ERRORS {
        eprintln!("[Warning] 尝试继续读取...");
        if wait_for_interrupt_or_timeout(
            app,
            Some(Duration::from_millis(DECODE_ERROR_RETRY_DELAY_MS)),
        )
        .await
        {
            return Some(cancelled_stream_result(state.content.thinking_open));
        }
        return None;
    }

    eprintln!("[Error] 响应流读取失败，返回已收集的内容");

    if state.content.thinking_open {
        let _ = write_stream_content(
            &format!("\n{}\n", markers.end_thinking_tag),
            app.writer.as_ref(),
            &mut state.render.markdown,
            false,
        );
        print!("\x1b[0m");
        let _ = io::stdout().flush();
    }

    let _ = state.render.markdown.flush_pending();

    Some(StreamResult {
        outcome: StreamOutcome::Completed,
        tool_calls: collect_valid_tool_calls(&mut state.content.tool_calls_map),
        assistant_text: std::mem::take(&mut state.content.assistant_text),
        hidden_meta: String::new(),
        reasoning_text: std::mem::take(&mut state.content.reasoning_text),
        skip_response_drain: true,
    })
}

fn normalize_tool_call_arguments(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Some("{}".to_string());
    }
    // 标准路径：整体就是合法 JSON。
    if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
        return Some(trimmed.to_string());
    }
    // 部分 provider（qwen3.7 等）在 delta.tool_calls.arguments 中混入 XML
    // parameter 标签（如 `{"k":"v"}</parameter><parameter-langs>...</parameter></function>`）。
    // 尝试用 Hermes body 解析器提取参数。
    if trimmed.contains("<parameter=") || trimmed.contains("</parameter>") {
        if let Some(args) = parse_hermes_function_body(trimmed) {
            return Some(args);
        }
    }
    // 尝试截取 JSON 对象前缀：从 '{' 开始找到最后一个配对的 '}'。
    if trimmed.starts_with('{') {
        if let Some(end) = find_json_object_end(trimmed) {
            let candidate = &trimmed[..=end];
            if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                return Some(candidate.to_string());
            }
        }
    }
    None
}

/// 从字符串开头的 `{` 开始，追踪大括号嵌套深度，跳过字符串字面量，返回配对 `}` 的索引。
fn find_json_object_end(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn collect_valid_tool_calls(
    builders: &mut rust_tools::cw::SkipMap<usize, ToolCallBuilder>,
) -> Vec<ToolCall> {
    builders
        .drain()
        .filter_map(|(_, mut builder)| {
            let Some(arguments) = normalize_tool_call_arguments(&builder.arguments) else {
                eprintln!(
                    "[Warning] dropping malformed tool call '{}' due to incomplete JSON arguments",
                    builder.function_name
                );
                return None;
            };
            builder.arguments = arguments;
            Some(builder.build())
        })
        .collect()
}

fn ensure_tool_calls_section_open(
    app: &mut App,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
) {
    if state.render.printed_tool_calls_header {
        return;
    }

    let _ = clear_waiting_hint(state);

    if state.content.thinking_open {
        // 如果折叠模式活跃，先执行折叠结束渲染
        if state.render.thinking_fold.active {
            let _ = finalize_thinking_fold(state);
        } else {
            let _ = write_stream_content(
                &format_end_thinking_line(markers, &state.render.markdown),
                app.writer.as_ref(),
                &mut state.render.markdown,
                false,
            );
        }
        state.content.thinking_open = false;
    }
    let _ = state.render.markdown.flush_pending();
    println!("{}", format_section_header("tool calls", None));
    state.render.printed_tool_calls_header = true;
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
    if state.render.current_printing_index.is_some() {
        println!("\x1b[0m)");
    }
    state.render.current_printing_index = Some(index);
    print!(
        "  {}│{} {}{}{}({}",
        ACCENT_RULE, RESET, ACCENT_MUTED, function_name, RESET, DIM
    );
    io::stdout().flush()
}

fn write_tool_call_arguments_stream(arguments: &str) -> io::Result<()> {
    if arguments.is_empty() {
        return Ok(());
    }

    let mut stdout = io::stdout();
    if stdout.is_terminal() {
        for ch in arguments.chars() {
            // 跳过换行符：模型可能输出 pretty-printed JSON，在终端上需要保持单行显示
            if ch == '\n' || ch == '\r' {
                continue;
            }
            let mut buf = [0u8; 4];
            stdout.write_all(ch.encode_utf8(&mut buf).as_bytes())?;
            stdout.flush()?;
        }
        return Ok(());
    }

    stdout.write_all(arguments.as_bytes())?;
    stdout.flush()
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
                    println!("\x1b[0m)");
                    state.render.current_printing_index = None;
                    let _ = io::stdout().flush();
                }
                state.content.internal_tool_call_idx += 1;
            }
        }
    }
}

fn commit_visible_content(
    app: &mut App,
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
        && content.contains(&markers.end_thinking_tag)
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
        app.writer.as_ref(),
        state,
        markers,
        state.content.thinking_open,
    )?;
    if state.content.thinking_open {
        return Ok(());
    }

    let text = if content.contains(&markers.end_thinking_tag) {
        content.replace(&markers.end_thinking_tag, "")
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

fn format_end_thinking_line(markers: &StreamMarkers, markdown: &MarkdownStreamRenderer) -> String {
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

/// Thinking 内容的折叠渲染：所有内容都先正常输出（进入 scrollback），
/// 超出限制后维护一个底部滚动窗口，只覆盖最近的 N 行区域。
/// 早期内容保留在 terminal scrollback 中，用户可以向上滚动查看。
fn write_thinking_content_folded(
    content: &str,
    state: &mut StreamProcessingState,
    markers: &StreamMarkers,
) -> io::Result<()> {
    if content.is_empty() {
        return Ok(());
    }
    let fold = &mut state.render.thinking_fold;

    // 检测 thinking 标题行（╭─ thinking）—— 直接输出不参与折叠计数
    if content.contains(&markers.thinking_tag) {
        fold.active = true;
        fold.header_printed = true;
        // 让 markdown renderer 处理标题行的样式
        return write_stream_content_to_terminal(content, &mut state.render.markdown, true);
    }

    if !fold.active {
        return write_stream_content_to_terminal(content, &mut state.render.markdown, true);
    }

    // 逐字符处理
    for ch in content.chars() {
        if ch == '\n' {
            // 一行完成
            fold.total_lines += 1;
            let completed_line = std::mem::take(&mut fold.current_line);
            fold.recent_lines.push_back(completed_line);
            // 只保留最近 max_visible_lines 行
            while fold.recent_lines.len() > fold.max_visible_lines {
                fold.recent_lines.pop_front();
            }

            if fold.total_lines <= fold.max_visible_lines {
                // 还没超限，正常输出换行
                print!("{DIM}\n{RESET}");
                // terminal_rows 记录从第一个数据行开始的已打印行数
                fold.terminal_rows += 1;
            } else {
                // 超限：覆盖底部滚动窗口区域
                thinking_fold_redraw(fold)?;
            }
        } else {
            fold.current_line.push(ch);
            // 实时流式输出字符（无论是否在折叠模式都输出，保证实时性）
            print!("{DIM}{ch}{RESET}");
        }
    }
    io::stdout().flush()?;
    Ok(())
}

/// 只覆盖底部的滚动窗口区域，不触碰 header 和早期已输出内容。
/// 窗口内容 = 折叠指示器(1行) + 最近 N 行数据。
fn thinking_fold_redraw(fold: &mut super::state::ThinkingFoldState) -> io::Result<()> {
    let mut out = io::stdout();

    // 计算需要回退的行数：
    // - 首次进入折叠模式：回退 max_visible_lines 行（之前正常输出的数据行 + 刚完成的行）
    // - 后续折叠：回退 window_rows 行（上次重绘的窗口）+ 1 行（当前刚完成的流式行）
    let erase_rows = if fold.window_rows == 0 {
        // 首次折叠：cursor 在刚完成行的 \n 之后（该行的字符已经用 print! 输出了）
        // 需要回退到最初打印的数据行位置
        // terminal_rows = 之前正常 print!(\n) 的次数 = max_visible_lines
        // 加上当前行（字符已输出但 \n 没输出）所以当前行算 1 行
        fold.terminal_rows + 1
    } else {
        // 后续折叠：回退窗口高度 + 当前流式行（字符已输出，无 \n）
        fold.window_rows + 1
    };

    if erase_rows > 0 {
        // 光标上移，回到行首，清除到屏幕末尾
        write!(out, "\x1b[{}A\r\x1b[0J", erase_rows)?;
    }

    let folded_count = fold.total_lines.saturating_sub(fold.max_visible_lines);

    // 折叠指示器
    write!(
        out,
        "{ACCENT_MUTED}  ··· {folded_count} lines folded ···\x1b[0m\n"
    )?;

    // 输出最近的 max_visible_lines 行（全部已完成）
    for line in &fold.recent_lines {
        write!(out, "{DIM}{line}{RESET}\n")?;
    }

    out.flush()?;

    // 更新 window_rows = 折叠指示器(1) + 可见数据行数
    fold.window_rows = 1 + fold.recent_lines.len();

    Ok(())
}

/// Thinking 结束时的最终渲染：覆盖底部窗口，输出最终折叠摘要 + "done thinking"。
fn finalize_thinking_fold(state: &mut StreamProcessingState) -> io::Result<()> {
    let fold = &mut state.render.thinking_fold;
    if !fold.active {
        return Ok(());
    }

    let mut out = io::stdout();

    // 擦除当前底部窗口区域（如果折叠已激活）
    let erase_rows = if fold.window_rows > 0 {
        // 有当前不完整行时多加 1
        fold.window_rows + if fold.current_line.is_empty() { 0 } else { 1 }
    } else {
        // 还没进入过折叠模式（thinking 行数 <= max_visible_lines），不需要特殊处理
        0
    };

    if erase_rows > 0 {
        write!(out, "\x1b[{}A\r\x1b[0J", erase_rows)?;
    }

    // 确保 "done thinking" 从新行开始：如果有未完成的当前行或之前有输出过内容
    if !fold.current_line.is_empty() || (fold.total_lines > 0 && erase_rows == 0) {
        write!(out, "\n")?;
    }

    if fold.total_lines > fold.max_visible_lines {
        let folded_count = fold.total_lines.saturating_sub(fold.max_visible_lines);
        write!(
            out,
            "{ACCENT_MUTED}  ··· {folded_count} lines folded ···\x1b[0m\n"
        )?;

        // 输出最近可见行
        for line in &fold.recent_lines {
            write!(out, "{DIM}{line}{RESET}\n")?;
        }
    }

    // "done thinking" 结尾标记
    write!(out, "{ACCENT_RULE}╰─\x1b[0m {ACCENT_MUTED}done thinking\x1b[0m\n")?;
    out.flush()?;

    // 重置折叠状态
    fold.reset();
    Ok(())
}

fn maybe_write_stream_content(
    content: &str,
    writer: Option<&Arc<std::sync::Mutex<std::fs::File>>>,
    state: &mut StreamProcessingState,
    markers: &StreamMarkers,
    dimmed: bool,
) -> io::Result<()> {
    if let Some(w) = writer {
        let mut guard = w.lock().unwrap();
        let clean = strip_ansi_codes(content);
        guard.write_all(clean.as_bytes())?;
        guard.flush()?;
    }

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
mod tests {
    use super::*;
    use crate::ai::{
        cli::ParsedCli,
        tools::os_tools::{GLOBAL_OS, init_os_tools_globals},
        types::{App, AppConfig},
    };
    use std::fs::File;
    use std::io::Read as _;
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex, atomic::AtomicBool, mpsc};

    #[test]
    fn prompt_cache_metrics_none_without_hit() {
        assert_eq!(format_prompt_cache_metrics(1000, 0), None);
        assert_eq!(format_prompt_cache_metrics(0, 0), None);
    }

    #[test]
    fn prompt_cache_metrics_reports_hit_rate() {
        let line = format_prompt_cache_metrics(1000, 750).unwrap();
        assert!(line.contains("750/1000"));
        assert!(line.contains("75% hit"));
    }

    fn test_app() -> App {
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
            session_id: String::new(),
            session_history_file: PathBuf::new(),
            client: reqwest::Client::builder().build().unwrap(),
            current_model: String::new(),
            current_agent: String::new(),
            current_agent_manifest: None,
            pending_files: None,
            pending_short_output: false,
            forced_skill: None,
            attached_image_files: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            streaming: Arc::new(AtomicBool::new(false)),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            ignore_next_prompt_interrupt: false,
            writer: None,
            prompt_editor: None,
            agent_context: None,
            last_skill_bias: None,
            os: crate::ai::driver::new_local_kernel(),
            agent_reload_counter: None,
            observers: vec![Box::new(
                crate::ai::driver::thinking::ThinkingOrchestrator::new(),
            )],
        }
    }

    #[tokio::test]
    async fn wait_for_interrupt_observes_request_interrupt_source() {
        let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        let app = test_app();
        init_os_tools_globals(app.os.clone());
        crate::ai::driver::signal::clear_request_interrupt();

        let waiter = wait_for_interrupt(&app);
        let trigger = async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            crate::ai::driver::signal::signal_request_interrupt();
        };

        tokio::join!(waiter, trigger);
        crate::ai::driver::signal::clear_request_interrupt();
        if let Ok(mut guard) = GLOBAL_OS.lock() {
            *guard = None;
        }
    }

    #[tokio::test]
    async fn wait_for_interrupt_or_timeout_returns_true_on_request_interrupt() {
        let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        let app = test_app();
        init_os_tools_globals(app.os.clone());
        crate::ai::driver::signal::clear_request_interrupt();

        let waiter = tokio::spawn(async move {
            wait_for_interrupt_or_timeout(&app, Some(Duration::from_secs(5))).await
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        crate::ai::driver::signal::signal_request_interrupt();

        let interrupted = tokio::time::timeout(Duration::from_millis(200), waiter)
            .await
            .expect("stream retry wait should wake on interrupt")
            .expect("waiter should complete");
        assert!(interrupted);

        crate::ai::driver::signal::clear_request_interrupt();
        if let Ok(mut guard) = GLOBAL_OS.lock() {
            *guard = None;
        }
    }

    #[test]
    fn closing_thinking_marker_starts_on_new_line_when_reasoning_line_is_open() {
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();

        state
            .render
            .markdown
            .write_chunk("still thinking", true)
            .unwrap();
        let mut content = format!("{}\nfinal", markers.end_thinking_tag);
        if content.starts_with(&markers.end_thinking_tag)
            && state.render.markdown.has_unfinished_line()
        {
            content.insert(0, '\n');
        }

        assert_eq!(content, format!("\n{}\nfinal", markers.end_thinking_tag));
    }

    #[test]
    fn closing_thinking_marker_keeps_compact_spacing_when_already_at_line_start() {
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();

        state
            .render
            .markdown
            .write_chunk("still thinking\n", true)
            .unwrap();
        let mut content = format!("{}\nfinal", markers.end_thinking_tag);
        normalize_end_thinking_boundary(&mut content, &markers, &state.render.markdown);

        assert_eq!(content, format!("{}\nfinal", markers.end_thinking_tag));
    }

    #[test]
    fn tool_call_boundary_closes_thinking_on_a_fresh_line() {
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();

        state
            .render
            .markdown
            .write_chunk("still thinking", true)
            .unwrap();

        assert_eq!(
            format_end_thinking_line(&markers, &state.render.markdown),
            format!("\n{}\n", markers.end_thinking_tag)
        );
    }

    #[test]
    fn first_reasoning_chunk_keeps_open_marker_and_first_text() {
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();
        let mut app = test_app();
        let mut current_history = String::new();
        let path =
            std::env::temp_dir().join(format!("ai-stream-thinking-{}.log", uuid::Uuid::new_v4()));
        let file = File::create(&path).unwrap();
        app.writer = Some(Arc::new(Mutex::new(file)));
        let payload = r#"{"choices":[{"delta":{"reasoning_content":"我判断这是首段"}}]}"#;

        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            normalize::StreamProviderAdapterKind::Compatible,
            None,
            payload,
        )
        .unwrap();

        assert!(state.content.thinking_open);
        assert!(current_history.is_empty());
        drop(app.writer.take());
        let written = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(written.contains(&markers.thinking_tag));
        assert!(written.contains("我判断这是首段"));
    }

    #[test]
    fn snapshot_content_only_appends_missing_suffix() {
        assert_eq!(unseen_suffix("hello wor", "hello world"), "ld");
        assert_eq!(unseen_suffix("hello world", "hello world"), "");
        assert_eq!(unseen_suffix("prefix", "suffix"), "suffix");
    }

    #[test]
    fn tool_call_render_chunk_only_streams_unprinted_suffix() {
        let mut builder = ToolCallBuilder::default();

        builder.arguments.push_str("{\"patch\":\"a");
        assert!(take_tool_call_render_chunk(None, 0, &mut builder).is_none());

        builder.function_name = "apply_patch".to_string();
        let first = take_tool_call_render_chunk(None, 0, &mut builder).unwrap();
        assert!(first.open_line);
        assert_eq!(first.function_name, "apply_patch");
        assert_eq!(first.arguments, "{\"patch\":\"a");

        builder.arguments.push('你');
        let second = take_tool_call_render_chunk(Some(0), 0, &mut builder).unwrap();
        assert!(!second.open_line);
        assert_eq!(second.arguments, "你");
    }

    #[test]
    fn normalize_tool_call_arguments_rejects_incomplete_json_and_canonicalizes_empty() {
        assert_eq!(normalize_tool_call_arguments(""), Some("{}".to_string()));
        assert_eq!(
            normalize_tool_call_arguments(" {\"command\":\"pwd\"} "),
            Some("{\"command\":\"pwd\"}".to_string())
        );
        assert_eq!(normalize_tool_call_arguments("{\"command\":"), None);
    }

    #[test]
    fn recover_inline_tool_calls_handles_bare_object() {
        // 模拟 qwen3.7-max 把 tool call 当成 content 输出的情况。
        let raw = r#"{"name":"read_file","arguments":{"path":"/tmp/x"}}"#;
        let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp/x"}"#);
        assert_eq!(calls[0].tool_type, "function");
    }

    #[test]
    fn recover_inline_tool_calls_handles_arguments_as_json_string() {
        let raw = r#"{"name":"read_file","arguments":"{\"path\":\"/tmp/x\"}"}"#;
        let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp/x"}"#);
    }

    #[test]
    fn recover_inline_tool_calls_handles_fenced_code_block() {
        let raw = "```json\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"/tmp/x\"}}\n```";
        let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
        assert_eq!(calls[0].function.name, "read_file");
    }

    #[test]
    fn recover_inline_tool_calls_handles_tool_call_xml_wrapper() {
        let raw =
            r#"<tool_call>{"name":"read_file","arguments":{"path":"/tmp/x"}}</tool_call>"#;
        let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
        assert_eq!(calls[0].function.name, "read_file");
    }

    #[test]
    fn recover_inline_tool_calls_handles_hermes_xml_json_body() {
        // 截图中模型实际输出的 Hermes/Qwen XML 形态（body 为 JSON）。
        let raw = "<tool_call>\n<function=read_file>\n{\"path\":\"/tmp/x\"}\n</function>\n</tool_call>";
        let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp/x"}"#);
    }

    #[test]
    fn recover_inline_tool_calls_handles_hermes_xml_parameter_tags() {
        let raw = "<function=read_file><parameter=path>/tmp/x</parameter><parameter=limit>200</parameter></function>";
        let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read_file");
        let args: serde_json::Value =
            serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["path"], "/tmp/x");
        // 数字参数被识别为 JSON 数字而非字符串。
        assert_eq!(args["limit"], 200);
    }

    #[test]
    fn recover_inline_tool_calls_handles_hermes_xml_no_args() {
        let raw = "<tool_call><function=list_agents></function></tool_call>";
        let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "list_agents");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn recover_inline_tool_calls_handles_hermes_xml_parallel_calls() {
        let raw = "<function=read_file>{\"path\":\"/a\"}</function><function=read_file>{\"path\":\"/b\"}</function>";
        let calls = recover_inline_tool_calls(raw).expect("should recover tool calls");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.arguments, r#"{"path":"/a"}"#);
        assert_eq!(calls[1].function.arguments, r#"{"path":"/b"}"#);
    }

    #[test]
    fn recover_inline_tool_calls_handles_array_of_calls() {
        let raw = r#"[{"name":"a","arguments":{}},{"name":"b","arguments":{"x":1}}]"#;
        let calls = recover_inline_tool_calls(raw).expect("should recover tool calls");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "a");
        assert_eq!(calls[1].function.name, "b");
        assert_eq!(calls[1].function.arguments, r#"{"x":1}"#);
    }

    #[test]
    fn recover_inline_tool_calls_handles_openai_function_wrapper() {
        let raw = r#"{"id":"call_123","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"/tmp/x\"}"}}"#;
        let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
        assert_eq!(calls[0].id, "call_123");
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments, r#"{"path":"/tmp/x"}"#);
    }

    #[test]
    fn recover_inline_tool_calls_rejects_plain_text() {
        // 普通文本回答绝不能被误判为 tool call。
        assert!(recover_inline_tool_calls("Hello world").is_none());
        assert!(recover_inline_tool_calls("").is_none());
        // 仅有 name 没有 arguments，且 name 不在合法对象中——这里应该也是不解析的，
        // 但为保留兼容性我们允许 name 单独存在时仍然识别。下面是真正的负样本：
        assert!(recover_inline_tool_calls("{\"foo\":\"bar\"}").is_none());
        assert!(recover_inline_tool_calls("12345").is_none());
        // 字符串形式的 args 必须本身也是合法 JSON，否则拒绝。
        assert!(
            recover_inline_tool_calls(r#"{"name":"x","arguments":"not-json"}"#).is_none()
        );
    }

    #[test]
    fn response_completed_event_does_not_block_late_snapshot_text() {
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();
        let mut app = test_app();
        let mut current_history = String::new();

        let should_stop = process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            normalize::StreamProviderAdapterKind::OpenAi,
            Some("response.completed"),
            r#"{"status":"completed"}"#,
        )
        .unwrap();
        assert!(!should_stop);

        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            normalize::StreamProviderAdapterKind::OpenAi,
            Some("response.output_text.done"),
            r#"{"text":"hello world"}"#,
        )
        .unwrap();

        assert_eq!(current_history, "hello world");
        assert_eq!(state.content.assistant_text, "hello world");
    }

    #[test]
    fn snapshot_done_chunk_does_not_duplicate_already_streamed_prefix() {
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();
        let mut app = test_app();
        let mut current_history = String::new();

        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            normalize::StreamProviderAdapterKind::OpenCode,
            Some("response.output_text.delta"),
            r#"{"delta":"hello wor"}"#,
        )
        .unwrap();

        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            normalize::StreamProviderAdapterKind::OpenCode,
            Some("response.output_text.done"),
            r#"{"text":"hello world"}"#,
        )
        .unwrap();

        assert_eq!(current_history, "hello world");
        assert_eq!(state.content.assistant_text, "hello world");
    }

    #[test]
    fn tool_call_snapshot_done_does_not_duplicate_already_streamed_prefix() {
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();
        let mut app = test_app();
        let mut current_history = String::new();

        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            normalize::StreamProviderAdapterKind::OpenAi,
            Some("response.output_item.added"),
            r#"{"output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"write_file","arguments":""}}"#,
        )
        .unwrap();

        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            normalize::StreamProviderAdapterKind::OpenAi,
            Some("response.function_call_arguments.delta"),
            r#"{"output_index":0,"delta":"{\"path\":\"a"}"#,
        )
        .unwrap();

        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            normalize::StreamProviderAdapterKind::OpenAi,
            Some("response.function_call_arguments.done"),
            r#"{"output_index":0,"arguments":"{\"path\":\"abc\"}"}"#,
        )
        .unwrap();

        let builder = state.content.tool_calls_map.get_ref(&0).unwrap();
        assert_eq!(builder.id, "call_1");
        assert_eq!(builder.function_name, "write_file");
        assert_eq!(builder.arguments, "{\"path\":\"abc\"}");
    }

    fn write_http_chunk(stream: &mut std::net::TcpStream, payload: &str) -> std::io::Result<()> {
        write!(stream, "{:X}\r\n", payload.len())?;
        stream.write_all(payload.as_bytes())?;
        stream.write_all(b"\r\n")?;
        stream.flush()
    }

    #[tokio::test]
    async fn stream_response_returns_after_finish_reason_without_eof() {
        let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request_buf = [0u8; 1024];
            let _ = stream.read(&mut request_buf);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
                )
                .unwrap();
            write_http_chunk(
                &mut stream,
                "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
            )
            .unwrap();
            write_http_chunk(
                &mut stream,
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            )
            .unwrap();
            let _ = done_rx.recv_timeout(Duration::from_secs(2));
        });

        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let mut response = client
            .post(format!("http://{addr}/chat"))
            .send()
            .await
            .unwrap();
        let mut app = test_app();
        init_os_tools_globals(app.os.clone());
        crate::ai::driver::signal::clear_request_interrupt();
        let mut current_history = String::new();

        let result = tokio::time::timeout(
            Duration::from_secs(2),
            stream_response(&mut app, &mut response, &mut current_history, None),
        )
        .await
        .expect("stream_response should return after the configured finish_reason grace window")
        .unwrap();

        assert_eq!(result.outcome, StreamOutcome::Completed);
        assert_eq!(result.assistant_text, "hello");
        assert_eq!(current_history, "hello");
        assert!(result.skip_response_drain);

        drop(response);
        let _ = done_tx.send(());
        server.join().unwrap();
        crate::ai::driver::signal::clear_request_interrupt();
        if let Ok(mut guard) = GLOBAL_OS.lock() {
            *guard = None;
        }
    }

    #[tokio::test]
    async fn stream_response_keeps_reading_delayed_chunks_after_finish_reason() {
        let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request_buf = [0u8; 1024];
            let _ = stream.read(&mut request_buf);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
                )
                .unwrap();
            write_http_chunk(
                &mut stream,
                "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
            )
            .unwrap();
            write_http_chunk(
                &mut stream,
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            )
            .unwrap();
            std::thread::sleep(Duration::from_millis(300));
            write_http_chunk(
                &mut stream,
                "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
            )
            .unwrap();
            write_http_chunk(&mut stream, "data: [DONE]\n\n").unwrap();
            let _ = done_rx.recv_timeout(Duration::from_secs(2));
        });

        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let mut response = client
            .post(format!("http://{addr}/chat"))
            .send()
            .await
            .unwrap();
        let mut app = test_app();
        init_os_tools_globals(app.os.clone());
        crate::ai::driver::signal::clear_request_interrupt();
        let mut current_history = String::new();

        let result = tokio::time::timeout(
            Duration::from_secs(2),
            stream_response(&mut app, &mut response, &mut current_history, None),
        )
        .await
        .expect("stream_response should keep reading delayed chunks after finish_reason")
        .unwrap();

        assert_eq!(result.outcome, StreamOutcome::Completed);
        assert_eq!(result.assistant_text, "hello world");
        assert_eq!(current_history, "hello world");
        assert!(result.skip_response_drain);

        drop(response);
        let _ = done_tx.send(());
        server.join().unwrap();
        crate::ai::driver::signal::clear_request_interrupt();
        if let Ok(mut guard) = GLOBAL_OS.lock() {
            *guard = None;
        }
    }
}

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

    if chunk.choices.is_empty() {
        state.content.empty_choice_chunks += 1;
        if should_show_opencode_waiting_hint(app) && state.content.empty_choice_chunks >= 3 {
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
        if existing.ends_with(&incoming[..split_idx]) {
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

fn write_stream_content(
    content: &str,
    writer: Option<&Arc<std::sync::Mutex<std::fs::File>>>,
    markdown: &mut MarkdownStreamRenderer,
    dimmed: bool,
) -> io::Result<()> {
    if let Some(w) = writer {
        let mut guard = w.lock().unwrap();
        let clean = strip_ansi_codes(content);
        guard.write_all(clean.as_bytes())?;
        guard.flush()?;
    }
    write_stream_content_to_terminal(content, markdown, dimmed)
}

fn write_stream_content_to_file(
    content: &str,
    mut writer: Option<&mut std::fs::File>,
) -> io::Result<()> {
    if let Some(file) = writer.as_mut() {
        let clean = strip_ansi_codes(content);
        file.write_all(clean.as_bytes())?;
        file.flush()?;
    }
    Ok(())
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
