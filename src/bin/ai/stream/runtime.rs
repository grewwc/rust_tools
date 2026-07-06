use std::io::{self, IsTerminal, Write};
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use regex::Regex;

use crate::ai::{
    config_schema::AiConfig,
    driver::print::format_section_header,
    models, provider,
    request::StreamChunk,
    theme::{ACCENT_MUTED, ACCENT_RULE, DIM, RESET},
    types::{App, StreamOutcome, StreamResult, ToolCall, take_stream_cancelled},
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

fn should_show_opencode_waiting_hint(app: &App) -> bool {
    if !io::stdout().is_terminal() {
        return false;
    }
    let provider = models::model_provider(&app.current_model);
    let endpoint = models::endpoint_for_model(&app.current_model, &app.config.endpoint);
    provider::adapter_for(provider, &endpoint).shows_waiting_hint()
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
        // 检测空响应：模型返回 200 OK 但没有文本、没有工具调用、没有推理内容。
        // 这种情况通常是 provider 端的问题（如限流、模型异常），需要触发重试
        // 而不是静默结束 turn。
        let has_text = !state.content.assistant_text.trim().is_empty();
        let has_reasoning = !state.content.reasoning_text.trim().is_empty();
        if !has_text && !has_reasoning {
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

type InlineToolCallParser = fn(&str) -> Option<Vec<ToolCall>>;

const INLINE_PARSERS: &[InlineToolCallParser] = &[
    recover_hermes_xml_tool_calls,
    recover_anthropic_xml_tool_calls,
    recover_json_tool_calls,
];

/// 把模型输出里的命名空间前缀 XML 标签归一化为标准 XML 标签，例如
/// `<|DSML|invoke name="x">` / `<｜｜DSML｜｜invoke name="x">` → `<invoke name="x">`，
/// `</|DSML|invoke>` / `</｜｜DSML｜｜invoke>` → `</invoke>`。
/// 这样 Hermes / Anthropic 解析器无需为每个 `<|PREFIX|>` 协议单独适配。
pub(super) fn normalize_inline_tool_call_markup(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains("<|") && !text.contains("<｜｜") && !text.contains("</｜｜") {
        return std::borrow::Cow::Borrowed(text);
    }
    static OPEN_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"<(?:\|([^>|]+)\|([^\s>]+)|｜｜([^＞>]+)｜｜([^\s>]+))"#)
            .expect("valid open-tag regex")
    });
    static CLOSE_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"</(?:\|([^>|]+)\|([^\s>]+)|｜｜([^＞>]+)｜｜([^\s>]+))>"#)
            .expect("valid close-tag regex")
    });
    let s = OPEN_RE.replace_all(text, |caps: &regex::Captures<'_>| {
        let local = caps
            .get(2)
            .or_else(|| caps.get(4))
            .map(|m| m.as_str())
            .unwrap_or("");
        format!("<{local}")
    });
    std::borrow::Cow::Owned(
        CLOSE_RE
            .replace_all(&s, |caps: &regex::Captures<'_>| {
                let local = caps
                    .get(2)
                    .or_else(|| caps.get(4))
                    .map(|m| m.as_str())
                    .unwrap_or("");
                format!("</{local}>")
            })
            .into_owned(),
    )
}

/// 尝试把整段 assistant 文本反向识别为一个/多个 tool_call。
/// 通过 parser 注册表 + 前置 XML 命名空间归一化，统一处理不同模型产出的
/// inline tool call 形态（Hermes XML、Anthropic XML、JSON、`<|PREFIX|>` 包装）。
/// 任一 parser 成功即返回；全部失败则视为普通文本回答。
fn recover_inline_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let normalized = normalize_inline_tool_call_markup(text);
    for parser in INLINE_PARSERS {
        if let Some(calls) = parser(&normalized) {
            return Some(calls);
        }
    }
    None
}

/// 从 assistant 文本里识别 JSON 形态的工具调用（单个对象或数组）。
fn recover_json_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
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
            let args = func_obj
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
            let id = obj
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            (name, args, id)
        } else {
            let name = obj.get("name")?.as_str()?.to_string();
            let args = obj
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
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
                    function: crate::ai::types::FunctionCall { name, arguments },
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

/// 解析 Anthropic / Claude 风格的 XML tool call。支持：
///   - 多个 `<invoke name="NAME"> ... </invoke>` 块（并行工具调用）
///   - 参数为 `<parameter name="key">value</parameter>` 标签集合
///   - 外层可有可无 `<function_calls>` / `<tool_calls>` 包裹
///   - 标签可带命名空间前缀（如 `antml:invoke`）
/// 任意一个 `<invoke ...>` 块解析成功即返回；全部失败返回 None。
fn recover_anthropic_xml_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let mut out: Vec<ToolCall> = Vec::new();
    let mut rest = text;
    let mut idx = 0usize;
    while let Some(open_rel) = rest.find("<invoke") {
        let after_tag = &rest[open_rel..];
        // 定位本 invoke 开标签的 '>'。
        let Some(open_gt) = after_tag.find('>') else {
            break;
        };
        let open_tag = &after_tag[..=open_gt];
        let name = parse_anthropic_xml_name_attr(open_tag);
        let body_start = open_rel + open_gt + 1;
        let body_region = &rest[body_start..];
        let (body, consumed_to) = match body_region.find("</invoke>") {
            Some(close_rel) => (
                &body_region[..close_rel],
                body_start + close_rel + "</invoke>".len(),
            ),
            None => (body_region, rest.len()),
        };
        if !name.trim().is_empty() {
            let arguments = parse_anthropic_invoke_body(body);
            out.push(ToolCall {
                id: format!("inline_anthropic_{idx}"),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall { name, arguments },
            });
            idx += 1;
        }
        if consumed_to >= rest.len() {
            break;
        }
        rest = &rest[consumed_to..];
    }
    if out.is_empty() { None } else { Some(out) }
}

/// 把 `<invoke>` body 里的 `<parameter name="key">value</parameter>` 解析为 JSON
/// arguments 字符串；无参数返回 `{}`。
fn parse_anthropic_invoke_body(body: &str) -> String {
    let mut map = serde_json::Map::new();
    let mut rest = body;
    while let Some(open_rel) = rest.find("<parameter") {
        let after_tag = &rest[open_rel..];
        let Some(open_gt) = after_tag.find('>') else {
            break;
        };
        let open_tag = &after_tag[..=open_gt];
        let key = parse_anthropic_xml_name_attr(open_tag);
        let value_region = &after_tag[open_gt + 1..];
        let (raw_value, consumed_in_after) = match value_region.find("</parameter>") {
            Some(close_rel) => (
                &value_region[..close_rel],
                open_gt + 1 + close_rel + "</parameter>".len(),
            ),
            None => break,
        };
        let raw_value = raw_value.trim();
        if !key.trim().is_empty() {
            let value = serde_json::from_str::<serde_json::Value>(raw_value)
                .unwrap_or_else(|_| serde_json::Value::String(raw_value.to_string()));
            map.insert(key, value);
        }
        let advance = open_rel + consumed_in_after;
        if advance >= rest.len() {
            break;
        }
        rest = &rest[advance..];
    }
    if map.is_empty() {
        "{}".to_string()
    } else {
        serde_json::Value::Object(map).to_string()
    }
}

/// 从 `<invoke name="x">` / `<parameter name="y">` 开标签里抽取 `name` 属性值，
/// 支持双引号或单引号。
fn parse_anthropic_xml_name_attr(open_tag: &str) -> String {
    let Some(pos) = open_tag.find("name") else {
        return String::new();
    };
    let after = open_tag[pos + "name".len()..].trim_start();
    let after = after.strip_prefix('=').unwrap_or(after).trim_start();
    let (quote, body) = if let Some(b) = after.strip_prefix('"') {
        ('"', b)
    } else if let Some(b) = after.strip_prefix('\'') {
        ('\'', b)
    } else {
        return String::new();
    };
    match body.find(quote) {
        Some(end) => body[..end].to_string(),
        None => String::new(),
    }
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
    _app: &mut App,
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
                &mut state.render.markdown,
                false,
            );
        }
        state.content.thinking_open = false;
    }
    let _ = state.render.markdown.flush_pending();
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
                    println!("\x1b[0m");
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
fn finalize_thinking_fold(state: &mut StreamProcessingState) -> io::Result<()> {
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
mod tests {
    use super::*;
    use crate::ai::{
        cli::ParsedCli,
        tools::os_tools::{GLOBAL_OS, init_os_tools_globals},
        types::{App, AppConfig},
    };
    use std::io::Read as _;
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool, mpsc};

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

    #[test]
    fn thinking_fold_defaults_to_configured_lines_for_tty() {
        assert_eq!(
            resolve_thinking_fold_max_visible_lines(true, None),
            DEFAULT_THINKING_MAX_VISIBLE_LINES
        );
        assert_eq!(
            resolve_thinking_fold_max_visible_lines(true, Some("12")),
            12
        );
        assert_eq!(
            resolve_thinking_fold_max_visible_lines(true, Some("0")),
            usize::MAX
        );
        assert_eq!(
            resolve_thinking_fold_max_visible_lines(true, Some("oops")),
            DEFAULT_THINKING_MAX_VISIBLE_LINES
        );
        assert_eq!(
            resolve_thinking_fold_max_visible_lines(false, Some("12")),
            usize::MAX
        );
    }

    fn test_app() -> App {
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
                agent_route_model_path: PathBuf::new(),
                skill_match_model_path: PathBuf::new(),
            },
            session_id: String::new(),
            session_history_file: PathBuf::new(),
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::builder().build().unwrap(),
            current_model: String::new(),
            current_agent: String::new(),
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
        let raw = r#"<tool_call>{"name":"read_file","arguments":{"path":"/tmp/x"}}</tool_call>"#;
        let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
        assert_eq!(calls[0].function.name, "read_file");
    }

    #[test]
    fn recover_inline_tool_calls_handles_hermes_xml_json_body() {
        // 截图中模型实际输出的 Hermes/Qwen XML 形态（body 为 JSON）。
        let raw =
            "<tool_call>\n<function=read_file>\n{\"path\":\"/tmp/x\"}\n</function>\n</tool_call>";
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
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
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
        assert!(recover_inline_tool_calls(r#"{"name":"x","arguments":"not-json"}"#).is_none());
    }

    #[test]
    fn recover_inline_tool_calls_handles_anthropic_xml_parameter_tags() {
        // deepseek-v4-flash 实际输出的 Anthropic 风格：<invoke name=...>/<parameter name=...>。
        let raw = r#"<function_calls><invoke name="read_file"><parameter name="path">/tmp/x</parameter><parameter name="limit">200</parameter></invoke></function_calls>"#;
        let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read_file");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["path"], "/tmp/x");
        assert_eq!(args["limit"], 200);
    }

    #[test]
    fn recover_inline_tool_calls_handles_anthropic_xml_namespaced_tags() {
        // 带命名空间前缀（antml:）且无外层包裹。
        let raw = r#"<invoke name="list_agents"></invoke>"#;
        let calls = recover_inline_tool_calls(raw).expect("should recover tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "list_agents");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    #[test]
    fn recover_inline_tool_calls_handles_anthropic_xml_parallel_calls() {
        let raw = r#"<tool_calls><invoke name="read_file"><parameter name="path">/a</parameter></invoke><invoke name="read_file"><parameter name="path">/b</parameter></invoke></tool_calls>"#;
        let calls = recover_inline_tool_calls(raw).expect("should recover tool calls");
        assert_eq!(calls.len(), 2);
        let a: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        let b: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(a["path"], "/a");
        assert_eq!(b["path"], "/b");
    }

    #[test]
    fn anthropic_xml_streamer_suppresses_markup_and_emits_events() {
        let mut streamer = super::super::splitter::AnthropicXmlToolCallStreamer::new();
        let (cleaned, events) = streamer.push(
            r#"Let me check.<invoke name="read_file"><parameter name="path">/tmp/x</parameter></invoke>"#,
        );
        // invoke 标记不外显，仅保留前置散文。
        assert_eq!(cleaned, "Let me check.");
        // 产出 Begin/Args/End 事件，与内部 tool_call 管线一致。
        assert_eq!(events.len(), 3);
        match (&events[0], &events[1], &events[2]) {
            (
                InternalToolCallStreamEvent::Begin(name),
                InternalToolCallStreamEvent::Args(args),
                InternalToolCallStreamEvent::End,
            ) => {
                assert_eq!(name, "read_file");
                let v: serde_json::Value = serde_json::from_str(args).unwrap();
                assert_eq!(v["path"], "/tmp/x");
            }
            _ => panic!("unexpected events: {events:?}"),
        }
    }

    #[test]
    fn anthropic_xml_streamer_handles_split_chunks() {
        let mut streamer = super::super::splitter::AnthropicXmlToolCallStreamer::new();
        let mut all_events = Vec::new();
        let mut all_cleaned = String::new();
        for chunk in [
            "pre <inv",
            "oke name=\"read_file\"><parameter name=\"pa",
            "th\">/tmp/x</parameter></in",
            "voke> post",
        ] {
            let (cleaned, events) = streamer.push(chunk);
            all_cleaned.push_str(&cleaned);
            all_events.extend(events);
        }
        assert_eq!(all_cleaned, "pre  post");
        assert_eq!(all_events.len(), 3);
        match &all_events[0] {
            InternalToolCallStreamEvent::Begin(name) => assert_eq!(name, "read_file"),
            other => panic!("unexpected first event: {other:?}"),
        }
    }

    #[test]
    fn anthropic_xml_streamer_leaves_prose_angle_brackets_intact() {
        let mut streamer = super::super::splitter::AnthropicXmlToolCallStreamer::new();
        let (cleaned, events) = streamer.push("a < b and c > d, also <div> here");
        assert_eq!(cleaned, "a < b and c > d, also <div> here");
        assert!(events.is_empty());
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
    fn thinking_fold_keeps_reasoning_buffer_intact() {
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();
        state.render.thinking_fold.max_visible_lines = 2;
        let mut app = test_app();
        let mut current_history = String::new();

        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            normalize::StreamProviderAdapterKind::OpenAi,
            Some("response.reasoning_text.delta"),
            r#"{"delta":"step 1\nstep 2\nstep 3"}"#,
        )
        .unwrap();

        assert_eq!(state.content.reasoning_text, "step 1\nstep 2\nstep 3");
        assert!(state.content.thinking_open);
        assert!(current_history.is_empty());
        assert!(state.content.assistant_text.is_empty());

        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            normalize::StreamProviderAdapterKind::OpenAi,
            Some("response.output_text.delta"),
            r#"{"delta":"final answer"}"#,
        )
        .unwrap();

        assert_eq!(state.content.reasoning_text, "step 1\nstep 2\nstep 3");
        assert_eq!(current_history, "final answer");
        assert_eq!(state.content.assistant_text, "final answer");
        assert!(!state.content.thinking_open);
        assert!(!state.render.thinking_fold.active);
    }

    #[test]
    fn thinking_fold_drops_interior_blank_lines() {
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();
        state.render.thinking_fold.max_visible_lines = 8;
        state.render.thinking_fold.active = true;

        // 模型常用空行分段：段间空行不应占用折叠窗口的可见行。
        write_thinking_content_folded("para 1\n\npara 2\n", &mut state, &markers).unwrap();

        let fold = &state.render.thinking_fold;
        assert_eq!(
            fold.recent_lines.iter().collect::<Vec<_>>(),
            vec!["para 1", "para 2"]
        );
        assert_eq!(fold.total_lines, 2);
    }

    #[test]
    fn thinking_fold_window_counts_current_line_inside_visible_budget() {
        // 锁定并置宽 COLUMNS：本用例断言正文/标记原样存在，须避免与 COLUMNS=12
        // 的折行用例并发时读到被泄漏的窄列宽而触发 clamp 截断。
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("COLUMNS", "200");
        }

        let mut state = StreamProcessingState::new();
        let fold = &mut state.render.thinking_fold;
        fold.max_visible_lines = 3;
        fold.total_lines = 3;
        fold.recent_lines.push_back("line-1".to_string());
        fold.recent_lines.push_back("line-2".to_string());
        fold.recent_lines.push_back("line-3".to_string());
        fold.current_line = "line-4".to_string();

        assert_eq!(thinking_fold_hidden_count(fold), 1);
        assert_eq!(
            thinking_fold_visible_lines(fold),
            vec!["line-2", "line-3", "line-4"]
        );

        let (window, _) = render_thinking_fold_window(fold);
        assert_eq!(window.matches("lines folded").count(), 1);
        assert!(!window.contains("line-1"));
        assert!(window.contains("line-2"));
        assert!(window.contains("line-3"));
        assert!(window.contains("line-4"));

        unsafe {
            std::env::remove_var("COLUMNS");
        }
    }

    #[test]
    fn thinking_fold_window_rows_follow_wrapped_terminal_height() {
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("COLUMNS", "12");
        }

        let mut state = StreamProcessingState::new();
        let fold = &mut state.render.thinking_fold;
        fold.max_visible_lines = 2;
        fold.total_lines = 2;
        fold.recent_lines.push_back("12345678901234567890".to_string());
        fold.recent_lines.push_back("abcdef".to_string());
        fold.current_line = "ghijklmnopqrst".to_string();

        let (window, rows) = render_thinking_fold_window(fold);

        // 每条可见行被 clamp 成单物理行，窗口物理行数恒等于逻辑行数：
        // 1 折叠标记 + 2 可见行 = 3。
        assert_eq!(rows, 3);
        // 每条渲染行都不超过终端列宽（12），确保 cursor-up 擦除精确。
        for line in window.lines() {
            let visible = crate::ai::stream::extract::strip_ansi_codes(line);
            assert!(
                unicode_width::UnicodeWidthStr::width(visible.as_str()) <= 12,
                "line exceeds terminal width: {visible:?}"
            );
        }
        // 未溢出的短行原样保留；溢出的超长行被截断为省略号结尾。
        assert!(window.contains("abcdef"));
        assert!(window.contains('…'));

        unsafe {
            std::env::remove_var("COLUMNS");
        }
    }

    #[test]
    fn thinking_fold_window_without_hidden_lines_has_no_fold_marker() {
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("COLUMNS", "200");
        }

        let mut state = StreamProcessingState::new();
        let fold = &mut state.render.thinking_fold;
        fold.max_visible_lines = 4;
        fold.total_lines = 2;
        fold.recent_lines.push_back("line-1".to_string());
        fold.recent_lines.push_back("line-2".to_string());
        fold.current_line = "line-3".to_string();

        let (window, rows) = render_thinking_fold_window(fold);

        // 无隐藏行、无 active header：窗口物理行数 == 可见逻辑行数（3）。
        assert!(!window.contains("lines folded"));
        assert!(window.contains("line-1"));
        assert!(window.contains("line-2"));
        assert!(window.contains("line-3"));
        assert_eq!(rows, 3);

        unsafe {
            std::env::remove_var("COLUMNS");
        }
    }

    #[test]
    fn thinking_fold_active_window_includes_header_in_row_budget() {
        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("COLUMNS", "200");
        }

        let mut state = StreamProcessingState::new();
        let fold = &mut state.render.thinking_fold;
        fold.active = true;
        fold.max_visible_lines = 2;
        fold.total_lines = 2;
        fold.recent_lines.push_back("line-1".to_string());
        fold.current_line = "line-2".to_string();

        let (window, rows) = render_thinking_fold_window(fold);

        // active header(1) + 折叠标记(1) + 可见行(1) + current(1) = 4 物理行。
        assert!(window.contains("thinking"));
        assert!(window.contains("lines folded"));
        assert!(window.contains("line-1"));
        assert!(window.contains("line-2"));
        assert_eq!(rows, 4);

        unsafe {
            std::env::remove_var("COLUMNS");
        }
    }

    #[test]
    fn cancelled_stream_result_finalizes_active_thinking_fold() {
        // 取消时若折叠窗口仍活跃，必须收口（finalize→reset），避免半截 `╭─ thinking`
        // 残留、下一轮重试在其下叠新 header（重复 header + 大段空白的跨轮根因）。
        let mut state = StreamProcessingState::new();
        {
            let fold = &mut state.render.thinking_fold;
            fold.active = true;
            fold.max_visible_lines = 2;
            fold.total_lines = 1;
            fold.recent_lines.push_back("partial".to_string());
            fold.window_rows = 2;
        }

        let result = cancelled_stream_result(&mut state);

        assert!(matches!(result.outcome, StreamOutcome::Cancelled));
        assert!(result.skip_response_drain);
        // finalize 后折叠状态被 reset：不再 active，窗口行数归零，无孤儿窗口残留。
        assert!(!state.render.thinking_fold.active);
        assert_eq!(state.render.thinking_fold.window_rows, 0);
        assert!(state.render.thinking_fold.recent_lines.is_empty());
    }

    #[test]
    fn standalone_stream_marker_requires_exact_control_line() {
        assert!(is_standalone_stream_marker(
            "\n╭─ thinking\n",
            "╭─ thinking"
        ));
        assert!(is_standalone_stream_marker(
            "\n╰─ done thinking\n",
            "╰─ done thinking"
        ));
        assert!(!is_standalone_stream_marker(
            "reasoning mentions ╭─ thinking literally",
            "╭─ thinking"
        ));
        assert!(!is_standalone_stream_marker(
            "prefix\n╰─ done thinking\nsuffix",
            "╰─ done thinking"
        ));
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

    #[test]
    fn recover_inline_tool_calls_normalizes_namespaced_xml_prefix() {
        // 某些前端/模型会把 Anthropic 风格的 invoke 包在 <|DSML|> 协议里输出。
        // 归一化后应被 Anthropic XML 解析器识别，无需为每种 <|PREFIX|> 单独写 parser。
        let raw = r#"<|DSML|tool_calls><|DSML|invoke name="apply_patch"><|DSML|parameter name="file_path">/tmp/x</|DSML|parameter><|DSML|parameter name="patch">---</|DSML|parameter></|DSML|invoke></|DSML|tool_calls>"#;
        let calls = recover_inline_tool_calls(raw).expect("should recover DSML-wrapped tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "apply_patch");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["file_path"], "/tmp/x");
        assert_eq!(args["patch"], "---");
    }

    #[test]
    fn recover_inline_tool_calls_normalizes_fullwidth_dsml_prefix() {
        // debug.md 里的 DeepSeek 实际会输出全角竖线版本：<｜｜DSML｜｜...>。
        let raw = r#"<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="apply_patch"><｜｜DSML｜｜parameter name="file_path">/tmp/x</｜｜DSML｜｜parameter><｜｜DSML｜｜parameter name="patch">---</｜｜DSML｜｜parameter></｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>"#;
        let calls =
            recover_inline_tool_calls(raw).expect("should recover fullwidth-DSML tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "apply_patch");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["file_path"], "/tmp/x");
        assert_eq!(args["patch"], "---");
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

fn write_stream_content(
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
