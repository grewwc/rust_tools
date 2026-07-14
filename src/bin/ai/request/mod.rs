use std::fs;
use std::time::{Duration, Instant};

use base64::Engine as _;
use reqwest::Response;
#[cfg(test)]
use reqwest::StatusCode;
use rust_tools::commonw;
use serde_json::{Map, Value, json};

use super::provider::adapter_for;
use super::{
    files,
    history::{
        Message, SessionStore, generate_session_summary, is_system_like_role, messages_to_markdown,
    },
    models,
    provider::ReasoningEffort,
    skills::SkillManifest,
    types::App,
};
use crate::ai::theme::{ACCENT_MUTED, ACCENT_PRIMARY, ACCENT_SUCCESS, ACCENT_WARN, RESET};
use crate::commonw::configw;

mod error;
mod normalize;
mod routing;
mod thinking;
mod types;

#[allow(unused_imports)]
pub(crate) use error::{
    AUTO_SUBAGENT_REQUEST_MAX_ATTEMPTS, AUTO_SUBAGENT_RESPONSE_HEADER_TIMEOUT_SECS,
    REQUEST_MAX_ATTEMPTS, REQUEST_MAX_ATTEMPTS_429, REQUEST_RETRY_429_MAX_MS, RequestError,
    RequestErrorKind, RequestRetryPolicy, STREAM_RESPONSE_HEADER_TIMEOUT_SECS,
    api_key_for_request_model, apply_request_auth, clear_stale_request_interrupt_before_request,
    config_bool_is_true, config_forces_thinking, control_model_for_aux_tasks,
    endpoint_for_request_model, is_retryable_reqwest_error, is_retryable_status_with_body,
    is_retryable_stream_error, is_transient_error, parse_retry_after, request_retry_policy,
    request_retry_policy_for_current_context, retry_delay, send_with_hedged_backup,
    should_abort_retry_wait, should_retry_status, should_rotate_key,
    should_temporarily_disable_auto_selected_model, should_temporarily_disable_model,
    should_try_model_fallback, sleep_with_cancel,
};
#[allow(unused_imports)]
use normalize::{
    agent_tools_for_request, normalize_message_content_for_text_only_model,
    normalize_messages_for_model, normalize_messages_for_request, request_tool_names_for_model,
    strip_unavailable_tool_hints_from_messages,
};
#[allow(unused_imports)]
pub(crate) use routing::{extract_router_content, strip_json_fence};
use thinking::resolve_thinking;
pub(crate) use thinking::strip_system_reminders;
#[cfg(test)]
pub(crate) use thinking::{
    latest_user_message_text, local_thinking_decision, parse_thinking_gate_output,
};
use types::RequestBody;
#[allow(unused_imports)]
pub(crate) use types::{
    StreamChoice, StreamChunk, StreamDelta, StreamFunctionCall, StreamToolCall, StreamUsage,
    merge_reasoning_fragments,
};

const SESSION_TITLE_REQUEST_TIMEOUT_SECS: u64 = 90;
const SESSION_TITLE_BODY_TIMEOUT_SECS: u64 = 45;

/// 并发请求（前台 turn + 各子代理）各自独立重试，`attempt N/M` 计数互相
/// 交错、无法区分归属。用 aios 调度 pid 作为作用域标签把每条重试日志绑定到
/// 具体进程；无 pid（无 TASK_PID 作用域）时返回空串，日志退化为原样。
fn retry_scope_tag() -> String {
    match aios_kernel::kernel::current_task_pid() {
        Some(pid) => format!("[pid {pid}] "),
        None => String::new(),
    }
}

/// Try a single API key for `do_request_messages`, with retry logic for
/// header timeout, network errors, and retryable server statuses (5xx / 400+upstream).
///
/// 429（配额/限流）**不**在此内部退避重试：直接携带（已钳制的）`retry_after`
/// 返回，交由上层 `do_request_messages` 先轮换其它 key，再决定是否退避重试。
/// Returns `Ok` on success or `Err` after exhausting per-key retries.
async fn request_messages_with_key(
    app: &mut App,
    api_key: &str,
    request_body: &RequestBody<'_>,
    retry_policy: &RequestRetryPolicy,
    endpoint: &str,
) -> Result<Response, RequestError> {
    for attempt in 1..=retry_policy.max_attempts {
        let build_request = || {
            apply_request_auth(app.client.post(endpoint), endpoint, api_key)
                .header("Content-Type", "application/json")
                .json(request_body)
        };
        let response = match tokio::time::timeout(
            Duration::from_secs(retry_policy.header_timeout_secs),
            send_with_hedged_backup(
                build_request,
                retry_policy.hedged_backup_after_secs(),
                retry_policy.hedged_max_sends(),
            ),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => {
                if attempt < retry_policy.max_attempts {
                    let delay = retry_delay(attempt);
                    eprintln!(
                        "[Warning] {}等待响应头超时 ({}s) - sleep {} 秒后重试 (attempt {}/{})",
                        retry_scope_tag(),
                        retry_policy.header_timeout_secs,
                        delay.as_secs_f32(),
                        attempt,
                        retry_policy.max_attempts
                    );
                    if sleep_with_cancel(app, delay).await {
                        return Err(RequestError::cancelled(
                            "request canceled by user during retry wait",
                        ));
                    }
                    continue;
                }
                return Err(RequestError {
                    kind: RequestErrorKind::Network,
                    message: format!(
                        "request timed out waiting for response headers after {} attempts",
                        retry_policy.max_attempts
                    ),
                    retry_after: None,
                });
            }
        };

        match response {
            Ok(response) => {
                if response.status().is_success() {
                    return Ok(response);
                }
                let status = response.status();
                let status_code = status.as_u16();
                let retry_after_delay = if status_code == 429 {
                    parse_retry_after(response.headers())
                } else {
                    None
                };
                let body = (response.text().await).unwrap_or_default();
                let mut err = RequestError::status(status, body);

                // 429（配额/限流）不在单 key 内退避：携带钳制后的 retry_after 立即返回，
                // 让上层先轮换其它 key，key 用尽后再由上层决定退避重试。
                if status_code == 429 {
                    err.retry_after = retry_after_delay;
                    return Err(err);
                }

                if is_retryable_status_with_body(status, &err.message)
                    && attempt < retry_policy.max_attempts
                {
                    let delay = retry_delay(attempt);
                    eprintln!(
                        "[Warning] {}{} - sleep {} 秒后重试 (attempt {}/{})",
                        retry_scope_tag(),
                        status,
                        delay.as_secs_f32(),
                        attempt,
                        retry_policy.max_attempts
                    );
                    if sleep_with_cancel(app, delay).await {
                        return Err(RequestError::cancelled(
                            "request canceled by user during retry wait",
                        ));
                    }
                    continue;
                }
                return Err(err);
            }
            Err(err) => {
                let retryable = is_retryable_reqwest_error(&err);
                let err = RequestError::network(err);
                if retryable && attempt < retry_policy.max_attempts {
                    let delay = retry_delay(attempt);
                    eprintln!(
                        "[Warning] {}网络错误 - sleep {} 秒后重试 (attempt {}/{})",
                        retry_scope_tag(),
                        delay.as_secs_f32(),
                        attempt,
                        retry_policy.max_attempts
                    );
                    if sleep_with_cancel(app, delay).await {
                        return Err(RequestError::cancelled(
                            "request canceled by user during retry wait",
                        ));
                    }
                    continue;
                }
                return Err(err);
            }
        }
    }
    unreachable!("retry loop always returns or breaks")
}

#[commonw::debug_measure_time("do_request_message")]
pub(super) async fn do_request_messages(
    app: &mut App,
    model: &str,
    messages: &[Message],
    stream: bool,
) -> Result<Response, RequestError> {
    clear_stale_request_interrupt_before_request(app);

    let mut normalized_messages = normalize_messages_for_model(model, messages);
    let request_tool_names = request_tool_names_for_model(app, model);
    strip_unavailable_tool_hints_from_messages(&mut normalized_messages, &request_tool_names);
    if prompt_cache_enabled_for_model(model) {
        apply_prompt_cache_breakpoint(&mut normalized_messages);
    }
    let (tools_value, tool_choice) = agent_tools_for_request(app, model);
    let thinking_start = Instant::now();
    let force_thinking_requested = config_forces_thinking();
    let enable_thinking = resolve_thinking(app, model, &normalized_messages).await;
    crate::ai::agent_hang_debug!(
        "pre-fix",
        "G",
        "request::do_request_messages:resolve_thinking:end",
        "[DEBUG] resolve thinking finished",
        {
            "enable_thinking": enable_thinking,
            "elapsed_ms": thinking_start.elapsed().as_secs_f64() * 1000.0,
        },
    );
    if force_thinking_requested && !enable_thinking {
        eprintln!(
            "[Info] thinking 已请求，但当前模型 `{}` 不支持 thinking；本轮将继续以普通模式输出。",
            model
        );
    }
    // DeepSeek/OpenCode 等协议要求：只要该模型 wire 需要 tool-call assistant
    // 历史回传 `reasoning_content` 字段，就必须在发请求前补齐字段形状。
    // 这与本轮 `enable_thinking` 判定不是同一回事：mid-turn 压缩会把较老
    // 的 reasoning 文本裁成 None，而该模型即使本轮 local gate 关掉 thinking，
    // 也可能仍因默认 `reasoning_effort` / 历史续写约束要求回传空字符串占位。
    ensure_reasoning_content_echo_for_thinking_model(model, &mut normalized_messages);
    let reasoning_effort = resolve_reasoning_effort(app, model).map(|e| e.as_str());
    let request_body = build_request_body(
        model,
        &normalized_messages,
        stream,
        enable_thinking,
        models::search_enabled(model).then_some(true),
        tools_value,
        tool_choice,
        reasoning_effort,
        app.cli.max_tokens_override,
        app.last_known_prompt_tokens,
    );
    let retry_policy = request_retry_policy_for_current_context();

    // --- Key rotation + 429 backoff ---
    // 每一轮先轮换尝试所有 key；只有当全部 key 都因 429（配额/限流）失败时，
    // 才整轮退避重试（最多 max_attempts_429 轮）。其它可轮换错误（401/403）
    // 轮换 key 后仍失败则直接返回，重试无意义。
    let endpoint = endpoint_for_request_model(app, model);
    let primary_key = api_key_for_request_model(app, model);
    let adapter = adapter_for(models::model_adapter(model), &endpoint);
    let keys_to_try = adapter.collect_api_keys(&primary_key);
    let total_keys = keys_to_try.len();

    let mut last_key_err: Option<RequestError> = None;
    for attempt in 1..=retry_policy.max_attempts_429 {
        let mut all_rate_limited = true;
        let mut round_retry_after: Option<Duration> = None;
        for (key_idx, api_key) in keys_to_try.iter().enumerate() {
            if key_idx > 0 {
                eprintln!(
                    "[{}] key #{} failed, trying next key #{} ({} remaining)",
                    adapter.label(),
                    key_idx - 1,
                    key_idx,
                    total_keys - key_idx
                );
            }
            match request_messages_with_key(app, api_key, &request_body, &retry_policy, &endpoint)
                .await
            {
                Ok(response) => return Ok(response),
                Err(err) if should_rotate_key(&err) => {
                    if err.is_rate_limited() {
                        round_retry_after = err.retry_after.or(round_retry_after);
                    } else {
                        all_rate_limited = false;
                    }
                    last_key_err = Some(err);
                }
                Err(err) => return Err(err),
            }
        }

        // 所有 key 均失败。仅当全部为 429 限流且仍有重试额度时才整轮退避重试。
        if !all_rate_limited || attempt >= retry_policy.max_attempts_429 {
            break;
        }
        let delay = round_retry_after.unwrap_or_else(|| retry_delay(attempt));
        eprintln!(
            "[Warning] {}429 Too Many Requests - {} 个 key 均配额超限，sleep {} 秒后重试 (attempt {}/{})",
            retry_scope_tag(),
            total_keys,
            delay.as_secs_f32(),
            attempt,
            retry_policy.max_attempts_429
        );
        if sleep_with_cancel(app, delay).await {
            return Err(RequestError::cancelled(
                "request canceled by user during retry wait",
            ));
        }
    }
    Err(last_key_err.unwrap_or_else(|| RequestError {
        kind: RequestErrorKind::Network,
        message: adapter.keys_exhausted_message().to_string(),
        retry_after: None,
    }))
}

pub(super) fn build_content(
    model: &str,
    question: &str,
    image_files: &[String],
) -> Result<Value, Box<dyn std::error::Error>> {
    if !models::supports_image_input(model) || image_files.is_empty() {
        return Ok(Value::String(question.to_string()));
    }

    let mut parts = Vec::new();
    for file in image_files {
        let bytes = fs::read(file)?;
        let mime = files::image_mime_type(file);
        let image = base64::engine::general_purpose::STANDARD.encode(bytes);
        parts.push(json!({
            "type": "image_url",
            "image_url": {
                "url": format!("data:{mime};base64,{image}")
            },
        }));
    }
    parts.push(json!({
        "type": "text",
        "text": question,
    }));
    Ok(Value::Array(parts))
}

pub(super) fn print_info(app: &App, model: &str) {
    let search = if models::search_enabled(model) {
        "true"
    } else {
        "false"
    };
    let effort_label = if app.cli.thinking_disabled_override {
        // 截断兜底已强制关闭 thinking（对 always-thinking 模型降 effort 无效时的
        // 最终手段），显式标注，避免与"auto/模型默认"混淆。
        "off"
    } else {
        match resolve_reasoning_effort(app, model) {
            Some(e) => e.as_str(),
            None => "auto",
        }
    };

    // 打印当前 session 摘要
    let store = SessionStore::new(&app.config.history_file);
    let summary = store
        .read_session_title(&app.session_id)
        .ok()
        .flatten()
        .or_else(|| {
            store
                .first_user_prompt(&app.session_id)
                .ok()
                .flatten()
                .map(|p| generate_session_summary(&p))
        });
    let session_part = summary
        .filter(|s| !s.is_empty())
        .map(|s| format!("{ACCENT_MUTED} · {ACCENT_WARN}{}{RESET}", s))
        .unwrap_or_default();

    // 使用 println! 避免手动 flush 的权限问题；模型与 session 合并为一行。
    println!(
        "{ACCENT_MUTED}[{ACCENT_SUCCESS}{}{ACCENT_MUTED} (search: {ACCENT_WARN}{search}{ACCENT_MUTED}, effort: {ACCENT_PRIMARY}{effort_label}{ACCENT_MUTED}){session_part}{ACCENT_MUTED}]{RESET}",
        models::model_display_label(model),
    );
}

fn build_request_body<'a>(
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
    enable_thinking: bool,
    enable_search: Option<bool>,
    tools: Option<Value>,
    tool_choice: Option<Value>,
    reasoning_effort: Option<&'a str>,
    max_tokens_override: Option<u32>,
    known_prompt_tokens: Option<u64>,
) -> RequestBody<'a> {
    let adapter_kind = models::model_adapter(model);
    let endpoint = models::endpoint_for_model(model, "");
    let adapter = super::provider::adapter_for(adapter_kind, &endpoint);
    let request_model = models::request_model_name(model);
    let (thinking, reasoning_effort, reasoning) =
        resolve_reasoning_wire_controls(model, &endpoint, enable_thinking, reasoning_effort);
    // 流式请求显式索取 usage：部分 adapter（DashScope compatible-mode）流式下
    // 默认不返回 usage，必须声明 stream_options.include_usage 才能统计 token。
    let stream_options = stream.then(|| json!({ "include_usage": true }));
    // 仅在模型声明了 max_output_tokens 时下发 max_tokens；并按「剩余上下文窗口」
    // 钳制，避免 prompt + 请求的输出上限一起挤爆 token 窗口（GLM 长上下文下反复
    // 截断 → 重试死循环的根因）。未声明 max_output_tokens 的模型保持不下发字段，
    // wire 行为不变。
    let max_tokens = models::max_output_tokens(model).map(|model_max| {
        clamp_max_tokens_for_prompt(
            model,
            messages,
            tools.as_ref(),
            model_max,
            known_prompt_tokens,
        )
    });
    // 零输出截断自适应：当上一轮检测到 completion=0 + finish_reason=length 时，
    // orchestrator 会把 max_tokens_override 设为更小的值。此处用该值替换 clamp 结果，
    // 让下一轮请求发送更小的 max_tokens，绕过服务端对超大 max_tokens 的空响应拒绝。
    let max_tokens = match (max_tokens, max_tokens_override) {
        (Some(_), Some(override_val)) => Some(override_val),
        (mt, _) => mt,
    };
    RequestBody {
        model: request_model,
        messages,
        stream,
        thinking,
        enable_search: adapter.enable_search_field(enable_search),
        tools,
        tool_choice,
        reasoning_effort,
        reasoning,
        stream_options,
        max_tokens,
    }
}

/// 中英文 + 代码混合语料下的保守「字符 → token」换算：每 token 约 2 个字符。
/// 偏保守（高估 prompt 占用），宁可提前钳小输出上限，也不让 prompt + 输出一起
/// 撞爆窗口。
const CHARS_PER_TOKEN_CONSERVATIVE: usize = 2;
/// 钳制后为输出保留的最小 token 数：即便 prompt 已接近窗口，也保证有一段可见
/// 输出空间，避免下发一个过小甚至为 0 的 max_tokens 反而立刻截断。
const MIN_OUTPUT_TOKENS_FLOOR: u32 = 1_024;
/// 为 provider 的隐性开销（模板 token、role 分隔符、思考链预留等）预留的安全余量。
const CONTEXT_WINDOW_SAFETY_MARGIN_TOKENS: usize = 2_048;

/// 估算 messages 的 prompt token 数（保守：高估）。无服务端 usage 反馈时以字符
/// 数近似（每 token ~2 字符）。
fn estimate_prompt_tokens(messages: &[Message]) -> usize {
    let chars: usize = messages
        .iter()
        .map(|m| super::history::value_to_string(&m.content).chars().count())
        .sum();
    chars.div_ceil(CHARS_PER_TOKEN_CONSERVATIVE)
}

/// 估算工具 schema 的 prompt token 数。工具定义（name/description/JSON Schema）
/// 会随每次请求发送并计入 prompt 占用；启用大量工具/MCP 时体积可观。以序列化后
/// 的字符数按同一保守换算折算。`None` / 空工具集贡献 0。
fn estimate_tools_tokens(tools: Option<&Value>) -> usize {
    let Some(tools) = tools else {
        return 0;
    };
    let chars = serde_json::to_string(tools)
        .map(|s| s.chars().count())
        .unwrap_or(0);
    chars.div_ceil(CHARS_PER_TOKEN_CONSERVATIVE)
}

/// 按「剩余上下文窗口」钳制单次请求的输出上限：
/// `min(model_max, window - est_prompt - safety_margin)`，并 floor 到
/// [`MIN_OUTPUT_TOKENS_FLOOR`]。这样即使模型声明了很大的 max_output_tokens，
/// 在高占用 prompt 下也不会 prompt + 输出一起超过 token 窗口。
fn clamp_max_tokens_for_prompt(
    model: &str,
    messages: &[Message],
    tools: Option<&Value>,
    model_max: u32,
    known_prompt_tokens: Option<u64>,
) -> u32 {
    let window = models::context_window_tokens(model);
    // 本轮实际消息量的字符估算（保守：每 token ~2 字符）。工具 schema 也会随请求
    // 一起发送、占用 prompt 窗口，故把其序列化长度折算进 prompt token——启用大量
    // 工具/MCP 时不计入会显著高估可用输出预算，导致 prompt+输出撞爆窗口。
    let est_prompt = estimate_prompt_tokens(messages) + estimate_tools_tokens(tools);
    // 优先使用服务端返回的实际 prompt_tokens，比字符估算精确得多。但该值来自
    // *上一轮* 请求：若这一轮刚发生历史压缩，prompt 骤降，而回填的 known 仍是
    // 压缩前的高值——直接用它会把 remaining 误算成接近 0，clamp 触底到
    // MIN_OUTPUT_TOKENS_FLOOR(1024)。always-thinking 模型(GLM)拿 1024 预算会被
    // reasoning 全部吃光 → completion=0 可见文本 → 截断重试死循环。
    // 因此对 known 设上界：正常轮 known ≈ est + tools(~1.2x est)，压缩轮 known 会
    // 远超 est(数倍)。超过 2×est 判定为陈旧，回退到本轮字符估算。
    let est_prompt = match known_prompt_tokens.map(|p| p as usize) {
        Some(known) if est_prompt > 0 => known.min(est_prompt.saturating_mul(2)),
        Some(known) => known,
        None => est_prompt,
    };
    let remaining = window
        .saturating_sub(est_prompt)
        .saturating_sub(CONTEXT_WINDOW_SAFETY_MARGIN_TOKENS);
    let remaining = u32::try_from(remaining).unwrap_or(u32::MAX);
    model_max.min(remaining).max(MIN_OUTPUT_TOKENS_FLOOR)
}

fn resolve_reasoning_wire_controls<'a>(
    model: &'a str,
    endpoint: &str,
    enable_thinking: bool,
    reasoning_effort: Option<&'a str>,
) -> (Map<String, Value>, Option<&'a str>, Option<Value>) {
    let adapter_kind = models::model_adapter(model);
    let adapter = super::provider::adapter_for(adapter_kind, &endpoint);
    let request_model = models::request_model_name(model);
    let thinking_dialect =
        super::provider::thinking_dialect_for(adapter_kind, &request_model, &endpoint);
    let top_level_reasoning_effort = adapter.reasoning_top_level(reasoning_effort);
    let thinking = thinking_dialect.fields(enable_thinking, top_level_reasoning_effort);
    let nested_reasoning = adapter.reasoning_nested(reasoning_effort);
    (thinking, top_level_reasoning_effort, nested_reasoning)
}

fn ensure_reasoning_content_echo_for_thinking_model(model: &str, messages: &mut [Message]) {
    let adapter_kind = models::model_adapter(model);
    let endpoint = models::endpoint_for_model(model, "");
    let request_model = models::request_model_name(model);
    let dialect = super::provider::thinking_dialect_for(adapter_kind, &request_model, &endpoint);
    if !dialect.requires_reasoning_content_echo() {
        return;
    }

    for message in messages.iter_mut() {
        if message.role != "assistant" || message.reasoning_content.is_some() {
            continue;
        }
        if message
            .tool_calls
            .as_ref()
            .is_some_and(|tool_calls| !tool_calls.is_empty())
        {
            message.reasoning_content = Some(String::new());
        }
    }
}

/// 把 provider adapter 给出的思考字段合并进辅助/后台请求体。
///
/// 辅助（非主链路）与后台请求固定关闭思考（`enable_thinking=false`），
/// 由各 adapter 决定具体写哪些 key（`enable_thinking:false` /
/// `thinking:{"type":"disabled"}` / 或空），核心层不再判别 provider。
pub(in crate::ai) fn apply_aux_thinking_fields(model: &str, body: &mut Value) {
    let endpoint = models::endpoint_for_model(model, "");
    let (fields, _, _) = resolve_reasoning_wire_controls(model, &endpoint, false, None);
    if fields.is_empty() {
        return;
    }
    if let Some(map) = body.as_object_mut() {
        for (key, value) in fields {
            map.insert(key, value);
        }
    }
}

/// 是否开启 opt-in 的显式 prompt cache 断点注入。
///
/// `cache_control` 是 provider/model 级能力，由 `models.json` 的
/// `explicit_prompt_cache` 字段声明；普通 OpenAI 兼容模型不一定接受该扩展字段。
fn prompt_cache_enabled_for_model(model: &str) -> bool {
    prompt_cache_config_enabled() && models::explicit_prompt_cache_enabled(model)
}

fn prompt_cache_config_enabled() -> bool {
    configw::get_all_config()
        .get(
            crate::ai::config_schema::AiConfig::PROMPT_CACHE_ENABLE,
            "true",
        )
        .trim()
        .eq_ignore_ascii_case("true")
}

/// 把首条 system / internal_note 消息的纯文本内容改写为带 `cache_control`
/// 的内容块数组，作为显式 prompt 缓存断点。仅在内容当前是字符串时转换，
/// 幂等且不会触碰其它消息。
fn apply_prompt_cache_breakpoint(messages: &mut [Message]) {
    for message in messages.iter_mut() {
        if !is_system_like_role(&message.role) {
            continue;
        }
        if let Value::String(text) = &message.content {
            message.content = json!([
                {
                    "type": "text",
                    "text": text,
                    "cache_control": { "type": "ephemeral" }
                }
            ]);
        }
        // 只在第一条 system-like 消息上设置断点即可。
        break;
    }
}

/// 解析当前会话生效的推理强度档位，按优先级从高到低：
/// 1. CLI 参数 `--reasoning-effort` 或 `/model effort <x>` 留下的覆盖
///    （存储在 [`App.cli.reasoning_effort_override`]，其中 `Some(None)`
///    表示用户显式关闭，`None` 表示未设置）；
/// 2. [models.json](../../../models.json) 中该模型的默认 `reasoning_effort`；
/// 3. `None` —— 不注入字段，保持服务端默认行为。
pub(super) fn resolve_reasoning_effort(app: &App, model: &str) -> Option<ReasoningEffort> {
    if let Some(override_value) = app.cli.reasoning_effort_override.as_ref() {
        return *override_value;
    }
    models::default_reasoning_effort(model)
}

/// 使用 LLM 进行 JSON 格式的请求（用于意图识别、知识库问答等场景）。
///
/// `skip_reasoning_effort`：为 `true` 时强制不注入 `reasoning_effort` 字段，
/// 即使模型默认配置了该参数也会被忽略。适用于知识库问答等轻量场景。
pub async fn do_request_json(
    app: &App,
    model: &str,
    messages: &[serde_json::Value],
    stream: bool,
    skip_reasoning_effort: bool,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    clear_stale_request_interrupt_before_request(app);

    let request_model = models::request_model_name(model);
    let mut request_body = json!({
        "model": request_model,
        "messages": messages,
        "stream": stream,
    });

    let endpoint = endpoint_for_request_model(app, model);
    let resolved_reasoning_effort = (!skip_reasoning_effort)
        .then(|| resolve_reasoning_effort(app, model).map(|effort| effort.as_str()))
        .flatten();
    let (thinking_fields, top_level_reasoning_effort, nested_reasoning) =
        resolve_reasoning_wire_controls(model, &endpoint, false, resolved_reasoning_effort);

    // 兼容（Qwen 等）provider 默认开启 thinking，会生成超长推理链。
    // 非流式辅助请求（意图识别、知识整理等）必须等整段生成完才返回响应头，
    // thinking 链一长就撑爆 60s 超时、重试也只是重复同样的慢生成。
    // 因此这里显式关闭 thinking，与后台 background_call 保持一致。
    if let Some(map) = request_body.as_object_mut() {
        for (key, value) in thinking_fields {
            map.insert(key, value);
        }
        if let Some(value) = top_level_reasoning_effort {
            map.insert("reasoning_effort".to_string(), json!(value));
        }
        if let Some(value) = nested_reasoning {
            map.insert("reasoning".to_string(), value);
        }
    }

    for attempt in 1..=REQUEST_MAX_ATTEMPTS {
        let endpoint = endpoint_for_request_model(app, model);
        let api_key = api_key_for_request_model(app, model);
        let t0 = Instant::now();
        // 非流式辅助请求：每次尝试 60 秒超时
        let send_future = async {
            let resp = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
                .header("Content-Type", "application/json")
                .json(&request_body)
                .send()
                .await?;
            Ok::<_, reqwest::Error>(resp)
        };
        let response = match tokio::time::timeout(Duration::from_secs(60), send_future).await {
            Ok(result) => result,
            Err(_) => {
                if attempt < REQUEST_MAX_ATTEMPTS {
                    eprintln!(
                        "[Warning] {}do_request_json timeout (60s), retrying (attempt {}/{})",
                        retry_scope_tag(),
                        attempt,
                        REQUEST_MAX_ATTEMPTS
                    );
                    continue;
                }
                return Err("do_request_json: all attempts timed out".into());
            }
        };

        match response {
            Ok(response) => {
                if response.status().is_success() {
                    let json: serde_json::Value = response.json().await?;
                    // AIOS: bridge non-stream usage to kernel `/dev/llm`.
                    if let Some(usage_val) = json.get("usage") {
                        if let Ok(usage) = serde_json::from_value::<StreamUsage>(usage_val.clone())
                        {
                            let usage = usage.normalized();
                            let latency_ms = t0.elapsed().as_millis().min(u64::MAX as u128) as u64;
                            let _ = charge_llm_usage_to_kernel(app, model, &usage, latency_ms);
                        }
                    }
                    return Ok(json);
                }
                let status = response.status();
                let status_code = status.as_u16();
                // 在消费 body 前读取 Retry-After 头
                let retry_after_delay = if status_code == 429 {
                    parse_retry_after(response.headers())
                } else {
                    None
                };
                let body = (response.text().await).unwrap_or_default();
                let err = RequestError::status(status, body);
                if should_retry_status(status) && attempt < REQUEST_MAX_ATTEMPTS {
                    let delay = retry_after_delay.unwrap_or_else(|| retry_delay(attempt));
                    if sleep_with_cancel(app, delay).await {
                        return Err(Box::new(RequestError::cancelled(
                            "request canceled by user during retry wait",
                        )));
                    }
                    continue;
                }
                return Err(err.into());
            }
            Err(err) => {
                if is_retryable_reqwest_error(&err) && attempt < REQUEST_MAX_ATTEMPTS {
                    let delay = retry_delay(attempt);
                    if sleep_with_cancel(app, delay).await {
                        return Err(Box::new(RequestError::cancelled(
                            "request canceled by user during retry wait",
                        )));
                    }
                    continue;
                }
                return Err(err.into());
            }
        }
    }

    Err("request failed after all attempts".into())
}

/// 流式聚合请求：发起 `stream: true` 请求，逐 chunk 累积 `delta.content`，
/// 最终返回拼接好的完整文本。
///
/// 相比非流式的 [`do_request_json`]，流式链路的响应头会立即返回、数据按
/// chunk 增量到达，因此**不会**被"等服务端整段 body 生成完"撑爆超时。
/// 这里用「单 chunk 空闲超时」兜底：只要持续有数据到达就一直读，连续
/// `STREAM_RESPONSE_HEADER_TIMEOUT_SECS` 秒没有任何 chunk 才判定卡死。
///
/// 适用于知识整理这类「只需要最终完整 JSON、不需要实时终端渲染」的辅助任务。
pub async fn do_request_text_streaming(
    app: &App,
    model: &str,
    messages: &[serde_json::Value],
) -> Result<String, Box<dyn std::error::Error>> {
    fn apply_stream_payload(
        payload: &str,
        content: &mut String,
        pending_usage: &mut Option<(String, StreamUsage)>,
    ) {
        let payload = payload.trim();
        if payload.is_empty() || payload == "[DONE]" {
            return;
        }
        if let Ok(chunk) = serde_json::from_str::<StreamChunk>(payload) {
            // 捕获 usage：OpenAI 兼容流式把最终 usage 放在 choices 为空的尾包，
            // 必须在取 choice 之前先 take 出来，否则会漏计。
            if let Some(usage) = chunk.usage {
                *pending_usage = Some((chunk.model.clone(), usage.normalized()));
            }
            if let Some(choice) = chunk.choices.into_iter().next() {
                content.push_str(&choice.delta.content);
            }
        }
    }

    clear_stale_request_interrupt_before_request(app);

    let request_model = models::request_model_name(model);
    let mut request_body = json!({
        "model": request_model,
        "messages": messages,
        "stream": true,
        // 显式索取流式 usage：DashScope compatible-mode 流式默认不返回 usage，
        // 不声明 include_usage 就无法统计 token、`/usage` 会漏计本次调用。
        "stream_options": { "include_usage": true },
    });

    // 兼容（Qwen 等）provider 默认开启 thinking：流式下推理链以 reasoning_content
    // 分片到达，而本函数只聚合 delta.content。思考阶段会让 content 长时间为空、
    // spinner 一直转，表现为"卡死"。consolidate 是结构化 JSON 任务，关闭 thinking。
    apply_aux_thinking_fields(model, &mut request_body);

    for attempt in 1..=REQUEST_MAX_ATTEMPTS {
        let endpoint = endpoint_for_request_model(app, model);
        let api_key = api_key_for_request_model(app, model);
        let build_request = || {
            apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
                .header("Content-Type", "application/json")
                .json(&request_body)
        };
        let retry_policy = request_retry_policy_for_current_context();
        // 等待响应头：握手 + 服务端开始返回。流式下这一步很快。
        // 使用 hedged backup：如果 primary 在短时间内没响应，自动发 backup 请求。
        // 与非流式路径一致，header 等待超时取自 retry_policy.header_timeout_secs
        // （auto 子 agent 走 30s 而非硬编码 90s），chunk 空闲超时仍用固定常量。
        let mut response = match tokio::time::timeout(
            Duration::from_secs(retry_policy.header_timeout_secs),
            send_with_hedged_backup(
                build_request,
                retry_policy.hedged_backup_after_secs(),
                retry_policy.hedged_max_sends(),
            ),
        )
        .await
        {
            Ok(Ok(resp)) => {
                if resp.status().is_success() {
                    resp
                } else {
                    let status = resp.status();
                    let body = (resp.text().await).unwrap_or_default();
                    let err = RequestError::status(status, body);
                    if should_retry_status(status) && attempt < REQUEST_MAX_ATTEMPTS {
                        let delay = retry_delay(attempt);
                        if sleep_with_cancel(app, delay).await {
                            return Err(Box::new(RequestError::cancelled(
                                "request canceled by user during retry wait",
                            )));
                        }
                        continue;
                    }
                    return Err(err.into());
                }
            }
            Ok(Err(err)) => {
                if is_retryable_reqwest_error(&err) && attempt < REQUEST_MAX_ATTEMPTS {
                    let delay = retry_delay(attempt);
                    if sleep_with_cancel(app, delay).await {
                        return Err(Box::new(RequestError::cancelled(
                            "request canceled by user during retry wait",
                        )));
                    }
                    continue;
                }
                return Err(err.into());
            }
            Err(_) => {
                if attempt < REQUEST_MAX_ATTEMPTS {
                    eprintln!(
                        "[Warning] {}do_request_text_streaming 等待响应头超时 ({}s), retrying (attempt {}/{})",
                        retry_scope_tag(),
                        retry_policy.header_timeout_secs,
                        attempt,
                        REQUEST_MAX_ATTEMPTS
                    );
                    continue;
                }
                return Err("do_request_text_streaming: all attempts timed out".into());
            }
        };

        // 逐 chunk 读取并聚合 delta.content。
        let mut content = String::new();
        let mut buffer: Vec<u8> = Vec::new();
        let mut sse_event_data = String::new();
        let mut idle_timed_out = false;
        // final chunk 携带的 usage（OpenAI 兼容流式：通常在 choices 为空的尾包返回）。
        let mut pending_usage: Option<(String, StreamUsage)> = None;
        let t0 = std::time::Instant::now();
        loop {
            let chunk = match tokio::time::timeout(
                Duration::from_secs(STREAM_RESPONSE_HEADER_TIMEOUT_SECS),
                response.chunk(),
            )
            .await
            {
                Ok(Ok(Some(bytes))) => bytes,
                Ok(Ok(None)) => break, // 流正常结束
                Ok(Err(_)) => break,   // 读取错误：用已聚合内容
                Err(_) => {
                    idle_timed_out = true;
                    break;
                }
            };
            buffer.extend_from_slice(&chunk);
            // 按 SSE 事件边界聚合 `data:` 行，兼容标准的多行 payload。
            while let Some(pos) = buffer.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = buffer.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line);
                let trimmed = line.trim_end_matches(['\r', '\n']);
                if trimmed.is_empty() {
                    apply_stream_payload(&sse_event_data, &mut content, &mut pending_usage);
                    sse_event_data.clear();
                    continue;
                }
                if trimmed.starts_with(':') {
                    continue;
                }
                let Some(payload) = trimmed.strip_prefix("data:") else {
                    continue;
                };
                let payload = payload.strip_prefix(' ').unwrap_or(payload);
                if !sse_event_data.is_empty() {
                    sse_event_data.push('\n');
                }
                sse_event_data.push_str(payload);
            }
        }
        if !buffer.is_empty() {
            let line = String::from_utf8_lossy(&buffer);
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if let Some(payload) = trimmed.strip_prefix("data:") {
                let payload = payload.strip_prefix(' ').unwrap_or(payload);
                if !sse_event_data.is_empty() {
                    sse_event_data.push('\n');
                }
                sse_event_data.push_str(payload);
            }
        }
        apply_stream_payload(&sse_event_data, &mut content, &mut pending_usage);

        // AIOS: 把本次流式辅助请求的 usage 落账到内核 `/dev/llm`，与主链路一致。
        if let Some((echoed_model, usage)) = pending_usage {
            let model_for_pricing = if echoed_model.is_empty() {
                model
            } else {
                echoed_model.as_str()
            };
            let latency_ms = t0.elapsed().as_millis().min(u64::MAX as u128) as u64;
            let _ = charge_llm_usage_to_kernel(app, model_for_pricing, &usage, latency_ms);
        }

        if idle_timed_out && content.is_empty() && attempt < REQUEST_MAX_ATTEMPTS {
            eprintln!(
                "[Warning] {}do_request_text_streaming chunk 空闲超时 ({}s) 且无内容, retrying (attempt {}/{})",
                retry_scope_tag(),
                STREAM_RESPONSE_HEADER_TIMEOUT_SECS,
                attempt,
                REQUEST_MAX_ATTEMPTS
            );
            continue;
        }
        return Ok(content);
    }

    Err("do_request_text_streaming: failed after all attempts".into())
}

pub(super) async fn summarize_history_via_model(
    app: &App,
    messages: &[Message],
    max_chars: usize,
) -> Option<String> {
    if messages.is_empty() || max_chars == 0 {
        return None;
    }

    let transcript = messages_to_markdown(messages, &app.session_id);
    // 三段式截断：head 12k + middle 关键命中 4k + tail 6k，总计 22k 字符。
    // 比原先 head 16k + tail 6k 多保留中段的 error/fix/decision 行，避免
    // 摘要器只看见"开头任务陈述 + 末尾收尾"而漏掉中段关键改动。
    let transcript = if transcript.chars().count() > 24_000 {
        let head: String = transcript.chars().take(12_000).collect();
        let tail: String = transcript
            .chars()
            .rev()
            .take(6_000)
            .collect::<String>()
            .chars()
            .rev()
            .collect();

        // 中段关键行抽取：从 head 之后、tail 之前的中间部分挑选 error/fail/panic/
        // fix/diff/apply_patch/decision 等关键标记行，控制在 4k 字符内。
        let total_chars = transcript.chars().count();
        let mid_start_chars = 12_000usize;
        let mid_end_chars = total_chars.saturating_sub(6_000);
        let middle_segment: String = if mid_end_chars > mid_start_chars {
            transcript
                .chars()
                .skip(mid_start_chars)
                .take(mid_end_chars - mid_start_chars)
                .collect()
        } else {
            String::new()
        };
        let mut keypoints = String::new();
        let mut keypoint_chars = 0usize;
        const MID_KEYPOINTS_BUDGET: usize = 4_000;
        for line in middle_segment.lines() {
            let lower = line.to_lowercase();
            let is_key = lower.contains("error")
                || lower.contains("fail")
                || lower.contains("panic")
                || lower.contains("fix")
                || lower.contains("diff")
                || lower.contains("apply_patch")
                || lower.contains("write_file")
                || lower.contains("decision")
                || lower.contains("conclusion")
                || lower.contains("结论")
                || lower.contains("修复")
                || lower.contains("错误");
            if !is_key {
                continue;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let chunk_len = trimmed.chars().count() + 1;
            if keypoint_chars + chunk_len > MID_KEYPOINTS_BUDGET {
                break;
            }
            keypoints.push_str(trimmed);
            keypoints.push('\n');
            keypoint_chars += chunk_len;
        }

        if keypoints.trim().is_empty() {
            format!("{head}\n\n[... older transcript omitted for summary budget ...]\n\n{tail}")
        } else {
            format!(
                "{head}\n\n[... middle segment compressed; keypoints below ...]\n{keypoints}\n[... end of middle keypoints ...]\n\n{tail}"
            )
        }
    } else {
        transcript
    };

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(format!(
                "你是一个软件开发对话历史压缩器。你的任务是把较早对话压缩成后续 coding agent 能继续工作的摘要。\n\
输出要求：\n\
- 只输出纯文本，不要 markdown 代码块，不要解释。\n\
- 必须保留：用户明确要求、文件路径/函数名/工具名、关键报错、修复结论、当前工作、未完成任务。\n\
- 优先保留事实和决定，删除寒暄、重复确认、冗长日志。\n\
- 使用下面这些标题，并且每个标题下用 `- ` 开头的短行：\n\
主要请求:\n关键上下文:\n错误与修复:\n当前工作:\n待办任务:\n已知工具结论:\n\
- 如果某项没有内容，写 `- 无`。\n\
- 总长度尽量控制在 {} 个字符以内。",
                max_chars
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(format!("请压缩下面的较早对话：\n\n{}", transcript)),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let control_model = control_model_for_aux_tasks(app);
    let request_body = build_request_body(
        &control_model,
        &messages,
        false,
        false,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let endpoint = endpoint_for_request_model(app, &control_model);
    let api_key = api_key_for_request_model(app, &control_model);
    // 历史摘要是 turn 收尾的后台辅助请求（任务边界压缩会在每次答案交付后触发）。
    // 主 client 只有 connect_timeout、没有整体 timeout，若摘要模型接受连接后迟迟
    // 不返回响应头，这里的裸 .send()/.text() 会永久阻塞、CPU 0，表现为“答案已输出
    // 但迟迟不回到提示符”的卡死。用显式超时兜底，超时即放弃摘要（保持原始历史）。
    let send_future = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send();
    let response = match tokio::time::timeout(Duration::from_secs(60), send_future).await {
        Ok(r) => r.ok()?,
        Err(_) => {
            eprintln!("[summary] timeout (60s) waiting for response headers, skipping");
            return None;
        }
    };
    if !response.status().is_success() {
        return None;
    }
    let text = match tokio::time::timeout(Duration::from_secs(30), response.text()).await {
        Ok(r) => r.ok()?,
        Err(_) => {
            eprintln!("[summary] timeout (30s) reading response body, skipping");
            return None;
        }
    };
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = extract_router_content(&v)?;
    let trimmed = content.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// 用 LLM 为当前对话生成一个简短的概括性标题（不超过 20 字）。
/// 供 session 列表和输入框顶部展示使用。
pub(super) async fn generate_session_title_via_model(
    app: &App,
    messages: &[crate::ai::history::Message],
) -> Option<String> {
    use crate::ai::history::{is_system_like_role, value_to_string};

    if messages.is_empty() {
        return None;
    }

    // 只取最近的对话内容用于生成标题（最多 8000 字符）
    let dialog: Vec<String> = messages
        .iter()
        .filter(|m| !is_system_like_role(&m.role))
        .map(|m| {
            let role = match m.role.as_str() {
                "user" => "用户",
                "assistant" => "助手",
                "tool" => "工具",
                _ => m.role.as_str(),
            };
            // 去掉图片内容，只保留文本，避免 LLM 看到 base64 数据生成无意义的标题
            let text_only = normalize_message_content_for_text_only_model(&m.content);
            let content = value_to_string(&text_only);
            format!("{role}: {content}")
        })
        .collect();

    if dialog.is_empty() {
        return None;
    }

    let mut transcript = dialog.join("\n");
    if transcript.chars().count() > 8000 {
        transcript = transcript.chars().take(8000).collect();
    }

    let system_prompt = "你是一个对话标题生成器。根据下面的对话内容，生成一个不超过20个字的简短标题，概括对话的核心主题。\n\
要求：\n\
- 只输出标题本身，不要引号，不要解释，不要前缀。\n\
- 标题要具体、有信息量，不要太笼统。\n\
- 优先用名词短语或动宾短语。\n\
- 如果是编程相关，提到关键技术或文件名。";

    let user_prompt = format!("对话内容：\n\n{transcript}\n\n请生成标题：");

    let title_messages = vec![
        crate::ai::history::Message {
            role: "system".to_string(),
            content: serde_json::Value::String(system_prompt.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        crate::ai::history::Message {
            role: "user".to_string(),
            content: serde_json::Value::String(user_prompt),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let control_model = control_model_for_aux_tasks(app);
    let request_body = build_request_body(
        &control_model,
        &title_messages,
        false,
        false,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let endpoint = endpoint_for_request_model(app, &control_model);
    let api_key = api_key_for_request_model(app, &control_model);

    let send_future = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send();

    let response = match tokio::time::timeout(
        std::time::Duration::from_secs(SESSION_TITLE_REQUEST_TIMEOUT_SECS),
        send_future,
    )
    .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            eprintln!("[session-title] request error: {e}");
            return None;
        }
        Err(_) => {
            eprintln!(
                "[session-title] timeout ({}s) sending request, skipping",
                SESSION_TITLE_REQUEST_TIMEOUT_SECS
            );
            return None;
        }
    };

    let status = response.status();
    if !status.is_success() {
        eprintln!("[session-title] HTTP {status}, skipping");
        return None;
    }

    let text = match tokio::time::timeout(
        std::time::Duration::from_secs(SESSION_TITLE_BODY_TIMEOUT_SECS),
        response.text(),
    )
    .await
    {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => {
            eprintln!("[session-title] body read error: {e}");
            return None;
        }
        Err(_) => {
            eprintln!(
                "[session-title] timeout ({}s) reading body, skipping",
                SESSION_TITLE_BODY_TIMEOUT_SECS
            );
            return None;
        }
    };

    let v: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[session-title] JSON parse error: {e}");
            return None;
        }
    };
    let content = match extract_router_content(&v) {
        Some(c) => c,
        None => {
            eprintln!("[session-title] extract_router_content returned None");
            return None;
        }
    };
    let trimmed = content.trim().to_string();

    // 清理：去掉引号、去掉换行、截断到 30 字符
    let cleaned = trimmed
        .trim_matches(|c: char| {
            c == '"' || c == '「' || c == '」' || c == '\'' || c.is_whitespace()
        })
        .lines()
        .next()
        .unwrap_or("")
        .to_string();

    if cleaned.is_empty() {
        return None;
    }

    // 截断到 30 字符（中文一个字算一个 char）
    let result: String = if cleaned.chars().count() > 30 {
        cleaned.chars().take(30).collect()
    } else {
        cleaned
    };

    Some(result)
}

/// AIOS bridge: take a parsed OpenAI-compatible `StreamUsage` (plus the
/// requested model name and latency) and hand it to the kernel's LLM device
/// for accounting. This is the single chokepoint where agent-land meets
/// `/dev/llm`; every LLM call site must route through here instead of
/// dropping usage on the floor.
///
/// The kernel takes care of:
///   - converting prompt/completion tokens to cost_micros (via `llm_price`)
///   - calling `rusage_charge` so rlimit enforcement stays authoritative
///   - emitting a `trace_event("llm.account", ...)` for observability
pub(crate) fn charge_llm_usage_to_kernel(
    app: &App,
    requested_model: &str,
    usage: &StreamUsage,
    latency_ms: u64,
) -> Option<aios_kernel::primitives::LlmAccountOutcome> {
    charge_llm_usage_via_kernel(&app.os, requested_model, usage, latency_ms)
}

/// 与 [`charge_llm_usage_to_kernel`] 等价，但直接接受一个 `SharedKernel`。
/// 供没有 `App` 句柄的调用方（如后台 reflection 的 `background_call`）使用——
/// `GLOBAL_OS` 与 `App.os` 共享同一把 `Arc<Mutex<Kernel>>`，落账语义一致。
pub(crate) fn charge_llm_usage_via_kernel(
    os: &aios_kernel::kernel::SharedKernel,
    requested_model: &str,
    usage: &StreamUsage,
    latency_ms: u64,
) -> Option<aios_kernel::primitives::LlmAccountOutcome> {
    // Fast path: a zero-usage report is noise.
    if usage.prompt_tokens == 0 && usage.completion_tokens == 0 {
        return None;
    }
    let cached = usage
        .prompt_tokens_details
        .as_ref()
        .map(|d| d.cached_tokens)
        .unwrap_or(0);
    let report = aios_kernel::primitives::LlmUsageReport {
        model: requested_model.to_string(),
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        cached_prompt_tokens: cached,
        latency_ms,
    };
    // 在内核里落账（计费 + rusage + trace + 追加审计账本），同时拿出本次需要
    // drain 落库的增量记录。SQLite I/O 放到 guard 释放之后，避免持内核锁做磁盘写。
    let (outcome, drained, head) = {
        let mut guard = match os.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let pid = guard.current_process_id()?;
        let outcome = guard.llm_account(pid, report);
        let cursor = crate::ai::tools::storage::token_usage_store::drain_cursor();
        let drained = guard.llm_usage_drain_since(cursor);
        let head = guard.llm_usage_head_seq();
        (outcome, drained, head)
    };
    // best-effort 落库到独立的 token 用量统计表，失败不影响主流程。
    crate::ai::tools::storage::token_usage_store::persist_drained(&drained, head);
    Some(outcome)
}

#[cfg(test)]
mod tests;
