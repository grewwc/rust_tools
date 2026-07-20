//! HTTP 传输层：请求发送、重试、超时、鉴权。
//!
//! 从 `request/mod.rs` 提取，职责仅限"把请求发出去并拿到响应"，
//! 与消息归一化、tool schema 构造、thinking dialect 等请求构建逻辑分离。

use std::time::{Duration, Instant};

use reqwest::Response;
use rust_tools::commonw;
use serde_json::json;

use super::super::{
    history::{Message, SessionStore, generate_session_summary},
    models,
    provider::adapter_for,
    types::App,
};
use crate::ai::theme::{ACCENT_MUTED, ACCENT_PRIMARY, ACCENT_SUCCESS, ACCENT_WARN, RESET};

use super::aux::charge_llm_usage_to_kernel;
use super::builder::{build_request_body, estimate_request_prompt_tokens};
use super::error::{
    REQUEST_MAX_ATTEMPTS, RequestError, RequestErrorKind, RequestRetryPolicy,
    STREAM_RESPONSE_HEADER_TIMEOUT_SECS, api_key_for_request_model, apply_request_auth,
    clear_stale_request_interrupt_before_request, config_forces_thinking,
    endpoint_for_request_model, is_retryable_reqwest_error, is_retryable_status_with_body,
    parse_retry_after, request_retry_policy_for_current_context, retry_delay, retry_delay_429,
    should_retry_status, should_rotate_key, sleep_with_cancel,
};
use super::normalize::{
    agent_tools_for_request, normalize_messages_for_model, request_tool_names_for_model,
    strip_unavailable_tool_hints_from_messages,
};
use super::reasoning::{
    apply_aux_thinking_fields, apply_prompt_cache_breakpoint,
    ensure_reasoning_content_echo_for_thinking_model, prompt_cache_enabled_for_model,
    resolve_reasoning_effort, resolve_reasoning_wire_controls,
};
use super::thinking::resolve_thinking;
use super::token_budget;
use super::types::{RequestBody, StreamChunk, StreamUsage};

/// 并发请求（前台 turn + 各子代理）各自独立重试，`attempt N/M` 计数互相
/// 交错、无法区分归属。用 aios 调度 pid 作为作用域标签把每条重试日志绑定到
/// 具体进程；无 pid（无 TASK_PID 作用域）时返回空串，日志退化为原样。
fn retry_scope_tag() -> String {
    match aios_kernel::kernel::current_task_pid() {
        Some(pid) => format!("[pid {pid}] "),
        None => String::new(),
    }
}

/// 带 TPM 预检的 hedged send。
///
/// 预算必须按 actual physical send 计，而不是按 logical request 计：hedged backup
/// 未触发时只占一次；长尾触发 backup 时，每个追加请求都会在发送前重新占一次。
/// 这样既避免 429，又不会因为按 `hedged_max_sends` 一次性预占而把正常吞吐压低 3 倍。
async fn send_with_budgeted_hedged_backup(
    app: &App,
    model: &str,
    endpoint: &str,
    request_model_label: &str,
    api_key: &str,
    estimated_prompt_tokens: usize,
    build_request: impl Fn() -> reqwest::RequestBuilder,
    backup_after_secs: u64,
    max_sends: usize,
) -> Result<Response, RequestError> {
    use futures_util::stream::{FuturesUnordered, StreamExt};

    let max_sends = max_sends.max(1);
    let hedge = Duration::from_secs(backup_after_secs);
    let mut in_flight = FuturesUnordered::new();

    for round in 1..=max_sends {
        token_budget::wait_for_request_budget(
            app,
            model,
            endpoint,
            request_model_label,
            api_key,
            estimated_prompt_tokens,
            1,
        )
        .await?;
        in_flight.push(build_request().send());
        tokio::select! {
            result = in_flight.next() => {
                return result
                    .expect("in_flight 非空")
                    .map_err(RequestError::network);
            }
            _ = tokio::time::sleep(hedge) => {
                if round < max_sends {
                    eprintln!(
                        "[Info] 第 {round} 次请求 {}s 内未返回响应头，发起 hedged backup request",
                        backup_after_secs
                    );
                }
            }
        }
    }

    match in_flight.next().await {
        Some(result) => result.map_err(RequestError::network),
        None => Err(RequestError {
            kind: RequestErrorKind::Network,
            message: "hedged request set unexpectedly empty".to_string(),
            retry_after: None,
        }),
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
    model: &str,
    api_key: &str,
    request_body: &RequestBody<'_>,
    retry_policy: &RequestRetryPolicy,
    endpoint: &str,
) -> Result<Response, RequestError> {
    let protocol = models::request_protocol_dialect(model, endpoint);
    let http_body = protocol.build_http_body(request_body);
    let estimated_prompt_tokens = token_budget::calibrate_prompt_tokens_for_budget(
        estimate_request_prompt_tokens(request_body.messages, request_body.tools.as_ref()),
        app.last_known_prompt_tokens,
        app.last_known_cached_prompt_tokens,
    );
    for attempt in 1..=retry_policy.max_attempts {
        let client = app.client.clone();
        let build_request = || {
            apply_request_auth(client.post(endpoint), endpoint, api_key)
                .header("Content-Type", "application/json")
                .json(&http_body)
        };
        let response = match tokio::time::timeout(
            Duration::from_secs(retry_policy.header_timeout_secs),
            send_with_budgeted_hedged_backup(
                app,
                model,
                endpoint,
                &request_body.model,
                api_key,
                estimated_prompt_tokens,
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
                let retryable = match &err.kind {
                    RequestErrorKind::Network => true,
                    _ => false,
                };
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
pub(crate) async fn do_request_messages(
    app: &mut App,
    model: &str,
    messages: &[Message],
    stream: bool,
) -> Result<Response, RequestError> {
    do_request_messages_with_tool_mode(app, model, messages, stream, true).await
}

/// 发起不暴露任何工具的请求。
///
/// 用于工具循环或迭代上限后的收口轮次。这里必须在 wire request 层移除
/// `tools`，而不能只依赖 prompt 要求模型停止调用工具。
pub(crate) async fn do_request_messages_without_tools(
    app: &mut App,
    model: &str,
    messages: &[Message],
    stream: bool,
) -> Result<Response, RequestError> {
    do_request_messages_with_tool_mode(app, model, messages, stream, false).await
}

async fn do_request_messages_with_tool_mode(
    app: &mut App,
    model: &str,
    messages: &[Message],
    stream: bool,
    tools_enabled: bool,
) -> Result<Response, RequestError> {
    clear_stale_request_interrupt_before_request(app);

    let mut normalized_messages = normalize_messages_for_model(model, messages);
    let request_tool_names = tools_enabled
        .then(|| request_tool_names_for_model(app, model))
        .unwrap_or_default();
    strip_unavailable_tool_hints_from_messages(&mut normalized_messages, &request_tool_names);
    if prompt_cache_enabled_for_model(model) {
        apply_prompt_cache_breakpoint(&mut normalized_messages);
    }
    let (tools_value, tool_choice) = if tools_enabled {
        agent_tools_for_request(app, model)
    } else {
        (None, None)
    };
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
    // 部分网关（如 bytedance modelhub）在 /v1/chat/completions 上拒绝
    // `tools` + `reasoning_effort` 同时出现（返回 400）。当模型声明了
    // `reasoning_effort_conflicts_with_tools` 且本轮请求携带 tools 时，
    // 自动省略 reasoning_effort 以避免 400；无 tools 时保留以维持 thinking。
    let reasoning_effort = if reasoning_effort.is_some()
        && tools_value.is_some()
        && models::reasoning_effort_conflicts_with_tools(model)
    {
        None
    } else {
        reasoning_effort
    };
    // 把 reasoning items 侧信道快照到局部，避免 `request_body` 长期持有对 `app`
    // 的不可变借用（后续 key 轮换循环需要 `&mut app`）。
    let turn_reasoning_items = app.turn_reasoning_items.clone();
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
        Some(&turn_reasoning_items),
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
            match request_messages_with_key(
                app,
                model,
                api_key,
                &request_body,
                &retry_policy,
                &endpoint,
            )
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
        let delay = round_retry_after.unwrap_or_else(|| retry_delay_429(attempt));
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

pub(crate) fn print_info(app: &App, model: &str) {
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
        .map(|title| crate::ai::history::normalize_generated_session_title(&title))
        .filter(|title| !title.is_empty())
        .or_else(|| {
            store
                .first_user_prompt(&app.session_id)
                .ok()
                .flatten()
                .map(|p| generate_session_summary(&p))
                .map(|summary| crate::ai::history::normalize_generated_session_title(&summary))
                .filter(|summary| !summary.is_empty())
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
        token_budget::wait_for_request_budget(
            app,
            model,
            &endpoint,
            &request_model,
            &api_key,
            token_budget::calibrate_prompt_tokens_for_budget(
                token_budget::estimate_json_request_tokens(&request_body),
                app.last_known_prompt_tokens,
                app.last_known_cached_prompt_tokens,
            ),
            1,
        )
        .await
        .map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })?;
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
        let retry_policy = request_retry_policy_for_current_context();
        let estimated_prompt_tokens = token_budget::estimate_json_request_tokens(&request_body);
        let client = app.client.clone();
        let build_request = || {
            apply_request_auth(client.post(&endpoint), &endpoint, &api_key)
                .header("Content-Type", "application/json")
                .json(&request_body)
        };
        // 等待响应头：握手 + 服务端开始返回。流式下这一步很快。
        // 使用 hedged backup：如果 primary 在短时间内没响应，自动发 backup 请求。
        // 与非流式路径一致，header 等待超时取自 retry_policy.header_timeout_secs
        // （auto 子 agent 走 30s 而非硬编码 90s），chunk 空闲超时仍用固定常量。
        let mut response = match tokio::time::timeout(
            Duration::from_secs(retry_policy.header_timeout_secs),
            send_with_budgeted_hedged_backup(
                app,
                model,
                &endpoint,
                &request_model,
                &api_key,
                estimated_prompt_tokens,
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
                let retryable = matches!(&err.kind, RequestErrorKind::Network);
                if retryable && attempt < REQUEST_MAX_ATTEMPTS {
                    let delay = retry_delay(attempt);
                    if sleep_with_cancel(app, delay).await {
                        return Err(Box::new(RequestError::cancelled(
                            "request canceled by user during retry wait",
                        )));
                    }
                    continue;
                }
                return Err(Box::new(err));
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
