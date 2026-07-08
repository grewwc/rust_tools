//! 请求错误类型、重试策略与请求配置辅助函数。
//!
//! 包含：
//! - `RequestError` / `RequestErrorKind`：统一的请求错误类型
//! - `RequestRetryPolicy`：重试次数/超时/对冲策略
//! - 重试辅助：`send_with_hedged_backup` / `sleep_with_cancel` / `retry_delay`
//! - 模型错误分类：`is_transient_error` / `should_temporarily_disable_model` / `should_try_model_fallback`
//! - 请求配置辅助：`endpoint_for_request_model` / `api_key_for_request_model` / `apply_request_auth`

use std::fmt;
use std::time::Duration;

use reqwest::{Response, StatusCode};

use crate::ai::config_schema::AiConfig;
use crate::ai::models;
use crate::ai::types::App;
use crate::commonw::configw;

#[derive(Debug)]
pub(crate) enum RequestErrorKind {
    Network,
    Status(StatusCode),
}

#[derive(Debug)]
pub(crate) struct RequestError {
    pub(crate) kind: RequestErrorKind,
    pub(crate) message: String,
}

impl RequestError {
    pub(crate) fn network(err: reqwest::Error) -> Self {
        Self {
            kind: RequestErrorKind::Network,
            message: err.to_string(),
        }
    }

    pub(crate) fn cancelled(message: impl Into<String>) -> Self {
        Self {
            kind: RequestErrorKind::Network,
            message: message.into(),
        }
    }

    pub(crate) fn status(status: StatusCode, body: String) -> Self {
        Self {
            kind: RequestErrorKind::Status(status),
            message: if body.trim().is_empty() {
                format!("request failed: {}", status)
            } else {
                format!("request failed: {} {}", status, body)
            },
        }
    }
}

impl fmt::Display for RequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RequestError {}

pub(crate) const REQUEST_MAX_ATTEMPTS: usize = 6;
pub(crate) const REQUEST_MAX_ATTEMPTS_429: usize = 32; // 429 错误重试 32 次
pub(crate) const REQUEST_RETRY_BASE_MS: u64 = 500;
pub(crate) const REQUEST_RETRY_MAX_MS: u64 = 16000;
/// 流式请求等待响应头（首字节）的超时。
///
/// 主 `app.client` 仅保留 `connect_timeout`（不设置整体 `.timeout()`，
/// 否则会误杀长时间的流式 body 读取）。但 `connect_timeout` 只覆盖 TCP/TLS
/// 握手，不覆盖“连接已建立、服务端迟迟不返回响应头”的场景——此时
/// `.send().await` 会永久阻塞、CPU 占用为 0，表现为 agent 卡死。
/// 因此对流式 `send()` 单独加一个响应头等待超时兜底。
pub(crate) const STREAM_RESPONSE_HEADER_TIMEOUT_SECS: u64 = 90;
/// 子 agent 自动选型有 fallback 兜底，首选模型迟迟不返回时应快速让位。
/// 显式指定模型不走 AUTO_MODEL_FALLBACK scope，仍保留常规重试策略。
pub(crate) const AUTO_SUBAGENT_RESPONSE_HEADER_TIMEOUT_SECS: u64 = 30;
pub(crate) const AUTO_SUBAGENT_REQUEST_MAX_ATTEMPTS: usize = 1;
pub(crate) const DEFAULT_AUTO_THINKING_THRESHOLD: f64 = 0.7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RequestRetryPolicy {
    pub(crate) max_attempts: usize,
    pub(crate) max_attempts_429: usize,
    pub(crate) header_timeout_secs: u64,
    /// hedged backup 最大发送次数（含 primary）。超过则最后一次不再设内部超时，
    /// 交给外层 `header_timeout_secs` 兜底。
    pub(crate) hedged_max_sends: usize,
}

impl RequestRetryPolicy {
    /// 对冲请求（hedged backup）触发阈值：primary 请求在这段时间内没收到响应头
    /// 就并发发起一次完全相同的 backup 请求，二者竞速，落败者被 drop。
    /// 取 header_timeout 的 1/9，clamp 到 [3, 15] 秒。
    ///
    /// 这样在服务端偶发长尾（连接已建立但迟迟不返回响应头）时，不必等满 90s
    /// 再重试，而是在 ~10s 就自动发起新请求，显著降低尾延迟。
    pub(crate) fn hedged_backup_after_secs(&self) -> u64 {
        (self.header_timeout_secs / 9).clamp(3, 15)
    }

    /// 主请求 hedged backup 总发送次数。
    /// 服务端长尾偶尔会连续命中坏实例，多一次 backup 能进一步压低尾延迟，
    /// 注意：真正的并发对冲下，长尾场景最多会有 `hedged_max_sends` 个并发连接。
    pub(crate) fn hedged_max_sends(&self) -> usize {
        self.hedged_max_sends
    }
}

pub(crate) fn request_retry_policy(auto_model_fallback: bool) -> RequestRetryPolicy {
    if auto_model_fallback {
        RequestRetryPolicy {
            max_attempts: AUTO_SUBAGENT_REQUEST_MAX_ATTEMPTS,
            max_attempts_429: AUTO_SUBAGENT_REQUEST_MAX_ATTEMPTS,
            header_timeout_secs: AUTO_SUBAGENT_RESPONSE_HEADER_TIMEOUT_SECS,
            hedged_max_sends: 2, // 子 agent 超时短（30s），1 primary + 1 backup 足够
        }
    } else {
        RequestRetryPolicy {
            max_attempts: REQUEST_MAX_ATTEMPTS,
            max_attempts_429: REQUEST_MAX_ATTEMPTS_429,
            header_timeout_secs: STREAM_RESPONSE_HEADER_TIMEOUT_SECS,
            hedged_max_sends: 3, // 主请求 1 primary + 2 backup
        }
    }
}

pub(crate) fn request_retry_policy_for_current_context() -> RequestRetryPolicy {
    request_retry_policy(crate::ai::driver::runtime_ctx::auto_model_fallback_spec().is_some())
}

pub(crate) fn config_bool_is_true(value: Option<String>) -> bool {
    value
        .map(|v| v.trim().to_ascii_lowercase())
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
}

pub(crate) fn config_forces_thinking() -> bool {
    let cfg = configw::get_all_config();
    config_bool_is_true(cfg.get_opt(AiConfig::MODEL_THINKING))
}

pub(crate) fn endpoint_for_request_model(app: &App, model: &str) -> String {
    models::endpoint_for_model(model, &app.config.endpoint)
}

pub(crate) fn api_key_for_request_model(app: &App, model: &str) -> String {
    models::api_key_for_model(model, &app.config.api_key)
}

pub(crate) fn apply_request_auth(
    builder: reqwest::RequestBuilder,
    endpoint: &str,
    api_key: &str,
) -> reqwest::RequestBuilder {
    if api_key.trim().is_empty() && models::endpoint_supports_anonymous_auth(endpoint) {
        return builder;
    }
    builder.bearer_auth(api_key)
}

pub(crate) fn should_retry_status(status: StatusCode) -> bool {
    status.as_u16() == 429 || status.is_server_error()
}

pub(crate) fn is_retryable_reqwest_error(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect() || err.is_request()
}

pub(crate) fn retry_delay(attempt: usize) -> Duration {
    let shift = attempt.saturating_sub(1).min(4) as u32;
    let backoff = REQUEST_RETRY_BASE_MS.saturating_mul(1u64 << shift);
    Duration::from_millis(backoff.min(REQUEST_RETRY_MAX_MS))
}

/// 解析 HTTP `Retry-After` 响应头，返回建议的等待时长。
/// 支持秒数格式（`120`）。HTTP 日期格式暂不支持。
pub(crate) fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let val = headers.get(reqwest::header::RETRY_AFTER)?;
    let s = val.to_str().ok()?;
    if let Ok(secs) = s.trim().parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    None
}

/// 对冲请求（hedged backup request）。
///
/// 先发起 primary 请求；若在 `backup_after_secs` 内未收到响应头，则【不丢弃 primary】，
/// 并发发起一个完全相同的 backup 请求，让二者竞速——谁先返回响应头谁赢，落败者被
/// drop（仅取消客户端等待，服务端可能仍在处理）。若 `max_sends > 2`，则继续按
/// `backup_after_secs` 间隔逐步追加并发请求（最多 `max_sends` 个同时在飞），任一返回
/// 即作为结果，其余被 drop。所有请求发完仍无返回时，等待首个完成（由外层
/// `header_timeout_secs` 兜底）。
///
/// 典型场景：服务端接受了 TCP 连接但因内部排队迟迟不返回响应头，primary 请求
/// 会卡在 `.send().await` 上等满 90s 才超时。hedged backup 在 ~10s 就并发发起
/// 新连接，新请求通常能立即被处理（命中不同的后端实例或脱离原队列），
/// 从而把尾延迟从 90s+ 降到 10s+。与"先 drop 再顺序重发"不同，真正的并发对冲
/// 不会浪费"只差一点就能返回"的 primary——它仍有机会赢得竞速。
///
/// 注意：竞速中落败的请求会被 drop，只取消客户端的等待，服务端可能仍在处理。
/// 但 LLM chat completions 是幂等的读操作，重复请求不会产生副作用。代价是长尾
/// 场景下最多会有 `max_sends` 个并发连接（原"顺序 drop"实现为 1 个）。
pub(crate) async fn send_with_hedged_backup(
    build_request: impl Fn() -> reqwest::RequestBuilder,
    backup_after_secs: u64,
    max_sends: usize,
) -> Result<Response, reqwest::Error> {
    use futures_util::stream::{FuturesUnordered, StreamExt};
    let max_sends = max_sends.max(1);
    let hedge = Duration::from_secs(backup_after_secs);
    // 所有在飞的请求；任一返回响应头即作为结果，其余被 drop（取消等待）。
    let mut in_flight = FuturesUnordered::new();
    for round in 1..=max_sends {
        in_flight.push(build_request().send());
        // 竞速：集合中首个响应头返回 vs hedge 计时器
        tokio::select! {
            // 集合中首个 future 完成（返回即移出集合），其余继续在飞
            result = in_flight.next() => return result.expect("in_flight 非空"),
            _ = tokio::time::sleep(hedge) => {
                // hedge 窗口内无任何响应头到达：若还有配额则追加一个并发请求，
                // 否则落到循环后的最终竞速（由外层 header_timeout 兜底）。
                if round < max_sends {
                    eprintln!(
                        "[Info] 第 {round} 次请求 {}s 内未返回响应头，发起 hedged backup",
                        backup_after_secs
                    );
                }
            }
        }
    }
    // 已发起全部 max_sends 个请求且仍在飞行中：等待首个完成（由外层 header_timeout 兜底）
    in_flight
        .next()
        .await
        .expect("in_flight 必非空：至少发起过一次请求")
}

pub(crate) fn should_abort_retry_wait(app: &App) -> bool {
    app.shutdown.load(std::sync::atomic::Ordering::Relaxed)
        || app.cancel_stream.load(std::sync::atomic::Ordering::Relaxed)
        || crate::ai::driver::signal::request_interrupt_ready()
}

pub(crate) async fn sleep_with_cancel(app: &App, delay: Duration) -> bool {
    if should_abort_retry_wait(app) {
        return true;
    }

    tokio::select! {
        _ = tokio::time::sleep(delay) => should_abort_retry_wait(app),
        _ = crate::ai::driver::signal::wait_for_interrupt_sources(None, None) => true,
    }
}

pub(crate) fn clear_stale_request_interrupt_before_request(app: &App) {
    // 若上一次 turn 的中断信号残留（但当前并未处于显式 cancel/shutdown），
    // 会导致本次网络重试在 attempt1 就被短路为 canceled。
    if !app.shutdown.load(std::sync::atomic::Ordering::Relaxed)
        && !app.cancel_stream.load(std::sync::atomic::Ordering::Relaxed)
        && crate::ai::driver::signal::request_interrupt_ready()
    {
        crate::ai::driver::signal::clear_request_interrupt();
    }
}

pub(crate) fn control_model_for_aux_tasks(app: &App) -> String {
    app.config
        .intent_model
        .as_deref()
        .filter(|v| !v.trim().is_empty())
        .map(models::determine_model)
        .unwrap_or_else(|| app.current_model.trim().to_string())
}

pub(crate) fn is_transient_error(err: &RequestError) -> bool {
    match err.kind {
        RequestErrorKind::Network => true,
        RequestErrorKind::Status(status) => should_retry_status(status),
    }
}

pub(crate) fn should_temporarily_disable_model(err: &RequestError) -> bool {
    match err.kind {
        RequestErrorKind::Network => false,
        RequestErrorKind::Status(status) => {
            matches!(status.as_u16(), 402 | 404 | 429) || status.is_server_error()
        }
    }
}

pub(crate) fn should_temporarily_disable_auto_selected_model(err: &RequestError) -> bool {
    if should_temporarily_disable_model(err) {
        return true;
    }
    if !matches!(err.kind, RequestErrorKind::Network) {
        return false;
    }
    let message = err.message.to_ascii_lowercase();
    message.contains("timed out") || message.contains("timeout")
}

pub(crate) fn should_try_model_fallback(err: &RequestError) -> bool {
    match err.kind {
        RequestErrorKind::Network => true,
        RequestErrorKind::Status(status) => {
            matches!(status.as_u16(), 401 | 402 | 403 | 404 | 429) || status.is_server_error()
        }
    }
}

/// 判断流式响应中途出现的错误是否值得重试。
///
/// 流式错误（`provider stream error: ...`）发生在响应头已返回、body 传输过程中，
/// 通常是服务端瞬态问题（后端取消、上游 RPC 失败、内部错误等）。对这些错误重试
/// 一次整条请求+流可以显著降低用户感知到的失败率。
pub(crate) fn is_retryable_stream_error(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("server_error")
        || lower.contains("cancelled")
        || lower.contains("canceled")
        || lower.contains("upstream")
        || lower.contains("rpc error")
        || lower.contains("internal")
        || lower.contains("overloaded")
        || lower.contains("timeout")
        || lower.contains("timed out")
        || lower.contains("temporarily unavailable")
        || lower.contains("try again")
        || lower.contains("connection reset")
        || lower.contains("broken pipe")
        || lower.contains("unexpected eof")
        || lower.contains("502")
        || lower.contains("503")
        || lower.contains("504")
}
