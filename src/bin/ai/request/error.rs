//! 请求错误类型、重试策略与请求配置辅助函数。
//!
//! 包含：
//! - `RequestError` / `RequestErrorKind`：统一的请求错误类型
//! - `RequestRetryPolicy`：重试次数/超时/对冲策略
//! - 重试辅助：`sleep_with_cancel` / `retry_delay`
//! - 模型错误分类：`is_transient_error` / `should_temporarily_disable_model` / `should_try_model_fallback`
//! - 请求配置辅助：`endpoint_for_request_model` / `api_key_for_request_model` / `apply_request_auth`

use std::fmt;
use std::time::Duration;

use reqwest::StatusCode;

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
    /// 429 场景下服务端建议的重试等待时长（已在 `parse_retry_after` 中钳制）。
    /// 供 key 轮换/退避的上层逻辑读取；其它错误为 `None`。
    pub(crate) retry_after: Option<Duration>,
}

impl RequestError {
    pub(crate) fn network(err: reqwest::Error) -> Self {
        Self {
            kind: RequestErrorKind::Network,
            message: err.to_string(),
            retry_after: None,
        }
    }

    pub(crate) fn cancelled(message: impl Into<String>) -> Self {
        Self {
            kind: RequestErrorKind::Network,
            message: message.into(),
            retry_after: None,
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
            retry_after: None,
        }
    }

    /// 是否为 429（配额/限流）错误。
    pub(crate) fn is_rate_limited(&self) -> bool {
        matches!(self.kind, RequestErrorKind::Status(status) if status.as_u16() == 429)
    }
}

impl fmt::Display for RequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RequestError {}

pub(crate) const REQUEST_MAX_ATTEMPTS: usize = 12;
pub(crate) const REQUEST_MAX_ATTEMPTS_429: usize = 32; // 429 错误重试 32 次
pub(crate) const REQUEST_RETRY_BASE_MS: u64 = 500;
pub(crate) const REQUEST_RETRY_MAX_MS: u64 = 16000;
/// 429（配额超限）退避起始值。所有 key 均返回 429 时，首次等待 4 秒再重试，
/// 而非走通用退避的 0.5s——配额限流通常需要等待服务端窗口刷新，起步太短只会白白
/// 消耗请求次数。
pub(crate) const REQUEST_RETRY_429_BASE_MS: u64 = 4_000;
/// 429（配额超限）单次退避等待上限。服务端可能在 `Retry-After` 里返回极大的值
/// （到下个配额窗口的秒数，可达数万秒），若原样 sleep 会让进程长时间“卡死”。
/// 统一钳制到该上限，配合 key 轮换快速让位。
pub(crate) const REQUEST_RETRY_429_MAX_MS: u64 = 10_000;
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
            // 自动选型的模型 fallback 已是备用请求。若再在 3 秒后发送 hedged
            // backup，会复制同一个子 agent 推理，造成重复调度和额外消耗。
            hedged_max_sends: 1,
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

/// 判断 HTTP 状态码 + 响应体是否值得重试。
/// 在 `should_retry_status` 基础上额外覆盖：400 + body 含 "upstream" ——
/// relay/兼容层把上游瞬态失败（如上游 RPC 错误、内部异常）包成了 400，
/// 实际是瞬态错误，重试可恢复。与 `is_retryable_stream_error` 的 "upstream"
/// 判定保持一致。
pub(crate) fn is_retryable_status_with_body(status: StatusCode, body: &str) -> bool {
    if should_retry_status(status) {
        return true;
    }
    status.as_u16() == 400 && body.to_ascii_lowercase().contains("upstream")
}

pub(crate) fn is_retryable_reqwest_error(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect() || err.is_request()
}

pub(crate) fn retry_delay(attempt: usize) -> Duration {
    let shift = attempt.saturating_sub(1).min(4) as u32;
    let backoff = REQUEST_RETRY_BASE_MS.saturating_mul(1u64 << shift);
    Duration::from_millis(backoff.min(REQUEST_RETRY_MAX_MS))
}

/// 429 专用退避：base 4s，指数递增到 `REQUEST_RETRY_MAX_MS`（16s）。
/// 与通用 `retry_delay` 共享上界，但起始值更高，适配配额窗口刷新场景。
pub(crate) fn retry_delay_429(attempt: usize) -> Duration {
    let shift = attempt.saturating_sub(1).min(4) as u32;
    let backoff = REQUEST_RETRY_429_BASE_MS.saturating_mul(1u64 << shift);
    Duration::from_millis(backoff.min(REQUEST_RETRY_MAX_MS))
}

/// 解析 HTTP `Retry-After` 响应头，返回建议的等待时长。
/// 支持秒数格式（`120`）。HTTP 日期格式暂不支持。
/// 返回值钳制到 `REQUEST_RETRY_429_MAX_MS`，避免服务端返回超大值把进程睡死。
pub(crate) fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let val = headers.get(reqwest::header::RETRY_AFTER)?;
    let s = val.to_str().ok()?;
    if let Ok(secs) = s.trim().parse::<u64>() {
        let capped = secs.saturating_mul(1000).min(REQUEST_RETRY_429_MAX_MS);
        return Some(Duration::from_millis(capped));
    }
    None
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
        _ = crate::ai::driver::signal::wait_for_interrupt_sources(None, None, Some(app.cancel_stream.as_ref())) => true,
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
        RequestErrorKind::Status(status) => is_retryable_status_with_body(status, &err.message),
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

/// 判断错误是否是「上下文/输入超出模型窗口」类的拒绝。
///
/// 这类错误不是瞬态的（重发同样的包只会再次被拒），也不该触发 key 轮换或换模型
/// ——唯一有意义的补救是**收缩上下文后重试**。因此它独立于 [`is_transient_error`]
/// / [`should_try_model_fallback`] 分类，供 driver 的 reactive 压缩重试路径专用。
///
/// provider 对「上下文过长」的表述五花八门，且状态码不统一（OpenAI 用
/// 400 `context_length_exceeded`，部分兼容层用 413 Payload Too Large，也有网关
/// 直接回 400 + "maximum context length" / "too many tokens" 文案）。这里同时按
/// 状态码与 body 文案做保守匹配：413 视为上下文超限；400 需 body 命中已知文案。
/// 429/5xx 不在此列（它们由既有瞬态/轮换/退避路径处理）。
pub(crate) fn is_context_overflow_error(err: &RequestError) -> bool {
    let RequestErrorKind::Status(status) = err.kind else {
        return false;
    };
    if status == StatusCode::PAYLOAD_TOO_LARGE {
        return true;
    }
    if status.as_u16() != 400 {
        return false;
    }
    let lower = err.message.to_ascii_lowercase();
    lower.contains("context_length_exceeded")
        || lower.contains("context length")
        || lower.contains("context window")
        || lower.contains("maximum context")
        || lower.contains("too many tokens")
        || lower.contains("reduce the length")
        || lower.contains("input is too long")
        || lower.contains("prompt is too long")
        || lower.contains("string too long")
}

/// Returns `true` if the error indicates an auth or quota issue with the API key
/// (401 Unauthorized, 403 Forbidden, 429 Too Many Requests),
/// which should trigger key rotation when alternative keys are available.
///
/// 扩展：对于其他 HTTP 错误（如 400），若响应体中包含账号级错误信号
/// （如 OpenCode/Xiaomi 返回的 `code:"441"` / `type:"risk_control"`），
/// 也应触发 key 轮换，尝试使用下一个可用的 API key。
pub(crate) fn should_rotate_key(err: &RequestError) -> bool {
    match err.kind {
        RequestErrorKind::Status(status) => {
            if matches!(status.as_u16(), 401 | 403 | 429) {
                return true;
            }
            // 非标准 HTTP 状态码也检查 body 中的账号级错误信号
            is_account_error_body(&err.message)
        }
        RequestErrorKind::Network => false,
    }
}

/// 检查错误消息体中是否包含账号级错误信号（如 code 441、type risk_control 等）。
fn is_account_error_body(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    lower.contains("risk_control") || lower.contains("\"code\":\"441\"")
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
        || lower.contains("risk_control")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_overflow_matches_413_regardless_of_body() {
        let err = RequestError::status(StatusCode::PAYLOAD_TOO_LARGE, String::new());
        assert!(is_context_overflow_error(&err));
    }

    #[test]
    fn context_overflow_matches_400_with_known_phrasing() {
        for body in [
            "{\"error\":{\"code\":\"context_length_exceeded\"}}",
            "This model's maximum context length is 262144 tokens",
            "please reduce the length of the messages",
            "input is too long for requested model",
        ] {
            let err = RequestError::status(StatusCode::BAD_REQUEST, body.to_string());
            assert!(
                is_context_overflow_error(&err),
                "expected overflow classification for body: {body}"
            );
        }
    }

    #[test]
    fn context_overflow_ignores_unrelated_400_and_transient_errors() {
        let unrelated_400 = RequestError::status(
            StatusCode::BAD_REQUEST,
            "invalid tool arguments".to_string(),
        );
        assert!(!is_context_overflow_error(&unrelated_400));

        let rate_limited = RequestError::status(StatusCode::TOO_MANY_REQUESTS, String::new());
        assert!(!is_context_overflow_error(&rate_limited));

        let server_error =
            RequestError::status(StatusCode::INTERNAL_SERVER_ERROR, "upstream".to_string());
        assert!(!is_context_overflow_error(&server_error));

        let network = RequestError::cancelled("canceled");
        assert!(!is_context_overflow_error(&network));
    }
}
