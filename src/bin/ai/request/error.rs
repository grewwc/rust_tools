//! Request error types, retry policy, and request configuration helpers.
//!
//! Contains:
//! - `RequestError` / `RequestErrorKind`: unified request error types
//! - `RequestRetryPolicy`: retry count / timeout / hedging policy
//! - Retry helpers: `sleep_with_cancel` / `retry_delay`
//! - Model error classification: `is_transient_error` / `should_temporarily_disable_model` / `should_try_model_fallback`
//! - Request configuration helpers: `endpoint_for_request_model` / `api_key_for_request_model` / `apply_request_auth`

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
    /// Server-suggested retry wait duration for 429 responses (already clamped
    /// in `parse_retry_after`). Read by upper-layer key rotation / backoff
    /// logic; `None` for other errors.
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

    /// Whether this is a 429 (quota / rate-limit) error.
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
pub(crate) const REQUEST_MAX_ATTEMPTS_429: usize = 32; // retry 429 errors up to 32 times
pub(crate) const REQUEST_RETRY_BASE_MS: u64 = 500;
pub(crate) const REQUEST_RETRY_MAX_MS: u64 = 16000;
/// Backoff starting value for 429 (quota exceeded). When every key returns
/// 429, wait 4 seconds before the first retry instead of the generic 0.5s
/// backoff — quota throttling usually requires waiting for the server-side
/// window to refresh, and starting too short just burns request attempts.
pub(crate) const REQUEST_RETRY_429_BASE_MS: u64 = 4_000;
/// Upper bound for a single 429 (quota exceeded) backoff wait. Servers may
/// return an enormous value in `Retry-After` (seconds until the next quota
/// window, possibly tens of thousands); sleeping as-is would stall the process
/// for a long time. Clamp to this bound and yield quickly via key rotation.
pub(crate) const REQUEST_RETRY_429_MAX_MS: u64 = 10_000;
/// Timeout for a streaming request to receive the response headers (first byte).
///
/// The main `app.client` only keeps `connect_timeout` (no overall
/// `.timeout()`, which would kill long streaming body reads). But
/// `connect_timeout` only covers the TCP/TLS handshake, not the case where the
/// connection is established and the server is slow to return response headers
/// — then `.send().await` blocks forever with 0 CPU usage, appearing as a hung
/// agent. So streaming `send()` gets its own header-wait timeout as a backstop.
pub(crate) const STREAM_RESPONSE_HEADER_TIMEOUT_SECS: u64 = 90;
/// Sub-agent auto model selection has a fallback backstop and should yield
/// quickly when the preferred model is slow to respond. Explicitly specified
/// models do not use the AUTO_MODEL_FALLBACK scope and keep normal retry policy.
pub(crate) const AUTO_SUBAGENT_RESPONSE_HEADER_TIMEOUT_SECS: u64 = 30;
pub(crate) const AUTO_SUBAGENT_REQUEST_MAX_ATTEMPTS: usize = 1;
pub(crate) const DEFAULT_AUTO_THINKING_THRESHOLD: f64 = 0.7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RequestRetryPolicy {
    pub(crate) max_attempts: usize,
    pub(crate) max_attempts_429: usize,
    pub(crate) header_timeout_secs: u64,
    /// Maximum number of hedged backup sends (including primary). Beyond this,
    /// the last send no longer sets an internal timeout and relies on the outer
    /// `header_timeout_secs` backstop.
    pub(crate) hedged_max_sends: usize,
}

impl RequestRetryPolicy {
    /// Hedged backup trigger threshold: if the primary request receives no
    /// response headers within this window, an identical backup request is
    /// issued concurrently; the two race and the loser is dropped.
    /// Set to header_timeout / 9, clamped to [3, 15] seconds.
    ///
    /// This way, occasional server-side long tails (connection established but
    /// response headers slow to arrive) do not wait the full 90s before
    /// retrying — a new request is issued automatically at ~10s, significantly
    /// reducing tail latency.
    pub(crate) fn hedged_backup_after_secs(&self) -> u64 {
        (self.header_timeout_secs / 9).clamp(3, 15)
    }

    /// Total number of hedged backup sends for the primary request.
    /// Server long tails occasionally hit bad instances consecutively; one more
    /// backup further reduces tail latency.
    /// Note: with true concurrent hedging, a long-tail scenario can have up to
    /// `hedged_max_sends` concurrent connections.
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
            // The auto-selection model fallback is already a backup request.
            // Sending a hedged backup 3 seconds later would duplicate the same
            // sub-agent inference, causing repeated scheduling and extra cost.
            hedged_max_sends: 1,
        }
    } else {
        RequestRetryPolicy {
            max_attempts: REQUEST_MAX_ATTEMPTS,
            max_attempts_429: REQUEST_MAX_ATTEMPTS_429,
            header_timeout_secs: STREAM_RESPONSE_HEADER_TIMEOUT_SECS,
            hedged_max_sends: 3, // primary request: 1 primary + 2 backup
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

/// Decides whether an HTTP status code + response body is worth retrying.
/// On top of `should_retry_status`, additionally covers 400 + body containing
/// "upstream" — relay/compat layers wrap upstream transient failures (upstream
/// RPC errors, internal exceptions) into 400s; these are actually transient
/// and recoverable by retry. Keeps the "upstream" detection consistent with
/// `is_retryable_stream_error`.
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

/// 429-specific backoff: base 4s, exponentially increasing up to
/// `REQUEST_RETRY_MAX_MS` (16s). Shares the upper bound with the generic
/// `retry_delay` but starts higher, fitting the quota-window refresh scenario.
pub(crate) fn retry_delay_429(attempt: usize) -> Duration {
    let shift = attempt.saturating_sub(1).min(4) as u32;
    let backoff = REQUEST_RETRY_429_BASE_MS.saturating_mul(1u64 << shift);
    Duration::from_millis(backoff.min(REQUEST_RETRY_MAX_MS))
}

/// Parses the HTTP `Retry-After` response header and returns the suggested
/// wait duration. Supports the seconds format (`120`); HTTP-date format is not
/// supported yet. The return value is clamped to `REQUEST_RETRY_429_MAX_MS` so
/// an oversized server value cannot put the process to sleep indefinitely.
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
    // A stale interrupt signal from the previous turn (without an explicit
    // cancel/shutdown in progress) would short-circuit this network retry as
    // canceled at attempt 1.
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

/// Decides whether an error is a "context/input exceeds the model window" rejection.
///
/// Such errors are not transient (resending the same payload only gets
/// rejected again) and should not trigger key rotation or model switching —
/// the only meaningful remedy is **shrinking the context and retrying**. They
/// are therefore classified separately from [`is_transient_error`] /
/// [`should_try_model_fallback`] and reserved for the driver's reactive
/// compaction retry path.
///
/// Providers phrase "context too long" in all kinds of ways with inconsistent
/// status codes (OpenAI uses 400 `context_length_exceeded`, some compat layers
/// use 413 Payload Too Large, and some gateways return 400 + "maximum context
/// length" / "too many tokens" text). Match conservatively on both the status
/// code and body text: 413 counts as context overflow; 400 requires the body
/// to hit known phrasing. 429/5xx are excluded here (handled by the existing
/// transient/rotation/backoff paths).
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
/// Extension: for other HTTP errors (e.g. 400), if the response body contains
/// account-level error signals (such as the `code:"441"` /
/// `type:"risk_control"` returned by OpenCode/Xiaomi), key rotation should
/// also be triggered to try the next available API key.
pub(crate) fn should_rotate_key(err: &RequestError) -> bool {
    match err.kind {
        RequestErrorKind::Status(status) => {
            if matches!(status.as_u16(), 401 | 403 | 429) {
                return true;
            }
            // Non-standard HTTP status codes also check the body for account-level error signals
            is_account_error_body(&err.message)
        }
        RequestErrorKind::Network => false,
    }
}

/// Checks whether the error message body contains account-level error signals (e.g. code 441, type risk_control).
fn is_account_error_body(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    lower.contains("risk_control") || lower.contains("\"code\":\"441\"")
}

/// Decides whether an error occurring mid-stream is worth retrying.
///
/// Stream errors (`provider stream error: ...`) happen after the response
/// headers have been returned, during body transfer, and are usually transient
/// server-side issues (backend cancellation, upstream RPC failure, internal
/// errors, etc.). Retrying the whole request + stream once for these errors
/// significantly reduces the user-perceived failure rate.
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
