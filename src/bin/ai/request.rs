use std::fmt;
use std::fs;
use std::time::{Duration, Instant};

use base64::Engine as _;
use reqwest::{Response, StatusCode};
use rust_tools::commonw;
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use super::{
    files,
    history::{
        Message, ROLE_SYSTEM, is_internal_note_role, is_system_like_role, messages_to_markdown,
    },
    models,
    provider::{ApiProvider, ReasoningEffort},
    skills::SkillManifest,
    types::App,
};
use crate::ai::config_schema::AiConfig;
use crate::ai::driver::intent_recognition;
use crate::ai::theme::{ACCENT_MUTED, ACCENT_PRIMARY, ACCENT_SUCCESS, ACCENT_WARN, RESET};
use crate::commonw::configw;

#[derive(Debug, Serialize)]
struct RequestBody<'a> {
    model: String,
    messages: &'a [Message],
    stream: bool,
    /// 思考开关的线缆字段，由 `thinking` 方言模块决定具体 key 与形状
    /// （`enable_thinking: bool` / `thinking: {"type":...}` / 或空）。
    /// 核心层只持有 provider 无关的 Map，wire 编码完全归属方言。
    #[serde(flatten)]
    thinking: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_search: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    /// OpenAI / OpenRouter / OpenCode 兼容协议的推理强度顶层字段。
    /// DashScope compatible provider 使用下方的嵌套 `reasoning.effort`。
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<Value>,
    /// 流式请求显式索取 usage 统计：`{ "include_usage": true }`。
    /// 部分 provider（已知 DashScope compatible-mode）流式下默认不返回 `usage`，
    /// 必须显式声明，否则 token 用量无法统计、`/usage` 会漏计。非流式时为 None。
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct StreamChunk {
    #[serde(default, deserialize_with = "vec_or_default")]
    pub(super) choices: Vec<StreamChoice>,
    /// OpenAI-compatible usage block. Only present on the final chunk
    /// (and only if `stream_options: { include_usage: true }` was requested,
    /// though many providers include it unconditionally).
    #[serde(default)]
    pub(super) usage: Option<StreamUsage>,
    /// Some providers echo the model name on every chunk.
    #[serde(default, deserialize_with = "string_or_default")]
    pub(super) model: String,
}

/// OpenAI-compatible `usage` object. We intentionally keep it permissive
/// (all optional / default=0) so that varied providers do not break parsing.
///
/// Field-name compatibility: providers diverge on naming. Besides the canonical
/// OpenAI `prompt_tokens` / `completion_tokens`, we accept the Anthropic-style
/// `input_tokens` / `output_tokens` (and a few common spellings) via serde
/// `alias`, so that non-OpenAI-shaped responses are not silently counted as 0.
/// After deserialization, always call [`StreamUsage::normalized`] to fill in
/// components that the provider omitted but can be derived from `total_tokens`.
#[derive(Debug, Default, Clone, Deserialize)]
pub(crate) struct StreamUsage {
    #[serde(default, alias = "input_tokens", alias = "prompt_token_count")]
    pub(crate) prompt_tokens: u64,
    #[serde(default, alias = "output_tokens", alias = "completion_token_count")]
    pub(crate) completion_tokens: u64,
    #[serde(default, alias = "total_token_count")]
    pub(crate) total_tokens: u64,
    /// OpenAI newer format: prompt_tokens_details.cached_tokens
    #[serde(default, alias = "input_tokens_details")]
    pub(crate) prompt_tokens_details: Option<StreamPromptTokensDetails>,
    /// OpenAI reasoning models / qwen thinking mode report the reasoning slice
    /// here. Most providers already fold this into `completion_tokens`, so we
    /// only use it as a floor (see [`StreamUsage::normalized`]) to avoid
    /// double-counting while still recovering output tokens when a provider
    /// reports reasoning separately and leaves `completion_tokens` at 0.
    #[serde(default, alias = "output_tokens_details")]
    pub(crate) completion_tokens_details: Option<StreamCompletionTokensDetails>,
}

impl StreamUsage {
    /// Backfill omitted token components from whatever the provider did report,
    /// so downstream accounting does not under-count.
    ///
    /// Rules (all conservative, never inflate beyond reported totals):
    /// - If exactly one of prompt/completion is missing but `total_tokens`
    ///   covers the other, derive the missing component as the remainder.
    /// - If `completion_tokens` is 0 but a separate `reasoning_tokens` slice is
    ///   present, treat reasoning tokens as the output floor.
    /// - Keep `total_tokens` consistent (>= prompt + completion) for display.
    pub(crate) fn normalized(mut self) -> Self {
        // Reasoning-only providers: recover output tokens from the details slice.
        if self.completion_tokens == 0 {
            if let Some(reasoning) = self
                .completion_tokens_details
                .as_ref()
                .map(|d| d.reasoning_tokens)
                .filter(|&r| r > 0)
            {
                self.completion_tokens = reasoning;
            }
        }

        // Derive a missing component from the total when the total is larger
        // than the single component we do have.
        if self.total_tokens > 0 {
            if self.completion_tokens == 0 && self.prompt_tokens > 0 {
                self.completion_tokens = self.total_tokens.saturating_sub(self.prompt_tokens);
            } else if self.prompt_tokens == 0 && self.completion_tokens > 0 {
                self.prompt_tokens = self.total_tokens.saturating_sub(self.completion_tokens);
            }
        }

        // Keep total at least the sum of the parts for honest display.
        let sum = self.prompt_tokens.saturating_add(self.completion_tokens);
        if self.total_tokens < sum {
            self.total_tokens = sum;
        }
        self
    }
}

#[derive(Debug, Default, Clone, Deserialize)]
pub(crate) struct StreamPromptTokensDetails {
    #[serde(default)]
    pub(crate) cached_tokens: u64,
}

#[derive(Debug, Default, Clone, Deserialize)]
pub(crate) struct StreamCompletionTokensDetails {
    #[serde(default, alias = "thinking_tokens")]
    pub(crate) reasoning_tokens: u64,
}

impl StreamChunk {
    pub(super) fn merge_reasoning(&mut self) {
        for choice in &mut self.choices {
            if choice.delta.reasoning_content.is_empty()
                && !choice.message.reasoning_content.is_empty()
            {
                choice.delta.reasoning_content = choice.message.reasoning_content.clone();
            }
            if choice.delta.content.is_empty() && !choice.message.content.is_empty() {
                choice.delta.content = std::mem::take(&mut choice.message.content);
            }
            if choice.delta.tool_calls.is_empty() && !choice.message.tool_calls.is_empty() {
                choice.delta.tool_calls = std::mem::take(&mut choice.message.tool_calls);
            }
            if choice.delta.reasoning_content.is_empty() && !choice.reasoning_content.is_empty() {
                choice.delta.reasoning_content = choice.reasoning_content.clone();
            }
            if choice.delta.reasoning_details.is_empty() && !choice.reasoning_details.is_empty() {
                choice.delta.reasoning_details = choice.reasoning_details.clone();
            }
            choice.delta.reasoning_content = merge_reasoning_fragments(
                &choice.delta.reasoning_details,
                &choice.delta.reasoning_content,
            );
            choice.delta.reasoning_details.clear();
        }
    }
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct StreamChoice {
    #[serde(default)]
    pub(super) delta: StreamDelta,
    /// 少数 OpenAI-compatible 网关会把快照块放在 `message` 而不是 `delta`。
    /// 统一折叠进 delta，后续流处理层不需要认识这种 wire 差异。
    #[serde(default)]
    pub(super) message: StreamDelta,
    #[serde(
        default,
        alias = "reasoning",
        alias = "reasoning_text",
        deserialize_with = "displayable_string_or_default"
    )]
    pub(super) reasoning_content: String,
    #[serde(default, deserialize_with = "reasoning_details_string_or_default")]
    pub(super) reasoning_details: String,
    #[serde(default)]
    pub(super) finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct StreamDelta {
    #[serde(default, deserialize_with = "displayable_string_or_default")]
    pub(super) content: String,
    #[serde(
        default,
        alias = "reasoning",
        alias = "reasoning_text",
        deserialize_with = "displayable_string_or_default"
    )]
    pub(super) reasoning_content: String,
    #[serde(default, deserialize_with = "reasoning_details_string_or_default")]
    pub(super) reasoning_details: String,
    #[serde(default, deserialize_with = "vec_or_default")]
    pub(super) tool_calls: Vec<StreamToolCall>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct StreamToolCall {
    #[serde(default)]
    pub(super) index: usize,
    #[serde(default, deserialize_with = "string_or_default")]
    pub(super) id: String,
    #[serde(rename = "type", default, deserialize_with = "string_or_default")]
    pub(super) tool_type: String,
    #[serde(default)]
    pub(super) function: StreamFunctionCall,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct StreamFunctionCall {
    #[serde(default, deserialize_with = "string_or_default")]
    pub(super) name: String,
    #[serde(default, deserialize_with = "string_or_default")]
    pub(super) arguments: String,
}

fn vec_or_default<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    Option::<Vec<T>>::deserialize(deserializer).map(|opt| opt.unwrap_or_default())
}

fn string_or_default<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value
        .as_ref()
        .map(json_value_to_string_lossy)
        .unwrap_or_default())
}

fn displayable_string_or_default<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value
        .as_ref()
        .map(extract_displayable_text)
        .unwrap_or_default())
}

fn reasoning_details_string_or_default<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value
        .as_ref()
        .map(extract_reasoning_details_text)
        .unwrap_or_default())
}

fn extract_displayable_text(value: &Value) -> String {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .map(extract_displayable_text)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(""),
        Value::Object(map) => extract_text_from_object(map, &["text", "content", "delta"]),
    }
}

fn extract_reasoning_details_text(value: &Value) -> String {
    match value {
        Value::Array(items) => items
            .iter()
            .map(extract_reasoning_details_text)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(""),
        Value::Object(map) => extract_text_from_object(
            map,
            &[
                "text",
                "content",
                "delta",
                "summary_text",
                "reasoning_text",
                "reasoning",
                "summary",
            ],
        ),
        Value::String(s) => s.clone(),
        _ => String::new(),
    }
}

fn extract_text_from_object(
    map: &serde_json::Map<String, Value>,
    preferred_keys: &[&str],
) -> String {
    for key in preferred_keys {
        if let Some(inner) = map.get(*key) {
            let extracted = match *key {
                "reasoning" | "summary" => extract_reasoning_details_text(inner),
                _ => extract_displayable_text(inner),
            };
            if !extracted.is_empty() {
                return extracted;
            }
        }
    }
    String::new()
}

pub(crate) fn merge_reasoning_fragments(details: &str, content: &str) -> String {
    if content.is_empty() {
        return details.to_string();
    }
    if details.is_empty() {
        return content.to_string();
    }
    if content.contains(details) {
        return content.to_string();
    }
    if details.contains(content) {
        return details.to_string();
    }

    let overlap = longest_suffix_prefix_overlap(details, content);
    if overlap > 0 {
        return format!("{}{}", details, &content[overlap..]);
    }

    let content_stripped = content.trim_start();
    let stripped_overlap = longest_suffix_prefix_overlap(details, content_stripped);
    if stripped_overlap > 0 {
        return format!("{}{}", details, &content_stripped[stripped_overlap..]);
    }

    if looks_like_continuation(details, content_stripped) {
        return format!("{details}{content}");
    }

    content.to_string()
}

fn looks_like_continuation(prefix: &str, continuation: &str) -> bool {
    let first = match continuation.chars().next() {
        Some(c) => c,
        None => return false,
    };

    if is_continuation_punctuation(first) {
        return true;
    }

    if first == '\'' {
        let rest = &continuation[first.len_utf8()..];
        if rest.starts_with('s')
            || rest.starts_with('t')
            || rest.starts_with("re ")
            || rest.starts_with("ve ")
            || rest.starts_with("ll ")
            || rest.starts_with("d ")
            || rest.starts_with("m ")
        {
            return true;
        }
    }

    if continuation.starts_with("n't") {
        let prefix_stripped = prefix.trim_end();
        if prefix_stripped.ends_with("is")
            || prefix_stripped.ends_with("was")
            || prefix_stripped.ends_with("were")
            || prefix_stripped.ends_with("would")
            || prefix_stripped.ends_with("could")
            || prefix_stripped.ends_with("should")
            || prefix_stripped.ends_with("do")
            || prefix_stripped.ends_with("does")
            || prefix_stripped.ends_with("did")
            || prefix_stripped.ends_with("has")
            || prefix_stripped.ends_with("have")
            || prefix_stripped.ends_with("had")
            || prefix_stripped.ends_with("ca")
        {
            return true;
        }
    }

    false
}

fn is_continuation_punctuation(ch: char) -> bool {
    matches!(
        ch,
        ',' | '.'
            | ';'
            | ':'
            | '!'
            | '?'
            | ')'
            | ']'
            | '}'
            | '，'
            | '。'
            | '；'
            | '：'
            | '！'
            | '？'
            | '）'
            | '】'
            | '》'
            | '…'
    )
}

fn longest_suffix_prefix_overlap(left: &str, right: &str) -> usize {
    let mut candidates = right.char_indices().map(|(idx, _)| idx).collect::<Vec<_>>();
    candidates.push(right.len());
    candidates.reverse();

    for overlap in candidates {
        if overlap == 0 || overlap > left.len() {
            continue;
        }
        if left.ends_with(&right[..overlap]) {
            return overlap;
        }
    }
    0
}

fn json_value_to_string_lossy(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Array(items) => items
            .iter()
            .map(json_value_to_string_lossy)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(""),
        Value::Object(map) => {
            for key in [
                "text",
                "content",
                "value",
                "reasoning_content",
                "reasoning",
                "arguments",
            ] {
                if let Some(inner) = map.get(key) {
                    let extracted = json_value_to_string_lossy(inner);
                    if !extracted.is_empty() {
                        return extracted;
                    }
                }
            }
            serde_json::to_string(value).unwrap_or_default()
        }
    }
}

#[derive(Debug)]
pub(super) enum RequestErrorKind {
    Network,
    Status(StatusCode),
}

#[derive(Debug)]
pub(super) struct RequestError {
    kind: RequestErrorKind,
    message: String,
}

impl RequestError {
    fn network(err: reqwest::Error) -> Self {
        Self {
            kind: RequestErrorKind::Network,
            message: err.to_string(),
        }
    }

    fn cancelled(message: impl Into<String>) -> Self {
        Self {
            kind: RequestErrorKind::Network,
            message: message.into(),
        }
    }

    fn status(status: StatusCode, body: String) -> Self {
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

const REQUEST_MAX_ATTEMPTS: usize = 6;
const REQUEST_MAX_ATTEMPTS_429: usize = 16; // 429 错误重试 16 次
const REQUEST_RETRY_BASE_MS: u64 = 500;
const REQUEST_RETRY_MAX_MS: u64 = 4000;
/// 流式请求等待响应头（首字节）的超时。
///
/// 主 `app.client` 仅保留 `connect_timeout`（不设置整体 `.timeout()`，
/// 否则会误杀长时间的流式 body 读取）。但 `connect_timeout` 只覆盖 TCP/TLS
/// 握手，不覆盖“连接已建立、服务端迟迟不返回响应头”的场景——此时
/// `.send().await` 会永久阻塞、CPU 占用为 0，表现为 agent 卡死。
/// 因此对流式 `send()` 单独加一个响应头等待超时兜底。
const STREAM_RESPONSE_HEADER_TIMEOUT_SECS: u64 = 90;
/// 子 agent 自动选型有 fallback 兜底，首选模型迟迟不返回时应快速让位。
/// 显式指定模型不走 AUTO_MODEL_FALLBACK scope，仍保留常规重试策略。
const AUTO_SUBAGENT_RESPONSE_HEADER_TIMEOUT_SECS: u64 = 30;
const AUTO_SUBAGENT_REQUEST_MAX_ATTEMPTS: usize = 1;
const DEFAULT_AUTO_THINKING_THRESHOLD: f64 = 0.7;
const DEFAULT_CONTROL_MODEL: &str = "qwen3.5-flash";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RequestRetryPolicy {
    max_attempts: usize,
    max_attempts_429: usize,
    header_timeout_secs: u64,
}

fn request_retry_policy(auto_model_fallback: bool) -> RequestRetryPolicy {
    if auto_model_fallback {
        RequestRetryPolicy {
            max_attempts: AUTO_SUBAGENT_REQUEST_MAX_ATTEMPTS,
            max_attempts_429: AUTO_SUBAGENT_REQUEST_MAX_ATTEMPTS,
            header_timeout_secs: AUTO_SUBAGENT_RESPONSE_HEADER_TIMEOUT_SECS,
        }
    } else {
        RequestRetryPolicy {
            max_attempts: REQUEST_MAX_ATTEMPTS,
            max_attempts_429: REQUEST_MAX_ATTEMPTS_429,
            header_timeout_secs: STREAM_RESPONSE_HEADER_TIMEOUT_SECS,
        }
    }
}

fn request_retry_policy_for_current_context() -> RequestRetryPolicy {
    request_retry_policy(crate::ai::driver::runtime_ctx::auto_model_fallback_spec().is_some())
}

fn config_bool_is_true(value: Option<String>) -> bool {
    value
        .map(|v| v.trim().to_ascii_lowercase())
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
}

fn config_forces_thinking() -> bool {
    let cfg = configw::get_all_config();
    config_bool_is_true(cfg.get_opt(AiConfig::MODEL_THINKING))
}

fn endpoint_for_request_model(app: &App, model: &str) -> String {
    models::endpoint_for_model(model, &app.config.endpoint)
}

fn api_key_for_request_model(app: &App, model: &str) -> String {
    models::api_key_for_model(model, &app.config.api_key)
}

pub(in crate::ai) fn apply_request_auth(
    builder: reqwest::RequestBuilder,
    endpoint: &str,
    api_key: &str,
) -> reqwest::RequestBuilder {
    if api_key.trim().is_empty() && models::endpoint_supports_anonymous_auth(endpoint) {
        return builder;
    }
    builder.bearer_auth(api_key)
}

fn should_retry_status(status: StatusCode) -> bool {
    status.as_u16() == 429 || status.is_server_error()
}

fn is_retryable_reqwest_error(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect() || err.is_request()
}

fn retry_delay(attempt: usize) -> Duration {
    let shift = attempt.saturating_sub(1).min(4) as u32;
    let backoff = REQUEST_RETRY_BASE_MS.saturating_mul(1u64 << shift);
    Duration::from_millis(backoff.min(REQUEST_RETRY_MAX_MS))
}

fn should_abort_retry_wait(app: &App) -> bool {
    app.shutdown.load(std::sync::atomic::Ordering::Relaxed)
        || app.cancel_stream.load(std::sync::atomic::Ordering::Relaxed)
        || crate::ai::driver::signal::request_interrupt_ready()
}

async fn sleep_with_cancel(app: &App, delay: Duration) -> bool {
    if should_abort_retry_wait(app) {
        return true;
    }

    tokio::select! {
        _ = tokio::time::sleep(delay) => should_abort_retry_wait(app),
        _ = crate::ai::driver::signal::wait_for_interrupt_sources(None, None) => true,
    }
}

pub(super) fn clear_stale_request_interrupt_before_request(app: &App) {
    // 若上一次 turn 的中断信号残留（但当前并未处于显式 cancel/shutdown），
    // 会导致本次网络重试在 attempt1 就被短路为 canceled。
    if !app.shutdown.load(std::sync::atomic::Ordering::Relaxed)
        && !app.cancel_stream.load(std::sync::atomic::Ordering::Relaxed)
        && crate::ai::driver::signal::request_interrupt_ready()
    {
        crate::ai::driver::signal::clear_request_interrupt();
    }
}

pub(super) fn control_model_for_aux_tasks(app: &App) -> String {
    app.config
        .intent_model
        .as_deref()
        .filter(|v| !v.trim().is_empty())
        .map(models::determine_model)
        .unwrap_or_else(|| match models::model_provider(&app.current_model) {
            ApiProvider::Alibaba | ApiProvider::Compatible => {
                models::determine_model(DEFAULT_CONTROL_MODEL)
            }
            _ => {
                let current_model = app.current_model.trim();
                if current_model.is_empty() {
                    "gpt-4o-mini".to_string()
                } else {
                    current_model.to_string()
                }
            }
        })
}

pub(super) fn is_transient_error(err: &RequestError) -> bool {
    match err.kind {
        RequestErrorKind::Network => true,
        RequestErrorKind::Status(status) => should_retry_status(status),
    }
}

pub(super) fn should_temporarily_disable_model(err: &RequestError) -> bool {
    match err.kind {
        RequestErrorKind::Network => false,
        RequestErrorKind::Status(status) => {
            matches!(status.as_u16(), 402 | 404 | 429) || status.is_server_error()
        }
    }
}

pub(super) fn should_temporarily_disable_auto_selected_model(err: &RequestError) -> bool {
    if should_temporarily_disable_model(err) {
        return true;
    }
    if !matches!(err.kind, RequestErrorKind::Network) {
        return false;
    }
    let message = err.message.to_ascii_lowercase();
    message.contains("timed out") || message.contains("timeout")
}

pub(super) fn should_try_model_fallback(err: &RequestError) -> bool {
    match err.kind {
        RequestErrorKind::Network => true,
        RequestErrorKind::Status(status) => {
            matches!(status.as_u16(), 401 | 402 | 403 | 404 | 429) || status.is_server_error()
        }
    }
}

/// Resolve whether to enable thinking mode for this request.
///
/// Decision order:
/// 1. Config `ai.model.thinking=true` forces thinking when the model supports it
/// 2. If model doesn't support thinking, return false
/// 3. If auto-thinking is disabled by config, return false
/// 4. Auto-detect based on question complexity
#[commonw::debug_measure_time("resolve_thinking")]
async fn resolve_thinking(app: &App, model: &str, messages: &[Message]) -> bool {
    let cfg = configw::get_all_config();
    let force_thinking = config_bool_is_true(cfg.get_opt(AiConfig::MODEL_THINKING));

    if force_thinking {
        return models::enable_thinking(model);
    }

    // Model must support thinking
    if !models::enable_thinking(model) {
        return false;
    }

    // Check config for auto-thinking override
    let auto_enabled = cfg
        .get_opt(AiConfig::MODEL_AUTO_THINKING_ENABLE)
        .map(|v| !v.trim().eq_ignore_ascii_case("false"))
        .unwrap_or(true); // Default: enabled

    if !auto_enabled {
        return false;
    }

    let raw_question = latest_user_message_text(messages).unwrap_or_default();
    // 注入的 `<system-reminder>...</system-reminder>` 上下文会被拼到当前
    // user message 最前面（见 prepare.rs / skill_runtime.rs）。它会把一句
    // "hi" 撑成上千字符的长文本，导致本地 thinking 短路（按问题长度判定）
    // 失效，进而落到耗时数秒的模型 gate。这里在判定前剥离这些 reminder 块，
    // 只用用户真正输入的内容做意图与 thinking 判定。
    let question = strip_system_reminders(&raw_question);
    let question = question.trim();
    if !question.is_empty() {
        let local_intent = intent_recognition::detect_intent_with_model_path(
            question,
            &app.config.intent_model_path,
        );
        if let Some(local_decision) = local_thinking_decision(question, &local_intent) {
            crate::ai::agent_hang_debug!(
                "post-fix",
                "G",
                "request::resolve_thinking:local_decision",
                "[DEBUG] resolve thinking decided locally",
                {
                    "core": format!("{:?}", local_intent.core),
                    "question_len": question.chars().count(),
                    "decision": local_decision,
                },
            );
            return local_decision;
        }
    }

    // Model-only decision path: if gate fails/uncertain, default to disabled.
    decide_thinking_via_model(app, model, messages)
        .await
        .unwrap_or(false)
}

fn latest_user_message_text(messages: &[Message]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .and_then(extract_message_text)
}

/// 剥离注入到 user message 中的 `<system-reminder>...</system-reminder>` 块。
///
/// prepare.rs / skill_runtime.rs 会把上下文提醒拼到当前 user message 最前面
/// （为保 prompt cache）。这些块体量很大，会污染意图/thinking 判定的输入，
/// 让一句 "hi" 看起来像是长文本。判定前去掉它们，只留用户真正输入的内容。
fn strip_system_reminders(text: &str) -> String {
    const OPEN: &str = "<system-reminder>";
    const CLOSE: &str = "</system-reminder>";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        out.push_str(&rest[..start]);
        let after_open = &rest[start + OPEN.len()..];
        match after_open.find(CLOSE) {
            Some(end) => rest = &after_open[end + CLOSE.len()..],
            // 没有闭合标签：丢弃剩余内容（视为未闭合的 reminder）。
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

fn local_thinking_decision(
    question: &str,
    intent: &intent_recognition::UserIntent,
) -> Option<bool> {
    let question = question.trim();
    if question.is_empty() {
        return Some(false);
    }

    let question_len = question.chars().count();
    let nonempty_lines: Vec<&str> = question
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    let line_count = nonempty_lines.len();
    let has_code_like_content = question.contains("```")
        || question.contains("::")
        || question.split_whitespace().any(|token| {
            token.contains('/')
                || token.contains('\\')
                || token.ends_with(".rs")
                || token.ends_with(".ts")
                || token.ends_with(".tsx")
                || token.ends_with(".js")
                || token.ends_with(".jsx")
                || token.ends_with(".py")
                || token.ends_with(".go")
                || token.ends_with(".java")
        });
    // 结构化诊断痕迹：后续行出现 `label: details` / 堆栈路径样式，
    // 不依赖具体错误关键词。
    let has_diagnostic_shape = line_count >= 2
        && nonempty_lines.iter().skip(1).any(|line| {
            line.contains(": ")
                || line.contains(" at ")
                || line.contains("->")
                || line.contains("::")
                || line.contains('/')
                || line.contains('\\')
        });
    let has_multistep_shape =
        line_count >= 3 || question.contains("\n- ") || question.contains("\n1.");
    let looks_like_complex_solution_request =
        matches!(intent.core, intent_recognition::CoreIntent::SeekSolution)
            && (question_len >= 48
                || has_code_like_content
                || has_multistep_shape
                || has_diagnostic_shape);
    let looks_like_complex_action_request =
        matches!(intent.core, intent_recognition::CoreIntent::RequestAction)
            && (question_len >= 96 || has_code_like_content || has_multistep_shape);

    if matches!(
        intent.core,
        intent_recognition::CoreIntent::Casual | intent_recognition::CoreIntent::QueryConcept
    ) && question_len <= 120
        && !has_code_like_content
    {
        return Some(false);
    }

    if looks_like_complex_solution_request
        || looks_like_complex_action_request
        || (question_len >= 220
            && matches!(
                intent.core,
                intent_recognition::CoreIntent::RequestAction
                    | intent_recognition::CoreIntent::SeekSolution
            ))
    {
        return Some(true);
    }

    None
}

/// Ask the model whether this request needs thinking mode.
///
/// Returns `Some(decision)` only when response parses successfully and confidence
/// passes configured threshold; otherwise returns `None` for local fallback.
#[crate::ai::agent_hang_span(
    "pre-fix",
    "G",
    "request::decide_thinking_via_model",
    "[DEBUG] thinking gate started",
    "[DEBUG] thinking gate finished",
    {
        "message_count": messages.len(),
    },
    {
        "decision": __agent_hang_result,
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
async fn decide_thinking_via_model(app: &App, _model: &str, messages: &[Message]) -> Option<bool> {
    let gate_start = Instant::now();
    let user_text = latest_user_message_text(messages).unwrap_or_default();
    let question = user_text.trim();
    if question.is_empty() {
        crate::ai::agent_hang_debug!(
            "pre-fix",
            "G",
            "request::decide_thinking_via_model:empty",
            "[DEBUG] thinking gate skipped empty question",
            {
                "elapsed_ms": gate_start.elapsed().as_secs_f64() * 1000.0,
            },
        );
        return None;
    }

    let clipped = if question.chars().count() > 1200 {
        question.chars().take(1200).collect::<String>()
    } else {
        question.to_string()
    };

    let gate_messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(
                "You are a request complexity gate. Decide whether this user request needs deliberate reasoning mode.\nOutput STRICT JSON only: {\"thinking\":true|false,\"confidence\":0.0}\nRules:\n- thinking=true for multi-step tasks, code changes, debugging, comparative analysis, or ambiguous complex intent.\n- thinking=false for greetings, simple factual asks, tiny rewrites, or short direct requests.\n- confidence is your certainty in [0,1]."
                    .to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(clipped),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let control_model = control_model_for_aux_tasks(app);
    let request_body = build_request_body(
        &control_model,
        &gate_messages,
        false,
        false,
        None,
        None,
        None,
        None,
    );

    let endpoint = endpoint_for_request_model(app, &control_model);
    let api_key = api_key_for_request_model(app, &control_model);
    // 辅助请求（thinking gate），15 秒超时兜底：主 client 无整体 timeout，
    // 仅 connect_timeout 不覆盖“连上但服务端不回响应头”的永久阻塞。
    let send_future = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send();
    let response = match tokio::time::timeout(Duration::from_secs(15), send_future).await {
        Ok(r) => r.ok()?,
        Err(_) => return None,
    };

    if !response.status().is_success() {
        crate::ai::agent_hang_debug!(
            "pre-fix",
            "G",
            "request::decide_thinking_via_model:http_non_success",
            "[DEBUG] thinking gate http non success",
            {
                "elapsed_ms": gate_start.elapsed().as_secs_f64() * 1000.0,
            },
        );
        return None;
    }

    let text = match tokio::time::timeout(Duration::from_secs(15), response.text()).await {
        Ok(r) => r.ok()?,
        Err(_) => return None,
    };
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = extract_router_content(&v)?;
    let (thinking, confidence) = parse_thinking_gate_output(&content)?;
    let cfg = configw::get_all_config();
    let threshold = cfg
        .get_opt(AiConfig::MODEL_AUTO_THINKING_THRESHOLD)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(DEFAULT_AUTO_THINKING_THRESHOLD);

    let result = if confidence >= threshold {
        Some(thinking)
    } else {
        None
    };
    result
}

fn parse_thinking_gate_output(s: &str) -> Option<(bool, f64)> {
    let s = strip_json_fence(s);
    let candidate = if let (Some(l), Some(r)) = (s.find('{'), s.rfind('}'))
        && r >= l
    {
        &s[l..=r]
    } else {
        s
    };

    let v: Value = serde_json::from_str(candidate).ok()?;
    let thinking = match v.get("thinking") {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => s.trim().eq_ignore_ascii_case("true"),
        _ => return None,
    };
    let confidence = v.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
    Some((thinking, confidence))
}

/// Extract text content from a message.
fn extract_message_text(msg: &Message) -> Option<String> {
    match &msg.content {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                if let Some(s) = part.get("text").and_then(|v| v.as_str()) {
                    out.push_str(s);
                }
            }
            if out.is_empty() { None } else { Some(out) }
        }
        _ => None,
    }
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
    if enable_thinking {
        ensure_reasoning_content_echo_for_thinking_model(model, &mut normalized_messages);
    }
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
    );
    let retry_policy = request_retry_policy_for_current_context();

    for attempt in 1..=retry_policy.max_attempts_429 {
        let endpoint = endpoint_for_request_model(app, model);
        let api_key = api_key_for_request_model(app, model);
        let send_future = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send();
        // 给“等待响应头”加超时兜底：connect_timeout 只覆盖握手，无法拦截
        // 服务端接受连接后迟迟不返回响应头导致的永久阻塞（CPU 0 卡死）。
        let response = match tokio::time::timeout(
            Duration::from_secs(retry_policy.header_timeout_secs),
            send_future,
        )
        .await
        {
            Ok(result) => result,
            Err(_) => {
                if attempt < retry_policy.max_attempts {
                    let delay = retry_delay(attempt);
                    eprintln!(
                        "[Warning] 等待响应头超时 ({}s) - sleep {} 秒后重试 (attempt {}/{})",
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
                let body = (response.text().await).unwrap_or_default();
                let err = RequestError::status(status, body);

                // 根据状态码确定最大重试次数
                let max_attempts_for_status = if status_code == 429 {
                    retry_policy.max_attempts_429
                } else {
                    retry_policy.max_attempts
                };

                if should_retry_status(status) && attempt < max_attempts_for_status {
                    // 打印 sleep 原因
                    let delay = retry_delay(attempt);

                    if status_code == 429 {
                        eprintln!(
                            "[Warning] 429 Too Many Requests - 配额超限，sleep {} 秒后重试 (attempt {}/{})",
                            delay.as_secs_f32(),
                            attempt,
                            max_attempts_for_status
                        );
                    } else {
                        eprintln!(
                            "[Warning] {} - sleep {} 秒后重试 (attempt {}/{})",
                            status,
                            delay.as_secs_f32(),
                            attempt,
                            max_attempts_for_status
                        );
                    }
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
                    // 打印 sleep 原因
                    let delay = retry_delay(attempt);
                    eprintln!(
                        "[Warning] 网络错误 - sleep {} 秒后重试 (attempt {}/{})",
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

    Err(RequestError {
        kind: RequestErrorKind::Network,
        message: "request failed".to_string(),
    })
}

fn strip_json_fence(s: &str) -> &str {
    let trimmed = s.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        let rest = rest.trim_start();
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        let rest = rest.trim_start_matches('\n').trim_start_matches('\r');
        if let Some(end) = rest.rfind("```") {
            return rest[..end].trim();
        }
    }
    trimmed
}

fn parse_router_output(s: &str) -> (Option<String>, f64) {
    let s = strip_json_fence(s);
    let candidate = if let (Some(l), Some(r)) = (s.find('{'), s.rfind('}'))
        && r >= l
    {
        &s[l..=r]
    } else {
        s
    };
    let v: Value = match serde_json::from_str(candidate) {
        Ok(v) => v,
        Err(_) => return (None, 0.0),
    };
    let name = v
        .get("skill")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let confidence = v.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
    if name.is_empty() || name == "none" || name == "null" {
        (None, confidence)
    } else {
        (Some(name), confidence)
    }
}

fn extract_router_content(v: &Value) -> Option<String> {
    let choices = v
        .get("choices")
        .or_else(|| v.get("output").and_then(|o| o.get("choices")))?;
    let msg = choices.get(0)?.get("message")?;
    let content = msg.get("content")?;
    match content {
        Value::String(s) => Some(s.to_string()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                if let Some(s) = part.get("text").and_then(|v| v.as_str()) {
                    out.push_str(s);
                }
            }
            Some(out)
        }
        _ => None,
    }
}

#[derive(Debug, Clone)]
struct SkillRouterDecision {
    skill: Option<String>,
    confidence: f64,
}

async fn select_skill_candidate_via_model(
    app: &App,
    question: &str,
    skills: &[SkillManifest],
) -> Option<SkillRouterDecision> {
    if question.trim().is_empty() || skills.is_empty() {
        return None;
    }

    let mut system_prompt = r#"You are a skill router for a code-focused assistant.
Your job is to decide whether the current request clearly needs one of the available skills.
Output schema: {"skill":"<exact skill name or empty>","confidence":0.0}
Rules:
- Route only when the request is explicitly about operating on source code, code artifacts, or a coding workflow that matches a listed skill.
- Abstain for general knowledge, documentation lookup, high-level discussion, non-code work, or ambiguous requests.
- Prefer abstaining over misrouting when the evidence is weak.
- Use only the exact skill names listed below.
- Return only valid JSON.
Skills:
"#.to_string();

    for s in skills {
        let desc = if s.description.trim().is_empty() {
            "(no description)".to_string()
        } else {
            s.description.trim().to_string()
        };
        system_prompt.push_str(&format!("- {}: {}\n", s.name, desc));
    }

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(system_prompt),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(question.to_string()),
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
    );

    let endpoint = endpoint_for_request_model(app, &control_model);
    let api_key = api_key_for_request_model(app, &control_model);
    // 辅助请求（skill 路由），15 秒超时兜底，理由同上。
    let send_future = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send();
    let response = match tokio::time::timeout(Duration::from_secs(15), send_future).await {
        Ok(r) => r.ok()?,
        Err(_) => return None,
    };
    if !response.status().is_success() {
        return None;
    }

    let text = match tokio::time::timeout(Duration::from_secs(15), response.text()).await {
        Ok(r) => r.ok()?,
        Err(_) => return None,
    };
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = extract_router_content(&v).unwrap_or_default();
    let (name, confidence) = parse_router_output(&content);
    Some(SkillRouterDecision {
        skill: name,
        confidence,
    })
}

#[crate::ai::agent_hang_span(
    "pre-fix",
    "R",
    "request::select_skill_via_model",
    "[DEBUG] model skill router started",
    "[DEBUG] model skill router finished",
    {
        "question_len": question.chars().count(),
        "skill_count": skills.len(),
    },
    {
        "selected": __agent_hang_result.as_deref(),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
pub(super) async fn select_skill_via_model(
    app: &mut App,
    _model: &str,
    question: &str,
    skills: &[SkillManifest],
) -> Option<String> {
    const SKILL_ROUTER_CHUNK_SIZE: usize = 32;
    let router_start = Instant::now();
    if question.trim().is_empty() {
        crate::ai::agent_hang_debug!(
            "pre-fix",
            "R",
            "request::select_skill_via_model:empty_question",
            "[DEBUG] model skill router skipped empty question",
            {
                "elapsed_ms": router_start.elapsed().as_secs_f64() * 1000.0,
            },
        );
        return None;
    }
    if skills.is_empty() {
        crate::ai::agent_hang_debug!(
            "pre-fix",
            "R",
            "request::select_skill_via_model:empty_skills",
            "[DEBUG] model skill router skipped empty skills",
            {
                "elapsed_ms": router_start.elapsed().as_secs_f64() * 1000.0,
            },
        );
        return None;
    }

    let cfg = configw::get_all_config();
    let threshold = cfg
        .get_opt("ai.skills.router_threshold")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.7);
    let decision = if skills.len() <= SKILL_ROUTER_CHUNK_SIZE {
        select_skill_candidate_via_model(app, question, skills).await
    } else {
        let mut chunk_best: Vec<(String, f64)> = Vec::new();
        for chunk in skills.chunks(SKILL_ROUTER_CHUNK_SIZE) {
            let Some(decision) = select_skill_candidate_via_model(app, question, chunk).await
            else {
                continue;
            };
            let Some(name) = decision.skill else {
                continue;
            };
            if let Some(existing) = chunk_best.iter_mut().find(|(n, _)| *n == name) {
                if decision.confidence > existing.1 {
                    existing.1 = decision.confidence;
                }
            } else {
                chunk_best.push((name, decision.confidence));
            }
        }
        chunk_best.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        if chunk_best.is_empty() {
            None
        } else if chunk_best.len() == 1 {
            Some(SkillRouterDecision {
                skill: Some(chunk_best[0].0.clone()),
                confidence: chunk_best[0].1,
            })
        } else {
            let finalists = chunk_best
                .iter()
                .filter_map(|(name, _)| skills.iter().find(|s| s.name == *name).cloned())
                .collect::<Vec<_>>();
            select_skill_candidate_via_model(app, question, &finalists)
                .await
                .or_else(|| {
                    Some(SkillRouterDecision {
                        skill: Some(chunk_best[0].0.clone()),
                        confidence: chunk_best[0].1,
                    })
                })
        }
    };
    let Some(decision) = decision else {
        return None;
    };
    if decision.confidence >= threshold {
        decision.skill
    } else {
        None
    }
}

fn agent_tools_for_request(app: &App, model: &str) -> (Option<Value>, Option<Value>) {
    if !models::tools_enabled(model) {
        return (None, None);
    }
    let Some(ctx) = app.agent_context.as_ref() else {
        return (None, None);
    };
    if ctx.tools.is_empty() {
        return (None, None);
    }
    let tools_value = serde_json::to_value(&ctx.tools).ok();
    let tool_choice = tools_value
        .as_ref()
        .map(|_| Value::String("auto".to_string()));
    (tools_value, tool_choice)
}

fn normalize_messages_for_request(messages: &[Message]) -> Vec<Message> {
    const MERGED_NOTES_MAX_CHARS: usize = 4_000;
    const MERGED_SINGLE_NOTE_MAX_CHARS: usize = 1_200;

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum InternalNoteKind {
        WorkingMemory,
        CodeDiscovery,
        CachedTools,
        Summary,
        SelfNote,
        Generic,
    }

    fn truncate_chars(s: &str, max_chars: usize) -> String {
        if s.chars().count() <= max_chars {
            return s.to_string();
        }
        let mut out = String::new();
        for ch in s.chars().take(max_chars.saturating_sub(1)) {
            out.push(ch);
        }
        out.push('…');
        out
    }

    fn truncate_note_text(text: &str, max_chars: usize) -> String {
        if text.chars().count() <= max_chars {
            return text.to_string();
        }

        let lines = text
            .lines()
            .map(str::trim_end)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>();
        if lines.is_empty() {
            return truncate_chars(text, max_chars);
        }

        // 结构化裁剪：优先保留头部条目 + 尾部少量最新条目，避免硬切半句。
        let mut selected = Vec::new();
        let mut used = 0usize;
        let head_budget = max_chars.saturating_mul(2).saturating_div(3);
        for line in lines.iter().take(24) {
            let line_chars = line.chars().count();
            let extra = if selected.is_empty() { 0 } else { 1 };
            if used + extra + line_chars > head_budget {
                break;
            }
            used += extra + line_chars;
            selected.push((*line).to_string());
        }

        let mut tail = Vec::new();
        for line in lines.iter().rev().take(6).rev() {
            if selected.iter().any(|existing| existing == line) {
                continue;
            }
            tail.push((*line).to_string());
        }

        let omitted = text.chars().count().saturating_sub(
            selected
                .iter()
                .map(|line| line.chars().count())
                .sum::<usize>(),
        );
        if !tail.is_empty() {
            selected.push(format!("... [truncated: {omitted} chars omitted]"));
            selected.extend(tail);
        }
        truncate_chars(&selected.join("\n"), max_chars)
    }

    fn detect_note_kind(text: &str) -> InternalNoteKind {
        let trimmed = text.trim();
        if trimmed.starts_with("Current code-inspection working memory:") {
            return InternalNoteKind::WorkingMemory;
        }
        if trimmed.starts_with("code_discovery:") {
            return InternalNoteKind::CodeDiscovery;
        }
        if trimmed.starts_with("Context note: reused cached tool results") {
            return InternalNoteKind::CachedTools;
        }
        if trimmed.starts_with("对话摘要（自动压缩")
            || trimmed.starts_with("历史摘要（自动压缩")
            || trimmed.starts_with("[mid-turn-summary]")
        {
            return InternalNoteKind::Summary;
        }
        if trimmed.starts_with("self_note:") {
            return InternalNoteKind::SelfNote;
        }
        InternalNoteKind::Generic
    }

    fn note_heading(kind: InternalNoteKind) -> &'static str {
        match kind {
            InternalNoteKind::WorkingMemory => "Working Memory",
            InternalNoteKind::CodeDiscovery => "Code Discoveries",
            InternalNoteKind::CachedTools => "Cached Tool Results",
            InternalNoteKind::Summary => "History Summary",
            InternalNoteKind::SelfNote => "Self Notes",
            InternalNoteKind::Generic => "Additional Notes",
        }
    }

    fn content_is_effectively_empty(content: &Value) -> bool {
        match content {
            Value::Null => true,
            Value::String(s) => s.trim().is_empty(),
            Value::Array(items) => items.is_empty(),
            Value::Object(map) => map.is_empty(),
            Value::Bool(_) | Value::Number(_) => false,
        }
    }

    fn sanitize_tool_call_for_request(
        tool_call: &crate::ai::types::ToolCall,
    ) -> Option<crate::ai::types::ToolCall> {
        let raw_args = tool_call.function.arguments.trim();
        let normalized_arguments = if raw_args.is_empty() {
            "{}".to_string()
        } else {
            serde_json::from_str::<Value>(raw_args).ok()?;
            raw_args.to_string()
        };

        let mut sanitized = tool_call.clone();
        sanitized.function.arguments = normalized_arguments;
        // 确保 tool_type 不为空（部分 provider 在 stream 中不返回 type）
        if sanitized.tool_type.is_empty() {
            sanitized.tool_type = "function".to_string();
        }
        Some(sanitized)
    }

    fn sanitize_tool_message_sequence(messages: Vec<Message>) -> Vec<Message> {
        fn build_unpaired_tool_evidence_note(
            reason: &str,
            tool_messages: &[Message],
        ) -> Option<Message> {
            if tool_messages.is_empty() {
                return None;
            }
            let mut lines = vec![
                "Context note: preserved unmatched tool outputs from prior rounds.".to_string(),
                format!("reason: {reason}"),
            ];
            for message in tool_messages.iter().take(8) {
                let tool_call_id = message
                    .tool_call_id
                    .as_deref()
                    .filter(|id| !id.trim().is_empty())
                    .unwrap_or("unknown");
                let text = message
                    .content
                    .as_str()
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .unwrap_or_default();
                if text.is_empty() {
                    continue;
                }
                let preview = if text.chars().count() > 240 {
                    let mut p = text.chars().take(239).collect::<String>();
                    p.push('…');
                    p
                } else {
                    text.to_string()
                };
                lines.push(format!("- tool_call_id={tool_call_id}: {preview}"));
            }
            if lines.len() <= 2 {
                return None;
            }
            Some(Message {
                role: "internal_note".to_string(),
                content: Value::String(lines.join("\n")),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            })
        }

        let mut out = Vec::with_capacity(messages.len());
        let mut idx = 0usize;

        while idx < messages.len() {
            let message = &messages[idx];
            if message.role == "tool" {
                idx += 1;
                continue;
            }

            let Some(tool_calls) = message
                .tool_calls
                .as_ref()
                .filter(|calls| !calls.is_empty())
            else {
                out.push(message.clone());
                idx += 1;
                continue;
            };

            let sanitized_tool_calls = tool_calls
                .iter()
                .filter_map(sanitize_tool_call_for_request)
                .collect::<Vec<_>>();
            let mut scan = idx + 1;

            if sanitized_tool_calls.is_empty() {
                let mut raw_tool_messages = Vec::new();
                while scan < messages.len() && messages[scan].role == "tool" {
                    raw_tool_messages.push(messages[scan].clone());
                    scan += 1;
                }
                let mut assistant_only = message.clone();
                assistant_only.tool_calls = None;
                if !content_is_effectively_empty(&assistant_only.content) {
                    out.push(assistant_only);
                }
                if let Some(note) = build_unpaired_tool_evidence_note(
                    "tool_calls dropped because arguments failed request sanitization",
                    &raw_tool_messages,
                ) {
                    out.push(note);
                }
                idx = scan.max(idx + 1);
                continue;
            }

            let expected_ids = sanitized_tool_calls
                .iter()
                .map(|tool_call| tool_call.id.as_str())
                .collect::<Vec<_>>();
            let mut matched_ids = Vec::new();
            let mut matched_tool_messages = Vec::new();
            let mut unmatched_tool_messages = Vec::new();

            while scan < messages.len() && messages[scan].role == "tool" {
                let tool_message = &messages[scan];
                if let Some(tool_call_id) = tool_message.tool_call_id.as_deref()
                    && expected_ids
                        .iter()
                        .any(|expected| *expected == tool_call_id)
                    && !matched_ids.iter().any(|seen| seen == tool_call_id)
                {
                    matched_ids.push(tool_call_id.to_string());
                    matched_tool_messages.push(tool_message.clone());
                } else {
                    unmatched_tool_messages.push(tool_message.clone());
                }
                scan += 1;
            }

            if matched_ids.is_empty() {
                let mut assistant_only = message.clone();
                assistant_only.tool_calls = None;
                if !content_is_effectively_empty(&assistant_only.content) {
                    out.push(assistant_only);
                }
                if let Some(note) = build_unpaired_tool_evidence_note(
                    "tool_call ids could not be matched with sanitized assistant tool_calls",
                    &unmatched_tool_messages,
                ) {
                    out.push(note);
                }
                idx = scan.max(idx + 1);
                continue;
            }

            let mut assistant_with_matched_calls = message.clone();
            assistant_with_matched_calls.tool_calls = Some(
                sanitized_tool_calls
                    .iter()
                    .filter(|tool_call| matched_ids.iter().any(|id| id == &tool_call.id))
                    .cloned()
                    .collect(),
            );
            out.push(assistant_with_matched_calls);
            out.extend(matched_tool_messages);
            if let Some(note) = build_unpaired_tool_evidence_note(
                "some tool outputs were unmatched and preserved as context note",
                &unmatched_tool_messages,
            ) {
                out.push(note);
            }
            idx = scan;
        }

        out
    }

    let first_system_idx = messages
        .iter()
        .position(|m| m.role == ROLE_SYSTEM || is_internal_note_role(&m.role));
    let Some(first_system_idx) = first_system_idx else {
        return sanitize_tool_message_sequence(messages.to_vec());
    };

    // Only merge system-like notes that sit BEFORE the first conversational
    // (user/assistant/tool) message — those are produced by history
    // compression at the very top and stay stable across turns. Notes that
    // arrive later (working memory / code discovery / self_note / cached
    // tool results, etc.) are kept in their original positions with role
    // rewritten to "system", so growing tail notes only invalidate the
    // suffix of the provider's prompt cache instead of the whole request.
    let first_body_idx = messages
        .iter()
        .position(|m| !is_system_like_role(&m.role))
        .unwrap_or(messages.len());

    let mut merged_notes: Vec<(usize, InternalNoteKind, String)> = Vec::new();
    for (idx, message) in messages.iter().enumerate().take(first_body_idx) {
        if idx == first_system_idx {
            continue;
        }
        let text = message
            .content
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or_default();
        if !text.is_empty() {
            merged_notes.push((
                idx,
                detect_note_kind(text),
                truncate_note_text(text, MERGED_SINGLE_NOTE_MAX_CHARS),
            ));
        }
    }
    merged_notes.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    let mut merged_first = messages[first_system_idx].clone();
    merged_first.role = ROLE_SYSTEM.to_string();
    if let Some(base) = merged_first.content.as_str() {
        let mut content = base.to_string();
        if !merged_notes.is_empty() {
            content.push_str("\n\n[Merged system notes from history/runtime]\n");
            let mut grouped = Vec::new();
            let mut current_kind: Option<InternalNoteKind> = None;
            for (_, kind, text) in merged_notes {
                if current_kind != Some(kind) {
                    grouped.push(format!("## {}", note_heading(kind)));
                    current_kind = Some(kind);
                }
                grouped.push(text);
            }
            let merged_blob = truncate_chars(&grouped.join("\n\n"), MERGED_NOTES_MAX_CHARS);
            content.push_str(&merged_blob);
        }
        merged_first.content = Value::String(content);
    }

    let mut out = Vec::with_capacity(messages.len());
    out.push(merged_first);
    for (idx, message) in messages.iter().enumerate() {
        if idx == first_system_idx {
            continue;
        }
        if idx < first_body_idx {
            // Already folded into merged_first above.
            continue;
        }
        if is_internal_note_role(&message.role) {
            // Keep the note in-place but normalize the role so the API
            // accepts it. Stable position preserves prompt-cache prefix:
            // older notes don't move when newer ones get appended.
            let mut promoted = message.clone();
            promoted.role = ROLE_SYSTEM.to_string();
            // Cap mid-stream notes to a reasonable budget to avoid bloating
            // the request with stale long notes (working memory / code
            // discoveries / self_note / cached-tool notes accumulated over
            // many tool rounds).
            if let Value::String(text) = &promoted.content {
                if text.chars().count() > MERGED_SINGLE_NOTE_MAX_CHARS {
                    promoted.content =
                        Value::String(truncate_note_text(text, MERGED_SINGLE_NOTE_MAX_CHARS));
                }
            }
            out.push(promoted);
            continue;
        }
        out.push(message.clone());
    }
    sanitize_tool_message_sequence(out)
}

fn normalize_message_content_for_text_only_model(content: &Value) -> Value {
    const IMAGE_PLACEHOLDER: &str = "[image omitted]";

    match content {
        Value::Array(items) => {
            let mut segments = Vec::new();
            for item in items {
                if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                    if !text.trim().is_empty() {
                        segments.push(text.to_string());
                    }
                    continue;
                }

                if item.get("image_url").is_some()
                    || item.get("type").and_then(|v| v.as_str()) == Some("image_url")
                {
                    segments.push(IMAGE_PLACEHOLDER.to_string());
                    continue;
                }

                let fallback = extract_displayable_text(item);
                if !fallback.trim().is_empty() {
                    segments.push(fallback);
                }
            }

            if segments.is_empty() {
                Value::String(IMAGE_PLACEHOLDER.to_string())
            } else {
                Value::String(segments.join("\n"))
            }
        }
        _ => content.clone(),
    }
}

fn normalize_messages_for_model(model: &str, messages: &[Message]) -> Vec<Message> {
    let mut normalized = normalize_messages_for_request(messages);
    if models::supports_image_input(model) {
        return normalized;
    }

    for message in &mut normalized {
        message.content = normalize_message_content_for_text_only_model(&message.content);
    }
    normalized
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
    let effort_label = match resolve_reasoning_effort(app, model) {
        Some(e) => e.as_str(),
        None => "auto",
    };
    // 使用 println! 避免手动 flush 的权限问题
    println!(
        "{ACCENT_MUTED}[{ACCENT_SUCCESS}{}{ACCENT_MUTED} (search: {ACCENT_WARN}{search}{ACCENT_MUTED}, effort: {ACCENT_PRIMARY}{effort_label}{ACCENT_MUTED})]{RESET}",
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
) -> RequestBody<'a> {
    let provider = models::model_provider(model);
    let endpoint = models::endpoint_for_model(model, "");
    let adapter = super::provider::adapter_for(provider, &endpoint);
    let request_model = models::request_model_name(model);
    let thinking_dialect =
        super::provider::thinking_dialect_for(provider, &request_model, &endpoint);
    // 流式请求显式索取 usage：部分 provider（DashScope compatible-mode）流式下
    // 默认不返回 usage，必须声明 stream_options.include_usage 才能统计 token。
    let stream_options = stream.then(|| json!({ "include_usage": true }));
    RequestBody {
        model: request_model,
        messages,
        stream,
        thinking: thinking_dialect.fields(enable_thinking),
        enable_search: adapter.enable_search_field(enable_search),
        tools,
        tool_choice,
        reasoning_effort: adapter.reasoning_top_level(reasoning_effort),
        reasoning: adapter.reasoning_nested(reasoning_effort),
        stream_options,
    }
}

fn ensure_reasoning_content_echo_for_thinking_model(model: &str, messages: &mut [Message]) {
    let provider = models::model_provider(model);
    let endpoint = models::endpoint_for_model(model, "");
    let request_model = models::request_model_name(model);
    let dialect = super::provider::thinking_dialect_for(provider, &request_model, &endpoint);
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
    let provider = models::model_provider(model);
    let endpoint = models::endpoint_for_model(model, "");
    let request_model = models::request_model_name(model);
    let dialect = super::provider::thinking_dialect_for(provider, &request_model, &endpoint);
    let fields = dialect.fields(false);
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
            "false",
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

    // 兼容（Qwen 等）provider 默认开启 thinking，会生成超长推理链。
    // 非流式辅助请求（意图识别、知识整理等）必须等整段生成完才返回响应头，
    // thinking 链一长就撑爆 60s 超时、重试也只是重复同样的慢生成。
    // 因此这里显式关闭 thinking，与后台 background_call 保持一致。
    apply_aux_thinking_fields(model, &mut request_body);

    // reasoning_effort：由 provider adapter 决定顶层或嵌套 wire 格式。
    if !skip_reasoning_effort {
        if let Some(effort) = resolve_reasoning_effort(app, model) {
            let endpoint = endpoint_for_request_model(app, model);
            let provider = models::model_provider(model);
            let adapter = super::provider::adapter_for(provider, &endpoint);
            if let Some(value) = adapter.reasoning_top_level(Some(effort.as_str())) {
                request_body["reasoning_effort"] = json!(value);
            }
            if let Some(value) = adapter.reasoning_nested(Some(effort.as_str())) {
                request_body["reasoning"] = value;
            }
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
                        "[Warning] do_request_json timeout (60s), retrying (attempt {}/{})",
                        attempt, REQUEST_MAX_ATTEMPTS
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
                let body = (response.text().await).unwrap_or_default();
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
        let send_future = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send();
        // 等待响应头：握手 + 服务端开始返回。流式下这一步很快。
        let mut response = match tokio::time::timeout(
            Duration::from_secs(STREAM_RESPONSE_HEADER_TIMEOUT_SECS),
            send_future,
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
                        "[Warning] do_request_text_streaming 等待响应头超时 ({}s), retrying (attempt {}/{})",
                        STREAM_RESPONSE_HEADER_TIMEOUT_SECS, attempt, REQUEST_MAX_ATTEMPTS
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
                "[Warning] do_request_text_streaming chunk 空闲超时 ({}s) 且无内容, retrying (attempt {}/{})",
                STREAM_RESPONSE_HEADER_TIMEOUT_SECS, attempt, REQUEST_MAX_ATTEMPTS
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

/// 通过 LLM 做用户意图识别（fallback 路径）。
///
/// 调用条件：本地 TF-IDF 给出 `Casual` 但问题文本看起来"非闲聊"
/// （比如带代码块、中等长度、显式问号 + 动词等）。这种情况下旧实现
/// 会被错分到 Casual，影响 thinking gate / skill 路由 / recall。
///
/// 接入要求：每次走到这里都会通过 [eprintln!] 打印 `[intent:llm]`
/// 标识，方便用户在终端可见地区分本地分类 vs 大模型分类。
///
/// 返回 `Some(core)` 仅当：
///   - HTTP 调用成功
///   - 返回的 JSON 能解析出 `intent` 字段
///   - confidence ≥ 0.6（与 thinking gate 一致的保守阈值）
pub async fn classify_intent_via_model(
    app: &App,
    question: &str,
) -> Option<crate::ai::driver::intent_recognition::CoreIntent> {
    use crate::ai::driver::intent_recognition::CoreIntent;

    let q = question.trim();
    if q.is_empty() {
        return None;
    }
    let clipped = if q.chars().count() > 800 {
        q.chars().take(800).collect::<String>()
    } else {
        q.to_string()
    };

    let gate_messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(
                "You are a user intent classifier. Output STRICT JSON only: \
{\"intent\":\"query_concept\"|\"request_action\"|\"seek_solution\"|\"casual\",\"confidence\":0.0}\n\
Definitions:\n\
- query_concept: 询问概念/定义（“是什么”、“什么意思”）\n\
- request_action: 请求执行操作（“帮我做”、“修复”、“实现”）\n\
- seek_solution: 寻求解决方案（“怎么处理”、“如何解决”、报错诊断）\n\
- casual: 闲聊或无明确意图\n\
confidence ∈ [0,1]，对边界样本请给低值（<0.6）。"
                    .to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(clipped),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let control_model = control_model_for_aux_tasks(app);

    // 用户可见标识：本次意图识别走的是 LLM 而非本地 TF-IDF。
    eprintln!(
        "[intent:llm] using model='{}' (local TF-IDF fell back to Casual on a non-trivial question)",
        control_model
    );

    let request_body = build_request_body(
        &control_model,
        &gate_messages,
        false,
        false,
        None,
        None,
        None,
        None,
    );

    let endpoint = endpoint_for_request_model(app, &control_model);
    let api_key = api_key_for_request_model(app, &control_model);

    // 意图分类是辅助请求，给 15 秒超时足够；避免因 API 无响应卡住整个 turn
    let request_future = async {
        let response = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .ok()?;
        if !response.status().is_success() {
            eprintln!("[intent:llm] http non-success status={}", response.status());
            return None;
        }
        response.text().await.ok()
    };
    let text = match tokio::time::timeout(Duration::from_secs(15), request_future).await {
        Ok(Some(t)) => t,
        Ok(None) => return None,
        Err(_) => {
            eprintln!("[intent:llm] timeout (15s), skipping");
            return None;
        }
    };
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = extract_router_content(&v)?;

    let s = strip_json_fence(&content);
    let candidate = if let (Some(l), Some(r)) = (s.find('{'), s.rfind('}'))
        && r >= l
    {
        &s[l..=r]
    } else {
        s
    };
    let parsed: Value = serde_json::from_str(candidate).ok()?;
    let intent_str = parsed.get("intent").and_then(|v| v.as_str())?;
    let confidence = parsed
        .get("confidence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    if confidence < 0.6 {
        eprintln!(
            "[intent:llm] low confidence ({:.2}); ignoring -> Casual",
            confidence
        );
        return None;
    }
    let core = match intent_str.to_ascii_lowercase().as_str() {
        "query_concept" => CoreIntent::QueryConcept,
        "request_action" => CoreIntent::RequestAction,
        "seek_solution" => CoreIntent::SeekSolution,
        "casual" => CoreIntent::Casual,
        _ => return None,
    };
    eprintln!("[intent:llm] -> {:?} (confidence={:.2})", core, confidence);
    Some(core)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::driver::intent_recognition::{CoreIntent, UserIntent};
    use crate::ai::tools::os_tools::{GLOBAL_OS, init_os_tools_globals};
    use crate::ai::{cli::ParsedCli, types::AppConfig};
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool};

    #[test]
    fn model_fallback_and_disable_statuses_are_separate() {
        let network = RequestError::cancelled("network timeout");
        assert!(should_try_model_fallback(&network));
        assert!(!should_temporarily_disable_model(&network));
        assert!(should_temporarily_disable_auto_selected_model(&network));

        let bad_request = RequestError::status(StatusCode::BAD_REQUEST, String::new());
        assert!(!should_try_model_fallback(&bad_request));
        assert!(!should_temporarily_disable_model(&bad_request));
        assert!(!should_temporarily_disable_auto_selected_model(
            &bad_request
        ));

        let unauthorized = RequestError::status(StatusCode::UNAUTHORIZED, String::new());
        assert!(should_try_model_fallback(&unauthorized));
        assert!(!should_temporarily_disable_model(&unauthorized));
        assert!(!should_temporarily_disable_auto_selected_model(
            &unauthorized
        ));

        let billing = RequestError::status(StatusCode::PAYMENT_REQUIRED, String::new());
        assert!(should_try_model_fallback(&billing));
        assert!(should_temporarily_disable_model(&billing));
        assert!(should_temporarily_disable_auto_selected_model(&billing));
    }

    #[test]
    fn auto_subagent_retry_policy_fails_fast_for_fallback() {
        let regular = request_retry_policy(false);
        assert_eq!(regular.max_attempts, REQUEST_MAX_ATTEMPTS);
        assert_eq!(regular.max_attempts_429, REQUEST_MAX_ATTEMPTS_429);
        assert_eq!(
            regular.header_timeout_secs,
            STREAM_RESPONSE_HEADER_TIMEOUT_SECS
        );

        let auto_subagent = request_retry_policy(true);
        assert_eq!(
            auto_subagent.max_attempts,
            AUTO_SUBAGENT_REQUEST_MAX_ATTEMPTS
        );
        assert_eq!(
            auto_subagent.max_attempts_429,
            AUTO_SUBAGENT_REQUEST_MAX_ATTEMPTS
        );
        assert_eq!(
            auto_subagent.header_timeout_secs,
            AUTO_SUBAGENT_RESPONSE_HEADER_TIMEOUT_SECS
        );
    }

    #[test]
    fn stream_usage_accepts_anthropic_style_field_aliases() {
        let usage: StreamUsage = serde_json::from_value(serde_json::json!({
            "input_tokens": 1200,
            "output_tokens": 345,
            "total_token_count": 1545,
        }))
        .unwrap();
        let usage = usage.normalized();
        assert_eq!(usage.prompt_tokens, 1200);
        assert_eq!(usage.completion_tokens, 345);
        assert_eq!(usage.total_tokens, 1545);
    }

    #[test]
    fn stream_usage_derives_missing_completion_from_total() {
        let usage = StreamUsage {
            prompt_tokens: 1000,
            completion_tokens: 0,
            total_tokens: 1234,
            ..Default::default()
        }
        .normalized();
        assert_eq!(usage.completion_tokens, 234);
        assert_eq!(usage.total_tokens, 1234);
    }

    #[test]
    fn stream_usage_recovers_output_from_reasoning_tokens() {
        let usage: StreamUsage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 800,
            "completion_tokens": 0,
            "completion_tokens_details": { "reasoning_tokens": 512 },
        }))
        .unwrap();
        let usage = usage.normalized();
        assert_eq!(usage.completion_tokens, 512);
        assert_eq!(usage.total_tokens, 1312);
    }

    #[test]
    fn stream_usage_does_not_double_count_reasoning_when_completion_present() {
        let usage: StreamUsage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 800,
            "completion_tokens": 600,
            "completion_tokens_details": { "reasoning_tokens": 512 },
        }))
        .unwrap();
        let usage = usage.normalized();
        assert_eq!(usage.completion_tokens, 600);
        assert_eq!(usage.total_tokens, 1400);
    }

    #[test]
    fn prompt_cache_breakpoint_wraps_first_system_message() {
        let mut messages = vec![
            Message {
                role: "system".to_string(),
                content: Value::String("you are helpful".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: Value::String("hi".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];
        apply_prompt_cache_breakpoint(&mut messages);

        // 第一条 system 消息被改写为内容块数组并带 cache_control。
        let blocks = messages[0].content.as_array().expect("array content");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "you are helpful");
        assert_eq!(blocks[0]["cache_control"]["type"], "ephemeral");
        // user 消息保持原样。
        assert_eq!(messages[1].content, Value::String("hi".to_string()));
    }

    #[test]
    fn prompt_cache_breakpoint_noop_without_system_message() {
        let mut messages = vec![Message {
            role: "user".to_string(),
            content: Value::String("hi".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        apply_prompt_cache_breakpoint(&mut messages);
        assert_eq!(messages[0].content, Value::String("hi".to_string()));
    }

    #[test]
    fn prompt_cache_model_support_uses_models_json_flag() {
        assert!(models::explicit_prompt_cache_enabled("qwen3.7-max"));
        assert!(models::explicit_prompt_cache_enabled("qwen3.7-plus"));
        assert!(models::explicit_prompt_cache_enabled("glm-5.1"));
    }

    #[test]
    fn prompt_cache_model_support_does_not_guess_by_name() {
        assert!(!models::explicit_prompt_cache_enabled(
            "anthropic/claude-sonnet-4"
        ));
        assert!(!models::explicit_prompt_cache_enabled("claude-3-5-sonnet"));
    }

    #[test]
    fn prompt_cache_model_support_rejects_plain_openai_model() {
        let Some(model) = first_openai_model_name() else {
            eprintln!(
                "[test] skipping prompt_cache_model_support_rejects_plain_openai_model: \
                 no OpenAi model present in models.json"
            );
            return;
        };
        assert!(!models::explicit_prompt_cache_enabled(&model));
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
                intent_model_path: PathBuf::new(),
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

    #[test]
    fn test_parse_thinking_gate_output_bool() {
        let s = r#"{"thinking":true,"confidence":0.91}"#;
        assert_eq!(parse_thinking_gate_output(s), Some((true, 0.91)));
    }

    #[test]
    fn test_parse_thinking_gate_output_string_bool() {
        let s = r#"{"thinking":"false","confidence":0.8}"#;
        assert_eq!(parse_thinking_gate_output(s), Some((false, 0.8)));
    }

    #[test]
    fn test_parse_thinking_gate_output_with_fence() {
        let s = "```json\n{\"thinking\":true,\"confidence\":0.73}\n```";
        assert_eq!(parse_thinking_gate_output(s), Some((true, 0.73)));
    }

    #[test]
    fn test_parse_thinking_gate_output_invalid() {
        let s = r#"{"confidence":0.73}"#;
        assert_eq!(parse_thinking_gate_output(s), None);
    }

    #[test]
    fn local_thinking_decision_skips_simple_concept_questions() {
        let intent = UserIntent::new(CoreIntent::QueryConcept);
        let decision = local_thinking_decision("Rust 的 trait 是什么？", &intent);
        assert_eq!(decision, Some(false));
    }

    #[test]
    fn local_thinking_decision_enables_for_debugging_requests() {
        let intent = UserIntent::new(CoreIntent::SeekSolution);
        let decision = local_thinking_decision(
            "帮我排查这个报错，并分析可能的修复方案\npanic: index out of bounds",
            &intent,
        );
        assert_eq!(decision, Some(true));
    }

    #[test]
    fn local_thinking_decision_leaves_ambiguous_short_actions_to_gate() {
        let intent = UserIntent::new(CoreIntent::RequestAction);
        let decision = local_thinking_decision("帮我写个函数", &intent);
        assert_eq!(decision, None);
    }

    #[test]
    fn strip_system_reminders_removes_injected_block() {
        let raw =
            "<system-reminder>\nlots of injected context\nmore lines\n</system-reminder>\n\nhi";
        assert_eq!(strip_system_reminders(raw), "\n\nhi");
    }

    #[test]
    fn strip_system_reminders_handles_multiple_and_unclosed() {
        let raw =
            "<system-reminder>a</system-reminder>real<system-reminder>b</system-reminder> text";
        assert_eq!(strip_system_reminders(raw), "real text");

        let unclosed = "<system-reminder>never closed and then the question hi";
        assert_eq!(strip_system_reminders(unclosed), "");
    }

    #[test]
    fn strip_system_reminders_passthrough_when_absent() {
        assert_eq!(strip_system_reminders("hi"), "hi");
    }

    #[test]
    fn reminder_polluted_greeting_decides_locally() {
        // 模拟被 system-reminder 撑长的 "hi"：剥离后应命中本地短路（Casual+短），
        // 而不是落到耗时的模型 gate。
        let polluted = format!(
            "<system-reminder>{}</system-reminder>\n\nhi",
            "x".repeat(2000)
        );
        let clean = strip_system_reminders(&polluted);
        let clean = clean.trim();
        let intent = UserIntent::new(CoreIntent::Casual);
        assert_eq!(local_thinking_decision(clean, &intent), Some(false));
    }

    #[test]
    fn thinking_gate_uses_latest_user_message_only() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Value::String(
                    "请帮我排查这个复杂报错，并分析可能的修复方案\npanic: index out of bounds"
                        .to_string(),
                ),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String("之前的复杂问题已经回答完毕。".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: Value::String("为什么天是蓝的？".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        assert_eq!(
            latest_user_message_text(&messages).as_deref(),
            Some("为什么天是蓝的？")
        );
    }

    #[tokio::test]
    async fn sleep_with_cancel_observes_request_interrupt_source() {
        let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        let app = test_app();
        init_os_tools_globals(app.os.clone());
        crate::ai::driver::signal::clear_request_interrupt();

        let waiter =
            tokio::spawn(async move { sleep_with_cancel(&app, Duration::from_secs(5)).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        crate::ai::driver::signal::signal_request_interrupt();

        let cancelled = tokio::time::timeout(Duration::from_millis(200), waiter)
            .await
            .expect("retry wait should wake on interrupt")
            .expect("waiter should complete");
        assert!(cancelled);

        crate::ai::driver::signal::clear_request_interrupt();
        if let Ok(mut guard) = GLOBAL_OS.lock() {
            *guard = None;
        }
    }

    #[test]
    fn clears_stale_interrupt_for_new_request_but_keeps_active_cancel() {
        let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        let app = test_app();
        init_os_tools_globals(app.os.clone());
        crate::ai::driver::signal::clear_request_interrupt();

        crate::ai::driver::signal::signal_request_interrupt();
        assert!(crate::ai::driver::signal::request_interrupt_ready());
        clear_stale_request_interrupt_before_request(&app);
        assert!(!crate::ai::driver::signal::request_interrupt_ready());

        app.cancel_stream
            .store(true, std::sync::atomic::Ordering::Relaxed);
        crate::ai::driver::signal::signal_request_interrupt();
        clear_stale_request_interrupt_before_request(&app);
        assert!(crate::ai::driver::signal::request_interrupt_ready());

        app.cancel_stream
            .store(false, std::sync::atomic::Ordering::Relaxed);
        crate::ai::driver::signal::clear_request_interrupt();
        if let Ok(mut guard) = GLOBAL_OS.lock() {
            *guard = None;
        }
    }

    /// 找一个真实存在的 OpenAi-provider 模型名做测试输入，避免硬编码
    /// 具体模型字符串导致 models.json 变更后测试失效。
    fn first_openai_model_name() -> Option<String> {
        crate::ai::model_names::all()
            .iter()
            .find(|m| m.provider == crate::ai::provider::ApiProvider::OpenAi)
            .map(|m| m.name.clone())
    }

    fn first_openai_vl_model_name() -> Option<String> {
        crate::ai::model_names::all()
            .iter()
            .find(|m| m.provider == crate::ai::provider::ApiProvider::OpenAi && m.is_vl)
            .map(|m| m.name.clone())
    }

    fn first_alibaba_vl_model_name() -> Option<String> {
        crate::ai::model_names::all()
            .iter()
            .find(|m| m.provider == crate::ai::provider::ApiProvider::Alibaba && m.is_vl)
            .map(|m| m.name.clone())
    }

    fn first_model_name_for_provider(provider: crate::ai::provider::ApiProvider) -> Option<String> {
        crate::ai::model_names::all()
            .iter()
            .find(|m| m.provider == provider)
            .map(|m| m.name.clone())
    }

    /// 逐字节 wire guard：锁死各 provider 的 `build_request_body` 序列化结果，
    /// 作为 provider adapter 重构「不破坏对外 wire 行为」的可执行回归网。
    /// 字段顺序由 [`RequestBody`] 声明顺序决定，serde 输出稳定可断言整串。
    #[test]
    fn build_request_body_wire_format_is_byte_stable_per_provider() {
        use crate::ai::provider::ApiProvider;

        let messages = vec![Message {
            role: "user".to_string(),
            content: Value::String("hi".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        // Alibaba：嵌套 reasoning.effort + enable_thinking/enable_search，无 stream_options（非流式）。
        let alibaba_model = first_model_name_for_provider(ApiProvider::Alibaba)
            .expect("models.json must contain an Alibaba model");
        let alibaba = build_request_body(
            &alibaba_model,
            &messages,
            false,
            true,
            Some(true),
            None,
            None,
            Some("high"),
        );
        assert_eq!(
            serde_json::to_string(&alibaba).unwrap(),
            format!(
                r#"{{"model":"{alibaba_model}","messages":[{{"role":"user","content":"hi"}}],"stream":false,"enable_thinking":true,"enable_search":true,"reasoning":{{"effort":"high"}}}}"#
            )
        );

        // OpenCode 非 DeepSeek：与 OpenAI 兼容族字段一致
        // （顶层 reasoning_effort、省略扩展字段）。DeepSeek 专属的 `thinking`
        // 字段由单独的 `deepseek_request_body_uses_thinking_object` 测试覆盖。
        let non_deepseek_opencode = crate::ai::model_names::all()
            .iter()
            .find(|m| {
                m.provider == ApiProvider::OpenCode
                    && !m.name.to_ascii_lowercase().contains("deepseek")
            })
            .map(|m| m.name.clone());
        if let Some(opencode_model) = non_deepseek_opencode {
            let opencode = build_request_body(
                &opencode_model,
                &messages,
                false,
                true,
                Some(true),
                None,
                None,
                Some("medium"),
            );
            assert_eq!(
                serde_json::to_string(&opencode).unwrap(),
                format!(
                    r#"{{"model":"{opencode_model}","messages":[{{"role":"user","content":"hi"}}],"stream":false,"reasoning_effort":"medium"}}"#
                )
            );
        }
    }

    #[test]
    fn build_request_body_sends_provider_model_name_for_key_handle() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Value::String("hi".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        let body = build_request_body(
            "deepseek-v4-flash-opencode",
            &messages,
            false,
            true,
            Some(true),
            None,
            None,
            Some("high"),
        );
        let json = serde_json::to_value(&body).unwrap();

        assert_eq!(
            json.get("model").and_then(|v| v.as_str()),
            Some("deepseek-v4-flash")
        );
        assert_eq!(
            json.pointer("/thinking/type").and_then(|v| v.as_str()),
            Some("enabled")
        );
    }

    #[test]
    fn deepseek_request_body_uses_thinking_object() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Value::String("hi".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        // 关闭：thinking={"type":"disabled"}
        let disabled = build_request_body(
            "deepseek-v4-flash-free",
            &messages,
            false,
            false,
            None,
            None,
            None,
            None,
        );
        let disabled = serde_json::to_value(&disabled).unwrap();
        assert_eq!(
            disabled.get("thinking"),
            Some(&json!({ "type": "disabled" }))
        );
        // DeepSeek 不应再发送 enable_thinking（避免与 thinking 对象冲突/无效）。
        assert!(disabled.get("enable_thinking").is_none());

        // 开启：thinking={"type":"enabled"}
        let enabled = build_request_body(
            "deepseek-v4-flash-free",
            &messages,
            false,
            true,
            None,
            None,
            None,
            None,
        );
        let enabled = serde_json::to_value(&enabled).unwrap();
        assert_eq!(enabled.get("thinking"), Some(&json!({ "type": "enabled" })));
    }

    #[test]
    fn non_deepseek_request_body_omits_thinking_object() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Value::String("hi".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let body = build_request_body(
            "qwen3.7-plus",
            &messages,
            true,
            false,
            None,
            None,
            None,
            None,
        );
        let value = serde_json::to_value(&body).unwrap();
        assert!(value.get("thinking").is_none());
    }

    #[test]
    fn deepseek_tool_call_messages_echo_empty_reasoning_content() {
        let mut messages = vec![Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![crate::ai::types::ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        }];

        ensure_reasoning_content_echo_for_thinking_model("deepseek-v4-flash-free", &mut messages);
        assert_eq!(messages[0].reasoning_content.as_deref(), Some(""));

        let value = serde_json::to_value(&messages[0]).unwrap();
        assert_eq!(
            value.get("reasoning_content").and_then(|v| v.as_str()),
            Some("")
        );
    }

    /// 核心回归：DashScope compatible-mode 端点的 Alibaba-provider 模型
    /// （deepseek-v4-pro/flash、kimi-k2.7-code）必须按 thinking gate 决策发送
    /// `enable_thinking`，否则「关闭」会被静默丢弃、模型仍 reasoning。
    #[test]
    fn dashscope_alibaba_provider_honors_enable_thinking_gate() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Value::String("hi".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        for model in ["deepseek-v4-pro", "deepseek-v4-flash", "kimi-k2.7-code"] {
            // gate 关闭 → enable_thinking:false
            let disabled =
                build_request_body(model, &messages, false, false, None, None, None, None);
            let disabled = serde_json::to_value(&disabled).unwrap();
            assert_eq!(
                disabled.get("enable_thinking").and_then(|v| v.as_bool()),
                Some(false),
                "{model} should emit enable_thinking:false when gate disables thinking"
            );
            // 走 enable_thinking 而非 deepseek 的 thinking 对象
            assert!(disabled.get("thinking").is_none(), "{model}");

            // gate 开启 → enable_thinking:true
            let enabled = build_request_body(model, &messages, false, true, None, None, None, None);
            let enabled = serde_json::to_value(&enabled).unwrap();
            assert_eq!(
                enabled.get("enable_thinking").and_then(|v| v.as_bool()),
                Some(true),
                "{model} should emit enable_thinking:true when gate enables thinking"
            );
        }
    }

    /// 辅助（非主链路）请求对 DashScope 端点模型必须显式关闭 thinking，
    /// 否则默认开启的长推理链会撑爆辅助任务超时。
    #[test]
    fn dashscope_aux_requests_disable_thinking_regardless_of_provider() {
        for model in ["qwen3.7-plus", "deepseek-v4-pro", "kimi-k2.7-code"] {
            let mut body = json!({ "model": model, "messages": [], "stream": false });
            apply_aux_thinking_fields(model, &mut body);
            assert_eq!(
                body.get("enable_thinking").and_then(|v| v.as_bool()),
                Some(false),
                "{model} aux request should disable thinking via enable_thinking:false"
            );
        }

        // OpenCode 的 deepseek 不靠 enable_thinking，aux 关闭走 thinking 对象。
        let mut deepseek =
            json!({ "model": "deepseek-v4-flash-free", "messages": [], "stream": false });
        apply_aux_thinking_fields("deepseek-v4-flash-free", &mut deepseek);
        assert_eq!(
            deepseek.get("thinking"),
            Some(&json!({ "type": "disabled" }))
        );
        assert!(deepseek.get("enable_thinking").is_none());

        // OpenCode 非 deepseek 无可靠关闭开关，aux 不注入任何思考字段。
        let mut mimo = json!({ "model": "mimo-v2.5-free", "messages": [], "stream": false });
        apply_aux_thinking_fields("mimo-v2.5-free", &mut mimo);
        assert!(mimo.get("thinking").is_none());
        assert!(mimo.get("enable_thinking").is_none());
    }

    #[test]
    fn openai_request_body_omits_nonstandard_flags() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Value::String("hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let Some(model) = first_openai_model_name() else {
            eprintln!(
                "[test] skipping openai_request_body_omits_nonstandard_flags: \
                 no OpenAi model present in models.json"
            );
            return;
        };
        let body = build_request_body(
            &model,
            &messages,
            true,
            true,
            Some(true),
            None,
            None,
            Some("high"),
        );
        let value = serde_json::to_value(&body).unwrap();

        // OpenAI-provider 不发送 DashScope 扩展字段，推理强度走顶层 reasoning_effort。
        assert!(value.get("enable_thinking").is_none());
        assert!(value.get("enable_search").is_none());
        assert_eq!(
            value.get("reasoning_effort").and_then(|v| v.as_str()),
            Some("high")
        );
        assert!(value.get("reasoning").is_none());
        assert_eq!(
            value.get("model").and_then(|v| v.as_str()),
            Some(model.as_str())
        );
    }

    #[test]
    fn alibaba_request_body_keeps_extension_flags() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Value::String("hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let body = build_request_body(
            "qwen3.7-plus",
            &messages,
            false,
            true,
            Some(true),
            None,
            None,
            Some("high"),
        );
        let value = serde_json::to_value(&body).unwrap();

        assert_eq!(
            value.get("enable_thinking").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            value.get("enable_search").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert!(value.get("reasoning_effort").is_none());
        assert_eq!(
            value
                .get("reasoning")
                .and_then(|v| v.get("effort"))
                .and_then(|v| v.as_str()),
            Some("high")
        );
    }

    #[test]
    fn normalize_messages_merges_non_leading_system_messages() {
        // Internal notes that appear AFTER the first conversational message
        // must remain in their original positions (with role normalized to
        // "system") so that older prompt-cache prefixes stay valid when new
        // notes are appended. Only notes that sit at the very top, before
        // any user/assistant/tool message, get folded into the first system.
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: Value::String("base system".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
                content: Value::String("history summary".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: Value::String("question".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String("answer".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
                content: Value::String("working memory".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        let normalized = normalize_messages_for_request(&messages);

        assert_eq!(normalized[0].role, "system");
        let head_text = normalized[0].content.as_str().unwrap();
        assert!(head_text.contains("base system"));
        assert!(head_text.contains("history summary"));
        assert!(!head_text.contains("working memory"));

        assert_eq!(normalized[1].role, "user");
        assert_eq!(normalized[2].role, "assistant");
        assert_eq!(normalized[3].role, "system");
        assert_eq!(normalized[3].content.as_str(), Some("working memory"));
    }

    #[test]
    fn normalize_messages_prioritizes_working_memory_before_summary_and_self_note() {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: Value::String("base system".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
                content: Value::String("self_note:\nremember style".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
                content: Value::String(
                    "对话摘要（自动压缩，以下为早期对话要点）：\nolder summary".to_string(),
                ),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
                content: Value::String(
                    "Current code-inspection working memory:\n- use code_search first".to_string(),
                ),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        let normalized = normalize_messages_for_request(&messages);
        let text = normalized[0].content.as_str().unwrap();
        let wm = text.find("## Working Memory").unwrap();
        let summary = text.find("## History Summary").unwrap();
        let self_note = text.find("## Self Notes").unwrap();
        assert!(wm < summary);
        assert!(summary < self_note);
    }

    #[test]
    fn normalize_messages_drops_orphan_tool_results_and_strips_broken_tool_calls() {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: Value::String("base system".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: Value::String("question".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![crate::ai::types::ToolCall {
                    id: "call_1".to_string(),
                    tool_type: "function".to_string(),
                    function: crate::ai::types::FunctionCall {
                        name: "read_file".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String("later answer".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("stale tool output".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                reasoning_content: None,
            },
        ];

        let normalized = normalize_messages_for_request(&messages);

        assert_eq!(normalized.len(), 3);
        assert_eq!(normalized[0].role, "system");
        assert_eq!(normalized[1].role, "user");
        assert_eq!(normalized[2].role, "assistant");
        assert_eq!(normalized[2].content.as_str(), Some("later answer"));
        assert!(normalized.iter().all(|message| message.role != "tool"));
    }

    #[test]
    fn normalize_messages_keeps_contiguous_tool_call_blocks() {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: Value::String("base system".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: Value::String("question".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![crate::ai::types::ToolCall {
                    id: "call_1".to_string(),
                    tool_type: "function".to_string(),
                    function: crate::ai::types::FunctionCall {
                        name: "read_file".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("fresh tool output".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                reasoning_content: None,
            },
        ];

        let normalized = normalize_messages_for_request(&messages);

        assert_eq!(normalized.len(), 4);
        assert_eq!(normalized[2].role, "assistant");
        assert_eq!(
            normalized[2].tool_calls.as_ref().map(|calls| calls.len()),
            Some(1)
        );
        assert_eq!(normalized[3].role, "tool");
        assert_eq!(normalized[3].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn normalize_messages_preserves_tool_evidence_when_malformed_tool_calls_are_dropped() {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: Value::String("base system".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: Value::String("question".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![crate::ai::types::ToolCall {
                    id: "call_1".to_string(),
                    tool_type: "function".to_string(),
                    function: crate::ai::types::FunctionCall {
                        name: "execute_command".to_string(),
                        arguments: "{\"command\":".to_string(),
                    },
                }]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("Error: failed to parse arguments".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String("later answer".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        let normalized = normalize_messages_for_request(&messages);

        assert_eq!(normalized.len(), 4);
        assert_eq!(normalized[2].role, "internal_note");
        let note = normalized[2].content.as_str().unwrap_or_default();
        assert!(note.contains("preserved unmatched tool outputs"));
        assert!(note.contains("failed to parse arguments"));
        assert_eq!(normalized[3].role, "assistant");
        assert_eq!(normalized[3].content.as_str(), Some("later answer"));
        assert!(normalized.iter().all(|message| message.role != "tool"));
        assert!(
            normalized
                .iter()
                .skip(1)
                .all(|message| message.tool_calls.is_none())
        );
    }

    #[test]
    fn normalize_messages_truncates_long_internal_notes_structurally() {
        let mut long_note_lines = Vec::new();
        long_note_lines.push("Current code-inspection working memory:".to_string());
        for i in 0..80usize {
            long_note_lines.push(format!("- finding {i:02}: {}", "x".repeat(40)));
        }

        let messages = vec![
            Message {
                role: "system".to_string(),
                content: Value::String("base system".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: Value::String("question".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
                content: Value::String(long_note_lines.join("\n")),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        let normalized = normalize_messages_for_request(&messages);
        assert_eq!(normalized.len(), 3);
        assert_eq!(normalized[2].role, "system");
        let text = normalized[2].content.as_str().unwrap_or_default();
        assert!(text.contains("Current code-inspection working memory:"));
        assert!(text.contains("[truncated:"));
        assert!(text.chars().count() <= 1_200);
    }

    #[test]
    fn openai_image_content_uses_object_image_url_shape() {
        // 仅当 models.json 中存在一个 OpenAi-provider 且 is_vl=true 的模型时
        // 才能验证"以 {image_url:{url:...}} 对象形状下发图像"的协议契约。
        // 真实环境下没有这种模型时（例如 models.json 只有 Compatible VL），
        // 这条契约无从验证，跳过即可。
        let Some(model) = first_openai_vl_model_name() else {
            eprintln!(
                "[test] skipping openai_image_content_uses_object_image_url_shape: \
                 no OpenAi+VL model present in models.json"
            );
            return;
        };

        let path =
            std::env::temp_dir().join(format!("ai-openai-image-{}.png", uuid::Uuid::new_v4()));
        std::fs::write(&path, b"fake").unwrap();

        let value =
            build_content(&model, "describe", &[path.to_string_lossy().to_string()]).unwrap();

        let first = value.as_array().and_then(|items| items.first()).unwrap();
        assert_eq!(
            first.get("type").and_then(|v| v.as_str()),
            Some("image_url")
        );
        assert!(
            first
                .get("image_url")
                .and_then(|v| v.get("url"))
                .and_then(|v| v.as_str())
                .map(|s| s.starts_with("data:image/png;base64,"))
                .unwrap_or(false)
        );
    }

    #[test]
    fn alibaba_image_content_also_uses_object_image_url_shape() {
        let Some(model) = first_alibaba_vl_model_name() else {
            eprintln!(
                "[test] skipping alibaba_image_content_also_uses_object_image_url_shape: \
                 no Alibaba+VL model present in models.json"
            );
            return;
        };

        let path =
            std::env::temp_dir().join(format!("ai-alibaba-image-{}.png", uuid::Uuid::new_v4()));
        std::fs::write(&path, b"fake").unwrap();

        let value =
            build_content(&model, "describe", &[path.to_string_lossy().to_string()]).unwrap();

        let first = value.as_array().and_then(|items| items.first()).unwrap();
        assert_eq!(
            first.get("type").and_then(|v| v.as_str()),
            Some("image_url")
        );
        assert!(
            first
                .get("image_url")
                .and_then(|v| v.get("url"))
                .and_then(|v| v.as_str())
                .map(|s| s.starts_with("data:image/png;base64,"))
                .unwrap_or(false)
        );
    }

    #[test]
    fn normalize_messages_downgrades_image_content_for_text_only_models() {
        let Some(model) = crate::ai::model_names::all()
            .iter()
            .find(|m| !m.is_vl)
            .map(|m| m.name.clone())
        else {
            eprintln!(
                "[test] skipping normalize_messages_downgrades_image_content_for_text_only_models: no text-only model present in models.json"
            );
            return;
        };

        let messages = vec![
            Message {
                role: "system".to_string(),
                content: Value::String("base system".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: Value::Array(vec![
                    serde_json::json!({
                        "type": "image_url",
                        "image_url": { "url": "data:image/png;base64,AAAA" }
                    }),
                    serde_json::json!({
                        "type": "text",
                        "text": "please explain"
                    }),
                ]),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        let normalized = normalize_messages_for_model(&model, &messages);

        assert!(
            normalized
                .iter()
                .all(|message| !matches!(message.content, Value::Array(_)))
        );
        let content = normalized[1].content.as_str().unwrap();
        assert!(content.contains("[image omitted]"));
        assert!(content.contains("please explain"));
    }
}
