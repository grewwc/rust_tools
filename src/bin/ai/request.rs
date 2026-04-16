use std::fmt;
use std::fs;
use std::time::{Duration, Instant};

use base64::Engine as _;
use colored::Colorize;
use reqwest::{Response, StatusCode};
use rust_tools::commonw;
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{
    files,
    history::{Message, ROLE_SYSTEM, is_internal_note_role, is_system_like_role, messages_to_markdown},
    models,
    provider::ApiProvider,
    skills::SkillManifest,
    types::App,
};
use crate::ai::config_schema::AiConfig;
use crate::ai::driver::intent_recognition;
use crate::commonw::configw;

#[derive(Debug, Serialize)]
struct RequestBody<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_search: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct StreamChunk {
    #[serde(default, deserialize_with = "vec_or_default")]
    pub(super) choices: Vec<StreamChoice>,
}

impl StreamChunk {
    pub(super) fn merge_reasoning(&mut self) {
        for choice in &mut self.choices {
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
        ',' | '.' | ';' | ':'
        | '!' | '?'
        | ')' | ']' | '}'
        | '，' | '。' | '；' | '：'
        | '！' | '？'
        | '）' | '】' | '》'
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
const DEFAULT_AUTO_THINKING_THRESHOLD: f64 = 0.7;
const DEFAULT_CONTROL_MODEL: &str = "qwen3.5-flash";

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

fn apply_request_auth(
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
}

async fn sleep_with_cancel(app: &App, delay: Duration) -> bool {
    let started_at = std::time::Instant::now();
    while started_at.elapsed() < delay {
        if should_abort_retry_wait(app) {
            return true;
        }
        let remaining = delay.saturating_sub(started_at.elapsed());
        tokio::time::sleep(remaining.min(Duration::from_millis(50))).await;
    }
    should_abort_retry_wait(app)
}

fn control_model_for_aux_tasks(app: &App) -> String {
    app.config
        .intent_model
        .as_deref()
        .filter(|v| !v.trim().is_empty())
        .map(models::determine_model)
        .unwrap_or_else(|| match models::model_provider(&app.current_model) {
            ApiProvider::Compatible => models::determine_model(DEFAULT_CONTROL_MODEL),
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

/// Resolve whether to enable thinking mode for this request.
///
/// Decision order:
/// 1. CLI `--thinking` flag always wins
/// 2. Config `ai.model.thinking=true` forces thinking when the model supports it
/// 3. If model doesn't support thinking, return false
/// 4. If auto-thinking is disabled by config, return false
/// 5. Auto-detect based on question complexity
#[commonw::debug_measure_time("resolve_thinking")]
async fn resolve_thinking(app: &App, model: &str, messages: &[Message]) -> bool {
    let cfg = configw::get_all_config();
    let force_thinking = app.cli.thinking || config_bool_is_true(cfg.get_opt(AiConfig::MODEL_THINKING));

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

    let question = messages
        .iter()
        .filter(|m| m.role == "user")
        .filter_map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let question = question.trim();
    if !question.is_empty() {
        let local_intent =
            intent_recognition::detect_intent_with_model_path(question, &app.config.intent_model_path);
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

fn local_thinking_decision(
    question: &str,
    intent: &intent_recognition::UserIntent,
) -> Option<bool> {
    let question = question.trim();
    if question.is_empty() {
        return Some(false);
    }

    let question_len = question.chars().count();
    let line_count = question.lines().count();
    let has_code_like_content = question.contains("```")
        || question.contains("::")
        || question.contains("fn ")
        || question.contains("panic")
        || question.contains("traceback")
        || question.contains("Exception")
        || question.contains("Error")
        || question.contains("error");
    let has_multistep_shape = line_count >= 3
        || question.contains("\n- ")
        || question.contains("\n1.")
        || question.contains("步骤")
        || question.contains("step by step");
    let looks_like_complex_solution_request = matches!(
        intent.core,
        intent_recognition::CoreIntent::SeekSolution
    ) && (question_len >= 48 || has_code_like_content || has_multistep_shape);
    let looks_like_complex_action_request = matches!(
        intent.core,
        intent_recognition::CoreIntent::RequestAction
    ) && (question_len >= 96 || has_code_like_content || has_multistep_shape);

    if matches!(
        intent.core,
        intent_recognition::CoreIntent::Casual | intent_recognition::CoreIntent::QueryConcept
    ) && question_len <= 120
        && !has_code_like_content
        && !intent.is_search_query()
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
async fn decide_thinking_via_model(
    app: &App,
    _model: &str,
    messages: &[Message],
) -> Option<bool> {
    let gate_start = Instant::now();
    let user_text: String = messages
        .iter()
        .filter(|m| m.role == "user")
        .filter_map(extract_message_text)
        .collect::<Vec<_>>()
        .join("\n");
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
        },
        Message {
            role: "user".to_string(),
            content: Value::String(clipped),
            tool_calls: None,
            tool_call_id: None,
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
    );

    let endpoint = endpoint_for_request_model(app, &control_model);
    let api_key = api_key_for_request_model(app, &control_model);
    let response = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await
        .ok()?;

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

    let text = response.text().await.ok()?;
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
            if out.is_empty() {
                None
            } else {
                Some(out)
            }
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
    let normalized_messages = normalize_messages_for_request(messages);
    let (tools_value, tool_choice) = agent_tools_for_request(app, model);
    let thinking_start = Instant::now();
    let force_thinking_requested = app.cli.thinking || config_forces_thinking();
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
    let request_body = build_request_body(
        model,
        &normalized_messages,
        stream,
        enable_thinking,
        models::search_enabled(model).then_some(true),
        tools_value,
        tool_choice,
    );

    for attempt in 1..=REQUEST_MAX_ATTEMPTS_429 {
        let endpoint = endpoint_for_request_model(app, model);
        let api_key = api_key_for_request_model(app, model);
        let response = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await;

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
                    REQUEST_MAX_ATTEMPTS_429
                } else {
                    REQUEST_MAX_ATTEMPTS
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
                if retryable && attempt < REQUEST_MAX_ATTEMPTS {
                    // 打印 sleep 原因
                    let delay = retry_delay(attempt);
                    eprintln!(
                        "[Warning] 网络错误 - sleep {} 秒后重试 (attempt {}/{})",
                        delay.as_secs_f32(),
                        attempt,
                        REQUEST_MAX_ATTEMPTS
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

    let mut system_prompt = r#"You are a skill router for a CODE development assistant. Your ONLY job is to route SOURCE CODE programming questions to appropriate skills.
ALL skills are for PROGRAMMING/CODING tasks only (writing, reviewing, debugging, refactoring code).
Output schema: {"skill":"<exact skill name or empty>","confidence":0.0}
Critical Rules:
- MANDATORY CHECK 1: Is the user asking to WRITE/REVIEW/DEBUG/REFACTOR specific SOURCE CODE (files or snippets)? If NO → return empty skill.
- MANDATORY CHECK 2: Is this a GENERAL KNOWLEDGE question about programming concepts/APIs/libraries (e.g., 'What is X?', 'How to use X?', 'X 是什么')? If YES → return empty skill. These are documentation/knowledge questions, NOT code tasks.
- NON-CODE TOPICS (ALWAYS return empty): documents/docs (文档，云文档，飞书文档), notes, wikis, data analysis, business questions, news, sports, weather, stocks, music, movies, general knowledge, project management, architecture design (without code), programming concept questions (API 说明，库的用法，框架特性).
- KEY INDICATOR: Does the question include ACTUAL CODE SNIPPET or SPECIFIC FILE PATH and ask to operate on it (review, fix, refactor, write)? If NO → likely not a code task.
- QUESTION TYPE MATTERS: 'X 是什么' / 'What is X' / '怎么用 X' / 'How to use X' = knowledge question (NO skill). '帮我 review 这段代码' / 'Fix this bug' / 'Refactor this function' = code task (YES skill).
- Chinese word traps: '看一下'、'检查'、'分析'、'审查'、'文档'、'架构' do NOT mean code tasks unless explicitly about operating on SOURCE CODE files/snippets.
- If unsure, return empty skill with confidence < 0.5. Better to skip than misroute.
- Use EXACT skill name from the list below.
- Return ONLY valid JSON, no explanations.
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
        },
        Message {
            role: "user".to_string(),
            content: Value::String(question.to_string()),
            tool_calls: None,
            tool_call_id: None,
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
    );

    let endpoint = endpoint_for_request_model(app, &control_model);
    let api_key = api_key_for_request_model(app, &control_model);
    let response = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }

    let text = response.text().await.ok()?;
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
            let Some(decision) = select_skill_candidate_via_model(app, question, chunk).await else {
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
            select_skill_candidate_via_model(app, question, &finalists).await.or_else(|| {
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
        if trimmed.starts_with("对话摘要（自动压缩") || trimmed.starts_with("历史摘要（自动压缩") {
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

    let first_system_idx = messages
        .iter()
        .position(|m| m.role == ROLE_SYSTEM || is_internal_note_role(&m.role));
    let Some(first_system_idx) = first_system_idx else {
        return messages.to_vec();
    };

    let mut merged_notes: Vec<(usize, InternalNoteKind, String)> = Vec::new();
    for (idx, message) in messages.iter().enumerate() {
        if idx == first_system_idx || !is_system_like_role(&message.role) {
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
                truncate_chars(text, MERGED_SINGLE_NOTE_MAX_CHARS),
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
        if idx == first_system_idx || is_system_like_role(&message.role) {
            continue;
        }
        out.push(message.clone());
    }
    out
}

pub(super) fn build_content(
    model: &str,
    question: &str,
    image_files: &[String],
) -> Result<Value, Box<dyn std::error::Error>> {
    if !models::supports_image_input(model) || image_files.is_empty() {
        return Ok(Value::String(question.to_string()));
    }

    let provider = models::model_provider(model);
    let mut parts = Vec::new();
    for file in image_files {
        let bytes = fs::read(file)?;
        let mime = files::image_mime_type(file);
        let image = base64::engine::general_purpose::STANDARD.encode(bytes);
        parts.push(match provider {
            ApiProvider::Compatible => json!({
                "type": "image_url",
                "image_url": format!("data:{mime};base64,{image}"),
            }),
            _ => json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{mime};base64,{image}")
                },
            }),
        });
    }
    parts.push(json!({
        "type": "text",
        "text": question,
    }));
    Ok(Value::Array(parts))
}

pub(super) fn print_info(model: &str) {
    let search = if models::search_enabled(model) {
        "true"
    } else {
        "false"
    };
    // 使用 println! 避免手动 flush 的权限问题
    println!("[{} (search: {})]", model.green(), search.red());
}

fn build_request_body<'a>(
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
    enable_thinking: bool,
    enable_search: Option<bool>,
    tools: Option<Value>,
    tool_choice: Option<Value>,
) -> RequestBody<'a> {
    let provider = models::model_provider(model);
    match provider {
        ApiProvider::Compatible => RequestBody {
            model,
            messages,
            stream,
            enable_thinking: Some(enable_thinking),
            enable_search,
            tools,
            tool_choice,
        },
        _ => RequestBody {
            model,
            messages,
            stream,
            enable_thinking: None,
            enable_search: None,
            tools,
            tool_choice,
        },
    }
}

/// 使用 LLM 进行 JSON 格式的请求（用于意图识别等场景）
pub async fn do_request_json(
    app: &App,
    model: &str,
    messages: &[serde_json::Value],
    stream: bool,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let request_body = json!({
        "model": model,
        "messages": messages,
        "stream": stream,
    });

    for attempt in 1..=REQUEST_MAX_ATTEMPTS {
        let endpoint = endpoint_for_request_model(app, model);
        let api_key = api_key_for_request_model(app, model);
        let response = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await;

        match response {
            Ok(response) => {
                if response.status().is_success() {
                    let json: serde_json::Value = response.json().await?;
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

pub(super) async fn summarize_history_via_model(
    app: &App,
    messages: &[Message],
    max_chars: usize,
) -> Option<String> {
    if messages.is_empty() || max_chars == 0 {
        return None;
    }

    let transcript = messages_to_markdown(messages, &app.session_id);
    let transcript = if transcript.chars().count() > 24_000 {
        let head: String = transcript.chars().take(16_000).collect();
        let tail: String = transcript
            .chars()
            .rev()
            .take(6_000)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        format!("{head}\n\n[... older transcript omitted for summary budget ...]\n\n{tail}")
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
- 总长度尽量控制在 {} 个字符以内。", max_chars
            )),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(format!(
                "请压缩下面的较早对话：\n\n{}",
                transcript
            )),
            tool_calls: None,
            tool_call_id: None,
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
    );
    let endpoint = endpoint_for_request_model(app, &control_model);
    let api_key = api_key_for_request_model(app, &control_model);
    let response = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    let text = response.text().await.ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = extract_router_content(&v)?;
    let trimmed = content.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::driver::intent_recognition::{CoreIntent, UserIntent};

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
    fn openai_request_body_omits_nonstandard_flags() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Value::String("hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
        }];
        let body = build_request_body(
            "gpt-4o",
            &messages,
            true,
            true,
            Some(true),
            None,
            None,
        );
        let value = serde_json::to_value(&body).unwrap();

        assert!(value.get("enable_thinking").is_none());
        assert!(value.get("enable_search").is_none());
        assert_eq!(value.get("model").and_then(|v| v.as_str()), Some("gpt-4o"));
    }

    #[test]
    fn compatible_request_body_keeps_extension_flags() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Value::String("hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
        }];
        let body = build_request_body(
            "qwen",
            &messages,
            false,
            true,
            Some(true),
            None,
            None,
        );
        let value = serde_json::to_value(&body).unwrap();

        assert_eq!(value.get("enable_thinking").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(value.get("enable_search").and_then(|v| v.as_bool()), Some(true));
    }

    #[test]
    fn normalize_messages_merges_non_leading_system_messages() {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: Value::String("base system".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: "user".to_string(),
                content: Value::String("question".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
                content: Value::String("history summary".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String("answer".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
                content: Value::String("working memory".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        let normalized = normalize_messages_for_request(&messages);

        assert_eq!(normalized[0].role, "system");
        assert_eq!(normalized.iter().filter(|m| m.role == "system").count(), 1);
        let text = normalized[0].content.as_str().unwrap();
        assert!(text.contains("base system"));
        assert!(text.contains("history summary"));
        assert!(text.contains("working memory"));
        assert_eq!(normalized[1].role, "user");
        assert_eq!(normalized[2].role, "assistant");
    }

    #[test]
    fn normalize_messages_prioritizes_working_memory_before_summary_and_self_note() {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: Value::String("base system".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
                content: Value::String("self_note:\nremember style".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
                content: Value::String(
                    "对话摘要（自动压缩，以下为早期对话要点）：\nolder summary".to_string(),
                ),
                tool_calls: None,
                tool_call_id: None,
            },
            Message {
                role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
                content: Value::String(
                    "Current code-inspection working memory:\n- use code_search first".to_string(),
                ),
                tool_calls: None,
                tool_call_id: None,
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
    fn openai_image_content_uses_object_image_url_shape() {
        let path = std::env::temp_dir().join(format!("ai-openai-image-{}.png", uuid::Uuid::new_v4()));
        std::fs::write(&path, b"fake").unwrap();

        let value = build_content(
            "gpt-4o",
            "describe",
            &[path.to_string_lossy().to_string()],
        )
        .unwrap();

        let first = value.as_array().and_then(|items| items.first()).unwrap();
        assert_eq!(first.get("type").and_then(|v| v.as_str()), Some("image_url"));
        assert!(first
            .get("image_url")
            .and_then(|v| v.get("url"))
            .and_then(|v| v.as_str())
            .map(|s| s.starts_with("data:image/png;base64,"))
            .unwrap_or(false));
    }
}
