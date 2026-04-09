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

use super::{files, history::Message, models, skills::SkillManifest, types::App};
use crate::ai::config_schema::AiConfig;
use crate::ai::driver::intent_recognition;
use crate::commonw::configw;

#[derive(Debug, Serialize)]
struct RequestBody<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
    enable_thinking: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_search: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct StreamChunk {
    #[serde(default)]
    pub(super) choices: Vec<StreamChoice>,
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
    #[serde(default, deserialize_with = "string_or_default")]
    pub(super) content: String,
    #[serde(default, deserialize_with = "string_or_default")]
    pub(super) reasoning_content: String,
    #[serde(default)]
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

fn string_or_default<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    Ok(value.unwrap_or_default())
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

fn control_model_for_aux_tasks(app: &App) -> String {
    app.config
        .intent_model
        .as_deref()
        .filter(|v| !v.trim().is_empty())
        .map(models::determine_model)
        .unwrap_or_else(|| models::determine_model(DEFAULT_CONTROL_MODEL))
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
/// 2. If model doesn't support thinking, return false
/// 3. If auto-thinking is disabled by config, return false
/// 4. Auto-detect based on question complexity
#[commonw::debug_measure_time("resolve_thinking")]
async fn resolve_thinking(app: &App, model: &str, messages: &[Message]) -> bool {
    // CLI flag always wins
    if app.cli.thinking {
        return true;
    }

    // Model must support thinking
    if !models::enable_thinking(model) {
        return false;
    }

    // Check config for auto-thinking override
    let cfg = configw::get_all_config();
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
        let question_len = question.chars().count();
        let skip_thinking_gate = matches!(
            local_intent.core,
            intent_recognition::CoreIntent::Casual | intent_recognition::CoreIntent::QueryConcept
        ) && question_len <= 64;
        if skip_thinking_gate {
            crate::ai::agent_hang_debug!(
                "post-fix",
                "G",
                "request::resolve_thinking:local_skip",
                "[DEBUG] resolve thinking skipped by local intent",
                {
                    "core": format!("{:?}", local_intent.core),
                    "question_len": question_len,
                },
            );
            return false;
        }
    }

    // Model-only decision path: if gate fails/uncertain, default to disabled.
    decide_thinking_via_model(app, model, messages)
        .await
        .unwrap_or(false)
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
    let request_body = RequestBody {
        model: &control_model,
        messages: &gate_messages,
        stream: false,
        enable_thinking: false,
        enable_search: None,
        tools: None,
        tool_choice: None,
    };

    let response = app
        .client
        .post(&app.config.endpoint)
        .bearer_auth(&app.config.api_key)
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
    let (tools_value, tool_choice) = agent_tools_for_request(app, model);
    let thinking_start = Instant::now();
    let enable_thinking = resolve_thinking(app, model, messages).await;
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
    let request_body = RequestBody {
        model,
        messages,
        stream,
        enable_thinking,
        enable_search: models::search_enabled(model).then_some(true),
        tools: tools_value,
        tool_choice,
    };

    for attempt in 1..=REQUEST_MAX_ATTEMPTS_429 {
        let http_start = Instant::now();
        let response = app
            .client
            .post(&app.config.endpoint)
            .bearer_auth(&app.config.api_key)
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
                    tokio::time::sleep(delay).await;
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
                    tokio::time::sleep(delay).await;
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

    // Use raw multi-line string for better readability
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
Examples:
- User: '帮我看一下今天 nba 的比赛' → {"skill":"","confidence":0.1} ❌ sports, not code
- User: '分析一下这个函数的时间复杂度' → {"skill":"","confidence":0.2} ❌ algorithm question without code
- User: 'python flask 中 g.request_base_info_dict 这个是什么' → {"skill":"","confidence":0.1} ❌ knowledge question about Flask API, not a code task
- User: 'Django 的 ORM 怎么用' → {"skill":"","confidence":0.1} ❌ knowledge question about Django, not a code task
- User: 'React hooks 是什么' → {"skill":"","confidence":0.1} ❌ knowledge question about React, not a code task
- User: 'Rust 的 borrow checker 怎么工作的' → {"skill":"","confidence":0.1} ❌ knowledge question about Rust, not a code task
- User: '帮我 review 这段 Rust 代码' → {"skill":"code-review","confidence":0.95} ✅ explicitly about code
- User: '这个编译错误怎么修复：error[E0382]...' → {"skill":"debugger","confidence":0.9} ✅ compilation error with code
- User: '优化一下这个函数的结构：fn foo() { ... }' → {"skill":"refactor","confidence":0.85} ✅ code refactoring with code
- User: '查询数据库中的用户数据' → {"skill":"","confidence":0.1} ❌ data/business query, not code
- User: '帮我一下我的飞书文档：dataagent 项目架构相关' → {"skill":"","confidence":0.1} ❌ about cloud documents, not code
- User: '查看项目架构文档' → {"skill":"","confidence":0.1} ❌ documentation, not code
Skills:
"#.to_string();

    for s in skills.iter().take(32) {
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
    let request_body = RequestBody {
        model: &control_model,
        messages: &messages,
        stream: false,
        enable_thinking: false,
        enable_search: None,
        tools: None,
        tool_choice: None,
    };

    let response = app
        .client
        .post(&app.config.endpoint)
        .bearer_auth(&app.config.api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await
        .ok()?;

    if !response.status().is_success() {
        crate::ai::agent_hang_debug!(
            "pre-fix",
            "R",
            "request::select_skill_via_model:http_non_success",
            "[DEBUG] model skill router http non success",
            {
                "elapsed_ms": router_start.elapsed().as_secs_f64() * 1000.0,
            },
        );
        return None;
    }

    let text = response.text().await.ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = extract_router_content(&v).unwrap_or_default();
    let (name, confidence) = parse_router_output(&content);
    let cfg = configw::get_all_config();
    let threshold = cfg
        .get_opt("ai.skills.router_threshold")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.7);
    let selected = if confidence >= threshold { name } else { None };
    selected
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

pub(super) fn build_content(
    model: &str,
    question: &str,
    image_files: &[String],
) -> Result<Value, Box<dyn std::error::Error>> {
    if !models::is_vl_model(model) || image_files.is_empty() {
        return Ok(Value::String(question.to_string()));
    }

    let mut parts = Vec::new();
    for file in image_files {
        let bytes = fs::read(file)?;
        let mime = files::image_mime_type(file);
        let image = base64::engine::general_purpose::STANDARD.encode(bytes);
        parts.push(json!({
            "type": "image_url",
            "image_url": format!("data:{mime};base64,{image}"),
        }));
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
        let response = app
            .client
            .post(&app.config.endpoint)
            .bearer_auth(&app.config.api_key)
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
                    tokio::time::sleep(delay).await;
                    continue;
                }
                return Err(err.into());
            }
            Err(err) => {
                if is_retryable_reqwest_error(&err) && attempt < REQUEST_MAX_ATTEMPTS {
                    let delay = retry_delay(attempt);
                    tokio::time::sleep(delay).await;
                    continue;
                }
                return Err(err.into());
            }
        }
    }

    Err("request failed after all attempts".into())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
