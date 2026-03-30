use std::fmt;
use std::fs;
use std::time::Duration;

use base64::Engine as _;
use colored::Colorize;
use reqwest::{StatusCode, blocking::Response};
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{files, history::Message, models, skills::SkillManifest, types::App};
use crate::common::configw;

#[derive(Debug, Serialize)]
struct RequestBody {
    model: String,
    messages: Vec<Message>,
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

const REQUEST_MAX_ATTEMPTS: usize = 3;
const REQUEST_MAX_ATTEMPTS_429: usize = 6; // 429 错误重试 6 次
const REQUEST_RETRY_BASE_MS: u64 = 500;
const REQUEST_RETRY_MAX_MS: u64 = 4000;

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

pub(super) fn is_transient_error(err: &RequestError) -> bool {
    match err.kind {
        RequestErrorKind::Network => true,
        RequestErrorKind::Status(status) => should_retry_status(status),
    }
}

pub(super) fn do_request_messages(
    app: &mut App,
    model: &str,
    messages: Vec<Message>,
    stream: bool,
) -> Result<Response, RequestError> {
    let (tools_value, tool_choice) = agent_tools_for_request(app, model);
    let request_body = RequestBody {
        model: model.to_string(),
        messages,
        stream,
        enable_thinking: app.cli.thinking || models::enable_thinking(model),
        enable_search: models::search_enabled(model).then_some(true),
        tools: tools_value,
        tool_choice,
    };

    for attempt in 1..=REQUEST_MAX_ATTEMPTS_429 {
        let response = app
            .client
            .post(&app.config.endpoint)
            .bearer_auth(&app.config.api_key)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send();

        match response {
            Ok(response) => {
                if response.status().is_success() {
                    return Ok(response);
                }
                let status = response.status();
                let body = response.text().unwrap_or_default();
                let err = RequestError::status(status, body);

                // 根据状态码确定最大重试次数
                let max_attempts_for_status = if status.as_u16() == 429 {
                    REQUEST_MAX_ATTEMPTS_429
                } else {
                    REQUEST_MAX_ATTEMPTS
                };

                if should_retry_status(status) && attempt < max_attempts_for_status {
                    // 打印 sleep 原因
                    let delay = retry_delay(attempt);

                    if status.as_u16() == 429 {
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
                    std::thread::sleep(delay);
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
                    std::thread::sleep(delay);
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

pub(super) fn select_skill_via_model(
    app: &mut App,
    model: &str,
    question: &str,
    skills: &[SkillManifest],
) -> Option<String> {
    if question.trim().is_empty() {
        return None;
    }
    if skills.is_empty() {
        return None;
    }

    let mut lines = Vec::new();
    lines.push("You are a skill router for a CODE development assistant. Your ONLY job is to route CODE-related questions to appropriate skills.".to_string());
    lines.push("Output schema: {\"skill\":\"<exact skill name or empty>\",\"confidence\":0.0}".to_string());
    lines.push("Critical Rules:".to_string());
    lines.push("- ALL skills in this system are for CODE only. If the question is NOT about programming/coding, ALWAYS return empty skill.".to_string());
    lines.push("- Check: Is the user asking about SOURCE CODE files (.rs, .py, .js, .java, .go, etc.)? If NO → empty skill.".to_string());
    lines.push("- Common NON-code topics that should get EMPTY skill: news, sports, weather, stocks, NBA, music, movies, general knowledge, data analysis, business questions.".to_string());
    lines.push("- Word traps: '看一下'、'检查'、'分析'、'审查' in Chinese do NOT mean code review unless explicitly about CODE.".to_string());
    lines.push("- If unsure, return empty skill with confidence < 0.5. Better to skip than misroute.".to_string());
    lines.push("- Use EXACT skill name from the list below.".to_string());
    lines.push("- Return ONLY valid JSON, no explanations.".to_string());
    lines.push("Examples:".to_string());
    lines.push("- User: '帮我看一下今天 nba 的比赛' → {\"skill\":\"\",\"confidence\":0.1} ❌ sports, not code".to_string());
    lines.push("- User: '分析一下这个函数的时间复杂度' → {\"skill\":\"\",\"confidence\":0.2} ❌ algorithm question, but no code provided".to_string());
    lines.push("- User: '帮我 review 这段 Rust 代码' → {\"skill\":\"code-review\",\"confidence\":0.95} ✅ explicitly about code".to_string());
    lines.push("- User: '这个编译错误怎么修复' → {\"skill\":\"debugger\",\"confidence\":0.9} ✅ compilation error".to_string());
    lines.push("- User: '优化一下这个函数的结构' → {\"skill\":\"refactor\",\"confidence\":0.85} ✅ code refactoring".to_string());
    lines.push("- User: '查询数据库中的用户数据' → {\"skill\":\"\",\"confidence\":0.1} ❌ data query, not code".to_string());
    lines.push("Skills:".to_string());

    for s in skills.iter().take(32) {
        let desc = if s.description.trim().is_empty() {
            "(no description)".to_string()
        } else {
            s.description.trim().to_string()
        };
        lines.push(format!("- {}: {}", s.name, desc));
    }

    let system_prompt = lines.join("\n");

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

    let request_body = RequestBody {
        model: model.to_string(),
        messages,
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
        .ok()?;

    if !response.status().is_success() {
        return None;
    }

    let text = response.text().ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = extract_router_content(&v).unwrap_or_default();
    let (name, confidence) = parse_router_output(&content);
    let cfg = configw::get_all_config();
    let threshold = cfg
        .get_opt("ai.skills.router_threshold")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.7);
    if confidence >= threshold { name } else { None }
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
