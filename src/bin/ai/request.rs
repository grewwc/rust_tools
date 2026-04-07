use std::fmt;
use std::fs;
use std::time::Duration;

use base64::Engine as _;
use colored::Colorize;
use reqwest::{Response, StatusCode};
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{files, history::Message, models, skills::SkillManifest, types::App};
use crate::ai::config_schema::AiConfig;
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

/// Resolve whether to enable thinking mode for this request.
///
/// Decision order:
/// 1. CLI `--thinking` flag always wins
/// 2. If model doesn't support thinking, return false
/// 3. If auto-thinking is disabled by config, return false
/// 4. Auto-detect based on question complexity
fn resolve_thinking(app: &App, model: &str, messages: &[Message]) -> bool {
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

    // Auto-detect based on question complexity
    should_enable_thinking_auto(messages)
}

/// Automatically decide whether to enable thinking based on question complexity.
///
/// Scoring factors:
/// - Question length (longer = more complex)
/// - Code-related keywords
/// - Reasoning/analysis indicators
/// - Multi-step task indicators
/// - Current agent type (plan mode favors thinking)
fn should_enable_thinking_auto(messages: &[Message]) -> bool {
    let mut score = 0;

    // Extract user question text from messages
    let user_text: String = messages
        .iter()
        .filter(|m| m.role == "user")
        .filter_map(|m| extract_message_text(m))
        .collect::<Vec<_>>()
        .join(" ");

    let text = user_text.to_lowercase();
    let char_count = text.chars().count();

    // Factor 1: Question length
    if char_count > 200 {
        score += 3;
    } else if char_count > 100 {
        score += 2;
    } else if char_count > 50 {
        score += 1;
    }

    // Factor 2: Code-related keywords (strong signal)
    let code_keywords = &[
        "implement", "实现", "重构", "refactor", "debug", "调试",
        "optimize", "优化", "架构", "architecture", "design pattern",
        "并发", "concurrent", "async", "异步", "内存", "memory",
        "性能", "performance", "algorithm", "算法", "数据结构",
        "trait", "lifetime", "borrow", "ownership", "泛型",
        "macro", "宏", "unsafe", "ffi", "序列化",
    ];
    for kw in code_keywords {
        if text.contains(kw) {
            score += 2;
        }
    }

    // Factor 3: Reasoning/analysis indicators
    let reasoning_keywords = &[
        "为什么", "分析", "explain", "why", "how does", "原理",
        "比较", "compare", "区别", "difference", "优劣",
        "pros and cons", "tradeoff", "权衡", "深入", "detailed",
        "详细", "完整", "comprehensive", "系统",
    ];
    for kw in reasoning_keywords {
        if text.contains(kw) {
            score += 2;
        }
    }

    // Factor 4: Multi-step task indicators
    let multistep_keywords = &[
        "第一步", "第二步", "然后", "接着", "步骤",
        "step 1", "step 2", "first", "then", "finally",
        "流程", "workflow", "pipeline", "完整流程",
    ];
    for kw in multistep_keywords {
        if text.contains(kw) {
            score += 2;
        }
    }

    // Factor 5: Simple chat indicators (negative signal)
    let chat_keywords = &[
        "你好", "hello", "hi ", "hey", "谢谢", "thanks",
        "再见", "bye", "天气", "weather", "新闻", "news",
        "你是谁", "who are you", "你叫什么",
    ];
    for kw in chat_keywords {
        if text.contains(kw) {
            score -= 3;
        }
    }

    // Threshold: score >= 3 enables thinking
    score >= 3
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

pub(super) async fn do_request_messages(
    app: &mut App,
    model: &str,
    messages: &[Message],
    stream: bool,
) -> Result<Response, RequestError> {
    let (tools_value, tool_choice) = agent_tools_for_request(app, model);
    let enable_thinking = resolve_thinking(app, model, messages);
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

pub(super) async fn select_skill_via_model(
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

    let request_body = RequestBody {
        model,
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
    use serde_json::json;

    fn make_user_message(text: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: json!(text),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn test_auto_thinking_simple_chat() {
        let messages = vec![make_user_message("你好")];
        assert!(!should_enable_thinking_auto(&messages));
    }

    #[test]
    fn test_auto_thinking_greeting() {
        let messages = vec![make_user_message("hello, how are you?")];
        assert!(!should_enable_thinking_auto(&messages));
    }

    #[test]
    fn test_auto_thinking_code_question() {
        let messages = vec![make_user_message(
            "帮我实现一个 Rust 的异步并发数据结构，需要支持多线程安全的内存管理",
        )];
        assert!(should_enable_thinking_auto(&messages));
    }

    #[test]
    fn test_auto_thinking_long_analysis() {
        let messages = vec![make_user_message(
            "请详细分析 Rust 中 borrow checker 的工作原理，并比较与其他语言内存管理的优劣",
        )];
        assert!(should_enable_thinking_auto(&messages));
    }

    #[test]
    fn test_auto_thinking_multistep() {
        let messages = vec![make_user_message(
            "第一步读取文件，然后解析 JSON，最后写入数据库，完整流程是怎样的",
        )];
        assert!(should_enable_thinking_auto(&messages));
    }

    #[test]
    fn test_auto_thinking_short_technical() {
        let messages = vec![make_user_message("Rust 的 trait 和 Go 的 interface 有什么区别？")];
        // Reasoning keyword "区别" + code keywords "trait", "interface"
        assert!(should_enable_thinking_auto(&messages));
    }

    #[test]
    fn test_auto_thinking_empty_messages() {
        let messages: Vec<Message> = vec![];
        assert!(!should_enable_thinking_auto(&messages));
    }

    #[test]
    fn test_auto_thinking_mixed_messages() {
        let messages = vec![
            make_user_message("帮我看看这段代码"),
            make_user_message("需要重构和性能优化"),
        ];
        assert!(should_enable_thinking_auto(&messages));
    }
}

