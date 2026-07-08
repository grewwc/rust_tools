//! 思考模式（thinking mode）解析逻辑。
//!
//! 决定单次请求是否启用模型的思考/推理模式：
//! - 配置强制开关
//! - 本地启发式短路（QuestionShape）
//! - 辅助模型 gate（`decide_thinking_via_model`）

use std::borrow::Cow;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::ai::config_schema::AiConfig;
use crate::ai::history::Message;
use crate::ai::models;
use crate::ai::types::App;
use rust_tools::commonw;
use crate::commonw::configw;

use super::{
    build_request_body,
};
use super::error::{
    api_key_for_request_model, apply_request_auth, config_bool_is_true,
    control_model_for_aux_tasks, endpoint_for_request_model, DEFAULT_AUTO_THINKING_THRESHOLD,
};
use super::routing::{extract_router_content, strip_json_fence};

// #region debug-point A:resolve-thinking-reporter
fn report_resolve_thinking_debug(
    run_id: &'static str,
    hypothesis_id: &'static str,
    location: &'static str,
    msg: &'static str,
    data: Value,
) {
    static DEBUG_TARGET: std::sync::LazyLock<Option<(String, String)>> = std::sync::LazyLock::new(
        || {
            let env_text = std::fs::read_to_string(".dbg/resolve-thinking-slow.env").ok()?;
            let mut debug_server_url = "http://127.0.0.1:7777/event".to_string();
            let mut debug_session_id = "resolve-thinking-slow".to_string();
            for line in env_text.lines() {
                if let Some(value) = line.strip_prefix("DEBUG_SERVER_URL=") {
                    if !value.trim().is_empty() {
                        debug_server_url = value.trim().to_string();
                    }
                } else if let Some(value) = line.strip_prefix("DEBUG_SESSION_ID=")
                    && !value.trim().is_empty()
                {
                    debug_session_id = value.trim().to_string();
                }
            }
            Some((debug_server_url, debug_session_id))
        },
    );

    let Some((debug_server_url, debug_session_id)) = DEBUG_TARGET.as_ref().cloned() else {
        return;
    };

    std::thread::spawn(move || {
        let payload = serde_json::json!({
            "sessionId": debug_session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": msg,
            "data": data,
            "ts": chrono::Utc::now().timestamp_millis(),
        });
        if let Ok(client) = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_millis(300))
            .build()
        {
            let _ = client.post(debug_server_url).json(&payload).send();
        }
    });
}
// #endregion

/// Resolve whether to enable thinking mode for this request.
///
/// Decision order:
/// 1. Config `ai.model.thinking=true` forces thinking when the model supports it
/// 2. If model doesn't support thinking, return false
/// 3. If auto-thinking is disabled by config, return false
/// 4. Auto-detect based on question complexity
#[commonw::debug_measure_time("resolve_thinking")]
pub(super) async fn resolve_thinking(app: &App, model: &str, messages: &[Message]) -> bool {
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
    // #region debug-point D:resolve-thinking-input
    report_resolve_thinking_debug(
        "pre-fix",
        "D",
        "request::resolve_thinking:prepared_input",
        "[DEBUG] resolve_thinking prepared input",
        serde_json::json!({
            "raw_question_len": raw_question.chars().count(),
            "clean_question_len": question.chars().count(),
            "reminder_removed": raw_question.len() != question.len(),
            "message_count": messages.len(),
            "model": model,
        }),
    );
    // #endregion
    if !question.is_empty() {
        if let Some(local_decision) = local_thinking_decision(question) {
            crate::ai::agent_hang_debug!(
                "post-fix",
                "G",
                "request::resolve_thinking:local_decision",
                "[DEBUG] resolve thinking decided locally",
                {
                    "question_len": question.chars().count(),
                    "decision": local_decision,
                },
            );
            // #region debug-point A:local-decision
            report_resolve_thinking_debug(
                "pre-fix",
                "A",
                "request::resolve_thinking:local_decision",
                "[DEBUG] resolve_thinking decided locally",
                serde_json::json!({
                    "question_len": question.chars().count(),
                    "decision": local_decision,
                }),
            );
            // #endregion
            return local_decision;
        }
    }

    // Model-only decision path: if gate fails/uncertain, default to disabled.
    // #region debug-point B:gate-fallback
    report_resolve_thinking_debug(
        "pre-fix",
        "B",
        "request::resolve_thinking:gate_fallback",
        "[DEBUG] resolve_thinking fell back to model gate",
        serde_json::json!({
            "question_len": question.chars().count(),
            "model": model,
        }),
    );
    // #endregion
    decide_thinking_via_model(app, model, messages)
        .await
        .unwrap_or(false)
}

pub(crate) fn latest_user_message_text(messages: &[Message]) -> Option<String> {
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
pub(crate) fn strip_system_reminders(text: &str) -> Cow<'_, str> {
    const OPEN: &str = "<system-reminder>";
    const CLOSE: &str = "</system-reminder>";
    if !text.contains(OPEN) {
        return Cow::Borrowed(text);
    }
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
    Cow::Owned(out)
}

pub(crate) fn local_thinking_decision(question: &str) -> Option<bool> {
    let question = question.trim();
    if question.is_empty() {
        return Some(false);
    }

    let nonempty_lines: Vec<&str> = question
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    let line_count = nonempty_lines.len();
    // 结构化诊断痕迹：后续行出现 `label: details` / 堆栈路径样式，
    // 不依赖具体错误关键词。QuestionShape 不覆盖此维度，内联计算后传入。
    let has_diagnostic_shape = line_count >= 2
        && nonempty_lines.iter().skip(1).any(|line| {
            line.contains(": ")
                || line.contains(" at ")
                || line.contains("->")
                || line.contains("::")
                || line.contains('/')
                || line.contains('\\')
        });

    let shape = crate::ai::driver::turn_runtime::QuestionShape::analyze(question);
    if shape.needs_deliberate_thinking(has_diagnostic_shape) {
        return Some(true);
    }

    // 兜底恒给决策：不再返回 None，避免 resolve_thinking 落到耗时数秒的
    // 模型 gate。中间地带（长但无结构）倒向 false（快），复杂输入已在上面
    // 稳定判 true。
    Some(false)
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
    // thinking gate 只需要真实用户问题，不需要 cache-preservation 用的
    // context reminder；否则会白白烧掉辅助模型 token。
    let question = strip_system_reminders(&user_text);
    let question = question.trim();
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
    let clipped_len = clipped.chars().count();

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
    // #region debug-point C:gate-request-prep
    report_resolve_thinking_debug(
        "pre-fix",
        "C",
        "request::decide_thinking_via_model:request_prep",
        "[DEBUG] thinking gate request prepared",
        serde_json::json!({
            "control_model": control_model,
            "question_len": question.chars().count(),
            "clipped_len": clipped_len,
        }),
    );
    // #endregion
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
    let send_start = Instant::now();
    let send_future = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send();
    let response = match tokio::time::timeout(Duration::from_secs(15), send_future).await {
        Ok(r) => {
            let outcome = if r.is_ok() { "ok" } else { "err" };
            // #region debug-point B:gate-send
            report_resolve_thinking_debug(
                "pre-fix",
                "B",
                "request::decide_thinking_via_model:send",
                "[DEBUG] thinking gate send finished",
                serde_json::json!({
                    "elapsed_ms": send_start.elapsed().as_secs_f64() * 1000.0,
                    "outcome": outcome,
                    "endpoint": endpoint,
                }),
            );
            // #endregion
            r.ok()?
        }
        Err(_) => {
            // #region debug-point B:gate-send-timeout
            report_resolve_thinking_debug(
                "pre-fix",
                "B",
                "request::decide_thinking_via_model:send_timeout",
                "[DEBUG] thinking gate send timed out",
                serde_json::json!({
                    "elapsed_ms": send_start.elapsed().as_secs_f64() * 1000.0,
                    "endpoint": endpoint,
                }),
            );
            // #endregion
            return None;
        }
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

    let body_start = Instant::now();
    let text = match tokio::time::timeout(Duration::from_secs(15), response.text()).await {
        Ok(r) => {
            let outcome = if r.is_ok() { "ok" } else { "err" };
            // #region debug-point B:gate-body
            report_resolve_thinking_debug(
                "pre-fix",
                "B",
                "request::decide_thinking_via_model:body",
                "[DEBUG] thinking gate body read finished",
                serde_json::json!({
                    "elapsed_ms": body_start.elapsed().as_secs_f64() * 1000.0,
                    "outcome": outcome,
                }),
            );
            // #endregion
            r.ok()?
        }
        Err(_) => {
            // #region debug-point B:gate-body-timeout
            report_resolve_thinking_debug(
                "pre-fix",
                "B",
                "request::decide_thinking_via_model:body_timeout",
                "[DEBUG] thinking gate body read timed out",
                serde_json::json!({
                    "elapsed_ms": body_start.elapsed().as_secs_f64() * 1000.0,
                }),
            );
            // #endregion
            return None;
        }
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
    // #region debug-point B:gate-result
    report_resolve_thinking_debug(
        "pre-fix",
        "B",
        "request::decide_thinking_via_model:result",
        "[DEBUG] thinking gate parsed result",
        serde_json::json!({
            "elapsed_ms": gate_start.elapsed().as_secs_f64() * 1000.0,
            "thinking": thinking,
            "confidence": confidence,
            "threshold": threshold,
            "accepted": result.is_some(),
        }),
    );
    // #endregion
    result
}

pub(crate) fn parse_thinking_gate_output(s: &str) -> Option<(bool, f64)> {
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
