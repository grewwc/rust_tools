//! Thinking-mode resolution logic.
//!
//! Decides whether a single request enables the model's thinking/reasoning mode:
//! - config force switch
//! - local heuristic short-circuit (QuestionShape)
//! - auxiliary model gate (`decide_thinking_via_model`)

use std::borrow::Cow;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::ai::config_schema::AiConfig;
use crate::ai::history::{Message, is_runtime_synthetic_user_message};
use crate::ai::models;
use crate::ai::provider::ThinkingOffCapability;
use crate::ai::types::App;
use crate::commonw::configw;
use rust_tools::commonw;

use super::builder::build_request_body;
use super::error::{
    DEFAULT_AUTO_THINKING_THRESHOLD, api_key_for_request_model, apply_request_auth,
    config_bool_is_true, control_model_for_aux_tasks, endpoint_for_request_model,
};
use super::reasoning::model_thinking_off_capability;
use super::routing::{extract_router_content, strip_json_fence};

/// Resolve whether to enable thinking mode for this request.
///
/// Decision order:
/// 1. Truncation fallback (`thinking_disabled_override`) forces thinking off
/// 2. Explicit user "effort off" (`/effort off`) turns thinking off on dialects
///    whose declared off capability is [`ThinkingOffCapability::RealSwitch`]
///    (DashScope `enable_thinking` / DeepSeek `thinking`)
/// 3. Config `ai.model.thinking=true` forces thinking when the model supports it
/// 4. If model doesn't support thinking, return false
/// 5. If auto-thinking is disabled by config, return false
/// 6. Models with a registry default reasoning effort always think — the
///    short-circuit that bypasses both the local heuristic and the model gate
/// 7. Auto-detect based on question complexity
#[commonw::debug_measure_time("resolve_thinking")]
pub(super) async fn resolve_thinking(app: &App, model: &str, messages: &[Message]) -> bool {
    // Truncation-retry fallback: after repeated truncation, thinking is forcibly
    // disabled with priority over everything (including force_thinking), leaving
    // the whole output budget to visible content. Restored by the orchestrator
    // at the end of the turn.
    if app.cli.thinking_disabled_override {
        return false;
    }

    // Explicit user "effort off" (`/effort off`, `/model effort off`,
    // `--reasoning-effort off`) also turns thinking off on dialects whose
    // declared off capability is a real wire off-switch (DashScope
    // `enable_thinking:false`, DeepSeek `thinking:{"type":"disabled"}`) — same
    // precedence tier as the truncation fallback above: a recent explicit
    // in-session command wins over the persistent `ai.model.thinking=true`
    // config. Effort-only dialects (capability [`ThinkingOffCapability::EffortNone`])
    // are not handled here: `resolve_reasoning_effort` expresses their off as
    // `reasoning_effort: "none"` instead. Unsupported dialects keep thinking on
    // and the `/effort off` handler reports that honestly.
    if app.cli.reasoning_effort_override == Some(None)
        && matches!(
            model_thinking_off_capability(model),
            ThinkingOffCapability::RealSwitch
        )
    {
        return false;
    }

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

    if models::default_reasoning_effort(model).is_some() {
        return true;
    }

    let raw_question = latest_user_message_text(messages).unwrap_or_default();
    // Injected `<system-reminder>...</system-reminder>` context is prepended to
    // the current user message (see prepare.rs / skill_runtime.rs). It can
    // inflate a one-word "hi" into a multi-thousand-character blob, defeating
    // the local thinking short-circuit (which keys off question length) and
    // falling through to the multi-second model gate. These reminder blocks are
    // stripped here before judging, using only what the user actually typed for
    // intent and thinking decisions.
    let question = strip_system_reminders(&raw_question);
    let question = question.trim();
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
            return local_decision;
        }
    }

    // Model-only decision path: if gate fails/uncertain, default to disabled.
    decide_thinking_via_model(app, model, messages)
        .await
        .unwrap_or(false)
}

pub(crate) fn latest_user_message_text(messages: &[Message]) -> Option<String> {
    // Skip synthetic user messages (task-evidence handoff, image followup, etc.):
    // they are injected after the real user message, and picking one up here
    // would take the handoff text instead of the user's actual question,
    // polluting both the thinking short-circuit and the model-gate decision.
    messages
        .iter()
        .rev()
        .find(|message| message.role == "user" && !is_runtime_synthetic_user_message(message))
        .and_then(extract_message_text)
}

/// Strips `<system-reminder>...</system-reminder>` blocks injected into user
/// messages.
///
/// prepare.rs / skill_runtime.rs prepend context reminders to the current user
/// message (to preserve prompt cache). These blocks are large and pollute the
/// intent/thinking judgment input, making a one-word "hi" look like long text.
/// They are removed before judgment, leaving only the user's actual input.
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
            // No closing tag: discard the rest (treated as an unterminated reminder).
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
    // Structured diagnostic traces: later lines match `label: details` /
    // stack-path shapes, without depending on specific error keywords.
    // QuestionShape does not cover this dimension, so it is computed inline
    // and passed in.
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

    // Always return a decision instead of None, so resolve_thinking never falls
    // through to the multi-second model gate. The middle ground (long but
    // unstructured) leans false (fast); complex inputs are already reliably
    // judged true above.
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
    // The thinking gate only needs the real user question, not the
    // cache-preservation context reminder; including it would waste the
    // auxiliary model's tokens.
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

    let gate_messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(include_str!("prompts/complexity_gate.md").to_string()),
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
    let mut request_body = build_request_body(
        &control_model,
        &gate_messages,
        false,
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );

    let endpoint = endpoint_for_request_model(app, &control_model);
    let api_key = api_key_for_request_model(app, &control_model);
    let http_body =
        super::protocol::build_http_body_for_request(&control_model, &endpoint, &mut request_body);
    // Auxiliary request (thinking gate), 15s timeout fallback: the main client
    // has no overall timeout, and connect_timeout alone does not cover the
    // "connected but the server never sends response headers" permanent block.
    let send_future =
        apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key, &app.session_id)
            .header("Content-Type", "application/json")
            .body(http_body)
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

    if confidence >= threshold {
        Some(thinking)
    } else {
        None
    }
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
