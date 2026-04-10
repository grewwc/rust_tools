use std::path::PathBuf;

use chrono::Local;
use serde_json::{Value, json};

use crate::ai::history::{Message, append_history_messages};
use crate::ai::request::{self, build_content};
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use crate::ai::types::{App, ToolDefinition};
use crate::commonw::configw;

use super::gates::{
    critic_filtered, model_should_revise, parse_reflect_flag, reflection_filtered_bg,
    turn_has_tool,
};

pub(crate) async fn maybe_append_self_reflection(
    app: &mut App,
    model: &str,
    question: &str,
    answer: &str,
    turn_messages: &mut Vec<Message>,
) {
    let q = question.trim();
    let a = answer.trim();
    if q.is_empty() || a.is_empty() {
        return;
    }
    let had_tool = turn_has_tool(turn_messages);
    let history_path = app.session_history_file.clone();
    let session_id = app.session_id.clone();
    let model_s = model.to_string();
    let q_s = q.to_string();
    let a_s = a.to_string();
    tokio::spawn(async move {
        run_self_reflection_background(history_path, session_id, model_s, q_s, a_s, had_tool)
            .await;
    });
}

pub(crate) async fn maybe_critic_and_revise(
    app: &mut App,
    model: &str,
    question: &str,
    draft: &str,
) -> Option<(String, String)> {
    use tokio::time::{Duration, timeout};
    let cfg = configw::get_all_config();
    let enabled = !cfg
        .get_opt("ai.critic_revise.enable")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if !enabled || question.trim().is_empty() || draft.trim().is_empty() {
        return None;
    }
    if critic_filtered(question, draft) {
        return None;
    }
    let to_ms = cfg
        .get_opt("ai.critic_revise.timeout_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(7000);
    let only_for_code = !cfg
        .get_opt("ai.critic_revise.only_for_code")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if only_for_code {
        let gate_fut = model_should_revise(app, model, question, draft);
        let should = match timeout(Duration::from_millis(to_ms / 2), gate_fut).await {
            Ok(v) => v.unwrap_or(false),
            Err(_) => false,
        };
        if !should {
            return None;
        }
    }
    let saved_tools: Option<Vec<ToolDefinition>> = app
        .agent_context
        .as_mut()
        .map(|ctx| std::mem::replace(&mut ctx.tools, Vec::new()));
    let critic_system = "You are a strict code assistant critic. Review the DRAFT answer for the user QUESTION.\nReturn a compact list of 3-8 actionable points focused on:\n- factual correctness and missing steps\n- tool usage and argument hygiene\n- clarity and structure of final message\nNo markdown fences. Use short bullets.";
    let critic_user = format!("QUESTION:\n{}\n\nDRAFT:\n{}", question.trim(), draft.trim());
    let critic_req = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(critic_system.to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".to_string(),
            content: build_content(model, &critic_user, &[])
                .unwrap_or(Value::String(critic_user.clone())),
            tool_calls: None,
            tool_call_id: None,
        },
    ];
    let critic_fut = request::do_request_messages(app, model, &critic_req, false);
    let critic_resp = match timeout(Duration::from_millis(to_ms), critic_fut).await {
        Ok(Ok(r)) => r,
        _ => {
            restore_tools(app, saved_tools);
            return None;
        }
    };
    let critic_text = critic_resp.text().await.ok()?;
    let critic_v: Value = serde_json::from_str(&critic_text).ok()?;
    let critic = extract_content(&critic_v).unwrap_or_default();
    if critic.trim().is_empty() {
        restore_tools(app, saved_tools);
        return None;
    }
    let revise_system = "You are a senior coding assistant. Rewrite the final answer for the QUESTION using the CRITIC points.\nRules:\n- Fix issues; add missing steps; keep answers concise and correct.\n- If code is needed, use proper markdown fences.\n- Do not mention the critic itself.";
    let revise_user = format!(
        "QUESTION:\n{}\n\nCRITIC:\n{}\n\nDRAFT:\n{}",
        question.trim(),
        critic.trim(),
        draft.trim()
    );
    let revise_req = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(revise_system.to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".to_string(),
            content: build_content(model, &revise_user, &[])
                .unwrap_or(Value::String(revise_user.clone())),
            tool_calls: None,
            tool_call_id: None,
        },
    ];
    let revise_fut = request::do_request_messages(app, model, &revise_req, false);
    let revised_resp = match timeout(Duration::from_millis(to_ms), revise_fut).await {
        Ok(Ok(r)) => r,
        _ => {
            restore_tools(app, saved_tools);
            return None;
        }
    };
    restore_tools(app, saved_tools);
    let revised_text = revised_resp.text().await.ok()?;
    let revised_v: Value = serde_json::from_str(&revised_text).ok()?;
    let revised = extract_content(&revised_v).unwrap_or_default();
    if revised.trim().is_empty() {
        None
    } else {
        Some((critic, revised))
    }
}

pub(crate) async fn run_critic_revise_background(
    history_path: PathBuf,
    model: String,
    question: String,
    draft: String,
) {
    use tokio::time::{Duration, timeout};
    let cfg = configw::get_all_config();
    let to_ms = cfg
        .get_opt("ai.critic_revise.timeout_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(7000);
    let system_c = "You are a strict code assistant critic. Review the DRAFT answer for the user QUESTION.\nReturn a compact list of 3-8 actionable points focused on:\n- factual correctness and missing steps\n- tool usage and argument hygiene\n- clarity and structure of final message\nNo markdown fences. Use short bullets.";
    let critic_user = format!("QUESTION:\n{}\n\nDRAFT:\n{}", question.trim(), draft.trim());
    let messages_c = vec![
        json!({"role":"system","content":system_c}),
        json!({"role":"user","content":critic_user}),
    ];
    let resp_c = match background_call(&model, &messages_c).await {
        Some(v) => v,
        None => return,
    };
    let content_c = extract_back_content(&resp_c).unwrap_or_default();
    if content_c.trim().is_empty() {
        return;
    }
    let system_r = "You are a senior coding assistant. Rewrite the final answer for the QUESTION using the CRITIC points.\nRules:\n- Fix issues; add missing steps; keep answers concise and correct.\n- If code is needed, use proper markdown fences.\n- Do not mention the critic itself.";
    let revise_user = format!(
        "QUESTION:\n{}\n\nCRITIC:\n{}\n\nDRAFT:\n{}",
        question.trim(),
        content_c.trim(),
        draft.trim()
    );
    let messages_r = vec![
        json!({"role":"system","content":system_r}),
        json!({"role":"user","content":revise_user}),
    ];
    let resp_r = match timeout(
        Duration::from_millis(to_ms),
        background_call(&model, &messages_r),
    )
    .await
    {
        Ok(v) => v.and_then(Some),
        Err(_) => None,
    };
    let Some(resp_r) = resp_r else {
        return;
    };
    let content_r = extract_back_content(&resp_r).unwrap_or_default();
    if content_r.trim().is_empty() {
        return;
    }
    let record = Message {
        role: "system".to_string(),
        content: Value::String(format!(
            "critic:\n{}\n\nrevised:\n{}",
            content_c.trim(),
            content_r.trim()
        )),
        tool_calls: None,
        tool_call_id: None,
    };
    let _ = append_history_messages(&history_path, &[record]);
}

pub(crate) async fn run_self_reflection_background(
    history_path: PathBuf,
    session_id: String,
    model: String,
    question: String,
    answer: String,
    had_tool: bool,
) {
    use tokio::time::{Duration, timeout};
    let cfg = configw::get_all_config();
    let enabled = !cfg
        .get_opt("ai.reflection.enable")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if !enabled {
        return;
    }
    let q = question.trim();
    let a = answer.trim();
    if q.is_empty() || a.is_empty() {
        return;
    }
    let model_gate_enabled = !cfg
        .get_opt("ai.reflection.model_gate.enable")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if model_gate_enabled {
        let to_ms = cfg
            .get_opt("ai.reflection.model_gate.timeout_ms")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(2000);
        let fut = background_model_should_reflect(&model, q, a, had_tool);
        let should = match timeout(Duration::from_millis(to_ms), fut).await {
            Ok(v) => v.unwrap_or(false),
            Err(_) => false,
        };
        if !should {
            return;
        }
    } else if reflection_filtered_bg(q, a, had_tool) {
        return;
    }
    let system = "You are an introspective meta-optimizer for a coding assistant. Produce a brief self note to improve future turns.\nRules:\n- Output 2-6 compact bullets grouped under 'Do:' and 'Avoid:' tuned to the given Q&A.\n- Focus on planning, tool usage, argument hygiene, and verification habits.\n- No apologies, no explanations, no markdown code fences.\n- Keep under 800 chars.";
    let user_payload = format!("question:\n{}\n\nanswer:\n{}", q, a);
    let messages = vec![
        json!({"role":"system","content":system}),
        json!({"role":"user","content":user_payload}),
    ];
    let to_ms_note = cfg
        .get_opt("ai.reflection.timeout_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(3000);
    let resp = match timeout(
        Duration::from_millis(to_ms_note),
        background_call(&model, &messages),
    )
    .await
    {
        Ok(v) => v,
        Err(_) => None,
    };
    let Some(resp) = resp else {
        return;
    };
    let content = extract_back_content(&resp).unwrap_or_default();
    let note = content.trim();
    if note.is_empty() {
        return;
    }
    let record = Message {
        role: crate::ai::history::ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(format!("self_note:\n{}", note)),
        tool_calls: None,
        tool_call_id: None,
    };
    let _ = append_history_messages(&history_path, &[record]);
    let entry = AgentMemoryEntry {
        id: None,
        timestamp: Local::now().to_rfc3339(),
        category: "self_note".to_string(),
        note: note.to_string(),
        tags: vec!["agent".to_string(), "policy".to_string()],
        source: Some(format!("session:{}", session_id)),
        priority: Some(255),
    };
    let store = MemoryStore::from_env_or_config();
    let _ = store.append(&entry);
    store.maintain_after_append();
}

pub(super) fn extract_content(v: &Value) -> Option<String> {
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

fn restore_tools(app: &mut App, saved_tools: Option<Vec<ToolDefinition>>) {
    if let Some(mut tools) = saved_tools
        && let Some(ctx) = app.agent_context.as_mut()
    {
        std::mem::swap(&mut ctx.tools, &mut tools);
    }
}

pub(super) async fn background_call(model: &str, messages: &Vec<Value>) -> Option<Value> {
    let cfg = configw::get_all_config();
    let endpoint =
        crate::ai::models::endpoint_for_model(model, &cfg.get_opt("ai.model.endpoint").unwrap_or_default());
    let api_key =
        crate::ai::models::api_key_for_model(model, &cfg.get_opt("api_key").unwrap_or_default());
    if api_key.trim().is_empty()
        && !crate::ai::models::endpoint_supports_anonymous_auth(&endpoint)
    {
        return None;
    }
    let body = match crate::ai::models::model_provider(model) {
        crate::ai::provider::ApiProvider::OpenAi => json!({
            "model": model,
            "messages": messages,
            "stream": false
        }),
        crate::ai::provider::ApiProvider::Compatible => json!({
            "model": model,
            "messages": messages,
            "stream": false,
            "enable_thinking": false
        }),
    };
    let client = reqwest::Client::new();
    let req = client.post(&endpoint);
    let req = if api_key.trim().is_empty()
        && crate::ai::models::endpoint_supports_anonymous_auth(&endpoint)
    {
        req
    } else {
        req.bearer_auth(api_key)
    };
    let resp = req
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .ok()?;
    let text = resp.text().await.ok()?;
    serde_json::from_str::<Value>(&text).ok()
}

pub(super) fn extract_back_content(v: &Value) -> Option<String> {
    let choices = v
        .get("choices")
        .or_else(|| v.get("output")?.get("choices"))?;
    let msg = choices.get(0)?.get("message")?;
    let content = msg.get("content")?;
    match content {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                if let Some(s) = p.get("text").and_then(|x| x.as_str()) {
                    out.push_str(s);
                }
            }
            Some(out)
        }
        _ => None,
    }
}

async fn background_model_should_reflect(
    model: &str,
    question: &str,
    answer: &str,
    had_tool: bool,
) -> Option<bool> {
    let system = "You are a binary classifier that decides whether to capture a short 'experience note' for future turns.\nReturn STRICT JSON ONLY with the shape: {\"reflect\": true|false}.\nRules:\n- reflect=true when Q/A contains non-trivial reasoning, code, multi-step instructions, tool usage outcomes, errors/diagnosis, or decisions that should guide future behavior.\n- reflect=false for greetings, acknowledgements, trivial answers, or very short single-turn exchanges with no actionable guidance.\nDo not include explanations or extra text.";
    let user = format!(
        "question:\n{}\n\nanswer:\n{}\n\nhad_tool:\n{}",
        question.trim(),
        answer.trim(),
        if had_tool { "true" } else { "false" }
    );
    let messages = vec![
        json!({"role":"system","content":system}),
        json!({"role":"user","content":user}),
    ];
    let resp = background_call(model, &messages).await?;
    let text = extract_back_content(&resp).unwrap_or_default();
    parse_reflect_flag(&text)
}
