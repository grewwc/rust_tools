use serde_json::Value;

use crate::ai::{history::Message, request::{self, build_content}, types::App};
use crate::commonw::configw;
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use rust_tools::cw::SkipSet;
use chrono::Local;

pub(super) async fn maybe_append_self_reflection(
    app: &mut App,
    model: &str,
    question: &str,
    answer: &str,
    turn_messages: &mut Vec<Message>,
) {
    let cfg = configw::get_all_config();
    let enabled = !cfg
        .get_opt("ai.reflection.enable")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if !enabled {
        return;
    }
    if question.trim().is_empty() || answer.trim().is_empty() {
        return;
    }

    let system = "You are an introspective meta-optimizer for a coding assistant. Produce a brief self note to improve future turns.\nRules:\n- Output 2-6 compact bullets grouped under 'Do:' and 'Avoid:' tuned to the given Q&A.\n- Focus on planning, tool usage, argument hygiene, and verification habits.\n- No apologies, no explanations, no markdown code fences.\n- Keep under 800 chars.";
    let user_payload = format!(
        "question:\n{}\n\nanswer:\n{}",
        question.trim(),
        answer.trim()
    );

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(system.to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".to_string(),
            content: build_content(model, &user_payload, &[]).unwrap_or(Value::String(
                user_payload.clone(),
            )),
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    let saved_tools = app
        .agent_context
        .as_mut()
        .map(|ctx| std::mem::take(&mut ctx.tools));

    let resp = request::do_request_messages(app, model, &messages, false).await;

    if let Some(mut tools) = saved_tools {
        if let Some(ctx) = app.agent_context.as_mut() {
            std::mem::swap(&mut ctx.tools, &mut tools);
        }
    }

    let Ok(response) = resp else {
        return;
    };
    let Ok(text) = response.text().await else {
        return;
    };
    let Ok(v) = serde_json::from_str::<Value>(&text) else {
        return;
    };
    let content = extract_content(&v).unwrap_or_default();
    let note = content.trim();
    if note.is_empty() {
        return;
    }
    let record = Message {
        role: "system".to_string(),
        content: Value::String(format!("self_note:\n{}", note)),
        tool_calls: None,
        tool_call_id: None,
    };
    turn_messages.push(record);

    let entry = AgentMemoryEntry {
        timestamp: Local::now().to_rfc3339(),
        category: "self_note".to_string(),
        note: note.to_string(),
        tags: vec!["agent".to_string(), "policy".to_string()],
        source: Some(format!("session:{}", app.session_id)),
    };
    let _ = MemoryStore::from_env_or_config().append(&entry);
}

fn extract_content(v: &Value) -> Option<String> {
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

pub(super) fn build_persistent_guidelines(question: &str, max_chars: usize) -> Option<String> {
    let store = MemoryStore::from_env_or_config();
    // 搜索更相关的 self_note，若未命中则回退 recent
    let entries = store.search(question, 120).ok().filter(|v| !v.is_empty())
        .or_else(|| store.recent(120).ok())
        .unwrap_or_default();
    let cfg = configw::get_all_config();
    let max_days: i64 = cfg.get_opt("ai.memory.guidelines_days")
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(30);
    let mut seen: SkipSet<String> = SkipSet::new(16);
    let mut selected: Vec<String> = Vec::new();
    for e in entries.into_iter().rev() {
        if e.category.to_lowercase() != "self_note" {
            continue;
        }
        let note = e.note.trim().to_string();
        if note.is_empty() {
            continue;
        }
        if max_days > 0 {
            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&e.timestamp) {
                let age_days = (chrono::Utc::now() - dt.with_timezone(&chrono::Utc)).num_seconds().max(0) as i64 / 86400;
                if age_days > max_days {
                    continue;
                }
            }
        }
        if seen.insert(note.clone()) {
            selected.push(note);
        }
    }
    if selected.is_empty() {
        return None;
    }
    selected.reverse();
    let mut out = String::from("Persistent Guidelines:\n");
    let mut used = out.len();
    for note in selected {
        for line in note.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let bullet = if line.starts_with('-') { format!("{line}\n") } else { format!("- {line}\n") };
            if used + bullet.len() > max_chars {
                break;
            }
            out.push_str(&bullet);
            used += bullet.len();
        }
        if used >= max_chars {
            break;
        }
    }
    Some(out)
}

pub(super) async fn maybe_critic_and_revise(
    app: &mut App,
    model: &str,
    question: &str,
    draft: &str,
) -> Option<(String, String)> {
    let cfg = configw::get_all_config();
    let enabled = !cfg
        .get_opt("ai.critic_revise.enable")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if !enabled || question.trim().is_empty() || draft.trim().is_empty() {
        return None;
    }
    // 禁用工具
    let saved_tools = app
        .agent_context
        .as_mut()
        .map(|ctx| std::mem::take(&mut ctx.tools));
    let critic_system = "You are a strict code assistant critic. Review the DRAFT answer for the user QUESTION.\nReturn a compact list of 3-8 actionable points focused on:\n- factual correctness and missing steps\n- tool usage and argument hygiene\n- clarity and structure of final message\nNo markdown fences. Use short bullets.";
    let critic_user = format!("QUESTION:\n{}\n\nDRAFT:\n{}", question.trim(), draft.trim());
    let critic_req = vec![
        Message { role: "system".to_string(), content: Value::String(critic_system.to_string()), tool_calls: None, tool_call_id: None },
        Message { role: "user".to_string(), content: build_content(model, &critic_user, &[]).unwrap_or(Value::String(critic_user.clone())), tool_calls: None, tool_call_id: None },
    ];
    let critic_resp = request::do_request_messages(app, model, &critic_req, false).await.ok()?;
    let critic_text = critic_resp.text().await.ok()?;
    let critic_v: Value = serde_json::from_str(&critic_text).ok()?;
    let critic = extract_content(&critic_v).unwrap_or_default();
    if critic.trim().is_empty() {
        // 恢复工具
        if let Some(mut tools) = saved_tools {
            if let Some(ctx) = app.agent_context.as_mut() {
                std::mem::swap(&mut ctx.tools, &mut tools);
            }
        }
        return None;
    }
    let revise_system = "You are a senior coding assistant. Rewrite the final answer for the QUESTION using the CRITIC points.\nRules:\n- Fix issues; add missing steps; keep answers concise and correct.\n- If code is needed, use proper markdown fences.\n- Do not mention the critic itself.";
    let revise_user = format!("QUESTION:\n{}\n\nCRITIC:\n{}\n\nDRAFT:\n{}", question.trim(), critic.trim(), draft.trim());
    let revise_req = vec![
        Message { role: "system".to_string(), content: Value::String(revise_system.to_string()), tool_calls: None, tool_call_id: None },
        Message { role: "user".to_string(), content: build_content(model, &revise_user, &[]).unwrap_or(Value::String(revise_user.clone())), tool_calls: None, tool_call_id: None },
    ];
    let revised_resp = request::do_request_messages(app, model, &revise_req, false).await.ok()?;
    // 恢复工具
    if let Some(mut tools) = saved_tools {
        if let Some(ctx) = app.agent_context.as_mut() {
            std::mem::swap(&mut ctx.tools, &mut tools);
        }
    }
    let revised_text = revised_resp.text().await.ok()?;
    let revised_v: Value = serde_json::from_str(&revised_text).ok()?;
    let revised = extract_content(&revised_v).unwrap_or_default();
    if revised.trim().is_empty() {
        None
    } else {
        Some((critic, revised))
    }
}
