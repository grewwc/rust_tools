use serde_json::Value;

use crate::ai::{history::Message, request::{self, build_content}, types::App};
use crate::commonw::configw;
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use crate::ai::history::append_history_messages;
use std::path::PathBuf;
use serde_json::json;
use rust_tools::cw::SkipSet;
use chrono::Local;

/// 反思触发条件 - 用于主动学习
#[derive(Debug, Clone, PartialEq)]
pub enum ReflectionTrigger {
    /// 工具调用失败
    ToolFailure,
    /// 模型回答置信度低
    LowConfidenceAnswer,
    /// 用户纠正
    UserCorrection,
    /// 重复问题（说明之前没解决）
    RepeatedQuestion,
    /// 超长对话轮次（>10 轮）
    LongTurn,
    /// 常规反思
    Routine,
}

/// 反思质量评估
#[derive(Debug, Clone)]
pub struct ReflectionQuality {
    /// 是否可执行
    pub actionable: bool,
    /// 是否具体
    pub specific: bool,
    /// 是否可推广
    pub generalizable: bool,
}

impl ReflectionQuality {
    pub fn score(&self) -> u8 {
        let mut score = 0;
        if self.actionable { score += 1; }
        if self.specific { score += 1; }
        if self.generalizable { score += 1; }
        score
    }
    
    pub fn is_high_quality(&self) -> bool {
        self.score() >= 2
    }
}

pub(super) async fn maybe_append_self_reflection(
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
        run_self_reflection_background(history_path, session_id, model_s, q_s, a_s, had_tool).await;
    });
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
    let cfg = configw::get_all_config();
    let max_days: i64 = cfg.get_opt("ai.memory.guidelines_days")
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(30);

    let hot_categories = [
        "self_note",
        "safety_rules",
        "common_sense",
        "coding_guideline",
        "best_practice",
        "user_preference",
        "preference",
    ];

    let mut entries: Vec<AgentMemoryEntry> = Vec::new();
    entries.extend(store.search(question, 160).ok().unwrap_or_default());
    for cat in hot_categories {
        if let Ok(mut v) = store.search(cat, 120) {
            entries.append(&mut v);
        }
    }
    if entries.is_empty() {
        entries = store.recent(200).ok().unwrap_or_default();
    }

    let mut seen: SkipSet<String> = SkipSet::new(16);

    fn parse_ts_utc(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
        chrono::DateTime::parse_from_rfc3339(s)
            .ok()
            .map(|dt| dt.with_timezone(&chrono::Utc))
    }

    fn hot_group(category: &str) -> u8 {
        match category {
            "safety_rules" => 0,
            "user_preference" | "preference" | "coding_guideline" | "best_practice"
            | "common_sense" => 1,
            "self_note" => 2,
            _ => 3,
        }
    }

    let mut ranked: Vec<(u8, u8, i64, String)> = Vec::with_capacity(entries.len());
    for e in entries {
        let category = e.category.to_lowercase();
        let priority = e.priority.unwrap_or(100);
        let group = hot_group(&category);
        if group >= 3 && priority < 200 {
            continue;
        }

        if max_days > 0 && group >= 2 {
            if let Some(dt) = parse_ts_utc(&e.timestamp) {
                let age_days =
                    (chrono::Utc::now() - dt).num_seconds().max(0) as i64 / 86400;
                if age_days > max_days {
                    continue;
                }
            }
        }

        let note = e.note.trim().to_string();
        if note.is_empty() {
            continue;
        }

        let ts_rank = parse_ts_utc(&e.timestamp)
            .map(|dt| dt.timestamp())
            .unwrap_or(0);
        ranked.push((group, priority, ts_rank, note));
    }

    ranked.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| b.2.cmp(&a.2))
    });

    let mut by_group: [Vec<String>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (group, _priority, _ts, note) in ranked {
        if seen.insert(note.clone()) {
            let g = group.min(3) as usize;
            by_group[g].push(note);
        }
    }
    if by_group.iter().all(|v| v.is_empty()) {
        return None;
    }

    let mut out = String::from("Persistent Guidelines:\n");
    let mut used = out.len();
    if used >= max_chars {
        return Some(out);
    }

    fn group_budget(max_chars: usize, used: usize, weights: [usize; 4], idx: usize) -> usize {
        let remaining = max_chars.saturating_sub(used);
        if remaining == 0 {
            return 0;
        }
        let total: usize = weights.iter().sum();
        let w = weights[idx];
        remaining.saturating_mul(w) / total.max(1)
    }

    fn append_notes(
        out: &mut String,
        used: &mut usize,
        max_chars: usize,
        budget: usize,
        notes: &[String],
    ) {
        if budget == 0 || *used >= max_chars {
            return;
        }
        let start_used = *used;
        for note in notes {
            for line in note.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let bullet = if line.starts_with('-') {
                    format!("{line}\n")
                } else {
                    format!("- {line}\n")
                };
                if *used + bullet.len() > max_chars {
                    return;
                }
                if *used - start_used + bullet.len() > budget {
                    return;
                }
                out.push_str(&bullet);
                *used += bullet.len();
                if *used >= max_chars {
                    return;
                }
            }
        }
    }

    let weights = [35usize, 40usize, 15usize, 10usize];
    for group_idx in 0..4 {
        if by_group[group_idx].is_empty() {
            continue;
        }
        let mut budget = group_budget(max_chars, used, weights, group_idx);
        budget = budget.max(120).min(max_chars.saturating_sub(used));
        append_notes(&mut out, &mut used, max_chars, budget, &by_group[group_idx]);
        if used >= max_chars {
            break;
        }
    }

    if used <= "Persistent Guidelines:\n".len() {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::build_persistent_guidelines;
    use crate::ai::test_support::ENV_LOCK;
    use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
    use chrono::Local;

    #[test]
    fn persistent_guidelines_include_safety_rules_and_high_priority_entries() {
        let _guard = ENV_LOCK.lock().unwrap();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_guidelines_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let store = MemoryStore::from_env_or_config();
        let timestamp = Local::now().to_rfc3339();
        store
            .append(&AgentMemoryEntry {
                id: None,
                timestamp: timestamp.clone(),
                category: "self_note".to_string(),
                note: "Do: validate tool arguments".to_string(),
                tags: vec![],
                source: Some("test".to_string()),
                priority: Some(100),
            })
            .unwrap();
        store
            .append(&AgentMemoryEntry {
                id: None,
                timestamp: timestamp.clone(),
                category: "safety_rules".to_string(),
                note: "Avoid: delete files without double checking".to_string(),
                tags: vec![],
                source: Some("test".to_string()),
                priority: Some(255),
            })
            .unwrap();
        store
            .append(&AgentMemoryEntry {
                id: None,
                timestamp: timestamp.clone(),
                category: "common_sense".to_string(),
                note: "Keep broadly applicable engineering habits in memory.".to_string(),
                tags: vec![],
                source: Some("test".to_string()),
                priority: Some(150),
            })
            .unwrap();
        store
            .append(&AgentMemoryEntry {
                id: None,
                timestamp: timestamp.clone(),
                category: "coding_guideline".to_string(),
                note: "Prefer cargo check before cargo test for quick feedback.".to_string(),
                tags: vec![],
                source: Some("test".to_string()),
                priority: Some(150),
            })
            .unwrap();
        store
            .append(&AgentMemoryEntry {
                id: None,
                timestamp: timestamp.clone(),
                category: "user_preference".to_string(),
                note: "Prefer concise, information-dense answers.".to_string(),
                tags: vec![],
                source: Some("test".to_string()),
                priority: Some(150),
            })
            .unwrap();
        store
            .append(&AgentMemoryEntry {
                id: None,
                timestamp: timestamp.clone(),
                category: "user_memory".to_string(),
                note: "Do: always ask before risky file operations".to_string(),
                tags: vec![],
                source: Some("test".to_string()),
                priority: Some(200),
            })
            .unwrap();
        store
            .append(&AgentMemoryEntry {
                id: None,
                timestamp,
                category: "user_memory".to_string(),
                note: "Ignore me".to_string(),
                tags: vec![],
                source: Some("test".to_string()),
                priority: Some(150),
            })
            .unwrap();

        let guidelines =
            build_persistent_guidelines("delete files safely", 1200).expect("guidelines");

        assert!(guidelines.contains("Do: validate tool arguments"));
        assert!(guidelines.contains("Avoid: delete files without double checking"));
        assert!(guidelines.contains("Keep broadly applicable engineering habits in memory."));
        assert!(guidelines.contains("Prefer cargo check before cargo test for quick feedback."));
        assert!(guidelines.contains("Prefer concise, information-dense answers."));
        assert!(guidelines.contains("Do: always ask before risky file operations"));
        assert!(!guidelines.contains("Ignore me"));

        let _ = std::fs::remove_file(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }
}

pub(super) async fn maybe_critic_and_revise(
    app: &mut App,
    model: &str,
    question: &str,
    draft: &str,
) -> Option<(String, String)> {
    use tokio::time::{timeout, Duration};
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
    let critic_fut = request::do_request_messages(app, model, &critic_req, false);
    let critic_resp = match timeout(Duration::from_millis(to_ms), critic_fut).await {
        Ok(Ok(r)) => r,
        _ => {
            // 恢复工具
            if let Some(mut tools) = saved_tools {
                if let Some(ctx) = app.agent_context.as_mut() {
                    std::mem::swap(&mut ctx.tools, &mut tools);
                }
            }
            return None;
        }
    };
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
    let revise_fut = request::do_request_messages(app, model, &revise_req, false);
    let revised_resp = match timeout(Duration::from_millis(to_ms), revise_fut).await {
        Ok(Ok(r)) => r,
        _ => {
            // 恢复工具
            if let Some(mut tools) = saved_tools {
                if let Some(ctx) = app.agent_context.as_mut() {
                    std::mem::swap(&mut ctx.tools, &mut tools);
                }
            }
            return None;
        }
    };
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

fn reflection_filtered(
    question: &str,
    answer: &str,
    turn_messages: &Vec<Message>,
) -> bool {
    let cfg = configw::get_all_config();
    let enabled = !cfg
        .get_opt("ai.reflection.filter.enable")
        .unwrap_or_else(|| "false".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if !enabled {
        return false;
    }
    let min_q = cfg
        .get_opt("ai.reflection.filter.min_question_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let min_a = cfg
        .get_opt("ai.reflection.filter.min_answer_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(80);
    let require_tool = cfg
        .get_opt("ai.reflection.filter.require_tool_or_long")
        .unwrap_or_else(|| "false".to_string())
        .trim()
        .eq_ignore_ascii_case("true");
    let q = question.trim();
    let a = answer.trim();
    if q.chars().count() < min_q && a.chars().count() < min_a {
        return true;
    }
    if require_tool && !turn_has_tool(turn_messages) && a.chars().count() < min_a {
        return true;
    }
    false
}

fn critic_filtered(question: &str, draft: &str) -> bool {
    let cfg = configw::get_all_config();
    let min_q = cfg
        .get_opt("ai.critic_revise.filter.min_question_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let min_a = cfg
        .get_opt("ai.critic_revise.filter.min_answer_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(120);
    let q = question.trim();
    let a = draft.trim();
    q.chars().count() < min_q && a.chars().count() < min_a
}

async fn model_should_reflect(
    app: &mut App,
    model: &str,
    question: &str,
    answer: &str,
    had_tool: bool,
) -> Option<bool> {
    use tokio::time::{timeout, Duration};
    let cfg = configw::get_all_config();
    let to_ms = cfg
        .get_opt("ai.reflection.model_gate.timeout_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(2000);
    let system = "You are a binary classifier that decides whether to capture a short 'experience note' for future turns.\nReturn STRICT JSON ONLY with the shape: {\"reflect\": true|false}.\nRules:\n- reflect=true when Q/A contains non-trivial reasoning, code, multi-step instructions, tool usage outcomes, errors/diagnosis, or decisions that should guide future behavior.\n- reflect=false for greetings, acknowledgements, trivial answers, or very short single-turn exchanges with no actionable guidance.\nDo not include explanations or extra text.";
    let user = format!(
        "question:\n{}\n\nanswer:\n{}\n\nhad_tool:\n{}",
        question.trim(),
        answer.trim(),
        if had_tool { "true" } else { "false" }
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
            content: build_content(model, &user, &[]).unwrap_or(Value::String(user)),
            tool_calls: None,
            tool_call_id: None,
        },
    ];
    let fut = request::do_request_messages(app, model, &messages, false);
    let resp = match timeout(Duration::from_millis(to_ms), fut).await {
        Ok(Ok(r)) => r,
        _ => return None,
    };
    let text = resp.text().await.ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = extract_content(&v).unwrap_or_default();
    parse_reflect_flag(&content)
}

fn parse_reflect_flag(s: &str) -> Option<bool> {
    let trimmed = s.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return v.get("reflect").and_then(|b| b.as_bool());
    }
    let l = trimmed.find('{')?;
    let r = trimmed.rfind('}')?;
    if r < l {
        return None;
    }
    let sub = &trimmed[l..=r];
    serde_json::from_str::<Value>(sub)
        .ok()
        .and_then(|v| v.get("reflect").and_then(|b| b.as_bool()))
}

fn turn_has_tool(messages: &Vec<Message>) -> bool {
    for m in messages {
        if m.role == "tool" {
            return true;
        }
        if let Some(calls) = m.tool_calls.as_ref() {
            if !calls.is_empty() {
                return true;
            }
        }
    }
    false
}

async fn model_should_revise(
    app: &mut App,
    model: &str,
    question: &str,
    draft: &str,
) -> Option<bool> {
    let system = "You decide if the DRAFT answer should be CRITIC→REVISE refined.\nReturn STRICT JSON ONLY: {\"revise\": true|false}.\nRules:\n- true ONLY for software engineering tasks: code writing/review/debug/refactor, tool execution results, build/test errors, patch proposals.\n- false for general knowledge, Q&A like weather/news/sports/finance, travel, generic suggestions, or casual chat.\n- false when the answer is short and already sufficient without code/steps.\nNo extra text.";
    let user = format!(
        "QUESTION:\n{}\n\nDRAFT:\n{}",
        question.trim(),
        draft.trim()
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
            content: build_content(model, &user, &[]).unwrap_or(Value::String(user)),
            tool_calls: None,
            tool_call_id: None,
        },
    ];
    let resp = request::do_request_messages(app, model, &messages, false).await.ok()?;
    let text = resp.text().await.ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = extract_content(&v).unwrap_or_default();
    if let Ok(v2) = serde_json::from_str::<Value>(content.trim()) {
        return v2.get("revise").and_then(|b| b.as_bool());
    }
    None
}

pub(super) async fn run_critic_revise_background(
    history_path: PathBuf,
    model: String,
    question: String,
    draft: String,
) {
    use tokio::time::{timeout, Duration};
    let cfg = configw::get_all_config();
    let to_ms = cfg
        .get_opt("ai.critic_revise.timeout_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(7000);
    // Perform critic
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
    // Perform revise
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
    let resp_r = match timeout(Duration::from_millis(to_ms), background_call(&model, &messages_r)).await {
        Ok(v) => v.and_then(|vv| Some(vv)),
        Err(_) => None,
    };
    let Some(resp_r) = resp_r else { return; };
    let content_r = extract_back_content(&resp_r).unwrap_or_default();
    if content_r.trim().is_empty() {
        return;
    }
    // Append as a single system record (background meta)
    let record = Message {
        role: "system".to_string(),
        content: Value::String(format!("critic:\n{}\n\nrevised:\n{}", content_c.trim(), content_r.trim())),
        tool_calls: None,
        tool_call_id: None,
    };
    let _ = append_history_messages(&history_path, &[record]);
}

async fn background_call(model: &str, messages: &Vec<Value>) -> Option<Value> {
    let cfg = configw::get_all_config();
    let endpoint = cfg.get_opt("ai.model.endpoint")?;
    let api_key = cfg.get_opt("api_key")?;
    let body = json!({
        "model": model,
        "messages": messages,
        "stream": false,
        "enable_thinking": false
    });
    let client = reqwest::Client::new();
    let resp = client
        .post(&endpoint)
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .ok()?;
    let text = resp.text().await.ok()?;
    serde_json::from_str::<Value>(&text).ok()
}

fn extract_back_content(v: &Value) -> Option<String> {
    let choices = v.get("choices").or_else(|| v.get("output")?.get("choices"))?;
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

pub(super) async fn run_self_reflection_background(
    history_path: PathBuf,
    session_id: String,
    model: String,
    question: String,
    answer: String,
    had_tool: bool,
) {
    use tokio::time::{timeout, Duration};
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
    let resp = match timeout(Duration::from_millis(to_ms_note), background_call(&model, &messages)).await {
        Ok(v) => v,
        Err(_) => None,
    };
    let Some(resp) = resp else { return; };
    let content = extract_back_content(&resp).unwrap_or_default();
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
    let _ = append_history_messages(&history_path, &[record]);
    let entry = AgentMemoryEntry {
        id: None,
        timestamp: Local::now().to_rfc3339(),
        category: "self_note".to_string(),
        note: note.to_string(),
        tags: vec!["agent".to_string(), "policy".to_string()],
        source: Some(format!("session:{}", session_id)),
        priority: Some(255), // Permanent: agent policies are never deleted
    };
    let store = MemoryStore::from_env_or_config();
    let _ = store.append(&entry);
    store.maintain_after_append();
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

fn reflection_filtered_bg(question: &str, answer: &str, had_tool: bool) -> bool {
    let cfg = configw::get_all_config();
    let enabled = !cfg
        .get_opt("ai.reflection.filter.enable")
        .unwrap_or_else(|| "false".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if !enabled {
        return false;
    }
    let min_q = cfg
        .get_opt("ai.reflection.filter.min_question_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let min_a = cfg
        .get_opt("ai.reflection.filter.min_answer_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(80);
    let require_tool = cfg
        .get_opt("ai.reflection.filter.require_tool_or_long")
        .unwrap_or_else(|| "false".to_string())
        .trim()
        .eq_ignore_ascii_case("true");
    let q = question.trim();
    let a = answer.trim();
    if q.chars().count() < min_q && a.chars().count() < min_a {
        return true;
    }
    if require_tool && !had_tool && a.chars().count() < min_a {
        return true;
    }
    false
}
