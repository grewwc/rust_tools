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

    // 在内核 daemon 登记表注册此后台 future，获得 handle + cancel token。
    // 用户态仍由 tokio::spawn 实际执行；退出时回调 daemon_exit 告知内核。
    use aios_kernel::primitives::DaemonKind;
    let (handle, cancel_token, kernel_arc, interrupt_futex) = {
        let kernel = app.os.clone();
        let (handle, token) = {
            let mut os = match kernel.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            let parent_pid = os.current_process_id();
            os.daemon_register(
                format!("self_reflection:{}", session_id),
                DaemonKind::Reflection,
                parent_pid,
            )
        };
        let interrupt_futex = crate::ai::driver::signal::alloc_interrupt_futex(format!(
            "background_reflection_interrupt:{}",
            session_id
        ));
        (handle, token, kernel, interrupt_futex)
    };

    tokio::spawn(async move {
        tokio::select! {
            _ = crate::ai::driver::signal::wait_for_interrupt_sources(
                Some(cancel_token.clone()),
                interrupt_futex,
            ) => {}
            _ = run_self_reflection_background(history_path, session_id, model_s, q_s, a_s, had_tool) => {}
        }
        let mut os = match kernel_arc.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        os.daemon_exit(handle, None);
        if let Some(addr) = interrupt_futex {
            crate::ai::driver::signal::destroy_interrupt_futex(addr);
        }
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
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: build_content(model, &critic_user, &[])
                .unwrap_or(Value::String(critic_user.clone())),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
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
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: build_content(model, &revise_user, &[])
                .unwrap_or(Value::String(revise_user.clone())),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
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
        reasoning_content: None,
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
    let system = "You are an introspective meta-optimizer and OS-level evolutionary engine for a coding assistant. Produce a brief self note and evolutionary policy to improve future runs.\nRules:\n- Output 2-6 compact bullets grouped under 'Do:' and 'Avoid:' tuned to the given Q&A.\n- Focus on planning, tool usage, goal decomposition, and verification habits.\n- Frame the learnings as 'System Evolution Policies' that will persist in memory and guide future agent processes.\n- No apologies, no explanations, no markdown code fences.\n- Keep under 800 chars.";
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
        reasoning_content: None,
    };
    let _ = append_history_messages(&history_path, &[record]);
    // ReflectionQuality 评分：低分仍写入但降低 priority，避免劣化召回排序
    // （只降不丢，保持原有"自我增强"行为；高分维持 150）。
    let quality = assess_reflection_quality(note);
    let priority = match quality.score() {
        0 => 90,  // 空泛/重复内容，几乎不会进 guideline 召回
        1 => 120, // 中等，参与召回但排序靠后
        _ => 150, // 高分，沿用原默认值
    };
    let entry = AgentMemoryEntry {
        id: None,
        timestamp: Local::now().to_rfc3339(),
        category: "self_note".to_string(),
        note: note.to_string(),
        tags: vec!["agent".to_string(), "policy".to_string()],
        source: Some(format!("session:{}", session_id)),
        // self_note 是会话期短期反思，不应作为永久记忆，让其参与正常 GC
        priority: Some(priority),
        owner_pid: None,
        owner_pgid: None,
    };
    let store = MemoryStore::from_env_or_config();
    // 矛盾检测：扫描近 100 条 self_note，若新 note 与既有条目语义相反
    // （Do/Avoid 翻转、关键短语相同极性相反），把旧条目降到 priority 60
    // 让 GC 回收它，避免新旧策略同时被召回造成 agent 行为摇摆。
    demote_contradicting_self_notes(&store, note);
    let _ = store.append(&entry);
    store.maintain_after_append();
}

/// 扫描近期 self_note，检测与新 note 的"反向极性"重复并降级旧条目。
///
/// 朴素启发式（保守，宁可漏报不可误伤）：
/// - 提取双方 "do:" / "avoid:" 之后的短语 token 集合
/// - 若 A.do ∩ B.avoid 中某 token 长度 ≥ 4 且非停用词，判定矛盾
/// - 把旧条目 priority 降到 60（低于普通 self_note 召回门槛）
fn demote_contradicting_self_notes(store: &MemoryStore, new_note: &str) {
    let Some((new_do, new_avoid)) = split_do_avoid(new_note) else {
        return;
    };
    let recent = match store.entries_by_category("self_note", 100) {
        Ok(e) => e,
        Err(_) => return,
    };
    for old in recent {
        let Some(id) = old.id.as_deref() else { continue };
        // 已经被降过的不再重复处理
        if old.priority.unwrap_or(150) <= 60 {
            continue;
        }
        let Some((old_do, old_avoid)) = split_do_avoid(&old.note) else {
            continue;
        };
        if has_polarity_conflict(&new_do, &old_avoid)
            || has_polarity_conflict(&new_avoid, &old_do)
        {
            // 用现有 update API 重写 priority，保持其他字段不变
            let _ = crate::ai::tools::service::memory::execute_memory_update(&serde_json::json!({
                "id": id,
                "priority": 60,
            }));
        }
    }
}

fn split_do_avoid(text: &str) -> Option<(Vec<String>, Vec<String>)> {
    let lower = text.to_lowercase();
    if !lower.contains("do:") && !lower.contains("avoid:") {
        return None;
    }
    let mut do_tokens: Vec<String> = Vec::new();
    let mut avoid_tokens: Vec<String> = Vec::new();
    for line in lower.lines() {
        let t = line.trim_start_matches(['-', '*', ' ', '\t']);
        if let Some(rest) = t.strip_prefix("do:") {
            do_tokens.extend(extract_keyword_tokens(rest));
        } else if let Some(rest) = t.strip_prefix("avoid:") {
            avoid_tokens.extend(extract_keyword_tokens(rest));
        }
    }
    Some((do_tokens, avoid_tokens))
}

fn extract_keyword_tokens(s: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "the", "and", "for", "with", "without", "into", "onto", "from", "this", "that",
        "your", "you", "are", "was", "were", "have", "has", "had", "but", "not", "can",
        "should", "would", "could", "may", "might", "will", "shall", "before", "after",
        "when", "where", "what", "which", "who", "whom",
    ];
    s.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter_map(|tok| {
            let t = tok.trim().to_string();
            if t.len() < 4 {
                None
            } else if STOP.contains(&t.as_str()) {
                None
            } else {
                Some(t)
            }
        })
        .collect()
}

fn has_polarity_conflict(a: &[String], b: &[String]) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    let set_a: std::collections::HashSet<&String> = a.iter().collect();
    b.iter().any(|t| set_a.contains(t))
}

/// 启发式评估 self-note 的"可执行性 / 具体性 / 可泛化性"。
///
/// 设计原则（保守，不丢弃）：
/// - actionable：包含 "Do:"/"Avoid:" 或动词导向的祈使句关键词
/// - specific：包含工具名 / 文件路径 / 函数名等具体标记（含 ` ` 反引号、
///   "()"、"::"、"/"、"."），或长度大于 80 字符
/// - generalizable：包含描述习惯/原则的关键词，且不是纯一次性事实
fn assess_reflection_quality(note: &str) -> super::ReflectionQuality {
    let lower = note.to_lowercase();
    let actionable = lower.contains("do:")
        || lower.contains("avoid:")
        || lower.contains("prefer ")
        || lower.contains("always ")
        || lower.contains("never ")
        || lower.contains("ensure ");
    let specific = note.contains('`')
        || note.contains("::")
        || note.contains("()")
        || note.contains('/')
        || note.chars().count() >= 80;
    let generalizable = lower.contains("when ")
        || lower.contains("before ")
        || lower.contains("after ")
        || lower.contains("instead ")
        || lower.contains("rather ")
        || lower.contains("habit")
        || lower.contains("policy")
        || lower.contains("pattern");
    super::ReflectionQuality {
        actionable,
        specific,
        generalizable,
    }
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

/// 共享的 reqwest 客户端：避免 background_call 每次重建连接池/TLS/DNS。
/// 后台 reflection / critic / revise 都走这里，给 50–300ms latency 让出。
static BACKGROUND_HTTP_CLIENT: std::sync::LazyLock<reqwest::Client> =
    std::sync::LazyLock::new(reqwest::Client::new);

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
        crate::ai::provider::ApiProvider::Compatible => json!({
            "model": model,
            "messages": messages,
            "stream": false,
            "enable_thinking": false
        }),
        _ => json!({
            "model": model,
            "messages": messages,
            "stream": false
        }),
    };
    let req = BACKGROUND_HTTP_CLIENT.post(&endpoint);
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
