use std::path::PathBuf;

use chrono::Local;
use serde_json::{Value, json};

use crate::ai::history::{Message, append_history_messages};
use crate::ai::request::{self, build_content};
use crate::ai::tools::service::memory::execute_memory_update;
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
    let Some(resp_c) = background_call(&model, &messages_c).await else {
        return;
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
    let _ = append_history_messages(history_path.as_path(), &[record]);
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
    let _ = append_history_messages(history_path.as_path(), &[record]);
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

    // 用真实 turn 信号更新进化策略健康度（pass/fail），驱动 canary 升级与回滚。
    apply_evolution_feedback(&store, q, a, had_tool);

    // 若新 note 与当前激活的进化 guideline 明显冲突，回滚到上一版稳定策略。
    maybe_rollback_promoted_guideline(&store, note);

    // 经验晋升：高质量且跨轮重复出现的 self_note 提升为稳定 guideline，
    // 让有效经验真正沉淀到长期策略层，而不是仅在短期反思层漂移。
    maybe_promote_stable_self_note(&store, note, &quality);

    store.maintain_after_append();
}

fn maybe_promote_stable_self_note(
    store: &MemoryStore,
    note: &str,
    quality: &super::ReflectionQuality,
) {
    if !quality.is_high_quality() {
        return;
    }

    let signature = normalize_evolution_note(note);
    if signature.is_empty() {
        return;
    }

    let recent_self_notes = match store.entries_by_category("self_note", 200) {
        Ok(entries) => entries,
        Err(_) => return,
    };

    let repeated_count = recent_self_notes
        .iter()
        .filter(|entry| normalize_evolution_note(&entry.note) == signature)
        .count();
    if repeated_count < 3 {
        return;
    }

    let guidelines = reflection_evolution_guidelines(store);

    // 同签名 guideline 已存在则不重复晋升。
    let exists = guidelines
        .iter()
        .any(|entry| {
            entry
                .tags
                .iter()
                .any(|tag| tag == &format!("evo_sig:{signature}"))
        });
    if exists {
        return;
    }

    let next_ver = next_evolution_version_from(&guidelines);
    let has_active = has_active_guideline(&guidelines);
    let has_canary = has_canary_guideline(&guidelines);
    // 有 active 时只允许单 canary 在评估中，避免并发试验导致策略抖动。
    if has_active && has_canary {
        return;
    }
    let next_state = if has_active { "canary" } else { "active" };
    let next_priority = if has_active { 155 } else { 170 };

    let promoted = AgentMemoryEntry {
        id: None,
        timestamp: Local::now().to_rfc3339(),
        category: "coding_guideline".to_string(),
        note: note.trim().to_string(),
        tags: vec![
            "agent".to_string(),
            "policy".to_string(),
            "evolution_promoted".to_string(),
            "evo:v1".to_string(),
            "evo_stream:reflection".to_string(),
            format!("evo_ver:{next_ver}"),
            format!("evo_state:{next_state}"),
            "evo_pass:0".to_string(),
            "evo_fail:0".to_string(),
            format!("evo_sig:{signature}"),
        ],
        source: Some(format!("auto_reflection_promotion:v{next_ver}")),
        priority: Some(next_priority),
        owner_pid: None,
        owner_pgid: None,
    };
    let _ = store.append(&promoted);
}

fn apply_evolution_feedback(store: &MemoryStore, question: &str, answer: &str, had_tool: bool) {
    let signal = evaluate_turn_feedback(question, answer, had_tool);

    let guidelines = reflection_evolution_guidelines(store);
    let target = current_canary_evolution_guideline_from(&guidelines)
        .or_else(|| current_active_evolution_guideline_from(&guidelines));
    let Some(target) = target else {
        return;
    };
    let Some(id) = target.id.as_deref() else {
        return;
    };

    let pass = parse_tag_u32(&target.tags, "evo_pass").unwrap_or(0);
    let fail = parse_tag_u32(&target.tags, "evo_fail").unwrap_or(0);
    let (pass, fail) = next_feedback_counters(pass, fail, signal);

    let mut tags = upsert_tag(&target.tags, "evo_pass", &pass.to_string());
    tags = upsert_tag(&tags, "evo_fail", &fail.to_string());
    let _ = execute_memory_update(&serde_json::json!({
        "id": id,
        "tags": tags,
    }));

    let state = tag_value(&target.tags, "evo_state").unwrap_or_default();
    if state == "canary" {
        let active = current_active_evolution_guideline_from(&guidelines);
        maybe_activate_canary(active, &target, pass, fail);
    } else if state == "active" {
        let active_ver = parse_evo_ver(&target.tags).unwrap_or(0);
        let previous = previous_evolution_guideline_from(&guidelines, active_ver);
        maybe_rollback_on_feedback(previous, &target, pass, fail);
    }
}

const EVO_FEEDBACK_COUNTER_WINDOW: u32 = 12;
const EVO_CANARY_REJECT_FAILS: u32 = 2;
const EVO_CANARY_ACTIVATE_PASSES: u32 = 3;
const EVO_ACTIVE_ROLLBACK_FAILS_MIN: u32 = 3;
const EVO_ACTIVE_FAIL_MARGIN: u32 = 2;

fn next_feedback_counters(pass: u32, fail: u32, signal: EvolutionFeedback) -> (u32, u32) {
    let mut pass = pass;
    let mut fail = fail;
    // 固定窗口衰减：把历史累计压缩到近似最近 N 次反馈，避免旧失败长期污染。
    if pass.saturating_add(fail) >= EVO_FEEDBACK_COUNTER_WINDOW {
        pass /= 2;
        fail /= 2;
    }
    match signal {
        EvolutionFeedback::Pass => (pass.saturating_add(1), fail),
        EvolutionFeedback::Fail => (pass, fail.saturating_add(1)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvolutionFeedback {
    Pass,
    Fail,
}

fn evaluate_turn_feedback(question: &str, answer: &str, had_tool: bool) -> EvolutionFeedback {
    let q = question.to_lowercase();
    let a = answer.to_lowercase();

    if question_looks_like_user_correction(q.as_str()) {
        return EvolutionFeedback::Fail;
    }

    let failure_markers = [
        "error", "failed", "failure", "panic", "exception", "traceback", "exit code",
        "超时", "失败", "报错", "异常", "未找到", "permission denied",
    ];
    if had_tool && failure_markers.iter().any(|m| a.contains(m)) {
        return EvolutionFeedback::Fail;
    }

    if a.trim().is_empty() {
        return EvolutionFeedback::Fail;
    }
    EvolutionFeedback::Pass
}

fn question_looks_like_user_correction(lower_q: &str) -> bool {
    const CORRECTION_MARKERS: &[&str] = &[
        "你错",
        "不对",
        "还是不对",
        "请重试",
        "重试一下",
        "重新回答",
        "重新生成",
        "纠正一下",
        "改正一下",
        "wrong answer",
        "incorrect",
        "not correct",
        "try again",
        "redo",
    ];
    CORRECTION_MARKERS.iter().any(|m| lower_q.contains(m))
}

fn maybe_activate_canary(
    active: Option<AgentMemoryEntry>,
    canary: &AgentMemoryEntry,
    pass: u32,
    fail: u32,
) {
    // 灰度策略：连续积累正反馈后再转 active；失败过多则拒绝该 canary。
    if fail >= EVO_CANARY_REJECT_FAILS {
        if let Some(id) = canary.id.as_deref() {
            let mut tags = upsert_tag(&canary.tags, "evo_state", "rejected");
            tags = upsert_tag(&tags, "evo_reject", &Local::now().format("%Y%m%d%H%M%S").to_string());
            let _ = execute_memory_update(&serde_json::json!({
                "id": id,
                "priority": 90,
                "tags": tags,
            }));
        }
        return;
    }

    if pass < EVO_CANARY_ACTIVATE_PASSES {
        return;
    }

    deactivate_active_evolution_guideline(active);
    if let Some(id) = canary.id.as_deref() {
        let tags = upsert_tag(&canary.tags, "evo_state", "active");
        let _ = execute_memory_update(&serde_json::json!({
            "id": id,
            "priority": 175,
            "tags": tags,
        }));
    }
}

fn maybe_rollback_on_feedback(
    previous: Option<AgentMemoryEntry>,
    active: &AgentMemoryEntry,
    pass: u32,
    fail: u32,
) {
    // 真实负反馈触发：失败累计达到阈值且明显劣于成功。
    if fail < EVO_ACTIVE_ROLLBACK_FAILS_MIN || fail < pass.saturating_add(EVO_ACTIVE_FAIL_MARGIN) {
        return;
    }
    if let Some(id) = active.id.as_deref() {
        let mut tags = upsert_tag(&active.tags, "evo_state", "rolled_back");
        tags = upsert_tag(&tags, "evo_feedback_rollback", &Local::now().format("%Y%m%d%H%M%S").to_string());
        let _ = execute_memory_update(&serde_json::json!({
            "id": id,
            "priority": 85,
            "tags": tags,
        }));
    }
    if let Some(previous) = previous
        && let Some(prev_id) = previous.id.as_deref()
    {
        let tags = upsert_tag(&previous.tags, "evo_state", "active");
        let _ = execute_memory_update(&serde_json::json!({
            "id": prev_id,
            "priority": 175,
            "tags": tags,
        }));
    }
}

fn maybe_rollback_promoted_guideline(store: &MemoryStore, new_note: &str) {
    let guidelines = reflection_evolution_guidelines(store);
    let Some(active) = current_active_evolution_guideline_from(&guidelines) else {
        return;
    };

    if !evolution_notes_conflict(new_note, &active.note) {
        return;
    }

    let active_ver = parse_evo_ver(&active.tags).unwrap_or(0);
    if let Some(id) = active.id.as_deref() {
        let mut tags = upsert_tag(&active.tags, "evo_state", "rolled_back");
        let ts = Local::now().format("%Y%m%d%H%M%S").to_string();
        tags = upsert_tag(&tags, "evo_rollback", &ts);
        let _ = execute_memory_update(&serde_json::json!({
            "id": id,
            "priority": 80,
            "tags": tags,
        }));
    }

    if let Some(previous) = previous_evolution_guideline_from(&guidelines, active_ver)
        && let Some(prev_id) = previous.id.as_deref()
    {
        let tags = upsert_tag(&previous.tags, "evo_state", "active");
        let _ = execute_memory_update(&serde_json::json!({
            "id": prev_id,
            "priority": 175,
            "tags": tags,
        }));
    }
}

fn next_evolution_version(store: &MemoryStore) -> u32 {
    let entries = reflection_evolution_guidelines(store);
    next_evolution_version_from(&entries)
}

fn next_evolution_version_from(entries: &[AgentMemoryEntry]) -> u32 {
    let max_ver = entries
        .iter()
        .filter_map(|entry| parse_evo_ver(&entry.tags))
        .max()
        .unwrap_or(0);
    max_ver.saturating_add(1)
}

fn deactivate_active_evolution_guideline(active: Option<AgentMemoryEntry>) {
    let Some(active) = active else {
        return;
    };
    let Some(id) = active.id.as_deref() else {
        return;
    };
    let tags = upsert_tag(&active.tags, "evo_state", "superseded");
    let _ = execute_memory_update(&serde_json::json!({
        "id": id,
        "priority": 140,
        "tags": tags,
    }));
}

fn current_active_evolution_guideline(store: &MemoryStore) -> Option<AgentMemoryEntry> {
    let entries = reflection_evolution_guidelines(store);
    current_active_evolution_guideline_from(&entries)
}

fn current_active_evolution_guideline_from(
    entries: &[AgentMemoryEntry],
) -> Option<AgentMemoryEntry> {
    entries
        .iter()
        .filter(|entry| tag_value(&entry.tags, "evo_state").as_deref() == Some("active"))
        .cloned()
        .max_by_key(|entry| parse_evo_ver(&entry.tags).unwrap_or(0))
}

fn current_canary_evolution_guideline(store: &MemoryStore) -> Option<AgentMemoryEntry> {
    let entries = reflection_evolution_guidelines(store);
    current_canary_evolution_guideline_from(&entries)
}

fn current_canary_evolution_guideline_from(
    entries: &[AgentMemoryEntry],
) -> Option<AgentMemoryEntry> {
    entries
        .iter()
        .filter(|entry| tag_value(&entry.tags, "evo_state").as_deref() == Some("canary"))
        .cloned()
        .max_by_key(|entry| parse_evo_ver(&entry.tags).unwrap_or(0))
}

fn previous_evolution_guideline(store: &MemoryStore, current_ver: u32) -> Option<AgentMemoryEntry> {
    let entries = reflection_evolution_guidelines(store);
    previous_evolution_guideline_from(&entries, current_ver)
}

fn previous_evolution_guideline_from(
    entries: &[AgentMemoryEntry],
    current_ver: u32,
) -> Option<AgentMemoryEntry> {
    entries
        .iter()
        .filter(|entry| parse_evo_ver(&entry.tags).unwrap_or(0) < current_ver)
        .cloned()
        .max_by_key(|entry| parse_evo_ver(&entry.tags).unwrap_or(0))
}

fn reflection_evolution_guidelines(store: &MemoryStore) -> Vec<AgentMemoryEntry> {
    store
        .entries_by_category("coding_guideline", 500)
        .ok()
        .unwrap_or_default()
        .into_iter()
        .filter(is_reflection_evolution_guideline)
        .collect()
}

fn is_reflection_evolution_guideline(entry: &AgentMemoryEntry) -> bool {
    entry.tags.iter().any(|tag| tag == "evolution_promoted")
        && tag_value(&entry.tags, "evo_stream").as_deref() == Some("reflection")
}

fn parse_evo_ver(tags: &[String]) -> Option<u32> {
    tag_value(tags, "evo_ver")?.parse::<u32>().ok()
}

fn parse_tag_u32(tags: &[String], key: &str) -> Option<u32> {
    tag_value(tags, key)?.parse::<u32>().ok()
}

fn tag_value(tags: &[String], key: &str) -> Option<String> {
    let prefix = format!("{key}:");
    tags.iter().find_map(|tag| {
        if tag.starts_with(&prefix) {
            Some(tag[prefix.len()..].to_string())
        } else {
            None
        }
    })
}

fn upsert_tag(tags: &[String], key: &str, value: &str) -> Vec<String> {
    let prefix = format!("{key}:");
    let mut out = Vec::with_capacity(tags.len() + 1);
    let mut replaced = false;
    for tag in tags {
        if tag.starts_with(&prefix) {
            if !replaced {
                out.push(format!("{key}:{value}"));
                replaced = true;
            }
            continue;
        }
        out.push(tag.clone());
    }
    if !replaced {
        out.push(format!("{key}:{value}"));
    }
    out
}

fn has_active_guideline(entries: &[AgentMemoryEntry]) -> bool {
    entries
        .iter()
        .any(|entry| tag_value(&entry.tags, "evo_state").as_deref() == Some("active"))
}

fn has_canary_guideline(entries: &[AgentMemoryEntry]) -> bool {
    entries
        .iter()
        .any(|entry| tag_value(&entry.tags, "evo_state").as_deref() == Some("canary"))
}

fn evolution_notes_conflict(a_note: &str, b_note: &str) -> bool {
    let Some((a_do, a_avoid)) = split_do_avoid(a_note) else {
        return false;
    };
    let Some((b_do, b_avoid)) = split_do_avoid(b_note) else {
        return false;
    };
    has_polarity_conflict(&a_do, &b_avoid) || has_polarity_conflict(&a_avoid, &b_do)
}

fn normalize_evolution_note(note: &str) -> String {
    note
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
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

/// 共享的学习/反思质量评估：
/// - 不是简单的单一 substring 命中，而是融合多类信号
/// - 同时服务于 background reflection 和 memory_save 长期记忆门禁
pub(crate) fn assess_learning_note_quality(note: &str) -> super::LearningNoteAssessment {
    let features = LearningNoteQualityFeatures::from_note(note);
    let actionable = features.directive_signals > 0
        || (features.condition_signals > 0 && features.word_count >= 8)
        || (features.nonempty_lines >= 2 && features.word_count >= 10);
    let specific = features.code_signals > 0
        || features.artifact_signals >= 2
        || (features.directive_signals > 0
            && features.word_count >= 6
            && features.unique_token_ratio >= 0.75)
        || (features.char_count >= 64
            && features.word_count >= 10
            && features.unique_token_ratio >= 0.55);
    let generalizable = features.one_off_signals == 0
        && (features.condition_signals > 0
            || features.abstraction_signals > 0
            || (features.word_count >= 10
                && features.unique_token_ratio >= 0.55
                && features.nonempty_lines >= 2));

    let quality = super::ReflectionQuality {
        actionable,
        specific,
        generalizable,
    };
    super::LearningNoteAssessment {
        actionable: quality.actionable,
        specific: quality.specific,
        generalizable: quality.generalizable,
        score: quality.score(),
        high_quality: quality.is_high_quality(),
        char_count: features.char_count,
        word_count: features.word_count,
        nonempty_lines: features.nonempty_lines,
        unique_token_ratio: features.unique_token_ratio,
        directive_signals: features.directive_signals,
        code_signals: features.code_signals,
        artifact_signals: features.artifact_signals,
        abstraction_signals: features.abstraction_signals,
        condition_signals: features.condition_signals,
        one_off_signals: features.one_off_signals,
    }
}

fn assess_reflection_quality(note: &str) -> super::ReflectionQuality {
    let assessment = assess_learning_note_quality(note);
    super::ReflectionQuality {
        actionable: assessment.actionable,
        specific: assessment.specific,
        generalizable: assessment.generalizable,
    }
}

struct LearningNoteQualityFeatures {
    char_count: usize,
    word_count: usize,
    nonempty_lines: usize,
    unique_token_ratio: f32,
    directive_signals: usize,
    code_signals: usize,
    artifact_signals: usize,
    abstraction_signals: usize,
    condition_signals: usize,
    one_off_signals: usize,
}

impl LearningNoteQualityFeatures {
    fn from_note(note: &str) -> Self {
        let trimmed = note.trim();
        if trimmed.is_empty() {
            return Self {
                char_count: 0,
                word_count: 0,
                nonempty_lines: 0,
                unique_token_ratio: 0.0,
                directive_signals: 0,
                code_signals: 0,
                artifact_signals: 0,
                abstraction_signals: 0,
                condition_signals: 0,
                one_off_signals: 1,
            };
        }

        let lower = trimmed.to_lowercase();
        let tokens = quality_tokens(trimmed);
        let word_count = tokens.len();
        let unique_token_ratio = if word_count == 0 {
            0.0
        } else {
            let unique = tokens.iter().collect::<std::collections::HashSet<_>>().len();
            unique as f32 / word_count as f32
        };

        let directive_signals = count_contains(
            &lower,
            &[
                "do:", "avoid:", "prefer ", "should ", "must ", "always ", "never ",
                "ensure ", "应该", "必须", "不要", "避免", "优先", "确保",
            ],
        );
        let condition_signals = count_contains(
            &lower,
            &[
                "when ", "if ", "before ", "after ", "instead ", "rather than ",
                "unless ", "当", "如果", "之前", "之后", "而不是", "否则",
            ],
        );
        let abstraction_signals = count_contains(
            &lower,
            &[
                "habit", "policy", "pattern", "rule", "guideline", "principle",
                "workflow", "strategy", "习惯", "规则", "准则", "原则", "策略", "流程",
            ],
        );
        let code_signals = count_code_signals(trimmed);
        let artifact_signals = count_artifact_signals(trimmed, &tokens);
        let one_off_signals = count_one_off_signals(trimmed, &tokens);

        Self {
            char_count: trimmed.chars().count(),
            word_count,
            nonempty_lines: trimmed.lines().filter(|line| !line.trim().is_empty()).count(),
            unique_token_ratio,
            directive_signals,
            code_signals,
            artifact_signals,
            abstraction_signals,
            condition_signals,
            one_off_signals,
        }
    }
}

fn quality_tokens(note: &str) -> Vec<String> {
    note.split(|ch: char| !(ch.is_alphanumeric() || ch == '_' || ch == '-' || ch == '.'))
        .map(str::trim)
        .filter(|token| token.chars().count() >= 2)
        .map(|token| token.to_lowercase())
        .collect()
}

fn count_contains(haystack: &str, needles: &[&str]) -> usize {
    needles.iter().filter(|needle| haystack.contains(**needle)).count()
}

fn count_code_signals(note: &str) -> usize {
    let markers = [
        '`', '/', '\\', '(', ')', ':', '[', ']', '{', '}',
    ];
    let mut count = markers.iter().filter(|marker| note.contains(**marker)).count();
    if note.contains("::") {
        count += 1;
    }
    if note.contains("->") || note.contains("=>") {
        count += 1;
    }
    count
}

fn count_artifact_signals(note: &str, tokens: &[String]) -> usize {
    let mut count = 0usize;
    if note.contains(".rs") || note.contains(".ts") || note.contains(".py") || note.contains(".md") {
        count += 1;
    }
    count += tokens
        .iter()
        .filter(|token| token.contains('_') || token.contains("::") || token.ends_with("()"))
        .count();
    count += tokens
        .iter()
        .filter(|token| token.chars().any(|ch| ch.is_ascii_digit()))
        .count()
        .min(1);
    count
}

fn count_one_off_signals(note: &str, tokens: &[String]) -> usize {
    let mut count = 0usize;
    if note.contains("session:") || note.contains("tmp") || note.contains("/var/") {
        count += 1;
    }
    count += tokens
        .iter()
        .filter(|token| token.len() >= 12 && token.chars().filter(|ch| ch.is_ascii_digit()).count() >= 4)
        .count();
    count += tokens
        .iter()
        .filter(|token| token.len() >= 16 && token.chars().all(|ch| ch.is_ascii_hexdigit() || ch == '-'))
        .count();
    count
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsert_tag_replaces_existing_value() {
        let tags = vec![
            "evo_stream:reflection".to_string(),
            "evo_state:active".to_string(),
        ];
        let out = upsert_tag(&tags, "evo_state", "rolled_back");
        assert!(out.iter().any(|tag| tag == "evo_state:rolled_back"));
        assert!(!out.iter().any(|tag| tag == "evo_state:active"));
    }

    #[test]
    fn parse_evo_ver_reads_numeric_tag() {
        let tags = vec![
            "evo_stream:reflection".to_string(),
            "evo_ver:7".to_string(),
        ];
        assert_eq!(parse_evo_ver(&tags), Some(7));
    }

    #[test]
    fn evolution_notes_conflict_detects_do_avoid_flip() {
        let newer = "Do: validate tool arguments before calling tools\nAvoid: guessing";
        let older = "Do: guessing quickly\nAvoid: validate tool arguments";
        assert!(evolution_notes_conflict(newer, older));
    }

    #[test]
    fn evaluate_turn_feedback_flags_tool_failures() {
        let feedback = evaluate_turn_feedback(
            "继续执行",
            "Error: command failed with exit code 1",
            true,
        );
        assert_eq!(feedback, EvolutionFeedback::Fail);
    }

    #[test]
    fn evaluate_turn_feedback_flags_user_corrections() {
        let feedback = evaluate_turn_feedback(
            "这个不对，请重试并修复",
            "我会继续处理",
            false,
        );
        assert_eq!(feedback, EvolutionFeedback::Fail);
    }

    #[test]
    fn evaluate_turn_feedback_keeps_normal_fix_requests_as_pass() {
        let feedback = evaluate_turn_feedback(
            "请帮我 fix 这个 rust 编译错误",
            "可以，我先定位报错并修复",
            false,
        );
        assert_eq!(feedback, EvolutionFeedback::Pass);
    }

    #[test]
    fn parse_tag_u32_reads_counter_tags() {
        let tags = vec!["evo_pass:3".to_string(), "evo_fail:1".to_string()];
        assert_eq!(parse_tag_u32(&tags, "evo_pass"), Some(3));
        assert_eq!(parse_tag_u32(&tags, "evo_fail"), Some(1));
    }

    #[test]
    fn next_feedback_counters_decay_then_add_new_signal() {
        let (pass, fail) = next_feedback_counters(10, 2, EvolutionFeedback::Fail);
        assert_eq!((pass, fail), (5, 2));

        let (pass, fail) = next_feedback_counters(2, 1, EvolutionFeedback::Pass);
        assert_eq!((pass, fail), (3, 1));
    }
}
