use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration as StdDuration, Instant};
use uuid::Uuid;

use crate::ai::history::Message;
use crate::ai::tools::service::memory::execute_memory_update;
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use crate::ai::types::App;
use crate::commonw::configw;
use chrono::Local;

use super::gates::{answer_looks_unstable_for_writeback, turn_uses_repo_inspection_tools};
use super::recall::current_project_name_hint;

/// 短期写回去重缓存：(project, q_hash, a_hash) -> last_seen_instant。
/// 5 分钟内相同 question+answer 不再触发 LLM 调用，避免连续多轮重复问答时
/// 反复发起 writeback LLM 请求；MemoryStore upsert 仍然是幂等的，这里只是省 token。
static WRITEBACK_RECENT: LazyLock<Mutex<HashMap<(String, u64, u64), Instant>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
const WRITEBACK_DEDUP_TTL: StdDuration = StdDuration::from_secs(5 * 60);

fn writeback_dedup_key(project: &str, question: &str, answer: &str) -> (String, u64, u64) {
    use std::hash::{Hash, Hasher};
    let mut q_h = rustc_hash::FxHasher::default();
    question.trim().hash(&mut q_h);
    let mut a_h = rustc_hash::FxHasher::default();
    answer.trim().hash(&mut a_h);
    (project.to_string(), q_h.finish(), a_h.finish())
}

fn writeback_recently_seen(key: &(String, u64, u64)) -> bool {
    let mut guard = match WRITEBACK_RECENT.lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    let now = Instant::now();
    guard.retain(|_, t| now.duration_since(*t) < WRITEBACK_DEDUP_TTL);
    if let Some(t) = guard.get(key) {
        if now.duration_since(*t) < WRITEBACK_DEDUP_TTL {
            return true;
        }
    }
    guard.insert(key.clone(), now);
    false
}

/// 把 reflection 写入的 AgentMemoryEntry 同步到向量索引，让 semantic search 能召回。
/// 失败仅打印 warning，不阻塞 JSONL 写入流程。
fn sync_agent_entry_to_vector(entry: &AgentMemoryEntry) {
    let knowledge_entry = crate::ai::knowledge::entry::KnowledgeEntry {
        id: entry.id.clone(),
        timestamp: entry.timestamp.clone(),
        category: entry.category.clone(),
        note: entry.note.clone(),
        tags: entry.tags.clone(),
        source: entry.source.clone(),
        priority: entry.priority,
    };
    let guard = match crate::ai::tools::storage::rag_store::get_rag_store() {
        Ok(g) => g,
        Err(_) => return,
    };
    let Some(store) = guard.as_ref() else {
        return;
    };
    if let Err(err) = crate::ai::knowledge::sync::knowledge_sync::sync_entry_to_vector(
        store,
        &knowledge_entry,
    ) {
        eprintln!("[Memory] writeback vector sync failed: {}", err);
    }
}

pub(super) async fn maybe_write_back_project_knowledge(
    _app: &mut App,
    model: &str,
    question: &str,
    answer: &str,
    turn_messages: &Vec<Message>,
) {
    let cfg = configw::get_all_config();
    let enabled = !cfg
        .get_opt("ai.project_writeback.enable")
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
    if !turn_uses_repo_inspection_tools(turn_messages) {
        return;
    }
    if answer_looks_unstable_for_writeback(a) {
        return;
    }

    let Some(project_name) = current_project_name_hint() else {
        return;
    };
    // 5 分钟内已对相同 (project, question, answer) 跑过 writeback：直接跳过 LLM 调用。
    let dedup_key = writeback_dedup_key(&project_name, q, a);
    if writeback_recently_seen(&dedup_key) {
        return;
    }
    let model_s = cfg
        .get_opt("ai.project_writeback.model")
        .unwrap_or_else(|| model.to_string());
    let q_s = q.to_string();
    let a_s = a.to_string();
    run_project_knowledge_writeback_background(project_name, model_s, q_s, a_s).await;
}

#[derive(Debug)]
struct ProjectWritebackPayload {
    content: String,
    tags: Vec<String>,
    priority: u8,
}

fn parse_project_writeback_payload(s: &str) -> Option<ProjectWritebackPayload> {
    let trimmed = s.trim();
    let candidate = if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        v
    } else {
        let l = trimmed.find('{')?;
        let r = trimmed.rfind('}')?;
        if r < l {
            return None;
        }
        serde_json::from_str::<Value>(&trimmed[l..=r]).ok()?
    };

    if !candidate
        .get("writeback")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return None;
    }
    let content = candidate
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if content.is_empty() {
        return None;
    }
    let tags = candidate
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .take(8)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let priority = candidate
        .get("priority")
        .and_then(|v| v.as_u64())
        .map(|v| if v > u8::MAX as u64 { u8::MAX } else { v as u8 })
        .unwrap_or(180)
        .max(150);
    Some(ProjectWritebackPayload {
        content,
        tags,
        priority,
    })
}

fn project_writeback_quality_ok(content: &str) -> bool {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.chars().count() < 40 {
        return false;
    }

    let lines = trimmed
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.len() < 2 {
        return false;
    }

    let lower = trimmed.to_lowercase();
    let uncertainty_markers = [
        "maybe",
        "might",
        "unsure",
        "probably",
        "not sure",
        "猜测",
        "可能",
        "不确定",
    ];
    let uncertain_hits = uncertainty_markers
        .iter()
        .filter(|marker| lower.contains(**marker))
        .count();
    uncertain_hits == 0
}

fn find_existing_project_writeback_entry(
    store: &MemoryStore,
    source: &str,
) -> Option<AgentMemoryEntry> {
    store
        .recent(500)
        .ok()
        .unwrap_or_default()
        .into_iter()
        .find(|entry| entry.category == "project_memory" && entry.source.as_deref() == Some(source))
}

pub(super) enum ProjectWritebackUpsert {
    Saved,
    Updated,
    Unchanged,
}

pub(super) fn upsert_project_writeback_entry(
    store: &MemoryStore,
    source: &str,
    content: &str,
    tags: Vec<String>,
    priority: u8,
) -> Result<ProjectWritebackUpsert, String> {
    if let Some(existing) = find_existing_project_writeback_entry(store, source) {
        if existing.note.trim() == content.trim() {
            return Ok(ProjectWritebackUpsert::Unchanged);
        }
        let Some(id) = existing.id.as_deref() else {
            return Ok(ProjectWritebackUpsert::Unchanged);
        };
        execute_memory_update(&json!({
            "id": id,
            "content": content,
            "category": "project_memory",
            "tags": tags,
            "source": source,
            "priority": priority,
        }))?;
        // 同步向量索引：用最新内容覆盖向量库中同 id 的条目，避免 semantic search 拿到旧版本。
        let updated = AgentMemoryEntry {
            id: Some(id.to_string()),
            timestamp: Local::now().to_rfc3339(),
            category: "project_memory".to_string(),
            note: content.to_string(),
            tags: tags.clone(),
            source: Some(source.to_string()),
            priority: Some(priority),
            owner_pid: None,
            owner_pgid: None,
        };
        sync_agent_entry_to_vector(&updated);
        return Ok(ProjectWritebackUpsert::Updated);
    }

    let entry = AgentMemoryEntry {
        id: Some(format!("mem_{}", Uuid::new_v4().simple())),
        timestamp: Local::now().to_rfc3339(),
        category: "project_memory".to_string(),
        note: content.to_string(),
        tags,
        source: Some(source.to_string()),
        priority: Some(priority),
        owner_pid: None,
        owner_pgid: None,
    };
    store.append(&entry)?;
    // 同步向量索引：让 semantic search 能立刻召回新写入的项目记忆。
    sync_agent_entry_to_vector(&entry);
    Ok(ProjectWritebackUpsert::Saved)
}

pub(super) async fn run_project_knowledge_writeback_background(
    project_name: String,
    model: String,
    question: String,
    answer: String,
) {
    use tokio::time::{Duration, timeout};
    let cfg = configw::get_all_config();
    let timeout_ms = cfg
        .get_opt("ai.project_writeback.timeout_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(3000);
    let system = "You extract durable project knowledge for future turns.\nReturn STRICT JSON ONLY.\nSchema:\n{\"writeback\":true|false,\"content\":\"...\",\"tags\":[\"...\"],\"priority\":180}\nRules:\n- writeback=true ONLY if the answer contains stable, project-specific facts useful for future Q&A.\n- Focus on repository structure, module responsibilities, build/test workflow, conventions, and architecture.\n- `content` must be 2-6 concise bullet lines of factual memory, no markdown fences.\n- Use only facts explicitly stated in the ANSWER. Do not speculate or add anything new.\n- Exclude transient status, temporary debugging details, one-off user requests, and uncertain statements.\n- If the answer is incomplete, vague, or not worth remembering, return {\"writeback\":false}.";
    let user = format!(
        "PROJECT:\n{}\n\nQUESTION:\n{}\n\nANSWER:\n{}",
        project_name.trim(),
        question.trim(),
        answer.trim()
    );
    let messages = vec![
        json!({"role":"system","content":system}),
        json!({"role":"user","content":user}),
    ];
    let resp = match timeout(
        Duration::from_millis(timeout_ms),
        super::background::background_call(&model, &messages),
    )
    .await
    {
        Ok(v) => v,
        Err(_) => None,
    };
    let Some(resp) = resp else {
        return;
    };
    let content = super::background::extract_back_content(&resp).unwrap_or_default();
    let Some(payload) = parse_project_writeback_payload(&content) else {
        return;
    };
    if !project_writeback_quality_ok(&payload.content) {
        return;
    }

    let source = format!("auto_project_writeback:{project_name}");
    let store = MemoryStore::from_env_or_config();
    let mut tags = vec![
        "project".to_string(),
        "auto_writeback".to_string(),
        project_name.clone(),
    ];
    for tag in payload.tags {
        if !tags.iter().any(|existing| existing == &tag) {
            tags.push(tag);
        }
    }
    match upsert_project_writeback_entry(&store, &source, &payload.content, tags, payload.priority)
    {
        Ok(ProjectWritebackUpsert::Saved) => {
            println!(
                "[Memory] writeback saved project={} source={} category=project_memory priority={}",
                project_name, source, payload.priority
            );
            store.maintain_after_append();
        }
        Ok(ProjectWritebackUpsert::Updated) => {
            println!(
                "[Memory] writeback updated project={} source={} category=project_memory priority={}",
                project_name, source, payload.priority
            );
            store.maintain_after_append();
        }
        Ok(ProjectWritebackUpsert::Unchanged) => {}
        Err(_) => {}
    }
}
