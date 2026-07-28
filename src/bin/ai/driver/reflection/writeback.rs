use serde_json::json;
use uuid::Uuid;

use crate::ai::tools::service::memory::execute_memory_update;
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use chrono::Local;

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
        image_path: entry.image_path.clone(),
    };
    let guard = match crate::ai::tools::storage::rag_store::get_rag_store() {
        Ok(g) => g,
        Err(_) => return,
    };
    let Some(store) = guard.as_ref() else {
        return;
    };
    if let Err(err) =
        crate::ai::knowledge::sync::knowledge_sync::sync_entry_to_vector(store, &knowledge_entry)
    {
        eprintln!("[Memory] writeback vector sync failed: {}", err);
    }
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
    if uncertain_hits > 0 {
        return false;
    }
    if crate::ai::knowledge::entry::note_has_local_env_path_leak(trimmed) {
        return false;
    }

    super::assess_learning_note_quality(trimmed).high_quality
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
    Rejected,
}

pub(super) fn upsert_project_writeback_entry(
    store: &MemoryStore,
    source: &str,
    content: &str,
    tags: Vec<String>,
    priority: u8,
) -> Result<ProjectWritebackUpsert, String> {
    if !project_writeback_quality_ok(content) {
        return Ok(ProjectWritebackUpsert::Rejected);
    }

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
            image_path: None,
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
        image_path: None,
    };
    store.append(&entry)?;
    // 同步向量索引：让 semantic search 能立刻召回新写入的项目记忆。
    sync_agent_entry_to_vector(&entry);
    Ok(ProjectWritebackUpsert::Saved)
}
