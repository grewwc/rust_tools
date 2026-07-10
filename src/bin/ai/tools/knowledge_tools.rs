/// 知识库工具 — 为用户提供知识管理功能
///
/// ## 与 Memory 系统的区别
/// - **Knowledge (知识库)**:
///   用户主动保存的项目知识、决策记录、偏好设置等
///   - 工具: `knowledge_save`, `knowledge_forget`, `knowledge_search`, `knowledge_list`
///   - 用途: 用户显式管理的事实性知识，以及明确要求长期记住的原则/偏好/约束
///   - 类别: `user_memory`, `project_info`, `architecture`, `decision_log`,
///     `common_sense`, `coding_guideline`, `preference`, `user_preference`, `safety_rules`
///
/// - **Memory (记忆)**: Agent 内部自动学习的行为规则、安全策略、自我反思
///   - 工具: `memory_save`, `memory_search`, `memory_recent` (内部使用)
///   - 用途: Agent 自动积累的行为指导，如安全规则、编码规范、自我反思笔记
///   - 类别: `safety_rules`, `coding_guideline`, `self_note`, `common_sense` 等
///
/// 两者共享底层存储 (`MemoryStore`)，但用途和访问方式不同。
/// 知识库面向用户，记忆面向 Agent 自身。
///
/// 包含工具：
/// - `knowledge_save` — 保存用户知识（自动同步到 RAG 向量索引）
/// - `knowledge_forget` — 删除指定知识（同步删除 RAG 向量）
/// - `knowledge_search` — 按关键词搜索知识
/// - `knowledge_list` — 列出最近的知识条目（默认 20 条，最多 100）
/// - `knowledge_consolidate` — AI 驱动的记忆整理（读全部 → 分析 → 执行整理）
///
/// ## 记忆整理流程（`knowledge_consolidate`）
///
/// `knowledge_consolidate` 是一个**双阶段工具**，由 Agent 协调完成：
///
/// **第一阶段 — 读取全部**
/// ```
/// knowledge_consolidate(action: "read_all")
/// ```
/// 工具返回所有知识条目（id, category, tags, source, priority, content, timestamp），
/// Agent 据此分析哪些条目有用、哪些过时、哪些可以合并。
///
/// **第二阶段 — 执行整理**
/// ```
/// knowledge_consolidate(
///   action: "execute",
///   delete_ids: ["id1", "id2"],
///   save_entries: [{ content: "合并后的内容", category: "user_memory", ... }]
/// )
/// ```
/// 工具把删除和新增合成为一次原子重写，避免“先删后增”中途失败导致丢数。
///
/// ## 整理原则（Agent 自行判断）
/// - **保留**：仍有参考价值的决策、偏好、项目信息、安全规则
/// - **删除**：过时的记录、重复的内容、不再相关的临时信息、低优先级的琐碎笔记
/// - **合并**：多条相关的内容可以合并为一条概括性的条目
///
use serde_json::Value;

use crate::ai::knowledge::retrieval::recall::is_guideline_category;
use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;
use crate::ai::tools::service::memory::{
    MemoryOwnerScope, next_memory_id, prepare_memory_save_entry,
};
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use crate::ai::tools::storage::rag_store::{
    RagEntry, ensure_rag_store, get_rag_store, legacy_rag_id_for_memory_entry,
    legacy_rebuild_rag_id_for_memory_entry,
};
use chrono::Local;

fn rag_timestamp_for_entry(entry: &AgentMemoryEntry) -> u64 {
    chrono::DateTime::parse_from_rfc3339(&entry.timestamp)
        .map(|dt| dt.timestamp_millis().max(0) as u64)
        .unwrap_or_else(|_| entry.timestamp.parse().unwrap_or(0))
}

// ─── knowledge_save ──────────────────────────────────────────────────────────

fn params_knowledge_save() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Content to save to memory."
            },
            "category": {
                "type": "string",
                "description": "Category label (default: \"user_memory\"). Use `common_sense` / `coding_guideline` / `preference` / `user_preference` / `safety_rules` for durable principles or constraints; use `user_memory` / `project_info` / `architecture` / `decision_log` for factual knowledge."
            },
            "tags": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Optional tags for categorization."
            },
            "source": {
                "type": "string",
                "description": "Optional source context (e.g. user command, project name)."
            },
            "priority": {
                "type": "integer",
                "description": "Priority level 0-255. 255=permanent (never delete), 100-200=high, 50-99=normal, 0-49=low. Default follows category semantics: `user_memory`=150, `common_sense`/`coding_guideline`/`preference`/`user_preference`=210, `safety_rules`=255."
            }
        },
        "required": ["content"]
    })
}

fn execute_knowledge_save(args: &Value) -> Result<String, String> {
    let prepared = prepare_memory_save_entry(
        args,
        "user_memory",
        &[],
        "knowledge_save",
        MemoryOwnerScope::Global,
        "knowledge_save_downgraded",
    )?;
    let crate::ai::tools::service::memory::PreparedMemorySave {
        requested_category,
        downgraded,
        assessment: _assessment,
        entry,
    } = prepared;

    let store = MemoryStore::from_env_or_config();
    store.append(&entry)?;

    // Sync to RAG vector index. Failure here is non-fatal (the entry is already
    // persisted to the JSONL store above), but it must not be silent: if the
    // vector upsert fails the entry won't be reachable via semantic search, so
    // surface a warning to the model rather than pretending it succeeded.
    let rag_warning = sync_entry_to_rag(&entry);

    let mut result = format!(
        "Saved to knowledge [{}]:\n  {}\n",
        entry.category, entry.note
    );
    if let Some(id) = entry.id.as_deref() {
        result.push_str(&format!("  ID: {}\n", id));
    }
    if !entry.tags.is_empty() {
        result.push_str(&format!("  Tags: {}\n", entry.tags.join(", ")));
    }
    if let Some(src) = &entry.source {
        result.push_str(&format!("  Source: {}\n", src));
    }
    result.push_str(&format!("  Priority: {}\n", entry.priority.unwrap_or(100)));
    if downgraded && entry.category == "self_note" {
        result.push_str(&format!(
            "  Note: requested category '{}' was downgraded to 'self_note' because the content is too generic for durable principle recall.\n",
            requested_category
        ));
        result.push_str(
              "  Saved as short-term self_note; it will not enter persistent guideline recall until the note is more specific and actionable.\n",
        );
    } else if is_guideline_category(&entry.category) {
        result.push_str("  This principle will participate in persistent guideline recall.\n");
    } else {
        result.push_str(
            "  The agent will automatically check this knowledge in future conversations.\n",
        );
    }

    if let Some(warning) = rag_warning {
        result.push_str(&format!(
            "  ⚠️ Warning: vector index sync failed ({warning}); this entry may not be retrievable via semantic knowledge_search until re-synced.\n"
        ));
    }

    Ok(result)
}

/// 把知识条目同步进 RAG 向量索引。成功返回 `None`；任一步骤失败返回
/// `Some(reason)`，由调用方决定如何提示模型。条目此前已落 JSONL 主存储，
/// 因此这里的失败是非致命的，但不能静默——否则模型会误以为该条目可被语义检索。
fn sync_entry_to_rag(entry: &AgentMemoryEntry) -> Option<String> {
    let Some(id) = entry.id.clone() else {
        // 没有 id 的条目无法进 RAG（也无法后续定位），但这属于上游生成问题，
        // 不在此处报错，交由主存储语义处理。
        return None;
    };
    if let Err(e) = ensure_rag_store() {
        return Some(format!("rag store unavailable: {e}"));
    }
    let guard = match get_rag_store() {
        Ok(guard) => guard,
        Err(e) => return Some(format!("rag store lock poisoned: {e}")),
    };
    let Some(rag) = guard.as_ref() else {
        return Some("rag store not initialized".to_string());
    };
    let embedding_text = format!("{}: {}", entry.category, entry.note);
    let embedding = match rag.embed_texts(&[embedding_text.clone()]) {
        Ok(embeddings) => match embeddings.into_iter().next() {
            Some(embedding) => embedding,
            None => return Some("embedding provider returned no vector".to_string()),
        },
        Err(e) => return Some(format!("embedding failed: {e}")),
    };
    if let Err(e) = rag.upsert(RagEntry {
        id,
        content: embedding_text,
        category: entry.category.clone(),
        tags: entry.tags.clone(),
        embedding,
        timestamp: rag_timestamp_for_entry(entry),
    }) {
        return Some(format!("vector upsert failed: {e}"));
    }
    None
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "knowledge_save",
        description: "Save user-directed content to the global knowledge base with optional category and tags. Use guideline categories like `common_sense`, `coding_guideline`, `preference`, `user_preference`, or `safety_rules` for durable principles/constraints so they participate in persistent recall.",
        parameters: params_knowledge_save,
        execute: execute_knowledge_save,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

// ─── knowledge_forget ────────────────────────────────────────────────────────

fn params_knowledge_forget() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Memory entry id to delete. Use knowledge_list or knowledge_search first to find the id."
            }
        },
        "required": ["id"]
    })
}

fn execute_knowledge_forget(args: &Value) -> Result<String, String> {
    let id = args["id"]
        .as_str()
        .ok_or("Missing 'id'. Provide the memory entry id to forget.")?;

    let store = MemoryStore::from_env_or_config();

    // Delete from memory_store
    let deleted = store
        .delete_by_id(id)
        .map_err(|e| format!("Failed to delete memory entry: {e}"))?
        .ok_or_else(|| {
            format!(
                "No memory entry found with id '{}'. Use knowledge_list to find valid ids.",
                id
            )
        })?;

    let preview: String = if deleted.note.chars().count() > 100 {
        let s: String = deleted.note.chars().take(100).collect();
        format!("{}...", s)
    } else {
        deleted.note.clone()
    };

    // Sync delete from RAG vector index
    if let Ok(_) = ensure_rag_store() {
        if let Ok(guard) = get_rag_store() {
            if let Some(rag) = guard.as_ref() {
                let rag_id = deleted
                    .id
                    .clone()
                    .unwrap_or_else(|| legacy_rag_id_for_memory_entry(&deleted));
                let _ = rag.delete(&rag_id);
                let legacy_ids = [
                    legacy_rag_id_for_memory_entry(&deleted),
                    legacy_rebuild_rag_id_for_memory_entry(&deleted),
                ];
                for legacy_id in legacy_ids {
                    if legacy_id != rag_id {
                        let _ = rag.delete(&legacy_id);
                    }
                }
            }
        }
    }

    Ok(format!(
        "Forgotten knowledge entry:\n  id: {}\n  category: {}\n  content: {}\n  (Also removed from RAG vector index)",
        id, deleted.category, preview
    ))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "knowledge_forget",
        description: "Delete a knowledge entry by id. Use knowledge_list or knowledge_search first to find the id.",
        parameters: params_knowledge_forget,
        execute: execute_knowledge_forget,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin"],
    }
});

// ─── knowledge_search ────────────────────────────────────────────────────────

fn params_knowledge_search() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query."
            },
            "category": {
                "type": "string",
                "description": "Filter by category."
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default: 10)."
            }
        },
        "required": ["query"]
    })
}

fn execute_knowledge_search(args: &Value) -> Result<String, String> {
    let query = args["query"]
        .as_str()
        .ok_or("Missing 'query'. Provide a search term.")?;
    let category = args["category"].as_str();
    let limit = args["limit"].as_u64().map(|v| v as usize).unwrap_or(10);

    let store = MemoryStore::from_env_or_config();
    let search_query = if let Some(cat) = category {
        format!("{} {}", query, cat)
    } else {
        query.to_string()
    };

    let results = store.search(&search_query, limit)?;
    let entries: Vec<_> = results.into_iter().map(|(e, _score)| e).collect();

    if entries.is_empty() {
        return Ok(format!("No knowledge entries found for query: '{}'", query));
    }

    let mut result = format!(
        "🔍 Search results for '{}' ({} found):\n\n",
        query,
        entries.len()
    );
    for (idx, entry) in entries.iter().enumerate() {
        let entry_id = entry.id.as_deref().unwrap_or("N/A");
        result.push_str(&format!(
            "{}. [id:{}] [{}] {}\n",
            idx + 1,
            entry_id,
            entry.category,
            entry.note
        ));
        if !entry.tags.is_empty() {
            result.push_str(&format!("   Tags: {}\n", entry.tags.join(", ")));
        }
        if let Some(src) = &entry.source {
            result.push_str(&format!("   Source: {}\n", src));
        }
        result.push_str(&format!("   Priority: {}\n", entry.priority.unwrap_or(100)));
        result.push('\n');
    }

    Ok(result)
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "knowledge_search",
        description: "Search knowledge base entries by keyword. Returns matching entries with their ids.",
        parameters: params_knowledge_search,
        execute: execute_knowledge_search,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::Spawnable,
        groups: &["builtin", "core"],
    }
});

// ─── knowledge_list ──────────────────────────────────────────────────────────

fn params_knowledge_list() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max entries to show (default: 20, max: 100)."
            }
        }
    })
}

fn execute_knowledge_list(args: &Value) -> Result<String, String> {
    let limit = args["limit"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or(20)
        .min(100);

    let store = MemoryStore::from_env_or_config();
    let path = store.path().to_path_buf();

    use std::fs;

    if !path.exists() {
        return Ok("Knowledge base is empty. Use knowledge_save to add entries.".to_string());
    }

    let content =
        fs::read_to_string(&path).map_err(|e| format!("Failed to read memory file: {e}"))?;

    let mut entries: Vec<AgentMemoryEntry> = Vec::new();
    let mut skipped_lines = 0usize;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<AgentMemoryEntry>(line) {
            Ok(entry) => entries.push(entry),
            // 坏行不能静默吞掉——否则模型会把"部分列表"当成"完整列表"。累计计数后
            // 在结果里明确提示，避免其基于不完整数据做决策。
            Err(_) => skipped_lines += 1,
        }
    }

    if entries.is_empty() {
        if skipped_lines > 0 {
            return Ok(format!(
                "Knowledge base has no readable entries, but {skipped_lines} line(s) were unparseable and skipped. The memory file may be corrupted."
            ));
        }
        return Ok("Knowledge base is empty. Use knowledge_save to add entries.".to_string());
    }

    // Show most recent first
    let start = entries.len().saturating_sub(limit);
    let recent: &[AgentMemoryEntry] = &entries[start..];

    let mut result = format!(
        "📋 Recent knowledge entries (showing {} of {}):\n\n",
        recent.len(),
        entries.len()
    );
    for entry in recent.iter().rev() {
        let entry_id = entry.id.as_deref().unwrap_or("N/A");
        let preview: String = if entry.note.chars().count() > 120 {
            let s: String = entry.note.chars().take(120).collect();
            format!("{}...", s)
        } else {
            entry.note.clone()
        };
        result.push_str(&format!(
            "• [id:{}] [{}] {}\n",
            entry_id, entry.category, preview
        ));
        if !entry.tags.is_empty() {
            result.push_str(&format!("  Tags: {}\n", entry.tags.join(", ")));
        }
        result.push_str(&format!(
            "  Priority: {}\n\n",
            entry.priority.unwrap_or(100)
        ));
    }

    if skipped_lines > 0 {
        result.push_str(&format!(
            "\n⚠️ Note: {skipped_lines} line(s) in the memory file were unparseable and skipped; this listing may be incomplete.\n"
        ));
    }

    Ok(result)
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "knowledge_list",
        description: "List recent knowledge base entries. Shows id, category, and content preview.",
        parameters: params_knowledge_list,
        execute: execute_knowledge_list,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::Spawnable,
        groups: &["builtin", "core"],
    }
});

// ─── knowledge_consolidate ─────────────────────────────────────────────────

fn params_knowledge_consolidate() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read_all", "execute"],
                "description": "\"read_all\" returns every knowledge entry for analysis; \"execute\" deletes & saves entries per your consolidation plan."
            },
            "delete_ids": {
                "type": "array",
                "items": { "type": "string" },
                "description": "List of entry IDs to delete (only used when action=\"execute\")."
            },
            "save_entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": { "type": "string", "description": "The consolidated note content." },
                        "category": { "type": "string", "description": "Category (default: user_memory)." },
                        "tags": { "type": "array", "items": { "type": "string" } },
                        "source": { "type": "string" },
                        "priority": { "type": "integer", "description": "0-255. 255=permanent." }
                    },
                    "required": ["content"]
                },
                "description": "New entries to save (only used when action=\"execute\"). Saved after deletions."
            }
        },
        "required": ["action"]
    })
}

fn execute_knowledge_consolidate(args: &Value) -> Result<String, String> {
    let action = args["action"].as_str().ok_or("Missing 'action'.")?;

    match action {
        "read_all" => read_all_entries(),
        "execute" => execute_consolidation(args),
        _ => Err(format!(
            "Unknown action '{}'. Use 'read_all' or 'execute'.",
            action
        )),
    }
}

fn read_all_entries() -> Result<String, String> {
    let store = MemoryStore::from_env_or_config();
    let entries = store.all()?;

    if entries.is_empty() {
        return Ok("Knowledge base is empty. Nothing to consolidate.".to_string());
    }

    let mut result = format!("📚 Total knowledge entries: {}\n\n", entries.len());

    // Group by category for better readability
    let mut by_category: std::collections::BTreeMap<String, Vec<&AgentMemoryEntry>> =
        std::collections::BTreeMap::new();
    for entry in &entries {
        by_category
            .entry(entry.category.clone())
            .or_default()
            .push(entry);
    }

    for (cat, cat_entries) in &by_category {
        result.push_str(&format!(
            "── [{}] ({} entries) ──\n",
            cat,
            cat_entries.len()
        ));
        for entry in cat_entries {
            let entry_id = entry.id.as_deref().unwrap_or("N/A");
            let ts = &entry.timestamp;
            let prio = entry.priority.unwrap_or(100);
            // 截断过长的内容用于预览
            let preview: String = if entry.note.chars().count() > 200 {
                entry.note.chars().take(200).collect::<String>()
                    + &format!(
                        "\n       …[truncated, total {} chars]",
                        entry.note.chars().count()
                    )
            } else {
                entry.note.clone()
            };

            result.push_str(&format!("   id: {}\n", entry_id));
            if !entry.tags.is_empty() {
                result.push_str(&format!("   tags: {}\n", entry.tags.join(", ")));
            }
            if let Some(src) = &entry.source {
                result.push_str(&format!("   source: {}\n", src));
            }
            result.push_str(&format!("   priority: {}  |  timestamp: {}\n", prio, ts));
            result.push_str(&format!("   content: {}\n\n", preview));
        }
    }

    result.push_str("──\n");
    result.push_str("Analysis: Review the entries above. Identify which to keep, delete (obsolete/duplicate), or merge into consolidated summaries.\n");
    result
        .push_str("Then call knowledge_consolidate with action=\"execute\" to apply your plan.\n");

    Ok(result)
}

fn execute_consolidation(args: &Value) -> Result<String, String> {
    let store = MemoryStore::from_env_or_config();

    // 收集待删除的 ID
    let delete_ids: Vec<String> = args["delete_ids"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    // 收集待保存的新条目
    let save_entries_raw: Vec<&Value> = args["save_entries"]
        .as_array()
        .map(|arr| arr.iter().collect())
        .unwrap_or_default();

    if delete_ids.is_empty() && save_entries_raw.is_empty() {
        return Err("No changes specified. Provide delete_ids, save_entries, or both.".to_string());
    }

    let mut report = String::new();
    report.push_str("📦 Consolidation report:\n");

    // 构建待保存的新条目
    let mut new_entries: Vec<AgentMemoryEntry> = Vec::new();
    for entry_val in &save_entries_raw {
        let content = entry_val["content"]
            .as_str()
            .ok_or("Each save_entries item must have a 'content' field.")?;
        let category = entry_val["category"].as_str().unwrap_or("user_memory");
        let tags: Vec<String> = entry_val["tags"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let source = entry_val["source"].as_str().map(String::from);
        let priority = entry_val["priority"]
            .as_u64()
            .map(|p| p as u8)
            .unwrap_or(150);

        new_entries.push(AgentMemoryEntry {
            id: Some(next_memory_id()),
            timestamp: Local::now().to_rfc3339(),
            category: category.to_string(),
            note: content.to_string(),
            tags,
            source,
            priority: Some(priority),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        });
    }

    let id_refs: Vec<&str> = delete_ids.iter().map(String::as_str).collect();
    let (deleted_count, saved_count) = match store.apply_batch_update(&id_refs, &new_entries) {
        Ok(result) => {
            if !delete_ids.is_empty() {
                report.push_str(&format!("   Deleted: {} entries\n", result.deleted));
            }
            if !new_entries.is_empty() {
                report.push_str(&format!("   Saved: {} new entries\n", result.appended));
            }
            // Sync RAG vector index: delete vectors for removed ids, upsert embeddings for new entries.
            // apply_batch_update only rewrites JSONL + rebuilds SQLite/FTS5, so the vector store
            // must be synced separately to avoid orphaned/missing embeddings.
            if let Ok(_) = ensure_rag_store() {
                if let Ok(guard) = get_rag_store() {
                    if let Some(rag) = guard.as_ref() {
                        for id in &delete_ids {
                            let _ = rag.delete(id);
                        }
                        for entry in &new_entries {
                            if let Some(id) = entry.id.clone() {
                                let embedding_text = format!("{}: {}", entry.category, entry.note);
                                if let Ok(embeddings) = rag.embed_texts(&[embedding_text.clone()]) {
                                    if let Some(embedding) = embeddings.into_iter().next() {
                                        let _ = rag.upsert(RagEntry {
                                            id,
                                            content: embedding_text,
                                            category: entry.category.clone(),
                                            tags: entry.tags.clone(),
                                            embedding,
                                            timestamp: rag_timestamp_for_entry(entry),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
            (result.deleted, result.appended)
        }
        Err(e) => {
            report.push_str(&format!("   Consolidation error: {}\n", e));
            (0, 0)
        }
    };

    if deleted_count == 0 && saved_count == 0 {
        report.push_str("   No changes were made.\n");
    }

    Ok(report)
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "knowledge_consolidate",
        description: "Two-phase knowledge consolidation. First call with action=\"read_all\" to get all entries; then call with action=\"execute\", delete_ids=[...], and/or save_entries=[...] to apply a consolidation plan. The agent analyzes which entries are useful, obsolete, or mergeable.",
        parameters: params_knowledge_consolidate,
        execute: execute_knowledge_consolidate,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin"],
    }
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::test_support::ENV_LOCK;
    use crate::ai::tools::storage::memory_store::AgentMemoryEntry;
    use std::path::Path;
    use std::sync::MutexGuard;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn env_lock_guard() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
    }

    fn cleanup_memory_artifacts(path: &Path) {
        let db_path = path.with_extension("db");
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_file(db_path.with_extension("db-wal"));
        let _ = std::fs::remove_file(db_path.with_extension("db-shm"));
    }

    fn read_entries(path: &Path) -> Vec<AgentMemoryEntry> {
        std::fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str::<AgentMemoryEntry>(line.trim()).ok())
            .collect()
    }

    #[test]
    fn test_knowledge_save_params() {
        let params = params_knowledge_save();
        assert!(
            params["required"]
                .as_array()
                .unwrap()
                .contains(&Value::String("content".to_string()))
        );
        assert!(params["properties"]["content"].is_object());
    }

    #[test]
    fn test_knowledge_forget_params() {
        let params = params_knowledge_forget();
        assert!(
            params["required"]
                .as_array()
                .unwrap()
                .contains(&Value::String("id".to_string()))
        );
    }

    #[test]
    fn test_knowledge_search_params() {
        let params = params_knowledge_search();
        assert!(
            params["required"]
                .as_array()
                .unwrap()
                .contains(&Value::String("query".to_string()))
        );
    }

    #[test]
    fn test_knowledge_list_params() {
        let params = params_knowledge_list();
        // limit is optional, so no required
        assert!(
            params["required"]
                .as_array()
                .map(|a| a.is_empty())
                .unwrap_or(true)
        );
    }

    #[test]
    fn test_knowledge_consolidate_params() {
        let params = params_knowledge_consolidate();
        assert!(
            params["required"]
                .as_array()
                .unwrap()
                .contains(&Value::String("action".to_string()))
        );
        assert_eq!(
            params["properties"]["action"]["enum"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        let actions: Vec<&str> = params["properties"]["action"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(actions.contains(&"read_all"));
        assert!(actions.contains(&"execute"));
    }

    #[test]
    fn test_knowledge_save_assigns_id_and_forget_can_delete_it() {
        let _guard = env_lock_guard();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_knowledge_save_forget_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let save_msg = execute_knowledge_save(&serde_json::json!({
            "content": "Always confirm before deleting user files or sessions.",
            "category": "common_sense",
            "tags": ["safety", "principle"]
        }))
        .unwrap();
        assert!(save_msg.contains("ID: mem_"));

        let entries = read_entries(&path);
        assert_eq!(entries.len(), 1);
        let id = entries[0]
            .id
            .clone()
            .expect("knowledge_save should assign id");

        let forget_msg = execute_knowledge_forget(&serde_json::json!({ "id": id })).unwrap();
        assert!(forget_msg.contains("Forgotten knowledge entry"));
        assert!(read_entries(&path).is_empty());

        cleanup_memory_artifacts(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    #[test]
    fn test_knowledge_save_guideline_category_uses_persistent_priority() {
        let _guard = env_lock_guard();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_knowledge_save_guideline_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let save_msg = execute_knowledge_save(&serde_json::json!({
            "content": "Do: require explicit confirmation before destructive file operations.",
            "category": "common_sense"
        }))
        .unwrap();
        assert!(save_msg.contains("persistent guideline recall"));

        let entries = read_entries(&path);
        assert_eq!(entries.len(), 1);
        assert!(
            entries[0]
                .id
                .as_deref()
                .is_some_and(|id| id.starts_with("mem_"))
        );
        assert_eq!(entries[0].category, "common_sense");
        assert_eq!(entries[0].priority, Some(210));

        cleanup_memory_artifacts(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    #[test]
    fn test_knowledge_save_downgrades_low_signal_long_term_entry() {
        let _guard = env_lock_guard();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_knowledge_save_downgrade_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let save_msg = execute_knowledge_save(&serde_json::json!({
            "content": "be careful",
            "category": "common_sense"
        }))
        .unwrap();
        assert!(save_msg.contains("downgraded to 'self_note'"));

        let entries = read_entries(&path);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].category, "self_note");
        assert_eq!(entries[0].priority, Some(120));
        assert!(entries[0].tags.iter().any(|tag| tag == "auto_downgraded"));

        cleanup_memory_artifacts(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    #[test]
    fn test_execute_consolidation_applies_delete_and_save_atomically() {
        let _guard = env_lock_guard();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_knowledge_consolidate_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let seed = [
            AgentMemoryEntry {
                id: Some("mem_old_1".to_string()),
                timestamp: "2025-01-01T00:00:00Z".to_string(),
                category: "user_memory".to_string(),
                note: "old memo 1".to_string(),
                tags: vec!["old".to_string()],
                source: None,
                priority: Some(150),
                owner_pid: None,
                owner_pgid: None,
                image_path: None,
            },
            AgentMemoryEntry {
                id: Some("mem_old_2".to_string()),
                timestamp: "2025-01-01T00:00:01Z".to_string(),
                category: "user_memory".to_string(),
                note: "old memo 2".to_string(),
                tags: vec!["old".to_string()],
                source: None,
                priority: Some(150),
                owner_pid: None,
                owner_pgid: None,
                image_path: None,
            },
        ];
        let mut buf = String::new();
        for entry in &seed {
            buf.push_str(&serde_json::to_string(entry).unwrap());
            buf.push('\n');
        }
        std::fs::write(&path, buf).unwrap();

        let report = execute_consolidation(&serde_json::json!({
            "delete_ids": ["mem_old_1", "mem_old_2"],
            "save_entries": [{
                "content": "merged memo",
                "category": "user_memory",
                "tags": ["consolidated"],
                "priority": 150
            }]
        }))
        .unwrap();

        assert!(report.contains("Deleted: 2 entries"));
        assert!(report.contains("Saved: 1 new entries"));

        let entries: Vec<AgentMemoryEntry> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .filter_map(|line| serde_json::from_str::<AgentMemoryEntry>(line.trim()).ok())
            .collect();
        assert_eq!(entries.len(), 1);
        assert!(
            entries[0]
                .id
                .as_deref()
                .is_some_and(|id| id.starts_with("mem_"))
        );
        assert_eq!(entries[0].note, "merged memo");
        assert_eq!(entries[0].category, "user_memory");

        cleanup_memory_artifacts(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }
}
