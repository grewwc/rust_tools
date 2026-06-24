/// 知识库工具 — 为用户提供知识管理功能
///
/// ## 与 Memory 系统的区别
/// - **Knowledge (知识库)**:
///   用户主动保存的项目知识、决策记录、偏好设置等
///   - 工具: `knowledge_save`, `knowledge_forget`, `knowledge_search`, `knowledge_list`
///   - 用途: 用户显式管理的事实性知识，如项目结构、技术决策、用户偏好
///   - 类别: `user_memory`, `project_info`, `architecture`, `decision_log` 等
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
/// 工具先批量删除指定 ID 的条目，再批量保存新条目（原子操作顺序：先删后增）。
///
/// ## 整理原则（Agent 自行判断）
/// - **保留**：仍有参考价值的决策、偏好、项目信息、安全规则
/// - **删除**：过时的记录、重复的内容、不再相关的临时信息、低优先级的琐碎笔记
/// - **合并**：多条相关的内容可以合并为一条概括性的条目
///
use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use crate::ai::tools::storage::rag_store::{RagEntry, ensure_rag_store, get_rag_store};
use chrono::Local;

/// 32 字符短指纹（取 SHA-256 前 16 字节）。
fn short_rag_id(bytes: &[u8]) -> String {
    let digest = <sha2::Sha256 as sha2::Digest>::digest(bytes);
    let mut s = String::with_capacity(32);
    for b in &digest[..16] {
        s.push_str(&format!("{:02x}", b));
    }
    s
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
                "description": "Category label (default: \"user_memory\")."
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
                "description": "Priority level 0-255. 255=permanent (never delete), 100-200=high, 50-99=normal, 0-49=low. Default: 150 for user-directed memory."
            }
        },
        "required": ["content"]
    })
}

fn execute_knowledge_save(args: &Value) -> Result<String, String> {
    let content = args["content"]
        .as_str()
        .ok_or("Missing 'content'. Provide the text to remember.")?;
    let category = args["category"].as_str().unwrap_or("user_memory");
    let tags: Vec<String> = args["tags"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let source = args["source"].as_str().map(String::from);
    let priority = args["priority"].as_u64().map(|p| p as u8).unwrap_or(150);

    let now = chrono::Local::now().to_rfc3339();

    let entry = AgentMemoryEntry {
        id: None,
        timestamp: now,
        category: category.to_string(),
        note: content.to_string(),
        tags,
        source,
        priority: Some(priority),
        owner_pid: None,
        owner_pgid: None,
        image_path: None,
    };

    let store = MemoryStore::from_env_or_config();
    store.append(&entry)?;

    // Sync to RAG vector index
    if let Ok(_) = ensure_rag_store() {
        if let Ok(guard) = get_rag_store() {
            if let Some(rag) = guard.as_ref() {
                let id = short_rag_id(format!("{}:{}", entry.timestamp, content).as_bytes());
                let embedding_text = format!("{}: {}", entry.category, entry.note);
                if let Ok(embeddings) = rag.embed_texts(&[embedding_text.clone()]) {
                    if let Some(embedding) = embeddings.into_iter().next() {
                        let _ = rag.upsert(RagEntry {
                            id,
                            content: embedding_text,
                            category: entry.category.clone(),
                            tags: entry.tags.clone(),
                            embedding,
                            timestamp: entry.timestamp.parse().unwrap_or(0),
                        });
                    }
                }
            }
        }
    }

    let mut result = format!("Saved to knowledge [{}]:\n  {}\n", category, content);
    if !entry.tags.is_empty() {
        result.push_str(&format!("  Tags: {}\n", entry.tags.join(", ")));
    }
    if let Some(src) = &entry.source {
        result.push_str(&format!("  Source: {}\n", src));
    }
    result.push_str(&format!("  Priority: {}\n", entry.priority.unwrap_or(100)));
    result.push_str("The agent will automatically check this knowledge in future conversations.");

    Ok(result)
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "knowledge_save",
        description: "Save user-directed content to the global knowledge base with optional category and tags. The agent automatically checks the knowledge base at the start of each conversation.",
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
                let rag_id =
                    short_rag_id(format!("{}:{}", deleted.timestamp, deleted.note).as_bytes());
                let _ = rag.delete(&rag_id);
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
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(line) {
            entries.push(entry);
        }
    }

    if entries.is_empty() {
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

    // 第一阶段：批量删除
    let deleted_count = if !delete_ids.is_empty() {
        let id_refs: Vec<&str> = delete_ids.iter().map(String::as_str).collect();
        match store.delete_by_ids(&id_refs) {
            Ok(n) => {
                report.push_str(&format!("   Deleted: {} entries\n", n));
                n
            }
            Err(e) => {
                report.push_str(&format!("   Delete error: {}\n", e));
                0
            }
        }
    } else {
        0
    };

    // 第二阶段：构建新条目
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
            id: None,
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

    // 第三阶段：批量保存新条目
    let saved_count = if !new_entries.is_empty() {
        match store.append_batch(&new_entries) {
            Ok(n) => {
                report.push_str(&format!("   Saved: {} new entries\n", n));
                n
            }
            Err(e) => {
                report.push_str(&format!("   Save error: {}\n", e));
                0
            }
        }
    } else {
        0
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
}
