/// 知识库工具 — 为用户提供知识管理功能
///
/// ## 与 Memory 系统的区别
/// - **Knowledge (知识库)**: 用户主动保存的项目知识、决策记录、偏好设置等
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
/// - `knowledge_list` — 列出最近的知识条目
use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use crate::ai::tools::storage::rag_store::{ensure_rag_store, get_rag_store, RagEntry};

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
    };

    let store = MemoryStore::from_env_or_config();
    store.append(&entry)?;

    // Sync to RAG vector index
    if let Ok(_) = ensure_rag_store() {
        if let Ok(guard) = get_rag_store() {
            if let Some(rag) = guard.as_ref() {
                let id = format!(
                    "{:x}",
                    md5::compute(&format!("{}:{}", entry.timestamp, content))
                );
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
        groups: &["builtin"],
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
                let rag_id = format!(
                    "{:x}",
                    md5::compute(&format!("{}:{}", deleted.timestamp, deleted.note))
                );
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
        description:
            "Search knowledge base entries by keyword. Returns matching entries with their ids.",
        parameters: params_knowledge_search,
        execute: execute_knowledge_search,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::Spawnable,
        groups: &["builtin"],
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
        groups: &["builtin"],
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_save_params() {
        let params = params_knowledge_save();
        assert!(params["required"]
            .as_array()
            .unwrap()
            .contains(&Value::String("content".to_string())));
        assert!(params["properties"]["content"].is_object());
    }

    #[test]
    fn test_knowledge_forget_params() {
        let params = params_knowledge_forget();
        assert!(params["required"]
            .as_array()
            .unwrap()
            .contains(&Value::String("id".to_string())));
    }

    #[test]
    fn test_knowledge_search_params() {
        let params = params_knowledge_search();
        assert!(params["required"]
            .as_array()
            .unwrap()
            .contains(&Value::String("query".to_string())));
    }

    #[test]
    fn test_knowledge_list_params() {
        let params = params_knowledge_list();
        // limit is optional, so no required
        assert!(params["required"]
            .as_array()
            .map(|a| a.is_empty())
            .unwrap_or(true));
    }
}
