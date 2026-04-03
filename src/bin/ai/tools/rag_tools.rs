/// RAG 语义检索工具 — 为 agent 提供向量语义搜索能力
///
/// 包含两个工具：
/// - `knowledge_semantic_search` — 语义相似度搜索（向量检索）
/// - `knowledge_rebuild_index` — 重建向量索引（从 memory_store 同步）

use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;
use crate::ai::tools::storage::rag_store::{ensure_rag_store, get_rag_store, RagStore};

// ─── knowledge_semantic_search ───────────────────────────────────────────────

fn params_semantic_search() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (semantic). The search understands meaning, not just keywords."
            },
            "category": {
                "type": "string",
                "description": "Filter by category (optional)."
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default: 5)."
            },
            "hybrid": {
                "type": "boolean",
                "description": "Use hybrid search (BM25 + vector). Default: true."
            }
        },
        "required": ["query"]
    })
}

fn execute_semantic_search(args: &Value) -> Result<String, String> {
    ensure_rag_store()?;
    let guard = get_rag_store()?;
    let store = guard.as_ref().ok_or("RAG store not initialized")?;

    let query = args["query"].as_str()
        .ok_or("Missing 'query'. Provide a semantic search query.")?;
    let category = args["category"].as_str();
    let limit = args["limit"].as_u64().map(|v| v as usize).unwrap_or(5);
    let use_hybrid = args["hybrid"].as_bool().unwrap_or(true);

    if use_hybrid {
        // 混合搜索：需要先获取 BM25 结果
        let bm25_results = hybrid_bm25_fallback(store, query, category)?;
        let results = store.hybrid_search(query, bm25_results, limit, category, 0.4)?;

        if results.is_empty() {
            return Ok(format!("🔍 No results found for: '{}'", query));
        }

        let mut output = format!("🧠 Semantic search results for '{}':\n\n", query);
        for (idx, (_id, entry, score)) in results.iter().enumerate() {
            output.push_str(&format!(
                "{}. [score: {:.3}] [{}] {}\n",
                idx + 1,
                score,
                entry.category,
                entry.content
            ));
            if !entry.tags.is_empty() {
                output.push_str(&format!("   Tags: {}\n", entry.tags.join(", ")));
            }
            output.push_str(&format!("   Vector ID: {}\n\n", entry.id));
        }
        Ok(output)
    } else {
        // 纯语义搜索
        let results = store.semantic_search(query, limit, category)?;

        if results.is_empty() {
            return Ok(format!("🔍 No results found for: '{}'", query));
        }

        let mut output = format!("🧠 Semantic search results for '{}':\n\n", query);
        for (idx, (entry, score)) in results.iter().enumerate() {
            output.push_str(&format!(
                "{}. [score: {:.3}] [{}] {}\n",
                idx + 1,
                score,
                entry.category,
                entry.content
            ));
            if !entry.tags.is_empty() {
                output.push_str(&format!("   Tags: {}\n", entry.tags.join(", ")));
            }
            output.push_str(&format!("   Vector ID: {}\n\n", entry.id));
        }
        Ok(output)
    }
}

/// BM25 回退 — 用 memory_store 的 BM25 搜索结果作为 hybrid 输入
fn hybrid_bm25_fallback(
    store: &RagStore,
    query: &str,
    _category: Option<&str>,
) -> Result<Vec<(String, f32)>, String> {
    use crate::ai::tools::storage::memory_store::MemoryStore;

    let mem_store = MemoryStore::from_env_or_config();
    let mem_results = mem_store.search(query, 20)?;

    let mut results = Vec::new();
    for (rank, entry) in mem_results.into_iter().enumerate() {
        let id = entry.id.clone().unwrap_or_else(|| {
            format!("{:x}", md5::compute(&entry.note))
        });

        // 检查向量索引中是否存在该条目
        if store.get_entry(&id)?.is_some() {
            // 用排名位置做粗略分数（排名越前分数越高）
            let pseudo_score = (1.0 - (rank as f32 * 0.05)).max(0.1);
            results.push((id, pseudo_score));
        }
    }
    Ok(results)
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "knowledge_semantic_search",
        description: "Search knowledge base using semantic (vector) similarity. Understands meaning beyond keywords. Use this when keyword search fails to find relevant results.",
        parameters: params_semantic_search,
        execute: execute_semantic_search,
        groups: &["builtin"],
    }
});

// ─── knowledge_rebuild_index ─────────────────────────────────────────────────

fn params_rebuild_index() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {}
    })
}

fn execute_rebuild_index(_args: &Value) -> Result<String, String> {
    ensure_rag_store()?;
    let guard = get_rag_store()?;
    let store = guard.as_ref().ok_or("RAG store not initialized")?;

    let count = store.rebuild_from_memory()?;
    Ok(format!("✅ Rebuilt RAG index: {} entries vectorized.", count))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "knowledge_rebuild_index",
        description: "Rebuild the vector index from the current memory store. Use this after bulk changes or if the index seems out of sync.",
        parameters: params_rebuild_index,
        execute: execute_rebuild_index,
        groups: &["builtin"],
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_search_params() {
        let params = params_semantic_search();
        assert!(params["required"].as_array().unwrap().contains(&Value::String("query".to_string())));
    }

    #[test]
    fn test_rebuild_index_params() {
        let params = params_rebuild_index();
        // 无必需参数
        assert!(params["required"].as_array().map(|a| a.is_empty()).unwrap_or(true));
    }
}
