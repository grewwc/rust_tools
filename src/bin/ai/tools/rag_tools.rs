/// RAG 语义检索工具 — 为 agent 提供向量语义搜索能力
///
/// 包含两个工具：
/// - `knowledge_semantic_search` — 语义相似度搜索（向量检索）
/// - `knowledge_rebuild_index` — 重建向量索引（从 memory_store 同步）
use serde_json::Value;

use crate::ai::knowledge::config::KnowledgeConfig;
use crate::ai::knowledge::storage::jsonl_store::JsonlStore;
use crate::ai::knowledge::sync::knowledge_sync;
use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;
use crate::ai::tools::storage::memory_store::MemoryStore;
use crate::ai::tools::storage::rag_store::{ensure_rag_store, get_rag_store};

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

    let query = args["query"]
        .as_str()
        .ok_or("Missing 'query'. Provide a semantic search query.")?;
    let category = args["category"].as_str();
    let limit = args["limit"].as_u64().map(|v| v as usize).unwrap_or(5);
    let use_hybrid = args["hybrid"].as_bool().unwrap_or(true);

    if use_hybrid {
        // Hybrid search: combine BM25 with vector
        let mem_store = MemoryStore::from_env_or_config();
        let jsonl_store = JsonlStore::new(mem_store.path().to_path_buf());
        let config = KnowledgeConfig::default();

        let bm25_results = crate::ai::knowledge::retrieval::keyword_search::keyword_search(
            &jsonl_store,
            query,
            limit * 3,
            &config,
        )?;

        let bm25_for_hybrid: Vec<(String, f32)> = bm25_results
            .iter()
            .filter_map(|(entry, score)| entry.id.as_ref().map(|id| (id.clone(), *score as f32)))
            .collect();

        let _results = store.hybrid_search(
            query,
            bm25_for_hybrid,
            limit,
            category,
            config.hybrid_vector_weight,
        )?;

        let bm25_for_hybrid: Vec<(String, f32)> = bm25_results
            .iter()
            .filter_map(|(entry, score)| entry.id.as_ref().map(|id| (id.clone(), *score as f32)))
            .collect();

        let results = store.hybrid_search(
            query,
            bm25_for_hybrid,
            limit,
            category,
            config.hybrid_vector_weight,
        )?;

        if results.is_empty() {
            return Ok(format!("No results found for: '{}'", query));
        }

        let mut output = format!("Semantic search results for '{}':\n\n", query);
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
        // Pure semantic search
        let results = store.semantic_search(query, limit, category)?;

        if results.is_empty() {
            return Ok(format!("No results found for: '{}'", query));
        }

        let mut output = format!("Semantic search results for '{}':\n\n", query);
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

    let mem_store = MemoryStore::from_env_or_config();
    let jsonl_store = JsonlStore::new(mem_store.path().to_path_buf());
    let count = knowledge_sync::rebuild_vector_index(&jsonl_store, store)?;

    Ok(format!("Rebuilt RAG index: {} entries vectorized.", count))
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
        assert!(params["required"]
            .as_array()
            .unwrap()
            .contains(&Value::String("query".to_string())));
    }

    #[test]
    fn test_rebuild_index_params() {
        let params = params_rebuild_index();
        // No required parameters
        assert!(params["required"]
            .as_array()
            .map(|a| a.is_empty())
            .unwrap_or(true));
    }
}
