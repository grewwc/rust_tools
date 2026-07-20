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
                "description": "Meaning-based query; falls back to keyword search without embeddings."
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
    // 确保 embedding provider 已初始化（与 note_search 一致：在入口处调用 warm_up）。
    // GLOBAL_PROVIDER 是 OnceLock，重复调用无副作用；未配置 key 时 is_ready() 仍为 false，
    // 下面会自动降级为 BM25/lexical 检索。
    crate::ai::knowledge::indexing::embedder::warm_up();

    ensure_rag_store()?;
    let guard = get_rag_store()?;
    let store = guard.as_ref().ok_or("RAG store not initialized")?;

    let query = args["query"]
        .as_str()
        .ok_or("Missing 'query'. Provide a semantic search query.")?;
    let category = args["category"].as_str();
    let limit = args["limit"].as_u64().map(|v| v as usize).unwrap_or(5);
    let use_hybrid = args["hybrid"].as_bool().unwrap_or(true);

    // 嵌入模型已经不再静态链接（fastembed/ONNX runtime 25MB 移除），
    // 当本地 embedder::is_ready() == false 时直接降级为 BM25/lexical 检索，
    // 不再返回错误，并在结果前缀里标记 [retrieval:bm25-fallback]。
    let semantic_available = crate::ai::knowledge::indexing::embedder::is_ready();
    if !semantic_available {
        let mem_store = MemoryStore::from_env_or_config();
        let jsonl_store = JsonlStore::new(mem_store.path().to_path_buf());
        let config = KnowledgeConfig::default();
        let bm25_results = crate::ai::knowledge::retrieval::keyword_search::keyword_search(
            &jsonl_store,
            query,
            limit,
            &config,
        )?;
        if bm25_results.is_empty() {
            return Ok(format!("No results found for: '{}'", query));
        }
        let mut output = format!(
            "[retrieval:bm25-fallback] semantic embedding unavailable, using BM25 + lexical similarity\n\nResults for '{}':\n\n",
            query
        );
        for (idx, (entry, score)) in bm25_results.iter().enumerate() {
            output.push_str(&format!(
                "{}. [score: {:.3}] [{}] {}\n",
                idx + 1,
                score,
                entry.category,
                entry.note
            ));
            if !entry.tags.is_empty() {
                output.push_str(&format!("   Tags: {}\n", entry.tags.join(", ")));
            }
        }
        return Ok(output);
    }

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
        description: "Search saved knowledge by meaning; falls back to keyword search without embeddings.",
        parameters: params_semantic_search,
        execute: execute_semantic_search,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::Spawnable,
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
    // 确保 embedding provider 已初始化（与 note_search / semantic_search 一致）。
    // 向量索引重建本质上依赖 embedding，无法降级；未配置 key 时给出明确提示。
    crate::ai::knowledge::indexing::embedder::warm_up();
    if !crate::ai::knowledge::indexing::embedder::is_ready() {
        return Err(
            "Embedding provider not available — vector index rebuild requires embeddings. \
             Configure embedding.api_key (or model.aliyun_api_key) to enable."
                .to_string(),
        );
    }

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
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin"],
    }
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_search_params() {
        let params = params_semantic_search();
        assert!(
            params["required"]
                .as_array()
                .unwrap()
                .contains(&Value::String("query".to_string()))
        );
    }

    #[test]
    fn test_rebuild_index_params() {
        let params = params_rebuild_index();
        // No required parameters
        assert!(
            params["required"]
                .as_array()
                .map(|a| a.is_empty())
                .unwrap_or(true)
        );
    }
}
