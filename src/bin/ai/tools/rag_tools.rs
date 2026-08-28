//! Semantic knowledge search tools backed by the embedding vector index.
//!
//! `knowledge_semantic_search` runs hybrid (BM25 + semantic) or pure semantic
//! search over durable knowledge. `knowledge_rebuild_index` forces an index
//! rebuild, which is required after changing the embedding model.

use serde_json::Value;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::storage::rag_store::{hybrid_search, rebuild_index, semantic_search};

fn execute_knowledge_semantic_search(args: &Value) -> Result<String, String> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if query.is_empty() {
        return Err("`query` is required for knowledge_semantic_search.".to_string());
    }
    let category = args
        .get("category")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(5)
        .clamp(1, 20) as usize;
    let hybrid = args
        .get("hybrid")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let hits = if hybrid {
        hybrid_search(&query, category.as_deref(), limit)?
    } else {
        semantic_search(&query, category.as_deref(), limit)?
    };

    if hits.is_empty() {
        return Ok("No relevant knowledge found.".to_string());
    }
    let mut out = Vec::with_capacity(hits.len());
    for h in hits {
        let tags = if h.tags.is_empty() {
            "-".to_string()
        } else {
            h.tags.join(", ")
        };
        out.push(format!(
            "- [{:.3}] {}: {}\n  ID: {} | Tags: {}",
            h.score, h.category, h.content, h.id, tags
        ));
    }
    Ok(out.join("\n"))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "knowledge_semantic_search",
        description: "",
        execute: execute_knowledge_semantic_search,
    }
});

fn execute_knowledge_rebuild_index(args: &Value) -> Result<String, String> {
    let _ = args;
    rebuild_index()
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "knowledge_rebuild_index",
        description: "",
        execute: execute_knowledge_rebuild_index,
    }
});
