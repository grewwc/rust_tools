/// Hybrid search — combines keyword (BM25) and semantic (vector) results.
use super::super::config::KnowledgeConfig;
use super::super::entry::KnowledgeEntry;
use super::super::storage::jsonl_store::JsonlStore;
use super::super::storage::vector_store::{VectorEntry, VectorStore};
use super::keyword_search;

/// Hybrid search result.
pub struct HybridResult {
    pub entry: KnowledgeEntry,
    pub score: f64,
    pub vector_entry: Option<VectorEntry>,
}

/// Perform hybrid search combining BM25 and vector similarity.
pub fn hybrid_search(
    jsonl_store: &JsonlStore,
    vector_store: &VectorStore,
    query: &str,
    limit: usize,
    category: Option<&str>,
    config: &KnowledgeConfig,
) -> Result<Vec<HybridResult>, String> {
    // Get BM25 results
    let bm25_results = keyword_search::keyword_search(jsonl_store, query, limit * 3, config)?;

    // Build BM25 score map for hybrid
    let bm25_for_hybrid: Vec<(String, f32)> = bm25_results
        .iter()
        .filter_map(|(entry, score)| entry.id.as_ref().map(|id| (id.clone(), *score as f32)))
        .collect();

    // Get vector store results via hybrid
    let query_embedding = vector_store.embed_text(query)?;
    let hybrid_results = vector_store.hybrid_search(
        &query_embedding,
        bm25_for_hybrid,
        limit,
        category,
        config.hybrid_vector_weight,
    )?;

    // Convert to HybridResult
    let mut results: Vec<HybridResult> = Vec::new();
    for (_id, ve, score) in hybrid_results {
        let entry = KnowledgeEntry {
            id: Some(ve.id.clone()),
            timestamp: String::new(),
            category: ve.category.clone(),
            note: ve.content.clone(),
            tags: ve.tags.clone(),
            source: None,
            priority: Some(100),
        };
        results.push(HybridResult {
            entry,
            score: score as f64,
            vector_entry: Some(ve),
        });
    }

    Ok(results)
}
