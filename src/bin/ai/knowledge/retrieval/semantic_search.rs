/// Semantic (vector) search over vector store.
use super::super::entry::KnowledgeEntry;
use super::super::storage::vector_store::{VectorEntry, VectorStore};

/// Search entries by semantic similarity.
pub fn semantic_search(
    store: &VectorStore,
    query: &str,
    limit: usize,
    category: Option<&str>,
) -> Result<Vec<(VectorEntry, f32)>, String> {
    let query_embedding = store.embed_text(query)?;
    store.semantic_search(&query_embedding, limit, category)
}

/// Convert vector entries to knowledge entries for display.
pub fn vector_to_knowledge_entries(entries: &[(VectorEntry, f32)]) -> Vec<(KnowledgeEntry, f64)> {
    entries
        .iter()
        .map(|(ve, score)| {
            let ke = KnowledgeEntry {
                id: Some(ve.id.clone()),
                timestamp: String::new(),
                category: ve.category.clone(),
                note: ve.content.clone(),
                tags: ve.tags.clone(),
                source: None,
                priority: Some(100),
            };
            (ke, *score as f64)
        })
        .collect()
}
