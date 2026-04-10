/// Knowledge sync — orchestrates JSONL ↔ vector store synchronization.
/// The ONLY place that coordinates between the two storage backends.
use super::super::entry::KnowledgeEntry;
use super::super::storage::jsonl_store::JsonlStore;
use super::id_generator;

/// Trait for vector stores that can accept entries.
pub trait VectorStoreSync {
    fn upsert_entry(
        &self,
        id: String,
        content: String,
        category: String,
        tags: Vec<String>,
        embedding: Vec<f32>,
    ) -> Result<(), String>;
    fn delete_entry(&self, id: &str) -> Result<bool, String>;
    fn embed_text(&self, text: &str) -> Result<Vec<f32>, String>;
    
    /// Batch embed multiple texts (default implementation calls embed_text individually)
    fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.embed_text(text)?);
        }
        Ok(embeddings)
    }
}

/// Sync a single entry to the vector store.
pub fn sync_entry_to_vector(
    _jsonl_store: &JsonlStore,
    vector_store: &dyn VectorStoreSync,
    entry: &KnowledgeEntry,
) -> Result<(), String> {
    let id = entry
        .id
        .clone()
        .unwrap_or_else(|| id_generator::generate_id(entry));
    let content = entry.search_text();
    let embedding = vector_store.embed_text(&content)?;

    vector_store.upsert_entry(
        id,
        content,
        entry.category.clone(),
        entry.tags.clone(),
        embedding,
    )?;

    Ok(())
}

/// Delete an entry from the vector store.
pub fn delete_entry_from_vector(
    vector_store: &dyn VectorStoreSync,
    id: &str,
) -> Result<bool, String> {
    vector_store.delete_entry(id)
}

/// Rebuild the entire vector index from the JSONL store.
pub fn rebuild_vector_index(
    jsonl_store: &JsonlStore,
    vector_store: &dyn VectorStoreSync,
) -> Result<usize, String> {
    let entries = jsonl_store.list_all()?;

    // Filter out empty entries
    let valid_entries: Vec<&KnowledgeEntry> = entries
        .iter()
        .filter(|e| !e.note.trim().is_empty())
        .collect();
    
    if valid_entries.is_empty() {
        return Ok(0);
    }
    
    // Batch embed all texts at once (much faster than individual API calls)
    let texts: Vec<String> = valid_entries.iter().map(|e| e.search_text()).collect();
    let embeddings = vector_store.embed_texts(&texts)?;
    
    // Upsert all entries with their embeddings
    let mut count = 0;
    for (entry, embedding) in valid_entries.iter().zip(embeddings.iter()) {
        let id = entry.id.clone().unwrap_or_else(|| id_generator::generate_id(entry));
        let content = entry.search_text();
        if vector_store.upsert_entry(id, content, entry.category.clone(), entry.tags.clone(), embedding.clone()).is_ok() {
            count += 1;
        }
    }

    Ok(count)
}
