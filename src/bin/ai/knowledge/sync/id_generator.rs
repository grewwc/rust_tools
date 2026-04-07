/// Canonical ID generation for knowledge entries.
/// Single source of truth — prevents ID mismatch between save and delete.
use super::super::entry::KnowledgeEntry;

/// Generate a unique ID for a knowledge entry.
/// Uses UUID for new entries, preserves existing IDs.
pub fn generate_id(entry: &KnowledgeEntry) -> String {
    if let Some(ref id) = entry.id {
        return id.clone();
    }
    format!("mem_{}", uuid::Uuid::new_v4().simple())
}

/// Generate a deterministic ID from content (for backward compatibility with RAG sync).
/// Only used when syncing to vector store where we need stable IDs.
pub fn generate_deterministic_id(timestamp: &str, content: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    timestamp.hash(&mut hasher);
    content.hash(&mut hasher);
    let hash = hasher.finish();
    format!("rag_{:x}", hash)
}
