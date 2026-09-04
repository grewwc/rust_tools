//! High-level semantic-search index on top of the canonical memory store.
//!
//! The vector index is a pure derived artifact: `rebuild_from_memory` embeds
//! every memory entry and stores the vectors in an SQLite-backed
//! `VectorStore`. The index is rebuilt lazily on first use when it is empty
//! or stale (the configured embedding model changed), or explicitly via the
//! `knowledge_rebuild_index` tool. The model fingerprint stored in the index
//! meta table is what detects a model change, so switching embedding models
//! cannot silently serve vectors from an incompatible index.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::ai::knowledge::config::knowledge_config;
use crate::ai::knowledge::indexing::embedder;
use crate::ai::knowledge::storage::vector_store::{VectorEntry, VectorStore};
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};

/// Meta key storing the embedding model used to build the index.
const META_MODEL: &str = "embedding_model";

/// Process-wide RAG store (created once by `ensure_rag_store`).
static RAG_STORE: OnceLock<Result<RagStore, String>> = OnceLock::new();

/// A search hit with display fields (no raw embedding).
#[derive(Debug, Clone)]
pub struct RagHit {
    pub id: String,
    pub category: String,
    pub content: String,
    pub tags: Vec<String>,
    pub timestamp: i64,
    pub score: f32,
}

/// Semantic index over memory entries plus the canonical store it mirrors.
pub struct RagStore {
    vec_store: Mutex<VectorStore>,
    memory_store: MemoryStore,
    hybrid_vector_weight: f32,
}

/// Location of the SQLite vector index (derived, rebuildable data).
fn index_path() -> std::path::PathBuf {
    let base = dirs::config_dir()
        .map(|p| p.join("rust_tools/rag_index"))
        .unwrap_or_else(std::env::temp_dir);
    let _ = std::fs::create_dir_all(&base);
    base.join("vec.db")
}

impl RagStore {
    /// Open the index and wire up the canonical memory store.
    fn new() -> Result<Self, String> {
        if !embedder::is_ready() {
            return Err(
                "Semantic search needs an embedding provider. Set ai.embedding.enable=true and \
                 ai.embedding.api_key (or ai.model.volcano.api_key), then run \
                 knowledge_rebuild_index."
                    .to_string(),
            );
        }
        let vec_store = VectorStore::new(&index_path())
            .map_err(|e| format!("Failed to open vector index: {e}"))?;
        let cfg = knowledge_config();
        Ok(Self {
            vec_store: Mutex::new(vec_store),
            memory_store: MemoryStore::from_env_or_config(),
            hybrid_vector_weight: cfg.hybrid_vector_weight,
        })
    }

    /// Text fed to the embedder for one entry (category + note + tags).
    fn embed_text_for(&self, e: &AgentMemoryEntry) -> String {
        let mut t = format!("{}: {}", e.category, e.note);
        if !e.tags.is_empty() {
            t.push_str(&format!(" [tags: {}]", e.tags.join(", ")));
        }
        t
    }

    /// (Re)build the index from all memory entries; returns the entry count.
    fn rebuild_from_memory(&self) -> Result<usize, String> {
        let entries: Vec<AgentMemoryEntry> = self.memory_store.all()?;
        let model = embedder::current_model().unwrap_or("").to_string();
        let mut vec_entries = Vec::with_capacity(entries.len());
        // Embed in batches to bound request size.
        for chunk in entries.chunks(32) {
            let texts: Vec<String> = chunk.iter().map(|e| self.embed_text_for(e)).collect();
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let vectors = embedder::embed_texts(&refs)
                .ok_or_else(|| "Embedding request failed during index rebuild.".to_string())?;
            for (entry, vec) in chunk.iter().zip(vectors.iter()) {
                vec_entries.push(VectorEntry {
                    id: entry.id.clone().unwrap_or_default(),
                    content: entry.note.clone(),
                    category: entry.category.clone(),
                    tags: entry.tags.clone(),
                    embedding: vec.clone(),
                    timestamp: entry.timestamp.parse().unwrap_or(0),
                });
            }
        }
        {
            let mut store = self
                .vec_store
                .lock()
                .map_err(|_| "vector index lock poisoned".to_string())?;
            store
                .upsert_batch(vec_entries)
                .map_err(|e| format!("Failed to write vector index: {e}"))?;
            // Prune index rows whose memory entries no longer exist.
            let current: HashMap<String, ()> = entries
                .iter()
                .filter_map(|e| e.id.clone())
                .map(|id| (id, ()))
                .collect();
            let stale: Vec<String> = store
                .all()
                .into_iter()
                .map(|e| e.id)
                .filter(|id| !current.contains_key(id))
                .collect();
            store
                .delete(&stale)
                .map_err(|e| format!("Failed to prune vector index: {e}"))?;
            store
                .set_meta(META_MODEL, &model)
                .map_err(|e| format!("Failed to write vector index meta: {e}"))?;
        }
        Ok(entries.len())
    }

    /// Ensure the index exists and matches the current embedding model.
    fn ensure_fresh(&self) -> Result<(), String> {
        let model = embedder::current_model().unwrap_or("").to_string();
        let stale = {
            let store = self
                .vec_store
                .lock()
                .map_err(|_| "vector index lock poisoned".to_string())?;
            store.is_empty() || store.get_meta(META_MODEL).as_deref() != Some(model.as_str())
        };
        if stale {
            self.rebuild_from_memory()?;
        }
        Ok(())
    }

    /// Pure vector (semantic) search over the index.
    pub fn semantic_search(
        &self,
        query: &str,
        category: Option<&str>,
        limit: usize,
    ) -> Result<Vec<RagHit>, String> {
        self.ensure_fresh()?;
        let qv =
            embedder::embed_text(query).ok_or_else(|| "Failed to embed the query.".to_string())?;
        let hits = {
            let store = self
                .vec_store
                .lock()
                .map_err(|_| "vector index lock poisoned".to_string())?;
            store.semantic_search(&qv, category, limit)
        };
        Ok(hits.into_iter().map(|(e, s)| self.to_hit(e, s)).collect())
    }

    /// Hybrid search: BM25 (canonical memory store) merged with semantic scores.
    pub fn hybrid_search(
        &self,
        query: &str,
        category: Option<&str>,
        limit: usize,
    ) -> Result<Vec<RagHit>, String> {
        self.ensure_fresh()?;
        let qv =
            embedder::embed_text(query).ok_or_else(|| "Failed to embed the query.".to_string())?;
        let w = self.hybrid_vector_weight.clamp(0.0, 1.0);

        // BM25 half from the canonical memory store.
        let bm25: Vec<(String, f32)> = self
            .memory_store
            .search(query, limit)?
            .into_iter()
            .map(|(e, s)| (e.id.clone().unwrap_or_default(), s as f32))
            .collect();
        // Semantic half from the vector index (more candidates so the merge
        // is not dominated by BM25-only ids).
        let sem: Vec<(VectorEntry, f32)> = {
            let store = self
                .vec_store
                .lock()
                .map_err(|_| "vector index lock poisoned".to_string())?;
            store.semantic_search(&qv, category, limit.saturating_mul(2))
        };

        // Normalize each half to [0, 1] by min-max over its own range, then
        // merge by id with the configured vector weight.
        let bm25_norm = normalize_scores(&bm25);
        let sem_raw: Vec<(String, f32)> = sem.iter().map(|(e, s)| (e.id.clone(), *s)).collect();
        let sem_norm = normalize_scores(&sem_raw);

        let mut merged: HashMap<String, (VectorEntry, f32)> = HashMap::new();
        for ((e, _), (id, s)) in sem.into_iter().zip(sem_norm.iter()) {
            merged.insert(id.clone(), (e, w * *s));
        }
        // BM25-only ids need display fields; load all memory entries once.
        let by_id: HashMap<String, AgentMemoryEntry> = self
            .memory_store
            .all()?
            .into_iter()
            .filter_map(|e| e.id.clone().map(|id| (id, e)))
            .collect();
        for (id, s) in bm25_norm.iter() {
            // The BM25 half (memory_store::search) has no category filter, so
            // entries that only matched BM25 would leak other categories into
            // a category-scoped hybrid search. Entries already in `merged`
            // came from the category-filtered semantic half and are safe.
            // Filter here (after BM25 normalization) to keep the score
            // distribution of the BM25 half untouched.
            if let Some(cat) = category {
                if by_id.get(id).map(|m| m.category.as_str()) != Some(cat) {
                    continue;
                }
            }
            let entry = merged.entry(id.clone()).or_insert_with(|| {
                let mem = by_id.get(id);
                let (category, content, tags, timestamp) = match mem {
                    Some(m) => (
                        m.category.clone(),
                        m.note.clone(),
                        m.tags.clone(),
                        m.timestamp.parse().unwrap_or(0),
                    ),
                    None => (String::new(), String::new(), Vec::new(), 0),
                };
                (
                    VectorEntry {
                        id: id.clone(),
                        content,
                        category,
                        tags,
                        embedding: Vec::new(),
                        timestamp,
                    },
                    0.0,
                )
            });
            entry.1 += (1.0 - w) * *s;
        }

        let mut out: Vec<RagHit> = merged
            .into_values()
            .map(|(e, s)| self.to_hit(e, s))
            .collect();
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(limit);
        Ok(out)
    }

    fn to_hit(&self, e: VectorEntry, score: f32) -> RagHit {
        RagHit {
            id: e.id,
            category: e.category,
            content: e.content,
            tags: e.tags,
            timestamp: e.timestamp,
            score,
        }
    }
}

/// Min-max normalize scores to [0, 1]; all-equal input maps to 0.5.
fn normalize_scores(scored: &[(String, f32)]) -> Vec<(String, f32)> {
    if scored.is_empty() {
        return Vec::new();
    }
    let min = scored.iter().map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
    let max = scored
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);
    if (max - min).abs() < 1e-9 {
        return scored.iter().map(|(id, _)| (id.clone(), 0.5)).collect();
    }
    scored
        .iter()
        .map(|(id, s)| (id.clone(), (s - min) / (max - min)))
        .collect()
}

/// Ensure the RAG store exists (lazy init + embedder warm-up), then return it.
pub fn ensure_rag_store() -> Result<&'static RagStore, String> {
    if let Some(res) = RAG_STORE.get() {
        return res.as_ref().map_err(|e| e.clone());
    }
    embedder::warm_up();
    let store = RagStore::new()?;
    let _ = RAG_STORE.set(Ok(store));
    RAG_STORE
        .get()
        .ok_or_else(|| "RAG store initialization failed.".to_string())?
        .as_ref()
        .map_err(|e| e.clone())
}

/// Explicit rebuild used by the `knowledge_rebuild_index` tool.
pub fn rebuild_index() -> Result<String, String> {
    let store = ensure_rag_store()?;
    let count = store.rebuild_from_memory()?;
    let model = embedder::current_model().unwrap_or("none");
    Ok(format!(
        "Rebuilt semantic index with {count} entries (embedding model: {model})."
    ))
}

/// Free-function entry points used by the tools.
pub fn semantic_search(
    query: &str,
    category: Option<&str>,
    limit: usize,
) -> Result<Vec<RagHit>, String> {
    ensure_rag_store()?.semantic_search(query, category, limit)
}

pub fn hybrid_search(
    query: &str,
    category: Option<&str>,
    limit: usize,
) -> Result<Vec<RagHit>, String> {
    ensure_rag_store()?.hybrid_search(query, category, limit)
}
