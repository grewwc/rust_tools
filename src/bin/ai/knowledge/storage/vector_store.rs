/// Vector store — pure vector CRUD operations.
/// Decoupled from JSONL store; sync is handled by the sync module.
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rust_tools::commonw::FastMap;
use serde::{Deserialize, Serialize};

use super::super::indexing::similarity;

/// Embedding dimension for the model.
const EMBEDDING_DIM: usize = 384;

/// A vector-indexed knowledge entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    pub id: String,
    pub content: String,
    pub category: String,
    pub tags: Vec<String>,
    pub embedding: Vec<f32>,
    pub timestamp: u64,
}

/// Embedder trait for vector store.
pub trait VectorEmbedder: Sync + Send {
    fn embed(&self, text: &str) -> Result<Vec<f32>, String>;
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;
}

/// FastEmbed-based embedder (lazy-loaded).
pub struct LazyEmbedder {
    inner: Mutex<Option<fastembed::TextEmbedding>>,
    cache_dir: PathBuf,
}

impl LazyEmbedder {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            inner: Mutex::new(None),
            cache_dir,
        }
    }

    fn get(&self) -> Result<&fastembed::TextEmbedding, String> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| format!("Lock poisoned: {}", e))?;
        if guard.is_none() {
            let embedder = fastembed::TextEmbedding::try_new(
                fastembed::InitOptions::default()
                    .with_cache_dir(self.cache_dir.clone())
                    .with_show_download_progress(true),
            )
            .map_err(|e| format!("Failed to load embedding model: {}", e))?;
            *guard = Some(embedder);
        }
        Ok(unsafe {
            std::mem::transmute::<&fastembed::TextEmbedding, &fastembed::TextEmbedding>(
                guard.as_ref().unwrap(),
            )
        })
    }
}

impl VectorEmbedder for LazyEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let embedder = self.get()?;
        let embeddings = embedder
            .embed(vec![text], None)
            .map_err(|e| format!("Failed to embed: {}", e))?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| "No embedding produced".to_string())
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let embedder = self.get()?;
        embedder
            .embed(texts.to_vec(), None)
            .map_err(|e| format!("Failed to embed texts: {}", e))
    }
}

/// Vector store backed by sled.
pub struct VectorStore {
    db: sled::Db,
    embedder: Box<dyn VectorEmbedder>,
    index_path: PathBuf,
}

impl VectorStore {
    pub fn new(path: &Path, embedder: Box<dyn VectorEmbedder>) -> Result<Self, String> {
        let db = sled::open(path)
            .map_err(|e| format!("Failed to open vector index at {:?}: {}", path, e))?;
        Ok(Self {
            db,
            embedder,
            index_path: path.to_path_buf(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.index_path
    }

    /// Upsert a vector entry.
    pub fn upsert(&self, entry: VectorEntry) -> Result<(), String> {
        let key = format!("vec:{}", entry.id);
        let value =
            serde_json::to_vec(&entry).map_err(|e| format!("Failed to serialize: {}", e))?;
        self.db
            .insert(key.as_bytes(), value)
            .map_err(|e| format!("Failed to write: {}", e))?;
        self.db
            .flush()
            .map_err(|e| format!("Failed to flush: {}", e))?;
        Ok(())
    }

    /// Delete a vector entry by ID.
    pub fn delete(&self, id: &str) -> Result<bool, String> {
        let key = format!("vec:{}", id);
        let existed = self
            .db
            .contains_key(key.as_bytes())
            .map_err(|e| format!("Failed to check key: {}", e))?;
        if existed {
            self.db
                .remove(key.as_bytes())
                .map_err(|e| format!("Failed to delete: {}", e))?;
            self.db
                .flush()
                .map_err(|e| format!("Failed to flush: {}", e))?;
        }
        Ok(existed)
    }

    /// Get a single entry by ID.
    pub fn get(&self, id: &str) -> Result<Option<VectorEntry>, String> {
        let key = format!("vec:{}", id);
        if let Some(val_bytes) = self
            .db
            .get(key.as_bytes())
            .map_err(|e| format!("Failed to get: {}", e))?
        {
            let entry: VectorEntry = serde_json::from_slice(&val_bytes)
                .map_err(|e| format!("Failed to deserialize: {}", e))?;
            Ok(Some(entry))
        } else {
            Ok(None)
        }
    }

    /// Semantic search — cosine similarity top-k.
    pub fn semantic_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        category: Option<&str>,
    ) -> Result<Vec<(VectorEntry, f32)>, String> {
        let mut candidates = Vec::new();

        for result in self.db.iter() {
            let (key_bytes, val_bytes) = result.map_err(|e| format!("Failed to iterate: {}", e))?;
            let key = String::from_utf8_lossy(&key_bytes);
            if !key.starts_with("vec:") {
                continue;
            }

            let entry: VectorEntry = serde_json::from_slice(&val_bytes)
                .map_err(|e| format!("Failed to deserialize: {}", e))?;

            if let Some(cat) = category {
                if entry.category != cat {
                    continue;
                }
            }

            let sim = similarity::cosine_similarity(query_embedding, &entry.embedding);
            candidates.push((entry, sim));
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(limit);
        Ok(candidates)
    }

    /// Hybrid search — combine BM25 scores with semantic scores.
    pub fn hybrid_search(
        &self,
        query_embedding: &[f32],
        bm25_results: Vec<(String, f32)>,
        limit: usize,
        category: Option<&str>,
        vector_weight: f32,
    ) -> Result<Vec<(String, VectorEntry, f32)>, String> {
        // Get semantic scores for all entries
        let mut all_entries = Vec::new();
        for result in self.db.iter() {
            let (key_bytes, val_bytes) = result.map_err(|e| format!("Failed to iterate: {}", e))?;
            let key = String::from_utf8_lossy(&key_bytes);
            if !key.starts_with("vec:") {
                continue;
            }
            let entry: VectorEntry = serde_json::from_slice(&val_bytes)
                .map_err(|e| format!("Failed to deserialize: {}", e))?;
            if let Some(cat) = category {
                if entry.category != cat {
                    continue;
                }
            }
            let sim = similarity::cosine_similarity(query_embedding, &entry.embedding);
            all_entries.push((entry.id.clone(), entry, sim));
        }

        let semantic_scores: FastMap<String, f32> = all_entries
            .iter()
            .map(|(id, _, score)| (id.clone(), *score))
            .collect();
        let entry_map: FastMap<String, VectorEntry> = all_entries
            .into_iter()
            .map(|(id, entry, _)| (id, entry))
            .collect();

        let bm25_max = bm25_results.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
        let bm25_normalized: FastMap<String, f32> = if bm25_max > 0.0 {
            bm25_results
                .into_iter()
                .map(|(id, score)| (id, score / bm25_max))
                .collect()
        } else {
            FastMap::default()
        };

        let mut all_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        all_ids.extend(bm25_normalized.keys().cloned());
        all_ids.extend(semantic_scores.keys().cloned());

        let mut combined: Vec<(String, f32)> = Vec::new();
        for id in all_ids {
            let bm25 = bm25_normalized.get(&id).copied().unwrap_or(0.0);
            let semantic = semantic_scores.get(&id).copied().unwrap_or(0.0);
            let final_score = (1.0 - vector_weight) * bm25 + vector_weight * semantic;
            combined.push((id, final_score));
        }

        combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        combined.truncate(limit);

        let mut results = Vec::new();
        for (id, score) in combined {
            if let Some(entry) = entry_map.get(&id) {
                results.push((id, entry.clone(), score));
            }
        }
        Ok(results)
    }

    /// Count entries.
    pub fn count(&self) -> Result<usize, String> {
        let mut count = 0;
        for result in self.db.iter() {
            let (key_bytes, _) = result.map_err(|e| format!("Failed to iterate: {}", e))?;
            let key = String::from_utf8_lossy(&key_bytes);
            if key.starts_with("vec:") {
                count += 1;
            }
        }
        Ok(count)
    }

    /// List all entry IDs.
    pub fn list_ids(&self) -> Result<Vec<String>, String> {
        let mut ids = Vec::new();
        for result in self.db.iter() {
            let (key_bytes, _) = result.map_err(|e| format!("Failed to iterate: {}", e))?;
            let key = String::from_utf8_lossy(&key_bytes);
            if key.starts_with("vec:") {
                ids.push(key.trim_start_matches("vec:").to_string());
            }
        }
        Ok(ids)
    }

    /// Embed text using the store's embedder.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>, String> {
        self.embedder.embed(text)
    }

    /// Embed multiple texts.
    pub fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        self.embedder.embed_batch(texts)
    }

    /// Rebuild index from a list of entries with their search texts.
    pub fn rebuild_from_entries(
        &self,
        entries: &[(String, String, String, Vec<String>)],
    ) -> Result<usize, String> {
        // entries: (id, category, note, tags)
        let texts: Vec<String> = entries
            .iter()
            .map(|(_, cat, note, tags)| {
                let mut text = format!("{}: {}", cat, note);
                if !tags.is_empty() {
                    text.push_str(&format!(" [tags: {}]", tags.join(", ")));
                }
                text
            })
            .collect();

        let embeddings = self.embed_texts(&texts)?;

        let mut count = 0;
        for ((id, category, note, tags), embedding) in entries.iter().zip(embeddings.into_iter()) {
            let content = format!("{}: {}", category, note);
            self.upsert(VectorEntry {
                id: id.clone(),
                content,
                category: category.clone(),
                tags: tags.clone(),
                embedding,
                timestamp: 0,
            })?;
            count += 1;
        }
        Ok(count)
    }
}
