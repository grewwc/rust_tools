/// Vector store — pure vector CRUD operations.
/// Decoupled from JSONL store; sync is handled by the sync module.
///
/// 后端已从 sled 迁移为 rusqlite（项目内已 bundled），单表 KV：
///   `vec_entries(id TEXT PRIMARY KEY, payload BLOB)`，与原 sled `vec:{id}` 语义对齐。
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::{Connection, OptionalExtension, params};
use rust_tools::commonw::FastMap;
use serde::{Deserialize, Serialize};

use super::super::indexing::{embedder, similarity};

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

struct GlobalEmbeddingAdapter;

impl VectorEmbedder for GlobalEmbeddingAdapter {
    fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        embedder::embed_text(text).ok_or_else(|| "embedding not available".to_string())
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        embedder::embed_texts(texts).ok_or_else(|| "embedding not available".to_string())
    }
}

pub struct VectorStore {
    /// SQLite 连接（rusqlite::Connection 非 Sync，包裹 Mutex 以支持并发）。
    conn: Mutex<Connection>,
    embedder: Box<dyn VectorEmbedder>,
    index_path: PathBuf,
}

impl VectorStore {
    pub fn new(path: &Path, embedder: Box<dyn VectorEmbedder>) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create vector index parent dir: {}", e))?;
        }
        let conn = Connection::open(path)
            .map_err(|e| format!("Failed to open vector index at {:?}: {}", path, e))?;
        // 与 memory_index 一致的并发优化。
        let _ = conn.pragma_update(None, "journal_mode", "WAL");
        let _ = conn.pragma_update(None, "synchronous", "NORMAL");
        conn.execute(
            "CREATE TABLE IF NOT EXISTS vec_entries (\
                id TEXT PRIMARY KEY,\
                payload BLOB NOT NULL\
            )",
            [],
        )
        .map_err(|e| format!("Failed to init vec_entries table: {}", e))?;
        Ok(Self {
            conn: Mutex::new(conn),
            embedder,
            index_path: path.to_path_buf(),
        })
    }

    pub fn with_global_provider(path: &Path) -> Result<Self, String> {
        Self::new(path, Box::new(GlobalEmbeddingAdapter))
    }

    pub fn path(&self) -> &Path {
        &self.index_path
    }

    fn lock_conn(&self) -> Result<std::sync::MutexGuard<'_, Connection>, String> {
        self.conn
            .lock()
            .map_err(|e| format!("vector store mutex poisoned: {}", e))
    }

    /// Upsert a vector entry.
    pub fn upsert(&self, entry: VectorEntry) -> Result<(), String> {
        let payload =
            serde_json::to_vec(&entry).map_err(|e| format!("Failed to serialize: {}", e))?;
        let conn = self.lock_conn()?;
        conn.execute(
            "INSERT INTO vec_entries (id, payload) VALUES (?1, ?2) \
             ON CONFLICT(id) DO UPDATE SET payload = excluded.payload",
            params![entry.id, payload],
        )
        .map_err(|e| format!("Failed to write: {}", e))?;
        Ok(())
    }

    /// Delete a vector entry by ID.
    pub fn delete(&self, id: &str) -> Result<bool, String> {
        let conn = self.lock_conn()?;
        let affected = conn
            .execute("DELETE FROM vec_entries WHERE id = ?1", params![id])
            .map_err(|e| format!("Failed to delete: {}", e))?;
        Ok(affected > 0)
    }

    /// Get a single entry by ID.
    pub fn get(&self, id: &str) -> Result<Option<VectorEntry>, String> {
        let conn = self.lock_conn()?;
        let payload: Option<Vec<u8>> = conn
            .query_row(
                "SELECT payload FROM vec_entries WHERE id = ?1",
                params![id],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()
            .map_err(|e| format!("Failed to get: {}", e))?;
        match payload {
            Some(bytes) => {
                let entry: VectorEntry = serde_json::from_slice(&bytes)
                    .map_err(|e| format!("Failed to deserialize: {}", e))?;
                Ok(Some(entry))
            }
            None => Ok(None),
        }
    }

    /// 加载所有条目（必要时按 category 过滤）。供 search/count 等共用。
    fn load_all(&self, category: Option<&str>) -> Result<Vec<VectorEntry>, String> {
        let conn = self.lock_conn()?;
        let mut stmt = conn
            .prepare("SELECT payload FROM vec_entries")
            .map_err(|e| format!("Failed to prepare: {}", e))?;
        let rows = stmt
            .query_map([], |row| row.get::<_, Vec<u8>>(0))
            .map_err(|e| format!("Failed to query: {}", e))?;
        let mut out = Vec::new();
        for row in rows {
            let bytes = row.map_err(|e| format!("Failed to iterate: {}", e))?;
            let entry: VectorEntry = serde_json::from_slice(&bytes)
                .map_err(|e| format!("Failed to deserialize: {}", e))?;
            if let Some(cat) = category {
                if entry.category != cat {
                    continue;
                }
            }
            out.push(entry);
        }
        Ok(out)
    }

    /// Semantic search — cosine similarity top-k.
    pub fn semantic_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        category: Option<&str>,
    ) -> Result<Vec<(VectorEntry, f32)>, String> {
        let entries = self.load_all(category)?;
        let mut candidates: Vec<(VectorEntry, f32)> = entries
            .into_iter()
            .map(|entry| {
                let sim = similarity::cosine_similarity(query_embedding, &entry.embedding);
                (entry, sim)
            })
            .collect();
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
        let entries = self.load_all(category)?;
        let mut all_entries: Vec<(String, VectorEntry, f32)> = entries
            .into_iter()
            .map(|entry| {
                let sim = similarity::cosine_similarity(query_embedding, &entry.embedding);
                (entry.id.clone(), entry, sim)
            })
            .collect();

        let semantic_scores: FastMap<String, f32> = all_entries
            .iter()
            .map(|(id, _, score)| (id.clone(), *score))
            .collect();
        let entry_map: FastMap<String, VectorEntry> = all_entries
            .drain(..)
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

        // 先按 entry_map 过滤掉孤儿 ID（FTS 命中但向量表已删除的条目），
        // 再做 truncate(limit)，避免最终返回数量远小于请求的 limit
        let mut results = Vec::new();
        for (id, score) in combined {
            if let Some(entry) = entry_map.get(&id) {
                results.push((id, entry.clone(), score));
                if results.len() >= limit {
                    break;
                }
            }
        }
        Ok(results)
    }

    /// Count entries.
    pub fn count(&self) -> Result<usize, String> {
        let conn = self.lock_conn()?;
        let cnt: i64 = conn
            .query_row("SELECT COUNT(*) FROM vec_entries", [], |row| row.get(0))
            .map_err(|e| format!("Failed to count: {}", e))?;
        Ok(cnt.max(0) as usize)
    }

    /// List all entry IDs.
    pub fn list_ids(&self) -> Result<Vec<String>, String> {
        let conn = self.lock_conn()?;
        let mut stmt = conn
            .prepare("SELECT id FROM vec_entries")
            .map_err(|e| format!("Failed to prepare: {}", e))?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|e| format!("Failed to query: {}", e))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(|e| format!("Failed to iterate: {}", e))?);
        }
        Ok(out)
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

#[allow(dead_code)]
const _ASSERT_DIM: usize = EMBEDDING_DIM;
