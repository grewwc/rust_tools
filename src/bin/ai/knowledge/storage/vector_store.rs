/// Vector store — pure vector CRUD operations.
/// Decoupled from JSONL store; sync is handled by the sync module.
///
/// 后端已从 sled 迁移为 rusqlite（项目内已 bundled），单表 KV：
///   `vec_entries(id TEXT PRIMARY KEY, payload BLOB)`，与原 sled `vec:{id}` 语义对齐。
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::{Connection, OptionalExtension, params};
use rust_tools::cw::SkipMap;
use rustc_hash::FxHashSet;
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
    /// 全表内存缓存（load_all 结果）及其 SQLite data_version。
    cache: Mutex<Option<CachedEntries>>,
}

struct CachedEntries {
    data_version: i64,
    entries: Vec<VectorEntry>,
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
            cache: Mutex::new(None),
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
        *self
            .cache
            .lock()
            .map_err(|e| format!("cache poisoned: {e}"))? = None;
        Ok(())
    }

    /// Delete a vector entry by ID.
    pub fn delete(&self, id: &str) -> Result<bool, String> {
        let conn = self.lock_conn()?;
        let affected = conn
            .execute("DELETE FROM vec_entries WHERE id = ?1", params![id])
            .map_err(|e| format!("Failed to delete: {}", e))?;
        *self
            .cache
            .lock()
            .map_err(|e| format!("cache poisoned: {e}"))? = None;
        Ok(affected > 0)
    }

    /// Delete all vector entries whose IDs are not present in `keep_ids`.
    pub fn delete_except_ids(&self, keep_ids: &[String]) -> Result<usize, String> {
        let keep: FxHashSet<&str> = keep_ids.iter().map(String::as_str).collect();
        let existing = self.list_ids()?;
        let mut removed = 0usize;
        for id in existing {
            if keep.contains(id.as_str()) {
                continue;
            }
            if self.delete(&id)? {
                removed += 1;
            }
        }
        Ok(removed)
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
        // 固定锁顺序为 conn -> cache，并保持 conn 锁直到缓存发布完成，避免同实例
        // 写入在查询完成与缓存发布之间插入。每次命中前用持久连接的 data_version
        // 校验跨进程/另一连接写入。
        let conn = self.lock_conn()?;
        let data_version = conn
            .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
            .map_err(|e| format!("Failed to read data_version: {e}"))?;
        {
            let guard = self
                .cache
                .lock()
                .map_err(|e| format!("cache poisoned: {e}"))?;
            if let Some(cached) = guard
                .as_ref()
                .filter(|cached| cached.data_version == data_version)
            {
                return Ok(match category {
                    Some(cat) => cached
                        .entries
                        .iter()
                        .filter(|e| e.category == cat)
                        .cloned()
                        .collect(),
                    None => cached.entries.clone(),
                });
            }
        }
        let mut stmt = conn
            .prepare("SELECT payload FROM vec_entries")
            .map_err(|e| format!("Failed to prepare: {}", e))?;
        let rows = stmt
            .query_map([], |row| row.get::<_, Vec<u8>>(0))
            .map_err(|e| format!("Failed to query: {}", e))?;
        // 始终加载全量条目并缓存完整集，避免首次调用带 category 时缓存了
        // 过滤子集导致后续 load_all(None) 返回不完整结果。
        let mut all = Vec::new();
        for row in rows {
            let bytes = row.map_err(|e| format!("Failed to iterate: {}", e))?;
            let entry: VectorEntry = serde_json::from_slice(&bytes)
                .map_err(|e| format!("Failed to deserialize: {}", e))?;
            all.push(entry);
        }
        drop(stmt);
        *self
            .cache
            .lock()
            .map_err(|e| format!("cache poisoned: {e}"))? = Some(CachedEntries {
            data_version,
            entries: all.clone(),
        });
        Ok(match category {
            Some(cat) => all.iter().filter(|e| e.category == cat).cloned().collect(),
            None => all,
        })
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

        let semantic_scores: SkipMap<String, f32> = all_entries
            .iter()
            .map(|(id, _, score)| (id.clone(), *score))
            .collect();
        let entry_map: SkipMap<String, VectorEntry> = all_entries
            .drain(..)
            .map(|(id, entry, _)| (id, entry))
            .collect();

        let bm25_max = bm25_results.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
        let bm25_normalized: SkipMap<String, f32> = if bm25_max > 0.0 {
            bm25_results
                .into_iter()
                .map(|(id, score)| (id, score / bm25_max))
                .collect()
        } else {
            SkipMap::default()
        };

        let mut all_ids: rust_tools::cw::SkipSet<String> = rust_tools::cw::SkipSet::default();
        all_ids.extend(bm25_normalized.keys().cloned());
        all_ids.extend(semantic_scores.keys().cloned());

        let mut combined: Vec<(String, f32)> = Vec::new();
        for id in all_ids.iter() {
            let bm25 = bm25_normalized.get_ref(id).copied().unwrap_or(0.0);
            let semantic = semantic_scores.get_ref(id).copied().unwrap_or(0.0);
            let final_score = (1.0 - vector_weight) * bm25 + vector_weight * semantic;
            combined.push((id.clone(), final_score));
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
        let keep_ids: Vec<String> = entries.iter().map(|(id, _, _, _)| id.clone()).collect();
        self.delete_except_ids(&keep_ids)?;

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

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeEmbedder;

    impl VectorEmbedder for FakeEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
            Ok(vec![text.len() as f32, 1.0])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
            texts.iter().map(|text| self.embed(text)).collect()
        }
    }

    fn cleanup_sqlite(path: &Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }

    #[test]
    fn rebuild_from_entries_deletes_stale_vector_ids() {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_vector_rebuild_{ts}.db"));
        cleanup_sqlite(&path);

        let store = VectorStore::new(&path, Box::new(FakeEmbedder)).unwrap();
        store
            .upsert(VectorEntry {
                id: "stale_legacy_hash".to_string(),
                content: "old content".to_string(),
                category: "user_memory".to_string(),
                tags: vec![],
                embedding: vec![1.0, 0.0],
                timestamp: 0,
            })
            .unwrap();

        let rebuilt = store
            .rebuild_from_entries(&[
                (
                    "mem_current".to_string(),
                    "coding_guideline".to_string(),
                    "Do: keep tests focused.".to_string(),
                    vec!["principle".to_string()],
                ),
                (
                    "mem_other".to_string(),
                    "user_memory".to_string(),
                    "Project fact".to_string(),
                    vec![],
                ),
            ])
            .unwrap();

        assert_eq!(rebuilt, 2);
        assert_eq!(store.count().unwrap(), 2);
        assert!(store.get("stale_legacy_hash").unwrap().is_none());
        assert!(store.get("mem_current").unwrap().is_some());

        cleanup_sqlite(&path);
    }

    #[test]
    fn load_all_caches_full_dataset_not_filtered_subset() {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_vector_cache_{ts}.db"));
        cleanup_sqlite(&path);

        let store = VectorStore::new(&path, Box::new(FakeEmbedder)).unwrap();
        store
            .upsert(VectorEntry {
                id: "e1".to_string(),
                content: "first".to_string(),
                category: "user_memory".to_string(),
                tags: vec![],
                embedding: vec![1.0, 0.0],
                timestamp: 0,
            })
            .unwrap();
        store
            .upsert(VectorEntry {
                id: "e2".to_string(),
                content: "second".to_string(),
                category: "coding_guideline".to_string(),
                tags: vec![],
                embedding: vec![0.0, 1.0],
                timestamp: 0,
            })
            .unwrap();

        // 首次调用带 category 过滤——修复前会缓存过滤子集。
        let filtered = store.load_all(Some("user_memory")).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "e1");

        // 后续无过滤调用必须返回全量，而非缓存的子集。
        let all = store.load_all(None).unwrap();
        assert_eq!(
            all.len(),
            2,
            "load_all(None) must return all entries, not cached subset"
        );

        // 交叉验证：换一个 category 也能命中缓存并正确过滤。
        let other = store.load_all(Some("coding_guideline")).unwrap();
        assert_eq!(other.len(), 1);
        assert_eq!(other[0].id, "e2");

        cleanup_sqlite(&path);
    }

    #[test]
    fn load_all_invalidates_cache_after_external_connection_write() {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_vector_external_cache_{ts}.db"));
        cleanup_sqlite(&path);

        let store = VectorStore::new(&path, Box::new(FakeEmbedder)).unwrap();
        store
            .upsert(VectorEntry {
                id: "e1".to_string(),
                content: "first".to_string(),
                category: "user_memory".to_string(),
                tags: vec![],
                embedding: vec![1.0, 0.0],
                timestamp: 0,
            })
            .unwrap();
        assert_eq!(store.load_all(None).unwrap().len(), 1);

        let external = Connection::open(&path).unwrap();
        let entry = VectorEntry {
            id: "e2".to_string(),
            content: "second".to_string(),
            category: "coding_guideline".to_string(),
            tags: vec![],
            embedding: vec![0.0, 1.0],
            timestamp: 0,
        };
        let payload = serde_json::to_vec(&entry).unwrap();
        external
            .execute(
                "INSERT INTO vec_entries (id, payload) VALUES (?1, ?2)",
                params![entry.id, payload],
            )
            .unwrap();

        assert_eq!(store.load_all(None).unwrap().len(), 2);
        cleanup_sqlite(&path);
    }

    #[test]
    fn concurrent_upsert_and_load_all_publish_fresh_cache() {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_vector_concurrent_cache_{ts}.db"));
        cleanup_sqlite(&path);

        let store = std::sync::Arc::new(VectorStore::new(&path, Box::new(FakeEmbedder)).unwrap());
        store
            .upsert(VectorEntry {
                id: "before".to_string(),
                content: "before".to_string(),
                category: "user_memory".to_string(),
                tags: vec![],
                embedding: vec![1.0, 0.0],
                timestamp: 0,
            })
            .unwrap();
        assert_eq!(store.load_all(None).unwrap().len(), 1);

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let reader_store = std::sync::Arc::clone(&store);
        let reader_barrier = std::sync::Arc::clone(&barrier);
        let reader = std::thread::spawn(move || {
            reader_barrier.wait();
            for _ in 0..100 {
                let _ = reader_store.load_all(None).unwrap();
            }
        });
        barrier.wait();
        store
            .upsert(VectorEntry {
                id: "after".to_string(),
                content: "after".to_string(),
                category: "user_memory".to_string(),
                tags: vec![],
                embedding: vec![0.0, 1.0],
                timestamp: 0,
            })
            .unwrap();
        reader.join().unwrap();

        let ids: std::collections::HashSet<_> = store
            .load_all(None)
            .unwrap()
            .into_iter()
            .map(|entry| entry.id)
            .collect();
        assert_eq!(
            ids,
            std::collections::HashSet::from(["before".to_string(), "after".to_string()])
        );
        cleanup_sqlite(&path);
    }
}

#[allow(dead_code)]
const _ASSERT_DIM: usize = EMBEDDING_DIM;
