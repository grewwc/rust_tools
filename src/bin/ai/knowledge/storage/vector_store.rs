//! SQLite-backed vector store for semantic knowledge search.
//!
//! Each row stores a `VectorEntry` (id + serialized JSON payload). The index
//! is a pure derived artifact: it is rebuilt from the canonical memory store
//! by `RagStore` and holds no source-of-truth data of its own.

use std::path::Path;

use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::ai::knowledge::indexing::similarity::cosine_similarity;

/// A single indexed memory entry with its embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    pub id: String,
    pub content: String,
    pub category: String,
    pub tags: Vec<String>,
    pub embedding: Vec<f32>,
    pub timestamp: i64,
}

/// SQLite-backed vector store.
pub struct VectorStore {
    conn: Connection,
}

impl VectorStore {
    /// Open (or create) the store at `path`.
    pub fn new(path: &Path) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS vec_entries (
                 id TEXT PRIMARY KEY,
                 payload BLOB NOT NULL
             );
             CREATE TABLE IF NOT EXISTS vec_meta (
                 key TEXT PRIMARY KEY,
                 value TEXT NOT NULL
             );",
        )?;
        Ok(Self { conn })
    }

    /// Number of indexed entries.
    pub fn len(&self) -> usize {
        self.conn
            .query_row("SELECT COUNT(*) FROM vec_entries", [], |r| r.get::<_, i64>(0))
            .unwrap_or(0) as usize
    }

    /// True when the index has no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert or replace entries in one transaction.
    pub fn upsert_batch(&mut self, entries: Vec<VectorEntry>) -> rusqlite::Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt =
                tx.prepare("INSERT OR REPLACE INTO vec_entries (id, payload) VALUES (?1, ?2)")?;
            for e in entries {
                let payload = serde_json::to_vec(&e).map_err(|err| {
                    rusqlite::Error::ToSqlConversionFailure(Box::new(err))
                })?;
                stmt.execute(rusqlite::params![e.id, payload])?;
            }
        }
        tx.commit()
    }

    /// Remove entries by id.
    pub fn delete(&mut self, ids: &[String]) -> rusqlite::Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare("DELETE FROM vec_entries WHERE id = ?1")?;
            for id in ids {
                stmt.execute(rusqlite::params![id])?;
            }
        }
        tx.commit()
    }

    /// Load all entries.
    pub fn all(&self) -> Vec<VectorEntry> {
        let mut stmt = self
            .conn
            .prepare("SELECT payload FROM vec_entries")
            .expect("select vec_entries");
        let rows = stmt
            .query_map([], |r| r.get::<_, Vec<u8>>(0))
            .expect("query vec_entries");
        rows.filter_map(|r| r.ok())
            .filter_map(|b| serde_json::from_slice(&b).ok())
            .collect()
    }

    /// Fetch one entry by id.
    pub fn get(&self, id: &str) -> Option<VectorEntry> {
        self.conn
            .query_row("SELECT payload FROM vec_entries WHERE id = ?1", [id], |r| {
                r.get::<_, Vec<u8>>(0)
            })
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
    }

    /// Store a key/value metadata row (e.g. the embedding model fingerprint).
    pub fn set_meta(&self, key: &str, value: &str) -> rusqlite::Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO vec_meta (key, value) VALUES (?1, ?2)",
            rusqlite::params![key, value],
        )?;
        Ok(())
    }

    /// Read a metadata row.
    pub fn get_meta(&self, key: &str) -> Option<String> {
        self.conn
            .query_row("SELECT value FROM vec_meta WHERE key = ?1", [key], |r| r.get(0))
            .ok()
    }

    /// Cosine-similarity top-k over all entries, optionally filtered by category.
    pub fn semantic_search(
        &self,
        query_embedding: &[f32],
        category: Option<&str>,
        limit: usize,
    ) -> Vec<(VectorEntry, f32)> {
        let mut scored: Vec<(VectorEntry, f32)> = self
            .all()
            .into_iter()
            .filter(|e| category.map_or(true, |c| e.category == c))
            .map(|e| {
                let s = cosine_similarity(query_embedding, &e.embedding);
                (e, s)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> VectorStore {
        // Unique path per test; SQLite keeps WAL files open while the
        // connection is alive, so deleting the file mid-test breaks I/O.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("vec_test_{}_{}.db", std::process::id(), nanos));
        VectorStore::new(&path).expect("open store")
    }

    #[test]
    fn upsert_semantic_search_rank_by_cosine() {
        let mut store = temp_store();
        store
            .upsert_batch(vec![
                VectorEntry {
                    id: "a".into(),
                    content: "rust".into(),
                    category: "general".into(),
                    tags: vec![],
                    embedding: vec![1.0, 0.0],
                    timestamp: 1,
                },
                VectorEntry {
                    id: "b".into(),
                    content: "python".into(),
                    category: "general".into(),
                    tags: vec![],
                    embedding: vec![0.0, 1.0],
                    timestamp: 2,
                },
            ])
            .expect("upsert");
        let hits = store.semantic_search(&[1.0, 0.1], None, 2);
        assert_eq!(hits[0].0.id, "a");
        assert!(hits[0].1 > hits[1].1);
        assert_eq!(store.get("a").unwrap().content, "rust");
        store.delete(&["a".to_string()]).unwrap();
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn meta_roundtrip() {
        let store = temp_store();
        store.set_meta("model", "doubao-embedding-vision").unwrap();
        assert_eq!(
            store.get_meta("model").as_deref(),
            Some("doubao-embedding-vision")
        );
    }
}
