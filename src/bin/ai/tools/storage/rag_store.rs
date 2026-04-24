/// RAG (Retrieval-Augmented Generation) 向量存储
///
/// 使用 AIOS 统一 embedding provider，
/// 使用 knowledge::storage::VectorStore 持久化存储向量索引。
/// 支持余弦相似度检索和混合 BM25 + 向量检索。
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use dirs;
use serde::{Deserialize, Serialize};

use crate::ai::knowledge::{
    storage::vector_store::{VectorEntry, VectorStore},
};
use crate::ai::tools::storage::memory_store::MemoryStore;

/// 带向量的知识条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagEntry {
    /// 与 memory_store 中的 id 对应
    pub id: String,
    /// 原始文本内容（用于 embedding 生成）
    pub content: String,
    /// 可选的类别信息，用于过滤
    pub category: String,
    /// 可选的标签，用于增强语义
    pub tags: Vec<String>,
    /// 向量（扁平化的 f32 数组）
    pub embedding: Vec<f32>,
    /// 时间戳（毫秒）
    pub timestamp: u64,
}

/// RAG 向量存储
pub struct RagStore {
    store: VectorStore,
    index_path: PathBuf,
}

impl RagStore {
    /// 从默认路径创建 RAG Store
    pub fn new() -> Result<Self, String> {
        let base = dirs::config_dir().ok_or("Cannot determine config directory")?;
        let index_path = base.join("rust_tools/rag_index");
        Self::with_path(&index_path)
    }

    /// 从指定路径创建
    pub fn with_path(path: &Path) -> Result<Self, String> {
        Ok(Self {
            store: VectorStore::with_global_provider(path)?,
            index_path: path.to_path_buf(),
        })
    }

    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>, String> {
        self.store.embed_text(text)
    }

    pub fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        self.store.embed_texts(texts)
    }

    pub fn upsert(&self, entry: RagEntry) -> Result<(), String> {
        self.store.upsert(entry.into())
    }

    pub fn delete(&self, id: &str) -> Result<bool, String> {
        self.store.delete(id)
    }

    /// 语义搜索 — 余弦相似度 top-k
    pub fn semantic_search(
        &self,
        query: &str,
        limit: usize,
        category: Option<&str>,
    ) -> Result<Vec<(RagEntry, f32)>, String> {
        // Lazy rebuild check: if index is empty, try to rebuild
        if self.count()? == 0 {
            let rebuilt = self.rebuild_from_memory()?;
            if rebuilt > 0 {
                eprintln!("[RAG lazy rebuild triggered: {} entries", rebuilt);
            }
        }

        let query_embedding = self.embed_text(query)?;
        self.store
            .semantic_search(&query_embedding, limit, category)
            .map(|results| {
                results
                    .into_iter()
                    .map(|(entry, score)| (entry.into(), score))
                    .collect()
            })
    }

    /// 混合搜索 — BM25 + 语义加权融合
    pub fn hybrid_search(
        &self,
        query: &str,
        bm25_results: Vec<(String, f32)>,
        limit: usize,
        category: Option<&str>,
        vector_weight: f32,
    ) -> Result<Vec<(String, RagEntry, f32)>, String> {
        let query_embedding = self.embed_text(query)?;
        self.store
            .hybrid_search(&query_embedding, bm25_results, limit, category, vector_weight)
            .map(|results| {
                results
                    .into_iter()
                    .map(|(id, entry, score)| (id, entry.into(), score))
                    .collect()
            })
    }

    pub fn count(&self) -> Result<usize, String> {
        self.store.count()
    }

    pub fn list_ids(&self) -> Result<Vec<String>, String> {
        self.store.list_ids()
    }

    pub fn get_entry(&self, id: &str) -> Result<Option<RagEntry>, String> {
        self.store.get(id).map(|entry| entry.map(Into::into))
    }

    /// 重建索引（从 memory_store 同步）
    pub fn rebuild_from_memory(&self) -> Result<usize, String> {
        let store = MemoryStore::from_env_or_config();
        let entries = store.all()?;

        let texts: Vec<String> = entries
            .iter()
            .map(|e| {
                let mut text = format!("{}: {}", e.category, e.note);
                if !e.tags.is_empty() {
                    text.push_str(&format!(" [tags: {}]", e.tags.join(", ")));
                }
                if let Some(src) = &e.source {
                    text.push_str(&format!(" (source: {})", src));
                }
                text
            })
            .collect();

        let embeddings = self.embed_texts(&texts)?;

        let mut count = 0;
        for (entry, embedding) in entries.into_iter().zip(embeddings.into_iter()) {
            let id = entry
                .id
                .clone()
                .unwrap_or_else(|| format!("{:x}", md5::compute(&entry.note)));
            let content = format!("{}: {}", entry.category, entry.note);
            self.upsert(RagEntry {
                id,
                content,
                category: entry.category.clone(),
                tags: entry.tags.clone(),
                embedding,
                timestamp: entry.timestamp.parse().unwrap_or(0),
            })?;
            count += 1;
        }
        Ok(count)
    }

    pub fn path(&self) -> &Path {
        &self.index_path
    }
}

impl From<RagEntry> for VectorEntry {
    fn from(value: RagEntry) -> Self {
        Self {
            id: value.id,
            content: value.content,
            category: value.category,
            tags: value.tags,
            embedding: value.embedding,
            timestamp: value.timestamp,
        }
    }
}

impl From<VectorEntry> for RagEntry {
    fn from(value: VectorEntry) -> Self {
        Self {
            id: value.id,
            content: value.content,
            category: value.category,
            tags: value.tags,
            embedding: value.embedding,
            timestamp: value.timestamp,
        }
    }
}

/// Implement the VectorStoreSync trait for compatibility with the new knowledge sync module.
impl crate::ai::knowledge::sync::knowledge_sync::VectorStoreSync for RagStore {
    fn upsert_entry(
        &self,
        id: String,
        content: String,
        category: String,
        tags: Vec<String>,
        embedding: Vec<f32>,
    ) -> Result<(), String> {
        self.upsert(RagEntry {
            id,
            content,
            category,
            tags,
            embedding,
            timestamp: 0,
        })
    }

    fn delete_entry(&self, id: &str) -> Result<bool, String> {
        self.delete(id)
    }

    fn embed_text(&self, text: &str) -> Result<Vec<f32>, String> {
        self.embed_text(text)
    }
    
    fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        // Use the optimized batch embedding
        self.embed_texts(texts)
    }
}

/// 全局 RAG Store
static GLOBAL_RAG_STORE: OnceLock<Mutex<Option<RagStore>>> = OnceLock::new();

pub fn get_rag_store() -> Result<std::sync::MutexGuard<'static, Option<RagStore>>, String> {
    let cell = GLOBAL_RAG_STORE.get_or_init(|| Mutex::new(None));
    cell.lock().map_err(|e| format!("Lock poisoned: {}", e))
}

pub fn init_rag_store() -> Result<(), String> {
    let store = RagStore::new()?;
    let mut guard = get_rag_store()?;
    *guard = Some(store);
    Ok(())
}

pub fn ensure_rag_store() -> Result<(), String> {
    let mut guard = get_rag_store()?;
    if guard.is_none() {
        let store = RagStore::new()?;

        // Lazy rebuild: 如果 RAG 索引为空但 memory_store 有数据，自动同步
        let index_count = store.count()?;
        if index_count == 0 {
            if let Ok(rebuilt) = store.rebuild_from_memory() {
                if rebuilt > 0 {
                    eprintln!(
                        "[RAG index auto-rebuilt: {} entries from memory store",
                        rebuilt
                    );
                }
            }
        }

        *guard = Some(store);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((crate::ai::knowledge::indexing::similarity::cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(crate::ai::knowledge::indexing::similarity::cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((crate::ai::knowledge::indexing::similarity::cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_eq!(crate::ai::knowledge::indexing::similarity::cosine_similarity(&[], &[]), 0.0);
    }

    fn make_entry(
        id: &str,
        content: &str,
        category: &str,
        tags: Vec<&str>,
        ts: u64,
        store: &RagStore,
    ) -> RagEntry {
        let emb = store.embed_text(content).unwrap();
        RagEntry {
            id: id.to_string(),
            content: content.to_string(),
            category: category.to_string(),
            tags: tags.into_iter().map(String::from).collect(),
            embedding: emb,
            timestamp: ts,
        }
    }

    #[test]
    #[ignore = "requires fastembed ONNX model"]
    fn test_rag_store_crud_and_semantic_search() {
        let tmp = std::env::temp_dir().join(format!("rag_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::create_dir_all(&tmp);

        let store = RagStore::with_path(&tmp).unwrap();

        // 插入条目
        let e1 = make_entry(
            "ci_cd_1",
            "CI/CD 使用 Jenkins 自动化部署",
            "deploy",
            vec!["ci", "jenkins"],
            1000,
            &store,
        );
        store.upsert(e1).unwrap();

        let e2 = make_entry(
            "review_1",
            "代码审查必须通过两个 reviewer 批准",
            "process",
            vec!["review"],
            2000,
            &store,
        );
        store.upsert(e2).unwrap();

        assert_eq!(store.count().unwrap(), 2);

        // 语义搜索：英文搜中文内容
        let results = store
            .semantic_search("how do we deploy code?", 5, None)
            .unwrap();
        assert!(!results.is_empty(), "semantic search should find results");
        assert_eq!(results[0].0.id, "ci_cd_1");
        assert!(results[0].1 > 0.0);

        // 按 category 过滤
        let filtered = store
            .semantic_search("code review", 5, Some("process"))
            .unwrap();
        assert!(!filtered.is_empty());
        assert_eq!(filtered[0].0.id, "review_1");

        let no_results = store
            .semantic_search("code review", 5, Some("nonexistent"))
            .unwrap();
        assert!(no_results.is_empty());

        // 删除
        store.delete("ci_cd_1").unwrap();
        assert_eq!(store.count().unwrap(), 1);

        let after_delete = store.semantic_search("deploy", 5, None).unwrap();
        assert!(after_delete.is_empty() || after_delete.iter().all(|(e, _)| e.id != "ci_cd_1"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    #[ignore = "requires fastembed ONNX model"]
    fn test_rag_store_get_entry() {
        let tmp = std::env::temp_dir().join(format!("rag_get_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::create_dir_all(&tmp);

        let store = RagStore::with_path(&tmp).unwrap();
        let e = make_entry("test_1", "test content here", "test", vec![], 0, &store);
        store.upsert(e).unwrap();

        let found = store.get_entry("test_1").unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().content, "test content here");

        let not_found = store.get_entry("nonexistent").unwrap();
        assert!(not_found.is_none());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    #[ignore = "requires fastembed ONNX model"]
    fn test_rag_store_hybrid_search() {
        let tmp = std::env::temp_dir().join(format!("rag_hybrid_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::create_dir_all(&tmp);

        let store = RagStore::with_path(&tmp).unwrap();

        let texts: Vec<String> = vec![
            "Jenkins CI/CD 自动化部署流程".into(),
            "单元测试使用 cargo test".into(),
            "代码风格使用 rustfmt 格式化".into(),
        ];
        let embeddings = store.embed_texts(&texts).unwrap();

        for (i, text) in texts.iter().enumerate() {
            store
                .upsert(RagEntry {
                    id: format!("entry_{}", i),
                    content: text.to_string(),
                    category: "misc".to_string(),
                    tags: vec![],
                    embedding: embeddings[i].clone(),
                    timestamp: i as u64,
                })
                .unwrap();
        }

        // hybrid_search 需要 BM25 results 作为输入
        let bm25_results: Vec<(String, f32)> =
            vec![("entry_0".to_string(), 0.8), ("entry_1".to_string(), 0.3)];
        let results = store
            .hybrid_search("部署", bm25_results, 5, None, 0.4)
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].1.id, "entry_0");

        for (_, _, score) in &results {
            assert!(
                *score >= 0.0 && *score <= 1.0,
                "score {} out of range",
                score
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    #[ignore = "requires fastembed ONNX model"]
    fn test_rag_store_rebuild_simulation() {
        let tmp = std::env::temp_dir().join(format!("rag_rebuild_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::create_dir_all(&tmp);

        let store = RagStore::with_path(&tmp).unwrap();
        assert_eq!(store.count().unwrap(), 0);

        // 模拟 rebuild：手动插入类似 memory_store 的条目
        let texts: Vec<String> = vec![
            "test: 第一条测试知识".into(),
            "test: 第二条测试知识".into(),
            "deploy: 部署用 Jenkins".into(),
        ];
        let embeddings = store.embed_texts(&texts).unwrap();

        for (i, text) in texts.iter().enumerate() {
            store
                .upsert(RagEntry {
                    id: format!("mem_entry_{}", i),
                    content: text.to_string(),
                    category: "test".to_string(),
                    tags: vec![],
                    embedding: embeddings[i].clone(),
                    timestamp: (i as u64) * 1000,
                })
                .unwrap();
        }

        assert_eq!(store.count().unwrap(), 3);

        let results = store.semantic_search("部署", 5, None).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].0.content.contains("部署"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    #[ignore = "requires fastembed ONNX model"]
    fn test_rag_store_persistence() {
        let tmp = std::env::temp_dir().join(format!("rag_persist_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::create_dir_all(&tmp);

        // 写入
        {
            let store = RagStore::with_path(&tmp).unwrap();
            let e = make_entry(
                "persist_1",
                "this should survive restart",
                "persist",
                vec![],
                999,
                &store,
            );
            store.upsert(e).unwrap();
        }

        // 重新打开
        {
            let store = RagStore::with_path(&tmp).unwrap();
            assert_eq!(store.count().unwrap(), 1);
            let found = store.get_entry("persist_1").unwrap();
            assert!(found.is_some());
            assert_eq!(found.unwrap().content, "this should survive restart");

            let results = store.semantic_search("persistent", 5, None).unwrap();
            assert!(!results.is_empty());
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_memory_store_delete_by_id() {
        use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};

        let tmp = std::env::temp_dir().join(format!("mem_del_{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&tmp).ok();

        let store = MemoryStore::for_tests_with_path(tmp.clone());

        let entry = AgentMemoryEntry {
            id: Some("del_test_1".to_string()),
            category: "test".to_string(),
            note: "to be deleted".to_string(),
            tags: vec![],
            timestamp: "1000".to_string(),
            source: None,
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
        };
        store.append(&entry).unwrap();

        let entry2 = AgentMemoryEntry {
            id: Some("del_test_2".to_string()),
            category: "test".to_string(),
            note: "to keep".to_string(),
            tags: vec![],
            timestamp: "2000".to_string(),
            source: None,
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
        };
        store.append(&entry2).unwrap();

        assert_eq!(store.all().unwrap().len(), 2);

        let deleted = store.delete_by_id("del_test_1").unwrap();
        assert!(deleted.is_some());
        assert_eq!(deleted.unwrap().note, "to be deleted");

        let remaining = store.all().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id.as_deref(), Some("del_test_2"));

        let not_found = store.delete_by_id("nonexistent").unwrap();
        assert!(not_found.is_none());

        let _ = std::fs::remove_file(&tmp);
    }
}
