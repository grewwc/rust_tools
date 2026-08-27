//! Knowledge module: shared types and the semantic search index.
//!
//! - `types.rs` / `entry.rs`: shared types and the path-leak guard.
//! - `indexing/similarity.rs`: lexical tokenization/stopwords (BM25 half of
//!   search) plus vector cosine similarity.
//! - `indexing/embedder.rs`: remote embedding provider (config-gated).
//! - `storage/vector_store.rs`: SQLite-backed vector index.
//! - `config.rs`: hybrid-search tuning.
//!
//! The vector index is a derived artifact rebuilt from the canonical memory
//! store (`tools/storage/memory_store.rs`); orchestration and the hybrid
//! merge live in `tools/storage/rag_store.rs`.

pub mod config;
pub mod entry;
pub mod indexing;
pub mod storage;
pub mod types;
