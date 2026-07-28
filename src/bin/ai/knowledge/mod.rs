/// Knowledge system — unified knowledge management for the AI agent.
///
/// ## Architecture
///
/// ```text
/// ┌─────────────────────────────────────────────────────────────┐
/// │                      TOOL LAYER                              │
/// │  knowledge_tools.rs → knowledge::retrieval                  │
/// │  rag_tools.rs       → knowledge::retrieval                  │
/// └──────────────────────────┬──────────────────────────────────┘
///                            │
/// ┌──────────────────────────▼──────────────────────────────────┐
/// │                       KNOWLEDGE MODULE                       │
/// │  ├── config.rs         — Search weights and defaults        │
/// │  ├── types.rs          — Category enum, KnowledgeType       │
/// │  ├── entry.rs          — KnowledgeEntry (unified)           │
/// │  ├── indexing/         — BM25, embeddings, similarity       │
/// │  ├── storage/          — JSONL store, vector store          │
/// │  ├── sync/             — ID generation, store coordination  │
/// │  └── retrieval/        — keyword, semantic, hybrid          │
/// └─────────────────────────────────────────────────────────────┘
/// ```
///
/// ## Key Design Decisions
///
/// 1. **Category enum** — Single source of truth, replaces 5 scattered string lists
/// 2. **KnowledgeSync** — Only place that coordinates JSONL ↔ vector sync
/// 3. **IdGenerator** — Only place that computes entry IDs
/// 4. **Config** — All magic numbers centralized and configurable
/// 5. **Decoupled stores** — VectorStore doesn't know about JsonlStore
///
/// ## Knowledge vs Memory
///
/// - **Knowledge**: User-facing, explicitly saved facts (project info, decisions, preferences)
/// - **Memory**: Saved behavior rules and session-scoped internal notes
///
/// Both share the same storage backend but are distinguished by category.
pub mod config;
pub mod entry;
pub mod indexing;
pub mod retrieval;
pub mod storage;
pub mod sync;
pub mod types;
