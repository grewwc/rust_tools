# Knowledge Guide

## Scope

Applies to `src/bin/ai/knowledge/**`. Shared types and the semantic-index
building blocks: `types.rs` (`Category`), `entry.rs` (path-leak guard),
`config.rs` (hybrid-search weight), `indexing/similarity.rs` (lexical scoring +
`cosine_similarity`), `indexing/embedder.rs` (remote OpenAI-compatible
embedding provider, default `doubao-embedding-vision`), and
`storage/vector_store.rs` (SQLite-backed vector index). The orchestration layer
lives in `tools/storage/rag_store.rs` (rebuild + hybrid merge) and the
`knowledge_semantic_search` / `knowledge_rebuild_index` tools in
`tools/rag_tools.rs`.

## Key invariants

1. **Public vs internal tools.** All registry tools are `knowledge_*`
   (`knowledge_save`/`forget`/`search`/`list`/`consolidate`); there are no separate
   `memory_*` tools — agent-internal memory uses the same `knowledge_*` tools
   scoped by `MemoryOwnerScope` (e.g. the `agent_memory_save` path in
   `tools/service/memory.rs`).
2. **Consolidation contract.** `knowledge_consolidate` is a two-phase flow:
   `read_all` then `execute`. Merges are lossless: `execute` `save_entries[].source_ids`
   must list every source entry being consolidated, and the tool appends each source's
   full original note to the merged entry (see `execute_consolidation` in
   `tools/knowledge_tools.rs`) while auto-deleting those sources (their content is fully
   preserved in the merged entry). Content is discarded only for entries removed via
   `delete_ids` without being listed in any `save_entries[].source_ids` — consolidation
   never silently compresses or drops original content.
3. **Semantic search is explicit and derived.** The vector index is a pure
   derived artifact of the canonical memory store, rebuilt lazily on first use
   (or when the configured embedding model changed — see the model fingerprint
   in `rag_store.rs`) and explicitly via `knowledge_rebuild_index`. It is never
   auto-injected into turns (invariant 4). Lexical ranking stays in
   `MemoryStore::search` (BM25 + priority weight); hybrid search merges that
   with the vector index using `hybrid_vector_weight`.
4. **No automatic recall.** Knowledge is read only through explicit
   `knowledge_*` tool calls - never scan or inject the store
   automatically while preparing a turn. Notebook is an independent tool-backed
   context source, not coupled to knowledge retrieval.
