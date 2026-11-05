# Knowledge Guide

## Scope

Applies to `src/bin/ai/knowledge/**`: shared entry types, derived semantic-index
building blocks, and verified distilled entries for automatic recall. Rebuild
and hybrid-search orchestration belongs to `tools/storage/rag_store.rs`; the
Tools guide owns schemas and execution for the `knowledge_*` family.

## Key invariants

1. **Public vs internal tools.** Registry knowledge tools follow the existing
   `knowledge_*` family, including `knowledge_save` and `knowledge_consolidate`
   (see **Names & paths** in `src/bin/ai/tools/AGENTS.md`); there are no separate
   `memory_*` tools — agent-internal memory uses the same `knowledge_*` tools
   scoped by `MemoryOwnerScope` (e.g. the `agent_memory_save` path in
   `tools/service/memory.rs`).
2. **Consolidation contract.** `knowledge_consolidate` is a two-phase flow:
   `read_all` then `execute`. Merges are lossless: `execute` `save_entries[].source_ids`
   must list every source entry being consolidated, and the tool appends each source's
   full original note to the merged entry while auto-deleting those sources (their
   content is fully preserved in the merged entry). Content is discarded only for
   entries removed via `delete_ids` without being listed in any
   `save_entries[].source_ids` — consolidation never silently compresses or drops
   original content.
3. **Semantic search is derived; automatic recall is restricted.** The vector
   index is a pure derived artifact of the canonical memory store, rebuilt lazily
   on first use (or when the configured embedding model changed — see the model
   fingerprint in `rag_store.rs`) and explicitly via `knowledge_rebuild_index`.
   General knowledge is read through explicit `knowledge_*` tool calls. The driver
   may automatically select verified, current-project distilled entries for real
   foreground tasks, subject to knowledge permissions. Recalled content is bounded
   historical reference data in the request projection under **History is truth**
   and **Derived-context provenance** in `src/bin/ai/AGENTS.md`. Lexical ranking
   stays in `MemoryStore::search` (BM25 + priority weight); hybrid search merges
   that with the vector index using `hybrid_vector_weight`. The notebook
   (`a -n` / `-ns` memo search, `driver/note_search.rs`) is an independent
   context source, not coupled to knowledge retrieval.
