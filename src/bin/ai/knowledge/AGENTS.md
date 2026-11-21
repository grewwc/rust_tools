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
   `memory_*` tools — agent-internal memory uses the same `knowledge_*` tools.
   The registered `knowledge_save` tool is fixed to the Global scope
   (`execute_knowledge_save` in `tools/knowledge_tools.rs`); the scoped
   `agent_memory_save` path (`execute_memory_save` in `tools/service/memory.rs`,
   `MemoryOwnerScope::Scoped`) is exercised only by test callers.
2. **Consolidation contract.** `knowledge_consolidate` is a two-phase flow:
   `read_all` then `execute`. `execute` appends each listed source's full original
   note (blank notes are skipped) to the merged entry while auto-deleting those
   sources. Preservation is bounded by the `read_all` view (`store.all()`):
   `source_ids` outside that view are reported as unknown and never appended,
   while `delete_ids` are deleted across rotation archives even when their
   content was never visible to `read_all`. Content is otherwise discarded only
   for entries removed via `delete_ids` without a `save_entries[].source_ids`
   listing — consolidation never silently compresses or drops listed sources.
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
