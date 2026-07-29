# Knowledge Guide

## Scope

Applies to `src/bin/ai/knowledge/**`.

## Key invariants

1. **Public vs internal tools.** User-facing persistence uses `knowledge_*`
   tools; `memory_*` tools are agent-internal.
2. **Consolidation contract.** `knowledge_consolidate` is a two-phase flow:
   `read_all` then `execute`.
3. **Graceful retrieval fallback.** Embedding is optional and explicit
   knowledge search must degrade to lexical/BM25 retrieval when unavailable or
   failing.
4. **Separated responsibilities.** Keep retrieval, indexing, storage, and sync
   responsibilities separated unless a change genuinely crosses those boundaries.
5. **No automatic recall.** Knowledge is read only through explicit
   `knowledge_*` / `memory_*` tool calls. Do not scan or inject the knowledge
   store automatically while preparing a turn. Notebook remains an independent
   tool-backed context source and must not be coupled to knowledge retrieval.

## Related code areas

- `indexing/` for embeddings and document indexing
- `retrieval/` for keyword, semantic, and hybrid search policy
- `storage/` for persistence
- `sync/` for cross-store synchronization
