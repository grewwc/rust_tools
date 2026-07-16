# Knowledge Guide

## Scope

Applies to `src/bin/ai/knowledge/**`.
Keep this file brief; inspect the touched module directly before broad refactors.

## Key invariants

1. **Public vs internal tools.** User-facing persistence uses `knowledge_*`
   tools; `memory_*` tools are agent-internal.
2. **Consolidation contract.** `knowledge_consolidate` is a two-phase flow:
   `read_all` then `execute`.
3. **Graceful recall fallback.** Embedding is optional and must degrade to
   lexical/BM25 recall when unavailable or failing.
4. **Separated responsibilities.** Keep retrieval, indexing, storage, and sync
   responsibilities separated unless a change genuinely crosses those boundaries.

## Related code areas

- `indexing/` for embeddings and document indexing
- `retrieval/` for recall policy
- `storage/` for persistence
- `sync/` for cross-store synchronization
