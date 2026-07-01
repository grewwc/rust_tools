# Knowledge Guide

## Scope

Applies to `src/bin/ai/knowledge/**`.
Keep this file brief; inspect the touched module directly before broad refactors.

## Key invariants

1. User-facing persistence uses `knowledge_*` tools; `memory_*` tools are
   agent-internal.
2. `knowledge_consolidate` is a two-phase flow (`read_all` -> `execute`); keep
   that contract intact.
3. Embedding is optional and must degrade gracefully to lexical/BM25 recall on
   failure.
4. Keep retrieval, indexing, and storage responsibilities separated unless the
   change genuinely crosses those boundaries.

## Related code areas

- `indexing/` for embeddings and document indexing
- `retrieval/` for recall policy
- `storage/` for persistence
- `sync/` for cross-store synchronization
