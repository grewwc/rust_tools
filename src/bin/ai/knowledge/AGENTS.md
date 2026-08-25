# Knowledge Guide

## Scope

Applies to `src/bin/ai/knowledge/**` (`indexing/`, `retrieval/`, `storage/`, `sync/`).

## Key invariants

1. **Public vs internal tools.** User-facing persistence uses `knowledge_*`
   tools; `memory_*` tools are agent-internal.
2. **Consolidation contract.** `knowledge_consolidate` is a two-phase flow:
   `read_all` then `execute`. Merges are lossless: `execute` `save_entries[].source_ids`
   must list every source entry being consolidated, and the tool appends each source's
   full original note to the merged entry (see `execute_consolidation` in
   `tools/knowledge_tools.rs`) while auto-deleting those sources (their content is fully
   preserved in the merged entry). Content is discarded only for entries removed via
   `delete_ids` without being listed in any `save_entries[].source_ids` — consolidation
   never silently compresses or drops original content.
3. **Graceful retrieval fallback.** Embedding is optional; explicit knowledge
   search must degrade to lexical/BM25 retrieval when unavailable or failing.
4. **Separated responsibilities.** Keep retrieval, indexing, storage, and sync
   separated unless a change genuinely crosses those boundaries.
5. **No automatic recall.** Knowledge is read only through explicit
   `knowledge_*` / `memory_*` tool calls - never scan or inject the store
   automatically while preparing a turn. Notebook is an independent tool-backed
   context source, not coupled to knowledge retrieval.
