# AI Module Guide

## Scope

Applies to `src/bin/ai/**` and nearby entry integration in `src/bin/a.rs`.
Keep this file short. Load the referenced detailed guide only when the task
actually touches that subsystem.

## Layout

- `driver/`: run loop, turn preparation/runtime, prompt building, thinking, reflection
  - `driver/tools/`: tool execution, async tool pipe, tool cache, barrier/oauth/sync_task submodules
- `request/`: LLM request execution (error/retry, thinking mode, skill routing, message normalization, stream types)
- `tools/`: tool registry, service implementations, storage helpers, progressive loading
  - `tools/task_tools/`: task_spawn/task_wait lifecycle + agent team orchestration + tests
- `knowledge/`: memory/knowledge storage, recall, indexing, embedding
- `mcp/`: stdio MCP client/init/connection/routing
- `provider/`: provider adapters and request/stream differences
- `stream/`: streaming response processing (inline_recovery, runtime, extract, splitter, render)
- `history/compress/`: history compression (text_utils, tool_groups, tool_overflow, llm_prune, tests)
- `builtin_agents/`, `builtin_skills/`: compiled-in prompt assets
- core manifests: `agents.rs`, `skills.rs`, `persona.rs`, `models.rs`, `types.rs`, `config_schema.rs`

## Core rules

1. Project instruction injection is decided in `driver/skill_runtime.rs`; startup
   preload in `src/bin/a.rs` is cache warmup only.
2. `load_skill` reads skill contents only; `activate_skill` changes turn behavior
   and tool availability.
3. LLM-guided pruning (`history/compress/llm_prune.rs`): the model marks outdated
   tool results via `<meta:self_note>prune:id1,id2</meta:self_note>`; after
   `PRUNE_THRESHOLD` (3) consecutive marks the content is replaced with a
   placeholder. It must reuse the existing tool-result protection policy:
   `is_non_compressible_tool` results and the most recent `KEEP_RECENT_TOOL_MESSAGES`
   tool messages are ignored even if marked. Injected in `prepare_turn`; marks are
   parsed from both tool-call and final-response model turns. Does not delete messages
   or alter existing compression logic.
4. Non-compressible tool results (`read_file` etc.) keep every **distinct** content
   version verbatim, but `dedup_repeated_tool_results` collapses **byte-identical**
   repeats (same name+args+content hash) into a re-read-suppressing stub. This breaks
   the "repeated full re-read" amnesia loop losslessly. Invariant: dedup MUST run
   **before** `prepare_tool_messages_structured` (offload) in every compression path —
   offload rewrites identical results into stubs with unique temp-file paths, which
   would defeat content-hash matching.

## On-Demand Guides

- driver / prompt assembly / general-knowledge mode:
  `docs/agent-guides/ai-driver.md`
- tool registration / execution / storage:
  `docs/agent-guides/ai-tools.md`
- MCP init / routing / OAuth / timeouts:
  `docs/agent-guides/ai-mcp.md`
- provider adapters and wire differences:
  `docs/agent-guides/ai-provider.md`

## Scoped Subdirectories

If the task is already under one of these paths, treat the closer `AGENTS.md`
as the next authority:

- `src/bin/ai/driver/`
- `src/bin/ai/tools/`
- `src/bin/ai/mcp/`
- `src/bin/ai/provider/`
- `src/bin/ai/knowledge/`
