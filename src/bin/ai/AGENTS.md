# AI Module Guide

## Scope

Applies to `src/bin/ai/**`. Keep this file runtime-wide; put subsystem-specific
rules in the nearest child `AGENTS.md`.

## Runtime layout

- `config*`: config loading, schema, model registry access
- `driver/`: turn orchestration, prompt/tool loop, skill runtime, history glue
- `request/`: LLM request execution, retry, error handling, routing,
  normalization, and thinking/reasoning support
- `provider/`: provider adapters and wire-format differences
- `tools/`: tool registry, service implementations, storage, and display/history
  policy metadata
- `mcp/`: MCP server lifecycle, clients, routing snapshots, and transport behavior
- `knowledge/`: durable knowledge indexing, retrieval, storage, and sync
- `stream/`: streaming protocol, chunk extraction, state machine, and
  terminal/TUI rendering (under `stream/render/`)

## Runtime-wide invariants

1. **No `cargo test` without user approval** - see root `AGENTS.md` Build/Test.
   Full-app test compilation triggers heavy deps (mongodb, rusqlite, image) and
   takes 2–8 min cold. Default to `cargo check --bin a`, always scoped.
2. **Driver-owned turn lifecycle.** Prompt assembly, model calls, tool loops,
   history updates, and final response handling flow through the driver. Do not
   bypass it with ad-hoc side effects.
3. **Clear provider/request boundary.** Request routing and normalization belong
   in `request/`; provider-specific wire behavior belongs in `provider/` adapter
   hooks, not scattered conditionals.
4. **Model metadata.** Treat `ApiProvider` as the request adapter axis. Platform
   naming and model metadata live in `models.json`.
5. **Tool contracts.** Tool names, schemas, display policy, and history policy
   are registry-driven. Execution logic stays out of registry metadata.
6. **Lazy capability loading.** Hidden tool/MCP catalogs, prompt hints, and
   `enable_tools` behavior must reflect the real configured registry names.
7. **Path/session authority.** Use `runtime_ctx::effective_cwd()` for user paths
   and runtime context helpers for session/temp state.
8. **History truthfulness.** Canonical session messages and rebuildable model
   context are separate layers. Compression may replace only the context snapshot,
   never canonical messages; only explicit user lifecycle operations may truncate
   canonical history. Preserve pruned evidence with explicit overflow/file pointers.
9. **Subagent ownership.** Child task results are evidence for the parent turn;
   the parent must summarize confirmed conclusions in its own final response.
10. **Derived-context provenance.** Runtime-owned policy/control notes may map
    to `system`; model-authored self-notes, checkpoints, and automatic summaries
    remain assistant-derived and marked unverified. Project assistant-derived
    context through request-only user/assistant handoff pairs (conventional role
    order). Never promote prior assistant wording into a system-level fact or
    verified conclusion.

## Scoped guides

Reference the nearest child `AGENTS.md` for area-specific invariants:

- `src/bin/ai/driver/`
- `src/bin/ai/tools/`
- `src/bin/ai/mcp/`
- `src/bin/ai/provider/`
- `src/bin/ai/knowledge/`
