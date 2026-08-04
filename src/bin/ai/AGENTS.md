# AI Module Guide

## Scope

Applies to `src/bin/ai/**`. Keep this file runtime-wide; put subsystem-specific
rules in the nearest child `AGENTS.md`.

## Runtime layout

- `config*` / `config/`: config loading and schema
- `models.rs` / `models.json`: model registry access, platform naming, metadata
- `prompt/`: prompt assembly and multiline extraction
- `skills.rs` / `agents.rs`: skill + agent manifests; `builtin_agents/` holds
  `.agent` files compiled in via `include_str!`
- `driver/`: turn orchestration, prompt/tool loop, and skill runtime
- `history/`: canonical persistence, context projection/compression, and task evidence
- `request/`: LLM request execution, retry, error handling, routing,
  normalization, and thinking/reasoning support
- `provider/`: provider adapters and wire-format differences
- `tools/`: tool registry, service implementations, storage, and display/history
  policy metadata
- `mcp/`: MCP server lifecycle, clients, routing snapshots, and transport behavior
- `knowledge/`: durable knowledge indexing, retrieval, storage, and sync
- `stream/`: streaming protocol, chunk extraction, state machine, and
  terminal/TUI rendering (under `stream/render/`)
- `cli.rs` / `theme.rs` / `background.rs`: CLI entry, theming, background tasks

## Runtime-wide invariants

1. **Verification.** Follow the root verification ladder; keep Cargo commands
   scoped, never bare workspace-wide checks.
2. **Driver-owned turn lifecycle.** Prompt assembly, model calls, tool loops,
   history updates, and final response all flow through the driver; no ad-hoc
   side effects.
3. **Provider/request boundary.** Request routing/normalization in `request/`;
   provider wire behavior in `provider/` adapter hooks, not scattered
   conditionals.
4. **Model metadata.** `ApiProvider` is the request adapter axis; platform
   naming and model metadata live in `models.json`.
5. **Tool contracts.** Tool names, schemas, display/history policy are
   registry-driven; execution logic stays out of registry metadata.
6. **Lazy capability loading.** Hidden tool/MCP catalogs, prompt hints, and
   `enable_tools` must reflect the real configured registry names.
7. **Path/session authority.** Use `runtime_ctx::effective_cwd()` for user paths
   and runtime context helpers for session/temp state.
8. **History truthfulness.** Canonical messages and rebuildable model context
   are separate layers: compression may replace only the context snapshot, never
   canonical messages; only explicit user lifecycle ops truncate history.
   Preserve pruned evidence via explicit overflow/file pointers.
9. **Subagent evidence lifecycle.** Persist delivered child results, restore
   unintegrated evidence after context rebuild, and require explicit parent
   integration before normal completion.
10. **Derived-context provenance.** Runtime-owned policy/control notes may map
    to `system`; model-authored self-notes, checkpoints, and automatic summaries
    stay assistant-derived and unverified. Never promote prior assistant wording
    into a system-level fact; project assistant-derived context via
    user/assistant handoff pairs only.

## Scoped guides

Reference the nearest child `AGENTS.md` for area-specific invariants:

- `src/bin/ai/driver/`
- `src/bin/ai/tools/`
- `src/bin/ai/mcp/`
- `src/bin/ai/provider/`
- `src/bin/ai/knowledge/`
