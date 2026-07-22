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
- `mcp/`: MCP server lifecycle, clients, routing snapshots, and OAuth/transport
  behavior
- `knowledge/`: durable knowledge indexing, retrieval, storage, and sync
- `ui*` / `render*`: terminal/TUI rendering and user-visible output

## Runtime-wide invariants

1. **🏆 No `cargo test` without user approval.** Never run `cargo test`,
   `cargo build --release`, or `cargo build` on your own initiative.
   Full-application test compilation triggers heavy dependencies (mongodb,
   rusqlite bundled, tree-sitter C parsers, ring, image, etc.) and takes
   2–8 minutes cold. Only run `cargo test` when the user explicitly asks or
   when fixing a regression with a known focused test. For all other
   verification, `cargo check --bin a` is the default. Always scope with
   `--bin`, `--lib`, or `-p`.

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
8. **History truthfulness.** Compression, pruning, and truncation must preserve
   evidence with explicit overflow/file pointers. Do not replace tool or subagent
   results with lossy summaries when the parent still needs them.
9. **Subagent ownership.** Child task results are evidence for the parent turn;
   the parent must summarize confirmed conclusions in its own final response.

## Scoped guides

Reference the nearest child `AGENTS.md` for area-specific invariants:

- `src/bin/ai/driver/`
- `src/bin/ai/tools/`
- `src/bin/ai/mcp/`
- `src/bin/ai/provider/`
- `src/bin/ai/knowledge/`
