# AI Module Guide

## Scope

Applies to `src/bin/ai/**`. Keep this file runtime-wide; put subsystem-specific
rules in the nearest child `AGENTS.md` and long explanations in
`docs/agent-guides/*.md`.

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

1. **Driver-owned turn lifecycle.** Prompt assembly, model calls, tool loops,
   history updates, and final response handling flow through the driver. Do not
   bypass it with ad-hoc side effects.
2. **Clear provider/request boundary.** Request routing and normalization belong
   in `request/`; provider-specific wire behavior belongs in `provider/` adapter
   hooks, not scattered conditionals.
3. **Model metadata.** Treat `ApiProvider` as the request adapter axis. Platform
   naming and model metadata live in `models.json`.
4. **Tool contracts.** Tool names, schemas, display policy, and history policy
   are registry-driven. Execution logic stays out of registry metadata.
5. **Lazy capability loading.** Hidden tool/MCP catalogs, prompt hints, and
   `enable_tools` behavior must reflect the real configured registry names.
6. **Path/session authority.** Use `runtime_ctx::effective_cwd()` for user paths
   and runtime context helpers for session/temp state.
7. **History truthfulness.** Compression, pruning, and truncation must preserve
   evidence with explicit overflow/file pointers. Do not replace tool or subagent
   results with lossy summaries when the parent still needs them.
8. **Subagent ownership.** Child task results are evidence for the parent turn;
   the parent must summarize confirmed conclusions in its own final response.
9. **Scoped verification.** For runtime changes, prefer `cargo check --bin a` or
   the narrowest relevant `cargo test --bin a <test_name>`.

## Scoped guides

Read the nearest guide before changing these areas:

- `src/bin/ai/driver/`
- `src/bin/ai/tools/`
- `src/bin/ai/mcp/`
- `src/bin/ai/provider/`
- `src/bin/ai/knowledge/`
