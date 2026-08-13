# AI Module Guide

## Scope

Applies to `src/bin/ai/**`. Keep this file runtime-wide; put subsystem-specific
rules in the nearest child `AGENTS.md`.

## Runtime layout

- `config.rs` / `config_schema.rs`: config loading and schema
- `models.rs` / `models.json`: model registry access, platform naming, metadata
- `prompt/`: prompt assembly and multiline extraction
- `skills.rs` / `agents.rs`: skill + agent manifests; `builtin_agents/` holds `.agent`
  files and `builtin_skills/` holds builtin `.skill` files, both compiled in via `include_str!`
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
- `persona.rs`: persona switching (per-session identity overlay, `PersonaProfile`/`PersonaStore`)
- `files.rs` / `types.rs`: file/text parse helpers and shared types (`FileParseResult`, ...)
- `model_names.rs`: `ModelDef` registry lookup and platform/model handle helpers
- `request_protocol.rs`: request wire-protocol dialects (chat-completions vs responses)
- `errors.rs`: structured `AiError` enum (alternative to pervasive `Result<T, String>`)
- `tool_descriptions/`: tool description JSONs, auto-discovered by `build.rs`

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
11. **Compression side effects.** Speculative context-fold candidates must remain
    pure until selected. Persist lossless evidence only for accepted candidates,
    use deterministic/idempotent asset paths, and retain raw messages when an
    archive commit fails.
12. **Synthetic user messages are not turn boundaries.** Any runtime-injected
    `role == "user"` message (subagent evidence handoff, auto image followup,
    etc.) must be built with `history::runtime_synthetic_user_message`; never
    infer runtime origin from user-controlled content. Request normalization must
    clear the internal origin sidecar before provider serialization. All
    current-turn boundary scans (scoped instruction targets, current-turn tool
    protection, dedupe bounds, compression/retention) must use
    `history::last_real_user_index` / `is_runtime_synthetic_user_message`,
    never a bare `rposition(role == "user")`.
13. **Model-visible runtime notes are English.** Base prompt, gate notes, and
    injected internal notes share one wording system (e.g. "unverified").
    New model-visible notes/warnings must be written in English; keep Chinese
    only in code comments, model-output keyword matching lists, and
    user-facing terminal/CLI text.

## Scoped guides

Reference the nearest child `AGENTS.md` for area-specific invariants:

- `src/bin/ai/driver/`
- `src/bin/ai/tools/`
- `src/bin/ai/mcp/`
- `src/bin/ai/provider/`
- `src/bin/ai/knowledge/`
