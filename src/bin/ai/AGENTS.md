# AI Module Guide

## Scope

Applies to `src/bin/ai/**`. Keep this file runtime-wide; subsystem rules live in
the nearest child `AGENTS.md`.

## Runtime layout

- `config.rs` / `config_schema.rs`: config loading and schema
- `models.rs` + `model_names.rs` + repo-root `models/` (user override `~/.config/rust_tools/models/`): model registry
- `prompt.rs` + `prompt/` (`completion.rs`, `multiline.rs`): prompt assembly and multiline extraction
- `skills.rs` / `agents.rs` + `builtin_skills/` / `builtin_agents/` (compiled in via `include_str!`): skill/agent manifests
- `driver/`: turn orchestration, prompt/tool loop, skill runtime
- `history/`: canonical persistence, context projection/compression, task evidence
- `request/`: LLM request execution, retry, routing, normalization; `request/wire_parse.rs` holds stream primitives shared with `stream/` (avoids provider↔stream cycle)
- `provider/`: provider adapters and wire-format differences
- `tools/`: registry, service implementations, storage, display/history policy
- `mcp/`: MCP lifecycle, clients, routing snapshots, transport
- `knowledge/`: indexing, retrieval, storage, sync
- `stream/`: streaming protocol, chunk extraction, state machine, `stream/render/` terminal rendering
- `cli.rs` / `theme.rs` / `background.rs`: CLI entry, theming, background tasks
- `persona.rs` / `files.rs` / `types.rs` / `errors.rs` / `request_protocol.rs`: persona, file helpers (`extract_key_lines`), shared types, `AiError`, request dialect
- `tool_descriptions/`: per-tool JSON schemas auto-discovered by `build.rs`

## Runtime-wide invariants

1. **Verification.** Follow root ladder; keep Cargo commands scoped.
2. **Driver owns the turn.** Prompt assembly, model calls, tool loops, history mutation, and final response flow through `driver/`; no ad-hoc side effects.
3. **Provider/request boundary.** Routing/normalization in `request/`; wire differences in `provider/` adapter hooks. `ApiProvider` is the adapter axis; model/platform metadata lives in `models/` + `model_names.rs`.
4. **Tool contracts.** Names/schemas/display/history policy are registry-driven. Per-turn visibility is progressive (`core` default, `enable_tools` for lazy `builtin`); hidden MCP/catalog hints must match real registry names.
5. **Path/session authority.** `runtime_ctx::effective_cwd()` for user paths; runtime helpers for session/temp state.
6. **History is truth.** Canonical `turn_messages` vs rebuildable context projection: compression replaces only the projection, never canonical history (only explicit user lifecycle ops truncate). Preserve pruned evidence via overflow/file pointers; persist delivered subagent results and require explicit `task_integrate`.
7. **Derived-context provenance.** Runtime policy notes may map to `system`; model-authored self-notes/checkpoints stay `assistant`-derived and unverified. Never promote prior assistant wording into system fact; project via user/assistant handoff pairs only.
8. **Synthetic user messages.** Runtime-injected `role=="user"` (subagent handoff, image followup, etc.) must use `history::runtime_synthetic_user_message` and be detected via `history::is_runtime_synthetic_user_message`/`last_real_user_index` — never bare `rposition(role=="user")`. Clear the origin sidecar in `request/normalize` before provider serialization.
9. **Compression purity.** Speculative fold candidates stay pure until selected; persist evidence only for accepted candidates with deterministic asset paths; retain raw messages if archive commit fails.
10. **Model-visible wording is English.** Base prompt / gate notes / injected notes use one English system; keep Chinese only in code comments, keyword-match lists, and terminal text.

## Scoped guides

- `src/bin/ai/driver/` · `src/bin/ai/tools/` · `src/bin/ai/mcp/` · `src/bin/ai/provider/` · `src/bin/ai/knowledge/`
