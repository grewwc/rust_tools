# AI Module Guide

## Scope

Applies to `src/bin/ai/**`. Keep this file runtime-wide; subsystem rules live in
the nearest child `AGENTS.md`.

## Runtime layout

- `config*`, `models*`, and repo/user `models/`: configuration and model registry.
- `skills.rs`, `agents.rs`, and `builtin_*`: runtime and compiled manifests.
- `driver/` owns turns; `history/` owns persistence and context projection.
- Keep auxiliary prompts beside consumers as uniquely named `prompts/*.md` files.
  `format!(include_str!(...))` templates need escaped literal braces and explicit
  named arguments; bare `include_str!` files keep single braces.
- `request/`, `provider/`, and `stream/`: request policy, provider wire adapters,
  and streaming; shared wire primitives stay in `request/wire_parse.rs` to avoid
  a provider/stream cycle. Add provider wire shapes to `stream/runtime/tests/`.
- `pipeline/`, `ports/`, and `middleware/`: pipeline stages, traits, and middleware.
- `tools/`, `mcp/`, and `knowledge/`: tool stack, MCP transport, and retrieval.
- `cli.rs`, `theme.rs`, `background.rs`, and `terminal_session/`: user-facing
  process and terminal infrastructure.
- `tool_descriptions/`: build-discovered per-tool JSON metadata.

## Session storage & sessionid debugging

Sessions are the persistence unit; IDs are 1–128 ASCII letters/digits/`-`/`_`,
not necessarily UUIDs. `SessionStore` derives `<history-stem>.sessions/`, where
`<id>.sqlite` is canonical history, `<id>.assets/` holds derived artifacts, and
lock/PID files coordinate live state. Session layout is independent of cwd.
The model may inspect session data only read-only and only through the
`session` tools (`session_list` / `session_read` / `session_search`, loaded via
`enable_tools`); `/sessions` owns lifecycle. Never advertise the storage layout
to the model: it invites raw sqlite/filesystem reads that bypass the WAL-aware
tool access (see `driver/skill_runtime.rs` `session_context_prompt`).
`build_skill_turn_guard` injects that `session_context` guidance when the turn
looks session-related; other turns carry a one-line session id only.

## Runtime-wide invariants

1. **Verification.** Follow root ladder; keep Cargo commands scoped.
2. **Driver owns the turn.** Prompt assembly, model calls, tool loops, history mutation, and final response flow through `driver/`; no ad-hoc side effects.
3. **Provider/request boundary.** Routing/normalization in `request/`; wire differences in `provider/` adapter hooks. `ApiProvider` is the adapter axis (details in `provider/AGENTS.md`).
4. **Tool contracts.** Names/schemas/display/history policy are registry-driven. Per-turn visibility is progressive (`core` default, `enable_tools` for lazy `builtin`); hidden MCP/catalog hints must match real registry names.
5. **Path/session authority.** `runtime_ctx::effective_cwd()` is the working-directory authority for user paths (tools and sub-agents); runtime helpers for session/temp state.
6. **History is truth.** Canonical `turn_messages` vs rebuildable context projection: compression replaces only the projection, never canonical history (only explicit user lifecycle ops truncate). Preserve pruned evidence via overflow/file pointers; persist delivered subagent results and require explicit `task_integrate`.
7. **Derived-context provenance.** Runtime policy notes may map to `system`; model-authored self-notes/checkpoints stay `assistant`-derived and unverified. Never promote prior assistant wording into system fact; project via user/assistant handoff pairs only.
8. **Synthetic user messages.** Runtime-injected `role=="user"` (subagent handoff, image followup, etc.) must use `history::runtime_synthetic_user_message` and be detected via `history::is_runtime_synthetic_user_message`/`last_real_user_index` — never bare `rposition(role=="user")`. Clear the origin sidecar in `request/normalize` before provider serialization.
9. **Compression purity.** Speculative fold/summary candidates stay pure until selected; persist evidence only for accepted candidates with deterministic asset paths; retain raw messages if archive commit fails. Summary increments bind the removed projection bytes in `summary-sources/`; prior summaries never become fresh summarizer input. Increments are compaction products too: their inline window is capped (`MAX_INCREMENTAL_SUMMARY_INLINE_CHARS`) and older records are archived verbatim behind one back-reference before removal — never merged into a single claim set, and never dropped without an archive write. Binding is a recovery locator, not claim verification. Bump the projection fingerprint when persisted projection semantics change.
10. **Model-visible wording is English.** Base prompt / gate notes / injected notes use one English system; keep Chinese only in keyword-match lists and terminal text.
11. **Output post-processing is display-only.** `ai.output.postprocess_command`
    is a best-effort final-body filter before rendering; failures show the
    original and canonical history never changes. See `scripts/AGENTS.md`.

12. **Registry-driven tests.** Discover `models/*.json` entries via
    `model_names::all()` and assert properties, never volatile identifiers. Select
    fixtures by required metadata (thinking channel, tier, protocol).

13. **Serve-chat parity.** Serve children have no terminal: route confirmations
    and text input through `serve_confirm`, fail closed without a channel, and
    never read the local TTY. Keep `serve_confirm.rs`, `serve/chat.rs`, and
    `serve/app.html` synchronized for user-facing behavior.

## Scoped guides

- `src/bin/ai/driver/` · `src/bin/ai/tools/` · `src/bin/ai/mcp/` · `src/bin/ai/provider/` · `src/bin/ai/knowledge/`
