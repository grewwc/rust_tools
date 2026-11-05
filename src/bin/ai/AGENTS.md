# AI Module Guide

## Scope

Runtime-wide contracts for the `a` binary, applying to `src/bin/ai/**`.

## Runtime layout

- `config*`, `models*`, and repo/user `models/`: configuration and model registry.
- `skills.rs`, `agents.rs`, and `builtin_*`: runtime and compiled manifests.
  `builtin_agents/*.agent` and `builtin_skills/*.skill` are compiled via
  `include_str!`; user `.skill` files load at runtime.
- `driver/` owns turns; `history/` owns persistence and context projection.
- Keep auxiliary prompts beside consumers as uniquely named `prompts/*.md` files.
  `format!(include_str!(...))` templates need escaped literal braces and explicit
  named arguments; bare `include_str!` files keep single braces.
- `request/`, `provider/`, and `stream/`: request policy, provider wire adapters,
  and streaming; shared wire primitives stay in `request/wire_parse.rs` to avoid
  a provider/stream cycle. Add provider wire shapes to `stream/runtime/tests/`.
- `pipeline/`, `ports/`, and `middleware/`: pipeline stages, traits, and middleware.
- `tools/`, `mcp/`, and `knowledge/`: tool stack, MCP transport, and retrieval.
- `serve/`: the HTTP/SSE server and the `--serve-*` process supervisor.
- `cli.rs`, `theme.rs`, `background.rs`, and `terminal_session/`: user-facing
  process and terminal infrastructure.
- `tool_descriptions/`: build-discovered per-tool JSON metadata.

## Session inspection & sessionid debugging

Sessions are the persistence unit; IDs are opaque names, not necessarily UUIDs,
and the session layout is independent of cwd. The model may inspect session data
only read-only and only through the `session` tools
(`session_list` / `session_read` / `session_search`, loaded via `enable_tools`);
`/sessions` owns lifecycle. The on-disk session storage (sqlite and asset
files) is not a sanctioned read path — `session_search` surfaces asset content
at runtime, and `history.sessions/` at the repo root is gitignored test data,
not the store. Capability-first: a question that names
a session id must be answered with the `session` tools — never by probing the
filesystem or sqlite. A top-level turn naming a session id auto-loads the
`session` group; session guidance follows the driver's prompt-residency contract.

## Runtime-wide invariants

1. **Configuration & tests (`a` only).** Add configuration keys in `AiConfig`
   (`src/bin/ai/config_schema.rs`). Serial tests use `test_support::ENV_LOCK`
   (`src/bin/ai/mod.rs`); these facilities do not govern standalone crates.
2. **Driver owns the turn.** Prompt assembly, model calls, tool loops, history mutation, and final response flow through `driver/`; no ad-hoc side effects.
3. **Provider/request boundary.** Routing/normalization in `request/`; wire differences in `provider/` adapter hooks. `ApiProvider` is the adapter axis (details in `provider/AGENTS.md`).
4. **Tool contracts.** Names/schemas/display/history policy are registry-driven; visibility and naming rules live in `tools/AGENTS.md` and `mcp/AGENTS.md`.
5. **Path/session authority.** `runtime_ctx::effective_cwd()` is the working-directory authority for user paths (tools and sub-agents); runtime helpers for session/temp state.
6. **History is truth.** Canonical `turn_messages` are distinct from the
   rebuildable context projection; only explicit user lifecycle operations may
   truncate canonical history. Compression, pruning, and offloading change the
   projection only. Archive truth-bearing reads, commands, and task evidence
   durably before replacing them with self-describing overflow/file pointers;
   retain raw messages when the archive commit fails. Persist delivered subagent
   results before IPC cleanup and retain child history/artifacts on merge failure
   or timeout. Task evidence becomes prunable only when the durable ledger proves
   every named task integrated through explicit `task_integrate`; reading results
   does not integrate them. Projection/recovery mechanisms live in the Driver
   guide; ledger and tool policies live in the Tools guide.
7. **Derived-context provenance.** Runtime policy notes may map to `system`; model-authored self-notes/checkpoints stay `assistant`-derived and unverified. Never promote prior assistant wording into system fact; project via user/assistant handoff pairs only.
8. **Synthetic user messages.** Runtime-injected `role=="user"` (subagent handoff, image followup, etc.) must use `history::runtime_synthetic_user_message` and be detected via `history::is_runtime_synthetic_user_message`/`last_real_user_index` — never bare `rposition(role=="user")`. Clear the origin sidecar in `request/normalize` before provider serialization. `reasoning_content` on user/internal-note messages is runtime metadata (origin, request reminder, fold provenance): check `history::has_runtime_message_metadata` before treating it as model reasoning.
9. **Compression purity.** Speculative fold/summary candidates stay pure until
   selected; persist evidence only for accepted candidates. Summary increments
   bind removed projection bytes in `summary-sources/`; prior summaries never
   become fresh summarizer input, and
   older records archive verbatim behind one back-reference before removal —
   never merged, never dropped without an archive write. Binding is a recovery
   locator, not claim verification. Bump the projection fingerprint when
   persisted projection semantics change.
10. **Model-visible wording is English.** Base prompt / gate notes / injected notes use one English system; keep Chinese only in keyword-match lists and terminal text.
11. **Output post-processing is display-only.** `ai.output.postprocess_command`
    is a best-effort final-body filter before rendering; failures show the
    original. History retention follows **History is truth**; script behavior
    lives in `scripts/AGENTS.md`.

12. **Registry-driven tests.** Discover `models/*.json` entries via
    `model_names::all()` and assert properties, never volatile identifiers. Select
    fixtures by required metadata (thinking channel, tier, protocol).

13. **Serve-chat parity.** Serve children have no terminal: route confirmations
    and text input through `serve_confirm`, fail closed without a channel, and
    never read the local TTY. Keep `serve_confirm.rs`, `serve/chat.rs`, and
    `serve/app.html` synchronized for user-facing behavior.

## Scoped guides

- [Driver](driver/AGENTS.md) · [Tools](tools/AGENTS.md) · [MCP](mcp/AGENTS.md) ·
  [Provider](provider/AGENTS.md) · [Knowledge](knowledge/AGENTS.md) ·
  [Serve](serve/AGENTS.md)
