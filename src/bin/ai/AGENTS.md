# AI Module Guide

## Scope

Applies to `src/bin/ai/**` and nearby entry integration in `src/bin/a.rs`.
Keep this file short. Load the referenced detailed guide only when the task
actually touches that subsystem.

## Layout

- `driver/`: run loop, turn preparation/runtime, prompt building, thinking, reflection
  - `driver/tools/`: tool execution, async tool pipe, barrier/oauth/sync_task submodules
- `request/`: LLM request execution (transport, retry, error, thinking, routing, normalization)
- `tools/`: tool registry, service implementations, storage helpers, progressive loading
  - `tools/task_tools/`: task_spawn/task_wait lifecycle + agent team orchestration
- `knowledge/`: memory/knowledge storage, recall, indexing, embedding
- `mcp/`: stdio MCP client/init/connection/routing
- `provider/`: provider adapters and request/stream differences
- `stream/`: streaming response processing (inline_recovery, runtime, extract, splitter, render)
- `history/compress/`: history compression (text_utils, tool_groups, tool_overflow, llm_prune)
- `builtin_agents/`, `builtin_skills/`: compiled-in prompt assets
- core manifests: `agents.rs`, `skills.rs`, `persona.rs`, `models.rs`, `types.rs`, `config_schema.rs`

## Multi-Agent Delegation

`build`, `plan`, and `explore` are `mode: all` (primary + subagent). `select_subagent`
(TF-IDF in `tools/task_tools.rs`) routes to them. The system prompt in
`driver/skill_runtime.rs` instructs proactive decomposition; two observer-driven nudges
in `driver/thinking/orchestrator.rs` reinforce this.

## Core rules

1. Project instruction injection is decided in `driver/skill_runtime.rs`; startup
   preload in `src/bin/a.rs` is cache warmup only.
2. `load_skill` reads skill contents only; `activate_skill` changes turn behavior
   and tool availability.
3. History compression has two orthogonal dimensions — **lossy compress** (controlled
   by `ToolHistoryPolicy.lossy_compress`; non-compressible tools like `read_file` keep
   distinct content verbatim) and **prune** (LLM-guided via `<meta:self_note>` marks,
   threshold-gated, governed by `ToolHistoryPolicy.prune`). Dedup of byte-identical
   tool results MUST run **before** offload (`prepare_tool_messages_structured`) in
   every compression path — offload rewrites results into stubs with unique temp-file
   paths, which defeats content-hash matching. Persisted history is a **sanitized**
   track, not a raw replay log: assistant `reasoning_content` is stripped on persist/load,
   and assistant messages with `tool_calls` keep the structured call metadata but must
   not retain process narration text.
4. `models.json` may declare per-model `api_key_config_key` for compatible gateways.
   **Encryption invariant:** `api_key_config_key` (like `name` and `endpoint`) may be
   encrypted (`enc:...`); `api_key_for_model` must decrypt before config lookup.
5. `models.json` separates request **adapter** from display/config **platform**.
   `adapter` drives request serialization, stream normalization, and endpoint;
   `platform` drives model handles, UI labels, and config semantics. Keep `provider`
   only as a backward-compatible alias.
6. Request-layer 429 prevention lives in `request/token_budget.rs`: every physical
   LLM HTTP send pre-reserves estimated prompt+tool-schema tokens in a per
   endpoint/model/key 60s TPM bucket before transport. Budgeting must be
   prompt-cache-aware: if the previous response reports `cached_tokens`, carry
   that through `StreamResult` / `App` and discount reusable prompt prefix
   instead of charging the whole prompt again. **Enable this only when the
   model explicitly declares `request_tpm_limit` in `models.json`; unconfigured
   models must skip waiting entirely rather than guessing a global default.**
   Do not rely on history compression or turn-loop limits as the primary
   rate-limit guard; those only reduce token volume, while the request budget
   controls send rate.
7. Responses API wire differs from chat-completions for multimodal content:
   internal `{"type":"text"}` / `{"type":"image_url"}` items must be remapped
   to Responses `input_text` / `input_image` shapes before send.

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
