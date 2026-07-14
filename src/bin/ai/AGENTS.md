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
   placeholder. Prune protection is driven by each tool's registered
   `ToolHistoryPolicy.prune` (see rule 6 in `tools/AGENTS.md`), **not** by
   `is_non_compressible_tool`: only tools declaring `prune: Never` (e.g. `plan`)
   plus the most recent `KEEP_RECENT_TOOL_MESSAGES` tool messages are ignored
   even if marked. Crucially, `read_file`/search results are lossy-incompressible
   yet **prunable** once superseded — the two dimensions are orthogonal. Injected
   in `prepare_turn`; marks are parsed from both tool-call and final-response model
   turns. Does not delete messages or alter existing compression logic.
4. Non-compressible tool results (`read_file` etc.) keep every **distinct** content
   version verbatim, but `dedup_repeated_tool_results` collapses **byte-identical**
   repeats (same name+args+content hash) into a re-read-suppressing stub. This breaks
   the "repeated full re-read" amnesia loop losslessly. "Non-compressible" is decided
   by each tool's registered `ToolHistoryPolicy.lossy_compress == Never`, queried via
   `is_non_compressible_tool` (a thin wrapper over `tool_history_policy`), not a
   hardcoded name list. Invariant: dedup MUST run **before**
   `prepare_tool_messages_structured` (offload) in every compression path —
   offload rewrites identical results into stubs with unique temp-file paths, which
   would defeat content-hash matching. **Unified recent-window invariant:** every
   lossy path must protect the same tail — the most recent `KEEP_RECENT_TOOL_MESSAGES`
   *tool messages*. `dedup`, offload, and `apply_pruning` already key off that count;
   `fold_early_tool_groups` counts by *tool group* instead, so it additionally clamps
   its fold boundary via `recent_tool_message_protection_anchor` so folding can never
   cross the shared window even at `keep_recent_groups=0`. Do NOT re-introduce a
   group-only fold window: a unit mismatch silently weakens a recent tool result that
   the other paths swear to keep, and the model re-runs the tool (the `read_file`/`execute_command` repeat-call loop). Ingress protection
   is the first line (`prepare_recent_tool_result` keeps the current turn's result raw
   instead of stubbing it); these compression windows are the second. **Request-time
   final line:** `sanitize_tool_call_for_request` (in `request/normalize.rs`) must NEVER
   drop a tool_call just because its `arguments` are not valid JSON — the tool already
   ran and produced a real result, so dropping the call breaks assistant/tool pairing
   and degrades the real `tool` result into a 240-char preview note. Repair malformed
   args into a valid object (`{"_malformed_arguments": "<raw>"}`) so pairing and the
   real result survive.
5. `models.json` may declare per-model `api_key_config_key` for compatible/openai-style
   gateways with provider-specific config names. Startup validation in `config.rs`
   must honor that field for the default model; do not assume only the built-in
   global keys (`compatible.api_key`, `openai.api_key`, etc.) can unlock startup.
6. `models.json` separates request `adapter` from display/config `platform`.
   `adapter` drives request serialization, stream normalization, default endpoint,
   and API-key fallback behavior; `platform` drives model handle suffixes, UI/log
   labels, and platform-specific config semantics. Keep `provider` only as a
   backward-compatible alias when reading old config, not as the canonical field.
7. Multiline inline viewport chrome must avoid stable decorative rule rows
   entirely. During resize/reflow/re-anchor, full-width divider lines are prone
   to being pushed into terminal scrollback and can accumulate as duplicate
   horizontal artifacts. Keep only essential model/help chrome, and keep it
   below the textarea top row instead of drawing a persistent divider.
8. Terminal markdown/table width math must follow real macOS terminal cell widths,
   not raw Unicode ambiguity defaults: emoji-style symbol blocks and score-delta
   markers such as `△▽▲▼` can render double-width and must be counted that way
   to avoid wrapped table borders and right-edge residue.
9. Session-title generation is a post-turn background-quality task, not a
   foreground UX path. Its model request/body timeouts may be longer than the
   interactive defaults (currently 90s / 45s) so slow control-model responses
   still have a chance to land without delaying the visible answer stream.
10. For DeepSeek/OpenCode continuation requests, `reasoning_content` echo on
    assistant tool-call messages is determined by the model's wire dialect, not
    by the current turn's `enable_thinking` decision. Mid-turn compression may
    clear older reasoning text to `None`; request preparation must still restore
    empty-string placeholders before sending or the provider can reject the
    continued conversation with a stable 400.
11. The streaming thinking-fold (`stream/runtime.rs`) must anchor its
    `╭─ thinking` header: print it exactly once on activation (`header_drawn`),
    then never retrace it. Per-chunk redraw (`\x1b[{window_rows}A\r\x1b[0J`) may
    only erase the bounded body region (fold summary + ≤`max_visible_lines`
    rows); `window_rows` counts body physical rows only, excluding the header.
    When the terminal is resized, previously rendered body lines can be reflowed
    by the terminal into more physical rows than the cached `window_rows`, so
    erase distance must be recomputed from the last rendered body under the
    **current** terminal width rather than trusting the cached row count alone.
    Rationale: a redraw window that spans the header can scroll into scrollback
    where cursor-up can't reach, leaving orphan headers that stack across chunks.
    Keeping the header outside the erased region makes a second header
    structurally impossible even if body erase ever slips.
12. Inline tool-call recovery in `stream/` must also recognize bare registered
    XML tool tags whose tag name is the tool name itself, such as
    `<execute_command>...</execute_command>`. This support is two-part: the
    streaming suppressors must strip the raw tag from visible output as soon as
    it appears, and finalize fallback must recover it into a structured tool
    call if the assistant text is pure markup. For raw-text bodies, only coerce
    them into arguments when the target tool has exactly one required `string`
    parameter (for example `execute_command.command`); otherwise leave the
    markup visible rather than guessing a schema.

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
