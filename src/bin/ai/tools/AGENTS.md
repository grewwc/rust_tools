# Tools Guide

## Scope

Applies to `src/bin/ai/tools/**`.

Read `docs/agent-guides/ai-tools.md` before changing tool registration,
execution policy, sandboxing, path resolution, or progressive loading.

## Key invariants

1. **Layer separation**: schema/metadata in `registry/`, execution in
   `service/`, shared helpers/state in `storage/`.
2. **Naming**: tool names use verb-first snake_case.
3. **Paths**: relative file paths must resolve against
   `runtime_ctx::effective_cwd()`.
4. **Progressive loading**: the default per-turn tool set is `core`
   (`DEFAULT_TURN_TOOL_GROUPS` in `driver/skill_runtime.rs`). Tools tagged
   only `builtin` (not `core`) are lazy — surfaced via `enable_tools` and
   aged out when idle. The `os_tools` process/IPC/shm/env suite is
   `builtin`+`executor` only (NOT `core`), so it is **deferred even for
   executor agents** — filtered via `groups_defer_eager_load`. Explicit
   `tools:` name lists are never filtered (naming pins eager).
5. **Validation**: prefer structural parsing over brittle string matching.
6. **ToolDisplayRegistration**: display behavior declared per-tool via
   optional `ToolDisplayRegistration` inventory submission, **not** by adding
   fields to `ToolSpec`. Defaults to all-`false`. Query via
   `tool_display_config(name)`.
7. **ToolHistoryPolicy**: retention declared per-tool via
   `ToolHistoryPolicyRegistration` (same pattern as rule 6). Two
   **orthogonal** dimensions: `lossy_compress` (`Allow`/`Never`) gates
   trimming/folding; `prune` (`Allow`/`Never`) gates LLM-guided pruning.
   Both default `Allow`. `plan`=`Never`/`Never`;
   `read_file`/`code_search`=`Never`/`Allow`. Query via
   `tool_history_policy(name)`. Do not reintroduce name-keyed `if`/`match`
   chains in `history/compress/`.
8. **Temp files**: use `write_file(temp=true)`, which writes under
   `runtime_ctx::temp_dir()` (per-session, outside project dir) and
   registers in a JSON registry. `file_path` must be relative; absolute
   paths rejected. `delete_path` only deletes registered files.
9. **Process groups**: `execute_command` runs via `setsid` in its own process
   group. Backgrounded pgids recorded in-memory keyed by `session_id`
   (`storage::process_registry`); `kill_session` `killpg`s at teardown. Do
   NOT persist pgids to disk (recycled across restarts) or key off
   `runtime_ctx::temp_dir()`.
10. **Self-describing truncation**: any truncating tool must let the model
    distinguish "done" from "incomplete". Canonical: `execute_command`
    (`format_command_output`). (a) Empty success returns explicit sentinel,
    never `""`; (b) failure prefixes `Exit code: N`; (c) truncation appends
    shown-vs-total counts and a narrow/page hint. Do not emit information-free
    truncation markers. Git/cargo run through `execute_command`.
11. **Patch hunk matching**: `apply_patch` anchors on remove lines, uses
    context lines for positioning. Strict match first; stale context tolerated
    only when removes uniquely identify the target.
12. **read_file dual cap**: paginates by lines (`offset`/`limit`) but also
    applies a character cap (`MAX_READ_FILE_RESULT_CHARS`, 64K). Truncation
    notice computes continue-`offset` from **actually rendered lines**, never
    from requested `limit`. When size-triggered, says so explicitly.
    Do not revert to `render_line_excerpt(..., None)`.
13. **code_search is the single navigation entry point**: wraps six LSP ops
    (`go_to_definition`, `find_references`, `hover`, `document_symbol`,
    `workspace_symbol`, `diagnostics`) plus `find_file`/`text_search`/
    `structural`. Legacy `lsp` maps to `code_search` via
    `canonical_tool_name`.
14. **Subagent spawn depth**: `task_spawn`/`task` reject when
    `child_depth > MAX_SUBAGENT_SPAWN_DEPTH` (2). Prevents unbounded
    recursive fanout.

## Related detailed guide

- `docs/agent-guides/ai-tools.md`
