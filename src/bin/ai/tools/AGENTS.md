# Tools Guide

## Scope

Applies to `src/bin/ai/tools/**`.

Read `docs/agent-guides/ai-tools.md` before changing tool registration,
execution policy, sandboxing, path resolution, or progressive loading.

## Key invariants

1. Keep schema/metadata in `registry/`, execution in `service/`, and shared
   helpers/state in `storage/`.
2. Tool names use verb-first snake_case.
3. Relative file paths must resolve against `runtime_ctx::effective_cwd()`.
4. Baseline/progressive-loading behavior should stay explicit; do not silently
   hide required entry-point tools. The default per-turn tool set is the `core`
   group (`DEFAULT_TURN_TOOL_GROUPS` in `driver/skill_runtime.rs`); tools tagged
   only `builtin` (not `core`) are lazy — surfaced by name via `enable_tools` and
   aged out when idle. The `os_tools` process/IPC/shm suite is intentionally
   `builtin`+`executor` only (NOT `core`): it stays available to executor agents
   and on-demand activation, but is kept out of the default request payload to
   cut per-turn input tokens. Do not re-add `core` to bulk low-frequency tools.
5. Prefer structural parsing over brittle string matching in validation paths.
6. Tool display behavior (whether to echo call args and/or result to the
   terminal) is declared per-tool via an optional `ToolDisplayRegistration`
   inventory submission in the tool's own file, **not** by adding fields to
   `ToolSpec`. `ToolDisplayConfig { print_args, print_result }` defaults to
   all-`false`; only tools with high user-facing visibility (e.g. `plan`)
   opt in. This keeps the 77+ existing `ToolSpec` initializers untouched and
   mirrors the `ToolStreamingRegistration` compatibility pattern. Query via
   `tool_display_config(name)`; never hardcode tool names in `print_run_status`
   or `driver/print.rs`.
7. Tool history-retention behavior is declared per-tool via an optional
   `ToolHistoryPolicyRegistration` inventory submission in the tool's own file
   (same bypass pattern as rule 6), **not** by hardcoding name lists in the
   compression code. `ToolHistoryPolicy` has two **orthogonal** dimensions:
   `lossy_compress` (`Allow`/`Never`) gates line-trimming/folding/summarizing;
   `prune` (`Allow`/`Never`) gates LLM-guided pruning of superseded results.
   Both default to `Allow` (unregistered tools). `plan` is `Never`/`Never`;
   `read_file`/`search_files`/`find_path`/`text_grep`/
   `code_search` are `Never`/`Allow` (precision results are never lossy-compressed
   but may still be pruned once superseded). Query via `tool_history_policy(name)`;
   the compression side wraps it as `is_non_compressible_tool` (lossy dimension)
   and `is_prune_protected_tool` (prune dimension). Do not reintroduce name-keyed
   `if`/`match` chains in `history/compress/`.
8. Temporary / scratch files should use `write_file(temp=true)`, which writes
   under `runtime_ctx::temp_dir()` (per-session, co-located with tool-overflow,
   outside the project dir) AND registers the file in a persistent JSON registry
   (`storage::temp_registry`). `file_path` must be a relative filename only;
   absolute paths and directory components are stripped/rejected. `delete_path`
   ONLY deletes files in this registry — unregistered files are always refused.
   Never rely on `execute_command` for deletion — `rm` is blocked by the sandbox.
9. `execute_command` runs each command via `setsid` in its own process group.
   If the command backgrounds a long-lived process (e.g. `python app.py &`),
   the foreground call returns and the surviving process-group pgid is recorded
   in an **in-memory, process-global** registry keyed by `session_id`
   (`storage::process_registry`) — `kill_session` `killpg`s them (SIGTERM then
   SIGKILL) at session teardown via `cleanup_one_shot` in `driver/mod.rs`. Do
   NOT persist pgids to disk (they get recycled across restarts and would
   mis-kill), and do NOT key this registry off `runtime_ctx::temp_dir()`
   (register-time is inside a turn, kill-time is outside — the paths differ).
   The lib-crate spawner (`cmd::run`) reports pgids outward via the
   `on_background_group` callback of
   `run_cmd_output_streaming_with_timeout_tracked`; it cannot reference the
   binary-side registry directly.
10. Any tool that truncates output must stay self-describing so the model can
    tell "done" from "incomplete" without re-running. Canonical impl:
    `execute_command` (`service/command.rs::format_command_output`). Invariants:
    (a) success with empty output returns an explicit
    "succeeded with exit code 0 and produced no output" sentinel, never `""`;
    (b) failure prefixes `Exit code: N`; (c) truncation (`truncate_chars`, cap
    `MAX_COMMAND_OUTPUT_CHARS`) appends shown-vs-total char/line counts plus a
    warning that unseen matches may be in the cut-off tail and a hint to
    narrow/page instead of retrying variants. This exists because bare
    `... (truncated)` + silent-empty output previously drove strong models into
    long near-identical retry loops (they couldn't tell if their target was
    cut off or truly absent — observed for both `execute_command` grep retries
    and the now-removed `git_diff` tool). Do not add a tool that emits an
    information-free truncation marker; if a tool caps output, its notice must
    carry shown-vs-total counts and a concrete narrow/page path. Git and cargo
    inspection are intentionally NOT separate tools — the model runs `git
    status`/`git diff`/`cargo check`/`cargo test` through `execute_command`,
    which already provides the self-describing truncation contract.
11. `apply_patch` hunk matching uses remove lines as the hard anchor and context
    lines as positioning evidence. Keep strict matching first, then tolerate
    stale context only when the remove lines uniquely identify the target or the
    remaining context gives a single best candidate. Ambiguous remove anchors
    must fail instead of silently editing the first match.
12. `read_file` (`service/file.rs`) paginates by **lines**
    (`offset`/`limit`), but line paging cannot bound **characters**: a minified
    or single-huge-line file produces an MB-scale result even at 1 line, which
    would blow the context window once it enters `messages` raw. The reader
    therefore also applies a character cap (`MAX_READ_FILE_RESULT_CHARS`, 64K) via
    `render_line_excerpt(..., Some(cap))`. The truncation notice MUST compute the
    continue-`offset` from the **actually rendered line count**
    (`start + excerpt.shown_lines`), never from the requested `limit` — using
    `limit` would point the model past unshown lines and silently drop the gap.
    When the cut is size-triggered, the notice says so explicitly (`output capped
    at N chars`) so the model knows to page on rather than assume EOF. Do not
    revert the reader to `render_line_excerpt(..., None)`. The former separate
    `read_file_lines` tool has been merged into `read_file` (offset/limit cover
    both overview and precision reads); `registry/common.rs::canonical_tool_name`
    maps the legacy `read_file_lines` name to `read_file` for old-session replay.
13. `code_search` is the single code-navigation entry point; there is no separate
    `lsp` tool. `code_search`'s six LSP operations (`go_to_definition`,
    `find_references`, `hover`, `document_symbol`, `workspace_symbol`,
    `diagnostics`) delegate to `execute_lsp` (a `pub(crate)` fn now living in
    `code_search.rs`), and it adds `find_file`/`text_search`/`structural`,
    no-match fallbacks, guidance, and path guards on top — a strict superset. The
    tree-sitter-backed LSP implementation and its tests were moved out of the
    deleted `lsp_tools.rs` into `code_search.rs`. `canonical_tool_name` maps the
    legacy `lsp` name to `code_search` for old-session replay (same
    `operation`/`file_path` arg shape).
14. **Subagent spawn depth guard**: `task_spawn` and `task` (sync) both call
    `spawn_subagent_kernel_task`, which reads `runtime_ctx::current_subagent_depth()`
    and rejects spawning when `child_depth > MAX_SUBAGENT_SPAWN_DEPTH` (2). The
    depth is propagated via `OsTaskGoal.spawn_depth` and scoped as
    `runtime_ctx::SUBAGENT_DEPTH` in both dispatch paths
    (`background_dispatch.rs`, `tools/sync_task.rs`). This prevents unbounded
    recursive fanout when `mode: all` agents (e.g. `build`) are spawned as
    subagents and themselves hold `task_spawn`.

## Related detailed guide

- `docs/agent-guides/ai-tools.md`
