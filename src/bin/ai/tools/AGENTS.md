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
   `read_file`/`read_file_lines`/`search_files`/`find_path`/`text_grep`/
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

## Related detailed guide

- `docs/agent-guides/ai-tools.md`
