7. Temporary / scratch files the agent creates during a task should use
   `write_file(temp=true)`, which writes under `runtime_ctx::temp_dir()`
   (per-session: `<sessions_root>/<session>.assets/tmp/`, co-located with
   tool-overflow, outside the project dir; falls back to
   `<std::env::temp_dir()>/.agent_tmp/<session>/` when DRIVER_CTX is unavailable,
   never inside the project dir)
   AND registers the file in a persistent JSON registry
   (`storage::temp_registry`). When `temp=true`, `file_path` MUST be a
   relative filename only (e.g. `script.py`); absolute paths and directory
   components are stripped/rejected so the file always lands inside the temp
   dir. `delete_path` ONLY deletes files in this
   registry — unregistered files (source code, configs, user data) are
   always refused. Never rely on `execute_command` for deletion — `rm` is
   blocked by the sandbox and intentionally not relaxed.
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
   hide required entry-point tools.
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
7. Temporary / scratch files the agent creates during a task should use the
   per-turn temp directory: pass `temp: true` to `write_file` (resolves
   `file_path` under `runtime_ctx::temp_dir()`, which is co-located with
   tool-overflow in `<sessions_root>/<session>.assets/tmp/`, outside the
   project dir). `file_path` must be a relative filename only — absolute
   paths are rejected and directory components are stripped to `file_name()`
   so the file never escapes the temp dir. For manual cleanup of files in the
   temp dir, use
   `delete_path` (structured deletion with sandbox + protected-dir checks).
   Never rely on `execute_command` for deletion — `rm` is blocked by the
   sandbox and intentionally not relaxed.

## Related detailed guide

- `docs/agent-guides/ai-tools.md`
