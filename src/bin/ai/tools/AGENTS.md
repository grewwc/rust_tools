7. Temporary / scratch files the agent creates during a task should use
   `write_file(temp=true)`, which writes under `runtime_ctx::temp_dir()`
   (per-session: `<effective_cwd>/.agent_tmp/<session>/`) AND registers
   the file in a persistent JSON registry (`storage::temp_registry`).
   `delete_path` ONLY deletes files in this registry — unregistered files
   (source code, configs, user data) are always refused. The registry
   survives session restarts, enabling cross-session temp cleanup. Never
   rely on `execute_command` for deletion — `rm` is blocked by the sandbox
   and intentionally not relaxed.
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
   `file_path` under `runtime_ctx::temp_dir()`, auto-cleaned at turn end by
   `orchestrator::run_turn_body`). For manual cleanup of files outside the
   temp dir, use `delete_path` (structured deletion with sandbox + protected-
   dir checks). Never rely on `execute_command` for deletion — `rm` is
   blocked by the sandbox and intentionally not relaxed.

## Related detailed guide

- `docs/agent-guides/ai-tools.md`
