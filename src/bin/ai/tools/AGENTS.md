# Tools Guide

## Scope

Applies to `src/bin/ai/tools/**`.
Module responsibilities: schema/metadata in `registry/`, execution in
`service/`, shared helpers/state in `storage/`.

## Key invariants

1. **Layer separation.** Keep schema/metadata in `registry/`, execution in
   `service/`, and shared helpers/state in `storage/`.
2. **Names and paths.** Tool names are verb-first `snake_case`; relative paths
   resolve through `runtime_ctx::effective_cwd()`.
3. **Progressive loading.** The default per-turn group is `core`
   (`DEFAULT_TURN_TOOL_GROUPS` in `driver/skill_runtime.rs`). Tools tagged only
   `builtin` are lazy-loaded via `enable_tools`. `os_tools` process/IPC/shm/env
   tools are `builtin` + `executor` only and stay deferred even for executor
   agents. Explicit `tools:` name lists pin eager visibility.
4. **Registry-driven metadata.** Display behavior uses
   `ToolDisplayRegistration`; history retention uses
   `ToolHistoryPolicyRegistration`. Do not add broad fields to `ToolSpec` or
   reintroduce name-keyed policy chains in `history/compress/`.
5. **History policy semantics.** `lossy_compress` and `prune` are orthogonal.
   Preserve truth for `plan`, `read_file`, `execute_command` diagnostics, and
   subagent task tools with explicit overflow stubs/file pointers instead of
   lossy summaries.
6. **Temp files.** `write_file(temp=true)` writes under `runtime_ctx::temp_dir()`
   and registers a relative path in the JSON temp registry. `delete_path` only
   deletes registered temp files. Delete existing project/source/config files
   through `apply_patch` with a `*** Delete File:` envelope section, including
   git-tracked files.
7. **Process groups.** `execute_command` runs in its own process group. Keep
   background pgids in the in-memory session registry and kill by process group
   at teardown; do not persist pgids across restarts.
8. **Self-describing truncation.** Truncating tools must distinguish complete,
   failed, and incomplete output. Include shown-vs-total counts and a concrete
   narrow/page hint when output is cut.
9. **Patch/read/search contracts.** `apply_patch` anchors on remove lines and
   tolerates stale context only when the removal is unique; line matching
   normalizes confusable typographic characters (smart quotes, dashes, NBSP) so
   model-introduced variants still match without corrupting output (context
   lines emit the original file text). A `ReplaceInLine` envelope op
   (`anchor:`/`old:`/`new:`) does anchored inline substring replacement outside
   the unified-diff path. `read_file` paginates by line and by character cap,
   computing continuation offsets from rendered lines. Text search lives in the
   dedicated grep/search tools.
10. **Subagent tools.** The `task` orchestration family
    (`task`, `task_spawn`, `task_wait`, `task_status`, `task_cancel`) is
    top-level only: subagents must not see it, `enable_tools` must not
    reintroduce it, and execution paths must reject it when
    `SUBAGENT_DEPTH > 0`. Results are session-scoped, and surfaced child
    outputs must remind the parent to produce its own summary. Subagent
    launches use a capped copy of the selected agent manifest
    (`SUBAGENT_MAX_ITERATIONS`) and wrap the prompt with leaf-task convergence
    constraints; do not lower the primary agent's manifest budget to tune
    subagent behavior.

11. **Wall-clock safety net.** Stuck subagents are reaped after
    `SUBAGENT_WALL_CLOCK_TIMEOUT` (30 min) via two paths: `task_wait`'s per-call
    check and the driver `run_loop`'s per-epoch `reap_timed_out_subagents()`
    scan (so a subagent is killed even if the parent never calls `task_wait`).
    The reaper kills the process and writes a `timeout`/`cancelled` terminal
    result but never destroys the channel/futex or removes the registry entry -
    that is left to the collecting `task_wait` ready path. `reap_timed_out_subagents`
    takes locks in two non-overlapping steps (registry-only then kernel-only) to
    avoid a lock cycle with `task_wait` (which holds registry -> kernel).
    `task_cancel` skips already-finished tasks (via `is_task_pending`) so it
    never overwrites/discards a real result.
