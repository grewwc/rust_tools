# Tools Guide

## Scope

Applies to `src/bin/ai/tools/**`. Layer separation: schema/metadata in
`registry/`, execution in `service/`, shared helpers/state in `storage/`.

## Key invariants

1. **Layer separation.** Schema/metadata in `registry/`, execution in `service/`,
   shared helpers/state in `storage/`.
2. **Names and paths.** Tool names are verb-first `snake_case`; relative paths
   resolve through `runtime_ctx::effective_cwd()`.
3. **Progressive loading.** Default per-turn group is `core`
   (`DEFAULT_TURN_TOOL_GROUPS` in `driver/skill_runtime.rs`). Tools tagged only
   `builtin` are lazy-loaded via `enable_tools`. `os_tools` (process/IPC/shm/env)
   are `builtin` + `executor` only and stay deferred even for executor agents.
   Explicit `tools:` name lists pin eager visibility.
4. **Registry-driven metadata.** Display via `ToolDisplayRegistration`; history
   retention via `ToolHistoryPolicyRegistration`. Do not add broad fields to
   `ToolSpec` or reintroduce name-keyed policy chains in `history/compress/`.
5. **History policy semantics.** `lossy_compress` and `prune` are orthogonal.
   Preserve truth for `plan`, `read_file`, `execute_command` diagnostics, and
   subagent task tools with explicit overflow stubs/file pointers, not lossy
   summaries.
6. **Temp files.** `write_file(temp=true)` writes under `runtime_ctx::temp_dir()`
   and registers a relative path in the JSON temp registry. `delete_path` only
   deletes registered temp files. Delete project/source/config files (incl.
   git-tracked) via `apply_patch` with a `*** Delete File:` section.
7. **Process groups.** `execute_command` runs in its own process group. Keep
   background pgids in the in-memory session registry and kill by process group at
   teardown; do not persist pgids across restarts.
8. **Self-describing truncation.** Truncating tools must distinguish complete,
   failed, and incomplete output. Include shown-vs-total counts and a concrete
   narrow/page hint when output is cut.
9. **Patch/read/search contracts.** `apply_patch` anchors on remove lines,
   normalizes confusable typographic chars (smart quotes, dashes, NBSP) so
   model-introduced variants match without corrupting output, and treats
   ambiguous hunk matches as a hard error. Prefer one `apply_patch` call with
   multiple `@@` hunks per file (one `*** Update File:` section); one envelope for
   multi-file edits. `read_file` paginates by line and by character cap. Text
   search lives in the dedicated grep/search tools, not here.
10. **Subagent tools are top-level only.** The `task` family
    (`task`/`task_spawn`/`task_wait`/`task_status`/`task_cancel`) must be hidden
    from subagents, not reintroduced by `enable_tools`, and rejected when
    `SUBAGENT_DEPTH > 0`. Results are scoped by session + owner pid; a process
    must not see/wait/cancel parent/sibling task ids. Surfaced child outputs must
    remind the parent to produce its own summary. Subagent launches use a capped
    copy of the agent manifest (`SUBAGENT_MAX_ITERATIONS`) with leaf-task
    convergence constraints; do not lower the primary agent's budget to tune
    subagent behavior. Never surface child thinking or streamed response bodies.
11. **Wall-clock safety net.** Stuck subagents are reaped after
    `SUBAGENT_WALL_CLOCK_TIMEOUT` (30 min) by both `task_wait` (per-call) and the
    driver `run_loop` (per-epoch `reap_timed_out_subagents()`), so they are killed
    even if the parent never calls `task_wait`. The reaper writes a terminal
    `timeout`/`cancelled` result but leaves registry cleanup to the collecting
    `task_wait`; `task_cancel` skips already-finished tasks so it never discards a
    real result.
12. **Interactive skill handoff.** `request_user_input` is a driver-owned control
    tool, visible only during an active skill turn. Scope its signal by
    `TURN_IDENTITY`; never infer a cross-turn continuation from response text.
