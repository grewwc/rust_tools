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
   Same-turn same-argument result reuse is opt-in via `ToolReplayRegistration`;
   never infer replay safety from read-like tool names.
5. **History policy semantics.** `lossy_compress` and `prune` are orthogonal.
   Preserve truth for `plan`, `read_file`, `execute_command` diagnostics, and
   subagent task tools with explicit overflow stubs/file pointers, not lossy
   summaries.
6. **Temp files.** `write_file(temp=true)` uses `runtime_ctx::temp_dir()` and the
   session temp registry. Delete project/source/config files via `apply_patch`
   with an explicit `*** Delete File:` section. Paths already registered in the
   session temp registry (e.g. a temp file created by a subagent in an isolated
   temp dir) are writable by `write_file`/`apply_patch` even when they fall
   outside `effective_cwd`/allowed roots — the registry is the authoritative
   same-session temp allowlist.
7. **Process groups.** `execute_command` runs in its own process group. Keep
   background pgids in the in-memory session registry and kill by process group at
   teardown; do not persist pgids across restarts.
8. **Self-describing truncation.** Truncating tools must distinguish complete,
   failed, and incomplete output. Include shown-vs-total counts and a concrete
   narrow/page hint when output is cut.
9. **Patch/read/search contracts.** `apply_patch` anchors on removed text,
   normalizes supported typographic confusables, and rejects ambiguous matches.
   Unified diffs are single-file; multi-file edits and deletion use an explicit
   envelope; a `*** Replace in line:` section is the low-friction path for a
   single-line substring edit. Context-mismatch / ambiguous / out-of-order errors
   each echo the current text as a prefix-free, paste-ready block (`<<<PATCH_TEXT`
   ... `PATCH_TEXT>>>`) so the model can rebuild without re-reading. Only an
   ambiguous match trips the stale-patch guard (context mismatch repairable
   directly, out-of-order is a hunk-ordering problem); classify the diagnostic
   before the block so source text cannot mimic an error, and for multi-file
   patches block only the failed target. `read_file` paginates by line and
   character cap; text search stays in dedicated search tools.
10. **Subagent tools are top-level only.** Hide and reject the `task` family when
    `SUBAGENT_DEPTH > 0`; `enable_tools` must not restore it. Scope tasks by
    session + owner pid, persist results before IPC cleanup, and require
    `task_integrate` after delivery. Cap only the child manifest and never expose
    child thinking or streamed response bodies.
11. **Wall-clock safety net.** Both `task_wait` and the driver loop reap tasks past
    `SUBAGENT_WALL_CLOCK_TIMEOUT`. Write a terminal result before collection,
    preserve already-finished results, and signal `cancel_stream` before aborting
    the Tokio task so synchronous child commands can stop their process group.
12. **Interactive skill handoff.** `request_user_input` is a driver-owned control
    tool, visible only during an active skill turn. Scope its signal by
    `TURN_IDENTITY`; never infer a cross-turn continuation from response text.
13. **Command execution defaults to non-interactive.** Non-PTY runners disable
    blocking pagers, editors, prompts, and color; explicit PTY runs preserve
    interactive terminal behavior. Analyze mutating commands before execution
    and automatically load scoped instructions for inferred project targets.
