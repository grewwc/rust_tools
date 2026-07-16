# Tools Guide

## Scope

Applies to `src/bin/ai/tools/**`.
Read `docs/agent-guides/ai-tools.md` before changing tool registration,
execution policy, sandboxing, path resolution, history/display policy, or
progressive loading.

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
   Preserve truth for `plan`, `read_file`, `code_search`, and subagent task
   tools with explicit overflow stubs/file pointers instead of lossy summaries.
6. **Temp files.** `write_file(temp=true)` writes under `runtime_ctx::temp_dir()`
   and registers a relative path in the JSON temp registry. `delete_path` only
   deletes registered temp files.
7. **Process groups.** `execute_command` runs in its own process group. Keep
   background pgids in the in-memory session registry and kill by process group
   at teardown; do not persist pgids across restarts.
8. **Self-describing truncation.** Truncating tools must distinguish complete,
   failed, and incomplete output. Include shown-vs-total counts and a concrete
   narrow/page hint when output is cut.
9. **Patch/read/search contracts.** `apply_patch` anchors on remove lines and
   tolerates stale context only when the removal is unique. `read_file` paginates
   by line and by character cap, computing continuation offsets from rendered
   lines. `code_search` is the single entry point for LSP, file, text, and
   structural navigation.
10. **Subagent tools.** `task`/`task_spawn` enforce the depth cap, results are
    session-scoped, and surfaced child outputs must remind the parent to produce
    its own summary.

## Related detailed guide

- `docs/agent-guides/ai-tools.md`
