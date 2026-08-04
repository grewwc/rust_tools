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
   session temp registry (e.g. a subagent's isolated temp dir) are writable by
   `write_file`/`apply_patch` even outside `effective_cwd`/allowed roots — the
   registry is the authoritative same-session temp allowlist.
7. **Process groups.** `execute_command` runs in its own process group. Keep
   background pgids in the in-memory session registry and kill by process group at
   teardown; do not persist pgids across restarts.
8. **Self-describing truncation.** Truncating tools must distinguish complete,
   failed, and incomplete output. Include shown-vs-total counts and a concrete
   narrow/page hint when output is cut.
9. **Patch/read/search contracts.** `apply_patch` anchors on removed text,
   normalizes supported typographic confusables, and rejects ambiguous matches.
   Unified diffs are single-file; multi-file edits and deletion use an explicit
   envelope; `*** Replace in line:` is the low-friction single-line substring
   edit. Context-mismatch / ambiguous / out-of-order errors echo current text as
   a prefix-free, paste-ready block (`<<<PATCH_TEXT` ... `PATCH_TEXT>>>`) so the
   model can rebuild without re-reading; classify the diagnostic before the
   block (so source text cannot mimic an error), and for multi-file patches
   block only the failed target. Only an ambiguous match trips the stale-patch
   guard (context mismatch repairable directly; out-of-order is a hunk-ordering
   problem). `patch` and `patch_file` are optional, mutually exclusive sources:
   missing, `null`, or empty-string values mean absent; exactly one non-empty
   string must remain after normalization, while non-string values are rejected.
   Validation is intrinsic to normal execution; do not expose a model-facing
   `dry_run` switch. The executor may honor legacy `dry_run: true` calls only to
   prevent historical no-write requests from silently becoming real writes.
   `read_file` paginates by line/char cap; text search stays in dedicated search
   tools. Large patches: inline `patch` hard-capped at 8K chars
   (split into multiple calls, or pass `patch_file` = a session temp file via
   `write_file(temp=true)` or a file under `effective_cwd`; `patch_file` has its
   own 64K safety cap, so it is the path for large patches); `@@ -0` normalizes
   to insert-at-start (line 1).
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
14. **Mutation log is best-effort and session-scoped.** `FileStore::write_all`
    and `apply_patch` delete/rollback paths append a JSONL entry to
    `<session_assets>/mutation_log.jsonl` (before/after capped, op, seq); it
    never affects the real write (failures silently dropped). Skip every path
    under the sessions root (assets, subagent scratch `subagent-cwd-*` siblings,
    checkpoints) - session runtime artifacts, not project changes; skipping the
    whole root keeps parallel subagent writes out of view. Appends are
    serialized by a process-global lock (main agent + parallel subagents share
    one log). `/audit` reads this log (via `current_session_assets_dir()`, needs
    `DRIVER_CTX`) to show the main agent's own changes without concurrent work
    contaminating the view; falls back to `git diff` only when the log is empty.
