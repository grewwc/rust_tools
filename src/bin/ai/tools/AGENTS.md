# Tools Guide

## Scope

Applies to `src/bin/ai/tools/**`. Layers: `registry/` (schema/metadata), `service/` (execution), `storage/` (helpers/state).

## Key invariants

1. **Layer separation.** Keep schema in `registry/`, logic in `service/`, helpers in `storage/`.
2. **Names & paths.** Verb-first `snake_case`; relative paths via `runtime_ctx::effective_cwd()`.
3. **Progressive loading.** Default `core` (`DEFAULT_TURN_TOOL_GROUPS` in `driver/skill_runtime.rs`). `builtin`-only tools are lazy via `enable_tools`; `os_tools` need `executor` plus `builtin`. Explicit `tools:` lists pin eager visibility.
4. **Registry-driven metadata.** Display=`ToolDisplayRegistration`, history=`ToolHistoryPolicyRegistration`, replay=`ToolReplayRegistration` (opt-in; never infer from names). Built-in metadata lives as `tool_descriptions/<tool>.json` auto-discovered by `build.rs`; a test fails if any registered tool lacks a non-empty JSON.
5. **History policy.** `lossy_compress` ⊥ `prune`. Preserve truth for `plan`/`read_file`/`execute_command`/task tools via overflow stubs, not lossy summaries.
6. **Temp & writable roots.** `write_file(temp=true)` → `runtime_ctx::temp_dir()` + session temp registry (authoritative allowlist for same-session isolated dirs). `*** Delete File:` via `apply_patch` for project deletes. Skills dir (`ai.skills.dir`, default `~/.config/rust_tools/skills`) is also always-writable; keep in sync with `configured_write_roots` in `storage/file_store.rs`.
7. **Process groups.** `execute_command` runs in its own pgid; track in session registry, kill by pgid at teardown, don't persist across restarts.
8. **Truncation.** Truncating tools must distinguish complete/failed/incomplete, include shown-vs-total counts and a narrow/page hint.
9. **Patch/read/search.** `apply_patch` anchors on removed text, normalizes typographic confusables, rejects ambiguous matches. Multi-file diffs auto-split per file; deletions need explicit envelope. `*** Replace in line:` tries exact then confusable-trimmed unique match. Failures prefix `Hunk N/M:` and echo paste-ready `<<<PATCH_TEXT` block; only ambiguous trips stale-patch guard. `patch`/`patch_file` are mutually exclusive (8K vs 64K caps; `patch_file` must be session-temp or under `effective_cwd`). Pure-insert hunks locate by line number only. Details in `storage/file_store.rs` / `service/patch_tools.rs`.
10. **Subagent tools are top-level only.** Hide `task` family when `SUBAGENT_DEPTH>0`; never restore via `enable_tools`. Scope by session+owner pid, persist before IPC cleanup, require `task_integrate`. `task_wait` lone-spawn hint `[tool_followup:lone_spawn]` is one-shot display-only.
11. **Wall-clock net.** Both `task_wait` and driver loop reap past `SUBAGENT_WALL_CLOCK_TIMEOUT` (60m); write terminal result, preserve finished results, signal `cancel_stream` before aborting.
12. **Interactive handoff.** `request_user_input` is visible only during active skill; scope by `TURN_IDENTITY`; never infer continuation from text.
13. **Commands are non-interactive by default.** Non-PTY disables pagers/prompts/color; PTY preserves interactive behavior. Mutating commands auto-load scoped instructions. `git commit` family gates in `service/command.rs`: prompt in TTY, fail-closed otherwise.
14. **Mutation log.** `FileStore::write_all`/`apply_patch` append best-effort JSONL to `<session_assets>/mutation_log.jsonl` (capped before/after, seq), skipping session root artifacts. Serialized by global lock; `/audit` reads via `DRIVER_CTX`, falls back to `git diff` only when empty.
15. **Team/Graph reuses task truth.** `manage_team`/`run_agent_graph` checkpoint orchestration only; all work flows through kernel task registry with integration/cancellation/depth guard.
