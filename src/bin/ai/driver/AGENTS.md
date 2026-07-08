# Driver Guide

## Module layout

- `mod.rs`: main run loop, session management, background dispatch, MCP init
- `tests.rs`: driver-level integration tests
- `tools/`: tool execution subsystem
  - `mod.rs`: tool routing, execution, retry, parallel batches, tool cache
  - `async_pipe.rs`: async tool pipe system (task_spawn/task_status/task_wait lifecycle, channel/futex, snapshot persistence)
  - `tests.rs`: tool execution tests
  - `barrier.rs`, `oauth.rs`, `sync_task.rs`: specialized submodules

## Scope

Applies to `src/bin/ai/driver/**`.

Read `docs/agent-guides/ai-driver.md` before changing the run loop, turn
preparation, prompt assembly, thinking, reflection, or runtime context.

## Key invariants

1. `skill_runtime.rs` owns system prompt assembly and project-instruction injection.
2. `build_project_instruction_prompt()` materializes auto-discovered
   `AGENTS.md` / `CLAUDE.md` docs from `agents.rs`.
3. `push_project_context()` must keep repo-local instruction docs available for
   repo-scoped work; it should not silently drop the actual scoped instruction
   docs.
4. `runtime_ctx::effective_cwd()` is the working-directory authority for tools
   and sub-agents.
5. One-shot knowledge maintenance flows run before the interactive `run_loop()`.
6. `runtime_ctx::temp_dir()` is per-session
   (`<sessions_root>/<session>.assets/tmp/`, co-located with tool-overflow,
   outside the project dir; falls back to `<effective_cwd>/.agent_tmp/<session>/`
   when DRIVER_CTX is unavailable) and persists across turns. There is no
   auto-cleanup; the agent explicitly deletes temp files via `delete_path`
   (which only works on files registered via `write_file(temp=true)`).
7. Process/correction notes (truncation-retry hints, cache-hit notes,
   discover_skills followups) are turn-scoped: push them to `messages` only,
   never via `append_message_pair`, so they are not persisted into
   `turn_messages` and cannot accumulate across turns.
8. `turn_runtime/persistence.rs` skips persisting turn messages only for
   *ephemeral* one-shot runs (`one_shot_mode && cli.session.is_none()`), i.e.
   the runs `cleanup_one_shot` will delete right after. Background mode
   (`a -bg`) and explicit `--session` one-shot (`a -ss <id> "q"`) keep the
   session, so they MUST persist — otherwise `/sessions` title and `/history`
   come up empty.

## Related detailed guide

- `docs/agent-guides/ai-driver.md`
