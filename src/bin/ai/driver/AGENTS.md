# Driver Guide

## Scope

Applies to `src/bin/ai/driver/**`.

Read `docs/agent-guides/ai-driver.md` before changing the run loop, turn
preparation, prompt assembly, general-knowledge mode, thinking, reflection, or
runtime context.

## Key invariants

1. `skill_runtime.rs` owns system prompt assembly and project-instruction injection.
2. `build_project_instruction_prompt()` materializes auto-discovered
   `AGENTS.md` / `CLAUDE.md` docs from `agents.rs`.
3. `push_project_context_for_turn()` must keep repo-local instruction docs
   available for repo-scoped work; general-knowledge mode may suppress helper
   facts, but should not silently drop the actual scoped instruction docs.
4. `runtime_ctx::effective_cwd()` is the working-directory authority for tools
   and sub-agents.
5. One-shot knowledge maintenance flows run before the interactive `run_loop()`.

## Related detailed guide

- `docs/agent-guides/ai-driver.md`
