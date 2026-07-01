# AI Driver Detailed Guide

## Scope

Long-form reference for `src/bin/ai/driver/**`.
Do not duplicate this whole document into prompt-loaded `AGENTS.md` files.

## Files to know

- `mod.rs`: entry orchestration and main run loop
- `skill_runtime.rs`: tool selection, skill routing, system prompt builder,
  project-context injection
- `turn_runtime/`: per-turn prepare / iterate / finalize flow
- `thinking/`: goal decomposition and thinking-engine helpers
- `reflection/`: background reflection and long-term-memory writeback
- `runtime_ctx.rs`: scoped cwd / memory overrides for sub-agents
- `tools/oauth.rs`: MCP OAuth orchestration glue

## Prompt and project instruction path

- `agents.rs::load_project_instruction_docs*` discovers scoped `AGENTS.md` /
  `CLAUDE.md` files along cwd ancestry inside the project boundary.
- `skill_runtime.rs::build_project_instruction_prompt()` turns those docs into
  prompt Policy text.
- `push_project_context_for_turn()` is the behavior boundary. Current rule:
  scoped project instruction docs remain available even when the turn is
  classified as general-knowledge mode; that mode only suppresses helper facts
  such as project-type hints.
- `src/bin/a.rs` may warm the cache early, but startup preload is not the
  semantic source of truth for prompt injection.

## General-knowledge mode

- This mode is meant to hide repo-specific discovery/tool noise for short
  conceptual turns.
- It should not suppress repo-local instruction documents when the user is
  operating inside the repository and those docs are part of the scoped
  behavior contract.
- If you change the heuristic, re-check both prompt construction and tool
  filtering paths, plus their tests.

## Runtime context and cwd

- `runtime_ctx::effective_cwd()` is the path authority used by tools, file
  resolution, recall helpers, and sub-agents.
- `SUBAGENT_CWD` overrides process cwd inside scoped sub-agent execution.
- If cwd semantics change, audit file tools, search tools, patch tools, command
  validation, recall, and sub-agent creation.

## One-shot and background flows

- One-shot knowledge maintenance commands such as
  `--consolidate-knowledge` / `--migrate-legacy-knowledge` run before the
  interactive `run_loop()`.
- Reflection runs after each turn and writes back to long-term memory; keep it
  decoupled from foreground prompt assembly.

## Useful tests

- `ai::driver::skill_runtime::tests::project_instruction_prompt_includes_repo_docs_from_cwd_scope`
- `ai::driver::skill_runtime::tests::project_instructions_remain_available_in_general_knowledge_mode`
- `ai::driver::skill_runtime::tests::prompt_introspection_query_still_keeps_project_instructions`
- `ai::driver::turn_runtime::prepare::*` for question-shape / recall heuristics
