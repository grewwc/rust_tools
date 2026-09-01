# Driver Guide

## Scope

Applies to `src/bin/ai/driver/**` and nearby driver glue: prompt assembly
(`skill_runtime.rs`), turn orchestration (all inside `turn_runtime/`:
`orchestrator.rs`, `loop_detection`, `checkpoint`, `progress`, `notes`, …), and
driver infrastructure:

- `scheduler.rs` / `background_dispatch.rs`: background scheduling
- `session.rs` / `process_context.rs` / `runtime_ctx.rs`: session, history paths, `effective_cwd`
- `agent_routing.rs`: skill manifests, primary agent, hot-reload
- `mcp_init.rs` / `mcp_lifecycle.rs`: MCP bootstrap
- `model.rs` / `input.rs` / `signal.rs` / `hooks.rs`: model resolution, input, SIGINT, hooks
- `observer.rs` / `decision_log.rs` / `note_search.rs` / `skill_watcher.rs`: observation, logging, search, watching
- `commands/` / `system_prompts/`: command impls (e.g. `/audit`) and `include_str!`
  prompt templates; supporting modules `thinking/`, `reflection/`, `embedding/`,
  `hook_registry.rs`, `session_pid.rs`, `side_note.rs`

## Key invariants

1. **Single turn coordinator.** Driver owns prompt assembly, model requests, tool loops, history mutation, and final response. No UI/tool shortcuts mutating state behind it.
2. **Prompt assembly is explicit.** Don't duplicate instructions across system prompts/catalogs/skill prompts/reminders. Trust-boundary block is injected once by `build_system_prompt` in `skill_runtime.rs`. Tool metadata may provide detailed `first_use_guidance`; the tool-result path emits it once after that tool's first call in the current user turn while keeping the resident schema compact.
3. **History is evidence.** Compression/pruning must preserve tool outputs, subagent results, and truncation via explicit stubs/file pointers — never silently summarize away truth. Sample progress from the pre-compression current tool-round snapshot, not compressed `messages`.
4. **Retry & streaming are contracts.** Retry only on classified transport/provider/context-window failures. Event ordering and final-response semantics are stable; terminal previews are not persisted truth.
5. **Skill runtime isolation.** Manifests determine visible tools, MCP allowlists, prompt overlays, and inheritance. Keep default `core` visibility and explicit pinning aligned with `tools/registry/`. Multi-skill: ordered active set, union tools/MCP, most-restrictive `disable_*` wins.
6. **Subagent lifecycle.** Scope by session+owner pid. Persist delivered results before IPC cleanup, keep delivery≠integration, restore unintegrated evidence after rebuild, preserve isolated-memory artifacts on merge failure. Hard timeouts must preserve child history, publish bounded recovery payload with last phase, and emit incremental checkpoints. **Depth guard:** only top-level may delegate (`SUBAGENT_DEPTH==0`).
7. **Code-grounding reads stay serial.** Don't batch `read_file` in driver/system-prompt path; use each result to refine the next lookup.
8. **Tool results are capped.** Oversized output goes to session overflow file with a self-describing stub; apply same cap when rebuilding context without mutating history. Results whose arguments prove they are historical/reference data (session storage paths, stale patch targets, `git log/show/blame`) get a `[reference: ...]` marker prepended to the model-visible content by `tool_result/execution/evidence_status.rs`; canonical history stays raw, and the model guidance lives in `system_prompts/tool_result_evidence.md`.
9. **Runtime environment is prompt context.** Base prompt includes OS/arch/shell/`effective_cwd`; keep cwd distinct from project root. Foreground owns the terminal — background subagents publish via task IPC only, never write live ANSI to stdout/stderr (one compact status line at scheduler safe points is allowed).
10. **Interactive handoffs are explicit, one-shot.** `request_user_input` preserves active skill(s) for exactly the next normal turn; an explicit skill pin overrides it. `activate_skill`/`deactivate_skill` manage the ordered active set; continuation preserves the whole set.
11. **Persist the actual response model.** Auto fallback doesn't rewrite `app.current_model`; projection/provenance use the model that produced each response.
12. **Mutation preflight.** Load target-scoped instructions root-to-deep before first write, prioritize paused mutation over observed targets, reject paths outside project root, keep preflight grace separate from tool/kernel quotas. Reasoning control needs a parsed control channel; redirect notes never quote user content.
13. **Completion and citation gates.** Re-check unsupported post-mutation completion claims once; temp-only side effects don't count. Gate acts only on provable `apply_patch`/`write_file` mutation; command-level "mutation" is intent classification (silent Allow). Any successful post-mutation activity (including unrecognized `python3` scripts or read-only tools) silently Allows; only provable zero activity reopens once then warns. A confirmed failed `cargo check` after mutation overrides and warns immediately; warning-only finals still get one no-tool synthesis grace. Separately, a conservative final `path:line` citation check reopens once only for locally provable bad conventional file citations; inaccessible or oversized files remain unknown, and a second invalid final receives an explicit unverified-citation warning.
14. **Scheduler is event-driven.** Foreground wakes via scheduler notifier (completion, shutdown, deadlines, `task_wait` budget) — no polling sleeps. Progress events are display-only (`notify_scheduler` for status line, never `wake_process`).
15. **Terminal & offloading.** Thinking and tool activity may stream as live preview, but a `Completed` assistant body is transactionally withheld until final-response gates accept it; rejected drafts are never committed to the terminal. Terminal preview never mutates `turn_messages`/`assistant_text`. Before each model request, apply prune marks to transient `messages` projection, archive full output before stubbing, never mutate canonical `turn_messages`; prune counts are session-scoped and filtered on rewind/switch. Don't accept model-authored limit claims without runtime/tool evidence. Question-guidance block is injected only when no skill active, `goal_mode==None`, and not background.
