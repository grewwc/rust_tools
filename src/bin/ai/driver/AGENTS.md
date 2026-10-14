# Driver Guide

## Scope

Applies to `src/bin/ai/driver/**` and nearby driver glue. The driver owns prompt
assembly, turn orchestration, scheduling, session/runtime context, agent and MCP
routing, commands, observation, and terminal output.

## Key invariants

These are driver-only additions to `src/bin/ai/AGENTS.md` invariant 2.

1. **Prompt assembly is explicit.** Don't duplicate instructions across system prompts/catalogs/skill prompts/reminders. Trust-boundary block is injected once by `build_system_prompt` in `skill_runtime.rs`.
2. **Progress sampling.** Sample progress from the pre-compression current tool-round snapshot, not compressed `messages`.
3. **Retry & streaming are contracts.** Retry only on classified transport/provider/context-window failures. Event ordering and final-response semantics are stable; terminal previews are not persisted truth.
4. **Skill runtime isolation.** Manifests determine visible tools, MCP allowlists, prompt overlays, and inheritance. Keep default `core` visibility and explicit pinning aligned with `tools/registry/`. Multi-skill: ordered active set, union tools/MCP, most-restrictive `disable_*` wins.
5. **Subagent recovery.** Restore unintegrated evidence after rebuild and preserve child history/artifacts on merge failure or timeout; timeouts also publish bounded last-phase recovery and incremental checkpoints. `tools/AGENTS.md` owns task scoping, persistence, integration, depth, and timeout protocols.
6. **Code-grounding reads stay serial.** Don't batch `read_file` in driver/system-prompt path; use each result to refine the next lookup.
7. **Tool-result projection.** Archive oversized output behind self-describing stubs and apply the same cap during context rebuild without mutating canonical history. `evidence_status.rs` marks provably historical or stale results only in model-visible content; guidance lives in `system_prompts/tool_result_evidence.md`.
8. **Runtime environment is prompt context.** Base prompt includes OS/arch/shell/`effective_cwd`; keep cwd distinct from project root. Foreground owns the terminal — background subagents publish via task IPC only, never write live ANSI to stdout/stderr (one compact status line at scheduler safe points is allowed).
9. **Interactive handoffs are explicit, one-shot.** `request_user_input` preserves active skill(s) for exactly the next normal turn; in non-skill turns the tool call is the explicit ask and the turn simply ends with the question. An explicit skill pin overrides the continuation. `activate_skill`/`deactivate_skill` manage the ordered active set; continuation preserves the whole set.
10. **Persist the actual response model.** Auto fallback doesn't rewrite `app.current_model`; projection/provenance use the model that produced each response. Provenance is per message and never inferred: `Message` carries no model (`source_model` travels beside it), overflow archives persist bodies only, so expanded messages report unknown instead of inheriting the stub's model, and `/history` labels assistant rows only.
11. **Mutation preflight.** Load target-scoped instructions root-to-deep before first write, prioritize paused mutation over observed targets, reject paths outside project root, keep preflight grace separate from tool/kernel quotas. Reasoning control needs a parsed control channel; redirect notes never quote user content.
12. **Final-response gates.** Quality gates share one bounded reopen; the structural outstanding-work/`complete` gate has an independent reopen. Final wording never controls gate decisions.
    - **Completion.** After a provable `apply_patch`/`write_file` mutation, tool records alone decide: command intent and temp-only effects do not count; any successful post-mutation activity allows, while provable zero activity reopens once then warns. Non-mutation action promises keep the dangling gate; reopen asks the model to inspect/check or state limits. A confirmed failed `cargo check` warns immediately; warning-only finals retain one no-tool synthesis grace.
    - **Citation.** Reopen once only for locally provable bad `path:line` citations; inaccessible/oversized files remain unknown, and a second invalid final receives an unverified-citation warning.
13. **Scheduler is event-driven.** Foreground wakes via scheduler notifier (completion, shutdown, deadlines, `task_wait` budget) — no polling sleeps. Progress events are display-only (`notify_scheduler` for status line, never `wake_process`).
14. **Terminal & offloading.** Withhold `Completed` bodies until final gates accept; previews are display-only. Before requests, reclaim eligible old tool evidence only from transient context within bounded limits, preserving recent/protected and task-integration evidence; archive before stubbing and never mutate canonical turns. Runtime evidence, not model claims, controls limits. `request_user_input` metadata is the sole ask protocol.
