# Driver Guide

## Scope

Applies to `src/bin/ai/driver/**` and nearby driver glue. The driver owns prompt
assembly, turn orchestration, scheduling, session/runtime context, agent and MCP
routing, commands, observation, and terminal output.

## Key invariants

These are driver-only additions to `src/bin/ai/AGENTS.md` invariant 2.

1. **Prompt assembly is explicit.** Don't duplicate instructions across system prompts/catalogs/skill prompts/reminders. Trust-boundary block is injected once by `build_system_prompt` in `skill_runtime.rs`. Keep earlier request bytes stable: scoped instructions and session guidance stay resident (`prompt_residency_v1`, dropped by history truncation) and are replaced in place — only a paused mutation's documents may evict resident ones from the budget; the per-turn reminder is recorded on the persisted user message and replayed byte-identically. Known exceptions: image digest substitution, text-attachment materialization, and edited instruction files.
2. **Progress sampling.** Sample progress from the pre-compression current tool-round snapshot, not compressed `messages`.
3. **Retry & streaming are contracts.** Retry only on classified transport/provider/context-window failures. Event ordering and final-response semantics are stable; terminal previews are not persisted truth.
4. **Skill runtime isolation.** Manifests determine visible tools, MCP allowlists, prompt overlays, and inheritance. Keep default `core` visibility and explicit pinning aligned with `tools/registry/`. Multi-skill: ordered active set, union tools/MCP, most-restrictive `disable_*` wins.
5. **Subagent recovery.** Restore unintegrated evidence after rebuild and preserve child history/artifacts on merge failure or timeout; timeouts also publish bounded last-phase recovery and incremental checkpoints. `tools/AGENTS.md` owns task scoping, persistence, integration, depth, and timeout protocols.
6. **Code-grounding reads stay serial.** Don't batch `read_file` in driver/system-prompt path; use each result to refine the next lookup.
7. **Tool-result projection.** Archive oversized output behind self-describing stubs and apply the same cap during context rebuild without mutating canonical history. `evidence_status.rs` marks provably historical or stale results only in model-visible content; guidance lives in `system_prompts/tool_result_evidence.md`.
8. **Runtime environment is prompt context.** Base prompt includes OS/arch/shell/`effective_cwd`; keep cwd distinct from project root. Foreground owns the terminal — background subagents publish via task IPC only, never write live ANSI to stdout/stderr (one compact status line at scheduler safe points is allowed).
9. **Interactive handoffs are explicit, one-shot.** `request_user_input` preserves active skill(s) for exactly the next normal turn; in non-skill turns the tool call is the explicit ask and the turn simply ends with the question. An explicit skill pin overrides the continuation; `activate_skill`/`deactivate_skill` manage the ordered active set, which continuation preserves whole.
10. **Persist the actual response model.** Auto fallback doesn't rewrite `app.current_model`; projection/provenance use the model that produced each response. Provenance is per message and never inferred: `Message` carries no model (`source_model` travels beside it), overflow archives persist bodies only, so expanded messages report unknown instead of inheriting the stub's model, and `/history` labels assistant rows only.
11. **Mutation preflight.** Load target-scoped instructions root-to-deep before first write, prioritize paused mutation over observed targets, reject paths outside project root, keep preflight grace separate from tool/kernel quotas. Reasoning control needs a parsed control channel; redirect notes never quote user content.
12. **Final-response gates.** Quality gates share one bounded reopen; the
    structural outstanding-work/`complete` gate has an independent reopen and
    binds only a plan written this turn (or resume/goal-mode turns); older plans
    project as non-binding carried context. Final wording never controls gate
    decisions. The per-turn self-check also carries the auditable-answer
    contract: consequential claims state their basis and scope where they
    stand, limitations sit next to the claim they limit, and a claim backed by
    only part of a set describes that part. Simple common-knowledge answers
    stay plain, with no added citations, qualifiers, or steps.
    - **Completion.** After a provable `apply_patch`/`write_file` mutation, tool
      records alone decide: command intent and temp-only effects do not count; any
      successful post-mutation activity allows, while provable zero activity
      reopens once then warns. Non-mutation action promises keep the dangling gate;
      reopen asks the model to inspect/check or state limits. A confirmed failed
      `cargo check` warns immediately; warning-only finals retain one no-tool
      synthesis grace.
    - **Citation.** Reopen once only for locally provable bad `path:line`
      citations; inaccessible/oversized files remain unknown, and a second invalid
      final receives an unverified-citation warning.
13. **Scheduler is event-driven.** Foreground wakes via scheduler notifier (completion, shutdown, deadlines, `task_wait` budget) — no polling sleeps. Progress events are display-only (`notify_scheduler` for status line, never `wake_process`).
14. **Terminal & offloading.** Withhold `Completed` bodies until final gates accept; previews are display-only. Before requests, reclaim eligible old tool evidence only from transient context within bounded limits, preserving recent/protected and task-integration evidence; archive before stubbing and never mutate canonical turns. Runtime evidence, not model claims, controls limits. `request_user_input` metadata is the sole ask protocol.
15. **Side-note listener lifetime.** The Ctrl+G side-note listener stays alive whenever the driver is not reading stdin: the whole turn AND the subagent wait window (foreground parked after `task_wait` PARKED, or background subagents still in flight). Exactly one listener / one stdin lease exists at a time: run_turn's turn-scoped guard during a turn, run_loop's wait-window guard (`spawn_wait_window_side_note_input`) during a wait, none while the input box is open. The wait-window guard is created once per wait window and kept alive across the whole scheduler wait — scheduler ticks elapse while it lives; it is dropped before the next iteration can start a turn or open the input box, restoring termios. Each spawn binds the current session's history file, so respawn with fresh values after session switches.
