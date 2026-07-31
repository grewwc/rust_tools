# Driver Guide

## Scope

Applies to `src/bin/ai/driver/**` and nearby driver-facing glue. Key areas are
prompt assembly in `skill_runtime.rs`, turn orchestration in `turn_runtime/`,
and driver-side subagent lifecycle in `turn_runtime/orchestrator.rs`.

## Key invariants

1. **Single turn coordinator.** The driver owns prompt assembly, model requests,
   tool execution loops, history mutation, and final response emission.
2. **No UI/tool shortcuts.** UI rendering and tool services may report events, but
   must not mutate conversation state behind the driver's back.
3. **Prompt assembly stays explicit.** Avoid duplicating instructions across
   system prompts, hidden catalogs, skill prompts, and reminders. If a capability
   is hidden/lazy, keep the reminder text consistent with the registry behavior.
4. **History is evidence.** Compression and pruning must preserve tool outputs,
   subagent results, and truncation state through explicit stubs or file pointers;
   never silently summarize away the only source of truth. Persist the raw turn
   separately from the bounded request-context projection.
5. **Retry behavior is intentional.** Retry only on well-classified transport,
   provider, or context-window failures. Treat retry/backoff changes as user-visible.
6. **Streaming protocol stability.** Event ordering and final-response semantics
   are part of the runtime contract; terminal/TUI previews are not a substitute for
   the persisted conversation state.
7. **Skill runtime isolation.** Skill/agent manifests determine visible tools,
   MCP allowlists, prompt overlays, and inheritance. Keep default visibility and
   explicit name pinning aligned with `tools/registry/`.
8. **Subagent lifecycle.** Scope tasks by session and owner pid. Persist delivered
   results before IPC cleanup, keep delivery distinct from integration, and restore
   unintegrated evidence after context rebuild. Preserve isolated-memory artifacts
   when permanent-memory merge fails.
9. **Depth guard.** Only the top-level agent may delegate to a child subagent;
   child subagents must work directly when orchestration tools are hidden.
10. **Code-grounding reads stay serial.** Do not batch `read_file` calls or
    encourage parallel reads in the driver path / system prompt; use each result to
    refine the next lookup so evidence stays narrow.
11. **Model-visible tool results have a hard cap.** Store oversized output in a
    session overflow file and keep a bounded, self-describing stub with call
    anchors. Apply the same cap when rebuilding context without mutating history.
12. **Progress truth comes from the raw current tool round.** Current-round
    mutation/progress is sampled from the pre-compression tool-call snapshot, not
    inferred solely from compressed `messages` (tool-loop / Progress Budget checks
    may run after mid-turn compression).
13. **Runtime environment is prompt context.** The base system prompt must include
    the current OS, architecture, and shell so generated commands target the actual
    execution platform.
14. **Foreground owns the terminal.** Background subagents must not write live
    model/thinking/tool/ANSI output to stdout/stderr; they publish results through
    task IPC for `task_wait` / `task_status` to aggregate. The driver may expose one
    compact foreground-owned status line, refreshed only at scheduler safe points.
15. **Interactive skill handoffs are explicit and one-shot.** Preserve an active
    skill across turns only after its `request_user_input` control tool succeeds;
    consume that continuation on the next normal turn, let an explicit skill pin
    override it, and never infer it from response wording or question marks.
16. **Persist the actual response model.** Automatic request fallback does not
    rewrite `app.current_model`; model-dependent projection and canonical-history
    provenance use the model that actually produced each response.
17. **Target-scoped instructions precede mutation.** Load applicable rules
    root-to-deep before the first write, reject paths outside the project root, and
    keep preflight grace separate from normal tool and kernel quotas.
18. **Reasoning control is separate from the answer.** A reasoning controller must
    use a parsed control channel before it can influence the user-facing prompt.
19. **Redirect notes never quote user content.** Keep redirect text fixed and
    runtime-owned; retain the exact input only in its user-role message.
20. **Completion claims require evidence.** Recheck unsupported post-mutation
    completion claims once; otherwise persist and display an explicit warning.
