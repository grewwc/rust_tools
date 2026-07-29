# Driver Guide

## Scope

Applies to `src/bin/ai/driver/**` and driver-facing glue in nearby modules.
Key areas: prompt assembly and tool loops in `skill_runtime.rs`,
history/compression in `turn_runtime/`, subagent flows in `turn_runtime/orchestrator.rs`.

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
8. **Subagent lifecycle.** Task registries are scoped by session and owner process
   pid. Surface completed child results once, mark them observed, and keep
   outstanding-task reminders limited to the current owner process.
9. **Depth guard.** Only the top-level agent may delegate to a child subagent;
   child subagents must work directly when orchestration tools are hidden.
10. **Code-grounding reads stay serial.** Do not batch `read_file` calls or
    encourage parallel reads in the driver path / system prompt; use each result to
    refine the next lookup so evidence stays narrow.
11. **Model-visible tool results have a hard cap.** Never put physically huge tool
    output directly into `messages`; write it to a session overflow file and keep a
    bounded, self-describing stub with original-call anchors. Rebuilt
    canonical-history tails apply the same absolute cap without mutating canonical rows.
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
17. **Target-scoped project instructions precede mutation.** Keep cwd-level
    instructions in the stable base prompt. `apply_patch` / `write_file` must pause
    before their first side effect when applicable scoped rules are not loaded;
    render root-to-deep, reject paths outside the project root, and avoid
    re-injecting root rules. Scoped preflight uses a separate bounded grace budget
    (currently 8 rounds per turn) so multi-directory work does not consume normal
    tool iterations; it must not extend kernel tool-call quotas.
18. **Thinking control is not a main-response protocol.** Do not register the
    current `ThinkingOrchestrator` in production while it requests strict JSON
    without consuming state transitions. Any future reasoning controller must use a
    separate parsed control call before it can influence the user-facing prompt.
19. **Redirect notes never quote user content.** A runtime redirect may tell the
    model that the final `role=user` message supersedes an unfinished tool loop,
    but the note must contain only fixed runtime-owned text. Keep the exact user
    input exclusively in its user-role message and preserve relevant verified
    history evidence for reuse.
