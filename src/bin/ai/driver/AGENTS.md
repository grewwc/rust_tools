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
   system prompts, hidden catalogs, skill prompts, and reminders; keep reminder
   text consistent with the registry behavior.
4. **History is evidence.** Compression and pruning must preserve tool outputs,
   subagent results, and truncation state through explicit stubs or file pointers;
   never silently summarize away the only source of truth.
5. **Retry behavior is intentional.** Retry only on well-classified transport,
   provider, or context-window failures; treat retry/backoff changes as
   user-visible.
6. **Streaming protocol stability.** Event ordering and final-response semantics
   are part of the runtime contract; terminal/TUI previews are not a substitute for
   the persisted conversation state.
7. **Skill runtime isolation.** Skill/agent manifests determine visible tools,
   MCP allowlists, prompt overlays, and inheritance. Keep default visibility and
   explicit name pinning aligned with `tools/registry/`.
8. **Subagent lifecycle.** Scope tasks by session and owner pid. Persist
   delivered results before IPC cleanup, keep delivery distinct from
   integration, restore unintegrated evidence after context rebuild, and
   preserve isolated-memory artifacts when permanent-memory merge fails.
   Sync/audit hard timeouts must preserve child history and publish a bounded
   recovery payload with the last runtime phase; audit work should emit
   incremental checkpoints so a missing final answer does not erase progress.
   **Depth guard:** only the top-level agent may delegate to a child; child
   subagents must work directly when orchestration tools are hidden.
9. **Code-grounding reads stay serial.** Do not batch `read_file` calls in the
    driver path / system prompt; use each result to refine the next lookup.
10. **Model-visible tool results have a hard cap.** Store oversized output in a
    session overflow file with a bounded, self-describing stub; apply the same cap
    when rebuilding context without mutating history.
11. **Progress truth comes from the raw current tool round.** Sample current-round
    mutation/progress from the pre-compression tool-call snapshot, not only from
    compressed `messages`.
12. **Runtime environment is prompt context.** The base system prompt must include
    the current OS, architecture, and shell so generated commands target the actual
    execution platform.
13. **Foreground owns the terminal.** Background subagents must not write live
    model/thinking/tool/ANSI output to stdout/stderr; they publish results through
    task IPC for `task_wait` / `task_status`. The driver may expose one compact
    foreground-owned status line, refreshed only at scheduler safe points.
14. **Interactive skill handoffs are explicit and one-shot.** Preserve an active
    skill across turns only after its `request_user_input` control tool succeeds;
    consume that continuation on the next normal turn, let an explicit skill pin
    override it, and never infer it from response wording or question marks.
15. **Persist the actual response model.** Automatic request fallback does not
    rewrite `app.current_model`; model-dependent projection and canonical-history
    provenance use the model that actually produced each response.
16. **Target-scoped instructions precede mutation.** Load applicable rules
    root-to-deep before the first write, prioritize a paused mutation over
    previously observed targets, reject paths outside the project root, and keep
    preflight grace separate from normal tool and kernel quotas.
17. **Reasoning control is separate from the answer.** A reasoning controller must
    use a parsed control channel before it can influence the user-facing prompt.
18. **Redirect notes never quote user content.** Keep redirect text fixed and
    runtime-owned; retain the exact input only in its user-role message.
19. **Completion claims require evidence.** Recheck unsupported post-project-
   mutation completion claims once; temp-only command side effects do not count.
   Otherwise append one explicit runtime warning to the user-visible final and
   persist the unverified state for later context. A warning-only response is still
   an incomplete final and receives the normal one-time synthesis grace.
20. **Scheduler blocking is event-driven.** Foreground waits, background-task
    completion, shutdown, and wall-clock deadlines wake the driver through the
    scheduler notifier; never restore fixed-interval polling sleeps in the main loop.
21. **Incomplete finals get one synthesis grace.** A tool-backed final that only
    promises another read/check is not completion: retry once with tools disabled
    (no reset of iteration/tool accounting). If a no-tool synthesis still emits a
    tool call, reject and retry synthesis once before warning and stopping. Never
    replace a required verification round with no-tool synthesis.
22. **Terminal rendering never mutates conversation state.** Visible assistant
    text streams live to the terminal as it arrives, including tool-round
    narration and candidate finals that a gate may later reject or warn on.
    Terminal output is a non-authoritative preview: the persisted
    `assistant_text`/`turn_messages` remain the single source of truth for
    history, context, and gate decisions, and rendering must never alter them.
23. **Model-guided offloading is request-boundary work.** Before every logical
    model request, apply eligible prune marks to the transient `messages`
    projection before normal context budgeting, archive full tool output before
    replacing it with a recall stub, and never mutate canonical `turn_messages`.
    Prune confirmation counts are session-scoped durable runtime state: restore
    and filter them against the live prunable ids, clear them on history rewind,
    and never leak them across session/persona switches.
24. **Execution limits require runtime evidence.** Do not accept a model-authored
    read-only phase/limit claim as the reason requested changes were skipped unless
    the current turn contains matching runtime or tool evidence. Reopen once with
    tools preserved; at a hard stop, make the unsupported claim visible.
