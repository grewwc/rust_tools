# Driver Guide

## Scope

Applies to `src/bin/ai/driver/**` and driver-facing glue in nearby modules.
Key areas: prompt assembly and tool loops in `skill_runtime.rs`,
history/compression in `turn_runtime/`, subagent flows in `turn_runtime/orchestrator.rs`.

## Key invariants

1. **Single turn coordinator.** The driver owns prompt assembly, model requests,
   tool execution loops, history mutation, and final response emission.
2. **No UI/tool shortcuts.** UI rendering and tool services may report events, but
   they must not mutate conversation state behind the driver's back.
3. **Prompt assembly stays explicit.** Avoid duplicating instructions across
   system prompts, hidden catalogs, skill prompts, and reminders. If a capability
   is hidden/lazy, keep the reminder text consistent with the registry behavior.
4. **History is evidence.** Compression and pruning must preserve tool outputs,
   subagent results, and truncation state through explicit stubs or file pointers;
   never silently summarize away the only source of truth.
5. **Retry behavior is intentional.** Retry only on well-classified transport,
   provider, or context-window failures. Treat retry/backoff changes as user
   visible behavior changes.
6. **Streaming protocol stability.** Event ordering and final-response semantics
   are part of the runtime contract; terminal/TUI previews are not a substitute
   for the persisted conversation state.
7. **Skill runtime isolation.** Skill/agent manifests determine visible tools,
   MCP allowlists, prompt overlays, and inheritance. Keep default visibility and
   explicit name pinning aligned with `tools/registry/`.
8. **Subagent lifecycle.** Task registries are session-scoped. Surface completed
   child results once, mark them observed, and keep outstanding-task reminders
   limited to the current session.
9. **Depth guard.** Only the top-level agent may delegate to a child subagent;
   child subagents must work directly when orchestration tools are hidden.
10. **Code-grounding reads stay serial.** `read_file` must not be encouraged or
    executed as a parallel batch in the driver path or system prompt guidance;
    use each result to refine the next lookup so evidence stays narrow and the
    model converges instead of flooding context.
11. **Current-turn tool results have a hard cap.** Keep normal recent precision
    results raw for recall, but never put physically huge tool output directly
    into `messages`; write it to a session overflow file and keep a bounded,
    self-describing stub with original-call anchors.
12. **Progress truth comes from the raw current tool round.** Tool-loop and
    Progress Budget checks may run after mid-turn compression, but current-round
    mutation/progress must be sampled from the pre-compression tool-call snapshot,
    not inferred solely from compressed `messages`.
13. **Runtime environment is prompt context.** The base system prompt must include
    the current OS, architecture, and shell so generated commands target the
    actual execution platform instead of a model-default platform.
