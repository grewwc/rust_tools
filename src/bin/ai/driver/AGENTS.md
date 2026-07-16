# Driver Guide

## Scope

Applies to `src/bin/ai/driver/**` and driver-facing glue in nearby modules.
Read `docs/agent-guides/ai-driver.md` before changing turn orchestration,
prompt assembly, skill/runtime setup, history compression, or subagent flows.

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

## Related detailed guide

- `docs/agent-guides/ai-driver.md`
