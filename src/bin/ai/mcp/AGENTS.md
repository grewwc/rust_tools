# MCP Guide

## Scope

Applies to `src/bin/ai/mcp/**` and nearby MCP-related execution glue.

Read `docs/agent-guides/ai-mcp.md` before changing server initialization,
client routing, transport error handling, OAuth flows, or timeout semantics.

## Key invariants

1. MCP servers are stdio JSON-RPC subprocesses, not HTTP services.
2. `routing_snapshot()` is for metadata/routing; real tool execution must use
   the live shared client/connection set.
3. Request timeouts and restart logic can kill long-running flows such as OAuth
   waits; treat timeout changes as behavior changes.
4. When MCP tool visibility is gated by progressive loading, prompt hints and
   enablement behavior must stay consistent with real configured tool names.

## Related detailed guide

- `docs/agent-guides/ai-mcp.md`
