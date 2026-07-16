# MCP Guide

## Scope

Applies to `src/bin/ai/mcp/**` and nearby MCP execution glue.
Read `docs/agent-guides/ai-mcp.md` before changing server initialization,
client routing, transport error handling, OAuth flows, schema loading, or timeout
semantics.

## Key invariants

1. **Transport model.** MCP servers are stdio JSON-RPC subprocesses, not HTTP
   services.
2. **Routing vs execution.** `routing_snapshot()` is metadata for discovery and
   routing; real tool execution must use the live shared client/connection set.
3. **Initialized notification.** `notifications/initialized` is a JSON-RPC
   notification: no `id`, no response expected. Send it with
   `send_notification_to_conn`, never `send_request_to_conn`.
4. **Response matching.** `send_request_to_conn` must ignore responses with
   missing or `null` ids; some servers emit spurious initialized acknowledgments
   that otherwise consume the next request's response slot.
5. **Timeouts are behavior.** Request timeouts and restart logic can interrupt
   long-running flows such as OAuth waits; treat changes as user-visible.
6. **Lazy MCP schemas.** Per-turn tool schemas include MCP tools only when a
   skill/agent declares an explicit `mcp_servers` allowlist. Otherwise expose the
   hidden MCP catalog and load tools on demand through `enable_tools`.
7. **Name consistency.** Prompt hints, hidden catalogs, and enablement behavior
   must stay consistent with the real configured MCP tool names.

## Related detailed guide

- `docs/agent-guides/ai-mcp.md`
