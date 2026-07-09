# MCP Guide

## Scope

Applies to `src/bin/ai/mcp/**` and nearby MCP-related execution glue.

Read `docs/agent-guides/ai-mcp.md` before changing server initialization,
client routing, transport error handling, OAuth flows, or timeout semantics.

## Key invariants

1. MCP servers are stdio JSON-RPC subprocesses, not HTTP services.
2. `routing_snapshot()` is for metadata/routing; real tool execution must use
   the live shared client/connection set.
3. `notifications/initialized` is a JSON-RPC **notification** (no `id`, no
   response expected). Send it via `send_notification_to_conn`, never
   `send_request_to_conn` — sending it as a request causes some servers
   (e.g. Feishu) to close the stdio stream.
4. `send_request_to_conn` must skip responses with `id: null` or missing `id`.
   Some servers (e.g. `mcp_ocr`) send a spurious `{"id":null,"result":{}}`
   acknowledgment after `notifications/initialized`. If accepted, it consumes
   the response slot for the next request, leaving the real response in the
   buffer and causing an id mismatch on the subsequent call.
5. Request timeouts and restart logic can kill long-running flows such as OAuth
   waits; treat timeout changes as behavior changes.
6. When MCP tool visibility is gated by progressive loading, prompt hints and
   enablement behavior must stay consistent with real configured tool names.

## Related detailed guide

- `docs/agent-guides/ai-mcp.md`
