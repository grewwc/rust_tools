# MCP Guide

## Scope

Applies to `src/bin/ai/mcp/**` (flat module: `client.rs`, `connection.rs`,
`config.rs`, `io.rs`, `jsonrpc.rs`).

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
   long-running flows; treat changes as user-visible.
6. **Lazy MCP schemas.** Per-turn tool schemas include MCP tools only when a
   skill/agent declares an explicit `mcp_servers` allowlist. Otherwise expose the
   hidden MCP catalog and load tools on demand through `enable_tools`.
7. **Name consistency.** Prompt hints, hidden catalogs, and enablement behavior
   must stay consistent with the real configured MCP tool names.
8. **The request deadline covers trickling data.** The read loop must check the
   deadline on every iteration, not only when the fd is idle: a server that
   streams one response line in many small writes must not hold the caller past
   `request_timeout_ms`.
9. **One malformed server config does not disable the rest.** A single typo in
   one `mcpServers` entry is skipped with a warning; it must not take down
   every other configured server.
10. **A restart invalidates the cached schemas.** The tool/resource/prompt
    cache is rebuilt after a connection restart, otherwise the catalog keeps
    serving pre-restart schemas while the live connection holds a new
    generation. Rebuild outside the per-server lock (outer → inner order).
