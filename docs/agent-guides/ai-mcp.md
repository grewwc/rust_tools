# AI MCP Detailed Guide

## Scope

Long-form reference for `src/bin/ai/mcp/**` and adjacent MCP execution glue in
driver/tool code.

## Core model

- MCP servers are stdio subprocesses launched from config (`mcp.json`), not
  local HTTP services.
- `driver/mcp_init.rs` handles preload and metadata discovery.
- `mcp/client.rs` owns request dispatch, tool calls, timeout/restart behavior,
  and cached metadata.
- `mcp/connection.rs` wraps process pipes plus timeout-aware reads.

## Routing vs execution

- `routing_snapshot()` is metadata-only. It is safe for tool selection and
  schema hints, but not for real MCP execution because it has no live `servers`
  map.
- Real tool calls must run through the shared live client/connection set.

## Timeout and restart pitfall

- Transport timeouts are not benign. `client.rs` treats them as
  restart-worthy transport failures.
- If a server is blocked in a legitimate long-running tool call (for example
  waiting for an OAuth loopback callback), a too-short `request_timeout_ms` can
  kill and restart the subprocess mid-flow.
- Treat per-server timeout changes as behavior changes; validate long waits
  explicitly.

## Feishu OAuth notes

- The loopback callback port (for example `127.0.0.1:8711/callback`) is a
  temporary OAuth listener created by the Feishu MCP server while
  `oauth_wait_local_code` is active.
- That port not listening all the time does not mean MCP failed to start.
- For Feishu server capabilities and examples, `docs/mcp-feishu.md` is the
  deeper product-specific reference.

## Progressive loading

- MCP metadata may be preloaded before the first turn, but tool exposure can
  still be gated.
- If exposure rules change, keep `enable_tools`, prompt hints, and actual
  configured tool names aligned.

## Validation checklist

- Can the server still initialize and list tools/resources/prompts?
- Does synchronous execution use the live shared client?
- Do long-running tool calls survive timeout/restart policy?
- Do prompt hints expose the same MCP tool names the runtime can actually enable?
