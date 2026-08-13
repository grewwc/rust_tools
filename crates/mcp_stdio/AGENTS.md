# AGENTS.md - mcp_stdio

## Scope

Shared **MCP-over-stdio protocol skeleton** (lib crate) reused by the standalone
MCP server binaries (`mcp_browser`, `mcp_excel`). Carries everything unrelated
to "which app is driven": JSON-RPC transport, result/error writing, text
capping, per-operation timeout, and the `run<S: McpServer>` dispatch loop.
Not a binary; light deps only (`tokio` io-std/io-util/time + `serde_json`).

## Layout

```text
src/lib.rs   # JsonRpcErr, text_content, cap_text (CAP_CHARS = 24_000),
             # with_timeout (default 90s), write_result / write_err,
             # McpServer trait, run() loop
```

## Build / Test

```bash
cargo check -p mcp_stdio
```

No focused unit tests - the gate is the consuming binaries' build + smoke test.

## Invariants (do not break)

1. **Domain-neutral.** This crate must stay free of app-specific logic or
   wording: the timeout message ("operation reached the N ms server cap") and
   `cap_text` truncation note are written to be reusable by any server.
2. **Error wording must avoid the host's transport trigger words.** Never let
   a returned error contain `mcp response timeout`, `broken pipe`,
   `closed the stream`, `process exited`, `failed to read response`, or
   `failed waiting for mcp response` - the host kills + restarts the subprocess
   on those (see `src/bin/ai/mcp/client.rs`).
3. **`run<S>` is only `block_on`-ed** by the consuming bin's `#[tokio::main]`,
   never `tokio::spawn` - this is why `async fn` sits directly on the trait
   (no `async-trait`, no `Send` bound). Do not add `Send` requirements.
4. **Only `content[0].text` reaches the model** - servers must return payloads
   via `text_content(...)`; `cap_text` caps extracted text/HTML at 24K chars
   (the host offloads results above ~32K to disk).
5. **Per-op timeout < host timeout.** `with_timeout` defaults to 90s so it
   fires before the host's `request_timeout_ms` (120s recommended); a server
   that needs longer must still stay under the host cap.
6. **New "drive an OS-native app" MCP servers** reuse this crate (per root
   AGENTS.md): implement `McpServer` + the app driver, nothing else.