# AGENTS.md - mcp_stdio

## Scope

Shared **MCP-over-stdio protocol skeleton** (lib crate) reused by every standalone
MCP server binary (`mcp_browser`, `mcp_computer`, `mcp_excel`, `mcp_pdf`).
Carries everything unrelated to "which app is driven": JSON-RPC transport,
result/error writing, text capping, per-operation timeout, and the
`run<S: McpServer>` dispatch loop. Not a binary; light deps only (`tokio`
io-std/io-util/time + `serde_json`).

## Layout

```text
src/lib.rs   # JsonRpcErr, text_content, cap_text (CAP_CHARS = 24_000),
             # with_timeout (per-op ms cap), write_result / write_err,
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
2. **Error wording must avoid the host's restart triggers.** The host kills +
   restarts the subprocess when a returned error matches any phrase in
   `is_transport_error`, `is_protocol_desync_error`, or `is_user_interrupt_error`
   (`src/bin/ai/mcp/client.rs`): `broken pipe`, `closed the stream
   unexpectedly`, `mcp response timeout`, `failed waiting for mcp response`,
   `failed to read response`, `process exited with status`, `failed to get
   stdin`, `failed to get stdout`, `mcp response id mismatch`, `failed to parse
   response`, `invalid json-rpc version`, `canceled by user`, `cancelled by
   user`, `interrupted by user`.
3. **`run<S>` is only `block_on`-ed** by the consuming bin's `#[tokio::main]`,
   never `tokio::spawn` - this is why `async fn` sits directly on the trait
   (no `async-trait`, no `Send` bound). Do not add `Send` requirements.
4. **Only `content[0].text` reaches the model** - servers must return payloads
   via `text_content(...)`; `cap_text` caps extracted text/HTML at 24K chars
   (the host offloads results above ~32K to disk).
5. **Per-op timeout < host timeout.** Servers wrap every op in
   `with_timeout(op_timeout_ms())`, defaulting that cap to 90s. The host's
   default `request_timeout_ms` is only 30s (`src/bin/ai/types.rs`), so the 90s
   cap fires first only when the host timeout is configured above 90s; a server
   that needs longer must still stay under the host's configured cap.
6. **New "drive an OS-native app" MCP servers** reuse this crate (per root
   AGENTS.md): implement `McpServer` + the app driver, nothing else.
