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
src/lib.rs   # JsonRpcErr, text_content, cap_text,
             # with_timeout (per-op ms cap), write_result / write_err,
             # McpServer trait, run() loop
```

## Build / Test

```bash
cargo check -p mcp_stdio
```

Changes to dispatch, notifications, result/error serialization, text capping, or
operation-timeout behavior additionally require a protocol smoke in one consuming
server. Build that binary when needed; exercise initialize, initialized
notification, tools/list, and a representative tools/call. Real-app round-trips
are required only when the consuming server's app integration is affected.

## Invariants (do not break)

1. **Domain-neutral.** Keep app-specific logic and wording out of this crate;
   shared timeout/truncation wording belongs in `with_timeout` and `cap_text`
   (`src/lib.rs`).
2. **Error wording must avoid the host's restart triggers.** The host kills +
   restarts the subprocess when a returned error matches any phrase in
   `is_transport_error`, `is_protocol_desync_error`, or `is_user_interrupt_error`
   (`src/bin/ai/mcp/client.rs`). These case-insensitive classifiers are the
   authoritative trigger set; do not maintain a second phrase list here or in
   consuming servers. Keep operation failures distinct from transport failures.
3. **`run<S>` is only `block_on`-ed** by the consuming bin's `#[tokio::main]`,
   never `tokio::spawn` - this is why `async fn` sits directly on the trait
   (no `async-trait`, no `Send` bound). Do not add `Send` requirements.
4. **Text delivery.** The `a` host exposes only `content[0].text` to the model.
   Return payloads through `text_content` and bound extracted text/HTML with
   `cap_text` (`src/lib.rs`); `CAP_CHARS` there owns the limit. Save large or binary
   output to disk and return its path.
5. **Per-op timeout < host timeout.** Wrap operations in
   `with_timeout(op_timeout_ms())` and keep the server cap strictly below the
   host's configured `request_timeout_ms`. Each server's `op_timeout_ms` owns its
   defaults and overrides; host/server defaults do not establish this ordering.
6. **Server boundary.** New OS-app MCP servers implement `McpServer` and keep
   their app driver in the consuming crate; reuse this dispatch loop instead of
   copying transport code.
7. **Notifications get no response.** A parsed request without an `id` is a
   JSON-RPC 2.0 notification: its side effects still run, but no response line
   is written — not even `-32601` for an unknown method. Never answer
   `notifications/initialized` with an ack. `run` (`src/lib.rs`) owns this rule.

## Host registration

For `a`, register server binaries under `mcpServers` in `~/.config/mcp.json`.
Set `request_timeout_ms` to satisfy **Per-op timeout < host timeout**; verify the
effective configured deadline rather than assuming either default is suitable.
