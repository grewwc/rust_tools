# AGENTS.md - mcp_pdf

## Scope

Standalone stdio JSON-RPC **MCP server** exposing PDF text/metadata extraction
(the `parse_pdf` tool) to MCP hosts. Stateless: each call parses the file
directly from disk; no session state. The parsing logic lives in the shared
root lib (`rust_tools::pdfw`, also used by `a` in-process and
`src/bin/mcp_ocr.rs`); this crate only wraps it in the MCP protocol.

## Layout

```text
src/main.rs   # PdfServer + impl McpServer (protocol loop via mcp_stdio::run)
src/tools.rs  # tools_list + handle_tools_call dispatch + parse_pdf tool
```

## Build / Test

```bash
cargo build -p mcp_pdf   # quick: tokio+serde_json + mcp_stdio + root lib
```

No unit tests (needs real PDF files). Gate: build + smoke-test a small PDF
round-trip through the server binary (or `mcp_stdio::run`) and check metadata +
text come back through `content[0].text`.

## Golden rules

1. All results go through `content[0].text` (host reads only that); cap long
   extracted text with `cap_text` (24K).
2. Every parse runs under `with_timeout` (default 90s < host request_timeout_ms
   120s); error messages must never contain the host's transport trigger words
   (see crates/mcp_stdio/AGENTS.md).
3. Error strings start with `open_failed` / `parse_failed` /
   `extract_text_failed` to mirror the three `PdfParseError` variants, so hosts
   can distinguish "file unreadable" from "text extraction failed".
4. Keep the server stateless and synchronous per call: PDF parsing is
   CPU-bound and fast; no background work, no long-lived handles.
