# AGENTS.md - mcp_pdf

## Scope

Standalone stdio JSON-RPC **MCP server** exposing PDF text/metadata extraction
(the `parse_pdf` tool) to MCP hosts. Stateless: each call parses the file
directly from disk; no session state. The parsing logic lives in the shared
root lib (`rust_tools::pdfw`, also used by `a` in-process and
`src/bin/mcp_ocr.rs`); this crate only wraps it in the MCP protocol.

Shared transport, text delivery, error wording, and timeout contracts live in
[the mcp_stdio guide](../mcp_stdio/AGENTS.md).

## Layout

```text
src/main.rs   # PdfServer + impl McpServer (protocol loop via mcp_stdio::run)
src/tools.rs  # tools_list + handle_tools_call dispatch + parse_pdf tool
```

## Build / Test

Crate type-check target: `cargo check -p mcp_pdf`. Protocol/schema dispatch
changes require a binary protocol smoke. Changes to parser integration, file
access, parse options, metadata/text output, or error mapping additionally require
`cargo build -p mcp_pdf` and a small PDF fixture round-trip through the server,
checking metadata and extracted text.

## Golden rules

1. **Wrapper boundary.** `parse_pdf_tool` (`src/tools.rs`) owns the MCP wrapper;
   shared result/error/timeout behavior belongs to `crates/mcp_stdio/AGENTS.md`.
2. Error strings start with `open_failed` / `parse_failed` /
   `extract_text_failed` via `pdf_err_kind` (`src/tools.rs`) to mirror
   `PdfParseError`, so hosts can distinguish "file unreadable" from
   "text extraction failed".
3. Keep the server stateless and synchronous per call: PDF parsing is
   CPU-bound and fast; no background work, no long-lived handles.

## Host registration

`op_timeout_ms` (`src/tools.rs`) owns the operation-cap setting. Shared host
configuration is described in **Host registration** in
`crates/mcp_stdio/AGENTS.md`.
