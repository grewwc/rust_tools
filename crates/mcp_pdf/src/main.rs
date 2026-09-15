//! mcp_pdf -- a stateless stdio JSON-RPC MCP server exposing PDF text/metadata
//! extraction (the `parse_pdf` tool) to MCP hosts such as the `a` agent.
//!
//! Mirrors the mcp_excel pattern (see crates/mcp_excel/AGENTS.md):
//! - **No long-lived session**: every call parses the PDF file directly from
//!   disk, so PdfServer holds no state at all.
//! - Protocol boilerplate (stdin main loop / method dispatch / write-back) is
//!   reused via `mcp_stdio::run`; this file only implements the `McpServer` trio.
//! - Per-operation timeout via `with_timeout` (default 90s < host
//!   request_timeout_ms 120s); timeouts return a clean JSON-RPC error without
//!   transport trigger words, so the host does not kill us.
//! - Parsing itself lives in the shared `rust_tools::pdfw` module (also used by
//!   `a` in-process and by `src/bin/mcp_ocr.rs`); the server only wraps it.

mod tools;

use mcp_stdio::{JsonRpcErr, McpServer};
use serde_json::Value;

/// PDF parsing server -- sessionless unit struct.
struct PdfServer;

impl McpServer for PdfServer {
    fn initialize_result(&self) -> Value {
        tools::initialize_result()
    }
    fn tools_list_result(&self) -> Value {
        tools::tools_list_result()
    }
    async fn handle_tools_call(&mut self, params: Option<Value>) -> Result<Value, JsonRpcErr> {
        tools::handle_tools_call(params).await
    }
    // shutdown uses the default no-op implementation (no session to close).
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    mcp_stdio::run(PdfServer).await;
}
