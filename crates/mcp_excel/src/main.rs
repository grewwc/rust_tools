//! mcp_excel -- a stdio JSON-RPC MCP server that drives an installed Microsoft Excel
//! via AppleScript (osascript), giving the `a` Agent the ability to operate the real
//! Excel app (open/read cells/read ranges/write cells/write ranges/export CSV/list sheets...).
//!
//! Design points are in crates/mcp_excel/AGENTS.md:
//! - **No long-lived session**: each osascript run is an independent subprocess; the
//!   Excel app itself keeps workbooks open across calls, so ExcelServer holds no session
//!   state at all, simpler than mcp_browser.
//! - Protocol boilerplate (stdin main loop / method dispatch / write-back) is reused via
//!   `mcp_stdio::run`; this file only implements the `McpServer` trio plus tool-related
//!   code; shutdown uses the default no-op.
//! - Per-operation timeouts are enforced by with_timeout (default 90s < host
//!   request_timeout_ms 120s); timeouts return a clean JSON-RPC error without transport
//!   trigger words, so the host does not kill us.
//! - Saving writes files on the Rust side (export_csv), bypassing the -50 limit of
//!   Excel's sandboxed save.

mod osa;
mod tools;

use mcp_stdio::{JsonRpcErr, McpServer};
use serde_json::Value;

/// Excel MCP server -- sessionless unit struct. The Excel app itself holds the open
/// workbook state, shared across independent osascript subprocesses, so no state is
/// needed here.
struct ExcelServer;

impl McpServer for ExcelServer {
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
    mcp_stdio::run(ExcelServer).await;
}
