//! mcp_computer — a stdio JSON-RPC MCP server that gives the `a` agent generic
//! computer-use control of the local desktop: screen capture, display and window
//! enumeration, and mouse / keyboard / scroll injection.
//!
//! Design notes are in crates/mcp_computer/AGENTS.md:
//! - Protocol boilerplate (stdin main loop / method dispatch / write-back) is
//!   reused from `mcp_stdio::run`; this file only supplies the `McpServer`
//!   implementation and selects the platform backend.
//! - The tool surface is platform-neutral; `backend/` holds one implementation
//!   per OS behind identical signatures, so a new platform is a new module and
//!   never a change to the tools the model sees.
//! - Screenshots are written to disk and reported by path. The host's `read_file`
//!   auto-upgrades image paths to image-input semantics, so the vision model does
//!   see the pixels without this server emitting image content blocks (the host
//!   only reads `content[0].text` from a tool result).
//! - Every operation is bounded by `with_timeout` (default 90s, below the host's
//!   `request_timeout_ms` of 120s) and every failure message avoids the host's
//!   transport trigger words, so a failing tool never costs the host the process.

mod backend;
mod tools;

#[tokio::main]
async fn main() {
    mcp_stdio::run(tools::ComputerServer).await;
}