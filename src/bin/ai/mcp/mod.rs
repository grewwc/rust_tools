mod client;
mod config;
pub(in crate::ai) mod connection;
mod http;
mod io;
mod jsonrpc;

pub(in crate::ai) use client::{McpClient, SharedMcpClient};
pub(in crate::ai) use client::{send_notification_to_conn, send_request_to_conn};
pub(super) use config::load_mcp_config_from_file;

/// MCP servers commonly return `-32601 Method not found` without echoing the
/// method name. Discovery must therefore key off the JSON-RPC code alone; a
/// missing optional capability is not a connection failure.
pub(in crate::ai) fn is_method_not_found(err: &str) -> bool {
    err.contains("-32601")
}
