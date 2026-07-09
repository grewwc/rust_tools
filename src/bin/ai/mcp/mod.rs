mod client;
mod config;
pub(in crate::ai) mod connection;
mod io;
mod jsonrpc;

pub(super) use client::{McpClient, SharedMcpClient};
pub(in crate::ai) use client::{send_notification_to_conn, send_request_to_conn};
pub(super) use config::load_mcp_config_from_file;
#[cfg(unix)]
pub(in crate::ai) use io::set_fd_nonblocking;
