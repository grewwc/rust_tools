//! mcp_browser — a stdio JSON-RPC MCP server that provides browser automation
//! (navigate/click/type/extract/screenshot...) for the `a` agent.
//!
//! Design notes are in crates/mcp_browser/AGENTS.md:
//! - Protocol boilerplate (stdin main loop / method dispatch / write-back) is
//!   reused from `mcp_stdio::run`; this file only supplies the `McpServer`
//!   implementation: the three tool-related sections plus a `shutdown` override
//!   for graceful session close.
//! - Dual driver modes: the default `applescript` drives the user's already-open
//!   Chrome (new tab, keeps cookies, never quits the user's browser);
//!   `MCP_BROWSER_DRIVER=cdp` (or setting MCP_BROWSER_WS_URL) uses the original
//!   controlled-instance path. BrowserServer holds an Option<Session> enum and
//!   dispatches by mode.
//! - Every operation is bounded by `with_timeout` (default 90s < host
//!   `request_timeout_ms` 120s); on timeout it returns a clean JSON-RPC error
//!   free of transport trigger words so the host does not kill the process.

mod applescript;
mod browser;
mod tools;

use browser::{DriverMode, Session};
use mcp_stdio::{JsonRpcErr, McpServer};
use serde_json::Value;

/// Browser MCP server — holds a lazily-started session per driver mode. In
/// applescript mode it drives the user's already-open Chrome (the session tab is
/// tracked by window_id+tab_id); in cdp mode it lazily starts a controlled
/// instance. When the main loop exits, `shutdown` closes the session gracefully
/// (applescript mode only closes the session tab, never quits the user's
/// browser).
struct BrowserServer {
    mode: DriverMode,
    session: Option<Session>,
}

impl BrowserServer {
    fn new() -> Self {
        Self {
            mode: DriverMode::from_env(),
            session: None,
        }
    }
}

impl McpServer for BrowserServer {
    fn initialize_result(&self) -> Value {
        tools::initialize_result()
    }
    fn tools_list_result(&self) -> Value {
        tools::tools_list_result()
    }
    async fn handle_tools_call(&mut self, params: Option<Value>) -> Result<Value, JsonRpcErr> {
        if self.mode.is_applescript() {
            // Reuse the 13 tool definitions/parameter names from tools.rs; the
            // implementation runs on the applescript driver.
            let s = match &mut self.session {
                Some(Session::AppleScript(a)) => a,
                other => {
                    *other = Some(Session::AppleScript(applescript::ApplescriptSession::new()));
                    match self.session.as_mut().unwrap() {
                        Session::AppleScript(a) => a,
                        Session::Cdp(_) => unreachable!(),
                    }
                }
            };
            applescript::handle_tools_call(s, params).await
        } else {
            // The CDP path fully reuses tools.rs: take out the Cdp variant and
            // pass it to the original implementation.
            let mut cdp = match self.session.take() {
                Some(Session::Cdp(c)) => Some(c),
                other => {
                    self.session = other;
                    None
                }
            };
            let r = tools::handle_tools_call(&mut cdp, params).await;
            if let Some(c) = cdp {
                self.session = Some(Session::Cdp(c));
            }
            r
        }
    }
    /// Override the default no-op: gracefully close the session when the main
    /// loop exits. applescript mode only closes the session tab (never quits the
    /// user's Chrome); CDP mode closes the controlled instance.
    async fn shutdown(&mut self) {
        if let Some(s) = self.session.take() {
            match s {
                Session::Cdp(c) => c.shutdown().await,
                Session::AppleScript(mut a) => a.shutdown().await,
            }
        }
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    // Startup garbage collection: remove leftover temp profiles from processes
    // that were killed before they could shut down gracefully. See
    // browser::gc_stale_profiles — SIGKILL cannot be caught, so the fallback is
    // "clean up on next startup". Kept as an explicit one-liner in main
    // (browser-specific, synchronous, idempotent) rather than a trait hook.
    browser::gc_stale_profiles();

    mcp_stdio::run(BrowserServer::new()).await;
}
