// =============================================================================
// McpPort - MCP call port (decouples the driver from a direct dependency on the mcp crate)
// =============================================================================
// The driver previously talked to MCP processes directly through `crate::ai::mcp::*`,
// which forced tests to spawn real processes and left no way to intercept middleware.
// Extracting the port gives:
// - backend: `LiveMcpPort` is the production adapter; tests can inject mock / recording /
//   circuit-breaking middleware;
// - upstream: the driver is injected via `&dyn McpPort` and can be replaced with mock /
//   recording / circuit-breaking middleware;
// - compatibility: `McpError` is passed through unchanged so callers handle it uniformly.

use std::collections::HashMap;

use crate::ai::mcp::{McpClient, SharedMcpClient};

#[derive(Debug, Clone)]
pub struct McpToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

pub trait McpPort: Send + Sync {
    fn tool_definitions(&self) -> Vec<McpToolDef>;
    fn call_tool(
        &self,
        server: &str,
        tool: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>>;
    fn is_available(&self, server: &str) -> bool;
}

/// Production implementation: delegates to the real lifecycle of the existing
/// `crate::ai::mcp::McpClient`. Holds a `SharedMcpClient` (`Arc<Mutex<McpClient>>`).
/// Design points:
/// - `tool_definitions` briefly holds the outer lock only to `Arc::clone` the cache (O(1)),
///   then releases it;
/// - `is_available` briefly holds the outer lock for a single `get_str_ref` lookup;
/// - `call_tool` delegates to `McpClient::call_tool_on_shared`; the outer lock is used only
///   to clone the `Arc<Mutex<Connection>>` and allocate `next_id`, and is never held across
///   the blocking `writeln`/`read` or `restart_connection` calls, so cross-server
///   `tool_definitions`/`is_available` are not blocked for long.
/// This type is the correct replacement for `DefaultMcpPort`: if "Default = production",
/// use this type when wiring it in.
pub struct LiveMcpPort {
    client: SharedMcpClient,
}

impl LiveMcpPort {
    pub fn new(client: SharedMcpClient) -> Self {
        Self { client }
    }
    pub fn client(&self) -> &SharedMcpClient {
        &self.client
    }
}

impl McpPort for LiveMcpPort {
    fn tool_definitions(&self) -> Vec<McpToolDef> {
        // Short lock: only clone the Arc, then map without holding the lock;
        // warn explicitly on poison instead of failing silently.
        let cached = match self.client.lock() {
            Ok(g) => g.cached_tool_definitions_arc(),
            Err(e) => {
                eprintln!("[warn] McpClient poisoned in tool_definitions: {}", e);
                return Vec::new();
            }
        };
        cached
            .iter()
            .map(|td| McpToolDef {
                name: td.function.name.clone(),
                description: td.function.description.clone(),
                input_schema: td.function.parameters.clone(),
            })
            .collect()
    }

    fn call_tool(
        &self,
        server: &str,
        tool: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        // The outer lock never spans blocking I/O: delegate to `call_tool_on_shared`,
        // which holds the outer lock only briefly to clone the per-server
        // `Arc<Mutex<Connection>>` and allocate a request id; during `send_request_to_conn`
        // only the per-server inner lock is held. On restart, follow the outer -> inner
        // locking order to avoid deadlocks.
        McpClient::call_tool_on_shared(&self.client, server, tool, args)
            .map(|content| serde_json::json!({ "content": content }))
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)) as _)
    }

    fn is_available(&self, server: &str) -> bool {
        match self.client.lock() {
            Ok(g) => g.servers.get_str_ref(server).is_some(),
            Err(e) => {
                eprintln!("[warn] McpClient poisoned in is_available: {}", e);
                false
            }
        }
    }
}

/// In-memory implementation for tests / recording.
pub struct InMemoryMcpPort {
    tools: HashMap<String, Vec<McpToolDef>>,
}

impl InMemoryMcpPort {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }
    pub fn with_tools(mut self, server: impl Into<String>, tools: Vec<McpToolDef>) -> Self {
        self.tools.insert(server.into(), tools);
        self
    }
}

impl Default for InMemoryMcpPort {
    fn default() -> Self {
        Self::new()
    }
}

impl McpPort for InMemoryMcpPort {
    fn tool_definitions(&self) -> Vec<McpToolDef> {
        self.tools.values().flatten().cloned().collect()
    }
    fn call_tool(
        &self,
        server: &str,
        tool: &str,
        _args: serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        if self.is_available(server) {
            Ok(serde_json::json!({"mock_tool": tool, "server": server, "status": "ok"}))
        } else {
            Err(format!("mock MCP: server '{server}' not found").into())
        }
    }
    fn is_available(&self, server: &str) -> bool {
        self.tools.contains_key(server)
    }
}
