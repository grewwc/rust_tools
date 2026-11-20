// =============================================================================
// McpPort - MCP discovery and tool execution port
// =============================================================================
// Production and in-memory adapters share discovery, call, and availability
// operations so tests can supply tool catalogs without spawning servers.

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
/// - `call_tool` delegates to `McpClient::call_tool_on_shared`, which releases the
///   outer lock before tool-request I/O. Restart and discovery hold the outer
///   lock and then the per-server lock, so they can block other metadata lookups.
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
        // Tool-request I/O holds only the per-server lock. Restart paths acquire
        // the outer lock first to preserve the outer -> inner lock order.
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

/// In-memory tool catalog with synthetic call results for tests.
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
