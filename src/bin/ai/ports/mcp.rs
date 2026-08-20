// =============================================================================
// McpPort — MCP 调用端口（解耦 driver 对 mcp crate 的直接依赖）
// =============================================================================
// driver 之前直接通过 `crate::ai::mcp::*` 与 MCP 进程交互，测试必须启动真实
// 进程且中间件无法插桩。抽端口后：
// - 后端：`LiveMcpPort` 为生产适配器，测试可注入 mock / 录制 / 熔断中间件；
// - 上层：driver 通过 `&dyn McpPort` 注入，可替换为 mock / 录制 / 熔断中间件；
// - 兼容：保留 `McpError` 透传，便于调用方统一处理。

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

/// 生产实现：委托给现有 `crate::ai::mcp::McpClient` 的真实 lifecycle。
/// 持有 `SharedMcpClient`（`Arc<Mutex<McpClient>>`）。设计要点：
/// - `tool_definitions` 仅短暂持有外锁 `Arc::clone` 缓存（O(1)），随即释放；
/// - `is_available` 仅短暂持有外锁做一次 `get_str_ref` 查找；
/// - `call_tool` 委托 `McpClient::call_tool_on_shared`，外锁仅用于克隆 `Arc<Mutex<Connection>>`
///   与分配 `next_id`，阻塞的 `writeln/read` 与 `restart_connection` 期间不持有外锁，
///   避免跨 server 的 `tool_definitions`/`is_available` 被长时间阻塞。
/// 该结构是 `DefaultMcpPort` 的正确替代——若按“Default=生产”接入，应使用此类型。
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
        // 短锁：仅克隆 Arc，随后在无锁状态下映射；poison 显式告警而非静默。
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
        // 外锁不跨阻塞 IO：委托 `call_tool_on_shared`，其内部仅短暂持有外锁克隆
        // per-server `Arc<Mutex<Connection>>` 与分配 request id，`send_request_to_conn`
        // 期间仅持有 per-server 内锁；restart 时遵循 外锁->内锁 顺序避免死锁。
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

/// 用于测试/录制的内存实现
pub struct InMemoryMcpPort {
    tools: HashMap<String, Vec<McpToolDef>>,
}

impl InMemoryMcpPort {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }
    pub fn with_tools(mut self, server: impl Into<String>, tools: Vec<McpToolDef>) -> Self {
        self.tools.insert(server.into(), tools);
        self
    }
}

impl Default for InMemoryMcpPort {
    fn default() -> Self { Self::new() }
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
