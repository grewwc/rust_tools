use std::{
    io::BufReader,
    process::{Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

#[cfg(unix)]
use libc;
#[cfg(unix)]
use std::os::unix::process::CommandExt;

use rust_tools::cw::SkipMap;
use serde_json::{Value, json};

use crate::ai::types::{
    FunctionDefinition, McpPrompt, McpResource, McpServerConfig, McpTool, ToolDefinition,
};

fn reject_server_request(
    conn: &mut McpServerConnection,
    id: Value,
    method: &str,
) -> Result<(), String> {
    let payload = unsupported_server_request_payload(id, method);
    writeln!(conn.stdin_mut(), "{}", payload).map_err(|e| {
        conn.decorate_transport_error(format!("Failed to reject server request: {}", e))
    })
}

/// 发送 JSON-RPC 请求到 MCP 服务器连接（独立函数，可在任何上下文中调用）
pub(in crate::ai) fn send_request_to_conn(
    conn: &mut McpServerConnection,
    id: u64,
    method: &str,
    params: Option<Value>,
) -> Result<Value, String> {
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id,
        method: method.to_string(),
        params,
    };

    let request_str = serde_json::to_string(&request)
        .map_err(|e| format!("Failed to serialize request: {}", e))?;

    writeln!(conn.stdin_mut(), "{}", request_str)
        .map_err(|e| conn.decorate_transport_error(format!("Failed to send request: {}", e)))?;

    let deadline = Instant::now() + Duration::from_millis(conn.request_timeout_ms);
    loop {
        let now = Instant::now();
        if now >= deadline {
            return Err(conn.decorate_transport_error(format!(
                "MCP response timeout after {} ms",
                conn.request_timeout_ms
            )));
        }
        let remaining_ms = deadline
            .saturating_duration_since(now)
            .as_millis()
            .clamp(1, u64::MAX as u128) as u64;
        let response_line = conn.read_response_line_with_timeout(remaining_ms)?;
        match classify_inbound_jsonrpc(&response_line)? {
            InboundJsonRpc::Notification { .. } => continue,
            InboundJsonRpc::Request { id: request_id, method } => {
                reject_server_request(conn, request_id, &method)?;
                continue;
            }
            InboundJsonRpc::Response(response) => {
                // 跳过无 id 的响应（id 为 null 或缺失）。
                // 部分 MCP 服务器（如 mcp_ocr）会对 notifications/initialized
                // 通知发送冗余确认响应 {"jsonrpc":"2.0","id":null,"result":{}}。
                // 若不跳过，该响应会被错误地当作下一个请求的响应消费，
                // 导致真正的响应滞留在缓冲区，引发 id mismatch 错误。
                let Some(resp_id) = response.id else {
                    continue;
                };
                if resp_id != id {
                    return Err(format!(
                        "MCP response id mismatch: expected {}, got {}",
                        id, resp_id
                    ));
                }

                if let Some(error) = response.error {
                    if let Some(data) = error.data {
                        return Err(format!(
                            "MCP error {}: {} ({})",
                            error.code, error.message, data
                        ));
                    } else {
                        return Err(format!("MCP error {}: {}", error.code, error.message));
                    }
                }

                return response.result.ok_or("No result in response".to_string());
            }
        }
    }
}

/// 发送 JSON-RPC 通知到 MCP 服务器连接（无 id，不等待响应）。
///
/// MCP 协议中 `notifications/initialized` 等消息是通知而非请求：
/// 不携带 `id` 字段，服务器不应返回响应。若错误地以请求形式发送
/// （带 `id` 并等待响应），部分 MCP 服务器（如 Feishu）会因协议
/// 违规而直接关闭 stdio 流，导致 "MCP server closed the stream
/// unexpectedly" 错误。
pub(in crate::ai) fn send_notification_to_conn(
    conn: &mut McpServerConnection,
    method: &str,
    params: Option<Value>,
) -> Result<(), String> {
    let mut payload = json!({
        "jsonrpc": "2.0",
        "method": method,
    });
    if let Some(p) = params {
        payload["params"] = p;
    }
    let payload_str = serde_json::to_string(&payload)
        .map_err(|e| format!("Failed to serialize notification: {}", e))?;

    writeln!(conn.stdin_mut(), "{}", payload_str).map_err(|e| {
        conn.decorate_transport_error(format!("Failed to send notification: {}", e))
    })?;
    Ok(())
}

use super::{
    connection::{McpServerConnection, spawn_stderr_drain},
    jsonrpc::{
        InboundJsonRpc, JsonRpcRequest, classify_inbound_jsonrpc,
        unsupported_server_request_payload,
    },
};

pub(in crate::ai) type SharedMcpClient = Arc<std::sync::Mutex<McpClient>>;

type ServerId = String;

pub(in crate::ai) struct McpClient {
    pub(in crate::ai) servers: SkipMap<ServerId, Mutex<McpServerConnection>>,
    next_id: AtomicU64,
    cached_tool_definitions: Vec<ToolDefinition>,
    cached_resources: Vec<(String, McpResource)>,
    cached_prompts: Vec<(String, McpPrompt)>,
    cached_server_prefixes: Vec<(String, String)>,
}

impl McpClient {
    pub(in crate::ai) fn new() -> Self {
        Self {
            servers: SkipMap::default(),
            next_id: AtomicU64::new(1),
            cached_tool_definitions: Vec::new(),
            cached_resources: Vec::new(),
            cached_prompts: Vec::new(),
            cached_server_prefixes: Vec::new(),
        }
    }

    pub(in crate::ai) fn routing_snapshot(&self) -> Self {
        Self {
            servers: SkipMap::default(),
            next_id: AtomicU64::new(self.next_id.load(Ordering::Relaxed)),
            cached_tool_definitions: self.cached_tool_definitions.clone(),
            cached_resources: self.cached_resources.clone(),
            cached_prompts: self.cached_prompts.clone(),
            cached_server_prefixes: self.cached_server_prefixes.clone(),
        }
    }

    pub(in crate::ai) fn connect_server(
        &mut self,
        name: &str,
        config: &McpServerConfig,
    ) -> Result<(), String> {
        if config.disabled {
            return Ok(());
        }

        let mut cmd = Command::new(&config.command);
        cmd.args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        unsafe {
            let _ = cmd.pre_exec(|| {
                {
                    libc::signal(libc::SIGINT, libc::SIG_IGN);
                    libc::signal(libc::SIGPIPE, libc::SIG_IGN);
                }
                Ok(())
            });
        }

        for (key, value) in &config.env {
            cmd.env(key, value);
        }

        let mut process = cmd
            .spawn()
            .map_err(|e| format!("Failed to start MCP server '{}': {}", name, e))?;

        let stdin = process.stdin.take().ok_or("Failed to get stdin")?;
        let stdout = process.stdout.take().ok_or("Failed to get stdout")?;
        let stderr = process.stderr.take().ok_or("Failed to get stderr")?;

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            super::io::set_fd_nonblocking(stdout.as_raw_fd())
                .map_err(|e| format!("Failed to set nonblocking mode: {}", e))?;
        }
        let stderr_tail = spawn_stderr_drain(stderr);

        let mut conn = McpServerConnection {
            config: config.clone(),
            process,
            stdin,
            stdout: BufReader::new(stdout),
            stderr_tail,
            request_timeout_ms: config.request_timeout_ms.max(100),
            tools: Vec::new(),
            resources: Vec::new(),
            prompts: Vec::new(),
        };

        self.initialize_server(&mut conn)?;
        conn.tools = self.list_tools(&mut conn)?;
        conn.resources = self.list_resources(&mut conn)?;
        conn.prompts = self.list_prompts(&mut conn)?;

        self.servers.insert(name.to_string(), Mutex::new(conn));
        self.rebuild_metadata_cache();
        Ok(())
    }

    fn is_transport_error(err: &str) -> bool {
        let e = err.to_lowercase();
        e.contains("broken pipe")
            || e.contains("closed the stream unexpectedly")
            || e.contains("mcp response timeout")
            || e.contains("failed waiting for mcp response")
            || e.contains("failed to read response")
            || e.contains("process exited with status")
            || e.contains("failed to get stdin")
            || e.contains("failed to get stdout")
    }

    fn is_protocol_desync_error(err: &str) -> bool {
        let e = err.to_lowercase();
        e.contains("mcp response id mismatch")
            || e.contains("failed to parse response")
            || e.contains("invalid json-rpc version")
    }

    fn is_user_interrupt_error(err: &str) -> bool {
        let e = err.to_lowercase();
        e.contains("canceled by user")
            || e.contains("cancelled by user")
            || e.contains("interrupted by user")
    }

    fn restart_connection(&self, conn: &mut McpServerConnection) -> Result<(), String> {
        let cfg = conn.config.clone();

        let _ = conn.process.kill();

        let mut cmd = Command::new(&cfg.command);
        cmd.args(&cfg.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        unsafe {
            let _ = cmd.pre_exec(|| {
                {
                    libc::signal(libc::SIGINT, libc::SIG_IGN);
                    libc::signal(libc::SIGPIPE, libc::SIG_IGN);
                }
                Ok(())
            });
        }
        for (key, value) in &cfg.env {
            cmd.env(key, value);
        }

        let mut process = cmd
            .spawn()
            .map_err(|e| format!("Failed to restart MCP server: {}", e))?;
        let stdin = process.stdin.take().ok_or("Failed to get stdin")?;
        let stdout = process.stdout.take().ok_or("Failed to get stdout")?;
        let stderr = process.stderr.take().ok_or("Failed to get stderr")?;

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            super::io::set_fd_nonblocking(stdout.as_raw_fd())
                .map_err(|e| format!("Failed to set nonblocking mode: {}", e))?;
        }
        let stderr_tail = spawn_stderr_drain(stderr);

        conn.process = process;
        conn.stdin = stdin;
        conn.stdout = BufReader::new(stdout);
        conn.stderr_tail = stderr_tail;
        conn.request_timeout_ms = cfg.request_timeout_ms.max(100);
        conn.tools.clear();
        conn.resources.clear();
        conn.prompts.clear();

        self.initialize_server(conn)?;
        conn.tools = self.list_tools(conn)?;
        conn.resources = self.list_resources(conn)?;
        conn.prompts = self.list_prompts(conn)?;
        Ok(())
    }

    pub(in crate::ai) fn reset_server(&self, server_name: &str) -> Result<(), String> {
        let conn_cell = self
            .servers
            .get_str_ref(server_name)
            .ok_or_else(|| format!("Server not found: {}", server_name))?;
        let mut conn = conn_cell
            .lock()
            .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
        self.restart_connection(&mut conn)
    }

    pub(in crate::ai) fn rebuild_metadata_cache(&mut self) {
        let mut tool_definitions = Vec::new();
        let mut resources = Vec::new();
        let mut prompts = Vec::new();
        let mut server_prefixes = Vec::with_capacity(self.servers.len());

        for (server_name, conn) in &self.servers {
            server_prefixes.push((server_name.clone(), format!("mcp_{server_name}_")));

            let Ok(conn) = conn.lock() else {
                continue;
            };

            tool_definitions.reserve(conn.tools.len());
            for tool in &conn.tools {
                tool_definitions.push(ToolDefinition {
                    tool_type: "function".to_string(),
                    function: FunctionDefinition {
                        name: format!("mcp_{}_{}", server_name, tool.name),
                        description: tool.description.clone().unwrap_or_default(),
                        parameters: tool.input_schema.clone(),
                    },
                });
            }

            resources.reserve(conn.resources.len());
            resources.extend(
                conn.resources
                    .iter()
                    .cloned()
                    .map(|resource| (server_name.clone(), resource)),
            );

            prompts.reserve(conn.prompts.len());
            prompts.extend(
                conn.prompts
                    .iter()
                    .cloned()
                    .map(|prompt| (server_name.clone(), prompt)),
            );
        }

        rust_tools::sortw::stable_sort_by(&mut server_prefixes, |a, b| b.1.len().cmp(&a.1.len()));
        self.cached_tool_definitions = tool_definitions;
        self.cached_resources = resources;
        self.cached_prompts = prompts;
        self.cached_server_prefixes = server_prefixes;
    }

    fn next_request_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    pub(in crate::ai) fn send_request_to_conn(
        conn: &mut McpServerConnection,
        id: u64,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, String> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        };

        let request_str = serde_json::to_string(&request)
            .map_err(|e| format!("Failed to serialize request: {}", e))?;

        writeln!(conn.stdin_mut(), "{}", request_str)
            .map_err(|e| conn.decorate_transport_error(format!("Failed to send request: {}", e)))?;

        let deadline = Instant::now() + Duration::from_millis(conn.request_timeout_ms);
        let response = loop {
            let now = Instant::now();
            if now >= deadline {
                return Err(conn.decorate_transport_error(format!(
                    "MCP response timeout after {} ms",
                    conn.request_timeout_ms
                )));
            }
            let remaining_ms = deadline
                .saturating_duration_since(now)
                .as_millis()
                .clamp(1, u64::MAX as u128) as u64;
            let response_line = conn.read_response_line_with_timeout(remaining_ms)?;
            match classify_inbound_jsonrpc(&response_line)? {
                InboundJsonRpc::Notification { .. } => continue,
                InboundJsonRpc::Request { id: request_id, method } => {
                    reject_server_request(conn, request_id, &method)?;
                    continue;
                }
                InboundJsonRpc::Response(resp) => {
                    // 跳过无 id 的响应（id 为 null 或缺失）
                    let Some(resp_id) = resp.id else { continue; };
                    if resp_id != id {
                        return Err(format!(
                            "MCP response id mismatch: expected {}, got {}",
                            id, resp_id
                        ));
                    }
                    break resp;
                }
            }
        };

        if let Some(error) = response.error {
            if let Some(data) = error.data {
                return Err(format!(
                    "MCP error {}: {} ({})",
                    error.code, error.message, data
                ));
            }
            return Err(format!("MCP error {}: {}", error.code, error.message));
        }

        response.result.ok_or("No result in response".to_string())
    }

    fn send_request(
        &self,
        conn: &mut McpServerConnection,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, String> {
        let id = self.next_request_id();
        Self::send_request_to_conn(conn, id, method, params)
    }

    fn initialize_server(&self, conn: &mut McpServerConnection) -> Result<(), String> {
        let params = json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {}
            },
            "clientInfo": {
                "name": "rust-tools-ai",
                "version": "1.0.0"
            }
        });

        let id1 = self.next_request_id();
        Self::send_request_to_conn(conn, id1, "initialize", Some(params))?;
        // notifications/initialized 是通知（无 id，不等待响应），不能用请求形式发送
        send_notification_to_conn(conn, "notifications/initialized", None)?;
        Ok(())
    }

    fn list_tools(&self, conn: &mut McpServerConnection) -> Result<Vec<McpTool>, String> {
        let result = match self.send_request(conn, "tools/list", None) {
            Ok(v) => v,
            Err(err) => {
                let is_method_not_found = err.contains("-32601") && err.contains("tools/list");
                if is_method_not_found {
                    return Ok(Vec::new());
                }
                return Err(err);
            }
        };

        let tools = result["tools"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| serde_json::from_value(t.clone()).ok())
                    .collect()
            })
            .unwrap_or_default();

        Ok(tools)
    }

    fn list_resources(&self, conn: &mut McpServerConnection) -> Result<Vec<McpResource>, String> {
        let result = match self.send_request(conn, "resources/list", None) {
            Ok(v) => v,
            Err(err) => {
                let is_method_not_found = err.contains("-32601") && err.contains("resources/list");
                if is_method_not_found {
                    return Ok(Vec::new());
                }
                return Err(err);
            }
        };

        let resources = result["resources"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|r| serde_json::from_value(r.clone()).ok())
                    .collect()
            })
            .unwrap_or_default();

        Ok(resources)
    }

    fn list_prompts(&self, conn: &mut McpServerConnection) -> Result<Vec<McpPrompt>, String> {
        let result = match self.send_request(conn, "prompts/list", None) {
            Ok(v) => v,
            Err(err) => {
                let is_method_not_found = err.contains("-32601") && err.contains("prompts/list");
                if is_method_not_found {
                    return Ok(Vec::new());
                }
                return Err(err);
            }
        };

        let prompts = result["prompts"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|p| serde_json::from_value(p.clone()).ok())
                    .collect()
            })
            .unwrap_or_default();

        Ok(prompts)
    }

    pub(in crate::ai) fn call_tool(
        &self,
        server_name: &str,
        tool_name: &str,
        arguments: Value,
    ) -> Result<String, String> {
        let make_params = || {
            let id = self.next_request_id();
            (id, json!({ "name": tool_name, "arguments": arguments }))
        };

        let conn_cell = self
            .servers
            .get_str_ref(server_name)
            .ok_or_else(|| format!("Server not found: {}", server_name))?;

        // First attempt
        let mut conn = conn_cell
            .lock()
            .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
        let (id1, params1) = make_params();
        let first = Self::send_request_to_conn(&mut conn, id1, "tools/call", Some(params1));
        match first {
            Ok(result) => {
                let content = result["content"]
                    .as_array()
                    .and_then(|arr| arr.first())
                    .and_then(|c| c["text"].as_str())
                    .unwrap_or("")
                    .to_string();
                return Ok(content);
            }
            Err(err) if Self::is_user_interrupt_error(&err) => {
                let restart_res = self.restart_connection(&mut conn);
                drop(conn);
                return match restart_res {
                    Ok(()) => Err(format!(
                        "MCP tool call canceled by user (Ctrl+C): {}",
                        tool_name
                    )),
                    Err(restart_err) => Err(format!(
                        "MCP tool call canceled by user (Ctrl+C): {} | restart failed: {}",
                        tool_name, restart_err
                    )),
                };
            }
            Err(err) if Self::is_transport_error(&err) || Self::is_protocol_desync_error(&err) => {
                // Try restart once and retry
                let restart_res = self.restart_connection(&mut conn);
                drop(conn);
                match restart_res {
                    Ok(()) => {
                        let mut conn2 = conn_cell
                            .lock()
                            .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
                        let (id2, params2) = make_params();
                        let result = Self::send_request_to_conn(
                            &mut conn2,
                            id2,
                            "tools/call",
                            Some(params2),
                        )?;
                        let content = result["content"]
                            .as_array()
                            .and_then(|arr| arr.first())
                            .and_then(|c| c["text"].as_str())
                            .unwrap_or("")
                            .to_string();
                        return Ok(content);
                    }
                    Err(restart_err) => {
                        let kind = if Self::is_protocol_desync_error(&err) {
                            "Protocol desync"
                        } else {
                            "Transport error"
                        };
                        return Err(format!(
                            "{}: {} | restart failed: {}",
                            kind, err, restart_err
                        ));
                    }
                }
            }
            Err(err) => return Err(err),
        }

        // unreachable
    }

    pub(in crate::ai) fn get_all_tools(&self) -> Vec<ToolDefinition> {
        self.cached_tool_definitions.clone()
    }

    pub(in crate::ai) fn parse_tool_name_for_known_server(
        &self,
        full_name: &str,
    ) -> Option<(String, String)> {
        if !full_name.starts_with("mcp_") {
            return None;
        }

        for (server_name, prefix) in &self.cached_server_prefixes {
            if let Some(tool_name) = full_name.strip_prefix(prefix)
                && !tool_name.is_empty()
            {
                return Some((server_name.clone(), tool_name.to_string()));
            }
        }

        None
    }

    pub(in crate::ai) fn get_all_resources(&self) -> Vec<(String, McpResource)> {
        self.cached_resources.clone()
    }

    pub(in crate::ai) fn get_all_prompts(&self) -> Vec<(String, McpPrompt)> {
        self.cached_prompts.clone()
    }

    pub(in crate::ai) fn disconnect_all(&mut self) {
        for (_, conn) in self.servers.drain() {
            let conn = conn.into_inner().unwrap_or_else(|e| e.into_inner());
            let mut process = conn.process;
            let _ = process.kill();
        }
        self.cached_tool_definitions.clear();
        self.cached_resources.clear();
        self.cached_prompts.clear();
        self.cached_server_prefixes.clear();
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        self.disconnect_all();
    }
}
