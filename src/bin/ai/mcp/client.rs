use std::{
    io::BufReader,
    process::{Command, Stdio},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

#[cfg(unix)]
use std::os::unix::process::CommandExt;
#[cfg(unix)]
use libc;

use serde_json::{Value, json};
use rust_tools::commonw::FastMap;

use crate::ai::types::{
    FunctionDefinition, McpPrompt, McpResource, McpServerConfig, McpTool, ToolDefinition,
};

use super::{
    connection::McpServerConnection,
    jsonrpc::{JsonRpcRequest, JsonRpcResponse},
};

type ServerId = String;

pub(in crate::ai) struct McpClient {
    servers: FastMap<ServerId, Mutex<McpServerConnection>>,
    next_id: AtomicU64,
    cached_tool_definitions: Vec<ToolDefinition>,
    cached_resources: Vec<(String, McpResource)>,
    cached_prompts: Vec<(String, McpPrompt)>,
    cached_server_prefixes: Vec<(String, String)>,
}

impl McpClient {
    pub(in crate::ai) fn new() -> Self {
        Self {
            servers: FastMap::default(),
            next_id: AtomicU64::new(1),
            cached_tool_definitions: Vec::new(),
            cached_resources: Vec::new(),
            cached_prompts: Vec::new(),
            cached_server_prefixes: Vec::new(),
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

        let mut conn = McpServerConnection {
            config: config.clone(),
            process,
            stdin,
            stdout: BufReader::new(stdout),
            stderr: BufReader::new(stderr),
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

        conn.process = process;
        conn.stdin = stdin;
        conn.stdout = BufReader::new(stdout);
        conn.stderr = BufReader::new(stderr);
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

    fn rebuild_metadata_cache(&mut self) {
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

    fn send_request_to_conn(
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

        let response_line = conn.read_response_line()?;

        let response: JsonRpcResponse = serde_json::from_str(&response_line)
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        if response.jsonrpc != "2.0" {
            return Err(format!("Invalid JSON-RPC version: {}", response.jsonrpc));
        }
        if let Some(resp_id) = response.id
            && resp_id != id
        {
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
        let id2 = self.next_request_id();
        if let Err(err) = Self::send_request_to_conn(conn, id2, "notifications/initialized", None) {
            let is_method_not_found = err.contains("-32601")
                && (err.contains("notifications/initialized") || err.contains("initialized"));
            if !is_method_not_found {
                return Err(err);
            }
        }
        Ok(())
    }

    fn list_tools(&self, conn: &mut McpServerConnection) -> Result<Vec<McpTool>, String> {
        let result = self.send_request(conn, "tools/list", None)?;

        let tools = result["tools"]
            .as_array()
            .ok_or("Invalid tools response")?
            .iter()
            .filter_map(|t| serde_json::from_value(t.clone()).ok())
            .collect();

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
            .get(server_name)
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
            Err(err) if Self::is_transport_error(&err) => {
                // Try restart once and retry
                let restart_res = self.restart_connection(&mut conn);
                drop(conn);
                match restart_res {
                    Ok(()) => {
                        let mut conn2 = conn_cell
                            .lock()
                            .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
                        let (id2, params2) = make_params();
                        let result =
                            Self::send_request_to_conn(&mut conn2, id2, "tools/call", Some(params2))?;
                        let content = result["content"]
                            .as_array()
                            .and_then(|arr| arr.first())
                            .and_then(|c| c["text"].as_str())
                            .unwrap_or("")
                            .to_string();
                        return Ok(content);
                    }
                    Err(restart_err) => {
                        return Err(format!(
                            "Transport error: {} | restart failed: {}",
                            err, restart_err
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
