use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[cfg(unix)]
use std::os::unix::io::{AsRawFd, FromRawFd};

use super::types::{
    FunctionDefinition, McpPrompt, McpResource, McpServerConfig, McpTool, ToolDefinition,
};

type ServerId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(default)]
    id: Option<u64>,
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
    #[serde(default)]
    data: Option<Value>,
}

pub(super) struct McpClient {
    servers: HashMap<ServerId, Mutex<McpServerConnection>>,
    next_id: AtomicU64,
}

struct McpServerConnection {
    transport: McpTransport,
    request_timeout_ms: u64,
    tools: Vec<McpTool>,
    resources: Vec<McpResource>,
    prompts: Vec<McpPrompt>,
}

enum McpTransport {
    Process {
        process: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
    },
    InProcess {
        stdin: File,
        stdout: BufReader<File>,
        join: Option<JoinHandle<()>>,
    },
}

impl McpServerConnection {
    fn stdin_mut(&mut self) -> &mut dyn Write {
        match &mut self.transport {
            McpTransport::Process { stdin, .. } => stdin,
            McpTransport::InProcess { stdin, .. } => stdin,
        }
    }

    fn stdout_mut(&mut self) -> &mut dyn BufRead {
        match &mut self.transport {
            McpTransport::Process { stdout, .. } => stdout,
            McpTransport::InProcess { stdout, .. } => stdout,
        }
    }

    fn read_response_line(&mut self) -> Result<String, String> {
        match &mut self.transport {
            McpTransport::Process { stdout, .. } => {
                let mut response_line = String::new();
                read_line_with_timeout_process(
                    stdout,
                    self.request_timeout_ms,
                    &mut response_line,
                )?;
                Ok(response_line)
            }
            McpTransport::InProcess { stdout, .. } => {
                let mut response_line = String::new();
                read_line_with_timeout_file(stdout, self.request_timeout_ms, &mut response_line)?;
                Ok(response_line)
            }
        }
    }
}

#[cfg(unix)]
fn wait_fd_readable(fd: i32, timeout_ms: u64) -> Result<(), String> {
    let timeout = if timeout_ms > i32::MAX as u64 {
        i32::MAX
    } else {
        timeout_ms as i32
    };

    loop {
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let rc = unsafe { libc::poll(&mut pfd, 1, timeout) };
        if rc > 0 {
            return Ok(());
        }
        if rc == 0 {
            return Err(format!("MCP response timeout after {} ms", timeout_ms));
        }

        let err = std::io::Error::last_os_error();
        if err.kind() == std::io::ErrorKind::Interrupted {
            continue;
        }
        return Err(format!("Failed waiting for MCP response: {}", err));
    }
}

#[cfg(not(unix))]
fn wait_fd_readable(_fd: i32, _timeout_ms: u64) -> Result<(), String> {
    Ok(())
}

fn read_line_with_timeout_process(
    stdout: &mut BufReader<ChildStdout>,
    timeout_ms: u64,
    response_line: &mut String,
) -> Result<(), String> {
    let line = read_line_with_timeout_buf(stdout, stdout.get_ref().as_raw_fd(), timeout_ms)?;
    response_line.push_str(&line);
    Ok(())
}

fn read_line_with_timeout_file(
    stdout: &mut BufReader<File>,
    timeout_ms: u64,
    response_line: &mut String,
) -> Result<(), String> {
    let line = read_line_with_timeout_buf(stdout, stdout.get_ref().as_raw_fd(), timeout_ms)?;
    response_line.push_str(&line);
    Ok(())
}

fn read_line_with_timeout_buf<R: std::io::Read>(
    stdout: &mut BufReader<R>,
    fd: i32,
    timeout_ms: u64,
) -> Result<String, String> {
    let timeout = Duration::from_millis(timeout_ms);
    let deadline = Instant::now() + timeout;
    let mut buf = Vec::<u8>::new();

    loop {
        let now = Instant::now();
        if now >= deadline {
            return Err(format!("MCP response timeout after {} ms", timeout_ms));
        }
        let remaining = deadline.saturating_duration_since(now).as_millis().max(1) as u64;

        #[cfg(unix)]
        {
            wait_fd_readable(fd, remaining)?;
        }

        let available = stdout
            .fill_buf()
            .map_err(|e| format!("Failed to read response: {}", e))?;
        if available.is_empty() {
            return Err("MCP server closed the stream unexpectedly".to_string());
        }

        if let Some(pos) = available.iter().position(|b| *b == b'\n') {
            buf.extend_from_slice(&available[..=pos]);
            stdout.consume(pos + 1);
            let line = String::from_utf8(buf)
                .map_err(|e| format!("MCP response is not valid UTF-8: {}", e))?;
            return Ok(line);
        }

        buf.extend_from_slice(available);
        let consumed = available.len();
        stdout.consume(consumed);
    }
}

fn create_pipe_pair() -> Result<(File, File), String> {
    #[cfg(unix)]
    {
        let mut fds = [0i32; 2];
        let rc = unsafe { libc::pipe(fds.as_mut_ptr()) };
        if rc != 0 {
            return Err(format!("pipe failed: {}", std::io::Error::last_os_error()));
        }
        let read = unsafe { File::from_raw_fd(fds[0]) };
        let write = unsafe { File::from_raw_fd(fds[1]) };
        Ok((read, write))
    }
    #[cfg(not(unix))]
    {
        Err("inprocess mcp is only supported on unix".to_string())
    }
}

impl McpClient {
    pub(super) fn new() -> Self {
        Self {
            servers: HashMap::new(),
            next_id: AtomicU64::new(1),
        }
    }

    pub(super) fn connect_server(
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
            .stderr(Stdio::null());

        for (key, value) in &config.env {
            cmd.env(key, value);
        }

        let mut process = cmd
            .spawn()
            .map_err(|e| format!("Failed to start MCP server '{}': {}", name, e))?;

        let stdin = process.stdin.take().ok_or("Failed to get stdin")?;
        let stdout = process.stdout.take().ok_or("Failed to get stdout")?;

        let mut conn = McpServerConnection {
            transport: McpTransport::Process {
                process,
                stdin,
                stdout: BufReader::new(stdout),
            },
            request_timeout_ms: config.request_timeout_ms.max(100),
            tools: Vec::new(),
            resources: Vec::new(),
            prompts: Vec::new(),
        };

        self.initialize_server(&mut conn)?;
        conn.tools = self.list_tools(&mut conn)?;

        self.servers.insert(name.to_string(), Mutex::new(conn));
        Ok(())
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
            .map_err(|e| format!("Failed to send request: {}", e))?;

        let response_line = conn.read_response_line()?;

        let response: JsonRpcResponse = serde_json::from_str(&response_line)
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        if let Some(error) = response.error {
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
        let result = self.send_request(conn, "resources/list", None)?;

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
        let result = self.send_request(conn, "prompts/list", None)?;

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

    pub(super) fn call_tool(
        &self,
        server_name: &str,
        tool_name: &str,
        arguments: Value,
    ) -> Result<String, String> {
        let id = self.next_request_id();
        let params = json!({
            "name": tool_name,
            "arguments": arguments
        });

        let conn = self
            .servers
            .get(server_name)
            .ok_or_else(|| format!("Server not found: {}", server_name))?;

        let mut conn = conn
            .lock()
            .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
        let result = Self::send_request_to_conn(&mut conn, id, "tools/call", Some(params))?;

        let content = result["content"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|c| c["text"].as_str())
            .unwrap_or("")
            .to_string();

        Ok(content)
    }

    pub(super) fn read_resource(&self, server_name: &str, uri: &str) -> Result<String, String> {
        let id = self.next_request_id();
        let params = json!({
            "uri": uri
        });

        let conn = self
            .servers
            .get(server_name)
            .ok_or_else(|| format!("Server not found: {}", server_name))?;

        let mut conn = conn
            .lock()
            .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
        let result = Self::send_request_to_conn(&mut conn, id, "resources/read", Some(params))?;

        let content = result["contents"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|c| c["text"].as_str())
            .unwrap_or("")
            .to_string();

        Ok(content)
    }

    pub(super) fn get_prompt(
        &self,
        server_name: &str,
        prompt_name: &str,
        arguments: HashMap<String, String>,
    ) -> Result<String, String> {
        let id = self.next_request_id();
        let params = json!({
            "name": prompt_name,
            "arguments": arguments
        });

        let conn = self
            .servers
            .get(server_name)
            .ok_or_else(|| format!("Server not found: {}", server_name))?;

        let mut conn = conn
            .lock()
            .map_err(|_| format!("Server connection poisoned: {}", server_name))?;
        let result = Self::send_request_to_conn(&mut conn, id, "prompts/get", Some(params))?;

        let content = result["messages"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|m| m["content"]["text"].as_str())
            .unwrap_or("")
            .to_string();

        Ok(content)
    }

    pub(super) fn get_all_tools(&self) -> Vec<ToolDefinition> {
        let mut result = Vec::new();
        for (server_name, conn) in &self.servers {
            let Ok(conn) = conn.lock() else {
                continue;
            };
            for tool in &conn.tools {
                result.push(ToolDefinition {
                    tool_type: "function".to_string(),
                    function: FunctionDefinition {
                        name: format!("mcp_{}_{}", server_name, tool.name),
                        description: tool.description.clone().unwrap_or_default(),
                        parameters: tool.input_schema.clone(),
                    },
                });
            }
        }
        result
    }

    pub(super) fn parse_tool_name_for_known_server(
        &self,
        full_name: &str,
    ) -> Option<(String, String)> {
        if !full_name.starts_with("mcp_") {
            return None;
        }

        let mut names: Vec<&String> = self.servers.keys().collect();
        names.sort_by_key(|n| std::cmp::Reverse(n.len()));

        for name in names {
            let prefix = format!("mcp_{}_", name);
            if let Some(tool_name) = full_name.strip_prefix(&prefix)
                && !tool_name.is_empty()
            {
                return Some((name.clone(), tool_name.to_string()));
            }
        }

        None
    }

    pub(super) fn get_all_resources(&self) -> Vec<(String, McpResource)> {
        let mut result = Vec::new();
        for (server_name, conn) in &self.servers {
            let Ok(conn) = conn.lock() else {
                continue;
            };
            for r in &conn.resources {
                result.push((server_name.clone(), r.clone()));
            }
        }
        result
    }

    pub(super) fn get_all_prompts(&self) -> Vec<(String, McpPrompt)> {
        let mut result = Vec::new();
        for (server_name, conn) in &self.servers {
            let Ok(conn) = conn.lock() else {
                continue;
            };
            for p in &conn.prompts {
                result.push((server_name.clone(), p.clone()));
            }
        }
        result
    }

    pub(super) fn disconnect_all(&mut self) {
        for (_, conn) in self.servers.drain() {
            let conn = conn.into_inner().unwrap_or_else(|e| e.into_inner());
            match conn.transport {
                McpTransport::Process { mut process, .. } => {
                    let _ = process.kill();
                }
                McpTransport::InProcess {
                    stdin,
                    stdout: _,
                    join,
                } => {
                    drop(stdin);
                    if let Some(handle) = join {
                        let _ = handle.join();
                    }
                }
            }
        }
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        self.disconnect_all();
    }
}

pub(super) fn parse_mcp_tool_name(full_name: &str) -> Option<(String, String)> {
    let prefix = "mcp_";
    if !full_name.starts_with(prefix) {
        return None;
    }

    let rest = &full_name[prefix.len()..];
    let parts: Vec<&str> = rest.splitn(2, '_').collect();
    if parts.len() != 2 {
        return None;
    }

    Some((parts[0].to_string(), parts[1].to_string()))
}

pub(super) fn load_mcp_config_from_file(
    path: &str,
) -> Result<HashMap<String, McpServerConfig>, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read MCP config: {}", e))?;

    let config: Value =
        serde_json::from_str(&content).map_err(|e| format!("Failed to parse MCP config: {}", e))?;

    let servers = config["mcpServers"]
        .as_object()
        .ok_or("Invalid mcpServers in config")?;

    let mut result = HashMap::new();
    for (name, value) in servers {
        let server_config: McpServerConfig = serde_json::from_value(value.clone())
            .map_err(|e| format!("Invalid server config for '{}': {}", name, e))?;
        result.insert(name.clone(), server_config);
    }

    Ok(result)
}
