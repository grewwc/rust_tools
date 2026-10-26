use std::{
    io::{BufReader, Read, Write},
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use serde_json::{Value, json};

use crate::ai::types::{McpPrompt, McpResource, McpServerConfig, McpTool, McpTransportKind};

use super::{
    http::HttpTransport,
    io::{read_line_with_timeout_process, set_fd_nonblocking},
    jsonrpc::{
        InboundJsonRpc, JsonRpcRequest, classify_inbound_jsonrpc,
        unsupported_server_request_payload,
    },
};

const STDERR_TAIL_LIMIT: usize = 16 * 1024;
const STDERR_SNIPPET_LIMIT: usize = 400;
const STDIO_PROTOCOL_VERSION: &str = "2024-11-05";

/// A connected MCP server. Only one transport is active for the lifetime of a
/// connection; restart replaces it while keeping the same `Arc<Mutex<...>>`
/// identity used by the shared client.
pub(in crate::ai) struct McpServerConnection {
    pub(in crate::ai) config: McpServerConfig,
    name: String,
    transport: McpTransport,
    pub(in crate::ai) request_timeout_ms: u64,
    pub(in crate::ai) tools: Vec<McpTool>,
    pub(in crate::ai) resources: Vec<McpResource>,
    pub(in crate::ai) prompts: Vec<McpPrompt>,
}

enum McpTransport {
    Stdio(StdioTransport),
    Http(HttpTransport),
}

struct StdioTransport {
    process: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr_tail: Arc<Mutex<String>>,
}

pub(in crate::ai) fn spawn_stderr_drain(stderr: ChildStderr) -> Arc<Mutex<String>> {
    let stderr_tail = Arc::new(Mutex::new(String::new()));
    let sink = stderr_tail.clone();
    thread::spawn(move || drain_stderr_into_tail(stderr, sink));
    stderr_tail
}

impl McpServerConnection {
    /// Build a connection for `config` without initializing the MCP session.
    /// The caller sends `initialize` and the discovery requests afterwards;
    /// keeping those out of the constructor lets a restart reuse the same
    /// initialize/list sequence without duplicating transport code.
    pub(in crate::ai) fn connect(name: &str, config: &McpServerConfig) -> Result<Self, String> {
        if config.disabled {
            return Err("Server is disabled".to_string());
        }
        let transport = build_transport(name, config)?;
        Ok(Self {
            config: config.clone(),
            name: name.to_string(),
            transport,
            request_timeout_ms: config.request_timeout_ms.max(100),
            tools: Vec::new(),
            resources: Vec::new(),
            prompts: Vec::new(),
        })
    }

    pub(in crate::ai) fn protocol_version(&self) -> &'static str {
        match &self.transport {
            McpTransport::Stdio(_) => STDIO_PROTOCOL_VERSION,
            McpTransport::Http(http) => http.protocol_version(),
        }
    }

    pub(in crate::ai) fn request(
        &mut self,
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
        let timeout_ms = self.request_timeout_ms;

        let result = match &mut self.transport {
            McpTransport::Stdio(stdio) => request_stdio(stdio, id, &request_str, timeout_ms),
            McpTransport::Http(http) => http.request(id, &request_str, timeout_ms),
        };
        result.map_err(|e| self.decorate_transport_error(e))
    }

    pub(in crate::ai) fn notify(
        &mut self,
        method: &str,
        params: Option<Value>,
    ) -> Result<(), String> {
        let mut payload = json!({
            "jsonrpc": "2.0",
            "method": method,
        });
        if let Some(params) = params {
            payload["params"] = params;
        }
        let payload_str = serde_json::to_string(&payload)
            .map_err(|e| format!("Failed to serialize notification: {}", e))?;
        let timeout_ms = self.request_timeout_ms;

        let result = match &mut self.transport {
            McpTransport::Stdio(stdio) => write_stdio(stdio, &payload_str),
            McpTransport::Http(http) => http.notify(&payload_str, timeout_ms),
        };
        result.map_err(|e| self.decorate_transport_error(e))
    }

    /// Replace the live transport after a timeout/desync. The seed metadata is
    /// cleared so a failed re-initialization cannot leave stale tools visible.
    pub(in crate::ai) fn restart(&mut self) -> Result<(), String> {
        self.shutdown_and_reap();
        self.transport = build_transport(&self.name, &self.config)?;
        self.request_timeout_ms = self.config.request_timeout_ms.max(100);
        self.tools.clear();
        self.resources.clear();
        self.prompts.clear();
        Ok(())
    }

    /// Best-effort shutdown. For stdio this terminates and reaps the child; for
    /// HTTP it stops the legacy SSE reader and asks the server to drop the
    /// session. Drop also calls this so a failed initialize does not leak.
    pub(in crate::ai) fn shutdown_and_reap(&mut self) {
        match &mut self.transport {
            McpTransport::Stdio(stdio) => stdio.shutdown_and_reap(),
            McpTransport::Http(http) => http.shutdown(),
        }
    }

    fn decorate_transport_error(&mut self, err: String) -> String {
        match &mut self.transport {
            McpTransport::Stdio(stdio) => stdio.decorate_transport_error(err),
            McpTransport::Http(_) => err,
        }
    }
}

impl Drop for McpServerConnection {
    fn drop(&mut self) {
        self.shutdown_and_reap();
    }
}

fn build_transport(name: &str, config: &McpServerConfig) -> Result<McpTransport, String> {
    match config.transport_kind()? {
        McpTransportKind::Stdio => Ok(McpTransport::Stdio(StdioTransport::spawn(name, config)?)),
        McpTransportKind::Http | McpTransportKind::Sse => {
            Ok(McpTransport::Http(HttpTransport::connect(config)?))
        }
    }
}

impl StdioTransport {
    fn spawn(name: &str, config: &McpServerConfig) -> Result<Self, String> {
        let mut cmd = Command::new(&config.command);
        cmd.args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        unsafe {
            use std::os::unix::process::CommandExt;
            let _ = cmd.pre_exec(|| {
                libc::signal(libc::SIGINT, libc::SIG_IGN);
                libc::signal(libc::SIGPIPE, libc::SIG_IGN);
                Ok(())
            });
        }

        for (key, value) in &config.env {
            cmd.env(key, value);
        }

        let mut process = crate::fork_guard::spawn(&mut cmd)
            .map_err(|e| format!("Failed to start MCP server '{}': {}", name, e))?;
        let stdin = process.stdin.take().ok_or("Failed to get stdin")?;
        let stdout = process.stdout.take().ok_or("Failed to get stdout")?;
        let stderr = process.stderr.take().ok_or("Failed to get stderr")?;
        let stderr_tail = spawn_stderr_drain(stderr);

        // Keep stdout non-blocking so the line reader can check cancellation
        // and the request deadline without blocking forever on an idle pipe.
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            set_fd_nonblocking(stdout.as_raw_fd())
                .map_err(|e| format!("Failed to set nonblocking mode: {}", e))?;
        }

        Ok(Self {
            process,
            stdin,
            stdout: BufReader::new(stdout),
            stderr_tail,
        })
    }

    fn write(&mut self, payload: &str) -> Result<(), String> {
        writeln!(self.stdin, "{}", payload).map_err(|e| format!("Failed to send request: {}", e))
    }

    fn read_response_line_with_timeout(&mut self, timeout_ms: u64) -> Result<String, String> {
        let mut response_line = String::new();
        read_line_with_timeout_process(&mut self.stdout, timeout_ms, &mut response_line)?;
        Ok(response_line)
    }

    fn reject_server_request(&mut self, id: Value, method: &str) -> Result<(), String> {
        let payload = unsupported_server_request_payload(id, method);
        let payload_str = serde_json::to_string(&payload)
            .map_err(|e| format!("Failed to serialize server-request rejection: {}", e))?;
        writeln!(self.stdin, "{}", payload_str)
            .map_err(|e| format!("Failed to reject server request: {}", e))
    }

    fn shutdown_and_reap(&mut self) {
        match self.process.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) | Err(_) => {}
        }
        let _ = self.process.kill();
        let _ = self.process.wait();
    }

    fn decorate_transport_error(&mut self, err: String) -> String {
        let mut detail = err;
        if let Ok(Some(status)) = self.process.try_wait() {
            detail.push_str(&format!(" | process exited with status {}", status));
        }
        let stderr = self.read_stderr_snippet();
        if !stderr.is_empty() {
            detail.push_str(" | stderr: ");
            detail.push_str(&stderr);
        }
        detail
    }

    fn read_stderr_snippet(&self) -> String {
        let text = self
            .stderr_tail
            .lock()
            .map(|tail| tail.trim().to_string())
            .unwrap_or_default();
        if text.chars().count() > STDERR_SNIPPET_LIMIT {
            let truncated = text.chars().take(STDERR_SNIPPET_LIMIT).collect::<String>();
            format!("{}...", truncated)
        } else {
            text
        }
    }
}

fn write_stdio(stdio: &mut StdioTransport, payload: &str) -> Result<(), String> {
    stdio.write(payload)
}

fn request_stdio(
    stdio: &mut StdioTransport,
    id: u64,
    request_str: &str,
    timeout_ms: u64,
) -> Result<Value, String> {
    stdio.write(request_str)?;
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    loop {
        let now = Instant::now();
        if now >= deadline {
            return Err(format!("MCP response timeout after {} ms", timeout_ms));
        }
        let remaining_ms = deadline
            .saturating_duration_since(now)
            .as_millis()
            .clamp(1, u64::MAX as u128) as u64;
        let response_line = stdio.read_response_line_with_timeout(remaining_ms)?;
        match classify_inbound_jsonrpc(&response_line)? {
            InboundJsonRpc::Notification { .. } => continue,
            InboundJsonRpc::Request {
                id: request_id,
                method,
            } => {
                stdio.reject_server_request(request_id, &method)?;
                continue;
            }
            InboundJsonRpc::Response(response) => {
                // Some servers send a redundant acknowledgement with a null id
                // for `notifications/initialized`; skipping it prevents it from
                // consuming the next request's response slot.
                let Some(response_id) = response.id else {
                    continue;
                };
                if response_id != id {
                    return Err(format!(
                        "MCP response id mismatch: expected {}, got {}",
                        id, response_id
                    ));
                }
                if let Some(error) = response.error {
                    return Err(match error.data {
                        Some(data) => {
                            format!("MCP error {}: {} ({})", error.code, error.message, data)
                        }
                        None => format!("MCP error {}: {}", error.code, error.message),
                    });
                }
                return response.result.ok_or("No result in response".to_string());
            }
        }
    }
}

fn drain_stderr_into_tail(mut stderr: ChildStderr, sink: Arc<Mutex<String>>) {
    let mut buf = [0u8; 1024];
    loop {
        match stderr.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => append_stderr_tail(&sink, &buf[..n]),
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
}

fn append_stderr_tail(sink: &Arc<Mutex<String>>, chunk: &[u8]) {
    let text = String::from_utf8_lossy(chunk);
    let Ok(mut tail) = sink.lock() else {
        return;
    };
    tail.push_str(&text);
    if tail.len() <= STDERR_TAIL_LIMIT {
        return;
    }

    let trim_from = tail.len().saturating_sub(STDERR_TAIL_LIMIT);
    let keep_start = tail
        .char_indices()
        .find(|(idx, _)| *idx >= trim_from)
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    tail.drain(..keep_start);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    #[test]
    fn dropping_mcp_connection_kills_and_reaps_child() {
        let mut process = crate::fork_guard::spawn(
            Command::new("/bin/sleep")
                .arg("30")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped()),
        )
        .unwrap();
        let pid = process.id() as libc::pid_t;
        let stdin = process.stdin.take().unwrap();
        let stdout = process.stdout.take().unwrap();
        let stderr = process.stderr.take().unwrap();
        let connection = McpServerConnection {
            config: McpServerConfig {
                command: "/bin/sleep".to_string(),
                args: vec!["30".to_string()],
                env: Default::default(),
                url: String::new(),
                headers: Default::default(),
                transport: Some(McpTransportKind::Stdio),
                request_timeout_ms: 100,
                disabled: false,
            },
            name: "test".to_string(),
            transport: McpTransport::Stdio(StdioTransport {
                process,
                stdin,
                stdout: BufReader::new(stdout),
                stderr_tail: spawn_stderr_drain(stderr),
            }),
            request_timeout_ms: 100,
            tools: Vec::new(),
            resources: Vec::new(),
            prompts: Vec::new(),
        };

        drop(connection);

        assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
    }
}
