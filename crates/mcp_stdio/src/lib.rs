//! `mcp_stdio` — shared protocol skeleton for the async stdio JSON-RPC MCP
//! servers (`mcp_browser`, `mcp_excel`, ...). Carries everything that is
//! unrelated to "which app is driven":
//!
//! - Transport: `JsonRpcErr`, `write_result`/`write_err` (async, flush per line),
//!   the `text_content` MCP content envelope, `cap_text` text truncation, and
//!   `with_timeout` per-operation timeouts.
//! - Main loop: `run<S: McpServer>()` — read line → parse → dispatch → write back.
//!   A new server only implements the `McpServer` trait (initialize / tools_list /
//!   tools_call / optional shutdown).
//!
//! Future "drive an OS-native app" MCP servers (word / photoshop / pdf preview...)
//! should reuse this crate for the protocol boilerplate and only write their own
//! tool set and driver layer.

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};

/// Character cap for extracted text/HTML. The host offloads results above ~32K to
/// disk and only shows the model an 800-char stub (see driver/turn_runtime/mod.rs),
/// so servers truncate here at 24K and append a hint.
pub const CAP_CHARS: usize = 24_000;

/// JSON-RPC error carrier.
pub struct JsonRpcErr {
    pub code: i32,
    pub message: String,
    pub data: Option<Value>,
}

impl JsonRpcErr {
    pub fn new(code: i32, message: &str, data: Option<Value>) -> Self {
        JsonRpcErr {
            code,
            message: message.to_string(),
            data,
        }
    }
}

/// Wrap a text payload in the MCP `{"content":[{"type":"text","text":...}]}`
/// envelope. The host's `call_tool` reads only `content[0].text`, so every payload
/// must go through this.
pub fn text_content(text: impl Into<String>) -> Value {
    json!({ "content": [ { "type": "text", "text": text.into() } ] })
}

/// Truncate text to `CAP_CHARS` characters (on char boundaries, so UTF-8 is never
/// split) and append a hint line when the input was cut. The hint wording stays
/// domain-neutral so every server (browser, spreadsheet, ...) can reuse it.
pub fn cap_text(s: &str) -> String {
    if s.chars().count() <= CAP_CHARS {
        return s.to_string();
    }
    let truncated: String = s.chars().take(CAP_CHARS).collect();
    format!(
        "{truncated}\n\n[... truncated: output exceeded {CAP_CHARS} chars; narrow the request (smaller selector/range) or write the full result to a file instead ...]"
    )
}

/// Write one successful JSON-RPC response line and flush.
pub async fn write_result<W: AsyncWrite + Unpin>(
    stdout: &mut W,
    id: Option<&Value>,
    result: Value,
) {
    let resp = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    });
    let mut line = resp.to_string();
    line.push('\n');
    let _ = stdout.write_all(line.as_bytes()).await;
    let _ = stdout.flush().await;
}

/// Write one JSON-RPC error response line and flush.
pub async fn write_err<W: AsyncWrite + Unpin>(
    stdout: &mut W,
    id: Option<&Value>,
    code: i32,
    message: &str,
    data: Option<Value>,
) {
    let mut err_obj = json!({ "code": code, "message": message });
    if let Some(d) = data {
        err_obj["data"] = d;
    }
    let resp = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": err_obj,
    });
    let mut line = resp.to_string();
    line.push('\n');
    let _ = stdout.write_all(line.as_bytes()).await;
    let _ = stdout.flush().await;
}

/// Bound one server-side operation (default 90s, below the host's configured
/// request_timeout_ms); a timeout returns a clean JSON-RPC error (-32001).
///
/// Critical: the timeout/error wording must never contain the host's transport
/// trigger phrases ("mcp response timeout" / "broken pipe" / "closed the stream
/// unexpectedly" / "process exited with status" / "failed to read response" /
/// "failed waiting for mcp response", see src/bin/ai/mcp/client.rs), or the host
/// kills + restarts the subprocess and the whole session is lost.
pub async fn with_timeout<F, T>(ms: u64, fut: F) -> Result<T, JsonRpcErr>
where
    F: std::future::Future<Output = Result<T, String>>,
{
    match tokio::time::timeout(std::time::Duration::from_millis(ms), fut).await {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(JsonRpcErr::new(-32000, &e, None)),
        Err(_) => Err(JsonRpcErr::new(
            -32001,
            &format!("operation reached the {ms} ms server cap"),
            None,
        )),
    }
}

/// Minimal interface a stdio MCP server must implement. The protocol boilerplate
/// (line read / dispatch / write back) is handled by [`run`]; implementors only
/// provide the three tool-related sections plus an optional shutdown hook.
///
/// `async fn` sits directly on the trait (no `async-trait`): `run` is only ever
/// `block_on`-ed by each binary's `#[tokio::main]`, never `tokio::spawn`, so the
/// server future does not need to be `Send`.
#[allow(async_fn_in_trait)]
pub trait McpServer {
    /// Return body of the `initialize` method (serverInfo / capabilities / protocolVersion).
    fn initialize_result(&self) -> Value;
    /// Return body of the `tools/list` method (the tool schema list).
    fn tools_list_result(&self) -> Value;
    /// Dispatch of the `tools/call` method: run the tool named in params and return its result.
    async fn handle_tools_call(&mut self, params: Option<Value>) -> Result<Value, JsonRpcErr>;
    /// Cleanup after the main loop exits (EOF or a read error). Defaults to a no-op;
    /// servers holding a long-lived session (like mcp_browser) override it for
    /// graceful shutdown.
    async fn shutdown(&mut self) {}
}

/// Shared stdio JSON-RPC main loop: read lines from stdin → parse → dispatch →
/// write back to stdout.
///
/// Only ever `block_on`-ed by each binary's `#[tokio::main]`, never `tokio::spawn`
/// (hence the future needs no `Send`; `&mut server` stays on the block_on thread
/// across `.await` and never crosses threads).
pub async fn run<S: McpServer>(mut server: S) {
    let mut reader = BufReader::new(tokio::io::stdin());
    let mut stdout = tokio::io::stdout();
    let mut line = String::new();

    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(_) => break,
        }
        let raw = line.trim();
        if raw.is_empty() {
            continue;
        }

        let req: Value = match serde_json::from_str(raw) {
            Ok(v) => v,
            Err(e) => {
                write_err(&mut stdout, None, -32700, &e.to_string(), None).await;
                continue;
            }
        };

        let id = req.get("id").cloned();
        // JSON-RPC 2.0 §2.2: a request without an "id" is a notification and must
        // never be answered — not even with an error for an unknown method.
        let is_notification = id.is_none();
        let method = req
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let params = req.get("params").cloned();

        let res = match method.as_str() {
            "initialize" => Ok(server.initialize_result()),
            "tools/list" => Ok(server.tools_list_result()),
            "tools/call" => server.handle_tools_call(params).await,
            "resources/list" => Ok(json!({ "resources": [] })),
            "prompts/list" => Ok(json!({ "prompts": [] })),
            _ => Err(JsonRpcErr::new(
                -32601,
                "Method not found",
                Some(json!({ "method": method })),
            )),
        };

        if is_notification {
            // Notifications still run their side effects (for known methods) but
            // produce no response line at all.
            continue;
        }

        match res {
            Ok(r) => write_result(&mut stdout, id.as_ref(), r).await,
            Err(e) => write_err(&mut stdout, id.as_ref(), e.code, &e.message, e.data).await,
        }
    }

    // Both EOF and read-error exits land here; servers without a session use the
    // default no-op shutdown.
    server.shutdown().await;
}
