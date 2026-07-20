//! `mcp_stdio` — 两个 async stdio JSON-RPC MCP server(`mcp_browser`、`mcp_excel`）
//! 共享的协议骨架。承载与"操作什么软件"无关的 MCP-over-stdio 样板：
//!
//! - 传输层：`JsonRpcErr`、`write_result`/`write_err`（async，每行写完 flush）、
//!   `text_content` MCP content 信封、`cap_text` 文本截断、`with_timeout` 每操作超时。
//! - 主循环：`run<S: McpServer>()` —— 读行 → 解析 → 方法分发 → 写回，各 server 只需
//!   实现 `McpServer` trait（initialize/tools_list/tools_call/可选 shutdown）。
//!
//! 未来新增"驱动 OS 本地 application"的 MCP server（word/photoshop/pdf preview...）
//! 时，只写自己的工具集与驱动层，协议样板零重复复用本 crate。

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader};

/// 提取文本/HTML 的字符上限。宿主超过约 32K 会把结果卸载到磁盘、只给模型看 800
/// 字符 stub（见 driver/turn_runtime/mod.rs），所以这里截断到 24K 并加提示。
pub const CAP_CHARS: usize = 24_000;

/// JSON-RPC 错误载体。
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

/// 把一段文本包成 MCP 的 `{"content":[{"type":"text","text":...}]}` 信封。
/// 宿主 `call_tool` 只会读取 `content[0].text`，所以一切都必须走这里回传。
pub fn text_content(text: impl Into<String>) -> Value {
    json!({ "content": [ { "type": "text", "text": text.into() } ] })
}

/// 把文本截断到 `CAP_CHARS` 个字符（按 char 边界，避免切坏 UTF-8），
/// 超长时追加一行说明。提示措辞保持领域中立，浏览器/表格等各 server 通用。
pub fn cap_text(s: &str) -> String {
    if s.chars().count() <= CAP_CHARS {
        return s.to_string();
    }
    let truncated: String = s.chars().take(CAP_CHARS).collect();
    format!(
        "{truncated}\n\n[... truncated: output exceeded {CAP_CHARS} chars; narrow the request (smaller selector/range) or write the full result to a file instead ...]"
    )
}

/// 写出一条成功的 JSON-RPC 响应行并 flush。
pub async fn write_result<W: AsyncWrite + Unpin>(stdout: &mut W, id: Option<&Value>, result: Value) {
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

/// 写出一条 JSON-RPC 错误响应行并 flush。
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

/// 给每个 server 内部操作套一个上限（默认 90s，短于宿主 120s 的
/// request_timeout_ms），超时返回干净的 JSON-RPC error（-32001）。
///
/// 关键：超时/错误消息**绝不能**包含宿主的 transport 触发词
/// （"mcp response timeout" / "broken pipe" / "closed the stream" /
///  "process exited" / "failed to read response" /
///  "failed waiting for mcp response"，见 mcp/client.rs），
/// 否则宿主会 kill+restart 子进程、丢掉整个会话。
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

/// 一个 stdio MCP server 需要实现的最小接口。协议样板（读行/分发/写回）由
/// [`run`] 统一承担，实现者只提供三段与自身工具集相关的内容 + 可选收尾。
///
/// `async fn` 直接写在 trait 上（不引 `async-trait`）：`run` 只被各 bin 的
/// `#[tokio::main]` `block_on`、绝不 `tokio::spawn`，故 server future 无需 `Send`。
#[allow(async_fn_in_trait)]
pub trait McpServer {
    /// `initialize` 方法的返回体（serverInfo / capabilities / protocolVersion）。
    fn initialize_result(&self) -> Value;
    /// `tools/list` 方法的返回体（工具 schema 列表）。
    fn tools_list_result(&self) -> Value;
    /// `tools/call` 方法的分发：按 params 里的工具名执行并回结果。
    async fn handle_tools_call(&mut self, params: Option<Value>) -> Result<Value, JsonRpcErr>;
    /// 主循环退出（EOF 或读错误）后的收尾。默认空实现；持有长活会话的
    /// server（如 mcp_browser）覆写它做优雅关闭。
    async fn shutdown(&mut self) {}
}

/// 共享的 stdio JSON-RPC 主循环：从 stdin 逐行读 → 解析 → 方法分发 → 写回 stdout。
///
/// 仅由各 bin 的 `#[tokio::main]` `block_on`，绝不 `tokio::spawn`（故 future 无需
/// `Send`；`&mut server` 跨 `.await` 始终在 block_on 线程上，不跨线程）。
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
        let method = req
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let params = req.get("params").cloned();

        let res = match method.as_str() {
            "initialize" => Ok(server.initialize_result()),
            "notifications/initialized" => Ok(json!({})),
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

        match res {
            Ok(r) => write_result(&mut stdout, id.as_ref(), r).await,
            Err(e) => write_err(&mut stdout, id.as_ref(), e.code, &e.message, e.data).await,
        }
    }

    // EOF 与读错误两种退出都会走到这里；无会话的 server 用默认空实现。
    server.shutdown().await;
}
