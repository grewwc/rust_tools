//! JSON-RPC over stdio 助手：错误类型、结果/错误写出、MCP content 信封、
//! 文本截断，以及每操作超时包装。
//!
//! 骨架照抄 `crates/mcp_browser/src/jsonrpc.rs`（同款 stdio MCP server 范式）。

use serde_json::{Value, json};
use tokio::io::{AsyncWrite, AsyncWriteExt};

/// 提取文本的字符上限。宿主超过约 32K 会把结果卸载到磁盘、只给模型看 800
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
/// 超长时追加一行说明。
pub fn cap_text(s: &str) -> String {
    if s.chars().count() <= CAP_CHARS {
        return s.to_string();
    }
    let truncated: String = s.chars().take(CAP_CHARS).collect();
    format!(
        "{truncated}\n\n[... truncated: output exceeded {CAP_CHARS} chars; read a smaller range or export to a file instead ...]"
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

/// 给每个 osascript 操作套一个 server 内部上限（默认 90s，短于宿主 120s 的
/// request_timeout_ms），超时返回干净的 JSON-RPC error（-32001）。
///
/// 关键：超时/错误消息**绝不能**包含宿主的 transport 触发词
/// （"mcp response timeout" / "broken pipe" / "closed the stream" /
///  "process exited" / "failed to read response" /
///  "failed waiting for mcp response"，见 mcp/client.rs），
/// 否则宿主会 kill+restart 子进程。
pub async fn with_timeout<F, T>(ms: u64, fut: F) -> Result<T, JsonRpcErr>
where
    F: std::future::Future<Output = Result<T, String>>,
{
    match tokio::time::timeout(std::time::Duration::from_millis(ms), fut).await {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(JsonRpcErr::new(-32000, &e, None)),
        Err(_) => Err(JsonRpcErr::new(
            -32001,
            &format!("operation reached the {ms} ms mcp_excel server cap"),
            None,
        )),
    }
}
