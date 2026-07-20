//! mcp_excel — 一个 stdio JSON-RPC MCP server，用 AppleScript(osascript) 驱动
//! 已安装的 Microsoft Excel，为 `a` Agent 提供操作真实 Excel 软件的能力
//! （打开/读单元格/读区域/写单元格/写区域/导出 CSV/列 sheet...）。
//!
//! 设计要点见 crates/mcp_excel/AGENTS.md：
//! - **无长活会话**：osascript 每次是独立子进程，Excel 应用自身维持工作簿的
//!   打开态，跨调用共享；故 main 不持有任何 session 状态，比 mcp_browser 简单。
//! - 顺序处理请求：Agent 一次只调一个工具，宿主 per-conn 串行化整个往返。
//! - 每操作超时由 with_timeout 兜底（默认 90s < 宿主 request_timeout_ms 120s），
//!   超时返回不含 transport 触发词的干净 JSON-RPC 错误，避免被宿主 kill。
//! - 存盘走 Rust 侧写文件（export_csv），绕过 Excel 沙盒版 save 的 -50 限制。

mod jsonrpc;
mod osa;
mod tools;

use jsonrpc::{JsonRpcErr, write_err, write_result};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};

#[tokio::main(flavor = "multi_thread")]
async fn main() {
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
            "initialize" => Ok(tools::initialize_result()),
            "notifications/initialized" => Ok(json!({})),
            "tools/list" => Ok(tools::tools_list_result()),
            "tools/call" => tools::handle_tools_call(params).await,
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
}
