//! mcp_excel — 一个 stdio JSON-RPC MCP server，用 AppleScript(osascript) 驱动
//! 已安装的 Microsoft Excel，为 `a` Agent 提供操作真实 Excel 软件的能力
//! （打开/读单元格/读区域/写单元格/写区域/导出 CSV/列 sheet...）。
//!
//! 设计要点见 crates/mcp_excel/AGENTS.md：
//! - **无长活会话**：osascript 每次是独立子进程，Excel 应用自身维持工作簿的
//!   打开态，跨调用共享；故 ExcelServer 不持有任何 session 状态，比 mcp_browser 简单。
//! - 协议样板（stdin 主循环 / 方法分发 / 写回）复用 `mcp_stdio::run`；本文件只提供
//!   `McpServer` 三段与工具集相关的实现，shutdown 用默认空实现。
//! - 每操作超时由 with_timeout 兜底（默认 90s < 宿主 request_timeout_ms 120s），
//!   超时返回不含 transport 触发词的干净 JSON-RPC 错误，避免被宿主 kill。
//! - 存盘走 Rust 侧写文件（export_csv），绕过 Excel 沙盒版 save 的 -50 限制。

mod osa;
mod tools;

use mcp_stdio::{JsonRpcErr, McpServer};
use serde_json::Value;

/// Excel MCP server——无会话，单元结构体。Excel 应用本身持有打开的工作簿态，
/// 跨独立 osascript 子进程共享，故这里无需任何状态。
struct ExcelServer;

impl McpServer for ExcelServer {
    fn initialize_result(&self) -> Value {
        tools::initialize_result()
    }
    fn tools_list_result(&self) -> Value {
        tools::tools_list_result()
    }
    async fn handle_tools_call(&mut self, params: Option<Value>) -> Result<Value, JsonRpcErr> {
        tools::handle_tools_call(params).await
    }
    // shutdown 用默认空实现（无会话可关）。
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    mcp_stdio::run(ExcelServer).await;
}
