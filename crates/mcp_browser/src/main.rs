//! mcp_browser — 一个 stdio JSON-RPC MCP server，为 `a` Agent 提供浏览器自动化
//! 能力（导航/点击/输入/提取/截图...）。
//!
//! 设计要点见 crates/mcp_browser/AGENTS.md：
//! - 协议样板（stdin 主循环 / 方法分发 / 写回）复用 `mcp_stdio::run`；本文件只提供
//!   `McpServer` 的实现：三段工具相关内容 + 覆写 shutdown 做会话优雅关闭。
//! - 双驱动模式：默认 `applescript` 复用用户已打开的 Chrome（新 tab、保留 cookie、
//!   绝不退出用户浏览器）；`MCP_BROWSER_DRIVER=cdp`（或设置 MCP_BROWSER_WS_URL）
//!   走原有受控实例路径。BrowserServer 持有 Option<Session> 枚举，按模式分发。
//! - 每操作超时由 with_timeout 兜底（默认 90s < 宿主 request_timeout_ms 120s），
//!   超时返回不含 transport 触发词的干净 JSON-RPC 错误，避免被宿主 kill。

mod applescript;
mod browser;
mod tools;

use browser::{DriverMode, Session};
use mcp_stdio::{JsonRpcErr, McpServer};
use serde_json::Value;

/// 浏览器 MCP server——按驱动模式持有懒启动的会话。applescript 模式驱动用户
/// 已打开的 Chrome（会话标签页 tracked by window_id+tab_id）；cdp 模式懒启动
/// 受控实例。主循环退出时经 shutdown 优雅关闭（applescript 模式只关会话标签页，
/// 绝不退出用户浏览器）。
struct BrowserServer {
    mode: DriverMode,
    session: Option<Session>,
}

impl BrowserServer {
    fn new() -> Self {
        Self {
            mode: DriverMode::from_env(),
            session: None,
        }
    }
}

impl McpServer for BrowserServer {
    fn initialize_result(&self) -> Value {
        tools::initialize_result()
    }
    fn tools_list_result(&self) -> Value {
        tools::tools_list_result()
    }
    async fn handle_tools_call(&mut self, params: Option<Value>) -> Result<Value, JsonRpcErr> {
        if self.mode.is_applescript() {
            // 复用 tools.rs 的 13 个工具定义/参数名，实现走 applescript 驱动。
            let s = match &mut self.session {
                Some(Session::AppleScript(a)) => a,
                other => {
                    *other = Some(Session::AppleScript(applescript::ApplescriptSession::new()));
                    match self.session.as_mut().unwrap() {
                        Session::AppleScript(a) => a,
                        Session::Cdp(_) => unreachable!(),
                    }
                }
            };
            applescript::handle_tools_call(s, params).await
        } else {
            // CDP 路径完全复用 tools.rs：把 Cdp 变体取出来传给原有实现。
            let mut cdp = match self.session.take() {
                Some(Session::Cdp(c)) => Some(c),
                other => {
                    self.session = other;
                    None
                }
            };
            let r = tools::handle_tools_call(&mut cdp, params).await;
            if let Some(c) = cdp {
                self.session = Some(Session::Cdp(c));
            }
            r
        }
    }
    /// 覆写默认空实现：主循环退出时优雅关闭会话。applescript 模式只关闭
    /// 会话标签页（绝不退出用户 Chrome）；CDP 模式关闭受控实例。
    async fn shutdown(&mut self) {
        if let Some(s) = self.session.take() {
            match s {
                Session::Cdp(c) => c.shutdown().await,
                Session::AppleScript(mut a) => a.shutdown().await,
            }
        }
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    // 启动时垃圾回收：清掉此前被 kill、未能优雅 shutdown 的残留临时 profile。
    // 见 browser::gc_stale_profiles —— SIGKILL 不可捕获，故靠“下次启动”兜底。
    // 保留为 main 显式一行（浏览器专属、同步、幂等），不下沉为 trait 钩子。
    browser::gc_stale_profiles();

    mcp_stdio::run(BrowserServer::new()).await;
}
