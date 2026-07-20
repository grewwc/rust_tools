//! mcp_browser — 一个 stdio JSON-RPC MCP server，用 chromiumoxide(CDP) 驱动
//! 已安装的 Chrome，为 `a` Agent 提供浏览器自动化能力（导航/点击/输入/提取/截图...）。
//!
//! 设计要点见 crates/mcp_browser/AGENTS.md：
//! - 协议样板（stdin 主循环 / 方法分发 / 写回）复用 `mcp_stdio::run`；本文件只提供
//!   `McpServer` 的实现：三段工具相关内容 + 覆写 shutdown 做会话优雅关闭。
//! - 会话持久化：BrowserServer 持有单个 Option<BrowserSession>，以 &mut 传入分发器；
//!   首次 navigate 时懒启动，登录/多步流程靠复用的单个 Page 保持。
//! - 每操作超时由 with_timeout 兜底（默认 90s < 宿主 request_timeout_ms 120s），
//!   超时返回不含 transport 触发词的干净 JSON-RPC 错误，避免被宿主 kill。

mod browser;
mod tools;

use browser::BrowserSession;
use mcp_stdio::{JsonRpcErr, McpServer};
use serde_json::Value;

/// 浏览器 MCP server——持有单个懒启动的 CDP 会话。首次 navigate 时启动，
/// 登录/多步流程靠复用同一个 Page 保持；主循环退出时经 shutdown 优雅关闭。
struct BrowserServer {
    session: Option<BrowserSession>,
}

impl BrowserServer {
    fn new() -> Self {
        Self { session: None }
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
        tools::handle_tools_call(&mut self.session, params).await
    }
    /// 覆写默认空实现：主循环退出时优雅关闭 CDP 会话。
    /// 注意与 `BrowserSession::shutdown(self)`（固有、消费 self）区分——这里先 take
    /// 出会话再消费，避免 &mut self 与消费型签名冲突。
    async fn shutdown(&mut self) {
        if let Some(s) = self.session.take() {
            s.shutdown().await;
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
