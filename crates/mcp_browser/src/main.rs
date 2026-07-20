//! mcp_browser — 一个 stdio JSON-RPC MCP server，用 chromiumoxide(CDP) 驱动
//! 已安装的 Chrome，为 `a` Agent 提供浏览器自动化能力（导航/点击/输入/提取/截图...）。
//!
//! 设计要点见 crates/mcp_browser/AGENTS.md：
//! - 顺序处理请求：Agent 一次只调一个工具，宿主 per-conn 串行化整个往返，
//!   故无需给 session 上锁；顺序写出也避免交错写坏 newline 协议。
//! - 会话持久化：main 持有单个 Option<BrowserSession>，以 &mut 传入分发器；
//!   首次 navigate 时懒启动，登录/多步流程靠复用的单个 Page 保持。
//! - 每操作超时由 with_timeout 兜底（默认 90s < 宿主 request_timeout_ms 120s），
//!   超时返回不含 transport 触发词的干净 JSON-RPC 错误，避免被宿主 kill。

mod browser;
mod jsonrpc;
mod tools;

use browser::BrowserSession;
use jsonrpc::{JsonRpcErr, write_err, write_result};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, BufReader};

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    // 启动时垃圾回收：清掉此前被 kill、未能优雅 shutdown 的残留临时 profile。
    // 见 browser::gc_stale_profiles —— SIGKILL 不可捕获，故靠“下次启动”兜底。
    browser::gc_stale_profiles();

    let mut reader = BufReader::new(tokio::io::stdin());
    let mut stdout = tokio::io::stdout();
    let mut session: Option<BrowserSession> = None;
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
            "tools/call" => tools::handle_tools_call(&mut session, params).await,
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

    if let Some(s) = session.take() {
        s.shutdown().await;
    }
}
