use std::process::Command;

use serde_json::json;

use crate::ai::mcp::McpClient;
use crate::commonw::prompt::{prompt_yes_or_no_interruptible, read_line};

pub fn try_handle_feishu_auth_command(
    mcp_client: &mut McpClient,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let normalized = if let Some(rest) = trimmed.strip_prefix('/') {
        rest
    } else if let Some(rest) = trimmed.strip_prefix(':') {
        rest
    } else {
        return Ok(false);
    };
    if normalized != "feishu-auth"
        && normalized != "feishu auth"
        && normalized != "feishu_auth"
    {
        return Ok(false);
    }

    let mut server = None;
    for tool in mcp_client.get_all_tools() {
        if let Some((server_name, tool_name)) =
            mcp_client.parse_tool_name_for_known_server(&tool.function.name)
            && tool_name == "oauth_authorize_url"
        {
            server = Some(server_name);
            break;
        }
    }
    let Some(server) = server else {
        println!("未检测到飞书 OAuth MCP 工具（oauth_authorize_url）。");
        println!("- 先运行：cargo run --bin a -- --list-mcp-tools");
        println!("- 再按文档配置：docs/mcp-feishu.md");
        return Ok(true);
    };

    let scope = read_line("OAuth scope (default: offline_access): ");
    let scope = if scope.trim().is_empty() {
        "offline_access".to_string()
    } else {
        scope.trim().to_string()
    };

    let port_input = read_line("Local callback port (default: 8711): ");
    let port = port_input
        .trim()
        .parse::<u16>()
        .ok()
        .filter(|p| *p > 0)
        .unwrap_or(8711);
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");

    let url = mcp_client.call_tool(
        &server,
        "oauth_authorize_url",
        json!({
            "redirect_uri": redirect_uri,
            "scope": scope,
            "prompt": "consent",
            "state": "rust-tools-ai"
        }),
    )?;
    let url = url.trim().to_string();
    println!("\n授权链接：\n{url}\n");

    let open_now = prompt_yes_or_no_interruptible("Open browser now? (y/n): ");
    if open_now == Some(true) {
        let program = if cfg!(target_os = "macos") {
            "open"
        } else {
            "xdg-open"
        };
        let _ = Command::new(program).arg(&url).status();
    }

    println!("等待授权回调（{redirect_uri}）...");
    let code_out = mcp_client.call_tool(
        &server,
        "oauth_wait_local_code",
        json!({
            "port": port,
            "timeout_sec": 180
        }),
    )?;
    let code = extract_code_from_wait_output(&code_out).unwrap_or_default();
    if code.is_empty() {
        println!("未获取到 code，原始输出：\n{code_out}");
        return Ok(true);
    }

    let exchange =
        mcp_client.call_tool(&server, "oauth_exchange_code", json!({ "code": code }))?;
    println!("{exchange}");
    Ok(true)
}

fn extract_code_from_wait_output(s: &str) -> Option<String> {
    for line in s.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("code:") {
            let v = rest.trim().to_string();
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    None
}
