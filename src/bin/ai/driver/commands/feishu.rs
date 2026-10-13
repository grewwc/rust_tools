use std::process::Command;

use serde_json::json;

use crate::ai::mcp::SharedMcpClient;
use crate::ai::driver::tools::oauth::extract_oauth_code_from_text;
use crate::commonw::prompt::{prompt_yes_or_no_interruptible, read_line};

pub fn try_handle_feishu_auth_command(
    mcp_client: &SharedMcpClient,
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
    if normalized != "feishu-auth" && normalized != "feishu auth" && normalized != "feishu_auth" {
        return Ok(false);
    }

    let mut server = None;
    {
        let mc = mcp_client.lock().unwrap();
        for tool in mc.get_all_tools() {
            if let Some((server_name, tool_name)) =
                mc.parse_tool_name_for_known_server(&tool.function.name)
                && tool_name == "oauth_authorize_url"
            {
                server = Some(server_name);
                break;
            }
        }
    }
    let Some(server) = server else {
        println!("未检测到飞书 OAuth MCP 工具（oauth_authorize_url）。");
        println!("- 先运行：cargo run --bin a -- --list-mcp-tools");
        println!("- 再按文档配置：docs/mcp-feishu.md");
        return Ok(true);
    };

    let serve = crate::ai::serve_confirm::active();

    let scope = if serve {
        // `None` is an explicit cancel (client Cancel button / channel gone),
        // not an empty answer: abort rather than silently using the default.
        let Some(scope) =
            crate::ai::serve_confirm::input("OAuth scope (default: offline_access): ")
        else {
            println!("已取消");
            return Ok(true);
        };
        scope
    } else {
        read_line("OAuth scope (default: offline_access): ")
    };
    let scope = if scope.trim().is_empty() {
        "offline_access".to_string()
    } else {
        scope.trim().to_string()
    };

    // In serve mode no local callback server is ever started (the client
    // pastes the code back instead), so the port only matters for building the
    // redirect_uri; the default is used.
    let port_input = if serve {
        String::new()
    } else {
        read_line("Local callback port (default: 8711): ")
    };
    let port = port_input
        .trim()
        .parse::<u16>()
        .ok()
        .filter(|p| *p > 0)
        .unwrap_or(8711);
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");

    let url = {
        let mc = mcp_client.lock().unwrap();
        mc.call_tool(
            &server,
            "oauth_authorize_url",
            json!({
                "redirect_uri": redirect_uri,
                "scope": scope,
                "prompt": "consent",
                "state": "rust-tools-ai"
            }),
        )?
    };
    let url = url.trim().to_string();
    println!("\n授权链接：\n{url}\n");

    let code = if serve {
        // The client opens the URL on its own device and pastes back the
        // redirect target; the local callback server would bind this machine's
        // 127.0.0.1, which a remote browser never reaches.
        let ok = crate::ai::serve_confirm::confirm(&format!(
            "飞书授权。请在浏览器中打开以下链接并完成授权：\n\n{url}\n\n已完成授权？(y/n): "
        ));
        if ok != Some(true) {
            println!("已取消授权");
            return Ok(true);
        }
        let pasted = crate::ai::serve_confirm::input(
            "授权完成后，请粘贴浏览器地址栏中的完整回调 URL（或其中的 code）:",
        );
        let Some(pasted) = pasted else {
            println!("已取消授权");
            return Ok(true);
        };
        match extract_oauth_code_from_text(&pasted) {
            Some(code) => code,
            None => {
                println!("未能从粘贴内容中提取 code：\n{pasted}");
                return Ok(true);
            }
        }
    } else {
        let open_now = prompt_yes_or_no_interruptible("Open browser now? (y/n): ");
        if open_now == Some(true) {
            let program = if cfg!(target_os = "macos") {
                "open"
            } else {
                "xdg-open"
            };
            let _ = crate::fork_guard::status(Command::new(program).arg(&url));
        }

        println!("等待授权回调（{redirect_uri}）...");
        let code_out = {
            let mc = mcp_client.lock().unwrap();
            match mc.call_tool(
                &server,
                "oauth_wait_local_code",
                json!({
                    "port": port,
                    "timeout_sec": 180
                }),
            ) {
                Ok(v) => v,
                Err(err) => {
                    let _ = mc.reset_server(&server);
                    return Err(err.into());
                }
            }
        };
        let code = extract_code_from_wait_output(&code_out).unwrap_or_default();
        if code.is_empty() {
            println!("未获取到 code，原始输出：\n{code_out}");
            return Ok(true);
        }
        code
    };

    let exchange = {
        let mc = mcp_client.lock().unwrap();
        match mc.call_tool(
            &server,
            "oauth_exchange_code",
            json!({
                "code": code,
                "redirect_uri": redirect_uri
            }),
        ) {
            Ok(v) => v,
            Err(err) => {
                let _ = mc.reset_server(&server);
                return Err(err.into());
            }
        }
    };
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

