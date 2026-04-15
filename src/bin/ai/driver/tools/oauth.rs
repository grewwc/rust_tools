use regex::Regex;
use serde_json::{Value, json};
use std::sync::LazyLock;

use crate::ai::{
    mcp::McpClient,
    types::{ToolCall, ToolResult},
};
use crate::commonw::prompt::prompt_yes_or_no_interruptible;

fn looks_like_feishu_oauth_required(err: &str) -> bool {
    let e = err.to_lowercase();
    e.contains("missing user_access_token")
        || e.contains("invalid access token")
        || e.contains("99991668")
        || e.contains("99991679")
        || e.contains("re-authorization")
}

fn extract_feishu_required_scopes(err: &str) -> Vec<String> {
    static RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\b[a-z][a-z0-9_-]*:[a-z0-9_-]+(?::[a-z0-9_-]+)*\b").unwrap());
    let mut out = Vec::new();
    for m in RE.find_iter(err) {
        let s = m.as_str();
        if s.starts_with("http") {
            continue;
        }
        if s.contains("open-apis") {
            continue;
        }
        if s.contains("bytedance") {
            continue;
        }
        out.push(s.to_string());
    }
    out.sort();
    out.dedup();
    out
}

fn extract_oauth_code(s: &str) -> Option<String> {
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

fn open_browser(url: &str) {
    let program = if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    let _ = std::process::Command::new(program).arg(url).status();
}

fn run_feishu_oauth_flow(
    mcp_client: &McpClient,
    server_name: &str,
    scope: &str,
) -> Result<(), String> {
    let confirm =
        prompt_yes_or_no_interruptible("Feishu authorization required. Authorize now? (y/n): ");
    if confirm != Some(true) {
        return Err(if confirm.is_none() {
            "feishu oauth canceled by user (Ctrl+C)".to_string()
        } else {
            "feishu oauth canceled by user".to_string()
        });
    }

    let port = 8711u16;
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");
    let auth_url = mcp_client.call_tool(
        server_name,
        "oauth_authorize_url",
        json!({
            "redirect_uri": redirect_uri,
            "scope": scope,
            "prompt": "consent",
            "state": "rust-tools-ai"
        }),
    )?;
    let auth_url = auth_url.trim().to_string();
    println!("\n[feishu] authorize url:\n{}\n", auth_url);

    let open_now = prompt_yes_or_no_interruptible("Open browser now? (y/n): ");
    if open_now == Some(true) {
        open_browser(&auth_url);
    }

    println!("[feishu] waiting for oauth callback on http://127.0.0.1:{port}/callback ...");
    let code_out = mcp_client.call_tool(
        server_name,
        "oauth_wait_local_code",
        json!({
            "port": port,
            "timeout_sec": 180
        }),
    )?;
    let Some(code) = extract_oauth_code(&code_out) else {
        return Err(format!(
            "failed to capture oauth code. raw output:\n{}",
            code_out.trim()
        ));
    };

    let exchange_out = mcp_client.call_tool(
        server_name,
        "oauth_exchange_code",
        json!({
            "code": code,
            "redirect_uri": redirect_uri
        }),
    )?;
    println!("\n[feishu] {}\n", exchange_out.trim());
    Ok(())
}

fn oauth_scope_from_error(err: &str) -> String {
    let required = extract_feishu_required_scopes(err);
    if required.is_empty() {
        return "offline_access".to_string();
    }

    let mut parts = Vec::with_capacity(required.len() + 1);
    parts.push("offline_access".to_string());
    parts.extend(required);
    parts.join(" ")
}

pub(super) fn execute_mcp_tool_call(
    mcp_client: &McpClient,
    tool_call: &ToolCall,
    server_name: &str,
    tool_name: &str,
    args: &Value,
) -> Result<ToolResult, String> {
    match mcp_client.call_tool(server_name, tool_name, args.clone()) {
        Ok(content) => Ok(ToolResult {
            tool_call_id: tool_call.id.clone(),
            content,
        }),
        Err(err) if !tool_name.starts_with("oauth_") && looks_like_feishu_oauth_required(&err) => {
            let scope = oauth_scope_from_error(&err);
            if let Err(oauth_err) = run_feishu_oauth_flow(mcp_client, server_name, &scope) {
                let _ = mcp_client.reset_server(server_name);
                return Err(format!("feishu oauth failed: {}", oauth_err));
            }
            let content = mcp_client.call_tool(server_name, tool_name, args.clone())?;
            Ok(ToolResult {
                tool_call_id: tool_call.id.clone(),
                content,
            })
        }
        Err(err) => Err(err),
    }
}
