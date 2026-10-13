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

/// Pull an OAuth code out of text a user pasted back: a full redirect URL
/// (the code in its query string or fragment), or a bare code. Serves the
/// serve-client flow, where the user authorizes on their own device and the
/// callback never reaches this machine. `None` when the text looks like a URL
/// but carries no code: fail closed instead of feeding the whole URL to the
/// exchange endpoint.
pub(crate) fn extract_oauth_code_from_text(s: &str) -> Option<String> {
    let s = s.trim();
    let code_from = |haystack: &str| {
        for part in haystack.split('&') {
            let mut it = part.splitn(2, '=');
            if it.next() == Some("code") {
                let v = it.next().unwrap_or("").trim();
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
        None
    };
    if let Some(qidx) = s.find('?') {
        let tail = &s[qidx + 1..];
        let (query, fragment) = match tail.split_once('#') {
            Some((q, f)) => (q, Some(f)),
            None => (tail, None),
        };
        return code_from(query).or_else(|| fragment.and_then(code_from));
    }
    if let Some(fidx) = s.find('#') {
        return code_from(&s[fidx + 1..]);
    }
    if !s.is_empty() {
        return Some(s.to_string());
    }
    None
}

fn open_browser(url: &str) {
    let program = if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    let _ = crate::fork_guard::status(std::process::Command::new(program).arg(url));
}

fn run_feishu_oauth_flow(
    mcp_client: &McpClient,
    server_name: &str,
    scope: &str,
) -> Result<(), String> {
    let serve = crate::ai::serve_confirm::active();
    let confirm = if serve {
        crate::ai::serve_confirm::confirm("Feishu authorization required. Authorize now? (y/n): ")
    } else {
        prompt_yes_or_no_interruptible("Feishu authorization required. Authorize now? (y/n): ")
    };
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
    let code = if serve {
        // No terminal and no browser we can open on the user's behalf: the
        // client opens the URL on its own device and pastes back the redirect
        // target. The local callback server would bind this machine's
        // 127.0.0.1, which a remote browser never reaches.
        let ok = crate::ai::serve_confirm::confirm(&format!(
            "Feishu authorization. Open this URL in your browser and authorize:\n\n{auth_url}\n\nAuthorized? (y/n): "
        ));
        if ok != Some(true) {
            return Err("feishu oauth canceled by user".to_string());
        }
        let pasted = crate::ai::serve_confirm::input(
            "Paste the full redirect URL from the address bar after authorizing (or just the code):",
        );
        let Some(pasted) = pasted else {
            return Err("feishu oauth canceled by user".to_string());
        };
        extract_oauth_code_from_text(&pasted).ok_or_else(|| {
            "failed to extract oauth code from the pasted text".to_string()
        })?
    } else {
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
        code
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

#[cfg(test)]
mod tests {
    use super::extract_oauth_code_from_text;

    #[test]
    fn extract_code_from_query_string() {
        assert_eq!(
            extract_oauth_code_from_text(
                "https://open.feishu.cn/open-apis/authen/v1/index?code=abc123&state=x"
            )
            .as_deref(),
            Some("abc123")
        );
        // `code` may appear after other parameters.
        assert_eq!(
            extract_oauth_code_from_text("https://host/cb?state=x&code=def456").as_deref(),
            Some("def456")
        );
    }

    #[test]
    fn extract_code_from_fragment() {
        // Fragment-only callback: no query at all.
        assert_eq!(
            extract_oauth_code_from_text("https://host/cb#code=abc123").as_deref(),
            Some("abc123")
        );
        // Query without code, code in the fragment after it.
        assert_eq!(
            extract_oauth_code_from_text("https://host/cb?state=x#code=def456").as_deref(),
            Some("def456")
        );
    }

    #[test]
    fn extract_code_from_bare_paste() {
        assert_eq!(extract_oauth_code_from_text("  abc123  ").as_deref(), Some("abc123"));
        assert_eq!(extract_oauth_code_from_text("").is_none(), true);
    }

    #[test]
    fn url_without_code_fails_closed() {
        // A URL-shaped paste without a code must not be passed through as the
        // code itself; the exchange would only fail later with a confusing
        // error. The `code` parameter must be present (and non-empty).
        assert_eq!(extract_oauth_code_from_text("https://host/cb?state=x").is_none(), true);
        assert_eq!(extract_oauth_code_from_text("https://host/cb?code=").is_none(), true);
        assert_eq!(extract_oauth_code_from_text("https://host/cb?state=x#frag").is_none(), true);
    }
}
