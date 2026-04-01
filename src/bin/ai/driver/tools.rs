use colored::Colorize;
use regex::Regex;
use serde_json::{Value, json};
use std::error::Error;
use std::sync::LazyLock;

use crate::{
    ai::{
        mcp::McpClient,
        tools as builtin_tools,
        types::{ToolCall, ToolResult},
    },
    common::prompt::prompt_yes_or_no_interruptible,
};

#[derive(Debug, Clone)]
enum ToolRoute {
    Builtin,
    Mcp {
        server_name: String,
        tool_name: String,
    },
}

#[derive(Debug, Clone)]
struct PreparedToolCall {
    route: ToolRoute,
    args: Value,
}

pub(super) struct ExecuteToolCallsResult {
    pub(super) executed_tool_calls: Vec<ToolCall>,
    pub(super) tool_results: Vec<ToolResult>,
}

pub(super) struct RunOneResult {
    pub(super) tool_result: ToolResult,
    pub(super) ok: bool,
    pub(super) executed: bool,
}

fn is_barrier_tool(tool_name: &str) -> bool {
    matches!(tool_name, "search_files" | "list_directory" | "web_search")
}

fn should_barrier_after(route: &ToolRoute, tool_call: &ToolCall, ok: bool, content: &str) -> bool {
    match route {
        ToolRoute::Mcp { .. } => true,
        ToolRoute::Builtin if tool_call.function.name == "search_files" => {
            ok && !content.trim().is_empty()
        }
        ToolRoute::Builtin => is_barrier_tool(&tool_call.function.name),
    }
}

fn route_tool_call(mcp_client: &McpClient, tool_name: &str) -> ToolRoute {
    if let Some((server_name, tool_name)) = mcp_client.parse_tool_name_for_known_server(tool_name) {
        ToolRoute::Mcp {
            server_name,
            tool_name,
        }
    } else {
        ToolRoute::Builtin
    }
}

fn parse_tool_args(tool_call: &ToolCall) -> Result<Value, ToolResult> {
    let raw_args = tool_call.function.arguments.trim();
    if raw_args.is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str(raw_args).map_err(|err| ToolResult {
        tool_call_id: tool_call.id.clone(),
        content: format!("Error: failed to parse arguments: {}", err),
    })
}

fn prepare_tool_call(
    mcp_client: &McpClient,
    tool_call: &ToolCall,
) -> Result<PreparedToolCall, ToolResult> {
    Ok(PreparedToolCall {
        route: route_tool_call(mcp_client, &tool_call.function.name),
        args: parse_tool_args(tool_call)?,
    })
}

fn requires_user_confirmation_for_tool(_tool_name: &str) -> bool {
    false
    // let lower = tool_name.to_lowercase();
    // let looks_like_feishu = lower.contains("feishu") || lower.contains("lark");
    // let looks_like_search = lower.contains("search") || lower.contains("docs_search");
    // looks_like_feishu && looks_like_search
}

fn confirm_tool_execution(tool_call: &ToolCall, args: &Value) -> Result<(), RunOneResult> {
    if !requires_user_confirmation_for_tool(&tool_call.function.name) {
        return Ok(());
    }

    let confirm =
        prompt_yes_or_no_interruptible(&format!("Confirm tool execution:{} (y/n): ", args));
    if confirm == Some(true) {
        return Ok(());
    }

    println!("canceled by user.");
    Err(RunOneResult {
        tool_result: ToolResult {
            tool_call_id: tool_call.id.clone(),
            content: if confirm.is_none() {
                format!(
                    "Error: {} canceled by user (Ctrl+C)",
                    tool_call.function.name
                )
            } else {
                format!("Error: {} canceled by user", tool_call.function.name)
            },
        },
        ok: false,
        executed: false,
    })
}

fn remediation_hint(tool_name: &str, err: &str) -> Option<String> {
    let err_lower = err.to_lowercase();

    if tool_name == "mcp_feishu_docs_get_text_by_url" && err_lower.contains("unsupported url") {
        return Some(
            "Suggestion: this tool only works for supported Feishu/Lark docs URLs. Do not retry with the same URL. Use mcp_feishu_docs_search to find the document first, or ask the user for a direct Feishu docs/wiki/sheet URL.".to_string(),
        );
    }

    if err_lower.contains("failed to parse arguments") || err_lower.contains("invalid type") {
        return Some(
            "Suggestion: fix the tool arguments to match the declared JSON schema before retrying."
                .to_string(),
        );
    }

    if err_lower.contains("no such file") || err_lower.contains("not found") {
        return Some(
            "Suggestion: verify the path or identifier first, or use a search/list tool to discover the correct target before retrying.".to_string(),
        );
    }

    if err_lower.contains("timeout") || err_lower.contains("timed out") {
        return Some(
            "Suggestion: retry once with a narrower query or a smaller scope. If it still fails, switch to another tool or ask the user.".to_string(),
        );
    }

    None
}

fn format_tool_error(tool_call: &ToolCall, err: &str) -> ToolResult {
    ToolResult {
        tool_call_id: tool_call.id.clone(),
        content: if let Some(hint) = remediation_hint(&tool_call.function.name, err) {
            format!(
                "Error: {} failed: {}\n{}",
                tool_call.function.name, err, hint
            )
        } else {
            format!("Error: {} failed: {}", tool_call.function.name, err)
        },
    }
}

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

    let exchange_out =
        mcp_client.call_tool(server_name, "oauth_exchange_code", json!({ "code": code }))?;
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

fn execute_mcp_tool_call(
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
            run_feishu_oauth_flow(mcp_client, server_name, &scope)
                .map_err(|e| format!("feishu oauth failed: {}", e))?;
            let content = mcp_client.call_tool(server_name, tool_name, args.clone())?;
            Ok(ToolResult {
                tool_call_id: tool_call.id.clone(),
                content,
            })
        }
        Err(err) => Err(err),
    }
}

fn execute_prepared_tool_call(
    mcp_client: &McpClient,
    tool_call: &ToolCall,
    prepared: &PreparedToolCall,
) -> Result<ToolResult, String> {
    match &prepared.route {
        ToolRoute::Builtin => builtin_tools::execute_tool_call_with_args(
            &tool_call.id,
            &tool_call.function.name,
            &prepared.args,
        )
        .map_err(|e| e.to_string()),
        ToolRoute::Mcp {
            server_name,
            tool_name,
        } => execute_mcp_tool_call(
            mcp_client,
            tool_call,
            server_name,
            tool_name,
            &prepared.args,
        ),
    }
}

fn run_one(mcp_client: &McpClient, tool_call: &ToolCall) -> (ToolRoute, RunOneResult) {
    let prepared = match prepare_tool_call(mcp_client, tool_call) {
        Ok(prepared) => prepared,
        Err(tool_result) => {
            return (
                route_tool_call(mcp_client, &tool_call.function.name),
                RunOneResult {
                    tool_result,
                    ok: false,
                    executed: true,
                },
            );
        }
    };

    if let Err(result) = confirm_tool_execution(tool_call, &prepared.args) {
        return (prepared.route, result);
    }

    let result = execute_prepared_tool_call(mcp_client, tool_call, &prepared);
    let run_result = match result {
        Ok(tool_result) => RunOneResult {
            tool_result,
            ok: true,
            executed: true,
        },
        Err(err) => RunOneResult {
            tool_result: format_tool_error(tool_call, &err),
            ok: false,
            executed: true,
        },
    };

    (prepared.route, run_result)
}

pub(super) fn execute_tool_calls(
    mcp_client: &McpClient,
    tool_calls: &[ToolCall],
) -> Result<ExecuteToolCallsResult, Box<dyn Error>> {
    let mut executed_tool_calls = Vec::with_capacity(tool_calls.len());
    let mut tool_results = Vec::with_capacity(tool_calls.len());

    for (idx, tool_call) in tool_calls.iter().enumerate() {
        let is_last = idx + 1 >= tool_calls.len();
        let (route, run_result) = tokio::task::block_in_place(|| run_one(mcp_client, tool_call));
        let should_barrier = should_barrier_after(
            &route,
            tool_call,
            run_result.ok,
            &run_result.tool_result.content,
        );

        executed_tool_calls.push(tool_call.clone());
        tool_results.push(run_result.tool_result);
        if run_result.executed {
            println!("\n[Executed] {}", tool_call.function.name.green());
        } else {
            println!("\n[Skipped] {}", tool_call.function.name.yellow());
        }

        if should_barrier && !is_last {
            for deferred in &tool_calls[idx + 1..] {
                println!("\n[Deferred] {}", deferred.function.name.yellow());
            }
            break;
        }
    }

    Ok(ExecuteToolCallsResult {
        executed_tool_calls,
        tool_results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_call(name: &str) -> ToolCall {
        ToolCall {
            id: "call-1".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: name.to_string(),
                arguments: "{}".to_string(),
            },
        }
    }

    #[test]
    fn barrier_builtin_search_files_requires_non_empty_success_output() {
        let tc = tool_call("search_files");
        assert!(!should_barrier_after(&ToolRoute::Builtin, &tc, true, "   "));
        assert!(!should_barrier_after(
            &ToolRoute::Builtin,
            &tc,
            false,
            "/tmp/a.rs"
        ));
        assert!(should_barrier_after(
            &ToolRoute::Builtin,
            &tc,
            true,
            "/tmp/a.rs"
        ));
    }

    #[test]
    fn barrier_builtin_and_mcp_rules_match_existing_behavior() {
        assert!(should_barrier_after(
            &ToolRoute::Builtin,
            &tool_call("list_directory"),
            true,
            ""
        ));
        assert!(!should_barrier_after(
            &ToolRoute::Builtin,
            &tool_call("read_file"),
            true,
            "content"
        ));
        assert!(should_barrier_after(
            &ToolRoute::Mcp {
                server_name: "foo".to_string(),
                tool_name: "bar".to_string(),
            },
            &tool_call("mcp_foo_bar"),
            false,
            ""
        ));
    }
}
