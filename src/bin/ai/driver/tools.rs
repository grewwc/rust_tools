use colored::Colorize;
use std::error::Error;

use crate::ai::{mcp::McpClient, types::ToolCall, types::ToolResult};

mod barrier {
    pub fn is_barrier_tool(tool_name: &str) -> bool {
        matches!(tool_name, "search_files" | "list_directory" | "web_search")
    }
}

mod args {
    use serde_json::Value;

    use crate::ai::types::{ToolCall, ToolResult};

    pub fn parse_args(tool_call: &ToolCall) -> Result<Value, ToolResult> {
        let raw_args = tool_call.function.arguments.trim();
        if raw_args.is_empty() {
            return Ok(serde_json::json!({}));
        }
        serde_json::from_str(raw_args).map_err(|err| ToolResult {
            tool_call_id: tool_call.id.clone(),
            content: format!("Error: failed to parse arguments: {}", err),
        })
    }
}

mod dispatch {
    use regex::Regex;
    use serde_json::{Value, json};
    use std::sync::LazyLock;

    use crate::{
        ai::{
            mcp::McpClient,
            tools as builtin_tools,
            types::{ToolCall, ToolResult},
        },
        common::prompt::prompt_yes_or_no_interruptible,
    };

    use super::args;

    fn requires_user_confirmation_for_tool(name: &str) -> bool {
        false
        // let lower = name.to_lowercase();
        // let looks_like_feishu = lower.contains("feishu") || lower.contains("lark");
        // let looks_like_search = lower.contains("search") || lower.contains("docs_search");
        // looks_like_feishu && looks_like_search
    }

    fn looks_like_feishu_oauth_required(err: &str) -> bool {
        let e = err.to_lowercase();
        e.contains("missing user_access_token")
            || e.contains("invalid access token")
            || e.contains("99991668")
            || e.contains("99991679")
            || e.contains("re-authorization")
    }

    fn remediation_hint(tool_name: &str, err: &str) -> Option<String> {
        let err_lower = err.to_lowercase();

        if tool_name == "mcp_feishu_docs_get_text_by_url"
            && err_lower.contains("unsupported url")
        {
            return Some(
                "Suggestion: this tool only works for supported Feishu/Lark docs URLs. Do not retry with the same URL. Use mcp_feishu_docs_search to find the document first, or ask the user for a direct Feishu docs/wiki/sheet URL.".to_string(),
            );
        }

        if err_lower.contains("failed to parse arguments") || err_lower.contains("invalid type") {
            return Some(
                "Suggestion: fix the tool arguments to match the declared JSON schema before retrying.".to_string(),
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

    fn extract_feishu_required_scopes(err: &str) -> Vec<String> {
        static RE: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r"\b[a-z][a-z0-9_-]*:[a-z0-9_-]+(?::[a-z0-9_-]+)*\b").unwrap()
        });
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

    pub struct RunOneResult {
        pub tool_result: ToolResult,
        pub ok: bool,
        pub executed: bool,
    }

    pub fn run_one(mcp_client: &McpClient, tool_call: &ToolCall) -> RunOneResult {
        let name = tool_call.function.name.as_str();

        if requires_user_confirmation_for_tool(name) {
            let args: Value = match args::parse_args(tool_call) {
                Ok(a) => a,
                Err(res) => {
                    return RunOneResult {
                        tool_result: res,
                        ok: false,
                        executed: true,
                    };
                }
            };

            let confirm =
                prompt_yes_or_no_interruptible(&format!("Confirm tool execution:{} (y/n): ", args));
            if confirm != Some(true) {
                println!("canceled by user.");
                return RunOneResult {
                    tool_result: ToolResult {
                        tool_call_id: tool_call.id.clone(),
                        content: if confirm.is_none() {
                            format!("Error: {} canceled by user (Ctrl+C)", name)
                        } else {
                            format!("Error: {} canceled by user", name)
                        },
                    },
                    ok: false,
                    executed: false,
                };
            }
        }

        let result: Result<ToolResult, String> = if let Some((server_name, tool_name)) =
            mcp_client.parse_tool_name_for_known_server(name)
        {
            let args_value: Value = match args::parse_args(tool_call) {
                Ok(a) => a,
                Err(res) => {
                    return RunOneResult {
                        tool_result: res,
                        ok: false,
                        executed: true,
                    };
                }
            };

            match mcp_client.call_tool(&server_name, &tool_name, args_value.clone()) {
                Ok(content) => Ok(ToolResult {
                    tool_call_id: tool_call.id.clone(),
                    content,
                }),
                Err(err) => {
                    let is_oauth_tool = tool_name.starts_with("oauth_");
                    if !is_oauth_tool && looks_like_feishu_oauth_required(&err) {
                        let required = extract_feishu_required_scopes(&err);
                        let scope = if required.is_empty() {
                            "offline_access".to_string()
                        } else {
                            let mut parts = Vec::with_capacity(required.len() + 1);
                            parts.push("offline_access".to_string());
                            parts.extend(required);
                            parts.join(" ")
                        };
                        match run_feishu_oauth_flow(mcp_client, &server_name, &scope) {
                            Ok(()) => {
                                match mcp_client.call_tool(&server_name, &tool_name, args_value) {
                                    Ok(content) => Ok(ToolResult {
                                        tool_call_id: tool_call.id.clone(),
                                        content,
                                    }),
                                    Err(err) => Err(err),
                                }
                            }
                            Err(e) => Err(format!("feishu oauth failed: {}", e)),
                        }
                    } else {
                        Err(err)
                    }
                }
            }
        } else {
            builtin_tools::execute_tool_call(tool_call).map_err(|e| e.to_string())
        };

        match result {
            Ok(res) => RunOneResult {
                tool_result: res,
                ok: true,
                executed: true,
            },
            Err(err) => RunOneResult {
                tool_result: ToolResult {
                    tool_call_id: tool_call.id.clone(),
                    content: if let Some(hint) = remediation_hint(&tool_call.function.name, &err) {
                        format!(
                            "Error: {} failed: {}\n{}",
                            tool_call.function.name, err, hint
                        )
                    } else {
                        format!("Error: {} failed: {}", tool_call.function.name, err)
                    },
                },
                ok: false,
                executed: true,
            },
        }
    }
}

pub(super) struct ExecuteToolCallsResult {
    pub(super) executed_tool_calls: Vec<ToolCall>,
    pub(super) tool_results: Vec<ToolResult>,
}

pub(super) fn execute_tool_calls(
    mcp_client: &McpClient,
    tool_calls: &[ToolCall],
) -> Result<ExecuteToolCallsResult, Box<dyn Error>> {
    let mut executed_tool_calls = Vec::with_capacity(tool_calls.len());
    let mut tool_results = Vec::with_capacity(tool_calls.len());

    for (idx, tool_call) in tool_calls.iter().enumerate() {
        let is_last = idx + 1 >= tool_calls.len();

        let is_mcp_tool = mcp_client
            .parse_tool_name_for_known_server(&tool_call.function.name)
            .is_some();

        let run_res = dispatch::run_one(mcp_client, tool_call);
        let res = run_res.tool_result;
        let ok = run_res.ok;

        let should_barrier_after = if is_mcp_tool {
            true
        } else if tool_call.function.name == "search_files" {
            ok && !res.content.trim().is_empty()
        } else {
            barrier::is_barrier_tool(&tool_call.function.name)
        };

        executed_tool_calls.push(tool_call.clone());
        tool_results.push(res);
        if run_res.executed {
            println!("\n[Executed] {}", tool_call.function.name.green());
        } else {
            println!("\n[Skipped] {}", tool_call.function.name.yellow());
        }

        if should_barrier_after && !is_last {
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
