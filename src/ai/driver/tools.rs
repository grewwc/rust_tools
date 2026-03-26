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
    use serde_json::Value;

    use crate::{
        ai::{
            mcp::McpClient,
            tools as builtin_tools,
            types::{ToolCall, ToolResult},
        },
        common::prompt::prompt_yes_or_no_interruptible,
    };

    use super::args;

    pub struct RunOneResult {
        pub tool_result: ToolResult,
        pub ok: bool,
        pub executed: bool,
    }

    pub fn run_one(mcp_client: &McpClient, tool_call: &ToolCall) -> RunOneResult {
        let name = tool_call.function.name.as_str();

        if name == "execute_command" {
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
                prompt_yes_or_no_interruptible(&format!("Execute command:{} (y/n): ", args));
            if confirm != Some(true) {
                println!("canceled by user.");
                return RunOneResult {
                    tool_result: ToolResult {
                        tool_call_id: tool_call.id.clone(),
                        content: if confirm.is_none() {
                            "Error: execute_command canceled by user (Ctrl+C)".to_string()
                        } else {
                            "Error: execute_command canceled by user".to_string()
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

            match mcp_client.call_tool(&server_name, &tool_name, args) {
                Ok(content) => Ok(ToolResult {
                    tool_call_id: tool_call.id.clone(),
                    content,
                }),
                Err(err) => Err(err),
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
                    content: format!("Error: {} failed: {}", tool_call.function.name, err),
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
