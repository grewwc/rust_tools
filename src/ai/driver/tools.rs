use colored::Colorize;
use serde_json::Value;
use std::error::Error;

use crate::ai::{
    mcp::McpClient,
    tools as builtin_tools,
    types::{ToolCall, ToolResult},
};

pub fn execute_tool_calls(
    mcp_client: &McpClient,
    tool_calls: &[ToolCall],
) -> Result<Vec<ToolResult>, Box<dyn Error>> {
    fn is_barrier_tool(tool_name: &str) -> bool {
        matches!(tool_name, "search_files" | "list_directory" | "web_search")
    }

    fn run_one(mcp_client: &McpClient, tool_call: &ToolCall) -> (ToolResult, bool) {
        let name = tool_call.function.name.as_str();
        let result: Result<ToolResult, String> = if let Some((server_name, tool_name)) =
            mcp_client.parse_tool_name_for_known_server(name)
        {
            let raw_args = tool_call.function.arguments.trim();
            let args: Value = if raw_args.is_empty() {
                serde_json::json!({})
            } else {
                match serde_json::from_str(raw_args) {
                    Ok(a) => a,
                    Err(err) => {
                        return (
                            ToolResult {
                                tool_call_id: tool_call.id.clone(),
                                content: format!("Error: failed to parse arguments: {}", err),
                            },
                            false,
                        );
                    }
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
            Ok(res) => (res, true),
            Err(err) => (
                ToolResult {
                    tool_call_id: tool_call.id.clone(),
                    content: format!("Error: {} failed: {}", tool_call.function.name, err),
                },
                false,
            ),
        }
    }

    let mut out = Vec::with_capacity(tool_calls.len());
    let mut skip_remaining = false;

    for tool_call in tool_calls {
        if skip_remaining {
            out.push(ToolResult {
                tool_call_id: tool_call.id.clone(),
                content: "Skipped: deferred to next iteration for dependency-aware execution"
                    .to_string(),
            });
            println!("\n[Skipped] {}", tool_call.function.name.yellow());
            continue;
        }

        let is_mcp_tool = mcp_client
            .parse_tool_name_for_known_server(&tool_call.function.name)
            .is_some();
        let (res, ok) = run_one(mcp_client, tool_call);
        if is_mcp_tool {
            skip_remaining = true;
        } else if tool_call.function.name == "search_files" {
            skip_remaining = ok && !res.content.trim().is_empty();
        } else if is_barrier_tool(&tool_call.function.name) {
            skip_remaining = true;
        }
        out.push(res);
        println!("\n[Executed] {}", tool_call.function.name.green());
    }

    Ok(out)
}
