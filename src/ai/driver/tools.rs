use colored::Colorize;
use serde_json::Value;
use std::error::Error;

use crate::ai::{
    mcp::McpClient,
    tools as builtin_tools,
    types::{ToolCall, ToolResult},
};

pub fn execute_tool_calls(
    mcp_client: &mut McpClient,
    tool_calls: &[ToolCall],
) -> Result<Vec<ToolResult>, Box<dyn Error>> {
    let mut results = Vec::new();

    for tool_call in tool_calls {
        let result = match if let Some((server_name, tool_name)) =
            mcp_client.parse_tool_name_for_known_server(&tool_call.function.name)
        {
            let raw_args = tool_call.function.arguments.trim();
            let args: Value = if raw_args.is_empty() {
                serde_json::json!({})
            } else {
                match serde_json::from_str(raw_args) {
                    Ok(a) => a,
                    Err(err) => {
                        eprintln!(
                            "[tool error] failed to parse arguments for {}: {}",
                            tool_call.function.name, err
                        );
                        results.push(ToolResult {
                            tool_call_id: tool_call.id.clone(),
                            content: format!("Error: failed to parse arguments: {}", err),
                        });
                        continue;
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
        } {
            Ok(res) => res,
            Err(err) => {
                eprintln!("[tool error] {} failed: {}", tool_call.function.name, err);
                ToolResult {
                    tool_call_id: tool_call.id.clone(),
                    content: format!("Error: {} failed: {}", tool_call.function.name, err),
                }
            }
        };
        println!("\n[Executed] {}", tool_call.function.name.green());
        results.push(result);
    }

    Ok(results)
}
