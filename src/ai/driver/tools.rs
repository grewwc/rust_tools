use colored::Colorize;
use serde_json::Value;
use std::error::Error;
use std::thread;

use crate::ai::{
    mcp::McpClient,
    tools as builtin_tools,
    types::{ToolCall, ToolResult},
};

pub fn execute_tool_calls(
    mcp_client: &McpClient,
    tool_calls: &[ToolCall],
) -> Result<Vec<ToolResult>, Box<dyn Error>> {
    fn run_one(mcp_client: &McpClient, tool_call: &ToolCall) -> ToolResult {
        let name = tool_call.function.name.as_str();
        let result = match if let Some((server_name, tool_name)) =
            mcp_client.parse_tool_name_for_known_server(name)
        {
            let raw_args = tool_call.function.arguments.trim();
            let args: Value = if raw_args.is_empty() {
                serde_json::json!({})
            } else {
                match serde_json::from_str(raw_args) {
                    Ok(a) => a,
                    Err(err) => {
                        return ToolResult {
                            tool_call_id: tool_call.id.clone(),
                            content: format!("Error: failed to parse arguments: {}", err),
                        };
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
            Err(err) => ToolResult {
                tool_call_id: tool_call.id.clone(),
                content: format!("Error: {} failed: {}", tool_call.function.name, err),
            },
        };
        result
    }

    let mut out = vec![
        ToolResult {
            tool_call_id: String::new(),
            content: String::new(),
        };
        tool_calls.len()
    ];

    thread::scope(|s| {
        let mut handles = Vec::with_capacity(tool_calls.len());
        for (idx, tool_call) in tool_calls.iter().enumerate() {
            let handle = s.spawn(move || run_one(mcp_client, tool_call));
            handles.push((idx, handle));
        }
        for (idx, h) in handles {
            match h.join() {
                Ok(res) => out[idx] = res,
                Err(_) => {
                    out[idx] = ToolResult {
                        tool_call_id: tool_calls
                            .get(idx)
                            .map(|c| c.id.clone())
                            .unwrap_or_default(),
                        content: "Error: tool execution panicked".to_string(),
                    };
                }
            }
        }
    });

    for tool_call in tool_calls {
        println!("\n[Executed] {}", tool_call.function.name.green());
    }

    Ok(out)
}
