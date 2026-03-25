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

mod approval {
    use serde_json::Value;
    use std::io::{self, IsTerminal, Write};

    use crate::ai::types::{ToolCall, ToolResult};

    pub enum ConfirmOutcome {
        Proceed,
        Skip,
        NotInteractive(ToolResult),
        Error(ToolResult),
    }

    pub fn confirm_execute_command(tool_call: &ToolCall, args: &Value) -> ConfirmOutcome {
        let command = args.get("command").and_then(|v| v.as_str()).unwrap_or("");
        let cwd = args.get("cwd").and_then(|v| v.as_str()).unwrap_or(".");

        if !io::stdin().is_terminal() {
            return ConfirmOutcome::NotInteractive(ToolResult {
                tool_call_id: tool_call.id.clone(),
                content: "Command not executed: confirmation required (non-interactive)".to_string(),
            });
        }

        print!(
            "\nExecute command? [y/N]\ncommand: {}\ncwd: {}\n> ",
            command, cwd
        );
        if let Err(err) = io::stdout().flush() {
            return ConfirmOutcome::Error(ToolResult {
                tool_call_id: tool_call.id.clone(),
                content: format!("Error: failed to flush stdout: {}", err),
            });
        }

        let mut line = String::new();
        if let Err(err) = io::stdin().read_line(&mut line) {
            return ConfirmOutcome::Error(ToolResult {
                tool_call_id: tool_call.id.clone(),
                content: format!("Error: failed to read confirmation: {}", err),
            });
        }

        let answer = line.trim().to_lowercase();
        if answer == "y" || answer == "yes" {
            ConfirmOutcome::Proceed
        } else {
            ConfirmOutcome::Skip
        }
    }
}

mod dispatch {
    use serde_json::Value;

    use crate::ai::{mcp::McpClient, tools as builtin_tools, types::ToolCall, types::ToolResult};

    use super::{approval, args};

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

            match approval::confirm_execute_command(tool_call, &args) {
                approval::ConfirmOutcome::Proceed => {}
                approval::ConfirmOutcome::Skip => {
                    return RunOneResult {
                        tool_result: ToolResult {
                            tool_call_id: tool_call.id.clone(),
                            content: "Command cancelled by user".to_string(),
                        },
                        ok: true,
                        executed: false,
                    };
                }
                approval::ConfirmOutcome::NotInteractive(res) => {
                    return RunOneResult {
                        tool_result: res,
                        ok: true,
                        executed: false,
                    };
                }
                approval::ConfirmOutcome::Error(res) => {
                    return RunOneResult {
                        tool_result: res,
                        ok: false,
                        executed: true,
                    };
                }
            }
        }

        let result: Result<ToolResult, String> =
            if let Some((server_name, tool_name)) = mcp_client.parse_tool_name_for_known_server(name)
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

pub fn execute_tool_calls(
    mcp_client: &McpClient,
    tool_calls: &[ToolCall],
) -> Result<Vec<ToolResult>, Box<dyn Error>> {
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

        let run_res = dispatch::run_one(mcp_client, tool_call);
        let res = run_res.tool_result;
        let ok = run_res.ok;

        if is_mcp_tool {
            skip_remaining = true;
        } else if tool_call.function.name == "search_files" {
            skip_remaining = ok && !res.content.trim().is_empty();
        } else if barrier::is_barrier_tool(&tool_call.function.name) {
            skip_remaining = true;
        }

        out.push(res);
        if run_res.executed {
            println!("\n[Executed] {}", tool_call.function.name.green());
        } else {
            println!("\n[Skipped] {}", tool_call.function.name.yellow());
        }
    }

    Ok(out)
}
