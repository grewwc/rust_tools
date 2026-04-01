use serde_json::Value;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::service::command::execute_command;

fn params_execute_command() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute. Destructive/network/escalation commands are blocked."
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory for the command (default: current directory)."
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (1-300; default: 30)."
            }
        },
        "required": ["command"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "execute_command",
        description: "Run a shell command with an optional working directory and timeout. Destructive/network/escalation commands are blocked; output is truncated and includes an exit code on failure.",
        parameters: params_execute_command,
        execute: execute_command,
        groups: &["builtin"],
    }
});
