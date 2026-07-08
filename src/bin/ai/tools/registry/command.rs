use serde_json::Value;

use crate::ai::tools::command_tools::execute_command_streaming;
use crate::ai::tools::common::{ToolRegistration, ToolSpec, ToolStreamingRegistration};
use crate::ai::tools::service::command::execute_command;

fn params_execute_command() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute. Destructive/network/escalation commands are blocked. To start a long-running server (e.g. `flask`/`python app.py`/`npm run dev`), background it and redirect its output to a log file, otherwise it will block until timeout: `python app.py > /tmp/app.log 2>&1 & sleep 2 && curl -s localhost:5566/...`. Never run a foreground server directly."
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

fn execute_command_streaming_registered(
    args: &Value,
    on_chunk: &mut crate::ai::tools::common::ToolStreamWriter<'_>,
) -> Result<String, String> {
    execute_command_streaming(args, |chunk| on_chunk(chunk))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "execute_command",
        description: "Run a shell command with an optional working directory and timeout. Destructive/network/escalation commands are blocked; output is truncated and includes an exit code on failure.",
        parameters: params_execute_command,
        execute: execute_command,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

inventory::submit!(ToolStreamingRegistration {
    name: "execute_command",
    execute_streaming: execute_command_streaming_registered,
});
