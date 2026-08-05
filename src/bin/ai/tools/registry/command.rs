use serde_json::Value;

use crate::ai::tools::command_tools::execute_command_streaming;
use crate::ai::tools::common::{
    ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
    ToolRegistration, ToolSpec, ToolStreamingRegistration,
};
use crate::ai::tools::service::command::execute_command;

fn params_execute_command() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run. For long-running servers, background and redirect output: `cmd > /tmp/app.log 2>&1 & sleep 2 && curl -s localhost:PORT/...`. Never run a foreground server directly."
            },
            "cwd": {
                "type": "string",
                "description": "Working directory (default: current directory)."
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds, 1-300 (default: 30)."
            },
            "pty": {
                "type": "boolean",
                "description": "Required. Use true for interactive CLIs (QR codes, prompts, full-screen); false otherwise. No keyboard input forwarding."
            }
        },
        "required": ["command", "pty"]
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
        description: "Run a shell command. Destructive/network/escalation commands are blocked. Output is truncated past a char cap with shown-vs-total counts; narrow or page instead of re-running with different variants. Failures include exit code.",
        parameters: params_execute_command,
        execute: execute_command,
        groups: &["builtin", "core"],
    }
});

// 命令退出状态与编译/测试诊断是 agent 后续修复的直接证据。压缩时必须先落到
// session asset 并留下回读路径，不能把失败日志降级为首行或仅一个 exit code。
// 旧日志仍可由模型主动标记为过时后裁剪，因此这不会让会话上下文单调增长。
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "execute_command",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Allow,
        counts_toward_precision_inline_budget: true,
    },
});

inventory::submit!(ToolStreamingRegistration {
    name: "execute_command",
    execute_streaming: execute_command_streaming_registered,
});
