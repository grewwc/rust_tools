use serde_json::Value;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};

fn params_compact_context() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "keep_recent_turns": {
                "type": "integer",
                "description": "Number of recent conversation turns to keep uncompressed (default: 5)."
            },
            "summary_max_chars": {
                "type": "integer",
                "description": "Maximum characters for the compressed summary (default: 2000)."
            }
        }
    })
}

fn params_context_status() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {}
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "compact_context",
        description: "Explain the runtime's automatic context compaction behavior. This informational tool does not apply manual compaction or its parameters; the runtime independently summarizes older turns when its token budget is exceeded.",
        parameters: params_compact_context,
        execute: execute_compact_context,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "context_status",
        description: "Show the runtime's context-management policy. Live token usage, message counts, and compression metrics are not available through this tool.",
        parameters: params_context_status,
        execute: execute_context_status,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin"],
    }
});

pub(crate) fn execute_compact_context(args: &Value) -> Result<String, String> {
    let keep_recent = args["keep_recent_turns"].as_u64().unwrap_or(5) as usize;
    let summary_max = args["summary_max_chars"].as_u64().unwrap_or(2000) as usize;

    if keep_recent == 0 {
        return Err("keep_recent_turns must be at least 1".to_string());
    }

    if summary_max < 100 {
        return Err("summary_max_chars must be at least 100".to_string());
    }

    Ok(format!(
        "Context compaction is handled automatically by the runtime when the token budget is exceeded. \
        This tool cannot apply keep_recent_turns={} or summary_max_chars={}; the runtime uses its \
        own mid-turn summarization strategy. To reduce context usage, avoid repeating large file \
        contents in messages.",
        keep_recent, summary_max
    ))
}

pub(crate) fn execute_context_status(_args: &Value) -> Result<String, String> {
    let status = "Context-management policy:\n\
                  - The runtime automatically summarizes older turns when its token budget is exceeded.\n\
                  - Live token usage, message count, and compression metrics are unavailable.\n\
                  - compact_context cannot manually trigger or configure compaction.";

    Ok(status.to_string())
}
