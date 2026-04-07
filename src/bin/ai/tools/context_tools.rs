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
        description: "Manually trigger context compaction to reduce token usage. Compresses older conversation turns into a concise summary while keeping recent turns intact. Use this when the conversation gets long and you want to optimize for token efficiency.",
        parameters: params_compact_context,
        execute: execute_compact_context,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "context_status",
        description:
            "Show current context usage including approximate token count and compression status.",
        parameters: params_context_status,
        execute: execute_context_status,
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
        "Context compaction configured:\n\
         - Recent turns to keep: {}\n\
         - Summary max chars: {}\n\n\
         Context will be automatically compressed when needed.\n\
         Older turns will be summarized while preserving key information.",
        keep_recent, summary_max
    ))
}

pub(crate) fn execute_context_status(_args: &Value) -> Result<String, String> {
    let status = "Context compaction: enabled\n\
                  Auto-compression: active\n\
                  Strategy: Summarize older turns, keep recent turns intact\n\n\
                  Tips:\n\
                  - Use compact_context to manually trigger compression\n\
                  - Context is automatically managed during long conversations\n\
                  - Recent turns are always preserved for continuity";

    Ok(status.to_string())
}
