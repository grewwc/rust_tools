use serde_json::Value;

use crate::ai::tools::common::{
    ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
    ToolRegistration, ToolSpec, ToolStreamingRegistration,
};
use crate::ai::tools::service::file::{
    execute_read_file, execute_write_file, execute_write_file_streaming,
};

fn params_read_file() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to a regular file to read (directories are not supported; some sensitive paths are blocked)."
            },
            "offset": {
                "type": "integer",
                "description": "1-based line number to start reading from (default: 1)."
            },
            "limit": {
                "type": "integer",
                "description": "Number of lines to read (default: 1000). During discovery, use a large limit for a broad overview; once you have located the relevant region (a symbol or line range), pass a small limit (e.g. 20-40) to read just that slice and avoid pulling unrelated lines. Very large results are additionally capped by a per-read character limit; when that happens the output ends with a truncation notice telling you the exact offset to continue from."
            }
        },
        "required": ["file_path"]
    })
}

fn params_write_file() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to write. When temp=false (default), an absolute path; parent directories are created if missing. When temp=true, a relative filename only (e.g. `script.py`) written under the per-session temp directory — an absolute path is rejected."
            },
            "content": {
                "type": "string",
                "description": "Full file content to write (overwrites existing file)."
            },
            "temp": {
                "type": "boolean",
                "description": "When true, write file_path (a relative filename) under the per-session temp directory. Use this for scratch/intermediate files (scripts, data dumps, test fixtures). An absolute path is rejected. Temp files are automatically cleaned up when the session ends. (default: false)"
            }
        },
        "required": ["file_path", "content"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "read_file",
        description: "Read a line-numbered excerpt from a local file (regular files only; directories are not supported; absolute paths only). Use offset/limit to page: a large limit for a broad overview during discovery, a small limit for a precise line-range read once you know the region you need.",
        parameters: params_read_file,
        execute: execute_read_file,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["executor", "builtin", "core"],
    }
});

// read_file 是高精度 grounding 结果：内容复现代价高，禁止有损压缩（只能零压缩
// 外溢到磁盘留指针）；但旧版本一旦被模型连续判定过时，就允许 LLM 裁剪释放上下文
// ——「不可有损压缩」不等于「不可裁剪」。
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "read_file",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Allow,
        counts_toward_precision_inline_budget: true,
    },
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "write_file",
        description: "Write content to a file, replacing its entire contents (no append, no merge). Returns the absolute path on success.\n- temp=false (default): write to an absolute path; parent directories are created if missing. For modifying an existing file, prefer apply_patch instead of a full rewrite.\n- temp=true: pass a relative filename (e.g. `script.py`); it is written under the per-session temp directory and cleaned up when the session ends. Absolute paths are rejected.",
        parameters: params_write_file,
        execute: execute_write_file,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

inventory::submit!(ToolStreamingRegistration {
    name: "write_file",
    execute_streaming: execute_write_file_streaming,
});


