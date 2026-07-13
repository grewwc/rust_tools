use serde_json::Value;

use crate::ai::tools::common::{
    ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
    ToolRegistration, ToolSpec, ToolStreamingRegistration,
};
use crate::ai::tools::service::delete::execute_delete_path;
use crate::ai::tools::service::file::{
    execute_read_file, execute_read_file_lines, execute_write_file, execute_write_file_streaming,
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
                "description": "Requested number of lines to read for a broad local excerpt. Prefer this when you already know the file and need more than a tiny snippet."
            }
        },
        "required": ["file_path"]
    })
}

fn params_read_file_lines() -> Value {
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
                "description": "Number of lines to return (1-400; default: 200). Use this after you have already located the relevant line range or symbol."
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
                "description": "When true, write file_path (a relative filename) under the per-session temp directory and register it so it can be cleaned up later via delete_path. Use this for scratch/intermediate files (scripts, data dumps, test fixtures). An absolute path is rejected. Files not created with temp=true cannot be deleted by delete_path. (default: false)"
            }
        },
        "required": ["file_path", "content"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "read_file",
        description: "Read a line-numbered excerpt from a local file (regular files only; directories are not supported; absolute paths only). Prefer this when you already know the file and want a broader local chunk before deciding whether a narrower follow-up read is necessary.",
        parameters: params_read_file,
        execute: execute_read_file,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::Spawnable,
        groups: &["builtin", "core"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "read_file_lines",
        description: "Read line-numbered text from a local file with configurable offset/limit (limit capped at 400). Prefer this only when you already know the relevant region and need a precise line-range read for inspection or patching.",
        parameters: params_read_file_lines,
        execute: execute_read_file_lines,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::Spawnable,
        groups: &["executor", "builtin", "core"],
    }
});

// read_file / read_file_lines 是高精度 grounding 结果：内容复现代价高，禁止
// 有损压缩（只能零压缩外溢到磁盘留指针）；但旧版本一旦被模型连续判定过时，
// 就允许 LLM 裁剪释放上下文——「不可有损压缩」不等于「不可裁剪」。
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "read_file",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Allow,
    },
});

inventory::submit!(ToolHistoryPolicyRegistration {
    name: "read_file_lines",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Allow,
    },
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "write_file",
        description: "Write a file. For scratch/intermediate files (scripts, data dumps, test fixtures that are not part of the project), pass temp=true with a relative filename to write under the per-session temp directory; the file is registered so it can be cleaned up later via delete_path. Without temp=true, this creates a new file or intentionally replaces an entire file at an absolute path — for modifying an existing project file, prefer apply_patch or a minimal localized edit instead of a full rewrite.",
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

fn params_delete_path() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file or directory to delete. Relative paths resolve against the working directory; absolute paths must stay within the sandbox."
            },
            "recursive": {
                "type": "boolean",
                "description": "When true, delete a directory and all its contents. Required for directories; regular files can be deleted without it. (default: false)"
            }
        },
        "required": ["path"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "delete_path",
        description: "Delete a temporary file or directory that was created via write_file(temp=true). Only files registered in the persistent temp-file registry can be deleted — source code, configs, and other project files are always refused. Use this to clean up scratch/intermediate files when done. Files created with write_file(temp=true) are tracked in a JSON registry that survives session restarts. Single-file deletes are undoable; recursive directory deletes are not.",
        parameters: params_delete_path,
        execute: execute_delete_path,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});
