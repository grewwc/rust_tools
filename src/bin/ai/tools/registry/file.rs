use serde_json::Value;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::service::file::{
    execute_read_file, execute_read_file_lines, execute_write_file,
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
                "description": "Requested number of lines to read; output is capped (currently 10 lines) and may be truncated."
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
                "description": "Number of lines to return (1-400; default: 200)."
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
                "description": "Absolute path to the file to write. Parent directories are created if missing."
            },
            "content": {
                "type": "string",
                "description": "Full file content to write (overwrites existing file)."
            }
        },
        "required": ["file_path", "content"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "read_file",
        description: "Read a small, line-numbered excerpt from a local file (regular files only; directories are not supported; absolute paths only). Use offset/limit to choose a range; output is capped (currently 10 lines) and may be truncated.",
        parameters: params_read_file,
        execute: execute_read_file,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "read_file_lines",
        description: "Read line-numbered text from a local file with configurable offset/limit (limit capped at 400). Prefer this before patching an existing file so edits can target the exact local region instead of rewriting the whole file.",
        parameters: params_read_file_lines,
        execute: execute_read_file_lines,
        groups: &["openclaw", "builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "write_file",
        description: "Create a new file or intentionally replace an entire file at an absolute path. For modifying an existing document or source file, prefer read_file_lines + apply_patch with the smallest localized diff instead of rewriting the whole file.",
        parameters: params_write_file,
        execute: execute_write_file,
        groups: &["builtin"],
    }
});
