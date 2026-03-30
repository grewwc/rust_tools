use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;

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
        description: "Read line-numbered text from a local file with configurable offset/limit (limit capped at 400).",
        parameters: params_read_file_lines,
        execute: execute_read_file_lines,
        groups: &["openclaw"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "write_file",
        description: "Create or overwrite a local file at an absolute path. Creates parent directories if missing; access to common secret locations is blocked.",
        parameters: params_write_file,
        execute: execute_write_file,
        groups: &["builtin"],
    }
});

fn is_sensitive_fs_path(path: &Path) -> bool {
    let s = path.to_string_lossy();
    let s = s.as_ref();
    if s.contains("/.ssh/")
        || s.ends_with("/.ssh")
        || s.contains("/.gnupg/")
        || s.ends_with("/.gnupg")
        || s.contains("/.aws/")
        || s.ends_with("/.aws")
        || s.contains("/.kube/")
        || s.ends_with("/.kube")
        || s.contains("/.configW")
        || s.ends_with("/.configW")
    {
        return true;
    }
    let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
        return false;
    };
    matches!(
        name,
        "id_rsa"
            | "id_rsa.pub"
            | "id_ed25519"
            | "id_ed25519.pub"
            | "authorized_keys"
            | "known_hosts"
            | ".netrc"
            | ".npmrc"
            | ".pypirc"
            | ".git-credentials"
            | "credentials"
            | "config.json"
    )
}

pub(crate) fn execute_read_file(args: &Value) -> Result<String, String> {
    let file_path = args["file_path"].as_str().ok_or("Missing file_path")?;
    let path = PathBuf::from(file_path);
    if !path.is_absolute() {
        return Err("file_path must be absolute".to_string());
    }
    if is_sensitive_fs_path(&path) {
        return Err("Access blocked: sensitive path".to_string());
    }

    if !path.exists() {
        return Err(format!("File not found: {}", file_path));
    }
    const MAX_NUM_LINES: usize = 10;
    let offset = args["offset"].as_u64().unwrap_or(1) as usize;
    let limit = args["limit"].as_u64().unwrap_or(1000) as usize;

    let content = fs::read_to_string(&path).map_err(|e| format!("Failed to read file: {}", e))?;

    let lines: Vec<&str> = content.lines().collect();
    let start = offset.saturating_sub(1).min(lines.len());
    let end = (start + limit).min(lines.len());

    let result: Vec<String> = lines[start..end]
        .iter()
        .take(MAX_NUM_LINES)
        .enumerate()
        .map(|(i, line)| format!("{:>6}\t{}", start + i + 1, line))
        .collect();

    Ok(result.join("\n"))
}

pub(crate) fn execute_read_file_lines(args: &Value) -> Result<String, String> {
    let file_path = args["file_path"].as_str().ok_or("Missing file_path")?;
    let path = PathBuf::from(file_path);
    if !path.is_absolute() {
        return Err("file_path must be absolute".to_string());
    }
    if is_sensitive_fs_path(&path) {
        return Err("Access blocked: sensitive path".to_string());
    }

    if !path.exists() {
        return Err(format!("File not found: {}", file_path));
    }
    let offset = args["offset"].as_u64().unwrap_or(1).max(1) as usize;
    let mut limit = args["limit"].as_u64().unwrap_or(200) as usize;
    limit = limit.clamp(1, 400);

    let content = fs::read_to_string(&path).map_err(|e| format!("Failed to read file: {}", e))?;
    let lines: Vec<&str> = content.lines().collect();
    let start = offset.saturating_sub(1);
    if start >= lines.len() {
        return Ok(String::new());
    }
    let end = (start + limit).min(lines.len());

    let result: Vec<String> = lines[start..end]
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{:>6}\t{}", start + i + 1, line))
        .collect();
    Ok(result.join("\n"))
}

pub(crate) fn execute_write_file(args: &Value) -> Result<String, String> {
    let file_path = args["file_path"].as_str().ok_or("Missing file_path")?;
    let content = args["content"].as_str().ok_or("Missing content")?;

    let path = PathBuf::from(file_path);
    if !path.is_absolute() {
        return Err("file_path must be absolute".to_string());
    }
    if is_sensitive_fs_path(&path) {
        return Err("Access blocked: sensitive path".to_string());
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
    }

    fs::write(&path, content).map_err(|e| format!("Failed to write file: {}", e))?;

    Ok(format!("Successfully wrote to {}", file_path))
}
