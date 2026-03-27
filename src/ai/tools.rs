use std::{
    collections::{HashMap, VecDeque},
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use regex::Regex;
use serde_json::{Value, json};
use std::sync::LazyLock;
use std::time::Duration;
use std::{io::Read, process::Stdio, time::Instant};

use super::types::{FunctionDefinition, ToolCall, ToolDefinition, ToolResult};

const HTTP_TOOL_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone, Copy)]
pub(super) struct ToolSpec {
    pub(super) name: &'static str,
    pub(super) description: &'static str,
    pub(super) parameters: fn() -> Value,
    pub(super) execute: fn(&Value) -> Result<String, String>,
    pub(super) groups: &'static [&'static str],
}

pub(super) struct ToolRegistration {
    pub(super) spec: ToolSpec,
}

inventory::collect!(ToolRegistration);

static TOOL_INDEX: LazyLock<HashMap<&'static str, &'static ToolSpec>> = LazyLock::new(|| {
    let mut index: HashMap<&'static str, &'static ToolSpec> = HashMap::new();
    for reg in inventory::iter::<ToolRegistration> {
        index.entry(reg.spec.name).or_insert(&reg.spec);
    }
    index
});

pub(super) fn tool_definitions_for_groups(groups: &[&str]) -> Vec<ToolDefinition> {
    let mut out: Vec<ToolDefinition> = Vec::new();
    for reg in inventory::iter::<ToolRegistration> {
        if !reg
            .spec
            .groups
            .iter()
            .any(|g| groups.iter().any(|x| x == g))
        {
            continue;
        }
        out.push(ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: reg.spec.name.to_string(),
                description: reg.spec.description.to_string(),
                parameters: (reg.spec.parameters)(),
            },
        });
    }
    out.sort_by(|a, b| a.function.name.cmp(&b.function.name));
    out
}

pub(super) fn get_tool_definitions_by_names(names: &[String]) -> Vec<ToolDefinition> {
    let mut out: Vec<ToolDefinition> = Vec::new();
    for name in names {
        let Some(spec) = TOOL_INDEX.get(name.as_str()).copied() else {
            continue;
        };
        out.push(ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: spec.name.to_string(),
                description: spec.description.to_string(),
                parameters: (spec.parameters)(),
            },
        });
    }
    out.sort_by(|a, b| a.function.name.cmp(&b.function.name));
    out
}

pub(super) fn get_builtin_tool_definitions() -> Vec<ToolDefinition> {
    tool_definitions_for_groups(&["builtin"])
}

fn get_tool_parameters(name: &str) -> Value {
    match name {
        "read_file" => json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "The line number to start reading from (1-based)"
                },
                "limit": {
                    "type": "integer",
                    "description": "The number of lines to read"
                }
            },
            "required": ["file_path"]
        }),
        "read_file_lines" => json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "The line number to start reading from (1-based)"
                },
                "limit": {
                    "type": "integer",
                    "description": "The number of lines to read (max: 400)"
                }
            },
            "required": ["file_path"]
        }),
        "write_file" => json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        }),
        "list_directory" => json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute path to the directory to list"
                }
            },
            "required": ["path"]
        }),
        "search_files" => json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Exact file name (preferred) or glob pattern to match. Examples: \"Cargo.toml\", \"*.rs\", \"**/*.md\""
                },
                "path": {
                    "type": "string",
                    "description": "The directory to search in (default: \".\"). Returned paths are absolute."
                }
            },
            "required": ["pattern"]
        }),
        "execute_command" => json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute"
                },
                "cwd": {
                    "type": "string",
                    "description": "The working directory for the command"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds"
                }
            },
            "required": ["command"]
        }),
        "grep_search" => json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "The file or directory to search in"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files"
                }
            },
            "required": ["pattern"]
        }),
        "web_search" => json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return"
                }
            },
            "required": ["query"]
        }),
        "web_fetch" => json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                }
            },
            "required": ["url"]
        }),
        "apply_patch" => json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to patch"
                },
                "patch": {
                    "type": "string",
                    "description": "Unified diff patch content"
                }
            },
            "required": ["file_path", "patch"]
        }),
        "git_status" => json!({
            "type": "object",
            "properties": {
                "cwd": {
                    "type": "string",
                    "description": "Working directory (default: \".\")"
                }
            }
        }),
        "git_diff" => json!({
            "type": "object",
            "properties": {
                "cwd": {
                    "type": "string",
                    "description": "Working directory (default: \".\")"
                },
                "cached": {
                    "type": "boolean",
                    "description": "Diff staged changes"
                },
                "pathspec": {
                    "type": "string",
                    "description": "Optional pathspec, like \"src\" or \"Cargo.toml\""
                }
            }
        }),
        "cargo_check" => json!({
            "type": "object",
            "properties": {
                "cwd": {
                    "type": "string",
                    "description": "Working directory (default: \".\")"
                },
                "workspace": {
                    "type": "boolean",
                    "description": "Run for workspace"
                },
                "all_features": {
                    "type": "boolean",
                    "description": "Enable all features"
                },
                "package": {
                    "type": "string",
                    "description": "Optional package name"
                }
            }
        }),
        "cargo_test" => json!({
            "type": "object",
            "properties": {
                "cwd": {
                    "type": "string",
                    "description": "Working directory (default: \".\")"
                },
                "workspace": {
                    "type": "boolean",
                    "description": "Run for workspace"
                },
                "all_features": {
                    "type": "boolean",
                    "description": "Enable all features"
                },
                "package": {
                    "type": "string",
                    "description": "Optional package name"
                }
            }
        }),
        _ => json!({"type": "object", "properties": {}}),
    }
}

fn params_read_file() -> Value {
    get_tool_parameters("read_file")
}

fn params_write_file() -> Value {
    get_tool_parameters("write_file")
}

fn params_search_files() -> Value {
    get_tool_parameters("search_files")
}

fn params_list_directory() -> Value {
    get_tool_parameters("list_directory")
}

fn params_execute_command() -> Value {
    get_tool_parameters("execute_command")
}

fn params_grep_search() -> Value {
    get_tool_parameters("grep_search")
}

fn params_web_search() -> Value {
    get_tool_parameters("web_search")
}

fn params_web_fetch() -> Value {
    get_tool_parameters("web_fetch")
}

fn params_read_file_lines() -> Value {
    get_tool_parameters("read_file_lines")
}

fn params_apply_patch() -> Value {
    get_tool_parameters("apply_patch")
}

fn params_git_status() -> Value {
    get_tool_parameters("git_status")
}

fn params_git_diff() -> Value {
    get_tool_parameters("git_diff")
}

fn params_cargo_check() -> Value {
    get_tool_parameters("cargo_check")
}

fn params_cargo_test() -> Value {
    get_tool_parameters("cargo_test")
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "read_file",
        description: "Read the contents of a file from the local filesystem",
        parameters: params_read_file,
        execute: execute_read_file,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "write_file",
        description: "Write content to a file on the local filesystem",
        parameters: params_write_file,
        execute: execute_write_file,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "search_files",
        description: "Search for files by exact file name or glob pattern (returns absolute paths)",
        parameters: params_search_files,
        execute: execute_search_files,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "list_directory",
        description: "List files and directories in a given path",
        parameters: params_list_directory,
        execute: execute_list_directory,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "execute_command",
        description: "Execute a shell command",
        parameters: params_execute_command,
        execute: execute_command,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "grep_search",
        description: "Search for patterns in file contents",
        parameters: params_grep_search,
        execute: execute_grep_search,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "web_search",
        description: "Search the web for information",
        parameters: params_web_search,
        execute: execute_web_search,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "web_fetch",
        description: "Fetch content from a URL",
        parameters: params_web_fetch,
        execute: execute_web_fetch,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "read_file_lines",
        description: "Read file contents with configurable line limits",
        parameters: params_read_file_lines,
        execute: execute_read_file_lines,
        groups: &["openclaw"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "apply_patch",
        description: "Apply a unified diff patch to a file",
        parameters: params_apply_patch,
        execute: execute_apply_patch,
        groups: &["openclaw"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "git_status",
        description: "Get git status (porcelain)",
        parameters: params_git_status,
        execute: execute_git_status,
        groups: &["openclaw"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "git_diff",
        description: "Get git diff",
        parameters: params_git_diff,
        execute: execute_git_diff,
        groups: &["openclaw"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "cargo_check",
        description: "Run cargo check",
        parameters: params_cargo_check,
        execute: execute_cargo_check,
        groups: &["openclaw"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "cargo_test",
        description: "Run cargo test",
        parameters: params_cargo_test,
        execute: execute_cargo_test,
        groups: &["openclaw"],
    }
});

pub(super) fn execute_tool_call(tool_call: &ToolCall) -> Result<ToolResult, String> {
    let raw_args = tool_call.function.arguments.trim();
    let args: Value = if raw_args.is_empty() {
        json!({})
    } else {
        serde_json::from_str(raw_args).map_err(|e| format!("Failed to parse arguments: {}", e))?
    };

    let name = tool_call.function.name.as_str();
    let Some(spec) = TOOL_INDEX.get(name).copied() else {
        return Err(format!("Unknown tool: {}", name));
    };
    let result = (spec.execute)(&args)?;

    Ok(ToolResult {
        tool_call_id: tool_call.id.clone(),
        content: result,
    })
}

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

fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out = String::with_capacity(max_chars + 32);
    for (i, ch) in s.chars().enumerate() {
        if i >= max_chars {
            break;
        }
        out.push(ch);
    }
    out.push_str("\n... (truncated)");
    out
}

fn output_to_strings(output: &std::process::Output) -> (String, String) {
    (
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

fn format_exit_code_output(output: &std::process::Output, stdout: &str, stderr: &str) -> String {
    format!(
        "Exit code: {}\n{}\n{}",
        output.status.code().unwrap_or(-1),
        stdout.trim(),
        stderr.trim()
    )
}

fn execute_read_file(args: &Value) -> Result<String, String> {
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
    let start = offset.saturating_sub(1);
    let end = (start + limit).min(lines.len());

    let result: Vec<String> = lines[start..end]
        .iter()
        .take(MAX_NUM_LINES)
        .enumerate()
        .map(|(i, line)| format!("{:>6}\t{}", start + i + 1, line))
        .collect();

    Ok(result.join("\n"))
}

fn execute_read_file_lines(args: &Value) -> Result<String, String> {
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

fn execute_write_file(args: &Value) -> Result<String, String> {
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

#[derive(Debug, Clone)]
struct UnifiedHunk {
    old_start: usize,
    lines: Vec<UnifiedLine>,
}

#[derive(Debug, Clone)]
enum UnifiedLine {
    Context(String),
    Remove(String),
    Add(String),
}

fn parse_unified_hunks(patch: &str) -> Result<Vec<UnifiedHunk>, String> {
    let mut hunks = Vec::new();
    let mut iter = patch.lines().peekable();
    while let Some(line) = iter.next() {
        let Some(rest) = line.strip_prefix("@@") else {
            continue;
        };
        let rest = rest.trim();
        let Some(rest) = rest.strip_prefix('-') else {
            return Err("invalid hunk header".to_string());
        };
        let mut parts = rest.split_whitespace();
        let old_part = parts.next().ok_or("invalid hunk header")?;
        let _new_part = parts.next().ok_or("invalid hunk header")?;

        let old_start = old_part
            .split(',')
            .next()
            .ok_or("invalid hunk header")?
            .parse::<isize>()
            .map_err(|_| "invalid hunk header")?;
        let old_start = if old_start <= 0 {
            0
        } else {
            old_start as usize
        };

        let mut lines = Vec::new();
        while let Some(next) = iter.peek().copied() {
            if next.starts_with("@@") {
                break;
            }
            let l = iter.next().unwrap_or_default();
            if l.starts_with("\\ No newline at end of file") {
                continue;
            }
            let (prefix, body) = l.split_at(1);
            match prefix {
                " " => lines.push(UnifiedLine::Context(body.to_string())),
                "-" => lines.push(UnifiedLine::Remove(body.to_string())),
                "+" => lines.push(UnifiedLine::Add(body.to_string())),
                _ => return Err(format!("invalid hunk line: {}", l)),
            }
        }
        hunks.push(UnifiedHunk { old_start, lines });
    }
    if hunks.is_empty() {
        return Err("no hunks found".to_string());
    }
    Ok(hunks)
}

fn apply_unified_patch(original: &str, patch: &str) -> Result<String, String> {
    let had_trailing_newline = original.ends_with('\n');
    let hunks = parse_unified_hunks(patch)?;
    let orig_lines: Vec<String> = original.lines().map(|s| s.to_string()).collect();

    let mut out: Vec<String> = Vec::new();
    let mut cursor = 0usize;

    for hunk in hunks {
        let apply_at = hunk.old_start.saturating_sub(1);
        if apply_at > orig_lines.len() {
            return Err("hunk start out of range".to_string());
        }
        if apply_at < cursor {
            return Err("hunks out of order".to_string());
        }

        out.extend_from_slice(&orig_lines[cursor..apply_at]);
        let mut idx = apply_at;

        for line in hunk.lines {
            match line {
                UnifiedLine::Context(s) => {
                    let cur = orig_lines.get(idx).ok_or("context out of range")?;
                    if cur != &s {
                        return Err("context mismatch".to_string());
                    }
                    out.push(s);
                    idx += 1;
                }
                UnifiedLine::Remove(s) => {
                    let cur = orig_lines.get(idx).ok_or("remove out of range")?;
                    if cur != &s {
                        return Err("remove mismatch".to_string());
                    }
                    idx += 1;
                }
                UnifiedLine::Add(s) => {
                    out.push(s);
                }
            }
        }

        cursor = idx;
    }

    out.extend_from_slice(&orig_lines[cursor..]);
    let mut s = out.join("\n");
    if had_trailing_newline {
        s.push('\n');
    }
    Ok(s)
}

fn execute_apply_patch(args: &Value) -> Result<String, String> {
    let file_path = args["file_path"].as_str().ok_or("Missing file_path")?;
    let patch = args["patch"].as_str().ok_or("Missing patch")?;

    let path = PathBuf::from(file_path);
    if !path.is_absolute() {
        return Err("file_path must be absolute".to_string());
    }
    if is_sensitive_fs_path(&path) {
        return Err("Access blocked: sensitive path".to_string());
    }
    let original = if path.exists() {
        fs::read_to_string(&path).map_err(|e| format!("Failed to read file: {}", e))?
    } else {
        String::new()
    };
    let next = apply_unified_patch(&original, patch)?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
    }
    fs::write(&path, next).map_err(|e| format!("Failed to write file: {}", e))?;
    Ok(format!("Successfully patched {}", file_path))
}

fn execute_list_directory(args: &Value) -> Result<String, String> {
    let path = args["path"].as_str().ok_or("Missing path")?;
    let dir_path = PathBuf::from(path);

    if !dir_path.exists() {
        return Err(format!("Directory not found: {}", path));
    }

    if !dir_path.is_dir() {
        return Err(format!("Not a directory: {}", path));
    }

    let entries: Vec<_> = fs::read_dir(&dir_path)
        .map_err(|e| format!("Failed to read directory: {}", e))?
        .filter_map(|e| e.ok())
        .map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if is_dir { format!("{}/", name) } else { name }
        })
        .collect();

    Ok(entries.join("\n"))
}

fn execute_search_files(args: &Value) -> Result<String, String> {
    let pattern = args["pattern"].as_str().ok_or("Missing pattern")?;
    let path = args["path"].as_str().unwrap_or(".");

    let cwd = std::env::current_dir().map_err(|e| format!("Failed to get cwd: {}", e))?;
    let base_dir = {
        let p = PathBuf::from(path);
        if p.is_absolute() { p } else { cwd.join(p) }
    };

    let is_exact_name = !pattern.contains('/')
        && !pattern.contains('\\')
        && !pattern.contains('*')
        && !pattern.contains('?')
        && !pattern.contains('[')
        && !pattern.contains(']')
        && !pattern.contains('{')
        && !pattern.contains('}');

    if is_exact_name {
        if let Some(found) = find_first_file_by_name(Path::new(path), pattern) {
            let abs = if found.is_absolute() {
                found
            } else {
                base_dir.join(found)
            };
            let abs = fs::canonicalize(&abs).unwrap_or(abs);
            return Ok(abs.to_string_lossy().trim().to_string());
        }
        return Ok(String::new());
    }

    let matches =
        crate::terminalw::glob_paths(pattern, path).map_err(|e| format!("glob failed: {e}"))?;
    let out: Vec<String> = matches
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            let p = PathBuf::from(s.trim());
            let abs = if p.is_absolute() { p } else { base_dir.join(p) };
            let abs = fs::canonicalize(&abs).unwrap_or(abs);
            abs.to_string_lossy().to_string()
        })
        .collect();
    Ok(out.join("\n").trim().to_string())
}

fn find_first_file_by_name(root: &Path, filename: &str) -> Option<PathBuf> {
    if filename.trim().is_empty() {
        return None;
    }

    if root.is_file() {
        let name = root.file_name().and_then(|s| s.to_str()).unwrap_or("");
        return (name == filename).then_some(root.to_path_buf());
    }

    if !root.is_dir() {
        return None;
    }

    let mut queue = VecDeque::new();
    queue.push_back(root.to_path_buf());

    let mut scanned_dirs = 0usize;
    let max_dirs = 50_000usize;

    while let Some(dir) = queue.pop_front() {
        scanned_dirs += 1;
        if scanned_dirs > max_dirs {
            return None;
        }

        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let file_name = file_name.to_string_lossy();
            let file_name = file_name.as_ref();
            if file_name == filename {
                return Some(entry.path());
            }

            let ft = entry.file_type().ok()?;
            if ft.is_dir() && !ft.is_symlink() {
                queue.push_back(entry.path());
            }
        }
    }

    None
}

pub(super) fn validate_execute_command(command: &str) -> Result<(), String> {
    let command = command.trim();
    if command.is_empty() {
        return Err("empty command".to_string());
    }
    if command.contains('\n') || command.contains('\r') {
        return Err("multi-line command is blocked".to_string());
    }
    if command.contains('|')
        || command.contains('>')
        || command.contains('<')
        || command.contains(';')
        || command.contains('&')
        || command.contains("&&")
        || command.contains("||")
        || command.contains("`")
        || command.contains("$(")
    {
        return Err("shell metacharacters are blocked".to_string());
    }

    let tokens = command.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return Err("empty command".to_string());
    }

    let program = tokens[0].to_lowercase();

    let denied_programs = [
        "fish",
        "jshell",
        "rm",
        "mv",
        "dd",
        "chmod",
        "chown",
        "chgrp",
        "kill",
        "pkill",
        "killall",
        "sudo",
        "su",
        "passwd",
        "shutdown",
        "reboot",
        "launchctl",
        "systemctl",
        "service",
        "diskutil",
        "mount",
        "umount",
        "ln",
        "truncate",
        "ssh",
        "scp",
    ];
    if denied_programs.contains(&program.as_str()) {
        return Err(format!("program '{program}' is blocked"));
    }

    let denied_tokens = [
        "-exec", "-delete", "--delete", "--remove", "rm", "mv", "chmod", "chown", "sudo", "ssh",
        "scp", "rsync",
    ];
    for token in tokens.iter().skip(1) {
        let t = token.to_lowercase();
        if denied_tokens.contains(&t.as_str()) {
            return Err(format!("argument '{t}' is blocked"));
        }
    }

    Ok(())
}

fn execute_command(args: &Value) -> Result<String, String> {
    let command = args["command"].as_str().ok_or("Missing command")?;
    let cwd = args["cwd"].as_str().filter(|dir| !dir.trim().is_empty());
    let timeout = args["timeout"].as_u64().unwrap_or(30).clamp(1, 300);

    if let Err(reason) = validate_execute_command(command) {
        return Ok(format!("Command blocked: {reason}"));
    }

    let mut cmd =
        crate::cmd::run::build_no_shell_command(command, crate::cmd::run::RunCmdOptions { cwd })
            .map_err(|e| format!("Failed to execute command: {}", e))?;
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to execute command: {}", e))?;
    let deadline = Instant::now() + Duration::from_secs(timeout);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok("Command blocked: timeout".to_string());
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => return Err(format!("Failed to execute command: {}", e)),
        }
    }
    let output = child
        .wait_with_output()
        .map_err(|e| format!("Failed to collect command output: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    let stdout_trimmed = stdout.trim();
    let stderr_trimmed = stderr.trim();

    if output.status.success() {
        let combined = if stdout_trimmed.is_empty() {
            stderr_trimmed.to_string()
        } else if stderr_trimmed.is_empty() {
            stdout_trimmed.to_string()
        } else {
            format!("{stdout_trimmed}\n{stderr_trimmed}")
        };
        Ok(truncate_chars(combined.trim(), 16_000))
    } else {
        Ok(truncate_chars(
            &format!(
                "Exit code: {}\n{}\n{}",
                output.status.code().unwrap_or(-1),
                stdout_trimmed,
                stderr_trimmed
            ),
            16_000,
        ))
    }
}

fn execute_grep_search(args: &Value) -> Result<String, String> {
    let pattern = args["pattern"].as_str().ok_or("Missing pattern")?;
    let path = args["path"].as_str().unwrap_or(".");
    let file_pattern = args["file_pattern"].as_str();

    let mut cmd = Command::new("rg");
    cmd.args(["-n", "--color=never", pattern, path]);

    if let Some(fp) = file_pattern {
        cmd.args(["-g", fp]);
    }

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to execute rg: {}", e))?;

    let result = String::from_utf8_lossy(&output.stdout).into_owned();
    Ok(truncate_chars(result.trim(), 16_000))
}

fn execute_git_status(args: &Value) -> Result<String, String> {
    let cwd = args["cwd"].as_str().unwrap_or(".");

    let output = Command::new("git")
        .args(["status", "--porcelain=v1", "--branch"])
        .current_dir(cwd)
        .output()
        .map_err(|e| format!("Failed to execute git: {}", e))?;

    let (stdout, stderr) = output_to_strings(&output);
    if output.status.success() {
        Ok(stdout.trim().to_string())
    } else {
        Ok(format_exit_code_output(&output, &stdout, &stderr))
    }
}

fn execute_git_diff(args: &Value) -> Result<String, String> {
    let cwd = args["cwd"].as_str().unwrap_or(".");
    let cached = args["cached"].as_bool().unwrap_or(false);
    let pathspec = args["pathspec"].as_str().unwrap_or("").trim().to_string();

    let mut cmd = Command::new("git");
    cmd.arg("diff");
    if cached {
        cmd.arg("--cached");
    }
    if !pathspec.is_empty() {
        cmd.arg("--").arg(pathspec);
    }
    let output = cmd
        .current_dir(cwd)
        .output()
        .map_err(|e| format!("Failed to execute git: {}", e))?;

    let (stdout, stderr) = output_to_strings(&output);
    let mut out = if output.status.success() {
        stdout
    } else {
        format_exit_code_output(&output, &stdout, &stderr)
    };

    const MAX_CHARS: usize = 16_000;
    if out.len() > MAX_CHARS {
        out.truncate(MAX_CHARS);
        out.push_str("\n... (truncated)");
    }
    Ok(out.trim().to_string())
}

fn cargo_common_args(args: &Value) -> (String, bool, bool, Option<String>) {
    let cwd = args["cwd"].as_str().unwrap_or(".").to_string();
    let workspace = args["workspace"].as_bool().unwrap_or(true);
    let all_features = args["all_features"].as_bool().unwrap_or(false);
    let package = args["package"].as_str().map(|s| s.trim().to_string());
    let package = package.filter(|s| !s.is_empty());
    (cwd, workspace, all_features, package)
}

fn execute_cargo_command(subcommand: &str, args: &Value) -> Result<String, String> {
    let (cwd, workspace, all_features, package) = cargo_common_args(args);

    let mut cmd = Command::new("cargo");
    cmd.arg(subcommand);
    if workspace {
        cmd.arg("--workspace");
    }
    if all_features {
        cmd.arg("--all-features");
    }
    if let Some(pkg) = package {
        cmd.args(["-p", &pkg]);
    }
    let output = cmd
        .current_dir(cwd)
        .output()
        .map_err(|e| format!("Failed to execute cargo: {}", e))?;

    let (stdout, stderr) = output_to_strings(&output);
    if output.status.success() {
        Ok(format!("{}\n{}", stdout.trim(), stderr.trim())
            .trim()
            .to_string())
    } else {
        Ok(format_exit_code_output(&output, &stdout, &stderr))
    }
}

fn execute_cargo_check(args: &Value) -> Result<String, String> {
    execute_cargo_command("check", args)
}

fn execute_cargo_test(args: &Value) -> Result<String, String> {
    execute_cargo_command("test", args)
}

fn execute_web_search(args: &Value) -> Result<String, String> {
    let query = args["query"].as_str().ok_or("Missing query")?;
    let num_results = args["num_results"]
        .as_u64()
        .or_else(|| args["num"].as_u64())
        .unwrap_or(5)
        .clamp(1, 10) as usize;

    let hits = duckduckgo_search(query, num_results)?;
    if hits.is_empty() {
        return Ok("No results found.".to_string());
    }

    let mut out = String::new();
    for (idx, hit) in hits.into_iter().enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        out.push_str(&format!("{}. {}\n", idx + 1, hit.title.trim()));
        out.push_str(&format!("{}\n", hit.url.trim()));
        if !hit.snippet.trim().is_empty() {
            out.push_str(&format!("{}\n", hit.snippet.trim()));
        }
    }
    Ok(out.trim_end().to_string())
}

fn execute_web_fetch(args: &Value) -> Result<String, String> {
    let url = args["url"].as_str().ok_or("Missing url")?;
    let parsed = reqwest::Url::parse(url).map_err(|_| "Invalid url".to_string())?;
    let scheme = parsed.scheme();
    if scheme != "http" && scheme != "https" {
        return Err("Only http/https urls are allowed".to_string());
    }
    let Some(host) = parsed.host_str() else {
        return Err("Invalid url host".to_string());
    };
    let host_lc = host.to_lowercase();
    if host_lc == "localhost" || host_lc.ends_with(".localhost") || host_lc.ends_with(".local") {
        return Err("Blocked url host".to_string());
    }
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        let blocked = match ip {
            std::net::IpAddr::V4(v4) => {
                v4.is_private() || v4.is_loopback() || v4.is_link_local() || v4.is_multicast()
            }
            std::net::IpAddr::V6(v6) => v6.is_loopback() || v6.is_unique_local(),
        };
        if blocked {
            return Err("Blocked url host".to_string());
        }
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(HTTP_TOOL_TIMEOUT)
        .user_agent("Mozilla/5.0 (compatible; rust-tools/1.0)")
        .build()
        .map_err(|e| format!("Failed to build http client: {}", e))?;

    let response = client
        .get(url)
        .send()
        .map_err(|e| format!("Failed to fetch URL: {}", e))?;

    const MAX_BYTES: usize = 512 * 1024;
    let mut buf = Vec::new();
    response
        .take((MAX_BYTES + 1) as u64)
        .read_to_end(&mut buf)
        .map_err(|e| format!("Failed to read response: {}", e))?;
    if buf.len() > MAX_BYTES {
        buf.truncate(MAX_BYTES);
    }
    let content = String::from_utf8_lossy(&buf).to_string();

    Ok(content)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WebSearchHit {
    title: String,
    url: String,
    snippet: String,
}

fn duckduckgo_search(query: &str, limit: usize) -> Result<Vec<WebSearchHit>, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(HTTP_TOOL_TIMEOUT)
        .user_agent("Mozilla/5.0 (compatible; rust-tools/1.0)")
        .build()
        .map_err(|e| format!("Failed to build http client: {}", e))?;

    let response = client
        .get("https://duckduckgo.com/html/")
        .query(&[("q", query)])
        .send()
        .map_err(|e| format!("Failed to perform web search: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        return Err(format!("Web search failed: HTTP {}", status.as_u16()));
    }

    let html = response
        .text()
        .map_err(|e| format!("Failed to read search response: {}", e))?;
    Ok(parse_duckduckgo_html(&html, limit))
}

fn parse_duckduckgo_html(html: &str, limit: usize) -> Vec<WebSearchHit> {
    let title_re = Regex::new(
        r#"(?s)<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>"#,
    )
    .ok();
    let snippet_re = Regex::new(r#"(?s)<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>|<div[^>]*class="result__snippet"[^>]*>(?P<snippet2>.*?)</div>"#).ok();

    let Some(title_re) = title_re else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for m in title_re.captures_iter(html) {
        if out.len() >= limit {
            break;
        }
        let raw_url = m.name("url").map(|m| m.as_str()).unwrap_or("").to_string();
        let url = normalize_duckduckgo_url(&raw_url);
        let title_html = m.name("title").map(|m| m.as_str()).unwrap_or("");
        let title = clean_html_text(title_html);

        let mut snippet = String::new();
        if let Some(snippet_re) = snippet_re.as_ref() {
            let window_start = m.get(0).map(|m| m.end()).unwrap_or(0);
            let mut window_end = (window_start + 4000).min(html.len());
            while window_end > window_start && !html.is_char_boundary(window_end) {
                window_end -= 1;
            }
            let window = html.get(window_start..window_end).unwrap_or("");
            if let Some(caps) = snippet_re.captures(window) {
                let snippet_html = caps
                    .name("snippet")
                    .or_else(|| caps.name("snippet2"))
                    .map(|m| m.as_str())
                    .unwrap_or("");
                snippet = clean_html_text(snippet_html);
            }
        }

        if title.trim().is_empty() || url.trim().is_empty() {
            continue;
        }
        out.push(WebSearchHit {
            title,
            url,
            snippet,
        });
    }
    out
}

fn normalize_duckduckgo_url(url: &str) -> String {
    let decoded_url = decode_html_entities(url.trim());
    if let Some(decoded) = extract_duckduckgo_uddg(&decoded_url) {
        return decoded;
    }
    decoded_url
}

fn extract_duckduckgo_uddg(url: &str) -> Option<String> {
    let idx = url.find("uddg=")?;
    let rest = &url[idx + 5..];
    let value = rest.split('&').next().unwrap_or(rest);
    let decoded = percent_decode(value)?;
    if decoded.trim().is_empty() {
        None
    } else {
        Some(decoded)
    }
}

fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let h1 = bytes[i + 1];
                let h2 = bytes[i + 2];
                let v1 = hex_value(h1)?;
                let v2 = hex_value(h2)?;
                out.push((v1 << 4) | v2);
                i += 3;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

fn hex_value(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn clean_html_text(s: &str) -> String {
    let without_tags = strip_html_tags(s);
    let decoded = decode_html_entities(&without_tags);
    decoded
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

fn strip_html_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out
}

fn decode_html_entities(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'&' {
            out.push(bytes[i]);
            i += 1;
            continue;
        }
        let mut end = i + 1;
        while end < bytes.len() && bytes[end] != b';' {
            end += 1;
        }
        if end >= bytes.len() {
            out.push(b'&');
            i += 1;
            continue;
        }

        let entity_bytes = &bytes[i + 1..end];
        let decoded = std::str::from_utf8(entity_bytes)
            .ok()
            .and_then(decode_single_entity);

        if let Some(decoded) = decoded {
            out.extend_from_slice(decoded.as_bytes());
        } else {
            out.push(b'&');
            out.extend_from_slice(entity_bytes);
            out.push(b';');
        }
        i = end + 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

fn decode_single_entity(entity: &str) -> Option<String> {
    match entity {
        "amp" => Some("&".to_string()),
        "lt" => Some("<".to_string()),
        "gt" => Some(">".to_string()),
        "quot" => Some("\"".to_string()),
        "apos" => Some("'".to_string()),
        _ if entity.starts_with("#x") || entity.starts_with("#X") => {
            let hex = &entity[2..];
            let v = u32::from_str_radix(hex, 16).ok()?;
            char::from_u32(v).map(|c| c.to_string())
        }
        _ if entity.starts_with('#') => {
            let dec = &entity[1..];
            let v = dec.parse::<u32>().ok()?;
            char::from_u32(v).map(|c| c.to_string())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::cmd;

    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_execute_cmd() {
        let command = json!({"command": "zip -r feishu-aeolus-ltc-exact-match.zip feishu-aeolus-ltc-exact-match"});
        let ret = execute_command(&command);
        println!("ret: {:?}", ret);
        let command = json!({"command": "diff ~/Downloads/1.csv ~/Downloads/2.csv"});
        let ret = execute_command(&command);
        println!("ret2: {:?}", ret);
        let ret = cmd::run_cmd("diff ~/Downloads/1.csv ~/Downloads/2.csv");
        println!("ret3: {:?}", ret);
    }

    #[test]
    fn parse_duckduckgo_html_extracts_title_url_snippet() {
        let html = r#"
        <div class="result results_links results_links_deep web-result">
          <h2 class="result__title">
            <a class="result__a" href="https://example.com/a?x=1&amp;y=2">A &amp; B</a>
          </h2>
          <a class="result__snippet">Hello <b>world</b> &gt; test</a>
        </div>
        <div class="result results_links results_links_deep web-result">
          <h2 class="result__title">
            <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Frust-lang.org%2F">Rust</a>
          </h2>
          <div class="result__snippet">The &quot;Rust&quot; language</div>
        </div>
        "#;

        let hits = parse_duckduckgo_html(html, 5);
        assert_eq!(hits.len(), 2);
        assert_eq!(
            hits[0],
            WebSearchHit {
                title: "A & B".to_string(),
                url: "https://example.com/a?x=1&y=2".to_string(),
                snippet: "Hello world > test".to_string()
            }
        );
        assert_eq!(hits[1].title, "Rust");
        assert_eq!(hits[1].url, "https://rust-lang.org/");
        assert_eq!(hits[1].snippet, "The \"Rust\" language");
    }

    #[test]
    fn decode_html_entities_handles_utf8_without_panicking() {
        let s = "你好 &amp; 标";
        assert_eq!(decode_html_entities(s), "你好 & 标");
    }

    #[test]
    fn parse_duckduckgo_html_does_not_panic_on_utf8_boundaries() {
        let html = format!(
            r#"<div class="result"><a class="result__a" href="https://example.com">Title</a>{}</div>"#,
            "标".repeat(2000)
        );
        let hits = parse_duckduckgo_html(&html, 1);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn search_files_returns_absolute_paths() {
        let tmp = std::env::temp_dir().join(format!(
            "rust_tools_search_files_returns_absolute_paths_{}",
            Uuid::new_v4()
        ));
        fs::create_dir_all(&tmp).unwrap();
        let f = tmp.join("hello.txt");
        fs::write(&f, "x").unwrap();

        let out = execute_search_files(&json!({
            "pattern": "hello.txt",
            "path": tmp.to_string_lossy()
        }))
        .unwrap();
        assert!(std::path::Path::new(out.trim()).is_absolute());

        let out2 = execute_search_files(&json!({
            "pattern": "*.txt",
            "path": tmp.to_string_lossy()
        }))
        .unwrap();
        for line in out2.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            assert!(std::path::Path::new(line).is_absolute());
        }

        let _ = fs::remove_dir_all(&tmp);
    }
}
