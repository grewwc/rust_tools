use std::{
    fs,
    path::PathBuf,
    process::Command,
};

use serde_json::{Value, json};

use super::types::{FunctionDefinition, ToolDefinition, ToolResult, ToolCall};

const BUILTIN_TOOLS: &[(&str, &str)] = &[
    ("read_file", "Read the contents of a file from the local filesystem"),
    ("write_file", "Write content to a file on the local filesystem"),
    ("list_directory", "List files and directories in a given path"),
    ("search_files", "Search for files matching a pattern"),
    ("execute_command", "Execute a shell command"),
    ("grep_search", "Search for patterns in file contents"),
    ("web_search", "Search the web for information"),
    ("web_fetch", "Fetch content from a URL"),
];

pub(super) fn get_builtin_tool_definitions() -> Vec<ToolDefinition> {
    BUILTIN_TOOLS
        .iter()
        .map(|(name, description)| ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.to_string(),
                description: description.to_string(),
                parameters: get_tool_parameters(name),
            },
        })
        .collect()
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
                    "description": "The glob pattern to match files"
                },
                "path": {
                    "type": "string",
                    "description": "The directory to search in"
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
        _ => json!({"type": "object", "properties": {}}),
    }
}

pub(super) fn execute_tool_call(tool_call: &ToolCall) -> Result<ToolResult, String> {
    let args: Value = serde_json::from_str(&tool_call.function.arguments)
        .map_err(|e| format!("Failed to parse arguments: {}", e))?;

    let result = match tool_call.function.name.as_str() {
        "read_file" => execute_read_file(&args)?,
        "write_file" => execute_write_file(&args)?,
        "list_directory" => execute_list_directory(&args)?,
        "search_files" => execute_search_files(&args)?,
        "execute_command" => execute_command(&args)?,
        "grep_search" => execute_grep_search(&args)?,
        "web_search" => execute_web_search(&args)?,
        "web_fetch" => execute_web_fetch(&args)?,
        name => return Err(format!("Unknown tool: {}", name)),
    };

    Ok(ToolResult {
        tool_call_id: tool_call.id.clone(),
        content: result,
    })
}

fn execute_read_file(args: &Value) -> Result<String, String> {
    let file_path = args["file_path"].as_str().ok_or("Missing file_path")?;
    let path = PathBuf::from(file_path);

    if !path.exists() {
        return Err(format!("File not found: {}", file_path));
    }
    const MAX_NUM_LINES: usize = 10;
    let offset = args["offset"].as_u64().unwrap_or(1) as usize;
    let limit = args["limit"].as_u64().unwrap_or(1000) as usize;

    let content = fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read file: {}", e))?;

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

fn execute_write_file(args: &Value) -> Result<String, String> {
    let file_path = args["file_path"].as_str().ok_or("Missing file_path")?;
    let content = args["content"].as_str().ok_or("Missing content")?;

    let path = PathBuf::from(file_path);

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory: {}", e))?;
    }

    fs::write(&path, content)
        .map_err(|e| format!("Failed to write file: {}", e))?;

    Ok(format!("Successfully wrote to {}", file_path))
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
            if is_dir {
                format!("{}/", name)
            } else {
                name
            }
        })
        .collect();

    Ok(entries.join("\n"))
}

fn execute_search_files(args: &Value) -> Result<String, String> {
    let pattern = args["pattern"].as_str().ok_or("Missing pattern")?;
    let path = args["path"].as_str().unwrap_or(".");

    let output = Command::new("find")
        .arg(path)
        .arg("-name")
        .arg(pattern)
        .output()
        .map_err(|e| format!("Failed to execute find: {}", e))?;

    let result = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(result.trim().to_string())
}

pub(super) fn validate_execute_command(command: &str) -> Result<(), String> {
    let command = command.trim();
    if command.is_empty() {
        return Err("empty command".to_string());
    }

    let forbidden_chars = [
        ';', '|', '&', '>', '<', '\n', '\r', '`', '$', '(', ')', '{', '}', '[', ']', '\\', '"',
        '\'',
    ];
    if command.chars().any(|c| forbidden_chars.contains(&c)) {
        return Err("contains forbidden shell characters".to_string());
    }

    let tokens = command.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return Err("empty command".to_string());
    }

    let program = tokens[0].to_lowercase();

    let denied_programs = [
        "rm",
        "mv",
        "cp",
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
        "tee",
        "truncate",
        "curl",
        "wget",
        "ssh",
        "scp",
        "rsync",
        "git",
        "python",
        "python3",
        "node",
        "npm",
        "pnpm",
        "yarn",
        "bash",
        "sh",
        "zsh",
        "fish",
        "pwsh",
        "powershell",
        "osascript",
        "open",
    ];
    if denied_programs.contains(&program.as_str()) {
        return Err(format!("program '{program}' is blocked"));
    }

    let allowed_programs = [
        "ls", "pwd", "cat", "head", "tail", "wc", "rg", "find", "stat", "du", "df", "uname",
        "whoami", "date", "echo", "printf", "which",
    ];
    if !allowed_programs.contains(&program.as_str()) {
        return Err(format!("program '{program}' is not allowed"));
    }

    let denied_tokens = [
        "-exec", "-delete", "--delete", "--remove", "rm", "mv", "cp", "chmod", "chown", "sudo",
        "ssh", "scp", "rsync", "curl", "wget", "bash", "sh", "zsh", "python", "python3", "node",
        "git",
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
    let cwd = args["cwd"].as_str();
    let _timeout = args["timeout"].as_u64().unwrap_or(30);

    if let Err(reason) = validate_execute_command(command) {
        return Ok(format!("Command blocked: {reason}"));
    }

    let mut cmd = if cfg!(target_os = "windows") {
        let mut c = Command::new("cmd");
        c.args(["/C", command]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", command]);
        c
    };

    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }

    let output = cmd.output()
        .map_err(|e| format!("Failed to execute command: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if output.status.success() {
        Ok(stdout.trim().to_string())
    } else {
        Ok(format!("Exit code: {}\n{}\n{}", 
            output.status.code().unwrap_or(-1), 
            stdout.trim(), 
            stderr.trim()))
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

    let output = cmd.output()
        .map_err(|e| format!("Failed to execute rg: {}", e))?;

    let result = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(result.trim().to_string())
}

fn execute_web_search(args: &Value) -> Result<String, String> {
    let query = args["query"].as_str().ok_or("Missing query")?;
    let _num_results = args["num_results"].as_u64().unwrap_or(5);

    Err(format!("Web search not implemented. Query: {}", query))
}

fn execute_web_fetch(args: &Value) -> Result<String, String> {
    let url = args["url"].as_str().ok_or("Missing url")?;

    let response = reqwest::blocking::get(url)
        .map_err(|e| format!("Failed to fetch URL: {}", e))?;

    let content = response.text()
        .map_err(|e| format!("Failed to read response: {}", e))?;

    Ok(content)
}

pub(super) fn merge_tool_definitions(
    builtin: Vec<ToolDefinition>,
    mcp_tools: Vec<ToolDefinition>,
    skill_tools: Vec<ToolDefinition>,
) -> Vec<ToolDefinition> {
    let mut tools = builtin;
    tools.extend(mcp_tools);
    tools.extend(skill_tools);
    tools
}

pub(super) fn tool_definitions_to_value(tools: &[ToolDefinition]) -> Value {
    serde_json::to_value(tools).unwrap_or(Value::Array(Vec::new()))
}
