use std::{
    collections::VecDeque,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use regex::Regex;
use serde_json::{Value, json};
use std::time::Duration;

use super::types::{FunctionDefinition, ToolCall, ToolDefinition, ToolResult};

const HTTP_TOOL_TIMEOUT: Duration = Duration::from_secs(2);

const BUILTIN_TOOLS: &[(&str, &str)] = &[
    (
        "read_file",
        "Read the contents of a file from the local filesystem",
    ),
    (
        "write_file",
        "Write content to a file on the local filesystem",
    ),
    (
        "search_files",
        "Search for files by exact file name or glob pattern",
    ),
    (
        "list_directory",
        "List files and directories in a given path",
    ),
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
                    "description": "Exact file name (preffered) or glob pattern to match. Examples: \"Cargo.toml\", \"*.rs\", \"**/*.md\""
                },
                "path": {
                    "type": "string",
                    "description": "The directory to search in (default: \".\")"
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

fn execute_write_file(args: &Value) -> Result<String, String> {
    let file_path = args["file_path"].as_str().ok_or("Missing file_path")?;
    let content = args["content"].as_str().ok_or("Missing content")?;

    let path = PathBuf::from(file_path);

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
    }

    fs::write(&path, content).map_err(|e| format!("Failed to write file: {}", e))?;

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
            if is_dir { format!("{}/", name) } else { name }
        })
        .collect();

    Ok(entries.join("\n"))
}

fn execute_search_files(args: &Value) -> Result<String, String> {
    let pattern = args["pattern"].as_str().ok_or("Missing pattern")?;
    let path = args["path"].as_str().unwrap_or(".");

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
            return Ok(found.to_string_lossy().trim().to_string());
        }
        return Ok(String::new());
    }

    let matches =
        crate::terminalw::glob_paths(pattern, path).map_err(|e| format!("glob failed: {e}"))?;
    Ok(matches.join("\n").trim().to_string())
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
        "truncate",
        "ssh",
        "scp",
        "rsync",
        "powershell",
        "osascript",
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
    let cwd = args["cwd"].as_str();
    let _timeout = args["timeout"].as_u64().unwrap_or(30);

    if let Err(reason) = validate_execute_command(command) {
        return Ok(format!("Command blocked: {reason}"));
    }

    let output = crate::cmd::run_cmd_output(command, crate::cmd::RunCmdOptions { cwd })
        .map_err(|e| format!("Failed to execute command: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    if output.status.success() {
        Ok(stdout.trim().to_string())
    } else {
        Ok(format!(
            "Exit code: {}\n{}\n{}",
            output.status.code().unwrap_or(-1),
            stdout.trim(),
            stderr.trim()
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

    let result = String::from_utf8_lossy(&output.stdout).to_owned();
    Ok(result.trim().to_string())
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

    let client = reqwest::blocking::Client::builder()
        .timeout(HTTP_TOOL_TIMEOUT)
        .user_agent("Mozilla/5.0 (compatible; rust-tools/1.0)")
        .build()
        .map_err(|e| format!("Failed to build http client: {}", e))?;

    let response = client
        .get(url)
        .send()
        .map_err(|e| format!("Failed to fetch URL: {}", e))?;

    let content = response
        .text()
        .map_err(|e| format!("Failed to read response: {}", e))?;

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
            let window_end = (window_start + 4000).min(html.len());
            let window = &html[window_start..window_end];
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
        let Some(end) = s[i..].find(';') else {
            out.push(b'&');
            i += 1;
            continue;
        };
        let end = i + end;
        let entity = &s[i + 1..end];
        if let Some(decoded) = decode_single_entity(entity) {
            out.extend_from_slice(decoded.as_bytes());
        } else {
            out.push(b'&');
            out.extend_from_slice(entity.as_bytes());
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
            let v = u32::from_str_radix(dec, 10).ok()?;
            char::from_u32(v).map(|c| c.to_string())
        }
        _ => None,
    }
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

#[cfg(test)]
mod tests {
    use crate::cmd;

    use super::*;

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
}
