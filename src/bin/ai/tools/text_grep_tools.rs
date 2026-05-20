use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

use regex::RegexBuilder;
use serde_json::Value;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};

const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "vendor",
    ".mypy_cache",
    ".pytest_cache",
    ".cargo",
];

const MAX_OUTPUT_CHARS: usize = 32_000;
const MAX_FILE_SIZE: u64 = 2 * 1024 * 1024;
const MAX_MATCHES: usize = 200;

fn params_text_grep() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Search pattern (literal substring by default, or regex when is_regex=true)."
            },
            "path": {
                "type": "string",
                "description": "Root directory or file to search in (default: \".\")."
            },
            "file_pattern": {
                "type": "string",
                "description": "Optional glob to filter files (e.g. \"*.rs\", \"*.{ts,tsx}\")."
            },
            "is_regex": {
                "type": "boolean",
                "description": "Treat pattern as a regular expression (default: false)."
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Case-sensitive matching (default: true)."
            },
            "context_lines": {
                "type": "integer",
                "description": "Number of context lines before and after each match (default: 2, max: 5)."
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching lines to return (default: 50, max: 200)."
            }
        },
        "required": ["pattern"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "text_grep",
        description: "Search file contents for a pattern (literal or regex) under a directory. Returns matching lines with file path, line number, and configurable context. Respects .gitignore-style skip directories. Use this when you need to find where specific text or patterns appear in source code.",
        parameters: params_text_grep,
        execute: execute_text_grep,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::Spawnable,
        groups: &["builtin", "core"],
    }
});

pub(crate) fn execute_text_grep(args: &Value) -> Result<String, String> {
    let pattern = args["pattern"]
        .as_str()
        .ok_or("Missing 'pattern' parameter")?;
    if pattern.is_empty() {
        return Err("pattern must not be empty".to_string());
    }

    let path = args["path"].as_str().unwrap_or(".");
    let file_pattern = args["file_pattern"].as_str();
    let is_regex = args["is_regex"].as_bool().unwrap_or(false);
    let case_sensitive = args["case_sensitive"].as_bool().unwrap_or(true);
    let context_lines = args["context_lines"].as_u64().unwrap_or(2).min(5) as usize;
    let max_results = args["max_results"].as_u64().unwrap_or(50).min(MAX_MATCHES as u64) as usize;

    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .map_err(|e| format!("Failed to get cwd: {}", e))?;
    let root = {
        let p = PathBuf::from(path);
        if p.is_absolute() {
            p
        } else {
            cwd.join(p)
        }
    };

    if !root.exists() {
        return Err(format!("Path not found: {}", root.display()));
    }

    let regex = if is_regex {
        RegexBuilder::new(pattern)
            .case_insensitive(!case_sensitive)
            .build()
            .map_err(|e| format!("Invalid regex: {}", e))?
    } else {
        let escaped = regex::escape(pattern);
        RegexBuilder::new(&escaped)
            .case_insensitive(!case_sensitive)
            .build()
            .map_err(|e| format!("Internal regex error: {}", e))?
    };

    let glob_matcher = file_pattern.map(|pat| build_glob_matcher(pat));

    let files = collect_files(&root, &glob_matcher)?;

    let mut results: Vec<MatchResult> = Vec::new();
    let mut total_matches = 0usize;

    for file_path in &files {
        if total_matches >= max_results {
            break;
        }

        let metadata = match fs::metadata(file_path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if metadata.len() > MAX_FILE_SIZE {
            continue;
        }

        let content = match fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let lines: Vec<String> = content.lines().map(|l| l.to_string()).collect();
        let mut file_matches: Vec<LineMatch> = Vec::new();

        for (idx, line) in lines.iter().enumerate() {
            if total_matches >= max_results {
                break;
            }
            if regex.is_match(line) {
                file_matches.push(LineMatch {
                    line_number: idx + 1,
                    line_index: idx,
                });
                total_matches += 1;
            }
        }

        if !file_matches.is_empty() {
            let display_path = file_path
                .strip_prefix(&cwd)
                .unwrap_or(file_path)
                .to_string_lossy()
                .to_string();

            results.push(MatchResult {
                file: display_path,
                matches: file_matches,
                lines,
                context_lines,
            });
        }
    }

    if results.is_empty() {
        return Ok("No matches found.".to_string());
    }

    let output = format_results(&results, total_matches, max_results);
    Ok(truncate_output(&output, MAX_OUTPUT_CHARS))
}

struct MatchResult {
    file: String,
    matches: Vec<LineMatch>,
    lines: Vec<String>,
    context_lines: usize,
}

struct LineMatch {
    line_number: usize,
    line_index: usize,
}

fn format_results(results: &[MatchResult], total_matches: usize, max_results: usize) -> String {
    let mut out = String::new();
    let file_count = results.len();

    out.push_str(&format!(
        "{} match(es) in {} file(s)",
        total_matches, file_count
    ));
    if total_matches >= max_results {
        out.push_str(" (limit reached, more matches may exist)");
    }
    out.push('\n');

    for result in results {
        out.push('\n');
        out.push_str(&result.file);
        out.push('\n');

        let ranges = merge_context_ranges(&result.matches, result.context_lines, result.lines.len());

        for range in &ranges {
            if range.start > 0 {
                out.push_str("...\n");
            }
            for idx in range.start..range.end {
                let line_num = idx + 1;
                let is_match = result.matches.iter().any(|m| m.line_index == idx);
                let prefix = if is_match { ">" } else { " " };
                let empty = String::new();
                let line_content = result.lines.get(idx).unwrap_or(&empty);
                out.push_str(&format!("{}{:>5}| {}\n", prefix, line_num, line_content));
            }
        }
    }

    out
}

struct LineRange {
    start: usize,
    end: usize,
}

fn merge_context_ranges(matches: &[LineMatch], context: usize, total_lines: usize) -> Vec<LineRange> {
    if matches.is_empty() {
        return Vec::new();
    }

    let mut ranges: Vec<LineRange> = Vec::new();

    for m in matches {
        let start = m.line_index.saturating_sub(context);
        let end = (m.line_index + context + 1).min(total_lines);

        if let Some(last) = ranges.last_mut() {
            if start <= last.end {
                last.end = last.end.max(end);
                continue;
            }
        }
        ranges.push(LineRange { start, end });
    }

    ranges
}

fn build_glob_matcher(pattern: &str) -> GlobMatcher {
    GlobMatcher {
        patterns: pattern
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect(),
    }
}

struct GlobMatcher {
    patterns: Vec<String>,
}

impl GlobMatcher {
    fn matches(&self, file_name: &str) -> bool {
        if self.patterns.is_empty() {
            return true;
        }
        for pat in &self.patterns {
            if glob_match_simple(pat, file_name) {
                return true;
            }
        }
        false
    }
}

fn glob_match_simple(pattern: &str, name: &str) -> bool {
    let pat = pattern.trim_start_matches("**/");
    if pat.starts_with("*.") {
        let ext = &pat[1..];
        return name.ends_with(ext);
    }
    if pat.contains('*') || pat.contains('?') {
        let parts: Vec<&str> = pat.split('*').collect();
        if parts.len() == 2 {
            return name.starts_with(parts[0]) && name.ends_with(parts[1]);
        }
    }
    name == pat || name.ends_with(pat)
}

fn collect_files(root: &Path, glob_matcher: &Option<GlobMatcher>) -> Result<Vec<PathBuf>, String> {
    if root.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }

    let mut files = Vec::new();
    let mut queue: VecDeque<PathBuf> = VecDeque::new();
    queue.push_back(root.to_path_buf());

    let max_files = 10_000usize;

    while let Some(dir) = queue.pop_front() {
        if files.len() >= max_files {
            break;
        }

        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            if path.is_dir() {
                if SKIP_DIRS.contains(&name_str.as_ref()) || name_str.starts_with('.') {
                    continue;
                }
                queue.push_back(path);
            } else if path.is_file() {
                if name_str.starts_with('.') {
                    continue;
                }
                if let Some(matcher) = glob_matcher {
                    if !matcher.matches(&name_str) {
                        continue;
                    }
                }
                files.push(path);
            }
        }
    }

    files.sort();
    Ok(files)
}

fn truncate_output(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }
    let mut out = String::with_capacity(max_chars + 32);
    for (i, ch) in s.chars().enumerate() {
        if i >= max_chars {
            break;
        }
        out.push(ch);
    }
    out.push_str("\n... (output truncated)");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("ai_text_grep_test_{}_{}", name, uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).expect("failed to create temp dir");
        path
    }

    #[test]
    fn test_text_grep_literal_match() {
        let dir = make_temp_dir("literal");
        fs::write(
            dir.join("hello.rs"),
            "fn main() {\n    println!(\"hello world\");\n}\n",
        )
        .unwrap();

        let args = serde_json::json!({
            "pattern": "hello world",
            "path": dir.to_string_lossy().to_string()
        });
        let result = execute_text_grep(&args);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("hello.rs"), "should show file name");
        assert!(output.contains("hello world"), "should show matched content");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_text_grep_regex_match() {
        let dir = make_temp_dir("regex");
        fs::write(
            dir.join("test.py"),
            "def foo():\n    return 42\n\ndef bar():\n    return 99\n",
        )
        .unwrap();

        let args = serde_json::json!({
            "pattern": "def \\w+\\(",
            "path": dir.to_string_lossy().to_string(),
            "is_regex": true
        });
        let result = execute_text_grep(&args);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("def foo()"), "should find foo");
        assert!(output.contains("def bar()"), "should find bar");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_text_grep_file_pattern_filter() {
        let dir = make_temp_dir("filter");
        fs::write(dir.join("code.rs"), "fn hello() {}\n").unwrap();
        fs::write(dir.join("readme.md"), "hello docs\n").unwrap();

        let args = serde_json::json!({
            "pattern": "hello",
            "path": dir.to_string_lossy().to_string(),
            "file_pattern": "*.rs"
        });
        let result = execute_text_grep(&args);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("code.rs"), "should find in .rs file");
        assert!(!output.contains("readme.md"), "should skip .md file");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_text_grep_case_insensitive() {
        let dir = make_temp_dir("case");
        fs::write(dir.join("test.txt"), "Hello World\nhello world\nHELLO WORLD\n").unwrap();

        let args = serde_json::json!({
            "pattern": "hello world",
            "path": dir.to_string_lossy().to_string(),
            "case_sensitive": false
        });
        let result = execute_text_grep(&args);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("3 match(es)"), "should find all 3 variants");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_text_grep_no_matches() {
        let dir = make_temp_dir("nomatch");
        fs::write(dir.join("test.txt"), "nothing special here\n").unwrap();

        let args = serde_json::json!({
            "pattern": "nonexistent_xyz_42",
            "path": dir.to_string_lossy().to_string()
        });
        let result = execute_text_grep(&args);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "No matches found.");

        let _ = fs::remove_dir_all(&dir);
    }
}
