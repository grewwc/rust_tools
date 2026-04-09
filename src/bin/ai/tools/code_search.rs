use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::ai::tools::ast_structural::execute_structural_search;
use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::lsp_tools::execute_lsp;
use crate::ai::tools::search_tools::execute_search_files;

fn params_code_search() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "find_file",
                    "text_search",
                    "workspace_symbol",
                    "document_symbol",
                    "go_to_definition",
                    "find_references",
                    "hover",
                    "diagnostics",
                    "structural"
                ],
                "description": "High-level code search intent. Prefer this over raw grep/read tools when locating files, symbols, definitions, references, diagnostics, structural code matches, or full-text content hits."
            },
            "file_path": {
                "type": "string",
                "description": "Absolute file path used as the primary LSP anchor. Required for go_to_definition, find_references, hover, document_symbol, diagnostics. Optional for workspace_symbol, structural, and text_search. When provided to text_search, search is limited to this file."
            },
            "path": {
                "type": "string",
                "description": "Root directory for file, text, or structural search. Also used to auto-pick an anchor file for workspace_symbol when file_path is omitted. Defaults to current directory."
            },
            "query": {
                "type": "string",
                "description": "Symbol name, raw tree-sitter structural query, or general search query depending on the operation."
            },
            "intent": {
                "type": "string",
                "enum": [
                    "find_functions",
                    "find_classes",
                    "find_methods",
                    "find_calls"
                ],
                "description": "High-level structural intent for operation=structural. Preferred over raw query when you want common code shapes. Example: use intent=find_calls with call_kind=method_call and receiver=app.view to find method calls on app.view."
            },
            "pattern": {
                "type": "string",
                "description": "Filename or glob pattern for find_file (for example: Cargo.toml, *.rs, **/*.md)."
            },
            "line": {
                "type": "integer",
                "description": "1-based line number for go_to_definition, find_references, and hover."
            },
            "column": {
                "type": "integer",
                "description": "1-based column number for go_to_definition, find_references, and hover. Defaults to 1."
            },
            "file_pattern": {
                "type": "string",
                "description": "Optional file glob restriction for text_search or structural search when searching a directory."
            },
            "name": {
                "type": "string",
                "description": "Optional structural filter. When operation=structural, keep only matches whose @name capture contains this text."
            },
            "contains_text": {
                "type": "string",
                "description": "Optional structural filter. When operation=structural, keep only matches whose captured text contains this substring."
            },
            "call_kind": {
                "type": "string",
                "enum": ["function_call", "method_call", "constructor_call"],
                "description": "Optional structural filter for operation=structural + intent=find_calls. Keeps only matches of the given normalized call kind. Example: call_kind=constructor_call."
            },
            "receiver": {
                "type": "string",
                "description": "Optional structural filter for operation=structural + intent=find_calls. Keeps only matches whose normalized receiver contains this text. Example: receiver=app.view."
            },
            "qualified_name": {
                "type": "string",
                "description": "Optional structural filter for operation=structural + intent=find_calls. Keeps only matches whose normalized qualified_name contains this text. Example: qualified_name=foo.bar.render."
            }
        },
        "required": ["operation"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "code_search",
        description: "High-level code navigation and search tool. Prefer this before raw grep/read tools. Internally routes to LSP for symbol/definition/reference lookups, to search_files for file discovery, to built-in content scanning for full-text search, and to built-in tree-sitter AST search for structural matching. For structural searches, prefer high-level intents like find_functions or find_calls over raw queries, and use name / contains_text / call_kind / receiver / qualified_name to narrow large result sets. Example: operation=structural, intent=find_calls, call_kind=method_call, receiver=app.view, name=render.",
        parameters: params_code_search,
        execute: execute_code_search,
        groups: &["builtin"],
    }
});

pub(crate) fn execute_code_search(args: &Value) -> Result<String, String> {
    let operation = args["operation"]
        .as_str()
        .ok_or("Missing 'operation' parameter")?;

    match operation {
        "find_file" => execute_code_find_file(args),
        "text_search" => execute_code_text_search(args),
        "workspace_symbol" => execute_code_workspace_symbol(args),
        "document_symbol" => execute_code_document_symbol(args),
        "go_to_definition" => execute_code_lsp_with_file(args, "go_to_definition"),
        "find_references" => execute_code_lsp_with_file(args, "find_references"),
        "hover" => execute_code_lsp_with_file(args, "hover"),
        "diagnostics" => execute_code_lsp_with_file(args, "diagnostics"),
        "structural" => execute_code_structural(args),
        other => Err(format!("Unknown code_search operation: {}", other)),
    }
}

fn execute_code_find_file(args: &Value) -> Result<String, String> {
    let pattern = args["pattern"]
        .as_str()
        .or_else(|| args["query"].as_str())
        .ok_or("find_file requires 'pattern' or 'query'")?;
    let path = args["path"].as_str().unwrap_or(".");
    let forwarded = serde_json::json!({
        "pattern": pattern,
        "path": path,
    });
    let result = execute_search_files(&forwarded)?;
    if result.trim().is_empty() {
        Ok(format!(
            "code_search route=search_files operation=find_file\nNo files matched '{}' under '{}'.",
            pattern, path
        ))
    } else {
        Ok(format!(
            "code_search route=search_files operation=find_file\n{}",
            result
        ))
    }
}

fn execute_code_text_search(args: &Value) -> Result<String, String> {
    let query = args["query"]
        .as_str()
        .or_else(|| args["pattern"].as_str())
        .ok_or("text_search requires 'query' or 'pattern'")?;
    let target = text_search_target(args)?;
    let result = execute_content_text_search(query, &target, args["file_pattern"].as_str())?;
    if result.trim().is_empty() {
        Ok(format!(
            "code_search route=content_search operation=text_search\nNo text matches for '{}' under '{}'.",
            query,
            target.display()
        ))
    } else {
        Ok(format!(
            "code_search route=content_search operation=text_search\n{}",
            result
        ))
    }
}

fn text_search_target(args: &Value) -> Result<PathBuf, String> {
    if let Some(file_path) = args["file_path"].as_str() {
        let path = PathBuf::from(file_path);
        if !path.exists() {
            return Err(format!("File not found: {}", file_path));
        }
        if !path.is_file() {
            return Err(format!("text_search file_path is not a file: {}", file_path));
        }
        return Ok(path);
    }

    let path = PathBuf::from(args["path"].as_str().unwrap_or("."));
    if !path.exists() {
        return Err(format!("Path not found: {}", path.display()));
    }
    Ok(path)
}

fn execute_content_text_search(
    query: &str,
    target: &Path,
    file_pattern: Option<&str>,
) -> Result<String, String> {
    if query.trim().is_empty() {
        return Err("text_search query is empty".to_string());
    }

    let files = collect_text_search_files(target, file_pattern)?;
    let mut matches = Vec::new();
    for file in files {
        let Ok(content) = fs::read_to_string(&file) else {
            continue;
        };
        for (idx, line) in content.lines().enumerate() {
            if !line.contains(query) {
                continue;
            }
            matches.push(format!("{}:{}: {}", file.display(), idx + 1, line));
            if matches.len() >= 200 {
                break;
            }
        }
        if matches.len() >= 200 {
            break;
        }
    }

    Ok(truncate_chars(&matches.join("\n"), 16_000))
}

fn collect_text_search_files(target: &Path, file_pattern: Option<&str>) -> Result<Vec<PathBuf>, String> {
    if target.is_file() {
        return Ok(vec![target.to_path_buf()]);
    }
    if !target.is_dir() {
        return Err(format!("text_search target is neither file nor directory: {}", target.display()));
    }

    if let Some(pattern) = file_pattern.filter(|value| !value.trim().is_empty()) {
        let matches = crate::terminalw::glob_paths(pattern, &target.to_string_lossy())
            .map_err(|e| format!("file_pattern glob failed: {}", e))?;
        let files = matches
            .into_iter()
            .map(PathBuf::from)
            .filter(|path| path.is_file())
            .collect::<Vec<_>>();
        return Ok(files);
    }

    let mut files = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(target.to_path_buf());
    let mut scanned_dirs = 0usize;
    let max_dirs = 10_000usize;

    while let Some(dir) = queue.pop_front() {
        scanned_dirs += 1;
        if scanned_dirs > max_dirs {
            break;
        }
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let ft = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            if ft.is_file() && is_text_search_file(&path) {
                files.push(path);
            } else if ft.is_dir() && !ft.is_symlink() {
                queue.push_back(path);
            }
        }
    }
    Ok(files)
}

fn is_text_search_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|s| s.to_str()),
        Some(
            "rs"
                | "ts"
                | "tsx"
                | "js"
                | "jsx"
                | "py"
                | "go"
                | "java"
                | "c"
                | "h"
                | "cpp"
                | "cc"
                | "cxx"
                | "hpp"
                | "rb"
                | "php"
                | "cs"
                | "json"
                | "yaml"
                | "yml"
                | "toml"
                | "md"
                | "txt"
                | "sql"
                | "sh"
        )
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

fn execute_code_workspace_symbol(args: &Value) -> Result<String, String> {
    let query = args["query"]
        .as_str()
        .ok_or("workspace_symbol requires 'query'")?;
    let anchor_file = resolve_anchor_file(args)?;
    let forwarded = serde_json::json!({
        "operation": "workspace_symbol",
        "file_path": anchor_file,
        "query": query,
    });
    let result = execute_lsp(&forwarded)?;
    Ok(format!(
        "code_search route=lsp operation=workspace_symbol\n{}",
        result
    ))
}

fn execute_code_document_symbol(args: &Value) -> Result<String, String> {
    let file_path = require_file_path(args, "document_symbol")?;
    let forwarded = serde_json::json!({
        "operation": "document_symbol",
        "file_path": file_path,
    });
    let result = execute_lsp(&forwarded)?;
    Ok(format!(
        "code_search route=lsp operation=document_symbol\n{}",
        result
    ))
}

fn execute_code_lsp_with_file(args: &Value, operation: &str) -> Result<String, String> {
    let file_path = require_file_path(args, operation)?;
    let mut forwarded = serde_json::json!({
        "operation": operation,
        "file_path": file_path,
    });
    if let Some(query) = args["query"].as_str() {
        forwarded["query"] = Value::String(query.to_string());
    }
    if let Some(line) = args["line"].as_u64() {
        forwarded["line"] = Value::Number(line.into());
    }
    if let Some(column) = args["column"].as_u64() {
        forwarded["column"] = Value::Number(column.into());
    }
    let result = execute_lsp(&forwarded)?;
    Ok(format!(
        "code_search route=lsp operation={}\n{}",
        operation, result
    ))
}

fn execute_code_structural(args: &Value) -> Result<String, String> {
    let target = structural_target(args)?;
    let filters = crate::ai::tools::ast_structural::StructuralFilters {
        name: args["name"].as_str(),
        contains_text: args["contains_text"].as_str(),
        call_kind: args["call_kind"].as_str(),
        receiver: args["receiver"].as_str(),
        qualified_name: args["qualified_name"].as_str(),
    };
    let result = if let Some(intent) = args["intent"].as_str() {
        execute_structural_search(
            crate::ai::tools::ast_structural::StructuralSearch::Intent(intent),
            &target,
            args["file_pattern"].as_str(),
            filters,
        )?
    } else {
        let query = args["query"]
            .as_str()
            .or_else(|| args["pattern"].as_str())
            .ok_or("structural requires 'intent' or 'query' or 'pattern'")?;
        execute_structural_search(
            crate::ai::tools::ast_structural::StructuralSearch::RawQuery(query),
            &target,
            args["file_pattern"].as_str(),
            filters,
        )?
    };
    Ok(format!(
        "code_search route=tree_sitter operation=structural\n{}",
        result
    ))
}

fn require_file_path<'a>(args: &'a Value, operation: &str) -> Result<&'a str, String> {
    let file_path = args["file_path"]
        .as_str()
        .ok_or_else(|| format!("{} requires 'file_path'", operation))?;
    if !Path::new(file_path).exists() {
        return Err(format!("File not found: {}", file_path));
    }
    Ok(file_path)
}

fn structural_target(args: &Value) -> Result<String, String> {
    if let Some(file_path) = args["file_path"].as_str() {
        return Ok(file_path.to_string());
    }
    Ok(args["path"].as_str().unwrap_or(".").to_string())
}

fn resolve_anchor_file(args: &Value) -> Result<String, String> {
    if let Some(file_path) = args["file_path"].as_str() {
        if Path::new(file_path).exists() {
            return Ok(file_path.to_string());
        }
        return Err(format!("File not found: {}", file_path));
    }

    let root = PathBuf::from(args["path"].as_str().unwrap_or("."));
    let anchor = find_code_anchor_file(&root).ok_or_else(|| {
        format!(
            "Could not find a source file under '{}' to use as an LSP workspace anchor.",
            root.display()
        )
    })?;
    Ok(fs::canonicalize(&anchor)
        .unwrap_or(anchor)
        .to_string_lossy()
        .to_string())
}

fn find_code_anchor_file(root: &Path) -> Option<PathBuf> {
    if root.is_file() {
        return is_code_file(root).then(|| root.to_path_buf());
    }
    if !root.exists() || !root.is_dir() {
        return None;
    }

    let mut queue = VecDeque::new();
    queue.push_back(root.to_path_buf());
    let mut scanned_dirs = 0usize;
    let max_dirs = 10_000usize;

    while let Some(dir) = queue.pop_front() {
        scanned_dirs += 1;
        if scanned_dirs > max_dirs {
            return None;
        }
        let entries = fs::read_dir(&dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            let ft = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            if ft.is_file() && is_code_file(&path) {
                return Some(path);
            }
            if ft.is_dir() && !ft.is_symlink() {
                queue.push_back(path);
            }
        }
    }
    None
}

fn is_code_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|s| s.to_str()),
        Some(
            "rs"
                | "ts"
                | "tsx"
                | "js"
                | "jsx"
                | "py"
                | "go"
                | "java"
                | "c"
                | "h"
                | "cpp"
                | "cc"
                | "cxx"
                | "hpp"
                | "rb"
                | "php"
                | "cs"
        )
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("ai_code_search_test_{}_{}", name, uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).expect("failed to create temp dir");
        path
    }

    #[test]
    fn code_search_params_require_operation() {
        let params = params_code_search();
        assert!(params["required"]
            .as_array()
            .unwrap()
            .contains(&Value::String("operation".to_string())));
    }

    #[test]
    fn is_code_file_detects_rust_source() {
        assert!(is_code_file(Path::new("/tmp/foo.rs")));
        assert!(!is_code_file(Path::new("/tmp/foo.txt")));
    }

    #[test]
    fn text_search_uses_file_path_as_single_file_scope() {
        let dir = make_temp_dir("file_path");
        let file = dir.join("sample.rs");
        fs::write(&file, "fn alpha() {}\nlet value = 1;\n").unwrap();

        let args = serde_json::json!({
            "operation": "text_search",
            "file_path": file.to_string_lossy(),
            "query": "fn "
        });
        let result = execute_code_text_search(&args).unwrap();

        assert!(result.contains("route=content_search"));
        assert!(result.contains("sample.rs:1: fn alpha() {}"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_search_honors_path_and_file_pattern_for_content_search() {
        let dir = make_temp_dir("file_pattern");
        let rs = dir.join("keep.rs");
        let txt = dir.join("skip.txt");
        fs::write(&rs, "fn beta() {}\n").unwrap();
        fs::write(&txt, "fn should_not_match() {}\n").unwrap();

        let args = serde_json::json!({
            "operation": "text_search",
            "path": dir.to_string_lossy(),
            "file_pattern": "*.rs",
            "query": "fn "
        });
        let result = execute_code_text_search(&args).unwrap();

        assert!(result.contains("keep.rs:1: fn beta() {}"));
        assert!(!result.contains("skip.txt"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_file_behavior_still_routes_to_search_files() {
        let dir = make_temp_dir("find_file");
        let file = dir.join("Cargo.toml");
        fs::write(&file, "[package]\nname = \"demo\"\n").unwrap();

        let args = serde_json::json!({
            "operation": "find_file",
            "pattern": "Cargo.toml",
            "path": dir.to_string_lossy()
        });
        let result = execute_code_find_file(&args).unwrap();

        assert!(result.contains("code_search route=search_files operation=find_file"));
        assert!(result.contains("Cargo.toml"));

        let _ = fs::remove_dir_all(&dir);
    }
}
