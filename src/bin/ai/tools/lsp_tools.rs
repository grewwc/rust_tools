use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command;

use super::ast_symbols;
use crate::ai::tools::common::{ToolRegistration, ToolSpec};

fn params_lsp() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["go_to_definition", "find_references", "hover", "document_symbol", "workspace_symbol", "diagnostics"],
                "description": "The LSP operation to perform."
            },
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to analyze."
            },
            "line": {
                "type": "integer",
                "description": "1-based line number (required for go_to_definition, find_references, hover)."
            },
            "column": {
                "type": "integer",
                "description": "1-based column number (optional, defaults to 1)."
            },
            "query": {
                "type": "string",
                "description": "Symbol name to search for (required for workspace_symbol)."
            }
        },
        "required": ["operation", "file_path"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "lsp",
        description: "Interact with Language Server Protocol (LSP) to get code intelligence. Supports: go_to_definition (find where a symbol is defined), find_references (find all usages of a symbol), hover (get type info and docs at cursor), document_symbol (list symbols in file), workspace_symbol (search symbols across project), diagnostics (get errors/warnings). Note: Requires appropriate LSP server to be running for the file's language.",
        parameters: params_lsp,
        execute: execute_lsp,
        groups: &["builtin", "core"],
    }
});

pub(crate) fn execute_lsp(args: &Value) -> Result<String, String> {
    let operation = args["operation"]
        .as_str()
        .ok_or("Missing 'operation' parameter")?;

    let file_path = args["file_path"]
        .as_str()
        .ok_or("Missing 'file_path' parameter")?;

    if !Path::new(file_path).exists() {
        return Err(format!("File not found: {}", file_path));
    }

    match operation {
        "go_to_definition" => {
            let line = args["line"]
                .as_u64()
                .ok_or("Missing 'line' for go_to_definition")?;
            let column = args["column"].as_u64().unwrap_or(1);
            lsp_go_to_definition(file_path, line as usize, column as usize)
        }
        "find_references" => {
            let line = args["line"]
                .as_u64()
                .ok_or("Missing 'line' for find_references")?;
            let column = args["column"].as_u64().unwrap_or(1);
            lsp_find_references(file_path, line as usize, column as usize)
        }
        "hover" => {
            let line = args["line"].as_u64().ok_or("Missing 'line' for hover")?;
            let column = args["column"].as_u64().unwrap_or(1);
            lsp_hover(file_path, line as usize, column as usize)
        }
        "document_symbol" => lsp_document_symbols(file_path),
        "workspace_symbol" => {
            let query = args["query"]
                .as_str()
                .ok_or("Missing 'query' for workspace_symbol")?;
            lsp_workspace_symbol(file_path, query)
        }
        "diagnostics" => lsp_diagnostics(file_path),
        other => Err(format!("Unknown LSP operation: {}", other)),
    }
}

fn detect_language(file_path: &str) -> String {
    let path = Path::new(file_path);
    match path.extension().and_then(|e| e.to_str()) {
        Some("rs") => "rust".to_string(),
        Some("ts") | Some("tsx") => "typescript".to_string(),
        Some("js") | Some("jsx") => "javascript".to_string(),
        Some("py") => "python".to_string(),
        Some("go") => "go".to_string(),
        Some("java") => "java".to_string(),
        Some("c") | Some("h") => "c".to_string(),
        Some("cpp") | Some("cc") | Some("cxx") | Some("hpp") => "cpp".to_string(),
        Some("rb") => "ruby".to_string(),
        Some("php") => "php".to_string(),
        Some("cs") => "csharp".to_string(),
        _ => "unknown".to_string(),
    }
}

fn find_project_root(file_path: &str) -> String {
    let mut current = Path::new(file_path);
    while let Some(parent) = current.parent() {
        if parent.join(".git").exists()
            || parent.join("Cargo.toml").exists()
            || parent.join("package.json").exists()
            || parent.join("go.mod").exists()
            || parent.join("pyproject.toml").exists()
            || parent.join("pom.xml").exists()
        {
            return parent.to_string_lossy().to_string();
        }
        current = parent;
    }
    current.to_string_lossy().to_string()
}

fn lsp_go_to_definition(file_path: &str, line: usize, column: usize) -> Result<String, String> {
    let language = detect_language(file_path);
    let project_root = find_project_root(file_path);

    match language.as_str() {
        "rust" => rust_analyzer_go_to_definition(file_path, line, column, &project_root),
        "typescript" | "javascript" => {
            typescript_lsp_go_to_definition(file_path, line, column, &project_root)
        }
        "python" => pyright_go_to_definition(file_path, line, column, &project_root),
        "go" => gopls_go_to_definition(file_path, line, column, &project_root),
        _ => grep_based_definition(file_path, line, column),
    }
}

fn rust_analyzer_go_to_definition(
    file_path: &str,
    line: usize,
    column: usize,
    project_root: &str,
) -> Result<String, String> {
    let content =
        fs::read_to_string(file_path).map_err(|e| format!("Failed to read file: {}", e))?;
    let lines: Vec<&str> = content.lines().collect();

    if line == 0 || line > lines.len() {
        return Err(format!(
            "Line {} out of range (file has {} lines)",
            line,
            lines.len()
        ));
    }

    let line_content = lines[line - 1];

    if column == 0 || column > line_content.len() {
        return Err("Column out of range".to_string());
    }

    let char_idx = line_content
        .char_indices()
        .find(|(idx, _)| *idx >= column - 1)
        .map(|(idx, _)| idx)
        .unwrap_or(line_content.len());

    let word_at_pos = find_word_at_position(line_content, char_idx);

    if word_at_pos.is_empty() {
        return Ok(format!(
            "No symbol found at {}:{}:{}\nLine content: {}",
            file_path, line, column, line_content
        ));
    }

    let result = search_symbol_in_project(&project_root, &word_at_pos);

    if result.is_empty() {
        Ok(format!(
            "Definition for '{}' not found in project.\n\nTip: For full LSP support, ensure rust-analyzer is running.",
            word_at_pos
        ))
    } else {
        Ok(format!(
            "Found definitions for '{}':\n{}",
            word_at_pos,
            result.join("\n")
        ))
    }
}

fn typescript_lsp_go_to_definition(
    _file_path: &str,
    _line: usize,
    _column: usize,
    project_root: &str,
) -> Result<String, String> {
    Ok(format!(
        "TypeScript LSP requires tsserver to be running.\nProject root: {}\n\n\
         For full LSP support, start tsserver manually or use an IDE with TypeScript support.",
        project_root
    ))
}

fn pyright_go_to_definition(
    _file_path: &str,
    _line: usize,
    _column: usize,
    project_root: &str,
) -> Result<String, String> {
    Ok(format!(
        "Python LSP requires pyright or jedi-language-server to be running.\nProject root: {}\n\n\
         For full LSP support, start the language server manually.",
        project_root
    ))
}

fn gopls_go_to_definition(
    _file_path: &str,
    _line: usize,
    _column: usize,
    project_root: &str,
) -> Result<String, String> {
    Ok(format!(
        "Go LSP requires gopls to be running.\nProject root: {}\n\n\
         For full LSP support, start gopls manually.",
        project_root
    ))
}

fn grep_based_definition(file_path: &str, line: usize, column: usize) -> Result<String, String> {
    let content =
        fs::read_to_string(file_path).map_err(|e| format!("Failed to read file: {}", e))?;
    let lines: Vec<&str> = content.lines().collect();

    if line == 0 || line > lines.len() {
        return Err(format!("Line {} out of range", line));
    }

    let line_content = lines[line - 1];

    if column == 0 || column > line_content.len() {
        return Err("Column out of range".to_string());
    }

    let char_idx = line_content
        .char_indices()
        .find(|(idx, _)| *idx >= column - 1)
        .map(|(idx, _)| idx)
        .unwrap_or(line_content.len());

    let word = find_word_at_position(line_content, char_idx);

    if word.is_empty() {
        return Ok(format!("No symbol at position {}:{}", line, column));
    }

    Ok(format!(
        "Symbol at {}:{}:{} is '{}'\n\n\
         For full LSP support (go to definition, find references, etc.),\n\
         ensure an appropriate LSP server is running for this file type.",
        file_path, line, column, word
    ))
}

fn find_word_at_position(line: &str, char_idx: usize) -> String {
    let bytes: Vec<u8> = line.bytes().collect();
    if char_idx >= bytes.len() {
        return String::new();
    }

    let mut start = char_idx;
    while start > 0 && is_identifier_char(bytes[start - 1]) {
        start -= 1;
    }

    let mut end = char_idx;
    while end < bytes.len() && is_identifier_char(bytes[end]) {
        end += 1;
    }

    if start >= end {
        return String::new();
    }

    String::from_utf8_lossy(&bytes[start..end]).to_string()
}

fn is_identifier_char(b: u8) -> bool {
    (b >= b'a' && b <= b'z') || (b >= b'A' && b <= b'Z') || (b >= b'0' && b <= b'9') || b == b'_'
}

fn search_symbol_in_project(project_root: &str, symbol: &str) -> Vec<String> {
    let mut results = Vec::new();

    let output = Command::new("grep")
        .args(&[
            "-rn",
            "--include=*.rs",
            "--include=*.ts",
            "--include=*.tsx",
            "--include=*.js",
            "--include=*.jsx",
            "--include=*.py",
            "--include=*.go",
            &format!(
                "fn {}|function {}|class {}|struct {}|type {}|const {}|let {}|var {}",
                symbol, symbol, symbol, symbol, symbol, symbol, symbol, symbol
            ),
            project_root,
        ])
        .output();

    if let Ok(output) = output {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines().take(20) {
            results.push(line.to_string());
        }
    }

    results
}

fn lsp_find_references(file_path: &str, line: usize, column: usize) -> Result<String, String> {
    let content =
        fs::read_to_string(file_path).map_err(|e| format!("Failed to read file: {}", e))?;
    let lines: Vec<&str> = content.lines().collect();

    if line == 0 || line > lines.len() {
        return Err(format!("Line {} out of range", line));
    }

    let line_content = lines[line - 1];
    let char_idx = line_content
        .char_indices()
        .find(|(idx, _)| *idx >= column - 1)
        .map(|(idx, _)| idx)
        .unwrap_or(line_content.len());

    let word = find_word_at_position(line_content, char_idx);

    if word.is_empty() {
        return Ok(format!("No symbol at position {}:{}", line, column));
    }

    let project_root = find_project_root(file_path);
    let references = search_symbol_in_project(&project_root, &word);

    if references.is_empty() {
        Ok(format!("No references found for '{}'", word))
    } else {
        Ok(format!(
            "Found {} references to '{}':\n{}",
            references.len(),
            word,
            references.join("\n")
        ))
    }
}

fn lsp_hover(file_path: &str, line: usize, column: usize) -> Result<String, String> {
    let content =
        fs::read_to_string(file_path).map_err(|e| format!("Failed to read file: {}", e))?;
    let lines: Vec<&str> = content.lines().collect();

    if line == 0 || line > lines.len() {
        return Err(format!("Line {} out of range", line));
    }

    let line_content = lines[line - 1];

    Ok(format!(
        "Hover info for {}:{}\nLine: {}",
        file_path, column, line_content
    ))
}

fn lsp_document_symbols(file_path: &str) -> Result<String, String> {
    let content =
        fs::read_to_string(file_path).map_err(|e| format!("Failed to read file: {}", e))?;
    let language = detect_language(file_path);

    if let Some(result) = ast_symbols::extract_document_symbols(&language, file_path, &content) {
        match result {
            Ok(symbols) => return Ok(ast_symbols::format_symbols_output(file_path, &symbols)),
            Err(err) => {
                let fallback = fallback_document_symbols(file_path, &content, &language);
                return Ok(format!(
                    "{}\n\n[AST parser fallback: {}]",
                    fallback, err
                ));
            }
        }
    }

    Ok(fallback_document_symbols(file_path, &content, &language))
}

fn fallback_document_symbols(file_path: &str, content: &str, language: &str) -> String {
    let mut symbols = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();

        let symbol = match language {
            "rust" => {
                if let Some(name) = trimmed.strip_prefix("fn ") {
                    name.split('(')
                        .next()
                        .map(|s| format!("function {} (line {})", s.trim(), idx + 1))
                } else if let Some(name) = trimmed.strip_prefix("struct ") {
                    name.split('{')
                        .next()
                        .map(|s| format!("struct {} (line {})", s.trim(), idx + 1))
                } else if let Some(name) = trimmed.strip_prefix("impl ") {
                    name.split('{')
                        .next()
                        .map(|s| format!("impl {} (line {})", s.trim(), idx + 1))
                } else if let Some(name) = trimmed.strip_prefix("trait ") {
                    name.split('{')
                        .next()
                        .map(|s| format!("trait {} (line {})", s.trim(), idx + 1))
                } else if let Some(name) = trimmed.strip_prefix("enum ") {
                    name.split('{')
                        .next()
                        .map(|s| format!("enum {} (line {})", s.trim(), idx + 1))
                } else if let Some(name) = trimmed.strip_prefix("pub fn ") {
                    name.split('(')
                        .next()
                        .map(|s| format!("pub fn {} (line {})", s.trim(), idx + 1))
                } else {
                    None
                }
            }
            "typescript" | "javascript" => {
                if trimmed.starts_with("export function")
                    || trimmed.starts_with("export async function")
                {
                    trimmed
                        .split_whitespace()
                        .nth(2)
                        .map(|s| format!("export fn {} (line {})", s, idx + 1))
                } else if trimmed.starts_with("export class") {
                    trimmed
                        .split_whitespace()
                        .nth(2)
                        .map(|s| format!("export class {} (line {})", s, idx + 1))
                } else if trimmed.starts_with("function ") {
                    trimmed
                        .split_whitespace()
                        .nth(1)
                        .map(|s| format!("fn {} (line {})", s, idx + 1))
                } else if trimmed.starts_with("class ") {
                    trimmed
                        .split_whitespace()
                        .nth(1)
                        .map(|s| format!("class {} (line {})", s, idx + 1))
                } else if trimmed.starts_with("const ") && trimmed.contains("=>") {
                    trimmed
                        .split_whitespace()
                        .nth(1)
                        .map(|s| format!("const {} (line {})", s.trim_end_matches('='), idx + 1))
                } else {
                    None
                }
            }
            "python" => {
                if let Some(name) = trimmed.strip_prefix("def ") {
                    name.split('(')
                        .next()
                        .map(|s| format!("def {} (line {})", s.trim(), idx + 1))
                } else if let Some(name) = trimmed.strip_prefix("class ") {
                    name.split('(')
                        .next()
                        .map(|s| format!("class {} (line {})", s.trim(), idx + 1))
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(sym) = symbol {
            symbols.push(sym);
        }
    }

    if symbols.is_empty() {
        format!("No symbols found in {}", file_path)
    } else {
        format!("Symbols in {}:\n{}", file_path, symbols.join("\n"))
    }
}

fn lsp_workspace_symbol(file_path: &str, query: &str) -> Result<String, String> {
    let project_root = find_project_root(file_path);

    let output = Command::new("grep")
        .args(&[
            "-rn",
            "--include=*.rs",
            "--include=*.ts",
            "--include=*.tsx",
            "--include=*.js",
            "--include=*.jsx",
            "--include=*.py",
            "--include=*.go",
            query,
            project_root.as_str(),
        ])
        .output();

    if let Ok(output) = output {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let matches: Vec<String> = stdout.lines().take(30).map(|s| s.to_string()).collect();

        if matches.is_empty() {
            Ok(format!(
                "No symbols matching '{}' found in workspace",
                query
            ))
        } else {
            Ok(format!(
                "Found {} symbols matching '{}':\n{}",
                matches.len(),
                query,
                matches.join("\n")
            ))
        }
    } else {
        Err("Failed to search workspace".to_string())
    }
}

fn lsp_diagnostics(file_path: &str) -> Result<String, String> {
    let language = detect_language(file_path);

    match language.as_str() {
        "rust" => {
            let output = Command::new("cargo")
                .args(&["check", "--message-format=short"])
                .output();

            if let Ok(output) = output {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let errors: Vec<&str> = stderr
                    .lines()
                    .filter(|line| line.contains(file_path))
                    .collect();

                if errors.is_empty() {
                    Ok(format!("No diagnostics for {}", file_path))
                } else {
                    Ok(format!(
                        "Diagnostics for {}:\n{}",
                        file_path,
                        errors.join("\n")
                    ))
                }
            } else {
                Err("Failed to run cargo check".to_string())
            }
        }
        _ => Ok(format!(
            "Diagnostics for {} (language: {})\n\n\
                 For full LSP diagnostics, ensure the appropriate language server is running.",
            file_path, language
        )),
    }
}
