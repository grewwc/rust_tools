use serde_json::Value;
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::ast_symbols::{self, SymbolEntry};
use crate::ai::tools::common::{ToolRegistration, ToolSpec};

const MAX_LSP_FILES: usize = 10_000;
const MAX_LSP_FILE_SIZE: u64 = 2 * 1024 * 1024;
const MAX_SYMBOL_HITS: usize = 60;
const MAX_REFERENCE_HITS: usize = 80;

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
                "description": "Absolute path to the file to analyze. Required for hover, document_symbol, diagnostics; used as the project anchor for go_to_definition/find_references/workspace_symbol."
            },
            "line": {
                "type": "integer",
                "description": "1-based line number (required for go_to_definition, find_references, hover unless 'query' is given)."
            },
            "column": {
                "type": "integer",
                "description": "1-based column number (optional, defaults to 1)."
            },
            "query": {
                "type": "string",
                "description": "Symbol name to search for (required for workspace_symbol; optional alternative to line/column for go_to_definition/find_references)."
            }
        },
        "required": ["operation", "file_path"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "lsp",
        description: "Code intelligence backed by tree-sitter AST analysis (no external language server required). Supports: go_to_definition (locate where a symbol is defined), find_references (find usages of a symbol across the project), hover (symbol kind and the line at a position), document_symbol (list symbols in a file), workspace_symbol (search symbols across the project), diagnostics (compiler diagnostics; Rust via cargo check). Works for Rust, TypeScript, JavaScript, Python, Go, Java, C, and C++.",
        parameters: params_lsp,
        execute: execute_lsp,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::Spawnable,
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
            let symbol = resolve_target_symbol(args, file_path)?;
            lsp_go_to_definition(file_path, &symbol)
        }
        "find_references" => {
            let symbol = resolve_target_symbol(args, file_path)?;
            lsp_find_references(file_path, &symbol)
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

/// 解析 go_to_definition / find_references 的目标符号：
/// 优先用显式 `query`，否则从 `line`/`column` 处取标识符。
fn resolve_target_symbol(args: &Value, file_path: &str) -> Result<String, String> {
    if let Some(query) = args["query"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return Ok(query.to_string());
    }

    let line = args["line"]
        .as_u64()
        .ok_or("Provide 'query', or 'line' (with optional 'column'), to identify the symbol")?
        as usize;
    let column = args["column"].as_u64().unwrap_or(1) as usize;

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
    let char_idx = byte_index_for_column(line_content, column);
    let word = find_word_at_position(line_content, char_idx);
    if word.is_empty() {
        return Err(format!(
            "No identifier found at {}:{}:{} (line: {})",
            file_path, line, column, line_content
        ));
    }
    Ok(word)
}

fn detect_language(file_path: &str) -> String {
    ast_symbols::language_for_path(file_path)
        .unwrap_or("unknown")
        .to_string()
}

fn find_project_root(file_path: &str) -> PathBuf {
    let mut current = Path::new(file_path);
    while let Some(parent) = current.parent() {
        if parent.join(".git").exists()
            || parent.join("Cargo.toml").exists()
            || parent.join("package.json").exists()
            || parent.join("go.mod").exists()
            || parent.join("pyproject.toml").exists()
            || parent.join("pom.xml").exists()
        {
            return parent.to_path_buf();
        }
        current = parent;
    }
    current.to_path_buf()
}

/// 一个项目级符号命中：文件、行号、符号种类、符号名。
struct SymbolHit {
    file: PathBuf,
    line: usize,
    kind: &'static str,
    name: String,
}

/// 扫描项目内所有 AST 可解析的源文件，收集符号名与 `query` 匹配的定义。
/// `exact` 为 true 时要求符号名完全相等，否则按子串匹配（大小写不敏感）。
fn collect_workspace_symbols(root: &Path, query: &str, exact: bool) -> Vec<SymbolHit> {
    let needle = query.to_lowercase();
    let files = collect_source_files(root);
    let mut hits = Vec::new();

    for file in files {
        if hits.len() >= MAX_SYMBOL_HITS {
            break;
        }
        let Some(language) = ast_symbols::language_for_path(&file.to_string_lossy()) else {
            continue;
        };
        let Ok(metadata) = fs::metadata(&file) else {
            continue;
        };
        if metadata.len() > MAX_LSP_FILE_SIZE {
            continue;
        }
        let Ok(content) = fs::read_to_string(&file) else {
            continue;
        };
        let Some(Ok(symbols)) =
            ast_symbols::extract_document_symbols(language, &file.to_string_lossy(), &content)
        else {
            continue;
        };
        for symbol in symbols {
            if symbol_matches(&symbol, &needle, exact) {
                hits.push(SymbolHit {
                    file: file.clone(),
                    line: symbol.line,
                    kind: symbol.kind,
                    name: symbol.name.clone(),
                });
                if hits.len() >= MAX_SYMBOL_HITS {
                    break;
                }
            }
        }
    }

    hits.sort_by(|a, b| a.file.cmp(&b.file).then_with(|| a.line.cmp(&b.line)));
    hits
}

fn symbol_matches(symbol: &SymbolEntry, needle_lower: &str, exact: bool) -> bool {
    let name = symbol.name.to_lowercase();
    if exact {
        name == needle_lower
    } else {
        name.contains(needle_lower)
    }
}

fn lsp_workspace_symbol(file_path: &str, query: &str) -> Result<String, String> {
    let query = query.trim();
    if query.is_empty() {
        return Err("workspace_symbol query is empty".to_string());
    }
    let root = find_project_root(file_path);
    let hits = collect_workspace_symbols(&root, query, false);

    if hits.is_empty() {
        return Ok(format!(
            "No symbols matching '{}' found in workspace",
            query
        ));
    }

    Ok(format!(
        "Found {} symbol(s) matching '{}':\n{}",
        hits.len(),
        query,
        render_symbol_hits(&hits, &root)
    ))
}

fn lsp_go_to_definition(file_path: &str, symbol: &str) -> Result<String, String> {
    let root = find_project_root(file_path);
    // 先精确匹配定义；没有再退回子串匹配，避免完全无结果。
    let mut hits = collect_workspace_symbols(&root, symbol, true);
    let exact = !hits.is_empty();
    if hits.is_empty() {
        hits = collect_workspace_symbols(&root, symbol, false);
    }

    if hits.is_empty() {
        return Ok(format!("Definition for '{}' not found in project.", symbol));
    }

    let header = if exact {
        format!("Found definition(s) for '{}':", symbol)
    } else {
        format!(
            "No exact definition for '{}'; showing symbols whose name contains it:",
            symbol
        )
    };
    Ok(format!("{}\n{}", header, render_symbol_hits(&hits, &root)))
}

fn lsp_find_references(file_path: &str, symbol: &str) -> Result<String, String> {
    let root = find_project_root(file_path);

    // 定义位置（若能用 AST 定位）。
    let definitions = collect_workspace_symbols(&root, symbol, true);

    // 引用：跨项目按全词标识符扫描。
    let references = scan_identifier_references(&root, symbol);

    if references.is_empty() && definitions.is_empty() {
        return Ok(format!("No references found for '{}'", symbol));
    }

    let mut out = String::new();
    if !definitions.is_empty() {
        out.push_str(&format!("Definition(s) for '{}':\n", symbol));
        out.push_str(&render_symbol_hits(&definitions, &root));
        out.push('\n');
    }
    out.push_str(&format!(
        "Found {} reference line(s) to '{}':\n{}",
        references.len(),
        symbol,
        references.join("\n")
    ));
    Ok(out)
}

/// 跨项目对一个标识符做全词扫描，返回 `relpath:line: trimmed-content` 列表。
fn scan_identifier_references(root: &Path, symbol: &str) -> Vec<String> {
    let files = collect_source_files(root);
    let mut out = Vec::new();

    for file in files {
        if out.len() >= MAX_REFERENCE_HITS {
            break;
        }
        let Ok(metadata) = fs::metadata(&file) else {
            continue;
        };
        if metadata.len() > MAX_LSP_FILE_SIZE {
            continue;
        }
        let Ok(content) = fs::read_to_string(&file) else {
            continue;
        };
        let rel = file.strip_prefix(root).unwrap_or(&file);
        for (idx, line) in content.lines().enumerate() {
            if !line_contains_whole_word(line, symbol) {
                continue;
            }
            out.push(format!("{}:{}: {}", rel.display(), idx + 1, line.trim()));
            if out.len() >= MAX_REFERENCE_HITS {
                break;
            }
        }
    }
    out
}

fn lsp_hover(file_path: &str, line: usize, column: usize) -> Result<String, String> {
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
    let char_idx = byte_index_for_column(line_content, column);
    let word = find_word_at_position(line_content, char_idx);

    // 尝试在本文件符号表里给出该标识符的种类信息。
    let mut symbol_detail = String::new();
    if !word.is_empty() {
        if let Some(language) = ast_symbols::language_for_path(file_path) {
            if let Some(Ok(symbols)) =
                ast_symbols::extract_document_symbols(language, file_path, &content)
            {
                if let Some(found) = symbols.iter().find(|s| s.name == word) {
                    symbol_detail = match &found.detail {
                        Some(detail) if !detail.trim().is_empty() => {
                            format!(
                                "\nSymbol: {} {} [{}] (line {})",
                                found.kind, found.name, detail, found.line
                            )
                        }
                        _ => format!(
                            "\nSymbol: {} {} (line {})",
                            found.kind, found.name, found.line
                        ),
                    };
                }
            }
        }
    }

    Ok(format!(
        "{}:{}:{}\nLine: {}{}",
        file_path,
        line,
        column,
        line_content.trim(),
        symbol_detail
    ))
}

fn lsp_document_symbols(file_path: &str) -> Result<String, String> {
    let content =
        fs::read_to_string(file_path).map_err(|e| format!("Failed to read file: {}", e))?;
    let language = detect_language(file_path);

    if let Some(result) = ast_symbols::extract_document_symbols(&language, file_path, &content) {
        match result {
            Ok(symbols) => return Ok(ast_symbols::format_symbols_output(file_path, &symbols)),
            Err(err) => return Err(format!("Failed to parse {}: {}", file_path, err)),
        }
    }

    Ok(format!(
        "No AST symbol support for {} (language: {}).",
        file_path, language
    ))
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
            "Diagnostics are only available for Rust (via cargo check). \
             For {} (language: {}), use document_symbol/workspace_symbol for structure or rely on the build system.",
            file_path, language
        )),
    }
}

fn render_symbol_hits(hits: &[SymbolHit], root: &Path) -> String {
    hits.iter()
        .map(|hit| {
            let rel = hit.file.strip_prefix(root).unwrap_or(&hit.file);
            format!("{}:{}: {} {}", rel.display(), hit.line, hit.kind, hit.name)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// 遍历项目源文件，仅收集 AST 可解析语言的文件，跳过依赖/构建目录。
fn collect_source_files(root: &Path) -> Vec<PathBuf> {
    if root.is_file() {
        return vec![root.to_path_buf()];
    }
    if !root.is_dir() {
        return Vec::new();
    }

    let mut files = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(root.to_path_buf());
    let mut scanned_dirs = 0usize;
    let max_dirs = 10_000usize;

    while let Some(dir) = queue.pop_front() {
        if files.len() >= MAX_LSP_FILES {
            break;
        }
        scanned_dirs += 1;
        if scanned_dirs > max_dirs {
            break;
        }
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            if files.len() >= MAX_LSP_FILES {
                break;
            }
            let path = entry.path();
            let ft = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            if ft.is_file() {
                if ast_symbols::language_for_path(&path.to_string_lossy()).is_some() {
                    files.push(path);
                }
            } else if ft.is_dir() && !ft.is_symlink() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !name.starts_with('.') && !rust_tools::commonw::is_skip_dir(name) {
                    queue.push_back(path);
                }
            }
        }
    }
    files
}

/// 把 1-based 列号转成字节索引，兼容多字节字符。
fn byte_index_for_column(line: &str, column: usize) -> usize {
    let col = column.max(1);
    line.char_indices()
        .nth(col - 1)
        .map(|(idx, _)| idx)
        .unwrap_or_else(|| line.len())
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
    b.is_ascii_alphanumeric() || b == b'_'
}

/// 判断一行里是否以"全词"形式出现 `word`（前后不是标识符字符）。
fn line_contains_whole_word(line: &str, word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    let bytes = line.as_bytes();
    let wb = word.as_bytes();
    let mut from = 0;
    while let Some(pos) = find_subslice(&bytes[from..], wb) {
        let start = from + pos;
        let end = start + wb.len();
        let left_ok = start == 0 || !is_identifier_char(bytes[start - 1]);
        let right_ok = end >= bytes.len() || !is_identifier_char(bytes[end]);
        if left_ok && right_ok {
            return true;
        }
        from = start + 1;
        if from >= bytes.len() {
            break;
        }
    }
    false
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("ai_lsp_test_{}_{}", name, uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).expect("failed to create temp dir");
        path
    }

    #[test]
    fn find_word_at_position_extracts_identifier() {
        let line = "let value = compute_total(items);";
        let idx = line.find("compute_total").unwrap();
        assert_eq!(find_word_at_position(line, idx), "compute_total");
    }

    #[test]
    fn line_contains_whole_word_respects_boundaries() {
        assert!(line_contains_whole_word("foo(bar)", "foo"));
        assert!(line_contains_whole_word("a = foo;", "foo"));
        assert!(!line_contains_whole_word("foobar()", "foo"));
        assert!(!line_contains_whole_word("do_foo()", "foo"));
    }

    #[test]
    fn workspace_symbol_finds_rust_function() {
        let dir = make_temp_dir("ws_symbol");
        fs::write(dir.join("Cargo.toml"), "[package]\nname=\"x\"\n").unwrap();
        fs::create_dir_all(dir.join("src")).unwrap();
        let file = dir.join("src/lib.rs");
        fs::write(&file, "fn alpha() {}\nfn beta_helper() {}\n").unwrap();

        let result = lsp_workspace_symbol(&file.to_string_lossy(), "beta").unwrap();
        assert!(result.contains("beta_helper"), "{}", result);
        assert!(result.contains("function"), "{}", result);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn workspace_symbol_no_match_uses_expected_prefix() {
        let dir = make_temp_dir("ws_symbol_none");
        fs::write(dir.join("Cargo.toml"), "[package]\nname=\"x\"\n").unwrap();
        let file = dir.join("lib.rs");
        fs::write(&file, "fn alpha() {}\n").unwrap();

        let result = lsp_workspace_symbol(&file.to_string_lossy(), "zzz_missing").unwrap();
        assert!(
            result.starts_with("No symbols matching '"),
            "fallback chains depend on this prefix: {}",
            result
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn go_to_definition_by_query_locates_symbol() {
        let dir = make_temp_dir("gotodef");
        fs::write(dir.join("Cargo.toml"), "[package]\nname=\"x\"\n").unwrap();
        let file = dir.join("main.rs");
        fs::write(&file, "fn target_fn() {}\nfn caller() { target_fn(); }\n").unwrap();

        let args = serde_json::json!({
            "operation": "go_to_definition",
            "file_path": file.to_string_lossy(),
            "query": "target_fn"
        });
        let result = execute_lsp(&args).unwrap();
        assert!(result.contains("target_fn"), "{}", result);
        assert!(
            result.contains(":1:"),
            "definition should be on line 1: {}",
            result
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_references_lists_usages() {
        let dir = make_temp_dir("findrefs");
        fs::write(dir.join("Cargo.toml"), "[package]\nname=\"x\"\n").unwrap();
        let file = dir.join("main.rs");
        fs::write(
            &file,
            "fn target_fn() {}\nfn caller() { target_fn(); }\nfn other() { target_fn(); }\n",
        )
        .unwrap();

        let args = serde_json::json!({
            "operation": "find_references",
            "file_path": file.to_string_lossy(),
            "query": "target_fn"
        });
        let result = execute_lsp(&args).unwrap();
        assert!(result.contains("Definition(s)"), "{}", result);
        assert!(result.contains("reference line(s)"), "{}", result);
        // 定义行 + 两处调用 = 3 行引用
        assert!(result.matches("target_fn").count() >= 3, "{}", result);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn document_symbol_lists_python_defs() {
        let dir = make_temp_dir("docsym");
        let file = dir.join("mod.py");
        fs::write(
            &file,
            "def first():\n    pass\n\nclass Thing:\n    def method(self):\n        pass\n",
        )
        .unwrap();

        let result = lsp_document_symbols(&file.to_string_lossy()).unwrap();
        assert!(result.contains("first"), "{}", result);
        assert!(result.contains("Thing"), "{}", result);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn hover_reports_symbol_kind() {
        let dir = make_temp_dir("hover");
        let file = dir.join("lib.rs");
        fs::write(&file, "fn hovered_fn() {}\n").unwrap();

        let args = serde_json::json!({
            "operation": "hover",
            "file_path": file.to_string_lossy(),
            "line": 1,
            "column": 4
        });
        let result = execute_lsp(&args).unwrap();
        assert!(result.contains("hovered_fn"), "{}", result);

        let _ = fs::remove_dir_all(&dir);
    }
}
