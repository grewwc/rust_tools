use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::ai::tools::ast_structural::execute_structural_search;
use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::lsp_tools::execute_lsp;
use crate::ai::tools::search_tools::execute_search_files;
use crate::ai::tools::text_grep_tools::{ContentSearchOptions, run_content_search};

const CODE_EXTENSIONS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "py", "go", "java", "c", "h", "cpp", "cc", "cxx", "hpp", "rb",
    "php", "cs",
];

const SKIP_DIRS: &[&str] = rust_tools::commonw::SKIP_DIRS;
const MAX_WALK_FILES: usize = 10_000;

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
                "description": "Root directory for file, text, or structural search. Also used to auto-pick an anchor file for workspace_symbol when file_path is omitted. Defaults to the current project (\".\"). Must be inside the active workspace — passing the filesystem root \"/\" or system paths like \"/System\", \"/Library\", \"/usr\" is rejected."
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
            "is_regex": {
                "type": "boolean",
                "description": "When true, treat 'query' as a regular expression for text_search. Defaults to false (literal substring match)."
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "When false, perform case-insensitive matching for text_search. Defaults to true."
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
        description: "High-level code navigation and search tool. Prefer this before raw grep/read flows when you need symbol lookup, definitions, references, file discovery, full-text search, or structural matching. For structural searches, set operation=structural and choose intent=find_functions|find_classes|find_methods|find_calls. Use name / contains_text / call_kind / receiver / qualified_name to narrow large result sets. Example: operation=structural, intent=find_calls, call_kind=method_call, receiver=app.view, name=render.",
        parameters: params_code_search,
        execute: execute_code_search,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::Spawnable,
        groups: &["builtin", "core"],
    }
});

pub(crate) fn execute_code_search(args: &Value) -> Result<String, String> {
    let operation = args["operation"]
        .as_str()
        .ok_or("Missing 'operation' parameter")?;

    if let Some(intent) = legacy_structural_intent(operation) {
        let mut normalized = args.clone();
        normalized["operation"] = Value::String("structural".to_string());
        normalized["intent"] = Value::String(intent.to_string());
        return execute_code_structural(&normalized);
    }

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

fn legacy_structural_intent(operation: &str) -> Option<&'static str> {
    match operation {
        "find_functions" => Some("find_functions"),
        "find_classes" => Some("find_classes"),
        "find_methods" => Some("find_methods"),
        "find_calls" => Some("find_calls"),
        _ => None,
    }
}

fn nonempty_str_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key)
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

/// 守卫一个 `path` 字符串参数：解析为绝对路径后调用
/// [`text_grep_tools::validate_search_root`]，拒绝 `/` 与系统目录。
/// 路径不存在时不在这里报错，留给下游函数按各自语义处理。
fn guard_path_arg(raw: &str) -> Result<(), String> {
    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .map_err(|e| format!("Failed to get cwd: {}", e))?;
    let p = PathBuf::from(raw);
    let abs = if p.is_absolute() { p } else { cwd.join(p) };
    super::text_grep_tools::validate_search_root(&abs, &cwd)
}

fn build_text_search_args(args: &Value, query: &str, forced_path: Option<&str>) -> Value {
    let mut forwarded = serde_json::json!({
        "operation": "text_search",
        "query": query,
    });

    if let Some(path) = forced_path.filter(|value| !value.trim().is_empty()) {
        forwarded["path"] = Value::String(path.to_string());
    } else if let Some(file_path) = nonempty_str_arg(args, "file_path") {
        forwarded["file_path"] = Value::String(file_path.to_string());
    } else if let Some(path) = nonempty_str_arg(args, "path") {
        forwarded["path"] = Value::String(path.to_string());
    }

    if let Some(file_pattern) = nonempty_str_arg(args, "file_pattern") {
        forwarded["file_pattern"] = Value::String(file_pattern.to_string());
    }
    if let Some(case_sensitive) = args.get("case_sensitive").and_then(|value| value.as_bool()) {
        forwarded["case_sensitive"] = Value::Bool(case_sensitive);
    }
    if let Some(is_regex) = args.get("is_regex").and_then(|value| value.as_bool()) {
        forwarded["is_regex"] = Value::Bool(is_regex);
    }

    forwarded
}

fn render_guidance_lines(lines: &[String]) -> String {
    if lines.is_empty() {
        return String::new();
    }
    let mut out = String::from("guidance:\n");
    for line in lines {
        out.push_str("- ");
        out.push_str(line);
        out.push('\n');
    }
    out.trim_end().to_string()
}

fn is_structural_no_match(result: &str) -> bool {
    result.starts_with("No AST structural matches")
        || result.starts_with("No supported source files found")
}

fn is_workspace_symbol_no_match(result: &str) -> bool {
    result.starts_with("No symbols matching '")
}

fn derive_structural_fallback_query<'a>(args: &'a Value) -> Option<&'a str> {
    nonempty_str_arg(args, "name")
        .or_else(|| nonempty_str_arg(args, "contains_text"))
        .or_else(|| {
            if args
                .get("intent")
                .and_then(|value| value.as_str())
                .is_some()
            {
                nonempty_str_arg(args, "query")
            } else {
                None
            }
        })
}

fn structural_guidance(args: &Value, target: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut scope = format!("path={target}");
    if let Some(file_path) = nonempty_str_arg(args, "file_path") {
        scope = format!("file_path={file_path}");
    }
    if let Some(name) = nonempty_str_arg(args, "name") {
        lines.push(format!(
            "Remove or broaden the `name` filter, then retry `code_search(operation=structural, intent={}, {}, name={})` only if the narrower symbol name is still likely correct.",
            nonempty_str_arg(args, "intent").unwrap_or("find_functions"),
            scope,
            name
        ));
    }
    if let Some(query) = derive_structural_fallback_query(args) {
        lines.push(format!(
            "Use `code_search(operation=text_search, query={}, {})` to search by raw text when AST structure is too narrow.",
            query, scope
        ));
    }
    lines.push("If you need broader discovery, switch intent between `find_functions`, `find_methods`, `find_calls`, and `find_classes` instead of repeating the same request.".to_string());
    lines
}

fn text_search_guidance(query: &str, target: &Path, file_pattern: Option<&str>) -> Vec<String> {
    let mut lines = Vec::new();
    if file_pattern.is_some() {
        lines.push("Remove or widen `file_pattern` if the current glob may be excluding the relevant files.".to_string());
    }
    lines.push(format!(
        "Retry with `case_sensitive=false` if '{}' may appear with different casing.",
        query
    ));
    lines.push(format!(
        "If '{}' is a symbol or type name, try `code_search(operation=workspace_symbol, query={}, path={})` or `code_search(operation=structural, intent=find_functions, path={})`.",
        query,
        query,
        target.display(),
        target.display()
    ));
    lines
}

fn find_file_guidance(pattern: &str, path: &str) -> Vec<String> {
    vec![
        format!(
            "Retry with a broader glob such as `**/*{pattern}*` if the exact filename is uncertain."
        ),
        format!(
            "If '{}' is content rather than a filename, use `code_search(operation=text_search, query={}, path={})`.",
            pattern, pattern, path
        ),
    ]
}

fn render_workspace_symbol_with_fallback(
    args: &Value,
    query: &str,
    result: String,
    fallback_path: &str,
) -> Result<String, String> {
    if !is_workspace_symbol_no_match(&result) {
        return Ok(format!(
            "code_search route=lsp operation=workspace_symbol\n{}",
            result
        ));
    }

    let fallback_args = build_text_search_args(args, query, Some(fallback_path));
    let fallback = execute_code_text_search(&fallback_args)?;
    Ok(format!(
        "code_search route=lsp operation=workspace_symbol\nsummary: No workspace symbols matched '{}'; ran text_search fallback in '{}'.\n{}\n\nFallback content search:\n{}",
        query, fallback_path, result, fallback
    ))
}

fn execute_code_find_file(args: &Value) -> Result<String, String> {
    let pattern = args["pattern"]
        .as_str()
        .or_else(|| args["query"].as_str())
        .ok_or("find_file requires 'pattern' or 'query'")?;
    let path = args["path"].as_str().unwrap_or(".");
    guard_path_arg(path)?;
    let forwarded = serde_json::json!({
        "pattern": pattern,
        "path": path,
    });
    let result = execute_search_files(&forwarded)?;
    if result.trim().is_empty() {
        let guidance = render_guidance_lines(&find_file_guidance(pattern, path));
        Ok(format!(
            "code_search route=search_files operation=find_file\nsummary: No files matched '{}' under '{}'.\nNo files matched '{}' under '{}'.\n{}",
            pattern, path, pattern, path, guidance
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
    let is_regex = args["is_regex"].as_bool().unwrap_or(false);
    let case_sensitive = args["case_sensitive"].as_bool().unwrap_or(true);
    let result = execute_content_text_search(
        query,
        &target,
        args["file_pattern"].as_str(),
        is_regex,
        case_sensitive,
    )?;
    if result.trim().is_empty() {
        let guidance = render_guidance_lines(&text_search_guidance(
            query,
            &target,
            args["file_pattern"].as_str(),
        ));
        Ok(format!(
            "code_search route=content_search operation=text_search\nsummary: No text matches for '{}' under '{}'.\nNo text matches for '{}' under '{}'.\n{}",
            query,
            target.display(),
            query,
            target.display(),
            guidance
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
            return Err(format!(
                "text_search file_path is not a file: {}",
                file_path
            ));
        }
        return Ok(path);
    }

    let raw = args["path"].as_str().unwrap_or(".");
    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .map_err(|e| format!("Failed to get cwd: {}", e))?;
    let path = {
        let p = PathBuf::from(raw);
        if p.is_absolute() { p } else { cwd.join(p) }
    };
    if !path.exists() {
        return Err(format!("Path not found: {}", path.display()));
    }
    super::text_grep_tools::validate_search_root(&path, &cwd)?;
    Ok(path)
}

fn execute_content_text_search(
    query: &str,
    target: &Path,
    file_pattern: Option<&str>,
    is_regex: bool,
    case_sensitive: bool,
) -> Result<String, String> {
    if query.trim().is_empty() {
        return Err("text_search query is empty".to_string());
    }

    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .map_err(|e| format!("Failed to get cwd: {}", e))?;

    let options = ContentSearchOptions {
        query,
        is_regex,
        case_sensitive,
        context_lines: 2,
        max_results: 200,
        file_pattern,
        // code_search 追求"最强代码搜索"：不按扩展名白名单过滤，覆盖面与
        // text_grep 一致；调用方仍可用可选的 file_pattern 收窄范围。
        extensions: None,
        display_root: Some(&cwd),
    };

    let result = run_content_search(target, &options)?;
    // 共享引擎无命中时返回 "No matches found."；这里归一成空串，
    // 让上层 `execute_code_text_search` 沿用 is_empty 判定挂 summary/guidance。
    if result == "No matches found." {
        return Ok(String::new());
    }
    Ok(result)
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
    let fallback_path = args["path"].as_str().unwrap_or(".");
    render_workspace_symbol_with_fallback(args, query, result, fallback_path)
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
    let file_path = match args["file_path"].as_str() {
        Some(fp) => {
            if !Path::new(fp).exists() {
                return Err(format!("File not found: {}", fp));
            }
            fp.to_string()
        }
        None => {
            if let Some(query) = args["query"].as_str() {
                return fallback_lsp_to_workspace_symbol(args, operation, query);
            }
            return Err(format!(
                "{} requires 'file_path' (with 'line'/'column') or 'query' to fall back to a workspace symbol search",
                operation
            ));
        }
    };
    let mut forwarded = serde_json::json!({
        "operation": operation,
        "file_path": file_path,
    });
    if let Some(query) = args["query"].as_str() {
        forwarded["query"] = Value::String(query.to_string());
    }
    let needs_position = matches!(operation, "go_to_definition" | "find_references" | "hover");
    if let Some(line) = args["line"].as_u64() {
        forwarded["line"] = Value::Number(line.into());
        if let Some(column) = args["column"].as_u64() {
            forwarded["column"] = Value::Number(column.into());
        }
    } else if needs_position {
        if let Some(query) = args["query"].as_str() {
            if let Some((line, col)) = find_symbol_position(&file_path, query) {
                forwarded["line"] = Value::Number(line.into());
                forwarded["column"] = Value::Number(col.into());
            }
        }
    }
    let result = execute_lsp(&forwarded)?;
    Ok(format!(
        "code_search route=lsp operation={}\n{}",
        operation, result
    ))
}

fn find_symbol_position(file_path: &str, symbol: &str) -> Option<(usize, usize)> {
    let content = fs::read_to_string(file_path).ok()?;
    for (idx, line) in content.lines().enumerate() {
        if let Some(col) = line.find(symbol) {
            return Some((idx + 1, col + 1));
        }
    }
    None
}

fn fallback_lsp_to_workspace_symbol(
    args: &Value,
    original_operation: &str,
    query: &str,
) -> Result<String, String> {
    let anchor_file = resolve_anchor_file(args)?;
    let forwarded = serde_json::json!({
        "operation": "workspace_symbol",
        "file_path": anchor_file,
        "query": query,
    });
    let result = execute_lsp(&forwarded)?;
    let fallback_path = args["path"].as_str().unwrap_or(".");
    let rendered = render_workspace_symbol_with_fallback(args, query, result, fallback_path)?;
    Ok(format!(
        "{}\ncontext: original operation was '{}' and file_path was not provided.",
        rendered, original_operation
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
    if !is_structural_no_match(&result) {
        return Ok(format!(
            "code_search route=tree_sitter operation=structural\n{}",
            result
        ));
    }

    let mut sections = vec![format!(
        "code_search route=tree_sitter operation=structural\nsummary: {}",
        result
    )];
    sections.push(result);

    if let Some(query) = derive_structural_fallback_query(args) {
        let fallback_args = build_text_search_args(args, query, Some(&target));
        let fallback = execute_code_text_search(&fallback_args)?;
        sections.push(format!(
            "Fallback content search for '{}':\n{}",
            query, fallback
        ));
    }

    let guidance = render_guidance_lines(&structural_guidance(args, &target));
    if !guidance.is_empty() {
        sections.push(guidance);
    }

    Ok(sections.join("\n\n"))
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
        let p = PathBuf::from(file_path);
        if p.is_dir() {
            guard_path_arg(file_path)?;
        }
        return Ok(file_path.to_string());
    }
    let path = args["path"].as_str().unwrap_or(".");
    guard_path_arg(path)?;
    Ok(path.to_string())
}

fn resolve_anchor_file(args: &Value) -> Result<String, String> {
    if let Some(file_path) = args["file_path"].as_str() {
        if Path::new(file_path).exists() {
            return Ok(file_path.to_string());
        }
        return Err(format!("File not found: {}", file_path));
    }

    let raw_path = args["path"].as_str().unwrap_or(".");
    guard_path_arg(raw_path)?;
    let root = PathBuf::from(raw_path);
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
    walk_files(root, is_code_file).into_iter().next()
}

fn is_code_file(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .map_or(false, |ext| CODE_EXTENSIONS.contains(&ext))
}

fn should_skip_dir(dir_name: &str) -> bool {
    dir_name.starts_with('.') || SKIP_DIRS.contains(&dir_name)
}

fn walk_files(root: &Path, predicate: fn(&Path) -> bool) -> Vec<PathBuf> {
    if !root.exists() || !root.is_dir() {
        return Vec::new();
    }
    let mut files = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(root.to_path_buf());
    let mut scanned_dirs = 0usize;
    let max_dirs = 10_000usize;

    while let Some(dir) = queue.pop_front() {
        if files.len() >= MAX_WALK_FILES {
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
            if files.len() >= MAX_WALK_FILES {
                break;
            }
            let path = entry.path();
            let ft = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            if ft.is_file() && predicate(&path) {
                files.push(path);
            } else if ft.is_dir() && !ft.is_symlink() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !should_skip_dir(name) {
                    queue.push_back(path);
                }
            }
        }
    }
    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "ai_code_search_test_{}_{}",
            name,
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&path).expect("failed to create temp dir");
        path
    }

    #[test]
    fn code_search_params_require_operation() {
        let params = params_code_search();
        assert!(
            params["required"]
                .as_array()
                .unwrap()
                .contains(&Value::String("operation".to_string()))
        );
    }

    #[test]
    fn legacy_structural_operation_is_mapped_to_intent() {
        assert_eq!(
            legacy_structural_intent("find_functions"),
            Some("find_functions")
        );
        assert_eq!(
            legacy_structural_intent("find_classes"),
            Some("find_classes")
        );
        assert_eq!(
            legacy_structural_intent("find_methods"),
            Some("find_methods")
        );
        assert_eq!(legacy_structural_intent("find_calls"), Some("find_calls"));
        assert_eq!(legacy_structural_intent("text_search"), None);
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
        assert!(result.contains("sample.rs"));
        assert!(result.contains("fn alpha() {}"));

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

        assert!(result.contains("keep.rs"));
        assert!(result.contains("fn beta() {}"));
        assert!(!result.contains("skip.txt"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_search_no_match_includes_summary_and_guidance() {
        let dir = make_temp_dir("text_search_no_match");
        let file = dir.join("sample.rs");
        fs::write(&file, "fn alpha() {}\n").unwrap();

        let args = serde_json::json!({
            "operation": "text_search",
            "path": dir.to_string_lossy(),
            "query": "missing_symbol",
            "file_pattern": "*.rs"
        });
        let result = execute_code_text_search(&args).unwrap();

        assert!(result.contains("summary: No text matches for 'missing_symbol'"));
        assert!(result.contains("guidance:"));
        assert!(result.contains("case_sensitive=false"));

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

    #[test]
    fn structural_no_match_runs_text_fallback() {
        let dir = make_temp_dir("structural_fallback");
        let file = dir.join("sample.rs");
        fs::write(&file, "fn alpha() {}\nfn beta() {}\n").unwrap();

        let args = serde_json::json!({
            "operation": "structural",
            "intent": "find_functions",
            "file_path": file.to_string_lossy(),
            "name": "gamma"
        });
        let result = execute_code_search(&args).unwrap();

        assert!(result.contains("summary: No AST structural matches"));
        assert!(result.contains("Fallback content search for 'gamma'"));
        assert!(result.contains("summary: No text matches for 'gamma'"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn workspace_symbol_no_match_runs_text_fallback() {
        let dir = make_temp_dir("workspace_symbol_fallback");
        let file = dir.join("sample.rs");
        fs::write(&file, "fn alpha() {}\n").unwrap();

        let args = serde_json::json!({
            "operation": "workspace_symbol",
            "path": dir.to_string_lossy(),
            "query": "alpha"
        });
        let result = render_workspace_symbol_with_fallback(
            &args,
            "alpha",
            "No symbols matching 'alpha' found in workspace".to_string(),
            &dir.to_string_lossy(),
        )
        .unwrap();

        assert!(result.contains("summary: No workspace symbols matched 'alpha'"));
        assert!(result.contains("Fallback content search:"));
        assert!(result.contains("sample.rs"));
        assert!(result.contains("fn alpha() {}"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn legacy_structural_operation_executes_structural_search() {
        let dir = make_temp_dir("legacy_structural");
        let file = dir.join("sample.rs");
        fs::write(&file, "fn alpha() {}\nfn beta() {}\n").unwrap();

        let args = serde_json::json!({
            "operation": "find_functions",
            "file_path": file.to_string_lossy(),
            "name": "alpha"
        });
        let result = execute_code_search(&args).unwrap();

        assert!(result.contains("code_search route=tree_sitter operation=structural"));
        assert!(result.contains("alpha"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn walk_files_skips_hidden_and_dependency_dirs() {
        let dir = make_temp_dir("walk_skip_dirs");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::create_dir_all(dir.join("target/debug")).unwrap();
        fs::create_dir_all(dir.join(".opencode/cache")).unwrap();
        fs::write(dir.join("src/lib.rs"), "fn keep() {}\n").unwrap();
        fs::write(dir.join("target/debug/generated.rs"), "fn generated() {}\n").unwrap();
        fs::write(
            dir.join(".opencode/cache/generated.js"),
            "function generated() {}\n",
        )
        .unwrap();

        let files = walk_files(&dir, is_code_file);
        let rendered = files
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("src/lib.rs"), "{}", rendered);
        assert!(
            !rendered.contains("target/debug/generated.rs"),
            "{}",
            rendered
        );
        assert!(
            !rendered.contains(".opencode/cache/generated.js"),
            "{}",
            rendered
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_search_skips_hidden_and_dependency_dirs() {
        let dir = make_temp_dir("content_skip_dirs");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::create_dir_all(dir.join("target/debug")).unwrap();
        fs::create_dir_all(dir.join(".opencode/cache")).unwrap();
        fs::write(dir.join("src/lib.rs"), "fn keep_marker() {}\n").unwrap();
        fs::write(
            dir.join("target/debug/generated.rs"),
            "fn keep_marker() {}\n",
        )
        .unwrap();
        fs::write(
            dir.join(".opencode/cache/generated.js"),
            "function keep_marker() {}\n",
        )
        .unwrap();

        let args = serde_json::json!({
            "operation": "text_search",
            "path": dir.to_string_lossy(),
            "query": "keep_marker"
        });
        let result = execute_code_text_search(&args).unwrap();

        assert!(result.contains("src/lib.rs"), "{}", result);
        assert!(!result.contains("target/debug"), "{}", result);
        assert!(!result.contains(".opencode/cache"), "{}", result);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_search_covers_non_whitelisted_extensions() {
        // 旧实现按 CODE_EXTENSIONS+EXTRA_TEXT 白名单过滤，覆盖面偏窄。
        // 放开后，`.env`、`.cfg` 这类非白名单文件也应被内容搜索覆盖。
        let dir = make_temp_dir("open_ext");
        fs::write(dir.join("config.env"), "SECRET_TOKEN=marker_value\n").unwrap();

        let args = serde_json::json!({
            "operation": "text_search",
            "path": dir.to_string_lossy(),
            "query": "marker_value"
        });
        let result = execute_code_text_search(&args).unwrap();

        assert!(result.contains("config.env"), "{}", result);
        assert!(result.contains("marker_value"), "{}", result);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_search_rejects_filesystem_root() {
        let args = serde_json::json!({
            "operation": "text_search",
            "path": "/",
            "query": "anything"
        });
        let err = execute_code_text_search(&args).expect_err("must reject /");
        assert!(err.contains("Refusing to search"), "{}", err);
    }

    #[test]
    fn find_file_rejects_filesystem_root() {
        let args = serde_json::json!({
            "operation": "find_file",
            "path": "/",
            "pattern": "Cargo.toml"
        });
        let err = execute_code_find_file(&args).expect_err("must reject /");
        assert!(err.contains("Refusing to search"), "{}", err);
    }

    #[test]
    fn resolve_anchor_file_rejects_filesystem_root() {
        let args = serde_json::json!({
            "path": "/",
            "query": "anything"
        });
        let err = resolve_anchor_file(&args).expect_err("must reject /");
        assert!(err.contains("Refusing to search"), "{}", err);
    }

    #[test]
    fn structural_rejects_directory_file_path_root() {
        let args = serde_json::json!({
            "operation": "structural",
            "file_path": "/",
            "intent": "find_functions"
        });
        let err = execute_code_search(&args).expect_err("must reject directory file_path=/");
        assert!(err.contains("Refusing to search"), "{}", err);
    }
}
