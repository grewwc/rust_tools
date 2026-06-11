mod c;
mod cpp;
mod go;
mod java;
mod javascript;
mod python;
mod rust;
mod typescript;

use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SymbolEntry {
    pub(crate) kind: &'static str,
    pub(crate) name: String,
    pub(crate) line: usize,
    pub(crate) detail: Option<String>,
    pub(crate) indent: usize,
}

impl SymbolEntry {
    pub(crate) fn new(
        kind: &'static str,
        name: impl Into<String>,
        line: usize,
        detail: Option<String>,
        indent: usize,
    ) -> Self {
        Self {
            kind,
            name: name.into(),
            line,
            detail,
            indent,
        }
    }
}

/// 按扩展名推断 AST 支持的语言；不支持的返回 None。
/// 与 lsp_tools::detect_language 不同，这里只覆盖 ast_symbols 真正能解析的 8 种语言，
/// 供 read_file 等路径判断是否值得生成符号大纲。
pub(crate) fn language_for_path(file_path: &str) -> Option<&'static str> {
    let ext = std::path::Path::new(file_path)
        .extension()
        .and_then(|e| e.to_str())?;
    let lang = match ext {
        "rs" => "rust",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" => "javascript",
        "py" => "python",
        "go" => "go",
        "java" => "java",
        "c" | "h" => "c",
        "cpp" | "cc" | "cxx" | "hpp" | "hxx" => "cpp",
        _ => return None,
    };
    Some(lang)
}

/// 为受支持的语言生成一段紧凑的符号大纲，供 read_file 输出时附带，
/// 让模型无需逐行 grep 即可获得结构化代码视图。
/// - 无法解析、无符号或语言不支持时返回 None；
/// - 大纲条目数上限为 `max_symbols`，超出时截断并提示。
pub(crate) fn document_symbol_outline(
    file_path: &str,
    content: &str,
    max_symbols: usize,
) -> Option<String> {
    let language = language_for_path(file_path)?;
    let symbols = extract_document_symbols(language, file_path, content)?.ok()?;
    if symbols.is_empty() {
        return None;
    }

    let total = symbols.len();
    let mut out = String::new();
    let _ = writeln!(&mut out, "Symbol outline ({} symbols):", total);
    for symbol in symbols.iter().take(max_symbols) {
        let indent = "  ".repeat(symbol.indent);
        match symbol.detail.as_deref() {
            Some(detail) if !detail.trim().is_empty() => {
                let _ = writeln!(
                    &mut out,
                    "{}{} {} [{}] (line {})",
                    indent, symbol.kind, symbol.name, detail, symbol.line
                );
            }
            _ => {
                let _ = writeln!(
                    &mut out,
                    "{}{} {} (line {})",
                    indent, symbol.kind, symbol.name, symbol.line
                );
            }
        }
    }
    if total > max_symbols {
        let _ = writeln!(
            &mut out,
            "... [{} more symbol(s) omitted; use the document_symbol LSP operation for the full list]",
            total - max_symbols
        );
    }
    Some(out.trim_end().to_string())
}

pub(crate) fn extract_document_symbols(
    language: &str,
    file_path: &str,
    content: &str,
) -> Option<Result<Vec<SymbolEntry>, String>> {
    match language {
        "c" => Some(c::extract(file_path, content)),
        "rust" => Some(rust::extract(file_path, content)),
        "python" => Some(python::extract(file_path, content)),
        "java" => Some(java::extract(file_path, content)),
        "cpp" => Some(cpp::extract(file_path, content)),
        "go" => Some(go::extract(file_path, content)),
        "javascript" => Some(javascript::extract(file_path, content)),
        "typescript" => Some(typescript::extract(file_path, content)),
        _ => None,
    }
}

pub(crate) fn format_symbols_output(file_path: &str, symbols: &[SymbolEntry]) -> String {
    if symbols.is_empty() {
        return format!("No symbols found in {}", file_path);
    }

    let mut out = String::new();
    let _ = writeln!(&mut out, "Symbols in {}:", file_path);
    for symbol in symbols {
        let indent = "  ".repeat(symbol.indent);
        match symbol.detail.as_deref() {
            Some(detail) if !detail.trim().is_empty() => {
                let _ = writeln!(
                    &mut out,
                    "{}{} {} [{}] (line {})",
                    indent, symbol.kind, symbol.name, detail, symbol.line
                );
            }
            _ => {
                let _ = writeln!(
                    &mut out,
                    "{}{} {} (line {})",
                    indent, symbol.kind, symbol.name, symbol.line
                );
            }
        }
    }
    out.trim_end().to_string()
}

pub(crate) fn text_for_node(node: tree_sitter::Node<'_>, source: &str) -> Option<String> {
    node.utf8_text(source.as_bytes())
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub(crate) fn name_from_field(
    node: tree_sitter::Node<'_>,
    field: &str,
    source: &str,
) -> Option<String> {
    node.child_by_field_name(field)
        .and_then(|name| text_for_node(name, source))
}

pub(crate) fn first_named_child_text(
    node: tree_sitter::Node<'_>,
    source: &str,
    wanted_kinds: &[&str],
) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if wanted_kinds.iter().any(|kind| *kind == child.kind()) {
            return text_for_node(child, source);
        }
        if let Some(found) = first_named_child_text(child, source, wanted_kinds) {
            return Some(found);
        }
    }
    None
}
