mod c;
mod cpp;
mod csharp;
mod go;
mod java;
mod javascript;
mod php;
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
        "csharp" => Some(csharp::extract(file_path, content)),
        "php" => Some(php::extract(file_path, content)),
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

pub(crate) fn name_from_field(node: tree_sitter::Node<'_>, field: &str, source: &str) -> Option<String> {
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
