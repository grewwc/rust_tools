use tree_sitter::Parser;

use super::{SymbolEntry, first_named_child_text, name_from_field, text_for_node};

pub(crate) fn extract(_file_path: &str, content: &str) -> Result<Vec<SymbolEntry>, String> {
    let mut parser = Parser::new();
    let language = tree_sitter::Language::from(tree_sitter_rust::LANGUAGE);
    parser
        .set_language(&language)
        .map_err(|e| format!("Failed to configure rust parser: {}", e))?;
    let tree = parser
        .parse(content, None)
        .ok_or("Failed to parse Rust source".to_string())?;

    let mut symbols = Vec::new();
    let root = tree.root_node();
    let mut cursor = root.walk();
    for child in root.named_children(&mut cursor) {
        visit_node(child, content, 0, &mut symbols);
    }
    Ok(symbols)
}

fn visit_node(node: tree_sitter::Node<'_>, source: &str, indent: usize, out: &mut Vec<SymbolEntry>) {
    match node.kind() {
        "function_item" => {
            if let Some(name) = name_from_field(node, "name", source) {
                out.push(SymbolEntry::new("function", name, line(node), None, indent));
            }
        }
        "struct_item" => push_named(node, source, "struct", indent, out),
        "enum_item" => push_named(node, source, "enum", indent, out),
        "trait_item" => {
            push_named(node, source, "trait", indent, out);
            recurse_named(node, source, indent + 1, out);
        }
        "mod_item" => {
            push_named(node, source, "mod", indent, out);
            recurse_named(node, source, indent + 1, out);
        }
        "type_item" => push_named(node, source, "type", indent, out),
        "const_item" => push_named(node, source, "const", indent, out),
        "static_item" => push_named(node, source, "static", indent, out),
        "macro_definition" => {
            if let Some(name) = name_from_field(node, "name", source)
                .or_else(|| first_named_child_text(node, source, &["identifier"])) {
                out.push(SymbolEntry::new("macro", name, line(node), None, indent));
            }
        }
        "impl_item" => {
            let target = impl_target(node, source);
            out.push(SymbolEntry::new("impl", target, line(node), impl_detail(node, source), indent));
            recurse_named(node, source, indent + 1, out);
        }
        "declaration_list" => {
            recurse_named(node, source, indent, out);
        }
        "associated_type" => {
            if let Some(name) = name_from_field(node, "name", source) {
                out.push(SymbolEntry::new("assoc type", name, line(node), None, indent));
            }
        }
        "function_signature_item" => {
            if let Some(name) = name_from_field(node, "name", source) {
                out.push(SymbolEntry::new("method", name, line(node), None, indent));
            }
        }
        _ => {}
    }
}

fn recurse_named(
    node: tree_sitter::Node<'_>,
    source: &str,
    indent: usize,
    out: &mut Vec<SymbolEntry>,
) {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "function_item" => {
                if let Some(name) = name_from_field(child, "name", source) {
                    out.push(SymbolEntry::new("method", name, line(child), None, indent));
                }
            }
            "function_signature_item" => {
                if let Some(name) = name_from_field(child, "name", source) {
                    out.push(SymbolEntry::new("method", name, line(child), None, indent));
                }
            }
            "const_item" => {
                if let Some(name) = name_from_field(child, "name", source) {
                    out.push(SymbolEntry::new("const", name, line(child), None, indent));
                }
            }
            "type_item" | "associated_type" => {
                if let Some(name) = name_from_field(child, "name", source) {
                    out.push(SymbolEntry::new("type", name, line(child), None, indent));
                }
            }
            "declaration_list" => recurse_named(child, source, indent, out),
            _ => {}
        }
    }
}

fn push_named(
    node: tree_sitter::Node<'_>,
    source: &str,
    kind: &'static str,
    indent: usize,
    out: &mut Vec<SymbolEntry>,
) {
    if let Some(name) = name_from_field(node, "name", source) {
        out.push(SymbolEntry::new(kind, name, line(node), None, indent));
    }
}

fn impl_target(node: tree_sitter::Node<'_>, source: &str) -> String {
    node.child_by_field_name("type")
        .and_then(|n| text_for_node(n, source))
        .or_else(|| first_named_child_text(node, source, &["type_identifier", "scoped_type_identifier"]))
        .unwrap_or_else(|| "<unknown>".to_string())
}

fn impl_detail(node: tree_sitter::Node<'_>, source: &str) -> Option<String> {
    let trait_name = node
        .child_by_field_name("trait")
        .and_then(|n| text_for_node(n, source));
    trait_name.filter(|s| !s.is_empty())
}

fn line(node: tree_sitter::Node<'_>) -> usize {
    node.start_position().row + 1
}
