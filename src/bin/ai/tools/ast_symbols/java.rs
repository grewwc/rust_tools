use tree_sitter::Parser;

use super::{SymbolEntry, first_named_child_text, name_from_field};

pub(crate) fn extract(_file_path: &str, content: &str) -> Result<Vec<SymbolEntry>, String> {
    let mut parser = Parser::new();
    let language = tree_sitter::Language::from(tree_sitter_java::LANGUAGE);
    parser
        .set_language(&language)
        .map_err(|e| format!("Failed to configure java parser: {}", e))?;
    let tree = parser
        .parse(content, None)
        .ok_or("Failed to parse Java source".to_string())?;

    let mut symbols = Vec::new();
    visit_node(tree.root_node(), content, 0, false, &mut symbols);
    Ok(symbols)
}

fn visit_node(
    node: tree_sitter::Node<'_>,
    source: &str,
    indent: usize,
    in_type: bool,
    out: &mut Vec<SymbolEntry>,
) {
    match node.kind() {
        "program" | "class_body" | "interface_body" | "enum_body" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                visit_node(child, source, indent, in_type, out);
            }
        }
        "class_declaration" => {
            push_named(node, source, "class", indent, out);
            recurse_body(node, source, indent + 1, out);
        }
        "interface_declaration" => {
            push_named(node, source, "interface", indent, out);
            recurse_body(node, source, indent + 1, out);
        }
        "enum_declaration" => {
            push_named(node, source, "enum", indent, out);
            recurse_body(node, source, indent + 1, out);
        }
        "annotation_type_declaration" => {
            push_named(node, source, "annotation", indent, out);
            recurse_body(node, source, indent + 1, out);
        }
        "record_declaration" => {
            push_named(node, source, "record", indent, out);
            recurse_body(node, source, indent + 1, out);
        }
        "method_declaration" => {
            if let Some(name) = name_from_field(node, "name", source)
                .or_else(|| first_named_child_text(node, source, &["identifier"])) {
                out.push(SymbolEntry::new(
                    if in_type { "method" } else { "function" },
                    name,
                    line(node),
                    None,
                    indent,
                ));
            }
        }
        "constructor_declaration" => {
            if let Some(name) = name_from_field(node, "name", source)
                .or_else(|| first_named_child_text(node, source, &["identifier"])) {
                out.push(SymbolEntry::new("constructor", name, line(node), None, indent));
            }
        }
        "field_declaration" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                if child.kind() == "variable_declarator"
                    && let Some(name) = name_from_field(child, "name", source)
                {
                    out.push(SymbolEntry::new("field", name, line(child), None, indent));
                }
            }
        }
        _ => {}
    }
}

fn recurse_body(node: tree_sitter::Node<'_>, source: &str, indent: usize, out: &mut Vec<SymbolEntry>) {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if matches!(child.kind(), "class_body" | "interface_body" | "enum_body") {
            visit_node(child, source, indent, true, out);
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
    if let Some(name) = name_from_field(node, "name", source)
        .or_else(|| first_named_child_text(node, source, &["identifier", "type_identifier"])) {
        out.push(SymbolEntry::new(kind, name, line(node), None, indent));
    }
}

fn line(node: tree_sitter::Node<'_>) -> usize {
    node.start_position().row + 1
}
