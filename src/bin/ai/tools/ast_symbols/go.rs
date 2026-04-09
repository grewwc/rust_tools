use tree_sitter::Parser;

use super::{SymbolEntry, first_named_child_text, name_from_field, text_for_node};

pub(crate) fn extract(_file_path: &str, content: &str) -> Result<Vec<SymbolEntry>, String> {
    let mut parser = Parser::new();
    let language = tree_sitter::Language::from(tree_sitter_go::LANGUAGE);
    parser
        .set_language(&language)
        .map_err(|e| format!("Failed to configure go parser: {}", e))?;
    let tree = parser
        .parse(content, None)
        .ok_or("Failed to parse Go source".to_string())?;

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
        "source_file" | "field_declaration_list" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                visit_node(child, source, indent, in_type, out);
            }
        }
        "function_declaration" => push_named(node, source, "function", indent, out),
        "method_declaration" => push_named(node, source, "method", indent, out),
        "type_declaration" => recurse_named(node, source, indent, true, out),
        "type_spec" => {
            if let Some(name) = name_from_field(node, "name", source) {
                let detail = node
                    .child_by_field_name("type")
                    .and_then(|n| text_for_node(n, source));
                out.push(SymbolEntry::new("type", name, line(node), detail, indent));
            }
            recurse_named(node, source, indent + 1, true, out);
        }
        "interface_type" => recurse_named(node, source, indent + 1, true, out),
        "method_spec" => push_named(node, source, "method", indent, out),
        "struct_type" => recurse_named(node, source, indent + 1, true, out),
        "field_declaration" if in_type => {
            if let Some(name) = first_named_child_text(node, source, &["field_identifier", "identifier"]) {
                out.push(SymbolEntry::new("field", name, line(node), None, indent));
            }
        }
        "const_declaration" | "var_declaration" => recurse_named(node, source, indent, in_type, out),
        "const_spec" => push_named(node, source, "const", indent, out),
        "var_spec" => push_named(node, source, "var", indent, out),
        _ => {}
    }
}

fn recurse_named(
    node: tree_sitter::Node<'_>,
    source: &str,
    indent: usize,
    in_type: bool,
    out: &mut Vec<SymbolEntry>,
) {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        visit_node(child, source, indent, in_type, out);
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
        .or_else(|| first_named_child_text(node, source, &["identifier", "field_identifier", "type_identifier"])) {
        out.push(SymbolEntry::new(kind, name, line(node), None, indent));
    }
}

fn line(node: tree_sitter::Node<'_>) -> usize {
    node.start_position().row + 1
}
