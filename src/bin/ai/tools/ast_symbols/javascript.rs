use tree_sitter::Parser;

use super::{SymbolEntry, first_named_child_text, name_from_field};

pub(crate) fn extract(_file_path: &str, content: &str) -> Result<Vec<SymbolEntry>, String> {
    let mut parser = Parser::new();
    let language = tree_sitter::Language::from(tree_sitter_javascript::LANGUAGE);
    parser
        .set_language(&language)
        .map_err(|e| format!("Failed to configure javascript parser: {}", e))?;
    let tree = parser
        .parse(content, None)
        .ok_or("Failed to parse JavaScript source".to_string())?;

    let mut symbols = Vec::new();
    visit_node(tree.root_node(), content, 0, false, &mut symbols);
    Ok(symbols)
}

fn visit_node(
    node: tree_sitter::Node<'_>,
    source: &str,
    indent: usize,
    in_class: bool,
    out: &mut Vec<SymbolEntry>,
) {
    match node.kind() {
        "program" | "statement_block" | "class_body" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                visit_node(child, source, indent, in_class, out);
            }
        }
        "function_declaration" => push_named(node, source, "function", indent, out),
        "generator_function_declaration" => push_named(node, source, "function", indent, out),
        "class_declaration" => {
            push_named(node, source, "class", indent, out);
            recurse_named(node, source, indent + 1, true, out);
        }
        "method_definition" => push_named(node, source, "method", indent, out),
        "public_field_definition" | "field_definition" => push_named(node, source, "field", indent, out),
        "lexical_declaration" | "variable_declaration" => recurse_named(node, source, indent, in_class, out),
        "variable_declarator" => {
            if let Some(name) = name_from_field(node, "name", source) {
                out.push(SymbolEntry::new("var", name, line(node), None, indent));
            }
        }
        _ => {}
    }
}

fn recurse_named(
    node: tree_sitter::Node<'_>,
    source: &str,
    indent: usize,
    in_class: bool,
    out: &mut Vec<SymbolEntry>,
) {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        visit_node(child, source, indent, in_class, out);
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
        .or_else(|| first_named_child_text(node, source, &["property_identifier", "identifier"])) {
        out.push(SymbolEntry::new(kind, name, line(node), None, indent));
    }
}

fn line(node: tree_sitter::Node<'_>) -> usize {
    node.start_position().row + 1
}
