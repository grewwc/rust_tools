use tree_sitter::Parser;

use super::{SymbolEntry, name_from_field};

pub(crate) fn extract(_file_path: &str, content: &str) -> Result<Vec<SymbolEntry>, String> {
    let mut parser = Parser::new();
    let language = tree_sitter::Language::from(tree_sitter_python::LANGUAGE);
    parser
        .set_language(&language)
        .map_err(|e| format!("Failed to configure python parser: {}", e))?;
    let tree = parser
        .parse(content, None)
        .ok_or("Failed to parse Python source".to_string())?;

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
        "module" | "block" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                visit_node(child, source, indent, in_class, out);
            }
        }
        "class_definition" => {
            if let Some(name) = name_from_field(node, "name", source) {
                out.push(SymbolEntry::new("class", name, line(node), None, indent));
            }
            if let Some(body) = node.child_by_field_name("body") {
                visit_node(body, source, indent + 1, true, out);
            }
        }
        "function_definition" => {
            if let Some(name) = name_from_field(node, "name", source) {
                out.push(SymbolEntry::new(
                    if in_class { "method" } else { "function" },
                    name,
                    line(node),
                    None,
                    indent,
                ));
            }
            if let Some(body) = node.child_by_field_name("body") {
                visit_node(body, source, indent + 1, false, out);
            }
        }
        "decorated_definition" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                visit_node(child, source, indent, in_class, out);
            }
        }
        _ => {}
    }
}

fn line(node: tree_sitter::Node<'_>) -> usize {
    node.start_position().row + 1
}
