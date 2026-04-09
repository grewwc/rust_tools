use tree_sitter::Parser;

use super::{SymbolEntry, first_named_child_text, name_from_field};

pub(crate) fn extract(_file_path: &str, content: &str) -> Result<Vec<SymbolEntry>, String> {
    let mut parser = Parser::new();
    let language = tree_sitter::Language::from(tree_sitter_c_sharp::LANGUAGE);
    parser
        .set_language(&language)
        .map_err(|e| format!("Failed to configure csharp parser: {}", e))?;
    let tree = parser
        .parse(content, None)
        .ok_or("Failed to parse C# source".to_string())?;

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
        "compilation_unit" | "declaration_list" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                visit_node(child, source, indent, in_type, out);
            }
        }
        "namespace_declaration" | "file_scoped_namespace_declaration" => {
            push_named(node, source, "namespace", indent, out);
            recurse_named(node, source, indent + 1, in_type, out);
        }
        "class_declaration" => {
            push_named(node, source, "class", indent, out);
            recurse_named(node, source, indent + 1, true, out);
        }
        "struct_declaration" => {
            push_named(node, source, "struct", indent, out);
            recurse_named(node, source, indent + 1, true, out);
        }
        "interface_declaration" => {
            push_named(node, source, "interface", indent, out);
            recurse_named(node, source, indent + 1, true, out);
        }
        "enum_declaration" => push_named(node, source, "enum", indent, out),
        "record_declaration" => {
            push_named(node, source, "record", indent, out);
            recurse_named(node, source, indent + 1, true, out);
        }
        "method_declaration" => push_named(node, source, "method", indent, out),
        "constructor_declaration" => push_named(node, source, "constructor", indent, out),
        "property_declaration" => push_named(node, source, "property", indent, out),
        "field_declaration" => {
            if let Some(name) = first_named_child_text(node, source, &["variable_declarator", "identifier"]) {
                out.push(SymbolEntry::new("field", name, line(node), None, indent));
            }
        }
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
        .or_else(|| first_named_child_text(node, source, &["identifier"])) {
        out.push(SymbolEntry::new(kind, name, line(node), None, indent));
    }
}

fn line(node: tree_sitter::Node<'_>) -> usize {
    node.start_position().row + 1
}
