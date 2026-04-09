use tree_sitter::Parser;

use super::{SymbolEntry, first_named_child_text, name_from_field, text_for_node};

pub(crate) fn extract(_file_path: &str, content: &str) -> Result<Vec<SymbolEntry>, String> {
    let mut parser = Parser::new();
    let language = tree_sitter::Language::from(tree_sitter_c::LANGUAGE);
    parser
        .set_language(&language)
        .map_err(|e| format!("Failed to configure c parser: {}", e))?;
    let tree = parser
        .parse(content, None)
        .ok_or("Failed to parse C source".to_string())?;

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
        "translation_unit" | "field_declaration_list" => {
            let mut cursor = node.walk();
            for child in node.named_children(&mut cursor) {
                visit_node(child, source, indent, in_type, out);
            }
        }
        "struct_specifier" => {
            push_named(node, source, "struct", indent, out);
            recurse_named(node, source, indent + 1, true, out);
        }
        "union_specifier" => {
            push_named(node, source, "union", indent, out);
            recurse_named(node, source, indent + 1, true, out);
        }
        "enum_specifier" => push_named(node, source, "enum", indent, out),
        "function_definition" => {
            if let Some(name) = extract_decl_name(node, source) {
                out.push(SymbolEntry::new(
                    if in_type { "method" } else { "function" },
                    name,
                    line(node),
                    None,
                    indent,
                ));
            }
        }
        "declaration" => {
            if let Some(name) = extract_decl_name(node, source) {
                out.push(SymbolEntry::new(
                    if in_type { "field" } else { "declaration" },
                    name,
                    line(node),
                    None,
                    indent,
                ));
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

fn extract_decl_name(node: tree_sitter::Node<'_>, source: &str) -> Option<String> {
    node.child_by_field_name("declarator")
        .and_then(|n| {
            first_named_child_text(
                n,
                source,
                &["identifier", "field_identifier", "function_declarator"],
            )
        })
        .or_else(|| name_from_field(node, "name", source))
        .or_else(|| first_named_child_text(node, source, &["identifier", "field_identifier"]))
}

fn push_named(
    node: tree_sitter::Node<'_>,
    source: &str,
    kind: &'static str,
    indent: usize,
    out: &mut Vec<SymbolEntry>,
) {
    if let Some(name) = name_from_field(node, "name", source)
        .or_else(|| first_named_child_text(node, source, &["type_identifier", "identifier"]))
        .or_else(|| text_for_node(node, source))
    {
        out.push(SymbolEntry::new(kind, name, line(node), None, indent));
    }
}

fn line(node: tree_sitter::Node<'_>) -> usize {
    node.start_position().row + 1
}
