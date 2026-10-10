//! Markup parsing helpers (headings, thematic breaks, blockquotes, ordered-list
//! prefixes). Pure string parsers; no terminal or rendering state.

use crate::ai::stream::render::table::split_indent;

pub(super) fn parse_heading(line: &str) -> Option<(usize, &str)> {
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i] == b'#' {
        i += 1;
    }
    if i == 0 || i > 6 {
        return None;
    }
    if i >= bytes.len() || bytes[i] != b' ' {
        return None;
    }
    Some((i, line[i + 1..].trim_end()))
}

pub(super) fn is_thematic_break(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.len() < 3 {
        return false;
    }
    let mut chars = trimmed.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !matches!(first, '-' | '*' | '_') {
        return false;
    }
    chars.all(|ch| ch == first)
}

pub(super) fn parse_blockquote(line: &str) -> Option<&str> {
    let body = line.strip_prefix("> ")?;
    Some(body.trim_end())
}

pub(super) fn split_list_prefix(line: &str) -> Option<(&str, &str, Option<bool>, &str)> {
    let (indent, rest) = split_indent(line);
    let rest = rest.trim_end();

    // Task list: - [ ] / - [x] / - [X]
    if rest.starts_with("- [ ] ") {
        return Some((indent, "- ", Some(false), &rest[6..]));
    }
    if rest.starts_with("- [x] ") || rest.starts_with("- [X] ") {
        return Some((indent, "- ", Some(true), &rest[6..]));
    }

    // Bullet list
    if rest.starts_with("- ") || rest.starts_with("* ") || rest.starts_with("+ ") {
        return Some((indent, &rest[..2], None, &rest[2..]));
    }

    // Ordered list
    let bytes = rest.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
        if i > 4 {
            break;
        }
    }
    if i == 0 || i + 1 >= bytes.len() {
        return None;
    }
    if bytes[i] == b'.' && bytes[i + 1] == b' ' {
        return Some((indent, &rest[..i + 2], None, &rest[i + 2..]));
    }
    None
}
