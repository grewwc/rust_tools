use crate::ai::stream::render::code::{MONOKAI_BG, MONOKAI_FG};

pub(super) fn render_inline_md(s: &str, base: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::new();
    let mut i = 0usize;
    let mut bold = false;
    let mut italic = false;
    let mut code = false;
    let mut math = false;
    let mut math_delim = "$";
    let mut math_buf = String::new();

    fn apply_style(out: &mut String, base: &str, bold: bool, italic: bool, code: bool, math: bool) {
        out.push_str("\x1b[0m");
        out.push_str(base);
        if bold {
            out.push_str("\x1b[1m");
        }
        if code {
            out.push_str(MONOKAI_BG);
            out.push_str(MONOKAI_FG);
        }
        if italic {
            out.push_str("\x1b[3m");
        }
        if math {
            out.push_str("\x1b[95m");
        }
    }

    fn is_url_start(bytes: &[u8], i: usize) -> bool {
        bytes
            .get(i..i + 8)
            .is_some_and(|s| s.eq_ignore_ascii_case(b"https://"))
            || bytes
                .get(i..i + 7)
                .is_some_and(|s| s.eq_ignore_ascii_case(b"http://"))
    }

    fn url_raw_end(bytes: &[u8], start: usize) -> usize {
        let mut end = start;
        while end < bytes.len() {
            let b = bytes[end];
            if b.is_ascii_whitespace()
                || b == b'<'
                || b == b'"'
                || b == b'\''
                || b == b'`'
                || b == b'\\'
            {
                break;
            }
            end += 1;
        }
        end
    }

    while i < bytes.len() {
        if bytes[i] == b'`' {
            code = !code;
            apply_style(&mut out, base, bold, italic, code, math);
            i += 1;
            continue;
        }

        if !code && !math && bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            bold = !bold;
            apply_style(&mut out, base, bold, italic, code, math);
            i += 2;
            continue;
        }

        if !code && !math && bytes[i] == b'*' {
            italic = !italic;
            apply_style(&mut out, base, bold, italic, code, math);
            i += 1;
            continue;
        }

        // ~~strikethrough~~
        if !code && !math && i + 1 < bytes.len() && bytes[i] == b'~' && bytes[i + 1] == b'~' {
            i += 2; // skip opening ~~
            let start = i;
            while i + 1 < bytes.len() && !(bytes[i] == b'~' && bytes[i + 1] == b'~') {
                let ch = s[i..].chars().next().unwrap();
                i += ch.len_utf8();
            }
            let content = &s[start..i];
            out.push_str("\x1b[0m");
            out.push_str(base);
            if bold {
                out.push_str("\x1b[1m");
            }
            if italic {
                out.push_str("\x1b[3m");
            }
            out.push_str("\x1b[9m"); // strikethrough
            out.push_str(content);
            out.push_str("\x1b[0m");
            out.push_str(base);
            if bold {
                out.push_str("\x1b[1m");
            }
            if italic {
                out.push_str("\x1b[3m");
            }
            if i + 1 < bytes.len() {
                i += 2; // skip closing ~~
            }
            continue;
        }

        if !code && bytes[i] == b'$' {
            let is_double = i + 1 < bytes.len() && bytes[i + 1] == b'$';
            let delim = if is_double { "$$" } else { "$" };

            if math {
                if delim == math_delim {
                    let rendered = crate::ai::stream::render_math_tex_to_unicode(math_buf.trim());
                    out.push_str(&rendered);
                    math_buf.clear();
                    math = false;
                    apply_style(&mut out, base, bold, italic, code, math);
                    i += delim.len();
                    continue;
                }
            } else {
                math = true;
                math_delim = delim;
                apply_style(&mut out, base, bold, italic, code, math);
                i += delim.len();
                continue;
            }
        }

        if !math && is_url_start(bytes, i) {
            let raw_end = url_raw_end(bytes, i);
            let mut end = raw_end;
            while end > i {
                match bytes[end - 1] {
                    b'.' | b',' | b';' | b':' | b')' | b']' => end -= 1,
                    _ => break,
                }
            }
            let url = &s[i..end];
            let trail = &s[end..raw_end];

            out.push_str("\x1b[0m");
            out.push_str(base);
            if bold {
                out.push_str("\x1b[1m");
            }
            if italic {
                out.push_str("\x1b[3m");
            }
            out.push_str("\x1b[4m\x1b[34m");
            out.push_str(url);
            apply_style(&mut out, base, bold, italic, code, math);
            out.push_str(trail);

            i = raw_end;
            continue;
        }

        let ch = s[i..].chars().next().unwrap();
        if math && !code {
            math_buf.push(ch);
        } else {
            out.push(ch);
        }
        i += ch.len_utf8();
    }

    if math && !math_buf.is_empty() {
        out.push_str(&crate::ai::stream::render_math_tex_to_unicode(math_buf.trim()));
    }

    out.push_str("\x1b[0m");
    out
}

fn strip_inline_md_markers(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::new();
    let mut i = 0usize;
    let mut code = false;
    let mut math = false;
    let mut math_delim = "$";
    let mut math_buf = String::new();
    while i < bytes.len() {
        if bytes[i] == b'`' {
            code = !code;
            i += 1;
            continue;
        }
        if !code && bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            i += 2;
            continue;
        }
        if !code && bytes[i] == b'*' {
            i += 1;
            continue;
        }
        if !code && i + 1 < bytes.len() && bytes[i] == b'~' && bytes[i + 1] == b'~' {
            i += 2;
            continue;
        }
        if !code && bytes[i] == b'$' {
            let is_double = i + 1 < bytes.len() && bytes[i + 1] == b'$';
            let delim = if is_double { "$$" } else { "$" };
            if math {
                if delim == math_delim {
                    out.push_str(&crate::ai::stream::render_math_tex_to_unicode(math_buf.trim()));
                    math_buf.clear();
                    math = false;
                    i += delim.len();
                    continue;
                }
            } else {
                math = true;
                math_delim = delim;
                i += delim.len();
                continue;
            }
        }
        let ch = s[i..].chars().next().unwrap();
        if math && !code {
            math_buf.push(ch);
        } else {
            out.push(ch);
        }
        i += ch.len_utf8();
    }
    if math && !math_buf.is_empty() {
        out.push_str(&crate::ai::stream::render_math_tex_to_unicode(math_buf.trim()));
    }
    out
}

pub(super) fn visible_width(s: &str) -> usize {
    unicode_width::UnicodeWidthStr::width(strip_inline_md_markers(s).as_str())
}

pub(super) fn wrap_md_cell(s: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    if s.trim().is_empty() {
        return vec![String::new()];
    }

    let mut bold = false;
    let mut italic = false;
    let mut cur = String::new();
    let mut cur_w = 0usize;
    let mut lines: Vec<String> = Vec::new();

    fn style_prefix(bold: bool, italic: bool) -> String {
        let mut p = String::new();
        if bold {
            p.push_str("**");
        }
        if italic {
            p.push('*');
        }
        p
    }

    fn style_suffix(bold: bool, italic: bool) -> String {
        let mut s = String::new();
        if italic {
            s.push('*');
        }
        if bold {
            s.push_str("**");
        }
        s
    }

    let start_new_line = |cur: &mut String, cur_w: &mut usize, bold: bool, italic: bool| {
        *cur = style_prefix(bold, italic);
        *cur_w = 0;
    };

    let close_line = |lines: &mut Vec<String>, cur: &mut String, bold: bool, italic: bool| {
        cur.push_str(&style_suffix(bold, italic));
        lines.push(std::mem::take(cur));
    };

    let mut i = 0usize;
    start_new_line(&mut cur, &mut cur_w, bold, italic);

    while i < s.len() {
        let rest = &s[i..];

        if rest.starts_with("**") {
            bold = !bold;
            cur.push_str("**");
            i += 2;
            continue;
        }

        if rest.starts_with('*') && !rest.starts_with("**") {
            italic = !italic;
            cur.push('*');
            i += 1;
            continue;
        }

        if let Some((piece, next)) = take_atomic_markdown_span(s, i) {
            let piece_width = visible_width(&piece);
            if cur_w > 0 && cur_w + piece_width > width {
                close_line(&mut lines, &mut cur, bold, italic);
                start_new_line(&mut cur, &mut cur_w, bold, italic);
            }
            cur.push_str(&piece);
            cur_w += piece_width;
            i = next;
            continue;
        }

        let ch = rest.chars().next().unwrap();
        let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if cur_w > 0 && cur_w + w > width {
            close_line(&mut lines, &mut cur, bold, italic);
            start_new_line(&mut cur, &mut cur_w, bold, italic);
        }
        cur.push(ch);
        cur_w += w;
        i += ch.len_utf8();
    }

    close_line(&mut lines, &mut cur, bold, italic);
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn take_atomic_markdown_span(s: &str, start: usize) -> Option<(String, usize)> {
    let rest = &s[start..];

    if rest.starts_with('`') {
        let end = find_unescaped_delim(s, start + 1, "`")?;
        return Some((s[start..end].to_string(), end));
    }

    if rest.starts_with("~~") {
        let end = find_unescaped_delim(s, start + 2, "~~")?;
        return Some((s[start..end].to_string(), end));
    }

    if rest.starts_with("$$") {
        let end = find_unescaped_delim(s, start + 2, "$$")?;
        return Some((s[start..end].to_string(), end));
    }

    if rest.starts_with('$') {
        let end = find_unescaped_delim(s, start + 1, "$")?;
        return Some((s[start..end].to_string(), end));
    }

    // Single `*` for italic — grab until matching closing `*`
    if rest.starts_with('*') && !rest.starts_with("**") {
        let end = find_unescaped_delim(s, start + 1, "*")?;
        return Some((s[start..end].to_string(), end));
    }

    if let Some(stripped) = rest.strip_prefix('\\') {
        let next = stripped.chars().next()?;
        let end = start + 1 + next.len_utf8();
        return Some((s[start..end].to_string(), end));
    }

    None
}

fn find_unescaped_delim(s: &str, mut i: usize, delim: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    while i < bytes.len() {
        if s[i..].starts_with(delim) && !is_escaped_at(s, i) {
            return Some(i + delim.len());
        }
        let ch = s[i..].chars().next()?;
        i += ch.len_utf8();
    }
    None
}

fn is_escaped_at(s: &str, idx: usize) -> bool {
    if idx == 0 {
        return false;
    }

    let mut backslashes = 0usize;
    let mut i = idx;
    while i > 0 {
        let prev = s[..i].chars().next_back().unwrap();
        if prev != '\\' {
            break;
        }
        backslashes += 1;
        i -= prev.len_utf8();
    }
    backslashes % 2 == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::stream::render::code::{MONOKAI_BG, MONOKAI_FG};

    #[test]
    fn wrap_md_cell_uses_visible_width_for_math_and_code_spans() {
        let math = wrap_md_cell(r#"$\frac{1}{2}$"#, 5);
        assert_eq!(math, vec![r#"$\frac{1}{2}$"#]);

        let code = wrap_md_cell(r#"`a|b`"#, 3);
        assert_eq!(code, vec![r#"`a|b`"#]);
    }

    #[test]
    fn inline_code_uses_monokai_colors() {
        let rendered = render_inline_md("use `cargo test` please", "");
        assert!(rendered.contains(MONOKAI_BG));
        assert!(rendered.contains(MONOKAI_FG));
        assert!(rendered.contains("cargo test"));
    }

    #[test]
    fn italic_rendering() {
        let rendered = render_inline_md("*hello*", "");
        assert!(!rendered.contains("*hello*")); // markers consumed
        assert!(rendered.contains("hello"));
    }

    #[test]
    fn strikethrough_rendering() {
        let rendered = render_inline_md("~~deleted~~", "");
        assert!(!rendered.contains("~~deleted~~")); // markers consumed
        assert!(rendered.contains("deleted"));
    }

    #[test]
    fn bold_italic_combined() {
        let rendered = render_inline_md("***bold italic***", "");
        assert!(rendered.contains("bold italic"));
    }

    #[test]
    fn strip_markers_with_strikethrough() {
        let stripped = strip_inline_md_markers("~~text~~");
        assert_eq!(stripped, "text");
    }
}
