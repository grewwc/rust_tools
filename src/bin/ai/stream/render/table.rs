use crate::ai::stream::{
    extract::strip_ansi_codes,
    render::inline::{render_inline_md, visible_width, wrap_md_cell},
};

#[derive(Clone)]
pub(super) enum TableState {
    None,
    PendingHeader {
        indent: String,
        header_line: String,
        preview_height: usize,
    },
    InTable {
        indent: String,
        header: Vec<String>,
        align: Vec<TableAlign>,
        rows: Vec<Vec<String>>,
        preview_height: usize,
    },
}

#[derive(Clone, Copy)]
pub(super) enum TableAlign {
    Left,
    Center,
    Right,
}

pub(super) fn table_preview_height(line: &str) -> usize {
    let visible = strip_ansi_codes(line);
    let cols = terminal_width().max(1);
    let mut lines = 1usize;
    let mut current_col = 0usize;

    for ch in visible.chars() {
        let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if current_col > 0 && current_col + w > cols {
            lines += 1;
            current_col = w;
        } else {
            current_col += w;
        }
    }

    lines
}

pub(super) fn split_indent(s: &str) -> (&str, &str) {
    let mut idx = 0usize;
    for (i, ch) in s.char_indices() {
        if ch == ' ' || ch == '\t' {
            idx = i + ch.len_utf8();
            continue;
        }
        idx = i;
        break;
    }
    if s.chars().all(|c| c == ' ' || c == '\t') {
        return (s, "");
    }
    s.split_at(idx)
}

pub(in crate::ai::stream) fn line_looks_like_table_preview(line: &str) -> bool {
    if line.trim().is_empty() {
        return false;
    }
    let (_, rest) = split_indent(line);
    let s = rest.trim_end();
    if s.starts_with("```") || s.starts_with("~~~") {
        return false;
    }
    s.contains('|')
}

pub(super) fn is_table_row_candidate(line: &str) -> bool {
    if line.trim().is_empty() {
        return false;
    }
    let (_, rest) = split_indent(line);
    let s = rest.trim_end();
    if !s.contains('|') {
        return false;
    }
    if s.starts_with("```") || s.starts_with("~~~") {
        return false;
    }
    let cells = parse_table_row(s);
    cells.len() >= 2
}

pub(super) fn is_table_row(line: &str) -> bool {
    let (_, rest) = split_indent(line);
    let s = rest.trim_end();
    if s.trim().is_empty() {
        return false;
    }
    if is_table_separator(s) {
        return false;
    }
    let cells = parse_table_row(s);
    cells.len() >= 2
}

pub(super) fn is_table_separator(line: &str) -> bool {
    let (_, rest) = split_indent(line);
    let mut s = rest.trim();
    if s.starts_with('|') {
        s = &s[1..];
    }
    if s.ends_with('|') && !s.is_empty() {
        s = &s[..s.len() - 1];
    }
    let parts = split_table_segments(s)
        .into_iter()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty());
    let mut count = 0usize;
    for p in parts {
        count += 1;
        let p = p.trim_matches(' ');
        let core = p.trim_matches(':');
        if core.is_empty() || !core.chars().all(|c| c == '-') {
            return false;
        }
    }
    count >= 2
}

pub(super) fn parse_table_row(line: &str) -> Vec<String> {
    let (_, rest) = split_indent(line);
    let s = rest.trim();
    let mut raw = split_table_segments(s);
    if s.starts_with('|') && !raw.is_empty() && raw.first().is_some_and(|x| x.is_empty()) {
        raw.remove(0);
    }
    if s.ends_with('|') && !raw.is_empty() && raw.last().is_some_and(|x| x.is_empty()) {
        raw.pop();
    }
    raw.into_iter().map(|x| x.trim().to_string()).collect()
}

pub(super) fn parse_table_align(line: &str, cols: usize) -> Vec<TableAlign> {
    let (_, rest) = split_indent(line);
    let s = rest.trim();
    let mut raw = split_table_segments(s);
    if s.starts_with('|') && !raw.is_empty() && raw.first().is_some_and(|x| x.is_empty()) {
        raw.remove(0);
    }
    if s.ends_with('|') && !raw.is_empty() && raw.last().is_some_and(|x| x.is_empty()) {
        raw.pop();
    }
    let mut out = Vec::with_capacity(cols);
    for seg in raw
        .iter()
        .map(|s| s.as_str())
        .chain(std::iter::repeat(""))
        .take(cols)
    {
        let seg = seg.trim();
        let left = seg.starts_with(':');
        let right = seg.ends_with(':');
        out.push(match (left, right) {
            (true, true) => TableAlign::Center,
            (false, true) => TableAlign::Right,
            _ => TableAlign::Left,
        });
    }
    out
}

fn split_table_segments(s: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut chars = s.chars().peekable();
    let mut in_code = false;
    let mut in_math = false;
    let mut escaped = false;

    while let Some(ch) = chars.next() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }

        if ch == '\\' {
            current.push(ch);
            escaped = true;
            continue;
        }

        if ch == '`' && !in_math {
            in_code = !in_code;
            current.push(ch);
            continue;
        }

        if ch == '$' && !in_code {
            if chars.peek().copied() == Some('$') {
                chars.next();
                in_math = !in_math;
                current.push('$');
                current.push('$');
                continue;
            }

            in_math = !in_math;
            current.push(ch);
            continue;
        }

        if ch == '|' && !in_code && !in_math {
            segments.push(std::mem::take(&mut current));
            continue;
        }

        current.push(ch);
    }

    segments.push(current);
    segments
}

fn pad_cell(s: &str, width: usize, align: TableAlign) -> String {
    let w = visible_width(s);
    let pad = width.saturating_sub(w);
    match align {
        TableAlign::Left => format!("{s}{}", " ".repeat(pad)),
        TableAlign::Right => format!("{}{}", " ".repeat(pad), s),
        TableAlign::Center => {
            let left = pad / 2;
            let right = pad - left;
            format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
        }
    }
}

pub(super) fn compute_table_widths(
    indent: &str,
    header: &[String],
    rows: &[Vec<String>],
) -> Vec<usize> {
    let cols = header
        .len()
        .max(rows.iter().map(|r| r.len()).max().unwrap_or(0));
    if cols == 0 {
        return Vec::new();
    }

    let mut widths = vec![3usize; cols];
    for (i, cell) in header.iter().enumerate() {
        widths[i] = widths[i].max(visible_width(cell));
    }
    for row in rows {
        for (width, cell) in widths
            .iter_mut()
            .zip(row.iter().map(|s| s.as_str()).chain(std::iter::repeat("")))
        {
            *width = (*width).max(visible_width(cell));
        }
    }
    for w in &mut widths {
        *w = (*w).max(3);
    }

    let term_cols = terminal_width();
    let indent_w = unicode_width::UnicodeWidthStr::width(indent);
    let max_total = term_cols.saturating_sub(indent_w).max(20);
    let avail = max_total.saturating_sub(3 * cols + 1);
    if avail == 0 {
        return widths;
    }

    let min_w = 3usize;
    let sum = widths.iter().sum::<usize>();
    if avail < min_w * cols {
        let base = (avail / cols).max(1);
        let mut rem = avail.saturating_sub(base * cols);
        for w in &mut widths {
            *w = base;
            if rem > 0 {
                *w += 1;
                rem -= 1;
            }
        }
        return widths;
    }

    if sum > avail {
        let mut excess = sum - avail;
        while excess > 0 {
            // Find the index of the column with the maximum width
            let mut max_idx = 0;
            let mut max_w = 0;
            for (i, &w) in widths.iter().enumerate() {
                if w > max_w {
                    max_w = w;
                    max_idx = i;
                }
            }
            
            if max_w <= min_w {
                break; // Cannot reduce further
            }
            
            widths[max_idx] -= 1;
            excess -= 1;
        }
    }

    widths
}

pub(super) fn render_table_top(indent: &str, widths: &[usize]) -> String {
    let cols = widths.len();
    if cols < 2 {
        return String::new();
    }
    let mut out = String::new();
    out.push_str(indent);
    out.push('┌');
    for (i, width) in widths.iter().enumerate() {
        out.push_str(&"─".repeat(*width + 2));
        out.push(if i + 1 == cols { '┐' } else { '┬' });
    }
    out.push('\n');
    out
}

pub(super) fn render_table_mid(indent: &str, widths: &[usize]) -> String {
    let cols = widths.len();
    if cols < 2 {
        return String::new();
    }
    let mut out = String::new();
    out.push_str(indent);
    out.push('├');
    for (i, width) in widths.iter().enumerate() {
        out.push_str(&"─".repeat(*width + 2));
        out.push(if i + 1 == cols { '┤' } else { '┼' });
    }
    out.push('\n');
    out
}

pub(super) fn render_table_bottom(indent: &str, widths: &[usize]) -> String {
    let cols = widths.len();
    if cols < 2 {
        return String::new();
    }
    let mut out = String::new();
    out.push_str(indent);
    out.push('└');
    for (i, width) in widths.iter().enumerate() {
        out.push_str(&"─".repeat(*width + 2));
        out.push(if i + 1 == cols { '┘' } else { '┴' });
    }
    out.push('\n');
    out
}

pub(super) fn render_table_header(
    indent: &str,
    header: &[String],
    align: &[TableAlign],
    widths: &[usize],
) -> String {
    let cols = widths.len();
    if cols < 2 {
        return String::new();
    }

    let header_lines = header
        .iter()
        .enumerate()
        .map(|(i, cell)| wrap_md_cell(cell, *widths.get(i).unwrap_or(&3)))
        .collect::<Vec<_>>();
    let header_height = header_lines.iter().map(|c| c.len()).max().unwrap_or(1);

    let mut out = String::new();
    for line_idx in 0..header_height {
        out.push_str(indent);
        out.push('│');
        for (i, width) in widths.iter().enumerate() {
            let cell_line = header_lines
                .get(i)
                .and_then(|ls| ls.get(line_idx))
                .map(|s| s.as_str())
                .unwrap_or("");
            let padded = pad_cell(
                cell_line,
                *width,
                align.get(i).copied().unwrap_or(TableAlign::Left),
            );
            out.push(' ');
            out.push_str("\x1b[1m\x1b[36m");
            out.push_str(&render_inline_md(&padded, "\x1b[1m\x1b[36m"));
            out.push_str("\x1b[0m");
            out.push(' ');
            out.push('│');
        }
        out.push('\n');
    }
    out
}

pub(super) fn render_table_row(
    indent: &str,
    row: &[String],
    align: &[TableAlign],
    widths: &[usize],
) -> String {
    let cols = widths.len();
    if cols < 2 {
        return String::new();
    }

    let wrapped = widths
        .iter()
        .enumerate()
        .map(|(i, width)| wrap_md_cell(row.get(i).map(|s| s.as_str()).unwrap_or(""), *width))
        .collect::<Vec<_>>();
    let height = wrapped.iter().map(|c| c.len()).max().unwrap_or(1);

    let mut out = String::new();
    for line_idx in 0..height {
        out.push_str(indent);
        out.push('│');
        for (i, width) in widths.iter().enumerate() {
            let cell_line = wrapped
                .get(i)
                .and_then(|ls| ls.get(line_idx))
                .map(|s| s.as_str())
                .unwrap_or("");
            let padded = pad_cell(
                cell_line,
                *width,
                align.get(i).copied().unwrap_or(TableAlign::Left),
            );
            out.push(' ');
            out.push_str(&render_inline_md(&padded, ""));
            out.push(' ');
            out.push('│');
        }
        out.push('\n');
    }
    out
}

fn terminal_width() -> usize {
    if let Some(cols) = std::env::var("COLUMNS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        && cols > 0
    {
        return cols;
    }

    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = std::io::stdout().as_raw_fd();
        let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &mut ws) };
        if rc == 0 && ws.ws_col > 0 {
            return ws.ws_col as usize;
        }
    }

    80
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::stream::render::inline::visible_width;

    #[test]
    fn parse_table_row_ignores_embedded_pipes() {
        assert_eq!(
            parse_table_row(r#"| `a|b` | x \| y | $p|q$ |"#),
            vec!["`a|b`", r#"x \| y"#, "$p|q$"]
        );
    }

    #[test]
    fn compute_table_widths_does_not_add_columns_for_embedded_pipes() {
        let header = parse_table_row("| name | value |");
        let rows = vec![parse_table_row(r#"| `a|b` | $\frac{1}{2}$ |"#)];
        let widths = compute_table_widths("", &header, &rows);

        assert_eq!(widths.len(), 2);
        assert!(widths[0] >= visible_width("`a|b`"));
        assert!(widths[1] >= visible_width(r#"$\frac{1}{2}$"#));
    }

    #[test]
    fn table_preview_height_ignores_ansi_sequences() {
        let plain = "a".repeat(200);
        let colored = format!("\x1b[2m{plain}\x1b[0m");
        assert_eq!(table_preview_height(&colored), table_preview_height(&plain));
    }
}
