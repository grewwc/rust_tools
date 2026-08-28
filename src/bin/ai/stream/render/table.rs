use crate::ai::stream::{
    extract::strip_ansi_codes,
    render::inline::{
        render_inline_md, strip_redundant_vs16, terminal_cell_width, terminal_display_width,
        visible_width, wrap_md_cell,
    },
};

const MIN_TABLE_CELL_WIDTH: usize = 6;

#[derive(Clone)]
pub(super) enum TableState {
    None,
    PendingHeader {
        indent: String,
        header_line: String,
    },
    InTable {
        indent: String,
        header: Vec<String>,
        align: Vec<TableAlign>,
        rows: Vec<Vec<String>>,
    },
}

#[derive(Clone, Copy)]
pub(super) enum TableAlign {
    Left,
    Center,
    Right,
}

pub(super) fn table_preview_height(line: &str) -> usize {
    // This counts how many physical rows the "preview lines written verbatim to the terminal"
    // occupy. The terminal wraps them at the **real** column width (raw_cols), so the margin-aware
    // terminal_width must not be used — it overestimates the row count and makes cursor-up erase past the top.
    let visible = strip_ansi_codes(line);
    let cols = raw_cols().max(1);
    let mut lines = 1usize;
    let mut current_col = 0usize;

    for ch in visible.chars() {
        let w = terminal_cell_width(ch);
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
    (cells.len() >= 2 || explicit_single_column_table_line(s, &cells))
        && header_candidate_has_clear_table_boundary(s, &cells)
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
    cells.len() >= 2 || explicit_single_column_table_line(s, &cells)
}

pub(super) fn is_table_separator(line: &str) -> bool {
    let (_, rest) = split_indent(line);
    let mut s = rest.trim();
    let explicit_boundary = has_explicit_table_boundaries(s);
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
    count >= 2 || (count == 1 && explicit_boundary)
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

fn header_candidate_has_clear_table_boundary(s: &str, cells: &[String]) -> bool {
    if s.starts_with('|') {
        return cells.iter().any(|cell| !cell.trim().is_empty());
    }

    let Some(first) = cells.first().map(|cell| cell.trim()) else {
        return false;
    };
    let Some(last) = cells.last().map(|cell| cell.trim()) else {
        return false;
    };
    if first.is_empty() || last.is_empty() {
        return false;
    }

    if starts_with_non_table_block_prefix(first) || ends_with_sentence_punctuation(first) {
        return false;
    }

    true
}

fn explicit_single_column_table_line(s: &str, cells: &[String]) -> bool {
    cells.len() == 1 && has_explicit_table_boundaries(s)
}

fn has_explicit_table_boundaries(s: &str) -> bool {
    let s = s.trim();
    s.len() >= 2 && s.starts_with('|') && s.ends_with('|')
}

fn starts_with_non_table_block_prefix(s: &str) -> bool {
    s.starts_with("> ")
        || s.starts_with("- ")
        || s.starts_with("* ")
        || s.starts_with("+ ")
        || has_ordered_list_prefix(s)
}

fn has_ordered_list_prefix(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
        if i > 4 {
            break;
        }
    }
    i > 0 && i + 1 < bytes.len() && bytes[i] == b'.' && bytes[i + 1] == b' '
}

fn ends_with_sentence_punctuation(s: &str) -> bool {
    matches!(
        s.chars().last(),
        Some(':' | '：' | '。' | '，' | '；' | '！' | '？')
    )
}

fn split_table_segments(s: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut chars = s.chars().peekable();
    let mut in_code = false;
    let mut in_math = false;
    let mut math_delim = ""; // "$" or "$$": the delimiter active when the math segment was entered
    let mut in_strike = false;
    let mut escaped = false;

    /// Scans forward for an unescaped target character. Used to check whether a delimiter has a
    /// matching pair, so an unclosed ` or $ cannot leave in_code/in_math stuck at true and make
    /// later | pipes unrecognized as column separators.
    fn has_matching_delim(chars: &std::iter::Peekable<std::str::Chars>, target: char) -> bool {
        let mut la = chars.clone();
        let mut esc = false;
        while let Some(c) = la.next() {
            if esc {
                esc = false;
                continue;
            }
            if c == '\\' {
                esc = true;
                continue;
            }
            if c == target {
                return true;
            }
        }
        false
    }

    /// Scans forward for an unescaped two-character delimiter (used for ~~ and $$).
    fn has_matching_delim_pair(
        chars: &std::iter::Peekable<std::str::Chars>,
        first: char,
        second: char,
    ) -> bool {
        let mut la = chars.clone();
        // Skip the character at the current peek position — it is part of the opening delimiter
        // (e.g. the second $ of $$) and must not be paired with the following characters into a
        // "closing pair". In $$$, the opening delimiter takes the first two $; if not skipped, the
        // lookahead would misread the 2nd and 3rd $ as a closing $$ and enter math mode forever.
        la.next();
        let mut esc = false;
        while let Some(c) = la.next() {
            if esc {
                esc = false;
                continue;
            }
            if c == '\\' {
                esc = true;
                continue;
            }
            if c == first && la.peek().copied() == Some(second) {
                return true;
            }
        }
        false
    }

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

        // ~~strikethrough~~: track the state so a | inside ~~ is not misread as a column separator
        if ch == '~' && !in_code && !in_math && chars.peek().copied() == Some('~') {
            if in_strike {
                // Already inside a strikethrough: this is the closing ~~
                chars.next();
                in_strike = false;
                current.push('~');
                current.push('~');
                continue;
            }
            // Not inside a strikethrough: check for a matching ~~ pair
            if has_matching_delim_pair(&chars, '~', '~') {
                chars.next();
                in_strike = true;
                current.push('~');
                current.push('~');
                continue;
            }
        }

        if ch == '`' {
            if in_code {
                // Already inside a code span: this is the closing backtick
                in_code = false;
                current.push(ch);
                continue;
            }
            if !in_math && has_matching_delim(&chars, '`') {
                // Not inside a code span and a matching backtick exists: enter the code span
                in_code = true;
                current.push(ch);
                continue;
            }
        }

        if ch == '$' && !in_code {
            if in_math {
                // Already inside a math segment: decide closure by the delimiter type entered with
                match math_delim {
                    "$$" if chars.peek().copied() == Some('$') => {
                        // A $$ segment meeting a $$ pair: close
                        chars.next();
                        in_math = false;
                        math_delim = "";
                        current.push('$');
                        current.push('$');
                        continue;
                    }
                    "$" => {
                        // A $ segment meeting a single $: close (even if a $ follows)
                        in_math = false;
                        math_delim = "";
                        current.push(ch);
                        continue;
                    }
                    _ => {
                        // A single $ inside a $$ segment (peek is not $), or an unknown state:
                        // treat it literally, do not close, and fall through to current.push(ch)
                    }
                }
            }
            // Not inside a math segment: check for a pair
            if chars.peek().copied() == Some('$') {
                if has_matching_delim_pair(&chars, '$', '$') {
                    chars.next();
                    in_math = true;
                    math_delim = "$$";
                    current.push('$');
                    current.push('$');
                    continue;
                }
            } else if has_matching_delim(&chars, '$') {
                in_math = true;
                math_delim = "$";
                current.push(ch);
                continue;
            }
        }

        if ch == '|' && !in_code && !in_math && !in_strike {
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

/// Render inline markdown first, then pad spaces based on the rendered display width.
///
/// The `pad_cell` + `render_inline_md(padded)` order must not be used: `pad_cell` relies on
/// `visible_width` (which strips unclosed `, **, and * markers), but `render_inline_md` emits
/// unclosed markers verbatim as literal characters, making the real width exceed the estimate and misaligning table borders.
fn render_and_pad_cell(cell_line: &str, width: usize, align: TableAlign, base: &str) -> String {
    let rendered = render_inline_md(cell_line, base);
    // Strip redundant VS16 from the rendered text. When VS16 follows an
    // is_ambiguous_emoji_block_char, the base already renders as 2-col emoji;
    // keeping VS16 in the string would add an extra column in the terminal.
    let rendered = strip_redundant_vs16(&rendered);
    let ansi_stripped = strip_ansi_codes(&rendered);
    let actual_w = terminal_display_width(ansi_stripped.as_str());
    let pad = width.saturating_sub(actual_w);
    match align {
        TableAlign::Left => format!("{rendered}{}", " ".repeat(pad)),
        TableAlign::Right => format!("{}{rendered}", " ".repeat(pad)),
        TableAlign::Center => {
            let left = pad / 2;
            let right = pad - left;
            format!("{}{rendered}{}", " ".repeat(left), " ".repeat(right))
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
        *w = (*w).max(MIN_TABLE_CELL_WIDTH);
    }

    let max_total = table_available_width(indent);
    let avail = max_total.saturating_sub(3 * cols + 1);

    let min_w = if avail >= MIN_TABLE_CELL_WIDTH * cols {
        MIN_TABLE_CELL_WIDTH
    } else {
        avail / cols
    };
    let sum = widths.iter().sum::<usize>();

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

pub(super) fn table_column_ranges(indent: &str, cols: usize) -> Vec<std::ops::Range<usize>> {
    if cols == 0 {
        return Vec::new();
    }
    let max_cols = max_columns_per_table_block(indent).max(1);
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < cols {
        let end = (start + max_cols).min(cols);
        ranges.push(start..end);
        start = end;
    }
    ranges
}

fn max_columns_per_table_block(indent: &str) -> usize {
    let max_total = table_available_width(indent);
    max_total
        .saturating_sub(1)
        .checked_div(MIN_TABLE_CELL_WIDTH + 3)
        .unwrap_or(0)
        .max(1)
}

fn table_available_width(indent: &str) -> usize {
    terminal_width()
        .saturating_sub(terminal_display_width(indent))
        .max(1)
}

pub(super) fn render_table_top(indent: &str, widths: &[usize]) -> String {
    let cols = widths.len();
    if cols == 0 {
        return String::new();
    }
    let mut out = String::new();
    out.push_str(indent);
    out.push('┌');
    for (i, width) in widths.iter().enumerate() {
        out.push_str(&"─".repeat(width + 2));
        out.push(if i + 1 == cols { '┐' } else { '┬' });
    }
    out.push('\n');
    out
}

pub(super) fn render_table_mid(indent: &str, widths: &[usize]) -> String {
    let cols = widths.len();
    if cols == 0 {
        return String::new();
    }
    let mut out = String::new();
    out.push_str(indent);
    out.push('├');
    for (i, width) in widths.iter().enumerate() {
        out.push_str(&"─".repeat(width + 2));
        out.push(if i + 1 == cols { '┤' } else { '┼' });
    }
    out.push('\n');
    out
}

pub(super) fn render_table_bottom(indent: &str, widths: &[usize]) -> String {
    let cols = widths.len();
    if cols == 0 {
        return String::new();
    }
    let mut out = String::new();
    out.push_str(indent);
    out.push('└');
    for (i, width) in widths.iter().enumerate() {
        out.push_str(&"─".repeat(width + 2));
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
    if cols == 0 {
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
            let padded = render_and_pad_cell(
                cell_line,
                *width,
                align.get(i).copied().unwrap_or(TableAlign::Left),
                "",
            );
            out.push(' ');
            out.push_str("\x1b[1m\x1b[36m");
            out.push_str(&padded);
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
    if cols == 0 {
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
            let padded = render_and_pad_cell(
                cell_line,
                *width,
                align.get(i).copied().unwrap_or(TableAlign::Left),
                "",
            );
            out.push(' ');
            out.push_str(&padded);
            out.push(' ');
            out.push('│');
        }
        out.push('\n');
    }
    out
}

fn terminal_width() -> usize {
    // Matches markdown.rs::preview_terminal_width: keep a 4-column right safety margin so the
    // table │ borders never sit flush against the terminal's right edge, which would trigger
    // automatic wrapping and break the box-drawing layout.
    const RIGHT_MARGIN: usize = 4;
    const MIN_WIDTH: usize = 20;

    let raw = raw_cols();
    raw.saturating_sub(RIGHT_MARGIN).max(MIN_WIDTH)
}

fn raw_cols() -> usize {
    // Matches markdown.rs::raw_terminal_cols: prefer a live ioctl query and use COLUMNS only as
    // a non-tty fallback. After a resident process is narrowed inside a panel, COLUMNS is a stale
    // snapshot; computing table width from it overflows the real panel, gets hard-wrapped by the
    // terminal, and misaligns the │ borders.
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

    if let Some(cols) = std::env::var("COLUMNS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        && cols > 0
    {
        return cols;
    }

    80
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::stream::render::inline::visible_width;
    use crate::ai::test_support::ENV_LOCK;

    #[test]
    fn parse_table_row_ignores_embedded_pipes() {
        assert_eq!(
            parse_table_row(r#"| `a|b` | x \| y | $p|q$ |"#),
            vec!["`a|b`", r#"x \| y"#, "$p|q$"]
        );
    }

    #[test]
    fn parse_table_row_handles_unpaired_backticks() {
        // An odd number of three backticks must not leave in_code stuck at true, otherwise the
        // trailing | can no longer be recognized as a column separator.
        // Before the fix: split_table_segments toggled in_code on every backtick; after ``` the
        //   in_code flag was true and later | characters were treated as literals, so the whole
        //   row parsed as a single cell.
        // After the fix: in_code is entered only when a paired backtick exists; unpaired ` is
        //   treated as a literal.
        assert_eq!(
            parse_table_row("| context 行漏前导空格 | ❌ invalid hunk line: ``` |"),
            vec!["context 行漏前导空格", "❌ invalid hunk line: ```"]
        );
        // A single unclosed backtick should also be treated as a literal
        assert_eq!(
            parse_table_row("| foo `bar | baz |"),
            vec!["foo `bar", "baz"]
        );
    }

    #[test]
    fn parse_table_row_handles_unpaired_dollar() {
        // An unpaired $ must not leave in_math stuck at true
        assert_eq!(
            parse_table_row("| foo $bar | baz |"),
            vec!["foo $bar", "baz"]
        );
        // Three $ characters (odd count) should also be handled correctly
        assert_eq!(parse_table_row("| a $$$ b | c |"), vec!["a $$$ b", "c"]);
    }

    #[test]
    fn parse_table_row_dollar_inside_code_is_literal() {
        // $ inside a backtick code span must not trigger math mode
        assert_eq!(parse_table_row("| `a$b` | c |"), vec!["`a$b`", "c"]);
    }

    #[test]
    fn parse_table_row_single_dollar_math_closes_at_single_dollar() {
        // For a math span entered with a single $, a $$ sequence must close the span at the
        // first $; the second $ is treated as a literal (it may open a new math span).
        // Key case: in $a$$b$, $a$ is one math span and $b$ is another.
        let cells = parse_table_row("| $a$$b$ | c |");
        // Two cells: the first contains "$a$$b$" (two adjacent math spans), the second is "c"
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[1], "c");
        // The first cell's content must contain all $ characters
        assert!(cells[0].contains("$$b$"));
    }

    #[test]
    fn parse_table_row_double_dollar_math_requires_double_dollar_to_close() {
        // For a math span entered with $$, a single $ must not trigger closure
        let cells = parse_table_row("| $$a$b$$ | c |");
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[1], "c");
        // In $$a$b$$, the $ is a literal inside the $$ span; the whole run is one math span
        assert_eq!(cells[0], "$$a$b$$");
    }

    #[test]
    fn parse_table_row_handles_unpaired_strikethrough() {
        // Unpaired ~~ must be treated as literals and must not leave in_strike stuck
        assert_eq!(parse_table_row("| ~~a | b |"), vec!["~~a", "b"]);
        // Three ~~ sequences (odd count) should also be handled correctly
        assert_eq!(parse_table_row("| ~~a~~b~~ | c |"), vec!["~~a~~b~~", "c"]);
    }

    #[test]
    fn parse_table_row_escaped_delimiters() {
        // An escaped backtick must not open a code span
        assert_eq!(parse_table_row(r#"| \`a\` | b |"#), vec![r#"\`a\`"#, "b"]);
        // An escaped $ must not open a math span
        assert_eq!(parse_table_row(r#"| \$a\$ | b |"#), vec![r#"\$a\$"#, "b"]);
    }

    #[test]
    fn parse_table_row_pipe_inside_math_is_literal() {
        // A | inside a $ math span must be treated as a literal
        assert_eq!(parse_table_row("| $a|b$ | c |"), vec!["$a|b$", "c"]);
    }

    #[test]
    fn parse_table_row_adjacent_code_spans() {
        // Adjacent backtick code spans: `a`b`c` should be parsed as two code spans `a` + `c`,
        // with the middle b as plain text; the | must not be swallowed
        let cells = parse_table_row("| `a`b`c` | d |");
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[1], "d");
        assert_eq!(cells[0], "`a`b`c`");
    }

    #[test]
    fn parse_table_row_ignores_pipes_inside_strikethrough() {
        assert_eq!(
            parse_table_row(r#"| ~~a|b~~ | normal |"#),
            vec!["~~a|b~~", "normal"]
        );
        // Multiple ~~ spans mixed with plain text
        assert_eq!(
            parse_table_row(r#"| ~~x|y~~ | z | ~~w|v~~ |"#),
            vec!["~~x|y~~", "z", "~~w|v~~"]
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
    fn compute_table_widths_respects_remaining_content_budget() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        unsafe { std::env::set_var("COLUMNS", "40") };

        let header = (0..10)
            .map(|idx| format!("very_long_column_name_{idx}"))
            .collect::<Vec<_>>();
        let widths = compute_table_widths("", &header, &[]);

        assert_eq!(widths.len(), 10);
        assert!(
            widths.iter().sum::<usize>() <= 5,
            "content widths must not exceed the remaining budget: {widths:?}"
        );
    }

    #[test]
    fn table_column_ranges_split_overwide_tables() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        unsafe { std::env::set_var("COLUMNS", "80") };

        let ranges = table_column_ranges("", 20);

        assert!(ranges.len() > 1, "{ranges:?}");
        assert!(
            ranges.iter().all(|range| range.len() <= 8),
            "each split table should keep readable minimum width: {ranges:?}"
        );
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, 20);
    }

    #[test]
    fn bare_table_candidate_accepts_simple_header_without_leading_pipe() {
        assert!(is_table_row_candidate("时间 | 线程 | 前置事件"));
    }

    #[test]
    fn bare_table_candidate_rejects_sentence_prefixed_line() {
        assert!(!is_table_row_candidate("两条记录对应: | 时间 | 线程 |"));
        assert!(!is_table_row_candidate("两条记录对应：| 时间 | 线程 |"));
    }

    #[test]
    fn bare_table_candidate_rejects_list_like_prefix() {
        assert!(!is_table_row_candidate("- 时间 | 线程 | 前置事件"));
        assert!(!is_table_row_candidate("1. 时间 | 线程 | 前置事件"));
    }

    #[test]
    fn explicit_single_column_table_is_recognized() {
        assert!(is_table_row_candidate("| 函数签名 |"));
        assert!(is_table_separator("| --- |"));
        assert!(is_table_row("| `processOrder()` |"));
        assert!(!is_table_row_candidate("函数签名 |"));
    }

    #[test]
    fn table_preview_height_ignores_ansi_sequences() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        unsafe { std::env::set_var("COLUMNS", "40") };
        let plain = "a".repeat(200);
        let colored = format!("\x1b[2m{plain}\x1b[0m");
        assert_eq!(table_preview_height(&colored), table_preview_height(&plain));
    }

    #[test]
    fn pad_cell_aligns_emojis_correctly() {
        let w1 = visible_width("🌧️天气");
        let w2 = visible_width("💧湿度");
        let w3 = visible_width("🍃空气质量");

        // The width model always adds +1 for U+FE0F, widening a base character that needs VS16 to
        // render as emoji to 2 columns, matching real terminals. 🌧️ = U+1F327(unicode-width=1) +
        // U+FE0F(+1) = 2 columns; "天气" is 4 columns, totaling 6. The old code counted VS16 as 0
        // columns, so 🌧️ measured 1 column — one narrower than the terminal. That was the root
        // cause of tables being hard-wrapped with header residue piling up.
        assert_eq!(w1, 6);
        assert_eq!(w2, 6); // 💧(2) + 湿(2) + 度(2)
        assert_eq!(w3, 10); // 🍃(2) + 空气质量(8)

        let p1 = pad_cell("🌧️天气", 10, TableAlign::Left);
        let p2 = pad_cell("💧湿度", 10, TableAlign::Left);

        assert_eq!(p1, "🌧️天气    "); // padded 4 spaces
        assert_eq!(p2, "💧湿度    "); // padded 4 spaces
    }

    #[test]
    fn table_borders_use_box_drawing_horizontal_fill_not_ascii_hyphen() {
        let widths = vec![4usize, 6usize];
        for border in [
            render_table_top("", &widths),
            render_table_mid("", &widths),
            render_table_bottom("", &widths),
        ] {
            assert!(
                border.contains('─'),
                "border must fill with box-drawing '─': {border:?}"
            );
            assert!(
                !border.contains('-'),
                "border must not contain ASCII hyphen: {border:?}"
            );
        }

        // The top border must form a continuous box-drawing line (┌─...─┬─...─┐), appearing as a
        // solid line rather than a dashed one.
        let top = render_table_top("", &widths);
        assert!(
            top.contains('┌') && top.contains('┬') && top.contains('┐'),
            "{top:?}"
        );
        assert!(top.contains("┌──────"), "{top:?}");
    }

    #[test]
    fn render_and_pad_cell_compensates_for_unclosed_marker_literal_output() {
        // Unclosed `, **, and * markers are emitted by render_inline_md as literal characters
        // (rather than stripped), so the rendered real display width exceeds the visible_width
        // estimate. render_and_pad_cell must pad based on the actual width after rendering.
        use crate::ai::stream::extract::strip_ansi_codes;

        fn rendered_display_width(s: &str) -> usize {
            let visible = strip_ansi_codes(s);
            terminal_display_width(visible.as_str())
        }

        // Target width 10; unclosed backtick: visible_width strips the ` but render_inline_md
        // outputs it verbatim
        let padded = render_and_pad_cell("`foo", 10, TableAlign::Left, "");
        let actual_w = rendered_display_width(&padded);
        assert_eq!(
            actual_w, 10,
            "unclosed backtick: padded width should be 10, got {actual_w}, padded={padded:?}"
        );

        // Unclosed **: visible_width strips it but render_inline_md outputs it verbatim (+2 cols)
        let padded = render_and_pad_cell("**foo", 10, TableAlign::Left, "");
        let actual_w = rendered_display_width(&padded);
        assert_eq!(
            actual_w, 10,
            "unclosed **: padded width should be 10, got {actual_w}, padded={padded:?}"
        );

        // Unclosed *: visible_width strips it but render_inline_md outputs it verbatim (+1 col)
        let padded = render_and_pad_cell("*foo", 10, TableAlign::Left, "");
        let actual_w = rendered_display_width(&padded);
        assert_eq!(
            actual_w, 10,
            "unclosed *: padded width should be 10, got {actual_w}, padded={padded:?}"
        );

        // Closed markers should work normally
        let padded = render_and_pad_cell("`foo`", 10, TableAlign::Left, "");
        let actual_w = rendered_display_width(&padded);
        assert_eq!(
            actual_w, 10,
            "closed backtick: padded width should be 10, got {actual_w}, padded={padded:?}"
        );
    }

    #[test]
    fn render_and_pad_cell_treats_box_drawing_as_single_width() {
        use crate::ai::stream::extract::strip_ansi_codes;

        let padded = render_and_pad_cell("────", 6, TableAlign::Left, "");
        let visible = strip_ansi_codes(&padded);

        assert_eq!(terminal_display_width(visible.as_str()), 6);
        assert_eq!(visible, "────  ");
    }

    /// Reproduce the exact table from the screenshot: 3 columns, emoji in first
    /// column of each data row. Verify that every `│` separator sits at the same
    /// column position across all rendered lines (header + data rows).
    #[test]
    fn screenshot_table_borders_align_with_emoji_first_column() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        unsafe { std::env::set_var("COLUMNS", "120") };
        let header = vec!["严重度".to_string(), "问题".to_string(), "状态".to_string()];
        let align = vec![TableAlign::Left, TableAlign::Left, TableAlign::Left];
        let rows = vec![
            vec![
                "🐛 上次 #1: switch -f 遗漏".to_string(),
                "已修复".to_string(),
                "✅".to_string(),
            ],
            vec![
                "⚠️ 标签值测试 v1.2.3".to_string(),
                "正确放行".to_string(),
                "✅".to_string(),
            ],
            vec![
                "📝 无扩展名文件的路径 checkout 局限".to_string(),
                "有注释，合理取舍".to_string(),
                "✅".to_string(),
            ],
        ];

        let widths = compute_table_widths("", &header, &rows);
        let top = render_table_top("", &widths);
        let mid = render_table_mid("", &widths);
        let hdr = render_table_header("", &header, &align, &widths);
        let bot = render_table_bottom("", &widths);

        let mut all_lines = Vec::new();
        all_lines.push(top);
        all_lines.push(hdr);
        all_lines.push(mid);
        for row in &rows {
            all_lines.push(render_table_row("", row, &align, &widths));
        }
        all_lines.push(bot);

        // Find the byte-offset of every `│` in each line and verify they all
        // occupy the same visual column positions.
        let mut separator_positions: Option<Vec<usize>> = None;
        for line in &all_lines {
            let stripped = strip_ansi_codes(line);
            let positions: Vec<usize> = stripped
                .char_indices()
                .filter(|&(_, ch)| ch == '│')
                .map(|(byte_idx, _)| {
                    // Convert byte index to visual column by summing widths of
                    // all preceding characters.
                    stripped[..byte_idx]
                        .chars()
                        .map(|c| terminal_cell_width(c))
                        .sum()
                })
                .collect();
            if positions.is_empty() {
                continue; // border-only lines (┌─┐, ├─┤, └─┘) have no │
            }
            if let Some(expected) = &separator_positions {
                assert_eq!(
                    &positions, expected,
                    "│ column positions differ on line: {stripped:?}"
                );
            } else {
                separator_positions = Some(positions);
            }
        }
    }

    #[test]
    fn score_delta_triangle_markers_do_not_push_right_border_past_terminal_width() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        unsafe { std::env::set_var("COLUMNS", "80") };

        let header = vec![
            "#".to_string(),
            "query".to_string(),
            "fast->slow".to_string(),
            "增量".to_string(),
            "倍率".to_string(),
            "得分变化".to_string(),
        ];
        let align = vec![TableAlign::Left; header.len()];
        let rows = vec![
            vec![
                "1".to_string(),
                "对十二个客户对应多个销售的情况, 不一致的比例有多高".to_string(),
                "148.0→419.4".to_string(),
                "+271.4s".to_string(),
                "2.83x".to_string(),
                "4.0→4.0 △".to_string(),
            ],
            vec![
                "2".to_string(),
                "帮我计算下载后删除被速率影响均值".to_string(),
                "119.9→192.6".to_string(),
                "+72.8s".to_string(),
                "1.61x".to_string(),
                "4.0→4.0 ▲".to_string(),
            ],
        ];

        let widths = compute_table_widths("", &header, &rows);
        let mut rendered = String::new();
        rendered.push_str(&render_table_top("", &widths));
        rendered.push_str(&render_table_header("", &header, &align, &widths));
        rendered.push_str(&render_table_mid("", &widths));
        for row in &rows {
            rendered.push_str(&render_table_row("", row, &align, &widths));
        }
        rendered.push_str(&render_table_bottom("", &widths));

        for line in rendered.lines() {
            let visible = strip_ansi_codes(line);
            let width = terminal_display_width(visible.as_str());
            assert!(
                width <= 80,
                "triangle score-delta table line exceeds terminal width ({width}):\n{visible}"
            );
        }
    }
}
