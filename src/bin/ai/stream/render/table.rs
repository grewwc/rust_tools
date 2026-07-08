use crate::ai::stream::{
    extract::strip_ansi_codes,
    render::inline::{
        render_inline_md, terminal_cell_width, terminal_display_width, visible_width, wrap_md_cell,
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
    // 数的是"原样写入终端的预览行"占多少物理行，终端按 **真实** 列宽（raw_cols）
    // 自动折行，因此不能用保留右边距的 terminal_width，否则会高估行数、导致 cursor-up
    // 越界清屏。
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
    let mut math_delim = ""; // "$" or "$$"，记录进入数学段时的分隔符
    let mut in_strike = false;
    let mut escaped = false;

    /// 向前查找未转义的目标字符。用于判断分隔符是否有配对，
    /// 避免未闭合的 `、$ 把 in_code/in_math 卡死在 true，
    /// 导致后续 | 无法被识别为列分隔符。
    fn has_matching_delim(
        chars: &std::iter::Peekable<std::str::Chars>,
        target: char,
    ) -> bool {
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

    /// 向前查找未转义的双字符分隔符（用于 ~~ 和 $$）。
    fn has_matching_delim_pair(
        chars: &std::iter::Peekable<std::str::Chars>,
        first: char,
        second: char,
    ) -> bool {
        let mut la = chars.clone();
        // 跳过当前 peek 位置的字符——它是开分隔符的一部分（如 $$ 的第二个 $），
        // 不能把它和后面的字符凑成"闭合对"。例如 $$$ 中，开分隔符占前两个 $，
        // 如果不跳过，lookahead 会把第 2、3 个 $ 误判为闭合 $$，
        // 导致错误地进入数学模式而永远无法退出。
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

        // ~~strikethrough~~：跟踪状态，避免 ~~ 内的 | 被误识别为列分隔符
        if ch == '~' && !in_code && !in_math && chars.peek().copied() == Some('~') {
            if in_strike {
                // 已在删除线内：这是闭合 ~~
                chars.next();
                in_strike = false;
                current.push('~');
                current.push('~');
                continue;
            }
            // 不在删除线内：检查是否有配对的 ~~
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
                // 已在代码段内：这是闭合反引号
                in_code = false;
                current.push(ch);
                continue;
            }
            if !in_math && has_matching_delim(&chars, '`') {
                // 不在代码段内且能找到配对反引号：进入代码段
                in_code = true;
                current.push(ch);
                continue;
            }
        }

        if ch == '$' && !in_code {
            if in_math {
                // 已在数学段内：按进入时的分隔符类型决定是否闭合
                match math_delim {
                    "$$" if chars.peek().copied() == Some('$') => {
                        // $$ 段遇到 $$ 对：闭合
                        chars.next();
                        in_math = false;
                        math_delim = "";
                        current.push('$');
                        current.push('$');
                        continue;
                    }
                    "$" => {
                        // $ 段遇到单个 $：闭合（即使后面跟着 $）
                        in_math = false;
                        math_delim = "";
                        current.push(ch);
                        continue;
                    }
                    _ => {
                        // $$ 段内的单个 $（peek 不是 $），或未知状态：
                        // 按字面量处理，不闭合，fall through 到 current.push(ch)
                    }
                }
            }
            // 不在数学段内：检查配对
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

/// 先渲染 inline markdown，再按渲染后的实际显示宽度补空格。
///
/// 不能用 `pad_cell` + `render_inline_md(padded)` 的顺序：`pad_cell` 基于
/// `visible_width`（会剥离未闭合的 `、**、* 标记），但 `render_inline_md` 遇到
/// 未闭合标记会原样输出为字面字符，导致实际宽度 > 预期，表格边框错位。
fn render_and_pad_cell(cell_line: &str, width: usize, align: TableAlign, base: &str) -> String {
    let rendered = render_inline_md(cell_line, base);
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
    // 与 markdown.rs::preview_terminal_width 保持一致：保留 4 列右安全边距，
    // 避免表格 │ 边框紧贴终端右缘触发自动换行，破坏 box-drawing。
    const RIGHT_MARGIN: usize = 4;
    const MIN_WIDTH: usize = 20;

    let raw = raw_cols();
    raw.saturating_sub(RIGHT_MARGIN).max(MIN_WIDTH)
}

fn raw_cols() -> usize {
    // 与 markdown.rs::raw_terminal_cols 保持一致：实时 ioctl 优先，COLUMNS 仅作非 tty
    // 回退。常驻进程在面板里被拖窄后 COLUMNS 是过时快照，用它算表格宽度会超出真实
    // 面板、被终端硬折行，导致 │ 边框错位。
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
        // 三个反引号（奇数）不应把 in_code 卡死在 true，否则尾随 | 无法被识别为列分隔符。
        // 修复前：split_table_segments 每遇到一个 ` 就翻转 in_code，
        //   ``` 后 in_code=true，后续 | 被当字面量，整行只解析出 1 个 cell。
        // 修复后：只有找到配对反引号才进入 in_code，未配对的 ` 按字面量处理。
        assert_eq!(
            parse_table_row("| context 行漏前导空格 | ❌ invalid hunk line: ``` |"),
            vec!["context 行漏前导空格", "❌ invalid hunk line: ```"]
        );
        // 单个未闭合反引号也应按字面量处理
        assert_eq!(
            parse_table_row("| foo `bar | baz |"),
            vec!["foo `bar", "baz"]
        );
    }

    #[test]
    fn parse_table_row_handles_unpaired_dollar() {
        // 未配对的 $ 不应把 in_math 卡死在 true
        assert_eq!(
            parse_table_row("| foo $bar | baz |"),
            vec!["foo $bar", "baz"]
        );
        // 三个 $（奇数）也应正确处理
        assert_eq!(
            parse_table_row("| a $$$ b | c |"),
            vec!["a $$$ b", "c"]
        );
    }

    #[test]
    fn parse_table_row_dollar_inside_code_is_literal() {
        // $ 在反引号代码段内不触发数学模式
        assert_eq!(
            parse_table_row("| `a$b` | c |"),
            vec!["`a$b`", "c"]
        );
    }

    #[test]
    fn parse_table_row_single_dollar_math_closes_at_single_dollar() {
        // 用单 $ 进入的数学段，遇到 $$ 时应在第一个 $ 处闭合，
        // 第二个 $ 按字面量处理（可能开启新的数学段）。
        // 关键：$a$$b$ 中，$a$ 是一个数学段，$b$ 是另一个。
        let cells = parse_table_row("| $a$$b$ | c |");
        // 两个 cell：第一个含 "$a$$b$"（两个相邻数学段），第二个是 "c"
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[1], "c");
        // 第一个 cell 的内容应包含所有 $ 符号
        assert!(cells[0].contains("$$b$"));
    }

    #[test]
    fn parse_table_row_double_dollar_math_requires_double_dollar_to_close() {
        // 用 $$ 进入的数学段，单个 $ 不应触发闭合
        let cells = parse_table_row("| $$a$b$$ | c |");
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[1], "c");
        // $$a$b$$ 中，$ 是 $$ 段内的字面量，整段是一个数学 span
        assert_eq!(cells[0], "$$a$b$$");
    }

    #[test]
    fn parse_table_row_handles_unpaired_strikethrough() {
        // 未配对的 ~~ 应按字面量处理，不应把 in_strike 卡死
        assert_eq!(
            parse_table_row("| ~~a | b |"),
            vec!["~~a", "b"]
        );
        // 三个 ~~ 序列（奇数）也应正确处理
        assert_eq!(
            parse_table_row("| ~~a~~b~~ | c |"),
            vec!["~~a~~b~~", "c"]
        );
    }

    #[test]
    fn parse_table_row_escaped_delimiters() {
        // 转义的反引号不应触发代码段
        assert_eq!(
            parse_table_row(r#"| \`a\` | b |"#),
            vec![r#"\`a\`"#, "b"]
        );
        // 转义的 $ 不应触发数学段
        assert_eq!(
            parse_table_row(r#"| \$a\$ | b |"#),
            vec![r#"\$a\$"#, "b"]
        );
    }

    #[test]
    fn parse_table_row_pipe_inside_math_is_literal() {
        // $ 段内的 | 应被当作字面量
        assert_eq!(
            parse_table_row("| $a|b$ | c |"),
            vec!["$a|b$", "c"]
        );
    }

    #[test]
    fn parse_table_row_adjacent_code_spans() {
        // 相邻的反引号代码段：`a`b`c` 应被解析为 `a` + `c` 两个代码段
        // 中间的 b 是普通文本，| 不应被误吞
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
        // 多个 ~~ 段 + 普通文本混合
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

        // 宽度模型对 U+FE0F 一律 +1，把需要 VS16 才呈现成 emoji 的 base 撑到 2 列，
        // 与真实终端一致。🌧️ = U+1F327(unicode-width=1) + U+FE0F(+1) = 2 列，
        // "天气" 4 列，合计 6。旧代码把 VS16 当 0 列，🌧️ 只算 1 列，比终端窄 1 列，
        // 正是表格被硬折行、表头残留堆叠的根因。
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

        // 顶部边框应连成 box-drawing 线段（┌─...─┬─...─┐），确保观感为实线而非虚线。
        let top = render_table_top("", &widths);
        assert!(
            top.contains('┌') && top.contains('┬') && top.contains('┐'),
            "{top:?}"
        );
        assert!(top.contains("┌──────"), "{top:?}");
    }

    #[test]
    fn render_and_pad_cell_compensates_for_unclosed_marker_literal_output() {
        // 未闭合的 `、**、* 标记会被 render_inline_md 当字面字符输出（而非剥离），
        // 导致渲染后的实际显示宽度 > visible_width 预估值。
        // render_and_pad_cell 必须基于渲染后的实际宽度补空格。
        use crate::ai::stream::extract::strip_ansi_codes;

        fn rendered_display_width(s: &str) -> usize {
            let visible = strip_ansi_codes(s);
            terminal_display_width(visible.as_str())
        }

        // 目标宽度 10；未闭合反引号：visible_width 剥离 `，但 render_inline_md 原样输出
        let padded = render_and_pad_cell("`foo", 10, TableAlign::Left, "");
        let actual_w = rendered_display_width(&padded);
        assert_eq!(
            actual_w, 10,
            "unclosed backtick: padded width should be 10, got {actual_w}, padded={padded:?}"
        );

        // 未闭合 **：visible_width 剥离 **，但 render_inline_md 原样输出（+2 列）
        let padded = render_and_pad_cell("**foo", 10, TableAlign::Left, "");
        let actual_w = rendered_display_width(&padded);
        assert_eq!(
            actual_w, 10,
            "unclosed **: padded width should be 10, got {actual_w}, padded={padded:?}"
        );

        // 未闭合 *：visible_width 剥离 *，但 render_inline_md 原样输出（+1 列）
        let padded = render_and_pad_cell("*foo", 10, TableAlign::Left, "");
        let actual_w = rendered_display_width(&padded);
        assert_eq!(
            actual_w, 10,
            "unclosed *: padded width should be 10, got {actual_w}, padded={padded:?}"
        );

        // 闭合标记应该正常工作
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
}
