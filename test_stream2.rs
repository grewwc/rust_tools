#[derive(Clone, Copy)]
enum TableAlign { Left, Center, Right }

enum TableState {
    None,
    PendingHeader { indent: String, header_line: String, preview_height: usize },
    InTable { indent: String, header: Vec<String>, align: Vec<TableAlign>, rows: Vec<Vec<String>>, preview_height: usize },
}

struct MarkdownStreamRenderer {
    table_state: TableState,
    in_code_block: bool,
    bol: bool,
}

fn split_indent(s: &str) -> (&str, &str) {
    let mut idx = 0usize;
    for (i, ch) in s.char_indices() {
        if ch == ' ' || ch == '\t' { idx = i + ch.len_utf8(); continue; }
        idx = i; break;
    }
    if s.chars().all(|c| c == ' ' || c == '\t') { return (s, ""); }
    s.split_at(idx)
}

fn split_table_segments(s: &str) -> Vec<String> {
    let mut segments = Vec::new(); let mut current = String::new();
    let mut chars = s.chars().peekable();
    let mut in_code = false; let mut in_math = false; let mut escaped = false;
    while let Some(ch) = chars.next() {
        if escaped { current.push(ch); escaped = false; continue; }
        if ch == '\\' { current.push(ch); escaped = true; continue; }
        if ch == '`' && !in_math { in_code = !in_code; current.push(ch); continue; }
        if ch == '$' && !in_code {
            if chars.peek().copied() == Some('$') {
                chars.next(); in_math = !in_math; current.push('$'); current.push('$'); continue;
            }
            in_math = !in_math; current.push(ch); continue;
        }
        if ch == '|' && !in_code && !in_math { segments.push(std::mem::take(&mut current)); continue; }
        current.push(ch);
    }
    segments.push(current); segments
}

fn parse_table_row(line: &str) -> Vec<String> {
    let (_, rest) = split_indent(line); let s = rest.trim();
    let mut raw = split_table_segments(s);
    if s.starts_with('|') && !raw.is_empty() && raw.first().is_some_and(|x| x.is_empty()) { raw.remove(0); }
    if s.ends_with('|') && !raw.is_empty() && raw.last().is_some_and(|x| x.is_empty()) { raw.pop(); }
    raw.into_iter().map(|x| x.trim().to_string()).collect()
}

fn is_table_row_candidate(line: &str) -> bool {
    if line.trim().is_empty() { return false; }
    let (_, rest) = split_indent(line); let s = rest.trim_end();
    if !s.contains('|') { return false; }
    if s.starts_with("```") || s.starts_with("~~~") { return false; }
    let cells = parse_table_row(s); cells.len() >= 2
}

fn is_table_separator(line: &str) -> bool {
    let (_, rest) = split_indent(line); let mut s = rest.trim();
    if s.starts_with('|') { s = &s[1..]; }
    if s.ends_with('|') && !s.is_empty() { s = &s[..s.len() - 1]; }
    let parts = split_table_segments(s).into_iter().map(|p| p.trim().to_string()).filter(|p| !p.is_empty());
    let mut count = 0usize;
    for p in parts {
        count += 1; let p = p.trim_matches(' '); let core = p.trim_matches(':');
        if core.is_empty() || !core.chars().all(|c| c == '-') { return false; }
    }
    count >= 2
}

fn is_table_row(line: &str) -> bool {
    let (_, rest) = split_indent(line);
    let s = rest.trim_end();
    if s.trim().is_empty() { return false; }
    if is_table_separator(s) { return false; }
    let cells = parse_table_row(s);
    cells.len() >= 2
}

fn parse_table_align(line: &str, cols: usize) -> Vec<TableAlign> {
    let (_, rest) = split_indent(line); let s = rest.trim();
    let mut raw = split_table_segments(s);
    if s.starts_with('|') && !raw.is_empty() && raw.first().is_some_and(|x| x.is_empty()) { raw.remove(0); }
    if s.ends_with('|') && !raw.is_empty() && raw.last().is_some_and(|x| x.is_empty()) { raw.pop(); }
    let mut out = Vec::with_capacity(cols);
    for seg in raw.iter().map(|s| s.as_str()).chain(std::iter::repeat("")).take(cols) {
        let seg = seg.trim(); let left = seg.starts_with(':'); let right = seg.ends_with(':');
        out.push(match (left, right) { (true, true) => TableAlign::Center, (false, true) => TableAlign::Right, _ => TableAlign::Left });
    }
    out
}

impl MarkdownStreamRenderer {
    fn consume_line(&mut self, line: &str) -> String {
        let state = std::mem::replace(&mut self.table_state, TableState::None);
        match state {
            TableState::None => {
                if !self.in_code_block && is_table_row_candidate(line) && !is_table_separator(line) {
                    let (indent, rest) = split_indent(line);
                    self.table_state = TableState::PendingHeader {
                        indent: indent.to_string(),
                        header_line: rest.trim_end().to_string(),
                        preview_height: 1,
                    };
                    return format!("-> [None -> PendingHeader] {:?}", line);
                }
                return format!("-> [None -> None] {:?}", line);
            }
            TableState::PendingHeader { indent, header_line, preview_height } => {
                if is_table_separator(line) {
                    let header_cells = parse_table_row(&header_line);
                    let align = parse_table_align(line, header_cells.len());
                    self.table_state = TableState::InTable { indent, header: header_cells, align, rows: Vec::new(), preview_height: preview_height + 1 };
                    return format!("-> [PendingHeader -> InTable] {:?}", line);
                }
                self.table_state = TableState::None;
                return format!("-> [PendingHeader -> None] Restored header and consumed line {:?}", line);
            }
            TableState::InTable { indent, header, align, mut rows, preview_height } => {
                if is_table_row(line) {
                    rows.push(parse_table_row(line));
                    self.table_state = TableState::InTable { indent, header, align, rows, preview_height: preview_height + 1 };
                    return format!("-> [InTable -> InTable] {:?}", line);
                }
                self.table_state = TableState::None;
                return format!("-> [InTable -> None] Broken out of table {:?}", line);
            }
        }
    }
}

fn main() {
    let mut r = MarkdownStreamRenderer { table_state: TableState::None, in_code_block: false, bol: true };
    let lines = [
        " ## 🎯 问题分析 ",
        " ",
        " ### 知识类型对比 ",
        " ",
        " | 知识类型 | 示例 | 变化频率 | 验证方式 | ",
        " |---------|------|---------|---------| ",
        " | **文件/代码** | 项目结构、函数实现 | 低（手动修改） | 文件指纹 | ",
        " | **时间敏感** | 今天天气、当前时间 | 高（自然变化） | 时间验证 | "
    ];
    for l in lines {
        println!("{}", r.consume_line(l));
    }
}
