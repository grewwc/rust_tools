fn main() {
    let line1 = " | 知识类型 | 示例 | 变化频率 | 验证方式 | ";
    println!("Candidate 1: {}", is_table_row_candidate(line1));
    println!("Separator 1: {}", is_table_separator(line1));
}

fn split_indent(s: &str) -> (&str, &str) {
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

fn split_table_segments(s: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut chars = s.chars().peekable();
    let mut in_code = false;
    let mut in_math = false;
    let mut escaped = false;

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
        if ch == '|' && !in_code && !in_math {
            segments.push(std::mem::take(&mut current)); continue;
        }
        current.push(ch);
    }
    segments.push(current);
    segments
}

fn parse_table_row(line: &str) -> Vec<String> {
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

fn is_table_row_candidate(line: &str) -> bool {
    if line.trim().is_empty() { return false; }
    let (_, rest) = split_indent(line);
    let s = rest.trim_end();
    if !s.contains('|') { return false; }
    if s.starts_with("```") || s.starts_with("~~~") { return false; }
    let cells = parse_table_row(s);
    cells.len() >= 2
}

fn is_table_separator(line: &str) -> bool {
    let (_, rest) = split_indent(line);
    let mut s = rest.trim();
    if s.starts_with('|') { s = &s[1..]; }
    if s.ends_with('|') && !s.is_empty() { s = &s[..s.len() - 1]; }
    let parts = split_table_segments(s)
        .into_iter()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty());
    let mut count = 0usize;
    for p in parts {
        count += 1;
        let p = p.trim_matches(' ');
        let core = p.trim_matches(':');
        if core.len() < 3 || !core.chars().all(|c| c == '-') {
            return false;
        }
    }
    count >= 2
}
