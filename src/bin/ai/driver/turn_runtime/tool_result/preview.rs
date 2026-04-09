const TOOL_TERMINAL_PREVIEW_MAX_CHARS: usize = 4_000;
const TOOL_TERMINAL_PREVIEW_MAX_LINES: usize = 80;
const TOOL_TERMINAL_PREVIEW_HEAD_LINES: usize = 40;
const TOOL_TERMINAL_PREVIEW_TAIL_LINES: usize = 20;
const READ_FILE_TERMINAL_PREVIEW_MAX_CHARS: usize = 2_200;
const READ_FILE_TERMINAL_PREVIEW_MAX_LINES: usize = 40;
const READ_FILE_TERMINAL_PREVIEW_HEAD_LINES: usize = 24;
const READ_FILE_TERMINAL_PREVIEW_TAIL_LINES: usize = 8;
const WEB_SEARCH_TERMINAL_PREVIEW_MAX_CHARS: usize = 1_800;
const WEB_SEARCH_TERMINAL_PREVIEW_MAX_LINES: usize = 18;

struct ToolTerminalPreviewPolicy {
    max_chars: usize,
    max_lines: usize,
    head_lines: usize,
    tail_lines: usize,
    summary_first: bool,
}

fn terminal_preview_policy(tool_name: &str) -> ToolTerminalPreviewPolicy {
    match tool_name {
        "read_file_lines" | "read_file" => ToolTerminalPreviewPolicy {
            max_chars: READ_FILE_TERMINAL_PREVIEW_MAX_CHARS,
            max_lines: READ_FILE_TERMINAL_PREVIEW_MAX_LINES,
            head_lines: READ_FILE_TERMINAL_PREVIEW_HEAD_LINES,
            tail_lines: READ_FILE_TERMINAL_PREVIEW_TAIL_LINES,
            summary_first: false,
        },
        "web_search" => ToolTerminalPreviewPolicy {
            max_chars: WEB_SEARCH_TERMINAL_PREVIEW_MAX_CHARS,
            max_lines: WEB_SEARCH_TERMINAL_PREVIEW_MAX_LINES,
            head_lines: 12,
            tail_lines: 0,
            summary_first: true,
        },
        _ => ToolTerminalPreviewPolicy {
            max_chars: TOOL_TERMINAL_PREVIEW_MAX_CHARS,
            max_lines: TOOL_TERMINAL_PREVIEW_MAX_LINES,
            head_lines: TOOL_TERMINAL_PREVIEW_HEAD_LINES,
            tail_lines: TOOL_TERMINAL_PREVIEW_TAIL_LINES,
            summary_first: false,
        },
    }
}

pub(super) fn truncate_chars(content: &str, max_chars: usize) -> String {
    if content.chars().count() <= max_chars {
        return content.to_string();
    }
    let mut out = String::with_capacity(max_chars + 32);
    for (i, ch) in content.chars().enumerate() {
        if i >= max_chars {
            break;
        }
        out.push(ch);
    }
    out.push('…');
    out
}

pub(super) fn tail_chars(content: &str, max_chars: usize) -> String {
    let total = content.chars().count();
    if total <= max_chars {
        return content.to_string();
    }
    let skip = total.saturating_sub(max_chars);
    let mut out = String::with_capacity(max_chars + 32);
    out.push('…');
    for (i, ch) in content.chars().enumerate() {
        if i < skip {
            continue;
        }
        out.push(ch);
    }
    out
}

pub(super) fn build_terminal_preview(tool_name: &str, content: &str) -> String {
    let policy = terminal_preview_policy(tool_name);
    let line_count = content.lines().count();
    let char_count = content.chars().count();
    if line_count <= policy.max_lines && char_count <= policy.max_chars {
        return content.to_string();
    }

    if policy.summary_first {
        let mut preview = String::new();
        let mut kept = 0usize;
        for line in content.lines().map(str::trim).filter(|line| !line.is_empty()) {
            preview.push_str(line);
            preview.push('\n');
            kept += 1;
            if kept >= policy.head_lines || preview.chars().count() >= policy.max_chars {
                break;
            }
        }
        if preview.is_empty() {
            preview = truncate_chars(content, policy.max_chars);
        }
        return format!(
            "{}\n... (summary-first terminal preview; {} lines, {} chars total)",
            preview.trim_end(),
            line_count,
            char_count
        );
    }

    if line_count <= 1 {
        let head_budget = policy.max_chars / 2;
        let tail_budget = policy.max_chars.saturating_sub(head_budget);
        let head = truncate_chars(content, head_budget);
        let tail = tail_chars(content, tail_budget);
        return format!(
            "{head}\n... (truncated for terminal preview; {} chars total)\n{tail}",
            char_count
        );
    }

    let lines = content.lines().collect::<Vec<_>>();
    let omitted = line_count.saturating_sub(policy.head_lines + policy.tail_lines);
    let mut head = String::new();
    for line in lines.iter().take(policy.head_lines) {
        head.push_str(line);
        head.push('\n');
    }
    let mut notice = format!(
        "... (truncated for terminal preview; {} lines, {} chars total",
        line_count, char_count
    );
    if omitted > 0 {
        notice.push_str(&format!(", {} lines omitted", omitted));
    }
    notice.push_str(")\n");
    let mut tail = String::new();
    if policy.tail_lines > 0 && line_count > policy.head_lines {
        let tail_start = line_count.saturating_sub(policy.tail_lines);
        for line in lines.iter().skip(tail_start) {
            tail.push_str(line);
            tail.push('\n');
        }
    }

    let mut preview = format!("{head}{notice}{tail}");
    if preview.chars().count() > policy.max_chars {
        let notice_len = notice.chars().count();
        let available = policy.max_chars.saturating_sub(notice_len);
        if !tail.is_empty() && available > 32 {
            let tail_budget = available.min(available / 3).max(available / 4).max(96);
            let tail_budget = tail_budget.min(available.saturating_sub(32));
            let head_budget = available.saturating_sub(tail_budget);
            let head = truncate_chars(head.trim_end(), head_budget);
            let tail = tail_chars(tail.trim_end(), tail_budget);
            preview = format!("{head}\n{notice}{tail}");
        } else {
            let truncated = truncate_chars(&preview, policy.max_chars);
            return format!("{truncated}\n... (truncated for terminal preview)");
        }
    }

    preview
}
