//! 纯文本截断与摘要工具函数。

pub(super) fn truncate_to_chars(s: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let end = s
        .char_indices()
        .nth(max_chars)
        .map(|(idx, _)| idx)
        .unwrap_or_else(|| s.len());
    let mut out = s[..end].to_string();
    out.push('\u{2026}');
    out
}

pub(super) fn summarize_text(text: &str, target_chars: usize) -> String {
    if target_chars == 0 {
        return String::new();
    }
    let char_count = text.chars().count();
    if char_count <= target_chars {
        return text.to_string();
    }

    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();

    if lines.len() <= 1 {
        return keep_ends_by_chars(text, target_chars);
    }

    let mut selected: Vec<&str> = Vec::new();
    let mut selected_chars = 0usize;

    let head_count = (lines.len().min(3)).min(target_chars / 20);
    for line in lines.iter().take(head_count) {
        if selected_chars + line.chars().count() + 1 > target_chars {
            break;
        }
        selected.push(line);
        selected_chars += line.chars().count() + 1;
    }

    let tail_budget = target_chars
        .saturating_sub(selected_chars)
        .max(target_chars / 3);
    let tail_count = lines.len().min(3).min(tail_budget / 20);
    let tail_start = lines.len().saturating_sub(tail_count);
    if tail_start > head_count {
        for line in lines.iter().skip(tail_start) {
            if selected_chars + line.chars().count() + 1 > target_chars {
                break;
            }
            selected.push(line);
            selected_chars += line.chars().count() + 1;
        }
    }

    if selected.is_empty() {
        return keep_ends_by_chars(text, target_chars);
    }

    let result = selected.join("\n");
    if result.chars().count() <= target_chars {
        return result;
    }

    keep_ends_by_chars(&result, target_chars)
}

pub(super) fn keep_ends_by_chars(text: &str, target_chars: usize) -> String {
    // 与 truncate_to_chars / summarize_text 一致：预算为 0 时无处放 head/tail，直接返回空串，
    // 否则下方 `target_chars - head_budget - 1` 会在 target_chars==0 时下溢（debug 崩溃、
    // release 回绕成 usize::MAX 导致 take/skip 反而放大文本）。
    if target_chars == 0 {
        return String::new();
    }
    let char_count = text.chars().count();
    if char_count <= target_chars {
        return text.to_string();
    }
    let head_budget = target_chars * 3 / 5;
    let tail_budget = target_chars - head_budget - 1;
    let head: String = text.chars().take(head_budget).collect();
    let tail: String = text
        .chars()
        .skip(char_count.saturating_sub(tail_budget))
        .collect();
    format!("{}\u{2026}{}", head, tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_ends_by_chars_zero_budget_returns_empty_without_overflow() {
        // 回归：prefix 撑爆 target 时预算会算成 0，旧实现在 `target - head - 1` 处下溢
        // （debug 崩溃 / release 回绕成 usize::MAX 反而放大文本）。0 预算必须安全返回空串。
        assert_eq!(keep_ends_by_chars("non-empty text", 0), "");
        assert_eq!(keep_ends_by_chars("", 0), "");
    }

    #[test]
    fn keep_ends_by_chars_small_budget_stays_within_target() {
        // 预算 >=1 时不得下溢，且结果字符数不超过 target。
        for target in 1..=8 {
            let out = keep_ends_by_chars("abcdefghijklmnopqrstuvwxyz", target);
            assert!(
                out.chars().count() <= target,
                "target={target} produced {out:?}"
            );
        }
    }
}
