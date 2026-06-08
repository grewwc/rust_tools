use colored::Colorize;
use std::path::Path;

/// Snap a byte index down to the nearest char boundary.
fn snap_down(s: &str, idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    if s.is_char_boundary(idx) {
        return idx;
    }
    (0..idx).rev().find(|&i| s.is_char_boundary(i)).unwrap_or(0)
}

/// Snap a byte index up to the nearest char boundary.
fn snap_up(s: &str, idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    if s.is_char_boundary(idx) {
        return idx;
    }
    (idx..=s.len()).find(|&i| s.is_char_boundary(i)).unwrap_or(s.len())
}

pub fn highlight_ranges(s: &str, mut ranges: Vec<(usize, usize)>) -> String {
    if ranges.is_empty() {
        return s.to_string();
    }
    ranges.sort_by_key(|(s, _)| *s);
    let mut out = String::new();
    let mut cursor = 0usize;
    for (start, end) in ranges {
        if start >= end || start >= s.len() {
            continue;
        }
        let end = end.min(s.len());
        // Snap range endpoints to char boundaries
        let start = snap_up(s, start);
        let end = snap_down(s, end);
        if start >= end {
            continue;
        }
        if cursor < start {
            out.push_str(&s[cursor..start]);
        }
        if let Some(seg) = s.get(start..end) {
            out.push_str(&seg.red().to_string());
        }
        cursor = end;
    }
    if cursor < s.len() {
        let tail_start = snap_up(s, cursor);
        if tail_start < s.len() {
            out.push_str(&s[tail_start..]);
        }
    }
    out
}

pub fn print_hit(abs: &Path, lineno: usize, preview: &str) {
    let abs_str = abs.to_string_lossy().replace('\\', "/");
    let (dir, base) = match abs_str.rsplit_once('/') {
        Some((d, b)) => (d.to_string(), b.to_string()),
        None => ("".to_string(), abs_str),
    };
    let sep = "/";
    println!(
        "{} \"{}{}{}\" [{}]:  {}\n",
        ">>".green(),
        dir,
        sep,
        base.yellow(),
        lineno,
        preview
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highlight_ranges() {
        let s = "abcXYZdefXYZ";
        let out = highlight_ranges(s, vec![(3, 6), (9, 12)]);
        assert!(out.contains("abc"));
        assert!(out.contains("def"));
    }

    #[test]
    fn test_highlight_ranges_utf8_boundary() {
        // "创建消息" — each CJK char is 3 bytes. "创" is bytes 0..3, "建" is 3..6, etc.
        let s = "创建消息到 aeolus_ada_messages, 返回 message_id";
        // Range ending mid-char (byte 42 is inside '返' at bytes 41..44)
        let out = highlight_ranges(s, vec![(0, 42)]);
        assert!(!out.is_empty());
    }
}
