use colored::Colorize;
use std::path::Path;

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
        if cursor < start {
            out.push_str(&s[cursor..start]);
        }
        if let Some(seg) = s.get(start..end) {
            out.push_str(&seg.red().to_string());
        }
        cursor = end;
    }
    if cursor < s.len() {
        out.push_str(&s[cursor..]);
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
}
