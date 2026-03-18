use crate::strw::search::kmp_search;

pub fn find_all(s: &str, substr: &str) -> Vec<usize> {
    kmp_search(s, substr, -1)
}

pub fn strip_prefix(s: &str, prefix: &str) -> String {
    if prefix.is_empty() {
        return s.to_string();
    }
    match s.find(prefix) {
        Some(idx) => String::from_utf8_lossy(&s.as_bytes()[idx + prefix.len()..]).to_string(),
        None => s.to_string(),
    }
}

pub fn strip_suffix(s: &str, suffix: &str) -> String {
    if suffix.is_empty() {
        return s.to_string();
    }
    match s.rfind(suffix) {
        Some(idx) => String::from_utf8_lossy(&s.as_bytes()[..idx]).to_string(),
        None => s.to_string(),
    }
}

pub fn substring_quiet(s: &str, beg: isize, end: isize) -> String {
    let len = s.len() as isize;
    if beg >= len || beg >= end {
        return String::new();
    }
    let b = beg.max(0) as usize;
    let e = end.min(len).max(0) as usize;
    if b >= e {
        return String::new();
    }
    match s.get(b..e) {
        Some(x) => x.to_string(),
        None => {
            let start = clamp_to_char_boundary(s, b, false);
            let stop = clamp_to_char_boundary(s, e, true);
            if start >= stop {
                String::new()
            } else {
                s.get(start..stop).unwrap_or("").to_string()
            }
        }
    }
}

fn clamp_to_char_boundary(s: &str, idx: usize, up: bool) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    if s.is_char_boundary(idx) {
        return idx;
    }
    if up {
        let mut i = idx + 1;
        while i < s.len() && !s.is_char_boundary(i) {
            i += 1;
        }
        i.min(s.len())
    } else {
        let mut i = idx.saturating_sub(1);
        while i > 0 && !s.is_char_boundary(i) {
            i = i.saturating_sub(1);
        }
        if s.is_char_boundary(i) { i } else { 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_prefix_suffix() {
        assert_eq!(strip_prefix("xxabczz", "abc"), "zz");
        assert_eq!(strip_prefix("abczz", "abc"), "zz");
        assert_eq!(strip_prefix("abczz", "x"), "abczz");
        assert_eq!(strip_suffix("xxabczz", "abc"), "xx");
        assert_eq!(strip_suffix("xxabczz", "zz"), "xxabc");
        assert_eq!(strip_suffix("xxabczz", "x"), "x");
    }

    #[test]
    fn test_substring_quiet() {
        assert_eq!(substring_quiet("abcdef", 1, 4), "bcd");
        assert_eq!(substring_quiet("abcdef", -2, 2), "ab");
        assert_eq!(substring_quiet("abcdef", 10, 12), "");
        assert_eq!(substring_quiet("abcdef", 4, 4), "");
    }
}
