pub fn slice_contains(slice: &[String], target: &str) -> bool {
    slice.iter().any(|s| s == target)
}

pub fn contains(s: &str, sub: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    if sub.is_empty() {
        return true;
    }
    !kmp_search(s, sub, 1).is_empty()
}

pub fn kmp_search(text: &str, pattern: &str, n: isize) -> Vec<usize> {
    if text.is_empty() {
        return Vec::new();
    }
    if pattern.is_empty() {
        return vec![0];
    }
    kmp_search_bytes(text.as_bytes(), pattern.as_bytes(), n)
}

pub fn kmp_search_bytes(text: &[u8], pattern: &[u8], n: isize) -> Vec<usize> {
    if text.is_empty() {
        return Vec::new();
    }
    if pattern.is_empty() {
        return vec![0];
    }
    let next = kmp_next(pattern);
    let mut matches: Vec<usize> = Vec::with_capacity(2);
    let mut i = 0usize;
    let mut j = 0usize;
    while i < text.len() {
        if text[i] == pattern[j] {
            i += 1;
            j += 1;
        } else if j == 0 {
            i += 1;
        } else {
            j = next[j - 1];
        }
        if j == pattern.len() {
            matches.push(i - j);
            j = next[j - 1];
            if n > 0 && matches.len() >= n as usize {
                break;
            }
        }
    }
    matches
}

pub fn copy_slice<T: Clone>(original: &[T]) -> Vec<T> {
    original.to_vec()
}

pub fn any_equals(target: &str, choices: &[&str]) -> bool {
    choices.contains(&target)
}

pub fn any_contains(s: &str, search_strings: &[&str]) -> bool {
    search_strings.iter().any(|search| contains(s, search))
}

pub fn find_first_substr(s: &str, skip_match: bool, search_strings: &[&str]) -> isize {
    for search in search_strings {
        if let Some(idx) = s.find(search)
            && idx > 0
        {
            let mut i = idx;
            if skip_match {
                i += search.len();
            }
            return i as isize;
        }
    }
    -1
}

pub fn all_contains(s: &str, search_strings: &[&str]) -> bool {
    search_strings.iter().all(|search| contains(s, search))
}

pub fn any_has_prefix(s: &str, subs: &[&str]) -> bool {
    subs.iter().any(|sub| s.starts_with(sub))
}

pub fn trim_after(s: &str, pattern: &str) -> String {
    let indices = kmp_search_bytes(s.as_bytes(), pattern.as_bytes(), 1);
    if indices.is_empty() {
        return s.to_string();
    }
    let idx = indices[0];
    String::from_utf8_lossy(&s.as_bytes()[..idx]).to_string()
}

pub fn trim_before(s: &str, pattern: &str) -> String {
    let indices = kmp_search_bytes(s.as_bytes(), pattern.as_bytes(), 1);
    if indices.is_empty() {
        return s.to_string();
    }
    let start = indices[0].saturating_add(pattern.len()).saturating_add(1);
    if start >= s.len() {
        return String::new();
    }
    String::from_utf8_lossy(&s.as_bytes()[start..]).to_string()
}

fn kmp_next(pattern: &[u8]) -> Vec<usize> {
    let mut next = vec![0usize; pattern.len()];
    let mut prefix_len = 0usize;
    let mut i = 1usize;
    while i < pattern.len() {
        if pattern[i] == pattern[prefix_len] {
            prefix_len += 1;
            next[i] = prefix_len;
            i += 1;
        } else if prefix_len == 0 {
            next[i] = 0;
            i += 1;
        } else {
            prefix_len = next[prefix_len - 1];
        }
    }
    next
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmp_search_overlap() {
        let matches = kmp_search("abababa", "aba", -1);
        assert_eq!(matches, vec![0, 2, 4]);
    }

    #[test]
    fn test_contains_empty_sub() {
        assert!(contains("abc", ""));
        assert!(!contains("", "a"));
    }

    #[test]
    fn test_trim_after_before() {
        assert_eq!(trim_after("a=b=c", "="), "a");
        assert_eq!(trim_before("a=b=c", "="), "=c");
        assert_eq!(trim_after("abc", "x"), "abc");
        assert_eq!(trim_before("abc", "x"), "abc");
    }
}
