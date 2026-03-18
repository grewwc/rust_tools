pub fn is_blank(s: &str) -> bool {
    if s.is_empty() {
        return true;
    }
    s.chars().all(|ch| ch.is_whitespace())
}

pub fn all_blank(strs: &[&str]) -> bool {
    if strs.is_empty() {
        return false;
    }
    strs.iter().all(|s| is_blank(s))
}

pub fn any_blank(strs: &[&str]) -> bool {
    strs.iter().any(|s| is_blank(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_blank() {
        assert!(is_blank(""));
        assert!(is_blank(" "));
        assert!(is_blank("\t\r\n"));
        assert!(!is_blank("a"));
        assert!(!is_blank(" a "));
    }

    #[test]
    fn test_all_blank() {
        assert!(!all_blank(&[]));
        assert!(all_blank(&["", " ", "\n"]));
        assert!(!all_blank(&["", "x"]));
    }

    #[test]
    fn test_any_blank() {
        assert!(!any_blank(&[]));
        assert!(any_blank(&["x", " "]));
        assert!(!any_blank(&["x", "y"]));
    }
}
