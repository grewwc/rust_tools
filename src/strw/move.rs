pub fn move_to_end_all(original: &str, target: &str) -> String {
    if target.is_empty() {
        return original.to_string();
    }
    let n = original.matches(target).count();
    if n == 0 {
        return original.to_string();
    }
    let new_string = original.replace(target, "");
    format!("{new_string}{}", target.repeat(n))
}

pub fn reverse(s: &str) -> String {
    s.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_to_end_all() {
        assert_eq!(move_to_end_all("a--b--c", "--"), "abc----");
        assert_eq!(move_to_end_all("abc", "--"), "abc");
        assert_eq!(move_to_end_all("abc", ""), "abc");
    }

    #[test]
    fn test_reverse() {
        assert_eq!(reverse("abc"), "cba");
        assert_eq!(reverse("你好rust"), "tsur好你");
    }
}
