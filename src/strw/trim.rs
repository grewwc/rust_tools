use std::borrow::Cow;

use crate::common::types::FastSet;

pub fn trim_cutset<'a>(text: &'a str, cutset: &'a str) -> Cow<'a, str> {
    if cutset.is_empty() {
        return Cow::Borrowed(text);
    }

    let char_set = FastSet::from_iter(cutset.chars());

    let x: Vec<char> = text.chars().filter(|ch| !char_set.contains(ch)).collect();

    Cow::Owned(x.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::trim_cutset;

    #[test]
    fn trim_set_returns_borrowed_when_set_is_empty() {
        let result = trim_cutset("hello", "");

        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result, "hello");
    }

    #[test]
    fn trim_set_removes_matching_characters() {
        assert_eq!(trim_cutset("abc123cab", "ac"), "b123b");
    }

    #[test]
    fn trim_set_preserves_order_and_duplicates_of_remaining_characters() {
        assert_eq!(trim_cutset("balloon", "bn"), "alloo");
    }

    #[test]
    fn trim_set_returns_original_when_nothing_matches() {
        assert_eq!(trim_cutset("rust", "xyz"), "rust");
    }

    #[test]
    fn trim_set_supports_unicode_characters() {
        assert_eq!(trim_cutset("你好rust世界好", "好世"), "你rust界");
    }

    #[test]
    fn trim_set_returns_empty_when_everything_matches() {
        assert_eq!(trim_cutset("rust", "rust"), "");
    }
}
