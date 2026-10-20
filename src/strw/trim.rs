//! # String Trimming Tools
//!
//! Provides removal of an arbitrary character set ([`trim_cutset`]) and of
//! whitespace ([`trim_all_whitespace`], [`collapse_whitespace`]).

use std::borrow::Cow;

use crate::commonw::types::FastSet;

/// 从字符串中移除所有属于指定字符集的字符
///
/// 与标准的 `trim` 方法不同，此函数会移除字符串中**所有**匹配的字符，
/// 而不仅仅是开头和结尾的字符。
///
/// # 参数
///
/// * `text` - 要处理的原始字符串
/// * `cutset` - 包含要移除字符的字符集
///
/// # 返回值
///
/// 返回 `Cow::Borrowed` 或 `Cow::Owned`：
/// - 如果 `cutset` 为空，返回 `Cow::Borrowed(text)`（零拷贝）
/// - 否则返回 `Cow::Owned` 包含处理后的新字符串
///
/// # 示例
///
/// ```rust
/// use rust_tools::strw::trim_cutset;
///
/// // 移除所有 'x' 字符
/// let result = trim_cutset("xxxhello worldxxx", "x");
/// assert_eq!(result, "hello world");
///
/// // 移除多个字符
/// let result = trim_cutset("abc123cab", "ac");
/// assert_eq!(result, "b123b");
///
/// // 支持 Unicode
/// let result = trim_cutset("你好 rust 世界好", "好世");
/// assert_eq!(result, "你 rust 界");
///
/// // 空字符集返回原字符串的借用
/// let result = trim_cutset("hello", "");
/// assert_eq!(result, "hello");
/// ```
///
/// # 性能特征
///
/// - 时间复杂度：O(n)，n 为字符串长度
/// - 空间复杂度：O(n)，需要创建新字符串
///
/// # 注意事项
///
/// - 此函数会移除字符串中**所有**匹配的字符，不仅仅是两端
/// - 如果需要只修剪两端，请使用标准库的 `trim_matches`
/// - 使用 `FastSet` 进行字符查找，提供 O(1) 平均查找时间
///
/// # 参见
///
/// - [`str::trim_matches`] - 标准库的修剪方法，只处理两端
pub fn trim_cutset<'a>(text: &'a str, cutset: &'a str) -> Cow<'a, str> {
    if cutset.is_empty() {
        return Cow::Borrowed(text);
    }

    // 使用 FastSet 存储要移除的字符集，提供 O(1) 查找
    let char_set = FastSet::from_iter(cutset.chars());

    // 过滤掉所有在字符集中的字符
    let x: Vec<char> = text.chars().filter(|ch| !char_set.contains(ch)).collect();

    Cow::Owned(x.into_iter().collect())
}

/// Remove every whitespace character from the text.
///
/// Unlike [`str::trim`], which only strips whitespace at both ends, this
/// removes whitespace anywhere in the text, including the middle.
///
/// Whitespace follows the Unicode `White_Space` property (spaces, tabs, line
/// breaks and the Unicode space separators). Zero-width characters such as
/// `U+200B` are not whitespace and are kept.
///
/// # Arguments
///
/// * `text` - The text to process
///
/// # Returns
///
/// Returns `Cow::Borrowed(text)` when the text holds no whitespace (zero-copy),
/// otherwise `Cow::Owned` with the whitespace removed.
///
/// # Examples
///
/// ```rust
/// use rust_tools::strw::trim_all_whitespace;
///
/// // Whitespace is removed everywhere, not just at the ends
/// let result = trim_all_whitespace("  a b\tc\n");
/// assert_eq!(result, "abc");
///
/// // Text without whitespace is returned unchanged
/// let result = trim_all_whitespace("abc");
/// assert_eq!(result, "abc");
/// ```
///
/// # Performance
///
/// - Time complexity: O(n), n being the text length
/// - Space complexity: O(n) when whitespace is present
pub fn trim_all_whitespace(text: &str) -> Cow<'_, str> {
    if !text.chars().any(char::is_whitespace) {
        return Cow::Borrowed(text);
    }

    Cow::Owned(text.chars().filter(|ch| !ch.is_whitespace()).collect())
}

/// Collapse every run of whitespace into a single space and drop the ends.
///
/// Whitespace at the very beginning and at the very end is removed entirely,
/// whitespace runs in the middle become one space each.
///
/// # Arguments
///
/// * `text` - The text to process
///
/// # Returns
///
/// Returns `Cow::Borrowed(text)` when the text holds no whitespace (zero-copy),
/// otherwise `Cow::Owned` with the collapsed text.
///
/// # Examples
///
/// ```rust
/// use rust_tools::strw::collapse_whitespace;
///
/// let result = collapse_whitespace("  hello \t\n  world  ");
/// assert_eq!(result, "hello world");
/// ```
///
/// # Performance
///
/// - Time complexity: O(n), n being the text length
/// - Space complexity: O(n) when whitespace is present
pub fn collapse_whitespace(text: &str) -> Cow<'_, str> {
    if !text.chars().any(char::is_whitespace) {
        return Cow::Borrowed(text);
    }

    let mut collapsed = String::with_capacity(text.len());
    let mut pending_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            // A separator is only needed between two non-whitespace characters.
            pending_space = !collapsed.is_empty();
            continue;
        }
        if pending_space {
            collapsed.push(' ');
            pending_space = false;
        }
        collapsed.push(ch);
    }

    Cow::Owned(collapsed)
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::{collapse_whitespace, trim_all_whitespace, trim_cutset};

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
        assert_eq!(trim_cutset("你好 rust 世界好", "好世"), "你 rust 界");
    }

    #[test]
    fn trim_set_returns_empty_when_everything_matches() {
        assert_eq!(trim_cutset("rust", "rust"), "");
    }

    #[test]
    fn trim_set_with_whitespace() {
        assert_eq!(trim_cutset("  hello  world  ", " "), "helloworld");
    }

    #[test]
    fn trim_set_with_special_chars() {
        assert_eq!(trim_cutset("a-b_c.d", "-_."), "abcd");
    }

    #[test]
    fn trim_all_whitespace_removes_whitespace_everywhere() {
        assert_eq!(trim_all_whitespace("  a b\tc\n"), "abc");
    }

    #[test]
    fn trim_all_whitespace_returns_borrowed_when_no_whitespace() {
        let result = trim_all_whitespace("hello");

        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result, "hello");
    }

    #[test]
    fn trim_all_whitespace_handles_unicode_spaces() {
        assert_eq!(trim_all_whitespace("a\u{3000}b\u{00a0}c"), "abc");
    }

    #[test]
    fn trim_all_whitespace_keeps_zero_width_characters() {
        assert_eq!(trim_all_whitespace("a\u{200b}b"), "a\u{200b}b");
    }

    #[test]
    fn trim_all_whitespace_returns_empty_for_blank_text() {
        assert_eq!(trim_all_whitespace(" \t\n "), "");
    }

    #[test]
    fn collapse_whitespace_collapses_runs_and_trims_ends() {
        assert_eq!(collapse_whitespace("  hello \t\n  world  "), "hello world");
    }

    #[test]
    fn collapse_whitespace_returns_borrowed_when_no_whitespace() {
        let result = collapse_whitespace("hello");

        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result, "hello");
    }

    #[test]
    fn collapse_whitespace_keeps_single_spaces() {
        assert_eq!(collapse_whitespace("a b c"), "a b c");
    }

    #[test]
    fn collapse_whitespace_returns_empty_for_blank_text() {
        assert_eq!(collapse_whitespace(" \t\n "), "");
    }
}
