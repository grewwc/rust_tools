//! 字符串修剪工具
//!
//! 提供去除字符串中指定字符集合的功能。

use std::borrow::Cow;

use crate::common::types::FastSet;

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
/// use rust_tools::strw::trim::trim_cutset;
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
}
