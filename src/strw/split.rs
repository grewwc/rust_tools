use std::{collections::HashSet, hash::BuildHasherDefault};

use rustc_hash::FxHasher;

use crate::common::types::FastSet;

const COMMON_WHITESPACES: &str = " \t\n\r\x0B\x0C";

/// Function that returns an iterator similar to std::str::Split
pub fn split_keep_symbol<'a>(
    s: &'a str,
    split_chars: &'a str,
    symbol_set: &'a str,
) -> QuoteSplit<'a> {
    QuoteSplit::new(s, split_chars, symbol_set)
}

pub fn split_space_keep_symbol<'a>(s: &'a str, symbol_set: &'a str) -> QuoteSplit<'a> {
    split_keep_symbol(s, COMMON_WHITESPACES, symbol_set)
}

/// Alternative implementation using a custom iterator that behaves more like Split
pub struct QuoteSplit<'a> {
    text: &'a str,
    position: usize,
    symbols: FastSet<char>,
    split_chars: FastSet<char>,
}

impl<'a> QuoteSplit<'a> {
    pub fn new(text: &'a str, split_chars: &'a str, symbols: &'a str) -> Self {
        let mut symbol_set: HashSet<char, BuildHasherDefault<FxHasher>> =
            HashSet::with_capacity_and_hasher(
                symbols.len(),
                BuildHasherDefault::<FxHasher>::default(),
            );

        symbols.chars().for_each(|ch| {
            symbol_set.insert(ch);
        });

        let mut split_chars_set = HashSet::with_capacity_and_hasher(
            split_chars.len(),
            BuildHasherDefault::<FxHasher>::default(),
        );
        split_chars.chars().for_each(|ch| {
            split_chars_set.insert(ch);
        });

        QuoteSplit {
            text,
            position: 0,
            split_chars: split_chars_set,
            symbols: symbol_set,
        }
    }
}

impl<'a> QuoteSplit<'a> {
    fn is_whitespace(&self, ch: char) -> bool {
        self.split_chars.contains(&ch)
    }
}

impl<'a> Iterator for QuoteSplit<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.text.len() {
            return None;
        }

        // Skip leading whitespace
        let mut char_indices = self.text[self.position..].char_indices();
        while let Some((_, ch)) = char_indices.next() {
            if self.is_whitespace(ch) {
                self.position += ch.len_utf8();
            } else {
                break;
            }
        }

        if self.position >= self.text.len() {
            return None;
        }

        let start = self.position;
        let mut in_quotes = false;
        let mut char_indices = self.text[self.position..].char_indices();

        while let Some((_, ch)) = char_indices.next() {
            if self.split_chars.is_empty() {
                self.position += ch.len_utf8();
                return Some(&self.text[start..self.position]);
            }

            if self.symbols.contains(&ch) {
                in_quotes = !in_quotes;
                self.position += ch.len_utf8();
            } else if !in_quotes && self.split_chars.contains(&ch) {
                break;
            } else {
                self.position += ch.len_utf8();
            }
        }

        Some(&self.text[start..self.position])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_keep_quotes_simple() {
        let input = r#"hello "world with spaces" test"#;
        let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
        assert_eq!(result, vec!["hello", r#""world with spaces""#, "test"]);
    }

    #[test]
    fn test_split_keep_symbol_iterator() {
        let input = r#"hello "world with spaces" test"#;
        let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
        assert_eq!(result, vec!["hello", r#""world with spaces""#, "test"]);
    }

    #[test]
    fn test_multiple_quotes() {
        let input = r#"command "arg one" "arg two" final"#;
        let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
        assert_eq!(
            result,
            vec!["command", r#""arg one""#, r#""arg two""#, "final"]
        );
    }

    #[test]
    fn test_no_quotes() {
        let input = "hello world test";
        let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
        assert_eq!(result, vec!["hello", "world", "test"]);
    }

    #[test]
    fn test_empty_quotes() {
        let input = r#"hello "" world"#;
        let result: Vec<&str> = split_keep_symbol(input, " ", r#"""#).collect();
        assert_eq!(result, vec!["hello", r#""""#, "world"]);
    }

    #[test]
    fn test_empty_split_chars() {
        let input = "hello world";
        let result = split_keep_symbol(input, "", "").collect::<Vec<&str>>();
        assert_eq!(
            result,
            vec!["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]
        );
    }

    #[test]
    fn test_split_chinese() {
        let s = "feature/20251103_27134199_fix_console_count_invoke_1 2025-11-03 fix:控制台计量统计问题";
        let s = "feature/20251030_27098361_fix_security_issues_1 2025-10-30 fix:校验是否是合法的ossendpoin";
        let s = "feature/20251030_27098361_fix_security_issues_1 2025-10-30 fix:校验是否是合法的ossendpoint";
        let result = split_space_keep_symbol(s, "\"");
        println!("result, {}", result.collect::<String>());
    }
}

