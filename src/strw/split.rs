use std::collections::HashSet;

const COMMON_WHITESPACES: &str = " \t\n\r\x0B\x0C";

/// Function that returns an iterator similar to std::str::Split
pub fn split_keep_symbol<'a>(s: &'a str, split_chars: &'a str, symbols: &'a str) -> QuoteSplit<'a> {
    QuoteSplit::new(s, split_chars, symbols)
}

pub fn split_space_keep_symbol<'a>(s: &'a str, symbols: &'a str) -> QuoteSplit<'a> {
    split_keep_symbol(s, COMMON_WHITESPACES, symbols)
}

/// Alternative implementation using a custom iterator that behaves more like Split
pub struct QuoteSplit<'a> {
    text: &'a str,
    position: usize,
    symbols: std::collections::HashSet<char>,
    split_chars: std::collections::HashSet<char>,
}

impl<'a> QuoteSplit<'a> {
    pub fn new(text: &'a str, split_chars: &'a str, symbols: &'a str) -> Self {
        let mut symbol_set = HashSet::new();
        symbol_set.reserve(symbols.len());
        symbols.chars().for_each(|ch| {
            symbol_set.insert(ch);
        });

        let mut split_chars_set = HashSet::new();
        split_chars_set.reserve(split_chars.len());
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
        while self.position < self.text.len()
            && self.is_whitespace(self.text.chars().nth(self.position).unwrap())
        {
            self.position += 1;
        }

        if self.position >= self.text.len() {
            return None;
        }

        let start = self.position;
        let mut in_quotes = false;

        while self.position < self.text.len() {
            if self.split_chars.is_empty() {
                self.position += 1;
                return Some(&self.text[start..self.position]);
            }

            let ch = self.text.chars().nth(self.position).unwrap();
            if self.symbols.contains(&ch) {
                in_quotes = !in_quotes;
                self.position += 1;
            } else if !in_quotes && self.split_chars.contains(&ch) {
                break;
            } else {
                self.position += 1;
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
}

