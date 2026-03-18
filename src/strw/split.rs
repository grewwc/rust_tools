use std::io::{BufRead, Read};
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

pub fn split_no_empty<'a>(s: &'a str, sep: &str) -> Vec<&'a str> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(sep).filter(|part| !part.is_empty()).collect()
}

pub fn split_by_cutset(s: &str, cutset: &str) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    if cutset.is_empty() {
        return vec![s.to_string()];
    }
    let set = FastSet::from_iter(cutset.chars());
    let mut res: Vec<String> = Vec::new();
    let mut buf = String::new();
    for ch in s.chars() {
        if set.contains(&ch) {
            if !buf.is_empty() {
                res.push(std::mem::take(&mut buf));
            }
        } else {
            buf.push(ch);
        }
    }
    if !buf.is_empty() {
        res.push(buf);
    }
    res
}

pub fn split_by_str_keep_quotes(
    s: &str,
    sep: &str,
    symbols: &str,
    keep_symbol: bool,
) -> Vec<String> {
    if sep.is_empty() {
        return vec![s.to_string()];
    }

    let mut symbol_set = FastSet::from_iter(symbols.chars());
    for ch in sep.chars() {
        symbol_set.remove(&ch);
    }
    if symbol_set.is_empty() {
        return vec![s.to_string()];
    }

    let sep_bytes = sep.as_bytes();
    let bytes = s.as_bytes();
    let mut res: Vec<String> = Vec::new();
    let mut in_quote = false;
    let mut buf: Vec<u8> = Vec::new();

    for (i, ch) in s.char_indices() {
        let prev_is_escape = i > 0 && bytes[i - 1] == b'\\';

        if symbol_set.contains(&ch) && !prev_is_escape {
            in_quote = !in_quote;
            if keep_symbol {
                let mut tmp = [0u8; 4];
                buf.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
            }
            continue;
        }

        let mut tmp = [0u8; 4];
        buf.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());

        if !in_quote
            && buf.len() >= sep_bytes.len()
            && buf[buf.len() - sep_bytes.len()..] == *sep_bytes
        {
            buf.truncate(buf.len() - sep_bytes.len());
            if !buf.is_empty() {
                res.push(String::from_utf8(std::mem::take(&mut buf)).unwrap_or_default());
            } else {
                buf.clear();
            }
        }
    }

    if !buf.is_empty() {
        res.push(String::from_utf8(buf).unwrap_or_default());
    }
    res
}

pub fn replace_all_in_quote_unchange(s: &str, old: char, new: char) -> String {
    let mut in_quote = false;
    let mut res = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch == '"' {
            in_quote = !in_quote;
            res.push(ch);
            continue;
        }
        if ch == old {
            if in_quote {
                res.push(old);
            } else {
                res.push(new);
            }
        } else {
            res.push(ch);
        }
    }
    res
}

pub fn replace_all_out_quote_unchange(s: &str, old: char, new: char) -> String {
    let mut in_quote = false;
    let mut res = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch == '"' {
            in_quote = !in_quote;
            res.push(ch);
            continue;
        }
        if ch == old {
            if in_quote {
                res.push(new);
            } else {
                res.push(old);
            }
        } else {
            res.push(ch);
        }
    }
    res
}

pub fn get_last_item<T>(slice: &[T]) -> T
where
    T: Copy + Default,
{
    match slice.last() {
        Some(v) => *v,
        None => T::default(),
    }
}

pub fn split_by_token<R: std::io::Read>(
    reader: R,
    token: &str,
    keep_token: bool,
) -> std::io::Result<Vec<String>> {
    if token.is_empty() {
        return Err(std::io::Error::other("token should not be empty"));
    }
    let mut r = std::io::BufReader::new(reader);
    let token_bytes = token.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk: Vec<u8> = Vec::new();

    loop {
        if buf.len() >= token_bytes.len() && buf[buf.len() - token_bytes.len()..] == *token_bytes {
            let mut s = buf.clone();
            if !keep_token {
                s.truncate(s.len() - token_bytes.len());
            }
            if !s.is_empty() {
                out.push(String::from_utf8_lossy(&s).to_string());
            }
            buf.clear();
        }

        chunk.clear();
        let n = {
            let mut byte = [0u8; 1];
            match r.read(&mut byte) {
                Ok(0) => 0,
                Ok(_) => {
                    chunk.push(byte[0]);
                    1
                }
                Err(e) => return Err(e),
            }
        };

        if n == 0 {
            break;
        }

        buf.extend_from_slice(&chunk);

        if token_bytes.len() > 1 {
            let mut rest = vec![0u8; token_bytes.len() - 1];
            match r.fill_buf() {
                Ok(avail) => {
                    let take = rest.len().min(avail.len());
                    rest[..take].copy_from_slice(&avail[..take]);
                    if take < rest.len() {
                        rest.truncate(take);
                    }
                }
                Err(e) => return Err(e),
            }
            if rest.len() == token_bytes.len() - 1 && rest == token_bytes[1..] {
                buf.extend_from_slice(&rest);
                r.consume(rest.len());
            }
        }
    }

    if !buf.is_empty() {
        out.push(String::from_utf8_lossy(&buf).to_string());
    }
    Ok(out)
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
        for (_, ch) in self.text[self.position..].char_indices() {
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
        for (_, ch) in self.text[self.position..].char_indices() {
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
        let s = "feature/20251030_27098361_fix_security_issues_1 2025-10-30 fix:校验是否是合法的ossendpoint";
        let result = split_space_keep_symbol(s, "\"");
        println!("result, {}", result.collect::<String>());
    }

    #[test]
    fn test_replace_all_in_quote_unchange() {
        let s = r#"a,"b,c",d"#;
        assert_eq!(replace_all_in_quote_unchange(s, ',', ';'), r#"a;"b,c";d"#);
        assert_eq!(replace_all_out_quote_unchange(s, ',', ';'), r#"a,"b;c",d"#);
    }

    #[test]
    fn test_split_by_token() {
        let input = "a<END>b<END>c";
        let parts = split_by_token(std::io::Cursor::new(input.as_bytes()), "<END>", false).unwrap();
        assert_eq!(parts, vec!["a", "b", "c"]);
        let parts = split_by_token(std::io::Cursor::new(input.as_bytes()), "<END>", true).unwrap();
        assert_eq!(parts, vec!["a<END>", "b<END>", "c"]);
    }
}
