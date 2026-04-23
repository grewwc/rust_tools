use crate::jsonw::types::ParseOptions;

pub fn sanitize_json_input(s: &str, options: ParseOptions) -> String {
    let mut out = s.to_string();
    if options.allow_comment {
        out = strip_comments(&out);
    }
    if options.remove_special_chars {
        // 移除控制字符，但保留换行和制表符
        // 使用 chars() 而不是 bytes() 来正确处理 UTF-8 编码的多字节中文字符
        out = out
            .chars()
            .filter(|c| !c.is_control() || *c == '\n' || *c == '\r' || *c == '\t')
            .collect();
    }
    out
}

fn strip_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_string = false;
    let mut escape = false;
    let mut chars = s.chars().peekable();

    while let Some(ch) = chars.next() {
        if escape {
            out.push(ch);
            escape = false;
            continue;
        }

        if in_string {
            if ch == '\\' {
                out.push(ch);
                escape = true;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            out.push(ch);
            continue;
        }

        if ch == '"' {
            in_string = true;
            out.push(ch);
            continue;
        }

        // Check for block comment /*
        if ch == '/' && chars.peek().copied() == Some('*') {
            chars.next();
            let mut newlines = 0;
            while let Some(c) = chars.next() {
                if c == '\n' {
                    newlines += 1;
                }
                if c == '*' && chars.peek().copied() == Some('/') {
                    chars.next();
                    break;
                }
            }
            // Preserve newlines to maintain line numbers
            for _ in 0..newlines {
                out.push('\n');
            }
            continue;
        }

        // Check for line comment //
        if ch == '/' && chars.peek().copied() == Some('/') {
            chars.next();
            for c in chars.by_ref() {
                if c == '\n' {
                    out.push('\n');
                    break;
                }
            }
            continue;
        }

        out.push(ch);
    }

    out
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    #[test]
    fn test_strip_line_comments_respects_quotes() {
        let s = r#"{ "a": "x//y", // comment
  "b": 1 }// tail"#;
        let sanitized = sanitize_json_input(s, ParseOptions::default());
        let v: Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(v["a"], "x//y");
        assert_eq!(v["b"], 1);
    }

    #[test]
    fn test_strip_block_comments() {
        let s = r#"{
    /* comment */
    "a": 1
}"#;
        let sanitized = sanitize_json_input(s, ParseOptions::default());
        let v: Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn test_strip_mixed_comments() {
        let s = r#"{
    // line comment
    "a": 1,
    /* block comment
       spanning multiple lines */
    "b": 2
}"#;
        let sanitized = sanitize_json_input(s, ParseOptions::default());
        let v: Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(v["a"], 1);
        assert_eq!(v["b"], 2);
    }

    #[test]
    fn test_block_comment_in_string() {
        let s = r#"{
    "a": "/* not a comment */"
}"#;
        let sanitized = sanitize_json_input(s, ParseOptions::default());
        let v: Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(v["a"], "/* not a comment */");
    }

    #[test]
    fn test_nested_block_comments_not_supported() {
        // Nested block comments are not supported, this tests that we stop at first */
        let s = r#"{
    /* outer /* inner */ still in outer */
    "a": 1
}"#;
        let sanitized = sanitize_json_input(s, ParseOptions::default());
        // The first */ closes the comment, so " still in outer */" becomes garbage
        // This is expected behavior - nested block comments are not standard
        assert!(serde_json::from_str::<Value>(&sanitized).is_err());
    }
}
