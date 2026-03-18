use crate::jsonw::types::ParseOptions;

pub fn sanitize_json_input(s: &str, options: ParseOptions) -> String {
    let mut out = s.to_string();
    if options.allow_comment {
        out = strip_line_comments(&out);
    }
    if options.remove_special_chars {
        let bytes = out
            .as_bytes()
            .iter()
            .copied()
            .filter(|b| *b > 31)
            .collect::<Vec<u8>>();
        out = String::from_utf8_lossy(&bytes).to_string();
    }
    out
}

fn strip_line_comments(s: &str) -> String {
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
    fn test_sanitize_keeps_utf8_chinese() {
        let s = "{\n\"msg\": \"你好，世界\"}\u{0001}";
        let sanitized = sanitize_json_input(s, ParseOptions::default());
        let v: Value = serde_json::from_str(&sanitized).unwrap();
        assert_eq!(v["msg"], "你好，世界");
    }
}
