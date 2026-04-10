/// Background color for code blocks.
/// A warm dark gray (#282828) that provides excellent contrast with Monokai colors.
pub(super) const MONOKAI_BG: &str = "\x1b[48;2;40;40;40m";
pub(super) const MONOKAI_FG: &str = "\x1b[38;2;248;248;242m";
pub(super) const MONOKAI_COMMENT: &str = "\x1b[38;2;117;113;94m";
pub(super) const MONOKAI_KEYWORD: &str = "\x1b[38;2;249;38;114m";
pub(super) const MONOKAI_STRING: &str = "\x1b[38;2;230;219;116m";
pub(super) const MONOKAI_NUMBER: &str = "\x1b[38;2;174;129;255m";
pub(super) const MONOKAI_TYPE: &str = "\x1b[38;2;102;217;239m";
pub(super) const MONOKAI_IDENT: &str = MONOKAI_FG;
pub(super) const MONOKAI_RESET: &str = "\x1b[0m";
/// Semantic alias: restore default foreground after a token.
pub(super) const MONOKAI_DEFAULT: &str = MONOKAI_FG;
/// Dimmed color for line numbers and fence borders (#6B6B6B)
pub(super) const MONOKAI_DIM: &str = "\x1b[38;2;107;107;107m";

// ---------------------------------------------------------------------------
// Parse code-block fence
// ---------------------------------------------------------------------------

pub(super) fn parse_code_block_language(trimmed: &str) -> Option<String> {
    let lang = trimmed
        .get(3..)
        .unwrap_or("")
        .split(|c: char| c.is_whitespace() || c == '{')
        .next()
        .unwrap_or("")
        .trim()
        .trim_matches('`')
        .trim_matches('~')
        .to_ascii_lowercase();
    if lang.is_empty() {
        return None;
    }
    Some(
        match lang.as_str() {
            "rs" => "rust",
            "py" => "python",
            "c++" | "cpp" | "cxx" | "cc" => "cpp",
            "c#" => "csharp",
            "js" => "javascript",
            "ts" => "typescript",
            "sh" | "shell" | "zsh" => "bash",
            other => other,
        }
        .to_string(),
    )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn lang_or(lang: Option<&str>) -> &str {
    lang.unwrap_or("")
}

fn line_comment_prefix(lang: Option<&str>) -> Option<(char, char)> {
    match lang_or(lang) {
        "rust" | "go" | "cpp" | "c" | "javascript" | "typescript" | "java" | "csharp" => {
            Some(('/', '/'))
        }
        _ => None,
    }
}

/// For unspecified language, `#` is treated as a comment start (fallback covers
/// Python/Bash which are common in AI output; this is intentional).
fn hash_starts_comment(lang: Option<&str>) -> bool {
    match lang {
        Some("python" | "bash") => true,
        None => true, // intentional fallback: treat `#` as comment when language unknown
        _ => false,
    }
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}

fn classify_identifier_color(ident: &str, lang: Option<&str>) -> &'static str {
    if is_keyword(ident, lang) {
        return MONOKAI_KEYWORD;
    }
    if is_type_like(ident, lang) {
        return MONOKAI_TYPE;
    }
    if matches!(ident, "true" | "false" | "nil" | "null" | "None" | "nullptr") {
        return MONOKAI_NUMBER;
    }
    MONOKAI_IDENT
}

// ---------------------------------------------------------------------------
// Keyword / type tables (sorted for potential binary_search)
// ---------------------------------------------------------------------------

const PYTHON_KEYWORDS: &[&str] = &[
    "and", "as", "async", "await", "break", "class", "continue", "def", "elif", "else", "except",
    "finally", "for", "from", "if", "import", "in", "is", "lambda", "not", "or", "pass", "raise",
    "return", "try", "while", "with", "yield",
];

const GO_KEYWORDS: &[&str] = &[
    "case", "chan", "const", "default", "defer", "else", "fallthrough", "for", "func", "go",
    "if", "import", "interface", "map", "package", "range", "return", "select", "struct",
    "switch", "type", "var",
];

const CPP_KEYWORDS: &[&str] = &[
    "auto", "case", "class", "const", "constexpr", "default", "delete", "else", "enum", "for",
    "if", "inline", "namespace", "new", "private", "protected", "public", "return", "struct",
    "switch", "template", "typedef", "typename", "using", "virtual", "void", "while",
];

const RUST_KEYWORDS: &[&str] = &[
    "as", "async", "await", "break", "const", "continue", "crate", "else", "enum", "extern",
    "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub",
    "ref", "return", "Self", "self", "static", "struct", "super", "trait", "type", "unsafe",
    "use", "where", "while",
];

const PYTHON_TYPES: &[&str] = &["bool", "dict", "float", "int", "list", "str", "tuple"];
const GO_TYPES: &[&str] = &["bool", "byte", "error", "int", "int64", "rune", "string", "uint64"];
const CPP_TYPES: &[&str] = &[
    "bool", "char", "double", "float", "int", "long", "short", "size_t", "std", "string",
];
const RUST_TYPES: &[&str] = &[
    "Option", "Result", "Self", "String", "Vec", "bool", "char", "f32", "f64", "i16", "i32",
    "i64", "i8", "isize", "str", "u16", "u32", "u64", "u8", "usize",
];

fn is_keyword(ident: &str, lang: Option<&str>) -> bool {
    let table: &[&str] = match lang_or(lang) {
        "python" => PYTHON_KEYWORDS,
        "go" => GO_KEYWORDS,
        "cpp" | "c" => CPP_KEYWORDS,
        _ => RUST_KEYWORDS,
    };
    table.contains(&ident)
}

fn is_type_like(ident: &str, lang: Option<&str>) -> bool {
    let table: &[&str] = match lang_or(lang) {
        "python" => PYTHON_TYPES,
        "go" => GO_TYPES,
        "cpp" | "c" => CPP_TYPES,
        _ => RUST_TYPES,
    };
    table.contains(&ident)
}

// ---------------------------------------------------------------------------
// Main highlight loop
// ---------------------------------------------------------------------------

pub(super) fn highlight_code_line(line: &str, lang: Option<&str>) -> String {
    let mut out = String::with_capacity(line.len() + 32);
    out.push_str(MONOKAI_DEFAULT);
    let mut chars = line.chars().peekable();
    let line_comment = line_comment_prefix(lang);

    while let Some(ch) = chars.next() {
        // --- line comment (//) ---
        if let Some(prefix) = line_comment
            && ch == prefix.0
            && chars.peek().copied() == Some(prefix.1)
        {
            out.push_str(MONOKAI_COMMENT);
            out.push(ch);
            out.push(chars.next().unwrap_or(prefix.1));
            for rest in chars.by_ref() {
                out.push(rest);
            }
            out.push_str(MONOKAI_DEFAULT);
            break;
        }

        // --- hash comment (#) ---
        if ch == '#' && hash_starts_comment(lang) {
            out.push_str(MONOKAI_COMMENT);
            out.push(ch);
            for rest in chars.by_ref() {
                out.push(rest);
            }
            out.push_str(MONOKAI_DEFAULT);
            break;
        }

        // --- strings ---
        if matches!(ch, '"' | '\'' | '`') {
            out.push_str(MONOKAI_STRING);
            out.push(ch);
            let quote = ch;
            let mut escaped = false;
            while let Some(next) = chars.next() {
                out.push(next);
                if escaped {
                    escaped = false;
                    continue;
                }
                if next == '\\' && quote != '`' {
                    escaped = true;
                    continue;
                }
                if next == quote {
                    break;
                }
            }
            out.push_str(MONOKAI_DEFAULT);
            continue;
        }

        // --- numbers ---
        if ch.is_ascii_digit() {
            out.push_str(MONOKAI_NUMBER);
            out.push(ch);
            // Only consume digits, separators, decimal point — do NOT blindly
            // accept x/o/b or all hex chars (avoids "10x" eating the 'x').
            while let Some(next) = chars.peek().copied() {
                if next.is_ascii_digit() || matches!(next, '.' | '_') {
                    out.push(next);
                    let _ = chars.next();
                } else {
                    break;
                }
            }
            out.push_str(MONOKAI_DEFAULT);
            continue;
        }

        // --- identifiers / keywords / types ---
        if is_ident_start(ch) {
            let mut ident = String::new();
            ident.push(ch);
            while let Some(next) = chars.peek().copied() {
                if is_ident_continue(next) {
                    ident.push(next);
                    let _ = chars.next();
                } else {
                    break;
                }
            }
            let color = classify_identifier_color(&ident, lang);
            out.push_str(color);
            out.push_str(&ident);
            out.push_str(MONOKAI_DEFAULT);
            continue;
        }

        out.push(ch);
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::stream::render::markdown::MarkdownStreamRenderer;

    #[test]
    fn rust_code_block_uses_monokai_like_colors() {
        let mut renderer = MarkdownStreamRenderer::new_with_tty(false);
        let fence = renderer.consume_line("```rust", false);
        assert!(fence.contains(MONOKAI_BG));
        assert_eq!(renderer.code_block_lang(), Some("rust"));

        let code = renderer.consume_line("fn main() { let x = 42; }", false);
        assert!(code.contains(MONOKAI_KEYWORD));
        assert!(code.contains(MONOKAI_NUMBER));

        let _ = renderer.consume_line("```", false);
        assert!(renderer.code_block_lang().is_none());
    }

    // ---- parse_code_block_language ----

    #[test]
    fn test_parse_rs_alias() {
        assert_eq!(parse_code_block_language("```rs"), Some("rust".to_string()));
    }

    #[test]
    fn test_parse_py_alias() {
        assert_eq!(parse_code_block_language("~~~py"), Some("python".to_string()));
    }

    #[test]
    fn test_parse_with_options() {
        assert_eq!(
            parse_code_block_language("```rust{some_option}"),
            Some("rust".to_string())
        );
    }

    #[test]
    fn test_parse_empty_fence() {
        assert_eq!(parse_code_block_language("```"), None);
        assert_eq!(parse_code_block_language("~~~"), None);
    }

    #[test]
    fn test_parse_too_short() {
        assert_eq!(parse_code_block_language("``"), None);
    }

    #[test]
    fn test_parse_unknown_lang_preserved() {
        assert_eq!(
            parse_code_block_language("```elixir"),
            Some("elixir".to_string())
        );
    }

    // ---- highlight_code_line ----

    #[test]
    fn test_highlight_comment_rust() {
        let out = highlight_code_line("// hello world", Some("rust"));
        assert!(out.contains(MONOKAI_COMMENT));
    }

    #[test]
    fn test_highlight_comment_python() {
        let out = highlight_code_line("# hello world", Some("python"));
        assert!(out.contains(MONOKAI_COMMENT));
    }

    #[test]
    fn test_highlight_comment_hash_fallback() {
        let out = highlight_code_line("# comment", None);
        assert!(out.contains(MONOKAI_COMMENT));
    }

    #[test]
    fn test_highlight_comment_hash_not_for_rust() {
        let out = highlight_code_line("# not a comment in rust", Some("rust"));
        // '#' is not a comment in rust, so MONOKAI_COMMENT should not appear
        assert!(!out.contains(MONOKAI_COMMENT));
    }

    #[test]
    fn test_highlight_string_with_escape() {
        let out = highlight_code_line(r#""hello \"world\"""#, Some("rust"));
        assert!(out.contains(MONOKAI_STRING));
    }

    #[test]
    fn test_highlight_single_quoted_string() {
        let out = highlight_code_line("'c'", Some("rust"));
        assert!(out.contains(MONOKAI_STRING));
    }

    #[test]
    fn test_highlight_number_decimal() {
        let out = highlight_code_line("42", Some("rust"));
        assert!(out.contains(MONOKAI_NUMBER));
    }

    #[test]
    fn test_highlight_number_underscore_sep() {
        let out = highlight_code_line("1_000_000", Some("rust"));
        assert!(out.contains(MONOKAI_NUMBER));
    }

    #[test]
    fn test_highlight_number_float() {
        let out = highlight_code_line("3.14", Some("rust"));
        assert!(out.contains(MONOKAI_NUMBER));
    }

    #[test]
    fn test_highlight_number_does_not_swallow_x() {
        // "10xyz" — old code consumed 'x' as part of the number.
        // Now only digits/./_ are consumed after a digit start.
        let out = highlight_code_line("10xyz", Some("rust"));
        let number_region = out.split(MONOKAI_DEFAULT).next().unwrap_or("");
        assert!(
            !number_region.contains('x'),
            "'x' should not be inside the number region, got: {number_region:?}"
        );
    }

    #[test]
    fn test_highlight_keyword_rust() {
        let out = highlight_code_line("fn main() {}", Some("rust"));
        assert!(out.contains(MONOKAI_KEYWORD));
    }

    #[test]
    fn test_highlight_keyword_python() {
        let out = highlight_code_line("def foo():", Some("python"));
        assert!(out.contains(MONOKAI_KEYWORD));
    }

    #[test]
    fn test_highlight_type_rust() {
        let out = highlight_code_line("let x: String;", Some("rust"));
        assert!(out.contains(MONOKAI_TYPE));
    }

    #[test]
    fn test_highlight_type_go() {
        let out = highlight_code_line("var s string", Some("go"));
        assert!(out.contains(MONOKAI_TYPE));
    }

    #[test]
    fn test_highlight_literal_bool() {
        let out = highlight_code_line("true", Some("rust"));
        assert!(out.contains(MONOKAI_NUMBER));
    }

    #[test]
    fn test_highlight_literal_null() {
        let out = highlight_code_line("null", Some("javascript"));
        assert!(out.contains(MONOKAI_NUMBER));
    }

    #[test]
    fn test_highlight_no_double_color_leak() {
        let out = highlight_code_line("fn main() {}", Some("rust"));
        let reset_count = out.matches(MONOKAI_DEFAULT).count();
        assert!(reset_count >= 2, "expected multiple resets, got {}", reset_count);
    }

    #[test]
    fn test_highlight_empty_line() {
        let out = highlight_code_line("", Some("rust"));
        assert_eq!(out, MONOKAI_DEFAULT);
    }
}
