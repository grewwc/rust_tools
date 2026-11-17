use mcp_stdio::{CAP_CHARS, cap_text};

// Preserve the original implementation as an independent byte-for-byte oracle.
fn original_cap_text(s: &str) -> String {
    if s.chars().count() <= CAP_CHARS {
        return s.to_string();
    }
    let truncated: String = s.chars().take(CAP_CHARS).collect();
    format!(
        "{truncated}\n\n[... truncated: output exceeded {CAP_CHARS} chars; narrow the request (smaller selector/range) or write the full result to a file instead ...]"
    )
}

fn assert_equivalence(char_count: usize) {
    assert_eq!(CAP_CHARS, 24_000);
    // Exercise one-, two-, three-, and four-byte scalars, including combining marks.
    for alphabet in ["a", "é", "界", "🦀", "aé界🦀e\u{0301}"] {
        let input: String = alphabet.chars().cycle().take(char_count).collect();
        assert_eq!(input.chars().count(), char_count);
        assert_eq!(
            cap_text(&input).as_bytes(),
            original_cap_text(&input).as_bytes(),
            "mismatch for {char_count} characters using {alphabet:?}"
        );
    }
}

#[test]
fn cap_text_matches_original_at_23999_chars() {
    assert_equivalence(23_999);
}

#[test]
fn cap_text_matches_original_at_24000_chars() {
    assert_equivalence(24_000);
}

#[test]
fn cap_text_matches_original_at_24001_chars() {
    assert_equivalence(24_001);
}
