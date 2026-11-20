//! Guard for a reasoning stream that decodes into garbage.
//!
//! `has_degenerate_repetition` covers verbatim loops. This guard covers the
//! other observed corruption shape: the provider's token stream decodes into
//! non-text (U+FFFD replacement characters, raw control characters or Unicode
//! noncharacters). Those bytes are machine-decidable evidence that the
//! generation is corrupt, so the stream is stopped and the turn loop retries
//! the request.
//!
//! The guard deliberately does not judge semantic incoherence or script mixing.
//! Normal reasoning legitimately quotes code, foreign languages and the exact
//! special-token placeholders this guard was built to catch, so placeholder
//! text alone is never a stop signal. `quoted_content_spans` masks fenced
//! blocks, inline code and quoted strings before anything is counted, and a
//! single replacement character is not enough.

use super::quoted_content_spans;

/// Finish reason recorded when reasoning corruption stops the stream. The turn
/// loop gives it a dedicated retry budget instead of the generic truncation
/// ladder.
pub(in crate::ai::stream) const CORRUPTED_REASONING_FINISH_REASON: &str = "corrupted_reasoning";

/// Two unquoted replacement characters rule out one isolated lossy quote.
const REPLACEMENT_CHAR_LIMIT: usize = 2;
/// A raw control character or Unicode noncharacter is never valid model text.
const NON_TEXT_CHAR_LIMIT: usize = 1;
/// Enough trailing bytes to see a candidate marker split across stream chunks.
const CANDIDATE_OVERLAP: usize = 64;

/// Incremental guard over the accumulated reasoning text.
///
/// The full scan runs only when the newly appended bytes contain a candidate
/// pattern, so ordinary reasoning costs one substring check per chunk instead
/// of rescanning the whole chain.
#[derive(Default)]
pub(in crate::ai::stream) struct ReasoningCorruptionScanner {
    scanned: usize,
    probe_tail: String,
    hit: bool,
}

impl ReasoningCorruptionScanner {
    pub(in crate::ai::stream) fn new() -> Self {
        Self::default()
    }

    /// Feed the current accumulated reasoning text. Returns true once the text
    /// carries enough unquoted corruption evidence to stop the stream.
    pub(in crate::ai::stream) fn observe(&mut self, text: &str) -> bool {
        if self.hit {
            return true;
        }
        if text.len() < self.scanned || !text.is_char_boundary(self.scanned) {
            *self = Self::default();
        }
        if text.len() == self.scanned {
            return false;
        }

        let base = self.scanned;
        self.scanned = text.len();

        // Keep a short tail from the previous chunk so a marker split across
        // chunks still triggers the full scan.
        let mut probe = std::mem::take(&mut self.probe_tail);
        probe.push_str(&text[base..]);
        self.probe_tail = clamp_suffix(&probe, CANDIDATE_OVERLAP).to_string();
        if !contains_corruption_candidate(&probe) {
            return false;
        }

        self.hit = has_unquoted_corruption(text);
        self.hit
    }
}

/// Count unquoted corruption evidence in the whole reasoning text. Kept
/// conservative: every trigger requires evidence that normal prose does not
/// produce, and quoted/code content is masked.
fn has_unquoted_corruption(text: &str) -> bool {
    let mut quoted = quoted_content_spans(text);
    // A fence still streaming (no closing ``` yet) must not hide corruption that
    // follows it. A closed fence ends with its closing delimiter and stays masked.
    if let Some(&(lo, hi)) = quoted.last()
        && hi == text.len()
        && text[lo..hi].starts_with("```")
        && !text[lo..hi].ends_with("```")
    {
        quoted.pop();
    }

    let mut replacement_chars = 0usize;
    let mut non_text_chars = 0usize;
    for (idx, ch) in text.char_indices() {
        if is_quoted(idx, &quoted) {
            continue;
        }
        if ch == '\u{FFFD}' {
            replacement_chars += 1;
            if replacement_chars >= REPLACEMENT_CHAR_LIMIT {
                return true;
            }
        } else if is_non_text_char(ch) {
            non_text_chars += 1;
            if non_text_chars >= NON_TEXT_CHAR_LIMIT {
                return true;
            }
        }
    }

    false
}

fn contains_corruption_candidate(text: &str) -> bool {
    text.contains('\u{FFFD}')
        || text.chars().any(is_non_text_char)
}

fn is_quoted(idx: usize, quoted: &[(usize, usize)]) -> bool {
    quoted.iter().any(|&(lo, hi)| idx >= lo && idx < hi)
}

fn clamp_suffix(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut start = text.len() - max_bytes;
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

/// Raw C0/C1 controls (except tab/newline/carriage return) and every Unicode
/// noncharacter, including the plane-ending U+?FFFE/U+?FFFF points. None are
/// valid in decoded model text.
fn is_non_text_char(ch: char) -> bool {
    let code = ch as u32;
    matches!(
        code,
        0x0000..=0x0008 | 0x000B..=0x000C | 0x000E..=0x001F | 0x007F..=0x009F
    ) || (0xFDD0..=0xFDEF).contains(&code)
        || (code & 0xFFFF) == 0xFFFE
        || (code & 0xFFFF) == 0xFFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screenshot_shaped_token_soup_is_detected() {
        let text = "enumerated RavDar/adminIndex fencing with 淵 and metsatzenorda\n\
                    < | box | > >spot 淵\n\
                    < /box >\n\
                    < | place__holder__no__114 | >\n\
                    TaiWake她还是chwitz compoundedovspich \u{FFFD}\u{FFFD}";
        assert!(has_unquoted_corruption(text));
    }

    #[test]
    fn scanner_detects_corruption_split_across_chunks() {
        let text = "normal start ... < | box | > and more junk < /box > and \u{FFFD}\u{FFFD}";
        let split = text.find("< /box >").unwrap() + 2;
        let mut scanner = ReasoningCorruptionScanner::new();
        assert!(!scanner.observe(&text[..split]));
        assert!(scanner.observe(text));
    }

    #[test]
    fn normal_bilingual_reasoning_is_not_corrupted() {
        let text = "需要检查 WebUI 的 session 状态：subagent 正在运行，但状态仍显示为 idle。\
                    I will read `src/bin/ai/serve/app.html` and compare the fetch handler, \
                    then verify the status transition in the browser UI.";
        assert!(!has_unquoted_corruption(text));
    }

    #[test]
    fn quoted_markers_and_replacement_chars_are_ignored() {
        let text = "The detector pattern is `place__holder__no__<n>`; the leaked form is:\n\
                    ```\n< | box | >\n< /box >\n\u{FFFD}\u{FFFD}\n```\n\
                    Those are captured examples, not live corruption.";
        assert!(!has_unquoted_corruption(text));
    }

    #[test]
    fn one_replacement_char_is_not_enough() {
        assert!(!has_unquoted_corruption(
            "A single lossy character \u{FFFD} can be a quoted artifact."
        ));
    }

    #[test]
    fn unclosed_fence_does_not_mask_later_replacement_chars() {
        let text = "example:\n```\nlet value = input;\n\u{FFFD}\u{FFFD}";
        assert!(has_unquoted_corruption(text));
    }

    #[test]
    fn placeholder_prose_without_decoded_garbage_is_not_corrupted() {
        let text = "The detector matches place__holder__no__114, < | box | > and \
                    < /box >; a review may list place__holder__no__114 and \
                    place__holder__no__114 while explaining the guard.";
        assert!(!has_unquoted_corruption(text));
    }

    #[test]
    fn raw_control_character_is_detected() {
        assert!(has_unquoted_corruption("normal text\u{0}tail"));
    }

    #[test]
    fn ordinary_line_breaks_and_tabs_are_not_control_evidence() {
        assert!(!has_unquoted_corruption("line one\nline two\tindented\r\nline three"));
    }
}
