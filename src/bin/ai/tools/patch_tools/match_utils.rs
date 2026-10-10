use super::*;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatchMode {
    /// Exact matching (allows trailing-whitespace differences), used by default.
    Strict,
    /// Ignores leading-indentation differences; only used as a fallback when strict matching
    /// fails to locate anything in the whole file.
    /// Aligns with `git apply --ignore-whitespace`: models often fail to reproduce the
    /// indentation of markdown/nested lists/code blocks exactly, causing strict matching to
    /// fail on the whole block with context mismatch.
    IgnoreIndent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContextPolicy {
    /// Context lines must match, and remove lines must also match.
    Require,
    /// Context lines are only a locating reference; when applying, keep the file's actual
    /// context, while remove lines must still match.
    Fuzz,
}

/// Strips the line-number prefix that read_file / grep and other tool outputs add
/// (single-argument fallback version).
/// Models sometimes accidentally copy the line-number prefix into a patch's context/remove lines.
///
/// This version is for scenarios with **no "real line" to anchor on** (e.g. IgnoreIndent
/// normalizes both sides independently). When a real line is available for comparison, prefer
/// the anchored [`strip_number_prefix_anchored`], which is separator-agnostic and has almost
/// zero false positives. Here, to avoid wrongly stripping code lines that genuinely start with
/// digits (e.g. `80:80`, `42px`, `3.14`), we take a **conservative** approach and only recognize
/// two highly deterministic line-number-column shapes:
/// - `digits + \t`: read_file's real format (`{:>6}\t{}`). The line content (including its own
///   indentation) follows directly after the TAB; only this single TAB is consumed.
/// - `digits + single non-alphanumeric separator + space`: grep-like (`42| `, `42: `). The
///   separator must be **followed by a space**, so `80:80` (`:` followed by a digit) and `3.14`
///   (`.` followed by a digit) are not wrongly stripped.
pub(crate) fn strip_line_number_prefix(s: &str) -> &str {
    let trimmed = s.trim_start();
    let digits_end = trimmed.find(|c: char| !c.is_ascii_digit()).unwrap_or(0);
    if digits_end == 0 {
        return s;
    }
    let after_digits = &trimmed[digits_end..];
    let mut chars = after_digits.chars();
    let sep = match chars.next() {
        Some(c) => c,
        None => return s,
    };
    // TAB: read_file's real separator; the content (including indentation) follows directly
    // after it; only this single TAB is consumed.
    if sep == '\t' {
        return &after_digits['\t'.len_utf8()..];
    }
    // Other separators: must be a single non-alphanumeric, non-space character followed
    // immediately by a space (`42| ` / `42: `). Requiring the trailing space avoids mistaking
    // `80:80` or `3.14` for a line-number column.
    if sep.is_alphanumeric() || sep == ' ' {
        return s;
    }
    let rest = &after_digits[sep.len_utf8()..];
    match rest.strip_prefix(' ') {
        Some(after_space) => after_space,
        None => s,
    }
}

/// Anchored line-number-prefix stripping: using `actual` (the file's real line, which never
/// contains a line-number column) as ground truth, decides whether `expected` (a patch line,
/// possibly with a model-miscopied line-number column) is **exactly equal** to `actual` after
/// removing the "digit column". If so, returns the de-columned content; otherwise returns
/// `expected` unchanged.
///
/// Compared with enumerating separators, this is separator-agnostic (`\t` `|` `:` space `.`
/// `)` are all compatible), and because it requires "the remainder to exactly equal the real
/// line", it can almost never wrongly hit code lines that genuinely start with digits — even if
/// it happens to, lines_match's multi-match (ambiguity) detection intercepts it.
pub(crate) fn strip_number_prefix_anchored<'a>(expected: &'a str, actual: &str) -> &'a str {
    // expected must start with "optional whitespace + digits", otherwise it cannot be
    // "line-number column + actual".
    let lead_ws_end = expected
        .find(|c: char| !c.is_whitespace())
        .unwrap_or(expected.len());
    let after_ws = &expected[lead_ws_end..];
    let digits_end = after_ws
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after_ws.len());
    if digits_end == 0 {
        return expected; // the digit part is empty: not a line-number column.
    }
    let after_digits = &after_ws[digits_end..];
    // The remainder after removing 1 separator (on char boundaries, to avoid multi-byte UTF-8
    // slice panics).
    let after_one_sep = after_digits
        .char_indices()
        .nth(1)
        .map(|(byte_idx, _)| &after_digits[byte_idx..])
        .unwrap_or("");
    // Try each candidate: does removing 0 or 1 separators (optionally with 1 space) equal the
    // real line? "The remainder exactly equals the real line" is the only criterion, so we don't
    // need to know what the separator actually is.
    let candidates = [
        after_digits,  // digits directly followed by content (rare)
        after_one_sep, // consume 1 separator
    ];
    for cand in candidates {
        if cand == actual || cand.trim_end() == actual.trim_end() {
            return cand;
        }
        if let Some(c2) = cand.strip_prefix(' ')
            && (c2 == actual || c2.trim_end() == actual.trim_end())
        {
            return c2;
        }
    }
    expected
}

/// Single-character confusable normalization (strict 1:1 mapping, no width expansion).
/// Shared with [`normalize_confusables`] to keep "whole-string normalization" and
/// "per-character equivalence checks" consistent.
pub(crate) fn normalize_confusable_char(c: char) -> char {
    match c {
        // --- dash family ---
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
        | '\u{2212}' => '-',
        // --- smart double quotes ---
        '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{2033}' => '"',
        // --- smart single quotes ---
        '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' | '\u{2032}' => '\'',
        // --- non-breaking space family ---
        '\u{00A0}' | '\u{202F}' | '\u{2007}' | '\u{2060}' => ' ',
        other => other,
    }
}

/// Normalizes common Unicode "confusable" characters to ASCII-equivalent forms.
///
/// Only used for **locating decisions** in patch matching (lines_match); never participates in
/// constructing output content.
/// All handled characters are purely typographic differences that don't affect semantics:
/// - dash family (— – ― etc.) -> '-'
/// - smart quotes (" " ' ' ‛ ‟) -> '"' / "'"
/// - non-breaking spaces (NBSP U+00A0, NNBSP U+202F, etc.) -> regular space
pub(crate) fn normalize_confusables(s: &str) -> String {
    s.chars().map(normalize_confusable_char).collect()
}

/// Locates `old` in `line` via **tolerant matching** (only for the `old` fallback locating in
/// `*** Replace in line:`):
/// - per-character confusable normalization equivalence (1:1, see [`normalize_confusable_char`]),
///   so em-dash/smart quotes/NBSP match their ASCII-equivalent forms;
/// - ignores leading/trailing whitespace in `old` (models often copy leading spaces into `old`
///   while reproducing indentation).
///
/// Returns the (byte_start, byte_end) of the match in the original line. Must be unique:
/// multiple matches return [`TolerantMatchError::Ambiguous`]. The replacement still slices on the
/// original byte range; the written content is constructed from `new`, so normalized characters
/// are never written into the file.
pub(crate) fn find_tolerant_old_match(line: &str, old: &str) -> Result<(usize, usize), TolerantMatchError> {
    let needle: Vec<char> = old.trim().chars().collect();
    let hay: Vec<char> = line.chars().collect();
    if needle.is_empty() || needle.len() > hay.len() {
        return Err(TolerantMatchError::NoMatch);
    }
    // Precompute each char's byte offset once in O(n) to avoid per-position nth().
    let byte_offsets: Vec<usize> = line.char_indices().map(|(b, _)| b).collect();
    let mut found: Vec<(usize, usize)> = Vec::new();
    'outer: for i in 0..=(hay.len() - needle.len()) {
        for (j, &nc) in needle.iter().enumerate() {
            if normalize_confusable_char(hay[i + j]) != normalize_confusable_char(nc) {
                continue 'outer;
            }
        }
        let byte_start = byte_offsets[i];
        let byte_end = byte_offsets
            .get(i + needle.len())
            .copied()
            .unwrap_or(line.len());
        found.push((byte_start, byte_end));
    }
    match found.len() {
        0 => Err(TolerantMatchError::NoMatch),
        1 => Ok(found[0]),
        n => Err(TolerantMatchError::Ambiguous(n)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TolerantMatchError {
    NoMatch,
    Ambiguous(usize),
}

pub(crate) fn lines_match_exact(actual: &str, expected: &str, mode: MatchMode) -> bool {
    if actual == expected || actual.trim_end() == expected.trim_end() {
        return true;
    }
    match mode {
        MatchMode::Strict => {
            // Models often copy the line-number prefix from read_file output (e.g.
            // `    42\t<code>`). Prefer the anchored approach: using actual (the real file line)
            // as the reference, check whether expected equals actual exactly after removing the
            // digit column — separator-agnostic and with almost zero false positives.
            let e = strip_number_prefix_anchored(expected, actual);
            if e == actual || e.trim_end() == actual.trim_end() {
                return true;
            }
            // Fallback: generic digit-column stripping on both sides when no actual anchor info
            // is available, also covering edge cases where the actual side carries a column too.
            let expected_stripped = strip_line_number_prefix(expected);
            let actual_stripped = strip_line_number_prefix(actual);
            expected_stripped == actual_stripped
                || expected_stripped.trim_end() == actual_stripped.trim_end()
        }
        MatchMode::IgnoreIndent => {
            // Try the anchored approach first (based on actual.trim), then fall back to generic
            // two-sided stripping + trim.
            let e = strip_number_prefix_anchored(expected.trim_start(), actual.trim());
            if e.trim() == actual.trim() {
                return true;
            }
            strip_line_number_prefix(actual).trim() == strip_line_number_prefix(expected).trim()
        }
    }
}

/// Common entry point for lines_match: exact matching first, then compare after normalizing
/// confusable characters.
///
/// Normalization only affects whether a line "can be located". Output content is constructed by
/// try_apply_hunk_at:
/// - Context lines output actual (the original file content)
/// - Remove lines are dropped after matching
/// - Add lines use the patch content directly
/// So when normalized matching succeeds, the file still receives the original file's Unicode
/// characters — content is never "replaced with the wrong thing".
pub(crate) fn lines_match(actual: &str, expected: &str, mode: MatchMode) -> bool {
    if lines_match_exact(actual, expected, mode) {
        return true;
    }
    let actual_n = normalize_confusables(actual);
    let expected_n = normalize_confusables(expected);
    if actual_n == expected_n || actual_n.trim_end() == expected_n.trim_end() {
        return true;
    }
    match mode {
        MatchMode::Strict => {
            let e = strip_number_prefix_anchored(&expected_n, &actual_n);
            if e == actual_n || e.trim_end() == actual_n.trim_end() {
                return true;
            }
            let a = strip_line_number_prefix(&actual_n);
            let e = strip_line_number_prefix(&expected_n);
            a == e || a.trim_end() == e.trim_end()
        }
        MatchMode::IgnoreIndent => {
            let e = strip_number_prefix_anchored(expected_n.trim_start(), actual_n.trim());
            if e.trim() == actual_n.trim() {
                return true;
            }
            strip_line_number_prefix(&actual_n).trim()
                == strip_line_number_prefix(&expected_n).trim()
        }
    }
}

/// Extracts the hunk's context+remove lines (i.e. the lines expected to match in the original file).
pub(crate) fn hunk_expected_lines(hunk: &UnifiedHunk) -> Vec<&str> {
    hunk.lines
        .iter()
        .filter_map(|line| match line {
            UnifiedLine::Context(s) | UnifiedLine::Remove(s) => Some(s.as_str()),
            _ => None,
        })
        .collect()
}

/// Counts, across the whole file, the positions (0-based line numbers) where the hunk's
/// context+remove block can match.
/// Used to detect "multiple matches" ambiguity, avoiding silently editing the wrong place.
pub(crate) fn all_hunk_match_positions(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    mode: MatchMode,
) -> Vec<usize> {
    let expected = hunk_expected_lines(hunk);
    if expected.is_empty() {
        return Vec::new();
    }
    let mut positions = Vec::new();
    let mut candidate = 0usize;
    while candidate + expected.len() <= orig_lines.len() {
        let all_match = expected
            .iter()
            .enumerate()
            .all(|(i, exp)| lines_match(&orig_lines[candidate + i], exp, mode));
        if all_match {
            positions.push(candidate);
        }
        candidate += 1;
    }
    positions
}
