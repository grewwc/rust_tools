use super::*;
/// The three single-line fields accepted in a `*** Replace in line:` body.
#[derive(Clone, Copy)]
pub(crate) enum InlineReplaceField {
    Anchor,
    Old,
    New,
}

impl InlineReplaceField {
    /// Field order, so a body line can index its slot directly.
    const ALL: [Self; 3] = [Self::Anchor, Self::Old, Self::New];

    fn index(self) -> usize {
        match self {
            Self::Anchor => 0,
            Self::Old => 1,
            Self::New => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Anchor => "anchor",
            Self::Old => "old",
            Self::New => "new",
        }
    }

    /// Parse one body line as `<name>: <value>`, or `None` when the line is not one of the three
    /// fields. Leading indentation before the name is formatting only and is ignored; exactly one
    /// space after the colon is the separator, so any further leading spaces stay in the value and
    /// a bare `name:` denotes an empty value.
    fn parse(line: &str) -> Option<(Self, &str)> {
        let line = line.trim_start();
        Self::ALL.iter().find_map(|field| {
            let rest = line.strip_prefix(field.name())?.strip_prefix(':')?;
            Some((*field, rest.strip_prefix(' ').unwrap_or(rest)))
        })
    }
}

/// Render a body line for an error message, capped so an over-long line cannot bloat the tool
/// result.
pub(crate) fn preview_patch_body_line(line: &str) -> String {
    const MAX_CHARS: usize = 80;
    if line.chars().count() <= MAX_CHARS {
        return format!("{line:?}");
    }
    let head: String = line.chars().take(MAX_CHARS).collect();
    format!("{head:?}...")
}

/// Inline substring replacement: use `anchor:` to locate the line, then exactly replace
/// `old:` with `new:` within that line.
///
/// Designed for the most common editing scenario — "change a few words inside one long
/// single-line string" — avoiding a full-line rewrite.
///
/// Safety design (to rule out "executed successfully but replaced the wrong position"):
/// - `anchor` locates the line via normalized substring matching (confusable-tolerant), but is
///   **only used for locating**;
/// - `old` uses **exact** substring matching (no normalization) to determine the byte range to
///   replace, ruling out positional drift;
/// - `anchor` must uniquely match one line, otherwise error (to avoid changing the wrong place);
/// - `old` must appear exactly once in that line, otherwise error (to avoid changing the wrong
///   position);
/// - if `old == new` (identical before/after), report an error as a no-op so it isn't mistaken
///   for success;
/// - the body must be three single-line fields plus optional blank lines: unrecognized non-blank
///   and repeated lines are rejected instead of ignored, because silently dropping them would
///   apply a partial edit (for example the first line of a multi-line `new:`) while still
///   reporting success.
pub(crate) fn apply_inline_replace(original: &str, envelope: &PatchEnvelope) -> Result<String, String> {
    // --- Parse the three fields: anchor / old / new ---
    let mut fields: [Option<String>; 3] = [None, None, None];
    for line in &envelope.body_lines {
        // Blank lines carry no content and can never express a value, so skipping them cannot hide
        // model-authored text — unlike an unrecognized non-blank line, which may be a dropped
        // continuation of a multi-line value.
        if line.trim().is_empty() {
            continue;
        }
        let Some((field, value)) = InlineReplaceField::parse(line) else {
            return Err(format!(
                "Replace in line: unexpected line {} — a `*** Replace in line:` section takes \
                 exactly three single-line fields (`anchor: <unique substring of the target \
                 line>`, `old: <substring to replace>`, `new: <replacement>`); `new:` cannot \
                 span multiple lines. To insert or replace whole lines, use a `*** Update File:` \
                 hunk with `@@` instead (or `write_file` for a full rewrite).",
                preview_patch_body_line(line)
            ));
        };
        if fields[field.index()].replace(value.to_string()).is_some() {
            return Err(format!(
                "Replace in line: duplicate `{}:` field — each of `anchor:`, `old:`, `new:` may \
                 appear exactly once.",
                field.name()
            ));
        }
    }
    let [anchor, old, new] = fields;
    let anchor = anchor.ok_or_else(|| {
        "Replace in line: missing `anchor:` field. \
         Expected `anchor: <unique substring of target line>`."
            .to_string()
    })?;
    if anchor.is_empty() {
        return Err("Replace in line: `anchor` field must not be empty.".to_string());
    }
    let old = old.ok_or_else(|| {
        "Replace in line: missing `old:` field. \
         Expected `old: <exact substring to replace>`."
            .to_string()
    })?;
    let new = new.ok_or_else(|| {
        "Replace in line: missing `new:` field. \
         Expected `new: <replacement substring>`."
            .to_string()
    })?;
    if old.is_empty() {
        return Err("Replace in line: `old` field must not be empty.".to_string());
    }
    if old == new {
        return Err(format!(
            "Replace in line: `old` and `new` are identical ({:?}). Nothing would change; fix the \
             patch or remove it. If you meant to insert or replace whole lines, use a \
             `*** Update File:` hunk with `@@` instead.",
            old
        ));
    }

    // --- Locate the line via normalized substring matching (confusable-tolerant), but only for locating ---
    let norm_anchor = normalize_confusables(&anchor);
    let matched_lines: Vec<usize> = original
        .lines()
        .enumerate()
        .filter(|(_, line)| normalize_confusables(line).contains(norm_anchor.as_str()))
        .map(|(i, _)| i)
        .collect();

    let line_idx = match matched_lines.len() {
        0 => {
            return Err(format!(
                "Replace in line: anchor not found. \
                 No line contains {:?} (after Unicode normalization).",
                anchor
            ));
        }
        1 => matched_lines[0],
        n => {
            let positions = matched_lines
                .iter()
                .map(|i| format!("{}", i + 1))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "Replace in line: anchor matched {n} lines (1-based: {positions}). \
                 Anchor must uniquely identify one line. Make `anchor` more specific."
            ));
        }
    };

    let original_lines: Vec<&str> = original.lines().collect();
    let target_line = original_lines[line_idx];

    // --- First use exact substring matching (no normalization) to determine the replacement position, ruling out positional drift ---
    let occurrences: Vec<usize> = target_line.match_indices(&old).map(|(i, _)| i).collect();
    let (pos, match_len) = match occurrences.len() {
        1 => (occurrences[0], old.len()),
        0 => {
            // Tolerant fallback when exact matching fails: confusable normalization + leading and
            // trailing whitespace tolerance.
            // Only used for locating; the replacement boundary follows the matched original byte
            // range, and the written content is constructed from `new`.
            match find_tolerant_old_match(target_line, &old) {
                Ok((start, end)) => (start, end - start),
                Err(TolerantMatchError::Ambiguous(n)) => {
                    return Err(format!(
                        "Replace in line: after Unicode normalization `old` matches {n} \
                         positions in line {}. It must be unique within the line. \
                         Make `old` longer or more specific. Line content: {:?}",
                        line_idx + 1,
                        target_line
                    ));
                }
                Err(TolerantMatchError::NoMatch) => {
                    return Err(format!(
                        "Replace in line: `old` substring not found in matched line {} \
                         (even with Unicode/whitespace tolerance). Line content: {:?}\n\
                         Tips: copy `old` from the actual file, not from memory. If you \
                         need fresh source text, re-read with `read_file` \
                         (use_line_numbers=false) so the output has no line-number \
                         prefixes and you can copy the exact line content. Watch for \
                         smart quotes / dashes / non-breaking \
                         spaces that may differ from the file.",
                        line_idx + 1,
                        target_line
                    ));
                }
            }
        }
        n => {
            return Err(format!(
                "Replace in line: `old` substring appears {n} times in line {}. \
                 It must be unique within the line. Make `old` longer to disambiguate. \
                 Line content: {:?}",
                line_idx + 1,
                target_line
            ));
        }
    };

    // Exact replacement: byte range [pos, pos+old.len()).
    // pos is the byte index returned by str::find; old is valid UTF-8, so pos and
    // pos+old.len() both lie on char boundaries and slicing is safe.
    let replaced_line = format!(
        "{}{}{}",
        &target_line[..pos],
        new,
        &target_line[pos + match_len..]
    );

    // Rebuild the file, preserving the original trailing-newline behavior
    let trailing_newline = original.ends_with('\n');
    let mut result = String::with_capacity(original.len() + new.len());
    for (i, line) in original_lines.iter().enumerate() {
        if i == line_idx {
            result.push_str(&replaced_line);
        } else {
            result.push_str(line);
        }
        if i < original_lines.len() - 1 || trailing_newline {
            result.push('\n');
        }
    }
    Ok(result)
}
