use super::*;
pub(crate) fn format_char_with_code_point(ch: char) -> String {
    format!("{ch:?} (U+{:04X})", ch as u32)
}

/// Describes the position and Unicode code point of the first differing character
/// Formats a char for mismatch diagnostics, rendering invisible characters
/// (space, tab) visibly so the difference cannot be collapsed by a display layer
/// or mistaken for an empty prefix in the model's view of the two lines.
pub(crate) fn format_char_visible(ch: char) -> String {
    match ch {
        ' ' => "'␣' (U+0020)".to_string(),
        '\t' => "'\\t' (U+0009)".to_string(),
        _ => format_char_with_code_point(ch),
    }
}

/// Finds the first character difference between `expected` and `actual`.
///
/// Returns `(1-based column, expected char, actual char)`; an `Option::None`
/// char means that side of the line ended first (e.g. `(Some(exp), None)` means
/// the actual line already ended where the expected line still has a char).
pub(crate) fn first_char_mismatch(
    expected: &str,
    actual: &str,
) -> Option<(usize, Option<char>, Option<char>)> {
    let mut column = 1usize;
    let mut expected_chars = expected.chars();
    let mut actual_chars = actual.chars();

    loop {
        match (expected_chars.next(), actual_chars.next()) {
            (Some(exp), Some(act)) if exp == act => {
                column += 1;
            }
            (Some(exp), Some(act)) => return Some((column, Some(exp), Some(act))),
            (Some(exp), None) => return Some((column, Some(exp), None)),
            (None, Some(act)) => return Some((column, None, Some(act))),
            (None, None) => return None,
        }
    }
}

/// between two lines of text, making it easy to spot "looks-alike" differences
/// such as smart quotes or full/half-width characters.
pub(crate) fn describe_first_char_mismatch(expected: &str, actual: &str) -> Option<String> {
    first_char_mismatch(expected, actual).map(|(column, exp, act)| match (exp, act) {
        (Some(exp), Some(act)) => format!(
            "column {column}: expected {}, found {}",
            format_char_visible(exp),
            format_char_visible(act)
        ),
        (Some(exp), None) => format!(
            "column {column}: expected {}, found end of line",
            format_char_visible(exp)
        ),
        (None, Some(act)) => format!(
            "column {column}: expected end of line, found {}",
            format_char_visible(act)
        ),
        (None, None) => unreachable!("first_char_mismatch returns None for identical lines"),
    })
}

/// Renders a caret line aligned under the first differing character of `expected`
/// (1-based `column`).
///
/// The padding replicates the `{:?}` debug rendering used for the full mismatch
/// line, so the caret points at the exact character even when the two lines look
/// identical at first glance (e.g. a space where the file has `<`). The found
/// line needs no separate caret: both lines are identical up to that column.
pub(crate) fn render_mismatch_caret(expected: &str, column: usize) -> String {
    let prefix: String = expected.chars().take(column.saturating_sub(1)).collect();
    let prefix_width = format!("{prefix:?}").chars().count();
    format!("{}^", " ".repeat(prefix_width))
}

pub(crate) fn describe_aligned_block_first_mismatch(
    expected_lines: &[&str],
    actual_lines: &[String],
    start: usize,
) -> Option<String> {
    for (offset, expected) in expected_lines.iter().enumerate() {
        let actual = actual_lines
            .get(start + offset)
            .map(String::as_str)
            .unwrap_or("");
        if expected == &actual {
            continue;
        }
        let detail = describe_first_char_mismatch(expected, actual)?;
        let line_no = start + offset + 1;
        return Some(format!(
            "First differing char near declared position is on line {} at {}.\n",
            line_no, detail
        ));
    }
    None
}

pub(crate) const PATCH_TEXT_BLOCK_START: &str =
    "Current file text at this location (copy verbatim, no line-number prefix):\n<<<PATCH_TEXT\n";

/// Renders a directly pasteable block of the current file text (0-based `start`,
/// `count` lines).
///
/// Unlike the other diagnostics in the error message (which carry a `<line>:` prefix
/// for human eyes), this block carries **no line-number prefix** and is meant to be
/// copied verbatim by the model into the new patch's context/removed lines — this
/// eliminates at the root the frequent error of copying read_file's `<number>\t`
/// line-number column into the patch. The `<<<PATCH_TEXT` / `PATCH_TEXT>>>` markers
/// clearly delimit the copyable region.
pub(crate) fn render_pasteable_current_block(orig_lines: &[String], start: usize, count: usize) -> String {
    let end = start.saturating_add(count).min(orig_lines.len());
    if start >= end {
        return String::new();
    }
    let mut out = String::from(PATCH_TEXT_BLOCK_START);
    for line in &orig_lines[start..end] {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str("PATCH_TEXT>>>\n");
    out
}

/// Constructs a "hunks out of order" error with diagnostics.
///
/// Previously this error returned only the bare string `"hunks out of order"` — no
/// line numbers, no indication of which hunk, no current text, no fix suggestion, so
/// the model could only guess blindly (in real sessions this caused 4 consecutive
/// failures).
///
/// Unified diffs require hunks to be ordered by **ascending** file line number;
/// `apply_unified_patch` locates each hunk with a monotonically increasing `cursor`.
/// When the unique match position of a hunk falls before `cursor` (the end line of
/// the previous hunk), the model wrote the hunks out of order (or overlapping).
/// This explains the cause, gives the matched position of this hunk vs the earliest
/// allowed position, appends pasteable current text, and clearly suggests reordering
/// by line number or switching to separate `*** Replace in line:` sections.
///
/// `matched_pos` is the 0-based line this hunk actually matched (if known); `cursor`
/// is the earliest currently allowed 0-based start.
pub(crate) fn describe_hunks_out_of_order(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    matched_pos: Option<usize>,
    cursor: usize,
    hunk_idx: usize,
    hunk_total: usize,
) -> String {
    let mut msg = format!(
        "Hunk {}/{}: hunks out of order: this hunk matches a location earlier in the file \
         than a previous hunk in the same section. Unified-diff hunks must be ordered by \
         ascending file line number. Reorder the hunks by their position in the file \
         (top to bottom), or split unrelated edits into separate `*** Replace in line:` \
         sections.\n",
        hunk_idx + 1,
        hunk_total
    );
    if hunk.old_start > 0 {
        msg.push_str(&format!(
            "This hunk declared @@ -{} (1-based line {}).\n",
            hunk.old_start, hunk.old_start
        ));
    }
    match matched_pos {
        Some(pos) => msg.push_str(&format!(
            "It matches at 1-based line {}, but the previous hunk already consumed through 1-based line {}; a following hunk must start at 1-based line {} or later.\n",
            pos + 1,
            cursor,
            cursor + 1
        )),
        None => msg.push_str(&format!(
            "The earliest position a following hunk may target is 1-based line {} (where the previous hunk ended).\n",
            cursor + 1
        )),
    }
    // Append pasteable current text at this hunk's expected block to help the model
    // reorder/rebuild accordingly.
    let anchor = matched_pos.unwrap_or_else(|| hunk.old_start.saturating_sub(1));
    let expected_len = hunk_expected_lines(hunk).len().max(1);
    let block = render_pasteable_current_block(orig_lines, anchor, expected_len);
    if !block.is_empty() {
        msg.push_str(&block);
    }
    msg
}

/// File lines that share the most text with any expected hunk line, used only to build a
/// diagnostic message. The nominal window printed by `describe_context_mismatch` is derived
/// from the declared line numbers, so it can point at unrelated content when the patch's
/// expected text is a truncated or edited copy of the real line. Scoring by shared text
/// (longest common prefix, or containment of one line in the other) surfaces the line the
/// patch most likely meant.
pub(crate) fn closest_file_lines<'a>(
    expected: &[&str],
    orig_lines: &'a [String],
    limit: usize,
) -> Vec<(usize, &'a str)> {
    const MIN_SHARED: usize = 12;
    let needles: Vec<&str> = expected
        .iter()
        .map(|line| line.trim())
        .filter(|line| line.len() >= MIN_SHARED)
        .collect();
    if needles.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(usize, usize)> = Vec::new();
    for (index, line) in orig_lines.iter().enumerate() {
        let hay = line.trim();
        if hay.len() < MIN_SHARED {
            continue;
        }
        let mut score = 0;
        for needle in &needles {
            score = score.max(shared_prefix_len(needle, hay));
            if needle.contains(hay) {
                score = score.max(hay.len());
            }
            if hay.contains(*needle) {
                score = score.max(needle.len());
            }
        }
        if score >= MIN_SHARED {
            scored.push((score, index));
        }
    }
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    scored
        .into_iter()
        .take(limit)
        .map(|(_, index)| (index + 1, orig_lines[index].trim()))
        .collect()
}

/// Number of leading bytes shared by two lines, used to rank diagnostic candidates.
pub(crate) fn shared_prefix_len(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

/// Constructs a "context mismatch" error with context: lists the lines the patch
/// expected to match plus the actual lines in the original file near the nominal
/// position, so the model can quickly self-correct instead of only seeing a bare
/// "context mismatch".
pub(crate) fn describe_context_mismatch(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    hunk_idx: usize,
    hunk_total: usize,
) -> String {
    let expected = hunk_expected_lines(hunk);
    let nominal = hunk.old_start.saturating_sub(1);

    let mut msg = format!(
        "Hunk {}/{}: context mismatch: patch hunk could not be located. Rebuild the patch \
         from the current file text shown below (re-read the file only if the shown context \
         is not enough).\n",
        hunk_idx + 1,
        hunk_total
    );
    // A hunk that cannot be located is often a duplicate of a patch that already
    // succeeded: every line the hunk would add already exists and none of the lines
    // it would remove remain. Name that case explicitly so the model verifies the
    // working tree instead of rebuilding and re-issuing the same patch in a loop.
    let adds: Vec<&str> = hunk
        .lines
        .iter()
        .filter_map(|line| match line {
            UnifiedLine::Add(text) => Some(text.trim()),
            _ => None,
        })
        .collect();
    let removes_exist = hunk.lines.iter().any(|line| match line {
        UnifiedLine::Remove(text) => {
            let trimmed = text.trim();
            orig_lines.iter().any(|orig| orig.trim() == trimmed)
        }
        _ => false,
    });
    if !adds.is_empty()
        && !removes_exist
        && adds
            .iter()
            .all(|add| orig_lines.iter().any(|orig| orig.trim() == *add))
    {
        msg.push_str(
            "Note: the lines this hunk adds already exist in the file and the lines it \
             removes do not. The change may already be applied: check the working tree \
             (e.g. `git diff`) before re-issuing the same patch; an identical patch will \
             keep failing.\n",
        );
    }
    if hunk.old_start == 0 {
        msg.push_str(
            "Hunk header declared no line number (bare `@@`); the hunk is located by full-file \
             context search.\n",
        );
    } else {
        msg.push_str(&format!(
            "Hunk header declared @@ -{} (1-based line {}).\n",
            hunk.old_start, hunk.old_start
        ));
    }

    // First try a best-effort partial match to pinpoint the mismatched lines. In
    // large replacements the most common failure is that only a few lines in the
    // block are not reproduced exactly; a partial match can tell the model "line X
    // expected A but is B".
    if let Some(best) = find_best_partial_match(orig_lines, hunk, MatchMode::IgnoreIndent) {
        msg.push_str(&format!(
            "Best partial match at line {} ({}/{} lines matched).\n",
            best.pos + 1,
            best.matches,
            best.total
        ));
        if best.mismatches.is_empty() {
            msg.push_str(
                "All expected lines matched at this position — \
                 the mismatch may be due to hunk ordering or a missing trailing line.\n",
            );
        } else {
            // Show the first 10 mismatched lines: the full offset pattern matters
            // more than a condensed word count for the model to fix the patch.
            let show = best.mismatches.len().min(10);
            msg.push_str(&format!(
                "Mismatched lines (showing {} of {}):\n",
                show,
                best.mismatches.len()
            ));
            for (file_line, exp, act) in best.mismatches.iter().take(show) {
                let first_diff = describe_first_char_mismatch(exp, act)
                    .map(|detail| format!("; first differing char at {detail}"))
                    .unwrap_or_default();
                msg.push_str(&format!(
                    "  line {}: expected {:?}, found {:?}{}\n",
                    file_line, exp, act, first_diff
                ));
                // Caret line under the first differing char. This is the one
                // piece of the diagnostic that survives display mangling of the
                // line content itself: it marks the exact position so the model
                // can fix the byte even when expected/found look identical.
                if let Some((column, _, _)) = first_char_mismatch(exp, act) {
                    let label_width = format!("  line {}: expected ", file_line).chars().count();
                    msg.push_str(&format!(
                        "{}{}\n",
                        " ".repeat(label_width),
                        render_mismatch_caret(exp, column)
                    ));
                }
            }
            if best.mismatches.len() > show {
                msg.push_str(&format!(
                    "  ... ({} more mismatches)\n",
                    best.mismatches.len() - show
                ));
            }
        }
        // A directly pasteable block of current text: covers the best-match region
        // (with a little extra margin before/after) so the model can rebuild the
        // patch in place without re-reading the whole file.
        let block =
            render_pasteable_current_block(orig_lines, best.pos, best.total.max(expected.len()));
        if !block.is_empty() {
            msg.push_str(&block);
        }
    } else {
        // No partial match found in the file — the block does not exist at all.
        // Echo the expected lines and the actual content near the nominal position.
        let candidates = closest_file_lines(&expected, orig_lines, 3);
        if !candidates.is_empty() {
            msg.push_str("Closest actual lines by shared text:\n");
            for (line_no, line) in candidates {
                msg.push_str(&format!("  line {line_no}: {line}\n"));
            }
            msg.push_str(
                "If one of these is the intended target, copy its text verbatim from the file; \
                 the expected line above may be truncated or edited.\n",
            );
        }
        msg.push_str("Patch expected these lines (context/removed):\n");
        for (i, line) in expected.iter().take(10).enumerate() {
            msg.push_str(&format!("  expected[{}]: {}\n", i, line));
        }
        if expected.len() > 10 {
            msg.push_str(&format!(
                "  ... ({} more expected lines)\n",
                expected.len() - 10
            ));
        }
        let win_start = nominal.saturating_sub(3);
        let win_end = (nominal + expected.len().max(1) + 3).min(orig_lines.len());
        if win_start < win_end {
            msg.push_str(&format!(
                "Actual file content around line {} (1-based):\n",
                win_start + 1
            ));
            for (offset, line) in orig_lines[win_start..win_end].iter().enumerate() {
                msg.push_str(&format!("  {:>6}: {}\n", win_start + offset + 1, line));
            }
            // Also append a pasteable block with no line-number prefix.
            let block = render_pasteable_current_block(orig_lines, win_start, win_end - win_start);
            if !block.is_empty() {
                msg.push_str(&block);
            }
        } else {
            msg.push_str(&format!(
                "File has {} line(s); declared position is out of range.\n",
                orig_lines.len()
            ));
        }
        if let Some(detail) = describe_aligned_block_first_mismatch(&expected, orig_lines, nominal)
        {
            msg.push_str(&detail);
        }
    }

    msg.push_str(
        "Hint: rebuild the patch from the copy-verbatim block above (the text between <<<PATCH_TEXT and PATCH_TEXT>>>), which is the exact current file content with NO line-number prefix. For a small change, a `*** Replace in line:` section (anchor/old/new) is the most reliable. If the shown context is insufficient, re-read the file with read_file(use_line_numbers=false) to get raw content without line-number prefixes, then copy the exact text.",
    );
    msg
}
