use super::*;
pub(crate) fn describe_ambiguous_hunk(
    orig_lines: &[String],
    positions: &[usize],
    hunk_idx: usize,
    hunk_total: usize,
) -> String {
    let shown: Vec<String> = positions
        .iter()
        .take(8)
        .map(|pos| (pos + 1).to_string())
        .collect();
    let mut msg = format!(
        "Hunk {}/{}: ambiguous patch: hunk context matched {} locations (1-based lines: {}{}). \
         Add more unique surrounding context, preferably both before and after the edit, \
         or split the edit around a uniquely matching removed line.\n",
        hunk_idx + 1,
        hunk_total,
        positions.len(),
        shown.join(", "),
        if positions.len() > 8 { ", ..." } else { "" }
    );
    // Echo the current first line at each candidate position, so the model can pick the right
    // anchor and add more unique context without a separate read_file.
    msg.push_str("Candidate locations (current first line at each):\n");
    for &pos in positions.iter().take(8) {
        if let Some(line) = orig_lines.get(pos) {
            msg.push_str(&format!("  line {}: {:?}\n", pos + 1, line));
        }
    }
    msg.push_str(
        "Hint: the first line at each candidate is shown above; add more unique surrounding \
         context (e.g. the preceding function signature or comment) around the intended \
         location. For a single-line change, `*** Replace in line:` (anchor/old/new) is the \
         most reliable. If the candidates are structurally similar blocks (e.g. repeated \
         closures), apply one patch per block instead of one multi-hunk patch.\n",
    );
    msg
}

pub(crate) const DECLARED_LINE_DISAMBIGUATION_MAX_DRIFT: usize = 12;

pub(crate) fn disambiguate_by_declared_line(positions: &[usize], hunk: &UnifiedHunk) -> Option<usize> {
    if hunk.old_start == 0 || positions.len() < 2 {
        return None;
    }
    let nominal = hunk.old_start.saturating_sub(1);
    let mut scored: Vec<(usize, usize)> = positions
        .iter()
        .map(|&pos| (pos.abs_diff(nominal), pos))
        .collect();
    scored.sort_unstable();
    let (best_dist, best_pos) = scored[0];
    let (second_dist, _) = scored[1];
    if best_dist <= DECLARED_LINE_DISAMBIGUATION_MAX_DRIFT
        && best_dist.saturating_mul(2) < second_dist
    {
        Some(best_pos)
    } else {
        None
    }
}

pub(crate) fn hunk_old_line_count(hunk: &UnifiedHunk) -> usize {
    hunk.lines
        .iter()
        .filter(|line| matches!(line, UnifiedLine::Context(_) | UnifiedLine::Remove(_)))
        .count()
}

pub(crate) fn hunk_remove_offsets(hunk: &UnifiedHunk) -> Vec<(usize, &str)> {
    let mut old_offset = 0usize;
    let mut offsets = Vec::new();
    for line in &hunk.lines {
        match line {
            UnifiedLine::Context(_) => old_offset += 1,
            UnifiedLine::Remove(s) => {
                offsets.push((old_offset, s.as_str()));
                old_offset += 1;
            }
            UnifiedLine::Add(_) => {}
        }
    }
    offsets
}

pub(crate) fn remove_lines_match_at(
    orig_lines: &[String],
    remove_offsets: &[(usize, &str)],
    start: usize,
    mode: MatchMode,
) -> bool {
    remove_offsets.iter().all(|(offset, expected)| {
        orig_lines
            .get(start + offset)
            .is_some_and(|actual| lines_match(actual, expected, mode))
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FuzzyContextMatch {
    pub(super) pos: usize,
    pub(super) context_matches: usize,
    pub(super) context_total: usize,
}

pub(crate) fn score_context_matches(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    start: usize,
    mode: MatchMode,
) -> (usize, usize) {
    let mut old_offset = 0usize;
    let mut matches = 0usize;
    let mut total = 0usize;
    for line in &hunk.lines {
        match line {
            UnifiedLine::Context(expected) => {
                total += 1;
                if orig_lines
                    .get(start + old_offset)
                    .is_some_and(|actual| lines_match(actual, expected, mode))
                {
                    matches += 1;
                }
                old_offset += 1;
            }
            UnifiedLine::Remove(_) => old_offset += 1,
            UnifiedLine::Add(_) => {}
        }
    }
    (matches, total)
}

pub(crate) fn fuzzy_context_candidates(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    cursor: usize,
    mode: MatchMode,
) -> Vec<FuzzyContextMatch> {
    let old_len = hunk_old_line_count(hunk);
    let remove_offsets = hunk_remove_offsets(hunk);
    if old_len == 0 || remove_offsets.is_empty() || old_len > orig_lines.len() {
        return Vec::new();
    }

    let (first_remove_offset, first_remove) = remove_offsets[0];
    let Some(first_scan_line) = cursor.checked_add(first_remove_offset) else {
        return Vec::new();
    };
    if first_scan_line >= orig_lines.len() {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    for file_line in first_scan_line..orig_lines.len() {
        if !lines_match(&orig_lines[file_line], first_remove, mode) {
            continue;
        }
        let Some(start) = file_line.checked_sub(first_remove_offset) else {
            continue;
        };
        if start < cursor || start + old_len > orig_lines.len() {
            continue;
        }
        if !remove_lines_match_at(orig_lines, &remove_offsets, start, mode) {
            continue;
        }
        let (context_matches, context_total) = score_context_matches(orig_lines, hunk, start, mode);
        candidates.push(FuzzyContextMatch {
            pos: start,
            context_matches,
            context_total,
        });
    }

    candidates.sort_by_key(|candidate| candidate.pos);
    candidates.dedup_by_key(|candidate| candidate.pos);
    candidates
}

/// Context lines are a locating aid and should not cause a hard failure once the
/// remove lines are precisely anchored. But to avoid mislocating common remove
/// lines (such as `}`), fuzz application is allowed only when the candidate is
/// unique or the remaining context can be scored uniquely.
pub(crate) fn locate_hunk_with_fuzzy_context(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    cursor: usize,
    mode: MatchMode,
) -> Result<Option<FuzzyContextMatch>, String> {
    let candidates = fuzzy_context_candidates(orig_lines, hunk, cursor, mode);
    if candidates.is_empty() {
        return Ok(None);
    }
    if candidates.len() == 1 {
        return Ok(candidates.first().copied());
    }

    let best_score = candidates
        .iter()
        .map(|candidate| candidate.context_matches)
        .max()
        .unwrap_or(0);
    let best: Vec<FuzzyContextMatch> = candidates
        .iter()
        .copied()
        .filter(|candidate| candidate.context_matches == best_score)
        .collect();
    if best.len() == 1 && best_score > 0 {
        return Ok(best.first().copied());
    }

    let nominal = hunk.old_start.saturating_sub(1);
    // Use old_start as a disambiguation signal: as long as the nominal
    // candidate's context score is close to the best (difference ≤ 1), trust the
    // line number the model annotated. Accept it also when best_score == 0 (the
    // context lines cannot distinguish candidate positions at all) — then
    // old_start is the only usable locating signal, and rejecting would just make
    // the model retry the same generic context endlessly.
    if hunk.old_start > 0 && nominal < orig_lines.len() {
        if let Some(nominal_candidate) = candidates.iter().find(|c| c.pos == nominal) {
            if best_score == 0 || nominal_candidate.context_matches + 1 >= best_score {
                return Ok(Some(*nominal_candidate));
            }
        }
    }

    let shown: Vec<String> = candidates
        .iter()
        .take(5)
        .map(|candidate| {
            format!(
                "{} (context {}/{})",
                candidate.pos + 1,
                candidate.context_matches,
                candidate.context_total
            )
        })
        .collect();
    Err(format!(
        "ambiguous patch: remove lines match {} locations under context-fuzz mode (1-based lines: {}{}). \
         Include more exact surrounding context (both before and after the edit), or split the hunk around a more unique removed line. A `*** Replace in line:` section with a unique anchor also avoids this.",
        candidates.len(),
        shown.join(", "),
        if candidates.len() > 5 { ", ..." } else { "" }
    ))
}

/// For large replacements (a hunk with many context/remove lines), all-or-nothing
/// exact matching easily fails entirely when a few lines are not reproduced
/// exactly. Here we first run a best-effort partial-match scan: find the start
/// with the most matching lines across the whole file, and report precisely which
/// lines differ (expected vs actual), so the model only needs to fix the few
/// wrong lines instead of re-guessing the whole block.
pub(crate) struct BestPartialMatch {
    /// Best matching start (0-based)
    pub(super) pos: usize,
    /// Number of matching lines
    pub(super) matches: usize,
    /// Total number of lines checked
    pub(super) total: usize,
    /// Mismatched lines: (1-based file line, expected content, actual content)
    pub(super) mismatches: Vec<(usize, String, String)>,
}

/// Finds the start position where the hunk's expected block matches best across
/// the whole file. Called only on the error path after exact matching failed,
/// using IgnoreIndent mode to tolerate indentation differences and focus on
/// content differences. Returns None when no line in the file can match the
/// expected block — the block does not exist at all.
pub(crate) fn find_best_partial_match(
    orig_lines: &[String],
    hunk: &UnifiedHunk,
    mode: MatchMode,
) -> Option<BestPartialMatch> {
    let expected = hunk_expected_lines(hunk);
    if expected.is_empty() || orig_lines.is_empty() {
        return None;
    }

    // Use the first line for a quick filter: only run the full alignment check at
    // candidate positions where the first line matches, avoiding an O(N*M) full
    // scan on large files. In large replacements the most common failure is a
    // correct first line with a few wrong lines after it.
    let mut candidates: Vec<usize> = (0..orig_lines.len())
        .filter(|&i| lines_match(&orig_lines[i], expected[0], mode))
        .collect();

    // When the first line does not match, use the last line as an anchor: a last
    // line matching at position i corresponds to start i - (len-1).
    if candidates.is_empty() && expected.len() > 1 {
        let last = expected.len() - 1;
        candidates = (last..orig_lines.len())
            .filter(|&i| lines_match(&orig_lines[i], expected[last], mode))
            .map(|i| i - last)
            .collect();
    }

    // When neither the first nor the last line matches, anchor on every expected
    // line and take the candidate with the most matching lines. This is the final
    // fallback, covering cases where the middle lines of the expected block are
    // correct but the first/last lines are wrong.
    if candidates.is_empty() {
        for (ei, exp) in expected.iter().enumerate() {
            for (fi, line) in orig_lines.iter().enumerate() {
                if lines_match(line, exp, mode) {
                    let start = fi.saturating_sub(ei);
                    if start < orig_lines.len() {
                        candidates.push(start);
                    }
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
    }

    // Limit the number of candidates to avoid performance issues in extreme cases.
    candidates.truncate(500);

    let mut best: Option<BestPartialMatch> = None;
    for &start in &candidates {
        let available = orig_lines.len().saturating_sub(start);
        let check_count = expected.len().min(available);
        if check_count == 0 {
            continue;
        }
        let mut matches = 0usize;
        let mut mismatches = Vec::new();
        for i in 0..check_count {
            let act = &orig_lines[start + i];
            if lines_match(act, expected[i], mode) {
                matches += 1;
            } else {
                mismatches.push((start + i + 1, expected[i].to_string(), act.clone()));
            }
        }
        let is_better = match &best {
            None => true,
            Some(b) => matches > b.matches,
        };
        if is_better {
            best = Some(BestPartialMatch {
                pos: start,
                matches,
                total: check_count,
                mismatches,
            });
        }
        // A perfect match should not occur on the error path, but keep the early
        // exit for safety.
        if matches == expected.len() {
            break;
        }
    }
    best.filter(|b| b.matches > 0)
}
