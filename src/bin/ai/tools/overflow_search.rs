//! search_overflow — ranked retrieval over the session overflow archive
//!
//! Content moved out of context is archived verbatim into session assets:
//! - `overflow-history.md`: original folded messages (user/assistant/tool results)
//! - `tool-overflow-compressed/`: full snapshots of individual tool results
//! - `folded-tool-groups/`: original messages of wholly folded tool-call groups
//! - `internal-note-overflow/`: internal context notes trimmed by budget
//! - `user-overflow-preserved/`, `image-overflow-preserved/`: kept user turns/images
//!
//! The model usually knows roughly *what* was archived but not the exact path or
//! wording, so retrieval must tolerate vocabulary drift. This tool therefore runs
//! a small ranking engine of its own instead of the shared single-pattern grep:
//!
//! 1. **Term fan-out**: a non-regex query is split into whitespace-separated
//!    terms searched as an OR, plus the full query as a phrase when it has more
//!    than one term. A near-miss on one word no longer produces zero hits.
//! 2. **TF-IDF-flavoured scoring**: rare-in-corpus terms outweigh common ones,
//!    whole-word and path hits get bonuses, and multi-term coverage lifts a
//!    snapshot file above single-term files.
//! 3. **Fair-share visibility**: files are picked globally by relevance, but one
//!    root can only take ~2× its fair share of the result budget while other
//!    roots still demand attention; ties spread round-robin. This replaces the
//!    previous static even quota split between roots and prevents either
//!    starvation mode: an early noisy root monopolising results, or equally a
//!    relevant-but-quota-starved archive section becoming permanently invisible.
//!
//! Results stay verbatim excerpts with absolute paths and line numbers so they
//! can be fed directly into `read_file`.
//!
//! Safety: the search root is never taken from caller input; it is derived only
//! from `current_session_assets_dir()`, and errors out when no driver context is
//! active, making cross-session reads impossible.

use std::fs;
use std::path::{Path, PathBuf};

use regex::{Regex, RegexBuilder};
use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;

use crate::ai::tools::common::{
    ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
    ToolRegistration, ToolSpec,
};
use crate::ai::tools::storage::file_store::current_session_assets_dir;

/// Hard ceiling of matched lines returned (mirrors the shared engine cap).
const MAX_MATCHES: usize = 200;
/// Per-file snippet ceiling: one huge repetitive log cannot occupy the whole
/// answer even when every one of its lines outranks the rest of the archive.
const MAX_SNIPPETS_PER_FILE: usize = 12;
/// While other roots still contribute candidates, a single root may consume at
/// most this multiple of the average share before yielding its turn.
const FAIR_SHARE_MULTIPLE: usize = 2;
/// Fixed weight of an exact whole-query phrase hit versus single-term IDF mass.
const PHRASE_WEIGHT: f64 = 6.0;
/// Rendered output size guard (archive files can be arbitrarily large).
const MAX_OUTPUT_CHARS: usize = 24_000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SearchScope {
    /// All session archive content.
    All,
    /// Only overflow-history.md (original folded messages).
    History,
    /// Only tool-overflow-compressed/ (per-tool-result snapshots).
    ToolOutputs,
}

impl SearchScope {
    fn parse(raw: &str) -> SearchScope {
        match raw.trim() {
            "history" => SearchScope::History,
            "tool_outputs" => SearchScope::ToolOutputs,
            _ => SearchScope::All, // "all" and unknown values fall back to full scope
        }
    }
}

struct OverflowSearchParams<'a> {
    query: &'a str,
    is_regex: bool,
    /// Tool-entry default keeps the historical strict behavior
    /// (`unwrap_or(true)`, preserved from the pre-ranking implementation);
    /// callers opting into fuzzy recall pass `case_sensitive: false`.
    case_sensitive: bool,
    context_lines: usize,
    max_results: usize,
    file_pattern: Option<&'a str>,
    scope: SearchScope,
}

fn execute_search_overflow(args: &Value) -> Result<String, String> {
    let query = args["query"].as_str().ok_or("Missing 'query' parameter")?;
    if query.trim().is_empty() {
        return Err("query must not be empty".to_string());
    }
    let assets_dir = current_session_assets_dir().ok_or(
        "No active session archive: cannot resolve the current session's overflow directory.",
    )?;

    let params = OverflowSearchParams {
        query,
        is_regex: args["is_regex"].as_bool().unwrap_or(false),
        // Parity with the previous implementation: unspecified means strict
        // case matching. The normalized ranking still honors the flag below.
        case_sensitive: args["case_sensitive"].as_bool().unwrap_or(true),
        context_lines: args["context_lines"].as_u64().unwrap_or(2).min(5) as usize,
        max_results: args["max_results"]
            .as_u64()
            .unwrap_or(50)
            .clamp(1, MAX_MATCHES as u64) as usize,
        file_pattern: args["file_pattern"].as_str(),
        scope: args["scope"]
            .as_str()
            .map(SearchScope::parse)
            .unwrap_or(SearchScope::All),
    };
    run_overflow_search(&assets_dir, &params)
}

// ─── Pattern construction ────────────────────────────────────────────────────

/// One searchable alternative derived from the user query.
struct TermPattern {
    /// Human-readable source of the pattern (for debug and length weighting).
    source: String,
    /// None marks the whole-query phrase; Some(i) is the index among split terms.
    term_id: Option<usize>,
    regex: Regex,
}

fn build_patterns(params: &OverflowSearchParams<'_>) -> Result<Vec<TermPattern>, String> {
    if params.is_regex {
        let regex = compile_pattern(params.query, params.is_regex, params.case_sensitive)?;
        return Ok(vec![TermPattern {
            source: params.query.to_string(),
            term_id: None,
            regex,
        }]);
    }

    let mut patterns: Vec<TermPattern> = Vec::new();
    let mut seen: FxHashMap<&str, ()> = FxHashMap::default();
    for (idx, term) in params.query.split_whitespace().enumerate() {
        if seen.insert(term, ()).is_some() {
            continue;
        }
        let regex = compile_pattern(term, false, params.case_sensitive)?;
        patterns.push(TermPattern {
            source: term.to_string(),
            term_id: Some(idx),
            regex,
        });
    }
    // The untouched query also competes as a phrase; single-term queries would
    // duplicate their own term here, hence the >1 guard.
    if params.query.split_whitespace().count() > 1 {
        let regex = compile_pattern(params.query, false, params.case_sensitive)?;
        patterns.push(TermPattern {
            source: params.query.to_string(),
            term_id: None,
            regex,
        });
    }
    Ok(patterns)
}

fn compile_pattern(source: &str, is_regex: bool, case_sensitive: bool) -> Result<Regex, String> {
    let body = if is_regex {
        source.to_string()
    } else {
        regex::escape(source)
    };
    RegexBuilder::new(&body)
        .case_insensitive(!case_sensitive)
        .build()
        .map_err(|e| {
            if is_regex {
                format!("Invalid regex: {}", e)
            } else {
                format!("Internal regex error: {}", e)
            }
        })
}

/// Minimal glob → regex translation for `file_pattern` ("*", "?", literals).
fn glob_to_regex(glob: &str) -> Regex {
    let mut body = String::from("^");
    for ch in glob.chars() {
        match ch {
            '*' => body.push_str(".*"),
            '?' => body.push('.'),
            c => body.push_str(&regex::escape(&c.to_string())),
        }
    }
    body.push('$');
    Regex::new(&body).expect("glob_to_regex always builds a valid regex")
}

// ─── Scanning & scoring ──────────────────────────────────────────────────────

/// Every matching line inside one file, kept raw; scoring happens corpus-wide.
struct RawHit {
    line_index: usize,
    /// Indices into `patterns` that matched this line (deduped, unordered).
    matched: Vec<usize>,
    /// IDF-independent line score, computed while the line text is in hand
    /// during the scan pass: whole-word (+2.0), lead-proximity, and exact
    /// phrase bonuses. Corpus-wide IDF weights are added in pass B.
    local_score: f64,
}

struct FileScan {
    /// Index into the per-scope roots vec; drives cross-root fair-share logic.
    root_idx: usize,
    /// Absolute path, ready for `read_file` round-trips.
    display_path: String,
    hits: Vec<RawHit>,
    /// Line count of the archive file at scan time. The file text itself is
    /// dropped after scanning and re-read only for the few files that survive
    /// selection, so resident memory scales with match count, not file size.
    total_lines: usize,
}

/// One scanned archive file with its aggregated relevance scores.
struct ScoredFile {
    root_idx: usize,
    scan: FileScan,
    /// Whole-file score: best line score + multi-term coverage bonus + path bonus.
    file_score: f64,
    /// Matched lines sorted by descending score, capped at MAX_SNIPPETS_PER_FILE.
    scored: Vec<(usize, f64)>,
    total_matches: usize,
}

/// Iterates concrete archive files under one root (single file or directory).
fn collect_files(root: &Path) -> Vec<PathBuf> {
    if root.is_file() {
        return vec![root.to_path_buf()];
    }
    let mut out: Vec<PathBuf> = Vec::new();
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    // `path.is_dir()` follows symlinks, so a symlink cycle (a directory
    // symlinked back into one of its ancestors) would otherwise push forever.
    // Canonical paths break the cycle: each physical directory is visited once.
    let mut visited: FxHashSet<PathBuf> = FxHashSet::default();
    while let Some(dir) = stack.pop() {
        let Ok(canon) = fs::canonicalize(&dir) else {
            continue;
        };
        if !visited.insert(canon) {
            continue;
        }
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        let mut names: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        // Deterministic traversal independent of filesystem order.
        names.sort();
        for path in names {
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

fn path_matches_glob(path: &Path, root: &Path, glob: Option<&Regex>) -> bool {
    let Some(glob) = glob else { return true };
    let rel = path.strip_prefix(root).unwrap_or(path);
    glob.is_match(rel.to_string_lossy().as_ref())
        || path
            .file_name()
            .is_some_and(|name| glob.is_match(name.to_string_lossy().as_ref()))
}

fn run_overflow_search(
    assets_dir: &Path,
    params: &OverflowSearchParams<'_>,
) -> Result<String, String> {
    let roots: Vec<PathBuf> = match params.scope {
        SearchScope::History => vec![assets_dir.join("overflow-history.md")],
        SearchScope::ToolOutputs => vec![assets_dir.join("tool-overflow-compressed")],
        SearchScope::All => vec![
            assets_dir.join("overflow-history.md"),
            assets_dir.join("tool-overflow-compressed"),
            assets_dir.join("folded-tool-groups"),
            assets_dir.join("internal-note-overflow"),
            assets_dir.join("user-overflow-preserved"),
            assets_dir.join("image-overflow-preserved"),
        ],
    };
    let roots: Vec<(usize, PathBuf)> = roots
        .into_iter()
        .filter(|root| root.exists())
        .enumerate()
        .collect();
    if roots.is_empty() {
        return Ok(format!(
            "No matches found in the session archive for query: '{}'",
            params.query
        ));
    }

    let patterns = build_patterns(params)?;

    // Pass A: scan every file once, collecting raw hits plus document
    // frequencies feeding the IDF weights.
    let mut scans: Vec<FileScan> = Vec::new();
    let mut df: Vec<usize> = vec![0; patterns.len()];
    let mut files_seen: usize = 0;

    // Compiled once: the glob is fixed for the whole search. A pattern that
    // trims to empty is treated as "no filter" — an empty glob would only match
    // a bare relative path "" (the History scope's file root) and silently
    // exclude every file under a directory root.
    let glob = params
        .file_pattern
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(glob_to_regex);

    for (root_idx, root) in &roots {
        for file in collect_files(root) {
            if !path_matches_glob(&file, root, glob.as_ref()) {
                continue;
            }
            let Ok(content) = fs::read_to_string(&file) else {
                // Unreadable or non-UTF-8 files cannot contain matches; skip
                // without counting them toward the corpus size used by IDF.
                continue;
            };
            files_seen += 1;

            let mut hits: Vec<RawHit> = Vec::new();
            let mut file_term_hits: Vec<bool> = vec![false; patterns.len()];
            let mut total_lines = 0usize;
            for (line_index, line) in content.lines().enumerate() {
                total_lines = line_index + 1;
                let mut matched: Vec<usize> = Vec::new();
                for (pid, pattern) in patterns.iter().enumerate() {
                    if pattern.regex.is_match(line) {
                        matched.push(pid);
                        file_term_hits[pid] = true;
                    }
                }
                if !matched.is_empty() {
                    // Line-local scoring happens here while the text is
                    // available; corpus-wide IDF weights are added in pass B.
                    let mut local_score = 0.0;
                    for &pid in &matched {
                        // Whole-word hits carry more information than substring hits.
                        let re = &patterns[pid].regex;
                        if let Some(m) = re.find(line) {
                            let bytes = line.as_bytes();
                            let left_ok =
                                m.start() == 0 || !is_identifier_byte(bytes[m.start() - 1]);
                            let right_ok =
                                m.end() >= line.len() || !is_identifier_byte(bytes[m.end()]);
                            if left_ok && right_ok {
                                local_score += 2.0;
                            }
                            // Lead-proximity bonus, mirroring the shared engine style.
                            let lead_chars = line[..m.start()].chars().count();
                            local_score += 2.0 * (1.0 - (lead_chars.min(40) as f64) / 40.0);
                        }
                    }
                    if line_has_exact_phrase_bonus(&patterns, &matched) {
                        // Exact whole-query phrase/regex hit.
                        local_score += PHRASE_WEIGHT;
                    }
                    hits.push(RawHit {
                        line_index,
                        matched,
                        local_score,
                    });
                }
            }
            if hits.is_empty() {
                continue;
            }
            for (pid, hit_all) in file_term_hits.into_iter().enumerate() {
                if hit_all {
                    df[pid] += 1;
                }
            }
            scans.push(FileScan {
                root_idx: *root_idx,
                display_path: file.to_string_lossy().to_string(),
                hits,
                total_lines,
            });
        }
    }

    if scans.is_empty() {
        return Ok(format!(
            "No matches found in the session archive for query: '{}'",
            params.query
        ));
    }

    // IDF over scanned files; +guards keep every weight finite and positive so
    // single-file corpora still discriminate by term rarity.
    let n = files_seen.max(1) as f64;
    let idf: Vec<f64> = df
        .iter()
        .map(|&d| ((n + 1.0) / (d as f64 + 0.5)).ln())
        .collect();

    // Pass B: score lines and files.
    let mut scored_files: Vec<ScoredFile> = Vec::new();
    for scan in scans.into_iter() {
        let mut distinct_terms: FxHashMap<usize, ()> = FxHashMap::default();
        let mut scored: Vec<(usize, f64)> = Vec::with_capacity(scan.hits.len());
        for hit in &scan.hits {
            let mut line_score = hit.local_score;
            for &pid in &hit.matched {
                distinct_terms.insert(pid, ());
                line_score += idf[pid];
            }
            scored.push((hit.line_index, line_score));
        }

        // Path-hit bonus weighted by term rarity.
        let hay_lower_path = scan.display_path.to_lowercase();
        let mut path_bonus = 0.0;
        for (pid, pattern) in patterns.iter().enumerate() {
            if !params.case_sensitive {
                if hay_lower_path.contains(&pattern.source.to_lowercase()) {
                    path_bonus += idf[pid];
                }
            } else if scan.display_path.contains(&pattern.source) {
                path_bonus += idf[pid];
            }
        }

        let file_score = scored.iter().map(|(_, s)| *s).fold(0.0_f64, f64::max)
            + 2.0 * distinct_terms.len() as f64
            + path_bonus;

        let total_matches = scored.len();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(MAX_SNIPPETS_PER_FILE);

        scored_files.push(ScoredFile {
            root_idx: scan.root_idx,
            scan,
            file_score,
            scored,
            total_matches,
        });
    }

    render_selection(scored_files, params, files_seen)
}

fn is_identifier_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Whether a hit line earns the flat whole-query bonus: any matched pattern
/// with `term_id: None` (the whole-query phrase in term-fanout mode, or the
/// single pattern in regex mode). Every matched pid must be inspected — the
/// phrase is appended *last* in pattern order, so the first match is always a
/// plain term, and checking only `matched[0]` would silently disable the bonus.
fn line_has_exact_phrase_bonus(patterns: &[TermPattern], matched: &[usize]) -> bool {
    matched.iter().any(|&pid| patterns[pid].term_id.is_none())
}

// ─── Fair-share selection & rendering ────────────────────────────────────────

/// Selects files/lines across roots with relevance-first ordering plus
/// fair-share visibility, then renders verbatim excerpt blocks.
///
/// Selection happens at line granularity: candidates enter a global pool, and
/// while the answer budget lasts, the pool is drained in relevance order, but a
/// root whose consumed share reaches `ceil(max_results/roots) * FAIR_SHARE_MULTIPLE`
/// sits out while other roots still have candidates left — and resumes once
/// they are spent, so the leftover budget is not wasted on an under-filled
/// answer. Equal-score ties rotate across files, so symmetric floods (e.g. the
/// same marker repeated in several archives) distribute visibly instead of
/// collapsing into a single dominant file.
fn render_selection(
    mut files: Vec<ScoredFile>,
    params: &OverflowSearchParams<'_>,
    files_seen: usize,
) -> Result<String, String> {
    files.sort_by(|a, b| {
        b.file_score
            .partial_cmp(&a.file_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.scan.display_path.cmp(&b.scan.display_path))
    });

    struct Candidate {
        file_pos: usize,
        line_index: usize,
    }

    let mut per_file_cursor = vec![0usize; files.len()];
    let mut root_consumed: FxHashMap<usize, usize> = FxHashMap::default();
    let roots_with_hits: Vec<usize> = {
        let mut v: Vec<usize> = files.iter().map(|f| f.root_idx).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    // A degenerate request of zero results is clamped to one so real matches
    // are never misreported as "No matches found".
    let max_results = params.max_results.max(1);
    let fair_share = (max_results / roots_with_hits.len().max(1)).max(1);
    let soft_cap_per_root = fair_share * FAIR_SHARE_MULTIPLE;

    let mut chosen: Vec<Candidate> = Vec::new();

    // Every pass visits all ranked files once; each visit either pops one
    // candidate or counts that file as settled. A capped root sits out only
    // while another root still holds an un-drained candidate (active_roots),
    // so the cap is a fairness valve rather than a hard quota: once every
    // other root is spent, the dominant root resumes draining and the full
    // budget is used. Equal-score ties rotate across files because each pass
    // re-visits files in stable global relevance order.
    while chosen.len() < max_results {
        // Roots that still hold at least one un-drained candidate this pass.
        let active_roots: std::collections::BTreeSet<usize> = files
            .iter()
            .enumerate()
            .filter(|(pos, f)| per_file_cursor[*pos] < f.scored.len())
            .map(|(_, f)| f.root_idx)
            .collect();

        let mut settled = 0usize;
        for file_pos in 0..files.len() {
            if chosen.len() >= max_results {
                break;
            }
            let file = &files[file_pos];
            let root_capped =
                root_consumed.get(&file.root_idx).copied().unwrap_or(0) >= soft_cap_per_root;
            if !active_roots.contains(&file.root_idx)
                || per_file_cursor[file_pos] >= file.scored.len()
                || (root_capped && active_roots.len() > 1)
            {
                // Capped roots sit out while other roots still demand room.
                settled += 1;
                continue;
            }
            let line_index = file.scored[per_file_cursor[file_pos]].0;
            per_file_cursor[file_pos] += 1;
            *root_consumed.entry(file.root_idx).or_insert(0) += 1;
            chosen.push(Candidate {
                file_pos,
                line_index,
            });
        }
        if settled == files.len() {
            break;
        }
    }

    if chosen.is_empty() {
        return Ok(format!(
            "No matches found in the session archive for query: '{}'",
            params.query
        ));
    }

    // Emit grouped by file in global relevance order, expanding context windows
    // around chosen lines; `'>'` marks matched lines, numbers are absolute
    // archive-file line numbers usable as `read_file` offsets.
    let mut out = String::new();
    let mut shown_matches = 0usize;
    let mut shown_files = 0usize;
    let mut total_hidden = 0usize;
    let total_matches_all: usize = files.iter().map(|f| f.total_matches).sum();

    for (file_pos, file) in files.iter().enumerate() {
        // Sorted ascending so context ranges merge correctly; dedup guards
        // against any duplicate selection.
        let mut lis: Vec<usize> = chosen
            .iter()
            .filter(|c| c.file_pos == file_pos)
            .map(|c| c.line_index)
            .collect();
        lis.sort_unstable();
        lis.dedup();
        if lis.is_empty() {
            continue;
        }
        shown_files += 1;
        shown_matches += lis.len();
        let hidden_here = file.total_matches.saturating_sub(lis.len());
        total_hidden += hidden_here;

        // The scan pass retained only indices and scores, never the archive
        // text. Re-read the file to render the selected excerpts so at most
        // one archive file is resident at a time — and only files that
        // survived selection are re-read at all.
        let Ok(content) = fs::read_to_string(&file.scan.display_path) else {
            // Vanished or unreadable between scan and render (should not
            // happen within one search): fall back to a pointer, never
            // fabricate excerpts.
            out.push_str(&format!(
                "### {} match(es) in {}\n",
                lis.len(),
                &file.scan.display_path
            ));
            out.push_str(
                "... [file unreadable during excerpt rendering; use read_file for surrounding context] ...\n\n",
            );
            continue;
        };
        let content_lines: Vec<&str> = content.lines().collect();
        let n_lines = content_lines.len();
        if n_lines == 0 {
            // Empty between scan and render (concurrent truncation): nothing
            // to excerpt, and indexing below would panic. Emit the header so
            // the footer's shown counts stay consistent with visible output.
            out.push_str(&format!(
                "### {} match(es) in {}\n",
                lis.len(),
                &file.scan.display_path
            ));
            out.push_str(
                "... [file empty during excerpt rendering; use read_file for surrounding context] ...\n\n",
            );
            continue;
        }

        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for li in &lis {
            let start = li.saturating_sub(params.context_lines);
            // Clamp against both the scan-time line count and the re-read
            // length so a concurrently truncated file can never cause an
            // out-of-bounds index.
            let end = (*li + params.context_lines)
                .min(file.scan.total_lines.saturating_sub(1))
                .min(n_lines.saturating_sub(1));
            if let Some(last) = ranges.last_mut() {
                if start <= last.1.saturating_add(1) {
                    last.1 = last.1.max(end);
                    continue;
                }
            }
            ranges.push((start, end));
        }

        out.push_str(&format!(
            "### {} match(es) in {}\n",
            lis.len(),
            &file.scan.display_path
        ));
        let mut match_set: std::collections::BTreeSet<usize> = lis.iter().copied().collect();
        for (start, end) in ranges {
            for li in start..=end {
                let marker = if match_set.remove(&li) { ">" } else { " " };
                out.push_str(&format!("{:>7}{} {}\n", li + 1, marker, content_lines[li]));
            }
        }
        if hidden_here > 0 {
            out.push_str(&format!(
                "... [{} more matching line(s) in this file not shown; narrow the query, raise max_results, or use file_pattern/scope] ...\n",
                hidden_here
            ));
        }
        out.push('\n');
    }

    if out.len() > MAX_OUTPUT_CHARS {
        // Archived content is arbitrary UTF-8, so MAX_OUTPUT_CHARS may land mid
        // codepoint; `String::truncate` panics off a char boundary (and the
        // release profile aborts). Snap down to the nearest boundary first.
        let mut cut = MAX_OUTPUT_CHARS;
        while cut > 0 && !out.is_char_boundary(cut) {
            cut -= 1;
        }
        out.truncate(cut);
        out.push_str("\n... [output truncated at character limit; narrow the query, scope, or file_pattern]\n");
    }
    out.push_str(&format!(
        "[archive search] showed {} matching line(s) across {} file(s); {} additional matching line(s) hidden (corpus total {}, files scanned {}). Use `read_file` on any listed absolute path for surrounding context.\n",
        shown_matches, shown_files, total_hidden.max(total_matches_all.saturating_sub(shown_matches)), total_matches_all, files_seen
    ));
    Ok(out)
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "search_overflow",
        description: "",
        execute: execute_search_overflow,
    }
});

// search_overflow results are localization pointers for recalled compressed
// content: reproducing them costs another full search, so lossy compression is
// forbidden and they spill verbatim with a pointer stub. Pruning stale results
// remains allowed (same policy as read_file). Hits themselves are never trimmed
// inline; the whole result spills to disk with a pointer only when the context
// budget forces it.
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "search_overflow",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Allow,
        counts_toward_precision_inline_budget: true,
    },
});

mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn make_temp_dir() -> PathBuf {
        // Must be unique: create_dir_all is idempotent, so colliding parallel
        // tests would silently share one directory and race on cleanup.
        let dir = std::env::temp_dir().join(format!(
            "search_overflow_test_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn seed_archive(dir: &Path) {
        fs::write(
            dir.join("overflow-history.md"),
            "## User\nOriginal question: implement a utility\n## Assistant\nDecision recorded about foo.\n## Tool result\n- some compressed command output\n",
        )
        .unwrap();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        fs::write(
            tool_dir.join("20260804T140000Z-execute_command-deadbeef.txt"),
            "original_command: grep -n foo\nfoo line 1\nfoo line 2\n",
        )
        .unwrap();
        fs::write(
            tool_dir.join("20260804T140000Z-read_file-deadbeef.txt"),
            "read_file content\nbar line\n",
        )
        .unwrap();
        let folded_dir = dir.join("folded-tool-groups");
        fs::create_dir_all(&folded_dir).unwrap();
        fs::write(folded_dir.join("group.md"), "folded foo evidence\n").unwrap();
        let note_dir = dir.join("internal-note-overflow");
        fs::create_dir_all(&note_dir).unwrap();
        fs::write(note_dir.join("note.md"), "internal foo state\n").unwrap();
        let user_dir = dir.join("user-overflow-preserved");
        fs::create_dir_all(&user_dir).unwrap();
        fs::write(user_dir.join("user.md"), "preserved foo request\n").unwrap();
        let image_dir = dir.join("image-overflow-preserved");
        fs::create_dir_all(&image_dir).unwrap();
        fs::write(image_dir.join("image.md"), "preserved foo image context\n").unwrap();
    }

    fn params(query: &str) -> OverflowSearchParams<'_> {
        OverflowSearchParams {
            query,
            is_regex: false,
            case_sensitive: true,
            context_lines: 1,
            max_results: 50,
            file_pattern: None,
            scope: SearchScope::All,
        }
    }

    #[test]
    fn search_all_scopes_both_locations() {
        let dir = make_temp_dir();
        seed_archive(&dir);
        let out = run_overflow_search(&dir, &params("foo")).unwrap();
        assert!(
            out.contains("overflow-history.md"),
            "history file in results: {out}"
        );
        assert!(
            out.contains("20260804T140000Z-execute_command-deadbeef.txt"),
            "tool output in results: {out}"
        );
        assert!(out.contains("foo line 1"));
        assert!(out.contains("folded-tool-groups/group.md"), "{out}");
        assert!(out.contains("internal-note-overflow/note.md"), "{out}");
        assert!(out.contains("user-overflow-preserved/user.md"), "{out}");
        assert!(out.contains("image-overflow-preserved/image.md"), "{out}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_history_scope_only() {
        let dir = make_temp_dir();
        seed_archive(&dir);
        let mut p = params("implement");
        p.scope = SearchScope::History;
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(out.contains("overflow-history.md"));
        assert!(!out.contains("tool-overflow-compressed"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_tool_outputs_scope_with_pattern() {
        let dir = make_temp_dir();
        seed_archive(&dir);
        let mut p = params("foo");
        p.scope = SearchScope::ToolOutputs;
        p.file_pattern = Some("*execute_command*");
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(
            out.contains("execute_command"),
            "command snapshot matched: {out}"
        );
        assert!(
            // Check the snapshot filename, not the bare word: the result
            // footer legitimately mentions the `read_file` tool by name.
            !out.contains("-read_file-"),
            "read_file snapshot excluded: {out}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_case_insensitive_flag_still_honored() {
        let dir = make_temp_dir();
        seed_archive(&dir);
        let mut p = params("FOO");
        p.case_sensitive = false;
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(out.contains("foo line 1"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_case_sensitive_default_misses_other_case() {
        let dir = make_temp_dir();
        seed_archive(&dir);
        let out = run_overflow_search(&dir, &params("FOO")).unwrap();
        assert!(out.contains("No matches found"), "exact-case miss: {out}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_no_matches_reports_cleanly() {
        let dir = make_temp_dir();
        seed_archive(&dir);
        let out = run_overflow_search(&dir, &params("zzz_absent")).unwrap();
        assert!(out.contains("No matches found"), "clean miss: {out}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn search_missing_archive_roots_are_skipped() {
        let dir = make_temp_dir(); // empty directory: no roots exist
        let out = run_overflow_search(&dir, &params("foo")).unwrap();
        assert!(out.contains("No matches found"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn multi_term_query_ranks_full_coverage_above_single_term() {
        let dir = make_temp_dir();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        // Both files match exactly one term apiece on many lines...
        fs::write(tool_dir.join("alpha-only.txt"), &"alpha token\n".repeat(30)).unwrap();
        // ...but this one matches BOTH terms plus the phrase on fewer lines.
        fs::write(
            tool_dir.join("both.txt"),
            "alpha beta\nalpha beta tail\nunrelated\n",
        )
        .unwrap();

        let mut p = params("alpha beta");
        p.context_lines = 0;
        p.max_results = 10;
        let out = run_overflow_search(&dir, &p).unwrap();
        let both_pos = out.find("both.txt").expect("both.txt must appear");
        let alpha_only_pos = out.find("alpha-only.txt").expect("alpha-only must appear");
        assert!(
            both_pos < alpha_only_pos,
            "multi-term coverage must rank first:\n{out}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn flood_does_not_starve_minority_roots() {
        let dir = make_temp_dir();
        fs::write(dir.join("overflow-history.md"), &"alpha\n".repeat(100)).unwrap();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        fs::write(tool_dir.join("a.txt"), &"alpha\n".repeat(100)).unwrap();
        let folded_dir = dir.join("folded-tool-groups");
        fs::create_dir_all(&folded_dir).unwrap();
        fs::write(folded_dir.join("a.md"), &"alpha\n".repeat(100)).unwrap();
        let note_dir = dir.join("internal-note-overflow");
        fs::create_dir_all(&note_dir).unwrap();
        fs::write(note_dir.join("a.md"), "alpha unique-marker\n").unwrap();

        let mut p = params("alpha");
        p.max_results = 5;
        p.context_lines = 0;
        let out = run_overflow_search(&dir, &p).unwrap();

        // Symmetric floods rotate across roots, and the low-volume note archive
        // must stay visible under the same tiny budget.
        let sections = out.matches("match(es) in").count();
        assert_eq!(sections, 4, "all four hit roots visible: {out}");
        assert!(
            out.contains("unique-marker"),
            "minority root visible: {out}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn capped_root_absorbs_leftover_budget_when_others_exhausted() {
        // Regression: with max_results=10 over three hit roots where two
        // minority roots have one line each, fair_share = 10/3 = 3 and the
        // soft cap is 6. The dominant root must absorb the remaining budget
        // once the minority roots are spent; the old code never lifted the
        // cap and returned only 8 of 10 lines.
        let dir = make_temp_dir();
        fs::write(dir.join("overflow-history.md"), "alpha\n").unwrap();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        fs::write(tool_dir.join("noisy.txt"), &"alpha\n".repeat(200)).unwrap();
        let folded_dir = dir.join("folded-tool-groups");
        fs::create_dir_all(&folded_dir).unwrap();
        fs::write(folded_dir.join("a.md"), "alpha\n").unwrap();

        let mut p = params("alpha");
        p.context_lines = 0;
        p.max_results = 10;
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(
            out.contains("[archive search] showed 10 matching line(s)"),
            "full budget must be used once minority roots are spent:\n{out}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn phrase_pattern_line_earns_exact_bonus() {
        // Regression: a line matching the whole query "alpha beta" matches the
        // phrase pattern AND both term patterns. Patterns are pid-ordered with
        // the phrase last, so inspecting only `matched[0]` (always a plain
        // term) made the exact-phrase bonus dead code; every matched pid must
        // be scanned.
        let mut p = params("alpha beta");
        p.case_sensitive = false;
        let patterns = build_patterns(&p).unwrap();
        let mut whole: Vec<usize> = Vec::new();
        for (pid, pat) in patterns.iter().enumerate() {
            if pat.regex.is_match("alpha beta") {
                whole.push(pid);
            }
        }
        assert!(
            line_has_exact_phrase_bonus(&patterns, &whole),
            "whole-query line must earn the phrase bonus"
        );
        let mut single: Vec<usize> = Vec::new();
        for (pid, pat) in patterns.iter().enumerate() {
            if pat.regex.is_match("alpha") {
                single.push(pid);
            }
        }
        assert!(
            !line_has_exact_phrase_bonus(&patterns, &single),
            "a lone term line must not earn the phrase bonus"
        );
    }

    #[test]
    fn word_boundary_bonus_survives_scan_time_scoring() {
        // Regression for moving line-local scoring (whole-word + lead-proximity
        // bonuses) from pass B into the scan pass, which let the archive text
        // be dropped after scanning. Without the +2 whole-word bonus the
        // substring line would win here (its lead bonus exceeds the long
        // whole-word line's), so the test discriminates the refactored path.
        let dir = make_temp_dir();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        // Line 1: whole-word "alpha" with a large lead (40 chars) -> lead bonus
        // 0, whole-word bonus +2. Line 2: substring "xxalpha" -> lead bonus
        // ~1.9, no whole-word bonus.
        fs::write(
            tool_dir.join("words.txt"),
            format!("{}alpha\nxxalpha\n", "a ".repeat(20)),
        )
        .unwrap();

        let mut p = params("alpha");
        p.scope = SearchScope::ToolOutputs;
        p.context_lines = 0;
        p.max_results = 1;
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(
            out.contains("1> a "),
            "whole-word line must win the single slot:\n{out}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn zero_max_results_clamps_to_at_least_one() {
        let dir = make_temp_dir();
        seed_archive(&dir);
        let mut p = params("foo");
        p.max_results = 0;
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(
            !out.contains("No matches found"),
            "degenerate max_results=0 must not hide real matches:\n{out}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn empty_file_pattern_is_ignored() {
        let dir = make_temp_dir();
        seed_archive(&dir);
        let mut p = params("foo");
        p.file_pattern = Some("");
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(
            out.contains("20260804T140000Z-execute_command-deadbeef.txt"),
            "an empty file_pattern must not filter out directory-root files:\n{out}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn symlink_cycle_terminates() {
        use std::os::unix::fs::symlink;
        let dir = make_temp_dir();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        fs::write(tool_dir.join("needle.txt"), "alpha needle\n").unwrap();
        // A directory symlinked back into itself; without cycle protection the
        // DFS would push forever and the search would hang.
        symlink(&tool_dir, tool_dir.join("loop")).unwrap();
        let mut p = params("needle");
        p.scope = SearchScope::ToolOutputs;
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(
            out.contains("needle.txt"),
            "search must terminate and still find hits:\n{out}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn repetitive_log_cannot_occupy_whole_answer() {
        let dir = make_temp_dir();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        fs::write(
            tool_dir.join("huge-log.txt"),
            &"spam spam spam\n".repeat(500),
        )
        .unwrap();
        fs::write(tool_dir.join("small-note.txt"), "spam context survivor\n").unwrap();

        let mut p = params("spam");
        p.scope = SearchScope::ToolOutputs;
        p.context_lines = 0;
        p.max_results = 50;
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(
            out.contains("small-note.txt"),
            "smaller file survives: {out}"
        );
        assert!(
            out.contains("not shown"),
            "per-file cap hides surplus lines: {out}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn footer_reports_shown_and_hidden_totals() {
        let dir = make_temp_dir();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        fs::write(tool_dir.join("big.txt"), &"needle\n".repeat(80)).unwrap();

        let mut p = params("needle");
        p.scope = SearchScope::ToolOutputs;
        p.context_lines = 0;
        p.max_results = 10;
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(
            out.contains("[archive search] showed "),
            "footer present: {out}"
        );
        assert!(
            out.contains("(corpus total 80"),
            "hidden-vs-total accounted: {out}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn regex_mode_single_pattern_passthrough() {
        let dir = make_temp_dir();
        seed_archive(&dir);
        let mut p = params(r"foo \w+ \d");
        p.is_regex = true;
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(out.contains("foo line 1"), "regex matches: {out}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn invalid_regex_surfaces_error() {
        let dir = make_temp_dir();
        seed_archive(&dir);
        let mut p = params("(unclosed");
        p.is_regex = true;
        let err = run_overflow_search(&dir, &p).unwrap_err();
        assert!(err.contains("Invalid regex"), "{err}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn oversized_multibyte_output_truncates_without_panicking() {
        // Regression: MAX_OUTPUT_CHARS is a byte cap, and archived content is
        // arbitrary UTF-8. A render whose cap byte falls mid-codepoint must snap
        // to a char boundary instead of panicking (release profile = abort).
        let dir = make_temp_dir();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        // Only MAX_SNIPPETS_PER_FILE lines survive per file, so each matching
        // line must be long enough that that many blow past MAX_OUTPUT_CHARS.
        // '好' is 3 bytes and straddles the byte cap regardless of gutter width.
        let line = format!("needle {}\n", "好".repeat(1_000));
        fs::write(
            tool_dir.join("wide.txt"),
            line.repeat(MAX_SNIPPETS_PER_FILE + 4),
        )
        .unwrap();

        let mut p = params("needle");
        p.scope = SearchScope::ToolOutputs;
        p.context_lines = 0;
        p.max_results = MAX_MATCHES;
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(
            out.len() > MAX_OUTPUT_CHARS,
            "test must exercise truncation"
        );
        assert!(
            out.contains("output truncated at character limit"),
            "truncation notice must survive"
        );
        fs::remove_dir_all(&dir).ok();
    }
}
