//! search_overflow — ranked retrieval over the session overflow archive
//!
//! Content moved out of context is archived into session assets (verbatim
//! unless noted):
//! - `overflow-history.md`: original folded messages (user/assistant/tool results)
//! - `summary-sources/`: raw message records behind compressed summary increments
//!   (one serialized message per line; searched alongside `overflow-history.md`
//!   rather than with the tool-output snapshots, since it holds the same class of
//!   content; written by `history/compress/incremental.rs`)
//! - `tool-overflow-compressed/`: full snapshots of individual tool results
//! - `folded-tool-groups/`: request-projection copies of wholly folded tool-call
//!   groups (high-precision tool results appear as spill stubs or as full-text
//!   copies kept verbatim at fold time; lossy results may be reduced to a summary)
//! - `internal-note-overflow/`: internal context notes trimmed by budget
//! - `context-checkpoints/`: assistant-derived checkpoint bodies (unverified)
//! - `tool-overflow/`: verbatim oversized live tool results
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
//!    roots still demand attention; the same reservation applies per file while
//!    other files still hold candidates, and ties spread round-robin. This
//!    replaces the previous static even quota split between roots and prevents
//!    either starvation mode: an early noisy root monopolising results, or
//!    equally a relevant-but-quota-starved archive section becoming permanently
//!    invisible.
//! 4. **Budget-aware excerpts**: candidate pools are never truncated by a fixed
//!    per-file ceiling — a hit ranked below the top dozen of one large file stays
//!    selectable — and rendering hides surplus lines with an accurate count
//!    instead of cutting excerpts away. Long lines are windowed around the first
//!    match (`…` marks elided text), so the term that caused a hit is never cut
//!    off.
//!
//! Results stay verbatim excerpts with absolute paths and line numbers so they
//! can be fed directly into `read_file`.
//!
//! Safety: the search root is never taken from caller input; it is derived only
//! from `current_session_assets_dir()`, and errors out when no driver context is
//! active. Discovered symlinks are excluded; both scanning and excerpt rendering
//! use the session-bound reader, which opens every path component without
//! following symlinks on Unix.

#[cfg(not(unix))]
use std::fs;
use std::path::{Path, PathBuf};

use regex::{Regex, RegexBuilder};
use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;

use crate::ai::history::compress::SOURCE_DIR;
use crate::ai::tools::common::{
    ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
    ToolRegistration, ToolSpec,
};
use crate::ai::tools::storage::file_store::current_session_assets_dir;

// Share the content-read boundary with automatic recall, but not its scan
// budgets: manual recovery must still reach large files and late/deep paths.
#[path = "../driver/turn_runtime/context_memory_archive.rs"]
mod archive_files;

fn read_archive_text(archive: &archive_files::ArchiveRoot, path: &Path) -> Option<String> {
    // Preserve full-file manual search semantics. Output is bounded separately;
    // scan work and resident hit metadata are not bounded by automatic-recall
    // limits. A fresh budget on each read prevents earlier files hiding later ones.
    let mut remaining_bytes = usize::MAX;
    archive.read_text(path, usize::MAX, &mut remaining_bytes)
}

/// Hard ceiling of matched lines returned (mirrors the shared engine cap).
const MAX_MATCHES: usize = 200;
/// Reservation ceiling for one file: a huge repetitive log cannot occupy the
/// whole answer while other files still hold candidates. It never makes a hit
/// unreachable: the ceiling is derived from `max_results` when that is larger,
/// and it lifts once the other files are spent.
const MAX_SNIPPETS_PER_FILE: usize = 12;
/// While other roots still contribute candidates, a single root may consume at
/// most this multiple of the average share before yielding its turn.
const FAIR_SHARE_MULTIPLE: usize = 2;
/// Fixed weight of an exact whole-query phrase hit versus single-term IDF mass.
const PHRASE_WEIGHT: f64 = 6.0;
/// Score weight of a derived CJK bigram relative to a term as typed. The
/// expansion exists to widen recall in unsegmented text; a line that merely
/// shares a two-character fragment must not outweigh one holding the full run.
const CJK_BIGRAM_WEIGHT: f64 = 0.35;
/// Ceiling on derived CJK bigram patterns per query, so a long Chinese sentence
/// cannot fan out into an unbounded set of per-line regex probes.
const MAX_CJK_BIGRAMS: usize = 24;
/// Rendered output size guard (archive files can be arbitrarily large).
const MAX_OUTPUT_CHARS: usize = 24_000;
/// Longest text rendered from a single archive line, in characters (the byte
/// budget around it is enforced separately). An arbitrarily long line (minified
/// JSON, a base64 payload) would otherwise consume the whole output budget and
/// could cut away the very term that matched it.
const MAX_RENDERED_LINE_CHARS: usize = 600;
/// Output space held back for the truncation notice and the footer, so excerpt
/// rendering stops early instead of the accounting line being dropped.
const FOOTER_RESERVE_CHARS: usize = 700;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SearchScope {
    /// All session archive content.
    All,
    /// Only original folded messages: overflow-history.md plus the raw records
    /// behind compressed summary increments (summary-sources/).
    History,
    /// Tool snapshots, folded groups, internal notes, and checkpoint bodies.
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
    /// Multiplier on this pattern's IDF weight in line, coverage, and path
    /// scoring: 1.0 for text the caller typed, `CJK_BIGRAM_WEIGHT` for the
    /// derived bigram expansions added by `add_cjk_bigram_patterns`.
    weight: f64,
    regex: Regex,
}

fn build_patterns(params: &OverflowSearchParams<'_>) -> Result<Vec<TermPattern>, String> {
    if params.is_regex {
        let regex = compile_pattern(params.query, params.is_regex, params.case_sensitive)?;
        return Ok(vec![TermPattern {
            source: params.query.to_string(),
            term_id: None,
            weight: 1.0,
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
            weight: 1.0,
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
            weight: 1.0,
            regex,
        });
    }
    // Unsegmented scripts get an extra recall path, added after the literal
    // terms above because those remain the decisive, higher-weighted matches.
    add_cjk_bigram_patterns(params, &mut patterns)?;
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

/// Whether `ch` belongs to a script written without inter-word spaces, where
/// whitespace splitting cannot yield useful terms. Covers Han (including
/// Extension A and the compatibility block) and kana; Hangul is excluded
/// because Korean text is space-separated, and Thai would need a dictionary.
fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F
    )
}

/// Whether a pattern is a recall expansion derived from the query rather than
/// text the caller typed (currently only the CJK bigrams).
fn is_derived_pattern(pattern: &TermPattern) -> bool {
    pattern.weight < 1.0
}

/// Adds the adjacent bigrams of every CJK run in the query as low-weight
/// alternatives.
///
/// A Chinese query has no spaces to split on, so "压缩的文档" is a single literal
/// term that never matches the compact "压缩文档": the miss is caused by
/// function words the caller cannot know are absent from the archive. Bigrams
/// restore that recall while the typed term stays decisive, because a line
/// holding the full run necessarily contains every one of its bigrams as well,
/// so it still outranks a line sharing only one fragment.
///
/// Runs shorter than three characters are skipped (their single bigram is the
/// run itself) and at most `MAX_CJK_BIGRAMS` patterns are added. A query with
/// no CJK text leaves the pattern set untouched.
fn add_cjk_bigram_patterns(
    params: &OverflowSearchParams<'_>,
    patterns: &mut Vec<TermPattern>,
) -> Result<(), String> {
    let mut seen: FxHashSet<String> = patterns
        .iter()
        .map(|pattern| pattern.source.clone())
        .collect();
    let mut added = 0usize;
    for (term_id, term) in params.query.split_whitespace().enumerate() {
        let mut run: Vec<char> = Vec::new();
        // The synthetic separator flushes a trailing run instead of duplicating
        // the expansion loop after the character loop.
        for ch in term.chars().chain(std::iter::once(' ')) {
            if is_cjk(ch) {
                run.push(ch);
                continue;
            }
            if run.len() >= 3 {
                for pair in run.windows(2) {
                    if added == MAX_CJK_BIGRAMS {
                        return Ok(());
                    }
                    let bigram: String = pair.iter().collect();
                    if !seen.insert(bigram.clone()) {
                        continue;
                    }
                    let regex = compile_pattern(&bigram, false, params.case_sensitive)?;
                    patterns.push(TermPattern {
                        source: bigram,
                        term_id: Some(term_id),
                        weight: CJK_BIGRAM_WEIGHT,
                        regex,
                    });
                    added += 1;
                }
            }
            run.clear();
        }
    }
    Ok(())
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
    /// Matched lines sorted by descending score. Not truncated here: selection
    /// applies the per-file ceiling against the live budget.
    scored: Vec<(usize, f64)>,
    total_matches: usize,
}

/// Iterates concrete archive files without automatic-recall candidate/depth caps.
#[cfg(unix)]
fn collect_files(archive: &archive_files::ArchiveRoot, root: &Path) -> Vec<PathBuf> {
    // Enumeration and content reads share the held session directory boundary;
    // checking a canonical path before read_dir would still allow symlink swaps.
    archive.collect_files_unbounded(root)
}

// Preserve the path-based fallback on platforms without the Unix directory
// descriptor primitives. Containment here is best-effort, not race-safe.
#[cfg(not(unix))]
fn collect_files(archive: &archive_files::ArchiveRoot, root: &Path) -> Vec<PathBuf> {
    let Ok(metadata) = fs::symlink_metadata(root) else {
        return Vec::new();
    };
    if metadata.file_type().is_symlink() {
        return Vec::new();
    }
    if metadata.is_file() {
        return vec![root.to_path_buf()];
    }
    let mut out: Vec<PathBuf> = Vec::new();
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    let mut visited: FxHashSet<PathBuf> = FxHashSet::default();
    while let Some(dir) = stack.pop() {
        let Ok(canon) = fs::canonicalize(&dir) else {
            continue;
        };
        if !canon.starts_with(archive.path()) || !visited.insert(canon) {
            continue;
        }
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        let mut names: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        // Deterministic traversal independent of filesystem order.
        names.sort();
        for path in names {
            let Ok(metadata) = fs::symlink_metadata(&path) else {
                continue;
            };
            if metadata.is_dir() {
                stack.push(path);
            } else if metadata.is_file() {
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
    let Some(archive) = archive_files::ArchiveRoot::new(assets_dir) else {
        return Ok("No matches found: the session archive root is missing or unavailable; nothing was scanned.".to_string());
    };
    let mut roots: Vec<PathBuf> = Vec::new();
    if params.scope != SearchScope::ToolOutputs {
        roots.push(archive.path().join("overflow-history.md"));
        // Raw records behind compressed summary increments. The incremental
        // summary path writes the removed span *only* here (plain compression,
        // used when summaries are disabled, appends to overflow-history.md
        // instead), so without this root those originals were unreachable. The
        // name comes from the writer, so renaming it cannot silently hide the
        // directory again.
        roots.push(archive.path().join(SOURCE_DIR));
    }
    if params.scope != SearchScope::History {
        roots.extend(
            archive_files::RECOVERY_DIRECTORIES
                .iter()
                .map(|name| archive.path().join(name)),
        );
    }
    if params.scope == SearchScope::All {
        roots.extend(
            ["user-overflow-preserved", "image-overflow-preserved"]
                .map(|name| archive.path().join(name)),
        );
    }
    let roots: Vec<(usize, PathBuf)> = roots
        .into_iter()
        .filter(|root| root.exists())
        .enumerate()
        .collect();
    if roots.is_empty() {
        return Ok(format!(
            "No matches found in the session archive for query: '{}'. Nothing was scanned: this session has no archive content yet, since files appear here only after content is moved out of context.",
            params.query,
        ));
    }

    let patterns = build_patterns(params)?;

    // Pass A: scan every file once, collecting raw hits plus document
    // frequencies feeding the IDF weights.
    let mut scans: Vec<FileScan> = Vec::new();
    let mut df: Vec<usize> = vec![0; patterns.len()];
    let mut files_seen: usize = 0;
    // Counted separately so a zero-hit answer can say whether the corpus was
    // empty, filtered away, or unreadable instead of implying absence.
    let mut files_excluded_by_pattern: usize = 0;
    let mut files_unreadable: usize = 0;

    // Compiled once: the glob is fixed for the whole search. A pattern that
    // trims to empty is treated as "no filter" — an empty glob would only match
    // a bare relative path "" (the History scope's file root) and silently
    // exclude every file under a directory root.
    let pattern_filter = params
        .file_pattern
        .map(str::trim)
        .filter(|p| !p.is_empty());
    let glob = pattern_filter.map(glob_to_regex);

    for (root_idx, root) in &roots {
        for file in collect_files(&archive, root) {
            if !path_matches_glob(&file, root, glob.as_ref()) {
                files_excluded_by_pattern += 1;
                continue;
            }
            let Some(content) = read_archive_text(&archive, &file) else {
                // Unreadable, unsafe, or non-UTF-8 files were not searched;
                // exclude them from the corpus size used by IDF.
                files_unreadable += 1;
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
        // A bare "No matches found" invites unwarranted absence claims: the
        // caller cannot distinguish an empty corpus or a filtered-away corpus
        // from a genuine miss. Report what was probed and which terms came up
        // empty, so a missing-evidence conclusion is not asserted from this.
        let probed = roots
            .iter()
            .map(|(_, path)| {
                path.strip_prefix(archive.path())
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_string()
            })
            .collect::<Vec<String>>()
            .join(", ");
        let pattern_note = match pattern_filter {
            Some(pattern) => format!(
                "; file_pattern '{pattern}' excluded {files_excluded_by_pattern} file(s)"
            ),
            None => String::new(),
        };
        let unreadable_note = if files_unreadable > 0 {
            format!("; {files_unreadable} file(s) were unreadable or unsafe to read")
        } else {
            String::new()
        };
        let scope_note = match params.scope {
            SearchScope::All => String::new(),
            SearchScope::History => "; scope=history searched original messages only".to_string(),
            SearchScope::ToolOutputs => {
                "; scope=tool_outputs skipped original messages".to_string()
            }
        };
        if files_seen == 0 {
            return Ok(format!(
                "No matches found in the session archive for query: '{}'. Nothing was scanned: 0 archive file(s) under the probed roots ({probed}){pattern_note}{unreadable_note}{scope_note}. Archived files appear only after content is moved out of context; canonical session history is not searched here.",
                params.query,
            ));
        }
        let mut terms: Vec<&str> = patterns
            .iter()
            .filter(|pattern| pattern.term_id.is_some() && !is_derived_pattern(pattern))
            .map(|pattern| pattern.source.as_str())
            .collect();
        if terms.is_empty() {
            // Regex mode: the single compiled pattern is the whole query.
            terms.push(params.query);
        }
        const MAX_LISTED_TERMS: usize = 12;
        let listed = if terms.len() > MAX_LISTED_TERMS {
            format!(
                "'{}' (first {MAX_LISTED_TERMS} of {})",
                terms[..MAX_LISTED_TERMS].join("', '"),
                terms.len()
            )
        } else {
            format!("'{}'", terms.join("', '"))
        };
        // Derived CJK bigrams are counted, not listed: the answer should name
        // what the caller asked for while still recording that the expansion was
        // tried, so the miss is not read as narrower than it was.
        let expansions = patterns
            .iter()
            .filter(|pattern| is_derived_pattern(pattern))
            .count();
        let expansion_note = if expansions > 0 {
            format!("; {expansions} derived CJK bigram pattern(s) also came up empty")
        } else {
            String::new()
        };
        let case_note = if params.case_sensitive {
            " Matching is case-sensitive here: retry with case_sensitive=false to widen it."
        } else {
            ""
        };
        return Ok(format!(
            "No matches found in the session archive for query: '{}'. {files_seen} file(s) were searched and no term matched (terms: {listed}{expansion_note}{pattern_note}{unreadable_note}{scope_note}); terms are ORed, so every one of them came up empty.{case_note} This archive holds only content moved out of context; canonical session history is not searched here.",
            params.query,
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
        // Pattern id → its scoring weight, deduplicated so the coverage bonus
        // still counts each matched pattern exactly once.
        let mut distinct_terms: FxHashMap<usize, f64> = FxHashMap::default();
        let mut scored: Vec<(usize, f64)> = Vec::with_capacity(scan.hits.len());
        for hit in &scan.hits {
            let mut line_score = hit.local_score;
            for &pid in &hit.matched {
                distinct_terms.insert(pid, patterns[pid].weight);
                line_score += idf[pid] * patterns[pid].weight;
            }
            scored.push((hit.line_index, line_score));
        }

        // Path-hit bonus weighted by term rarity.
        let hay_lower_path = scan.display_path.to_lowercase();
        let mut path_bonus = 0.0;
        for (pid, pattern) in patterns.iter().enumerate() {
            if !params.case_sensitive {
                if hay_lower_path.contains(&pattern.source.to_lowercase()) {
                    path_bonus += idf[pid] * pattern.weight;
                }
            } else if scan.display_path.contains(&pattern.source) {
                path_bonus += idf[pid] * pattern.weight;
            }
        }

        let file_score = scored.iter().map(|(_, s)| *s).fold(0.0_f64, f64::max)
            + 2.0 * distinct_terms.values().sum::<f64>()
            + path_bonus;

        let total_matches = scored.len();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        // Deliberately not truncated: the per-file ceiling is applied during
        // selection against the live budget, so a hit ranked below the top
        // dozen stays selectable instead of becoming permanently unreachable.

        scored_files.push(ScoredFile {
            root_idx: scan.root_idx,
            scan,
            file_score,
            scored,
            total_matches,
        });
    }

    render_selection(scored_files, params, &patterns, files_seen, &archive)
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

/// Byte offset of the earliest pattern match on `line`, used to keep the
/// matching text inside a windowed excerpt. Recomputed at render time because
/// the scan pass retains only line indices and scores, never the text.
fn first_match_offset(patterns: &[TermPattern], line: &str) -> Option<usize> {
    patterns
        .iter()
        .filter_map(|pattern| pattern.regex.find(line).map(|m| m.start()))
        .min()
}

/// Renders one archive line, windowed around `first_match` when the line is
/// longer than `MAX_RENDERED_LINE_CHARS`. An arbitrarily long line (minified
/// JSON, a base64 payload) can exceed the whole output budget, and trimming it
/// from the front would cut away the very term that matched it. Leading and
/// trailing `…` mark elided text, so a window is never mistaken for the whole
/// line.
fn render_line_window(line: &str, first_match: Option<usize>) -> String {
    let total = line.chars().count();
    if total <= MAX_RENDERED_LINE_CHARS {
        return line.to_string();
    }
    let match_chars = first_match
        .and_then(|byte| line.get(..byte))
        .map(|prefix| prefix.chars().count())
        .unwrap_or(0);
    let start = match_chars
        .saturating_sub(MAX_RENDERED_LINE_CHARS / 2)
        .min(total - MAX_RENDERED_LINE_CHARS);
    let mut window = String::with_capacity(MAX_RENDERED_LINE_CHARS + 8);
    if start > 0 {
        window.push('…');
    }
    window.extend(line.chars().skip(start).take(MAX_RENDERED_LINE_CHARS));
    if start + MAX_RENDERED_LINE_CHARS < total {
        window.push('…');
    }
    window
}

// ─── Fair-share selection & rendering ────────────────────────────────────────

/// Selects files/lines across roots with relevance-first ordering plus
/// fair-share visibility, then renders verbatim excerpt blocks.
///
/// Selection happens at line granularity: candidates enter a global pool, and
/// while the answer budget lasts, the pool is drained in relevance order, but a
/// root whose consumed share reaches `ceil(max_results/roots) * FAIR_SHARE_MULTIPLE`
/// sits out while other roots can still use that room — and resumes once those
/// roots are spent or equally capped, so the leftover budget is never wasted on
/// an under-filled answer. Equal-score ties rotate across files, so symmetric
/// floods (e.g. the same marker repeated in several archives) distribute
/// visibly instead of collapsing into a single dominant file.
///
/// The same valve applies per file: a file may consume more than its share of
/// the budget only while no other file still holds a candidate, so the soft
/// per-file ceiling (`MAX_SNIPPETS_PER_FILE`, raised to the per-file share when
/// the budget allows) reserves room without making surplus lines unreachable.
fn render_selection(
    mut files: Vec<ScoredFile>,
    params: &OverflowSearchParams<'_>,
    patterns: &[TermPattern],
    files_seen: usize,
    archive: &archive_files::ArchiveRoot,
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
    let mut file_consumed = vec![0usize; files.len()];
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
    // Per-file ceiling: the historical snippet cap, raised to the per-file
    // share of the budget when that is larger, so a wider `max_results` really
    // does expose more of one large file instead of being absorbed by the
    // first dozen lines.
    let files_holding_candidates = files
        .iter()
        .filter(|file| !file.scored.is_empty())
        .count()
        .max(1);
    let fair_share_per_file = (max_results / files_holding_candidates).max(1);
    let soft_cap_per_file = MAX_SNIPPETS_PER_FILE.max(fair_share_per_file * FAIR_SHARE_MULTIPLE);

    let mut chosen: Vec<Candidate> = Vec::new();

    // Every pass visits all ranked files once; each visit either pops one
    // candidate or counts that file as settled. A file is "free" while it holds
    // an un-drained candidate and is under both ceilings, and a capped file
    // sits out only while some file is still free: the caps decide who goes
    // first, never how much of the budget is spent. Once nothing is free, every
    // ceiling yields so the remaining candidates drain in relevance order — a
    // capped root cannot stall the answer even when a second root is capped at
    // the same time. Equal-score ties rotate across files because each pass
    // re-visits files in stable global relevance order.
    while chosen.len() < max_results {
        // A file that is neither drained nor over a ceiling is what the
        // ceilings reserve room for. Computed once per pass: a file that
        // exhausts its candidates mid-pass only delays the lift by one pass.
        let free_anywhere = files.iter().enumerate().any(|(pos, f)| {
            per_file_cursor[pos] < f.scored.len()
                && file_consumed[pos] < soft_cap_per_file
                && root_consumed.get(&f.root_idx).copied().unwrap_or(0) < soft_cap_per_root
        });

        let mut settled = 0usize;
        for file_pos in 0..files.len() {
            if chosen.len() >= max_results {
                break;
            }
            let file = &files[file_pos];
            if per_file_cursor[file_pos] >= file.scored.len() {
                settled += 1;
                continue;
            }
            let root_capped =
                root_consumed.get(&file.root_idx).copied().unwrap_or(0) >= soft_cap_per_root;
            let file_capped = file_consumed[file_pos] >= soft_cap_per_file;
            if (root_capped || file_capped) && free_anywhere {
                // Capped while another file can still use the room this one
                // would consume.
                settled += 1;
                continue;
            }
            let line_index = file.scored[per_file_cursor[file_pos]].0;
            per_file_cursor[file_pos] += 1;
            file_consumed[file_pos] += 1;
            *root_consumed.entry(file.root_idx).or_insert(0) += 1;
            chosen.push(Candidate {
                file_pos,
                line_index,
            });
        }
        if settled == files.len() {
            // Nothing progressed and nothing is left that any ceiling allows.
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
    // Counted only when an excerpt is really written below: the previous flow
    // counted every selected line and then truncated the rendered text, so the
    // footer could report lines that the caller never received.
    let mut shown_matches = 0usize;
    let mut shown_files = 0usize;
    let total_matches_all: usize = files.iter().map(|f| f.total_matches).sum();
    // Excerpt rendering stops before the truncation notice and the footer would
    // no longer fit.
    let render_budget = MAX_OUTPUT_CHARS.saturating_sub(FOOTER_RESERVE_CHARS);
    let mut budget_exhausted = false;

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
        let hidden_here = file.total_matches.saturating_sub(lis.len());
        // One file's section is assembled in `chunk` and committed once its
        // lines are known to fit the excerpt budget, so the shown counts always
        // describe text that is really present in the answer.
        let mut chunk = String::new();
        let mut chunk_matches = 0usize;
        let mut chunk_complete = true;
        // The section header is written before any excerpt, so it is charged to
        // the same budget: a session-asset path long enough to fill the excerpt
        // budget on its own stops the rendering instead of pushing the answer
        // past it.
        let header = format!(
            "### {} match(es) in {}\n",
            lis.len(),
            &file.scan.display_path
        );
        if out.len() + header.len() > render_budget {
            budget_exhausted = true;
            break;
        }
        chunk.push_str(&header);

        // The scan pass retained only indices and scores, never the archive
        // text. Re-read the file to render the selected excerpts so at most
        // one archive file is resident at a time — and only files that
        // survived selection are re-read at all.
        let Some(content) = read_archive_text(archive, Path::new(&file.scan.display_path)) else {
            // Vanished or unreadable between scan and render (should not
            // happen within one search): fall back to a pointer, never
            // fabricate excerpts.
            chunk.push_str(
                "... [file unreadable during excerpt rendering; use read_file for surrounding context] ...\n\n",
            );
            if out.len() + chunk.len() > render_budget {
                budget_exhausted = true;
                break;
            }
            out.push_str(&chunk);
            continue;
        };
        let content_lines: Vec<&str> = content.lines().collect();
        let n_lines = content_lines.len();
        if n_lines == 0 {
            // Empty between scan and render (concurrent truncation): nothing
            // to excerpt, and indexing below would panic. Keep the header so the
            // caller sees that the file was reached but held no lines.
            chunk.push_str(
                "... [file empty during excerpt rendering; use read_file for surrounding context] ...\n\n",
            );
            if out.len() + chunk.len() > render_budget {
                budget_exhausted = true;
                break;
            }
            out.push_str(&chunk);
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

        let mut match_set: std::collections::BTreeSet<usize> = lis.iter().copied().collect();
        'lines: for (start, end) in ranges {
            for li in start..=end {
                let is_match = match_set.remove(&li);
                let line = content_lines[li];
                let window = render_line_window(
                    line,
                    if is_match {
                        first_match_offset(patterns, line)
                    } else {
                        None
                    },
                );
                let rendered = format!(
                    "{:>7}{} {}\n",
                    li + 1,
                    if is_match { ">" } else { " " },
                    window
                );
                if out.len() + chunk.len() + rendered.len() > render_budget {
                    // Whole lines only: cutting the rendered text afterwards
                    // could remove the matching text while still counting the
                    // line as shown.
                    budget_exhausted = true;
                    chunk_complete = false;
                    break 'lines;
                }
                chunk.push_str(&rendered);
                if is_match {
                    chunk_matches += 1;
                }
            }
        }
        if chunk_complete && hidden_here > 0 {
            let hint = format!(
                "... [{} more matching line(s) in this file not shown; narrow the query, raise max_results, or use file_pattern/scope] ...\n",
                hidden_here
            );
            if out.len() + chunk.len() + hint.len() <= render_budget {
                chunk.push_str(&hint);
            }
        }
        chunk.push('\n');
        out.push_str(&chunk);
        if chunk_matches > 0 {
            shown_files += 1;
            shown_matches += chunk_matches;
        }
        if !chunk_complete {
            break;
        }
    }

    // Rendering above stopped at a whole-line boundary, so nothing is cut
    // mid-line or mid-codepoint here; the notice replaces the excerpts that the
    // budget could not hold instead of retroactively dropping counted text.
    if budget_exhausted {
        out.push_str(&format!(
            "\n... [output truncated at character limit; {} selected matching line(s) not written; narrow the query, scope, or file_pattern] ...\n",
            chosen.len().saturating_sub(shown_matches)
        ));
    }
    out.push_str(&format!(
        "[archive search] showed {} matching line(s) across {} file(s); {} additional matching line(s) hidden (corpus total {}, files scanned {}). Use `read_file` on any listed absolute path for surrounding context.\n",
        shown_matches,
        shown_files,
        total_matches_all.saturating_sub(shown_matches),
        total_matches_all,
        files_seen
    ));
    debug_assert!(
        out.len() <= MAX_OUTPUT_CHARS,
        "the excerpt budget must leave room for the notice and the footer"
    );
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

    /// Shown count parsed from the footer, so tests can assert that the
    /// accounting line describes the markers actually rendered.
    fn shown_count(out: &str) -> usize {
        out.rsplit_once("[archive search] showed ")
            .and_then(|(_, rest)| rest.split_whitespace().next())
            .and_then(|value| value.parse().ok())
            .expect("footer must report how many lines were shown")
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
    fn archive_search_finds_checkpoint_only_bodies_in_both_scopes() {
        let dir = make_temp_dir();
        fs::create_dir_all(dir.join("context-checkpoints")).unwrap();
        fs::create_dir_all(dir.join("tool-overflow")).unwrap();
        fs::write(
            dir.join("context-checkpoints/opaque.md"),
            "header\nrare_checkpoint_keyword decision\n",
        )
        .unwrap();
        fs::write(dir.join("tool-overflow/live.txt"), "rare_tool_keyword result\n").unwrap();
        for scope in [SearchScope::All, SearchScope::ToolOutputs] {
            let mut p = params("rare_checkpoint_keyword");
            p.scope = scope;
            let out = run_overflow_search(&dir, &p).unwrap();
            assert!(out.contains("context-checkpoints/opaque.md"), "{out}");
            assert!(out.contains("2> rare_checkpoint_keyword decision"), "{out}");
            p.query = "rare_tool_keyword";
            assert!(run_overflow_search(&dir, &p).unwrap().contains("tool-overflow/live.txt"));
        }
        let mut p = params("rare_checkpoint_keyword");
        p.scope = SearchScope::History;
        assert!(run_overflow_search(&dir, &p).unwrap().contains("No matches found"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn archive_search_large_history_keeps_late_matches() {
        let dir = make_temp_dir();
        // The only hit occurs after 4 MiB; neither a file-size skip nor a prefix
        // scan can recover it. Wide filler keeps the regression inexpensive.
        let filler_lines = 4_097;
        let mut body = format!("{}\n", "x".repeat(1_023)).repeat(filler_lines);
        body.push_str("late_history_keyword recovered decision\n");
        assert!(body.len() > 4 * 1024 * 1024);
        fs::write(dir.join("overflow-history.md"), body).unwrap();
        for scope in [SearchScope::History, SearchScope::All] {
            let mut p = params("late_history_keyword");
            p.scope = scope;
            p.context_lines = 0;
            let out = run_overflow_search(&dir, &p).unwrap();
            assert!(out.contains("4098> late_history_keyword recovered decision"), "{out}");
            assert!(out.contains("files scanned 1"), "{out}");
        }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn archive_search_file_pattern_reaches_late_and_deep_paths() {
        let dir = make_temp_dir();
        let snapshots = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&snapshots).unwrap();
        // Filtering must reach the wanted file even when more than 1,024
        // unrelated files precede it; a capped candidate prefix would hide it.
        for index in 0..1_025 {
            fs::write(snapshots.join(format!("a-{index:04}.txt")), "noise\n").unwrap();
        }
        fs::write(snapshots.join("zz-target.txt"), "late_path_keyword evidence\n").unwrap();
        let deep = dir.join("context-checkpoints/a/b/c/d/e/f/g/h/i");
        fs::create_dir_all(&deep).unwrap();
        fs::write(deep.join("zz-target.txt"), "late_path_keyword checkpoint\n").unwrap();
        let mut p = params("late_path_keyword");
        p.file_pattern = Some("zz-target.txt");
        for scope in [SearchScope::All, SearchScope::ToolOutputs] {
            p.scope = scope;
            let out = run_overflow_search(&dir, &p).unwrap();
            assert!(out.contains("tool-overflow-compressed/zz-target.txt"), "{out}");
            assert!(out.contains("context-checkpoints/a/b/c/d/e/f/g/h/i/zz-target.txt"), "{out}");
            assert!(out.contains("files scanned 2"), "{out}");
        }
        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn archive_search_collects_nested_files_without_internal_symlinks() {
        use std::os::unix::fs::symlink;
        let dir = make_temp_dir();
        let snapshots = dir.join("tool-overflow-compressed");
        let nested = snapshots.join("nested/deeper");
        fs::create_dir_all(&nested).unwrap();
        let first = snapshots.join("a.txt");
        let second = nested.join("b.txt");
        fs::write(&first, "nested_keyword first\n").unwrap();
        fs::write(&second, "nested_keyword second\n").unwrap();
        symlink(&second, snapshots.join("file-link.txt")).unwrap();
        symlink(&nested, snapshots.join("directory-link")).unwrap();
        symlink(&snapshots, nested.join("loop")).unwrap();

        let archive = archive_files::ArchiveRoot::new(&dir).unwrap();
        for _ in 0..2 {
            assert_eq!(
                collect_files(&archive, &snapshots),
                vec![first.clone(), second.clone()]
            );
        }
        let out = run_overflow_search(&dir, &params("nested_keyword")).unwrap();
        assert!(out.contains("nested/deeper/b.txt"), "{out}");
        assert!(out.contains("files scanned 2"), "{out}");
        assert!(!out.contains("file-link"), "{out}");
        assert!(!out.contains("directory-link"), "{out}");
        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn archive_search_rejects_outside_file_directory_and_root_symlinks() {
        use std::os::unix::fs::symlink;
        let dir = make_temp_dir();
        let outside = make_temp_dir();
        fs::create_dir_all(dir.join("context-checkpoints")).unwrap();
        fs::write(outside.join("external.md"), "external_secret_keyword\n").unwrap();
        symlink(outside.join("external.md"), dir.join("context-checkpoints/link.md")).unwrap();
        symlink(&outside, dir.join("context-checkpoints/nested")).unwrap();
        symlink(&outside, dir.join("tool-overflow")).unwrap();
        symlink(outside.join("external.md"), dir.join("overflow-history.md")).unwrap();
        let out = run_overflow_search(&dir, &params("external_secret_keyword")).unwrap();
        assert!(out.contains("No matches found"), "{out}");
        assert!(!out.contains("external.md"), "{out}");
        let root_link = dir.join("session-link");
        symlink(&outside, &root_link).unwrap();
        let out = run_overflow_search(&root_link, &params("external_secret_keyword")).unwrap();
        assert!(out.contains("nothing was scanned"), "{out}");
        fs::remove_dir_all(&dir).ok();
        fs::remove_dir_all(&outside).ok();
    }

    #[cfg(unix)]
    #[test]
    fn archive_search_render_rejects_symlink_swapped_after_scan() {
        let dir = make_temp_dir();
        let outside = make_temp_dir();
        let archive = archive_files::ArchiveRoot::new(&dir).unwrap();
        let path = dir.join("overflow-history.md");
        fs::write(&path, "needle original\n").unwrap();
        assert!(read_archive_text(&archive, &path).is_some());
        let file = ScoredFile {
            root_idx: 0,
            scan: FileScan {
                root_idx: 0,
                display_path: path.to_string_lossy().into_owned(),
                hits: Vec::new(),
                total_lines: 1,
            },
            file_score: 1.0,
            scored: vec![(0, 1.0)],
            total_matches: 1,
        };
        fs::rename(&path, dir.join("saved.md")).unwrap();
        fs::write(outside.join("external.md"), "needle external_secret_body\n").unwrap();
        std::os::unix::fs::symlink(outside.join("external.md"), &path).unwrap();
        let needle = params("needle");
        let patterns = build_patterns(&needle).unwrap();
        let out = render_selection(vec![file], &needle, &patterns, 1, &archive).unwrap();
        assert!(!out.contains("external_secret_body"), "{out}");
        assert!(out.contains("file unreadable during excerpt rendering"), "{out}");
        fs::remove_dir_all(&dir).ok();
        fs::remove_dir_all(&outside).ok();
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
    fn two_capped_roots_share_the_leftover_budget() {
        // Regression: with max_results=40 over five hit roots, fair_share = 8
        // and the soft cap is 16 per root. Both dominant roots reach the cap
        // while each still holds candidates, so the old `active_roots.len() > 1`
        // valve benched every file at once and the pass ended with
        // settled == files.len(). Restoring that valve makes this test report
        // "showed 35 matching line(s) ... 168 additional ... (corpus total 203)"
        // instead of the requested 40.
        let dir = make_temp_dir();
        fs::write(dir.join("overflow-history.md"), &"alpha\n".repeat(100)).unwrap();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        fs::write(tool_dir.join("noisy.txt"), &"alpha\n".repeat(100)).unwrap();
        for (sub, name) in [
            ("folded-tool-groups", "a.md"),
            ("internal-note-overflow", "a.md"),
            ("tool-overflow", "a.txt"),
        ] {
            let root = dir.join(sub);
            fs::create_dir_all(&root).unwrap();
            fs::write(root.join(name), "alpha minority-marker\n").unwrap();
        }

        let mut p = params("alpha");
        p.context_lines = 0;
        p.max_results = 40;
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(
            out.contains("[archive search] showed 40 matching line(s)"),
            "two capped roots must keep draining until the budget is used:\n{out}"
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
    fn oversized_multibyte_output_stops_at_budget_without_panicking() {
        // Regression: archived content is arbitrary UTF-8 while MAX_OUTPUT_CHARS
        // is a byte cap. Rendering must stop at whole-line boundaries instead of
        // cutting a string mid-codepoint (the release profile aborts) or cutting
        // away matches that the footer already counted as shown.
        let dir = make_temp_dir();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        // One matching line per file, each longer than the per-line window; the
        // CJK payload straddles the byte cap regardless of gutter width.
        for index in 0..MAX_SNIPPETS_PER_FILE + 4 {
            let line = format!("needle {index} {}\n", "好".repeat(1_000));
            fs::write(tool_dir.join(format!("wide-{index:02}.txt")), line).unwrap();
        }

        let mut p = params("needle");
        p.scope = SearchScope::ToolOutputs;
        p.context_lines = 0;
        p.max_results = MAX_MATCHES;
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(
            out.contains("output truncated at character limit"),
            "test must exercise the budget stop:\n{out}"
        );
        let shown = shown_count(&out);
        assert!(shown >= 1, "at least one windowed line must render:\n{out}");
        assert_eq!(
            shown,
            out.matches("> ").count(),
            "reported shown count must match the rendered markers:\n{out}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn archive_search_covers_summary_source_records() {
        // The incremental-summary path archives the removed span *only* under
        // summary-sources/ (plain compression writes overflow-history.md
        // instead), so without that root the originals behind a compressed
        // summary were unreachable from this tool. It holds original folded
        // messages, hence the history side of the scope split.
        let dir = make_temp_dir();
        let source_dir = dir.join("summary-sources");
        fs::create_dir_all(&source_dir).unwrap();
        let record = serde_json::json!({
            "role": "user",
            "content": "incremental_source_keyword: keep the original wording",
        });
        fs::write(source_dir.join("deadbeef.jsonl"), format!("{record}\n")).unwrap();

        for scope in [SearchScope::All, SearchScope::History] {
            let mut p = params("incremental_source_keyword");
            p.scope = scope;
            let out = run_overflow_search(&dir, &p).unwrap();
            assert!(
                out.contains("summary-sources/deadbeef.jsonl"),
                "{scope:?} must search summary source records:\n{out}"
            );
            assert!(out.contains("incremental_source_keyword"), "{out}");
        }
        let mut p = params("incremental_source_keyword");
        p.scope = SearchScope::ToolOutputs;
        assert!(
            run_overflow_search(&dir, &p)
                .unwrap()
                .contains("No matches found"),
            "tool_outputs keeps skipping original messages"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn windowed_long_line_keeps_the_match_visible() {
        // Regression: one unbounded line used to be emitted whole, and the
        // output cap then cut it from the front — removing the very term that
        // matched while still counting the line as shown.
        let dir = make_temp_dir();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        let line = format!(
            "{}needle_tail_keyword{}\n",
            "x".repeat(20_000),
            "y".repeat(20_000)
        );
        fs::write(tool_dir.join("long.txt"), line).unwrap();

        let mut p = params("needle_tail_keyword");
        p.scope = SearchScope::ToolOutputs;
        p.context_lines = 0;
        p.max_results = 1;
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(
            out.contains("needle_tail_keyword"),
            "the matched term must stay inside the window:\n{out}"
        );
        assert!(out.contains('…'), "elision must be marked:\n{out}");
        assert!(
            !out.contains(&"x".repeat(MAX_RENDERED_LINE_CHARS + 1)),
            "line must be windowed, not emitted whole:\n{out}"
        );
        assert!(
            !out.contains(&"y".repeat(MAX_RENDERED_LINE_CHARS + 1)),
            "line must be windowed, not emitted whole:\n{out}"
        );
        assert_eq!(shown_count(&out), 1, "{out}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn per_file_ceiling_lifts_when_no_other_file_has_candidates() {
        // Regression: the twelfth line used to be the last reachable one from a
        // single file even with max_results=200, and the hint merely suggested
        // raising a budget that could not expose the surplus.
        let dir = make_temp_dir();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        let body: String = (1..=MAX_SNIPPETS_PER_FILE + 1)
            .map(|index| format!("needle line {index}\n"))
            .collect();
        fs::write(tool_dir.join("many.txt"), body).unwrap();

        let mut p = params("needle");
        p.scope = SearchScope::ToolOutputs;
        p.context_lines = 0;
        p.max_results = MAX_MATCHES;
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(
            out.contains(&format!("needle line {}", MAX_SNIPPETS_PER_FILE + 1)),
            "a line past the old per-file ceiling must be reachable:\n{out}"
        );
        assert_eq!(shown_count(&out), MAX_SNIPPETS_PER_FILE + 1, "{out}");
        assert!(!out.contains("not shown"), "nothing is hidden here:\n{out}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn zero_hit_search_reports_terms_and_retry_hints() {
        // A bare "No matches found" reads like proof of absence; the answer must
        // say what was searched and how to widen the search.
        let dir = make_temp_dir();
        seed_archive(&dir);
        let out = run_overflow_search(&dir, &params("zzz_absent")).unwrap();
        assert!(out.contains("No matches found"), "{out}");
        assert!(out.contains("file(s) were searched"), "{out}");
        assert!(out.contains("'zzz_absent'"), "{out}");
        assert!(out.contains("case_sensitive=false"), "{out}");
        assert!(!out.contains("Nothing was scanned"), "{out}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unscanned_archive_reports_probed_roots() {
        // Nothing reachable: say so explicitly instead of implying that the
        // archived content does not exist.
        let dir = make_temp_dir();
        fs::create_dir_all(dir.join("tool-overflow-compressed")).unwrap();
        let mut p = params("foo");
        p.file_pattern = Some("*absent*");
        let out = run_overflow_search(&dir, &p).unwrap();
        assert!(out.contains("No matches found"), "{out}");
        assert!(out.contains("Nothing was scanned"), "{out}");
        assert!(out.contains("file_pattern '*absent*'"), "{out}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn footer_counts_match_rendered_markers() {
        // Every line reported as shown must really be present, in each regime:
        // budget-limited, per-file-limited, and context-expanded.
        let dir = make_temp_dir();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        fs::write(tool_dir.join("big.txt"), "needle\n".repeat(80)).unwrap();
        fs::write(tool_dir.join("other.txt"), "needle\n".repeat(40)).unwrap();

        for (max_results, context_lines) in [(10usize, 0usize), (50, 2), (MAX_MATCHES, 1)] {
            let mut p = params("needle");
            p.scope = SearchScope::ToolOutputs;
            p.context_lines = context_lines;
            p.max_results = max_results;
            let out = run_overflow_search(&dir, &p).unwrap();
            assert_eq!(
                shown_count(&out),
                out.matches("> ").count(),
                "max_results={max_results} context_lines={context_lines}:\n{out}"
            );
        }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ascii_only_query_gains_no_derived_patterns() {
        // Regression guard for the CJK expansion: an ASCII query must still
        // produce exactly the pattern set it produced before expansion existed.
        let patterns = build_patterns(&params("foo bar")).unwrap();
        let summary: Vec<(&str, Option<usize>, f64)> = patterns
            .iter()
            .map(|p| (p.source.as_str(), p.term_id, p.weight))
            .collect();
        assert_eq!(
            summary,
            vec![
                ("foo", Some(0), 1.0),
                ("bar", Some(1), 1.0),
                ("foo bar", None, 1.0),
            ],
            "ASCII query pattern set changed"
        );
    }

    #[test]
    fn regex_query_gains_no_cjk_expansion() {
        // Expansion is a literal-mode recall aid: a regex query must keep its
        // single caller-authored pattern (build_patterns returns early).
        let mut p = params("压缩.文档");
        p.is_regex = true;
        let patterns = build_patterns(&p).unwrap();
        assert_eq!(patterns.len(), 1, "regex query gained derived patterns");
        assert_eq!(patterns[0].source, "压缩.文档");
        assert!(!is_derived_pattern(&patterns[0]));
    }

    #[test]
    fn cjk_run_boundaries_follow_is_cjk() {
        // Hangul is written with spaces, so Korean text is not a CJK run and
        // falls back to plain whitespace terms.
        let hangul = build_patterns(&params("문서")).unwrap();
        assert!(
            !hangul.iter().any(is_derived_pattern),
            "Hangul must not expand"
        );
        // Kana has no inter-word spaces either, so it does expand.
        let kana = build_patterns(&params("ドキュメント")).unwrap();
        assert!(kana.iter().any(is_derived_pattern), "kana run must expand");
        // Non-CJK characters break a run and a 2-character run has no useful
        // bigram, so only the 3-character run contributes here.
        let mixed = build_patterns(&params("压缩的abc文档")).unwrap();
        let derived: Vec<&str> = mixed
            .iter()
            .filter(|p| is_derived_pattern(p))
            .map(|p| p.source.as_str())
            .collect();
        assert_eq!(derived, vec!["压缩", "缩的"], "run split or short-run leak");
    }

    #[test]
    fn cjk_query_recalls_text_with_function_words_elided() {
        let dir = make_temp_dir();
        fs::write(
            dir.join("overflow-history.md"),
            "## User\n部署时需要保证压缩文档的完整性\n",
        )
        .unwrap();
        // "压缩的文档" is a single literal term, so without bigram expansion it can
        // never match this line: the archive never holds the phrase as typed.
        let out = run_overflow_search(&dir, &params("压缩的文档")).unwrap();
        assert!(out.contains("压缩文档的完整性"), "{out}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cjk_literal_run_outranks_a_bigram_only_line() {
        let dir = make_temp_dir();
        let tool_dir = dir.join("tool-overflow-compressed");
        fs::create_dir_all(&tool_dir).unwrap();
        fs::write(tool_dir.join("literal-run.txt"), "这里讨论压缩的文档结构\n").unwrap();
        fs::write(tool_dir.join("bigram-only.txt"), "文档目录与压缩无关的说明\n").unwrap();
        let out = run_overflow_search(&dir, &params("压缩的文档")).unwrap();
        let literal_at = out
            .find("literal-run.txt")
            .expect("the file holding the full run must be shown");
        let bigram_at = out
            .find("bigram-only.txt")
            .expect("the fragment-only file must still be reachable");
        assert!(
            literal_at < bigram_at,
            "the full run must rank above a shared fragment:\n{out}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cjk_bigram_hit_keeps_the_matching_text_inside_the_window() {
        let dir = make_temp_dir();
        // 800 filler characters push the match past MAX_RENDERED_LINE_CHARS, so
        // the excerpt is windowed; the bigram that matched must stay visible,
        // exactly like a literal term hit does.
        let filler = "无关内容".repeat(200);
        fs::write(
            dir.join("overflow-history.md"),
            format!("## User\n{filler}压缩文档的完整性校验\n"),
        )
        .unwrap();
        let out = run_overflow_search(&dir, &params("压缩的文档")).unwrap();
        assert!(out.contains('…'), "long line must be windowed:\n{out}");
        assert!(
            out.contains("压缩文档"),
            "the matching run must survive windowing:\n{out}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn short_cjk_runs_gain_no_bigram_expansion() {
        // A one- or two-character run has no adjacent pair worth adding: its
        // only bigram is the run itself.
        let patterns = build_patterns(&params("压缩 文档 的")).unwrap();
        let derived: Vec<&str> = patterns
            .iter()
            .filter(|p| is_derived_pattern(p))
            .map(|p| p.source.as_str())
            .collect();
        assert!(derived.is_empty(), "unexpected expansion: {derived:?}");
    }

    #[test]
    fn cjk_expansion_is_deduplicated_and_capped() {
        // Repeating one ten-character block reuses the same adjacent pairs, so
        // only ten distinct bigrams appear no matter how long the query grows.
        let repeated = "一二三四五六七八九十".repeat(4);
        let patterns = build_patterns(&params(&repeated)).unwrap();
        let derived = patterns.iter().filter(|p| is_derived_pattern(p)).count();
        assert_eq!(derived, 10, "repeated pairs must be deduplicated");

        // Forty distinct consecutive Han code points yield 39 distinct bigrams,
        // which the cap trims: the per-line pattern fan-out must not scale with
        // query length.
        let distinct: String = (0x4E00u32..0x4E00 + 40)
            .filter_map(char::from_u32)
            .collect();
        let patterns = build_patterns(&params(&distinct)).unwrap();
        let derived = patterns.iter().filter(|p| is_derived_pattern(p)).count();
        assert_eq!(derived, MAX_CJK_BIGRAMS, "expansion must stop at the cap");
    }

    #[test]
    fn zero_hit_cjk_answer_reports_the_expansion_separately() {
        let dir = make_temp_dir();
        seed_archive(&dir);
        let out = run_overflow_search(&dir, &params("压缩的文档")).unwrap();
        assert!(out.contains("No matches found"), "{out}");
        assert!(
            out.contains(
                "terms: '压缩的文档'; 4 derived CJK bigram pattern(s) also came up empty"
            ),
            "the typed term is listed and the expansion disclosed separately:\n{out}"
        );
        fs::remove_dir_all(&dir).ok();
    }
}
