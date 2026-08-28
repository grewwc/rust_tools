use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

use regex::{Regex, RegexBuilder};
#[cfg(test)]
use serde_json::Value;

const MAX_OUTPUT_CHARS: usize = 32_000;
const MAX_FILE_SIZE: u64 = 2 * 1024 * 1024;
const MAX_MATCHES: usize = 200;
const MAX_WALK_FILES: usize = 10_000;
/// How many snippets to keep per file at most (top N after relevance sorting).
const MAX_SNIPPETS_PER_FILE: usize = 3;

/// System-level directory blocklist: search roots inside these prefixes are
/// always rejected, to avoid accidentally scanning the whole disk. Only explicit
/// "system/platform" directories are listed; `/var`, `/private`, and `/tmp` are
/// deliberately excluded, because macOS temp dirs live under `/var/folders/...`
/// (after canonicalize: `/private/var/folders/...`) and including them would
/// reject legitimate temporary working directories.
const FORBIDDEN_ROOT_PREFIXES: &[&str] = &[
    "/System",
    "/Library",
    "/usr",
    "/bin",
    "/sbin",
    "/dev",
    "/proc",
    "/sys",
    "/etc",
    "/Applications",
    "/cores",
    "/Network",
];

/// Validate the search root, rejecting the filesystem root `/` and system-level
/// directories.
///
/// Design goal: stop LLM mistakes like `path="/"` or `path="/System"` that would
/// trigger a full-disk scan at 100% CPU. `root` must already be absolute (the
/// caller joins cwd first).
pub(crate) fn validate_search_root(root: &Path, cwd: &Path) -> Result<(), String> {
    // 1. Reject the filesystem root (`/` or a Windows drive root).
    let component_count = root.components().count();
    if component_count <= 1 {
        return Err(format!(
            "Refusing to search filesystem root '{}'. Pass a path inside the current project (cwd: {}).",
            root.display(),
            cwd.display()
        ));
    }

    // 2. Reject system-level prefixes. Prefer canonicalize for the comparison;
    // fall back to a literal comparison on failure.
    let canonical = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let canonical_str = canonical.to_string_lossy();
    for prefix in FORBIDDEN_ROOT_PREFIXES {
        if canonical_str == *prefix || canonical_str.starts_with(&format!("{}/", prefix)) {
            return Err(format!(
                "Refusing to search system path '{}'. Pass a path inside the current project (cwd: {}).",
                root.display(),
                cwd.display()
            ));
        }
    }

    Ok(())
}

// ============================================================================
// Shared content search engine
//
// `run_content_search` is the content-search core of the text search tools:
// BFS-collect files → match line by line → rerank by relevance → aggregate per
// file (top-N snippets + context + `>` markers on matched lines per file).
// Plain case-sensitive literal queries go straight through `str::find`; only
// regex and case-insensitive queries build a Regex.
//
// Design notes:
// - File collection uses BFS (keeping the `*.rs` recursive semantics;
//   `terminalw::glob_paths` is non-recursive and cannot replace it directly).
// - `extensions=None` means no extension filtering (text_search's default
//   behavior); `Some(&[...])` means only whitelisted extensions are searched
//   (reserved for future per-language filtering).
// - Relevance scoring: whole-word hits > substring hits; exact case match is
//   preferred; files whose name/path matches get a whole-file bonus; earlier
//   in-line matches are preferred.
// ============================================================================

/// Configurable knobs for content search. Constructed separately by each caller.
pub(crate) struct ContentSearchOptions<'a> {
    /// The raw query string (used to detect literal case and whole-word hits
    /// during relevance scoring).
    pub(crate) query: &'a str,
    /// Whether to treat the query as a regex. When false, matches as a literal
    /// (escaped) substring.
    pub(crate) is_regex: bool,
    /// Whether matching is case-sensitive.
    pub(crate) case_sensitive: bool,
    /// How many context lines to keep on each side of a matched line.
    pub(crate) context_lines: usize,
    /// Maximum number of matched lines returned (across all files).
    pub(crate) max_results: usize,
    /// Optional file-name glob filter (supports comma-separated patterns and
    /// `*.{ts,tsx}` brace expansion).
    pub(crate) file_pattern: Option<&'a str>,
    /// Optional extension whitelist. None = no extension filtering; Some = only
    /// these extensions.
    pub(crate) extensions: Option<&'a [&'a str]>,
    /// Prefix stripped for display paths (usually cwd) so output uses relative paths.
    pub(crate) display_root: Option<&'a Path>,
    /// Per-file size cap in bytes; larger files are skipped. Plain search uses
    /// 2 MiB; session archive searches raise it because overflow files can be far
    /// larger than ordinary source files.
    pub(crate) max_file_size: u64,
}

/// One matched line, together with its in-file relevance score.
struct ScoredLine {
    line_index: usize,
    score: i64,
}

/// Aggregated result for one file (snippet lines already sorted by relevance and
/// capped).
struct FileHits {
    /// Path string used for display (possibly relativized).
    display_path: String,
    /// Whole-file relevance score (used for ordering between files).
    file_score: i64,
    /// Matched lines (line_index, score), sorted by relevance descending.
    scored: Vec<ScoredLine>,
    /// **All** matched line indices in the file (ascending), so rendering can
    /// mark `>` correctly. Note `scored` keeps only the top-N snippets, but other
    /// hits that fall inside a context window must also be marked `>`; otherwise
    /// they would be wrongly shown as ordinary context lines.
    all_match_indices: Vec<usize>,
    /// Raw file content. Kept only for files with hits; context rendering borrows
    /// slices by line start instead of allocating a String for every scanned line.
    content: String,
    /// Line-start byte offsets with the same semantics as `str::lines()`.
    line_starts: Vec<usize>,
}

enum SearchMatcher {
    /// Default path: case-sensitive literal substring search; no regex is built or run.
    Literal,
    /// Regex queries and case-insensitive literal queries still go through the
    /// regex crate to preserve semantics.
    Regex(Regex),
}

impl SearchMatcher {
    fn new(options: &ContentSearchOptions<'_>) -> Result<Self, String> {
        if !options.is_regex && options.case_sensitive {
            Ok(Self::Literal)
        } else {
            build_regex(options.query, options.is_regex, options.case_sensitive).map(Self::Regex)
        }
    }

    fn find(&self, line: &str, query: &str) -> Option<(usize, usize)> {
        match self {
            Self::Literal => line.find(query).map(|start| (start, start + query.len())),
            Self::Regex(regex) => regex
                .find(line)
                .map(|matched| (matched.start(), matched.end())),
        }
    }
}

/// Run the shared content search and return the fully formatted result string
/// (including truncation).
/// With no hits it returns `Ok("No matches found.")` so callers can wrap it in
/// their own semantics.
pub(crate) fn run_content_search(
    root: &Path,
    options: &ContentSearchOptions<'_>,
) -> Result<String, String> {
    if options.query.is_empty() {
        return Err("pattern must not be empty".to_string());
    }

    let matcher = SearchMatcher::new(options)?;
    let glob_matcher = options.file_pattern.map(build_glob_matcher);
    let files = collect_content_files(root, glob_matcher.as_ref(), options.extensions)?;

    let mut file_hits: Vec<FileHits> = Vec::new();
    let mut total_matches = 0usize;

    for file_path in &files {
        if total_matches >= options.max_results {
            break;
        }

        let metadata = match fs::metadata(file_path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if metadata.len() > options.max_file_size {
            continue;
        }

        let content = match fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let display_path = display_path_for(file_path, options.display_root);
        let name_path_bonus = path_match_bonus(&display_path, options);

        let mut scored: Vec<ScoredLine> = Vec::new();
        for (idx, line) in content.lines().enumerate() {
            if total_matches >= options.max_results {
                break;
            }
            let Some((match_start, match_end)) = matcher.find(line, options.query) else {
                continue;
            };
            let score = score_line(line, options, match_start, match_end) + name_path_bonus;
            scored.push(ScoredLine {
                line_index: idx,
                score,
            });
            total_matches += 1;
        }

        if scored.is_empty() {
            continue;
        }

        // Collect all matched line indices (ascending) before truncation, for `>`
        // marking at render time.
        let all_match_indices: Vec<usize> = scored.iter().map(|s| s.line_index).collect();

        // File score = its best hit score + path-hit bonus; used as the inter-file
        // sort key.
        let file_score = scored.iter().map(|s| s.score).max().unwrap_or(0) + name_path_bonus;
        // Within the file: relevance descending, then line index ascending
        // (stable, nearest first).
        scored.sort_by(|a, b| {
            b.score
                .cmp(&a.score)
                .then_with(|| a.line_index.cmp(&b.line_index))
        });
        scored.truncate(MAX_SNIPPETS_PER_FILE);

        file_hits.push(FileHits {
            display_path,
            file_score,
            scored,
            all_match_indices,
            line_starts: collect_line_starts(&content),
            content,
        });
    }

    if file_hits.is_empty() {
        return Ok("No matches found.".to_string());
    }

    // Between files: relevance descending; equal scores ordered stably by path
    // lexicographically.
    file_hits.sort_by(|a, b| {
        b.file_score
            .cmp(&a.file_score)
            .then_with(|| a.display_path.cmp(&b.display_path))
    });

    let output = format_content_results(
        &file_hits,
        total_matches,
        options.max_results,
        options.context_lines,
    );
    Ok(truncate_output(&output, MAX_OUTPUT_CHARS))
}

fn build_regex(query: &str, is_regex: bool, case_sensitive: bool) -> Result<Regex, String> {
    if is_regex {
        RegexBuilder::new(query)
            .case_insensitive(!case_sensitive)
            .build()
            .map_err(|e| format!("Invalid regex: {}", e))
    } else {
        let escaped = regex::escape(query);
        RegexBuilder::new(&escaped)
            .case_insensitive(!case_sensitive)
            .build()
            .map_err(|e| format!("Internal regex error: {}", e))
    }
}

/// Display path: relativize when possible, otherwise the full path.
fn display_path_for(file_path: &Path, display_root: Option<&Path>) -> String {
    if let Some(root) = display_root {
        if let Ok(rel) = file_path.strip_prefix(root) {
            return rel.to_string_lossy().to_string();
        }
    }
    file_path.to_string_lossy().to_string()
}

/// Whole-file bonus when the query matches the file name/path (+3 for a file-name
/// hit, +1 for a directory-path hit).
fn path_match_bonus(display_path: &str, options: &ContentSearchOptions<'_>) -> i64 {
    if options.is_regex {
        return 0;
    }
    let needle = options.query;
    let (hay, needle) = if options.case_sensitive {
        (display_path.to_string(), needle.to_string())
    } else {
        (display_path.to_lowercase(), needle.to_lowercase())
    };
    if needle.is_empty() || !hay.contains(&needle) {
        return 0;
    }
    let file_name = Path::new(&hay)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if file_name.contains(&needle) { 3 } else { 1 }
}

/// Score a single matched line: whole-word hit +4; exact literal-case match +2;
/// matches closer to the line start rank higher (up to +2).
fn score_line(
    line: &str,
    options: &ContentSearchOptions<'_>,
    match_start: usize,
    match_end: usize,
) -> i64 {
    let mut score = 1; // base hit score
    let matched = &line[match_start..match_end];

    // Whole-word hit bonus.
    let left_ok =
        match_start == 0 || !is_identifier_byte(line.as_bytes()[match_start.saturating_sub(1)]);
    let right_ok = match_end >= line.len() || !is_identifier_byte(line.as_bytes()[match_end]);
    if left_ok && right_ok {
        score += 4;
    }

    // For literal (non-regex) queries, an exact-case match against the query
    // adds another bonus.
    if !options.is_regex && matched == options.query {
        score += 2;
    }

    // Proximity: the earlier the match, the better.
    let lead = line[..match_start]
        .chars()
        .filter(|c| !c.is_whitespace())
        .count();
    score += match lead {
        0 => 2,
        1..=8 => 1,
        _ => 0,
    };

    score
}

fn is_identifier_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Collect line starts with the same semantics as `str::lines()`. A trailing
/// newline does not produce an extra empty line.
fn collect_line_starts(content: &str) -> Vec<usize> {
    if content.is_empty() {
        return Vec::new();
    }

    let mut starts = Vec::new();
    starts.push(0);
    for (index, _) in content.match_indices('\n') {
        let next = index + 1;
        if next < content.len() {
            starts.push(next);
        }
    }
    starts
}

/// Return the line slice without newlines for a given line start, stripping the
/// CR of CRLF just like `str::lines()`.
fn line_at<'a>(content: &'a str, starts: &[usize], index: usize) -> Option<&'a str> {
    let start = *starts.get(index)?;
    let mut end = starts
        .get(index + 1)
        .map_or(content.len(), |next| next.saturating_sub(1));
    if index + 1 == starts.len() && content.as_bytes().last() == Some(&b'\n') {
        end = end.saturating_sub(1);
    }
    let line = content.get(start..end)?;
    Some(line.strip_suffix('\r').unwrap_or(line))
}

fn format_content_results(
    file_hits: &[FileHits],
    total_matches: usize,
    max_results: usize,
    context_lines: usize,
) -> String {
    let mut out = String::new();
    let file_count = file_hits.len();

    out.push_str(&format!(
        "{} match(es) in {} file(s)",
        total_matches, file_count
    ));
    if total_matches >= max_results {
        out.push_str(" (limit reached, more matches may exist)");
    }
    out.push('\n');

    for hit in file_hits {
        out.push('\n');
        out.push_str(&hit.display_path);
        out.push('\n');

        // Render snippet lines in ascending line order (relevance was already used
        // for capping and file ordering).
        let mut match_indices: Vec<usize> = hit.scored.iter().map(|s| s.line_index).collect();
        match_indices.sort_unstable();

        let ranges = merge_context_ranges(&match_indices, context_lines, hit.line_starts.len());
        for range in &ranges {
            if range.start > 0 {
                out.push_str("...\n");
            }
            for idx in range.start..range.end {
                let line_num = idx + 1;
                // Mark `>` using all matched line indices, so a truncated hit falling
                // inside a context window is not mistaken for an ordinary context line.
                let is_match = hit.all_match_indices.binary_search(&idx).is_ok();
                let prefix = if is_match { ">" } else { " " };
                let line_content = line_at(&hit.content, &hit.line_starts, idx).unwrap_or("");
                out.push_str(&format!("{}{:>5}| {}\n", prefix, line_num, line_content));
            }
        }
    }

    out
}

struct LineRange {
    start: usize,
    end: usize,
}

fn merge_context_ranges(
    match_indices: &[usize],
    context: usize,
    total_lines: usize,
) -> Vec<LineRange> {
    if match_indices.is_empty() {
        return Vec::new();
    }

    let mut ranges: Vec<LineRange> = Vec::new();
    for &line_index in match_indices {
        let start = line_index.saturating_sub(context);
        let end = (line_index + context + 1).min(total_lines);

        if let Some(last) = ranges.last_mut() {
            if start <= last.end {
                last.end = last.end.max(end);
                continue;
            }
        }
        ranges.push(LineRange { start, end });
    }

    ranges
}

// ----------------------------------------------------------------------------
// File collection + glob matching
// ----------------------------------------------------------------------------

fn build_glob_matcher(pattern: &str) -> GlobMatcher {
    let mut patterns = Vec::new();
    // First split on "top-level commas" (commas not inside `{}`) into multiple
    // globs, then brace-expand each one; otherwise `*.{ts,tsx}` would be wrongly
    // split on the comma inside the braces.
    for part in split_top_level_commas(pattern) {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Brace expansion: `*.{ts,tsx}` → `*.ts`, `*.tsx`
        for expanded in expand_braces(trimmed) {
            patterns.push(expanded);
        }
    }
    GlobMatcher { patterns }
}

/// Split on commas not inside `{}`, so `a,*.{ts,tsx}` → [`a`, `*.{ts,tsx}`].
fn split_top_level_commas(pattern: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32;
    for ch in pattern.chars() {
        match ch {
            '{' => {
                depth += 1;
                current.push(ch);
            }
            '}' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                parts.push(std::mem::take(&mut current));
            }
            _ => current.push(ch),
        }
    }
    parts.push(current);
    parts
}

/// Expand a single one-level `{a,b,c}` brace (only one brace site supported,
/// enough for patterns like `*.{ts,tsx}`).
fn expand_braces(pattern: &str) -> Vec<String> {
    let (Some(open), Some(close)) = (pattern.find('{'), pattern.find('}')) else {
        return vec![pattern.to_string()];
    };
    if close < open {
        return vec![pattern.to_string()];
    }
    let prefix = &pattern[..open];
    let suffix = &pattern[close + 1..];
    let inner = &pattern[open + 1..close];
    inner
        .split(',')
        .map(|alt| format!("{}{}{}", prefix, alt.trim(), suffix))
        .collect()
}

struct GlobMatcher {
    patterns: Vec<String>,
}

impl GlobMatcher {
    fn matches(&self, relative_path: &Path) -> bool {
        if self.patterns.is_empty() {
            return true;
        }
        let relative_path = relative_path.to_string_lossy().replace('\\', "/");
        let file_name = relative_path.rsplit('/').next().unwrap_or(&relative_path);
        self.patterns.iter().any(|pattern| {
            if pattern.contains('/') {
                glob_match_path(pattern, &relative_path)
            } else {
                glob_match_simple(pattern, file_name)
            }
        })
    }
}

/// Path globs additionally support `**/` matching zero directory levels, so
/// `src/**/mod.rs` should also match `src/mod.rs`. The existing semantics of a
/// plain `*` are unchanged and still handled by `glob_match_simple`.
fn glob_match_path(pattern: &str, path: &str) -> bool {
    if glob_match_simple(pattern, path) {
        return true;
    }

    let mut offset = 0;
    while let Some(found) = pattern[offset..].find("**/") {
        let index = offset + found;
        let without_globstar_dir = format!("{}{}", &pattern[..index], &pattern[index + 3..]);
        if glob_match_path(&without_globstar_dir, path) {
            return true;
        }
        offset = index + 3;
    }
    false
}

fn glob_match_simple(pattern: &str, name: &str) -> bool {
    let pat = pattern.trim_start_matches("**/");
    // Classic two-pointer glob matching with backtracking, correctly supporting
    // `*` (matches any length, including empty) and `?` (matches exactly one
    // character). Patterns without wildcards degrade to exact matching, avoiding
    // the old implementation's bug where `name.ends_with(pat)` made `Cargo.toml`
    // match `my_cargo.toml`.
    let p: Vec<char> = pat.chars().collect();
    let n: Vec<char> = name.chars().collect();
    let mut pi = 0usize;
    let mut ni = 0usize;
    let mut star_pi: Option<usize> = None;
    let mut star_ni = 0usize;
    while ni < n.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == n[ni]) {
            pi += 1;
            ni += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star_pi = Some(pi);
            star_ni = ni;
            pi += 1;
        } else if let Some(sp) = star_pi {
            pi = sp + 1;
            star_ni += 1;
            ni = star_ni;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

/// Recursively collect candidate files via BFS, skipping hidden directories and
/// dependency/build directories.
/// - `glob_matcher`: file-name or relative-path glob filter (None = no filter).
/// - `extensions`: extension whitelist (None = no extension filtering).
fn collect_content_files(
    root: &Path,
    glob_matcher: Option<&GlobMatcher>,
    extensions: Option<&[&str]>,
) -> Result<Vec<PathBuf>, String> {
    if root.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }

    let mut files = Vec::new();
    let mut queue: VecDeque<PathBuf> = VecDeque::new();
    queue.push_back(root.to_path_buf());

    while let Some(dir) = queue.pop_front() {
        if files.len() >= MAX_WALK_FILES {
            break;
        }

        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => continue,
            };

            if file_type.is_symlink() {
                // Do not recurse into directory symlinks; otherwise scanning may
                // repeat, or even run away on a directory loop.
                let meta = match fs::metadata(&path) {
                    Ok(meta) => meta,
                    Err(_) => continue,
                };
                if !meta.is_file() {
                    continue;
                }
            }

            if file_type.is_dir() {
                if rust_tools::commonw::is_skip_dir(name_str.as_ref()) || name_str.starts_with('.')
                {
                    continue;
                }
                queue.push_back(path);
            } else if file_type.is_file() || file_type.is_symlink() {
                if name_str.starts_with('.') {
                    continue;
                }
                if let Some(exts) = extensions {
                    let ext_ok = path
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|ext| exts.contains(&ext))
                        .unwrap_or(false);
                    if !ext_ok {
                        continue;
                    }
                }
                if let Some(matcher) = glob_matcher {
                    let relative_path = path.strip_prefix(root).unwrap_or(&path);
                    if !matcher.matches(relative_path) {
                        continue;
                    }
                }
                files.push(path);
            }
        }
    }

    files.sort();
    Ok(files)
}

fn truncate_output(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out = String::with_capacity(max_chars + 32);
    for (i, ch) in s.chars().enumerate() {
        if i >= max_chars {
            break;
        }
        out.push(ch);
    }
    out.push_str("\n... (output truncated)");
    out
}

// execute_text_grep is kept as a test-only entry point: the text_grep tool is
// retired, but regression tests for the shared engine run_content_search still
// drive it through this arg-parsing path.
#[cfg(test)]
fn execute_text_grep(args: &Value) -> Result<String, String> {
    let pattern = args["pattern"]
        .as_str()
        .ok_or("Missing 'pattern' parameter")?;
    if pattern.is_empty() {
        return Err("pattern must not be empty".to_string());
    }

    let path = args["path"].as_str().unwrap_or(".");
    let file_pattern = args["file_pattern"].as_str();
    let is_regex = args["is_regex"].as_bool().unwrap_or(false);
    let case_sensitive = args["case_sensitive"].as_bool().unwrap_or(true);
    let context_lines = args["context_lines"].as_u64().unwrap_or(2).min(5) as usize;
    let max_results = args["max_results"]
        .as_u64()
        .unwrap_or(50)
        .min(MAX_MATCHES as u64) as usize;

    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .map_err(|e| format!("Failed to get cwd: {}", e))?;
    let root = {
        let p = PathBuf::from(path);
        if p.is_absolute() { p } else { cwd.join(p) }
    };

    if !root.exists() {
        return Err(format!("Path not found: {}", root.display()));
    }

    validate_search_root(&root, &cwd)?;

    let options = ContentSearchOptions {
        query: pattern,
        is_regex,
        case_sensitive,
        context_lines,
        max_results,
        file_pattern,
        // text_grep does not filter by extension — only the optional file_pattern
        // constrains it.
        extensions: None,
        display_root: Some(&cwd),
        max_file_size: MAX_FILE_SIZE,
    };

    run_content_search(&root, &options)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::symlink;

    fn make_temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "ai_text_grep_test_{}_{}",
            name,
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&path).expect("failed to create temp dir");
        path
    }

    #[test]
    fn test_text_grep_literal_match() {
        let dir = make_temp_dir("literal");
        fs::write(
            dir.join("hello.rs"),
            "fn main() {\n    println!(\"hello world\");\n}\n",
        )
        .unwrap();

        let args = serde_json::json!({
            "pattern": "hello world",
            "path": dir.to_string_lossy().to_string()
        });
        let result = execute_text_grep(&args);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("hello.rs"), "should show file name");
        assert!(
            output.contains("hello world"),
            "should show matched content"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_text_grep_literal_metacharacters_are_not_regex() {
        let dir = make_temp_dir("literal_metacharacters");
        fs::write(
            dir.join("literal.txt"),
            "not a regex: a+b[0]\nregex-like alternative: aaab0\n",
        )
        .unwrap();

        let args = serde_json::json!({
            "pattern": "a+b[0]",
            "path": dir.to_string_lossy().to_string()
        });
        let output = execute_text_grep(&args).unwrap();
        assert!(output.contains("not a regex: a+b[0]"), "{}", output);
        assert!(output.contains("1 match(es)"), "{}", output);
        assert!(
            !output
                .lines()
                .any(|line| line.starts_with('>') && line.contains("regex-like alternative")),
            "{}",
            output
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_text_grep_regex_match() {
        let dir = make_temp_dir("regex");
        fs::write(
            dir.join("test.py"),
            "def foo():\n    return 42\n\ndef bar():\n    return 99\n",
        )
        .unwrap();

        let args = serde_json::json!({
            "pattern": "def \\w+\\(",
            "path": dir.to_string_lossy().to_string(),
            "is_regex": true
        });
        let result = execute_text_grep(&args);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("def foo()"), "should find foo");
        assert!(output.contains("def bar()"), "should find bar");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_text_grep_file_pattern_filter() {
        let dir = make_temp_dir("filter");
        fs::write(dir.join("code.rs"), "fn hello() {}\n").unwrap();
        fs::write(dir.join("readme.md"), "hello docs\n").unwrap();

        let args = serde_json::json!({
            "pattern": "hello",
            "path": dir.to_string_lossy().to_string(),
            "file_pattern": "*.rs"
        });
        let result = execute_text_grep(&args);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("code.rs"), "should find in .rs file");
        assert!(!output.contains("readme.md"), "should skip .md file");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_text_grep_brace_glob_expands() {
        let dir = make_temp_dir("brace");
        fs::write(dir.join("a.ts"), "const found = 1;\n").unwrap();
        fs::write(dir.join("b.tsx"), "const found = 2;\n").unwrap();
        fs::write(dir.join("c.js"), "const found = 3;\n").unwrap();

        let args = serde_json::json!({
            "pattern": "found",
            "path": dir.to_string_lossy().to_string(),
            "file_pattern": "*.{ts,tsx}"
        });
        let output = execute_text_grep(&args).unwrap();
        assert!(output.contains("a.ts"), "{}", output);
        assert!(output.contains("b.tsx"), "{}", output);
        assert!(!output.contains("c.js"), "{}", output);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_text_grep_recurses_subdirectories() {
        let dir = make_temp_dir("recurse");
        fs::create_dir_all(dir.join("nested/deep")).unwrap();
        fs::write(dir.join("nested/deep/inner.rs"), "fn needle() {}\n").unwrap();

        let args = serde_json::json!({
            "pattern": "needle",
            "path": dir.to_string_lossy().to_string(),
            "file_pattern": "*.rs"
        });
        let output = execute_text_grep(&args).unwrap();
        assert!(output.contains("inner.rs"), "should recurse: {}", output);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_text_grep_file_pattern_matches_relative_path() {
        let dir = make_temp_dir("relative_pattern");
        let matched = dir.join("src/bin/ai/history/compress/mod.rs");
        let matched_without_intermediate_dir = dir.join("src/bin/ai/history/lib.rs");
        let skipped = dir.join("src/bin/ai/driver/mod.rs");
        fs::create_dir_all(matched.parent().unwrap()).unwrap();
        fs::create_dir_all(skipped.parent().unwrap()).unwrap();
        fs::write(&matched, "fn compress_marker() {}\n").unwrap();
        fs::write(
            &matched_without_intermediate_dir,
            "fn compress_marker() {}\n",
        )
        .unwrap();
        fs::write(&skipped, "fn compress_marker() {}\n").unwrap();

        let args = serde_json::json!({
            "pattern": "compress_marker",
            "path": dir.to_string_lossy(),
            "file_pattern": "src/bin/ai/history/**/*.rs"
        });
        let output = execute_text_grep(&args).unwrap();

        assert!(output.contains("history/compress/mod.rs"), "{output}");
        assert!(output.contains("history/lib.rs"), "{output}");
        assert!(!output.contains("driver/mod.rs"), "{output}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_text_grep_case_insensitive() {
        let dir = make_temp_dir("case");
        fs::write(
            dir.join("test.txt"),
            "Hello World\nhello world\nHELLO WORLD\n",
        )
        .unwrap();

        let args = serde_json::json!({
            "pattern": "hello world",
            "path": dir.to_string_lossy().to_string(),
            "case_sensitive": false
        });
        let result = execute_text_grep(&args);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("3 match(es)"), "should find all 3 variants");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_text_grep_no_matches() {
        let dir = make_temp_dir("nomatch");
        fs::write(dir.join("test.txt"), "nothing special here\n").unwrap();

        let args = serde_json::json!({
            "pattern": "nonexistent_xyz_42",
            "path": dir.to_string_lossy().to_string()
        });
        let result = execute_text_grep(&args);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "No matches found.");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_content_search_ranks_whole_word_first() {
        let dir = make_temp_dir("rank_word");
        // substring hit
        fs::write(dir.join("a_sub.rs"), "let foobar = 1;\n").unwrap();
        // whole-word hit — should rank first
        fs::write(dir.join("z_word.rs"), "let foo = 2;\n").unwrap();

        let args = serde_json::json!({
            "pattern": "foo",
            "path": dir.to_string_lossy().to_string(),
            "file_pattern": "*.rs"
        });
        let output = execute_text_grep(&args).unwrap();
        let word_pos = output.find("z_word.rs").expect("word file present");
        let sub_pos = output.find("a_sub.rs").expect("sub file present");
        assert!(
            word_pos < sub_pos,
            "whole-word file should rank before substring file:\n{}",
            output
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_content_search_ranks_filename_hit_first() {
        let dir = make_temp_dir("rank_name");
        fs::write(dir.join("unrelated.rs"), "// router used here\n").unwrap();
        fs::write(
            dir.join("router.rs"),
            "// some other content\nlet router = 1;\n",
        )
        .unwrap();

        let args = serde_json::json!({
            "pattern": "router",
            "path": dir.to_string_lossy().to_string(),
            "file_pattern": "*.rs"
        });
        let output = execute_text_grep(&args).unwrap();
        let name_pos = output.find("router.rs").expect("router.rs present");
        let other_pos = output.find("unrelated.rs").expect("unrelated.rs present");
        assert!(
            name_pos < other_pos,
            "filename hit should rank first:\n{}",
            output
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_expand_braces_basic() {
        let mut out = expand_braces("*.{ts,tsx}");
        out.sort();
        assert_eq!(out, vec!["*.ts".to_string(), "*.tsx".to_string()]);
        assert_eq!(expand_braces("*.rs"), vec!["*.rs".to_string()]);
    }

    #[test]
    fn test_validate_search_root_rejects_filesystem_root() {
        let cwd = std::env::temp_dir();
        let err = validate_search_root(Path::new("/"), &cwd).expect_err("must reject /");
        assert!(err.contains("Refusing to search"), "{}", err);
    }

    #[test]
    fn test_validate_search_root_rejects_system_prefix() {
        let cwd = std::env::temp_dir();
        let err = validate_search_root(Path::new("/System/Library"), &cwd)
            .expect_err("must reject /System/...");
        assert!(err.contains("system path"), "{}", err);
    }

    #[test]
    fn test_validate_search_root_allows_user_dir() {
        let dir = make_temp_dir("allow");
        let cwd = std::env::temp_dir();
        validate_search_root(&dir, &cwd).expect("user temp dir should be allowed");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_text_grep_rejects_filesystem_root() {
        let args = serde_json::json!({
            "pattern": "anything",
            "path": "/"
        });
        let result = execute_text_grep(&args);
        assert!(result.is_err(), "expected error for path=/");
        let msg = result.unwrap_err();
        assert!(msg.contains("Refusing to search"), "{}", msg);
    }

    // Bug 1 regression: when one file has more hits than MAX_SNIPPETS_PER_FILE(3),
    // truncated hits falling inside the context window must still be marked `>`
    // instead of being shown as ordinary context lines.
    #[test]
    fn test_text_grep_truncated_match_marked_in_context() {
        let dir = make_temp_dir("trunc");
        // 5 consecutive hits; with the default context_lines of 2, the top-3
        // window covers all 5 lines.
        fs::write(
            dir.join("f.txt"),
            "match a\nmatch b\nmatch c\nmatch d\nmatch e\n",
        )
        .unwrap();
        let args = serde_json::json!({
            "pattern": "match",
            "path": dir.to_string_lossy().to_string(),
        });
        let out = execute_text_grep(&args).unwrap();
        // header count should be 5
        assert!(out.contains("5 match(es)"), "{}", out);
        // all 5 matched lines should be marked `>`
        let marked = out.lines().filter(|l| l.starts_with('>')).count();
        assert_eq!(marked, 5, "all 5 matches should be marked `>`:\n{}", out);
        let _ = fs::remove_dir_all(&dir);
    }

    // Bug 2 regression: a literal file_pattern without wildcards must match
    // exactly, never via ends_with.
    #[test]
    fn test_glob_literal_pattern_exact_only() {
        // `Cargo.toml` must not match `my_cargo.toml`
        assert!(glob_match_simple("Cargo.toml", "Cargo.toml"));
        assert!(!glob_match_simple("Cargo.toml", "my_cargo.toml"));
        // `main.rs` must not match `domain.rs`
        assert!(glob_match_simple("main.rs", "main.rs"));
        assert!(!glob_match_simple("main.rs", "domain.rs"));
        // `*.rs` should still match by suffix
        assert!(glob_match_simple("*.rs", "foo.rs"));
        assert!(!glob_match_simple("*.rs", "foo.txt"));
    }

    // Bug 3 regression: the `?` wildcard matches exactly one character.
    #[test]
    fn test_glob_question_mark_single_char() {
        assert!(glob_match_simple("foo?bar", "fooXbar"));
        assert!(glob_match_simple("foo?bar", "foo_bar"));
        assert!(!glob_match_simple("foo?bar", "foobar")); // ? matches at least one character
        assert!(!glob_match_simple("foo?bar", "fooXYbar")); // ? matches exactly one character
        // Combined with `*`
        assert!(glob_match_simple("?at.rs", "cat.rs"));
        assert!(!glob_match_simple("?at.rs", "at.rs"));
        assert!(glob_match_simple("?.rs", "a.rs"));
    }

    #[test]
    fn test_truncate_output_counts_chars_consistently() {
        let content = "你".repeat(20_000);
        let out = truncate_output(&content, 32_000);
        assert_eq!(out, content, "should not truncate under char budget");
        assert!(!out.contains("output truncated"), "{}", out);
    }

    #[test]
    fn test_line_offsets_match_str_lines_semantics() {
        for content in ["", "one", "one\n", "one\n\n", "one\r\ntwo\r\n"] {
            let starts = collect_line_starts(content);
            let actual: Vec<&str> = (0..starts.len())
                .map(|index| line_at(content, &starts, index).unwrap())
                .collect();
            let expected: Vec<&str> = content.lines().collect();
            assert_eq!(actual, expected, "content={content:?}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn test_collect_content_files_skips_symlink_dirs() {
        let dir = make_temp_dir("symlink_dir");
        let real_dir = dir.join("real");
        fs::create_dir_all(&real_dir).unwrap();
        fs::write(real_dir.join("needle.rs"), "fn needle() {}\n").unwrap();
        symlink(&real_dir, dir.join("alias")).unwrap();

        let files = collect_content_files(&dir, None, None).unwrap();
        assert_eq!(files.len(), 1, "symlink dir should not duplicate scan");
        assert_eq!(files[0], real_dir.join("needle.rs"));

        let _ = fs::remove_dir_all(&dir);
    }
}
