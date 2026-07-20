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
/// 每个文件最多保留多少条 snippet（按相关性排序后取前 N）。
const MAX_SNIPPETS_PER_FILE: usize = 3;

/// 系统级目录黑名单：搜索根落在这些前缀内一律拒绝，避免误把整个磁盘扫了。
/// 仅列出明确的"系统/平台"目录；故意不收 `/var`、`/private`、`/tmp`，因为
/// macOS 的临时目录就在 `/var/folders/...`（canonicalize 后为
/// `/private/var/folders/...`），收进去会误伤合法的临时工作目录。
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

/// 校验搜索根目录是否合法，拒绝文件系统根 `/` 与系统级目录。
///
/// 设计目标：阻止 LLM 误传 `path="/"` 或 `path="/System"` 这类会导致全盘扫描、
/// CPU 100% 的调用。`root` 必须已经是绝对路径（调用方先 join 过 cwd）。
pub(crate) fn validate_search_root(root: &Path, cwd: &Path) -> Result<(), String> {
    // 1. 拒绝文件系统根（`/` 或 Windows 盘根）。
    let component_count = root.components().count();
    if component_count <= 1 {
        return Err(format!(
            "Refusing to search filesystem root '{}'. Pass a path inside the current project (cwd: {}).",
            root.display(),
            cwd.display()
        ));
    }

    // 2. 拒绝系统级前缀。比较时优先用 canonicalize，失败则按字面比较。
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
// 共享内容搜索引擎
//
// `run_content_search` 是 code_search.text_search operation 调用的内容搜索核心：
// BFS 递归收集文件 → 逐行匹配 → 相关性重排 → 按文件聚合（每文件 top-N
// snippet + context + `>` 标记匹配行）。普通大小写敏感字面查询直接走
// `str::find`，仅正则和大小写不敏感查询构造 Regex。
//
// 设计要点：
// - 文件收集走 BFS（保留 `*.rs` 递归语义；`terminalw::glob_paths` 非递归，
//   不能直接替代）。
// - `extensions=None` 表示不按扩展名过滤（text_search 的默认行为）；
//   `Some(&[...])` 表示只搜白名单扩展名（保留给未来按语言过滤用）。
// - 相关性评分：whole-word 命中 > 子串命中；大小写一致优先；命中出现在
//   文件名/路径中的文件整体加权；行内匹配靠前的优先。
// ============================================================================

/// 内容搜索的可配置项。由两个调用方各自构造。
pub(crate) struct ContentSearchOptions<'a> {
    /// 原始查询串（用于相关性打分时识别字面大小写、whole-word）。
    pub(crate) query: &'a str,
    /// 是否把 query 当正则。false 时按字面子串（已转义）匹配。
    pub(crate) is_regex: bool,
    /// 是否大小写敏感。
    pub(crate) case_sensitive: bool,
    /// 每个匹配行上下各保留多少 context 行。
    pub(crate) context_lines: usize,
    /// 最多返回多少条匹配行（跨所有文件）。
    pub(crate) max_results: usize,
    /// 可选的文件名 glob 过滤（支持逗号分隔、`*.{ts,tsx}` brace 展开）。
    pub(crate) file_pattern: Option<&'a str>,
    /// 可选的扩展名白名单。None=不按扩展名过滤；Some=只收这些扩展名。
    pub(crate) extensions: Option<&'a [&'a str]>,
    /// 展示路径时用于裁掉前缀（一般是 cwd），让输出用相对路径。
    pub(crate) display_root: Option<&'a Path>,
}

/// 一行匹配，连同它在文件内的相关性分数。
struct ScoredLine {
    line_index: usize,
    score: i64,
}

/// 单个文件的聚合结果（已按相关性排序、限流的 snippet 行）。
struct FileHits {
    /// 用于展示的（可能相对化的）路径字符串。
    display_path: String,
    /// 文件的整体相关性分数（用于文件间排序）。
    file_score: i64,
    /// 命中的行（line_index, score），已按相关性降序。
    scored: Vec<ScoredLine>,
    /// 文件内**全部**命中行号（升序），用于在渲染时正确标记 `>`。
    /// 注意 `scored` 仅保留 top-N snippet，但落到 context 窗口内的
    /// 其余命中行也应以 `>` 标注，否则会被误显示为普通 context 行。
    all_match_indices: Vec<usize>,
    /// 原始文件内容。只为有命中的文件保留，渲染 context 时按行起点借用切片，
    /// 避免为扫描过的每一行单独分配 String。
    content: String,
    /// 与 `str::lines()` 语义一致的行起始字节偏移。
    line_starts: Vec<usize>,
}

enum SearchMatcher {
    /// 默认路径：大小写敏感的字面子串搜索，不构造也不执行正则。
    Literal,
    /// 正则查询以及大小写不敏感的字面查询仍由 regex crate 处理，以保持语义。
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

/// 运行共享内容搜索，返回已格式化好的结果字符串（含 truncate）。
/// 无命中时返回 `Ok("No matches found.")`，让调用方按各自语义包装。
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
        if metadata.len() > MAX_FILE_SIZE {
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

        // 在 truncate 前收集全部命中行号（升序），供渲染时标注 `>`。
        let all_match_indices: Vec<usize> = scored.iter().map(|s| s.line_index).collect();

        // 文件分数取其命中行的最高分 + 路径命中加权，作为文件间排序键。
        let file_score = scored.iter().map(|s| s.score).max().unwrap_or(0) + name_path_bonus;
        // 文件内按相关性降序，再按行号升序（稳定、就近）。
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

    // 文件间按相关性降序，分数相同按路径字典序稳定排列。
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

/// 展示用路径：能相对化就相对化，否则用完整路径。
fn display_path_for(file_path: &Path, display_root: Option<&Path>) -> String {
    if let Some(root) = display_root {
        if let Ok(rel) = file_path.strip_prefix(root) {
            return rel.to_string_lossy().to_string();
        }
    }
    file_path.to_string_lossy().to_string()
}

/// 当查询命中文件名/路径时给文件整体加权（文件名命中 +3，目录路径命中 +1）。
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

/// 对单个匹配行打分：whole-word 命中 +4；字面大小写完全一致 +2；
/// 匹配越靠近行首越优先（最多 +2）。
fn score_line(
    line: &str,
    options: &ContentSearchOptions<'_>,
    match_start: usize,
    match_end: usize,
) -> i64 {
    let mut score = 1; // 基础命中分
    let matched = &line[match_start..match_end];

    // 全词命中加权。
    let left_ok =
        match_start == 0 || !is_identifier_byte(line.as_bytes()[match_start.saturating_sub(1)]);
    let right_ok = match_end >= line.len() || !is_identifier_byte(line.as_bytes()[match_end]);
    if left_ok && right_ok {
        score += 4;
    }

    // 字面（非正则）时，匹配片段与 query 大小写完全一致再加权。
    if !options.is_regex && matched == options.query {
        score += 2;
    }

    // 就近：匹配越靠前越好。
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

/// 收集与 `str::lines()` 一致的行起点。尾部换行不会产生额外空行。
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

/// 根据行起点返回不含换行符的行切片，并与 `str::lines()` 一样去掉 CRLF 中的 CR。
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

        // snippet 行按行号升序渲染（相关性已用于截断与文件排序）。
        let mut match_indices: Vec<usize> = hit.scored.iter().map(|s| s.line_index).collect();
        match_indices.sort_unstable();

        let ranges = merge_context_ranges(&match_indices, context_lines, hit.line_starts.len());
        for range in &ranges {
            if range.start > 0 {
                out.push_str("...\n");
            }
            for idx in range.start..range.end {
                let line_num = idx + 1;
                // 用全部命中行号标注 `>`，避免被截断的命中行落入 context 窗口时
                // 被误显示为普通 context 行。
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
// 文件收集 + glob 匹配
// ----------------------------------------------------------------------------

fn build_glob_matcher(pattern: &str) -> GlobMatcher {
    let mut patterns = Vec::new();
    // 先按"顶层逗号"（不在 `{}` 内的逗号）拆成多个 glob，再各自做 brace 展开，
    // 否则 `*.{ts,tsx}` 会被 brace 内的逗号错误拆开。
    for part in split_top_level_commas(pattern) {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        // brace 展开：`*.{ts,tsx}` → `*.ts`, `*.tsx`
        for expanded in expand_braces(trimmed) {
            patterns.push(expanded);
        }
    }
    GlobMatcher { patterns }
}

/// 按不在 `{}` 内的逗号拆分，使 `a,*.{ts,tsx}` → [`a`, `*.{ts,tsx}`]。
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

/// 展开单层 `{a,b,c}` brace（仅支持一处 brace，足够覆盖 `*.{ts,tsx}` 这类）。
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
    fn matches(&self, file_name: &str) -> bool {
        if self.patterns.is_empty() {
            return true;
        }
        self.patterns
            .iter()
            .any(|pat| glob_match_simple(pat, file_name))
    }
}

fn glob_match_simple(pattern: &str, name: &str) -> bool {
    let pat = pattern.trim_start_matches("**/");
    // 经典两指针 + 回溯的 glob 匹配，正确支持 `*`（匹配任意长度，含空）与
    // `?`（匹配恰好一个字符）。不含通配符时退化为精确匹配，避免旧实现里
    // `name.ends_with(pat)` 把 `Cargo.toml` 误匹配到 `my_cargo.toml`。
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

/// BFS 递归收集候选文件，跳过隐藏目录与依赖/构建目录。
/// - `glob_matcher`：文件名 glob 过滤（None=不过滤）。
/// - `extensions`：扩展名白名单（None=不按扩展名过滤）。
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
                // 不递归目录符号链接；否则可能重复扫描，甚至在目录环上失控。
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
                    if !matcher.matches(&name_str) {
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

// execute_text_grep 保留为测试专用入口：text_grep 工具已退役，但共享引擎
// run_content_search 的回归测试仍通过这条 arg-parsing 路径驱动。
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
        // text_grep 不按扩展名过滤——只受可选的 file_pattern 约束。
        extensions: None,
        display_root: Some(&cwd),
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
        // substring 命中
        fs::write(dir.join("a_sub.rs"), "let foobar = 1;\n").unwrap();
        // whole-word 命中——应排在前面
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

    // Bug 1 回归：单文件命中超过 MAX_SNIPPETS_PER_FILE(3) 时，落入 context 窗口
    // 的被截断命中行也必须以 `>` 标注，而非被误显示为普通 context 行。
    #[test]
    fn test_text_grep_truncated_match_marked_in_context() {
        let dir = make_temp_dir("trunc");
        // 5 条连续命中，context_lines 默认 2，top-3 的窗口会覆盖全部 5 行。
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
        // header 计数应为 5
        assert!(out.contains("5 match(es)"), "{}", out);
        // 全部 5 行命中都应以 `>` 标注
        let marked = out.lines().filter(|l| l.starts_with('>')).count();
        assert_eq!(marked, 5, "all 5 matches should be marked `>`:\n{}", out);
        let _ = fs::remove_dir_all(&dir);
    }

    // Bug 2 回归：无通配符的字面 file_pattern 必须精确匹配，不得走 ends_with。
    #[test]
    fn test_glob_literal_pattern_exact_only() {
        // `Cargo.toml` 不应匹配 `my_cargo.toml`
        assert!(glob_match_simple("Cargo.toml", "Cargo.toml"));
        assert!(!glob_match_simple("Cargo.toml", "my_cargo.toml"));
        // `main.rs` 不应匹配 `domain.rs`
        assert!(glob_match_simple("main.rs", "main.rs"));
        assert!(!glob_match_simple("main.rs", "domain.rs"));
        // `*.rs` 仍应按后缀匹配
        assert!(glob_match_simple("*.rs", "foo.rs"));
        assert!(!glob_match_simple("*.rs", "foo.txt"));
    }

    // Bug 3 回归：`?` 通配符匹配恰好一个字符。
    #[test]
    fn test_glob_question_mark_single_char() {
        assert!(glob_match_simple("foo?bar", "fooXbar"));
        assert!(glob_match_simple("foo?bar", "foo_bar"));
        assert!(!glob_match_simple("foo?bar", "foobar")); // ? 至少匹配一个字符
        assert!(!glob_match_simple("foo?bar", "fooXYbar")); // ? 恰好一个字符
        // 与 `*` 组合
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
