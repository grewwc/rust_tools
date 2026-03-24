use std::sync::LazyLock;

use regex::Regex;

use crate::common::types::{FastMap, FastSet};
use crate::memo::model::MemoRecord;

const SEARCH_MIN_INFORMATIVE_COVERAGE: f64 = 0.2;
const SEARCH_MIN_INFORMATIVE_COVERAGE_WITH_URL: f64 = 0.26;

const SEARCH_SYNONYM_GROUPS: &[&[&str]] = &[
    &[
        "链接", "地址", "网址", "url", "uri", "link", "endpoint", "host", "域名", "官网", "site",
        "http", "https",
    ],
    &["数据库", "database", "db"],
];

const SEARCH_GENERIC_SUFFIXES: &[&str] = &[
    "相关", "有关", "内容", "方面", "资料", "信息", "情况", "问题", "事项", "记录",
];

static SEARCH_URL_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)https?://\S+").expect("invalid search url regex"));
static SEARCH_SYNONYM_INDEX: LazyLock<FastMap<String, usize>> = LazyLock::new(|| {
    let mut index = FastMap::default();
    for (group_idx, group) in SEARCH_SYNONYM_GROUPS.iter().enumerate() {
        for term in *group {
            let normalized = compact_search_text(term);
            if !normalized.is_empty() {
                index.insert(normalized, group_idx);
            }
        }
    }
    index
});
static SEARCH_KNOWN_TERMS: LazyLock<Vec<String>> = LazyLock::new(|| {
    let mut seen = FastSet::default();
    let mut terms = Vec::new();
    for group in SEARCH_SYNONYM_GROUPS {
        for term in *group {
            let normalized = compact_search_text(term);
            if !normalized.is_empty() && seen.insert(normalized.clone()) {
                terms.push(normalized);
            }
        }
    }
    terms
});
static SEARCH_IGNORED_TOKENS: LazyLock<FastSet<String>> = LazyLock::new(|| {
    SEARCH_GENERIC_SUFFIXES
        .iter()
        .map(|token| compact_search_text(token))
        .filter(|token| !token.is_empty())
        .collect()
});

#[derive(Clone, Debug, Default)]
struct SearchDocument {
    normalized: String,
    compact: String,
    tokens: Vec<String>,
    ngrams: FastMap<String, usize>,
    has_url: bool,
    requests_url: bool,
}

pub fn highlight_pattern(query: &str) -> Option<Regex> {
    let mut parts = tokenize_search_text(query)
        .into_iter()
        .filter(|token| token.chars().count() >= 2)
        .collect::<Vec<_>>();
    if parts.is_empty() {
        let compact = compact_search_text(query);
        if compact.is_empty() {
            return None;
        }
        parts.push(compact);
    }
    parts.sort_by_key(|token| std::cmp::Reverse(token.chars().count()));
    let escaped = parts
        .iter()
        .map(|token| regex::escape(token))
        .collect::<Vec<_>>();
    Regex::new(&format!("(?i){}", escaped.join("|"))).ok()
}

pub fn score_record(record: &MemoRecord, query: &str) -> f64 {
    let query_doc = new_search_query_document(query);
    if query_doc.compact.is_empty() {
        return 0.0;
    }

    let tags_text = record.tags.join(" ");
    let title_score = score_search_document(&query_doc, &new_search_document(&record.title));
    let tag_score = score_search_document(&query_doc, &new_search_document(&tags_text));
    let combined_score = score_search_document(
        &query_doc,
        &new_search_document(&format!("{} {}", record.title, tags_text)),
    )
    .max(score_search_document(
        &query_doc,
        &new_search_document(&format!("{} {}", tags_text, record.title)),
    ));
    if title_score <= 0.0 && tag_score <= 0.0 && combined_score <= 0.0 {
        return 0.0;
    }

    let mut best = title_score.max(combined_score).max(tag_score * 0.94);
    if title_score > 0.0 && tag_score > 0.0 {
        best += 0.35 * title_score.min(tag_score);
    }
    if tag_score > 0.0 {
        best += 0.12 * tag_score;
    }
    best
}

pub fn build_preview_lines(record: &MemoRecord, query: &str, max_lines: usize) -> Vec<String> {
    let query = compact_search_text(query);
    if query.is_empty() {
        return Vec::new();
    }

    let lines = normalized_record_lines(&record.title)
        .into_iter()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();

    let mut out = Vec::new();
    let matched = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| compact_search_text(line).contains(&query))
        .map(|(idx, _)| idx)
        .collect::<Vec<_>>();

    if !matched.is_empty() {
        let mut selected = vec![false; lines.len()];
        for idx in matched {
            let start = idx.saturating_sub(1);
            let end = (idx + 1).min(lines.len().saturating_sub(1));
            for j in start..=end {
                selected[j] = true;
            }
        }
        for (idx, line) in lines.iter().enumerate() {
            if !selected[idx] {
                continue;
            }
            out.push(line.to_string());
            if out.len() >= max_lines {
                break;
            }
        }
    }
    if !out.is_empty() {
        return out;
    }

    if record
        .tags
        .iter()
        .any(|tag| compact_search_text(tag).contains(&query))
    {
        out.extend(lines.into_iter().take(max_lines));
    }

    out
}

fn new_search_document(text: &str) -> SearchDocument {
    new_search_document_with_options(text, true)
}

fn new_search_query_document(text: &str) -> SearchDocument {
    new_search_document_with_options(text, !looks_like_explicit_url(text))
}

fn new_search_document_with_options(text: &str, trim_urls: bool) -> SearchDocument {
    let normalized = normalize_search_text_with_options(text, trim_urls);
    let compact = compact_normalized_text(&normalized);
    let tokens = tokenize_normalized_search_text(&normalized);
    SearchDocument {
        normalized: normalized.clone(),
        compact: compact.clone(),
        tokens: tokens.clone(),
        ngrams: build_search_ngrams(&compact),
        has_url: contains_url(text),
        requests_url: search_requests_url_from_tokens(&tokens),
    }
}

fn score_search_document(query: &SearchDocument, candidate: &SearchDocument) -> f64 {
    if query.compact.is_empty() || candidate.compact.is_empty() {
        return 0.0;
    }

    let mut coverage = search_coverage_score(&query.tokens, &candidate.tokens);
    if coverage == 0.0 {
        return 0.0;
    }

    let informative_tokens = filter_search_informative_tokens(&query.tokens);
    if !informative_tokens.is_empty() {
        let informative_coverage = search_coverage_score(&informative_tokens, &candidate.tokens);
        let min_informative_coverage = if query.requests_url {
            SEARCH_MIN_INFORMATIVE_COVERAGE_WITH_URL
        } else {
            SEARCH_MIN_INFORMATIVE_COVERAGE
        };
        if informative_coverage < min_informative_coverage {
            return 0.0;
        }
        coverage = coverage.max(0.72 * coverage + 0.38 * informative_coverage);
    }

    let ngram = dice_coefficient(&query.ngrams, &candidate.ngrams);
    let mut whole = search_token_similarity(&query.compact, &candidate.compact);
    if candidate.compact.chars().count() > 96 {
        whole *= 0.4;
    }
    let exact = if candidate.compact.contains(&query.compact)
        || candidate.normalized.contains(&query.normalized)
    {
        1.25
    } else {
        0.0
    };
    let prefix = if candidate.compact.starts_with(&query.compact) {
        0.22
    } else {
        0.0
    };
    let url_bonus = if query.requests_url && candidate.has_url {
        0.18
    } else {
        0.0
    };

    1.35 * coverage + 0.75 * ngram + 0.35 * whole + exact + prefix + url_bonus
}

fn search_coverage_score(query_tokens: &[String], candidate_tokens: &[String]) -> f64 {
    if query_tokens.is_empty() || candidate_tokens.is_empty() {
        return 0.0;
    }

    let mut total_weight = 0.0;
    let mut total_score = 0.0;
    for query_token in query_tokens {
        let weight = clamp_int(query_token.chars().count(), 1, 6) as f64;
        let best = candidate_tokens
            .iter()
            .map(|candidate_token| search_token_similarity(query_token, candidate_token))
            .fold(0.0, f64::max);
        if requires_strong_search_token_match(query_token) && best < 0.85 {
            return 0.0;
        }
        total_weight += weight;
        total_score += best * weight;
    }

    if total_weight == 0.0 {
        0.0
    } else {
        total_score / total_weight
    }
}

fn filter_search_informative_tokens(tokens: &[String]) -> Vec<String> {
    let mut result = Vec::new();
    let mut seen = FastSet::default();
    for token in tokens {
        let token = compact_search_text(token);
        if token.is_empty() || SEARCH_IGNORED_TOKENS.contains(&token) {
            continue;
        }
        if SEARCH_SYNONYM_INDEX.get(&token).copied() == Some(0) {
            continue;
        }
        let stripped = strip_search_url_affixes(&token);
        let chosen = if !stripped.is_empty() && stripped != token {
            stripped
        } else {
            token
        };
        if seen.insert(chosen.clone()) {
            result.push(chosen);
        }
    }
    result
}

fn strip_search_url_affixes(token: &str) -> String {
    let token = compact_search_text(token);
    if token.is_empty() {
        return String::new();
    }

    let mut best = token.clone();
    for term in SEARCH_SYNONYM_GROUPS[0] {
        let affix = compact_search_text(term);
        if affix.is_empty() || affix == token {
            continue;
        }
        if let Some(candidate) = token.strip_prefix(&affix)
            && candidate.chars().count() >= 2
            && candidate.chars().count() < best.chars().count()
        {
            best = candidate.to_string();
        }
        if let Some(candidate) = token.strip_suffix(&affix)
            && candidate.chars().count() >= 2
            && candidate.chars().count() < best.chars().count()
        {
            best = candidate.to_string();
        }
    }
    best
}

fn search_token_similarity(left: &str, right: &str) -> f64 {
    let left = compact_search_text(left);
    let right = compact_search_text(right);
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    if left == right {
        return 1.0;
    }

    if let (Some(left_group), Some(right_group)) = (
        SEARCH_SYNONYM_INDEX.get(&left),
        SEARCH_SYNONYM_INDEX.get(&right),
    ) && left_group == right_group
    {
        return 0.92;
    }

    if right.contains(&left) || left.contains(&right) {
        let shorter = left.chars().count().min(right.chars().count()) as f64;
        let longer = left.chars().count().max(right.chars().count()) as f64;
        if is_ascii_alnum_search_token(&left) && is_ascii_alnum_search_token(&right) {
            let ratio = shorter / longer;
            if ratio < 0.75 {
                return 0.18 + 0.62 * ratio;
            }
            return 0.42 + 0.5 * ratio;
        }
        return 0.82 + 0.18 * shorter / longer;
    }

    let ngram = dice_coefficient(&build_search_ngrams(&left), &build_search_ngrams(&right));
    let left_chars = left.chars().collect::<Vec<_>>();
    let right_chars = right.chars().collect::<Vec<_>>();
    if left_chars.len() > 64 || right_chars.len() > 64 {
        return ngram;
    }
    let distance = levenshtein_distance(&left_chars, &right_chars);
    let max_len = left_chars.len().max(right_chars.len());
    if max_len == 0 {
        return 0.0;
    }
    let mut edit = 1.0 - distance as f64 / max_len as f64;
    if edit < 0.0 {
        edit = 0.0;
    }
    0.68 * edit + 0.32 * ngram
}

fn requires_strong_search_token_match(token: &str) -> bool {
    let token = compact_search_text(token);
    !token.is_empty() && is_ascii_alnum_search_token(&token) && token.chars().count() <= 2
}

fn normalize_search_text(text: &str) -> String {
    normalize_search_text_with_options(text, true)
}

fn normalize_search_text_with_options(text: &str, trim_urls: bool) -> String {
    let source = sanitize_search_source(text, trim_urls)
        .trim()
        .to_lowercase();
    if source.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    let mut last_space = false;
    for ch in source.chars() {
        if ch.is_alphanumeric() || is_han_char(ch) {
            out.push(ch);
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }

    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn sanitize_search_source(text: &str, trim_urls: bool) -> String {
    if !trim_urls {
        return text.to_string();
    }
    SEARCH_URL_PATTERN.replace_all(text, " url ").into_owned()
}

fn compact_search_text(text: &str) -> String {
    compact_normalized_text(&normalize_search_text(text))
}

fn compact_normalized_text(normalized: &str) -> String {
    normalized
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect()
}

fn tokenize_search_text(text: &str) -> Vec<String> {
    tokenize_normalized_search_text(&normalize_search_text(text))
}

fn tokenize_normalized_search_text(normalized: &str) -> Vec<String> {
    if normalized.is_empty() {
        return Vec::new();
    }

    let mut tokens = Vec::new();
    let mut seen = FastSet::default();
    for chunk in normalized.split_whitespace() {
        let parts = split_mixed_search_token(chunk);
        for part in parts {
            append_search_token(&mut tokens, &mut seen, &part);
            append_known_search_terms(&part, &mut tokens, &mut seen);
            append_derived_search_terms(&part, &mut tokens, &mut seen);
        }
        append_known_search_terms(chunk, &mut tokens, &mut seen);
        append_derived_search_terms(chunk, &mut tokens, &mut seen);
    }
    tokens
}

fn append_search_token(tokens: &mut Vec<String>, seen: &mut FastSet<String>, token: &str) {
    let token = compact_search_text(token);
    if token.is_empty() || SEARCH_IGNORED_TOKENS.contains(&token) || !seen.insert(token.clone()) {
        return;
    }
    tokens.push(token);
}

fn append_known_search_terms(token: &str, out: &mut Vec<String>, seen: &mut FastSet<String>) {
    let compact = compact_search_text(token);
    if compact.is_empty() {
        return;
    }
    for known_term in SEARCH_KNOWN_TERMS.iter() {
        if known_term != &compact && compact.contains(known_term) {
            append_search_token(out, seen, known_term);
        }
    }
}

fn append_derived_search_terms(token: &str, out: &mut Vec<String>, seen: &mut FastSet<String>) {
    let compact = compact_search_text(token);
    if compact.is_empty() {
        return;
    }

    let stripped = strip_generic_search_suffixes(&compact);
    if !stripped.is_empty() && stripped != compact {
        append_search_token(out, seen, &stripped);
        append_known_search_terms(&stripped, out, seen);
    }
    if is_han_search_token(&compact) {
        append_han_search_prefixes(&compact, out, seen);
    }
}

fn strip_generic_search_suffixes(token: &str) -> String {
    let mut stripped = token.to_string();
    loop {
        let mut updated = stripped.clone();
        for suffix in SEARCH_GENERIC_SUFFIXES {
            if let Some(candidate) = updated.strip_suffix(suffix)
                && candidate.chars().count() >= 2
            {
                updated = candidate.to_string();
            }
        }
        if updated == stripped {
            break;
        }
        stripped = updated;
    }
    stripped
}

fn append_han_search_prefixes(token: &str, out: &mut Vec<String>, seen: &mut FastSet<String>) {
    let runes = token.chars().collect::<Vec<_>>();
    if runes.len() < 3 {
        return;
    }
    let max_prefix = 4.min(runes.len() - 1);
    for size in 2..=max_prefix {
        let prefix = runes.iter().take(size).collect::<String>();
        append_search_token(out, seen, &prefix);
    }
}

fn is_han_search_token(token: &str) -> bool {
    !token.is_empty() && token.chars().all(is_han_char)
}

fn split_mixed_search_token(token: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut last_class = 0;
    for ch in token.chars() {
        let class = search_rune_class(ch);
        if class == 0 {
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            last_class = 0;
            continue;
        }
        if last_class != 0 && class != last_class && !current.is_empty() {
            parts.push(std::mem::take(&mut current));
        }
        current.push(ch);
        last_class = class;
    }
    if !current.is_empty() {
        parts.push(current);
    }
    if parts.is_empty() && !token.is_empty() {
        parts.push(token.to_string());
    }
    parts
}

fn looks_like_explicit_url(text: &str) -> bool {
    contains_url(text)
}

fn search_requests_url_from_tokens(tokens: &[String]) -> bool {
    tokens
        .iter()
        .any(|token| SEARCH_SYNONYM_INDEX.get(token).copied() == Some(0))
}

fn contains_url(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("http://") || lower.contains("https://")
}

fn build_search_ngrams(compact: &str) -> FastMap<String, usize> {
    if compact.is_empty() {
        return FastMap::default();
    }

    let runes = compact.chars().collect::<Vec<_>>();
    let mut ngrams = FastMap::default();
    if runes.len() == 1 {
        ngrams.insert(compact.to_string(), 1);
        return ngrams;
    }

    for size in [2, 3] {
        if runes.len() < size {
            continue;
        }
        for idx in 0..=runes.len() - size {
            *ngrams
                .entry(runes[idx..idx + size].iter().collect::<String>())
                .or_insert(0) += 1;
        }
    }

    if ngrams.is_empty() {
        ngrams.insert(compact.to_string(), 1);
    }
    ngrams
}

fn dice_coefficient(left: &FastMap<String, usize>, right: &FastMap<String, usize>) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }

    let left_total: usize = left.values().sum();
    let right_total: usize = right.values().sum();
    let mut intersection = 0_usize;
    for (token, left_count) in left {
        if let Some(right_count) = right.get(token) {
            intersection += (*left_count).min(*right_count);
        }
    }
    if left_total + right_total == 0 {
        0.0
    } else {
        2.0 * intersection as f64 / (left_total + right_total) as f64
    }
}

fn levenshtein_distance(left: &[char], right: &[char]) -> usize {
    if left.is_empty() {
        return right.len();
    }
    if right.is_empty() {
        return left.len();
    }

    let mut prev = (0..=right.len()).collect::<Vec<_>>();
    let mut curr = vec![0_usize; right.len() + 1];
    for (left_idx, left_ch) in left.iter().enumerate() {
        curr[0] = left_idx + 1;
        for (right_idx, right_ch) in right.iter().enumerate() {
            let cost = if left_ch == right_ch { 0 } else { 1 };
            curr[right_idx + 1] = (prev[right_idx + 1] + 1)
                .min(curr[right_idx] + 1)
                .min(prev[right_idx] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[right.len()]
}

fn search_rune_class(ch: char) -> i32 {
    if is_han_char(ch) {
        1
    } else if ch.is_ascii_alphanumeric() {
        2
    } else if ch.is_alphanumeric() {
        3
    } else {
        0
    }
}

fn is_ascii_alnum_search_token(token: &str) -> bool {
    !token.is_empty() && token.chars().all(|ch| ch.is_ascii_alphanumeric())
}

fn is_han_char(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0x30000..=0x3134F
    )
}

fn clamp_int(value: usize, lower: usize, upper: usize) -> usize {
    value.clamp(lower, upper)
}

fn normalized_record_lines(title: &str) -> Vec<String> {
    title
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .lines()
        .map(|line| line.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{build_preview_lines, score_record, search_token_similarity};
    use crate::memo::model::MemoRecord;

    fn make_record(title: &str, tags: &[&str]) -> MemoRecord {
        MemoRecord {
            id: "id".to_string(),
            add_date: 0,
            modified_date: 0,
            finished: false,
            hold: false,
            title: title.to_string(),
            tags: tags.iter().map(|tag| tag.to_string()).collect(),
        }
    }

    #[test]
    fn score_record_matches_query_in_tags() {
        let record = make_record("ORDER BY\nshard_skew_ratio DESC", &["判断分片键是否倾斜"]);

        assert!(score_record(&record, "倾斜") > 0.0);
    }

    #[test]
    fn score_record_recognizes_url_synonyms() {
        let record = make_record("官方链接 https://example.com/workbench", &["database"]);

        assert!(score_record(&record, "url") > 0.0);
        assert!(search_token_similarity("url", "链接") >= 0.92);
    }

    #[test]
    fn preview_falls_back_to_title_when_query_only_hits_tags() {
        let record = make_record("ORDER BY\nshard_skew_ratio DESC", &["判断分片键是否倾斜"]);

        assert_eq!(
            build_preview_lines(&record, "倾斜", 3),
            vec!["ORDER BY".to_string(), "shard_skew_ratio DESC".to_string()]
        );
    }

    #[test]
    fn preview_includes_context_around_matched_lines() {
        let record = make_record("line 1\nline 2 keyword\nline 3\nline 4", &[]);

        assert_eq!(
            build_preview_lines(&record, "keyword", 3),
            vec![
                "line 1".to_string(),
                "line 2 keyword".to_string(),
                "line 3".to_string()
            ]
        );
    }
}
