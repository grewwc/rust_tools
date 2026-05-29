//! 面向 **skill / agent routing** 的轻量文本相似度工具。
//!
//! 与 `knowledge::indexing::similarity` 是两个并列、独立的相似度库，使用场景与
//! 数据结构完全不同，**调用方不可混用**：
//!
//! | 模块                                       | 主要用户                       | 归一化            | 集合数据结构 |
//! |--------------------------------------------|--------------------------------|-------------------|--------------|
//! | `driver::text_similarity` (本文件)         | skill_match / agent_router     | 保留单空格        | `SkipSet`    |
//! | `knowledge::indexing::similarity`          | memory / RAG 检索              | 完全去除空格      | `Vec`/`HashSet` |
//!
//! 两套实现历史上来自不同代码路径，目前算法精度也不同（本文件用 `cosine_tfidf` 结合
//! 2-4 元 char-ngram，knowledge 端用基于词的 jaccard / dice）。如果将来要合并，
//! 必须同时验证 routing 与召回两个 pipeline 的回归数据。

use rust_tools::cw::SkipMap;
use rust_tools::cw::SkipSet;

/// 在 `haystack` 中按 ASCII 词边界查找 `needle`。
///
/// 词边界 = 该位置左右两侧不是 ASCII 字母 / 数字 / `_`。
/// 该函数仅适用于 ASCII needle；对包含 CJK / emoji 的 needle，
/// 调用方应自行回退到子串匹配，因为这些字符并无"词"的概念。
pub fn ascii_word_contains(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    let bytes = haystack.as_bytes();
    let needle_bytes = needle.as_bytes();
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut start = 0usize;
    while let Some(rel) = haystack[start..].find(needle) {
        let pos = start + rel;
        let left_ok = pos == 0 || !is_word(bytes[pos - 1]);
        let right = pos + needle_bytes.len();
        let right_ok = right >= bytes.len() || !is_word(bytes[right]);
        if left_ok && right_ok {
            return true;
        }
        start = pos + 1;
        if start >= bytes.len() {
            break;
        }
    }
    false
}

/// `pattern` 是否完全由 ASCII 字母 / 数字 / `_` / `-` 组成。
/// 若是，调用方可使用 `ascii_word_contains`；否则应回退为 `contains`。
pub fn pattern_is_ascii_word(pattern: &str) -> bool {
    !pattern.is_empty()
        && pattern
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
}

pub struct TextSimilarityFeatures {
    pub normalized: String,
    pub token_set: SkipSet<String>,
    pub char_bigrams: SkipSet<String>,
    pub ngram_tf: SkipMap<String, f64>,
}

impl TextSimilarityFeatures {
    pub fn from_text(input: &str) -> Self {
        let normalized = normalize_text_for_similarity(input);
        let token_set = token_set_from_normalized(&normalized);
        let char_bigrams = char_ngram_set(&normalized, 2);
        let ngram_tf = extract_ngram_tf_from_normalized(&normalized);
        Self {
            normalized,
            token_set,
            char_bigrams,
            ngram_tf,
        }
    }
}

pub fn normalize_text_for_similarity(input: &str) -> String {
    let mut normalized = String::new();
    let mut prev_space = false;
    for ch in input.to_lowercase().chars() {
        if ch == '\r' {
            continue;
        }
        if ch.is_whitespace() {
            if !prev_space {
                normalized.push(' ');
            }
            prev_space = true;
        } else {
            normalized.push(ch);
            prev_space = false;
        }
    }
    normalized.trim().to_string()
}

pub fn extract_ngram_tf_from_normalized(normalized: &str) -> SkipMap<String, f64> {
    if normalized.is_empty() {
        return SkipMap::default();
    }
    let chars = format!("^{normalized}$").chars().collect::<Vec<_>>();
    let mut counts = SkipMap::default();
    for n in 2..=4 {
        if chars.len() < n {
            continue;
        }
        for window in chars.windows(n) {
            let token = window.iter().collect::<String>();
            if token.trim().is_empty() {
                continue;
            }
            *counts.entry(token).or_insert(0.0) += 1.0;
        }
    }
    let total = counts.values().sum::<f64>();
    if total > f64::EPSILON {
        for value in counts.values_mut() {
            *value /= total;
        }
    }
    counts
}

pub fn build_idf_from_documents(docs: &[&SkipMap<String, f64>]) -> SkipMap<String, f64> {
    let mut df = SkipMap::default();
    for doc in docs {
        for token in doc.keys() {
            *df.entry(token.clone()).or_insert(0usize) += 1;
        }
    }

    let total_docs = docs.len().max(1) as f64;
    let mut idf = SkipMap::default();
    for (token, freq) in df {
        let value = ((1.0 + total_docs) / (1.0 + freq as f64)).ln() + 1.0;
        idf.insert(token, value);
    }
    idf
}

pub fn cosine_tfidf_similarity(
    query_tf: &SkipMap<String, f64>,
    doc_tf: &SkipMap<String, f64>,
    idf: &SkipMap<String, f64>,
) -> f64 {
    let mut dot = 0.0;
    let mut query_norm = 0.0;
    let mut doc_norm = 0.0;

    for (token, tf) in query_tf {
        let weight = *tf * idf.get(token).unwrap_or(1.0);
        query_norm += weight * weight;
        if let Some(doc_tf_val) = doc_tf.get(token) {
            let doc_weight = doc_tf_val * idf.get(token).unwrap_or(1.0);
            dot += weight * doc_weight;
        }
    }
    for (token, tf) in doc_tf {
        let weight = *tf * idf.get(token).unwrap_or(1.0);
        doc_norm += weight * weight;
    }
    if query_norm <= f64::EPSILON || doc_norm <= f64::EPSILON {
        return 0.0;
    }
    dot / (query_norm.sqrt() * doc_norm.sqrt())
}

pub fn token_set_from_normalized(normalized: &str) -> SkipSet<String> {
    let mut set = SkipSet::new(16);
    for token in normalized.split_whitespace() {
        set.insert(token.to_string());
    }
    set
}

pub fn char_ngram_set(normalized: &str, n: usize) -> SkipSet<String> {
    let chars = normalized.chars().collect::<Vec<_>>();
    let mut set = SkipSet::new(chars.len().max(4));
    if n == 0 {
        return set;
    }
    if chars.len() < n {
        if !normalized.is_empty() {
            set.insert(normalized.to_string());
        }
        return set;
    }
    for window in chars.windows(n) {
        set.insert(window.iter().collect::<String>());
    }
    set
}

pub fn set_intersection_count(a: &SkipSet<String>, b: &SkipSet<String>) -> usize {
    a.iter().filter(|item| b.contains(item)).count()
}

pub fn jaccard_similarity_for_sets(a: &SkipSet<String>, b: &SkipSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = set_intersection_count(a, b);
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}
