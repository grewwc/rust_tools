//! 面向 **memory / knowledge / RAG 检索** 的相似度库（向量、Jaccard、Dice、Han
//! 分词）。本模块抽自 `memory_store.rs` / `rag_store.rs` 以消除两份重复实现。
//!
//! 与 `driver::text_similarity` 是两个并列、独立的相似度库，使用场景与数据结构
//! 完全不同，**调用方不可混用**。详见 `driver::text_similarity` 模块顶部说明。
//! 简要差异：本模块归一化时 **完全删除空格**（适合中文 token 粒度的相似度），
//! 而 `driver::text_similarity` 归一化时 **保留单个空格**（适合英文/路由风格的
//! token 比较）。
//!
//! 注意：如果将来要将两套实现合并，需要同时验证 RAG / memory 召回 与 skill /
//! agent routing 两侧的回归数据。

use rust_tools::cw::SkipMap;
use rust_tools::cw::SkipSet;

/// Cosine similarity between two f32 vectors.
/// Returns 0.0 for empty or mismatched vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        let x = a[i];
        let y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-8 {
        0.0
    } else {
        (dot / denom).max(-1.0).min(1.0)
    }
}

/// Normalize text by removing whitespace and lowercasing.
pub fn norm_text(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Generate character bigrams.
pub fn bigrams(s: &str) -> Vec<(char, char)> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 2 {
        return Vec::new();
    }
    let mut v = Vec::with_capacity(chars.len() - 1);
    for i in 0..(chars.len() - 1) {
        v.push((chars[i], chars[i + 1]));
    }
    v
}

/// Dice coefficient between two bigram lists.
pub fn dice_coefficient(a: &[(char, char)], b: &[(char, char)]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut count = 0usize;
    let mut map: SkipMap<(char, char), usize> = SkipMap::default();
    for x in a {
        *map.entry(*x).or_insert(0) += 1;
    }
    for y in b {
        if let Some(c) = map.get_mut(y) {
            if *c > 0 {
                count += 1;
                *c -= 1;
            }
        }
    }
    (2.0 * count as f64) / ((a.len() + b.len()) as f64)
}

/// Check if a character is a Han (Chinese) character.
pub fn is_han_char(ch: char) -> bool {
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

/// Tokenize text into words and individual Chinese characters.
pub fn tokenize(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut buf = String::new();
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            buf.push(ch.to_ascii_lowercase());
        } else {
            if !buf.is_empty() {
                tokens.push(buf.clone());
                buf.clear();
            }
            if is_han_char(ch) {
                tokens.push(ch.to_string());
            }
        }
    }
    if !buf.is_empty() {
        tokens.push(buf);
    }
    tokens
}

/// Expand tokens by deduplicating (preserving order).
pub fn expand_tokens(tokens: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut seen: SkipSet<String> = SkipSet::default();
    for t in tokens {
        let tnorm = t.to_lowercase();
        if seen.insert(tnorm.clone()) {
            out.push(tnorm);
        }
    }
    out
}

/// Jaccard similarity between two token lists.
pub fn jaccard(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    use std::collections::HashSet;
    let sa: HashSet<&String> = a.iter().collect();
    let sb: HashSet<&String> = b.iter().collect();
    let inter = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    if union == 0.0 { 0.0 } else { inter / union }
}

/// Character overlap ratio.
pub fn char_overlap(a: &str, b: &str) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    use std::collections::HashSet;
    let sa: HashSet<char> = a.chars().collect();
    let sb: HashSet<char> = b.chars().collect();
    let inter = sa.intersection(&sb).count() as f64;
    let denom = sa.len().min(sb.len()) as f64;
    if denom == 0.0 { 0.0 } else { inter / denom }
}

/// Jaccard similarity for note deduplication.
pub fn note_similarity(a: &str, b: &str) -> f64 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    if a_lower == b_lower {
        return 1.0;
    }
    let a_words: std::collections::HashSet<&str> = a_lower.split_whitespace().collect();
    let b_words: std::collections::HashSet<&str> = b_lower.split_whitespace().collect();
    if a_words.is_empty() || b_words.is_empty() {
        return 0.0;
    }
    let intersection = a_words.intersection(&b_words).count();
    let union = a_words.union(&b_words).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}
