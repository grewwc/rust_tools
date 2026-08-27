//! Lexical similarity library for memory / knowledge search.
//!
//! Scope: exact-term, character-level matching only — there is no semantic
//! (embedding) layer in this codebase anymore. These primitives feed the BM25
//! term space in `memory_store.rs` and the token-coverage bonus in
//! `service/memory.rs`.
//!
//! Tokenization strategy (dependency-free, standard for CJK + Latin text):
//!   - Han (CJK) runs become **character bigrams** (`搜索算法` → `搜索 索算
//!     算法`). Bigrams are the classic substitute for word segmentation when no
//!     dictionary is bundled; they keep adjacent-character (word) information
//!     that single-character tokens throw away.
//!   - Latin/digit runs become whole words (lowercased).
//!   - Common function words (stopwords) are dropped from both sides so they
//!     cannot inflate similarity.
//!
//! `norm_text` preserves word boundaries (single spaces) instead of stripping
//! all whitespace: the previous version removed every space, which merged
//! distinct words ("rust tools" → "rusttools") and fabricated cross-word
//! bigrams.

use rustc_hash::FxHashSet;

/// Normalize text: full-width ASCII → half-width, lowercase, collapse whitespace
/// runs to a single space. Word boundaries are preserved.
pub fn norm_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pending_space = false;
    for ch in s.chars() {
        let ch = match ch {
            '\u{3000}' => ' ', // ideographic space (U+3000)
            '\u{FF01}'..='\u{FF5E}' => {
                // Full-width ASCII block maps to U+0021..=U+007E.
                char::from_u32(ch as u32 - 0xFEE0).unwrap_or(ch)
            }
            _ => ch,
        };
        if ch.is_whitespace() {
            pending_space = true;
            continue;
        }
        if pending_space && !out.is_empty() {
            out.push(' ');
        }
        pending_space = false;
        for lc in ch.to_lowercase() {
            out.push(lc);
        }
    }
    out
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

/// Tokenize text into search terms.
///
/// Han runs → character bigrams (single char when the run is one char);
/// Latin/digit runs → whole words. Tokens are normalized and stopword-filtered;
/// they may repeat, so callers dedup with `expand_tokens` when a set is wanted.
///
/// Note: Han characters are Unicode `Alphabetic`, so `is_han_char` must be
/// checked before `is_alphanumeric` — otherwise Han runs merge into Latin runs
/// and "搜索算法" becomes one opaque token with no shared terms with "搜索".
pub fn tokenize(s: &str) -> Vec<String> {
    let norm = norm_text(s);
    let mut tokens: Vec<String> = Vec::new();
    let mut latin = String::new();
    let mut han: Vec<char> = Vec::new();

    for ch in norm.chars() {
        if is_han_char(ch) {
            if !latin.is_empty() {
                push_token(std::mem::take(&mut latin), &mut tokens);
            }
            han.push(ch);
        } else if ch.is_alphanumeric() {
            if !han.is_empty() {
                push_han_tokens(&han, &mut tokens);
                han.clear();
            }
            latin.push(ch);
        } else {
            // Whitespace / punctuation: flush both runs.
            if !latin.is_empty() {
                push_token(std::mem::take(&mut latin), &mut tokens);
            }
            if !han.is_empty() {
                push_han_tokens(&han, &mut tokens);
                han.clear();
            }
        }
    }
    if !latin.is_empty() {
        push_token(std::mem::take(&mut latin), &mut tokens);
    }
    if !han.is_empty() {
        push_han_tokens(&han, &mut tokens);
    }
    tokens
}

fn push_token(token: String, tokens: &mut Vec<String>) {
    if !is_stopword(&token) {
        tokens.push(token);
    }
}

fn push_han_tokens(han: &[char], tokens: &mut Vec<String>) {
    if han.len() == 1 {
        let t = han[0].to_string();
        if !is_stopword(&t) {
            tokens.push(t);
        }
    } else {
        for pair in han.windows(2) {
            let t: String = pair.iter().collect();
            if !is_stopword(&t) {
                tokens.push(t);
            }
        }
    }
}

/// Deduplicate tokens preserving first-occurrence order.
pub fn expand_tokens(tokens: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut seen: FxHashSet<String> = FxHashSet::default();
    for t in tokens {
        let tnorm = t.to_lowercase();
        if seen.insert(tnorm.clone()) {
            out.push(tnorm);
        }
    }
    out
}

/// Chinese single-character function words (particles, pronouns,
/// demonstratives, copula, conjunctions, prepositions). Only reachable as
/// tokens when a Han run is exactly one char; meaningful chars ("能", "要",
/// "有", "中", "上", "下" …) are deliberately excluded.
const STOPWORDS_ZH_SINGLE: &[&str] = &[
    "之", "了", "于", "也", "从", "他", "以", "们", "但", "你", "其", "则", "却", "各", "和", "在",
    "她", "如", "就", "或", "所", "把", "的", "着", "等", "而", "被", "让", "该", "这", "那", "都",
    "呢", "吗", "吧", "啊", "呀", "个", "些", "对", "向", "为", "与", "及", "且", "若", "因", "由",
    "没", "是",
];

/// Chinese two-character function words / connectives that carry little search
/// signal. Meaning-bearing words ("可以", "需要", "没有", "然后" …) are excluded.
const STOPWORDS_ZH_BIGRAM: &[&str] = &[
    "我们", "你们", "他们", "她们", "它们", "这个", "那个", "这些", "那些", "一个", "一些", "什么",
    "怎么", "怎样", "这样", "那样", "这么", "那么", "因为", "所以", "但是", "然而", "如果", "虽然",
    "而且", "于是", "因而", "以及", "并且", "或者", "之间", "之后", "之前", "之中", "以来", "以后",
    "以前",
];

/// English function words. Deliberately excludes negation ("not" is included,
/// "no" is not) and content words.
const STOPWORDS_EN: &[&str] = &[
    "a", "about", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "his", "i", "if", "in",
    "into", "is", "it", "its", "may", "me", "might", "must", "my", "not", "of", "on", "or", "our",
    "shall", "she", "should", "so", "than", "that", "the", "their", "them", "these", "they", "this",
    "those", "to", "us", "was", "we", "were", "what", "when", "where", "which", "who", "will", "with",
    "would", "you", "your",
];

fn is_stopword(t: &str) -> bool {
    STOPWORDS_ZH_SINGLE.contains(&t)
        || STOPWORDS_ZH_BIGRAM.contains(&t)
        || STOPWORDS_EN.contains(&t)
}

/// Cosine similarity between two embedding vectors.
///
/// Pure vector math on dense float embeddings (used by semantic search); this
/// is unrelated to the lexical character-similarity helpers above.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_text_maps_fullwidth_and_collapses_spaces() {
        assert_eq!(norm_text("ＡＢＣ  ｄｅｆ"), "abc def");
        assert_eq!(norm_text("Rust\tTools\n\n2024"), "rust tools 2024");
        assert_eq!(norm_text("　中文　全角空格"), "中文 全角空格");
        assert_eq!(norm_text("  leading and trailing  "), "leading and trailing");
    }

    #[test]
    fn tokenize_han_runs_to_bigrams() {
        assert_eq!(tokenize("知识库"), vec!["知识", "识库"]);
        assert_eq!(tokenize("搜索算法"), vec!["搜索", "索算", "算法"]);
        assert_eq!(tokenize("知"), vec!["知"]);
    }

    #[test]
    fn tokenize_latin_and_digits_to_words() {
        assert_eq!(tokenize("rust tools"), vec!["rust", "tools"]);
        assert_eq!(tokenize("Rust_Tools v2"), vec!["rust", "tools", "v2"]);
        assert_eq!(tokenize("C++"), vec!["c"]);
    }

    #[test]
    fn tokenize_mixed_han_latin() {
        assert_eq!(tokenize("部署到macOS"), vec!["部署", "署到", "macos"]);
    }

    #[test]
    fn tokenize_filters_stopwords() {
        assert!(tokenize("的 了 是").is_empty());
        assert!(!tokenize("我们的项目").contains(&"我们".to_string()));
        assert!(tokenize("我们的项目").contains(&"项目".to_string()));
        assert!(tokenize("the and of").is_empty());
    }

    #[test]
    fn tokenize_keeps_meaningful_terms() {
        let t = tokenize("部署流程");
        assert!(t.contains(&"部署".to_string()));
        assert!(t.contains(&"流程".to_string()));
    }
}
