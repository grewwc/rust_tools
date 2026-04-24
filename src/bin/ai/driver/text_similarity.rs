use rust_tools::commonw::FastMap;
use rust_tools::cw::SkipSet;

pub struct TextSimilarityFeatures {
    pub normalized: String,
    pub token_set: SkipSet<String>,
    pub char_bigrams: SkipSet<String>,
    pub ngram_tf: FastMap<String, f64>,
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

pub fn extract_ngram_tf_from_normalized(normalized: &str) -> FastMap<String, f64> {
    if normalized.is_empty() {
        return FastMap::default();
    }
    let chars = format!("^{normalized}$").chars().collect::<Vec<_>>();
    let mut counts = FastMap::default();
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

pub fn build_idf_from_documents(docs: &[&FastMap<String, f64>]) -> FastMap<String, f64> {
    let mut df = FastMap::default();
    for doc in docs {
        for token in doc.keys() {
            *df.entry(token.clone()).or_insert(0usize) += 1;
        }
    }

    let total_docs = docs.len().max(1) as f64;
    let mut idf = FastMap::default();
    for (token, freq) in df {
        let value = ((1.0 + total_docs) / (1.0 + freq as f64)).ln() + 1.0;
        idf.insert(token, value);
    }
    idf
}

pub fn cosine_tfidf_similarity(
    query_tf: &FastMap<String, f64>,
    doc_tf: &FastMap<String, f64>,
    idf: &FastMap<String, f64>,
) -> f64 {
    let mut dot = 0.0;
    let mut query_norm = 0.0;
    let mut doc_norm = 0.0;

    for (token, tf) in query_tf {
        let weight = *tf * idf.get(token.as_str()).copied().unwrap_or(1.0);
        query_norm += weight * weight;
        if let Some(doc_tf) = doc_tf.get(token.as_str()) {
            let doc_weight = *doc_tf * idf.get(token.as_str()).copied().unwrap_or(1.0);
            dot += weight * doc_weight;
        }
    }
    for (token, tf) in doc_tf {
        let weight = *tf * idf.get(token.as_str()).copied().unwrap_or(1.0);
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
