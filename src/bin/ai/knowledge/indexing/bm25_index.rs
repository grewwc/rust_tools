/// BM25 indexing and scoring — extracted from memory_store.rs.
/// Pure BM25 implementation, no embedding or similarity logic.
use rust_tools::commonw::FastMap;
use rust_tools::commonw::FastSet;

use super::similarity;

/// A document for BM25 scoring.
pub struct Bm25Doc {
    pub tokens: Vec<String>,
    pub doc_id: usize,
}

/// BM25 parameters
pub struct Bm25Params {
    pub k1: f64,
    pub b: f64,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// Compute BM25 scores for all documents against a query.
/// Returns (doc_id, score) pairs sorted by score descending.
pub fn compute_bm25_scores(
    docs: &[Bm25Doc],
    query: &str,
    params: &Bm25Params,
) -> Vec<(usize, f64)> {
    if docs.is_empty() {
        return Vec::new();
    }

    let query_lc = query.to_lowercase();
    let nq_tokens = similarity::expand_tokens(&similarity::tokenize(&query_lc));

    // Compute document frequency and average document length
    let mut df: FastMap<String, usize> = FastMap::default();
    let mut avgdl = 0.0f64;

    for doc in docs {
        avgdl += doc.tokens.len() as f64;
        let mut set = FastSet::default();
        for t in &doc.tokens {
            if set.insert(t.clone()) {
                *df.entry(t.clone()).or_insert(0) += 1;
            }
        }
    }
    avgdl /= docs.len() as f64;

    let n_docs = docs.len() as f64;
    let mut scored: Vec<(usize, f64)> = Vec::with_capacity(docs.len());

    for doc in docs {
        let mut tf: FastMap<&str, usize> = FastMap::default();
        for t in &doc.tokens {
            *tf.entry(t.as_str()).or_insert(0) += 1;
        }

        let mut bm25 = 0.0f64;
        let dl = doc.tokens.len() as f64;

        for qt in &nq_tokens {
            let dfv = *df.get(qt).unwrap_or(&0) as f64;
            if dfv <= 0.0 {
                continue;
            }
            let idf = ((n_docs - dfv + 0.5) / (dfv + 0.5) + 1.0).ln();
            let tfv = *tf.get(qt.as_str()).unwrap_or(&0) as f64;
            if tfv <= 0.0 {
                continue;
            }
            let denom = tfv + params.k1 * (1.0 - params.b + params.b * (dl / avgdl.max(1e-6)));
            bm25 += idf * (tfv * (params.k1 + 1.0)) / denom;
        }

        scored.push((doc.doc_id, bm25));
    }

    scored
}

/// Normalize BM25 scores to [0, 1] range.
pub fn normalize_bm25_scores(scores: &[f64]) -> Vec<f64> {
    let max_score = scores.iter().cloned().fold(0.0f64, f64::max);
    if max_score <= 0.0 {
        return vec![0.0; scores.len()];
    }
    scores.iter().map(|s| *s / max_score).collect()
}
