/// Keyword (BM25) search over JSONL store.
/// Combines BM25 scoring with text similarity for ranking.
use super::super::config::KnowledgeConfig;
use super::super::entry::KnowledgeEntry;
use super::super::indexing::{bm25_index, similarity};
use super::super::storage::jsonl_store::JsonlStore;

/// Search entries by keyword query.
pub fn keyword_search(
    store: &JsonlStore,
    query: &str,
    limit: usize,
    config: &KnowledgeConfig,
) -> Result<Vec<(KnowledgeEntry, f64)>, String> {
    let entries = store.list_all()?;
    if entries.is_empty() {
        return Ok(Vec::new());
    }

    let query_lc = query.to_lowercase();

    // Build BM25 docs
    let docs: Vec<bm25_index::Bm25Doc> = entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let full = entry.search_text().to_lowercase();
            let tokens = similarity::expand_tokens(&similarity::tokenize(&full));
            bm25_index::Bm25Doc { tokens, doc_id: i }
        })
        .collect();

    // Compute BM25 scores
    let bm25_scores = bm25_index::compute_bm25_scores(
        &docs,
        query,
        &bm25_index::Bm25Params {
            k1: config.similarity.bm25_k1,
            b: config.similarity.bm25_b,
        },
    );
    let bm25_vals: Vec<f64> = bm25_scores.iter().map(|(_, s)| *s).collect();
    let bm25_normalized = bm25_index::normalize_bm25_scores(&bm25_vals);

    // Compute text similarity scores
    let mut pre_scores: Vec<f64> = Vec::with_capacity(entries.len());
    for entry in &entries {
        let sim = compute_entry_similarity(entry, &query_lc, config);
        let pre = config.similarity.pre_score_blend * sim;
        pre_scores.push(pre);
    }

    // Blend BM25 and pre-scores
    let mut scored: Vec<(f64, usize)> = entries
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let s = config.similarity.bm25_blend * bm25_normalized[i]
                + (1.0 - config.similarity.bm25_blend) * pre_scores[i];
            (s, i)
        })
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Rescore top candidates with embedding if available
    let cap = limit.saturating_mul(10).min(200).max(limit);
    let mut top_idx: Vec<(f64, usize)> = scored.iter().take(cap).copied().collect();

    if let Some(qv) = super::super::indexing::embedder::embed_text(&query_lc) {
        let mut rescored: Vec<(f64, usize)> = Vec::with_capacity(top_idx.len());
        for &(_s, i) in &top_idx {
            let full = entries[i].search_text();
            let emb = super::super::indexing::embedder::embed_text(&full)
                .as_ref()
                .map(|v| similarity::cosine_similarity(&qv, v))
                .unwrap_or(0.0);
            let final_s = (1.0 - config.similarity.embedding_blend) * _s
                + config.similarity.embedding_blend * emb as f64;
            rescored.push((final_s, i));
        }
        rescored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        top_idx = rescored.into_iter().take(limit).collect();
    } else {
        top_idx.truncate(limit);
    }

    let mut out = Vec::with_capacity(top_idx.len());
    for (s, i) in top_idx {
        out.push((entries[i].clone(), s));
    }
    Ok(out)
}

/// Compute text similarity between an entry and query.
fn compute_entry_similarity(
    entry: &KnowledgeEntry,
    query_lc: &str,
    config: &KnowledgeConfig,
) -> f64 {
    let base_contains = if entry.note.to_lowercase().contains(query_lc)
        || entry.category.to_lowercase().contains(query_lc)
        || entry
            .tags
            .iter()
            .any(|t| t.to_lowercase().contains(query_lc))
        || entry
            .source
            .as_ref()
            .is_some_and(|s| s.to_lowercase().contains(query_lc))
    {
        config.similarity.base_contains_bonus
    } else {
        0.0
    };

    let full = entry.search_text();
    let nq = similarity::norm_text(query_lc);
    let ne = similarity::norm_text(&full);
    let d = similarity::dice_coefficient(&similarity::bigrams(&nq), &similarity::bigrams(&ne));
    let tq = similarity::expand_tokens(&similarity::tokenize(query_lc));
    let te = similarity::expand_tokens(&similarity::tokenize(&full.to_lowercase()));
    let j = similarity::jaccard(&tq, &te);
    let co = similarity::char_overlap(&nq, &ne);

    let s = config.similarity.dice_weight * d
        + config.similarity.jaccard_weight * j
        + config.similarity.char_overlap_weight * co
        + base_contains;

    s.clamp(0.0, 1.0)
}
