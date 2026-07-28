/// Centralized configuration for the knowledge system.
/// All thresholds, weights, TTLs, and magic numbers live here.

#[derive(Debug, Clone)]
pub struct SimilarityWeights {
    pub bm25_blend: f64,
    pub pre_score_blend: f64,
    pub embedding_blend: f64,
    pub dice_weight: f64,
    pub jaccard_weight: f64,
    pub char_overlap_weight: f64,
    pub base_contains_bonus: f64,
    pub bm25_k1: f64,
    pub bm25_b: f64,
}

impl Default for SimilarityWeights {
    fn default() -> Self {
        Self {
            bm25_blend: 0.45,
            pre_score_blend: 0.55,
            embedding_blend: 0.15,
            dice_weight: 0.5,
            jaccard_weight: 0.3,
            char_overlap_weight: 0.15,
            base_contains_bonus: 0.35,
            bm25_k1: 1.2,
            bm25_b: 0.75,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KnowledgeConfig {
    pub similarity: SimilarityWeights,
    pub hybrid_vector_weight: f32,
}

impl Default for KnowledgeConfig {
    fn default() -> Self {
        Self {
            similarity: SimilarityWeights::default(),
            hybrid_vector_weight: 0.4,
        }
    }
}
