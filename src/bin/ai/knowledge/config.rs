/// Centralized configuration for the knowledge system.
/// All thresholds, weights, TTLs, and magic numbers live here.

#[derive(Debug, Clone)]
pub struct SearchThresholds {
    pub min_score_guideline: f64,
    pub min_score_knowledge: f64,
    pub dedup_similarity_guideline: f64,
    pub dedup_similarity_knowledge: f64,
    pub max_entries_with_project: usize,
    pub max_entries_without_project: usize,
    pub high_confidence_min_matches: usize,
    pub high_confidence_min_priority: u8,
}

impl Default for SearchThresholds {
    fn default() -> Self {
        Self {
            min_score_guideline: 0.15,
            min_score_knowledge: 0.3,
            dedup_similarity_guideline: 0.7,
            dedup_similarity_knowledge: 0.65,
            max_entries_with_project: 10,
            max_entries_without_project: 6,
            high_confidence_min_matches: 2,
            high_confidence_min_priority: 180,
        }
    }
}

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
pub struct TtlConfig {
    pub project_structure: u64,
    pub code_content: u64,
    pub api_doc: u64,
    pub general: u64,
}

impl Default for TtlConfig {
    fn default() -> Self {
        Self {
            project_structure: 1800,
            code_content: 300,
            api_doc: 600,
            general: 3600,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaintenanceConfig {
    pub auto_rotate_max_bytes: u64,
    pub auto_gc_days: i64,
    pub auto_gc_min_keep: usize,
    pub auto_maintain_probability: f64,
    pub archives_retain_days: i64,
    pub archives_keep_last: usize,
    pub archives_max_bytes: u64,
    pub guidelines_max_days: i64,
    pub guidelines_max_chars: usize,
    pub guidelines_group_weights: [usize; 4],
}

impl Default for MaintenanceConfig {
    fn default() -> Self {
        Self {
            auto_rotate_max_bytes: 8 * 1024 * 1024,
            auto_gc_days: 30,
            auto_gc_min_keep: 200,
            auto_maintain_probability: 0.05,
            archives_retain_days: 60,
            archives_keep_last: 10,
            archives_max_bytes: 64 * 1024 * 1024,
            guidelines_max_days: 30,
            guidelines_max_chars: 1200,
            guidelines_group_weights: [35, 40, 15, 10],
        }
    }
}

#[derive(Debug, Clone)]
pub struct KnowledgeConfig {
    pub thresholds: SearchThresholds,
    pub similarity: SimilarityWeights,
    pub ttl: TtlConfig,
    pub maintenance: MaintenanceConfig,
    pub embedding_dim: usize,
    pub hybrid_vector_weight: f32,
    pub default_search_limit: usize,
}

impl Default for KnowledgeConfig {
    fn default() -> Self {
        Self {
            thresholds: SearchThresholds::default(),
            similarity: SimilarityWeights::default(),
            ttl: TtlConfig::default(),
            maintenance: MaintenanceConfig::default(),
            embedding_dim: 384,
            hybrid_vector_weight: 0.4,
            default_search_limit: 5,
        }
    }
}

impl KnowledgeConfig {
    pub fn from_config_file() -> Self {
        use crate::commonw::configw;
        let cfg = configw::get_all_config();
        let mut config = Self::default();

        if let Some(v) = cfg
            .get_opt("ai.knowledge.thresholds.min_score_guideline")
            .and_then(|v| v.parse::<f64>().ok())
        {
            config.thresholds.min_score_guideline = v;
        }
        if let Some(v) = cfg
            .get_opt("ai.knowledge.thresholds.min_score_knowledge")
            .and_then(|v| v.parse::<f64>().ok())
        {
            config.thresholds.min_score_knowledge = v;
        }
        if let Some(v) = cfg
            .get_opt("ai.knowledge.maintenance.guidelines_max_chars")
            .and_then(|v| v.parse::<usize>().ok())
        {
            config.maintenance.guidelines_max_chars = v;
        }
        if let Some(v) = cfg
            .get_opt("ai.knowledge.search.default_limit")
            .and_then(|v| v.parse::<usize>().ok())
        {
            config.default_search_limit = v;
        }

        config
    }
}
