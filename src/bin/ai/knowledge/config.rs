//! Knowledge search configuration.

use crate::ai::config_schema::AiConfig;
use crate::commonw::configw;

/// Tuning parameters for knowledge retrieval.
#[derive(Debug, Clone, Copy)]
pub struct KnowledgeConfig {
    /// Weight of vector (semantic) similarity in hybrid search, in [0.0, 1.0].
    /// 1.0 = semantic only, 0.0 = BM25 only. Default 0.4.
    pub hybrid_vector_weight: f32,
}

impl Default for KnowledgeConfig {
    fn default() -> Self {
        Self {
            hybrid_vector_weight: 0.4,
        }
    }
}

/// Load the knowledge config from the runtime config
/// (`ai.knowledge.hybrid_vector_weight`).
pub fn knowledge_config() -> KnowledgeConfig {
    let mut cfg = KnowledgeConfig::default();
    if let Some(v) = configw::get_all_config()
        .get_opt(AiConfig::KNOWLEDGE_HYBRID_VECTOR_WEIGHT)
    {
        if let Ok(w) = v.parse::<f32>() {
            cfg.hybrid_vector_weight = w.clamp(0.0, 1.0);
        }
    }
    cfg
}
