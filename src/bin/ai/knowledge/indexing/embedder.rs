//! Embedding provider for semantic knowledge search.
//!
//! Calls a remote OpenAI-compatible `/embeddings` endpoint (default: Doubao
//! `doubao-embedding-vision`, 384 dims), configured via the `ai.embedding.*`
//! keys. When no API key is configured or the endpoint is unreachable the
//! provider stays absent (`is_ready() == false`) and knowledge search
//! degrades to BM25-only hybrid results.

use std::sync::OnceLock;

use reqwest::blocking::Client;
use serde_json::json;

use crate::ai::config_schema::AiConfig;
use crate::commonw::configw;

pub const DEFAULT_EMBEDDING_ENDPOINT: &str =
    "https://ark.cn-beijing.volces.com/api/coding/v3/embeddings";
pub const DEFAULT_EMBEDDING_MODEL: &str = "doubao-embedding-vision";
pub const DEFAULT_EMBEDDING_DIM: usize = 384;

/// Embedding provider trait; a provider produces dense vectors for texts.
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a single text; `None` on failure or empty input.
    fn embed(&self, text: &str) -> Option<Vec<f32>>;
    /// Embed a batch of texts; `None` on failure.
    fn embed_texts(&self, texts: &[&str]) -> Option<Vec<Vec<f32>>>;
    /// Dimension of the produced vectors.
    fn dimension(&self) -> usize;
}

/// Remote OpenAI-compatible embedding provider (Doubao Ark / generic).
pub struct RemoteEmbeddingProvider {
    client: Client,
    endpoint: String,
    api_key: String,
    model: String,
}

impl RemoteEmbeddingProvider {
    /// Build from runtime config; `None` when embedding is disabled or the
    /// API key is missing.
    pub fn from_config() -> Option<Self> {
        let cfg = configw::get_all_config();
        if cfg
            .get_opt(AiConfig::EMBEDDING_ENABLE)
            .map_or(true, |v| v != "true")
        {
            return None;
        }
        let api_key = cfg
            .get_opt(AiConfig::EMBEDDING_API_KEY)
            .or_else(|| cfg.get_opt(AiConfig::MODEL_VOLCANO_API_KEY))
            .or_else(|| cfg.get_opt(AiConfig::MODEL_ALIYUN_API_KEY))?
            .trim()
            .to_string();
        if api_key.is_empty() {
            return None;
        }
        let endpoint = cfg
            .get_opt(AiConfig::EMBEDDING_ENDPOINT)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_EMBEDDING_ENDPOINT.to_string());
        let model = cfg
            .get_opt(AiConfig::EMBEDDING_MODEL)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_EMBEDDING_MODEL.to_string());
        let timeout_ms = cfg
            .get_opt(AiConfig::EMBEDDING_TIMEOUT_MS)
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(30_000);
        let client = Client::builder()
            .timeout(std::time::Duration::from_millis(timeout_ms))
            .build()
            .ok()?;
        Some(Self {
            client,
            endpoint,
            api_key,
            model,
        })
    }
}

impl EmbeddingProvider for RemoteEmbeddingProvider {
    fn embed(&self, text: &str) -> Option<Vec<f32>> {
        self.embed_texts(&[text]).map(|mut v| v.remove(0))
    }

    fn embed_texts(&self, texts: &[&str]) -> Option<Vec<Vec<f32>>> {
        let trimmed: Vec<&str> = texts
            .iter()
            .map(|t| t.trim())
            .filter(|t| !t.is_empty())
            .collect();
        if trimmed.is_empty() {
            return Some(Vec::new());
        }
        let body = json!({ "model": self.model, "input": trimmed });
        let resp = self
            .client
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let payload: serde_json::Value = resp.json().ok()?;
        let data = payload.get("data")?.as_array()?;
        let mut out = Vec::with_capacity(data.len());
        for item in data {
            let emb = item.get("embedding")?.as_array()?;
            let vec: Vec<f32> = emb
                .iter()
                .filter_map(|x| x.as_f64().map(|v| v as f32))
                .collect();
            out.push(vec);
        }
        Some(out)
    }

    fn dimension(&self) -> usize {
        DEFAULT_EMBEDDING_DIM
    }
}

/// Process-wide provider, installed once by `warm_up`.
static PROVIDER: OnceLock<Option<RemoteEmbeddingProvider>> = OnceLock::new();

/// Install the embedding provider from config (best-effort; no-op on failure).
pub fn warm_up() {
    let provider = RemoteEmbeddingProvider::from_config();
    let _ = PROVIDER.set(provider);
}

/// True when a usable embedding provider is installed.
pub fn is_ready() -> bool {
    PROVIDER.get().is_some_and(|p| p.is_some())
}

/// Model name of the installed provider (used to fingerprint the vector index).
pub fn current_model() -> Option<&'static str> {
    PROVIDER
        .get()
        .and_then(|p| p.as_ref())
        .map(|p| p.model.as_str())
}

/// Embed a single text for semantic search.
pub fn embed_text(text: &str) -> Option<Vec<f32>> {
    PROVIDER.get().and_then(|p| p.as_ref())?.embed(text)
}

/// Embed a batch of texts for index building.
pub fn embed_texts(texts: &[&str]) -> Option<Vec<Vec<f32>>> {
    PROVIDER.get().and_then(|p| p.as_ref())?.embed_texts(texts)
}
