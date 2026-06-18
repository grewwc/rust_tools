//! OpenRouter 适配器。
//!
//! OpenRouter 不是独立的 `ApiProvider` 变体，而是 OpenAI 协议的 endpoint 变体
//! （endpoint 含 `openrouter.ai`）。请求体字段与 OpenAI 完全一致，仅日志标签和
//! 默认 endpoint / API key 候选链不同。

use crate::ai::config_schema::AiConfig;

use super::{OPENROUTER_ENDPOINT, ProviderAdapter};

pub(super) struct OpenRouterAdapter;

impl ProviderAdapter for OpenRouterAdapter {
    fn label(&self) -> &'static str {
        "openrouter"
    }

    fn default_endpoint(&self) -> &'static str {
        OPENROUTER_ENDPOINT
    }

    fn api_key_candidates(&self) -> &'static [&'static str] {
        &[
            AiConfig::MODEL_OPENROUTER_API_KEY,
            AiConfig::MODEL_OPENAI_API_KEY,
            AiConfig::MODEL_API_KEY,
        ]
    }
}
