//! OpenRouter adapter.
//!
//! OpenRouter is not an independent `ApiProvider` variant but an endpoint variant of the
//! OpenAI protocol (the endpoint contains `openrouter.ai`). The request-body fields are
//! identical to OpenAI; only the log label and the default endpoint / API key candidate
//! chain differ.

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
