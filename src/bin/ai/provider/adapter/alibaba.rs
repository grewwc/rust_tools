//! Alibaba Bailian (DashScope) compatible-mode adapter.
//!
//! Endpoint: `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`.
//! By default no reasoning-effort field is sent, and the `enable_search` extension field
//! is accepted. Reasoning-effort wire behavior for models such as DeepSeek / GLM is
//! covered by the model-level capability declarations in the model registry.
//! The thinking toggle is handled uniformly by the `thinking` dialect module, not here.

use crate::ai::config_schema::AiConfig;

use super::{ALIBABA_DEFAULT_ENDPOINT, ProviderAdapter};

pub(super) struct AlibabaAdapter;

impl ProviderAdapter for AlibabaAdapter {
    fn label(&self) -> &'static str {
        "alibaba"
    }

    fn enable_search_field(&self, requested: Option<bool>) -> Option<bool> {
        requested
    }

    fn reasoning_top_level<'a>(&self, _effort: Option<&'a str>) -> Option<&'a str> {
        None
    }

    fn default_endpoint(&self) -> &'static str {
        ALIBABA_DEFAULT_ENDPOINT
    }

    fn api_key_candidates(&self) -> &'static [&'static str] {
        &[
            AiConfig::MODEL_ALIBABA_API_KEY,
            AiConfig::MODEL_ALIYUN_API_KEY,
            AiConfig::MODEL_COMPATIBLE_API_KEY,
            AiConfig::MODEL_API_KEY,
        ]
    }
}
