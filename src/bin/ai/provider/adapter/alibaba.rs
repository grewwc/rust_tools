//! 阿里云百炼（DashScope）compatible-mode 适配器。
//!
//! 端点：`https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`。
//! 默认不发送推理强度字段，并接受 `enable_search` 扩展字段。DeepSeek / GLM
//! 等模型的推理强度 wire 由模型注册表的模型级能力声明覆盖。
//! 思考开关由 `thinking` 方言模块统一处理，不在此实现。

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
