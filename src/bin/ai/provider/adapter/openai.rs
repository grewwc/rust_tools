//! OpenAI 协议适配器。
//!
//! 默认实现即「OpenAI 兼容族」通用行为（顶层 `reasoning_effort`、不发送扩展
//! 字段）。思考开关的 wire 编码由 `thinking` 方言模块统一按 endpoint/model 处理
//! （含「openai-provider 模型托管在 DashScope 端点」这一情形），本适配器不再
//! 参与思考字段，因此无需认识 DashScope 的 wire 格式。

use crate::ai::config_schema::AiConfig;

use super::{OPENAI_DEFAULT_ENDPOINT, ProviderAdapter};

pub(super) struct OpenAiAdapter;

impl ProviderAdapter for OpenAiAdapter {
    fn label(&self) -> &'static str {
        "openai"
    }

    fn default_endpoint(&self) -> &'static str {
        OPENAI_DEFAULT_ENDPOINT
    }

    fn api_key_candidates(&self) -> &'static [&'static str] {
        &[
            AiConfig::MODEL_OPENROUTER_API_KEY,
            AiConfig::MODEL_OPENAI_API_KEY,
            AiConfig::MODEL_API_KEY,
        ]
    }
}
