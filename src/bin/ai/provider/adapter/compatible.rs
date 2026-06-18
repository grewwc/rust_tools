//! DashScope（阿里云百炼）compatible-mode 适配器。
//!
//! 端点：`https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`。
//! 推理强度走嵌套 `reasoning: { effort }`，并接受 `enable_search` 扩展字段。
//! 思考开关由 `thinking` 方言模块统一处理，不在此实现。

use serde_json::{Value, json};

use crate::ai::config_schema::AiConfig;

use super::{COMPATIBLE_DEFAULT_ENDPOINT, ProviderAdapter};

pub(super) struct CompatibleAdapter;

impl ProviderAdapter for CompatibleAdapter {
    fn label(&self) -> &'static str {
        "compatible"
    }

    fn enable_search_field(&self, requested: Option<bool>) -> Option<bool> {
        requested
    }

    fn reasoning_top_level<'a>(&self, _effort: Option<&'a str>) -> Option<&'a str> {
        None
    }

    fn reasoning_nested(&self, effort: Option<&str>) -> Option<Value> {
        effort.map(|effort| json!({ "effort": effort }))
    }

    fn default_endpoint(&self) -> &'static str {
        COMPATIBLE_DEFAULT_ENDPOINT
    }

    fn api_key_candidates(&self) -> &'static [&'static str] {
        &[
            AiConfig::MODEL_COMPATIBLE_API_KEY,
            AiConfig::MODEL_ALIYUN_API_KEY,
            AiConfig::MODEL_API_KEY,
        ]
    }
}
