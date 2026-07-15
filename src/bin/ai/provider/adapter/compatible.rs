//! 「Compatible」通用适配器。
//!
//! 历史上本适配器专门服务 DashScope compatible-mode（阿里云百炼），所以默认
//! wire 形状是：
//! - 嵌套 `reasoning: { effort }`（无顶层 `reasoning_effort`）
//! - 透传 `enable_search`
//! - 思考开关由 `thinking` 方言模块以 `enable_thinking: bool` 发送
//!
//! 但 `models.json` 里也把大量「纯 OpenAI 兼容」端点（如内部 modelhub、其他
//! 第三方 OpenAI 兼容网关）挂在 `adapter: "compatible"` 下。这些端点严格遵循
//! OpenAI 协议，只接受顶层 `reasoning_effort`，不识别 `enable_thinking` /
//! `enable_search` / 嵌套 `reasoning`，发送这些字段会直接 400
//! （`unknown_parameter: 'enable_thinking'`）。
//!
//! 因此本文件提供两组静态 helper：
//! - [`dashscope_defaults`]：DashScope compatible-mode 的默认 wire 形状；
//! - [`openai_compatible_defaults`]：纯 OpenAI 兼容端点的默认 wire 形状。
//!
//! 具体使用哪一组，由调用方
//! （`super::super::request::reasoning::resolve_reasoning_wire_controls`）
//! 按 provider + endpoint 判定，和 [`super::thinking::thinking_dialect_for`] 的
//! 分支保持一一对应。trait 本身仍保留 DashScope 默认，以维持对「未显式传 endpoint」
//! 场景的向后兼容。

use serde_json::{Value, json};

use crate::ai::config_schema::AiConfig;

use super::{COMPATIBLE_DEFAULT_ENDPOINT, ProviderAdapter};

pub(super) struct CompatibleAdapter;

impl ProviderAdapter for CompatibleAdapter {
    fn label(&self) -> &'static str {
        "compatible"
    }

    fn enable_search_field(&self, requested: Option<bool>) -> Option<bool> {
        // 默认（DashScope）透传 enable_search。非 DashScope 端点由调用方
        // 直接覆盖为 None，见 compatible_wire_shapes。
        requested
    }

    fn reasoning_top_level<'a>(&self, _effort: Option<&'a str>) -> Option<&'a str> {
        // 默认（DashScope）不发顶层 reasoning_effort。
        None
    }

    fn reasoning_nested(&self, effort: Option<&str>) -> Option<Value> {
        // 默认（DashScope）发嵌套 reasoning: { effort }。
        effort.map(|effort| json!({ "effort": effort }))
    }

    fn default_endpoint(&self) -> &'static str {
        COMPATIBLE_DEFAULT_ENDPOINT
    }

    fn api_key_candidates(&self) -> &'static [&'static str] {
        &[
            AiConfig::MODEL_COMPATIBLE_API_KEY,
            AiConfig::MODEL_ALIBABA_API_KEY,
            AiConfig::MODEL_ALIYUN_API_KEY,
            AiConfig::MODEL_API_KEY,
        ]
    }
}

/// endpoint 是否命中 DashScope compatible-mode。
///
/// 注意：和 [`super::thinking::is_dashscope_endpoint`] 保持一致，两处使用
/// 相同判定规则；若扩展更多兼容网关，在同一处加条件即可。
pub(in crate::ai) fn is_dashscope_endpoint(endpoint: &str) -> bool {
    endpoint
        .trim()
        .to_ascii_lowercase()
        .contains("dashscope.aliyuncs.com")
}

/// compatible 端点的 wire 形状：根据 endpoint 返回三元组
/// `(enable_search, top_level_effort, nested_reasoning)`。
///
/// - DashScope 端点 → (透传 requested, None, Some({effort}))
/// - 其他 OpenAI 兼容端点 → (None, Some(effort), None)
///
/// 注意：思考开关字段（`enable_thinking` / `thinking` 对象）不在此处理，
/// 由 [`super::thinking::thinking_dialect_for`] 负责。
pub(in crate::ai) fn compatible_wire_shapes<'a>(
    endpoint: &str,
    requested_enable_search: Option<bool>,
    effort: Option<&'a str>,
) -> (Option<bool>, Option<&'a str>, Option<Value>) {
    if is_dashscope_endpoint(endpoint) {
        (
            requested_enable_search,
            None,
            effort.map(|e| json!({ "effort": e })),
        )
    } else {
        (None, effort, None)
    }
}
