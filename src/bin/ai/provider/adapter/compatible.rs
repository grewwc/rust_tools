//! Generic compatible adapter with DashScope defaults.
//!
//! [`CompatibleAdapter`] passes through `enable_search` and nests effort under
//! `reasoning`. [`compatible_wire_shapes`] selects these defaults for DashScope
//! endpoints and top-level `reasoning_effort` for other compatible endpoints.
//!
//! Endpoint-specific shapes keep DashScope extension fields out of requests to
//! other gateways. Thinking-switch fields belong to [`super::thinking`].

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
            AiConfig::MODEL_ALIBABA_API_KEY,
            AiConfig::MODEL_ALIYUN_API_KEY,
            AiConfig::MODEL_API_KEY,
        ]
    }
}

/// Matches DashScope compatible-mode endpoints.
///
/// The thinking dialect reuses this predicate to keep its endpoint selection
/// consistent with [`compatible_wire_shapes`].
pub(in crate::ai) fn is_dashscope_endpoint(endpoint: &str) -> bool {
    endpoint
        .trim()
        .to_ascii_lowercase()
        .contains("dashscope.aliyuncs.com")
}

/// Selects compatible wire fields by endpoint as
/// `(enable_search, top_level_effort, nested_reasoning)`.
///
/// DashScope preserves search and nests an effort when supplied. Other
/// endpoints omit search and forward the effort at the top level.
///
/// Thinking-switch fields belong to [`super::thinking::thinking_dialect_for`].
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
