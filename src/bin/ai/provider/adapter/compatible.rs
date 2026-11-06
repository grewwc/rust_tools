//! General Compatible adapter.
//!
//! The legacy trait defaults target DashScope compatible-mode: nested
//! `reasoning: { effort }` and caller-controlled `enable_search`. Plain
//! OpenAI-compatible gateways instead use top-level `reasoning_effort` and omit
//! `enable_search`.
//!
//! The provider-owned [`super::request_wire_shapes`] facade selects this endpoint
//! policy without exposing it to request callers. Thinking-field encoding stays
//! independent in [`super::thinking::thinking_dialect_for`], which shares the
//! exact DashScope endpoint predicate. The trait defaults remain unchanged for
//! callers without an explicit endpoint.

use serde_json::{Value, json};

use crate::ai::config_schema::AiConfig;

use super::{COMPATIBLE_DEFAULT_ENDPOINT, ProviderAdapter};

pub(super) struct CompatibleAdapter;

impl ProviderAdapter for CompatibleAdapter {
    fn label(&self) -> &'static str {
        "compatible"
    }

    fn enable_search_field(&self, requested: Option<bool>) -> Option<bool> {
        // Preserve DashScope defaults; the facade handles other endpoints.
        requested
    }

    fn reasoning_top_level<'a>(&self, _effort: Option<&'a str>) -> Option<&'a str> {
        // DashScope defaults omit top-level reasoning_effort.
        None
    }

    fn reasoning_nested(&self, effort: Option<&str>) -> Option<Value> {
        // DashScope defaults send nested reasoning: { effort }.
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

/// Match DashScope compatible-mode endpoints with the exact predicate shared
/// by the independent thinking dialect.
pub(super) fn is_dashscope_endpoint(endpoint: &str) -> bool {
    endpoint
        .trim()
        .to_ascii_lowercase()
        .contains("dashscope.aliyuncs.com")
}

/// Compatible endpoint policy: `(enable_search, top_level_effort, nested_reasoning)`.
///
/// DashScope forwards search and nests effort; other OpenAI-compatible
/// endpoints omit search and place effort at the top level.
///
/// Thinking fields are owned by [`super::thinking::thinking_dialect_for`].
pub(super) fn compatible_wire_shapes<'a>(
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
