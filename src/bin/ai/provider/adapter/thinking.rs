//! The "wire dialect" abstraction for the thinking switch.
//!
//! Whether thinking is enabled is a logical switch, but different gateways
//! express it with different request-body fields. This is an axis **orthogonal**
//! to provider auth / response consumption: the core layer only passes the
//! logical switch `enable`, and [`thinking_dialect_for`] selects the dialect,
//! which is responsible for writing the keys. This keeps provider adapters from
//! having to know each other's thinking wire format.
//!
//! [`thinking_dialect_for`] shares its axis with [`super::adapter_for`] (both
//! are keyed on provider + endpoint), but the Compatible axis is finer:
//! `adapter_for` returns one adapter for all compatible endpoints, while the
//! dialect further splits by endpoint into DashScope / official DeepSeek /
//! everything else, so the dialect cannot be derived from the adapter alone.
//! Apart from that subdivision, the wire output for every model in the model
//! registry is byte-identical to the old per-adapter `thinking_fields`.
//!
//! The three dialects:
//! - [`EnableThinkingDialect`]: DashScope compatible-mode (Bailian) → `enable_thinking: bool`
//! - [`DeepSeekThinkingDialect`]: DeepSeek (OpenCode Zen gateway / official api.deepseek.com) → `thinking: {"type":...}`
//! - [`NoThinkingDialect`]: plain OpenAI / OpenRouter / MiniMax etc. → sends no fields

use serde_json::{Map, Value, json};

use super::super::ApiProvider;

/// Wire encoding dialect for the thinking switch. Zero-state singleton.
pub(in crate::ai) trait ThinkingDialect: Sync {
    /// Encodes the logical switch `enable` into request-body fields. An empty
    /// Map means this dialect sends no fields. `top_level_reasoning_effort`
    /// indicates whether the request will ultimately send a top-level
    /// `reasoning_effort`; only a few special dialects need it to adjust the
    /// wire shape.
    fn fields(&self, enable: bool, top_level_reasoning_effort: Option<&str>) -> Map<String, Value>;

    /// Whether this dialect requires assistant tool-call messages to echo back
    /// a `reasoning_content` field. DeepSeek thinking-mode validates this field
    /// when continuing after a tool round; even when the gateway produced no
    /// non-empty reasoning text, the field shape must be preserved to pass the
    /// protocol check.
    fn requires_reasoning_content_echo(&self) -> bool {
        false
    }

    /// Whether lowering `reasoning_effort` actually shortens the thinking chain
    /// under this dialect.
    ///
    /// Top-level `reasoning_effort` dialects (the OpenAI-compatible family)
    /// return `true`: lowering the effort compresses the reasoning budget.
    /// Dialects like [`EnableThinkingDialect`], where thinking is controlled
    /// solely by the `enable_thinking` boolean and effort is ignored, return
    /// `false` — lowering effort for them is a no-op, so truncation retries
    /// must turn thinking off directly to give the output budget back to the
    /// visible content.
    fn reasoning_effort_reduces_thinking(&self) -> bool {
        true
    }
}

/// DashScope protocol: `{"enable_thinking": bool}`.
pub(in crate::ai) struct EnableThinkingDialect;

impl ThinkingDialect for EnableThinkingDialect {
    fn fields(
        &self,
        enable: bool,
        _top_level_reasoning_effort: Option<&str>,
    ) -> Map<String, Value> {
        let mut map = Map::new();
        map.insert("enable_thinking".to_string(), Value::Bool(enable));
        map
    }

    fn reasoning_effort_reduces_thinking(&self) -> bool {
        // Thinking is controlled solely by the `enable_thinking` boolean; the
        // request body never carries an effort, so lowering effort has zero
        // effect on the thinking-chain length.
        false
    }
}

/// DeepSeek (OpenCode Zen gateway / official api.deepseek.com):
/// `{"thinking": {"type": "enabled"|"disabled"}}`.
/// DeepSeek natively ignores `enable_thinking` and only honors this object.
/// Gateway testing (2026-08) showed the `thinking` object and the top-level
/// `reasoning_effort` can coexist without wire conflicts, and
/// `thinking:{"type":"disabled"}` takes precedence over any `reasoning_effort`,
/// making it the only reliable thinking-off switch. This dialect therefore
/// always sends the `thinking` object (the request layer still sends the
/// top-level `reasoning_effort` per adapter rules as usual).
pub(in crate::ai) struct DeepSeekThinkingDialect;

impl ThinkingDialect for DeepSeekThinkingDialect {
    fn fields(
        &self,
        enable: bool,
        _top_level_reasoning_effort: Option<&str>,
    ) -> Map<String, Value> {
        let kind = if enable { "enabled" } else { "disabled" };
        let mut map = Map::new();
        map.insert("thinking".to_string(), json!({ "type": kind }));
        map
    }

    fn requires_reasoning_content_echo(&self) -> bool {
        true
    }

    fn reasoning_effort_reduces_thinking(&self) -> bool {
        // Gateway testing: low and max reasoning_effort both keep thinking on
        // with no gradation — thinking is a binary switch (the thinking
        // object). Lowering effort on truncation retries is ineffective;
        // thinking must be turned off directly.
        false
    }
}

/// Sends no thinking fields (plain OpenAI / OpenRouter rely only on
/// `reasoning_effort`; MiniMax M2.x has always-on reasoning with no reliable
/// gateway off switch).
pub(in crate::ai) struct NoThinkingDialect;

impl ThinkingDialect for NoThinkingDialect {
    fn fields(
        &self,
        _enable: bool,
        _top_level_reasoning_effort: Option<&str>,
    ) -> Map<String, Value> {
        Map::new()
    }
}

static ENABLE_THINKING: EnableThinkingDialect = EnableThinkingDialect;
static DEEPSEEK_THINKING: DeepSeekThinkingDialect = DeepSeekThinkingDialect;
static NO_THINKING: NoThinkingDialect = NoThinkingDialect;

/// Whether the endpoint is DashScope (Alibaba Cloud Bailian) compatible-mode,
/// which controls thinking via `enable_thinking: bool`. The matching logic is
/// defined in [`super::compatible::is_dashscope_endpoint`] and reused here to
/// stay consistent with the compatible module's wire-shape decision
/// (case-insensitive / trimmed).
use super::compatible::is_dashscope_endpoint;

/// Whether the endpoint is the official DeepSeek API (`api.deepseek.com`).
/// Like the OpenCode Zen gateway, the official endpoint controls thinking with
/// the `thinking: {"type":"enabled"|"disabled"}` object (the official sample
/// curl sends this object).
fn is_deepseek_official_endpoint(endpoint: &str) -> bool {
    endpoint
        .trim()
        .to_ascii_lowercase()
        .contains("api.deepseek.com")
}

/// Picks the thinking dialect by gateway (provider + endpoint) and model,
/// decoupled from the provider auth / response-consumption axis. The dispatch
/// strictly mirrors [`super::adapter_for`]:
/// - OpenRouter endpoint → sends nothing (same as OpenAI, relies only on reasoning_effort)
/// - Alibaba → `enable_thinking` (DashScope compatible-mode)
/// - Compatible → DashScope endpoint uses `enable_thinking`; official DeepSeek
///   endpoint (api.deepseek.com) uses the `thinking` object; everything else
///   (e.g. internal modelhub, other OpenAI-compatible gateways) sends no
///   thinking field, relying only on top-level `reasoning_effort`
/// - OpenAi → DashScope endpoint uses `enable_thinking`; plain OpenAI endpoints send nothing
/// - OpenCode → DeepSeek models use the `thinking` object; everything else sends nothing
///
/// Historically `Compatible` always used the DashScope dialect, so pure
/// OpenAI-compatible endpoints mounted under the compatible provider (e.g.
/// internal modelhub) received the unknown `enable_thinking` parameter and
/// returned 400.
pub(in crate::ai) fn thinking_dialect_for(
    provider: ApiProvider,
    model: &str,
    endpoint: &str,
) -> &'static dyn ThinkingDialect {
    if endpoint
        .trim()
        .to_ascii_lowercase()
        .contains("openrouter.ai")
    {
        return &NO_THINKING;
    }
    match provider {
        ApiProvider::Alibaba => &ENABLE_THINKING,
        ApiProvider::Compatible => {
            if is_dashscope_endpoint(endpoint) {
                &ENABLE_THINKING
            } else if is_deepseek_official_endpoint(endpoint) {
                &DEEPSEEK_THINKING
            } else {
                &NO_THINKING
            }
        }
        ApiProvider::OpenAi => {
            if is_dashscope_endpoint(endpoint) {
                &ENABLE_THINKING
            } else {
                &NO_THINKING
            }
        }
        ApiProvider::OpenCode => {
            if model.to_ascii_lowercase().contains("deepseek") {
                &DEEPSEEK_THINKING
            } else {
                &NO_THINKING
            }
        }
    }
}

/// For the given model, whether lowering `reasoning_effort` actually shortens
/// the thinking chain.
///
/// Boolean `enable_thinking`-switch dialects (DashScope / compatible, e.g.
/// GLM) return `false`: the effort step in the truncation-retry ladder is a
/// no-op for them, so thinking must be turned off directly to converge.
pub(in crate::ai) fn reasoning_effort_reduces_thinking_for(
    provider: ApiProvider,
    model: &str,
    endpoint: &str,
) -> bool {
    thinking_dialect_for(provider, model, endpoint).reasoning_effort_reduces_thinking()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enable_thinking_dialect_effort_is_noop() {
        // DashScope compatible endpoints use the enable_thinking switch;
        // lowering effort is a no-op.
        assert!(!reasoning_effort_reduces_thinking_for(
            ApiProvider::Compatible,
            "glm5.2-super-relay",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ));
        assert!(!reasoning_effort_reduces_thinking_for(
            ApiProvider::Alibaba,
            "qwen-max",
            super::super::ALIBABA_DEFAULT_ENDPOINT,
        ));
    }

    #[test]
    fn non_dashscope_compatible_uses_openai_dialect() {
        // Truly OpenAI-compatible endpoints — ByteDance modelhub / Ollama /
        // self-hosted vLLM — should follow the OpenAI dialect even under the
        // `compatible` provider (reasoning_effort applies, no enable_thinking).
        assert!(reasoning_effort_reduces_thinking_for(
            ApiProvider::Compatible,
            "Kimi-K2.5",
            "https://dataagent-dev-llm.bytedance.net/api/chat/completions",
        ));
    }

    #[test]
    fn openai_family_effort_reduces_thinking() {
        // Top-level reasoning_effort dialect: lowering the effort compresses
        // the reasoning budget.
        assert!(reasoning_effort_reduces_thinking_for(
            ApiProvider::OpenAi,
            "gpt-5",
            super::super::OPENAI_DEFAULT_ENDPOINT,
        ));
    }

    #[test]
    fn deepseek_official_endpoint_uses_thinking_object_dialect() {
        // The official endpoint (api.deepseek.com) uses the DeepSeek `thinking`
        // object dialect, matching the official sample curl
        // (`"thinking": {"type": "enabled"}`): enabled when on, disabled when
        // off, independent of the top-level reasoning_effort.
        let endpoint = "https://api.deepseek.com/chat/completions";
        let dialect = thinking_dialect_for(ApiProvider::Compatible, "deepseek-flash", endpoint);
        assert_eq!(
            dialect.fields(true, Some("high")),
            json!({ "thinking": { "type": "enabled" } })
                .as_object()
                .unwrap()
                .clone()
        );
        assert_eq!(
            dialect.fields(false, None),
            json!({ "thinking": { "type": "disabled" } })
                .as_object()
                .unwrap()
                .clone()
        );
        // Thinking is a binary switch (the thinking object); lowering effort
        // does not shorten the chain.
        assert!(!reasoning_effort_reduces_thinking_for(
            ApiProvider::Compatible,
            "deepseek-flash",
            endpoint,
        ));
    }
}
