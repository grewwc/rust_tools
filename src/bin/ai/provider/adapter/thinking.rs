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
//! The dialects:
//! - [`EnableThinkingDialect`]: DashScope compatible-mode (Bailian) → `enable_thinking: bool`
//! - [`DeepSeekThinkingDialect`]: DeepSeek (OpenCode Zen gateway / official api.deepseek.com) → `thinking: {"type":...}`
//! - [`NoThinkingDialect`]: plain OpenAI / OpenRouter etc. → sends no fields
//! - [`NoOffSwitchDialect`]: effort-only models with no reliable gateway off
//!   switch (MiniMax M2.x ("MiMo") always-on reasoning; beam, whose gateway
//!   rejects `reasoning_effort: "none"`) → sends no fields, declares
//!   [`ThinkingOffCapability::Unsupported`]
//!
//! Each dialect also declares how (or whether) it can express "thinking off"
//! via [`ThinkingDialect::thinking_off_capability`] — the per-vendor adaptation
//! table the request layer maps the user's single "off" intent onto.

use serde_json::{Map, Value, json};

use super::super::ApiProvider;

/// How this dialect can express "thinking off" on the wire.
///
/// One value per vendor dialect — the per-vendor adaptation table for the
/// logical off-switch. The request layer maps a single user intent (turn
/// thinking off, e.g. `/effort off`) onto these capabilities; it never invents
/// a wire shape on its own.
pub(in crate::ai) enum ThinkingOffCapability {
    /// A dedicated switch in the request body: DashScope `enable_thinking: false`
    /// or DeepSeek `thinking: {"type": "disabled"}`. Vendor-verified and reliable.
    RealSwitch,
    /// The only lever is `reasoning_effort`; the lowest tier `"none"` is this
    /// vendor family's off value (OpenAI / OpenRouter / Responses / plain
    /// OpenAI-compatible gateways). The request layer sends
    /// `ReasoningEffort::None`.
    EffortNone,
    /// The gateway offers no reliable way to turn thinking off (e.g. MiniMax
    /// M2.x always-on reasoning, or beam whose gateway rejects "none"). The
    /// request layer omits the thinking/effort fields on explicit off and
    /// keeps the current tier on automatic force-off, reporting the
    /// limitation honestly instead of sending a bogus switch.
    Unsupported,
}

/// Wire encoding dialect for the thinking switch. Zero-state singleton.
pub(in crate::ai) trait ThinkingDialect: Sync {
    /// Encodes the logical switch `enable` into request-body fields. An empty
    /// Map means this dialect sends no fields. `top_level_reasoning_effort`
    /// indicates whether the request will ultimately send a top-level
    /// `reasoning_effort`; only a few special dialects need it to adjust the
    /// wire shape.
    fn fields(&self, enable: bool, top_level_reasoning_effort: Option<&str>) -> Map<String, Value>;

    /// Whether this dialect requires assistant tool-call messages to echo back
    /// a `reasoning_content` field. DeepSeek thinking-mode validates the field
    /// shape when continuing after a tool round (even when the gateway produced
    /// no non-empty reasoning text). The Anthropic Messages dialect keeps the
    /// previous thinking chain so the protocol layer can replay `thinking`
    /// blocks on tool rounds. Without the echo the field is stripped before
    /// the protocol layer builds the body.
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

    /// How this dialect turns thinking off on the wire. Defaults to
    /// [`ThinkingOffCapability::Unsupported`]: a dialect must explicitly declare
    /// an off mechanism — claiming one that was never implemented would silently
    /// send wrong wire shapes.
    fn thinking_off_capability(&self) -> ThinkingOffCapability {
        ThinkingOffCapability::Unsupported
    }

    /// Maps a local reasoning-effort tier onto this vendor's wire value.
    ///
    /// The request layer resolves the tier (`low`..`max`, or `"none"` for the
    /// force-off fallback) and hands the string here; the dialect owns the
    /// per-vendor adaptation and the request layer never invents a value.
    /// Defaults to passthrough: gateways whose accepted value set is unverified
    /// keep the tier verbatim rather than guessing a mapping.
    ///
    /// Binary-switch dialects ([`EnableThinkingDialect`],
    /// [`DeepSeekThinkingDialect`]) return `None`: for models **without** a
    /// registry-declared wire placement, gateway testing shows effort has no
    /// gradation, so the tier is omitted from the wire entirely — a meaningless
    /// value would only risk a gateway rejecting non-standard tiers like
    /// `xhigh`/`max`. Models that *do* declare `reasoning_effort_wire` in the
    /// registry are vendor-verified graded and bypass this method in
    /// `resolve_reasoning_wire_controls` (request/reasoning.rs), so their tier
    /// reaches the wire. A dialect with a verified tier set may also clamp here
    /// (e.g. `max` -> `high`) in its own impl.
    fn adapt_effort<'a>(&self, effort: Option<&'a str>) -> Option<&'a str> {
        effort
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

    fn thinking_off_capability(&self) -> ThinkingOffCapability {
        ThinkingOffCapability::RealSwitch
    }

    fn adapt_effort<'a>(&self, _effort: Option<&'a str>) -> Option<&'a str> {
        // Default for models without a registry-declared `reasoning_effort_wire`:
        // DashScope thinking is a binary `enable_thinking` switch, the request
        // body never carries a meaningful effort tier (see
        // `reasoning_effort_reduces_thinking`), so omit the value. Models that
        // declare the wire placement (e.g. DashScope DeepSeek v4) bypass this
        // in resolve_reasoning_wire_controls and send the tier verbatim.
        None
    }
}

/// DeepSeek (OpenCode Zen gateway / official api.deepseek.com):
/// `{"thinking": {"type": "enabled"|"disabled"}}`.
/// DeepSeek natively ignores `enable_thinking` and only honors this object.
/// The official API documents `reasoning_effort` as `[none, low, high, max]`
/// (minimal→low, medium/xhigh→high), and its sample curl sends the `thinking`
/// object together with a top-level `reasoning_effort` — the two coexist
/// without conflicts, and `thinking:{"type":"disabled"}` takes precedence over
/// any `reasoning_effort`, making it the only reliable thinking-off switch.
/// This dialect therefore always sends the `thinking` object; the top-level
/// `reasoning_effort` is sent only when the model declares
/// `reasoning_effort_wire` in the registry (bypassed in
/// resolve_reasoning_wire_controls), and omitted for undeclared models where
/// effort has no gradation (see [`ThinkingDialect::adapt_effort`]).
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
        // Default for models without a registry-declared `reasoning_effort_wire`:
        // low and max reasoning_effort both keep thinking on with no gradation —
        // thinking is a binary switch (the thinking object). Lowering effort on
        // truncation retries is ineffective; thinking must be turned off
        // directly. Registry-declared models (DeepSeek v4) are graded and
        // handled by models::reasoning_effort_reduces_thinking instead.
        false
    }

    fn thinking_off_capability(&self) -> ThinkingOffCapability {
        ThinkingOffCapability::RealSwitch
    }

    fn adapt_effort<'a>(&self, _effort: Option<&'a str>) -> Option<&'a str> {
        // Default for models without a registry-declared `reasoning_effort_wire`:
        // low and max reasoning_effort both keep thinking on with no gradation —
        // thinking is a binary switch (the `thinking` object). A tier value is
        // a wire no-op, and omitting it avoids sending locally-invented tiers
        // (xhigh/max) whose acceptance this gateway has not verified.
        // Registry-declared models (DeepSeek v4, vendor-verified
        // `[none, low, high, max]`) bypass this in resolve_reasoning_wire_controls.
        None
    }
}

/// Sends no thinking fields (plain OpenAI / OpenRouter rely only on
/// `reasoning_effort`). Effort tiers pass through verbatim via the trait
/// default `adapt_effort` — this family's accepted value set is unverified,
/// so the request layer does not guess a mapping.
pub(in crate::ai) struct NoThinkingDialect;

impl ThinkingDialect for NoThinkingDialect {
    fn fields(
        &self,
        _enable: bool,
        _top_level_reasoning_effort: Option<&str>,
    ) -> Map<String, Value> {
        Map::new()
    }

    fn thinking_off_capability(&self) -> ThinkingOffCapability {
        // Plain OpenAI-family / OpenRouter / Responses and other effort-only
        // gateways express off as `reasoning_effort: "none"` (the value the
        // truncation force-off fallback already emits for this family).
        ThinkingOffCapability::EffortNone
    }
}

/// Effort-only models with no reliable gateway off switch. Sends no thinking
/// fields (like [`NoThinkingDialect`]) and declares
/// [`ThinkingOffCapability::Unsupported`], so the request layer never sends
/// `reasoning_effort: "none"` for these models: `/effort off` omits the
/// effort field (and the command handler reports the limitation honestly),
/// and the truncation/recovery force-off fallback keeps the current tier.
/// Effort tiers pass through verbatim via the trait default `adapt_effort`.
/// Covers the MiniMax M2.x family ("MiMo", e.g. mimo-v2.5-pro, whose
/// always-on reasoning ignores or rejects "none") and beam (e.g. beam-free,
/// whose gateway rejects "none" — accepted tiers are max/xhigh/high/medium/low).
pub(in crate::ai) struct NoOffSwitchDialect;

impl ThinkingDialect for NoOffSwitchDialect {
    fn fields(
        &self,
        _enable: bool,
        _top_level_reasoning_effort: Option<&str>,
    ) -> Map<String, Value> {
        Map::new()
    }

    fn thinking_off_capability(&self) -> ThinkingOffCapability {
        ThinkingOffCapability::Unsupported
    }
}

/// Anthropic Messages wire (`/v1/messages`) as served by the OpenCode Zen
/// gateway. The gateway does not accept the stock Anthropic
/// `thinking.type: "enabled"` + `budget_tokens` shape; it requires
/// `thinking: {"type": "adaptive"}` and controls intensity via
/// `output_config.effort` (the resolved reasoning tier, sent verbatim).
/// Thinking is off via `thinking: {"type": "disabled"}`.
pub(in crate::ai) struct AnthropicThinkingDialect;

impl ThinkingDialect for AnthropicThinkingDialect {
    fn fields(&self, enable: bool, effort: Option<&str>) -> Map<String, Value> {
        let mut map = Map::new();
        map.insert(
            "thinking".to_string(),
            json!({ "type": if enable { "adaptive" } else { "disabled" } }),
        );
        if enable {
            map.insert(
                "output_config".to_string(),
                json!({ "effort": effort.unwrap_or("medium") }),
            );
        }
        map
    }

    fn reasoning_effort_reduces_thinking(&self) -> bool {
        // The resolved tier is sent verbatim as `output_config.effort`, so
        // lowering effort genuinely reduces thinking intensity.
        true
    }

    fn thinking_off_capability(&self) -> ThinkingOffCapability {
        // `thinking: {"type": "disabled"}` is the explicit off switch.
        ThinkingOffCapability::RealSwitch
    }

    fn requires_reasoning_content_echo(&self) -> bool {
        // Keep the previous turn's thinking text on assistant messages so the
        // Messages builder can replay `thinking` blocks ahead of `tool_use`
        // rounds, preserving reasoning continuity across tool hops. The zen
        // adaptive dialect accepts plain-text thinking blocks without the
        // strict API's `signature` field.
        true
    }
}

static ENABLE_THINKING: EnableThinkingDialect = EnableThinkingDialect;
static ANTHROPIC_THINKING: AnthropicThinkingDialect = AnthropicThinkingDialect;
static DEEPSEEK_THINKING: DeepSeekThinkingDialect = DeepSeekThinkingDialect;
static NO_THINKING: NoThinkingDialect = NoThinkingDialect;
static NO_OFF_SWITCH_THINKING: NoOffSwitchDialect = NoOffSwitchDialect;

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

/// Whether the model is a MiniMax model (official "MiniMax-M*" / "MiMo"
/// family). MiniMax models share gateway endpoints with other models (e.g. the
/// OpenCode Zen gateway), so they cannot be identified by endpoint alone; the
/// model name is the discriminator, mirroring the OpenCode deepseek check.
fn is_minimax_model(model: &str) -> bool {
    let m = model.trim().to_ascii_lowercase();
    m.contains("mimo") || m.contains("minimax")
}

/// Whether the model is a beam model (e.g. beam-free). Like MiniMax, beam
/// shares gateway endpoints with models whose "none" does work (the OpenCode
/// inference gateway), so the model name is the discriminator. Wire evidence:
/// the gateway rejects `reasoning_effort: "none"` with 400 while accepting
/// max/xhigh/high/medium/low.
fn is_beam_model(model: &str) -> bool {
    model.trim().to_ascii_lowercase().contains("beam")
}

/// Picks the thinking dialect by gateway (provider + endpoint) and model,
/// decoupled from the provider auth / response-consumption axis. The dispatch
/// strictly mirrors [`super::adapter_for`]:
/// - OpenRouter endpoint → sends nothing (same as OpenAI, relies only on reasoning_effort)
/// - Alibaba → `enable_thinking` (DashScope compatible-mode)
/// - Compatible → DashScope endpoint uses `enable_thinking`; official DeepSeek
///   endpoint (api.deepseek.com) uses the `thinking` object; MiniMax/beam models use
///   [`NoOffSwitchDialect`]; everything else (e.g. internal modelhub, other
///   OpenAI-compatible gateways) sends no thinking field, relying only on
///   top-level `reasoning_effort`
/// - OpenAi → DashScope endpoint uses `enable_thinking`; MiniMax/beam models use
///   [`NoOffSwitchDialect`]; plain OpenAI endpoints send nothing
/// - OpenCode → DeepSeek models use the `thinking` object; MiniMax/beam models use
///   [`NoOffSwitchDialect`]; Anthropic Messages endpoints (`/v1/messages`)
///   use [`AnthropicThinkingDialect`]; everything else sends nothing
///
/// Historically `Compatible` always used the DashScope dialect, so pure
/// OpenAI-compatible endpoints mounted under the compatible provider (e.g.
/// internal modelhub) received the unknown `enable_thinking` parameter and
/// returned 400.
/// Whether the endpoint speaks the Anthropic Messages wire (`/v1/messages`).
/// The `thinking` object is valid on this wire regardless of the provider
/// label under which the gateway is mounted.
fn is_anthropic_messages_endpoint(endpoint: &str) -> bool {
    endpoint.trim().to_ascii_lowercase().contains("/v1/messages")
}

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
    if is_anthropic_messages_endpoint(endpoint) {
        return &ANTHROPIC_THINKING;
    }
    match provider {
        ApiProvider::Alibaba => &ENABLE_THINKING,
        ApiProvider::Compatible => {
            if is_dashscope_endpoint(endpoint) {
                &ENABLE_THINKING
            } else if is_deepseek_official_endpoint(endpoint) {
                &DEEPSEEK_THINKING
            } else if is_minimax_model(model) || is_beam_model(model) {
                &NO_OFF_SWITCH_THINKING
            } else {
                &NO_THINKING
            }
        }
        ApiProvider::OpenAi => {
            if is_dashscope_endpoint(endpoint) {
                &ENABLE_THINKING
            } else if is_minimax_model(model) || is_beam_model(model) {
                &NO_OFF_SWITCH_THINKING
            } else {
                &NO_THINKING
            }
        }
        ApiProvider::OpenCode => {
            if model.to_ascii_lowercase().contains("deepseek") {
                &DEEPSEEK_THINKING
            } else if is_minimax_model(model) || is_beam_model(model) {
                &NO_OFF_SWITCH_THINKING
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

/// Per-vendor thinking-off capability for the given model, dispatched exactly
/// like [`thinking_dialect_for`] (see [`ThinkingDialect::thinking_off_capability`]).
pub(in crate::ai) fn thinking_off_capability_for(
    provider: ApiProvider,
    model: &str,
    endpoint: &str,
) -> ThinkingOffCapability {
    thinking_dialect_for(provider, model, endpoint).thinking_off_capability()
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

    #[test]
    fn thinking_off_capability_is_per_vendor() {
        // Real switch: DashScope (Alibaba + compatible), official DeepSeek,
        // OpenCode DeepSeek.
        assert!(matches!(
            thinking_off_capability_for(
                ApiProvider::Alibaba,
                "qwen-max",
                super::super::ALIBABA_DEFAULT_ENDPOINT
            ),
            ThinkingOffCapability::RealSwitch
        ));
        assert!(matches!(
            thinking_off_capability_for(
                ApiProvider::Compatible,
                "deepseek-flash",
                "https://api.deepseek.com/chat/completions"
            ),
            ThinkingOffCapability::RealSwitch
        ));
        assert!(matches!(
            thinking_off_capability_for(
                ApiProvider::OpenCode,
                "deepseek-v4",
                "https://opencode.ai/v1"
            ),
            ThinkingOffCapability::RealSwitch
        ));
        // Effort-only: plain OpenAI, volcano-style compatible endpoints,
        // OpenCode non-DeepSeek models.
        assert!(matches!(
            thinking_off_capability_for(
                ApiProvider::OpenAi,
                "gpt-5.5",
                super::super::OPENAI_DEFAULT_ENDPOINT
            ),
            ThinkingOffCapability::EffortNone
        ));
        assert!(matches!(
            thinking_off_capability_for(
                ApiProvider::Compatible,
                "deepseek-v4-flash",
                "https://ark.cn-beijing.volces.com/api/v3"
            ),
            ThinkingOffCapability::EffortNone
        ));
        assert!(matches!(
            thinking_off_capability_for(
                ApiProvider::OpenCode,
                "glm-5.3",
                "https://opencode.ai/v1"
            ),
            ThinkingOffCapability::EffortNone
        ));
    }

    #[test]
    fn minimax_off_capability_is_unsupported() {
        // MiniMax M2.x has always-on reasoning with no reliable gateway off
        // switch; `reasoning_effort: "none"` may be ignored or rejected, so the
        // capability must be Unsupported — never EffortNone. Detected by model
        // name because the MiniMax family shares the OpenCode Zen gateway with
        // models whose "none" does work.
        assert!(matches!(
            thinking_off_capability_for(
                ApiProvider::OpenCode,
                "mimo-v2.5-pro",
                "https://opencode.ai/inference/go/v1/chat/completions"
            ),
            ThinkingOffCapability::Unsupported
        ));
        assert!(matches!(
            thinking_off_capability_for(
                ApiProvider::Compatible,
                "MiniMax-M2",
                "https://api.minimaxi.com/v1"
            ),
            ThinkingOffCapability::Unsupported
        ));
        assert!(matches!(
            thinking_off_capability_for(
                ApiProvider::OpenAi,
                "mimo-v2.5-free",
                super::super::OPENAI_DEFAULT_ENDPOINT
            ),
            ThinkingOffCapability::Unsupported
        ));
        // The dialect sends no thinking fields (like NoThinkingDialect) and
        // passes effort tiers through verbatim (no verified value set).
        let dialect = thinking_dialect_for(
            ApiProvider::OpenCode,
            "mimo-v2.5-pro",
            "https://opencode.ai/inference/go/v1/chat/completions",
        );
        assert!(dialect.fields(true, Some("high")).is_empty());
        assert_eq!(dialect.adapt_effort(Some("high")), Some("high"));
    }

    #[test]
    fn beam_off_capability_is_unsupported() {
        // beam-free shares the OpenCode inference gateway with models whose
        // "none" works, but its own gateway rejects `reasoning_effort: "none"`
        // with 400 (accepted tiers: max/xhigh/high/medium/low) — detected by
        // model name, like MiniMax. The graded tiers still reduce thinking.
        assert!(matches!(
            thinking_off_capability_for(
                ApiProvider::OpenCode,
                "beam-free",
                "https://opencode.ai/inference/go/v1/chat/completions"
            ),
            ThinkingOffCapability::Unsupported
        ));
        assert!(reasoning_effort_reduces_thinking_for(
            ApiProvider::OpenCode,
            "beam-free",
            "https://opencode.ai/inference/go/v1/chat/completions"
        ));
        // No thinking fields; tiers pass through verbatim.
        let dialect = thinking_dialect_for(
            ApiProvider::OpenCode,
            "beam-free",
            "https://opencode.ai/inference/go/v1/chat/completions",
        );
        assert!(dialect.fields(true, Some("max")).is_empty());
        assert_eq!(dialect.adapt_effort(Some("max")), Some("max"));
    }

    #[test]
    fn effort_value_adaptation_is_per_vendor() {
        // Dialect default for models without a registry-declared
        // reasoning_effort_wire: binary-switch vendors (DashScope / DeepSeek)
        // omit the tier value — gateway testing showed effort has no gradation
        // there, and a meaningless value would only risk a gateway rejecting
        // non-standard tiers. Registry-declared models bypass adapt_effort in
        // resolve_reasoning_wire_controls (request/reasoning.rs).
        assert!(thinking_dialect_for(
            ApiProvider::Alibaba,
            "qwen-max",
            super::super::ALIBABA_DEFAULT_ENDPOINT
        )
        .adapt_effort(Some("max"))
        .is_none());
        assert!(thinking_dialect_for(
            ApiProvider::Compatible,
            "deepseek-flash",
            "https://api.deepseek.com/chat/completions"
        )
        .adapt_effort(Some("high"))
        .is_none());
        // Effort-only / OpenAI-family dialects pass the tier through verbatim:
        // their accepted value set is unverified, and guessing a mapping would
        // be worse than sending the tier.
        assert_eq!(
            thinking_dialect_for(
                ApiProvider::OpenAi,
                "gpt-5.5",
                super::super::OPENAI_DEFAULT_ENDPOINT
            )
            .adapt_effort(Some("max")),
            Some("max")
        );
        assert_eq!(
            thinking_dialect_for(
                ApiProvider::Compatible,
                "deepseek-v4-flash",
                "https://ark.cn-beijing.volces.com/api/v3"
            )
            .adapt_effort(Some("xhigh")),
            Some("xhigh")
        );
    }

    #[test]
    fn dialect_without_declared_off_is_unsupported() {
        // A dialect that never declares an off mechanism must not claim one:
        // the safe default is Unsupported, so a missing override surfaces as a
        // honest "cannot turn off" instead of a wrong wire shape.
        struct UnknownDialect;
        impl ThinkingDialect for UnknownDialect {
            fn fields(&self, _enable: bool, _effort: Option<&str>) -> Map<String, Value> {
                Map::new()
            }
        }
        assert!(matches!(
            UnknownDialect.thinking_off_capability(),
            ThinkingOffCapability::Unsupported
        ));
    }

    #[test]
    fn anthropic_dialect_enabled_sends_adaptive_thinking_and_effort() {
        let dialect = thinking_dialect_for(
            ApiProvider::OpenCode,
            "claude-opus-5-5",
            "https://opencode.ai/inference/go/v1/messages",
        );
        let fields = dialect.fields(true, Some("xhigh"));
        assert_eq!(fields["thinking"]["type"], "adaptive");
        assert_eq!(fields["output_config"]["effort"], "xhigh");
        // The resolved tier is sent verbatim.
        assert_eq!(
            dialect.fields(true, Some("low"))["output_config"]["effort"],
            "low"
        );
        assert!(dialect.reasoning_effort_reduces_thinking());
        assert!(matches!(
            dialect.thinking_off_capability(),
            ThinkingOffCapability::RealSwitch
        ));
    }

    #[test]
    fn anthropic_dialect_disabled_sends_the_disabled_type() {
        let dialect = thinking_dialect_for(
            ApiProvider::OpenCode,
            "claude-opus-5-5",
            "https://opencode.ai/inference/go/v1/messages",
        );
        let fields = dialect.fields(false, None);
        assert_eq!(fields["thinking"]["type"], "disabled");
        assert!(!fields.contains_key("output_config"));
    }

    #[test]
    fn messages_endpoints_get_anthropic_dialect_other_wires_stay_unchanged() {
        // The same model on a chat-completions endpoint keeps the old dialect:
        // the Anthropic `thinking` object is only valid on the Messages wire.
        assert!(thinking_dialect_for(
            ApiProvider::OpenCode,
            "claude-opus-5-5",
            "https://opencode.ai/inference/go/v1/chat/completions"
        )
        .fields(true, Some("high"))
        .is_empty());
    }
}
