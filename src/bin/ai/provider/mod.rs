mod adapter;

#[cfg(test)]
pub(in crate::ai) use adapter::{
    ALIBABA_DEFAULT_ENDPOINT, OPENCODE_DEFAULT_ENDPOINT, OPENROUTER_ENDPOINT, alibaba_adapter,
    openai_adapter, opencode_adapter,
};
pub(in crate::ai) use adapter::{
    ProviderAdapter, adapter_for, compatible_wire_shapes, reasoning_effort_reduces_thinking_for,
    thinking_dialect_for,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub(super) enum ApiProvider {
    #[default]
    Compatible,
    #[serde(alias = "aliyun", alias = "dashscope")]
    Alibaba,
    #[serde(alias = "openai")]
    OpenAi,
    #[serde(alias = "opencode")]
    OpenCode,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    serde::Deserialize,
    serde::Serialize,
    Default,
)]
#[serde(rename_all = "snake_case")]
pub(super) enum ModelQualityTier {
    Basic,
    #[default]
    Standard,
    Strong,
    Flagship,
}

/// LLM reasoning-effort tiers. Protocols like OpenAI / OpenRouter / OpenCode use a top-level
/// `reasoning_effort`; the DashScope compatible provider uses a nested
/// `reasoning.effort`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ReasoningEffort {
    /// Explicit lowest tier: sends `reasoning_effort: "none"` (`reasoning: {"effort":"none"}` on
    /// the Responses wire). Differs from "omit the field" (`override = Some(None)`, where the
    /// server default takes over and keeps thinking on): `None` is the true reasoning floor,
    /// used by gpt-5.x in place of the removed `minimal`. It is emitted only by the truncation
    /// ladder's last-resort force-off fallback (`thinking_disabled_override`, see
    /// `request::reasoning::apply_thinking_force_off_effort`); the graduated ladder itself
    /// deliberately never sends it (orchestrator.rs) because an explicit none during convergence
    /// retries would castrate reasoning instead of shrinking it.
    None,
    Minimal,
    Low,
    Medium,
    High,
    #[serde(rename = "xhigh", alias = "x_high", alias = "extra_high")]
    XHigh,
    #[serde(alias = "maximum")]
    Max,
}

impl ReasoningEffort {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }

    /// Parses a string from the CLI / `/model effort` command. Only recognizes the five
    /// tier literals case-insensitively; control semantics like `off`/`none`/`auto` are left to the caller.
    pub(super) fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "minimal" | "min" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" | "mid" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" | "extra_high" | "extra-high" => Some(Self::XHigh),
            "max" | "maximum" => Some(Self::Max),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ApiProvider, ModelQualityTier, ReasoningEffort};
    use crate::ai::model_names::ModelDef;

    #[test]
    fn adapter_defaults_to_compatible() {
        let def: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","is_vl":false,"search_enabled":true,"tools_default_enabled":true}"#,
        )
        .unwrap();
        assert_eq!(def.adapter, ApiProvider::Compatible);
    }

    #[test]
    fn quality_tier_defaults_to_standard() {
        let def: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","is_vl":false,"search_enabled":true,"tools_default_enabled":true}"#,
        )
        .unwrap();
        assert_eq!(def.quality_tier, ModelQualityTier::Standard);
    }

    #[test]
    fn parses_openai_adapter_and_flagship_tier() {
        let def: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","adapter":"openai","quality_tier":"flagship","is_vl":true,"search_enabled":false,"tools_default_enabled":true}"#,
        )
        .unwrap();
        assert_eq!(def.adapter, ApiProvider::OpenAi);
        assert_eq!(def.quality_tier, ModelQualityTier::Flagship);
    }

    #[test]
    fn parses_provider_alias_into_adapter() {
        let def: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"alibaba","quality_tier":"strong","is_vl":true,"search_enabled":false,"tools_default_enabled":true}"#,
        )
        .unwrap();
        assert_eq!(def.adapter, ApiProvider::Alibaba);
    }

    #[test]
    fn parses_opencode_adapter_alias() {
        let def: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","adapter":"opencode","quality_tier":"basic","is_vl":false,"search_enabled":false,"tools_default_enabled":true}"#,
        )
        .unwrap();
        assert_eq!(def.adapter, ApiProvider::OpenCode);
    }

    #[test]
    fn parses_platform_independently_from_adapter() {
        let def: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","platform":"volcano","adapter":"compatible","is_vl":false,"search_enabled":true,"tools_default_enabled":true}"#,
        )
        .unwrap();
        assert_eq!(def.adapter, ApiProvider::Compatible);
        assert_eq!(def.platform.as_deref(), Some("volcano"));
    }

    #[test]
    fn reasoning_effort_parses_canonical_aliases() {
        assert_eq!(
            ReasoningEffort::parse("minimal"),
            Some(ReasoningEffort::Minimal)
        );
        assert_eq!(
            ReasoningEffort::parse("MIN"),
            Some(ReasoningEffort::Minimal)
        );
        assert_eq!(ReasoningEffort::parse("low"), Some(ReasoningEffort::Low));
        assert_eq!(ReasoningEffort::parse("Mid"), Some(ReasoningEffort::Medium));
        assert_eq!(
            ReasoningEffort::parse("medium"),
            Some(ReasoningEffort::Medium)
        );
        assert_eq!(ReasoningEffort::parse("HIGH"), Some(ReasoningEffort::High));
        assert_eq!(
            ReasoningEffort::parse("xhigh"),
            Some(ReasoningEffort::XHigh)
        );
        assert_eq!(
            ReasoningEffort::parse("extra-high"),
            Some(ReasoningEffort::XHigh)
        );
        assert_eq!(ReasoningEffort::parse("max"), Some(ReasoningEffort::Max));
        assert_eq!(
            ReasoningEffort::parse("MAXIMUM"),
            Some(ReasoningEffort::Max)
        );
        assert_eq!(ReasoningEffort::parse(""), None);
        assert_eq!(ReasoningEffort::parse("bogus"), None);
    }

    #[test]
    fn reasoning_effort_as_str_round_trip() {
        for level in [
            ReasoningEffort::Minimal,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::XHigh,
            ReasoningEffort::Max,
        ] {
            assert_eq!(ReasoningEffort::parse(level.as_str()), Some(level));
        }
    }

    #[test]
    fn reasoning_effort_none_is_explicit_low_bound_not_user_parseable() {
        // `None` is the explicit lowest tier, sending `reasoning_effort: "none"`.
        assert_eq!(ReasoningEffort::None.as_str(), "none");
        // But `"none"` as user/config input keeps the "omit the field" control semantics,
        // and does not map to the `None` tier, which is only used internally by the truncation downgrade ladder.
        assert_eq!(ReasoningEffort::parse("none"), None);
    }

    #[test]
    fn model_def_reasoning_effort_field_optional() {
        // Defaults to None when the field is absent
        let def: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true}"#,
        )
        .unwrap();
        assert!(def.reasoning_effort.is_none());

        // Deserializes correctly with the field present (new name)
        let def2: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"default_reasoning_effort":"high"}"#,
        )
        .unwrap();
        assert_eq!(def2.reasoning_effort, Some(ReasoningEffort::High));

        // xhigh tier deserializes correctly
        let def_xhigh: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"default_reasoning_effort":"xhigh"}"#,
        )
        .unwrap();
        assert_eq!(def_xhigh.reasoning_effort, Some(ReasoningEffort::XHigh));

        // max as the highest tier deserializes correctly
        let def_max: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"default_reasoning_effort":"max"}"#,
        )
        .unwrap();
        assert_eq!(def_max.reasoning_effort, Some(ReasoningEffort::Max));

        // "auto" is equivalent to unset
        let def3: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"default_reasoning_effort":"auto"}"#,
        )
        .unwrap();
        assert!(def3.reasoning_effort.is_none());

        // "off" is equivalent to unset
        let def4: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"default_reasoning_effort":"off"}"#,
        )
        .unwrap();
        assert!(def4.reasoning_effort.is_none());

        // Compatible with the old field name reasoning_effort
        let def5: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"reasoning_effort":"low"}"#,
        )
        .unwrap();
        assert_eq!(def5.reasoning_effort, Some(ReasoningEffort::Low));

        // Invalid values error out
        let bad: Result<ModelDef, _> = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"default_reasoning_effort":"bogus"}"#,
        );
        assert!(bad.is_err());
    }

    #[test]
    fn model_def_max_output_tokens_field_optional() {
        // When the field is absent, defaults to None and the request omits max_tokens (historical behavior).
        let def: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true}"#,
        )
        .unwrap();
        assert!(def.max_output_tokens.is_none());

        // Canonical field name.
        let def2: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"max_output_tokens":32768}"#,
        )
        .unwrap();
        assert_eq!(def2.max_output_tokens, Some(32768));

        // Compatible aliases: max_tokens / max_completion_tokens.
        let def3: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"max_tokens":16000}"#,
        )
        .unwrap();
        assert_eq!(def3.max_output_tokens, Some(16000));

        let def4: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"max_completion_tokens":8192}"#,
        )
        .unwrap();
        assert_eq!(def4.max_output_tokens, Some(8192));
    }
}
