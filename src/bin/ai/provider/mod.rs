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

/// LLM 推理强度档位。OpenAI / OpenRouter / OpenCode 等协议使用顶层
/// `reasoning_effort`；DashScope compatible provider 使用嵌套
/// `reasoning.effort`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ReasoningEffort {
    /// 显式最低档：下发 `reasoning_effort: "none"`。注意与「不下发字段」
    /// （`override = Some(None)`，服务端默认档位接管）语义不同——`None` 是真正
    /// 的推理下限。新一代 gpt-5.6 用它取代已被移除的 `minimal`。
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

    /// 解析 CLI / `/model effort` 命令传入的字符串。仅识别五个档位的字面量，
    /// 大小写不敏感；`off`/`none`/`auto` 等控制语义由调用方自行处理。
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
        assert_eq!(
            ReasoningEffort::parse("max"),
            Some(ReasoningEffort::Max)
        );
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
        // `None` 是显式最低档，下发 `reasoning_effort: "none"`。
        assert_eq!(ReasoningEffort::None.as_str(), "none");
        // 但 `"none"` 作为用户/config 输入仍保留「省略字段」的控制语义，
        // 不映射到 `None` 档位——该档位仅由截断降档阶梯内部使用。
        assert_eq!(ReasoningEffort::parse("none"), None);
    }

    #[test]
    fn model_def_reasoning_effort_field_optional() {
        // 不带字段时默认为 None
        let def: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true}"#,
        )
        .unwrap();
        assert!(def.reasoning_effort.is_none());

        // 带字段时正确反序列化（新名）
        let def2: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"default_reasoning_effort":"high"}"#,
        )
        .unwrap();
        assert_eq!(def2.reasoning_effort, Some(ReasoningEffort::High));

        // xhigh 档位正确反序列化
        let def_xhigh: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"default_reasoning_effort":"xhigh"}"#,
        )
        .unwrap();
        assert_eq!(def_xhigh.reasoning_effort, Some(ReasoningEffort::XHigh));

        // max 作为最高档位正确反序列化
        let def_max: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"default_reasoning_effort":"max"}"#,
        )
        .unwrap();
        assert_eq!(def_max.reasoning_effort, Some(ReasoningEffort::Max));

        // "auto" 等同于未设置
        let def3: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"default_reasoning_effort":"auto"}"#,
        )
        .unwrap();
        assert!(def3.reasoning_effort.is_none());

        // "off" 等同于未设置
        let def4: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"default_reasoning_effort":"off"}"#,
        )
        .unwrap();
        assert!(def4.reasoning_effort.is_none());

        // 兼容旧的字段名 reasoning_effort
        let def5: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"reasoning_effort":"low"}"#,
        )
        .unwrap();
        assert_eq!(def5.reasoning_effort, Some(ReasoningEffort::Low));

        // 非法值会报错
        let bad: Result<ModelDef, _> = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"default_reasoning_effort":"bogus"}"#,
        );
        assert!(bad.is_err());
    }

    #[test]
    fn model_def_max_output_tokens_field_optional() {
        // 不带字段时默认为 None，请求不下发 max_tokens（沿用历史行为）。
        let def: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true}"#,
        )
        .unwrap();
        assert!(def.max_output_tokens.is_none());

        // 规范字段名。
        let def2: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","is_vl":false,"search_enabled":false,"tools_default_enabled":true,"max_output_tokens":32768}"#,
        )
        .unwrap();
        assert_eq!(def2.max_output_tokens, Some(32768));

        // 兼容别名 max_tokens / max_completion_tokens。
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
