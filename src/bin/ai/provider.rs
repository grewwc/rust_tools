#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub(super) enum ApiProvider {
    #[default]
    Compatible,
    #[serde(alias = "openai")]
    OpenAi,
    #[serde(alias = "opencode")]
    OpenCode,
}

impl ApiProvider {
    pub(super) fn is_openai(self) -> bool {
        matches!(self, Self::OpenAi | Self::OpenCode)
    }
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

/// LLM 推理强度档位，对应 OpenAI / OpenRouter / OpenCode 等兼容协议的
/// `reasoning_effort` 顶层字段。Qwen DashScope (`Compatible` provider) 当前
/// 不支持此字段，所以只对 `is_openai()` 为真的 provider 注入。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    /// 解析 CLI / `/model effort` 命令传入的字符串。仅识别四个档位的字面量，
    /// 大小写不敏感；`off`/`none`/`auto` 等控制语义由调用方自行处理。
    pub(super) fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "minimal" | "min" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" | "mid" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ApiProvider, ModelQualityTier, ReasoningEffort};
    use crate::ai::model_names::ModelDef;

    #[test]
    fn provider_defaults_to_compatible() {
        let def: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","is_vl":false,"search_enabled":true,"tools_default_enabled":true}"#,
        )
        .unwrap();
        assert_eq!(def.provider, ApiProvider::Compatible);
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
    fn parses_openai_provider_and_flagship_tier() {
        let def: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"openai","quality_tier":"flagship","is_vl":true,"search_enabled":false,"tools_default_enabled":true}"#,
        )
        .unwrap();
        assert_eq!(def.provider, ApiProvider::OpenAi);
        assert_eq!(def.quality_tier, ModelQualityTier::Flagship);
    }

    #[test]
    fn parses_opencode_provider_alias() {
        let def: ModelDef = serde_json::from_str(
            r#"{"key":"X","name":"x","provider":"opencode","quality_tier":"basic","is_vl":false,"search_enabled":false,"tools_default_enabled":true}"#,
        )
        .unwrap();
        assert_eq!(def.provider, ApiProvider::OpenCode);
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
        ] {
            assert_eq!(ReasoningEffort::parse(level.as_str()), Some(level));
        }
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
}
