#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub(super) enum ApiProvider {
    #[default]
    Compatible,
    #[serde(alias = "openai")]
    OpenAi,
}

impl ApiProvider {
    pub(super) fn is_openai(self) -> bool {
        matches!(self, Self::OpenAi)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Deserialize, serde::Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub(super) enum ModelQualityTier {
    Basic,
    #[default]
    Standard,
    Strong,
    Flagship,
}

#[cfg(test)]
mod tests {
    use super::{ApiProvider, ModelQualityTier};
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
}
