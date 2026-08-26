//! Request protocol dialect.
//!
//! This is an axis orthogonal to the provider adapter: under the same adapter, different
//! models/endpoints may also use different HTTP wires (e.g. chat-completions vs responses).

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RequestProtocolDialect {
    #[default]
    ChatCompletions,
    Responses,
}

impl RequestProtocolDialect {
    pub(crate) fn infer_from_endpoint(endpoint: &str) -> Self {
        if endpoint.trim_end_matches('/').ends_with("/v1/responses") {
            Self::Responses
        } else {
            Self::ChatCompletions
        }
    }
}
