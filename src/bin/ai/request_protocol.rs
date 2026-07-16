//! 请求协议类型。
//!
//! 这是与 provider adapter 正交的另一根轴：同一 adapter 下，不同模型/endpoint
//! 也可能走不同的 HTTP wire（例如 chat-completions 与 responses）。

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
