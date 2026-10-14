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
    /// Anthropic Messages API wire (`/v1/messages`): `x-api-key` auth, a
    /// `messages` array with content blocks, and `message_*` / `content_block_*`
    /// SSE events. Used by Anthropic-native models served through gateways that
    /// only expose this protocol (e.g. OpenCode Zen serving claude-opus-5-5).
    Messages,
}

impl RequestProtocolDialect {
    pub(crate) fn infer_from_endpoint(endpoint: &str) -> Self {
        let endpoint = endpoint.trim_end_matches('/');
        if endpoint.ends_with("/v1/responses") {
            Self::Responses
        } else if endpoint.ends_with("/v1/messages") {
            Self::Messages
        } else {
            Self::ChatCompletions
        }
    }
}
