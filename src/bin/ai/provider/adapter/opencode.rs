//! OpenCode Zen 网关适配器。
//!
//! 端点：`https://opencode.ai/zen/v1/chat/completions`。首 token 较慢，需打印
//! 等待提示；流式 payload 解析更宽松。思考开关（DeepSeek 的 `thinking` 对象 /
//! MiniMax always-on）由 `thinking` 方言模块统一处理，不在此实现。

use crate::ai::config_schema::AiConfig;
use crate::ai::stream::{ParsedStreamPayload, try_parse_stream_chunk_loose};

use super::{OPENCODE_DEFAULT_ENDPOINT, ProviderAdapter};

pub(super) struct OpenCodeAdapter;

impl ProviderAdapter for OpenCodeAdapter {
    fn label(&self) -> &'static str {
        "opencode"
    }

    fn default_endpoint(&self) -> &'static str {
        OPENCODE_DEFAULT_ENDPOINT
    }

    fn api_key_candidates(&self) -> &'static [&'static str] {
        &[AiConfig::MODEL_OPENCODE_API_KEY, AiConfig::MODEL_API_KEY]
    }

    fn shows_waiting_hint(&self) -> bool {
        true
    }

    fn parse_provider_chunk(&self, payload: &str) -> ParsedStreamPayload {
        let trimmed = payload.trim();
        if trimmed.is_empty() || trimmed == "[DONE]" {
            return ParsedStreamPayload::Ignore;
        }
        match try_parse_stream_chunk_loose(trimmed) {
            Some(chunk) => ParsedStreamPayload::Chunk(chunk),
            None => {
                eprintln!(
                    "[opencode] ignored payload, length: {}, starts_with: {:.30}",
                    trimmed.len(),
                    trimmed
                );
                ParsedStreamPayload::Ignore
            }
        }
    }
}
