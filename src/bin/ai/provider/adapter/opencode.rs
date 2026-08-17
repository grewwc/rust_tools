//! OpenCode Zen 网关适配器。
//!
//! 端点：`https://opencode.ai/zen/v1/chat/completions`。首 token 较慢，需打印
//! 等待提示；流式 payload 解析更宽松。思考开关（DeepSeek 的 `thinking` 对象 /
//! MiniMax always-on）由 `thinking` 方言模块统一处理，不在此实现。

use crate::ai::config_schema::AiConfig;
use crate::ai::request::{ParsedStreamPayload, try_parse_stream_chunk_loose};

use super::{OPENCODE_DEFAULT_ENDPOINT, ProviderAdapter};

pub(super) struct OpenCodeAdapter;

fn collect_api_keys_from_config(
    cfg: &crate::commonw::configw::ConfigW,
    primary_key: &str,
) -> Vec<String> {
    let mut provider_keys = Vec::new();
    for (key, _) in cfg.entries() {
        if key.starts_with("opencode.api_key")
            && key != "opencode.api_key"
            && let Some(value) = cfg
                .get_opt(key)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
            && value != primary_key
            && !provider_keys.contains(&value)
        {
            provider_keys.push(value);
        }
    }

    // `opencode.api_key` 未配置时，通用 `api_key` 只是兜底；专属轮换 key 应优先。
    if cfg.get_opt(AiConfig::MODEL_API_KEY).as_deref() == Some(primary_key)
        && !provider_keys.is_empty()
    {
        provider_keys.push(primary_key.to_string());
        provider_keys
    } else {
        let mut keys = vec![primary_key.to_string()];
        keys.extend(provider_keys);
        keys
    }
}

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

    fn collect_api_keys(&self, primary_key: &str) -> Vec<String> {
        collect_api_keys_from_config(&crate::commonw::configw::get_all_config(), primary_key)
    }

    fn keys_exhausted_message(&self) -> &'static str {
        "all opencode keys exhausted"
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
                crate::ai::request::emit_request_diagnostic(format_args!(
                    "[opencode] ignored payload, length: {}, starts_with: {:.30}",
                    trimmed.len(),
                    trimmed
                ));
                ParsedStreamPayload::Ignore
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commented_keys_are_ignored_and_active_named_key_is_normalized() {
        let cfg = crate::commonw::configw::ConfigW::parse(
            "opencode.api_key_active = \"active-key\" # account\n\
             api_key = \"global-fallback\"\n\
             # opencode.api_key = \"disabled-key\" # disabled\n\
             # opencode.api_key_old = \"old-key\" # disabled",
        );

        assert_eq!(
            collect_api_keys_from_config(&cfg, "global-fallback"),
            vec!["active-key", "global-fallback"]
        );
    }
}
