use crate::ai::{provider::ApiProvider, request::StreamChunk};

use super::state::ParsedStreamPayload;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StreamProviderAdapterKind {
    Compatible,
    OpenAi,
    OpenRouter,
    OpenCode,
}

pub(super) fn resolve_adapter_kind(
    provider: ApiProvider,
    endpoint: &str,
) -> StreamProviderAdapterKind {
    let endpoint = endpoint.trim().to_ascii_lowercase();
    if endpoint.contains("openrouter.ai") {
        return StreamProviderAdapterKind::OpenRouter;
    }

    match provider {
        ApiProvider::OpenAi => StreamProviderAdapterKind::OpenAi,
        ApiProvider::OpenCode => StreamProviderAdapterKind::OpenCode,
        ApiProvider::Compatible => StreamProviderAdapterKind::Compatible,
    }
}

pub(super) fn parse_stream_payload(
    adapter_kind: StreamProviderAdapterKind,
    payload: &str,
) -> ParsedStreamPayload {
    let payload = payload.trim();
    if payload.is_empty() {
        return ParsedStreamPayload::Ignore;
    }
    if payload == "[DONE]" {
        return ParsedStreamPayload::Done;
    }

    match adapter_kind {
        StreamProviderAdapterKind::Compatible => parse_compatible_payload(payload),
        StreamProviderAdapterKind::OpenAi => parse_openai_payload(payload),
        StreamProviderAdapterKind::OpenRouter => parse_openrouter_payload(payload),
        StreamProviderAdapterKind::OpenCode => parse_opencode_payload(payload),
    }
}

fn parse_compatible_payload(payload: &str) -> ParsedStreamPayload {
    parse_stream_chunk("compatible", payload)
}

fn parse_openai_payload(payload: &str) -> ParsedStreamPayload {
    parse_stream_chunk("openai", payload)
}

fn parse_openrouter_payload(payload: &str) -> ParsedStreamPayload {
    parse_stream_chunk("openrouter", payload)
}

fn parse_opencode_payload(payload: &str) -> ParsedStreamPayload {
    let trimmed = payload.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return ParsedStreamPayload::Ignore;
    }

    match serde_json::from_str::<StreamChunk>(trimmed) {
        Ok(chunk) => ParsedStreamPayload::Chunk(chunk),
        Err(_) => {
            eprintln!(
                "[opencode] ignored payload, length: {}, starts_with: {:.30}",
                trimmed.len(),
                trimmed
            );
            ParsedStreamPayload::Ignore
        }
    }
}

fn parse_stream_chunk(adapter_label: &str, payload: &str) -> ParsedStreamPayload {
    match serde_json::from_str::<StreamChunk>(payload) {
        Ok(chunk) => ParsedStreamPayload::Chunk(chunk),
        Err(err) => {
            eprintln!("handleResponse error [{adapter_label}] {err}");
            eprintln!("======> response: ");
            eprintln!("{payload}");
            eprintln!("<======");
            ParsedStreamPayload::Ignore
        }
    }
}

fn try_parse_stream_chunk(payload: &str) -> Option<StreamChunk> {
    serde_json::from_str::<StreamChunk>(payload)
        .ok()
        // 静默解析，解析失败时不打印错误
}

#[cfg(test)]
mod tests {
    use super::{StreamProviderAdapterKind, parse_stream_payload, resolve_adapter_kind};
    use crate::ai::{provider::ApiProvider, stream::state::ParsedStreamPayload};

    #[test]
    fn parse_stream_payload_accepts_plain_json_payload() {
        let payload = r#"{"choices":[{"delta":{"content":"hello"}}]}"#;
        match parse_stream_payload(StreamProviderAdapterKind::OpenAi, payload) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.content, "hello");
            }
            _ => panic!("expected parsed chunk"),
        }
    }

    #[test]
    fn openrouter_endpoint_uses_openrouter_adapter() {
        let adapter = resolve_adapter_kind(
            ApiProvider::OpenAi,
            "https://openrouter.ai/api/v1/chat/completions",
        );
        assert_eq!(adapter, StreamProviderAdapterKind::OpenRouter);
    }

    #[test]
    fn opencode_provider_uses_opencode_adapter() {
        let adapter = resolve_adapter_kind(
            ApiProvider::OpenCode,
            "https://opencode.ai/zen/v1/chat/completions",
        );
        assert_eq!(adapter, StreamProviderAdapterKind::OpenCode);
    }

    #[test]
    fn opencode_payload_accepts_structured_content_chunks() {
        let payload = r#"{"id":"chatcmpl-1","choices":[{"delta":{"content":[{"type":"output_text","text":"hi"}]}}]}"#;
        match parse_stream_payload(StreamProviderAdapterKind::OpenCode, payload) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.content, "hi");
            }
            _ => panic!("expected parsed chunk"),
        }
    }
}
