use crate::ai::{
    provider::ApiProvider,
    request::{StreamChoice, StreamChunk, StreamDelta},
};

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
    event_type: Option<&str>,
) -> ParsedStreamPayload {
    let payload = payload.trim();
    if payload.is_empty() {
        return ParsedStreamPayload::Ignore;
    }
    if payload == "[DONE]" {
        return ParsedStreamPayload::Done;
    }
    if let Some(event_type) = event_type {
        if let Some(parsed) = parse_sse_event_payload(event_type, payload) {
            return parsed;
        }
    }

    match adapter_kind {
        StreamProviderAdapterKind::Compatible => parse_compatible_payload(payload),
        StreamProviderAdapterKind::OpenAi => parse_openai_payload(payload),
        StreamProviderAdapterKind::OpenRouter => parse_openrouter_payload(payload),
        StreamProviderAdapterKind::OpenCode => parse_opencode_payload(payload),
    }
}

fn parse_sse_event_payload(event_type: &str, payload: &str) -> Option<ParsedStreamPayload> {
    let event_type = event_type.trim().to_ascii_lowercase();
    if event_type.is_empty() {
        return None;
    }
    if event_type == "done" || event_type == "[done]" || event_type == "response.completed" {
        return Some(ParsedStreamPayload::Done);
    }
    if event_type.ends_with(".done") || event_type.ends_with(".added") || event_type.ends_with(".part.done") {
        return Some(ParsedStreamPayload::Ignore);
    }

    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    if event_type.contains("reasoning") && event_type.ends_with(".delta") {
        let text = extract_event_text(&value, &["delta", "text", "summary_text", "content"]);
        if text.is_empty() {
            return Some(ParsedStreamPayload::Ignore);
        }
        return Some(ParsedStreamPayload::Chunk(single_delta_chunk("", &text)));
    }
    if (event_type.contains("output_text") || event_type.contains("content"))
        && event_type.ends_with(".delta")
    {
        let text = extract_event_text(&value, &["delta", "text", "content"]);
        if text.is_empty() {
            return Some(ParsedStreamPayload::Ignore);
        }
        return Some(ParsedStreamPayload::Chunk(single_delta_chunk(&text, "")));
    }

    None
}

fn single_delta_chunk(content: &str, reasoning_content: &str) -> StreamChunk {
    StreamChunk {
        choices: vec![StreamChoice {
            delta: StreamDelta {
                content: content.to_string(),
                reasoning_content: reasoning_content.to_string(),
                reasoning_details: String::new(),
                tool_calls: Vec::new(),
            },
            finish_reason: None,
        }],
    }
}

fn extract_event_text(value: &serde_json::Value, preferred_keys: &[&str]) -> String {
    match value {
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
            String::new()
        }
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(items) => items
            .iter()
            .map(|item| extract_event_text(item, preferred_keys))
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(""),
        serde_json::Value::Object(map) => {
            for key in preferred_keys {
                if let Some(inner) = map.get(*key) {
                    let extracted = extract_event_text(inner, preferred_keys);
                    if !extracted.is_empty() {
                        return extracted;
                    }
                }
            }
            String::new()
        }
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
        Ok(mut chunk) => {
            chunk.merge_reasoning();
            ParsedStreamPayload::Chunk(chunk)
        }
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
        Ok(mut chunk) => {
            chunk.merge_reasoning();
            ParsedStreamPayload::Chunk(chunk)
        }
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
    let mut chunk = serde_json::from_str::<StreamChunk>(payload).ok()?;
    chunk.merge_reasoning();
    Some(chunk)
}

#[cfg(test)]
mod tests {
    use super::{StreamProviderAdapterKind, parse_stream_payload, resolve_adapter_kind};
    use crate::ai::{provider::ApiProvider, stream::state::ParsedStreamPayload};

    #[test]
    fn parse_stream_payload_accepts_plain_json_payload() {
        let payload = r#"{"choices":[{"delta":{"content":"hello"}}]}"#;
        match parse_stream_payload(StreamProviderAdapterKind::OpenAi, payload, None) {
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
        match parse_stream_payload(StreamProviderAdapterKind::OpenCode, payload, None) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.content, "hi");
            }
            _ => panic!("expected parsed chunk"),
        }
    }

    #[test]
    fn reasoning_event_delta_maps_to_reasoning_chunk() {
        let payload = r#"{"delta":"step one"}"#;
        match parse_stream_payload(
            StreamProviderAdapterKind::OpenAi,
            payload,
            Some("response.reasoning_text.delta"),
        ) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.reasoning_content, "step one");
                assert_eq!(chunk.choices[0].delta.content, "");
            }
            _ => panic!("expected reasoning chunk"),
        }
    }

    #[test]
    fn output_text_event_delta_maps_to_content_chunk() {
        let payload = r#"{"delta":"hello"}"#;
        match parse_stream_payload(
            StreamProviderAdapterKind::OpenAi,
            payload,
            Some("response.output_text.delta"),
        ) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.content, "hello");
                assert_eq!(chunk.choices[0].delta.reasoning_content, "");
            }
            _ => panic!("expected content chunk"),
        }
    }
}
