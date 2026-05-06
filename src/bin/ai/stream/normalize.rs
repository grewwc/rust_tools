use crate::ai::{
    provider::ApiProvider,
    request::{StreamChoice, StreamChunk, StreamDelta, StreamFunctionCall, StreamToolCall},
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
    if event_type == "done" || event_type == "[done]" {
        return Some(ParsedStreamPayload::Done);
    }
    if event_type == "response.completed" {
        return Some(ParsedStreamPayload::Ignore);
    }

    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    if let Some(parsed) = parse_function_call_arguments_event(event_type.as_str(), &value) {
        return Some(parsed);
    }
    if let Some(parsed) = parse_output_item_event(event_type.as_str(), &value) {
        return Some(parsed);
    }
    if let Some(parsed) = parse_content_part_event(event_type.as_str(), &value) {
        return Some(parsed);
    }
    if let Some(parsed) = parse_refusal_event(event_type.as_str(), &value) {
        return Some(parsed);
    }
    if event_type.contains("reasoning")
        && (event_type.ends_with(".delta") || event_type.ends_with(".done"))
    {
        let text = extract_event_text(&value, &["delta", "text", "summary_text", "content"]);
        if text.is_empty() {
            return Some(ParsedStreamPayload::Ignore);
        }
        return Some(textual_event_chunk(event_type.as_str(), "", &text));
    }
    if (event_type.contains("output_text") || event_type.contains("content"))
        && (event_type.ends_with(".delta") || event_type.ends_with(".done"))
    {
        let text = extract_event_text(&value, &["delta", "text", "content"]);
        if text.is_empty() {
            return Some(ParsedStreamPayload::Ignore);
        }
        return Some(textual_event_chunk(event_type.as_str(), &text, ""));
    }

    if event_type.ends_with(".done")
        || event_type.ends_with(".added")
        || event_type.ends_with(".part.done")
    {
        return Some(ParsedStreamPayload::Ignore);
    }

    None
}

fn parse_function_call_arguments_event(
    event_type: &str,
    value: &serde_json::Value,
) -> Option<ParsedStreamPayload> {
    if !event_type.contains("function_call_arguments")
        || !(event_type.ends_with(".delta") || event_type.ends_with(".done"))
    {
        return None;
    }

    let mut tool_call = extract_function_call_item(value, extract_output_index(value));
    let arguments = extract_event_text(value, &["delta", "arguments", "text", "content"]);
    if let Some(existing) = tool_call.as_mut() {
        if !arguments.is_empty() {
            existing.function.arguments = arguments;
        }
    } else if !arguments.is_empty() {
        tool_call = Some(StreamToolCall {
            index: extract_output_index(value),
            id: extract_call_identifier(value),
            tool_type: "function".to_string(),
            function: StreamFunctionCall {
                name: extract_function_name(value),
                arguments,
            },
        });
    }

    tool_call.map(|tool_call| tool_call_event_chunk(event_type, tool_call))
}

fn parse_output_item_event(
    event_type: &str,
    value: &serde_json::Value,
) -> Option<ParsedStreamPayload> {
    if !(event_type == "response.output_item.added" || event_type == "response.output_item.done") {
        return None;
    }

    let item = value.get("item").unwrap_or(value);
    let item_type = item
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();

    if item_type != "function_call" && item_type != "function" {
        return Some(ParsedStreamPayload::Ignore);
    }

    let Some(tool_call) = extract_function_call_item(item, extract_output_index(value)) else {
        return Some(ParsedStreamPayload::Ignore);
    };
    Some(tool_call_event_chunk(event_type, tool_call))
}

fn parse_content_part_event(
    event_type: &str,
    value: &serde_json::Value,
) -> Option<ParsedStreamPayload> {
    if !(event_type == "response.content_part.added" || event_type == "response.content_part.done") {
        return None;
    }

    let part = value.get("part").unwrap_or(value);
    let text = extract_event_text(part, &["delta", "text", "content"]);
    if text.is_empty() {
        return Some(ParsedStreamPayload::Ignore);
    }
    Some(textual_event_chunk(event_type, &text, ""))
}

fn parse_refusal_event(event_type: &str, value: &serde_json::Value) -> Option<ParsedStreamPayload> {
    if !event_type.contains("refusal")
        || !(event_type.ends_with(".delta") || event_type.ends_with(".done"))
    {
        return None;
    }

    let text = extract_event_text(value, &["delta", "text", "content", "refusal"]);
    if text.is_empty() {
        return Some(ParsedStreamPayload::Ignore);
    }
    Some(textual_event_chunk(event_type, &text, ""))
}

fn textual_event_chunk(
    event_type: &str,
    content: &str,
    reasoning_content: &str,
) -> ParsedStreamPayload {
    let chunk = stream_chunk_with_delta(StreamDelta {
        content: content.to_string(),
        reasoning_content: reasoning_content.to_string(),
        reasoning_details: String::new(),
        tool_calls: Vec::new(),
    });
    if event_type.ends_with(".done") {
        ParsedStreamPayload::SnapshotChunk(chunk)
    } else {
        ParsedStreamPayload::Chunk(chunk)
    }
}

fn single_delta_chunk(content: &str, reasoning_content: &str) -> StreamChunk {
    stream_chunk_with_delta(StreamDelta {
        content: content.to_string(),
        reasoning_content: reasoning_content.to_string(),
        reasoning_details: String::new(),
        tool_calls: Vec::new(),
    })
}

fn tool_call_event_chunk(event_type: &str, tool_call: StreamToolCall) -> ParsedStreamPayload {
    let chunk = stream_chunk_with_delta(StreamDelta {
        content: String::new(),
        reasoning_content: String::new(),
        reasoning_details: String::new(),
        tool_calls: vec![tool_call],
    });
    if event_type.ends_with(".done") {
        ParsedStreamPayload::SnapshotChunk(chunk)
    } else {
        ParsedStreamPayload::Chunk(chunk)
    }
}

fn stream_chunk_with_delta(delta: StreamDelta) -> StreamChunk {
    StreamChunk {
        choices: vec![StreamChoice {
            delta,
            finish_reason: None,
        }],
        usage: None,
        model: String::new(),
    }
}

fn extract_output_index(value: &serde_json::Value) -> usize {
    value.get("output_index")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize
}

fn extract_function_call_item(
    value: &serde_json::Value,
    fallback_index: usize,
) -> Option<StreamToolCall> {
    let item_type = value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if !item_type.is_empty() && item_type != "function_call" && item_type != "function" {
        return None;
    }

    let name = extract_function_name(value);
    let arguments = extract_stringish_field(value, &["arguments"]);
    let id = extract_call_identifier(value);
    if name.is_empty() && arguments.is_empty() && id.is_empty() {
        return None;
    }

    Some(StreamToolCall {
        index: fallback_index,
        id,
        tool_type: "function".to_string(),
        function: StreamFunctionCall { name, arguments },
    })
}

fn extract_call_identifier(value: &serde_json::Value) -> String {
    for key in ["call_id", "id", "item_id"] {
        let extracted = extract_stringish_field(value, &[key]);
        if !extracted.is_empty() {
            return extracted;
        }
    }
    String::new()
}

fn extract_function_name(value: &serde_json::Value) -> String {
    let direct = extract_stringish_field(value, &["name"]);
    if !direct.is_empty() {
        return direct;
    }
    value.get("function")
        .map(|function| extract_stringish_field(function, &["name"]))
        .unwrap_or_default()
}

fn extract_stringish_field(value: &serde_json::Value, keys: &[&str]) -> String {
    for key in keys {
        let Some(inner) = value.get(*key) else {
            continue;
        };
        let extracted = match inner {
            serde_json::Value::Null => String::new(),
            serde_json::Value::String(text) => text.clone(),
            other => serde_json::to_string(other).unwrap_or_default(),
        };
        if !extracted.is_empty() {
            return extracted;
        }
    }
    String::new()
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

fn parse_stream_chunk(adapter_label: &str, payload: &str) -> ParsedStreamPayload {
    match try_parse_stream_chunk_loose(payload) {
        Some(chunk) => ParsedStreamPayload::Chunk(chunk),
        None => {
            let err = serde_json::from_str::<serde_json::Value>(payload)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_else(|| "unable to parse stream payload".to_string());
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

fn try_parse_stream_chunk_loose(payload: &str) -> Option<StreamChunk> {
    if let Some(chunk) = try_parse_stream_chunk(payload) {
        return Some(chunk);
    }

    let trimmed = payload.trim();
    let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) else {
        return None;
    };
    if start >= end {
        return None;
    }

    let candidate = &trimmed[start..=end];
    try_parse_stream_chunk(candidate)
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
    fn opencode_payload_with_wrapped_json_still_parses() {
        let payload = r#"noise {"choices":[{"delta":{"content":"hello"}}]} trailing"#;
        match parse_stream_payload(StreamProviderAdapterKind::OpenCode, payload, None) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.content, "hello");
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

    #[test]
    fn output_text_done_event_maps_to_snapshot_chunk() {
        let payload = r#"{"text":"hello world"}"#;
        match parse_stream_payload(
            StreamProviderAdapterKind::OpenCode,
            payload,
            Some("response.output_text.done"),
        ) {
            ParsedStreamPayload::SnapshotChunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.content, "hello world");
                assert_eq!(chunk.choices[0].delta.reasoning_content, "");
            }
            _ => panic!("expected snapshot content chunk"),
        }
    }

    #[test]
    fn function_call_arguments_delta_maps_to_tool_call_chunk() {
        let payload = r#"{"output_index":2,"item_id":"fc_item_1","delta":"{\"path\":\"a"}"#;
        match parse_stream_payload(
            StreamProviderAdapterKind::OpenAi,
            payload,
            Some("response.function_call_arguments.delta"),
        ) {
            ParsedStreamPayload::Chunk(chunk) => {
                let tool_call = &chunk.choices[0].delta.tool_calls[0];
                assert_eq!(tool_call.index, 2);
                assert_eq!(tool_call.id, "fc_item_1");
                assert_eq!(tool_call.tool_type, "function");
                assert_eq!(tool_call.function.arguments, "{\"path\":\"a");
            }
            _ => panic!("expected tool-call delta chunk"),
        }
    }

    #[test]
    fn function_call_arguments_done_maps_to_snapshot_tool_call_chunk() {
        let payload = r#"{"output_index":2,"arguments":"{\"path\":\"abc\"}"}"#;
        match parse_stream_payload(
            StreamProviderAdapterKind::OpenAi,
            payload,
            Some("response.function_call_arguments.done"),
        ) {
            ParsedStreamPayload::SnapshotChunk(chunk) => {
                let tool_call = &chunk.choices[0].delta.tool_calls[0];
                assert_eq!(tool_call.index, 2);
                assert_eq!(tool_call.function.arguments, "{\"path\":\"abc\"}");
            }
            _ => panic!("expected tool-call snapshot chunk"),
        }
    }

    #[test]
    fn output_item_added_maps_function_call_metadata() {
        let payload = r#"{"output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"write_file","arguments":""}}"#;
        match parse_stream_payload(
            StreamProviderAdapterKind::OpenAi,
            payload,
            Some("response.output_item.added"),
        ) {
            ParsedStreamPayload::Chunk(chunk) => {
                let tool_call = &chunk.choices[0].delta.tool_calls[0];
                assert_eq!(tool_call.index, 1);
                assert_eq!(tool_call.id, "call_1");
                assert_eq!(tool_call.function.name, "write_file");
                assert_eq!(tool_call.function.arguments, "");
            }
            _ => panic!("expected tool-call metadata chunk"),
        }
    }

    #[test]
    fn output_item_done_maps_final_function_call_snapshot() {
        let payload = r#"{"output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"write_file","arguments":"{\"path\":\"a.rs\"}"}}"#;
        match parse_stream_payload(
            StreamProviderAdapterKind::OpenAi,
            payload,
            Some("response.output_item.done"),
        ) {
            ParsedStreamPayload::SnapshotChunk(chunk) => {
                let tool_call = &chunk.choices[0].delta.tool_calls[0];
                assert_eq!(tool_call.index, 1);
                assert_eq!(tool_call.id, "call_1");
                assert_eq!(tool_call.function.name, "write_file");
                assert_eq!(tool_call.function.arguments, "{\"path\":\"a.rs\"}");
            }
            _ => panic!("expected tool-call final snapshot chunk"),
        }
    }

    #[test]
    fn content_part_added_event_maps_to_content_chunk() {
        let payload = r#"{"part":{"type":"output_text","text":"hello"}}"#;
        match parse_stream_payload(
            StreamProviderAdapterKind::OpenAi,
            payload,
            Some("response.content_part.added"),
        ) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.content, "hello");
            }
            _ => panic!("expected content-part chunk"),
        }
    }

    #[test]
    fn refusal_done_event_maps_to_snapshot_content_chunk() {
        let payload = r#"{"refusal":"cannot comply"}"#;
        match parse_stream_payload(
            StreamProviderAdapterKind::OpenAi,
            payload,
            Some("response.refusal.done"),
        ) {
            ParsedStreamPayload::SnapshotChunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.content, "cannot comply");
            }
            _ => panic!("expected refusal snapshot chunk"),
        }
    }

    #[test]
    fn response_completed_event_is_not_treated_as_immediate_stop() {
        let payload = r#"{"status":"completed"}"#;
        match parse_stream_payload(
            StreamProviderAdapterKind::OpenAi,
            payload,
            Some("response.completed"),
        ) {
            ParsedStreamPayload::Ignore => {}
            _ => panic!("response.completed should not terminate stream before EOF"),
        }
    }
}
