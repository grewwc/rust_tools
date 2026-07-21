use crate::ai::{
    provider::ProviderAdapter,
    request::{StreamChoice, StreamChunk, StreamDelta, StreamFunctionCall, StreamToolCall},
};

use super::state::ParsedStreamPayload;

pub(super) fn parse_stream_payload(
    adapter: &'static dyn ProviderAdapter,
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

    // 在调用 adapter 解析之前，先检测 provider 在流中途返回的 error 对象。
    // StreamChunk 所有字段都是 #[serde(default)]，{"error":{...}} 会被静默反序列化
    // 为空 chunk 然后丢弃，导致用户看到空响应且无任何错误提示。
    if let Some(err_msg) = extract_provider_error(payload) {
        return ParsedStreamPayload::Error(err_msg);
    }

    if let Some(event_type) = event_type {
        if let Some(parsed) = parse_sse_event_payload(event_type, payload) {
            return parsed;
        }
    }

    adapter.parse_provider_chunk(payload)
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
        // Responses API 的最终用量嵌在 response.usage，而不是兼容流的顶层
        // usage。将其包装成普通 chunk，复用既有的用量落账路径；仍不能把该
        // 事件视为 [DONE]，因为连接关闭才是流结束信号。
        let value: serde_json::Value = serde_json::from_str(payload).ok()?;
        let usage = value.get("response")?.get("usage")?.clone();
        let usage = serde_json::from_value(usage).ok()?;
        return Some(ParsedStreamPayload::Chunk(StreamChunk {
            usage: Some(usage),
            ..Default::default()
        }));
    }
    // OpenAI Responses API 错误/不完整事件——必须显式处理，否则会 fallthrough
    // 到 parse_provider_chunk 被当成空 chunk 静默丢弃。
    if event_type == "response.failed" {
        let value: serde_json::Value = serde_json::from_str(payload).ok()?;
        let msg = value
            .get("response")
            .and_then(|r| r.get("error"))
            .and_then(extract_error_message)
            .unwrap_or_else(|| "response failed (no error detail)".to_string());
        return Some(ParsedStreamPayload::Error(msg));
    }
    if event_type == "response.incomplete" {
        let value: serde_json::Value = serde_json::from_str(payload).ok()?;
        let reason = value
            .get("response")
            .and_then(|r| r.get("incomplete_details"))
            .and_then(|d| d.get("reason"))
            .and_then(|r| r.as_str())
            .unwrap_or("unknown");
        return Some(ParsedStreamPayload::Error(format!(
            "response incomplete: {reason}"
        )));
    }
    // 部分 provider 用 SSE event: error 携带错误对象
    if event_type == "error" {
        let value: serde_json::Value = serde_json::from_str(payload).ok()?;
        let msg = extract_error_message(&value)
            .unwrap_or_else(|| "stream error event (no detail)".to_string());
        return Some(ParsedStreamPayload::Error(msg));
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
        let text = extract_event_text(
            &value,
            &[
                "delta",
                "text",
                "summary_text",
                "content",
                "summary",
                "reasoning",
            ],
        );
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

    // 完整的 reasoning output item 只在 `.done` 落地（`.added` 时 encrypted_content
    // 尚未填充）。只捕获真正带 `encrypted_content` 的 item：没有加密载荷就无法
    // 忠实回放，交回上层退化为不回传，避免送半截 item 触发 400。
    if item_type == "reasoning" {
        if event_type == "response.output_item.done"
            && item
                .get("encrypted_content")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|content| !content.is_empty())
        {
            return Some(ParsedStreamPayload::ReasoningItem(item.clone()));
        }
        return Some(ParsedStreamPayload::Ignore);
    }

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
    if !(event_type == "response.content_part.added" || event_type == "response.content_part.done")
    {
        return None;
    }

    let part = value.get("part").unwrap_or(value);
    let part_type = part
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let text = extract_event_text(part, &["delta", "text", "content"]);
    if text.is_empty() {
        return Some(ParsedStreamPayload::Ignore);
    }
    if part_type == "summary_text" {
        // summary_text 的 content_part 事件（added/done）都是已流式输出过的推理
        // 摘要重发，而非模型增量。统一按 SnapshotChunk 处理，走未见后缀去重，
        // 避免 added 事件携带的完整文本按 Append 模式用原文重复累积 reasoning_text，
        // 污染退化检测并可能诱发 thinking 重复渲染（gpt-5.5/5.6 多发此路径）。
        return Some(ParsedStreamPayload::SnapshotChunk(single_delta_chunk(
            "", &text,
        )));
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
            message: StreamDelta::default(),
            reasoning_content: String::new(),
            reasoning_details: String::new(),
            finish_reason: None,
        }],
        usage: None,
        model: String::new(),
    }
}

fn extract_output_index(value: &serde_json::Value) -> usize {
    // 优先使用 provider 显式提供的 output_index
    if let Some(idx) = value
        .get("output_index")
        .and_then(serde_json::Value::as_u64)
    {
        return idx as usize;
    }
    // output_index 缺失时，使用 call_id/item_id 的哈希作为合成索引，
    // 避免多个并行工具调用全部碰撞到 index 0 互相覆盖。
    // 哈希值映射到 [10000, usize::MAX) 区间，不与真实 output_index（通常 0-9）冲突。
    let id = extract_call_identifier(value);
    if !id.is_empty() {
        let mut hash = 10000u64;
        for byte in id.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        return hash as usize;
    }
    0
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
    value
        .get("function")
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

/// 检测 payload JSON 顶层的 `error` 字段，提取可读错误信息。
///
/// StreamChunk 所有字段都是 `#[serde(default)]` 且无 `deny_unknown_fields`，
/// 所以 `{"error":{...}}` 会被静默反序列化为空 chunk。此函数在解析前拦截
/// 这类 provider 错误对象，返回可读错误信息。
fn extract_provider_error(payload: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    let error = value.get("error")?;
    extract_error_message(error)
}

/// 从一个 JSON value（通常是 `error` 字段的值）提取可读错误信息。
///
/// 支持的格式：
/// - `{"message": "..."}` / `{"message": "...", "type": "..."}` / `{"code": "...", "message": "..."}`
/// - `"string message"`
/// - 其他对象：回退到 JSON 序列化
fn extract_error_message(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => {
            if s.is_empty() {
                None
            } else {
                Some(s.clone())
            }
        }
        serde_json::Value::Object(obj) => {
            let msg = obj.get("message").and_then(|v| v.as_str());
            let typ = obj
                .get("type")
                .and_then(|v| v.as_str())
                .or_else(|| obj.get("code").and_then(|v| v.as_str()));
            match (msg, typ) {
                (Some(m), Some(t)) => Some(format!("{t}: {m}")),
                (Some(m), None) => Some(m.to_string()),
                (None, Some(t)) => Some(t.to_string()),
                (None, None) => {
                    let s = value.to_string();
                    if s == "{}" { None } else { Some(s) }
                }
            }
        }
        _ => None,
    }
}

fn try_parse_stream_chunk(payload: &str) -> Option<StreamChunk> {
    let mut chunk = serde_json::from_str::<StreamChunk>(payload).ok()?;
    chunk.merge_reasoning();
    Some(chunk)
}

pub(in crate::ai) fn try_parse_stream_chunk_loose(payload: &str) -> Option<StreamChunk> {
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
    use super::parse_stream_payload;
    use crate::ai::{provider, stream::state::ParsedStreamPayload};

    #[test]
    fn parse_stream_payload_accepts_plain_json_payload() {
        let payload = r#"{"choices":[{"delta":{"content":"hello"}}]}"#;
        match parse_stream_payload(provider::openai_adapter(), payload, None) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.content, "hello");
            }
            _ => panic!("expected parsed chunk"),
        }
    }

    #[test]
    fn openrouter_endpoint_uses_openrouter_adapter() {
        let adapter = provider::adapter_for(
            crate::ai::provider::ApiProvider::OpenAi,
            "https://openrouter.ai/api/v1/chat/completions",
        );
        assert_eq!(adapter.label(), "openrouter");
    }

    #[test]
    fn alibaba_provider_uses_alibaba_adapter() {
        let adapter = provider::adapter_for(
            crate::ai::provider::ApiProvider::Alibaba,
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        );
        assert_eq!(adapter.label(), "alibaba");
    }

    #[test]
    fn opencode_provider_uses_opencode_adapter() {
        let adapter = provider::adapter_for(
            crate::ai::provider::ApiProvider::OpenCode,
            "https://opencode.ai/zen/v1/chat/completions",
        );
        assert_eq!(adapter.label(), "opencode");
    }

    #[test]
    fn opencode_payload_accepts_structured_content_chunks() {
        let payload = r#"{"id":"chatcmpl-1","choices":[{"delta":{"content":[{"type":"output_text","text":"hi"}]}}]}"#;
        match parse_stream_payload(provider::opencode_adapter(), payload, None) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.content, "hi");
            }
            _ => panic!("expected parsed chunk"),
        }
    }

    #[test]
    fn structured_content_summary_text_stays_in_reasoning_channel() {
        let payload = r#"{"choices":[{"delta":{"content":[{"type":"summary_text","text":"先检查测试配置。"},{"type":"output_text","text":"结论：这是陈旧测试。"}]}}]}"#;
        match parse_stream_payload(provider::openai_adapter(), payload, None) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.reasoning_content, "先检查测试配置。");
                assert_eq!(chunk.choices[0].delta.content, "结论：这是陈旧测试。");
            }
            _ => panic!("expected parsed chunk"),
        }
    }

    #[test]
    fn opencode_payload_accepts_message_snapshot_reasoning() {
        let payload =
            r#"{"choices":[{"message":{"reasoning_content":"step","content":"answer"}}]}"#;
        match parse_stream_payload(provider::opencode_adapter(), payload, None) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.reasoning_content, "step");
                assert_eq!(chunk.choices[0].delta.content, "answer");
            }
            _ => panic!("expected parsed chunk"),
        }
    }

    #[test]
    fn opencode_payload_with_wrapped_json_still_parses() {
        let payload = r#"noise {"choices":[{"delta":{"content":"hello"}}]} trailing"#;
        match parse_stream_payload(provider::opencode_adapter(), payload, None) {
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
            provider::openai_adapter(),
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
    fn reasoning_event_with_summary_array_maps_to_reasoning_chunk() {
        let payload = r#"{"summary":[{"text":"step 1"},{"text":" step 2"}]}"#;
        match parse_stream_payload(
            provider::openai_adapter(),
            payload,
            Some("response.reasoning_summary_text.delta"),
        ) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.reasoning_content, "step 1 step 2");
                assert_eq!(chunk.choices[0].delta.content, "");
            }
            _ => panic!("expected reasoning chunk"),
        }
    }

    #[test]
    fn output_text_event_delta_maps_to_content_chunk() {
        let payload = r#"{"delta":"hello"}"#;
        match parse_stream_payload(
            provider::openai_adapter(),
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
            provider::opencode_adapter(),
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
            provider::openai_adapter(),
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
            provider::openai_adapter(),
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
            provider::openai_adapter(),
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
            provider::openai_adapter(),
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
            provider::openai_adapter(),
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
    fn content_part_summary_text_maps_to_reasoning_snapshot_chunk() {
        // summary_text 的 content_part 事件（added/done）是对已流式输出的推理
        // 摘要重发，统一按 SnapshotChunk 处理以走未见后缀去重，避免重复累积
        // reasoning_text（gpt-5.5/5.6 多发此路径）。
        let payload = r#"{"part":{"type":"summary_text","text":"step summary"}}"#;
        match parse_stream_payload(
            provider::openai_adapter(),
            payload,
            Some("response.content_part.added"),
        ) {
            ParsedStreamPayload::SnapshotChunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.content, "");
                assert_eq!(chunk.choices[0].delta.reasoning_content, "step summary");
            }
            _ => panic!("expected reasoning content-part snapshot chunk"),
        }
    }

    #[test]
    fn refusal_done_event_maps_to_snapshot_content_chunk() {
        let payload = r#"{"refusal":"cannot comply"}"#;
        match parse_stream_payload(
            provider::openai_adapter(),
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
    fn response_completed_event_preserves_responses_api_usage() {
        let payload = r#"{
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 128,
                    "output_tokens": 64,
                    "total_tokens": 192,
                    "input_tokens_details": {"cached_tokens": 32},
                    "output_tokens_details": {"reasoning_tokens": 48}
                }
            }
        }"#;
        match parse_stream_payload(
            provider::openai_adapter(),
            payload,
            Some("response.completed"),
        ) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert!(chunk.choices.is_empty());
                let usage = chunk
                    .usage
                    .expect("response.completed should contain usage");
                assert_eq!(usage.prompt_tokens, 128);
                assert_eq!(usage.completion_tokens, 64);
                assert_eq!(usage.total_tokens, 192);
                assert_eq!(
                    usage
                        .prompt_tokens_details
                        .expect("cached details")
                        .cached_tokens,
                    32
                );
                assert_eq!(
                    usage
                        .completion_tokens_details
                        .expect("reasoning details")
                        .reasoning_tokens,
                    48
                );
            }
            _ => panic!("response.completed should yield a usage chunk"),
        }
    }

    #[test]
    fn error_object_in_payload_is_not_silently_swallowed() {
        // provider 在流中途返回 {"error":{"message":"rate limited","type":"server_error"}}
        // 此前 StreamChunk 的 #[serde(default)] 会把它反序列化为空 chunk 静默丢弃。
        let payload = r#"{"error":{"message":"rate limited","type":"server_error"}}"#;
        match parse_stream_payload(provider::openai_adapter(), payload, None) {
            ParsedStreamPayload::Error(msg) => {
                assert!(msg.contains("rate limited"), "msg was: {msg}");
                assert!(msg.contains("server_error"), "msg was: {msg}");
            }
            _ => panic!("expected Error for provider error object, got something else"),
        }
    }

    #[test]
    fn error_object_with_string_value_is_extracted() {
        let payload = r#"{"error":"internal server error"}"#;
        match parse_stream_payload(provider::openai_adapter(), payload, None) {
            ParsedStreamPayload::Error(msg) => {
                assert_eq!(msg, "internal server error");
            }
            _ => panic!("expected Error for string error"),
        }
    }

    #[test]
    fn error_object_with_code_and_message_is_extracted() {
        let payload = r#"{"error":{"code":"429","message":"Too Many Requests"}}"#;
        match parse_stream_payload(provider::alibaba_adapter(), payload, None) {
            ParsedStreamPayload::Error(msg) => {
                assert!(msg.contains("429"), "msg was: {msg}");
                assert!(msg.contains("Too Many Requests"), "msg was: {msg}");
            }
            _ => panic!("expected Error for code+message error"),
        }
    }

    #[test]
    fn normal_chunk_without_error_field_still_parses() {
        // 确保正常 chunk 不被 extract_provider_error 误判
        let payload = r#"{"choices":[{"delta":{"content":"hello"}}]}"#;
        match parse_stream_payload(provider::openai_adapter(), payload, None) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.content, "hello");
            }
            _ => panic!("normal chunk should parse as Chunk, not Error"),
        }
    }

    #[test]
    fn usage_only_chunk_without_error_field_still_ignored() {
        // OpenAI 尾包：choices 为空但带 usage，不应被误判为 error
        let payload = r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        match parse_stream_payload(provider::openai_adapter(), payload, None) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert!(chunk.choices.is_empty());
                assert!(chunk.usage.is_some());
            }
            _ => panic!("usage-only chunk should parse as Chunk, not Error"),
        }
    }

    #[test]
    fn response_failed_event_surfaces_error() {
        let payload = r#"{"type":"response.failed","response":{"error":{"code":"server_error","message":"model overloaded"}}}"#;
        match parse_stream_payload(provider::openai_adapter(), payload, Some("response.failed")) {
            ParsedStreamPayload::Error(msg) => {
                assert!(msg.contains("model overloaded"), "msg was: {msg}");
            }
            _ => panic!("response.failed should surface as Error"),
        }
    }

    #[test]
    fn response_incomplete_event_surfaces_reason() {
        let payload = r#"{"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}}}"#;
        match parse_stream_payload(
            provider::openai_adapter(),
            payload,
            Some("response.incomplete"),
        ) {
            ParsedStreamPayload::Error(msg) => {
                assert!(msg.contains("max_output_tokens"), "msg was: {msg}");
            }
            _ => panic!("response.incomplete should surface as Error"),
        }
    }
}
