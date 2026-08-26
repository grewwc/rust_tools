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
    if let Some(event_type) = event_type {
        let normalized_event_type = event_type.trim();
        if normalized_event_type.eq_ignore_ascii_case("done")
            || normalized_event_type.eq_ignore_ascii_case("[done]")
        {
            return ParsedStreamPayload::Done;
        }
        if (normalized_event_type.eq_ignore_ascii_case("error")
            || normalized_event_type.eq_ignore_ascii_case("response.failed")
            || normalized_event_type.eq_ignore_ascii_case("response.incomplete"))
            && let Some(parsed) = parse_sse_event_payload(event_type, payload)
        {
            return parsed;
        }
    }

    // 部分网关（opencode zen / enc 加密通道）缺少准确的 SSE `event:` 名，事件类型仅
    // 内嵌在 JSON 顶层 `type` 字段里。`response.output_item.done` 携带的
    // encrypted reasoning 会被 event 名分支忽略或被 adapter 宽松 chunk 解析静默吞掉，
    // 导致无法在下一轮 tool 请求中回放；未来网关若把完整载荷移到 `.added` 也需覆盖。
    // 这里先按 JSON type 补一次 output_item 解析，仅 reasoning 捕获会命中，其余 item
    // 类型维持既有路径；用 contains 预筛避免对普通 delta chunk 多做 JSON 解析。
    if payload.contains("response.output_item.done")
        || payload.contains("response.output_item.added")
    {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) {
            if let Some(err_msg) = value.get("error").and_then(extract_error_message) {
                return ParsedStreamPayload::Error(err_msg);
            }
            if let Some(event_type) = value.get("type").and_then(serde_json::Value::as_str) {
                if event_type == "response.output_item.done"
                    || event_type == "response.output_item.added"
                {
                    if let Some(parsed) = parse_output_item_event(event_type, &value) {
                        match &parsed {
                            ParsedStreamPayload::ReasoningItem(_) => return parsed,
                            ParsedStreamPayload::Ignore => {
                                // 仅 reasoning 的 stub 需要在此截获为 Ignore；其余类型（如 message）
                                // 必须落回 adapter 宽松解析，避免把 message done 吞成 Ignore。
                                let is_reasoning = value
                                    .get("item")
                                    .and_then(|v| v.get("type"))
                                    .and_then(serde_json::Value::as_str)
                                    .is_some_and(|t| t.eq_ignore_ascii_case("reasoning"));
                                if is_reasoning {
                                    return parsed;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    if let Some(event_type) = event_type {
        if let Some(parsed) = parse_sse_event_payload(event_type, payload) {
            return parsed;
        }
    }

    // 非 SSE 事件路径：在调用 adapter 解析之前，先检测 provider 在流中途返回的
    // error 对象。StreamChunk 所有字段都是 #[serde(default)]，{"error":{...}}
    // 会被静默反序列化为空 chunk 然后丢弃，导致用户看到空响应且无任何错误提示。
    // SSE 路径的 error 检测已并入 parse_sse_event_payload 的同一次 JSON 解析，
    // 避免每个 chunk 双重解析。
    if let Some(err_msg) = extract_provider_error(payload) {
        return ParsedStreamPayload::Error(err_msg);
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
    // 统一解析一次并复用：各事件分支共享同一 Value，避免每个 chunk 双重 JSON
    // 解析；顶层 error 对象检测也在此完成（StreamChunk 全字段 #[serde(default)]，
    // 纯 error payload 会被静默反序列化为空 chunk 丢弃）。
    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    if let Some(err_msg) = value.get("error").and_then(extract_error_message) {
        return Some(ParsedStreamPayload::Error(err_msg));
    }
    if event_type == "response.completed" {
        // Responses API 的最终用量嵌在 response.usage，而不是兼容流的顶层
        // usage。将其包装成普通 chunk，复用既有的用量落账路径；仍不能把该
        // 事件视为 [DONE]，因为连接关闭才是流结束信号。
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
        let msg = value
            .get("response")
            .and_then(|r| r.get("error"))
            .and_then(extract_error_message)
            .unwrap_or_else(|| "response failed (no error detail)".to_string());
        return Some(ParsedStreamPayload::Error(msg));
    }
    if event_type == "response.incomplete" {
        let reason = value
            .get("response")
            .and_then(|r| r.get("incomplete_details"))
            .and_then(|d| d.get("reason"))
            .and_then(|r| r.as_str())
            .unwrap_or("unknown");
        // Mirrors @ai-sdk/openai's mapOpenAIResponseFinishReason: max_output_tokens
        // truncation maps to finish_reason=length, keeping the partial text
        // produced so far; usage is embedded in response.usage just like
        // response.completed. Reuses the existing length-truncation decision:
        // visible text finishes as a normal completion; only with no visible
        // output at all does it escalate to a retryable Truncated.
        if reason.eq_ignore_ascii_case("max_output_tokens") {
            let mut chunk = stream_chunk_with_delta(StreamDelta::default());
            if let Some(choice) = chunk.choices.first_mut() {
                choice.finish_reason = Some("length".to_string());
            }
            if let Some(usage) = value.get("response").and_then(|r| r.get("usage")) {
                chunk.usage = serde_json::from_value(usage.clone()).ok();
            }
            return Some(ParsedStreamPayload::Chunk(chunk));
        }
        return Some(ParsedStreamPayload::Error(format!(
            "response incomplete: {reason}"
        )));
    }
    // 部分 provider 用 SSE event: error 携带错误对象
    if event_type == "error" {
        let msg = extract_error_message(&value)
            .unwrap_or_else(|| "stream error event (no detail)".to_string());
        return Some(ParsedStreamPayload::Error(msg));
    }

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
            index: Some(extract_output_index(value)),
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

    // 捕获 reasoning item：`.done` 上的完整 `encrypted_content` 始终可回放；`.added`
    // 在现网是固定短串 stub（不可回放、回放会 400），但若未来网关把完整载荷移到
    // `.added` 需前向兼容。约定阈值：real 载荷实测 900-1500 长度，stub 恒为短串
    // （<100），以 256 为分界足以区分二者且不误伤短推理链的加密块。
    if item_type == "reasoning" {
        let encrypted_len = item
            .get("encrypted_content")
            .and_then(serde_json::Value::as_str)
            .map(|s| s.len())
            .unwrap_or(0);
        if encrypted_len == 0 {
            return Some(ParsedStreamPayload::Ignore);
        }
        const REASONING_ENCRYPTED_ADDED_MIN_LEN: usize = 256;
        let is_real = if event_type == "response.output_item.done" {
            true
        } else if event_type == "response.output_item.added" {
            encrypted_len >= REASONING_ENCRYPTED_ADDED_MIN_LEN
        } else {
            false
        };
        if is_real {
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
    if event_type.ends_with(".added") {
        // output_text 类型的 content_part.added 同样是协议重发：携带该 part 当前
        // 已存在的完整文本，与 output_text.delta 增量重叠。按增量格式解析但标记为
        // 重发（ReplayedChunk），由流层对 content 做未见后缀去重，避免正文跨事件
        // 路径重复渲染（用户可见"结论输出两遍"）。output_text 的 .done 保持
        // SnapshotChunk 不变（快照去重已覆盖）。
        return Some(ParsedStreamPayload::ReplayedChunk(single_delta_chunk(&text, "")));
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
        index: Some(fallback_index),
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
    fn embedded_output_item_done_captures_reasoning_despite_sse_event_name() {
        // 模拟缺少或发送不准确 SSE `event:` 名的网关：事件类型内嵌在 JSON 顶层
        // `type` 字段，encrypted reasoning 必须仍能被捕获，否则无法在下一轮
        // tool 请求中回放。`unknown.done` 还覆盖通用 SSE 分支提前 Ignore 的情况。
        let payload = r#"{"type":"response.output_item.done","sequence_number":1,"item":{"id":"rs_reason","type":"reasoning","encrypted_content":"enc-xyz"}}"#;
        for event_type in [None, Some(""), Some("message"), Some("unknown.done")] {
            match parse_stream_payload(provider::opencode_adapter(), payload, event_type) {
                ParsedStreamPayload::ReasoningItem(item) => {
                    assert_eq!(item["type"], "reasoning");
                    assert_eq!(item["encrypted_content"], "enc-xyz");
                }
                _ => panic!("expected reasoning item for event type {event_type:?}"),
            }
        }
    }

    #[test]
    fn terminal_event_name_takes_precedence_over_embedded_reasoning_item() {
        let payload = r#"{"type":"response.output_item.done","item":{"type":"reasoning","encrypted_content":"enc-xyz"}}"#;
        for event_type in ["done", "[DONE]", " done "] {
            assert!(matches!(
                parse_stream_payload(provider::opencode_adapter(), payload, Some(event_type)),
                ParsedStreamPayload::Done
            ));
        }
    }

    #[test]
    fn provider_error_takes_precedence_over_embedded_reasoning_item() {
        let payload = r#"{"type":"response.output_item.done","error":{"message":"boom"},"item":{"type":"reasoning","encrypted_content":"enc-xyz"}}"#;
        for event_type in [None, Some("message"), Some("unknown.done")] {
            assert!(matches!(
                parse_stream_payload(provider::opencode_adapter(), payload, event_type),
                ParsedStreamPayload::Error(_)
            ));
        }
    }

    #[test]
    fn sse_error_event_name_takes_precedence_over_embedded_reasoning_item() {
        let payload = r#"{"type":"response.output_item.done","item":{"type":"reasoning","encrypted_content":"enc-xyz"}}"#;
        for event_type in ["error", "response.failed", "response.incomplete"] {
            assert!(matches!(
                parse_stream_payload(provider::opencode_adapter(), payload, Some(event_type)),
                ParsedStreamPayload::Error(_)
            ));
        }
    }

    #[test]
    fn no_event_line_non_reasoning_done_falls_through_to_adapter() {
        // 非 reasoning 的 output_item.done 不应被补路径截获，仍走 adapter 宽松解析。
        let payload = r#"{"type":"response.output_item.done","sequence_number":1,"item":{"id":"msg_1","type":"message","content":[{"type":"output_text","text":"hi"}]}}"#;
        match parse_stream_payload(provider::opencode_adapter(), payload, None) {
            ParsedStreamPayload::Chunk(_) => {}
            _ => panic!("expected chunk from adapter path"),
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
    fn reasoning_added_with_short_stub_is_ignored() {
        // 现网 `.added` 为短 stub，不可回放；长度 <256 应被忽略
        let payload = r#"{"type":"response.output_item.added","item":{"type":"reasoning","encrypted_content":"short-stub"}}"#;
        for event_type in [Some("response.output_item.added"), None, Some(""), Some("message")] {
            match parse_stream_payload(provider::opencode_adapter(), payload, event_type) {
                ParsedStreamPayload::Ignore => {}
                _ => panic!("expected Ignore for short stub added for {event_type:?}"),
            }
        }
    }

    #[test]
    fn reasoning_added_with_long_encrypted_content_is_captured_via_fallback() {
        // 前向兼容：若网关把完整加密载荷移到 `.added`，长度 >=256 应被捕获
        let long = "a".repeat(512);
        let payload = format!(
            r#"{{"type":"response.output_item.added","item":{{"id":"rs_1","type":"reasoning","encrypted_content":"{long}"}}}}"#
        );
        for event_type in [Some("response.output_item.added"), None, Some(""), Some("message")] {
            match parse_stream_payload(provider::opencode_adapter(), &payload, event_type) {
                ParsedStreamPayload::ReasoningItem(item) => {
                    assert_eq!(item["type"], "reasoning");
                    assert_eq!(item["encrypted_content"].as_str().unwrap().len(), 512);
                }
                _ => panic!("expected ReasoningItem for long added for {event_type:?}"),
            }
        }
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
                assert_eq!(tool_call.index, Some(2));
                assert_eq!(tool_call.id, "fc_item_1");
                assert_eq!(tool_call.tool_type, "function");
                assert_eq!(tool_call.function.arguments, "{\"path\":\"a");
            }
            _ => panic!("expected tool-call delta chunk"),
        }
    }

    #[test]
    fn chat_completion_tool_call_without_index_keeps_none() {
        // When the gateway omits index, it must not default to 0 (parallel calls
        // would overwrite each other); keep None and let the stream layer compose
        // grouping keys by call id.
        let payload = r#"{"choices":[{"delta":{"tool_calls":[{"id":"call_a","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}"#;
        match parse_stream_payload(provider::openai_adapter(), payload, None) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.tool_calls[0].index, None);
            }
            _ => panic!("expected parsed chunk"),
        }
    }

    #[test]
    fn response_incomplete_max_output_tokens_maps_to_length_chunk() {
        let payload = r#"{"response":{"incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":3,"output_tokens":7}}}"#;
        match parse_stream_payload(
            provider::openai_adapter(),
            payload,
            Some("response.incomplete"),
        ) {
            ParsedStreamPayload::Chunk(chunk) => {
                assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
                assert!(chunk.usage.is_some());
            }
            _ => panic!("expected length-truncation chunk"),
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
                assert_eq!(tool_call.index, Some(2));
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
                assert_eq!(tool_call.index, Some(1));
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
                assert_eq!(tool_call.index, Some(1));
                assert_eq!(tool_call.id, "call_1");
                assert_eq!(tool_call.function.name, "write_file");
                assert_eq!(tool_call.function.arguments, "{\"path\":\"a.rs\"}");
            }
            _ => panic!("expected tool-call final snapshot chunk"),
        }
    }

    #[test]
    fn content_part_added_event_maps_to_replayed_content_chunk() {
        let payload = r#"{"part":{"type":"output_text","text":"hello"}}"#;
        match parse_stream_payload(
            provider::openai_adapter(),
            payload,
            Some("response.content_part.added"),
        ) {
            // added 事件携带 part 已存在的完整文本，属协议重发，标记为
            // ReplayedChunk 由流层对 content 做未见后缀去重。
            ParsedStreamPayload::ReplayedChunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.content, "hello");
            }
            _ => panic!("expected replayed content-part chunk"),
        }
    }

    #[test]
    fn content_part_done_event_stays_snapshot_content_chunk() {
        // output_text 的 .done 保持 SnapshotChunk（快照去重已覆盖），不受 added
        // 重发标记影响。
        let payload = r#"{"part":{"type":"output_text","text":"done text"}}"#;
        match parse_stream_payload(
            provider::openai_adapter(),
            payload,
            Some("response.content_part.done"),
        ) {
            ParsedStreamPayload::SnapshotChunk(chunk) => {
                assert_eq!(chunk.choices[0].delta.content, "done text");
            }
            _ => panic!("expected snapshot content-part chunk"),
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
        // max_output_tokens truncation is already mapped to finish_reason=length
        // (see the dedicated test above); other unknown reasons still surface as
        // hard errors, keeping the reason text for debugging.
        let payload = r#"{"type":"response.incomplete","response":{"incomplete_details":{"reason":"content_filter"}}}"#;
        match parse_stream_payload(
            provider::openai_adapter(),
            payload,
            Some("response.incomplete"),
        ) {
            ParsedStreamPayload::Error(msg) => {
                assert!(msg.contains("content_filter"), "msg was: {msg}");
            }
            _ => panic!("response.incomplete should surface as Error"),
        }
    }
}
