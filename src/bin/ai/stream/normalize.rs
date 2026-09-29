use crate::ai::{
    provider::ProviderAdapter,
    request::{
        ResponseTerminalStatus, StreamChoice, StreamChunk, StreamDelta, StreamFunctionCall,
        StreamToolCall, try_parse_stream_chunk_from_value,
    },
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
    // Sentinel compared ASCII-case-insensitively, matching the `done` / `[done]` SSE event-name
    // handling: a body-only stream that varies the sentinel's case must still stop the loop.
    if payload.eq_ignore_ascii_case("[DONE]") {
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

    // Some gateways (opencode zen / encrypted channel) omit a usable SSE `event:` name and carry
    // the event type only in the JSON top-level `type` field. The `response.output_item.done`
    // payload holding the encrypted reasoning would otherwise be ignored by the event-name branch
    // or silently swallowed by the adapter's loose chunk parse, which breaks replaying it in the
    // next tool request; a future gateway moving the full payload into `.added` is covered too.
    // Only the reasoning capture reacts to the JSON `type` fallback added here, other item types
    // keep their existing path. The pre-screen and the `type` comparisons it guards are
    // ASCII-case-insensitive, like every other protocol token in this file: a gateway that varies
    // the case of these markers reaches the same branches, while ordinary delta chunks still skip
    // the JSON parse.
    if ascii_contains(payload, "response.output_item.done")
        || ascii_contains(payload, "response.output_item.added")
        || ascii_contains(payload, "response.completed")
        || ascii_contains(payload, "response.incomplete")
        || ascii_contains(payload, "response.failed")
    {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) {
            if let Some(err_msg) = value.get("error").and_then(extract_error_message) {
                return ParsedStreamPayload::Error(err_msg);
            }
            if let Some(event_type) = value.get("type").and_then(serde_json::Value::as_str) {
                // Terminal events can also arrive without a usable SSE `event:` name; missing them
                // would turn a stopped response into an empty chunk that is silently swallowed
                // (same reasoning as the output_item branch below).
                let is_terminal_event = event_type.eq_ignore_ascii_case("response.completed")
                    || event_type.eq_ignore_ascii_case("response.incomplete")
                    || event_type.eq_ignore_ascii_case("response.failed");
                if is_terminal_event
                    && let Some(parsed) = parse_sse_event_payload(event_type, payload)
                {
                    return parsed;
                }
                if event_type.eq_ignore_ascii_case("response.output_item.done")
                    || event_type.eq_ignore_ascii_case("response.output_item.added")
                {
                    if let Some(parsed) = parse_output_item_event(event_type, &value) {
                        match &parsed {
                            // Reasoning items are terminal for this parse: they never fall through to
                            // the adapter's loose parse, and the open/closed state must reach the stream
                            // layer even when the payload itself is a non-replayable stub.
                            ParsedStreamPayload::ReasoningItem { .. } => return parsed,
                            // Other ignored items (e.g. a message) must fall back to the adapter's
                            // loose parse so a message `.done` is not swallowed as an ignored chunk.
                            ParsedStreamPayload::Ignore => {}
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

    // Non-SSE event path: parse the payload once and reuse the same Value for both
    // error detection and chunk construction (aligned with the SSE path, which
    // merges error detection into its single parse, so each chunk is no longer
    // parsed twice). StreamChunk fields are all #[serde(default)], so {"error":{...}}
    // is intercepted by the error check first; only when parsing fails or the JSON
    // cannot deserialize into a chunk (e.g. noisy wrapped JSON) do we fall back to
    // the adapter's loose parse — semantics are identical to before.
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) {
        if let Some(err_msg) = value.get("error").and_then(extract_error_message) {
            return ParsedStreamPayload::Error(err_msg);
        }
        if let Some(chunk) = try_parse_stream_chunk_from_value(value) {
            return ParsedStreamPayload::Chunk(chunk);
        }
    }

    adapter.parse_provider_chunk(payload)
}

fn parse_sse_event_payload(event_type: &str, payload: &str) -> Option<ParsedStreamPayload> {
    let event_type = event_type.trim();
    if event_type.is_empty() {
        return None;
    }
    if event_type.eq_ignore_ascii_case("done") || event_type.eq_ignore_ascii_case("[done]") {
        return Some(ParsedStreamPayload::Done);
    }
    // 统一解析一次并复用：各事件分支共享同一 Value，避免每个 chunk 双重 JSON
    // 解析；顶层 error 对象检测也在此完成（StreamChunk 全字段 #[serde(default)]，
    // 纯 error payload 会被静默反序列化为空 chunk 丢弃）。
    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    if let Some(err_msg) = value.get("error").and_then(extract_error_message) {
        return Some(ParsedStreamPayload::Error(err_msg));
    }
    if event_type.eq_ignore_ascii_case("response.completed") {
        // Some gateways report the terminal state inside this event instead of emitting a
        // dedicated `response.incomplete` / `response.failed` (status other than `completed`, an
        // `incomplete_details` block, or an error object). Such a response was stopped early, so it
        // must be surfaced as such: reading it as a normal completion is what leaves a cut-off
        // turn looking finished.
        if let Some(response) = value.get("response") {
            if let Some(err_msg) = response.get("error").and_then(extract_error_message) {
                return Some(ParsedStreamPayload::Error(err_msg));
            }
            if response_declares_incomplete(response) {
                return Some(incomplete_response_payload(response));
            }
        }
        // The Responses API puts the final usage in response.usage instead of the top-level usage
        // of compatible streams. Wrap it in an ordinary chunk to reuse the existing usage
        // accounting, and record the declared terminal state: the stream layer uses it to stop at
        // the response's own end marker rather than waiting for the socket to close, so a
        // connection that stays open afterwards cannot discard the tool calls already delivered.
        // The marker must not depend on the usage block: a gateway that omits usage still declared
        // the response finished, and the stream layer needs that declaration to end here.
        let usage = value
            .get("response")
            .and_then(|response| response.get("usage"))
            .and_then(|usage| serde_json::from_value(usage.clone()).ok());
        return Some(ParsedStreamPayload::ResponseTerminal {
            status: ResponseTerminalStatus::Completed,
            chunk: StreamChunk {
                usage,
                ..Default::default()
            },
        });
    }
    // OpenAI Responses API 错误/不完整事件——必须显式处理，否则会 fallthrough
    // 到 parse_provider_chunk 被当成空 chunk 静默丢弃。
    if event_type.eq_ignore_ascii_case("response.failed") {
        let msg = value
            .get("response")
            .and_then(|r| r.get("error"))
            .and_then(extract_error_message)
            .unwrap_or_else(|| "response failed (no error detail)".to_string());
        return Some(ParsedStreamPayload::Error(msg));
    }
    if event_type.eq_ignore_ascii_case("response.incomplete") {
        let fallback = serde_json::Value::Null;
        let response = value.get("response").unwrap_or(&fallback);
        return Some(incomplete_response_payload(response));
    }
    // 部分 provider 用 SSE event: error 携带错误对象
    if event_type.eq_ignore_ascii_case("error") {
        let msg = extract_error_message(&value)
            .unwrap_or_else(|| "stream error event (no detail)".to_string());
        return Some(ParsedStreamPayload::Error(msg));
    }

    if let Some(parsed) = parse_function_call_arguments_event(event_type, &value) {
        return Some(parsed);
    }
    if let Some(parsed) = parse_output_item_event(event_type, &value) {
        return Some(parsed);
    }
    if let Some(parsed) = parse_content_part_event(event_type, &value) {
        return Some(parsed);
    }
    if let Some(parsed) = parse_refusal_event(event_type, &value) {
        return Some(parsed);
    }
    if ascii_contains(event_type, "reasoning")
        && (ascii_ends_with(event_type, ".delta") || ascii_ends_with(event_type, ".done"))
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
        return Some(textual_event_chunk(event_type, "", &text));
    }
    if (ascii_contains(event_type, "output_text") || ascii_contains(event_type, "content"))
        && (ascii_ends_with(event_type, ".delta") || ascii_ends_with(event_type, ".done"))
    {
        let text = extract_event_text(&value, &["delta", "text", "content"]);
        if text.is_empty() {
            return Some(ParsedStreamPayload::Ignore);
        }
        return Some(textual_event_chunk(event_type, &text, ""));
    }

    if ascii_ends_with(event_type, ".done")
        || ascii_ends_with(event_type, ".added")
        || ascii_ends_with(event_type, ".part.done")
    {
        return Some(ParsedStreamPayload::Ignore);
    }

    None
}

/// Case-insensitive `contains` for an ASCII needle on a borrowed string. Avoids the
/// per-event `to_ascii_lowercase()` allocation on the SSE hot path; `eq_ignore_ascii_case`
/// leaves non-ASCII bytes unchanged, exactly matching `to_ascii_lowercase` semantics.
fn ascii_contains(haystack: &str, needle: &str) -> bool {
    haystack
        .as_bytes()
        .windows(needle.len())
        .any(|w| w.eq_ignore_ascii_case(needle.as_bytes()))
}

/// Case-insensitive `ends_with` for an ASCII needle on a borrowed string.
fn ascii_ends_with(haystack: &str, needle: &str) -> bool {
    let h = haystack.as_bytes();
    let n = needle.as_bytes();
    h.len() >= n.len() && h[h.len() - n.len()..].eq_ignore_ascii_case(n)
}

fn parse_function_call_arguments_event(
    event_type: &str,
    value: &serde_json::Value,
) -> Option<ParsedStreamPayload> {
    if !ascii_contains(event_type, "function_call_arguments")
        || !(ascii_ends_with(event_type, ".delta") || ascii_ends_with(event_type, ".done"))
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
    if !(event_type.eq_ignore_ascii_case("response.output_item.added")
        || event_type.eq_ignore_ascii_case("response.output_item.done"))
    {
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
        // Capture the item for same-turn tool-chain replay: a `.done` payload carries the real
        // `encrypted_content`, an `.added` payload is a fixed short stub in production (not
        // replayable — replaying it returns 400) but is captured too in case a future gateway moves
        // the full payload there. Fixed threshold: real payloads measure 900-1500 chars while stubs
        // stay short (<100), so 256 separates them without misjudging a short chain's encrypted block.
        let encrypted_len = item
            .get("encrypted_content")
            .and_then(serde_json::Value::as_str)
            .map(|s| s.len())
            .unwrap_or(0);
        const REASONING_ENCRYPTED_ADDED_MIN_LEN: usize = 256;
        let is_done_event = event_type.eq_ignore_ascii_case("response.output_item.done");
        let is_real = encrypted_len > 0
            && (is_done_event || encrypted_len >= REASONING_ENCRYPTED_ADDED_MIN_LEN);
        // The event name is the primary open/close signal; a terminal item status closes the item
        // too, so a provider that reports the final state on an `.added` line still resolves.
        let terminal_status = item
            .get("status")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|status| {
                matches!(
                    status.trim().to_ascii_lowercase().as_str(),
                    "completed" | "incomplete" | "failed"
                )
            });
        let open = !is_done_event && !terminal_status;
        return Some(ParsedStreamPayload::ReasoningItem {
            item: is_real.then(|| item.clone()),
            open,
        });
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
    if !(event_type.eq_ignore_ascii_case("response.content_part.added")
        || event_type.eq_ignore_ascii_case("response.content_part.done"))
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
    if ascii_ends_with(event_type, ".added") {
        // output_text 类型的 content_part.added 同样是协议重发：携带该 part 当前
        // 已存在的完整文本，与 output_text.delta 增量重叠。按增量格式解析但标记为
        // 重发（ReplayedChunk），由流层对 content 做未见后缀去重，避免正文跨事件
        // 路径重复渲染（用户可见"结论输出两遍"）。output_text 的 .done 保持
        // SnapshotChunk 不变（快照去重已覆盖）。
        return Some(ParsedStreamPayload::ReplayedChunk(single_delta_chunk(
            &text, "",
        )));
    }
    Some(textual_event_chunk(event_type, &text, ""))
}

fn parse_refusal_event(event_type: &str, value: &serde_json::Value) -> Option<ParsedStreamPayload> {
    if !ascii_contains(event_type, "refusal")
        || !(ascii_ends_with(event_type, ".delta") || ascii_ends_with(event_type, ".done"))
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
    if ascii_ends_with(event_type, ".done") {
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
    if ascii_ends_with(event_type, ".done") {
        ParsedStreamPayload::SnapshotChunk(chunk)
    } else {
        ParsedStreamPayload::Chunk(chunk)
    }
}

/// Whether a Responses payload declares the response stopped rather than finished: a populated
/// `incomplete_details` block, or any status other than `completed`. Gateways differ on where the
/// terminal state is carried (a dedicated `response.incomplete` event, or the final event's own
/// payload), and a stopped response must never be read as a normal end.
fn response_declares_incomplete(response: &serde_json::Value) -> bool {
    // The field is present-but-null on a normal completion, so presence alone must not count.
    if response
        .get("incomplete_details")
        .is_some_and(|details| !details.is_null())
    {
        return true;
    }
    response
        .get("status")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|status| !status.trim().eq_ignore_ascii_case("completed"))
}

/// Payload for a response the provider declared incomplete. `max_output_tokens` maps onto
/// `finish_reason=length` (mirrors @ai-sdk/openai's mapOpenAIResponseFinishReason): the partial
/// text produced so far is kept, usage stays available like on a completed response, and the
/// existing truncation ladders keep reading the shape they already handle.
///
/// A *named* stop reason other than the output cap stays an error, because a retry cannot fix a
/// content filter or a server-side cancel. A stop declared without any reason (a gateway that keeps
/// a non-`completed` status while dropping `incomplete_details`) is reported as a declared stop
/// instead: nothing proves a retry is useless there, and turning it into a terminal error is what
/// ends a turn whose partial body was already produced.
fn incomplete_response_payload(response: &serde_json::Value) -> ParsedStreamPayload {
    let reason = response
        .get("incomplete_details")
        .and_then(|details| details.get("reason"))
        .and_then(serde_json::Value::as_str)
        .filter(|reason| !reason.trim().is_empty());
    if let Some(reason) = reason.filter(|reason| !reason.eq_ignore_ascii_case("max_output_tokens")) {
        return ParsedStreamPayload::Error(format!("response incomplete: {reason}"));
    }
    // Reaching here means the output cap (named in `reason`) or an unnamed stop. Only the output cap
    // claims `finish_reason=length`; an unnamed stop leaves the field unset, because the provider
    // never said the model hit the cap and a fabricated `length` would drive cap-specific repair
    // (halving max_tokens on a zero-output stop) for a cause that was never reported.
    let finish_reason = reason.is_some().then(|| "length".to_string());
    let mut chunk = stream_chunk_with_delta(StreamDelta::default());
    if let Some(choice) = chunk.choices.first_mut() {
        choice.finish_reason = finish_reason;
    }
    if let Some(usage) = response.get("usage") {
        chunk.usage = serde_json::from_value(usage.clone()).ok();
    }
    ParsedStreamPayload::ResponseTerminal {
        status: ResponseTerminalStatus::Incomplete,
        chunk,
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
    use crate::ai::{
        provider,
        request::ResponseTerminalStatus,
        stream::state::ParsedStreamPayload,
    };

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
        // Gateways that omit or mislabel the SSE `event:` name embed the type in the JSON `type`
        // field; the encrypted reasoning item must still be captured, otherwise it cannot be
        // replayed on the next tool request. `unknown.done` also covers the generic SSE branch that
        // would otherwise resolve to an ignored chunk first. The closed state travels with the
        // payload (see `StreamContentState::open_output_items`).
        let payload = r#"{"type":"response.output_item.done","sequence_number":1,"item":{"id":"rs_reason","type":"reasoning","encrypted_content":"enc-xyz"}}"#;
        for event_type in [None, Some(""), Some("message"), Some("unknown.done")] {
            match parse_stream_payload(provider::opencode_adapter(), payload, event_type) {
                ParsedStreamPayload::ReasoningItem { item: Some(item), open } => {
                    assert!(!open, "a `.done` item is closed");
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
        for event_type in ["error", "response.failed"] {
            assert!(matches!(
                parse_stream_payload(provider::opencode_adapter(), payload, Some(event_type)),
                ParsedStreamPayload::Error(_)
            ));
        }
        // The `response.incomplete` event name still wins over the embedded item, but it declares a
        // stop rather than an error: with no reason in the payload there is nothing proving a retry
        // is useless, so the caller must be able to retry instead of failing the turn.
        assert!(matches!(
            parse_stream_payload(
                provider::opencode_adapter(),
                payload,
                Some("response.incomplete")
            ),
            ParsedStreamPayload::ResponseTerminal { .. }
        ));
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
    fn body_type_markers_match_ascii_case_insensitively() {
        // A gateway that omits the SSE `event:` name may vary the case of the JSON `type` marker
        // too. Such a payload must reach the same branch as its lowercase form instead of falling
        // through to the loose parse, which turns a stopped response into an empty chunk that is
        // silently swallowed (leaving the stream to end on the socket instead of on the provider's
        // own end marker).
        let payload = r#"{"type":"RESPONSE.COMPLETED","response":{"id":"r1","status":"INCOMPLETE","incomplete_details":{"reason":"MAX_OUTPUT_TOKENS"}}}"#;
        assert!(matches!(
            parse_stream_payload(provider::opencode_adapter(), payload, None),
            ParsedStreamPayload::ResponseTerminal {
                status: ResponseTerminalStatus::Incomplete,
                ..
            }
        ));

        // The output_item capture branch exists for the same no-event-name gateways.
        let payload = r#"{"type":"RESPONSE.OUTPUT_ITEM.DONE","item":{"type":"REASONING","encrypted_content":"enc-xyz"}}"#;
        assert!(matches!(
            parse_stream_payload(provider::opencode_adapter(), payload, None),
            ParsedStreamPayload::ReasoningItem { .. }
        ));
    }

    #[test]
    fn body_done_sentinel_matches_ascii_case_insensitively() {
        for payload in ["[DONE]", "[done]", "[Done]"] {
            assert!(matches!(
                parse_stream_payload(provider::opencode_adapter(), payload, None),
                ParsedStreamPayload::Done
            ));
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
    fn reasoning_added_with_short_stub_reports_open_item_without_replay_payload() {
        // The live `.added` payload is a short stub (not replayable — replaying it returns 400), so
        // no payload is captured, but the open state must still be reported: the stream layer uses
        // it to keep a hidden-thinking silence from being read as a stalled stream.
        let payload = r#"{"type":"response.output_item.added","item":{"type":"reasoning","encrypted_content":"short-stub"}}"#;
        for event_type in [
            Some("response.output_item.added"),
            None,
            Some(""),
            Some("message"),
        ] {
            match parse_stream_payload(provider::opencode_adapter(), payload, event_type) {
                ParsedStreamPayload::ReasoningItem { item: None, open: true } => {}
                _ => panic!("expected an open, non-replayable reasoning item for {event_type:?}"),
            }
        }
    }

    #[test]
    fn reasoning_added_with_long_encrypted_content_is_captured_via_fallback() {
        // Forward compatibility: if a gateway moves the full encrypted payload to `.added`
        // (>=256 chars), it must be captured for replay and reported as an open item.
        let long = "a".repeat(512);
        let payload = format!(
            r#"{{"type":"response.output_item.added","item":{{"id":"rs_1","type":"reasoning","encrypted_content":"{long}"}}}}"#
        );
        for event_type in [
            Some("response.output_item.added"),
            None,
            Some(""),
            Some("message"),
        ] {
            match parse_stream_payload(provider::opencode_adapter(), &payload, event_type) {
                ParsedStreamPayload::ReasoningItem { item: Some(item), open: true } => {
                    assert_eq!(item["type"], "reasoning");
                    assert_eq!(item["encrypted_content"].as_str().unwrap().len(), 512);
                }
                _ => panic!("expected an open reasoning item for long added for {event_type:?}"),
            }
        }
    }

    #[test]
    fn reasoning_added_without_encrypted_content_reports_open_item() {
        // Thinking whose content is not streamed at all (no encrypted block, empty summary): the
        // item still reports that the provider holds it open.
        let payload = r#"{"type":"response.output_item.added","item":{"id":"rs_1","type":"reasoning","status":"in_progress","summary":[]}}"#;
        match parse_stream_payload(
            provider::opencode_adapter(),
            payload,
            Some("response.output_item.added"),
        ) {
            ParsedStreamPayload::ReasoningItem {
                item: None,
                open: true,
            } => {}
            _ => panic!("expected an open reasoning item without a replay payload"),
        }
    }

    #[test]
    fn reasoning_done_without_encrypted_content_reports_closed_item() {
        let payload = r#"{"type":"response.output_item.done","item":{"id":"rs_1","type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":"done"}]}}"#;
        match parse_stream_payload(
            provider::opencode_adapter(),
            payload,
            Some("response.output_item.done"),
        ) {
            ParsedStreamPayload::ReasoningItem {
                item: None,
                open: false,
            } => {}
            _ => panic!("expected a closed reasoning item without a replay payload"),
        }
    }

    #[test]
    fn reasoning_added_with_terminal_status_reports_closed_item() {
        // A provider that reports the final state on the `.added` line must not leave the item open
        // forever: a terminal status closes it.
        let long = "a".repeat(512);
        let payload = format!(
            r#"{{"type":"response.output_item.added","item":{{"id":"rs_1","type":"reasoning","status":"completed","encrypted_content":"{long}"}}}}"#
        );
        match parse_stream_payload(
            provider::opencode_adapter(),
            &payload,
            Some("response.output_item.added"),
        ) {
            ParsedStreamPayload::ReasoningItem {
                item: Some(_),
                open: false,
            } => {}
            _ => panic!("expected a closed reasoning item for a terminal status"),
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
            ParsedStreamPayload::ResponseTerminal {
                status: ResponseTerminalStatus::Incomplete,
                chunk,
            } => {
                assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
                assert!(chunk.usage.is_some());
            }
            _ => panic!("expected an incomplete terminal declaration"),
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
            ParsedStreamPayload::ResponseTerminal {
                status: ResponseTerminalStatus::Completed,
                chunk,
            } => {
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
            _ => panic!("response.completed should declare a completed terminal state"),
        }
    }

    #[test]
    fn response_completed_with_null_incomplete_details_stays_completed() {
        // Real gateway shape: the completed event carries `incomplete_details` as an explicit null
        // (and `error` as null) while the status is `completed`. Presence alone must not be read as
        // an incomplete declaration, or every completed turn would abort as a stream error.
        let payload = r#"{
            "type": "response.completed",
            "response": {
                "status": "completed",
                "error": null,
                "incomplete_details": null,
                "usage": {"input_tokens": 40, "output_tokens": 12}
            }
        }"#;
        match parse_stream_payload(
            provider::openai_adapter(),
            payload,
            Some("response.completed"),
        ) {
            ParsedStreamPayload::ResponseTerminal {
                status: ResponseTerminalStatus::Completed,
                chunk,
            } => {
                assert!(chunk.usage.is_some());
            }
            _ => panic!("a completed response with null details must stay completed"),
        }
    }

    #[test]
    fn response_completed_without_usage_still_declares_completion() {
        let payload = r#"{"type":"response.completed","response":{"status":"completed","incomplete_details":null}}"#;
        match parse_stream_payload(
            provider::openai_adapter(),
            payload,
            Some("response.completed"),
        ) {
            ParsedStreamPayload::ResponseTerminal {
                status: ResponseTerminalStatus::Completed,
                chunk,
            } => {
                assert!(chunk.usage.is_none());
            }
            _ => panic!("the completion marker must not depend on a usage block"),
        }
    }

    #[test]
    fn response_completed_with_incomplete_status_maps_to_incomplete_terminal() {
        // Gateways may report the terminal state inside `response.completed` instead of emitting a
        // dedicated `response.incomplete`; reading it as a normal completion is what ends a cut-off
        // turn silently, so it must surface as an incomplete declaration.
        let payload = r#"{
            "type": "response.completed",
            "response": {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 5, "output_tokens": 9}
            }
        }"#;
        match parse_stream_payload(
            provider::openai_adapter(),
            payload,
            Some("response.completed"),
        ) {
            ParsedStreamPayload::ResponseTerminal {
                status: ResponseTerminalStatus::Incomplete,
                chunk,
            } => {
                assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
                assert!(chunk.usage.is_some());
            }
            _ => panic!("an incomplete response must not be read as a normal completion"),
        }
    }

    #[test]
    fn response_incomplete_without_event_name_is_not_swallowed() {
        // Same signal with no usable SSE `event:` name: the type has to be read from the JSON body,
        // otherwise the payload deserializes into an empty chunk and the cut disappears.
        let payload =
            r#"{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}"#;
        match parse_stream_payload(provider::openai_adapter(), payload, None) {
            ParsedStreamPayload::ResponseTerminal {
                status: ResponseTerminalStatus::Incomplete,
                chunk,
            } => {
                assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
            }
            _ => panic!("response.incomplete without an event name must still surface"),
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
        // (see the dedicated test above); a named policy stop still surfaces as a
        // hard error, keeping the reason text for debugging.
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

    #[test]
    fn response_incomplete_without_a_reason_is_a_declared_stop() {
        // A gateway can declare the stop while dropping the reason (on this wire even a normal
        // completion carries `incomplete_details: null`, so an absent or empty block must not become
        // a terminal error). Nothing proves a retry is useless for an unnamed stop, so it stays a
        // declared stop instead of ending a turn whose partial body was already produced.
        for (event_type, payload) in [
            (
                "response.incomplete",
                r#"{"type":"response.incomplete","response":{"status":"incomplete"}}"#,
            ),
            (
                "response.incomplete",
                r#"{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":null}}"#,
            ),
            (
                "response.incomplete",
                r#"{"type":"response.incomplete","response":{"incomplete_details":{"reason":"   "}}}"#,
            ),
            ("response.incomplete", r#"{"type":"response.incomplete"}"#),
            (
                "response.completed",
                r#"{"type":"response.completed","response":{"status":"incomplete","incomplete_details":null,"usage":{"input_tokens":5,"output_tokens":9}}}"#,
            ),
        ] {
            match parse_stream_payload(provider::openai_adapter(), payload, Some(event_type)) {
                ParsedStreamPayload::ResponseTerminal {
                    status: ResponseTerminalStatus::Incomplete,
                    chunk,
                } => {
                    // The unnamed stop must not claim the output cap: no fabricated finish reason,
                    // otherwise cap-specific repair would run for a cause nobody reported.
                    assert!(
                        chunk.choices[0].finish_reason.is_none(),
                        "an unnamed stop must not fabricate a finish reason: {payload}"
                    );
                }
                _ => panic!("an unnamed stop must stay a declared stop, not an error: {payload}"),
            }
        }
    }
}
