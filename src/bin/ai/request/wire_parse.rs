//! Streaming wire-protocol parsing primitives (shared between the provider adapter
//! layer and the stream normalization layer).
//!
//! `ParsedStreamPayload` was originally defined in `stream/state.rs`, and
//! `try_parse_stream_chunk` / `try_parse_stream_chunk_loose` in `stream/normalize.rs`.
//! The provider adapter layer needs these two parsing primitives, while the stream
//! normalization layer depends on the provider `ProviderAdapter` trait, forming a
//! provider ↔ stream circular dependency. Once the primitives moved down into this
//! neutral request submodule, both sides reference them uniformly through
//! `crate::ai::request` without importing each other's module trees.

use super::StreamChunk;
use serde_json::{Value, json};

/// How a provider declared a whole response ended, for wires whose end-of-response marker is not
/// a chat-completions `finish_reason` (OpenAI Responses: `response.completed` /
/// `response.incomplete`). It lets the stream layer tell "everything has been delivered" from "the
/// model was stopped early" without inspecting model names.
#[derive(Clone, Copy)]
pub(in crate::ai) enum ResponseTerminalStatus {
    /// The response finished normally: text, tool calls and usage have all been delivered.
    Completed,
    /// The provider stopped the response early (typically the output cap). The partial output is a
    /// truncated generation, not an answer.
    Incomplete,
}

pub(in crate::ai) enum ParsedStreamPayload {
    Ignore,
    Done,
    Chunk(StreamChunk),
    /// A provider terminal-state declaration. `chunk` carries what the plain [`Self::Chunk`]
    /// variant would have carried (usage when the terminal event reports it, and
    /// `finish_reason=length` for a response stopped at the output cap); the status only describes
    /// how the response ended, so a stop that named no reason carries neither field.
    ResponseTerminal {
        status: ResponseTerminalStatus,
        chunk: StreamChunk,
        /// Server-assembled visible text from the terminal event (`response.output[]`
        /// message content). Empty for wires without a server-side copy (chat
        /// completions) and for incomplete terminals. The finalize path reconciles
        /// it against the delta-assembled text: a strict-prefix shortfall means the
        /// transport lost the tail even though the provider declared completion.
        server_text: String,
    },
    /// `content_part.added` (output_text type) carries the complete text that currently
    /// exists in that part, overlapping with the incremental `output_text.delta`. This is
    /// a multi-path protocol re-delivery rather than new model content. It is still parsed
    /// in delta form (think_demux splitting, etc. still apply), but the stream layer
    /// additionally deduplicates content against the unseen suffix to avoid rendering the
    /// body twice across event paths.
    ReplayedChunk(StreamChunk),
    SnapshotChunk(StreamChunk),
    /// A `reasoning` output item of the Responses protocol, opened or finished. `item` carries the
    /// payload only when it is replayable (`.done`, or an `.added` with a real `encrypted_content`
    /// block); a replayable item is passed through verbatim into the next request's input so the
    /// model retains the previous hop's reasoning context, and is never persisted into history.
    /// `open` reports whether the provider still holds the item: this wire streams no content for
    /// the thinking itself, so silence while an item is open is work in progress, not a stall.
    ReasoningItem {
        item: Option<serde_json::Value>,
        open: bool,
    },
    /// The provider returned an error object or error event mid-stream, carrying a
    /// human-readable error message.
    Error(String),
}

fn try_parse_stream_chunk(payload: &str) -> Option<StreamChunk> {
    let mut chunk = serde_json::from_str::<StreamChunk>(payload).ok()?;
    chunk.merge_reasoning();
    Some(chunk)
}

/// Build a [`StreamChunk`] from an already-deserialized JSON value, applying the
/// same `merge_reasoning` post-processing as [`try_parse_stream_chunk`]. Callers
/// that already parsed the payload as a `Value` (e.g. for error detection) reuse
/// it here to avoid a second full JSON parse per stream chunk.
pub(in crate::ai) fn try_parse_stream_chunk_from_value(
    value: serde_json::Value,
) -> Option<StreamChunk> {
    let mut chunk = serde_json::from_value::<StreamChunk>(value).ok()?;
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

/// Anthropic Messages API SSE event names.
const ANTHROPIC_EVENTS: [&str; 7] = [
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
    "ping",
];

/// Parse an Anthropic Messages SSE event into a [`ParsedStreamPayload`].
///
/// The Messages wire (`/v1/messages`) streams `message_start` /
/// `content_block_start` / `content_block_delta` (text, thinking and tool-input
/// fragments) / `message_delta` (stop reason + usage) / `message_stop` events.
/// They are translated into the OpenAI-shaped [`StreamChunk`] vocabulary the
/// stream layer already understands: text → `delta.content`, thinking →
/// `delta.reasoning_content`, tool input → `delta.tool_calls`, `message_stop` →
/// [`ParsedStreamPayload::Done`]. Returns `None` when the payload is not an
/// Anthropic event, so callers fall back to the chat-completions parse path.
pub(in crate::ai) fn try_parse_anthropic_event(
    payload: &str,
    event_type: Option<&str>,
) -> Option<ParsedStreamPayload> {
    let value: Value = serde_json::from_str(payload).ok()?;
    let event = event_type
        .map(str::trim)
        .filter(|name| ANTHROPIC_EVENTS.iter().any(|known| known.eq_ignore_ascii_case(name)))
        .or_else(|| {
            value
                .get("type")
                .and_then(Value::as_str)
                .filter(|name| ANTHROPIC_EVENTS.iter().any(|known| known.eq_ignore_ascii_case(name)))
        })?;
    match event.to_ascii_lowercase().as_str() {
        "message_start" => value
            .pointer("/message/usage")
            .cloned()
            .and_then(|usage| try_parse_stream_chunk_from_value(json!({ "usage": usage })))
            .map(ParsedStreamPayload::Chunk),
        "content_block_start" => {
            let block = value.get("content_block")?;
            if block.get("type").and_then(Value::as_str) != Some("tool_use") {
                return Some(ParsedStreamPayload::Ignore);
            }
            let index = value.get("index").and_then(Value::as_u64);
            let id = block.get("id").and_then(Value::as_str).unwrap_or_default();
            let name = block.get("name").and_then(Value::as_str).unwrap_or_default();
            try_parse_stream_chunk_from_value(json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": index,
                            "id": id,
                            "function": { "name": name, "arguments": "" },
                        }],
                    },
                }],
            }))
            .map(ParsedStreamPayload::Chunk)
        }
        "content_block_delta" => {
            let delta = value.get("delta")?;
            let index = value.get("index").and_then(Value::as_u64);
            let chunk = match delta.get("type").and_then(Value::as_str) {
                Some("text_delta") => json!({
                    "choices": [{ "delta": {
                        "content": delta.get("text").and_then(Value::as_str).unwrap_or_default(),
                    } }],
                }),
                Some("thinking_delta") => json!({
                    "choices": [{ "delta": {
                        "reasoning_content": delta.get("thinking").and_then(Value::as_str).unwrap_or_default(),
                    } }],
                }),
                Some("input_json_delta") => json!({
                    "choices": [{ "delta": {
                        "tool_calls": [{
                            "index": index,
                            "function": {
                                "arguments": delta.get("partial_json").and_then(Value::as_str).unwrap_or_default(),
                            },
                        }],
                    } }],
                }),
                _ => return Some(ParsedStreamPayload::Ignore),
            };
            try_parse_stream_chunk_from_value(chunk).map(ParsedStreamPayload::Chunk)
        }
        "content_block_stop" | "ping" => Some(ParsedStreamPayload::Ignore),
        "message_delta" => {
            let mut chunk = serde_json::Map::new();
            if let Some(stop_reason) = value
                .pointer("/delta/stop_reason")
                .and_then(Value::as_str)
            {
                let finish_reason = match stop_reason {
                    "tool_use" => "tool_calls",
                    "max_tokens" => "length",
                    _ => "stop",
                };
                chunk.insert("choices".to_string(), json!([{ "finish_reason": finish_reason }]));
            }
            if let Some(usage) = value.get("usage").cloned() {
                chunk.insert("usage".to_string(), usage);
            }
            if chunk.is_empty() {
                Some(ParsedStreamPayload::Ignore)
            } else {
                try_parse_stream_chunk_from_value(Value::Object(chunk))
                    .map(ParsedStreamPayload::Chunk)
            }
        }
        "message_stop" => Some(ParsedStreamPayload::Done),
        _ => None,
    }
}

#[cfg(test)]
mod anthropic_event_tests {
    use super::*;

    // Golden payloads captured from a live stream against the OpenCode Zen
    // gateway (`/zen/v1/messages`, model claude-opus-5-5).

    #[test]
    fn parses_text_and_thinking_deltas() {
        let chunk = match try_parse_anthropic_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi! 👋 "}}"#,
            None,
        ) {
            Some(ParsedStreamPayload::Chunk(c)) => c,
            _ => panic!("expected a text chunk"),
        };
        assert_eq!(chunk.choices[0].delta.content, "Hi! 👋 ");

        let chunk = match try_parse_anthropic_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}"#,
            None,
        ) {
            Some(ParsedStreamPayload::Chunk(c)) => c,
            _ => panic!("expected a thinking chunk"),
        };
        assert_eq!(chunk.choices[0].delta.reasoning_content, "hmm");
    }

    #[test]
    fn parses_tool_use_blocks_and_stop_reason() {
        let chunk = match try_parse_anthropic_event(
            r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"web_search","input":{}}}"#,
            None,
        ) {
            Some(ParsedStreamPayload::Chunk(c)) => c,
            _ => panic!("expected a tool_use chunk"),
        };
        let call = &chunk.choices[0].delta.tool_calls[0];
        assert_eq!(call.index, Some(2));
        assert_eq!(call.id, "toolu_1");
        assert_eq!(call.function.name, "web_search");
        assert_eq!(call.function.arguments, "");

        let chunk = match try_parse_anthropic_event(
            r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"q\": \"ru"}}"#,
            None,
        ) {
            Some(ParsedStreamPayload::Chunk(c)) => c,
            _ => panic!("expected an input_json chunk"),
        };
        let call = &chunk.choices[0].delta.tool_calls[0];
        assert_eq!(call.index, Some(2));
        assert_eq!(call.function.arguments, "{\"q\": \"ru");

        let chunk = match try_parse_anthropic_event(
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null,"stop_details":null},"usage":{"input_tokens":11,"output_tokens":17}}"#,
            None,
        ) {
            Some(ParsedStreamPayload::Chunk(c)) => c,
            _ => panic!("expected a message_delta chunk"),
        };
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("tool_calls"));
        let usage = chunk.usage.expect("usage present");
        assert_eq!(usage.prompt_tokens, 11);
        assert_eq!(usage.completion_tokens, 17);
    }

    #[test]
    fn parses_terminal_events_and_usage() {
        assert!(matches!(
            try_parse_anthropic_event(r#"{"type":"message_stop"}"#, None),
            Some(ParsedStreamPayload::Done)
        ));
        assert!(matches!(
            try_parse_anthropic_event(r#"{"type":"ping"}"#, None),
            Some(ParsedStreamPayload::Ignore)
        ));

        let chunk = match try_parse_anthropic_event(
            r#"{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":11,"output_tokens":0}}}"#,
            None,
        ) {
            Some(ParsedStreamPayload::Chunk(c)) => c,
            _ => panic!("expected a message_start chunk"),
        };
        assert_eq!(chunk.usage.expect("usage present").prompt_tokens, 11);
    }

    #[test]
    fn recognizes_events_by_sse_event_name() {
        let parsed = try_parse_anthropic_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}"#,
            Some("content_block_delta"),
        );
        assert!(matches!(parsed, Some(ParsedStreamPayload::Chunk(_))));
    }

    #[test]
    fn ignores_non_anthropic_payloads() {
        assert!(try_parse_anthropic_event(
            r#"{"choices":[{"delta":{"content":"x"}}]}"#,
            None
        )
        .is_none());
        assert!(try_parse_anthropic_event("not json", None).is_none());
    }
}
