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
