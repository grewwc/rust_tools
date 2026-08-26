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

pub(in crate::ai) enum ParsedStreamPayload {
    Ignore,
    Done,
    Chunk(StreamChunk),
    /// `content_part.added` (output_text type) carries the complete text that currently
    /// exists in that part, overlapping with the incremental `output_text.delta`. This is
    /// a multi-path protocol re-delivery rather than new model content. It is still parsed
    /// in delta form (think_demux splitting, etc. still apply), but the stream layer
    /// additionally deduplicates content against the unseen suffix to avoid rendering the
    /// body twice across event paths.
    ReplayedChunk(StreamChunk),
    SnapshotChunk(StreamChunk),
    /// A complete `reasoning` output item returned by the Responses protocol (with `id` /
    /// `encrypted_content` / `summary`). Used for same-turn tool-chain replay: passed
    /// through verbatim into the next request's input so the model retains the previous
    /// hop's reasoning context. Never persisted into history.
    ReasoningItem(serde_json::Value),
    /// The provider returned an error object or error event mid-stream, carrying a
    /// human-readable error message.
    Error(String),
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
