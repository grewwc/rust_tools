mod extract;
mod framing;
mod normalize;
mod render;
mod runtime;
mod splitter;
mod state;

pub(super) use render::markdown::MarkdownStreamRenderer;

use crate::ai::{request::StreamChunk, types::{App, StreamResult}};

pub(super) fn extract_chunk_text(
    chunk: &StreamChunk,
    thinking_tag: &str,
    end_thinking_tag: &str,
    thinking_open: &mut bool,
) -> String {
    extract::extract_chunk_text(chunk, thinking_tag, end_thinking_tag, thinking_open)
}

pub(super) async fn stream_response(
    app: &mut App,
    response: &mut reqwest::Response,
    current_history: &mut String,
    terminal_dedupe_candidate: Option<&str>,
) -> Result<StreamResult, Box<dyn std::error::Error>> {
    runtime::stream_response(app, response, current_history, terminal_dedupe_candidate).await
}

pub(super) fn line_looks_like_table_preview(line: &str) -> bool {
    render::table::line_looks_like_table_preview(line)
}

fn render_math_tex_to_unicode(s: &str) -> String {
    render::math::render_math_tex_to_unicode(s)
}
