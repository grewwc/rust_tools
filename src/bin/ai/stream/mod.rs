mod extract;
mod framing;
mod inline_recovery;
mod normalize;
mod render;
mod runtime;
pub(crate) mod side_note_input;
mod splitter;
mod state;
mod think_demux;

pub(super) use render::markdown::MarkdownStreamRenderer;
pub(in crate::ai) use render::markdown::clamp_line_to_terminal_row_with_reserve;

use crate::ai::{
    request::StreamChunk,
    types::{App, StreamResult},
};

/// Renders a complete block of Markdown text to stdout in one go (used in
/// non-streaming scenarios, such as `-ns` search results).
pub(crate) fn render_markdown_block(text: &str) -> std::io::Result<()> {
    use std::io::IsTerminal;
    let tty = std::io::stdout().is_terminal();
    let mut renderer = MarkdownStreamRenderer::new_with_tty(tty);
    renderer.write_block(text, false)?;
    renderer.flush_pending()
}

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
