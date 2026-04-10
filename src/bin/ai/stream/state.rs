use rust_tools::commonw::FastMap;

use crate::ai::{
    request::StreamChunk,
    theme::{ACCENT_MUTED, ACCENT_RULE, RESET},
    types::{FunctionCall, StreamResult, ToolCall},
};

use super::MarkdownStreamRenderer;

pub(super) struct StreamMarkers {
    pub(super) thinking_tag: String,
    pub(super) end_thinking_tag: String,
    pub(super) hidden_begin: &'static str,
    pub(super) hidden_end: &'static str,
}

impl StreamMarkers {
    pub(super) fn new() -> Self {
        Self {
            thinking_tag: format!("{ACCENT_RULE}╭─{RESET} {ACCENT_MUTED}thinking{RESET}"),
            end_thinking_tag: format!("{ACCENT_RULE}╰─{RESET} {ACCENT_MUTED}done thinking{RESET}"),
            hidden_begin: "<meta:self_note>",
            hidden_end: "</meta:self_note>",
        }
    }
}

pub(super) struct StreamProcessingState {
    pub(super) thinking_open: bool,
    pub(super) markdown: MarkdownStreamRenderer,
    pub(super) waiting_hint_active: bool,
    pub(super) waiting_hint_buffering: bool,
    pub(super) empty_choice_chunks: usize,
    pub(super) saw_reasoning_output: bool,
    pub(super) tool_calls_map: FastMap<usize, ToolCallBuilder>,
    pub(super) assistant_text: String,
    pub(super) hidden_meta: String,
    pub(super) hidden_open: bool,
    pub(super) hidden_begin_match: usize,
    pub(super) hidden_end_match: usize,
    pub(super) internal_tool_call_idx: usize,
    pub(super) printed_tool_calls_header: bool,
    pub(super) current_printing_index: Option<usize>,
    pub(super) decode_error_count: usize,
    pub(super) pending: Vec<u8>,
}

impl StreamProcessingState {
    pub(super) fn new() -> Self {
        Self {
            thinking_open: false,
            markdown: MarkdownStreamRenderer::new(),
            waiting_hint_active: false,
            waiting_hint_buffering: false,
            empty_choice_chunks: 0,
            saw_reasoning_output: false,
            tool_calls_map: FastMap::default(),
            assistant_text: String::new(),
            hidden_meta: String::new(),
            hidden_open: false,
            hidden_begin_match: 0,
            hidden_end_match: 0,
            internal_tool_call_idx: 0,
            printed_tool_calls_header: false,
            current_printing_index: None,
            decode_error_count: 0,
            pending: Vec::with_capacity(4096),
        }
    }
}

pub(super) enum StreamChunkStep {
    Continue,
    Stop,
    Return(StreamResult),
}

pub(super) enum ParsedStreamPayload {
    Ignore,
    Done,
    Chunk(StreamChunk),
}

#[derive(Default)]
pub(super) struct ToolCallBuilder {
    pub(super) id: String,
    pub(super) tool_type: String,
    pub(super) function_name: String,
    pub(super) arguments: String,
}

impl ToolCallBuilder {
    pub(super) fn build(self) -> ToolCall {
        ToolCall {
            id: self.id,
            tool_type: self.tool_type,
            function: FunctionCall {
                name: self.function_name,
                arguments: self.arguments,
            },
        }
    }
}

pub(super) struct InternalToolCall {
    pub(super) id: String,
    pub(super) tool_type: String,
    pub(super) function_name: String,
    pub(super) arguments: String,
}
