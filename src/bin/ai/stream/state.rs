use rust_tools::commonw::FastMap;

use crate::ai::{
    request::StreamChunk,
    types::{FunctionCall, StreamResult, ToolCall},
};

use super::{
    MarkdownStreamRenderer,
    splitter::{InternalToolCallStreamer, StreamSplitter},
};

pub(super) const THINKING_TAG_TEXT: &str = "╭─ thinking";
pub(super) const END_THINKING_TAG_TEXT: &str = "╰─ done thinking";

pub(super) struct StreamMarkers {
    pub(super) thinking_tag: String,
    pub(super) end_thinking_tag: String,
    pub(super) hidden_begin: &'static str,
    pub(super) hidden_end: &'static str,
}

impl StreamMarkers {
    pub(super) fn new() -> Self {
        Self {
            thinking_tag: THINKING_TAG_TEXT.to_string(),
            end_thinking_tag: END_THINKING_TAG_TEXT.to_string(),
            hidden_begin: "<meta:self_note>",
            hidden_end: "</meta:self_note>",
        }
    }
}

pub(super) struct StreamProcessingState {
    pub(super) framing: StreamFramingState,
    pub(super) render: StreamRenderState,
    pub(super) content: StreamContentState,
    /// Last-seen `(echoed_model, usage)` from any chunk during this stream.
    /// Handed to the kernel's `/dev/llm` when the stream finalizes.
    pub(super) pending_llm_usage: Option<(String, super::super::request::StreamUsage)>,
}

impl StreamProcessingState {
    pub(super) fn new() -> Self {
        Self {
            framing: StreamFramingState::new(),
            render: StreamRenderState::new(),
            content: StreamContentState::new(),
            pending_llm_usage: None,
        }
    }
}

pub(super) struct StreamFramingState {
    pub(super) decode_error_count: usize,
    pub(super) pending: Vec<u8>,
    pub(super) sse_event_type: Option<String>,
    pub(super) sse_event_data: String,
}

impl StreamFramingState {
    fn new() -> Self {
        Self {
            decode_error_count: 0,
            pending: Vec::with_capacity(4096),
            sse_event_type: None,
            sse_event_data: String::with_capacity(4096),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct SseEvent {
    pub(super) event_type: Option<String>,
    pub(super) payload: String,
}

pub(super) struct StreamRenderState {
    pub(super) markdown: MarkdownStreamRenderer,
    pub(super) waiting_hint_active: bool,
    pub(super) waiting_hint_buffering: bool,
    pub(super) printed_tool_calls_header: bool,
    pub(super) current_printing_index: Option<usize>,
    pub(super) terminal_dedupe: Option<TerminalDedupeState>,
    pub(super) terminal_splitter: StreamSplitter,
}

impl StreamRenderState {
    fn new() -> Self {
        Self {
            markdown: MarkdownStreamRenderer::new(),
            waiting_hint_active: false,
            waiting_hint_buffering: false,
            printed_tool_calls_header: false,
            current_printing_index: None,
            terminal_dedupe: None,
            terminal_splitter: StreamSplitter::new(),
        }
    }
}

pub(super) struct TerminalDedupeState {
    pub(super) candidate: String,
    pub(super) buffered_terminal_output: String,
}

pub(super) struct StreamContentState {
    pub(super) thinking_open: bool,
    pub(super) empty_choice_chunks: usize,
    pub(super) finish_reason_seen: bool,
    pub(super) saw_reasoning_output: bool,
    pub(super) tool_calls_map: FastMap<usize, ToolCallBuilder>,
    pub(super) assistant_text: String,
    pub(super) hidden_meta: String,
    /// 累积模型返回的 reasoning_content 原文（不含展示用的 thinking 标记），
    /// 终轮结束后通过 StreamResult 透传给 history，
    /// 以便下一轮请求把它原样回传给后端（DeepSeek thinking-mode 必须）。
    pub(super) reasoning_text: String,
    pub(super) hidden_meta_parse: HiddenMetaParseState,
    pub(super) internal_tool_call_idx: usize,
    pub(super) internal_tool_call_streamer: InternalToolCallStreamer,
}

impl StreamContentState {
    fn new() -> Self {
        Self {
            thinking_open: false,
            empty_choice_chunks: 0,
            finish_reason_seen: false,
            saw_reasoning_output: false,
            tool_calls_map: FastMap::default(),
            assistant_text: String::new(),
            hidden_meta: String::new(),
            reasoning_text: String::new(),
            hidden_meta_parse: HiddenMetaParseState::default(),
            internal_tool_call_idx: 0,
            internal_tool_call_streamer: InternalToolCallStreamer::new(),
        }
    }
}

#[derive(Default)]
pub(super) struct HiddenMetaParseState {
    pub(super) hidden_open: bool,
    pub(super) hidden_begin_match: usize,
    pub(super) hidden_end_match: usize,
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
    SnapshotChunk(StreamChunk),
}

#[derive(Default)]
pub(super) struct ToolCallBuilder {
    pub(super) id: String,
    pub(super) tool_type: String,
    pub(super) function_name: String,
    pub(super) arguments: String,
    pub(super) printed_arguments_len: usize,
}

impl ToolCallBuilder {
    pub(super) fn build(self) -> ToolCall {
        ToolCall {
            id: self.id,
            // 部分 provider 在 stream delta 中不返回 type 字段，默认为 "function"
            // 以符合 OpenAI 协议要求，避免发送 "type":"" 导致 400 错误。
            tool_type: if self.tool_type.is_empty() {
                "function".to_string()
            } else {
                self.tool_type
            },
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
