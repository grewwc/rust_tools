use std::collections::VecDeque;

use rust_tools::cw::SkipMap;

use crate::ai::{
    request::StreamChunk,
    types::{FunctionCall, StreamResult, ToolCall},
};

use super::{
    MarkdownStreamRenderer,
    splitter::{
        AnthropicXmlToolCallStreamer, BareXmlToolCallStreamer, HermesXmlToolCallStreamer,
        InternalToolCallStreamer, StreamSplitter,
    },
};

pub(super) const THINKING_TAG_TEXT: &str = "╭─ thinking";
pub(super) const END_THINKING_TAG_TEXT: &str = "╰─ done thinking";

pub(super) struct StreamMarkers {
    pub(super) thinking_tag: String,
    pub(super) end_thinking_tag: String,
    pub(super) hidden_begin: &'static str,
    pub(super) hidden_end: &'static str,
    pub(super) subagent_fold_header: Option<String>,
    pub(super) subagent_fold_footer: Option<String>,
}

impl StreamMarkers {
    pub(super) fn new() -> Self {
        Self {
            thinking_tag: THINKING_TAG_TEXT.to_string(),
            end_thinking_tag: END_THINKING_TAG_TEXT.to_string(),
            hidden_begin: "<meta:self_note>",
            hidden_end: "</meta:self_note>",
            subagent_fold_header: None,
            subagent_fold_footer: None,
        }
    }

    pub(super) fn enable_subagent_preview(&mut self, agent_name: &str) {
        let agent_name = agent_name.trim();
        let suffix = if agent_name.is_empty() {
            String::new()
        } else {
            format!(" {agent_name}")
        };
        self.subagent_fold_header = Some(format!("subagent{suffix}"));
        self.subagent_fold_footer = Some(format!("done subagent{suffix}"));
    }

    pub(super) fn subagent_preview_enabled(&self) -> bool {
        self.subagent_fold_header.is_some() && self.subagent_fold_footer.is_some()
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
    pub(super) thinking_fold: ThinkingFoldState,
    pub(super) subagent_fold: ThinkingFoldState,
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
            thinking_fold: ThinkingFoldState::new(),
            subagent_fold: ThinkingFoldState::new_with_labels("subagent", "done subagent", false),
        }
    }
}

/// Thinking 折叠状态：维护一个滚动窗口，只在终端展示最近 N 行 thinking 内容，
/// 旧的行被折叠起来，同时保持流式实时输出。
pub(super) struct ThinkingFoldState {
    /// 最大可见行数（不含折叠提示行）
    pub(super) max_visible_lines: usize,
    /// 已完成的 thinking 行（ring buffer，只保留最近 max_visible_lines 行）
    pub(super) recent_lines: VecDeque<String>,
    /// 当前正在流式输出的不完整行
    pub(super) current_line: String,
    /// 总完成行数（含已被折叠的）
    pub(super) total_lines: usize,
    /// 当前折叠窗口（仅正文，不含 header）占用的 terminal 物理行数
    pub(super) window_rows: usize,
    /// 上次真正写到 terminal 的正文纯文本行（不含 ANSI / header），用于在 terminal
    /// resize 后按**当前**列宽重算旧窗口会占多少物理行，避免 cursor-up 擦不干净。
    pub(super) rendered_body_lines: Vec<String>,
    /// 是否处于活跃的 thinking 折叠模式
    pub(super) active: bool,
    /// header（`╭─ thinking`）是否已落地。header 只打印一次并被锚定在重画区域之上，
    /// 绝不随正文一起被 cursor-up 擦除重画——这样即便正文擦除失步也无法再生出第二个
    /// header，从根上杜绝「孤儿 header 叠加」的渲染 bug。
    pub(super) header_drawn: bool,
    /// 折叠块 header 文案（如 `thinking` / `subagent explore`）。
    pub(super) header_label: String,
    /// 折叠块 footer 文案（如 `done thinking` / `done subagent explore`）。
    pub(super) footer_label: String,
    /// 是否在折叠窗口里跳过空白行。thinking 适合紧凑展示；subagent 正文保持原样。
    pub(super) skip_blank_lines: bool,
}

impl ThinkingFoldState {
    pub(super) fn new() -> Self {
        Self::new_with_labels("thinking", "done thinking", true)
    }

    pub(super) fn new_with_labels(
        header_label: impl Into<String>,
        footer_label: impl Into<String>,
        skip_blank_lines: bool,
    ) -> Self {
        Self {
            max_visible_lines: usize::MAX,
            recent_lines: VecDeque::new(),
            current_line: String::new(),
            total_lines: 0,
            window_rows: 0,
            rendered_body_lines: Vec::new(),
            active: false,
            header_drawn: false,
            header_label: header_label.into(),
            footer_label: footer_label.into(),
            skip_blank_lines,
        }
    }

    pub(super) fn set_labels(
        &mut self,
        header_label: impl Into<String>,
        footer_label: impl Into<String>,
    ) {
        self.header_label = header_label.into();
        self.footer_label = footer_label.into();
    }

    pub(super) fn reset(&mut self) {
        self.recent_lines.clear();
        self.current_line.clear();
        self.total_lines = 0;
        self.window_rows = 0;
        self.rendered_body_lines.clear();
        self.active = false;
        self.header_drawn = false;
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
    /// 最近一个非空 `finish_reason` 的具体值（如 `stop` / `length` / `tool_calls`）。
    /// `length` 表示服务端因输出上限截断，是比"工具 JSON 解析失败"更早、更准的
    /// 截断信号，用于把本轮 outcome 升级为可重试的 `Truncated`。
    pub(super) finish_reason_value: Option<String>,
    /// 本轮是否发生过「因 arguments JSON 不完整而丢弃工具调用」。大文件 `write_file`
    /// 撞上输出上限被截断时最典型：JSON 半截 → 被丢弃 → 本轮无有效工具调用。
    /// 若仅凭"无工具调用 + 有文本"会被误判为正常完成而静默结束。
    pub(super) dropped_malformed_tool_call: bool,
    pub(super) saw_reasoning_output: bool,
    pub(super) tool_calls_map: SkipMap<usize, ToolCallBuilder>,
    pub(super) assistant_text: String,
    pub(super) hidden_meta: String,
    /// 累积模型返回的 reasoning_content 原文（不含展示用的 thinking 标记），
    /// 终轮结束后通过 StreamResult 透传给 history，
    /// 以便下一轮请求把它原样回传给后端（DeepSeek thinking-mode 必须）。
    pub(super) reasoning_text: String,
    /// 本轮从 Responses 流捕获的完整 `reasoning` output items（含 encrypted_content）。
    /// 仅用于同 turn 工具链回放，不落持久化历史。
    pub(super) reasoning_items: Vec<serde_json::Value>,
    pub(super) hidden_meta_parse: HiddenMetaParseState,
    pub(super) internal_tool_call_idx: usize,
    pub(super) internal_tool_call_streamer: InternalToolCallStreamer,
    pub(super) hermes_tool_call_streamer: HermesXmlToolCallStreamer,
    pub(super) anthropic_tool_call_streamer: AnthropicXmlToolCallStreamer,
    pub(super) bare_xml_tool_call_streamer: BareXmlToolCallStreamer,
}

impl StreamContentState {
    fn new() -> Self {
        Self {
            thinking_open: false,
            empty_choice_chunks: 0,
            finish_reason_seen: false,
            finish_reason_value: None,
            dropped_malformed_tool_call: false,
            saw_reasoning_output: false,
            tool_calls_map: SkipMap::default(),
            assistant_text: String::new(),
            hidden_meta: String::new(),
            reasoning_text: String::new(),
            reasoning_items: Vec::new(),
            hidden_meta_parse: HiddenMetaParseState::default(),
            internal_tool_call_idx: 0,
            internal_tool_call_streamer: InternalToolCallStreamer::new(),
            hermes_tool_call_streamer: HermesXmlToolCallStreamer::new(),
            anthropic_tool_call_streamer: AnthropicXmlToolCallStreamer::new(),
            bare_xml_tool_call_streamer: BareXmlToolCallStreamer::new(),
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

pub(in crate::ai) enum ParsedStreamPayload {
    Ignore,
    Done,
    Chunk(StreamChunk),
    SnapshotChunk(StreamChunk),
    /// Responses 协议返回的完整 `reasoning` output item（含 `id` /
    /// `encrypted_content` / `summary`）。用于同 turn 工具链回放：原样透传给
    /// 后续请求的 input，使模型保留上一跳推理上下文。不进持久化历史。
    ReasoningItem(serde_json::Value),
    /// provider 在流中途返回了 error 对象或 error 事件，携带可读错误信息。
    Error(String),
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
