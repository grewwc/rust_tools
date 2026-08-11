use std::collections::VecDeque;

use rust_tools::cw::SkipMap;

use crate::ai::{
    request::{StreamChunk, DIGEST_BEGIN, DIGEST_END},
    types::{FunctionCall, StreamResult, ToolCall},
};

use super::{
    MarkdownStreamRenderer,
    inline_recovery::InlineMarkupNormalizer,
    splitter::{
        AnthropicXmlToolCallStreamer, BareXmlToolCallStreamer, HermesXmlToolCallStreamer,
        InternalToolCallStreamer, StreamSplitter,
    },
    think_demux::ContentThinkDemuxer,
};

/// 流文本协议标记（嵌入 history / assistant_text，终端展示时会再映射为
/// header/footer 标签，见 `ThinkingFoldState` 与 markdown 渲染器）。
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
    pub(super) waiting_hint_tool_call: bool,
    pub(super) printed_tool_calls_header: bool,
    pub(super) current_printing_index: Option<usize>,
    pub(super) terminal_dedupe: Option<TerminalDedupeState>,
    pub(super) terminal_splitter: StreamSplitter,
    pub(super) thinking_fold: ThinkingFoldState,
    pub(super) subagent_fold: ThinkingFoldState,
    /// 终端展示专用：剥离 `<<<IMAGE_DIGEST>>> ... <<<END_IMAGE_DIGEST>>>` 区间
    /// （模型可见文本 / 历史不受影响），并处理哨兵被跨 chunk 拆散的情况。
    pub(super) digest_filter: DigestTerminalFilter,
}

impl StreamRenderState {
    fn new() -> Self {
        Self {
            markdown: MarkdownStreamRenderer::new(),
            waiting_hint_active: false,
            waiting_hint_buffering: false,
            waiting_hint_tool_call: false,
            printed_tool_calls_header: false,
            current_printing_index: None,
            terminal_dedupe: None,
            terminal_splitter: StreamSplitter::new(),
            thinking_fold: ThinkingFoldState::new(),
            subagent_fold: ThinkingFoldState::new_with_labels("subagent", "done subagent", false),
            digest_filter: DigestTerminalFilter::new(),
        }
    }
}

/// 终端展示用的图片摘要区域过滤器：把 digest 区间从**终端输出**里剥离，
/// 跨 chunk 正确处理哨兵被拆散的情况；模型可见文本不走这里。
pub(super) struct DigestTerminalFilter {
    /// 是否已越过 BEGIN 哨兵（正在 digest 区间内）。
    in_digest: bool,
    /// 尾部暂存：等待确认是否构成哨兵前缀（最多保留最长哨兵长度 - 1）。
    pending: String,
    /// 已确认进入 digest 区间后暂存的文本（仅在流结束时仍未闭合才回退输出）。
    suppressed: String,
}

impl DigestTerminalFilter {
    fn new() -> Self {
        Self {
            in_digest: false,
            pending: String::new(),
            suppressed: String::new(),
        }
    }

    /// 输入一段流式内容，返回其中**可以写终端**的部分。
    pub(super) fn push(&mut self, content: &str) -> String {
        let mut out = String::with_capacity(content.len());
        self.pending.push_str(content);
        loop {
            let target = if self.in_digest { DIGEST_END } else { DIGEST_BEGIN };
            let Some(idx) = self.pending.find(target) else { break };
            if self.in_digest {
                // 丢弃 digest 区间内容（含 END 哨兵），回到普通状态
                self.in_digest = false;
                self.suppressed.clear();
                self.pending.drain(..idx + target.len());
            } else {
                // 输出 BEGIN 之前的文本，进入 digest 区间
                out.push_str(&self.pending[..idx]);
                self.in_digest = true;
                self.pending.drain(..idx + target.len());
            }
        }
        if self.in_digest {
            // digest 区间内只保留确实可能组成 END 的后缀，其余移入 suppressed。
            let hold_len = marker_prefix_suffix_len(&self.pending, DIGEST_END);
            let commit_len = self.pending.len() - hold_len;
            if commit_len > 0 {
                let committed: String = self.pending.drain(..commit_len).collect();
                self.suppressed.push_str(&committed);
            }
        } else {
            // 普通状态只保留确实可能组成 BEGIN 的后缀。普通正文必须立即放行，
            // 不能固定扣留尾巴后在 flush 时绕过去重/样式管线。
            let hold_len = marker_prefix_suffix_len(&self.pending, DIGEST_BEGIN);
            let emit_len = self.pending.len() - hold_len;
            out.push_str(&self.pending[..emit_len]);
            self.pending.drain(..emit_len);
        }
        out
    }

    /// 流结束时冲刷：若 digest 区间从未闭合，把暂存内容回退输出
    /// （宁可展示完整叙述，也不静默丢内容）；哨兵前缀尾巴也一并输出。
    pub(super) fn flush(&mut self) -> String {
        let mut out = String::new();
        if self.in_digest {
            out.push_str(&std::mem::take(&mut self.suppressed));
            self.in_digest = false;
        }
        out.push_str(&std::mem::take(&mut self.pending));
        out
    }
}

/// 返回 `text` 末尾与 `marker` 前缀重合的最长字节数。
fn marker_prefix_suffix_len(text: &str, marker: &str) -> usize {
    let max_len = text.len().min(marker.len().saturating_sub(1));
    for len in (1..=max_len).rev() {
        let start = text.len() - len;
        if text.is_char_boundary(start) && marker.starts_with(&text[start..]) {
            return len;
        }
    }
    0
}

/// Thinking 折叠状态：维护一个滚动窗口，只在终端展示最近 N 条正文物理行，
/// 旧内容被折叠起来，同时保持流式实时输出。
pub(super) struct ThinkingFoldState {
    /// 最大可见正文物理行数（不含单行折叠提示）
    pub(super) max_visible_lines: usize,
    /// 已完成的 thinking 逻辑行（ring buffer，只保留最近 max_visible_lines 个候选行）
    pub(super) recent_lines: VecDeque<String>,
    /// 当前正在流式输出的不完整行
    pub(super) current_line: String,
    /// 总完成行数（含已被折叠的）
    pub(super) total_lines: usize,
    /// 当前折叠窗口（仅正文，不含 header）占用的 terminal 物理行数。光标停在正文
    /// 最后一行，而不是窗口下方的空白行；重画时只需向上移动 `window_rows - 1`。
    pub(super) window_rows: usize,
    /// 上次真正写到 terminal 的正文纯文本物理行（含缩进/包裹，不含 ANSI / header），用于在
    /// terminal resize 后按**当前**列宽重算旧窗口会占多少物理行，避免 cursor-up 擦不干净。
    pub(super) rendered_body_lines: Vec<String>,
    /// 重画正文时额外保留的右侧列数。xterm.js 在最后一列使用 delayed-wrap，若正文恰好
    /// 写满整行，下一次换行可能多占一个未计入 cursor-up 的物理行并把旧帧推入 scrollback。
    pub(super) rewrite_right_margin_cols: usize,
    /// 是否处于活跃的 thinking 折叠模式
    pub(super) active: bool,
    /// header（`○ thinking`）是否已落地。流式重画绝不随正文一起擦除/重画；收尾时才会
    /// 在原位将它改为 `✓ thinking`。这样即便正文擦除失步也无法再生出第二个 header，
    /// 从根上杜绝「孤儿 header 叠加」的渲染 bug。
    pub(super) header_drawn: bool,
    /// 折叠块 header 文案（如 `○ thinking` / `subagent explore`）。
    pub(super) header_label: String,
    /// 折叠块 footer 文案（如 `✓ thinking` / `done subagent explore`）。
    pub(super) footer_label: String,
    /// 是否在折叠窗口里跳过空白行。thinking 适合紧凑展示；subagent 正文保持原样。
    pub(super) skip_blank_lines: bool,
}

impl ThinkingFoldState {
    pub(super) fn new() -> Self {
        // thinking 过程用 `○ thinking` 标识进行中，结束后用 `✓ thinking` 收口
        // （对勾代替 "done"），避免「thinking / done thinking」成对文字的冗余。
        Self::new_with_labels("○ thinking", "✓ thinking", true)
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
            rewrite_right_margin_cols: 0,
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
    /// 未收到 finish_reason 前，响应流连续无有效进展并触发 idle timeout。
    /// 这属于传输中断，不能把未确认完整的工具调用交给执行层。
    pub(super) stream_idle_timed_out: bool,
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
    /// content 通道已消费过的原始文本（先于 think demux）。Responses 兼容网关会把
    /// `output_text.delta` 已经流过的 part 再通过 `content_part.added` 全量重发；
    /// 这里按原始 content 去重，避免 demux 关闭后把完整 `<think>...</think>正文`
    /// 当成新正文再次追加。
    pub(super) content_replay_text: String,
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
    /// 有状态命名空间 marker 归一化器：跨 chunk 复原被截断的 `<｜｜DSML｜｜…>`。
    pub(super) inline_markup_normalizer: InlineMarkupNormalizer,
    /// 把内联在 content 通道里的推理链（预填 `<think>` 模板）用悬空 `</think>`
    /// 拆回 reasoning。默认直通（未 arm 的模型零影响）；仅对声明
    /// `reasoning_in_content` 的模型在 `stream_response` 里 arm。
    pub(super) content_think_demuxer: ContentThinkDemuxer,
}

impl StreamContentState {
    fn new() -> Self {
        Self {
            thinking_open: false,
            empty_choice_chunks: 0,
            finish_reason_seen: false,
            stream_idle_timed_out: false,
            finish_reason_value: None,
            dropped_malformed_tool_call: false,
            saw_reasoning_output: false,
            tool_calls_map: SkipMap::default(),
            assistant_text: String::new(),
            content_replay_text: String::new(),
            hidden_meta: String::new(),
            reasoning_text: String::new(),
            reasoning_items: Vec::new(),
            hidden_meta_parse: HiddenMetaParseState::default(),
            internal_tool_call_idx: 0,
            internal_tool_call_streamer: InternalToolCallStreamer::new(),
            hermes_tool_call_streamer: HermesXmlToolCallStreamer::new(),
            anthropic_tool_call_streamer: AnthropicXmlToolCallStreamer::new(),
            bare_xml_tool_call_streamer: BareXmlToolCallStreamer::new(),
            inline_markup_normalizer: InlineMarkupNormalizer::new(),
            content_think_demuxer: ContentThinkDemuxer::new(),
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
    Continue { meaningful_progress: bool },
    Stop,
    Return(StreamResult),
}

pub(in crate::ai) enum ParsedStreamPayload {
    Ignore,
    Done,
    Chunk(StreamChunk),
    /// content_part.added（output_text 类型）携带的是该 part 当前已存在的完整文本，
    /// 与 output_text.delta 增量重叠，属于协议多路径重发而非模型新增内容。
    /// 按增量格式解析（think_demux 拆分等仍生效），但流层会对 content 额外做
    /// 未见后缀去重，避免正文跨事件路径重复渲染。
    ReplayedChunk(StreamChunk),
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

#[cfg(test)]
mod tests {
    use super::DigestTerminalFilter;
    use crate::ai::request::{DIGEST_BEGIN, DIGEST_END};

    #[test]
    fn digest_filter_strips_region_across_chunk_boundaries() {
        let mut f = DigestTerminalFilter::new();
        let mut out = String::new();
        // 普通叙述立即透出。
        out.push_str(&f.push("我先看一下界面。"));
        // BEGIN 哨兵被拆散到两个 chunk
        out.push_str(&f.push(&DIGEST_BEGIN[..10]));
        out.push_str(&f.push(&DIGEST_BEGIN[10..]));
        // digest 区间内容被吞掉
        out.push_str(&f.push("界面上有一个搜索框，右下角是按钮"));
        // END 哨兵被拆散
        out.push_str(&f.push(&DIGEST_END[..8]));
        out.push_str(&f.push(&DIGEST_END[8..]));
        // 后续叙述恢复透出
        out.push_str(&f.push("接下来我操作一下。"));
        out.push_str(&f.flush());
        assert_eq!(out, "我先看一下界面。接下来我操作一下。");
    }

    #[test]
    fn digest_filter_flush_recovers_unclosed_region() {
        let mut f = DigestTerminalFilter::new();
        let mut out = String::new();
        out.push_str(&f.push("叙述开始"));
        out.push_str(&f.push(DIGEST_BEGIN));
        let body = "被截断的摘要正文很长，必须保持原始顺序，不能把尾部移到开头。";
        out.push_str(&f.push(body));
        // 流结束仍未闭合：回退输出暂存内容，避免静默丢叙述
        out.push_str(&f.flush());
        assert_eq!(out, format!("叙述开始{body}"));
        assert_eq!(f.flush(), "");
    }

    #[test]
    fn digest_filter_multiple_regions_and_adjacent_text() {
        let mut f = DigestTerminalFilter::new();
        let text = format!(
            "a{DIGEST_BEGIN}1{DIGEST_END}b{DIGEST_BEGIN}2{DIGEST_END}c"
        );
        let mut out = f.push(&text);
        out.push_str(&f.flush());
        assert_eq!(out, "abc");
    }

    #[test]
    fn digest_filter_keeps_partial_sentinel_tail_until_flush() {
        let mut f = DigestTerminalFilter::new();
        // 流在普通文本中结束：尾部可能是哨兵前缀的部分在 flush 时透出
        assert_eq!(f.push("结尾写着 <<<IMAGE_"), "结尾写着 ");
        assert_eq!(f.flush(), "<<<IMAGE_");
    }
}
