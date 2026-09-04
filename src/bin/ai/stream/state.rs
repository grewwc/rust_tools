use std::collections::VecDeque;

use rust_tools::cw::SkipMap;

pub(in crate::ai) use crate::ai::request::ParsedStreamPayload;
use crate::ai::{
    ports::stream::FilterChain,
    request::{DIGEST_BEGIN, DIGEST_END},
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

/// Stream text protocol markers (embedded into history / assistant_text, and re-mapped to
/// header/footer labels at terminal display time; see `ThinkingFoldState` and the markdown renderer).
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
    /// Pluggable stream filter chain (Step 6: applied at the visible-content commit point of
    /// `process_stream_payload`; empty chain = pass-through, zero behavior change).
    pub(super) filters: FilterChain,
    /// Last-seen `(echoed_model, usage)` from any chunk during this stream.
    /// Handed to the kernel's `/dev/llm` when the stream finalizes.
    pub(super) pending_llm_usage: Option<(String, super::super::request::StreamUsage)>,
}

impl StreamProcessingState {
    pub(super) fn new() -> Self {
        Self::with_filters(FilterChain::new())
    }

    pub(super) fn with_filters(filters: FilterChain) -> Self {
        Self {
            framing: StreamFramingState::new(),
            render: StreamRenderState::new(),
            content: StreamContentState::new(),
            filters,
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
    /// Withhold non-thinking assistant prose until the stream outcome and the
    /// driver's final-response gates are known.
    pub(super) defer_assistant_body: bool,
    pub(super) waiting_hint_active: bool,
    pub(super) waiting_hint_buffering: bool,
    pub(super) waiting_hint_tool_call: bool,
    pub(super) printed_tool_calls_header: bool,
    pub(super) current_printing_index: Option<usize>,
    pub(super) terminal_dedupe: Option<TerminalDedupeState>,
    pub(super) terminal_splitter: StreamSplitter,
    pub(super) thinking_fold: ThinkingFoldState,
    pub(super) subagent_fold: ThinkingFoldState,
    /// Terminal display only: strips `<<<IMAGE_DIGEST>>> ... <<<END_IMAGE_DIGEST>>>` ranges
    /// (model-visible text / history are unaffected), and handles sentinels split across chunks.
    pub(super) digest_filter: DigestTerminalFilter,
}

impl StreamRenderState {
    fn new() -> Self {
        Self {
            markdown: MarkdownStreamRenderer::new(),
            defer_assistant_body: false,
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

/// Image-digest range filter for terminal display: strips digest ranges from **terminal output**,
/// correctly handling sentinels split across chunks; model-visible text does not go through here.
pub(super) struct DigestTerminalFilter {
    /// Whether the BEGIN sentinel has been passed (currently inside the digest range).
    in_digest: bool,
    /// Pending tail: held while confirming whether it forms a sentinel prefix (at most longest sentinel length - 1).
    pending: String,
    /// Text staged after the digest range is confirmed entered (only flushed back out if the stream ends while still unclosed).
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

    /// Feeds in a chunk of streamed content and returns the part that **can be written to the terminal**.
    pub(super) fn push(&mut self, content: &str) -> String {
        let mut out = String::with_capacity(content.len());
        self.pending.push_str(content);
        loop {
            let target = if self.in_digest {
                DIGEST_END
            } else {
                DIGEST_BEGIN
            };
            let Some(idx) = self.pending.find(target) else {
                break;
            };
            if self.in_digest {
                // Drop the digest range content (including the END sentinel) and return to normal state
                self.in_digest = false;
                self.suppressed.clear();
                self.pending.drain(..idx + target.len());
            } else {
                // Emit the text before BEGIN and enter the digest range
                out.push_str(&self.pending[..idx]);
                self.in_digest = true;
                self.pending.drain(..idx + target.len());
            }
        }
        if self.in_digest {
            // Inside the digest range, keep only the suffix that could genuinely form END; move the rest into suppressed.
            let hold_len = marker_prefix_suffix_len(&self.pending, DIGEST_END);
            let commit_len = self.pending.len() - hold_len;
            if commit_len > 0 {
                let committed: String = self.pending.drain(..commit_len).collect();
                self.suppressed.push_str(&committed);
            }
        } else {
            // Normal state keeps only the suffix that could genuinely form BEGIN. Ordinary body text must pass through immediately,
            // not be held back as a fixed tail only to bypass the dedup/style pipeline at flush time.
            let hold_len = marker_prefix_suffix_len(&self.pending, DIGEST_BEGIN);
            let emit_len = self.pending.len() - hold_len;
            out.push_str(&self.pending[..emit_len]);
            self.pending.drain(..emit_len);
        }
        out
    }

    /// Flush at stream end: if the digest range was never closed, flush the staged content back out
    /// (prefer showing the full narration over silently dropping content); the sentinel-prefix tail is emitted too.
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

/// Returns the longest number of bytes at the end of `text` that overlap the prefix of `marker`.
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

/// Thinking fold state: maintains a rolling window so only the most recent N body physical lines are shown in the terminal,
/// older content is folded away, while streaming output stays real-time.
pub(super) struct ThinkingFoldState {
    /// Maximum visible body physical lines (excluding the one-line fold hint)
    pub(super) max_visible_lines: usize,
    /// Completed thinking logical lines (ring buffer holding only the most recent max_visible_lines candidate lines)
    pub(super) recent_lines: VecDeque<String>,
    /// The incomplete line currently being streamed
    pub(super) current_line: String,
    /// Total completed line count (including folded lines)
    pub(super) total_lines: usize,
    /// Number of terminal physical rows occupied by the current fold window (body only, excluding the header). The cursor sits on the last
    /// body line, not on the blank line below the window; redrawing only needs to move up `window_rows - 1`.
    pub(super) window_rows: usize,
    /// Body plain-text physical lines actually written to the terminal last time (including indentation/wrapping, excluding ANSI / header), used to
    /// recompute how many physical rows the old window occupies at the **current** column width after a terminal resize, so cursor-up leaves no residue.
    pub(super) rendered_body_lines: Vec<String>,
    /// Extra right-side columns reserved when redrawing the body. xterm.js uses delayed-wrap on the last column; if the body
    /// exactly fills a line, the next line break may occupy an extra physical row not counted by cursor-up and push the old frame into scrollback.
    pub(super) rewrite_right_margin_cols: usize,
    /// Whether active thinking fold mode is in effect
    pub(super) active: bool,
    /// Whether the header (`○ thinking`) has been laid down. Streaming redraws never erase/repaint it along with the body; only at teardown is it
    /// changed in place to `✓ thinking`. This way, even if body erasure drifts out of sync, a second header can never appear,
    /// eliminating the "orphan header stacking" rendering bug at its root.
    pub(super) header_drawn: bool,
    /// Fold-block header text (e.g. `○ thinking` / `subagent explore`).
    pub(super) header_label: String,
    /// Fold-block footer text (e.g. `✓ thinking` / `done subagent explore`).
    pub(super) footer_label: String,
    /// Whether to skip blank lines inside the fold window. Thinking suits compact display; subagent body text stays as-is.
    pub(super) skip_blank_lines: bool,
}

impl ThinkingFoldState {
    pub(super) fn new() -> Self {
        // The thinking phase is marked in progress with `○ thinking` and closed out with `✓ thinking`
        // (a checkmark instead of "done"), avoiding the redundant "thinking / done thinking" pair of words.
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
    /// Before finish_reason was received, the response stream made no valid progress for a sustained period and hit the idle timeout.
    /// This is a transport interruption; unconfirmed-incomplete tool calls must not be handed to the execution layer.
    pub(super) stream_idle_timed_out: bool,
    /// The exact value of the most recent non-empty `finish_reason` (e.g. `stop` / `length` / `tool_calls`).
    /// `length` means the server truncated due to the output limit — an earlier, more accurate truncation
    /// signal than "tool JSON parse failure", used to upgrade this turn's outcome to retryable `Truncated`.
    pub(super) finish_reason_value: Option<String>,
    /// Whether this turn had tool calls dropped due to incomplete arguments JSON. Typical case: a large `write_file`
    /// hits the output limit and gets truncated: half a JSON → dropped → the turn has no valid tool calls.
    /// Judging only by "no tool calls + some text" would misread it as normal completion and end silently.
    pub(super) dropped_malformed_tool_call: bool,
    /// Accumulated tool-call arguments exceeded the `MAX_TOOL_ARG_BYTES` cap and the stream was force-stopped. Same reasoning as
    /// `stream_idle_timed_out`: the stream was cut off by the runtime and the model may still be generating,
    /// so JSON that merely happens to be valid at the cut-off moment must not be handed to the execution layer as a complete tool call.
    pub(super) tool_args_cap_exceeded: bool,
    pub(super) saw_reasoning_output: bool,
    pub(super) tool_calls_map: SkipMap<usize, ToolCallBuilder>,
    /// Composite key resolved for the most recent tool call without an `index`,
    /// used to attach later parameter-continuation deltas (which have neither id
    /// nor index) onto that same tool call.
    pub(super) last_indexless_tool_call_key: Option<usize>,
    pub(super) assistant_text: String,
    /// Raw text already consumed on the content channel (prior to think demux).
    /// Responses-compatible gateways re-send parts already streamed via
    /// `output_text.delta` through `content_part.added` in full; dedupe by raw
    /// content here so a complete `thinking...` response body is not appended
    /// again as new text once demux is closed.
    pub(super) content_replay_text: String,
    pub(super) hidden_meta: String,
    /// Raw `reasoning_content` returned by the model (without the display-only
    /// `thinking` markers), forwarded to history via StreamResult at the end of
    /// the turn so the next turn can send it back verbatim to the backend
    /// (required for DeepSeek thinking-mode).
    pub(super) reasoning_text: String,
    /// Complete `reasoning` output items captured from the Responses stream this
    /// turn (including encrypted_content). Only used for same-turn tool-chain
    /// replay; never persisted to history.
    pub(super) reasoning_items: Vec<serde_json::Value>,
    pub(super) hidden_meta_parse: HiddenMetaParseState,
    pub(super) internal_tool_call_idx: usize,
    pub(super) internal_tool_call_streamer: InternalToolCallStreamer,
    pub(super) hermes_tool_call_streamer: HermesXmlToolCallStreamer,
    pub(super) anthropic_tool_call_streamer: AnthropicXmlToolCallStreamer,
    pub(super) bare_xml_tool_call_streamer: BareXmlToolCallStreamer,
    /// Stateful namespace marker normalizer: reassembles a truncated `<｜｜DSML｜｜…>` across chunks.
    pub(super) inline_markup_normalizer: InlineMarkupNormalizer,
    /// Splits reasoning chains inlined in the content channel (pre-filled `<think>` template) back out into reasoning using a dangling `</think>`.
    /// Pass-through by default (zero impact on models that are not armed); armed only in `stream_response` for models declaring
    /// `reasoning_in_content`.
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
            tool_args_cap_exceeded: false,
            saw_reasoning_output: false,
            tool_calls_map: SkipMap::default(),
            last_indexless_tool_call_key: None,
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
            // Some providers do not return the type field in stream deltas; default to "function"
            // to satisfy the OpenAI protocol and avoid a 400 error from sending "type":"".
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
        // Ordinary narration passes through immediately.
        out.push_str(&f.push("我先看一下界面。"));
        // BEGIN sentinel split across two chunks
        out.push_str(&f.push(&DIGEST_BEGIN[..10]));
        out.push_str(&f.push(&DIGEST_BEGIN[10..]));
        // digest range content is swallowed
        out.push_str(&f.push("界面上有一个搜索框，右下角是按钮"));
        // END sentinel split apart
        out.push_str(&f.push(&DIGEST_END[..8]));
        out.push_str(&f.push(&DIGEST_END[8..]));
        // subsequent narration passes through again
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
        // Stream ended while still unclosed: flush the staged content back out to avoid silently losing narration
        out.push_str(&f.flush());
        assert_eq!(out, format!("叙述开始{body}"));
        assert_eq!(f.flush(), "");
    }

    #[test]
    fn digest_filter_multiple_regions_and_adjacent_text() {
        let mut f = DigestTerminalFilter::new();
        let text = format!("a{DIGEST_BEGIN}1{DIGEST_END}b{DIGEST_BEGIN}2{DIGEST_END}c");
        let mut out = f.push(&text);
        out.push_str(&f.flush());
        assert_eq!(out, "abc");
    }

    #[test]
    fn digest_filter_keeps_partial_sentinel_tail_until_flush() {
        let mut f = DigestTerminalFilter::new();
        // Stream ends inside ordinary text: a tail that may be part of a sentinel prefix is emitted at flush
        assert_eq!(f.push("结尾写着 <<<IMAGE_"), "结尾写着 ");
        assert_eq!(f.flush(), "<<<IMAGE_");
    }
}
