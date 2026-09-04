use super::state::InternalToolCall;

const TOOL_CALL_BEGIN_MARKER: &str = "<|tool_call_begin|>";
const TOOL_CALL_ARGS_MARKER: &str = "<|tool_call_args|>";
const TOOL_CALL_END_MARKER: &str = "<|tool_call_end|>";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum InternalToolCallStreamEvent {
    Begin(String),
    Args(String),
    End,
    /// System-internal tool-protocol markers the model emits in visible body
    /// text (e.g. `<function_results>`). The system itself never emits such
    /// results — the deterministic fingerprint of a degenerate self-play loop. The
    /// streamer strips the whole block from display; this event tells upper layers
    /// the turn is degraded, so stop streaming and downgrade-retry (reuse the degenerate_repetition path) instead of persisting hallucinated text that would poison the next request. Unlike text-statistics thresholds this is a zero-false-positive signal: legitimate repeated code or wording never contains this marker.
    HallucinatedProtocolMarker,
}

#[derive(Default)]
enum InternalToolCallStreamerPhase {
    #[default]
    Idle,
    AwaitingName,
    StreamingArgs,
    SkipUntilEnd,
}

#[derive(Default)]
pub(super) struct InternalToolCallStreamer {
    pending: String,
    phase: InternalToolCallStreamerPhase,
}

impl InternalToolCallStreamer {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn push(&mut self, chunk: &str) -> (String, Vec<InternalToolCallStreamEvent>) {
        self.pending.push_str(chunk);
        let mut cleaned = String::new();
        let mut events = Vec::new();

        loop {
            match &self.phase {
                InternalToolCallStreamerPhase::Idle => {
                    if let Some(pos) = self.pending.find(TOOL_CALL_BEGIN_MARKER) {
                        cleaned.push_str(&self.pending[..pos]);
                        let after = pos + TOOL_CALL_BEGIN_MARKER.len();
                        self.pending.drain(..after);
                        self.phase = InternalToolCallStreamerPhase::AwaitingName;
                        continue;
                    }
                    let keep =
                        longest_marker_suffix_prefix(&self.pending, &[TOOL_CALL_BEGIN_MARKER]);
                    let emit_len = self.pending.len().saturating_sub(keep);
                    if emit_len > 0 {
                        cleaned.push_str(&self.pending[..emit_len]);
                        self.pending.drain(..emit_len);
                    }
                    break;
                }
                InternalToolCallStreamerPhase::AwaitingName => {
                    let candidates = [
                        TOOL_CALL_ARGS_MARKER,
                        TOOL_CALL_END_MARKER,
                        TOOL_CALL_BEGIN_MARKER,
                    ];
                    let brace_pos = self.pending.find('{');
                    let marker_hit = earliest_substring_match(&self.pending, &candidates);
                    let boundary = match (brace_pos, marker_hit) {
                        (Some(b), Some((m_pos, m_idx, m_len))) => {
                            if b <= m_pos {
                                Some(BoundaryHit::Brace(b))
                            } else {
                                Some(BoundaryHit::Marker {
                                    pos: m_pos,
                                    marker: candidates[m_idx],
                                    len: m_len,
                                })
                            }
                        }
                        (Some(b), None) => Some(BoundaryHit::Brace(b)),
                        (None, Some((m_pos, m_idx, m_len))) => Some(BoundaryHit::Marker {
                            pos: m_pos,
                            marker: candidates[m_idx],
                            len: m_len,
                        }),
                        (None, None) => None,
                    };

                    match boundary {
                        Some(BoundaryHit::Brace(pos)) => {
                            let raw_before = self.pending[..pos].to_string();
                            self.pending.drain(..pos);
                            let name = sanitize_internal_tool_call_name(&raw_before);
                            if name.is_empty() {
                                self.phase = InternalToolCallStreamerPhase::SkipUntilEnd;
                            } else {
                                events.push(InternalToolCallStreamEvent::Begin(name));
                                self.phase = InternalToolCallStreamerPhase::StreamingArgs;
                            }
                            continue;
                        }
                        Some(BoundaryHit::Marker { pos, marker, len })
                            if marker == TOOL_CALL_ARGS_MARKER =>
                        {
                            let raw_before = self.pending[..pos].to_string();
                            let after = pos + len;
                            self.pending.drain(..after);
                            let name = sanitize_internal_tool_call_name(&raw_before);
                            if name.is_empty() {
                                self.phase = InternalToolCallStreamerPhase::SkipUntilEnd;
                            } else {
                                events.push(InternalToolCallStreamEvent::Begin(name));
                                self.phase = InternalToolCallStreamerPhase::StreamingArgs;
                            }
                            continue;
                        }
                        Some(BoundaryHit::Marker { pos, marker, len })
                            if marker == TOOL_CALL_END_MARKER =>
                        {
                            let raw_before = self.pending[..pos].to_string();
                            let after = pos + len;
                            self.pending.drain(..after);
                            let name = sanitize_internal_tool_call_name(&raw_before);
                            if !name.is_empty() {
                                events.push(InternalToolCallStreamEvent::Begin(name));
                                events.push(InternalToolCallStreamEvent::End);
                            }
                            self.phase = InternalToolCallStreamerPhase::Idle;
                            continue;
                        }
                        Some(BoundaryHit::Marker { pos, len, .. }) => {
                            let after = pos + len;
                            self.pending.drain(..after);
                            self.phase = InternalToolCallStreamerPhase::AwaitingName;
                            continue;
                        }
                        None => {
                            let keep = longest_marker_suffix_prefix(&self.pending, &candidates);
                            let _ = keep;
                            break;
                        }
                    }
                }
                InternalToolCallStreamerPhase::StreamingArgs => {
                    if let Some(pos) = self.pending.find(TOOL_CALL_END_MARKER) {
                        if pos > 0 {
                            let chunk = self.pending[..pos].to_string();
                            events.push(InternalToolCallStreamEvent::Args(chunk));
                        }
                        let after = pos + TOOL_CALL_END_MARKER.len();
                        self.pending.drain(..after);
                        events.push(InternalToolCallStreamEvent::End);
                        self.phase = InternalToolCallStreamerPhase::Idle;
                        continue;
                    }
                    let keep = longest_marker_suffix_prefix(&self.pending, &[TOOL_CALL_END_MARKER]);
                    let emit_len = self.pending.len().saturating_sub(keep);
                    if emit_len > 0 {
                        let chunk = self.pending[..emit_len].to_string();
                        self.pending.drain(..emit_len);
                        events.push(InternalToolCallStreamEvent::Args(chunk));
                    }
                    break;
                }
                InternalToolCallStreamerPhase::SkipUntilEnd => {
                    if let Some(pos) = self.pending.find(TOOL_CALL_END_MARKER) {
                        let after = pos + TOOL_CALL_END_MARKER.len();
                        self.pending.drain(..after);
                        self.phase = InternalToolCallStreamerPhase::Idle;
                        continue;
                    }
                    let keep = longest_marker_suffix_prefix(&self.pending, &[TOOL_CALL_END_MARKER]);
                    let emit_len = self.pending.len().saturating_sub(keep);
                    if emit_len > 0 {
                        self.pending.drain(..emit_len);
                    }
                    break;
                }
            }
        }

        (cleaned, events)
    }
}

enum BoundaryHit {
    Brace(usize),
    Marker {
        pos: usize,
        marker: &'static str,
        len: usize,
    },
}

const FN_OPEN_MARKER: &str = "<function=";
const FN_CLOSE_MARKER: &str = "</function>";
const TC_OPEN_MARKER: &str = "<tool_call>";
const TC_CLOSE_MARKER: &str = "</tool_call>";

#[derive(Default)]
enum HermesXmlPhase {
    #[default]
    Idle,
    /// Consumed `<function=`, waiting for the `>` after the function name.
    AwaitingName,
    /// Function name captured; buffering the body until `</function>` (nothing is echoed meanwhile).
    InBody { name: String },
}

/// Suppress Hermes / Qwen-style XML tool calls (`<function=NAME>...</function>`,
/// possibly wrapped in `<tool_call>`) mid-stream: strip the markup from the visible output
/// and convert it on the fly into the same Begin/Args/End events as `<|tool_call_begin|>` for the unified pipeline to render.
/// This way the terminal never flashes the raw `<function=...>` markup when the model calls a tool.
#[derive(Default)]
pub(super) struct HermesXmlToolCallStreamer {
    pending: String,
    phase: HermesXmlPhase,
}

impl HermesXmlToolCallStreamer {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn push(&mut self, chunk: &str) -> (String, Vec<InternalToolCallStreamEvent>) {
        self.pending.push_str(chunk);
        let mut cleaned = String::new();
        let mut events = Vec::new();

        loop {
            match &self.phase {
                HermesXmlPhase::Idle => {
                    let candidates = [FN_OPEN_MARKER, TC_OPEN_MARKER, TC_CLOSE_MARKER];
                    match earliest_substring_match(&self.pending, &candidates) {
                        Some((pos, idx, len)) => {
                            // Content before the marker is normal visible text; but trailing whitespace
                            // right before the marker is wrapper/call noise — trim it to avoid extra blank lines.
                            let before =
                                self.pending[..pos].trim_end_matches([' ', '\t', '\r', '\n']);
                            cleaned.push_str(before);
                            let after = pos + len;
                            self.pending.drain(..after);
                            if candidates[idx] == FN_OPEN_MARKER {
                                self.phase = HermesXmlPhase::AwaitingName;
                            }
                            // `<tool_call>` / `</tool_call>` are wrapper markers only; suppress and continue.
                            continue;
                        }
                        None => {
                            // Keep only the tail that could still be a marker prefix; emit the rest safely.
                            let mut keep = longest_marker_suffix_prefix(&self.pending, &candidates);
                            // When holding a potential marker prefix, also hold the whitespace right
                            // before it, so a split like `<tool_call>\n<func` never flashes the middle `\n`
                            // first. The whitespace is only deferred by one frame; ordering is unchanged.
                            if keep > 0 {
                                let head = &self.pending[..self.pending.len() - keep];
                                let trimmed = head.trim_end_matches([' ', '\t', '\r', '\n']);
                                keep += head.len() - trimmed.len();
                            }
                            let emit_len = self.pending.len().saturating_sub(keep);
                            if emit_len > 0 {
                                cleaned.push_str(&self.pending[..emit_len]);
                                self.pending.drain(..emit_len);
                            }
                            break;
                        }
                    }
                }
                HermesXmlPhase::AwaitingName => {
                    if let Some(pos) = self.pending.find('>') {
                        let name = self.pending[..pos].trim().to_string();
                        self.pending.drain(..pos + 1);
                        self.phase = HermesXmlPhase::InBody { name };
                        continue;
                    }
                    // Function name not fully arrived yet: wait for later chunks (never echo a partial name).
                    break;
                }
                HermesXmlPhase::InBody { name } => {
                    if let Some(pos) = self.pending.find(FN_CLOSE_MARKER) {
                        let body = self.pending[..pos].to_string();
                        let after = pos + FN_CLOSE_MARKER.len();
                        self.pending.drain(..after);
                        let name = name.clone();
                        if !name.is_empty() {
                            let args = super::inline_recovery::parse_hermes_function_body(&body)
                                .unwrap_or_else(|| "{}".to_string());
                            events.push(InternalToolCallStreamEvent::Begin(name));
                            events.push(InternalToolCallStreamEvent::Args(args));
                            events.push(InternalToolCallStreamEvent::End);
                        }
                        self.phase = HermesXmlPhase::Idle;
                        continue;
                    }
                    // Body still unclosed: keep buffering everything (no echo) and wait for `</function>`.
                    break;
                }
            }
        }

        (cleaned, events)
    }
}

/// Anthropic / Claude-style XML tool calls:
/// ```text
/// <function_calls>
///   <invoke name="read_file">
///     <parameter name="path">/x</parameter>
///   </invoke>
/// </function_calls>
/// ```
/// Unlike the Hermes form (`<function=NAME>` / `<parameter=key>`), this one uses
/// a `name="..."` attribute, the outer wrapper tag is `function_calls` or `tool_calls`,
/// and tags may carry a namespace prefix (e.g. `antml:invoke`). Some models
/// (deepseek-v4-flash) emit tool calls in this format; unrecognized, they would be printed as plain text and the turn would be
/// judged Completed and end right away — showing up as "it suddenly stopped and the tool never ran".
#[derive(Default)]
enum AnthropicXmlPhase {
    #[default]
    Idle,
    InInvoke {
        name: String,
        params: serde_json::Map<String, serde_json::Value>,
    },
    InParamValue {
        name: String,
        params: serde_json::Map<String, serde_json::Value>,
        key: String,
        force_string: bool,
        value: String,
    },
    /// Swallowing a model-hallucinated "tool result" block until the matching closing tag or end of stream.
    /// Nothing inside the block (nested tags, hallucinated result text, prose) is echoed. Upper layers stop
    /// and downgrade-retry as soon as HallucinatedProtocolMarker arrives, so the block is usually swallowed only until that stop.
    InResultBlock,
}

#[derive(Default)]
pub(super) struct AnthropicXmlToolCallStreamer {
    pending: String,
    phase: AnthropicXmlPhase,
}

enum AnthropicTagClass {
    /// `function_calls` / `tool_calls` (open or close): wrapper only, suppress directly.
    Wrapper,
    InvokeOpen(String),
    InvokeClose,
    ParamOpen {
        key: String,
        /// `string="true"` attribute: the parameter value must always stay a string, even when it looks like a JSON scalar.
        force_string: bool,
    },
    ParamClose,
    /// System-internal "tool result" protocol markers (`function_results` / `tool_result` etc.).
    /// The system never emits these; seeing one means model hallucination. An open tag starts swallowing the whole block until the matching
    /// closing tag (or end of stream); none of the hallucinated "result" text inside is echoed.
    ResultBlockOpen,
    /// Closing tag of the result block above; ends the whole-block swallow.
    ResultBlockClose,
    /// Non-tool tags (`<...>` in ordinary prose): passed through as-is.
    Other,
}

/// Local names of model-hallucinated "tool result" protocol markers (no angle brackets or namespace prefix).
/// Real system tool results never reach the assistant's visible body in this inline XML form — the moment the model
/// emits one, it is the deterministic fingerprint of a fabricated "call → result" loop. An open tag triggers the whole-block swallow and
/// also reports HallucinatedProtocolMarker for upper layers to downgrade-retry. The list covers singular/plural and common hallucinated variants.
const HALLUCINATED_RESULT_TAG_NAMES: &[&str] = &[
    "function_results",
    "function_result",
    "tool_results",
    "tool_result",
];

impl AnthropicXmlToolCallStreamer {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn push(&mut self, chunk: &str) -> (String, Vec<InternalToolCallStreamEvent>) {
        self.pending.push_str(chunk);
        let mut cleaned = String::new();
        let mut events = Vec::new();

        loop {
            let phase_kind = match &self.phase {
                AnthropicXmlPhase::Idle => 0,
                AnthropicXmlPhase::InInvoke { .. } => 1,
                AnthropicXmlPhase::InParamValue { .. } => 2,
                AnthropicXmlPhase::InResultBlock => 3,
            };

            match phase_kind {
                0 => {
                    let Some(lt) = self.pending.find('<') else {
                        cleaned.push_str(&self.pending);
                        self.pending.clear();
                        break;
                    };
                    let before = self.pending[..lt].to_string();

                    let gt_rel = self.pending[lt..].find('>');
                    let Some(gt_rel) = gt_rel else {
                        // Tag not closed yet: decide whether this `<...` fragment could still become a tool/ordinary tag.
                        if could_be_tag_name_prefix(&self.pending[lt..]) {
                            // Looks like a tag name being written: emit the visible text before `<`, hold the rest.
                            if lt > 0 {
                                cleaned.push_str(&before);
                                self.pending.drain(..lt);
                            }
                            break;
                        }
                        // Not a tag (e.g. `<` in prose): emit `<` as an ordinary character.
                        cleaned.push_str(&before);
                        cleaned.push('<');
                        self.pending.drain(..lt + '<'.len_utf8());
                        continue;
                    };

                    let tag_start = lt;
                    let tag_end = lt + gt_rel; // points at '>'
                    let tag = self.pending[tag_start..=tag_end].to_string();
                    let class = classify_anthropic_tag(&tag);
                    let is_tool_tag = !matches!(class, AnthropicTagClass::Other);

                    if is_tool_tag {
                        // Trailing whitespace before a tool tag is layout noise; trim it to avoid extra blank lines.
                        let trimmed = before.trim_end_matches([' ', '\t', '\r', '\n']);
                        cleaned.push_str(trimmed);
                    } else {
                        cleaned.push_str(&before);
                        cleaned.push_str(&tag);
                    }
                    self.pending.drain(..=tag_end);

                    match class {
                        AnthropicTagClass::InvokeOpen(name) if !name.is_empty() => {
                            self.phase = AnthropicXmlPhase::InInvoke {
                                name,
                                params: serde_json::Map::new(),
                            };
                        }
                        AnthropicTagClass::ResultBlockOpen => {
                            // Model-hallucinated "tool result" block: start the whole-block swallow and report the degeneration fingerprint.
                            self.phase = AnthropicXmlPhase::InResultBlock;
                            events.push(InternalToolCallStreamEvent::HallucinatedProtocolMarker);
                        }
                        AnthropicTagClass::ResultBlockClose => {
                            // Orphan closing tag in Idle (repetition broke the pairing): the tag is already suppressed,
                            // report the fingerprint anyway — ordinary body text never contains `</function_results>`.
                            events.push(InternalToolCallStreamEvent::HallucinatedProtocolMarker);
                        }
                        // Wrapper / empty-name invoke / stray closing tag / Other: already handled, continue.
                        _ => {}
                    }
                    continue;
                }
                1 => {
                    let Some(lt) = self.pending.find('<') else {
                        // Whitespace/newlines between tags inside invoke are suppressed directly.
                        self.pending.clear();
                        break;
                    };
                    if lt > 0 {
                        self.pending.drain(..lt);
                    }
                    let Some(gt_rel) = self.pending.find('>') else {
                        break;
                    };
                    let tag = self.pending[..=gt_rel].to_string();
                    let class = classify_anthropic_tag(&tag);
                    self.pending.drain(..=gt_rel);
                    match class {
                        AnthropicTagClass::ParamOpen { key, force_string } => {
                            if let AnthropicXmlPhase::InInvoke { name, params } =
                                std::mem::take(&mut self.phase)
                            {
                                self.phase = AnthropicXmlPhase::InParamValue {
                                    name,
                                    params,
                                    key,
                                    force_string,
                                    value: String::new(),
                                };
                            }
                        }
                        AnthropicTagClass::InvokeClose => {
                            if let AnthropicXmlPhase::InInvoke { name, params } =
                                std::mem::take(&mut self.phase)
                            {
                                emit_anthropic_invoke(&mut events, name, params);
                            }
                            self.phase = AnthropicXmlPhase::Idle;
                        }
                        // All other tags (including unknown) inside invoke are suppressed.
                        _ => {}
                    }
                    continue;
                }
                2 => {
                    // InParamValue: accumulate the raw value until the `</parameter>` closing tag.
                    let Some(lt) = self.pending.find('<') else {
                        if let AnthropicXmlPhase::InParamValue { value, .. } = &mut self.phase {
                            value.push_str(&self.pending);
                        }
                        self.pending.clear();
                        break;
                    };
                    let Some(gt_rel) = self.pending.find('>') else {
                        // `<` seen but the tag is not closed: text before `<` is value content; hold from `<` onward.
                        if lt > 0 {
                            if let AnthropicXmlPhase::InParamValue { value, .. } = &mut self.phase {
                                value.push_str(&self.pending[..lt]);
                            }
                            self.pending.drain(..lt);
                        }
                        break;
                    };
                    let tag = self.pending[lt..=gt_rel].to_string();
                    if matches!(classify_anthropic_tag(&tag), AnthropicTagClass::ParamClose) {
                        if let AnthropicXmlPhase::InParamValue {
                            name,
                            params,
                            key,
                            force_string,
                            value,
                        } = std::mem::take(&mut self.phase)
                        {
                            let mut value = value;
                            value.push_str(&self.pending[..lt]);
                            let mut params = params;
                            insert_anthropic_param(&mut params, key, &value, force_string);
                            self.phase = AnthropicXmlPhase::InInvoke { name, params };
                        }
                        self.pending.drain(..=gt_rel);
                    } else {
                        // `<...>` belongs to the value (e.g. a code snippet); fold it into the value and continue.
                        if let AnthropicXmlPhase::InParamValue { value, .. } = &mut self.phase {
                            value.push_str(&self.pending[..=gt_rel]);
                        }
                        self.pending.drain(..=gt_rel);
                    }
                    continue;
                }
                _ => {
                    // InResultBlock: swallow everything inside the hallucinated "tool result" block until the matching closing
                    // tag (`</function_results>` etc.); nested tags, result text, and prose inside are never echoed.
                    let Some(lt) = self.pending.find('<') else {
                        // No tag start: it is all in-block text; discard.
                        self.pending.clear();
                        break;
                    };
                    // Content before `<` is still in-block text; discard.
                    if lt > 0 {
                        self.pending.drain(..lt);
                    }
                    let Some(gt_rel) = self.pending.find('>') else {
                        // Tag not closed: could be a `<` inside block text or a partial closing tag.
                        // If it could still be a tag-name prefix, hold and wait for the next chunk; otherwise discard this `<`.
                        if could_be_tag_name_prefix(&self.pending) {
                            break;
                        }
                        self.pending.drain(..'<'.len_utf8());
                        continue;
                    };
                    let tag = self.pending[..=gt_rel].to_string();
                    self.pending.drain(..=gt_rel);
                    if matches!(
                        classify_anthropic_tag(&tag),
                        AnthropicTagClass::ResultBlockClose
                    ) {
                        // Block ended, back to Idle. The closing tag itself is swallowed too, never echoed.
                        self.phase = AnthropicXmlPhase::Idle;
                    }
                    // All other tags (including nested opens/unknown) inside the result block are suppressed.
                    continue;
                }
            }
        }

        (cleaned, events)
    }
}

#[derive(Default)]
enum BareXmlPhase {
    #[default]
    Idle,
    InBody {
        name: String,
    },
}

/// Bare tool-name XML: `<execute_command>pwd</execute_command>`.
/// Only takes effect when the tag name itself is a registered tool name; otherwise pass through untouched to avoid false positives on ordinary HTML/XML.
#[derive(Default)]
pub(super) struct BareXmlToolCallStreamer {
    pending: String,
    phase: BareXmlPhase,
}

impl BareXmlToolCallStreamer {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn push(&mut self, chunk: &str) -> (String, Vec<InternalToolCallStreamEvent>) {
        self.pending.push_str(chunk);
        let mut cleaned = String::new();
        let mut events = Vec::new();

        loop {
            match &self.phase {
                BareXmlPhase::Idle => {
                    let Some(lt) = self.pending.find('<') else {
                        cleaned.push_str(&self.pending);
                        self.pending.clear();
                        break;
                    };
                    let before = self.pending[..lt].to_string();

                    let Some(gt_rel) = self.pending[lt..].find('>') else {
                        if could_be_tag_name_prefix(&self.pending[lt..]) {
                            if lt > 0 {
                                cleaned.push_str(&before);
                                self.pending.drain(..lt);
                            }
                            break;
                        }
                        cleaned.push_str(&before);
                        cleaned.push('<');
                        self.pending.drain(..lt + '<'.len_utf8());
                        continue;
                    };

                    let tag_end = lt + gt_rel;
                    let tag = self.pending[lt..=tag_end].to_string();
                    if let Some(name) = super::inline_recovery::parse_bare_xml_open_tag(&tag) {
                        let trimmed = before.trim_end_matches([' ', '\t', '\r', '\n']);
                        cleaned.push_str(trimmed);
                        self.pending.drain(..=tag_end);
                        self.phase = BareXmlPhase::InBody { name };
                        continue;
                    }

                    cleaned.push_str(&before);
                    cleaned.push_str(&tag);
                    self.pending.drain(..=tag_end);
                    continue;
                }
                BareXmlPhase::InBody { name } => {
                    let close_tag = format!("</{name}>");
                    if let Some(pos) = self.pending.find(&close_tag) {
                        let body = self.pending[..pos].to_string();
                        let after = pos + close_tag.len();
                        self.pending.drain(..after);
                        let name = name.clone();
                        if let Some(args) =
                            super::inline_recovery::parse_bare_xml_tool_body(&name, &body)
                        {
                            events.push(InternalToolCallStreamEvent::Begin(name));
                            events.push(InternalToolCallStreamEvent::Args(args));
                            events.push(InternalToolCallStreamEvent::End);
                        } else {
                            cleaned.push_str(&format!("<{name}>{body}</{name}>"));
                        }
                        self.phase = BareXmlPhase::Idle;
                        continue;
                    }
                    break;
                }
            }
        }

        (cleaned, events)
    }
}

fn emit_anthropic_invoke(
    events: &mut Vec<InternalToolCallStreamEvent>,
    name: String,
    params: serde_json::Map<String, serde_json::Value>,
) {
    if name.trim().is_empty() {
        return;
    }
    let args = if params.is_empty() {
        "{}".to_string()
    } else {
        serde_json::Value::Object(params).to_string()
    };
    events.push(InternalToolCallStreamEvent::Begin(name));
    events.push(InternalToolCallStreamEvent::Args(args));
    events.push(InternalToolCallStreamEvent::End);
}

fn insert_anthropic_param(
    params: &mut serde_json::Map<String, serde_json::Value>,
    key: String,
    raw_value: &str,
    force_string: bool,
) {
    if key.is_empty() {
        return;
    }
    let raw = raw_value.trim();
    if force_string {
        params.insert(key, serde_json::Value::String(raw.to_string()));
        return;
    }
    // Try to parse the value as a JSON scalar/structure (number, bool, object, array); otherwise treat it as a string.
    let value = serde_json::from_str::<serde_json::Value>(raw)
        .unwrap_or_else(|_| serde_json::Value::String(raw.to_string()));
    params.insert(key, value);
}

/// Decide whether an unclosed `<...` fragment could still be a tag-name prefix (used for cross-chunk buffering decisions).
/// A real tag name is a run of name characters; any whitespace or other character means this is not a tag start (prose).
fn could_be_tag_name_prefix(after_lt: &str) -> bool {
    let body = after_lt.strip_prefix('<').unwrap_or(after_lt);
    let body = body.strip_prefix('/').unwrap_or(body);
    if body.is_empty() {
        return true;
    }
    if body.len() > 40 {
        return false;
    }
    body.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == ':' || c == '_' || c == '-')
}

/// Classify a single `<...>` tag. Tag names may carry a namespace prefix (the local name after the last `:` is used).
fn classify_anthropic_tag(tag: &str) -> AnthropicTagClass {
    let inner = tag.trim_start_matches('<').trim_end_matches('>').trim();
    let is_close = inner.starts_with('/');
    let inner = inner.trim_start_matches('/').trim_start();
    let inner = inner.trim_end_matches('/').trim_end();
    let (raw_name, attrs) = match inner.find(char::is_whitespace) {
        Some(i) => (&inner[..i], inner[i..].trim()),
        None => (inner, ""),
    };
    let local = raw_name.rsplit(':').next().unwrap_or(raw_name);
    if HALLUCINATED_RESULT_TAG_NAMES.contains(&local) {
        return if is_close {
            AnthropicTagClass::ResultBlockClose
        } else {
            AnthropicTagClass::ResultBlockOpen
        };
    }
    match local {
        "function_calls" | "tool_calls" => AnthropicTagClass::Wrapper,
        "invoke" => {
            if is_close {
                AnthropicTagClass::InvokeClose
            } else {
                AnthropicTagClass::InvokeOpen(parse_anthropic_name_attr(attrs))
            }
        }
        "parameter" => {
            if is_close {
                AnthropicTagClass::ParamClose
            } else {
                AnthropicTagClass::ParamOpen {
                    key: parse_anthropic_name_attr(attrs),
                    force_string: parse_anthropic_bool_attr(attrs, "string"),
                }
            }
        }
        _ => AnthropicTagClass::Other,
    }
}

/// Parse the `name="..."` or `name='...'` value out of a tag's attribute string.
fn parse_anthropic_name_attr(attrs: &str) -> String {
    parse_xml_attr_value(attrs, "name")
        .unwrap_or_default()
        .to_string()
}

fn parse_anthropic_bool_attr(attrs: &str, name: &str) -> bool {
    parse_xml_attr_value(attrs, name).is_some_and(|value| value.eq_ignore_ascii_case("true"))
}

/// Scan attribute values along XML attribute boundaries so the target name is never mismatched into other attribute names or already-quoted values.
pub(super) fn parse_xml_attr_value<'a>(input: &'a str, target: &str) -> Option<&'a str> {
    let bytes = input.as_bytes();
    let mut cursor = 0;

    while cursor < bytes.len() {
        while cursor < bytes.len()
            && (bytes[cursor].is_ascii_whitespace() || matches!(bytes[cursor], b'<' | b'>' | b'/'))
        {
            cursor += 1;
        }
        let name_start = cursor;
        while cursor < bytes.len()
            && !bytes[cursor].is_ascii_whitespace()
            && !matches!(bytes[cursor], b'=' | b'>' | b'/')
        {
            cursor += 1;
        }
        if name_start == cursor {
            cursor += 1;
            continue;
        }
        let candidate = &input[name_start..cursor];

        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }
        if bytes.get(cursor) != Some(&b'=') {
            continue;
        }
        cursor += 1;
        while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
            cursor += 1;
        }

        let (value_start, value_end) = match bytes.get(cursor).copied() {
            Some(quote @ (b'"' | b'\'')) => {
                cursor += 1;
                let start = cursor;
                while cursor < bytes.len() && bytes[cursor] != quote {
                    cursor += 1;
                }
                if cursor == bytes.len() {
                    return None;
                }
                let end = cursor;
                cursor += 1;
                (start, end)
            }
            Some(_) => {
                let start = cursor;
                while cursor < bytes.len()
                    && !bytes[cursor].is_ascii_whitespace()
                    && !matches!(bytes[cursor], b'>' | b'/')
                {
                    cursor += 1;
                }
                (start, cursor)
            }
            None => return None,
        };

        if candidate == target {
            return Some(&input[value_start..value_end]);
        }
    }

    None
}

fn sanitize_internal_tool_call_name(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '<' {
            let mut peeked = String::new();
            for next in chars.by_ref() {
                peeked.push(next);
                if next == '>' {
                    break;
                }
            }
            let _ = peeked;
            continue;
        }
        out.push(ch);
    }
    out.trim().to_string()
}

fn earliest_substring_match(s: &str, needles: &[&str]) -> Option<(usize, usize, usize)> {
    needles
        .iter()
        .enumerate()
        .filter_map(|(idx, needle)| s.find(needle).map(|pos| (pos, idx, needle.len())))
        .min_by_key(|(pos, _, _)| *pos)
}

fn longest_marker_suffix_prefix(s: &str, markers: &[&str]) -> usize {
    if s.is_empty() || markers.is_empty() {
        return 0;
    }
    let mut best = 0usize;
    let mut starts = s.char_indices().map(|(idx, _)| idx).collect::<Vec<_>>();
    starts.push(s.len());
    for start in starts {
        let suffix = &s[start..];
        if suffix.is_empty() {
            continue;
        }
        if markers
            .iter()
            .any(|marker| marker.starts_with(suffix) && marker.len() > suffix.len())
        {
            best = best.max(suffix.len());
        }
    }
    best
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum StreamSplitSegment {
    Text(String),
    Marker { marker_index: usize, text: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum WrappedSplitSegment {
    Text(String),
    Marker(String),
}

#[derive(Default)]
pub(super) struct StreamSplitter {
    pending: String,
    /// Scratch reused by `longest_marker_prefix_suffix` for its char-offset
    /// collection. Cleared at every use; keeps repeated `take_segments` calls from
    /// allocating a fresh `Vec` each time. Purely an allocation-reuse cache: the
    /// scan result (and therefore the emitted segments) is unaffected.
    char_offset_scratch: Vec<usize>,
}

impl StreamSplitter {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn push(&mut self, chunk: &str, markers: &[&str]) -> Vec<StreamSplitSegment> {
        self.pending.push_str(chunk);
        self.take_segments(markers, false)
    }

    pub(super) fn flush(&mut self, markers: &[&str]) -> Vec<StreamSplitSegment> {
        self.take_segments(markers, true)
    }

    fn take_segments(&mut self, markers: &[&str], flush_all: bool) -> Vec<StreamSplitSegment> {
        let mut segments = Vec::new();
        loop {
            if let Some((marker_pos, marker_index, marker_len)) =
                earliest_marker_match(&self.pending, markers)
            {
                if marker_pos > 0 {
                    segments.push(StreamSplitSegment::Text(
                        self.pending[..marker_pos].to_string(),
                    ));
                }
                let marker_end = marker_pos + marker_len;
                segments.push(StreamSplitSegment::Marker {
                    marker_index,
                    text: self.pending[marker_pos..marker_end].to_string(),
                });
                self.pending.drain(..marker_end);
                continue;
            }

            let keep_len = if flush_all {
                0
            } else {
                longest_marker_prefix_suffix(&self.pending, markers, &mut self.char_offset_scratch)
            };
            let emit_len = self.pending.len().saturating_sub(keep_len);
            if emit_len == 0 {
                break;
            }

            segments.push(StreamSplitSegment::Text(
                self.pending[..emit_len].to_string(),
            ));
            self.pending.drain(..emit_len);
            if !flush_all {
                break;
            }
        }
        segments
    }
}

fn earliest_marker_match(s: &str, markers: &[&str]) -> Option<(usize, usize, usize)> {
    markers
        .iter()
        .enumerate()
        .filter_map(|(marker_index, marker)| {
            s.find(marker)
                .map(|marker_pos| (marker_pos, marker_index, marker.len()))
        })
        .min_by_key(|(marker_pos, _, _)| *marker_pos)
}

fn longest_marker_prefix_suffix(
    s: &str,
    markers: &[&str],
    char_offsets_scratch: &mut Vec<usize>,
) -> usize {
    if s.is_empty() || markers.is_empty() {
        return 0;
    }

    char_offsets_scratch.clear();
    let mut best = 0usize;
    char_offsets_scratch.extend(s.char_indices().map(|(idx, _)| idx));
    char_offsets_scratch.push(s.len());
    for start in char_offsets_scratch.iter().copied() {
        let suffix = &s[start..];
        if markers
            .iter()
            .any(|marker| marker.starts_with(suffix) && marker.len() > suffix.len())
        {
            best = best.max(suffix.len());
        }
    }
    best
}

pub(super) fn extract_internal_tool_calls(s: &str) -> (String, Vec<InternalToolCall>) {
    let segments = split_wrapped_markers(s, "<|", "|>");
    let mut result = String::with_capacity(s.len());
    let mut tool_calls = Vec::new();
    let mut pending_tool_call_begin = false;

    for segment in segments {
        match segment {
            WrappedSplitSegment::Text(text) => {
                if pending_tool_call_begin {
                    if let Some((tool_call, consumed)) =
                        parse_internal_tool_call_payload(&text, tool_calls.len())
                    {
                        tool_calls.push(tool_call);
                        pending_tool_call_begin = false;
                        if consumed < text.len() {
                            result.push_str(&text[consumed..]);
                        }
                    } else {
                        pending_tool_call_begin = false;
                        result.push_str(&text);
                    }
                } else {
                    result.push_str(&text);
                }
            }
            WrappedSplitSegment::Marker(marker) => {
                pending_tool_call_begin = marker == "<|tool_call_begin|>";
            }
        }
    }

    (result, tool_calls)
}

fn split_wrapped_markers(s: &str, start: &str, end: &str) -> Vec<WrappedSplitSegment> {
    let mut segments = Vec::new();
    let mut offset = 0usize;

    while let Some(start_rel) = s[offset..].find(start) {
        let marker_start = offset + start_rel;
        if marker_start > offset {
            segments.push(WrappedSplitSegment::Text(
                s[offset..marker_start].to_string(),
            ));
        }

        let body_start = marker_start + start.len();
        let Some(end_rel) = s[body_start..].find(end) else {
            segments.push(WrappedSplitSegment::Text(s[marker_start..].to_string()));
            return segments;
        };
        let marker_end = body_start + end_rel + end.len();
        segments.push(WrappedSplitSegment::Marker(
            s[marker_start..marker_end].to_string(),
        ));
        offset = marker_end;
    }

    if offset < s.len() {
        segments.push(WrappedSplitSegment::Text(s[offset..].to_string()));
    }
    segments
}

fn parse_internal_tool_call_payload(
    s: &str,
    tool_call_index: usize,
) -> Option<(InternalToolCall, usize)> {
    let (name, name_consumed) = parse_tool_call_name(s);
    let name = name?;
    let mut tool_call = InternalToolCall {
        id: format!("internal_{tool_call_index}"),
        tool_type: "function".to_string(),
        function_name: name,
        arguments: String::new(),
    };

    let mut total_consumed = name_consumed;
    if let Some((args, args_consumed)) = parse_tool_call_args(&s[name_consumed..]) {
        tool_call.arguments = args;
        total_consumed += args_consumed;
    }

    Some((tool_call, total_consumed))
}

fn parse_tool_call_name(s: &str) -> (Option<String>, usize) {
    let mut i = 0usize;
    let mut name = String::new();

    while i < s.len() {
        let Some(ch) = s[i..].chars().next() else {
            break;
        };
        if ch == '<' || ch == '{' {
            break;
        }
        name.push(ch);
        i += ch.len_utf8();
    }

    let name = name.trim().to_string();
    if name.is_empty() {
        (None, 0)
    } else {
        (Some(name), i)
    }
}

fn parse_tool_call_args(s: &str) -> Option<(String, usize)> {
    let bytes = s.as_bytes();
    let mut i = 0usize;

    while i < bytes.len()
        && (bytes[i] == b' ' || bytes[i] == b'\n' || bytes[i] == b'\r' || bytes[i] == b'\t')
    {
        i += 1;
    }

    if i >= bytes.len() || bytes[i] != b'{' {
        return None;
    }

    let json_start = i;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escape = false;

    while i < bytes.len() {
        let b = bytes[i];

        if escape {
            escape = false;
            i += 1;
            continue;
        }

        match b {
            b'\\' if in_string => escape = true,
            b'"' => in_string = !in_string,
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    i += 1;
                    break;
                }
            }
            _ => {}
        }
        i += 1;
    }

    Some((s[json_start..i].to_string(), i))
}

#[cfg(test)]
mod tests {
    use super::{
        BareXmlToolCallStreamer, HermesXmlToolCallStreamer, InternalToolCallStreamEvent,
        InternalToolCallStreamer, StreamSplitSegment, StreamSplitter, WrappedSplitSegment,
        extract_internal_tool_calls, split_wrapped_markers,
    };

    #[test]
    fn hermes_streamer_suppresses_markup_and_emits_events_single_chunk() {
        let mut s = HermesXmlToolCallStreamer::new();
        let (cleaned, events) =
            s.push("<tool_call><function=read_file>{\"path\":\"/x\"}</function></tool_call>");
        assert_eq!(cleaned, "", "markup must not appear in visible output");
        assert_eq!(
            events,
            vec![
                InternalToolCallStreamEvent::Begin("read_file".to_string()),
                InternalToolCallStreamEvent::Args("{\"path\":\"/x\"}".to_string()),
                InternalToolCallStreamEvent::End,
            ]
        );
    }

    #[test]
    fn hermes_streamer_emits_visible_text_before_call() {
        let mut s = HermesXmlToolCallStreamer::new();
        let (cleaned, events) = s.push("done.<function=list_agents></function>");
        assert_eq!(cleaned, "done.");
        assert_eq!(
            events.first(),
            Some(&InternalToolCallStreamEvent::Begin(
                "list_agents".to_string()
            ))
        );
        // No parameters → empty object.
        assert!(events.contains(&InternalToolCallStreamEvent::Args("{}".to_string())));
    }

    #[test]
    fn hermes_streamer_holds_markup_split_across_chunks() {
        let mut s = HermesXmlToolCallStreamer::new();
        // The marker arrives split in half; no partial marker may ever be echoed midway.
        let (c1, e1) = s.push("<tool_call>\n<func");
        assert_eq!(c1, "");
        assert!(e1.is_empty());
        let (c2, e2) = s.push("tion=read_file>\n{\"path\":");
        assert_eq!(c2, "", "body must be buffered, not shown");
        assert!(e2.is_empty(), "no events until </function> arrives");
        let (c3, e3) = s.push("\"/x\"}\n</function>\n</tool_call>");
        assert_eq!(c3, "");
        assert_eq!(
            e3,
            vec![
                InternalToolCallStreamEvent::Begin("read_file".to_string()),
                InternalToolCallStreamEvent::Args("{\"path\":\"/x\"}".to_string()),
                InternalToolCallStreamEvent::End,
            ]
        );
    }

    #[test]
    fn hermes_streamer_passes_through_plain_prose() {
        let mut s = HermesXmlToolCallStreamer::new();
        let (cleaned, events) = s.push("just some normal text with a < bracket and 2 < 3");
        assert_eq!(cleaned, "just some normal text with a < bracket and 2 < 3");
        assert!(events.is_empty());
    }

    #[test]
    fn hermes_streamer_handles_parameter_tags() {
        let mut s = HermesXmlToolCallStreamer::new();
        let (cleaned, events) =
            s.push("<function=read_file><parameter=path>/x</parameter></function>");
        assert_eq!(cleaned, "");
        assert_eq!(
            events[0],
            InternalToolCallStreamEvent::Begin("read_file".to_string())
        );
        let InternalToolCallStreamEvent::Args(args) = &events[1] else {
            panic!("expected args event");
        };
        let parsed: serde_json::Value = serde_json::from_str(args).unwrap();
        assert_eq!(parsed["path"], "/x");
    }

    #[test]
    fn bare_xml_streamer_suppresses_registered_tool_markup_and_emits_events() {
        let mut s = BareXmlToolCallStreamer::new();
        let (cleaned, events) = s.push("让我检查一下。<execute_command>pwd</execute_command>");
        assert_eq!(cleaned, "让我检查一下。");
        assert_eq!(
            events,
            vec![
                InternalToolCallStreamEvent::Begin("execute_command".to_string()),
                InternalToolCallStreamEvent::Args("{\"command\":\"pwd\"}".to_string()),
                InternalToolCallStreamEvent::End,
            ]
        );
    }

    #[test]
    fn bare_xml_streamer_leaves_unregistered_tags_visible() {
        let mut s = BareXmlToolCallStreamer::new();
        let (cleaned, events) = s.push("a <div>demo</div> b");
        assert_eq!(cleaned, "a <div>demo</div> b");
        assert!(events.is_empty());
    }

    // ── Hallucinated "tool result" protocol markers (bug A/B fix) ──────────────────────────

    #[test]
    fn anthropic_streamer_swallows_hallucinated_result_block_and_signals() {
        // Model-fabricated "tool results": the whole block (tag + hallucinated result text) is never echoed,
        // and HallucinatedProtocolMarker is reported so upper layers can downgrade-retry.
        let mut s = super::AnthropicXmlToolCallStreamer::new();
        let (cleaned, events) = s.push(
            "正在读取文件<function_results>File: a.rs\n3 matches found</function_results>完成",
        );
        assert_eq!(cleaned, "正在读取文件完成", "幻觉结果块不得进入可见正文");
        assert!(
            events.contains(&InternalToolCallStreamEvent::HallucinatedProtocolMarker),
            "必须上报幻觉标记指纹，events={events:?}"
        );
    }

    #[test]
    fn anthropic_streamer_swallows_result_block_split_across_chunks() {
        // Cross-chunk: open tag, result text, and closing tag spread across multiple chunks are still swallowed as one block.
        let mut s = super::AnthropicXmlToolCallStreamer::new();
        let mut all_cleaned = String::new();
        let mut all_events = Vec::new();
        for chunk in [
            "pre <function_res",
            "ults>File: x.rs\n1 match",
            " found</function_",
            "results> post",
        ] {
            let (cleaned, events) = s.push(chunk);
            all_cleaned.push_str(&cleaned);
            all_events.extend(events);
        }
        assert_eq!(all_cleaned, "pre  post");
        assert!(
            all_events.contains(&InternalToolCallStreamEvent::HallucinatedProtocolMarker),
            "跨 chunk 也必须上报幻觉标记，events={all_events:?}"
        );
    }

    #[test]
    fn anthropic_streamer_signals_on_orphan_result_close_tag() {
        // Repetition breaking the pairing produces orphan closing tags; they are the same zero-false-positive fingerprint — suppress and report.
        let mut s = super::AnthropicXmlToolCallStreamer::new();
        let (cleaned, events) = s.push("尾部残留</function_results>之后");
        assert_eq!(cleaned, "尾部残留之后");
        assert!(events.contains(&InternalToolCallStreamEvent::HallucinatedProtocolMarker));
    }

    #[test]
    fn anthropic_streamer_handles_namespaced_result_tag() {
        // Hallucinated markers with a namespace prefix (e.g. `antml:`) also match by local name.
        let mut s = super::AnthropicXmlToolCallStreamer::new();
        let (cleaned, events) = s.push("x<tool_result>garbage</tool_result>y");
        assert_eq!(cleaned, "xy");
        assert!(events.contains(&InternalToolCallStreamEvent::HallucinatedProtocolMarker));
    }

    #[test]
    fn anthropic_streamer_keeps_legit_html_and_normal_invoke_untouched() {
        // Zero-false-positive regression: ordinary HTML is kept as-is and produces no hallucination signal.
        let mut s = super::AnthropicXmlToolCallStreamer::new();
        let (cleaned, events) = s.push("see <div>result</div> and <span>ok</span>");
        assert_eq!(cleaned, "see <div>result</div> and <span>ok</span>");
        assert!(
            !events.contains(&InternalToolCallStreamEvent::HallucinatedProtocolMarker),
            "普通 HTML 不得触发幻觉信号"
        );
        // Legitimate invoke tool calls still parse normally, unaffected by the result-block logic.
        let mut s2 = super::AnthropicXmlToolCallStreamer::new();
        let (cleaned2, events2) = s2.push(
            r#"go<invoke name="read_file"><parameter name="path">/x</parameter></invoke>done"#,
        );
        assert_eq!(cleaned2, "godone");
        assert_eq!(
            events2.first(),
            Some(&InternalToolCallStreamEvent::Begin("read_file".to_string()))
        );
        assert!(!events2.contains(&InternalToolCallStreamEvent::HallucinatedProtocolMarker));
    }

    #[test]
    fn anthropic_streamer_respects_string_true_attr() {
        // DSML `string="true"`: values that look like JSON scalars still stay strings.
        let mut s = super::AnthropicXmlToolCallStreamer::new();
        let (cleaned, events) = s.push(
            r#"<tool_calls><invoke name="enable_tools"><parameter name="operation" string="true">enable</parameter><parameter name="tools" string="false">["read_file"]</parameter><parameter name="flag" string="true">true</parameter></invoke></tool_calls>"#,
        );
        assert_eq!(cleaned, "");
        let mut it = events.into_iter();
        assert_eq!(
            it.next(),
            Some(InternalToolCallStreamEvent::Begin(
                "enable_tools".to_string()
            ))
        );
        let InternalToolCallStreamEvent::Args(args) = it.next().expect("expected args event")
        else {
            panic!("expected args event");
        };
        let parsed: serde_json::Value = serde_json::from_str(&args).unwrap();
        assert_eq!(parsed["operation"], "enable");
        assert!(parsed["operation"].is_string());
        assert_eq!(parsed["tools"], serde_json::json!(["read_file"]));
        assert_eq!(parsed["flag"], "true");
        assert!(parsed["flag"].is_string());
        assert_eq!(it.next(), Some(InternalToolCallStreamEvent::End));
    }

    #[test]
    fn anthropic_streamer_matches_bool_attributes_by_exact_name() {
        let mut s = super::AnthropicXmlToolCallStreamer::new();
        let (cleaned, events) = s.push(
            r#"<invoke name="enable_tools"><parameter name="string" string="true">123</parameter><parameter name="count" notstring="true" string="false">456</parameter></invoke>"#,
        );
        assert_eq!(cleaned, "");

        let InternalToolCallStreamEvent::Args(args) = &events[1] else {
            panic!("expected args event");
        };
        let parsed: serde_json::Value = serde_json::from_str(args).unwrap();
        assert_eq!(parsed["string"], "123");
        assert_eq!(parsed["count"], 456);
    }

    #[test]
    fn bare_xml_streamer_handles_split_chunks() {
        let mut s = BareXmlToolCallStreamer::new();
        let mut all_cleaned = String::new();
        let mut all_events = Vec::new();
        for chunk in [
            "前缀 <",
            "execute_command>",
            "pwd</execute_",
            "command> 后缀",
        ] {
            let (cleaned, events) = s.push(chunk);
            all_cleaned.push_str(&cleaned);
            all_events.extend(events);
        }
        assert_eq!(all_cleaned, "前缀  后缀");
        assert_eq!(all_events.len(), 3);
        assert_eq!(
            all_events[0],
            InternalToolCallStreamEvent::Begin("execute_command".to_string())
        );
    }

    #[test]
    fn push_splits_marker_and_text_in_same_chunk() {
        let mut splitter = StreamSplitter::new();
        let segments = splitter.push("hello<done>world", &["<done>"]);

        assert_eq!(
            segments,
            vec![
                StreamSplitSegment::Text("hello".to_string()),
                StreamSplitSegment::Marker {
                    marker_index: 0,
                    text: "<done>".to_string(),
                },
                StreamSplitSegment::Text("world".to_string()),
            ]
        );
    }

    #[test]
    fn push_preserves_partial_marker_across_chunks() {
        let mut splitter = StreamSplitter::new();

        let first = splitter.push("hello<do", &["<done>"]);
        let second = splitter.push("ne>world", &["<done>"]);

        assert_eq!(first, vec![StreamSplitSegment::Text("hello".to_string())]);
        assert_eq!(
            second,
            vec![
                StreamSplitSegment::Marker {
                    marker_index: 0,
                    text: "<done>".to_string(),
                },
                StreamSplitSegment::Text("world".to_string()),
            ]
        );
    }

    #[test]
    fn flush_releases_unfinished_marker_prefix_as_text() {
        let mut splitter = StreamSplitter::new();

        let first = splitter.push("hello<do", &["<done>"]);
        let tail = splitter.flush(&["<done>"]);

        assert_eq!(first, vec![StreamSplitSegment::Text("hello".to_string())]);
        assert_eq!(tail, vec![StreamSplitSegment::Text("<do".to_string())]);
    }

    #[test]
    fn push_supports_multiple_markers() {
        let mut splitter = StreamSplitter::new();
        let segments = splitter.push("a<one>b<two>c", &["<one>", "<two>"]);

        assert_eq!(
            segments,
            vec![
                StreamSplitSegment::Text("a".to_string()),
                StreamSplitSegment::Marker {
                    marker_index: 0,
                    text: "<one>".to_string(),
                },
                StreamSplitSegment::Text("b".to_string()),
                StreamSplitSegment::Marker {
                    marker_index: 1,
                    text: "<two>".to_string(),
                },
                StreamSplitSegment::Text("c".to_string()),
            ]
        );
    }

    #[test]
    fn wrapped_marker_splitter_extracts_text_and_markers() {
        let segments = split_wrapped_markers("a<|x|>b<|y|>", "<|", "|>");

        assert_eq!(
            segments,
            vec![
                WrappedSplitSegment::Text("a".to_string()),
                WrappedSplitSegment::Marker("<|x|>".to_string()),
                WrappedSplitSegment::Text("b".to_string()),
                WrappedSplitSegment::Marker("<|y|>".to_string()),
            ]
        );
    }

    #[test]
    fn internal_tool_call_extraction_uses_splitter_logic() {
        let (cleaned, tool_calls) = extract_internal_tool_calls(
            "before<|tool_call_begin|>execute_command {\"command\":\"pwd\"}<|tool_call_end|>after",
        );

        assert_eq!(cleaned, "beforeafter");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function_name, "execute_command");
        assert_eq!(tool_calls[0].arguments, "{\"command\":\"pwd\"}");
    }

    #[test]
    fn internal_tool_call_extraction_skips_unknown_wrapped_markers() {
        let (cleaned, tool_calls) = extract_internal_tool_calls("a<|unknown|>b");

        assert_eq!(cleaned, "ab");
        assert!(tool_calls.is_empty());
    }

    #[test]
    fn internal_tool_call_streamer_emits_args_incrementally_across_chunks() {
        let mut streamer = InternalToolCallStreamer::new();

        let (cleaned1, events1) =
            streamer.push("intro<|tool_call_begin|>write_file<|tool_call_args|>{\"path\":\"a\"");
        assert_eq!(cleaned1, "intro");
        assert_eq!(
            events1,
            vec![
                InternalToolCallStreamEvent::Begin("write_file".to_string()),
                InternalToolCallStreamEvent::Args("{\"path\":\"a\"".to_string()),
            ]
        );

        let (cleaned2, events2) = streamer.push(",\"content\":\"hi\"}");
        assert_eq!(cleaned2, "");
        assert_eq!(
            events2,
            vec![InternalToolCallStreamEvent::Args(
                ",\"content\":\"hi\"}".to_string()
            )]
        );

        let (cleaned3, events3) = streamer.push("<|tool_call_end|>after");
        assert_eq!(cleaned3, "after");
        assert_eq!(events3, vec![InternalToolCallStreamEvent::End]);
    }

    #[test]
    fn internal_tool_call_streamer_handles_split_begin_marker() {
        let mut streamer = InternalToolCallStreamer::new();

        let (cleaned1, events1) = streamer.push("hello<|tool_call_be");
        assert_eq!(cleaned1, "hello");
        assert!(events1.is_empty());

        let (cleaned2, events2) =
            streamer.push("gin|>do_work<|tool_call_args|>{\"x\":1}<|tool_call_end|>");
        assert_eq!(cleaned2, "");
        assert_eq!(
            events2,
            vec![
                InternalToolCallStreamEvent::Begin("do_work".to_string()),
                InternalToolCallStreamEvent::Args("{\"x\":1}".to_string()),
                InternalToolCallStreamEvent::End,
            ]
        );
    }

    #[test]
    fn internal_tool_call_streamer_falls_back_when_args_marker_missing() {
        let mut streamer = InternalToolCallStreamer::new();

        let (cleaned, events) = streamer
            .push("<|tool_call_begin|>execute_command {\"command\":\"pwd\"}<|tool_call_end|>");
        assert_eq!(cleaned, "");
        assert_eq!(
            events,
            vec![
                InternalToolCallStreamEvent::Begin("execute_command".to_string()),
                InternalToolCallStreamEvent::Args("{\"command\":\"pwd\"}".to_string()),
                InternalToolCallStreamEvent::End,
            ]
        );
    }

    #[test]
    fn internal_tool_call_streamer_does_not_leak_partial_end_marker() {
        let mut streamer = InternalToolCallStreamer::new();

        let (_, events1) = streamer.push("<|tool_call_begin|>tool<|tool_call_args|>{\"a\":1}");
        assert!(matches!(
            events1.last(),
            Some(InternalToolCallStreamEvent::Args(_))
        ));

        let (cleaned2, events2) = streamer.push("<|tool_call_e");
        assert_eq!(cleaned2, "");
        assert!(events2.is_empty());

        let (cleaned3, events3) = streamer.push("nd|>tail");
        assert_eq!(cleaned3, "tail");
        assert_eq!(events3, vec![InternalToolCallStreamEvent::End]);
    }
}
