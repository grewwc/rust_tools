use super::state::InternalToolCall;

const TOOL_CALL_BEGIN_MARKER: &str = "<|tool_call_begin|>";
const TOOL_CALL_ARGS_MARKER: &str = "<|tool_call_args|>";
const TOOL_CALL_END_MARKER: &str = "<|tool_call_end|>";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum InternalToolCallStreamEvent {
    Begin(String),
    Args(String),
    End,
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
    /// 已吞掉 `<function=`，正在等待函数名后的 `>`。
    AwaitingName,
    /// 已捕获函数名，正在缓冲 body 直到 `</function>`（期间不外显任何字符）。
    InBody {
        name: String,
    },
}

/// 流式抑制 Hermes / Qwen 风格的 XML tool call（`<function=NAME>...</function>`，
/// 可被 `<tool_call>` 包裹），在生成期间就把这段标记从可见输出里剥掉，并即时
/// 转换成与 `<|tool_call_begin|>` 相同的 Begin/Args/End 事件交由统一管线渲染。
/// 这样模型每轮调用工具时，终端不会先闪现一段 `<function=...>` 原始标记。
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
                            // marker 之前的内容是正常可见文本；但紧邻 marker 的尾随
                            // 空白只是包裹/调用前的噪声，去掉以免出现多余空行。
                            let before = self.pending[..pos]
                                .trim_end_matches([' ', '\t', '\r', '\n']);
                            cleaned.push_str(before);
                            let after = pos + len;
                            self.pending.drain(..after);
                            if candidates[idx] == FN_OPEN_MARKER {
                                self.phase = HermesXmlPhase::AwaitingName;
                            }
                            // `<tool_call>` / `</tool_call>` 仅作包裹标记，直接抑制后继续。
                            continue;
                        }
                        None => {
                            // 仅保留可能是 marker 前缀的尾巴，其余安全外显。
                            let mut keep = longest_marker_suffix_prefix(&self.pending, &candidates);
                            // 若正握着一个潜在 marker 前缀，则把紧邻其前的空白也一起
                            // 暂存，避免 `<tool_call>\n<func` 这类拆包时把中间的 `\n`
                            // 先闪出来。空白只是被推迟一帧，顺序不变。
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
                    // 函数名尚未完整到达，等待后续 chunk（不外显半截名字）。
                    break;
                }
                HermesXmlPhase::InBody { name } => {
                    if let Some(pos) = self.pending.find(FN_CLOSE_MARKER) {
                        let body = self.pending[..pos].to_string();
                        let after = pos + FN_CLOSE_MARKER.len();
                        self.pending.drain(..after);
                        let name = name.clone();
                        if !name.is_empty() {
                            let args = super::runtime::parse_hermes_function_body(&body)
                                .unwrap_or_else(|| "{}".to_string());
                            events.push(InternalToolCallStreamEvent::Begin(name));
                            events.push(InternalToolCallStreamEvent::Args(args));
                            events.push(InternalToolCallStreamEvent::End);
                        }
                        self.phase = HermesXmlPhase::Idle;
                        continue;
                    }
                    // body 未闭合，整体继续缓冲（不外显），等待 `</function>`。
                    break;
                }
            }
        }

        (cleaned, events)
    }
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
                longest_marker_prefix_suffix(&self.pending, markers)
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

fn longest_marker_prefix_suffix(s: &str, markers: &[&str]) -> usize {
    if s.is_empty() || markers.is_empty() {
        return 0;
    }

    let mut best = 0usize;
    let mut starts = s.char_indices().map(|(idx, _)| idx).collect::<Vec<_>>();
    starts.push(s.len());
    for start in starts {
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
        HermesXmlToolCallStreamer, InternalToolCallStreamEvent, InternalToolCallStreamer,
        StreamSplitSegment, StreamSplitter, WrappedSplitSegment, extract_internal_tool_calls,
        split_wrapped_markers,
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
        assert_eq!(events.first(), Some(&InternalToolCallStreamEvent::Begin("list_agents".to_string())));
        // 无参数 → 空对象。
        assert!(events.contains(&InternalToolCallStreamEvent::Args("{}".to_string())));
    }

    #[test]
    fn hermes_streamer_holds_markup_split_across_chunks() {
        let mut s = HermesXmlToolCallStreamer::new();
        // marker 被切成两半到达，中途不得外显任何半截标记。
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
        assert_eq!(events[0], InternalToolCallStreamEvent::Begin("read_file".to_string()));
        let InternalToolCallStreamEvent::Args(args) = &events[1] else {
            panic!("expected args event");
        };
        let parsed: serde_json::Value = serde_json::from_str(args).unwrap();
        assert_eq!(parsed["path"], "/x");
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
