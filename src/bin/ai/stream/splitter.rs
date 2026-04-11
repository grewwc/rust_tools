use super::state::InternalToolCall;

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
                    segments.push(StreamSplitSegment::Text(self.pending[..marker_pos].to_string()));
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

            segments.push(StreamSplitSegment::Text(self.pending[..emit_len].to_string()));
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
            segments.push(WrappedSplitSegment::Text(s[offset..marker_start].to_string()));
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
        StreamSplitSegment, StreamSplitter, WrappedSplitSegment, extract_internal_tool_calls,
        split_wrapped_markers,
    };

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
}
