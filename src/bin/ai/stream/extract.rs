use super::state::InternalToolCall;
use crate::ai::request::StreamChunk;

pub(super) fn extract_chunk_text(
    chunk: &StreamChunk,
    thinking_tag: &str,
    end_thinking_tag: &str,
    thinking_open: &mut bool,
) -> String {
    let (content, _) =
        extract_chunk_text_with_tools(chunk, thinking_tag, end_thinking_tag, thinking_open);
    content
}

pub(super) fn extract_chunk_text_with_tools(
    chunk: &StreamChunk,
    thinking_tag: &str,
    end_thinking_tag: &str,
    thinking_open: &mut bool,
) -> (String, Vec<InternalToolCall>) {
    let Some(choice) = chunk.choices.first() else {
        return (String::new(), Vec::new());
    };
    let delta = &choice.delta;

    if delta.content.is_empty() && !delta.reasoning_content.is_empty() {
        let (cleaned, tool_calls) = extract_internal_tool_calls(&delta.reasoning_content);
        if cleaned.is_empty() && tool_calls.is_empty() {
            return (String::new(), Vec::new());
        }
        if !*thinking_open {
            *thinking_open = true;
            return (format!("\n{thinking_tag}\n{cleaned}"), tool_calls);
        }
        return (cleaned, tool_calls);
    }

    if *thinking_open {
        *thinking_open = false;
        return (format!("\n{end_thinking_tag}\n{}", delta.content), Vec::new());
    }
    (delta.content.clone(), Vec::new())
}

fn extract_internal_tool_calls(s: &str) -> (String, Vec<InternalToolCall>) {
    let mut result = String::with_capacity(s.len());
    let mut tool_calls = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        if bytes[i] == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'|' {
            let marker_start = i;
            let mut marker_end = i + 2;
            while marker_end < bytes.len() {
                if bytes[marker_end] == b'|'
                    && marker_end + 1 < bytes.len()
                    && bytes[marker_end + 1] == b'>'
                {
                    break;
                }
                marker_end += 1;
            }

            let marker = &s[marker_start..marker_end + 2];

            if marker == "<|tool_call_begin|>" {
                let (name, consumed) = parse_tool_call_name(s, marker_end + 2);
                if let Some(name) = name {
                    let mut tc = InternalToolCall {
                        id: format!("internal_{}", tool_calls.len()),
                        tool_type: "function".to_string(),
                        function_name: name,
                        arguments: String::new(),
                    };

                    let remaining_start = marker_end + 2 + consumed;
                    if let Some((args, args_consumed)) = parse_tool_call_args(s, remaining_start) {
                        tc.arguments = args;
                        i = remaining_start + args_consumed;
                    } else {
                        i = remaining_start;
                    }

                    tool_calls.push(tc);
                    continue;
                }
            } else if marker == "<|tool_call_end|>"
                || marker == "<|tool_calls_section_end|>"
                || marker == "<|tool_call_argument_begin|>"
            {
                i = marker_end + 2;
                continue;
            }

            i = marker_end + 2;
            continue;
        }

        let Some(ch) = s[i..].chars().next() else {
            break;
        };
        result.push(ch);
        i += ch.len_utf8();
    }

    (result, tool_calls)
}

fn parse_tool_call_name(s: &str, start: usize) -> (Option<String>, usize) {
    let bytes = s.as_bytes();
    let mut i = start;
    let mut name = String::new();

    while i < bytes.len() {
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
        (Some(name), i - start)
    }
}

fn parse_tool_call_args(s: &str, start: usize) -> Option<(String, usize)> {
    let bytes = s.as_bytes();
    let mut i = start;

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

    let json_str = s[json_start..i].to_string();
    Some((json_str, i - start))
}

pub(super) fn strip_ansi_codes(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut result = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            i += 2;
            while i < bytes.len() {
                let b = bytes[i];
                i += 1;
                if (b as char) >= '@' && (b as char) <= '~' {
                    break;
                }
            }
            continue;
        }
        let Some(ch) = s[i..].chars().next() else {
            break;
        };
        result.push(ch);
        i += ch.len_utf8();
    }
    result
}
