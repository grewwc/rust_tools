use super::{
    splitter,
    state::{HiddenMetaParseState, InternalToolCall},
};
use crate::ai::request::StreamChunk;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum StreamTextEvent {
    OpenThinking,
    AppendThinking(String),
    CloseThinking,
    AppendContent(String),
    AppendHiddenMeta(String),
}

pub(super) fn extract_chunk_text(
    chunk: &StreamChunk,
    thinking_tag: &str,
    end_thinking_tag: &str,
    thinking_open: &mut bool,
) -> String {
    let mut hidden_meta_parse = HiddenMetaParseState::default();
    let (events, _) = extract_chunk_events_with_tools(
        chunk,
        "<meta:self_note>",
        "</meta:self_note>",
        thinking_open,
        &mut hidden_meta_parse,
    );
    render_legacy_stream_text(&events, thinking_tag, end_thinking_tag)
}

pub(super) fn extract_chunk_events_with_tools(
    chunk: &StreamChunk,
    hidden_begin: &str,
    hidden_end: &str,
    thinking_open: &mut bool,
    hidden_meta_parse: &mut HiddenMetaParseState,
) -> (Vec<StreamTextEvent>, Vec<InternalToolCall>) {
    let Some(choice) = chunk.choices.first() else {
        return (Vec::new(), Vec::new());
    };
    let delta = &choice.delta;

    if delta.content.is_empty() && !delta.reasoning_content.is_empty() {
        let (cleaned, tool_calls) = splitter::extract_internal_tool_calls(&delta.reasoning_content);
        if cleaned.is_empty() && tool_calls.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let mut events = Vec::new();
        if !*thinking_open {
            *thinking_open = true;
            events.push(StreamTextEvent::OpenThinking);
        }
        if !cleaned.is_empty() {
            push_text_with_hidden_meta(
                &mut events,
                cleaned,
                true,
                hidden_begin,
                hidden_end,
                hidden_meta_parse,
            );
        }
        return (events, tool_calls);
    }

    let mut events = Vec::new();
    if *thinking_open {
        *thinking_open = false;
        events.push(StreamTextEvent::CloseThinking);
    }
    if !delta.content.is_empty() {
        push_text_with_hidden_meta(
            &mut events,
            delta.content.clone(),
            false,
            hidden_begin,
            hidden_end,
            hidden_meta_parse,
        );
    }
    (events, Vec::new())
}

pub(super) fn render_stream_text_events(
    events: &[StreamTextEvent],
    thinking_tag: &str,
    end_thinking_tag: &str,
) -> String {
    let mut content = String::new();
    for event in events {
        match event {
            StreamTextEvent::OpenThinking => {
                content.push('\n');
                content.push_str(thinking_tag);
                content.push('\n');
            }
            StreamTextEvent::AppendThinking(text) | StreamTextEvent::AppendContent(text) => {
                content.push_str(text);
            }
            StreamTextEvent::AppendHiddenMeta(_) => {}
            StreamTextEvent::CloseThinking => {
                content.push_str(end_thinking_tag);
                content.push('\n');
            }
        }
    }
    content
}

fn render_legacy_stream_text(
    events: &[StreamTextEvent],
    thinking_tag: &str,
    end_thinking_tag: &str,
) -> String {
    let mut content = String::new();
    for event in events {
        match event {
            StreamTextEvent::OpenThinking => {
                content.push('\n');
                content.push_str(thinking_tag);
                content.push('\n');
            }
            StreamTextEvent::AppendThinking(text) | StreamTextEvent::AppendContent(text) => {
                content.push_str(text);
            }
            StreamTextEvent::AppendHiddenMeta(_) => {}
            StreamTextEvent::CloseThinking => {
                content.push('\n');
                content.push_str(end_thinking_tag);
                content.push('\n');
            }
        }
    }
    content
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::request::{StreamChoice, StreamDelta};

    #[test]
    fn closing_thinking_does_not_add_extra_blank_line_before_end_marker() {
        let chunk = StreamChunk {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: "next".to_string(),
                    reasoning_content: String::new(),
                    reasoning_details: String::new(),
                    tool_calls: Vec::new(),
                },
                finish_reason: None,
            }],
        };

        let mut thinking_open = true;
        let mut hidden_meta_parse = HiddenMetaParseState::default();
        let (events, _) = extract_chunk_events_with_tools(
            &chunk,
            "<meta:self_note>",
            "</meta:self_note>",
            &mut thinking_open,
            &mut hidden_meta_parse,
        );
        let content = render_stream_text_events(&events, "╭─ thinking", "╰─ done thinking");

        assert_eq!(content, "╰─ done thinking\nnext");
        assert!(!thinking_open);
    }

    #[test]
    fn reasoning_chunk_emits_structured_stream_events() {
        let chunk = StreamChunk {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: String::new(),
                    reasoning_content: "step one".to_string(),
                    reasoning_details: String::new(),
                    tool_calls: Vec::new(),
                },
                finish_reason: None,
            }],
        };

        let mut thinking_open = false;
        let mut hidden_meta_parse = HiddenMetaParseState::default();
        let (events, tool_calls) = extract_chunk_events_with_tools(
            &chunk,
            "<meta:self_note>",
            "</meta:self_note>",
            &mut thinking_open,
            &mut hidden_meta_parse,
        );

        assert_eq!(
            events,
            vec![
                StreamTextEvent::OpenThinking,
                StreamTextEvent::AppendThinking("step one".to_string())
            ]
        );
        assert!(tool_calls.is_empty());
        assert!(thinking_open);
    }

    #[test]
    fn hidden_meta_is_emitted_as_events_not_visible_text() {
        let chunk = StreamChunk {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: "before<meta:self_note>secret</meta:self_note>after".to_string(),
                    reasoning_content: String::new(),
                    reasoning_details: String::new(),
                    tool_calls: Vec::new(),
                },
                finish_reason: None,
            }],
        };

        let mut thinking_open = false;
        let mut hidden_meta_parse = HiddenMetaParseState::default();
        let (events, _) = extract_chunk_events_with_tools(
            &chunk,
            "<meta:self_note>",
            "</meta:self_note>",
            &mut thinking_open,
            &mut hidden_meta_parse,
        );

        assert_eq!(
            events,
            vec![
                StreamTextEvent::AppendContent("before".to_string()),
                StreamTextEvent::AppendHiddenMeta("secret".to_string()),
                StreamTextEvent::AppendContent("after".to_string()),
            ]
        );
    }

    #[test]
    fn hidden_meta_markers_can_span_multiple_chunks() {
        let first_chunk = StreamChunk {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: "before<meta:self".to_string(),
                    reasoning_content: String::new(),
                    reasoning_details: String::new(),
                    tool_calls: Vec::new(),
                },
                finish_reason: None,
            }],
        };
        let second_chunk = StreamChunk {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: "_note>secret</meta:self_note>after".to_string(),
                    reasoning_content: String::new(),
                    reasoning_details: String::new(),
                    tool_calls: Vec::new(),
                },
                finish_reason: None,
            }],
        };

        let mut thinking_open = false;
        let mut hidden_meta_parse = HiddenMetaParseState::default();
        let (first_events, _) = extract_chunk_events_with_tools(
            &first_chunk,
            "<meta:self_note>",
            "</meta:self_note>",
            &mut thinking_open,
            &mut hidden_meta_parse,
        );
        let (second_events, _) = extract_chunk_events_with_tools(
            &second_chunk,
            "<meta:self_note>",
            "</meta:self_note>",
            &mut thinking_open,
            &mut hidden_meta_parse,
        );

        assert_eq!(first_events, vec![StreamTextEvent::AppendContent("before".to_string())]);
        assert_eq!(
            second_events,
            vec![
                StreamTextEvent::AppendHiddenMeta("secret".to_string()),
                StreamTextEvent::AppendContent("after".to_string()),
            ]
        );
    }
}

fn push_text_with_hidden_meta(
    events: &mut Vec<StreamTextEvent>,
    text: String,
    is_thinking: bool,
    hidden_begin: &str,
    hidden_end: &str,
    hidden_meta_parse: &mut HiddenMetaParseState,
) {
    if text.is_empty() {
        return;
    }

    let hb: Vec<char> = hidden_begin.chars().collect();
    let he: Vec<char> = hidden_end.chars().collect();
    let mut visible = String::new();
    let mut hidden = String::new();

    let flush_visible = |events: &mut Vec<StreamTextEvent>, visible: &mut String, is_thinking: bool| {
        if visible.is_empty() {
            return;
        }
        let chunk = std::mem::take(visible);
        if is_thinking {
            events.push(StreamTextEvent::AppendThinking(chunk));
        } else {
            events.push(StreamTextEvent::AppendContent(chunk));
        }
    };
    let flush_hidden = |events: &mut Vec<StreamTextEvent>, hidden: &mut String| {
        if hidden.is_empty() {
            return;
        }
        events.push(StreamTextEvent::AppendHiddenMeta(std::mem::take(hidden)));
    };

    for ch in text.chars() {
        if !hidden_meta_parse.hidden_open {
            if hidden_meta_parse.hidden_begin_match < hb.len()
                && ch == hb[hidden_meta_parse.hidden_begin_match]
            {
                hidden_meta_parse.hidden_begin_match += 1;
                if hidden_meta_parse.hidden_begin_match == hb.len() {
                    flush_visible(events, &mut visible, is_thinking);
                    hidden_meta_parse.hidden_open = true;
                    hidden_meta_parse.hidden_begin_match = 0;
                }
            } else {
                if hidden_meta_parse.hidden_begin_match > 0 {
                    for k in 0..hidden_meta_parse.hidden_begin_match {
                        visible.push(hb[k]);
                    }
                    hidden_meta_parse.hidden_begin_match = 0;
                }
                visible.push(ch);
            }
        } else if hidden_meta_parse.hidden_end_match < he.len()
            && ch == he[hidden_meta_parse.hidden_end_match]
        {
            hidden_meta_parse.hidden_end_match += 1;
            if hidden_meta_parse.hidden_end_match == he.len() {
                flush_hidden(events, &mut hidden);
                hidden_meta_parse.hidden_open = false;
                hidden_meta_parse.hidden_end_match = 0;
            }
        } else {
            if hidden_meta_parse.hidden_end_match > 0 {
                for k in 0..hidden_meta_parse.hidden_end_match {
                    hidden.push(he[k]);
                }
                hidden_meta_parse.hidden_end_match = 0;
            }
            hidden.push(ch);
        }
    }

    flush_visible(events, &mut visible, is_thinking);
    flush_hidden(events, &mut hidden);
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
