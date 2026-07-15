use super::{
    splitter::{
        self, AnthropicXmlToolCallStreamer, BareXmlToolCallStreamer, HermesXmlToolCallStreamer,
        InternalToolCallStreamEvent, InternalToolCallStreamer,
    },
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

    let mut events = Vec::new();
    let mut all_tool_calls = Vec::new();

    // 先处理 reasoning_content（即使 content 也非空，两者可共存于同一 chunk）
    if !delta.reasoning_content.is_empty() {
        let (cleaned, tool_calls) = splitter::extract_internal_tool_calls(&delta.reasoning_content);
        let cleaned = normalize_stream_text(cleaned);
        if !cleaned.is_empty() || !tool_calls.is_empty() {
            if !cleaned.is_empty() {
                if !*thinking_open {
                    *thinking_open = true;
                    events.push(StreamTextEvent::OpenThinking);
                }
                push_text_with_hidden_meta(
                    &mut events,
                    cleaned,
                    true,
                    hidden_begin,
                    hidden_end,
                    hidden_meta_parse,
                );
            }
            all_tool_calls = tool_calls;
        }
        // content 为空时保持 thinking 开启并返回
        if delta.content.is_empty() {
            return (events, all_tool_calls);
        }
    }

    // 处理 content（先关闭 thinking 标签）
    if *thinking_open {
        *thinking_open = false;
        events.push(StreamTextEvent::CloseThinking);
    }
    if !delta.content.is_empty() {
        let content = normalize_stream_text(delta.content.clone());
        push_text_with_hidden_meta(
            &mut events,
            content,
            false,
            hidden_begin,
            hidden_end,
            hidden_meta_parse,
        );
    }
    (events, all_tool_calls)
}

/// Stateful streaming variant that incrementally emits internal tool call
/// Begin/Args/End events as soon as bytes arrive, instead of buffering until
/// the closing `<|tool_call_end|>` marker is observed.
pub(super) fn extract_chunk_events_streaming(
    chunk: &StreamChunk,
    hidden_begin: &str,
    hidden_end: &str,
    thinking_open: &mut bool,
    hidden_meta_parse: &mut HiddenMetaParseState,
    streamer: &mut InternalToolCallStreamer,
    hermes_streamer: &mut HermesXmlToolCallStreamer,
    anthropic_streamer: &mut AnthropicXmlToolCallStreamer,
    bare_xml_streamer: &mut BareXmlToolCallStreamer,
) -> (Vec<StreamTextEvent>, Vec<InternalToolCallStreamEvent>) {
    let Some(choice) = chunk.choices.first() else {
        return (Vec::new(), Vec::new());
    };
    let delta = &choice.delta;

    let mut events = Vec::new();
    let mut all_tool_events = Vec::new();

    // 先处理 reasoning_content（即使 content 也非空，两者可共存于同一 chunk）
    if !delta.reasoning_content.is_empty() {
        let (cleaned, tool_events) = streamer.push(&delta.reasoning_content);
        let cleaned = normalize_stream_text(cleaned);
        if !cleaned.is_empty() || !tool_events.is_empty() {
            if !cleaned.is_empty() {
                if !*thinking_open {
                    *thinking_open = true;
                    events.push(StreamTextEvent::OpenThinking);
                }
                push_text_with_hidden_meta(
                    &mut events,
                    cleaned,
                    true,
                    hidden_begin,
                    hidden_end,
                    hidden_meta_parse,
                );
            }
            all_tool_events = tool_events;
        }
        // content 为空时保持 thinking 开启并返回
        if delta.content.is_empty() {
            return (events, all_tool_events);
        }
    }

    // 处理 content（先关闭 thinking 标签）
    if *thinking_open {
        *thinking_open = false;
        events.push(StreamTextEvent::CloseThinking);
    }
    if !delta.content.is_empty() {
        let normalized = super::inline_recovery::normalize_inline_tool_call_markup(&delta.content);
        let (cleaned, mut hermes_events) = hermes_streamer.push(&normalized);
        // 再把 Hermes 抽离后的可见文本交给 Anthropic（`<invoke name=...>`）解析器，
        // 兼容 deepseek-v4-flash 等用该格式输出工具调用的模型。
        let (cleaned, anthropic_events) = anthropic_streamer.push(&cleaned);
        let (cleaned, bare_xml_events) = bare_xml_streamer.push(&cleaned);
        hermes_events.extend(anthropic_events);
        hermes_events.extend(bare_xml_events);
        if !cleaned.is_empty() {
            let content = normalize_stream_text(cleaned);
            push_text_with_hidden_meta(
                &mut events,
                content,
                false,
                hidden_begin,
                hidden_end,
                hidden_meta_parse,
            );
        }
        all_tool_events.extend(hermes_events);
        return (events, all_tool_events);
    }
    (events, all_tool_events)
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
                ..Default::default()
            }],
            ..Default::default()
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
                ..Default::default()
            }],
            ..Default::default()
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
                ..Default::default()
            }],
            ..Default::default()
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
                ..Default::default()
            }],
            ..Default::default()
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
                ..Default::default()
            }],
            ..Default::default()
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

        assert_eq!(
            first_events,
            vec![StreamTextEvent::AppendContent("before".to_string())]
        );
        assert_eq!(
            second_events,
            vec![
                StreamTextEvent::AppendHiddenMeta("secret".to_string()),
                StreamTextEvent::AppendContent("after".to_string()),
            ]
        );
    }

    #[test]
    fn extract_normalizes_carriage_returns_before_emitting_events() {
        let reasoning_chunk = StreamChunk {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: String::new(),
                    reasoning_content: "step 1\rstep 2\r\nstep 3".to_string(),
                    reasoning_details: String::new(),
                    tool_calls: Vec::new(),
                },
                finish_reason: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let content_chunk = StreamChunk {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: "line 1\rline 2\r\nline 3".to_string(),
                    reasoning_content: String::new(),
                    reasoning_details: String::new(),
                    tool_calls: Vec::new(),
                },
                finish_reason: None,
                ..Default::default()
            }],
            ..Default::default()
        };

        let mut thinking_open = false;
        let mut hidden_meta_parse = HiddenMetaParseState::default();
        let (reasoning_events, _) = extract_chunk_events_with_tools(
            &reasoning_chunk,
            "<meta:self_note>",
            "</meta:self_note>",
            &mut thinking_open,
            &mut hidden_meta_parse,
        );
        let (content_events, _) = extract_chunk_events_with_tools(
            &content_chunk,
            "<meta:self_note>",
            "</meta:self_note>",
            &mut thinking_open,
            &mut hidden_meta_parse,
        );

        assert_eq!(
            reasoning_events,
            vec![
                StreamTextEvent::OpenThinking,
                StreamTextEvent::AppendThinking("step 1\nstep 2\nstep 3".to_string()),
            ]
        );
        assert_eq!(
            content_events,
            vec![
                StreamTextEvent::CloseThinking,
                StreamTextEvent::AppendContent("line 1\nline 2\nline 3".to_string()),
            ]
        );
    }

    #[test]
    fn streaming_extract_normalizes_fullwidth_dsml_markup_before_tool_parsing() {
        let chunk = StreamChunk {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: r#"prefix<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="read_file"><｜｜DSML｜｜parameter name="path">/tmp/x</｜｜DSML｜｜parameter></｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>"#.to_string(),
                    reasoning_content: String::new(),
                    reasoning_details: String::new(),
                    tool_calls: Vec::new(),
                },
                finish_reason: None,
                ..Default::default()
            }],
            ..Default::default()
        };

        let mut thinking_open = false;
        let mut hidden_meta_parse = HiddenMetaParseState::default();
        let mut streamer = InternalToolCallStreamer::new();
        let mut hermes_streamer = HermesXmlToolCallStreamer::new();
        let mut anthropic_streamer = AnthropicXmlToolCallStreamer::new();
        let mut bare_xml_streamer = BareXmlToolCallStreamer::new();
        let (events, tool_events) = extract_chunk_events_streaming(
            &chunk,
            "<meta:self_note>",
            "</meta:self_note>",
            &mut thinking_open,
            &mut hidden_meta_parse,
            &mut streamer,
            &mut hermes_streamer,
            &mut anthropic_streamer,
            &mut bare_xml_streamer,
        );

        assert_eq!(
            events,
            vec![StreamTextEvent::AppendContent("prefix".to_string())]
        );
        assert_eq!(tool_events.len(), 3);
        match (&tool_events[0], &tool_events[1], &tool_events[2]) {
            (
                InternalToolCallStreamEvent::Begin(name),
                InternalToolCallStreamEvent::Args(args),
                InternalToolCallStreamEvent::End,
            ) => {
                assert_eq!(name, "read_file");
                let v: serde_json::Value = serde_json::from_str(args).unwrap();
                assert_eq!(v["path"], "/tmp/x");
            }
            other => panic!("unexpected tool events: {other:?}"),
        }
    }

    #[test]
    fn streaming_extract_emits_thinking_even_when_chunk_also_carries_content() {
        // 复现「有时完全不输出 thinking」：部分 provider 的 message 快照经
        // merge_reasoning 折叠后，单个 chunk 会同时带 content 与 reasoning_content。
        // 旧的互斥分支会整段丢掉 thinking 显示；修复后二者都应被输出。
        let chunk = StreamChunk {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: "answer".to_string(),
                    reasoning_content: "step one".to_string(),
                    reasoning_details: String::new(),
                    tool_calls: Vec::new(),
                },
                finish_reason: None,
                ..Default::default()
            }],
            ..Default::default()
        };

        let mut thinking_open = false;
        let mut hidden_meta_parse = HiddenMetaParseState::default();
        let mut streamer = InternalToolCallStreamer::new();
        let mut hermes_streamer = HermesXmlToolCallStreamer::new();
        let mut anthropic_streamer = AnthropicXmlToolCallStreamer::new();
        let mut bare_xml_streamer = BareXmlToolCallStreamer::new();
        let (events, _) = extract_chunk_events_streaming(
            &chunk,
            "<meta:self_note>",
            "</meta:self_note>",
            &mut thinking_open,
            &mut hidden_meta_parse,
            &mut streamer,
            &mut hermes_streamer,
            &mut anthropic_streamer,
            &mut bare_xml_streamer,
        );

        assert_eq!(
            events,
            vec![
                StreamTextEvent::OpenThinking,
                StreamTextEvent::AppendThinking("step one".to_string()),
                StreamTextEvent::CloseThinking,
                StreamTextEvent::AppendContent("answer".to_string()),
            ]
        );
        assert!(!thinking_open);
    }

    #[test]
    fn streaming_extract_suppresses_bare_registered_xml_tool_markup() {
        let chunk = StreamChunk {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: "先看一下。<execute_command>pwd</execute_command>".to_string(),
                    reasoning_content: String::new(),
                    reasoning_details: String::new(),
                    tool_calls: Vec::new(),
                },
                finish_reason: None,
                ..Default::default()
            }],
            ..Default::default()
        };

        let mut thinking_open = false;
        let mut hidden_meta_parse = HiddenMetaParseState::default();
        let mut streamer = InternalToolCallStreamer::new();
        let mut hermes_streamer = HermesXmlToolCallStreamer::new();
        let mut anthropic_streamer = AnthropicXmlToolCallStreamer::new();
        let mut bare_xml_streamer = BareXmlToolCallStreamer::new();
        let (events, tool_events) = extract_chunk_events_streaming(
            &chunk,
            "<meta:self_note>",
            "</meta:self_note>",
            &mut thinking_open,
            &mut hidden_meta_parse,
            &mut streamer,
            &mut hermes_streamer,
            &mut anthropic_streamer,
            &mut bare_xml_streamer,
        );

        assert_eq!(
            events,
            vec![StreamTextEvent::AppendContent("先看一下。".to_string())]
        );
        assert_eq!(tool_events.len(), 3);
        match (&tool_events[0], &tool_events[1], &tool_events[2]) {
            (
                InternalToolCallStreamEvent::Begin(name),
                InternalToolCallStreamEvent::Args(args),
                InternalToolCallStreamEvent::End,
            ) => {
                assert_eq!(name, "execute_command");
                let v: serde_json::Value = serde_json::from_str(args).unwrap();
                assert_eq!(v["command"], "pwd");
            }
            other => panic!("unexpected tool events: {other:?}"),
        }
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

    let flush_visible =
        |events: &mut Vec<StreamTextEvent>, visible: &mut String, is_thinking: bool| {
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

fn normalize_stream_text(text: String) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
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
