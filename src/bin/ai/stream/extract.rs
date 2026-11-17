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

    // 只有真正产生可见正文时才结束 thinking。重复的 reasoning 快照在上游去重后
    // 会变成空 chunk；若在此处关闭，会让后续 reasoning 反复重新打开折叠块。
    if !delta.content.is_empty() {
        let content = normalize_stream_text(delta.content.clone());
        let mut content_events = Vec::new();
        push_text_with_hidden_meta(
            &mut content_events,
            content,
            false,
            hidden_begin,
            hidden_end,
            hidden_meta_parse,
        );
        close_thinking_before_visible_content(&mut events, thinking_open, &content_events);
        events.extend(content_events);
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
    inline_markup_normalizer: &mut super::inline_recovery::InlineMarkupNormalizer,
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

    // 只有真正产生可见正文时才结束 thinking。空 delta、被去重的 reasoning 快照、
    // 以及被工具解析器完全消费的 XML 都不应让终端折叠块闪烁地关闭又重新打开。
    if !delta.content.is_empty() {
        let normalized = inline_markup_normalizer.push(&delta.content);
        if normalized.is_empty() {
            // 整段被判定为「可能截断的 marker 前缀」而暂存，等待下一个 chunk。
            return (events, all_tool_events);
        }
        let (cleaned, mut hermes_events) = hermes_streamer.push(&normalized);
        // 再把 Hermes 抽离后的可见文本交给 Anthropic（`<invoke name=...>`）解析器，
        // 兼容 deepseek-v4-flash 等用该格式输出工具调用的模型。
        let (cleaned, anthropic_events) = anthropic_streamer.push(&cleaned);
        let (cleaned, bare_xml_events) = bare_xml_streamer.push(&cleaned);
        hermes_events.extend(anthropic_events);
        hermes_events.extend(bare_xml_events);
        if !cleaned.is_empty() {
            let content = normalize_stream_text(cleaned);
            let mut content_events = Vec::new();
            push_text_with_hidden_meta(
                &mut content_events,
                content,
                false,
                hidden_begin,
                hidden_end,
                hidden_meta_parse,
            );
            close_thinking_before_visible_content(&mut events, thinking_open, &content_events);
            events.extend(content_events);
        }
        all_tool_events.extend(hermes_events);
        return (events, all_tool_events);
    }
    (events, all_tool_events)
}

/// 在可见 assistant 正文首次落盘前才关闭 thinking，忽略不产生正文的协议噪声。
fn close_thinking_before_visible_content(
    events: &mut Vec<StreamTextEvent>,
    thinking_open: &mut bool,
    content_events: &[StreamTextEvent],
) {
    if !*thinking_open
        || !content_events
            .iter()
            .any(|event| matches!(event, StreamTextEvent::AppendContent(_)))
    {
        return;
    }

    *thinking_open = false;
    events.push(StreamTextEvent::CloseThinking);
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
    fn streaming_extract_keeps_thinking_open_across_empty_reasoning_snapshot() {
        // OpenAI Responses 的 summary delta 后可能紧跟一个已在 runtime 去重为空的
        // snapshot chunk。它不是正文开始，不能结束折叠块；否则下一段 summary
        // 又会重新打开 thinking，导致终端连续闪出多个折叠框。
        let chunk = |content: &str, reasoning_content: &str| StreamChunk {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: content.to_string(),
                    reasoning_content: reasoning_content.to_string(),
                    reasoning_details: String::new(),
                    tool_calls: Vec::new(),
                },
                finish_reason: None,
                ..Default::default()
            }],
            ..Default::default()
        };
        let first_reasoning = chunk("", "first");
        let duplicate_snapshot = chunk("", "");
        let next_reasoning = chunk("", "second");
        let answer = chunk("answer", "");

        let mut thinking_open = false;
        let mut hidden_meta_parse = HiddenMetaParseState::default();
        let mut streamer = InternalToolCallStreamer::new();
        let mut hermes_streamer = HermesXmlToolCallStreamer::new();
        let mut anthropic_streamer = AnthropicXmlToolCallStreamer::new();
        let mut bare_xml_streamer = BareXmlToolCallStreamer::new();
        let mut inline_markup_normalizer =
            super::super::inline_recovery::InlineMarkupNormalizer::new();
        let mut extract = |stream_chunk: &StreamChunk| {
            extract_chunk_events_streaming(
                stream_chunk,
                "<meta:self_note>",
                "</meta:self_note>",
                &mut thinking_open,
                &mut hidden_meta_parse,
                &mut streamer,
                &mut hermes_streamer,
                &mut anthropic_streamer,
                &mut bare_xml_streamer,
                &mut inline_markup_normalizer,
            )
            .0
        };

        assert_eq!(
            extract(&first_reasoning),
            vec![
                StreamTextEvent::OpenThinking,
                StreamTextEvent::AppendThinking("first".to_string()),
            ]
        );
        assert!(extract(&duplicate_snapshot).is_empty());
        assert_eq!(
            extract(&next_reasoning),
            vec![StreamTextEvent::AppendThinking("second".to_string())]
        );
        assert_eq!(
            extract(&answer),
            vec![
                StreamTextEvent::CloseThinking,
                StreamTextEvent::AppendContent("answer".to_string()),
            ]
        );
        drop(extract);
        assert!(!thinking_open);
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

    fn hidden_meta_stream_output(chunks: &[&str]) -> (String, String) {
        let mut thinking_open = false;
        let mut state = HiddenMetaParseState::default();
        let mut streamer = InternalToolCallStreamer::new();
        let mut hermes = HermesXmlToolCallStreamer::new();
        let mut anthropic = AnthropicXmlToolCallStreamer::new();
        let mut bare_xml = BareXmlToolCallStreamer::new();
        let mut normalizer = super::super::inline_recovery::InlineMarkupNormalizer::new();
        let mut all_events = Vec::new();
        for text in chunks {
            let chunk = StreamChunk {
                choices: vec![StreamChoice {
                    delta: StreamDelta {
                        content: (*text).to_string(),
                        ..Default::default()
                    },
                    ..Default::default()
                }],
                ..Default::default()
            };
            let (events, tools) = extract_chunk_events_streaming(
                &chunk,
                "<meta:self_note>",
                "</meta:self_note>",
                &mut thinking_open,
                &mut state,
                &mut streamer,
                &mut hermes,
                &mut anthropic,
                &mut bare_xml,
                &mut normalizer,
            );
            assert!(tools.is_empty());
            all_events.extend(events);
        }
        let literal_markers = ["<meta:self_note>", "</meta:self_note>"];
        let (mut tail, tools) = hermes.push(&normalizer.flush());
        assert!(tools.is_empty());
        tail.push_str(&hermes.flush_literal_marker_prefix(&literal_markers));
        let (mut tail, tools) = anthropic.push(&tail);
        assert!(tools.is_empty());
        tail.push_str(&anthropic.flush_literal_marker_prefix(&literal_markers));
        let (tail, tools) = bare_xml.push(&tail);
        assert!(tools.is_empty());
        for tail in [tail, bare_xml.flush()] {
            push_text_with_hidden_meta(
                &mut all_events,
                normalize_stream_text(tail),
                false,
                "<meta:self_note>",
                "</meta:self_note>",
                &mut state,
            );
        }
        flush_hidden_meta(
            &mut all_events,
            "<meta:self_note>",
            "</meta:self_note>",
            &mut state,
        );
        let mut visible = String::new();
        let mut hidden = String::new();
        for event in all_events {
            match event {
                StreamTextEvent::AppendContent(text) => visible.push_str(&text),
                StreamTextEvent::AppendHiddenMeta(text) => hidden.push_str(&text),
                other => panic!("unexpected event: {other:?}"),
            }
        }
        (visible, hidden)
    }

    fn assert_hidden_meta_stream_splits(text: &str, visible: &str, hidden: &str) {
        for cut in text
            .char_indices()
            .map(|(index, _)| index)
            .chain([text.len()])
        {
            assert_eq!(
                hidden_meta_stream_output(&[&text[..cut], &text[cut..]]),
                (visible.to_string(), hidden.to_string()),
                "stream split at {cut}: {text:?}"
            );
        }
        let chunks: Vec<_> = text
            .char_indices()
            .map(|(index, ch)| &text[index..index + ch.len_utf8()])
            .collect();
        assert_eq!(
            hidden_meta_stream_output(&chunks),
            (visible.to_string(), hidden.to_string()),
            "one character per delta: {text:?}"
        );
    }

    #[test]
    fn hidden_meta_fenced_examples_preserve_following_prose() {
        let text = "Example:\n```text\n<meta:self_note>private</meta:self_note>\n```\nMore prose.";
        assert_hidden_meta_stream_splits(text, text, "");
    }

    #[test]
    fn hidden_meta_inline_examples_preserve_following_prose() {
        let text = "- 内部反思写在 `<meta:self_note>` followed by the rest of the answer.";
        assert_hidden_meta_stream_splits(text, text, "");
    }

    #[test]
    fn hidden_meta_prose_markers_survive_every_stream_split() {
        let text = "before<meta:self_note>secret</meta:self_note>after";
        assert_hidden_meta_stream_splits(text, "beforeafter", "secret");
        assert_hidden_meta_stream_splits(
            "before<meta:self<meta:self_note>secret</meta:self</meta:self_note>after",
            "before<meta:selfafter",
            "secret</meta:self",
        );
    }

    #[test]
    fn hidden_meta_code_markers_survive_every_stream_split() {
        for example in [
            "~~~~text\n<meta:self_note>literal</meta:self_note>\n~~~~\n",
            "   ```text\n<meta:self_note>\n   ```\n",
            "````text\n```\n<meta:self_note>\n````\n",
            "~~~text\n```\n<meta:self_note>\n~~~\n",
            "```text\n``` trailing text\n<meta:self_note>\n```\n",
            "`` <meta:self_note> ` literal </meta:self_note> ``\n",
            "`<meta:self` and `<meta:self_note>literal</meta:self_note>`\n",
            "    <meta:self_note>indented</meta:self_note>\n",
            "\t<meta:self_note>tab indented</meta:self_note>\n",
            "```text\n<meta:self\n```\n",
        ] {
            let text = format!("{example}after<meta:self_note>secret</meta:self_note>done");
            assert_hidden_meta_stream_splits(&text, &format!("{example}afterdone"), "secret");
        }
        let crlf = "~~~text\r\n<meta:self_note>literal</meta:self_note>\r\n~~~\r\nafter";
        assert_eq!(
            hidden_meta_stream_output(&[crlf]),
            (crlf.replace("\r\n", "\n"), String::new())
        );
    }

    #[test]
    fn hidden_meta_unfinished_prefixes_are_flushed() {
        for end in 1.."<meta:self_note>".len() {
            let text = format!("answer {}", &"<meta:self_note>"[..end]);
            assert_hidden_meta_stream_splits(&text, &text, "");
        }
        assert_hidden_meta_stream_splits(
            "answer<meta:self_note>secret</meta:self",
            "answer",
            "secret</meta:self",
        );
    }

    #[test]
    fn hidden_meta_container_fences_survive_every_stream_split() {
        for example in [
            "> ~~~text\n> <meta:self_note>literal</meta:self_note>\n> ~~~\n",
            ">~~~text\n><meta:self_note>literal</meta:self_note>\n>~~~\n",
            "  > > ~~~~text\n>> <meta:self_note>literal</meta:self_note>\n> > ~~~~\n",
            "- ~~~text\n  <meta:self_note>literal</meta:self_note>\n  ~~~\n",
            "+ ~~~text\n  <meta:self_note>literal</meta:self_note>\n  ~~~\n",
            "* ~~~text\n  <meta:self_note>literal</meta:self_note>\n  ~~~\n",
            "123. ~~~text\n     <meta:self_note>literal</meta:self_note>\n     ~~~\n",
            "1) ~~~text\n   <meta:self_note>literal</meta:self_note>\n   ~~~\n",
            "- - ~~~text\n    <meta:self_note>literal</meta:self_note>\n    ~~~\n",
            "> - ~~~text\n>   <meta:self_note>literal</meta:self_note>\n>   ~~~\n",
            "- > ~~~text\n  > <meta:self_note>literal</meta:self_note>\n  > ~~~\n",
            "> ```text\n> <meta:self_note>literal</meta:self_note>\n> ```\n",
            "> ~~~text\n> <meta:self\n> ~~~\n",
            "~~~text\n> ~~~\n<meta:self_note>literal</meta:self_note>\n~~~\n",
            "> ~~~text\n> - ~~~\n> <meta:self_note>literal</meta:self_note>\n> ~~~\n",
        ] {
            let text = format!("{example}after<meta:self_note>secret</meta:self_note>done");
            assert_hidden_meta_stream_splits(&text, &format!("{example}afterdone"), "secret");
        }
    }

    #[test]
    fn hidden_meta_container_prefixes_do_not_turn_prose_into_fences() {
        for prose in [
            "ordinary > ~~~\n",
            "-~~~\n",
            "1.~~~\n",
            "    ~~~\n",
            ">     ~~~\n",
            "> ~~~text\n> literal\n",
            "- ~~~text\n  literal\n",
        ] {
            let text = format!("{prose}<meta:self_note>secret</meta:self_note>after");
            assert_hidden_meta_stream_splits(&text, &format!("{prose}after"), "secret");
        }
    }

    #[test]
    fn hidden_meta_container_fences_preserve_crlf_stream_splits() {
        let text = "> ~~~text\r\n> <meta:self_note>literal</meta:self_note>\r\n> ~~~\r\nafter";
        for cut in 0..=text.len() {
            let expected = normalize_stream_text(text[..cut].to_string())
                + &normalize_stream_text(text[cut..].to_string());
            assert_eq!(
                hidden_meta_stream_output(&[&text[..cut], &text[cut..]]),
                (expected, String::new())
            );
        }
    }

    #[test]
    fn hidden_meta_crlf_code_markers_survive_every_stream_split() {
        let text = "~~~text\r\n<meta:self_note>literal</meta:self_note>\r\n~~~\r\nafter";
        for cut in 0..=text.len() {
            // Preserve the existing per-delta CRLF normalization, including a
            // CR and LF received separately, while checking code protection.
            let expected = normalize_stream_text(text[..cut].to_string())
                + &normalize_stream_text(text[cut..].to_string());
            assert_eq!(
                hidden_meta_stream_output(&[&text[..cut], &text[cut..]]),
                (expected, String::new())
            );
        }
    }

    #[test]
    fn hidden_meta_escaped_backticks_do_not_open_code() {
        assert_hidden_meta_stream_splits(
            "\\`before<meta:self_note>secret</meta:self_note>after",
            "\\`beforeafter",
            "secret",
        );
        assert_hidden_meta_stream_splits(
            "\\\\`<meta:self_note>`after",
            "\\\\`<meta:self_note>`after",
            "",
        );
        assert_hidden_meta_stream_splits(
            "`\\`<meta:self_note>secret</meta:self_note>",
            "`\\`",
            "secret",
        );
    }

    #[test]
    fn hidden_meta_markdown_state_is_separate_for_reasoning_and_content() {
        let mut state = HiddenMetaParseState::default();
        let mut events = Vec::new();
        for (text, thinking) in [
            ("`reasoning", true),
            ("<meta:self_note>secret</meta:self_note>answer", false),
        ] {
            push_text_with_hidden_meta(
                &mut events,
                text.to_string(),
                thinking,
                "<meta:self_note>",
                "</meta:self_note>",
                &mut state,
            );
        }
        assert_eq!(
            events,
            vec![
                StreamTextEvent::AppendThinking("`reasoning".to_string()),
                StreamTextEvent::AppendHiddenMeta("secret".to_string()),
                StreamTextEvent::AppendContent("answer".to_string())
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
            &mut super::super::inline_recovery::InlineMarkupNormalizer::new(),
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
            &mut super::super::inline_recovery::InlineMarkupNormalizer::new(),
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
            &mut super::super::inline_recovery::InlineMarkupNormalizer::new(),
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

    #[test]
    fn split_dsml_marker_across_chunks_is_recovered_not_leaked() {
        // 回归：全角 DSML marker（`<｜｜DSML｜｜…>`）被拆到两个 delta chunk 边界时，
        // 逐 chunk 无状态归一化无法还原，会把原始 marker 泄漏成正文且解析失效。
        // 有状态的 InlineMarkupNormalizer 应把半截 marker 暂存到下一个 chunk 再归一化。
        let full = "<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name=\"read_file\"><｜｜DSML｜｜parameter name=\"file_path\" string=\"true\">/tmp/x</｜｜DSML｜｜parameter></｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>";
        // 在多个字符边界切开，覆盖 marker 中间被截断的各种位置。
        for cut_char in [1usize, 3, 5, 8, 20] {
            let Some((cut, _)) = full.char_indices().nth(cut_char) else {
                continue;
            };
            let (a, b) = full.split_at(cut);

            let mut thinking_open = false;
            let mut hidden_meta_parse = HiddenMetaParseState::default();
            let mut streamer = InternalToolCallStreamer::new();
            let mut hermes_streamer = HermesXmlToolCallStreamer::new();
            let mut anthropic_streamer = AnthropicXmlToolCallStreamer::new();
            let mut bare_xml_streamer = BareXmlToolCallStreamer::new();
            let mut inline_markup_normalizer =
                super::super::inline_recovery::InlineMarkupNormalizer::new();

            let mut visible = String::new();
            let mut tool_events_total = Vec::new();
            for piece in [a, b] {
                let chunk = StreamChunk {
                    choices: vec![StreamChoice {
                        delta: StreamDelta {
                            content: piece.to_string(),
                            reasoning_content: String::new(),
                            reasoning_details: String::new(),
                            tool_calls: Vec::new(),
                        },
                        finish_reason: None,
                        ..Default::default()
                    }],
                    ..Default::default()
                };
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
                    &mut inline_markup_normalizer,
                );
                for ev in &events {
                    if let StreamTextEvent::AppendContent(t) = ev {
                        visible.push_str(t);
                    }
                }
                tool_events_total.extend(tool_events);
            }
            assert!(
                !visible.contains("DSML") && !visible.contains("invoke"),
                "DSML markup leaked to visible content (cut={cut_char}): {visible:?}"
            );
            assert!(
                tool_events_total
                    .iter()
                    .any(|ev| matches!(ev, InternalToolCallStreamEvent::Begin(name) if name == "read_file")),
                "tool call should be recovered from split chunks (cut={cut_char}): {tool_events_total:?}"
            );
        }
    }

    #[test]
    fn spaced_fullwidth_dsml_marker_is_recovered_not_leaked() {
        // deepseek-v4-flash 有时把 DSML 命名空间分隔符输出成带空格的全角竖线，
        // 如 `<｜｜DSML ｜ invoke ...>`。归一化应仍把它还原为普通 `<invoke ...>`。
        let full = "<｜｜DSML ｜ tool_calls>\n<｜｜DSML ｜ invoke name=\"mcp_browser_get_text\">\n\n</｜｜DSML ｜ invoke>\n</｜｜DSML ｜ tool_calls>";

        let mut thinking_open = false;
        let mut hidden_meta_parse = HiddenMetaParseState::default();
        let mut streamer = InternalToolCallStreamer::new();
        let mut hermes_streamer = HermesXmlToolCallStreamer::new();
        let mut anthropic_streamer = AnthropicXmlToolCallStreamer::new();
        let mut bare_xml_streamer = BareXmlToolCallStreamer::new();
        let mut inline_markup_normalizer =
            super::super::inline_recovery::InlineMarkupNormalizer::new();

        let mut visible = String::new();
        let mut tool_events_total = Vec::new();
        for piece in [
            "<｜｜DSML ｜ tool",
            "_calls>\n<｜｜DSML ｜ invoke name=\"mcp_browser_get_text\">\n\n",
            "</｜｜DSML ｜ invoke>\n</｜｜DSML ｜ tool_calls>",
        ] {
            let chunk = StreamChunk {
                choices: vec![StreamChoice {
                    delta: StreamDelta {
                        content: piece.to_string(),
                        reasoning_content: String::new(),
                        reasoning_details: String::new(),
                        tool_calls: Vec::new(),
                    },
                    finish_reason: None,
                    ..Default::default()
                }],
                ..Default::default()
            };
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
                &mut inline_markup_normalizer,
            );
            for ev in &events {
                if let StreamTextEvent::AppendContent(t) = ev {
                    visible.push_str(t);
                }
            }
            tool_events_total.extend(tool_events);
        }
        assert!(
            !visible.contains("DSML") && !visible.contains("invoke"),
            "DSML markup leaked to visible content: {visible:?}; source={full:?}"
        );
        assert_eq!(
            tool_events_total,
            vec![
                InternalToolCallStreamEvent::Begin("mcp_browser_get_text".to_string()),
                InternalToolCallStreamEvent::Args("{}".to_string()),
                InternalToolCallStreamEvent::End,
            ]
        );
    }

    #[test]
    fn spaced_ascii_dsml_marker_is_recovered_not_leaked() {
        // The same model can emit ASCII bars with a space after `<`:
        // `< | | DSML | | tool_calls>`. The old guard recognized only `<|` /
        // `<｜`, so this form reached the visible body unparsed.
        let full = "< | | DSML | | calls>\n\
                    < | | DSML | | invoke name=\"mcp_browser_get_text\">\n\n\
                    </ | | DSML | | invoke>\n\
                    </ | | DSML | | calls>";
        let pieces = [
            "< | | DSML | | calls>\n< | | DSML | | invoke name=",
            "\"mcp_browser_get_text\">\n\n</ | | DSML | | invoke>\n</ | | DSML | | calls>",
        ];

        let mut thinking_open = false;
        let mut hidden_meta_parse = HiddenMetaParseState::default();
        let mut streamer = InternalToolCallStreamer::new();
        let mut hermes_streamer = HermesXmlToolCallStreamer::new();
        let mut anthropic_streamer = AnthropicXmlToolCallStreamer::new();
        let mut bare_xml_streamer = BareXmlToolCallStreamer::new();
        let mut inline_markup_normalizer =
            super::super::inline_recovery::InlineMarkupNormalizer::new();

        let mut visible = String::new();
        let mut tool_events_total = Vec::new();
        for piece in pieces {
            let chunk = StreamChunk {
                choices: vec![StreamChoice {
                    delta: StreamDelta {
                        content: piece.to_string(),
                        reasoning_content: String::new(),
                        reasoning_details: String::new(),
                        tool_calls: Vec::new(),
                    },
                    finish_reason: None,
                    ..Default::default()
                }],
                ..Default::default()
            };
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
                &mut inline_markup_normalizer,
            );
            for ev in &events {
                if let StreamTextEvent::AppendContent(t) = ev {
                    visible.push_str(t);
                }
            }
            tool_events_total.extend(tool_events);
        }
        assert!(
            !visible.contains("DSML") && !visible.contains("invoke") && !visible.contains("calls"),
            "spaced ASCII DSML markup leaked to visible content: {visible:?}; source={full:?}"
        );
        assert_eq!(
            tool_events_total,
            vec![
                InternalToolCallStreamEvent::Begin("mcp_browser_get_text".to_string()),
                InternalToolCallStreamEvent::Args("{}".to_string()),
                InternalToolCallStreamEvent::End,
            ]
        );
    }

    #[test]
    fn truncated_spaced_dsml_marker_is_dropped_at_flush() {
        // The reported second line is truncated at `invoke name=`. A protocol
        // prefix that never closes must not be flushed as answer text.
        let mut normalizer = super::super::inline_recovery::InlineMarkupNormalizer::new();
        let emitted = normalizer.push("< | | DSML | | calls>\n< | | DSML | | invoke name=");
        assert_eq!(emitted, "<calls>\n");

        let mut anthropic = AnthropicXmlToolCallStreamer::new();
        let (cleaned, events) = anthropic.push(&emitted);
        assert!(
            cleaned.trim().is_empty(),
            "wrapper tag must be suppressed: {cleaned:?}"
        );
        assert!(events.is_empty());

        assert!(
            normalizer.flush().is_empty(),
            "truncated protocol marker must be dropped at stream end"
        );
    }

    #[test]
    fn ordinary_bar_led_text_is_not_flushed_away() {
        // A bar after `<` is not proof of a protocol marker; ordinary prose
        // such as a comparison must survive the stream-end flush.
        let mut normalizer = super::super::inline_recovery::InlineMarkupNormalizer::new();
        assert_eq!(normalizer.push("compare < |x|"), "compare ");
        assert_eq!(normalizer.flush(), "< |x|");
    }

    #[test]
    fn incomplete_marker_before_another_tag_is_still_held() {
        // Holding from the last `<` would emit the earlier incomplete marker.
        let mut normalizer = super::super::inline_recovery::InlineMarkupNormalizer::new();
        assert_eq!(normalizer.push("< | | DSML | | invoke name= <"), "");
        assert_eq!(normalizer.flush(), "");
    }

    #[test]
    fn unicode_whitespace_in_dsml_marker_is_normalized() {
        assert_eq!(
            super::super::inline_recovery::normalize_inline_tool_call_markup(
                "<　|　|　DSML　|　|　calls>"
            ),
            "<calls>"
        );
    }
}

/// Tracks only Markdown boundaries that make self-note tags literal text.
/// Delimiter runs remain visible immediately; their meaning is resolved when
/// the next character arrives, even when that character is in another delta.
/// Unmatched code delimiters conservatively keep subsequent tags literal.
#[derive(Default)]
pub(super) struct HiddenMetaMarkdownState {
    fence: Option<HiddenMetaFence>,
    inline_backticks: usize,
    delimiter_run: Option<(char, usize, bool)>,
    line_indent: usize,
    line_has_text: bool,
    container_prefix: HiddenMetaContainerPrefix,
    container_indent: usize,
    quote_depth: usize,
    indented: bool,
    escaped: bool,
    closing_fence: bool,
}

#[derive(Clone, Copy)]
struct HiddenMetaFence {
    marker: char,
    length: usize,
    container_indent: usize,
    quote_depth: usize,
}

#[derive(Default, Clone, Copy)]
enum HiddenMetaContainerPrefix {
    #[default]
    Start,
    QuotePadding,
    ListMarker {
        width: usize,
    },
    OrderedDigits {
        width: usize,
        digits: usize,
    },
    OrderedMarker {
        width: usize,
    },
}

impl HiddenMetaMarkdownState {
    /// Recognize only line-start container prefixes. List markers require
    /// whitespace, and each container permits at most three spaces before it.
    fn push_line_prefix(&mut self, ch: char) -> bool {
        if self.line_has_text {
            return false;
        }
        match self.container_prefix {
            HiddenMetaContainerPrefix::QuotePadding => {
                self.container_prefix = HiddenMetaContainerPrefix::Start;
                if ch == ' ' || ch == '\t' {
                    return false;
                }
            }
            HiddenMetaContainerPrefix::ListMarker { width }
            | HiddenMetaContainerPrefix::OrderedMarker { width } => {
                if ch == ' ' || ch == '\t' {
                    self.container_indent += width + if ch == '\t' { 4 - width % 4 } else { 1 };
                    self.line_indent = 0;
                    self.container_prefix = HiddenMetaContainerPrefix::Start;
                    return false;
                }
                self.line_has_text = true;
                self.leave_finished_container();
                return false;
            }
            HiddenMetaContainerPrefix::OrderedDigits { width, digits } => {
                if ch.is_ascii_digit() && digits < 9 {
                    self.container_prefix = HiddenMetaContainerPrefix::OrderedDigits {
                        width: width + 1,
                        digits: digits + 1,
                    };
                    return false;
                }
                if ch == '.' || ch == ')' {
                    self.container_prefix =
                        HiddenMetaContainerPrefix::OrderedMarker { width: width + 1 };
                    return false;
                }
                self.line_has_text = true;
                self.leave_finished_container();
                return false;
            }
            HiddenMetaContainerPrefix::Start => {}
        }

        let fence_indent = self.fence.map_or(0, |fence| fence.container_indent);
        let can_start_container =
            self.line_indent <= 3 + fence_indent && !self.indented && self.inline_backticks == 0;
        match ch {
            ' ' => self.line_indent += 1,
            '\t' => self.line_indent += 4 - self.line_indent % 4,
            '>' if can_start_container
                && self
                    .fence
                    .is_none_or(|fence| self.quote_depth < fence.quote_depth) =>
            {
                self.container_indent += self.line_indent.min(fence_indent);
                self.quote_depth += 1;
                self.line_indent = 0;
                self.container_prefix = HiddenMetaContainerPrefix::QuotePadding;
                return false;
            }
            '-' | '+' | '*' if can_start_container && self.fence.is_none() => {
                self.container_prefix = HiddenMetaContainerPrefix::ListMarker {
                    width: self.line_indent + 1,
                };
                return false;
            }
            '0'..='9' if can_start_container && self.fence.is_none() => {
                self.container_prefix = HiddenMetaContainerPrefix::OrderedDigits {
                    width: self.line_indent + 1,
                    digits: 1,
                };
                return false;
            }
            _ => {
                self.line_has_text = true;
                self.leave_finished_container();
                return self.line_indent
                    <= 3 + self.fence.map_or(0, |fence| fence.container_indent);
            }
        }
        if self.line_indent >= 4 && self.fence.is_none() && self.inline_backticks == 0 {
            self.indented = true;
        }
        false
    }

    fn leave_finished_container(&mut self) {
        if let Some(fence) = self.fence
            && (self.quote_depth < fence.quote_depth
                || self.container_indent + self.line_indent < fence.container_indent)
        {
            // A fenced block cannot continue after its quote or list container ends.
            self.fence = None;
            self.closing_fence = false;
            self.indented = self.line_indent >= 4;
        }
    }

    fn open_fence(&mut self, marker: char, length: usize) {
        self.fence = Some(HiddenMetaFence {
            marker,
            length,
            container_indent: self.container_indent,
            quote_depth: self.quote_depth,
        });
    }

    fn push(&mut self, ch: char) -> bool {
        if let Some((marker, length, line_start)) = self.delimiter_run
            && ch != marker
        {
            self.delimiter_run = None;
            if let Some(fence) = self.fence {
                self.closing_fence = line_start && marker == fence.marker && length >= fence.length;
            } else if marker == '`' {
                if self.inline_backticks != 0 {
                    if length == self.inline_backticks {
                        self.inline_backticks = 0;
                    }
                } else if line_start && length >= 3 {
                    self.open_fence(marker, length);
                } else {
                    self.inline_backticks = length;
                }
            } else if line_start && length >= 3 {
                self.open_fence(marker, length);
            }
        }

        let in_code = self.fence.is_some() || self.inline_backticks != 0 || self.indented;
        if ch == '\n' || ch == '\r' {
            if self.closing_fence {
                self.fence = None;
            }
            self.line_indent = 0;
            self.line_has_text = false;
            self.container_prefix = HiddenMetaContainerPrefix::Start;
            self.container_indent = 0;
            self.quote_depth = 0;
            self.indented = false;
            self.escaped = false;
            self.closing_fence = false;
            return in_code;
        }

        let line_start = self.push_line_prefix(ch);
        let in_code = self.fence.is_some() || self.inline_backticks != 0 || self.indented;

        if let Some((marker, length, _)) = &mut self.delimiter_run {
            if ch == *marker {
                *length += 1;
            }
        } else if !self.indented
            && (!self.escaped || in_code)
            && (ch == '`' || ch == '~')
            && self.fence.map_or_else(
                || ch == '`' || (self.inline_backticks == 0 && line_start),
                |fence| ch == fence.marker && line_start,
            )
        {
            self.delimiter_run = Some((ch, 1, line_start));
        }

        if self.closing_fence && ch != ' ' && ch != '\t' {
            // A fence closes only if the rest of its line is whitespace.
            self.closing_fence = false;
        }
        self.escaped = !in_code && ch == '\\' && !self.escaped;
        in_code || self.indented
    }
}

pub(super) fn flush_hidden_meta(
    events: &mut Vec<StreamTextEvent>,
    hidden_begin: &str,
    hidden_end: &str,
    state: &mut HiddenMetaParseState,
) {
    if state.hidden_begin_match != 0 {
        let text = hidden_begin
            .chars()
            .take(state.hidden_begin_match)
            .collect();
        events.push(if state.hidden_begin_thinking {
            StreamTextEvent::AppendThinking(text)
        } else {
            StreamTextEvent::AppendContent(text)
        });
        state.hidden_begin_match = 0;
    }
    if state.hidden_end_match != 0 {
        events.push(StreamTextEvent::AppendHiddenMeta(
            hidden_end.chars().take(state.hidden_end_match).collect(),
        ));
        state.hidden_end_match = 0;
    }
}

pub(super) fn push_text_with_hidden_meta(
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
            let markdown = if is_thinking {
                &mut hidden_meta_parse.thinking_markdown
            } else {
                &mut hidden_meta_parse.content_markdown
            };
            let in_code = markdown.push(ch);
            if hidden_meta_parse.hidden_begin_match > 0
                && (in_code || ch != hb[hidden_meta_parse.hidden_begin_match])
            {
                visible.extend(hb.iter().take(hidden_meta_parse.hidden_begin_match));
                hidden_meta_parse.hidden_begin_match = 0;
            }
            if in_code {
                visible.push(ch);
                continue;
            }
            if hidden_meta_parse.hidden_begin_match < hb.len()
                && ch == hb[hidden_meta_parse.hidden_begin_match]
            {
                if hidden_meta_parse.hidden_begin_match == 0 {
                    hidden_meta_parse.hidden_begin_thinking = is_thinking;
                }
                hidden_meta_parse.hidden_begin_match += 1;
                if hidden_meta_parse.hidden_begin_match == hb.len() {
                    flush_visible(events, &mut visible, is_thinking);
                    hidden_meta_parse.hidden_open = true;
                    hidden_meta_parse.hidden_begin_match = 0;
                }
            } else {
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
                hidden.extend(he.iter().take(hidden_meta_parse.hidden_end_match));
                hidden_meta_parse.hidden_end_match = 0;
            }
            if he.first() == Some(&ch) {
                hidden_meta_parse.hidden_end_match = 1;
            } else {
                hidden.push(ch);
            }
        }
    }

    flush_visible(events, &mut visible, is_thinking);
    flush_hidden(events, &mut hidden);
}

pub(super) fn normalize_stream_text(text: String) -> String {
    // Most chunks contain no bare '\r': short-circuit to avoid two unconditional
    // whole-string replace allocations.
    if !text.contains('\r') {
        return text;
    }
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
