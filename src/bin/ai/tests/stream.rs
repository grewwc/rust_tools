//! Stream parsing / rendering and reasoning-fragment merge tests.

use serde_json::Value;

use super::super::request::{StreamChoice, StreamChunk, StreamDelta};
use super::super::*;

#[test]
fn thinking_chunks_are_wrapped_once() {
    colored::control::set_override(false);
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
    let text =
        stream::extract_chunk_text(&chunk, "<thinking>", "<end thinking>", &mut thinking_open);
    assert_eq!(text, "\n<thinking>\nstep one");
    assert!(thinking_open);

    let chunk = StreamChunk {
        choices: vec![StreamChoice {
            delta: StreamDelta {
                content: "final".to_string(),
                reasoning_content: String::new(),
                reasoning_details: String::new(),
                tool_calls: Vec::new(),
            },
            finish_reason: None,
            ..Default::default()
        }],
        ..Default::default()
    };
    let text =
        stream::extract_chunk_text(&chunk, "<thinking>", "<end thinking>", &mut thinking_open);
    assert_eq!(text, "\n<end thinking>\nfinal");
    assert!(!thinking_open);
}

#[test]
fn table_preview_lines_are_not_double_printed_after_live_emit() {
    let mut renderer = stream::MarkdownStreamRenderer::new_with_tty(true);

    // Streaming table rendering: the header row enters silent buffering (waiting for a separator row to confirm a table); while
    // buffering, a "generating table" placeholder preview is emitted instead of echoing the raw text — otherwise the
    // final table would print the rendered header twice.
    let header_out = renderer.consume_line("| name | value |", false);
    assert!(header_out.contains("\x1b["));
    assert!(!header_out.contains("| name | value |"));

    // Once the separator row confirms a table, table content keeps buffering silently (no more echoing raw rows).
    let sep_out = renderer.consume_line("| --- | --- |", true);
    assert_eq!(sep_out, "");
    let row_out = renderer.consume_line("| foo | bar |", true);
    assert_eq!(row_out, "");

    // When the table ends (non-table row "done"), clear the placeholder preview first, then render the complete table in one shot.
    let end_out = renderer.consume_line("done", false);
    // ANSI sequence that clears the placeholder preview
    assert!(end_out.contains("\x1b[1A"));
    // The rendered table contains header and data, but the raw markdown text does not appear separately
    assert!(end_out.contains("name"));
    assert!(end_out.contains("value"));
    assert!(end_out.contains("foo"));
    assert!(end_out.contains("bar"));
    assert!(!end_out.contains("| name | value |"));
    assert!(!end_out.contains("| --- | --- |"));
    assert!(!end_out.contains("| foo | bar |"));
    // Plain text after the table outputs normally
    assert!(end_out.contains("done"));
}

#[test]
fn table_live_preview_detection_requires_table_like_content() {
    assert!(stream::line_looks_like_table_preview("| col1 | col2"));
    assert!(stream::line_looks_like_table_preview("  col1 | col2"));
    assert!(!stream::line_looks_like_table_preview("plain text"));
    assert!(!stream::line_looks_like_table_preview("```| not table"));
}

#[test]
fn math_frac_renders_with_nested_braces() {
    let mut renderer = stream::MarkdownStreamRenderer::new_with_tty(true);
    // Block math lines now buffer: lines between `$$`/`\[` and `$$`/`\]` accumulate first and render
    // in one shot when the closing delimiter arrives, preventing the live preview from emitting raw TeX and then appending the Unicode result after a newline (duplicate output).
    assert_eq!(renderer.consume_line("$$", false), "");
    assert_eq!(
        renderer.consume_line(r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}", false),
        ""
    );
    assert_eq!(
        renderer.consume_line(r"y = \frac{1}{\frac{2}{3}}", false),
        ""
    );

    let out = renderer.consume_line("$$", false);
    assert!(out.contains("x ="));
    assert!(out.contains("(-b ± √(b² - 4ac))/2a"));
    assert!(!out.contains("\\frac"));
    assert!(out.contains("y ="));
    assert!(out.contains("1/(2/3)"));
}

#[test]
fn math_renderer_preserves_longer_commands_and_literal_braces() {
    let mut renderer = stream::MarkdownStreamRenderer::new_with_tty(true);
    // Buffered block math lines render in one shot at the closing delimiter (see above).
    assert_eq!(renderer.consume_line("$$", false), "");
    assert_eq!(
        renderer.consume_line(
            r"\leftarrow \rightarrow \leftrightarrow \subseteq \supseteq \sqrt[3]{x} \sqrt[5]{y} \left\{a\right\}",
            false,
        ),
        ""
    );

    let out = renderer.consume_line("$$", false);
    assert!(out.contains("←"));
    assert!(out.contains("→"));
    assert!(out.contains("↔"));
    assert!(out.contains("⊆"));
    assert!(out.contains("⊇"));
    assert!(out.contains("∛(x)"));
    assert!(out.contains("√[5](y)"));
    assert!(out.contains("{a}"));
    assert!(!out.contains("arrow"));
    assert!(!out.contains("⊂eq"));
    assert!(!out.contains("⊃eq"));
}

#[test]
fn math_renderer_maps_mathbb_and_preserves_unknown_commands() {
    let mut renderer = stream::MarkdownStreamRenderer::new_with_tty(true);
    // Buffered block math lines render in one shot at the closing delimiter (see above).
    assert_eq!(renderer.consume_line("$$", false), "");
    assert_eq!(
        renderer.consume_line(r"\mathbb{R} \customcmd \alpha", false),
        ""
    );

    let out = renderer.consume_line("$$", false);
    assert!(out.contains("ℝ"));
    assert!(out.contains(r"\customcmd"));
    assert!(out.contains("α"));
}

#[test]
fn stream_chunk_accepts_null_content() {
    let payload = r#"{"choices":[{"delta":{"content":null,"reasoning_content":null}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices.len(), 1);
    assert_eq!(parsed.choices[0].delta.content, "");
    assert_eq!(parsed.choices[0].delta.reasoning_content, "");
}

#[test]
fn stream_chunk_accepts_reasoning_alias() {
    // OpenCode/OpenRouter providers often stream reasoning under `delta.reasoning`.
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning":"step by step"}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices.len(), 1);
    assert_eq!(parsed.choices[0].delta.content, "");
    assert_eq!(parsed.choices[0].delta.reasoning_content, "step by step");
}

#[test]
fn stream_chunk_accepts_reasoning_text_alias() {
    // Some provider shims expose the same field as `delta.reasoning_text`.
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning_text":"step by step"}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices.len(), 1);
    assert_eq!(parsed.choices[0].delta.content, "");
    assert_eq!(parsed.choices[0].delta.reasoning_content, "step by step");
}

#[test]
fn stream_chunk_ignores_structured_reasoning_object_without_text() {
    let payload =
        r#"{"choices":[{"delta":{"content":"","reasoning":{"confidence":0.9,"thinking":true}}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.reasoning_content, "");
}

#[test]
fn stream_chunk_extracts_text_from_reasoning_object() {
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning":{"type":"thinking","text":"step by step"}}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.reasoning_content, "step by step");
}

#[test]
fn stream_chunk_extracts_nested_reasoning_delta_text() {
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning":{"type":"reasoning_text","delta":"No"}}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.reasoning_content, "No");
}

#[test]
fn stream_chunk_extracts_reasoning_summary_object() {
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning":{"type":"summary","summary":[{"text":"step 1"},{"text":" step 2"}]}}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.reasoning_content, "step 1 step 2");
}

#[test]
fn stream_chunk_ignores_bool_and_number_reasoning() {
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning":42}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.reasoning_content, "");
}

#[test]
fn stream_chunk_merges_reasoning_details_into_reasoning_content() {
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning_details":[{"text":"step 1"},{"text":" step 2"}]}}]}"#;
    let mut parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.reasoning_content, "");
    parsed.merge_reasoning();
    assert_eq!(parsed.choices[0].delta.reasoning_content, "step 1 step 2");
}

#[test]
fn stream_chunk_merges_reasoning_details_prefix_with_punctuation_continuation() {
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning":", that's not really necessary.","reasoning_details":[{"delta":"No"}]}}]}"#;
    let mut parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(
        parsed.choices[0].delta.reasoning_content,
        ", that's not really necessary."
    );
    parsed.merge_reasoning();
    assert_eq!(
        parsed.choices[0].delta.reasoning_content,
        "No, that's not really necessary."
    );
}

#[test]
fn stream_chunk_reasoning_content_takes_priority_over_details() {
    let payload = r#"{"choices":[{"delta":{"content":"","reasoning":"from reasoning field","reasoning_details":[{"text":"from details"}]}}]}"#;
    let mut parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(
        parsed.choices[0].delta.reasoning_content,
        "from reasoning field"
    );
    parsed.merge_reasoning();
    assert_eq!(
        parsed.choices[0].delta.reasoning_content,
        "from reasoning field"
    );
}

#[test]
fn merge_reasoning_fragments_stripped_overlap() {
    use crate::ai::request::merge_reasoning_fragments;
    assert_eq!(
        merge_reasoning_fragments("I think", " think this is right"),
        "I think this is right"
    );
}

#[test]
fn merge_reasoning_fragments_cjk_punctuation_continuation() {
    use crate::ai::request::merge_reasoning_fragments;
    assert_eq!(
        merge_reasoning_fragments("是的", "，这很重要"),
        "是的，这很重要"
    );
    assert_eq!(merge_reasoning_fragments("注意", "！危险"), "注意！危险");
}

#[test]
fn merge_reasoning_fragments_english_contraction_continuation() {
    use crate::ai::request::merge_reasoning_fragments;
    assert_eq!(
        merge_reasoning_fragments("It is", "n't necessary"),
        "It isn't necessary"
    );
    assert_eq!(
        merge_reasoning_fragments("I", "'ve already checked"),
        "I've already checked"
    );
    assert_eq!(
        merge_reasoning_fragments("They", "'re coming"),
        "They're coming"
    );
}

#[test]
fn merge_reasoning_fragments_no_false_positive_on_independent_sentence() {
    use crate::ai::request::merge_reasoning_fragments;
    let result = merge_reasoning_fragments("First step done", "Second step begins");
    assert_eq!(result, "Second step begins");
}

#[test]
fn merge_reasoning_fragments_ellipsis_continuation() {
    use crate::ai::request::merge_reasoning_fragments;
    assert_eq!(
        merge_reasoning_fragments("等等", "…还有更多"),
        "等等…还有更多"
    );
}

#[test]
fn stream_chunk_opencode_structured_content_extracts_text() {
    let payload = r#"{"choices":[{"delta":{"content":[{"type":"output_text","text":"hi"}]}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.content, "hi");
}

#[test]
fn stream_tool_call_maps_type_field() {
    let payload = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"mcp_demo_get_time","arguments":""}}]}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    let call = &parsed.choices[0].delta.tool_calls[0];
    assert_eq!(call.id, "call_1");
    assert_eq!(call.tool_type, "function");
    assert_eq!(call.function.name, "mcp_demo_get_time");
}

#[test]
fn stream_tool_call_defaults_when_nulls_present() {
    let payload = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":null,"type":null,"function":{"name":null,"arguments":null}}]}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    let call = &parsed.choices[0].delta.tool_calls[0];
    assert_eq!(call.id, "");
    assert_eq!(call.tool_type, "");
    assert_eq!(call.function.name, "");
    assert_eq!(call.function.arguments, "");
}

#[test]
fn stream_chunk_accepts_structured_content_arrays() {
    let payload = r#"{"choices":[{"delta":{"content":[{"type":"output_text","text":"hel"},{"type":"output_text","text":"lo"}]}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed.choices[0].delta.content, "hello");
}

#[test]
fn stream_tool_call_accepts_object_arguments() {
    let payload = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"apply_patch","arguments":{"file":"a.rs","patch":"..."}}}]}}]}"#;
    let parsed: StreamChunk = serde_json::from_str(payload).unwrap();
    let args: Value =
        serde_json::from_str(&parsed.choices[0].delta.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["file"], "a.rs");
    assert_eq!(args["patch"], "...");
}
