use super::*;

#[test]
fn closing_thinking_marker_starts_on_new_line_when_reasoning_line_is_open() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();

    state
        .render
        .markdown
        .write_chunk("still thinking", true)
        .unwrap();
    let mut content = format!("{}\nfinal", markers.end_thinking_tag);
    if content.starts_with(&markers.end_thinking_tag) && state.render.markdown.has_unfinished_line()
    {
        content.insert(0, '\n');
    }

    assert_eq!(content, format!("\n{}\nfinal", markers.end_thinking_tag));
}

#[test]
fn closing_thinking_marker_keeps_compact_spacing_when_already_at_line_start() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();

    state
        .render
        .markdown
        .write_chunk("still thinking\n", true)
        .unwrap();
    let mut content = format!("{}\nfinal", markers.end_thinking_tag);
    normalize_end_thinking_boundary(&mut content, &markers, &state.render.markdown);

    assert_eq!(content, format!("{}\nfinal", markers.end_thinking_tag));
}

#[test]
fn tool_call_boundary_closes_thinking_on_a_fresh_line() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();

    state
        .render
        .markdown
        .write_chunk("still thinking", true)
        .unwrap();

    assert_eq!(
        format_end_thinking_line(&markers, &state.render.markdown),
        format!("\n{}\n", markers.end_thinking_tag)
    );
}

#[test]
fn snapshot_content_only_appends_missing_suffix() {
    assert_eq!(unseen_suffix("hello wor", "hello world"), "ld");
    assert_eq!(unseen_suffix("hello world", "hello world"), "");
    assert_eq!(unseen_suffix("hello world", "\n\nhello world"), "");
    assert_eq!(unseen_suffix("hello world", "\n\nhello world!"), "!");
    assert_eq!(unseen_suffix("prefix", "suffix"), "suffix");
}

#[test]
fn tool_call_render_chunk_only_streams_unprinted_suffix() {
    let mut builder = ToolCallBuilder::default();

    builder.arguments.push_str("{\"patch\":\"a");
    assert!(take_tool_call_render_chunk(None, 0, &mut builder).is_none());

    builder.function_name = "apply_patch".to_string();
    let first = take_tool_call_render_chunk(None, 0, &mut builder).unwrap();
    assert!(first.open_line);
    assert_eq!(first.function_name, "apply_patch");
    assert_eq!(first.arguments, "{\"patch\":\"a");

    builder.arguments.push('你');
    let second = take_tool_call_render_chunk(Some(0), 0, &mut builder).unwrap();
    assert!(!second.open_line);
    assert_eq!(second.arguments, "你");
}
