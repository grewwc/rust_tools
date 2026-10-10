use super::*;

#[test]
fn standalone_stream_marker_requires_exact_control_line() {
    assert!(is_standalone_stream_marker(
        "\n╭─ thinking\n",
        "╭─ thinking"
    ));
    assert!(is_standalone_stream_marker(
        "\n╰─ done thinking\n",
        "╰─ done thinking"
    ));
    assert!(!is_standalone_stream_marker(
        "reasoning mentions ╭─ thinking literally",
        "╭─ thinking"
    ));
    assert!(!is_standalone_stream_marker(
        "prefix\n╰─ done thinking\nsuffix",
        "╰─ done thinking"
    ));
}

#[test]
fn snapshot_done_chunk_does_not_duplicate_already_streamed_prefix() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::opencode_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"hello wor"}"#,
    )
    .unwrap();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::opencode_adapter(),
        Some("response.output_text.done"),
        r#"{"text":"hello world"}"#,
    )
    .unwrap();

    assert_eq!(current_history, "hello world");
    assert_eq!(state.content.assistant_text, "hello world");
}

#[test]
fn reasoning_item_added_done_same_id_keeps_full_payload() {
    // The gateway re-delivers .added (partial payload) and .done (complete payload) for the same
    // reasoning resource: same id, different encrypted_content lengths (a real partial payload of
    // >=256 delivered via .added is captured too). The accumulator must converge by id and keep the
    // longest payload, otherwise the same resource id appearing twice triggers modelhub 400 (-4003 Duplicate item found).
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    let added_payload = format!(
        r#"{{"output_index":0,"item":{{"type":"reasoning","id":"rs_same","encrypted_content":"{}"}}}}"#,
        "A".repeat(300)
    );
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_item.added"),
        &added_payload,
    )
    .unwrap();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_item.done"),
        r#"{"output_index":0,"item":{"type":"reasoning","id":"rs_same","encrypted_content":"FULL_LONGER_PAYLOAD"}}"#,
    )
    .unwrap();

    assert_eq!(
        state.content.reasoning_items.len(),
        1,
        "同 id 的 reasoning item 必须收敛为一项"
    );
    assert_eq!(
        state.content.reasoning_items[0]
            .get("encrypted_content")
            .and_then(serde_json::Value::as_str),
        Some("FULL_LONGER_PAYLOAD"),
        "必须保留最长（完整）载荷"
    );
}

#[test]
fn tool_call_snapshot_done_does_not_duplicate_already_streamed_prefix() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            provider::openai_adapter(),
            Some("response.output_item.added"),
            r#"{"output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"write_file","arguments":""}}"#,
        )
        .unwrap();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.function_call_arguments.delta"),
        r#"{"output_index":0,"delta":"{\"path\":\"a"}"#,
    )
    .unwrap();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.function_call_arguments.done"),
        r#"{"output_index":0,"arguments":"{\"path\":\"abc\"}"}"#,
    )
    .unwrap();

    let builder = state.content.tool_calls_map.get_ref(&0).unwrap();
    assert_eq!(builder.id, "call_1");
    assert_eq!(builder.function_name, "write_file");
    assert_eq!(builder.arguments, "{\"path\":\"abc\"}");
}
