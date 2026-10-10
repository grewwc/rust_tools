use super::*;

#[test]
fn response_completed_event_does_not_block_late_snapshot_text() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    let outcome = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.completed"),
        r#"{"status":"completed"}"#,
    )
    .unwrap();
    assert!(!outcome.should_stop);
    assert!(!outcome.meaningful_progress);

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.done"),
        r#"{"text":"hello world"}"#,
    )
    .unwrap();

    assert_eq!(current_history, "hello world");
    assert_eq!(state.content.assistant_text, "hello world");
}

#[test]
fn hidden_thinking_item_keeps_the_long_allowance_and_counts_as_liveness() {
    // The Responses wire streams no content for hidden thinking: the provider opens a reasoning
    // item and then goes quiet for a long stretch (measured 111s on muse-spark xhigh). That silence
    // is work in progress, not a stalled stream, so a provider-held open item already selects the
    // long allowance, and both the opening and closing events count as liveness (restarting the timer).
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    assert_eq!(
        stream_silence_timeout_secs(&state),
        STREAM_FIRST_CHUNK_TIMEOUT_SECS
    );

    let opened = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_item.added"),
        r#"{"type":"response.output_item.added","item":{"id":"rs_1","type":"reasoning","status":"in_progress","summary":[]}}"#,
    )
    .unwrap();
    assert!(!opened.should_stop);
    assert!(opened.meaningful_progress);
    assert_eq!(state.content.open_output_items, 1);
    assert_eq!(
        stream_silence_timeout_secs(&state),
        STREAM_DECLARED_ITEM_TIMEOUT_SECS
    );
    assert!(
        state.content.reasoning_items.is_empty(),
        "a content-free `.added` stub is not replayable and must not be captured"
    );

    let closed = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_item.done"),
        r#"{"type":"response.output_item.done","item":{"id":"rs_1","type":"reasoning","status":"completed","encrypted_content":"enc-xyz"}}"#,
    )
    .unwrap();
    assert!(closed.meaningful_progress);
    assert_eq!(state.content.open_output_items, 0);
    assert_eq!(state.content.reasoning_items.len(), 1);
    assert_eq!(
        stream_silence_timeout_secs(&state),
        STREAM_FIRST_CHUNK_TIMEOUT_SECS
    );

    // A close without a matching open must saturate instead of underflowing.
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_item.done"),
        r#"{"type":"response.output_item.done","item":{"id":"rs_1","type":"reasoning","status":"completed","encrypted_content":"enc-xyz"}}"#,
    )
    .unwrap();
    assert_eq!(state.content.open_output_items, 0);
}

#[test]
fn silence_window_selection_uses_the_connection_bound_once_output_arrived() {
    let mut state = StreamProcessingState::new();
    assert_eq!(
        stream_silence_timeout_secs(&state),
        STREAM_FIRST_CHUNK_TIMEOUT_SECS
    );
    state.content.assistant_text.push_str("working");
    assert_eq!(stream_silence_timeout_secs(&state), STREAM_IDLE_TIMEOUT_SECS);
    state.content.assistant_text.clear();
    state.content.finish_reason_seen = true;
    assert_eq!(stream_silence_timeout_secs(&state), STREAM_IDLE_TIMEOUT_SECS);

    // A model that declares a longer allowance (`stream_silence_timeout_secs`) widens only the
    // no-declaration case; a provider-held open item keeps the declared-item allowance.
    state.model_silence_timeout_secs = Some(180);
    assert_eq!(stream_silence_timeout_secs(&state), 180);
    state.content.open_output_items = 1;
    assert_eq!(
        stream_silence_timeout_secs(&state),
        STREAM_DECLARED_ITEM_TIMEOUT_SECS
    );
}

#[test]
fn open_function_call_item_selects_the_declared_item_allowance() {
    // Incident shape (muse-spark-1.3, single-file artifact): the model streams one narration sentence, closes
    // the message item, opens a `function_call` item and then generates ~20KB of arguments server-side, sending
    // nothing until that payload is ready. Tracking only reasoning items left that silence to the connection
    // bound, which cut the response 45s after the narration and replayed the whole generation.
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
        r#"{"type":"response.output_item.added","output_index":2,"item":{"id":"fc_1","type":"function_call","status":"in_progress","name":"write_file","call_id":"call_1","arguments":""}}"#,
    )
    .unwrap();
    assert_eq!(state.content.open_output_items, 1);
    assert_eq!(
        stream_silence_timeout_secs(&state),
        STREAM_DECLARED_ITEM_TIMEOUT_SECS
    );

    // Argument deltas carry the payload, not the item lifecycle: they must not count the same item twice.
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.function_call_arguments.delta"),
        r#"{"type":"response.function_call_arguments.delta","output_index":2,"delta":"{\"file_path\":"}"#,
    )
    .unwrap();
    assert_eq!(state.content.open_output_items, 1);

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_item.done"),
        r#"{"type":"response.output_item.done","output_index":2,"item":{"id":"fc_1","type":"function_call","status":"completed","name":"write_file","call_id":"call_1","arguments":"{\"file_path\":\"/tmp/pelican.html\",\"content\":\"hi\"}"}}"#,
    )
    .unwrap();
    assert_eq!(state.content.open_output_items, 0);
    assert_eq!(stream_silence_timeout_secs(&state), STREAM_IDLE_TIMEOUT_SECS);
}

#[test]
fn replayed_content_part_added_does_not_duplicate_visible_text() {
    // User-visible "conclusion printed twice": a compatibility gateway re-delivers the full text of
    // content_part.added (output_text) after the output_text.delta increments. Rendering as-is in
    // Append mode duplicates the body; ReplayedChunk must compute the unseen suffix for content and render only the new part.
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    // 1) delta increments render part of the body first
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"修复完成"}"#,
    )
    .unwrap();
    assert_eq!(state.content.assistant_text, "修复完成");

    // 2) content_part.added re-sends the part's full text (multi-path delivery by the protocol)
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.content_part.added"),
        r#"{"part":{"type":"output_text","text":"修复完成，验证通过。"}}"#,
    )
    .unwrap();
    // The seen prefix is swallowed; only the unseen suffix is appended
    assert_eq!(state.content.assistant_text, "修复完成，验证通过。");
    assert_eq!(current_history, "修复完成，验证通过。");

    // 3) Re-sending the exact same text: full overlap, nothing more is appended
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.content_part.added"),
        r#"{"part":{"type":"output_text","text":"修复完成，验证通过。"}}"#,
    )
    .unwrap();
    assert_eq!(state.content.assistant_text, "修复完成，验证通过。");
    assert_eq!(current_history, "修复完成，验证通过。");
}

#[test]
fn stream_payload_meaningful_progress_includes_new_reasoning_chunks() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    let usage_only = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        None,
        r#"{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}"#,
    )
    .unwrap();
    assert!(!usage_only.should_stop);
    assert!(!usage_only.meaningful_progress);

    let reasoning_only = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.reasoning_summary_text.delta"),
        r#"{"delta":"thinking step"}"#,
    )
    .unwrap();
    assert!(!reasoning_only.should_stop);
    assert!(reasoning_only.meaningful_progress);
    assert_eq!(state.content.reasoning_text, "thinking step");
    assert!(current_history.is_empty());

    let duplicate_reasoning_snapshot = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.reasoning_summary_text.done"),
        r#"{"text":"thinking step"}"#,
    )
    .unwrap();
    assert!(!duplicate_reasoning_snapshot.meaningful_progress);

    let content = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"answer"}"#,
    )
    .unwrap();
    assert!(content.meaningful_progress);
    assert_eq!(current_history, "answer");

    let duplicate_snapshot = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.done"),
        r#"{"text":"answer"}"#,
    )
    .unwrap();
    assert!(!duplicate_snapshot.meaningful_progress);

    let tool_call = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        None,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"/tmp/x\"}"}}]}}]}"#,
    )
    .unwrap();
    assert!(tool_call.meaningful_progress);

    let finish_reason = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        None,
        r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
    )
    .unwrap();
    assert!(finish_reason.meaningful_progress);
    assert_eq!(state.content.finish_reason_value.as_deref(), Some("stop"));
}

#[test]
fn repeated_reasoning_deltas_preserve_model_output() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    for _ in 0..2 {
        let outcome = process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            provider::openai_adapter(),
            Some("response.reasoning_summary_text.delta"),
            r#"{"delta":"same step"}"#,
        )
        .unwrap();
        assert!(outcome.meaningful_progress);
    }

    assert_eq!(state.content.reasoning_text, "same stepsame step");
    assert!(current_history.is_empty());
}

#[tokio::test]
async fn process_chunk_result_marks_empty_sse_as_no_meaningful_progress() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    let empty_step = process_chunk_result(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Ok(Some(
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":0,\"total_tokens\":1}}\n\n"
                .as_slice(),
        )),
    )
    .await
    .unwrap();
    match empty_step {
        StreamChunkStep::Continue {
            meaningful_progress,
        } => assert!(!meaningful_progress),
        _ => panic!("empty SSE should keep streaming without refreshing watchdog"),
    }

    let content_step = process_chunk_result(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Ok(Some(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n".as_slice(),
        )),
    )
    .await
    .unwrap();
    match content_step {
        StreamChunkStep::Continue {
            meaningful_progress,
        } => assert!(meaningful_progress),
        _ => panic!("content SSE should keep streaming and refresh watchdog"),
    }
    assert_eq!(current_history, "hello");
}

#[test]
fn suppressed_terminal_output_still_collects_subagent_response() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    crate::ai::driver::runtime_ctx::SUPPRESS_TERMINAL_OUTPUT.sync_scope(true, || {
        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            provider::openai_adapter(),
            Some("response.output_text.done"),
            r#"{"text":"subagent result"}"#,
        )
        .unwrap();
    });

    assert_eq!(current_history, "subagent result");
    assert_eq!(state.content.assistant_text, "subagent result");
}

#[test]
fn output_text_done_snapshot_with_leading_whitespace_does_not_duplicate_answer() {
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
        Some("response.output_text.delta"),
        r#"{"delta":"结论：history 文件结构本身正常。"}"#,
    )
    .unwrap();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.done"),
        r#"{"text":"\n\n结论：history 文件结构本身正常。"}"#,
    )
    .unwrap();

    assert_eq!(current_history, "结论：history 文件结构本身正常。");
    assert_eq!(
        state.content.assistant_text,
        "结论：history 文件结构本身正常。"
    );
}

#[test]
fn output_text_stream_preserves_inter_paragraph_newlines_without_snapshot_duplication() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    // The responses protocol emits in segments: body -> bare newline -> next segment. A bare newline is part of the body format,
    // must go into assistant_text verbatim; the final .done snapshot is only for dedup and must not append the whole content again.
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"有问题，而且问题不在 a.rs 本身。"}"#,
    )
    .unwrap();
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"\n\n"}"#,
    )
    .unwrap();
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"核心是 Agent 的收敛机制过于宽松。"}"#,
    )
    .unwrap();
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.done"),
        r#"{"text":"有问题，而且问题不在 a.rs 本身。\n\n核心是 Agent 的收敛机制过于宽松。"}"#,
    )
    .unwrap();

    assert_eq!(
        current_history,
        "有问题，而且问题不在 a.rs 本身。\n\n核心是 Agent 的收敛机制过于宽松。"
    );
    assert_eq!(
        state.content.assistant_text,
        "有问题，而且问题不在 a.rs 本身。\n\n核心是 Agent 的收敛机制过于宽松。"
    );
}

#[test]
fn process_stream_payload_suppresses_bare_registered_xml_tool_markup() {
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
        Some("response.output_text.delta"),
        r#"{"delta":"先确认一下。<execute_command>pwd</execute_command>"}"#,
    )
    .unwrap();

    assert_eq!(current_history, "先确认一下。");
    assert_eq!(state.content.assistant_text, "先确认一下。");
    let builder = state.content.tool_calls_map.get_ref(&0).unwrap();
    assert_eq!(builder.function_name, "execute_command");
    assert_eq!(builder.arguments, r#"{"command":"pwd"}"#);
}

#[test]
fn opencode_message_snapshot_recovers_reported_fullwidth_dsml_before_rendering() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();
    let payload = serde_json::json!({
        "choices": [{
            "message": {
                "content": REPORTED_FULLWIDTH_DSML_TOOL_CALL
            }
        }]
    })
    .to_string();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::opencode_adapter(),
        None,
        &payload,
    )
    .unwrap();

    assert!(current_history.is_empty());
    assert!(state.content.assistant_text.is_empty());
    assert!(state.content.hidden_meta.is_empty());
    assert_eq!(state.content.tool_calls_map.len(), 1);
    let builder = state.content.tool_calls_map.get_ref(&0).unwrap();
    assert_eq!(builder.function_name, "read_file");
    let args: serde_json::Value = serde_json::from_str(&builder.arguments).unwrap();
    assert_eq!(
        args["file_path"],
        "/Users/bytedance/rust_tools/src/bin/ai/driver/turn_runtime/iteration.rs"
    );
    assert_eq!(args["limit"], 80);
    assert_eq!(args["offset"], 110);
}

#[test]
fn fullwidth_dsml_done_snapshot_does_not_duplicate_delta_tool_call() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    for event_type in ["response.output_text.delta", "response.output_text.done"] {
        let payload = if event_type.ends_with(".delta") {
            serde_json::json!({ "delta": REPORTED_FULLWIDTH_DSML_TOOL_CALL }).to_string()
        } else {
            serde_json::json!({ "text": REPORTED_FULLWIDTH_DSML_TOOL_CALL }).to_string()
        };
        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            provider::openai_adapter(),
            Some(event_type),
            &payload,
        )
        .unwrap();
    }

    assert!(current_history.is_empty());
    assert!(state.content.assistant_text.is_empty());
    assert_eq!(state.content.tool_calls_map.len(), 1);
    assert_eq!(state.content.internal_tool_call_idx, 1);
}

#[test]
fn inline_tool_call_fallback_does_not_persist_protocol_as_hidden_meta() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.content.assistant_text = REPORTED_FULLWIDTH_DSML_TOOL_CALL.to_string();
    let mut app = test_app();
    let mut current_history = String::new();

    let result = finalize_stream_response(&mut app, &mut current_history, &markers, state).unwrap();

    assert_eq!(result.outcome, StreamOutcome::ToolCall);
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(result.tool_calls[0].function.name, "read_file");
    assert!(result.assistant_text.is_empty());
    assert!(
        result.hidden_meta.is_empty(),
        "工具协议不是 self_note，不得进入 hidden_meta"
    );
}

#[test]
fn process_stream_payload_halts_and_downshifts_on_hallucinated_result_marker() {
    // Reproduces this incident: the model fabricated a "tool call -> tool result" sequence in its
    // visible body, emitting `<function_results>` protocol markers the system never generates. Requirements:
    // (1) the hallucinated result block is stripped whole and never persisted; (2) the stream stops
    // (should_stop=true); (3) degenerate_repetition finish_reason is set to take the downgrade-retry path, keeping hallucinated body text from poisoning the next request.
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut app = test_app();
    let mut current_history = String::new();

    let outcome = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"我再读一遍<function_results>File: a.rs\n3 matches found</function_results>"}"#,
    )
    .unwrap();

    assert!(outcome.should_stop, "检出幻觉标记必须停流");
    assert!(
        outcome.meaningful_progress,
        "退化停流已设置 finish_reason，应视为语义进展"
    );
    assert_eq!(
        state.content.finish_reason_value.as_deref(),
        Some("degenerate_repetition"),
        "必须走 degenerate_repetition 降档重试路径"
    );
    assert!(
        !state.content.assistant_text.contains("function_results"),
        "幻觉协议标记不得落入 assistant_text：{}",
        state.content.assistant_text
    );
    assert!(
        !state.content.assistant_text.contains("matches found"),
        "幻觉结果文本不得落盘：{}",
        state.content.assistant_text
    );
}
