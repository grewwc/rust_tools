use super::*;

#[test]
fn terminal_dedupe_recognizes_exact_replayed_tool_round_narration() {
    let mut state = StreamProcessingState::new();
    state.render.terminal_dedupe = Some(TerminalDedupeState {
        candidate: "结论已经在工具调用前展示。".to_string(),
        buffered_terminal_output: "结论已经在工具调用前".to_string(),
    });

    assert!(terminal_dedupe_still_matches(&state));
    assert!(!terminal_dedupe_buffer_is_complete_match(&state));

    let dedupe = state.render.terminal_dedupe.as_mut().unwrap();
    dedupe.buffered_terminal_output.push_str("展示。");
    state.content.assistant_text = dedupe.buffered_terminal_output.clone();

    assert!(terminal_dedupe_buffer_is_complete_match(&state));
    assert!(final_assistant_matches_terminal_dedupe(&state));
}

#[test]
fn terminal_dedupe_ignores_digest_blocks_in_final_assistant_text() {
    let mut state = StreamProcessingState::new();
    state.render.terminal_dedupe = Some(TerminalDedupeState {
        candidate: "结论已经展示。".to_string(),
        buffered_terminal_output: "结论已经展示。".to_string(),
    });
    state.content.assistant_text = format!(
        "结论已经展示。{}内部图片摘要{}",
        crate::ai::request::DIGEST_BEGIN,
        crate::ai::request::DIGEST_END
    );

    assert!(final_assistant_matches_terminal_dedupe(&state));
}

#[test]
fn terminal_dedupe_releases_content_after_visible_divergence() {
    let mut state = StreamProcessingState::new();
    state.render.terminal_dedupe = Some(TerminalDedupeState {
        candidate: "旧结论".to_string(),
        buffered_terminal_output: "新结论".to_string(),
    });

    assert!(!terminal_dedupe_still_matches(&state));
    assert!(!terminal_dedupe_buffer_is_complete_match(&state));
}

#[test]
fn completed_assistant_body_is_withheld_for_final_gates() {
    let mut app = test_app();
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.render.defer_assistant_body = true;
    state.render.terminal_dedupe = Some(TerminalDedupeState {
        candidate: "older visible narration".to_string(),
        buffered_terminal_output: String::new(),
    });
    let mut current_history = String::new();

    commit_visible_content(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        "provisional conclusion".to_string(),
    )
    .unwrap();

    assert_eq!(state.content.assistant_text, "provisional conclusion");
    assert_eq!(current_history, "provisional conclusion");
    assert_eq!(
        state
            .render
            .terminal_dedupe
            .as_ref()
            .unwrap()
            .buffered_terminal_output,
        "",
        "a provisional final must not enter the live terminal pipeline"
    );

    let result = finalize_stream_response(&mut app, &mut current_history, &markers, state).unwrap();
    assert_eq!(result.outcome, StreamOutcome::Completed);
}

#[test]
fn waiting_hint_tool_name_is_single_line_and_terminal_safe() {
    assert_eq!(
        sanitize_waiting_hint_tool_name("apply_\x1b[31mpatch\n next\tstep"),
        "apply_patch next step"
    );
    assert_eq!(sanitize_waiting_hint_tool_name("\n\t"), "tool");
}

#[test]
fn idle_timeout_discards_unconfirmed_tool_call_and_marks_stream_error() {
    let mut app = test_app();
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut current_history = String::new();
    state.content.stream_idle_timed_out = true;
    state.content.tool_calls_map.insert(
        0,
        ToolCallBuilder {
            id: "call-timeout".to_string(),
            tool_type: "function".to_string(),
            function_name: "apply_patch".to_string(),
            arguments: r#"{"patch":"partial but currently valid"}"#.to_string(),
            printed_arguments_len: 0,
        },
    );

    let result = finalize_stream_response(&mut app, &mut current_history, &markers, state).unwrap();

    assert_eq!(result.outcome, StreamOutcome::Truncated);
    assert!(result.stream_error);
    assert!(result.tool_calls.is_empty());
}

#[test]
fn tool_arg_cap_discards_unconfirmed_tool_call_and_marks_truncated() {
    // When accumulated tool arguments exceed the cap and get cut off, the half-finished tool call
    // must not reach the execution layer even if the JSON happens to be valid at the cutoff instant
    // (same principle as idle timeout): drop it and take degenerate_repetition's retryable Truncated path instead of executing partial args.
    let mut app = test_app();
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    let mut current_history = String::new();
    state.content.finish_reason_seen = true;
    state.content.finish_reason_value = Some(DEGENERATE_REPETITION_FINISH_REASON.to_string());
    state.content.tool_args_cap_exceeded = true;
    state.content.tool_calls_map.insert(
        0,
        ToolCallBuilder {
            id: "call-cap".to_string(),
            tool_type: "function".to_string(),
            function_name: "apply_patch".to_string(),
            // Even when the JSON is valid at the cutoff instant, it must not be executed.
            arguments: r#"{"patch":"partial but currently valid"}"#.to_string(),
            printed_arguments_len: 0,
        },
    );
    // JSON in assistant_text that looks like an inline tool call (recognized by recover_inline_tool_calls)
    // is equally untrusted in this over-limit scenario and must not be recovered for execution, otherwise it would bypass the drop logic above.
    state.content.assistant_text =
        r#"{"function":{"name":"apply_patch","arguments":"{}"},"id":"call-recover"}"#.to_string();

    let result = finalize_stream_response(&mut app, &mut current_history, &markers, state).unwrap();

    assert_eq!(result.outcome, StreamOutcome::Truncated);
    assert!(result.tool_calls.is_empty());
}
