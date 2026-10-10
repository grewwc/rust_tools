use super::*;

#[test]
fn recover_inline_tool_calls_normalizes_namespaced_xml_prefix() {
    // Some frontends/models wrap Anthropic-style invokes in the <|DSML|> protocol.
    // After normalization the Anthropic XML parser should recognize them; no per-<|PREFIX|> parser needed.
    let raw = r#"<|DSML|tool_calls><|DSML|invoke name="apply_patch"><|DSML|parameter name="file_path">/tmp/x</|DSML|parameter><|DSML|parameter name="patch">---</|DSML|parameter></|DSML|invoke></|DSML|tool_calls>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover DSML-wrapped tool calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "apply_patch");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["file_path"], "/tmp/x");
    assert_eq!(args["patch"], "---");
}

#[test]
fn recover_inline_tool_calls_normalizes_fullwidth_dsml_prefix() {
    // Per debug.md, DeepSeek actually emits the fullwidth-vertical-bar variant: <｜｜DSML｜｜...>.
    let raw = r#"<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="apply_patch"><｜｜DSML｜｜parameter name="file_path">/tmp/x</｜｜DSML｜｜parameter><｜｜DSML｜｜parameter name="patch">---</｜｜DSML｜｜parameter></｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>"#;
    let calls = recover_inline_tool_calls(raw).expect("should recover fullwidth-DSML tool calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "apply_patch");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["file_path"], "/tmp/x");
    assert_eq!(args["patch"], "---");
}

/// Reproduces the session bc1f2e88 failure: a reasoner with a prefilled `<think>` template writes
/// the chain of thought into the content channel, ending only with a dangling `</think>`. With the
/// splitter armed, leaked reasoning before `</think>` must go into reasoning_text (rendered in the
/// thinking fold); only the real answer after `</think>` enters assistant_text, so the final answer is never output twice.
#[test]
fn armed_demuxer_splits_leaked_reasoning_from_visible_content() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.content.content_think_demuxer.arm();
    let mut app = test_app();
    let mut current_history = String::new();

    // The chain of thought arrives across multiple content chunks, with `</think>` split at a chunk boundary.
    for payload in [
        r#"{"choices":[{"delta":{"content":"Let me consolidate. "}}]}"#,
        r#"{"choices":[{"delta":{"content":"I have enough evidence.</thi"}}]}"#,
        r#"{"choices":[{"delta":{"content":"nk>## 结论\n不是 bug。"}}]}"#,
    ] {
        process_stream_payload(
            &mut app,
            &mut current_history,
            &markers,
            &mut state,
            provider::openai_adapter(),
            None,
            payload,
        )
        .unwrap();
    }

    // The real answer after `</think>` is the only visible body; the chain of thought must not leak into assistant_text.
    assert_eq!(state.content.assistant_text, "## 结论\n不是 bug。");
    assert!(!state.content.assistant_text.contains("Let me consolidate"));
    assert!(!state.content.assistant_text.contains("</think>"));
    // The leaked reasoning is split back into the reasoning channel.
    assert!(
        state
            .content
            .reasoning_text
            .contains("Let me consolidate. I have enough evidence.")
    );
}

#[test]
fn demuxer_buffered_content_counts_as_stream_progress() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.content.content_think_demuxer.arm();
    let mut app = test_app();
    let mut current_history = String::new();

    let outcome = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        None,
        r#"{"choices":[{"delta":{"content":"long reasoning without close yet"}}]}"#,
    )
    .unwrap();

    assert!(outcome.meaningful_progress);
    assert!(state.content.assistant_text.is_empty());
    assert!(state.content.reasoning_text.is_empty());
}

#[test]
fn demuxer_flush_without_close_tag_commits_visible_content() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.content.content_think_demuxer.arm();
    let mut app = test_app();
    let mut current_history = String::new();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        None,
        r#"{"choices":[{"delta":{"content":"visible fallback without close"}}]}"#,
    )
    .unwrap();
    assert!(state.content.assistant_text.is_empty());

    let result = finalize_stream_response(&mut app, &mut current_history, &markers, state).unwrap();

    assert_eq!(result.assistant_text, "visible fallback without close");
    assert_eq!(current_history, "visible fallback without close");
}

#[test]
fn replayed_content_part_after_demux_close_does_not_replay_reasoning_prefix() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.content.content_think_demuxer.arm();
    let mut app = test_app();
    let mut current_history = String::new();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        None,
        r#"{"choices":[{"delta":{"content":"reasoning</think>answer"}}]}"#,
    )
    .unwrap();
    assert_eq!(state.content.assistant_text, "answer");

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.content_part.added"),
        r#"{"part":{"type":"output_text","text":"reasoning</think>answer"}}"#,
    )
    .unwrap();

    assert_eq!(state.content.assistant_text, "answer");
    assert_eq!(current_history, "answer");
    assert_eq!(state.content.reasoning_text, "reasoning");
}

#[test]
fn output_text_snapshot_after_demux_close_does_not_replay_reasoning_prefix() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.content.content_think_demuxer.arm();
    let mut app = test_app();
    let mut current_history = String::new();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"reasoning</think>answer"}"#,
    )
    .unwrap();
    assert_eq!(state.content.assistant_text, "answer");

    let snapshot = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.done"),
        r#"{"text":"reasoning</think>answer"}"#,
    )
    .unwrap();

    assert!(!snapshot.meaningful_progress);
    assert_eq!(state.content.assistant_text, "answer");
    assert_eq!(current_history, "answer");
    assert_eq!(state.content.reasoning_text, "reasoning");
}

#[test]
fn output_text_snapshot_can_finish_a_partially_streamed_demux_capture() {
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    state.content.content_think_demuxer.arm();
    let mut app = test_app();
    let mut current_history = String::new();

    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.delta"),
        r#"{"delta":"reasoning"}"#,
    )
    .unwrap();
    assert!(state.content.assistant_text.is_empty());

    let snapshot = process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        Some("response.output_text.done"),
        r#"{"text":"reasoning</think>answer"}"#,
    )
    .unwrap();

    assert!(snapshot.meaningful_progress);
    assert_eq!(state.content.assistant_text, "answer");
    assert_eq!(current_history, "answer");
    assert_eq!(state.content.reasoning_text, "reasoning");
}

/// Reverse assertion: for a normal model without the splitter armed (using the separate reasoning_content
/// field), behavior is unchanged — a literal `</think>` in content lands verbatim in the visible body and is never swallowed.
#[test]
fn stream_filters_rewrite_visible_content_before_commit() {
    // Filter: rewrite "secret" into "[REDACTED]".
    struct RedactFilter;
    impl crate::ai::ports::stream::StreamFilter for RedactFilter {
        fn filter(&self, chunk: &str) -> Option<String> {
            Some(chunk.replace("secret", "[REDACTED]"))
        }
        fn name(&self) -> &'static str {
            "redact"
        }
    }
    let markers = StreamMarkers::new();
    let mut state = StreamProcessingState::new();
    // Step 6: inject the filter chain and verify that `process_stream_payload`'s visible-content commit
    // point applies the filters (the rewrite lands in assistant_text / history, the original text does not appear).
    state.filters = crate::ai::ports::stream::FilterChain::new().push(RedactFilter);
    let mut app = test_app();
    let mut current_history = String::new();
    process_stream_payload(
        &mut app,
        &mut current_history,
        &markers,
        &mut state,
        provider::openai_adapter(),
        None,
        r#"{"choices":[{"delta":{"content":"hello secret world"}}]}"#,
    )
    .unwrap();
    assert_eq!(state.content.assistant_text, "hello [REDACTED] world");
    assert_eq!(current_history, "hello [REDACTED] world");
    assert!(!state.content.assistant_text.contains("secret"));
}

fn unarmed_demuxer_leaves_content_untouched() {
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
        None,
        r#"{"choices":[{"delta":{"content":"see </think> literal"}}]}"#,
    )
    .unwrap();

    assert_eq!(state.content.assistant_text, "see </think> literal");
    assert!(state.content.reasoning_text.is_empty());
}

// =============================================================================
// Golden wire→parse harness
// =============================================================================
// Regression guard for the SSE wire-shape → StreamResult contract. Unlike the
// process_stream_payload unit tests above (which feed one payload at a time), and
// unlike the ad-hoc loopback tests earlier in this file, this harness scripts a
// full multi-event SSE response over a real loopback socket and drives the real
// `stream_response` state machine end to end. It is the deterministic, offline
// core of the "golden transcript" idea: fixtures describe the exact bytes a
// provider would stream; assertions pin the parsed outcome. No network, no live
// model — so it is safe to run in CI and catches regressions in chunk framing,
// tool-call assembly, finish_reason handling, and reasoning demux.
//
// Reusability: `ScriptedSse::spawn(events)` takes a list of raw SSE `data:`
// payloads (JSON strings, or the literal "[DONE]") and serves them over a real
// loopback socket; `drive(events)` runs the parser and returns the StreamResult.
// New golden cases only add fixtures; the harness is fixed.
#[test]
fn serve_thinking_chunk_split_keeps_glued_close_marker_on_time() {
    use crate::ai::background::ServeLiveKind;

    let open = "╭─ thinking";
    let close = "╰─ done thinking";
    // Standalone markers keep their one-frame mapping.
    let open_case = format!("\n{open}\n");
    assert_eq!(
        split_serve_thinking_chunk(&open_case, open, close),
        vec![(ServeLiveKind::ThinkingStart, "")]
    );
    let close_case = format!("{close}\n");
    assert_eq!(
        split_serve_thinking_chunk(&close_case, open, close),
        vec![(ServeLiveKind::ThinkingDone, "")]
    );
    // Plain body passes through untouched.
    assert_eq!(
        split_serve_thinking_chunk("half a thought", open, close),
        vec![(ServeLiveKind::Thinking, "half a thought")]
    );
    // A close marker glued to body text still closes on time: the head stays
    // thinking and the answer tail becomes a delta instead of being swallowed
    // by the fold.
    assert_eq!(
        split_serve_thinking_chunk(
            "half a thought\n╰─ done thinking\nAnd the answer",
            open,
            close
        ),
        vec![
            (ServeLiveKind::Thinking, "half a thought\n"),
            (ServeLiveKind::ThinkingDone, ""),
            (ServeLiveKind::Delta, "\nAnd the answer"),
        ]
    );
}
