    use super::*;
    #[test]
    fn model_silence_timeout_is_resolved_from_the_registry() {
        // The per-model allowance travels through the registry accessor: every declared entry resolves
        // to its declared value, and a model without an entry keeps the runtime default instead of
        // failing. Nothing pins a model name, so registry churn cannot break this test.
        let registry = crate::ai::model_names::all();
        assert!(
            !registry.is_empty(),
            "the model registry must load for this check to mean anything"
        );
        for def in registry {
            assert_eq!(
                crate::ai::models::stream_silence_timeout_for_model(&def.key),
                def.stream_silence_timeout_secs.filter(|secs| *secs > 0)
            );
        }
        assert_eq!(
            crate::ai::models::stream_silence_timeout_for_model("no-such-model"),
            None
        );
    }

    /// One scripted SSE server for a single response. Splits each event across
    /// its own HTTP chunk so the framing/boundary logic is exercised the same way
    /// a real streaming provider drives it.
    struct ScriptedSse {
        addr: std::net::SocketAddr,
        done_tx: mpsc::Sender<()>,
        handle: std::thread::JoinHandle<()>,
    }

    impl ScriptedSse {
        /// Spawn a loopback server that emits `events` as `data: <event>\n\n`
        /// SSE frames, then blocks until the response is dropped. Each entry is a
        /// raw payload: a JSON chunk body, or "[DONE]" for the terminator.
        fn spawn(events: Vec<String>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let (done_tx, done_rx) = mpsc::channel::<()>();
            let handle = std::thread::spawn(move || {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request_buf = [0u8; 1024];
                let _ = stream.read(&mut request_buf);
                stream
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
                    )
                    .unwrap();
                for event in &events {
                    write_http_chunk(&mut stream, &format!("data: {event}\n\n")).unwrap();
                }
                // Keep the socket open until the test drops the response, matching
                // the finish_reason-grace behavior of real providers.
                let _ = done_rx.recv_timeout(Duration::from_secs(2));
            });
            Self {
                addr,
                done_tx,
                handle,
            }
        }

        async fn response(&self) -> reqwest::Response {
            let client = reqwest::Client::builder().no_proxy().build().unwrap();
            client
                .post(format!("http://{}/chat", self.addr))
                .send()
                .await
                .unwrap()
        }

        fn shutdown(self, response: reqwest::Response) {
            drop(response);
            let _ = self.done_tx.send(());
            self.handle.join().unwrap();
        }
    }

    /// Drive `stream_response` against a scripted event list and return the parsed
    /// result. Centralizes the env-lock, os-globals, and interrupt hygiene the
    /// existing loopback tests each repeat.
    async fn drive(events: Vec<String>) -> crate::ai::types::StreamResult {
        let _signal_guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let server = ScriptedSse::spawn(events);
        let mut response = server.response().await;
        let mut app = test_app();
        init_os_tools_globals(app.os.clone());
        crate::ai::driver::signal::clear_request_interrupt();
        let mut current_history = String::new();

        let result = tokio::time::timeout(
            Duration::from_secs(3),
            stream_response(&mut app, &mut response, &mut current_history, None),
        )
        .await
        .expect("stream_response should finish within the grace window")
        .unwrap();

        server.shutdown(response);
        crate::ai::driver::signal::clear_request_interrupt();
        if let Ok(mut guard) = GLOBAL_OS.lock() {
            *guard = None;
        }
        result
    }

    #[tokio::test]
    async fn golden_multi_chunk_text_assembles_in_order() {
        let result = drive(vec![
            r#"{"choices":[{"delta":{"content":"Hel"}}]}"#.to_string(),
            r#"{"choices":[{"delta":{"content":"lo, "}}]}"#.to_string(),
            r#"{"choices":[{"delta":{"content":"world"}}]}"#.to_string(),
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#.to_string(),
            "[DONE]".to_string(),
        ])
        .await;

        assert_eq!(result.outcome, StreamOutcome::Completed);
        assert_eq!(result.assistant_text, "Hello, world");
        assert!(result.tool_calls.is_empty());
        assert!(!result.truncated_by_length);
    }

    #[tokio::test]
    async fn golden_streamed_tool_call_is_reassembled() {
        // OpenAI-style streaming tool call: name arrives first, arguments arrive
        // as fragments across subsequent deltas, then finish_reason=tool_calls.
        let result = drive(vec![
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":""}}]}}]}"#.to_string(),
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"file_path\":"}}]}}]}"#.to_string(),
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Cargo.toml\"}"}}]}}]}"#.to_string(),
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#.to_string(),
            "[DONE]".to_string(),
        ])
        .await;

        assert_eq!(result.outcome, StreamOutcome::ToolCall);
        assert_eq!(result.tool_calls.len(), 1);
        let call = &result.tool_calls[0];
        assert_eq!(call.id, "call_1");
        assert_eq!(call.function.name, "read_file");
        // Fragmented arguments must concatenate into valid JSON.
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments)
            .expect("reassembled tool-call arguments must be valid JSON");
        assert_eq!(args["file_path"], "Cargo.toml");
    }

    #[tokio::test]
    async fn golden_finish_reason_length_marks_truncation() {
        let result = drive(vec![
            r#"{"choices":[{"delta":{"content":"partial answer"}}]}"#.to_string(),
            r#"{"choices":[{"delta":{},"finish_reason":"length"}]}"#.to_string(),
            "[DONE]".to_string(),
        ])
        .await;

        // Even with visible text, a length finish_reason must flag truncation so
        // upper layers can inject a shrink hint / retry.
        assert_eq!(result.assistant_text, "partial answer");
        assert!(
            result.truncated_by_length,
            "length finish_reason => truncated_by_length"
        );
    }

    #[tokio::test]
    async fn golden_reasoning_is_split_from_visible_content() {
        // reasoning_content on the delta must land in reasoning_text, not the
        // visible assistant answer.
        let result = drive(vec![
            r#"{"choices":[{"delta":{"reasoning_content":"thinking step"}}]}"#.to_string(),
            r#"{"choices":[{"delta":{"content":"final answer"}}]}"#.to_string(),
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#.to_string(),
            "[DONE]".to_string(),
        ])
        .await;

        assert_eq!(result.outcome, StreamOutcome::Completed);
        assert_eq!(result.assistant_text, "final answer");
        assert!(
            result.reasoning_text.contains("thinking step"),
            "reasoning must be captured separately, got: {:?}",
            result.reasoning_text
        );
        assert!(!result.assistant_text.contains("thinking step"));
    }

    #[test]
    fn tool_args_open_at_starts_on_open_tool_call_and_resets_on_finish() {
        let app = crate::ai::middleware::test_util::test_app();
        let mut content = initial_stream_processing_state(&app).content;

        // No tool call open yet: the stall timer stays unset.
        assert_eq!(update_tool_args_open_at(&content, None), None);

        // A tool call opens: the timer starts at that moment.
        content
            .tool_calls_map
            .insert(0, ToolCallBuilder::default());
        let open_at = update_tool_args_open_at(&content, None);
        assert!(open_at.is_some(), "timer must start when a tool call opens");

        // While the call stays open the original timestamp is preserved.
        assert_eq!(update_tool_args_open_at(&content, open_at), open_at);

        // Finish reason clears the timer even if the map is not empty yet.
        content.finish_reason_seen = true;
        assert_eq!(update_tool_args_open_at(&content, open_at), None);

        // Empty map also clears it.
        content.finish_reason_seen = false;
        content.tool_calls_map.clear();
        assert_eq!(update_tool_args_open_at(&content, open_at), None);
    }

    #[test]
    fn tool_args_stall_detects_pathological_trickle_stream() {
        let timeout = Duration::from_secs(STREAM_TOOL_ARGS_STALL_TIMEOUT_SECS);
        let now = Instant::now();

        // No open tool call: never stalled.
        assert!(!tool_args_stream_stalled(None, now, timeout));
        // Still inside the window: not stalled.
        assert!(!tool_args_stream_stalled(
            Some(now - timeout + Duration::from_secs(1)),
            now,
            timeout
        ));
        // Just below the boundary: not stalled.
        assert!(!tool_args_stream_stalled(
            Some(now - timeout + Duration::from_millis(1)),
            now,
            timeout
        ));
        // At and beyond the boundary: stalled — this is the trickle-stream
        // case where the idle timer never fires because every delta counts
        // as meaningful progress.
        assert!(tool_args_stream_stalled(Some(now - timeout), now, timeout));
        assert!(tool_args_stream_stalled(
            Some(now - timeout - Duration::from_secs(1)),
            now,
            timeout
        ));
    }

/// Width source for these tests: they run without a TTY, so `raw_terminal_cols()` falls back to
/// `COLUMNS`. Mutating the process environment is `unsafe` in edition 2024; each test holds `ENV_LOCK`
/// for the whole width-sensitive section, so no other test observes the value.
fn set_test_columns(cols: &str) {
    unsafe {
        std::env::set_var("COLUMNS", cols);
    }
}

fn clear_test_columns() {
    unsafe {
        std::env::remove_var("COLUMNS");
    }
}

/// Rows of a live region are wrapped at the live terminal width, not at an artificially narrowed bound,
/// so a long logical row still shows as much text as the terminal can hold. An artificial bound (live
/// rows capped at 100 columns) kept the stored row count resize-proof but cut every long row short on
/// wide terminals; the resize case is handled on the erase side instead, by recomputing the footprint
/// from the stored text (`live_region_fold_header_erase_covers_re_wrapped_rows`,
/// `live_region_waiting_hint_erase_covers_re_wrapped_rows`).
#[test]
fn live_region_rows_wrap_at_the_live_terminal_width() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    set_test_columns("200");
    let rows = wrap_line_to_terminal_rows_with_reserve(&"x".repeat(400), 4);
    assert!(rows.len() > 1, "the line must be wrapped, got {} rows", rows.len());
    let widest = rows.iter().map(|row| row.chars().count()).max().unwrap_or(0);
    assert!(
        widest > 150,
        "rows must use the live terminal width, got {widest} columns: {rows:?}"
    );
    clear_test_columns();
}

/// A header written on a wide terminal is one physical row; a narrowed terminal re-wraps that row into
/// several. The erase must clear all of them — moving up exactly one row left the previous
/// `○ thinking · …` header on screen on every redraw, which is the stack of headers seen when the
/// terminal is resized while the model streams.
#[test]
fn live_region_fold_header_erase_covers_re_wrapped_rows() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let mut state = StreamProcessingState::new();
    state.render.thinking_fold.active = true;
    state.render.thinking_fold.set_labels(
        "thinking about a header label that is wider than a forty column terminal row",
        "done",
    );

    set_test_columns("200");
    let mut header = Vec::new();
    // The runtime call sites keep the returned text on the fold; the erase reads it back from there.
    state.render.thinking_fold.header_rendered_line = write_fold_header(
        &mut header,
        Some("12.3 tok/s"),
        &state.render.thinking_fold,
    )
    .unwrap();
    assert_eq!(
        thinking_fold_header_rendered_rows(&state.render.thinking_fold),
        1,
        "a wide terminal keeps the header on one row"
    );

    set_test_columns("40");
    let rows = thinking_fold_header_rendered_rows(&state.render.thinking_fold);
    assert!(
        rows > 1,
        "the narrowed terminal re-wraps the header, got {rows} rows"
    );

    state.render.thinking_fold.header_drawn = true;
    let mut redraw = Vec::new();
    thinking_fold_redraw_to(&mut redraw, None, &mut state.render.thinking_fold).unwrap();
    let redraw = String::from_utf8_lossy(&redraw).into_owned();
    assert!(
        redraw.contains(&format!("\x1b[{}A", rows)),
        "the erase must move up over all {rows} header rows: {redraw:?}"
    );
    assert!(
        redraw.matches("\x1b[2K").count() >= rows,
        "the erase must clear all {rows} header rows: {redraw:?}"
    );
    clear_test_columns();
}

/// The waiting hint owns its own line and is erased by moving back up over it, so the erase must clear
/// every physical row the hint occupies — including rows a narrowed terminal created by re-wrapping the
/// hint row that was already drawn.
#[test]
fn live_region_waiting_hint_erase_covers_re_wrapped_rows() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    set_test_columns("200");
    let hint = clamp_line_to_terminal_row(&format!(
        "  ⠋ receiving `read_file` arguments…{}",
        "x".repeat(200)
    ));
    assert_eq!(live_preview_cursor_rows(&hint), 1);

    set_test_columns("60");
    let rows = live_preview_cursor_rows(&hint);
    assert!(
        rows > 1,
        "the narrowed terminal re-wraps the hint, got {rows} rows"
    );

    let mut out = Vec::new();
    erase_rows_above_cursor(&mut out, rows).unwrap();
    let emitted = String::from_utf8_lossy(&out).into_owned();
    assert_eq!(
        emitted.matches("\x1b[2K").count(),
        rows,
        "emitted: {emitted:?}"
    );
    assert!(
        emitted.starts_with(&format!("\r\x1b[{}A", rows)),
        "emitted: {emitted:?}"
    );
    clear_test_columns();
}
