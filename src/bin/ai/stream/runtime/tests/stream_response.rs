use super::*;

#[test]
fn stream_response_discards_valid_tools_on_any_explicit_truncation_signal() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    crate::ai::driver::signal::clear_request_interrupt();

    for (finish_reason, incomplete, malformed) in [
        (Some("stop"), false, true),
        (Some("length"), false, false),
        (None, true, false),
        (Some("length"), true, false),
    ] {
        let mut app = test_app();
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();
        let mut current_history = String::new();
        state.content.assistant_text = "partial output".to_string();
        state.content.finish_reason_seen = finish_reason.is_some();
        state.content.finish_reason_value = finish_reason.map(str::to_string);
        state.content.response_incomplete = incomplete;
        state.content.tool_calls_map.insert(
            0,
            ToolCallBuilder {
                id: "call-valid".to_string(),
                tool_type: "function".to_string(),
                function_name: "run_shell".to_string(),
                arguments: r#"{"command":"pwd"}"#.to_string(),
                ..Default::default()
            },
        );
        if malformed {
            state.content.tool_calls_map.insert(
                1,
                ToolCallBuilder {
                    id: "call-cut".to_string(),
                    tool_type: "function".to_string(),
                    function_name: "write_file".to_string(),
                    arguments: r#"{"path":"/tmp/x","content":"aaa"#.to_string(),
                    ..Default::default()
                },
            );
        }

        let result =
            finalize_stream_response(&mut app, &mut current_history, &markers, state).unwrap();

        assert_eq!(result.outcome, StreamOutcome::Truncated);
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.assistant_text, "partial output");
        assert_eq!(result.truncated_by_length, finish_reason == Some("length"));
        assert!(!result.stream_error);
    }
}

#[test]
fn stream_response_does_not_recover_inline_tools_from_incomplete_output() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    crate::ai::driver::signal::clear_request_interrupt();
    let inline_call = r#"{"name":"read_file","arguments":{"path":"/tmp/x"}}"#;
    for finish_reason in [Some("length"), None] {
        let mut app = test_app();
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();
        let mut current_history = String::new();
        state.content.assistant_text = inline_call.to_string();
        state.content.finish_reason_seen = finish_reason.is_some();
        state.content.finish_reason_value = finish_reason.map(str::to_string);
        state.content.response_incomplete = finish_reason.is_none();

        let result =
            finalize_stream_response(&mut app, &mut current_history, &markers, state).unwrap();

        assert_eq!(result.outcome, StreamOutcome::Truncated);
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.assistant_text, inline_call);
    }
}

#[tokio::test]
async fn stream_response_decode_error_fallback_always_retries_and_discards_tools() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    crate::ai::driver::signal::clear_request_interrupt();
    for tool_arguments in [None, Some(r#"{"command":"pwd"}"#), Some(r#"{"command":"#)] {
        let mut app = test_app();
        let markers = StreamMarkers::new();
        let mut state = StreamProcessingState::new();
        state.framing.decode_error_count = MAX_DECODE_ERRORS;
        state.content.assistant_text = "partial output".to_string();
        state.content.reasoning_text = "partial reasoning".to_string();
        state.content.finish_reason_seen = true;
        state.content.finish_reason_value = Some("stop".to_string());
        if let Some(arguments) = tool_arguments {
            state.content.tool_calls_map.insert(
                0,
                ToolCallBuilder {
                    id: "call-unconfirmed".to_string(),
                    tool_type: "function".to_string(),
                    function_name: "run_shell".to_string(),
                    arguments: arguments.to_string(),
                    ..Default::default()
                },
            );
        }

        let result = handle_stream_decode_error(&mut app, &markers, &mut state, "read failed")
            .await
            .expect("exceeding the consecutive read-error threshold must return a retryable result");

        assert_eq!(result.outcome, StreamOutcome::Truncated);
        assert!(result.stream_error);
        assert!(result.tool_calls.is_empty());
        assert!(state.content.tool_calls_map.is_empty());
        assert_eq!(result.assistant_text, "partial output");
        assert_eq!(result.reasoning_text, "partial reasoning");
        assert!(result.skip_response_drain);
        assert!(!result.truncated_by_length);
        assert_eq!(state.framing.decode_error_count, MAX_DECODE_ERRORS + 1);
    }
}

#[tokio::test]
async fn stream_response_returns_after_finish_reason_without_eof() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request_buf = [0u8; 1024];
        let _ = stream.read(&mut request_buf);
        stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
                )
                .unwrap();
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
        )
        .unwrap();
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        )
        .unwrap();
        let _ = done_rx.recv_timeout(Duration::from_secs(2));
    });

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let mut response = client
        .post(format!("http://{addr}/chat"))
        .send()
        .await
        .unwrap();
    let mut app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();
    let mut current_history = String::new();

    let result = tokio::time::timeout(
        Duration::from_secs(2),
        stream_response(&mut app, &mut response, &mut current_history, None),
    )
    .await
    .expect("stream_response should return after the configured finish_reason grace window")
    .unwrap();

    assert_eq!(result.outcome, StreamOutcome::Completed);
    assert_eq!(result.assistant_text, "hello");
    assert_eq!(current_history, "hello");
    assert!(result.skip_response_drain);

    drop(response);
    let _ = done_tx.send(());
    server.join().unwrap();
    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[tokio::test]
async fn stream_response_marks_length_finish_reason_with_visible_text_as_truncated() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request_buf = [0u8; 1024];
        let _ = stream.read(&mut request_buf);
        stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
                )
                .unwrap();
        // Visible text present but the server truncated at the output cap: finish_reason=length.
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{\"content\":\"partial output\"}}]}\n\n",
        )
        .unwrap();
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
        )
        .unwrap();
        let _ = done_rx.recv_timeout(Duration::from_secs(2));
    });

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let mut response = client
        .post(format!("http://{addr}/chat"))
        .send()
        .await
        .unwrap();
    let mut app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();
    let mut current_history = String::new();

    let result = tokio::time::timeout(
        Duration::from_secs(2),
        stream_response(&mut app, &mut response, &mut current_history, None),
    )
    .await
    .expect("stream_response should return after finish_reason grace window")
    .unwrap();

    // An explicit output-cap stop must retry even when the response already contains visible text.
    assert_eq!(result.outcome, StreamOutcome::Truncated);
    assert_eq!(result.assistant_text, "partial output");
    assert!(result.truncated_by_length);
    assert!(!result.stream_error);

    drop(response);
    let _ = done_tx.send(());
    server.join().unwrap();
    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[tokio::test]
async fn stream_response_escalates_provider_incomplete_with_visible_text() {
    // Silent-stop regression: the provider stopped the model at the output cap while the visible
    // body already contained text (typically an announcement such as "starting the code change").
    // Reading that as Completed ends the turn with the work never started, so the provider's own
    // incomplete declaration must escalate to the retryable truncation path instead.
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request_buf = [0u8; 1024];
        let _ = stream.read(&mut request_buf);
        stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
                )
                .unwrap();
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{\"content\":\"starting the code change\"}}]}\n\n",
        )
        .unwrap();
        write_http_chunk(
            &mut stream,
            "data: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"usage\":{\"input_tokens\":5,\"output_tokens\":9}}}\n\n",
        )
        .unwrap();
        let _ = done_rx.recv_timeout(Duration::from_secs(2));
    });

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let mut response = client
        .post(format!("http://{addr}/responses"))
        .send()
        .await
        .unwrap();
    let mut app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();
    let mut current_history = String::new();

    let result = tokio::time::timeout(
        Duration::from_secs(2),
        stream_response(&mut app, &mut response, &mut current_history, None),
    )
    .await
    .expect("stream_response should return after finish_reason grace window")
    .unwrap();

    assert_eq!(result.outcome, StreamOutcome::Truncated);
    assert_eq!(result.assistant_text, "starting the code change");
    assert!(result.truncated_by_length);
    assert!(!result.stream_error);

    drop(response);
    let _ = done_tx.send(());
    server.join().unwrap();
    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[tokio::test]
async fn stream_response_retries_unattributed_provider_incomplete() {
    // The gateway kept the non-`completed` status but dropped the reason (real shape on this wire:
    // `incomplete_details` present but null). The stop is still the provider's own declaration, and
    // because nothing named it as a policy stop the turn must stay retryable instead of ending as a
    // terminal stream error that discards the partial body. The server also holds the socket open
    // after the declaration: a provider-declared end must stop at its own marker instead of letting
    // the idle timer report the stop as a stream failure.
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request_buf = [0u8; 1024];
        let _ = stream.read(&mut request_buf);
        stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
                )
                .unwrap();
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{\"content\":\"starting the code change\"}}]}\n\n",
        )
        .unwrap();
        write_http_chunk(
            &mut stream,
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":9}}}\n\n",
        )
        .unwrap();
        // Hold the socket well past the client-side timeout: returning early can then only come from
        // the declared stop's grace window, never from the connection closing.
        let _ = done_rx.recv_timeout(Duration::from_secs(10));
    });

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let mut response = client
        .post(format!("http://{addr}/responses"))
        .send()
        .await
        .unwrap();
    let mut app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();
    let mut current_history = String::new();

    let result = tokio::time::timeout(
        Duration::from_secs(2),
        stream_response(&mut app, &mut response, &mut current_history, None),
    )
    .await
    .expect("a declared stop must end at the grace window, not wait for the socket to close")
    .unwrap();

    assert_eq!(result.outcome, StreamOutcome::Truncated);
    assert_eq!(result.assistant_text, "starting the code change");
    assert!(!result.truncated_by_length);
    assert!(!result.stream_error);

    drop(response);
    let _ = done_tx.send(());
    server.join().unwrap();
    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[tokio::test]
async fn stream_response_keeps_tool_call_delivered_before_provider_completed() {
    // The server holds the socket open after `response.completed`. The provider declared the
    // response finished, so the runtime must end at that marker (grace window) instead of waiting
    // for the close: waiting would let the idle timer cut the stream and drop the tool call that
    // was already delivered in full.
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request_buf = [0u8; 1024];
        let _ = stream.read(&mut request_buf);
        stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
                )
                .unwrap();
        write_http_chunk(
            &mut stream,
            "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":2,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"status\":\"in_progress\",\"name\":\"run_shell\",\"call_id\":\"call_1\",\"arguments\":\"\"}}\n\n",
        )
        .unwrap();
        write_http_chunk(
            &mut stream,
            "event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":2,\"item_id\":\"fc_1\",\"delta\":\"{\\\"command\\\":\\\"ls\\\"}\"}\n\n",
        )
        .unwrap();
        write_http_chunk(
            &mut stream,
            "event: response.function_call_arguments.done\ndata: {\"type\":\"response.function_call_arguments.done\",\"output_index\":2,\"item_id\":\"fc_1\",\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\",\"name\":\"run_shell\"}\n\n",
        )
        .unwrap();
        write_http_chunk(
            &mut stream,
            "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":2,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"status\":\"completed\",\"name\":\"run_shell\",\"call_id\":\"call_1\",\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\"}}\n\n",
        )
        .unwrap();
        write_http_chunk(
            &mut stream,
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"error\":null,\"incomplete_details\":null,\"usage\":{\"input_tokens\":40,\"output_tokens\":12,\"total_tokens\":52}}}\n\n",
        )
        .unwrap();
        let _ = done_rx.recv_timeout(Duration::from_secs(2));
    });

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let mut response = client
        .post(format!("http://{addr}/responses"))
        .send()
        .await
        .unwrap();
    let mut app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();
    let mut current_history = String::new();

    let result = tokio::time::timeout(
        Duration::from_secs(2),
        stream_response(&mut app, &mut response, &mut current_history, None),
    )
    .await
    .expect("a provider-declared completion must end the stream without waiting for the socket")
    .unwrap();

    assert_eq!(result.outcome, StreamOutcome::ToolCall);
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(result.tool_calls[0].function.name, "run_shell");
    assert!(!result.stream_error);

    drop(response);
    let _ = done_tx.send(());
    server.join().unwrap();
    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[tokio::test]
async fn stream_response_marks_reasoning_only_early_stop_as_truncated() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request_buf = [0u8; 1024];
        let _ = stream.read(&mut request_buf);
        stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
                )
                .unwrap();
        // Only reasoning was emitted, no visible content was ever produced, no finish_reason was ever
        // sent, and then the connection closes outright (early EOF) — simulating the early-stop of GLM-style
        // enable_thinking models cut off mid chain-of-thought.
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Hmm\"}}]}\n\n",
        )
        .unwrap();
        // Close the chunked body (0-size chunk) then drop the stream to produce an EOF.
        let _ = stream.write_all(b"0\r\n\r\n");
        let _ = stream.flush();
    });

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let mut response = client
        .post(format!("http://{addr}/chat"))
        .send()
        .await
        .unwrap();
    let mut app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();
    let mut current_history = String::new();

    let result = tokio::time::timeout(
        Duration::from_secs(2),
        stream_response(&mut app, &mut response, &mut current_history, None),
    )
    .await
    .expect("stream_response should return promptly on reasoning-only early stop")
    .unwrap();

    // Key assertion: an early stop with only reasoning, no visible text and no finish_reason must
    // escalate to Truncated so the upper layer retries with a downgraded model / thinking off, not silently Completed.
    assert_eq!(result.outcome, StreamOutcome::Truncated);
    assert!(result.assistant_text.trim().is_empty());
    assert_eq!(result.reasoning_text, "Hmm");
    assert!(!result.truncated_by_length);
    assert!(!result.stream_error);

    drop(response);
    server.join().unwrap();
    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[tokio::test]
async fn stream_response_keeps_reading_delayed_chunks_after_finish_reason() {
    let _signal_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request_buf = [0u8; 1024];
        let _ = stream.read(&mut request_buf);
        stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n",
                )
                .unwrap();
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
        )
        .unwrap();
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        )
        .unwrap();
        std::thread::sleep(Duration::from_millis(300));
        write_http_chunk(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
        )
        .unwrap();
        write_http_chunk(&mut stream, "data: [DONE]\n\n").unwrap();
        let _ = done_rx.recv_timeout(Duration::from_secs(2));
    });

    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let mut response = client
        .post(format!("http://{addr}/chat"))
        .send()
        .await
        .unwrap();
    let mut app = test_app();
    init_os_tools_globals(app.os.clone());
    crate::ai::driver::signal::clear_request_interrupt();
    let mut current_history = String::new();

    let result = tokio::time::timeout(
        Duration::from_secs(2),
        stream_response(&mut app, &mut response, &mut current_history, None),
    )
    .await
    .expect("stream_response should keep reading delayed chunks after finish_reason")
    .unwrap();

    assert_eq!(result.outcome, StreamOutcome::Completed);
    assert_eq!(result.assistant_text, "hello world");
    assert_eq!(current_history, "hello world");
    assert!(result.skip_response_drain);

    drop(response);
    let _ = done_tx.send(());
    server.join().unwrap();
    crate::ai::driver::signal::clear_request_interrupt();
    if let Ok(mut guard) = GLOBAL_OS.lock() {
        *guard = None;
    }
}
