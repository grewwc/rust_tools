use super::*;

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
async fn stream_response_marks_length_finish_reason_as_truncated() {
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

    // Key assertion: text present but finish_reason=length is treated as Completed. Reasoning models
    // often exhaust the output budget on reasoning tokens, yielding finish_reason=length while the
    // visible assistant_text is actually complete; retrying would only truncate again for nothing. Only
    // truncated tool call arguments JSON (dropped_malformed_tool_call) or no visible output at all
    // should escalate to Truncated and trigger a retry.
    assert_eq!(result.outcome, StreamOutcome::Completed);
    assert_eq!(result.assistant_text, "partial output");

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
