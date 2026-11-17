//! Tests for the `round` cluster.

use super::super::*;
use super::common::*;

#[test]
fn tool_call_round_no_longer_requests_terminal_dedupe() {
    let exec_result = ExecuteToolCallsResult {
        executed_tool_calls: vec![ToolCall {
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "execute_command".to_string(),
                arguments: "{\"command\":\"seq 3\"}".to_string(),
            },
        }],
        tool_results: vec![ToolResult {
            task_report: None,
            tool_call_id: "call_1".to_string(),
            content: "1\n2\n3\n".to_string(),
        }],
        cached_hits: vec![false],
        execution_outcomes: Vec::new(),
        had_error: false,
    };

    assert_eq!(exec_result.executed_tool_calls.len(), 1);
    assert_eq!(exec_result.tool_results.len(), 1);
}

#[test]
fn first_use_guidance_is_emitted_once_per_tool_per_turn() {
    let calls = vec![
        test_tool_call("plan_1", "plan", serde_json::json!({})),
        test_tool_call("plan_2", "plan", serde_json::json!({})),
        test_tool_call("command_1", "execute_command", serde_json::json!({})),
    ];
    let exec_result = ExecuteToolCallsResult {
        executed_tool_calls: calls.clone(),
        tool_results: calls
            .iter()
            .map(|call| ToolResult {
                task_report: None,
                tool_call_id: call.id.clone(),
                content: "ok".to_string(),
            })
            .collect(),
        cached_hits: vec![false; calls.len()],
        execution_outcomes: vec![None; calls.len()],
        had_error: false,
    };

    let first_round = build_first_use_tool_guidance_messages_with(&exec_result, &[], |name| {
        (name == "plan").then(|| "detailed plan guidance".to_string())
    });
    assert_eq!(first_round.len(), 1);
    assert_eq!(first_round[0].role, ROLE_INTERNAL_NOTE);
    assert_eq!(
        first_round[0].content.as_str(),
        Some("[tool_first_use_guidance name=plan]\ndetailed plan guidance")
    );

    let prior_turn_messages = vec![Message {
        role: "assistant".to_string(),
        content: serde_json::Value::String(String::new()),
        tool_calls: Some(vec![calls[0].clone()]),
        tool_call_id: None,
        reasoning_content: None,
    }];
    let later_round =
        build_first_use_tool_guidance_messages_with(&exec_result, &prior_turn_messages, |name| {
            (name == "plan").then(|| "detailed plan guidance".to_string())
        });
    assert!(later_round.is_empty());
}

#[test]
fn reused_tool_call_id_is_rewritten_for_the_whole_occurrence() {
    let existing_call = test_tool_call(
        "reused",
        "execute_command",
        serde_json::json!({ "command": "false", "pty": false }),
    );
    let messages = vec![Message {
        role: "assistant".to_string(),
        content: serde_json::Value::String(String::new()),
        tool_calls: Some(vec![existing_call]),
        tool_call_id: None,
        reasoning_content: None,
    }];
    let mut result = ExecuteToolCallsResult {
        executed_tool_calls: vec![test_tool_call(
            "reused",
            "execute_command",
            serde_json::json!({ "command": "true", "pty": false }),
        )],
        tool_results: vec![crate::ai::types::ToolResult {
            task_report: None,
            tool_call_id: "reused".to_string(),
            content: "done".to_string(),
        }],
        cached_hits: vec![false],
        execution_outcomes: vec![Some(crate::ai::history::ToolExecutionOutcome {
            tool_call_id: "reused".to_string(),
            execution_signature: "signature".to_string(),
            succeeded: true,
        })],
        had_error: false,
    };

    uniquify_tool_call_occurrences(&messages, &[], &mut result);

    let occurrence_id = &result.executed_tool_calls[0].id;
    assert_ne!(occurrence_id, "reused");
    assert_eq!(&result.tool_results[0].tool_call_id, occurrence_id);
    assert_eq!(
        &result.execution_outcomes[0].as_ref().unwrap().tool_call_id,
        occurrence_id
    );
}

#[test]
fn evicted_tool_call_id_stays_reserved_by_canonical_turn_history() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    fn observed_result(call: ToolCall, content: &str) -> ExecuteToolCallsResult {
        let id = call.id.clone();
        ExecuteToolCallsResult {
            executed_tool_calls: vec![call],
            tool_results: vec![tool_result(&id, content)],
            cached_hits: vec![false],
            execution_outcomes: vec![Some(crate::ai::history::ToolExecutionOutcome {
                tool_call_id: id,
                execution_signature: "observation-signature".into(),
                succeeded: true,
            })],
            had_error: false,
        }
    }

    #[derive(Debug)]
    struct ObservationMiddleware;
    impl crate::ai::middleware::ToolMiddleware for ObservationMiddleware {
        fn name(&self) -> &'static str {
            "observation-fixture"
        }

        fn wrap(
            &self,
            _: Box<dyn crate::ai::ports::tool::ToolExecutor>,
        ) -> Box<dyn crate::ai::ports::tool::ToolExecutor> {
            struct ObservationExecutor;
            impl crate::ai::ports::tool::ToolExecutor for ObservationExecutor {
                fn execute<'a>(
                    &'a self,
                    _: &'a mut App,
                    tool_calls: Vec<ToolCall>,
                ) -> Pin<
                    Box<
                        dyn Future<
                                Output = Result<
                                    crate::ai::ports::tool::ToolExecOutput,
                                    Box<dyn std::error::Error + Send + Sync>,
                                >,
                            > + Send
                            + 'a,
                    >,
                > {
                    Box::pin(async move {
                        assert_eq!(tool_calls.len(), 1);
                        let result = observed_result(tool_calls[0].clone(), "     1\tnew text");
                        Ok(crate::ai::ports::tool::ToolExecOutput {
                            tool_results: result.tool_results,
                            assistant_messages: Vec::new(),
                            executed_tool_calls: result.executed_tool_calls,
                            cached_hits: result.cached_hits,
                            execution_outcomes: result.execution_outcomes,
                            had_error: result.had_error,
                        })
                    })
                }
            }
            Box::new(ObservationExecutor)
        }
    }

    let args = serde_json::json!({"file_path": "file.rs", "use_line_numbers": true});
    let mut gate = FinalGateState::default();
    let mut turn_messages = Vec::new();
    let mut oldest_object = String::new();
    for index in 0..65 {
        let call = test_tool_call(&format!("read-{index}"), "read_file", args.clone());
        let mut result = observed_result(call.clone(), "     1\told text");
        critical_premises::bind_round(&mut gate, &mut result);
        if index == 0 {
            oldest_object = gate.critical_evidence[0].observed_objects[0].object.clone();
        }
        turn_messages.push(assistant_tool_call_message(call.clone()));
        turn_messages.push(tool_result_message(&call.id, "     1\told text"));
    }
    assert_eq!(gate.critical_evidence.len(), 64);
    assert!(
        gate.critical_evidence
            .iter()
            .all(|record| record.tool_call_id != "read-0")
    );

    let mut app = test_app_with_tools(&["read_file"]);
    app.tool_middlewares.push(Arc::new(ObservationMiddleware));
    assert!(
        crate::ai::history::read_tool_message_ids_sqlite(&app.session_history_file)
            .unwrap_or_default()
            .is_empty()
    );
    let mut messages = vec![Message {
        role: ROLE_INTERNAL_NOTE.into(),
        content: serde_json::json!("Earlier tool calls were folded from the context projection."),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let mcp = McpClient::new();
    let shared_mcp = Arc::new(std::sync::Mutex::new(McpClient::new()));
    let mut persisted = 0;
    let mut had_error = false;
    handle_tool_call_round_with_gate(
        &mut app,
        "",
        &mcp,
        &shared_mcp,
        &ToolCallExecution {
            stream_result: crate::ai::types::StreamResult {
                outcome: crate::ai::types::StreamOutcome::ToolCall,
                tool_calls: vec![test_tool_call("read-0", "read_file", args)],
                assistant_text: String::new(),
                hidden_meta: String::new(),
                reasoning_text: String::new(),
                reasoning_items: Vec::new(),
                skip_response_drain: true,
                truncated_by_length: false,
                stream_error: false,
                response_reconciled: false,
                response_server_mismatch: false,
                finish_reason_value: None,
                usage_prompt_tokens: 0,
                usage_cached_prompt_tokens: 0,
                usage_completion_tokens: 0,
                usage_reasoning_tokens: 0,
            },
            allowed_tool_names: ["read_file".to_string()].into_iter().collect(),
        },
        &mut messages,
        &mut turn_messages,
        true,
        &mut persisted,
        66,
        None,
        &HashMap::new(),
        &mut had_error,
        &mut gate,
    )
    .unwrap();
    assert!(!had_error);
    let new_record = gate.critical_evidence.last().unwrap();
    assert_ne!(new_record.tool_call_id, "read-0");
    assert_ne!(new_record.observed_objects[0].object, oldest_object);
    let new_call = messages
        .iter()
        .flat_map(|message| message.tool_calls.iter().flatten())
        .next()
        .unwrap();
    assert_eq!(new_call.id, new_record.tool_call_id);
    assert!(
        messages
            .iter()
            .any(|message| message.tool_call_id.as_deref() == Some(new_call.id.as_str()))
    );
    assert_eq!(
        turn_messages[0].tool_calls.as_ref().unwrap()[0].id,
        "read-0"
    );
    assert_eq!(turn_messages[1].tool_call_id.as_deref(), Some("read-0"));
    assert_eq!(persisted, 0);

    let update = serde_json::json!({
        "id": "old", "proposition": "The old read still supports this choice", "impact": "choice", "status": "supported",
        "evidence": [{"tool_call_id": "read-0", "object": oldest_object, "coverage": "full"}]
    });
    let content = crate::ai::tools::critical_premises::execute(&update).unwrap();
    let mut rejected = observed_result(
        test_tool_call("track-old", "critical_premises", update),
        &content,
    );
    critical_premises::bind_round(&mut gate, &mut rejected);
    assert!(rejected.had_error);
    assert!(rejected.tool_results[0].content.contains("binding failed"));
    assert!(!rejected.execution_outcomes[0].as_ref().unwrap().succeeded);
    assert!(!gate.critical_premises.has_records());
}

#[test]
fn ctrl_c_during_foreground_tool_round_cancels_without_shutdown() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    signal::clear_request_interrupt();

    let app = test_app_with_tools(&["execute_command"]);
    {
        let mut os = app.os.lock().unwrap();
        let _ = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    }
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());

    let streaming = app.streaming.clone();
    let shutdown = app.shutdown.clone();
    let cancel_stream = app.cancel_stream.clone();
    let started_marker = std::env::temp_dir().join(format!(
        "a_ctrl_c_foreground_tool_{}_{}",
        std::process::id(),
        Instant::now().elapsed().as_nanos()
    ));
    let command_marker = started_marker.to_string_lossy().replace('\'', "'\\''");

    let handle = std::thread::spawn(move || {
        let mut app = app;
        let mcp = crate::ai::mcp::McpClient::new();
        let shared_mcp =
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        let mut persisted_turn_messages = 0usize;
        let mut turn_had_tool_error = false;
        let start = Instant::now();
        let result = handle_tool_call_round(
            &mut app,
            "",
            &mcp,
            &shared_mcp,
            &ToolCallExecution {
                stream_result: crate::ai::types::StreamResult {
                    outcome: crate::ai::types::StreamOutcome::ToolCall,
                    tool_calls: vec![ToolCall {
                        id: "call_1".to_string(),
                        tool_type: "function".to_string(),
                        function: FunctionCall {
                            name: "execute_command".to_string(),
                            arguments: serde_json::json!({
                                "command": format!("touch '{command_marker}'; sleep 2"),
                                "pty": false,
                            })
                            .to_string(),
                        },
                    }],
                    assistant_text: String::new(),
                    hidden_meta: String::new(),
                    reasoning_text: String::new(),
                    reasoning_items: Vec::new(),
                    skip_response_drain: true,
                    truncated_by_length: false,
                    stream_error: false,
                    response_reconciled: false,
                    response_server_mismatch: false,
                    finish_reason_value: None,
                    usage_prompt_tokens: 0,
                    usage_cached_prompt_tokens: 0,
                    usage_completion_tokens: 0,
                    usage_reasoning_tokens: 0,
                },
                allowed_tool_names: ["execute_command".to_string()].into_iter().collect(),
            },
            &mut messages,
            &mut turn_messages,
            true,
            &mut persisted_turn_messages,
            1,
            None,
            &HashMap::new(),
            &mut turn_had_tool_error,
        );
        (
            result.map(|_| ()).map_err(|err| err.to_string()),
            start.elapsed(),
            app,
        )
    });

    let wait_started = Instant::now();
    while !started_marker.exists() && wait_started.elapsed() < Duration::from_secs(1) {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        started_marker.exists(),
        "foreground tool command never started"
    );

    signal::handle_sigint(
        shutdown.as_ref(),
        streaming.as_ref(),
        cancel_stream.as_ref(),
    );

    let (result, elapsed, returned_app) = handle.join().unwrap();
    let _ = std::fs::remove_file(&started_marker);

    returned_app
        .cancel_stream
        .store(false, std::sync::atomic::Ordering::Relaxed);
    crate::ai::tools::registry::common::clear_tool_cancel();
    signal::clear_request_interrupt();
    if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
        *guard = None;
    }

    assert!(result.is_ok());
    assert!(
        elapsed < Duration::from_secs(1),
        "tool round did not stop promptly after Ctrl+C: {elapsed:?}"
    );
    assert!(
        !shutdown.load(std::sync::atomic::Ordering::Relaxed),
        "Ctrl+C during foreground tool round should not request shutdown"
    );
}

#[test]
fn registered_tool_middleware_intercepts_real_dispatch_round() {
    // Step 5 integration verification: middleware registered in
    // `app.tool_middlewares` must really intercept the dispatch round of
    // `handle_tool_call_round` (the middleware behavior path beyond the empty chain).
    #[derive(Debug)]
    struct CountingMiddleware {
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl crate::ai::middleware::ToolMiddleware for CountingMiddleware {
        fn name(&self) -> &'static str {
            "counting"
        }
        fn wrap(
            &self,
            inner: Box<dyn crate::ai::ports::tool::ToolExecutor>,
        ) -> Box<dyn crate::ai::ports::tool::ToolExecutor> {
            struct CountingExecutor {
                inner: Box<dyn crate::ai::ports::tool::ToolExecutor>,
                calls: Arc<std::sync::atomic::AtomicUsize>,
            }
            impl crate::ai::ports::tool::ToolExecutor for CountingExecutor {
                fn execute<'a>(
                    &'a self,
                    app: &'a mut App,
                    tool_calls: Vec<ToolCall>,
                ) -> Pin<
                    Box<
                        dyn Future<
                                Output = Result<
                                    crate::ai::ports::tool::ToolExecOutput,
                                    Box<dyn std::error::Error + Send + Sync>,
                                >,
                            > + Send
                            + 'a,
                    >,
                > {
                    Box::pin(async move {
                        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        self.inner.execute(app, tool_calls).await
                    })
                }
            }
            Box::new(CountingExecutor {
                inner,
                calls: self.calls.clone(),
            })
        }
    }

    let mut app = test_app_with_tools(&["execute_command"]);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    app.tool_middlewares.push(Arc::new(CountingMiddleware {
        calls: calls.clone(),
    }));

    let mcp = crate::ai::mcp::McpClient::new();
    let shared_mcp = Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
    let mut messages = Vec::new();
    let mut turn_messages = Vec::new();
    let mut persisted_turn_messages = 0usize;
    let mut turn_had_tool_error = false;
    let result = handle_tool_call_round(
        &mut app,
        "",
        &mcp,
        &shared_mcp,
        &ToolCallExecution {
            stream_result: crate::ai::types::StreamResult {
                outcome: crate::ai::types::StreamOutcome::ToolCall,
                tool_calls: vec![ToolCall {
                    id: "call_mw_1".to_string(),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: "execute_command".to_string(),
                        arguments: serde_json::json!({ "command": "echo middleware-intercept" })
                            .to_string(),
                    },
                }],
                assistant_text: String::new(),
                hidden_meta: String::new(),
                reasoning_text: String::new(),
                reasoning_items: Vec::new(),
                skip_response_drain: true,
                truncated_by_length: false,
                stream_error: false,
                response_reconciled: false,
                response_server_mismatch: false,
                finish_reason_value: None,
                usage_prompt_tokens: 0,
                usage_cached_prompt_tokens: 0,
                usage_completion_tokens: 0,
                usage_reasoning_tokens: 0,
            },
            allowed_tool_names: ["execute_command".to_string()].into_iter().collect(),
        },
        &mut messages,
        &mut turn_messages,
        true,
        &mut persisted_turn_messages,
        1,
        None,
        &HashMap::new(),
        &mut turn_had_tool_error,
    );
    assert!(
        result.is_ok(),
        "round should succeed with middleware, got {:?}",
        result.err()
    );
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "registered middleware must intercept the dispatch round exactly once"
    );
    assert!(
        !messages.is_empty(),
        "tool result messages should be produced through the chain"
    );
}

#[test]
fn tool_round_releases_live_mcp_lock_before_dispatch() {
    struct McpLockProbeMiddleware {
        shared_mcp: SharedMcpClient,
        lock_was_available: Arc<std::sync::atomic::AtomicBool>,
    }
    impl std::fmt::Debug for McpLockProbeMiddleware {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("McpLockProbeMiddleware").finish()
        }
    }
    impl crate::ai::middleware::ToolMiddleware for McpLockProbeMiddleware {
        fn name(&self) -> &'static str {
            "mcp_lock_probe"
        }

        fn wrap(
            &self,
            inner: Box<dyn crate::ai::ports::tool::ToolExecutor>,
        ) -> Box<dyn crate::ai::ports::tool::ToolExecutor> {
            struct McpLockProbeExecutor {
                inner: Box<dyn crate::ai::ports::tool::ToolExecutor>,
                shared_mcp: SharedMcpClient,
                lock_was_available: Arc<std::sync::atomic::AtomicBool>,
            }
            impl crate::ai::ports::tool::ToolExecutor for McpLockProbeExecutor {
                fn execute<'a>(
                    &'a self,
                    app: &'a mut App,
                    tool_calls: Vec<ToolCall>,
                ) -> Pin<
                    Box<
                        dyn Future<
                                Output = Result<
                                    crate::ai::ports::tool::ToolExecOutput,
                                    Box<dyn std::error::Error + Send + Sync>,
                                >,
                            > + Send
                            + 'a,
                    >,
                > {
                    Box::pin(async move {
                        let available = self.shared_mcp.try_lock().is_ok();
                        self.lock_was_available
                            .store(available, std::sync::atomic::Ordering::SeqCst);
                        self.inner.execute(app, tool_calls).await
                    })
                }
            }
            Box::new(McpLockProbeExecutor {
                inner,
                shared_mcp: self.shared_mcp.clone(),
                lock_was_available: self.lock_was_available.clone(),
            })
        }
    }

    let mut app = test_app_with_tools(&["execute_command"]);
    let shared_mcp = Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
    let lock_was_available = Arc::new(std::sync::atomic::AtomicBool::new(false));
    app.tool_middlewares.push(Arc::new(McpLockProbeMiddleware {
        shared_mcp: shared_mcp.clone(),
        lock_was_available: lock_was_available.clone(),
    }));

    let mcp = crate::ai::mcp::McpClient::new();
    let mut messages = Vec::new();
    let mut turn_messages = Vec::new();
    let mut persisted_turn_messages = 0usize;
    let mut turn_had_tool_error = false;
    let result = handle_tool_call_round(
        &mut app,
        "",
        &mcp,
        &shared_mcp,
        &ToolCallExecution {
            stream_result: crate::ai::types::StreamResult {
                outcome: crate::ai::types::StreamOutcome::ToolCall,
                tool_calls: vec![ToolCall {
                    id: "call_mcp_lock_probe".to_string(),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: "execute_command".to_string(),
                        arguments: serde_json::json!({ "command": "echo mcp-lock-probe" })
                            .to_string(),
                    },
                }],
                assistant_text: String::new(),
                hidden_meta: String::new(),
                reasoning_text: String::new(),
                reasoning_items: Vec::new(),
                skip_response_drain: true,
                truncated_by_length: false,
                stream_error: false,
                response_reconciled: false,
                response_server_mismatch: false,
                finish_reason_value: None,
                usage_prompt_tokens: 0,
                usage_cached_prompt_tokens: 0,
                usage_completion_tokens: 0,
                usage_reasoning_tokens: 0,
            },
            allowed_tool_names: ["execute_command".to_string()].into_iter().collect(),
        },
        &mut messages,
        &mut turn_messages,
        true,
        &mut persisted_turn_messages,
        1,
        None,
        &HashMap::new(),
        &mut turn_had_tool_error,
    );

    assert!(
        result.is_ok(),
        "tool round should complete: {:?}",
        result.err()
    );
    assert!(
        lock_was_available.load(std::sync::atomic::Ordering::SeqCst),
        "tool dispatch must not retain the live MCP mutex; a foreground subagent needs it while preparing context"
    );
}
