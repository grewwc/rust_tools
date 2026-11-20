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
#[cfg(unix)]
fn reused_tool_call_id_real_dispatch_matches_mutation_and_check_locators() {
    use crate::ai::driver::runtime_ctx;
    use crate::ai::tools::storage::{command_session, mutation_log};
    use std::os::unix::fs::PermissionsExt;

    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let _session_guard = command_session::tests::TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    struct SessionCleanup(Vec<String>);
    impl Drop for SessionCleanup {
        fn drop(&mut self) {
            for id in &self.0 {
                let _ = command_session::close(id);
            }
        }
    }

    #[derive(Debug)]
    struct CaptureIds(Arc<std::sync::Mutex<Vec<String>>>);
    impl crate::ai::middleware::ToolMiddleware for CaptureIds {
        fn name(&self) -> &'static str {
            "capture-occurrence-ids"
        }

        fn wrap(&self, inner: Box<dyn ToolExecutor>) -> Box<dyn ToolExecutor> {
            struct CaptureExecutor {
                inner: Box<dyn ToolExecutor>,
                ids: Arc<std::sync::Mutex<Vec<String>>>,
            }
            impl ToolExecutor for CaptureExecutor {
                fn execute<'a>(
                    &'a self,
                    app: &'a mut App,
                    calls: Vec<ToolCall>,
                ) -> Pin<
                    Box<
                        dyn Future<
                                Output = Result<
                                    ToolExecOutput,
                                    Box<dyn std::error::Error + Send + Sync>,
                                >,
                            > + Send
                            + 'a,
                    >,
                > {
                    self.ids
                        .lock()
                        .unwrap()
                        .extend(calls.iter().map(|call| call.id.clone()));
                    self.inner.execute(app, calls)
                }
            }
            Box::new(CaptureExecutor {
                inner,
                ids: self.0.clone(),
            })
        }
    }

    let root = std::env::temp_dir().join(format!("a-round-id-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let _cleanup = Cleanup(root.clone());
    // A local check fixture runs through the real command service. Release files
    // control completion order without depending on compiler timing or network.
    let script = root.join("pytest");
    fs::write(
        &script,
        "#!/bin/sh\nstate=$(cat a.rs)\n: > \"started-$state\"\nwhile [ ! -f \"release-$state\" ]; do sleep 0.01; done\nif [ \"$state\" = first ]; then printf '1 failed in 0.01s\\n'; exit 1; fi\nprintf '1 passed in 0.01s\\n'\n",
    )
    .unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    let names = [
        "write_file",
        "execute_command",
        "command_session",
        "read_file",
    ];
    let mut app = test_app_with_tools(&names);
    app.session_id = "reused-call-round".into();
    app.config.history_file = root.join(".git/runtime/history.sqlite");
    app.config.base_history_file = app.config.history_file.clone();
    app.session_history_file = crate::ai::history::SessionStore::new(&app.config.history_file)
        .session_history_file(&app.session_id);
    let seen_ids = Arc::new(std::sync::Mutex::new(Vec::new()));
    app.tool_middlewares
        .push(Arc::new(CaptureIds(seen_ids.clone())));
    let shared_mcp = Arc::new(std::sync::Mutex::new(McpClient::new()));
    let mcp = mcp_snapshot(&shared_mcp);
    let context = runtime_ctx::DriverContext::from_app_snapshot(
        &app,
        shared_mcp.clone(),
        Arc::new(Vec::new()),
        Arc::new(Vec::new()),
    );
    runtime_ctx::DRIVER_CTX.sync_scope(context, || {
        runtime_ctx::SUBAGENT_CWD.sync_scope(root.clone(), || {
            runtime_ctx::SUPPRESS_TERMINAL_OUTPUT.sync_scope(true, || {
                let mut sessions = SessionCleanup(Vec::new());
                let mut messages = Vec::new();
                let mut turn_messages = Vec::new();
                let mut persisted = 0;
                let mut had_error = false;
                let mut gate = FinalGateState::default();
                let mut iteration = 0;
                let mut run = |name: &str, args: Value, suppressed: &HashMap<String, String>| {
                    iteration += 1;
                    handle_tool_call_round_with_gate(
                        &mut app,
                        "",
                        &mcp,
                        &shared_mcp,
                        &ToolCallExecution {
                            stream_result: crate::ai::types::StreamResult {
                                outcome: crate::ai::types::StreamOutcome::ToolCall,
                                tool_calls: vec![test_tool_call("reused", name, args)],
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
                            allowed_tool_names: names.iter().map(|name| name.to_string()).collect(),
                        },
                        &mut messages,
                        &mut turn_messages,
                        false,
                        &mut persisted,
                        iteration,
                        None,
                        suppressed,
                        &mut had_error,
                        &mut gate,
                    )
                    .unwrap();
                    let result = turn_messages
                        .iter()
                        .rev()
                        .find(|message| message.role == "tool")
                        .unwrap();
                    (
                        result.tool_call_id.clone().unwrap(),
                        result.content.as_str().unwrap().to_string(),
                    )
                };
                let empty = HashMap::new();
                let (write_old, old_write) = run(
                    "write_file",
                    serde_json::json!({"file_path": "a.rs", "content": "first"}),
                    &empty,
                );
                assert!(old_write.contains("Successfully wrote"), "{old_write}");
                let check_args = serde_json::json!({
                    "command": "./pytest -q", "pty": false, "yield_time_ms": 5000,
                    "cwd": root.to_string_lossy(),
                });
                let (check_old, old_start) = run("execute_command", check_args.clone(), &empty);
                let (old_session, old_status) =
                    crate::ai::driver::turn_runtime::command_session_observation(&old_start).unwrap();
                assert!(old_start.contains("[command session running]"), "{old_start}");
                let old_session = old_session.unwrap();
                sessions.0.push(old_session.clone());
                assert!(fs::metadata(root.join("started-first")).is_ok());
                let (write_new, new_write) = run(
                    "write_file",
                    serde_json::json!({"file_path": "a.rs", "content": "second"}),
                    &empty,
                );
                assert!(new_write.contains("Successfully wrote"), "{new_write}");
                let (check_new, new_start) = run("execute_command", check_args, &empty);
                let (new_session, new_status) =
                    crate::ai::driver::turn_runtime::command_session_observation(&new_start).unwrap();
                assert!(new_start.contains("[command session running]"), "{new_start}");
                assert_eq!(old_status, new_status);
                let new_session = new_session.unwrap();
                sessions.0.push(new_session.clone());
                assert_ne!(old_session, new_session);
                assert!(fs::metadata(root.join("started-second")).is_ok());
                fs::write(root.join("release-second"), "").unwrap();
                let (poll_new, new_result) = run(
                    "command_session",
                    serde_json::json!({"action": "poll", "session_id": new_session, "yield_time_ms": 5000}),
                    &empty,
                );
                assert!(new_result.contains("exit code 0"), "{new_result}");
                fs::write(root.join("release-first"), "").unwrap();
                let (poll_old, old_result) = run(
                    "command_session",
                    serde_json::json!({"action": "poll", "session_id": old_session, "yield_time_ms": 5000}),
                    &empty,
                );
                assert!(old_result.contains("exit code 1"), "{old_result}");
                let (suppressed_id, suppressed_result) = run(
                    "read_file",
                    serde_json::json!({"file_path": "does-not-exist"}),
                    &[("reused".to_string(), "Read suppressed by prior observation".to_string())]
                        .into_iter()
                        .collect(),
                );
                assert_eq!(suppressed_result, "Read suppressed by prior observation");
                drop(run);

                let expected_ids = [
                    write_old.clone(), check_old.clone(), write_new.clone(), check_new.clone(),
                    poll_new.clone(), poll_old.clone(), suppressed_id,
                ];
                assert!(expected_ids.iter().all(|id| id != "reused"));
                assert_eq!(expected_ids.iter().collect::<FastSet<_>>().len(), expected_ids.len());
                assert_eq!(*seen_ids.lock().unwrap(), expected_ids);
                let entries = mutation_log::read_all();
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0].tool_call_id.as_deref(), Some(write_old.as_str()));
                assert_eq!(entries[1].tool_call_id.as_deref(), Some(write_new.as_str()));
                let checks = mutation_log::read_checks();
                for (origin, poll, revision, status) in [
                    (&check_old, &poll_old, 1, mutation_log::CheckStatus::Failure),
                    (&check_new, &poll_new, 2, mutation_log::CheckStatus::Success),
                ] {
                    let observations = checks.iter()
                        .filter(|check| &check.originating_command_call_id == origin)
                        .collect::<Vec<_>>();
                    assert_eq!(observations.len(), 3);
                    assert!(observations.iter().all(|check| check.covers_revision == revision));
                    assert_eq!(&observations[0].tool_call_id, origin);
                    assert_eq!(&observations[1].tool_call_id, origin);
                    assert_eq!(&observations[2].tool_call_id, poll);
                    assert_eq!(observations[2].status, status);
                }
                let summary = mutation_log::verification_summary(2, &checks, true);
                assert_eq!(summary["verification"]["status"], "success");
                assert_eq!(summary["verification_gaps"]["count"], 0);
                assert_eq!(summary["verification"]["checks"].as_array().unwrap().len(), 1);
                assert_eq!(
                    summary["verification"]["checks"][0]["observation"]["originating_command_call_id"],
                    check_new,
                );
                let history = crate::ai::history::read_all_messages_sqlite(&app.session_history_file).unwrap();
                assert_eq!(history, turn_messages);
                assert_eq!(persisted, turn_messages.len());
                let persisted_calls = history.iter()
                    .flat_map(|message| message.tool_calls.iter().flatten())
                    .map(|call| call.id.clone())
                    .collect::<Vec<_>>();
                assert_eq!(persisted_calls, expected_ids);
                for check in &checks {
                    assert_eq!(check.evidence_locator, format!("tool_call_id:{}", check.tool_call_id));
                    assert!(persisted_calls.contains(&check.originating_command_call_id));
                    assert!(history.iter().any(|message| {
                        message.tool_call_id.as_deref() == Some(check.tool_call_id.as_str())
                    }));
                }
                let outcomes = crate::ai::history::read_tool_execution_outcomes_sqlite(&app.session_history_file).unwrap();
                assert!(outcomes.iter().all(|outcome| persisted_calls.contains(&outcome.tool_call_id)));
            })
        })
    });
}

#[test]
fn fresh_tool_call_ids_do_not_depend_on_saved_history() {
    let provider_id = "call_00000000000040008000000000000000";
    let result = || ExecuteToolCallsResult {
        executed_tool_calls: vec![test_tool_call(
            provider_id,
            "execute_command",
            serde_json::json!({"command": "true"}),
        )],
        tool_results: vec![ToolResult {
            task_report: None,
            tool_call_id: provider_id.into(),
            content: "done".into(),
        }],
        cached_hits: vec![false],
        execution_outcomes: vec![None],
        had_error: false,
    };
    let mut first = result();
    let mut recovered = result();
    uniquify_tool_call_occurrences(&[], &[], &mut first);
    uniquify_tool_call_occurrences(&[], &[], &mut recovered);
    let first_id = &first.executed_tool_calls[0].id;
    let recovered_id = &recovered.executed_tool_calls[0].id;
    assert_ne!(first_id, provider_id);
    assert_ne!(recovered_id, provider_id);
    assert_ne!(first_id, recovered_id);
    assert_eq!(&first.tool_results[0].tool_call_id, first_id);
    assert_eq!(&recovered.tool_results[0].tool_call_id, recovered_id);
}

#[cfg(unix)]
fn check_identity_after_uncommitted_round_recovery(same_command: bool, delayed_poll: bool) {
    use crate::ai::driver::runtime_ctx;
    use crate::ai::tools::storage::{command_session, mutation_log};
    use std::os::unix::fs::PermissionsExt;

    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let _session_guard = command_session::tests::TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());

    struct Cleanup {
        root: PathBuf,
        sessions: Vec<String>,
    }
    impl Drop for Cleanup {
        fn drop(&mut self) {
            for session in &self.sessions {
                let _ = command_session::close(session);
            }
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[derive(Debug)]
    struct InterruptAfterDispatch;
    impl crate::ai::middleware::ToolMiddleware for InterruptAfterDispatch {
        fn name(&self) -> &'static str {
            "interrupt-before-history-commit"
        }

        fn wrap(&self, inner: Box<dyn ToolExecutor>) -> Box<dyn ToolExecutor> {
            struct InterruptExecutor(Box<dyn ToolExecutor>);
            impl ToolExecutor for InterruptExecutor {
                fn execute<'a>(
                    &'a self,
                    app: &'a mut App,
                    calls: Vec<ToolCall>,
                ) -> Pin<
                    Box<
                        dyn Future<
                                Output = Result<
                                    ToolExecOutput,
                                    Box<dyn std::error::Error + Send + Sync>,
                                >,
                            > + Send
                            + 'a,
                    >,
                > {
                    Box::pin(async move {
                        self.0.execute(app, calls).await?;
                        Err(std::io::Error::other("interrupted before history commit").into())
                    })
                }
            }
            Box::new(InterruptExecutor(inner))
        }
    }

    #[derive(Default)]
    struct RoundState {
        messages: Vec<Message>,
        turn_messages: Vec<Message>,
        persisted: usize,
        gate: FinalGateState,
        had_error: bool,
    }
    impl RoundState {
        fn run(
            &mut self,
            app: &mut App,
            shared_mcp: &SharedMcpClient,
            name: &str,
            args: Value,
        ) -> Result<(String, String), String> {
            handle_tool_call_round_with_gate(
                app,
                "",
                &mcp_snapshot(shared_mcp),
                shared_mcp,
                &ToolCallExecution {
                    stream_result: crate::ai::types::StreamResult {
                        outcome: crate::ai::types::StreamOutcome::ToolCall,
                        tool_calls: vec![test_tool_call("reused", name, args)],
                        skip_response_drain: true,
                        ..Default::default()
                    },
                    allowed_tool_names: ["write_file", "execute_command", "command_session"]
                        .into_iter()
                        .map(str::to_string)
                        .collect(),
                },
                &mut self.messages,
                &mut self.turn_messages,
                false,
                &mut self.persisted,
                1,
                None,
                &HashMap::new(),
                &mut self.had_error,
                &mut self.gate,
            )
            .map_err(|error| error.to_string())?;
            let result = self
                .turn_messages
                .iter()
                .rev()
                .find(|message| message.role == "tool")
                .unwrap();
            Ok((
                result.tool_call_id.clone().unwrap(),
                result.content.as_str().unwrap().to_string(),
            ))
        }
    }

    let root = std::env::temp_dir().join(format!("a-round-recovery-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&root).unwrap();
    let mut cleanup = Cleanup {
        root: root.clone(),
        sessions: Vec::new(),
    };
    let script = root.join("pytest");
    fs::write(
        &script,
        "#!/bin/sh\nstate=$(cat a.rs)\n: > \"started-$state\"\nif [ -f delayed ] && [ \"$state\" = first ]; then while [ ! -f release-first ]; do sleep 0.01; done; fi\nif [ \"$state\" = first ]; then printf '1 failed in 0.01s\\n'; exit 1; fi\nprintf '1 passed in 0.01s\\n'\n",
    ).unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    if delayed_poll {
        fs::write(root.join("delayed"), "").unwrap();
    }
    let make_app = || {
        let mut app = test_app_with_tools(&["write_file", "execute_command", "command_session"]);
        app.session_id = "recovered-call-round".into();
        app.config.history_file = root.join(".git/runtime/history.sqlite");
        app.config.base_history_file = app.config.history_file.clone();
        app.session_history_file = crate::ai::history::SessionStore::new(&app.config.history_file)
            .session_history_file(&app.session_id);
        app
    };
    let mut app = make_app();
    let shared_mcp = Arc::new(std::sync::Mutex::new(McpClient::new()));
    let context = runtime_ctx::DriverContext::from_app_snapshot(
        &app,
        shared_mcp.clone(),
        Arc::new(Vec::new()),
        Arc::new(Vec::new()),
    );
    runtime_ctx::DRIVER_CTX.sync_scope(context, || {
        runtime_ctx::SUBAGENT_CWD.sync_scope(root.clone(), || {
            runtime_ctx::SUPPRESS_TERMINAL_OUTPUT.sync_scope(true, || {
                let mut state = RoundState::default();
                let (_, written) = state.run(&mut app, &shared_mcp, "write_file",
                    serde_json::json!({"file_path": "a.rs", "content": "first"})).unwrap();
                assert!(written.contains("Successfully wrote"), "{written}");
                let saved_history = crate::ai::history::read_all_messages_sqlite(&app.session_history_file).unwrap();
                let old_args = serde_json::json!({
                    "command": "./pytest -q", "pty": false, "yield_time_ms": 5000,
                    "cwd": root.to_string_lossy(),
                });
                app.tool_middlewares.push(Arc::new(InterruptAfterDispatch));
                let error = state.run(&mut app, &shared_mcp, "execute_command", old_args.clone()).unwrap_err();
                assert!(error.contains("interrupted before history commit"), "{error}");
                let old_checks = mutation_log::read_checks();
                assert_eq!(old_checks.len(), 2);
                let old_origin = old_checks[0].originating_command_call_id.clone();
                assert_ne!(old_origin, "reused");
                assert!(old_checks.iter().all(|check| check.covers_revision == 1 && check.originating_command_call_id == old_origin));
                if let Some(session) = &old_checks[1].command_session_id {
                    cleanup.sessions.push(session.clone());
                }
                assert_eq!(old_checks[1].status, if delayed_poll { mutation_log::CheckStatus::Running } else { mutation_log::CheckStatus::Failure });
                assert!(fs::metadata(root.join("started-first")).is_ok());
                assert_eq!(crate::ai::history::read_all_messages_sqlite(&app.session_history_file).unwrap(), saved_history);
                assert_eq!(state.turn_messages, saved_history);
                assert_eq!(state.persisted, saved_history.len());
                assert!(!crate::ai::history::read_tool_message_ids_sqlite(&app.session_history_file).unwrap().contains(&old_origin));

                // Discard the interrupted runtime and preserve exactly the durable log.
                // Replacing its inode also forces the incremental index to recover it.
                let log_path = mutation_log::log_path().unwrap();
                let saved_log = fs::read(&log_path).unwrap();
                fs::rename(&log_path, log_path.with_extension("saved")).unwrap();
                fs::write(&log_path, &saved_log).unwrap();
                drop(state);
                drop(app);
                let mut app = make_app();
                let mut state = RoundState {
                    messages: saved_history.clone(),
                    turn_messages: saved_history.clone(),
                    persisted: saved_history.len(),
                    ..Default::default()
                };
                assert!(state.gate.critical_evidence.is_empty());
                assert!(state.gate.critical_prior_tool_call_ids.is_empty());
                assert_eq!(fs::read(&log_path).unwrap(), saved_log);
                let (_, written) = state.run(&mut app, &shared_mcp, "write_file",
                    serde_json::json!({"file_path": "a.rs", "content": "second"})).unwrap();
                assert!(written.contains("Successfully wrote"), "{written}");
                let mut new_args = old_args;
                if !same_command {
                    new_args["command"] = serde_json::json!("./pytest -q new");
                }
                let (new_origin, new_result) = state.run(&mut app, &shared_mcp, "execute_command", new_args).unwrap();
                assert_ne!(new_origin, old_origin);
                assert_ne!(new_origin, "reused");
                assert!(new_result.contains("exit code 0"), "{new_result}");
                let poll_id = if delayed_poll {
                    fs::write(root.join("release-first"), "").unwrap();
                    let (id, result) = state.run(&mut app, &shared_mcp, "command_session",
                        serde_json::json!({"action": "poll", "session_id": cleanup.sessions[0], "yield_time_ms": 5000})).unwrap();
                    assert!(result.contains("exit code 1"), "{result}");
                    assert_ne!(id, old_origin);
                    assert_ne!(id, new_origin);
                    assert_ne!(id, "reused");
                    Some(id)
                } else { None };
                let checks = mutation_log::read_checks();
                let old = checks.iter().filter(|check| check.originating_command_call_id == old_origin).collect::<Vec<_>>();
                let new = checks.iter().filter(|check| check.originating_command_call_id == new_origin).collect::<Vec<_>>();
                assert_eq!(old.len(), if delayed_poll { 3 } else { 2 });
                assert_eq!(new.len(), 2);
                assert_eq!(&checks[..2], old_checks.as_slice());
                assert!(old.iter().all(|check| check.command == "./pytest -q" && check.covers_revision == 1));
                assert_eq!(old.last().unwrap().status, mutation_log::CheckStatus::Failure);
                assert!(new.iter().all(|check| check.covers_revision == 2));
                assert_eq!(new.last().unwrap().status, mutation_log::CheckStatus::Success);
                if let Some(id) = poll_id {
                    let terminal = old.last().unwrap();
                    assert_eq!(terminal.tool_call_id, id);
                    assert_eq!(terminal.evidence_locator, format!("tool_call_id:{id}"));
                }
                let summary = mutation_log::verification_summary(2, &checks, true);
                assert_eq!(summary["verification"]["status"], if same_command { "success" } else { "stale" });
                assert_eq!(summary["verification_gaps"]["count"], if same_command { 0 } else { 1 });
                assert_eq!(summary["verification"]["checks"].as_array().unwrap().len(), if same_command { 1 } else { 2 });
                assert!(summary["verification"]["checks"].as_array().unwrap().iter().any(|check| check["observation"]["originating_command_call_id"] == new_origin && check["observation"]["status"] == "success"));
                let reminder = mutation_log::verification_reminder_summary();
                assert_eq!(reminder.revision, 2);
                assert_eq!(reminder.gaps, if same_command { 0 } else { 1 });
                assert_eq!(reminder.stale_checks, if same_command { 0 } else { 1 });
                let history = crate::ai::history::read_all_messages_sqlite(&app.session_history_file).unwrap();
                assert_eq!(history, state.turn_messages);
                assert!(history.iter().any(|message| message.tool_call_id.as_deref() == Some(new_origin.as_str())));
                assert!(!history.iter().any(|message| message.tool_call_id.as_deref() == Some(old_origin.as_str())));
            })
        })
    });
}

#[test]
#[cfg(unix)]
fn recovered_tool_call_id_same_command_keeps_independent_origins() {
    check_identity_after_uncommitted_round_recovery(true, false);
}

#[test]
#[cfg(unix)]
fn recovered_tool_call_id_different_command_retains_stale_failure() {
    check_identity_after_uncommitted_round_recovery(false, false);
}

#[test]
#[cfg(unix)]
fn recovered_tool_call_id_delayed_old_poll_cannot_replace_new_check() {
    check_identity_after_uncommitted_round_recovery(true, true);
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
    // Middleware registered in app.tool_middlewares must intercept the real
    // handle_tool_call_round dispatch, including a non-empty chain.
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
