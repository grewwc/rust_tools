//! Tests for the `final_recovery` cluster.

use super::common::*;
use super::super::*;

#[test]
fn task_evidence_reopen_marker_counts_and_survives_reopen_retain() {
    let mut messages: Vec<Message> = Vec::new();
    assert_eq!(task_evidence_reopen_count(&messages), 0);

    // Inject one count marker per reopen; the count accumulates with it.
    push_task_evidence_reopen_marker(&mut messages, 1);
    assert_eq!(task_evidence_reopen_count(&messages), 1);
    push_task_evidence_reopen_marker(&mut messages, 2);
    assert_eq!(task_evidence_reopen_count(&messages), 2);

    // Key invariant: the unintegrated-evidence reopen retain must not clear the count
    // markers, or the quota would never fill and we would regress to endless reopens.
    reopen_turn_for_unintegrated_task_evidence(&mut messages, "[task-evidence-ledger]\ntask_id=t");
    assert_eq!(
        task_evidence_reopen_count(&messages),
        2,
        "reopen must not erase the reopen-count markers"
    );
}

#[test]
fn task_evidence_reopen_quota_is_bounded() {
    // The quota cap exists and is far below the iteration hard cap (DEFAULT_MAX_ITERATIONS
    // = 64*64 = 4096), guaranteeing that dead ends (TIMED_OUT / refusing to integrate)
    // never reopen forever.
    assert!(TASK_EVIDENCE_REOPEN_MAX >= 1);
    assert!(TASK_EVIDENCE_REOPEN_MAX < 64 * 64);

    let mut messages: Vec<Message> = Vec::new();
    for attempt in 1..=TASK_EVIDENCE_REOPEN_MAX {
        assert!(
            task_evidence_reopen_count(&messages) < TASK_EVIDENCE_REOPEN_MAX,
            "budget must not be exhausted before the cap"
        );
        push_task_evidence_reopen_marker(&mut messages, attempt);
    }
    assert_eq!(task_evidence_reopen_count(&messages), TASK_EVIDENCE_REOPEN_MAX);
    assert!(
        task_evidence_reopen_count(&messages) >= TASK_EVIDENCE_REOPEN_MAX,
        "after the cap, reopen budget is exhausted and the turn finalizes"
    );
}

#[test]
fn reasoning_only_final_response_retries_once_with_full_capabilities() {
    let mut app = test_app_with_tools(&["read_file"]);
    let mcp = crate::ai::mcp::McpClient::new();
    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(mcp));
    let mut messages = Vec::new();
    let mut turn_messages = Vec::new();
    let mut persisted_turn_messages = 0usize;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut force_final_response = false;
    let mut terminal_dedupe_candidate = None;

    let step = handle_iteration_execution(
        &mut app,
        "compare two yaml files",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        IterationExecution::FinalResponse(crate::ai::types::StreamResult {
            outcome: crate::ai::types::StreamOutcome::Completed,
            tool_calls: Vec::new(),
            assistant_text: String::new(),
            hidden_meta: String::new(),
            reasoning_text: "I should read both files first.".to_string(),
            reasoning_items: Vec::new(),
            skip_response_drain: true,
            truncated_by_length: false,
            stream_error: false,
            finish_reason_value: None,
            usage_prompt_tokens: 0,
            usage_cached_prompt_tokens: 0,
            usage_completion_tokens: 0,
            usage_reasoning_tokens: 0,
        }),
        &mut messages,
        &mut turn_messages,
        false,
        &mut persisted_turn_messages,
        &mut final_assistant_text,
        &mut final_assistant_recorded,
        &mut force_final_response,
        &mut terminal_dedupe_candidate,
        true,
        1,
        16,
        0,
        &mut false,
    )
    .unwrap();

    assert!(matches!(step, TurnLoopStep::Continue));
    assert!(!force_final_response);
    assert!(!app.cli.thinking_disabled_override);
    assert!(final_assistant_text.is_empty());
    assert!(!final_assistant_recorded);
    assert!(messages.iter().any(|message| {
        message.role == ROLE_INTERNAL_NOTE
            && message
                .content
                .as_str()
                .is_some_and(|text| text.starts_with(REASONING_ONLY_RETRY_MARKER))
    }));
    assert!(turn_messages.is_empty());
}

#[test]
fn reasoning_only_final_response_forces_no_thinking_synthesis_after_normal_retry() {
    let mut app = test_app_with_tools(&["read_file"]);
    let mcp = crate::ai::mcp::McpClient::new();
    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(mcp));
    let mut messages = vec![Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: serde_json::Value::String(format!(
            "{REASONING_ONLY_RETRY_MARKER}\n{REASONING_ONLY_RETRY_NOTE}"
        )),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let mut turn_messages = Vec::new();
    let mut persisted_turn_messages = 0usize;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut force_final_response = true;
    let mut terminal_dedupe_candidate = None;

    let step = handle_iteration_execution(
        &mut app,
        "compare two yaml files",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        IterationExecution::FinalResponse(crate::ai::types::StreamResult {
            outcome: crate::ai::types::StreamOutcome::Completed,
            tool_calls: Vec::new(),
            assistant_text: String::new(),
            hidden_meta: String::new(),
            reasoning_text: "I should read both files first.".to_string(),
            reasoning_items: Vec::new(),
            skip_response_drain: true,
            truncated_by_length: false,
            stream_error: false,
            finish_reason_value: None,
            usage_prompt_tokens: 0,
            usage_cached_prompt_tokens: 0,
            usage_completion_tokens: 0,
            usage_reasoning_tokens: 0,
        }),
        &mut messages,
        &mut turn_messages,
        false,
        &mut persisted_turn_messages,
        &mut final_assistant_text,
        &mut final_assistant_recorded,
        &mut force_final_response,
        &mut terminal_dedupe_candidate,
        true,
        2,
        16,
        0,
        &mut false,
    )
    .unwrap();

    assert!(matches!(step, TurnLoopStep::Continue));
    assert!(app.cli.thinking_disabled_override);
    assert!(force_final_response);
    assert!(final_assistant_text.is_empty());
    assert!(!final_assistant_recorded);
    assert!(messages.iter().any(|message| {
        message.role == ROLE_INTERNAL_NOTE
            && message
                .content
                .as_str()
                .is_some_and(|text| text.starts_with(REASONING_ONLY_SYNTHESIS_MARKER))
    }));
    assert!(turn_messages.is_empty());

    let second_step = handle_iteration_execution(
        &mut app,
        "compare two yaml files",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        IterationExecution::FinalResponse(crate::ai::types::StreamResult {
            outcome: crate::ai::types::StreamOutcome::Completed,
            tool_calls: Vec::new(),
            assistant_text: String::new(),
            hidden_meta: String::new(),
            reasoning_text: "Still hidden reasoning".to_string(),
            reasoning_items: Vec::new(),
            skip_response_drain: true,
            truncated_by_length: false,
            stream_error: false,
            finish_reason_value: None,
            usage_prompt_tokens: 0,
            usage_cached_prompt_tokens: 0,
            usage_completion_tokens: 0,
            usage_reasoning_tokens: 0,
        }),
        &mut messages,
        &mut turn_messages,
        false,
        &mut persisted_turn_messages,
        &mut final_assistant_text,
        &mut final_assistant_recorded,
        &mut force_final_response,
        &mut terminal_dedupe_candidate,
        true,
        3,
        16,
        0,
        &mut false,
    )
    .unwrap();

    // After the forced synthesis the model still returns reasoning-only: do not stop
    // early; keep the forced state and continue auto-retrying without re-injecting the
    // synthesis note; but inject one lightweight synthesis-retry marker per attempt
    // (counted against REASONING_ONLY_POST_SYNTHESIS_MAX_RETRIES) to avoid empty
    // spins on identical byte-for-byte requests.
    assert!(matches!(second_step, TurnLoopStep::Continue));
    assert!(app.cli.thinking_disabled_override);
    assert!(force_final_response);
    assert!(final_assistant_text.is_empty());
    let synthesis_markers = messages
        .iter()
        .filter(|message| {
            message.role == ROLE_INTERNAL_NOTE
                && message
                    .content
                    .as_str()
                    .is_some_and(|text| text.starts_with(REASONING_ONLY_SYNTHESIS_MARKER))
        })
        .count();
    assert_eq!(synthesis_markers, 1);
    let synthesis_retry_markers = messages
        .iter()
        .filter(|message| {
            message.role == ROLE_INTERNAL_NOTE
                && message
                    .content
                    .as_str()
                    .is_some_and(|text| text.starts_with(REASONING_ONLY_SYNTHESIS_RETRY_MARKER))
        })
        .count();
    assert_eq!(synthesis_retry_markers, 1);
}

#[test]
fn reasoning_only_final_response_stops_after_bounded_post_synthesis_retries() {
    // After the forced no-reasoning synthesis the model still returns reasoning-only:
    // only a limited number of retries with fresh markers is allowed
    // (REASONING_ONLY_POST_SYNTHESIS_MAX_RETRIES); past that, stop the round with a
    // user-visible error — avoiding empty spins on identical byte-for-byte requests
    // up to max_iterations.
    let mut app = test_app_with_tools(&["read_file"]);
    let mcp = crate::ai::mcp::McpClient::new();
    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(mcp));
    let mut messages = vec![Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: serde_json::Value::String(format!(
            "{REASONING_ONLY_SYNTHESIS_MARKER}\n{REASONING_ONLY_SYNTHESIS_NOTE}"
        )),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let mut turn_messages = Vec::new();
    let mut persisted_turn_messages = 0usize;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut force_final_response = true;
    let mut terminal_dedupe_candidate = None;

    let stream_result = || {
        IterationExecution::FinalResponse(crate::ai::types::StreamResult {
            outcome: crate::ai::types::StreamOutcome::Completed,
            tool_calls: Vec::new(),
            assistant_text: String::new(),
            hidden_meta: String::new(),
            reasoning_text: "Still hidden reasoning".to_string(),
            reasoning_items: Vec::new(),
            skip_response_drain: true,
            truncated_by_length: false,
            stream_error: false,
            finish_reason_value: None,
            usage_prompt_tokens: 0,
            usage_cached_prompt_tokens: 0,
            usage_completion_tokens: 0,
            usage_reasoning_tokens: 0,
        })
    };
    fn synthesis_retry_markers(messages: &[Message]) -> usize {
        messages
            .iter()
            .filter(|message| {
                message.role == ROLE_INTERNAL_NOTE
                    && message.content.as_str().is_some_and(|text| {
                        text.starts_with(REASONING_ONLY_SYNTHESIS_RETRY_MARKER)
                    })
            })
            .count()
    }

    // First hit (no synthesis-retry marker yet): inject a new marker and continue.
    let step = handle_iteration_execution(
        &mut app,
        "compare two yaml files",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        stream_result(),
        &mut messages,
        &mut turn_messages,
        false,
        &mut persisted_turn_messages,
        &mut final_assistant_text,
        &mut final_assistant_recorded,
        &mut force_final_response,
        &mut terminal_dedupe_candidate,
        true,
        3,
        16,
        0,
        &mut false,
    )
    .unwrap();
    assert!(matches!(step, TurnLoopStep::Continue));
    assert!(final_assistant_text.is_empty());
    assert_eq!(synthesis_retry_markers(&messages), 1);

    // Second hit: inject a second marker and continue.
    let second_step = handle_iteration_execution(
        &mut app,
        "compare two yaml files",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        stream_result(),
        &mut messages,
        &mut turn_messages,
        false,
        &mut persisted_turn_messages,
        &mut final_assistant_text,
        &mut final_assistant_recorded,
        &mut force_final_response,
        &mut terminal_dedupe_candidate,
        true,
        4,
        16,
        0,
        &mut false,
    )
    .unwrap();
    assert!(matches!(second_step, TurnLoopStep::Continue));
    assert!(final_assistant_text.is_empty());
    assert_eq!(synthesis_retry_markers(&messages), 2);

    // Third hit: the cap is reached; stop the round with a user-visible error.
    let last_step = handle_iteration_execution(
        &mut app,
        "compare two yaml files",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        stream_result(),
        &mut messages,
        &mut turn_messages,
        false,
        &mut persisted_turn_messages,
        &mut final_assistant_text,
        &mut final_assistant_recorded,
        &mut force_final_response,
        &mut terminal_dedupe_candidate,
        true,
        5,
        16,
        0,
        &mut false,
    )
    .unwrap();
    assert!(matches!(last_step, TurnLoopStep::Break));
    assert_eq!(
        final_assistant_text,
        "[Model returned only reasoning content without a final answer; please retry or switch models]"
    );
}

#[test]
fn reasoning_only_final_response_max_iterations_is_final_backstop() {
    // The iteration hard cap remains the final fallback: even if the post-synthesis
    // retries have not hit their cap, reaching max_iterations also stops the round
    // with a user-visible error.
    let mut app = test_app_with_tools(&["read_file"]);
    let mcp = crate::ai::mcp::McpClient::new();
    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(mcp));
    let mut messages = vec![Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: serde_json::Value::String(format!(
            "{REASONING_ONLY_SYNTHESIS_MARKER}\n{REASONING_ONLY_SYNTHESIS_NOTE}"
        )),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let mut turn_messages = Vec::new();
    let mut persisted_turn_messages = 0usize;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut force_final_response = true;
    let mut terminal_dedupe_candidate = None;

    let stream_result = || {
        IterationExecution::FinalResponse(crate::ai::types::StreamResult {
            outcome: crate::ai::types::StreamOutcome::Completed,
            tool_calls: Vec::new(),
            assistant_text: String::new(),
            hidden_meta: String::new(),
            reasoning_text: "Still hidden reasoning".to_string(),
            reasoning_items: Vec::new(),
            skip_response_drain: true,
            truncated_by_length: false,
            stream_error: false,
            finish_reason_value: None,
            usage_prompt_tokens: 0,
            usage_cached_prompt_tokens: 0,
            usage_completion_tokens: 0,
            usage_reasoning_tokens: 0,
        })
    };

    // Post-synthesis retries have not hit their cap, but max_iterations was reached:
    // stop the round with a user-visible error.
    let last_step = handle_iteration_execution(
        &mut app,
        "compare two yaml files",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        stream_result(),
        &mut messages,
        &mut turn_messages,
        false,
        &mut persisted_turn_messages,
        &mut final_assistant_text,
        &mut final_assistant_recorded,
        &mut force_final_response,
        &mut terminal_dedupe_candidate,
        true,
        16,
        16,
        0,
        &mut false,
    )
    .unwrap();
    assert!(matches!(last_step, TurnLoopStep::Break));
    assert_eq!(
        final_assistant_text,
        "[Model returned only reasoning content without a final answer; please retry or switch models]"
    );
}

#[test]
fn reasoning_only_final_response_retries_up_to_max_before_forcing_synthesis() {
    let mut app = test_app_with_tools(&["read_file"]);
    let mcp = crate::ai::mcp::McpClient::new();
    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(mcp));
    // With MAX-1 ordinary retries already used, another hit should still continue
    // ordinary retries rather than entering synthesis early.
    let mut messages: Vec<Message> = (0..REASONING_ONLY_MAX_RETRIES - 1)
        .map(|_| Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: serde_json::Value::String(format!(
                "{REASONING_ONLY_RETRY_MARKER}\n{REASONING_ONLY_RETRY_NOTE}"
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        })
        .collect();
    let mut turn_messages = Vec::new();
    let mut persisted_turn_messages = 0usize;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut force_final_response = false;
    let mut terminal_dedupe_candidate = None;

    let stream_result = |reasoning: &str| {
        IterationExecution::FinalResponse(crate::ai::types::StreamResult {
            outcome: crate::ai::types::StreamOutcome::Completed,
            tool_calls: Vec::new(),
            assistant_text: String::new(),
            hidden_meta: String::new(),
            reasoning_text: reasoning.to_string(),
            reasoning_items: Vec::new(),
            skip_response_drain: true,
            truncated_by_length: false,
            stream_error: false,
            finish_reason_value: None,
            usage_prompt_tokens: 0,
            usage_cached_prompt_tokens: 0,
            usage_completion_tokens: 0,
            usage_reasoning_tokens: 0,
        })
    };

    let step = handle_iteration_execution(
        &mut app,
        "compare two yaml files",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        stream_result("Still hidden reasoning"),
        &mut messages,
        &mut turn_messages,
        false,
        &mut persisted_turn_messages,
        &mut final_assistant_text,
        &mut final_assistant_recorded,
        &mut force_final_response,
        &mut terminal_dedupe_candidate,
        true,
        2,
        16,
        0,
        &mut false,
    )
    .unwrap();

    assert!(matches!(step, TurnLoopStep::Continue));
    assert!(!force_final_response);
    assert!(!app.cli.thinking_disabled_override);
    let retry_markers = messages
        .iter()
        .filter(|message| {
            message.role == ROLE_INTERNAL_NOTE
                && message
                    .content
                    .as_str()
                    .is_some_and(|text| text.starts_with(REASONING_ONLY_RETRY_MARKER))
        })
        .count();
    assert_eq!(retry_markers, REASONING_ONLY_MAX_RETRIES);
    assert!(messages.iter().all(|message| {
        message.role != ROLE_INTERNAL_NOTE
            || !message
                .content
                .as_str()
                .is_some_and(|text| text.starts_with(REASONING_ONLY_SYNTHESIS_MARKER))
    }));

    // After reaching the cap, the next hit enters the no-reasoning synthesis.
    let second_step = handle_iteration_execution(
        &mut app,
        "compare two yaml files",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        stream_result("Still hidden reasoning again"),
        &mut messages,
        &mut turn_messages,
        false,
        &mut persisted_turn_messages,
        &mut final_assistant_text,
        &mut final_assistant_recorded,
        &mut force_final_response,
        &mut terminal_dedupe_candidate,
        true,
        3,
        16,
        0,
        &mut false,
    )
    .unwrap();

    assert!(matches!(second_step, TurnLoopStep::Continue));
    assert!(app.cli.thinking_disabled_override);
    assert!(force_final_response);
    assert!(messages.iter().any(|message| {
        message.role == ROLE_INTERNAL_NOTE
            && message
                .content
                .as_str()
                .is_some_and(|text| text.starts_with(REASONING_ONLY_SYNTHESIS_MARKER))
    }));
}

#[test]
fn forced_final_hallucinated_tool_call_is_rejected_without_consuming_quota() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_tools(&["read_file"]);
    let pid = {
        let mut os = app.os.lock().unwrap();
        let pid =
            os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        let mut lim = ResourceLimit::unlimited();
        lim.max_tool_calls = 64;
        os.rlimit_set(pid, lim).unwrap();
        pid
    };
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());

    let path = std::env::temp_dir().join(format!("forced-final-{}.txt", pid));
    std::fs::write(&path, "hello").unwrap();

    let shared_mcp =
        std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
    let mut messages = Vec::new();
    let mut turn_messages = Vec::new();
    let mut persisted_turn_messages = 0usize;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut force_final_response = true;
    let mut terminal_dedupe_candidate = None;

    let step = handle_iteration_execution(
        &mut app,
        "summarize findings",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        IterationExecution::ToolCall(ToolCallExecution {
            stream_result: crate::ai::types::StreamResult {
                outcome: crate::ai::types::StreamOutcome::ToolCall,
                tool_calls: vec![ToolCall {
                    id: "call_1".to_string(),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: "read_file".to_string(),
                        arguments: format!(r#"{{"file_path":"{}"}}"#, path.to_string_lossy()),
                    },
                }],
                assistant_text: String::new(),
                hidden_meta: String::new(),
                reasoning_text: String::new(),
                reasoning_items: Vec::new(),
                skip_response_drain: true,
                truncated_by_length: false,
                stream_error: false,
                finish_reason_value: None,
                usage_prompt_tokens: 0,
                usage_cached_prompt_tokens: 0,
                usage_completion_tokens: 0,
                usage_reasoning_tokens: 0,
            },
            allowed_tool_names: ["read_file".to_string()].into_iter().collect(),
        }),
        &mut messages,
        &mut turn_messages,
        false,
        &mut persisted_turn_messages,
        &mut final_assistant_text,
        &mut final_assistant_recorded,
        &mut force_final_response,
        &mut terminal_dedupe_candidate,
        true,
        3,
        16,
        0,
        &mut false,
    )
    .unwrap();

    assert!(matches!(step, TurnLoopStep::Continue));
    assert!(force_final_response);
    assert!(final_assistant_text.is_empty());
    assert!(!final_assistant_recorded);
    {
        let os = app.os.lock().unwrap();
        assert_eq!(os.rusage_get(pid).unwrap().tool_calls, 0);
    }
    let joined = turn_messages
        .iter()
        .map(|msg| msg.content.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(joined.contains("disabled in no-tool handoff mode"));
    assert!(!joined.contains("exceeded kernel rlimit"));
    // The no-tool synthesis retry marker is model-visible context injected into the
    // request projection (`messages`), but is deliberately kept out of canonical
    // `turn_messages`: the retry budget is owned in-memory by `FinalGateState`, so
    // persisting the marker would only risk a stale note surviving into a later turn.
    let request_joined = messages
        .iter()
        .map(|msg| msg.content.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(request_joined.contains(NO_TOOL_SYNTHESIS_RETRY_MARKER));
    assert!(!joined.contains(NO_TOOL_SYNTHESIS_RETRY_MARKER));

    let step = handle_iteration_execution(
        &mut app,
        "summarize findings",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        IterationExecution::ToolCall(ToolCallExecution {
            stream_result: crate::ai::types::StreamResult {
                outcome: crate::ai::types::StreamOutcome::ToolCall,
                tool_calls: vec![ToolCall {
                    id: "call_2".to_string(),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: "read_file".to_string(),
                        arguments: format!(r#"{{"file_path":"{}"}}"#, path.to_string_lossy()),
                    },
                }],
                assistant_text: "I still need one more read.".to_string(),
                hidden_meta: String::new(),
                reasoning_text: String::new(),
                reasoning_items: Vec::new(),
                skip_response_drain: true,
                truncated_by_length: false,
                stream_error: false,
                finish_reason_value: None,
                usage_prompt_tokens: 0,
                usage_cached_prompt_tokens: 0,
                usage_completion_tokens: 0,
                usage_reasoning_tokens: 0,
            },
            allowed_tool_names: ["read_file".to_string()].into_iter().collect(),
        }),
        &mut messages,
        &mut turn_messages,
        false,
        &mut persisted_turn_messages,
        &mut final_assistant_text,
        &mut final_assistant_recorded,
        &mut force_final_response,
        &mut terminal_dedupe_candidate,
        true,
        4,
        16,
        0,
        &mut false,
    )
    .unwrap();

    assert!(matches!(step, TurnLoopStep::Break));
    assert!(final_assistant_text.contains("I still need one more read."));
    assert!(final_assistant_text.contains(NO_TOOL_SYNTHESIS_WARNING));
    {
        let os = app.os.lock().unwrap();
        assert_eq!(os.rusage_get(pid).unwrap().tool_calls, 0);
    }

    let _ = std::fs::remove_file(&path);
    if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[test]
fn unsupported_read_only_phase_limit_claim_reopens_once_with_tools() {
    let turn_messages = vec![Message {
        role: "tool".to_string(),
        content: Value::String("read completed".to_string()),
        tool_calls: None,
        tool_call_id: Some("call-1".to_string()),
        reasoning_content: None,
    }];
    let mut messages = turn_messages.clone();
    let final_text = "本轮执行环境在代码修改前触发了只读阶段上限，尚未写入文件。";

    assert_eq!(
        unsupported_runtime_limit_action(
            "继续修复吧",
            &mut messages,
            &turn_messages,
            final_text,
            false,
            false,
            2,
            16,
        ),
        UnsupportedRuntimeLimitAction::ReopenWithTools
    );
    assert_eq!(
        messages
            .iter()
            .filter(|message| {
                message.content.as_str().is_some_and(|text| {
                    text.starts_with(UNSUPPORTED_RUNTIME_LIMIT_RETRY_MARKER)
                })
            })
            .count(),
        1
    );
    assert_eq!(
        unsupported_runtime_limit_action(
            "继续修复吧",
            &mut messages,
            &turn_messages,
            final_text,
            false,
            false,
            3,
            16,
        ),
        UnsupportedRuntimeLimitAction::Warn
    );

    let supported_turn = vec![Message {
        role: "tool".to_string(),
        content: Value::String("Error: 触发了只读阶段上限".to_string()),
        tool_calls: None,
        tool_call_id: Some("call-2".to_string()),
        reasoning_content: None,
    }];
    let mut untrusted_messages = supported_turn.clone();
    assert_eq!(
        unsupported_runtime_limit_action(
            "继续修复吧",
            &mut untrusted_messages,
            &supported_turn,
            final_text,
            false,
            false,
            2,
            16,
        ),
        UnsupportedRuntimeLimitAction::ReopenWithTools,
        "tool text alone is not trusted as runtime failure evidence"
    );

    let mut supported_messages = supported_turn.clone();
    assert_eq!(
        unsupported_runtime_limit_action(
            "继续修复吧",
            &mut supported_messages,
            &supported_turn,
            final_text,
            true,
            false,
            2,
            16,
        ),
        UnsupportedRuntimeLimitAction::Allow,
        "observed tool evidence must preserve legitimate failure reporting"
    );

    let mut plan_messages = turn_messages.clone();
    assert_eq!(
        unsupported_runtime_limit_action(
            "Give me a plan for fixing this",
            &mut plan_messages,
            &turn_messages,
            final_text,
            false,
            false,
            2,
            16,
        ),
        UnsupportedRuntimeLimitAction::Allow,
        "a plan-only request must never be upgraded into mutation work"
    );
}

#[test]
fn dangling_action_final_gets_exactly_one_no_tool_recovery() {
    let turn_messages = vec![Message {
        role: "tool".to_string(),
        content: Value::String("existing scheduler evidence".to_string()),
        tool_calls: None,
        tool_call_id: Some("call-1".to_string()),
        reasoning_content: None,
    }];
    let mut messages = turn_messages.clone();
    let final_text = "Now I understand the SchedulerClock::wait mechanism. Let me read the full run loop body to see how it uses next_wakeup_tick and advance_ticks";

    assert_eq!(
        dangling_final_recovery_action(
            "Audit the scheduler changes",
            &mut messages,
            &turn_messages,
            final_text,
        ),
        DanglingFinalRecoveryAction::RetryWithoutTools
    );
    assert_eq!(
        messages
            .iter()
            .filter(|message| {
                message
                    .content
                    .as_str()
                    .is_some_and(|text| text.starts_with(DANGLING_FINAL_RECOVERY_MARKER))
            })
            .count(),
        1
    );
    assert_eq!(
        dangling_final_recovery_action(
            "Audit the scheduler changes",
            &mut messages,
            &turn_messages,
            final_text,
        ),
        DanglingFinalRecoveryAction::Warn
    );
}

#[test]
fn dangling_action_detection_preserves_normal_finals_and_plan_answers() {
    let turn_messages = vec![Message {
        role: "tool".to_string(),
        content: Value::String("evidence".to_string()),
        tool_calls: None,
        tool_call_id: Some("call-1".to_string()),
        reasoning_content: None,
    }];

    assert!(!looks_like_dangling_action_final(
        "Audit the scheduler changes",
        &turn_messages,
        "Conclusion: the scheduler wake path is covered. Let me explain the remaining risk.",
    ));
    assert!(!looks_like_dangling_action_final(
        "Give me a plan for auditing the scheduler",
        &turn_messages,
        "Next steps: let me inspect the run loop, then check the kernel wake path.",
    ));
    assert!(!looks_like_dangling_action_final(
        "Audit the scheduler changes",
        &[],
        "Let me inspect the run loop first.",
    ));
    assert!(looks_like_dangling_action_final(
        "Audit the scheduler changes",
        &turn_messages,
        "Now I understand the flow. Let me inspect the final dispatch branch.\n\n[Runtime warning] Completion claim is unverified.",
    ));
    assert!(looks_like_dangling_action_final(
        "Don't give me next steps; audit the scheduler changes",
        &turn_messages,
        "Let me inspect the final dispatch branch.",
    ));
    assert!(looks_like_dangling_action_final(
        "Execute the existing next steps and report findings",
        &turn_messages,
        "Let me inspect the final dispatch branch.",
    ));
    assert!(looks_like_dangling_action_final(
        "The phrase \"give me a plan\" is an example; audit the scheduler changes",
        &turn_messages,
        "Let me inspect the final dispatch branch.",
    ));
    assert!(looks_like_dangling_action_final(
        "Audit the scheduler changes",
        &turn_messages,
        "[Runtime warning] Completion claim is unverified.",
    ));
    assert!(!looks_like_dangling_action_final(
        "Audit the scheduler changes",
        &turn_messages,
        "[Runtime warning] Completion claim is unverified.\n\nConclusion: no drift was found.",
    ));

    let mut warning_only_messages = turn_messages.clone();
    assert_eq!(
        dangling_final_recovery_action(
            "Audit the scheduler changes",
            &mut warning_only_messages,
            &turn_messages,
            "[Runtime warning] Completion claim is unverified.",
        ),
        DanglingFinalRecoveryAction::RetryWithoutTools
    );

    let mut warning_text = DANGLING_FINAL_WARNING.to_string();
    append_runtime_warning_once(&mut warning_text, DANGLING_FINAL_WARNING);
    assert_eq!(warning_text.matches(DANGLING_FINAL_WARNING).count(), 1);
}

#[test]
fn prose_sentence_counter_ignores_code_symbol_dots() {
    // Dots inside code symbols must not count as sentence endings: in
    // `driver/mod.rs`, `.ok().flatten()`, and line ranges like `1057-1080`, the `.`
    // is never followed by whitespace or end-of-text.
    assert_eq!(
        prose_sentence_terminator_count(
            "检查 driver/mod.rs:1057-1080 的 .ok().flatten() 吞错逻辑"
        ),
        0
    );
    // Genuine sentence endings (. followed by whitespace, or the CJK
    // full-stop/exclamation/question marks) still count.
    assert_eq!(
        prose_sentence_terminator_count("First done. Second done! Third?"),
        3
    );
    assert_eq!(prose_sentence_terminator_count("第一。第二！第三？"), 3);
    // A trailing . also counts as a sentence ending (followed by the end of the text).
    assert_eq!(prose_sentence_terminator_count("Done."), 1);
}

#[test]
fn strip_inline_code_spans_removes_paired_backticks_only() {
    assert_eq!(
        strip_inline_code_spans("检查 `driver/mod.rs` 的 `.ok()` 逻辑"),
        "检查  的  逻辑"
    );
    // When backticks are unpaired (odd count), return the text unchanged to avoid
    // deleting the tail of the prose.
    assert_eq!(
        strip_inline_code_spans("half `open span"),
        "half `open span"
    );
}

#[test]
fn dangling_final_detects_mid_introduction_colon_stop() {
    // Real regression: session b884d15f message id=455. At the end of a long tool
    // chain the model stopped on the aside "first look at... check...:" — a
    // colon-terminated promise of a tool call with no tool call — which previously
    // slipped through both the stream classifier (judged Completed) and the dangling
    // gate (code symbols polluting the sentence count + wording not in the word list),
    // being silently accepted as a final response and forcing the user to nudge it.
    let turn_messages = vec![Message {
        role: "tool".to_string(),
        content: Value::String("git status output".to_string()),
        tool_calls: None,
        tool_call_id: Some("call-1".to_string()),
        reasoning_content: None,
    }];
    let final_text = "11 个文件与 review.md 声称一致。现在逐项检查 review.md 列出的问题。先看 P1-a（图片解析失败静默丢失）——检查 `driver/mod.rs:1057-1080` 的 `.ok().flatten()` 吞错逻辑：";
    assert!(
        looks_like_dangling_action_final(
            "分析这个 agent 的会话历史",
            &turn_messages,
            final_text,
        ),
        "以冒号收尾、代码符号密集的悬空预告必须被识别为 dangling final"
    );
}

#[test]
fn dangling_final_colon_signal_respects_conclusion_and_structure_guards() {
    let turn_messages = vec![Message {
        role: "tool".to_string(),
        content: Value::String("evidence".to_string()),
        tool_calls: None,
        tool_call_id: Some("call-1".to_string()),
        reasoning_content: None,
    }];

    // Colon-terminated but a conclusion was delivered: the conclusion marker takes
    // priority, not dangling.
    assert!(!looks_like_dangling_action_final(
        "审查这段代码",
        &turn_messages,
        "结论：run loop 的 wake 路径已覆盖，没有缺陷。补充说明如下：",
    ));
    // Colon-terminated but followed by a delivered list: the structured_lines guard
    // runs first, not dangling.
    assert!(!looks_like_dangling_action_final(
        "审查这段代码",
        &turn_messages,
        "发现两个问题：\n- 第一个问题\n- 第二个问题",
    ));
    // Body ending with a code span (last char is a backtick, not a colon) = content
    // delivered; no misjudgment.
    assert!(!looks_like_dangling_action_final(
        "审查这段代码",
        &turn_messages,
        "修复点在 `foo.rs` 的 `bar()`",
    ));
    // A bare colon-terminated teaser with nothing after = dangling.
    assert!(looks_like_dangling_action_final(
        "审查这段代码",
        &turn_messages,
        "现在开始逐项核对第一处改动：",
    ));
}

#[test]
fn injected_context_echo_is_detected_only_when_it_is_the_whole_answer() {
    // Real regression: session 7ac3d771 message id=263. The model regurgitated the
    // completion-evidence reopen hint + self_note header verbatim as its answer,
    // leaking to the terminal and persisting as final.
    let echoed = "[Model-authored note from an earlier turn; this is not authoritative evidence. Treat every claim as unverified unless it is backed by tool output or a cited source, and re-check it before using it as a conclusion.]\nself_note:completion_evidence_required\nA successful project mutation occurred in the current user turn, but no successful post-mutation verification was observed.";
    assert!(looks_like_injected_context_echo(echoed));

    // The [Runtime warning] section appended post-hoc does not affect the verdict —
    // only the model's body is considered.
    let echoed_with_warning = format!(
        "{echoed}\n\n[Runtime warning] Completion/impact claim is unverified: no successful post-mutation check was observed."
    );
    assert!(looks_like_injected_context_echo(&echoed_with_warning));

    // Bare self_note: prefix.
    assert!(looks_like_injected_context_echo(
        "self_note:completion_evidence_required\ninspect the diff first."
    ));
    // History-summary header / handoff header.
    assert!(looks_like_injected_context_echo(
        "[Compressed history summary for task continuity. Use it to ...]\nearlier work"
    ));
    assert!(looks_like_injected_context_echo(
        "[Runtime context handoff, not a new end-user request. ...]"
    ));
    // A real answer: even quoting these prefixes, as long as they are not at the
    // start, it is not an echo.
    assert!(!looks_like_injected_context_echo(
        "修复完成。运行时会注入形如 self_note: 的提示，但那是内部上下文。"
    ));
    assert!(!looks_like_injected_context_echo(
        "P2-a 已修完，62 个 fold 测试全绿。"
    ));
    // Pure [Runtime warning] (no model body) is handled by the other gates; not an
    // echo.
    assert!(!looks_like_injected_context_echo(
        "\n\n[Runtime warning] Completion/impact claim is unverified."
    ));
}

#[test]
fn injected_context_echo_gets_exactly_one_no_tool_recovery_then_stops() {
    let echoed = "[Model-authored note from an earlier turn; this is not authoritative evidence.]\nself_note:completion_evidence_required\nThis is not a final answer.";
    let mut messages: Vec<Message> = Vec::new();

    // First hit: inject one no-tool retry hint.
    assert_eq!(
        injected_context_echo_recovery_action(&mut messages, echoed),
        DanglingFinalRecoveryAction::RetryWithoutTools
    );
    assert_eq!(
        messages
            .iter()
            .filter(|message| {
                message
                    .content
                    .as_str()
                    .is_some_and(|text| text.starts_with(INJECTED_CONTEXT_ECHO_RETRY_MARKER))
            })
            .count(),
        1
    );
    // Second time still regurgitating: stop the round (Warn), no infinite retries.
    assert_eq!(
        injected_context_echo_recovery_action(&mut messages, echoed),
        DanglingFinalRecoveryAction::Warn
    );
    // A normal answer passes.
    assert_eq!(
        injected_context_echo_recovery_action(&mut messages, "修复完成，测试全绿。"),
        DanglingFinalRecoveryAction::Allow
    );
}
