//! Tests for the `followup` cluster.

use super::super::*;
use super::common::*;

#[test]
fn runtime_synthetic_user_unintegrated_task_evidence_keeps_provenance() {
    let mut messages = vec![Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(no_tool_handoff_note().to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    reopen_turn_for_unintegrated_task_evidence(
        &mut messages,
        "[task-evidence-ledger]\ntask_id=task-1",
    );

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "user");
    assert!(is_runtime_synthetic_user_message(&messages[0]));
    assert_eq!(messages[1].role, "assistant");
    assert!(
        messages[1]
            .content
            .as_str()
            .unwrap()
            .contains("task_id=task-1")
    );
}

#[test]
fn extract_image_paths_from_file_read_tool_calls_collects_image_reads() {
    let tool_calls = vec![
        ToolCall {
            id: "call_1".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: r#"{"file_path":"/tmp/shot.png"}"#.to_string(),
            },
        },
        ToolCall {
            id: "call_2".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "read_file".to_string(),
                arguments: r#"{"file_path":"/tmp/notes.txt"}"#.to_string(),
            },
        },
    ];
    assert_eq!(
        extract_image_paths_from_file_read_tool_calls(&tool_calls),
        vec!["/tmp/shot.png".to_string()]
    );
}

#[test]
fn final_response_reopens_until_delivered_task_is_integrated() {
    let root = std::env::temp_dir().join(format!(
        "task-evidence-final-gate-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let history_file = root.join("history.sqlite");
    let session_id = format!("task-evidence-{}", uuid::Uuid::new_v4().simple());
    let mut app = test_app_with_tools(&["task_integrate"]);
    app.config.history_file = history_file.clone();
    app.session_id = session_id.clone();
    crate::ai::history::record_delivered_task_evidence(
        &history_file,
        &session_id,
        crate::ai::history::DeliveredTaskEvidence {
            task_id: "task-1",
            description: "review parser",
            agent_name: "build",
            model: "test-model",
            status: "completed",
            payload: "[Subagent final answer]\nconfirmed conclusion",
        },
    )
    .unwrap();

    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
    let mut messages = Vec::new();
    let mut turn_messages = Vec::new();
    let mut persisted_turn_messages = 0usize;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut force_final_response = false;
    let mut terminal_dedupe_candidate = None;
    let mut turn_had_tool_error = false;
    let final_response = || {
        IterationExecution::FinalResponse(crate::ai::types::StreamResult {
            outcome: crate::ai::types::StreamOutcome::Completed,
            assistant_text: "done".to_string(),
            skip_response_drain: true,
            ..Default::default()
        })
    };

    let first = handle_iteration_execution(
        &mut app,
        "finish",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        final_response(),
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
        &mut turn_had_tool_error,
    )
    .unwrap();
    assert!(matches!(first, TurnLoopStep::Continue));
    assert!(messages.iter().any(|message| {
        message
            .content
            .as_str()
            .is_some_and(|text| text.starts_with(UNINTEGRATED_TASK_EVIDENCE_PREFIX))
            && crate::ai::history::is_runtime_synthetic_user_message(message)
    }));

    assert!(
        crate::ai::history::integrate_task_evidence(
            &history_file,
            &session_id,
            "task-1",
            "accepted",
            "used confirmed conclusion"
        )
        .unwrap()
    );
    let second = handle_iteration_execution(
        &mut app,
        "finish",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        final_response(),
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
        &mut turn_had_tool_error,
    )
    .unwrap();
    assert!(matches!(second, TurnLoopStep::Break));
    assert_eq!(final_assistant_text, "done");

    let sessions_root = crate::ai::history::SessionStore::new(&history_file)
        .sessions_root()
        .to_path_buf();
    let _ = std::fs::remove_dir_all(sessions_root);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn final_response_with_outstanding_subagent_task_reopens_turn_and_clears_no_tool_handoff() {
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_tools(&["task_wait", "task_status"]);
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (pid, result_channel_id) = {
        let mut os = app.os.lock().unwrap();
        let pid = os.begin_foreground(
            "child".to_string(),
            "goal".to_string(),
            10,
            usize::MAX,
            None,
        );
        let channel = os.channel_create(Some(pid), 1, "task-result".to_string());
        (pid, channel.raw())
    };
    crate::ai::tools::task_tools::insert_task_entry_for_test(
        task_id.clone(),
        crate::ai::tools::task_tools::AsyncTaskEntry {
            session_id: app.session_id.clone(),
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            result_observed: false,
            owner_pid: pid,
            pid,
            result_channel_id,
            completion_futex_addr: aios_kernel::primitives::FutexAddr(1),
            description: "inspect parser".to_string(),
            agent_name: "build".to_string(),
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: crate::ai::tools::task_tools::InheritOptions::default(),
            abort_handle: None,
            cancel_stream: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
        },
    );

    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
    let mut messages = vec![Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(no_tool_handoff_note().to_string()),
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
        "wrap up",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        IterationExecution::FinalResponse(crate::ai::types::StreamResult {
            outcome: crate::ai::types::StreamOutcome::Completed,
            tool_calls: Vec::new(),
            assistant_text: "done".to_string(),
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
    assert!(!force_final_response);
    assert!(final_assistant_text.is_empty());
    assert!(!final_assistant_recorded);
    assert!(turn_messages.is_empty());
    let joined = messages
        .iter()
        .map(|message| message.content.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(joined.contains(PENDING_SUBAGENT_TASKS_FOLLOWUP_PREFIX.trim_end()));
    assert!(joined.contains(&task_id));
    assert!(joined.contains("Immediate next step: call `task_wait` or `task_status`"));
    assert!(!joined.contains(no_tool_handoff_note()));

    let _ = crate::ai::tools::task_tools::remove_task_entry(&task_id);
    if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[test]
fn final_response_at_iteration_ceiling_finishes_despite_outstanding_task() {
    // The iteration hard cap is the authoritative ceiling: even with unclosed
    // subagent tasks remaining, finalization cannot be bounced indefinitely
    // (otherwise it would livelock when a subtask never reaches a terminal state and
    // repeatedly knock out the safety brakes).
    let _env_guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut app = test_app_with_tools(&["task_wait", "task_status"]);
    app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
    crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());

    let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
    let (pid, result_channel_id) = {
        let mut os = app.os.lock().unwrap();
        let pid = os.begin_foreground(
            "child".to_string(),
            "goal".to_string(),
            10,
            usize::MAX,
            None,
        );
        let channel = os.channel_create(Some(pid), 1, "task-result".to_string());
        (pid, channel.raw())
    };
    crate::ai::tools::task_tools::insert_task_entry_for_test(
        task_id.clone(),
        crate::ai::tools::task_tools::AsyncTaskEntry {
            session_id: app.session_id.clone(),
            last_progress_notification_at: None,
            last_progress_persisted_at: None,
            result_observed: false,
            owner_pid: pid,
            pid,
            result_channel_id,
            completion_futex_addr: aios_kernel::primitives::FutexAddr(1),
            description: "inspect parser".to_string(),
            agent_name: "build".to_string(),
            model: "qwen3.7-max".to_string(),
            is_model_auto_selected: false,
            auto_model_fallback: None,
            selection_explanation: "explicit override".to_string(),
            inherit: crate::ai::tools::task_tools::InheritOptions::default(),
            abort_handle: None,
            cancel_stream: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
        },
    );

    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
    let mut messages = Vec::new();
    let mut turn_messages = Vec::new();
    let mut persisted_turn_messages = 0usize;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut force_final_response = true;
    let mut terminal_dedupe_candidate = None;

    let max_iterations = 16;
    let step = handle_iteration_execution(
        &mut app,
        "wrap up",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        IterationExecution::FinalResponse(crate::ai::types::StreamResult {
            outcome: crate::ai::types::StreamOutcome::Completed,
            tool_calls: Vec::new(),
            assistant_text: "done".to_string(),
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
        max_iterations,
        max_iterations,
        0,
        &mut false,
    )
    .unwrap();

    // Hard cap reached: no more bounces; allow finalization.
    assert!(matches!(step, TurnLoopStep::Break));
    assert!(final_assistant_text.starts_with("done\n\n"));
    assert!(final_assistant_text.contains("1 spawned subagent task(s) were still outstanding"));
    assert!(final_assistant_text.contains(&task_id));
    assert!(final_assistant_text.contains("Required follow-up: re-run this turn"));
    let joined = messages
        .iter()
        .map(|message| message.content.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!joined.contains(PENDING_SUBAGENT_TASKS_FOLLOWUP_PREFIX.trim_end()));

    let _ = crate::ai::tools::task_tools::remove_task_entry(&task_id);
    if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
        *guard = None;
    }
}

#[test]
fn truncated_response_retries_and_injects_shrink_note() {
    let mut app = test_app_with_tools(&["write_file"]);
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
        "write a big script",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        IterationExecution::Truncated(crate::ai::types::StreamResult {
            outcome: crate::ai::types::StreamOutcome::Truncated,
            tool_calls: Vec::new(),
            assistant_text: "现在让我来编写一个综合脚本".to_string(),
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
        1,
        &mut false,
    )
    .unwrap();

    // Truncation should auto-retry (Continue), never complete silently.
    assert!(matches!(step, TurnLoopStep::Continue));
    assert!(final_assistant_text.is_empty());
    assert!(!final_assistant_recorded);
    // Partial visible text is preserved as assistant context.
    assert!(
        messages
            .iter()
            .any(|m| m.role == "assistant"
                && m.content.as_str() == Some("现在让我来编写一个综合脚本"))
    );
    // Partial text must not be written to the persisted turn_messages track — with
    // consecutive truncations, multiple large half-finished texts would pollute the
    // history file and cause the next turn's normal history to be compressed away.
    assert!(
        !turn_messages
            .iter()
            .any(|m| m.role == "assistant"
                && m.content.as_str() == Some("现在让我来编写一个综合脚本")),
        "partial text must not leak into turn_messages (persistence track)"
    );
    // A shrink-and-rewrite hint was injected.
    assert!(messages.iter().any(|m| {
        m.role == ROLE_INTERNAL_NOTE
            && m.content
                .as_str()
                .is_some_and(|c| c.starts_with(TRUNCATION_RETRY_NOTE_PREFIX))
    }));
}

#[test]
fn truncation_retry_note_replaces_with_updated_count() {
    let mut app = test_app_with_tools(&["write_file"]);
    let mcp = crate::ai::mcp::McpClient::new();
    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(mcp));
    let mut messages = Vec::new();
    let mut turn_messages = Vec::new();
    let mut persisted_turn_messages = 0usize;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut force_final_response = false;
    let mut terminal_dedupe_candidate = None;

    for consecutive in 1..=2 {
        handle_iteration_execution(
            &mut app,
            "write a big script",
            &mcp_snapshot(&shared_mcp),
            &shared_mcp,
            IterationExecution::Truncated(crate::ai::types::StreamResult {
                outcome: crate::ai::types::StreamOutcome::Truncated,
                tool_calls: Vec::new(),
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
            consecutive,
            &mut false,
        )
        .unwrap();
    }

    let note_count = messages
        .iter()
        .filter(|m| {
            m.role == ROLE_INTERNAL_NOTE
                && m.content
                    .as_str()
                    .is_some_and(|c| c.starts_with(TRUNCATION_RETRY_NOTE_PREFIX))
        })
        .count();
    // The old note is removed and a new one injected, so there is always exactly 1
    // (not 2 stacked).
    assert_eq!(note_count, 1, "重复截断应替换旧 note 而非堆叠");
    // The second-truncation note should carry count "2" so the model perceives
    // escalating severity.
    let note = messages.iter().find(|m| {
        m.role == ROLE_INTERNAL_NOTE
            && m.content
                .as_str()
                .is_some_and(|c| c.starts_with(TRUNCATION_RETRY_NOTE_PREFIX))
    });
    assert!(
        note.and_then(|m| m.content.as_str())
            .is_some_and(|c| c.contains("Truncated 2 times")),
        "the second truncation note should carry the count"
    );
}

#[test]
fn stream_error_truncation_skips_shrink_note_and_partial_text() {
    let mut app = test_app_with_tools(&["write_file"]);
    let mcp = crate::ai::mcp::McpClient::new();
    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(mcp));
    let mut messages = vec![Message {
        role: "user".to_string(),
        content: Value::String("write a big script".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];
    let mut turn_messages = Vec::new();
    let mut persisted_turn_messages = 0;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut force_final_response = false;
    let mut terminal_dedupe_candidate: Option<String> = None;

    let step = handle_iteration_execution(
        &mut app,
        "write a big script",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        IterationExecution::Truncated(crate::ai::types::StreamResult {
            outcome: crate::ai::types::StreamOutcome::Truncated,
            tool_calls: Vec::new(),
            assistant_text: "partial content from broken stream".to_string(),
            hidden_meta: String::new(),
            reasoning_text: String::new(),
            reasoning_items: Vec::new(),
            skip_response_drain: true,
            truncated_by_length: false,
            stream_error: true,
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
        1,
        &mut false,
    )
    .unwrap();

    // Should keep retrying
    assert!(matches!(step, TurnLoopStep::Continue));
    // Should not inject a shrink hint — stream errors are unrelated to output size
    let has_shrink_note = messages.iter().any(|m| {
        m.role == ROLE_INTERNAL_NOTE
            && m.content
                .as_str()
                .is_some_and(|c| c.starts_with(TRUNCATION_RETRY_NOTE_PREFIX))
    });
    assert!(!has_shrink_note, "stream_error 截断不应注入收缩提示");
    // Should not keep partial text — partial from an interrupted stream is unreliable
    let has_partial = messages.iter().any(|m| {
        m.role == "assistant"
            && m.content
                .as_str()
                .is_some_and(|c| c.contains("partial content from broken stream"))
    });
    assert!(!has_partial, "stream_error 截断不应保留 partial text");
}

#[test]
fn runtime_synthetic_user_auto_image_followup_is_multimodal() {
    let mut app = test_app_with_tools(&[]);
    let dir = std::env::temp_dir();
    let path = dir.join(format!("tool-followup-{}.png", uuid::Uuid::new_v4()));
    std::fs::write(&path, b"fake").unwrap();
    app.current_model = crate::ai::model_names::all()
        .iter()
        .find(|m| m.is_vl)
        .map(|m| m.name.clone())
        .expect("model registry must contain at least one VL model");

    let mut messages = Vec::new();
    let mut turn_messages = Vec::new();
    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
    append_auto_image_followup_message(
        &app,
        "describe the file",
        &shared_mcp,
        &[path.to_string_lossy().to_string()],
        &mut messages,
        &mut turn_messages,
    )
    .unwrap();

    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, "user");
    assert!(is_runtime_synthetic_user_message(&messages[0]));
    assert!(messages[0].content.is_array());

    let _ = std::fs::remove_file(&path);
}
