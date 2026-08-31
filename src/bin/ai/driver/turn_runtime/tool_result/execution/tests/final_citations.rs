//! Tests for the `final_citations` cluster.

use super::common::*;
use super::super::*;

#[test]
fn final_response_citation_parser_ignores_urls_and_non_file_colon_forms() {
    let citations = final_response_citations(
        "Evidence: src/lib.rs:2-3, Cargo.toml:1:4, phase:2, https://example.com/file.rs:5, and 127.0.0.1:8080.",
    );
    assert_eq!(
        citations
            .iter()
            .map(|citation| citation.text.as_str())
            .collect::<Vec<_>>(),
        vec!["src/lib.rs:2-3", "Cargo.toml:1:4"]
    );
}

#[test]
fn final_response_citation_parser_skips_fenced_code_blocks() {
    let citations = final_response_citations(
        "See src/lib.rs:2.\n\n\
         ```rust\n\
         # src/nonexistent_example.rs:12\n\
         // Cargo.toml:8 in a diff example\n\
         ```\n\n\
         Also Cargo.toml:1:4.\n",
    );
    assert_eq!(
        citations
            .iter()
            .map(|citation| citation.text.as_str())
            .collect::<Vec<_>>(),
        vec!["src/lib.rs:2", "Cargo.toml:1:4"]
    );

    // An unclosed fence skips everything after it (conservative direction).
    assert!(final_response_citations("```text\nsrc/missing.rs:9\nmore\n").is_empty());
}

#[test]
fn final_response_citation_parser_ignores_prose_qualifier_extensions() {
    let citations = final_response_citations(
        "Rollout phase.alpha:2, build.release:3, retry.beta:4 vs src/main.rs:4.",
    );
    assert_eq!(
        citations
            .iter()
            .map(|citation| citation.text.as_str())
            .collect::<Vec<_>>(),
        vec!["src/main.rs:4"]
    );
}

#[test]
fn citation_line_check_falsifies_lines_beyond_scan_cap_cheaply() {
    let root = std::env::temp_dir().join(format!(
        "final-citation-line-check-{}",
        uuid::Uuid::new_v4().simple()
    ));
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/lib.rs"), "first\nsecond\n").unwrap();
    let path = root.join("src/lib.rs");

    // A 2-line file: the line number is provably past EOF (a file of S bytes
    // has at most S lines) even though it exceeds the line-scan cap.
    assert_eq!(
        citation_file_contains_line(&path, MAX_FINAL_CITATION_LINE_SCAN + 1),
        Some(false)
    );
    // Missing files stay provably invalid regardless of the line number.
    assert_eq!(
        citation_file_contains_line(
            &root.join("src/missing.rs"),
            MAX_FINAL_CITATION_LINE_SCAN + 1
        ),
        Some(false)
    );
    // A file large enough that the line could exist stays unknown (no scan).
    let big = root.join("src/big.txt");
    fs::write(&big, "\n".repeat(1_200_000)).unwrap();
    assert_eq!(
        citation_file_contains_line(&big, MAX_FINAL_CITATION_LINE_SCAN + 1),
        None
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn final_citation_resolution_failure_is_unknown_not_invalid() {
    // No cwd: relative citations cannot be resolved and must be skipped as
    // unknown, never flagged as provably bad.
    assert!(unvalidated_final_response_citations("See src/lib.rs:2.", None).is_empty());
    // Same for ~/ citations without HOME.
    assert_eq!(resolve_final_citation_path("~/notes.rs", None, None), None);
    assert_eq!(
        resolve_final_citation_path("~/notes.rs", None, Some(std::ffi::OsStr::new("/home/u"))),
        Some(std::path::PathBuf::from("/home/u/notes.rs"))
    );
}

#[test]
fn final_citation_resolves_basename_from_observed_tool_path() {
    let root = std::env::temp_dir().join(format!(
        "final-citation-observed-path-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let file = root.join("aida/adaptor/output_rendering.py");
    fs::create_dir_all(file.parent().unwrap()).unwrap();
    fs::write(&file, "first\nsecond\n").unwrap();
    let messages = vec![assistant_tool_call_message(test_tool_call(
        "call_read",
        "read_file",
        serde_json::json!({"file_path": file}),
    ))];

    let base_dirs = final_citation_base_dirs(&messages, Some(&root));
    assert!(unvalidated_final_response_citations_with_bases(
        "See output_rendering.py:2.",
        &base_dirs,
    )
    .is_empty());

    let _ = fs::remove_dir_all(root);
}

#[test]
fn final_citation_gate_accepts_relative_paths_from_inline_cd_command() {
    let root = std::env::temp_dir().join(format!(
        "final-citation-inline-cd-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let repository = root.join("dataagent_be");
    let session_state_store = repository.join("aida/core/session_state_store.py");
    let upload_file_service = repository
        .join("aeolus_llm/tests/service/deep_research/agent/test_agent_upload_file_service.py");
    fs::create_dir_all(session_state_store.parent().unwrap()).unwrap();
    fs::create_dir_all(upload_file_service.parent().unwrap()).unwrap();
    fs::write(&session_state_store, "line\n".repeat(46)).unwrap();
    fs::write(&upload_file_service, "line\n".repeat(1_759)).unwrap();

    let command = format!("cd {} && rg -n session_state_store", repository.display());
    let mut messages = vec![
        assistant_tool_call_message(test_tool_call(
            "call_command",
            "execute_command",
            serde_json::json!({ "command": command }),
        )),
        assistant_tool_call_message(test_tool_call(
            "call_first_read",
            "read_file",
            serde_json::json!({ "file_path": session_state_store }),
        )),
        assistant_tool_call_message(test_tool_call(
            "call_second_read",
            "read_file",
            serde_json::json!({ "file_path": upload_file_service }),
        )),
    ];
    let final_text = "See aida/core/session_state_store.py:46 and \
                      aeolus_llm/tests/service/deep_research/agent/test_agent_upload_file_service.py:1759.";
    let base_dirs = final_citation_base_dirs(&messages, Some(&root));

    assert!(base_dirs.contains(&repository));
    assert!(unvalidated_final_response_citations_with_bases(final_text, &base_dirs).is_empty());
    assert_eq!(
        final_response_citation_gate_action(&mut messages, final_text, Some(&root), false, 16, 16,),
        FinalCitationGateAction::Allow
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn inline_execute_command_cd_dir_rejects_newlines_and_shell_comments() {
    for command in [
        "cd\n/tmp/repository && rg -n citation",
        "cd \n/tmp/repository && rg -n citation",
    ] {
        let messages = vec![assistant_tool_call_message(test_tool_call(
            "call_command",
            "execute_command",
            serde_json::json!({ "command": command }),
        ))];
        assert!(
            !final_citation_base_dirs(&messages, None).contains(&std::path::PathBuf::from("/tmp/repository"))
        );
    }

    let root = std::path::PathBuf::from("/tmp/final-citation-inline-cd-comment");
    let messages = vec![assistant_tool_call_message(test_tool_call(
        "call_comment",
        "execute_command",
        serde_json::json!({ "command": "cd #&& ignored" }),
    ))];
    assert!(!final_citation_base_dirs(&messages, Some(&root)).contains(&root.join("#")));
}

#[test]
fn final_response_citation_gate_reopens_once_then_warns_for_an_invalid_line() {
    let root = std::env::temp_dir().join(format!(
        "final-citation-gate-{}",
        uuid::Uuid::new_v4().simple()
    ));
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/lib.rs"), "first\nsecond\n").unwrap();

    SUBAGENT_CWD.sync_scope(root.clone(), || {
        let final_text = "Implemented the change in src/lib.rs:9.";
        let effective_cwd = crate::ai::driver::runtime_ctx::effective_cwd().unwrap();
        assert_eq!(
            unvalidated_final_response_citations(final_text, Some(&effective_cwd)),
            vec!["src/lib.rs:9"]
        );

        let mut messages = Vec::new();
        assert_eq!(
            final_response_citation_gate_action(
                &mut messages,
                final_text,
                Some(&effective_cwd),
                false,
                1,
                16,
            ),
            FinalCitationGateAction::Reopen
        );
        assert!(messages.iter().any(|message| {
            message.role == ROLE_INTERNAL_NOTE
                && message.content.as_str().is_some_and(|text| {
                    text.starts_with(FINAL_CITATION_RETRY_MARKER)
                        && text.contains("`src/lib.rs:9`")
                })
        }));
        assert_eq!(
            final_response_citation_gate_action(
                &mut messages,
                final_text,
                Some(&effective_cwd),
                false,
                2,
                16,
            ),
            FinalCitationGateAction::Warn
        );
        assert!(unvalidated_final_response_citations(
            "Implemented the change in src/lib.rs:2.",
            Some(&effective_cwd)
        )
        .is_empty());
    });

    let _ = fs::remove_dir_all(root);
}

#[test]
fn final_response_citation_gate_warns_only_after_one_recovery_final() {
    let root = std::env::temp_dir().join(format!(
        "final-citation-finalize-{}",
        uuid::Uuid::new_v4().simple()
    ));
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/lib.rs"), "first\nsecond\n").unwrap();

    SUBAGENT_CWD.sync_scope(root.clone(), || {
        let mut app = test_app_with_tools(&["read_file"]);
        let shared_mcp =
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
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
                assistant_text: "Implemented the change in src/lib.rs:9.".to_string(),
                skip_response_drain: true,
                ..Default::default()
            })
        };

        let first_step = handle_iteration_execution(
            &mut app,
            "fix the bug",
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

        assert!(matches!(first_step, TurnLoopStep::Continue));
        assert!(!final_assistant_recorded);
        assert_eq!(terminal_dedupe_candidate, None);

        let second_step = handle_iteration_execution(
            &mut app,
            "fix the bug",
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

        assert!(matches!(second_step, TurnLoopStep::Break));
        assert!(final_assistant_recorded);
        assert!(final_assistant_text.contains(FINAL_CITATION_WARNING));
        assert_eq!(
            terminal_dedupe_candidate.as_deref(),
            Some(final_assistant_text.as_str())
        );
        assert!(turn_messages.iter().any(|message| {
            message.role == ROLE_INTERNAL_NOTE
                && message.content.as_str().is_some_and(|text| {
                    text.contains(FINAL_CITATION_UNVERIFIED_NOTE)
                })
        }));

    let _ = fs::remove_dir_all(root);
    });
}

#[test]
fn citation_rewrite_commits_only_the_accepted_final() {
    let root = std::env::temp_dir().join(format!(
        "final-citation-transactional-{}",
        uuid::Uuid::new_v4().simple()
    ));
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/lib.rs"), "first\nsecond\n").unwrap();

    SUBAGENT_CWD.sync_scope(root.clone(), || {
        let mut app = test_app_with_tools(&["read_file"]);
        let shared_mcp =
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        let mut persisted_turn_messages = 0usize;
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut force_final_response = false;
        let mut terminal_dedupe_candidate = None;
        let mut turn_had_tool_error = false;
        let response = |text: &str| {
            IterationExecution::FinalResponse(crate::ai::types::StreamResult {
                outcome: crate::ai::types::StreamOutcome::Completed,
                assistant_text: text.to_string(),
                skip_response_drain: true,
                ..Default::default()
            })
        };

        let first_step = handle_iteration_execution(
            &mut app,
            "fix the bug",
            &mcp_snapshot(&shared_mcp),
            &shared_mcp,
            response("Implemented the change in src/lib.rs:9."),
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
        assert!(matches!(first_step, TurnLoopStep::Continue));
        assert_eq!(
            terminal_dedupe_candidate, None,
            "the rejected draft was never visible and must not be replayed or deduped"
        );

        let accepted = "Implemented the change in src/lib.rs:2.";
        let second_step = handle_iteration_execution(
            &mut app,
            "fix the bug",
            &mcp_snapshot(&shared_mcp),
            &shared_mcp,
            response(accepted),
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

        assert!(matches!(second_step, TurnLoopStep::Break));
        assert_eq!(final_assistant_text, accepted);
        assert_eq!(terminal_dedupe_candidate.as_deref(), Some(accepted));
        assert!(final_assistant_recorded);
    });

    let _ = fs::remove_dir_all(root);
}

#[test]
fn citation_reopen_does_not_resurrect_rejected_draft_after_tool_round() {
    // A rejected conclusion is transactionally hidden. A verification tool round
    // may establish its own live-narration dedupe candidate, but it must never
    // resurrect the rejected draft as already-visible terminal content.
    let root = std::env::temp_dir().join(format!(
        "citation-reopen-dedupe-{}",
        uuid::Uuid::new_v4().simple()
    ));
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/lib.rs"), "first\nsecond\n").unwrap();

    SUBAGENT_CWD.sync_scope(root.clone(), || {
        let mut app = test_app_with_tools(&[TEST_REPLAY_TOOL]);
        let shared_mcp =
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        let mut persisted_turn_messages = 0usize;
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut force_final_response = false;
        let mut terminal_dedupe_candidate = None;
        let mut turn_had_tool_error = false;
        let draft = "Implemented the change in src/lib.rs:9.";
        let final_response = || {
            IterationExecution::FinalResponse(crate::ai::types::StreamResult {
                outcome: crate::ai::types::StreamOutcome::Completed,
                assistant_text: draft.to_string(),
                skip_response_drain: true,
                ..Default::default()
            })
        };

        // Step 1: the citation gate cannot validate src/lib.rs:9 (the file only has
        // two lines) and reopens without committing the draft.
        let first_step = handle_iteration_execution(
            &mut app,
            "fix the bug",
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
        assert!(matches!(first_step, TurnLoopStep::Continue));
        assert_eq!(terminal_dedupe_candidate, None);

        // Step 2: the model verifies with a tool round. Its narration is visible
        // and therefore becomes the only valid dedupe candidate.
        let second_step = handle_iteration_execution(
            &mut app,
            "fix the bug",
            &mcp_snapshot(&shared_mcp),
            &shared_mcp,
            IterationExecution::ToolCall(ToolCallExecution {
                stream_result: crate::ai::types::StreamResult {
                    outcome: crate::ai::types::StreamOutcome::ToolCall,
                    assistant_text: "核对计数器定义区块的行号，确保最终引用精确。".to_string(),
                    tool_calls: vec![test_tool_call(
                        "call_verify",
                        TEST_REPLAY_TOOL,
                        serde_json::json!({ "file_path": "src/lib.rs" }),
                    )],
                    skip_response_drain: true,
                    ..Default::default()
                },
                allowed_tool_names: rust_tools::commonw::FastSet::from_iter([
                    TEST_REPLAY_TOOL.to_string()
                ]),
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
            &mut turn_had_tool_error,
        )
        .unwrap();
        assert!(matches!(second_step, TurnLoopStep::Continue));
        assert_eq!(
            terminal_dedupe_candidate.as_deref(),
            Some("核对计数器定义区块的行号，确保最终引用精确。")
        );

        // Step 3: the model re-answers verbatim. The accepted answer plus warning
        // is now the sole terminal commit; the rejected first draft was never shown.
        let third_step = handle_iteration_execution(
            &mut app,
            "fix the bug",
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
            4,
            16,
            0,
            &mut turn_had_tool_error,
        )
        .unwrap();
        assert!(matches!(third_step, TurnLoopStep::Break));
        assert!(final_assistant_recorded);
        assert!(final_assistant_text.starts_with(draft));
        assert_eq!(
            terminal_dedupe_candidate.as_deref(),
            Some(final_assistant_text.as_str())
        );

        let _ = fs::remove_dir_all(root);
    });
}
