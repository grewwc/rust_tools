//! Tests for the `completion_gate` cluster.

use super::super::*;
use super::common::*;

#[test]
fn completion_evidence_gate_reopens_once_then_warns_on_second_final() {
    let mutation = test_tool_call(
        "call_patch",
        "apply_patch",
        serde_json::json!({
            "patch": "*** Begin Patch\n*** Update File: /tmp/project/lib.rs\n@@\n-old\n+new\n*** End Patch"
        }),
    );
    let evidence_messages = vec![
        assistant_tool_call_message(mutation),
        tool_result_message("call_patch", "Successfully patched 1 file."),
    ];
    let mut app = test_app_with_tools(&["apply_patch", "execute_command"]);
    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
    let mut messages = evidence_messages.clone();
    let mut turn_messages = evidence_messages;
    let mut persisted_turn_messages = 0usize;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut force_final_response = false;
    let mut terminal_dedupe_candidate = None;
    let mut turn_had_tool_error = false;

    let final_response = || {
        IterationExecution::FinalResponse(crate::ai::types::StreamResult {
            outcome: crate::ai::types::StreamOutcome::Completed,
            assistant_text: "已修复。".to_string(),
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
    assert!(final_assistant_text.is_empty());
    assert!(!final_assistant_recorded);
    assert_eq!(terminal_dedupe_candidate, None);
    assert_eq!(
        messages
            .iter()
            .filter(|message| {
                message.role == ROLE_INTERNAL_NOTE
                    && message
                        .content
                        .as_str()
                        .is_some_and(|text| text.starts_with(COMPLETION_EVIDENCE_REQUIRED_MARKER))
            })
            .count(),
        1
    );

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
    assert!(final_assistant_text.starts_with("已修复。"));
    assert!(final_assistant_text.contains(COMPLETION_EVIDENCE_WARNING));
    assert_eq!(
        terminal_dedupe_candidate.as_deref(),
        Some(final_assistant_text.as_str()),
        "only the accepted final, including its runtime warning, is committed to the terminal"
    );
    assert!(messages.iter().any(|message| {
        message.role == "assistant"
            && message
                .content
                .as_str()
                .is_some_and(|text| text.contains(COMPLETION_EVIDENCE_WARNING))
    }));
    assert_eq!(
        messages
            .iter()
            .filter(|message| {
                message.role == ROLE_INTERNAL_NOTE
                    && message
                        .content
                        .as_str()
                        .is_some_and(|text| text.starts_with(COMPLETION_EVIDENCE_REQUIRED_MARKER))
            })
            .count(),
        1
    );
    assert!(turn_messages.iter().any(|message| {
        message.role == ROLE_INTERNAL_NOTE
            && message
                .content
                .as_str()
                .is_some_and(|text| text.contains(COMPLETION_EVIDENCE_UNVERIFIED_NOTE))
    }));
}

#[test]
fn completion_evidence_gate_allows_unrecognized_post_mutation_activity_silently() {
    // The model verified after the mutation only with commands the classifier cannot
    // recognize (python3 scripts): there is real post-mutation activity but no
    // “recognized check”. Silently Allow here — neither Reopen nor append a false
    // “no check observed” warning (and record no internal note either), otherwise the
    // model defensively restates its conclusions. This is exactly the root of
    // “repeated conclusions”, and the runtime must never be its source.
    let mutation = test_tool_call(
        "call_patch",
        "apply_patch",
        serde_json::json!({
            "patch": "*** Begin Patch\n*** Update File: /tmp/project/lib.rs\n@@\n-old\n+new\n*** End Patch"
        }),
    );
    let unrecognized_check = test_tool_call(
        "call_verify",
        "execute_command",
        serde_json::json!({ "command": "python3 /tmp/project/verify.py" }),
    );
    let evidence_messages = vec![
        assistant_tool_call_message(mutation),
        tool_result_message("call_patch", "Successfully patched 1 file."),
        assistant_tool_call_message(unrecognized_check),
        tool_result_message("call_verify", "all checks passed"),
    ];
    let evidence = completion_evidence_state(&evidence_messages);
    assert!(evidence.successful_mutation);
    assert!(!evidence.successful_post_mutation_verification);
    assert!(
        evidence.successful_post_mutation_activity,
        "python3 校验虽未被识别为检查，也应记为变更后活动"
    );

    let mut app = test_app_with_tools(&["apply_patch", "execute_command"]);
    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
    let mut messages = evidence_messages.clone();
    let mut turn_messages = evidence_messages;
    let mut persisted_turn_messages = 0usize;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut force_final_response = false;
    let mut terminal_dedupe_candidate = None;
    let mut turn_had_tool_error = false;

    let final_response = || {
        IterationExecution::FinalResponse(crate::ai::types::StreamResult {
            outcome: crate::ai::types::StreamOutcome::Completed,
            assistant_text: "已修复。".to_string(),
            skip_response_drain: true,
            ..Default::default()
        })
    };
    let step = handle_iteration_execution(
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

    // The first final is accepted directly (silent Allow) with no Reopen and no
    // warning appended, so the model never restates its conclusion.
    assert!(matches!(step, TurnLoopStep::Break));
    assert!(final_assistant_recorded);
    assert!(final_assistant_text.starts_with("已修复。"));
    assert!(
        !final_assistant_text.contains(COMPLETION_EVIDENCE_WARNING),
        "变更后活动静默 Allow，不应追加'未观察到检查'的虚假警告"
    );
    assert!(
        !turn_messages.iter().any(|message| {
            message.role == ROLE_INTERNAL_NOTE
                && message
                    .content
                    .as_str()
                    .is_some_and(|text| text.starts_with(COMPLETION_EVIDENCE_UNVERIFIED_NOTE))
        }),
        "变更后活动静默 Allow，不应记入'未观察到验证'的内部注记"
    );
    assert_eq!(
        messages
            .iter()
            .filter(|message| {
                message.role == ROLE_INTERNAL_NOTE
                    && message
                        .content
                        .as_str()
                        .is_some_and(|text| text.starts_with(COMPLETION_EVIDENCE_REQUIRED_MARKER))
            })
            .count(),
        0,
        "有变更后活动时不应注入 completion_evidence_required 重开笔记"
    );
}

#[test]
fn dangling_action_gate_takes_over_when_mutation_final_has_no_completion_claim() {
    let mutation = test_tool_call(
        "call_patch",
        "apply_patch",
        serde_json::json!({
            "patch": "*** Begin Patch\n*** Update File: /tmp/project/lib.rs\n@@\n-old\n+new\n*** End Patch"
        }),
    );
    let evidence_messages = vec![
        assistant_tool_call_message(mutation),
        tool_result_message("call_patch", "Successfully patched 1 file."),
    ];
    let mut app = test_app_with_tools(&["apply_patch", "execute_command"]);
    let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
    let mut messages = evidence_messages.clone();
    let mut turn_messages = evidence_messages;
    let mut persisted_turn_messages = 0usize;
    let mut final_assistant_text = String::new();
    let mut final_assistant_recorded = false;
    let mut force_final_response = false;
    let mut terminal_dedupe_candidate = None;
    let mut turn_had_tool_error = false;

    let step = handle_iteration_execution(
        &mut app,
        "fix the bug",
        &mcp_snapshot(&shared_mcp),
        &shared_mcp,
        IterationExecution::FinalResponse(crate::ai::types::StreamResult {
            outcome: crate::ai::types::StreamOutcome::Completed,
            assistant_text: "Let me inspect the diff and run the targeted test.".to_string(),
            skip_response_drain: true,
            ..Default::default()
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
        &mut turn_had_tool_error,
    )
    .unwrap();

    assert!(matches!(step, TurnLoopStep::Continue));
    // The completion gate only reopens finals that CLAIM completion
    // (FinalClaimKind::NoClaim passes the evidence check unverified). A final
    // that announces an action but ends the turn instead falls through to the
    // dangling-final gate, which forces one no-tool synthesis retry and
    // records the no-tool wrap-up root cause.
    assert!(
        force_final_response,
        "a dangling-action final must force the no-tool synthesis retry"
    );
    assert!(
        messages.iter().any(|message| {
            message.role == ROLE_INTERNAL_NOTE
                && message
                    .content
                    .as_str()
                    .is_some_and(|text| text.starts_with("[runtime-tool-stop]"))
        }),
        "the dangling-action retry must record its force-final reason"
    );
    assert!(
        !messages.iter().any(|message| {
            message
                .content
                .as_str()
                .is_some_and(|text| text.starts_with(COMPLETION_EVIDENCE_REQUIRED_MARKER))
        }),
        "no completion claim means the completion gate must stay silent"
    );
}

#[test]
fn completion_evidence_gate_requires_check_after_generic_mutation_claim() {
    let mutation = test_tool_call(
        "call_write",
        "write_file",
        serde_json::json!({ "file_path": "/tmp/project/lib.rs", "content": "new" }),
    );
    let turn_messages = vec![
        assistant_tool_call_message(mutation),
        tool_result_message("call_write", "Successfully wrote to /tmp/project/lib.rs"),
    ];
    let mut messages = turn_messages.clone();

    assert_eq!(
        completion_evidence_gate_action(
            &mut messages,
            &turn_messages,
            "Changes are ready.",
            false,
            2,
            16,
        ),
        CompletionEvidenceGateAction::Reopen
    );
}

#[test]
fn completion_evidence_gate_ignores_temp_write_file() {
    let temp_write = test_tool_call(
        "call_temp",
        "write_file",
        serde_json::json!({ "file_path": "scratch.txt", "content": "x", "temp": true }),
    );
    assert!(!tool_call_is_successful_mutation_candidate(&temp_write));
}

#[test]
fn completion_evidence_gate_ignores_execute_command_temp_redirections() {
    let command = test_tool_call(
        "call_command",
        "execute_command",
        serde_json::json!({
            "command": "grep -rhoE 'name: \"[a-z_]+\"' src/bin/ai/tools/ | sed 's/name: //' | tr -d '\"' | sort -u > /tmp/registered.txt; ls src/bin/ai/tool_descriptions/ | sed 's/.json//' | sort -u > /tmp/jsonnames.txt; comm -23 /tmp/registered.txt /tmp/jsonnames.txt",
            "cwd": crate::ai::driver::runtime_ctx::effective_cwd().unwrap(),
        }),
    );
    let turn_messages = vec![
        assistant_tool_call_message(command),
        tool_result_message("call_command", "48 registrations match 48 JSON files"),
    ];
    let evidence = completion_evidence_state(&turn_messages);
    let mut messages = turn_messages.clone();

    assert!(
        !evidence.successful_mutation,
        "系统临时文件不应触发项目变更证据门"
    );
    assert_eq!(
        completion_evidence_gate_action(
            &mut messages,
            &turn_messages,
            "名称完全对齐，没有发现漂移。",
            false,
            2,
            16,
        ),
        CompletionEvidenceGateAction::Allow
    );
}

#[test]
fn completion_evidence_gate_accepts_successful_post_mutation_check() {
    let mutation = test_tool_call(
        "call_write",
        "write_file",
        serde_json::json!({ "file_path": "/tmp/project/lib.rs", "content": "new" }),
    );
    let verification = test_tool_call(
        "call_check",
        "execute_command",
        serde_json::json!({ "command": "cargo check --bin a" }),
    );
    let turn_messages = vec![
        assistant_tool_call_message(mutation),
        tool_result_message("call_write", "Successfully wrote to /tmp/project/lib.rs"),
        assistant_tool_call_message(verification),
        tool_result_message("call_check", "Finished dev profile"),
    ];
    let mut messages = turn_messages.clone();

    assert_eq!(
        completion_evidence_gate_action(
            &mut messages,
            &turn_messages,
            "Implemented and fixed.",
            false,
            2,
            16,
        ),
        CompletionEvidenceGateAction::Allow
    );
    assert!(!messages.iter().any(|message| {
        message.role == ROLE_INTERNAL_NOTE
            && message
                .content
                .as_str()
                .is_some_and(|text| text.starts_with(COMPLETION_EVIDENCE_REQUIRED_MARKER))
    }));
}

#[test]
fn completion_evidence_gate_accepts_piped_check_with_success_sentinel() {
    let command = "cargo test --bin a replayed_content_part_added 2>&1 | tail -6";
    let mutation = test_tool_call(
        "call_write",
        "write_file",
        serde_json::json!({ "file_path": "/tmp/project/lib.rs", "content": "new" }),
    );
    let verification = test_tool_call(
        "call_check",
        "execute_command",
        serde_json::json!({ "command": command }),
    );
    let args = serde_json::json!({ "command": command });
    let effects =
        super::super::super::super::iteration::execute_command_segment_effects_for_args(&args);
    assert!(
        effects.iter().any(|effect| effect.behavior_check),
        "expected behavior check effect for {command:?}: {effects:?}"
    );
    assert!(
        !effects
            .iter()
            .any(|effect| effect.project_mutation && !effect.behavior_check),
        "non-check segment must not reset verification after the check: {effects:?}"
    );
    let turn_messages = vec![
        assistant_tool_call_message(mutation),
        tool_result_message("call_write", "Successfully wrote to /tmp/project/lib.rs"),
        assistant_tool_call_message(verification),
        tool_result_message(
            "call_check",
            "running 1 test\n\
             test ai::stream::runtime::tests::replayed_content_part_added_does_not_duplicate_visible_text ... ok\n\n\
             test result: ok. 1 passed; 0 failed; 0 ignored; 1748 filtered out; finished in 0.00s",
        ),
    ];
    assert!(behavior_check_output_confirms_success(
        &turn_messages[3].content
    ));
    let evidence = completion_evidence_state(&turn_messages);
    assert!(evidence.successful_mutation);
    assert!(evidence.successful_post_mutation_verification);
    let mut messages = turn_messages.clone();

    assert_eq!(
        completion_evidence_gate_action(
            &mut messages,
            &turn_messages,
            "Implemented and verified.",
            false,
            2,
            16,
        ),
        CompletionEvidenceGateAction::Allow
    );
}

#[test]
fn completion_evidence_gate_warns_on_piped_check_without_success_sentinel() {
    // `cargo check 2>&1 | tail -5` output is an error message: the check really ran,
    // but the output cannot confirm success, which counts as a “failed known check”
    // (provable fact). Claiming completion here deserves an honest Warn, not a
    // Reopen — the model already tried the check, and pushing it to “run the check”
    // again would produce repeated output; warning + internal note is enough to drive
    // the next round toward convergence.
    let mutation = test_tool_call(
        "call_write",
        "write_file",
        serde_json::json!({ "file_path": "/tmp/project/lib.rs", "content": "new" }),
    );
    let verification = test_tool_call(
        "call_check",
        "execute_command",
        serde_json::json!({
            "command": "cargo check --bin a 2>&1 | tail -5"
        }),
    );
    let turn_messages = vec![
        assistant_tool_call_message(mutation),
        tool_result_message("call_write", "Successfully wrote to /tmp/project/lib.rs"),
        assistant_tool_call_message(verification),
        tool_result_message(
            "call_check",
            "error[E0425]: cannot find value `x` in this scope",
        ),
    ];
    let mut messages = turn_messages.clone();

    assert_eq!(
        completion_evidence_gate_action(
            &mut messages,
            &turn_messages,
            "Implemented and verified.",
            false,
            2,
            16,
        ),
        CompletionEvidenceGateAction::Warn
    );
}

#[test]
fn completion_evidence_gate_allows_command_level_mutation_with_same_command_check() {
    // Pure command-level mutation + a successful check inside the same command
    // (printf > file && cargo check). Command-level “mutations” are intent
    // classification; the gate only accepts provable tool-level mutations, so this
    // is always Allowed; the successful check is not punished, but it is no longer a
    // basis for the gate to allow either.
    let command = test_tool_call(
        "call_command",
        "execute_command",
        serde_json::json!({
            "command": "printf x > src/generated.txt && cargo check --bin a"
        }),
    );
    let turn_messages = vec![
        assistant_tool_call_message(command),
        tool_result_message("call_command", "Finished dev profile"),
    ];
    let mut messages = turn_messages.clone();

    assert_eq!(
        completion_evidence_gate_action(
            &mut messages,
            &turn_messages,
            "Changes are ready.",
            false,
            2,
            16,
        ),
        CompletionEvidenceGateAction::Allow
    );
}

#[test]
fn completion_evidence_gate_warns_after_failed_check_even_with_later_activity() {
    // apply_patch → known check failure (cargo check output does not confirm success)
    // → later benign command (ls). The benign call resets activity to true, but the
    // failure is provable fact and must not be silently allowed: the gate should Warn
    // (an honest warning, not classification uncertainty, so no false repetition)
    // rather than Allow.
    let mutation = test_tool_call(
        "call_patch",
        "apply_patch",
        serde_json::json!({
            "patch": "*** Begin Patch\n*** Update File: /tmp/project/lib.rs\n@@\n-old\n+new\n*** End Patch"
        }),
    );
    let failed_check = test_tool_call(
        "call_check",
        "execute_command",
        serde_json::json!({ "command": "cargo check --bin a 2>&1 | tail -5" }),
    );
    let benign = test_tool_call(
        "call_ls",
        "execute_command",
        serde_json::json!({ "command": "ls" }),
    );
    let turn_messages = vec![
        assistant_tool_call_message(mutation),
        tool_result_message("call_patch", "Successfully patched 1 file."),
        assistant_tool_call_message(failed_check),
        tool_result_message(
            "call_check",
            "error[E0425]: cannot find value `x` in this scope",
        ),
        assistant_tool_call_message(benign),
        tool_result_message("call_ls", "src  target"),
    ];
    let mut messages = turn_messages.clone();

    let evidence = completion_evidence_state(&turn_messages);
    assert!(evidence.successful_tool_level_mutation);
    assert!(evidence.successful_post_mutation_failed_check);
    assert!(evidence.successful_post_mutation_activity);

    assert_eq!(
        completion_evidence_gate_action(&mut messages, &turn_messages, "已修复。", false, 2, 16,),
        CompletionEvidenceGateAction::Warn
    );
}

#[test]
fn completion_evidence_gate_allows_command_level_mutation_without_tool_evidence() {
    // Pure command-level mutation (sed -i ... ; cargo check): there is no provable
    // tool-level mutation like apply_patch / write_file. Command-level “mutations”
    // are intent classification and may misjudge read-only commands as mutations
    // (the allowlist can never be complete); Reopen based on them would force the
    // model to repeat conclusions. So the gate silently Allows any pure command-level
    // mutation — convergence strength yields to the higher-priority invariant of
    // “never wrongly producing repeated output”.
    let command = test_tool_call(
        "call_command",
        "execute_command",
        serde_json::json!({
            "command": "sed -i '' -e 's/old/new/' missing.rs; cargo check --bin a"
        }),
    );
    let turn_messages = vec![
        assistant_tool_call_message(command),
        tool_result_message("call_command", "Finished dev profile"),
    ];
    let mut messages = turn_messages.clone();

    assert_eq!(
        completion_evidence_gate_action(
            &mut messages,
            &turn_messages,
            "Changes are ready.",
            false,
            2,
            16,
        ),
        CompletionEvidenceGateAction::Allow
    );
}

#[test]
fn completion_evidence_records_direct_nonzero_check_after_mutation() {
    let turn_messages = vec![
        assistant_tool_call_message(test_tool_call(
            "call_patch",
            "apply_patch",
            serde_json::json!({"patch": "*** Begin Patch\n*** End Patch"}),
        )),
        tool_result_message("call_patch", "Successfully patched src/lib.rs"),
        assistant_tool_call_message(test_tool_call(
            "call_check",
            "execute_command",
            serde_json::json!({"command": "cargo check --bin a"}),
        )),
        tool_result_message(
            "call_check",
            "Exit code: 101\n\nerror: could not compile `rust_tools`",
        ),
    ];

    let evidence = completion_evidence_state(&turn_messages);
    assert!(evidence.successful_mutation);
    assert!(evidence.successful_post_mutation_failed_check);
    assert!(!evidence.successful_post_mutation_verification);
}
