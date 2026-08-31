//! Iteration execution entry points: drives one tool-call round inside
//! the turn loop and applies every final-response quality gate.

use super::*;

pub(in crate::ai::driver::turn_runtime) fn handle_iteration_execution(
    app: &mut App,
    question: &str,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    execution: IterationExecution,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    final_assistant_text: &mut String,
    final_assistant_recorded: &mut bool,
    force_final_response: &mut bool,
    terminal_dedupe_candidate: &mut Option<String>,
    _no_active_skill: bool,
    iteration: usize,
    max_iterations: usize,
    consecutive_truncations: usize,
    turn_had_tool_error: &mut bool,
) -> Result<TurnLoopStep, Box<dyn std::error::Error>> {
    let source_model = app.current_model.clone();
    let mut final_gate_state = FinalGateState::from_current_turn_markers(messages);
    handle_iteration_execution_for_model(
        app,
        &source_model,
        question,
        mcp_client,
        shared_mcp_client,
        execution,
        messages,
        turn_messages,
        one_shot_mode,
        persisted_turn_messages,
        final_assistant_text,
        final_assistant_recorded,
        force_final_response,
        terminal_dedupe_candidate,
        &mut final_gate_state,
        _no_active_skill,
        iteration,
        max_iterations,
        consecutive_truncations,
        turn_had_tool_error,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::ai::driver::turn_runtime) fn handle_iteration_execution_for_model(
    app: &mut App,
    source_model: &str,
    question: &str,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    execution: IterationExecution,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    final_assistant_text: &mut String,
    final_assistant_recorded: &mut bool,
    force_final_response: &mut bool,
    terminal_dedupe_candidate: &mut Option<String>,
    final_gate_state: &mut FinalGateState,
    _no_active_skill: bool,
    iteration: usize,
    max_iterations: usize,
    consecutive_truncations: usize,
    turn_had_tool_error: &mut bool,
) -> Result<TurnLoopStep, Box<dyn std::error::Error>> {
    match execution {
        IterationExecution::Exit(outcome) => Ok(TurnLoopStep::Return(outcome)),
        // Pre-timeout finalization is intercepted by the orchestrator before this
        // function is called; this arm is only an exhaustive-match fallback.
        IterationExecution::WrapUpFinal => Ok(TurnLoopStep::Continue),
        IterationExecution::RequestFailed(text) => {
            *final_assistant_text = text;
            Ok(TurnLoopStep::Break)
        }
        IterationExecution::EmptyResponse => {
            // The model returned an empty response (no text, no tool calls, no reasoning
            // content); retry automatically.
            Ok(TurnLoopStep::Continue)
        }
        IterationExecution::Truncated(stream_result) => {
            if stream_result.stream_error {
                // Truncation caused by a stream read error (unstable server): no shrink
                // hint is injected and no partial text is kept (partial from an
                // interrupted stream is unreliable); just retry. Logging already happens
                // at the orchestrator layer.
                Ok(TurnLoopStep::Continue)
            } else {
                append_truncation_retry_note(&stream_result, messages, consecutive_truncations);
                Ok(TurnLoopStep::Continue)
            }
        }
        IterationExecution::FinalResponse(mut stream_result) => {
            // Completion veto: while unclosed subagent tasks remain, bounce the round
            // back to force collecting their results. But the iteration hard cap must be
            // respected — otherwise, when a subtask never reaches a terminal state and
            // the model refuses task_wait, this would livelock forever, and resetting
            // force_final_response every round would keep knocking out the
            // orchestrator's safety brakes (tool-loop / progress-budget /
            // iteration-limit hard-stop). Past the hard cap, finalization is allowed so
            // max_iterations stays the authoritative ceiling.
            //
            // On top of the hard cap, a **separate reopen quota** is added: once the
            // reopen count within one turn reaches TASK_EVIDENCE_REOPEN_MAX, no more
            // bounces happen. Otherwise dead ends that can never be integrated (like
            // TIMED_OUT), or a model that keeps refusing task_integrate, would reopen
            // forever and spin to 4096.
            let reopen_count = task_evidence_reopen_count(messages);
            let reopen_budget_exhausted = reopen_count >= TASK_EVIDENCE_REOPEN_MAX;
            if iteration < max_iterations
                && !reopen_budget_exhausted
                && reopen_turn_for_outstanding_subagent_tasks(messages, &app.session_id)
            {
                push_task_evidence_reopen_marker(messages, reopen_count + 1);
                *force_final_response = false;
                return Ok(TurnLoopStep::Continue);
            }
            let (task_evidence_ledger, task_evidence_warning) =
                crate::ai::history::render_unintegrated_task_evidence_resilient(
                    app.config.history_file.as_path(),
                    &app.session_id,
                );
            if let Some(warning) = task_evidence_warning {
                stream_result
                    .assistant_text
                    .push_str(&format!("\n\n[Runtime warning] {warning}"));
            }
            if let Some(ledger) = task_evidence_ledger {
                if iteration < max_iterations && !reopen_budget_exhausted {
                    reopen_turn_for_unintegrated_task_evidence(messages, &ledger);
                    push_task_evidence_reopen_marker(messages, reopen_count + 1);
                    *force_final_response = false;
                    return Ok(TurnLoopStep::Continue);
                }
                // Hard cap or reopen quota exhausted: allow finalization, attaching the
                // unintegrated evidence as warning + ledger to the visible answer for the
                // user/later rounds to handle, instead of spinning on endless reopens.
                stream_result.assistant_text.push_str(
                    "\n\n[Runtime warning] Subagent results remain unintegrated after repeated reopen attempts.\n",
                );
                stream_result.assistant_text.push_str(&ledger);
            }
            let reasoning_only_completion = stream_result.assistant_text.trim().is_empty()
                && !stream_result.reasoning_text.trim().is_empty()
                && stream_result.tool_calls.is_empty();
            if reasoning_only_completion {
                // Count the retries via the number of retry markers, so multiple
                // automatic retries are supported.
                let turn_start = crate::ai::history::last_real_user_index(messages).unwrap_or(0);
                let retry_count = messages
                    .iter()
                    .skip(turn_start)
                    .filter(|message| {
                        message.role == ROLE_INTERNAL_NOTE
                            && message
                                .content
                                .as_str()
                                .is_some_and(|text| text.starts_with(REASONING_ONLY_RETRY_MARKER))
                    })
                    .count();
                let already_forced_synthesis = current_turn_has_internal_marker(
                    messages,
                    REASONING_ONLY_SYNTHESIS_MARKER,
                );
                // The iteration hard cap remains the final fallback: at max_iterations the
                // round stops with a user-visible error.
                if iteration >= max_iterations {
                    *final_assistant_text = "[Model returned only reasoning content without a final answer; please retry or switch models]"
                        .to_string();
                    return Ok(TurnLoopStep::Break);
                }
                if already_forced_synthesis {
                    // Still spinning after a forced no-reasoning synthesis: keep the
                    // synthesis note and force_final_response / thinking_disabled_override
                    // in place and do not re-inject the synthesis note. But on this path
                    // force_final_response is already set, so the orchestrator's secondary
                    // brakes (tool-loop / progress-budget / checkpoint) are all disabled,
                    // and since this is classified as FinalResponse, the consecutive_empty /
                    // truncation / stream_error fallback counters never count either — if
                    // the same byte-for-byte request were repeated each round, a
                    // deterministic model would uselessly spin to max_iterations. So a
                    // lightweight retry marker counts explicitly: one new marker per round
                    // (which also gives each round's request fresh context), and after
                    // REASONING_ONLY_POST_SYNTHESIS_MAX_RETRIES the round stops with a
                    // user-visible error.
                    let post_synthesis_retries = messages
                        .iter()
                        .skip(turn_start)
                        .filter(|message| {
                            message.role == ROLE_INTERNAL_NOTE
                                && message.content.as_str().is_some_and(|text| {
                                    text.starts_with(REASONING_ONLY_SYNTHESIS_RETRY_MARKER)
                                })
                        })
                        .count();
                    if post_synthesis_retries >= REASONING_ONLY_POST_SYNTHESIS_MAX_RETRIES {
                        *final_assistant_text = "[Model returned only reasoning content without a final answer; please retry or switch models]"
                            .to_string();
                        return Ok(TurnLoopStep::Break);
                    }
                    messages.push(Message {
                        role: ROLE_INTERNAL_NOTE.to_string(),
                        content: serde_json::Value::String(format!(
                            "{REASONING_ONLY_SYNTHESIS_RETRY_MARKER}\n{REASONING_ONLY_SYNTHESIS_RETRY_NOTE}\n(Automatic recovery attempt {}/{} after forced synthesis)",
                            post_synthesis_retries + 1,
                            REASONING_ONLY_POST_SYNTHESIS_MAX_RETRIES
                        )),
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    });
                    return Ok(TurnLoopStep::Continue);
                }
                if retry_count >= REASONING_ONLY_MAX_RETRIES || *force_final_response {
                    messages.push(Message {
                        role: ROLE_INTERNAL_NOTE.to_string(),
                        content: serde_json::Value::String(format!(
                            "{REASONING_ONLY_SYNTHESIS_MARKER}\n{REASONING_ONLY_SYNTHESIS_NOTE}"
                        )),
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    });
                    app.cli.thinking_disabled_override = true;
                    *force_final_response = true;
                    return Ok(TurnLoopStep::Continue);
                }
                messages.push(Message {
                    role: ROLE_INTERNAL_NOTE.to_string(),
                    content: serde_json::Value::String(format!(
                        "{REASONING_ONLY_RETRY_MARKER}\n{REASONING_ONLY_RETRY_NOTE}\n(Automatic recovery attempt {attempt}/{REASONING_ONLY_MAX_RETRIES})",
                        attempt = retry_count + 1,
                    )),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
                return Ok(TurnLoopStep::Continue);
            }
            // Injected-note regurgitation gate: takes priority over the other final gates.
            // When the model spits a runtime context note back verbatim as its answer
            // (especially common with weak models after the earlier kinds of reopen), the
            // text has no answer value and leaks internal prompts to the terminal and into
            // the persisted final. On a hit, give one no-tool synthesis retry; if it still
            // regurgitates, stop the round with a user-visible error instead of accepting it.
            let final_gate_reopen_allowed =
                final_gate_state.can_reopen(*force_final_response, iteration, max_iterations);
            let echo_action = if !final_gate_reopen_allowed
                && looks_like_injected_context_echo(&stream_result.assistant_text)
            {
                DanglingFinalRecoveryAction::Warn
            } else {
                injected_context_echo_recovery_action(messages, &stream_result.assistant_text)
            };
            match echo_action {
                DanglingFinalRecoveryAction::Allow => {}
                DanglingFinalRecoveryAction::RetryWithoutTools => {
                    final_gate_state.consume_retry();
                    record_force_final_reason(messages, "injected_context_echo", iteration, None);
                    *force_final_response = true;
                    return Ok(TurnLoopStep::Continue);
                }
                DanglingFinalRecoveryAction::Warn => {
                    *final_assistant_text = INJECTED_CONTEXT_ECHO_STOP.to_string();
                    return Ok(TurnLoopStep::Break);
                }
            }
            let warn_unsupported_runtime_limit = match unsupported_runtime_limit_action(
                question,
                messages,
                turn_messages,
                &stream_result.assistant_text,
                *turn_had_tool_error,
                !final_gate_reopen_allowed,
                iteration,
                max_iterations,
            ) {
                UnsupportedRuntimeLimitAction::Allow => false,
                UnsupportedRuntimeLimitAction::ReopenWithTools => {
                    final_gate_state.consume_retry();
                    *force_final_response = false;
                    return Ok(TurnLoopStep::Continue);
                }
                UnsupportedRuntimeLimitAction::Warn => true,
            };
            let warn_unverified_completion = match completion_evidence_gate_action(
                messages,
                turn_messages,
                &stream_result.assistant_text,
                !final_gate_reopen_allowed,
                iteration,
                max_iterations,
            ) {
                CompletionEvidenceGateAction::Allow => false,
                CompletionEvidenceGateAction::Reopen => {
                    final_gate_state.consume_retry();
                    // Completed drafts are transactionally deferred and were never
                    // user-visible, so they must not become a dedupe candidate.
                    *terminal_dedupe_candidate = None;
                    return Ok(TurnLoopStep::Continue);
                }
                CompletionEvidenceGateAction::Warn => true,
            };
            let effective_cwd = crate::ai::driver::runtime_ctx::effective_cwd().ok();
            let warn_unvalidated_final_citation = match final_response_citation_gate_action(
                messages,
                &stream_result.assistant_text,
                effective_cwd.as_deref(),
                !final_gate_reopen_allowed,
                iteration,
                max_iterations,
            ) {
                FinalCitationGateAction::Allow => false,
                FinalCitationGateAction::Reopen => {
                    final_gate_state.consume_retry();
                    *terminal_dedupe_candidate = None;
                    return Ok(TurnLoopStep::Continue);
                }
                FinalCitationGateAction::Warn => true,
            };
            let dangling_action = if !final_gate_reopen_allowed
                && looks_like_dangling_action_final(
                    question,
                    turn_messages,
                    &stream_result.assistant_text,
                )
            {
                DanglingFinalRecoveryAction::Warn
            } else {
                dangling_final_recovery_action(
                    question,
                    messages,
                    turn_messages,
                    &stream_result.assistant_text,
                )
            };
            let warn_dangling_final = match dangling_action {
                DanglingFinalRecoveryAction::Allow => false,
                DanglingFinalRecoveryAction::RetryWithoutTools => {
                    final_gate_state.consume_retry();
                    record_force_final_reason(messages, "dangling_action_final", iteration, None);
                    *force_final_response = true;
                    return Ok(TurnLoopStep::Continue);
                }
                DanglingFinalRecoveryAction::Warn => true,
            };
            let previously_rendered_body_matches = terminal_dedupe_candidate
                .as_deref()
                .is_some_and(|candidate| {
                    crate::ai::request::strip_digest_blocks(&stream_result.assistant_text).trim()
                        == candidate.trim()
                });
            // The current response has passed the final gate. The slot is filled with the
            // complete accepted answer below so finalize renders it exactly once; if an
            // identical tool-round narration was already visible, only new warnings remain.
            *terminal_dedupe_candidate = None;
            if warn_unsupported_runtime_limit {
                append_runtime_warning_once(
                    &mut stream_result.assistant_text,
                    UNSUPPORTED_RUNTIME_LIMIT_WARNING,
                );
                append_user_visible_final_notice(
                    terminal_dedupe_candidate,
                    UNSUPPORTED_RUNTIME_LIMIT_WARNING,
                );
            }
            if warn_dangling_final {
                append_runtime_warning_once(
                    &mut stream_result.assistant_text,
                    DANGLING_FINAL_WARNING,
                );
                append_user_visible_final_notice(terminal_dedupe_candidate, DANGLING_FINAL_WARNING);
            }
            if warn_unverified_completion {
                append_runtime_warning_once(
                    &mut stream_result.assistant_text,
                    COMPLETION_EVIDENCE_WARNING,
                );
                append_user_visible_final_notice(
                    terminal_dedupe_candidate,
                    COMPLETION_EVIDENCE_WARNING,
                );
                record_hidden_self_note(app, turn_messages, COMPLETION_EVIDENCE_UNVERIFIED_NOTE);
            }
            if warn_unvalidated_final_citation {
                append_runtime_warning_once(
                    &mut stream_result.assistant_text,
                    FINAL_CITATION_WARNING,
                );
                append_user_visible_final_notice(
                    terminal_dedupe_candidate,
                    FINAL_CITATION_WARNING,
                );
                record_hidden_self_note(app, turn_messages, FINAL_CITATION_UNVERIFIED_NOTE);
            }
            // At the hard cap we no longer reopen, but unreaped subtasks must still enter
            // both the canonical final and the terminal redraw.
            if iteration >= max_iterations {
                if let Ok(Some(notice)) =
                    task_tools::build_abandoned_tasks_notice(&app.session_id, max_iterations)
                {
                    append_runtime_warning_once(&mut stream_result.assistant_text, &notice);
                    append_user_visible_final_notice(terminal_dedupe_candidate, &notice);
                }
            }
            if !previously_rendered_body_matches {
                *terminal_dedupe_candidate = Some(stream_result.assistant_text.clone());
            }
            let was_truncated_by_length = stream_result.truncated_by_length;
            record_final_stream_response(
                app,
                stream_result,
                messages,
                turn_messages,
                final_assistant_text,
                final_assistant_recorded,
            );
            // finish_reason=length but with visible text: accept as Completed, but inject
            // a light hint so the model knows the output may be incomplete. No retry is
            // triggered (avoiding a pointless loop when a reasoning model fills its budget
            // with reasoning); the hint only reminds the model to self-check/complete in
            // the next request.
            if was_truncated_by_length {
                let note = "self_note:output_length_warning\n\
                            The previous response hit the output length limit (finish_reason=length).\n\
                            Visible text so far was kept as this round's answer. If you judge the content may be incomplete (e.g., a file write cut off mid-way),\n\
                            proactively check and complete it in the next step; if the content is already complete, ignore this note.";
                messages.push(Message {
                    role: ROLE_INTERNAL_NOTE.to_string(),
                    content: serde_json::Value::String(note.to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            Ok(TurnLoopStep::Break)
        }
        IterationExecution::ToolCall(tool_call_execution) => {
            let patch_retry_needs_fresh_read = !*force_final_response
                && patch_retry_requires_fresh_read(
                    &app.stale_patch_targets,
                    &tool_call_execution.stream_result.tool_calls,
                );
            let scoped_preflight_targets =
                if !*force_final_response && !patch_retry_needs_fresh_read {
                    mutation_scoped_instruction_preflight_targets(
                        messages,
                        &tool_call_execution.stream_result.tool_calls,
                    )
                } else {
                    Vec::new()
                };
            let scoped_preflight_needed = !scoped_preflight_targets.is_empty();
            let rejection_reason = if *force_final_response {
                Some(ToolCallRejectionReason::NoToolHandoff)
            } else if patch_retry_needs_fresh_read {
                Some(ToolCallRejectionReason::PatchRetryNeedsFreshRead)
            } else if scoped_preflight_needed {
                Some(ToolCallRejectionReason::ScopedInstructionsNeedReload)
            } else {
                None
            };
            let suppressed_read_only_results = if rejection_reason.is_none() {
                let mut results = duplicate_read_only_suppressions(
                    messages,
                    turn_messages,
                    &tool_call_execution.stream_result.tool_calls,
                );
                for call_id in duplicate_knowledge_search_call_ids(
                    messages,
                    &tool_call_execution.stream_result.tool_calls,
                ) {
                    results
                        .entry(call_id)
                        .or_insert_with(duplicate_knowledge_search_message);
                }
                results
            } else {
                HashMap::new()
            };
            let image_read_paths = if rejection_reason.is_none() {
                extract_image_paths_from_file_read_tool_calls(
                    &tool_call_execution.stream_result.tool_calls,
                )
            } else {
                Vec::new()
            };
            // Pre-tool-round hook (on_before_tools → ExecuteTools.before).
            app.fire_before_tools_hooks();
            let tool_round_candidate = handle_tool_call_round(
                app,
                source_model,
                mcp_client,
                shared_mcp_client,
                &tool_call_execution,
                messages,
                turn_messages,
                one_shot_mode,
                persisted_turn_messages,
                iteration,
                rejection_reason,
                &suppressed_read_only_results,
                turn_had_tool_error,
            )?;
            // A candidate armed by a final gate reopen (the draft conclusion was already
            // streamed live) must survive this verification tool round: the model is
            // expected to re-answer verbatim after verification, and only the draft
            // (not the tool round's short narration) can suppress that redraw in the
            // next round's terminal dedupe. Only fill the slot when it is empty.
            if terminal_dedupe_candidate.is_none() {
                *terminal_dedupe_candidate = tool_round_candidate;
            }
            append_auto_image_followup_message(
                app,
                question,
                shared_mcp_client,
                &image_read_paths,
                messages,
                turn_messages,
            )?;

            crate::ai::driver::input::clear_stdin_buffer();

            if scoped_preflight_needed {
                return Ok(TurnLoopStep::ScopedPreflightContinue(
                    scoped_preflight_targets,
                ));
            }

            if *force_final_response {
                if iteration < max_iterations && !final_gate_state.no_tool_retry_consumed() {
                    final_gate_state.consume_no_tool_retry();
                    let retry_note = Message {
                        role: ROLE_INTERNAL_NOTE.to_string(),
                        content: serde_json::Value::String(format!(
                            "{NO_TOOL_SYNTHESIS_RETRY_MARKER}\n{NO_TOOL_SYNTHESIS_RETRY_NOTE}"
                        )),
                        tool_calls: None,
                        tool_call_id: None,
                        reasoning_content: None,
                    };
                    messages.push(retry_note);
                    return Ok(TurnLoopStep::Continue);
                }

                // Stop after the second violation so the model cannot retry forever in the
                // finalization phase with disabled tools.
                let partial = tool_call_execution.stream_result.assistant_text.trim();
                *final_assistant_text = if partial.is_empty() {
                    NO_TOOL_SYNTHESIS_WARNING.to_string()
                } else {
                    format!("{partial}\n\n{NO_TOOL_SYNTHESIS_WARNING}")
                };
                *terminal_dedupe_candidate = None;
                return Ok(TurnLoopStep::Break);
            }

            {
                let mut os = app.os.lock().unwrap();
                if os.consume_yield_requested() {
                    return Ok(TurnLoopStep::Return(
                        crate::ai::driver::turn_runtime::types::TurnOutcome::Continue,
                    ));
                }
            }

            if iteration >= max_iterations {
                if *force_final_response {
                    let mut text = format!(
                        "Agent reached the tool iteration limit ({max_iterations}) without producing a final answer."
                    );
                    // At the hard cap, allow finalization: attach the still-unreaped
                    // subtask state to the final answer as a visibility fallback. The model
                    // is not bounced again (avoiding an infinite livelock when a subtask
                    // never reaches a terminal state); we only ensure unreaped results are
                    // not silently dropped.
                    if let Ok(Some(notice)) =
                        task_tools::build_abandoned_tasks_notice(&app.session_id, max_iterations)
                    {
                        text.push_str("\n\n");
                        text.push_str(&notice);
                    }
                    *final_assistant_text = text;
                    return Ok(TurnLoopStep::Break);
                }
                record_force_final_reason(messages, "iteration_limit", iteration, None);
                *force_final_response = true;
            } else {
                // AIOS: kernel is the authoritative source for tool-call quota.
                // Whether the current usage is already over budget or the next tool call
                // would exceed it, we should switch to force-final; but the tool-call quota
                // itself must not block a tool-free final answer.
                use aios_kernel::primitives::{ResourceUsageDelta, RlimitDim, RlimitVerdict};
                let os = app.os.lock().unwrap();
                if let Some(pid) = os.current_process_id() {
                    let current_verdict = os.rlimit_check(pid, &Default::default());
                    let next_tool_verdict = os.rlimit_check(
                        pid,
                        &ResourceUsageDelta {
                            tool_calls: 1,
                            ..Default::default()
                        },
                    );
                    drop(os);
                    if let RlimitVerdict::Exceeded {
                        dimension,
                        used,
                        limit,
                    } = current_verdict
                    {
                        match dimension {
                            RlimitDim::Turns => {
                                if *force_final_response {
                                    *final_assistant_text = format!(
                                        "Agent exceeded kernel rlimit ({:?}: used={} limit={}).",
                                        dimension, used, limit
                                    );
                                    return Ok(TurnLoopStep::Break);
                                }
                                record_force_final_reason(
                                    messages,
                                    "kernel_turn_rlimit",
                                    iteration,
                                    None,
                                );
                                *force_final_response = true;
                            }
                            RlimitDim::ToolCalls => {
                                record_force_final_reason(
                                    messages,
                                    "kernel_tool_call_rlimit",
                                    iteration,
                                    None,
                                );
                                *force_final_response = true;
                            }
                            _ => {}
                        }
                    }
                    if matches!(
                        next_tool_verdict,
                        RlimitVerdict::Exceeded {
                            dimension: RlimitDim::ToolCalls,
                            ..
                        }
                    ) {
                        record_force_final_reason(messages, "kernel_tool_call_rlimit", iteration, None);
                        *force_final_response = true;
                    }
                }
            }

            Ok(TurnLoopStep::Continue)
        }
    }
}
