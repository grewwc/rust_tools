//! Completion-evidence gate: verifies that final answers claiming
//! task completion are backed by observed tool evidence.

use super::*;

pub(in crate::ai::driver::turn_runtime) const COMPLETION_EVIDENCE_REQUIRED_MARKER: &str =
    "self_note:completion_evidence_required";
pub(in crate::ai::driver::turn_runtime) const COMPLETION_EVIDENCE_UNVERIFIED_NOTE: &str = "runtime:completion_evidence_unverified\nA final response was recorded after a project mutation without observed post-mutation verification.";
pub(in crate::ai::driver::turn_runtime) const COMPLETION_EVIDENCE_WARNING: &str = "[Runtime warning] Completion/impact claim is unverified: no successful post-mutation check, test, diff, or status command was observed.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai::driver::turn_runtime) enum CompletionEvidenceGateAction {
    Allow,
    Reopen,
    Warn,
}

#[derive(Default)]
pub(in crate::ai::driver::turn_runtime) struct CompletionEvidenceState {
    /// Whether any change occurred in the session (tool-level or command-level). Kept
    /// for downstream consumers such as checkpoint-phase hints; the gate decision only
    /// uses `successful_tool_level_mutation` (see below), because command-level “changes”
    /// are intent classification and may misjudge read-only commands as changes.
    pub(in crate::ai::driver::turn_runtime) successful_mutation: bool,
    /// Whether a provable tool-level change occurred (apply_patch / write_file succeeded).
    /// This is the only trusted change evidence for the gate: command-level “changes” can
    /// be false positives, and Reopen/Warn based on them would force the model to repeat
    /// conclusions (the allowlist can never be complete, so we drop reliance on that class).
    pub(super) successful_tool_level_mutation: bool,
    pub(in crate::ai::driver::turn_runtime) successful_post_mutation_verification: bool,
    successful_post_mutation_scope_review: bool,
    successful_post_mutation_behavior_check: bool,
    /// Whether any successful tool call ran after the mutation (a command or a read-only
    /// tool such as read_file). The classifier cannot exhaustively recognize verification
    /// commands (e.g. python3 scripts); such calls do not prove the check passed, but they
    /// do prove the model did post-mutation work. When set, the gate silently Allows —
    /// asserting “no check observed” would be false and would tempt the model to
    /// defensively restate its conclusions.
    pub(super) successful_post_mutation_activity: bool,
    /// Whether a known check failure occurred after the mutation (e.g. cargo check output
    /// that does not confirm success). This is provable fact, not classification
    /// uncertainty; a failure is not cleared by later benign calls. When set, the gate
    /// goes Warn — the model claimed completion after a known check failure, and an
    /// honest warning causes no false repeat.
    pub(super) successful_post_mutation_failed_check: bool,
}

/// Scan only the canonical messages of the current user turn, pairing each call with its
/// result by `tool_call_id`. Only provable tool-level mutations (apply_patch / write_file
/// succeeding) invalidate earlier verification; command-level “changes” are intent
/// classification and may be false positives, so they no longer reset gate signals.
/// Within one compound command, only later checks in a pure `&&` success chain can
/// cover the latest change.
pub(in crate::ai::driver::turn_runtime) fn completion_evidence_state(
    turn_messages: &[Message],
) -> CompletionEvidenceState {
    let mut state = CompletionEvidenceState::default();
    let mut calls_by_id: HashMap<String, ToolCall> = HashMap::new();

    for message in turn_messages {
        if let Some(tool_calls) = &message.tool_calls {
            for tool_call in tool_calls {
                calls_by_id.insert(tool_call.id.clone(), tool_call.clone());
            }
        }
        if message.role != "tool" {
            continue;
        }
        let Some(tool_call) = message
            .tool_call_id
            .as_deref()
            .and_then(|tool_call_id| calls_by_id.get(tool_call_id))
        else {
            continue;
        };

        let tool_succeeded = completion_tool_result_succeeded(&message.content);
        if tool_call.function.name == "execute_command" {
            let effects = serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
                .ok()
                .map(|args| {
                    crate::ai::driver::turn_runtime::iteration::execute_command_segment_effects_for_args(&args)
                })
                .unwrap_or_default();
            let output_confirms_behavior_check =
                behavior_check_output_confirms_success(&message.content);
            // Command-level determination: whether the whole command output a failed known check.
            // When `cargo check | tail -5` fails, the tail segment itself is not a check; if
            // judged per segment it would be misrecorded as “post-mutation activity”, so the
            // whole command must be considered together.
            let output_reports_behavior_failure =
                behavior_check_output_reports_failure(&message.content);
            let mut command_has_failed_known_check = false;
            for effect in &effects {
                command_has_failed_known_check |=
                    effect.behavior_check && (!tool_succeeded || output_reports_behavior_failure);
            }
            let had_mutation_before_command = state.successful_mutation;
            for effect in &effects {
                let had_mutation = state.successful_mutation;
                // Command-level changes are only recorded into successful_mutation (for
                // downstream consumers such as checkpoint-phase hints) and no longer reset
                // gate signals: command-level “changes” are intent classification and may
                // misjudge read-only commands as changes; resetting would make the gate
                // think “nothing was done after the mutation”, causing false Reopen/Warn
                // and forcing the model to repeat conclusions.
                if effect.project_mutation {
                    state.successful_mutation = true;
                }
                if tool_succeeded
                    && had_mutation
                    && (effect.success_guaranteed
                        || (effect.behavior_check && output_confirms_behavior_check))
                    && (effect.scope_review || effect.behavior_check)
                {
                    state.successful_post_mutation_verification = true;
                    state.successful_post_mutation_scope_review |= effect.scope_review;
                    state.successful_post_mutation_behavior_check |= effect.behavior_check;
                }
            }
            // Command-level “post-mutation activity”: a successful command that ran after
            // the mutation and output no failed known check is recorded as post-mutation
            // activity. Verification commands the classifier cannot recognize (python3
            // scripts) and command-level changes themselves land here; when set, the gate
            // silently Allows — it proves the model did post-mutation work, so injecting
            // an “no check observed” assertion would be false and would tempt the model
            // to restate conclusions.
            // Conversely, a known check failure (e.g. cargo check output that does not
            // confirm success) is provable fact recorded separately; later benign calls
            // must not clear it.
            if had_mutation_before_command {
                if command_has_failed_known_check {
                    state.successful_post_mutation_failed_check = true;
                } else if tool_succeeded {
                    state.successful_post_mutation_activity = true;
                }
            }
        } else if !tool_succeeded {
            continue;
        } else if tool_call_is_successful_mutation_candidate(tool_call) {
            // Tool-level mutations (apply_patch / write_file) are the gate's only trusted
            // change evidence; each success invalidates earlier verification.
            state.successful_mutation = true;
            state.successful_tool_level_mutation = true;
            state.successful_post_mutation_verification = false;
            state.successful_post_mutation_scope_review = false;
            state.successful_post_mutation_behavior_check = false;
            state.successful_post_mutation_activity = false;
            state.successful_post_mutation_failed_check = false;
        } else if state.successful_mutation {
            // Successful read-only/informational tools after the mutation (read_file,
            // search_overflow, etc.) also count as post-mutation activity; otherwise
            // apply_patch → read_file → final would be misjudged as “nothing done” and
            // Reopened, forcing the model to repeat conclusions.
            state.successful_post_mutation_activity = true;
        }
    }

    state
}

pub(in crate::ai::driver::turn_runtime) fn behavior_check_output_confirms_success(
    content: &serde_json::Value,
) -> bool {
    let text = content.as_str().unwrap_or_default().to_ascii_lowercase();
    if behavior_check_output_reports_failure(content) {
        return false;
    }

    text.contains("test result: ok")
        || (text.contains("finished") && text.contains("target(s)"))
        || text.contains("all tests passed")
}

pub(in crate::ai::driver::turn_runtime) fn behavior_check_output_reports_failure(
    content: &serde_json::Value,
) -> bool {
    let text = content.as_str().unwrap_or_default().to_ascii_lowercase();
    text.contains("test result: failed")
        || text.contains("\nfailures:")
        || text.contains("error:")
        || text.contains("error[")
        || text.contains("could not compile")
}

pub(in crate::ai::driver::turn_runtime) fn completion_tool_result_succeeded(
    content: &serde_json::Value,
) -> bool {
    let text = content.as_str().unwrap_or_default().trim_start();
    !text.starts_with("Error") && !text.starts_with("Exit code:")
}

pub(in crate::ai::driver::turn_runtime) fn tool_call_is_successful_mutation_candidate(
    tool_call: &ToolCall,
) -> bool {
    match tool_call.function.name.as_str() {
        "apply_patch" => serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
            .ok()
            .is_some_and(|args| {
                !args
                    .get("dry_run")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            }),
        "write_file" => serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
            .ok()
            .is_some_and(|args| {
                !args
                    .get("temp")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            }),
        "execute_command" => {
            serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
                .ok()
                .and_then(|args| {
                    args.get("command")
                        .and_then(serde_json::Value::as_str)
                        .map(crate::ai::driver::turn_runtime::iteration::execute_command_may_mutate)
                })
                .unwrap_or(false)
        }
        _ => false,
    }
}

pub(in crate::ai::driver::turn_runtime) fn contains_non_negated_completion_word(
    text: &str,
    word: &str,
) -> bool {
    text.match_indices(word).any(|(start, _)| {
        let bytes = text.as_bytes();
        let end = start + word.len();
        let bounded_before = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
        let bounded_after = end == bytes.len() || !bytes[end].is_ascii_alphanumeric();
        if !bounded_before || !bounded_after {
            return false;
        }
        !text[..start]
            .split(|ch: char| !ch.is_ascii_alphabetic() && ch != '\'')
            .filter(|token| !token.is_empty())
            .rev()
            .take(3)
            .any(|token| matches!(token, "not" | "never" | "without") || token.ends_with("n't"))
    })
}

pub(in crate::ai::driver::turn_runtime) fn completion_evidence_gate_action(
    messages: &mut Vec<Message>,
    turn_messages: &[Message],
    final_text: &str,
    force_final_response: bool,
    iteration: usize,
    max_iterations: usize,
) -> CompletionEvidenceGateAction {
    let evidence = completion_evidence_state(turn_messages);
    let claim = final_text_claim_kind(final_text);
    let evidence_is_sufficient = match claim {
        FinalClaimKind::NoClaim => true,
        FinalClaimKind::Completion => evidence.successful_post_mutation_verification,
        FinalClaimKind::NoImpact => {
            evidence.successful_post_mutation_scope_review
                && evidence.successful_post_mutation_behavior_check
        }
    };
    if !evidence.successful_mutation || evidence_is_sufficient {
        return CompletionEvidenceGateAction::Allow;
    }

    // The gate only acts on “provable tool-level mutations”. Command-level “changes” are
    // intent classification and may misjudge read-only commands as changes (the allowlist
    // can never be complete); Reopen/Warn based on them would force the model to repeat
    // conclusions — the only source of erroneous repetition the runtime can fully avoid.
    if !evidence.successful_tool_level_mutation {
        return CompletionEvidenceGateAction::Allow;
    }

    // A known check failure (provable fact, not classification uncertainty) takes
    // precedence over “post-mutation activity”: even if later benign tool calls set
    // activity back to true, the failure fact is kept and we go Warn — the model
    // claimed completion after a known check failure, and an honest warning causes
    // no false repetition.
    if evidence.successful_post_mutation_failed_check {
        return CompletionEvidenceGateAction::Warn;
    }

    // Any successful post-mutation work counts (whether or not it is recognized as
    // “verification”): verification commands the classifier cannot recognize (python3
    // scripts) and read-only tools (read_file) both qualify. Silently Allow here —
    // injecting a “no check observed” assertion would be false and would make the model
    // defensively restate its conclusions; only provable “zero post-mutation activity”
    // deserves Reopen/Warn.
    if evidence.successful_post_mutation_activity {
        return CompletionEvidenceGateAction::Allow;
    }

    let already_fired =
        current_turn_has_internal_marker(messages, COMPLETION_EVIDENCE_REQUIRED_MARKER);
    if already_fired || force_final_response || iteration >= max_iterations {
        return CompletionEvidenceGateAction::Warn;
    }

    let note = format!(
        "{COMPLETION_EVIDENCE_REQUIRED_MARKER}\n\
         A successful project mutation occurred in the current user turn, but no successful post-mutation verification was observed.\n\
         This is not a final answer. Inspect the current diff, then run the narrowest targeted check/test/diff/status command.\n\
         Only then report completion or impact; if verification is impossible, report that limitation explicitly."
    );
    messages.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: serde_json::Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    CompletionEvidenceGateAction::Reopen
}
