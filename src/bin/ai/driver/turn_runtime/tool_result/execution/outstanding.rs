//! Outstanding-work gate: a turn may land its final response only when no
//! machine-verifiable work is left unsettled.
//!
//! The gate reads the session plan (`plan_state`) and treats every step that is
//! not `done`/`skipped` as outstanding — including `failed`, which needs a
//! retry/redo or an explicit `skipped` before the turn can end; silently landing
//! on a failure would drop it. Tool-assisted turns without a binding plan use
//! the same `complete` handshake, while plain no-tool Q&A passes untouched.
//!
//! Deliberately no wording classification lives here (same rule as the other
//! final-response gates): whether the final answer claims completion is never
//! inferred from prose. The model asserts completion structurally with the
//! `complete` tool; the gate checks that assertion against the plan:
//!   - no binding plan, no tools -> Allow (plain Q&A rounds pass through untouched)
//!   - no binding plan + tools   -> Require `complete` after the last tool result
//!   - plan settled, fresh claim -> Allow (assertion verified, consumed as spent)
//!   - plan settled, no claim    -> Reopen, demanding a `complete` call
//!   - work open, fresh claim    -> Reopen, naming the contradiction
//!   - work open, no/stale claim -> Reopen, naming the open steps
//!   - forced / at cap          -> Warn (the final lands, annotated with steps)
//! The claim is single-use — consumed on every evaluation — so an assertion
//! never leaks into a later round, and any `plan`/`plan_update` in between
//! silently invalidates it via the `updated_at_ms` snapshot.

use super::*;
use crate::ai::tools::plan_state::{
    StepStatus, consume_completion_claim, load_completion_claim, load_plan_state,
    plan_binds_current_turn,
};
use crate::ai::types::App;

pub(in crate::ai::driver::turn_runtime) const OUTSTANDING_WORK_REQUIRED_MARKER: &str =
    "[outstanding-work-retry]";
pub(in crate::ai::driver::turn_runtime) const OUTSTANDING_WORK_WARNING: &str = "[Runtime warning] The session plan still has unsettled steps, so this final response may describe incomplete work.";
pub(in crate::ai::driver::turn_runtime) const OUTSTANDING_WORK_UNASSERTED_WARNING: &str = "[Runtime warning] No `complete` call asserted completion, so this final response may end unfinished work.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::ai::driver::turn_runtime) enum OutstandingWorkGateAction {
    Allow,
    Reopen,
    /// Forced to land with work still open: carries the unsettled-steps summary
    /// so the user-visible warning names what is left instead of a generic note.
    Warn(String),
    /// The plan is settled but nothing called `complete`: the final lands
    /// annotated, same forced/cap semantics as [`OutstandingWorkGateAction::Warn`].
    WarnUnasserted,
}

#[derive(Debug, Default)]
struct ToolAssistedCompletionState {
    last_actionable_result: Option<usize>,
    last_complete_receipt: Option<usize>,
}

fn tool_assisted_completion_state(turn_messages: &[Message]) -> ToolAssistedCompletionState {
    let mut calls = rust_tools::commonw::FastMap::<String, String>::default();
    let mut state = ToolAssistedCompletionState::default();

    for (index, message) in turn_messages.iter().enumerate() {
        if message.role == "assistant" {
            for call in message.tool_calls.as_deref().unwrap_or_default() {
                calls.insert(call.id.clone(), call.function.name.clone());
            }
            continue;
        }
        if message.role != "tool" {
            continue;
        }
        let Some(id) = message.tool_call_id.as_deref() else {
            continue;
        };
        let Some(name) = calls.remove(id) else {
            continue;
        };
        let text = message.content.as_str().unwrap_or_default();
        match name.as_str() {
            "complete" if text.starts_with(crate::ai::tools::COMPLETE_CLAIM_RECORDED_PREFIX) => {
                state.last_complete_receipt = Some(index);
            }
            "complete" | "request_user_input" => {}
            // A pause becomes the latest unfinished condition and is owned by
            // `recoverable_pause_gate_action`. Successful activity after it will
            // establish a new actionable result and require a fresh receipt.
            _ if is_recoverable_pause_reply(text) => state.last_actionable_result = None,
            _ => state.last_actionable_result = Some(index),
        }
    }

    state
}

#[cfg(test)]
pub(in crate::ai::driver::turn_runtime) fn outstanding_work_gate_action(
    app: &App,
    messages: &mut Vec<Message>,
    force_final_response: bool,
    iteration: usize,
    max_iterations: usize,
) -> OutstandingWorkGateAction {
    // These compatibility tests exercise the plan-state matrix; supply the
    // current-turn receipt that production obtains from `turn_messages`.
    let turn_messages = vec![
        Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: "test_complete".to_string(),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "complete".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "tool".to_string(),
            content: serde_json::Value::String(format!(
                "{} test",
                crate::ai::tools::COMPLETE_CLAIM_RECORDED_PREFIX
            )),
            tool_calls: None,
            tool_call_id: Some("test_complete".to_string()),
            reasoning_content: None,
        },
    ];
    outstanding_work_gate_action_with_turn_messages(
        app,
        messages,
        &turn_messages,
        force_final_response,
        iteration,
        max_iterations,
    )
}

pub(in crate::ai::driver::turn_runtime) fn outstanding_work_gate_action_with_turn_messages(
    app: &App,
    messages: &mut Vec<Message>,
    turn_messages: &[Message],
    force_final_response: bool,
    iteration: usize,
    max_iterations: usize,
) -> OutstandingWorkGateAction {
    // The claim is single-use: consume it on every evaluation, even when there
    // is no plan to check against, so an assertion never leaks into a later round.
    let claim = load_completion_claim(app).ok().flatten();
    consume_completion_claim(app);
    // An unreadable plan must not brick the turn: `plan_update` already surfaces
    // the underlying error, and failing open here keeps this gate advisory.
    let plan = match load_plan_state(app) {
        Ok(plan) => plan,
        Err(_) => return OutstandingWorkGateAction::Allow,
    };
    let plan_binds = plan
        .as_ref()
        .is_some_and(|plan| plan_binds_current_turn(app, plan));
    let state = tool_assisted_completion_state(turn_messages);
    if !plan_binds {
        let Some(last_actionable_result) = state.last_actionable_result else {
            return OutstandingWorkGateAction::Allow;
        };
        let expected_snapshot = plan.as_ref().map(|plan| plan.updated_at_ms);
        let has_current_receipt = state
            .last_complete_receipt
            .is_some_and(|index| index > last_actionable_result);
        let claim_fresh = claim
            .as_ref()
            .is_some_and(|claim| claim.plan_updated_at_ms == expected_snapshot);
        if has_current_receipt && claim_fresh {
            return OutstandingWorkGateAction::Allow;
        }
        if force_final_response || iteration >= max_iterations {
            return OutstandingWorkGateAction::WarnUnasserted;
        }
        log_outstanding_work_reopen(app, "tool_assisted_turn_no_claim", None, iteration);
        messages.push(Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: serde_json::Value::String(format!(
                "{OUTSTANDING_WORK_REQUIRED_MARKER}\n\
                 This turn has tool results, but no successful `complete` assertion was recorded after the latest tool result.\n\
                 This is not a final answer. Inspect the user's request and the tool evidence: if work remains, continue using tools; otherwise call `complete`, then write the final response."
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
        return OutstandingWorkGateAction::Reopen;
    }
    let plan = plan.expect("binding plan must exist");
    let outstanding: Vec<(u64, String)> = plan
        .steps
        .iter()
        .filter(|step| {
            matches!(
                step.status,
                StepStatus::Pending | StepStatus::Running | StepStatus::Failed
            )
        })
        .map(|step| (step.step, step.action.clone()))
        .collect();

    let receipt_covers_activity = state.last_complete_receipt.is_some_and(|receipt| {
        state
            .last_actionable_result
            .is_none_or(|action| receipt > action)
    });
    let claim_fresh = claim.is_some_and(|claim| {
        claim.plan_updated_at_ms == Some(plan.updated_at_ms) && receipt_covers_activity
    });
    let steps = outstanding
        .iter()
        .map(|(number, action)| format!("step {number} ({action})"))
        .collect::<Vec<_>>()
        .join(", ");

    if outstanding.is_empty() {
        // A settled plan is not a completion assertion on its own: without a
        // fresh `complete` claim, a turn that stops early (no tool call at all)
        // would land as a final answer and silently end. Demand the assertion
        // until forced or at the iteration cap, then let the final land.
        if claim_fresh {
            return OutstandingWorkGateAction::Allow;
        }
        if force_final_response || iteration >= max_iterations {
            return OutstandingWorkGateAction::WarnUnasserted;
        }
        log_outstanding_work_reopen(app, "settled_plan_no_claim", None, iteration);
        messages.push(Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: serde_json::Value::String(format!(
                "{OUTSTANDING_WORK_REQUIRED_MARKER}\n\
                 The session plan has no unsettled steps, but no `complete` call asserts completion for this turn.\n\
                 This is not a final answer. If the work is done, call `complete` and write the final response; otherwise keep working with tools."
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
        return OutstandingWorkGateAction::Reopen;
    }

    if force_final_response || iteration >= max_iterations {
        return OutstandingWorkGateAction::Warn(steps);
    }

    let note = if claim_fresh {
        log_outstanding_work_reopen(app, "open_steps_with_claim", Some(&steps), iteration);
        format!(
            "{OUTSTANDING_WORK_REQUIRED_MARKER}\n\
             You called `complete` asserting all work is done, but the session plan still has unsettled steps: {steps}.\n\
             This is not a final answer. Finish or explicitly skip the open steps with `plan_update` (a step that cannot be done must be marked skipped, not left failed), then call `complete` again and write the final response."
        )
    } else {
        log_outstanding_work_reopen(app, "open_steps_no_claim", Some(&steps), iteration);
        format!(
            "{OUTSTANDING_WORK_REQUIRED_MARKER}\n\
             The session plan still has unsettled steps: {steps}.\n\
             This is not a final answer. Finish or explicitly skip the open steps with `plan_update`, then call `complete` and write the final response."
        )
    };
    messages.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: serde_json::Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    OutstandingWorkGateAction::Reopen
}

/// Make the outstanding-work reopen observable after the fact. The gate's
/// model-visible note lives only in the request projection (it must not enter
/// canonical history, or a later round would promote it to a system rule), so
/// without a side channel the reopen loop is invisible in the session record.
/// The decision log is the session side channel for runtime gate decisions.
fn log_outstanding_work_reopen(app: &App, reason: &str, target: Option<&str>, iteration: usize) {
    crate::ai::driver::decision_log::log_runtime_stop(
        crate::ai::driver::decision_log::get_decision_log_store(),
        &app.session_id,
        crate::ai::driver::runtime_ctx::current_turn_id_or_zero(),
        &format!("outstanding_work_{reason}"),
        target,
        iteration,
    );
}
