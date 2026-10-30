//! Outstanding-work gate: a turn may land its final response only when no
//! machine-verifiable work is left unsettled.
//!
//! The gate reads the session plan (`plan_state`) and treats every step that is
//! not `done`/`skipped` as outstanding — including `failed`, which needs a
//! retry/redo or an explicit `skipped` before the turn can end; silently landing
//! on a failure would drop it. Tool-assisted turns without a binding plan use
//! no `complete` handshake at all: the assertion is only required for planned
//! work, so turns without a binding plan land freely.
//!
//! Deliberately no wording classification lives here (same rule as the other
//! final-response gates): whether the final answer claims completion is never
//! inferred from prose. The model asserts completion structurally with the
//! `complete` tool; the gate checks that assertion against the plan:
//!   - no binding plan           -> Allow (the `complete` assertion is optional)
//!   - plan settled, fresh claim -> Allow (assertion verified, consumed as spent)
//!   - plan settled, no claim    -> Reopen, demanding a `complete` call
//!   - work open, fresh claim    -> Reopen, naming the contradiction
//!   - work open, no/stale claim -> Reopen, naming the open steps
//!   - forced / retry cap / iteration cap -> Warn (the final lands, annotated with steps)
//! The claim is single-use. Standalone evaluations consume it immediately;
//! iteration execution first performs a non-consuming preflight, then consumes
//! only when the final is accepted or outstanding work rejects the assertion.
//! This lets another quality gate request a replacement draft without losing
//! the completion assertion that draft still needs to validate.

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

fn upsert_outstanding_work_note(messages: &mut Vec<Message>, note: String) {
    let turn_start = crate::ai::history::last_real_user_index(messages).unwrap_or(0);
    let mut index = messages.len();
    while index > turn_start {
        index -= 1;
        let is_existing = messages[index].role == ROLE_INTERNAL_NOTE
            && messages[index]
                .content
                .as_str()
                .is_some_and(|text| text.starts_with(OUTSTANDING_WORK_REQUIRED_MARKER));
        if is_existing {
            messages.remove(index);
        }
    }
    messages.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: serde_json::Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

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

fn completion_assertion_is_fresh(
    claim_snapshot: Option<Option<u64>>,
    expected_snapshot: Option<u64>,
    state: &ToolAssistedCompletionState,
) -> bool {
    let receipt_covers_activity = state.last_complete_receipt.is_some_and(|receipt| {
        state
            .last_actionable_result
            .is_none_or(|action| receipt > action)
    });
    receipt_covers_activity && claim_snapshot == Some(expected_snapshot)
}

pub(in crate::ai::driver::turn_runtime) fn has_successful_complete_receipt(
    turn_messages: &[Message],
) -> bool {
    tool_assisted_completion_state(turn_messages)
        .last_complete_receipt
        .is_some()
}

/// Non-consuming preflight used by final-response integrity gates. The
/// outstanding-work gate remains the sole consumer of the single-use claim.
pub(in crate::ai::driver::turn_runtime) fn has_fresh_completion_assertion(
    app: &App,
    turn_messages: &[Message],
) -> bool {
    let Ok(plan) = load_plan_state(app) else {
        return false;
    };
    let Ok(Some(claim)) = load_completion_claim(app) else {
        return false;
    };
    completion_assertion_is_fresh(
        Some(claim.plan_updated_at_ms),
        plan.as_ref().map(|plan| plan.updated_at_ms),
        &tool_assisted_completion_state(turn_messages),
    )
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
    outstanding_work_gate_action_impl(
        app,
        messages,
        turn_messages,
        force_final_response,
        iteration,
        max_iterations,
        true,
    )
}

pub(in crate::ai::driver::turn_runtime) fn preflight_outstanding_work_gate_action(
    app: &App,
    messages: &mut Vec<Message>,
    turn_messages: &[Message],
    force_final_response: bool,
    iteration: usize,
    max_iterations: usize,
) -> OutstandingWorkGateAction {
    outstanding_work_gate_action_impl(
        app,
        messages,
        turn_messages,
        force_final_response,
        iteration,
        max_iterations,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn outstanding_work_gate_action_impl(
    app: &App,
    messages: &mut Vec<Message>,
    turn_messages: &[Message],
    force_final_response: bool,
    iteration: usize,
    max_iterations: usize,
    consume_claim: bool,
) -> OutstandingWorkGateAction {
    let claim = load_completion_claim(app).ok().flatten();
    if consume_claim {
        consume_completion_claim(app);
    }
    // An unreadable plan must not brick the turn: `plan_update` already surfaces
    // the underlying error, and failing open here keeps this gate advisory.
    let plan = match load_plan_state(app) {
        Ok(plan) => plan,
        Err(_) => return OutstandingWorkGateAction::Allow,
    };
    let plan_binds = plan
        .as_ref()
        .is_some_and(|plan| plan_binds_current_turn(app, plan));
    // Without a binding plan there is no machine-verifiable work contract to
    // check, so the final lands freely; the `complete` assertion is optional.
    // A plan carried from an earlier turn stays visible context only until this
    // turn writes it.
    if !plan_binds {
        return OutstandingWorkGateAction::Allow;
    }
    let state = tool_assisted_completion_state(turn_messages);
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

    let claim_fresh = completion_assertion_is_fresh(
        claim.as_ref().map(|claim| claim.plan_updated_at_ms),
        Some(plan.updated_at_ms),
        &state,
    );
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
        upsert_outstanding_work_note(
            messages,
            format!(
                "{OUTSTANDING_WORK_REQUIRED_MARKER}\n\
                 The session plan has no unsettled steps, but no `complete` call asserts completion for this turn. Your final response above was REJECTED and discarded; it was not delivered.\n\
                 This is not a final answer. Do not write prose yet: your next message must be exactly one `complete` tool call with no other content (if work remains, keep using tools instead). After the runtime records the receipt, rewrite the final response. Any further prose final without a fresh `complete` receipt will be rejected again (at most {OUTSTANDING_WORK_MAX_REOPENS} reopens; then the final lands with an unverified warning)."
            ),
        );
        return OutstandingWorkGateAction::Reopen;
    }

    if force_final_response || iteration >= max_iterations {
        return OutstandingWorkGateAction::Warn(steps);
    }

    let note = if claim_fresh {
        log_outstanding_work_reopen(app, "open_steps_with_claim", Some(&steps), iteration);
        format!(
            "{OUTSTANDING_WORK_REQUIRED_MARKER}\n\
             You called `complete` asserting all work is done, but the session plan still has unsettled steps: {steps}. Your final response above was REJECTED and discarded; it was not delivered.\n\
             This is not a final answer. Finish or explicitly skip the open steps with `plan_update` (a step that cannot be done must be marked skipped, not left failed), then reply with exactly one `complete` tool call and no prose; after the runtime records it, rewrite the final response. Any further prose final without a fresh `complete` receipt will be rejected again."
        )
    } else {
        log_outstanding_work_reopen(app, "open_steps_no_claim", Some(&steps), iteration);
        format!(
            "{OUTSTANDING_WORK_REQUIRED_MARKER}\n\
             The session plan still has unsettled steps: {steps}. Your final response above was REJECTED and discarded; it was not delivered.\n\
             This is not a final answer. Finish or explicitly skip the open steps with `plan_update`, then reply with exactly one `complete` tool call and no prose; after the runtime records it, rewrite the final response. Any further prose final without a fresh `complete` receipt will be rejected again."
        )
    };
    upsert_outstanding_work_note(messages, note);
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
