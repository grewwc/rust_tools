//! Outstanding-work gate: a turn may land its final response only when no
//! machine-verifiable work is left unsettled.
//!
//! The gate reads the session plan (`plan_state`) and treats every step that is
//! not `done`/`skipped` as outstanding — including `failed`, which needs a
//! retry/redo or an explicit `skipped` before the turn can end; silently landing
//! on a failure would drop it. Turns without a plan file have nothing
//! outstanding, so plain Q&A rounds pass through untouched.
//!
//! Deliberately no wording classification lives here (same rule as the other
//! final-response gates): whether the final answer claims completion is never
//! inferred from prose. The model asserts completion structurally with the
//! `complete` tool; the gate checks that assertion against the plan:
//!   - no plan file              -> Allow (plain Q&A rounds pass through untouched)
//!   - plan from an earlier turn -> Allow (carried context; binds again once this
//!                                  turn writes it, or on resume / goal-mode turns)
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

pub(in crate::ai::driver::turn_runtime) fn outstanding_work_gate_action(
    app: &App,
    messages: &mut Vec<Message>,
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
    let Some(plan) = plan else {
        return OutstandingWorkGateAction::Allow;
    };
    // A plan written by an earlier turn is carried context, not this turn's
    // commitment: a new, unrelated request must not be forced back into it.
    if !plan_binds_current_turn(app, &plan) {
        return OutstandingWorkGateAction::Allow;
    }
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

    let claim_fresh =
        claim.is_some_and(|claim| claim.plan_updated_at_ms == Some(plan.updated_at_ms));
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
