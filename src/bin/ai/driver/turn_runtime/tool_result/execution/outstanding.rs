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
//!   - plan clean              -> Allow (any claim is consumed as spent)
//!   - work open, fresh claim   -> Reopen once, naming the contradiction
//!   - work open, no/stale claim -> Reopen once, naming the open steps
//!   - budget spent / forced    -> Warn (the final lands, annotated)
//! The claim is single-use — consumed on every evaluation — so an assertion
//! never leaks into a later round, and any `plan`/`plan_update` in between
//! silently invalidates it via the `updated_at_ms` snapshot.

use super::*;
use crate::ai::tools::plan_state::{
    StepStatus, consume_completion_claim, load_completion_claim, load_plan_state,
};
use crate::ai::types::App;

pub(in crate::ai::driver::turn_runtime) const OUTSTANDING_WORK_REQUIRED_MARKER: &str =
    "[outstanding-work-retry]";
pub(in crate::ai::driver::turn_runtime) const OUTSTANDING_WORK_WARNING: &str = "[Runtime warning] The session plan still has unsettled steps, so this final response may describe incomplete work.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::ai::driver::turn_runtime) enum OutstandingWorkGateAction {
    Allow,
    Reopen,
    Warn,
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

    if outstanding.is_empty() {
        return OutstandingWorkGateAction::Allow;
    }

    let claim_fresh =
        claim.is_some_and(|claim| claim.plan_updated_at_ms == Some(plan.updated_at_ms));
    let already_fired =
        current_turn_has_internal_marker(messages, OUTSTANDING_WORK_REQUIRED_MARKER);
    if already_fired || force_final_response || iteration >= max_iterations {
        return OutstandingWorkGateAction::Warn;
    }

    let steps = outstanding
        .iter()
        .map(|(number, action)| format!("step {number} ({action})"))
        .collect::<Vec<_>>()
        .join(", ");
    let note = if claim_fresh {
        format!(
            "{OUTSTANDING_WORK_REQUIRED_MARKER}\n\
             You called `complete` asserting all work is done, but the session plan still has unsettled steps: {steps}.\n\
             This is not a final answer. Finish or explicitly skip the open steps with `plan_update` (a step that cannot be done must be marked skipped, not left failed), then call `complete` again and write the final response."
        )
    } else {
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
