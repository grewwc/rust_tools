//! Outstanding-work gate: unsettled steps in a binding session plan reopen a
//! final response within a bounded budget. Done/skipped steps allow it directly.
//!
//! A plan binds the turn that writes it and resume/goal-mode turns; older plans
//! are carried context only. Failed steps stay outstanding until retried or
//! explicitly skipped. Decisions use durable plan state, never final wording.

use super::*;
use crate::ai::tools::plan_state::{StepStatus, load_plan_state, plan_binds_current_turn};

pub(in crate::ai::driver::turn_runtime) const OUTSTANDING_WORK_REQUIRED_MARKER: &str =
    "[outstanding-work-retry]";
pub(in crate::ai::driver::turn_runtime) const OUTSTANDING_WORK_WARNING: &str = "[Runtime warning] The session plan still has unsettled steps, so this final response may describe incomplete work.";

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
}

pub(in crate::ai::driver::turn_runtime) fn outstanding_work_gate_action(
    app: &App,
    messages: &mut Vec<Message>,
    force_final_response: bool,
    iteration: usize,
    max_iterations: usize,
) -> OutstandingWorkGateAction {
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
    // check, so the final lands freely.
    // A plan carried from an earlier turn stays visible context only until this
    // turn writes it.
    if !plan_binds {
        return OutstandingWorkGateAction::Allow;
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

    let steps = outstanding
        .iter()
        .map(|(number, action)| format!("step {number} ({action})"))
        .collect::<Vec<_>>()
        .join(", ");

    if outstanding.is_empty() {
        return OutstandingWorkGateAction::Allow;
    }

    if force_final_response || iteration >= max_iterations {
        return OutstandingWorkGateAction::Warn(steps);
    }

    log_outstanding_work_reopen(app, "open_steps", Some(&steps), iteration);
    let note = format!(
        "{OUTSTANDING_WORK_REQUIRED_MARKER}\n\
         The session plan still has unsettled steps: {steps}. Your final response above was REJECTED and discarded; it was not delivered.\n\
         This is not a final answer. Finish or explicitly skip the open steps with `plan_update` (a step that cannot be done must be marked skipped, not left failed), then write the final response."
    );
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
