mod execution;
mod messaging;
pub(in crate::ai) mod overflow;
mod preview;

pub(in crate::ai::driver) use execution::stale_patch_targets_from_messages;
pub(super) use execution::{
    FinalGateState, completion_evidence_state, completion_tool_result_succeeded,
    handle_iteration_execution_for_model, tool_call_is_successful_mutation_candidate,
};
#[cfg(test)]
pub(super) use execution::{prepare_recent_tool_result, prepare_tool_result};
