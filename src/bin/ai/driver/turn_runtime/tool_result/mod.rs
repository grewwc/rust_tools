mod execution;
mod messaging;
pub(in crate::ai) mod overflow;
mod preview;

pub(in crate::ai::driver) use execution::stale_patch_targets_from_messages;
pub(super) use execution::{
    FinalGateState, audit_evidence_gate_action, completion_evidence_state,
    completion_tool_result_succeeded, handle_iteration_execution_for_model,
    is_evidence_gated_audit_agent, tool_call_is_successful_mutation_candidate,
    validated_claims_gate_action,
};
#[cfg(test)]
pub(super) use execution::{prepare_recent_tool_result, prepare_tool_result};
