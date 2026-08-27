pub(crate) mod command_tools;
mod common;
pub(crate) mod enable_tools;
mod knowledge_tools;
pub mod os_tools;
mod overflow_search;
mod patch_tools;
mod rag_tools;
mod permissions;
mod plan_state;
mod plan_tools;
pub(crate) mod registry;
pub(crate) mod service;
pub(crate) mod skill_tools;
pub(crate) mod storage;
pub(crate) mod task_tools;
mod text_grep_tools;
mod tree_tools;

#[cfg(test)]
pub use command_tools::validate_execute_command;

pub(crate) use patch_tools::{PATCH_TEXT_BLOCK_START, apply_patch_target_paths_from_patch};
#[cfg(test)]
pub(crate) use registry::common::ToolReplayRegistration;
#[cfg(test)]
pub(crate) use registry::common::execute_tool_call;
pub(crate) use registry::common::execute_tool_call_with_args_streaming;
pub(crate) use registry::common::{
    deferred_eager_load_tool_summaries, get_tool_definitions_by_names,
    manifest_unknown_tool_names_warning,
    tool_allows_same_turn_replay, tool_defers_eager_load, tool_definitions_for_groups,
    tool_history_policy, tool_summaries_for_groups,
};
const BASELINE_TOOL_NAMES: &[&str] = &[
    "list_skills",
    "load_skill",
    "enable_tools",
    "read_file",
    "task",
    "task_spawn",
    "task_spawn_batch",
    "task_retry",
    "task_wait",
    "task_status",
    "task_evidence_read",
    "task_integrate",
];

// Process-level allowlist needs to retain the full self-service capability above, but the manifest's resident schema only restores
// the real execution baseline; skill discovery, task orchestration, and knowledge tools are loaded on demand via `enable_tools`.
const EAGER_BASELINE_TOOL_NAMES: &[&str] = &[
    "enable_tools",
    "read_file",
];

pub(crate) fn baseline_tool_names() -> &'static [&'static str] {
    BASELINE_TOOL_NAMES
}

pub(crate) fn eager_baseline_tool_names() -> &'static [&'static str] {
    EAGER_BASELINE_TOOL_NAMES
}

const SUBAGENT_ORCHESTRATION_TOOL_NAMES: &[&str] = &[
    "task",
    "task_spawn",
    "task_spawn_batch",
    "task_retry",
    "task_wait",
    "task_status",
    "task_evidence_read",
    "task_integrate",
    "task_audit",
    "task_cancel",
    "manage_team",
    "run_agent_graph",
];

pub(crate) fn is_subagent_orchestration_tool_name(name: &str) -> bool {
    SUBAGENT_ORCHESTRATION_TOOL_NAMES.contains(&name)
}
