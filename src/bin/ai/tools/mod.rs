mod ast_structural;
pub(crate) mod ast_symbols;
mod code_search;
pub(crate) mod command_tools;
mod common;
mod context_tools;
pub(crate) mod enable_tools;
mod knowledge_tools;
pub mod os_tools;
mod patch_tools;
mod permissions;
mod plan_tools;
mod rag_tools;
pub(crate) mod registry;
mod search_tools;
pub(crate) mod service;
pub(crate) mod skill_tools;
pub(crate) mod storage;
pub(crate) mod task_tools;
mod text_grep_tools;
mod tree_tools;
mod undo_tools;
mod web_tools;

#[cfg(test)]
pub use command_tools::validate_execute_command;

#[cfg(test)]
pub(crate) use registry::common::execute_tool_call;
pub(crate) use registry::common::execute_tool_call_with_args_streaming;
pub(crate) use registry::common::{
    deferred_eager_load_tool_summaries, get_tool_definitions_by_names, tool_defers_eager_load,
    tool_definitions_for_groups, tool_summaries_for_groups,
};

const BASELINE_TOOL_NAMES: &[&str] = &[
    "discover_skills",
    "load_skill",
    "enable_tools",
    "read_file",
    "list_directory",
    "find_path",
    "text_grep",
    "code_search",
    "task",
    "task_spawn",
    "task_wait",
    "task_status",
    "agent_team",
];

pub(crate) fn baseline_tool_names() -> &'static [&'static str] {
    BASELINE_TOOL_NAMES
}
