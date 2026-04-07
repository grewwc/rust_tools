mod cargo_tools;
pub(crate) mod command_tools;
mod common;
mod context_tools;
mod git_tools;
mod knowledge_tools;
mod lsp_tools;
mod permissions;
mod rag_tools;
mod plan_tools;
mod patch_tools;
pub(crate) mod registry;
mod search_tools;
pub(crate) mod service;
mod skill_tools;
pub(crate) mod storage;
mod task_tools;
mod undo_tools;
mod web_tools;

#[cfg(test)]
pub use command_tools::validate_execute_command;

#[cfg(test)]
pub(crate) use registry::common::execute_tool_call;
pub(crate) use registry::common::execute_tool_call_with_args;
pub(crate) use registry::common::{
    get_builtin_tool_definitions, get_tool_definitions_by_names, tool_definitions_for_groups,
};
