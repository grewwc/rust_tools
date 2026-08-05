// memory_* 工具已废弃（2026-07-16），功能由 knowledge_* 系列工具完全覆盖。
// service/memory.rs 中的 execute_memory_* 函数保留，作为 knowledge_tools / reflection /
// memory_store 等内部模块的基础设施使用。

use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::service::knowledge_update::execute_knowledge_cache_manage;


inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "knowledge_cache_manage",
        description: "",

        execute: execute_knowledge_cache_manage,
        groups: &["builtin"],
    }
});
