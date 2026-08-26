// The memory_* tools are deprecated (2026-07-16); the knowledge_* tool family
// fully covers their functionality. The execute_memory_* functions in
// service/memory.rs are kept as infrastructure for internal modules such as
// knowledge_tools / reflection / memory_store.

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
