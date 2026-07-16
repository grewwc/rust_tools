// memory_* 工具已废弃（2026-07-16），功能由 knowledge_* 系列工具完全覆盖。
// service/memory.rs 中的 execute_memory_* 函数保留，作为 knowledge_tools / reflection /
// memory_store 等内部模块的基础设施使用。

use serde_json::Value;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::service::knowledge_update::execute_knowledge_cache_manage;

fn params_knowledge_cache_manage() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action: stats, clear_volatile, refresh",
                "enum": ["stats", "clear_volatile", "refresh"]
            },
            "topic": {
                "type": "string",
                "description": "Topic to refresh (required for refresh action)."
            },
            "category": {
                "type": "string",
                "description": "Category for filtering."
            },
            "query": {
                "type": "string",
                "description": "Search query for refresh."
            },
            "limit": {
                "type": "string",
                "description": "Limit for refresh results."
            }
        },
        "required": ["action"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "knowledge_cache_manage",
        description: "Manage knowledge cache: view stats, clear volatile cache, or force refresh.",
        parameters: params_knowledge_cache_manage,
        execute: execute_knowledge_cache_manage,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin"],
    }
});
