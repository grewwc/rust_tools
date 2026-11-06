use crate::ai::tools::common::{
    ToolCallPolicy, ToolCallPolicyRegistration, ToolPersistentCachePolicy, ToolRegistration,
    ToolSpec,
};
use crate::ai::tools::service::side_note::execute_send_side_note;

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "send_side_note",
        description: "",
        execute: execute_send_side_note,
    }
});

inventory::submit!(ToolCallPolicyRegistration {
    name: "send_side_note",
    policy: ToolCallPolicy {
        argument_aliases: &[
            ("content", "note"),
            ("target_task_id", "task_id"),
            ("target", "task_id"),
        ],
        persistent_cache: ToolPersistentCachePolicy::Disabled,
    },
});
