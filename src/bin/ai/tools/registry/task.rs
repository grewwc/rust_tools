use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::service::task::{
    execute_task_assign, execute_task_close, execute_task_collect, execute_task_start,
    execute_task_room_status,
};

// Host-side multi-worker task orchestration. `task_collect` drains the host
// inbox and `task_assign` advances shared room state, so all five stay on the
// fail-closed defaults: serial execution, no retry, never replayed
// (`task_room_status` is read-only but keeps the same policy for uniformity).
inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_start",
        description: "",
        execute: execute_task_start,
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_assign",
        description: "",
        execute: execute_task_assign,
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_collect",
        description: "",
        execute: execute_task_collect,
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_close",
        description: "",
        execute: execute_task_close,
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_room_status",
        description: "",
        execute: execute_task_room_status,
    }
});
