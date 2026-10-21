use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::service::discuss::{
    execute_discuss_broadcast, execute_discuss_collect, execute_discuss_end,
    execute_discuss_start, execute_discuss_status,
};

// Host-side discuss orchestration. `discuss_collect` drains the host inbox and
// `discuss_broadcast` advances shared room state, so all five stay on the
// fail-closed defaults: serial execution, no retry, never replayed
// (`discuss_status` is read-only but keeps the same policy for uniformity).
inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "discuss_start",
        description: "",
        execute: execute_discuss_start,
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "discuss_collect",
        description: "",
        execute: execute_discuss_collect,
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "discuss_broadcast",
        description: "",
        execute: execute_discuss_broadcast,
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "discuss_end",
        description: "",
        execute: execute_discuss_end,
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "discuss_status",
        description: "",
        execute: execute_discuss_status,
    }
});
