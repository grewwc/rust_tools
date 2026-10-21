use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::service::talk::{execute_talk_read, execute_talk_send};

// Cross-session mailbox tools. Both drain or append file state, so they stay
// on the fail-closed defaults: serial execution, no retry. `talk_read` in
// particular drains the inbox and must never be replayed.
inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "talk_send",
        description: "",
        execute: execute_talk_send,
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "talk_read",
        description: "",
        execute: execute_talk_read,
    }
});
