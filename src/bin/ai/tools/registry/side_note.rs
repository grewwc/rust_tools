use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::service::side_note::execute_send_side_note;

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "send_side_note",
        description: "",
        execute: execute_send_side_note,
    }
});
