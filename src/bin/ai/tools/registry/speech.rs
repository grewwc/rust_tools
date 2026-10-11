use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::service::speech::execute_generate_speech;

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "generate_speech",
        description: "",
        execute: execute_generate_speech,
    }
});
