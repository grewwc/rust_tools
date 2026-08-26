use serde_json::Value;

use crate::ai::tools::command_tools::execute_command_streaming;
use crate::ai::tools::common::{
    ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
    ToolRegistration, ToolReplayRegistration, ToolSpec, ToolStreamingRegistration,
};
use crate::ai::tools::service::command::execute_command;

fn execute_command_streaming_registered(
    args: &Value,
    on_chunk: &mut crate::ai::tools::common::ToolStreamWriter<'_>,
) -> Result<String, String> {
    execute_command_streaming(args, |chunk| on_chunk(chunk))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "execute_command",
        description: "",

        execute: execute_command,
        groups: &["builtin", "core"],
    }
});

// The command exit status and compile/test diagnostics are direct evidence the agent
// uses when fixing issues later. When compressing, the full output must first be written
// to a session asset with a read-back path; failure logs must not be degraded to just the
// first line or a bare exit code. Old logs can still be pruned once the model explicitly
// marks them stale, so this does not make the session context grow monotonically.
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "execute_command",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Allow,
        counts_toward_precision_inline_budget: true,
    },
});

// execute_command is only registered as a same-turn reusable snapshot when the command
// is provably read-only (whitelisted programs / read-only git subcommands); mutating
// commands are caught by read_only_tool_signature's read-only gate, still really execute,
// and invalidate any existing read snapshot.
inventory::submit!(ToolReplayRegistration {
    name: "execute_command",
});

inventory::submit!(ToolStreamingRegistration {
    name: "execute_command",
    execute_streaming: execute_command_streaming_registered,
});
