use crate::ai::tools::common::{
    ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
    ToolRegistration, ToolReplayRegistration, ToolSpec, ToolStreamingRegistration,
};
use crate::ai::tools::service::file::{
    execute_read_file, execute_write_file, execute_write_file_streaming,
};

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "read_file",
        description: "",

        execute: execute_read_file,
    }
});

// read_file is a high-precision grounding result: reproducing its content is
// expensive, so lossy compression is forbidden (only zero-compression spill to disk
// with a pointer is allowed). But once the model repeatedly deems an old version
// stale, the LLM may prune it to free context — "no lossy compression" does not mean
// "no pruning".
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "read_file",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Allow,
        counts_toward_precision_inline_budget: true,
    },
});

// read_file is a pure read and is treated as a stable snapshot within a turn: replay
// is allowed only when no state-changing call ran between the two reads, and the
// suppression message only points at the original result in context, never fabricates
// new data. Paths are normalized in read_only_tool_signature, so `./x` and `x` count
// as the same read.
inventory::submit!(ToolReplayRegistration {
    name: "read_file",
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "write_file",
        description: "",

        execute: execute_write_file,
    }
});

inventory::submit!(ToolStreamingRegistration {
    name: "write_file",
    execute_streaming: execute_write_file_streaming,
});
