use crate::ai::tools::common::{
    ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
    ToolRegistration, ToolSpec, ToolStreamingRegistration,
};
use crate::ai::tools::service::file::{
    execute_read_file, execute_write_file, execute_write_file_streaming,
};

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "read_file",
        description: "",

        execute: execute_read_file,
        groups: &["executor", "builtin", "core"],
    }
});

// read_file 是高精度 grounding 结果：内容复现代价高，禁止有损压缩（只能零压缩
// 外溢到磁盘留指针）；但旧版本一旦被模型连续判定过时，就允许 LLM 裁剪释放上下文
// ——「不可有损压缩」不等于「不可裁剪」。
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "read_file",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Allow,
        counts_toward_precision_inline_budget: true,
    },
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "write_file",
        description: "",

        execute: execute_write_file,
        groups: &["builtin", "core"],
    }
});

inventory::submit!(ToolStreamingRegistration {
    name: "write_file",
    execute_streaming: execute_write_file_streaming,
});
