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

// 命令退出状态与编译/测试诊断是 agent 后续修复的直接证据。压缩时必须先落到
// session asset 并留下回读路径，不能把失败日志降级为首行或仅一个 exit code。
// 旧日志仍可由模型主动标记为过时后裁剪，因此这不会让会话上下文单调增长。
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "execute_command",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Allow,
        counts_toward_precision_inline_budget: true,
    },
});

// execute_command 只有在命令可证明只读（白名单程序 / 只读 git 子命令）时才登记为
// 同轮可复用快照；变更型命令被 read_only_tool_signature 的只读闸门拦截，仍真实执行
// 并失效既有读快照。
inventory::submit!(ToolReplayRegistration {
    name: "execute_command",
});

inventory::submit!(ToolStreamingRegistration {
    name: "execute_command",
    execute_streaming: execute_command_streaming_registered,
});
