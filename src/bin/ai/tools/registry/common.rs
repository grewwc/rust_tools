use std::sync::LazyLock;

use rust_tools::{commonw::FastSet, cw::SkipMap};
use serde_json::Value;

use crate::ai::tools::os_tools::GLOBAL_OS;
use crate::ai::tools::permissions::ToolPermissions;
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use crate::ai::types::{FunctionDefinition, ToolCall, ToolDefinition, ToolResult};
use aios_kernel::{
    kernel::{Kernel, Signal},
    primitives::FutexAddr,
};
use chrono::Local;

/// Static specification for a builtin tool, including its name,
/// description, parameter schema, execution function, and group memberships.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ToolAsyncPolicy {
    SyncOnly,
    Spawnable,
}

#[derive(Clone, Copy)]
pub(crate) struct ToolSpec {
    pub(crate) name: &'static str,
    pub(crate) description: &'static str,
    pub(crate) parameters: fn() -> Value,
    pub(crate) execute: fn(&Value) -> Result<String, String>,
    pub(crate) async_policy: ToolAsyncPolicy,
    pub(crate) groups: &'static [&'static str],
}

/// Registry entry submitted via `inventory!` to register a tool
/// at compile time for runtime discovery.
pub(crate) struct ToolRegistration {
    pub(crate) spec: ToolSpec,
}

inventory::collect!(ToolRegistration);

pub(crate) type ToolStreamWriter<'a> = dyn FnMut(&[u8]) + 'a;
pub(crate) type ToolStreamExecutor =
    for<'a> fn(&Value, &mut ToolStreamWriter<'a>) -> Result<String, String>;

/// 可选的流式执行注册：只给确实需要实时 terminal 反馈的 builtin tool 使用。
/// 未注册的工具仍沿用原有同步 `execute` 路径，不需要改动现有 ToolSpec。
pub(crate) struct ToolStreamingRegistration {
    pub(crate) name: &'static str,
    pub(crate) execute_streaming: ToolStreamExecutor,
}

inventory::collect!(ToolStreamingRegistration);

/// 终端回显配置：控制工具调用时是否把入参 / 输出结果打印到终端。
/// 默认全部为 `false`，仅对用户可见性价值较高的工具（如 `plan`）显式开启。
/// 通过独立的 `ToolDisplayRegistration` 提交，不改动现有 `ToolSpec`，
/// 保持向后兼容。
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct ToolDisplayConfig {
    /// 是否在终端打印工具调用入参。
    pub(crate) print_args: bool,
    /// 是否在终端打印工具输出结果。
    pub(crate) print_result: bool,
}

/// 可选的终端回显注册：只给需要回显入参/结果的 builtin tool 使用。
/// 未注册的工具沿用默认配置（均不回显），不需要改动现有 `ToolSpec`。
pub(crate) struct ToolDisplayRegistration {
    pub(crate) name: &'static str,
    pub(crate) config: ToolDisplayConfig,
}

inventory::collect!(ToolDisplayRegistration);

static TOOL_DISPLAY_INDEX: LazyLock<SkipMap<String, ToolDisplayConfig>> = LazyLock::new(|| {
    let mut index: SkipMap<String, ToolDisplayConfig> = SkipMap::default();
    for reg in inventory::iter::<ToolDisplayRegistration> {
        let name = reg.name.to_string();
        if !index.contains_key(&name) {
            index.insert(name, reg.config);
        }
    }
    index
});

/// 查询某个工具的终端回显配置；未注册的工具返回全 `false` 默认值。
pub(crate) fn tool_display_config(name: &str) -> ToolDisplayConfig {
    TOOL_DISPLAY_INDEX
        .get_ref(&name.to_string())
        .copied()
        .unwrap_or_default()
}

/// 有损压缩策略：控制该工具结果是否允许被行裁剪 / 折叠 / 摘要等有损压缩。
/// `Never` 表示 precision 结果，压缩路径只能零压缩外溢到磁盘并留指针 stub。
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) enum ToolLossyCompressPolicy {
    /// 默认：允许有损压缩（普通概览型工具结果）。
    #[default]
    Allow,
    /// 禁止有损压缩：内容复现代价高（如 `read_file` / 检索类 / `execute_command`），
    /// 一旦被裁剪模型会反复重跑同一次操作，表现为失忆/原地打转。
    Never,
}

/// LLM 引导裁剪策略：控制该工具结果是否允许被模型标记后裁剪成占位符。
/// 与有损压缩正交——「不可有损压缩」不等于「不可裁剪」：`read_file` 的旧
/// 版本一旦被模型连续判定过时，就应允许裁剪以释放上下文。`plan` 允许有损
/// 压缩但禁止 LLM 裁剪：最新一版由最近工具组保护窗口完整保留，旧版可摘要
/// 压缩以释放上下文，但模型不应单方"判废"既有规划。
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) enum ToolPrunePolicy {
    /// 默认：允许被 LLM 引导裁剪（仍受最近窗口保护与连续标记阈值约束）。
    #[default]
    Allow,
    /// 永不裁剪（如 `plan`）。
    Never,
}

/// 工具的历史保留策略：把「有损压缩」与「LLM 裁剪」两个正交维度合并声明。
/// 通过独立的 `ToolHistoryPolicyRegistration` 提交，不改动 `ToolSpec`，
/// 未注册的工具取默认值（两维度均 `Allow`）。
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct ToolHistoryPolicy {
    pub(crate) lossy_compress: ToolLossyCompressPolicy,
    pub(crate) prune: ToolPrunePolicy,
    /// 最近工具组过大时，该工具是否占用高精度结果的 inline 预算。
    /// 聚合型工具（如 task_wait）即使同样禁止有损压缩，也不占用此预算。
    pub(crate) counts_toward_precision_inline_budget: bool,
}

impl ToolHistoryPolicy {
    /// 是否允许对该工具结果做有损压缩（行裁剪/折叠/摘要）。
    pub(crate) fn allows_lossy_compress(&self) -> bool {
        matches!(self.lossy_compress, ToolLossyCompressPolicy::Allow)
    }

    /// 是否允许该工具结果被 LLM 引导裁剪。
    pub(crate) fn allows_prune(&self) -> bool {
        matches!(self.prune, ToolPrunePolicy::Allow)
    }

    pub(crate) fn counts_toward_precision_inline_budget(&self) -> bool {
        self.counts_toward_precision_inline_budget
    }
}

/// 可选的历史保留策略注册：只给需要偏离默认（`Allow`/`Allow`）的工具使用。
/// 未注册的工具沿用默认策略，不需要改动现有 `ToolSpec`，与
/// `ToolDisplayRegistration` / `ToolStreamingRegistration` 的兼容模式一致。
pub(crate) struct ToolHistoryPolicyRegistration {
    pub(crate) name: &'static str,
    pub(crate) policy: ToolHistoryPolicy,
}

inventory::collect!(ToolHistoryPolicyRegistration);

static TOOL_HISTORY_POLICY_INDEX: LazyLock<SkipMap<String, ToolHistoryPolicy>> =
    LazyLock::new(|| {
        let mut index: SkipMap<String, ToolHistoryPolicy> = SkipMap::default();
        for reg in inventory::iter::<ToolHistoryPolicyRegistration> {
            let name = reg.name.to_string();
            if !index.contains_key(&name) {
                index.insert(name, reg.policy);
            }
        }
        index
    });

/// 查询某个工具的历史保留策略；未注册的工具返回默认值（两维度均 `Allow`）。
pub(crate) fn tool_history_policy(name: &str) -> ToolHistoryPolicy {
    TOOL_HISTORY_POLICY_INDEX
        .get_ref(&name.to_string())
        .copied()
        .unwrap_or_default()
}

const TOOL_CANCEL_FUTEX_ENV: &str = "__ai_tool_cancel_futex_addr";

pub(crate) fn ensure_process_tool_cancel_futex(
    os: &mut dyn Kernel,
    pid: u64,
) -> Result<FutexAddr, String> {
    if let Some(addr) = os
        .get_process(pid)
        .and_then(|proc| proc.env.get(TOOL_CANCEL_FUTEX_ENV))
        .and_then(|raw| raw.parse::<u64>().ok())
        .map(FutexAddr)
    {
        if os.futex_load(addr).is_some() {
            return Ok(addr);
        }
    }
    let addr = os.futex_create(0, format!("tool_cancel:pid={pid}"));
    let Some(proc) = os.get_process_mut(pid) else {
        return Err(format!("process {pid} not found for tool cancel futex"));
    };
    proc.env
        .insert(TOOL_CANCEL_FUTEX_ENV.to_string(), addr.raw().to_string());
    Ok(addr)
}

pub(crate) fn current_process_tool_cancel_futex(
    os: &mut dyn Kernel,
) -> Result<Option<FutexAddr>, String> {
    let Some(pid) = os.current_process_id() else {
        return Ok(None);
    };
    ensure_process_tool_cancel_futex(os, pid).map(Some)
}

pub(crate) fn request_tool_cancel() {
    with_current_process_kernel(|os, pid| {
        let addr = ensure_process_tool_cancel_futex(os, pid)?;
        let _ = os.futex_store(addr, 1);
        os.signal_process(pid, Signal::SigCancel)?;
        Ok(())
    });
}

pub(crate) fn clear_tool_cancel() {
    with_current_process_kernel(|os, pid| {
        let addr = ensure_process_tool_cancel_futex(os, pid)?;
        let _ = os.futex_store(addr, 0);
        Ok(())
    });
    with_current_process_mut(|proc| {
        proc.pending_signals
            .retain(|signal| *signal != Signal::SigCancel);
    });
}

pub(crate) fn is_tool_cancel_requested() -> bool {
    with_current_process_ref(|proc| {
        proc.pending_signals
            .iter()
            .any(|signal| *signal == Signal::SigCancel)
    })
    .unwrap_or(false)
}

fn with_current_process<T>(
    f: impl FnOnce(&mut dyn aios_kernel::kernel::Syscall, u64) -> Result<T, String>,
) -> Option<T> {
    let guard = GLOBAL_OS.lock().ok()?;
    let os = guard.as_ref()?.clone();
    let mut os = os.lock().ok()?;
    let pid = os.current_process_id()?;
    f(os.as_mut(), pid).ok()
}

fn with_current_process_kernel<T>(
    f: impl FnOnce(&mut dyn Kernel, u64) -> Result<T, String>,
) -> Option<T> {
    let guard = GLOBAL_OS.lock().ok()?;
    let os = guard.as_ref()?.clone();
    let mut os = os.lock().ok()?;
    let pid = os.current_process_id()?;
    f(os.as_mut(), pid).ok()
}

fn with_current_process_mut(f: impl FnOnce(&mut aios_kernel::kernel::Process)) {
    let Ok(guard) = GLOBAL_OS.lock() else {
        return;
    };
    let Some(os) = guard.as_ref() else {
        return;
    };
    let Ok(mut os) = os.lock() else {
        return;
    };
    let Some(pid) = os.current_process_id() else {
        return;
    };
    if let Some(proc) = os.get_process_mut(pid) {
        f(proc);
    }
}

fn with_current_process_ref<T>(f: impl FnOnce(&aios_kernel::kernel::Process) -> T) -> Option<T> {
    let guard = GLOBAL_OS.lock().ok()?;
    let os = guard.as_ref()?.clone();
    let os = os.lock().ok()?;
    let pid = os.current_process_id()?;
    os.get_process(pid).map(f)
}

static TOOL_INDEX: LazyLock<SkipMap<String, &'static ToolSpec>> = LazyLock::new(|| {
    let mut index: SkipMap<String, &'static ToolSpec> = SkipMap::default();
    for reg in inventory::iter::<ToolRegistration> {
        let name = reg.spec.name.to_string();
        if !index.contains_key(&name) {
            index.insert(name, &reg.spec);
        }
    }
    index
});

static TOOL_STREAM_INDEX: LazyLock<SkipMap<String, ToolStreamExecutor>> = LazyLock::new(|| {
    let mut index: SkipMap<String, ToolStreamExecutor> = SkipMap::default();
    for reg in inventory::iter::<ToolStreamingRegistration> {
        let name = reg.name.to_string();
        if !index.contains_key(&name) {
            index.insert(name, reg.execute_streaming);
        }
    }
    index
});

static REGISTERED_TOOL_NAMES: LazyLock<FastSet<&'static str>> = LazyLock::new(|| {
    let mut names = FastSet::default();
    for reg in inventory::iter::<ToolRegistration> {
        names.insert(reg.spec.name);
    }
    names
});

fn expanded_tool_groups<'a>(groups: &'a [&'a str]) -> Vec<&'a str> {
    let mut expanded_groups: Vec<&str> = groups.to_vec();
    if groups.contains(&"executor") && !expanded_groups.contains(&"openclaw") {
        expanded_groups.push("openclaw");
    }
    if groups.contains(&"openclaw") && !expanded_groups.contains(&"executor") {
        expanded_groups.push("executor");
    }
    expanded_groups
}

/// Returns tool definitions for all registered tools that belong
/// to at least one of the specified groups.
pub(crate) fn tool_definitions_for_groups(groups: &[&str]) -> Vec<ToolDefinition> {
    let mut tools: Box<SkipMap<String, ToolDefinition>> =
        SkipMap::new(16, |a: &String, b: &String| a.cmp(b) as i32);
    let expanded_groups = expanded_tool_groups(groups);

    for reg in inventory::iter::<ToolRegistration> {
        if !reg
            .spec
            .groups
            .iter()
            .any(|g| expanded_groups.iter().any(|x| x == g))
        {
            continue;
        }
        let tool_def = ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: reg.spec.name.to_string(),
                description: reg.spec.description.to_string(),
                parameters: (reg.spec.parameters)(),
            },
        };
        tools.insert(tool_def.function.name.clone(), tool_def);
    }
    tools.into_iter().map(|(_, v)| v).collect()
}

pub(crate) fn tool_summaries_for_groups(groups: &[&str]) -> Vec<(String, String)> {
    let mut tools: Box<SkipMap<String, String>> =
        SkipMap::new(16, |a: &String, b: &String| a.cmp(b) as i32);
    let expanded_groups = expanded_tool_groups(groups);

    for reg in inventory::iter::<ToolRegistration> {
        if !reg
            .spec
            .groups
            .iter()
            .any(|g| expanded_groups.iter().any(|x| x == g))
        {
            continue;
        }
        tools.insert(reg.spec.name.to_string(), reg.spec.description.to_string());
    }

    tools.into_iter().collect()
}

pub(crate) fn get_tool_definitions_by_names(names: &[String]) -> Vec<ToolDefinition> {
    let mut tools: Box<SkipMap<String, ToolDefinition>> =
        SkipMap::new(16, |a: &String, b: &String| a.cmp(b) as i32);

    for name in names {
        let Some(spec) = TOOL_INDEX.get_ref(&name.to_string()).copied() else {
            continue;
        };
        let tool_def = ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: spec.name.to_string(),
                description: spec.description.to_string(),
                parameters: (spec.parameters)(),
            },
        };
        tools.insert(tool_def.function.name.clone(), tool_def);
    }
    tools.into_iter().map(|(_, v)| v).collect()
}

pub(crate) fn get_builtin_tool_definitions() -> Vec<ToolDefinition> {
    tool_definitions_for_groups(&["builtin"])
}

/// 判断一组 group 归属是否代表「按需加载的重执行原语」：隶属 executor/openclaw
/// 组但不属于 core 组。这类工具（进程 / IPC / 共享内存 / 环境原语）schema 体积大、
/// 使用频率低，默认不随每轮请求常驻，改由模型经 `enable_tools` 按需启用，压缩每轮
/// 发送的 tools schema token。core∩executor 的工具（如 apply_patch / write_file /
/// read_file）因同属 core 仍然常驻，不受影响。
fn groups_defer_eager_load(groups: &[&str]) -> bool {
    let in_executor = groups.contains(&"executor") || groups.contains(&"openclaw");
    in_executor && !groups.contains(&"core")
}

/// 某个已注册工具是否为「按需加载的重执行原语」。turn 级工具集在按 tool_groups
/// 展开时用它剔除这些工具；显式 `tools:` 点名的工具不走这里（点名即常驻）。
pub(crate) fn tool_defers_eager_load(name: &str) -> bool {
    TOOL_INDEX
        .get_ref(&name.to_string())
        .copied()
        .map(|spec| groups_defer_eager_load(spec.groups))
        .unwrap_or(false)
}

/// 全部「按需加载的重执行原语」（名称 + 描述），按名称排序。用于在 system prompt
/// 里生成「未加载但可 enable」的能力目录，保证模型对这些工具可感知、可按需启用。
pub(crate) fn deferred_eager_load_tool_summaries() -> Vec<(String, String)> {
    let mut tools: Box<SkipMap<String, String>> =
        SkipMap::new(16, |a: &String, b: &String| a.cmp(b) as i32);
    for reg in inventory::iter::<ToolRegistration> {
        if groups_defer_eager_load(reg.spec.groups) {
            tools.insert(reg.spec.name.to_string(), reg.spec.description.to_string());
        }
    }
    tools.into_iter().collect()
}

pub(crate) fn get_tool_spec(name: &str) -> Option<&'static ToolSpec> {
    TOOL_INDEX.get_ref(&name.to_string()).copied()
}

pub(crate) fn is_registered_tool_name(name: &str) -> bool {
    REGISTERED_TOOL_NAMES.contains(name)
}

/// 把已废弃/合并的历史工具名规整到当前规范名，用于旧会话回放兼容。
/// `read_file_lines` 已并入 `read_file`（两者都接受 offset/limit）。
/// `lsp` 已并入 `code_search`（后者是超集：同样的 operation/file_path 语义，
/// 6 个 LSP 操作内部委托给 `execute_lsp`）。
fn canonical_tool_name(name: &str) -> &str {
    match name {
        "read_file_lines" => "read_file",
        "lsp" => "code_search",
        other => other,
    }
}

/// Executes a tool call by parsing its arguments and dispatching
/// to the registered tool implementation.
pub(crate) fn execute_tool_call(tool_call: &ToolCall) -> Result<ToolResult, String> {
    let raw_args = tool_call.function.arguments.trim();
    let args: Value = if raw_args.is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(raw_args).map_err(|e| format!("Failed to parse arguments: {}", e))?
    };

    execute_tool_call_with_args(&tool_call.id, &tool_call.function.name, &args)
}

pub(crate) fn execute_tool_call_with_args(
    tool_call_id: &str,
    name: &str,
    args: &Value,
) -> Result<ToolResult, String> {
    execute_tool_call_with_args_impl(tool_call_id, name, args, None)
}

pub(crate) fn execute_tool_call_with_args_streaming(
    tool_call_id: &str,
    name: &str,
    args: &Value,
    on_chunk: &mut ToolStreamWriter<'_>,
) -> Result<ToolResult, String> {
    execute_tool_call_with_args_impl(tool_call_id, name, args, Some(on_chunk))
}

fn execute_tool_call_with_args_impl(
    tool_call_id: &str,
    name: &str,
    args: &Value,
    on_chunk: Option<&mut ToolStreamWriter<'_>>,
) -> Result<ToolResult, String> {
    // 旧会话回放兼容：read_file_lines 已并入 read_file（同样支持 offset/limit），
    // lsp 已并入 code_search（同样的 operation/file_path 语义）。老历史里残留的
    // 调用名映射到规范名，避免回放时命中 "Unknown tool"。
    let name = canonical_tool_name(name);
    let Some(spec) = TOOL_INDEX.get_ref(&name.to_string()).copied() else {
        record_tool_stat(name, false);
        record_tool_decision(name, false, "unknown_tool");
        return Err(format!("Unknown tool: {}", name));
    };
    let started = std::time::Instant::now();
    let exec = if let Some(stream_exec) = TOOL_STREAM_INDEX.get_ref(&name.to_string()).copied() {
        let mut sink = |_chunk: &[u8]| {};
        let writer = match on_chunk {
            Some(writer) => writer,
            None => &mut sink,
        };
        stream_exec(args, writer)
    } else {
        (spec.execute)(args)
    };
    let elapsed_ms = started.elapsed().as_millis() as u64;
    match exec {
        Ok(result) => {
            record_tool_stat(name, true);
            record_tool_decision_with_time(name, true, "ok", elapsed_ms);
            Ok(ToolResult {
                tool_call_id: tool_call_id.to_string(),
                content: result,
            })
        }
        Err(err) => {
            record_tool_stat(name, false);
            record_tool_decision_with_time(name, false, &err, elapsed_ms);
            Err(err)
        }
    }
}

/// Tier A1：把工具调用结果写进 DecisionLog（只写，下游消费另起 PR）。
fn record_tool_decision(name: &str, success: bool, message: &str) {
    record_tool_decision_with_time(name, success, message, 0);
}

fn record_tool_decision_with_time(name: &str, success: bool, message: &str, elapsed_ms: u64) {
    let store = crate::ai::driver::decision_log::get_decision_log_store();
    let session_id = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
    let turn_id = crate::ai::driver::runtime_ctx::current_turn_id_or_zero();
    store.log(crate::ai::driver::decision_log::DecisionLog {
        timestamp: 0,
        session_id,
        turn_id,
        decision_type: crate::ai::driver::decision_log::DecisionType::ToolInvocation,
        context: String::new(),
        alternatives_considered: Vec::new(),
        chosen_option: name.to_string(),
        reasoning: String::new(),
        confidence: None,
        outcome: Some(crate::ai::driver::decision_log::Outcome {
            success,
            message: {
                // 截断长错误，避免 DecisionLog 内存膨胀
                if message.len() > 240 {
                    let mut end = 240;
                    while !message.is_char_boundary(end) && end > 0 {
                        end -= 1;
                    }
                    format!("{}...", &message[..end])
                } else {
                    message.to_string()
                }
            },
            user_feedback: None,
        }),
        execution_time_ms: Some(elapsed_ms),
    });
}

fn record_tool_stat(name: &str, ok: bool) {
    let entry = AgentMemoryEntry {
        id: None,
        timestamp: Local::now().to_rfc3339(),
        category: "tool_stat".to_string(),
        note: format!("name={} result={}", name, if ok { "ok" } else { "err" }),
        tags: vec![
            name.to_string(),
            if ok {
                "ok".to_string()
            } else {
                "err".to_string()
            },
        ],
        source: None,
        priority: Some(50),
        owner_pid: None,
        owner_pgid: None,
        image_path: None,
    };
    let store = MemoryStore::from_env_or_config();
    let _ = store.append(&entry);
    store.maintain_after_append();
}

/// Executes a tool call with permission checking.
/// - If denied: returns an error immediately.
/// - If ask: prompts the user for confirmation before executing.
/// - If allowed: proceeds to execute directly.
pub(crate) fn execute_tool_call_with_permissions(
    tool_call: &ToolCall,
    permissions: &ToolPermissions,
) -> Result<ToolResult, String> {
    let tool_name = &tool_call.function.name;

    if permissions.is_denied(tool_name) {
        return Err(format!("Tool '{}' is denied by permissions", tool_name));
    }

    if permissions.needs_ask(tool_name) {
        let confirmed = crate::commonw::prompt::prompt_yes_or_no_interruptible(&format!(
            "Confirm tool execution: {} (y/n): ",
            tool_name
        ));
        if !confirmed.unwrap_or(false) {
            return Err(format!("Tool '{}' execution cancelled by user", tool_name));
        }
    }

    execute_tool_call(tool_call)
}

#[cfg(test)]
mod history_policy_tests {
    use super::*;

    #[test]
    fn plan_allows_lossy_compress_but_blocks_prune() {
        // plan 最新一版由最近工具组保护窗口 (`KEEP_RECENT_TOOL_GROUPS`) 完整保留；
        // 旧版 plan 触发上下文压力时允许有损压缩摘要。LLM 单方裁剪 prunes 始终禁止，
        // 防止模型"判废"既有规划而原地打转。
        let policy = tool_history_policy("plan");
        assert!(policy.allows_lossy_compress());
        assert!(!policy.allows_prune());
        assert!(!policy.counts_toward_precision_inline_budget());
    }

    #[test]
    fn legacy_read_file_lines_name_canonicalizes_to_read_file() {
        // 旧会话历史里残留的 read_file_lines 调用名必须映射到 read_file，
        // 回放时不能命中 "Unknown tool"。
        assert_eq!(canonical_tool_name("read_file_lines"), "read_file");
        assert_eq!(canonical_tool_name("lsp"), "code_search");
        assert_eq!(canonical_tool_name("read_file"), "read_file");
        assert_eq!(canonical_tool_name("execute_command"), "execute_command");
    }

    #[test]
    fn read_and_search_tools_block_lossy_but_allow_prune() {
        for name in ["read_file", "find_path", "code_search"] {
            let policy = tool_history_policy(name);
            assert!(
                !policy.allows_lossy_compress(),
                "{name} should block lossy compression"
            );
            assert!(policy.allows_prune(), "{name} should allow LLM pruning");
        }
    }

    #[test]
    fn execute_command_blocks_lossy_compression_but_allows_pruning() {
        let policy = tool_history_policy("execute_command");
        assert!(!policy.allows_lossy_compress());
        assert!(policy.allows_prune());
        assert!(policy.counts_toward_precision_inline_budget());
    }

    #[test]
    fn unregistered_tool_defaults_to_allow_both() {
        let policy = tool_history_policy("unregistered_tool");
        assert!(policy.allows_lossy_compress());
        assert!(policy.allows_prune());
    }
}
