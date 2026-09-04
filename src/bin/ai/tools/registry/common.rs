use std::sync::LazyLock;

use rust_tools::{commonw::FastSet, cw::SkipMap};
use serde_json::Value;

use crate::ai::tools::os_tools::GLOBAL_OS;
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use crate::ai::types::{FunctionDefinition, ToolCall, ToolDefinition, ToolResult};
use aios_kernel::{
    kernel::{Kernel, Signal},
    primitives::FutexAddr,
};
use chrono::Local;

#[derive(Clone, Copy)]
pub(crate) struct ToolSpec {
    pub(crate) name: &'static str,
    pub(crate) description: &'static str,
    pub(crate) execute: fn(&Value) -> Result<String, String>,
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

/// Optional streaming-execution registration: only for builtin tools that
/// genuinely need real-time terminal feedback. Unregistered tools keep the
/// original synchronous `execute` path; no change to the existing ToolSpec.
pub(crate) struct ToolStreamingRegistration {
    pub(crate) name: &'static str,
    pub(crate) execute_streaming: ToolStreamExecutor,
}

inventory::collect!(ToolStreamingRegistration);

/// Terminal echo configuration: controls whether a tool call's arguments /
/// output are printed to the terminal. Defaults to all `false`; only tools
/// with high user-visibility value (e.g. `plan`) opt in explicitly. Submitted
/// via a separate `ToolDisplayRegistration` without touching the existing
/// `ToolSpec`, keeping backward compatibility.

/// Terminal echo content transform: compresses the full tool result into
/// compact echo text. Signature `fn(full result, call args) -> echo text`;
/// `None` echoes the full result as-is.
pub(crate) type ToolDisplayTransform = fn(&str, &Value) -> String;

#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct ToolDisplayConfig {
    /// Whether to print tool call arguments to the terminal.
    pub(crate) print_args: bool,
    /// Whether to print tool output to the terminal.
    pub(crate) print_result: bool,
    /// Whether the result body is echoed at regular brightness; defaults to
    /// the low-priority `DIM`. Only for the few tool results that need slight
    /// emphasis without high-saturation color or bold.
    pub(crate) emphasize_result: bool,
    /// Optional terminal echo transform: the model still receives the full `content`; the terminal only echoes the transformed compact text.
    pub(crate) display: Option<ToolDisplayTransform>,
}

/// Optional terminal-echo registration: only for builtin tools that need to
/// echo arguments/results. Unregistered tools keep the default configuration
/// (nothing echoed); no change to the existing `ToolSpec`.
pub(crate) struct ToolDisplayRegistration {
    pub(crate) name: &'static str,
    pub(crate) config: ToolDisplayConfig,
}

inventory::collect!(ToolDisplayRegistration);

/// Optional same-args result reuse registration. Only tools whose results can
/// be treated as a stable snapshot within the current user turn and whose
/// repeated execution has no consuming side effects may register; unregistered
/// tools must actually execute by default.
pub(crate) struct ToolReplayRegistration {
    pub(crate) name: &'static str,
}

inventory::collect!(ToolReplayRegistration);

static TOOL_REPLAY_INDEX: LazyLock<SkipMap<String, ()>> = LazyLock::new(|| {
    let mut index: SkipMap<String, ()> = SkipMap::default();
    for reg in inventory::iter::<ToolReplayRegistration> {
        let name = reg.name.to_string();
        if !index.contains_key(&name) {
            index.insert(name, ());
        }
    }
    index
});

/// Whether a successful result with the same name and args may be reused directly within the current user turn.
pub(crate) fn tool_allows_same_turn_replay(name: &str) -> bool {
    TOOL_REPLAY_INDEX.contains_key(&name.to_string())
}

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

/// Query a tool's terminal echo configuration; unregistered tools return the all-`false` default.
pub(crate) fn tool_display_config(name: &str) -> ToolDisplayConfig {
    TOOL_DISPLAY_INDEX
        .get_ref(&name.to_string())
        .copied()
        .unwrap_or_default()
}

/// Lossy-compression policy: controls whether this tool's results may undergo
/// lossy compression such as line trimming / folding / summarization. `Never`
/// marks precision results; the compression path may only zero-compress them
/// out to disk, leaving a pointer stub.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) enum ToolLossyCompressPolicy {
    /// Default: lossy compression allowed (ordinary overview-type tool results).
    #[default]
    Allow,
    /// Lossy compression forbidden: content is expensive to reproduce (e.g.
    /// `read_file` / retrieval / `execute_command`); once trimmed, the model
    /// re-runs the same operation repeatedly, appearing amnesiac / stuck in place.
    Never,
}

/// LLM-guided pruning policy: controls whether this tool's results may be
/// marked by the model and pruned into placeholders. Orthogonal to lossy
/// compression — "cannot be lossily compressed" does not mean "cannot be
/// pruned": old versions of `read_file` should become prunable once the model
/// has repeatedly judged them outdated, to free context. `plan` allows lossy
/// compression but forbids LLM pruning: the latest version is fully preserved
/// by the recent-tool-group protection window, while old versions may be
/// summarized to free context, but the model should not unilaterally
/// invalidate existing plans.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) enum ToolPrunePolicy {
    /// Default: may be pruned under LLM guidance (still subject to recent-window protection and the consecutive-mark threshold).
    #[default]
    Allow,
    /// Never pruned (e.g. `plan`).
    Never,
}

/// Tool history retention policy: declares both orthogonal dimensions, "lossy
/// compression" and "LLM pruning", together. Submitted via a separate
/// `ToolHistoryPolicyRegistration` without touching `ToolSpec`; unregistered
/// tools take the defaults (both dimensions `Allow`).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct ToolHistoryPolicy {
    pub(crate) lossy_compress: ToolLossyCompressPolicy,
    pub(crate) prune: ToolPrunePolicy,
    /// When the recent tool group is oversized, whether this tool occupies the
    /// inline budget for high-precision results. Aggregation tools (e.g.
    /// task_wait) do not occupy this budget even though they likewise forbid
    /// lossy compression.
    pub(crate) counts_toward_precision_inline_budget: bool,
}

impl ToolHistoryPolicy {
    /// Whether lossy compression (line trimming / folding / summarization) may be applied to this tool's results.
    pub(crate) fn allows_lossy_compress(&self) -> bool {
        matches!(self.lossy_compress, ToolLossyCompressPolicy::Allow)
    }

    /// Whether this tool's results may be pruned under LLM guidance.
    pub(crate) fn allows_prune(&self) -> bool {
        matches!(self.prune, ToolPrunePolicy::Allow)
    }

    pub(crate) fn counts_toward_precision_inline_budget(&self) -> bool {
        self.counts_toward_precision_inline_budget
    }
}

/// Optional history-retention-policy registration: only for tools that
/// deviate from the default (`Allow`/`Allow`). Unregistered tools keep the
/// default policy, no change to the existing `ToolSpec` needed, consistent
/// with the compatibility mode of `ToolDisplayRegistration` /
/// `ToolStreamingRegistration`.
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

/// Query a tool's history retention policy; unregistered tools return the default (`Allow` for both dimensions).
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

/// The SIGINT handler path must not block on the kernel mutex. The
/// stream/request cancel flag is the primary cancellation signal; this is only
/// a best-effort side channel for tools currently polling SigCancel.
pub(crate) fn try_request_tool_cancel() -> bool {
    try_with_current_process_kernel(|os, pid| {
        let addr = ensure_process_tool_cancel_futex(os, pid)?;
        let _ = os.futex_store(addr, 1);
        os.signal_process(pid, Signal::SigCancel)?;
        Ok(())
    })
    .is_some()
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

fn try_with_current_process_kernel<T>(
    f: impl FnOnce(&mut dyn Kernel, u64) -> Result<T, String>,
) -> Option<T> {
    let guard = GLOBAL_OS.try_lock().ok()?;
    let os = guard.as_ref()?.clone();
    drop(guard);
    let mut os = os.try_lock().ok()?;
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

/// Returns tool definitions for all registered tools that belong
/// to at least one of the specified groups.
pub(crate) fn tool_definitions_for_groups(
    groups: &[super::tool_groups::ToolGroup],
) -> Vec<ToolDefinition> {
    let mut tools: Box<SkipMap<String, ToolDefinition>> =
        SkipMap::new(16, |a: &String, b: &String| a.cmp(b) as i32);

    for reg in inventory::iter::<ToolRegistration> {
        let tags = super::tool_metadata::tool_groups(reg.spec.name);
        if !tags.iter().any(|g| groups.contains(g)) {
            continue;
        }
        let tool_def = ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: reg.spec.name.to_string(),
                description: super::tool_metadata::tool_description(
                    reg.spec.name,
                    reg.spec.description,
                ),
                parameters: super::tool_metadata::tool_parameters(reg.spec.name),
            },
        };
        tools.insert(tool_def.function.name.clone(), tool_def);
    }
    tools.into_iter().map(|(_, v)| v).collect()
}

pub(crate) fn tool_summaries_for_groups(
    groups: &[super::tool_groups::ToolGroup],
) -> Vec<(String, String)> {
    let mut tools: Box<SkipMap<String, String>> =
        SkipMap::new(16, |a: &String, b: &String| a.cmp(b) as i32);

    for reg in inventory::iter::<ToolRegistration> {
        let tags = super::tool_metadata::tool_groups(reg.spec.name);
        if !tags.iter().any(|g| groups.contains(g)) {
            continue;
        }
        tools.insert(
            reg.spec.name.to_string(),
            super::tool_metadata::tool_description(reg.spec.name, reg.spec.description),
        );
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
                description: super::tool_metadata::tool_description(spec.name, spec.description),
                parameters: super::tool_metadata::tool_parameters(spec.name),
            },
        };
        tools.insert(tool_def.function.name.clone(), tool_def);
    }
    tools.into_iter().map(|(_, v)| v).collect()
}

pub(crate) fn get_builtin_tool_definitions() -> Vec<ToolDefinition> {
    tool_definitions_for_groups(&[super::tool_groups::ToolGroup::Builtin])
}

/// Whether a registered tool is a "deferred eager-load heavy execution
/// primitive", i.e. it carries the `hidden` metadata flag. These tools
/// (process / IPC / shared-memory / environment primitives) have large schemas
/// and low usage, so they are not resident in each turn's request; the model
/// enables them on demand via `enable_tools`, shrinking per-turn tools-schema
/// tokens. The same flag also excludes them from the default catalog
/// (`tool_metadata::tool_is_hidden`), so `hidden` is the single source of
/// truth for both eager-load deferral and default-agent visibility.
pub(crate) fn tool_defers_eager_load(name: &str) -> bool {
    super::tool_metadata::tool_is_hidden(name)
}

/// All deferred eager-load primitives (name + description), sorted by name.
/// Feeds the "loaded on demand" capability catalog in the system prompt so the
/// model stays aware of these tools and can enable them when needed.
pub(crate) fn deferred_eager_load_tool_summaries() -> Vec<(String, String)> {
    let mut tools: Box<SkipMap<String, String>> =
        SkipMap::new(16, |a: &String, b: &String| a.cmp(b) as i32);
    for reg in inventory::iter::<ToolRegistration> {
        if super::tool_metadata::tool_is_hidden(reg.spec.name) {
            tools.insert(
                reg.spec.name.to_string(),
                super::tool_metadata::tool_description(reg.spec.name, reg.spec.description),
            );
        }
    }
    tools.into_iter().collect()
}

/// Whether a turn-group gates hidden tools: at least one registered tool that
/// belongs to `group` carries the `hidden` metadata flag. Agents that declare
/// such a group (e.g. the executor group in a skill/agent manifest) may see
/// and enable the group's hidden primitives; everyone else stays blocked. This
/// is the single source of truth for "privileged group" — tagging a new tool
/// hidden automatically makes its group privileged, with no hardcoded
/// group-name list.
pub(crate) fn group_gates_hidden_tools(group: super::tool_groups::ToolGroup) -> bool {
    inventory::iter::<ToolRegistration>.into_iter().any(|reg| {
        super::tool_metadata::tool_is_hidden(reg.spec.name)
            && super::tool_metadata::tool_groups(reg.spec.name).contains(&group)
    })
}

pub(crate) fn get_tool_spec(name: &str) -> Option<&'static ToolSpec> {
    TOOL_INDEX.get_ref(&name.to_string()).copied()
}

pub(crate) fn is_registered_tool_name(name: &str) -> bool {
    REGISTERED_TOOL_NAMES.contains(name)
}

/// Returns the skill/agent manifest-declared tool names that match no
/// registered builtin tool. `mcp_*` entries are excluded deliberately: MCP
/// availability is resolved against live servers at turn time (`select_mcp_tools`
/// / `enable_tools`), not against this static registry, so an `mcp_*` prefix can
/// be valid even though this process has never seen the underlying server.
pub(crate) fn unknown_manifest_tool_names(tool_names: &[String]) -> Vec<String> {
    tool_names
        .iter()
        .filter(|name| !name.starts_with("mcp_") && !is_registered_tool_name(name))
        .cloned()
        .collect()
}

/// Comma-joined form of [`unknown_manifest_tool_names`] for load-time warnings,
/// or `None` when every declared name resolves.
pub(crate) fn manifest_unknown_tool_names_warning(tool_names: &[String]) -> Option<String> {
    let unknown = unknown_manifest_tool_names(tool_names);
    (!unknown.is_empty()).then(|| unknown.join(", "))
}

#[cfg(test)]
mod manifest_tool_name_tests {
    use super::*;

    #[test]
    fn known_and_mcp_prefixed_names_pass_typo_is_flagged() {
        assert_eq!(
            unknown_manifest_tool_names(&[
                "read_file".to_string(),
                "read_fil".to_string(),
                "".to_string()
            ]),
            vec!["read_fil".to_string(), "".to_string()]
        );
        assert_eq!(
            unknown_manifest_tool_names(&["mcp_browser_navigate".to_string()]),
            Vec::<String>::new()
        );
    }

    #[test]
    fn warning_is_none_when_all_resolved() {
        assert_eq!(
            manifest_unknown_tool_names_warning(&["apply_patch".to_string()]),
            None
        );
        assert_eq!(
            manifest_unknown_tool_names_warning(&["nope".to_string(), "also_nope".to_string()]),
            Some("nope, also_nope".to_string())
        );
    }
}

/// Normalize deprecated/merged historical tool names to their current
/// canonical names, for old-session replay compatibility. `read_file_lines`
/// has been merged into `read_file` (both accept offset/limit).
fn canonical_tool_name(name: &str) -> &str {
    match name {
        "read_file_lines" => "read_file",
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
    // Old-session replay compatibility: read_file_lines was merged into
    // read_file (also supports offset/limit). Legacy call names left over in
    // old history map to the canonical name to avoid hitting "Unknown tool"
    // during replay.
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

/// Tier A1: write tool call results into the DecisionLog (write-only; downstream consumption is a separate PR).
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
                // Truncate long errors to avoid DecisionLog memory bloat
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

#[cfg(test)]
mod history_policy_tests {
    use super::*;

    #[test]
    fn plan_allows_lossy_compress_but_blocks_prune() {
        // The latest plan version is fully preserved by the recent-tool-group
        // protection window (`KEEP_RECENT_TOOL_GROUPS`); older plan versions
        // may be lossily summarized under context pressure. LLM-driven pruning
        // of plan results is always forbidden, to prevent the model from
        // unilaterally invalidating existing plans and spinning in place.
        let policy = tool_history_policy("plan");
        assert!(policy.allows_lossy_compress());
        assert!(!policy.allows_prune());
        assert!(!policy.counts_toward_precision_inline_budget());
    }

    #[test]
    fn legacy_read_file_lines_name_canonicalizes_to_read_file() {
        // The read_file_lines call name left over in old-session history must
        // map to read_file; replay must not hit "Unknown tool".
        assert_eq!(canonical_tool_name("read_file_lines"), "read_file");
        assert_eq!(canonical_tool_name("lsp"), "lsp");
        assert_eq!(canonical_tool_name("read_file"), "read_file");
        assert_eq!(canonical_tool_name("execute_command"), "execute_command");
    }

    #[test]
    fn read_file_blocks_lossy_but_allows_prune() {
        let policy = tool_history_policy("read_file");
        assert!(!policy.allows_lossy_compress());
        assert!(policy.allows_prune());
    }

    #[test]
    fn execute_command_blocks_lossy_compression_but_allows_pruning() {
        let policy = tool_history_policy("execute_command");
        assert!(!policy.allows_lossy_compress());
        assert!(policy.allows_prune());
        assert!(policy.counts_toward_precision_inline_budget());
    }

    #[test]
    fn apply_patch_blocks_lossy_compression_and_counts_toward_precision() {
        // apply_patch results are the only precise evidence of "what I just
        // changed" in the current turn; failure diagnostics also echo the full
        // file text for rebuilding the patch. Missing registration would fall
        // back to the default (Allow/Allow, precision=false), so the current
        // turn's patch result would neither enter the protection set nor be
        // safe from lossy trimming — the model could not see whether the patch
        // landed and would retry in place. This assertion pins that contract,
        // preventing another regression to defaults.
        let policy = tool_history_policy("apply_patch");
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

    #[test]
    fn subagent_spawn_tools_block_lossy_and_prune() {
        // task_spawn / task_spawn_batch arguments (subagent prompt / response
        // schema) and return values (task_id list) are required inputs for
        // later wait/status/integrate. Missing registration would fall back to
        // the default policy (Allow/Allow); after folding, evidence degrades
        // to the first characters of the result (the original args do not
        // participate in recall), and the main agent loses grounding on
        // already-spawned subtasks. This assertion pins the contract.
        for name in ["task_spawn", "task_spawn_batch"] {
            let policy = tool_history_policy(name);
            assert!(
                !policy.allows_lossy_compress(),
                "{name} must block lossy compress"
            );
            assert!(!policy.allows_prune(), "{name} must block prune");
        }
    }
}
