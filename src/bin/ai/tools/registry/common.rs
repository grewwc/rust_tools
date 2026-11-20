use std::sync::LazyLock;

use rust_tools::{
    commonw::{FastMap, FastSet},
    cw::SkipMap,
};
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

/// Registry entry submitted via `inventory::submit!` to register a tool
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


/// Terminal echo content transform: compresses the full tool result into
/// compact echo text. Signature `fn(full result, call args) -> echo text`;
/// `None` echoes the full result as-is.
pub(crate) type ToolDisplayTransform = fn(&str, &Value) -> String;

/// Inline status-target extractor: derives the short target shown after the
/// tool name on the terminal status line (file path, skill name, ...) from
/// the call arguments. Signature `fn(call args) -> Option<target text>`;
/// `None` means the tool carries no inline target.
pub(crate) type ToolStatusTarget = fn(&Value) -> Option<String>;

/// Controls terminal echo of call arguments and results. Tools opt in through
/// `ToolDisplayRegistration`; the default echoes neither.
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
    /// Optional inline status target shown on both the running row and the
    /// completion row. Defaults to `None` (no target printed); only tools
    /// that register an extractor get the `· target` prompt.
    pub(crate) status_target: Option<ToolStatusTarget>,
}

/// Keep only the last two path segments (directory + file name), truncating over-long paths.
pub(crate) fn short_path(p: &str) -> String {
    // Prefer the last two segments: /a/b/c/foo.rs → c/foo.rs
    let parts: Vec<&str> = p.split('/').filter(|s| !s.is_empty()).collect();
    let short = if parts.len() >= 2 {
        format!("{}/{}", parts[parts.len() - 2], parts[parts.len() - 1])
    } else {
        p.to_string()
    };
    let char_count = short.chars().count();
    if char_count > 50 {
        let byte_idx = short
            .char_indices()
            .nth(char_count - 48)
            .map(|(i, _)| i)
            .unwrap_or(0);
        format!("…{}", &short[byte_idx..])
    } else {
        short
    }
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
    /// Prunable only once the durable subagent evidence ledger records every task
    /// the result refers to as integrated (see `PruneAuthorization`).
    ///
    /// A delivered subagent result is live state until the parent integrates its
    /// conclusion: the parent may still quote, cross-check, or re-read it
    /// verbatim. Integration moves it into history, where the ledger keeps the
    /// byte-identical payload retrievable through `task_evidence_read`. Callers
    /// that cannot read the ledger fail closed rather than treat this as `Allow`.
    AfterIntegration,
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

    /// Whether this tool's results may be pruned under LLM guidance *without
    /// consulting session state*. [`ToolPrunePolicy::AfterIntegration`] is
    /// deliberately excluded: callers that cannot read the evidence ledger must
    /// fail closed. Use [`ToolHistoryPolicy::prune_policy`] when they can.
    pub(crate) fn allows_prune(&self) -> bool {
        matches!(self.prune, ToolPrunePolicy::Allow)
    }

    /// The declared prune policy, for callers that can consult session state.
    pub(crate) fn prune_policy(&self) -> ToolPrunePolicy {
        self.prune
    }

    /// Whether a stub created by an earlier offload keeps its mark.
    ///
    /// This states how the stub was created, not current eligibility: an
    /// `AfterIntegration` result reached a stub only while authorized, so
    /// re-deriving authorization from a stub (which no longer carries the task
    /// markers of the original text) must not drop the mark and make the offload
    /// flap. `Never` tools can never reach a stub through this path.
    pub(crate) fn retains_stub_mark(&self) -> bool {
        !matches!(self.prune, ToolPrunePolicy::Never)
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

/// Whether a builtin tool may run in a parallel batch with other read-only
/// calls. Registered explicitly: tool names do not encode side effects, and a
/// substring such as `get` in `forget` must not make a deleting tool parallel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ToolParallelPolicy {
    /// Run through the sequential path. This is the fail-closed default.
    #[default]
    Serial,
    /// Pure read-only work that is independent of other parallel calls.
    ReadOnly,
}

/// Optional parallel-execution-policy registration. Co-located with each
/// tool's registration so side effects and concurrency are reviewed together;
/// unregistered tools stay serial.
pub(crate) struct ToolParallelPolicyRegistration {
    pub(crate) name: &'static str,
    pub(crate) policy: ToolParallelPolicy,
}

inventory::collect!(ToolParallelPolicyRegistration);

static TOOL_PARALLEL_POLICY_INDEX: LazyLock<SkipMap<String, ToolParallelPolicy>> =
    LazyLock::new(|| {
        let mut index: SkipMap<String, ToolParallelPolicy> = SkipMap::default();
        for reg in inventory::iter::<ToolParallelPolicyRegistration> {
            let name = reg.name.to_string();
            if !index.contains_key(&name) {
                index.insert(name, reg.policy);
            }
        }
        index
    });

/// Query a tool's parallel-execution policy; unregistered tools are serial.
pub(crate) fn tool_parallel_policy(name: &str) -> ToolParallelPolicy {
    TOOL_PARALLEL_POLICY_INDEX
        .get_ref(&name.to_string())
        .copied()
        .unwrap_or_default()
}

/// Whether a builtin tool may be retried once after a transient error.
///
/// Retry safety is not the same dimension as parallel safety: `read_file` is
/// serial for evidence precision but idempotent, while `read_mailbox` is serial
/// because it drains state and must never be replayed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ToolRetryPolicy {
    /// Do not retry. This is the fail-closed default for mutating or
    /// non-idempotent tools.
    #[default]
    NoRetry,
    /// Idempotent read-only work that is safe to execute twice.
    RetrySafe,
}

/// Optional retry-policy registration. Co-located with each tool's
/// registration so side effects and retry semantics are reviewed together;
/// unregistered tools are never retried.
pub(crate) struct ToolRetryPolicyRegistration {
    pub(crate) name: &'static str,
    pub(crate) policy: ToolRetryPolicy,
}

inventory::collect!(ToolRetryPolicyRegistration);

static TOOL_RETRY_POLICY_INDEX: LazyLock<SkipMap<String, ToolRetryPolicy>> = LazyLock::new(|| {
    let mut index: SkipMap<String, ToolRetryPolicy> = SkipMap::default();
    for reg in inventory::iter::<ToolRetryPolicyRegistration> {
        let name = reg.name.to_string();
        if !index.contains_key(&name) {
            index.insert(name, reg.policy);
        }
    }
    index
});

/// Query a tool's retry policy; unregistered tools are never retried.
pub(crate) fn tool_retry_policy(name: &str) -> ToolRetryPolicy {
    TOOL_RETRY_POLICY_INDEX
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

static AGENT_TEAM_SCHEMA_EXTENSIONS: LazyLock<
    FastMap<&'static str, serde_json::Map<String, Value>>,
> = LazyLock::new(|| {
    [
        (
            "task_evidence_read",
            include_str!("../../tool_descriptions/task_evidence_read.json"),
        ),
        (
            "task_status",
            include_str!("../../tool_descriptions/task_status.json"),
        ),
        (
            "manage_team",
            include_str!("../../tool_descriptions/manage_team.json"),
        ),
        (
            "run_agent_graph",
            include_str!("../../tool_descriptions/run_agent_graph.json"),
        ),
    ]
    .into_iter()
    .map(|(name, content)| {
        let metadata: Value = serde_json::from_str(content).expect("embedded tool metadata");
        let properties = metadata["agent_team_optimization_properties"]
            .as_object()
            .expect("embedded agent-team schema extensions")
            .clone();
        (name, properties)
    })
    .collect()
});

/// The declarative base schema includes explicit reads for every metadata consumer.
/// Request builders add opt-in properties using one configuration snapshot per batch.
pub(crate) fn tool_parameters_with_optimizations(name: &str, enabled: bool) -> Value {
    let mut parameters = super::tool_metadata::tool_parameters(name);
    if !enabled {
        return parameters;
    }
    let Some(extensions) = AGENT_TEAM_SCHEMA_EXTENSIONS.get(name) else {
        return parameters;
    };
    if let Some(properties) = parameters
        .get_mut("properties")
        .and_then(Value::as_object_mut)
    {
        for (field, schema) in extensions {
            properties
                .entry(field.clone())
                .or_insert_with(|| schema.clone());
        }
    }
    parameters
}

/// Returns tool definitions for all registered tools that belong
/// to at least one of the specified groups.
pub(crate) fn tool_definitions_for_groups(
    groups: &[super::tool_groups::ToolGroup],
) -> Vec<ToolDefinition> {
    let optimizations = crate::ai::config::agent_team_optimizations();
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
                parameters: tool_parameters_with_optimizations(
                    reg.spec.name,
                    optimizations.optimizations_enabled(),
                ),
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
    let optimizations = crate::ai::config::agent_team_optimizations();
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
                parameters: tool_parameters_with_optimizations(
                    spec.name,
                    optimizations.optimizations_enabled(),
                ),
            },
        };
        tools.insert(tool_def.function.name.clone(), tool_def);
    }
    tools.into_iter().map(|(_, v)| v).collect()
}

pub(crate) fn get_builtin_tool_definitions() -> Vec<ToolDefinition> {
    tool_definitions_for_groups(&[super::tool_groups::ToolGroup::Builtin])
}

/// Whether `hidden` metadata defers a tool's eager schema loading. Enabling a
/// hidden tool also requires its manifest-declared gating group; see
/// `agent_may_enable_tool` in `tools/enable_tools.rs`.
pub(crate) fn tool_defers_eager_load(name: &str) -> bool {
    super::tool_metadata::tool_is_hidden(name)
}

/// Schema residency for the opt-in lazy orchestration group. This does not
/// change hidden-tool visibility or the privileged-group gate.
pub(crate) fn tool_has_lazy_group_schema(name: &str, lazy_group_schemas: bool) -> bool {
    lazy_group_schemas
        && super::tool_metadata::tool_groups(name)
            .contains(&super::tool_groups::ToolGroup::AgentTeam)
}

/// Task-group members used after spawning. Explicit name-based loads still
/// work before a spawn, including reads of previously delivered evidence.
pub(crate) fn is_task_lifecycle_tool(name: &str) -> bool {
    name != "task_spawn"
        && super::tool_metadata::tool_groups(name).contains(&super::tool_groups::ToolGroup::Task)
}

pub(crate) fn task_lifecycle_tool_names() -> Vec<String> {
    let mut names = inventory::iter::<ToolRegistration>
        .into_iter()
        .filter(|reg| is_task_lifecycle_tool(reg.spec.name))
        .map(|reg| reg.spec.name.to_string())
        .collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    names
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

    const HEAD_AGENT_TEAM_DEFINITION_BYTES: &[(&str, &str)] = &[
        (
            "task_evidence_read",
            r##"{"type":"function","function":{"name":"task_evidence_read","description":"Read evidence for one task. source=progress (default) returns its live phase/checkpoint snapshot. source=delivered is read-only: with agent-team optimizations enabled, defaults to a summary of at most 2 KiB including payload preview, verification availability, and advisory assessment; otherwise defaults to raw. Full persisted payload: task_evidence_read(source=\"delivered\", view=\"raw\", exact task_id), omitting fields/max_chars/cursor; fields selects metadata and cursor/max_chars page it. Explicit parameters work regardless of the optimization switch. Top-level verification is normalized and assessment is runtime-derived advisory requiring parent review; body-authored assessment values remain unverified worker content in payload/output; limits remain 8 KiB and 16 checks. Reading never collects or integrates. Use task_audit for recorded IDs.","parameters":{"type":"object","properties":{"task_id":{"type":"string","description":"Exact task_id returned by task_spawn or listed by task_audit. Delivered lookup preserves the ID verbatim."},"source":{"type":"string","enum":["progress","delivered"],"default":"progress","description":"progress preserves the existing live snapshot behavior; delivered performs a read-only lookup of a previously delivered result."},"fields":{"type":"array","minItems":1,"items":{"type":"string","minLength":1},"description":"Select top-level evidence fields, including verification and assessment when available. Unavailable names list available fields. With pagination, payload is serialized selected JSON; repeat identical fields for continuation. Explicit selection works with optimizations disabled."},"max_chars":{"type":"integer","minimum":0,"description":"Limit raw payload or serialized selected JSON per page in Unicode scalar values. Metadata is outside this limit; verification/assessment remain visible. Returns cursor, shown_chars, total_chars, truncated, next_cursor, and a recovery hint. 0 returns metadata and an empty payload; increase max_chars to advance. Explicit limits work with optimizations disabled."},"view":{"type":"string","enum":["summary","raw"],"description":"Delivered view: summary returns at most 2 KiB with preview and metadata discovery; raw returns the full original envelope when fields, max_chars, and cursor are omitted. Default is summary with agent-team optimizations enabled, raw otherwise. Projection parameters imply raw; explicit summary cannot be combined with them. Progress supports raw only."},"cursor":{"type":"integer","minimum":0,"description":"Raw page start as a Unicode scalar offset, default 0. Pass next_cursor and identical fields to continue. Without max_chars, a cursor requests a 2048-character page. Cursor equal to total_chars returns an empty complete page; larger offsets error. Full evidence: task_evidence_read(source=\"delivered\", view=\"raw\", exact task_id), omitting fields/max_chars/cursor."}},"required":["task_id"]}}}"##,
        ),
        (
            "task_status",
            r##"{"type":"function","function":{"name":"task_status","description":"Show status of all asynchronously spawned tasks: full task_id, agent, model, current state (ready/running/waiting/sleeping/stopped/terminated/completed/failed/timeout/cancelled), and each task's latest live progress/checkpoint snapshot, without blocking. Finished tasks' output is included inline and collected immediately, so you can use the result and integrate its full task_id without calling task_wait or task_audit.","parameters":{"type":"object","properties":{}}}}"##,
        ),
        (
            "manage_team",
            r##"{"type":"function","function":{"name":"manage_team","description":"Create and operate a checkpointed agent team with a shared task board. Create a team once, then add, claim, dispatch, advance, complete, message, inspect, or cancel its dependency-aware tasks.","parameters":{"type":"object","required":["action"],"properties":{"action":{"type":"string","enum":["create","add_task","claim","dispatch","advance","complete","send_message","status","cancel"],"description":"Operation. create requires name, goal, and members; add_task requires team_id and task; claim requires team_id and member_id; complete requires team_id, task_id, member_id, and status; send_message requires team_id, from, to, and body."},"team_id":{"type":"string","description":"Team UUID returned by create; required for every non-create action.","pattern":"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"},"name":{"type":"string","minLength":1,"description":"Non-empty team name for create."},"goal":{"type":"string","minLength":1,"description":"Non-empty team goal for create."},"members":{"type":"array","minItems":1,"maxItems":32,"description":"Member definitions for create (1 to 32 required when action is create).","items":{"type":"object","required":["id","role"],"properties":{"id":{"type":"string","pattern":"^[A-Za-z0-9_.-]{1,96}$","maxLength":96,"description":"Unique member identifier within the team."},"role":{"type":"string","minLength":1,"description":"Member responsibility or specialty."},"agent":{"type":"string","description":"Optional agent profile name for dispatched work."},"model":{"type":"string","description":"Optional model override for dispatched work."},"inherit":{"type":"string","description":"Context inherited by dispatched work: none, all, cwd, memory, history, skills, or a comma-separated combination. Default cwd,skills."},"capabilities":{"type":"array","description":"Optional capability labels assigned to the member.","items":{"type":"string"}}}}},"budget":{"type":"object","description":"Optional create-time team limits; omitted fields use their stated defaults.","properties":{"max_parallel":{"type":"integer","minimum":1,"maximum":8,"description":"Maximum concurrently dispatched tasks. Default 4."},"max_tasks":{"type":"integer","minimum":1,"maximum":512,"description":"Maximum tasks on the team board. Default 128."},"max_total_attempts":{"type":"integer","minimum":1,"maximum":4096,"description":"Total spawn attempts across the team. Default 256; must be >= max_tasks."},"max_messages":{"type":"integer","minimum":1,"maximum":2048,"description":"Maximum retained team messages. Default 512."}}},"task":{"type":"object","description":"Task definition for add_task.","required":["title","prompt"],"properties":{"id":{"type":"string","pattern":"^[A-Za-z0-9_.-]{1,96}$","maxLength":96,"description":"Optional unique task identifier; generated when omitted."},"title":{"type":"string","minLength":1,"description":"Short task title."},"prompt":{"type":"string","minLength":1,"maxLength":64000,"description":"Instructions given to the member that executes the task."},"assignee":{"type":"string","description":"Optional member id restricting who may claim or receive the task."},"depends_on":{"type":"array","description":"Task ids that must complete before this task becomes ready.","items":{"type":"string"}}}},"task_id":{"type":"string","pattern":"^[A-Za-z0-9_.-]{1,96}$","maxLength":96,"description":"Target task id. Optional for claim; required for complete."},"member_id":{"type":"string","pattern":"^[A-Za-z0-9_.-]{1,96}$","maxLength":96,"description":"Member claiming or completing the task."},"lease_secs":{"type":"integer","minimum":1,"maximum":86400,"description":"Claim lease in seconds. Default 300; clamped to 1..=86400."},"status":{"type":"string","enum":["completed","failed","cancelled"],"description":"Terminal task state for complete."},"output":{"type":"string","description":"Optional successful result recorded by complete."},"error":{"type":"string","description":"Optional failure or cancellation detail recorded by complete."},"from":{"type":"string","description":"Message sender: parent or a member id."},"to":{"type":"string","description":"Message recipient: parent, a member id, or * for broadcast."},"body":{"type":"string","minLength":1,"maxLength":16000,"description":"Non-empty message body for send_message."},"reason":{"type":"string","description":"Optional cancel reason recorded on cancelled tasks. Default team cancelled by parent."},"dispatch":{"type":"boolean","description":"For advance, dispatch newly ready tasks after collecting terminal work. Default true."}}}}}"##,
        ),
        (
            "run_agent_graph",
            r##"{"type":"function","function":{"name":"run_agent_graph","description":"Start, advance, inspect, or cancel a checkpointed agent graph. Static mode runs a conditional DAG and reduces node outputs into shared state. Dynamic mode selects relevant candidates, propagates messages, and pools a final answer.","parameters":{"type":"object","required":["action"],"properties":{"action":{"type":"string","enum":["start","advance","status","cancel"],"description":"Operation. start requires mode, question, and `static_graph` or `dynamic_graph` matching mode; advance, status, and cancel require graph_id. Call advance until the graph is terminal."},"graph_id":{"type":"string","description":"Graph UUID returned by start; required for advance, status, and cancel."},"mode":{"type":"string","enum":["static","dynamic"],"description":"Graph strategy for start: static runs static_graph; dynamic runs dynamic_graph."},"name":{"type":"string","description":"Optional graph name for start. Default agent-graph."},"question":{"type":"string","description":"Non-empty problem or objective given to the graph."},"max_parallel":{"type":"integer","minimum":1,"maximum":8,"description":"Maximum concurrent graph tasks. Default 4; clamped to 1..=8."},"static_graph":{"type":"object","description":"Static-mode definition: shared initial state, executable nodes, and conditional dependency edges.","required":["nodes"],"properties":{"initial_state":{"type":"object","description":"Initial shared JSON state. Defaults to an empty object."},"nodes":{"type":"array","maxItems":128,"description":"DAG nodes to execute (maximum 128). Node ids must be unique.","items":{"type":"object","required":["id","description","prompt"],"properties":{"id":{"type":"string","description":"Unique node identifier."},"description":{"type":"string","description":"Short explanation of the node's responsibility."},"prompt":{"type":"string","description":"Instructions executed by this node."},"agent":{"type":"string","description":"Optional agent profile name for the node task."},"model":{"type":"string","description":"Optional model override for the node task."},"inherit":{"type":"string","description":"Context inherited by the node task: none, all, cwd, memory, history, skills, or a comma-separated combination. Default cwd,skills."},"response_schema":{"type":"object","description":"Optional JSON Schema constraining the node's final output."},"output_key":{"type":"string","description":"Shared-state key receiving the reduced node output. Defaults to the node id."},"reducer":{"type":"string","enum":["replace","append_array","merge_object","concat"],"description":"How to combine this output with the current output_key value. Default replace."},"max_retries":{"type":"integer","minimum":0,"description":"Maximum retries after a failed node attempt. Default 0."}}}},"edges":{"type":"array","maxItems":1024,"description":"Directed dependencies between nodes (maximum 1024). Defaults to no edges.","items":{"type":"object","required":["from","to"],"properties":{"from":{"type":"string","description":"Predecessor node id."},"to":{"type":"string","description":"Successor node id."},"condition":{"type":"object","description":"Optional gate. Omit for always; on_success runs only after a completed predecessor, on_failure only after a failed predecessor, output_contains requires text, and json_pointer_equals requires pointer and value.","required":["kind"],"properties":{"kind":{"type":"string","enum":["always","on_success","on_failure","output_contains","json_pointer_equals"],"description":"Gate type."},"text":{"type":"string","description":"Required when kind is output_contains: substring in the predecessor output."},"pointer":{"type":"string","description":"Required when kind is json_pointer_equals: JSON Pointer into the predecessor output."},"value":{"description":"Required when kind is json_pointer_equals: expected JSON value."}}}}}}}},"dynamic_graph":{"type":"object","description":"Dynamic-mode definition: candidate agents plus optional topology and pooling policy.","required":["candidates"],"properties":{"candidates":{"type":"array","minItems":2,"maxItems":16,"description":"Candidate agents considered for the graph (2 to 16). Candidate ids must be unique.","items":{"type":"object","required":["id","role"],"properties":{"id":{"type":"string","description":"Unique candidate identifier."},"role":{"type":"string","description":"Candidate specialty used during relevance scoring and answer generation."},"prompt":{"type":"string","description":"Optional candidate-specific instructions."},"agent":{"type":"string","description":"Optional agent profile name for this candidate."},"model":{"type":"string","description":"Optional model override for this candidate."},"inherit":{"type":"string","description":"Context inherited by candidate tasks: none, all, cwd, memory, history, skills, or a comma-separated combination. Default cwd,skills."}}}},"policy":{"type":"object","description":"Optional candidate-selection, edge-selection, and final-pooling policy.","properties":{"min_selected":{"type":"integer","minimum":2,"description":"Minimum candidates retained after relevance scoring. Default 2."},"max_selected":{"type":"integer","minimum":2,"description":"Maximum candidates retained after relevance scoring. Default 4."},"relevance_threshold":{"type":"number","minimum":0,"maximum":1,"description":"Minimum candidate relevance score. Default 0.2."},"edge_threshold":{"type":"number","minimum":0,"maximum":1,"description":"Minimum pairwise relevance score for a message edge. Default 0.35."},"pool":{"type":"string","enum":["max","judge"],"description":"Final-answer strategy: max selects the highest-scored answer; judge asks a pooling agent to synthesize one. Default judge."},"pool_agent":{"type":"string","description":"Optional agent profile for judge pooling."},"pool_model":{"type":"string","description":"Optional model override for judge pooling."}}}}}}}}}"##,
        ),
    ];

    #[test]
    fn complete_evidence_recovery_tool_descriptions_are_effective() {
        for enabled in [false, true] {
            crate::ai::tools::enable_tools::with_agent_team_optimizations_for_test(
                &format!("ai.agent_team.optimizations_enabled = {enabled}\n"),
                || {
                    let names = ["task_evidence_read", "task_wait"].map(String::from);
                    for definition in get_tool_definitions_by_names(&names) {
                        crate::ai::tools::task_tools::assert_full_evidence_recovery_hints(
                            &definition.function.description,
                        );
                    }
                },
            );
        }
    }

    #[test]
    fn agent_team_master_off_schemas_match_head_bytes() {
        for config in [
            "",
            "ai.agent_team.optimizations_enabled = false\nai.agent_team.lazy_group_schemas = true\nai.agent_team.persist_status_cursor = true\nai.agent_team.auto_mount_lifecycle_tools = true\n",
        ] {
            crate::ai::tools::enable_tools::with_agent_team_optimizations_for_test(config, || {
                let names = HEAD_AGENT_TEAM_DEFINITION_BYTES
                    .iter()
                    .map(|(name, _)| name.to_string())
                    .collect::<Vec<_>>();
                let definitions = get_tool_definitions_by_names(&names);
                assert_eq!(definitions.len(), names.len());
                let grouped =
                    tool_definitions_for_groups(&[super::super::tool_groups::ToolGroup::Builtin]);
                for (name, expected) in HEAD_AGENT_TEAM_DEFINITION_BYTES {
                    let definition = definitions
                        .iter()
                        .find(|tool| tool.function.name == *name)
                        .unwrap();
                    assert_eq!(
                        serde_json::to_string(definition).unwrap().as_bytes(),
                        expected.as_bytes(),
                        "{name}"
                    );
                    let from_group = grouped
                        .iter()
                        .find(|tool| tool.function.name == *name)
                        .unwrap();
                    assert_eq!(
                        serde_json::to_string(from_group).unwrap().as_bytes(),
                        expected.as_bytes(),
                        "{name}"
                    );
                    let expected: ToolDefinition = serde_json::from_str(expected).unwrap();
                    assert_eq!(
                        serde_json::to_vec(&super::super::tool_metadata::tool_parameters(name))
                            .unwrap(),
                        serde_json::to_vec(&expected.function.parameters).unwrap(),
                        "direct metadata matches the base schema for {name}",
                    );
                }
            });
        }
    }

    #[test]
    fn agent_team_master_on_schemas_expose_extension_properties() {
        crate::ai::tools::enable_tools::with_agent_team_optimizations_for_test(
            "ai.agent_team.optimizations_enabled = true\n",
            || {
                let names = HEAD_AGENT_TEAM_DEFINITION_BYTES
                    .iter()
                    .map(|(name, _)| name.to_string())
                    .collect::<Vec<_>>();
                let definitions = get_tool_definitions_by_names(&names);
                let grouped =
                    tool_definitions_for_groups(&[super::super::tool_groups::ToolGroup::Builtin]);
                for (name, expected) in HEAD_AGENT_TEAM_DEFINITION_BYTES {
                    let definition = definitions
                        .iter()
                        .find(|tool| tool.function.name == *name)
                        .unwrap();
                    let from_group = grouped
                        .iter()
                        .find(|tool| tool.function.name == *name)
                        .unwrap();
                    assert_eq!(
                        serde_json::to_vec(definition).unwrap(),
                        serde_json::to_vec(from_group).unwrap()
                    );
                    let expected: ToolDefinition = serde_json::from_str(expected).unwrap();
                    let extensions = &AGENT_TEAM_SCHEMA_EXTENSIONS[name];
                    for (field, schema) in extensions {
                        assert_eq!(
                            &definition.function.parameters["properties"][field], schema,
                            "{name}.{field}"
                        );
                    }
                    let mut base = definition.function.parameters.clone();
                    for field in extensions.keys() {
                        base["properties"].as_object_mut().unwrap().remove(field);
                    }
                    assert_eq!(
                        serde_json::to_vec(&base).unwrap(),
                        serde_json::to_vec(&expected.function.parameters).unwrap()
                    );
                }
            },
        );
    }

    #[test]
    fn agent_team_schema_policies_keep_hidden_metadata_independent() {
        for name in ["manage_team", "run_agent_graph"] {
            assert!(!tool_has_lazy_group_schema(name, false));
            assert!(tool_has_lazy_group_schema(name, true));
            assert!(!tool_defers_eager_load(name));
        }
        assert!(!group_gates_hidden_tools(
            super::super::tool_groups::ToolGroup::AgentTeam
        ));
        assert!(group_gates_hidden_tools(
            super::super::tool_groups::ToolGroup::Executor
        ));
        assert!(tool_defers_eager_load("spawn_process"));
        assert!(!tool_has_lazy_group_schema("spawn_process", true));
        assert!(!is_task_lifecycle_tool("task_spawn"));
        assert!(!tool_has_lazy_group_schema("task_spawn", true));
        assert_eq!(
            task_lifecycle_tool_names(),
            [
                "send_side_note",
                "task_audit",
                "task_cancel",
                "task_evidence_read",
                "task_integrate",
                "task_retry",
                "task_status",
                "task_wait"
            ]
            .map(String::from)
            .to_vec(),
        );
    }

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

    #[test]
    fn meta_prefixed_unknown_tool_gets_correction_message() {
        let msg = unknown_tool_error_message("meta:self_note");
        assert!(msg.contains("not a tool"));
        assert!(msg.contains("<meta:self_note>"));
        assert!(!msg.starts_with("Unknown tool"));
        assert_eq!(unknown_tool_error_message("nope"), "Unknown tool: nope");
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

/// Error message for an unregistered tool name. `meta:*` names (e.g. the
/// hidden `<meta:self_note>` channel) are plain-text markers, not tools:
/// issuing one as a tool call is a serialization mistake, so the error tells
/// the model to write the tags in the message body instead of a tool_call.
fn unknown_tool_error_message(name: &str) -> String {
    if name.starts_with("meta:") {
        format!(
            "`{name}` is not a tool; it is a plain-text marker. Write the content between \
             `<{name}>` and `</{name}>` in the message body instead of issuing a tool call."
        )
    } else {
        format!("Unknown tool: {}", name)
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
        return Err(unknown_tool_error_message(name));
    };
    let started = std::time::Instant::now();
    let pending_check =
        crate::ai::tools::storage::mutation_log::begin_check(tool_call_id, name, args);
    let exec = crate::ai::tools::storage::mutation_log::with_tool_call(tool_call_id, || {
        run_tool_guarded(&name, || {
            if let Some(stream_exec) = TOOL_STREAM_INDEX.get_ref(&name.to_string()).copied() {
                let mut sink = |_chunk: &[u8]| {};
                let writer = match on_chunk {
                    Some(writer) => writer,
                    None => &mut sink,
                };
                stream_exec(args, writer)
            } else {
                (spec.execute)(args)
            }
        })
    });
    if pending_check.is_some() {
        match &exec {
            Ok(text) => crate::ai::tools::storage::mutation_log::finish_check(
                pending_check,
                name,
                args,
                text,
            ),
            Err(text) => crate::ai::tools::storage::mutation_log::finish_check(
                pending_check,
                name,
                args,
                &format!("Error: {text}"),
            ),
        }
    }
    let elapsed_ms = started.elapsed().as_millis() as u64;
    match exec {
        Ok(result) => {
            record_tool_stat(name, true);
            record_tool_decision_with_time(name, true, "ok", elapsed_ms);
            Ok(ToolResult {
                task_report: None,
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

/// Run a tool implementation, turning a panic into an ordinary tool error.
///
/// Contain unwinding tool panics so the turn can retain its other results and
/// the model can respond to the error.
fn run_tool_guarded<F>(tool_name: &str, run: F) -> Result<String, String>
where
    F: FnOnce() -> Result<String, String>,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(result) => result,
        Err(payload) => Err(format!(
            "tool '{tool_name}' panicked: {}",
            panic_payload_message(payload.as_ref())
        )),
    }
}

/// Recover the readable message from a panic payload.
fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "unknown panic payload".to_string()
}

/// Record a tool invocation outcome in the session decision log.
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
        distilled: None,
        owner_pid: None,
        owner_pgid: None,
        image_path: None,
        // Rule-emitted stats; no model produced this content.
        model: None,
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
    fn a_panicking_tool_becomes_a_tool_error_instead_of_killing_the_turn() {
        // A tool that panics used to unwind out of the turn process, so one
        // buggy tool destroyed the whole turn and every tool result in it.
        let error = run_tool_guarded("boom", || -> Result<String, String> {
            panic!("tool exploded");
        })
        .expect_err("a panicking tool must report an error");
        assert!(error.contains("boom"), "{error}");
        assert!(error.contains("tool exploded"), "{error}");
    }

    #[test]
    fn a_panicking_tool_with_a_string_payload_keeps_its_message() {
        let error = run_tool_guarded("boom", || -> Result<String, String> {
            panic!("{}", "formatted payload");
        })
        .expect_err("a panicking tool must report an error");
        assert!(error.contains("formatted payload"), "{error}");
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
    fn subagent_spawn_tools_block_lossy_and_gate_prune_on_integration() {
        // task_spawn arguments (subagent prompt / response
        // schema) and return values (task_id list) are required inputs for
        // later wait/status/integrate. Missing registration would fall back to
        // the default policy (Allow/Allow); after folding, evidence degrades
        // to the first characters of the result (the original args do not
        // participate in recall), and the main agent loses grounding on
        // already-spawned subtasks. Pruning stays possible, but only after
        // integration: see `ToolPrunePolicy::AfterIntegration`. This assertion
        // pins the contract.
        for name in ["task_spawn"] {
            let policy = tool_history_policy(name);
            assert!(
                !policy.allows_lossy_compress(),
                "{name} must block lossy compress"
            );
            assert!(
                !policy.allows_prune(),
                "{name} must not be prunable without consulting the evidence ledger"
            );
            assert_eq!(
                policy.prune_policy(),
                ToolPrunePolicy::AfterIntegration,
                "{name} must gate pruning on integration"
            );
        }
    }

    #[test]
    fn parallel_policy_defaults_to_serial_for_unregistered_tools() {
        assert_eq!(tool_parallel_policy("tree"), ToolParallelPolicy::Serial);
        assert_eq!(
            tool_parallel_policy("not_a_registered_tool"),
            ToolParallelPolicy::Serial
        );
    }

    #[test]
    fn every_parallel_policy_names_a_registered_tool() {
        // A typo in a policy registration must fail here instead of silently
        // leaving the intended tool serial forever.
        for reg in inventory::iter::<ToolParallelPolicyRegistration> {
            assert!(
                REGISTERED_TOOL_NAMES.contains(reg.name),
                "parallel policy registered for unknown tool: {}",
                reg.name
            );
        }
    }

    #[test]
    fn retry_policy_defaults_to_no_retry() {
        // Side effects are not encoded in names. In particular, "forget" and
        // "mailbox" must not inherit retry safety from "get"/"read" substrings.
        assert_eq!(
            tool_retry_policy("knowledge_forget"),
            ToolRetryPolicy::NoRetry
        );
        assert_eq!(tool_retry_policy("read_mailbox"), ToolRetryPolicy::NoRetry);
        assert_eq!(
            tool_retry_policy("task_evidence_read"),
            ToolRetryPolicy::NoRetry
        );
        assert_eq!(
            tool_retry_policy("execute_command"),
            ToolRetryPolicy::NoRetry
        );
        assert_eq!(
            tool_retry_policy("not_a_registered_tool"),
            ToolRetryPolicy::NoRetry
        );
    }

    #[test]
    fn retry_safe_policy_is_explicit_and_names_registered_tools() {
        assert_eq!(tool_retry_policy("read_file"), ToolRetryPolicy::RetrySafe);
        assert_eq!(
            tool_retry_policy("knowledge_search"),
            ToolRetryPolicy::RetrySafe
        );
        for reg in inventory::iter::<ToolRetryPolicyRegistration> {
            assert!(
                REGISTERED_TOOL_NAMES.contains(reg.name),
                "retry policy registered for unknown tool: {}",
                reg.name
            );
        }
    }
}
