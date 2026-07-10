use serde_json::Value;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::time::Instant;

use crate::ai::tools::os_tools::GLOBAL_OS;
use crate::ai::{
    agents::{self, AgentManifest, AgentModelTier},
    driver::{
        TextSimilarityFeatures, build_idf_from_documents, cosine_tfidf_similarity,
        normalize_text_for_similarity,
    },
    models,
    tools::common::{ToolRegistration, ToolSpec},
    tools::registry::common::current_process_tool_cancel_futex,
};
use aios_kernel::SharedKernel;
use aios_kernel::{
    kernel::{EventId, Kernel, ProcessState, WaitPolicy},
    primitives::{
        ChannelId, ChannelOwnerTag, EpollEventMask, EpollSource, EpollWaitResult, FutexAddr,
        IpcRecvResult,
    },
};
use rust_tools::cw::{SkipMap, SkipSet};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const MAX_TASK_REGISTRY_SIZE: usize = 100;
const DEFAULT_TASK_PRIORITY: u8 = 20;
const DEFAULT_TASK_QUOTA_TURNS: usize = 10;
const TASK_GOAL_PREFIX: &str = "AIOS_SUBAGENT_TASK:";
const MAX_AGENT_TEAM_MEMBERS: usize = 8;
/// 单次 `task_wait` 调用的默认等待预算（秒）。这只是 **本次调用的最长阻塞时间**，
/// 不是 subagent 的总寿命：超时仅意味着"这次没等到结果"，主 agent 可以继续调
/// `task_wait` 续等，subagent 仍在后台运行，channel/futex 也不会被销毁。
///
/// 之前默认 120s，太容易被 LLM 误判为"subagent 卡住"——很多正常 subagent 跑一轮
/// LLM + 多个工具调用就需要 2~5 分钟。提高到 600s 让等待与正常运行时长更匹配。
const DEFAULT_TASK_WAIT_TIMEOUT_SECS: u64 = 600;
/// `task_wait.timeout_secs` 的硬上限，避免模型把 timeout 设成天文数字时彻底
/// 阻塞 driver。上限高于默认值，允许模型在确有需要时显式等待更久（与
/// `params_task_wait` schema 中标称的 `[1, 900]` 保持一致）。超时只表示本次调用
/// 没等到、subagent 仍在跑，因此单次阻塞不宜过长，以保证对中断/事件的响应性。
const MAX_TASK_WAIT_TIMEOUT_SECS: u64 = 900;

/// Granular control over which slices of the parent agent's execution
/// context are inherited by a spawned sub-agent. Defaults are
/// history+cwd+skills=true and memory=false (private memory) unless the
/// caller specifies an `inherit` argument on the tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InheritOptions {
    pub(crate) history: bool,
    pub(crate) memory: bool,
    pub(crate) cwd: bool,
    pub(crate) skills: bool,
}

impl Default for InheritOptions {
    fn default() -> Self {
        // 修复点：sub-agent 默认私有 memory。
        // 历史默认 `memory: true` 会让所有 sub-agent 直接写到主 memory 文件，
        // 一次大型 sub-agent run 产生的 task_event/log 会污染主记忆，
        // 也削弱了召回准确性。现在默认私有：sub-agent 写入独立 jsonl，
        // finalize 时只把白名单条目（is_permanent_memory）merge 回主文件。
        // 调用方仍可显式传 `inherit: "all"` 或 `inherit: "memory"` 退回旧行为。
        Self {
            history: true,
            memory: false,
            cwd: true,
            skills: true,
        }
    }
}

impl InheritOptions {
    /// Parse the optional `inherit` field from a tool call.
    /// Recognised forms:
    ///   - missing / null -> default (history+cwd+skills, **memory private**)
    ///   - "all"          -> full inheritance (incl. memory)
    ///   - "none"         -> no inheritance (fresh sub-agent)
    ///   - comma-separated list of: history, memory, cwd, skills
    pub(crate) fn from_value(value: &Value) -> Result<Self, String> {
        let Some(raw) = value.as_str() else {
            if value.is_null() {
                return Ok(Self::default());
            }
            return Err("'inherit' must be a string".to_string());
        };
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Ok(Self::default());
        }
        if trimmed.eq_ignore_ascii_case("all") {
            return Ok(Self {
                history: true,
                memory: true,
                cwd: true,
                skills: true,
            });
        }
        if trimmed.eq_ignore_ascii_case("none") {
            return Ok(Self {
                history: false,
                memory: false,
                cwd: false,
                skills: false,
            });
        }
        let mut opts = Self {
            history: false,
            memory: false,
            cwd: false,
            skills: false,
        };
        for part in trimmed.split(',') {
            match part.trim().to_ascii_lowercase().as_str() {
                "history" => opts.history = true,
                "memory" => opts.memory = true,
                "cwd" => opts.cwd = true,
                "skills" => opts.skills = true,
                "" => {}
                other => {
                    return Err(format!(
                        "Unknown inherit option '{}'. Allowed: history, memory, cwd, skills, all, none",
                        other
                    ));
                }
            }
        }
        Ok(opts)
    }

    pub(crate) fn describe(&self) -> String {
        if self.history && self.memory && self.cwd && self.skills {
            return "all".to_string();
        }
        if !self.history && !self.memory && !self.cwd && !self.skills {
            return "none".to_string();
        }
        let mut parts = Vec::new();
        if self.history {
            parts.push("history");
        }
        if self.memory {
            parts.push("memory");
        }
        if self.cwd {
            parts.push("cwd");
        }
        if self.skills {
            parts.push("skills");
        }
        parts.join(",")
    }
}

/// Agent 层为每个异步子任务维护的注册表条目，用于 `task_spawn` / `task_wait` 流程。
///
/// **与 AIOS Kernel `Process` 的关系**：本结构体的部分字段（`pid`、`agent_name`、
/// `description`、`started_at`）在 kernel `Process` 中已有等价物（`pid` / `name` /
/// `goal` / `created_at_tick`），存在 **概念重叠**。重叠保留的原因：
///
/// 1. agent 特有字段（`result_channel_id`、`completion_futex_addr`、`inherit`、
///    `selection_explanation`、`model`）在 kernel 进程表中没有对应位置；
/// 2. agent 层需要在 task_id 这个稳定字符串键下做查询，而 kernel 用的是数值 pid；
/// 3. kernel `created_at_tick` 是 logical tick，不能直接换算回 wall-clock 用于
///    `prune_completed_tasks` 的 LRU 决策。
///
/// **不变量**：本注册表中的 `pid` 必须始终对应 kernel process table 里同一个
/// 进程；如果 kernel 端进程被 reap，此注册表里的对应条目应在
/// `prune_completed_tasks`（容量上限）或 `task_wait` 完成时被移除。
pub(crate) struct AsyncTaskEntry {
    /// 与 kernel `Process.pid` 一致；agent 端额外保存便于通过 task_id 反查 pid。
    pub(crate) pid: u64,
    pub(crate) result_channel_id: u64,
    pub(crate) completion_futex_addr: FutexAddr,
    /// 描述性文本；与 kernel `Process.goal` 不同——后者会带 TASK_GOAL_PREFIX
    /// 前缀和完整 prompt。
    pub(crate) description: String,
    /// 子 agent 的逻辑名（"explore" / "plan" 等）；与 kernel `Process.name` 同源
    /// 但 kernel 端 name 仅作显示。
    pub(crate) agent_name: String,
    pub(crate) model: String,
    pub(crate) is_model_auto_selected: bool,
    pub(crate) auto_model_fallback: Option<models::AutoModelFallbackSpec>,
    pub(crate) selection_explanation: String,
    pub(crate) inherit: InheritOptions,
    /// wall-clock 起始时间，用于 `prune_completed_tasks` LRU；不能由 kernel
    /// `created_at_tick` 替代。
    pub(crate) started_at: Instant,
}

/// 异步子任务注册表，键为 task_id（UUID 字符串），值见 [`AsyncTaskEntry`]。
///
/// 与 AIOS kernel process table 是 **平行存储**：两者通过 `pid` 字段关联，但
/// 各自有独立的字段集（参见 `AsyncTaskEntry` 注释）。访问方应通过 `with_task_entry`
/// / `take_task_entry` 等 helper 函数来读写这里，避免直接持有 lock guard。
static TASK_REGISTRY: LazyLock<Mutex<SkipMap<String, AsyncTaskEntry>>> =
    LazyLock::new(|| Mutex::new(SkipMap::default()));

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct OsTaskGoal {
    pub(crate) task_id: String,
    pub(crate) result_channel_id: u64,
    pub(crate) completion_futex_addr: u64,
    pub(crate) description: String,
    pub(crate) prompt: String,
    pub(crate) agent_name: String,
    pub(crate) model: String,
    #[serde(default)]
    pub(crate) is_model_auto_selected: bool,
    #[serde(default)]
    pub(crate) auto_model_fallback: Option<models::AutoModelFallbackSpec>,
    pub(crate) selection_explanation: String,
}

fn prune_completed_tasks(registry: &mut SkipMap<String, AsyncTaskEntry>) {
    if registry.len() <= MAX_TASK_REGISTRY_SIZE {
        return;
    }
    // 仅驱逐已完成的任务（process 已终止），绝不驱逐仍在运行的任务。
    // 注意：这里只从注册表中移除条目，内核 channel/futex 的释放由 task_wait /
    // task_status 的正常收集路径完成。如果 process 已终止但结果未被收集，
    // 下次 task_wait 会进入失败路径并释放 channel/futex（包括 producer holder）。
    // 按时间排序避免 O(n²) 全量扫描。
    let now = Instant::now();
    let mut candidates: Vec<(String, Instant)> = Vec::new();
    for (key, entry) in registry.iter() {
        // 通过 kernel 检查进程是否已终止；若无法访问 kernel 则保守跳过。
        let terminated = with_os_kernel(|os| {
            match os.get_process(entry.pid) {
                None => Ok(true), // 进程不存在 → 已终止
                Some(proc) => Ok(matches!(proc.state, ProcessState::Terminated)),
            }
        })
        .unwrap_or(false);
        if terminated {
            candidates.push((key.clone(), entry.started_at));
        }
    }
    candidates.sort_by_key(|(_, t)| *t);
    let to_remove = candidates
        .len()
        .min(registry.len().saturating_sub(MAX_TASK_REGISTRY_SIZE));
    let _ = now; // suppress unused
    for (key, _) in candidates.into_iter().take(to_remove) {
        // 在移除注册表条目前，尝试释放内核资源（best-effort）。
        if let Some(entry) = registry.get_ref(&key) {
            let _ = with_os_kernel(|os| {
                let _ = os.channel_close(None, ChannelId(entry.result_channel_id));
                let _ = os.channel_release_named(
                    ChannelId(entry.result_channel_id),
                    "task_result.consumer",
                );
                let _ = os.channel_release_named(
                    ChannelId(entry.result_channel_id),
                    "task_result.producer",
                );
                let _ = os.channel_destroy(None, ChannelId(entry.result_channel_id));
                let _ = os.futex_destroy(entry.completion_futex_addr);
                Ok::<(), String>(())
            });
        }
        registry.remove(&key);
    }
}

fn next_task_id() -> String {
    format!("task_{}", Uuid::new_v4().simple())
}

pub(crate) fn encode_os_task_goal(goal: &OsTaskGoal) -> Result<String, String> {
    serde_json::to_string(goal)
        .map(|payload| format!("{TASK_GOAL_PREFIX}{payload}"))
        .map_err(|err| format!("Failed to encode task goal: {err}"))
}

pub(crate) fn is_encoded_task_goal(goal: &str) -> bool {
    goal.starts_with(TASK_GOAL_PREFIX)
}

pub(crate) fn decode_os_task_goal(goal: &str) -> Option<OsTaskGoal> {
    let payload = goal.strip_prefix(TASK_GOAL_PREFIX)?;
    serde_json::from_str(payload).ok()
}

/// 在 AIOS kernel 上执行一段 mutable 操作。
///
/// 优先路径：从 `DRIVER_CTX` task-local 取出当前 turn 持有的 `SharedKernel`，
/// 这样 `task_wait` / `task_spawn` 等高频路径直接复用 turn scope 已经持有的 Arc，
/// 避免 `GLOBAL_OS` 这个全局 static 的额外锁与间接寻址。
///
/// 回退路径：当调用方不在 `DRIVER_CTX` scope 中（例如 driver 启动早期或单测从同步
/// 上下文调用 tool），仍使用 `GLOBAL_OS`，保证向后兼容。
fn with_os_kernel<T>(f: impl FnOnce(&mut dyn Kernel) -> Result<T, String>) -> Result<T, String> {
    let shared: SharedKernel = match crate::ai::driver::runtime_ctx::try_current() {
        Some(ctx) => ctx.app_proto.os.clone(),
        None => {
            let guard = GLOBAL_OS
                .lock()
                .map_err(|e| format!("Failed to lock AIOS kernel handle: {e}"))?;
            guard
                .as_ref()
                .cloned()
                .ok_or("AIOS kernel is not initialized.".to_string())?
        }
    };
    let mut kernel = shared
        .lock()
        .map_err(|e| format!("Failed to lock AIOS kernel: {e}"))?;
    f(kernel.as_mut())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EpollWaitManyOutcome {
    pub(crate) ready_sources: Vec<WaitManySource>,
    pub(crate) pending_sources: Vec<WaitManySource>,
    pub(crate) event_ids: Vec<EventId>,
    pub(crate) suspended: bool,
    pub(crate) timeout_tick: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum WaitManySource {
    Channel(u64),
    Event(EventId),
    Futex { addr: FutexAddr, expected: u64 },
}

pub(crate) fn wait_sources_for_channel_and_futex(
    os: &mut dyn Kernel,
    channel_id: u64,
    completion_futex_addr: Option<FutexAddr>,
) -> Result<Vec<WaitManySource>, String> {
    let mut sources = vec![WaitManySource::Channel(channel_id)];
    let channel_event = os
        .channel_event_id(ChannelId(channel_id))
        .ok_or_else(|| format!("Channel {} has no waitable event id.", channel_id))?;
    sources.push(WaitManySource::Event(channel_event));
    if let Some(addr) = completion_futex_addr {
        sources.push(WaitManySource::Futex { addr, expected: 0 });
    }
    Ok(sources)
}

pub(crate) fn append_current_process_cancel_source(
    os: &mut dyn Kernel,
    sources: &mut Vec<WaitManySource>,
) -> Result<(), String> {
    if let Some(addr) = current_process_tool_cancel_futex(os)? {
        sources.push(WaitManySource::Futex { addr, expected: 0 });
    }
    Ok(())
}

impl WaitManySource {
    fn epoll_source(self) -> EpollSource {
        match self {
            Self::Channel(channel_id) => EpollSource::Channel(ChannelId(channel_id)),
            Self::Event(event_id) => EpollSource::Event(event_id),
            Self::Futex { addr, expected } => EpollSource::Futex { addr, expected },
        }
    }

    fn epoll_mask(self) -> EpollEventMask {
        match self {
            Self::Channel(_) => EpollEventMask::IN | EpollEventMask::HUP | EpollEventMask::ERR,
            Self::Event(_) | Self::Futex { .. } => EpollEventMask::IN | EpollEventMask::ERR,
        }
    }
}

fn wait_many_snapshot(
    os: &mut dyn Kernel,
    sources: &[WaitManySource],
) -> Result<(Vec<WaitManySource>, Vec<WaitManySource>, Vec<EventId>), String> {
    let mut ready = Vec::new();
    let mut pending = Vec::new();
    let mut event_ids = Vec::new();
    for source in sources {
        let event_id = match *source {
            WaitManySource::Channel(channel_id) => {
                let channel = ChannelId(channel_id);
                let meta = os
                    .channel_meta(channel)
                    .ok_or_else(|| format!("Channel {} no longer exists.", channel_id))?;
                if meta.queued_len > 0 || meta.closed {
                    ready.push(*source);
                    continue;
                }
                os.channel_event_id(channel)
                    .ok_or_else(|| format!("Channel {} has no waitable event id.", channel_id))?
            }
            WaitManySource::Event(event_id) => {
                if os.event_is_completed(event_id) {
                    ready.push(*source);
                    continue;
                }
                event_id
            }
            WaitManySource::Futex { addr, expected } => {
                if os.futex_try_wait(addr, expected).is_some() {
                    ready.push(*source);
                    continue;
                }
                os.futex_event_id(addr)
                    .ok_or_else(|| format!("Futex {} has no waitable event id.", addr.raw()))?
            }
        };
        pending.push(*source);
        event_ids.push(event_id);
    }
    Ok((ready, pending, event_ids))
}

/// 在 agent 层组合 kernel 提供的 epoll / channel / futex / event 原语，实现
/// **跨多种等待源** 的 "等待任意一个完成" 语义，主要服务于 `task_wait` 工具。
///
/// **设计定位**：本函数 *不是* 重新实现 kernel 的等待原语，而是把若干低层 API
/// （`epoll_create` / `epoll_ctl` / `epoll_wait` / `wait_on_events`）按 agent
/// 业务语义拼装：
/// 1. 为 channel/futex 类等待源建立短暂的 epoll 集合，再 `epoll_wait` 取就绪集合；
/// 2. 为 event 类等待源直接 `wait_on_events`；
/// 3. 把两类结果归一化到 `EpollWaitManyOutcome`。
///
/// **未来下沉建议**：当 kernel 加入对 `Vec<WaitManySource>` 的原生 syscall 支持
/// （类似 epoll_pwait2 + EVENTFD 的混合模式）后，本函数可以变成对单次 syscall
/// 的轻量包装。在迁移前，本函数保留当前的多步组合实现；任何对其行为的修改
/// **必须保证 task_wait 在如下场景的回归**：
/// - 全部 ready 立即返回（不会调用 epoll_wait）；
/// - 全部 pending 时按 `wait_policy` 决定是否真正 suspend；
/// - 混合就绪 + pending 时只返回就绪集，不引入额外阻塞。
pub(crate) fn epoll_wait_many(
    os: &mut dyn Kernel,
    label: &str,
    sources: &[WaitManySource],
    wait_policy: WaitPolicy,
    timeout_ticks: Option<u64>,
) -> Result<EpollWaitManyOutcome, String> {
    if sources.is_empty() {
        return Ok(EpollWaitManyOutcome {
            ready_sources: Vec::new(),
            pending_sources: Vec::new(),
            event_ids: Vec::new(),
            suspended: false,
            timeout_tick: None,
        });
    }

    let epoll = os.epoll_create(label.to_string());
    let result = (|| {
        for (index, source) in sources.iter().enumerate() {
            os.epoll_ctl_add(
                epoll,
                source.epoll_source(),
                source.epoll_mask(),
                index as u64,
            )?;
        }

        let (ready_sources, pending_sources, event_ids) = wait_many_snapshot(os, sources)?;
        let satisfied = match wait_policy {
            WaitPolicy::Any => !ready_sources.is_empty(),
            WaitPolicy::All => pending_sources.is_empty(),
        };
        if satisfied {
            return Ok(EpollWaitManyOutcome {
                ready_sources,
                pending_sources,
                event_ids,
                suspended: false,
                timeout_tick: None,
            });
        }

        match wait_policy {
            WaitPolicy::Any => match os.epoll_wait(epoll, sources.len(), timeout_ticks)? {
                EpollWaitResult::Ready(_) => {
                    let (ready_sources, pending_sources, event_ids) =
                        wait_many_snapshot(os, sources)?;
                    Ok(EpollWaitManyOutcome {
                        ready_sources,
                        pending_sources,
                        event_ids,
                        suspended: false,
                        timeout_tick: None,
                    })
                }
                EpollWaitResult::Suspended { timeout_tick } => {
                    // epoll_wait 内部已 consume 了 yield_requested 标志用于判定挂起；
                    // 必须把它重新置位，否则 turn-loop 的 consume_yield_requested()
                    // 读到 false，控制权无法交还调度器，已就绪的子 agent 永远不被派发。
                    os.request_yield();
                    Ok(EpollWaitManyOutcome {
                        ready_sources,
                        pending_sources,
                        event_ids,
                        suspended: true,
                        timeout_tick,
                    })
                }
            },
            WaitPolicy::All => {
                let wake_tick =
                    os.wait_on_events(event_ids.clone(), WaitPolicy::All, timeout_ticks)?;
                let suspended = os.consume_yield_requested() || wake_tick.is_some();
                if suspended {
                    // 同上：本分支用 consume_yield_requested() 探测挂起，会清掉让出
                    // 意图。确认挂起后重新置位，保证 turn-loop 能感知并交还调度权。
                    os.request_yield();
                }
                let (ready_sources, pending_sources, refreshed_event_ids) =
                    wait_many_snapshot(os, sources)?;
                Ok(EpollWaitManyOutcome {
                    ready_sources,
                    pending_sources,
                    event_ids: if suspended {
                        event_ids
                    } else {
                        refreshed_event_ids
                    },
                    suspended,
                    timeout_tick: wake_tick,
                })
            }
        }
    })();
    let _ = os.epoll_destroy(epoll);
    result
}

pub(crate) fn epoll_wait_many_channels(
    os: &mut dyn Kernel,
    label: &str,
    channel_ids: &[u64],
    wait_policy: WaitPolicy,
    timeout_ticks: Option<u64>,
) -> Result<EpollWaitManyOutcome, String> {
    let sources = channel_ids
        .iter()
        .copied()
        .map(WaitManySource::Channel)
        .collect::<Vec<_>>();
    epoll_wait_many(os, label, &sources, wait_policy, timeout_ticks)
}

fn task_inherit_schema_description() -> &'static str {
    "Optional inheritance control. Accepts 'all' (inherit history, memory, cwd, skills), 'none' (fresh sub-agent context), or a comma-separated list selecting some of: history, memory, cwd, skills. If omitted, defaults to history+cwd+skills with private memory (memory not inherited). For narrow leaf tasks, prefer 'none' or 'cwd' to give the subagent a focused context; only use 'all' when the subtask genuinely depends on the full conversation."
}

fn params_task() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short description of what this task will do (3-10 words)."
            },
            "prompt": {
                "type": "string",
                "description": "The task/prompt to send to the subagent. Be specific about what you want accomplished."
            },
            "agent": {
                "type": "string",
                "description": "Optional subagent name. Leave empty to let the runtime auto-select the best subagent for this task."
            },
            "model": {
                "type": "string",
                "description": "Optional model override. By default the subagent reuses your (parent) model; only override when the subtask is clearly lighter (use a lighter model to save cost/latency) or heavier than your own."
            },
            "inherit": {
                "type": "string",
                "description": task_inherit_schema_description()
            }
        },
        "required": ["description", "prompt"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task",
        description: "Launch a specialized subagent synchronously and return its final output. The current agent blocks until the subagent finishes. Use this for a single focused side investigation when you need the result before continuing. For multiple parallel subagents prefer task_spawn + task_wait. The runtime auto-selects a subagent when 'agent' is omitted.",
        parameters: params_task,
        execute: execute_task,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

pub(crate) fn execute_task(_args: &Value) -> Result<String, String> {
    Err("task is handled by the runtime".to_string())
}

fn params_task_spawn() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short description of what this task will do (3-10 words)."
            },
            "prompt": {
                "type": "string",
                "description": "The task/prompt to send to the subagent. Be specific about what you want accomplished."
            },
            "agent": {
                "type": "string",
                "description": "Optional subagent name. Leave empty to let the runtime auto-select the best subagent."
            },
            "model": {
                "type": "string",
                "description": "Optional model override. By default the subagent reuses your (parent) model; only override when the subtask is clearly lighter (use a lighter model to save cost/latency) or heavier than your own."
            },
            "inherit": {
                "type": "string",
                "description": task_inherit_schema_description()
            }
        },
        "required": ["description", "prompt"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_spawn",
        description: "Launch a subagent task asynchronously and return immediately with a task_id. Use this whenever you need a delegated task's result returned to you — for a single task or to fan out several in parallel. Unlike spawn_process (fire-and-forget, no result), task_spawn produces a collectable structured final answer. The returned task_id is a long-lived handle: collect results with task_wait (re-callable until the result is consumed) or peek non-blockingly with task_status. Hitting task_wait's timeout does NOT mean the subagent is stuck — it only means the wait budget for that single call elapsed.",
        parameters: params_task_spawn,
        execute: execute_task_spawn,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

fn params_agent_team() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["start", "challenge", "synthesize"],
                "description": "Team phase to launch. 'start' fans out independent members. 'challenge' asks members to critique a transcript. 'synthesize' asks one or more members to produce a final conclusion from a transcript."
            },
            "goal": {
                "type": "string",
                "description": "Shared objective for the team deliberation."
            },
            "members": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "description": "Role name for this team member, for example implementer, reviewer, skeptic, domain expert, synthesizer."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Role-specific instructions. For start, this is the member's investigation brief. For challenge/synthesize, this describes the critique or synthesis angle."
                        },
                        "agent": {
                            "type": "string",
                            "description": "Optional subagent name. Leave empty to auto-select."
                        },
                        "model": {
                            "type": "string",
                            "description": "Optional model override for this member."
                        }
                    },
                    "required": ["role"]
                },
                "description": "Team members to launch in this phase. Use 2-8 members for start; challenge/synthesize may use 1-8."
            },
            "transcript": {
                "type": "string",
                "description": "Collected outputs from prior team phases. Required for challenge/synthesize so no agent relies on direct peer messaging."
            },
            "inherit": {
                "type": "string",
                "description": task_inherit_schema_description()
            }
        },
        "required": ["operation", "goal", "members"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "agent_team",
        description: "Launch a parent-mediated multi-agent team phase over the existing task_spawn/task_wait substrate. Use operation='start' to fan out multiple members, then task_wait to collect all outputs; use operation='challenge' with that transcript to have agents challenge assumptions; use operation='synthesize' with the updated transcript for final consensus. Team members do NOT message each other directly: the parent passes full transcripts between phases, avoiding mailbox/drain/routing bugs while reusing AIOS kernel processes, result channels, futex wakeups, and task_wait.",
        parameters: params_agent_team,
        execute: execute_agent_team,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

/// Pre-flight subagent task spec produced from a `task` / `task_spawn` tool
/// call before the kernel actually spawns the new process.
pub(crate) struct PreparedSubagentTask {
    pub(crate) description: String,
    pub(crate) prompt: String,
    pub(crate) agent_name: String,
    pub(crate) model: String,
    pub(crate) is_model_auto_selected: bool,
    pub(crate) auto_model_fallback: Option<models::AutoModelFallbackSpec>,
    pub(crate) selection_explanation: String,
    pub(crate) inherit: InheritOptions,
}

/// Parse and validate a `task` / `task_spawn` tool call payload, run subagent
/// auto-selection, and resolve the model. Used both by the async `task_spawn`
/// path and by the synchronous `task` interception in the driver.
pub(crate) fn prepare_subagent_task(args: &Value) -> Result<PreparedSubagentTask, String> {
    let description = args["description"]
        .as_str()
        .ok_or("Missing 'description' parameter")?;
    let prompt = args["prompt"]
        .as_str()
        .ok_or("Missing 'prompt' parameter")?;
    let agent = args["agent"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let model_override = args["model"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty());

    if description.trim().is_empty() {
        return Err("description cannot be empty".to_string());
    }
    if prompt.trim().is_empty() {
        return Err("prompt cannot be empty".to_string());
    }

    let inherit = InheritOptions::from_value(&args["inherit"])?;

    // 优先从 DRIVER_CTX 中拿已缓存的 agent_manifests，避免每次 task_spawn 都重读磁盘。
    // 当不在 DRIVER_CTX scope 中（极少见，例如单测），回退到 load_all_agents()。
    let cached = crate::ai::driver::runtime_ctx::try_current();
    let owned_fallback;
    let all_agents: &[AgentManifest] = if let Some(ref ctx) = cached {
        ctx.agent_manifests.as_slice()
    } else {
        owned_fallback = agents::load_all_agents();
        &owned_fallback
    };
    let selected = select_subagent(all_agents, agent, description, prompt)?;
    let (selected_model, is_model_auto_selected, auto_model_fallback, inherited_parent_model) =
        if let Some(model_override) = model_override {
            (models::determine_model(model_override), false, None, false)
        } else {
            let parent_model = cached
                .as_ref()
                .map(|ctx| ctx.app_proto.current_model.as_str());
            let choice = models::choose_model_for_subagent(
                parent_model,
                selected.agent,
                description,
                prompt,
            );
            (
                choice.model,
                choice.is_auto_selected,
                choice.fallback,
                !choice.is_auto_selected,
            )
        };
    let selection_explanation = build_selection_explanation(
        &selected,
        &selected_model,
        model_override,
        inherited_parent_model,
    );

    Ok(PreparedSubagentTask {
        description: description.to_string(),
        prompt: prompt.to_string(),
        agent_name: selected.agent.name.clone(),
        model: selected_model,
        is_model_auto_selected,
        auto_model_fallback,
        selection_explanation,
        inherit,
    })
}

pub(crate) struct SpawnedSubagentTask {
    pub(crate) task_id: String,
    pub(crate) pid: u64,
    pub(crate) result_channel_id: u64,
    pub(crate) completion_futex_addr: FutexAddr,
}

/// Spawn a subagent kernel process and register it in `TASK_REGISTRY`. The
/// returned handle exposes the IPC channel + futex that the caller can wait
/// on. Used by both `task_spawn` (async) and the synchronous `task` runtime
/// interception path.
pub(crate) fn spawn_subagent_kernel_task(
    prepared: &PreparedSubagentTask,
) -> Result<SpawnedSubagentTask, String> {
    let task_id = next_task_id();
    let (pid, result_channel_id, completion_futex_addr) = with_os_kernel(|os| {
        let parent_pid = os
            .current_process_id()
            .ok_or("subagent task requires an active AIOS process context.".to_string())?;
        let result_channel = os.channel_create_tagged_with_holders(
            Some(parent_pid),
            1,
            format!("task_result:{task_id}"),
            ChannelOwnerTag::TaskResult,
            vec![
                "task_result.producer".to_string(),
                "task_result.consumer".to_string(),
            ],
        );
        let completion_futex = os.futex_create(0, format!("task_completion:{task_id}"));
        let process_goal = encode_os_task_goal(&OsTaskGoal {
            task_id: task_id.clone(),
            result_channel_id: result_channel.raw(),
            completion_futex_addr: completion_futex.raw(),
            description: prepared.description.clone(),
            prompt: prepared.prompt.clone(),
            agent_name: prepared.agent_name.clone(),
            model: prepared.model.clone(),
            is_model_auto_selected: prepared.is_model_auto_selected,
            auto_model_fallback: prepared.auto_model_fallback,
            selection_explanation: prepared.selection_explanation.clone(),
        })?;
        let pid = os.spawn(
            Some(parent_pid),
            prepared.agent_name.clone(),
            process_goal,
            DEFAULT_TASK_PRIORITY,
            DEFAULT_TASK_QUOTA_TURNS,
            None,
            None,
        )?;
        Ok((pid, result_channel.raw(), completion_futex))
    })?;

    {
        let mut registry = TASK_REGISTRY.lock().unwrap();
        registry.insert(
            task_id.clone(),
            AsyncTaskEntry {
                pid,
                result_channel_id,
                completion_futex_addr,
                description: prepared.description.clone(),
                agent_name: prepared.agent_name.clone(),
                model: prepared.model.clone(),
                is_model_auto_selected: prepared.is_model_auto_selected,
                auto_model_fallback: prepared.auto_model_fallback,
                selection_explanation: prepared.selection_explanation.clone(),
                inherit: prepared.inherit,
                started_at: Instant::now(),
            },
        );
        prune_completed_tasks(&mut registry);
    }

    Ok(SpawnedSubagentTask {
        task_id,
        pid,
        result_channel_id,
        completion_futex_addr,
    })
}

/// Look up a registered async task entry. Used by the driver-side sync `task`
/// interception to retrieve the channel/futex/inherit info after spawning.
pub(crate) fn with_task_entry<R>(task_id: &str, f: impl FnOnce(&AsyncTaskEntry) -> R) -> Option<R> {
    let registry = TASK_REGISTRY.lock().unwrap();
    registry.get_ref(&task_id.to_string()).map(f)
}

pub(crate) fn with_task_entry_by_pid<R>(
    pid: u64,
    mut f: impl FnMut(&AsyncTaskEntry) -> R,
) -> Option<R> {
    let registry = TASK_REGISTRY.lock().unwrap();
    for (_task_id, entry) in registry.iter() {
        if entry.pid == pid {
            return Some(f(entry));
        }
    }
    None
}

/// Remove a task entry from the registry. Called by the synchronous `task`
/// interception once it has consumed the result.
pub(crate) fn remove_task_entry(task_id: &str) -> Option<AsyncTaskEntry> {
    let mut registry = TASK_REGISTRY.lock().unwrap();
    registry.take(&task_id.to_string())
}
/// channel. Re-exported for the synchronous `task` interception so that both
/// paths produce identical output.
pub(crate) fn format_finished_task(entry: &AsyncTaskEntry, result: StoredTaskResult) -> String {
    format_task_result(entry, result)
}

pub(crate) fn execute_task_spawn(args: &Value) -> Result<String, String> {
    let prepared = prepare_subagent_task(args)?;
    let spawned = spawn_subagent_kernel_task(&prepared)?;

    println!(
        "\n[TaskSpawn] Launched AIOS task pid={} subagent '{}' with model '{}' inherit={} for: {} (task_id: {})",
        spawned.pid,
        prepared.agent_name,
        prepared.model,
        prepared.inherit.describe(),
        prepared.description,
        spawned.task_id,
    );

    Ok(format!(
        "Task spawned: task_id={}, pid={}, agent={}, model={}, inherit={}\nUse task_wait to collect results when ready.",
        spawned.task_id,
        spawned.pid,
        prepared.agent_name,
        prepared.model,
        prepared.inherit.describe()
    ))
}

fn params_task_wait() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "task_ids": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Array of task_id strings returned by task_spawn."
            },
            "timeout_secs": {
                "type": "integer",
                "description": "Wait budget for THIS call (clamped to [1, 900], default 600). Hitting this budget does NOT cancel or stall the subagent — it only means the wait policy was not satisfied within this call. The subagent keeps running, its result channel/futex stay alive, and you can call task_wait again with the same task_ids to keep waiting (or use task_status for a non-blocking snapshot)."
            },
            "wait_policy": {
                "type": "string",
                "enum": ["all", "any"],
                "description": "Optional. 'all' (default) returns when every pending task has finished. 'any' returns as soon as the first task finishes — useful for fan-out where you want to start processing the fastest result while others continue. Remaining tasks stay alive and can be retrieved by another task_wait call."
            }
        },
        "required": ["task_ids"]
    })
}

fn params_tool_spawn() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Builtin or MCP tool name to run asynchronously. The tool must support async spawning."
            },
            "arguments": {
                "type": "object",
                "description": "JSON arguments for the target tool."
            }
        },
        "required": ["tool_name", "arguments"]
    })
}

fn params_tool_wait() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "task_ids": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Array of task ids returned by tool_spawn."
            },
            "max_wait_ms": {
                "type": "integer",
                "description": "Short wait window in milliseconds before returning control to the model. Default 1500."
            },
            "wait_policy": {
                "type": "string",
                "enum": ["any", "all"],
                "description": "When used with AIOS suspend/resume, wake when any waited event finishes or only after all waited events finish. Default is all."
            },
            "timeout_ticks": {
                "type": "integer",
                "description": "Optional AIOS scheduler timeout in ticks when suspending on waited events."
            },
            "timeout_secs": {
                "type": "integer",
                "description": "Legacy alias for wait budget. If max_wait_ms is absent, timeout_secs will be converted to milliseconds."
            }
        },
        "required": ["task_ids"]
    })
}

fn params_tool_status() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "task_ids": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Optional array of tool task ids. If omitted, returns all async tool tasks for the current session."
            }
        }
    })
}

fn params_tool_cancel() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "task_ids": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Array of async tool task ids to cancel."
            },
            "reason": {
                "type": "string",
                "description": "Optional reason for canceling these tasks."
            }
        },
        "required": ["task_ids"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_wait",
        description: "Wait for one or more asynchronously spawned subagent tasks to complete and collect their results. Polls all tasks in parallel so total wait time equals the slowest task, not the sum. The `timeout_secs` argument is a per-call wait budget — when it elapses without satisfying the policy, the call returns with already-collected results AND a clear note that the remaining subagents are still running; you can call task_wait again with the same task_ids to keep waiting (or pass `wait_policy=\"any\"` to wake on the first finisher). Use task_status for a non-blocking snapshot.",
        parameters: params_task_wait,
        execute: execute_task_wait,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "tool_spawn",
        description: "Launch a builtin or MCP tool asynchronously and return immediately with a task id. Use this when the tool call is independent from the current step and you want to fan out multiple lookups in parallel. Preferred cases: reading multiple files, querying multiple MCP tools, fetching several URLs, or launching several unrelated searches before comparing results. Do NOT use this when the next tool depends on this result immediately, when the tool mutates state, or when the calls must happen in strict order. Typical pattern: call tool_spawn several times first, continue reasoning or launch other independent work, then use tool_status or tool_wait later.",
        parameters: params_tool_spawn,
        execute: execute_tool_spawn_placeholder,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "tool_wait",
        description: "Wait for one or more async tool tasks started by tool_spawn. When running inside AIOS process scheduling, this tool suspends the current process by calling wait_on_events and yields control until the wait condition is satisfied or timeout_ticks is reached. When AIOS process context is unavailable, it falls back to a short non-blocking wait window and returns partial progress. Use wait_policy=all to join a batch, or wait_policy=any when you want to resume as soon as any branch finishes.",
        parameters: params_tool_wait,
        execute: execute_tool_wait_placeholder,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "tool_status",
        description: "Inspect async tool tasks started by tool_spawn without blocking. Use this when you want to check progress before deciding whether to wait, continue other reasoning, or spawn more independent work. Preferred cases: long-running MCP requests, background searches, or when only some spawned tasks may have finished and you want to opportunistically use completed results first. Do NOT use this when you already know you must have the final outputs right now; use tool_wait instead.",
        parameters: params_tool_status,
        execute: execute_tool_status_placeholder,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "tool_cancel",
        description: "Cancel one or more async tool tasks started by tool_spawn. Use this when a background lookup is no longer needed, when another result already answered the question, or when the model wants to stop waiting on a low-value branch. This is a best-effort cancel from the runtime perspective: the task becomes canceled and future wait/status calls report it as canceled, but already-running underlying work may continue in the background and its final result will be discarded.",
        parameters: params_tool_cancel,
        execute: execute_tool_cancel_placeholder,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

fn execute_tool_spawn_placeholder(_args: &Value) -> Result<String, String> {
    Err("tool_spawn is handled by the runtime".to_string())
}

fn execute_tool_wait_placeholder(_args: &Value) -> Result<String, String> {
    Err("tool_wait is handled by the runtime".to_string())
}

fn execute_tool_status_placeholder(_args: &Value) -> Result<String, String> {
    Err("tool_status is handled by the runtime".to_string())
}

fn execute_tool_cancel_placeholder(_args: &Value) -> Result<String, String> {
    Err("tool_cancel is handled by the runtime".to_string())
}

pub(crate) fn execute_task_wait(args: &Value) -> Result<String, String> {
    let task_ids = args["task_ids"]
        .as_array()
        .ok_or("Missing 'task_ids' array parameter")?
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect::<Vec<_>>();

    if task_ids.is_empty() {
        return Err("task_ids array cannot be empty".to_string());
    }

    // 单次 task_wait 调用的等待预算。详见 DEFAULT_TASK_WAIT_TIMEOUT_SECS 注释——
    // 超时只意味着本次没等到，subagent 仍在跑、资源不会被释放。
    let timeout_secs = args
        .get("timeout_secs")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_TASK_WAIT_TIMEOUT_SECS)
        .clamp(1, MAX_TASK_WAIT_TIMEOUT_SECS);
    // 启发式：driver 主循环 idle 路径每 ~10ms 调一次 advance_tick()，故 100
    // ticks ≈ 1 秒。这里宁可早醒不晚醒，使用 100 ticks/sec。
    let timeout_ticks = Some(timeout_secs.saturating_mul(100));

    // wait_policy: "any" | "all"，默认 "all"（与历史行为一致）。
    // - all  — 等到所有 pending 任务都完成才返回（适合需要汇总）；
    // - any  — 任一 pending 任务完成即返回，其余仍在跑、可继续 task_wait
    //          （适合 fan-out 后想边收边推进）。
    let wait_policy = match args.get("wait_policy").and_then(Value::as_str) {
        Some("any") => WaitPolicy::Any,
        Some("all") | None => WaitPolicy::All,
        Some(other) => {
            return Err(format!(
                "Unknown wait_policy: {} (expected 'any' or 'all')",
                other
            ));
        }
    };

    let mut registry = TASK_REGISTRY.lock().unwrap();
    // task_id 不在 registry 中，说明它在 *上一次* task_wait 调用里已经被收集并清理
    // 掉了（ready 任务一旦读到结果就会从 registry 删除）。PARKED / BUDGET-ELAPSED
    // 提示以及 driver 唤醒消息都让模型"用 same task_ids 继续调"，所以"已收集 +
    // 仍 pending"混合的一组 id 是预期输入，**绝不能**整调用 hard-fail（否则多子任务
    // 编排会在第二次 task_wait 时因 Unknown task_id 直接崩掉）。这里静默丢弃已收集
    // 的 id，只对仍被跟踪的 id 继续等待。
    let already_collected = task_ids
        .iter()
        .filter(|tid| !registry.contains_key(*tid))
        .count();
    let task_ids: Vec<String> = task_ids
        .into_iter()
        .filter(|tid| registry.contains_key(tid))
        .collect();
    if task_ids.is_empty() {
        // 所有引用的 task 都已在之前的调用里收集完毕——模型其实已经拿到这些结果。
        // 返回中性提示而非报错，让它停止重复等待、直接基于已有结果继续推理。
        return Ok(format!(
            "[task_wait] All {already_collected} referenced task(s) already completed and \
             their results were delivered by an earlier task_wait call. No tasks remain to \
             wait on; continue reasoning with the results you already collected."
        ));
    }

    let mut ready = Vec::new();
    let mut pending = Vec::new();
    // 收集本次调用中已完成（成功 / 失败、channel/futex 已销毁、需要从 registry
    // 删除）的 task_id；suspended 与 budget-elapsed 早返回路径也会用它清理。
    let mut finished: Vec<String> = Vec::new();
    // closure 默认按引用借用 wait_policy / registry / pending / ready / finished，
    // 不加 `move`，保证 closure 返回后外层 `if !pending.is_empty()` 等代码仍可访问。
    let wait_message = with_os_kernel(|os| {
        for tid in &task_ids {
            let entry = registry.get_ref(tid).expect("validated");
            // ⚠️ 这里之前曾按 `entry.started_at.elapsed() >= timeout_secs`
            // 直接把任务标记为 TIMEOUT 并销毁 channel/futex —— 这是 bug：
            // `started_at` 是 spawn 时间，不是本次 task_wait 的开始时间。如果
            // 主 agent 在 spawn 后 600s 才第一次调 task_wait，所有任务都会
            // **立刻** 被报为 TIMEOUT 且 result_channel 被销毁，subagent
            // 真实结果永久丢失，主 agent 自然会以为 "subagent 卡住"。
            //
            // 现在的做法：只看 channel 上有没有就绪 payload；如果还没有，统一
            // 走 pending 分支，让 epoll_wait_many 在本次调用 budget 内挂起。
            // 真正的"等待预算耗尽"只在 epoll_wait_many 的 wait.suspended /
            // 返回空 ready 时体现，并且 **绝不销毁 channel/futex**，主 agent
            // 可以继续调 task_wait 续等。
            if let Some(result) = read_task_result(os, entry.result_channel_id, true)? {
                ready.push(format_task_result(entry, result));
                let _ = os.channel_close(None, ChannelId(entry.result_channel_id));
                let _ = os.channel_release_named(
                    ChannelId(entry.result_channel_id),
                    "task_result.consumer",
                );
                let _ = os.channel_destroy(None, ChannelId(entry.result_channel_id));
                let _ = os.futex_destroy(entry.completion_futex_addr);
                finished.push(tid.clone());
            } else if is_task_pending(os, entry.pid)? {
                pending.push((tid.clone(), entry.pid));
            } else {
                // Process is no longer pending and never wrote a result.
                // Treat as failed-without-output and free the kernel
                // resources so we do not leak channels/futexes.
                // 子 agent 进程终止但未发布结果时，它不会运行自己的清理代码
                // 来释放 producer holder，因此这里必须同时释放 consumer 和 producer，
                // 否则 channel_destroy 因 ref_count != 0 失败，channel + futex 永久泄漏。
                let _ = os.channel_close(None, ChannelId(entry.result_channel_id));
                let _ = os.channel_release_named(
                    ChannelId(entry.result_channel_id),
                    "task_result.consumer",
                );
                let _ = os.channel_release_named(
                    ChannelId(entry.result_channel_id),
                    "task_result.producer",
                );
                let _ = os.channel_destroy(None, ChannelId(entry.result_channel_id));
                let _ = os.futex_destroy(entry.completion_futex_addr);
                ready.push(format!(
                    "[Task: {} via {} @ {}] FAILED: process pid={} terminated without publishing any output.",
                    entry.description, entry.agent_name, entry.model, entry.pid
                ));
                finished.push(tid.clone());
            }
        }

        if !pending.is_empty() {
            let pending_ids = pending
                .iter()
                .map(|(tid, _)| tid.clone())
                .collect::<Vec<_>>();
            let wait_sources = task_wait_sources(os, &pending_ids, &registry)?;
            // `task_wait` 的 `wait_policy=all` 是工具层语义：返回前要收齐所有
            // task 结果。但底层 park 不能用 `WaitPolicy::All` 等所有事件源，
            // 因为 sources 里还包含用于中断当前进程的 cancel futex，它在正常路径
            // 不会完成。这里等待“任一 task 事件”唤醒，再重新扫描所有 task 状态；
            // 若还没收齐，模型可用相同 task_ids 继续调用 task_wait。
            let wait = epoll_wait_many(
                os,
                &format!("task_wait:{}", pending_ids.join(",")),
                &wait_sources,
                WaitPolicy::Any,
                timeout_ticks,
            )?;
            // 无论 epoll_wait_many 是否 suspended，都先 re-scan 收集在等待期间
            // 变为就绪的结果。如果 suspended 且所有任务都已完成，直接返回结果
            // （而不是 PARKED），避免 wait_policy=all 时模型被迫反复调用 task_wait。
            // 仅当 re-scan 后仍有 pending 且确为 suspended 时才返回 PARKED。
            pending.clear();
            for tid in &pending_ids {
                let entry = registry.get_ref(tid).expect("validated after wait");
                if let Some(result) = read_task_result(os, entry.result_channel_id, true)? {
                    ready.push(format_task_result(entry, result));
                    let _ = os.channel_close(None, ChannelId(entry.result_channel_id));
                    let _ = os.channel_release_named(
                        ChannelId(entry.result_channel_id),
                        "task_result.consumer",
                    );
                    let _ = os.channel_destroy(None, ChannelId(entry.result_channel_id));
                    let _ = os.futex_destroy(entry.completion_futex_addr);
                    finished.push(tid.clone());
                } else if is_task_pending(os, entry.pid)? {
                    pending.push((tid.clone(), entry.pid));
                } else {
                    let _ = os.channel_close(None, ChannelId(entry.result_channel_id));
                    let _ = os.channel_release_named(
                        ChannelId(entry.result_channel_id),
                        "task_result.consumer",
                    );
                    let _ = os.channel_release_named(
                        ChannelId(entry.result_channel_id),
                        "task_result.producer",
                    );
                    let _ = os.channel_destroy(None, ChannelId(entry.result_channel_id));
                    let _ = os.futex_destroy(entry.completion_futex_addr);
                    ready.push(format!(
                        "[Task: {} via {} @ {}] FAILED: process pid={} terminated without publishing any output.",
                        entry.description, entry.agent_name, entry.model, entry.pid
                    ));
                    finished.push(tid.clone());
                }
            }
            // Re-scan 后仍有 pending 且确为 suspend（协作式让出，非预算耗尽），
            // 返回 PARKED 并附带已收集的部分结果。这里 **绝不能** 用
            // "BUDGET ELAPSED" 之类的终态措辞：suspend 是毫秒级同步返回的（不是
            // 真的等满 timeout_secs），否则模型会把"刚发起等待就超时"误判成
            // "子任务卡住"，从而提前放弃并转手动分析。
            if !pending.is_empty() && wait.suspended {
                let mut parts = Vec::new();
                if !ready.is_empty() {
                    parts.push(ready.join("\n\n---\n\n"));
                }
                let policy_label = match wait_policy {
                    WaitPolicy::Any => "any",
                    WaitPolicy::All => "all",
                };
                let still_pending: Vec<&str> =
                    pending.iter().map(|(tid, _)| tid.as_str()).collect();
                parts.push(format!(
                    "[task_wait PARKED] Yielded CPU so {} pending subagent task(s) can run. \
                    This is normal cooperative scheduling, NOT a timeout and NOT a stall — the wait budget \
                    ({timeout_secs}s, wait_policy={policy_label}) has NOT elapsed. The scheduler will wake this \
                    agent as soon as a result is ready. \
                    Pending task_ids: [{}]. event_ids={}. \
                    Do NOT assume the subagents are stuck and do NOT abandon them to work around this; \
                    when woken, re-call `task_wait` with the same task_ids to collect results, or use \
                    `task_status` for a non-blocking snapshot.",
                    still_pending.len(),
                    still_pending.join(", "),
                    wait.event_ids
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                return Ok(Some(parts.join("\n\n---\n\n")));
            }
        }
        Ok(None)
    })?;
    for tid in &finished {
        registry.remove(tid);
    }
    if let Some(message) = wait_message {
        return Ok(message);
    }
    if !pending.is_empty() {
        // Surface partial progress instead of dropping it on the floor.
        let mut parts = Vec::new();
        if !ready.is_empty() {
            parts.push(ready.join("\n\n---\n\n"));
        }
        let pending_ids: Vec<&str> = pending.iter().map(|(tid, _)| tid.as_str()).collect();
        let policy_label = match wait_policy {
            WaitPolicy::Any => "any",
            WaitPolicy::All => "all",
        };
        parts.push(format!(
            "[task_wait BUDGET ELAPSED] {} pending subagent task(s) still running in the background. \
            wait_policy={policy_label}, timeout_secs={timeout_secs}. The subagent(s) are NOT stalled and NOT cancelled; \
            their result channels and completion futexes remain alive. \
            Pending task_ids: [{}]. \
            Next steps: call `task_status` for a snapshot, or call `task_wait` again with the same task_ids to keep waiting \
            (consider `wait_policy=\"any\"` if you only need the first finisher).",
            pending.len(),
            pending_ids.join(", ")
        ));
        // 仅清理已经 ready 的 task_id 对应的 registry 条目；pending 任务必须保留，
        // 否则下次 task_wait 会因 "Unknown task_id" 失败。
        let pending_set: SkipSet<&str> = pending_ids.iter().copied().collect();
        for tid in &task_ids {
            if !pending_set.contains(&tid.as_str()) {
                registry.remove(tid);
            }
        }
        return Ok(parts.join("\n\n---\n\n"));
    }

    for tid in &task_ids {
        registry.remove(tid);
    }
    Ok(ready.join("\n\n---\n\n"))
}

fn params_task_status() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {}
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_status",
        description: "Show status of all asynchronously spawned tasks. Lists task_id, agent, model, and current state (running/completed/failed) without blocking. For tasks that have already finished, their output is included inline so you can use completed results immediately without calling task_wait.",
        parameters: params_task_status,
        execute: execute_task_status,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});

pub(crate) fn execute_task_status(_args: &Value) -> Result<String, String> {
    let registry = TASK_REGISTRY.lock().unwrap();
    if registry.is_empty() {
        return Ok("No async tasks currently tracked.".to_string());
    }

    let mut lines = vec![
        "TaskID              PID      Agent          Model          State       Description"
            .to_string(),
    ];
    // 对已经把结果写回 channel 的子任务，额外用 **非消费式 peek** 读出正文，附在
    // 表格后面。否则模型即使看到 state=completed，也只能回头再调 task_wait 才能拿到
    // 输出——而 task_wait 在协作让出时又只会回一条 PARKED 提示，形成"看得到完成、
    // 拿不到结果"的踢皮球，是诱发"子任务卡住"误判的次要原因。peek 不消费消息，
    // 后续 task_wait 仍能正常 consume 并清理资源。
    let mut completed_outputs: Vec<String> = Vec::new();
    with_os_kernel(|os| {
        for (tid, entry) in registry.iter() {
            let state_str = task_state_string(os, entry.result_channel_id, entry.pid)?;
            let short_id = if tid.len() > 19 { &tid[..19] } else { tid };
            lines.push(format!(
                "{:<19} {:<8} {:<14} {:<14} {:<11} {}",
                short_id, entry.pid, entry.agent_name, entry.model, state_str, entry.description
            ));
            if let Some(result) = read_task_result(os, entry.result_channel_id, false)? {
                completed_outputs.push(format_task_result(entry, result));
            }
        }
        Ok(())
    })?;

    if !completed_outputs.is_empty() {
        lines.push(String::new());
        lines.push(
            "Completed task results below (already available — no need to wait for these):"
                .to_string(),
        );
        lines.push(completed_outputs.join("\n\n---\n\n"));
    }

    Ok(lines.join("\n"))
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StoredTaskResult {
    pub(crate) status: String,
    pub(crate) output: String,
    pub(crate) error: Option<String>,
}

fn read_task_result(
    os: &mut dyn Kernel,
    result_channel_id: u64,
    consume: bool,
) -> Result<Option<StoredTaskResult>, String> {
    let payload = match if consume {
        os.channel_try_recv(None, ChannelId(result_channel_id))
    } else {
        os.channel_peek(None, ChannelId(result_channel_id))
    }? {
        IpcRecvResult::Message(payload) => payload,
        IpcRecvResult::Empty | IpcRecvResult::Closed => return Ok(None),
    };
    serde_json::from_str(&payload).map(Some).map_err(|err| {
        format!(
            "Failed to decode stored task result from channel {}: {}",
            result_channel_id, err
        )
    })
}

fn task_wait_sources(
    os: &mut dyn Kernel,
    task_ids: &[String],
    registry: &SkipMap<String, AsyncTaskEntry>,
) -> Result<Vec<WaitManySource>, String> {
    let mut sources = Vec::new();
    for tid in task_ids {
        let entry = registry
            .get_ref(tid)
            .ok_or_else(|| format!("Unknown task_id: {}", tid))?;
        sources.extend(wait_sources_for_channel_and_futex(
            os,
            entry.result_channel_id,
            Some(entry.completion_futex_addr),
        )?);
    }
    append_current_process_cancel_source(os, &mut sources)?;
    Ok(sources)
}

fn is_task_pending(os: &mut dyn Kernel, pid: u64) -> Result<bool, String> {
    let Some(proc) = os.get_process(pid) else {
        return Ok(false);
    };
    Ok(matches!(
        proc.state,
        ProcessState::Ready
            | ProcessState::Running
            | ProcessState::Waiting { .. }
            | ProcessState::Sleeping { .. }
    ))
}

fn task_state_string(
    os: &mut dyn Kernel,
    result_channel_id: u64,
    pid: u64,
) -> Result<String, String> {
    if let Some(result) = read_task_result(os, result_channel_id, false)? {
        return Ok(result.status);
    }
    let state = match os.get_process(pid) {
        Some(proc) => match proc.state {
            ProcessState::Ready => "ready",
            ProcessState::Running => "running",
            ProcessState::Waiting { .. } => "waiting",
            ProcessState::Sleeping { .. } => "sleeping",
            ProcessState::Stopped => "stopped",
            ProcessState::Terminated => "terminated",
        },
        None => "unknown",
    };
    Ok(state.to_string())
}

fn format_task_result(entry: &AsyncTaskEntry, result: StoredTaskResult) -> String {
    let duration_secs = entry.started_at.elapsed().as_secs_f64();
    let mut parts = vec![format!(
        "[Task: {} via {} @ {}] {} after {:.1}s",
        entry.description,
        entry.agent_name,
        entry.model,
        result.status.to_uppercase(),
        duration_secs
    )];
    parts.push(entry.selection_explanation.clone());
    if let Some(error) = result.error
        && !error.trim().is_empty()
    {
        parts.push(format!("Error: {}", error));
    }
    if !result.output.trim().is_empty() {
        parts.push(result.output.trim().to_string());
    } else {
        parts.push("(subagent did not produce any final assistant text)".to_string());
    }
    parts.join("\n")
}

fn subagent_document_text(agent: &AgentManifest) -> String {
    let mut parts = vec![agent.name.clone(), agent.description.clone()];
    if !agent.routing_tags.is_empty() {
        parts.push(agent.routing_tags.join(" "));
    }
    if !agent.prompt.trim().is_empty() {
        parts.push(agent.prompt.chars().take(1500).collect());
    }
    normalize_text_for_similarity(&parts.join("\n"))
}

fn auto_subagent_score(
    agent: &AgentManifest,
    task_text: &str,
    idf: &FxHashMap<String, f64>,
) -> f64 {
    let query = TextSimilarityFeatures::from_text(task_text);
    let doc = TextSimilarityFeatures::from_text(&subagent_document_text(agent));
    cosine_tfidf_similarity(&query.ngram_tf, &doc.ngram_tf, idf)
}

#[derive(Debug)]
struct SelectedSubagent<'a> {
    agent: &'a AgentManifest,
    auto_selected: bool,
    matched_tags: Vec<String>,
    score: i32,
}

fn matched_routing_tags(agent: &AgentManifest, task_text: &str) -> Vec<String> {
    let task = task_text.to_ascii_lowercase();
    agent
        .routing_tags_normalized()
        .into_iter()
        .filter(|tag| task.contains(tag.as_str()))
        .collect()
}

fn select_subagent<'a>(
    all_agents: &'a [AgentManifest],
    requested_agent: Option<&str>,
    description: &str,
    prompt: &str,
) -> Result<SelectedSubagent<'a>, String> {
    let subagents = agents::get_subagents(all_agents);
    if subagents.is_empty() {
        return Err(
            "No subagents are available. Add at least one agent with mode: subagent or all."
                .to_string(),
        );
    }

    if let Some(requested) = requested_agent {
        if let Some(agent) = subagents
            .iter()
            .copied()
            .find(|agent| agent.name.eq_ignore_ascii_case(requested))
        {
            return Ok(SelectedSubagent {
                agent,
                auto_selected: false,
                matched_tags: Vec::new(),
                score: 0,
            });
        }

        if let Some(agent) = agents::find_agent_by_name(all_agents, requested) {
            return Err(format!(
                "Agent '{}' exists but is not a subagent. Use a subagent or omit the agent field for auto-selection.",
                agent.name
            ));
        }

        let available = subagents
            .iter()
            .map(|agent| agent.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "Unknown subagent '{}'. Available subagents: {}",
            requested, available
        ));
    }

    let task_text = format!("{description}\n{prompt}");
    let doc_tfs: Vec<FxHashMap<String, f64>> = subagents
        .iter()
        .map(|agent| TextSimilarityFeatures::from_text(&subagent_document_text(agent)).ngram_tf)
        .collect();
    let doc_refs: Vec<&FxHashMap<String, f64>> = doc_tfs.iter().collect();
    let idf = build_idf_from_documents(&doc_refs);

    subagents
        .into_iter()
        .max_by(|a, b| {
            auto_subagent_score(a, &task_text, &idf)
                .total_cmp(&auto_subagent_score(b, &task_text, &idf))
                .then_with(|| b.name.cmp(&a.name))
        })
        .map(|agent| {
            let score = auto_subagent_score(agent, &task_text, &idf);
            SelectedSubagent {
                agent,
                auto_selected: true,
                matched_tags: matched_routing_tags(agent, &task_text),
                score: (score * 100.0) as i32,
            }
        })
        .ok_or_else(|| "No subagents are available.".to_string())
}

fn format_agent_model_tier(agent: &AgentManifest) -> &'static str {
    match agent.model_tier {
        Some(AgentModelTier::Light) => "light",
        Some(AgentModelTier::Standard) | None => "standard",
        Some(AgentModelTier::Heavy) => "heavy",
    }
}

fn format_quality_tier(tier: crate::ai::provider::ModelQualityTier) -> &'static str {
    match tier {
        crate::ai::provider::ModelQualityTier::Basic => "basic",
        crate::ai::provider::ModelQualityTier::Standard => "standard",
        crate::ai::provider::ModelQualityTier::Strong => "strong",
        crate::ai::provider::ModelQualityTier::Flagship => "flagship",
    }
}

fn format_provider(provider: crate::ai::provider::ApiProvider) -> &'static str {
    match provider {
        crate::ai::provider::ApiProvider::Alibaba => "alibaba",
        crate::ai::provider::ApiProvider::Compatible => "compatible",
        crate::ai::provider::ApiProvider::OpenAi => "openai",
        _ => "opencode",
    }
}

fn build_selection_explanation(
    selected: &SelectedSubagent<'_>,
    selected_model: &str,
    model_override: Option<&str>,
    inherited_parent_model: bool,
) -> String {
    let agent_reason = if selected.auto_selected {
        if selected.matched_tags.is_empty() {
            "agent_reason=auto-selected as the best available subagent".to_string()
        } else {
            format!(
                "agent_reason=auto-selected by routing_tags [{}] (score={})",
                selected.matched_tags.join(", "),
                selected.score
            )
        }
    } else {
        "agent_reason=explicit agent override".to_string()
    };

    let model_reason = if model_override
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .is_some()
    {
        "model_reason=explicit model override".to_string()
    } else if inherited_parent_model {
        "model_reason=inherited parent agent current model".to_string()
    } else {
        format!(
            "model_reason=auto-selected for agent_tier={} using {} provider and {} quality_tier",
            format_agent_model_tier(selected.agent),
            format_provider(models::model_provider(selected_model)),
            format_quality_tier(models::model_quality_tier(selected_model))
        )
    };

    format!("{agent_reason}\n{model_reason}")
}

mod agent_team;

use agent_team::execute_agent_team;
#[cfg(test)]
pub(crate) use agent_team::{
    AgentTeamMemberSpec, AgentTeamOperation, build_agent_team_prompt,
    build_agent_team_selection_prompt, parse_agent_team_members, resolve_agent_team_model_override,
};

#[cfg(test)]
mod tests;
