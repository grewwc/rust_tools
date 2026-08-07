use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use crate::ai::tools::os_tools::GLOBAL_OS;
use crate::ai::{
    agents::{self, AgentManifest, AgentModelTier},
    models,
    tools::common::{
        ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
    },
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
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const MAX_TASK_REGISTRY_SIZE: usize = 100;
const DEFAULT_TASK_PRIORITY: u8 = 20;
const DEFAULT_TASK_QUOTA_TURNS: usize = 10;
/// 子代理最大嵌套深度。depth=1 是顶层 agent 直接 spawn 的子代理。
/// 子代理不允许继续 spawn 孙代理，避免递归扇出与结果无人收集。
pub(crate) const MAX_SUBAGENT_SPAWN_DEPTH: usize = 1;
/// Subagent 是父 agent 的叶子取证/执行单元，不应继承主 agent 的完整长循环预算。
/// 主 agent 仍保留自身 max_steps；这里只有 `task` / `task_spawn` 启动路径会钳制。
pub(crate) const SUBAGENT_MAX_ITERATIONS: usize = 32;
/// 显式声明 `max_steps` 的 agent（如深度审计 `/audit`）可以突破默认 32 轮，
/// 但任何子代理都不能超过这个绝对硬帽，防止失控子代理无限迭代。
pub(crate) const SUBAGENT_MAX_ITERATIONS_HARD_CAP: usize = 256;
/// 单次批量委派的硬上限，与后台调度默认最大批次保持一致。
/// schema 与执行入口都校验，避免绕过父级单次 tool-call 配额造成无界扇出。
const MAX_SUBAGENT_SPAWN_BATCH_SIZE: usize = 8;
const TASK_GOAL_PREFIX: &str = "AIOS_SUBAGENT_TASK:";
/// 子代理结果只是主 agent 的证据输入，不是最终对用户的直接回答。
/// 主 agent 拿到 payload 后仍需自行汇总结论、风险与下一步，再面向用户输出。
pub(crate) const SUBAGENT_PARENT_SUMMARY_REMINDER: &str = "Parent-agent follow-up: summarize the confirmed subagent conclusions in your own response to the user. Do not rely on the raw subagent transcript or terminal fold as the final user-facing answer.";
/// 单次 `task_wait` 调用的默认等待预算（秒）。这只是 **本次调用的最长阻塞时间**，
/// 不是 subagent 的总寿命：超时仅意味着"这次没等到结果"，主 agent 可以继续调
/// `task_wait` 续等，subagent 仍在后台运行，channel/futex 也不会被销毁。
///
/// 前台等待只提供短暂的收集窗口；长时间运行的子任务应与父 agent 的独立工作重叠，
/// 而不是让父 agent 在 task_spawn 后立即挂起数分钟。
const DEFAULT_TASK_WAIT_TIMEOUT_SECS: u64 = 30;
/// `task_wait.timeout_secs` 的硬上限，避免模型把 timeout 设成天文数字时彻底
/// 阻塞 driver。最长 60 秒后必须把控制权还给父 agent，重新评估是继续本地工作、
/// 非阻塞查看状态，还是确实需要再次等待。
const MAX_TASK_WAIT_TIMEOUT_SECS: u64 = 60;

/// Subagent 的 wall-clock 总寿命上限。与单次 task_wait 的 `timeout_secs`（默认
/// 30s、上限 60s）不同，这是进程级硬上限：subagent 存活超过此值（典型如卡在单个永不
/// 返回的工具执行里、单 turn 内无 wall-clock 超时），task_wait 入口会主动
/// 终止它并写入 timeout 终态结果，避免主 agent 陷入"超时->续等->再超时"空转
/// 或后台进程永久占用资源。1 小时远大于正常完成时长，仅在真正卡死时兜底。
const SUBAGENT_WALL_CLOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60 * 60);

/// Granular control over which slices of the parent agent's execution
/// context are inherited by a spawned sub-agent. Defaults are cwd+skills=true
/// and history+memory=false unless the caller specifies an `inherit` argument
/// on the tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InheritOptions {
    pub(crate) history: bool,
    pub(crate) memory: bool,
    pub(crate) cwd: bool,
    pub(crate) skills: bool,
}

impl Default for InheritOptions {
    fn default() -> Self {
        // 默认只继承执行所需的 cwd/skills，不继承整段对话历史与 memory。
        // 窄任务由父 agent 在 prompt 中显式传入必要上下文，避免 token 膨胀、注意力偏移，
        // 以及 sub-agent 直接污染主 memory 文件。调用方仍可显式传 `inherit: "all"`
        // 或 `inherit: "history,cwd,skills"` 退回旧行为。
        Self {
            history: false,
            memory: false,
            cwd: true,
            skills: true,
        }
    }
}

impl InheritOptions {
    /// Parse the optional `inherit` field from a tool call.
    /// Recognised forms:
    ///   - missing / null -> default (cwd+skills, history/memory private)
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
/// 进程；结果必须先由 `task_wait` / `task_status` 持久化到 evidence ledger，
/// 再移除注册表条目和 IPC 资源。容量满时拒绝新任务，绝不驱逐未收取结果。
pub(crate) struct AsyncTaskEntry {
    pub(crate) session_id: String,
    pub(crate) result_observed: bool,
    /// 直接拥有该 task 的父进程 pid。task_wait/status/cancel 只允许 owner
    /// 进程观察自己 spawn 的子任务，避免同一 session 内父/兄弟任务互相污染。
    pub(crate) owner_pid: u64,
    /// 与 kernel `Process.pid` 一致；agent 端额外保存便于通过 task_id 反查 pid。
    pub(crate) pid: u64,
    pub(crate) result_channel_id: u64,
    pub(crate) completion_futex_addr: FutexAddr,
    /// 描述性文本；与 kernel `Process.goal` 不同——后者会带 TASK_GOAL_PREFIX
    /// 前缀和完整 prompt。
    pub(crate) description: String,
    /// 子 agent 的逻辑名（用于查找注册的 AgentManifest，如 `"build"`）；与 kernel
    /// `Process.name` 同源但 kernel 端 name 仅作显示。注意区分：`plan` 是工具名，
    /// 不是 agent 名（仓库内未注册 `plan` subagent），不要将 `agent_name` 填为
    /// `"plan"`——会把派名指向不存在的 manifest。
    pub(crate) agent_name: String,
    pub(crate) model: String,
    pub(crate) is_model_auto_selected: bool,
    pub(crate) auto_model_fallback: Option<models::AutoModelFallbackSpec>,
    pub(crate) selection_explanation: String,
    pub(crate) inherit: InheritOptions,
    /// 真实 Tokio 子任务的取消句柄。kernel process 终止时必须同步 abort，
    /// 否则网络请求或工具 Future 仍会在后台继续运行。
    pub(crate) abort_handle: Option<tokio::task::AbortHandle>,
    /// 与子代理 App 共享的取消标志。同步执行中的命令无法被 Tokio abort 立即打断，
    /// 因此 timeout/cancel 必须先置位，让命令 runner 杀掉实际 OS 进程组。
    pub(crate) cancel_stream: Arc<AtomicBool>,
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
static TASK_PROGRESS_REGISTRY: LazyLock<
    Mutex<SkipMap<String, crate::ai::driver::runtime_ctx::SubagentPhaseSlot>>,
> = LazyLock::new(|| Mutex::new(SkipMap::default()));
static TASK_RETRY_REGISTRY: LazyLock<Mutex<SkipMap<String, RetryableTaskSpec>>> =
    LazyLock::new(|| Mutex::new(SkipMap::default()));

#[derive(Clone)]
struct RetryableTaskSpec {
    session_id: String,
    owner_pid: u64,
    prepared: PreparedSubagentTask,
    retry_root: String,
    terminal_status: Option<String>,
    recorded_at: Instant,
}

pub(crate) fn task_progress_slot(
    task_id: &str,
) -> Option<crate::ai::driver::runtime_ctx::SubagentPhaseSlot> {
    TASK_PROGRESS_REGISTRY
        .lock()
        .ok()?
        .get_ref(&task_id.to_string())
        .cloned()
}

fn current_task_progress(task_id: &str) -> Option<String> {
    let slot = task_progress_slot(task_id)?;
    let value = crate::ai::driver::runtime_ctx::subagent_progress_snapshot(&slot)?;
    (!value.is_empty()).then_some(value)
}

fn register_retry_spec(
    task_id: &str,
    session_id: String,
    owner_pid: u64,
    prepared: PreparedSubagentTask,
    retry_root: String,
) {
    let mut registry = TASK_RETRY_REGISTRY.lock().unwrap();
    while registry.len() >= MAX_TASK_REGISTRY_SIZE {
        let Some(oldest) = registry
            .iter()
            .min_by_key(|(_, spec)| spec.recorded_at)
            .map(|(id, _)| id.clone())
        else {
            break;
        };
        registry.remove(&oldest);
    }
    registry.insert(
        task_id.to_string(),
        RetryableTaskSpec {
            session_id,
            owner_pid,
            prepared,
            retry_root,
            terminal_status: None,
            recorded_at: Instant::now(),
        },
    );
}

fn mark_task_retry_status(task_id: &str, status: &str) {
    if let Some(spec) = TASK_RETRY_REGISTRY
        .lock()
        .unwrap()
        .get_mut(&task_id.to_string())
    {
        spec.terminal_status = Some(status.to_string());
    }
}

fn is_retryable_task_status(status: &str) -> bool {
    matches!(status, "failed" | "timeout" | "cancelled")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TaskWaitPolicyKey {
    Any,
    All,
}

impl From<&WaitPolicy> for TaskWaitPolicyKey {
    fn from(value: &WaitPolicy) -> Self {
        match value {
            WaitPolicy::Any => Self::Any,
            WaitPolicy::All => Self::All,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TaskWaitKey {
    session_id: String,
    owner_pid: u64,
    wait_policy: TaskWaitPolicyKey,
    task_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct TaskWaitState {
    deadline: Instant,
    timeout_secs: u64,
    expired: bool,
}

static TASK_WAIT_STATES: LazyLock<Mutex<FxHashMap<TaskWaitKey, TaskWaitState>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

const OUTSTANDING_SUBAGENT_TASKS_NOTE_PREFIX: &str = "[pending-subagent-tasks]";

/// 最近一次 `task_spawn` / `task_spawn_batch` 成功产生的任务 id 列表，用于检测
/// "lone spawn" 反模式：spawn 单个任务后立即 `task_wait` 收集它。该场景没有并发
/// 收益，应该用同步 `task` 工具（spawn + wait 只会更慢）。提示是轻量规范引导：
/// 只提示一次、绝不拒绝或阻塞，模型可忽略（例如它确实在 spawn 与 wait 之间插入了
/// 父侧工作）。
struct LastSpawnBatch {
    task_ids: Vec<String>,
    hinted: bool,
}
static LAST_SPAWN_BATCH: LazyLock<Mutex<Option<LastSpawnBatch>>> =
    LazyLock::new(|| Mutex::new(None));

fn record_last_spawn_batch(task_ids: Vec<String>) {
    *LAST_SPAWN_BATCH.lock().unwrap() = Some(LastSpawnBatch {
        task_ids,
        hinted: false,
    });
}

/// 若本次 wait 的 task_ids 命中"最近一次 spawn 是单个任务"且尚未提示过，
/// 返回一次规范提示文本（消费 hinted 标志，保证整轮会话只提示一次）。
fn lone_spawn_hint_note(waited_task_ids: &[String]) -> Option<String> {
    let mut guard = match LAST_SPAWN_BATCH.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let record = guard.as_mut()?;
    if record.hinted || record.task_ids.len() != 1 {
        return None;
    }
    let sole_id = &record.task_ids[0];
    if !waited_task_ids.contains(sole_id) {
        return None;
    }
    record.hinted = true;
    Some(
        "[tool_followup:lone_spawn]\n\
         This task_wait collects the only task spawned by the most recent task_spawn call. \
         Spawning a single task and waiting on it adds no concurrency: for one-task handoffs use \
         the synchronous `task` tool instead of task_spawn + task_wait. \
         (If you intentionally ran parent-side work between spawn and wait, ignore this hint.)"
            .to_string(),
    )
}

#[cfg(test)]
pub(crate) fn reset_last_spawn_batch_for_test() {
    *LAST_SPAWN_BATCH.lock().unwrap() = None;
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OutstandingTaskSnapshot {
    task_id: String,
    status: String,
    agent_name: String,
    model: String,
    description: String,
}

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
    /// 子代理嵌套深度：顶层 spawn 为 1，逐层递增。用于防止递归扇出。
    #[serde(default)]
    pub(crate) spawn_depth: usize,
    /// 可选的子代理最终响应 JSON Schema；旧任务载荷缺失时保持兼容。
    #[serde(default)]
    pub(crate) response_schema: Option<Value>,
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

fn current_task_owner_pid() -> Result<u64, String> {
    with_os_kernel(|os| {
        os.current_process_id()
            .ok_or("task orchestration requires an active AIOS process context.".to_string())
    })
}

fn current_task_owner_pid_opt() -> Option<u64> {
    with_os_kernel(|os| Ok(os.current_process_id()))
        .ok()
        .flatten()
}

fn active_foreground_owner_pid(os: &mut dyn Kernel) -> Option<u64> {
    if let Some(pid) = os.current_process_id()
        && os.get_process(pid).is_some_and(|proc| proc.is_foreground)
    {
        return Some(pid);
    }
    os.list_processes()
        .into_iter()
        .find(|proc| proc.is_foreground && !matches!(proc.state, ProcessState::Terminated))
        .map(|proc| proc.pid)
}

fn task_entry_owned_by(entry: &AsyncTaskEntry, session_id: &str, owner_pid: u64) -> bool {
    entry.session_id == session_id && entry.owner_pid == owner_pid
}

fn task_wait_key(
    session_id: &str,
    owner_pid: u64,
    wait_policy: &WaitPolicy,
    task_ids: &[String],
) -> TaskWaitKey {
    let mut normalized = task_ids.to_vec();
    normalized.sort();
    normalized.dedup();
    TaskWaitKey {
        session_id: session_id.to_string(),
        owner_pid,
        wait_policy: wait_policy.into(),
        task_ids: normalized,
    }
}

fn load_or_create_task_wait_state(key: &TaskWaitKey, timeout_secs: u64) -> TaskWaitState {
    let now = Instant::now();
    let mut states = TASK_WAIT_STATES.lock().unwrap();
    let mut inserted = false;
    let state = states.entry(key.clone()).or_insert_with(|| {
        inserted = true;
        TaskWaitState {
            deadline: now + Duration::from_secs(timeout_secs),
            timeout_secs,
            expired: false,
        }
    });
    if now >= state.deadline {
        state.expired = true;
    }
    let state = *state;
    drop(states);
    if inserted {
        crate::ai::driver::notify_scheduler_after(Duration::from_secs(timeout_secs));
    }
    state
}

fn clear_task_wait_state(key: &TaskWaitKey) {
    let mut states = TASK_WAIT_STATES.lock().unwrap();
    states.remove(key);
}

#[cfg(test)]
pub(crate) fn expire_task_wait_states_for_test() {
    let mut states = TASK_WAIT_STATES.lock().unwrap();
    let expired_at = Instant::now() - Duration::from_secs(1);
    for state in states.values_mut() {
        state.deadline = expired_at;
        state.expired = false;
    }
}

pub(crate) fn wake_expired_task_waits() {
    let expired = {
        let now = Instant::now();
        let mut states = TASK_WAIT_STATES.lock().unwrap();
        states
            .iter_mut()
            .filter_map(|(key, state)| {
                if state.expired || now < state.deadline {
                    return None;
                }
                state.expired = true;
                Some((key.clone(), *state))
            })
            .collect::<Vec<_>>()
    };
    if expired.is_empty() {
        return;
    }

    let _ = with_os_kernel(|os| {
        let mut woken: SkipSet<u64> = SkipSet::default();
        for (key, state) in expired {
            if !woken.insert(key.owner_pid) {
                continue;
            }
            let task_ids = key.task_ids.join(", ");
            let _ = os.wake_process(
                key.owner_pid,
                format!(
                    "[TASK_WAIT_TIMEOUT]\nWall-clock task_wait budget elapsed after {}s. Re-call `task_wait` with the same task_ids to collect any ready results and receive the budget-elapsed status. task_ids=[{}]",
                    state.timeout_secs, task_ids
                ),
            );
        }
        Ok(())
    });
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

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task",
        description: "",

        execute: execute_task,
        groups: &["builtin", "core"],
    }
});

// `task` / `task_wait` / `task_status` 都可能承载 subagent 的唯一可见结果。
// 这些结果一旦被有损压缩或 LLM prune，主 agent 就可能失去对已完成子任务的
// grounding 感知。统一禁止 lossy 与 prune；若内容过大，交给 overflow stub +
// file_path 承接，而不是删成不可复原的摘要。
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "task",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Never,
        counts_toward_precision_inline_budget: false,
    },
});

pub(crate) fn execute_task(_args: &Value) -> Result<String, String> {
    Err("task is handled by the runtime".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_spawn",
        description: "",

        execute: execute_task_spawn,
        groups: &["builtin", "core"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_spawn_batch",
        description: "",

        execute: execute_task_spawn_batch,
        groups: &["builtin", "core"],
    }
});

/// Pre-flight subagent task spec produced from a `task` / `task_spawn` tool
/// call before the kernel actually spawns the new process.
#[derive(Clone)]
pub(crate) struct PreparedSubagentTask {
    pub(crate) description: String,
    pub(crate) prompt: String,
    pub(crate) response_schema: Option<Value>,
    pub(crate) agent_name: String,
    pub(crate) model: String,
    pub(crate) is_model_auto_selected: bool,
    pub(crate) auto_model_fallback: Option<models::AutoModelFallbackSpec>,
    pub(crate) selection_explanation: String,
    pub(crate) inherit: InheritOptions,
}

pub(in crate::ai) fn capped_subagent_manifest(agent: &AgentManifest) -> AgentManifest {
    let mut capped = agent.clone();
    let max_steps = agent
        .max_steps
        .unwrap_or(SUBAGENT_MAX_ITERATIONS)
        .min(SUBAGENT_MAX_ITERATIONS_HARD_CAP)
        .max(1);
    capped.max_steps = Some(max_steps);
    capped
}

fn wrap_subagent_prompt(
    description: &str,
    prompt: &str,
    response_schema: Option<&Value>,
) -> String {
    let response_contract = response_schema
        .map(|schema| {
            let schema =
                serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());
            format!(
                "Required response contract:\n\
                 - Return exactly one JSON value matching the schema below.\n\
                 - Do not wrap the JSON in Markdown fences or add prose before or after it.\n\
                 <response_schema>\n{schema}\n</response_schema>\n\n"
            )
        })
        .unwrap_or_default();
    format!(
        "Subagent task: {}\n\n\
         Runtime constraints:\n\
         - Treat this as a bounded leaf task for the parent agent. Do not expand scope beyond the task.\n\
         - Reuse observed evidence and avoid equivalent read/search/list/command variants unless omitted text is needed; prefer one targeted broad call over many small ones.\n\
         - Ground factual claims in observed evidence. For review or diagnosis, trace the relevant path and check likely counter-evidence before reporting a finding.\n\
         - If evidence is incomplete, return a concise partial result separating confirmed conclusions, unresolved hypotheses, missing evidence, and the next verification step.\n\n\
         {}Parent task prompt:\n{}",
        description.trim(),
        response_contract,
        prompt.trim()
    )
}

fn parse_response_schema(args: &Value) -> Result<Option<Value>, String> {
    let Some(schema) = args.get("response_schema") else {
        return Ok(None);
    };
    if schema.is_null() {
        return Ok(None);
    }
    if !schema.is_object() {
        return Err("'response_schema' must be a JSON Schema object".to_string());
    }
    jsonschema::validator_for(schema)
        .map_err(|error| format!("Invalid 'response_schema': {error}"))?;
    Ok(Some(schema.clone()))
}

pub(crate) fn validate_subagent_response(
    response_schema: Option<&Value>,
    output: &str,
) -> Result<(), String> {
    let Some(schema) = response_schema else {
        return Ok(());
    };
    let instance: Value = serde_json::from_str(output.trim()).map_err(|error| {
        format!("Subagent response is not valid JSON required by response_schema: {error}")
    })?;
    let validator = jsonschema::validator_for(schema)
        .map_err(|error| format!("Invalid response_schema: {error}"))?;
    validator
        .validate(&instance)
        .map_err(|error| format!("Subagent response did not match response_schema: {error}"))
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
    let response_schema = parse_response_schema(args)?;

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
        prompt: wrap_subagent_prompt(description, prompt, response_schema.as_ref()),
        response_schema,
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
    spawn_subagent_kernel_task_attempt(prepared, None)
}

fn spawn_subagent_kernel_task_attempt(
    prepared: &PreparedSubagentTask,
    retry_of: Option<&str>,
) -> Result<SpawnedSubagentTask, String> {
    let parent_depth = crate::ai::driver::runtime_ctx::current_subagent_depth();
    let child_depth = parent_depth + 1;
    if child_depth > MAX_SUBAGENT_SPAWN_DEPTH {
        return Err(format!(
            "Subagent nesting depth {} exceeds maximum {}. \
             The current agent is already a nested subagent; further delegation \
             would risk unbounded recursion. Execute the work directly instead.",
            child_depth, MAX_SUBAGENT_SPAWN_DEPTH,
        ));
    }
    {
        let registry = TASK_REGISTRY.lock().unwrap();
        if registry.len() >= MAX_TASK_REGISTRY_SIZE {
            return Err(format!(
                "Subagent task registry is full ({MAX_TASK_REGISTRY_SIZE}). \
                 Collect and integrate existing task results before spawning another task."
            ));
        }
    }
    let task_id = next_task_id();
    let (owner_pid, pid, result_channel_id, completion_futex_addr) = with_os_kernel(|os| {
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
            spawn_depth: child_depth,
            response_schema: prepared.response_schema.clone(),
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
        Ok((parent_pid, pid, result_channel.raw(), completion_futex))
    })?;

    {
        let mut registry = TASK_REGISTRY.lock().unwrap();
        registry.insert(
            task_id.clone(),
            AsyncTaskEntry {
                session_id: crate::ai::driver::runtime_ctx::current_session_id_or_empty(),
                result_observed: false,
                owner_pid,
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
                abort_handle: None,
                cancel_stream: Arc::new(AtomicBool::new(false)),
            },
        );
    }
    TASK_PROGRESS_REGISTRY.lock().unwrap().insert(
        task_id.clone(),
        crate::ai::driver::runtime_ctx::new_subagent_progress_slot(),
    );
    register_retry_spec(
        &task_id,
        crate::ai::driver::runtime_ctx::current_session_id_or_empty(),
        owner_pid,
        prepared.clone(),
        retry_of.unwrap_or(&task_id).to_string(),
    );
    crate::ai::driver::notify_scheduler_after(SUBAGENT_WALL_CLOCK_TIMEOUT);

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

/// 关联实际执行子代理的 Tokio task，使取消和超时能够停止后台 Future，
/// 而不只是终止 kernel 中的逻辑进程。
pub(crate) fn set_task_abort_handle(task_id: &str, abort_handle: tokio::task::AbortHandle) -> bool {
    let mut registry = TASK_REGISTRY.lock().unwrap();
    let Some(entry) = registry.get_mut(&task_id.to_string()) else {
        return false;
    };
    entry.abort_handle = Some(abort_handle);
    true
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

/// 前台状态栏使用的只读快照。只暴露展示所需字段；subagent 正文仍只通过
/// `task_wait` / `task_status` 返回，避免后台任务争用 terminal。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubagentTerminalStatus {
    pub(crate) description: String,
    pub(crate) agent_name: String,
    pub(crate) state: String,
    pub(crate) elapsed_secs: u64,
    pub(crate) progress: Option<String>,
}

pub(crate) fn subagent_terminal_statuses(
    os: &mut dyn Kernel,
    session_id: &str,
) -> Vec<SubagentTerminalStatus> {
    let Some(owner_pid) = active_foreground_owner_pid(os) else {
        return Vec::new();
    };
    let registry = TASK_REGISTRY.lock().unwrap();
    let mut statuses = registry
        .iter()
        .filter(|(_, entry)| task_entry_owned_by(entry, session_id, owner_pid))
        .map(|(task_id, entry)| SubagentTerminalStatus {
            description: entry.description.clone(),
            agent_name: entry.agent_name.clone(),
            state: task_state_string(os, entry.result_channel_id, entry.pid)
                .unwrap_or_else(|_| "unknown".to_string()),
            elapsed_secs: entry.started_at.elapsed().as_secs(),
            progress: current_task_progress(task_id),
        })
        .collect::<Vec<_>>();
    statuses.sort_by(|a, b| a.description.cmp(&b.description));
    statuses
}

/// Remove a task entry from the registry. Called by the synchronous `task`
/// interception once it has consumed the result.
pub(crate) fn remove_task_entry(task_id: &str) -> Option<AsyncTaskEntry> {
    let mut registry = TASK_REGISTRY.lock().unwrap();
    registry.take(&task_id.to_string())
}

#[cfg(test)]
pub(crate) fn insert_task_entry_for_test(task_id: String, entry: AsyncTaskEntry) {
    let mut registry = TASK_REGISTRY.lock().unwrap();
    registry.insert(task_id, entry);
}
/// channel. Re-exported for the synchronous `task` interception so that both
/// paths produce identical output.
pub(crate) fn format_finished_task(entry: &AsyncTaskEntry, result: StoredTaskResult) -> String {
    format_task_result(entry, result)
}

pub(crate) fn execute_task_spawn(args: &Value) -> Result<String, String> {
    ensure_top_level_task_orchestration("task_spawn")?;
    let prepared = prepare_subagent_task(args)?;
    let spawned = spawn_subagent_kernel_task(&prepared)?;
    record_last_spawn_batch(vec![spawned.task_id.clone()]);

    Ok(format!(
        "Task spawned: task_id={}, pid={}, agent={}, model={}, inherit={}\nContinue independent parent-side work now. Do not call task_wait immediately unless the parent is blocked on this result; use task_status for a non-blocking snapshot.",
        spawned.task_id,
        spawned.pid,
        prepared.agent_name,
        prepared.model,
        prepared.inherit.describe()
    ))
}

pub(crate) fn execute_task_spawn_batch(args: &Value) -> Result<String, String> {
    ensure_top_level_task_orchestration("task_spawn_batch")?;
    let tasks = args
        .get("tasks")
        .and_then(Value::as_array)
        .ok_or_else(|| "task_spawn_batch requires a 'tasks' array".to_string())?;
    if tasks.is_empty() {
        return Err("task_spawn_batch requires at least one task".to_string());
    }
    if tasks.len() > MAX_SUBAGENT_SPAWN_BATCH_SIZE {
        return Err(format!(
            "task_spawn_batch accepts at most {MAX_SUBAGENT_SPAWN_BATCH_SIZE} tasks per call"
        ));
    }

    // 先完成整批 preflight，避免后续条目参数无效时前面的 child 已经启动。
    let prepared = tasks
        .iter()
        .enumerate()
        .map(|(index, task)| {
            prepare_subagent_task(task)
                .map_err(|error| format!("task_spawn_batch tasks[{index}]: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut spawned_count = 0usize;
    let mut entries = Vec::with_capacity(prepared.len());
    let mut spawned_ids = Vec::with_capacity(prepared.len());
    for (index, task) in prepared.iter().enumerate() {
        match spawn_subagent_kernel_task(task) {
            Ok(spawned) => {
                spawned_count += 1;
                spawned_ids.push(spawned.task_id.clone());
                entries.push(serde_json::json!({
                    "index": index,
                    "status": "spawned",
                    "task_id": spawned.task_id,
                    "pid": spawned.pid,
                    "agent": task.agent_name,
                    "model": task.model,
                    "inherit": task.inherit.describe(),
                }));
            }
            Err(error) => entries.push(serde_json::json!({
                "index": index,
                "status": "failed",
                "error": error,
                "agent": task.agent_name,
                "model": task.model,
                "inherit": task.inherit.describe(),
            })),
        }
    }
    if !spawned_ids.is_empty() {
        record_last_spawn_batch(spawned_ids);
    }

    Ok(serde_json::to_string_pretty(&serde_json::json!({
        "spawned": spawned_count,
        "failed": entries.len() - spawned_count,
        "tasks": entries,
        "next": "Continue independent parent-side work; use task_status for snapshots and task_wait only when blocked on results."
    }))
    .expect("serializing task_spawn_batch result cannot fail"))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_retry",
        description: "",

        execute: execute_task_retry,
        groups: &["builtin", "core"],
    }
});

fn execute_task_retry(args: &Value) -> Result<String, String> {
    ensure_top_level_task_orchestration("task_retry")?;
    let task_id = args
        .get("task_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "task_retry requires non-empty 'task_id'".to_string())?;
    let record = TASK_RETRY_REGISTRY
        .lock()
        .unwrap()
        .get_ref(&task_id.to_string())
        .cloned()
        .ok_or_else(|| format!("task_retry: unknown task_id '{task_id}'"))?;
    if record.session_id != crate::ai::driver::runtime_ctx::current_session_id_or_empty() {
        return Err(format!(
            "task_retry: task_id '{task_id}' belongs to another session"
        ));
    }

    let current_owner_pid = current_task_owner_pid()?;
    if record.owner_pid != current_owner_pid {
        return Err(format!(
            "task_retry: task_id '{task_id}' belongs to a different parent process"
        ));
    }
    match record.terminal_status.as_deref() {
        Some(status) if is_retryable_task_status(status) => {}
        Some(status) => {
            return Err(format!(
                "task_retry: task_id '{task_id}' ended with status '{status}' and is not retryable"
            ));
        }
        None => {
            return Err(format!(
                "task_retry: task_id '{task_id}' has no collected terminal failure yet; collect it with task_wait or task_status first"
            ));
        }
    }
    let spawned = spawn_subagent_kernel_task_attempt(&record.prepared, Some(&record.retry_root))?;
    Ok(format!(
        "Task retried: retry_of={}, new_task_id={}, pid={}, agent={}, model={}, inherit={}\nThis is a distinct attempt. Collect and integrate both task ids separately.",
        record.retry_root,
        spawned.task_id,
        spawned.pid,
        record.prepared.agent_name,
        record.prepared.model,
        record.prepared.inherit.describe()
    ))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_wait",
        description: "",

        execute: execute_task_wait,
        groups: &["builtin", "core"],
    }
});

inventory::submit!(ToolHistoryPolicyRegistration {
    name: "task_wait",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Never,
        counts_toward_precision_inline_budget: false,
    },
});

fn ensure_top_level_task_orchestration(tool_name: &str) -> Result<(), String> {
    if crate::ai::driver::runtime_ctx::current_subagent_depth() == 0 {
        return Ok(());
    }
    Err(format!(
        "{tool_name} is only available to top-level agents. This subagent is a leaf task; complete the assigned work directly instead of waiting on, inspecting, or cancelling parent-owned subagent tasks."
    ))
}

fn parse_task_wait_options(args: &Value) -> Result<(u64, WaitPolicy), String> {
    let timeout_secs = args
        .get("timeout_secs")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_TASK_WAIT_TIMEOUT_SECS)
        .clamp(1, MAX_TASK_WAIT_TIMEOUT_SECS);
    let wait_policy = match args.get("wait_policy").and_then(Value::as_str) {
        Some("any") | None => WaitPolicy::Any,
        Some("all") => WaitPolicy::All,
        Some(other) => {
            return Err(format!(
                "Unknown wait_policy: {} (expected 'any' or 'all')",
                other
            ));
        }
    };
    Ok((timeout_secs, wait_policy))
}

pub(crate) fn execute_task_wait(args: &Value) -> Result<String, String> {
    ensure_top_level_task_orchestration("task_wait")?;
    let current_session_id = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
    let requested_task_ids = args["task_ids"]
        .as_array()
        .ok_or("Missing 'task_ids' array parameter")?
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect::<Vec<_>>();

    if requested_task_ids.is_empty() {
        return Err("task_ids array cannot be empty".to_string());
    }

    // 单次 task_wait 调用的等待预算。详见 DEFAULT_TASK_WAIT_TIMEOUT_SECS 注释——
    // 超时只意味着本次没等到，subagent 仍在跑、资源不会被释放。
    let (timeout_secs, wait_policy) = parse_task_wait_options(args)?;

    // wait_policy: "any" | "all"，默认 "any"，避免前台被最慢任务拖住。
    // - all  — 等到所有 pending 任务都完成才返回（适合需要汇总）；
    // - any  — 任一 pending 任务完成即返回，其余仍在跑、可继续 task_wait
    //          （适合 fan-out 后想边收边推进）。
    let current_owner_pid = current_task_owner_pid()?;
    let mut registry = TASK_REGISTRY.lock().unwrap();
    let mut foreign_session_task_ids = Vec::new();
    let mut foreign_owner_task_ids = Vec::new();
    let mut missing_task_ids = Vec::new();
    let mut task_ids_filtered = Vec::new();
    for tid in &requested_task_ids {
        match registry.get_ref(&tid) {
            Some(entry) if task_entry_owned_by(entry, &current_session_id, current_owner_pid) => {
                task_ids_filtered.push(tid.clone());
            }
            Some(entry) if entry.session_id == current_session_id => {
                foreign_owner_task_ids.push(tid.clone());
            }
            Some(_) => foreign_session_task_ids.push(tid.clone()),
            None => missing_task_ids.push(tid.clone()),
        }
    }
    if !foreign_session_task_ids.is_empty() {
        return Err(format!(
            "Refusing to wait on task_id(s) owned by another session: {}",
            foreign_session_task_ids.join(", ")
        ));
    }
    if !foreign_owner_task_ids.is_empty() {
        return Err(format!(
            "Refusing to wait on task_id(s) not owned by current process pid={}: {}",
            current_owner_pid,
            foreign_owner_task_ids.join(", ")
        ));
    }
    // registry miss 既可能是已交付后正常清理，也可能是模型拼错、跨进程旧 id 或
    // 注册表异常。只有 durable evidence ledger 中确有 tombstone 才能判定为已交付。
    let mut already_delivered = Vec::new();
    let mut unknown_task_ids = Vec::new();
    if !missing_task_ids.is_empty() {
        let context = crate::ai::driver::runtime_ctx::try_current()
            .ok_or("task_wait cannot classify missing task_ids outside an active driver turn")?;
        for task_id in missing_task_ids {
            let delivered = crate::ai::history::task_evidence_exists(
                context.app_proto.config.history_file.as_path(),
                &current_session_id,
                &task_id,
            )
            .map_err(|error| {
                format!("failed to inspect durable task evidence for {task_id}: {error}")
            })?;
            if delivered {
                already_delivered.push(task_id);
            } else {
                unknown_task_ids.push(task_id);
            }
        }
    }
    if !unknown_task_ids.is_empty() {
        return Err(format!(
            "Unknown task_id(s): {}. These ids are neither active in the current session/process \
             nor present in its durable delivered-task ledger.",
            unknown_task_ids.join(", ")
        ));
    }
    // 已交付 id 与仍 pending id 混用是预期输入：PARKED / BUDGET-ELAPSED 会要求模型
    // 用同一组 ids 续等。这里只丢弃 ledger 已确认的 delivered ids。
    let task_ids = task_ids_filtered;
    if task_ids.is_empty() {
        return Ok(format!(
            "[task_wait] All {} referenced task(s) already completed and \
             their results were delivered by an earlier task result tool call. No tasks remain to \
             wait on; continue reasoning with the results you already collected.",
            already_delivered.len()
        ));
    }
    // lone-spawn 规范提示：只计算一次（消费 hinted 标志），后续返回点统一附加。
    let lone_spawn_hint = lone_spawn_hint_note(&task_ids);
    let wait_key = task_wait_key(
        &current_session_id,
        current_owner_pid,
        &wait_policy,
        &requested_task_ids,
    );
    let wait_state = load_or_create_task_wait_state(&wait_key, timeout_secs);
    let wait_budget_elapsed = wait_state.expired;

    let mut ready = Vec::new();
    let mut pending = Vec::new();
    // 收集本次调用中已完成（成功 / 失败、channel/futex 已销毁、需要从 registry
    // 删除）的 task_id；suspended 与 budget-elapsed 早返回路径也会用它清理。
    let mut finished: Vec<String> = Vec::new();
    // `write_terminal_subagent_result` 只终止 kernel process，不会停止宿主 Tokio
    // Future。先逐个 abort 已超出总寿命的 worker，再进入 kernel 临界区发布终态。
    for tid in &task_ids {
        let entry = registry.get_ref(tid).expect("validated");
        if entry.started_at.elapsed() > SUBAGENT_WALL_CLOCK_TIMEOUT {
            entry.cancel_stream.store(true, Ordering::Release);
            if let Some(handle) = &entry.abort_handle {
                handle.abort();
            }
        }
    }
    // closure 默认按引用借用 wait_policy / registry / pending / ready / finished，
    // 不加 `move`，保证 closure 返回后外层 `if !pending.is_empty()` 等代码仍可访问。
    let wait_message = with_os_kernel(|os| {
        for tid in &task_ids {
            let entry = registry.get_ref(tid).expect("validated");
            // ⚠️ 这里之前曾按 `entry.started_at.elapsed() >= timeout_secs`
            // 直接把任务标记为 TIMEOUT 并销毁 channel/futex —— 这是 bug：
            // `started_at` 是 spawn 时间，不是本次 task_wait 的开始时间。如果
            // 主 agent 在 spawn 后很久才第一次调 task_wait，所有任务都会
            // **立刻** 被报为 TIMEOUT 且 result_channel 被销毁，subagent
            // 真实结果永久丢失，主 agent 自然会以为 "subagent 卡住"。
            //
            // 现在的做法：只看 channel 上有没有就绪 payload；如果还没有，统一
            // 走 pending 分支。单次 task_wait 预算由 TASK_WAIT_STATES 的真实
            // wall-clock deadline 控制；driver run_loop 到期后唤醒 owner 进程，
            // 下一次 task_wait 才返回 BUDGET ELAPSED。预算耗尽也 **绝不销毁
            // channel/futex**，主 agent 可以继续调 task_wait 续等。
            // wall-clock 总寿命检查：subagent 若超过 SUBAGENT_WALL_CLOCK_TIMEOUT
            // 仍无结果（典型如卡在单个永不返回的工具执行里），主动终止并写入
            // timeout 终态，使紧随其后的 read_task_result 立即读到结果，避免主
            // agent 陷入"超时->续等->再超时"空转。区别于历史上用 started_at 对比
            // 单次 timeout_secs 的 bug：这里用独立的、远大于单次 wait 预算的总
            // 寿命上限，且写入失败结果而非销毁 channel，结果不会丢失。
            if entry.started_at.elapsed() > SUBAGENT_WALL_CLOCK_TIMEOUT {
                write_terminal_subagent_result(
                    os,
                    tid,
                    entry.pid,
                    entry.result_channel_id,
                    entry.completion_futex_addr,
                    "timeout",
                    &format!(
                        "Subagent exceeded wall-clock lifetime of {}s (likely stuck in a non-returning tool execution)",
                        SUBAGENT_WALL_CLOCK_TIMEOUT.as_secs()
                    ),
                );
            }
            if let Some(rendered) = collect_ready_task_result(os, tid, entry)? {
                ready.push(rendered);
                cleanup_collected_task(os, entry, "subagent result collected");
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
                let rendered = collect_missing_task_result(tid, entry)?;
                cleanup_collected_task(os, entry, "subagent terminated without output");
                ready.push(rendered);
                finished.push(tid.clone());
            }
        }

        // `any` 在首次扫描已拿到结果时必须立即返回，不能再被其余 pending 任务挂起。
        if !pending.is_empty()
            && !wait_budget_elapsed
            && !(wait_policy == WaitPolicy::Any && !ready.is_empty())
        {
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
                None,
            )?;
            // 无论 epoll_wait_many 是否 suspended，都先 re-scan 收集在等待期间
            // 变为就绪的结果。如果 suspended 且所有任务都已完成，直接返回结果
            // （而不是 PARKED），避免 wait_policy=all 时模型被迫反复调用 task_wait。
            // 仅当 re-scan 后仍有 pending 且确为 suspended 时才返回 PARKED。
            pending.clear();
            for tid in &pending_ids {
                let entry = registry.get_ref(tid).expect("validated after wait");
                if let Some(rendered) = collect_ready_task_result(os, tid, entry)? {
                    ready.push(rendered);
                    cleanup_collected_task(os, entry, "subagent result collected after wait");
                    finished.push(tid.clone());
                } else if is_task_pending(os, entry.pid)? {
                    pending.push((tid.clone(), entry.pid));
                } else {
                    let rendered = collect_missing_task_result(tid, entry)?;
                    cleanup_collected_task(
                        os,
                        entry,
                        "subagent terminated without output after wait",
                    );
                    ready.push(rendered);
                    finished.push(tid.clone());
                }
            }
            // Re-scan 后仍有 pending 且确为 suspend（协作式让出，非预算耗尽），
            // 返回 PARKED 并附带已收集的部分结果。这里 **绝不能** 用
            // "BUDGET ELAPSED" 之类的终态措辞：suspend 是毫秒级同步返回的（不是
            // 真的等满 timeout_secs），否则模型会把"刚发起等待就超时"误判成
            // "子任务卡住"，从而提前放弃并转手动分析。
            if !pending.is_empty()
                && wait.suspended
                && !(wait_policy == WaitPolicy::Any && !ready.is_empty())
            {
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
        return Ok(append_lone_spawn_hint(message, lone_spawn_hint.as_deref()));
    }
    if wait_policy == WaitPolicy::Any && !ready.is_empty() {
        clear_task_wait_state(&wait_key);
        return Ok(append_lone_spawn_hint(
            ready.join("\n\n---\n\n"),
            lone_spawn_hint.as_deref(),
        ));
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
        clear_task_wait_state(&wait_key);
        return Ok(append_lone_spawn_hint(
            parts.join("\n\n---\n\n"),
            lone_spawn_hint.as_deref(),
        ));
    }

    for tid in &task_ids {
        registry.remove(tid);
    }
    clear_task_wait_state(&wait_key);
    Ok(append_lone_spawn_hint(
        ready.join("\n\n---\n\n"),
        lone_spawn_hint.as_deref(),
    ))
}

fn append_lone_spawn_hint(mut text: String, hint: Option<&str>) -> String {
    if let Some(hint) = hint {
        text.push_str("\n\n");
        text.push_str(hint);
    }
    text
}

/// 向 subagent 的 result channel 写入一条终态结果并终止其 kernel 进程。用于
/// task_cancel（主动取消）与 wall-clock 总寿命超时。结果采用与
/// `publish_background_task_failure` 相同的 status/output/error 格式，使
/// task_wait / task_status 的收集路径能正常读到。本函数只释放 producer 端命名
/// 所有权并 store futex 唤醒等待方；channel/futex 的 destroy 留给收集方（task_wait
/// 的 ready 路径或 task_cancel 自身）完成，避免重复释放。
fn write_terminal_subagent_result(
    os: &mut dyn aios_kernel::kernel::Kernel,
    task_id: &str,
    pid: u64,
    result_channel_id: u64,
    completion_futex_addr: aios_kernel::primitives::FutexAddr,
    status: &str,
    error: &str,
) {
    // 调用方必须先 abort 实际执行 subagent 的 Tokio task；kernel process 状态本身
    // 不会停止宿主进程里的 Future。随后再终止 kernel 进程并发布终态结果。
    let _ = os.kill_process(pid, format!("{}: {}", status, error));
    // 再以 subagent 身份写终态结果并释放 producer 端（result channel 的 producer
    // 所有权校验要求 current == pid）。进程虽已 terminated，但 channel/futex 资源
    // 尚未回收（回收发生在 drop_terminated），故仍可写入。
    let original = os.current_process_id();
    os.set_current_pid(Some(pid));
    let payload = serde_json::json!({
        "status": status,
        "output": "",
        "error": error,
        "progress": current_task_progress(task_id),
    })
    .to_string();
    let _ = os.channel_send(Some(pid), ChannelId(result_channel_id), payload);
    let _ = os.channel_close(Some(pid), ChannelId(result_channel_id));
    let _ = os.channel_release_named(ChannelId(result_channel_id), "task_result.producer");
    let _ = os.futex_store(completion_futex_addr, 1);
    os.set_current_pid(original);
}

/// 调度器每 epoch 调用：扫描 TASK_REGISTRY 中超过 wall-clock 总寿命上限且仍在
/// 运行的 subagent，终止其进程并写入 timeout 终态结果。
///
/// 与 `task_wait` 内的 wall-clock 检查互补：task_wait 只在主 agent 主动调用时触发；
/// 本函数在 driver run_loop 每 epoch 主动扫描，即使主 agent 去做别的事（长期不调
/// task_wait），卡死的 subagent 进程也能被及时终止，避免永久占用调度器资源。
///
/// 资源语义：只 kill 进程 + 写终态结果，**不**销毁 channel/futex、**不**从 registry
/// 移除——这些留给收集方（task_wait 的 ready 路径）完成，避免重复释放。进程被 kill
/// 后 `is_task_pending` 返回 false，后续 epoch 扫描到同一 entry 会跳过，不会重复 kill。
///
/// 锁顺序：分三步以避免与 task_wait（registry -> kernel）形成锁环——
/// 1. 仅锁 TASK_REGISTRY 收集候选，立即释放；
/// 2. 不持任何锁，abort 实际执行 subagent 的 Tokio task；
/// 3. 仅锁 kernel（via with_os_kernel）执行 kill + 写结果。
/// 两步绝不同时持有 registry 与 kernel（GLOBAL_OS 与 App.os 是同一把锁，参见
/// os_tools.rs 的重入死锁警告）。
pub(crate) fn reap_timed_out_subagents() {
    // Step 1：仅持 registry 锁，收集超时候选（pid / channel / futex），立即释放。
    let candidates = {
        let registry = TASK_REGISTRY.lock().unwrap();
        registry
            .iter()
            .filter(|(_, e)| e.started_at.elapsed() > SUBAGENT_WALL_CLOCK_TIMEOUT)
            .map(|(task_id, e)| {
                (
                    task_id.clone(),
                    e.pid,
                    e.result_channel_id,
                    e.completion_futex_addr,
                    e.abort_handle.clone(),
                    e.cancel_stream.clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    if candidates.is_empty() {
        return;
    }
    // Step 2：不持任何锁，先停止真实 Tokio Future，避免它与 timeout 终态并发写结果。
    for (_, _, _, _, abort_handle, cancel_stream) in &candidates {
        cancel_stream.store(true, Ordering::Release);
        if let Some(handle) = abort_handle {
            handle.abort();
        }
    }
    // Step 3：仅持 kernel 锁，逐个检查是否仍在运行，是则 kill + 写 timeout 终态。
    let _ = with_os_kernel(|os| {
        for (task_id, pid, result_channel_id, completion_futex_addr, _, _) in candidates {
            if !is_task_pending(os, pid)? {
                // 进程已结束（正常完成 / 失败 / 被他人 kill），跳过；其结果与资源
                // 清理交给收集方处理。
                continue;
            }
            write_terminal_subagent_result(
                os,
                &task_id,
                pid,
                result_channel_id,
                completion_futex_addr,
                "timeout",
                &format!(
                    "Subagent exceeded wall-clock lifetime of {}s (reaped by scheduler; likely stuck in a non-returning tool execution)",
                    SUBAGENT_WALL_CLOCK_TIMEOUT.as_secs()
                ),
            );
        }
        Ok(())
    });
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_status",
        description: "",

        execute: execute_task_status,
        groups: &["builtin", "core"],
    }
});

inventory::submit!(ToolHistoryPolicyRegistration {
    name: "task_status",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Never,
        counts_toward_precision_inline_budget: false,
    },
});

fn execute_task_integrate(args: &Value) -> Result<String, String> {
    ensure_top_level_task_orchestration("task_integrate")?;
    let tasks = args
        .get("tasks")
        .and_then(Value::as_array)
        .ok_or("Missing 'tasks' array parameter")?;
    if tasks.is_empty() {
        return Err("tasks array cannot be empty".to_string());
    }
    let context = crate::ai::driver::runtime_ctx::try_current()
        .ok_or("task_integrate requires an active driver turn")?;
    let history_file = context.app_proto.config.history_file.as_path();
    let session_id = context.app_proto.session_id.as_str();
    let mut integrated = Vec::new();
    for task in tasks {
        let task_id = task
            .get("task_id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or("Each task integration requires a non-empty task_id")?;
        let disposition = task
            .get("disposition")
            .and_then(Value::as_str)
            .ok_or("Each task integration requires disposition")?;
        if !matches!(disposition, "accepted" | "rejected" | "superseded") {
            return Err(format!("Invalid disposition for {task_id}: {disposition}"));
        }
        let summary = task
            .get("summary")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or("Each task integration requires a non-empty summary")?;
        if summary.chars().count() > 6_000 {
            return Err(format!(
                "Integration summary for {task_id} exceeds 6000 characters"
            ));
        }
        let found = crate::ai::history::integrate_task_evidence(
            history_file,
            session_id,
            task_id,
            disposition,
            summary,
        )
        .map_err(|error| format!("failed to integrate {task_id}: {error}"))?;
        if !found {
            return Err(format!(
                "Unknown task_id in durable task evidence ledger: {task_id}"
            ));
        }
        integrated.push(task_id.to_string());
    }
    Ok(format!(
        "Integrated {} task result(s): {}",
        integrated.len(),
        integrated.join(", ")
    ))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_integrate",
        description: "",

        execute: execute_task_integrate,
        groups: &["builtin", "core"],
    }
});

inventory::submit!(ToolHistoryPolicyRegistration {
    name: "task_integrate",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Never,
        counts_toward_precision_inline_budget: false,
    },
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_cancel",
        description: "",

        execute: execute_task_cancel,
        groups: &["builtin", "core"],
    }
});

inventory::submit!(ToolHistoryPolicyRegistration {
    name: "task_cancel",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Never,
        counts_toward_precision_inline_budget: false,
    },
});

/// 销毁一个 session 时终止并移除其全部异步 subagent。
///
/// 这条路径不保留可收集结果：父 session 已被用户显式删除，继续保留 registry、IPC
/// 或后台 Future 只会让已删除的派生历史被重新创建。
pub(crate) fn discard_tasks_for_session(session_id: &str) {
    let candidates = {
        let registry = TASK_REGISTRY.lock().unwrap();
        registry
            .iter()
            .filter(|(_, entry)| entry.session_id == session_id)
            .map(|(task_id, entry)| {
                (
                    task_id.clone(),
                    entry.pid,
                    entry.result_channel_id,
                    entry.completion_futex_addr,
                    entry.abort_handle.clone(),
                    entry.cancel_stream.clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    for (_, _, _, _, abort_handle, cancel_stream) in &candidates {
        cancel_stream.store(true, Ordering::Release);
        if let Some(handle) = abort_handle {
            handle.abort();
        }
    }

    if !candidates.is_empty() {
        let _ = with_os_kernel(|os| {
            for (_, pid, result_channel_id, completion_futex_addr, _, _) in &candidates {
                let _ = os.cleanup_process_resources(*pid);
                let _ = os.kill_process(*pid, "parent session deleted".to_string());
                let _ = os.drop_terminated(*pid);
                let channel_id = ChannelId(*result_channel_id);
                let _ = os.channel_close(None, channel_id);
                let _ = os.channel_release_named(channel_id, "task_result.consumer");
                let _ = os.channel_release_named(channel_id, "task_result.producer");
                let _ = os.channel_destroy(None, channel_id);
                let _ = os.futex_destroy(*completion_futex_addr);
            }
            Ok::<(), String>(())
        });
    }

    let task_ids = candidates
        .into_iter()
        .map(|(task_id, _, _, _, _, _)| task_id)
        .collect::<Vec<_>>();
    {
        let mut registry = TASK_REGISTRY.lock().unwrap();
        for task_id in &task_ids {
            registry.remove(task_id);
        }
    }
    {
        let mut progress_registry = TASK_PROGRESS_REGISTRY.lock().unwrap();
        for task_id in &task_ids {
            progress_registry.remove(task_id);
        }
    }
    {
        let mut retry_registry = TASK_RETRY_REGISTRY.lock().unwrap();
        let retry_task_ids = retry_registry
            .iter()
            .filter(|(_, spec)| spec.session_id == session_id)
            .map(|(task_id, _)| task_id.clone())
            .collect::<Vec<_>>();
        for task_id in retry_task_ids {
            retry_registry.remove(&task_id);
        }
    }
    TASK_WAIT_STATES
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .retain(|key, _| key.session_id != session_id);
}

pub(crate) fn execute_task_cancel(args: &Value) -> Result<String, String> {
    ensure_top_level_task_orchestration("task_cancel")?;
    let task_ids = args["task_ids"]
        .as_array()
        .ok_or("Missing 'task_ids' array parameter")?
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect::<Vec<_>>();
    if task_ids.is_empty() {
        return Err("task_ids array cannot be empty".to_string());
    }
    let reason = args
        .get("reason")
        .and_then(Value::as_str)
        .map(String::from)
        .unwrap_or_else(|| "cancelled by parent agent".to_string());
    let current_session_id = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
    let current_owner_pid = current_task_owner_pid()?;

    let mut cancelled: Vec<String> = Vec::new();
    let mut already_finished: Vec<String> = Vec::new();
    let mut not_found: Vec<String> = Vec::new();
    // 先只持 registry 锁复制取消所需信息，随后立即释放；Tokio task 的 abort 和
    // kernel 终态写入都不能与 registry 锁重叠，避免与收集路径形成锁环。
    let candidates = {
        let registry = TASK_REGISTRY.lock().unwrap();
        task_ids
            .iter()
            .filter_map(|tid| match registry.get_ref(tid) {
                Some(entry)
                    if task_entry_owned_by(entry, &current_session_id, current_owner_pid) =>
                {
                    Some((
                        tid.clone(),
                        entry.pid,
                        entry.result_channel_id,
                        entry.completion_futex_addr,
                        entry.abort_handle.clone(),
                        entry.cancel_stream.clone(),
                    ))
                }
                _ => {
                    not_found.push(tid.clone());
                    None
                }
            })
            .collect::<Vec<_>>()
    };

    // 必须先停止实际 Tokio Future，再进入 kernel 写 cancelled 终态；否则逻辑进程
    // 已终止后，网络请求或工具调用仍可能在后台继续运行并与终态写入竞争。
    for (_, _, _, _, abort_handle, cancel_stream) in &candidates {
        cancel_stream.store(true, Ordering::Release);
        if let Some(handle) = abort_handle {
            handle.abort();
        }
    }

    for (tid, pid, result_channel_id, completion_futex_addr, _, _) in candidates {
        // 仅对仍在运行的 subagent 执行取消。已结束（正常完成 / 失败 / 进程终止）的
        // 任务不再 kill、也不再写终态结果——否则会向 channel 追加一条 "cancelled"
        // 消息并销毁 channel，遮蔽/丢弃 subagent 的真实结果，且让后续 task_wait 拿
        // 到错误的 cancelled 状态。已结束任务的 channel/futex/registry 清理留给收集
        // 方（task_wait 的 ready / 失败路径，或 task_status 后的 task_wait）。
        let was_pending = with_os_kernel(|os| {
            if !is_task_pending(os, pid)? {
                return Ok(false);
            }
            write_terminal_subagent_result(
                os,
                &tid,
                pid,
                result_channel_id,
                completion_futex_addr,
                "cancelled",
                &reason,
            );
            Ok(true)
        })?;
        if was_pending {
            cancelled.push(tid);
        } else {
            already_finished.push(tid);
        }
    }

    let mut msg = String::new();
    if !cancelled.is_empty() {
        msg.push_str(&format!(
            "[task_cancel] Cancelled {} task(s): {}. The subagent processes were terminated and \
             their result slots were filled with a 'cancelled' terminal result. Required next step: \
             collect these terminal results with task_wait or task_status so the runtime can clean up \
             their registry entries and IPC resources.",
            cancelled.len(),
            cancelled.join(", ")
        ));
    }
    if !already_finished.is_empty() {
        if !msg.is_empty() {
            msg.push('\n');
        }
        msg.push_str(&format!(
            "[task_cancel] {} task_id(s) were already finished (completed/failed/cancelled) and \
             were left untouched - use task_wait/task_status to collect their real terminal results: {}",
            already_finished.len(),
            already_finished.join(", ")
        ));
    }
    if !not_found.is_empty() {
        if !msg.is_empty() {
            msg.push('\n');
        }
        msg.push_str(&format!(
            "[task_cancel] {} task_id(s) not found or not owned by this process/session (already \
             collected or never spawned): {}",
            not_found.len(),
            not_found.join(", ")
        ));
    }
    Ok(msg)
}

pub(crate) fn execute_task_status(_args: &Value) -> Result<String, String> {
    ensure_top_level_task_orchestration("task_status")?;
    let current_session_id = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
    let current_owner_pid = current_task_owner_pid()?;
    let tracked = {
        let registry = TASK_REGISTRY.lock().unwrap();
        registry
            .iter()
            .filter(|(_, entry)| task_entry_owned_by(entry, &current_session_id, current_owner_pid))
            .map(|(tid, entry)| {
                (
                    tid.clone(),
                    entry.owner_pid,
                    entry.pid,
                    entry.result_channel_id,
                    entry.completion_futex_addr,
                    entry.description.clone(),
                    entry.agent_name.clone(),
                    entry.model.clone(),
                    entry.started_at,
                )
            })
            .collect::<Vec<_>>()
    };
    if tracked.is_empty() {
        return Ok("No async tasks currently tracked.".to_string());
    }

    let mut lines = vec![
        "TaskID              PID      Agent          Model          State       Description"
            .to_string(),
    ];
    // 对已经把结果写回 channel 的子任务，直接 **消费并清理** 正文，附在表格后面。
    // 否则模型即使看到 state=completed，也只能回头再调 task_wait 才能拿到输出；
    // 更糟的是如果它把"seen completed in task_status"视为已处理，就会绕过收口守卫，
    // 留下 registry 条目和 channel/futex 资源。这里既然已经把结果返回给模型，
    // 就应视为已收集完成。
    let mut completed_outputs: Vec<String> = Vec::new();
    let mut finished_ids: Vec<String> = Vec::new();
    with_os_kernel(|os| {
        for (
            tid,
            owner_pid,
            pid,
            result_channel_id,
            completion_futex_addr,
            description,
            agent_name,
            model,
            started_at,
        ) in &tracked
        {
            let state_str = task_state_string(os, *result_channel_id, *pid)?;
            let short_id = if tid.len() > 19 { &tid[..19] } else { tid };
            lines.push(format!(
                "{:<19} {:<8} {:<14} {:<14} {:<11} {}",
                short_id, pid, agent_name, model, state_str, description
            ));
            let entry = AsyncTaskEntry {
                session_id: current_session_id.clone(),
                result_observed: false,
                owner_pid: *owner_pid,
                pid: *pid,
                result_channel_id: *result_channel_id,
                completion_futex_addr: *completion_futex_addr,
                description: description.clone(),
                agent_name: agent_name.clone(),
                model: model.clone(),
                is_model_auto_selected: false,
                auto_model_fallback: None,
                selection_explanation: String::new(),
                inherit: InheritOptions::default(),
                started_at: *started_at,
                abort_handle: None,
                cancel_stream: Arc::new(AtomicBool::new(false)),
            };
            if let Some(rendered) = collect_ready_task_result(os, tid, &entry)? {
                completed_outputs.push(rendered);
                cleanup_collected_task(os, &entry, "subagent result collected by task_status");
                finished_ids.push(tid.clone());
            } else if !is_task_pending(os, *pid)? {
                // 与 task_wait 保持一致：进程已终止但没有写回结果时，也必须把任务
                // 收口并释放双方的 channel ownership，避免仅轮询 task_status 时泄漏。
                let rendered = collect_missing_task_result(tid, &entry)?;
                completed_outputs.push(rendered);
                cleanup_collected_task(
                    os,
                    &entry,
                    "subagent terminated without output before task_status collection",
                );
                finished_ids.push(tid.clone());
            }
        }
        Ok(())
    })?;
    if !finished_ids.is_empty() {
        let mut registry = TASK_REGISTRY.lock().unwrap();
        for task_id in &finished_ids {
            registry.remove(task_id);
        }
    }

    if !completed_outputs.is_empty() {
        lines.push(String::new());
        lines.push(
            "Completed task results below (already collected — no need to wait for these):"
                .to_string(),
        );
        lines.push(completed_outputs.join("\n\n---\n\n"));
    }

    Ok(lines.join("\n"))
}

fn collect_outstanding_task_snapshots(
    session_id: &str,
    owner_pid: u64,
) -> Result<Vec<OutstandingTaskSnapshot>, String> {
    let registry = TASK_REGISTRY.lock().unwrap();
    let tracked = registry
        .iter()
        .filter(|(_, entry)| {
            task_entry_owned_by(entry, session_id, owner_pid) && !entry.result_observed
        })
        .map(|(tid, entry)| {
            (
                tid.clone(),
                entry.result_channel_id,
                entry.pid,
                entry.agent_name.clone(),
                entry.model.clone(),
                entry.description.clone(),
            )
        })
        .collect::<Vec<_>>();
    drop(registry);

    if tracked.is_empty() {
        return Ok(Vec::new());
    }

    with_os_kernel(|os| {
        let mut snapshots = Vec::with_capacity(tracked.len());
        for (task_id, result_channel_id, pid, agent_name, model, description) in &tracked {
            snapshots.push(OutstandingTaskSnapshot {
                task_id: task_id.clone(),
                status: task_state_string(os, *result_channel_id, *pid)?,
                agent_name: agent_name.clone(),
                model: model.clone(),
                description: description.clone(),
            });
        }
        Ok(snapshots)
    })
}

fn render_outstanding_task_anchor(snapshots: &[OutstandingTaskSnapshot]) -> String {
    let mut lines = vec![
        OUTSTANDING_SUBAGENT_TASKS_NOTE_PREFIX.to_string(),
        format!(
            "You still have {} spawned subagent task(s) tracked in this session. Do not silently forget them or finish the user-facing answer before handling them.",
            snapshots.len()
        ),
        format!(
            "Outstanding task_ids: [{}]",
            snapshots
                .iter()
                .map(|snapshot| snapshot.task_id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    ];
    for snapshot in snapshots {
        lines.push(format!(
            "- task_id={} status={} agent={} model={} desc={}",
            snapshot.task_id,
            snapshot.status,
            snapshot.agent_name,
            snapshot.model,
            snapshot.description
        ));
    }
    lines.push(
        "Required next step: use `task_wait` with the same task_ids to collect results, or `task_status` for a non-blocking snapshot. Before ending the turn, ensure every listed task_id has been handled — including failures."
            .to_string(),
    );
    lines.join("\n")
}

pub(crate) fn build_outstanding_task_anchor(session_id: &str) -> Result<Option<String>, String> {
    let Some(owner_pid) = current_task_owner_pid_opt() else {
        return Ok(None);
    };
    let snapshots = collect_outstanding_task_snapshots(session_id, owner_pid)?;
    if snapshots.is_empty() {
        return Ok(None);
    }
    Ok(Some(render_outstanding_task_anchor(&snapshots)))
}

/// 已到达迭代硬上限、不再打回模型时，把仍未回收的子任务状态拼进最终回答，
/// 避免未回收结果被静默抛弃。与 `build_outstanding_task_anchor` 共用 snapshot
/// 收集，但文案面向最终输出：此时不会再给模型继续的机会，因此不再要求模型
/// “下一步调用 task_wait”，而是告知用户哪些子任务结果未被回收、需要重跑收集。
pub(crate) fn build_abandoned_tasks_notice(
    session_id: &str,
    iteration_limit: usize,
) -> Result<Option<String>, String> {
    let Some(owner_pid) = current_task_owner_pid_opt() else {
        return Ok(None);
    };
    let snapshots = collect_outstanding_task_snapshots(session_id, owner_pid)?;
    if snapshots.is_empty() {
        return Ok(None);
    }
    let mut lines = vec![format!(
        "The following {} spawned subagent task(s) were still outstanding when the tool iteration limit ({}) was reached; their results were NOT collected and are not reflected in this answer:",
        snapshots.len(),
        iteration_limit
    )];
    for snapshot in &snapshots {
        lines.push(format!(
            "- task_id={} status={} agent={} model={} desc={}",
            snapshot.task_id,
            snapshot.status,
            snapshot.agent_name,
            snapshot.model,
            snapshot.description
        ));
    }
    lines.push(
        "Required follow-up: re-run this turn and collect these results with `task_wait` / `task_status` for the listed task_ids."
            .to_string(),
    );
    Ok(Some(lines.join("\n")))
}

pub(crate) fn outstanding_task_anchor_prefix() -> &'static str {
    OUTSTANDING_SUBAGENT_TASKS_NOTE_PREFIX
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StoredTaskResult {
    pub(crate) status: String,
    pub(crate) output: String,
    pub(crate) error: Option<String>,
    #[serde(default)]
    pub(crate) progress: Option<String>,
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
        parts.push(format!("Failure reason: {}", error));
    }
    if let Some(progress) = result.progress
        && !progress.trim().is_empty()
    {
        parts.push(format!("Last known progress: {}", progress));
    }
    if !result.output.trim().is_empty() {
        if result.status == "completed" {
            parts.push(result.output.trim().to_string());
        } else {
            parts.push(format!("Partial output:\n{}", result.output.trim()));
        }
    } else {
        parts.push("(subagent did not produce any final assistant text)".to_string());
    }
    parts.push(SUBAGENT_PARENT_SUMMARY_REMINDER.to_string());
    parts.join("\n")
}

fn format_task_result_with_id(
    task_id: &str,
    entry: &AsyncTaskEntry,
    result: StoredTaskResult,
) -> String {
    let retryable = is_retryable_task_status(&result.status);
    let mut rendered = format!("[task_id={task_id}]\n{}", format_task_result(entry, result));
    if retryable {
        rendered.push_str(&format!(
            "\nRetry available: call `task_retry` with `task_id=\"{task_id}\"` to rerun the same subagent configuration as a new linked attempt."
        ));
    }
    rendered
}

fn persist_rendered_task_evidence(
    task_id: &str,
    entry: &AsyncTaskEntry,
    status: &str,
    rendered: &str,
) -> Result<(), String> {
    let Some(context) = crate::ai::driver::runtime_ctx::try_current() else {
        return Ok(());
    };
    crate::ai::history::record_delivered_task_evidence(
        context.app_proto.config.history_file.as_path(),
        &entry.session_id,
        crate::ai::history::DeliveredTaskEvidence {
            task_id,
            description: &entry.description,
            agent_name: &entry.agent_name,
            model: &entry.model,
            status,
            payload: rendered,
        },
    )
    .map_err(|error| format!("failed to persist task evidence for {task_id}: {error}"))
}

fn collect_missing_task_result(task_id: &str, entry: &AsyncTaskEntry) -> Result<String, String> {
    let result = StoredTaskResult {
        status: "failed".to_string(),
        output: String::new(),
        error: Some(format!(
            "Subagent process pid={} terminated without publishing any output.",
            entry.pid
        )),
        progress: current_task_progress(task_id),
    };
    let rendered = format_task_result_with_id(task_id, entry, result.clone());
    persist_rendered_task_evidence(task_id, entry, &result.status, &rendered)?;
    mark_task_retry_status(task_id, &result.status);
    TASK_PROGRESS_REGISTRY
        .lock()
        .unwrap()
        .take(&task_id.to_string());
    Ok(rendered)
}

fn collect_ready_task_result(
    os: &mut dyn Kernel,
    task_id: &str,
    entry: &AsyncTaskEntry,
) -> Result<Option<String>, String> {
    let Some(result) = read_task_result(os, entry.result_channel_id, false)? else {
        return Ok(None);
    };
    let rendered = format_task_result_with_id(task_id, entry, result.clone());
    persist_rendered_task_evidence(task_id, entry, &result.status, &rendered)?;
    let consumed = read_task_result(os, entry.result_channel_id, true)?;
    if consumed.is_none() {
        return Err(format!(
            "task result for {task_id} disappeared after durable persistence"
        ));
    }
    mark_task_retry_status(task_id, &result.status);
    TASK_PROGRESS_REGISTRY
        .lock()
        .unwrap()
        .take(&task_id.to_string());
    Ok(Some(rendered))
}

/// 收集终态结果后统一释放 IPC，并把对应 kernel 进程终止、回收。
///
/// result payload 可在 driver 完成最终进程状态更新前唤醒父 agent；因此收集方不能只
/// 尝试 `drop_terminated`。无论此时进程仍是 Ready/Running 还是已经 Terminated，
/// terminal collection 都负责把“一次性 subagent task”收口，避免进程表持续增长。
fn cleanup_collected_task(os: &mut dyn Kernel, entry: &AsyncTaskEntry, reason: &str) {
    let channel_id = ChannelId(entry.result_channel_id);
    let _ = os.channel_close(None, channel_id);
    let _ = os.channel_release_named(channel_id, "task_result.consumer");
    let _ = os.channel_release_named(channel_id, "task_result.producer");
    let _ = os.channel_destroy(None, channel_id);
    let _ = os.futex_destroy(entry.completion_futex_addr);

    if os.get_process(entry.pid).is_none() {
        return;
    }

    // 防御测试/损坏注册表把前台 owner 自身登记成 task pid；收集 subagent 结果绝不能
    // 终止仍在运行的父进程。若它已经终止，下面仍会正常清理并 drop。
    if entry.pid == entry.owner_pid
        && !matches!(
            os.get_process(entry.pid).map(|process| &process.state),
            Some(ProcessState::Terminated)
        )
    {
        return;
    }

    // kill_process 依赖 current pid 做父子权限校验。正常路径保留 owner 作为 current；
    // session owner 已消失时退回到子进程自杀，确保删除 session 也不会遗留孤儿。
    let collector_pid = if os.get_process(entry.owner_pid).is_some() {
        entry.owner_pid
    } else {
        entry.pid
    };
    os.set_current_pid(Some(collector_pid));
    let _ = os.cleanup_process_resources(entry.pid);
    if !matches!(
        os.get_process(entry.pid).map(|process| &process.state),
        Some(ProcessState::Terminated)
    ) {
        let _ = os.kill_process(entry.pid, reason.to_string());
    }
    let _ = os.drop_terminated(entry.pid);

    if entry.owner_pid != entry.pid && os.get_process(entry.owner_pid).is_some() {
        os.set_current_pid(Some(entry.owner_pid));
    } else {
        os.set_current_pid(None);
    }
}

fn subagent_document_text(agent: &AgentManifest) -> String {
    let mut parts = vec![agent.name.clone(), agent.description.clone()];
    if !agent.prompt.trim().is_empty() {
        parts.push(agent.prompt.chars().take(1500).collect());
    }
    parts.join("\n")
}

/// 从文本提取 2-4 元字符 n-gram 集合（小写、空白折叠归一化）。
/// 仅用于集合相似度计算，不含词频 / 逆文档频率等权重。
fn char_ngram_set_from_text(input: &str) -> FxHashSet<String> {
    let mut normalized = String::with_capacity(input.len());
    let mut prev_space = false;
    for ch in input.to_lowercase().chars() {
        if ch.is_whitespace() {
            if !prev_space {
                normalized.push(' ');
            }
            prev_space = true;
        } else {
            normalized.push(ch);
            prev_space = false;
        }
    }
    let normalized = normalized.trim();
    if normalized.is_empty() {
        return FxHashSet::default();
    }
    let chars: Vec<char> = format!("^{normalized}$").chars().collect();
    let mut set = FxHashSet::default();
    for n in 2..=4 {
        if chars.len() < n {
            continue;
        }
        for window in chars.windows(n) {
            let token: String = window.iter().collect();
            if token.trim().is_empty() {
                continue;
            }
            set.insert(token);
        }
    }
    set
}

/// 子代理自动选择：基于归一化文本的字符 n-gram 集合重叠度（Jaccard）打分。
fn auto_subagent_score(
    agent: &AgentManifest,
    task_text: &str,
) -> f64 {
    let query = char_ngram_set_from_text(task_text);
    let doc = char_ngram_set_from_text(&subagent_document_text(agent));
    if query.is_empty() || doc.is_empty() {
        return 0.0;
    }
    let intersection = query.intersection(&doc).count();
    let union = query.len() + doc.len() - intersection;
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

#[derive(Debug)]
struct SelectedSubagent<'a> {
    agent: &'a AgentManifest,
    auto_selected: bool,
    score: i32,
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

    subagents
        .into_iter()
        .max_by(|a, b| {
            auto_subagent_score(a, &task_text)
                .total_cmp(&auto_subagent_score(b, &task_text))
                .then_with(|| b.name.cmp(&a.name))
        })
        .map(|agent| {
            let score = auto_subagent_score(agent, &task_text);
            SelectedSubagent {
                agent,
                auto_selected: true,
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

fn build_selection_explanation(
    selected: &SelectedSubagent<'_>,
    selected_model: &str,
    model_override: Option<&str>,
    inherited_parent_model: bool,
) -> String {
    let agent_reason = if selected.auto_selected {
        format!(
            "agent_reason=auto-selected as the best available subagent (score={})",
            selected.score
        )
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
            "model_reason=auto-selected for agent_tier={} using {} platform via {} adapter and {} quality_tier",
            format_agent_model_tier(selected.agent),
            models::model_platform_label(selected_model),
            crate::ai::model_names::adapter_slug(models::model_adapter(selected_model)),
            format_quality_tier(models::model_quality_tier(selected_model))
        )
    };

    format!("{agent_reason}\n{model_reason}")
}

#[cfg(test)]
mod tests;
