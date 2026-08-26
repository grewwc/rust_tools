use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use crate::ai::tools::os_tools::GLOBAL_OS;
use crate::ai::tools::storage::file_store::current_session_assets_dir;
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

mod agent_team;

const MAX_TASK_REGISTRY_SIZE: usize = 100;
const DEFAULT_TASK_PRIORITY: u8 = 20;
const DEFAULT_TASK_QUOTA_TURNS: usize = 10;
/// Maximum subagent nesting depth. depth=1 means a subagent spawned directly by a top-level agent.
/// Subagents may not spawn their own sub-subagents, preventing recursive fan-out and results that
/// nobody collects.
pub(crate) const MAX_SUBAGENT_SPAWN_DEPTH: usize = 1;
/// A subagent is a leaf evidence-gathering/execution unit of its parent agent and must not inherit
/// the main agent's full long-running loop budget. The main agent keeps its own max_steps; only
/// the `task` / `task_spawn` launch path is clamped here.
pub(crate) const SUBAGENT_MAX_ITERATIONS: usize = 32;
/// Agents that explicitly declare `max_steps` (such as the deep audit `/audit`) may exceed the
/// default 32 rounds, but no subagent may go beyond this absolute hard cap, preventing a runaway
/// subagent from iterating forever.
pub(crate) const SUBAGENT_MAX_ITERATIONS_HARD_CAP: usize = 256;
/// Hard cap on the number of tasks in a single batch delegation, matching the default maximum
/// batch size of background scheduling. Both the schema and the execution entry point enforce it,
/// so the parent's per-call tool quota cannot be bypassed to cause unbounded fan-out.
const MAX_SUBAGENT_SPAWN_BATCH_SIZE: usize = 8;
const TASK_GOAL_PREFIX: &str = "AIOS_SUBAGENT_TASK:";
/// A subagent's result is only evidence input for the main agent, not a final direct answer to the
/// user. After receiving the payload, the main agent must still synthesize conclusions, risks, and
/// next steps on its own before responding to the user.
pub(crate) const SUBAGENT_PARENT_SUMMARY_REMINDER: &str = "Parent-agent follow-up: summarize the confirmed subagent conclusions in your own response to the user. Do not rely on the raw subagent transcript or terminal fold as the final user-facing answer.";
/// Default wait budget for a single `task_wait` call (seconds). This is only the **maximum block
/// time for this one call**, not the subagent's total lifetime: a timeout merely means "this call
/// did not get the result yet". The main agent can keep calling `task_wait` to wait again; the
/// subagent keeps running in the background and the channel/futex are not destroyed.
///
/// Foreground waits only provide a short collection window; long-running subtasks should overlap
/// with the parent agent's own work instead of suspending the parent for minutes right after
/// task_spawn.
const DEFAULT_TASK_WAIT_TIMEOUT_SECS: u64 = 30;
/// Hard upper bound for `task_wait.timeout_secs`, so the model cannot set an astronomically large
/// timeout that would block the driver indefinitely. After at most 60 seconds, control must return
/// to the parent agent, which re-evaluates whether to continue local work, check status
/// non-blockingly, or genuinely wait again.
const MAX_TASK_WAIT_TIMEOUT_SECS: u64 = 60;

/// Wall-clock lifetime cap for a subagent. Unlike a single task_wait's `timeout_secs` (default 30s,
/// max 60s), this is a process-level hard limit: when a subagent outlives it (typically by getting
/// stuck in a single tool execution that never returns, where a single turn has no wall-clock
/// timeout), the task_wait entry point proactively terminates it and writes a timeout terminal
/// result, so the main agent does not spin in a "timeout -> wait again -> timeout" loop or leave
/// background processes holding resources forever. One hour far exceeds normal completion times
/// and only acts as a safety net for genuinely stuck subagents.
const SUBAGENT_WALL_CLOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60 * 60);
const SUBAGENT_PROGRESS_NOTIFY_INTERVAL: Duration = Duration::from_secs(15);
const SUBAGENT_PROGRESS_PERSIST_INTERVAL: Duration = Duration::from_secs(10);

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
        // By default, inherit only the execution essentials (cwd/skills), not the full conversation
        // history or memory. Narrow tasks get their necessary context passed explicitly by the
        // parent agent in the prompt, avoiding token bloat, attention drift, and a subagent
        // polluting the main memory file. Callers can still explicitly pass `inherit: "all"` or
        // `inherit: "history,cwd,skills"` to restore the old behavior.
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

/// Registry entry maintained by the agent layer for each asynchronous subtask, used by the
/// `task_spawn` / `task_wait` flows.
///
/// **Relationship to the AIOS Kernel `Process`**: some fields of this struct (`pid`, `agent_name`,
/// `description`, `started_at`) already have equivalents in the kernel `Process` (`pid` / `name` /
/// `goal` / `created_at_tick`), so there is **conceptual overlap**. Reasons the overlap is kept:
///
/// 1. Agent-specific fields (`result_channel_id`, `completion_futex_addr`, `inherit`,
///    `selection_explanation`, `model`) have no place in the kernel process table;
/// 2. The agent layer needs to query under the stable string key task_id, while the kernel uses
///    numeric pids;
/// 3. The kernel's `created_at_tick` is a logical tick and cannot be converted back to wall-clock
///    time for the LRU decision in `prune_completed_tasks`.
///
/// **Invariant**: the `pid` in this registry must always correspond to the same process in the
/// kernel process table; results must first be persisted to the evidence ledger by `task_wait` /
/// `task_status` before the registry entry and IPC resources are removed. When at capacity, new
/// tasks are rejected — uncollected results are never evicted.
#[derive(Clone)]
pub(crate) struct AsyncTaskEntry {
    pub(crate) session_id: String,
    pub(crate) result_observed: bool,
    /// pid of the parent process that directly owns this task. task_wait/status/cancel only allow
    /// the owner process to observe its own spawned subtasks, preventing parent/sibling tasks
    /// within the same session from interfering with each other.
    pub(crate) owner_pid: u64,
    /// Matches the kernel `Process.pid`; the agent side additionally stores it so the pid can be
    /// looked up from a task_id.
    pub(crate) pid: u64,
    pub(crate) result_channel_id: u64,
    pub(crate) completion_futex_addr: FutexAddr,
    /// Descriptive text; unlike kernel `Process.goal`, which carries the TASK_GOAL_PREFIX prefix
    /// and the full prompt.
    pub(crate) description: String,
    /// Logical name of the subagent (used to look up the registered AgentManifest, e.g. `"build"`);
    /// shares its source with the kernel `Process.name`, but the kernel-side name is display-only.
    /// Note the distinction: `plan` is a tool name, not an agent name (no `plan` subagent is
    /// registered in this repo), so `agent_name` must not be set to `"plan"` — that would point the
    /// dispatch at a manifest that does not exist.
    pub(crate) agent_name: String,
    pub(crate) model: String,
    pub(crate) is_model_auto_selected: bool,
    pub(crate) auto_model_fallback: Option<models::AutoModelFallbackSpec>,
    pub(crate) selection_explanation: String,
    pub(crate) inherit: InheritOptions,
    /// Cancellation handle of the real Tokio subtask. When the kernel process is terminated, the
    /// handle must be aborted too, otherwise the network request or tool Future keeps running in
    /// the background.
    pub(crate) abort_handle: Option<tokio::task::AbortHandle>,
    /// Cancellation flag shared with the subagent App. A synchronously executing command cannot be
    /// interrupted immediately by a Tokio abort, so timeout/cancel must set this flag first, letting
    /// the command runner kill the actual OS process group.
    pub(crate) cancel_stream: Arc<AtomicBool>,
    /// Wall-clock start time, used by the `prune_completed_tasks` LRU; it cannot be replaced by the
    /// kernel `created_at_tick`.
    pub(crate) started_at: Instant,
    pub(crate) last_progress_notification_at: Option<Instant>,
    pub(crate) last_progress_persisted_at: Option<Instant>,
}

/// Registry of asynchronous subtasks, keyed by task_id (a UUID string); values are
/// [`AsyncTaskEntry`].
///
/// This is **parallel storage** to the AIOS kernel process table: the two are linked through the
/// `pid` field but each keeps its own independent set of fields (see the `AsyncTaskEntry`
/// comments). Accessors should read/write this registry through helpers such as `with_task_entry`
/// / `take_task_entry` rather than holding a lock guard directly.
static TASK_REGISTRY: LazyLock<Mutex<SkipMap<String, AsyncTaskEntry>>> =
    LazyLock::new(|| Mutex::new(SkipMap::default()));
static TASK_PROGRESS_REGISTRY: LazyLock<
    Mutex<SkipMap<String, crate::ai::driver::runtime_ctx::SubagentPhaseSlot>>,
> = LazyLock::new(|| Mutex::new(SkipMap::default()));
/// Progress evidence is written infrequently and the files are small, so a single in-process lock
/// guarantees that snapshots for the same task cannot truncate each other through the shared temp
/// path; ordering is also compared before persisting, so a late stale snapshot cannot overwrite
/// newer evidence.
static TASK_PROGRESS_FILE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
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

fn append_task_progress_snapshots(output: &mut String, task_ids: &[String]) {
    let snapshots = task_ids
        .iter()
        .filter_map(|task_id| {
            current_task_progress(task_id).map(|progress| format!("- {task_id}: {progress}"))
        })
        .collect::<Vec<_>>();
    if snapshots.is_empty() {
        return;
    }
    output.push_str("\nLatest progress snapshots:\n");
    output.push_str(&snapshots.join("\n"));
}

/// Receives the unified structured snapshots published by subagents. In-memory state updates are
/// not throttled; status-line refresh notifications and disk snapshots are each edge-throttled
/// separately, while checkpoints always propagate immediately — so long tasks neither stay silent
/// nor cause an event storm.
pub(crate) fn record_subagent_progress_update(
    task_id: &str,
    snapshot: &crate::ai::driver::runtime_ctx::SubagentProgressSnapshot,
    kind: crate::ai::driver::runtime_ctx::SubagentProgressEventKind,
) {
    let now = Instant::now();
    let update = {
        let mut registry = TASK_REGISTRY.lock().unwrap();
        let Some(entry) = registry.get_mut(&task_id.to_string()) else {
            return;
        };
        let persist = kind
            == crate::ai::driver::runtime_ctx::SubagentProgressEventKind::Checkpoint
            || entry.last_progress_persisted_at.is_none_or(|last| {
                now.saturating_duration_since(last) >= SUBAGENT_PROGRESS_PERSIST_INTERVAL
            });
        let notify = kind
            == crate::ai::driver::runtime_ctx::SubagentProgressEventKind::Checkpoint
            || entry.last_progress_notification_at.is_none_or(|last| {
                now.saturating_duration_since(last) >= SUBAGENT_PROGRESS_NOTIFY_INTERVAL
            });
        if persist {
            entry.last_progress_persisted_at = Some(now);
        }
        if notify {
            entry.last_progress_notification_at = Some(now);
        }
        (persist, notify)
    };

    if update.0 && persist_subagent_progress_snapshot(task_id, snapshot).is_err() {
        let mut registry = TASK_REGISTRY.lock().unwrap();
        if let Some(entry) = registry.get_mut(&task_id.to_string())
            && entry.last_progress_persisted_at == Some(now)
        {
            // Undo the throttle marker when this persist fails, so the next progress event retries
            // immediately.
            entry.last_progress_persisted_at = None;
        }
    }
    if update.1 {
        // Progress events only refresh the foreground status line (notify_scheduler makes the
        // scheduling loop tick once to redraw the subagent status line); they must **never**
        // wake_process a parked parent agent back to Ready. Otherwise a progress event every 15s
        // would force the parent through a full model-call round trip just to park again, turning
        // the foreground into busy-wait spinning that burns an inference every ~9s (see the
        // agent-team fan-out stall). Only three terminal paths genuinely need to wake the parent,
        // and each triggers independently without relying on this code:
        //   1. Subtask finished/failed: channel_send + futex_store -> notify_events_completed;
        //   2. task_wait budget exhausted: wake_expired_task_waits;
        //   3. task_cancel: flips the cancel futex directly.
        crate::ai::driver::notify_scheduler();
    }
}

fn task_progress_file_path(task_id: &str) -> Result<PathBuf, String> {
    if task_id.is_empty()
        || !task_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        return Err("invalid task id for progress evidence".to_string());
    }
    let assets = current_session_assets_dir()
        .ok_or_else(|| "session assets directory is unavailable".to_string())?;
    Ok(assets.join("task-progress").join(format!("{task_id}.json")))
}

fn persist_subagent_progress_snapshot(
    task_id: &str,
    snapshot: &crate::ai::driver::runtime_ctx::SubagentProgressSnapshot,
) -> Result<(), String> {
    let path = task_progress_file_path(task_id)?;
    persist_subagent_progress_snapshot_at(&path, task_id, snapshot)
}

fn persist_subagent_progress_snapshot_at(
    path: &std::path::Path,
    task_id: &str,
    snapshot: &crate::ai::driver::runtime_ctx::SubagentProgressSnapshot,
) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "progress evidence path has no parent".to_string())?;
    let elapsed_ms = snapshot.elapsed.as_millis().min(u64::MAX as u128) as u64;
    let timeline = snapshot
        .timeline
        .iter()
        .map(|event| {
            serde_json::json!({
                "sequence": event.sequence,
                "kind": event.kind.as_str(),
                "elapsed_ms": event.elapsed.as_millis().min(u64::MAX as u128) as u64,
                "summary": event.summary,
            })
        })
        .collect::<Vec<_>>();
    let value = serde_json::json!({
        "version": 1,
        "task_id": task_id,
        "phase": snapshot.phase,
        "checkpoint_summary": snapshot.checkpoint_summary,
        "elapsed_ms": elapsed_ms,
        "stale_for_ms": snapshot.stale_for.as_millis().min(u64::MAX as u128) as u64,
        "checkpoint_due": snapshot.checkpoint_due,
        "sequence": snapshot.sequence,
        "timeline": timeline,
    });
    let bytes = serde_json::to_vec_pretty(&value).map_err(|err| err.to_string())?;
    let candidate_order = (snapshot.sequence, elapsed_ms);
    let _guard = TASK_PROGRESS_FILE_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if let Ok(existing) = fs::read(path)
        && let Ok(existing) = serde_json::from_slice::<Value>(&existing)
        && existing.get("task_id").and_then(Value::as_str) == Some(task_id)
        && let (Some(sequence), Some(elapsed_ms)) = (
            existing.get("sequence").and_then(Value::as_u64),
            existing.get("elapsed_ms").and_then(Value::as_u64),
        )
        && (sequence, elapsed_ms) >= candidate_order
    {
        return Ok(());
    }

    fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    let temp_path = path.with_extension(format!("json.{}.tmp", Uuid::new_v4().simple()));
    fs::write(&temp_path, bytes).map_err(|err| err.to_string())?;
    if let Err(error) = fs::rename(&temp_path, path) {
        let _ = fs::remove_file(&temp_path);
        return Err(error.to_string());
    }
    Ok(())
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

/// Task id list produced by the most recent successful `task_spawn` / `task_spawn_batch`, used to
/// detect the "lone spawn" anti-pattern: spawning a single task and immediately `task_wait`-ing to
/// collect it. That scenario gains no concurrency and should use the synchronous `task` tool
/// (spawn + wait is only slower). The hint is light normative guidance: it fires once, never
/// rejects or blocks, and the model may ignore it (e.g. when it really did interleave parent-side
/// work between spawn and wait).
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

/// If this wait's task_ids match "the most recent spawn was a single task" and the hint has not
/// been shown yet, return the normative hint text once (consuming the hinted flag so it fires at
/// most once per whole session turn).
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
    progress: Option<String>,
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
    /// Subagent nesting depth: 1 for a top-level spawn, incrementing per level. Used to prevent
    /// recursive fan-out.
    #[serde(default)]
    pub(crate) spawn_depth: usize,
    /// Optional JSON Schema for the subagent's final response; stays compatible with old task
    /// payloads where it is missing.
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

/// Runs a mutable operation on the AIOS kernel.
///
/// Preferred path: take the `SharedKernel` held by the current turn from the `DRIVER_CTX`
/// task-local, so high-frequency paths such as `task_wait` / `task_spawn` reuse the Arc the turn
/// scope already holds, avoiding the extra lock and indirection of the `GLOBAL_OS` global static.
///
/// Fallback path: when the caller is not inside a `DRIVER_CTX` scope (e.g. early driver startup or
/// a unit test invoking the tool from a synchronous context), fall back to `GLOBAL_OS` for backward
/// compatibility.
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

/// Time remaining until the scheduler must next check task_wait wall-clock deadlines.
///
/// A delayed notify is only an early wake-up signal and cannot be the sole source of truth for
/// the deadline: if the notification task was not registered successfully or a notify race
/// occurs, the scheduler must still wake itself up at the real deadline returned here.
pub(crate) fn next_task_wait_wakeup_delay() -> Option<Duration> {
    let now = Instant::now();
    TASK_WAIT_STATES
        .lock()
        .unwrap()
        .values()
        .filter(|state| !state.expired)
        .map(|state| state.deadline.saturating_duration_since(now))
        .min()
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

#[cfg(test)]
pub(crate) fn task_wait_state_count_for_test() -> usize {
    TASK_WAIT_STATES.lock().unwrap().len()
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

    // Wake each expired owner one by one; meanwhile record the wait keys whose owner process no
    // longer exists or has terminated. wake_process only returns an owner in the Waiting state to
    // Ready (when not Waiting the owner is already scheduled, so a lost wake-up is harmless), but
    // once the owner is Terminated the wake-up misses entirely and its wait state stays in the
    // global table forever with expired=true — leaking memory and leaving no cleanup path beyond
    // next_task_wait_wakeup_delay. These orphan keys are collected here and then deleted in bulk
    // to self-heal.
    let orphaned = with_os_kernel(|os| {
        let mut woken: SkipSet<u64> = SkipSet::default();
        let mut orphaned: Vec<TaskWaitKey> = Vec::new();
        for (key, state) in expired {
            let owner_alive = os
                .get_process(key.owner_pid)
                .is_some_and(|proc| !matches!(proc.state, ProcessState::Terminated));
            if !owner_alive {
                orphaned.push(key);
                continue;
            }
            if !woken.insert(key.owner_pid) {
                continue;
            }
            let task_ids = key.task_ids.join(", ");
            let mut message = format!(
                "[TASK_WAIT_TIMEOUT]\nWall-clock task_wait budget elapsed after {}s. Re-call `task_wait` with the same task_ids to collect any ready results and receive the budget-elapsed status. task_ids=[{}]",
                state.timeout_secs, task_ids
            );
            append_task_progress_snapshots(&mut message, &key.task_ids);
            let _ = os.wake_process(key.owner_pid, message);
        }
        Ok(orphaned)
    })
    .unwrap_or_default();

    if !orphaned.is_empty() {
        let mut states = TASK_WAIT_STATES.lock().unwrap();
        for key in orphaned {
            states.remove(&key);
        }
    }
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

/// Combines the kernel's epoll / channel / futex / event primitives at the agent layer to
/// implement a "wait for any of several sources to complete" semantic across **multiple wait
/// source kinds**, primarily serving the `task_wait` tool.
///
/// **Design positioning**: this function does *not* re-implement the kernel's wait primitives; it
/// assembles several low-level APIs (`epoll_create` / `epoll_ctl` / `epoll_wait` /
/// `wait_on_events`) to the agent's business semantics:
/// 1. Build a short-lived epoll set for channel/futex-style wait sources, then `epoll_wait` for
///    the ready set;
/// 2. For event-style wait sources, call `wait_on_events` directly;
/// 3. Normalize both kinds of results into `EpollWaitManyOutcome`.
///
/// **Future lowering suggestion**: once the kernel gains native syscall support for
/// `Vec<WaitManySource>` (similar to a hybrid of epoll_pwait2 + EVENTFD), this function can become
/// a thin wrapper around a single syscall. Until that migration, this function keeps the current
/// multi-step composite implementation; any behavior change **must keep task_wait regression-free
/// in the following scenarios**:
/// - all sources ready: return immediately (epoll_wait is not called);
/// - all sources pending: decide whether to actually suspend according to `wait_policy`;
/// - mixed ready + pending: return only the ready set, without adding extra blocking.
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
                    // epoll_wait internally consumed the yield_requested flag to decide whether it
                    // suspended; it must be re-set here, otherwise the turn-loop's
                    // consume_yield_requested() reads false, control is never returned to the
                    // scheduler, and a ready subagent is never dispatched.
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
                    // Same as above: this branch probes suspension via consume_yield_requested(),
                    // which clears the yield intent. Once suspension is confirmed, re-set the flag
                    // so the turn-loop can notice it and hand control back to the scheduler.
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
        groups: &["builtin", "task"],
    }
});

// `task` / `task_spawn` / `task_spawn_batch` / `task_wait` / `task_status`
// may all carry the only visible result of a subagent: the spawn-family arguments (subagent
// prompt / response schema) and return values (the task_id list) are required inputs for the
// later wait/status/integrate calls; once a result is lossy-compressed or LLM-pruned, the main
// agent can lose its grounding on already-finished subtasks. Lossy compression and pruning are
// uniformly banned here; oversized content goes to an overflow stub + file_path instead of being
// reduced to an unrecoverable summary.
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
        groups: &["builtin", "task"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_spawn_batch",
        description: "",

        execute: execute_task_spawn_batch,
        groups: &["builtin", "task"],
    }
});

inventory::submit!(ToolHistoryPolicyRegistration {
    name: "task_spawn",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Never,
        counts_toward_precision_inline_budget: false,
    },
});

inventory::submit!(ToolHistoryPolicyRegistration {
    name: "task_spawn_batch",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Never,
        counts_toward_precision_inline_budget: false,
    },
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

    // Prefer the agent_manifests cached in DRIVER_CTX, so each task_spawn does not re-read the
    // disk. When not inside a DRIVER_CTX scope (rare, e.g. unit tests), fall back to
    // load_all_agents().
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
                last_progress_notification_at: None,
                last_progress_persisted_at: None,
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

/// Associates the Tokio task that actually runs the subagent, so cancellation and timeout can stop
/// the background Future instead of only terminating the logical process in the kernel.
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

/// Read-only snapshot used by the foreground status bar. Only exposes the fields needed for
/// display; the subagent body text is still returned exclusively through `task_wait` /
/// `task_status`, keeping background tasks from contending for the terminal.
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
    record_subagent_spawn_audit(&spawned.task_id, &prepared);

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

    // Complete the whole batch preflight first, so earlier children are not already started when a
    // later entry has invalid arguments.
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
                record_subagent_spawn_audit(&spawned.task_id, task);
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
        groups: &["builtin", "task"],
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
        groups: &["builtin", "task"],
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

    // Wait budget for this single task_wait call. See the DEFAULT_TASK_WAIT_TIMEOUT_SECS comment —
    // a timeout only means this call did not get the result; the subagent keeps running and no
    // resources are released.
    let (timeout_secs, wait_policy) = parse_task_wait_options(args)?;

    // wait_policy: "any" | "all", default "any", so the foreground is not held up by the slowest
    // task.
    // - all  — return only after every pending task completes (for when results must be
    //          aggregated);
    // - any  — return as soon as any pending task completes, the rest keep running and can be
    //          collected by further task_wait calls (for fan-out where results are gathered as
    //          they arrive).
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
    // A registry miss can mean either the task was normally cleaned up after delivery, or the
    // model mistyped an id, referenced a stale cross-process id, or hit a registry anomaly. Only a
    // tombstone actually present in the durable evidence ledger proves the task was delivered.
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
    // Mixing already-delivered ids with still-pending ones is expected input: PARKED /
    // BUDGET-ELAPSED asks the model to keep waiting with the same set of ids. Only ids confirmed
    // delivered by the ledger are dropped here.
    let task_ids = task_ids_filtered;
    if task_ids.is_empty() {
        return Ok(format!(
            "[task_wait] All {} referenced task(s) already completed and \
             their results were delivered by an earlier task result tool call. No tasks remain to \
             wait on; continue reasoning with the results you already collected.",
            already_delivered.len()
        ));
    }
    // lone-spawn normative hint: computed only once (consuming the hinted flag), then uniformly
    // appended at every later return point.
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
    // Collect the task_ids finished in this call (success / failure, channel/futex destroyed,
    // needing removal from the registry); the suspended and budget-elapsed early-return paths also
    // use it for cleanup.
    let mut finished: Vec<String> = Vec::new();
    // `write_terminal_subagent_result` only terminates the kernel process; it does not stop the
    // hosting Tokio Future. So first abort every worker that exceeded its total lifetime, then
    // enter the kernel critical section to publish the terminal state.
    for tid in &task_ids {
        let entry = registry.get_ref(tid).expect("validated");
        if entry.started_at.elapsed() > SUBAGENT_WALL_CLOCK_TIMEOUT {
            entry.cancel_stream.store(true, Ordering::Release);
            if let Some(handle) = &entry.abort_handle {
                handle.abort();
            }
        }
    }
    // The closure borrows wait_policy / registry / pending / ready / finished by reference by
    // default; no `move` is added, so code after the closure (e.g. `if !pending.is_empty()`) can
    // still access them.
    let wait_message = with_os_kernel(|os| {
        for tid in &task_ids {
            let entry = registry.get_ref(tid).expect("validated");
            // ⚠️ This block previously marked a task TIMEOUT and destroyed its channel/futex as
            // soon as `entry.started_at.elapsed() >= timeout_secs` — that was a bug:
            // `started_at` is the spawn time, not the start of this task_wait call. If the main
            // agent first calls task_wait long after spawning, every task would be **immediately**
            // reported as TIMEOUT and its result_channel destroyed, permanently losing the real
            // subagent result while the main agent naturally concludes "subagent is stuck".
            //
            // Current behavior: only look at whether a ready payload exists on the channel; if
            // not, uniformly take the pending branch. The per-call task_wait budget is governed by
            // the real wall-clock deadline in TASK_WAIT_STATES; the driver run_loop wakes the
            // owner process when it expires, and only the next task_wait returns BUDGET ELAPSED.
            // Budget exhaustion also **never destroys the channel/futex**, so the main agent can
            // keep calling task_wait to wait longer.
            // Wall-clock total-lifetime check: if a subagent still has no result past
            // SUBAGENT_WALL_CLOCK_TIMEOUT (typically stuck in a single tool execution that never
            // returns), terminate it proactively and write a timeout terminal result, so the
            // immediately following read_task_result reads a result and the main agent does not
            // spin in a "timeout -> wait again -> timeout" loop. Unlike the historical bug that
            // compared started_at against the per-call timeout_secs, this uses an independent
            // total-lifetime cap far larger than a single wait budget, and writes a failure result
            // instead of destroying the channel, so no result is lost.
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
                // When a subagent process terminated without publishing a result, it never ran its
                // own cleanup to release the producer holder, so both the consumer and the producer
                // must be released here; otherwise channel_destroy fails on a non-zero ref_count
                // and the channel + futex leak permanently.
                let rendered = collect_missing_task_result(tid, entry)?;
                cleanup_collected_task(os, entry, "subagent terminated without output");
                ready.push(rendered);
                finished.push(tid.clone());
            }
        }

        // With `any`, once the first scan already collected a result, we must return immediately
        // instead of being suspended by the remaining pending tasks.
        if !pending.is_empty()
            && !wait_budget_elapsed
            && !(wait_policy == WaitPolicy::Any && !ready.is_empty())
        {
            let pending_ids = pending
                .iter()
                .map(|(tid, _)| tid.clone())
                .collect::<Vec<_>>();
            let wait_sources = task_wait_sources(os, &pending_ids, &registry)?;
            // `task_wait`'s `wait_policy=all` is tool-layer semantics: all task results must be
            // collected before returning. The underlying park cannot use `WaitPolicy::All` to wait
            // on every event source, because the sources also include the cancel futex used to
            // interrupt the current process, which never completes on the normal path. So we wait
            // for "any task event" to wake us, then re-scan all task states; if everything is not
            // collected yet, the model can call task_wait again with the same task_ids.
            let wait = epoll_wait_many(
                os,
                &format!("task_wait:{}", pending_ids.join(",")),
                &wait_sources,
                WaitPolicy::Any,
                None,
            )?;
            // Whether or not epoll_wait_many suspended, always re-scan first to collect results
            // that became ready during the wait. If it suspended and all tasks are now complete,
            // return the results directly (instead of PARKED), so the model is not forced to call
            // task_wait repeatedly under wait_policy=all. PARKED is returned only when tasks are
            // still pending after the re-scan and the wait really did suspend.
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
            // Tasks are still pending after the re-scan and the wait really suspended (a
            // cooperative yield, not budget exhaustion), so return PARKED with the partial results
            // collected so far. Terminal wording like "BUDGET ELAPSED" must **never** be used
            // here: a suspend returns synchronously within milliseconds (it does not really wait
            // the full timeout_secs), otherwise the model would misread "timeout right after
            // starting to wait" as "subtask stuck" and give up early to fall back to manual
            // analysis.
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
                let still_pending = pending
                    .iter()
                    .map(|(tid, _)| tid.clone())
                    .collect::<Vec<_>>();
                let mut parked_message = format!(
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
                );
                append_task_progress_snapshots(&mut parked_message, &still_pending);
                parts.push(parked_message);
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
        let pending_ids = pending
            .iter()
            .map(|(tid, _)| tid.clone())
            .collect::<Vec<_>>();
        let policy_label = match wait_policy {
            WaitPolicy::Any => "any",
            WaitPolicy::All => "all",
        };
        let mut elapsed_message = format!(
            "[task_wait BUDGET ELAPSED] {} pending subagent task(s) still running in the background. \
            wait_policy={policy_label}, timeout_secs={timeout_secs}. The subagent(s) are NOT stalled and NOT cancelled; \
            their result channels and completion futexes remain alive. \
            Pending task_ids: [{}]. \
            Next steps: call `task_status` for a snapshot, or call `task_wait` again with the same task_ids to keep waiting \
            (consider `wait_policy=\"any\"` if you only need the first finisher).",
            pending.len(),
            pending_ids.join(", ")
        );
        append_task_progress_snapshots(&mut elapsed_message, &pending_ids);
        parts.push(elapsed_message);
        // Only remove the registry entries for task_ids that are already ready; pending tasks must
        // be kept, otherwise the next task_wait fails with "Unknown task_id".
        let pending_set: SkipSet<&str> = pending_ids.iter().map(String::as_str).collect();
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

/// Writes a terminal result to a subagent's result channel and terminates its kernel process.
/// Used for task_cancel (explicit cancellation) and wall-clock total-lifetime timeout. The result
/// uses the same status/output/error format as `publish_background_task_failure`, so the
/// task_wait / task_status collection paths read it normally. This function only releases the
/// producer-side named ownership and stores the futex to wake the waiter; destroying the
/// channel/futex is left to the collector (task_wait's ready path or task_cancel itself) to avoid
/// double release.
fn write_terminal_subagent_result(
    os: &mut dyn aios_kernel::kernel::Kernel,
    task_id: &str,
    pid: u64,
    result_channel_id: u64,
    completion_futex_addr: aios_kernel::primitives::FutexAddr,
    status: &str,
    error: &str,
) {
    // The caller must first abort the Tokio task actually running the subagent; the kernel process
    // state alone does not stop the Future inside the hosting process. Only then terminate the
    // kernel process and publish the terminal result.
    let _ = os.kill_process(pid, format!("{}: {}", status, error));
    // Then write the terminal result as the subagent and release the producer side (the result
    // channel's producer ownership check requires current == pid). Although the process is already
    // terminated, the channel/futex resources are not reclaimed yet (that happens in
    // drop_terminated), so the write is still valid.
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

/// Called by the scheduler every epoch: scans TASK_REGISTRY for subagents still running past the
/// wall-clock total-lifetime cap, terminates their processes, and writes a timeout terminal
/// result.
///
/// Complements the wall-clock check inside `task_wait`: task_wait only triggers when the main
/// agent actively calls it; this function proactively scans every epoch of the driver run_loop,
/// so even if the main agent is busy elsewhere (not calling task_wait for a long time), a stuck
/// subagent process is still terminated promptly instead of occupying scheduler resources
/// forever.
///
/// Resource semantics: only kill the process + write the terminal result; the channel/futex are
/// **not** destroyed and the entry is **not** removed from the registry — those are left to the
/// collector (task_wait's ready path) to avoid double release. After the process is killed,
/// `is_task_pending` returns false, so a later epoch scanning the same entry skips it and does not
/// kill it again.
///
/// Lock ordering: three steps to avoid forming a lock cycle with task_wait (registry -> kernel) —
/// 1. Lock only TASK_REGISTRY to collect the candidates, then release immediately;
/// 2. Holding no lock, abort the Tokio task actually running the subagent;
/// 3. Lock only the kernel (via with_os_kernel) to kill + write the result.
/// The registry and the kernel are never held simultaneously (GLOBAL_OS and App.os are the same
/// lock; see the reentrant-deadlock warning in os_tools.rs).
pub(crate) fn reap_timed_out_subagents() {
    // Step 1: hold only the registry lock to collect timed-out candidates (pid / channel /
    // futex), then release immediately.
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
    // Step 2: holding no lock, stop the real Tokio Future first so it cannot write a result
    // concurrently with the timeout terminal state.
    for (_, _, _, _, abort_handle, cancel_stream) in &candidates {
        cancel_stream.store(true, Ordering::Release);
        if let Some(handle) = abort_handle {
            handle.abort();
        }
    }
    // Step 3: hold only the kernel lock and check each process; if still running, kill it + write
    // the timeout terminal state.
    let _ = with_os_kernel(|os| {
        for (task_id, pid, result_channel_id, completion_futex_addr, _, _) in candidates {
            if !is_task_pending(os, pid)? {
                // The process already ended (completed normally / failed / killed by someone
                // else); skip it — result and resource cleanup are handled by the collector.
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
        groups: &["builtin", "task"],
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

pub(crate) fn execute_task_evidence_read(args: &Value) -> Result<String, String> {
    ensure_top_level_task_orchestration("task_evidence_read")?;
    let task_id = args
        .get("task_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or("Missing non-empty 'task_id' parameter")?;
    let owner_pid = current_task_owner_pid()?;
    if let Some(entry_owner) = with_task_entry(task_id, |entry| entry.owner_pid) {
        if entry_owner != owner_pid {
            return Err(format!("Task {task_id} is owned by another process"));
        }
    }

    if let Some(slot) = task_progress_slot(task_id)
        && let Some(snapshot) =
            crate::ai::driver::runtime_ctx::subagent_progress_state_snapshot(&slot)
    {
        persist_subagent_progress_snapshot(task_id, &snapshot).map_err(|error| {
            format!("Failed to persist current progress evidence for task {task_id}: {error}")
        })?;
    }
    let path = task_progress_file_path(task_id)?;
    match fs::read_to_string(&path) {
        Ok(content) => Ok(content),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Err(format!(
            "No persisted progress evidence for task {task_id}; the task may not have emitted a phase transition or checkpoint yet"
        )),
        Err(error) => Err(format!(
            "Failed to read progress evidence for task {task_id}: {error}"
        )),
    }
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_evidence_read",
        description: "",
        execute: execute_task_evidence_read,
        groups: &["builtin", "task"],
    }
});

inventory::submit!(ToolHistoryPolicyRegistration {
    name: "task_evidence_read",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Never,
        counts_toward_precision_inline_budget: false,
    },
});

/// Presents the complete subagent call ledger of the current session (spawn audit).
///
/// Complements `task_evidence_read` (per-task progress evidence): this is a model-visible audit
/// view answering "which subagents were called in this agent session, when, with which
/// agent/model, and whether results were delivered/integrated" — still queryable even after the
/// results were collected long ago or the history was compressed.
pub(crate) fn execute_task_audit(_args: &Value) -> Result<String, String> {
    ensure_top_level_task_orchestration("task_audit")?;
    let Some(context) = crate::ai::driver::runtime_ctx::try_current() else {
        return Err("No driver context available for task audit".to_string());
    };
    let records = crate::ai::history::read_task_spawn_audit(
        context.app_proto.config.history_file.as_path(),
        &context.app_proto.session_id,
    )
    .map_err(|error| format!("failed to read subagent spawn audit: {error}"))?;
    if records.is_empty() {
        return Ok(
            "No subagent calls recorded in this session yet. Use `task` / `task_spawn` / \
             `task_spawn_batch` to delegate work; every spawn is persisted here."
                .to_string(),
        );
    }
    let mut lines = Vec::with_capacity(records.len());
    for record in records.iter().rev() {
        let delivered = record.delivered_at_unix_ms;
        let integrated = match record.integrated_at_unix_ms {
            Some(ts) => format!("{ts}"),
            None => "no".to_string(),
        };
        lines.push(format!(
            "task_id={} status={} agent={} model={} delivered_ms={} integrated={}\n  description: {}",
            record.task_id, record.status, record.agent_name, record.model, delivered, integrated,
            record.description,
        ));
        if let Some(disposition) = &record.disposition {
            lines.push(format!("  disposition: {disposition}"));
        }
    }
    Ok(format!(
        "[subagent-call-audit] {} record(s) in this session (newest delivered first; undelivered at bottom):\n{}",
        records.len(),
        lines.join("\n")
    ))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_audit",
        description: "",
        execute: execute_task_audit,
        groups: &["builtin", "task"],
    }
});

inventory::submit!(ToolHistoryPolicyRegistration {
    name: "task_audit",
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
        groups: &["builtin", "task"],
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
        groups: &["builtin", "task"],
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

/// Terminates and removes all asynchronous subagents of a session when it is destroyed.
///
/// This path does not preserve collectable results: the parent session was explicitly deleted by
/// the user, so keeping the registry, IPC, or background Futures would only let the deleted
/// derived history be recreated.
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
    // First copy the cancellation info under the registry lock only, then release immediately;
    // neither the Tokio task abort nor the kernel terminal write may overlap with the registry
    // lock, to avoid forming a lock cycle with the collection path.
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

    // The real Tokio Future must be stopped before entering the kernel to write the cancelled
    // terminal state; otherwise, after the logical process is terminated, network requests or
    // tool calls can still run in the background and race with the terminal write.
    for (_, _, _, _, abort_handle, cancel_stream) in &candidates {
        cancel_stream.store(true, Ordering::Release);
        if let Some(handle) = abort_handle {
            handle.abort();
        }
    }

    for (tid, pid, result_channel_id, completion_futex_addr, _, _) in candidates {
        // Only cancel subagents that are still running. Tasks that already ended (completed
        // normally / failed / process terminated) are neither killed nor given a terminal result —
        // otherwise a "cancelled" message would be appended to the channel and the channel
        // destroyed, masking/discarding the subagent's real result and making a later task_wait
        // read a bogus cancelled state. Cleanup of channel/futex/registry for already-ended tasks
        // is left to the collector (task_wait's ready / failure path, or a task_wait after
        // task_status).
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
    // For subtasks that already wrote their result back to the channel, **consume and clean up**
    // the body directly and append it after the table. Otherwise, even if the model sees
    // state=completed, it can only get the output by calling task_wait again; worse, if it treats
    // "seen completed in task_status" as handled, it would bypass the collection guard and leave
    // registry entries and channel/futex resources behind. Since the result is already returned
    // to the model here, treat it as collected.
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
            if let Some(progress) = current_task_progress(tid) {
                lines.push(format!("  progress[{tid}]: {progress}"));
            }
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
                last_progress_notification_at: None,
                last_progress_persisted_at: None,
                abort_handle: None,
                cancel_stream: Arc::new(AtomicBool::new(false)),
            };
            if let Some(rendered) = collect_ready_task_result(os, tid, &entry)? {
                completed_outputs.push(rendered);
                cleanup_collected_task(os, &entry, "subagent result collected by task_status");
                finished_ids.push(tid.clone());
            } else if !is_task_pending(os, *pid)? {
                // Consistent with task_wait: when the process terminated without writing back a
                // result, the task must also be closed out and both sides of the channel
                // ownership released, so polling only task_status does not leak.
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
                progress: current_task_progress(task_id),
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
        if let Some(progress) = &snapshot.progress {
            lines.push(format!("  progress: {progress}"));
        }
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

/// When the iteration hard cap is reached and control will no longer be handed back to the model,
/// fold the still-uncollected subtask states into the final answer so uncollected results are not
/// silently dropped. Shares the snapshot collection with `build_outstanding_task_anchor`, but the
/// wording targets the final output: the model gets no further chance to act, so instead of asking
/// it to "call task_wait next", it tells the user which subtask results were not collected and
/// need to be re-collected.
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
        if let Some(progress) = &snapshot.progress {
            lines.push(format!("  progress: {progress}"));
        }
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

/// After a successful spawn, writes a placeholder audit record (delivered_at=0); when the result
/// is delivered, `record_delivered_task_evidence` overwrites the status fields for the same
/// task_id, so the ledger can always answer "whether the current agent called a subagent, when,
/// and with which agent/model". The audit write is best-effort: failure does not block the spawn
/// itself.
fn record_subagent_spawn_audit(task_id: &str, prepared: &PreparedSubagentTask) {
    let Some(context) = crate::ai::driver::runtime_ctx::try_current() else {
        return;
    };
    let _ = crate::ai::history::record_task_spawn_audit(
        context.app_proto.config.history_file.as_path(),
        &context.app_proto.session_id,
        task_id,
        &prepared.description,
        &prepared.agent_name,
        &prepared.model,
    );
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

/// Reads the persisted evidence state (status + payload) for the given task_id in the current
/// session. Returns None when there is no driver context or nothing has been persisted (a lenient
/// semantic consistent with the persisting side).
fn read_persisted_task_evidence_status(
    session_id: &str,
    task_id: &str,
) -> Result<Option<(String, String)>, String> {
    let Some(context) = crate::ai::driver::runtime_ctx::try_current() else {
        return Ok(None);
    };
    crate::ai::history::read_task_evidence_status_payload(
        context.app_proto.config.history_file.as_path(),
        session_id,
        task_id,
    )
    .map_err(|error| format!("failed to read task evidence for {task_id}: {error}"))
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

pub(super) enum OwnedTaskPoll {
    Pending {
        state: String,
    },
    Terminal {
        result: StoredTaskResult,
        rendered: String,
    },
}

/// Non-blocking collection entry point used by the Team/Graph orchestrator.
///
/// Follows the same truth path as `task_status`: persist evidence first, then
/// consume the channel, and finally clean up IPC and the registry. This way the
/// graph executor does not build a second subagent result protocol and cannot
/// bypass the durable evidence that context rebuild depends on.
pub(super) fn poll_owned_task_result(task_id: &str) -> Result<OwnedTaskPoll, String> {
    let current_session_id = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
    let current_owner_pid = current_task_owner_pid()?;
    let entry = {
        let registry = TASK_REGISTRY.lock().unwrap();
        let entry = match registry.get_ref(&task_id.to_string()) {
            Some(entry) => entry,
            None => {
                // Entry missing: in the normal flow an entry is only removed
                // after the task is polled to Terminal and its evidence is
                // persisted. If this session already persisted evidence for the
                // task (the only path for the H1 regression "polled but
                // checkpoint not saved"), return an idempotent Terminal so the
                // graph/team checkpoint can self-heal instead of hanging
                // forever; without evidence this is still a genuine error.
                if let Some((status, payload)) =
                    read_persisted_task_evidence_status(&current_session_id, task_id)?
                {
                    return Ok(OwnedTaskPoll::Terminal {
                        result: StoredTaskResult {
                            status,
                            output: payload.clone(),
                            error: None,
                            progress: None,
                        },
                        rendered: payload,
                    });
                }
                return Err(format!("Unknown graph-managed task_id: {task_id}"));
            }
        };
        if !task_entry_owned_by(entry, &current_session_id, current_owner_pid) {
            return Err(format!(
                "Task {task_id} is not owned by the current process/session"
            ));
        }
        entry.clone()
    };

    let poll = with_os_kernel(|os| {
        if let Some(result) = read_task_result(os, entry.result_channel_id, false)? {
            let rendered = format_task_result_with_id(task_id, &entry, result.clone());
            persist_rendered_task_evidence(task_id, &entry, &result.status, &rendered)?;
            if read_task_result(os, entry.result_channel_id, true)?.is_none() {
                return Err(format!(
                    "task result for {task_id} disappeared after durable persistence"
                ));
            }
            mark_task_retry_status(task_id, &result.status);
            TASK_PROGRESS_REGISTRY
                .lock()
                .unwrap()
                .take(&task_id.to_string());
            cleanup_collected_task(os, &entry, "subagent result collected by agent graph");
            return Ok(OwnedTaskPoll::Terminal { result, rendered });
        }

        if is_task_pending(os, entry.pid)? {
            return Ok(OwnedTaskPoll::Pending {
                state: task_state_string(os, entry.result_channel_id, entry.pid)?,
            });
        }

        let result = StoredTaskResult {
            status: "failed".to_string(),
            output: String::new(),
            error: Some(format!(
                "Subagent process pid={} terminated without publishing any output.",
                entry.pid
            )),
            progress: current_task_progress(task_id),
        };
        let rendered = format_task_result_with_id(task_id, &entry, result.clone());
        persist_rendered_task_evidence(task_id, &entry, &result.status, &rendered)?;
        mark_task_retry_status(task_id, &result.status);
        TASK_PROGRESS_REGISTRY
            .lock()
            .unwrap()
            .take(&task_id.to_string());
        cleanup_collected_task(
            os,
            &entry,
            "graph-managed subagent terminated without output",
        );
        Ok(OwnedTaskPoll::Terminal { result, rendered })
    })?;

    if matches!(poll, OwnedTaskPoll::Terminal { .. }) {
        TASK_REGISTRY.lock().unwrap().remove(&task_id.to_string());
    }
    Ok(poll)
}

/// After collecting a terminal result, release all IPC and terminate and reap
/// the corresponding kernel process.
///
/// The result payload may wake the parent agent before the driver finishes
/// updating the final process state; therefore the collector cannot rely on
/// `drop_terminated` alone. Whether the process is still Ready/Running or
/// already Terminated, terminal collection closes out the one-shot subagent
/// task so the process table does not grow indefinitely.
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

    // Defend against tests or a corrupted registry registering the foreground
    // owner itself as a task pid; collecting a subagent result must never
    // terminate a still-running parent process. If it has already terminated,
    // the cleanup below still runs normally and drops it.
    if entry.pid == entry.owner_pid
        && !matches!(
            os.get_process(entry.pid).map(|process| &process.state),
            Some(ProcessState::Terminated)
        )
    {
        return;
    }

    // kill_process uses the current pid for parent/child permission checks. The
    // normal path keeps the owner as current; when the session owner is gone we
    // fall back to the child terminating itself, so deleting a session never
    // leaves an orphan behind.
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

/// Extract the set of 2-4 character n-grams from text (lowercased, whitespace
/// collapsed and normalized).
/// Used only for set-similarity scoring; carries no term-frequency /
/// inverse-document-frequency weights.
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

/// Subagent auto-selection: score by Jaccard overlap of the normalized-text
/// character n-gram sets.
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
