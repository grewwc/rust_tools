use super::*;

pub(super) const MAX_TASK_REGISTRY_SIZE: usize = 100;
pub(super) const DEFAULT_TASK_PRIORITY: u8 = 20;
pub(super) const DEFAULT_TASK_QUOTA_TURNS: usize = 10;
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
pub(super) const MAX_SUBAGENT_SPAWN_BATCH_SIZE: usize = 8;
pub(super) const TASK_GOAL_PREFIX: &str = "AIOS_SUBAGENT_TASK:";
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
pub(super) const DEFAULT_TASK_WAIT_TIMEOUT_SECS: u64 = 30;
/// Hard upper bound for `task_wait.timeout_secs`, so the model cannot set an astronomically large
/// timeout that would block the driver indefinitely. After at most 60 seconds, control must return
/// to the parent agent, which re-evaluates whether to continue local work, check status
/// non-blockingly, or genuinely wait again.
pub(super) const MAX_TASK_WAIT_TIMEOUT_SECS: u64 = 60;

/// Wall-clock lifetime cap for a subagent. Unlike a single task_wait's `timeout_secs` (default 30s,
/// max 60s), this is a process-level hard limit: when a subagent outlives it (typically by getting
/// stuck in a single tool execution that never returns, where a single turn has no wall-clock
/// timeout), the task_wait entry point proactively terminates it and writes a timeout terminal
/// result, so the main agent does not spin in a "timeout -> wait again -> timeout" loop or leave
/// background processes holding resources forever. One hour far exceeds normal completion times
/// and only acts as a safety net for genuinely stuck subagents.
pub(super) const SUBAGENT_WALL_CLOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60 * 60);
pub(super) const SUBAGENT_PROGRESS_NOTIFY_INTERVAL: Duration = Duration::from_secs(15);
pub(super) const SUBAGENT_PROGRESS_PERSIST_INTERVAL: Duration = Duration::from_secs(10);

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
    ///   - list of: history, memory, cwd, skills. The separators ',', '+' and '/'
    ///     are all accepted, so a value copied out of prose style ("cwd+skills",
    ///     "history/cwd") parses the same as the canonical "cwd,skills". Models
    ///     tend to transcribe the description's own wording verbatim, and strict
    ///     comma-only parsing made that copy a hard tool-call failure.
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
        for part in trimmed.split(&[',', '+', '/'][..]) {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum TaskWaitPolicyKey {
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
pub(super) struct TaskWaitKey {
    pub(super) session_id: String,
    pub(super) owner_pid: u64,
    pub(super) wait_policy: TaskWaitPolicyKey,
    pub(super) task_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TaskWaitState {
    pub(super) deadline: Instant,
    pub(super) timeout_secs: u64,
    pub(super) expired: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct OutstandingTaskSnapshot {
    pub(super) task_id: String,
    pub(super) status: String,
    pub(super) agent_name: String,
    pub(super) model: String,
    pub(super) description: String,
    pub(super) progress: Option<String>,
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
