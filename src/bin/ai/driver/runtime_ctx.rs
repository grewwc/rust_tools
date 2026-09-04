// =============================================================================
// AIOS Driver Runtime Context - Sub-agent dispatch context bridge
// =============================================================================
// `DRIVER_CTX` is a `tokio::task_local!` that exposes a snapshot of the
// pieces required to spawn a sub-agent's `run_turn` from inside a tool
// invocation.
//
// It is set up once per foreground/background turn in `driver::run_loop`
// and inherited by every nested `tokio::spawn` that participates in
// sub-agent dispatch (see `task_tools::execute_task`).
//
// Holding `Arc<DriverContext>` keeps the structure cheap to clone while
// still letting tools synthesise a fresh `task_app` for the spawned
// sub-agent without having to plumb additional parameters through every
// tool call.
//
// In addition to the parent-runtime snapshot, this module exposes several
// finer-grained task-locals that drive persona isolation plus the
// `inherit.memory` / `inherit.cwd` flags of the `task` / `task_spawn`
// tools:
//
//   - `PERSONA_MEMORY_PATH` overrides `MemoryStore::from_env_or_config`
//     for the whole foreground turn so each persona gets an isolated
//     long-term memory / memo store.
//
//   - `SUBAGENT_MEMORY_PATH` overrides `MemoryStore::from_env_or_config`
//     more strongly than `PERSONA_MEMORY_PATH`, so a sub-agent that opted
//     out of `inherit.memory` writes / reads its own jsonl file instead of
//     the persona-shared one.
//
//   - `SUBAGENT_CWD` overrides the project-wide `effective_cwd()` helper
//     so tools that consult it (e.g. ripgrep / find / fingerprint) honour
//     the sub-agent's scoped working directory instead of the parent's.
//
//   - `AUTO_MODEL_FALLBACK` marks sub-agent turns whose model was chosen
//     automatically. Request failures in that scope may retry with another
//     healthy auto-selected model; explicit model overrides do not.
// =============================================================================

use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use crate::ai::{
    agents::AgentManifest, mcp::SharedMcpClient, models::AutoModelFallbackSpec,
    skills::SkillManifest, types::App,
};
use tokio::sync::Mutex;

/// Result published by a subagent to its parent. `parent_payload` may contain tool evidence for the parent to reuse;
/// `final_assistant_text` keeps the model's original final body for `response_schema` validation.
/// The two must stay separate so evidence wrapping cannot break structured JSON output.
#[derive(Clone, Debug, Default)]
pub(crate) struct SubagentResult {
    pub(crate) parent_payload: String,
    pub(crate) final_assistant_text: String,
}

/// Slot used by a sub-agent's `finalize_turn` to publish its result back to
/// the caller. The parent installs a fresh slot before invoking `run_turn`,
/// then reads it once the child returns.
pub(crate) type SubagentResultSlot = Arc<Mutex<SubagentResult>>;

const SUBAGENT_PROGRESS_TIMELINE_CAPACITY: usize = 12;
const SUBAGENT_CHECKPOINT_INTERVAL: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SubagentProgressEventKind {
    Phase,
    Checkpoint,
}

impl SubagentProgressEventKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Phase => "phase",
            Self::Checkpoint => "checkpoint",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SubagentProgressEvent {
    pub(crate) sequence: u64,
    pub(crate) kind: SubagentProgressEventKind,
    pub(crate) elapsed: Duration,
    pub(crate) summary: String,
}

/// Structured snapshot shared by all parent-observable paths. Display, wakeup, persistence, and hard-timeout recovery
/// all project from here, preventing each path from assembling a semantically inconsistent state of its own.
#[derive(Debug, Clone)]
pub(crate) struct SubagentProgressSnapshot {
    pub(crate) phase: String,
    pub(crate) checkpoint_summary: Option<String>,
    pub(crate) elapsed: Duration,
    pub(crate) stale_for: Duration,
    pub(crate) checkpoint_due: bool,
    pub(crate) sequence: u64,
    pub(crate) timeline: Vec<SubagentProgressEvent>,
}

impl SubagentProgressSnapshot {
    pub(crate) fn display(&self) -> String {
        let mut display = match self.checkpoint_summary.as_deref() {
            Some(summary) if !summary.is_empty() => {
                format!("{} · plan: {summary}", self.phase)
            }
            _ => self.phase.clone(),
        };
        display.push_str(&format!(
            " · elapsed {} · last activity {} ago",
            format_progress_duration(self.elapsed),
            format_progress_duration(self.stale_for)
        ));
        if self.checkpoint_due {
            display.push_str(" · checkpoint due");
        }
        display
    }
}

/// A subagent's live progress. The event timeline stays bounded; long-term diagnostic evidence is persisted at low frequency by the task layer.
#[derive(Debug, Clone)]
pub(crate) struct SubagentProgress {
    phase: String,
    checkpoint_summary: Option<String>,
    started_at: Instant,
    updated_at: Instant,
    last_checkpoint_at: Option<Instant>,
    last_checkpoint_reminder_at: Option<Instant>,
    sequence: u64,
    timeline: VecDeque<SubagentProgressEvent>,
}

impl SubagentProgress {
    fn new(phase: &str) -> Self {
        let now = Instant::now();
        let phase = compact_progress_text(phase, 80);
        let mut timeline = VecDeque::with_capacity(SUBAGENT_PROGRESS_TIMELINE_CAPACITY);
        timeline.push_back(SubagentProgressEvent {
            sequence: 1,
            kind: SubagentProgressEventKind::Phase,
            elapsed: Duration::ZERO,
            summary: phase.clone(),
        });
        Self {
            phase,
            checkpoint_summary: None,
            started_at: now,
            updated_at: now,
            last_checkpoint_at: None,
            last_checkpoint_reminder_at: None,
            sequence: 1,
            timeline,
        }
    }

    fn push_event(&mut self, kind: SubagentProgressEventKind, summary: String, now: Instant) {
        self.sequence = self.sequence.saturating_add(1);
        self.updated_at = now;
        if self.timeline.len() == SUBAGENT_PROGRESS_TIMELINE_CAPACITY {
            self.timeline.pop_front();
        }
        self.timeline.push_back(SubagentProgressEvent {
            sequence: self.sequence,
            kind,
            elapsed: now.saturating_duration_since(self.started_at),
            summary,
        });
    }

    fn snapshot(&self, now: Instant) -> SubagentProgressSnapshot {
        let checkpoint_reference = self.last_checkpoint_at.unwrap_or(self.started_at);
        SubagentProgressSnapshot {
            phase: self.phase.clone(),
            checkpoint_summary: self.checkpoint_summary.clone(),
            elapsed: now.saturating_duration_since(self.started_at),
            stale_for: now.saturating_duration_since(self.updated_at),
            checkpoint_due: now.saturating_duration_since(checkpoint_reference)
                >= SUBAGENT_CHECKPOINT_INTERVAL,
            sequence: self.sequence,
            timeline: self.timeline.iter().cloned().collect(),
        }
    }
}

/// Unlike `SubagentResultSlot`, this uses a synchronous lock: subagent writes never span an `.await`,
/// and the parent's foreground wait/refresh loop only takes a short snapshot.
pub(crate) type SubagentPhaseSlot = Arc<std::sync::Mutex<SubagentProgress>>;

/// One-shot wrap-up signal: the parent wait loop sets it when the reserved wrap-up time arrives, and the subagent consumes it before its next
/// model request, switching to the tool-free final answer mode.
pub(crate) type SubagentWrapUpSignal = Arc<AtomicBool>;

/// Snapshot of the live runtime that a sub-agent dispatch needs.
///
/// All fields are independently cloneable so that downstream consumers can
/// take what they need without holding a long-lived borrow on the
/// foreground turn.
pub(crate) struct DriverContext {
    /// Prototype `App` cloned from the parent turn. Mutate the clone, never
    /// the prototype.
    pub(crate) app_proto: App,
    pub(crate) mcp_client: SharedMcpClient,
    pub(crate) skill_manifests: Arc<Vec<SkillManifest>>,
    pub(crate) agent_manifests: Arc<Vec<AgentManifest>>,
}

impl DriverContext {
    pub(crate) fn from_app_snapshot(
        app: &App,
        mcp_client: SharedMcpClient,
        skill_manifests: Arc<Vec<SkillManifest>>,
        agent_manifests: Arc<Vec<AgentManifest>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            app_proto: app.snapshot_for_driver_context(),
            mcp_client,
            skill_manifests,
            agent_manifests,
        })
    }

    #[cfg(test)]
    pub(crate) fn new(
        app_proto: App,
        mcp_client: SharedMcpClient,
        skill_manifests: Arc<Vec<SkillManifest>>,
        agent_manifests: Arc<Vec<AgentManifest>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            app_proto,
            mcp_client,
            skill_manifests,
            agent_manifests,
        })
    }
}

tokio::task_local! {
    pub(crate) static DRIVER_CTX: Arc<DriverContext>;
    /// Memory file bound to the current persona. The foreground turn / one-shot note flows scope it
    /// in, keeping long-term memory fully isolated across personas.
    pub(crate) static PERSONA_MEMORY_PATH: PathBuf;
    /// When set, every `MemoryStore::from_env_or_config()` inside this
    /// task scope reads/writes from this path instead of the shared
    /// `RUST_TOOLS_MEMORY_FILE` / `ai.memory.file` location. Used by
    /// `inherit.memory == false` to give the sub-agent a private memory
    /// jsonl.
    pub(crate) static SUBAGENT_MEMORY_PATH: PathBuf;
    /// When set, every `runtime_ctx::effective_cwd()` consumer inside this
    /// task scope sees this directory as the active working directory
    /// instead of `std::env::current_dir()`. Used by `inherit.cwd ==
    /// false` to scope the sub-agent to a per-task scratch workspace.
    pub(crate) static SUBAGENT_CWD: PathBuf;
    /// When set, the sub-agent's `finalize_turn` publishes its final
    /// assistant text into this slot so the spawning tool can return it
    /// to the parent agent. Absence means "no parent is interested".
    pub(crate) static SUBAGENT_RESULT_SLOT: SubagentResultSlot;
    /// When set, `runtime_ctx::publish_subagent_phase` writes the sub-agent's
    /// current execution phase here so the spawning `task` tool's heartbeat
    /// line can surface it. Absence means "no parent is showing a heartbeat".
    pub(crate) static SUBAGENT_PHASE: SubagentPhaseSlot;
    /// Stable task id for background async tasks. The progress publisher uses it to persist the unified snapshot at low frequency and wake the
    /// corresponding parent process; a synchronous `task` has no such scope but can still show heartbeats via the shared slot.
    pub(crate) static SUBAGENT_TASK_ID: String;
    /// Set when the parent's synchronous wait is about to hit its hard timeout; the subagent consumes it once to request immediate wrap-up.
    pub(crate) static SUBAGENT_WRAP_UP_SIGNAL: SubagentWrapUpSignal;
    /// Background subagents do not own the terminal. Their full responses are still parsed, persisted, and returned to the parent through the
    /// result slot as usual, but streaming body, thinking, tool status, etc. must never write directly to stdout/stderr,
    /// or concurrent tasks would overwrite each other's cursor and garble foreground output.
    pub(crate) static SUPPRESS_TERMINAL_OUTPUT: bool;
    /// Subagent nesting depth. Unset for the top-level agent (equivalent to 0); incremented each time
    /// a subagent is spawned. Guards against recursive fan-out of `mode: all` heavy agents
    /// exhausting resources — `task_spawn` / `task` refuse to delegate further once `MAX_SUBAGENT_SPAWN_DEPTH`
    /// is exceeded.
    pub(crate) static SUBAGENT_DEPTH: usize;
    /// The current turn's (session_id, turn_id) tuple. The driver run_loop enters it before each
    /// turn's scheduling; DecisionLog / feedback write paths read it to attribute tool call
    /// results back to the right (session, turn). Downstream sees ("", 0) when unset.
    pub(crate) static TURN_IDENTITY: (String, usize);
    pub(crate) static AUTO_MODEL_FALLBACK: AutoModelFallbackSpec;
    /// When set, marks the current turn as resumed execution after a foreground process wakeup
    /// (rather than direct user input). `prepare_turn` uses this to mark the persisted question message
    /// as `internal_note` instead of `user`, so the wakeup prompt is not counted in
    /// `/history user`, history compression's user-turn counts, or misread by the model as
    /// the user asking again.
    pub(crate) static IS_RESUME_TURN: bool;
}

/// Read the current turn's session_id; returns an empty string outside a turn.
pub(crate) fn current_session_id_or_empty() -> String {
    TURN_IDENTITY
        .try_with(|(s, _)| s.clone())
        .unwrap_or_default()
}

/// Read the current turn's turn_id; returns 0 outside a turn.
pub(crate) fn current_turn_id_or_zero() -> usize {
    TURN_IDENTITY.try_with(|(_, t)| *t).unwrap_or(0)
}

/// Whether the current turn is resumed execution after a foreground process wakeup.
pub(crate) fn is_resume_turn() -> bool {
    IS_RESUME_TURN.try_with(|v| *v).unwrap_or(false)
}

/// Publish the parent-side payload together with the original final body into the active result slot. A top-level
/// foreground turn without an installed slot silently skips this.
pub(crate) async fn publish_subagent_result(parent_payload: &str, final_assistant_text: &str) {
    if parent_payload.trim().is_empty() {
        return;
    }
    let slot = match SUBAGENT_RESULT_SLOT.try_with(|slot| slot.clone()) {
        Ok(slot) => slot,
        Err(_) => return,
    };
    let mut guard = slot.lock().await;
    *guard = SubagentResult {
        parent_payload: parent_payload.to_string(),
        final_assistant_text: final_assistant_text.to_string(),
    };
}

pub(crate) fn has_subagent_result_slot() -> bool {
    SUBAGENT_RESULT_SLOT.try_with(|_| ()).is_ok()
}

/// Whether the current task owns terminal output. Allowed by default; background subagent scopes turn it off explicitly.
pub(crate) fn terminal_output_enabled() -> bool {
    !SUPPRESS_TERMINAL_OUTPUT
        .try_with(|value| *value)
        .unwrap_or(false)
}

/// Publish the sub-agent's current execution phase into the active phase
/// slot if one was installed by the spawning tool. Silent no-op when no
/// slot is set (top-level foreground turn, unit tests, …).
pub(crate) fn publish_subagent_phase(phase: &str) {
    let Ok(slot) = SUBAGENT_PHASE.try_with(|slot| slot.clone()) else {
        return;
    };
    let phase = compact_progress_text(phase, 80);
    let snapshot = match slot.lock() {
        Ok(mut guard) if guard.phase != phase => {
            guard.phase = phase.clone();
            guard.push_event(SubagentProgressEventKind::Phase, phase, Instant::now());
            Some(guard.snapshot(Instant::now()))
        }
        _ => None,
    };
    if let Some(snapshot) = snapshot {
        publish_progress_update(&snapshot, SubagentProgressEventKind::Phase);
    }
}

/// The most recent work plan successfully persisted. Phase changes only update `phase`, never losing this
/// task-level progress available for parent display.
pub(crate) fn publish_subagent_checkpoint_summary(summary: &str) {
    let Ok(slot) = SUBAGENT_PHASE.try_with(|slot| slot.clone()) else {
        return;
    };
    let summary = summary
        .strip_prefix("working_checkpoint:")
        .unwrap_or(summary)
        .trim();
    let summary = compact_progress_text(summary, 120);
    let snapshot = match slot.lock() {
        Ok(mut guard) => {
            let now = Instant::now();
            guard.checkpoint_summary = Some(summary.clone());
            guard.last_checkpoint_at = Some(now);
            guard.push_event(SubagentProgressEventKind::Checkpoint, summary, now);
            Some(guard.snapshot(now))
        }
        Err(_) => None,
    };
    if let Some(snapshot) = snapshot {
        publish_progress_update(&snapshot, SubagentProgressEventKind::Checkpoint);
    }
}

fn publish_progress_update(snapshot: &SubagentProgressSnapshot, kind: SubagentProgressEventKind) {
    let Ok(task_id) = SUBAGENT_TASK_ID.try_with(Clone::clone) else {
        return;
    };
    crate::ai::tools::task_tools::record_subagent_progress_update(&task_id, snapshot, kind);
}

pub(crate) fn new_subagent_progress_slot() -> SubagentPhaseSlot {
    Arc::new(std::sync::Mutex::new(SubagentProgress::new("starting")))
}

pub(crate) fn subagent_progress_snapshot(slot: &SubagentPhaseSlot) -> Option<String> {
    subagent_progress_state_snapshot(slot).map(|progress| progress.display())
}

pub(crate) fn subagent_progress_state_snapshot(
    slot: &SubagentPhaseSlot,
) -> Option<SubagentProgressSnapshot> {
    slot.lock()
        .ok()
        .map(|progress| progress.snapshot(Instant::now()))
}

/// Checkpoints for long tasks are driven by runtime state, not by the model remembering fixed turn counts.
/// At most one reminder is injected per due window; the timer restarts after a checkpoint is published successfully.
pub(crate) fn take_subagent_checkpoint_due_reminder() -> bool {
    let Ok(slot) = SUBAGENT_PHASE.try_with(|slot| slot.clone()) else {
        return false;
    };
    let Ok(mut progress) = slot.lock() else {
        return false;
    };
    let now = Instant::now();
    let checkpoint_reference = progress.last_checkpoint_at.unwrap_or(progress.started_at);
    if now.saturating_duration_since(checkpoint_reference) < SUBAGENT_CHECKPOINT_INTERVAL {
        return false;
    }
    if progress
        .last_checkpoint_reminder_at
        .is_some_and(|last| now.saturating_duration_since(last) < SUBAGENT_CHECKPOINT_INTERVAL)
    {
        return false;
    }
    progress.last_checkpoint_reminder_at = Some(now);
    true
}

fn compact_progress_text(value: &str, max_chars: usize) -> String {
    // Progress eventually reaches the parent's TTY; at the shared-slot write boundary, strip ESC, BEL, carriage returns, and other
    // terminal control characters so no foreground render path mistakes model-provided tool args for control sequences.
    let sanitized: String = value
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect();
    let normalized = sanitized.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = normalized.chars();
    let prefix: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{prefix}…")
    } else {
        prefix
    }
}

pub(crate) fn format_progress_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds < 60 {
        format!("{seconds}s")
    } else if seconds < 3_600 {
        format!("{}m{:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h{:02}m", seconds / 3_600, (seconds % 3_600) / 60)
    }
}

/// Consume one wrap-up request from the parent. If this task-local is not installed, the current turn has no
/// pre-timeout wrap-up policy and keeps the normal execution path.
pub(crate) fn take_subagent_wrap_up_request() -> bool {
    SUBAGENT_WRAP_UP_SIGNAL
        .try_with(|signal| signal.swap(false, Ordering::AcqRel))
        .unwrap_or(false)
}

/// Non-consuming check: whether the parent has already sent a pre-timeout wrap-up request. Used to interrupt the in-flight
/// model request and immediately enter the forced wrap-up iteration (see the request-interrupt branch in `iteration.rs`).
pub(crate) fn has_subagent_wrap_up_pending() -> bool {
    SUBAGENT_WRAP_UP_SIGNAL
        .try_with(|signal| signal.load(Ordering::Acquire))
        .unwrap_or(false)
}

// =============================================================================
// DRIVER_CTX fallback for parallel read-only batch threads
// =============================================================================
// `run_parallel_readonly_batch` runs read-only tools on raw OS threads via `std::thread::scope`.
// The tokio task-local `DRIVER_CTX` is visible only to the tokio task that installed it; on those
// threads `try_with` always fails, hard-failing tools that depend on session context (such as search_overflow's
// `current_session_assets_dir`) during batched parallelism. Before running, a batch thread installs the parent task's
// context into the fallback slot below, and `try_current()` reads it when the task-local is missing;
// on thread exit (guard drop) the original value is restored and nothing leaks to other threads.
thread_local! {
    static THREAD_CTX_FALLBACK: RefCell<Option<Arc<DriverContext>>> =
        const { RefCell::new(None) };
}

/// RAII guard: temporarily installs/restores the `DRIVER_CTX` fallback inside a `std::thread::scope` batch thread,
/// letting read-only tools still resolve session context on those threads.
pub(crate) struct DriverCtxThreadFallback {
    previous: Option<Arc<DriverContext>>,
}

impl DriverCtxThreadFallback {
    /// Install the fallback context; the returned guard restores the thread's previous value on drop.
    pub(crate) fn install(ctx: Option<Arc<DriverContext>>) -> Self {
        let previous =
            THREAD_CTX_FALLBACK.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), ctx));
        Self { previous }
    }
}

impl Drop for DriverCtxThreadFallback {
    fn drop(&mut self) {
        THREAD_CTX_FALLBACK.with(|slot| {
            *slot.borrow_mut() = self.previous.take();
        });
    }
}

/// Try to read the current `DRIVER_CTX`. Returns `None` when called from a
/// thread that has no active scope (e.g. unit tests or one-shot tool
/// invocations outside a turn). Falls back to the thread-installed
/// [`DriverCtxThreadFallback`] value on raw threads (parallel readonly batch).
pub(crate) fn try_current() -> Option<Arc<DriverContext>> {
    DRIVER_CTX
        .try_with(Arc::clone)
        .ok()
        .or_else(|| THREAD_CTX_FALLBACK.with(|slot| slot.borrow().clone()))
}

pub(crate) fn auto_model_fallback_spec() -> Option<AutoModelFallbackSpec> {
    AUTO_MODEL_FALLBACK.try_with(|value| *value).ok()
}

/// Read the current sub-agent nesting depth. `0` means top-level agent
/// (no `SUBAGENT_DEPTH` task-local set). Each spawn level increments by 1.
pub(crate) fn current_subagent_depth() -> usize {
    SUBAGENT_DEPTH.try_with(|d| *d).unwrap_or(0)
}

/// Read the current sub-agent task id when inside a sub-agent dispatch.
/// Returns None on foreground turns. Exposed for `side_note::current_target_id`
/// so file-queue routing can distinguish foreground vs per-task queues.
pub(crate) fn try_subagent_task_id() -> Option<String> {
    SUBAGENT_TASK_ID.try_with(|id| id.clone()).ok()
}

/// Read the optional sub-agent memory path override. `None` means
/// "fall back to persona memory file / shared memory file".
pub(crate) fn override_memory_path() -> Option<PathBuf> {
    SUBAGENT_MEMORY_PATH
        .try_with(|p| p.clone())
        .ok()
        .or_else(|| PERSONA_MEMORY_PATH.try_with(|p| p.clone()).ok())
}

/// Resolve the effective working directory for tools that consult the
/// process cwd. Honours `SUBAGENT_CWD` first, then falls back to
/// `std::env::current_dir()`.
pub(crate) fn effective_cwd() -> std::io::Result<PathBuf> {
    if let Ok(p) = SUBAGENT_CWD.try_with(|p| p.clone()) {
        return Ok(p);
    }
    std::env::current_dir()
}

// =============================================================================
// Per-session temp directory + persistent temp-file registry
// =============================================================================
// Agents often need to write temp/intermediate files while working (scripts, snippet output, dumps, etc.).
// `temp_dir()` provides a unified, per-session isolated temp directory, created on demand.
//
// Prefer the session assets dir (same source as tool-overflow) at
// `~/.history_file.sessions/<session>.assets/tmp/` — outside the project and isolated per session,
// so the workspace stays clean. When DRIVER_CTX is unavailable (tests / one-shot calls), fall back to
// `<std::env::temp_dir()/.agent_tmp/<session>/` (system temp dir; does not pollute the project).
//
// Files written into this dir via `write_file(temp=true)` are recorded in a persistent registry
// (`storage::temp_registry`) for audit tracking. Temp files are cleaned up by the runtime when the session ends.
// The registry is persisted as a JSON file and remains readable after restarts following a session termination.
// =============================================================================

/// Return the current session's temp dir path, creating the directory on demand. Used by `write_file(temp=true)`
/// and other flows that need to write temp files.
///
/// Prefers `<sessions_root>/<session>.assets/tmp/` (same source as tool-overflow,
/// outside the project); falls back to `<std::env::temp_dir()/.agent_tmp/<session>/` when `DRIVER_CTX` is unavailable.
pub(crate) fn temp_dir() -> std::io::Result<PathBuf> {
    // Prefer the session assets dir (same source as tool-overflow) so temp files live in
    // the per-session isolated ~/.history_file.sessions/<id>.assets/tmp/ outside the project.
    if let Some(ctx) = try_current() {
        let history_file = ctx.app_proto.config.history_file.clone();
        let session_id = ctx.app_proto.session_id.clone();
        let store = crate::ai::history::SessionStore::new(&history_file);
        store.ensure_root_dir()?;
        let dir = store.session_assets_dir(&session_id).join("tmp");
        std::fs::create_dir_all(&dir)?;
        return Ok(dir);
    }

    // Fallback: without DRIVER_CTX (tests / one-shot calls), use the system temp dir,
    // never under effective_cwd, keeping the project workspace clean.
    let base = std::env::temp_dir();
    let session = current_session_id_or_empty();
    let session_part = if session.is_empty() {
        "default".to_string()
    } else {
        session
    };
    let dir = base.join(".agent_tmp").join(session_part);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Deterministic scratch-workspace path for a sub-agent that opted out of
/// `inherit.cwd`. Single source of truth for the path rule so that the
/// creation site ([`make_subagent_cwd`]) and the cleanup site (the history
/// guard) can never drift out of sync.
pub(crate) fn subagent_cwd_path(base: &Path, task_id: &str) -> PathBuf {
    base.join(format!("subagent-cwd-{task_id}"))
}

/// Build a default scratch workspace path for a sub-agent that opted out
/// of `inherit.cwd`. The directory is created on demand. Returns `None`
/// if the directory cannot be created (caller should fall back to
/// inheriting cwd in that case).
pub(crate) fn make_subagent_cwd(base: &Path, task_id: &str) -> Option<PathBuf> {
    let dir = subagent_cwd_path(base, task_id);
    std::fs::create_dir_all(&dir).ok().map(|_| dir)
}

/// Build the per-subagent memory file path next to the parent's history
/// file. Used by `inherit.memory == false`.
pub(crate) fn make_subagent_memory_path(base_history: &Path, task_id: &str) -> PathBuf {
    let parent = base_history.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!("agent_memory.subagent-{task_id}.jsonl"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subagent_wrap_up_request_is_consumed_once() {
        let signal = Arc::new(AtomicBool::new(true));
        let (first, second) = SUBAGENT_WRAP_UP_SIGNAL.sync_scope(signal, || {
            (
                take_subagent_wrap_up_request(),
                take_subagent_wrap_up_request(),
            )
        });

        assert!(first);
        assert!(!second);
    }

    #[test]
    fn override_memory_path_is_none_outside_scope() {
        assert!(override_memory_path().is_none());
    }

    #[test]
    fn terminal_output_is_disabled_only_inside_suppressed_scope() {
        assert!(terminal_output_enabled());
        let enabled = SUPPRESS_TERMINAL_OUTPUT.sync_scope(true, terminal_output_enabled);
        assert!(!enabled);
        assert!(terminal_output_enabled());
    }

    #[test]
    fn subagent_progress_keeps_saved_plan_while_phase_changes() {
        let slot = new_subagent_progress_slot();
        let rendered = SUBAGENT_PHASE.sync_scope(slot.clone(), || {
            publish_subagent_checkpoint_summary("working_checkpoint: inspect the task path");
            publish_subagent_phase("calling model");
            subagent_progress_snapshot(&slot)
        });

        let rendered = rendered.expect("progress snapshot");
        assert!(rendered.starts_with("calling model · plan: inspect the task path"));
        assert!(rendered.contains("elapsed "));
        assert!(rendered.contains("last activity "));
    }

    #[test]
    fn subagent_progress_timeline_is_bounded_and_keeps_latest_events() {
        let mut progress = SubagentProgress::new("starting");
        let start = progress.started_at;
        for index in 0..20 {
            progress.push_event(
                SubagentProgressEventKind::Phase,
                format!("phase-{index}"),
                start + Duration::from_secs(index + 1),
            );
        }

        let snapshot = progress.snapshot(start + Duration::from_secs(21));
        assert_eq!(snapshot.timeline.len(), SUBAGENT_PROGRESS_TIMELINE_CAPACITY);
        assert_eq!(snapshot.timeline.first().unwrap().summary, "phase-8");
        assert_eq!(snapshot.timeline.last().unwrap().summary, "phase-19");
    }

    #[tokio::test]
    async fn published_subagent_result_keeps_parent_payload_and_raw_final_separate() {
        let slot: SubagentResultSlot = Arc::new(Mutex::new(SubagentResult::default()));
        SUBAGENT_RESULT_SLOT
            .scope(slot.clone(), async {
                publish_subagent_result(
                    "[Subagent tool evidence]\nread_file(...)\n\n[Subagent final answer]\n{\"ok\":true}",
                    "{\"ok\":true}",
                )
                .await;
            })
            .await;

        let result = slot.lock().await.clone();
        assert!(
            result
                .parent_payload
                .starts_with("[Subagent tool evidence]")
        );
        assert_eq!(result.final_assistant_text, "{\"ok\":true}");
    }

    #[test]
    fn override_memory_path_returns_value_inside_scope() {
        let want = PathBuf::from("/tmp/agent_memory.subagent-test.jsonl");
        let got = SUBAGENT_MEMORY_PATH.sync_scope(want.clone(), || override_memory_path());
        assert_eq!(got, Some(want));
    }

    #[test]
    fn override_memory_path_falls_back_to_persona_scope() {
        let want = PathBuf::from("/tmp/agent_memory.persona-test.jsonl");
        let got = PERSONA_MEMORY_PATH.sync_scope(want.clone(), || override_memory_path());
        assert_eq!(got, Some(want));
    }

    #[test]
    fn effective_cwd_falls_back_to_process_cwd() {
        let process_cwd = std::env::current_dir().unwrap();
        let got = effective_cwd().unwrap();
        assert_eq!(got, process_cwd);
    }

    #[test]
    fn effective_cwd_honours_subagent_override() {
        let want = std::env::temp_dir();
        let got = SUBAGENT_CWD.sync_scope(want.clone(), || effective_cwd().unwrap());
        assert_eq!(got, want);
    }

    #[test]
    fn make_subagent_memory_path_lands_next_to_parent_history() {
        let parent = PathBuf::from("/tmp/sessions/session-foo.jsonl");
        let got = make_subagent_memory_path(&parent, "abc123");
        assert_eq!(
            got,
            PathBuf::from("/tmp/sessions/agent_memory.subagent-abc123.jsonl")
        );
    }

    #[test]
    fn make_subagent_memory_path_handles_root_history() {
        let parent = PathBuf::from("session.jsonl");
        let got = make_subagent_memory_path(&parent, "abc");
        assert_eq!(got, PathBuf::from("agent_memory.subagent-abc.jsonl"));
    }

    #[test]
    fn make_subagent_cwd_creates_scoped_directory() {
        let base = std::env::temp_dir().join(format!(
            "rust_tools_runtime_ctx_test_{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&base).unwrap();
        let got = make_subagent_cwd(&base, "tid").unwrap();
        assert!(got.is_dir());
        assert!(got.starts_with(&base));
        assert!(got.ends_with("subagent-cwd-tid"));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn is_resume_turn_defaults_false_outside_scope() {
        assert!(!is_resume_turn());
    }

    #[test]
    fn is_resume_turn_true_inside_scope() {
        let got = IS_RESUME_TURN.sync_scope(true, || is_resume_turn());
        assert!(got);
    }

    #[test]
    fn try_current_falls_back_to_thread_installed_ctx_on_raw_threads() {
        // Regression: parallel read-only batches run on raw OS threads via `std::thread::scope`, where the tokio
        // task-local `DRIVER_CTX` does not exist (batched parallel search_overflow once
        // hard-failed because of this). After a batch thread installs the `DriverCtxThreadFallback`,
        // `try_current()` must still resolve the context; the fallback is restored automatically on thread exit,
        // and nothing leaks between threads.
        let ctx = DriverContext::new(
            crate::ai::driver::tests::test_app("build"),
            Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new())),
            Arc::new(Vec::new()),
            Arc::new(Vec::new()),
        );
        assert!(try_current().is_none(), "测试线程默认无 DRIVER_CTX");

        std::thread::scope(|scope| {
            scope.spawn({
                let ctx = Arc::clone(&ctx);
                move || {
                    assert!(try_current().is_none(), "原始线程上默认无 DRIVER_CTX");
                    let guard = DriverCtxThreadFallback::install(Some(ctx));
                    assert!(
                        try_current().is_some(),
                        "安装回退后 try_current() 必须能解析上下文"
                    );
                    drop(guard);
                    assert!(try_current().is_none(), "guard drop 后恢复无上下文");
                }
            });
            scope.spawn(move || {
                // Concurrent threads do not affect each other: the fallback is thread-local.
                assert!(try_current().is_none());
            });
        });

        assert!(try_current().is_none(), "线程退出后回退不泄漏到测试线程");
    }
}
