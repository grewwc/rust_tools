use super::*;

/// Registry of asynchronous subtasks, keyed by task_id (a UUID string); values are
/// [`AsyncTaskEntry`].
///
/// This is **parallel storage** to the AIOS kernel process table: the two are linked through the
/// `pid` field but each keeps its own independent set of fields (see the `AsyncTaskEntry`
/// comments). Accessors should read/write this registry through helpers such as `with_task_entry`
/// / `take_task_entry` rather than holding a lock guard directly.
pub(super) static TASK_REGISTRY: LazyLock<Mutex<SkipMap<String, AsyncTaskEntry>>> =
    LazyLock::new(|| Mutex::new(SkipMap::default()));
static TASK_REASONING_EFFORT_REGISTRY: LazyLock<Mutex<SkipMap<String, String>>> =
    LazyLock::new(|| Mutex::new(SkipMap::default()));
pub(super) static TASK_RETRY_REGISTRY: LazyLock<Mutex<SkipMap<String, RetryableTaskSpec>>> =
    LazyLock::new(|| Mutex::new(SkipMap::default()));

/// Lock a task-tool mutex without propagating poisoning. A panic while one task tool held a lock
/// must not permanently disable every later `task_*` call in the session.
pub(super) fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poison| poison.into_inner())
}

pub(super) fn lock_task_registry() -> std::sync::MutexGuard<'static, SkipMap<String, AsyncTaskEntry>>
{
    lock_unpoisoned(&TASK_REGISTRY)
}

#[derive(Clone)]
pub(super) struct RetryableTaskSpec {
    pub(super) session_id: String,
    pub(super) owner_pid: u64,
    pub(super) prepared: PreparedSubagentTask,
    pub(super) retry_root: String,
    pub(super) terminal_status: Option<String>,
    pub(super) recorded_at: Instant,
}
pub(super) fn register_retry_spec(
    task_id: &str,
    session_id: String,
    owner_pid: u64,
    prepared: PreparedSubagentTask,
    retry_root: String,
) {
    let mut registry = lock_unpoisoned(&TASK_RETRY_REGISTRY);
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

pub(super) fn mark_task_retry_status(task_id: &str, status: &str) {
    if let Some(spec) = lock_unpoisoned(&TASK_RETRY_REGISTRY).get_mut(&task_id.to_string()) {
        spec.terminal_status = Some(status.to_string());
    }
}

pub(super) fn is_retryable_task_status(status: &str) -> bool {
    TaskTerminalStatus::parse(status).is_some_and(TaskTerminalStatus::is_retryable)
}
/// Look up a registered async task entry. Used by the driver-side sync `task`
/// interception to retrieve the channel/futex/inherit info after spawning.
pub(crate) fn with_task_entry<R>(task_id: &str, f: impl FnOnce(&AsyncTaskEntry) -> R) -> Option<R> {
    let registry = lock_task_registry();
    registry.get_ref(&task_id.to_string()).map(f)
}

/// Associates the Tokio task that actually runs the subagent, so cancellation and timeout can stop
/// the background Future instead of only terminating the logical process in the kernel.
pub(crate) fn set_task_abort_handle(task_id: &str, abort_handle: tokio::task::AbortHandle) -> bool {
    let mut registry = lock_task_registry();
    let Some(entry) = registry.get_mut(&task_id.to_string()) else {
        return false;
    };
    entry.abort_handle = Some(abort_handle);
    true
}

/// Record a subagent's effective effort display label. Kernel children resolve
/// it at preparation; external backends record their dispatched effort. `None`
/// leaves an external backend's default unknown.
pub(crate) fn set_task_reasoning_effort(task_id: &str, reasoning_effort: Option<String>) {
    let mut registry = lock_unpoisoned(&TASK_REASONING_EFFORT_REGISTRY);
    match reasoning_effort.filter(|effort| !effort.trim().is_empty()) {
        Some(effort) => {
            registry.insert(task_id.to_string(), effort);
        }
        None => {
            registry.remove(&task_id.to_string());
        }
    }
}

pub(crate) fn task_reasoning_effort(task_id: &str) -> Option<String> {
    lock_unpoisoned(&TASK_REASONING_EFFORT_REGISTRY).get(&task_id.to_string())
}

pub(crate) fn with_task_entry_by_pid<R>(
    pid: u64,
    mut f: impl FnMut(&AsyncTaskEntry) -> R,
) -> Option<R> {
    let registry = lock_task_registry();
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
    pub(crate) task_id: String,
    pub(crate) started_at: Instant,
    pub(crate) description: String,
    pub(crate) agent_name: String,
    pub(crate) backend: Option<String>,
    /// Model actually serving an external backend. The kernel runner does not
    /// disclose a model in the status block.
    pub(crate) model: Option<String>,
    /// Effective effort display label. `None` preserves an external backend's
    /// unknown/default behavior.
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) state: String,
    pub(crate) elapsed_secs: u64,
    pub(crate) progress: Option<String>,
    /// Sampled once with the text above so the shared tree can use typed activity.
    pub(crate) progress_snapshot: Option<crate::ai::driver::runtime_ctx::SubagentProgressSnapshot>,
}

/// Registry-only half of the status-line snapshot. Kept separate from the kernel query so a caller
/// that already holds the kernel lock cannot invert the registry -> kernel order used elsewhere.
pub(crate) struct SubagentStatusSnapshot {
    task_id: String,
    owner_pid: u64,
    pid: u64,
    result_channel_id: u64,
    description: String,
    agent_name: String,
    backend: Option<String>,
    display_model: Option<String>,
    reasoning_effort: Option<String>,
    started_at: Instant,
    progress: Option<String>,
    progress_snapshot: Option<crate::ai::driver::runtime_ctx::SubagentProgressSnapshot>,
}

/// Snapshot the registry under its own lock, then resolve progress after releasing it.
pub(crate) fn subagent_status_snapshots(session_id: &str) -> Vec<SubagentStatusSnapshot> {
    let mut snapshots = {
        let registry = lock_task_registry();
        registry
            .iter()
            .filter(|(_, entry)| entry.session_id == session_id)
            .map(|(task_id, entry)| SubagentStatusSnapshot {
                task_id: task_id.clone(),
                owner_pid: entry.owner_pid,
                pid: entry.pid,
                result_channel_id: entry.result_channel_id,
                description: entry.description.clone(),
                agent_name: entry.agent_name.clone(),
                backend: entry.backend.map(|backend| backend.as_str().to_string()),
                // Only external backends disclose their model in the status block.
                display_model: entry.backend.map(|_| entry.model.clone()),
                reasoning_effort: None,
                started_at: entry.started_at,
                progress: None,
                progress_snapshot: None,
            })
            .collect::<Vec<_>>()
    };
    {
        let effort_registry = lock_unpoisoned(&TASK_REASONING_EFFORT_REGISTRY);
        for snapshot in &mut snapshots {
            snapshot.reasoning_effort = effort_registry.get(&snapshot.task_id);
        }
    }
    for snapshot in &mut snapshots {
        snapshot.progress_snapshot = task_progress_slot(&snapshot.task_id).and_then(|slot| {
            crate::ai::driver::runtime_ctx::subagent_progress_state_snapshot(&slot)
        });
        let progress = snapshot
            .progress_snapshot
            .as_ref()
            .map(crate::ai::driver::runtime_ctx::SubagentProgressSnapshot::display_status_line)
            .filter(|value| !value.is_empty());
        snapshot.progress = match (progress, pending_task_side_note_progress(&snapshot.task_id)) {
            (Some(progress), Some(pending)) => Some(format!("{progress} · {pending}")),
            (progress, pending) => progress.or(pending),
        };
    }
    snapshots
}

pub(crate) fn subagent_terminal_statuses(
    os: &mut dyn Kernel,
    snapshots: Vec<SubagentStatusSnapshot>,
) -> Vec<SubagentTerminalStatus> {
    let Some(owner_pid) = active_foreground_owner_pid(os) else {
        return Vec::new();
    };
    let mut statuses = snapshots
        .into_iter()
        .filter(|snapshot| snapshot.owner_pid == owner_pid)
        .map(|snapshot| SubagentTerminalStatus {
            task_id: snapshot.task_id,
            started_at: snapshot.started_at,
            description: snapshot.description,
            agent_name: snapshot.agent_name,
            backend: snapshot.backend,
            model: snapshot.display_model,
            reasoning_effort: snapshot.reasoning_effort,
            state: task_state_string(os, snapshot.result_channel_id, snapshot.pid)
                .unwrap_or_else(|_| "unknown".to_string()),
            elapsed_secs: snapshot.started_at.elapsed().as_secs(),
            progress: snapshot.progress,
            progress_snapshot: snapshot.progress_snapshot,
        })
        .collect::<Vec<_>>();
    statuses.sort_by(|a, b| a.description.cmp(&b.description));
    statuses
}

/// Validate a `send_side_note` target before queueing the note.
/// Foreground targets need no validation and always pass. A concrete task id
/// must name an active subagent owned by the current session/process, and the
/// subagent must not have published its final result yet.
pub(crate) fn validate_side_note_target(task_id: &str) -> Result<(), String> {
    let trimmed = task_id.trim();
    if trimmed.is_empty() || trimmed == "foreground" {
        return Ok(());
    }
    let session_id = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
    recover_current_foreground_tasks(&session_id)?;
    let owner_pid_opt = current_task_owner_pid_opt();
    let entry = with_task_entry(trimmed, |entry| entry.clone()).ok_or_else(|| {
        format!(
            "Unknown task_id `{trimmed}`: no active subagent with this id in the current session."
        )
    })?;
    if entry.session_id != session_id {
        return Err(format!(
            "Refusing side-note to task_id `{trimmed}` owned by another session."
        ));
    }
    if let Some(owner_pid) = owner_pid_opt
        && entry.owner_pid != owner_pid
    {
        return Err(format!(
            "Refusing side-note to task_id `{trimmed}` not owned by current process pid={owner_pid}."
        ));
    }
    let finished_status = match with_os_kernel(|os| {
        if let Some(result) = read_task_result(os, entry.result_channel_id, false)? {
            return Ok(Some(result.status));
        }
        if is_task_pending(os, entry.pid)? {
            Ok(None)
        } else {
            Ok(Some(task_state_string(
                os,
                entry.result_channel_id,
                entry.pid,
            )?))
        }
    }) {
        Ok(status) => status,
        Err(_) => return Ok(()),
    };
    if let Some(status) = finished_status {
        if task_backend(entry.backend).descriptor().side_notes == SideNoteMode::DeferredContinuation
        {
            return Err(format!(
                "Task `{trimmed}` already finished (status={status}); put new guidance in a follow-up `task`/`task_spawn` with `backend_continue_from`. Previously queued side-notes are retained for that continuation."
            ));
        }
        return Err(format!(
            "Task `{trimmed}` already finished (status={status}); a side-note would never be seen. Spawn a follow-up task or use `task_retry` for a new linked attempt instead."
        ));
    }
    Ok(())
}

/// Resolve which history file owns the queue a side-note must be written to.
///
/// `Ok(None)` keeps the note in `foreground_history`: the foreground turn and
/// empty targets drain that queue. A concrete task id resolves to the child's
/// own history file (`<stem>.proc-<pid>.<ext>`), because a background subagent
/// drains the queue under its own history file; writing to the lead's file
/// would leave the note unread while still reporting success.
///
/// Deferred queues stay under the parent's history and are drained when
/// `backend_continue_from` dispatches a continuation, not during the active run.
pub(crate) fn resolve_side_note_history_file(
    task_id: &str,
    foreground_history: &std::path::Path,
) -> Result<Option<std::path::PathBuf>, String> {
    let trimmed = task_id.trim();
    if trimmed.is_empty() || trimmed == "foreground" {
        return Ok(None);
    }
    validate_side_note_target(trimmed)?;
    let (pid, backend) = with_task_entry(trimmed, |entry| (entry.pid, entry.backend))
        .ok_or_else(|| {
            format!(
                "Unknown task_id `{trimmed}`: no active subagent with this id in the current session."
            )
        })?;
    if task_backend(backend).descriptor().side_notes == SideNoteMode::DeferredContinuation {
        return Ok(Some(foreground_history.to_path_buf()));
    }
    Ok(Some(crate::ai::driver::process_history_path(
        foreground_history,
        pid,
    )))
}

pub(crate) fn task_backend_id(task_id: &str) -> BackendId {
    with_task_entry(task_id, |entry| entry.backend.unwrap_or_default().id())
        .unwrap_or(BackendId::KERNEL)
}

/// Remove a task entry from the registry. Called by the synchronous `task`
/// interception once it has consumed the result.
pub(crate) fn remove_task_entry(task_id: &str) -> Option<AsyncTaskEntry> {
    lock_unpoisoned(&TASK_REASONING_EFFORT_REGISTRY).remove(&task_id.to_string());
    let mut registry = lock_task_registry();
    registry.take(&task_id.to_string())
}

#[cfg(test)]
pub(crate) fn insert_task_entry_for_test(task_id: String, entry: AsyncTaskEntry) {
    let mut registry = lock_task_registry();
    registry.insert(task_id, entry);
}
/// channel. Re-exported for the synchronous `task` interception so that both
/// paths produce identical output.
pub(crate) fn format_finished_task(entry: &AsyncTaskEntry, result: StoredTaskResult) -> String {
    format_task_result(entry, result, None)
}
