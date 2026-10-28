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
pub(super) static TASK_RETRY_REGISTRY: LazyLock<Mutex<SkipMap<String, RetryableTaskSpec>>> =
    LazyLock::new(|| Mutex::new(SkipMap::default()));

/// Lock a task-tool mutex without propagating poisoning. A panic while one task tool held a lock
/// must not permanently disable every later `task_*` call in the session.
pub(super) fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poison| poison.into_inner())
}

pub(super) fn lock_task_registry()
-> std::sync::MutexGuard<'static, SkipMap<String, AsyncTaskEntry>> {
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
    pub(crate) description: String,
    pub(crate) agent_name: String,
    pub(crate) state: String,
    pub(crate) elapsed_secs: u64,
    pub(crate) progress: Option<String>,
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
    started_at: Instant,
    progress: Option<String>,
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
                started_at: entry.started_at,
                progress: None,
            })
            .collect::<Vec<_>>()
    };
    for snapshot in &mut snapshots {
        snapshot.progress = current_task_progress(&snapshot.task_id);
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
            description: snapshot.description,
            agent_name: snapshot.agent_name,
            state: task_state_string(os, snapshot.result_channel_id, snapshot.pid)
                .unwrap_or_else(|_| "unknown".to_string()),
            elapsed_secs: snapshot.started_at.elapsed().as_secs(),
            progress: snapshot.progress,
        })
        .collect::<Vec<_>>();
    statuses.sort_by(|a, b| a.description.cmp(&b.description));
    statuses
}

/// Remove a task entry from the registry. Called by the synchronous `task`
/// interception once it has consumed the result.
pub(crate) fn remove_task_entry(task_id: &str) -> Option<AsyncTaskEntry> {
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
    format_task_result(entry, result)
}
