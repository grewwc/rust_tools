//! Destroy subagent tasks when their owning foreground process dies unexpectedly.

use super::*;
use rust_tools::commonw::FastSet;

/// Registry data captured before the kernel is locked. Task tools take the registry lock before
/// the kernel lock, so both are never held at the same time here.
struct DeadTask {
    task_id: String,
    owner_pid: u64,
    pid: u64,
    result_channel_id: u64,
    completion_futex_addr: FutexAddr,
    abort_handle: Option<tokio::task::AbortHandle>,
    cancel_stream: Arc<AtomicBool>,
    backend: BackendId,
}

/// Terminate every pending subagent task that belongs to the foreground process `root_pid` or one
/// of its kernel descendants.
///
/// A foreground that dies unexpectedly must not leave subagents running: they would keep consuming
/// model/tool resources, mutate the workspace, and a replacement foreground could adopt them and
/// continue work the user never asked for. Finished tasks keep their registry entries so a
/// replacement foreground can still collect results they already produced; only pending tasks are
/// destroyed. Destruction aborts the hosting Tokio task, signals blocking command runners to kill
/// their OS process groups, terminates the kernel process subtree, and removes the registry, IPC,
/// retry, and wait state so nothing can be adopted, retried, or collected afterwards.
///
/// The caller must not hold the kernel lock. Returns the number of destroyed tasks.
pub(crate) fn destroy_pending_tasks_of_dead_foreground(
    kernel: &SharedKernel,
    session_id: &str,
    root_pid: u64,
    reason: &str,
) -> usize {
    let snapshot = {
        let registry = lock_task_registry();
        registry
            .iter()
            .filter(|(_, entry)| entry.session_id == session_id)
            .map(|(task_id, entry)| DeadTask {
                task_id: task_id.clone(),
                owner_pid: entry.owner_pid,
                pid: entry.pid,
                result_channel_id: entry.result_channel_id,
                completion_futex_addr: entry.completion_futex_addr,
                abort_handle: entry.abort_handle.clone(),
                cancel_stream: entry.cancel_stream.clone(),
                backend: entry.backend.unwrap_or_default(),
            })
            .collect::<Vec<_>>()
    };
    if snapshot.is_empty() {
        return 0;
    }

    let processes = {
        let os = kernel.lock().unwrap_or_else(|poison| poison.into_inner());
        os.list_processes()
    };
    let mut dead = FastSet::default();
    collect_subtree(&processes, root_pid, &mut dead);
    let victims = snapshot
        .into_iter()
        .filter(|task| dead.contains(&task.owner_pid) || dead.contains(&task.pid))
        .collect::<Vec<_>>();
    if victims.is_empty() {
        return 0;
    }

    // Stop the hosting futures and blocking runners before the kernel teardown, mirroring
    // `task_cancel`: aborting the Tokio task alone does not stop a synchronous backend, while the
    // cancel flag makes its command runner kill the OS process group.
    for task in &victims {
        task.cancel_stream.store(true, Ordering::Release);
        if let Some(handle) = &task.abort_handle {
            handle.abort();
        }
    }

    let destroyed = {
        let mut os = kernel.lock().unwrap_or_else(|poison| poison.into_inner());
        // Sample pending state before termination flips every process to `Terminated`.
        let pending = victims
            .iter()
            .filter(|task| is_task_pending(os.as_mut(), task.pid).unwrap_or(false))
            .collect::<Vec<_>>();
        if pending.is_empty() {
            return 0;
        }

        let mut kill_set = FastSet::default();
        for task in &pending {
            collect_subtree(&processes, task.pid, &mut kill_set);
        }
        for pid in kill_set.iter().copied() {
            if pid != root_pid
                && os
                    .get_process(pid)
                    .is_some_and(|process| process.state != ProcessState::Terminated)
            {
                // The kernel exposes subtree termination only as `kill_process`, which is
                // capability-scoped to the caller. Teardown of a dead owner is system work, so
                // terminate each process through the current-process slot instead.
                os.set_current_pid(Some(pid));
                os.terminate_current(format!("Aborted with its foreground: {reason}"));
            }
        }
        for task in &pending {
            let _ = os.cleanup_process_resources(task.pid);
            let _ = os.drop_terminated(task.pid);
            let channel_id = ChannelId(task.result_channel_id);
            let _ = os.channel_close(None, channel_id);
            let _ = os.channel_release_named(channel_id, "task_result.consumer");
            let _ = os.channel_release_named(channel_id, "task_result.producer");
            let _ = os.channel_destroy(None, channel_id);
            let _ = os.futex_destroy(task.completion_futex_addr);
        }
        // Drop the remaining tombstone entries so none of these processes can be re-parented to
        // the next foreground and become reachable again.
        for pid in kill_set.iter().copied() {
            let _ = os.drop_terminated(pid);
        }

        pending
            .into_iter()
            .map(|task| (task.task_id.clone(), task.backend))
            .collect::<Vec<_>>()
    };

    for (task_id, backend) in &destroyed {
        remove_task_entry(task_id);
        backend_session::forget_session(*backend, task_id);
        lock_unpoisoned(&TASK_PROGRESS_REGISTRY).remove(task_id);
        lock_unpoisoned(&TASK_RETRY_REGISTRY).remove(task_id);
    }
    TASK_WAIT_STATES
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .retain(|key, _| key.session_id != session_id || !dead.contains(&key.owner_pid));
    TASK_WAIT_NOOP_COUNTS
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .retain(|key, _| key.session_id != session_id || !dead.contains(&key.owner_pid));

    destroyed.len()
}

/// Collect `root` and every descendant reachable through `parent_pid`.
fn collect_subtree(processes: &[aios_kernel::kernel::Process], root: u64, out: &mut FastSet<u64>) {
    out.insert(root);
    // Repeat until no further process is reachable so the result does not depend on the order of
    // `list_processes` (a grandchild may be listed before its parent).
    loop {
        let mut changed = false;
        for process in processes {
            if let Some(parent) = process.parent_pid
                && out.contains(&parent)
                && out.insert(process.pid)
            {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}
