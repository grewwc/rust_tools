//! Foreground replacement keeps task ownership and kernel resources in agreement.

use rust_tools::commonw::{FastMap, FastSet};

use super::*;

pub(crate) fn recover_current_foreground_tasks(session_id: &str) -> Result<Vec<String>, String> {
    if session_id.is_empty() || crate::ai::driver::runtime_ctx::current_subagent_depth() != 0 {
        return Ok(Vec::new());
    }
    let Some(owner_pid) = with_os_kernel(|os| Ok(os.current_process_id()))? else {
        return Ok(Vec::new());
    };
    recover_orphaned_task_owners(session_id, owner_pid)
}

/// Reconcile only this session's tasks, and only after the previous owner exits. Registry
/// snapshots, kernel mutations, and registry commits use separate locks to avoid inversion.
pub(super) fn recover_orphaned_task_owners(
    session_id: &str,
    owner_pid: u64,
) -> Result<Vec<String>, String> {
    if session_id.is_empty() || crate::ai::driver::runtime_ctx::current_subagent_depth() != 0 {
        return Ok(Vec::new());
    }
    let (candidates, task_owners) = {
        let registry = lock_task_registry();
        let entries = registry
            .iter()
            .filter(|(_, entry)| entry.session_id == session_id);
        let mut candidates = Vec::new();
        let mut task_owners = FastMap::default();
        for (task_id, entry) in entries {
            task_owners.insert(task_id.clone(), entry.owner_pid);
            if entry.owner_pid != owner_pid {
                candidates.push((task_id.clone(), entry.clone()));
            }
        }
        (candidates, task_owners)
    };
    let retry_candidates = {
        let registry = lock_unpoisoned(&TASK_RETRY_REGISTRY);
        registry
            .iter()
            .filter(|(_, spec)| spec.session_id == session_id && spec.owner_pid != owner_pid)
            .map(|(task_id, spec)| {
                (
                    task_id.clone(),
                    spec.owner_pid,
                    spec.terminal_status.is_some(),
                )
            })
            .collect::<Vec<_>>()
    };
    if candidates.is_empty() && retry_candidates.is_empty() {
        return Ok(Vec::new());
    }

    let (adopted, retry_ids, dead_owners, failures) = with_os_kernel(|os| {
        let mut adopted = Vec::new();
        let mut retry_ids = Vec::new();
        let mut dead_owners = FastSet::default();
        let mut failures = Vec::new();
        if !os.get_process(owner_pid).is_some_and(|process| {
            process.is_foreground && process.state != ProcessState::Terminated
        }) {
            return Ok((adopted, retry_ids, dead_owners, failures));
        }
        for (task_id, entry) in &candidates {
            if os
                .get_process(entry.owner_pid)
                .is_some_and(|process| process.state != ProcessState::Terminated)
            {
                continue;
            }
            dead_owners.insert(entry.owner_pid);
            // An absent channel is recovered from the durable terminal publication during
            // collection. Existing channels still undergo the kernel's owner validation.
            let channels = os
                .channel_meta(ChannelId(entry.result_channel_id))
                .map(|metadata| vec![metadata.channel])
                .unwrap_or_default();
            match os.adopt_process_resources(entry.pid, entry.owner_pid, owner_pid, &channels) {
                Ok(()) => adopted.push(task_id.clone()),
                Err(error) => failures.push(format!("{task_id}: {error}")),
            }
        }
        for (task_id, previous_owner, collected) in &retry_candidates {
            if os
                .get_process(*previous_owner)
                .is_some_and(|process| process.state != ProcessState::Terminated)
            {
                continue;
            }
            dead_owners.insert(*previous_owner);
            if adopted.contains(task_id)
                || task_owners.get(task_id) == Some(&owner_pid)
                || (*collected && !task_owners.contains_key(task_id))
            {
                retry_ids.push(task_id.clone());
            }
        }
        Ok((adopted, retry_ids, dead_owners, failures))
    })?;

    {
        let mut registry = lock_task_registry();
        for (task_id, snapshot) in &candidates {
            if adopted.contains(task_id)
                && let Some(entry) = registry.get_mut(task_id)
                && entry.session_id == session_id
                && entry.owner_pid == snapshot.owner_pid
                && entry.pid == snapshot.pid
                && entry.result_channel_id == snapshot.result_channel_id
            {
                entry.owner_pid = owner_pid;
            }
        }
    }
    {
        let mut registry = lock_unpoisoned(&TASK_RETRY_REGISTRY);
        for task_id in retry_ids {
            if let Some(spec) = registry.get_mut(&task_id)
                && spec.session_id == session_id
                && dead_owners.contains(&spec.owner_pid)
            {
                spec.owner_pid = owner_pid;
            }
        }
    }
    // The old turn's waits have ended; its tasks have not. Never turn an expired wait budget
    // into a task timeout, or carry its owner identity into the replacement turn.
    lock_unpoisoned(&TASK_WAIT_STATES)
        .retain(|key, _| key.session_id != session_id || !dead_owners.contains(&key.owner_pid));
    lock_unpoisoned(&TASK_WAIT_NOOP_COUNTS)
        .retain(|key, _| key.session_id != session_id || !dead_owners.contains(&key.owner_pid));
    if !failures.is_empty() {
        return Err(format!(
            "Task ownership recovery failed: {}. The original tasks are retained. Do not retry or resume a backend while its original run may still be active. Once it has stopped, collect with task_wait/task_status before task_retry; if the runtime is unavailable, start a new task only after confirming the original run has stopped.",
            failures.join("; ")
        ));
    }
    Ok(adopted)
}
