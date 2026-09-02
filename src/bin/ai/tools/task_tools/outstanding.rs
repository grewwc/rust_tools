use super::*;

pub(super) fn collect_outstanding_task_snapshots(
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

pub(super) fn render_outstanding_task_anchor(snapshots: &[OutstandingTaskSnapshot]) -> String {
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

pub(super) fn read_task_result(
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
