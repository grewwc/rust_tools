use super::*;

pub(super) fn collect_outstanding_task_snapshots(
    session_id: &str,
    owner_pid: u64,
) -> Result<Vec<OutstandingTaskSnapshot>, String> {
    let registry = lock_task_registry();
    let tracked = registry
        .iter()
        .filter(|(_, entry)| task_entry_owned_by(entry, session_id, owner_pid))
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
                reasoning_effort: task_reasoning_effort(task_id),
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
        let effort = snapshot
            .reasoning_effort
            .as_deref()
            .map(|effort| format!(" effort={effort}"))
            .unwrap_or_default();
        lines.push(format!(
            "- task_id={} status={} agent={} model={}{} desc={}",
            snapshot.task_id,
            snapshot.status,
            snapshot.agent_name,
            snapshot.model,
            effort,
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

struct AgentTeamOutstandingTaskDigest {
    task_id: String,
    status: String,
    phase: String,
    idle_secs: u64,
}

fn render_agent_team_outstanding_task_anchor(
    tasks: &[AgentTeamOutstandingTaskDigest],
) -> Option<String> {
    if tasks.is_empty() {
        return None;
    }
    let mut lines = vec![OUTSTANDING_SUBAGENT_TASKS_NOTE_PREFIX.to_string()];
    for task in tasks {
        let task_id = if task
            .task_id
            .chars()
            .any(|ch| ch.is_control() || ch.is_whitespace())
        {
            serde_json::to_string(&task.task_id).expect("serializing a string cannot fail")
        } else {
            task.task_id.clone()
        };
        let status = task.status.split_whitespace().collect::<Vec<_>>().join(" ");
        let phase = task.phase.split_whitespace().collect::<Vec<_>>().join(" ");
        let mut phase_chars = phase.chars();
        let mut phase: String = phase_chars.by_ref().take(80).collect();
        if phase_chars.next().is_some() {
            phase.pop();
            phase.push('…');
        }
        lines.push(format!(
            "{} status={} phase={} idle={}s",
            task_id, status, phase, task.idle_secs
        ));
    }
    lines.push(
        "Collect with `task_wait`; full evidence: `task_evidence_read` (source=\"delivered\", task_id). Handle failures before finishing."
            .to_string(),
    );
    Some(lines.join("\n"))
}

pub(crate) fn build_agent_team_outstanding_task_anchor(
    session_id: &str,
) -> Result<Option<String>, String> {
    let Some(owner_pid) = with_os_kernel(|os| Ok(os.current_process_id()))? else {
        return Ok(None);
    };
    // Copy only the identity/state inputs, then release the registry before taking the kernel lock.
    let tracked = {
        let registry = lock_task_registry();
        registry
            .iter()
            .filter(|(_, entry)| task_entry_owned_by(entry, session_id, owner_pid))
            .map(|(task_id, entry)| {
                (
                    task_id.clone(),
                    entry.result_channel_id,
                    entry.pid,
                    entry.started_at,
                )
            })
            .collect::<Vec<_>>()
    };
    if tracked.is_empty() {
        return Ok(None);
    }
    with_os_kernel(|os| {
        let tasks = tracked
            .iter()
            .map(|(task_id, channel, pid, started_at)| {
                let progress = task_progress_slot(task_id).and_then(|slot| {
                    crate::ai::driver::runtime_ctx::subagent_progress_state_snapshot(&slot)
                });
                AgentTeamOutstandingTaskDigest {
                    task_id: task_id.clone(),
                    // Preserve the ID and diagnostic even when one child's stored result is invalid.
                    status: task_state_string(os, *channel, *pid)
                        .unwrap_or_else(|error| format!("error({error})")),
                    phase: progress
                        .as_ref()
                        .map_or_else(|| "-".to_string(), |progress| progress.phase.clone()),
                    idle_secs: progress
                        .map_or_else(|| started_at.elapsed(), |progress| progress.stale_for)
                        .as_secs(),
                }
            })
            .collect::<Vec<_>>();
        Ok(render_agent_team_outstanding_task_anchor(&tasks))
    })
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
    build_uncollected_tasks_notice(
        session_id,
        &format!("the tool iteration limit ({iteration_limit}) was reached"),
    )
}

/// Same snapshot fold as [`build_abandoned_tasks_notice`], but with a caller
/// supplied reason instead of the iteration-limit wording. Used when the turn
/// finalizes early for a reason other than the hard cap (e.g. no-tool
/// wrap-up): the model gets no further tool rounds, so still-running subtasks
/// are reported in the final answer rather than silently dropped or
/// re-demanded via a reopen that would fight the wrap-up brake.
pub(crate) fn build_uncollected_tasks_notice(
    session_id: &str,
    reason: &str,
) -> Result<Option<String>, String> {
    let Some(owner_pid) = current_task_owner_pid_opt() else {
        return Ok(None);
    };
    let snapshots = collect_outstanding_task_snapshots(session_id, owner_pid)?;
    if snapshots.is_empty() {
        return Ok(None);
    }
    for snapshot in &snapshots {
        if TaskTerminalStatus::parse(&snapshot.status).is_some() {
            record_uncollected_terminal_status(session_id, &snapshot.task_id, &snapshot.status);
        }
    }
    let mut lines = vec![format!(
        "The following {} spawned subagent task(s) were still outstanding when {}; their results were NOT collected and are not reflected in this answer:",
        snapshots.len(),
        reason
    )];
    for snapshot in &snapshots {
        let effort = snapshot
            .reasoning_effort
            .as_deref()
            .map(|effort| format!(" effort={effort}"))
            .unwrap_or_default();
        lines.push(format!(
            "- task_id={} status={} agent={} model={}{} desc={}",
            snapshot.task_id,
            snapshot.status,
            snapshot.agent_name,
            snapshot.model,
            effort,
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
    #[serde(default)]
    pub(crate) backend_session: Option<String>,
}

pub(super) fn read_task_result(
    os: &mut dyn Kernel,
    result_channel_id: u64,
    consume: bool,
) -> Result<Option<StoredTaskResult>, String> {
    if os.channel_meta(ChannelId(result_channel_id)).is_none() {
        return Ok(None);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_team_parent_context_task_note_is_compact_and_keeps_terminal_state() {
        let tasks = vec![
            AgentTeamOutstandingTaskDigest {
                task_id: "task-running".to_string(),
                status: "running".to_string(),
                phase: format!("read\nfiles {}", "界".repeat(100)),
                idle_secs: 12,
            },
            AgentTeamOutstandingTaskDigest {
                task_id: "task-failed".to_string(),
                status: "failed".to_string(),
                phase: "verify".to_string(),
                idle_secs: 25,
            },
            AgentTeamOutstandingTaskDigest {
                task_id: "task-invalid".to_string(),
                status: "error(Failed to decode\nstored task result)".to_string(),
                phase: "-".to_string(),
                idle_secs: 3,
            },
        ];
        let note = render_agent_team_outstanding_task_anchor(&tasks).unwrap();
        assert!(note.starts_with(OUTSTANDING_SUBAGENT_TASKS_NOTE_PREFIX));
        assert_eq!(note.lines().count(), tasks.len() + 2);
        assert!(note.contains("task-running status=running phase=read files "));
        assert!(note.contains("… idle=12s"));
        assert!(note.contains("task-failed status=failed phase=verify idle=25s"));
        assert!(note.contains("task-invalid status=error(Failed to decode stored task result)"));
        assert!(note.contains("task_evidence_read"));
        assert!(note.contains("source=\"delivered\""));
        assert!(!note.contains("model="));
        assert!(!note.contains("progress:"));
    }

    #[test]
    fn agent_team_parent_context_task_note_empty_is_absent() {
        assert!(render_agent_team_outstanding_task_anchor(&[]).is_none());
    }
}
