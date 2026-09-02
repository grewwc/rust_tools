use super::*;

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_retry",
        description: "",

        execute: execute_task_retry,
    }
});

pub(super) fn execute_task_retry(args: &Value) -> Result<String, String> {
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

pub(super) fn ensure_top_level_task_orchestration(tool_name: &str) -> Result<(), String> {
    if crate::ai::driver::runtime_ctx::current_subagent_depth() == 0 {
        return Ok(());
    }
    Err(format!(
        "{tool_name} is only available to top-level agents. This subagent is a leaf task; complete the assigned work directly instead of waiting on, inspecting, or cancelling parent-owned subagent tasks."
    ))
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

pub(super) fn execute_task_integrate(args: &Value) -> Result<String, String> {
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
            .filter(|value| !value.is_empty());
        let Some(task_id) = task_id else {
            return Err(missing_task_id_error(history_file, session_id));
        };
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

/// Error text for a missing or empty `task_id`. Appends the ids of delivered but
/// still-unintegrated subagent results so the model can copy one verbatim instead of
/// re-guessing (the `[task_id=...]` markers in earlier results may be off-screen).
pub(super) fn missing_task_id_error(history_file: &std::path::Path, session_id: &str) -> String {
    let mut message = "Each task integration requires a non-empty task_id".to_string();
    // Use the full audit plus the same delivered-but-unintegrated filter as
    // `read_unintegrated_task_evidence` (which is not re-exported at history level).
    match crate::ai::history::read_task_spawn_audit(history_file, session_id) {
        Ok(records) => {
            let ids: Vec<&str> = records
                .iter()
                .filter(|record| {
                    record.delivered_at_unix_ms > 0 && record.integrated_at_unix_ms.is_none()
                })
                .map(|record| record.task_id.as_str())
                .collect();
            if ids.is_empty() {
                message.push_str(
                    "\nNo delivered-but-unintegrated subagent results found; run `task_audit` to list task ids, then retry with the exact task_id.",
                );
                return message;
            }
            message.push_str(&format!(
                "\nDelivered task results not yet integrated (copy one of these task_ids verbatim): {}",
                ids.join(", ")
            ));
        }
        Err(_) => {
            message.push_str(
                "\nRun `task_audit` to list delivered task ids, then retry with the exact task_id.",
            );
        }
    }
    message
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_integrate",
        description: "",

        execute: execute_task_integrate,
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
    // Same session-scoped cleanup for the task_wait no-op escalation counter: keys keyed by a
    // deleted session's id can never be bumped or reset again, so drop them to avoid accumulating
    // dead entries in the long-lived process.
    TASK_WAIT_NOOP_COUNTS
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
