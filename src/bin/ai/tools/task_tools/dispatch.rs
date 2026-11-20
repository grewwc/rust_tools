use super::*;
use crate::ai::tools::common::{ToolParallelPolicy, ToolParallelPolicyRegistration};

#[path = "task_evidence.rs"]
mod task_evidence;
pub(in crate::ai) use task_evidence::initial_agent_team_task_status_cursor;
pub(super) use task_evidence::{
    EvidenceReadOptions, StatusCursor, merge_status_cursor, read_status_cursor, write_status_cursor,
};

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_retry",
        description: "",

        execute: execute_task_retry,
    }
});

inventory::submit!(ToolHistoryPolicyRegistration {
    name: "task_retry",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::AfterIntegration,
        counts_toward_precision_inline_budget: false,
    },
});

pub(super) fn execute_task_retry(args: &Value) -> Result<String, String> {
    start_task_report(false);
    ensure_top_level_task_orchestration("task_retry")?;
    let task_id = args
        .get("task_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "task_retry requires non-empty 'task_id'".to_string())?;
    let current_owner_pid = current_task_owner_pid()?;
    let record = lock_unpoisoned(&TASK_RETRY_REGISTRY)
        .get_ref(&task_id.to_string())
        .cloned()
        .ok_or_else(|| format!("task_retry: unknown task_id '{task_id}'"))?;
    if record.session_id != crate::ai::driver::runtime_ctx::current_session_id_or_empty() {
        return Err(format!(
            "task_retry: task_id '{task_id}' belongs to another session"
        ));
    }

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
    if backend_session::task_backend_run_is_active(record.prepared.backend, task_id) {
        return Err(format!(
            "task_retry: task_id '{task_id}' has been collected but its backend run is still stopping; wait for it to end before retrying"
        ));
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
        prune: ToolPrunePolicy::AfterIntegration,
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
        let registry = lock_task_registry();
        registry
            .iter()
            .filter(|(_, e)| e.started_at.elapsed() > SUBAGENT_WALL_CLOCK_TIMEOUT)
            .map(|(task_id, e)| {
                (
                    task_id.clone(),
                    e.session_id.clone(),
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
    for (_, _, _, _, _, abort_handle, cancel_stream) in &candidates {
        cancel_stream.store(true, Ordering::Release);
        if let Some(handle) = abort_handle {
            handle.abort();
        }
    }
    // Step 3: hold only the kernel lock and check each process; if still running, kill it + write
    // the timeout terminal state.
    let _ = with_os_kernel(|os| {
        for (task_id, session_id, pid, result_channel_id, completion_futex_addr, _, _) in candidates
        {
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
                TaskTerminalStatus::Timeout.as_str(),
                &format!(
                    "Subagent exceeded wall-clock lifetime of {}s (reaped by scheduler; likely stuck in a non-returning tool execution)",
                    SUBAGENT_WALL_CLOCK_TIMEOUT.as_secs()
                ),
            );
            record_uncollected_terminal_status(
                &session_id,
                &task_id,
                TaskTerminalStatus::Timeout.as_str(),
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
        prune: ToolPrunePolicy::AfterIntegration,
        counts_toward_precision_inline_budget: false,
    },
});

pub(crate) fn execute_task_evidence_read(args: &Value) -> Result<String, String> {
    ensure_top_level_task_orchestration("task_evidence_read")?;
    let optimizations = crate::ai::config::agent_team_optimizations();
    let source = match args.get("source") {
        None => "progress",
        Some(Value::String(source)) if matches!(source.as_str(), "progress" | "delivered") => {
            source.as_str()
        }
        Some(_) => return Err("source must be 'progress' or 'delivered'".to_string()),
    };
    if source == "progress" && args.get("view").and_then(Value::as_str) == Some("summary") {
        return Err("view=summary is available only with source=delivered".to_string());
    }
    let task_id = args
        .get("task_id")
        .and_then(Value::as_str)
        // Delivered IDs are database keys, not filenames. Preserve their exact bytes;
        // trimming or normalizing them would make some durable records unretrievable.
        .map(|value| {
            if source == "progress" {
                value.trim()
            } else {
                value
            }
        })
        .filter(|value| !value.is_empty())
        .ok_or("Missing non-empty 'task_id' parameter")?;
    let owner_pid = current_task_owner_pid()?;
    let session_id = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
    if with_task_entry(task_id, |entry| {
        task_entry_owned_by(entry, &session_id, owner_pid)
    }) == Some(false)
    {
        return Err(format!(
            "Task {task_id} is owned by another process/session"
        ));
    }
    let options =
        EvidenceReadOptions::for_optimizations(args, optimizations.optimizations_enabled())?;

    if source == "delivered" {
        let context = crate::ai::driver::runtime_ctx::try_current()
            .ok_or("Delivered evidence reads require an active driver session")?;
        // Return before the progress branch: this lookup must neither refresh progress
        // nor collect a task result, acknowledge integration, or recreate a missing store.
        // After registry cleanup, durable evidence retains its existing session scope.
        let (status, payload) = crate::ai::history::read_task_evidence_status_payload(
            context.app_proto.config.history_file.as_path(),
            &context.app_proto.session_id,
            task_id,
        ).map_err(|error| format!("Failed to read delivered evidence for task {task_id}: {error}"))?
            .ok_or_else(|| format!("No delivered evidence for task {task_id} in this session; use task_audit to list recorded IDs"))?;
        let report = task_evidence::read_delivered_task_report(
            context.app_proto.config.history_file.as_path(),
            &context.app_proto.session_id,
            task_id,
        )
        // Optional display metadata cannot block access to the durable payload.
        .ok()
        .flatten()
        .filter(|report| report.status == status);
        let mut evidence = serde_json::json!({
            "task_id": task_id,
            "source": "delivered",
            "status": status,
            "payload": payload,
        });
        if let Some(verification) = report
            .as_ref()
            .and_then(|report| report.verification.as_ref())
        {
            // Only the full raw payload guarantees that matching standalone metadata
            // remains visible. Summaries, pages, and field selections need it separately.
            // Also inspect the original body: a task header can hide a handoff prefix.
            if !options.needs_full_delivered()
                || !super::wait::verification_is_in_task_output(&payload, verification)
                || !report
                    .as_ref()
                    .and_then(|report| report.output.as_deref())
                    .is_some_and(|output| {
                        super::wait::verification_is_in_task_output(output, verification)
                    })
            {
                evidence["verification"] = serde_json::to_value(verification)
                    .expect("verification metadata is a JSON value");
            }
            evidence["assessment"] = serde_json::to_value(verification.assessment())
                .expect("verification assessment is serializable");
        }
        let full = if options.needs_full_delivered() {
            serde_json::to_string_pretty(&evidence)
                .map_err(|error| format!("Failed to render delivered evidence: {error}"))?
        } else {
            String::new()
        };
        return options.render_delivered(task_id, full, evidence, report.as_ref());
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
        Ok(content) => options.render_progress(task_id, content),
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

inventory::submit!(ToolParallelPolicyRegistration {
    name: "task_evidence_read",
    // The default `progress` source persists a refreshed snapshot before it
    // reads, so this is not a pure read even though the caller only observes
    // evidence.
    policy: ToolParallelPolicy::Serial,
});

inventory::submit!(ToolHistoryPolicyRegistration {
    name: "task_evidence_read",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::AfterIntegration,
        counts_toward_precision_inline_budget: false,
    },
});

/// Presents the complete subagent call ledger of the current session (spawn audit).
///
/// Complements `task_evidence_read` (per-task progress or delivered evidence): this is a model-visible audit
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
            "No subagent calls recorded in this session yet. Use `task_spawn` \
             to delegate work; every spawn is persisted here."
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
    // Validate the whole batch before touching the store. A later invalid item must not leave
    // earlier items integrated while the tool reports an overall failure.
    let mut integrated = Vec::new();
    let mut seen = FxHashSet::default();
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
        if !seen.insert(task_id.to_string()) {
            return Err(format!(
                "Duplicate task_id in task_integrate batch: {task_id}"
            ));
        }
        integrated.push((
            task_id.to_string(),
            disposition.to_string(),
            summary.to_string(),
        ));
    }
    let unknown =
        crate::ai::history::integrate_task_evidence_batch(history_file, session_id, &integrated)
            .map_err(|error| format!("failed to integrate task results: {error}"))?;
    if let Some(task_id) = unknown.first() {
        return Err(format!(
            "Unknown task_id in durable task evidence ledger (no delivered result): {task_id}. \
             No integrations were applied."
        ));
    }
    let mut reply = format!(
        "Integrated {} task result(s): {}",
        integrated.len(),
        integrated
            .iter()
            .map(|(task_id, _, _)| task_id.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    // The parent decision and atomic transaction above remain authoritative. Metadata reads
    // and advisory gaps must never turn successful integration into a tool failure.
    let mut advisory_bytes = 0;
    let mut omitted = 0;
    for (task_id, disposition, _) in &integrated {
        if disposition != "accepted" {
            continue;
        }
        let report = task_evidence::read_delivered_task_report(history_file, session_id, task_id)
            .ok()
            .flatten();
        let Some(verification) = report
            .as_ref()
            .and_then(|report| report.verification.as_ref())
        else {
            continue;
        };
        if verification.assessment().gaps.is_empty() {
            continue;
        }
        let label: String = task_id.chars().take(96).collect();
        let advisory = format!(
            "\nVerification advisory for task_id={label}:\n{}",
            verification.inline_assessment()
        );
        if advisory_bytes + advisory.len() <= 2048 {
            advisory_bytes += advisory.len();
            reply.push_str(&advisory);
        } else {
            omitted += 1;
        }
    }
    if omitted > 0 {
        reply.push_str(&format!("\n{omitted} additional verification advisories; read each delivered assessment for details."));
    }
    crate::ai::tools::enable_tools::settle_task_lifecycle_mount();
    Ok(reply)
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
        prune: ToolPrunePolicy::AfterIntegration,
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
        let registry = lock_task_registry();
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
                    entry.backend.unwrap_or_default().id(),
                )
            })
            .collect::<Vec<_>>()
    };
    for (_, _, _, _, abort_handle, cancel_stream, _) in &candidates {
        cancel_stream.store(true, Ordering::Release);
        if let Some(handle) = abort_handle {
            handle.abort();
        }
    }

    if !candidates.is_empty() {
        let _ = with_os_kernel(|os| {
            for (_, pid, result_channel_id, completion_futex_addr, _, _, _) in &candidates {
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
        .map(|(task_id, _, _, _, _, _, backend_id)| {
            remove_task_entry(&task_id);
            backend_session::forget_session(backend_id, &task_id);
            task_id
        })
        .collect::<Vec<_>>();
    {
        let mut progress_registry = lock_unpoisoned(&TASK_PROGRESS_REGISTRY);
        for task_id in &task_ids {
            progress_registry.remove(task_id);
        }
    }
    {
        let mut retry_registry = lock_unpoisoned(&TASK_RETRY_REGISTRY);
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
    execute_task_cancel_with_progress(args, |task_id| {
        task_progress_slot(task_id).and_then(|slot| {
            crate::ai::driver::runtime_ctx::subagent_progress_state_snapshot(&slot)
        })
    })
}

fn execute_task_cancel_with_progress(
    args: &Value,
    snapshot_for: impl Fn(&str) -> Option<crate::ai::driver::runtime_ctx::SubagentProgressSnapshot>,
) -> Result<String, String> {
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
    let force = match args.get("force") {
        None => false,
        Some(Value::Bool(force)) => *force,
        Some(_) => return Err("'force' must be a boolean".to_string()),
    };
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
        let registry = lock_task_registry();
        task_ids
            .iter()
            .filter_map(|tid| match registry.get_ref(tid) {
                Some(entry)
                    if task_entry_owned_by(entry, &current_session_id, current_owner_pid) =>
                {
                    Some((tid.clone(), entry.clone()))
                }
                _ => {
                    not_found.push(tid.clone());
                    None
                }
            })
            .collect::<Vec<_>>()
    };

    // Sample terminal truth before progress, holding neither the task registry nor the kernel
    // lock while acquiring a progress slot. Every requested task is checked before any abort.
    let candidates = with_os_kernel(|os| {
        candidates
            .into_iter()
            .map(|(tid, entry)| {
                let (_, active) = task_state_snapshot(os, entry.result_channel_id, entry.pid)?;
                Ok((tid, entry, active))
            })
            .collect::<Result<Vec<_>, String>>()
    })?;
    let recent = candidates
        .iter()
        .filter(|(_, _, active)| *active)
        .filter_map(|(tid, _, _)| {
            let snapshot = snapshot_for(tid)?;
            if force || snapshot.stale_for >= TASK_CANCEL_RECENT_PROGRESS {
                return None;
            }
            let progress = TaskReportProgress::from(&snapshot);
            Some(format!(
                "{tid}: last activity {}ms ago (stale_for_ms={}, elapsed_ms={}, sequence={}, phase={:?}, latest_activity={:?})",
                progress.stale_for_ms,
                progress.stale_for_ms,
                progress.elapsed_ms,
                progress.sequence,
                progress.phase,
                progress.latest_activity.as_deref().unwrap_or(&progress.phase),
            ))
        })
        .collect::<Vec<_>>();
    if !recent.is_empty() {
        return Err(format!(
            "[task_cancel] Cancelled nothing: recent progress protects these active tasks:\n{}\nRun task_status and keep waiting while progress advances. Pass force: true only after checking progress and confirming that work is superseded or no longer needed, never to shorten a wait.",
            recent.join("\n")
        ));
    }

    for (tid, entry, active) in candidates {
        if !active {
            already_finished.push(tid);
            continue;
        }
        // Only cancel subagents that are still running. Tasks that already ended (completed
        // normally / failed / process terminated) are neither killed nor given a terminal result —
        // otherwise a "cancelled" message would be appended to the channel and the channel
        // destroyed, masking/discarding the subagent's real result and making a later task_wait
        // read a bogus cancelled state. Cleanup of channel/futex/registry for already-ended tasks
        // is left to the collector (task_wait's ready / failure path, or a task_wait after
        // task_status).
        let was_pending = with_os_kernel(|os| {
            if !task_state_snapshot(os, entry.result_channel_id, entry.pid)?.1 {
                return Ok(false);
            }
            // Abort is nonblocking. Rechecking while holding the kernel lock protects a result
            // published between preflight and cancellation without touching finished workers.
            entry.cancel_stream.store(true, Ordering::Release);
            if let Some(handle) = &entry.abort_handle {
                handle.abort();
            }
            write_terminal_subagent_result(
                os,
                &tid,
                entry.pid,
                entry.result_channel_id,
                entry.completion_futex_addr,
                TaskTerminalStatus::Cancelled.as_str(),
                &reason,
            );
            Ok(true)
        })?;
        if was_pending {
            record_uncollected_terminal_status(
                &entry.session_id,
                &tid,
                TaskTerminalStatus::Cancelled.as_str(),
            );
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

fn task_status_filter(args: &Value, enabled: bool) -> Result<Option<&str>, String> {
    if !enabled {
        return Ok(None);
    }
    Ok(match args.get("task_id") {
        None => None,
        Some(Value::String(task_id)) if !task_id.trim().is_empty() => Some(task_id.trim()),
        Some(_) => return Err("task_status requires non-empty string 'task_id'".to_string()),
    })
}

pub(crate) fn execute_task_status(args: &Value) -> Result<String, String> {
    start_task_report(false);
    ensure_top_level_task_orchestration("task_status")?;
    let optimizations = crate::ai::config::agent_team_optimizations();
    let task_id = task_status_filter(args, optimizations.optimizations_enabled())?;
    let current_session_id = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
    let current_owner_pid = current_task_owner_pid()?;
    let tracked = {
        let registry = lock_task_registry();
        registry
            .iter()
            .filter(|(_, entry)| task_entry_owned_by(entry, &current_session_id, current_owner_pid))
            .filter(|(tid, _)| task_id.is_none_or(|task_id| tid.as_str() == task_id))
            .map(|(tid, entry)| (tid.clone(), entry.clone()))
            .collect::<Vec<_>>()
    };
    let result = if is_agent_team_mode() {
        match task_id {
            None => super::quiet::execute_agent_team_task_status(&tracked),
            Some(_) => super::quiet::execute_agent_team_task_status_filtered(&tracked, task_id),
        }
    } else {
        execute_full_task_status(tracked)
    };
    if task_id.is_some() {
        // A filtered sample must not remove unrelated tasks from the display's live snapshot.
        start_task_report(false);
    }
    result
}

pub(super) fn execute_full_task_status(
    tracked: Vec<(String, AsyncTaskEntry)>,
) -> Result<String, String> {
    start_task_report(false);
    if tracked.is_empty() {
        start_task_report(true);
        return Ok("No async tasks currently tracked.".to_string());
    }

    let model_width = tracked
        .iter()
        .map(|(_, entry)| entry.model.len())
        .max()
        .unwrap_or(0)
        .max(14);
    let mut lines = vec![format!(
        "{:<37} {:<8} {:<14} {:<model_width$} {:<17} {:<11} {}",
        "TaskID", "PID", "Agent", "Model", "Effort", "State", "Description"
    )];
    // For subtasks that already wrote their result back to the channel, **consume and clean up**
    // the body directly and append it after the table. Otherwise, even if the model sees
    // state=completed, it can only get the output by calling task_wait again; worse, if it treats
    // "seen completed in task_status" as handled, it would bypass the collection guard and leave
    // registry entries and channel/futex resources behind. Since the result is already returned
    // to the model here, treat it as collected.
    let mut completed_outputs: Vec<String> = Vec::new();
    let mut finished_ids: Vec<String> = Vec::new();
    let mut printed_progress = false;
    let kernel_result = with_os_kernel(|os| {
        for (tid, entry) in &tracked {
            let state_str = task_state_string(os, entry.result_channel_id, entry.pid)?;
            let reasoning_effort = task_reasoning_effort(tid);
            lines.push(format!(
                "{:<37} {:<8} {:<14} {:<model_width$} {:<17} {:<11} {}",
                tid,
                entry.pid,
                entry.agent_name,
                entry.model,
                reasoning_effort.as_deref().unwrap_or("-"),
                state_str,
                entry.description
            ));
            let progress = match (
                current_task_parent_progress(os, tid),
                pending_task_side_note_progress(tid),
            ) {
                (Some(progress), Some(pending)) => Some(format!("{progress} · {pending}")),
                (progress, pending) => progress.or(pending),
            };
            if let Some(progress) = progress {
                lines.push(format!("  progress[{tid}]: {progress}"));
                printed_progress = true;
            }
            if let Some(collected) =
                drain_task_result(os, tid, entry, "subagent result collected by task_status")?
            {
                completed_outputs.push(collected.rendered);
                finished_ids.push(tid.clone());
            }
        }
        Ok(())
    });
    // Remove collected entries even when a later task failed mid-batch. Their channels and
    // futexes are already destroyed, so a stranded entry would make every later task_status /
    // task_wait call fail on a missing channel.
    for task_id in &finished_ids {
        remove_task_entry(task_id);
    }
    kernel_result?;

    if printed_progress {
        lines.push(
            "Progress notes: recent last activity means the task is still working; a terminal state or a long-stale snapshot means it is stalled. Use task_wait only when blocked on a result."
                .to_string(),
        );
    }

    if !completed_outputs.is_empty() {
        lines.push(String::new());
        lines.push(
            "Completed task results below (already collected — no need to wait for these):"
                .to_string(),
        );
        lines.push(completed_outputs.join("\n\n---\n\n"));
    }

    start_task_report(true);
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod optimization_gate_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn task_evidence_master_gate_controls_runtime_projection() {
        for config in [
            "",
            "ai.agent_team.optimizations_enabled = false\nai.agent_team.persist_status_cursor = true\n",
            "ai.agent_team.optimizations_enabled = true\n",
        ] {
            super::super::agent_team::tests::with_status_test_context_with_config(config, || {
                let task_id = "gate-evidence";
                let path = task_progress_file_path(task_id).unwrap();
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                let original = " {\"phase\":\"read\",\"checkpoint_excerpt\":\"α界🙂z\"} \n";
                fs::write(&path, original).unwrap();
                let full = execute_task_evidence_read(&json!({"task_id": task_id})).unwrap();
                assert_eq!(full.as_bytes(), original.as_bytes());
                let selected = execute_task_evidence_read(
                    &json!({"task_id": task_id, "fields": ["checkpoint_excerpt"]}),
                )
                .unwrap();
                let bounded =
                    execute_task_evidence_read(&json!({"task_id": task_id, "max_chars": 0}))
                        .unwrap();
                let invalid = execute_task_evidence_read(
                    &json!({"task_id": task_id, "fields": [], "max_chars": -1}),
                );
                let selected: Value = serde_json::from_str(&selected).unwrap();
                assert_eq!(selected["payload"], json!({"checkpoint_excerpt": "α界🙂z"}));
                let bounded: Value = serde_json::from_str(&bounded).unwrap();
                assert_eq!(bounded["payload"], "");
                assert_eq!(bounded["truncated"], true);
                assert_eq!(bounded["next_cursor"], 0);
                assert!(invalid.is_err());
                assert!(
                    execute_task_evidence_read(&json!({"task_id": task_id, "view": "summary"}))
                        .unwrap_err()
                        .contains("source=delivered")
                );
            });
        }
    }

    struct RegisteredTasks(Vec<String>);

    impl Drop for RegisteredTasks {
        fn drop(&mut self) {
            for task_id in &self.0 {
                remove_task_entry(task_id);
                lock_unpoisoned(&TASK_PROGRESS_REGISTRY).remove(task_id);
            }
        }
    }

    fn register_pending_tasks() -> RegisteredTasks {
        let session_id = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
        let owner_pid = current_task_owner_pid().unwrap();
        let mut tasks = RegisteredTasks(Vec::new());
        for _ in 0..2 {
            let task_id = uuid::Uuid::new_v4().to_string();
            let (pid, channel, futex) = with_os_kernel(|os| {
                let pid = os.spawn(
                    Some(owner_pid),
                    "worker".to_string(),
                    "pending gate test".to_string(),
                    20,
                    4,
                    None,
                    None,
                )?;
                let channel = os.channel_create(Some(owner_pid), 1, "gate-result".to_string());
                let futex = os.futex_create(0, "gate-complete".to_string());
                Ok((pid, channel.raw(), futex))
            })
            .unwrap();
            insert_task_entry_for_test(
                task_id.clone(),
                AsyncTaskEntry {
                    session_id: session_id.clone(),
                    evidence_history_file: None,
                    owner_pid,
                    pid,
                    result_channel_id: channel,
                    completion_futex_addr: futex,
                    description: "pending gate test".to_string(),
                    agent_name: "worker".to_string(),
                    backend: None,
                    model: "test-model".to_string(),
                    is_model_auto_selected: false,
                    auto_model_fallback: None,
                    selection_explanation: "gate fixture".to_string(),
                    inherit: InheritOptions::default(),
                    abort_handle: None,
                    cancel_stream: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                    started_at: Instant::now(),
                    last_progress_notification_at: None,
                    last_progress_persisted_at: None,
                },
            );
            tasks.0.push(task_id);
        }
        tasks
    }

    fn with_cancellation_test_context<T>(operation: impl FnOnce() -> T) -> T {
        super::super::agent_team::tests::with_status_test_context_with_config(
            "ai.agent_team.optimizations_enabled = true\n",
            operation,
        )
    }

    fn cancellation_progress(task_id: &str, checkpoint: Option<&str>) {
        use crate::ai::driver::runtime_ctx;
        let slot = runtime_ctx::new_subagent_progress_slot();
        runtime_ctx::publish_subagent_activity_to_slot(
            &slot,
            "working",
            "read_file cancellation evidence",
        );
        if let Some(checkpoint) = checkpoint {
            runtime_ctx::publish_subagent_checkpoint_to_slot(&slot, "Saved finding", checkpoint);
        }
        lock_unpoisoned(&TASK_PROGRESS_REGISTRY).insert(task_id.to_string(), slot);
    }

    fn assert_cancellation_left_active(task_id: &str) {
        let entry = with_task_entry(task_id, Clone::clone).unwrap();
        assert!(!entry.cancel_stream.load(Ordering::Acquire));
        with_os_kernel(|os| {
            assert!(is_task_pending(os, entry.pid)?);
            assert!(read_task_result(os, entry.result_channel_id, false)?.is_none());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn task_cancel_rejects_recent_activity_without_force() {
        with_cancellation_test_context(|| {
            let tasks = register_pending_tasks();
            cancellation_progress(&tasks.0[0], None);
            let error = execute_task_cancel(&json!({"task_ids": [&tasks.0[0]]})).unwrap_err();
            for expected in [
                tasks.0[0].as_str(),
                "Cancelled nothing",
                "stale_for_ms=",
                "elapsed_ms=",
                "sequence=",
                "working",
                "read_file cancellation evidence",
                "task_status",
                "force: true",
            ] {
                assert!(error.contains(expected), "{error}");
            }
            assert_cancellation_left_active(&tasks.0[0]);
        });
    }

    #[test]
    fn task_cancel_mixed_batch_cancels_nothing_when_one_task_is_recent() {
        with_cancellation_test_context(|| {
            let tasks = register_pending_tasks();
            cancellation_progress(&tasks.0[1], None);
            let error =
                execute_task_cancel(&json!({"task_ids": tasks.0, "force": false})).unwrap_err();
            assert!(error.contains(&tasks.0[1]));
            for task_id in &tasks.0 {
                assert_cancellation_left_active(task_id);
            }
        });
    }

    #[test]
    fn task_cancel_force_preserves_checkpoint_in_wait_report_and_evidence() {
        with_cancellation_test_context(|| {
            let tasks = register_pending_tasks();
            let task_id = &tasks.0[0];
            let checkpoint = "Candidate finding: cancellation must preserve collected evidence.";
            cancellation_progress(task_id, Some(checkpoint));
            let output = execute_task_cancel(&json!({
                "task_ids": [task_id], "force": true, "reason": "work superseded"
            }))
            .unwrap();
            assert!(output.contains("Cancelled 1 task(s)"));
            let entry = with_task_entry(task_id, Clone::clone).unwrap();
            assert!(entry.cancel_stream.load(Ordering::Acquire));
            let raw = with_os_kernel(|os| {
                assert!(!is_task_pending(os, entry.pid)?);
                os.channel_peek(None, ChannelId(entry.result_channel_id))
            })
            .unwrap();
            let IpcRecvResult::Message(raw) = raw else {
                panic!("cancelled result must remain in its channel");
            };
            let terminal: Value = serde_json::from_str(&raw).unwrap();
            let result: StoredTaskResult = serde_json::from_str(&raw).unwrap();
            assert_eq!(terminal["status"], "cancelled");
            assert_eq!(terminal["cancel_reason"], "work superseded");
            assert!(terminal.get("error").is_none());
            assert!(
                result
                    .partial_output
                    .as_deref()
                    .unwrap()
                    .contains(checkpoint)
            );
            assert!(
                result
                    .partial_output
                    .as_deref()
                    .unwrap()
                    .contains("Saved checkpoint excerpt")
            );
            assert_eq!(
                result.partial_output.as_deref(),
                Some(result.output.as_str())
            );

            let (wait, report) = capture_task_report("task_wait", || {
                execute_task_wait(&json!({"task_ids": [task_id]}))
            });
            let wait = wait.unwrap();
            assert!(wait.contains("Cancellation reason: work superseded"));
            assert!(wait.contains("Partial output:"));
            assert!(wait.contains(checkpoint));
            assert!(!wait.contains("Failure reason:"));
            let report = serde_json::to_value(report.unwrap()).unwrap();
            assert_eq!(report["tasks"][0]["cancel_reason"], "work superseded");
            assert!(report["tasks"][0].get("error").is_none());
            assert_eq!(
                report["tasks"][0]["partial_output"],
                terminal["partial_output"]
            );

            let evidence: Value = serde_json::from_str(
                &execute_task_evidence_read(
                    &json!({"task_id": task_id, "source": "delivered", "view": "raw"}),
                )
                .unwrap(),
            )
            .unwrap();
            assert_eq!(evidence["cancel_reason"], "work superseded");
            assert_eq!(evidence["partial_output"], terminal["partial_output"]);
            assert!(evidence.get("error").is_none());
            let summary: Value = serde_json::from_str(
                &execute_task_evidence_read(&json!({"task_id": task_id, "source": "delivered"}))
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(summary["cancel_reason"], "work superseded");
            assert!(summary.get("error").is_none());
            let selected: Value = serde_json::from_str(
                &execute_task_evidence_read(&json!({
                    "task_id": task_id, "source": "delivered",
                    "fields": ["cancel_reason", "partial_output"]
                }))
                .unwrap(),
            )
            .unwrap();
            assert_eq!(selected["payload"]["cancel_reason"], "work superseded");
            assert_eq!(
                selected["payload"]["partial_output"],
                terminal["partial_output"]
            );
        });
    }

    #[test]
    fn task_cancel_allows_stale_progress_at_and_beyond_threshold() {
        with_cancellation_test_context(|| {
            let tasks = register_pending_tasks();
            for (task_id, stale_ms) in tasks.0.iter().zip([60_000, 120_000]) {
                let output =
                    execute_task_cancel_with_progress(&json!({"task_ids": [task_id]}), |_| {
                        Some(
                            serde_json::from_value(json!({
                                "phase": "working", "checkpoint_summary": null,
                                "elapsed_ms": 376_000, "stale_for_ms": stale_ms,
                                "checkpoint_due": true, "sequence": 142, "timeline": []
                            }))
                            .unwrap(),
                        )
                    })
                    .unwrap();
                assert!(output.contains("Cancelled 1 task(s)"));
                let entry = with_task_entry(task_id, Clone::clone).unwrap();
                assert!(entry.cancel_stream.load(Ordering::Acquire));
                assert_eq!(
                    with_os_kernel(|os| read_task_result(os, entry.result_channel_id, false))
                        .unwrap()
                        .unwrap()
                        .status,
                    "cancelled"
                );
            }
        });
    }

    #[test]
    fn task_cancel_allows_missing_progress_without_fabricating_output() {
        with_cancellation_test_context(|| {
            let tasks = register_pending_tasks();
            execute_task_cancel(&json!({"task_ids": [&tasks.0[0]]})).unwrap();
            let entry = with_task_entry(&tasks.0[0], Clone::clone).unwrap();
            let result = with_os_kernel(|os| read_task_result(os, entry.result_channel_id, false))
                .unwrap()
                .unwrap();
            assert_eq!(
                result.cancel_reason.as_deref(),
                Some("cancelled by parent agent")
            );
            assert!(result.error.is_none());
            assert!(result.partial_output.is_none());
            assert!(result.output.is_empty());
        });
    }

    #[test]
    fn task_cancel_force_preserves_latest_activity_in_task_status_report() {
        with_cancellation_test_context(|| {
            let tasks = register_pending_tasks();
            let task_id = &tasks.0[0];
            cancellation_progress(task_id, None);
            execute_task_cancel(&json!({"task_ids": [task_id], "force": true})).unwrap();
            let (status, report) = capture_task_report("task_status", || {
                execute_task_status(&json!({"task_id": task_id}))
            });
            let status = status.unwrap();
            assert!(status.contains("Cancellation reason: cancelled by parent agent"));
            assert!(status.contains("Last recorded activity (partial progress):"));
            assert!(status.contains("read_file cancellation evidence"));
            assert!(!status.contains("Failure reason:"));
            let report = report.unwrap();
            assert_eq!(report.tasks.len(), 1);
            assert_eq!(report.tasks[0].status, "cancelled");
            assert!(report.tasks[0].error.is_none());
            assert!(
                report.tasks[0]
                    .partial_output
                    .as_deref()
                    .unwrap()
                    .contains("read_file")
            );
        });
    }

    #[test]
    fn task_cancel_leaves_terminal_tasks_and_published_assistant_output_untouched() {
        with_cancellation_test_context(|| {
            let tasks = register_pending_tasks();
            for task_id in &tasks.0 {
                cancellation_progress(task_id, Some("unfinished checkpoint"));
            }
            let entry = with_task_entry(&tasks.0[0], Clone::clone).unwrap();
            with_os_kernel(|os| {
                os.channel_send(
                    Some(entry.pid),
                    ChannelId(entry.result_channel_id),
                    json!({
                        "status": "completed", "output": "exact final assistant text", "error": null
                    })
                    .to_string(),
                )?;
                let terminated = with_task_entry(&tasks.0[1], Clone::clone).unwrap();
                os.kill_process(terminated.pid, "already terminated".into())?;
                Ok(())
            })
            .unwrap();
            let output = execute_task_cancel(
                &json!({"task_ids": [&tasks.0[0], &tasks.0[1], "missing-task"]}),
            )
            .unwrap();
            assert!(output.contains("2 task_id(s) were already finished"));
            assert!(output.contains("missing-task"));
            for task_id in &tasks.0 {
                assert!(
                    !with_task_entry(task_id, |entry| entry.cancel_stream.load(Ordering::Acquire))
                        .unwrap()
                );
            }
            let result = with_os_kernel(|os| read_task_result(os, entry.result_channel_id, false))
                .unwrap()
                .unwrap();
            assert_eq!(result.status, "completed");
            assert_eq!(result.output, "exact final assistant text");
            assert!(result.cancel_reason.is_none());
        });
    }

    #[test]
    fn task_cancel_rechecks_results_published_after_preflight() {
        with_cancellation_test_context(|| {
            let tasks = register_pending_tasks();
            let task_id = &tasks.0[0];
            let entry = with_task_entry(task_id, Clone::clone).unwrap();
            let output = execute_task_cancel_with_progress(&json!({"task_ids": [task_id]}), |_| {
                with_os_kernel(|os| {
                    os.channel_send(Some(entry.pid), ChannelId(entry.result_channel_id), json!({
                        "status": "completed", "output": "won the publication race", "error": null
                    }).to_string())
                })
                .unwrap();
                None
            })
            .unwrap();
            assert!(output.contains("already finished"));
            assert!(!entry.cancel_stream.load(Ordering::Acquire));
            assert_eq!(
                with_os_kernel(|os| read_task_result(os, entry.result_channel_id, false))
                    .unwrap()
                    .unwrap()
                    .output,
                "won the publication race"
            );
        });
    }

    #[test]
    fn task_cancel_requires_a_boolean_force_and_advertises_default_false() {
        let schema: Value =
            serde_json::from_str(include_str!("../../tool_descriptions/task_cancel.json")).unwrap();
        assert_eq!(
            schema["parameters"]["properties"]["force"]["type"],
            "boolean"
        );
        assert_eq!(
            schema["parameters"]["properties"]["force"]["default"],
            false
        );
        with_cancellation_test_context(|| {
            let tasks = register_pending_tasks();
            assert_eq!(
                execute_task_cancel(&json!({"task_ids": tasks.0, "force": "true"})).unwrap_err(),
                "'force' must be a boolean"
            );
            for task_id in &tasks.0 {
                assert_cancellation_left_active(task_id);
            }
        });
    }

    #[test]
    fn task_status_master_gate_controls_filtering_and_validation() {
        for (config, enabled) in [
            ("", false),
            (
                "ai.agent_team.optimizations_enabled = false\nai.agent_team.persist_status_cursor = true\n",
                false,
            ),
            ("ai.agent_team.optimizations_enabled = true\n", true),
        ] {
            super::super::agent_team::tests::with_status_test_context_with_config(config, || {
                let tasks = register_pending_tasks();
                let filtered =
                    execute_task_status(&json!({"task_id": format!(" {} ", tasks.0[0])})).unwrap();
                assert!(filtered.contains(&tasks.0[0]));
                assert_eq!(filtered.contains(&tasks.0[1]), !enabled);
                for invalid in [
                    json!({"task_id": " "}),
                    json!({"task_id": null}),
                    json!({"task_id": 7}),
                ] {
                    let result = execute_task_status(&invalid);
                    if enabled {
                        assert!(result.unwrap_err().contains("non-empty string 'task_id'"));
                    } else {
                        let output = result.unwrap();
                        assert!(output.contains(&tasks.0[0]));
                        assert!(output.contains(&tasks.0[1]));
                    }
                }
                assert!(with_task_entry(&tasks.0[0], |_| ()).is_some());
                assert!(with_task_entry(&tasks.0[1], |_| ()).is_some());
            });
        }
    }
}

/// Read-only verification access for on-demand runtime navigation hints.
pub(crate) fn read_delivered_verification_report(
    history_file: &std::path::Path,
    session_id: &str,
    task_id: &str,
) -> Result<Option<super::TaskReportTask>, String> {
    task_evidence::read_delivered_task_report(history_file, session_id, task_id)
}
