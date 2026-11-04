use super::*;

pub(super) static TASK_PROGRESS_REGISTRY: LazyLock<
    Mutex<SkipMap<String, crate::ai::driver::runtime_ctx::SubagentPhaseSlot>>,
> = LazyLock::new(|| Mutex::new(SkipMap::default()));
/// Progress evidence is written infrequently and the files are small, so a single in-process lock
/// guarantees that snapshots for the same task cannot truncate each other through the shared temp
/// path; ordering is also compared before persisting, so a late stale snapshot cannot overwrite
/// newer evidence.
pub(super) static TASK_PROGRESS_FILE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
pub(crate) fn task_progress_slot(
    task_id: &str,
) -> Option<crate::ai::driver::runtime_ctx::SubagentPhaseSlot> {
    lock_unpoisoned(&TASK_PROGRESS_REGISTRY)
        .get_ref(&task_id.to_string())
        .cloned()
}

pub(super) fn current_task_progress(task_id: &str) -> Option<String> {
    let progress = task_progress_slot(task_id)
        .and_then(|slot| crate::ai::driver::runtime_ctx::subagent_progress_snapshot(&slot))
        .filter(|value| !value.is_empty());
    match (progress, pending_task_side_note_progress(task_id)) {
        (Some(progress), Some(pending)) => Some(format!("{progress} · {pending}")),
        (progress, pending) => progress.or(pending),
    }
}

pub(super) fn pending_task_side_note_progress(task_id: &str) -> Option<String> {
    let deferred = with_task_entry(task_id, |entry| {
        entry.backend.is_some_and(SubagentBackend::is_traecli)
    })?;
    if !deferred {
        return None;
    }
    let ctx = crate::ai::driver::runtime_ctx::try_current()?;
    let count = crate::ai::driver::side_note::pending_side_note_count(
        &ctx.app_proto.session_history_file,
        Some(task_id),
    );
    (count > 0).then(|| format!("pending side-notes: {count} (queued for backend_continue_from)"))
}

pub(super) fn current_task_parent_progress(os: &mut dyn Kernel, task_id: &str) -> Option<String> {
    // Parent delivery paths pass their already-locked kernel borrow so terminal-state checks
    // never acquire the kernel mutex again. The progress registry only supplies a cloned slot;
    // its lock is released before the slot consumes the checkpoint revision for this response.
    let slot = task_progress_slot(task_id)?;
    let pending = with_task_entry(task_id, |entry| (entry.pid, entry.result_channel_id))
        .is_some_and(|(pid, channel)| {
            read_task_result(os, channel, false).is_ok_and(|result| result.is_none())
                && is_task_pending(os, pid).unwrap_or(false)
        });
    let snapshot =
        crate::ai::driver::runtime_ctx::take_subagent_parent_progress_snapshot(&slot, pending)?;
    let value = snapshot.display();
    (!value.is_empty()).then_some(value)
}

pub(super) fn append_task_progress_snapshots(
    os: &mut dyn Kernel,
    output: &mut String,
    task_ids: &[String],
) {
    let snapshots = task_ids
        .iter()
        .filter_map(|task_id| {
            let progress = current_task_parent_progress(os, task_id)?;
            Some(format!("- {task_id}: {progress}"))
        })
        .collect::<Vec<_>>();
    if snapshots.is_empty() {
        return;
    }
    output.push_str("\nLatest progress snapshots:\n");
    output.push_str(&snapshots.join("\n"));
}

/// Receives the unified structured snapshots published by subagents. In-memory state updates are
/// not throttled; status-line refresh notifications and disk snapshots are each edge-throttled
/// separately, while checkpoints always propagate immediately — so long tasks neither stay silent
/// nor cause an event storm.
pub(crate) fn record_subagent_progress_update(
    task_id: &str,
    snapshot: &crate::ai::driver::runtime_ctx::SubagentProgressSnapshot,
    kind: crate::ai::driver::runtime_ctx::SubagentProgressEventKind,
) {
    let now = Instant::now();
    let update = {
        let mut registry = lock_task_registry();
        let Some(entry) = registry.get_mut(&task_id.to_string()) else {
            return;
        };
        let persist = kind == crate::ai::driver::runtime_ctx::SubagentProgressEventKind::Checkpoint
            || entry.last_progress_persisted_at.is_none_or(|last| {
                now.saturating_duration_since(last) >= SUBAGENT_PROGRESS_PERSIST_INTERVAL
            });
        let notify = kind == crate::ai::driver::runtime_ctx::SubagentProgressEventKind::Checkpoint
            || entry.last_progress_notification_at.is_none_or(|last| {
                now.saturating_duration_since(last) >= SUBAGENT_PROGRESS_NOTIFY_INTERVAL
            });
        if persist {
            entry.last_progress_persisted_at = Some(now);
        }
        if notify {
            entry.last_progress_notification_at = Some(now);
        }
        (persist, notify)
    };

    if update.0 && persist_subagent_progress_snapshot(task_id, snapshot).is_err() {
        let mut registry = lock_task_registry();
        if let Some(entry) = registry.get_mut(&task_id.to_string())
            && entry.last_progress_persisted_at == Some(now)
        {
            // Undo the throttle marker when this persist fails, so the next progress event retries
            // immediately.
            entry.last_progress_persisted_at = None;
        }
    }
    if update.1 {
        // Progress events only refresh the foreground status line (notify_scheduler makes the
        // scheduling loop tick once to redraw the subagent status line); they must **never**
        // wake_process a parked parent agent back to Ready. Otherwise a progress event every 15s
        // would force the parent through a full model-call round trip just to park again, turning
        // the foreground into busy-wait spinning that burns an inference every ~9s (see the
        // agent-team fan-out stall). Only three terminal paths genuinely need to wake the parent,
        // and each triggers independently without relying on this code:
        //   1. Subtask finished/failed: channel_send + futex_store -> notify_events_completed;
        //   2. task_wait budget exhausted: wake_expired_task_waits;
        //   3. task_cancel: flips the cancel futex directly.
        crate::ai::driver::notify_scheduler();
    }
}

pub(super) fn task_progress_file_path(task_id: &str) -> Result<PathBuf, String> {
    if task_id.is_empty()
        || !task_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        return Err("invalid task id for progress evidence".to_string());
    }
    let assets = current_session_assets_dir()
        .ok_or_else(|| "session assets directory is unavailable".to_string())?;
    Ok(assets.join("task-progress").join(format!("{task_id}.json")))
}

pub(super) fn persist_subagent_progress_snapshot(
    task_id: &str,
    snapshot: &crate::ai::driver::runtime_ctx::SubagentProgressSnapshot,
) -> Result<(), String> {
    let path = task_progress_file_path(task_id)?;
    persist_subagent_progress_snapshot_at(&path, task_id, snapshot)
}

pub(super) fn persist_subagent_progress_snapshot_at(
    path: &std::path::Path,
    task_id: &str,
    snapshot: &crate::ai::driver::runtime_ctx::SubagentProgressSnapshot,
) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "progress evidence path has no parent".to_string())?;
    let elapsed_ms = snapshot.elapsed.as_millis().min(u64::MAX as u128) as u64;
    let timeline = snapshot
        .timeline
        .iter()
        .map(|event| {
            serde_json::json!({
                "sequence": event.sequence,
                "kind": event.kind.as_str(),
                "elapsed_ms": event.elapsed.as_millis().min(u64::MAX as u128) as u64,
                "summary": event.summary,
            })
        })
        .collect::<Vec<_>>();
    let value = serde_json::json!({
        "version": 1,
        "task_id": task_id,
        "source": "progress",
        "phase": snapshot.phase,
        "checkpoint_summary": snapshot.checkpoint_summary,
        "checkpoint_excerpt": snapshot.checkpoint_excerpt,
        "checkpoint_revision": snapshot.checkpoint_revision,
        "elapsed_ms": elapsed_ms,
        "stale_for_ms": snapshot.stale_for.as_millis().min(u64::MAX as u128) as u64,
        "checkpoint_due": snapshot.checkpoint_due,
        "sequence": snapshot.sequence,
        "timeline": timeline,
    });
    let bytes = serde_json::to_vec_pretty(&value).map_err(|err| err.to_string())?;
    let candidate_order = (snapshot.sequence, elapsed_ms);
    let _guard = TASK_PROGRESS_FILE_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if let Ok(existing) = fs::read(path)
        && let Ok(existing) = serde_json::from_slice::<Value>(&existing)
        && existing.get("task_id").and_then(Value::as_str) == Some(task_id)
        && let (Some(sequence), Some(elapsed_ms)) = (
            existing.get("sequence").and_then(Value::as_u64),
            existing.get("elapsed_ms").and_then(Value::as_u64),
        )
        && (sequence, elapsed_ms) >= candidate_order
    {
        return Ok(());
    }

    fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    let temp_path = path.with_extension(format!("json.{}.tmp", Uuid::new_v4().simple()));
    fs::write(&temp_path, bytes).map_err(|err| err.to_string())?;
    if let Err(error) = fs::rename(&temp_path, path) {
        let _ = fs::remove_file(&temp_path);
        return Err(error.to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::driver::runtime_ctx::{
        SUBAGENT_PHASE, new_subagent_progress_slot, publish_subagent_activity_to_slot,
        publish_subagent_checkpoint_summary, subagent_progress_state_snapshot,
    };

    struct ProgressTestGuard {
        previous_kernel: Option<SharedKernel>,
        task_id: String,
    }

    impl Drop for ProgressTestGuard {
        fn drop(&mut self) {
            *lock_unpoisoned(&GLOBAL_OS) = self.previous_kernel.take();
            remove_task_entry(&self.task_id);
            lock_unpoisoned(&TASK_PROGRESS_REGISTRY).remove(&self.task_id);
        }
    }

    #[test]
    fn checkpoint_excerpts_emit_once_and_stop_at_terminal_result() {
        let _env_guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let kernel = crate::ai::driver::new_local_kernel();
        let task_id = format!("checkpoint-test-{}", Uuid::new_v4().simple());
        let _guard = ProgressTestGuard {
            previous_kernel: lock_unpoisoned(&GLOBAL_OS).replace(kernel.clone()),
            task_id: task_id.clone(),
        };
        let (pid, channel, futex) = {
            let mut os = kernel.lock().unwrap();
            let pid = os.begin_foreground("test".into(), "test".into(), 10, 8, None);
            os.set_current_pid(Some(pid));
            let channel = os.channel_create(Some(pid), 1, "checkpoint-test".into());
            let futex = os.futex_create(0, "checkpoint-test".into());
            (pid, channel.raw(), futex)
        };
        insert_task_entry_for_test(
            task_id.clone(),
            AsyncTaskEntry {
                session_id: String::new(),
                owner_pid: pid,
                pid,
                result_channel_id: channel,
                completion_futex_addr: futex,
                description: "checkpoint test".into(),
                agent_name: "build".into(),
                backend: None,
                model: String::new(),
                is_model_auto_selected: false,
                auto_model_fallback: None,
                selection_explanation: String::new(),
                inherit: InheritOptions::default(),
                abort_handle: None,
                cancel_stream: Arc::new(AtomicBool::new(false)),
                started_at: Instant::now(),
                last_progress_notification_at: None,
                last_progress_persisted_at: None,
            },
        );
        let slot = new_subagent_progress_slot();
        lock_unpoisoned(&TASK_PROGRESS_REGISTRY).insert(task_id.clone(), slot.clone());
        SUBAGENT_PHASE.sync_scope(slot.clone(), || {
            publish_subagent_checkpoint_summary("checkpoint", "first saved body");
            let mut locked_output = String::new();
            {
                let mut kernel_lock = kernel.lock().unwrap();
                append_task_progress_snapshots(
                    kernel_lock.as_mut(),
                    &mut locked_output,
                    &[task_id.clone()],
                );
            }
            // A cooperative wait owns the kernel lock while building its parent response;
            // borrowing that kernel must deliver the new revision without recursive locking.
            assert!(locked_output.contains("\n    first saved body"));
            let mut first = String::new();
            append_task_progress_snapshots(
                kernel.lock().unwrap().as_mut(),
                &mut first,
                &[task_id.clone()],
            );
            assert!(!first.contains("first saved body"));
            publish_subagent_activity_to_slot(&slot, "working", "reading next file");
            let mut repeated = String::new();
            append_task_progress_snapshots(
                kernel.lock().unwrap().as_mut(),
                &mut repeated,
                &[task_id.clone()],
            );
            assert!(!repeated.contains("first saved body"));
            assert!(repeated.contains("reading next file"));
            publish_subagent_checkpoint_summary("checkpoint", "second saved body");
            let mut second = String::new();
            append_task_progress_snapshots(
                kernel.lock().unwrap().as_mut(),
                &mut second,
                &[task_id.clone()],
            );
            assert!(second.contains("\n    second saved body"));

            let root = std::env::temp_dir()
                .join(format!("checkpoint-persist-{}", Uuid::new_v4().simple()));
            let snapshot = subagent_progress_state_snapshot(&slot).unwrap();
            let path = root.join("task.json");
            persist_subagent_progress_snapshot_at(&path, &task_id, &snapshot).unwrap();
            assert_eq!(
                serde_json::from_slice::<Value>(&fs::read(&path).unwrap()).unwrap()["source"],
                "progress"
            );
            let persisted: crate::ai::driver::runtime_ctx::SubagentProgressSnapshot =
                serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            assert_eq!(persisted.checkpoint_excerpt, snapshot.checkpoint_excerpt);
            assert_eq!(persisted.checkpoint_revision, 2);
            fs::remove_dir_all(root).unwrap();

            publish_subagent_checkpoint_summary("checkpoint", "undelivered terminal excerpt");
            kernel
                .lock()
                .unwrap()
                .channel_send(
                    Some(pid),
                    ChannelId(channel),
                    serde_json::json!({"status": "completed", "output": "final result"})
                        .to_string(),
                )
                .unwrap();
            let mut terminal = String::new();
            append_task_progress_snapshots(
                kernel.lock().unwrap().as_mut(),
                &mut terminal,
                &[task_id.clone()],
            );
            assert!(!terminal.contains("undelivered terminal excerpt"));
            assert!(
                !current_task_progress(&task_id)
                    .unwrap()
                    .contains("undelivered terminal excerpt")
            );
        });
    }
}
