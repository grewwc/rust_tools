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
    TASK_PROGRESS_REGISTRY
        .lock()
        .ok()?
        .get_ref(&task_id.to_string())
        .cloned()
}

pub(super) fn current_task_progress(task_id: &str) -> Option<String> {
    let slot = task_progress_slot(task_id)?;
    let value = crate::ai::driver::runtime_ctx::subagent_progress_snapshot(&slot)?;
    (!value.is_empty()).then_some(value)
}

pub(super) fn append_task_progress_snapshots(output: &mut String, task_ids: &[String]) {
    let snapshots = task_ids
        .iter()
        .filter_map(|task_id| {
            current_task_progress(task_id).map(|progress| format!("- {task_id}: {progress}"))
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
        let mut registry = TASK_REGISTRY.lock().unwrap();
        let Some(entry) = registry.get_mut(&task_id.to_string()) else {
            return;
        };
        let persist = kind
            == crate::ai::driver::runtime_ctx::SubagentProgressEventKind::Checkpoint
            || entry.last_progress_persisted_at.is_none_or(|last| {
                now.saturating_duration_since(last) >= SUBAGENT_PROGRESS_PERSIST_INTERVAL
            });
        let notify = kind
            == crate::ai::driver::runtime_ctx::SubagentProgressEventKind::Checkpoint
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
        let mut registry = TASK_REGISTRY.lock().unwrap();
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
        "phase": snapshot.phase,
        "checkpoint_summary": snapshot.checkpoint_summary,
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
