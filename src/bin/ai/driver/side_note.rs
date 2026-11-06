// side_note.rs — durable side-note file queue with coalesced foreground scheduling
// A new foreground queue notifies the scheduler after its append is synced. At
// the next scheduler boundary, only the parked foreground process is woken;
// further notes coalesce until the queue is drained. Native subagent notes are
// injected before their next model iteration; traecli notes wait for continuation.
// If the turn ends before a queued foreground note is drained, the main input loop
// (run_loop in driver/mod.rs) flushes the queue and runs the note as the next user
// input instead of waiting for the user's next typed message.
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{LazyLock, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::ai::history::{Message, runtime_synthetic_user_message};
use rust_tools::commonw::FastSet;

const SIDE_NOTE_DIR: &str = "side_notes";
const FOREGROUND_FILE: &str = "foreground.jsonl";

/// Existing JSONL queue bytes awaiting their durable commit. Taking a snapshot
/// never removes durable input; acknowledgement removes only this exact prefix.
pub(super) struct PendingSideNotes {
    file: PathBuf,
    content: String,
    notes: Vec<SideNote>,
}

impl PendingSideNotes {
    pub(super) fn notes(&self) -> &[SideNote] {
        &self.notes
    }

    fn acknowledge_locked(&self) -> Result<(), String> {
        let current = fs::read_to_string(&self.file)
            .map_err(|error| format!("read side-note acknowledgement: {error}"))?;
        let remaining = current
            .strip_prefix(&self.content)
            .ok_or_else(|| "side-note queue prefix changed; retained pending input".to_string())?;
        // A suffix may contain guidance appended since the input snapshot. Sync
        // its replacement before rename so a crash cannot lose that new input.
        let tmp = self.file.with_extension(format!(
            "drain.{}.{}.tmp",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        let result = (|| {
            let mut output = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&tmp)
                .map_err(|error| format!("open side-note acknowledgement: {error}"))?;
            output
                .write_all(remaining.as_bytes())
                .map_err(|error| format!("write side-note acknowledgement: {error}"))?;
            output
                .sync_all()
                .map_err(|error| format!("sync side-note acknowledgement: {error}"))?;
            fs::rename(&tmp, &self.file)
                .map_err(|error| format!("replace side-note queue: {error}"))?;
            if let Some(parent) = self.file.parent() {
                fs::File::open(parent)
                    .and_then(|directory| directory.sync_all())
                    .map_err(|error| format!("sync side-note directory: {error}"))?;
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(tmp);
        }
        result
    }

    #[cfg(test)]
    fn acknowledge(&self) -> Result<(), String> {
        let mut woken = WOKEN_FOREGROUND_QUEUES
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let _lock = lock_side_note_queue(&self.file)?;
        self.acknowledge_locked()?;
        woken.remove(&self.file);
        if fs::metadata(&self.file).is_ok_and(|metadata| metadata.len() > 0) {
            super::notify_scheduler();
        }
        Ok(())
    }
}

/// The empty lock file coordinates processes across queue replacement. It stores
/// no session data; the queue retains the existing SideNote JSONL encoding.
fn lock_side_note_queue(file: &Path) -> Result<fs::File, String> {
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(file.with_extension("lock"))
        .map_err(|error| format!("open side-note queue lock: {error}"))?;
    lock.lock()
        .map_err(|error| format!("lock side-note queue: {error}"))?;
    Ok(lock)
}

fn read_pending_side_notes_locked(
    file: PathBuf,
    target: Option<&str>,
) -> Result<Option<PendingSideNotes>, String> {
    let content = match fs::read_to_string(&file) {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("read side-note queue: {error}")),
    };
    let notes: Vec<SideNote> = content
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return None;
            }
            Some(serde_json::from_str(trimmed).unwrap_or_else(|_| SideNote {
                from: "user".to_string(),
                content: trimmed.to_string(),
                ts: now_ts(),
                target: target.map(str::to_string),
            }))
        })
        .collect();
    Ok((!notes.is_empty()).then_some(PendingSideNotes {
        file,
        content,
        notes,
    }))
}

#[cfg(test)]
pub(super) fn read_pending_side_notes(
    history_file: &Path,
    target: Option<&str>,
) -> Result<Option<PendingSideNotes>, String> {
    let file = side_note_file(history_file, target);
    if !file.exists() {
        return Ok(None);
    }
    let _lock = lock_side_note_queue(&file)?;
    read_pending_side_notes_locked(file, target)
}

// The same lock serializes foreground appends, drains, and wake decisions so a
// drain cannot reset coalescing while an append or scheduler wake is in progress.
static WOKEN_FOREGROUND_QUEUES: LazyLock<Mutex<FastSet<PathBuf>>> =
    LazyLock::new(|| Mutex::new(FastSet::default()));

fn is_foreground_target(target: Option<&str>) -> bool {
    matches!(target, None | Some("foreground" | ""))
}

fn should_wake_foreground(target: Option<&str>, already_pending: bool) -> bool {
    is_foreground_target(target) && !already_pending
}

/// Coalesce scheduler wakes until the foreground queue is consumed. Inspecting
/// the durable queue also sees notes written by a serve or attach process, whose
/// in-memory scheduler notification cannot reach this process.
pub(super) fn wake_foreground_for_pending_notes(
    history_file: &Path,
    wake: impl FnOnce() -> bool,
) -> bool {
    let file = side_note_file(history_file, None);
    let mut woken = WOKEN_FOREGROUND_QUEUES
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    if !has_pending_side_notes(history_file, None) {
        woken.remove(&file);
        return false;
    }
    if !should_wake_foreground(None, woken.contains(&file)) || !wake() {
        return false;
    }
    woken.insert(file);
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideNote {
    pub from: String, // "user" | "lead" | task_id
    pub content: String,
    pub ts: u64,
    #[serde(default)]
    pub target: Option<String>, // None=foreground, Some(task_id)
}

fn now_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// history file -> session assets dir: shares the same session assets root as `plan_state` / `checkpoint`.
// `history_file` is always `SessionStore::session_history_file(id)` (i.e. `<sessions_root>/<id>.sqlite`),
// so `parent` is sessions_root and `{stem}.assets` is `SessionStore::session_assets_dir(id)`.
// The old implementation used `"{file_name}.assets"` (including the `.sqlite` suffix), producing
// `"<id>.sqlite.assets"`, which diverged from plan's `"<id>.assets"` and dropped notes. The new
// implementation uniformly uses `file_stem`.
pub(crate) fn assets_dir_for_history(history_file: &Path) -> PathBuf {
    let parent = history_file.parent().unwrap_or_else(|| Path::new("."));
    let stem = history_file
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("default");
    parent.join(format!("{stem}.assets"))
}

pub fn side_note_dir(history_file: &Path) -> PathBuf {
    assets_dir_for_history(history_file).join(SIDE_NOTE_DIR)
}

pub fn side_note_file(history_file: &Path, target: Option<&str>) -> PathBuf {
    let dir = side_note_dir(history_file);
    let name = match target {
        None | Some("foreground") | Some("") => FOREGROUND_FILE.to_string(),
        Some(id) => format!("{}.jsonl", sanitize_task_id(id)),
    };
    dir.join(name)
}

/// Whether the target queue currently contains at least one side-note.
///
/// This is a boundary snapshot: callers use it to decide whether a serve child
/// must stay alive for another model iteration. The subsequent drain remains
/// the authority for the actual notes.
pub fn has_pending_side_notes(history_file: &Path, target: Option<&str>) -> bool {
    fs::metadata(side_note_file(history_file, target))
        .map(|metadata| metadata.len() > 0)
        .unwrap_or(false)
}

/// Read-only count for deferred-backend progress; plain-text queue lines also count.
pub(crate) fn pending_side_note_count(history_file: &Path, target: Option<&str>) -> usize {
    fs::read_to_string(side_note_file(history_file, target))
        .map(|content| {
            content
                .lines()
                .filter(|line| !line.trim().is_empty())
                .count()
        })
        .unwrap_or(0)
}

fn sanitize_task_id(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Durably append one side-note, then notify the scheduler for new foreground guidance.
///
/// `from` should be "user" or the lead agent's task_id / "lead".
/// `target`: None means foreground; Some(task_id) means a specific subagent.
pub fn push_side_note(
    history_file: &Path,
    content: &str,
    from: &str,
    target: Option<&str>,
) -> Result<PathBuf, String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Err("side-note content is empty".to_string());
    }
    if trimmed.chars().count() > 8000 {
        return Err("side-note exceeds 8000 characters (single note limit)".to_string());
    }
    let file = side_note_file(history_file, target);
    if let Some(parent) = file.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create side-note dir: {e}"))?;
    }
    let note = SideNote {
        from: from.to_string(),
        content: trimmed.to_string(),
        ts: now_ts(),
        target: target.map(|s| s.to_string()),
    };
    let line = serde_json::to_string(&note).map_err(|e| format!("encode side-note: {e}"))?;
    let mut wake_state = is_foreground_target(target).then(|| {
        WOKEN_FOREGROUND_QUEUES
            .lock()
            .unwrap_or_else(|err| err.into_inner())
    });
    // Persist the append before publishing any foreground scheduling notification.
    let _queue_lock = lock_side_note_queue(&file)?;
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file)
        .map_err(|e| format!("open side-note file: {e}"))?;
    let already_pending = f
        .metadata()
        .map_err(|e| format!("inspect side-note queue: {e}"))?
        .len()
        > 0;
    writeln!(f, "{}", line).map_err(|e| format!("write side-note: {e}"))?;
    f.sync_all().map_err(|e| format!("sync side-note: {e}"))?;
    if should_wake_foreground(target, already_pending) {
        if let Some(woken) = wake_state.as_mut() {
            woken.remove(&file);
        }
        super::notify_scheduler();
    }
    Ok(file)
}

/// Durably commit input before acknowledging its queue prefix. Append failures
/// and process exits leave the original JSONL available for the next delivery.
pub(super) fn consume_side_notes(
    history_file: &Path,
    target: Option<&str>,
    commit: impl FnOnce(&[SideNote]) -> Result<(), String>,
) -> Result<usize, String> {
    let file = side_note_file(history_file, target);
    if !file.exists() {
        return Ok(0);
    }
    let mut wake_state = is_foreground_target(target).then(|| {
        WOKEN_FOREGROUND_QUEUES
            .lock()
            .unwrap_or_else(|err| err.into_inner())
    });
    let _queue_lock = lock_side_note_queue(&file)?;
    let Some(batch) = read_pending_side_notes_locked(file.clone(), target)? else {
        return Ok(0);
    };
    commit(batch.notes())?;
    batch.acknowledge_locked()?;
    if let Some(woken) = wake_state.as_mut() {
        woken.remove(&file);
    }
    Ok(batch.notes.len())
}

/// The next-input path can handle a local command without appending canonical
/// dialogue. Archive its exact guidance using the existing overflow format
/// before dequeue, then let the ordinary input path retain its original framing.
pub(super) fn drain_side_notes_to_archive(
    history_file: &Path,
    target: Option<&str>,
) -> Result<Vec<SideNote>, String> {
    let mut notes = Vec::new();
    consume_side_notes(history_file, target, |pending| {
        let archived = side_notes_to_messages(pending.to_vec());
        let root = assets_dir_for_history(history_file);
        crate::ai::history::archive_messages_to_overflow(&archived, Some(&root))
            .ok_or_else(|| "side-note archive commit failed; input remains queued".to_string())?;
        notes = pending.to_vec();
        Ok(())
    })?;
    Ok(notes)
}

/// Queue-only fixture helper; production consumers require a durable commit.
#[cfg(test)]
pub fn drain_side_notes(history_file: &Path, target: Option<&str>) -> Vec<SideNote> {
    let mut notes = Vec::new();
    consume_side_notes(history_file, target, |pending| {
        notes = pending.to_vec();
        Ok(())
    })
    .expect("drain side-note fixture");
    notes
}

/// Convert side-notes to runtime-synthetic messages for the LLM request.
/// Compression gives their complete contents the same retention protection as
/// user input while preserving their author identity and turn boundaries.
pub fn side_notes_to_messages(notes: Vec<SideNote>) -> Vec<Message> {
    notes
        .into_iter()
        .map(|n| {
            let header = match n.from.as_str() {
                "user" => "Live guidance (side-note) from user".to_string(),
                other => format!("Live guidance (side-note) from upstream agent `{}`", other),
            };
            let text = format!(
                "[side-note from={} ts={}] {}\n\n{}",
                n.from, n.ts, header, n.content
            );
            runtime_synthetic_user_message(serde_json::Value::String(text))
        })
        .collect()
}

/// The current process's target identifier: None for foreground; subagents learn their task_id
/// via task_local or environment variables.
pub fn current_target_id() -> Option<String> {
    // Prefer task_local (in-process subagents get SUBAGENT_TASK_ID injected by background_dispatch).
    if let Some(id) = crate::ai::driver::runtime_ctx::try_subagent_task_id() {
        let t = id.trim().to_string();
        if !t.is_empty() {
            return Some(t);
        }
    }
    // Fall back to env vars: supports process isolation or external injection; both namings accepted.
    for key in ["AIOS_SUBAGENT_TASK_ID", "SUBAGENT_TASK_ID", "AIOS_TASK_ID"] {
        if let Ok(v) = std::env::var(key) {
            let t = v.trim().to_string();
            if !t.is_empty() {
                return Some(t);
            }
        }
    }
    None
}

#[cfg(test)]
#[path = "side_note_delivery_tests.rs"]
mod delivery_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foreground_side_note_wake_decision_is_targeted_and_coalesced() {
        for target in [None, Some("foreground"), Some("")] {
            assert!(should_wake_foreground(target, false));
            assert!(!should_wake_foreground(target, true));
        }
        assert!(!should_wake_foreground(Some("task_abc"), false));
        assert!(!should_wake_foreground(Some("task_abc"), true));
    }

    #[test]
    fn foreground_side_note_wakes_parked_parent_without_touching_children() {
        use aios_kernel::kernel::{EventId, ProcessState, WaitPolicy};

        use super::super::agent_routing::{
            PARKED_SIDE_NOTE_MESSAGE, wake_pending_foreground_side_note,
        };

        let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        let dir = temp_dir();
        let mut app = super::super::tests::test_app("foreground");
        app.session_history_file = dir.join("sess.sqlite");
        let event = EventId::new(42);
        let (parent, child) = {
            let mut os = app.os.lock().unwrap();
            let parent = os.begin_foreground("foreground".into(), "goal".into(), 10, 8, None);
            let child = os
                .spawn(
                    Some(parent),
                    "child".into(),
                    "child goal".into(),
                    20,
                    4,
                    None,
                    None,
                )
                .unwrap();
            os.wait_on_events(vec![event], WaitPolicy::Any, None)
                .unwrap();
            assert!(os.consume_yield_requested());
            assert_eq!(os.pop_all_ready(1)[0].pid, child);
            (parent, child)
        };

        push_side_note(
            &app.session_history_file,
            "child guidance",
            "lead",
            Some("task_abc"),
        )
        .unwrap();
        assert!(!wake_pending_foreground_side_note(&app));
        assert!(matches!(
            app.os.lock().unwrap().get_process(parent).unwrap().state,
            ProcessState::Waiting { .. }
        ));

        push_side_note(&app.session_history_file, "first guidance", "user", None).unwrap();
        assert!(wake_pending_foreground_side_note(&app));
        {
            let mut os = app.os.lock().unwrap();
            let process = os.get_process(parent).unwrap();
            assert_eq!(process.state, ProcessState::Ready);
            assert_eq!(
                process.mailbox.iter().collect::<Vec<_>>(),
                vec![PARKED_SIDE_NOTE_MESSAGE]
            );
            assert_eq!(os.get_process(child).unwrap().state, ProcessState::Running);
            assert!(os.get_process(child).unwrap().mailbox.is_empty());
            assert!(os.get_process(child).unwrap().result.is_none());
            assert!(!os.event_is_completed(event));
            assert_eq!(os.pop_foreground_ready().unwrap().pid, parent);
            os.wait_on_events(vec![event], WaitPolicy::Any, None)
                .unwrap();
            assert!(os.consume_yield_requested());
        }

        push_side_note(&app.session_history_file, "second guidance", "user", None).unwrap();
        assert!(!wake_pending_foreground_side_note(&app));
        assert_eq!(drain_side_notes(&app.session_history_file, None).len(), 2);
        push_side_note(&app.session_history_file, "new guidance", "user", None).unwrap();
        assert!(wake_pending_foreground_side_note(&app));
        assert_eq!(drain_side_notes(&app.session_history_file, None).len(), 1);
        assert_eq!(
            drain_side_notes(&app.session_history_file, Some("task_abc"))[0].content,
            "child guidance"
        );
        assert!(!app.cancel_stream.load(std::sync::atomic::Ordering::Relaxed));
        let _ = fs::remove_dir_all(dir);
    }

    // No tempfile dependency in this project; reuse the existing std::env::temp_dir + uuid pattern.
    fn temp_dir() -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("side_note_test_{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn push_and_drain_roundtrip() {
        let dir = temp_dir();
        let hist = dir.join("sess.sqlite");
        push_side_note(&hist, "hello", "user", None).unwrap();
        push_side_note(&hist, "world", "lead", None).unwrap();
        assert!(has_pending_side_notes(&hist, None));
        let notes = drain_side_notes(&hist, None);
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0].content, "hello");
        // should be empty after drain
        assert!(drain_side_notes(&hist, None).is_empty());
        assert!(!has_pending_side_notes(&hist, None));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn target_isolation() {
        let dir = temp_dir();
        let hist = dir.join("sess.sqlite");
        push_side_note(&hist, "fg", "user", None).unwrap();
        push_side_note(&hist, "sub", "lead", Some("task_abc")).unwrap();
        assert_eq!(drain_side_notes(&hist, None).len(), 1);
        assert_eq!(drain_side_notes(&hist, Some("task_abc")).len(), 1);
        assert!(drain_side_notes(&hist, Some("task_abc")).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
