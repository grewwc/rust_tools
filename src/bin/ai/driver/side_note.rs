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
    path::{Path, PathBuf},
    sync::{LazyLock, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::ai::history::{Message, runtime_synthetic_user_message};
use rust_tools::commonw::FastSet;

const SIDE_NOTE_DIR: &str = "side_notes";
const FOREGROUND_FILE: &str = "foreground.jsonl";

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
    use std::io::Write;
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

/// Drain and clear all pending notes for the target queue. Called by the caller at iteration
/// boundaries; new notes are injected immediately.
/// Drain uses an atomic rename: push appends, while the drain side renames the old file to a temp
/// file and reads it, avoiding notes pushed during the "read-then-truncate" window being dropped.
pub fn drain_side_notes(history_file: &Path, target: Option<&str>) -> Vec<SideNote> {
    let file = side_note_file(history_file, target);
    let mut wake_state = is_foreground_target(target).then(|| {
        WOKEN_FOREGROUND_QUEUES
            .lock()
            .unwrap_or_else(|err| err.into_inner())
    });
    // Atomic drain: try renaming the queue file to a temp file; failure means nothing is pending.
    let tmp = file.with_extension(format!("drain.{}.tmp", std::process::id()));
    let renamed = fs::rename(&file, &tmp).is_ok();
    let content = if renamed {
        if let Some(woken) = wake_state.as_mut() {
            woken.remove(&file);
        }
        match fs::read_to_string(&tmp) {
            Ok(c) => {
                let _ = fs::remove_file(&tmp);
                c
            }
            Err(_) => {
                let _ = fs::remove_file(&tmp);
                return Vec::new();
            }
        }
    } else {
        if !file.exists() {
            if let Some(woken) = wake_state.as_mut() {
                woken.remove(&file);
            }
            return Vec::new();
        }
        // Fallback path: rename can fail on cross-device moves etc.; degrade to atomic truncation
        // (a tiny race remains, but only triggered on cross-device).
        match fs::read_to_string(&file) {
            Ok(c) => {
                // Best-effort atomic truncation: write an empty temp file first, then rename over it.
                let empty_tmp = file.with_extension(format!("empty.{}.tmp", std::process::id()));
                let _ = fs::write(&empty_tmp, "");
                if fs::rename(&empty_tmp, &file).is_ok()
                    && let Some(woken) = wake_state.as_mut()
                {
                    woken.remove(&file);
                }
                c
            }
            Err(_) => return Vec::new(),
        }
    };
    let mut out = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(note) = serde_json::from_str::<SideNote>(trimmed) {
            out.push(note);
        } else {
            // Backward compatibility: a plain-text line is a single note.
            out.push(SideNote {
                from: "user".to_string(),
                content: trimmed.to_string(),
                ts: now_ts(),
                target: target.map(|s| s.to_string()),
            });
        }
    }
    out
}

/// Convert SideNote into user messages that can be pushed straight into LLM `messages`.
/// Uses `runtime_synthetic_user_message` to avoid being misjudged as a real user turn boundary.
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
