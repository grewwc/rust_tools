//! Process-local session indices and deferred continuation-note transactions.
//! Continuations wait for terminal publication and for the actual backend run to end.

use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};

use rust_tools::commonw::FastMap;

use super::backend::{BackendId, Continuation, ContinueContext};
use super::registry::lock_unpoisoned;

pub(crate) type SessionSlot = Arc<Mutex<Option<String>>>;

pub(super) enum SessionState {
    Pending(SessionSlot),
    Terminal(String),
}

pub(super) type SessionIndex = Arc<Mutex<FastMap<String, SessionState>>>;

static SESSIONS: LazyLock<Mutex<FastMap<BackendId, SessionIndex>>> =
    LazyLock::new(|| Mutex::new(FastMap::default()));
static ACTIVE_RUNS: LazyLock<Mutex<FastMap<(BackendId, String), SessionSlot>>> =
    LazyLock::new(|| Mutex::new(FastMap::default()));

/// Owned by the actual blocking runner, including time spent queued on the blocking pool.
/// Aborting its async waiter does not make a still-running CLI resumable.
pub(crate) struct BackendRunGuard {
    backend: BackendId,
    task_id: String,
    session_slot: SessionSlot,
}

impl BackendRunGuard {
    pub(crate) fn start(
        backend: BackendId,
        task_id: &str,
        session_slot: SessionSlot,
        resume_session: Option<&str>,
    ) -> Result<Self, String> {
        if let Some(session) = resume_session {
            *lock_unpoisoned(&session_slot) = Some(session.to_string());
        }
        let key = (backend, task_id.to_string());
        let mut active = lock_unpoisoned(&ACTIVE_RUNS);
        if active.contains_key(&key)
            || resume_session.is_some_and(|session| {
                active.iter().any(|((running_backend, _), slot)| {
                    *running_backend == backend
                        && captured_session(slot).as_deref() == Some(session)
                })
            })
        {
            return Err(format!(
                "Backend run for task {task_id} or its session is still active; wait for it to stop before retrying or continuing."
            ));
        }
        active.insert(key, session_slot.clone());
        Ok(Self {
            backend,
            task_id: task_id.to_string(),
            session_slot,
        })
    }
}

impl Drop for BackendRunGuard {
    fn drop(&mut self) {
        let key = (self.backend, self.task_id.clone());
        let mut active = lock_unpoisoned(&ACTIVE_RUNS);
        if active
            .get(&key)
            .is_some_and(|slot| Arc::ptr_eq(slot, &self.session_slot))
        {
            active.remove(&key);
        }
        drop(active);
        crate::ai::driver::notify_scheduler();
    }
}

pub(crate) fn task_backend_run_is_active(backend: BackendId, task_id: &str) -> bool {
    lock_unpoisoned(&ACTIVE_RUNS).contains_key(&(backend, task_id.to_string()))
}

/// A separate index per backend prevents task IDs and raw sessions from crossing
/// backend boundaries. Creating an index performs no I/O or discovery.
pub(super) fn session_index(backend: BackendId) -> SessionIndex {
    lock_unpoisoned(&SESSIONS)
        .entry(backend)
        .or_insert_with(|| Arc::new(Mutex::new(FastMap::default())))
        .clone()
}

pub(crate) fn track_session(backend: BackendId, task_id: &str) -> SessionSlot {
    let slot = Arc::new(Mutex::new(None));
    track_session_slot(backend, task_id, slot.clone());
    slot
}

pub(crate) fn track_session_slot(backend: BackendId, task_id: &str, slot: SessionSlot) {
    lock_unpoisoned(&session_index(backend))
        .insert(task_id.to_string(), SessionState::Pending(slot));
}

pub(crate) fn captured_session(slot: &SessionSlot) -> Option<String> {
    lock_unpoisoned(slot).clone()
}

/// Terminal publication promotes explicit or streamed sessions, even on failure.
/// Discarding a task must instead use `forget_session`, which never promotes it.
pub(crate) fn record_session(
    backend: BackendId,
    task_id: &str,
    session_id: Option<String>,
) -> Option<String> {
    let index = session_index(backend);
    let mut sessions = lock_unpoisoned(&index);
    let streamed_session = match sessions.remove(task_id) {
        Some(SessionState::Pending(slot)) => captured_session(&slot),
        Some(SessionState::Terminal(session)) => Some(session),
        None => None,
    };
    let session = session_id
        .filter(|session| !session.trim().is_empty())
        .or(streamed_session)?;
    sessions.insert(task_id.to_string(), SessionState::Terminal(session.clone()));
    Some(session)
}

pub(crate) fn forget_session(backend: BackendId, task_id: &str) {
    lock_unpoisoned(&session_index(backend)).remove(task_id);
}

pub(crate) fn resolve_continue_target(
    backend: BackendId,
    value: &str,
    accepts_raw_session: fn(&str) -> bool,
    error_template: &str,
) -> Result<String, String> {
    let trimmed = value.trim();
    let index = session_index(backend);
    let sessions = lock_unpoisoned(&index);
    let session = match sessions.get(trimmed) {
        Some(SessionState::Terminal(session)) => session.clone(),
        None if accepts_raw_session(trimmed) => trimmed.to_string(),
        _ => return Err(error_template.replace("{value}", trimmed)),
    };
    let pending = sessions.values().any(|state| match state {
        SessionState::Pending(slot) => captured_session(slot).as_deref() == Some(session.as_str()),
        SessionState::Terminal(_) => false,
    });
    let active = lock_unpoisoned(&ACTIVE_RUNS)
        .iter()
        .any(|((running_backend, _), slot)| {
            *running_backend == backend
                && captured_session(slot).as_deref() == Some(session.as_str())
        });
    if pending || active {
        return Err(format!(
            "Backend session {session} still has an active run. Wait for it to stop and collect its result before backend_continue_from."
        ));
    }
    Ok(session)
}

/// Reading notes does not consume them. The runner commits only after a process
/// successfully spawns; resolution, launch failure, and dropped preparations
/// leave the existing queues intact.
#[derive(Debug, Clone, Default)]
pub(crate) struct DeferredNotes {
    pub(crate) files: Vec<PathBuf>,
}

impl DeferredNotes {
    pub(crate) fn commit_after_spawn(self) {
        Self::consume_after_spawn(&self.files);
    }

    pub(crate) fn consume_after_spawn(files: &[PathBuf]) {
        for file in files {
            let _ = std::fs::remove_file(file);
        }
    }
}

pub(crate) fn prepare_continuation(
    backend: BackendId,
    context: &ContinueContext<'_>,
    accepts_raw_session: fn(&str) -> bool,
    error_template: &str,
) -> Result<Continuation, String> {
    let Some(value) = context
        .continue_from
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(Continuation::fresh(context.prompt));
    };
    let session = resolve_continue_target(backend, value, accepts_raw_session, error_template)?;
    let mut task_ids = {
        let index = session_index(backend);
        let sessions = lock_unpoisoned(&index);
        if matches!(sessions.get(value), Some(SessionState::Terminal(_))) {
            vec![value.to_string()]
        } else {
            sessions
                .iter()
                .filter_map(|(task_id, state)| match state {
                    SessionState::Terminal(recorded) if recorded == &session => {
                        Some(task_id.clone())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        }
    };
    task_ids.sort();
    let mut notes = Vec::new();
    let mut files = Vec::new();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    for task_id in task_ids {
        let file =
            crate::ai::driver::side_note::side_note_file(context.parent_history, Some(&task_id));
        let Ok(content) = std::fs::read_to_string(&file) else {
            continue;
        };
        for line in content
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
        {
            notes.push(
                serde_json::from_str::<crate::ai::driver::side_note::SideNote>(line)
                    .unwrap_or_else(|_| crate::ai::driver::side_note::SideNote {
                        from: "user".to_string(),
                        content: line.to_string(),
                        ts: timestamp,
                        target: Some(task_id.clone()),
                    }),
            );
        }
        files.push(file);
    }
    let prompt = if notes.is_empty() {
        context.prompt.to_string()
    } else {
        notes.sort_by_key(|note| note.ts);
        let mut augmented = String::from("Main-agent follow-ups:\n");
        for note in notes {
            augmented.push_str("- ");
            augmented.push_str(
                &serde_json::to_string(&note.content).expect("side-note string serialization"),
            );
            augmented.push('\n');
        }
        augmented.push('\n');
        augmented.push_str(context.prompt);
        augmented
    };
    Ok(Continuation {
        session_id: Some(session),
        prompt,
        deferred_notes: DeferredNotes { files },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_continuation_waits_for_actual_run_after_terminal_publication() {
        let backend = BackendId::new("active-session-test");
        let slot = track_session(backend, "task-a");
        let guard =
            BackendRunGuard::start(backend, "task-a", slot.clone(), Some("session-a")).unwrap();
        assert!(
            resolve_continue_target(backend, "session-a", |_| true, "invalid {value}").is_err()
        );
        record_session(backend, "task-a", None);
        assert!(task_backend_run_is_active(backend, "task-a"));
        for target in ["task-a", "session-a"] {
            assert!(
                resolve_continue_target(backend, target, |_| true, "invalid {value}")
                    .unwrap_err()
                    .contains("active run")
            );
        }
        assert!(
            BackendRunGuard::start(
                backend,
                "task-b",
                Arc::new(Mutex::new(None)),
                Some("session-a")
            )
            .is_err()
        );
        drop(guard);
        assert!(!task_backend_run_is_active(backend, "task-a"));
        assert_eq!(
            resolve_continue_target(backend, "task-a", |_| true, "invalid {value}").unwrap(),
            "session-a"
        );
        assert_eq!(
            resolve_continue_target(backend, "session-a", |_| true, "invalid {value}").unwrap(),
            "session-a"
        );
        forget_session(backend, "task-a");
    }

    #[test]
    fn backend_continuation_rejects_raw_pending_session_without_run_guard() {
        let backend = BackendId::new("pending-session-test");
        let other = BackendId::new("pending-session-other");
        let slot = track_session(backend, "task-a");
        *lock_unpoisoned(&slot) = Some("session-a".to_string());
        assert!(
            resolve_continue_target(backend, "session-a", |_| true, "invalid {value}").is_err()
        );
        assert_eq!(
            resolve_continue_target(other, "session-a", |_| true, "invalid {value}").unwrap(),
            "session-a"
        );
        record_session(backend, "task-a", None);
        assert_eq!(
            resolve_continue_target(backend, "session-a", |_| true, "invalid {value}").unwrap(),
            "session-a"
        );
        forget_session(backend, "task-a");
    }

    #[test]
    fn backend_sessions_isolate_pending_and_terminal_states() {
        let first = BackendId::new("session-test-first");
        let second = BackendId::new("session-test-second");
        let task = "raw-session";
        let accepts_raw = |value: &str| value == "raw-session";
        let slot = track_session(first, task);
        *lock_unpoisoned(&slot) = Some("first-session".to_string());
        assert!(resolve_continue_target(first, task, accepts_raw, "invalid {value}").is_err());
        assert_eq!(
            resolve_continue_target(second, task, accepts_raw, "invalid {value}").unwrap(),
            task
        );
        assert_eq!(
            record_session(second, task, Some("second-session".to_string())).as_deref(),
            Some("second-session")
        );
        assert!(resolve_continue_target(first, task, accepts_raw, "invalid {value}").is_err());
        assert_eq!(
            record_session(first, task, None).as_deref(),
            Some("first-session")
        );
        assert_eq!(
            resolve_continue_target(first, task, accepts_raw, "invalid {value}").unwrap(),
            "first-session"
        );
        assert_eq!(
            resolve_continue_target(second, task, accepts_raw, "invalid {value}").unwrap(),
            "second-session"
        );
        assert_eq!(
            record_session(first, task, None).as_deref(),
            Some("first-session")
        );
        forget_session(first, task);
        forget_session(second, task);
    }
}
