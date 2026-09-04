// side_note.rs — real-time side-note file queue + in-memory notification
// Written by the user or a lead agent during a turn; the running turn drains and
// injects them into the LLM context before the next iteration.
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::ai::history::{Message, runtime_synthetic_user_message};

const SIDE_NOTE_DIR: &str = "side_notes";
const FOREGROUND_FILE: &str = "foreground.jsonl";

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

/// Append one side-note (file append; single-line append atomicity is guaranteed by the OS).
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
    // Append-only write; single-line append atomicity avoids concurrent truncation.
    use std::io::Write;
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file)
        .map_err(|e| format!("open side-note file: {e}"))?;
    writeln!(f, "{}", line).map_err(|e| format!("write side-note: {e}"))?;
    Ok(file)
}

/// Drain and clear all pending notes for the target queue. Called by the caller at iteration
/// boundaries; new notes are injected immediately.
/// Drain uses an atomic rename: push appends, while the drain side renames the old file to a temp
/// file and reads it, avoiding notes pushed during the "read-then-truncate" window being dropped.
pub fn drain_side_notes(history_file: &Path, target: Option<&str>) -> Vec<SideNote> {
    let file = side_note_file(history_file, target);
    // Atomic drain: try renaming the queue file to a temp file; failure means nothing is pending.
    let tmp = file.with_extension(format!("drain.{}.tmp", std::process::id()));
    let renamed = fs::rename(&file, &tmp).is_ok();
    let content = if renamed {
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
            return Vec::new();
        }
        // Fallback path: rename can fail on cross-device moves etc.; degrade to atomic truncation
        // (a tiny race remains, but only triggered on cross-device).
        match fs::read_to_string(&file) {
            Ok(c) => {
                // Best-effort atomic truncation: write an empty temp file first, then rename over it.
                let empty_tmp = file.with_extension(format!("empty.{}.tmp", std::process::id()));
                let _ = fs::write(&empty_tmp, "");
                let _ = fs::rename(&empty_tmp, &file);
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

/// Convenience: drain and directly return Messages (empty Vec when nothing pending).
pub fn drain_side_notes_as_messages(history_file: &Path, target: Option<&str>) -> Vec<Message> {
    let notes = drain_side_notes(history_file, target);
    if notes.is_empty() {
        return Vec::new();
    }
    side_notes_to_messages(notes)
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

/// Called by the turn loop before each model request: drains pending notes for the current target
/// and injects them into messages. Returns the number injected, so the caller can decide whether
/// to print a hint.
pub fn poll_and_inject(history_file: &Path, messages: &mut Vec<Message>) -> usize {
    let target = current_target_id();
    let target_ref = target.as_deref();
    let injected = drain_side_notes_as_messages(history_file, target_ref);
    let n = injected.len();
    if n > 0 {
        messages.extend(injected);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let notes = drain_side_notes(&hist, None);
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0].content, "hello");
        // should be empty after drain
        assert!(drain_side_notes(&hist, None).is_empty());
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
