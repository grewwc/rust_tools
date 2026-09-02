use std::{
    fs::{self},
    io,
    path::Path,
};

use crate::commonw::utils::open_file_for_append;

use super::{
    sqlite,
    types::{COLON, Message, NEWLINE, ROLE_INTERNAL_NOTE, WAKE_NOTE_DEDUP_SCAN},
};

pub(in crate::ai) fn build_message_arr(
    history_count: usize,
    history_file: &Path,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    if is_sqlite_path(history_file) {
        return sqlite::build_message_arr_sqlite(history_count, history_file);
    }
    let history = match fs::read_to_string(history_file) {
        Ok(history) => history,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.into()),
    };

    // Split and filter out empty trailing line (caused by trailing newline)
    let mut lines: Vec<&str> = history.split(NEWLINE).collect();
    if lines.last().map_or(false, |s| s.is_empty()) {
        lines.pop();
    }
    let mut parsed_messages = Vec::new();
    for line in &lines {
        if let Some(message) = parse_history_line(line) {
            parsed_messages.push(message);
        }
    }

    if history_count >= parsed_messages.len() {
        return Ok(parsed_messages);
    }
    Ok(parsed_messages[parsed_messages.len() - history_count..].to_vec())
}

pub(in crate::ai) fn append_history(path: &Path, content: &str) -> io::Result<()> {
    if is_sqlite_path(path) {
        return sqlite::append_history_sqlite(path, parse_history_blob(content));
    }
    append_history_blob(path, content)
}

pub(in crate::ai) fn append_history_messages(path: &Path, messages: &[Message]) -> io::Result<()> {
    if messages.is_empty() {
        return Ok(());
    }

    let newline = NEWLINE.to_string();
    let mut records = Vec::with_capacity(messages.len());
    for message in messages {
        let record = serde_json::to_string(message).map_err(|e| io::Error::other(e.to_string()))?;
        records.push(record);
    }
    let blob = format!("{}{}", records.join(&newline), newline);

    if is_sqlite_path(path) {
        return sqlite::append_history_sqlite(path, messages.to_vec());
    }
    append_history(path, &blob)
}

/// Writes this round's raw messages together with the model that produced them to the
/// canonical history.
/// `source_model` serves only as provenance for reasoning-request projections; it never
/// rewrites message bodies.
pub(in crate::ai) fn append_history_messages_for_model(
    path: &Path,
    messages: &[Message],
    source_model: &str,
) -> io::Result<()> {
    if messages.is_empty() {
        return Ok(());
    }
    if is_sqlite_path(path) {
        return sqlite::append_history_sqlite_for_model(
            path,
            messages.to_vec(),
            Some(source_model),
        );
    }
    append_history_messages(path, messages)
}

pub(in crate::ai) fn replace_history_messages(path: &Path, messages: &[Message]) -> io::Result<()> {
    if is_sqlite_path(path) {
        return sqlite::replace_all_messages_sqlite(path, messages);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    atomic_write_history(path, serialize_history_messages(messages).as_bytes())
}

pub(in crate::ai) fn truncate_history_messages(path: &Path, keep: usize) -> io::Result<()> {
    if is_sqlite_path(path) {
        return sqlite::truncate_messages_sqlite(path, keep);
    }
    let mut messages = match fs::read_to_string(path) {
        Ok(content) => parse_history_blob(&content),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    messages.truncate(keep);
    replace_history_messages(path, &messages)
}

fn append_history_blob(path: &Path, content: &str) -> io::Result<()> {
    let mut file = open_file_for_append(path, 0o664)?;
    use std::io::Write;
    file.write_all(content.as_bytes())
}

fn serialize_history_messages(messages: &[Message]) -> String {
    let newline = NEWLINE.to_string();
    let mut records = Vec::with_capacity(messages.len());
    for message in messages {
        if let Ok(record) = serde_json::to_string(message) {
            records.push(record);
        }
    }
    if records.is_empty() {
        String::new()
    } else {
        format!("{}{}", records.join(&newline), newline)
    }
}

pub(in crate::ai) fn serialize_history_messages_for_storage(messages: &[Message]) -> String {
    serialize_history_messages(messages)
}

pub(in crate::ai) fn delete_history_artifacts(path: &Path) -> io::Result<()> {
    fn remove_one(path: &Path) -> io::Result<()> {
        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err),
        }
    }

    remove_one(path)?;

    let base = path.to_string_lossy().to_string();
    remove_one(Path::new(&format!("{base}-wal")))?;
    remove_one(Path::new(&format!("{base}-shm")))?;
    remove_one(Path::new(&format!("{base}-journal")))?;
    // Every history-deletion entry point eventually reaches here; also reclaim the in-process
    // revision cache so stale entries do not keep accumulating under per-session/sub-agent paths.
    super::sqlite::remove_history_revision_cache_entry(path);
    Ok(())
}

pub(in crate::ai) fn delete_assets_dir(path: &Path) -> io::Result<()> {
    match fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

pub(in crate::ai) fn is_sqlite_path(path: &Path) -> bool {
    // Hard-timeout recovery preserves a child history by appending this suffix to the original
    // filename. Classify the preserved artifact by its original extension; otherwise a SQLite
    // database such as `child.sqlite.timeout-preserved` is read as UTF-8 text and recovery loses
    // all of the subagent's recorded progress.
    let path_text = path.to_string_lossy();
    let logical_path = path_text
        .strip_suffix(".timeout-preserved")
        .unwrap_or(&path_text);
    matches!(
        Path::new(logical_path).extension().and_then(|s| s.to_str()),
        Some("sqlite") | Some("db")
    )
}

pub(in crate::ai) fn parse_history_blob(content: &str) -> Vec<Message> {
    let mut out = Vec::new();
    for line in content.split(NEWLINE) {
        if let Some(message) = parse_history_line(line) {
            out.push(message);
        }
    }
    out
}

fn parse_history_line(line: &str) -> Option<Message> {
    if line.is_empty() {
        return None;
    }
    if let Ok(message) = serde_json::from_str::<Message>(line) {
        return Some(message);
    }

    let last_colon = line.rfind(COLON)?;
    if last_colon == 0 || last_colon + COLON.len_utf8() >= line.len() {
        return None;
    }
    let role = &line[..last_colon];
    if !matches!(role, "user" | "assistant" | "system" | "tool") {
        return None;
    }
    let content = &line[last_colon + COLON.len_utf8()..];
    Some(Message {
        role: role.to_string(),
        content: serde_json::Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    })
}

/// Wake-note dedup (plan 1): for the same process and the same batch of task_ids, only the
/// latest TASK_WAIT_TIMEOUT "still waiting" wake note is kept. Callers invoke this function
/// right before appending an introspection note:
///
/// - The note is not a "still waiting" wake note (a normal question, a real-result wake, or a
///   non-internal_note): returns `Ok(false)` without touching the file, and the caller appends
///   as usual;
/// - It is such a wake note: deletes every old waiting note with the same identity
///   (pid, task_ids) within the last `WAKE_NOTE_DEDUP_SCAN` messages, returns `Ok(true)`;
///   the caller then appends the latest one at the tail, so the whole waiting chain keeps only
///   the newest progress snapshot in history.
///
/// Read/rebuild failures return `Ok(false)` on a best-effort basis and never block normal appends.
pub(in crate::ai) fn coalesce_repeated_wait_wake_notes(
    path: &Path,
    note: &Message,
) -> io::Result<bool> {
    if is_sqlite_path(path) {
        return sqlite::coalesce_repeated_wait_wake_notes_sqlite(path, note);
    }
    // fast path: no I/O at all when this is not a "still waiting" wake note
    if note.role != ROLE_INTERNAL_NOTE {
        return Ok(false);
    }
    let Some(text) = note.content.as_str() else {
        return Ok(false);
    };
    let Some(identity) = super::types::parse_still_waiting_wake_identity(text) else {
        return Ok(false);
    };

    let data = match fs::read_to_string(path) {
        Ok(data) => data,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err),
    };
    let mut lines: Vec<&str> = data.split(NEWLINE).collect();
    if lines.last().map_or(false, |s| s.is_empty()) {
        lines.pop();
    }
    let scan_from = lines.len().saturating_sub(WAKE_NOTE_DEDUP_SCAN);
    let mut removed = 0usize;
    let mut rebuilt = String::with_capacity(data.len());
    for (i, line) in lines.iter().enumerate() {
        if i >= scan_from {
            if let Some(prev) = parse_history_line(line) {
                let is_dup = prev.role == ROLE_INTERNAL_NOTE
                    && prev
                        .content
                        .as_str()
                        .and_then(super::types::parse_still_waiting_wake_identity)
                        .as_ref()
                        == Some(&identity);
                if is_dup {
                    removed += 1;
                    continue;
                }
            }
        }
        rebuilt.push_str(line);
        rebuilt.push(NEWLINE);
    }
    if removed > 0 {
        atomic_write_history(path, rebuilt.as_bytes()).map_err(|err| {
            eprintln!(
                "[history] coalesce_repeated_wait_wake_notes: rewrite {} failed: {err}",
                path.display()
            );
            err
        })?;
        return Ok(true);
    }
    Ok(false)
}

/// Atomically replace the whole history file: write to a unique temp file in the same directory,
/// fsync it, then rename over the target. A crash at any point leaves either the previous complete
/// history or no file — never a truncated canonical history (which `fs::write`'s in-place truncate
/// could produce). Every text-backend full rewrite (replace/truncate/compaction/wake-note
/// coalescing) funnels through here; the SQLite backend needs no equivalent because its
/// replace/truncate run inside a transaction and are protected by the state lock.
pub(in crate::ai) fn atomic_write_history(path: &Path, contents: &[u8]) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("history");
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    // pid + timestamp keeps temp names unique across concurrent writers in the same session, so
    // concurrent rewrites never clobber each other's temp file; the last rename wins with a complete file.
    let tmp = parent.join(format!(".{}.tmp.{}.{}", file_name, pid, nanos));
    let result = (|| {
        use std::io::Write;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp)?;
        file.write_all(contents)?;
        file.flush()?;
        // fsync before rename so the rename never publishes unflushed data.
        file.sync_all()?;
        // rename replaces the inode, which would silently reset the file's permission bits
        // (the append path creates history with 0o664). Carry the previous mode over when the
        // target already exists; a first-ever write keeps the temp file's default mode.
        if let Ok(meta) = fs::metadata(path) {
            let _ = fs::set_permissions(&tmp, meta.permissions());
        }
        fs::rename(&tmp, path)
    })();
    if result.is_err() {
        // Best-effort cleanup of the temp file on failure so no partial artifact is left behind.
        let _ = fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn msg(role: &str, text: &str) -> Message {
        Message {
            role: role.to_string(),
            content: Value::String(text.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn wake_note_text(pid: u64, ids: &[&str], checkpoint: &str) -> String {
        format!(
            "[Process {pid} Woke Up] Original goal: test goal\nNew mailbox messages:\n[TASK_WAIT_TIMEOUT]\nWall-clock task_wait budget elapsed after 30s. Re-call `task_wait` with the same task_ids to collect any ready results and receive the budget-elapsed status. task_ids=[{}]\nProgress: {checkpoint}\n\nWake-up handling rules:\n- rule\n\nResume execution based on the goal and these messages.",
            ids.join(", ")
        )
    }

    #[test]
    fn preserved_sqlite_history_remains_readable_as_sqlite() {
        let dir = std::env::temp_dir().join(format!(
            "preserved_sqlite_history_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let original = dir.join("child.sqlite");
        append_history_messages(&original, &[msg("assistant", "verified partial finding")])
            .unwrap();

        let preserved = crate::ai::history::preserve_subagent_history(&original).unwrap();
        assert!(is_sqlite_path(&preserved));
        let messages = build_message_arr(10, &preserved).unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(
            messages[0].content.as_str(),
            Some("verified partial finding")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wait_wake_notes_coalesce_keeps_latest_in_blob_history() {
        let dir = std::env::temp_dir().join(format!(
            "blob_wake_dedup_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.txt");

        // The history tail already holds 2 old waiting notes with the same identity (pid=6, same
        // task_ids batch) plus 1 with a different identity (pid=7).
        let history = serialize_history_messages_for_storage(&[
            msg("user", "goal"),
            msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a", "task_b"], "checkpoint-1")),
            msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a", "task_b"], "checkpoint-2")),
            msg(ROLE_INTERNAL_NOTE, &wake_note_text(7, &["task_x"], "checkpoint-3")),
        ]);
        std::fs::write(&path, history).unwrap();

        let latest =
            msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a", "task_b"], "checkpoint-4"));
        assert!(coalesce_repeated_wait_wake_notes(&path, &latest).unwrap());
        // The caller then appends the latest note at the tail.
        append_history_messages(&path, &[latest]).unwrap();

        let messages = parse_history_blob(&std::fs::read_to_string(&path).unwrap());
        let notes: Vec<_> = messages
            .into_iter()
            .filter(|m| m.role == ROLE_INTERNAL_NOTE)
            .collect();
        // The two old pid=6 notes were coalesced; only pid=7 plus the latest pid=6 remain.
        assert_eq!(notes.len(), 2);
        assert!(notes[0].content.as_str().unwrap().contains("checkpoint-3"));
        assert!(notes[1].content.as_str().unwrap().contains("checkpoint-4"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wait_wake_coalesce_blob_window_is_last_total_messages() {
        let dir = std::env::temp_dir().join(format!(
            "blob_wake_dedup_window_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.txt");

        // Window semantics pinned: scan the last WAKE_NOTE_DEDUP_SCAN messages in the history tail
        // (any role), not the "most recent WAKE_NOTE_DEDUP_SCAN internal_notes".
        // The old waiting note is the 1st message, followed by WAKE_NOTE_DEDUP_SCAN+1 user messages,
        // so it falls outside the window.
        let mut history = vec![msg(
            ROLE_INTERNAL_NOTE,
            &wake_note_text(6, &["task_a"], "checkpoint-old"),
        )];
        history.extend(
            (0..WAKE_NOTE_DEDUP_SCAN as usize + 1).map(|i| msg("user", &format!("filler {i}"))),
        );
        std::fs::write(&path, serialize_history_messages_for_storage(&history)).unwrap();

        let latest =
            msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a"], "checkpoint-new"));
        assert!(!coalesce_repeated_wait_wake_notes(&path, &latest).unwrap());

        let messages = parse_history_blob(&std::fs::read_to_string(&path).unwrap());
        let notes: Vec<_> = messages
            .into_iter()
            .filter(|m| m.role == ROLE_INTERNAL_NOTE)
            .collect();
        // The old waiting note outside the window is kept, not wrongly deleted.
        assert_eq!(notes.len(), 1);
        assert!(notes[0].content.as_str().unwrap().contains("checkpoint-old"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wait_wake_coalesce_blob_is_noop_when_nothing_matches() {
        let dir = std::env::temp_dir().join(format!(
            "blob_wake_dedup_noop_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.txt");
        std::fs::write(
            &path,
            serialize_history_messages_for_storage(&[
                msg("user", "goal"),
                msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a"], "checkpoint-1")),
            ]),
        )
        .unwrap();

        // Same pid but a different task set: different identity, so no dedup.
        let other = msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_z"], "checkpoint-x"));
        assert!(!coalesce_repeated_wait_wake_notes(&path, &other).unwrap());

        // Non-internal_note message: the fast path performs no I/O.
        assert!(!coalesce_repeated_wait_wake_notes(&path, &msg("user", "hello")).unwrap());

        // Real-result wake (parsed as None): no dedup.
        let result_wake = msg(
            ROLE_INTERNAL_NOTE,
            "[Process 6 Woke Up] Original goal: g\nNew mailbox messages:\n[EVENT_WAKE]\nready\n\nWake-up handling rules:\n- rule\n\nResume execution based on the goal and these messages.",
        );
        assert!(!coalesce_repeated_wait_wake_notes(&path, &result_wake).unwrap());

        // History file missing: best-effort returns false without erroring.
        let missing = dir.join("missing.txt");
        let wait_note = msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a"], "c"));
        assert!(!coalesce_repeated_wait_wake_notes(&missing, &wait_note).unwrap());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_backend_rewrites_are_atomic_with_no_temp_leftovers() {
        let dir = std::env::temp_dir().join(format!(
            "blob_atomic_rewrite_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.txt");
        std::fs::write(&path, serialize_history_messages_for_storage(&[msg("user", "first")]))
            .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        }

        // Both text-backend rewrite entry points must replace the file completely and leave no
        // intermediate `.tmp.` artifact behind (the atomic-write contract for canonical history).
        replace_history_messages(
            &path,
            &[msg("user", "second"), msg("assistant", "reply")],
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o640,
                "atomic rewrite must preserve the history file's permission bits"
            );
        }
        let messages = parse_history_blob(&std::fs::read_to_string(&path).unwrap());
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content.as_str(), Some("second"));

        truncate_history_messages(&path, 1).unwrap();
        let messages = parse_history_blob(&std::fs::read_to_string(&path).unwrap());
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content.as_str(), Some("second"));

        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| {
                let name = entry.ok()?.file_name().to_string_lossy().into_owned();
                name.contains(".tmp.").then_some(name)
            })
            .collect();
        assert!(
            leftovers.is_empty(),
            "atomic rewrites must not leave temp files: {leftovers:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
