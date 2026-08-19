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

/// 把本轮原始消息连同生成它们的模型写入 canonical history。
/// `source_model` 只作为 reasoning 请求投影的来源证明，不会改写消息正文。
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
    fs::write(path, serialize_history_messages(messages))
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
    // 所有 history 删除入口最终都会走到这里；同步回收进程内 revision 缓存，
    // 避免按 session/sub-agent 唯一路径持续累积陈旧条目。
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
    matches!(
        path.extension().and_then(|s| s.to_str()),
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

/// 唤醒笔记去重（方案1）：同一进程、同一批 task_ids 的 TASK_WAIT_TIMEOUT "仍在等待"
/// 唤醒笔记只保留最新一条。调用方在准备追加一条内省笔记时调用本函数：
///
/// - 该笔记不是"仍在等待"唤醒笔记（普通问题、真实结果唤醒、非 internal_note）：
///   返回 `Ok(false)`，不动文件，调用方照常追加；
/// - 是上述唤醒笔记：删除历史尾部 `WAKE_NOTE_DEDUP_SCAN` 条消息内所有同身份
///   (pid, task_ids) 的旧等待笔记，返回 `Ok(true)`；随后调用方把最新一条追加到尾部，
///   从而整条等待链在历史中只保留最新进度快照。
///
/// 读取/重建失败按 best-effort 返回 `Ok(false)`，绝不阻塞正常追加。
pub(in crate::ai) fn coalesce_repeated_wait_wake_notes(
    path: &Path,
    note: &Message,
) -> io::Result<bool> {
    if is_sqlite_path(path) {
        return sqlite::coalesce_repeated_wait_wake_notes_sqlite(path, note);
    }
    // fast path：非"仍在等待"唤醒笔记时不做任何 IO
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
        fs::write(path, rebuilt.as_bytes()).map_err(|err| {
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

        // 历史尾部已有 2 条同身份（pid=6, 同一批 task_ids）旧等待笔记 + 1 条不同身份（pid=7）。
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
        // 调用方随后把最新一条追加到尾部。
        append_history_messages(&path, &[latest]).unwrap();

        let messages = parse_history_blob(&std::fs::read_to_string(&path).unwrap());
        let notes: Vec<_> = messages
            .into_iter()
            .filter(|m| m.role == ROLE_INTERNAL_NOTE)
            .collect();
        // pid=6 的两条旧笔记被折叠，只剩 pid=7 + 最新 pid=6。
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

        // 窗口语义钉死：扫描历史尾部 WAKE_NOTE_DEDUP_SCAN 条消息（不限角色），
        // 而不是“最近 WAKE_NOTE_DEDUP_SCAN 条 internal_note”。
        // 旧等待笔记在第 1 条，其后跟 WAKE_NOTE_DEDUP_SCAN+1 条 user 消息，故其在窗口外。
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
        // 窗口外的旧等待笔记保留，未被误删。
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

        // 同一 pid 但不同 task 集合：身份不同，不去重。
        let other = msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_z"], "checkpoint-x"));
        assert!(!coalesce_repeated_wait_wake_notes(&path, &other).unwrap());

        // 非 internal_note 消息：fast path 不做任何 IO。
        assert!(!coalesce_repeated_wait_wake_notes(&path, &msg("user", "hello")).unwrap());

        // 真实结果唤醒（parse 为 None）：不去重。
        let result_wake = msg(
            ROLE_INTERNAL_NOTE,
            "[Process 6 Woke Up] Original goal: g\nNew mailbox messages:\n[EVENT_WAKE]\nready\n\nWake-up handling rules:\n- rule\n\nResume execution based on the goal and these messages.",
        );
        assert!(!coalesce_repeated_wait_wake_notes(&path, &result_wake).unwrap());

        // 历史文件不存在：best-effort 返回 false，不报错。
        let missing = dir.join("missing.txt");
        let wait_note = msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a"], "c"));
        assert!(!coalesce_repeated_wait_wake_notes(&missing, &wait_note).unwrap());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
