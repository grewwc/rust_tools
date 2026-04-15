use std::{
    fs::{self},
    io,
    path::{Path, PathBuf},
};

use crate::commonw::utils::open_file_for_append;

use super::{
    compress,
    sqlite,
    types::{COLON, Message, NEWLINE},
};

pub(in crate::ai) fn build_message_arr(
    history_count: usize,
    history_file: &PathBuf,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    if is_sqlite_path(history_file) {
        return sqlite::build_message_arr_sqlite(history_count, history_file.as_path());
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

    let compacted = compress::compact_persisted_history(parsed_messages.clone());
    if compacted != parsed_messages {
        fs::write(history_file, serialize_history_messages(&compacted))?;
    }
    let messages = compacted;

    if history_count >= messages.len() {
        return Ok(messages);
    }
    Ok(messages[messages.len() - history_count..].to_vec())
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

pub(in crate::ai) fn append_history_messages_uncompacted(
    path: &Path,
    messages: &[Message],
) -> io::Result<()> {
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
        return sqlite::append_history_sqlite_uncompacted(path, messages.to_vec());
    }
    append_history_blob(path, &blob)
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
    })
}
