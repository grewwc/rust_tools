use std::{
    fs::{self},
    fs::File,
    io,
    path::Path,
    path::PathBuf,
};

use chrono::{DateTime, Local};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::common::utils::open_file_for_append;

use super::types::ToolCall;

pub(super) const MAX_HISTORY_LINES: usize = 200;
pub(super) const COLON: char = '\0';
pub(super) const NEWLINE: char = '\x01';

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(super) struct Message {
    pub(super) role: String,
    pub(super) content: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_call_id: Option<String>,
}

pub(super) fn compress_messages_for_context(
    messages: Vec<Message>,
    max_chars: usize,
    keep_last: usize,
    summary_max_chars: usize,
) -> Vec<Message> {
    if max_chars == 0 || messages.is_empty() {
        return messages;
    }

    let keep_last = keep_last.min(messages.len());
    if keep_last == 0 {
        return shrink_messages_to_fit(messages, max_chars);
    }

    let split_at = messages.len().saturating_sub(keep_last);
    let (older, recent) = messages.split_at(split_at);
    if older.is_empty() {
        return shrink_messages_to_fit(recent.to_vec(), max_chars);
    }

    let mut out = Vec::new();
    if summary_max_chars > 0 {
        let summary = build_summary_text(older, summary_max_chars);
        if !summary.trim().is_empty() {
            out.push(Message {
                role: "system".to_string(),
                content: Value::String(format!(
                    "对话摘要（自动压缩，以下为早期对话要点）：\n{summary}"
                )),
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }
    out.extend_from_slice(recent);
    shrink_messages_to_fit(out, max_chars)
}

pub(super) fn build_message_arr(
    history_count: usize,
    history_file: &PathBuf,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    if is_sqlite_path(history_file) {
        return build_message_arr_sqlite(history_count, history_file.as_path());
    }
    let history = match fs::read_to_string(history_file) {
        Ok(history) => history,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.into()),
    };

    let newline = NEWLINE.to_string();
    let lines: Vec<&str> = history.split(NEWLINE).collect();
    let mut messages = Vec::new();

    for line in &lines {
        if let Some(message) = parse_history_line(line) {
            messages.push(message);
        }
    }

    if lines.len() > MAX_HISTORY_LINES {
        let start = lines.len() - MAX_HISTORY_LINES;
        let trimmed = lines[start..].join(&newline);
        fs::write(history_file, trimmed)?;
    }

    if history_count >= messages.len() {
        return Ok(messages);
    }
    Ok(messages[messages.len() - history_count..].to_vec())
}

pub(super) fn append_history(path: &Path, content: &str) -> io::Result<()> {
    if is_sqlite_path(path) {
        return append_history_sqlite(path, content);
    }
    append_history_blob(path, content)
}

pub(super) fn append_history_messages(path: &Path, messages: &[Message]) -> io::Result<()> {
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
        return append_history_sqlite(path, &blob);
    }
    append_history(path, &blob)
}

fn append_history_blob(path: &Path, content: &str) -> io::Result<()> {
    let mut file = open_file_for_append(path, 0o664)?;
    use std::io::Write;
    file.write_all(content.as_bytes())
}

pub(super) fn delete_history_artifacts(path: &Path) -> io::Result<()> {
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

fn delete_assets_dir(path: &Path) -> io::Result<()> {
    match fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

fn is_sqlite_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|s| s.to_str()),
        Some("sqlite") | Some("db")
    )
}

fn parse_history_blob(content: &str) -> Vec<Message> {
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
        content: Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: None,
    })
}

fn shrink_messages_to_fit(mut messages: Vec<Message>, max_chars: usize) -> Vec<Message> {
    if max_chars == 0 {
        return messages;
    }

    if messages.is_empty() {
        return Vec::new();
    }

    // 如果总长度已经在限制内，直接返回
    if messages_total_chars(&messages) <= max_chars {
        return messages;
    }

    // 从前往后移除消息，直到总长度满足限制（至少保留最后一条）
    let mut start = 0usize;
    while start + 1 < messages.len() && messages_total_chars(&messages[start..]) > max_chars {
        start += 1;
    }
    if start > 0 {
        messages = messages[start..].to_vec();
    }

    // 如果仍然超出限制，截断第一条消息的内容（而不是删除其他消息）
    if messages_total_chars(&messages) > max_chars {
        truncate_first_message_to_fit(&mut messages, max_chars);
    }

    messages
}

/// 截断第一条消息的内容，使其适应剩余字符限制
/// 至少保留 50 个字符给第一条消息
fn truncate_first_message_to_fit(messages: &mut [Message], max_chars: usize) {
    if messages.is_empty() {
        return;
    }

    let remaining_chars = max_chars
        .saturating_sub(messages_total_chars(&messages[1..]))
        .max(50);

    let first = &mut messages[0];
    let text = value_to_string(&first.content);
    let truncated = truncate_to_chars(&text, remaining_chars);
    first.content = Value::String(truncated);
}

fn messages_total_chars(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|m| value_len_chars(&m.content))
        .sum::<usize>()
}

fn value_len_chars(v: &Value) -> usize {
    v.as_str()
        .map(|s| s.len())
        .unwrap_or_else(|| v.to_string().len())
}

fn value_to_string(v: &Value) -> String {
    v.as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| v.to_string())
}

fn build_summary_text(messages: &[Message], max_chars: usize) -> String {
    let mut lines = Vec::new();
    for m in messages {
        let role = match m.role.as_str() {
            "user" => "用户",
            "assistant" => "助手",
            "tool" => "工具",
            other => other,
        };
        let text = normalize_whitespace(&value_to_string(&m.content));

        // Include tool call information if present
        let tool_info = if let Some(ref tool_calls) = m.tool_calls {
            let tool_names: Vec<&str> = tool_calls
                .iter()
                .map(|tc| tc.function.name.as_str())
                .collect();
            if !tool_names.is_empty() {
                format!(" [tools: {}]", tool_names.join(", "))
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        if text.is_empty() && tool_info.is_empty() {
            continue;
        }

        let snippet = truncate_to_chars(&text, 200);
        lines.push(format!("{role}: {snippet}{tool_info}"));
        if lines.join("\n").len() >= max_chars {
            break;
        }
    }
    let joined = lines.join("\n");
    truncate_to_chars(&joined, max_chars)
}

fn normalize_whitespace(s: &str) -> String {
    let mut out = String::new();
    let mut in_ws = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !in_ws {
                out.push(' ');
                in_ws = true;
            }
        } else {
            out.push(ch);
            in_ws = false;
        }
    }
    out.trim().to_string()
}

fn truncate_to_chars(s: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let end = s
        .char_indices()
        .nth(max_chars)
        .map(|(idx, _)| idx)
        .unwrap_or_else(|| s.len());
    let mut out = s[..end].to_string();
    out.push('…');
    out
}

fn open_history_db(path: &Path) -> Result<Connection, io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Connection::open(path).map_err(|e| io::Error::other(e.to_string()))
}

fn init_history_schema(conn: &Connection) -> Result<(), io::Error> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_calls TEXT,
            tool_call_id TEXT,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );",
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    add_column_if_missing(conn, "messages", "tool_calls", "TEXT")?;
    add_column_if_missing(conn, "messages", "tool_call_id", "TEXT")?;
    Ok(())
}

fn add_column_if_missing(
    conn: &Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> Result<(), io::Error> {
    let sql = format!("ALTER TABLE {table} ADD COLUMN {column} {definition}");
    match conn.execute(&sql, []) {
        Ok(_) => Ok(()),
        Err(err) if err.to_string().contains("duplicate column name") => Ok(()),
        Err(err) => Err(io::Error::other(err.to_string())),
    }
}

fn append_history_sqlite(path: &Path, content: &str) -> io::Result<()> {
    let mut conn = open_history_db(path)?;
    init_history_schema(&conn)?;
    let entries = parse_history_blob(content);
    if entries.is_empty() {
        return Ok(());
    }
    let first_user_in_blob = entries
        .iter()
        .find(|message| message.role == "user")
        .map(|message| value_to_string(&message.content));
    let tx = conn
        .transaction()
        .map_err(|e| io::Error::other(e.to_string()))?;
    {
        let existing_first: Option<String> = tx
            .query_row(
                "SELECT value FROM meta WHERE key='first_user_prompt' LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()
            .unwrap_or(None);
        if existing_first.is_none() {
            let first_existing_user: Option<String> = tx
                .query_row(
                    "SELECT content FROM messages WHERE role='user' ORDER BY id ASC LIMIT 1",
                    [],
                    |row| row.get(0),
                )
                .optional()
                .unwrap_or(None);
            let first_user_prompt = first_existing_user.or(first_user_in_blob.clone());
            if let Some(v) = first_user_prompt.as_deref() {
                let _ = tx.execute(
                    "INSERT OR IGNORE INTO meta (key, value) VALUES ('first_user_prompt', ?1)",
                    params![v],
                );
            }
        }
        let mut stmt = tx
            .prepare(
                "INSERT INTO messages (role, content, tool_calls, tool_call_id)
                 VALUES (?1, ?2, ?3, ?4)",
            )
            .map_err(|e| io::Error::other(e.to_string()))?;
        for message in entries {
            let content = serde_json::to_string(&message.content)
                .map_err(|e| io::Error::other(e.to_string()))?;
            let tool_calls = message
                .tool_calls
                .as_ref()
                .map(serde_json::to_string)
                .transpose()
                .map_err(|e| io::Error::other(e.to_string()))?;
            stmt.execute(params![
                message.role,
                content,
                tool_calls,
                message.tool_call_id
            ])
            .map_err(|e| io::Error::other(e.to_string()))?;
        }
    }
    tx.execute(
        "DELETE FROM messages
         WHERE id NOT IN (SELECT id FROM messages ORDER BY id DESC LIMIT ?1)",
        params![MAX_HISTORY_LINES as i64],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))
}

fn build_message_arr_sqlite(
    history_count: usize,
    history_file: &Path,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let conn = match open_history_db(history_file) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.into()),
    };
    init_history_schema(&conn)?;

    let mut stmt = conn.prepare(
        "SELECT role, content, tool_calls, tool_call_id
         FROM messages
         ORDER BY id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![MAX_HISTORY_LINES as i64], |row| {
        let role: String = row.get(0)?;
        let content: String = row.get(1)?;
        let tool_calls: Option<String> = row.get(2)?;
        let tool_call_id: Option<String> = row.get(3)?;
        Ok((role, content, tool_calls, tool_call_id))
    })?;
    let mut messages: Vec<Message> = Vec::new();
    for row in rows {
        let (role, content, tool_calls, tool_call_id) = row?;
        messages.push(Message {
            role,
            content: decode_message_content(&content),
            tool_calls: decode_tool_calls(tool_calls.as_deref()),
            tool_call_id,
        });
    }
    messages.reverse();

    if history_count >= messages.len() {
        return Ok(messages);
    }
    Ok(messages[messages.len() - history_count..].to_vec())
}

#[derive(Debug, Clone)]
pub(super) struct SessionStore {
    root: PathBuf,
}

#[derive(Debug, Clone)]
pub(super) struct SessionInfo {
    pub(super) id: String,
    pub(super) modified_local: Option<DateTime<Local>>,
    pub(super) size_bytes: u64,
    pub(super) first_user_prompt: Option<String>,
}

impl SessionStore {
    pub(super) fn new(history_file: &Path) -> Self {
        Self {
            root: sessions_root_from_history_file(history_file),
        }
    }

    pub(super) fn ensure_root_dir(&self) -> io::Result<()> {
        fs::create_dir_all(&self.root)
    }

    pub(super) fn session_history_file(&self, session_id: &str) -> PathBuf {
        let id = sanitize_session_id(session_id);
        self.root.join(format!("{id}.sqlite"))
    }

    pub(super) fn session_assets_dir(&self, session_id: &str) -> PathBuf {
        let id = sanitize_session_id(session_id);
        self.root.join(format!("{id}.assets"))
    }

    pub(super) fn list_sessions(&self) -> io::Result<Vec<SessionInfo>> {
        let entries = match fs::read_dir(&self.root) {
            Ok(v) => v,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(err),
        };
        let mut out = Vec::new();
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("sqlite") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let metadata = match entry.metadata() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let modified_local = metadata.modified().ok().map(DateTime::<Local>::from);
            let first_user_prompt = read_first_user_prompt_sqlite(&path).unwrap_or(None);
            out.push(SessionInfo {
                id: stem.to_string(),
                modified_local,
                size_bytes: metadata.len(),
                first_user_prompt,
            });
        }
        out.sort_by(|a, b| {
            b.modified_local
                .cmp(&a.modified_local)
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(out)
    }

    pub(super) fn delete_session(&self, session_id: &str) -> io::Result<bool> {
        let path = self.session_history_file(session_id);
        let assets = self.session_assets_dir(session_id);
        let existed = path.exists();
        delete_history_artifacts(&path)?;
        let _ = delete_assets_dir(&assets);
        Ok(existed)
    }

    pub(super) fn clear_session(&self, session_id: &str) -> io::Result<()> {
        let path = self.session_history_file(session_id);
        let assets = self.session_assets_dir(session_id);
        let _ = delete_history_artifacts(&path);
        let _ = delete_assets_dir(&assets);
        Ok(())
    }

    pub(super) fn clear_all_sessions(&self) -> io::Result<usize> {
        let sessions = self.list_sessions()?;
        let mut deleted = 0usize;
        for s in sessions {
            if self.delete_session(&s.id).is_ok() {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    pub(super) fn first_user_prompt(&self, session_id: &str) -> io::Result<Option<String>> {
        let path = self.session_history_file(session_id);
        if !path.exists() {
            return Ok(None);
        }
        read_first_user_prompt_sqlite(&path)
    }

    pub(super) fn read_all_messages(&self, session_id: &str) -> io::Result<Vec<Message>> {
        let path = self.session_history_file(session_id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        read_all_messages_sqlite(&path)
    }

    pub(super) fn export_session_to_markdown(
        &self,
        session_id: &str,
        output_path: &Path,
    ) -> io::Result<()> {
        let messages = self.read_all_messages(session_id)?;
        if messages.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Session '{}' not found or empty", session_id),
            ));
        }

        let markdown = messages_to_markdown(&messages, session_id);

        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = File::create(output_path)?;
        use std::io::Write;
        file.write_all(markdown.as_bytes())?;

        Ok(())
    }

    pub(super) fn migrate_legacy_if_needed(&self, legacy_history_file: &PathBuf) -> io::Result<()> {
        self.ensure_root_dir()?;
        self.migrate_txt_sessions_to_sqlite()?;

        let legacy_session_id = "legacy";
        let legacy_session_path = self.session_history_file(legacy_session_id);
        if legacy_session_path.exists() {
            return Ok(());
        }
        if is_sqlite_path(legacy_history_file) {
            return Ok(());
        }
        let history = match fs::read_to_string(legacy_history_file) {
            Ok(v) => v,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(err) => return Err(err),
        };
        if history.trim().is_empty() {
            return Ok(());
        }
        append_history_sqlite(&legacy_session_path, &history)?;
        Ok(())
    }

    fn migrate_txt_sessions_to_sqlite(&self) -> io::Result<()> {
        let entries = match fs::read_dir(&self.root) {
            Ok(v) => v,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(err) => return Err(err),
        };
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("txt") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let sqlite_path = self.root.join(format!("{stem}.sqlite"));
            if sqlite_path.exists() {
                let _ = fs::remove_file(&path);
                continue;
            }
            let history = match fs::read_to_string(&path) {
                Ok(v) => v,
                Err(err) if err.kind() == io::ErrorKind::NotFound => continue,
                Err(err) => return Err(err),
            };
            append_history_sqlite(&sqlite_path, &history)?;
            let _ = fs::remove_file(&path);
        }
        Ok(())
    }
}

fn sessions_root_from_history_file(history_file: &Path) -> PathBuf {
    let parent = history_file.parent().unwrap_or_else(|| Path::new("."));
    let name = history_file
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("history");
    parent.join(format!("{name}.sessions"))
}

fn sanitize_session_id(session_id: &str) -> String {
    let mut out = String::new();
    for ch in session_id.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else if ch.is_whitespace() {
            out.push('_');
        }
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        "session".to_string()
    } else {
        out
    }
}

fn read_first_user_prompt_sqlite(path: &Path) -> io::Result<Option<String>> {
    let conn = Connection::open(path).map_err(|e| io::Error::other(e.to_string()))?;
    let meta: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key='first_user_prompt' LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .unwrap_or(None);
    if meta.is_some() {
        return Ok(meta);
    }
    let fallback: Option<String> = conn
        .query_row(
            "SELECT content FROM messages WHERE role='user' ORDER BY id ASC LIMIT 1",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .unwrap_or(None);
    Ok(fallback.map(|content| value_to_string(&decode_message_content(&content))))
}

fn decode_message_content(content: &str) -> Value {
    serde_json::from_str(content).unwrap_or_else(|_| Value::String(content.to_string()))
}

fn decode_tool_calls(tool_calls: Option<&str>) -> Option<Vec<ToolCall>> {
    tool_calls.and_then(|raw| serde_json::from_str(raw).ok())
}

fn read_all_messages_sqlite(path: &Path) -> io::Result<Vec<Message>> {
    let conn = Connection::open(path).map_err(|e| io::Error::other(e.to_string()))?;

    let mut stmt = conn
        .prepare(
            "SELECT role, content, tool_calls, tool_call_id
             FROM messages
             ORDER BY id ASC",
        )
        .map_err(|e| io::Error::other(e.to_string()))?;

    let rows = stmt
        .query_map([], |row| {
            let role: String = row.get(0)?;
            let content: String = row.get(1)?;
            let tool_calls: Option<String> = row.get(2)?;
            let tool_call_id: Option<String> = row.get(3)?;
            Ok((role, content, tool_calls, tool_call_id))
        })
        .map_err(|e| io::Error::other(e.to_string()))?;

    let mut messages: Vec<Message> = Vec::new();
    for row in rows {
        let (role, content, tool_calls, tool_call_id) =
            row.map_err(|e| io::Error::other(e.to_string()))?;
        messages.push(Message {
            role,
            content: decode_message_content(&content),
            tool_calls: decode_tool_calls(tool_calls.as_deref()),
            tool_call_id,
        });
    }

    Ok(messages)
}

pub(super) fn messages_to_markdown(messages: &[Message], session_id: &str) -> String {
    let mut md = String::new();
    md.push_str(&format!("# Session: {}\n\n", session_id));
    md.push_str(&format!("**Total messages:** {}\n\n", messages.len()));
    md.push_str("---\n\n");

    for (i, msg) in messages.iter().enumerate() {
        let role_emoji = match msg.role.as_str() {
            "user" => "👤",
            "assistant" => "🤖",
            "system" => "⚙️",
            "tool" => "🔧",
            _ => "📝",
        };

        md.push_str(&format!("### {} {}\n\n", role_emoji, msg.role.to_uppercase()));

        let content_str = value_to_string(&msg.content);
        if !content_str.is_empty() {
            md.push_str(&content_str);
            md.push_str("\n\n");
        }

        if let Some(ref tool_calls) = msg.tool_calls {
            md.push_str("**Tool Calls:**\n");
            for tc in tool_calls {
                md.push_str(&format!("- `{}`", tc.function.name));
                if !tc.function.arguments.trim().is_empty() {
                    // Try to parse as JSON for pretty printing
                    if let Ok(args_val) = serde_json::from_str::<Value>(&tc.function.arguments) {
                        md.push_str(&format!("({})", args_val));
                    } else {
                        md.push_str(&format!("({})", tc.function.arguments));
                    }
                }
                md.push('\n');
            }
            md.push('\n');
        }

        if let Some(ref tool_call_id) = msg.tool_call_id {
            md.push_str(&format!("**Tool Call ID:** `{}`\n\n", tool_call_id));
        }

        if i < messages.len() - 1 {
            md.push_str("---\n\n");
        }
    }

    md
}
