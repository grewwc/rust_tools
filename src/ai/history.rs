use std::{
    fs::{self, OpenOptions},
    io,
    path::Path,
    path::PathBuf,
};

use chrono::{DateTime, Local};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::strw::split::split_by_str_keep_quotes;

use super::types::ToolCall;

const MAX_HISTORY_LINES: usize = 100;
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
    let lines = split_by_str_keep_quotes(&history, &newline, "\"", false);
    let mut messages = Vec::new();

    for line in &lines {
        if line.is_empty() {
            continue;
        }
        let Some(last_colon) = line.rfind(COLON) else {
            continue;
        };
        if last_colon == 0 || last_colon + COLON.len_utf8() >= line.len() {
            continue;
        }
        let role = &line[..last_colon];
        if role != "user" && role != "assistant" {
            continue;
        }
        let content = &line[last_colon + COLON.len_utf8()..];
        messages.push(Message {
            role: role.to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        });
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
    let mut options = OpenOptions::new();
    options.create(true).append(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o664);
    }
    let mut file = options.open(path)?;
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

fn is_sqlite_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|s| s.to_str()),
        Some("sqlite") | Some("db")
    )
}

fn parse_history_blob(content: &str) -> Vec<(String, String)> {
    let newline = NEWLINE.to_string();
    let lines = split_by_str_keep_quotes(content, &newline, "\"", false);
    let mut out = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let Some(last_colon) = line.rfind(COLON) else {
            continue;
        };
        if last_colon == 0 || last_colon + COLON.len_utf8() >= line.len() {
            continue;
        }
        let role = &line[..last_colon];
        if role != "user" && role != "assistant" {
            continue;
        }
        let msg = &line[last_colon + COLON.len_utf8()..];
        out.push((role.to_string(), msg.to_string()));
    }
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
    Ok(())
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
        .find(|(role, _)| role == "user")
        .map(|(_, msg)| msg.clone());
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
            .prepare("INSERT INTO messages (role, content) VALUES (?1, ?2)")
            .map_err(|e| io::Error::other(e.to_string()))?;
        for (role, msg) in entries {
            stmt.execute(params![role, msg])
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
        "SELECT role, content
         FROM messages
         ORDER BY id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![MAX_HISTORY_LINES as i64], |row| {
        let role: String = row.get(0)?;
        let content: String = row.get(1)?;
        Ok((role, content))
    })?;
    let mut messages: Vec<Message> = Vec::new();
    for row in rows {
        let (role, content) = row?;
        messages.push(Message {
            role,
            content: Value::String(content),
            tool_calls: None,
            tool_call_id: None,
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
        let existed = path.exists();
        delete_history_artifacts(&path)?;
        Ok(existed)
    }

    pub(super) fn clear_session(&self, session_id: &str) -> io::Result<()> {
        let path = self.session_history_file(session_id);
        let _ = delete_history_artifacts(&path);
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
            |row| row.get(0),
        )
        .optional()
        .unwrap_or(None);
    Ok(fallback)
}
