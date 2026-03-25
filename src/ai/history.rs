use std::{
    fs::{self, OpenOptions},
    io,
    path::Path,
    path::PathBuf,
};

use chrono::{DateTime, Local};
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

pub(super) fn append_history(path: &PathBuf, content: &str) -> io::Result<()> {
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

#[derive(Debug, Clone)]
pub(super) struct SessionStore {
    root: PathBuf,
}

#[derive(Debug, Clone)]
pub(super) struct SessionInfo {
    pub(super) id: String,
    pub(super) modified_local: Option<DateTime<Local>>,
    pub(super) size_bytes: u64,
}

impl SessionStore {
    pub(super) fn new(history_file: &PathBuf) -> Self {
        Self {
            root: sessions_root_from_history_file(history_file),
        }
    }

    pub(super) fn root(&self) -> &PathBuf {
        &self.root
    }

    pub(super) fn ensure_root_dir(&self) -> io::Result<()> {
        fs::create_dir_all(&self.root)
    }

    pub(super) fn session_history_file(&self, session_id: &str) -> PathBuf {
        let id = sanitize_session_id(session_id);
        self.root.join(format!("{id}.txt"))
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
            if path.extension().and_then(|s| s.to_str()) != Some("txt") {
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
            out.push(SessionInfo {
                id: stem.to_string(),
                modified_local,
                size_bytes: metadata.len(),
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
        match fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(err) => Err(err),
        }
    }

    pub(super) fn clear_session(&self, session_id: &str) -> io::Result<()> {
        let path = self.session_history_file(session_id);
        let _ = fs::remove_file(path);
        Ok(())
    }

    pub(super) fn migrate_legacy_if_needed(&self, legacy_history_file: &PathBuf) -> io::Result<()> {
        let legacy_session_id = "legacy";
        let legacy_session_path = self.session_history_file(legacy_session_id);
        if legacy_session_path.exists() {
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
        self.ensure_root_dir()?;
        fs::write(legacy_session_path, history)
    }
}

fn sessions_root_from_history_file(history_file: &PathBuf) -> PathBuf {
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
