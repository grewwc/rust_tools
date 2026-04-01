use std::{
    fs::File,
    fs::{self},
    io,
    path::{Path, PathBuf},
};

use chrono::{DateTime, Local};
use rust_tools::cw::SkipMap;

use super::{
    blob::{delete_assets_dir, delete_history_artifacts, is_sqlite_path, parse_history_blob},
    markdown::messages_to_markdown,
    sqlite::{read_all_messages_sqlite, read_first_user_prompt_sqlite},
    types::Message,
};

#[derive(Debug, Clone)]
pub(in crate::ai) struct SessionStore {
    root: PathBuf,
}

#[derive(Debug, Clone)]
pub(in crate::ai) struct SessionInfo {
    pub(in crate::ai) id: String,
    pub(in crate::ai) modified_local: Option<DateTime<Local>>,
    pub(in crate::ai) size_bytes: u64,
    pub(in crate::ai) first_user_prompt: Option<String>,
}

impl SessionStore {
    pub(in crate::ai) fn new(history_file: &Path) -> Self {
        Self {
            root: sessions_root_from_history_file(history_file),
        }
    }

    pub(in crate::ai) fn ensure_root_dir(&self) -> io::Result<()> {
        fs::create_dir_all(&self.root)
    }

    pub(in crate::ai) fn session_history_file(&self, session_id: &str) -> PathBuf {
        let id = sanitize_session_id(session_id);
        self.root.join(format!("{id}.sqlite"))
    }

    pub(in crate::ai) fn session_assets_dir(&self, session_id: &str) -> PathBuf {
        let id = sanitize_session_id(session_id);
        self.root.join(format!("{id}.assets"))
    }

    pub(in crate::ai) fn list_sessions(&self) -> io::Result<Vec<SessionInfo>> {
        let entries = match fs::read_dir(&self.root) {
            Ok(v) => v,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(err),
        };
        let mut sessions: Box<SkipMap<(u64, String), SessionInfo>> =
            SkipMap::new(16, |a: &(u64, String), b: &(u64, String)| {
                match b.0.cmp(&a.0) {
                    std::cmp::Ordering::Equal => a.1.cmp(&b.1) as i32 * -1,
                    std::cmp::Ordering::Less => 1,
                    std::cmp::Ordering::Greater => -1,
                }
            });
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
            let id = stem.to_string();
            let timestamp = modified_local
                .map(|dt| dt.timestamp_millis() as u64)
                .unwrap_or(0);
            sessions.insert(
                (timestamp, id.clone()),
                SessionInfo {
                    id,
                    modified_local,
                    size_bytes: metadata.len(),
                    first_user_prompt,
                },
            );
        }
        Ok(sessions.into_iter().map(|(_, v)| v.clone()).collect())
    }

    pub(in crate::ai) fn delete_session(&self, session_id: &str) -> io::Result<bool> {
        let path = self.session_history_file(session_id);
        let assets = self.session_assets_dir(session_id);
        let existed = path.exists();
        delete_history_artifacts(&path)?;
        let _ = delete_assets_dir(&assets);
        Ok(existed)
    }

    pub(in crate::ai) fn clear_session(&self, session_id: &str) -> io::Result<()> {
        let path = self.session_history_file(session_id);
        let assets = self.session_assets_dir(session_id);
        let _ = delete_history_artifacts(&path);
        let _ = delete_assets_dir(&assets);
        Ok(())
    }

    pub(in crate::ai) fn clear_all_sessions(&self) -> io::Result<usize> {
        let sessions = self.list_sessions()?;
        let mut deleted = 0usize;
        for s in sessions {
            if self.delete_session(&s.id).is_ok() {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    pub(in crate::ai) fn first_user_prompt(&self, session_id: &str) -> io::Result<Option<String>> {
        let path = self.session_history_file(session_id);
        if !path.exists() {
            return Ok(None);
        }
        read_first_user_prompt_sqlite(&path)
    }

    pub(in crate::ai) fn read_all_messages(&self, session_id: &str) -> io::Result<Vec<Message>> {
        let path = self.session_history_file(session_id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        read_all_messages_sqlite(&path)
    }

    pub(in crate::ai) fn export_session_to_markdown(
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

    pub(in crate::ai) fn migrate_legacy_if_needed(
        &self,
        legacy_history_file: &PathBuf,
    ) -> io::Result<()> {
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
        super::sqlite::append_history_sqlite(&legacy_session_path, parse_history_blob(&history))?;
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
            super::sqlite::append_history_sqlite(&sqlite_path, parse_history_blob(&history))?;
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
