use std::{
    ffi::CStr,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

use chrono::Local;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::commonw::utils::{expanduser, get_config_dir, open_file_for_write_truncate};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(in crate::ai) struct SuspendedSessionEntry {
    pub(in crate::ai) terminal_key: String,
    pub(in crate::ai) session_id: String,
    pub(in crate::ai) history_file: PathBuf,
    pub(in crate::ai) persona_id: String,
    #[serde(default)]
    pub(in crate::ai) suspended_at: String,
}

#[derive(Debug, Clone)]
pub(in crate::ai) struct SuspendedSessionStore {
    root: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum SuspendedSessionFile {
    Single(SuspendedSessionEntry),
    Many(Vec<SuspendedSessionEntry>),
}

impl SuspendedSessionStore {
    pub(in crate::ai) fn new() -> Self {
        Self {
            root: suspended_sessions_root(),
        }
    }

    #[cfg(test)]
    pub(in crate::ai) fn for_tests_with_root(root: PathBuf) -> Self {
        Self { root }
    }

    pub(in crate::ai) fn suspend_current_terminal(
        &self,
        session_id: &str,
        history_file: &Path,
        persona_id: &str,
    ) -> io::Result<SuspendedSessionEntry> {
        let key = current_terminal_key().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "当前 terminal 不可识别，无法挂起；可改用 `a --session <id>` 手动回到该会话",
            )
        })?;
        self.save_for_terminal_key(&key, session_id, history_file, persona_id)
    }

    pub(in crate::ai) fn take_current_terminal(&self) -> io::Result<Option<SuspendedSessionEntry>> {
        let key = current_terminal_key().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "当前 terminal 不可识别，无法恢复挂起会话",
            )
        })?;
        self.take_for_terminal_key(&key)
    }

    pub(in crate::ai) fn list_current_terminal(&self) -> io::Result<Vec<SuspendedSessionEntry>> {
        let key = current_terminal_key().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "当前 terminal 不可识别，无法查看挂起会话",
            )
        })?;
        self.list_for_terminal_key(&key)
    }

    pub(in crate::ai) fn save_for_terminal_key(
        &self,
        terminal_key: &str,
        session_id: &str,
        history_file: &Path,
        persona_id: &str,
    ) -> io::Result<SuspendedSessionEntry> {
        let terminal_key = terminal_key.trim();
        let session_id = session_id.trim();
        let persona_id = persona_id.trim();
        if terminal_key.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "terminal key cannot be empty",
            ));
        }
        if session_id.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "session id cannot be empty",
            ));
        }
        if history_file.as_os_str().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "history file cannot be empty",
            ));
        }
        if persona_id.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "persona id cannot be empty",
            ));
        }

        let entry = SuspendedSessionEntry {
            terminal_key: terminal_key.to_string(),
            session_id: session_id.to_string(),
            history_file: history_file.to_path_buf(),
            persona_id: persona_id.to_string(),
            suspended_at: Local::now().to_rfc3339(),
        };
        let mut entries = self
            .read_entries(terminal_key)?
            .unwrap_or_default();
        entries.retain(|existing| !same_binding(existing, &entry));
        entries.push(entry.clone());
        self.write_entries(terminal_key, &entries)?;
        Ok(entry)
    }

    #[cfg(test)]
    pub(in crate::ai) fn peek_for_terminal_key(
        &self,
        terminal_key: &str,
    ) -> io::Result<Option<SuspendedSessionEntry>> {
        Ok(self
            .read_entries(terminal_key)?
            .and_then(|entries| entries.last().cloned()))
    }

    #[cfg(test)]
    pub(in crate::ai) fn peek_entries_for_terminal_key(
        &self,
        terminal_key: &str,
    ) -> io::Result<Vec<SuspendedSessionEntry>> {
        self.list_for_terminal_key(terminal_key)
    }

    pub(in crate::ai) fn take_for_terminal_key(
        &self,
        terminal_key: &str,
    ) -> io::Result<Option<SuspendedSessionEntry>> {
        let Some(mut entries) = self.read_entries(terminal_key)? else {
            return Ok(None);
        };
        let Some(entry) = entries.pop() else {
            self.write_entries(terminal_key, &entries)?;
            return Ok(None);
        };
        self.write_entries(terminal_key, &entries)?;
        Ok(Some(entry))
    }

    pub(in crate::ai) fn list_for_terminal_key(
        &self,
        terminal_key: &str,
    ) -> io::Result<Vec<SuspendedSessionEntry>> {
        let mut entries = self.read_entries(terminal_key)?.unwrap_or_default();
        entries.reverse();
        Ok(entries)
    }

    pub(in crate::ai) fn take_selected_current_terminal(
        &self,
        selected: &SuspendedSessionEntry,
    ) -> io::Result<Option<SuspendedSessionEntry>> {
        let key = current_terminal_key().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "当前 terminal 不可识别，无法恢复挂起会话",
            )
        })?;
        self.take_selected_for_terminal_key(&key, selected)
    }

    pub(in crate::ai) fn take_selected_for_terminal_key(
        &self,
        terminal_key: &str,
        selected: &SuspendedSessionEntry,
    ) -> io::Result<Option<SuspendedSessionEntry>> {
        let Some(mut entries) = self.read_entries(terminal_key)? else {
            return Ok(None);
        };
        let Some(index) = entries.iter().rposition(|entry| same_binding(entry, selected)) else {
            return Ok(None);
        };
        let entry = entries.remove(index);
        self.write_entries(terminal_key, &entries)?;
        Ok(Some(entry))
    }

    fn read_entries(&self, terminal_key: &str) -> io::Result<Option<Vec<SuspendedSessionEntry>>> {
        let path = self.entry_path(terminal_key);
        let content = match fs::read_to_string(&path) {
            Ok(content) => content,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err),
        };
        let parsed: SuspendedSessionFile = serde_json::from_str(&content).map_err(|err| {
            let _ = fs::remove_file(&path);
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to parse suspended session entry: {err}"),
            )
        })?;
        let entries = match parsed {
            SuspendedSessionFile::Single(entry) => vec![entry],
            SuspendedSessionFile::Many(entries) => entries,
        };
        if entries.iter().any(|entry| entry.terminal_key != terminal_key) {
            let _ = fs::remove_file(&path);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "suspended session terminal key mismatch",
            ));
        }
        Ok(Some(entries))
    }

    fn write_entries(
        &self,
        terminal_key: &str,
        entries: &[SuspendedSessionEntry],
    ) -> io::Result<()> {
        let path = self.entry_path(terminal_key);
        if entries.is_empty() {
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                Err(err) => return Err(err),
            }
            return Ok(());
        }
        fs::create_dir_all(&self.root)?;
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("session.json");
        let tmp = self
            .root
            .join(format!("{file_name}.tmp-{}", Uuid::new_v4().simple()));
        let content = serde_json::to_vec_pretty(entries).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to serialize suspended session entry: {err}"),
            )
        })?;
        {
            let mut file = open_file_for_write_truncate(&tmp, 0o600)?;
            file.write_all(&content)?;
        }
        fs::rename(&tmp, &path)?;
        Ok(())
    }

    fn entry_path(&self, terminal_key: &str) -> PathBuf {
        self.root
            .join(format!("{}.json", hex_encode(terminal_key.as_bytes())))
    }
}

fn suspended_sessions_root() -> PathBuf {
    if let Ok(path) = std::env::var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR") {
        let path = path.trim();
        if !path.is_empty() {
            return PathBuf::from(expanduser(path).as_ref());
        }
    }
    get_config_dir()
        .unwrap_or_else(|| PathBuf::from("~/.config"))
        .join("rust_tools")
        .join("suspended_sessions")
}

pub(in crate::ai) fn current_terminal_key() -> Option<String> {
    resolve_terminal_key_from_sources(
        |name| std::env::var(name).ok(),
        current_tty_path().as_deref(),
    )
}

fn resolve_terminal_key_from_sources<F>(mut getenv: F, tty: Option<&str>) -> Option<String>
where
    F: FnMut(&str) -> Option<String>,
{
    for (prefix, name) in [
        ("tmux", "TMUX_PANE"),
        ("wezterm", "WEZTERM_PANE"),
        ("iterm", "ITERM_SESSION_ID"),
        ("terminal", "TERM_SESSION_ID"),
    ] {
        if let Some(value) = getenv(name) {
            let value = value.trim();
            if !value.is_empty() {
                return Some(format!("{prefix}:{value}"));
            }
        }
    }
    tty.map(|value| format!("tty:{value}"))
}

#[cfg(unix)]
fn current_tty_path() -> Option<String> {
    for fd in [libc::STDIN_FILENO, libc::STDOUT_FILENO, libc::STDERR_FILENO] {
        let path = tty_path_for_fd(fd);
        if path.is_some() {
            return path;
        }
    }
    None
}

#[cfg(not(unix))]
fn current_tty_path() -> Option<String> {
    None
}

#[cfg(unix)]
fn tty_path_for_fd(fd: libc::c_int) -> Option<String> {
    if unsafe { libc::isatty(fd) } != 1 {
        return None;
    }
    let mut buf = vec![0u8; 4096];
    if unsafe { libc::ttyname_r(fd, buf.as_mut_ptr() as *mut libc::c_char, buf.len()) } != 0 {
        return None;
    }
    let cstr = unsafe { CStr::from_ptr(buf.as_ptr() as *const libc::c_char) };
    let value = cstr.to_string_lossy().trim().to_string();
    if value.is_empty() { None } else { Some(value) }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(hex_digit(byte >> 4));
        out.push(hex_digit(byte & 0x0f));
    }
    out
}

fn hex_digit(value: u8) -> char {
    match value {
        0..=9 => (b'0' + value) as char,
        10..=15 => (b'a' + (value - 10)) as char,
        _ => '0',
    }
}

fn same_binding(left: &SuspendedSessionEntry, right: &SuspendedSessionEntry) -> bool {
    left.terminal_key == right.terminal_key
        && left.session_id == right.session_id
        && left.history_file == right.history_file
        && left.persona_id == right.persona_id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_terminal_key_prefers_mux_like_ids() {
        let key = resolve_terminal_key_from_sources(
            |name| match name {
                "TMUX_PANE" => Some("%12".to_string()),
                "ITERM_SESSION_ID" => Some("worse-choice".to_string()),
                _ => None,
            },
            Some("/dev/ttys001"),
        );
        assert_eq!(key.as_deref(), Some("tmux:%12"));
    }

    #[test]
    fn resolve_terminal_key_falls_back_to_tty() {
        let key = resolve_terminal_key_from_sources(|_| None, Some("/dev/ttys009"));
        assert_eq!(key.as_deref(), Some("tty:/dev/ttys009"));
    }

    #[test]
    fn store_round_trips_entry_by_terminal_key() {
        let root = std::env::temp_dir().join(format!(
            "rust-tools-suspended-session-{}",
            Uuid::new_v4()
        ));
        let store = SuspendedSessionStore::for_tests_with_root(root.clone());
        let history = root.join("history.sqlite");
        let entry = store
            .save_for_terminal_key("tty:/dev/ttys001", "sess-1", &history, "default")
            .unwrap();
        assert_eq!(entry.session_id, "sess-1");

        let peeked = store
            .peek_for_terminal_key("tty:/dev/ttys001")
            .unwrap()
            .expect("entry should exist");
        assert_eq!(peeked.session_id, "sess-1");
        assert_eq!(peeked.history_file, history);

        let taken = store
            .take_for_terminal_key("tty:/dev/ttys001")
            .unwrap()
            .expect("entry should exist");
        assert_eq!(taken.persona_id, "default");
        assert!(
            store
                .take_for_terminal_key("tty:/dev/ttys001")
                .unwrap()
                .is_none()
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn store_keeps_multiple_entries_per_terminal_key() {
        let root = std::env::temp_dir().join(format!(
            "rust-tools-suspended-session-stack-{}",
            Uuid::new_v4()
        ));
        let store = SuspendedSessionStore::for_tests_with_root(root.clone());
        let history = root.join("history.sqlite");
        let other = root.join("other.sqlite");
        store
            .save_for_terminal_key("tty:/dev/ttys002", "sess-1", &history, "default")
            .unwrap();
        store
            .save_for_terminal_key("tty:/dev/ttys002", "sess-2", &other, "reviewer")
            .unwrap();

        let entries = store
            .peek_entries_for_terminal_key("tty:/dev/ttys002")
            .unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].session_id, "sess-2");
        assert_eq!(entries[1].session_id, "sess-1");

        let taken = store
            .take_selected_for_terminal_key("tty:/dev/ttys002", &entries[1])
            .unwrap()
            .expect("selected entry should exist");
        assert_eq!(taken.session_id, "sess-1");

        let remaining = store
            .peek_entries_for_terminal_key("tty:/dev/ttys002")
            .unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].session_id, "sess-2");

        let _ = fs::remove_dir_all(root);
    }
}
