use std::path::{Path, PathBuf};

use crate::ai::history::{Message, SessionStore, read_all_messages_sqlite_read_only};

#[cfg(test)]
#[path = "input_tests.rs"]
mod tests;

/// The CLI accepts either an existing archive path or a complete local session ID.
/// The agent tool keeps its archive-only contract and constructs `Archive` directly.
#[derive(Debug)]
pub(in crate::ai) enum DistillInput {
    Archive(PathBuf),
    Session { id: String, path: PathBuf },
}

impl DistillInput {
    pub(in crate::ai) fn resolve(
        value: &str,
        cwd: &Path,
        store: &SessionStore,
    ) -> Result<Self, String> {
        if value.trim().is_empty() {
            return Err("--distill-session requires an archive path or session ID".into());
        }
        let path = cwd.join(value);
        // Existing paths win, including extensionless archives that resemble IDs.
        match std::fs::metadata(&path) {
            Ok(metadata) if metadata.is_file() => return Ok(Self::Archive(path)),
            Ok(_) => return Err(format!("Archive path is not a file: {}", path.display())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("Cannot inspect {}: {error}", path.display())),
        }
        SessionStore::validate_session_id(value).map_err(|error| {
            format!("Archive not found at {}; invalid session ID: {error}", path.display())
        })?;
        // Resolve the validated ID exactly; legacy filename sanitization trims
        // underscores and can redirect an otherwise valid ID to another session.
        let path = store.sessions_root().join(format!("{value}.sqlite"));
        if !path.is_file() {
            return Err(format!(
                "Local session '{value}' not found in {} (use a complete session ID or an existing ZIP path)",
                store.sessions_root().display()
            ));
        }
        Ok(Self::Session {
            id: value.to_owned(),
            path,
        })
    }

    pub(in crate::ai) fn path(&self) -> &Path {
        match self {
            Self::Archive(path) | Self::Session { path, .. } => path,
        }
    }

    pub(super) fn read_messages(&self) -> Result<(String, Vec<Message>), String> {
        match self {
            Self::Archive(path) => super::read_archive_messages(path),
            Self::Session { id, path } => read_all_messages_sqlite_read_only(path)
                .map(|messages| (id.clone(), messages))
                .map_err(|error| format!("Failed to read local session '{id}': {error}")),
        }
    }
}