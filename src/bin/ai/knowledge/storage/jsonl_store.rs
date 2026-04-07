/// Pure JSONL storage — CRUD operations without search logic.
/// Search is handled by the retrieval module.
use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::super::entry::KnowledgeEntry;

/// JSONL-based knowledge store.
pub struct JsonlStore {
    path: PathBuf,
}

impl JsonlStore {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append a single entry to the file.
    pub fn append(&self, entry: &KnowledgeEntry) -> Result<(), String> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create dir: {e}"))?;
        }

        let serialized =
            serde_json::to_string(entry).map_err(|e| format!("Failed to serialize: {e}"))?;

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&self.path)
            .map_err(|e| format!("Failed to open file: {e}"))?;

        let needs_newline = file
            .metadata()
            .map_err(|e| format!("Failed to read metadata: {e}"))?
            .len()
            > 0
            && {
                file.seek(SeekFrom::End(-1))
                    .map_err(|e| format!("Failed to seek: {e}"))?;
                let mut last = [0u8; 1];
                file.read_exact(&mut last)
                    .map_err(|e| format!("Failed to read: {e}"))?;
                last[0] != b'\n'
            };

        if needs_newline {
            file.write_all(b"\n")
                .map_err(|e| format!("Failed to write: {e}"))?;
        }
        file.write_all(serialized.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .map_err(|e| format!("Failed to write: {e}"))?;

        Ok(())
    }

    /// Get the most recent N entries (no scoring).
    pub fn recent(&self, limit: usize) -> Result<Vec<KnowledgeEntry>, String> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = fs::File::open(&self.path).map_err(|e| format!("Failed to open file: {e}"))?;
        let reader = BufReader::new(file);

        let mut window: VecDeque<KnowledgeEntry> = VecDeque::with_capacity(limit + 1);
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read: {e}"))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(entry) = serde_json::from_str::<KnowledgeEntry>(line) {
                window.push_back(entry);
                if window.len() > limit {
                    window.pop_front();
                }
            }
        }

        let mut entries: Vec<KnowledgeEntry> = window.into_iter().collect();
        entries.reverse();
        Ok(entries)
    }

    /// Get ALL entries without any scoring or filtering.
    /// This is the correct replacement for the old `all()` method.
    pub fn list_all(&self) -> Result<Vec<KnowledgeEntry>, String> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = fs::File::open(&self.path).map_err(|e| format!("Failed to open file: {e}"))?;
        let reader = BufReader::new(file);

        let mut entries = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read: {e}"))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(entry) = serde_json::from_str::<KnowledgeEntry>(line) {
                entries.push(entry);
            }
        }
        Ok(entries)
    }

    /// Delete an entry by ID. Returns the deleted entry if found.
    pub fn delete_by_id(&self, id: &str) -> Result<Option<KnowledgeEntry>, String> {
        let content = std::fs::read_to_string(&self.path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let mut entries: Vec<KnowledgeEntry> = Vec::new();
        let mut deleted_entry: Option<KnowledgeEntry> = None;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(entry) = serde_json::from_str::<KnowledgeEntry>(line) {
                let entry_id = entry.id.as_deref().unwrap_or("");
                if entry_id == id {
                    deleted_entry = Some(entry);
                    continue;
                }
                entries.push(entry);
            }
        }

        if deleted_entry.is_none() {
            return Ok(None);
        }

        let mut output = String::new();
        for entry in &entries {
            if let Ok(s) = serde_json::to_string(entry) {
                output.push_str(&s);
                output.push('\n');
            }
        }

        std::fs::write(&self.path, output).map_err(|e| format!("Failed to write file: {}", e))?;

        Ok(deleted_entry)
    }

    /// Rewrite the entire file with the given entries.
    pub fn rewrite(&self, entries: &[KnowledgeEntry]) -> Result<(), String> {
        let tmp = self.path.with_extension("jsonl.tmp");
        {
            let mut f =
                fs::File::create(&tmp).map_err(|e| format!("Failed to create tmp: {}", e))?;
            for entry in entries {
                let line = serde_json::to_string(entry).map_err(|e| format!("{}", e))?;
                f.write_all(line.as_bytes())
                    .and_then(|_| f.write_all(b"\n"))
                    .map_err(|e| format!("Failed to write tmp: {}", e))?;
            }
        }
        std::fs::rename(&tmp, &self.path).map_err(|e| format!("Failed to replace file: {}", e))?;
        Ok(())
    }
}
