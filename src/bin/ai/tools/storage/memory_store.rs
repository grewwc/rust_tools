use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AgentMemoryEntry {
    pub(crate) timestamp: String,
    pub(crate) category: String,
    pub(crate) note: String,
    pub(crate) tags: Vec<String>,
    pub(crate) source: Option<String>,
}

pub(crate) struct MemoryStore {
    path: PathBuf,
}

impl MemoryStore {
    pub(crate) fn from_env_or_config() -> Self {
        Self {
            path: resolve_memory_file(),
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn append(&self, entry: &AgentMemoryEntry) -> Result<(), String> {
        super::with_memory_file_lock(&self.path, || {
            if let Some(parent) = self.path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create memory dir: {e}"))?;
            }
            let serialized = serde_json::to_string(entry)
                .map_err(|e| format!("Failed to serialize memory entry: {e}"))?;

            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .read(true)
                .open(&self.path)
                .map_err(|e| format!("Failed to open memory file: {e}"))?;

            let needs_newline = file
                .metadata()
                .map_err(|e| format!("Failed to read memory file metadata: {e}"))?
                .len()
                > 0
                && {
                    file.seek(SeekFrom::End(-1))
                        .map_err(|e| format!("Failed to seek memory file: {e}"))?;
                    let mut last = [0u8; 1];
                    file.read_exact(&mut last)
                        .map_err(|e| format!("Failed to read memory file: {e}"))?;
                    last[0] != b'\n'
                };

            if needs_newline {
                file.write_all(b"\n")
                    .map_err(|e| format!("Failed to write memory file: {e}"))?;
            }
            file.write_all(serialized.as_bytes())
                .and_then(|_| file.write_all(b"\n"))
                .map_err(|e| format!("Failed to write memory file: {e}"))?;

            Ok(())
        })
    }

    pub(crate) fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<AgentMemoryEntry>, String> {
        let query_lc = query.to_lowercase();
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = fs::File::open(&self.path).map_err(|e| format!("Failed to read memory file: {e}"))?;
        let reader = BufReader::new(file);

        let mut window: VecDeque<AgentMemoryEntry> = VecDeque::with_capacity(limit + 1);
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read memory file: {e}"))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(line) else {
                continue;
            };
            let matched = entry.note.to_lowercase().contains(&query_lc)
                || entry.category.to_lowercase().contains(&query_lc)
                || entry.tags.iter().any(|tag| tag.to_lowercase().contains(&query_lc))
                || entry
                    .source
                    .as_ref()
                    .is_some_and(|source| source.to_lowercase().contains(&query_lc));
            if !matched {
                continue;
            }
            window.push_back(entry);
            if window.len() > limit {
                window.pop_front();
            }
        }

        let mut entries: Vec<AgentMemoryEntry> = window.into_iter().collect();
        entries.reverse();
        Ok(entries)
    }

    pub(crate) fn recent(&self, limit: usize) -> Result<Vec<AgentMemoryEntry>, String> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = fs::File::open(&self.path).map_err(|e| format!("Failed to read memory file: {e}"))?;
        let reader = BufReader::new(file);

        let mut window: VecDeque<AgentMemoryEntry> = VecDeque::with_capacity(limit + 1);
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read memory file: {e}"))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(line) else {
                continue;
            };
            window.push_back(entry);
            if window.len() > limit {
                window.pop_front();
            }
        }

        let mut entries: Vec<AgentMemoryEntry> = window.into_iter().collect();
        entries.reverse();
        Ok(entries)
    }
}

fn resolve_memory_file() -> PathBuf {
    if let Ok(path) = std::env::var("RUST_TOOLS_MEMORY_FILE") {
        let path = path.trim();
        if !path.is_empty() {
            return PathBuf::from(crate::commonw::utils::expanduser(path).as_ref());
        }
    }
    let cfg = crate::commonw::configw::get_all_config();
    let raw = cfg
        .get_opt("ai.memory.file")
        .unwrap_or_else(|| "~/.config/rust_tools/agent_memory.jsonl".to_string());
    PathBuf::from(crate::commonw::utils::expanduser(&raw).as_ref())
}
