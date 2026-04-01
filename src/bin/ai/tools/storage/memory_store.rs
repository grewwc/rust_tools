use std::fs;
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
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create memory dir: {e}"))?;
        }
        let serialized = serde_json::to_string(entry)
            .map_err(|e| format!("Failed to serialize memory entry: {e}"))?;
        let mut existing = if self.path.exists() {
            fs::read_to_string(&self.path)
                .map_err(|e| format!("Failed to read memory file: {e}"))?
        } else {
            String::new()
        };
        if !existing.is_empty() && !existing.ends_with('\n') {
            existing.push('\n');
        }
        existing.push_str(&serialized);
        existing.push('\n');
        fs::write(&self.path, existing).map_err(|e| format!("Failed to write memory file: {e}"))?;
        Ok(())
    }

    pub(crate) fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<AgentMemoryEntry>, String> {
        let query_lc = query.to_lowercase();
        let mut entries = self.load_entries()?;
        entries.retain(|entry| {
            entry.note.to_lowercase().contains(&query_lc)
                || entry.category.to_lowercase().contains(&query_lc)
                || entry.tags.iter().any(|tag| tag.to_lowercase().contains(&query_lc))
                || entry
                    .source
                    .as_ref()
                    .is_some_and(|source| source.to_lowercase().contains(&query_lc))
        });
        entries.reverse();
        entries.truncate(limit);
        Ok(entries)
    }

    pub(crate) fn recent(&self, limit: usize) -> Result<Vec<AgentMemoryEntry>, String> {
        let mut entries = self.load_entries()?;
        entries.reverse();
        entries.truncate(limit);
        Ok(entries)
    }

    fn load_entries(&self) -> Result<Vec<AgentMemoryEntry>, String> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let content =
            fs::read_to_string(&self.path).map_err(|e| format!("Failed to read memory file: {e}"))?;
        let mut entries = Vec::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(item) = serde_json::from_str::<AgentMemoryEntry>(line) {
                entries.push(item);
            }
        }
        Ok(entries)
    }
}

fn resolve_memory_file() -> PathBuf {
    if let Ok(path) = std::env::var("RUST_TOOLS_MEMORY_FILE") {
        let path = path.trim();
        if !path.is_empty() {
            return PathBuf::from(crate::common::utils::expanduser(path).as_ref());
        }
    }
    let cfg = crate::common::configw::get_all_config();
    let raw = cfg
        .get_opt("ai.memory.file")
        .unwrap_or_else(|| "~/.config/rust_tools/agent_memory.jsonl".to_string());
    PathBuf::from(crate::common::utils::expanduser(&raw).as_ref())
}
