use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use crate::commonw::configw;
use super::with_memory_file_lock;
use crate::ai::tools::service::memory::{execute_memory_dedup, execute_memory_gc};
use serde_json::json;
use std::time::{SystemTime, Duration};
use std::ffi::OsStr;

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
        let mut window: VecDeque<AgentMemoryEntry> = VecDeque::with_capacity(limit + 1);
        let cfg = configw::get_all_config();
        let search_archives = cfg
            .get_opt("ai.memory.search_archives.enable")
            .unwrap_or_else(|| "false".to_string())
            .trim()
            .eq_ignore_ascii_case("true");
        let keep_last_archives = cfg
            .get_opt("ai.memory.search_archives.keep_last")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3);

        let mut files: Vec<PathBuf> = Vec::new();
        if search_archives {
            if let Some(parent) = self.path.parent() {
                let base = self
                    .path
                    .file_name()
                    .and_then(OsStr::to_str)
                    .unwrap_or("")
                    .to_string();
                let mut archives = Vec::new();
                for entry in fs::read_dir(parent).map_err(|e| format!("{}", e))? {
                    let entry = entry.map_err(|e| format!("{}", e))?;
                    let file_name = entry.file_name().to_str().unwrap_or("").to_string();
                    if !file_name.starts_with(&(base.clone() + ".")) {
                        continue;
                    }
                    let meta = entry.metadata().map_err(|e| format!("{}", e))?;
                    if !meta.is_file() {
                        continue;
                    }
                    let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                    archives.push((entry.path(), modified));
                }
                archives.sort_by_key(|(_, m)| *m);
                let take_from = archives.len().saturating_sub(keep_last_archives);
                for (p, _) in archives.into_iter().skip(take_from) {
                    files.push(p);
                }
            }
        }
        files.push(self.path.clone());

        for p in files {
            if !p.exists() {
                continue;
            }
            let file = fs::File::open(&p).map_err(|e| format!("Failed to read memory file: {e}"))?;
            let reader = BufReader::new(file);
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
                    || entry.tags.iter().any(|t| t.to_lowercase().contains(&query_lc))
                    || entry
                        .source
                        .as_ref()
                        .is_some_and(|s| s.to_lowercase().contains(&query_lc));
                if !matched {
                    continue;
                }
                window.push_back(entry);
                if window.len() > limit {
                    window.pop_front();
                }
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

impl MemoryStore {
    pub(crate) fn rotate_if_exceeds(&self, max_bytes: u64) -> Result<bool, String> {
        let path = self.path().to_path_buf();
        with_memory_file_lock(&path, || {
            let meta = std::fs::metadata(&path).ok();
            if let Some(meta) = meta {
                if meta.len() > max_bytes {
                    let ts = chrono::Local::now().format("%Y%m%d%H%M%S").to_string();
                    let mut new_name = path.clone();
                    let ext = new_name
                        .extension()
                        .and_then(|s| s.to_str())
                        .unwrap_or("jsonl")
                        .to_string();
                    new_name.set_extension(format!("{ext}.{}", ts));
                    std::fs::rename(&path, &new_name)
                        .map_err(|e| format!("Failed to rotate file: {}", e))?;
                    std::fs::File::create(&path)
                        .map_err(|e| format!("Failed to create new memory file: {}", e))?;
                    return Ok(true);
                }
            }
            Ok(false)
        })
    }

    pub(crate) fn maintain_after_append(&self) {
        let cfg = configw::get_all_config();
        let max_bytes = cfg
            .get_opt("ai.memory.auto_rotate.max_bytes")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(8 * 1024 * 1024);
        let gc_days = cfg
            .get_opt("ai.memory.auto_gc.days")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(30);
        let min_keep = cfg
            .get_opt("ai.memory.auto_gc.min_keep")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(200);
        let prob = cfg
            .get_opt("ai.memory.auto_maintain.probability")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.05);
        let rotated = self.rotate_if_exceeds(max_bytes).unwrap_or(false);
        let _ = if rotated {
            self.cleanup_archives_auto()
        } else {
            Ok(())
        };
        let roll = rand::random::<f64>();
        if roll < prob {
            let _ = execute_memory_dedup(&json!({}));
            let _ = execute_memory_gc(&json!({ "max_days": gc_days, "min_keep": min_keep }));
            let _ = self.cleanup_archives_auto();
        }
    }

    fn cleanup_archives_auto(&self) -> Result<(), String> {
        let cfg = configw::get_all_config();
        let retain_days = cfg
            .get_opt("ai.memory.archives.retain_days")
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(60);
        let keep_last = cfg
            .get_opt("ai.memory.archives.keep_last")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(10);
        let max_total = cfg
            .get_opt("ai.memory.archives.max_bytes")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(64 * 1024 * 1024);
        self.cleanup_archives(retain_days, keep_last, max_total)
    }

    pub(crate) fn cleanup_archives(
        &self,
        retain_days: i64,
        keep_last: usize,
        max_total_bytes: u64,
    ) -> Result<(), String> {
        let path = self.path().to_path_buf();
        let parent = match path.parent() {
            Some(p) => p.to_path_buf(),
            None => return Ok(()),
        };
        let base = match path.file_name().and_then(OsStr::to_str) {
            Some(s) => s.to_string(),
            None => return Ok(()),
        };

        let mut archives = Vec::new();
        for entry in std::fs::read_dir(&parent).map_err(|e| format!("{}", e))? {
            let entry = entry.map_err(|e| format!("{}", e))?;
            let file_name = entry
                .file_name()
                .to_str()
                .unwrap_or("")
                .to_string();
            if !file_name.starts_with(&(base.clone() + ".")) {
                continue;
            }
            let meta = entry.metadata().map_err(|e| format!("{}", e))?;
            if !meta.is_file() {
                continue;
            }
            let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            let size = meta.len();
            archives.push((entry.path(), modified, size));
        }

        if archives.is_empty() {
            return Ok(());
        }

        archives.sort_by_key(|(_, modified, _)| *modified);

        // Age-based cleanup
        if retain_days > 0 {
            let cutoff = SystemTime::now()
                .checked_sub(Duration::from_secs((retain_days as u64) * 86400))
                .unwrap_or(SystemTime::UNIX_EPOCH);
            for (p, m, _) in archives.clone() {
                if m < cutoff {
                    let _ = std::fs::remove_file(&p);
                }
            }
        }

        // Refresh list after potential deletions
        let mut archives2 = Vec::new();
        for (p, m, s) in archives.into_iter() {
            if p.exists() {
                archives2.push((p, m, s));
            }
        }
        if archives2.is_empty() {
            return Ok(());
        }
        archives2.sort_by_key(|(_, modified, _)| *modified);

        // Keep last N
        if archives2.len() > keep_last {
            let to_delete = archives2.len() - keep_last;
            for i in 0..to_delete {
                let (p, _, _) = &archives2[i];
                let _ = std::fs::remove_file(p);
            }
        }

        // Size cap
        let mut archives3: Vec<(std::path::PathBuf, SystemTime, u64)> = archives2
            .into_iter()
            .filter(|(p, _, _)| p.exists())
            .collect();
        archives3.sort_by_key(|(_, m, _)| *m);
        let mut total: u64 = archives3.iter().map(|(_, _, s)| *s).sum();
        let mut idx = 0usize;
        while total > max_total_bytes && idx < archives3.len() {
            let (p, _, s) = &archives3[idx];
            if std::fs::remove_file(p).is_ok() {
                total = total.saturating_sub(*s);
            }
            idx += 1;
        }
        Ok(())
    }
}
