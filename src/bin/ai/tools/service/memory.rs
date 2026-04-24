use chrono::{DateTime, Local, Utc};
use std::io::{BufRead, Write};
use serde_json::Value;
use uuid::Uuid;
use rust_tools::commonw::FastSet;

use crate::ai::tools::os_tools::GLOBAL_OS;
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};

fn current_owner_tags() -> (Option<u64>, Option<u64>) {
    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os_arc) = guard.as_ref() {
            if let Ok(os) = os_arc.lock() {
                if let Some(pid) = os.current_process_id() {
                    let pgid = os.get_process(pid).and_then(|p| p.process_group);
                    return (Some(pid), pgid);
                }
            }
        }
    }
    (None, None)
}

fn is_memory_visible_to(entry: &AgentMemoryEntry, viewer_pid: Option<u64>) -> bool {
    let Some(viewer) = viewer_pid else {
        return true;
    };
    let Some(owner) = entry.owner_pid else {
        return true;
    };
    if owner == viewer {
        return true;
    }
    let Ok(guard) = GLOBAL_OS.lock() else {
        return false;
    };
    let Some(os_arc) = guard.as_ref() else {
        return false;
    };
    let Ok(os) = os_arc.lock() else {
        return false;
    };
    if let Some(entry_pgid) = entry.owner_pgid {
        if let Some(vpgid) = os.get_process(viewer).and_then(|p| p.process_group) {
            if entry_pgid == vpgid {
                return true;
            }
        }
    }
    let mut cursor = owner;
    while let Some(proc) = os.get_process(cursor) {
        if proc.parent_pid == Some(viewer) {
            return true;
        }
        match proc.parent_pid {
            Some(parent) => cursor = parent,
            None => break,
        }
    }
    false
}

struct ViewerContext {
    viewer_pid: Option<u64>,
    viewer_pgid: Option<u64>,
}

impl ViewerContext {
    fn current() -> Self {
        let (pid, pgid) = current_owner_tags();
        Self { viewer_pid: pid, viewer_pgid: pgid }
    }

    fn can_see(&self, entry: &AgentMemoryEntry) -> bool {
        let Some(viewer) = self.viewer_pid else { return true; };
        let Some(owner) = entry.owner_pid else { return true; };
        if owner == viewer { return true; }
        if let (Some(entry_pgid), Some(vpgid)) = (entry.owner_pgid, self.viewer_pgid) {
            if entry_pgid == vpgid { return true; }
        }
        let Ok(guard) = GLOBAL_OS.lock() else { return false; };
        let Some(os_arc) = guard.as_ref() else { return false; };
        let Ok(os) = os_arc.lock() else { return false; };
        let mut cursor = owner;
        while let Some(proc) = os.get_process(cursor) {
            if proc.parent_pid == Some(viewer) { return true; }
            match proc.parent_pid {
                Some(parent) => cursor = parent,
                None => break,
            }
        }
        false
    }
}

fn parse_string_array(value: &Value) -> Vec<String> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .map(|item| item.trim().to_string())
                .filter(|item| !item.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn render_memory_entries(entries: &[AgentMemoryEntry]) -> String {
    if entries.is_empty() {
        return "No memory entries found.".to_string();
    }
    let mut output = String::new();
    for (idx, entry) in entries.iter().enumerate() {
        if idx > 0 {
            output.push('\n');
        }
        output.push_str(&format!(
            "{}. [{}] {}\n{}",
            idx + 1,
            entry.timestamp,
            entry.category,
            entry.note
        ));
        if let Some(id) = &entry.id
            && !id.trim().is_empty()
        {
            output.push_str(&format!("\nID: {}", id));
        }
        if !entry.tags.is_empty() {
            output.push_str(&format!("\nTags: {}", entry.tags.join(", ")));
        }
        if let Some(source) = &entry.source
            && !source.trim().is_empty()
        {
            output.push_str(&format!("\nSource: {}", source));
        }
    }
    output
}

fn next_memory_id() -> String {
    format!("mem_{}", Uuid::new_v4().simple())
}

fn normalized_category(raw: Option<&str>, fallback: &str) -> String {
    let value = raw.unwrap_or(fallback).trim().to_lowercase();
    if value.is_empty() {
        fallback.to_string()
    } else {
        value
    }
}

fn default_priority_for_category(category: &str) -> u8 {
    match category {
        "common_sense" | "coding_guideline" | "best_practice" | "user_preference" | "preference" => {
            210
        }
        "safety_rules" => 255,
        _ => 150,
    }
}

fn parse_priority_arg(args: &Value, field: &str) -> Result<Option<u8>, String> {
    match args.get(field).and_then(|value| value.as_u64()) {
        Some(priority) if priority > u8::MAX as u64 => Err("priority out of range".to_string()),
        Some(priority) => Ok(Some(priority as u8)),
        None => Ok(None),
    }
}

fn load_memory_entries(path: &std::path::Path) -> Result<Vec<AgentMemoryEntry>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = std::io::BufReader::new(file);
    let mut entries = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read memory file: {}", e))?;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<AgentMemoryEntry>(&line) {
            entries.push(entry);
        }
    }
    Ok(entries)
}

fn write_memory_entries(
    path: &std::path::Path,
    entries: &[AgentMemoryEntry],
) -> Result<(), String> {
    let tmp = path.with_extension("jsonl.tmp");
    {
        let mut f =
            std::fs::File::create(&tmp).map_err(|e| format!("Failed to create tmp: {}", e))?;
        for entry in entries {
            let line = serde_json::to_string(entry).map_err(|e| format!("{}", e))?;
            f.write_all(line.as_bytes())
                .and_then(|_| f.write_all(b"\n"))
                .map_err(|e| format!("Failed to write tmp: {}", e))?;
        }
    }
    std::fs::rename(&tmp, path).map_err(|e| format!("Failed to replace memory file: {}", e))?;
    Ok(())
}

pub(crate) fn execute_memory_append(args: &Value) -> Result<String, String> {
    let note = args["note"].as_str().ok_or("Missing note")?.trim();
    if note.is_empty() {
        return Err("note is empty".to_string());
    }
    let category = normalized_category(args["category"].as_str(), "general");
    let tags = parse_string_array(&args["tags"]);
    let source = args["source"]
        .as_str()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let priority = parse_priority_arg(args, "priority")?
        .or_else(|| Some(default_priority_for_category(&category)));
    let (owner_pid, owner_pgid) = current_owner_tags();
    let entry = AgentMemoryEntry {
        id: Some(next_memory_id()),
        timestamp: Local::now().to_rfc3339(),
        category,
        note: note.to_string(),
        tags,
        source,
        priority,
        owner_pid,
        owner_pgid,
    };

    let store = MemoryStore::from_env_or_config();
    store.append(&entry)?;
    Ok(format!(
        "Memory appended: {} (id: {})",
        store.path().display(),
        entry.id.as_deref().unwrap_or("")
    ))
}

pub(crate) fn execute_memory_search(args: &Value) -> Result<String, String> {
    let query = args["query"].as_str().ok_or("Missing query")?.trim();
    if query.is_empty() {
        return Err("query is empty".to_string());
    }
    let limit = args["limit"].as_u64().unwrap_or(8).clamp(1, 50) as usize;
    let category_filter = args["category"].as_str().map(|s| s.trim().to_lowercase());
    let tags_any = parse_string_array(&args["tags_any"])
        .into_iter()
        .map(|s| s.to_lowercase())
        .collect::<Vec<_>>();
    let tags_all = parse_string_array(&args["tags_all"])
        .into_iter()
        .map(|s| s.to_lowercase())
        .collect::<Vec<_>>();
    let source_sub = args["source_substring"]
        .as_str()
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty());
    let debug_score = args["debug_score"].as_bool().unwrap_or(false);
    let store = MemoryStore::from_env_or_config();
    let results = store.search(query, 10_000)?;
    let viewer = ViewerContext::current();

    let mut scored = Vec::with_capacity(results.len());
    for (e, _search_score) in results {
        if !viewer.can_see(&e) {
            continue;
        }
        if let Some(cat) = category_filter.as_ref() {
            if e.category.to_lowercase() != *cat {
                continue;
            }
        }
        if !tags_any.is_empty()
            && !e
                .tags
                .iter()
                .any(|t| tags_any.iter().any(|x| t.to_lowercase() == *x))
        {
            continue;
        }
        if !tags_all.is_empty()
            && !tags_all
                .iter()
                .all(|x| e.tags.iter().any(|t| t.to_lowercase() == *x))
        {
            continue;
        }
        if let Some(sub) = source_sub.as_ref() {
            if !e
                .source
                .as_ref()
                .map(|s| s.to_lowercase().contains(sub))
                .unwrap_or(false)
            {
                continue;
            }
        }
        let mut score = 0.0_f64;
        let qlc = query.to_lowercase();
        if e.note.to_lowercase().contains(&qlc) {
            score += 3.0;
            score += (qlc.len() as f64).min(20.0) * 0.05;
        }
        if e.category.to_lowercase().contains(&qlc) {
            score += 1.5;
        }
        if e.tags.iter().any(|t| t.to_lowercase().contains(&qlc)) {
            score += 1.2;
        }
        if e.source
            .as_ref()
            .map(|s| s.to_lowercase().contains(&qlc))
            .unwrap_or(false)
        {
            score += 0.8;
        }
        let recency_bonus = parse_rfc3339_ts(&e.timestamp)
            .map(|ts| {
                let age_secs = (Utc::now() - ts).num_seconds().max(0) as f64;
                if age_secs <= 7.0 * 86400.0 {
                    1.0
                } else if age_secs <= 30.0 * 86400.0 {
                    0.3
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);
        score += recency_bonus;
        scored.push((score, e));
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut top_scored = scored;
    top_scored.truncate(limit);
    let top: Vec<AgentMemoryEntry> = top_scored.iter().map(|(_, e)| e.clone()).collect();
    let mut out = render_memory_entries(&top);
    if debug_score {
        out.push_str("\n");
        out.push_str("--- scores ---\n");
        for (idx, (s, e)) in top_scored.iter().enumerate() {
            out.push_str(&format!(
                "{}. score={:.2} [{}] {}\n",
                idx + 1,
                s,
                e.category,
                e.note
            ));
        }
    }
    Ok(out)
}

pub(crate) fn execute_memory_recent(args: &Value) -> Result<String, String> {
    let limit = args["limit"].as_u64().unwrap_or(8).clamp(1, 50) as usize;
    let store = MemoryStore::from_env_or_config();
    let entries = store.recent(limit)?;
    let viewer = ViewerContext::current();
    let visible: Vec<AgentMemoryEntry> = entries
        .into_iter()
        .filter(|e| viewer.can_see(e))
        .collect();
    Ok(render_memory_entries(&visible))
}

pub(crate) fn execute_memory_list_json(args: &Value) -> Result<String, String> {
    let limit = args["limit"].as_u64().unwrap_or(50).clamp(1, 200) as usize;
    let offset = args["offset"].as_u64().unwrap_or(0) as usize;
    let store = MemoryStore::from_env_or_config();
    let entries = store.recent(limit + offset)?;
    let viewer = ViewerContext::current();
    let visible: Vec<AgentMemoryEntry> = entries
        .into_iter()
        .filter(|e| viewer.can_see(e))
        .collect();
    let sliced = if offset >= visible.len() {
        Vec::new()
    } else {
        visible.into_iter().skip(offset).collect::<Vec<_>>()
    };
    serde_json::to_string(&sliced).map_err(|e| format!("{}", e))
}

pub(crate) fn execute_memory_rotate(args: &Value) -> Result<String, String> {
    let max_bytes = args["max_bytes"].as_u64().ok_or("Missing max_bytes")? as u64;
    let store = MemoryStore::from_env_or_config();
    let path = store.path().to_path_buf();
    super::super::storage::with_memory_file_lock(&path, || {
        let meta = std::fs::metadata(&path).ok();
        if let Some(meta) = meta {
            if meta.len() > max_bytes {
                let ts = Local::now().format("%Y%m%d%H%M%S").to_string();
                let mut new_name = path.clone();
                new_name.set_extension(format!("jsonl.{}", ts));
                std::fs::rename(&path, &new_name)
                    .map_err(|e| format!("Failed to rotate file: {}", e))?;
                std::fs::File::create(&path)
                    .map_err(|e| format!("Failed to create new memory file: {}", e))?;
                return Ok(format!(
                    "Rotated: {} -> {}",
                    path.display(),
                    new_name.display()
                ));
            }
        }
        Ok("Rotate skipped: size within limit".to_string())
    })
}

pub(crate) fn execute_memory_gc(args: &Value) -> Result<String, String> {
    let max_days = args["max_days"].as_u64().ok_or("Missing max_days")? as i64;
    let min_keep = args["min_keep"].as_u64().unwrap_or(200) as usize;
    let store = MemoryStore::from_env_or_config();
    let path = store.path().to_path_buf();
    super::super::storage::with_memory_file_lock(&path, || {
        if !path.exists() {
            return Ok("No memory file".to_string());
        }
        let file = std::fs::File::open(&path).map_err(|e| format!("Failed to open file: {}", e))?;
        let reader = std::io::BufReader::new(file);
        let mut entries = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read memory file: {}", e))?;
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }
            if let Ok(e) = serde_json::from_str::<AgentMemoryEntry>(&line) {
                entries.push(e);
            }
        }
        if entries.is_empty() {
            return Ok("No entries".to_string());
        }
        
        // Separate permanent entries (priority=255) - these are never deleted
        let mut permanent: Vec<AgentMemoryEntry> = entries
            .iter()
            .filter(|e| e.priority.unwrap_or(100) == 255)
            .cloned()
            .collect();
        let mut deletable: Vec<AgentMemoryEntry> = entries
            .into_iter()
            .filter(|e| e.priority.unwrap_or(100) != 255)
            .collect();
        
        let now = Utc::now();
        let cutoff_secs = max_days * 86400;
        
        // Apply age filter to deletable entries
        deletable.retain(|e| {
            parse_rfc3339_ts(&e.timestamp)
                .map(|ts| (now - ts).num_seconds() <= cutoff_secs)
                .unwrap_or(true)
        });
        
        // Sort deletable entries by priority (ascending) then by timestamp (ascending)
        // This ensures low priority and old entries are deleted first
        deletable.sort_by(|a, b| {
            let prio_a = a.priority.unwrap_or(100);
            let prio_b = b.priority.unwrap_or(100);
            prio_a.cmp(&prio_b).then_with(|| {
                let ts_a = parse_rfc3339_ts(&a.timestamp);
                let ts_b = parse_rfc3339_ts(&b.timestamp);
                ts_a.cmp(&ts_b)
            })
        });
        
        // Ensure minimum keep count (but never delete permanent entries)
        let total_permanent = permanent.len();
        if deletable.len() + total_permanent < min_keep {
            // Need to restore some entries, but prefer higher priority ones
            let all_entries = store.recent(min_keep)?;
            let new_deletable: Vec<AgentMemoryEntry> = all_entries
                .iter()
                .filter(|e| e.priority.unwrap_or(100) != 255)
                .cloned()
                .collect();
            let new_permanent: Vec<AgentMemoryEntry> = all_entries
                .iter()
                .filter(|e| e.priority.unwrap_or(100) == 255)
                .cloned()
                .collect();
            permanent = new_permanent;
            deletable = new_deletable;
        }
        
        // Combine permanent and deletable entries
        let permanent_count = permanent.len();
        let mut final_entries = permanent;
        final_entries.append(&mut deletable);
        
        let tmp = path.with_extension("jsonl.tmp");
        {
            let mut f =
                std::fs::File::create(&tmp).map_err(|e| format!("Failed to create tmp: {}", e))?;
            for e in &final_entries {
                let line = serde_json::to_string(e).map_err(|e| format!("{}", e))?;
                f.write_all(line.as_bytes())
                    .and_then(|_| f.write_all(b"\n"))
                    .map_err(|e| format!("Failed to write tmp: {}", e))?;
            }
        }
        std::fs::rename(&tmp, &path).map_err(|e| format!("Failed to replace memory file: {}", e))?;
        Ok(format!("GC done: {} entries kept (including {} permanent)", final_entries.len(), permanent_count))
    })
}

fn parse_rfc3339_ts(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

pub(crate) fn execute_memory_dedup(_args: &Value) -> Result<String, String> {
    let store = MemoryStore::from_env_or_config();
    let path = store.path().to_path_buf();
    super::super::storage::with_memory_file_lock(&path, || {
        if !path.exists() {
            return Ok("No memory file".to_string());
        }
        let entries = load_memory_entries(&path)?;
        if entries.is_empty() {
            return Ok("No entries".to_string());
        }
        let mut seen: FastSet<(String, String, Vec<String>, Option<String>)> = FastSet::default();
        let mut deduped: Vec<AgentMemoryEntry> = Vec::with_capacity(entries.len());
        for e in entries.into_iter().rev() {
            let key = (
                e.note.clone(),
                e.category.clone(),
                {
                    let mut t = e.tags.clone();
                    t.sort();
                    t
                },
                e.source.clone(),
            );
            if seen.insert(key) {
                deduped.push(e);
            }
        }
        deduped.reverse();
        write_memory_entries(&path, &deduped)?;
        Ok("Dedup done".to_string())
    })
}

pub(crate) fn execute_memory_update(args: &Value) -> Result<String, String> {
    let id = args["id"].as_str().ok_or("Missing id")?.trim();
    if id.is_empty() {
        return Err("id is empty".to_string());
    }

    let has_content = args.get("content").is_some();
    let has_category = args.get("category").is_some();
    let has_tags = args.get("tags").is_some();
    let has_source = args.get("source").is_some();
    let has_priority = args.get("priority").is_some();
    if !has_content && !has_category && !has_tags && !has_source && !has_priority {
        return Err("no fields to update".to_string());
    }

    let new_content = if has_content {
        let value = args["content"].as_str().ok_or("content must be a string")?.trim();
        if value.is_empty() {
            return Err("content is empty".to_string());
        }
        Some(value.to_string())
    } else {
        None
    };
    let new_category = if has_category {
        let value = args["category"]
            .as_str()
            .ok_or("category must be a string")?
            .trim();
        if value.is_empty() {
            return Err("category is empty".to_string());
        }
        Some(value.to_string())
    } else {
        None
    };
    let new_tags = if has_tags {
        Some(parse_string_array(&args["tags"]))
    } else {
        None
    };
    let new_source = if has_source {
        match args["source"].as_str() {
            Some(value) => {
                let value = value.trim();
                if value.is_empty() {
                    Some(None)
                } else {
                    Some(Some(value.to_string()))
                }
            }
            None if args["source"].is_null() => Some(None),
            None => return Err("source must be a string or null".to_string()),
        }
    } else {
        None
    };
    let new_priority = if has_priority {
        let priority = args["priority"]
            .as_u64()
            .ok_or("priority must be an integer")?;
        if priority > u8::MAX as u64 {
            return Err("priority out of range".to_string());
        }
        Some(priority as u8)
    } else {
        None
    };

    let store = MemoryStore::from_env_or_config();
    let path = store.path().to_path_buf();
    super::super::storage::with_memory_file_lock(&path, || {
        if !path.exists() {
            return Err("No memory file".to_string());
        }

        let mut entries = load_memory_entries(&path)?;

        let Some(entry) = entries
            .iter_mut()
            .rev()
            .find(|entry| entry.id.as_deref() == Some(id))
        else {
            return Err(format!("memory id not found: {id}"));
        };

        if let Some(content) = new_content.as_ref() {
            entry.note = content.clone();
        }
        if let Some(category) = new_category.as_ref() {
            entry.category = category.clone();
        }
        if let Some(tags) = new_tags.as_ref() {
            entry.tags = tags.clone();
        }
        if let Some(source) = new_source.as_ref() {
            entry.source = source.clone();
        }
        if let Some(priority) = new_priority {
            entry.priority = Some(priority);
        }
        entry.timestamp = Local::now().to_rfc3339();

        write_memory_entries(&path, &entries)?;

        Ok(format!("Memory updated: {} (id: {})", path.display(), id))
    })
}

pub(crate) fn execute_memory_delete(args: &Value) -> Result<String, String> {
    let id = args["id"].as_str().ok_or("Missing id")?.trim();
    if id.is_empty() {
        return Err("id is empty".to_string());
    }

    let store = MemoryStore::from_env_or_config();
    let path = store.path().to_path_buf();
    super::super::storage::with_memory_file_lock(&path, || {
        if !path.exists() {
            return Err("No memory file".to_string());
        }

        let mut entries = load_memory_entries(&path)?;
        let before_len = entries.len();
        entries.retain(|entry| entry.id.as_deref() != Some(id));
        if entries.len() == before_len {
            return Err(format!("memory id not found: {id}"));
        }

        write_memory_entries(&path, &entries)?;
        Ok(format!("Memory deleted: {} (id: {})", path.display(), id))
    })
}

/// 用户主动保存记忆到全局 memory store
pub(crate) fn execute_memory_save(args: &Value) -> Result<String, String> {
    let content = args["content"].as_str().ok_or("Missing content")?.trim();
    if content.is_empty() {
        return Err("content is empty".to_string());
    }

    let category = normalized_category(args["category"].as_str(), "user_memory");
    let tags = args["tags"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_else(|| vec!["user_directed".to_string()]);
    
    let source = args["source"]
        .as_str()
        .map(|s| s.trim().to_string())
        .or(Some("user_command".to_string()));

    let priority = parse_priority_arg(args, "priority")?
        .or_else(|| Some(default_priority_for_category(&category)));
    let (owner_pid, owner_pgid) = current_owner_tags();
    let entry = AgentMemoryEntry {
        id: Some(next_memory_id()),
        timestamp: Local::now().to_rfc3339(),
        category,
        note: content.to_string(),
        tags,
        source,
        priority,
        owner_pid,
        owner_pgid,
    };
    
    let store = MemoryStore::from_env_or_config();
    store.append(&entry)?;
    Ok(format!(
        "Memory saved: {} (category: {}, id: {})",
        store.path().display(),
        entry.category,
        entry.id.as_deref().unwrap_or("")
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        execute_memory_delete, execute_memory_list_json, execute_memory_save, execute_memory_update,
        is_memory_visible_to,
    };
    use crate::ai::os::kernel::{KernelInternal, Syscall};
    use crate::ai::test_support::ENV_LOCK;
    use crate::ai::tools::storage::memory_store::AgentMemoryEntry;
    use chrono::Local;

    #[test]
    fn memory_save_assigns_id_and_update_rewrites_entry() {
        let _guard = ENV_LOCK.lock().unwrap();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_memory_update_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let save_args = serde_json::json!({
            "content": "delete carefully",
            "category": "safety_rules",
            "tags": ["safety"],
            "source": "test",
            "priority": 255
        });
        let save_msg = execute_memory_save(&save_args).unwrap();
        assert!(save_msg.contains("id: mem_"));

        let list_before = execute_memory_list_json(&serde_json::json!({ "limit": 10 })).unwrap();
        let before: serde_json::Value = serde_json::from_str(&list_before).unwrap();
        let id = before[0]["id"].as_str().unwrap().to_string();
        assert_eq!(before[0]["note"].as_str().unwrap(), "delete carefully");

        let update_args = serde_json::json!({
            "id": id,
            "content": "delete carefully after confirmation",
            "priority": 200,
            "tags": ["safety", "confirmed"]
        });
        let update_msg = execute_memory_update(&update_args).unwrap();
        assert!(update_msg.contains("Memory updated"));

        let list_after = execute_memory_list_json(&serde_json::json!({ "limit": 10 })).unwrap();
        let after: serde_json::Value = serde_json::from_str(&list_after).unwrap();
        assert_eq!(
            after[0]["note"].as_str().unwrap(),
            "delete carefully after confirmation"
        );
        assert_eq!(after[0]["priority"].as_u64().unwrap(), 200);
        assert_eq!(after[0]["tags"].as_array().unwrap().len(), 2);

        let _ = std::fs::remove_file(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    #[test]
    fn memory_save_common_sense_defaults_to_persistent_priority_and_can_delete() {
        let _guard = ENV_LOCK.lock().unwrap();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("rt_memory_delete_{ts}.jsonl"));
        unsafe {
            std::env::set_var("RUST_TOOLS_MEMORY_FILE", &path);
        }

        let save_args = serde_json::json!({
            "content": "Files should end with a trailing newline",
            "category": "common_sense",
            "tags": ["editing"]
        });
        execute_memory_save(&save_args).unwrap();

        let list_before = execute_memory_list_json(&serde_json::json!({ "limit": 10 })).unwrap();
        let before: serde_json::Value = serde_json::from_str(&list_before).unwrap();
        let id = before[0]["id"].as_str().unwrap().to_string();
        assert_eq!(before[0]["category"].as_str().unwrap(), "common_sense");
        assert_eq!(before[0]["priority"].as_u64().unwrap(), 210);

        let delete_args = serde_json::json!({ "id": id });
        let delete_msg = execute_memory_delete(&delete_args).unwrap();
        assert!(delete_msg.contains("Memory deleted"));

        let list_after = execute_memory_list_json(&serde_json::json!({ "limit": 10 })).unwrap();
        let after: serde_json::Value = serde_json::from_str(&list_after).unwrap();
        assert!(after.as_array().unwrap().is_empty());

        let _ = std::fs::remove_file(&path);
        unsafe {
            std::env::remove_var("RUST_TOOLS_MEMORY_FILE");
        }
    }

    #[test]
    fn memory_visibility_unowned_entries_visible_to_all() {
        let entry = AgentMemoryEntry {
            id: None,
            timestamp: Local::now().to_rfc3339(),
            category: "general".to_string(),
            note: "public note".to_string(),
            tags: vec![],
            source: None,
            priority: Some(100),
            owner_pid: None,
            owner_pgid: None,
        };
        assert!(is_memory_visible_to(&entry, None));
        assert!(is_memory_visible_to(&entry, Some(1)));
        assert!(is_memory_visible_to(&entry, Some(999)));
    }

    #[test]
    fn memory_visibility_owner_sees_own_entries() {
        let entry = AgentMemoryEntry {
            id: None,
            timestamp: Local::now().to_rfc3339(),
            category: "general".to_string(),
            note: "my note".to_string(),
            tags: vec![],
            source: None,
            priority: Some(100),
            owner_pid: Some(42),
            owner_pgid: None,
        };
        assert!(is_memory_visible_to(&entry, Some(42)));
        assert!(!is_memory_visible_to(&entry, Some(99)));
    }

    #[test]
    fn memory_visibility_foreground_sees_all() {
        let entry = AgentMemoryEntry {
            id: None,
            timestamp: Local::now().to_rfc3339(),
            category: "general".to_string(),
            note: "tagged note".to_string(),
            tags: vec![],
            source: None,
            priority: Some(100),
            owner_pid: Some(42),
            owner_pgid: Some(10),
        };
        assert!(is_memory_visible_to(&entry, None));
    }

    #[test]
    fn memory_visibility_same_process_group() {
        let _guard = ENV_LOCK.lock().unwrap();
        let kernel = crate::ai::driver::new_local_kernel();
        let (root, child_a, child_b) = {
            let mut os = kernel.lock().unwrap();
            let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
            let child_a = os.spawn(Some(root), "a".to_string(), "goal a".to_string(), 20, 4, None, None).unwrap();
            let child_b = os.spawn(Some(root), "b".to_string(), "goal b".to_string(), 20, 4, None, None).unwrap();
            os.set_process_group(child_a, 100).unwrap();
            os.set_process_group(child_b, 100).unwrap();
            (root, child_a, child_b)
        };
        crate::ai::tools::os_tools::init_os_tools_globals(kernel.clone());

        let entry_a = AgentMemoryEntry {
            id: None,
            timestamp: Local::now().to_rfc3339(),
            category: "general".to_string(),
            note: "a's note".to_string(),
            tags: vec![],
            source: None,
            priority: Some(100),
            owner_pid: Some(child_a),
            owner_pgid: Some(100),
        };

        assert!(is_memory_visible_to(&entry_a, Some(child_b)));

        {
            if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
                *guard = None;
            }
        }
    }

    #[test]
    fn memory_visibility_ancestor_sees_descendant() {
        let _guard = ENV_LOCK.lock().unwrap();
        let kernel = crate::ai::driver::new_local_kernel();
        let (root, child) = {
            let mut os = kernel.lock().unwrap();
            let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
            let child = os.spawn(Some(root), "child".to_string(), "goal child".to_string(), 20, 4, None, None).unwrap();
            (root, child)
        };
        crate::ai::tools::os_tools::init_os_tools_globals(kernel.clone());

        let child_entry = AgentMemoryEntry {
            id: None,
            timestamp: Local::now().to_rfc3339(),
            category: "general".to_string(),
            note: "child's secret".to_string(),
            tags: vec![],
            source: None,
            priority: Some(100),
            owner_pid: Some(child),
            owner_pgid: None,
        };

        assert!(is_memory_visible_to(&child_entry, Some(root)));

        {
            if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
                *guard = None;
            }
        }
    }

    #[test]
    fn memory_visibility_unrelated_process_blocked() {
        let _guard = ENV_LOCK.lock().unwrap();
        let kernel = crate::ai::driver::new_local_kernel();
        let (child_a, child_b) = {
            let mut os = kernel.lock().unwrap();
            let root1 = os.begin_foreground("fg1".to_string(), "goal1".to_string(), 10, usize::MAX, None);
            let child_a = os.spawn(Some(root1), "a".to_string(), "goal a".to_string(), 20, 4, None, None).unwrap();

            let root2 = os.begin_foreground("fg2".to_string(), "goal2".to_string(), 10, usize::MAX, None);
            let child_b = os.spawn(Some(root2), "b".to_string(), "goal b".to_string(), 20, 4, None, None).unwrap();
            (child_a, child_b)
        };
        crate::ai::tools::os_tools::init_os_tools_globals(kernel.clone());

        let entry_a = AgentMemoryEntry {
            id: None,
            timestamp: Local::now().to_rfc3339(),
            category: "general".to_string(),
            note: "a's private note".to_string(),
            tags: vec![],
            source: None,
            priority: Some(100),
            owner_pid: Some(child_a),
            owner_pgid: None,
        };

        assert!(!is_memory_visible_to(&entry_a, Some(child_b)));

        {
            if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
                *guard = None;
            }
        }
    }
}
