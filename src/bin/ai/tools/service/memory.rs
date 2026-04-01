use chrono::{DateTime, Local, Utc};
use std::io::{BufRead, Write};
use serde_json::Value;

use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};

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

pub(crate) fn execute_memory_append(args: &Value) -> Result<String, String> {
    let note = args["note"].as_str().ok_or("Missing note")?.trim();
    if note.is_empty() {
        return Err("note is empty".to_string());
    }
    let category = args["category"].as_str().unwrap_or("general").trim();
    let tags = parse_string_array(&args["tags"]);
    let source = args["source"]
        .as_str()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let priority = args["priority"].as_u64().map(|p| p as u8);
    let entry = AgentMemoryEntry {
        timestamp: Local::now().to_rfc3339(),
        category: if category.is_empty() {
            "general".to_string()
        } else {
            category.to_string()
        },
        note: note.to_string(),
        tags,
        source,
        priority,
    };

    let store = MemoryStore::from_env_or_config();
    store.append(&entry)?;
    Ok(format!("Memory appended: {}", store.path().display()))
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
    let entries = store.search(query, 10_000)?;

    let mut scored = Vec::with_capacity(entries.len());
    for e in entries {
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
    Ok(render_memory_entries(&entries))
}

pub(crate) fn execute_memory_list_json(args: &Value) -> Result<String, String> {
    let limit = args["limit"].as_u64().unwrap_or(50).clamp(1, 200) as usize;
    let offset = args["offset"].as_u64().unwrap_or(0) as usize;
    let store = MemoryStore::from_env_or_config();
    let entries = store.recent(limit + offset)?;
    let sliced = if offset >= entries.len() {
        Vec::new()
    } else {
        entries.into_iter().skip(offset).collect::<Vec<_>>()
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
    use std::collections::HashSet;
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
        let mut seen: HashSet<(String, String, Vec<String>, Option<String>)> = HashSet::new();
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
        let tmp = path.with_extension("jsonl.tmp");
        {
            let mut f =
                std::fs::File::create(&tmp).map_err(|e| format!("Failed to create tmp: {}", e))?;
            for e in &deduped {
                let line = serde_json::to_string(e).map_err(|e| format!("{}", e))?;
                f.write_all(line.as_bytes())
                    .and_then(|_| f.write_all(b"\n"))
                    .map_err(|e| format!("Failed to write tmp: {}", e))?;
            }
        }
        std::fs::rename(&tmp, &path).map_err(|e| format!("Failed to replace memory file: {}", e))?;
        Ok("Dedup done".to_string())
    })
}

/// 用户主动保存记忆到全局 memory store
pub(crate) fn execute_memory_save(args: &Value) -> Result<String, String> {
    let content = args["content"].as_str().ok_or("Missing content")?.trim();
    if content.is_empty() {
        return Err("content is empty".to_string());
    }
    
    let category = args["category"].as_str().unwrap_or("user_memory").trim();
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
    
    let priority = args["priority"].as_u64().map(|p| p as u8).or(Some(150)); // Default high priority for user-directed memory
    let entry = AgentMemoryEntry {
        timestamp: Local::now().to_rfc3339(),
        category: if category.is_empty() { "user_memory".to_string() } else { category.to_string() },
        note: content.to_string(),
        tags,
        source,
        priority,
    };
    
    let store = MemoryStore::from_env_or_config();
    store.append(&entry)?;
    Ok(format!("Memory saved: {} (category: {})", store.path().display(), entry.category))
}
