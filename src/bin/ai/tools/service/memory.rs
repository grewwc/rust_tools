use chrono::Local;
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
    let store = MemoryStore::from_env_or_config();
    let entries = store.search(query, limit)?;
    Ok(render_memory_entries(&entries))
}

pub(crate) fn execute_memory_recent(args: &Value) -> Result<String, String> {
    let limit = args["limit"].as_u64().unwrap_or(8).clamp(1, 50) as usize;
    let store = MemoryStore::from_env_or_config();
    let entries = store.recent(limit)?;
    Ok(render_memory_entries(&entries))
}
