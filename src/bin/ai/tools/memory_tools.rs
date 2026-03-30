use std::fs;
use std::path::Path;

use chrono::Local;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentMemoryEntry {
    timestamp: String,
    category: String,
    note: String,
    tags: Vec<String>,
    source: Option<String>,
}

fn params_memory_append() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "description": "Memory note text to store."
            },
            "category": {
                "type": "string",
                "description": "Category label for the note (default: \"general\")."
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for later retrieval."
            },
            "source": {
                "type": "string",
                "description": "Optional source/context string (e.g. URL, project name, ticket id)."
            }
        },
        "required": ["note"]
    })
}

fn params_memory_search() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword query (case-insensitive) matched against note/category/tags/source."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (1-50; default: 8)."
            }
        },
        "required": ["query"]
    })
}

fn params_memory_recent() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of entries to return (1-50; default: 8)."
            }
        }
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "memory_append",
        description: "Append a structured memory entry (timestamp/category/tags/source/note) to the agent memory store (JSONL).",
        parameters: params_memory_append,
        execute: execute_memory_append,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "memory_search",
        description: "Search the agent memory store for a keyword across note/category/tags/source and return recent matches.",
        parameters: params_memory_search,
        execute: execute_memory_search,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "memory_recent",
        description: "Show the most recent entries from the agent memory store.",
        parameters: params_memory_recent,
        execute: execute_memory_recent,
        groups: &["builtin"],
    }
});

fn resolve_memory_file() -> std::path::PathBuf {
    if let Ok(path) = std::env::var("RUST_TOOLS_MEMORY_FILE") {
        let path = path.trim();
        if !path.is_empty() {
            return std::path::PathBuf::from(crate::common::utils::expanduser(path).as_ref());
        }
    }
    let cfg = crate::common::configw::get_all_config();
    let raw = cfg
        .get_opt("ai.memory.file")
        .unwrap_or_else(|| "~/.config/rust_tools/agent_memory.jsonl".to_string());
    std::path::PathBuf::from(crate::common::utils::expanduser(&raw).as_ref())
}

fn parse_string_array(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn load_memory_entries(path: &Path) -> Result<Vec<AgentMemoryEntry>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let content =
        fs::read_to_string(path).map_err(|e| format!("Failed to read memory file: {e}"))?;
    let mut out = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(item) = serde_json::from_str::<AgentMemoryEntry>(line) {
            out.push(item);
        }
    }
    Ok(out)
}

fn render_memory_entries(entries: &[AgentMemoryEntry]) -> String {
    if entries.is_empty() {
        return "No memory entries found.".to_string();
    }
    let mut out = String::new();
    for (idx, entry) in entries.iter().enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        out.push_str(&format!(
            "{}. [{}] {}\n{}",
            idx + 1,
            entry.timestamp,
            entry.category,
            entry.note
        ));
        if !entry.tags.is_empty() {
            out.push_str(&format!("\nTags: {}", entry.tags.join(", ")));
        }
        if let Some(source) = &entry.source
            && !source.trim().is_empty()
        {
            out.push_str(&format!("\nSource: {}", source));
        }
    }
    out
}

pub(crate) fn execute_memory_append(args: &Value) -> Result<String, String> {
    let note = args["note"].as_str().ok_or("Missing note")?.trim();
    if note.is_empty() {
        return Err("note is empty".to_string());
    }
    let category = args["category"].as_str().unwrap_or("general").trim();
    let category = if category.is_empty() { "general" } else { category };
    let tags = parse_string_array(&args["tags"]);
    let source = args["source"]
        .as_str()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let entry = AgentMemoryEntry {
        timestamp: Local::now().to_rfc3339(),
        category: category.to_string(),
        note: note.to_string(),
        tags,
        source,
    };

    let path = resolve_memory_file();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create memory dir: {e}"))?;
    }
    let serialized =
        serde_json::to_string(&entry).map_err(|e| format!("Failed to serialize memory entry: {e}"))?;
    let mut existing = if path.exists() {
        fs::read_to_string(&path).map_err(|e| format!("Failed to read memory file: {e}"))?
    } else {
        String::new()
    };
    if !existing.is_empty() && !existing.ends_with('\n') {
        existing.push('\n');
    }
    existing.push_str(&serialized);
    existing.push('\n');
    fs::write(&path, existing).map_err(|e| format!("Failed to write memory file: {e}"))?;

    Ok(format!("Memory appended: {}", path.display()))
}

pub(crate) fn execute_memory_search(args: &Value) -> Result<String, String> {
    let query = args["query"].as_str().ok_or("Missing query")?.trim();
    if query.is_empty() {
        return Err("query is empty".to_string());
    }
    let query_lc = query.to_lowercase();
    let limit = args["limit"].as_u64().unwrap_or(8).clamp(1, 50) as usize;
    let path = resolve_memory_file();
    let mut entries = load_memory_entries(&path)?;
    entries.retain(|e| {
        e.note.to_lowercase().contains(&query_lc)
            || e.category.to_lowercase().contains(&query_lc)
            || e.tags.iter().any(|t| t.to_lowercase().contains(&query_lc))
            || e.source
                .as_ref()
                .is_some_and(|s| s.to_lowercase().contains(&query_lc))
    });
    entries.reverse();
    entries.truncate(limit);
    Ok(render_memory_entries(&entries))
}

pub(crate) fn execute_memory_recent(args: &Value) -> Result<String, String> {
    let limit = args["limit"].as_u64().unwrap_or(8).clamp(1, 50) as usize;
    let path = resolve_memory_file();
    let mut entries = load_memory_entries(&path)?;
    entries.reverse();
    entries.truncate(limit);
    Ok(render_memory_entries(&entries))
}
