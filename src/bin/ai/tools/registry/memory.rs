use serde_json::Value;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::service::memory::{
    execute_memory_append, execute_memory_dedup, execute_memory_gc, execute_memory_list_json,
    execute_memory_recent, execute_memory_rotate, execute_memory_search,
};

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
            },
            "category": {
                "type": "string",
                "description": "Optional category filter (exact match)."
            },
            "tags_any": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags filter: any of these tags."
            },
            "tags_all": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags filter: all of these tags."
            },
            "source_substring": {
                "type": "string",
                "description": "Optional substring to match in source."
            },
            "debug_score": {
                "type": "boolean",
                "description": "If true, append a score line per entry for debugging ranking."
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

fn params_memory_list_json() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of entries to return (1-200; default: 50)."
            },
            "offset": {
                "type": "integer",
                "description": "Skip the first N most recent entries (default: 0)."
            }
        }
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "memory_list_json",
        description: "Return recent memory entries as a JSON array (for programmatic use).",
        parameters: params_memory_list_json,
        execute: execute_memory_list_json,
        groups: &["builtin"],
    }
});

fn params_memory_dedup() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {}
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "memory_dedup",
        description: "Deduplicate the memory file by (note,category,tags,source), keeping the most recent for each key.",
        parameters: params_memory_dedup,
        execute: execute_memory_dedup,
        groups: &["builtin"],
    }
});

fn params_memory_rotate() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "max_bytes": {
                "type": "integer",
                "description": "Rotate if file size exceeds this many bytes."
            }
        },
        "required": ["max_bytes"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "memory_rotate",
        description: "Rotate the memory jsonl file if its size exceeds max_bytes; archive current file and create an empty one.",
        parameters: params_memory_rotate,
        execute: execute_memory_rotate,
        groups: &["builtin"],
    }
});

fn params_memory_gc() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "max_days": {
                "type": "integer",
                "description": "Remove entries older than this number of days."
            },
            "min_keep": {
                "type": "integer",
                "description": "Keep at least this many recent entries even if older than max_days (default: 200)."
            }
        },
        "required": ["max_days"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "memory_gc",
        description: "Garbage collect old memory entries: keep recent entries and drop entries older than max_days.",
        parameters: params_memory_gc,
        execute: execute_memory_gc,
        groups: &["builtin"],
    }
});
