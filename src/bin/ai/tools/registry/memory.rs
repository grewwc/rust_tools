use serde_json::Value;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::service::memory::{
    execute_memory_append, execute_memory_recent, execute_memory_search,
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
