use serde_json::Value;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::service::memory::{
    execute_memory_append, execute_memory_dedup, execute_memory_gc, execute_memory_list_json,
    execute_memory_recent, execute_memory_rotate, execute_memory_search, execute_memory_save,
    execute_memory_update,
};
use crate::ai::tools::service::knowledge_update::execute_knowledge_cache_manage;

fn params_memory_append() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "description": "Content to append to memory."
            },
            "category": {
                "type": "string",
                "description": "Category label (default: \"general\")."
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for categorization."
            },
            "source": {
                "type": "string",
                "description": "Optional source context."
            },
            "priority": {
                "type": "integer",
                "description": "Priority level 0-255. 255=permanent (never delete), 100-200=high, 50-99=normal, 0-49=low. Default: 100."
            }
        },
        "required": ["note"]
    })
}

fn params_memory_save() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Content to save to memory."
            },
            "category": {
                "type": "string",
                "description": "Category label (default: \"user_memory\")."
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for categorization."
            },
            "source": {
                "type": "string",
                "description": "Optional source context (e.g. user command, project name)."
            },
            "priority": {
                "type": "integer",
                "description": "Priority level 0-255. 255=permanent (never delete), 100-200=high, 50-99=normal, 0-49=low. Default: 150 for user-directed memory."
            }
        },
        "required": ["content"]
    })
}

fn params_memory_search() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query."
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default: 10)."
            },
            "category": {
                "type": "string",
                "description": "Filter by category."
            }
        },
        "required": ["query"]
    })
}

fn params_memory_update() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Memory entry id to update."
            },
            "content": {
                "type": "string",
                "description": "New memory content."
            },
            "category": {
                "type": "string",
                "description": "New category."
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New tags. Use an empty array to clear tags."
            },
            "source": {
                "type": ["string", "null"],
                "description": "New source. Use null or an empty string to clear source."
            },
            "priority": {
                "type": "integer",
                "description": "New priority level 0-255."
            }
        },
        "required": ["id"]
    })
}

fn params_memory_recent() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Number of recent entries (default: 10, max: 50)."
            }
        }
    })
}

fn params_memory_list_json() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max entries (default: 100)."
            }
        }
    })
}

fn params_memory_rotate() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "max_entries": {
                "type": "integer",
                "description": "Max entries to keep (default: 1000)."
            }
        }
    })
}

fn params_memory_gc() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "older_than_days": {
                "type": "integer",
                "description": "Remove entries older than N days (default: 90)."
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true, only report what would be removed."
            }
        }
    })
}

fn params_memory_dedup() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "dry_run": {
                "type": "boolean",
                "description": "If true, only report duplicates."
            }
        }
    })
}

fn params_knowledge_cache_manage() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action: stats, clear_volatile, refresh",
                "enum": ["stats", "clear_volatile", "refresh"]
            },
            "topic": {
                "type": "string",
                "description": "Topic to refresh (required for refresh action)."
            },
            "category": {
                "type": "string",
                "description": "Category for filtering."
            },
            "query": {
                "type": "string",
                "description": "Search query for refresh."
            },
            "limit": {
                "type": "string",
                "description": "Limit for refresh results."
            }
        },
        "required": ["action"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "memory_append",
        description: "Append content to the global memory store.",
        parameters: params_memory_append,
        execute: execute_memory_append,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "memory_save",
        description: "Save user-directed content to the global memory store with optional category and tags.",
        parameters: params_memory_save,
        execute: execute_memory_save,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "memory_search",
        description: "Search memory entries by keyword.",
        parameters: params_memory_search,
        execute: execute_memory_search,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "memory_update",
        description: "Update an existing memory entry by id.",
        parameters: params_memory_update,
        execute: execute_memory_update,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "memory_recent",
        description: "Get recent memory entries.",
        parameters: params_memory_recent,
        execute: execute_memory_recent,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "memory_list_json",
        description: "List memory entries as JSON.",
        parameters: params_memory_list_json,
        execute: execute_memory_list_json,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "memory_rotate",
        description: "Rotate memory file to keep only recent entries.",
        parameters: params_memory_rotate,
        execute: execute_memory_rotate,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "memory_gc",
        description: "Garbage collect old memory entries.",
        parameters: params_memory_gc,
        execute: execute_memory_gc,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "memory_dedup",
        description: "Remove duplicate memory entries.",
        parameters: params_memory_dedup,
        execute: execute_memory_dedup,
        groups: &["builtin"],
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "knowledge_cache_manage",
        description: "Manage knowledge cache: view stats, clear volatile cache, or force refresh.",
        parameters: params_knowledge_cache_manage,
        execute: execute_knowledge_cache_manage,
        groups: &["builtin"],
    }
});
