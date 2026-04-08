use rust_tools::commonw::FastMap;
use std::sync::LazyLock;

use rust_tools::cw::SkipMap;
use serde_json::Value;

use crate::ai::tools::permissions::ToolPermissions;
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use crate::ai::types::{FunctionDefinition, ToolCall, ToolDefinition, ToolResult};
use chrono::Local;

/// Static specification for a builtin tool, including its name,
/// description, parameter schema, execution function, and group memberships.
#[derive(Clone, Copy)]
pub(crate) struct ToolSpec {
    pub(crate) name: &'static str,
    pub(crate) description: &'static str,
    pub(crate) parameters: fn() -> Value,
    pub(crate) execute: fn(&Value) -> Result<String, String>,
    pub(crate) groups: &'static [&'static str],
}

/// Registry entry submitted via `inventory!` to register a tool
/// at compile time for runtime discovery.
pub(crate) struct ToolRegistration {
    pub(crate) spec: ToolSpec,
}

inventory::collect!(ToolRegistration);

static TOOL_INDEX: LazyLock<FastMap<&'static str, &'static ToolSpec>> = LazyLock::new(|| {
    let mut index: FastMap<&'static str, &'static ToolSpec> = FastMap::default();
    for reg in inventory::iter::<ToolRegistration> {
        index.entry(reg.spec.name).or_insert(&reg.spec);
    }
    index
});

/// Returns tool definitions for all registered tools that belong
/// to at least one of the specified groups.
pub(crate) fn tool_definitions_for_groups(groups: &[&str]) -> Vec<ToolDefinition> {
    let mut tools: Box<SkipMap<String, ToolDefinition>> =
        SkipMap::new(16, |a: &String, b: &String| a.cmp(b) as i32);

    for reg in inventory::iter::<ToolRegistration> {
        if !reg
            .spec
            .groups
            .iter()
            .any(|g| groups.iter().any(|x| x == g))
        {
            continue;
        }
        let tool_def = ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: reg.spec.name.to_string(),
                description: reg.spec.description.to_string(),
                parameters: (reg.spec.parameters)(),
            },
        };
        tools.insert(tool_def.function.name.clone(), tool_def);
    }
    tools.into_iter().map(|(_, v)| v).collect()
}

pub(crate) fn get_tool_definitions_by_names(names: &[String]) -> Vec<ToolDefinition> {
    let mut tools: Box<SkipMap<String, ToolDefinition>> =
        SkipMap::new(16, |a: &String, b: &String| a.cmp(b) as i32);

    for name in names {
        let Some(spec) = TOOL_INDEX.get(name.as_str()).copied() else {
            continue;
        };
        let tool_def = ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: spec.name.to_string(),
                description: spec.description.to_string(),
                parameters: (spec.parameters)(),
            },
        };
        tools.insert(tool_def.function.name.clone(), tool_def);
    }
    tools.into_iter().map(|(_, v)| v).collect()
}

pub(crate) fn get_builtin_tool_definitions() -> Vec<ToolDefinition> {
    tool_definitions_for_groups(&["builtin"])
}

/// Executes a tool call by parsing its arguments and dispatching
/// to the registered tool implementation.
pub(crate) fn execute_tool_call(tool_call: &ToolCall) -> Result<ToolResult, String> {
    let raw_args = tool_call.function.arguments.trim();
    let args: Value = if raw_args.is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(raw_args).map_err(|e| format!("Failed to parse arguments: {}", e))?
    };

    execute_tool_call_with_args(&tool_call.id, &tool_call.function.name, &args)
}

pub(crate) fn execute_tool_call_with_args(
    tool_call_id: &str,
    name: &str,
    args: &Value,
) -> Result<ToolResult, String> {
    let Some(spec) = TOOL_INDEX.get(name).copied() else {
        record_tool_stat(name, false);
        return Err(format!("Unknown tool: {}", name));
    };
    let exec = (spec.execute)(args);
    match exec {
        Ok(result) => {
            record_tool_stat(name, true);
            Ok(ToolResult {
                tool_call_id: tool_call_id.to_string(),
                content: result,
            })
        }
        Err(err) => {
            record_tool_stat(name, false);
            Err(err)
        }
    }
}

fn record_tool_stat(name: &str, ok: bool) {
    let entry = AgentMemoryEntry {
        id: None,
        timestamp: Local::now().to_rfc3339(),
        category: "tool_stat".to_string(),
        note: format!("name={} result={}", name, if ok { "ok" } else { "err" }),
        tags: vec![
            name.to_string(),
            if ok {
                "ok".to_string()
            } else {
                "err".to_string()
            },
        ],
        source: None,
        priority: Some(50), // Normal priority: tool stats can be GC'd normally
    };
    let store = MemoryStore::from_env_or_config();
    let _ = store.append(&entry);
    store.maintain_after_append();
}

/// Executes a tool call with permission checking.
/// - If denied: returns an error immediately.
/// - If ask: prompts the user for confirmation before executing.
/// - If allowed: proceeds to execute directly.
pub(crate) fn execute_tool_call_with_permissions(
    tool_call: &ToolCall,
    permissions: &ToolPermissions,
) -> Result<ToolResult, String> {
    let tool_name = &tool_call.function.name;

    if permissions.is_denied(tool_name) {
        return Err(format!("Tool '{}' is denied by permissions", tool_name));
    }

    if permissions.needs_ask(tool_name) {
        let confirmed = crate::commonw::prompt::prompt_yes_or_no_interruptible(&format!(
            "Confirm tool execution: {} (y/n): ",
            tool_name
        ));
        if !confirmed.unwrap_or(false) {
            return Err(format!("Tool '{}' execution cancelled by user", tool_name));
        }
    }

    execute_tool_call(tool_call)
}
