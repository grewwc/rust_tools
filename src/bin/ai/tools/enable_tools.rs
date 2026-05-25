use std::sync::{LazyLock, Mutex};

use serde_json::Value;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::types::ToolDefinition;

static PENDING_ENABLE: LazyLock<Mutex<Vec<String>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

static PENDING_MCP_ENABLE: LazyLock<Mutex<Vec<String>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

static ACTIVE_TOOL_NAMES: LazyLock<Mutex<Vec<String>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

static EXPLICIT_ENABLED_TOOL_NAMES: LazyLock<Mutex<Vec<String>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

/// 每个 explicit-enabled tool 的"未使用计数"。
/// 每个 turn 末由调度器调用 `age_unused_explicit_tools` 老化：
///   - 在本 turn 被实际调用过的 tool → 计数清零
///   - 没用过 → 计数 +1，达到 EXPLICIT_TOOL_DEMOTE_AGE 就从 explicit list demote
/// 这样可以让"启用一次后永久焊接到所有 turn"的旧行为变成"用就保留，闲置就降级"，
/// 减少 prompt cache 失效与 tool schema 占用。
static EXPLICIT_TOOL_AGE: LazyLock<Mutex<rustc_hash::FxHashMap<String, u32>>> =
    LazyLock::new(|| Mutex::new(rustc_hash::FxHashMap::default()));

const EXPLICIT_TOOL_DEMOTE_AGE: u32 = 4;

static AVAILABLE_MCP_TOOLS: LazyLock<Mutex<Vec<ToolDefinition>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

pub(crate) fn set_active_tool_names(names: Vec<String>) {
    if let Ok(mut guard) = ACTIVE_TOOL_NAMES.lock() {
        *guard = names;
    }
}

pub(crate) fn set_available_mcp_tools(tools: Vec<ToolDefinition>) {
    if let Ok(mut guard) = AVAILABLE_MCP_TOOLS.lock() {
        *guard = tools;
    }
}

pub(crate) fn explicit_enabled_tool_names() -> Vec<String> {
    EXPLICIT_ENABLED_TOOL_NAMES
        .lock()
        .map(|guard| guard.clone())
        .unwrap_or_default()
}

/// 清空 explicit-enabled tool 列表。
/// 由 session 切换 / clear-history 等流程调用，避免上一 session 启用过的 tool
/// 永久焊接到后续所有 session 的请求 tools 数组（每个 schema 几百~上千 token，
/// 还会让 prompt cache 失效）。
pub(crate) fn clear_explicitly_enabled_tools() {
    if let Ok(mut guard) = EXPLICIT_ENABLED_TOOL_NAMES.lock() {
        guard.clear();
    }
    if let Ok(mut guard) = EXPLICIT_TOOL_AGE.lock() {
        guard.clear();
    }
}

/// 在 turn 末调用：把"本 turn 被实际调用过"的 explicit tool 计数清零，
/// 其它 explicit tool 计数 +1；超过 `EXPLICIT_TOOL_DEMOTE_AGE` 就从 explicit
/// list 中 demote。
///
/// 这是对"enable_tools 一旦启用就永久挂载"行为的温和约束：用就保留，闲置就降级。
pub(crate) fn age_unused_explicit_tools(used_in_turn: &[String]) {
    let Ok(mut names) = EXPLICIT_ENABLED_TOOL_NAMES.lock() else {
        return;
    };
    let Ok(mut ages) = EXPLICIT_TOOL_AGE.lock() else {
        return;
    };
    // 1. 老化或清零
    let mut to_remove: Vec<String> = Vec::new();
    for name in names.iter() {
        if used_in_turn.iter().any(|u| u == name) {
            ages.insert(name.clone(), 0);
        } else {
            let entry = ages.entry(name.clone()).or_insert(0);
            *entry = entry.saturating_add(1);
            if *entry >= EXPLICIT_TOOL_DEMOTE_AGE {
                to_remove.push(name.clone());
            }
        }
    }
    // 2. demote
    if !to_remove.is_empty() {
        names.retain(|n| !to_remove.contains(n));
        for n in &to_remove {
            ages.remove(n);
        }
    }
}

fn mark_explicitly_enabled_tools(names: &[String]) {
    if names.is_empty() {
        return;
    }
    if let Ok(mut guard) = EXPLICIT_ENABLED_TOOL_NAMES.lock() {
        for name in names {
            if !guard.contains(name) {
                guard.push(name.clone());
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn set_explicit_enabled_tool_names(names: Vec<String>) {
    if let Ok(mut guard) = EXPLICIT_ENABLED_TOOL_NAMES.lock() {
        *guard = names;
    }
}

pub(crate) fn drain_pending_mcp_names() -> Vec<String> {
    PENDING_MCP_ENABLE
        .lock()
        .map(|mut guard| guard.drain(..).collect())
        .unwrap_or_default()
}

pub(crate) fn drain_pending_enable() -> Vec<ToolDefinition> {
    let names: Vec<String> = PENDING_ENABLE
        .lock()
        .map(|mut guard| guard.drain(..).collect())
        .unwrap_or_default();
    if names.is_empty() {
        return Vec::new();
    }
    let mut defs = Vec::new();
    for reg in inventory::iter::<ToolRegistration> {
        if names.iter().any(|n| n == reg.spec.name) {
            defs.push(ToolDefinition {
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionDefinition {
                    name: reg.spec.name.to_string(),
                    description: reg.spec.description.to_string(),
                    parameters: (reg.spec.parameters)(),
                },
            });
        }
    }
    if let Ok(mut guard) = ACTIVE_TOOL_NAMES.lock() {
        for d in &defs {
            if !guard.contains(&d.function.name) {
                guard.push(d.function.name.clone());
            }
        }
    }
    defs
}

fn available_tools_not_active() -> Vec<(String, String)> {
    let active = ACTIVE_TOOL_NAMES
        .lock()
        .map(|g| g.clone())
        .unwrap_or_default();
    let mut result = Vec::new();
    for reg in inventory::iter::<ToolRegistration> {
        if !active.iter().any(|a| a == reg.spec.name) {
            result.push((reg.spec.name.to_string(), reg.spec.description.to_string()));
        }
    }
    let mcp_tools = AVAILABLE_MCP_TOOLS
        .lock()
        .map(|g| g.clone())
        .unwrap_or_default();
    for tool in mcp_tools {
        if !active.iter().any(|a| a == &tool.function.name) {
            result.push((tool.function.name, tool.function.description));
        }
    }
    result.sort_by(|a, b| a.0.cmp(&b.0));
    result.dedup_by(|a, b| a.0 == b.0);
    result
}

fn params_enable_tools() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["list", "enable"],
                "description": "list: show available but not yet loaded tools. enable: activate tools by name so they become available in subsequent calls."
            },
            "tools": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Tool names to enable (required for 'enable' operation)."
            }
        },
        "required": ["operation"]
    })
}

fn execute_enable_tools(args: &Value) -> Result<String, String> {
    let operation = args["operation"]
        .as_str()
        .ok_or("Missing 'operation' parameter")?;

    match operation {
        "list" => {
            let available = available_tools_not_active();
            if available.is_empty() {
                return Ok("All available tools are already loaded.".to_string());
            }
            let mut lines = Vec::with_capacity(available.len() + 1);
            lines.push(format!("{} additional tools available:", available.len()));
            for (name, desc) in available {
                let short = if desc.len() > 80 {
                    desc[..80].to_string()
                } else {
                    desc
                };
                lines.push(format!("  - {}: {}", name, short));
            }
            Ok(lines.join("\n"))
        }
        "enable" => {
            let tool_names: Vec<String> = args["tools"]
                .as_array()
                .ok_or("'enable' requires a 'tools' array")?
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            if tool_names.is_empty() {
                return Err("'tools' array is empty".to_string());
            }
            let active = ACTIVE_TOOL_NAMES
                .lock()
                .map(|g| g.clone())
                .unwrap_or_default();
            let mut known_builtin: Vec<&str> = Vec::new();
            for reg in inventory::iter::<ToolRegistration> {
                known_builtin.push(reg.spec.name);
            }
            let known_mcp: Vec<String> = AVAILABLE_MCP_TOOLS
                .lock()
                .map(|g| g.iter().map(|t| t.function.name.clone()).collect())
                .unwrap_or_default();
            let already: Vec<String> = tool_names
                .iter()
                .filter(|n| active.iter().any(|a| a == n.as_str()))
                .cloned()
                .collect();
            let unknown: Vec<String> = tool_names
                .iter()
                .filter(|n| {
                    !known_builtin.iter().any(|k| k == n)
                        && !known_mcp.iter().any(|k| k == n.as_str())
                })
                .cloned()
                .collect();
            let explicitly_requested: Vec<String> = tool_names
                .iter()
                .filter(|n| !unknown.iter().any(|u| u == *n))
                .cloned()
                .collect();
            let to_enable: Vec<String> = tool_names
                .into_iter()
                .filter(|n| !active.iter().any(|a| a == n.as_str()))
                .collect();
            let (mcp_names, builtin_names): (Vec<String>, Vec<String>) = to_enable
                .iter()
                .cloned()
                .partition(|n| n.starts_with("mcp_"));
            if let Ok(mut guard) = PENDING_ENABLE.lock() {
                for name in &builtin_names {
                    if !guard.contains(name) {
                        guard.push(name.clone());
                    }
                }
            }
            if let Ok(mut guard) = PENDING_MCP_ENABLE.lock() {
                for name in &mcp_names {
                    if !guard.contains(name) {
                        guard.push(name.clone());
                    }
                }
            }
            mark_explicitly_enabled_tools(&explicitly_requested);
            let mut msg = Vec::new();
            if !to_enable.is_empty() {
                msg.push(format!(
                    "Enabled {} tool(s): {}. They will be available in your next call.",
                    to_enable.len(),
                    to_enable.join(", ")
                ));
            }
            if !already.is_empty() {
                msg.push(format!("Already active: {}", already.join(", ")));
            }
            if !unknown.is_empty() {
                msg.push(format!("Unknown tools (ignored): {}", unknown.join(", ")));
            }
            Ok(msg.join("\n"))
        }
        other => Err(format!("Unknown operation '{}'. Use 'list' or 'enable'.", other)),
    }
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "enable_tools",
        description: "List or activate additional tools that are not loaded by default. Use 'list' to see available tools, 'enable' to activate specific tools by name. Enabled tools become available in subsequent calls. Use this when you need specialized capabilities like memory, knowledge base, undo, web browsing, or MCP server tools.",
        parameters: params_enable_tools,
        execute: execute_enable_tools,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
        groups: &["builtin", "core"],
    }
});
