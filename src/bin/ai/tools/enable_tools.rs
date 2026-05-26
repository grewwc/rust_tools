use std::sync::{LazyLock, RwLock};

use serde_json::Value;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::types::ToolDefinition;

/// 把以前 6 把独立的 Mutex（pending_enable/pending_mcp_enable/active_tool_names/
/// explicit_enabled_tool_names/explicit_tool_age/available_mcp_tools）合并到单个
/// `RwLock<EnableState>`：
///   - 减少锁加解次数与潜在的锁顺序隐患；
///   - 读多写少的场景（available_tools_not_active 等）允许并发读；
///   - execute_enable_tools 是 SyncOnly，所有持锁路径都不跨 await，使用
///     std::sync::RwLock 安全。
#[derive(Default)]
struct EnableState {
    /// 等待激活的内置 tool 名称
    pending_enable: Vec<String>,
    /// 等待激活的 MCP tool 名称
    pending_mcp_enable: Vec<String>,
    /// 已经激活、出现在请求 tools 数组里的 tool 名称
    active_tool_names: Vec<String>,
    /// 通过 enable_tools 显式启用、应该跨 turn 保留的 tool 名称
    explicit_enabled_tool_names: Vec<String>,
    /// 每个 explicit-enabled tool 的"未使用计数"
    explicit_tool_age: rustc_hash::FxHashMap<String, u32>,
    /// MCP server 暴露但当前未必已启用的全部 tool 列表
    available_mcp_tools: Vec<ToolDefinition>,
}

static STATE: LazyLock<RwLock<EnableState>> = LazyLock::new(|| RwLock::new(EnableState::default()));

const EXPLICIT_TOOL_DEMOTE_AGE: u32 = 4;

pub(crate) fn set_active_tool_names(names: Vec<String>) {
    if let Ok(mut s) = STATE.write() {
        s.active_tool_names = names;
    }
}

pub(crate) fn set_available_mcp_tools(tools: Vec<ToolDefinition>) {
    if let Ok(mut s) = STATE.write() {
        s.available_mcp_tools = tools;
    }
}

pub(crate) fn explicit_enabled_tool_names() -> Vec<String> {
    STATE
        .read()
        .map(|s| s.explicit_enabled_tool_names.clone())
        .unwrap_or_default()
}

/// 清空 explicit-enabled tool 列表。
/// 由 session 切换 / clear-history 等流程调用，避免上一 session 启用过的 tool
/// 永久焊接到后续所有 session 的请求 tools 数组（每个 schema 几百~上千 token，
/// 还会让 prompt cache 失效）。
pub(crate) fn clear_explicitly_enabled_tools() {
    if let Ok(mut s) = STATE.write() {
        s.explicit_enabled_tool_names.clear();
        s.explicit_tool_age.clear();
    }
}

/// 在 turn 末调用：把"本 turn 被实际调用过"的 explicit tool 计数清零，
/// 其它 explicit tool 计数 +1；超过 `EXPLICIT_TOOL_DEMOTE_AGE` 就从 explicit
/// list 中 demote。
///
/// 这是对"enable_tools 一旦启用就永久挂载"行为的温和约束：用就保留，闲置就降级。
pub(crate) fn age_unused_explicit_tools<I, S>(used_in_turn: I)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    // 把 used_in_turn 收成 FxHashSet，O(1) 查询；调用方可能传 Vec/HashSet/迭代器。
    let used: rustc_hash::FxHashSet<String> = used_in_turn
        .into_iter()
        .map(|s| s.as_ref().to_string())
        .collect();
    let Ok(mut s) = STATE.write() else {
        return;
    };
    // 1. 老化或清零
    let mut to_remove: Vec<String> = Vec::new();
    // 这里需要同时可变借用 s.explicit_enabled_tool_names 和 s.explicit_tool_age，
    // 通过 split borrow 拆出来避免借用冲突。
    let EnableState {
        explicit_enabled_tool_names,
        explicit_tool_age,
        ..
    } = &mut *s;
    for name in explicit_enabled_tool_names.iter() {
        if used.contains(name) {
            explicit_tool_age.insert(name.clone(), 0);
        } else {
            let entry = explicit_tool_age.entry(name.clone()).or_insert(0);
            *entry = entry.saturating_add(1);
            if *entry >= EXPLICIT_TOOL_DEMOTE_AGE {
                to_remove.push(name.clone());
            }
        }
    }
    // 2. demote
    if !to_remove.is_empty() {
        explicit_enabled_tool_names.retain(|n| !to_remove.contains(n));
        for n in &to_remove {
            explicit_tool_age.remove(n);
        }
    }
}

fn mark_explicitly_enabled_tools(s: &mut EnableState, names: &[String]) {
    if names.is_empty() {
        return;
    }
    for name in names {
        if !s.explicit_enabled_tool_names.contains(name) {
            s.explicit_enabled_tool_names.push(name.clone());
        }
    }
}

#[cfg(test)]
pub(crate) fn set_explicit_enabled_tool_names(names: Vec<String>) {
    if let Ok(mut s) = STATE.write() {
        s.explicit_enabled_tool_names = names;
    }
}

pub(crate) fn drain_pending_mcp_names() -> Vec<String> {
    STATE
        .write()
        .map(|mut s| s.pending_mcp_enable.drain(..).collect())
        .unwrap_or_default()
}

pub(crate) fn drain_pending_enable() -> Vec<ToolDefinition> {
    let names: Vec<String> = match STATE.write() {
        Ok(mut s) => s.pending_enable.drain(..).collect(),
        Err(_) => return Vec::new(),
    };
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
    if let Ok(mut s) = STATE.write() {
        for d in &defs {
            if !s.active_tool_names.contains(&d.function.name) {
                s.active_tool_names.push(d.function.name.clone());
            }
        }
    }
    defs
}

fn available_tools_not_active() -> Vec<(String, String)> {
    let (active, mcp_tools) = STATE
        .read()
        .map(|s| (s.active_tool_names.clone(), s.available_mcp_tools.clone()))
        .unwrap_or_default();
    let mut result = Vec::new();
    for reg in inventory::iter::<ToolRegistration> {
        if !active.iter().any(|a| a == reg.spec.name) {
            result.push((reg.spec.name.to_string(), reg.spec.description.to_string()));
        }
    }
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
            // 一次写锁内完成读 active/known_mcp + 写 pending_enable/pending_mcp_enable
            // + mark_explicitly_enabled，避免多次锁切换造成的状态拼接错位。
            let mut known_builtin: Vec<&str> = Vec::new();
            for reg in inventory::iter::<ToolRegistration> {
                known_builtin.push(reg.spec.name);
            }
            let mut s = match STATE.write() {
                Ok(g) => g,
                Err(_) => return Err("enable_tools state poisoned".to_string()),
            };
            let active = s.active_tool_names.clone();
            let known_mcp: Vec<String> = s
                .available_mcp_tools
                .iter()
                .map(|t| t.function.name.clone())
                .collect();
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
            for name in &builtin_names {
                if !s.pending_enable.contains(name) {
                    s.pending_enable.push(name.clone());
                }
            }
            for name in &mcp_names {
                if !s.pending_mcp_enable.contains(name) {
                    s.pending_mcp_enable.push(name.clone());
                }
            }
            mark_explicitly_enabled_tools(&mut s, &explicitly_requested);
            drop(s);
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
        other => Err(format!(
            "Unknown operation '{}'. Use 'list' or 'enable'.",
            other
        )),
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
