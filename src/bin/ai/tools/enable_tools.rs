use std::sync::{LazyLock, RwLock};

use rust_tools::commonw::FastMap;
use rust_tools::cw::SkipSet;
use serde_json::Value;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::registry::tool_groups::ToolGroup;
use crate::ai::types::ToolDefinition;

type TurnIdentity = (String, usize);
type EnableOwner = (String, Option<usize>);

#[derive(Default)]
struct TurnEnableState {
    pending_enable: Vec<String>,
    pending_mcp_enable: Vec<String>,
    active_tool_names: Vec<String>,
    available_mcp_tools: Vec<ToolDefinition>,
    /// Whether the current turn's active agent/skills declare the
    /// executor tool group. Set by the driver when the skill turn
    /// guard is built; gates visibility of heavy execution primitives in the
    /// `enable_tools` catalog (hidden for agents that never declare the group).
    // Set when the current agent/skills declare a group that gates hidden
    // tools (see `registry::common::group_gates_hidden_tools`).
    hidden_group_declared: bool,
}

#[derive(Default)]
struct ExplicitEnableState {
    tool_names: Vec<String>,
    tool_age: FastMap<String, u32>,
}

/// active/catalog/pending 都是 turn 级状态，避免并发 Agent 覆盖或偷取消费。
/// 前台显式启用的工具按 session 保留以维持跨 turn 语义；子 Agent 则绑定其
/// 独立 turn，并在 turn 结束时清理。
#[derive(Default)]
struct EnableState {
    turns: FastMap<TurnIdentity, TurnEnableState>,
    explicit: FastMap<EnableOwner, ExplicitEnableState>,
}

static STATE: LazyLock<RwLock<EnableState>> = LazyLock::new(|| RwLock::new(EnableState::default()));

const EXPLICIT_TOOL_DEMOTE_AGE: u32 = 4;

fn clear_turn_state(s: &mut EnableState, turn: &TurnIdentity, owner: &EnableOwner) {
    s.turns.remove(turn);
    if owner.1.is_some() {
        s.explicit.remove(owner);
    }
}

/// 把 enable_tools 的 turn 级状态绑定到 driver turn future 的生命周期。
/// 即使 turn 提前报错、被 abort 或发生 unwind，Drop 也会清理对应条目。
#[must_use = "guard 必须持有到 turn future 结束"]
pub(crate) struct EnableTurnStateGuard {
    turn: TurnIdentity,
    owner: EnableOwner,
}

impl EnableTurnStateGuard {
    pub(crate) fn enter() -> Self {
        let turn = current_turn_identity();
        let owner = current_enable_owner(&turn);
        Self { turn, owner }
    }
}

impl Drop for EnableTurnStateGuard {
    fn drop(&mut self) {
        if let Ok(mut s) = STATE.write() {
            clear_turn_state(&mut s, &self.turn, &self.owner);
        }
    }
}

fn subagent_may_enable_tool(name: &str) -> bool {
    crate::ai::driver::runtime_ctx::current_subagent_depth() == 0
        || !super::is_subagent_orchestration_tool_name(name)
}

/// On-demand enablement only exposes generic `builtin` tools. Skill-specific
/// driver control tools are injected by the driver during an active-skill turn
/// and cannot be obtained through `enable_tools` in a normal turn.
fn registration_may_be_dynamically_enabled(reg: &ToolRegistration) -> bool {
    crate::ai::tools::registry::tool_metadata::tool_groups(reg.spec.name)
        .contains(&ToolGroup::Builtin)
}

/// Record whether the current turn's active agent/skills declare a group that
/// gates hidden tools (see `registry::common::group_gates_hidden_tools`). The
/// driver computes this from the same manifest source that decides the hidden
/// execution-primitive system-prompt catalog, so the `enable_tools` catalog
/// and that hint can never disagree.
pub(crate) fn set_hidden_group_declared(declared: bool) {
    let turn = current_turn_identity();
    if let Ok(mut s) = STATE.write() {
        s.turns.entry(turn).or_default().hidden_group_declared = declared;
    }
}

fn hidden_group_declared() -> bool {
    let turn = current_turn_identity();
    STATE
        .read()
        .ok()
        .and_then(|s| s.turns.get(&turn).map(|state| state.hidden_group_declared))
        .unwrap_or(false)
}

/// Whether a deferred heavy execution primitive may be enabled in the current
/// agent context. Hidden tools (spawn_process / spawn_daemon / shm_* / ...)
/// are only meaningful to agents that manage kernel processes, so for everyone
/// else they stay out of the catalog and are rejected by name; agents that
/// declare a hidden-gating group (see `group_gates_hidden_tools`) may enable
/// them. Core tools like apply_patch / read_file are never hidden, so they are
/// unaffected.
fn agent_may_enable_tool(name: &str) -> bool {
    !super::tool_defers_eager_load(name) || hidden_group_declared()
}

/// Whether a group name may be used as an `enable` shortcut (and appear in the
/// group-shortcut catalog line) for the current agent. Groups that gate hidden
/// tools (`group_gates_hidden_tools`) are privileged: for agents that never
/// declare one, the whole group is hidden — including any always-available
/// core members — so the catalog never advertises a system-level family the
/// agent cannot meaningfully load.
fn group_visible_to_agent(group: ToolGroup) -> bool {
    !super::group_gates_hidden_tools(group) || hidden_group_declared()
}

fn current_turn_identity() -> TurnIdentity {
    crate::ai::driver::runtime_ctx::TURN_IDENTITY
        .try_with(Clone::clone)
        .unwrap_or_default()
}

fn current_enable_owner(turn: &TurnIdentity) -> EnableOwner {
    let is_subagent = crate::ai::driver::runtime_ctx::current_subagent_depth() > 0
        || crate::ai::driver::runtime_ctx::has_subagent_result_slot();
    (turn.0.clone(), is_subagent.then_some(turn.1))
}

pub(crate) fn set_active_tool_names(names: Vec<String>) {
    let turn = current_turn_identity();
    if let Ok(mut s) = STATE.write() {
        s.turns.entry(turn).or_default().active_tool_names = names;
    }
}

pub(crate) fn set_available_mcp_tools(tools: Vec<ToolDefinition>) {
    let turn = current_turn_identity();
    if let Ok(mut s) = STATE.write() {
        s.turns.entry(turn).or_default().available_mcp_tools = tools;
    }
}

pub(crate) fn explicit_enabled_tool_names() -> Vec<String> {
    let turn = current_turn_identity();
    let owner = current_enable_owner(&turn);
    STATE
        .read()
        .ok()
        .and_then(|s| s.explicit.get(&owner).map(|state| state.tool_names.clone()))
        .unwrap_or_default()
}

/// 清空 explicit-enabled tool 列表。
/// 由 session 切换 / clear-history 等流程调用，避免上一 session 启用过的 tool
/// 永久焊接到后续所有 session 的请求 tools 数组（每个 schema 几百~上千 token，
/// 还会让 prompt cache 失效）。
pub(crate) fn clear_explicitly_enabled_tools(session_id: &str) {
    if let Ok(mut s) = STATE.write() {
        s.explicit.remove(&(session_id.to_string(), None));
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
    // 把 used_in_turn 收成 SkipSet，O(log n) 查询；调用方可能传 Vec/HashSet/迭代器。
    let used: SkipSet<String> = used_in_turn
        .into_iter()
        .map(|s| s.as_ref().to_string())
        .collect();
    let turn = current_turn_identity();
    let owner = current_enable_owner(&turn);
    let Ok(mut s) = STATE.write() else {
        return;
    };
    if let Some(explicit) = s.explicit.get_mut(&owner) {
        let mut to_remove: Vec<String> = Vec::new();
        for name in &explicit.tool_names {
            if used.contains(name) {
                explicit.tool_age.insert(name.clone(), 0);
            } else {
                let entry = explicit.tool_age.entry(name.clone()).or_insert(0);
                *entry = entry.saturating_add(1);
                if *entry >= EXPLICIT_TOOL_DEMOTE_AGE {
                    to_remove.push(name.clone());
                }
            }
        }
        if !to_remove.is_empty() {
            explicit.tool_names.retain(|n| !to_remove.contains(n));
            for name in &to_remove {
                explicit.tool_age.remove(name);
            }
        }
    }
    clear_turn_state(&mut s, &turn, &owner);
}

fn mark_explicitly_enabled_tools(s: &mut EnableState, owner: EnableOwner, names: &[String]) {
    if names.is_empty() {
        return;
    }
    let explicit = s.explicit.entry(owner).or_default();
    for name in names {
        if !explicit.tool_names.contains(name) {
            explicit.tool_names.push(name.clone());
        }
    }
}

#[cfg(test)]
pub(crate) fn set_explicit_enabled_tool_names(names: Vec<String>) {
    let turn = current_turn_identity();
    let owner = current_enable_owner(&turn);
    if let Ok(mut s) = STATE.write() {
        s.explicit.entry(owner).or_default().tool_names = names;
    }
}

pub(crate) fn drain_pending_mcp_names() -> Vec<String> {
    let turn = current_turn_identity();
    STATE
        .write()
        .ok()
        .and_then(|mut s| {
            s.turns
                .get_mut(&turn)
                .map(|state| std::mem::take(&mut state.pending_mcp_enable))
        })
        .unwrap_or_default()
}

pub(crate) fn drain_pending_enable() -> Vec<ToolDefinition> {
    let turn = current_turn_identity();
    let mut names: Vec<String> = match STATE.write() {
        Ok(mut s) => s
            .turns
            .get_mut(&turn)
            .map(|state| std::mem::take(&mut state.pending_enable))
            .unwrap_or_default(),
        Err(_) => return Vec::new(),
    };
    names.retain(|name| subagent_may_enable_tool(name) && agent_may_enable_tool(name));
    if names.is_empty() {
        return Vec::new();
    }
    let mut defs = Vec::new();
    for reg in inventory::iter::<ToolRegistration> {
        if registration_may_be_dynamically_enabled(reg) && names.iter().any(|n| n == reg.spec.name)
        {
            defs.push(ToolDefinition {
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionDefinition {
                    name: reg.spec.name.to_string(),
                    description: crate::ai::tools::registry::tool_metadata::tool_description(
                        reg.spec.name,
                        reg.spec.description,
                    ),
                    parameters: crate::ai::tools::registry::tool_metadata::tool_parameters(
                        reg.spec.name,
                    ),
                },
            });
        }
    }
    if let Ok(mut s) = STATE.write() {
        let state = s.turns.entry(turn).or_default();
        for d in &defs {
            if !state.active_tool_names.contains(&d.function.name) {
                state.active_tool_names.push(d.function.name.clone());
            }
        }
    }
    defs
}

fn available_tools_not_active() -> Vec<(String, String)> {
    let turn = current_turn_identity();
    let (active, mcp_tools) = STATE
        .read()
        .ok()
        .and_then(|s| {
            s.turns.get(&turn).map(|state| {
                (
                    state.active_tool_names.clone(),
                    state.available_mcp_tools.clone(),
                )
            })
        })
        .unwrap_or_default();
    let mut result = Vec::new();
    for reg in inventory::iter::<ToolRegistration> {
        if registration_may_be_dynamically_enabled(reg)
            && subagent_may_enable_tool(reg.spec.name)
            && agent_may_enable_tool(reg.spec.name)
            && !active.iter().any(|a| a == reg.spec.name)
        {
            result.push((
                reg.spec.name.to_string(),
                crate::ai::tools::registry::tool_metadata::tool_description(
                    reg.spec.name,
                    reg.spec.description,
                ),
            ));
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
                let short = if desc.chars().count() > 80 {
                    desc.chars().take(80).collect::<String>()
                } else {
                    desc.to_string()
                };
                lines.push(format!("  - {}: {}", name, short));
            }
            // One-line group catalog: the model can load a whole tool family
            // by passing the group name to 'enable' instead of listing every
            // member. Counts cover enable-able members only, matching what
            // the expansion actually loads.
            let mut group_counts: std::collections::BTreeMap<&str, usize> = Default::default();
            for reg in inventory::iter::<ToolRegistration> {
                if registration_may_be_dynamically_enabled(reg) {
                    for tag in crate::ai::tools::registry::tool_metadata::tool_groups(reg.spec.name)
                    {
                        if !tag.is_enable_ability_flag() && group_visible_to_agent(*tag) {
                            *group_counts.entry(tag.as_str()).or_insert(0) += 1;
                        }
                    }
                }
            }
            if !group_counts.is_empty() {
                let parts: Vec<String> = group_counts
                    .iter()
                    .map(|(group, count)| format!("{group}({count})"))
                    .collect();
                lines.push(format!(
                    "Group shortcuts (a group name may be passed to 'enable'): {}",
                    parts.join(", ")
                ));
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
            // Group shortcuts: an entry that matches no registered tool but
            // names a group expands to every enable-able tool carrying that
            // tag. `builtin` is the enable-ability flag rather than a loadable
            // unit and never expands, so a single entry can never dump the
            // whole catalog; per-name enables stay the fine-grained path.
            let mut expanded_from_groups: Vec<(String, usize)> = Vec::new();
            let mut resolved_names: Vec<String> = Vec::new();
            for entry in tool_names {
                let mut is_tool = false;
                for reg in inventory::iter::<ToolRegistration> {
                    if reg.spec.name == entry {
                        is_tool = true;
                        break;
                    }
                }
                if is_tool {
                    resolved_names.push(entry);
                    continue;
                }
                let mut members: Vec<String> = Vec::new();
                if let Some(group) = ToolGroup::from_name(&entry)
                    && !group.is_enable_ability_flag()
                    && group_visible_to_agent(group)
                {
                    for reg in inventory::iter::<ToolRegistration> {
                        let tags =
                            crate::ai::tools::registry::tool_metadata::tool_groups(reg.spec.name);
                        if tags.contains(&ToolGroup::Builtin) && tags.contains(&group) {
                            members.push(reg.spec.name.to_string());
                        }
                    }
                }
                if !members.is_empty() {
                    expanded_from_groups.push((entry.clone(), members.len()));
                    resolved_names.extend(members);
                } else {
                    // Not a tool and not a known group: keep as-is so the
                    // existing unknown/MCP handling below reports it.
                    resolved_names.push(entry);
                }
            }
            let tool_names = resolved_names;
            let blocked_in_subagent: Vec<String> = tool_names
                .iter()
                .filter(|name| !subagent_may_enable_tool(name))
                .cloned()
                .collect();
            let blocked_for_agent: Vec<String> = tool_names
                .iter()
                .filter(|name| !agent_may_enable_tool(name))
                .cloned()
                .collect();
            // 一次写锁内完成读 active/known_mcp + 写 pending_enable/pending_mcp_enable
            // + mark_explicitly_enabled，避免多次锁切换造成的状态拼接错位。
            let mut known_builtin: Vec<&str> = Vec::new();
            for reg in inventory::iter::<ToolRegistration> {
                if registration_may_be_dynamically_enabled(reg) {
                    known_builtin.push(reg.spec.name);
                }
            }
            let mut s = match STATE.write() {
                Ok(g) => g,
                Err(_) => return Err("enable_tools state poisoned".to_string()),
            };
            let turn = current_turn_identity();
            let owner = current_enable_owner(&turn);
            let state = s.turns.entry(turn.clone()).or_default();
            let active = state.active_tool_names.clone();
            let known_mcp: Vec<String> = state
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
                    !blocked_in_subagent.iter().any(|blocked| blocked == *n)
                        && !blocked_for_agent.iter().any(|blocked| blocked == *n)
                        && !known_builtin.iter().any(|k| k == n)
                        && !known_mcp.iter().any(|k| k == n.as_str())
                })
                .cloned()
                .collect();
            let explicitly_requested: Vec<String> = tool_names
                .iter()
                .filter(|n| !unknown.iter().any(|u| u == *n))
                .filter(|n| !blocked_in_subagent.iter().any(|blocked| blocked == *n))
                .filter(|n| !blocked_for_agent.iter().any(|blocked| blocked == *n))
                .cloned()
                .collect();
            let to_enable: Vec<String> = tool_names
                .into_iter()
                .filter(|n| !active.iter().any(|a| a == n.as_str()))
                .filter(|n| !blocked_in_subagent.iter().any(|blocked| blocked == n))
                .filter(|n| !blocked_for_agent.iter().any(|blocked| blocked == n))
                .filter(|n| {
                    known_builtin.iter().any(|k| k == n)
                        || known_mcp.iter().any(|k| k == n.as_str())
                })
                .collect();
            let (mcp_names, builtin_names): (Vec<String>, Vec<String>) = to_enable
                .iter()
                .cloned()
                .partition(|n| n.starts_with("mcp_"));
            for name in &builtin_names {
                if !state.pending_enable.contains(name) {
                    state.pending_enable.push(name.clone());
                }
            }
            for name in &mcp_names {
                if !state.pending_mcp_enable.contains(name) {
                    state.pending_mcp_enable.push(name.clone());
                }
            }
            mark_explicitly_enabled_tools(&mut s, owner, &explicitly_requested);
            drop(s);
            let mut msg = Vec::new();
            if !to_enable.is_empty() {
                msg.push(format!(
                    "Enabled {} tool(s): {}. They will be available in your next call.",
                    to_enable.len(),
                    to_enable.join(", ")
                ));
            }
            if !expanded_from_groups.is_empty() {
                let parts: Vec<String> = expanded_from_groups
                    .iter()
                    .map(|(group, count)| format!("{group} ({count} tools)"))
                    .collect();
                msg.push(format!("Expanded group shortcut(s): {}", parts.join(", ")));
            }
            if !already.is_empty() {
                msg.push(format!("Already active: {}", already.join(", ")));
            }
            if !unknown.is_empty() {
                msg.push(format!("Unknown tools (ignored): {}", unknown.join(", ")));
            }
            if !blocked_in_subagent.is_empty() {
                msg.push(format!(
                    "Unavailable in subagent context (ignored): {}",
                    blocked_in_subagent.join(", ")
                ));
            }
            if !blocked_for_agent.is_empty() {
                msg.push(format!(
                    "Unavailable for the current agent (ignored): {}",
                    blocked_for_agent.join(", ")
                ));
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
        description: "",

        execute: execute_enable_tools,
    }
});

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::ai::types::FunctionDefinition;

    fn reset_state_for_tests() {
        if let Ok(mut s) = STATE.write() {
            *s = EnableState::default();
        }
    }

    fn tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.to_string(),
                description: format!("{name} description"),
                parameters: json!({"type": "object"}),
            },
        }
    }

    #[test]
    fn list_includes_available_mcp_tools() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        reset_state_for_tests();
        set_active_tool_names(vec!["enable_tools".to_string()]);
        set_available_mcp_tools(vec![tool("mcp_feishu_docs_get_text_by_url")]);

        let output = execute_enable_tools(&json!({"operation": "list"})).unwrap();

        assert!(output.contains("mcp_feishu_docs_get_text_by_url"));
    }

    #[test]
    fn skill_discovery_tools_remain_dynamically_enabled() {
        // skill 发现/激活工具已从 core 组降级为 builtin-only：默认不随每轮
        // 常驻，但仍必须在 enable_tools 目录中可见、可按需启用，不能"隐形"。
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        reset_state_for_tests();
        set_active_tool_names(vec!["enable_tools".to_string()]);

        let output = execute_enable_tools(&json!({"operation": "list"})).unwrap();
        for name in ["activate_skill", "list_skills", "load_skill", "save_skill"] {
            assert!(
                output.contains(&format!("  - {name}:")),
                "{name} must remain discoverable via enable_tools"
            );
        }

        let enabled = execute_enable_tools(
            &json!({"operation": "enable", "tools": ["activate_skill", "list_skills"]}),
        )
        .unwrap();
        assert!(enabled.contains("Enabled 2 tool(s): activate_skill, list_skills"));
    }

    #[test]
    fn group_name_enable_expands_to_group_members() {
        // Group shortcut: an 'enable' entry naming a group expands to every
        // enable-able member of that group. `builtin` is the enable-ability
        // flag rather than a loadable unit, so it must stay unknown and can
        // never load the whole catalog in one entry.
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        reset_state_for_tests();
        set_active_tool_names(vec!["enable_tools".to_string()]);

        let enabled =
            execute_enable_tools(&json!({"operation": "enable", "tools": ["knowledge"]})).unwrap();
        assert!(
            enabled.contains("Expanded group shortcut(s): knowledge (7 tools)"),
            "{enabled}"
        );
        assert!(enabled.contains("Enabled 7 tool(s)"), "{enabled}");
        for name in [
            "knowledge_search",
            "knowledge_save",
            "knowledge_semantic_search",
        ] {
            assert!(
                enabled.contains(name),
                "{name} should be enabled via the knowledge group"
            );
        }

        let builtin =
            execute_enable_tools(&json!({"operation": "enable", "tools": ["builtin"]})).unwrap();
        assert!(
            builtin.contains("Unknown tools (ignored): builtin"),
            "{builtin}"
        );

        let listed = execute_enable_tools(&json!({"operation": "list"})).unwrap();
        assert!(listed.contains("Group shortcuts"), "{listed}");
        assert!(
            listed.contains("skills(") && listed.contains("task("),
            "{listed}"
        );
    }

    #[test]
    fn task_and_knowledge_tools_remain_dynamically_enabled() {
        // task 编排系列与 knowledge 记忆系列已从 core 组降级为 builtin-only：
        // 默认不随每轮常驻（省 token），但必须仍在 enable_tools 目录中可见、
        // 可按需启用，保证模型能渐进式发现，而不是"隐形"。
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        reset_state_for_tests();
        set_active_tool_names(vec!["enable_tools".to_string()]);

        let output = execute_enable_tools(&json!({"operation": "list"})).unwrap();
        for name in [
            "task",
            "task_spawn",
            "task_spawn_batch",
            "task_retry",
            "task_wait",
            "task_status",
            "task_evidence_read",
            "task_audit",
            "task_integrate",
            "task_cancel",
            "knowledge_save",
            "knowledge_search",
            "knowledge_list",
        ] {
            assert!(
                output.contains(&format!("  - {name}:")),
                "{name} must remain discoverable via enable_tools"
            );
        }

        let enabled = execute_enable_tools(
            &json!({"operation": "enable", "tools": ["task_spawn", "knowledge_search"]}),
        )
        .unwrap();
        assert!(enabled.contains("Enabled 2 tool(s): task_spawn, knowledge_search"));
    }

    #[test]
    fn executor_primitives_hidden_for_default_agent() {
        // Heavy execution primitives (executor group, non-core) only matter to
        // agents that declare the group: a default agent's enable catalog must
        // hide them entirely — no by-name enable and no group shortcut — or
        // the model just sees unusable system-level schema noise.
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        reset_state_for_tests();
        set_active_tool_names(vec!["enable_tools".to_string()]);

        let output = execute_enable_tools(&json!({"operation": "list"})).unwrap();
        for hidden in [
            "spawn_daemon",
            "spawn_process",
            "shm_read",
            "send_ipc_message",
            "set_env",
        ] {
            assert!(
                !output.contains(&format!("  - {hidden}:")),
                "{hidden} must not be listed for a default agent"
            );
        }
        // 组快捷键一行也不能宣传 executor 组（enable 它只会得到 Unknown）。
        assert!(!output.contains("executor("), "{output}");
        // 非原语工具不受影响：task 家族仍在目录中。
        assert!(output.contains("  - task_spawn:"), "{output}");

        let enabled =
            execute_enable_tools(&json!({"operation": "enable", "tools": ["spawn_daemon"]}))
                .unwrap();
        assert!(
            enabled.contains("Unavailable for the current agent (ignored): spawn_daemon"),
            "{enabled}"
        );
        assert!(drain_pending_enable().is_empty());
        assert!(explicit_enabled_tool_names().is_empty());

        let group =
            execute_enable_tools(&json!({"operation": "enable", "tools": ["executor"]})).unwrap();
        assert!(
            group.contains("Unknown tools (ignored): executor"),
            "{group}"
        );
    }

    #[test]
    fn executor_primitives_visible_when_group_declared() {
        // An agent declaring the executor group must still discover and enable
        // heavy execution primitives: visible in the catalog, enableable by
        // name, and the group shortcut expands.
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        reset_state_for_tests();
        set_active_tool_names(vec!["enable_tools".to_string()]);
        set_hidden_group_declared(true);

        let output = execute_enable_tools(&json!({"operation": "list"})).unwrap();
        assert!(output.contains("  - spawn_daemon:"), "{output}");
        assert!(output.contains("executor("), "{output}");

        let enabled =
            execute_enable_tools(&json!({"operation": "enable", "tools": ["spawn_daemon"]}))
                .unwrap();
        assert!(
            enabled.contains("Enabled 1 tool(s): spawn_daemon"),
            "{enabled}"
        );
        let drained = drain_pending_enable();
        assert!(drained.iter().any(|d| d.function.name == "spawn_daemon"));

        let group =
            execute_enable_tools(&json!({"operation": "enable", "tools": ["executor"]})).unwrap();
        assert!(
            group.contains("Expanded group shortcut(s): executor"),
            "{group}"
        );
    }

    #[test]
    fn enable_known_mcp_tool_queues_mcp_activation() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        reset_state_for_tests();
        set_active_tool_names(vec!["enable_tools".to_string()]);
        set_available_mcp_tools(vec![tool("mcp_feishu_docs_get_text_by_url")]);

        let output = execute_enable_tools(
            &json!({"operation": "enable", "tools": ["mcp_feishu_docs_get_text_by_url"]}),
        )
        .unwrap();
        let pending = drain_pending_mcp_names();

        assert!(output.contains("mcp_feishu_docs_get_text_by_url"));
        assert_eq!(pending, vec!["mcp_feishu_docs_get_text_by_url".to_string()]);
    }

    #[test]
    fn skill_control_tools_cannot_be_dynamically_enabled() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        reset_state_for_tests();
        set_active_tool_names(vec!["enable_tools".to_string()]);

        let listed = execute_enable_tools(&json!({"operation": "list"})).unwrap();
        assert!(
            !listed.contains("  - request_user_input:"),
            "skill-only control tools must not appear in the normal enable catalog"
        );

        let output =
            execute_enable_tools(&json!({"operation": "enable", "tools": ["request_user_input"]}))
                .unwrap();
        assert!(output.contains("Unknown tools (ignored): request_user_input"));
        assert!(drain_pending_enable().is_empty());
        assert!(explicit_enabled_tool_names().is_empty());
    }

    #[test]
    fn subagent_list_hides_task_orchestration_tools() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        reset_state_for_tests();
        set_active_tool_names(vec!["enable_tools".to_string()]);

        crate::ai::driver::runtime_ctx::SUBAGENT_DEPTH.sync_scope(1, || {
            let output = execute_enable_tools(&json!({"operation": "list"})).unwrap();

            for hidden in [
                "task",
                "task_spawn",
                "task_wait",
                "task_status",
                "task_integrate",
                "task_cancel",
            ] {
                assert!(
                    !output.contains(&format!("  - {hidden}:")),
                    "{hidden} should not be listed for subagents"
                );
            }
        });
    }

    #[test]
    fn subagent_enable_ignores_task_orchestration_tools() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        reset_state_for_tests();
        set_active_tool_names(vec!["enable_tools".to_string()]);

        crate::ai::driver::runtime_ctx::SUBAGENT_DEPTH.sync_scope(1, || {
            let output = execute_enable_tools(
                &json!({"operation": "enable", "tools": ["task_wait", "task_cancel"]}),
            )
            .unwrap();

            assert!(output.contains("Unavailable in subagent context"));
            assert!(output.contains("task_wait"));
            assert!(output.contains("task_cancel"));
            assert!(drain_pending_enable().is_empty());
            assert!(explicit_enabled_tool_names().is_empty());

            if let Ok(mut s) = STATE.write() {
                s.turns
                    .entry(current_turn_identity())
                    .or_default()
                    .pending_enable
                    .push("task_wait".to_string());
            }
            assert!(
                drain_pending_enable().is_empty(),
                "subagent drain must drop task orchestration tools queued by stale state"
            );
        });
    }

    #[test]
    fn concurrent_subagent_turns_keep_enable_state_isolated() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        reset_state_for_tests();
        let turn_a = ("shared-session".to_string(), 11);
        let turn_b = ("shared-session".to_string(), 12);

        crate::ai::driver::runtime_ctx::SUBAGENT_DEPTH.sync_scope(1, || {
            crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(turn_a.clone(), || {
                set_active_tool_names(vec!["enable_tools".to_string()]);
                set_available_mcp_tools(vec![tool("mcp_agent_a")]);
                execute_enable_tools(&json!({"operation": "enable", "tools": ["mcp_agent_a"]}))
                    .unwrap();
            });

            crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(turn_b, || {
                set_active_tool_names(vec!["enable_tools".to_string()]);
                set_available_mcp_tools(vec![tool("mcp_agent_b")]);
                let listed = execute_enable_tools(&json!({"operation": "list"})).unwrap();
                assert!(listed.contains("mcp_agent_b"));
                assert!(!listed.contains("mcp_agent_a"));
                assert!(drain_pending_mcp_names().is_empty());
                assert!(explicit_enabled_tool_names().is_empty());
            });

            crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(turn_a, || {
                assert_eq!(drain_pending_mcp_names(), vec!["mcp_agent_a".to_string()]);
                assert_eq!(
                    explicit_enabled_tool_names(),
                    vec!["mcp_agent_a".to_string()]
                );
            });
        });
    }

    #[test]
    fn enable_turn_state_guard_cleans_subagent_state_on_drop() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        reset_state_for_tests();
        let turn = ("guard-cleanup-session".to_string(), 21);
        let owner = (turn.0.clone(), Some(turn.1));

        crate::ai::driver::runtime_ctx::SUBAGENT_DEPTH.sync_scope(1, || {
            crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(turn.clone(), || {
                let turn_guard = EnableTurnStateGuard::enter();
                set_active_tool_names(vec!["enable_tools".to_string()]);
                set_explicit_enabled_tool_names(vec!["read_file".to_string()]);

                let s = STATE.read().unwrap_or_else(|poison| poison.into_inner());
                assert!(s.turns.contains_key(&turn));
                assert!(s.explicit.contains_key(&owner));
                drop(s);

                drop(turn_guard);
            });
        });

        let s = STATE.read().unwrap_or_else(|poison| poison.into_inner());
        assert!(!s.turns.contains_key(&turn));
        assert!(!s.explicit.contains_key(&owner));
    }
}
