use std::sync::{LazyLock, RwLock};

use rust_tools::commonw::FastMap;
use rust_tools::cw::SkipSet;
use serde_json::Value;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};
use crate::ai::types::ToolDefinition;

type TurnIdentity = (String, usize);
type EnableOwner = (String, Option<usize>);

#[derive(Default)]
struct TurnEnableState {
    pending_enable: Vec<String>,
    pending_mcp_enable: Vec<String>,
    active_tool_names: Vec<String>,
    available_mcp_tools: Vec<ToolDefinition>,
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

/// 按需启用只暴露通用 builtin 工具。skill 专属的 driver control tool 只能在
/// active skill turn 由 driver 注入，不能通过普通 turn 的 `enable_tools` 获得。
fn registration_may_be_dynamically_enabled(reg: &ToolRegistration) -> bool {
    reg.spec.groups.contains(&"builtin")
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
    names.retain(|name| subagent_may_enable_tool(name));
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
            let blocked_in_subagent: Vec<String> = tool_names
                .iter()
                .filter(|name| !subagent_may_enable_tool(name))
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
                        && !known_builtin.iter().any(|k| k == n)
                        && !known_mcp.iter().any(|k| k == n.as_str())
                })
                .cloned()
                .collect();
            let explicitly_requested: Vec<String> = tool_names
                .iter()
                .filter(|n| !unknown.iter().any(|u| u == *n))
                .filter(|n| !blocked_in_subagent.iter().any(|blocked| blocked == *n))
                .cloned()
                .collect();
            let to_enable: Vec<String> = tool_names
                .into_iter()
                .filter(|n| !active.iter().any(|a| a == n.as_str()))
                .filter(|n| !blocked_in_subagent.iter().any(|blocked| blocked == n))
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
        groups: &["builtin", "core"],
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
