use crate::ai::{
    mcp::McpClient,
    request,
    skills::SkillManifest,
    types::{App, ToolDefinition},
};
use crate::common::configw;
use std::ptr::NonNull;

use super::{DEFAULT_MAX_ITERATIONS, OPENCLAW_MAX_ITERATIONS, match_skill};

type ToolDef = ToolDefinition;

pub(super) struct SkillTurnGuard {
    app: NonNull<App>,
    restore_agent_context: Option<(Vec<ToolDef>, usize)>,
    system_prompt: String,
    matched_skill_name: Option<String>,
}

impl SkillTurnGuard {
    pub(super) fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    pub(super) fn matched_skill_name(&self) -> Option<&str> {
        self.matched_skill_name.as_deref()
    }
}

impl Drop for SkillTurnGuard {
    fn drop(&mut self) {
        if let Some((tools, max_iterations)) = self.restore_agent_context.take() {
            let app = unsafe { self.app.as_mut() };
            if let Some(ctx) = app.agent_context.as_mut() {
                ctx.tools = tools;
                ctx.max_iterations = max_iterations;
            }
        }
    }
}

fn activate_skill_context(
    app: &mut App,
    builtin_tools: Vec<ToolDef>,
    mcp_tools: Vec<ToolDef>,
    openclaw_active: bool,
) -> Option<(Vec<ToolDef>, usize)> {
    let mut restore = None;
    if let Some(ctx) = app.agent_context.as_mut() {
        let mut all_tools = builtin_tools;
        all_tools.extend(mcp_tools);
        let prev_tools = std::mem::replace(&mut ctx.tools, all_tools);
        let prev_max_iterations = std::mem::replace(
            &mut ctx.max_iterations,
            if openclaw_active {
                OPENCLAW_MAX_ITERATIONS
            } else {
                DEFAULT_MAX_ITERATIONS
            },
        );
        restore = Some((prev_tools, prev_max_iterations));
    }
    restore
}

fn builtin_tools_for_skill(
    prompt_optimizer_active: bool,
    skill: Option<&SkillManifest>,
) -> Vec<ToolDef> {
    if prompt_optimizer_active {
        return Vec::new();
    }
    if let Some(skill) = skill {
        if !skill.tool_groups.is_empty() {
            let groups: Vec<&str> = skill.tool_groups.iter().map(|s| s.as_str()).collect();
            return super::super::tools::tool_definitions_for_groups(&groups);
        }
        if !skill.tools.is_empty() {
            return super::super::tools::get_tool_definitions_by_names(&skill.tools);
        }
    }
    super::super::tools::get_builtin_tool_definitions()
}

fn tool_uses_mcp_server(tool_name: &str, allowed_servers: &[String]) -> bool {
    if !tool_name.starts_with("mcp_") {
        return false;
    }

    let mut names = allowed_servers
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    rust_tools::sortw::stable_sort_by(&mut names, |a, b| b.len().cmp(&a.len()));

    names.into_iter().any(|server_name| {
        let prefix = format!("mcp_{server_name}_");
        tool_name
            .strip_prefix(&prefix)
            .is_some_and(|tool_part| !tool_part.is_empty())
    })
}

fn mcp_tools_for_skill(
    mcp_client: &McpClient,
    prompt_optimizer_active: bool,
    skill: Option<&SkillManifest>,
) -> Vec<ToolDef> {
    if prompt_optimizer_active {
        return Vec::new();
    }

    let all_tools = mcp_client.get_all_tools();
    let Some(skill) = skill else {
        return all_tools;
    };
    if skill.mcp_servers.is_empty() {
        return all_tools;
    }

    all_tools
        .into_iter()
        .filter(|tool| tool_uses_mcp_server(&tool.function.name, &skill.mcp_servers))
        .collect()
}

fn build_system_prompt(skill: Option<&SkillManifest>) -> String {
    let mut system_prompt = if let Some(skill) = skill {
        let mut p = "You are a helpful assistant.".to_string();
        p.push_str("\n\n");
        p.push_str("Skill enforcement:\n- You MUST follow the active skill instructions precisely.\n- Do not ignore, weaken, or bypass the skill behavior.\n- If the user request conflicts with the skill, ask a brief clarification aligned with the skill.");
        let extra = skill.build_system_prompt();
        if !extra.trim().is_empty() {
            p.push_str("\n\n");
            p.push_str(extra.trim());
        }
        p
    } else {
        "You are a helpful assistant.".to_string()
    };

    system_prompt.push_str("\n\n");
    system_prompt.push_str("Tool recovery mode:\n- If a tool call fails, read the error message and correct course before answering.\n- Prefer retrying with corrected arguments or switching to a more appropriate tool.\n- Do not repeat the exact same failing tool call unless the error indicates a transient retry is appropriate.\n- If a URL-based docs fetch tool says the URL is unsupported, switch to a search tool or ask for a supported docs URL instead of retrying the same call.\n- Only stop and ask the user when the error is ambiguous or missing required information.");
    system_prompt
}

pub(super) async fn prepare_skill_for_turn(
    app: &mut App,
    mcp_client: &McpClient,
    skill_manifests: &[SkillManifest],
    question: &str,
) -> SkillTurnGuard {
    let cfg = configw::get_all_config();
    let router_enabled = !cfg
        .get_opt("ai.skills.router")
        .unwrap_or_else(|| "true".to_string())
        .trim()
        .eq_ignore_ascii_case("false");

    let router_selected = if router_enabled {
        let model = app.current_model.clone();
        request::select_skill_via_model(app, &model, question, skill_manifests).await
    } else {
        None
    };

    let heuristic_skill = match_skill(skill_manifests, question);
    let router_skill = router_selected
        .as_deref()
        .and_then(|name| skill_manifests.iter().find(|s| s.name == name));
    let skill = router_skill.or(heuristic_skill);
    let matched_skill_name = skill.map(|s| s.name.clone());
    let prompt_optimizer_active = skill
        .as_ref()
        .is_some_and(|s| s.name.as_str() == "prompt-optimizer");
    let openclaw_active = skill.as_ref().is_some_and(|s| {
        s.name.as_str() == "openclaw" || s.tool_groups.iter().any(|g| g == "openclaw")
    });

    let builtin_tools = builtin_tools_for_skill(prompt_optimizer_active, skill);
    let mcp_tools = mcp_tools_for_skill(mcp_client, prompt_optimizer_active, skill);
    let restore_agent_context =
        activate_skill_context(app, builtin_tools, mcp_tools, openclaw_active);

    SkillTurnGuard {
        app: NonNull::from(&mut *app),
        restore_agent_context,
        system_prompt: build_system_prompt(skill),
        matched_skill_name,
    }
}

#[cfg(test)]
mod tests {
    use super::tool_uses_mcp_server;

    #[test]
    fn mcp_server_filter_matches_longest_server_name_prefix() {
        let allowed = vec!["foo".to_string(), "foo_bar".to_string()];
        assert!(tool_uses_mcp_server("mcp_foo_bar_search", &allowed));
        assert!(tool_uses_mcp_server("mcp_foo_lookup", &allowed));
        assert!(!tool_uses_mcp_server("mcp_bar_search", &allowed));
    }
}
