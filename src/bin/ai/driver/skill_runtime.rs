use crate::ai::{
    mcp::McpClient,
    request,
    skills::SkillManifest,
    types::{App, ToolDefinition},
};
use crate::commonw::configw;
use std::ptr::NonNull;
use rust_tools::cw::SkipMap;
use chrono::{DateTime, Utc};

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

    pub(super) fn append_system_prompt(&mut self, extra: &str) {
        self.system_prompt.push_str(extra);
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
        let all_tools = reorder_tools_by_stats(builtin_tools, mcp_tools);
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

fn reorder_tools_by_stats(mut builtin: Vec<ToolDef>, mut mcp: Vec<ToolDef>) -> Vec<ToolDef> {
    let mut all = Vec::new();
    all.append(&mut builtin);
    all.append(&mut mcp);
    if all.is_empty() {
        return all;
    }
    let scores = load_tool_scores();
    all.sort_by(|a, b| {
        let sa = (*scores).get(&a.function.name).unwrap_or(0.0);
        let sb = (*scores).get(&b.function.name).unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.function.name.cmp(&b.function.name))
    });
    all
}

fn load_tool_scores() -> Box<SkipMap<String, f64>> {
    use crate::ai::tools::storage::memory_store::MemoryStore;
    let store = MemoryStore::from_env_or_config();
    let entries = store.recent(600).unwrap_or_default();
    let mut ok: Box<SkipMap<String, f64>> = SkipMap::new(16, |a: &String, b: &String| a.cmp(b) as i32);
    let mut err: Box<SkipMap<String, f64>> = SkipMap::new(16, |a: &String, b: &String| a.cmp(b) as i32);
    for e in entries {
        if e.category.to_lowercase() != "tool_stat" {
            continue;
        }
        if e.tags.is_empty() {
            continue;
        }
        let name = e.tags[0].clone();
        let is_ok = e.tags.iter().any(|t| t == "ok");
        let is_err = e.tags.iter().any(|t| t == "err");
        let weight = recency_weight(&e.timestamp);
        if is_ok {
            let cur = ok.get(&name).unwrap_or(0.0);
            ok.insert(name.clone(), cur + 1.0 * weight);
        }
        if is_err {
            let cur = err.get(&name).unwrap_or(0.0);
            err.insert(name.clone(), cur + 1.0 * weight);
        }
    }
    let mut score: Box<SkipMap<String, f64>> =
        SkipMap::new(16, |a: &String, b: &String| a.cmp(b) as i32);
    for (k, v) in (&*ok).into_iter() {
        let cur = score.get(k).unwrap_or(0.0);
        score.insert(k.clone(), cur + *v);
    }
    for (k, v) in (&*err).into_iter() {
        let cur = score.get(k).unwrap_or(0.0);
        score.insert(k.clone(), cur - 1.5 * *v);
    }
    score
}

fn recency_weight(ts: &str) -> f64 {
    let parsed: Option<DateTime<Utc>> = chrono::DateTime::parse_from_rfc3339(ts).ok().map(|dt| dt.with_timezone(&Utc));
    let Some(t) = parsed else { return 1.0; };
    let age_days = (Utc::now() - t).num_seconds().max(0) as f64 / 86400.0;
    f64::exp(-age_days / 14.0)
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
    system_prompt.push_str("\n\n");
    system_prompt.push_str("File editing policy:\n- When modifying an existing file or document, DO NOT rewrite the whole file unless the user explicitly asks for a full rewrite or the change truly affects most of the file.\n- First inspect the relevant region with read_file or read_file_lines, then use apply_patch to make the smallest localized edit that preserves the surrounding content.\n- Use write_file mainly for creating new files or for deliberate full-file replacement.\n- This rule applies equally to prose documents, markdown notes, configuration files, and source code.");
    system_prompt
}

pub(super) async fn prepare_skill_for_turn(
    app: &mut App,
    mcp_client: &McpClient,
    skill_manifests: &[SkillManifest],
    question: &str,
) -> SkillTurnGuard {
    let cfg = configw::get_all_config();
    let debug = cfg
        .get_opt("ai.skills.debug")
        .unwrap_or_default()
        .trim()
        .eq_ignore_ascii_case("true");
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

    let heuristic_skill = match_skill(skill_manifests, question, None);
    let router_skill = router_selected
        .as_deref()
        .and_then(|name| skill_manifests.iter().find(|s| s.name == name));
    let penalty_enabled = cfg
        .get_opt("ai.skills.penalty.enable")
        .unwrap_or_default()
        .trim()
        .eq_ignore_ascii_case("true");
    let mut skill = if penalty_enabled {
        match (router_skill, heuristic_skill) {
            (Some(r), Some(h)) => {
                let pr = super::tools::penalty_for_skill_tools(r);
                let ph = super::tools::penalty_for_skill_tools(h);
                if ph + 1e-6 < pr {
                    Some(h)
                } else {
                    Some(r)
                }
            }
            (Some(r), None) => Some(r),
            (None, Some(h)) => Some(h),
            (None, None) => None,
        }
    } else {
        router_skill.or(heuristic_skill)
    };
    if debug {
        if let Some(name) = router_selected.as_deref() {
            eprintln!("[skills] router selected: {}", name);
        }
        if let Some(s) = heuristic_skill.as_ref() {
            eprintln!("[skills] heuristic candidate: {}", s.name);
        }
        if let Some(s) = skill.as_ref() {
            eprintln!("[skills] final: {}", s.name);
        } else {
            eprintln!("[skills] final: <none>");
        }
    }
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

    let guard = SkillTurnGuard {
        app: NonNull::from(&mut *app),
        restore_agent_context,
        system_prompt: build_system_prompt(skill),
        matched_skill_name,
    };
    guard
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
