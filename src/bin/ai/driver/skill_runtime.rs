use crate::ai::{
    agents::AgentManifest,
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
use super::intent_recognition::{self, UserIntent};

type ToolDef = ToolDefinition;

pub(super) struct SkillTurnGuard {
    app: NonNull<App>,
    restore_agent_context: Option<(Vec<ToolDef>, usize)>,
    system_prompt: String,
    matched_skill_name: Option<String>,
    intent: UserIntent,
    skip_recall_by_skill: bool,
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

    pub(super) fn intent(&self) -> &UserIntent {
        &self.intent
    }

    pub(super) fn skip_recall_by_skill(&self) -> bool {
        self.skip_recall_by_skill
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
    max_iterations: usize,
) -> Option<(Vec<ToolDef>, usize)> {
    let mut restore = None;
    if let Some(ctx) = app.agent_context.as_mut() {
        let all_tools = reorder_tools_by_stats(builtin_tools, mcp_tools);
        let prev_tools = std::mem::replace(&mut ctx.tools, all_tools);
        let prev_max_iterations = std::mem::replace(&mut ctx.max_iterations, max_iterations);
        restore = Some((prev_tools, prev_max_iterations));
    }
    restore
}

fn manifest_tool_definitions(
    tool_groups: &[String],
    tools: &[String],
) -> Option<Vec<ToolDef>> {
    if !tool_groups.is_empty() {
        let groups: Vec<&str> = tool_groups.iter().map(|s| s.as_str()).collect();
        return Some(super::super::tools::tool_definitions_for_groups(&groups));
    }
    if !tools.is_empty() {
        return Some(super::super::tools::get_tool_definitions_by_names(tools));
    }
    None
}

fn resolve_max_iterations(active_agent: Option<&AgentManifest>, openclaw_active: bool) -> usize {
    active_agent
        .and_then(|agent| agent.max_steps)
        .unwrap_or(if openclaw_active {
            OPENCLAW_MAX_ITERATIONS
        } else {
            DEFAULT_MAX_ITERATIONS
        })
}

fn builtin_tools_for_skill(
    prompt_optimizer_active: bool,
    skill: Option<&SkillManifest>,
    active_agent: Option<&AgentManifest>,
) -> Vec<ToolDef> {
    if prompt_optimizer_active {
        return Vec::new();
    }
    if let Some(skill) = skill {
        if let Some(tool_defs) = manifest_tool_definitions(&skill.tool_groups, &skill.tools) {
            return tool_defs;
        }
    }
    if let Some(agent) = active_agent
        && let Some(tool_defs) = manifest_tool_definitions(&agent.tool_groups, &agent.tools)
    {
        return tool_defs;
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

fn build_system_prompt(active_agent: Option<&AgentManifest>, skill: Option<&SkillManifest>) -> String {
    let mut system_prompt = "You are a helpful assistant.".to_string();
    if let Some(agent) = active_agent {
        let extra = agent.build_system_prompt();
        if !extra.trim().is_empty() {
            system_prompt.push_str("\n\n");
            system_prompt.push_str("Agent enforcement:\n- You MUST follow the active agent profile for behavior, workflow, and safety boundaries.\n- Treat the active agent as the default operating mode for this turn.\n- When a skill is also active, satisfy both the agent profile and the skill instructions.\n\n");
            system_prompt.push_str(extra.trim());
        }
    }
    if let Some(skill) = skill {
        system_prompt.push_str("\n\n");
        system_prompt.push_str("Skill enforcement:\n- You MUST follow the active skill instructions precisely.\n- Do not ignore, weaken, or bypass the skill behavior.\n- If the user request conflicts with the skill, ask a brief clarification aligned with the skill.");
        let extra = skill.build_system_prompt();
        if !extra.trim().is_empty() {
            system_prompt.push_str("\n\n");
            system_prompt.push_str(extra.trim());
        }
    }

    system_prompt.push_str("\n\n");
    system_prompt.push_str("Tool recovery mode:\n- If a tool call fails, read the error message and correct course before answering.\n- Prefer retrying with corrected arguments or switching to a more appropriate tool.\n- Do not repeat the exact same failing tool call unless the error indicates a transient retry is appropriate.\n- If a URL-based docs fetch tool says the URL is unsupported, switch to a search tool or ask for a supported docs URL instead of retrying the same call.\n- Only stop and ask the user when the error is ambiguous or missing required information.");
    system_prompt.push_str("\n\n");
    system_prompt.push_str("Code navigation policy:\n- Prefer `code_search` for locating files, symbols, definitions, references, diagnostics, structural code matches, or full-text content hits.\n- When you need to find where a symbol is declared or used, prefer `code_search` with `workspace_symbol`, `go_to_definition`, or `find_references`.\n- When you need to find literal text, log messages, SQL fragments, config keys, or other exact content, prefer `code_search` with `text_search`.\n- For structural searches, prefer high-level `code_search` intents like `find_functions`, `find_classes`, `find_methods`, and `find_calls` before writing a raw tree-sitter query.\n- When structural results are too broad, add `name` to filter the `@name` capture or `contains_text` to filter by captured snippet text.\n- For call searches, you can further narrow matches with `call_kind`, `receiver`, and `qualified_name`.\n- Use `code_search` before raw `grep_search` / `read_file_lines` when you are still narrowing down where relevant code lives.\n- Use `read_file` or `read_file_lines` only after `code_search` or `lsp` has identified the exact file or region you need to inspect.\n- Use raw `grep_search` mainly as a fallback when higher-level code navigation does not apply.\n- If a recent system note labeled `Current code-inspection working memory` is present, treat it as authoritative current-turn context and avoid re-reading the same file range or rerunning equivalent raw searches unless you need verification or a narrower slice.\n- If you have already used several raw repo-inspection tools in a row and are still locating code, switch to `code_search` before making more broad `read_file_lines` or `grep_search` calls.");
    system_prompt.push_str("\n\n");
    system_prompt.push_str("File editing policy:\n- When modifying an existing file or document, DO NOT rewrite the whole file unless the user explicitly asks for a full rewrite or the change truly affects most of the file.\n- First inspect the relevant region with read_file or read_file_lines, then use apply_patch to make the smallest localized edit that preserves the surrounding content.\n- Use write_file mainly for creating new files or for deliberate full-file replacement.\n- This rule applies equally to prose documents, markdown notes, configuration files, and source code.");
    system_prompt.push_str("\n\n");
    system_prompt.push_str("Planning before acting:\n- For simple tasks (read a file, answer a question, run a single command, quick lookup), act directly — do NOT call the plan tool.\n- For complex tasks (multi-step refactoring, debugging across files, building a feature, investigating an unfamiliar codebase), call the `plan` tool first to create a step-by-step plan, then execute it step by step.\n- After each tool execution, review the result and adjust the plan if needed. You do not need to re-plan for minor adjustments.\n- A good rule of thumb: if the task requires 3+ tool calls across different tools/files, plan first.");
    system_prompt.push_str("\n\n");
    system_prompt.push_str("Knowledge base auto-check:\n- At the start of each conversation, briefly consider whether the user's request might benefit from checking the knowledge base.\n- If the request relates to past decisions, project context, preferences, or remembered information, use `knowledge_search` to look up relevant entries.\n- You do NOT need to check the knowledge base for every single query — only when the user's request seems to reference prior context, preferences, or accumulated knowledge.\n- Use `knowledge_list` to browse recent entries when the user asks \"what do you remember\" or similar.");
    system_prompt.push_str("\n\n");
    system_prompt.push_str("Semantic knowledge retrieval:\n- Use `knowledge_semantic_search` when keyword search doesn't find relevant results — it understands meaning, not just exact words.\n- For example, searching \"how to deploy\" can find entries about \"CI/CD pipeline\" even without matching keywords.\n- Use `knowledge_rebuild_index` to sync the vector index if it seems out of date.\n- The knowledge base supports both BM25 keyword search and vector semantic search, combined in hybrid mode.");
    system_prompt.push_str("\n\n");
    system_prompt.push_str("Web search policy:\n- For questions about real-time or time-sensitive information (weather, news, stock prices, sports scores, current events, recent developments, product releases), you MUST use the `web_search` tool to find up-to-date answers.\n- Do NOT attempt to answer time-sensitive questions from your training data alone, as the information is likely outdated or unknown.\n- Use `web_fetch` to retrieve detailed content from specific URLs when search results point to relevant pages.\n- If the user asks about anything that could have changed since your training cutoff, search first.");
    system_prompt
}

fn should_skip_recall_for_skill(skill: Option<&SkillManifest>) -> bool {
    let Some(skill) = skill else {
        return false;
    };
    matches!(
        skill.name.as_str(),
        "debugger" | "code-review" | "refactor" | "prompt-optimizer" | "openclaw"
    ) || skill.tool_groups.iter().any(|g| g == "openclaw")
}

#[crate::ai::agent_hang_span(
    "pre-fix",
    "R",
    "skill_runtime::prepare_skill_for_turn:intent",
    "[DEBUG] intent recognition started",
    "[DEBUG] intent recognition finished",
    {
        "question_len": question.chars().count(),
    },
    {
        "core": format!("{:?}", __agent_hang_result.core),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
fn detect_turn_intent(
    question: &str,
    intent_model_path: &std::path::Path,
) -> intent_recognition::UserIntent {
    intent_recognition::detect_intent_with_model_path(question, intent_model_path)
}

#[crate::ai::agent_hang_span(
    "pre-fix",
    "R",
    "skill_runtime::prepare_skill_for_turn:router",
    "[DEBUG] skill router started",
    "[DEBUG] skill router finished",
    {
        "router_enabled": router_enabled,
        "skip_router_for_local_intent": skip_router_for_local_intent,
        "skill_count": skill_manifests.len(),
    },
    {
        "router_enabled": router_enabled,
        "skip_router_for_local_intent": skip_router_for_local_intent,
        "selected": __agent_hang_result.as_deref(),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
async fn route_skill_for_turn(
    app: &mut App,
    question: &str,
    skill_manifests: &[SkillManifest],
    router_enabled: bool,
    skip_router_for_local_intent: bool,
) -> Option<String> {
    if router_enabled && !skip_router_for_local_intent {
        let model = app.current_model.clone();
        request::select_skill_via_model(app, &model, question, skill_manifests).await
    } else {
        None
    }
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

    let intent = detect_turn_intent(question, &app.config.intent_model_path);
    let question_len = question.chars().count();
    let skip_router_for_local_intent = matches!(
        intent.core,
        intent_recognition::CoreIntent::Casual | intent_recognition::CoreIntent::QueryConcept
    ) && question_len <= 64;
    let router_selected = route_skill_for_turn(
        app,
        question,
        skill_manifests,
        router_enabled,
        skip_router_for_local_intent,
    )
    .await;

    let heuristic_skill = match_skill(skill_manifests, question, Some(&intent));
    let router_skill = router_selected
        .as_deref()
        .and_then(|name| skill_manifests.iter().find(|s| s.name == name));

    let skill = resolve_skill_selection(router_skill, heuristic_skill, &intent);

    if debug {
        if let Some(name) = router_selected.as_deref() {
            eprintln!("[skills] router selected: {}", name);
        }
        if let Some(s) = heuristic_skill.as_ref() {
            eprintln!("[skills] heuristic candidate: {}", s.name);
        }
        eprintln!("[skills] intent: {:?}", intent.core);
        if let Some(s) = skill.as_ref() {
            eprintln!("[skills] final: {}", s.name);
        } else {
            eprintln!("[skills] final: <none>");
        }
    }
    let matched_skill_name = skill.as_ref().map(|s| s.name.clone());
    let skip_recall_by_skill = should_skip_recall_for_skill(skill);
    let prompt_optimizer_active = skill
        .as_ref()
        .is_some_and(|s| s.name.as_str() == "prompt-optimizer");
    let active_agent = app.current_agent_manifest.clone();
    let openclaw_active = skill.as_ref().is_some_and(|s| {
        s.name.as_str() == "openclaw" || s.tool_groups.iter().any(|g| g == "openclaw")
    }) || active_agent.as_ref().is_some_and(|agent| {
        agent.name.as_str() == "openclaw" || agent.tool_groups.iter().any(|g| g == "openclaw")
    });

    let builtin_tools = builtin_tools_for_skill(prompt_optimizer_active, skill, active_agent.as_ref());
    let mcp_tools = mcp_tools_for_skill(mcp_client, prompt_optimizer_active, skill);
    let system_prompt = build_system_prompt(active_agent.as_ref(), skill);
    let max_iterations = resolve_max_iterations(active_agent.as_ref(), openclaw_active);
    let restore_agent_context =
        activate_skill_context(app, builtin_tools, mcp_tools, max_iterations);

    let guard = SkillTurnGuard {
        app: NonNull::from(&mut *app),
        restore_agent_context,
        system_prompt,
        matched_skill_name,
        intent,
        skip_recall_by_skill,
    };
    guard
}

fn resolve_skill_selection<'a>(
    router_skill: Option<&'a SkillManifest>,
    heuristic_skill: Option<&'a SkillManifest>,
    intent: &UserIntent,
) -> Option<&'a SkillManifest> {
    match (router_skill, heuristic_skill) {
        (Some(r), Some(h)) => {
            if r.name == h.name {
                return Some(r);
            }
            if intent.is_searching_resource("skill") {
                return None;
            }
            let pr = super::tools::penalty_for_skill_tools(r);
            let ph = super::tools::penalty_for_skill_tools(h);
            if ph + 1e-6 < pr {
                Some(h)
            } else {
                Some(r)
            }
        }
        (Some(r), None) => {
            if intent.is_searching_resource("skill") {
                None
            } else {
                Some(r)
            }
        }
        (None, Some(h)) => Some(h),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{resolve_max_iterations, tool_uses_mcp_server};
    use crate::ai::agents::{AgentManifest, AgentMode};

    #[test]
    fn mcp_server_filter_matches_longest_server_name_prefix() {
        let allowed = vec!["foo".to_string(), "foo_bar".to_string()];
        assert!(tool_uses_mcp_server("mcp_foo_bar_search", &allowed));
        assert!(tool_uses_mcp_server("mcp_foo_lookup", &allowed));
        assert!(!tool_uses_mcp_server("mcp_bar_search", &allowed));
    }

    #[test]
    fn active_agent_max_steps_override_default_iterations() {
        let agent = AgentManifest {
            name: "openclaw".to_string(),
            description: String::new(),
            mode: AgentMode::Primary,
            model: None,
            temperature: None,
            max_steps: Some(17),
            prompt: String::new(),
            system_prompt: None,
            tools: Vec::new(),
            tool_groups: vec!["builtin".to_string(), "openclaw".to_string()],
            mcp_servers: Vec::new(),
            routing_tags: Vec::new(),
            model_tier: None,
            disabled: false,
            hidden: false,
            color: None,
            source_path: None,
        };

        assert_eq!(resolve_max_iterations(Some(&agent), false), 17);
        assert_eq!(resolve_max_iterations(Some(&agent), true), 17);
        assert_eq!(resolve_max_iterations(None, true), super::super::OPENCLAW_MAX_ITERATIONS);
        assert_eq!(resolve_max_iterations(None, false), super::super::DEFAULT_MAX_ITERATIONS);
    }
}
