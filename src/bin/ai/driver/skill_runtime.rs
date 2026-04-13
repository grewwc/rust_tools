use crate::ai::{
    agents::AgentManifest,
    mcp::McpClient,
    skills::SkillManifest,
    types::{App, SkillBiasMemory, ToolDefinition},
};
use crate::commonw::configw;
use std::ptr::NonNull;
use rust_tools::cw::SkipMap;
use rust_tools::cw::SkipSet;
use chrono::{DateTime, Utc};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use super::{
    DEFAULT_MAX_ITERATIONS, EXECUTOR_MAX_ITERATIONS, TextSimilarityFeatures,
    jaccard_similarity_for_sets, rank_skills_locally, set_intersection_count,
};
use super::intent_recognition::{self, UserIntent};

type ToolDef = ToolDefinition;
type ToolScoreMap = SkipMap<String, f64>;

const TOOL_SCORE_CACHE_TTL: Duration = Duration::from_secs(20);
const DEFAULT_TURN_TOOL_GROUPS: &[&str] = &["core"];
const CURRENT_SKILL_STICKY_BONUS: f64 = 1.25;
const CURRENT_SKILL_KEEP_FLOOR_DELTA: f64 = 0.75;
const SKILL_SWITCH_MARGIN: f64 = 1.5;
const CROSS_TURN_SKILL_STICKY_BONUS: f64 = 0.35;
const CROSS_TURN_SKILL_KEEP_FLOOR_DELTA: f64 = 0.25;
const CROSS_TURN_SKILL_SWITCH_MARGIN: f64 = 0.6;

static TOOL_SCORE_CACHE: LazyLock<Mutex<Option<(Instant, Arc<Box<ToolScoreMap>>)>>> =
    LazyLock::new(|| Mutex::new(None));

#[derive(Clone, Copy)]
enum PreferenceStrength {
    StrongSticky,
    CrossTurnBias,
}

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

    pub(super) fn take_restore_agent_context(&mut self) -> Option<(Vec<ToolDef>, usize)> {
        self.restore_agent_context.take()
    }

    pub(super) fn set_restore_agent_context(&mut self, restore: Option<(Vec<ToolDef>, usize)>) {
        self.restore_agent_context = restore;
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
        let all_tools = merge_with_runtime_enabled_tools(builtin_tools, mcp_tools, &ctx.tools);
        let names: Vec<String> = all_tools.iter().map(|t| t.function.name.clone()).collect();
        super::super::tools::enable_tools::set_active_tool_names(names);
        let prev_tools = std::mem::replace(&mut ctx.tools, all_tools);
        let prev_max_iterations = std::mem::replace(&mut ctx.max_iterations, max_iterations);
        restore = Some((prev_tools, prev_max_iterations));
    }
    restore
}

fn merge_with_runtime_enabled_tools(
    builtin_tools: Vec<ToolDef>,
    mcp_tools: Vec<ToolDef>,
    current_tools: &[ToolDef],
) -> Vec<ToolDef> {
    let mut merged = reorder_tools_by_stats(builtin_tools, mcp_tools);
    let explicit_enabled = super::super::tools::enable_tools::explicit_enabled_tool_names()
        .into_iter()
        .collect::<Box<SkipSet<_>>>();
    if explicit_enabled.is_empty() {
        return merged;
    }
    let known_names: Box<SkipSet<String>> = merged
        .iter()
        .map(|tool| tool.function.name.clone())
        .collect();
    let runtime_extra = current_tools
        .iter()
        .filter(|tool| explicit_enabled.contains(&tool.function.name))
        .filter(|tool| !known_names.contains(&tool.function.name))
        .cloned()
        .collect::<Vec<_>>();
    if runtime_extra.is_empty() {
        return merged;
    }
    merged.extend(runtime_extra);
    rust_tools::sortw::stable_sort_by(&mut merged, |a, b| a.function.name.cmp(&b.function.name));
    dedupe_tools_by_name(merged)
}

fn dedupe_tools_by_name(tools: Vec<ToolDef>) -> Vec<ToolDef> {
    let mut seen = SkipSet::new(16);
    let mut result = Vec::new();
    for tool in tools {
        if seen.insert(tool.function.name.clone()) {
            result.push(tool);
        }
    }
    result
}

fn required_discovery_tool_names() -> Vec<String> {
    vec![
        "enable_tools".to_string(),
        "discover_skills".to_string(),
    ]
}

fn ensure_required_discovery_tools(mut tools: Vec<ToolDef>) -> Vec<ToolDef> {
    let existing = tools
        .iter()
        .map(|tool| tool.function.name.clone())
        .collect::<Box<SkipSet<_>>>();
    let missing = required_discovery_tool_names()
        .into_iter()
        .filter(|name| !existing.contains(name))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return dedupe_tools_by_name(tools);
    }
    let extra = super::super::tools::get_tool_definitions_by_names(&missing);
    tools.extend(extra);
    dedupe_tools_by_name(tools)
}

fn manifest_tool_definitions(
    tool_groups: &[String],
    tools: &[String],
) -> Option<Vec<ToolDef>> {
    if !tool_groups.is_empty() {
        let groups: Vec<&str> = tool_groups.iter().map(|s| s.as_str()).collect();
        return Some(ensure_required_discovery_tools(
            super::super::tools::tool_definitions_for_groups(&groups),
        ));
    }
    if !tools.is_empty() {
        return Some(ensure_required_discovery_tools(
            super::super::tools::get_tool_definitions_by_names(tools),
        ));
    }
    None
}

fn is_executor_name(name: &str) -> bool {
    matches!(name, "executor" | "openclaw")
}

fn has_executor_group(tool_groups: &[String]) -> bool {
    tool_groups
        .iter()
        .any(|group| matches!(group.as_str(), "executor" | "openclaw"))
}

fn resolve_max_iterations(active_agent: Option<&AgentManifest>, executor_active: bool) -> usize {
    active_agent
        .and_then(|agent| agent.max_steps)
        .unwrap_or(if executor_active {
            EXECUTOR_MAX_ITERATIONS
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
    super::super::tools::tool_definitions_for_groups(DEFAULT_TURN_TOOL_GROUPS)
}

fn available_tool_names(builtin_tools: &[ToolDef], mcp_tools: &[ToolDef]) -> Box<SkipSet<String>> {
    builtin_tools
        .iter()
        .chain(mcp_tools.iter())
        .map(|tool| tool.function.name.clone())
        .collect()
}

fn has_tool(available: &Box<SkipSet<String>>, name: &str) -> bool {
    available.contains_str(name)
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
        let sa = scores.get_ref(&a.function.name).copied().unwrap_or(0.0);
        let sb = scores.get_ref(&b.function.name).copied().unwrap_or(0.0);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.function.name.cmp(&b.function.name))
    });
    all
}

fn load_tool_scores() -> Box<ToolScoreMap> {
    if let Ok(cache) = TOOL_SCORE_CACHE.lock()
        && let Some((created_at, scores)) = cache.as_ref()
        && created_at.elapsed() < TOOL_SCORE_CACHE_TTL
    {
        return (**scores).clone();
    }

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
    let scores: Box<SkipMap<String, f64>> = (&*score)
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect::<Box<SkipMap<_, _>>>();
    let result = scores.clone();
    if let Ok(mut cache) = TOOL_SCORE_CACHE.lock() {
        *cache = Some((Instant::now(), Arc::new(scores)));
    }
    result
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

    let Some(skill) = skill else {
        return Vec::new();
    };
    if skill.mcp_servers.is_empty() {
        return Vec::new();
    }

    let all_tools = mcp_client.get_all_tools();
    all_tools
        .into_iter()
        .filter(|tool| tool_uses_mcp_server(&tool.function.name, &skill.mcp_servers))
        .collect()
}

fn build_system_prompt(
    active_agent: Option<&AgentManifest>,
    skill: Option<&SkillManifest>,
    available_tools: &Box<SkipSet<String>>,
) -> String {
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

    if !available_tools.is_empty() {
        let available = available_tools
            .iter()
            .map(|name| format!("`{name}`"))
            .collect::<Vec<_>>()
            .join(", ");
        system_prompt.push_str("\n\n");
        system_prompt.push_str("Tool availability:\n- Only rely on tools that are actually available in this turn.\n- Available tools: ");
        system_prompt.push_str(&available);
        system_prompt.push('\n');
    }

    system_prompt.push_str("\n\n");
    system_prompt.push_str("Tool recovery mode:\n- If a tool call fails, read the error message and correct course before answering.\n- Prefer retrying with corrected arguments or switching to a more appropriate tool.\n- Do not repeat the exact same failing tool call unless the error indicates a transient retry is appropriate.\n- If `code_search` returns only `No ...` style results, do not rerun the same request unchanged; broaden the scope, remove filters, change the operation, or use the fallback guidance returned by `code_search`.\n- If a URL-based docs fetch tool says the URL is unsupported, switch to a search tool or ask for a supported docs URL instead of retrying the same call.\n- Only stop and ask the user when the error is ambiguous or missing required information.");
    system_prompt.push_str("\n\n");
    system_prompt.push_str("Tool selection policy:\n- If the answer depends on code or repo contents and code/file tools are available, inspect with tools before concluding.\n- If the user asks you to modify files and editing tools are available, make the change with tools instead of only describing what to change.\n- If the user asks to run, build, test, or reproduce behavior and command tools are available, execute the relevant command instead of only suggesting it.\n- If a path, URL, symbol, or query is provided and there is a matching tool for that resource, prefer using the tool over guessing.");

    if has_tool(available_tools, "enable_tools") {
        system_prompt.push_str("\n\n");
        system_prompt.push_str("Tool discovery policy:\n- Do NOT assume every possible tool is already loaded for this turn.\n- When you need a capability that is not currently available, call `enable_tools(operation=list)` to discover additional tools, then `enable_tools(operation=enable, tools=[...])` to load only the tools you need.");
    }

    if has_tool(available_tools, "discover_skills") {
        system_prompt.push_str("\n\n");
        system_prompt.push_str("Skill discovery policy:\n- Do NOT assume all skill prompts are already visible in this turn.\n- When you need to discover specialized skills, call `discover_skills` to inspect skill metadata only.\n- Use skill discovery to find relevant skill names and capabilities without loading every skill prompt into context.");
    }

    if has_tool(available_tools, "code_search") {
        system_prompt.push_str("\n\n");
        system_prompt.push_str("Code navigation policy:\n- ALWAYS use `code_search` as your FIRST tool when exploring code. Do NOT start with `read_file`, `read_file_lines`, `grep_search`, or `search_files`.\n- The only exception: use `read_file_lines` directly when you already know the exact file path AND line range from a previous `code_search` result.\n- When you need to find where a symbol is declared or used, use `code_search` with `operation=workspace_symbol`, `operation=go_to_definition`, or `operation=find_references`.\n- When you need to find literal text, log messages, SQL fragments, config keys, or other exact content, use `code_search` with `operation=text_search`.\n- For structural searches, ALWAYS call `code_search` with `operation=structural` and set `intent` to one of `find_functions`, `find_classes`, `find_methods`, or `find_calls`.\n- Never send `find_functions`, `find_classes`, `find_methods`, or `find_calls` as the `operation` value; those belong in `intent`.\n- When structural results are too broad, add `name` to filter the `@name` capture or `contains_text` to filter by captured snippet text.\n- For call searches, you can further narrow matches with `call_kind`, `receiver`, and `qualified_name`.\n- Use raw `grep_search` ONLY as a last resort when `code_search` does not apply.\n- If a recent system note labeled `Current code-inspection working memory` is present, treat it as authoritative current-turn context and avoid re-reading the same file range or rerunning equivalent raw searches unless you need verification or a narrower slice.\n- If you have already used raw repo-inspection tools and are still locating code, you MUST switch to `code_search` immediately.");
    }

    if has_tool(available_tools, "read_file")
        || has_tool(available_tools, "read_file_lines")
        || has_tool(available_tools, "apply_patch")
        || has_tool(available_tools, "write_file")
    {
        system_prompt.push_str("\n\n");
        system_prompt.push_str("File editing policy:\n- When modifying an existing file or document, DO NOT rewrite the whole file unless the user explicitly asks for a full rewrite or the change truly affects most of the file.\n- First inspect the relevant region with read_file or read_file_lines, then use apply_patch to make the smallest localized edit that preserves the surrounding content.\n- Use write_file mainly for creating new files or for deliberate full-file replacement.\n- This rule applies equally to prose documents, markdown notes, configuration files, and source code.");
    }

    if has_tool(available_tools, "plan") {
        system_prompt.push_str("\n\n");
        system_prompt.push_str("Planning before acting:\n- For simple tasks (read a file, answer a question, run a single command, quick lookup), act directly — do NOT call the plan tool.\n- For complex tasks (multi-step refactoring, debugging across files, building a feature, investigating an unfamiliar codebase), call the `plan` tool first to create a step-by-step plan, then execute it step by step.\n- After each tool execution, review the result and adjust the plan if needed. You do not need to re-plan for minor adjustments.\n- A good rule of thumb: if the task requires 3+ tool calls across different tools/files, plan first.");
    }

    if has_tool(available_tools, "knowledge_search") || has_tool(available_tools, "knowledge_list") {
        system_prompt.push_str("\n\n");
        system_prompt.push_str("Knowledge base auto-check:\n- At the start of each conversation, briefly consider whether the user's request might benefit from checking the knowledge base.\n- If the request relates to past decisions, project context, preferences, or remembered information, use `knowledge_search` to look up relevant entries.\n- You do NOT need to check the knowledge base for every single query — only when the user's request seems to reference prior context, preferences, or accumulated knowledge.\n- Use `knowledge_list` to browse recent entries when the user asks \"what do you remember\" or similar.");
    }

    if has_tool(available_tools, "knowledge_semantic_search") {
        system_prompt.push_str("\n\n");
        system_prompt.push_str("Semantic knowledge retrieval:\n- Use `knowledge_semantic_search` when keyword search doesn't find relevant results — it understands meaning, not just exact words.\n- For example, searching \"how to deploy\" can find entries about \"CI/CD pipeline\" even without matching keywords.\n- Use `knowledge_rebuild_index` to sync the vector index if it seems out of date.\n- The knowledge base supports both BM25 keyword search and vector semantic search, combined in hybrid mode.");
    }

    if has_tool(available_tools, "web_search") || has_tool(available_tools, "web_fetch") {
        system_prompt.push_str("\n\n");
        system_prompt.push_str("Web search policy:\n- For questions about real-time or time-sensitive information (weather, news, stock prices, sports scores, current events, recent developments, product releases), you MUST use the `web_search` tool to find up-to-date answers.\n- Do NOT attempt to answer time-sensitive questions from your training data alone, as the information is likely outdated or unknown.\n- Use `web_fetch` to retrieve detailed content from specific URLs when search results point to relevant pages.\n- If the user asks about anything that could have changed since your training cutoff, search first.");
    }
    system_prompt
}

fn should_skip_recall_for_skill(skill: Option<&SkillManifest>) -> bool {
    let Some(skill) = skill else {
        return false;
    };
    matches!(
        skill.name.as_str(),
        "debugger" | "code-review" | "refactor" | "prompt-optimizer"
    ) || is_executor_name(skill.name.as_str())
        || has_executor_group(&skill.tool_groups)
}

fn skill_selection_threshold(intent: &UserIntent, skill_count: usize) -> f64 {
    let base: f64 = if skill_count > 10 {
        5.0
    } else if skill_count > 5 {
        4.5
    } else {
        4.0
    };
    match intent.core {
        intent_recognition::CoreIntent::RequestAction => base.max(4.5),
        intent_recognition::CoreIntent::SeekSolution => base.max(4.25),
        intent_recognition::CoreIntent::QueryConcept => base.max(5.5),
        intent_recognition::CoreIntent::Casual => base.max(5.5),
    }
}

fn looks_like_follow_up_or_same_topic(question: &str, previous_question: &str) -> bool {
    let current = question.trim();
    let previous = previous_question.trim();
    if current.is_empty() || previous.is_empty() {
        return false;
    }

    let current_features = TextSimilarityFeatures::from_text(current);
    let previous_features = TextSimilarityFeatures::from_text(previous);
    let token_similarity = jaccard_similarity_for_sets(
        &current_features.token_set,
        &previous_features.token_set,
    );
    if token_similarity >= 0.34 {
        return true;
    }

    let bigram_similarity = jaccard_similarity_for_sets(
        &current_features.char_bigrams,
        &previous_features.char_bigrams,
    );
    if bigram_similarity >= 0.2 {
        return true;
    }

    if current_features.token_set.is_empty() || previous_features.token_set.is_empty() {
        return false;
    }

    let overlap =
        set_intersection_count(&current_features.token_set, &previous_features.token_set);
    let new_tokens = current_features.token_set.len().saturating_sub(overlap);
    let new_token_ratio = new_tokens as f64 / current_features.token_set.len().max(1) as f64;
    let current_chars = current.chars().count();
    let previous_chars = previous.chars().count();

    (current_chars <= 64 && overlap >= 1 && new_token_ratio <= 0.5)
        || (current_chars <= 32 && current_chars <= previous_chars && overlap >= 1)
        || (current_chars <= 24 && (token_similarity >= 0.18 || bigram_similarity >= 0.12))
}

fn cross_turn_preferred_skill_name(
    app: &App,
    question: &str,
    intent: &UserIntent,
) -> Option<String> {
    if intent.is_searching_resource("skill") {
        return None;
    }
    let memory = app.last_skill_bias.as_ref()?;
    if looks_like_follow_up_or_same_topic(question, &memory.question) {
        Some(memory.skill_name.clone())
    } else {
        None
    }
}

fn update_cross_turn_skill_bias(app: &mut App, question: &str, skill: Option<&SkillManifest>) {
    app.last_skill_bias = skill.map(|selected| SkillBiasMemory {
        skill_name: selected.name.clone(),
        question: question.trim().to_string(),
    });
}

fn select_skill_with_preference<'a>(
    skill_manifests: &'a [SkillManifest],
    question: &str,
    intent: &UserIntent,
    preferred_skill_name: Option<&str>,
) -> Option<&'a SkillManifest> {
    select_skill_with_preference_strength(
        skill_manifests,
        question,
        intent,
        preferred_skill_name,
        PreferenceStrength::StrongSticky,
    )
}

fn select_skill_with_preference_strength<'a>(
    skill_manifests: &'a [SkillManifest],
    question: &str,
    intent: &UserIntent,
    preferred_skill_name: Option<&str>,
    strength: PreferenceStrength,
) -> Option<&'a SkillManifest> {
    if question.trim().is_empty() || skill_manifests.is_empty() || intent.is_searching_resource("skill") {
        return None;
    }

    let ranked = rank_skills_locally(skill_manifests, question, Some(intent));
    let Some(best) = ranked.first() else {
        return None;
    };
    let threshold = skill_selection_threshold(intent, skill_manifests.len());
    let preferred = preferred_skill_name.and_then(|name| ranked.iter().find(|item| item.skill.name == name));

    if let Some(current) = preferred {
        let (sticky_bonus, keep_floor_delta, switch_margin, keep_current_when_best, keep_below_threshold) =
            match strength {
                PreferenceStrength::StrongSticky => (
                    CURRENT_SKILL_STICKY_BONUS,
                    CURRENT_SKILL_KEEP_FLOOR_DELTA,
                    SKILL_SWITCH_MARGIN,
                    true,
                    true,
                ),
                PreferenceStrength::CrossTurnBias => (
                    CROSS_TURN_SKILL_STICKY_BONUS,
                    CROSS_TURN_SKILL_KEEP_FLOOR_DELTA,
                    CROSS_TURN_SKILL_SWITCH_MARGIN,
                    true,
                    false,
                ),
            };
        let current_has_signal = current.score >= (threshold - keep_floor_delta).max(0.0)
            || current.heuristic_score >= 3.0
            || current.model_score >= 0.08;
        if best.skill.name == current.skill.name {
            return if keep_current_when_best || current.score >= threshold || current_has_signal {
                Some(current.skill)
            } else {
                None
            };
        }

        let effective_current = current.score + sticky_bonus;
        let best_clearly_wins = best.score >= effective_current + switch_margin;
        if keep_below_threshold && best.score < threshold {
            return Some(current.skill);
        }
        if current_has_signal && !best_clearly_wins {
            return Some(current.skill);
        }
        if best.score >= threshold {
            return Some(best.skill);
        }
        if current_has_signal {
            return Some(current.skill);
        }
        return None;
    }

    (best.score >= threshold).then_some(best.skill)
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

fn build_skill_turn_guard(
    app: &mut App,
    mcp_client: &McpClient,
    skill: Option<&SkillManifest>,
    intent: UserIntent,
) -> SkillTurnGuard {
    super::super::tools::enable_tools::set_available_mcp_tools(mcp_client.get_all_tools());
    let matched_skill_name = skill.as_ref().map(|s| s.name.clone());
    let skip_recall_by_skill = should_skip_recall_for_skill(skill);
    let prompt_optimizer_active = skill
        .as_ref()
        .is_some_and(|s| s.name.as_str() == "prompt-optimizer");
    let active_agent = app.current_agent_manifest.clone();
    let executor_active = skill.as_ref().is_some_and(|s| {
        is_executor_name(s.name.as_str()) || has_executor_group(&s.tool_groups)
    }) || active_agent.as_ref().is_some_and(|agent| {
        is_executor_name(agent.name.as_str()) || has_executor_group(&agent.tool_groups)
    });

    let builtin_tools = builtin_tools_for_skill(prompt_optimizer_active, skill, active_agent.as_ref());
    let mcp_tools = mcp_tools_for_skill(mcp_client, prompt_optimizer_active, skill);
    let available_tools = available_tool_names(&builtin_tools, &mcp_tools);
    let system_prompt = build_system_prompt(active_agent.as_ref(), skill, &available_tools);
    let max_iterations = resolve_max_iterations(active_agent.as_ref(), executor_active);
    let restore_agent_context =
        activate_skill_context(app, builtin_tools, mcp_tools, max_iterations);

    SkillTurnGuard {
        app: NonNull::from(&mut *app),
        restore_agent_context,
        system_prompt,
        matched_skill_name,
        intent,
        skip_recall_by_skill,
    }
}

pub(super) fn rebuild_skill_turn_with_existing_selection(
    app: &mut App,
    mcp_client: &McpClient,
    skill_manifests: &[SkillManifest],
    question: &str,
    preferred_skill_name: Option<&str>,
    intent: &UserIntent,
) -> SkillTurnGuard {
    let skill =
        select_skill_with_preference(skill_manifests, question, intent, preferred_skill_name);
    build_skill_turn_guard(app, mcp_client, skill, intent.clone())
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

    let intent = detect_turn_intent(question, &app.config.intent_model_path);
    let ranked = rank_skills_locally(skill_manifests, question, Some(&intent));
    let cross_turn_preference = cross_turn_preferred_skill_name(app, question, &intent);
    let skill = select_skill_with_preference_strength(
        skill_manifests,
        question,
        &intent,
        cross_turn_preference.as_deref(),
        PreferenceStrength::CrossTurnBias,
    );

    if debug {
        if let Some(top) = ranked.first() {
            eprintln!(
                "[skills] local top: {} total={:.3} heuristic={:.3} model={:.3}",
                top.skill.name, top.score, top.heuristic_score, top.model_score
            );
        }
        if let Some(preferred) = cross_turn_preference.as_deref() {
            eprintln!("[skills] cross-turn preferred: {}", preferred);
        }
        eprintln!("[skills] intent: {:?}", intent.core);
        if let Some(s) = skill.as_ref() {
            eprintln!("[skills] final: {}", s.name);
        } else {
            eprintln!("[skills] final: <none>");
        }
    }
    update_cross_turn_skill_bias(app, question, skill);
    let matched_skill_name = skill.as_ref().map(|s| s.name.clone());
    let skip_recall_by_skill = should_skip_recall_for_skill(skill);
    let mut guard = build_skill_turn_guard(app, mcp_client, skill, intent);
    guard.matched_skill_name = matched_skill_name;
    guard.skip_recall_by_skill = skip_recall_by_skill;
    guard
}

#[cfg(test)]
mod tests {
    use super::{
        build_system_prompt, builtin_tools_for_skill, ensure_required_discovery_tools,
        looks_like_follow_up_or_same_topic, merge_with_runtime_enabled_tools,
        resolve_max_iterations, select_skill_with_preference,
        select_skill_with_preference_strength, tool_uses_mcp_server, PreferenceStrength,
    };
    use crate::ai::agents::{AgentManifest, AgentMode};
    use crate::ai::driver::intent_recognition::{CoreIntent, UserIntent};
    use crate::ai::skills::SkillManifest;
    use crate::ai::tools::enable_tools::set_explicit_enabled_tool_names;
    use crate::ai::types::{FunctionDefinition, ToolDefinition};
    use rust_tools::cw::SkipSet;
    use std::sync::{LazyLock, Mutex};

    static EXPLICIT_TOOL_TEST_GUARD: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
    
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
            name: "executor".to_string(),
            description: String::new(),
            mode: AgentMode::Primary,
            model: None,
            temperature: None,
            max_steps: Some(17),
            prompt: String::new(),
            system_prompt: None,
            tools: Vec::new(),
            tool_groups: vec!["builtin".to_string(), "executor".to_string()],
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
        assert_eq!(resolve_max_iterations(None, true), super::super::EXECUTOR_MAX_ITERATIONS);
        assert_eq!(resolve_max_iterations(None, false), super::super::DEFAULT_MAX_ITERATIONS);
    }

    #[test]
    fn default_tools_start_with_core_discovery_and_editing() {
        let tools = builtin_tools_for_skill(false, None, None);
        let names = tools
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();
        assert!(names.iter().any(|name| name == "enable_tools"));
        assert!(names.iter().any(|name| name == "discover_skills"));
        assert!(names.iter().any(|name| name == "code_search"));
        assert!(names.iter().any(|name| name == "read_file"));
        assert!(!names.iter().any(|name| name == "web_search"));
        assert!(!names.iter().any(|name| name == "knowledge_search"));
    }

    #[test]
    fn system_prompt_only_mentions_tools_available_this_turn() {
        let mut available = SkipSet::new(16);
        available.insert("code_search".to_string());
        available.insert("read_file".to_string());
        available.insert("apply_patch".to_string());
        available.insert("enable_tools".to_string());
        available.insert("discover_skills".to_string());

        let prompt = build_system_prompt(None, None, &Box::new(available));
        assert!(prompt.contains("Available tools:"));
        assert!(prompt.contains("Code navigation policy"));
        assert!(prompt.contains("File editing policy"));
        assert!(prompt.contains("Tool discovery policy"));
        assert!(prompt.contains("Skill discovery policy"));
        assert!(!prompt.contains("Web search policy"));
        assert!(!prompt.contains("Knowledge base auto-check"));
    }

    fn tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.to_string(),
                description: String::new(),
                parameters: serde_json::json!({}),
            },
        }
    }

    fn skill(name: &str, description: &str) -> SkillManifest {
        SkillManifest {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            description: description.to_string(),
            author: None,
            tools: Vec::new(),
            tool_groups: Vec::new(),
            mcp_servers: Vec::new(),
            prompt: String::new(),
            system_prompt: None,
            priority: 0,
            source_path: Some(format!("builtin:{name}.skill")),
        }
    }

    #[test]
    fn runtime_enabled_tools_are_preserved_when_refreshing_context() {
        let _guard = EXPLICIT_TOOL_TEST_GUARD.lock().unwrap();
        set_explicit_enabled_tool_names(vec!["enable_tools".to_string(), "web_search".to_string()]);
        let merged = merge_with_runtime_enabled_tools(
            vec![tool("code_search"), tool("read_file"), tool("enable_tools")],
            vec![],
            &[tool("code_search"), tool("enable_tools"), tool("web_search")],
        );
        let names = merged
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"code_search".to_string()));
        assert!(names.contains(&"enable_tools".to_string()));
        assert!(names.contains(&"web_search".to_string()));
        set_explicit_enabled_tool_names(Vec::new());
    }

    #[test]
    fn non_explicit_skill_tools_do_not_leak_into_next_context() {
        let _guard = EXPLICIT_TOOL_TEST_GUARD.lock().unwrap();
        set_explicit_enabled_tool_names(vec!["web_search".to_string()]);
        let merged = merge_with_runtime_enabled_tools(
            vec![tool("code_search")],
            vec![],
            &[tool("code_search"), tool("apply_patch"), tool("web_search")],
        );
        let names = merged
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"code_search".to_string()));
        assert!(names.contains(&"web_search".to_string()));
        assert!(!names.contains(&"apply_patch".to_string()));
        set_explicit_enabled_tool_names(Vec::new());
    }

    #[test]
    fn explicit_tool_lists_keep_only_discovery_entry_available() {
        let merged = ensure_required_discovery_tools(vec![tool("code_search")]);
        let names = merged
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"enable_tools".to_string()));
        assert!(names.contains(&"discover_skills".to_string()));
        assert!(names.contains(&"code_search".to_string()));
        assert!(!names.contains(&"plan".to_string()));
        assert!(!names.contains(&"read_file".to_string()));
        assert!(!names.contains(&"search_files".to_string()));
    }

    #[test]
    fn local_selector_chooses_review_skill_for_review_request() {
        let intent = UserIntent::new(CoreIntent::RequestAction);
        let skills = vec![
            skill("code-review", "Review code changes and highlight bugs"),
            skill("debugger", "Debug runtime failures and collect traces"),
        ];
        let selected = select_skill_with_preference(&skills, "帮我 review 这段 Rust 代码", &intent, None);
        assert_eq!(selected.map(|item| item.name.as_str()), Some("code-review"));
    }

    #[test]
    fn local_selector_prefers_current_skill_without_clear_winner() {
        let intent = UserIntent::new(CoreIntent::RequestAction);
        let skills = vec![
            skill("code-review", "Review code changes and summarize defects"),
            skill("debugger", "Debug runtime failures, panic, and stack traces"),
        ];
        let selected =
            select_skill_with_preference(&skills, "帮我看一下这段代码哪里有问题", &intent, Some("code-review"));
        assert_eq!(selected.map(|item| item.name.as_str()), Some("code-review"));
    }

    #[test]
    fn local_selector_switches_when_new_skill_is_significantly_better() {
        let intent = UserIntent::new(CoreIntent::RequestAction);
        let skills = vec![
            skill("code-review", "Review code changes and summarize defects"),
            skill("debugger", "Debug panic crash stack trace runtime failure logs"),
        ];
        let selected = select_skill_with_preference(
            &skills,
            "程序 panic 了，帮我调试这个 crash 和 stack trace",
            &intent,
            Some("code-review"),
        );
        assert_eq!(selected.map(|item| item.name.as_str()), Some("debugger"));
    }

    #[test]
    fn follow_up_detector_recognizes_short_continuation_question() {
        assert!(looks_like_follow_up_or_same_topic(
            "那这个 panic 呢？",
            "帮我调试一下这个 Rust panic"
        ));
    }

    #[test]
    fn cross_turn_bias_prefers_previous_skill_on_follow_up() {
        let intent = UserIntent::new(CoreIntent::RequestAction);
        let skills = vec![
            skill("code-review", "Review code changes and summarize defects"),
            skill("debugger", "Debug runtime failures and panic stack traces"),
        ];
        let selected = select_skill_with_preference_strength(
            &skills,
            "那这个报错顺便再看一下",
            &intent,
            Some("debugger"),
            PreferenceStrength::CrossTurnBias,
        );
        assert_eq!(selected.map(|item| item.name.as_str()), Some("debugger"));
    }

    #[test]
    fn cross_turn_bias_still_switches_when_new_skill_is_clearly_better() {
        let intent = UserIntent::new(CoreIntent::RequestAction);
        let skills = vec![
            skill("code-review", "Review code changes and summarize defects"),
            skill("debugger", "Debug panic crash stack trace runtime failure logs"),
        ];
        let selected = select_skill_with_preference_strength(
            &skills,
            "继续这个问题，不过现在请直接 review 这段实现有没有逻辑 bug",
            &intent,
            Some("debugger"),
            PreferenceStrength::CrossTurnBias,
        );
        assert_eq!(selected.map(|item| item.name.as_str()), Some("code-review"));
    }
}
