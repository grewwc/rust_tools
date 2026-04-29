use crate::ai::{
    agents::AgentManifest,
    mcp::McpClient,
    skills::SkillManifest,
    types::{App, SkillBiasMemory, ToolDefinition},
};
use crate::commonw::configw;
use rust_tools::cw::SkipMap;
use rust_tools::cw::SkipSet;
use chrono::{DateTime, Utc};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use super::{
    DEFAULT_MAX_ITERATIONS, EXECUTOR_MAX_ITERATIONS, TextSimilarityFeatures,
    jaccard_similarity_for_sets, rank_skills_locally_with_model_path, set_intersection_count,
};
use super::intent_recognition::{self, UserIntent};

type ToolDef = ToolDefinition;
type ToolScoreMap = SkipMap<String, f64>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ContextKind {
    Identity,
    Behavior,
    Policy,
    Fact,
}

#[derive(Clone)]
pub(super) struct SystemPromptBuilder {
    sections: Vec<(ContextKind, Option<String>, String)>,
}

impl SystemPromptBuilder {
    fn new() -> Self {
        Self { sections: Vec::new() }
    }

    fn push(&mut self, kind: ContextKind, content: impl Into<String>) {
        let content = content.into();
        if !content.trim().is_empty() {
            self.sections.push((kind, None, content));
        }
    }

    fn push_labeled(&mut self, kind: ContextKind, label: &str, content: impl Into<String>) {
        let content = content.into();
        if !content.trim().is_empty() {
            self.sections.push((kind, Some(label.to_string()), content));
        }
    }

    fn render_system_prompt(&self) -> String {
        let mut out = String::new();
        for (kind, _, content) in &self.sections {
            if *kind == ContextKind::Fact {
                continue;
            }
            let tag = match kind {
                ContextKind::Identity => "identity",
                ContextKind::Behavior => "behavior",
                ContextKind::Policy => "policy",
                ContextKind::Fact => unreachable!(),
            };
            out.push_str(&format!("<{}>\n{}\n</{}>\n", tag, content.trim(), tag));
        }
        out
    }

    fn render_context_reminder(&self) -> Option<String> {
        let facts: Vec<(&Option<String>, &str)> = self
            .sections
            .iter()
            .filter(|(k, _, _)| *k == ContextKind::Fact)
            .filter_map(|(_kind, label, content)| {
                if content.trim().is_empty() {
                    None
                } else {
                    Some((label, content.as_str()))
                }
            })
            .collect();
        if facts.is_empty() {
            return None;
        }
        let mut out = String::from("<system-reminder>\nAs you answer the user's questions, you can use the following context:\n");
        for (label, content) in &facts {
            if let Some(key) = label {
                out.push_str(&format!("# {}\n{}\n\n", key, content.trim()));
            } else {
                out.push_str(&format!("{}\n\n", content.trim()));
            }
        }
        out.push_str("IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.\n</system-reminder>");
        Some(out)
    }
}

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
    restore_agent_context: Option<(Vec<ToolDef>, usize)>,
    builder: SystemPromptBuilder,
    cached_system_prompt: Option<String>,
    cached_context_reminder: Option<Option<String>>,
    matched_skill_name: Option<String>,
    intent: UserIntent,
    skip_recall_by_skill: bool,
}

impl SkillTurnGuard {
    pub(super) fn system_prompt(&mut self) -> &str {
        if self.cached_system_prompt.is_none() {
            self.cached_system_prompt = Some(self.builder.render_system_prompt());
        }
        self.cached_system_prompt.as_deref().unwrap_or_default()
    }

    pub(super) fn context_reminder(&mut self) -> Option<String> {
        if self.cached_context_reminder.is_none() {
            self.cached_context_reminder = Some(self.builder.render_context_reminder());
        }
        self.cached_context_reminder.clone().flatten()
    }

    pub(super) fn push_section(&mut self, kind: ContextKind, content: &str) {
        self.cached_system_prompt = None;
        self.cached_context_reminder = None;
        self.builder.push(kind, content);
    }

    pub(super) fn push_labeled_section(&mut self, kind: ContextKind, label: &str, content: &str) {
        self.cached_system_prompt = None;
        self.cached_context_reminder = None;
        self.builder.push_labeled(kind, label, content);
    }

    pub(super) fn append_system_prompt(&mut self, extra: &str) {
        self.push_section(ContextKind::Fact, extra);
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

    pub(super) fn restore_agent_context(self, app: &mut App) {
        if let Some((tools, max_iterations)) = self.restore_agent_context {
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
    // AIOS: mirror the user-space max_iterations into kernel rlimit so that
    // quota enforcement lives in one place (kernel), not scattered across driver.
    // We map max_iterations -> ResourceLimit.max_tool_calls since each iteration
    // typically issues <= 1 tool call batch.
    {
        use aios_kernel::primitives::ResourceLimit;
        let mut os = app.os.lock().unwrap();
        if let Some(pid) = os.current_process_id() {
            let mut lim = os.rlimit_get(pid).unwrap_or_else(ResourceLimit::unlimited);
            lim.max_tool_calls = max_iterations as u64;
            let _ = os.rlimit_set(pid, lim);
        }
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
        "discover_skills".to_string(),
        "enable_tools".to_string(),
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

fn is_executor_agent(agent: &AgentManifest) -> bool {
    agent.mode == crate::ai::agents::AgentMode::Primary
        && agent
            .tool_groups
            .iter()
            .any(|group| group.eq_ignore_ascii_case("executor") || group.eq_ignore_ascii_case("openclaw"))
}

fn is_executor_skill(skill: &SkillManifest) -> bool {
    skill.tool_groups
        .iter()
        .any(|group| group.eq_ignore_ascii_case("executor") || group.eq_ignore_ascii_case("openclaw"))
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
    skill: Option<&SkillManifest>,
    active_agent: Option<&AgentManifest>,
) -> Vec<ToolDef> {
    if skill.is_some_and(|skill| skill.disable_builtin_tools) {
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

fn resolved_mcp_servers(
    skill: Option<&SkillManifest>,
    active_agent: Option<&AgentManifest>,
) -> Vec<String> {
    let mut servers = Vec::new();
    if let Some(skill) = skill {
        for server in &skill.mcp_servers {
            let server = server.trim();
            if !server.is_empty() && !servers.iter().any(|existing| existing == server) {
                servers.push(server.to_string());
            }
        }
    }
    if let Some(agent) = active_agent {
        for server in &agent.mcp_servers {
            let server = server.trim();
            if !server.is_empty() && !servers.iter().any(|existing| existing == server) {
                servers.push(server.to_string());
            }
        }
    }
    servers
}

fn filter_mcp_tools_by_allowed_servers(
    tools: Vec<ToolDef>,
    allowed_servers: &[String],
) -> Vec<ToolDef> {
    tools.into_iter()
        .filter(|tool| tool_uses_mcp_server(&tool.function.name, allowed_servers))
        .collect()
}

fn mcp_tools_for_turn(
    mcp_client: &McpClient,
    skill: Option<&SkillManifest>,
    active_agent: Option<&AgentManifest>,
) -> Vec<ToolDef> {
    if skill.is_some_and(|skill| skill.disable_mcp_tools) {
        return Vec::new();
    }

    let allowed_servers = resolved_mcp_servers(skill, active_agent);
    if allowed_servers.is_empty() {
        return Vec::new();
    }

    filter_mcp_tools_by_allowed_servers(mcp_client.get_all_tools(), &allowed_servers)
}

struct CapabilityEntry {
    trigger: &'static str,
    tools: &'static [&'static str],
    hint: &'static str,
}

const CAPABILITY_CATALOG: &[CapabilityEntry] = &[
    CapabilityEntry {
        trigger: "remember, memorize, save, store, 记住, 记忆, 保存",
        tools: &["memory_save"],
        hint: "Do NOT just acknowledge — actually call memory_save so the information persists.",
    },
    CapabilityEntry {
        trigger: "recall, search memory, 回忆, 查找记忆",
        tools: &["memory_search", "memory_recent"],
        hint: "",
    },
    CapabilityEntry {
        trigger: "knowledge, decisions, preferences, 知识库, 决策, 偏好",
        tools: &["knowledge_search", "knowledge_list"],
        hint: "",
    },
    CapabilityEntry {
        trigger: "semantic search, vector search, 语义搜索",
        tools: &["knowledge_semantic_search"],
        hint: "",
    },
    CapabilityEntry {
        trigger: "web, real-time, URL, 实时, 网页, 链接",
        tools: &["web_search", "web_fetch"],
        hint: "",
    },
    CapabilityEntry {
        trigger: "Feishu, Lark, 飞书, 文档, docx, wiki, sheet, 多维表格",
        tools: &[
            "mcp_feishu_docs_search",
            "mcp_feishu_docs_get_text_by_url",
            "mcp_feishu_doc_create_from_markdown",
        ],
        hint: "For Feishu/Lark docs or sheets, enable the relevant MCP tools before proceeding.",
    },
    CapabilityEntry {
        trigger: "git status, git diff",
        tools: &["git_status", "git_diff"],
        hint: "",
    },
    CapabilityEntry {
        trigger: "cargo check, cargo test, 编译, 测试",
        tools: &["cargo_check", "cargo_test"],
        hint: "",
    },
    CapabilityEntry {
        trigger: "undo, redo, 撤销, 重做",
        tools: &["undo", "redo"],
        hint: "",
    },
    CapabilityEntry {
        trigger: "compact context, 压缩上下文",
        tools: &["compact_context"],
        hint: "",
    },
    CapabilityEntry {
        trigger: "list directory, 列出目录",
        tools: &["list_directory"],
        hint: "",
    },
];

fn build_capability_catalog(available_tools: &Box<SkipSet<String>>) -> Option<String> {
    let mut lines = Vec::new();
    for entry in CAPABILITY_CATALOG {
        let missing: Vec<String> = entry
            .tools
            .iter()
            .filter(|name| !available_tools.contains_str(name))
            .map(|name| (*name).to_string())
            .collect();
        if missing.is_empty() {
            continue;
        }
        let mut line = format!("- \"{}\" → enable {:?}", entry.trigger, missing);
        if !entry.hint.is_empty() {
            line.push_str(" — ");
            line.push_str(entry.hint);
        }
        lines.push(line);
    }
    if lines.is_empty() {
        return None;
    }
    let mut out = String::from(
        "Capability catalog (not yet loaded — enable as needed):\n\
         When the user's request matches a trigger below, call `enable_tools(operation=enable, tools=[...])` \
         with the listed tools before proceeding.\n",
    );
    for line in lines {
        out.push_str(&line);
        out.push('\n');
    }
    Some(out)
}

fn build_system_prompt(
    active_agent: Option<&AgentManifest>,
    skill: Option<&SkillManifest>,
    available_tools: &Box<SkipSet<String>>,
) -> SystemPromptBuilder {
    let mut b = SystemPromptBuilder::new();

    b.push(ContextKind::Identity, "You are a helpful assistant.");

    if let Some(agent) = active_agent {
        let extra = agent.build_system_prompt();
        if !extra.trim().is_empty() {
            b.push(ContextKind::Identity, format!("Agent enforcement:\n- You MUST follow the active agent profile for behavior, workflow, and safety boundaries.\n- Treat the active agent as the default operating mode for this turn.\n- When a skill is also active, satisfy both the agent profile and the skill instructions.\n\n{}", extra.trim()));
        }
    }
    if let Some(skill) = skill {
        b.push(ContextKind::Identity, "Skill enforcement:\n- You MUST follow the active skill instructions precisely.\n- Do not ignore, weaken, or bypass the skill behavior.\n- If the user request conflicts with the skill, ask a brief clarification aligned with the skill.");
        let extra = skill.build_system_prompt();
        if !extra.trim().is_empty() {
            b.push(ContextKind::Identity, extra.trim());
        }
    }

    b.push(ContextKind::Behavior, "Tool usage:\n- Only rely on tools available in this turn's tool schema.\n- If the answer depends on repo/code facts, inspect with tools before concluding.\n- If the user asks for edits, perform edits with tools instead of only describing them.\n- If the user asks to run/build/test/reproduce, run commands with tools when available.\n- On tool failure, read the error and correct course before answering. Retry with fixed args or switch tools; avoid repeating the same failing call.\n- If `code_search` returns only `No ...` results, broaden scope instead of rerunning unchanged.");

    if has_tool(available_tools, "enable_tools") {
        let mut discovery_policy = String::from(
            "Tool discovery:\n- Do NOT assume all tools are already loaded.\n- Before answering a non-trivial request from scratch, especially an action request or a task that may benefit from a specialized workflow, check whether a skill fits the task. Call `discover_skills` early instead of guessing.\n- If you are unsure how to solve the user's request, feel stuck, or do not know the best next step, use `discover_skills` before continuing. Use `discover_skills` to inspect the currently available skills.\n- If a capability is missing, call `enable_tools(operation=list)` then `enable_tools(operation=enable, tools=[...])` for only what you need.\n- This also applies to MCP tools. For external systems such as Feishu/Lark, docs/wiki/sheets, media generation, presentations, or third-party services, discover and enable the matching `mcp_*` tools before proceeding.",
        );
        if skill.is_none() {
            discovery_policy.push_str(
                "\n- No skill is active yet. For any request that is not obviously a simple direct answer, prefer calling `discover_skills` before giving a freeform response.",
            );
        }
        b.push(ContextKind::Policy, discovery_policy);
        if let Some(catalog) = build_capability_catalog(available_tools) {
            b.push(ContextKind::Fact, catalog);
        }
    }

    if has_tool(available_tools, "memory_save") {
        b.push(ContextKind::Policy, "Memory:\n- When the user asks to remember/save something, call `memory_save`. Do NOT skip the save step.\n- When the user asks to recall saved memory, call `memory_search` or `memory_recent`.");
    }

    if has_tool(available_tools, "plan") || has_tool(available_tools, "spawn_process") {
        b.push(ContextKind::Behavior, "Planning & Sub-process Execution:\n- Simple tasks: act directly without `plan`.\n- Complex multi-step tasks: use `plan` first.\n- If a task can be delegated or run autonomously in the background (e.g. searching broadly, running a heavy test suite), use `spawn_process` to let the Agent OS handle it asynchronously.\n- Child processes can be created with reduced capabilities for least-privilege execution.\n- After spawning, you can either continue your work in parallel, or use `wait_process` to yield control until the child finishes.\n- Use `sleep_process` to suspend yourself for future scheduler ticks.\n- Use `kill_process` to terminate a descendant that is no longer needed.\n- Use `reap_process` to collect a terminated descendant and remove it from the process table.\n- Use `send_ipc_message` and `read_mailbox` to communicate across processes.");
    }

    if has_tool(available_tools, "tool_spawn") || has_tool(available_tools, "tool_wait") {
        b.push(ContextKind::Behavior, "Async Tool Orchestration:\n- Use `tool_spawn` to launch independent builtin or MCP tool calls in parallel.\n- Use `tool_wait` with `wait_policy=all` to join a full batch, or `wait_policy=any` to resume when any branch finishes.\n- When a process wakes because async tools finished, inspect the wake-up mailbox message carefully.\n- After wake-up, prefer `tool_status` to inspect all task states, `tool_wait` to collect newly finished results, or `tool_cancel` to stop low-value still-running branches.\n- Do not blindly wait again if the completed results already support an answer.\n- If mailbox messages already identify the relevant finished tasks, continue reasoning immediately instead of re-querying everything.");
    }

    if has_tool(available_tools, "knowledge_search") || has_tool(available_tools, "knowledge_semantic_search") {
        b.push(ContextKind::Policy, "Knowledge retrieval:\n- If the request involves prior decisions/context/preferences, use `knowledge_search` or `knowledge_semantic_search`.\n- Use `knowledge_list` when the user asks what is remembered.\n- If semantic index seems stale, run `knowledge_rebuild_index`.");
    }

    if has_tool(available_tools, "web_search") || has_tool(available_tools, "web_fetch") {
        b.push(ContextKind::Policy, "Web search:\n- For real-time or time-sensitive topics, use `web_search` first. Do not answer from memory alone.\n- Use `web_fetch` for detailed content from selected URLs.");
    }

    b
}

fn should_skip_recall_for_skill(skill: Option<&SkillManifest>) -> bool {
    skill.is_some_and(|skill| skill.skip_recall || is_executor_skill(skill))
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
    model_path: &std::path::Path,
) -> Option<&'a SkillManifest> {
    select_skill_with_preference_strength(
        skill_manifests,
        question,
        intent,
        preferred_skill_name,
        PreferenceStrength::StrongSticky,
        model_path,
    )
}

fn select_skill_with_preference_strength<'a>(
    skill_manifests: &'a [SkillManifest],
    question: &str,
    intent: &UserIntent,
    preferred_skill_name: Option<&str>,
    strength: PreferenceStrength,
    model_path: &std::path::Path,
) -> Option<&'a SkillManifest> {
    if question.trim().is_empty() || skill_manifests.is_empty() || intent.is_searching_resource("skill") {
        return None;
    }

    let ranked = rank_skills_locally_with_model_path(skill_manifests, question, Some(intent), model_path);
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
                    true,
                ),
            };
        let current_has_signal = current.score >= (threshold - keep_floor_delta).max(0.0)
            || has_skill_signal(current);
        if best.skill.name == current.skill.name {
            return if keep_current_when_best || current.score >= threshold || current_has_signal {
                Some(current.skill)
            } else {
                None
            };
        }

        let effective_current = current.score + sticky_bonus;
        let best_clearly_wins = best.score >= effective_current + switch_margin;
        let best_has_positive_signal = is_positive_skill_winner(best);
        if best_clearly_wins && best_has_positive_signal {
            return Some(best.skill);
        }
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

    if should_abstain_from_skill(best) {
        return None;
    }
    (best.score >= threshold).then_some(best.skill)
}

fn has_skill_signal(item: &super::skill_ranking::ScoredSkill<'_>) -> bool {
    item.embedding_score >= 0.08
        || item.fallback_semantic_score >= 0.08
        || item.model_prior_score >= 0.08
        || item.blended_score >= 0.08
}

fn is_positive_skill_winner(item: &super::skill_ranking::ScoredSkill<'_>) -> bool {
    item.embedding_score >= 0.35
        || item.fallback_semantic_score >= 0.18
        || (item.model_prior_score >= 0.45 && item.model_prior_score > item.none_score)
        || item.blended_score >= 0.35
}

fn should_abstain_from_skill(item: &super::skill_ranking::ScoredSkill<'_>) -> bool {
    item.none_score >= item.blended_score && item.none_score >= 0.5
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
    let active_agent = app.current_agent_manifest.clone();
    let executor_active = skill.as_ref().is_some_and(|s| {
        is_executor_skill(s)
    }) || active_agent.as_ref().is_some_and(is_executor_agent);

    let builtin_tools = builtin_tools_for_skill(skill, active_agent.as_ref());
    let mcp_tools = mcp_tools_for_turn(
        mcp_client,
        skill,
        active_agent.as_ref(),
    );
    let available_tools = available_tool_names(&builtin_tools, &mcp_tools);
    let builder = build_system_prompt(active_agent.as_ref(), skill, &available_tools);
    let max_iterations = resolve_max_iterations(active_agent.as_ref(), executor_active);
    let restore_agent_context =
        activate_skill_context(app, builtin_tools, mcp_tools, max_iterations);

    SkillTurnGuard {
        restore_agent_context,
        builder,
        cached_system_prompt: None,
        cached_context_reminder: None,
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
        select_skill_with_preference(skill_manifests, question, intent, preferred_skill_name, &app.config.skill_match_model_path);
    build_skill_turn_guard(app, mcp_client, skill, intent.clone())
}

pub(super) fn prepare_skill_for_turn(
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
    let cross_turn_preference = cross_turn_preferred_skill_name(app, question, &intent);
    let skill = select_skill_with_preference_strength(
        skill_manifests,
        question,
        &intent,
        cross_turn_preference.as_deref(),
        PreferenceStrength::CrossTurnBias,
        &app.config.skill_match_model_path,
    );

    if debug {
        let ranked = rank_skills_locally_with_model_path(
            skill_manifests,
            question,
            Some(&intent),
            &app.config.skill_match_model_path,
        );
        if let Some(top) = ranked.first() {
            eprintln!(
                "[skills] local top: {} total={:.3} embed={:.3} prior={:.3} fallback={:.3} blend={:.3} none={:.3}",
                top.skill.name,
                top.score,
                top.embedding_score,
                top.model_prior_score,
                top.fallback_semantic_score,
                top.blended_score,
                top.none_score
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
        filter_mcp_tools_by_allowed_servers, looks_like_follow_up_or_same_topic,
        merge_with_runtime_enabled_tools, resolve_max_iterations, select_skill_with_preference,
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
        let tools = builtin_tools_for_skill(None, None);
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

        let prompt = build_system_prompt(None, None, &Box::new(available)).render_system_prompt();
        assert!(prompt.contains("Tool usage:"));
        assert!(prompt.contains("Tool discovery:"));
        assert!(!prompt.contains("Web search:"));
        assert!(!prompt.contains("Knowledge retrieval:"));
    }

    #[test]
    fn system_prompt_mentions_mcp_discovery_when_enable_tools_available() {
        let mut available = SkipSet::new(16);
        available.insert("enable_tools".to_string());
        let prompt = build_system_prompt(None, None, &Box::new(available)).render_system_prompt();
        assert!(prompt.contains("This also applies to MCP tools"));
    }

    #[test]
    fn system_prompt_reminds_model_to_check_skills_when_unsure() {
        let mut available = SkipSet::new(16);
        available.insert("enable_tools".to_string());
        available.insert("discover_skills".to_string());
        let prompt = build_system_prompt(None, None, &Box::new(available)).render_system_prompt();
        assert!(prompt.contains("If you are unsure how to solve the user's request"));
        assert!(prompt.contains("Use `discover_skills` to inspect the currently available skills"));
    }

    #[test]
    fn system_prompt_prefers_discover_skills_before_freeform_when_no_skill_active() {
        let mut available = SkipSet::new(16);
        available.insert("enable_tools".to_string());
        available.insert("discover_skills".to_string());
        let prompt = build_system_prompt(None, None, &Box::new(available)).render_system_prompt();
        assert!(prompt.contains("Call `discover_skills` early instead of guessing"));
        assert!(prompt.contains("No skill is active yet"));
        assert!(prompt.contains("prefer calling `discover_skills` before giving a freeform response"));
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
            triggers: Vec::new(),
            tools: Vec::new(),
            tool_groups: Vec::new(),
            mcp_servers: Vec::new(),
            skip_recall: false,
            disable_builtin_tools: false,
            disable_mcp_tools: false,
            prompt: String::new(),
            system_prompt: None,
            priority: 0,
            source_path: Some(format!("builtin:{name}.skill")),
        }
    }

    fn agent(name: &str, mcp_servers: Vec<&str>) -> AgentManifest {
        AgentManifest {
            name: name.to_string(),
            description: String::new(),
            mode: AgentMode::Primary,
            model: None,
            temperature: None,
            max_steps: None,
            prompt: String::new(),
            system_prompt: None,
            tools: Vec::new(),
            tool_groups: Vec::new(),
            mcp_servers: mcp_servers.into_iter().map(|s| s.to_string()).collect(),
            routing_tags: Vec::new(),
            model_tier: None,
            disabled: false,
            hidden: false,
            color: None,
            source_path: None,
        }
    }

    #[test]
    fn active_agent_mcp_servers_auto_load_matching_mcp_tools() {
        let all_tools = vec![
            tool("mcp_feishu_docs_search"),
            tool("mcp_feishu_docs_get_text_by_url"),
            tool("mcp_other_lookup"),
        ];
        let build_agent = agent("build", vec!["feishu"]);
        let allowed_servers = build_agent.mcp_servers.clone();

        let tools = filter_mcp_tools_by_allowed_servers(all_tools, &allowed_servers);
        let names = tools
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();

        assert!(names.contains(&"mcp_feishu_docs_search".to_string()));
        assert!(names.contains(&"mcp_feishu_docs_get_text_by_url".to_string()));
        assert!(!names.contains(&"mcp_other_lookup".to_string()));
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

    fn model_path() -> std::path::PathBuf {
        crate::ai::driver::skill_match_model::default_model_path()
    }

    #[test]
    fn local_selector_chooses_review_skill_for_review_request() {
        let intent = UserIntent::new(CoreIntent::RequestAction);
        let skills = vec![
            skill("code-review", "Review code changes and highlight bugs"),
            skill("debugger", "Debug runtime failures and collect traces"),
        ];
        let selected = select_skill_with_preference(&skills, "帮我 review 这段 Rust 代码", &intent, None, &model_path());
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
            select_skill_with_preference(&skills, "帮我看一下这段代码哪里有问题", &intent, Some("code-review"), &model_path());
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
            &model_path(),
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
            &model_path(),
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
            &model_path(),
        );
        assert_eq!(selected.map(|item| item.name.as_str()), Some("code-review"));
    }
}
