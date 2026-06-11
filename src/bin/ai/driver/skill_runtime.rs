use crate::ai::{
    agents::{AgentManifest, load_project_instruction_docs},
    mcp::McpClient,
    skills::SkillManifest,
    types::{App, SkillBiasMemory, ToolDefinition},
};
use crate::commonw::configw;
use rust_tools::cw::SkipSet;
use std::sync::{LazyLock, Mutex};

use super::intent_recognition::{self, UserIntent};
use super::{
    DEFAULT_MAX_ITERATIONS, EXECUTOR_MAX_ITERATIONS, TextSimilarityFeatures,
    jaccard_similarity_for_sets, rank_skills_locally_with_model_path, set_intersection_count,
};

type ToolDef = ToolDefinition;

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
        Self {
            sections: Vec::new(),
        }
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
        let mut out = String::from(
            "<system-reminder>\nAs you answer the user's questions, you can use the following context:\n",
        );
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

const DEFAULT_TURN_TOOL_GROUPS: &[&str] = &["core"];
const CURRENT_SKILL_STICKY_BONUS: f64 = 0.7;
const CURRENT_SKILL_KEEP_FLOOR_DELTA: f64 = 0.4;
const SKILL_SWITCH_MARGIN: f64 = 0.8;
const CROSS_TURN_SKILL_STICKY_BONUS: f64 = 0.2;
const CROSS_TURN_SKILL_KEEP_FLOOR_DELTA: f64 = 0.15;
const CROSS_TURN_SKILL_SWITCH_MARGIN: f64 = 0.35;

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

    /// 用于 LLM intent fallback 路径：本地 TF-IDF 给 Casual 但 question
    /// 明显不像闲聊时，turn 准备阶段会调 LLM 升级，结果通过这里回写。
    pub(super) fn set_intent(&mut self, intent: UserIntent) {
        self.intent = intent;
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

fn required_baseline_tool_names() -> Vec<String> {
    vec![
        // discovery / 自助能力：让模型在白名单 skill 下仍能发现并启用更多工具。
        "discover_skills".to_string(),
        "enable_tools".to_string(),
        // 子 Agent 编排：task_* 属于 core 组的常驻能力。skill 用 tools:/tool_groups:
        // 白名单替换工具集时，若不补回，主 Agent 在 skill 激活期间就完全失去委派
        // 子 Agent 的能力（且 :685 的 has_tool 提示门也会一并消失），导致"skill
        // 用得多 → 子 Agent 用得少"。这里像 discover/enable 一样常驻补回。
        // 注意：skill 显式 disable_builtin_tools 时 builtin_tools_for_skill 会提前
        // 返回空集，根本不会走到这里，故该退出语义自然得到尊重。
        "task".to_string(),
        "task_spawn".to_string(),
        "task_wait".to_string(),
        "task_status".to_string(),
    ]
}

fn ensure_required_baseline_tools(mut tools: Vec<ToolDef>) -> Vec<ToolDef> {
    let existing = tools
        .iter()
        .map(|tool| tool.function.name.clone())
        .collect::<Box<SkipSet<_>>>();
    let missing = required_baseline_tool_names()
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

fn manifest_tool_definitions(tool_groups: &[String], tools: &[String]) -> Option<Vec<ToolDef>> {
    if !tool_groups.is_empty() {
        let groups: Vec<&str> = tool_groups.iter().map(|s| s.as_str()).collect();
        return Some(ensure_required_baseline_tools(
            super::super::tools::tool_definitions_for_groups(&groups),
        ));
    }
    if !tools.is_empty() {
        return Some(ensure_required_baseline_tools(
            super::super::tools::get_tool_definitions_by_names(tools),
        ));
    }
    None
}

fn is_executor_agent(agent: &AgentManifest) -> bool {
    agent.mode == crate::ai::agents::AgentMode::Primary
        && agent.tool_groups.iter().any(|group| {
            group.eq_ignore_ascii_case("executor") || group.eq_ignore_ascii_case("openclaw")
        })
}

fn is_executor_skill(skill: &SkillManifest) -> bool {
    skill.tool_groups.iter().any(|group| {
        group.eq_ignore_ascii_case("executor") || group.eq_ignore_ascii_case("openclaw")
    })
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
    // Tools are part of the request payload that providers hash for prompt
    // caching. Reordering on every turn (e.g. by sliding 14-day usage stats)
    // silently invalidates the tools-section of the prompt cache. Pick a
    // deterministic order instead: keep the natural builtin-first/MCP-second
    // grouping and sort each bucket alphabetically by tool name. This is
    // stable across turns regardless of recent tool_stat memory.
    rust_tools::sortw::stable_sort_by(&mut builtin, |a, b| a.function.name.cmp(&b.function.name));
    rust_tools::sortw::stable_sort_by(&mut mcp, |a, b| a.function.name.cmp(&b.function.name));
    let mut all = Vec::with_capacity(builtin.len() + mcp.len());
    all.append(&mut builtin);
    all.append(&mut mcp);
    all
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
    tools
        .into_iter()
        .filter(|tool| tool_uses_mcp_server(&tool.function.name, allowed_servers))
        .collect()
}

fn select_mcp_tools(
    all_tools: Vec<ToolDef>,
    skill: Option<&SkillManifest>,
    active_agent: Option<&AgentManifest>,
) -> Vec<ToolDef> {
    if skill.is_some_and(|skill| skill.disable_mcp_tools) {
        return Vec::new();
    }

    let allowed_servers = resolved_mcp_servers(skill, active_agent);
    if allowed_servers.is_empty() {
        return all_tools;
    }

    filter_mcp_tools_by_allowed_servers(all_tools, &allowed_servers)
}

fn mcp_tools_for_turn(
    mcp_client: &McpClient,
    skill: Option<&SkillManifest>,
    active_agent: Option<&AgentManifest>,
) -> Vec<ToolDef> {
    select_mcp_tools(mcp_client.get_all_tools(), skill, active_agent)
}

struct CapabilityEntry {
    use_case: &'static str,
    tools: &'static [&'static str],
    hint: &'static str,
}

const CAPABILITY_CATALOG: &[CapabilityEntry] = &[
    CapabilityEntry {
        use_case: "Persist user-facing facts, preferences, or decisions for later recall.",
        tools: &["knowledge_save"],
        hint: "Do NOT just acknowledge — actually call knowledge_save so the information persists in the user-facing knowledge base.",
    },
    CapabilityEntry {
        use_case: "Recall previously saved user-facing information.",
        tools: &["knowledge_search", "knowledge_list"],
        hint: "",
    },
    CapabilityEntry {
        use_case: "Inspect saved knowledge such as prior decisions, preferences, or known context.",
        tools: &["knowledge_search", "knowledge_list"],
        hint: "",
    },
    CapabilityEntry {
        use_case: "Search remembered knowledge by semantic similarity rather than exact wording.",
        tools: &["knowledge_semantic_search"],
        hint: "",
    },
    CapabilityEntry {
        use_case: "Handle live web information, current events, or URL-based research.",
        tools: &["web_search", "web_fetch"],
        hint: "",
    },
    CapabilityEntry {
        use_case: "Work with Feishu/Lark documents, wikis, sheets, or doc exports.",
        tools: &[
            "mcp_feishu_docs_search",
            "mcp_feishu_docs_get_text_by_url",
            "mcp_feishu_doc_create_from_markdown",
        ],
        hint: "For Feishu/Lark docs or sheets, enable the relevant MCP tools before proceeding.",
    },
    CapabilityEntry {
        use_case: "Inspect repository status or diffs.",
        tools: &["git_status", "git_diff"],
        hint: "",
    },
    CapabilityEntry {
        use_case: "Build, compile, or run Rust test workflows.",
        tools: &["cargo_check", "cargo_test"],
        hint: "",
    },
    CapabilityEntry {
        use_case: "Undo or redo prior editor changes.",
        tools: &["undo", "redo"],
        hint: "",
    },
    CapabilityEntry {
        use_case: "Compact or trim conversation context to recover token budget.",
        tools: &["compact_context"],
        hint: "",
    },
    CapabilityEntry {
        use_case: "Inspect directory contents or enumerate files in a folder.",
        tools: &["list_directory"],
        hint: "",
    },
];

fn build_capability_catalog(available_tools: &Box<SkipSet<String>>) -> Option<String> {
    // 缓存键：可用工具集合的哈希。同一会话/同一 agent 在工具集不变时复用。
    let cache_key = capability_catalog_cache_key(available_tools);
    if let Some(hit) = capability_catalog_cache_get(cache_key) {
        return hit;
    }

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
        let mut line = format!("- Need: {} → enable {:?}", entry.use_case, missing);
        if !entry.hint.is_empty() {
            line.push_str(" — ");
            line.push_str(entry.hint);
        }
        lines.push(line);
    }
    let result = if lines.is_empty() {
        None
    } else {
        let mut out = String::from(
            "Capability catalog (not yet loaded — enable as needed):\n\
             When the task clearly requires one of the capabilities below, call `enable_tools(operation=enable, tools=[...])` \
             with the listed tools before proceeding.\n",
        );
        for line in lines {
            out.push_str(&line);
            out.push('\n');
        }
        Some(out)
    };
    capability_catalog_cache_put(cache_key, result.clone());
    result
}

fn capability_catalog_cache_key(available_tools: &Box<SkipSet<String>>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut tools: Vec<String> = available_tools.iter().map(|s| s.to_string()).collect();
    tools.sort();
    let mut hasher = rustc_hash::FxHasher::default();
    for t in &tools {
        t.hash(&mut hasher);
        0u8.hash(&mut hasher);
    }
    hasher.finish()
}

static CAPABILITY_CATALOG_CACHE: LazyLock<Mutex<Option<(u64, Option<String>)>>> =
    LazyLock::new(|| Mutex::new(None));

fn capability_catalog_cache_get(key: u64) -> Option<Option<String>> {
    let cache = CAPABILITY_CATALOG_CACHE.lock().ok()?;
    cache
        .as_ref()
        .filter(|(k, _)| *k == key)
        .map(|(_, v)| v.clone())
}

fn capability_catalog_cache_put(key: u64, value: Option<String>) {
    if let Ok(mut cache) = CAPABILITY_CATALOG_CACHE.lock() {
        *cache = Some((key, value));
    }
}

fn build_project_instruction_prompt() -> Option<String> {
    let docs = load_project_instruction_docs();
    if docs.is_empty() {
        return None;
    }

    let mut out = String::from(
        "Project-local instructions:\n\
         - The current working directory provides project-specific instruction documents.\n\
         - Follow these repo-local constraints and preferences unless they conflict with higher-priority system, developer, or user instructions.\n",
    );
    for doc in docs {
        out.push_str(&format!("\nFrom {}:\n{}\n", doc.path, doc.content.trim()));
    }
    Some(out)
}

fn build_system_prompt(
    active_agent: Option<&AgentManifest>,
    skill: Option<&SkillManifest>,
    available_tools: &Box<SkipSet<String>>,
) -> SystemPromptBuilder {
    let mut b = SystemPromptBuilder::new();

    // Identity 段：合并通用 identity + agent / skill enforcement，避免 4 段
    // 重复 "you must follow ..." 充斥 prompt cache。
    let mut identity = String::from(
        "You are a highly capable general-purpose AI assistant. You help users plan, research, write, analyze, and act across any domain — code is one of many, not the only one. Adapt your approach to the task at hand: use code/tooling when the request is technical, and plain reasoning or research when it is not. Aim to be sharp and to the point — answer what was asked, not more.",
    );

    let agent_extra = active_agent
        .map(|agent| agent.build_system_prompt())
        .filter(|s| !s.trim().is_empty());
    let skill_extra = skill
        .map(|skill| skill.build_system_prompt())
        .filter(|s| !s.trim().is_empty());

    if agent_extra.is_some() || skill_extra.is_some() {
        identity.push_str("\n\nEnforcement: follow the active agent profile (if any) and skill instructions (if any) precisely; if both are active, satisfy both; on conflict with the user request, ask one brief clarification aligned with the more specific (skill > agent) layer.");
    }
    if let Some(extra) = agent_extra {
        identity.push_str("\n\n[Agent profile]\n");
        identity.push_str(extra.trim());
    }
    if let Some(extra) = skill_extra {
        identity.push_str("\n\n[Skill instructions]\n");
        identity.push_str(extra.trim());
    }
    b.push(ContextKind::Identity, identity);

    b.push(ContextKind::Behavior, "Response style:\n- Lead with the answer or the action; skip preamble, restating the question, and meta commentary like \"Let me…\" / \"I'll now…\".\n- One sentence beats three. Drop filler, transitions, and recaps the user can already see.\n- Default to short, direct prose. Use lists/sections only when they materially help; never pad output to look thorough.\n- Do not narrate tool calls before or after running them — let the calls and their results speak. A brief status line is only appropriate at real milestones or when a plan changes.\n- Conciseness must NOT come at the cost of correctness: for non-trivial requests, think the problem through (read code, check facts) BEFORE answering. If you are uncertain, say so in one line and gather evidence rather than guessing tersely.\n- When stating a conclusion that depends on code, cite the exact file/line; do not summarize from memory.");
    b.push(ContextKind::Behavior, "Tool usage:\n- Only rely on tools available in this turn's tool schema.\n- If the answer depends on repo/code facts, inspect with tools before concluding.\n- If the user asks for edits, perform edits with tools instead of only describing them.\n- If the user asks to run/build/test/reproduce, run commands with tools when available.\n- On tool failure, read the error and correct course before answering. Retry with fixed args or switch tools; avoid repeating the same failing call.\n- If `code_search` returns only `No ...` results, broaden scope instead of rerunning unchanged.");

    if let Some(project_prompt) = build_project_instruction_prompt() {
        b.push(ContextKind::Policy, project_prompt);
    }

    if let Some(kind) = crate::ai::agents::detect_project_kind_from_cwd() {
        // 把识别出的项目类型 + 默认构建/测试约定作为 Fact 段注入，
        // 让 LLM 不必猜测 `cargo` / `npm` / `go` 该用哪个。
        b.push_labeled(
            ContextKind::Fact,
            "Project Type",
            kind.prompt_hint().to_string(),
        );
    }

    if has_tool(available_tools, "enable_tools") {
        // Capability catalog（如有）按 trigger→tool 给出事实性精确映射；
        // discovery 段是 actionable 政策。两者 ContextKind 不同，挂在一起
        // 重叠较小且各自承担不同职责（catalog 让模型知道有什么，policy
        // 告诉模型何时去发现/启用）。保持双挂以避免回归测试用例期望的
        // mcp 提示与 "No skill is active yet" 提示丢失。
        if let Some(catalog) = build_capability_catalog(available_tools) {
            b.push(ContextKind::Fact, catalog);
        }
        let discovery_policy = if skill.is_none() {
            "Tool discovery:\n- Not all tools are loaded. Use `discover_skills` for specialized workflows, or `enable_tools(operation=list)` then `enable_tools(operation=enable, tools=[...])` for specific tools.\n- If the user names a workflow/tool/domain that may have a local skill (for example an internal service, CLI, log system, or incident workflow), call `discover_skills` with the named keyword before inventing commands.\n- After `discover_skills`, if one listed skill clearly matches the task, call `activate_skill(name=...)` to load its prompt and tools. Do not activate a skill that does not clearly match.\n- For external systems (Feishu/Lark, web, etc.), discover and enable matching `mcp_*` tools first.\n- No skill is active yet. For non-trivial requests, prefer calling `discover_skills` before giving a freeform response."
        } else {
            "Tool discovery:\n- Not all tools are loaded. If a capability is missing, use `enable_tools(operation=list)` then `enable_tools(operation=enable, tools=[...])` for only what you need.\n- For external systems (Feishu/Lark, web, etc.), discover and enable matching `mcp_*` tools first."
        };
        b.push(ContextKind::Policy, discovery_policy);
    }

    if has_tool(available_tools, "knowledge_save") {
        b.push(ContextKind::Policy, "Knowledge save:\n- When the user explicitly asks to remember/save/store something for later use, call `knowledge_save`. Do NOT skip the save step.\n- If the user states a durable preference, standing constraint, or stable decision that will likely matter later, proactively call `knowledge_save` even without explicit remember wording.\n- When the user asks what has been remembered or asks to recall saved information, use `knowledge_search` or `knowledge_list`.");
    }

    if has_tool(available_tools, "plan") || has_tool(available_tools, "spawn_process") {
        b.push(ContextKind::Behavior, "Planning & Sub-process Execution:\n- Simple tasks: act directly without `plan`.\n- Complex multi-step tasks: use `plan` first.\n- If a task can be delegated or run autonomously in the background (e.g. searching broadly, running a heavy test suite), use `spawn_process` to let the Agent OS handle it asynchronously.\n- Child processes can be created with reduced capabilities for least-privilege execution.\n- After spawning, you can either continue your work in parallel, or use `wait_process` to yield control until the child finishes.\n- Use `sleep_process` to suspend yourself for future scheduler ticks.\n- Use `kill_process` to terminate a descendant that is no longer needed.\n- Use `reap_process` to collect a terminated descendant and remove it from the process table.\n- Use `send_ipc_message` and `read_mailbox` to communicate across processes.");
    }

    if has_tool(available_tools, "tool_spawn") || has_tool(available_tools, "tool_wait") {
        b.push(ContextKind::Behavior, "Async Tool Orchestration:\n- Use `tool_spawn` to launch independent builtin or MCP tool calls in parallel.\n- Use `tool_wait` with `wait_policy=all` to join a full batch, or `wait_policy=any` to resume when any branch finishes.\n- When a process wakes because async tools finished, inspect the wake-up mailbox message carefully.\n- After wake-up, prefer `tool_status` to inspect all task states, `tool_wait` to collect newly finished results, or `tool_cancel` to stop low-value still-running branches.\n- Do not blindly wait again if the completed results already support an answer.\n- If mailbox messages already identify the relevant finished tasks, continue reasoning immediately instead of re-querying everything.");
    }

    if has_tool(available_tools, "task_spawn") || has_tool(available_tools, "task_wait") {
        b.push(ContextKind::Behavior, "Async Subagent Orchestration (task_*):\n- `task_spawn` launches a subagent task asynchronously and returns a task_id immediately. Fan out multiple subagent tasks in parallel when their work is independent.\n- `task_wait` collects results. Its `timeout_secs` is a per-call wait budget — when it elapses without satisfying the policy, the call returns the already-collected results plus a clear note that the remaining subagents are still running. This is NOT a stall: re-call `task_wait` with the same task_ids to keep waiting (or pass `wait_policy=\"any\"` to wake on the first finisher).\n- `task_status` is non-blocking: use it to peek at progress without consuming results.\n- There is NO `task_cancel`. Do not invent it. If you need to abandon work, just stop calling `task_wait` and let the subagent run to completion in the background; its result will be reaped by the kernel.\n- Do not confuse the `task_*` family with `tool_*` — they are distinct. `tool_wait` cannot consume a task_id, and `task_wait` cannot consume a tool ticket.");
    }

    if has_tool(available_tools, "knowledge_search")
        || has_tool(available_tools, "knowledge_semantic_search")
    {
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
        2.5
    } else if skill_count > 5 {
        2.0
    } else {
        1.75
    };
    match intent.core {
        intent_recognition::CoreIntent::RequestAction => base.max(2.0),
        intent_recognition::CoreIntent::SeekSolution => base.max(1.85),
        intent_recognition::CoreIntent::QueryConcept => base.max(2.5),
        intent_recognition::CoreIntent::Casual => base.max(2.5),
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
    let token_similarity =
        jaccard_similarity_for_sets(&current_features.token_set, &previous_features.token_set);
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

    let overlap = set_intersection_count(&current_features.token_set, &previous_features.token_set);
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
    let _ = intent;
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
    if question.trim().is_empty() || skill_manifests.is_empty() {
        return None;
    }

    let ranked =
        rank_skills_locally_with_model_path(skill_manifests, question, Some(intent), model_path);
    let Some(best) = ranked.first() else {
        return None;
    };
    let threshold = skill_selection_threshold(intent, skill_manifests.len());
    let preferred =
        preferred_skill_name.and_then(|name| ranked.iter().find(|item| item.skill.name == name));

    if let Some(current) = preferred {
        let (
            sticky_bonus,
            keep_floor_delta,
            switch_margin,
            keep_current_when_best,
            keep_below_threshold,
        ) = match strength {
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
        let current_has_signal =
            current.score >= (threshold - keep_floor_delta).max(0.0) || has_skill_signal(current);
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
        let allow_switch = match strength {
            PreferenceStrength::StrongSticky => best_clearly_wins && best_has_positive_signal,
            PreferenceStrength::CrossTurnBias => best_clearly_wins,
        };
        if allow_switch {
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
        || item.blended_score >= 0.08
}

fn is_positive_skill_winner(item: &super::skill_ranking::ScoredSkill<'_>) -> bool {
    item.embedding_score >= 0.30
        || item.fallback_semantic_score >= 0.15
        || item.blended_score >= 0.30
}

fn should_abstain_from_skill(item: &super::skill_ranking::ScoredSkill<'_>) -> bool {
    item.blended_score < 0.08
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
    let executor_active = skill.as_ref().is_some_and(|s| is_executor_skill(s))
        || active_agent.as_ref().is_some_and(is_executor_agent);

    let builtin_tools = builtin_tools_for_skill(skill, active_agent.as_ref());
    let mcp_tools = mcp_tools_for_turn(mcp_client, skill, active_agent.as_ref());
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
    let skill = select_skill_with_preference(
        skill_manifests,
        question,
        intent,
        preferred_skill_name,
        &app.config.skill_match_model_path,
    );
    build_skill_turn_guard(app, mcp_client, skill, intent.clone())
}

/// 模型通过 `activate_skill` 工具显式请求激活某个 skill 时走这里：直接按名字
/// 命中并强制激活其 prompt + 工具集，跳过自动路由的打分/阈值/门控。
///
/// "别乱用"由工具侧（名字必须真实存在、描述明确要求"clearly matches"才调用）和
/// 这里的名字校验共同兜底；命中后写入 cross-turn bias，让后续 iteration 通过
/// sticky 机制保持，不会被自动路由立刻切走。
pub(super) fn force_activate_named_skill(
    app: &mut App,
    mcp_client: &McpClient,
    skill_manifests: &[SkillManifest],
    question: &str,
    requested_name: &str,
    intent: &UserIntent,
) -> Option<SkillTurnGuard> {
    let skill = skill_manifests.iter().find(|s| s.name == requested_name)?;
    update_cross_turn_skill_bias(app, question, Some(skill));
    let mut guard = build_skill_turn_guard(app, mcp_client, Some(skill), intent.clone());
    guard.matched_skill_name = Some(skill.name.clone());
    guard.skip_recall_by_skill = should_skip_recall_for_skill(Some(skill));
    Some(guard)
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

    // 用户通过 `@skills:<name>` 在输入框中显式选择的强制 skill 优先于自动路由。
    // 这是 per-turn 语义：消费后立即清空，下一轮不再强制注入。
    if let Some(forced) = app.forced_skill.take() {
        if let Some(skill) = skill_manifests
            .iter()
            .find(|s| s.name == forced)
            .or_else(|| {
                skill_manifests
                    .iter()
                    .find(|s| s.name.eq_ignore_ascii_case(&forced))
            })
        {
            let name = skill.name.clone();
            if let Some(guard) =
                force_activate_named_skill(app, mcp_client, skill_manifests, question, &name, &intent)
            {
                if debug {
                    eprintln!("[skills] forced via @skills: {}", name);
                }
                return guard;
            }
        } else if debug {
            eprintln!("[skills] forced @skills:{} not found, falling back to auto-route", forced);
        }
    }

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
        PreferenceStrength, build_project_instruction_prompt, build_system_prompt,
        builtin_tools_for_skill, ensure_required_baseline_tools,
        filter_mcp_tools_by_allowed_servers, looks_like_follow_up_or_same_topic,
        merge_with_runtime_enabled_tools, resolve_max_iterations, select_mcp_tools,
        select_skill_with_preference, select_skill_with_preference_strength, tool_uses_mcp_server,
    };
    use crate::ai::agents::{AgentManifest, AgentMode};
    use crate::ai::driver::intent_recognition::{CoreIntent, UserIntent};
    use crate::ai::driver::runtime_ctx::SUBAGENT_CWD;
    use crate::ai::skills::SkillManifest;
    use crate::ai::tools::enable_tools::set_explicit_enabled_tool_names;
    use crate::ai::types::{FunctionDefinition, ToolDefinition};
    use rust_tools::cw::SkipSet;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{LazyLock, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};

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
        assert_eq!(
            resolve_max_iterations(None, true),
            super::super::EXECUTOR_MAX_ITERATIONS
        );
        assert_eq!(
            resolve_max_iterations(None, false),
            super::super::DEFAULT_MAX_ITERATIONS
        );
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
        assert!(names.iter().any(|name| name == "knowledge_save"));
        assert!(names.iter().any(|name| name == "knowledge_search"));
        assert!(!names.iter().any(|name| name == "web_search"));
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
    fn system_prompt_enforces_concise_response_style_with_correctness_safeguard() {
        let available = SkipSet::new(16);
        let prompt = build_system_prompt(None, None, &Box::new(available)).render_system_prompt();
        // 风格段必须存在，且要求"先答后说、不啰嗦"
        assert!(prompt.contains("Response style:"));
        assert!(prompt.contains("Lead with the answer"));
        // 必须保留"简洁不能换错"的安全垫，避免过度精简导致错误判断
        assert!(prompt.contains("Conciseness must NOT come at the cost of correctness"));
    }

    #[test]
    fn system_prompt_mentions_mcp_discovery_when_enable_tools_available() {
        let mut available = SkipSet::new(16);
        available.insert("enable_tools".to_string());
        let prompt = build_system_prompt(None, None, &Box::new(available)).render_system_prompt();
        assert!(prompt.contains("discover and enable matching `mcp_*` tools first"));
    }

    #[test]
    fn system_prompt_reminds_model_to_check_skills_when_unsure() {
        let mut available = SkipSet::new(16);
        available.insert("enable_tools".to_string());
        available.insert("discover_skills".to_string());
        let prompt = build_system_prompt(None, None, &Box::new(available)).render_system_prompt();
        assert!(prompt.contains("discover_skills"));
        assert!(prompt.contains("enable_tools"));
    }

    #[test]
    fn system_prompt_prefers_discover_skills_before_freeform_when_no_skill_active() {
        let mut available = SkipSet::new(16);
        available.insert("enable_tools".to_string());
        available.insert("discover_skills".to_string());
        let prompt = build_system_prompt(None, None, &Box::new(available)).render_system_prompt();
        assert!(prompt.contains("discover_skills"));
        assert!(prompt.contains("No skill is active yet"));
        assert!(
            prompt.contains("prefer calling `discover_skills` before giving a freeform response")
        );
        assert!(prompt.contains("call `discover_skills` with the named keyword"));
    }

    #[test]
    fn system_prompt_uses_knowledge_save_for_user_memory_requests() {
        let mut available = SkipSet::new(16);
        available.insert("knowledge_save".to_string());
        available.insert("knowledge_search".to_string());
        available.insert("knowledge_list".to_string());

        let prompt = build_system_prompt(None, None, &Box::new(available)).render_system_prompt();
        assert!(prompt.contains("Knowledge save:"));
        assert!(prompt.contains("call `knowledge_save`"));
        assert!(prompt.contains("`knowledge_search` or `knowledge_list`"));
        assert!(!prompt.contains("call `memory_save`"));
    }

    #[test]
    fn capability_catalog_routes_remember_requests_to_knowledge_save() {
        let mut available = SkipSet::new(16);
        available.insert("enable_tools".to_string());

        let catalog = super::build_capability_catalog(&Box::new(available)).expect("catalog");
        assert!(catalog.contains("knowledge_save"));
        assert!(!catalog.contains("memory_save"));
    }

    fn temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        path.push(format!(
            "rust_tools_skill_runtime_{name}_{}_{}",
            std::process::id(),
            nanos
        ));
        path
    }

    #[test]
    fn project_instruction_prompt_includes_repo_docs_from_cwd_scope() {
        let root = temp_dir("project_prompt");
        let nested = root.join("apps/web/src");
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(&nested).unwrap();
        fs::write(root.join("AGENTS.md"), "Use cargo fmt before commit.\n").unwrap();
        fs::write(root.join("apps/web/claude.md"), "Web app uses pnpm.\n").unwrap();

        let prompt = SUBAGENT_CWD
            .sync_scope(nested.clone(), build_project_instruction_prompt)
            .expect("project instruction prompt");

        assert!(prompt.contains("Project-local instructions:"));
        assert!(prompt.contains("AGENTS.md"));
        assert!(prompt.contains("Use cargo fmt before commit."));
        assert!(prompt.contains("claude.md"));
        assert!(prompt.contains("Web app uses pnpm."));

        let _ = fs::remove_dir_all(root);
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
            excludes: Vec::new(),
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
    fn agent_without_mcp_servers_field_exposes_all_mcp_tools() {
        let all_tools = vec![
            tool("mcp_feishu_docs_search"),
            tool("mcp_ocr_extract"),
            tool("mcp_other_lookup"),
        ];
        let build_agent = agent("build", vec![]);

        let tools = select_mcp_tools(all_tools, None, Some(&build_agent));
        let names = tools
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();

        assert!(names.contains(&"mcp_feishu_docs_search".to_string()));
        assert!(names.contains(&"mcp_ocr_extract".to_string()));
        assert!(names.contains(&"mcp_other_lookup".to_string()));
    }

    #[test]
    fn no_active_agent_or_skill_falls_back_to_all_mcp_tools() {
        let all_tools = vec![tool("mcp_feishu_docs_search"), tool("mcp_other_lookup")];

        let tools = select_mcp_tools(all_tools, None, None);
        let names = tools
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();

        assert_eq!(names.len(), 2);
        assert!(names.contains(&"mcp_feishu_docs_search".to_string()));
        assert!(names.contains(&"mcp_other_lookup".to_string()));
    }

    #[test]
    fn skill_disable_mcp_tools_overrides_agent_default_fallback() {
        let all_tools = vec![tool("mcp_feishu_docs_search"), tool("mcp_other_lookup")];
        let build_agent = agent("build", vec![]);
        let mut s = skill("focus", "");
        s.disable_mcp_tools = true;

        let tools = select_mcp_tools(all_tools, Some(&s), Some(&build_agent));
        assert!(tools.is_empty());
    }

    #[test]
    fn explicit_agent_whitelist_still_narrows_when_set() {
        let all_tools = vec![tool("mcp_feishu_docs_search"), tool("mcp_ocr_extract")];
        let agent = agent("build", vec!["feishu"]);

        let tools = select_mcp_tools(all_tools, None, Some(&agent));
        let names = tools
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["mcp_feishu_docs_search".to_string()]);
    }

    #[test]
    fn runtime_enabled_tools_are_preserved_when_refreshing_context() {
        let _guard = EXPLICIT_TOOL_TEST_GUARD.lock().unwrap();
        set_explicit_enabled_tool_names(vec!["enable_tools".to_string(), "web_search".to_string()]);
        let merged = merge_with_runtime_enabled_tools(
            vec![tool("code_search"), tool("read_file"), tool("enable_tools")],
            vec![],
            &[
                tool("code_search"),
                tool("enable_tools"),
                tool("web_search"),
            ],
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
    fn explicit_tool_lists_keep_baseline_entries_available() {
        let merged = ensure_required_baseline_tools(vec![tool("code_search")]);
        let names = merged
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"enable_tools".to_string()));
        assert!(names.contains(&"discover_skills".to_string()));
        assert!(names.contains(&"code_search".to_string()));
        // 子 Agent 编排能力应作为 baseline 常驻补回，避免 skill 白名单把 task_*
        // 全部剔除导致主 Agent 失去委派子 Agent 的能力。
        assert!(names.contains(&"task".to_string()));
        assert!(names.contains(&"task_spawn".to_string()));
        assert!(names.contains(&"task_wait".to_string()));
        assert!(names.contains(&"task_status".to_string()));
        // 其它非 baseline 的内置工具仍不应被无端带入白名单。
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
        let selected = select_skill_with_preference(
            &skills,
            "帮我 review 这段 Rust 代码",
            &intent,
            None,
            &model_path(),
        );
        assert_eq!(selected.map(|item| item.name.as_str()), Some("code-review"));
    }

    #[test]
    fn local_selector_prefers_current_skill_without_clear_winner() {
        let intent = UserIntent::new(CoreIntent::RequestAction);
        let skills = vec![
            skill("code-review", "Review code changes and summarize defects"),
            skill(
                "debugger",
                "Debug runtime failures, panic, and stack traces",
            ),
        ];
        let selected = select_skill_with_preference(
            &skills,
            "帮我看一下这段代码哪里有问题",
            &intent,
            Some("code-review"),
            &model_path(),
        );
        assert_eq!(selected.map(|item| item.name.as_str()), Some("code-review"));
    }

    #[test]
    fn local_selector_switches_when_new_skill_is_significantly_better() {
        let intent = UserIntent::new(CoreIntent::RequestAction);
        let debugger = skill(
            "debugger",
            "Debug panic crash stack trace runtime failure logs",
        );
        let skills = vec![
            skill("code-review", "Review code changes and summarize defects"),
            debugger,
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
            skill(
                "debugger",
                "Debug panic crash stack trace runtime failure logs",
            ),
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
