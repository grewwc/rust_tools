use crate::ai::{
    agents::{AgentManifest, load_project_instruction_docs},
    mcp::McpClient,
    skills::SkillManifest,
    types::{App, SkillBiasMemory, ToolDefinition},
};
use crate::commonw::configw;
use rust_tools::cw::SkipSet;
use std::sync::{LazyLock, Mutex};

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
        // 按语义类别分组渲染：同一 kind（identity/behavior/policy）的所有段落
        // 合并进同一对 tag，组内保持插入顺序。这样 persona 等"在 build_system_prompt
        // 之后追加"的 identity 段不会被甩到 prompt 末尾，而是与通用 identity 聚拢；
        // behavior/policy 也不再因 push 时机裂成多簇，减少 tag 噪音、让优先级层次
        // 对模型更清晰。Fact 段不在 system prompt 渲染（走 context reminder 注入当前
        // user 消息），故不在白名单内、自然被排除。
        const RENDER_ORDER: [(ContextKind, &str); 3] = [
            (ContextKind::Identity, "identity"),
            (ContextKind::Behavior, "behavior"),
            (ContextKind::Policy, "policy"),
        ];
        let mut out = String::new();
        for (group_kind, tag) in RENDER_ORDER {
            let mut group = String::new();
            for (kind, _, content) in &self.sections {
                if *kind != group_kind {
                    continue;
                }
                let trimmed = content.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if !group.is_empty() {
                    group.push_str("\n\n");
                }
                group.push_str(trimmed);
            }
            if !group.is_empty() {
                out.push_str(&format!("<{}>\n{}\n</{}>\n", tag, group, tag));
            }
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
    // discovery / 自助能力 + 基础只读 / 检索能力 + 子 Agent 编排能力：
    // 这些都是白名单 skill 替换工具集时也必须常驻补回的 baseline。与进程级
    // allowed_tools whitelist 共享同一份清单，避免"schema 里可见，但执行时被
    // whitelist 拦掉"的分叉。
    crate::ai::tools::baseline_tool_names()
        .iter()
        .map(|name| (*name).to_string())
        .collect()
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

fn push_tool_guidance_section(
    builder: &mut SystemPromptBuilder,
    kind: ContextKind,
    title: &str,
    lines: Vec<String>,
) {
    if lines.is_empty() {
        return;
    }

    let mut section = String::from(title);
    section.push('\n');
    for line in lines {
        section.push_str("- ");
        section.push_str(&line);
        section.push('\n');
    }
    if section.ends_with('\n') {
        section.pop();
    }
    builder.push(kind, section);
}

fn backticked_tool(name: &str) -> String {
    format!("`{name}`")
}

fn format_tool_names(names: &[&str]) -> String {
    match names {
        [] => String::new(),
        [only] => backticked_tool(only),
        [first, second] => format!("{} and {}", backticked_tool(first), backticked_tool(second)),
        _ => {
            let mut rendered = names
                .iter()
                .map(|name| backticked_tool(name))
                .collect::<Vec<_>>();
            let last = rendered.pop().unwrap_or_default();
            format!("{}, and {}", rendered.join(", "), last)
        }
    }
}

fn available_tool_names_in_order<'a>(
    available_tools: &Box<SkipSet<String>>,
    candidates: &'a [&'a str],
) -> Vec<&'a str> {
    candidates
        .iter()
        .copied()
        .filter(|name| has_tool(available_tools, name))
        .collect()
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
    if let Some(agent) = active_agent
        && !agent.disable_mcp_tools
    {
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
    let skill_declares_mcp_servers = skill.is_some_and(|skill| !skill.mcp_servers.is_empty());
    if active_agent.is_some_and(|agent| agent.disable_mcp_tools) && !skill_declares_mcp_servers {
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

fn build_hidden_mcp_tool_catalog(
    all_mcp_tools: &[ToolDef],
    loaded_mcp_tools: &[ToolDef],
) -> Option<String> {
    let loaded_names = loaded_mcp_tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect::<Box<SkipSet<_>>>();
    let mut hidden: Vec<String> = all_mcp_tools
        .iter()
        .map(|tool| tool.function.name.clone())
        .filter(|name| !loaded_names.contains_str(name))
        .collect();
    if hidden.is_empty() {
        return None;
    }
    rust_tools::sortw::stable_sort_by(&mut hidden, |a, b| a.cmp(b));
    hidden.dedup();

    const MAX_DISPLAY: usize = 8;
    let displayed = hidden
        .iter()
        .take(MAX_DISPLAY)
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ");
    let remaining = hidden.len().saturating_sub(MAX_DISPLAY);

    let mut out = format!(
        "Configured MCP tools are available but not loaded in this turn.\n\
         If the task needs an external system or MCP-backed capability, call `enable_tools(operation=list)` first, then \
         `enable_tools(operation=enable, tools=[...])` with the exact names you need.\n\
         Example available MCP tools: {}",
        displayed
    );
    if remaining > 0 {
        out.push_str(&format!(", and {remaining} more"));
    }
    out.push('.');
    Some(out)
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

fn push_project_instruction_context(builder: &mut SystemPromptBuilder) {
    if let Some(project_prompt) = build_project_instruction_prompt() {
        builder.push(ContextKind::Policy, project_prompt);
    }
}

fn push_project_type_context(builder: &mut SystemPromptBuilder) {
    if let Some(kind) = crate::ai::agents::detect_project_kind_from_cwd() {
        // 把识别出的项目类型 + 默认构建/测试约定作为 Fact 段注入，
        // 让 LLM 不必猜测 `cargo` / `npm` / `go` 该用哪个。
        builder.push_labeled(
            ContextKind::Fact,
            "Project Type",
            kind.prompt_hint().to_string(),
        );
    }
}

fn push_project_context(builder: &mut SystemPromptBuilder) {
    push_project_instruction_context(builder);
    push_project_type_context(builder);
}

fn build_system_prompt(
    active_agent: Option<&AgentManifest>,
    skill: Option<&SkillManifest>,
    available_tools: &Box<SkipSet<String>>,
) -> SystemPromptBuilder {
    let mut b = SystemPromptBuilder::new();

    // Identity 段：合并通用 identity + agent / skill enforcement，避免 4 段
    // 重复 "you must follow ..." 充斥 prompt cache。
    let agent_extra = active_agent
        .map(|agent| agent.build_system_prompt())
        .filter(|s| !s.trim().is_empty());
    let skill_extra = skill
        .map(|skill| skill.build_system_prompt())
        .filter(|s| !s.trim().is_empty());

    let identity = if let Some(skill_text) = &skill_extra {
        let skill_name = skill.map(|skill| skill.name.as_str()).unwrap_or("unknown");
        let mut s = format!(
            "Active skill: {skill_name}\n\
             You are operating under this skill for the current turn. Treat the active skill \
             instructions as the primary behavior contract for this turn."
        );
        s.push_str("\n\n[Skill instructions]\n");
        s.push_str(skill_text.trim());
        if let Some(agent_text) = &agent_extra {
            s.push_str("\n\n[Agent instructions]\n");
            s.push_str(agent_text.trim());
            s.push_str("\n\nEnforcement: skill instructions override agent instructions when they differ. Use agent instructions only for capabilities, workflow, and defaults not covered by the active skill.");
        } else {
            s.push_str("\n\nEnforcement: skill instructions override generic assistant guidelines when they differ.");
        }
        s
    } else {
        agent_extra.unwrap_or_else(|| {
            String::from(
                "You are a highly capable general-purpose AI assistant. Adapt your approach to the task at hand: use code/tooling when the request is technical, and plain reasoning or research when it is not. Aim to be sharp and to the point — answer what was asked, not more.",
            )
        })
    };
    b.push(ContextKind::Identity, identity);

    if let Some(resource_path) = skill
        .and_then(|skill| skill.resource_path.as_deref())
        .filter(|path| !path.trim().is_empty())
    {
        b.push_labeled(
            ContextKind::Fact,
            "Active Skill Resources",
            format!(
                "The active skill includes bundled resources at `{}`. When the skill instructions refer to bundled files, scripts, references, examples, or assets, inspect this directory with available file tools and use the relevant resources.",
                resource_path.trim()
            ),
        );
    }

    b.push(ContextKind::Behavior, "Response style:\n- Lead with answer or action; skip preamble, restatements, and meta-commentary.\n- Default to short, direct prose. Use lists/sections only when they materially improve clarity.\n- Be concise but not at the cost of correctness: verify facts with tools before concluding. When citing code, include file/line.\n- Do not narrate tool calls before/during execution — let their output speak. Brief status lines only at real milestones or when the plan changes.");
    b.push(ContextKind::Behavior, "Tool usage:\n- Only rely on tools available in this turn's tool schema.\n- Prefer tool-backed evidence over speculation: inspect the relevant sources, artifacts, or system state and use the available tools before concluding.\n- If the user asks to run, build, test, reproduce, inspect, or modify something, use the relevant tools available in this turn. If the needed capability is unavailable, say so clearly instead of pretending you executed it.\n- On failure: read the error, adjust approach, retry up to twice before escalating.\n- When modifying files or structured content, prefer minimal, localized changes over broad rewrites.");
    b.push(ContextKind::Behavior, "Correctness guardrails:\n- Do not hallucinate: never present guesses, imagined evidence, or unverified assumptions as established truth.\n- Before concluding about code behavior, root cause, API contracts, repository state, or command results, gather sufficient evidence from tool output, source code, tests, logs, or explicit user input.\n- If evidence is incomplete, conflicting, or unavailable, say exactly what is uncertain.\n- Ask a clarifying question or state the missing verification step instead of guessing.\n- Distinguish clearly between verified facts, working hypotheses, and open questions.");

    if has_tool(available_tools, "enable_tools") {
        // Capability catalog（如有）按 trigger→tool 给出事实性精确映射；
        // discovery 段是 actionable 政策。两者 ContextKind 不同，挂在一起
        // 重叠较小且各自承担不同职责（catalog 让模型知道有什么，policy
        // 告诉模型何时去发现/启用）。保持双挂以避免回归测试用例期望的
        // mcp 提示与 "No skill is active yet" 提示丢失。
        if let Some(catalog) = build_capability_catalog(available_tools) {
            b.push(ContextKind::Fact, catalog);
        }
        let mut discovery_lines = Vec::new();
        if skill.is_none() {
            if has_tool(available_tools, "discover_skills") {
                discovery_lines.push(
                    "Not all tools are loaded. Use `discover_skills` for specialized workflows, or `enable_tools(operation=list)` then `enable_tools(operation=enable, tools=[...])` for specific tools.".to_string(),
                );
                discovery_lines.push(
                    "If the user names a workflow, tool, product, or domain that likely maps to a local skill (for example an internal service, CLI, log system, or incident workflow), call `discover_skills` with the named keyword before inventing commands.".to_string(),
                );
                if has_tool(available_tools, "activate_skill") {
                    discovery_lines.push(
                        "After `discover_skills`, if one listed skill clearly matches the task, call `activate_skill(name=...)` to load its prompt and tools. Do not activate a skill that does not clearly match.".to_string(),
                    );
                }
                discovery_lines.push(
                    "No skill is active yet. Prefer `discover_skills` before a freeform response only when the task is specialized, tool-heavy, or likely covered by an installed skill.".to_string(),
                );
            } else {
                discovery_lines.push(
                    "Not all tools are loaded. If a capability is missing, use `enable_tools(operation=list)` then `enable_tools(operation=enable, tools=[...])` for only what you need.".to_string(),
                );
                discovery_lines.push(
                    "No skill is active yet. Prefer enabling only the specific tools you need when the task is specialized or tool-heavy.".to_string(),
                );
            }
        } else {
            discovery_lines.push(
                "Not all tools are loaded. If a capability is missing, use `enable_tools(operation=list)` then `enable_tools(operation=enable, tools=[...])` for only what you need.".to_string(),
            );
        }
        discovery_lines.push(
            "For external systems (Feishu/Lark, web, etc.), discover and enable matching `mcp_*` tools first.".to_string(),
        );
        push_tool_guidance_section(&mut b, ContextKind::Policy, "Tool discovery:", discovery_lines);
    }

    if has_tool(available_tools, "knowledge_save") {
        let mut lines = vec![
            "If the user asks to remember or states a durable preference/constraint, call `knowledge_save`.".to_string(),
            "When saving a durable principle, preference, safety rule, or coding rule, choose a guideline category such as `common_sense`, `coding_guideline`, `preference`, `user_preference`, or `safety_rules` so it participates in persistent recall.".to_string(),
            "Use `user_memory` / `project_info` / `architecture` / `decision_log` for factual knowledge.".to_string(),
        ];
        let retrieval_tools =
            available_tool_names_in_order(available_tools, &["knowledge_search", "knowledge_list"]);
        if !retrieval_tools.is_empty() {
            lines.push(format!(
                "When asked about remembered info, use {}.",
                format_tool_names(&retrieval_tools)
            ));
        }
        push_tool_guidance_section(&mut b, ContextKind::Policy, "Knowledge save:", lines);
    }

    if has_tool(available_tools, "plan")
        || has_tool(available_tools, "spawn_process")
        || has_tool(available_tools, "task_spawn")
        || has_tool(available_tools, "task_wait")
        || has_tool(available_tools, "wait_process")
        || has_tool(available_tools, "kill_process")
        || has_tool(available_tools, "reap_process")
        || has_tool(available_tools, "send_ipc_message")
        || has_tool(available_tools, "read_mailbox")
    {
        let mut lines = Vec::new();
        if has_tool(available_tools, "plan") {
            lines.push("Simple tasks: act directly. Complex ones: call `plan` first.".to_string());
        }
        if has_tool(available_tools, "spawn_process") {
            lines.push(
                "Use `spawn_process` only for fire-and-forget background work whose result you do NOT need back (long-running processes, two-way IPC collaboration). It returns a PID, not a result.".to_string(),
            );
        }
        if has_tool(available_tools, "task_spawn") && has_tool(available_tools, "task_wait") {
            lines.push(
                "If you need the delegated work's result returned to you, use `task_spawn` + `task_wait` instead — even for a single task.".to_string(),
            );
        }
        let process_tools =
            available_tool_names_in_order(available_tools, &["wait_process", "kill_process", "reap_process"]);
        if !process_tools.is_empty() {
            lines.push(format!(
                "Use {} to manage child processes.",
                format_tool_names(&process_tools)
            ));
        }
        let ipc_tools =
            available_tool_names_in_order(available_tools, &["send_ipc_message", "read_mailbox"]);
        if !ipc_tools.is_empty() {
            lines.push(format!(
                "Use {} for cross-process communication.",
                format_tool_names(&ipc_tools)
            ));
        }
        push_tool_guidance_section(
            &mut b,
            ContextKind::Behavior,
            "Planning & Sub-process Execution:",
            lines,
        );
    }

    if has_tool(available_tools, "tool_spawn")
        || has_tool(available_tools, "tool_wait")
        || has_tool(available_tools, "tool_status")
        || has_tool(available_tools, "tool_cancel")
    {
        let mut lines = Vec::new();
        if has_tool(available_tools, "tool_spawn") {
            lines.push("Use `tool_spawn` for parallel independent tool calls.".to_string());
        }
        let async_tool_controls =
            available_tool_names_in_order(available_tools, &["tool_wait", "tool_status", "tool_cancel"]);
        if !async_tool_controls.is_empty() {
            lines.push(format!(
                "Use {} to join, inspect, or drop async tool branches as needed.",
                format_tool_names(&async_tool_controls)
            ));
        }
        if has_tool(available_tools, "read_mailbox") {
            lines.push(
                "If wake-up messages identify finished tasks, act on them immediately instead of re-querying.".to_string(),
            );
        }
        push_tool_guidance_section(
            &mut b,
            ContextKind::Behavior,
            "Async Tool Orchestration:",
            lines,
        );
    }

    if has_tool(available_tools, "task_spawn")
        || has_tool(available_tools, "task_wait")
        || has_tool(available_tools, "task_status")
    {
        let mut lines = Vec::new();
        if has_tool(available_tools, "task_spawn") {
            lines.push("Use `task_spawn` to launch a subagent task and fan out parallel independent subtasks.".to_string());
        }
        if has_tool(available_tools, "task_wait") {
            lines.push("Use `task_wait` to collect results. Timeout is per-call — re-call or use `wait_policy=\"any\"` for early wake-up.".to_string());
        }
        if has_tool(available_tools, "task_status") {
            lines.push("Use `task_status` for a non-blocking peek. There is no `task_cancel`; let orphaned tasks finish.".to_string());
        }
        if has_tool(available_tools, "tool_spawn")
            || has_tool(available_tools, "tool_wait")
            || has_tool(available_tools, "tool_status")
        {
            lines.push("`task_*` and `tool_*` are distinct families: do not confuse their IDs.".to_string());
        }
        push_tool_guidance_section(
            &mut b,
            ContextKind::Behavior,
            "Async Subagent Orchestration (task_*):",
            lines,
        );
    }

    if has_tool(available_tools, "agent_team") {
        let mut lines = vec![
            "Use `agent_team(operation=\"start\")` for complex decisions that benefit from several roles or independent perspectives.".to_string(),
        ];
        if has_tool(available_tools, "task_wait") {
            lines.push(
                "Collect the returned task_ids with `task_wait(wait_policy=\"all\")`, then pass the complete transcript into `agent_team(operation=\"challenge\")` so agents can challenge assumptions.".to_string(),
            );
        }
        lines.push(
            "For final consensus, pass initial outputs plus challenges into `agent_team(operation=\"synthesize\")`.".to_string(),
        );
        lines.push(
            "Team communication is parent-mediated; do not expect peer agents to receive direct mailbox messages.".to_string(),
        );
        push_tool_guidance_section(
            &mut b,
            ContextKind::Behavior,
            "Agent Team Deliberation:",
            lines,
        );
    }

    if has_tool(available_tools, "knowledge_search")
        || has_tool(available_tools, "knowledge_semantic_search")
        || has_tool(available_tools, "knowledge_list")
    {
        let mut lines = Vec::new();
        let search_tools = available_tool_names_in_order(
            available_tools,
            &["knowledge_search", "knowledge_semantic_search"],
        );
        if !search_tools.is_empty() {
            lines.push(format!(
                "Before answering from memory, search with {}.",
                format_tool_names(&search_tools)
            ));
        }
        if has_tool(available_tools, "knowledge_list") {
            lines.push("Use `knowledge_list` when asked what is remembered.".to_string());
        }
        push_tool_guidance_section(&mut b, ContextKind::Policy, "Knowledge retrieval:", lines);
    }

    if has_tool(available_tools, "web_search") || has_tool(available_tools, "web_fetch") {
        let mut lines = Vec::new();
        if has_tool(available_tools, "web_search") {
            lines.push("For real-time or time-sensitive topics, use `web_search` first (not memory).".to_string());
        }
        if has_tool(available_tools, "web_fetch") {
            lines.push("Use `web_fetch` for detailed content from selected URLs.".to_string());
        }
        push_tool_guidance_section(&mut b, ContextKind::Policy, "Web search:", lines);
    }

    b
}

fn should_skip_recall_for_skill(skill: Option<&SkillManifest>) -> bool {
    skill.is_some_and(|skill| skill.skip_recall || is_executor_skill(skill))
}

fn skill_selection_threshold(skill_count: usize) -> f64 {
    let base: f64 = if skill_count > 10 {
        2.5
    } else if skill_count > 5 {
        2.0
    } else {
        1.75
    };
    // 无 intent 后统一下限取 2.0（原 RequestAction 档），避免小 skill 集阈值
    // 反而更低导致误激活。
    base.max(2.0)
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

fn cross_turn_preferred_skill_name(app: &App, question: &str) -> Option<String> {
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
    preferred_skill_name: Option<&str>,
    model_path: &std::path::Path,
) -> Option<&'a SkillManifest> {
    select_skill_with_preference_strength(
        skill_manifests,
        question,
        preferred_skill_name,
        PreferenceStrength::StrongSticky,
        model_path,
    )
}

fn select_skill_with_preference_strength<'a>(
    skill_manifests: &'a [SkillManifest],
    question: &str,
    preferred_skill_name: Option<&str>,
    strength: PreferenceStrength,
    model_path: &std::path::Path,
) -> Option<&'a SkillManifest> {
    if question.trim().is_empty() || skill_manifests.is_empty() {
        return None;
    }

    let ranked = rank_skills_locally_with_model_path(skill_manifests, question, model_path);
    let Some(best) = ranked.first() else {
        return None;
    };
    let threshold = skill_selection_threshold(skill_manifests.len());
    let preferred =
        preferred_skill_name.and_then(|name| ranked.iter().find(|item| item.skill.name == name));
    let best_is_valid = !should_abstain_from_skill(best);

    if let Some(current) = preferred {
        let current_is_valid = !should_abstain_from_skill(current);
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
        let current_has_signal = current_is_valid
            && (current.score >= (threshold - keep_floor_delta).max(0.0)
                || has_skill_signal(current));
        if best.skill.name == current.skill.name {
            return if current_is_valid
                && (keep_current_when_best || current.score >= threshold || current_has_signal)
            {
                Some(current.skill)
            } else {
                None
            };
        }

        let effective_current = current.score + sticky_bonus;
        let best_clearly_wins = best.score >= effective_current + switch_margin;
        let best_has_positive_signal = best_is_valid && is_positive_skill_winner(best);
        let allow_switch = match strength {
            PreferenceStrength::StrongSticky => best_clearly_wins && best_has_positive_signal,
            PreferenceStrength::CrossTurnBias => best_clearly_wins,
        };
        if allow_switch {
            return Some(best.skill);
        }
        if keep_below_threshold && best.score < threshold && current_is_valid {
            return Some(current.skill);
        }
        if current_has_signal && !best_clearly_wins {
            return Some(current.skill);
        }
        if best_is_valid && best.score >= threshold {
            return Some(best.skill);
        }
        if current_has_signal {
            return Some(current.skill);
        }
        return None;
    }

    if !best_is_valid {
        return None;
    }
    (best.score >= threshold).then_some(best.skill)
}

fn has_skill_signal(item: &super::skill_ranking::ScoredSkill<'_>) -> bool {
    item.embedding_score >= 0.08
        || item.fallback_semantic_score >= 0.08
        || item.blended_score >= 0.08
        || non_identity_skill_signal(item) >= 0.08
}

fn is_positive_skill_winner(item: &super::skill_ranking::ScoredSkill<'_>) -> bool {
    item.embedding_score >= 0.30
        || item.fallback_semantic_score >= 0.15
        || item.blended_score >= 0.30
        || non_identity_skill_signal(item) >= 0.15
}

fn identity_skill_signal(item: &super::skill_ranking::ScoredSkill<'_>) -> f64 {
    item.embedding_identity_score
        .max(item.fallback_identity_score)
}

fn non_identity_skill_signal(item: &super::skill_ranking::ScoredSkill<'_>) -> f64 {
    item.embedding_capability_score
        .max(item.embedding_behavior_score)
        .max(item.fallback_capability_score)
        .max(item.fallback_behavior_score)
        .max(item.model_prior_score)
}

fn should_abstain_from_skill(item: &super::skill_ranking::ScoredSkill<'_>) -> bool {
    let identity_signal = identity_skill_signal(item);
    let non_identity_signal = non_identity_skill_signal(item);
    let likely_identity_only_match = identity_signal >= 0.12
        && non_identity_signal < 0.12
        && non_identity_signal + 0.04 < identity_signal;

    likely_identity_only_match
        && item.none_score >= item.model_prior_score + 0.08
        && non_identity_signal < 0.10
}

fn build_skill_turn_guard(
    app: &mut App,
    mcp_client: &McpClient,
    skill: Option<&SkillManifest>,
) -> SkillTurnGuard {
    let all_mcp_tools = mcp_client.get_all_tools();
    super::super::tools::enable_tools::set_available_mcp_tools(all_mcp_tools.clone());
    let matched_skill_name = skill.as_ref().map(|s| s.name.clone());
    let skip_recall_by_skill = should_skip_recall_for_skill(skill);
    let active_agent = app.current_agent_manifest.clone();
    let executor_active = skill.as_ref().is_some_and(|s| is_executor_skill(s))
        || active_agent.as_ref().is_some_and(is_executor_agent);

    let builtin_tools = builtin_tools_for_skill(skill, active_agent.as_ref());
    let mcp_tools = select_mcp_tools(all_mcp_tools.clone(), skill, active_agent.as_ref());
    let available_tools = available_tool_names(&builtin_tools, &mcp_tools);
    let mut builder = build_system_prompt(active_agent.as_ref(), skill, &available_tools);
    if has_tool(&available_tools, "enable_tools")
        && let Some(catalog) = build_hidden_mcp_tool_catalog(&all_mcp_tools, &mcp_tools)
    {
        builder.push(ContextKind::Fact, catalog);
    }
    push_project_context(&mut builder);
    if !app.active_persona.is_default() {
        let mut persona_prompt = format!(
            "Persistent persona:\n- Name: {}\n",
            app.active_persona.name.trim()
        );
        if !app.active_persona.avatar.trim().is_empty() {
            persona_prompt.push_str(&format!("- Avatar: {}\n", app.active_persona.avatar.trim()));
        }
        if !app.active_persona.prompt.trim().is_empty() {
            persona_prompt.push_str("\nPersona instructions:\n");
            persona_prompt.push_str(app.active_persona.prompt.trim());
        }
        persona_prompt.push_str(
            "\n\nApply this persona consistently across turns, but never let it override higher-priority agent, skill, policy, or user instructions.",
        );
        builder.push(ContextKind::Identity, persona_prompt);
    }
    let max_iterations = resolve_max_iterations(active_agent.as_ref(), executor_active);
    let restore_agent_context =
        activate_skill_context(app, builtin_tools, mcp_tools, max_iterations);

    SkillTurnGuard {
        restore_agent_context,
        builder,
        cached_system_prompt: None,
        cached_context_reminder: None,
        matched_skill_name,
        skip_recall_by_skill,
    }
}

pub(super) fn rebuild_skill_turn_with_existing_selection(
    app: &mut App,
    mcp_client: &McpClient,
    skill_manifests: &[SkillManifest],
    question: &str,
    preferred_skill_name: Option<&str>,
) -> SkillTurnGuard {
    let skill = select_skill_with_preference(
        skill_manifests,
        question,
        preferred_skill_name,
        &app.config.skill_match_model_path,
    );
    build_skill_turn_guard(app, mcp_client, skill)
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
) -> Option<SkillTurnGuard> {
    let skill = skill_manifests.iter().find(|s| s.name == requested_name)?;
    update_cross_turn_skill_bias(app, question, Some(skill));
    let mut guard = build_skill_turn_guard(app, mcp_client, Some(skill));
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
                force_activate_named_skill(app, mcp_client, skill_manifests, question, &name)
            {
                if debug {
                    eprintln!("[skills] forced via @skills: {}", name);
                }
                return guard;
            }
        } else if debug {
            eprintln!(
                "[skills] forced @skills:{} not found, falling back to auto-route",
                forced
            );
        }
    }

    let cross_turn_preference = cross_turn_preferred_skill_name(app, question);
    let skill = select_skill_with_preference_strength(
        skill_manifests,
        question,
        cross_turn_preference.as_deref(),
        PreferenceStrength::CrossTurnBias,
        &app.config.skill_match_model_path,
    );

    if debug {
        let ranked = rank_skills_locally_with_model_path(
            skill_manifests,
            question,
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
        if let Some(s) = skill.as_ref() {
            eprintln!("[skills] final: {}", s.name);
        } else {
            eprintln!("[skills] final: <none>");
        }
    }
    update_cross_turn_skill_bias(app, question, skill);
    let matched_skill_name = skill.as_ref().map(|s| s.name.clone());
    let skip_recall_by_skill = should_skip_recall_for_skill(skill);
    let mut guard = build_skill_turn_guard(app, mcp_client, skill);
    guard.matched_skill_name = matched_skill_name;
    guard.skip_recall_by_skill = skip_recall_by_skill;
    guard
}

#[cfg(test)]
mod tests {
    use super::{
        ContextKind, PreferenceStrength, SystemPromptBuilder, available_tool_names,
        build_hidden_mcp_tool_catalog, build_project_instruction_prompt, build_system_prompt,
        builtin_tools_for_skill, ensure_required_baseline_tools,
        filter_mcp_tools_by_allowed_servers, has_tool, looks_like_follow_up_or_same_topic,
        merge_with_runtime_enabled_tools, push_project_context, resolve_max_iterations,
        select_mcp_tools, select_skill_with_preference, select_skill_with_preference_strength,
        tool_uses_mcp_server,
    };
    use crate::ai::agents::{AgentManifest, AgentMode};
    use crate::ai::driver::runtime_ctx::SUBAGENT_CWD;
    use crate::ai::driver::skill_ranking::ScoredSkill;
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
            disable_mcp_tools: false,
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
        assert!(!prompt.contains("cargo_test"));
        assert!(!prompt.contains("execute_command"));
        assert!(!prompt.contains("apply_patch"));
    }

    #[test]
    fn system_prompt_enforces_concise_response_style_with_correctness_safeguard() {
        let available = SkipSet::new(16);
        let prompt = build_system_prompt(None, None, &Box::new(available)).render_system_prompt();
        // 风格段必须存在，且要求"先答后说、不啰嗦"
        assert!(prompt.contains("Response style:"));
        assert!(prompt.contains("Lead with answer"));
        // 必须保留"简洁不能换错"的安全垫，避免过度精简导致错误判断
        assert!(prompt.contains("concise but not at the cost of correctness"));
    }

    #[test]
    fn system_prompt_forbids_guessing_without_sufficient_evidence() {
        let available = SkipSet::new(16);
        let prompt = build_system_prompt(None, None, &Box::new(available)).render_system_prompt();
        assert!(prompt.contains("Correctness guardrails:"));
        assert!(prompt.contains("Do not hallucinate"));
        assert!(prompt.contains("gather sufficient evidence"));
        assert!(prompt.contains("instead of guessing"));
        assert!(prompt.contains("verified facts, working hypotheses, and open questions"));
    }

    #[test]
    fn generic_system_prompt_does_not_hardcode_repo_specific_tool_names() {
        let available = SkipSet::new(16);
        let prompt = build_system_prompt(None, None, &Box::new(available)).render_system_prompt();
        assert!(!prompt.contains("cargo_test"));
        assert!(!prompt.contains("execute_command / cargo_test"));
        assert!(!prompt.contains("execute_command"));
        assert!(!prompt.contains("apply_patch"));
        assert!(prompt.contains("relevant tools available in this turn"));
        assert!(prompt.contains("instead of pretending you executed it"));
    }

    #[test]
    fn system_prompt_mentions_mcp_discovery_when_enable_tools_available() {
        let mut available = SkipSet::new(16);
        available.insert("enable_tools".to_string());
        let prompt = build_system_prompt(None, None, &Box::new(available)).render_system_prompt();
        assert!(prompt.contains("discover and enable matching `mcp_*` tools first"));
    }

    #[test]
    fn hidden_mcp_tool_catalog_lists_real_available_tools() {
        let catalog = build_hidden_mcp_tool_catalog(
            &[
                tool("mcp_feishu_docs_search"),
                tool("mcp_feishu_docs_get_text_by_url"),
                tool("mcp_pdf-extract_pdf_extract_text"),
            ],
            &[tool("mcp_feishu_docs_search")],
        )
        .unwrap();

        assert!(catalog.contains("Configured MCP tools are available but not loaded"));
        assert!(catalog.contains("enable_tools(operation=list)"));
        assert!(catalog.contains("`mcp_feishu_docs_get_text_by_url`"));
        assert!(catalog.contains("`mcp_pdf-extract_pdf_extract_text`"));
        assert!(!catalog.contains("`mcp_feishu_docs_search`"));
    }

    #[test]
    fn hidden_mcp_tool_catalog_omits_prompt_when_everything_is_loaded() {
        let catalog = build_hidden_mcp_tool_catalog(
            &[tool("mcp_feishu_docs_search")],
            &[tool("mcp_feishu_docs_search")],
        );
        assert!(catalog.is_none());
    }

    #[test]
    fn narrow_skill_whitelist_still_lets_model_discover_explicitly_requested_mcp_tools() {
        // 场景复现：用户显式要求"用 mcp 工具写飞书"，但当前 skill 用窄 tools:
        // 白名单把工具集替换成只有一个专用工具，且默认 agent 带 disable_mcp_tools
        // （mcp_* 不预挂载）。修复前：窄白名单会把 enable_tools 一并挤掉，hidden MCP
        // catalog 的注入门（has_tool("enable_tools")）随之关闭，模型三条发现 MCP 的
        // 路径全断，物理上无法响应"用 mcp 工具"。修复后：enable_tools 作为 baseline
        // 常驻补回，catalog 注入门重新成立，模型能发现并启用 mcp_feishu_*。
        let mut narrow_skill = skill("feishu-upload", "Upload markdown into Feishu docs");
        narrow_skill.tools = vec!["write_file".to_string()];

        // 1) 窄白名单替换工具集后，baseline 兜底仍补回发现/加载与基础只读入口。
        let builtin_tools = builtin_tools_for_skill(Some(&narrow_skill), None);
        let builtin_names = builtin_tools
            .iter()
            .map(|tool| tool.function.name.clone())
            .collect::<Vec<_>>();
        assert!(
            builtin_names.contains(&"write_file".to_string()),
            "skill 白名单里显式声明的工具应保留"
        );
        assert!(
            builtin_names.contains(&"enable_tools".to_string()),
            "enable_tools 必须作为 baseline 常驻补回，否则模型无法发现/启用 MCP 工具"
        );
        assert!(
            builtin_names.contains(&"read_file".to_string()),
            "read_file 应作为基础只读能力常驻，读取用户点名的 test.md"
        );

        // 2) 默认 agent disable_mcp_tools => 本轮 mcp_* 一个都没预挂载。
        let all_mcp_tools = vec![
            tool("mcp_feishu_doc_create_from_markdown"),
            tool("mcp_feishu_docs_get_text_by_url"),
            tool("mcp_pdf-extract_pdf_extract_text"),
        ];
        let loaded_mcp_tools: Vec<ToolDefinition> = Vec::new();

        // 3) available_tools 含 enable_tools => catalog 注入门成立（生产代码里的
        //    has_tool("enable_tools") 判断）。
        let available_tools = available_tool_names(&builtin_tools, &loaded_mcp_tools);
        assert!(
            has_tool(&available_tools, "enable_tools"),
            "catalog 注入门依赖 available_tools 里存在 enable_tools"
        );

        // 4) hidden MCP catalog 会把用户想用的 mcp_feishu_* 暴露给模型作为发现入口。
        let catalog = build_hidden_mcp_tool_catalog(&all_mcp_tools, &loaded_mcp_tools)
            .expect("存在未加载的 mcp_* 时必须给出发现提示");
        assert!(catalog.contains("enable_tools(operation=list)"));
        assert!(catalog.contains("`mcp_feishu_doc_create_from_markdown`"));
        assert!(catalog.contains("`mcp_feishu_docs_get_text_by_url`"));
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
        assert!(prompt.contains("only when the task is specialized, tool-heavy"));
        assert!(prompt.contains("call `discover_skills` with the named keyword"));
    }

    #[test]
    fn system_prompt_omits_discover_skills_guidance_when_tool_is_unavailable() {
        let mut available = SkipSet::new(16);
        available.insert("enable_tools".to_string());
        let prompt = build_system_prompt(None, None, &Box::new(available)).render_system_prompt();
        assert!(prompt.contains("enable_tools(operation=list)"));
        assert!(!prompt.contains("call `discover_skills` with the named keyword"));
        assert!(!prompt.contains("After `discover_skills`"));
    }

    #[test]
    fn render_groups_same_kind_sections_into_single_tag_block() {
        // identity/behavior/policy 各应只出现一对 tag，且按 identity→behavior→policy
        // 排布。这保证 persona 这类"事后追加"的 identity 段不会被甩到 prompt 末尾，
        // 而是与通用 identity 聚拢成一块。
        let mut builder = SystemPromptBuilder::new();
        builder.push(ContextKind::Identity, "Generic identity.");
        builder.push(ContextKind::Behavior, "Behavior one.");
        builder.push(ContextKind::Policy, "Policy one.");
        builder.push(ContextKind::Behavior, "Behavior two.");
        // 模拟 persona 在 build_system_prompt 之后追加：
        builder.push(ContextKind::Identity, "Persona identity.");

        let prompt = builder.render_system_prompt();

        assert_eq!(prompt.matches("<identity>").count(), 1);
        assert_eq!(prompt.matches("<behavior>").count(), 1);
        assert_eq!(prompt.matches("<policy>").count(), 1);

        let identity_pos = prompt.find("<identity>").unwrap();
        let behavior_pos = prompt.find("<behavior>").unwrap();
        let policy_pos = prompt.find("<policy>").unwrap();
        assert!(identity_pos < behavior_pos && behavior_pos < policy_pos);

        // 两段 identity 必须落在同一个 identity 块内、且保持插入顺序。
        let generic_pos = prompt.find("Generic identity.").unwrap();
        let persona_pos = prompt.find("Persona identity.").unwrap();
        let identity_close = prompt.find("</identity>").unwrap();
        assert!(generic_pos < persona_pos);
        assert!(persona_pos < identity_close);

        // 同 kind 段落之间用空行分隔，组内保持插入顺序。
        let behavior_one = prompt.find("Behavior one.").unwrap();
        let behavior_two = prompt.find("Behavior two.").unwrap();
        assert!(behavior_one < behavior_two);
    }

    #[test]
    fn render_excludes_fact_sections_from_system_prompt() {
        let mut builder = SystemPromptBuilder::new();
        builder.push(ContextKind::Identity, "Identity.");
        builder.push_labeled(ContextKind::Fact, "Project Type", "Rust project.");

        let prompt = builder.render_system_prompt();
        assert!(prompt.contains("Identity."));
        assert!(!prompt.contains("Rust project."));
        assert!(!prompt.contains("Project Type"));
    }

    #[test]
    fn active_skill_prompt_precedes_agent_prompt_and_declares_priority() {
        let available = SkipSet::new(16);
        let mut build_agent = agent("build", vec![]);
        build_agent.prompt = "You are the build agent.".to_string();
        let mut humanizer = skill("humanizer", "Rewrite text naturally");
        humanizer.prompt = "You are a writing editor.".to_string();

        let prompt =
            build_system_prompt(Some(&build_agent), Some(&humanizer), &Box::new(available))
                .render_system_prompt();

        let skill_pos = prompt.find("Active skill: humanizer").unwrap();
        let agent_pos = prompt.find("[Agent instructions]").unwrap();
        assert!(skill_pos < agent_pos);
        assert!(prompt.contains("primary behavior contract"));
        assert!(prompt.contains("skill instructions override agent instructions"));
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
        assert!(prompt.contains("`common_sense`, `coding_guideline`"));
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

    #[test]
    fn project_context_is_appended_separately_from_base_prompt() {
        let root = temp_dir("project_context");
        let nested = root.join("apps/web/src");
        fs::create_dir_all(&nested).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname='demo'\nversion='0.1.0'\n",
        )
        .unwrap();
        fs::write(root.join("AGENTS.md"), "Use cargo fmt before commit.\n").unwrap();

        let (base_prompt, enriched_prompt, reminder) =
            SUBAGENT_CWD.sync_scope(nested.clone(), || {
                let available = SkipSet::new(16);
                let mut builder = build_system_prompt(None, None, &Box::new(available));
                let base_prompt = builder.render_system_prompt();
                push_project_context(&mut builder);
                let enriched_prompt = builder.render_system_prompt();
                let reminder = builder.render_context_reminder().unwrap_or_default();
                (base_prompt, enriched_prompt, reminder)
            });

        assert!(!base_prompt.contains("Use cargo fmt before commit."));
        assert!(enriched_prompt.contains("Use cargo fmt before commit."));
        assert!(reminder.contains("Project Type"));
        assert!(reminder.contains("Rust project"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn project_instructions_remain_available() {
        let root = temp_dir("project_context_general_mode");
        let nested = root.join("apps/web/src");
        fs::create_dir_all(&nested).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname='demo'\nversion='0.1.0'\n",
        )
        .unwrap();
        fs::write(root.join("AGENTS.md"), "Always follow repo safety rules.\n").unwrap();

        let prompt = SUBAGENT_CWD.sync_scope(nested.clone(), || {
            let available = SkipSet::new(16);
            let mut builder = build_system_prompt(None, None, &Box::new(available));
            push_project_context(&mut builder);
            builder.render_system_prompt()
        });

        assert!(prompt.contains("Project-local instructions:"));
        assert!(prompt.contains("Always follow repo safety rules."));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn project_action_intent_keeps_project_context_available() {
        let root = temp_dir("project_context_work_signal");
        let nested = root.join("apps/web/src");
        fs::create_dir_all(&nested).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname='demo'\nversion='0.1.0'\n",
        )
        .unwrap();
        fs::write(root.join("AGENTS.md"), "Always follow repo safety rules.\n").unwrap();

        let prompt = SUBAGENT_CWD.sync_scope(nested.clone(), || {
            let available = SkipSet::new(16);
            let mut builder = build_system_prompt(None, None, &Box::new(available));
            push_project_context(&mut builder);
            builder.render_system_prompt()
        });

        assert!(prompt.contains("Project-local instructions:"));
        assert!(prompt.contains("Always follow repo safety rules."));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn prompt_introspection_query_still_keeps_project_instructions() {
        let root = temp_dir("project_context_prompt_query");
        let nested = root.join("apps/web/src");
        fs::create_dir_all(&nested).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname='demo'\nversion='0.1.0'\n",
        )
        .unwrap();
        fs::write(root.join("AGENTS.md"), "Always follow repo safety rules.\n").unwrap();

        let prompt = SUBAGENT_CWD.sync_scope(nested.clone(), || {
            let available = SkipSet::new(16);
            let mut builder = build_system_prompt(None, None, &Box::new(available));
            push_project_context(&mut builder);
            builder.render_system_prompt()
        });

        assert!(prompt.contains("Project-local instructions:"));
        assert!(prompt.contains("Always follow repo safety rules."));

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
            resource_path: None,
        }
    }

    fn skill_with_prompt(name: &str, description: &str, prompt: &str) -> SkillManifest {
        let mut skill = skill(name, description);
        skill.prompt = prompt.to_string();
        skill
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
            disable_mcp_tools: false,
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
    fn agent_disable_mcp_tools_hides_default_mcp_tools() {
        let all_tools = vec![tool("mcp_feishu_docs_search"), tool("mcp_ocr_extract")];
        let mut build_agent = agent("build", vec![]);
        build_agent.disable_mcp_tools = true;

        let tools = select_mcp_tools(all_tools, None, Some(&build_agent));

        assert!(tools.is_empty());
    }

    #[test]
    fn skill_mcp_servers_can_opt_in_when_agent_disables_mcp_tools() {
        let all_tools = vec![tool("mcp_feishu_docs_search"), tool("mcp_ocr_extract")];
        let mut build_agent = agent("build", vec![]);
        build_agent.disable_mcp_tools = true;
        let mut s = skill("feishu-docs", "");
        s.mcp_servers = vec!["feishu".to_string()];

        let tools = select_mcp_tools(all_tools, Some(&s), Some(&build_agent));
        let names = tools
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["mcp_feishu_docs_search".to_string()]);
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
        // 基础只读 / 检索能力应作为 baseline 常驻补回，避免窄白名单 skill 把
        // read_file 等最基本的阅读工具剔除，导致主 Agent 连用户点名的文件都读不了。
        assert!(names.contains(&"read_file".to_string()));
        assert!(names.contains(&"read_file_lines".to_string()));
        assert!(names.contains(&"list_directory".to_string()));
        assert!(names.contains(&"find_path".to_string()));
        assert!(names.contains(&"text_grep".to_string()));
        // 子 Agent 编排能力应作为 baseline 常驻补回，避免 skill 白名单把 task_*
        // 全部剔除导致主 Agent 失去委派子 Agent 的能力。
        assert!(names.contains(&"task".to_string()));
        assert!(names.contains(&"task_spawn".to_string()));
        assert!(names.contains(&"task_wait".to_string()));
        assert!(names.contains(&"task_status".to_string()));
        assert!(names.contains(&"agent_team".to_string()));
        // 其它非 baseline 的内置工具仍不应被无端带入白名单。
        assert!(!names.contains(&"plan".to_string()));
        assert!(!names.contains(&"write_file".to_string()));
        assert!(!names.contains(&"apply_patch".to_string()));
    }

    fn model_path() -> std::path::PathBuf {
        crate::ai::driver::skill_match_model::default_model_path()
    }

    #[test]
    fn local_selector_chooses_review_skill_for_review_request() {
        let skills = vec![
            skill("code-review", "Review code changes and highlight bugs"),
            skill("debugger", "Debug runtime failures and collect traces"),
        ];
        let selected = select_skill_with_preference(
            &skills,
            "帮我 review 这段 Rust 代码",
            None,
            &model_path(),
        );
        assert_eq!(selected.map(|item| item.name.as_str()), Some("code-review"));
    }

    #[test]
    fn local_selector_prefers_current_skill_without_clear_winner() {
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
            Some("code-review"),
            &model_path(),
        );
        assert_eq!(selected.map(|item| item.name.as_str()), Some("code-review"));
    }

    #[test]
    fn local_selector_switches_when_new_skill_is_significantly_better() {
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
            Some("code-review"),
            &model_path(),
        );
        assert_eq!(selected.map(|item| item.name.as_str()), Some("debugger"));
    }

    #[test]
    fn local_selector_abstains_from_identity_only_match() {
        // 纯 identity 命中（名字/描述里恰好出现 skill 名）但缺乏 capability/behavior
        // 信号时应弃权：intent 移除后由 identity-only 打分门控兜底防误激活。
        let item = ScoredSkill {
            skill: &skill("skill-creator", "Create new skills"),
            score: 0.0,
            embedding_score: 0.0,
            embedding_identity_score: 0.30,
            embedding_capability_score: 0.02,
            embedding_behavior_score: 0.02,
            fallback_semantic_score: 0.0,
            fallback_identity_score: 0.30,
            fallback_capability_score: 0.02,
            fallback_behavior_score: 0.02,
            model_prior_score: 0.05,
            blended_score: 0.0,
            none_score: 0.20,
        };
        assert!(super::should_abstain_from_skill(&item));
    }

    #[test]
    fn local_selector_still_routes_skill_creator_like_skill_for_creation_request() {
        let skills = vec![
            skill_with_prompt(
                "skill-creator",
                "Create new skills and skill templates for the workspace",
                "Use this skill when the user wants to create or add a new skill to the workspace.",
            ),
            skill("debugger", "Debug runtime failures and collect traces"),
        ];
        let selected = select_skill_with_preference(
            &skills,
            "帮我创建一个新的 CI skill 模板",
            None,
            &model_path(),
        );
        assert_eq!(
            selected.map(|item| item.name.as_str()),
            Some("skill-creator")
        );
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
        let skills = vec![
            skill("code-review", "Review code changes and summarize defects"),
            skill("debugger", "Debug runtime failures and panic stack traces"),
        ];
        let selected = select_skill_with_preference_strength(
            &skills,
            "那这个报错顺便再看一下",
            Some("debugger"),
            PreferenceStrength::CrossTurnBias,
            &model_path(),
        );
        assert_eq!(selected.map(|item| item.name.as_str()), Some("debugger"));
    }

    #[test]
    fn cross_turn_bias_still_switches_when_new_skill_is_clearly_better() {
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
            Some("debugger"),
            PreferenceStrength::CrossTurnBias,
            &model_path(),
        );
        assert_eq!(selected.map(|item| item.name.as_str()), Some("code-review"));
    }
}
