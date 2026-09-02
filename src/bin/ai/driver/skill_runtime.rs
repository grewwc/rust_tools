use crate::ai::{
    agents::{
        AgentManifest, load_project_instruction_docs,
        load_scoped_project_instruction_docs_for_target_priority,
        load_scoped_project_instruction_docs_for_targets,
    },
    history::{self, SkillActivationEvent},
    mcp::McpClient,
    skills::SkillManifest,
    tools::ToolGroup,
    types::{App, ForcedSkillSource, ToolDefinition},
};
use crate::commonw::configw;
use rust_tools::cw::SkipSet;
use std::path::{Path, PathBuf};

use super::{DEFAULT_MAX_ITERATIONS, EXECUTOR_MAX_ITERATIONS};

type ToolDef = ToolDefinition;

/// Runtime context passed into build_system_prompt to enable conditional rendering.
/// Currently has goal_mode and is_background; extensible later with task_type / persona etc.
#[derive(Clone, Default)]
pub(super) struct PromptContext {
    /// Some(_) means goal mode is active; the value is the goal description text.
    pub goal_mode: Option<String>,
    /// Background mode (-bg): the terminal is detached, so do not inject "ask the user" guidance.
    pub is_background: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ContextKind {
    Identity,
    Behavior,
    Capability,
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
            let label = label.trim();
            self.sections.push((
                kind,
                (!label.is_empty()).then(|| label.to_string()),
                content,
            ));
        }
    }

    fn render_system_prompt(&self) -> String {
        // Render grouped by semantic category: all sections of the same kind
        // (identity/behavior/capability/policy) merge into one tag pair, keeping insertion order within the group, so identity sections
        // appended "after build_system_prompt" (e.g. persona) are not pushed to the end of the prompt but cluster with the generic identity;
        // behavior/policy no longer split into multiple clusters by push timing, reducing tag noise and making the priority
        // hierarchy clearer to the model. Fact sections are not rendered in the system prompt (they go through the context reminder injected into the current
        // user message), so they are not in the whitelist and are naturally excluded.
        const RENDER_ORDER: [(ContextKind, &str); 4] = [
            (ContextKind::Identity, "identity"),
            (ContextKind::Behavior, "behavior"),
            (ContextKind::Capability, "capabilities"),
            (ContextKind::Policy, "policy"),
        ];
        let mut out = String::new();
        for (group_kind, tag) in RENDER_ORDER {
            let mut group = String::new();
            for (kind, label, content) in &self.sections {
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
                if let Some(label) = label {
                    let tag = label.trim();
                    group.push_str(&format!("<{tag}>\n"));
                }
                group.push_str(trimmed);
                if let Some(label) = label {
                    let tag = label.trim();
                    group.push_str(&format!("\n</{tag}>"));
                }
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
                out.push_str(&format!("## {}\n{}\n\n", key, content.trim()));
            } else {
                out.push_str(&format!("{}\n\n", content.trim()));
            }
        }
        out.push_str("IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.\n</system-reminder>");
        Some(out)
    }
}

const DEFAULT_TURN_TOOL_GROUPS: &[ToolGroup] = &[ToolGroup::Core];

fn runtime_environment_prompt() -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let os_label = match os {
        "macos" => "macOS",
        "linux" => "Linux",
        "windows" => "Windows",
        other => other,
    };
    let shell = std::env::var("SHELL")
        .ok()
        .and_then(|shell| {
            let shell_name = Path::new(&shell)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(shell.as_str())
                .trim()
                .to_string();
            (!shell_name.is_empty()).then_some(shell_name)
        })
        .unwrap_or_else(|| "unknown".to_string());
    let effective_cwd = super::runtime_ctx::effective_cwd()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|error| format!("<unavailable: {error}>"));

    format!(
        include_str!("system_prompts/runtime_environment.md"),
        os_label = os_label,
        os = os,
        arch = arch,
        shell = shell,
        effective_cwd = effective_cwd,
    )
}

pub(super) struct SkillTurnGuard {
    restore_agent_context: Option<(Vec<ToolDef>, usize)>,
    builder: SystemPromptBuilder,
    cached_system_prompt: Option<String>,
    cached_context_reminder: Option<Option<String>>,
    /// Currently active skill list (ordered; multiple skills are peers with no ranking).
    matched_skill_names: Vec<String>,
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
            let mut parts: Vec<String> = Vec::new();
            // "End-of-context pointer" for active skills: in long contexts (multi-turn dialogue + tool loops) the system
            // prompt sits at the very front, where skill instructions get diluted by the long middle; injecting a short pointer at the
            // start of the last user message re-anchors "the skill list in effect at turn start (snapshot)"
            // near the request (recency position), so the model still follows the skill contract
            // in long contexts. The pointer deliberately says "at turn start" rather than "for this turn": the user message is
            // built only once at turn start, while mid-turn activate_skill/deactivate_skill
            // can change the effective set, so the snapshot wording keeps the pointer never false; the currently effective set is always governed by the
            // system prompt's `<skill_instructions>` (rebuilt each iteration, authoritative, cache-friendly)
            // This pointer only enters the request projection (turn_messages does not contain the reminder), and the current user
            // message is already a cache miss anyway, so the upstream prompt cache is not broken. Injected only when
            // there are active skills.
            if !self.matched_skill_names.is_empty() {
                let mut pointer = String::new();
                if self.matched_skill_names.len() == 1 {
                    pointer.push_str(&format!(
                        "Active skill at turn start: {}.\n",
                        self.matched_skill_names[0]
                    ));
                    pointer.push_str(
                        "Treat its instructions in the system prompt's <skill_instructions> \
                         section as the primary behavior contract for this turn.",
                    );
                } else {
                    pointer.push_str(
                        "Active skills at turn start (in activation order):\n",
                    );
                    for (i, name) in self.matched_skill_names.iter().enumerate() {
                        use std::fmt::Write;
                        let _ = writeln!(pointer, "  {}. {}", i + 1, name);
                    }
                    pointer.push_str(
                        "Treat their instructions in the system prompt's <skill_instructions> \
                         section as the primary behavior contract for this turn; all active \
                         skills are equal peers that compose additively, and guardrails \
                         always take precedence.",
                    );
                }
                parts.push(format!("<system-reminder>\n{}\n</system-reminder>", pointer));
            }
            if let Some(rest) = self.builder.render_context_reminder() {
                parts.push(rest);
            }
            self.cached_context_reminder = Some(if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n\n"))
            });
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

    pub(super) fn push_scoped_project_instructions(
        &mut self,
        required_targets: &[PathBuf],
        observed_targets: &[PathBuf],
    ) -> bool {
        if let Some(prompt) = build_scoped_project_instruction_prompt_with_priority(
            required_targets,
            observed_targets,
        ) {
            self.push_labeled_section(
                ContextKind::Policy,
                "target_scoped_project_instructions",
                &prompt,
            );
        }
        !scoped_project_instructions_missing(self.system_prompt(), required_targets)
    }

    /// Returns the names of all currently active skills (ordered; peers with no ranking).
    pub(super) fn matched_skill_names(&self) -> &[String] {
        &self.matched_skill_names
    }

    /// Returns the first skill name in the active list, for display/logging only (peers, no ranking).
    pub(super) fn primary_skill_name(&self) -> Option<&str> {
        self.matched_skill_names.first().map(|s| s.as_str())
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
    // max_iterations is the per-turn iteration cap (TurnSupervisor.iteration resets to 0 each turn),
    // while the kernel's max_tool_calls is cumulative over the process lifetime (tool_calls_used is never reset).
    // Mapping the per-turn max_iterations onto the cumulative max_tool_calls would make long sessions hit the
    // cumulative tool-call ceiling first (build=2048 / executor=128), force-wrapping the turn
    // even without any single-turn overrun, showing up as "tool limit reached for this turn". The per-turn iteration cap is already enforced by
    // execution.rs's `iteration >= max_iterations` check, and the process-level turn cap is enforced by max_turns
    // (from quota_turns), so we no longer override max_tool_calls here and keep it unlimited.
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
        return filter_subagent_hidden_tools(merged);
    }
    let known_names: Box<SkipSet<String>> = merged
        .iter()
        .map(|tool| tool.function.name.clone())
        .collect();
    let mut runtime_extra = current_tools
        .iter()
        .filter(|tool| explicit_enabled.contains(&tool.function.name))
        .filter(|tool| !known_names.contains(&tool.function.name))
        .cloned()
        .collect::<Vec<_>>();
    let runtime_extra_names = runtime_extra
        .iter()
        .map(|tool| tool.function.name.clone())
        .collect::<Box<SkipSet<_>>>();
    let missing_builtin_names = explicit_enabled
        .iter()
        .filter(|name| !known_names.contains(*name))
        .filter(|name| !runtime_extra_names.contains(*name))
        .cloned()
        .collect::<Vec<_>>();
    // The result of enable_tools and the next skill refresh must not depend on ctx.tools having been written back in time.
    // Builtin tools can be restored directly from the registry; MCP tools are still kept by current_tools.
    runtime_extra.extend(super::super::tools::get_tool_definitions_by_names(
        &missing_builtin_names,
    ));
    if runtime_extra.is_empty() {
        return filter_subagent_hidden_tools(merged);
    }
    merged.extend(runtime_extra);
    rust_tools::sortw::stable_sort_by(&mut merged, |a, b| a.function.name.cmp(&b.function.name));
    filter_subagent_hidden_tools(dedupe_tools_by_name(merged))
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
    // Only re-add the execution baseline that must be resident every turn. Low-frequency skill-discovery tools stay in the process-level
    // allowlist but join the schema on demand via `enable_tools`, preventing the manifest path from bypassing
    // the lazy-loading policy.
    crate::ai::tools::eager_baseline_tool_names()
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

fn should_hide_task_tools_for_subagent() -> bool {
    super::runtime_ctx::current_subagent_depth() > 0
}

fn filter_subagent_hidden_tools(mut tools: Vec<ToolDef>) -> Vec<ToolDef> {
    if should_hide_task_tools_for_subagent() {
        tools.retain(|tool| {
            !super::super::tools::is_subagent_orchestration_tool_name(&tool.function.name)
        });
    }
    tools
}

fn manifest_tool_definitions(tool_groups: &[String], tools: &[String]) -> Option<Vec<ToolDef>> {
    if !tool_groups.is_empty() {
        // Manifest group names resolve against the closed ToolGroup vocabulary;
        // unknown names are skipped (matching how unknown names have always
        // behaved as no-ops here, e.g. typos).
        let groups: Vec<ToolGroup> = tool_groups
            .iter()
            .filter_map(|name| ToolGroup::from_name(name))
            .collect();
        // When expanding via tool_groups, drop the deferred heavy execution
        // primitives (tools carrying the `hidden` metadata flag: process / IPC
        // / shm / env primitives). Their schemas are large and usage rare, so
        // they do not ride along in every turn's request; the model enables
        // them on demand via `enable_tools`, shrinking per-turn tools tokens.
        // Tools named explicitly via `tools:` take the branch below and are
        // never dropped (naming a tool pins it as resident). Core tools like
        // apply_patch / write_file are never marked hidden, so editing
        // capability is unaffected.
        let expanded = super::super::tools::tool_definitions_for_groups(&groups)
            .into_iter()
            .filter(|tool| !super::super::tools::tool_defers_eager_load(&tool.function.name))
            .collect::<Vec<_>>();
        return Some(ensure_required_baseline_tools(expanded));
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
        && manifest_declares_hidden_group(&agent.tool_groups)
}

fn is_executor_skill(skills: &[&SkillManifest]) -> bool {
    // Any active skill declaring a hidden-gating group counts.
    skills
        .iter()
        .any(|skill| manifest_declares_hidden_group(&skill.tool_groups))
}

/// Whether a skill / agent manifest declares a group that gates hidden tools
/// (see `crate::ai::tools::group_gates_hidden_tools`). Unlike the
/// `is_executor_*` helpers this is mode-agnostic: a `mode: all` agent carrying
/// the executor group (e.g. `build`) counts too. Drives the "available on
/// demand via enable_tools" hint for turns whose resident tool set had the
/// heavy execution primitives deferred out by `manifest_tool_definitions`, so
/// read-only agents (plan / explore) never get irrelevant process/IPC hints.
fn declares_hidden_group(
    skills: &[&SkillManifest],
    active_agent: Option<&AgentManifest>,
) -> bool {
    skills
        .iter()
        .any(|s| manifest_declares_hidden_group(&s.tool_groups))
        || active_agent.is_some_and(|a| manifest_declares_hidden_group(&a.tool_groups))
}

/// Case-insensitive hidden-gating-group check: a manifest declares the
/// capability iff any of its group names resolves to a group that gates hidden
/// tools (today that is the executor group). Names resolve against the closed
/// `ToolGroup` vocabulary (manifests historically spelled group names loosely).
fn manifest_declares_hidden_group(groups: &[String]) -> bool {
    groups
        .iter()
        .filter_map(|g| ToolGroup::from_name(g))
        .any(crate::ai::tools::group_gates_hidden_tools)
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
    skills: &[&SkillManifest],
    active_agent: Option<&AgentManifest>,
) -> Vec<ToolDef> {
    // If any active skill disables builtin, disable builtin entirely (most-restrictive)
    if skills.iter().any(|s| s.disable_builtin_tools) {
        return filter_subagent_hidden_tools(Vec::new());
    }
    // Merge tool_groups and tools across all active skills (dedup, order-preserving)
    let mut merged_groups: Vec<String> = Vec::new();
    let mut merged_tools: Vec<String> = Vec::new();
    for skill in skills {
        for g in &skill.tool_groups {
            if !merged_groups.iter().any(|x| x.eq_ignore_ascii_case(g)) {
                merged_groups.push(g.clone());
            }
        }
        for t in &skill.tools {
            if !merged_tools.iter().any(|x| x.eq_ignore_ascii_case(t)) {
                merged_tools.push(t.clone());
            }
        }
    }
    if !merged_groups.is_empty() || !merged_tools.is_empty() {
        if let Some(tool_defs) = manifest_tool_definitions(&merged_groups, &merged_tools) {
            return filter_subagent_hidden_tools(tool_defs);
        }
    }
    if let Some(agent) = active_agent
        && let Some(tool_defs) = manifest_tool_definitions(&agent.tool_groups, &agent.tools)
    {
        return filter_subagent_hidden_tools(tool_defs);
    }
    filter_subagent_hidden_tools(super::super::tools::tool_definitions_for_groups(
        DEFAULT_TURN_TOOL_GROUPS,
    ))
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

    let mut section = String::new();
    for line in lines {
        section.push_str("- ");
        section.push_str(&line);
        section.push('\n');
    }
    if section.ends_with('\n') {
        section.pop();
    }
    builder.push_labeled(kind, title, section);
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
    skills: &[&SkillManifest],
    active_agent: Option<&AgentManifest>,
) -> Vec<String> {
    let mut servers = Vec::new();
    for skill in skills {
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
    skills: &[&SkillManifest],
    active_agent: Option<&AgentManifest>,
) -> Vec<ToolDef> {
    // If any active skill disables mcp, disable mcp entirely
    if skills.iter().any(|skill| skill.disable_mcp_tools) {
        return Vec::new();
    }
    let skill_declares_mcp_servers = skills.iter().any(|skill| !skill.mcp_servers.is_empty());
    if active_agent.is_some_and(|agent| agent.disable_mcp_tools) && !skill_declares_mcp_servers {
        return Vec::new();
    }

    let allowed_servers = resolved_mcp_servers(skills, active_agent);
    if allowed_servers.is_empty() {
        // Lazy by default: do not pre-mount every MCP tool's schema into each request (each schema
        // costs hundreds to thousands of tokens; all MCP tools are the largest and most trimmable chunk of the per-turn tools array,
        // easily hitting the TPM cap). The model can still sense these tools — `build_hidden_mcp_tool_catalog`
        // lists unloaded MCP tool names in the system prompt, and the model loads them on demand via
        // `enable_tools(operation=list/enable)`; already-enabled tools persist across turns
        // via `explicit_enabled_tool_names` + `merge_with_runtime_enabled_tools`.
        // Only when a skill / agent explicitly declares an `mcp_servers` whitelist do we take the
        // eager branch below and mount the matching server tools directly.
        return Vec::new();
    }

    filter_mcp_tools_by_allowed_servers(all_tools, &allowed_servers)
}

fn mcp_tools_for_turn(
    mcp_client: &McpClient,
    skills: &[&SkillManifest],
    active_agent: Option<&AgentManifest>,
) -> Vec<ToolDef> {
    select_mcp_tools(mcp_client.get_all_tools(), skills, active_agent)
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
         If the task needs an external system or MCP-backed capability, discover and enable matching \
         `mcp_*` tools via `enable_tools` first.\n\
         Available: {}",
        displayed
    );
    if remaining > 0 {
        out.push_str(&format!(", and {remaining} more"));
    }
    out.push('.');
    Some(out)
}

fn build_hidden_execution_primitive_catalog(
    available_tools: &Box<SkipSet<String>>,
) -> Option<String> {
    // Deferred heavy-execution primitives (process / IPC / shared-memory / env)
    // are not loaded into every turn by default. List the registered-but-not-
    // loaded names here so the model stays aware they exist and can enable them
    // on demand via `enable_tools`. Mirrors `build_hidden_mcp_tool_catalog`
    // (same MAX_DISPLAY truncation; silent when everything is already loaded).
    let mut hidden: Vec<String> = super::super::tools::deferred_eager_load_tool_summaries()
        .into_iter()
        .map(|(name, _desc)| name)
        .filter(|name| !available_tools.contains_str(name))
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
        "Process / IPC / shared-memory primitives are available but not loaded this turn.\n\
         For multi-process orchestration, background daemons, cross-process IPC, shared memory, \
         or per-process env/working-dir control, enable the needed tools via `enable_tools`.\n\
         Available: {}",
        displayed
    );
    if remaining > 0 {
        out.push_str(&format!(", and {remaining} more"));
    }
    out.push('.');
    Some(out)
}

/// Lazily-loaded subagent orchestration family: task-group tools (group
/// `["builtin", "task"]`) are not part of the default `core` turn schema, and
/// unlike executor primitives they are not covered by `tool_defers_eager_load`
/// (which only defers tools carrying the `hidden` metadata flag), so without
/// this catalog the model has no prompt-level awareness that subagent tools
/// exist and can be enabled on demand — broad multi-branch tasks silently
/// degrade to serial
/// parent work. Mirrors `build_hidden_execution_primitive_catalog`.
/// Top-level agents only: the task family is hidden from subagents
/// (`SUBAGENT_DEPTH > 0`), so the nudge must never reach them.
fn build_hidden_task_tool_catalog(available_tools: &Box<SkipSet<String>>) -> Option<String> {
    if crate::ai::driver::runtime_ctx::current_subagent_depth() != 0 {
        return None;
    }
    const TASK_FAMILY: &[&str] = &[
        "task",
        "task_spawn",
        "task_spawn_batch",
        "task_wait",
        "task_status",
        "task_retry",
        "task_cancel",
        "task_evidence_read",
        "task_audit",
        "task_integrate",
        "manage_team",
        "run_agent_graph",
    ];
    let hidden: Vec<&str> = TASK_FAMILY
        .iter()
        .copied()
        .filter(|name| !available_tools.contains_str(name))
        .collect();
    if hidden.is_empty() {
        return None;
    }

    const MAX_DISPLAY: usize = 8;
    let displayed = hidden
        .iter()
        .take(MAX_DISPLAY)
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ");
    let remaining = hidden.len().saturating_sub(MAX_DISPLAY);

    let mut out = format!(
        "Subagent orchestration tools are available but not loaded in this turn.\n\
         When the task splits into multiple independent branches that would benefit from \
         subagent parallelism (broad discovery, cross-module mapping, independent verification, \
         or concurrent research), enable the needed tools via `enable_tools`.\n\
         Available: {}",
        displayed
    );
    if remaining > 0 {
        out.push_str(&format!(", and {remaining} more"));
    }
    out.push('.');
    Some(out)
}

/// XML attribute-value escaping: `&` `<` `>` `"` `'`.
/// Used for the path attribute of `<instructions path="...">` so quotes/angle brackets in paths cannot break the XML structure.
fn escape_xml_attr(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

fn build_project_instruction_prompt() -> Option<String> {
    let docs = load_project_instruction_docs();
    if docs.is_empty() {
        return None;
    }

    let mut out = String::from(
        "- The current working directory provides project-specific instruction documents.\n\
         - Follow these repo-local constraints and preferences unless they conflict with higher-priority system, developer, or user instructions.\n",
    );
    for doc in docs {
        out.push_str(&format!(
            "\n<instructions path=\"{}\">\n{}\n</instructions>\n",
            escape_xml_attr(&doc.path),
            doc.content.trim()
        ));
    }
    Some(out)
}

fn build_scoped_project_instruction_prompt(targets: &[PathBuf]) -> Option<String> {
    build_scoped_project_instruction_prompt_with_priority(targets, &[])
}

fn build_scoped_project_instruction_prompt_with_priority(
    required_targets: &[PathBuf],
    observed_targets: &[PathBuf],
) -> Option<String> {
    let docs = load_scoped_project_instruction_docs_for_target_priority(
        required_targets,
        observed_targets,
    );
    if docs.is_empty() {
        return None;
    }
    let mut out = String::from(
        "- These documents apply to files already touched in this turn.\n\
         - A rule from a deeper directory is more specific and overrides a conflicting general project rule.\n",
    );
    for doc in docs {
        out.push_str(&format!(
            "\n<instructions path=\"{}\">\n{}\n</instructions>\n",
            escape_xml_attr(&doc.path),
            doc.content.trim()
        ));
    }
    Some(out)
}

pub(super) fn scoped_project_instructions_missing(
    system_prompt: &str,
    targets: &[PathBuf],
) -> bool {
    load_scoped_project_instruction_docs_for_targets(targets)
        .iter()
        .any(|doc| {
            !system_prompt.contains(&format!(
                "<instructions path=\"{}\">",
                escape_xml_attr(&doc.path)
            ))
        })
}

fn push_project_instruction_context(builder: &mut SystemPromptBuilder) {
    if let Some(project_prompt) = build_project_instruction_prompt() {
        builder.push_labeled(
            ContextKind::Policy,
            "project_local_instructions",
            project_prompt,
        );
    }
}

fn push_project_type_context(builder: &mut SystemPromptBuilder) {
    if let Some(kind) = crate::ai::agents::detect_project_kind_from_cwd() {
        // Inject the detected project type and default build/test conventions as a Fact section,
        // so the LLM does not have to guess whether `cargo` / `npm` / `go` applies.
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

/// Session context: tells the model the current session and its data layout, so that in any project
/// (even directories unrelated to rust_tools) it can locate and debug sessionid problems and
/// interact read-only with a given session's content. The model is read-only: no writing, modifying, or deleting session data.
fn session_context_prompt(session_id: &str, session_history_file: &Path, history_file: &Path) -> String {
    let sessions_root = crate::ai::history::SessionStore::new(history_file)
        .sessions_root()
        .display()
        .to_string();
    format!(
        include_str!("system_prompts/session_context.md"),
        session_id,
        session_history_file.display(),
        sessions_root,
    )
}

const MAX_SKILL_ACTIVATION_HISTORY_ENTRIES: usize = 6;

/// Project a successful explicit skill selection into a bounded runtime fact. The raw side-channel record does not enter
/// canonical messages; this only lets later models distinguish "was once selected" from "still currently active".
fn build_skill_activation_history_reminder(events: &[SkillActivationEvent]) -> Option<String> {
    let mut recent: Vec<(&str, &str)> = Vec::with_capacity(MAX_SKILL_ACTIVATION_HISTORY_ENTRIES);
    for event in events.iter().rev() {
        if event.outcome != "injected" {
            continue;
        }
        let Some(skill_name) = event
            .injected_skill
            .as_deref()
            .filter(|name| !name.trim().is_empty())
        else {
            continue;
        };
        if recent.iter().any(|entry| entry.0 == skill_name) {
            continue;
        }
        recent.push((skill_name, event.source.as_str()));
        if recent.len() == MAX_SKILL_ACTIVATION_HISTORY_ENTRIES {
            break;
        }
    }
    if recent.is_empty() {
        return None;
    }

    recent.reverse();
    let mut reminder = String::from(
        "Successful skill selections earlier in this session:\n\
         - These are historical records only. They do not reactivate a skill or change the current turn.\n\
         - Treat only the `<identity>` block's `Active skill` or `Active skills` declaration as current active-skill state.\n",
    );
    for (skill_name, source) in recent {
        reminder.push_str(&format!(
            "- {skill_name:?} was successfully selected via {source:?}.\n"
        ));
    }
    Some(reminder)
}

fn build_system_prompt(
    active_agent: Option<&AgentManifest>,
    skills: &[&SkillManifest],
    available_tools: &Box<SkipSet<String>>,
    ctx: &PromptContext,
) -> SystemPromptBuilder {
    let mut b = SystemPromptBuilder::new();

    // Identity section: merge the generic identity with agent / skill enforcement, avoiding 4 copies of
    // repeated "you must follow ..." flooding the prompt cache.
    let agent_extra = active_agent
        .map(|agent| agent.build_system_prompt())
        .filter(|s| !s.trim().is_empty());

    // Multi-skill stacking: concatenate all skills' prompts in activation order
    let skill_prompts: Vec<String> = skills
        .iter()
        .map(|skill| skill.build_system_prompt())
        .filter(|s| !s.trim().is_empty())
        .collect();
    let skill_extra: Option<String> = if skill_prompts.is_empty() {
        None
    } else {
        Some(skill_prompts.join("\n\n"))
    };

    let identity = if let Some(skill_text) = &skill_extra {
        let mut s = if skills.len() == 1 {
            // Single skill: keep the original compact format
            let skill_name = skills[0].name.as_str();
            format!(
                "Active skill: {skill_name}\n\
                 Its instructions are the primary behavior contract for this turn."
            )
        } else {
            // Multiple skills: list all and state the stacking rules
            let mut header = String::from(
                "Active skills (activation order; equal peers):\n\
                 Their instructions compose additively and form the primary behavior contract \
                 for this turn.\n",
            );
            for (i, skill) in skills.iter().enumerate() {
                use std::fmt::Write;
                let _ = writeln!(header, "  {}. {}", i + 1, skill.name);
            }
            header.push_str(
                "No active skill overrides another; guardrails always take precedence.",
            );
            header
        };
        s.push_str("\n\n<skill_instructions>\n");
        s.push_str(skill_text.trim());
        s.push_str("\n</skill_instructions>");
        if let Some(agent_text) = &agent_extra {
            s.push_str("\n\n<agent_instructions>\n");
            s.push_str(agent_text.trim());
            s.push_str("\n</agent_instructions>");
            s.push_str("\n\nEnforcement: skill instructions override agent instructions when they differ. Use agent instructions only for capabilities, workflow, and defaults not covered by the active skill. Neither skill nor agent instructions override the correctness guardrails (including git-safety rules), safety redlines, or policy sections, which always take precedence.");
        } else {
            s.push_str("\n\nEnforcement: skill instructions override generic assistant guidelines when they differ, except the correctness guardrails (including git-safety rules), safety redlines, and policy sections, which always take precedence.");
        }
        s
    } else {
        agent_extra.unwrap_or_else(|| {
            String::from(
                "You are a general-purpose AI assistant. Match the task: use tools for technical work and reasoning or research otherwise. Answer only what was asked, clearly and concisely.",
            )
        })
    };
    b.push(ContextKind::Identity, identity);
    if !skills.is_empty() && has_tool(available_tools, "request_user_input") {
        b.push(
            ContextKind::Behavior,
            include_str!("system_prompts/interactive_skill_handoff.md"),
        );
    }
    // Question-prompt guidance for non-skill turns, injected only on the default
    // interactive path: skill turns already have the request_user_input handoff
    // protocol, goal mode demands autonomous progress, and background mode has
    // no attached terminal — so none of the three inject it.
    // The guidance also anchors anti-hallucination: when a conclusion depends on
    // the user's private information, ask rather than guess/fabricate; but info
    // that can be checked ourselves is still looked up with tools first — never
    // skip autonomous investigation just because asking is allowed.
    if skills.is_empty() && ctx.goal_mode.is_none() && !ctx.is_background {
        b.push(
            ContextKind::Behavior,
            include_str!("system_prompts/asking_the_user.md"),
        );
    }
    b.push_labeled(
        ContextKind::Behavior,
        "execution_environment",
        runtime_environment_prompt(),
    );

    // Multiple skills: output each resource_path (only when non-empty)
    for skill in skills {
        if let Some(resource_path) = skill.resource_path.as_deref() {
            let trimmed = resource_path.trim();
            if !trimmed.is_empty() {
                b.push(
                    ContextKind::Capability,
                    format!(
                        "<active_skill_resources>\nThe active skill `{}` includes bundled resources at `{}`. When the skill instructions refer to bundled files, scripts, references, examples, or assets, inspect this directory with available file tools and use the relevant resources.\n</active_skill_resources>",
                        skill.name, trimmed
                    ),
                );
            }
        }
    }

    b.push(
        ContextKind::Behavior,
        include_str!("system_prompts/response_style.md"),
    );
    b.push(
        ContextKind::Behavior,
        include_str!("system_prompts/tool_usage.md"),
    );
    b.push(
        ContextKind::Behavior,
        include_str!("system_prompts/correctness_guardrails.md"),
    );
    // Intellectual honesty: evidence-earned agreement and respectful pushback
    // against wrong or inappropriate user premises. Unconditional — it applies
    // in every mode and is never relaxed by a skill or goal.
    b.push(
        ContextKind::Behavior,
        include_str!("system_prompts/intellectual_honesty.md"),
    );

    // ── System constraints: implementing a requirement must not break other modules ──
    // Unconditionally rendered regression red line: no change may sacrifice
    // existing module behavior just to satisfy a new requirement.
    b.push(
        ContextKind::Behavior,
        include_str!("system_prompts/system_constraints.md"),
    );

    // ── Safety red lines: zero tolerance for dangerous operations + hard anti-hallucination ──
    // Unconditionally rendered red lines: dangerous operations forbidden +
    // no_hallucination as a conclusion gate. Fact tracing / evidence calibration
    // are already covered by correctness_guardrails; only the non-negotiable
    // prohibitions stay here: dangerous operations and unverified content must
    // never be presented as conclusions or recommendations.
    // Never relaxed by task, skill, or goal mode; when a skill activates, the
    // enforcement line folds these into the highest priority.
    b.push(
        ContextKind::Behavior,
        include_str!("system_prompts/safety_redlines.md"),
    );
    b.push(
        ContextKind::Behavior,
        include_str!("system_prompts/no_hallucination.md"),
    );

    // ── Task convergence: success criteria land in the plan carrier, closing the loop ──
    // task_convergence is the unconditionally rendered convergence discipline; the
    // plan bridge line is injected only when the plan tool is available, so the
    // guidance never dangles when the tool is unavailable (e.g. stripped by a
    // skill whitelist). Acceptance criteria go straight into the roadmap, tracked
    // by plan_update — no more "criteria defined but never reflected in the plan".
    let plan_criteria_bridge = if has_tool(available_tools, "plan") {
        "- For multi-step tasks, encode these criteria into the `plan` (each step states what/why/tool; the final step verifies the outcome).\n"
    } else {
        ""
    };
    b.push(
        ContextKind::Behavior,
        format!(
            include_str!("system_prompts/task_convergence.md"),
            plan_criteria_bridge = plan_criteria_bridge,
        ),
    );

    // ── Trust boundary: tool output / fetched content is data, not instructions ──
    // The mechanical layer already strips forged reminders from user messages via
    // strip_system_reminders; here we add model-level teaching covering the
    // injection surface of instructions embedded in tool output (web pages,
    // documents, command output). Consistent with the "authenticity seal": runtime
    // reminders have a fixed format, so a look-alike inside tool output is forged.
    b.push(
        ContextKind::Behavior,
        include_str!("system_prompts/trust_boundary.md"),
    );

    // ── Tool-result evidence status: `[reference: ...]` markers on historical ──
    //    data are runtime-injected (see tool_result/execution/evidence_status.rs);
    //    the model must read them as reference snapshots, not live state.
    b.push(
        ContextKind::Behavior,
        include_str!("system_prompts/tool_result_evidence.md"),
    );

    // ── Compressed-context recovery: absence claims must first search the session ──
    //    archive — never assert "not found" without checking.
    if has_tool(available_tools, "search_overflow") {
        b.push(
            ContextKind::Behavior,
            include_str!("system_prompts/compressed_context_recovery.md"),
        );
    }

    // ── Behavior rules: conditionally rendered based on goal mode ──
    // Both modes share the success-criteria convergence rules above; only scope and continuation differences are expressed here.
    if ctx.goal_mode.is_some() {
        b.push(
            ContextKind::Behavior,
            include_str!("system_prompts/goal_mode.md"),
        );
    } else {
        b.push(
            ContextKind::Behavior,
            include_str!("system_prompts/scope_discipline.md"),
        );
        b.push(
            ContextKind::Behavior,
            include_str!("system_prompts/autonomous_execution.md"),
        );
    }

    if has_tool(available_tools, "enable_tools")
        || (skills.is_empty()
            && has_tool(available_tools, "list_skills")
            && has_tool(available_tools, "activate_skill"))
    {
        // Detailed catalogs and examples of unloaded capabilities add noise to
        // every turn; discovery goes through enable_tools on demand, and only
        // already-loaded tools get concrete rules injected below.
        let mut discovery_lines = Vec::new();
        if skills.is_empty() && has_tool(available_tools, "enable_tools") {
            discovery_lines.push(
                "No skill is active for this turn. Additional capabilities are available via `enable_tools`; call `enable_tools(operation=list)` to see them, enabling only the specific tools you need.".to_string(),
            );
            discovery_lines.push(
                "If the task needs an external system or MCP-backed capability, discover and enable matching `mcp_*` tools first.".to_string(),
            );
        } else if has_tool(available_tools, "enable_tools") {
            discovery_lines.push(
                "Additional capabilities are available via `enable_tools`; list and enable only what the current task needs.".to_string(),
            );
        }
        if skills.is_empty()
            && has_tool(available_tools, "list_skills")
            && has_tool(available_tools, "activate_skill")
        {
            discovery_lines.push(
                "Skills are optional. Call `list_skills` when the user asks about them; otherwise assess the task first."
                    .to_string(),
            );
            discovery_lines.push(
                "After assessing the task, proactively call `list_skills` only for a concrete, genuinely specialized need for domain context, an established workflow, bundled resources, or dedicated tools. Call it only when you do not know which skill is available."
                    .to_string(),
            );
            discovery_lines.push(
                "Do not browse skills as a routine opening step.".to_string(),
            );
            discovery_lines.push(
                "A routine source-code, repository, file, or terminal investigation—or technical keywords alone—is not evidence that a skill is needed."
                    .to_string(),
            );
            discovery_lines.push(
                "Call `activate_skill(name=...)` only when one listed skill clearly and materially improves the task. Do not activate for generic work, loose keyword overlap, or just in case.".to_string(),
            );
            discovery_lines.push(
                "Skill activation is limited to the current user turn and unloads automatically after it ends; rediscover and reactivate only when a later turn clearly needs it.".to_string(),
            );
        }
        push_tool_guidance_section(
            &mut b,
            ContextKind::Policy,
            "tool_discovery",
            discovery_lines,
        );
    }

    if has_tool(available_tools, "knowledge_save") {
        let mut lines = vec![
            "If the user asks to remember or states a durable preference/constraint, call `knowledge_save`.".to_string(),
            "When saving a durable principle, preference, safety rule, or coding rule, choose a guideline category such as `common_sense`, `coding_guideline`, `preference`, `user_preference`, or `safety_rules`.".to_string(),
            "Use `user_memory` / `project_info` / `architecture` / `decision_log` for factual knowledge.".to_string(),
            "Save each distinct durable fact at most once per turn. Do not save temporary work notes, raw tool output, or speculative conclusions as global knowledge.".to_string(),
        ];
        let retrieval_tools =
            available_tool_names_in_order(available_tools, &["knowledge_search", "knowledge_list"]);
        if !retrieval_tools.is_empty() {
            lines.push(format!(
                "When asked about remembered info, use {}.",
                format_tool_names(&retrieval_tools)
            ));
        }
        push_tool_guidance_section(&mut b, ContextKind::Policy, "knowledge_save", lines);
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
            lines.push("Simple tasks: act directly. Complex ones: call `plan` first — before the first tool call, so the plan is the roadmap for the whole task.".to_string());
            if has_tool(available_tools, "plan_update") {
                // plan_update's own tool description carries the per-step status
                // semantics; only the track-as-you-go cadence is restated here.
                lines.push("Track step progress with `plan_update` as you work (per-step status semantics live in its tool description).".to_string());
            } else {
                lines.push("Track step progress with `plan_update`: mark a step `running` before starting it and `done` when finished; use `failed`/`skipped` when a step cannot be completed as planned. Each `plan_update` echoes the full plan with per-step status.".to_string());
            }
            lines.push("Treat the plan as a living roadmap: when findings, changed requirements, or a dead end reshape the task, call `plan` again instead of drifting; the latest plan is preserved in full as the task anchor while older versions may be summarized.".to_string());
            if has_tool(available_tools, "task_spawn") {
                lines.push("When planning, mark `delegate: true` on every substantive step, serial or parallel: subagents start with a clean, focused context and the parent reviews results. Keep in the parent only trivial single-tool steps, tightly coupled overlapping edits, and final review/synthesis. `parallelizable: true` means no dependency on earlier steps (concurrent task_spawn); delegated steps without it run one at a time via the synchronous `task`, with the parent handing the needed context in the prompt.".to_string());
            }
        }
        if has_tool(available_tools, "spawn_process") {
            lines.push(
                "Use `spawn_process` only for fire-and-forget background work whose result you do NOT need back (long-running processes, two-way IPC collaboration). It returns a PID, not a result.".to_string(),
            );
        }
        let process_tools = available_tool_names_in_order(
            available_tools,
            &["wait_process", "kill_process", "reap_process"],
        );
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
            "planning_subprocess_execution",
            lines,
        );
    }

    if has_tool(available_tools, "task_spawn")
        || has_tool(available_tools, "task_wait")
        || has_tool(available_tools, "task_status")
        || has_tool(available_tools, "task_integrate")
    {
        let mut lines = Vec::new();
        if has_tool(available_tools, "task_spawn") {
            if has_tool(available_tools, "task") {
                lines.push("Use `task_spawn` to fan out MULTIPLE focused, independent subtasks concurrently. For a single delegated subtask whose result you need back, prefer the synchronous `task` (one spawned task immediately joined by `task_wait` gains no concurrency and just adds overhead).".to_string());
            } else {
                lines.push("Use `task_spawn` to fan out MULTIPLE focused, independent subtasks concurrently. For a single delegated subtask, one spawned task immediately joined by `task_wait` gains no concurrency and just adds overhead.".to_string());
            }
            lines.push("Qualify a subtask when it has a distinct, bounded goal and its expected latency or context-isolation benefit outweighs handoff overhead — a serial step qualifies too when it keeps substantial intermediate reads, searches, logs, or experiments out of the parent context while returning a concise result; parallel branches are not required.".to_string());
            lines.push("Keep pre-division shared discovery sequential; never run dependent steps concurrently, do not delegate merely to create parallelism, and keep work in the parent when net benefit is marginal or uncertain.".to_string());
            lines.push("Prefer delegating broad read-only discovery, cross-module caller or consumer mapping, noisy log or dependency research, and independent adversarial verification. Keep final decisions, tightly coupled overlapping edits, unresolved coupled work, and end-to-end synthesis in the parent; iteration limits, tool failures, and recovery steps are not delegation benefits.".to_string());
            lines.push("Give each subagent an explicit result contract: return a concise conclusion, the key evidence paths/lines or commands, remaining uncertainty, and suggested verification; do not return raw logs, exhaustive search output, or large source excerpts unless requested.".to_string());
            if has_tool(available_tools, "task_spawn_batch") {
                lines.push("Once you identify multiple qualifying subtasks with no data dependency, prefer one `task_spawn_batch` call so dispatch and returned task ids preserve input order. Then continue every independent parent-side step while they run. Do NOT call `task_wait` merely because tasks are running, and do not spawn-wait-spawn-wait serially.".to_string());
            } else {
                lines.push("Once you identify multiple qualifying subtasks with no data dependency, spawn ALL of them in the same response (multiple `task_spawn` calls in one turn). Then continue every independent parent-side step while they run. Do NOT call `task_wait` merely because tasks are running, and do not spawn-wait-spawn-wait serially.".to_string());
            }
        }
        if has_tool(available_tools, "task_wait") {
            lines.push("Call `task_wait` only when the parent is blocked on subagent results or has no productive independent work left. Keep its per-call timeout short (normally 30-60 seconds) and prefer `wait_policy=\"any\"` so the parent resumes on the first useful result.".to_string());
        }
        if has_tool(available_tools, "task_status") {
            lines.push(
                "Use `task_status` for a non-blocking peek while continuing parent-side work."
                    .to_string(),
            );
            lines.push("Before finishing your answer, call `task_status` to confirm no spawned subagent is still running. Never silently drop a spawned task.".to_string());
        }
        if has_tool(available_tools, "task_integrate") {
            lines.push("After `task`, `task_wait`, or `task_status` delivers a result, call `task_integrate` with that task_id, a disposition, and the parent conclusion. Delivery alone is not integration, and normal final answers are blocked while delivered results remain unintegrated.".to_string());
        }
        if has_tool(available_tools, "task_cancel") {
            lines.push("Use `task_cancel` to abandon a stuck or no-longer-needed background subagent instead of repeatedly calling `task_wait` - it terminates the subagent process and writes a cancelled terminal result, but you still must collect that result later with `task_wait` or `task_status`.".to_string());
        }

        if has_tool(available_tools, "task_spawn") {
            lines.push("By default a subagent reuses your (parent) model; only override the `model` field when the subtask is clearly lighter or heavier than your own.".to_string());
            lines.push("Give each subagent a focused context: omitting `inherit` applies the default \"cwd,skills\" (no history/memory), which is right for delegated steps that touch the workspace; use `inherit=\"none\"` only for pure analysis that never touches the workspace; use `inherit=\"all\"` only when the subtask genuinely needs the full conversation.".to_string());
        }
        push_tool_guidance_section(
            &mut b,
            ContextKind::Behavior,
            "async_subagent_orchestration",
            lines,
        );
    }

    if has_tool(available_tools, "knowledge_search")
        || has_tool(available_tools, "knowledge_list")
    {
        let mut lines = Vec::new();
        let search_tools = available_tool_names_in_order(available_tools, &["knowledge_search"]);
        if !search_tools.is_empty() {
            lines.push(format!(
                "Only when the user explicitly asks about remembered or saved knowledge, search with {}.",
                format_tool_names(&search_tools)
            ));
            lines.push(
                "Reuse a successful knowledge search for the rest of the turn. Search again only after knowledge changes or when the query is materially different."
                    .to_string(),
            );
            lines.push(
                "Never fabricate memory: when retrieval returns nothing or insufficient evidence, state that plainly; do not present unretrieved details as remembered facts."
                    .to_string(),
            );
        }
        if has_tool(available_tools, "knowledge_list") {
            lines.push("Use `knowledge_list` when asked what is remembered.".to_string());
        }
        push_tool_guidance_section(&mut b, ContextKind::Policy, "knowledge_retrieval", lines);
    }

    if has_tool(available_tools, "write_file") {
        let mut lines = Vec::new();
        lines.push(
            "To run a script, dump intermediate data, or write a test fixture, create it with `write_file(temp=true)` first, then run it with `execute_command`. Prefer this over inline `python -c '...'` whenever the code is more than a few lines or you need to inspect/edit the file.".to_string(),
        );
        lines.push(
            "Do NOT use `execute_command` to create temp files (e.g. `echo > /tmp/foo`, `python -c '...' > out.json`) — files created outside `write_file(temp=true)` will accumulate. `execute_command` cannot run `rm` either — that is a command-policy blacklist, not a filesystem sandbox: allowed commands run directly against the real workspace.".to_string(),
        );
        if has_tool(available_tools, "apply_patch") {
            lines.push(
                "Do NOT use `write_file(temp=true)` on existing project files; reserve `write_file` without `temp` for genuine full rewrites (localized edits go through `apply_patch`).".to_string(),
            );
            lines.push(
                "When one file needs several localized edits, read the relevant span once. Make ONE `apply_patch` call with multiple `@@` hunks in a single `*** Update File:` section only when every hunk has a unique anchor (distinct surrounding context)."
                    .to_string(),
            );
            lines.push(
                "For several files, use one Begin Patch envelope with one section per target. Do not split related edits into serial read/patch cycles unless a previous patch failed or a later edit truly depends on the earlier edit's result."
                    .to_string(),
            );
            lines.push(
                "For structurally similar blocks (for example, repeated closures with identical bodies), apply one at a time with a distinctive anchor line such as a function name or comment."
                    .to_string(),
            );
            lines.push(
                "Keep each patch well under the size limit. Split large edits into multiple `apply_patch` calls, or write the patch to a temp file and pass `patch_file`."
                    .to_string(),
            );
        } else {
            lines.push(
                "When modifying an existing project file, do NOT use `write_file` with `temp=true`; use `write_file` without `temp` only when a full rewrite is genuinely necessary.".to_string(),
            );
        }
        if has_tool(available_tools, "apply_patch") {
            let line = "To remove an existing project/source/config file, including a git-tracked file, use `apply_patch` with a Begin Patch envelope and a `*** Delete File: <path>` section.".to_string();
            lines.push(line);
        }
        push_tool_guidance_section(&mut b, ContextKind::Behavior, "temporary_files", lines);
    }

    if has_tool(available_tools, "tree") {
        push_tool_guidance_section(
            &mut b,
            ContextKind::Behavior,
            "codebase_navigation",
            vec![
                "Use `tree` to grasp directory layout before reading files, instead of repeatedly listing directories via shell or guessing paths. Then open specific files with `read_file`.".to_string(),
            ],
        );
    }

    b
}

fn build_skill_turn_guard(
    app: &mut App,
    mcp_client: &McpClient,
    skills: &[&SkillManifest],
) -> SkillTurnGuard {
    let all_mcp_tools = mcp_client.get_all_tools();
    super::super::tools::enable_tools::set_available_mcp_tools(all_mcp_tools.clone());
    let matched_skill_names: Vec<String> = skills.iter().map(|s| s.name.clone()).collect();
    let active_agent = app.current_agent_manifest.clone();
    let executor_active = is_executor_skill(skills)
        || active_agent.as_ref().is_some_and(is_executor_agent);
    // Record whether the current turn declares a hidden-gating group into
    // enable_tools' turn-level state, so the heavy execution primitives stay
    // out of the enable catalog for agents that do not declare one (same
    // source of truth as the hidden-catalog hint below, so the two can never
    // disagree).
    let hidden_group_declared = declares_hidden_group(skills, active_agent.as_ref());
    super::super::tools::enable_tools::set_hidden_group_declared(hidden_group_declared);

    let mut builtin_tools = builtin_tools_for_skill(skills, active_agent.as_ref());
    // Externally downloaded skills cannot pre-declare this runtime's continuation protocol; inject this driver-owned tool
    // only while a skill is active, so ordinary turns gain no schema noise or behavior branches.
    if !skills.is_empty() {
        builtin_tools.extend(crate::ai::tools::get_tool_definitions_by_names(&[
            "request_user_input".to_string(),
        ]));
    }
    let mcp_tools = select_mcp_tools(all_mcp_tools.clone(), skills, active_agent.as_ref());
    let available_tools = available_tool_names(&builtin_tools, &mcp_tools);
    let ctx = PromptContext {
        goal_mode: app.goal_mode.clone(),
        is_background: app.cli.background,
    };
    let mut builder = build_system_prompt(active_agent.as_ref(), skills, &available_tools, &ctx);
    if has_tool(&available_tools, "enable_tools")
        && let Some(catalog) = build_hidden_mcp_tool_catalog(&all_mcp_tools, &mcp_tools)
    {
        builder.push(ContextKind::Capability, catalog);
    }
    // For executor agents the heavy execution primitives are deferred out of
    // the resident set by manifest_tool_definitions; add an on-demand enable
    // hint here so the model can still sense them (read-only agents never
    // declare the group, so nothing is injected).
    if has_tool(&available_tools, "enable_tools")
        && hidden_group_declared
        && let Some(catalog) = build_hidden_execution_primitive_catalog(&available_tools)
    {
        builder.push(ContextKind::Capability, catalog);
    }
    // Task-family tools are lazy for every top-level turn (not just executor
    // agents); advertise them so the model loads subagent tools exactly when the
    // task warrants parallelism. Gated internally on top-level depth.
    if has_tool(&available_tools, "enable_tools")
        && let Some(catalog) = build_hidden_task_tool_catalog(&available_tools)
    {
        builder.push(ContextKind::Capability, catalog);
    }
    push_project_context(&mut builder);
    builder.push_labeled(
        ContextKind::Behavior,
        "session_context",
        session_context_prompt(&app.session_id, &app.session_history_file, &app.config.history_file),
    );
    if let Ok(events) = history::read_skill_activation_events_sqlite(&app.session_history_file)
        && let Some(reminder) = build_skill_activation_history_reminder(&events)
    {
        builder.push_labeled(
            ContextKind::Fact,
            "Session Skill Activation History",
            reminder,
        );
    }
    if !app.active_persona.is_default() {
        let mut persona_prompt = format!("- Name: {}\n", app.active_persona.name.trim());
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
        builder.push_labeled(ContextKind::Identity, "persistent_persona", persona_prompt);
    }
    let max_iterations = resolve_max_iterations(active_agent.as_ref(), executor_active);
    let restore_agent_context =
        activate_skill_context(app, builtin_tools, mcp_tools, max_iterations);

    SkillTurnGuard {
        restore_agent_context,
        builder,
        cached_system_prompt: None,
        cached_context_reminder: None,
        matched_skill_names,
    }
}

pub(super) fn rebuild_skill_turn_with_existing_selection(
    app: &mut App,
    mcp_client: &McpClient,
    skill_manifests: &[SkillManifest],
    _question: &str,
    preferred_skill_names: &[String],
) -> SkillTurnGuard {
    // For iteration > 1, keep the previous turn's skills by name only; no more text-similarity re-routing.
    // If the model needs to switch, it can request explicitly via activate_skill.
    let skills: Vec<&SkillManifest> = preferred_skill_names
        .iter()
        .filter_map(|name| skill_manifests.iter().find(|s| &s.name == name))
        .collect();
    build_skill_turn_guard(app, mcp_client, &skills)
}

/// Path taken when the model explicitly requests a skill via the `activate_skill` tool: match by name
/// directly and force-activate its prompt + tool set, skipping auto-routing's scoring/threshold/gating.
///
/// "Don't abuse it" is enforced by the tool side (the name must really exist, and the description requires "clearly matches" before calling) plus
/// the name validation here. After a hit, the active set is kept within the turn by the per-iteration rebuild
/// (`refresh_skill_turn_for_iteration` only adjusts by pending action and does not re-score).
pub(super) fn force_activate_named_skill(
    app: &mut App,
    mcp_client: &McpClient,
    skill_manifests: &[SkillManifest],
    _question: &str,
    requested_names: &[String],
) -> Option<SkillTurnGuard> {
    // Resolve each name into a manifest one by one (skipping misses)
    let skills: Vec<&SkillManifest> = requested_names
        .iter()
        .filter_map(|name| skill_manifests.iter().find(|s| &s.name == name))
        .collect();
    if skills.is_empty() {
        return None;
    }
    let mut guard = build_skill_turn_guard(app, mcp_client, &skills);
    guard.matched_skill_names = skills.iter().map(|s| s.name.clone()).collect();
    Some(guard)
}

fn record_forced_skill_activation(
    app: &App,
    source: ForcedSkillSource,
    requested_skill: &str,
    injected_skill: Option<&str>,
    outcome: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    history::append_skill_activation_event_sqlite(
        &app.session_history_file,
        &SkillActivationEvent {
            requested_skill: requested_skill.to_string(),
            injected_skill: injected_skill.map(str::to_string),
            source: source.label().to_string(),
            outcome: outcome.to_string(),
        },
    )?;
    crate::ai::driver::print::print_skill_activation_note(
        requested_skill,
        injected_skill,
        source.label(),
        outcome,
    );
    Ok(())
}

pub(super) fn prepare_skill_for_turn(
    app: &mut App,
    mcp_client: &McpClient,
    skill_manifests: &[SkillManifest],
    question: &str,
) -> Result<SkillTurnGuard, Box<dyn std::error::Error>> {
    let cfg = configw::get_all_config();
    let debug = cfg
        .get_opt("ai.skills.debug")
        .unwrap_or_default()
        .trim()
        .eq_ignore_ascii_case("true");

    // Skills explicitly forced by the user via `@skills:<name>` or `/skills <name>...` in the input box have the
    // highest priority.
    // This is per-turn semantics: cleared immediately after consumption; not force-injected next turn.
    // It is also the user's signal of explicitly leaving a waiting skill; an old continuation must not grab the turn back.
    let forced_skills = std::mem::take(&mut app.forced_skills);
    let forced_source = app.forced_skill_source.take();
    if !forced_skills.is_empty() {
        app.pending_skill_continuation = None;
    }
    if !forced_skills.is_empty() {
        // Resolve one by one: keep input order, normalize names per manifest and dedup; misses are recorded separately,
        // since one bad name must not sink the whole set (force_activate_named_skill skips unknowns one by one too).
        let mut valid: Vec<String> = Vec::with_capacity(forced_skills.len());
        let mut not_found: Vec<String> = Vec::new();
        for forced in &forced_skills {
            if let Some(skill) = skill_manifests
                .iter()
                .find(|s| s.name == *forced)
                .or_else(|| {
                    skill_manifests
                        .iter()
                        .find(|s| s.name.eq_ignore_ascii_case(forced))
                })
            {
                if !valid.iter().any(|n| n == &skill.name) {
                    valid.push(skill.name.clone());
                }
            } else {
                not_found.push(forced.clone());
            }
        }
        if let Some(source) = forced_source {
            for missing in &not_found {
                record_forced_skill_activation(app, source, missing, None, "not-found")?;
            }
        }
        if !valid.is_empty() {
            if let Some(guard) =
                force_activate_named_skill(app, mcp_client, skill_manifests, question, &valid)
            {
                if let Some(source) = forced_source {
                    for name in &valid {
                        record_forced_skill_activation(app, source, name, Some(name), "injected")?;
                    }
                }
                if debug {
                    eprintln!("[skills] forced via @skills: {}", valid.join(", "));
                }
                return Ok(guard);
            }
            if let Some(source) = forced_source {
                for name in &valid {
                    record_forced_skill_activation(app, source, name, None, "activation-failed")?;
                }
            }
        } else if debug {
            eprintln!(
                "[skills] forced skills not found: {}, no auto-activation",
                forced_skills.join(", ")
            );
        }
    }

    // Consume the explicit continuation created by `request_user_input` exactly once. Re-resolving manifests by name here both
    // avoids treating deleted/renamed external skills as valid state and does not fall back to
    // the old text-similarity-based cross-turn sticky routing.
    if let Some(continuation) = app.pending_skill_continuation.take() {
        let requested_names = continuation.skill_names;
        // Restore only skills that still resolve: one deleted/renamed skill in the set must not sink the whole
        // continuation. force_activate_named_skill skips unknown names one by one internally too; filtering here first
        // lets us distinguish "partially restored" from "all invalid".
        let valid_names: Vec<String> = requested_names
            .iter()
            .filter(|name| skill_manifests.iter().any(|s| &s.name == *name))
            .cloned()
            .collect();
        if !valid_names.is_empty() {
            if let Some(guard) =
                force_activate_named_skill(app, mcp_client, skill_manifests, question, &valid_names)
            {
                if debug {
                    if valid_names.len() < requested_names.len() {
                        eprintln!(
                            "[skills] continuing {} of {} requested skill(s): {} ({} not found)",
                            valid_names.len(),
                            requested_names.len(),
                            valid_names.join(", "),
                            requested_names.len() - valid_names.len()
                        );
                    } else {
                        eprintln!(
                            "[skills] continuing requested skills: {}",
                            valid_names.join(", ")
                        );
                    }
                }
                return Ok(guard);
            }
        } else if debug {
            eprintln!(
                "[skills] pending continuation skill(s) '{}' not found; continuing without it",
                requested_names.join(", ")
            );
        }
    }

    // No automatic skill activation: leave it to the LLM to explicitly choose via activate_skill when needed,
    // or complete the task with existing tools. Cross-turn sticky is also gone — shallow Jaccard
    // matching cannot tell "follow-up on the same skill" from "a different topic that happens to share tokens".
    let skills: &[&SkillManifest] = &[];

    if debug {
        eprintln!("[skills] no auto-activation; explicit activate_skill only");
    }
    let mut guard = build_skill_turn_guard(app, mcp_client, skills);
    guard.matched_skill_names = Vec::new();
    Ok(guard)
}

#[cfg(test)]
#[path = "skill_runtime_tests.rs"]
mod tests;

