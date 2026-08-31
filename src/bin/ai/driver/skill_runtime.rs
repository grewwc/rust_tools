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
                "Skills are optional. Call `list_skills` when the user asks about them; otherwise assess the task first and proactively call `list_skills` only when you identify a concrete, genuinely specialized need for domain context, an established workflow, bundled resources, or dedicated tools and do not know which skill is available. Do not browse skills as a routine opening step; a routine source-code, repository, file, or terminal investigation—or technical keywords alone—is not evidence that a skill is needed.".to_string(),
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
            lines.push("Give each subagent a focused context: the default `inherit` (cwd + skills, no history/memory) is right for delegated steps that touch the workspace; use `inherit=\"none\"` only for pure analysis that never touches the workspace; use `inherit=\"all\"` only when the subtask genuinely needs the full conversation.".to_string());
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
                "When one file needs several localized edits, read the relevant span once and make ONE `apply_patch` call with multiple `@@` hunks in a single `*** Update File:` section — only when every hunk has a unique anchor (distinct surrounding context). For several files, use one Begin Patch envelope with one section per target. Do not split related edits into serial read/patch cycles unless a previous patch failed or a later edit truly depends on the earlier edit's result. For structurally similar blocks (e.g. repeated closures with identical bodies), apply one at a time, each hunk with a distinctive anchor line (function name or comment). Keep each patch under ~4KB: split large edits into multiple apply_patch calls, or write the patch to a temp file and pass `patch_file`.".to_string(),
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
mod tests {
    use super::{
        ContextKind, PromptContext, SystemPromptBuilder, available_tool_names,
        build_hidden_execution_primitive_catalog, build_hidden_mcp_tool_catalog,
        build_hidden_task_tool_catalog, build_project_instruction_prompt,
        build_scoped_project_instruction_prompt, build_system_prompt, builtin_tools_for_skill,
        declares_hidden_group, ensure_required_baseline_tools, escape_xml_attr,
        filter_mcp_tools_by_allowed_servers, has_tool, manifest_tool_definitions,
        merge_with_runtime_enabled_tools, push_project_context, resolve_max_iterations,
        select_mcp_tools, session_context_prompt, tool_uses_mcp_server, ToolGroup,
    };
    use crate::ai::agents::{AgentManifest, AgentMode};
    use crate::ai::driver::runtime_ctx::{SUBAGENT_CWD, SUBAGENT_DEPTH};
    use crate::ai::history::SkillActivationEvent;
    use crate::ai::mcp::McpClient;
    use crate::ai::skills::SkillManifest;
    use crate::ai::tools::enable_tools::set_explicit_enabled_tool_names;
    use crate::ai::types::{FunctionDefinition, PendingSkillContinuation, ToolDefinition};
    use rust_tools::cw::SkipSet;
    use std::fs;
    use std::path::Path;
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
            name: "build".to_string(),
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
    fn default_core_tools_exclude_lazy_skill_discovery_tools() {
        let tools = builtin_tools_for_skill(&[], None);
        let names = tools
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();
        assert!(names.iter().any(|name| name == "enable_tools"));
        assert!(names.iter().any(|name| name == "read_file"));
        // Skill discovery/activation, task orchestration, and knowledge memory are low-frequency capabilities: by default they do not
        // expand into the resident core set; `enable_tools` enables them on demand (the builtin group stays, discoverable dynamically).
        assert!(!names.iter().any(|name| name == "activate_skill"));
        assert!(!names.iter().any(|name| name == "list_skills"));
        assert!(!names.iter().any(|name| name == "load_skill"));
        assert!(!names.iter().any(|name| name == "save_skill"));
        assert!(!names.iter().any(|name| name == "task_spawn"));
        assert!(!names.iter().any(|name| name == "task_integrate"));
        assert!(!names.iter().any(|name| name == "knowledge_save"));
        assert!(!names.iter().any(|name| name == "knowledge_search"));
    }

    #[test]
    fn manifest_tools_keep_skill_discovery_lazy() {
        let groups = vec!["core".to_string(), "executor".to_string()];
        let tools = manifest_tool_definitions(&groups, &[])
            .expect("non-empty manifest groups should resolve tool definitions");
        let names = tools
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();

        assert!(names.iter().any(|name| name == "enable_tools"));
        assert!(names.iter().any(|name| name == "read_file"));
        assert!(!names.iter().any(|name| name == "activate_skill"));
        assert!(!names.iter().any(|name| name == "list_skills"));
        assert!(!names.iter().any(|name| name == "load_skill"));
    }

    #[test]
    fn skill_declaring_task_group_eagerly_loads_only_the_task_family() {
        // Regression guard: task* were moved out of core (lazy builtin-only) so the
        // default turn stays slim, but agent-team / audit_own_changes hard-require the
        // task family in their flows (delegation / spawning the audit subagent). Their
        // manifests declare the narrow `task` group so expansion loads exactly that
        // family. Heavy non-flow orchestration tools (`manage_team`, `run_agent_graph`)
        // live in the narrow `agent_team` group instead of `builtin`: `[core, task]`
        // stays slim (the broad `builtin` tag means "nearly the whole registry",
        // ~15KB of unrelated schemas), while the agent-team skill opts into them by
        // declaring `agent_team` in its manifest.
        let names_for = |groups: Vec<String>| {
            manifest_tool_definitions(&groups, &[])
                .expect("non-empty manifest groups should resolve tool definitions")
                .into_iter()
                .map(|tool| tool.function.name)
                .collect::<Vec<_>>()
        };

        let skill_groups =
            names_for(vec!["core".to_string(), "task".to_string()]);
        assert!(
            skill_groups.iter().any(|name| name == "task_spawn"),
            "skill with [core, task] must eagerly expose task_spawn"
        );
        assert!(skill_groups.iter().any(|name| name == "task"));
        assert!(skill_groups.iter().any(|name| name == "task_integrate"));
        assert!(
            !skill_groups.iter().any(|name| name == "run_agent_graph"),
            "[core, task] must not drag in heavyweight non-flow orchestration tools"
        );
        assert!(!skill_groups.iter().any(|name| name == "manage_team"));
        assert!(!skill_groups.iter().any(|name| name == "save_skill"));
        assert!(!skill_groups.iter().any(|name| name == "knowledge_save"));

        let plain_groups = names_for(vec!["core".to_string(), "executor".to_string()]);
        assert!(
            !plain_groups.iter().any(|name| name == "task_spawn"),
            "default [core, executor] manifest must keep task* lazy"
        );

        let agent_team_groups = names_for(vec![
            "core".to_string(),
            "task".to_string(),
            "agent_team".to_string(),
        ]);
        assert!(
            agent_team_groups.iter().any(|name| name == "manage_team"),
            "[core, task, agent_team] (agent-team skill) must expose manage_team"
        );
        assert!(agent_team_groups.iter().any(|name| name == "run_agent_graph"));
        assert!(
            agent_team_groups.iter().any(|name| name == "send_side_note"),
            "send_side_note rides in via the task group for orchestration skills"
        );
    }

    #[test]
    fn active_skill_prompt_uses_explicit_user_input_handoff_only_for_skills() {
        let active_skill = skill("external-skill", "external workflow");
        let available = {
            let mut available = SkipSet::new(16);
            available.insert("request_user_input".to_string());
            Box::new(available)
        };

        let active_prompt = build_system_prompt(
            None,
            &[&active_skill],
            &available,
            &PromptContext::default(),
        )
        .render_system_prompt();
        assert!(active_prompt.contains("<interactive_skill_handoff>"));
        assert!(active_prompt.contains("`request_user_input`"));

        let ordinary_prompt =
            build_system_prompt(None, &[], &available, &PromptContext::default())
                .render_system_prompt();
        assert!(!ordinary_prompt.contains("<interactive_skill_handoff>"));
    }

    #[test]
    fn explicit_user_input_continuation_restores_only_the_next_turn() {
        let external = skill("external-skill", "external workflow");
        let replacement = skill("replacement", "replacement workflow");
        let mcp_client = McpClient::new();
        let mut app = super::super::tests::test_app("build");

        app.pending_skill_continuation = Some(PendingSkillContinuation {
            skill_names: vec![external.name.clone()],
        });
        let guard = super::prepare_skill_for_turn(
            &mut app,
            &mcp_client,
            std::slice::from_ref(&external),
            "the requested answer",
        )
        .unwrap();
        assert_eq!(guard.primary_skill_name(), Some(external.name.as_str()));
        assert!(app.pending_skill_continuation.is_none());
        drop(guard);

        let next_guard = super::prepare_skill_for_turn(
            &mut app,
            &mcp_client,
            std::slice::from_ref(&external),
            "an unrelated new request",
        )
        .unwrap();
        assert!(next_guard.matched_skill_names().is_empty());
        drop(next_guard);

        app.pending_skill_continuation = Some(PendingSkillContinuation {
            skill_names: vec![external.name.clone()],
        });
        app.forced_skills = vec![replacement.name.clone()];
        let forced_guard = super::prepare_skill_for_turn(
            &mut app,
            &mcp_client,
            &[external, replacement],
            "use the explicitly selected skill",
        )
        .unwrap();
        assert_eq!(forced_guard.primary_skill_name(), Some("replacement"));
        assert!(app.pending_skill_continuation.is_none());
    }

    #[tokio::test]
    async fn subagent_builtin_tools_hide_task_orchestration_family() {
        SUBAGENT_DEPTH
            .scope(1, async {
                let tools = builtin_tools_for_skill(&[], None);
                let names = tools
                    .into_iter()
                    .map(|tool| tool.function.name)
                    .collect::<Box<SkipSet<_>>>();

                for hidden in [
                    "task",
                    "task_spawn",
                    "task_wait",
                    "task_status",
                    "task_integrate",
                    "task_cancel",
                ] {
                    assert!(!names.contains_str(hidden), "{hidden} should be hidden");
                }
                assert!(names.contains_str("read_file"));
            })
            .await;
    }

    fn executor_group_agent(name: &str) -> AgentManifest {
        let mut agent = agent(name, Vec::new());
        agent.mode = AgentMode::All;
        agent.tool_groups = vec!["core".to_string(), "executor".to_string()];
        agent.disable_mcp_tools = true;
        agent
    }

    #[test]
    fn executor_group_defers_process_primitives_but_keeps_core_editing() {
        // build/executor use tool_groups: [core, executor]. Execution primitives (process/IPC/shm/env)
        // are lazy by default and stay out of the resident tool set; apply_patch/write_file (core∩executor) are kept,
        // so editing capability is preserved with zero loss.
        let build_agent = executor_group_agent("build");
        let tools = builtin_tools_for_skill(&[], Some(&build_agent));
        let names = tools
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();

        // Resident: core editing/retrieval capabilities
        assert!(names.iter().any(|n| n == "apply_patch"));
        assert!(names.iter().any(|n| n == "write_file"));
        assert!(names.iter().any(|n| n == "read_file"));
        // Resident: baseline self-service capabilities (enable_tools is the progressive-discovery entry point)
        assert!(names.iter().any(|n| n == "enable_tools"));

        // Lazy: heavy execution primitives and task orchestration (task_spawn etc.) are not resident
        assert!(!names.iter().any(|n| n == "task_spawn"));
        for deferred in [
            "spawn_process",
            "spawn_daemon",
            "send_ipc_message",
            "read_mailbox",
            "shm_create",
            "shm_read",
            "shm_write",
            "shm_delete",
            "signal_process",
            "kill_process",
            "wait_process",
            "reap_process",
            "set_env",
            "set_working_dir",
            "set_process_group",
            "signal_process_group",
            "sleep_process",
            "ps_processes",
            "ps_ipc",
        ] {
            assert!(
                !names.iter().any(|n| n == deferred),
                "executor primitive `{deferred}` should be lazy-loaded, not eagerly mounted"
            );
        }
    }

    #[test]
    fn explicit_tools_list_is_not_filtered_by_lazy_load() {
        // Tools explicitly named via `tools:` become resident: even a named execution primitive is not culled.
        let mut agent = agent("custom", Vec::new());
        agent.tool_groups = Vec::new();
        agent.tools = vec!["read_file".to_string(), "spawn_process".to_string()];

        let tools = builtin_tools_for_skill(&[], Some(&agent));
        let names = tools
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();

        assert!(names.iter().any(|n| n == "read_file"));
        assert!(
            names.iter().any(|n| n == "spawn_process"),
            "explicitly named executor primitive must stay eager"
        );
    }

    #[test]
    fn hidden_execution_primitive_catalog_advertises_deferred_tools() {
        // After lazy loading the model must still stay aware: when these
        // primitives are not loaded this turn, the catalog must list them and
        // give the enable_tools path.
        let mut available = SkipSet::new(16);
        available.insert("read_file".to_string());
        available.insert("enable_tools".to_string());

        let catalog = build_hidden_execution_primitive_catalog(&Box::new(available))
            .expect("deferred primitives should produce a catalog");
        assert!(catalog.contains("enable the needed tools via `enable_tools`"));
        // The catalog shows the first MAX_DISPLAY(8) entries in sorted order and
        // folds the rest into "and N more". `kill_process` sorts first and must
        // be inside the display window; assert on it rather than `spawn_process`
        // (the latter sorts last and falls outside the truncation).
        assert!(catalog.contains("kill_process"));
        assert!(catalog.contains("more."));
    }

    #[test]
    fn hidden_execution_primitive_catalog_suppressed_when_all_loaded() {
        // When every executor primitive is already loaded (e.g. an explicit
        // full whitelist), the catalog is suppressed: nothing to advertise.
        let mut available = SkipSet::new(16);
        for (name, _desc) in crate::ai::tools::deferred_eager_load_tool_summaries() {
            available.insert(name);
        }
        assert!(build_hidden_execution_primitive_catalog(&Box::new(available)).is_none());
    }

    #[test]
    fn hidden_task_tool_catalog_advertises_lazy_subagent_family() {
        // A default core turn carries no task group: the catalog must list the
        // unloaded subagent tool names and give the enable_tools path so the
        // model knows subagent tools can be loaded on demand.
        let mut available = SkipSet::new(16);
        available.insert("read_file".to_string());
        available.insert("enable_tools".to_string());

        let catalog = build_hidden_task_tool_catalog(&Box::new(available))
            .expect("unloaded task family should produce a catalog");
        assert!(catalog.contains("enable the needed tools via `enable_tools`"));
        assert!(catalog.contains("task_spawn"));
        assert!(catalog.contains("task_spawn_batch"));
        // The display window takes the first MAX_DISPLAY(8) entries in family
        // order and folds the rest into "and N more".
        assert!(catalog.contains("more."));
    }

    #[test]
    fn hidden_task_tool_catalog_suppressed_when_family_loaded() {
        // Once the task family is loaded (explicit enable or a skill/agent
        // declaration), the catalog must not repeat the nudge.
        let mut available = SkipSet::new(16);
        for name in [
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
        ] {
            available.insert(name.to_string());
        }
        assert!(build_hidden_task_tool_catalog(&Box::new(available)).is_none());
    }

    #[tokio::test]
    async fn hidden_task_tool_catalog_suppressed_inside_subagent() {
        // Subagents can never enable the task family (it is hidden when
        // SUBAGENT_DEPTH > 0), so the nudge would be misleading there; the
        // catalog must stay silent at depth > 0.
        SUBAGENT_DEPTH.scope(1, async {
            let mut available = SkipSet::new(16);
            available.insert("read_file".to_string());
            available.insert("enable_tools".to_string());
            assert!(build_hidden_task_tool_catalog(&Box::new(available)).is_none());
        });
    }

    #[test]
    fn declares_hidden_group_only_true_for_executor_agents() {
        let build_agent = executor_group_agent("build");
        assert!(declares_hidden_group(&[], Some(&build_agent)));

        // plan/explore use an explicit tools list without any executor group.
        let mut plan_agent = agent("plan", Vec::new());
        plan_agent.mode = AgentMode::All;
        plan_agent.tool_groups = Vec::new();
        plan_agent.tools = vec!["read_file".to_string()];
        assert!(!declares_hidden_group(&[], Some(&plan_agent)));

        assert!(!declares_hidden_group(&[], None));
    }

    #[test]
    fn lazy_load_measurably_shrinks_build_agent_tool_payload() {
        // Quantify, via the production serialization path, the actual per-request tools-token savings of
        // lazy-loading culling of execution primitives: tools in the request body is compact JSON of Vec<ToolDefinition>, and
        // request/builder.rs's estimate_tools_tokens counts serde_json::to_string characters / 2 (conservative conversion,
        // CHARS_PER_TOKEN_CONSERVATIVE) into each turn's prompt. Compare:
        //   baseline = the executor group fully expanded via tool_groups (pre-lazy-loading behavior)
        //   optimized = current builtin_tools_for_skill (after lazy loading, deferred primitives culled)
        const CHARS_PER_TOKEN_CONSERVATIVE: usize = 2;
        let build_agent = executor_group_agent("build");

        // baseline: expand [core, executor] fully with no deferred filtering
        // (the pre-lazy-load behavior).
        let groups = [ToolGroup::Core, ToolGroup::Executor];
        let baseline_tools =
            ensure_required_baseline_tools(crate::ai::tools::tool_definitions_for_groups(&groups));
        // optimized: current production path (manifest_tool_definitions culls deferred primitives).
        let optimized_tools = builtin_tools_for_skill(&[], Some(&build_agent));

        let ser = |tools: &[ToolDefinition]| -> usize {
            serde_json::to_string(tools)
                .map(|s| s.chars().count())
                .unwrap_or(0)
        };
        let baseline_chars = ser(&baseline_tools);
        let optimized_chars = ser(&optimized_tools);
        let saved_chars = baseline_chars.saturating_sub(optimized_chars);
        let saved_tokens = saved_chars.div_ceil(CHARS_PER_TOKEN_CONSERVATIVE);
        let pct = (saved_chars as f64 / baseline_chars as f64) * 100.0;

        eprintln!(
            "[lazy-load measurement] build agent tools: baseline={} tools / {} chars (~{} tok), \
             optimized={} tools / {} chars (~{} tok), saved={} chars (~{} tok, {:.1}%)",
            baseline_tools.len(),
            baseline_chars,
            baseline_chars.div_ceil(CHARS_PER_TOKEN_CONSERVATIVE),
            optimized_tools.len(),
            optimized_chars,
            optimized_chars.div_ceil(CHARS_PER_TOKEN_CONSERVATIVE),
            saved_chars,
            saved_tokens,
            pct,
        );

        // The optimization must genuinely shrink the payload: culled primitive count == deferred catalog size, and the token
        // savings must be significant (conservative lower bound 800 tok/turn, measured ~1.1–1.2k). If anyone ever adds these
        // primitives back to core or removes the filter, this assertion goes red immediately.
        let deferred_count = crate::ai::tools::deferred_eager_load_tool_summaries().len();
        assert_eq!(
            baseline_tools.len() - optimized_tools.len(),
            deferred_count,
            "optimized set must drop exactly the deferred executor primitives"
        );
        assert!(
            saved_tokens >= 800,
            "expected >=800 tokens saved per turn, got {saved_tokens}"
        );
    }

    #[test]
    fn system_prompt_only_mentions_tools_available_this_turn() {
        let mut available = SkipSet::new(16);
        available.insert("read_file".to_string());
        available.insert("apply_patch".to_string());
        available.insert("enable_tools".to_string());

        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();
        assert!(prompt.contains("<tool_usage>"));
        assert!(prompt.contains("<tool_discovery>"));
        assert!(prompt.contains("<trust_boundary>"));
        assert!(prompt.contains("Additional capabilities are available via `enable_tools`"));
        assert!(!prompt.contains("Capability catalog (not yet loaded"));
        assert!(!prompt.contains("Configured MCP tools are available"));
        assert!(!prompt.contains("Process / IPC / shared-memory primitives are available"));
        assert!(!prompt.contains("Feishu/Lark"));
        assert!(!prompt.contains("Web search:"));
        assert!(!prompt.contains("<knowledge_retrieval>"));
        assert!(!prompt.contains("cargo_test"));
        assert!(!prompt.contains("execute_command"));
        assert!(!prompt.contains("apply_patch"));
        assert!(!prompt.contains("<compressed_context_recovery>"));
    }

    #[test]
    fn system_prompt_forbids_breaking_other_modules_to_satisfy_a_requirement() {
        // Unconditionally rendered system constraint: implementing a requirement must not come at the cost of breaking other modules; conflicts must be reported, not silently broken.
        // Goal mode is the most likely to "sacrifice existing functionality to reach the goal", so both the default and goal paths must verify the constraint is present.
        let cases = [
            PromptContext::default(),
            PromptContext {
                goal_mode: Some("finish the goal".to_string()),
                is_background: false,
            },
        ];
        for ctx in cases {
            let prompt =
                build_system_prompt(None, &[], &Box::new(SkipSet::new(16)), &ctx)
                    .render_system_prompt();
            assert!(prompt.contains("<system_constraints>"));
            assert!(prompt.contains(
                "Never break another module's functionality to satisfy a requirement"
            ));
            assert!(prompt.contains("surface the conflict"));
        }
    }

    #[test]
    fn system_prompt_bridges_compressed_context_recovery_via_search_overflow() {
        // search_overflow is a core resident tool but must be explicitly bridged to the compression pipeline:
        // after context compression, absence claims must first search the session archive instead of directly asserting "not found".
        let mut available = SkipSet::new(16);
        available.insert("search_overflow".to_string());

        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();

        assert!(prompt.contains("<compressed_context_recovery>"));
        assert!(prompt.contains("search the session archive with `search_overflow`"));
        assert!(prompt.contains("absence claims must cover the archived scope"));
        assert!(prompt.contains("verbatim excerpts"));
        assert!(prompt.contains("scope=all"));
    }

    #[test]
    fn system_prompt_includes_runtime_environment_and_effective_cwd() {
        let available = SkipSet::new(16);
        let effective_cwd = std::env::temp_dir().join("rust_tools_prompt_cwd");
        let prompt = SUBAGENT_CWD.sync_scope(effective_cwd.clone(), || {
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt()
        });

        assert!(prompt.contains("<execution_environment>"));
        assert!(prompt.contains("Operating system:"));
        assert!(prompt.contains(std::env::consts::OS));
        assert!(prompt.contains(std::env::consts::ARCH));
        assert!(prompt.contains("Shell:"));
        assert!(prompt.contains(&format!(
            "Effective working directory: `{}`",
            effective_cwd.display()
        )));
        assert!(prompt.contains("Relative tool paths resolve against this directory"));
        assert!(prompt.contains("not necessarily the project root"));
        assert!(prompt.contains("Write commands for this OS/shell"));
    }

    #[test]
    fn session_context_prompt_mentions_id_layout_and_read_only_rule() {
        let mut builder = SystemPromptBuilder::new();
        builder.push_labeled(
            ContextKind::Behavior,
            "session_context",
            session_context_prompt(
                "f6bb0f1c-ce48-4283-96ab-27ab297ed6b7",
                Path::new(
                    "/Users/u/.history_file.sessions/f6bb0f1c-ce48-4283-96ab-27ab297ed6b7.sqlite",
                ),
                Path::new("/Users/u/.history_file.sqlite"),
            ),
        );
        let rendered = builder.render_system_prompt();
        assert!(
            rendered.contains("<session_context>")
                && !rendered.contains("<session_context>\n<session_context>")
                && rendered.contains("f6bb0f1c-ce48-4283-96ab-27ab297ed6b7")
                && rendered.contains(".history_file.sessions")
                && rendered.contains("<id>.sqlite")
                && rendered.contains("sqlite3")
                && rendered.contains("Read-only rule"),
            "session_context must carry the id, storage layout, read paths and the read-only rule; got:\n{rendered}"
        );
    }

    #[test]
    fn system_prompt_routes_project_file_deletes_to_apply_patch() {
        let mut available = SkipSet::new(16);
        available.insert("write_file".to_string());
        available.insert("apply_patch".to_string());

        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();

        assert!(prompt.contains("<temporary_files>"));
        assert!(prompt.contains("git-tracked file"));
        assert!(prompt.contains("`apply_patch`"));
        assert!(prompt.contains("ONE `apply_patch` call with multiple `@@` hunks"));
        assert!(prompt.contains("unique anchor"));
        assert!(prompt.contains("structurally similar blocks"));
        assert!(prompt.contains("Do not split related edits into serial read/patch cycles"));
        assert!(prompt.contains("`*** Delete File: <path>`"));
    }

    #[test]
    fn system_prompt_enforces_concise_response_style_with_correctness_safeguard() {
        let available = SkipSet::new(16);
        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();
        // The style section must exist and require "answer first, no rambling"
        assert!(prompt.contains("<response_style>"));
        assert!(prompt.contains("Lead with the answer or action"));
        // Must keep the "conciseness must not trade away correctness" safety pad to prevent over-compression from causing wrong judgments
        assert!(prompt.contains("Be concise without sacrificing correctness"));
        assert!(prompt.contains("Skip preambles, restatements, meta-commentary"));
        assert!(prompt.contains("status only at real milestones or plan changes"));
    }

    #[test]
    fn system_prompt_renders_safety_redlines_and_no_hallucination() {
        let available = SkipSet::new(16);
        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();
        // Dangerous-operation red lines: unconditionally rendered, containing all three elements — forbidden / bypass / confirmation
        assert!(prompt.contains("<safety_redlines>"));
        assert!(prompt.contains("Never perform dangerous operations"));
        assert!(prompt.contains("Never bypass or work around safety mechanisms"));
        assert!(prompt.contains("state the exact command and its consequences and wait for approval"));
        // Anti-hallucination red line: unconditionally rendered; fact tracing /
        // evidence calibration is covered by correctness_guardrails, so this only
        // verifies the purely prohibitive phrasing and the duty to label
        // inference / unknown.
        assert!(prompt.contains("<no_hallucination>"));
        assert!(prompt.contains("Never present unverified content"));
        assert!(prompt.contains(
            "label inferences with their basis and state unknowns as unknown"
        ));
    }

    #[test]
    fn system_prompt_uses_criterion_based_parallel_delegation() {
        let mut available = SkipSet::new(16);
        available.insert("plan".to_string());
        available.insert("task_spawn".to_string());
        available.insert("task_wait".to_string());
        available.insert("task_status".to_string());

        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();

        // Delegation is the default choice for substantive steps (serial or parallel), but with clear boundaries; the parent keeps
        // trivial steps, tightly coupled edits, and final review, and never runs dependent steps concurrently.
        assert!(prompt.contains("fan out MULTIPLE focused, independent subtasks concurrently"));
        assert!(prompt.contains("mark `delegate: true` on every substantive step"));
        assert!(prompt.contains("delegated steps without it run one at a time via the synchronous `task`"));
        assert!(prompt.contains("distinct, bounded goal"));
        assert!(prompt
            .contains("latency or context-isolation benefit outweighs handoff overhead"));
        // Pre-division shared discovery must complete serially; never run
        // dependent steps concurrently, never delegate just to create
        // parallelism, and keep work in the parent when the net benefit is
        // unclear.
        assert!(prompt.contains("Keep pre-division shared discovery sequential"));
        assert!(prompt.contains("never run dependent steps concurrently"));
        assert!(prompt.contains("do not delegate merely to create parallelism"));
        assert!(prompt.contains("net benefit is marginal or uncertain"));
        // The parent keeps final decisions, tightly coupled edits, and end-to-end
        // synthesis.
        assert!(prompt.contains("Keep final decisions, tightly coupled overlapping edits"));
        assert!(prompt.contains("broad read-only discovery"));
        assert!(prompt.contains("explicit result contract"));
        // Serial steps may also be delegated for context isolation (parallel
        // branches are not required).
        assert!(prompt
            .contains("experiments out of the parent context"));
        assert!(prompt.contains("parallel branches are not required"));
        assert!(prompt.contains("continue every independent parent-side step while they run"));
        assert!(prompt.contains("only when the parent is blocked on subagent results"));
        assert!(prompt.contains("Use `task_status` for a non-blocking peek while continuing"));
        assert!(!prompt.contains("certainty is not required"));
    }

    #[test]
    fn system_prompt_routes_single_delegation_to_sync_task_when_available() {
        // The Async Subagent Orchestration section is the single authority for
        // task routing. When the synchronous `task` tool is also available, a
        // SINGLE delegated subtask must be steered to `task` instead of a lone
        // task_spawn+task_wait; the Planning section must not repeat that rule
        // (dedup keeps token cost down without losing guidance).
        let mut available = SkipSet::new(16);
        available.insert("task".to_string());
        available.insert("task_spawn".to_string());
        available.insert("task_wait".to_string());
        available.insert("task_status".to_string());
        available.insert("plan".to_string());

        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();

        // Canonical home: Async Subagent Orchestration section (task branch).
        assert!(
            prompt.contains(
                "For a single delegated subtask whose result you need back, prefer the synchronous `task`"
            )
        );
        // The Planning section no longer duplicates the routing rule...
        assert!(!prompt.contains("for a SINGLE subtask use the synchronous `task`"));
        assert!(!prompt.contains("is just a slower `task`"));
        // ...but still keeps its own unique plan/delegate guidance.
        assert!(prompt.contains(
            "Simple tasks: act directly. Complex ones: call `plan` first — before the first tool call"
        ));
        assert!(prompt.contains("mark `delegate: true` on every substantive step"));
    }

    #[test]
    fn system_prompt_task_only_tools_skip_empty_planning_section() {
        // Edge case: only task_* tools available (no plan / spawn_process /
        // process / ipc). The Planning section's outer gate includes
        // task_spawn/task_wait, so the section *would* open - but with no
        // plan/sub-process/ipc tools every inner block is skipped, `lines`
        // stays empty, and push_tool_guidance_section early-returns. Routing
        // guidance must come solely from the Async section, with no empty
        // Planning header left over.
        let mut available = SkipSet::new(16);
        available.insert("task".to_string());
        available.insert("task_spawn".to_string());
        available.insert("task_wait".to_string());
        available.insert("task_status".to_string());
        available.insert("task_integrate".to_string());
        available.insert("task_cancel".to_string());
        available.insert("task_spawn_batch".to_string());

        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();

        // No empty Planning section header.
        assert!(!prompt.contains("<planning_subprocess_execution>"));
        // Routing guidance still present, from the Async section alone.
        assert!(
            prompt.contains(
                "For a single delegated subtask whose result you need back, prefer the synchronous `task`"
            )
        );
        assert!(prompt.contains("<async_subagent_orchestration>"));
    }

    #[test]
    fn system_prompt_forbids_guessing_without_sufficient_evidence() {
        let available = SkipSet::new(16);
        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();
        assert!(prompt.contains("<correctness_guardrails>"));
        assert!(prompt.contains("Ground factual claims in observed evidence"));
        assert!(prompt.contains("state what is verified, what is unknown"));
        assert!(prompt.contains("Calibrate verification effort to a claim's consequence"));
        assert!(prompt.contains("prefer direct evidence when reasonably accessible"));
        assert!(prompt.contains("separate evidence-backed premises from judgment"));
        assert!(prompt.contains("navigation aids rather than independent proof"));
        assert!(prompt.contains("reopen underlying evidence only when it could materially change"));
        assert!(prompt.contains("limit absence claims to the scope actually searched"));
        assert!(prompt.contains("locate relevant callers and dependents"));
        assert!(prompt.contains("compilation and tests prove only covered behavior"));
        assert!(prompt.contains("consequences supported by traced evidence"));
        assert!(prompt.contains("keep unresolved hypotheses separate"));
        assert!(prompt.contains("distinguish introduced behavior from pre-existing behavior"));
        assert!(prompt.contains("reset, checkout, restore, stash drop"));
        assert!(prompt.contains("temporary branch/worktree or stash push then pop"));
        // Anti-hallucination bullet: every concrete specific must trace to
        // session-observed evidence, with explicit abstention allowed. The old
        // in-bullet meta-sentence guarding against re-verifying settled facts
        // was trimmed as redundant; the efficiency guard now lives in
        // task_convergence's stopping rule and must keep rendering.
        assert!(prompt.contains("must trace to evidence observed in this session"));
        assert!(prompt.contains("not to memory or plausibility"));
        assert!(prompt.contains("beats a confident guess"));
        assert!(prompt.contains("Do not pursue perfect certainty or unrelated detail"));
    }

    #[test]
    fn system_prompt_requires_self_contained_comments() {
        // Comments rule lives in correctness_guardrails (always-precedence
        // section), not in an agent/skill manifest — it must render for every
        // session regardless of active agent or skill.
        let available = SkipSet::new(16);
        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();
        assert!(prompt.contains("<correctness_guardrails>"));
        assert!(prompt.contains("Write comments for a reader who only has the code"));
        assert!(prompt.contains("Never reference a discussion-only shorthand or codename"));
        assert!(prompt.contains("state what was decided and why"));
    }

    #[test]
    fn system_prompt_scope_discipline_bullets_have_no_leaked_indentation() {
        // Regression: the non-goal Scope Discipline block once dropped the
        // `\n\` line-continuation on two bullets, baking ~13 spaces of source
        // indentation into the rendered prompt. Assert every rendered line is
        // left-trimmed (no leading whitespace leaks from the source literal).
        let available = SkipSet::new(16);
        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();
        assert!(prompt.contains("<scope_discipline>"));
        // The three bullets must each start at column 0 (bullet marker), not be
        // prefixed by leaked source indentation.
        assert!(prompt
            .contains("\n- Investigate the user's explicit request plus only the direct dependencies"));
        assert!(prompt
            .contains("\n- Do not implement refactors or optimizations unrelated to the task;"));
        assert!(prompt.contains("\n- For broad requests, define investigation boundaries"));
        // Guard against the exact defect: no bullet prefixed by leading spaces.
        assert!(!prompt.contains("\n             - Do not implement refactors"));
        assert!(!prompt.contains("\n             - For broad requests, define"));
    }

    #[test]
    fn system_prompt_defines_an_end_to_end_behavior_contract() {
        let available = SkipSet::new(16);
        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();

        assert!(prompt.contains("current plan and interpretation as hypotheses"));
        assert!(prompt.contains("user correction, failed check, or new evidence"));
        assert!(prompt.contains("conclusions and actions that depended on it"));
        assert!(prompt.contains("Do not patch only the literal symptom"));
        assert!(prompt.contains("approval of adjacent behavior"));
        assert!(prompt.contains("observable outcomes and preserved invariants"));
        assert!(prompt.contains("what must change, what must stay unchanged"));
        assert!(prompt.contains("disappearance of the original symptom"));
    }

    #[test]
    fn system_prompt_links_task_convergence_criteria_to_plan_when_available() {
        // When plan is available: task_convergence injects the "write acceptance
        // criteria into the plan" bridge line; the planning block injected under
        // the same condition carries the plan_update usage notes, and the two
        // together close the plan → execute → accept loop.
        let mut available = SkipSet::new(16);
        available.insert("plan".to_string());
        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();
        assert!(prompt.contains(
            "For multi-step tasks, encode these criteria into the `plan`"
        ));
        assert!(prompt.contains("Track step progress with `plan_update`"));
        assert!(prompt.contains("Treat the plan as a living roadmap"));
        assert!(prompt.contains("before the first tool call, so the plan is the roadmap"));

        // When plan is unavailable (e.g. culled by a skill whitelist): the bridging line is absent, but the task_convergence body remains.
        let empty =
            build_system_prompt(None, &[], &Box::new(SkipSet::new(16)), &PromptContext::default())
                .render_system_prompt();
        assert!(empty.contains("<task_convergence>"));
        assert!(!empty.contains("encode these criteria into the `plan`"));
    }

    #[test]
    fn system_prompt_bounds_tool_exploration() {
        let available = SkipSet::new(16);
        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();
        assert!(prompt.contains("Give every call a concrete decision goal"));
        assert!(prompt.contains("no further call can change the decision"));
        assert!(prompt.contains("Before exploration, state the question it can answer"));
        assert!(prompt.contains("Do not batch code reads or reread visible content"));
    }

    #[test]
    fn system_prompt_uses_success_criteria_for_normal_and_goal_convergence() {
        let available = SkipSet::new(16);
        let normal = build_system_prompt(
            None,
            &[],
            &Box::new(available.clone()),
            &PromptContext::default(),
        )
        .render_system_prompt();
        assert!(normal.contains("<task_convergence>"));
        assert!(normal.contains("task-level success criteria"));
        assert!(normal.contains(
            "the next call can verify it, rule out a live hypothesis, or complete required work"
        ));
        assert!(
            normal.contains("Stop when all criteria are verified or a specific blocker remains")
        );
        assert!(normal.contains("evidence count alone is not a stopping rule"));
        assert!(!normal.contains("3+ pieces of converging evidence"));

        let goal = build_system_prompt(
            None,
            &[],
            &Box::new(available),
            &PromptContext {
                goal_mode: Some("analyze the design".to_string()),
                is_background: false,
            },
        )
        .render_system_prompt();
        assert!(goal.contains("Analysis-only goals are complete"));
        assert!(!goal.contains("your job is to act, not analyze"));
        assert!(!goal.contains("every detail of the goal is complete"));
    }

    #[test]
    fn asking_user_guidance_only_on_default_interactive_path() {
        let available = Box::new(SkipSet::new(16));
        let build_agent = agent("build", vec![]);

        let interactive = build_system_prompt(
            Some(&build_agent),
            &[],
            &available,
            &PromptContext::default(),
        )
        .render_system_prompt();
        assert!(interactive.contains("<asking_the_user>"));
        assert!(interactive.contains("information only the user can provide"));

        let goal = build_system_prompt(
            Some(&build_agent),
            &[],
            &available,
            &PromptContext {
                goal_mode: Some("finish the goal".to_string()),
                is_background: false,
            },
        )
        .render_system_prompt();
        assert!(!goal.contains("<asking_the_user>"));

        let background = build_system_prompt(
            Some(&build_agent),
            &[],
            &available,
            &PromptContext {
                goal_mode: None,
                is_background: true,
            },
        )
        .render_system_prompt();
        assert!(!background.contains("<asking_the_user>"));

        let skill_turn = build_system_prompt(
            Some(&build_agent),
            &[&skill("humanizer", "Rewrite text naturally")],
            &available,
            &PromptContext::default(),
        )
        .render_system_prompt();
        assert!(!skill_turn.contains("<asking_the_user>"));
    }

    #[test]
    fn system_prompt_stops_repeating_failed_approach_without_ending_task() {
        let available = SkipSet::new(16);
        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();

        assert!(prompt.contains("On failure, diagnose before retrying"));
        assert!(prompt.contains("switch to a materially different safe recovery"));
        assert!(!prompt.contains("after 3 failed attempts on the same issue, stop and report"));
    }

    #[test]
    fn system_prompt_keeps_code_grounding_calls_serial() {
        let available = SkipSet::new(16);
        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();
        assert!(prompt.contains("Navigate code serially"));
        assert!(prompt.contains(
            "read one sufficiently broad needed region, then patch it"
        ));
        assert!(prompt.contains("Do not batch code reads"));
        assert!(
            !prompt
                .contains("Work in batches: when several independent read-only lookups are needed")
        );
    }

    #[test]
    fn generic_system_prompt_does_not_hardcode_repo_specific_tool_names() {
        let available = SkipSet::new(16);
        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();
        assert!(!prompt.contains("cargo_test"));
        assert!(!prompt.contains("execute_command / cargo_test"));
        assert!(!prompt.contains("execute_command"));
        assert!(!prompt.contains("apply_patch"));
        assert!(prompt.contains("Use tools for requested work"));
        assert!(prompt.contains("if unavailable, say so instead of pretending"));
    }

    #[test]
    fn system_prompt_mentions_mcp_discovery_when_enable_tools_available() {
        let mut available = SkipSet::new(16);
        available.insert("enable_tools".to_string());
        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();
        assert!(prompt.contains("discover and enable matching `mcp_*` tools first"));
    }

    #[test]
    fn system_prompt_guides_tree_for_layout_when_available() {
        // When no tree, do not inject the navigation section; with a tree, prompt to grasp the structure with tree first, then read_file,
        // avoiding blind ls / recursive reads by the model.
        let without = build_system_prompt(
            None,
            &[],
            &Box::new(SkipSet::new(16)),
            &PromptContext::default(),
        )
        .render_system_prompt();
        assert!(!without.contains("<codebase_navigation>"));

        let mut available = SkipSet::new(16);
        available.insert("tree".to_string());
        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();
        assert!(prompt.contains("<codebase_navigation>"));
        assert!(prompt.contains("Use `tree` to grasp directory layout before reading files"));
    }

    #[test]
    fn hidden_mcp_tool_catalog_lists_real_available_tools() {
        let catalog = build_hidden_mcp_tool_catalog(
            &[
                tool("mcp_feishu_docs_get_text_by_url"),
                tool("mcp_feishu_doc_create_from_markdown"),
                tool("mcp_pdf-extract_pdf_extract_text"),
            ],
            &[tool("mcp_feishu_doc_create_from_markdown")],
        )
        .unwrap();

        assert!(catalog.contains("Configured MCP tools are available but not loaded"));
        assert!(catalog.contains("discover and enable matching `mcp_*` tools"));
        assert!(catalog.contains("`enable_tools`"));
        assert!(catalog.contains("`mcp_feishu_docs_get_text_by_url`"));
        assert!(catalog.contains("`mcp_pdf-extract_pdf_extract_text`"));
        assert!(!catalog.contains("`mcp_feishu_doc_create_from_markdown`"));
    }

    #[test]
    fn hidden_mcp_tool_catalog_omits_prompt_when_everything_is_loaded() {
        let catalog = build_hidden_mcp_tool_catalog(
            &[tool("mcp_feishu_docs_get_text_by_url")],
            &[tool("mcp_feishu_docs_get_text_by_url")],
        );
        assert!(catalog.is_none());
    }

    #[test]
    fn narrow_skill_whitelist_still_lets_model_discover_explicitly_requested_mcp_tools() {
        // Scenario: the user explicitly asks to "write a Feishu doc with MCP
        // tools", but the active skill's narrow tools: whitelist replaces the
        // tool set with a single dedicated tool, and the default agent carries
        // disable_mcp_tools (no mcp_* pre-mounted). Before the fix: the narrow
        // whitelist squeezed out enable_tools too, the hidden MCP catalog gate
        // (has_tool("enable_tools")) closed with it, and all three MCP discovery
        // paths were severed, making the request physically impossible to serve.
        // After the fix: enable_tools is restored as an always-on baseline, the
        // catalog gate holds again, and the model can discover and enable
        // mcp_feishu_*.
        let mut narrow_skill = skill("feishu-upload", "Upload markdown into Feishu docs");
        narrow_skill.tools = vec!["write_file".to_string()];

        // 1) After the narrow whitelist replaces the tool set, the baseline
        // fallback still restores discovery/loading and basic read-only entries.
        let builtin_tools = builtin_tools_for_skill(&[&narrow_skill], None);
        let builtin_names = builtin_tools
            .iter()
            .map(|tool| tool.function.name.clone())
            .collect::<Vec<_>>();
        assert!(
            builtin_names.contains(&"write_file".to_string()),
            "explicitly declared tools in the skill whitelist must be preserved"
        );
        assert!(
            builtin_names.contains(&"enable_tools".to_string()),
            "enable_tools must be restored as an always-on baseline, otherwise the model cannot discover/enable MCP tools"
        );
        assert!(
            builtin_names.contains(&"read_file".to_string()),
            "read_file must stay as a baseline read-only capability to read the user-named test.md"
        );

        // 2) The default agent's disable_mcp_tools => no mcp_* is pre-mounted
        // this turn.
        let all_mcp_tools = vec![
            tool("mcp_feishu_doc_create_from_markdown"),
            tool("mcp_feishu_docs_get_text_by_url"),
            tool("mcp_pdf-extract_pdf_extract_text"),
        ];
        let loaded_mcp_tools: Vec<ToolDefinition> = Vec::new();

        // 3) available_tools contains enable_tools => the catalog injection gate
        // holds (the production code's has_tool("enable_tools") check).
        let available_tools = available_tool_names(&builtin_tools, &loaded_mcp_tools);
        assert!(
            has_tool(&available_tools, "enable_tools"),
            "the catalog injection gate depends on enable_tools being present in available_tools"
        );

        // 4) The hidden MCP catalog exposes the user-requested mcp_feishu_* to
        // the model as the discovery entry point.
        let catalog = build_hidden_mcp_tool_catalog(&all_mcp_tools, &loaded_mcp_tools)
            .expect("a discovery hint is required when unloaded mcp_* tools exist");
        assert!(catalog.contains("discover and enable matching `mcp_*` tools"));
        assert!(catalog.contains("`enable_tools`"));
        assert!(catalog.contains("`mcp_feishu_doc_create_from_markdown`"));
        assert!(catalog.contains("`mcp_feishu_docs_get_text_by_url`"));
    }

    #[test]
    fn system_prompt_guides_optional_skill_discovery_when_available() {
        let mut available = SkipSet::new(16);
        available.insert("enable_tools".to_string());
        available.insert("activate_skill".to_string());
        available.insert("list_skills".to_string());
        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();
        assert!(prompt.contains("activate_skill"));
        assert!(prompt.contains("list_skills"));
        assert!(prompt.contains("Skills are optional"));
        assert!(prompt.contains("proactively call `list_skills`"));
        assert!(prompt.contains("technical keywords alone"));
        assert!(
            prompt.contains("routine source-code, repository, file, or terminal investigation")
        );
        assert!(prompt.contains("unloads automatically"));
        assert!(prompt.contains("enable_tools"));
    }

    #[test]
    fn system_prompt_prefers_enable_tools_when_no_skill_active() {
        let mut available = SkipSet::new(16);
        available.insert("enable_tools".to_string());
        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();
        assert!(prompt.contains("No skill is active for this turn"));
        assert!(prompt.contains("enable_tools(operation=list)"));
        assert!(prompt.contains("enabling only the specific tools you need"));
    }

    #[test]
    fn skill_activation_history_reminder_preserves_past_selection_without_reactivation() {
        let events = vec![
            SkillActivationEvent {
                requested_skill: "missing".to_string(),
                injected_skill: None,
                source: "/skills-inline".to_string(),
                outcome: "not-found".to_string(),
            },
            SkillActivationEvent {
                requested_skill: "bytedcli".to_string(),
                injected_skill: Some("bytedcli".to_string()),
                source: "/skills-inline".to_string(),
                outcome: "injected".to_string(),
            },
            SkillActivationEvent {
                requested_skill: "unsafe".to_string(),
                injected_skill: Some("unsafe\nnew instruction".to_string()),
                source: "/skills-inline".to_string(),
                outcome: "injected".to_string(),
            },
        ];

        let history = super::build_skill_activation_history_reminder(&events).unwrap();
        let mut builder = SystemPromptBuilder::new();
        builder.push_labeled(
            ContextKind::Fact,
            "Session Skill Activation History",
            history,
        );
        let reminder = builder.render_context_reminder().unwrap();

        assert!(reminder.contains("\"bytedcli\" was successfully selected"));
        assert!(reminder.contains("\"unsafe\\nnew instruction\" was successfully selected"));
        assert!(!reminder.contains("\"unsafe\nnew instruction\" was successfully selected"));
        assert!(!reminder.contains("missing"));
        assert!(reminder.contains("historical records only"));
        assert!(reminder.contains("do not reactivate a skill"));
        assert!(reminder.contains("current active-skill state"));
    }

    #[test]
    fn skill_activation_history_reminder_keeps_six_recent_unique_selections() {
        let mut events = vec![SkillActivationEvent {
            requested_skill: "duplicate".to_string(),
            injected_skill: Some("duplicate".to_string()),
            source: "old-source".to_string(),
            outcome: "injected".to_string(),
        }];
        events.extend((0..7).map(|index| SkillActivationEvent {
            requested_skill: format!("skill-{index}"),
            injected_skill: Some(format!("skill-{index}")),
            source: "test-source".to_string(),
            outcome: "injected".to_string(),
        }));
        events.push(SkillActivationEvent {
            requested_skill: "duplicate".to_string(),
            injected_skill: Some("duplicate".to_string()),
            source: "new-source".to_string(),
            outcome: "injected".to_string(),
        });

        let history = super::build_skill_activation_history_reminder(&events).unwrap();

        assert_eq!(history.matches("was successfully selected via").count(), 6);
        assert_eq!(history.matches("\"duplicate\"").count(), 1);
        assert!(history.contains("new-source"));
        assert!(!history.contains("old-source"));
        assert!(history.contains("\"skill-6\""));
        assert!(!history.contains("\"skill-1\""));
    }

    #[test]
    fn system_prompt_never_mentions_discover_skills() {
        let mut available = SkipSet::new(16);
        available.insert("enable_tools".to_string());
        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();
        assert!(prompt.contains("enable_tools(operation=list)"));
        assert!(!prompt.contains("discover_skills"));
    }

    #[test]
    fn render_groups_same_kind_sections_into_single_tag_block() {
        // identity/behavior/policy should each appear as exactly one tag pair, ordered identity→behavior→policy,
        // This guarantees that "appended afterwards" identity sections like persona are not pushed to the end of the prompt
        // but cluster with the generic identity into one block.
        let mut builder = SystemPromptBuilder::new();
        builder.push(ContextKind::Identity, "Generic identity.");
        builder.push(ContextKind::Behavior, "Behavior one.");
        builder.push(ContextKind::Policy, "Policy one.");
        builder.push(ContextKind::Behavior, "Behavior two.");
        // Simulate persona being appended after build_system_prompt:
        builder.push(ContextKind::Identity, "Persona identity.");

        let prompt = builder.render_system_prompt();

        assert_eq!(prompt.matches("<identity>").count(), 1);
        assert_eq!(prompt.matches("<behavior>").count(), 1);
        assert_eq!(prompt.matches("<policy>").count(), 1);

        let identity_pos = prompt.find("<identity>").unwrap();
        let behavior_pos = prompt.find("<behavior>").unwrap();
        let policy_pos = prompt.find("<policy>").unwrap();
        assert!(identity_pos < behavior_pos && behavior_pos < policy_pos);

        // The two identity sections must land inside the same identity block, in insertion order.
        let generic_pos = prompt.find("Generic identity.").unwrap();
        let persona_pos = prompt.find("Persona identity.").unwrap();
        let identity_close = prompt.find("</identity>").unwrap();
        assert!(generic_pos < persona_pos);
        assert!(persona_pos < identity_close);

        // Sections of the same kind are separated by blank lines; insertion order is kept within the group.
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
    fn render_uses_xml_tags_for_labeled_system_sections() {
        let mut builder = SystemPromptBuilder::new();
        builder.push_labeled(
            ContextKind::Behavior,
            "runtime_guard",
            "Verify before claiming completion.",
        );
        builder.push_labeled(ContextKind::Behavior, "  ", "Unlabeled fallback.");

        let prompt = builder.render_system_prompt();

        assert!(prompt.contains(
            "<behavior>\n<runtime_guard>\nVerify before claiming completion.\n</runtime_guard>\n\nUnlabeled fallback.\n</behavior>"
        ));
        assert!(!prompt.contains("## \n"));
    }

    #[test]
    fn render_keeps_capabilities_in_system_prompt_out_of_fact_reminder() {
        let mut builder = SystemPromptBuilder::new();
        builder.push(
            ContextKind::Capability,
            "Configured MCP tools can be loaded with `enable_tools`.",
        );
        builder.push_labeled(ContextKind::Fact, "Project Type", "Rust project.");

        let prompt = builder.render_system_prompt();
        let reminder = builder.render_context_reminder().unwrap();

        assert!(prompt.contains("<capabilities>"));
        assert!(prompt.contains("Configured MCP tools can be loaded"));
        assert!(!prompt.contains("You should not respond to this context"));
        assert!(!reminder.contains("Configured MCP tools can be loaded"));
        assert!(reminder.contains("Project Type"));
        assert!(reminder.contains("## Project Type"));
    }

    #[test]
    fn active_skill_resources_are_system_capabilities() {
        let available = SkipSet::new(16);
        let mut active_skill = skill("resource-skill", "Uses bundled references");
        active_skill.resource_path = Some("/private/resource-skill/resources".to_string());

        let builder = build_system_prompt(
            None,
            &[&active_skill],
            &Box::new(available),
            &PromptContext::default(),
        );
        let prompt = builder.render_system_prompt();
        let reminder = builder.render_context_reminder().unwrap_or_default();

        assert!(prompt.contains("<active_skill_resources>"));
        assert!(prompt.contains("/private/resource-skill/resources"));
        assert!(!reminder.contains("/private/resource-skill/resources"));
    }

    #[test]
    fn active_skill_prompt_precedes_agent_prompt_and_declares_priority() {
        let available = SkipSet::new(16);
        let mut build_agent = agent("build", vec![]);
        build_agent.prompt = "You are the build agent.".to_string();
        let mut humanizer = skill("humanizer", "Rewrite text naturally");
        humanizer.prompt = "You are a writing editor.".to_string();

        let prompt = build_system_prompt(
            Some(&build_agent),
            &[&humanizer],
            &Box::new(available),
            &PromptContext::default(),
        )
        .render_system_prompt();

        let skill_pos = prompt.find("Active skill: humanizer").unwrap();
        let agent_pos = prompt.find("<agent_instructions>").unwrap();
        assert!(skill_pos < agent_pos);
        assert!(prompt.contains("<skill_instructions>\nYou are a writing editor.\n</skill_instructions>"));
        assert!(prompt.contains("<agent_instructions>\nYou are the build agent.\n</agent_instructions>"));
        assert!(prompt.contains("primary behavior contract"));
        assert!(prompt.contains("skill instructions override agent instructions"));
    }

    #[test]
    fn skill_only_prompt_keeps_guardrails_non_overridable() {
        let available = SkipSet::new(16);
        let mut humanizer = skill("humanizer", "Rewrite text naturally");
        humanizer.prompt = "You are a writing editor.".to_string();

        let prompt = build_system_prompt(None, &[&humanizer], &Box::new(available), &PromptContext::default())
            .render_system_prompt();

        assert!(prompt.contains("skill instructions override generic assistant guidelines"));
        assert!(prompt.contains("except the correctness guardrails (including git-safety rules), safety redlines, and policy sections, which always take precedence"));
        assert!(!prompt.contains("<agent_instructions>"));
    }

    #[test]
    fn system_prompt_uses_knowledge_save_for_user_memory_requests() {
        let mut available = SkipSet::new(16);
        available.insert("knowledge_save".to_string());
        available.insert("knowledge_search".to_string());
        available.insert("knowledge_list".to_string());

        let prompt =
            build_system_prompt(None, &[], &Box::new(available), &PromptContext::default())
                .render_system_prompt();
        assert!(prompt.contains("<knowledge_save>"));
        assert!(prompt.contains("call `knowledge_save`"));
        assert!(prompt.contains("`common_sense`, `coding_guideline`"));
        assert!(prompt.contains("Save each distinct durable fact at most once per turn"));
        assert!(prompt.contains("<knowledge_retrieval>"));
        assert!(prompt.contains("Only when the user explicitly asks"));
        assert!(prompt.contains("Reuse a successful knowledge search"));
        assert!(prompt.contains("Never fabricate memory"));
        assert!(prompt.contains("Use `knowledge_list` when asked what is remembered"));
        assert!(!prompt.contains("call `memory_save`"));
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

        assert!(
            prompt.contains("- The current working directory provides project-specific instruction documents.")
        );
        assert!(prompt.contains("<instructions path="));
        assert!(prompt.contains("AGENTS.md"));
        assert!(prompt.contains("Use cargo fmt before commit."));
        assert!(prompt.contains("claude.md"));
        assert!(prompt.contains("Web app uses pnpm."));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn escape_xml_attr_escapes_all_five_special_chars() {
        assert_eq!(
            escape_xml_attr("a&b<c>d\"e'f"),
            "a&amp;b&lt;c&gt;d&quot;e&apos;f"
        );
        // Ordinary path preserved as-is
        assert_eq!(
            escape_xml_attr("/src/bin/ai/AGENTS.md"),
            "/src/bin/ai/AGENTS.md"
        );
        // Empty string is safe
        assert_eq!(escape_xml_attr(""), "");
    }

    #[test]
    fn scoped_project_instruction_prompt_follows_observed_target_path() {
        let root = temp_dir("target_project_prompt");
        let target = root.join("src/bin/ai/driver/iteration.rs");
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(root.join("AGENTS.md"), "Root safety rules.\n").unwrap();
        fs::write(root.join("src/bin/ai/AGENTS.md"), "AI runtime rules.\n").unwrap();
        fs::write(
            root.join("src/bin/ai/driver/AGENTS.md"),
            "Driver-specific rules.\n",
        )
        .unwrap();
        fs::write(&target, "// source\n").unwrap();

        let prompt = SUBAGENT_CWD
            .sync_scope(root.clone(), || {
                build_scoped_project_instruction_prompt(std::slice::from_ref(&target))
            })
            .expect("target-scoped instruction prompt");

        assert!(
            prompt.contains("- These documents apply to files already touched in this turn.")
        );
        assert!(prompt.contains("<instructions path="));
        assert!(prompt.contains("AI runtime rules."));
        assert!(prompt.contains("Driver-specific rules."));
        assert!(!prompt.contains("Root safety rules."));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn scoped_project_instruction_push_confirms_required_target_is_loaded() {
        let root = temp_dir("required_target_project_prompt");
        let target = root.join("src/bin/ai/driver/iteration.rs");
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(root.join("AGENTS.md"), "Root safety rules.\n").unwrap();
        fs::write(
            root.join("src/bin/ai/driver/AGENTS.md"),
            "Required driver rules.\n",
        )
        .unwrap();
        fs::write(&target, "// source\n").unwrap();

        SUBAGENT_CWD.sync_scope(root.clone(), || {
            let mut guard = super::SkillTurnGuard {
                restore_agent_context: None,
                builder: SystemPromptBuilder::new(),
                cached_system_prompt: None,
                cached_context_reminder: None,
                matched_skill_names: Vec::new(),
            };

            assert!(guard.push_scoped_project_instructions(
                std::slice::from_ref(&target),
                &[]
            ));
            assert!(guard.system_prompt().contains("Required driver rules."));
        });

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn context_reminder_injects_active_skill_pointer_into_user_message_reminder() {
        let mut guard = super::SkillTurnGuard {
            restore_agent_context: None,
            builder: SystemPromptBuilder::new(),
            cached_system_prompt: None,
            cached_context_reminder: None,
            matched_skill_names: vec![
                "alpha".to_string(),
                "beta".to_string(),
                "gamma".to_string(),
            ],
        };
        let reminder = guard
            .context_reminder()
            .expect("reminder should be present with active skills");
        assert!(reminder.contains("<system-reminder>"));
        assert!(reminder.contains(
            "Active skills at turn start (in activation order):"
        ));
        assert!(reminder.contains("  1. alpha"));
        assert!(reminder.contains("  2. beta"));
        assert!(reminder.contains("  3. gamma"));
        assert!(reminder.contains("<skill_instructions>"));
        assert!(reminder.contains("primary behavior contract for this turn"));
    }

    #[test]
    fn context_reminder_single_active_skill_uses_singular_pointer() {
        let mut guard = super::SkillTurnGuard {
            restore_agent_context: None,
            builder: SystemPromptBuilder::new(),
            cached_system_prompt: None,
            cached_context_reminder: None,
            matched_skill_names: vec!["solo".to_string()],
        };
        let reminder = guard
            .context_reminder()
            .expect("reminder should be present with an active skill");
        assert!(reminder.contains("Active skill at turn start: solo."));
        assert!(!reminder.contains("in activation order"));
    }

    #[test]
    fn context_reminder_omits_skill_pointer_without_active_skills() {
        let mut guard = super::SkillTurnGuard {
            restore_agent_context: None,
            builder: SystemPromptBuilder::new(),
            cached_system_prompt: None,
            cached_context_reminder: None,
            matched_skill_names: Vec::new(),
        };
        assert!(guard.context_reminder().is_none());
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
                let mut builder = build_system_prompt(
                    None,
                    &[],
                    &Box::new(available),
                    &PromptContext::default(),
                );
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
            let mut builder =
                build_system_prompt(None, &[], &Box::new(available), &PromptContext::default());
            push_project_context(&mut builder);
            builder.render_system_prompt()
        });

        assert!(prompt.contains("<project_local_instructions>"));
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
            let mut builder =
                build_system_prompt(None, &[], &Box::new(available), &PromptContext::default());
            push_project_context(&mut builder);
            builder.render_system_prompt()
        });

        assert!(prompt.contains("<project_local_instructions>"));
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
            let mut builder =
                build_system_prompt(None, &[], &Box::new(available), &PromptContext::default());
            push_project_context(&mut builder);
            builder.render_system_prompt()
        });

        assert!(prompt.contains("<project_local_instructions>"));
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
            disable_builtin_tools: false,
            disable_mcp_tools: false,
            prompt: String::new(),
            system_prompt: None,
            priority: 0,
            excludes: Vec::new(),
            parent: None,
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
            model_tier: None,
            disabled: false,
            hidden: false,
            color: None,
            source_path: None,
        }
    }

    #[test]
    fn manifest_skill_group_does_not_expose_request_user_input_without_active_skill() {
        let mut active_agent = agent("ordinary", vec![]);
        active_agent.tool_groups.push("skill".to_string());

        let tools = super::builtin_tools_for_skill(&[], Some(&active_agent));

        assert!(
            !tools
                .iter()
                .any(|tool| tool.function.name == "request_user_input"),
            "manifest tool_groups must not expose the skill-only handoff control tool"
        );
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
    fn agent_without_mcp_servers_field_lazy_loads_mcp_tools() {
        // With no mcp_servers whitelist, lazy-load by default: pre-mount no MCP tool schemas;
        // the model loads them on demand via the hidden MCP catalog + enable_tools.
        let all_tools = vec![
            tool("mcp_feishu_docs_search"),
            tool("mcp_ocr_extract"),
            tool("mcp_other_lookup"),
        ];
        let build_agent = agent("build", vec![]);

        let tools = select_mcp_tools(all_tools, &[], Some(&build_agent));

        assert!(tools.is_empty());
    }

    #[test]
    fn agent_disable_mcp_tools_hides_default_mcp_tools() {
        let all_tools = vec![tool("mcp_feishu_docs_search"), tool("mcp_ocr_extract")];
        let mut build_agent = agent("build", vec![]);
        build_agent.disable_mcp_tools = true;

        let tools = select_mcp_tools(all_tools, &[], Some(&build_agent));

        assert!(tools.is_empty());
    }

    #[test]
    fn skill_mcp_servers_can_opt_in_when_agent_disables_mcp_tools() {
        let all_tools = vec![tool("mcp_feishu_docs_search"), tool("mcp_ocr_extract")];
        let mut build_agent = agent("build", vec![]);
        build_agent.disable_mcp_tools = true;
        let mut s = skill("feishu-docs", "");
        s.mcp_servers = vec!["feishu".to_string()];

        let tools = select_mcp_tools(all_tools, &[&s], Some(&build_agent));
        let names = tools
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["mcp_feishu_docs_search".to_string()]);
    }

    #[test]
    fn no_active_agent_or_skill_lazy_loads_mcp_tools() {
        // With neither a skill nor an agent whitelist, lazy-load by default too: return empty and leave loading to
        // the hidden MCP catalog + enable_tools on demand.
        let all_tools = vec![tool("mcp_feishu_docs_search"), tool("mcp_other_lookup")];

        let tools = select_mcp_tools(all_tools, &[], None);

        assert!(tools.is_empty());
    }

    #[test]
    fn skill_disable_mcp_tools_overrides_agent_default_fallback() {
        let all_tools = vec![tool("mcp_feishu_docs_search"), tool("mcp_other_lookup")];
        let build_agent = agent("build", vec![]);
        let mut s = skill("focus", "");
        s.disable_mcp_tools = true;

        let tools = select_mcp_tools(all_tools, &[&s], Some(&build_agent));
        assert!(tools.is_empty());
    }

    #[test]
    fn explicit_agent_whitelist_still_narrows_when_set() {
        let all_tools = vec![tool("mcp_feishu_docs_search"), tool("mcp_ocr_extract")];
        let agent = agent("build", vec!["feishu"]);

        let tools = select_mcp_tools(all_tools, &[], Some(&agent));
        let names = tools
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["mcp_feishu_docs_search".to_string()]);
    }

    #[test]
    fn runtime_enabled_tools_are_preserved_when_refreshing_context() {
        let _guard = EXPLICIT_TOOL_TEST_GUARD.lock().unwrap();
        set_explicit_enabled_tool_names(vec![
            "enable_tools".to_string(),
            "knowledge_search".to_string(),
        ]);
        let merged = merge_with_runtime_enabled_tools(
            vec![tool("read_file"), tool("enable_tools")],
            vec![],
            &[
                tool("read_file"),
                tool("enable_tools"),
                tool("knowledge_search"),
            ],
        );
        let names = merged
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"read_file".to_string()));
        assert!(names.contains(&"enable_tools".to_string()));
        assert!(names.contains(&"knowledge_search".to_string()));
        set_explicit_enabled_tool_names(Vec::new());
    }

    #[test]
    fn runtime_enabled_builtin_is_restored_when_current_context_missed_writeback() {
        let _guard = EXPLICIT_TOOL_TEST_GUARD.lock().unwrap();
        set_explicit_enabled_tool_names(vec!["knowledge_consolidate".to_string()]);

        let merged = merge_with_runtime_enabled_tools(vec![tool("read_file")], vec![], &[]);
        let names = merged
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();

        assert!(names.contains(&"read_file".to_string()));
        assert!(names.contains(&"knowledge_consolidate".to_string()));
        set_explicit_enabled_tool_names(Vec::new());
    }

    #[test]
    fn subagent_runtime_enabled_task_tools_stay_hidden() {
        let _guard = EXPLICIT_TOOL_TEST_GUARD.lock().unwrap();
        SUBAGENT_DEPTH.sync_scope(1, || {
            // The explicit-enabled list is isolated per owner (including subagent context); it must be written
            // within sync_scope so reads use the same owner.
            set_explicit_enabled_tool_names(vec![
                "task_wait".to_string(),
                "task_cancel".to_string(),
                "knowledge_search".to_string(),
            ]);
            let merged = merge_with_runtime_enabled_tools(
                vec![tool("read_file"), tool("enable_tools")],
                vec![],
                &[
                    tool("task_wait"),
                    tool("task_cancel"),
                    tool("knowledge_search"),
                ],
            );
            let names = merged
                .into_iter()
                .map(|tool| tool.function.name)
                .collect::<Vec<_>>();

            assert!(names.contains(&"read_file".to_string()));
            assert!(names.contains(&"knowledge_search".to_string()));
            assert!(!names.contains(&"task_wait".to_string()));
            assert!(!names.contains(&"task_cancel".to_string()));
            set_explicit_enabled_tool_names(Vec::new());
        });
    }

    #[test]
    fn non_explicit_skill_tools_do_not_leak_into_next_context() {
        let _guard = EXPLICIT_TOOL_TEST_GUARD.lock().unwrap();
        set_explicit_enabled_tool_names(vec!["knowledge_search".to_string()]);
        let merged = merge_with_runtime_enabled_tools(
            vec![tool("read_file")],
            vec![],
            &[
                tool("read_file"),
                tool("apply_patch"),
                tool("knowledge_search"),
            ],
        );
        let names = merged
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"read_file".to_string()));
        assert!(names.contains(&"knowledge_search".to_string()));
        assert!(!names.contains(&"apply_patch".to_string()));
        set_explicit_enabled_tool_names(Vec::new());
    }

    #[test]
    fn explicit_tool_lists_keep_baseline_entries_available() {
        let merged = ensure_required_baseline_tools(vec![tool("read_file")]);
        let names = merged
            .into_iter()
            .map(|tool| tool.function.name)
            .collect::<Vec<_>>();
        assert!(names.contains(&"enable_tools".to_string()));
        // Basic read-only / retrieval capabilities should be re-added as the resident baseline, so a narrow-whitelist skill cannot
        // cull the most basic reading tools like read_file and leave the main Agent unable to read user-named files.
        assert!(names.contains(&"read_file".to_string()));
        // Task orchestration / knowledge memory series have been moved out of the baseline: under narrow-whitelist skills they are no longer
        // auto-re-added, but the main Agent can still progressively discover and enable them via the resident enable_tools,
        // so delegation of subagents is never truly lost.
        assert!(!names.contains(&"task".to_string()));
        assert!(!names.contains(&"task_spawn".to_string()));
        assert!(!names.contains(&"task_wait".to_string()));
        assert!(!names.contains(&"task_status".to_string()));
        assert!(!names.contains(&"knowledge_save".to_string()));
        // Other non-baseline builtin tools still must not be dragged into the whitelist without cause.
        assert!(!names.contains(&"plan".to_string()));
        assert!(!names.contains(&"write_file".to_string()));
        assert!(!names.contains(&"apply_patch".to_string()));
    }

    #[test]
    fn multi_skill_merges_tool_groups_from_all_skills() {
        let mut skill_a = skill("alpha", "alpha skill");
        skill_a.tool_groups.push("skill".to_string());
        let mut skill_b = skill("beta", "beta skill");
        skill_b.tool_groups.push("builtin".to_string());
        let tools = super::builtin_tools_for_skill(&[&skill_a, &skill_b], None);
        let names: Vec<String> = tools.iter().map(|t| t.function.name.clone()).collect();
        assert!(names.contains(&"enable_tools".to_string()));
        // The skill group provides list_skills / activate_skill; the builtin group provides enable_tools
        assert!(names.contains(&"list_skills".to_string()));
        assert!(names.contains(&"activate_skill".to_string()));
    }

    #[test]
    fn multi_skill_system_prompt_lists_all_active_skills() {
        let skill_a = skill_with_prompt("alpha", "alpha skill", "Do alpha things.");
        let skill_b = skill_with_prompt("beta", "beta skill", "Do beta things.");
        let available_tools: Box<SkipSet<String>> = Box::new(SkipSet::new(16));
        let ctx = PromptContext { goal_mode: None, is_background: false };
        let builder = super::build_system_prompt(None, &[&skill_a, &skill_b], &available_tools, &ctx);
        let prompt = builder.render_system_prompt();
        assert!(prompt.contains("alpha"));
        assert!(prompt.contains("beta"));
        assert!(prompt.contains("activation order"));
        assert!(prompt.contains("Do alpha things."));
        assert!(prompt.contains("Do beta things."));
    }

    #[test]
    fn multi_skill_any_disable_builtin_tools_disables_all() {
        let skill_a = skill("alpha", "alpha skill");
        let mut skill_b = skill("beta", "beta skill");
        skill_b.disable_builtin_tools = true;
        let tools = super::builtin_tools_for_skill(&[&skill_a, &skill_b], None);
        assert!(tools.is_empty());
    }

    #[test]
    fn multi_skill_any_disable_mcp_tools_disables_all() {
        // select_mcp_tools checks disable_mcp_tools: any skill disabling it returns empty
        let mcp_tool = tool("mcp_server_a_lookup");
        let mut skill_a = skill("alpha", "alpha skill");
        skill_a.mcp_servers.push("server_a".to_string());
        let mut skill_b = skill("beta", "beta skill");
        skill_b.disable_mcp_tools = true;
        let tools = super::select_mcp_tools(vec![mcp_tool], &[&skill_a, &skill_b], None);
        assert!(tools.is_empty());
    }

    #[test]
    fn multi_skill_merges_mcp_servers_deduplicated() {
        let mut skill_a = skill("alpha", "alpha skill");
        skill_a.mcp_servers.push("server_a".to_string());
        skill_a.mcp_servers.push("server_b".to_string());
        let mut skill_b = skill("beta", "beta skill");
        skill_b.mcp_servers.push("server_b".to_string());
        skill_b.mcp_servers.push("server_c".to_string());
        let servers = super::resolved_mcp_servers(&[&skill_a, &skill_b], None);
        assert_eq!(servers, vec!["server_a", "server_b", "server_c"]);
    }

}
