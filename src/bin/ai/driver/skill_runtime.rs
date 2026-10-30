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
use regex::Regex;
use rust_tools::cw::SkipSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

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

#[derive(Clone, Default)]
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

    /// Replaces the content of the first section carrying `label` in place,
    /// keeping its position so the rendered prompt only changes inside that
    /// section; appends when no such section exists yet.
    fn replace_labeled(&mut self, kind: ContextKind, label: &str, content: impl Into<String>) {
        let content = content.into();
        if content.trim().is_empty() {
            return;
        }
        let label = label.trim();
        if let Some(section) = self
            .sections
            .iter_mut()
            .find(|(existing_kind, existing_label, _)| {
                *existing_kind == kind && existing_label.as_deref() == Some(label)
            })
        {
            section.2 = content;
            return;
        }
        self.push_labeled(kind, label, content);
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
        // Keep a constant near-end pointer to the base behavior contracts,
        // outside the optional fact context. Do not define a second acceptance
        // standard here; recorded reminders still replay byte-identically.
        out.push_str("\n<system-reminder>\nBehavior check (mandatory, every turn): apply the base evidence and outcome-reporting rules. Resolve only gaps that materially affect delivery; otherwise reuse sufficient evidence and keep conclusions within its support.\n</system-reminder>");
        Some(out)
    }
}

const DEFAULT_TURN_TOOL_GROUPS: &[ToolGroup] = &[ToolGroup::Core];

/// Core tools, plus phone actions only for a serve turn whose client declared
/// the native bridge. Other clients never see `phone_action`.
fn default_turn_tool_groups() -> Vec<ToolGroup> {
    let mut groups = DEFAULT_TURN_TOOL_GROUPS.to_vec();
    if crate::ai::background::native_client() {
        groups.push(ToolGroup::Phone);
    }
    groups
}

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

#[derive(Default)]
pub(super) struct SkillTurnGuard {
    restore_agent_context: Option<(Vec<ToolDef>, usize)>,
    builder: SystemPromptBuilder,
    cached_system_prompt: Option<String>,
    cached_context_reminder: Option<Option<String>>,
    /// Currently active skill list (ordered; multiple skills are peers with no ranking).
    matched_skill_names: Vec<String>,
    /// Sections pushed after the guard was built (turn-preparation facts,
    /// critic and observer guidance), in push order. A mid-turn rebuild
    /// replays them onto the fresh guard so the turn keeps the same
    /// instructions and the same system-prompt bytes.
    turn_sections: Vec<(ContextKind, Option<String>, String)>,
    /// Directories whose target-scoped instruction documents the section
    /// currently renders (seeded from storage, updated on every load).
    scoped_instruction_dirs: Vec<PathBuf>,
    /// Whether `scoped_instruction_dirs` differs from what storage holds.
    scoped_instruction_dirs_changed: bool,
    /// Whether this guard rendered the full session-tools guidance.
    session_context_guidance: bool,
    /// Whether this guard renders the pure prompt, which carries no project
    /// context and therefore never accepts scoped instruction documents.
    pure_mode: bool,
    /// Whether the session had already made that guidance resident.
    session_context_guidance_was_sticky: bool,
}

/// Label of the system-prompt section holding target-scoped instructions.
const SCOPED_PROJECT_INSTRUCTIONS_LABEL: &str = "target_scoped_project_instructions";

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
                    pointer.push_str("Active skills at turn start (in activation order):\n");
                    for (i, name) in self.matched_skill_names.iter().enumerate() {
                        use std::fmt::Write;
                        let _ = writeln!(pointer, "  {}. {}", i + 1, name);
                    }
                    pointer.push_str(
                        "Treat their instructions in the system prompt's <skill_instructions> \
                         section as the primary behavior contract for this turn.",
                    );
                }
                parts.push(format!(
                    "<system-reminder>\n{}\n</system-reminder>",
                    pointer
                ));
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
        if !content.trim().is_empty() {
            self.turn_sections.push((kind, None, content.to_string()));
        }
    }

    pub(super) fn push_labeled_section(&mut self, kind: ContextKind, label: &str, content: &str) {
        self.cached_system_prompt = None;
        self.cached_context_reminder = None;
        self.builder.push_labeled(kind, label, content);
        if !content.trim().is_empty() {
            let label = label.trim();
            self.turn_sections.push((
                kind,
                (!label.is_empty()).then(|| label.to_string()),
                content.to_string(),
            ));
        }
    }

    /// Replays the sections `previous` received after it was built onto this
    /// freshly built guard, in their original order.
    fn inherit_turn_sections(&mut self, previous: &SkillTurnGuard) {
        for (kind, label, content) in &previous.turn_sections {
            match label {
                Some(label) => self.push_labeled_section(*kind, label, content),
                None => self.push_section(*kind, content),
            }
        }
    }

    pub(super) fn append_system_prompt(&mut self, extra: &str) {
        self.push_section(ContextKind::Fact, extra);
    }

    pub(super) fn push_scoped_project_instructions(
        &mut self,
        required_targets: &[PathBuf],
        observed_targets: &[PathBuf],
    ) -> bool {
        // Pure mode owns its whole prompt: project instruction documents would
        // make it larger at exactly the moment it is supposed to stay minimal.
        // Reporting "nothing missing" also stops the preflight from pausing.
        if self.pure_mode {
            return true;
        }
        if let Some((prompt, doc_dirs)) = build_scoped_project_instruction_prompt_with_priority(
            required_targets,
            observed_targets,
            &self.scoped_instruction_dirs,
        ) {
            // Record exactly what the section renders: a paused mutation's
            // documents outrank resident ones and may evict them when the
            // budget is full, and storage must not claim documents the
            // prompt no longer holds.
            if self.scoped_instruction_dirs != doc_dirs {
                self.scoped_instruction_dirs = doc_dirs;
                self.scoped_instruction_dirs_changed = true;
            }
            // Replaced in place rather than appended: the section keeps its
            // position, and it is rebuilt from the guard's own state instead
            // of being replayed through `turn_sections`.
            self.cached_system_prompt = None;
            self.cached_context_reminder = None;
            self.builder.replace_labeled(
                ContextKind::Policy,
                SCOPED_PROJECT_INSTRUCTIONS_LABEL,
                prompt,
            );
        }
        !scoped_project_instructions_missing(self.system_prompt(), required_targets)
    }

    /// Seeds the directories whose scoped instruction documents earlier turns
    /// of this session already loaded, and renders them. Documents stay
    /// resident instead of being dropped at every turn boundary and re-added
    /// on the next touch, so the model keeps seeing the rules it already
    /// followed and the system-prompt prefix stays stable. Only a paused
    /// mutation's documents can displace them when the budget is full.
    pub(super) fn seed_scoped_instruction_dirs(&mut self, dirs: Vec<PathBuf>) {
        if dirs.is_empty() {
            return;
        }
        let seeded = dirs;
        self.scoped_instruction_dirs = seeded.clone();
        self.push_scoped_project_instructions(&[], &[]);
        if !self.has_scoped_project_instructions() {
            // None of the stored documents can be rendered anymore (deleted,
            // outside the project, or over budget). Drop the stale directories
            // so storage stops claiming a section the prompt does not hold.
            self.scoped_instruction_dirs.clear();
            self.scoped_instruction_dirs_changed = true;
            return;
        }
        // Seeded directories came from storage, but rendering may drop entries
        // whose documents disappeared or no longer fit the budget. Report that
        // shrinkage so storage never claims documents the prompt does not hold.
        self.scoped_instruction_dirs_changed = self.scoped_instruction_dirs != seeded;
    }

    fn has_scoped_project_instructions(&self) -> bool {
        self.builder.sections.iter().any(|(kind, label, _)| {
            *kind == ContextKind::Policy
                && label.as_deref() == Some(SCOPED_PROJECT_INSTRUCTIONS_LABEL)
        })
    }

    /// Directories whose scoped instruction documents are resident, when this
    /// turn changed them from what storage holds.
    pub(super) fn changed_scoped_instruction_dirs(&self) -> Option<&[PathBuf]> {
        self.scoped_instruction_dirs_changed
            .then_some(self.scoped_instruction_dirs.as_slice())
    }

    /// Whether this turn rendered the full session-tools guidance for the
    /// first time in the session, so later turns should keep it resident.
    pub(super) fn session_context_guidance_newly_rendered(&self) -> bool {
        self.session_context_guidance && !self.session_context_guidance_was_sticky
    }

    /// Carries turn-scoped state from the guard this one replaces: sections
    /// pushed after the old guard was built, and the resident scoped
    /// instruction directories. Must run before scoped instructions are
    /// pushed onto this guard.
    pub(super) fn inherit_turn_state(&mut self, previous: &SkillTurnGuard) {
        self.inherit_turn_sections(previous);
        self.scoped_instruction_dirs = previous.scoped_instruction_dirs.clone();
        self.scoped_instruction_dirs_changed = previous.scoped_instruction_dirs_changed;
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

/// Install the turn's resolved tool set (already merged with runtime-enabled
/// tools by the caller) into the agent context, returning the previous tools
/// and iteration cap for restoration.
fn activate_skill_context(
    app: &mut App,
    all_tools: Vec<ToolDef>,
    max_iterations: usize,
) -> Option<(Vec<ToolDef>, usize)> {
    let mut restore = None;
    if let Some(ctx) = app.agent_context.as_mut() {
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
        // Builtin and MCP inputs can overlap (a skill may name a tool the
        // driver also injects), so the no-explicit branch still dedupes.
        return filter_subagent_hidden_tools(dedupe_tools_by_name(merged));
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
        return filter_subagent_hidden_tools(dedupe_tools_by_name(merged));
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

/// Native phone actions are injected only for a top-level turn whose serve
/// client declared the Android bridge; manifests cannot grant them to desktop
/// or subagent turns.
fn append_native_tools(mut tools: Vec<ToolDef>) -> Vec<ToolDef> {
    if !crate::ai::background::native_client() || should_hide_task_tools_for_subagent() {
        return tools;
    }
    for def in super::super::tools::tool_definitions_for_groups(&[ToolGroup::Phone]) {
        if !tools.iter().any(|tool| tool.function.name == def.function.name) {
            tools.push(def);
        }
    }
    tools
}

fn manifest_tool_definitions(tool_groups: &[String], tools: &[String]) -> Option<Vec<ToolDef>> {
    let mut defs: Vec<ToolDef> = Vec::new();
    // Case-insensitive dedup so a tool named both via a group and by an
    // explicit `tools:` entry rides along only once.
    let mut names_seen: Vec<String> = Vec::new();
    // Set when a manifest pinned a driver-injected control tool, which is
    // dropped below; it still counts as a declared tool set.
    let mut pinned_control_tool = false;
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
        // Tools named explicitly via `tools:` (below) are never dropped:
        // naming a tool pins it as resident. Core tools like apply_patch /
        // write_file are never marked hidden, so editing capability is
        // unaffected.
        for tool in super::super::tools::tool_definitions_for_groups(&groups) {
            if !super::super::tools::tool_defers_eager_load(&tool.function.name) {
                names_seen.push(tool.function.name.clone());
                defs.push(tool);
            }
        }
    }
    if !tools.is_empty() {
        // Explicitly named tools pin a tool into every turn even when its
        // group is also declared, bypassing the deferred-eager-load pruning
        // above (a manifest listing a name opts that tool in).
        for tool in super::super::tools::get_tool_definitions_by_names(tools) {
            // Driver control tools are injected by name per turn, never
            // granted by a manifest: pinning one would bypass the per-turn
            // ask policy and duplicate the driver's own injection.
            if super::super::tools::is_driver_injected_control_tool(&tool.function.name) {
                pinned_control_tool = true;
                continue;
            }
            if !names_seen
                .iter()
                .any(|n| n.eq_ignore_ascii_case(&tool.function.name))
            {
                names_seen.push(tool.function.name.clone());
                defs.push(tool);
            }
        }
    }
    if defs.is_empty() {
        // Pinning only driver-injected control tools is still a declared set:
        // returning None would make the caller fall back to the default
        // groups, widening the manifest's narrow pin to the whole Core set.
        if pinned_control_tool {
            return Some(ensure_required_baseline_tools(Vec::new()));
        }
        return None;
    }
    Some(ensure_required_baseline_tools(defs))
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
/// `is_executor_*` helpers this is mode-agnostic: any manifest — agent or
/// skill — carrying the executor group counts (no builtin agent declares it
/// today; only opt-in skills do). Drives the "available on
/// demand via enable_tools" hint for turns whose resident tool set had the
/// heavy execution primitives deferred out by `manifest_tool_definitions`, so
/// read-only agents (plan / explore) never get irrelevant process/IPC hints.
fn declares_hidden_group(skills: &[&SkillManifest], active_agent: Option<&AgentManifest>) -> bool {
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
        return append_native_tools(filter_subagent_hidden_tools(Vec::new()));
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
            return append_native_tools(filter_subagent_hidden_tools(tool_defs));
        }
    }
    if let Some(agent) = active_agent
        && let Some(tool_defs) = manifest_tool_definitions(&agent.tool_groups, &agent.tools)
    {
        return append_native_tools(filter_subagent_hidden_tools(tool_defs));
    }
    filter_subagent_hidden_tools(super::super::tools::tool_definitions_for_groups(
        &default_turn_tool_groups(),
    ))
}

/// Whether the driver injects the `request_user_input` control tool for this
/// turn. Present whenever a human is attached to an interactive turn: skills
/// need it for the cross-turn continuation handoff, and ordinary turns get the
/// explicit ask protocol (tool metadata) instead of a prose guidance block.
/// Goal mode demands autonomous progress and background turns have no attached
/// terminal, and subagent turns have no attached user, so all three
/// deliberately disable asking and never inject the tool.
fn should_inject_request_user_input(
    skills: &[&SkillManifest],
    goal_mode: &Option<String>,
    is_background: bool,
) -> bool {
    crate::ai::driver::runtime_ctx::current_subagent_depth() == 0
        && (!skills.is_empty() || (goal_mode.is_none() && !is_background))
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
    rust_tools::sortw::stable_sort_by(&mut hidden, |a, b| a.cmp(b));
    hidden.dedup();

    render_hidden_tool_catalog(
        "Configured MCP tools are available but not loaded in this turn.\n\
         If the task needs an external system or MCP-backed capability, discover and enable matching \
         `mcp_*` tools via `enable_tools` first.",
        &hidden,
    )
}

fn build_hidden_execution_primitive_catalog(
    available_tools: &Box<SkipSet<String>>,
) -> Option<String> {
    // Deferred heavy-execution primitives (process / IPC / shared-memory / env)
    // are not loaded into every turn by default. List the registered-but-not-
    // loaded names here so the model stays aware they exist and can enable them
    // on demand via `enable_tools`. The shared renderer handles truncation and
    // stays silent when everything is already loaded.
    let mut hidden: Vec<String> = super::super::tools::deferred_eager_load_tool_summaries()
        .into_iter()
        .map(|(name, _desc)| name)
        .filter(|name| !available_tools.contains_str(name))
        .collect();
    rust_tools::sortw::stable_sort_by(&mut hidden, |a, b| a.cmp(b));
    hidden.dedup();

    render_hidden_tool_catalog(
        "Process / IPC / shared-memory primitives are available but not loaded this turn.\n\
         For multi-process orchestration, background daemons, cross-process IPC, shared memory, \
         or per-process env/working-dir control, enable the needed tools via `enable_tools`.",
        &hidden,
    )
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

    render_hidden_tool_catalog(
        "Subagent orchestration tools are available but not loaded in this turn.\n\
         When the task splits into multiple independent branches that would benefit from \
         subagent parallelism (broad discovery, cross-module mapping, independent verification, \
         or concurrent research), enable the needed tools via `enable_tools`.",
        &hidden,
    )
}

/// Render names in caller-supplied order: MCP and execution catalogs sort and
/// deduplicate first, while the task catalog deliberately keeps family order.
fn render_hidden_tool_catalog(
    introduction: &str,
    hidden_names: &[impl AsRef<str>],
) -> Option<String> {
    if hidden_names.is_empty() {
        return None;
    }

    const MAX_DISPLAY: usize = 8;
    let displayed = hidden_names
        .iter()
        .take(MAX_DISPLAY)
        .map(|name| format!("`{}`", name.as_ref()))
        .collect::<Vec<_>>()
        .join(", ");
    let remaining = hidden_names.len().saturating_sub(MAX_DISPLAY);

    let mut out = format!("{introduction}\nAvailable: {displayed}");
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

// Keep project and skill precedence declarations aligned on the protected rules.
const NON_OVERRIDABLE_PROMPT_CONSTRAINTS: &str =
    "correctness guardrails (including git-safety rules), intellectual honesty, safety redlines";

fn build_project_instruction_prompt() -> Option<String> {
    let docs = load_project_instruction_docs();
    if docs.is_empty() {
        return None;
    }

    let mut out = format!(
        "- The current working directory provides project-specific instruction documents.\n\
         - Follow these repo-local constraints and preferences; only {NON_OVERRIDABLE_PROMPT_CONSTRAINTS}, and explicit user instructions take precedence over them, and they override agent, skill, and generic guidance.\n",
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

#[cfg(test)]
fn build_scoped_project_instruction_prompt(targets: &[PathBuf]) -> Option<String> {
    build_scoped_project_instruction_prompt_with_priority(targets, &[], &[]).map(|(prompt, _)| prompt)
}

/// Renders the target-scoped instruction section. Also returns the directories
/// holding the rendered documents, so the caller can keep them resident for
/// the rest of the session.
fn build_scoped_project_instruction_prompt_with_priority(
    required_targets: &[PathBuf],
    observed_targets: &[PathBuf],
    carried_dirs: &[PathBuf],
) -> Option<(String, Vec<PathBuf>)> {
    let docs = load_scoped_project_instruction_docs_for_target_priority(
        required_targets,
        observed_targets,
        carried_dirs,
    );
    if docs.is_empty() {
        return None;
    }
    let mut out = String::from(
        "- These documents apply to files already touched in this session.\n\
         - A rule from a deeper directory is more specific and overrides a conflicting general project rule.\n",
    );
    let mut dirs: Vec<PathBuf> = Vec::new();
    for doc in docs {
        out.push_str(&format!(
            "\n<instructions path=\"{}\">\n{}\n</instructions>\n",
            escape_xml_attr(&doc.path),
            doc.content.trim()
        ));
        if let Some(dir) = Path::new(&doc.path).parent().map(Path::to_path_buf)
            && !dirs.contains(&dir)
        {
            dirs.push(dir);
        }
    }
    Some((out, dirs))
}

/// Marker carried by the authored pure prompt. Scoped-instruction handling
/// recognises pure mode from the rendered text so it needs no extra plumbing
/// through the turn loop.
const PURE_MODE_PROMPT_MARKER: &str = "<pure_mode>";

pub(super) fn scoped_project_instructions_missing(
    system_prompt: &str,
    targets: &[PathBuf],
) -> bool {
    // Pure mode renders no project context by design, so nothing can be
    // missing. Answering "not missing" keeps the mutation preflight from
    // pausing the turn to inject instructions the mode deliberately omits.
    if system_prompt.contains(PURE_MODE_PROMPT_MARKER) {
        return false;
    }
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

/// Session context: grounds the model in its current session id and steers
/// session inspection through the standard `session` tools (`session_list`
/// to discover ids, `session_read` / `session_search` for content; loaded
/// via `enable_tools` when not visible). The model is read-only: no writing,
/// modifying, or deleting session data. The raw storage layout (sessions
/// root, `<id>.sqlite`, assets dirs) is deliberately not advertised so the
/// model cannot bypass the tools with direct sqlite or filesystem reads;
/// `session_search` surfaces asset paths at runtime. Injected when the request
/// looks session-related (`session_issue_likely`), references a concrete session
/// id (`session_id_referenced`), or prior session guidance is already resident.
/// Otherwise the opaque id is omitted because it has no actionable meaning.
fn session_context_prompt(session_id: &str) -> String {
    format!(include_str!("system_prompts/session_context.md"), session_id)
}

/// Relevance gate for the session-debugging walkthrough: decides whether the
/// turn injects the session-context guidance (session id + standard session
/// tools). Matching is deliberately broad
/// substring — a missed debug session costs more than the ~500 carried
/// characters of an occasional over-injection (a general "history of X"
/// knowledge question also matches `history`; accepted). Chinese is matched
/// literally here alongside English; this keyword list is the one place where
/// non-English prompt text is intentional.
fn session_issue_likely(question: &str) -> bool {
    let lowered = question.to_lowercase();
    [
        // "session" also covers sessionid, session_id and /sessions paths.
        "session", "history", "resume", "attach", "background", "会话", "历史",
        "恢复", "重连",
    ]
    .iter()
    .any(|kw| lowered.contains(kw))
}

/// Deterministic session-id detection: a UUID-shaped token in the user question
/// is an unambiguous reference to a concrete runtime session. Unlike the
/// keyword gate above, this needs no vocabulary, so a bare id like
/// `ddb7e6be-7076-43b3-898f-8370e64312da` (with no "session"/"会话" words)
/// still routes the turn to the session tools. Session store ids are
/// 1-128 ASCII letters/digits/`-`/`_`; the dominant real-world form is a UUID,
/// which is what this matches. Non-UUID ids (e.g. `test-session-<hex>`) are
/// still caught by `session_issue_likely` via the "session" keyword.
fn session_id_referenced(question: &str) -> bool {
    static SESSION_ID_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
            .expect("session-id regex must compile")
    });
    SESSION_ID_RE.is_match(question)
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
            header.push_str("No active skill overrides another.");
            header
        };
        s.push_str("\n\n<skill_instructions>\n");
        s.push_str(skill_text.trim());
        s.push_str("\n</skill_instructions>");
        // Precedence names only the sections that must never yield. The
        // `<policy>` group also carries tool heuristics (knowledge retrieval,
        // discovery) that a skill contract may legitimately override, so it is
        // not named wholesale; project-local instructions are named instead,
        // and their own preamble states the same ordering.
        if let Some(agent_text) = &agent_extra {
            s.push_str("\n\n<agent_instructions>\n");
            s.push_str(agent_text.trim());
            s.push_str("\n</agent_instructions>");
            s.push_str(&format!("\n\nEnforcement: skill instructions override agent instructions when they differ. Use agent instructions only for capabilities, workflow, and defaults not covered by the active skill. Neither skill nor agent instructions override the {NON_OVERRIDABLE_PROMPT_CONSTRAINTS}, or project-local instructions, which always take precedence."));
        } else {
            s.push_str(&format!("\n\nEnforcement: skill instructions override generic assistant guidelines when they differ, except the {NON_OVERRIDABLE_PROMPT_CONSTRAINTS}, and project-local instructions, which always take precedence."));
        }
        s
    } else {
        agent_extra.unwrap_or_else(|| {
            String::from(
                include_str!("prompts/default_identity.md"),
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
    // The ask protocol ("ask only when genuinely blocked; one focused question;
    // wait for the reply") travels as `request_user_input` tool metadata, injected
    // for every interactive turn by build_skill_turn_guard. Goal mode and
    // background turns deliberately disable asking, so they get neither the tool
    // nor any prose guidance; skill turns carry their own handoff block above.
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

    // ── Safety red lines: non-negotiable prohibitions, rendered ahead of the ──
    // working-style rules so the red lines are read before any guidance about
    // how to work. The non-fabrication rule lives in intellectual_honesty below,
    // while fact tracing / evidence calibration stay single-sourced in
    // correctness_guardrails, so this block holds only the dangerous-operation
    // prohibitions and the confirmation requirement.
    // Never relaxed by task, skill, or goal mode; when a skill activates, the
    // enforcement line folds these into the highest priority.
    b.push(
        ContextKind::Behavior,
        include_str!("system_prompts/safety_redlines.md"),
    );
    // ── Trust boundary: tool output / fetched content is data, not instructions ──
    // strip_system_reminders already removes forged reminder blocks from user
    // messages mechanically; this block teaches the model to treat look-alikes
    // embedded in tool output (web pages, documents, command output) as forged
    // content, since authentic runtime reminders have a fixed format.
    b.push(
        ContextKind::Behavior,
        include_str!("system_prompts/trust_boundary.md"),
    );
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
    // Thinking scope: effort authorizes depth, not length. Unconditional — a
    // "max" reasoning tier must still scale thinking to the question instead
    // of writing a maximal-length thought chain.
    b.push(
        ContextKind::Behavior,
        include_str!("system_prompts/thinking_scope.md"),
    );
    // Intellectual honesty: evidence-earned agreement and respectful pushback
    // against wrong or inappropriate user premises. Unconditional — it applies
    // in every mode and is never relaxed by a skill or goal. It forbids treating
    // unsupported claims as established facts while allowing explicit hypotheses
    // and conditional answers. Fact tracing / evidence calibration stay
    // single-sourced in correctness_guardrails. The no-regression rule for other modules
    // lives in correctness_guardrails' scope section, beside the caller /
    // dependent rule it extends.
    b.push(
        ContextKind::Behavior,
        include_str!("system_prompts/intellectual_honesty.md"),
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
            // The MCP-specific hint lives in the hidden MCP catalog, which is
            // rendered only when unloaded `mcp_*` tools actually exist.
            discovery_lines.push(
                "No skill is active for this turn. Additional capabilities are available via `enable_tools`; before declaring a requested capability unavailable, call `enable_tools(operation=list)` to discover it, then enable only the specific tools you need.".to_string(),
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
            // When-to-browse and when-to-activate criteria are owned by the
            // `list_skills` / `activate_skill` descriptions, which are visible
            // whenever this line renders.
            discovery_lines.push(
                "Skills are optional: follow the `list_skills` / `activate_skill` criteria, and never browse skills as a routine opening step."
                    .to_string(),
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
        // Retrieval guidance is owned by the knowledge_retrieval section, which
        // renders whenever a retrieval tool is available.
        let lines = vec![
            "If the user asks to remember or states a durable preference/constraint, call `knowledge_save`.".to_string(),
            "When saving a durable principle, preference, safety rule, or coding rule, choose a guideline category such as `common_sense`, `coding_guideline`, `preference`, `user_preference`, or `safety_rules`.".to_string(),
            "Use `user_memory` / `project_info` / `architecture` / `decision_log` for factual knowledge.".to_string(),
            "Save each distinct durable fact at most once per turn. Do not save temporary work notes, raw tool output, or speculative conclusions as global knowledge.".to_string(),
        ];
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
            // Step-status cadence and semantics are owned by the `plan_update`
            // description and the plan tool's first-use guidance.
            lines.push("Treat the plan as a living roadmap: when findings, changed requirements, or a dead end reshape the task, call `plan` again instead of drifting; the latest plan is preserved in full as the task anchor while older versions may be summarized.".to_string());
            if has_tool(available_tools, "task_spawn") {
                // The delegation decision is owned by the async_subagent_orchestration
                // section (rendered under the same `task_spawn` gate below); this line
                // defers to it and keeps only the plan-schema field semantics. Per-shape
                // orchestration mechanics come from the plan-time footer in
                // `plan_state::render::delegation_guidance`.
                lines.push("When planning, decide delegation per the async_subagent_orchestration guidance below. A delegated step runs in a subagent with a clean, focused context and the parent reviews its result; `parallelizable: true` means no dependency on earlier steps (run those concurrently via `task_spawn`); delegated steps without it run one at a time via the synchronous `task`.".to_string());
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

    if has_tool(available_tools, "task_spawn") || has_tool(available_tools, "task_status") {
        // Cross-tool delegation decisions only. Tool-local semantics (task vs
        // task_spawn routing, model/inherit defaults, wait timeout/wait_policy,
        // task_integrate disposition, task_cancel collect-later) live in the
        // always-visible tool schemas and runtime follow-ups, so repeating them
        // here would add resident tokens without adding constraints. Final-response
        // gates check outstanding tasks and unintegrated evidence with bounded retries.
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
            lines.push("Once you identify multiple qualifying subtasks with no data dependency, spawn ALL of them in one `task_spawn` call (1-8 tasks) so preflight, dispatch, and returned task ids preserve input order; continue every independent parent-side step while they run — never spawn-wait-spawn-wait serially.".to_string());
        }
        if has_tool(available_tools, "task_status") {
            lines.push("Use `task_status` when unresolved task state matters to the next decision, not as a mandatory finishing step. Reuse delivered results; never substitute a status query for reviewing and explicitly integrating evidence. Never silently drop a spawned task; runtime final-response gates provide bounded recovery, not a substitute for closing task work.".to_string());
        }
        push_tool_guidance_section(
            &mut b,
            ContextKind::Behavior,
            "async_subagent_orchestration",
            lines,
        );
    }

    if has_tool(available_tools, "knowledge_search") || has_tool(available_tools, "knowledge_list")
    {
        let mut lines = Vec::new();
        let search_tools = available_tool_names_in_order(available_tools, &["knowledge_search"]);
        if !search_tools.is_empty() {
            lines.push(format!(
                "Only when the user explicitly asks about remembered or saved knowledge, search with {}.",
                format_tool_names(&search_tools)
            ));
            lines.push(
                "Exception: when the current task concerns past project decisions, preferences, or architecture, one proactive search using task keywords is allowed (entries distilled from session archives carry a session-distill source tag)."
                    .to_string(),
            );
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
        // Patch-size limits, envelope syntax, and multi-hunk mechanics are owned
        // by the `apply_patch` schema and first-use guidance; this section keeps
        // only the cross-tool routing between write_file, execute_command, and
        // apply_patch.
        let mut lines = Vec::new();
        lines.push(
            "To run a script, dump intermediate data, or write a test fixture, create it with `write_file(temp=true)` first, then run it with `execute_command`. Prefer this over inline `python -c '...'` whenever the code is more than a few lines or you need to inspect/edit the file.".to_string(),
        );
        lines.push(
            "Do NOT use `execute_command` to create temp files (e.g. `echo > /tmp/foo`, `python -c '...' > out.json`) — files created outside `write_file(temp=true)` will accumulate. `execute_command` may `rm` only inside the session temp directory; that is a command-policy blacklist, not a filesystem sandbox: allowed commands run directly against the real workspace.".to_string(),
        );
        lines.push(
            "Save content downloaded from the network in the session temp directory, not the current working directory. Reserve the destination with `write_file(temp=true)`, then pass the returned path to the downloader; use a project or user-selected path only when the user explicitly requests that destination.".to_string(),
        );
        if has_tool(available_tools, "apply_patch") {
            lines.push(
                "Do NOT use `write_file(temp=true)` on existing project files; reserve `write_file` without `temp` for genuine full rewrites (localized edits go through `apply_patch`).".to_string(),
            );
            lines.push(
                "When one file needs several localized edits, read the span once. Make ONE `apply_patch` call with multiple `@@` hunks only when every hunk has a unique anchor (distinct context); for structurally similar blocks, patch one at a time with a distinctive anchor line such as a function name or comment."
                    .to_string(),
            );
            lines.push(
                "To remove an existing project/source/config file, including a git-tracked file, use `apply_patch` with a `*** Delete File: <path>` section.".to_string(),
            );
        } else {
            lines.push(
                "When modifying an existing project file, do NOT use `write_file` with `temp=true`; use `write_file` without `temp` only when a full rewrite is genuinely necessary.".to_string(),
            );
        }
        push_tool_guidance_section(&mut b, ContextKind::Behavior, "temporary_files", lines);
    }

    b
}

/// Pure mode renders the pure agent's authored baseline plus the runtime
/// environment. The baseline already owns identity, non-fabrication,
/// tool-evidence, safety, and response style, so re-adding those blocks here
/// would duplicate contracts. Only the environment block is added because it
/// is per-process fact that no static prompt can carry.
fn build_pure_system_prompt(agent: &AgentManifest) -> SystemPromptBuilder {
    let mut b = SystemPromptBuilder::new();
    // Behavior, not Identity: the authored block is a whole behaviour contract
    // (identity, honesty, tool evidence, safety, style), and rendering it into
    // the identity group would leave that group carrying more than identity.
    b.push(ContextKind::Behavior, agent.build_system_prompt());
    b.push_labeled(
        ContextKind::Behavior,
        "execution_environment",
        runtime_environment_prompt(),
    );
    b
}

/// Chooses the prompt builder for one turn. A pure agent renders only its own
/// baseline: the skill and runtime context inputs are ignored because none of
/// their guidance renders.
fn prompt_builder_for_turn(
    active_agent: Option<&AgentManifest>,
    skills: &[&SkillManifest],
    available_tools: &Box<SkipSet<String>>,
    ctx: &PromptContext,
) -> SystemPromptBuilder {
    match active_agent {
        Some(agent) if agent.pure => build_pure_system_prompt(agent),
        _ => build_system_prompt(active_agent, skills, available_tools, ctx),
    }
}

fn build_skill_turn_guard(
    app: &mut App,
    mcp_client: &McpClient,
    skills: &[&SkillManifest],
    question: &str,
) -> SkillTurnGuard {
    let all_mcp_tools = mcp_client.get_all_tools();
    super::super::tools::enable_tools::set_available_mcp_tools(all_mcp_tools.clone());
    let matched_skill_names: Vec<String> = skills.iter().map(|s| s.name.clone()).collect();
    let active_agent = app.current_agent_manifest.clone();
    let executor_active =
        is_executor_skill(skills) || active_agent.as_ref().is_some_and(is_executor_agent);
    // Record whether the current turn declares a hidden-gating group into
    // enable_tools' turn-level state, so the heavy execution primitives stay
    // out of the enable catalog for agents that do not declare one (same
    // source of truth as the hidden-catalog hint below, so the two can never
    // disagree).
    let hidden_group_declared = declares_hidden_group(skills, active_agent.as_ref());
    super::super::tools::enable_tools::set_hidden_group_declared(hidden_group_declared);

    let mut builtin_tools = builtin_tools_for_skill(skills, active_agent.as_ref());
    // Externally downloaded skills cannot pre-declare this runtime's ask/handoff
    // protocol via manifest tool lists; the driver injects request_user_input by
    // name for every interactive turn. Skill turns need the cross-turn
    // continuation handoff; ordinary turns get the explicit ask protocol in place
    // of prose-only guidance. Goal mode, background, and subagent turns
    // deliberately disable asking (subagent turns have no attached user), so
    // none of them injects the tool.
    if should_inject_request_user_input(skills, &app.goal_mode, app.cli.background) {
        builtin_tools.extend(crate::ai::tools::get_tool_definitions_by_names(&[
            "request_user_input".to_string(),
        ]));
    }
    // Deterministic session-tool routing: a user message referencing a concrete
    // session id (UUID-shaped) auto-loads the `session` group (inspection plus talk mailbox) into the
    // resident tool set for top-level turns, so the model never has to remember
    // `enable_tools` before it can answer a session question (a real failure
    // mode: it probes the filesystem instead). Subagent and disabled-builtin
    // turns keep the lazy behavior; the keyword gate below still injects the
    // guidance for vaguer session/history wording.
    if crate::ai::driver::runtime_ctx::current_subagent_depth() == 0
        && !builtin_tools.is_empty()
        && session_id_referenced(question)
    {
        // Only append tools that are not already resident (a skill manifest may
        // have declared the session group).
        let known: Box<SkipSet<String>> = builtin_tools
            .iter()
            .map(|tool| tool.function.name.clone())
            .collect();
        let missing: Vec<String> = ["session_list", "session_read", "session_search", "talk_send", "talk_read", "discuss_start", "discuss_collect", "discuss_broadcast", "discuss_end", "discuss_status", "task_start", "task_assign", "task_collect", "task_close", "task_room_status"]
            .iter()
            .filter(|name| !known.contains_str(*name))
            .map(|s| s.to_string())
            .collect();
        if !missing.is_empty() {
            builtin_tools.extend(crate::ai::tools::get_tool_definitions_by_names(&missing));
        }
    }
    let mcp_tools = select_mcp_tools(all_mcp_tools.clone(), skills, active_agent.as_ref());
    // Prompt gating must see exactly the tool set this request will carry,
    // including tools enabled via `enable_tools` in earlier turns; otherwise
    // the hidden-tool catalogs keep advertising loaded tools as "not loaded"
    // and tool-gated guidance sections never render for them.
    let loaded_tools: Vec<ToolDef> = match app.agent_context.as_ref() {
        Some(ctx) => merge_with_runtime_enabled_tools(builtin_tools, mcp_tools, &ctx.tools),
        None => builtin_tools.into_iter().chain(mcp_tools).collect(),
    };
    let available_tools = available_tool_names(&loaded_tools, &[]);
    // Pure mode is a prompt-construction decision only: tool selection, session
    // state, and persistence are unchanged. It is driven by the active agent's
    // `pure` flag, so switching agents (CLI, /agent, task) switches rendering.
    let pure_mode = active_agent.as_ref().is_some_and(|agent| agent.pure);
    let ctx = PromptContext {
        goal_mode: app.goal_mode.clone(),
        is_background: app.cli.background,
    };
    let mut builder = prompt_builder_for_turn(
        active_agent.as_ref(),
        skills,
        &available_tools,
        &ctx,
    );
    // Everything appended below steers a full agent: tool catalogs, project and
    // session context, skill history, persona, and scoped instructions. Pure mode
    // keeps the prompt to the safety/protocol core, so none of them render.
    if !pure_mode
        && has_tool(&available_tools, "enable_tools")
        && let Some(catalog) = build_hidden_mcp_tool_catalog(&all_mcp_tools, &loaded_tools)
    {
        builder.push(ContextKind::Capability, catalog);
    }
    // For executor agents the heavy execution primitives are deferred out of
    // the resident set by manifest_tool_definitions; add an on-demand enable
    // hint here so the model can still sense them (read-only agents never
    // declare the group, so nothing is injected).
    if !pure_mode
        && has_tool(&available_tools, "enable_tools")
        && hidden_group_declared
        && let Some(catalog) = build_hidden_execution_primitive_catalog(&available_tools)
    {
        builder.push(ContextKind::Capability, catalog);
    }
    // Task-family tools are lazy for every top-level turn (not just executor
    // agents); advertise them so the model loads subagent tools exactly when the
    // task warrants parallelism. Gated internally on top-level depth.
    if !pure_mode
        && has_tool(&available_tools, "enable_tools")
        && let Some(catalog) = build_hidden_task_tool_catalog(&available_tools)
    {
        builder.push(ContextKind::Capability, catalog);
    }
    if !pure_mode {
        push_project_context(&mut builder);
    }
    let residency = history::read_prompt_residency_sqlite(&app.session_history_file)
        .ok()
        .flatten()
        .unwrap_or_default();
    // Session debugging guidance is low-frequency knowledge. Keep the opaque id
    // out of unrelated turns; inject it with the session-tools walkthrough once
    // the request looks session-related. After that the guidance stays resident
    // for the rest of the session, so later turns keep the system-prompt prefix.
    let session_context_guidance = !pure_mode
        && (residency.session_context_guidance
            || session_issue_likely(question)
            || session_id_referenced(question));
    if session_context_guidance {
        builder.push_labeled(
            ContextKind::Behavior,
            "session_context",
            session_context_prompt(&app.session_id),
        );
    }
    if !pure_mode
        && let Ok(events) = history::read_skill_activation_events_sqlite(&app.session_history_file)
        && let Some(reminder) = build_skill_activation_history_reminder(&events)
    {
        builder.push_labeled(
            ContextKind::Fact,
            "Session Skill Activation History",
            reminder,
        );
    }
    if !pure_mode && !app.active_persona.is_default() {
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
    let restore_agent_context = activate_skill_context(app, loaded_tools, max_iterations);

    let mut guard = SkillTurnGuard {
        restore_agent_context,
        builder,
        matched_skill_names,
        pure_mode,
        session_context_guidance,
        session_context_guidance_was_sticky: residency.session_context_guidance,
        ..SkillTurnGuard::default()
    };
    // Rendered last so the section is the final policy entry, the same position
    // a mid-turn load appends it at.
    if !pure_mode {
        guard.seed_scoped_instruction_dirs(residency.scoped_instruction_dirs);
    }
    guard
}

pub(super) fn rebuild_skill_turn_with_existing_selection(
    app: &mut App,
    mcp_client: &McpClient,
    skill_manifests: &[SkillManifest],
    question: &str,
    preferred_skill_names: &[String],
) -> SkillTurnGuard {
    // For iteration > 1, keep the previous turn's skills by name only; no more text-similarity re-routing.
    // If the model needs to switch, it can request explicitly via activate_skill.
    let skills: Vec<&SkillManifest> = preferred_skill_names
        .iter()
        .filter_map(|name| skill_manifests.iter().find(|s| &s.name == name))
        .collect();
    build_skill_turn_guard(app, mcp_client, &skills, question)
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
    question: &str,
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
    let mut guard = build_skill_turn_guard(app, mcp_client, &skills, question);
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
    let mut guard = build_skill_turn_guard(app, mcp_client, skills, question);
    guard.matched_skill_names = Vec::new();
    Ok(guard)
}

#[cfg(test)]
mod pure_prompt_tests {
    use super::*;
    use serde_json::json;

    /// Every tool that gates a guidance section, so a mode's omissions are the
    /// mode's own decision rather than a missing tool precondition.
    const FULL_TOOL_SET: &[&str] = &[
        "request_user_input",
        "enable_tools",
        "list_skills",
        "activate_skill",
        "knowledge_save",
        "knowledge_search",
        "knowledge_list",
        "plan",
        "plan_update",
        "task_spawn",
        "task_status",
        "write_file",
        "apply_patch",
        "execute_command",
        "search_overflow",
    ];

    fn tools(names: &[&str]) -> Box<SkipSet<String>> {
        let mut available = SkipSet::new(names.len().max(1));
        for name in names {
            available.insert((*name).to_string());
        }
        Box::new(available)
    }

    fn skill_with_prompt(prompt: &str) -> SkillManifest {
        SkillManifest {
            name: "pure-sample-skill".to_string(),
            version: "1.0.0".to_string(),
            description: String::new(),
            author: None,
            tools: Vec::new(),
            tool_groups: Vec::new(),
            mcp_servers: Vec::new(),
            disable_builtin_tools: false,
            disable_mcp_tools: false,
            prompt: prompt.to_string(),
            system_prompt: None,
            priority: 0,
            excludes: Vec::new(),
            parent: None,
            source_path: None,
            resource_path: None,
        }
    }

    fn render_turn(
        agent: Option<&AgentManifest>,
        skills: &[&SkillManifest],
        available: &Box<SkipSet<String>>,
        ctx: &PromptContext,
    ) -> String {
        prompt_builder_for_turn(agent, skills, available, ctx).render_system_prompt()
    }

    fn pure_agent_manifest() -> AgentManifest {
        crate::ai::agents::load_all_agents()
            .into_iter()
            .find(|agent| agent.name == "pure")
            .expect("the pure agent is registered")
    }

    #[test]
    fn pure_mode_renders_only_the_authored_baseline_and_environment() {
        let pure_agent = pure_agent_manifest();
        let prompt = render_turn(
            Some(&pure_agent),
            &[],
            &tools(FULL_TOOL_SET),
            &PromptContext::default(),
        );

        for kept in [
            "You are a general-purpose AI assistant",
            "<honesty>",
            "<tool_evidence>",
            "<safety>",
            "<execution_environment>",
        ] {
            assert!(prompt.contains(kept), "pure mode must keep {kept}:\n{prompt}");
        }

        // Guidance that steers a general agent rather than preventing a wrong
        // answer, so pure mode drops it.
        for dropped in [
            "<tool_usage>",
            "<correctness_guardrails>",
            "<thinking_scope>",
            "<task_convergence>",
            "<tool_result_evidence>",
            "<compressed_context_recovery>",
            "<scope_discipline>",
            "<autonomous_execution>",
            "<tool_discovery>",
            "<knowledge_save>",
            "<knowledge_retrieval>",
            "<planning_subprocess_execution>",
            "<async_subagent_orchestration>",
            "<temporary_files>",
            "<interactive_skill_handoff>",
            "<intellectual_honesty>",
            "<safety_redlines>",
            "<trust_boundary>",
        ] {
            assert!(
                !prompt.contains(dropped),
                "pure mode must drop {dropped}:\n{prompt}"
            );
        }
    }

    #[test]
    fn standard_mode_keeps_the_full_guidance_catalog() {
        let prompt = render_turn(
            None,
            &[],
            &tools(FULL_TOOL_SET),
            &PromptContext::default(),
        );

        for kept in [
            "<response_style>",
            "<tool_usage>",
            "<correctness_guardrails>",
            "<task_convergence>",
            "<scope_discipline>",
            "<autonomous_execution>",
            "<tool_discovery>",
            "<planning_subprocess_execution>",
            "<async_subagent_orchestration>",
            "<temporary_files>",
        ] {
            assert!(
                prompt.contains(kept),
                "standard mode must keep {kept}:\n{prompt}"
            );
        }
    }

    #[test]
    fn pure_agent_renders_baseline_and_skips_skill_instructions() {
        let pure_agent = pure_agent_manifest();
        let skill = skill_with_prompt("SKILL_SCOPE_MARKER");
        let available = tools(FULL_TOOL_SET);
        let skills = [&skill];

        let pure = render_turn(
            Some(&pure_agent),
            &skills,
            &available,
            &PromptContext::default(),
        );
        // The pure agent's own body is the whole baseline, so it stays.
        assert!(
            pure.contains("You are a general-purpose AI assistant"),
            "pure mode must keep the authored baseline:\n{pure}"
        );
        for dropped in [
            "SKILL_SCOPE_MARKER",
            "<skill_instructions>",
            "<agent_instructions>",
        ] {
            assert!(
                !pure.contains(dropped),
                "pure mode must skip {dropped}:\n{pure}"
            );
        }

        // A non-pure agent keeps the standard composition: its own
        // instructions and skill guidance both render.
        let agent: AgentManifest = serde_json::from_value(json!({
            "name": "pure-sample-agent",
            "description": "",
            "prompt": "AGENT_SCOPE_MARKER",
        }))
        .expect("agent manifest deserializes from its documented fields");
        let standard = render_turn(
            Some(&agent),
            &skills,
            &available,
            &PromptContext::default(),
        );
        for kept in [
            "AGENT_SCOPE_MARKER",
            "SKILL_SCOPE_MARKER",
            "<skill_instructions>",
        ] {
            assert!(
                standard.contains(kept),
                "standard mode must keep {kept}:\n{standard}"
            );
        }
    }

    #[test]
    fn pure_mode_is_a_small_fraction_of_the_standard_prompt() {
        let pure_agent = pure_agent_manifest();
        let pure = build_pure_system_prompt(&pure_agent).render_system_prompt();
        let standard = render_turn(
            None,
            &[],
            &tools(FULL_TOOL_SET),
            &PromptContext::default(),
        );

        assert!(!pure.is_empty());
        assert!(
            pure.len() * 3 < standard.len(),
            "pure prompt must stay a small fraction of the standard prompt; \
             pure={} bytes, standard={} bytes",
            pure.len(),
            standard.len()
        );
    }

    #[test]
    fn pure_mode_ignores_scoped_project_instructions() {
        // A file under the project's scoped-instruction directory: the loader
        // only collects documents for targets below the effective working
        // directory, and `driver/AGENTS.md` supplies a real document here.
        let target = std::env::current_dir()
            .expect("test working directory")
            .join("src/bin/ai/driver/skill_runtime.rs");
        let targets = std::slice::from_ref(&target);

        let standard = render_turn(
            None,
            &[],
            &tools(&["apply_patch"]),
            &PromptContext::default(),
        );
        assert!(
            scoped_project_instructions_missing(&standard, targets),
            "the standard prompt starts without the target's scoped documents"
        );

        let pure_agent = pure_agent_manifest();
        let pure = build_pure_system_prompt(&pure_agent).render_system_prompt();
        assert!(
            !scoped_project_instructions_missing(&pure, targets),
            "pure mode must never report scoped instructions as missing, \
             otherwise the mutation preflight pauses the turn"
        );

        let mut guard = SkillTurnGuard {
            pure_mode: true,
            ..SkillTurnGuard::default()
        };
        assert!(
            guard.push_scoped_project_instructions(targets, &[]),
            "a pure guard reports the target as satisfied instead of pausing"
        );
        assert!(
            !guard.system_prompt().contains("<instructions path="),
            "pure mode must not render project instruction documents"
        );
    }

    /// FNV-1a, so the snapshot is stable across toolchains and dependency
    /// versions instead of depending on `Hash`'s randomised state.
    fn stable_hash(bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    /// Drops the environment block: it carries the host OS, shell, and working
    /// directory, which differ per machine and are not part of the profile.
    fn without_environment_block(prompt: &str) -> String {
        const OPEN: &str = "<execution_environment>";
        const CLOSE: &str = "</execution_environment>";
        match (prompt.find(OPEN), prompt.find(CLOSE)) {
            (Some(start), Some(close)) if close > start => {
                let mut out = String::with_capacity(prompt.len());
                out.push_str(&prompt[..start]);
                out.push_str("<execution_environment/>");
                out.push_str(&prompt[close + CLOSE.len()..]);
                out
            }
            _ => prompt.to_string(),
        }
    }

    #[test]
    fn default_profile_prompt_bytes_are_unchanged() {
        const DEFAULT_PROFILE_PROMPT_HASH: u64 = 7_767_455_692_574_641_592;
        let prompt = render_turn(
            None,
            &[],
            &tools(FULL_TOOL_SET),
            &PromptContext::default(),
        );
        assert_eq!(
            stable_hash(without_environment_block(&prompt).as_bytes()),
            DEFAULT_PROFILE_PROMPT_HASH,
            "the default profile's prompt bytes drifted"
        );
    }
}

#[cfg(test)]
#[path = "skill_runtime_tests.rs"]
mod tests;
