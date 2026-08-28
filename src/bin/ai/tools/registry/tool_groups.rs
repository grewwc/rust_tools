//! Closed vocabulary of tool turn-groups.
//!
//! Group membership is declarative metadata: each `tool_descriptions/<tool>.json`
//! carries a `groups` array whose entries must name a [`ToolGroup`] variant, and
//! skill/agent manifests (`tool_groups:`) resolve their group names through the
//! same enum. This module is the single place that lists the existing groups;
//! adding one means adding a variant here (and, where relevant, teaching
//! `enable_tools` / `driver::skill_runtime` about its load semantics). Unknown
//! names are rejected at the metadata-parse boundary and skipped at the
//! manifest boundary, so a typo cannot silently detach a tool from every turn.

/// A tool group as declared in metadata JSON and skill/agent manifests.
///
/// Names are matched case-insensitively at both boundaries (`from_name`);
/// `as_str` is the canonical spelling used in JSON, manifests, and catalog
/// lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolGroup {
    /// Resident in every turn unless a manifest overrides the tool set.
    Core,
    /// Subagent orchestration family (`task` / `task_spawn` / ...). Lazy for
    /// every top-level turn; loaded eagerly when a manifest declares the group.
    Task,
    /// Kernel process / IPC / shared-memory / environment primitives for agents
    /// that manage OS processes. Members carrying the `hidden` metadata flag
    /// are deferred out of the resident turn set and the default catalog
    /// (`tool_defers_eager_load`) and must be enabled on demand by agents that
    /// declare this group.
    Executor,
    /// Long-term memory / knowledge-base tools.
    Knowledge,
    /// Skill management tools (`activate_skill` / `list_skills` / ...).
    Skills,
    /// Multi-agent team orchestration (`manage_team` / `run_agent_graph`),
    /// resident only when a manifest declares the group (agent-team skill).
    AgentTeam,
    /// Enable-ability flag rather than a loadable group: membership marks a
    /// tool as obtainable via `enable_tools`. It never expands as a group
    /// shortcut and is not meant to be declared by manifests.
    Builtin,
}

impl ToolGroup {
    /// All groups, for exhaustive iteration (catalogs, docs, tests).
    pub(crate) const ALL: &'static [ToolGroup] = &[
        Self::Core,
        Self::Task,
        Self::Executor,
        Self::Knowledge,
        Self::Skills,
        Self::AgentTeam,
        Self::Builtin,
    ];

    /// Canonical spelling used in metadata JSON, manifests, and user-facing
    /// catalog lines.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Task => "task",
            Self::Executor => "executor",
            Self::Knowledge => "knowledge",
            Self::Skills => "skills",
            Self::AgentTeam => "agent_team",
            Self::Builtin => "builtin",
        }
    }

    /// Resolve a group by name, case-insensitively. Returns `None` for names
    /// outside the vocabulary.
    pub(crate) fn from_name(name: &str) -> Option<ToolGroup> {
        Self::ALL
            .iter()
            .copied()
            .find(|group| group.as_str().eq_ignore_ascii_case(name))
    }

    /// True for the `builtin` enable-ability flag: it marks enable-ability and
    /// never acts as a loadable group (no expansion, no manifest declaration).
    pub(crate) const fn is_enable_ability_flag(self) -> bool {
        matches!(self, Self::Builtin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_name_is_case_insensitive_and_rejects_unknown() {
        assert_eq!(ToolGroup::from_name("core"), Some(ToolGroup::Core));
        assert_eq!(ToolGroup::from_name("Executor"), Some(ToolGroup::Executor));
        assert_eq!(ToolGroup::from_name("AGENT_TEAM"), Some(ToolGroup::AgentTeam));
        // Removed legacy group name must no longer resolve anywhere.
        assert_eq!(ToolGroup::from_name("openclaw"), None);
        assert_eq!(ToolGroup::from_name("no_such_group"), None);
    }

    #[test]
    fn all_names_are_unique_and_builtin_is_flag_only() {
        let mut names: Vec<&str> = ToolGroup::ALL.iter().map(|g| g.as_str()).collect();
        names.sort_unstable();
        let unique = names.len() == names.iter().len();
        assert!(unique, "duplicate canonical group name in ALL: {names:?}");
        for group in ToolGroup::ALL {
            assert_eq!(
                ToolGroup::from_name(group.as_str()),
                Some(*group),
                "ALL entry {} does not round-trip through from_name",
                group.as_str()
            );
        }
        assert!(ToolGroup::Builtin.is_enable_ability_flag());
        assert!(!ToolGroup::Core.is_enable_ability_flag());
    }
}
