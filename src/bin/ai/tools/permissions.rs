use rust_tools::commonw::FastMap;

/// Permission level for a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolPermission {
    /// Always prompt user before executing.
    Ask,
    /// Execute without prompting.
    Allow,
    /// Block execution entirely.
    Deny,
}

/// Tool permission manager.
pub struct ToolPermissions {
    /// Per-tool overrides (tool_name -> permission).
    overrides: FastMap<String, ToolPermission>,
    /// Default permission for unknown tools.
    default: ToolPermission,
    /// Patterns with wildcards (e.g. "execute_*" → Deny).
    patterns: Vec<(String, ToolPermission)>,
}

impl ToolPermissions {
    /// Create a new manager with default Allow for all tools.
    pub fn new() -> Self {
        Self {
            overrides: FastMap::default(),
            default: ToolPermission::Allow,
            patterns: Vec::new(),
        }
    }

    /// Set the default permission for unknown tools.
    pub fn with_default(mut self, perm: ToolPermission) -> Self {
        self.default = perm;
        self
    }

    /// Override permission for a specific tool.
    pub fn set_tool(&mut self, name: &str, perm: ToolPermission) {
        self.overrides.insert(name.to_string(), perm);
    }

    /// Add a wildcard pattern (simple glob: `*` matches any suffix).
    pub fn set_pattern(&mut self, pattern: &str, perm: ToolPermission) {
        self.patterns.push((pattern.to_string(), perm));
    }

    /// Check permission for a tool. Most specific match wins: exact > pattern > default.
    pub fn check(&self, name: &str) -> ToolPermission {
        if let Some(&perm) = self.overrides.get(name) {
            return perm;
        }
        for (pattern, perm) in &self.patterns {
            if matches_pattern(pattern, name) {
                return *perm;
            }
        }
        self.default
    }

    /// Convenience: returns true if the tool is allowed to execute without prompting.
    pub fn is_allowed(&self, name: &str) -> bool {
        self.check(name) == ToolPermission::Allow
    }

    /// Convenience: returns true if the tool is blocked.
    pub fn is_denied(&self, name: &str) -> bool {
        self.check(name) == ToolPermission::Deny
    }

    /// Convenience: returns true if the tool requires user confirmation.
    pub fn needs_ask(&self, name: &str) -> bool {
        self.check(name) == ToolPermission::Ask
    }
}

impl Default for ToolPermissions {
    fn default() -> Self {
        Self::new()
    }
}

/// Match a tool name against a simple glob pattern where `*` matches any suffix.
fn matches_pattern(pattern: &str, name: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix('*') {
        name.starts_with(prefix)
    } else {
        pattern == name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_allow() {
        let perms = ToolPermissions::new();
        assert!(perms.is_allowed("any_tool"));
        assert!(!perms.is_denied("any_tool"));
        assert!(!perms.needs_ask("any_tool"));
    }

    #[test]
    fn test_exact_override() {
        let mut perms = ToolPermissions::new();
        perms.set_tool("dangerous_tool", ToolPermission::Deny);
        assert!(perms.is_denied("dangerous_tool"));
        assert!(perms.is_allowed("other_tool"));
    }

    #[test]
    fn test_pattern_matching() {
        let mut perms = ToolPermissions::new();
        perms.set_pattern("execute_*", ToolPermission::Deny);
        assert!(perms.is_denied("execute_command"));
        assert!(perms.is_denied("execute_script"));
        assert!(perms.is_allowed("run_command"));
    }

    #[test]
    fn test_exact_takes_priority_over_pattern() {
        let mut perms = ToolPermissions::new();
        perms.set_pattern("execute_*", ToolPermission::Deny);
        perms.set_tool("execute_safe", ToolPermission::Allow);
        assert!(perms.is_allowed("execute_safe"));
        assert!(perms.is_denied("execute_unsafe"));
    }

    #[test]
    fn test_deny_blocks_execution() {
        let mut perms = ToolPermissions::new();
        perms.set_tool("blocked_tool", ToolPermission::Deny);
        assert!(perms.is_denied("blocked_tool"));
        assert!(!perms.is_allowed("blocked_tool"));
        assert!(!perms.needs_ask("blocked_tool"));
    }

    #[test]
    fn test_with_default() {
        let perms = ToolPermissions::new().with_default(ToolPermission::Ask);
        assert!(perms.needs_ask("unknown_tool"));
        assert!(!perms.is_allowed("unknown_tool"));
    }

    #[test]
    fn test_default_trait() {
        let perms = ToolPermissions::default();
        assert!(perms.is_allowed("any_tool"));
    }
}
