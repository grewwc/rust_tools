use std::future::Future;
use std::io::IsTerminal;
use std::pin::Pin;
use std::sync::Arc;

use rust_tools::cw::SkipMap;

use crate::ai::middleware::tool::ToolMiddleware;
use crate::ai::ports::tool::{ToolExecOutput, ToolExecutor};
use crate::ai::types::{App, ToolCall, ToolResult};

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

impl ToolPermission {
    /// Parse a policy token (`allow` / `ask` / `deny`, case-insensitive).
    /// Returns `None` for unrecognized tokens so the caller can skip a
    /// malformed rule rather than silently coercing it to a default.
    fn parse(token: &str) -> Option<Self> {
        match token.trim().to_ascii_lowercase().as_str() {
            "allow" => Some(ToolPermission::Allow),
            "ask" => Some(ToolPermission::Ask),
            "deny" => Some(ToolPermission::Deny),
            _ => None,
        }
    }
}

/// Tool permission manager.
pub struct ToolPermissions {
    /// Per-tool overrides (tool_name -> permission).
    overrides: SkipMap<String, ToolPermission>,
    /// Default permission for unknown tools.
    default: ToolPermission,
    /// Patterns with wildcards (e.g. "execute_*" → Deny).
    patterns: Vec<(String, ToolPermission)>,
}

impl ToolPermissions {
    /// Create a new manager with default Allow for all tools.
    pub fn new() -> Self {
        Self {
            overrides: SkipMap::default(),
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

    /// Check permission for a tool. Precedence: an exact tool-name override
    /// wins; otherwise the FIRST inserted pattern that matches wins (insertion
    /// order, not longest/most-specific); otherwise the default. Callers that
    /// want a specific `prefix*` to beat a broader one must insert it first.
    pub fn check(&self, name: &str) -> ToolPermission {
        if let Some(&perm) = self.overrides.get_ref(&name.to_string()) {
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

impl ToolPermissions {
    /// Build a permission set from the config strings.
    ///
    /// `rules` is a comma-separated list of `pattern:policy` entries (e.g.
    /// `execute_command:ask, write_file:ask, apply_patch:deny`); `default` is an
    /// optional `allow`/`ask`/`deny` fallback for tools matched by no rule.
    ///
    /// Returns `None` when `rules` is empty/whitespace so callers can skip
    /// installing the middleware entirely on the zero-config path (preserving
    /// the current all-allow behavior). A malformed rule or default is skipped
    /// individually rather than failing the whole parse; the reason is returned
    /// in `warnings` for surfacing to the user.
    pub fn from_config(rules: &str, default: &str) -> Option<(Self, Vec<String>)> {
        let rules = rules.trim();
        if rules.is_empty() {
            return None;
        }
        let mut warnings = Vec::new();
        let mut perms = ToolPermissions::new();

        let default = default.trim();
        if !default.is_empty() {
            match ToolPermission::parse(default) {
                Some(policy) => perms.default = policy,
                None => warnings.push(format!(
                    "ignored invalid tool-permission default '{default}' (want allow/ask/deny)"
                )),
            }
        }

        for raw in rules.split(',') {
            let entry = raw.trim();
            if entry.is_empty() {
                continue;
            }
            // Split on the last ':' so a (hypothetical) tool name containing ':'
            // keeps the trailing token as the policy. Tool names are verb-first
            // snake_case and MCP names use `mcp_<server>_` prefixes, so neither
            // contains ':' today; rsplit is a defensive choice.
            let Some((pattern, policy_token)) = entry.rsplit_once(':') else {
                warnings.push(format!("ignored tool-permission rule '{entry}' (want pattern:policy)"));
                continue;
            };
            let pattern = pattern.trim();
            if pattern.is_empty() {
                warnings.push(format!("ignored tool-permission rule '{entry}' (empty pattern)"));
                continue;
            }
            let Some(policy) = ToolPermission::parse(policy_token) else {
                warnings.push(format!(
                    "ignored tool-permission rule '{entry}' (policy must be allow/ask/deny)"
                ));
                continue;
            };
            if pattern.ends_with('*') {
                perms.set_pattern(pattern, policy);
            } else {
                perms.set_tool(pattern, policy);
            }
        }

        Some((perms, warnings))
    }
}

/// Synthesize a denied `ToolResult` for a single tool call. Mirrors the
/// `reject_tool_calls` convention in the turn runtime: the content begins with
/// `Error:` so downstream success detection (`starts_with("Error:")`) never
/// treats a blocked call as reusable success evidence.
fn denied_tool_result(tool_call: &ToolCall, policy_note: &str) -> ToolResult {
    ToolResult {
        tool_call_id: tool_call.id.clone(),
        content: format!(
            "Error: tool '{}' was blocked by the configured tool-permission policy ({policy_note}). \
             Adjust `ai.tools.permissions` to allow it, or choose a different approach.",
            tool_call.function.name
        ),
    }
}

/// Tool-execution middleware that enforces a [`ToolPermissions`] policy before
/// delegating to the inner executor.
///
/// Gating is per call, not per batch: allowed calls are forwarded to `inner` in
/// one dispatch while denied calls are synthesized locally, then the two are
/// merged back in the original request order so the parallel result vectors stay
/// positionally consistent with what a full inner dispatch would produce.
///
/// `Ask` requires an interactive stdin; in a non-TTY foreground (or any context
/// without a terminal) it fails closed and is treated as `Deny`, matching the
/// fail-closed posture of the `git commit` gate.
pub(crate) struct PermissionMiddleware {
    permissions: Arc<ToolPermissions>,
}

impl PermissionMiddleware {
    pub(crate) fn new(permissions: ToolPermissions) -> Self {
        Self {
            permissions: Arc::new(permissions),
        }
    }
}

impl ToolMiddleware for PermissionMiddleware {
    fn name(&self) -> &'static str {
        "permissions"
    }

    fn wrap(&self, inner: Box<dyn ToolExecutor>) -> Box<dyn ToolExecutor> {
        Box::new(PermissionExecutor {
            inner,
            permissions: Arc::clone(&self.permissions),
        })
    }
}

struct PermissionExecutor {
    inner: Box<dyn ToolExecutor>,
    permissions: Arc<ToolPermissions>,
}

/// Per-call permission decision resolved before dispatch.
enum CallDecision {
    /// Forward to the inner executor.
    Allow,
    /// Block locally with the given note (deny, or ask that failed closed / was declined).
    Deny(String),
}

impl PermissionExecutor {
    /// Resolve each call's decision. `Ask` prompts synchronously; a non-TTY
    /// context or a declined/interrupted prompt fails closed to `Deny`.
    fn decide(&self, tool_calls: &[ToolCall]) -> Vec<CallDecision> {
        tool_calls
            .iter()
            .map(|call| match self.permissions.check(&call.function.name) {
                ToolPermission::Allow => CallDecision::Allow,
                ToolPermission::Deny => CallDecision::Deny("policy=deny".to_string()),
                ToolPermission::Ask => {
                    if !std::io::stdin().is_terminal() {
                        return CallDecision::Deny(
                            "policy=ask but no interactive terminal; failing closed".to_string(),
                        );
                    }
                    match crate::commonw::prompt::prompt_yes_or_no_interruptible(&format!(
                        "Allow tool '{}' to run? (y/n): ",
                        call.function.name
                    )) {
                        Some(true) => CallDecision::Allow,
                        Some(false) => CallDecision::Deny("declined by user".to_string()),
                        None => CallDecision::Deny("prompt interrupted".to_string()),
                    }
                }
            })
            .collect()
    }
}

impl ToolExecutor for PermissionExecutor {
    fn execute<'a>(
        &'a self,
        app: &'a mut App,
        tool_calls: Vec<ToolCall>,
    ) -> Pin<Box<dyn Future<Output = Result<ToolExecOutput, Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>>
    {
        Box::pin(async move {
            let decisions = self.decide(&tool_calls);

            // Fast path: everything allowed → single inner dispatch, zero reassembly.
            if decisions.iter().all(|d| matches!(d, CallDecision::Allow)) {
                return self.inner.execute(app, tool_calls).await;
            }

            // Partition, preserving each call's original index for later merge.
            let mut allowed: Vec<ToolCall> = Vec::new();
            let mut allowed_slots: Vec<usize> = Vec::new();
            let mut denied: Vec<(usize, ToolResult)> = Vec::new();
            for (idx, (call, decision)) in tool_calls.iter().zip(decisions.iter()).enumerate() {
                match decision {
                    CallDecision::Allow => {
                        allowed.push(call.clone());
                        allowed_slots.push(idx);
                    }
                    CallDecision::Deny(note) => {
                        denied.push((idx, denied_tool_result(call, note)));
                    }
                }
            }

            // Dispatch the allowed subset (if any) through the inner chain.
            let inner_out = if allowed.is_empty() {
                ToolExecOutput::default()
            } else {
                self.inner.execute(app, allowed).await?
            };

            // Reassemble all four parallel vectors in the original request order.
            // Slot arrays are keyed by the original index so denied and allowed
            // results interleave exactly where their calls appeared.
            let total = tool_calls.len();
            let mut results_by_slot: Vec<Option<ToolResult>> = (0..total).map(|_| None).collect();
            let mut executed_by_slot: Vec<Option<ToolCall>> = (0..total).map(|_| None).collect();
            let mut cached_by_slot: Vec<Option<bool>> = (0..total).map(|_| None).collect();
            let mut outcome_by_slot: Vec<Option<Option<crate::ai::history::ToolExecutionOutcome>>> =
                (0..total).map(|_| None).collect();

            // Denied slots: synthesized result, not executed, not cached, no outcome.
            for (idx, result) in denied {
                results_by_slot[idx] = Some(result);
                executed_by_slot[idx] = Some(tool_calls[idx].clone());
                cached_by_slot[idx] = Some(false);
                outcome_by_slot[idx] = Some(None);
            }

            // Allowed slots: map the inner output back by position. The inner
            // executor preserves input order 1:1, but it may cancel mid-batch
            // and return fewer entries; only fill as many slots as it produced.
            let ToolExecOutput {
                tool_results,
                assistant_messages,
                executed_tool_calls,
                cached_hits,
                execution_outcomes,
                had_error: inner_had_error,
            } = inner_out;
            for (i, result) in tool_results.into_iter().enumerate() {
                if let Some(&slot) = allowed_slots.get(i) {
                    results_by_slot[slot] = Some(result);
                }
            }
            for (i, call) in executed_tool_calls.into_iter().enumerate() {
                if let Some(&slot) = allowed_slots.get(i) {
                    executed_by_slot[slot] = Some(call);
                }
            }
            for (i, cached) in cached_hits.into_iter().enumerate() {
                if let Some(&slot) = allowed_slots.get(i) {
                    cached_by_slot[slot] = Some(cached);
                }
            }
            for (i, outcome) in execution_outcomes.into_iter().enumerate() {
                if let Some(&slot) = allowed_slots.get(i) {
                    outcome_by_slot[slot] = Some(outcome);
                }
            }

            // Flatten, dropping slots the inner executor never produced (e.g. a
            // cancelled tail). Each vector is filtered independently but they
            // stay mutually consistent because they are keyed on the same slots.
            let merged_results: Vec<ToolResult> = results_by_slot.into_iter().flatten().collect();
            let merged_executed: Vec<ToolCall> = executed_by_slot.into_iter().flatten().collect();
            let merged_cached: Vec<bool> = cached_by_slot.into_iter().flatten().collect();
            let merged_outcomes: Vec<Option<crate::ai::history::ToolExecutionOutcome>> =
                outcome_by_slot.into_iter().flatten().collect();

            Ok(ToolExecOutput {
                tool_results: merged_results,
                assistant_messages,
                executed_tool_calls: merged_executed,
                cached_hits: merged_cached,
                execution_outcomes: merged_outcomes,
                // Any blocked call is a turn-level error signal, consistent with
                // reject_tool_calls setting had_error = true.
                had_error: inner_had_error || allowed_slots.len() != total,
            })
        })
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

    // ── from_config parsing ──────────────────────────────

    #[test]
    fn from_config_empty_rules_returns_none() {
        assert!(ToolPermissions::from_config("", "deny").is_none());
        assert!(ToolPermissions::from_config("   ", "").is_none());
    }

    #[test]
    fn from_config_parses_rules_and_default() {
        let (perms, warnings) = ToolPermissions::from_config(
            "execute_command:ask, apply_patch:deny, read_*:allow",
            "deny",
        )
        .expect("non-empty rules install a policy");
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        assert!(perms.needs_ask("execute_command"));
        assert!(perms.is_denied("apply_patch"));
        assert!(perms.is_allowed("read_file"), "prefix rule read_* should allow");
        // Unmatched tool falls back to the parsed default (deny).
        assert!(perms.is_denied("some_other_tool"));
    }

    #[test]
    fn from_config_skips_malformed_rules_with_warnings() {
        let (perms, warnings) = ToolPermissions::from_config(
            "good_tool:deny, no_colon_here, empty_policy:, :empty_pattern, bad:whatever",
            "",
        )
        .expect("at least one token present");
        assert!(perms.is_denied("good_tool"));
        // 4 malformed entries → 4 warnings; the valid one is applied.
        assert_eq!(warnings.len(), 4, "warnings: {warnings:?}");
        // Malformed entries must not silently become a policy.
        assert!(perms.is_allowed("no_colon_here"));
    }

    #[test]
    fn from_config_invalid_default_warns_but_keeps_allow() {
        let (perms, warnings) =
            ToolPermissions::from_config("x:deny", "not_a_policy").expect("rules present");
        assert_eq!(warnings.len(), 1);
        // Invalid default is ignored → falls back to the built-in Allow.
        assert!(perms.is_allowed("unmatched"));
    }

    #[test]
    fn from_config_pattern_precedence_is_first_registered() {
        // Two overlapping patterns; the earlier one in config order must win for
        // a name both match. `read_special_*` is listed first, so it beats the
        // broader `read_*` for `read_special_file`.
        let (perms, _) =
            ToolPermissions::from_config("read_special_*:deny, read_*:allow", "").expect("rules");
        assert!(
            perms.is_denied("read_special_file"),
            "first-registered matching pattern wins"
        );
        // A name only the broad pattern matches still resolves to it.
        assert!(perms.is_allowed("read_file"));
        // Reversing the order flips the outcome — proves it is order-driven, not
        // specificity-driven.
        let (perms_rev, _) =
            ToolPermissions::from_config("read_*:allow, read_special_*:deny", "").expect("rules");
        assert!(
            perms_rev.is_allowed("read_special_file"),
            "broad pattern listed first now wins"
        );
    }

    // ── PermissionMiddleware gating ──────────────────────

    use crate::ai::middleware::test_util::test_app;
    use crate::ai::types::{FunctionCall, ToolCall};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: "{}".to_string(),
            },
        }
    }

    /// Inner executor that echoes one successful ToolResult per received call and
    /// records how many calls it was asked to run.
    struct EchoExecutor {
        seen: Arc<AtomicUsize>,
    }
    impl ToolExecutor for EchoExecutor {
        fn execute<'a>(
            &'a self,
            _app: &'a mut App,
            tool_calls: Vec<ToolCall>,
        ) -> Pin<Box<dyn Future<Output = Result<ToolExecOutput, Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>>
        {
            self.seen.fetch_add(tool_calls.len(), Ordering::SeqCst);
            Box::pin(async move {
                let n = tool_calls.len();
                Ok(ToolExecOutput {
                    tool_results: tool_calls
                        .iter()
                        .map(|c| ToolResult {
                            tool_call_id: c.id.clone(),
                            content: format!("ok:{}", c.function.name),
                        })
                        .collect(),
                    assistant_messages: Vec::new(),
                    executed_tool_calls: tool_calls,
                    cached_hits: vec![false; n],
                    execution_outcomes: (0..n).map(|_| None).collect(),
                    had_error: false,
                })
            })
        }
    }

    fn deny_only(tool: &str) -> PermissionMiddleware {
        let mut perms = ToolPermissions::new();
        perms.set_tool(tool, ToolPermission::Deny);
        PermissionMiddleware::new(perms)
    }

    #[tokio::test]
    async fn all_allowed_takes_fast_path() {
        let seen = Arc::new(AtomicUsize::new(0));
        let mw = deny_only("never_called");
        let exec = mw.wrap(Box::new(EchoExecutor { seen: Arc::clone(&seen) }));
        let mut app = test_app();
        let out = exec
            .execute(&mut app, vec![call("a", "read_file"), call("b", "tree")])
            .await
            .unwrap();
        assert_eq!(seen.load(Ordering::SeqCst), 2, "both calls reach inner");
        assert_eq!(out.tool_results.len(), 2);
        assert!(!out.had_error);
    }

    #[tokio::test]
    async fn denied_call_is_synthesized_and_not_dispatched() {
        let seen = Arc::new(AtomicUsize::new(0));
        let mw = deny_only("apply_patch");
        let exec = mw.wrap(Box::new(EchoExecutor { seen: Arc::clone(&seen) }));
        let mut app = test_app();
        let out = exec
            .execute(&mut app, vec![call("only", "apply_patch")])
            .await
            .unwrap();
        assert_eq!(seen.load(Ordering::SeqCst), 0, "denied call never reaches inner");
        assert_eq!(out.tool_results.len(), 1);
        assert!(out.tool_results[0].content.starts_with("Error:"));
        assert_eq!(out.tool_results[0].tool_call_id, "only");
        assert!(out.had_error, "a blocked call flags turn error");
    }

    #[tokio::test]
    async fn partial_gating_preserves_request_order() {
        let seen = Arc::new(AtomicUsize::new(0));
        let mw = deny_only("apply_patch");
        let exec = mw.wrap(Box::new(EchoExecutor { seen: Arc::clone(&seen) }));
        let mut app = test_app();
        // Denied call sits in the MIDDLE to prove slot-based reassembly, not append.
        let out = exec
            .execute(
                &mut app,
                vec![
                    call("c0", "read_file"),
                    call("c1", "apply_patch"),
                    call("c2", "tree"),
                ],
            )
            .await
            .unwrap();
        assert_eq!(seen.load(Ordering::SeqCst), 2, "only allowed calls dispatched");
        // All four parallel vectors stay aligned and in original order.
        let ids: Vec<&str> = out
            .tool_results
            .iter()
            .map(|r| r.tool_call_id.as_str())
            .collect();
        assert_eq!(ids, vec!["c0", "c1", "c2"], "result order matches request order");
        let exec_ids: Vec<&str> = out
            .executed_tool_calls
            .iter()
            .map(|c| c.id.as_str())
            .collect();
        assert_eq!(exec_ids, vec!["c0", "c1", "c2"]);
        assert_eq!(out.cached_hits.len(), 3);
        assert_eq!(out.execution_outcomes.len(), 3);
        // Middle slot is the synthesized denial.
        assert!(out.tool_results[1].content.starts_with("Error:"));
        assert!(out.tool_results[0].content.starts_with("ok:"));
        assert!(out.tool_results[2].content.starts_with("ok:"));
        assert!(out.had_error);
    }
}
