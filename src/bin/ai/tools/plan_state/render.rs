//! plan rendering: decoupled from the state model / persistence. `PlanState::render`
//! is the only rendering entry point (shared by `plan` / `plan_update`); the
//! delegation/parallel orchestration hints live in `delegation_guidance`, separate from
//! structural rendering.

use super::model::{PlanState, StepStatus};

impl StepStatus {
    fn suffix(self) -> &'static str {
        match self {
            Self::Pending => "",
            Self::Running => " (running)",
            Self::Done => " (done)",
            Self::Failed => " (failed)",
            Self::Skipped => " (skipped)",
        }
    }
}

impl PlanState {
    /// Renders the full plan text (plan / plan_update share the same rendering path).
    ///
    /// The format stays consistent with the legacy `execute_plan`; only when non-pending
    /// steps exist does it append a `Progress: x/y steps done` line plus a per-step status
    /// suffix (including an optional note). `compact_plan_update_echo` (mod.rs) extracts
    /// step/progress lines using this function's line format, so changing this format
    /// requires updating its matching logic in sync (the
    /// `compact_echo_stays_in_sync_with_render_format` test locks in that coupling).
    pub(crate) fn render(&self) -> String {
        let total = self.steps.len();
        let done = self.done_count();
        let running = self.running_count();
        let failed = self.failed_count();
        let skipped = self.skipped_count();
        let has_progress = self.has_progress();

        let mut formatted = String::new();
        if !self.summary.is_empty() {
            formatted.push_str(&format!("Plan: {}\n", self.summary));
            if has_progress {
                formatted.push_str(&format!(
                    "Progress: {}\n",
                    progress_summary(done, running, failed, skipped, total)
                ));
            }
            formatted.push('\n');
        } else if has_progress {
            formatted.push_str(&format!(
                "Progress: {}\n\n",
                progress_summary(done, running, failed, skipped, total)
            ));
        }

        for step in &self.steps {
            let mut tags = String::new();
            if step.delegate {
                tags.push_str(" [delegate]");
            }
            let mut prefix = String::new();
            if step.parallelizable {
                prefix.push_str("  || ");
            }
            formatted.push_str(&format!(
                "{}Step {}. [{}]{} {}{}\n",
                prefix,
                step.step,
                step.tool,
                tags,
                step.action,
                render_status_suffix(step.status, step.note.as_deref()),
            ));
            if !step.reason.is_empty() {
                formatted.push_str(&format!("  Reason: {}\n", step.reason));
            }
        }

        formatted.push('\n');
        formatted.push_str(&format!("---\n{} step(s) planned.", total));
        let delegated = self.steps.iter().filter(|s| s.delegate).count();
        if delegated > 0 {
            formatted.push_str(&format!(" {} step(s) marked for delegation.", delegated));
        }
        let parallel = self.steps.iter().filter(|s| s.parallelizable).count();
        if parallel > 0 {
            formatted.push_str(&format!(" {} step(s) can run in parallel.", parallel));
        }
        let parallel_delegated = self
            .steps
            .iter()
            .filter(|s| s.parallelizable && s.delegate)
            .count();
        let serial_delegated = self
            .steps
            .iter()
            .filter(|s| !s.parallelizable && s.delegate)
            .count();
        // Delegation/parallel orchestration hint copy lives in `delegation_guidance` (separate from structural rendering).
        formatted.push_str(&delegation_guidance(
            delegated,
            parallel_delegated,
            serial_delegated,
        ));
        formatted.push('\n');
        formatted
    }
}

/// Summarizes the progress copy: `2/5 steps done, 1 running, 1 failed.`
fn progress_summary(
    done: usize,
    running: usize,
    failed: usize,
    skipped: usize,
    total: usize,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    if running > 0 {
        parts.push(format!("{running} running"));
    }
    if failed > 0 {
        parts.push(format!("{failed} failed"));
    }
    if skipped > 0 {
        parts.push(format!("{skipped} skipped"));
    }
    if parts.is_empty() {
        format!("{done}/{total} steps done.")
    } else {
        format!("{done}/{total} steps done, {}.", parts.join(", "))
    }
}

/// Render suffix for a step status plus an optional note: `(done)` / `(running, note: xx)` / empty string.
fn render_status_suffix(status: StepStatus, note: Option<&str>) -> String {
    let base = status.suffix();
    match note.filter(|n| !n.trim().is_empty()) {
        Some(n) if base.is_empty() => format!(" ({n})"),
        Some(n) => {
            // Merge the note into the same pair of parentheses: `(running, note: xx)`.
            let label = base.trim().trim_start_matches('(').trim_end_matches(')');
            format!(" ({label}, note: {n})")
        }
        None => base.to_string(),
    }
}

/// Delegation/parallel orchestration hint copy (operation suggestions visible to the model),
/// separate from structural plan rendering. Gives orchestration advice only when delegated
/// steps exist; without delegation, it suggests proceeding but considering delegation.
fn delegation_guidance(
    delegated: usize,
    parallel_delegated: usize,
    serial_delegated: usize,
) -> String {
    if delegated == 0 {
        return " Proceed to execute, but reconsider: any substantive step with real intermediate reads or commands is usually better delegated to a subagent (cleaner, focused context); keep only trivial single-tool steps and final review in the parent."
            .to_string();
    }
    let mut s = String::new();
    if parallel_delegated >= 2 {
        s.push_str(" Launch the parallel delegated steps concurrently via task_spawn, then collect with a single task_wait.");
    } else if parallel_delegated == 1 && serial_delegated == 0 {
        s.push_str(" Run the single delegated step with the synchronous `task` tool (async spawn+wait adds overhead without concurrency for one task).");
    } else if parallel_delegated == 1 {
        s.push_str(" Spawn the single parallel delegated step via task_spawn while you run the serial ones, then collect it with task_wait.");
    }
    if serial_delegated > 0 {
        s.push_str(" Run serial delegated steps one at a time with the synchronous `task`, passing the needed context from prior results in the prompt; never run dependent steps concurrently.");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_status_suffix_and_progress() {
        let raw = serde_json::json!([
            { "step": 1, "action": "Read", "tool": "read_file" },
            { "step": 2, "action": "Patch", "tool": "apply_patch" },
            { "step": 3, "action": "Check", "tool": "execute_command" }
        ]);
        let mut state = PlanState::build("Demo", raw.as_array().unwrap(), None).unwrap();
        let fresh = state.render();
        // Fresh plan: no progress line, no status suffixes.
        assert!(fresh.contains("Plan: Demo"));
        assert!(!fresh.contains("Progress:"));
        assert!(fresh.contains("3 step(s) planned."));

        state.apply_update(1, StepStatus::Done, None).unwrap();
        state
            .apply_update(2, StepStatus::Running, Some("on it".to_string()))
            .unwrap();
        state.apply_update(3, StepStatus::Failed, None).unwrap();
        let out = state.render();
        assert!(out.contains("Progress: 1/3 steps done, 1 running, 1 failed."));
        assert!(out.contains("Step 1. [read_file] Read (done)"));
        assert!(out.contains("Step 2. [apply_patch] Patch (running, note: on it)"));
        assert!(out.contains("Step 3. [execute_command] Check (failed)"));
    }
}
