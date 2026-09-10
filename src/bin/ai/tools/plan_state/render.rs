//! plan rendering: decoupled from the state model / persistence. `PlanState::render` is the
//! full-plan entry point used by `plan`; `plan_update` returns a delta through
//! `PlanState::render_update_delta`, and working checkpoints store their single full copy
//! through `PlanState::render_recovery_snapshot`. The delegation/parallel orchestration
//! hints live in `delegation_guidance`, separate from structural rendering.

use super::model::{PlanState, PlanStepState, StepStatus};

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
        let has_progress = self.has_progress();

        let mut formatted = String::new();
        if !self.summary.is_empty() {
            formatted.push_str(&format!("Plan: {}\n", self.summary));
            if has_progress {
                formatted.push_str(&format!("Progress: {}\n", self.progress_line()));
            }
            formatted.push('\n');
        } else if has_progress {
            formatted.push_str(&format!("Progress: {}\n\n", self.progress_line()));
        }

        formatted.push_str(&self.render_step_lines());

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

    /// Model-facing result of one `plan_update`.
    ///
    /// Carries what a status flip changes without re-echoing the whole plan: the aggregate
    /// progress line, the steps that are not terminal yet (pending/running) with their reasons,
    /// the steps that failed with their reasons, and — when the flip closed a step out — that
    /// step with its reason on a line of its own. Every step appears at most once. Reasons stay
    /// attached because after context compression nothing re-injects the full plan
    /// automatically (the model would have to remember to read `plan-state.json`): the most
    /// recent delta lives in the compression tail window and survives it, so it must stay a
    /// self-contained roadmap. A failure is unfinished work (retry, re-plan, or an explicit
    /// skip), not a closed chapter, so failed steps stay in every later delta too, while
    /// completed/skipped steps other than the changed one and the plan-time orchestration
    /// footer stay out.
    pub(crate) fn render_update_delta(&self, step: u64) -> String {
        let mut formatted = String::new();
        if !self.summary.is_empty() {
            formatted.push_str(&format!("Plan: {}\n", self.summary));
        }
        // A step that just reached a completed/skipped state is listed nowhere else below, so it
        // gets an explicit line here: the model must still see which step closed out and why it
        // existed. A non-terminal changed step already appears in the remaining list and a
        // failed one in the failed list, so echoing either here would print the same step twice
        // in one result.
        if let Some(changed) = self.steps.iter().find(|candidate| {
            candidate.step == step
                && matches!(candidate.status, StepStatus::Done | StepStatus::Skipped)
        }) {
            formatted.push_str(&render_step_headline(changed));
            if !changed.reason.is_empty() {
                formatted.push_str(&format!("  Reason: {}\n", changed.reason));
            }
        }
        formatted.push_str(&format!("Progress: {}\n", self.progress_line()));
        // Failures persist in every later delta: the model must see outstanding work without
        // reading `plan-state.json` once the update that produced the failure scrolls out.
        let failed = self
            .steps
            .iter()
            .filter(|candidate| candidate.status == StepStatus::Failed)
            .collect::<Vec<_>>();
        if !failed.is_empty() {
            formatted.push_str("Failed steps:\n");
            formatted.push_str(&render_steps_with_reasons(failed));
        }
        let remaining = self
            .steps
            .iter()
            .filter(|candidate| !candidate.status.is_terminal())
            .collect::<Vec<_>>();
        if remaining.is_empty() {
            formatted.push_str("Remaining steps: none.\n");
        } else {
            formatted.push_str("Remaining steps:\n");
            formatted.push_str(&render_steps_with_reasons(remaining));
        }
        formatted
    }

    /// Single full copy of the plan for recovery checkpoints: the progress line followed by
    /// every step with its live status and reason.
    ///
    /// Checkpoints store this instead of keeping a step list *and* a second copy of the
    /// `plan` tool output, which described the same steps twice.
    pub(crate) fn render_recovery_snapshot(&self) -> String {
        let mut formatted = String::new();
        if !self.summary.is_empty() {
            formatted.push_str(&format!("Plan: {}\n", self.summary));
        }
        formatted.push_str(&format!("Progress: {}\n", self.progress_line()));
        formatted.push_str(&self.render_step_lines());
        formatted
    }

    /// Per-step lines used by the full render and the recovery snapshot: a headline plus an
    /// indented `Reason:` line when the step carries one.
    fn render_step_lines(&self) -> String {
        render_steps_with_reasons(&self.steps)
    }

    /// Aggregate progress copy for the current step counts, e.g. `2/5 steps done, 1 running, 1 failed.`
    fn progress_line(&self) -> String {
        progress_summary(
            self.done_count(),
            self.running_count(),
            self.failed_count(),
            self.skipped_count(),
            self.steps.len(),
        )
    }
}

/// Renders one or more steps as headline plus indented `Reason:` line, in order.
///
/// Shared by the full render, the recovery snapshot, and the update delta so all three keep
/// a single line format for steps that carry a reason.
fn render_steps_with_reasons<'a>(steps: impl IntoIterator<Item = &'a PlanStepState>) -> String {
    let mut rendered = String::new();
    for step in steps {
        rendered.push_str(&render_step_headline(step));
        if !step.reason.is_empty() {
            rendered.push_str(&format!("  Reason: {}\n", step.reason));
        }
    }
    rendered
}

/// Renders one step headline, e.g. `Step 2. [apply_patch] Patch [delegate] (running, note: xx)`.
///
/// Shared by the full render, the update delta, and the recovery snapshot so all three keep
/// a single line format. Parallel steps keep their `  || ` marker.
fn render_step_headline(step: &PlanStepState) -> String {
    let tags = if step.delegate { " [delegate]" } else { "" };
    let prefix = if step.parallelizable { "  || " } else { "" };
    format!(
        "{prefix}Step {}. [{}]{tags} {}{}\n",
        step.step,
        step.tool,
        step.action,
        render_status_suffix(step.status, step.note.as_deref()),
    )
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

    #[test]
    fn test_render_update_delta_stays_a_delta() {
        let raw = serde_json::json!([
            { "step": 1, "action": "Read", "tool": "read_file", "reason": "Locate the writer" },
            { "step": 2, "action": "Patch", "tool": "apply_patch", "reason": "Fix the writer" },
            { "step": 3, "action": "Check", "tool": "execute_command", "reason": "Prove the fix" }
        ]);
        let mut state = PlanState::build("Demo", raw.as_array().unwrap(), None).unwrap();
        state.apply_update(1, StepStatus::Done, None).unwrap();
        state
            .apply_update(2, StepStatus::Running, Some("editing".to_string()))
            .unwrap();

        // A non-terminal flip needs no line of its own: the remaining list shows the step in
        // place, so the delta still prints it exactly once.
        let delta = state.render_update_delta(2);
        assert!(delta.starts_with("Plan: Demo\n"));
        assert!(delta.contains("Progress: 1/3 steps done, 1 running."));
        assert_eq!(delta.matches("Step 2.").count(), 1, "got: {delta}");

        let remaining = delta.split("Remaining steps:\n").nth(1).unwrap();
        assert!(remaining.contains("Step 2. [apply_patch] Patch (running, note: editing)"));
        assert!(remaining.contains("Step 3. [execute_command] Check"));
        // Terminal steps are only counted in the progress line, never re-listed.
        assert!(!remaining.contains("Step 1."));
        // Every remaining step keeps its reason: the most recent delta must stay a
        // self-contained roadmap after compression (nothing re-injects the full plan), so the
        // reasons of the work still ahead stay visible. Terminal steps' reasons and the
        // plan-time footer stay out.
        assert!(delta.contains("  Reason: Fix the writer\n"));
        assert!(!delta.contains("Reason: Locate the writer"));
        assert!(delta.contains("  Reason: Prove the fix\n"));
        assert!(!delta.contains("step(s) planned."));

        // A flip that closes a step out does give it an explicit line with its reason: terminal
        // steps are excluded from the remaining list, so this line is the only place where the
        // model can still see which step just finished and why it existed.
        let closed = state.render_update_delta(1);
        assert!(
            closed.contains("Step 1. [read_file] Read (done)\n  Reason: Locate the writer\n"),
            "got: {closed}"
        );
        assert_eq!(closed.matches("Step 1.").count(), 1, "got: {closed}");

        // A failure keeps its place in every later delta with its reason and note: it is work
        // still outstanding (retry, re-plan, or explicit skip), so it must not fall out of the
        // model's view while the rest of the plan is worked through.
        state
            .apply_update(3, StepStatus::Failed, Some("tests red".to_string()))
            .unwrap();
        let failure = state.render_update_delta(3);
        assert!(
            failure.contains(
                "Failed steps:\nStep 3. [execute_command] Check (failed, note: tests red)\n  Reason: Prove the fix\n"
            ),
            "got: {failure}"
        );
        assert_eq!(failure.matches("Step 3.").count(), 1, "got: {failure}");

        let later = state.render_update_delta(1);
        assert!(later.contains("Failed steps:\n"), "got: {later}");
        assert!(
            later.contains("Step 3. [execute_command] Check (failed, note: tests red)"),
            "got: {later}"
        );
        assert!(later.contains("  Reason: Prove the fix\n"), "got: {later}");
        assert_eq!(later.matches("Step 3.").count(), 1, "got: {later}");
    }

    #[test]
    fn test_render_update_delta_reports_a_finished_plan() {
        let raw = serde_json::json!([{ "step": 1, "action": "Read", "tool": "read_file" }]);
        let mut state = PlanState::build("Demo", raw.as_array().unwrap(), None).unwrap();
        state.apply_update(1, StepStatus::Done, None).unwrap();

        let delta = state.render_update_delta(1);
        assert!(delta.contains("Progress: 1/1 steps done."));
        assert!(delta.contains("Remaining steps: none."));
    }

    #[test]
    fn test_render_recovery_snapshot_keeps_one_full_copy() {
        let raw = serde_json::json!([
            { "step": 1, "action": "Read", "tool": "read_file", "reason": "Locate the writer" },
            { "step": 2, "action": "Patch", "tool": "apply_patch", "reason": "Fix the writer" }
        ]);
        let mut state = PlanState::build("Demo", raw.as_array().unwrap(), None).unwrap();
        state.apply_update(1, StepStatus::Done, None).unwrap();

        let snapshot = state.render_recovery_snapshot();
        assert!(snapshot.starts_with("Plan: Demo\nProgress: 1/2 steps done.\n"));
        assert!(snapshot.contains("Step 1. [read_file] Read (done)\n  Reason: Locate the writer\n"));
        assert!(snapshot.contains("Step 2. [apply_patch] Patch\n  Reason: Fix the writer\n"));
        // The recovery snapshot is the checkpoint's single copy of the plan: no second
        // description, no plan-time footer bookkeeping around the step list.
        assert_eq!(snapshot.matches("Step 1.").count(), 1);
        assert!(!snapshot.contains("step(s) planned."));
        assert!(!snapshot.contains("Progress: 0/2"));
    }
}
