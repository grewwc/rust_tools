//! plan 渲染：与状态模型 / 持久化解耦。`PlanState::render` 是唯一渲染入口
//! （`plan` / `plan_update` 共用）；委派/并行编排提示文案在 `delegation_guidance`，
//! 与结构渲染分离。

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

    /// 渲染完整计划文本（plan / plan_update 共用同一渲染路径）。
    ///
    /// 格式与旧版 `execute_plan` 保持一致；仅在存在非 pending 步骤时追加
    /// `Progress: x/y steps done` 行与每步状态后缀（含可选 note）。
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
        let parallel_delegated = self.steps.iter().filter(|s| s.parallelizable && s.delegate).count();
        let serial_delegated = self.steps.iter().filter(|s| !s.parallelizable && s.delegate).count();
        // 委派/并行编排提示文案见 `delegation_guidance`（与结构渲染分离）。
        formatted.push_str(&delegation_guidance(delegated, parallel_delegated, serial_delegated));
        formatted.push('\n');
        formatted
    }
}

/// 汇总进度文案：`2/5 steps done, 1 running, 1 failed.`
fn progress_summary(done: usize, running: usize, failed: usize, skipped: usize, total: usize) -> String {
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

/// 步骤状态 + 可选备注的渲染后缀：`(done)` / `(running, note: xx)` / 空字符串。
fn render_status_suffix(status: StepStatus, note: Option<&str>) -> String {
    let base = status.suffix();
    match note.filter(|n| !n.trim().is_empty()) {
        Some(n) if base.is_empty() => format!(" ({n})"),
        Some(n) => {
            // note 并入同一对括号：`(running, note: xx)`。
            let label = base.trim().trim_start_matches('(').trim_end_matches(')');
            format!(" ({label}, note: {n})")
        }
        None => base.to_string(),
    }
}

/// 委派/并行编排提示文案（模型可见的操作建议），与计划结构渲染分离。
/// 仅在存在委派步骤时给出编排建议；无委派时给出"推进但可考虑委派"的提示。
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
        // 全新计划：无进度行、无状态后缀。
        assert!(fresh.contains("Plan: Demo"));
        assert!(!fresh.contains("Progress:"));
        assert!(fresh.contains("3 step(s) planned."));

        state.apply_update(1, StepStatus::Done, None).unwrap();
        state.apply_update(2, StepStatus::Running, Some("on it".to_string())).unwrap();
        state.apply_update(3, StepStatus::Failed, None).unwrap();
        let out = state.render();
        assert!(out.contains("Progress: 1/3 steps done, 1 running, 1 failed."));
        assert!(out.contains("Step 1. [read_file] Read (done)"));
        assert!(out.contains("Step 2. [apply_patch] Patch (running, note: on it)"));
        assert!(out.contains("Step 3. [execute_command] Check (failed)"));
    }
}
