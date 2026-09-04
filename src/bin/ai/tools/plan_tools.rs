use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;
use crate::ai::tools::common::{ToolDisplayConfig, ToolDisplayRegistration};
use crate::ai::tools::common::{
    ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
};

fn execute_plan(args: &Value) -> Result<String, String> {
    let steps = args["steps"]
        .as_array()
        .ok_or("Missing 'steps' array. Provide a JSON array of plan steps.")?;
    let summary = args["summary"].as_str().unwrap_or("");

    if steps.is_empty() {
        return Err("Plan must contain at least one step.".to_string());
    }

    // Render through the shared state model and `PlanState::render`, which is also
    // used by `plan_update`, rather than maintaining duplicate formatting here.
    // Then register the plan as session state for later updates. Registration
    // failures do not suppress the plan output, but must remain visible as a warning.
    let rendered = crate::ai::tools::plan_state::PlanState::build(summary, steps, None)?.render();
    if let Some(ctx) = crate::ai::driver::runtime_ctx::try_current() {
        match crate::ai::tools::plan_state::record_plan(&ctx.app_proto, summary, steps) {
            // Render persisted state so replanning preserves completed-step suffixes and progress.
            Ok(state) => Ok(state.render()),
            Err(e) => Ok(format!(
                "{rendered}\nWarning: could not register this plan as session state (plan_update will fail until this is fixed): {e}\n"
            )),
        }
    } else {
        Ok(rendered)
    }
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "plan",
        description: "",

        execute: execute_plan,
    }
});

// Plan output has high user-facing value, so show the result.
inventory::submit!(ToolDisplayRegistration {
    name: "plan",
    config: ToolDisplayConfig {
        print_args: false,
        print_result: true,
        emphasize_result: false,
        display: None,
    },
});

// The latest plan is the task-roadmap anchor and must remain complete. The recent
// tool-group retention window (`KEEP_RECENT_TOOL_GROUPS`) protects it from lossy
// compression. Superseded plans may be summarized, but model-directed pruning is
// still forbidden so the model cannot invalidate the roadmap on its own.
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "plan",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Allow,
        prune: ToolPrunePolicy::Never,
        counts_toward_precision_inline_budget: false,
    },
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_basic() {
        let args = serde_json::json!({
            "steps": [
                {
                    "step": 1,
                    "action": "Read src/main.rs to understand structure",
                    "reason": "Need to know entry point before making changes",
                    "tool": "read_file"
                },
                {
                    "step": 2,
                    "action": "Apply patch to fix the bug",
                    "reason": "Fix the identified issue",
                    "tool": "apply_patch"
                }
            ],
            "summary": "Fix bug in main.rs"
        });
        let result = execute_plan(&args).unwrap();
        assert!(result.contains("Fix bug in main.rs"));
        assert!(result.contains("Step 1."));
        assert!(result.contains("Step 2."));
        assert!(result.contains("read_file"));
        assert!(result.contains("apply_patch"));
    }

    #[test]
    fn test_plan_empty_steps() {
        let args = serde_json::json!({
            "steps": []
        });
        let result = execute_plan(&args);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("at least one step"));
    }

    #[test]
    fn test_plan_missing_steps() {
        let args = serde_json::json!({
            "summary": "no steps"
        });
        let result = execute_plan(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_plan_with_parallel_and_delegate() {
        let args = serde_json::json!({
            "summary": "Parallel fix across two modules",
            "steps": [
                {
                    "step": 1,
                    "action": "Fix module A",
                    "tool": "apply_patch",
                    "parallelizable": true,
                    "delegate": true
                },
                {
                    "step": 2,
                    "action": "Fix module B",
                    "tool": "apply_patch",
                    "parallelizable": true,
                    "delegate": true
                }
            ]
        });
        let result = execute_plan(&args).unwrap();
        // Explicit parallelizable delegated steps should use concurrent task_spawn branches.
        assert!(result.contains("||"));
        assert!(result.contains("[delegate]"));
        assert!(result.contains("2 step(s) marked for delegation."));
        assert!(result.contains("2 step(s) can run in parallel."));
        assert!(result.contains("task_spawn"));
        assert!(!result.contains("Proceed to execute."));
    }

    #[test]
    fn test_plan_delegate_without_parallelizable_stays_serial() {
        // delegate=true without explicit parallelizable: the step is SERIAL — it must not
        // get the parallel prefix or parallel counting, and guidance routes it to the
        // sequential synchronous `task`, not concurrent task_spawn.
        let args = serde_json::json!({
            "summary": "Delegate a single independent module fix",
            "steps": [
                {
                    "step": 1,
                    "action": "Fix module A",
                    "tool": "apply_patch",
                    "delegate": true
                }
            ]
        });
        let result = execute_plan(&args).unwrap();
        assert!(!result.contains("||"));
        assert!(result.contains("[delegate]"));
        assert!(result.contains("1 step(s) marked for delegation."));
        assert!(!result.contains("can run in parallel."));
        // Serial delegated steps use synchronous `task` calls one at a time.
        assert!(result.contains("synchronous `task`"));
        assert!(result.contains("one at a time"));
        assert!(!result.contains("task_spawn"));
        assert!(!result.contains("Proceed to execute."));
    }

    #[test]
    fn test_plan_no_delegate_no_parallel_advises_proceed_to_execute() {
        // Zero delegated steps: guidance falls back to "Proceed to execute." but nudges
        // the model to reconsider delegating substantive steps.
        let args = serde_json::json!({
            "summary": "Sequential read then patch",
            "steps": [
                {
                    "step": 1,
                    "action": "Read file A",
                    "tool": "read_file"
                },
                {
                    "step": 2,
                    "action": "Patch file A",
                    "tool": "apply_patch"
                }
            ]
        });
        let result = execute_plan(&args).unwrap();
        assert!(result.contains("Proceed to execute"));
        assert!(result.contains("reconsider"));
        assert!(!result.contains("task_spawn"));
        assert!(!result.contains("[delegate]"));
    }

    #[test]
    fn test_plan_multiple_parallel_delegates_advise_concurrent_task_spawn() {
        // Two delegated steps with explicit parallelizable: guidance recommends concurrent
        // task_spawn + a single task_wait, not the synchronous `task`.
        let args = serde_json::json!({
            "summary": "Two independent module fixes",
            "steps": [
                { "step": 1, "action": "Fix module A", "tool": "apply_patch", "delegate": true, "parallelizable": true },
                { "step": 2, "action": "Fix module B", "tool": "apply_patch", "delegate": true, "parallelizable": true }
            ]
        });
        let result = execute_plan(&args).unwrap();
        assert!(result.contains("2 step(s) marked for delegation."));
        assert!(result.contains("concurrently via task_spawn"));
        assert!(result.contains("single task_wait"));
        assert!(!result.contains("synchronous `task`"));
    }

    #[test]
    fn test_plan_multiple_serial_delegates_advise_sequential_task() {
        // Two delegated steps WITHOUT parallelizable are dependent/serial: they must run
        // one at a time via the synchronous `task` — never concurrently via task_spawn.
        let args = serde_json::json!({
            "summary": "Two dependent module fixes",
            "steps": [
                { "step": 1, "action": "Fix module A", "tool": "apply_patch", "delegate": true },
                { "step": 2, "action": "Fix module B", "tool": "apply_patch", "delegate": true }
            ]
        });
        let result = execute_plan(&args).unwrap();
        assert!(result.contains("2 step(s) marked for delegation."));
        assert!(result.contains("one at a time with the synchronous `task`"));
        assert!(result.contains("never run dependent steps concurrently"));
        assert!(!result.contains("can run in parallel."));
        assert!(!result.contains("task_spawn"));
    }

    #[test]
    fn test_plan_mixed_parallel_and_serial_delegates() {
        // Mixed plans should recommend task_spawn for the parallel step and
        // synchronous `task` for the serial step.
        let args = serde_json::json!({
            "summary": "One independent fix plus one dependent step",
            "steps": [
                { "step": 1, "action": "Fix module A", "tool": "apply_patch", "delegate": true, "parallelizable": true },
                { "step": 2, "action": "Fix module B", "tool": "apply_patch", "delegate": true }
            ]
        });
        let result = execute_plan(&args).unwrap();
        assert!(result.contains("Spawn the single parallel delegated step via task_spawn"));
        assert!(result.contains("one at a time with the synchronous `task`"));
        assert!(result.contains("1 step(s) can run in parallel."));
        assert!(result.contains("2 step(s) marked for delegation."));
    }

    #[test]
    fn test_replan_echo_keeps_completed_steps_status() {
        use crate::ai::driver::runtime_ctx::{DRIVER_CTX, DriverContext};
        use crate::ai::mcp::{McpClient, SharedMcpClient};
        use crate::ai::tools::plan_state::StepStatus;
        use std::sync::{Arc, Mutex};
        use std::time::{SystemTime, UNIX_EPOCH};

        // The drop guard cleans the temporary directory even when an assertion panics.
        struct TempDirGuard(std::path::PathBuf);
        impl Drop for TempDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        // Regression: after plan_update marks a step done, invoking `plan` again must
        // preserve the persisted status suffix and progress line. Rendering only the
        // raw arguments made completed steps appear pending after replanning.
        let base = std::env::temp_dir().join(format!(
            "plan-tools-replan-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let _guard = TempDirGuard(base.clone());
        let mut app = crate::ai::middleware::test_util::test_app();
        // The plan state path is derived from `session_history_file`, not
        // `config.history_file`. Keep it under the temporary directory to prevent
        // state leaking through the shared `default.assets/` fallback between runs.
        app.session_history_file = base.join("session.sqlite");
        app.session_id = "plan-tools-replan-test".to_string();

        let args = serde_json::json!({
            "summary": "Sequential read then patch",
            "steps": [
                { "step": 1, "action": "Read file A", "tool": "read_file" },
                { "step": 2, "action": "Patch file A", "tool": "apply_patch" }
            ]
        });

        let ctx = DriverContext::new(
            app.clone(),
            Arc::new(Mutex::new(McpClient::new())) as SharedMcpClient,
            Arc::new(Vec::new()),
            Arc::new(Vec::new()),
        );

        let result = DRIVER_CTX.sync_scope(ctx, || {
            // A new plan starts with every step pending and no status suffix.
            let first = execute_plan(&args).unwrap();
            assert!(!first.contains("(done)"));
            assert!(!first.contains("Progress: 1/2 steps done."));

            // Persist step 1 through the same load-mutate-save path as plan_update.
            crate::ai::tools::plan_state::update_plan_step(&app, 1, StepStatus::Done, None)
                .unwrap();

            // Replanning with the same arguments must preserve the completed suffix
            // and progress line while leaving the pending step suffix-free.
            let replan = execute_plan(&args).unwrap();
            assert!(
                replan.contains("Step 1. [read_file] Read file A (done)"),
                "completed step must keep (done) suffix in re-plan echo:\n{replan}"
            );
            assert!(
                replan.contains("Progress: 1/2 steps done."),
                "re-plan echo must show progress line:\n{replan}"
            );
            assert!(
                replan.contains("Step 2. [apply_patch] Patch file A\n"),
                "pending step must stay suffix-free in re-plan echo:\n{replan}"
            );
            replan
        });

        assert!(
            result.contains("(done)"),
            "re-plan echo must keep (done) for the completed step:\n{result}"
        );
    }
}
