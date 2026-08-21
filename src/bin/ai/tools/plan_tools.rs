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

    // 渲染走统一路径：按当前参数构建状态模型并经 `PlanState::render` 输出（与
    // `plan_update` 共用同一渲染路径，不再在 execute_plan 里维护一套重复格式化）。
    // 随后尝试把本次规划登记为会话级状态（plan_update 据此更新）；登记失败不阻塞
    // plan 输出本身，但追加提示行，避免损坏/不可写的状态文件被静默吞没。
    let rendered = crate::ai::tools::plan_state::PlanState::build(summary, steps, None)?.render();
    if let Some(app) = crate::ai::driver::runtime_ctx::try_current().map(|ctx| ctx.app_proto.clone()) {
        match crate::ai::tools::plan_state::record_plan(&app, summary, steps) {
            // 登记成功：按持久化状态渲染（重规划时保留已完成步骤的后缀与进度行）。
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
        groups: &["builtin", "core"],
    }
});

// plan 工具的输出对用户有较高可见性价值，开启结果回显。
inventory::submit!(ToolDisplayRegistration {
    name: "plan",
    config: ToolDisplayConfig {
        print_args: false,
        print_result: true,
        emphasize_result: false,
        display: None,
    },
});

// plan 是任务路线图锚点：最新一版必须完整保留（不受有损压缩，也不被 LLM 裁剪），
// 这由最近工具组保护窗口 (`KEEP_RECENT_TOOL_GROUPS`) 自动实现。旧版 plan 一旦被
// 新版替换，可被有损压缩摘要以释放上下文；但仍禁止 LLM 单方裁剪为占位符，避免
// 模型自己否定既有规划。
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
        // 显式 parallelizable=true + delegate=true：两个并行委派步 → 并发 task_spawn 分支。
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
        // 串行委派步 → 逐个同步 `task`；绝不并发。
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
        // 一个并行委派步 + 一个串行委派步：并行步用 task_spawn 并发跑，串行步逐个同步
        // `task`，两条建议都出现。
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

        // Drop guard：即使断言 panic 也保证临时目录被清理，避免测试失败泄漏垃圾目录。
        struct TempDirGuard(std::path::PathBuf);
        impl Drop for TempDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        // 回归：plan_update 把步骤标记为 done 后，重新调用 `plan`（修订计划、续会话
        // 压缩后重锚定都会触发）的回显必须带持久化的状态后缀与进度行。此前 execute_plan
        // 只按原始参数渲染，重规划后已完成步骤的状态从回显中消失，agent 会误以为进度
        // 丢失而从头开始。
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
        // plan 状态路径由 `session_history_file` 推导（见 plan_state::store::plan_state_path），
        // 不是 `config.history_file`。必须把它指到临时目录内，否则测试会把 plan-state.json
        // 写到共享的 `default.assets/`（session_history_file 为空时的回退路径），造成跨测试
        // 运行的状态泄漏与"首轮通过、次轮必挂"的抖动。
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
            // 首次规划：全新计划，全部 pending，无状态后缀。
            let first = execute_plan(&args).unwrap();
            assert!(!first.contains("(done)"));
            assert!(!first.contains("Progress: 1/2 steps done."));

            // 完成第 1 步并落盘（即 plan_update 走过的 load→mutate→save 路径）。
            crate::ai::tools::plan_state::update_plan_step(
                &app,
                1,
                StepStatus::Done,
                None,
            )
            .unwrap();

            // 重新调用 plan（同参数重新规划）：回显必须保留已完成步骤的 (done) 后缀，
            // 并显示进度行；未完成的第 2 步保持无状态后缀。
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
