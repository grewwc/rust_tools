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

    let mut formatted = String::new();
    if !summary.is_empty() {
        formatted.push_str(&format!("Plan: {}\n\n", summary));
    }

    for step_val in steps {
        let step_obj = step_val
            .as_object()
            .ok_or("Each step must be a JSON object.")?;

        let step_num = step_obj
            .get("step")
            .and_then(|v| v.as_u64())
            .ok_or("Each step must have an integer 'step' field.")?;

        let action = step_obj
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or("Each step must have an 'action' string field.")?;

        let reason = step_obj
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let tool = step_obj
            .get("tool")
            .and_then(|v| v.as_str())
            .unwrap_or("unspecified");

        // delegate 与 parallelizable 相互独立：delegate 表示"该步交给 subagent 做"（串行或
        // 并行均可）；parallelizable 表示"与前置步无数据依赖，可并发"（显式声明，不再由
        // delegate 蕴含）。串行委派步应逐个走同步 `task`，绝不能因 delegate 被当成并发步。
        let delegate = step_obj
            .get("delegate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let parallelizable = step_obj
            .get("parallelizable")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let prefix = if parallelizable { "  || " } else { "" };
        let tags = if delegate { " [delegate]" } else { "" };
        formatted.push_str(&format!(
            "{}Step {}. [{}]{} {}\n",
            prefix, step_num, tool, tags, action
        ));
        if !reason.is_empty() {
            formatted.push_str(&format!("  Reason: {}\n", reason));
        }
    }

    // 统计可委派/可并行步骤。delegate 不再自动计入 parallelizable。
    let delegate_count: usize = steps
        .iter()
        .filter_map(|s| s.get("delegate").and_then(|v| v.as_bool()))
        .filter(|&b| b)
        .count();
    let parallel_count: usize = steps
        .iter()
        .filter_map(|s| s.get("parallelizable").and_then(|v| v.as_bool()))
        .filter(|&b| b)
        .count();
    let parallel_delegates: usize = steps
        .iter()
        .filter(|s| {
            s.get("delegate").and_then(|v| v.as_bool()).unwrap_or(false)
                && s.get("parallelizable")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
        })
        .count();
    let serial_delegates: usize = steps
        .iter()
        .filter(|s| {
            s.get("delegate").and_then(|v| v.as_bool()).unwrap_or(false)
                && !s
                    .get("parallelizable")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
        })
        .count();

    formatted.push_str(&format!("\n---\n{} step(s) planned.", steps.len()));
    if delegate_count > 0 {
        formatted.push_str(&format!(
            " {} step(s) marked for delegation.",
            delegate_count
        ));
    }
    if parallel_count > 0 {
        formatted.push_str(&format!(" {} step(s) can run in parallel.", parallel_count));
    }
    // 委派派发建议：并行委派步（delegate && parallelizable）走并发 task_spawn；串行委派步
    // （delegate && !parallelizable）逐个走同步 `task`，父进程在 prompt 中传递所需上下文。
    if delegate_count > 0 {
        if parallel_delegates >= 2 {
            formatted.push_str(" Launch the parallel delegated steps concurrently via task_spawn, then collect with a single task_wait.");
        } else if parallel_delegates == 1 && serial_delegates == 0 {
            formatted.push_str(" Run the single delegated step with the synchronous `task` tool (async spawn+wait adds overhead without concurrency for one task).");
        } else if parallel_delegates == 1 {
            formatted.push_str(" Spawn the single parallel delegated step via task_spawn while you run the serial ones, then collect it with task_wait.");
        }
        if serial_delegates > 0 {
            formatted.push_str(" Run serial delegated steps one at a time with the synchronous `task`, passing the needed context from prior results in the prompt; never run dependent steps concurrently.");
        }
    } else {
        formatted.push_str(" Proceed to execute, but reconsider: any substantive step with real intermediate reads or commands is usually better delegated to a subagent (cleaner, focused context); keep only trivial single-tool steps and final review in the parent.");
    }
    formatted.push('\n');

    Ok(formatted)
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
}
