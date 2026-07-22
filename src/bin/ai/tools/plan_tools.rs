use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;
use crate::ai::tools::common::{ToolDisplayConfig, ToolDisplayRegistration};
use crate::ai::tools::common::{
    ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
};

fn params_plan() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {
                            "type": "integer",
                            "description": "Step number (1-based)."
                        },
                        "action": {
                            "type": "string",
                            "description": "What to do in this step (e.g., 'read src/main.rs', 'run cargo build', 'apply patch to fix X')."
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why this step is needed and what we expect to learn or achieve."
                        },
                        "tool": {
                            "type": "string",
                            "description": "The primary tool you plan to use for this step (e.g., 'read_file', 'execute_command', 'apply_patch', 'web_search'). Use 'none' if no tool is needed."
                        },
                        "parallelizable": {
                            "type": "boolean",
                            "description": "Whether this step can run in parallel with the previous step (no data dependency). Default: false."
                        },
                        "delegate": {
                            "type": "boolean",
                            "description": "Whether this step should be delegated to a subagent via task_spawn. A delegated step implies `parallelizable: true` (a subagent should not block the parent synchronously), so `delegate: true` automatically counts as parallelizable; set `parallelizable` explicitly only when it is true without delegation."
                        }
                    },
                    "required": ["step", "action"]
                },
                "description": "Ordered list of steps to accomplish the task."
            },
            "summary": {
                "type": "string",
                "description": "Brief one-line summary of the overall plan."
            }
        },
        "required": ["steps"]
    })
}

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

        // delegate implies parallelizable — subagent dispatch is inherently async, a
        // delegate=true without parallelizable=true would block the parent unnecessarily.
        let delegate = step_obj
            .get("delegate")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let parallelizable = step_obj
            .get("parallelizable")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            || delegate;

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

    // 统计可委派/可并行步骤。delegate 自动计入 parallelizable。
    let delegate_count: usize = steps
        .iter()
        .filter_map(|s| s.get("delegate").and_then(|v| v.as_bool()))
        .filter(|&b| b)
        .count();
    let parallel_count: usize = steps
        .iter()
        .filter_map(|s| {
            let d = s.get("delegate").and_then(|v| v.as_bool()).unwrap_or(false);
            let p = s.get("parallelizable").and_then(|v| v.as_bool()).unwrap_or(false);
            Some(d || p)
        })
        .filter(|&b| b)
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
    // 仅当既有 delegate 步又有真正可并行步时才建议并行 spawn；只有 delegate 但
    // 没有 parallelizable 步时（应由 schema/delegate 蕴含规则避免，但仍兜底）应
    // 单步派发而非误导模型并行 spawn 各步。
    if delegate_count > 0 && parallel_count > 0 {
        formatted.push_str(" Launch delegated steps via task_spawn in parallel, then task_wait to collect results.");
    } else if delegate_count > 0 {
        formatted.push_str(" Launch delegated steps via task_spawn and collect with task_wait.");
    } else {
        formatted.push_str(" Proceed to execute.");
    }
    formatted.push('\n');

    Ok(formatted)
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "plan",
        description: "Create a step-by-step plan for complex tasks. Use this BEFORE executing tools when a task has multiple steps, involves unfamiliar code, or requires coordination across files/systems. Each step should specify what to do, why, and which tool to use. A step marked `delegate: true` implies `parallelizable: true` (delegation is inherently async). Simple tasks (read one file, answer a question, run one command) do NOT need a plan - just act directly.",
        parameters: params_plan,
        execute: execute_plan,
        async_policy: crate::ai::tools::common::ToolAsyncPolicy::SyncOnly,
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
        // delegate=true should imply parallelizable=true (both prefix same), and the
        // delegate-guidance branch must fire because delegate>0 AND parallel>0.
        assert!(result.contains("||"));
        assert!(result.contains("[delegate]"));
        assert!(result.contains("2 step(s) marked for delegation."));
        assert!(result.contains("2 step(s) can run in parallel."));
        assert!(result.contains("task_spawn"));
        assert!(!result.contains("Proceed to execute."));
    }

    #[test]
    fn test_plan_delegate_without_explicit_parallelizable_counts_as_parallel() {
        // delegate=true without explicit parallelizable: the implicit-implication rule
        // must still treat the step as parallelizable, so parallel-aware guidance fires.
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
        // delegate-only step still gets parallel prefix and parallel-aware guidance.
        assert!(result.contains("||"));
        assert!(result.contains("[delegate]"));
        assert!(result.contains("1 step(s) marked for delegation."));
        assert!(result.contains("1 step(s) can run in parallel."));
        assert!(result.contains("task_spawn"));
        assert!(!result.contains("Proceed to execute."));
    }

    #[test]
    fn test_plan_delegate_one_step_no_parallelizable_does_not_advise_parallel_spawn() {
        // Old bug: when only delegate steps exist (>=1) and parallel_count happened to
        // be 0, the old code still printed "in parallel" unconditionally. Confirm
        // the gating now requires parallel_count>0 for the parallel-spawn advice.
        // 这里通过 neither delegate nor parallelizable 制造零 delegate + 零 parallel 的
        // fallback 分支，验证提示回到 "Proceed to execute"。
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
        assert!(result.contains("Proceed to execute."));
        assert!(!result.contains("task_spawn"));
        assert!(!result.contains("[delegate]"));
    }
}
