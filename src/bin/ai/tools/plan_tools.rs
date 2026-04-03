use serde_json::Value;

use crate::ai::tools::common::ToolRegistration;
use crate::ai::tools::common::ToolSpec;

fn params_plan() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {
                            "type": "integer",
                            "description": "Step number (1-based, sequential)."
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
                            "description": "The primary tool you plan to use for this step (e.g., 'read_file_lines', 'execute_command', 'apply_patch', 'web_search'). Use 'none' if no tool is needed."
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
    let steps = args["steps"].as_array().ok_or("Missing 'steps' array. Provide a JSON array of plan steps.")?;
    let summary = args["summary"].as_str().unwrap_or("");

    if steps.is_empty() {
        return Err("Plan must contain at least one step.".to_string());
    }

    let mut formatted = String::new();
    if !summary.is_empty() {
        formatted.push_str(&format!("Plan: {}\n\n", summary));
    }

    for step_val in steps {
        let step_obj = step_val.as_object().ok_or("Each step must be a JSON object.")?;
        
        let step_num = step_obj.get("step")
            .and_then(|v| v.as_u64())
            .ok_or("Each step must have a numeric 'step' field.")?;
        
        let action = step_obj.get("action")
            .and_then(|v| v.as_str())
            .ok_or("Each step must have a 'action' string field.")?;
        
        let reason = step_obj.get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        
        let tool = step_obj.get("tool")
            .and_then(|v| v.as_str())
            .unwrap_or("unspecified");

        formatted.push_str(&format!("Step {}. [{}] {}\n", step_num, tool, action));
        if !reason.is_empty() {
            formatted.push_str(&format!("  Reason: {}\n", reason));
        }
    }

    formatted.push_str(&format!("\n---\n{} step(s) planned. Proceed to execute.\n", steps.len()));

    Ok(formatted)
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "plan",
        description: "Create a step-by-step plan for complex tasks. Use this BEFORE executing tools when a task has multiple steps, involves unfamiliar code, or requires coordination across files/systems. Each step should specify what to do, why, and which tool to use. Simple tasks (read one file, answer a question, run one command) do NOT need a plan — just act directly.",
        parameters: params_plan,
        execute: execute_plan,
        groups: &["builtin"],
    }
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
                    "tool": "read_file_lines"
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
        assert!(result.contains("read_file_lines"));
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
}
