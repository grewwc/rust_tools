use serde_json::Value;
use std::process::Command;

use crate::ai::tools::common::{ToolRegistration, ToolSpec};

fn params_task() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short description of what this task will do (3-10 words)."
            },
            "prompt": {
                "type": "string",
                "description": "The task/prompt to send to the subagent. Be specific about what you want accomplished."
            },
            "agent": {
                "type": "string",
                "description": "Which subagent to use (e.g., 'explore', 'general'). Defaults to 'general' if not specified."
            },
            "model": {
                "type": "string",
                "description": "Optional model override for this subagent task."
            }
        },
        "required": ["description", "prompt"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task",
        description: "Launch a specialized subagent to handle a specific task in parallel. Use this for complex multi-step work, codebase exploration, or tasks that benefit from specialized agents. The subagent runs independently and returns its full response.",
        parameters: params_task,
        execute: execute_task,
        groups: &["builtin"],
    }
});

pub(crate) fn execute_task(args: &Value) -> Result<String, String> {
    let description = args["description"]
        .as_str()
        .ok_or("Missing 'description' parameter")?;

    let prompt = args["prompt"]
        .as_str()
        .ok_or("Missing 'prompt' parameter")?;

    let agent = args["agent"].as_str().unwrap_or("general");
    let _model = args["model"].as_str();

    if description.trim().is_empty() {
        return Err("description cannot be empty".to_string());
    }

    if prompt.trim().is_empty() {
        return Err("prompt cannot be empty".to_string());
    }

    execute_subagent_task(description, prompt, agent, _model)
}

fn execute_subagent_task(
    description: &str,
    prompt: &str,
    agent: &str,
    model: Option<&str>,
) -> Result<String, String> {
    use std::time::Instant;

    let start = Instant::now();

    println!(
        "\n[Task] Launching subagent '{}' for: {}",
        agent, description
    );

    let mut cmd_args = vec!["--".to_string(), "--no-skills".to_string()];

    if let Some(m) = model {
        cmd_args.push("--model".to_string());
        cmd_args.push(m.to_string());
    }

    cmd_args.push(prompt.to_string());

    let output = Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
        .args(&cmd_args)
        .output()
        .map_err(|e| format!("Failed to launch subagent: {}", e))?;

    let duration = start.elapsed();

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result = format!(
            "[Task: {}] (completed in {:.1}s)\n{}",
            description,
            duration.as_secs_f64(),
            stdout.trim()
        );
        Ok(result)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!(
            "[Task: {}] failed after {:.1}s:\n{}",
            description,
            duration.as_secs_f64(),
            stderr.trim()
        ))
    }
}

fn params_question() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user."
            },
            "header": {
                "type": "string",
                "description": "Very short label (max 30 chars) for context."
            },
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Display text for this option (1-5 words)."
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief explanation of what this option means."
                        }
                    },
                    "required": ["label", "description"]
                },
                "description": "Available choices for the user."
            },
            "multiple": {
                "type": "boolean",
                "description": "Allow selecting multiple choices (default: false)."
            }
        },
        "required": ["question", "header", "options"]
    })
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "question",
        description: "Ask the user questions during execution. Use this to gather preferences, clarify ambiguous instructions, get decisions on implementation choices, or offer choices about direction. Returns the user's selected answer(s).",
        parameters: params_question,
        execute: execute_question,
        groups: &["builtin"],
    }
});

pub(crate) fn execute_question(args: &Value) -> Result<String, String> {
    let question = args["question"]
        .as_str()
        .ok_or("Missing 'question' parameter")?;

    let header = args["header"]
        .as_str()
        .ok_or("Missing 'header' parameter")?;

    let options = args["options"]
        .as_array()
        .ok_or("Missing 'options' parameter (must be an array)")?;

    if options.is_empty() {
        return Err("options array cannot be empty".to_string());
    }

    let multiple = args["multiple"].as_bool().unwrap_or(false);

    println!("\n--- Question: {} ---", header);
    println!("{}", question);
    println!();

    for (i, opt) in options.iter().enumerate() {
        let label = opt["label"].as_str().unwrap_or("?");
        let desc = opt["description"].as_str().unwrap_or("");
        println!("  {}. {} - {}", i + 1, label, desc);
    }
    println!();

    if multiple {
        println!("Enter option numbers separated by commas (or type your own answer):");
    } else {
        println!("Enter option number (or type your own answer):");
    }

    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .map_err(|e| format!("Failed to read input: {}", e))?;

    let input = input.trim();

    if input.is_empty() {
        return Err("No answer provided".to_string());
    }

    if multiple {
        let selections: Vec<&str> = input.split(',').map(|s| s.trim()).collect();
        let mut selected_labels = Vec::new();

        for sel in &selections {
            if let Ok(idx) = sel.parse::<usize>() {
                if idx > 0 && idx <= options.len() {
                    if let Some(label) = options[idx - 1]["label"].as_str() {
                        selected_labels.push(label.to_string());
                    }
                } else {
                    return Ok(format!("[User answer] {}", input));
                }
            } else {
                return Ok(format!("[User answer] {}", input));
            }
        }

        Ok(format!("[User selected] {}", selected_labels.join(", ")))
    } else {
        if let Ok(idx) = input.parse::<usize>() {
            if idx > 0 && idx <= options.len() {
                if let Some(label) = options[idx - 1]["label"].as_str() {
                    return Ok(format!("[User selected] {}", label));
                }
            }
        }

        Ok(format!("[User answer] {}", input))
    }
}
