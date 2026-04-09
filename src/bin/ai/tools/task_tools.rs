use serde_json::Value;
use std::process::Command;

use crate::ai::{
    agents::{self, AgentManifest, AgentModelTier},
    models,
    tools::common::{ToolRegistration, ToolSpec},
};

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
                "description": "Optional subagent name. Leave empty to let the runtime auto-select the best subagent for this task."
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
        description: "Launch a specialized subagent to handle a focused task. Use this for complex work, codebase exploration, independent side investigations, or when multiple subtasks can be delegated. If agent is omitted, the runtime auto-selects a suitable subagent.",
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

    let agent = args["agent"].as_str().map(str::trim).filter(|s| !s.is_empty());
    let model_override = args["model"].as_str();

    if description.trim().is_empty() {
        return Err("description cannot be empty".to_string());
    }

    if prompt.trim().is_empty() {
        return Err("prompt cannot be empty".to_string());
    }

    execute_subagent_task(description, prompt, agent, model_override)
}

fn auto_subagent_score(agent: &AgentManifest, task_text: &str) -> i32 {
    let task = task_text.to_ascii_lowercase();
    let mut score = 0i32;

    for tag in agent.routing_tags_normalized() {
        if tag.is_empty() {
            continue;
        }
        if task.contains(&tag) {
            score += 24;
        } else if tag.contains('-') || tag.contains(' ') {
            let parts = tag
                .split(['-', ' '])
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>();
            if !parts.is_empty() && parts.iter().all(|part| task.contains(part)) {
                score += 14;
            }
        }
    }

    score
}

#[derive(Debug)]
struct SelectedSubagent<'a> {
    agent: &'a AgentManifest,
    auto_selected: bool,
    matched_tags: Vec<String>,
    score: i32,
}

fn matched_routing_tags(agent: &AgentManifest, task_text: &str) -> Vec<String> {
    let task = task_text.to_ascii_lowercase();
    agent
        .routing_tags_normalized()
        .into_iter()
        .filter(|tag| {
            if task.contains(tag) {
                return true;
            }
            if tag.contains('-') || tag.contains(' ') {
                let parts = tag
                    .split(['-', ' '])
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .collect::<Vec<_>>();
                return !parts.is_empty() && parts.iter().all(|part| task.contains(part));
            }
            false
        })
        .collect()
}

fn select_subagent<'a>(
    all_agents: &'a [AgentManifest],
    requested_agent: Option<&str>,
    description: &str,
    prompt: &str,
) -> Result<SelectedSubagent<'a>, String> {
    let subagents = agents::get_subagents(all_agents);
    if subagents.is_empty() {
        return Err("No subagents are available. Add at least one agent with mode: subagent or all."
            .to_string());
    }

    if let Some(requested) = requested_agent {
        if let Some(agent) = subagents
            .iter()
            .copied()
            .find(|agent| agent.name.eq_ignore_ascii_case(requested))
        {
            return Ok(SelectedSubagent {
                agent,
                auto_selected: false,
                matched_tags: Vec::new(),
                score: 0,
            });
        }

        if let Some(agent) = agents::find_agent_by_name(all_agents, requested) {
            return Err(format!(
                "Agent '{}' exists but is not a subagent. Use a subagent or omit the agent field for auto-selection.",
                agent.name
            ));
        }

        let available = subagents
            .iter()
            .map(|agent| agent.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "Unknown subagent '{}'. Available subagents: {}",
            requested, available
        ));
    }

    let task_text = format!("{description}\n{prompt}");
    subagents
        .into_iter()
        .max_by(|a, b| {
            auto_subagent_score(a, &task_text)
                .cmp(&auto_subagent_score(b, &task_text))
                .then_with(|| b.name.cmp(&a.name))
        })
        .map(|agent| SelectedSubagent {
            agent,
            auto_selected: true,
            matched_tags: matched_routing_tags(agent, &task_text),
            score: auto_subagent_score(agent, &task_text),
        })
        .ok_or_else(|| "No subagents are available.".to_string())
}

fn format_agent_model_tier(agent: &AgentManifest) -> &'static str {
    match agent.model_tier {
        Some(AgentModelTier::Light) => "light",
        Some(AgentModelTier::Standard) | None => "standard",
        Some(AgentModelTier::Heavy) => "heavy",
    }
}

fn format_quality_tier(tier: crate::ai::provider::ModelQualityTier) -> &'static str {
    match tier {
        crate::ai::provider::ModelQualityTier::Basic => "basic",
        crate::ai::provider::ModelQualityTier::Standard => "standard",
        crate::ai::provider::ModelQualityTier::Strong => "strong",
        crate::ai::provider::ModelQualityTier::Flagship => "flagship",
    }
}

fn format_provider(provider: crate::ai::provider::ApiProvider) -> &'static str {
    match provider {
        crate::ai::provider::ApiProvider::Compatible => "compatible",
        crate::ai::provider::ApiProvider::OpenAi => "openai",
    }
}

fn build_selection_explanation(
    selected: &SelectedSubagent<'_>,
    selected_model: &str,
    model_override: Option<&str>,
) -> String {
    let agent_reason = if selected.auto_selected {
        if selected.matched_tags.is_empty() {
            "agent_reason=auto-selected as the best available subagent".to_string()
        } else {
            format!(
                "agent_reason=auto-selected by routing_tags [{}] (score={})",
                selected.matched_tags.join(", "),
                selected.score
            )
        }
    } else {
        "agent_reason=explicit agent override".to_string()
    };

    let model_reason = if model_override.map(str::trim).filter(|value| !value.is_empty()).is_some() {
        "model_reason=explicit model override".to_string()
    } else {
        format!(
            "model_reason=auto-selected for agent_tier={} using {} provider and {} quality_tier",
            format_agent_model_tier(selected.agent),
            format_provider(models::model_provider(selected_model)),
            format_quality_tier(models::model_quality_tier(selected_model))
        )
    };

    format!("{agent_reason}\n{model_reason}")
}

fn execute_subagent_task(
    description: &str,
    prompt: &str,
    agent: Option<&str>,
    model: Option<&str>,
) -> Result<String, String> {
    use std::time::Instant;

    let start = Instant::now();
    let all_agents = agents::load_all_agents();
    let selected = select_subagent(&all_agents, agent, description, prompt)?;
    let selected_model = model
        .map(models::determine_model)
        .unwrap_or_else(|| models::auto_subagent_model_for_agent(selected.agent, description, prompt));
    let selection_explanation = build_selection_explanation(&selected, &selected_model, model);

    println!(
        "\n[Task] Launching subagent '{}' with model '{}' for: {}\n{}",
        selected.agent.name, selected_model, description, selection_explanation
    );

    let mut cmd_args = vec!["--".to_string(), "--no-skills".to_string()];

    cmd_args.push("--model".to_string());
    cmd_args.push(selected_model.clone());
    cmd_args.push("--agent".to_string());
    cmd_args.push(selected.agent.name.clone());
    cmd_args.push(prompt.to_string());

    let output = Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
        .args(&cmd_args)
        .output()
        .map_err(|e| format!("Failed to launch subagent: {}", e))?;

    let duration = start.elapsed();

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result = format!(
            "[Task: {} via {} @ {}] (completed in {:.1}s)\n{}\n{}",
            description,
            selected.agent.name,
            selected_model,
            duration.as_secs_f64(),
            selection_explanation,
            stdout.trim()
        );
        Ok(result)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!(
            "[Task: {} via {} @ {}] failed after {:.1}s:\n{}\n{}",
            description,
            selected.agent.name,
            selected_model,
            duration.as_secs_f64(),
            selection_explanation,
            stderr.trim()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{SelectedSubagent, build_selection_explanation, select_subagent};
    use crate::ai::agents::{AgentManifest, AgentMode, AgentModelTier};

    fn manifest(name: &str, description: &str, mode: AgentMode) -> AgentManifest {
        AgentManifest {
            name: name.to_string(),
            description: description.to_string(),
            mode,
            model: None,
            temperature: None,
            max_steps: None,
            prompt: String::new(),
            system_prompt: None,
            tools: Vec::new(),
            tool_groups: Vec::new(),
            mcp_servers: Vec::new(),
            routing_tags: Vec::new(),
            model_tier: Some(AgentModelTier::Standard),
            disabled: false,
            hidden: false,
            color: None,
            source_path: None,
        }
    }

    #[test]
    fn auto_select_prefers_explore_for_codebase_investigation() {
        let mut build = manifest("build", "Main build agent", AgentMode::Primary);
        build.routing_tags = vec!["implement".to_string(), "fix".to_string()];
        build.model_tier = Some(AgentModelTier::Heavy);
        let mut explore = manifest(
            "explore",
            "Read-only codebase exploration agent",
            AgentMode::Subagent,
        );
        explore.routing_tags = vec![
            "find".to_string(),
            "search".to_string(),
            "read-only".to_string(),
            "understand".to_string(),
        ];
        explore.model_tier = Some(AgentModelTier::Light);
        let mut review = manifest("review", "Read-only review agent", AgentMode::Subagent);
        review.routing_tags = vec!["review".to_string(), "audit".to_string()];

        let all_agents = vec![build, explore, review];

        let selected = select_subagent(
            &all_agents,
            None,
            "Locate routing logic",
            "Find where automatic agent routing happens and summarize the files involved.",
        )
        .unwrap();

        assert_eq!(selected.agent.name, "explore");
        assert!(selected.auto_selected);
        assert!(!selected.matched_tags.is_empty());
    }

    #[test]
    fn explicit_primary_agent_is_rejected_for_task_tool() {
        let mut build = manifest("build", "Main build agent", AgentMode::Primary);
        build.routing_tags = vec!["implement".to_string()];
        let mut explore = manifest(
            "explore",
            "Read-only codebase exploration agent",
            AgentMode::Subagent,
        );
        explore.routing_tags = vec!["find".to_string(), "search".to_string()];
        let all_agents = vec![build, explore];

        let err = select_subagent(&all_agents, Some("build"), "Inspect code", "Look up files")
            .unwrap_err();

        assert!(err.contains("not a subagent"));
    }

    #[test]
    fn routing_tags_drive_auto_selection_without_name_special_cases() {
        let mut explore = manifest(
            "navigator",
            "Read-only codebase exploration agent",
            AgentMode::Subagent,
        );
        explore.routing_tags = vec!["find".to_string(), "search".to_string(), "locate".to_string()];
        explore.model_tier = Some(AgentModelTier::Light);

        let mut review = manifest("critic", "Code review agent", AgentMode::Subagent);
        review.routing_tags = vec!["review".to_string(), "audit".to_string()];
        let all_agents = vec![explore, review];

        let selected = select_subagent(
            &all_agents,
            None,
            "Find handler",
            "Search the codebase and locate where the request handler is defined.",
        )
        .unwrap();

        assert_eq!(selected.agent.name, "navigator");
    }

    #[test]
    fn selection_explanation_mentions_quality_tier_for_auto_model_choice() {
        let agent = manifest("build", "Main build agent", AgentMode::Subagent);
        let selected = SelectedSubagent {
            agent: &agent,
            auto_selected: true,
            matched_tags: vec!["implement".to_string(), "fix".to_string()],
            score: 48,
        };

        let explanation = build_selection_explanation(&selected, "qwen3-max", None);

        assert!(explanation.contains("routing_tags [implement, fix]"));
        assert!(explanation.contains("quality_tier"));
        assert!(explanation.contains("flagship"));
        assert!(explanation.contains("compatible"));
    }

    #[test]
    fn selection_explanation_mentions_explicit_overrides() {
        let agent = manifest("explore", "Read-only codebase exploration agent", AgentMode::Subagent);
        let selected = SelectedSubagent {
            agent: &agent,
            auto_selected: false,
            matched_tags: Vec::new(),
            score: 0,
        };

        let explanation = build_selection_explanation(&selected, "gpt-4o", Some("gpt-4o"));

        assert!(explanation.contains("explicit agent override"));
        assert!(explanation.contains("explicit model override"));
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
