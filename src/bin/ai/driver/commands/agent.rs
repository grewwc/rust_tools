use crate::ai::{
    agents::{self, AgentManifest},
    types::App,
};

pub fn try_handle_agent_command(
    app: &mut App,
    input: &str,
    agent_manifests: &[AgentManifest],
) -> Result<bool, Box<dyn std::error::Error>> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let normalized = if let Some(rest) = trimmed.strip_prefix('/') {
        rest
    } else if let Some(rest) = trimmed.strip_prefix(':') {
        rest
    } else {
        return Ok(false);
    };
    let mut parts = normalized.split_whitespace();
    let Some(cmd) = parts.next() else {
        return Ok(false);
    };
    if cmd != "agents" && cmd != "agent" {
        return Ok(false);
    }
    let action = parts.next().unwrap_or("list");

    match action {
        "help" | "h" => {
            println!("agent commands:");
            println!("  /agents");
            println!("  /agents list");
            println!("  /agents current");
            println!("  /agents use <name>");
        }
        "list" | "ls" => {
            let primary_agents = agents::get_primary_agents(agent_manifests);
            let subagents = agents::get_subagents(agent_manifests);

            if !primary_agents.is_empty() {
                println!("\nPrimary agents (Tab to switch):");
                for agent in &primary_agents {
                    let mark = if agent.name == app.current_agent {
                        ">>>"
                    } else {
                        "   "
                    };
                    let color_info = agent.color.as_deref().unwrap_or("default");
                    println!(
                        "  {} {} [{}] - {}",
                        mark, agent.name, color_info, agent.description
                    );
                }
            }

            if !subagents.is_empty() {
                println!("\nSubagents (@mention to invoke):");
                for agent in &subagents {
                    println!("    {} - {}", agent.name, agent.description);
                }
            }
            println!();
        }
        "current" | "cur" => {
            if let Some(agent) = agents::find_agent_by_name(agent_manifests, &app.current_agent) {
                println!("Current agent: {}", app.current_agent);
                println!("Description: {}", agent.description);
                println!("Mode: {:?}", agent.mode);
                if let Some(model) = &agent.model {
                    println!("Model: {}", model);
                }
                if let Some(color) = &agent.color {
                    println!("Color: {}", color);
                }
            } else {
                println!(
                    "Current agent: {} (not found in manifests)",
                    app.current_agent
                );
            }
        }
        "use" | "select" | "switch" => {
            let Some(name) = parts.next() else {
                println!("missing agent name. try: /agents use <name>");
                println!("\nAvailable primary agents:");
                for agent in agents::get_primary_agents(agent_manifests) {
                    println!("  {} - {}", agent.name, agent.description);
                }
                return Ok(true);
            };

            if let Some(agent) = agents::find_agent_by_name(agent_manifests, name) {
                if !agent.is_primary() {
                    println!(
                        "Agent '{}' is not a primary agent. Use @mention to invoke subagents.",
                        name
                    );
                    return Ok(true);
                }
                if agent.disabled {
                    println!("Agent '{}' is disabled.", name);
                    return Ok(true);
                }

                let old_agent = app.current_agent.clone();
                app.current_agent = agent.name.clone();

                if let Some(model) = &agent.model {
                    app.current_model = model.clone();
                }

                println!("Switched agent: {} -> {}", old_agent, agent.name);
                if let Some(model) = &agent.model {
                    println!("Model: {}", model);
                }
            } else {
                println!("Agent not found: {}", name);
                println!("\nAvailable primary agents:");
                for agent in agents::get_primary_agents(agent_manifests) {
                    println!("  {} - {}", agent.name, agent.description);
                }
            }
        }
        _ => {
            println!("unknown action: {}. try: /agents help", action);
        }
    }
    Ok(true)
}
