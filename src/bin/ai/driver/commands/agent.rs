use crate::ai::{
    agents::{self, AgentManifest},
    types::App,
};

pub fn try_handle_agent_command(
    app: &mut App,
    input: &str,
    agent_manifests: &mut Vec<AgentManifest>,
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
            println!("Agent management commands:");
            println!();
            println!("  /agents                   list available agents");
            println!("  /agents list              list available agents");
            println!("  /agents current           show current agent");
            println!("  /agents use <name>        switch to an agent");
            println!("  /agents auto              restore automatic agent routing");
            println!("  /agents reload            reload agents from disk (hot-discovery)");
            println!();
        }
        "list" | "ls" | "" => {
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
                println!(
                    "Routing mode: {}",
                    if app.cli.agent.is_some() {
                        "manual"
                    } else {
                        "auto"
                    }
                );
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
        "auto" => {
            let was_manual = app.cli.agent.take().is_some();
            println!("Agent auto-routing is now enabled.");
            if was_manual {
                println!(
                    "Current agent remains '{}' for now and may auto-switch on the next user request.",
                    app.current_agent
                );
            } else {
                println!("Agent routing was already in auto mode.");
            }
        }
        "reload" => {
            let old_count = agent_manifests.len();
            *agent_manifests = agents::load_all_agents();
            let new_count = agent_manifests.len();
            let delta = new_count as i64 - old_count as i64;
            if delta > 0 {
                println!("[Agent 发现] 重新扫描完成，新发现 {} 个 agent(s)，当前共 {} 个", delta, new_count);
            } else if delta < 0 {
                println!("[Agent 发现] 重新扫描完成，{} 个 agent 已移除，当前共 {} 个", -delta, new_count);
            } else {
                println!("[Agent 发现] 重新扫描完成，共 {} 个 agent (无变化)", new_count);
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
                app.current_agent_manifest = Some(agent.clone());
                app.cli.agent = Some(agent.name.clone());

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
            println!("unknown action: '{}'. try: /agents help", action);
        }
    }
    Ok(true)
}
