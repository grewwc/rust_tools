use crate::ai::agents::{self, AgentManifest};

pub fn try_handle_help_command(input: &str) -> bool {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return false;
    }
    let normalized = if let Some(rest) = trimmed.strip_prefix('/') {
        rest
    } else if let Some(rest) = trimmed.strip_prefix(':') {
        rest
    } else {
        return false;
    };
    if normalized != "help" && normalized != "h" {
        return false;
    }
    println!("interactive commands:");
    println!("  /help");
    println!("  /feishu-auth");
    println!("  /agents                     list available agents");
    println!("  /agents use <name>          switch to an agent");
    println!("  /agents current             show current agent");
    println!("  /sessions");
    println!("  /sessions export <id> [output.md]");
    println!("  /sessions export-current [output.md]");
    println!("  /sessions export-last [output.md]");
    println!("  /sessions list");
    println!("  /sessions current");
    println!("  /sessions new");
    println!("  /sessions use <id>");
    println!("  /sessions delete <id>");
    println!("  /sessions clear-all");
    println!("  /share [output.md]              export current session as shareable markdown");
    println!();
    true
}

pub fn print_agents_list(agent_manifests: &[AgentManifest]) {
    let primary_agents = agents::get_primary_agents(agent_manifests);
    let subagents = agents::get_subagents(agent_manifests);

    println!("Available agents:\n");

    if !primary_agents.is_empty() {
        println!("Primary agents (use --agent <name> or /agents use <name>):");
        for agent in &primary_agents {
            let color_info = agent.color.as_deref().unwrap_or("default");
            let model_info = agent
                .model
                .as_deref()
                .map(|m| format!(" [{}]", m))
                .unwrap_or_default();
            println!(
                "  {}{} - {} ({})",
                agent.name, model_info, agent.description, color_info
            );
        }
    }

    if !subagents.is_empty() {
        println!("\nSubagents (use @<name> in conversation or task tool):");
        for agent in &subagents {
            let color_info = agent.color.as_deref().unwrap_or("default");
            println!("  {} - {} ({})", agent.name, agent.description, color_info);
        }
    }

    println!();
}
