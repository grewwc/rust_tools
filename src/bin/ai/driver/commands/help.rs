use crate::ai::agents::{self, AgentManifest};

/// 统一的交互式命令帮助信息
pub fn print_interactive_help() {
    println!("Interactive commands:");
    println!();
    println!("  General:");
    println!("    /help, /h                 show this help message");
    println!("    /model [name]             list or switch models");
    println!("    /usage [models|today|7d|30d|all|daily]   show LLM token usage statistics");
    println!("    /history [full|user|assistant|tool|system] [N]     show recent session messages");
    println!("    /history grep <keyword>      search recent messages by keyword");
    println!("    /history rewind u<N>|last    remove a user input and all following messages");
    println!("    /history export [file.txt]   export current preview to a file");
    println!("    /history copy                copy current preview to clipboard");
    println!("    /history replay              replay the last turn's assistant conclusion (text only)");
    println!("    /feishu-auth              authenticate with Feishu");
    println!("    /share [output.md]        export current session as shareable markdown");
    println!("    /close                    close and delete current session, then exit");
    println!();
    println!("  Persona management:");
    println!("    /personas                 list personas");
    println!("    /personas current         show current persona");
    println!("    /personas create          interactively create a persona");
    println!("    /personas use <name|id>   switch to a persona");
    println!("    /personas delete <name|id> delete a persona");
    println!("    /personas help            show persona command help");
    println!();
    println!("  Agent management:");
    println!("    /agents                   list available agents");
    println!("    /agents list              list available agents");
    println!("    /agents current           show current agent");
    println!("    /agents use <name>        switch to an agent");
    println!("    /agents auto              restore automatic agent routing");
    println!();
    println!("  Skill management:");
    println!("    /skills                   list available skills");
    println!("    /skills <name>            select & activate a skill");
    println!();
    println!("  Session management:");
    println!("    /sessions                 list all sessions");
    println!("    /sessions list            list all sessions");
    println!("    /sessions current         show current session info");
    println!("    /sessions new             create and switch to new session");
    println!("    /sessions use <id>        switch to specified session");
    println!("    /sessions suspend         suspend current session and return to shell (or /suspend, /bg, /detach, /susp)");
    println!("    /sessions bound           list suspended sessions bound to current terminal");
    println!("    /sessions delete <id>     delete specified session");
    println!("    /sessions clear-bound     clear suspended sessions bound to current terminal");
    println!("    /sessions clear-history   clear current session history (keeps session alive)");
    println!("    /sessions clear-all       delete all sessions");
    println!("    /sessions export <id> [output.md]       export session to Markdown");
    println!("    /sessions export-current [output.md]    export current session to Markdown");
    println!("    /sessions export-last [output.md]       export latest session to Markdown");
    println!("    /sessions fork [src=<id>] [as=<id>]      copy session to a new branch");
    println!("    /sessions branch <keep_messages> [src=<id>] [as=<id>]");
    println!();
    println!("  Notes:");
    println!("    - Commands support both / and : prefix (e.g., /help or :help)");
    println!("    - Use `/personas` for persona management");
    println!("    - To create a persona: start `a`, then run `/personas create`");
    println!("    - Run bare `a` in the same terminal to resume/select suspended sessions");
    println!("    - Run `a --new-session` to start fresh without consuming suspended sessions");
    println!("    - Press Ctrl+C to interrupt streaming or exit");
    println!();
}

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
    print_interactive_help();
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
