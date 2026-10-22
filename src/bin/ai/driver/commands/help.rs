use crate::ai::agents::{self, AgentManifest};

use super::discuss::print_discuss_help;
use super::task::print_task_help;

/// Command sections of the interactive help (everything except shortcuts/notes).
fn print_commands_help() {
    println!("Interactive commands:");
    println!();
    println!("  General:");
    println!("    /help [cmd|shortcuts|discuss|task]    show this help message (optional topic)");
    println!("    /model [name]             list or switch models");
    println!(
        "    /effort [level]           show or set reasoning effort (shortcut for /model effort)"
    );
    println!("    /theme [name]             list or switch terminal themes live (session-only)");
    println!(
        "    /cd [dir]                 show or change the session working directory (\"-\" goes back)"
    );
    println!("    /pwd                      show the session working directory (same as bare /cd)");
    println!("    /audit <instruction>      run the audit subagent synchronously (up to 15 min), then verify and fix findings");
    println!(
        "    /changes [/diff] [--stat|--json|--patch|--open]  show session file changes & open diff externally"
    );
    println!("    /clear                    clear the terminal screen");
    println!("    /compact                  compact session context; preserve full history");
    println!("    /distill-session [id|zip] [--dry-run] [--limit N]  distill into the knowledge base (default: current session)");
    println!("    /usage [models|today|7d|30d|all|daily]   show LLM token usage statistics");
    println!("    /history [full|user|assistant|tool|system] [N]     show session messages (role filters: up to 2000)");
    println!("    /history grep <keyword>      search recent messages by keyword");
    println!("    /history rewind u<N>|last    remove a user input and all following messages");
    println!("    /history export [file.txt]   export current preview to a file");
    println!("    /history copy                copy current preview to clipboard");
    println!(
        "    /history last [N]            replay the Nth most recent assistant message with markdown rendering"
    );
    println!(
        "    /history replay              replay the last turn's assistant conclusion (text only)"
    );
    println!(
        "    /history <session> [N]         show last N messages of another session (no switch; Tab completes ids)"
    );
    println!(
        "    /history <session> last [N]    replay the Nth recent assistant message of another session with markdown rendering"
    );
    println!(
        "    /history <session> replay     replay the last assistant conclusion of another session"
    );
    println!("    /share [output.md]        export current session as shareable markdown");
    println!("    /close                    close and delete current session, then exit");
    println!("    /proc                     show running sessions (interactive + background)");
    println!(
        "    /memo [text]              save last assistant conclusion (or given text) as memo to knowledge base"
    );
    println!(
        "    /export [file.md]         save last assistant conclusion to a markdown file (default: _summary.md)"
    );
    println!(
        "    /goal [target]            enter goal mode: agent persists until the goal is done"
    );
    println!(
        "    /discuss [N] <topic>      host a multi-session debate (N members, default 3)"
    );
    println!("    /discuss status [room]    show discussion progress (read-only)");
    println!(
        "    /task [N] <goal>          split a goal across worker sessions (N workers, default 3)"
    );
    println!("    /task status [room]       show task progress (read-only)");
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
    println!("    /agent                    list available agents");
    println!("    /agent <name>             switch to an agent");
    println!("    /agent current            show current agent");
    println!();
    println!("  Skill management:");
    println!("    /skills                   list available skills");
    println!("    /skills <name>...          select & activate skills (multiple allowed)");
    println!();
    println!("  Session management:");
    println!("    /sessions [list]          list all sessions (default, no sizes)");
    println!("    /sessions verbose         list all sessions with per-session sizes");
    println!("    /sessions current         show current session info");
    println!("    /sessions new             create and switch to new session");
    println!("    /sessions use <id>        switch to specified session");
    println!(
        "    /fork                     fork current session into a new branch (keeps original) and switch"
    );
    println!("    /sessions suspend         suspend current session and return to shell");
    println!("    /sessions bound           list suspended sessions bound to current terminal");
    println!("    /sessions delete <id> [more...]     delete one or more sessions");
    println!(
        "    /sessions unbind          remove this terminal's suspended-session bindings (sessions are kept)"
    );
    println!("    /sessions clear-history   clear current session history (keeps session alive)");
    println!("    /sessions clear-all       delete all sessions");
    println!("    /sessions dump-history <id>              dump session history to JSON");
    println!("    /sessions export <id|current|last> [output.md]   export session to Markdown");
    println!(
        "    /sessions archive <id|current|last> [output.zip] full session archive for migration"
    );
    println!("    /sessions import <file.zip> [as=<id>]           import session from archive");
    println!("    /sessions fork [src=<id>] [as=<id>]      copy session to a new branch");
    println!("    /sessions branch <keep_turns> [src=<id>] [as=<id>] retain complete user turns");
    println!(
        "    /title [text]             set or show the current session title (bare /title shows it)"
    );
    println!();
}

fn print_shortcuts_help() {
    println!("  Keyboard shortcuts:");
    println!("    Enter                 newline | Alt+Enter/F2      send");
    println!("    Ctrl+C                cancel  | Up/Down,Ctrl+P/N  history");
    println!("    Cmd/Ctrl+U            delete line | Ctrl+V  paste");
    println!("    F8  clear screen | F9  copy last answer | F10  copy all");
    println!("    Tab                   refresh completion | Esc  close completion");
}

fn print_help_notes() {
    println!("  Notes:");
    println!("    - Commands support both / and : prefix (e.g., /help or :help)");
    println!("    - Use `/personas` for persona management");
    println!("    - To create a persona: start `a`, then run `/personas create`");
    println!("    - Run bare `a` in the same terminal to resume/select suspended sessions");
    println!("    - Run `a --new-session` to start fresh without consuming suspended sessions");
    println!("    - Press Ctrl+C to interrupt streaming or exit");
}

/// Unified interactive command help message (same output as the three sections above).
pub fn print_interactive_help() {
    print_commands_help();
    println!();
    print_shortcuts_help();
    println!();
    print_help_notes();
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
    let mut parts = normalized.split_whitespace();
    match parts.next() {
        Some("help") | Some("h") => {}
        _ => return false,
    }
    // From here the head word matched, so the input is always consumed
    // locally: topics print their section, anything else is a usage error.
    // Nothing falls through to the model.
    match parts.next() {
        None => print_interactive_help(),
        Some(topic) if matches!(topic, "shortcuts" | "cmd" | "commands") => {
            if parts.next().is_some() {
                println!("Too many arguments. Usage: /help [cmd|shortcuts|discuss|task]");
            } else if topic == "shortcuts" {
                print_shortcuts_help();
                println!();
            } else {
                print_commands_help();
            }
        }
        Some(topic)
            if matches!(
                topic.trim_start_matches(['/', ':']),
                "discuss" | "task"
            ) =>
        {
            if parts.next().is_some() {
                println!("Too many arguments. Usage: /help [cmd|shortcuts|discuss|task]");
            } else if topic.trim_start_matches(['/', ':']) == "discuss" {
                print_discuss_help();
            } else {
                print_task_help();
            }
        }
        Some(other) => {
            println!(
                "Unknown help topic: '{other}'. Available topics: cmd, shortcuts, discuss, task"
            );
            println!("Usage: /help [cmd|shortcuts|discuss|task]");
        }
    }
    true
}

pub fn print_agents_list(agent_manifests: &[AgentManifest]) {
    let primary_agents = agents::get_primary_agents(agent_manifests);
    let subagents = agents::get_subagents(agent_manifests);

    println!("Available agents:\n");

    if !primary_agents.is_empty() {
        println!("Primary agents (use --agent <name> or /agent <name>):");
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

#[cfg(test)]
mod tests {
    use super::try_handle_help_command;

    #[test]
    fn help_consumes_bare_topics_and_rejects_unknown_words_locally() {
        // Handled locally (never forwarded to the model).
        assert!(try_handle_help_command("/help"));
        assert!(try_handle_help_command(":help"));
        assert!(try_handle_help_command("/h"));
        assert!(try_handle_help_command("/help shortcuts"));
        assert!(try_handle_help_command("/help cmd"));
        assert!(try_handle_help_command("/help commands"));
        // Usage errors are also consumed locally, not sent to the model.
        assert!(try_handle_help_command("/help bogus"));
        assert!(try_handle_help_command("/help shortcuts extra"));
        // Non-help input still falls through.
        assert!(!try_handle_help_command("/helpx"));
        assert!(!try_handle_help_command("help"));
        assert!(!try_handle_help_command(""));
    }
}
