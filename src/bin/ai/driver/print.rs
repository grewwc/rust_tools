use colored::Colorize;

use crate::ai::{driver::McpInitReport, mcp::McpClient, skills::SkillManifest, types::App};

pub fn print_assistant_banner() {
    println!("\n{}", "[Assistant]".bright_blue().bold());
}

pub fn print_tool_output_block(content: &str) {
    if content.trim().is_empty() {
        println!("  {} {}", "│".bright_black(), "(empty)".dimmed());
        return;
    }
    for line in content.lines() {
        if line.is_empty() {
            println!("  {}", "│".bright_black());
        } else {
            println!("  {} {}", "│".bright_black(), line.dimmed());
        }
    }
}

pub fn print_builtin_tools(app: &App) {
    println!("{}", "[builtin tools]".yellow());
    let tools = app
        .agent_context
        .as_ref()
        .map(|c| c.tools.clone())
        .unwrap_or_default();
    for t in tools {
        if t.function.name.starts_with("mcp_") {
            continue;
        }
        println!(" - {}: {}", t.function.name.cyan(), t.function.description);
    }
}

pub fn print_skills(skill_manifests: &[SkillManifest]) {
    let dir = crate::ai::skills::skills_dir();

    println!("{}", "[skills]".yellow());
    println!(
        "register: put *.skill (or *.md) into {}",
        dir.display().to_string().cyan()
    );
    for s in skill_manifests {
        println!(" - {}: {}", s.name.cyan(), s.description);
    }
}

pub fn print_mcp_tools(report: &McpInitReport, mcp_client: &McpClient) {
    if !report.loaded {
        println!("{}", "[mcp] no config or failed to load".yellow());
        println!("config path: {}", report.config_path);
        return;
    }
    println!(
        "{}",
        format!(
            "[mcp] {} servers, {} tools",
            report.server_count, report.tool_count
        )
        .yellow()
    );
    for failure in &report.failures {
        println!(" - {}", format!("[mcp failed] {failure}").red());
    }
    for t in mcp_client.get_all_tools() {
        println!(" - {}: {}", t.function.name.cyan(), t.function.description);
    }
}
