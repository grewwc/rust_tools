use crate::ai::{
    driver::McpInitReport,
    mcp::McpClient,
    skills::SkillManifest,
    theme::{ACCENT_MUTED, ACCENT_PRIMARY, ACCENT_RULE, ACCENT_SUCCESS, BOLD, DIM, RESET},
    types::App,
};

pub fn print_assistant_banner() {
    print_assistant_banner_with_app(None);
}

pub fn print_assistant_banner_with_app(app: Option<&App>) {
    println!("\n{}", format_assistant_banner(app.map(|a| a.current_agent.as_str())));
}

pub fn print_tool_output_block(content: &str) {
    for line in format_tool_output_block(content) {
        println!("{line}");
    }
}

pub(in crate::ai) fn format_section_header(title: &str, detail: Option<&str>) -> String {
    let mut out = String::new();
    out.push_str(ACCENT_RULE);
    out.push_str("╭─");
    out.push_str(RESET);
    out.push(' ');
    out.push_str(BOLD);
    out.push_str(ACCENT_SUCCESS);
    out.push_str(title);
    out.push_str(RESET);
    if let Some(detail) = detail.filter(|detail| !detail.is_empty()) {
        out.push(' ');
        out.push_str(ACCENT_MUTED);
        out.push('·');
        out.push(' ');
        out.push_str(detail);
        out.push_str(RESET);
    }
    out
}

pub(in crate::ai) fn format_section_note(note: &str) -> String {
    format!("  {}│{} {}{}{}", ACCENT_RULE, RESET, ACCENT_MUTED, note, RESET)
}

pub(in crate::ai) fn format_section_item(label: &str, description: &str) -> String {
    if description.is_empty() {
        return format!("  {}│{} {}{}{}", ACCENT_RULE, RESET, ACCENT_PRIMARY, label, RESET);
    }
    format!(
        "  {}│{} {}{}{} {}· {}{}{}",
        ACCENT_RULE,
        RESET,
        ACCENT_PRIMARY,
        label,
        RESET,
        ACCENT_MUTED,
        RESET,
        DIM,
        description
    ) + RESET
}

pub(in crate::ai) fn format_empty_state(label: &str) -> String {
    format!("{}╰─{} {}{}{}", ACCENT_RULE, RESET, ACCENT_MUTED, label, RESET)
}

pub(in crate::ai) fn format_assistant_banner(agent_name: Option<&str>) -> String {
    format_section_header("assistant", agent_name)
}

pub(in crate::ai) fn format_tool_header(tool_name: &str) -> String {
    format!(
        "{}├─{} {}{}tool{} {}{}{}",
        ACCENT_RULE, RESET, BOLD, ACCENT_SUCCESS, RESET, ACCENT_PRIMARY, tool_name, RESET
    )
}

pub(in crate::ai) fn format_tool_output_block(content: &str) -> Vec<String> {
    if content.trim().is_empty() {
        return vec![format!("  {}│{} {}(empty){}", ACCENT_RULE, RESET, ACCENT_MUTED, RESET)];
    }

    content
        .lines()
        .map(|line| {
            if line.is_empty() {
                format!("  {}│{}", ACCENT_RULE, RESET)
            } else {
                format!("  {}│{} {}{}{}", ACCENT_RULE, RESET, DIM, line, RESET)
            }
        })
        .collect()
}

pub fn print_builtin_tools(app: &App) {
    println!("{}", format_section_header("builtin tools", None));
    let tools = app
        .agent_context
        .as_ref()
        .map(|c| c.tools.clone())
        .unwrap_or_default();
    for t in tools {
        if t.function.name.starts_with("mcp_") {
            continue;
        }
        println!("{}", format_section_item(&t.function.name, &t.function.description));
    }
}

pub fn print_skills(skill_manifests: &[SkillManifest]) {
    let dir = crate::ai::skills::skills_dir();

    println!("{}", format_section_header("skills", None));
    println!(
        "{}",
        format_section_note(&format!(
            "register: put *.skill (or *.md) into {}",
            dir.display()
        ))
    );
    for s in skill_manifests {
        println!("{}", format_section_item(&s.name, &s.description));
    }
}

pub fn print_mcp_tools(report: &McpInitReport, mcp_client: &McpClient) {
    if !report.loaded {
        println!(
            "{}",
            format_section_header("mcp", Some("no config or failed to load"))
        );
        println!(
            "{}",
            format_section_note(&format!("config path: {}", report.config_path))
        );
        return;
    }
    println!(
        "{}",
        format_section_header(
            "mcp",
            Some(&format!("{} servers, {} tools", report.server_count, report.tool_count))
        )
    );
    for failure in &report.failures {
        println!("{}", format_section_note(&format!("failed: {failure}")));
    }
    for t in mcp_client.get_all_tools() {
        println!("{}", format_section_item(&t.function.name, &t.function.description));
    }
}

#[cfg(test)]
mod tests {
    use super::{
        format_assistant_banner, format_empty_state, format_section_header, format_section_item,
        format_section_note, format_tool_header, format_tool_output_block,
    };

    fn strip_ansi_for_test(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\x1b' && chars.peek() == Some(&'[') {
                let _ = chars.next();
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
                continue;
            }
            out.push(ch);
        }
        out
    }

    #[test]
    fn assistant_banner_uses_modern_single_line_label() {
        let rendered = strip_ansi_for_test(&format_assistant_banner(Some("planner")));
        assert_eq!(rendered, "╭─ assistant · planner");
    }

    #[test]
    fn tool_header_and_block_use_muted_gutter() {
        let header = strip_ansi_for_test(&format_tool_header("search_codebase"));
        let lines = format_tool_output_block("line 1\n\nline 3")
            .into_iter()
            .map(|line| strip_ansi_for_test(&line))
            .collect::<Vec<_>>();

        assert_eq!(header, "├─ tool search_codebase");
        assert_eq!(lines, vec!["  │ line 1", "  │", "  │ line 3"]);
    }

    #[test]
    fn section_helpers_keep_text_content_stable() {
        let header = strip_ansi_for_test(&format_section_header("mcp", Some("2 servers, 5 tools")));
        let note = strip_ansi_for_test(&format_section_note("config path: ~/.config/mcp.json"));
        let item = strip_ansi_for_test(&format_section_item("search_codebase", "semantic code search"));
        let empty = strip_ansi_for_test(&format_empty_state("no response"));

        assert_eq!(header, "╭─ mcp · 2 servers, 5 tools");
        assert_eq!(note, "  │ config path: ~/.config/mcp.json");
        assert_eq!(item, "  │ search_codebase · semantic code search");
        assert_eq!(empty, "╰─ no response");
    }
}
