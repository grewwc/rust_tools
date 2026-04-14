use crate::ai::{
    driver::model::OcrExtraction,
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

pub fn print_tool_output_line(line: &str) {
    println!("{}", format_tool_output_line(line));
}

pub fn print_tool_note_line(label: &str, value: &str) {
    println!("{}", format_tool_note_line(label, value));
}

pub fn print_ocr_summary(extraction: &OcrExtraction) {
    for line in format_ocr_summary_block(extraction) {
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

pub(in crate::ai) fn format_tool_output_prefix() -> String {
    format!("  {}│{} {}", ACCENT_RULE, RESET, DIM)
}

pub(in crate::ai) fn format_tool_output_line(line: &str) -> String {
    let sanitized = sanitize_for_terminal(line);
    if sanitized.is_empty() {
        format!("  {}│{}", ACCENT_RULE, RESET)
    } else {
        format!("  {}│{} {}{}{}", ACCENT_RULE, RESET, DIM, sanitized, RESET)
    }
}

pub(in crate::ai) fn sanitize_for_terminal(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut result = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == 0x1b {
            if i + 1 < bytes.len() && bytes[i + 1] == b'[' {
                i += 2;
                while i < bytes.len() {
                    let b = bytes[i];
                    i += 1;
                    if b >= 0x40 && b <= 0x7e {
                        break;
                    }
                }
            } else if i + 1 < bytes.len() && bytes[i + 1] == b']' {
                i += 2;
                while i < bytes.len() {
                    if bytes[i] == 0x07 || (bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'\\') {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
            } else {
                i += 1;
                if i < bytes.len() {
                    i += 1;
                }
            }
            continue;
        }
        let Some(ch) = text[i..].chars().next() else {
            break;
        };
        if ch.is_control() && ch != '\n' && ch != '\t' {
            i += ch.len_utf8();
            continue;
        }
        result.push(ch);
        i += ch.len_utf8();
    }
    result
}

pub(in crate::ai) fn format_tool_note_line(label: &str, value: &str) -> String {
    format!(
        "  {}│{} {}{}:{} {}{}{}",
        ACCENT_RULE, RESET, BOLD, label, RESET, ACCENT_PRIMARY, value, RESET
    )
}

pub(in crate::ai) fn format_tool_output_block(content: &str) -> Vec<String> {
    if content.trim().is_empty() {
        return vec![format_tool_note_line("result", "no output")];
    }

    content.lines().map(format_tool_output_line).collect()
}

pub(in crate::ai) fn format_ocr_summary_block(extraction: &OcrExtraction) -> Vec<String> {
    let mut lines = Vec::with_capacity(extraction.images.len() + 1);
    let ok_count = extraction.images.iter().filter(|img| img.error.is_none()).count();
    let failed_count = extraction.images.len().saturating_sub(ok_count);
    let detail = if failed_count == 0 {
        format!("{} images · {}", extraction.images.len(), extraction.tool_name)
    } else {
        format!(
            "{} images · {} ok · {} failed · {}",
            extraction.images.len(),
            ok_count,
            failed_count,
            extraction.tool_name
        )
    };
    lines.push(format_section_header("ocr", Some(&detail)));
    for image in &extraction.images {
        let description = if let Some(err) = &image.error {
            format!("failed · {}", truncate_terminal_detail(err, 120))
        } else if image.extracted_chars == 0 {
            "ok · empty text".to_string()
        } else {
            format!("ok · {} chars", image.extracted_chars)
        };
        lines.push(format_section_item(&image.file_name, &description));
    }
    lines
}

fn truncate_terminal_detail(s: &str, max_chars: usize) -> String {
    let total = s.chars().count();
    if total <= max_chars {
        return s.to_string();
    }
    let kept: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{kept}…")
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
        format_assistant_banner, format_empty_state, format_ocr_summary_block,
        format_section_header, format_section_item, format_section_note, format_tool_header,
        format_tool_output_block, format_tool_output_line, format_tool_output_prefix,
        sanitize_for_terminal,
    };
    use crate::ai::driver::model::{OcrExtraction, OcrImageSummary};

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
    fn empty_tool_output_uses_explicit_no_output_note() {
        let lines = format_tool_output_block("")
            .into_iter()
            .map(|line| strip_ansi_for_test(&line))
            .collect::<Vec<_>>();

        assert_eq!(lines, vec!["  │ result: no output"]);
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

    #[test]
    fn ocr_summary_block_is_compact_and_informative() {
        let extraction = OcrExtraction {
            tool_name: "mcp_ocr_extract_ocr_image".to_string(),
            content: String::new(),
            images: vec![
                OcrImageSummary {
                    file_name: "a.png".to_string(),
                    extracted_chars: 128,
                    error: None,
                },
                OcrImageSummary {
                    file_name: "b.png".to_string(),
                    extracted_chars: 42,
                    error: Some("timeout talking to OCR service".to_string()),
                },
            ],
        };

        let visible = format_ocr_summary_block(&extraction)
            .into_iter()
            .map(|line| strip_ansi_for_test(&line))
            .collect::<Vec<_>>();

        assert_eq!(visible[0], "╭─ ocr · 2 images · 1 ok · 1 failed · mcp_ocr_extract_ocr_image");
        assert_eq!(visible[1], "  │ a.png · ok · 128 chars");
        assert!(visible[2].starts_with("  │ b.png · failed · timeout talking to OCR service"));
    }

    #[test]
    fn tool_output_line_formats_single_line() {
        let visible = strip_ansi_for_test(&format_tool_output_line("hello"));
        assert_eq!(visible, "  │ hello");
    }

    #[test]
    fn tool_output_prefix_formats_gutter_without_closing_line() {
        let visible = strip_ansi_for_test(&format_tool_output_prefix());
        assert_eq!(visible, "  │ ");
    }

    #[test]
    fn sanitize_strips_control_characters_but_keeps_newline_and_tab() {
        let input = "hello\x03world\x04\nline\t2\x07";
        let sanitized = sanitize_for_terminal(input);
        assert_eq!(sanitized, "helloworld\nline\t2");
    }

    #[test]
    fn sanitize_strips_ansi_csi_sequences() {
        let input = "\x1b[31mred text\x1b[0m normal";
        let sanitized = sanitize_for_terminal(input);
        assert_eq!(sanitized, "red text normal");
    }

    #[test]
    fn sanitize_strips_ansi_osc_sequences() {
        let input = "\x1b]0;window title\x07content";
        let sanitized = sanitize_for_terminal(input);
        assert_eq!(sanitized, "content");
    }

    #[test]
    fn sanitize_strips_bare_esc_sequences() {
        let input = "before\x1bXafter";
        let sanitized = sanitize_for_terminal(input);
        assert_eq!(sanitized, "beforeafter");
    }

    #[test]
    fn sanitize_preserves_normal_text() {
        let input = "正常文本 hello world 123";
        let sanitized = sanitize_for_terminal(input);
        assert_eq!(sanitized, input);
    }

    #[test]
    fn tool_output_line_strips_control_characters() {
        let visible = strip_ansi_for_test(&format_tool_output_line("line\x03with\x04ctrl"));
        assert_eq!(visible, "  │ linewithctrl");
    }
}
