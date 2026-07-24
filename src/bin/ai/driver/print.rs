use crate::ai::{
    driver::McpInitReport,
    driver::model::OcrExtraction,
    mcp::McpClient,
    skills::SkillManifest,
    theme::{
        ACCENT_COMMAND, ACCENT_DANGER, ACCENT_MUTED, ACCENT_PRIMARY, ACCENT_RULE, ACCENT_SECONDARY,
        ACCENT_SUCCESS, ACCENT_TOOL_NAME, ACCENT_WARN, BOLD, DIM, RESET,
    },
    types::App,
};

pub fn print_assistant_banner() {
    print_assistant_banner_with_app(None);
}

pub fn print_assistant_banner_with_app(app: Option<&App>) {
    print_assistant_banner_with_app_and_skill(app, None);
}

/// 终端不再打印 assistant 角色标记行（如 "assistant · build"），信息冗余。
/// 角色信息仍通过 messages 传递给模型，不影响 agent 效果。
pub fn print_assistant_banner_with_app_and_skill(_app: Option<&App>, _skill_name: Option<&str>) {
    // no-op
}

/// 终端不再打印工具输出内容，只保留工具调用状态行。
/// 工具结果仍会写入 messages 供模型使用，不影响 agent 效果。
pub fn print_tool_output_block(_content: &str) {}

pub fn print_tool_output_line(line: &str) {
    if !crate::ai::driver::runtime_ctx::terminal_output_enabled() {
        return;
    }
    println!("{}", format_tool_output_line(line));
}

pub fn print_tool_note_line(label: &str, value: &str) {
    if !crate::ai::driver::runtime_ctx::terminal_output_enabled() {
        return;
    }
    println!("{}", format_tool_note_line(label, value));
}

pub fn print_tool_command_line(command: &str) {
    if !crate::ai::driver::runtime_ctx::terminal_output_enabled() {
        return;
    }
    println!("{}", format_tool_command_line(command));
}

pub fn print_ocr_summary(extraction: &OcrExtraction) {
    if !crate::ai::driver::runtime_ctx::terminal_output_enabled() {
        return;
    }
    for line in format_ocr_summary_block(extraction) {
        println!("{line}");
    }
}

pub(in crate::ai) fn format_section_header(title: &str, detail: Option<&str>) -> String {
    let mut out = String::new();
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
    format!("  {}{}{}", ACCENT_MUTED, note, RESET)
}

pub(in crate::ai) fn format_section_item(label: &str, description: &str) -> String {
    if description.is_empty() {
        return format!("  {}{}{}", ACCENT_PRIMARY, label, RESET);
    }
    format!(
        "  {}{}{} {}· {}{}{}",
        ACCENT_PRIMARY, label, RESET, ACCENT_MUTED, RESET, DIM, description
    ) + RESET
}

pub(in crate::ai) fn format_empty_state(label: &str) -> String {
    format!("  {}{}{}", ACCENT_MUTED, label, RESET)
}

pub(in crate::ai) fn format_assistant_banner(
    agent_name: Option<&str>,
    skill_name: Option<&str>,
) -> String {
    let detail = match (agent_name, skill_name) {
        (Some(agent), Some(skill)) => Some(format!("{agent} · skill:{skill}")),
        (Some(agent), None) => Some(agent.to_string()),
        (None, Some(skill)) => Some(format!("skill:{skill}")),
        (None, None) => None,
    };
    let mut out = format!("{BOLD}{ACCENT_SUCCESS}assistant{RESET}");
    if let Some(detail) = detail.as_deref() {
        out.push_str(&format!(" {ACCENT_MUTED}· {detail}{RESET}"));
    }
    out
}

pub(in crate::ai) fn format_tool_header(tool_name: &str) -> String {
    format!(
        "{}{}tool{} {}{}{}",
        BOLD, ACCENT_SUCCESS, RESET, ACCENT_TOOL_NAME, tool_name, RESET
    )
}

pub(in crate::ai) fn format_tool_status(status: &str, tool_name: &str, accent: &str) -> String {
    format!(
        "  {}{}{} {}{}{}",
        accent, status, RESET, ACCENT_TOOL_NAME, tool_name, RESET
    )
}

pub(in crate::ai) fn format_tool_status_with_file_target(
    status_line: String,
    target: &str,
) -> String {
    let target = sanitize_for_terminal(target);
    // 当 target 包含行号信息（read_file 的 "path  lines X..Y"）时，
    // 把路径和行号分开着色，避免同一行两种信息颜色重复。
    if let Some((path_part, line_part)) = target.split_once("  lines ") {
        format!(
            "{}  {}·{} {}{}{} {}{}{}",
            status_line,
            ACCENT_MUTED, RESET, ACCENT_SECONDARY, path_part, RESET,
            ACCENT_COMMAND, line_part, RESET,
        )
    } else {
        format!(
            "{}  {}·{} {}{}{}",
            status_line, ACCENT_MUTED, RESET, ACCENT_SECONDARY, target, RESET
        )
    }
}

pub(in crate::ai) fn format_tool_status_running(tool_name: &str) -> String {
    format_tool_status("●", tool_name, ACCENT_PRIMARY)
}

pub(in crate::ai) fn format_tool_status_cached(tool_name: &str) -> String {
    format_tool_status("◇", tool_name, ACCENT_SECONDARY)
}

pub(in crate::ai) fn format_tool_status_skipped(tool_name: &str) -> String {
    format_tool_status("–", tool_name, ACCENT_WARN)
}

pub(in crate::ai) fn format_tool_status_completed(tool_name: &str) -> String {
    format_tool_status("✓", tool_name, ACCENT_SUCCESS)
}

pub(in crate::ai) fn format_tool_status_failed(tool_name: &str) -> String {
    format_tool_status("×", tool_name, ACCENT_DANGER)
}

pub(in crate::ai) fn format_tool_status_deferred(tool_name: &str) -> String {
    format_tool_status("◷", tool_name, ACCENT_WARN)
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
                    if bytes[i] == 0x07
                        || (bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'\\')
                    {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
            } else {
                // Unknown ESC sequence. Preserve the old behavior of dropping the following
                // payload as well, but do it in char units instead of raw bytes so we never
                // slice inside a UTF-8 code point.
                i += 1;
                while i < bytes.len() && !text.is_char_boundary(i) {
                    i += 1;
                }
                if i < bytes.len() {
                    if let Some(ch) = text[i..].chars().next() {
                        i += ch.len_utf8();
                    } else {
                        break;
                    }
                }
            }
            continue;
        }

        // `i` is a byte index. In rare cases (e.g. binary blobs containing ESC + UTF8 leading
        // bytes), previous byte-based skipping can leave `i` in the middle of a UTF8 sequence.
        // Never slice `text[i..]` unless `i` is a valid char boundary.
        while i < bytes.len() && !text.is_char_boundary(i) {
            i += 1;
        }
        if i >= bytes.len() {
            break;
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
        ACCENT_RULE, RESET, BOLD, label, RESET, ACCENT_MUTED, value, RESET
    )
}

pub(in crate::ai) fn format_tool_command_line(command: &str) -> String {
    format!(
        "  {}│{} {}${} {}{}{}",
        ACCENT_RULE, RESET, ACCENT_MUTED, RESET, ACCENT_COMMAND, command, RESET
    )
}

pub(in crate::ai) fn format_tool_output_block(content: &str) -> Vec<String> {
    if content.trim().is_empty() {
        return vec![format_tool_note_line("result", "no output")];
    }

    content.lines().map(format_tool_output_line).collect()
}

/// 从文件类工具的 arguments JSON 中提取目标文件名，用于终端提示。
/// - `read_file` / `write_file`: `file_path` 字段
/// - `apply_patch`: 优先 `file_path` / `path`，否则从 patch 文本的 `*** Update File:` 等行提取
pub(in crate::ai) fn format_file_tool_target(tool_name: &str, args_json: &str) -> Option<String> {
    let args: serde_json::Value = serde_json::from_str(args_json.trim()).ok()?;
    match tool_name {
        "read_file" | "write_file" => args
            .get("file_path")
            .or_else(|| args.get("path"))
            .and_then(|v| v.as_str())
            .map(|path| {
                let short = short_path(path);
                if tool_name == "read_file" {
                    let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(1);
                    let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(0);
                    if limit > 0 {
                        return format!(
                            "{}  lines {}..{}",
                            short,
                            offset,
                            offset + limit.saturating_sub(1)
                        );
                    }
                }
                short
            }),
        "delete_path" => args.get("path").and_then(|v| v.as_str()).map(short_path),
        "apply_patch" => {
            // 优先从 file_path / path 参数取
            for key in &["file_path", "path"] {
                if let Some(p) = args
                    .get(*key)
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                {
                    return Some(short_path(p));
                }
            }
            // 其次从 patch 文本中提取第一个目标文件
            let patch = args.get("patch").and_then(|v| v.as_str())?;
            extract_first_patch_target(patch).map(|s| short_path(&s))
        }
        _ => None,
    }
}

/// 从 patch 文本中提取第一个 `*** Update File:` / `*** Add File:` 行的目标路径。
fn extract_first_patch_target(patch: &str) -> Option<String> {
    for line in patch.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed
            .strip_prefix("*** Update File:")
            .or_else(|| trimmed.strip_prefix("*** Add File:"))
        {
            return Some(rest.trim().to_string());
        }
    }
    None
}

/// 只保留路径最后两段（目录+文件名），截断过长的路径。
fn short_path(p: &str) -> String {
    // 优先取最后两段 /a/b/c/foo.rs → b/foo.rs
    let parts: Vec<&str> = p.split('/').filter(|s| !s.is_empty()).collect();
    let short = if parts.len() >= 2 {
        format!("{}/{}", parts[parts.len() - 2], parts[parts.len() - 1])
    } else {
        p.to_string()
    };
    let char_count = short.chars().count();
    if char_count > 50 {
        let byte_idx = short
            .char_indices()
            .nth(char_count - 48)
            .map(|(i, _)| i)
            .unwrap_or(0);
        format!("…{}", &short[byte_idx..])
    } else {
        short
    }
}

/// 若工具配置了 `print_args`，把调用入参回显到终端；否则为空操作。
/// 是否回显由工具自身提交的 `ToolDisplayConfig` 决定，调用方无需感知具体工具名。
pub(in crate::ai) fn echo_tool_args(tool_name: &str, args_json: &str) {
    if !crate::ai::driver::runtime_ctx::terminal_output_enabled() {
        return;
    }
    let config = crate::ai::tools::registry::common::tool_display_config(tool_name);
    if !config.print_args {
        return;
    }
    let trimmed = args_json.trim();
    // 尝试美化打印 JSON；解析失败时回退到原始文本。
    let pretty = serde_json::from_str::<serde_json::Value>(trimmed)
        .ok()
        .and_then(|v| serde_json::to_string_pretty(&v).ok())
        .unwrap_or_else(|| trimmed.to_string());
    for line in format_tool_output_block(&pretty) {
        println!("{line}");
    }
}

/// 若工具配置了 `print_result`，把输出内容回显到终端；否则为空操作。
/// 是否回显由工具自身提交的 `ToolDisplayConfig` 决定，调用方无需感知具体工具名。
pub(in crate::ai) fn echo_tool_output(tool_name: &str, content: &str) {
    if !crate::ai::driver::runtime_ctx::terminal_output_enabled() {
        return;
    }
    let config = crate::ai::tools::registry::common::tool_display_config(tool_name);
    if !config.print_result {
        return;
    }
    for line in format_tool_output_block(content) {
        println!("{line}");
    }
}

pub(in crate::ai) fn format_ocr_summary_block(extraction: &OcrExtraction) -> Vec<String> {
    let mut lines = Vec::with_capacity(extraction.images.len() + 1);
    let ok_count = extraction
        .images
        .iter()
        .filter(|img| img.error.is_none())
        .count();
    let failed_count = extraction.images.len().saturating_sub(ok_count);
    let detail = if failed_count == 0 {
        format!(
            "{} images · {}",
            extraction.images.len(),
            extraction.tool_name
        )
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

pub fn print_builtin_tool_summaries(tools: &[(String, String)]) {
    println!("{}", format_section_header("builtin tools", None));
    for (name, description) in tools {
        if name.starts_with("mcp_") {
            continue;
        }
        println!("{}", format_section_item(name, description));
    }
}

pub fn print_builtin_tools(app: &App) {
    let tools = app
        .agent_context
        .as_ref()
        .map(|c| c.tools.clone())
        .unwrap_or_default();
    let summaries = tools
        .into_iter()
        .map(|tool| (tool.function.name, tool.function.description))
        .collect::<Vec<_>>();
    print_builtin_tool_summaries(&summaries);
}

pub fn print_skills(skill_manifests: &[SkillManifest]) {
    let dir = crate::ai::skills::skills_dir();

    println!("{}", format_section_header("skills", None));
    println!(
        "{}",
        format_section_note(&format!(
            "local dir: {}; packaged SKILL.md skills from standard install roots are also auto-discovered",
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
            Some(&format!(
                "{} servers, {} tools",
                report.server_count, report.tool_count
            ))
        )
    );
    for failure in &report.failures {
        println!("{}", format_section_note(&format!("failed: {failure}")));
    }
    for t in mcp_client.get_all_tools() {
        println!(
            "{}",
            format_section_item(&t.function.name, &t.function.description)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{
        format_assistant_banner, format_empty_state, format_ocr_summary_block,
        format_section_header, format_section_item, format_section_note, format_tool_command_line,
        format_tool_header, format_tool_output_block, format_tool_output_line,
        format_tool_output_prefix, format_tool_status_completed,
        format_tool_status_with_file_target, sanitize_for_terminal,
    };
    use crate::ai::driver::model::{OcrExtraction, OcrImageSummary};
    use crate::ai::theme::{ACCENT_COMMAND, ACCENT_SECONDARY};

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
        let rendered = strip_ansi_for_test(&format_assistant_banner(Some("planner"), None));
        assert_eq!(rendered, "assistant · planner");
    }

    #[test]
    fn assistant_banner_includes_active_skill_when_present() {
        let rendered =
            strip_ansi_for_test(&format_assistant_banner(Some("build"), Some("humanizer")));
        assert_eq!(rendered, "assistant · build · skill:humanizer");
    }

    #[test]
    fn tool_header_and_block_use_muted_gutter() {
        let header = strip_ansi_for_test(&format_tool_header("search_codebase"));
        let lines = format_tool_output_block("line 1\n\nline 3")
            .into_iter()
            .map(|line| strip_ansi_for_test(&line))
            .collect::<Vec<_>>();

        assert_eq!(header, "tool search_codebase");
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
        let item = strip_ansi_for_test(&format_section_item(
            "search_codebase",
            "semantic code search",
        ));
        let empty = strip_ansi_for_test(&format_empty_state("no response"));

        assert_eq!(header, "mcp · 2 servers, 5 tools");
        assert_eq!(note, "  config path: ~/.config/mcp.json");
        assert_eq!(item, "  search_codebase · semantic code search");
        assert_eq!(empty, "  no response");
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

        assert_eq!(
            visible[0],
            "ocr · 2 images · 1 ok · 1 failed · mcp_ocr_extract_ocr_image"
        );
        assert_eq!(visible[1], "  a.png · ok · 128 chars");
        assert!(visible[2].starts_with("  b.png · failed · timeout talking to OCR service"));
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
    fn tool_command_line_keeps_text_stable_and_uses_distinct_accent() {
        let rendered = format_tool_command_line("cargo check --bin a");
        let visible = strip_ansi_for_test(&rendered);
        assert_eq!(visible, "  │ $ cargo check --bin a");
        assert!(rendered.contains(ACCENT_COMMAND));
    }

    #[test]
    fn tool_status_file_target_stays_on_status_line_with_distinct_accent() {
        let rendered = format_tool_status_with_file_target(
            format_tool_status_completed("read_file"),
            "tool_result/execution.rs",
        );
        let visible = strip_ansi_for_test(&rendered);

        assert_eq!(visible, "  ✓ read_file  · tool_result/execution.rs");
        assert!(rendered.contains(&format!("{ACCENT_SECONDARY}tool_result/execution.rs")));
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
    fn sanitize_does_not_panic_when_esc_is_followed_by_utf8_bytes() {
        let input = "prefix\x1b中文 suffix";
        let sanitized = sanitize_for_terminal(input);
        assert_eq!(sanitized, "prefix文 suffix");
    }

    #[test]
    fn tool_output_line_strips_control_characters() {
        let visible = strip_ansi_for_test(&format_tool_output_line("line\x03with\x04ctrl"));
        assert_eq!(visible, "  │ linewithctrl");
    }
}
