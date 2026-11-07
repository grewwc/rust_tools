// =============================================================================
// /export-md interactive command -- exports the last model conclusion to a markdown file
// =============================================================================
//   /export-md           -- export to _summary.md in the current directory
//   /export-md xxx.md    -- export to the given markdown file
// =============================================================================

use std::path::PathBuf;

use super::status_line::show_status;
use crate::ai::driver::input;
use crate::ai::types::App;

pub fn try_handle_export_markdown_command(
    app: &mut App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    if !is_export_markdown_command(input) {
        return Ok(false);
    }
    let arg = extract_export_markdown_arg(input);
    execute_export_markdown(app, arg)?;
    Ok(true)
}

fn extract_export_markdown_arg(input: &str) -> &str {
    let trimmed = input.trim();
    let rest = trimmed
        .strip_prefix("/export-md")
        .or_else(|| trimmed.strip_prefix(":export-md"))
        .unwrap_or("");
    rest.trim()
}

fn execute_export_markdown(app: &App, arg: &str) -> Result<(), Box<dyn std::error::Error>> {
    // 1) Determine the output path
    let target_path = if arg.is_empty() {
        default_export_markdown_path()
    } else {
        let p = PathBuf::from(arg);
        if p.is_relative() {
            std::env::current_dir()?.join(p)
        } else {
            p
        }
    };

    // 2) Fetch the last assistant conclusion
    let text = match input::last_assistant_conclusion_text(app)? {
        Some(t) => t,
        None => {
            show_status("[export-md] 未找到模型结论，请先进行一次对话。");
            return Ok(());
        }
    };

    if text.trim().is_empty() {
        show_status("[export-md] 模型结论为空，已取消导出。");
        return Ok(());
    }

    // 3) Write the file
    if let Some(parent) = target_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(&target_path, &text)?;

    let display_path = target_path.display();
    let line_count = text.lines().count();
    let char_count = text.chars().count();
    show_status(&format!(
        "[export-md] 已导出到 {} ({} 行, {} 字符)",
        display_path, line_count, char_count
    ));

    Ok(())
}

fn default_export_markdown_path() -> PathBuf {
    match std::env::current_dir() {
        Ok(cwd) => cwd.join("_summary.md"),
        Err(_) => PathBuf::from("_summary.md"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_export_markdown_path_is_summary_md_in_cwd() {
        let p = default_export_markdown_path();
        assert!(p.ends_with("_summary.md"));
        assert!(p.is_absolute());
    }

    #[test]
    fn is_export_markdown_command_matches_expected_prefixes() {
        assert!(is_export_markdown_command("/export-md"));
        assert!(is_export_markdown_command("/export-md "));
        assert!(is_export_markdown_command("/export-md out.md"));
        assert!(is_export_markdown_command("/export-md\tfoo.md"));
        assert!(is_export_markdown_command(":export-md"));
        assert!(is_export_markdown_command(":export-md foo.md"));

        // Cases that must not match
        assert!(!is_export_markdown_command("/export"));
        assert!(!is_export_markdown_command(":export foo.md"));
        assert!(!is_export_markdown_command("/export-md_foo"));
        assert!(!is_export_markdown_command("/foo"));
        assert!(!is_export_markdown_command(""));
        assert!(!is_export_markdown_command("export-md"));
    }
}

/// Pure function: decides whether the input is a `/export-md` / `:export-md` command
/// (without performing the export).
/// Split out so it can be unit-tested and reused by the dispatcher.
fn is_export_markdown_command(input: &str) -> bool {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return false;
    }
    let rest = if let Some(r) = trimmed.strip_prefix("/export-md") {
        r
    } else if let Some(r) = trimmed.strip_prefix(":export-md") {
        r
    } else {
        return false;
    };
    rest.is_empty() || rest.starts_with(' ') || rest.starts_with('\t')
}
