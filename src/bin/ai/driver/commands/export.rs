// =============================================================================
// /export interactive command -- exports the last model conclusion to a markdown file
// =============================================================================
//   /export              -- export to _summary.md in the current directory
//   /export xxx.md       -- export to the given markdown file
// =============================================================================

use std::path::PathBuf;

use super::status_line::show_status;
use crate::ai::driver::input;
use crate::ai::types::App;

pub fn try_handle_export_command(
    app: &mut App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    if !is_export_command(input) {
        return Ok(false);
    }
    let arg = extract_export_arg(input);
    execute_export(app, arg)?;
    Ok(true)
}

fn extract_export_arg(input: &str) -> &str {
    let trimmed = input.trim();
    let rest = trimmed
        .strip_prefix("/export")
        .or_else(|| trimmed.strip_prefix(":export"))
        .unwrap_or("");
    rest.trim()
}

fn execute_export(app: &App, arg: &str) -> Result<(), Box<dyn std::error::Error>> {
    // 1) Determine the output path
    let target_path = export_target_path(arg)?;

    // 2) Fetch the last assistant conclusion
    let text = match input::last_assistant_conclusion_text(app)? {
        Some(t) => t,
        None => {
            show_status("[export] 未找到模型结论，请先进行一次对话。");
            return Ok(());
        }
    };

    if text.trim().is_empty() {
        show_status("[export] 模型结论为空，已取消导出。");
        return Ok(());
    }

    // 3) Write the file
    let (line_count, char_count) = write_export_conclusion(&target_path, &text)?;
    let display_path = target_path.display();
    show_status(&format!(
        "[export] 已导出到 {} ({} 行, {} 字符)",
        display_path, line_count, char_count
    ));

    Ok(())
}

/// Resolve the destination for `/export [path]`: the argument as given
/// (relative paths join this process's working directory), or `_summary.md`
/// there when no path was supplied. Shared with the serve-chat client, whose
/// export writes to the machine running the client rather than the server.
pub(crate) fn export_target_path(arg: &str) -> std::io::Result<PathBuf> {
    if arg.is_empty() {
        return Ok(default_export_path());
    }
    let path = PathBuf::from(arg);
    if path.is_relative() {
        Ok(std::env::current_dir()?.join(path))
    } else {
        Ok(path)
    }
}

/// Write exported conclusion text, creating missing parent directories, and
/// return `(line_count, char_count)` for the caller's status line.
pub(crate) fn write_export_conclusion(
    target_path: &std::path::Path,
    text: &str,
) -> std::io::Result<(usize, usize)> {
    if let Some(parent) = target_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(target_path, text)?;
    Ok((text.lines().count(), text.chars().count()))
}

fn default_export_path() -> PathBuf {
    match std::env::current_dir() {
        Ok(cwd) => cwd.join("_summary.md"),
        Err(_) => PathBuf::from("_summary.md"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_export_path_is_summary_md_in_cwd() {
        let p = default_export_path();
        assert!(p.ends_with("_summary.md"));
        assert!(p.is_absolute());
    }

    #[test]
    fn is_export_command_matches_expected_prefixes() {
        assert!(is_export_command("/export"));
        assert!(is_export_command("/export "));
        assert!(is_export_command("/export out.md"));
        assert!(is_export_command("/export\tfoo.md"));
        assert!(is_export_command(":export"));
        assert!(is_export_command(":export foo.md"));

        // Cases that must not match
        assert!(!is_export_command("/export_foo"));
        assert!(!is_export_command("/foo"));
        assert!(!is_export_command(""));
        assert!(!is_export_command("export"));
    }

    #[test]
    fn export_target_path_keeps_absolute_and_joins_relative_to_cwd() {
        assert_eq!(
            export_target_path("/tmp/out.md").unwrap(),
            PathBuf::from("/tmp/out.md")
        );
        let relative = export_target_path("sub/out.md").unwrap();
        assert!(relative.is_absolute());
        assert!(relative.ends_with("sub/out.md"));
        assert!(export_target_path("").unwrap().ends_with("_summary.md"));
    }
}

/// Pure function: decides whether the input is a `/export` / `:export` command
/// (without performing the export).
/// Split out so it can be unit-tested and reused by the dispatcher.
fn is_export_command(input: &str) -> bool {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return false;
    }
    let rest = if let Some(r) = trimmed.strip_prefix("/export") {
        r
    } else if let Some(r) = trimmed.strip_prefix(":export") {
        r
    } else {
        return false;
    };
    rest.is_empty() || rest.starts_with(' ') || rest.starts_with('\t')
}
