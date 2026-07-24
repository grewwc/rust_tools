// =============================================================================
// /export 交互命令 —— 把最后一条模型结论导出为 markdown 文件
// =============================================================================
//   /export              —— 导出到当前目录 _summary.md
//   /export xxx.md       —— 导出到指定 markdown 文件
// =============================================================================

use std::path::PathBuf;

use crate::ai::driver::input;
use crate::ai::types::App;
use super::status_line::show_status;

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
    // 1) 确定输出路径
    let target_path = if arg.is_empty() {
        default_export_path()
    } else {
        let p = PathBuf::from(arg);
        if p.is_relative() {
            std::env::current_dir()?.join(p)
        } else {
            p
        }
    };

    // 2) 取最后一条 assistant 结论
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

    // 3) 写文件
    if let Some(parent) = target_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(&target_path, &text)?;

    let display_path = target_path.display();
    let line_count = text.lines().count();
    let char_count = text.chars().count();
    show_status(&format!("[export] 已导出到 {} ({} 行, {} 字符)", display_path, line_count, char_count));

    Ok(())
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

        // 不应匹配的
        assert!(!is_export_command("/export_foo"));
        assert!(!is_export_command("/foo"));
        assert!(!is_export_command(""));
        assert!(!is_export_command("export"));
    }
}

/// 纯函数：判断输入是否为 `/export` / `:export` 命令（不执行导出）。
/// 拆分出来便于单测和分发器复用。
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
