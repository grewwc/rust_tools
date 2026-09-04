//! `/changes` 与 `/diff`：查看本会话文件变更并支持外部工具打开 diff。
//!
//! 交互：
//! - `/changes` / `/diff`            → 摘要视图（mutation log 优先，git 回退）
//! - `/changes --stat` / `/diff --stat` → `git diff --stat`（仅 git 仓库）
//! - `/changes --json`               → JSON 列表（便于脚本）
//! - `/changes --patch [path]`       → 生成 `session_assets/changes.patch` 并打印路径，可指定输出路径
//! - `/changes --open [editor]`      → 生成 patch 后用外部编辑器打开；editor 可为 code/vscode/cursor/idea/git/open/auto
//!   git 支持：若不在 git 仓库，`--stat`/`--open=git` 会提示不可用；其余模式自动回退到 mutation 视图。
//! - `/changes --help` / `-h`        → 用法说明
//! 前缀同时支持 `/` 与 `:`。

use std::path::PathBuf;

use crate::ai::tools::storage::changes;

// ---------------------------------------------------------------------------
// Parse
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ChangesCommand {
    Help,
    Summary,
    Stat,
    Json,
    Patch { dest: Option<PathBuf> },
    Open { editor: Option<changes::EditorKind> },
}

pub(crate) fn parse_changes_command(input: &str) -> Option<ChangesCommand> {
    let trimmed = input.trim();
    let normalized = trimmed
        .strip_prefix('/')
        .or_else(|| trimmed.strip_prefix(':'))?;
    let (_head, rest) = if let Some(r) = normalized.strip_prefix("changes") {
        ("changes", r)
    } else if let Some(r) = normalized.strip_prefix("diff") {
        ("diff", r)
    } else {
        return None;
    };
    // head 后必须为空白或结束，避免 `/changesxxx`
    if rest.chars().next().is_some_and(|c| !c.is_whitespace()) {
        return None;
    }
    let args = rest.trim();
    if args.is_empty() {
        return Some(ChangesCommand::Summary);
    }
    let mut tokens = args.split_whitespace().peekable();
    let first = tokens.peek().copied().unwrap_or("");
    match first {
        "-h" | "--help" | "help" => Some(ChangesCommand::Help),
        "--stat" | "-s" | "stat" => Some(ChangesCommand::Stat),
        "--json" | "json" => Some(ChangesCommand::Json),
        "--patch" | "patch" => {
            tokens.next();
            let dest = tokens.next().map(PathBuf::from);
            Some(ChangesCommand::Patch { dest })
        }
        "--open" | "open" => {
            tokens.next();
            let editor_raw = tokens.next().unwrap_or("");
            let editor = if editor_raw.is_empty() {
                None
            } else {
                // 支持 --open=vscode 形式
                let v = if let Some(eq) = editor_raw.strip_prefix('=') {
                    eq
                } else if editor_raw.starts_with('-') {
                    // `--open --stat` 这类不应把 --stat 当编辑器名
                    ""
                } else {
                    editor_raw
                };
                // 处理 `--open=vscode` 粘连在 first token 的情况：`--open=code`
                if v.is_empty() {
                    // 尝试从 original args 解析 `--open=xxx`
                    if let Some(pos) = args.find("--open=") {
                        let tail = &args[pos + "--open=".len()..];
                        let tok = tail.split_whitespace().next().unwrap_or("");
                        if tok.is_empty() {
                            None
                        } else {
                            changes::EditorKind::from_str(tok)
                        }
                    } else {
                        None
                    }
                } else {
                    changes::EditorKind::from_str(v)
                }
            };
            // 若用户传了未知编辑器名，给出 Help 而非静默忽略，便于纠正
            if let Some(tok) = tokens.peek() {
                // 多余参数忽略
                let _ = tok;
            }
            // 校验：如果用户显式给了编辑器名但未识别，视为用法错误
            if !editor_raw.is_empty() && editor.is_none() {
                // 尝试提取等号后的原始值用于提示
                let raw = if let Some(eq) = args.find("--open=") {
                    &args[eq + "--open=".len()..]
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                } else if editor_raw.starts_with("--") {
                    editor_raw
                } else {
                    editor_raw
                };
                if !raw.is_empty()
                    && changes::EditorKind::from_str(raw).is_none()
                    && !raw.starts_with('-')
                {
                    // 仍按 Help 返回，上层会打印支持列表
                    return Some(ChangesCommand::Help);
                }
            }
            Some(ChangesCommand::Open { editor })
        }
        _ if first.starts_with("--open=") => {
            let val = first.trim_start_matches("--open=");
            let editor = if val.is_empty() {
                None
            } else {
                changes::EditorKind::from_str(val)
            };
            if val.is_empty() || editor.is_some() {
                Some(ChangesCommand::Open { editor })
            } else {
                Some(ChangesCommand::Help)
            }
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Handle (local command)
// ---------------------------------------------------------------------------

pub fn try_handle_changes_command(input: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let Some(cmd) = parse_changes_command(input) else {
        return Ok(false);
    };
    match cmd {
        ChangesCommand::Help => {
            print_changes_help();
        }
        ChangesCommand::Summary => {
            let summary = changes::combined_summary();
            println!("{summary}");
        }
        ChangesCommand::Stat => match changes::git_stat() {
            Some(stat) => println!("{stat}"),
            None => {
                eprintln!("提示：不在 git 仓库内或无差异，无法展示 --stat；以下为会话级摘要：\n");
                println!("{}", changes::combined_summary());
            }
        },
        ChangesCommand::Json => {
            let json = changes_json();
            println!("{json}");
        }
        ChangesCommand::Patch { dest } => match write_patch(dest) {
            Ok(path) => println!("patch 已生成：{}", path.display()),
            Err(e) => eprintln!("生成 patch 失败：{e}"),
        },
        ChangesCommand::Open { editor } => match changes::open_changes(editor) {
            Ok(msg) => println!("{msg}"),
            Err(e) => eprintln!("{e}"),
        },
    }
    Ok(true)
}

fn write_patch(dest: Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(p) = dest {
        let patch = changes::combined_patch()
            .ok_or_else(|| "当前会话无可导出的变更（无 mutation log 且 git 无差异）".to_string())?;
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| format!("创建目录失败: {e}"))?;
            }
        }
        std::fs::write(&p, patch.as_bytes()).map_err(|e| format!("写入 patch 失败: {e}"))?;
        Ok(p)
    } else {
        changes::write_combined_patch()
    }
}

fn changes_json() -> String {
    let entries = crate::ai::tools::storage::mutation_log::read_all();
    if !entries.is_empty() {
        let grouped = changes::session_grouped_changes();
        let mut items = Vec::new();
        for g in grouped {
            let obj = serde_json::json!({
                "path": g.path,
                "rel": g.rel,
                "net": g.net,
                "writes": g.write_count,
                "deletes": g.delete_count,
            });
            items.push(obj);
        }
        return serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string());
    }
    // 回退：git status
    let cwd = match crate::ai::driver::runtime_ctx::effective_cwd() {
        Ok(p) => p,
        Err(e) => {
            return serde_json::json!({"error": format!("无法确定工作目录: {e}")}).to_string();
        }
    };
    if !changes::is_inside_git_work_tree(&cwd) {
        return serde_json::json!({"changes": [], "note": "不在 git 仓库且无 mutation log"})
            .to_string();
    }
    let status = crate::fork_guard::output(
        std::process::Command::new("git")
            .args(["status", "--porcelain=v1"])
            .current_dir(&cwd),
    )
    .ok()
    .filter(|o| o.status.success())
    .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
    .unwrap_or_default();
    let files: Vec<serde_json::Value> = status
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::json!({"status": l[..2].to_string(), "path": l[3..].to_string()}))
        .collect();
    serde_json::to_string_pretty(&serde_json::json!({"git_status": files}))
        .unwrap_or_else(|_| "[]".into())
}

fn print_changes_help() {
    println!(
        r#"changes / diff — 查看本会话文件变更并支持外部打开

用法:
  /changes                摘要视图（mutation 优先，无则回退 git）
  /diff                   同上（alias）
  /changes --stat         git diff --stat（需在 git 仓库内）
  /changes --json         JSON 列表
  /changes --patch [path] 生成 patch；无 path 时写入 <session_assets>/changes.patch
                          （无活动会话时回退到系统临时目录）
  /changes --open [editor] 用外部工具打开 diff（自动生成 patch）
  /changes --help         显示本帮助

--open 的 editor 可选值:
  auto   自动探测（默认）：code → cursor → git difftool → open
  code / vscode           VS Code（单文件时用 code --diff 双栏对比）
  cursor                  Cursor
  idea                    JetBrains IDE（idea diff）
  git                     git difftool --dir-diff
  open                    系统默认打开（macOS open / Linux xdg-open）

配置:
  ai.diff.editor = auto | code | cursor | idea | git | open
  例：a config set ai.diff.editor code

示例:
  /changes
  /changes --open
  /changes --open=code
  /changes --open=git
  /changes --patch /tmp/my.patch
"#
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_summary_and_alias() {
        assert_eq!(
            parse_changes_command("/changes"),
            Some(ChangesCommand::Summary)
        );
        assert_eq!(
            parse_changes_command(":changes"),
            Some(ChangesCommand::Summary)
        );
        assert_eq!(
            parse_changes_command("/diff"),
            Some(ChangesCommand::Summary)
        );
        assert_eq!(
            parse_changes_command("/diff  "),
            Some(ChangesCommand::Summary)
        );
    }

    #[test]
    fn parse_help() {
        assert_eq!(
            parse_changes_command("/changes --help"),
            Some(ChangesCommand::Help)
        );
        assert_eq!(
            parse_changes_command("/diff -h"),
            Some(ChangesCommand::Help)
        );
    }

    #[test]
    fn parse_stat_json() {
        assert_eq!(
            parse_changes_command("/changes --stat"),
            Some(ChangesCommand::Stat)
        );
        assert_eq!(
            parse_changes_command("/changes --json"),
            Some(ChangesCommand::Json)
        );
    }

    #[test]
    fn parse_patch() {
        assert_eq!(
            parse_changes_command("/changes --patch"),
            Some(ChangesCommand::Patch { dest: None })
        );
        assert_eq!(
            parse_changes_command("/changes --patch /tmp/a.patch"),
            Some(ChangesCommand::Patch {
                dest: Some(PathBuf::from("/tmp/a.patch"))
            })
        );
    }

    #[test]
    fn parse_open() {
        assert_eq!(
            parse_changes_command("/changes --open"),
            Some(ChangesCommand::Open { editor: None })
        );
        assert_eq!(
            parse_changes_command("/changes --open=code"),
            Some(ChangesCommand::Open {
                editor: Some(changes::EditorKind::Vscode)
            })
        );
        assert_eq!(
            parse_changes_command("/changes --open code"),
            Some(ChangesCommand::Open {
                editor: Some(changes::EditorKind::Vscode)
            })
        );
        assert_eq!(
            parse_changes_command("/changes --open=git"),
            Some(ChangesCommand::Open {
                editor: Some(changes::EditorKind::Git)
            })
        );
    }

    #[test]
    fn rejects_unknown_prefix() {
        assert_eq!(parse_changes_command("/changesxxx"), None);
        assert_eq!(parse_changes_command("/diffx"), None);
    }
}
