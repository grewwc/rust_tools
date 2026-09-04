//! 变更查看服务：`show_changes` / `open_diff` 两个工具的实现。

use serde_json::{Value, json};

use crate::ai::tools::registry::common::{ToolRegistration, ToolSpec};
use crate::ai::tools::storage::changes::{self, EditorKind};

fn execute_show_changes(args: &Value) -> Result<String, String> {
    let v = handle_show_changes(args);
    serde_json::to_string_pretty(&v).map_err(|e| format!("failed to serialize: {e}"))
}

fn execute_open_diff(args: &Value) -> Result<String, String> {
    let v = handle_open_diff(args);
    serde_json::to_string_pretty(&v).map_err(|e| format!("failed to serialize: {e}"))
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "show_changes",
        description: "",
        execute: execute_show_changes,
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "open_diff",
        description: "",
        execute: execute_open_diff,
        // User-triggered only (opens a diff in an external editor); lazy via
        // `enable_tools` instead of a resident `core` schema every turn.
    }
});

// ── show_changes ───────────────────────────────────────────────────────
pub(crate) fn handle_show_changes(args: &Value) -> Value {
    let path_filter = args
        .get("path")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim();
    let include_git = args
        .get("include_git")
        .and_then(Value::as_bool)
        .unwrap_or_else(|| {
            crate::commonw::configw::get_all_config()
                .get_opt(crate::ai::config_schema::AiConfig::DIFF_INCLUDE_GIT)
                .filter(|v| !v.trim().is_empty())
                .map(|v| {
                    let low = v.trim().to_ascii_lowercase();
                    !(low == "false" || low == "0" || low == "no" || low == "off")
                })
                .unwrap_or(true)
        });
    let limit_snippet = args
        .get("limit_snippet")
        .and_then(Value::as_i64)
        .map(|v| v.max(0) as usize)
        .unwrap_or(120);

    // 读取真实会话链路
    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    let is_git_repo = changes::is_inside_git_work_tree(&cwd);
    let entries = crate::ai::tools::storage::mutation_log::read_all();
    let has_session_mutations = !entries.is_empty();
    let (patch_opt, text_raw) = if include_git {
        (changes::combined_patch(), changes::combined_summary())
    } else {
        // 仅会话，不回退 git
        let patch = if entries.is_empty() {
            None
        } else {
            changes::mutation_patch(&entries)
        };
        let text = if entries.is_empty() {
            "当前会话无文件变更（include_git=false，仅统计 mutation log）。".to_string()
        } else {
            // include_git=false 时不混入 git 状态提示，避免与 schema 承诺不一致
            changes::format_session_summary_with_git(false)
        };
        (patch, text)
    };
    let patch_bytes = patch_opt.as_ref().map(|s| s.len()).unwrap_or(0);

    let mut text = text_raw;
    if !path_filter.is_empty() {
        // 按路径子串过滤文本：仅保留命中行与上下文
        let mut filtered_lines = Vec::new();
        for line in text.lines() {
            if line.contains(path_filter) {
                filtered_lines.push(line.to_string());
            }
        }
        if filtered_lines.is_empty() {
            text = format!(
                "no changes matching path filter '{}'\n{}",
                path_filter, text
            );
        } else {
            // 仍附带原摘要头 + 过滤行
            let header = text.lines().take(4).collect::<Vec<_>>().join("\n");
            text = format!(
                "{}\n[filter: {}]\n{}",
                header,
                path_filter,
                filtered_lines.join("\n")
            );
        }
    }
    if limit_snippet > 0 && text.len() > limit_snippet * 80 {
        let cap = limit_snippet * 80;
        let truncated = text.chars().take(cap).collect::<String>();
        text = format!(
            "{}…\n[truncated to {} chars, patch {} bytes]",
            truncated, cap, patch_bytes
        );
    }

    let changed_files = changes::session_grouped_changes().len();
    json!({
        "text": text,
        "changed_files": changed_files,
        "is_git_repo": is_git_repo,
        "has_session_mutations": has_session_mutations,
        "patch_bytes": patch_bytes,
    })
}

// ── open_diff ──────────────────────────────────────────────────────────
pub(crate) fn handle_open_diff(args: &Value) -> Value {
    let editor_str = args.get("editor").and_then(Value::as_str).unwrap_or("auto");
    let patch_file_arg = args.get("patch_file").and_then(Value::as_str);
    let as_patch_opt = args.get("as_patch").and_then(Value::as_bool).or_else(|| {
        crate::commonw::configw::get_all_config()
            .get_opt(crate::ai::config_schema::AiConfig::DIFF_OPEN_PATCH_FILE)
            .filter(|v| !v.trim().is_empty())
            .map(|v| {
                let low = v.trim().to_ascii_lowercase();
                !(low == "false" || low == "0" || low == "no" || low == "off")
            })
    });

    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    let is_git_repo = changes::is_inside_git_work_tree(&cwd);
    if changes::session_grouped_changes().is_empty() && !is_git_repo {
        return json!({
            "ok": false,
            "error": "no changes to open (no session mutations and not a git repo)",
            "is_git_repo": is_git_repo,
        });
    }
    let patch_opt = changes::combined_patch();
    let patch = match patch_opt {
        Some(p) if !p.trim().is_empty() => p,
        _ => {
            return json!({
                "ok": false,
                "error": "generated patch is empty",
                "is_git_repo": is_git_repo,
            });
        }
    };

    // 当 as_patch == false 时，尝试按文件的 before/after 走 code --diff（仅单文件 + vscode/cursor）
    if as_patch_opt == Some(false) {
        let grouped = changes::session_grouped_changes();
        if grouped.len() == 1 {
            let normalized = if editor_str.eq_ignore_ascii_case("vscode") {
                "code"
            } else {
                editor_str
            };
            let kind = if editor_str.eq_ignore_ascii_case("auto") {
                // auto 场景下探测；若探测到 vscode/cursor 则允许 diff，否则回退 patch
                changes::configured_editor()
            } else {
                EditorKind::from_str(normalized).unwrap_or_else(changes::configured_editor)
            };
            if matches!(kind, EditorKind::Vscode | EditorKind::Cursor) {
                let g = &grouped[0];
                // 构造 before/after 临时文件
                if let Some(res) = try_open_single_file_diff(g, &kind) {
                    return match res {
                        Ok(cmd) => json!({
                            "ok": true,
                            "patch_file": null,
                            "mode": "diff",
                            "file": g.rel,
                            "editor": kind.label(),
                            "requested_editor": editor_str,
                            "command": cmd,
                            "is_git_repo": is_git_repo,
                        }),
                        Err(e) => json!({
                            "ok": false,
                            "error": e,
                            "mode": "diff",
                            "is_git_repo": is_git_repo,
                        }),
                    };
                }
            }
        }
        // 多文件或非 vscode/cursor / 无 before/after 时回退到 patch file 路径
    }

    // 明确指定 patch_file 时，覆盖写入该路径而非走会话资产目录
    if let Some(custom) = patch_file_arg {
        let dest = std::path::PathBuf::from(custom);
        // 安全校验：patch_file 必须落在可写根（effective_cwd / allowed_roots / session temp / skills）
        // 或为已注册的隔离临时文件（子代理隔离目录），与 FileStore::validate_write_access 保持一致
        let in_allowed = crate::ai::tools::storage::file_store::path_within_allowed_roots(&dest);
        let is_registered =
            crate::ai::tools::storage::temp_registry::is_registered(&dest.display().to_string());
        // 相对路径已在 path_within_allowed_roots 内按 effective_cwd 归一化；绝对路径孤立注册文件也放行
        // 额外兼容：绝对路径若恰好位于 runtime_ctx::temp_dir() 下也视为已授权的 scratch
        let in_session_tmp = dest.is_absolute()
            && dest.starts_with(
                crate::ai::driver::runtime_ctx::temp_dir().unwrap_or_else(|_| std::env::temp_dir()),
            );
        if !(in_allowed || is_registered || in_session_tmp) {
            return json!({
                "ok": false,
                "error": format!("patch_file 被沙箱拦截：{} 不在可写根（effective_cwd / allowed_roots / session temp）内", dest.display()),
                "is_git_repo": is_git_repo,
            });
        }
        if let Some(parent) = dest.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return json!({
                        "ok": false,
                        "error": format!("failed to create parent dir {}: {e}", parent.display()),
                        "is_git_repo": is_git_repo,
                    });
                }
            }
        }
        if let Err(e) = std::fs::write(&dest, patch.as_bytes()) {
            return json!({
                "ok": false,
                "error": format!("failed to write patch to {}: {e}", dest.display()),
                "is_git_repo": is_git_repo,
            });
        }
        // 自定义路径：尝试以 resolved 编辑器打开
        let kind = if editor_str.eq_ignore_ascii_case("auto") {
            changes::configured_editor()
        } else {
            let normalized = if editor_str.eq_ignore_ascii_case("vscode") {
                "code"
            } else {
                editor_str
            };
            EditorKind::from_str(normalized).unwrap_or_else(changes::configured_editor)
        };
        // 复用存储层 open_patch 逻辑：此处直接调用 open_changes 会忽略自定义路径，因此手动打开
        let open_res = open_patch_path_with_editor(&dest, &kind);
        return match open_res {
            Ok(cmd) => json!({
                "ok": true,
                "patch_file": dest.display().to_string(),
                "editor": kind.label(),
                "requested_editor": editor_str,
                "command": cmd,
                "is_git_repo": is_git_repo,
            }),
            Err(e) => json!({
                "ok": false,
                "error": e,
                "patch_file": dest.display().to_string(),
                "editor": kind.label(),
                "is_git_repo": is_git_repo,
            }),
        };
    }

    // 默认路径：走存储层统一入口（写入 <session_assets>/changes.patch 并打开）
    let requested_kind = if editor_str.eq_ignore_ascii_case("auto") {
        None
    } else {
        let normalized = if editor_str.eq_ignore_ascii_case("vscode") {
            "code"
        } else {
            editor_str
        };
        EditorKind::from_str(normalized)
    };
    match changes::open_changes(requested_kind.clone()) {
        Ok(cmd) => {
            // open_changes 已写入 <session_assets>/changes.patch（或 temp 回退），避免重复落盘
            // 通过 resolved kind 保持返回一致；patch_file 优先取 open_changes 返回中的路径
            let resolved = requested_kind.unwrap_or_else(changes::configured_editor);
            // 从存储层推断实际落盘路径：优先 session_assets，失败则回退 temp
            let patch_path = crate::ai::tools::storage::file_store::current_session_assets_dir()
                .map(|d| d.join("changes.patch"))
                .unwrap_or_else(|| {
                    crate::ai::driver::runtime_ctx::temp_dir()
                        .unwrap_or_else(|_| std::path::PathBuf::from("."))
                        .join(format!("changes.{}.patch", std::process::id()))
                });
            json!({
                "ok": true,
                "patch_file": patch_path.display().to_string(),
                "editor": resolved.label(),
                "requested_editor": editor_str,
                "command": cmd,
                "is_git_repo": is_git_repo,
            })
        }
        Err(e) => json!({
            "ok": false,
            "error": e,
            "is_git_repo": is_git_repo,
        }),
    }
}

fn try_open_single_file_diff(
    g: &crate::ai::tools::storage::changes::FileChange,
    kind: &EditorKind,
) -> Option<Result<String, String>> {
    // 仅当存在 before/after 差异时才有意义；两者皆 None 表示纯 meta
    if g.before_first.is_none() && g.after_last.is_none() {
        return None;
    }
    // Truncated snapshots are not the full file, so code --diff would show a
    // wrong comparison; fall back to the patch path (the patch is built from the
    // authoritative diff recorded at write time).
    if !crate::ai::tools::storage::changes::snapshots_full(&g.before_first, &g.after_last) {
        return None;
    }
    // 落盘到会话临时目录（与 file_store 沙箱保持一致，避免直接写系统 /tmp 越界）
    let base_tmp =
        crate::ai::driver::runtime_ctx::temp_dir().unwrap_or_else(|_| std::env::temp_dir());
    let tmp_dir = base_tmp.join(format!("rust_tools_diff_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp_dir);
    let safe = g.rel.replace('/', "_").replace('\\', "_");
    let before_path = tmp_dir.join(format!("{}.before", safe));
    let after_path = tmp_dir.join(format!("{}.after", safe));
    if let Some(b) = g.before_first.as_deref() {
        if std::fs::write(&before_path, b).is_err() {
            return None;
        }
    } else {
        let _ = std::fs::write(&before_path, "");
    }
    if let Some(a) = g.after_last.as_deref() {
        if std::fs::write(&after_path, a).is_err() {
            return None;
        }
    } else {
        let _ = std::fs::write(&after_path, "");
    }
    let prog = match kind {
        EditorKind::Vscode => "code",
        EditorKind::Cursor => "cursor",
        _ => return None,
    };
    let cmd_display = format!(
        "{} --diff {} {}",
        prog,
        before_path.display(),
        after_path.display()
    );
    let res = std::process::Command::new(prog)
        .args([
            "--diff",
            &before_path.display().to_string(),
            &after_path.display().to_string(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map(|_| cmd_display.clone())
        .map_err(|e| format!("failed to launch editor '{}': {e}", prog));
    Some(res)
}

fn open_patch_path_with_editor(
    patch_path: &std::path::Path,
    kind: &EditorKind,
) -> Result<String, String> {
    // 复刻 storage::changes::open_patch 但允许自定义路径
    let patch_str = patch_path.display().to_string();
    let (prog, args): (&str, Vec<String>) = match kind {
        EditorKind::Vscode => ("code", vec![patch_str.clone()]),
        EditorKind::Cursor => ("cursor", vec![patch_str.clone()]),
        EditorKind::Idea => ("idea", vec!["diff".to_string(), patch_str.clone()]),
        EditorKind::Git => {
            // git difftool 需要仓库上下文；自定义 patch 无仓库则退化为 open
            return Err("git difftool requires a git work tree; use code/cursor/open".to_string());
        }
        EditorKind::SystemOpen => {
            #[cfg(target_os = "macos")]
            {
                ("open", vec![patch_str.clone()])
            }
            #[cfg(not(target_os = "macos"))]
            {
                ("xdg-open", vec![patch_str.clone()])
            }
        }
        EditorKind::Auto => {
            #[cfg(target_os = "macos")]
            {
                ("open", vec![patch_str.clone()])
            }
            #[cfg(not(target_os = "macos"))]
            {
                ("xdg-open", vec![patch_str.clone()])
            }
        }
    };
    let cmd_display = format!("{} {}", prog, args.join(" "));
    std::process::Command::new(prog)
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("failed to launch editor '{}': {e}", prog))?;
    Ok(cmd_display)
}
