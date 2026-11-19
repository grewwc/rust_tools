//! Change viewing service for `show_changes` and `open_diff`.

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

    // Read the current session context.
    let cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .unwrap_or_else(|_| std::path::PathBuf::from("."));
    let is_git_repo = changes::is_inside_git_work_tree(&cwd);
    let snapshot = crate::ai::tools::storage::mutation_log::read_committed_snapshot();
    let entries = &snapshot.mutations;
    let has_session_mutations = !entries.is_empty();
    let (patch_opt, text_raw) = if include_git {
        (changes::combined_patch(), changes::combined_summary())
    } else {
        // Use session mutations only, without a Git fallback.
        let patch = if entries.is_empty() {
            None
        } else {
            changes::mutation_patch(entries)
        };
        let text = if entries.is_empty() {
            "当前会话无文件变更（include_git=false，仅统计 mutation log）。".to_string()
        } else {
            // Keep the session-only view free of Git-status hints.
            changes::format_session_summary_with_git(false)
        };
        (patch, text)
    };
    let patch_bytes = patch_opt.as_ref().map(|s| s.len()).unwrap_or(0);

    let mut text = text_raw;
    if !path_filter.is_empty() {
        // Filter the summary by path substring.
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
            // Retain the summary header alongside matching lines.
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

    let changed_files = changes::grouped_changes(entries).len();
    let mut result = json!({
        "text": text,
        "changed_files": changed_files,
        "is_git_repo": is_git_repo,
        "has_session_mutations": has_session_mutations,
        "patch_bytes": patch_bytes,
    });
    if include_git {
        let inventory = changes::collect_current_changes(true);
        result["coverage"] = json!({
            "observed_paths": inventory.files.len(),
            "session_changed_files": inventory.session_changes.len(),
            "collection_complete": inventory.collection_complete(),
            "collection_errors": inventory.collection_errors,
            "coverage_gaps": inventory.coverage_gaps,
        });
        result["changed_paths"] = json!(inventory.files);
    }
    use crate::ai::tools::storage::mutation_log::{self, VerificationDetailsPage};
    let full = args
        .get("verification_full")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let details = full
        || args
            .get("verification_details")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let mut page = VerificationDetailsPage::default();
    if full {
        page.limit = 0;
        page.max_bytes = 0;
    } else {
        let read_usize = |name: &str, default: usize| {
            args.get(name)
                .and_then(Value::as_u64)
                .map(|value| usize::try_from(value).unwrap_or(usize::MAX))
                .unwrap_or(default)
        };
        page.offset = read_usize("verification_offset", page.offset);
        page.limit = read_usize("verification_limit", page.limit);
        page.max_bytes = read_usize("verification_max_bytes", page.max_bytes);
    }
    let verification = mutation_log::verification_summary_page(
        snapshot.revision(),
        &snapshot.checks,
        details.then_some(&page),
    );
    for (key, value) in verification.as_object().unwrap() {
        result[key] = value.clone();
    }
    result
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

    // Prefer before/after comparison for a single file in VS Code or Cursor.
    if as_patch_opt == Some(false) {
        let grouped = changes::session_grouped_changes();
        if grouped.len() == 1 {
            let normalized = if editor_str.eq_ignore_ascii_case("vscode") {
                "code"
            } else {
                editor_str
            };
            let kind = if editor_str.eq_ignore_ascii_case("auto") {
                // Automatic selection allows comparisons only in VS Code or Cursor.
                changes::configured_editor()
            } else {
                EditorKind::from_str(normalized).unwrap_or_else(changes::configured_editor)
            };
            if matches!(kind, EditorKind::Vscode | EditorKind::Cursor) {
                let g = &grouped[0];
                // Prepare temporary before/after files.
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
        // Fall back to a patch when per-file comparison is unavailable.
    }

    // An explicit patch path overrides the session-assets destination.
    if let Some(custom) = patch_file_arg {
        let dest = std::path::PathBuf::from(custom);
        // Require a writable root or registered isolated temporary file,
        // matching FileStore::validate_write_access.
        let in_allowed = crate::ai::tools::storage::file_store::path_within_allowed_roots(&dest);
        let is_registered =
            crate::ai::tools::storage::temp_registry::is_registered(&dest.display().to_string());
        // Root validation resolves relative paths through effective_cwd. Absolute
        // paths within the runtime temp directory are also authorized scratch.
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
        // Open the explicit patch path with the resolved viewer.
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
        // Open directly because open_changes would ignore the explicit path.
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

    // Use the storage entry point for the default patch destination.
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
            // open_changes already wrote the patch. Retain the resolved viewer and
            // prefer its returned path without writing the patch again.
            let resolved = requested_kind.unwrap_or_else(changes::configured_editor);
            // Infer the patch destination from session assets or the temp fallback.
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
    // A comparison needs at least one captured side.
    if g.before_first.is_none() && g.after_last.is_none() {
        return None;
    }
    // Unavailable or truncated snapshots cannot be represented by empty files
    // in code --diff; use the patch path with its explicit limitations instead.
    if !g.snapshots_full() {
        return None;
    }
    // Write comparison files within the session temp directory sandbox.
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

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    use crate::ai::tools::storage::mutation_log::{BeforeState, MutationEntry};

    #[test]
    fn audit_default_show_changes_keeps_legacy_keys_and_session_text() {
        let result = handle_show_changes(&json!({"include_git": false, "limit_snippet": 0}));
        let mut keys = result
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "changed_files",
                "has_session_mutations",
                "is_git_repo",
                "patch_bytes",
                "revision",
                "text",
                "verification",
                "verification_gaps",
            ]
        );
        assert_eq!(
            result["text"],
            "当前会话无文件变更（include_git=false，仅统计 mutation log）。"
        );
        assert_eq!(result["changed_files"], 0);
        assert_eq!(result["has_session_mutations"], false);
        assert_eq!(result["patch_bytes"], 0);
        assert_eq!(result["revision"], 0);
        assert_eq!(result["verification"]["status"], "no_changes");
        assert_eq!(result["verification_gaps"]["count"], 0);
        assert!(result["verification"].get("checks").is_none());
    }

    #[test]
    fn show_changes_verification_paging_arguments_are_observable() {
        use crate::ai::driver::runtime_ctx;
        use crate::ai::tools::storage::mutation_log;
        let root =
            std::env::temp_dir().join(format!("show-changes-pages-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let mut app = crate::ai::middleware::test_util::test_app();
        app.config.history_file = root.join("history.jsonl");
        app.session_id = "show-changes-pages".into();
        let context = runtime_ctx::DriverContext::new(
            app,
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new())),
            std::sync::Arc::new(Vec::new()),
            std::sync::Arc::new(Vec::new()),
        );
        runtime_ctx::SUBAGENT_CWD.sync_scope(root.clone(), || runtime_ctx::DRIVER_CTX.sync_scope(context, || {
            for index in 0..10 {
                assert!(mutation_log::begin_check(&format!("check-{index}"), "execute_command", &json!({"command": format!("cargo check --package project-{index}")})).is_some());
            }
            let path = mutation_log::log_path().unwrap();
            let original = std::fs::read(&path).unwrap();
            let first = handle_show_changes(&json!({"include_git": false, "verification_details": true}));
            assert_eq!(first["verification"]["checks_page"]["shown"], 8);
            assert_eq!(first["verification"]["checks_page"]["total"], 10);
            let next = handle_show_changes(&json!({"include_git": false, "verification_details": true, "verification_offset": 8, "verification_limit": 1, "verification_max_bytes": 0}));
            assert_eq!(next["verification"]["checks_page"]["shown"], 1);
            assert_eq!(next["verification"]["checks_page"]["next_offset"], 9);
            assert_eq!(next["verification"]["checks"][0]["observation"]["originating_command_call_id"], "check-8");
            for max_bytes in [1, 2] {
                let tiny = handle_show_changes(&json!({"include_git": false, "verification_details": true, "verification_max_bytes": max_bytes}));
                assert_eq!(tiny["verification"]["checks"], json!([]));
                let page = &tiny["verification"]["checks_page"];
                assert_eq!(page["max_bytes"], 2);
                assert_eq!(page["shown_bytes"], 2);
                assert!(page["shown_bytes"].as_u64().unwrap() <= page["max_bytes"].as_u64().unwrap());
                assert_eq!(page["next_offset"], 0);
                assert_eq!(page["complete"], false);
            }
            let full = handle_show_changes(&json!({"include_git": false, "verification_full": true, "verification_offset": 9, "verification_limit": 1, "verification_max_bytes": 2}));
            assert_eq!(full["verification"]["checks_page"]["offset"], 0);
            assert_eq!(full["verification"]["checks_page"]["shown"], 10);
            assert_eq!(full["verification"]["checks_page"]["complete"], true);
            assert_eq!(full["verification_gaps"], first["verification_gaps"]);
            assert_eq!(std::fs::read(&path).unwrap(), original);
        }));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn show_changes_verification_uses_committed_log_snapshot() {
        use crate::ai::tools::storage::mutation_log;
        use std::io::Write;

        let root = std::env::temp_dir().join(format!(
            "show-changes-committed-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let mut app = crate::ai::middleware::test_util::test_app();
        app.config.history_file = root.join("history.jsonl");
        app.session_id = "show-changes-committed".into();
        let context = crate::ai::driver::runtime_ctx::DriverContext::new(
            app,
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new())),
            std::sync::Arc::new(Vec::new()),
            std::sync::Arc::new(Vec::new()),
        );
        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(root.clone(), || {
            crate::ai::driver::runtime_ctx::DRIVER_CTX.sync_scope(context, || {
                let path = mutation_log::log_path().unwrap();
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                let entry = MutationEntry {
                    entry_id: None,
                    turn_id: None,
                    task_id: None,
                    tool_call_id: None,
                    seq: 1,
                    ts: "t".into(),
                    path: root.join("file.rs").display().to_string(),
                    op: "write".into(),
                    before: None,
                    before_state: Some(BeforeState::Absent),
                    after: Some("changed\n".into()),
                    diff: None,
                };
                std::fs::write(&path, serde_json::to_string(&entry).unwrap() + "\n").unwrap();
                let pending = mutation_log::begin_check(
                    "check",
                    "execute_command",
                    &json!({"command": "cargo check"}),
                );
                assert!(pending.is_some());
                let mut terminal = mutation_log::read_checks().last().unwrap().clone();
                terminal.status = mutation_log::CheckStatus::Success;
                terminal.observed_success = true;
                let before = mutation_log::recovered_records_at(&path);
                let mut writer = std::fs::OpenOptions::new()
                    .append(true)
                    .open(&path)
                    .unwrap();
                writer
                    .write_all(&serde_json::to_vec(&terminal).unwrap())
                    .unwrap();
                let args = json!({
                    "include_git": false,
                    "limit_snippet": 0,
                    "verification_details": true,
                });
                for _ in 0..3 {
                    let result = handle_show_changes(&args);
                    assert_eq!(result["revision"], 1);
                    assert_eq!(result["changed_files"], 1);
                    assert_eq!(result["verification"]["status"], "running");
                    assert_eq!(result["verification_gaps"]["count"], 1);
                    assert_eq!(
                        result["verification"]["checks"][0]["observation"]["status"],
                        "running"
                    );
                    assert_eq!(mutation_log::verification_reminder_summary().gaps, 1);
                    assert_eq!(mutation_log::recovered_records_at(&path), before);
                }
                writer.write_all(b"\n").unwrap();
                for _ in 0..3 {
                    let result = handle_show_changes(&args);
                    assert_eq!(result["revision"], 1);
                    assert_eq!(result["verification"]["status"], "success");
                    assert_eq!(result["verification_gaps"]["count"], 0);
                    assert_eq!(
                        result["verification"]["checks"][0]["observation"],
                        json!(terminal)
                    );
                    assert_eq!(mutation_log::verification_reminder_summary().gaps, 0);
                    assert_eq!(mutation_log::recovered_records_at(&path), before + 1);
                }
            });
        });
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn show_changes_with_git_exposes_union_inventory_without_changing_legacy_fields() {
        let root = std::env::temp_dir().join(format!(
            "show-changes-inventory-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let root = std::fs::canonicalize(root).unwrap();
        let git = crate::fork_guard::output(
            std::process::Command::new("git")
                .args(["init", "-q"])
                .current_dir(&root),
        )
        .unwrap();
        assert!(git.status.success(), "{:?}", git.stderr);
        let git_path = root.join("git-only\npath.rs");
        std::fs::write(&git_path, "untracked\n").unwrap();
        let entry = MutationEntry {
            entry_id: None,
            turn_id: None,
            task_id: None,
            tool_call_id: None,
            seq: 1,
            ts: "t".into(),
            path: root.join("mutation-only.rs").display().to_string(),
            op: "delete".into(),
            before: Some("removed\n".into()),
            before_state: Some(BeforeState::Present),
            after: None,
            diff: None,
        };
        let mut app = crate::ai::middleware::test_util::test_app();
        app.config.history_file = root.join(".git/audit-test/history.jsonl");
        app.session_id = "show-changes-test".into();
        let context = crate::ai::driver::runtime_ctx::DriverContext::new(
            app,
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new())),
            std::sync::Arc::new(Vec::new()),
            std::sync::Arc::new(Vec::new()),
        );
        crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(root.clone(), || {
            crate::ai::driver::runtime_ctx::DRIVER_CTX.sync_scope(context, || {
                let log_path = crate::ai::tools::storage::mutation_log::log_path().unwrap();
                assert!(log_path.starts_with(root.join(".git")));
                std::fs::create_dir_all(log_path.parent().unwrap()).unwrap();
                let log_entry = serde_json::to_string(&entry).unwrap() + "\n";
                std::fs::write(&log_path, &log_entry).unwrap();
                let patch_bytes = changes::combined_patch().unwrap().len();
                let full = handle_show_changes(&json!({"include_git": true, "limit_snippet": 0}));
                assert_eq!(full["text"], changes::combined_summary());
                let result = handle_show_changes(&json!({
                    "include_git": true,
                    "path": "mutation-only",
                    "limit_snippet": 1,
                }));
                assert!(
                    result["text"]
                        .as_str()
                        .unwrap()
                        .contains("[truncated to 80 chars")
                );
                assert_eq!(result["changed_files"], 1);
                assert_eq!(result["is_git_repo"], true);
                assert_eq!(result["has_session_mutations"], true);
                assert_eq!(result["patch_bytes"], patch_bytes);
                let inventory = changes::collect_current_changes(true);
                assert_eq!(
                    result["changed_paths"],
                    serde_json::to_value(&inventory.files).unwrap()
                );
                assert_eq!(result["changed_paths"], full["changed_paths"]);
                assert_eq!(result["coverage"]["observed_paths"], 2);
                assert_eq!(result["coverage"]["session_changed_files"], 1);
                assert_eq!(result["coverage"]["collection_complete"], true);
                assert_eq!(result["coverage"]["collection_errors"], json!([]));
                assert_eq!(
                    result["coverage"]["coverage_gaps"],
                    json!(inventory.coverage_gaps)
                );
                assert!(
                    !result["coverage"]["coverage_gaps"]
                        .as_array()
                        .unwrap()
                        .is_empty()
                );
                for path in [&entry.path, &git_path.display().to_string()] {
                    assert!(
                        result["changed_paths"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|file| {
                                file["path"] == *path && file["ownership"] == "unknown"
                            })
                    );
                }
                std::fs::write(&log_path, log_entry + "invalid mutation entry\n").unwrap();
                let incomplete =
                    handle_show_changes(&json!({"include_git": true, "limit_snippet": 1}));
                let inventory = changes::collect_current_changes(true);
                assert_eq!(incomplete["coverage"]["collection_complete"], false);
                assert!(!inventory.collection_errors.is_empty());
                assert_eq!(
                    incomplete["coverage"]["collection_errors"],
                    json!(inventory.collection_errors)
                );
                assert_eq!(incomplete["changed_paths"], result["changed_paths"]);
            });
        });
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn external_diff_rejects_unavailable_initial_snapshot_before_spawning() {
        for state in [BeforeState::Present, BeforeState::Unknown] {
            let entries = [MutationEntry {
                entry_id: None,
                turn_id: None,
                task_id: None,
                tool_call_id: None,
                seq: 0,
                ts: "t".into(),
                path: "/proj/file.rs".into(),
                op: "write".into(),
                before: None,
                before_state: Some(state),
                after: Some("replacement\n".into()),
                diff: None,
            }];
            let grouped = changes::grouped_changes(&entries);
            for editor in [EditorKind::Vscode, EditorKind::Cursor] {
                assert!(try_open_single_file_diff(&grouped[0], &editor).is_none());
            }
        }
    }
}

fn open_patch_path_with_editor(
    patch_path: &std::path::Path,
    kind: &EditorKind,
) -> Result<String, String> {
    // Match storage::changes::open_patch while allowing an explicit path.
    let patch_str = patch_path.display().to_string();
    let (prog, args): (&str, Vec<String>) = match kind {
        EditorKind::Vscode => ("code", vec![patch_str.clone()]),
        EditorKind::Cursor => ("cursor", vec![patch_str.clone()]),
        EditorKind::Idea => ("idea", vec!["diff".to_string(), patch_str.clone()]),
        EditorKind::Git => {
            // Git difftool requires repository context for the explicit patch.
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
