use uuid::Uuid;

use crate::ai::{history::SessionStore, types::App};

pub fn try_handle_session_command(
    app: &mut App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let normalized = if let Some(rest) = trimmed.strip_prefix('/') {
        rest
    } else if let Some(rest) = trimmed.strip_prefix(':') {
        rest
    } else {
        return Ok(false);
    };
    let mut parts = normalized.split_whitespace();
    let Some(cmd) = parts.next() else {
        return Ok(false);
    };
    if cmd != "sessions" && cmd != "session" && cmd != "ss" {
        return Ok(false);
    }
    let action = parts.next().unwrap_or("list");
    let store = SessionStore::new(app.config.history_file.as_path());
    let _ = store.ensure_root_dir();

    match action {
        "help" | "h" => {
            println!("Session management commands:");
            println!();
            println!("  /sessions                 list all sessions");
            println!("  /sessions list            list all sessions");
            println!("  /sessions current         show current session info");
            println!("  /sessions new             create and switch to new session");
            println!("  /sessions use <id>        switch to specified session");
            println!("  /sessions delete <id>     delete specified session");
            println!(
                "  /sessions clear-history   clear current session history (keeps session alive)"
            );
            println!("  /sessions clear-all       delete all sessions");
            println!("  /sessions export <id> [output.md]       export session to Markdown");
            println!(
                "  /sessions export-current [output.md]    export current session to Markdown"
            );
            println!("  /sessions export-last [output.md]       export latest session to Markdown");
            println!("  /sessions fork [src=<id>] [as=<id>]      copy session to a new branch");
            println!("  /sessions branch <keep_messages> [src=<id>] [as=<id>]");
            println!(
                "                                          fork then truncate to first N messages"
            );
            println!();
        }
        "list" | "ls" | "" => {
            let sessions = store.list_sessions()?;
            if sessions.is_empty() {
                println!("No sessions.");
            } else {
                // 计算最大 ID 长度用于对齐
                let max_id_len = sessions.iter().map(|s| s.id.len()).max().unwrap_or(36);
                for s in &sessions {
                    let mark = if s.id == app.session_id { "*" } else { " " };
                    let time = s
                        .modified_local
                        .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                        .unwrap_or_else(|| "-".to_string());
                    let summary = s
                        .summary
                        .as_deref()
                        .filter(|v| !v.is_empty())
                        .unwrap_or("-");
                    println!(
                        "{} {:<width$}  {}  {:>8}  {}",
                        mark, s.id, time, format_size(s.size_bytes), summary, width = max_id_len
                    );
                }
            }
        }
        "current" | "cur" => {
            println!("session: {}", app.session_id);
            println!("history: {}", app.session_history_file.display());
            // 显示 session 摘要
            let sessions = store.list_sessions().unwrap_or_default();
            if let Some(current) = sessions.iter().find(|s| s.id == app.session_id) {
                if let Some(summary) = &current.summary {
                    println!("summary: {}", summary);
                }
                println!("size: {}", format_size(current.size_bytes));
                if let Some(t) = current.modified_local {
                    println!("modified: {}", t.format("%Y-%m-%d %H:%M:%S"));
                }
            }
        }
        "new" | "create" => {
            let new_id = Uuid::new_v4().to_string();
            // 切换前清掉旧 session 的 history cache 与 explicit-enabled tools，
            // 防止下个 turn 携带跨 session 脏状态。
            crate::ai::history::invalidate_context_history_cache_for(&app.session_history_file);
            crate::ai::tools::enable_tools::clear_explicitly_enabled_tools();
            app.session_id = new_id.clone();
            app.session_history_file = store.session_history_file(&new_id);
            println!("Switched to new session: {}", new_id);
        }
        "use" | "select" => {
            let Some(id) = parts.next() else {
                println!("missing session id. try: /sessions use <id>");
                return Ok(true);
            };
            crate::ai::history::invalidate_context_history_cache_for(&app.session_history_file);
            crate::ai::tools::enable_tools::clear_explicitly_enabled_tools();
            app.session_id = id.to_string();
            app.session_history_file = store.session_history_file(id);
            println!("Switched session: {}", id);
            // 显示 session 摘要
            let sessions = store.list_sessions().unwrap_or_default();
            if let Some(session) = sessions.iter().find(|s| s.id == id) {
                if let Some(summary) = &session.summary {
                    println!("summary: {}", summary);
                }
            }
        }
        "delete" | "del" | "rm" => {
            let Some(id) = parts.next() else {
                println!("missing session id. try: /sessions delete <id>");
                return Ok(true);
            };
            let deleted_path = store.session_history_file(id);
            let deleted = store.delete_session(id)?;
            if deleted {
                crate::ai::history::invalidate_context_history_cache_for(&deleted_path);
                if id == app.session_id {
                    crate::ai::tools::enable_tools::clear_explicitly_enabled_tools();
                    let new_id = Uuid::new_v4().to_string();
                    app.session_id = new_id.clone();
                    app.session_history_file = store.session_history_file(&new_id);
                    println!(
                        "Deleted current session. Switched to new session: {}",
                        new_id
                    );
                } else {
                    println!("Deleted session: {}", id);
                }
            } else {
                println!("Session not found: {}", id);
            }
        }
        "export" => {
            let Some(id) = parts.next() else {
                println!("missing session id. try: /sessions export <id> [output.md]");
                return Ok(true);
            };
            let output_path = parts.next().unwrap_or("session_export.md");
            let output_path = std::path::Path::new(output_path);

            match store.export_session_to_markdown(id, output_path) {
                Ok(()) => {
                    println!("Exported session '{}' to '{}'", id, output_path.display());
                }
                Err(err) => {
                    eprintln!("Failed to export session: {}", err);
                }
            }
        }
        "export-current" | "export-cur" => {
            let output_path = parts.next().unwrap_or("session_export.md");
            let output_path = std::path::Path::new(output_path);

            match store.export_session_to_markdown(&app.session_id, output_path) {
                Ok(()) => {
                    println!(
                        "Exported current session '{}' to '{}'",
                        app.session_id,
                        output_path.display()
                    );
                }
                Err(err) => {
                    eprintln!("Failed to export session: {}", err);
                }
            }
        }
        "export-last" | "export-latest" => {
            let sessions = store.list_sessions()?;
            let Some(last) = sessions.first() else {
                println!("No sessions found to export.");
                return Ok(true);
            };
            let output_path = parts.next().unwrap_or("session_export.md");
            let output_path = std::path::Path::new(output_path);

            match store.export_session_to_markdown(&last.id, output_path) {
                Ok(()) => {
                    println!(
                        "Exported latest session '{}' to '{}'",
                        last.id,
                        output_path.display()
                    );
                }
                Err(err) => {
                    eprintln!("Failed to export session: {}", err);
                }
            }
        }
        "clear-history" | "clear_history" | "ch" => {
            let confirm = crate::commonw::prompt::prompt_yes_or_no_interruptible(
                "Clear current session history? (y/n): ",
            );
            if confirm != Some(true) {
                println!("canceled by user.");
                return Ok(true);
            }

            store.clear_session_history(&app.session_id)?;
            // 清掉关联的 history cache 与 explicit-enabled tools，避免下个 turn
            // 命中陈旧缓存或携带已经无意义的工具列表。
            crate::ai::history::invalidate_context_history_cache_for(&app.session_history_file);
            crate::ai::tools::enable_tools::clear_explicitly_enabled_tools();
            if let Some(ctx) = app.agent_context.as_mut() {
                ctx.tools.clear();
            }
            println!(
                "Cleared history for session: {} (session preserved)",
                app.session_id
            );
        }
        "clear-all" | "clear_all" | "clear" | "wipe" => {
            let confirm = crate::commonw::prompt::prompt_yes_or_no_interruptible(
                "Delete ALL sessions? (y/n): ",
            );
            if confirm != Some(true) {
                println!("canceled by user.");
                return Ok(true);
            }

            let deleted = store.clear_all_sessions()?;
            crate::ai::history::clear_context_history_cache();
            crate::ai::tools::enable_tools::clear_explicitly_enabled_tools();
            let new_id = Uuid::new_v4().to_string();
            app.session_id = new_id.clone();
            app.session_history_file = store.session_history_file(&new_id);
            println!("Deleted {deleted} session(s). Switched to new session: {new_id}");
        }
        "fork" => {
            // 解析 src=<id> / as=<id>，未指定 src 时默认基于当前 session。
            let mut src: Option<String> = None;
            let mut dst: Option<String> = None;
            for arg in parts.by_ref() {
                if let Some(v) = arg.strip_prefix("src=") {
                    src = Some(v.to_string());
                } else if let Some(v) = arg.strip_prefix("as=") {
                    dst = Some(v.to_string());
                }
            }
            let src_id = src.unwrap_or_else(|| app.session_id.clone());
            let dst_id = dst.unwrap_or_else(|| Uuid::new_v4().to_string());
            match store.fork_session(&src_id, &dst_id) {
                Ok(()) => {
                    crate::ai::history::invalidate_context_history_cache_for(
                        &app.session_history_file,
                    );
                    crate::ai::tools::enable_tools::clear_explicitly_enabled_tools();
                    app.session_id = dst_id.clone();
                    app.session_history_file = store.session_history_file(&dst_id);
                    if let Some(ctx) = app.agent_context.as_mut() {
                        ctx.tools.clear();
                    }
                    println!(
                        "Forked '{}' -> '{}', switched to new branch.",
                        src_id, dst_id
                    );
                }
                Err(err) => {
                    eprintln!("Failed to fork session: {}", err);
                }
            }
        }
        "branch" => {
            // 用法: /sessions branch <keep_messages> [src=<id>] [as=<id>]
            let Some(keep_str) = parts.next() else {
                println!(
                    "missing keep count. try: /sessions branch <keep_messages> [src=<id>] [as=<id>]"
                );
                return Ok(true);
            };
            let Ok(keep) = keep_str.parse::<usize>() else {
                println!("invalid keep count: '{}'", keep_str);
                return Ok(true);
            };
            let mut src: Option<String> = None;
            let mut dst: Option<String> = None;
            for arg in parts.by_ref() {
                if let Some(v) = arg.strip_prefix("src=") {
                    src = Some(v.to_string());
                } else if let Some(v) = arg.strip_prefix("as=") {
                    dst = Some(v.to_string());
                }
            }
            let src_id = src.unwrap_or_else(|| app.session_id.clone());
            let dst_id = dst.unwrap_or_else(|| Uuid::new_v4().to_string());
            match store.branch_session(&src_id, &dst_id, keep) {
                Ok(()) => {
                    app.session_id = dst_id.clone();
                    app.session_history_file = store.session_history_file(&dst_id);
                    if let Some(ctx) = app.agent_context.as_mut() {
                        ctx.tools.clear();
                    }
                    println!(
                        "Branched '{}' -> '{}' (kept first {} message(s)), switched to new branch.",
                        src_id, dst_id, keep
                    );
                }
                Err(err) => {
                    eprintln!("Failed to branch session: {}", err);
                }
            }
        }
        _ => {
            println!("unknown action: '{}'. try: /sessions help", action);
        }
    }
    Ok(true)
}

fn sanitize_session_prompt(s: &str) -> String {
    s.lines()
        .next()
        .unwrap_or(s)
        .replace('\n', " ")
        .replace('\r', "")
}

fn truncate_session_prompt(s: &str, max_len: usize) -> String {
    if max_len == 0 {
        return String::new();
    }
    let char_count = s.chars().count();
    if char_count <= max_len {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_len).collect();
    out.push_str("...");
    out
}

/// 格式化文件大小为人类可读格式（KB/MB/GB）。
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.1}GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1}KB", bytes as f64 / KB as f64)
    } else {
        format!("{}B", bytes)
    }
}
