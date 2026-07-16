use aios_kernel::primitives::{DaemonKind, DaemonState};
use uuid::Uuid;

use crate::ai::{
    history::{
        SessionStore, SuspendedSessionEntry, SuspendedSessionStore,
        format_suspended_timestamp_label,
    },
    types::App,
};

pub(in crate::ai) fn cancel_current_process_reflection_daemons(app: &App) -> usize {
    let Ok(mut os) = app.os.lock() else {
        return 0;
    };
    let current_pid = os.current_process_id();
    let handles = os
        .list_daemons()
        .into_iter()
        .filter(|entry| {
            entry.parent_pid == current_pid
                && entry.kind == DaemonKind::Reflection
                && entry.state == DaemonState::Running
        })
        .map(|entry| entry.handle)
        .collect::<Vec<_>>();

    let mut cancelled = 0usize;
    for handle in handles {
        if os.cancel_daemon(handle) {
            cancelled += 1;
        }
    }
    cancelled
}

pub(in crate::ai) fn clear_session_local_runtime_state(app: &mut App) {
    cancel_current_process_reflection_daemons(app);
    crate::ai::tools::enable_tools::clear_explicitly_enabled_tools();
    if let Some(ctx) = app.agent_context.as_mut() {
        ctx.tools.clear();
    }
    app.attached_image_files.clear();
    app.forced_skill = None;
    app.forced_question = None;
    app.last_skill_bias = None;
}

fn suspended_session_summary(entry: &SuspendedSessionEntry) -> Option<String> {
    SessionStore::new(entry.history_file.as_path())
        .list_sessions()
        .ok()?
        .into_iter()
        .find(|session| session.id == entry.session_id)
        .and_then(|session| session.summary)
        .filter(|summary| !summary.is_empty())
}

fn print_current_terminal_suspended_sessions(entries: &[SuspendedSessionEntry]) {
    if entries.is_empty() {
        println!("No suspended sessions bound to the current terminal.");
        return;
    }

    println!(
        "Current terminal has {} suspended session(s):",
        entries.len()
    );
    let max_id_len = entries
        .iter()
        .map(|entry| entry.session_id.len())
        .max()
        .unwrap_or(36);
    for (index, entry) in entries.iter().enumerate() {
        println!(
            "  {}. {:<width$}  persona={}  suspended={}",
            index + 1,
            entry.session_id,
            entry.persona_id,
            format_suspended_timestamp_label(&entry.suspended_at),
            width = max_id_len
        );
        if let Some(summary) = suspended_session_summary(entry) {
            println!("     {summary}");
        }
        println!("     history: {}", entry.history_file.display());
    }
}

/// `/clear`：仅清屏（清除终端显示），不触及任何对话历史或会话状态。
pub fn try_handle_clear_command(input: &str) -> bool {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return false;
    }
    let normalized = if let Some(rest) = trimmed.strip_prefix('/') {
        rest
    } else if let Some(rest) = trimmed.strip_prefix(':') {
        rest
    } else {
        return false;
    };
    let mut parts = normalized.split_whitespace();
    let Some(cmd) = parts.next() else {
        return false;
    };
    if cmd != "clear" {
        return false;
    }

    // 清屏：ANSI escape - 清除整屏 + 光标回到左上角
    use std::io::Write;
    print!("\x1b[2J\x1b[H");
    let _ = std::io::stdout().flush();
    true
}

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
    let top_level_suspend = matches!(cmd, "suspend" | "bg" | "detach" | "susp");
    let top_level_close = cmd == "close";
    if cmd != "sessions"
        && cmd != "session"
        && cmd != "ss"
        && !top_level_suspend
        && !top_level_close
    {
        return Ok(false);
    }
    let action = if top_level_suspend {
        "suspend"
    } else if top_level_close {
        "close"
    } else {
        parts.next().unwrap_or("list")
    };
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
            println!(
                "  /sessions suspend         suspend current session and return to shell (or /suspend, /bg, /detach, /susp)"
            );
            println!(
                "  /close                    close and delete current session, then exit (or :close)"
            );
            println!(
                "  /sessions bound           list suspended sessions bound to current terminal"
            );
            println!("  /sessions delete <id> [more...]     delete one or more sessions");
            println!(
                "  /sessions clear-bound     clear suspended sessions bound to current terminal"
            );
            println!(
                "  /sessions clear-history   clear current session history (keeps session alive)"
            );
            println!("  /sessions clear-all       delete all sessions");
            println!("  /sessions export <id> [output.md]       export session to Markdown");
            println!(
                "  /sessions dump-history <id>                dump session history to JSON (<id>-history.json)"
            );
            println!(
                "  /sessions export-current [output.md]    export current session to Markdown"
            );
            println!("  /sessions export-last [output.md]       export latest session to Markdown");
            println!(
                "  /sessions export-archive <id> [output.zip]       full session archive for migration"
            );
            println!("  /sessions export-archive-current [output.zip]    archive current session");
            println!("  /sessions export-archive-last [output.zip]       archive latest session");
            println!(
                "  /sessions import <file.zip> [as=<id>]           import session from archive"
            );
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
                        mark,
                        s.id,
                        time,
                        format_size(s.size_bytes),
                        summary,
                        width = max_id_len
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
            clear_session_local_runtime_state(app);
            app.session_id = new_id.clone();
            app.session_history_file = store.session_history_file(&new_id);
            app.sync_persona_session_binding();
            println!("Switched to new session: {}", new_id);
        }
        "use" | "select" => {
            let Some(id) = parts.next() else {
                println!("missing session id. try: /sessions use <id>");
                return Ok(true);
            };
            crate::ai::history::invalidate_context_history_cache_for(&app.session_history_file);
            clear_session_local_runtime_state(app);
            app.session_id = id.to_string();
            app.session_history_file = store.session_history_file(id);
            app.sync_persona_session_binding();
            println!("Switched session: {}", id);
            // 显示 session 摘要
            let sessions = store.list_sessions().unwrap_or_default();
            if let Some(session) = sessions.iter().find(|s| s.id == id) {
                if let Some(summary) = &session.summary {
                    println!("summary: {}", summary);
                }
            }
        }
        "suspend" | "bg" | "detach" => {
            match SuspendedSessionStore::new().suspend_current_terminal(
                &app.session_id,
                app.config.history_file.as_path(),
                &app.active_persona.id,
                // 保存当前模型，恢复时使用
                &app.current_model,
            ) {
                Ok(entry) => {
                    println!("Suspended session: {}", entry.session_id);
                    println!("Run `a` in this terminal to resume/select it.");
                    println!("Run `a --new-session` to start a fresh session instead.");
                    crate::ai::driver::signal::request_shutdown(app.shutdown.as_ref());
                }
                Err(err) => {
                    eprintln!("[suspend] {}", err);
                }
            }
        }
        "close" => {
            // /close：删除当前 session 后退出交互式对话（与 /suspend 的"保留并回到 shell"
            // 相反，这里直接销毁 session）。复用 store（已在上方构造）。
            let current_id = app.session_id.clone();
            let deleted_path = app.session_history_file.clone();
            match store.delete_session(&current_id) {
                Ok(true) => {
                    crate::ai::history::invalidate_context_history_cache_for(&deleted_path);
                    println!("Closed and deleted session: {}", current_id);
                }
                Ok(false) => {
                    println!("Session already removed: {}", current_id);
                }
                Err(err) => {
                    eprintln!("[close] failed to delete session: {}", err);
                }
            }
            crate::ai::driver::signal::request_shutdown(app.shutdown.as_ref());
        }
        "bound" | "bindings" | "suspended" => {
            match SuspendedSessionStore::new().list_current_terminal() {
                Ok(entries) => print_current_terminal_suspended_sessions(&entries),
                Err(err) => eprintln!("[sessions bound] {}", err),
            }
        }
        "delete" | "del" | "rm" => {
            let ids: Vec<&str> = parts.collect();
            if ids.is_empty() {
                println!("missing session id(s). try: /sessions delete <id1> [<id2> ...]");
                return Ok(true);
            };
            let mut deleted_count = 0;
            let mut not_found_count = 0;
            let mut deleted_current = false;
            for id in &ids {
                let deleted_path = store.session_history_file(id);
                let deleted = store.delete_session(id)?;
                if deleted {
                    crate::ai::history::invalidate_context_history_cache_for(&deleted_path);
                    deleted_count += 1;
                    if *id == app.session_id {
                        deleted_current = true;
                    }
                    println!("Deleted session: {}", id);
                } else {
                    not_found_count += 1;
                    println!("Session not found: {}", id);
                }
            }
            if ids.len() > 1 {
                println!("Summary: {deleted_count} deleted, {not_found_count} not found.");
            }
            if deleted_current {
                clear_session_local_runtime_state(app);
                let new_id = Uuid::new_v4().to_string();
                app.session_id = new_id.clone();
                app.session_history_file = store.session_history_file(&new_id);
                app.sync_persona_session_binding();
                println!("Switched to new session: {}", new_id);
            }
        }
        "clear-bound" | "clear_bound" | "clear-suspended" | "clear_suspended" => {
            let suspended_store = SuspendedSessionStore::new();
            let entries = match suspended_store.list_current_terminal() {
                Ok(entries) => entries,
                Err(err) => {
                    eprintln!("[sessions clear-bound] {}", err);
                    return Ok(true);
                }
            };
            if entries.is_empty() {
                println!("No suspended sessions bound to the current terminal.");
                return Ok(true);
            }

            let confirm = crate::commonw::prompt::prompt_yes_or_no_interruptible(
                "Clear ALL suspended sessions bound to the current terminal? (y/n): ",
            );
            if confirm != Some(true) {
                println!("canceled by user.");
                return Ok(true);
            }

            match suspended_store.clear_current_terminal() {
                Ok(cleared) => {
                    println!(
                        "Cleared {cleared} suspended session(s) bound to the current terminal."
                    );
                }
                Err(err) => {
                    eprintln!("[sessions clear-bound] {}", err);
                }
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
        "dump-history" | "dump" => {
            let Some(id) = parts.next() else {
                println!("missing session id. try: /sessions dump-history <id>");
                return Ok(true);
            };
            let output_path = format!("{}-history.json", id);
            let output_path = std::path::Path::new(&output_path);

            match store.read_all_messages(id) {
                Ok(messages) => {
                    let json = serde_json::to_string_pretty(&messages)?;
                    if let Some(parent) = output_path.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::write(output_path, json)?;
                    println!(
                        "Dumped history of session '{}' to '{}'",
                        id,
                        output_path.display()
                    );
                }
                Err(err) => {
                    eprintln!("Failed to dump history: {}", err);
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
        "export-archive" | "export-bundle" | "pack" => {
            let Some(id) = parts.next() else {
                println!("missing session id. try: /sessions export-archive <id> [output.zip]");
                return Ok(true);
            };
            let output_path = parts.next().unwrap_or("session_archive.zip");
            let output_path = std::path::Path::new(output_path);

            match store.export_session_archive(id, output_path) {
                Ok(()) => {
                    println!("Archived session '{}' to '{}'", id, output_path.display());
                }
                Err(err) => {
                    eprintln!("Failed to export archive: {}", err);
                }
            }
        }
        "export-archive-current" | "export-bundle-current" | "pack-current" | "pack-cur" => {
            let output_path = parts.next().unwrap_or("session_archive.zip");
            let output_path = std::path::Path::new(output_path);

            match store.export_session_archive(&app.session_id, output_path) {
                Ok(()) => {
                    println!(
                        "Archived current session '{}' to '{}'",
                        app.session_id,
                        output_path.display()
                    );
                }
                Err(err) => {
                    eprintln!("Failed to export archive: {}", err);
                }
            }
        }
        "export-archive-last" | "export-bundle-last" | "pack-last" | "pack-latest" => {
            let sessions = store.list_sessions()?;
            let Some(last) = sessions.first() else {
                println!("No sessions found to archive.");
                return Ok(true);
            };
            let output_path = parts.next().unwrap_or("session_archive.zip");
            let output_path = std::path::Path::new(output_path);

            match store.export_session_archive(&last.id, output_path) {
                Ok(()) => {
                    println!(
                        "Archived latest session '{}' to '{}'",
                        last.id,
                        output_path.display()
                    );
                }
                Err(err) => {
                    eprintln!("Failed to export archive: {}", err);
                }
            }
        }
        "import" | "import-archive" | "unpack" => {
            let Some(file) = parts.next() else {
                println!("missing archive file. try: /sessions import <file.zip> [as=<id>]");
                return Ok(true);
            };
            let archive_path = std::path::Path::new(file);
            // 可选 as=<id> 指定导入后的 session id
            let mut dst: Option<String> = None;
            for arg in parts.by_ref() {
                if let Some(v) = arg.strip_prefix("as=") {
                    dst = Some(v.to_string());
                }
            }
            let dst_id = dst.unwrap_or_else(|| Uuid::new_v4().to_string());

            match store.import_session_archive(archive_path, &dst_id) {
                Ok(id) => {
                    crate::ai::history::invalidate_context_history_cache_for(
                        &app.session_history_file,
                    );
                    clear_session_local_runtime_state(app);
                    app.session_id = id.clone();
                    app.session_history_file = store.session_history_file(&id);
                    app.sync_persona_session_binding();
                    println!(
                        "Imported session from '{}' -> '{}', switched to it.",
                        file, id
                    );
                }
                Err(err) => {
                    eprintln!("Failed to import session: {}", err);
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
            clear_session_local_runtime_state(app);
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
            clear_session_local_runtime_state(app);
            let new_id = Uuid::new_v4().to_string();
            app.session_id = new_id.clone();
            app.session_history_file = store.session_history_file(&new_id);
            app.sync_persona_session_binding();
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
                    clear_session_local_runtime_state(app);
                    app.session_id = dst_id.clone();
                    app.session_history_file = store.session_history_file(&dst_id);
                    app.sync_persona_session_binding();
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
                    crate::ai::history::invalidate_context_history_cache_for(
                        &app.session_history_file,
                    );
                    clear_session_local_runtime_state(app);
                    app.session_id = dst_id.clone();
                    app.session_history_file = store.session_history_file(&dst_id);
                    app.sync_persona_session_binding();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{
        cli::ParsedCli,
        history::{Message, SuspendedSessionStore, append_history_messages},
        types::{AgentContext, AppConfig, SkillBiasMemory},
    };
    use serde_json::Value;
    use std::{
        fs,
        path::PathBuf,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
    };

    fn test_history_root() -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("rust_tools-session-tests-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn test_app(root: &std::path::Path) -> App {
        let history_file = root.join("history.sqlite");
        let session_store = SessionStore::new(history_file.as_path());
        let session_id = "sess-old".to_string();
        App {
            cli: ParsedCli::default(),
            config: AppConfig {
                api_key: String::new(),
                base_history_file: history_file.clone(),
                history_file: history_file.clone(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 12000,
                history_keep_last: 8,
                history_summary_max_chars: 4000,
                intent_model: None,
                agent_route_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/agent_route/agent_route_model.json"),
                skill_match_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/skill_match/skill_match_model.json"),
            },
            session_id: session_id.clone(),
            session_history_file: session_store.session_history_file(&session_id),
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::new(),
            current_model: crate::ai::model_names::all()
                .first()
                .map(|model| crate::ai::model_names::model_handle(model))
                .expect("models.json is empty"),
            current_agent: "build".to_string(),
            current_agent_manifest: None,
            pending_files: None,
            forced_skill: Some("feishu-upload-md".to_string()),
            forced_question: Some("把 markdown 发到飞书".to_string()),
            attached_image_files: vec!["/tmp/demo.png".to_string()],
            shutdown: Arc::new(AtomicBool::new(false)),
            streaming: Arc::new(AtomicBool::new(false)),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            ignore_next_prompt_interrupt: false,
            prompt_editor: None,
            agent_context: Some(AgentContext::default()),
            last_skill_bias: Some(SkillBiasMemory {
                skill_name: "feishu-upload-md".to_string(),
                question: "把 markdown 发到飞书".to_string(),
            }),
            os: crate::ai::driver::new_local_kernel(),
            agent_reload_counter: None,
            observers: vec![Box::new(
                crate::ai::driver::thinking::ThinkingOrchestrator::new(),
            )],
            last_known_prompt_tokens: None,
            last_known_cached_prompt_tokens: None,
            goal_mode: None,
            last_turn_had_tool_calls: false,
            last_turn_interrupted: false,
            prune_marks: Default::default(),
            turn_reasoning_items: Default::default(),
        }
    }

    #[test]
    fn sessions_new_clears_session_local_runtime_state() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let mut app = test_app(&root);
        if let Some(ctx) = app.agent_context.as_mut() {
            ctx.tools.push(crate::ai::types::ToolDefinition {
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionDefinition {
                    name: "read_file".to_string(),
                    description: String::new(),
                    parameters: serde_json::json!({}),
                },
            });
        }
        crate::ai::tools::enable_tools::set_explicit_enabled_tool_names(vec![
            "mcp_feishu_doc_create_from_markdown".to_string(),
        ]);

        try_handle_session_command(&mut app, "/sessions new").unwrap();

        assert!(app.last_skill_bias.is_none());
        assert!(app.forced_skill.is_none());
        assert!(app.forced_question.is_none());
        assert!(app.attached_image_files.is_empty());
        assert!(
            app.agent_context
                .as_ref()
                .is_some_and(|ctx| ctx.tools.is_empty())
        );
        assert!(crate::ai::tools::enable_tools::explicit_enabled_tool_names().is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn sessions_branch_also_clears_stale_skill_bias_and_explicit_tools() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let mut app = test_app(&root);
        let store = SessionStore::new(app.config.history_file.as_path());
        let src_path = store.session_history_file(&app.session_id);
        append_history_messages(
            &src_path,
            &[
                Message {
                    role: "user".to_string(),
                    content: Value::String("u0".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
                Message {
                    role: "assistant".to_string(),
                    content: Value::String("a0".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
            ],
        )
        .unwrap();
        crate::ai::tools::enable_tools::set_explicit_enabled_tool_names(vec![
            "mcp_feishu_doc_create_from_markdown".to_string(),
        ]);

        try_handle_session_command(&mut app, "/sessions branch 1").unwrap();

        assert!(app.last_skill_bias.is_none());
        assert!(app.forced_skill.is_none());
        assert!(app.forced_question.is_none());
        assert!(crate::ai::tools::enable_tools::explicit_enabled_tool_names().is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn sessions_suspend_persists_entry_and_requests_shutdown() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let suspended_root = root.join("suspended");
        unsafe {
            std::env::set_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR", &suspended_root);
            std::env::set_var("TERM_SESSION_ID", "term-123");
        }

        let mut app = test_app(&root);
        try_handle_session_command(&mut app, "/sessions suspend").unwrap();

        assert!(app.shutdown.load(Ordering::Relaxed));
        let entry = SuspendedSessionStore::new()
            .take_for_terminal_key("terminal:term-123")
            .unwrap()
            .expect("suspended session entry should exist");
        assert_eq!(entry.session_id, app.session_id);
        assert_eq!(entry.history_file, app.config.history_file);
        assert_eq!(entry.persona_id, app.active_persona.id);

        unsafe {
            std::env::remove_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR");
            std::env::remove_var("TERM_SESSION_ID");
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn sessions_bound_lists_current_terminal_entries_without_consuming_them() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let suspended_root = root.join("suspended");
        unsafe {
            std::env::set_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR", &suspended_root);
            std::env::set_var("TERM_SESSION_ID", "term-bound");
        }

        let mut app = test_app(&root);
        let other_history = root.join("other.sqlite");
        SuspendedSessionStore::new()
            .save_for_terminal_key(
                "terminal:term-bound",
                &app.session_id,
                &app.config.history_file,
                &app.active_persona.id,
                "test-model",
            )
            .unwrap();
        SuspendedSessionStore::new()
            .save_for_terminal_key(
                "terminal:term-bound",
                "sess-2",
                &other_history,
                "reviewer",
                "other-model",
            )
            .unwrap();

        assert!(try_handle_session_command(&mut app, "/sessions bound").unwrap());

        let entries = SuspendedSessionStore::new()
            .peek_entries_for_terminal_key("terminal:term-bound")
            .unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].session_id, "sess-2");
        assert_eq!(entries[1].session_id, app.session_id);

        unsafe {
            std::env::remove_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR");
            std::env::remove_var("TERM_SESSION_ID");
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn export_import_archive_roundtrip_preserves_messages() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let app = test_app(&root);
        let store = SessionStore::new(app.config.history_file.as_path());

        // 写入测试消息
        let src_path = store.session_history_file(&app.session_id);
        let original_messages = [
            Message {
                role: "user".to_string(),
                content: Value::String("hello world".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String("hi there".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];
        append_history_messages(&src_path, &original_messages).unwrap();

        // 导出到 zip
        let archive_path = root.join("export.zip");
        store
            .export_session_archive(&app.session_id, &archive_path)
            .expect("export should succeed");
        assert!(archive_path.exists(), "archive file should exist");

        // 导入为新 session
        let dst_id = "imported-session".to_string();
        let result = store.import_session_archive(&archive_path, &dst_id);
        assert!(result.is_ok(), "import should succeed: {:?}", result.err());

        // 验证导入后的消息与原始消息一致
        let imported_messages = store.read_all_messages(&dst_id).unwrap();
        assert_eq!(imported_messages.len(), original_messages.len());
        assert_eq!(imported_messages[0].role, "user");
        assert_eq!(
            imported_messages[0].content,
            Value::String("hello world".to_string())
        );
        assert_eq!(imported_messages[1].role, "assistant");
        assert_eq!(
            imported_messages[1].content,
            Value::String("hi there".to_string())
        );

        let _ = fs::remove_dir_all(root);
    }
}
