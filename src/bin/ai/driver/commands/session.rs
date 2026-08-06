use aios_kernel::primitives::{DaemonKind, DaemonState};
use uuid::Uuid;

use crate::ai::{
    history::{
        PruneSessionDeleteResult, SessionInfo, SessionStore, SessionTitleOrigin,
        SuspendedSessionEntry, SuspendedSessionStore, format_suspended_timestamp_label,
        generate_session_summary,
    },
    types::App,
};

/// 公开给帮助与补全的规范二级命令；旧别名仅为兼容保留，不再主动展示。
pub(in crate::ai) const CANONICAL_SESSION_SUBCOMMANDS: &[&str] = &[
    "help",
    "list",
    "verbose",
    "current",
    "new",
    "use",
    "suspend",
    "bound",
    "unbind",
    "delete",
    "prune",
    "clear-history",
    "clear-all",
    "dump-history",
    "export",
    "archive",
    "import",
    "fork",
    "branch",
];

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
    crate::ai::tools::enable_tools::clear_explicitly_enabled_tools(&app.session_id);
    if let Some(ctx) = app.agent_context.as_mut() {
        ctx.tools.clear();
    }
    app.attached_image_files.clear();
    app.forced_skill = None;
    app.pending_skill_continuation = None;
    app.forced_question = None;
    app.last_skill_bias = None;
    app.stale_patch_targets.clear();
}

fn load_stale_patch_targets(
    store: &SessionStore,
    session_id: &str,
    history_file: &std::path::Path,
) -> std::io::Result<rustc_hash::FxHashSet<std::path::PathBuf>> {
    if let Some(targets) = crate::ai::history::read_stale_patch_targets_sqlite(history_file)? {
        return Ok(targets);
    }

    // 兼容升级前没有专用 meta 的 session：只在首次加载时从尚存的结构化消息
    // 回放一次，随后写回 meta。以后即使历史被压缩，账本也不再依赖消息形态。
    let messages = store.read_all_messages(session_id)?;
    let targets = crate::ai::driver::turn_runtime::stale_patch_targets_from_messages(&messages);
    if history_file.exists() {
        crate::ai::history::write_stale_patch_targets_sqlite(history_file, &targets)?;
    }
    Ok(targets)
}

/// 恢复当前 App 对应 session 的持久化运行时状态。启动恢复、persona 切换与
/// `/sessions use` 必须统一走这里，避免 stale-patch 状态跨 session 污染或丢失。
pub(in crate::ai) fn restore_session_local_runtime_state(app: &mut App) -> std::io::Result<()> {
    let store = SessionStore::new(app.config.history_file.as_path());
    app.stale_patch_targets =
        load_stale_patch_targets(&store, &app.session_id, &app.session_history_file)?;
    Ok(())
}

fn switch_app_to_session(
    app: &mut App,
    store: &SessionStore,
    session_id: &str,
    require_existing: bool,
) -> std::io::Result<()> {
    let history_file = store.session_history_file(session_id);
    if let Err((path, err)) = crate::ai::driver::session_pid::mark_session_pid(
        store.sessions_root(),
        session_id,
        require_existing,
    ) {
        if require_existing {
            return Err(err);
        }
        eprintln!(
            "[Warning] 无法写入 session PID 文件 ({}): {err}",
            path.display()
        );
    }
    // 先登记活跃标记，再加载目标状态，避免 prune 在切换窗口内删除目标 session。
    let stale_patch_targets = load_stale_patch_targets(store, &session_id, &history_file)?;
    clear_session_local_runtime_state(app);
    app.session_id = session_id.to_string();
    app.session_history_file = history_file;
    app.stale_patch_targets = stale_patch_targets;
    app.sync_persona_session_binding();
    Ok(())
}

/// 历史 rewind 会原地替换当前 session 的消息，必须同步重建并持久化账本，不能沿用
/// rewind 之前的 meta，也不能简单清空后让下一次 patch 绕过 fresh-read 门控。
pub(in crate::ai) fn reset_stale_patch_targets_from_messages(
    app: &mut App,
    messages: &[crate::ai::history::Message],
) -> std::io::Result<()> {
    let targets = crate::ai::driver::turn_runtime::stale_patch_targets_from_messages(messages);
    crate::ai::history::write_stale_patch_targets_sqlite(&app.session_history_file, &targets)?;
    app.stale_patch_targets = targets;
    Ok(())
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

/// 解析 export/archive 使用的规范目标选择器，并确保普通 ID 指向现有会话。
fn resolve_existing_session_selector(
    store: &SessionStore,
    current_session_id: &str,
    selector: &str,
) -> std::io::Result<String> {
    let session_id = match selector {
        "current" => current_session_id.to_string(),
        "last" => store
            .list_sessions()?
            .first()
            .map(|session| session.id.clone())
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, "no sessions found")
            })?,
        id => {
            SessionStore::validate_session_id(id)?;
            id.to_string()
        }
    };
    if !store.session_exists(&session_id)? {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("session '{session_id}' not found"),
        ));
    }
    Ok(session_id)
}

/// `/sessions prune <days>` 允许的最大天数（~10k 年）。远小于 `chrono` 的
/// `Duration::days` 与 `DateTime` 运算溢出阈值，用来把非法的超大输入挡在 panic 之前。
/// 有效区间为 `1..=MAX_PRUNE_DAYS`：`0` 会退化成"删除所有非当前 session"，被显式拒绝。
pub(crate) const MAX_PRUNE_DAYS: i64 = 3_650_000;

/// 选出 N 天未活动的 session：`modified_local` 早于 `cutoff` 的视为过期。
/// 当前 session 永不删除；缺少时间戳的 session 无法判定新旧，保守跳过。
pub(crate) fn select_stale_sessions<'a>(
    sessions: &'a [SessionInfo],
    current_session_id: &str,
    cutoff: chrono::DateTime<chrono::Local>,
) -> Vec<&'a SessionInfo> {
    sessions
        .iter()
        .filter(|s| {
            s.id != current_session_id && s.modified_local.map(|t| t < cutoff).unwrap_or(false)
        })
        .collect()
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
    let top_level_fork = cmd == "fork";
    if cmd != "sessions"
        && cmd != "session"
        && cmd != "ss"
        && !top_level_suspend
        && !top_level_close
        && !top_level_fork
    {
        return Ok(false);
    }
    let action = if top_level_suspend {
        "suspend"
    } else if top_level_close {
        "close"
    } else if top_level_fork {
        "fork"
    } else {
        parts.next().unwrap_or("list")
    };
    let store = SessionStore::new(app.config.history_file.as_path());
    let _ = store.ensure_root_dir();

    match action {
        "help" | "h" => {
            println!("Session management commands:");
            println!();
            println!("  /sessions [list]          list all sessions (default, no sizes)");
            println!("  /sessions verbose         list all sessions with per-session sizes");
            println!("  /sessions current         show current session info");
            println!("  /sessions new             create and switch to new session");
            println!("  /sessions use <id>        switch to specified session");
            println!("  /sessions suspend         suspend current session and return to shell");
            println!(
                "  /close                    close and delete current session, then exit (or :close)"
            );
            println!(
                "  /fork                     fork current session into a new branch (keeps original) and switch"
            );
            println!(
                "  /sessions bound           list suspended sessions bound to current terminal"
            );
            println!("  /sessions delete <id> [more...]     delete one or more sessions");
            println!(
                "  /sessions unbind          remove suspended-session bindings for this terminal (sessions are kept)"
            );
            println!(
                "  /sessions clear-history   clear current session history (keeps session alive)"
            );
            println!("  /sessions clear-all       delete all sessions");
            println!(
                "  /sessions prune <days>    delete sessions inactive for N days (current session kept)"
            );
            println!(
                "  /sessions dump-history <id>              dump session history to JSON (<id>-history.json)"
            );
            println!(
                "  /sessions export <id|current|last> [output.md]   export session to Markdown"
            );
            println!(
                "  /sessions archive <id|current|last> [output.zip] full session archive for migration"
            );
            println!(
                "  /sessions import <file.zip> [as=<id>]           import session from archive"
            );
            println!("  /sessions fork [src=<id>] [as=<id>]      copy session to a new branch");
            println!("  /sessions branch <keep_turns> [src=<id>] [as=<id>]");
            println!(
                "                                          fork then retain the first N complete user turns"
            );
            println!();
        }
        "list" | "ls" | "" | "verbose" => {
            // 默认不递归统计每个 session 的大小（assets 递归是 `/ss` 的唯一重活）；
            // 只有显式 `verbose`（`/ss verbose` 或 `/ss list verbose`）才并行统计。
            let verbose =
                action == "verbose" || parts.next().map(|t| t == "verbose").unwrap_or(false);
            let mut sessions = store.list_sessions()?;
            if verbose {
                // 大小按需并行统计：list_sessions 只读元数据，assets 递归统计放到这里多核并行。
                store.attach_session_sizes(&mut sessions)?;
            }
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
                    let size = if verbose {
                        format_size(s.size_bytes)
                    } else {
                        "-".to_string()
                    };
                    println!(
                        "{} {:<width$}  {}  {:>8}  {}",
                        mark,
                        s.id,
                        time,
                        size,
                        summary,
                        width = max_id_len
                    );
                }
                if !verbose {
                    println!("(run `/ss verbose` to include per-session sizes)");
                }
            }
        }
        "current" | "cur" => {
            println!("session: {}", app.session_id);
            println!("history: {}", app.session_history_file.display());
            // 显示 session 摘要
            // 只读当前 session 的预览，避免扫描并统计全部 session。
            if let Ok(Some((summary, modified_local))) = store.read_session_preview(&app.session_id)
            {
                if let Some(summary) = &summary {
                    println!("summary: {}", summary);
                }
                let size = store.session_total_size(&app.session_id).unwrap_or(0);
                println!("size: {}", format_size(size));
                if let Some(t) = modified_local {
                    println!("modified: {}", t.format("%Y-%m-%d %H:%M:%S"));
                }
            }
        }
        "new" | "create" => {
            let new_id = Uuid::new_v4().to_string();
            // 切换前清掉旧 session 的 history cache 与 explicit-enabled tools，
            // 防止下个 turn 携带跨 session 脏状态。
            crate::ai::history::invalidate_context_history_cache_for(&app.session_history_file);
            switch_app_to_session(app, &store, &new_id, false)?;
            println!("Switched to new session: {}", new_id);
        }
        "use" | "select" => {
            let Some(id) = parts.next() else {
                println!("missing session id. try: /sessions use <id>");
                return Ok(true);
            };
            if let Err(error) = SessionStore::validate_session_id(id) {
                println!("invalid session id: {error}");
                return Ok(true);
            }
            if !store.session_exists(id)? {
                println!("Session not found: {id}");
                return Ok(true);
            }
            crate::ai::history::invalidate_context_history_cache_for(&app.session_history_file);
            switch_app_to_session(app, &store, id, true)?;
            println!("Switched session: {}", id);
            // 显示 session 摘要
            // 只读目标 session 预览，避免扫描全部 session。
            if let Ok(Some((summary, _))) = store.read_session_preview(id) {
                if let Some(summary) = &summary {
                    println!("summary: {}", summary);
                }
            }
        }
        "suspend" | "bg" | "detach" => {
            // 与 Ctrl+C 挂起路径（`should_suspend_session_on_sigint`）保持一致：
            // `--session` 显式指定的 id 总是挂起；否则若 session 还没有任何用户消息，
            // 主循环退出时的 `cleanup_one_shot` 会把这个空 session 删除，若此处仍写入
            // 挂起条目，就会留下指向已删除 session 的悬空绑定，下次启动 `a` 会尝试恢复
            // 一个不存在的会话。
            let should_suspend = if app.cli.session.is_some() {
                true
            } else {
                let session_store = SessionStore::new(app.config.history_file.as_path());
                !session_store
                    .is_empty_session(&app.session_id)
                    .unwrap_or(false)
            };
            if !should_suspend {
                println!("当前 session 为空，直接退出（不挂起）。");
                crate::ai::driver::signal::request_shutdown(app.shutdown.as_ref());
                return Ok(true);
            }
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
            if let Some(error) = ids
                .iter()
                .find_map(|id| SessionStore::validate_session_id(id).err())
            {
                println!("invalid session id: {error}");
                return Ok(true);
            }
            let mut deleted_count = 0;
            let mut not_found_count = 0;
            let mut deleted_current = false;
            for id in &ids {
                let deleted_path = store.session_history_file(id);
                // 必须先终止该 session 尚在执行的 subagent；否则删除 SQLite 后，
                // 活跃 Future 仍可能再次写入并重建派生历史。
                crate::ai::tools::task_tools::discard_tasks_for_session(id);
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
                let new_id = Uuid::new_v4().to_string();
                switch_app_to_session(app, &store, &new_id, false)?;
                println!("Switched to new session: {}", new_id);
            }
        }
        "unbind" | "clear-bound" | "clear_bound" | "clear-suspended" | "clear_suspended" => {
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
                "Remove ALL suspended-session bindings for the current terminal? Sessions will be kept. (y/n): ",
            );
            if confirm != Some(true) {
                println!("canceled by user.");
                return Ok(true);
            }

            match suspended_store.clear_current_terminal() {
                Ok(cleared) => {
                    println!(
                        "Removed {cleared} suspended-session binding(s) for the current terminal; sessions were kept."
                    );
                }
                Err(err) => {
                    eprintln!("[sessions clear-bound] {}", err);
                }
            }
        }
        "export" | "export-current" | "export-cur" | "export-last" | "export-latest" => {
            let (selector, output) = match action {
                "export" => {
                    let Some(selector) = parts.next() else {
                        println!(
                            "missing session selector. try: /sessions export <id|current|last> [output.md]"
                        );
                        return Ok(true);
                    };
                    (selector, parts.next())
                }
                "export-current" | "export-cur" => ("current", parts.next()),
                "export-last" | "export-latest" => ("last", parts.next()),
                _ => unreachable!("matched export action"),
            };
            let id = match resolve_existing_session_selector(&store, &app.session_id, selector) {
                Ok(id) => id,
                Err(error) => {
                    eprintln!("Failed to export session: {error}");
                    return Ok(true);
                }
            };
            let output_path = std::path::Path::new(output.unwrap_or("session_export.md"));

            match store.export_session_to_markdown(&id, output_path) {
                Ok(()) => println!("Exported session '{id}' to '{}'", output_path.display()),
                Err(error) => eprintln!("Failed to export session: {error}"),
            }
        }
        "dump-history" | "dump" => {
            let Some(id) = parts.next() else {
                println!("missing session id. try: /sessions dump-history <id>");
                return Ok(true);
            };
            let id = match resolve_existing_session_selector(&store, &app.session_id, id) {
                Ok(id) => id,
                Err(error) => {
                    eprintln!("Failed to dump history: {error}");
                    return Ok(true);
                }
            };
            let output_path = format!("{}-history.json", id);
            let output_path = std::path::Path::new(&output_path);

            match store.read_all_messages(&id) {
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
        "archive"
        | "export-archive"
        | "export-bundle"
        | "pack"
        | "export-archive-current"
        | "export-bundle-current"
        | "pack-current"
        | "pack-cur"
        | "export-archive-last"
        | "export-bundle-last"
        | "pack-last"
        | "pack-latest" => {
            let (selector, output) = match action {
                "archive" | "export-archive" | "export-bundle" | "pack" => {
                    let Some(selector) = parts.next() else {
                        println!(
                            "missing session selector. try: /sessions archive <id|current|last> [output.zip]"
                        );
                        return Ok(true);
                    };
                    (selector, parts.next())
                }
                "export-archive-current"
                | "export-bundle-current"
                | "pack-current"
                | "pack-cur" => ("current", parts.next()),
                "export-archive-last" | "export-bundle-last" | "pack-last" | "pack-latest" => {
                    ("last", parts.next())
                }
                _ => unreachable!("matched archive action"),
            };
            let id = match resolve_existing_session_selector(&store, &app.session_id, selector) {
                Ok(id) => id,
                Err(error) => {
                    eprintln!("Failed to archive session: {error}");
                    return Ok(true);
                }
            };
            let output_path = std::path::Path::new(output.unwrap_or("session_archive.zip"));

            match store.export_session_archive(&id, output_path) {
                Ok(()) => println!("Archived session '{id}' to '{}'", output_path.display()),
                Err(error) => eprintln!("Failed to archive session: {error}"),
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
                    switch_app_to_session(app, &store, &id, true)?;
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
                "Clear current session history and checkpoints? (y/n): ",
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
                "Cleared history and checkpoints for session: {} (session preserved)",
                app.session_id
            );
        }
        "clear-all" | "clear_all" | "wipe" => {
            let confirm =
                crate::commonw::prompt::prompt_yes_or_no_danger("Delete ALL sessions? (y/n): ");
            if confirm != Some(true) {
                println!("canceled by user.");
                return Ok(true);
            }

            let deleted = store.clear_all_sessions()?;
            crate::ai::history::clear_context_history_cache();
            let new_id = Uuid::new_v4().to_string();
            switch_app_to_session(app, &store, &new_id, false)?;
            println!("Deleted {deleted} session(s). Switched to new session: {new_id}");
        }
        "prune" | "delete-old" | "delete_old" | "purge" | "gc" | "stale" => {
            // /sessions prune <days>：删除 N 天未活动的 session（当前 session 永不删除）。
            let Some(days_str) = parts.next() else {
                println!("missing days. try: /sessions prune <days>");
                return Ok(true);
            };
            let Ok(days) = days_str.parse::<i64>() else {
                println!("invalid days: '{}'", days_str);
                return Ok(true);
            };
            // 下界为 1：`prune 0` 的 cutoff 落在 now，等于删除所有非当前 session，
            // 与"清理 N 天未活动"的语义相悖，属于误伤级 footgun，直接拒绝。上界防止
            // `Duration::days` / `DateTime` 运算溢出 panic；~10k 年足够任何清理场景。
            if !(1..=MAX_PRUNE_DAYS).contains(&days) {
                println!(
                    "days must be between 1 and {MAX_PRUNE_DAYS}. try: /sessions prune <days>"
                );
                return Ok(true);
            }
            let cutoff = chrono::Local::now() - chrono::Duration::days(days);
            let sessions = store.list_sessions()?;
            let stale = select_stale_sessions(&sessions, &app.session_id, cutoff);
            if stale.is_empty() {
                println!("No sessions inactive for {} day(s).", days);
                return Ok(true);
            }
            println!(
                "Found {} session(s) inactive for {} day(s) (current session kept):",
                stale.len(),
                days
            );
            let max_id_len = stale.iter().map(|s| s.id.len()).max().unwrap_or(36);
            for s in &stale {
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
                    "  {:<width$}  {}  {}",
                    s.id,
                    time,
                    summary,
                    width = max_id_len
                );
            }
            let confirm = crate::commonw::prompt::prompt_yes_or_no_danger(&format!(
                "Delete these {} session(s)? (y/n): ",
                stale.len()
            ));
            if confirm != Some(true) {
                println!("canceled by user.");
                return Ok(true);
            }
            let mut deleted_count = 0;
            let mut skipped_count = 0;
            let mut failed_count = 0;
            for s in &stale {
                let deleted_path = store.session_history_file(&s.id);
                let session_id = s.id.clone();
                match store.delete_session_if_unchanged(s, cutoff, || {
                    let active =
                        crate::ai::driver::session_pid::scan_session_pids(store.sessions_root())?
                            .into_iter()
                            .any(|(id, _, alive)| alive && id == session_id);
                    if !active {
                        // 最终校验后再终止该 session 的子代理，避免删除后 Future 写回。
                        crate::ai::tools::task_tools::discard_tasks_for_session(&session_id);
                    }
                    Ok(active)
                }) {
                    Ok(PruneSessionDeleteResult::Deleted) => {
                        crate::ai::history::invalidate_context_history_cache_for(&deleted_path);
                        deleted_count += 1;
                    }
                    Ok(PruneSessionDeleteResult::Missing) => {
                        skipped_count += 1;
                        println!("[prune] skipped {}: session no longer exists", s.id);
                    }
                    Ok(PruneSessionDeleteResult::Changed) => {
                        skipped_count += 1;
                        println!(
                            "[prune] skipped {}: session changed after confirmation",
                            s.id
                        );
                    }
                    Ok(PruneSessionDeleteResult::NotExpired) => {
                        skipped_count += 1;
                        println!("[prune] skipped {}: session is no longer expired", s.id);
                    }
                    Ok(PruneSessionDeleteResult::Active) => {
                        skipped_count += 1;
                        println!("[prune] skipped {}: session is active", s.id);
                    }
                    Err(err) => {
                        failed_count += 1;
                        eprintln!("[prune] failed to delete session {}: {}", s.id, err);
                    }
                }
            }
            println!(
                "Prune complete: deleted {deleted_count}, skipped {skipped_count}, failed {failed_count} (inactive for {days} day(s))."
            );
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
                    // 为 fork 出的新 session 写入带深度的 fork 标记标题（支持 fork 的 fork）。
                    if let Err(err) = apply_fork_title(&store, &src_id, &dst_id) {
                        eprintln!("[fork] 无法写入 fork 标记标题: {err}");
                    }
                    crate::ai::history::invalidate_context_history_cache_for(
                        &app.session_history_file,
                    );
                    switch_app_to_session(app, &store, &dst_id, true)?;
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
            // 用法: /sessions branch <keep_turns> [src=<id>] [as=<id>]
            let Some(keep_str) = parts.next() else {
                println!(
                    "missing keep count. try: /sessions branch <keep_turns> [src=<id>] [as=<id>]"
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
                    switch_app_to_session(app, &store, &dst_id, true)?;
                    println!(
                        "Branched '{}' -> '{}' (kept first {} complete user turn(s)), switched to new branch.",
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

/// 解析标题开头的 fork 标记，返回 (已 fork 深度, 去除标记后的内层标题)。
/// 无标记或格式不符时返回 (0, 原标题)。`[fork]` 视为深度 1，`[fork N]` 视为深度 N。
fn parse_fork_marker(title: &str) -> (usize, &str) {
    let trimmed = title.trim_start();
    let Some(rest) = trimmed.strip_prefix('[') else {
        return (0, title);
    };
    let Some(close) = rest.find(']') else {
        return (0, title);
    };
    let tag = &rest[..close];
    let after = rest[close + 1..].trim_start();
    let Some(suffix) = tag.strip_prefix("fork") else {
        return (0, title);
    };
    let suffix = suffix.trim();
    let depth = if suffix.is_empty() {
        1
    } else {
        match suffix.parse::<usize>() {
            Ok(n) if n >= 1 => n,
            _ => return (0, title),
        }
    };
    (depth, after)
}

/// 构造带 fork 深度标记的标题。深度 1 显示为 `[fork]`，N≥2 显示为 `[fork N]`。
/// 总长度限制在 40 字符内，避免被判定为低质量标题而在后续被模型重新生成覆盖。
fn format_fork_title(depth: usize, inner: &str) -> String {
    let marker = if depth <= 1 {
        "[fork]".to_string()
    } else {
        format!("[fork {}]", depth)
    };
    let inner = inner.trim();
    if inner.is_empty() {
        return marker;
    }
    const MAX_TITLE_CHARS: usize = 40;
    let budget = MAX_TITLE_CHARS.saturating_sub(marker.chars().count() + 1);
    let inner = if budget == 0 {
        String::new()
    } else if inner.chars().count() <= budget {
        inner.to_string()
    } else {
        let mut out: String = inner.chars().take(budget - 1).collect();
        out.push('…');
        out
    };
    if inner.is_empty() {
        marker
    } else {
        format!("{} {}", marker, inner)
    }
}

/// 读取源 session 的有效标题作为 fork 标记的基底：优先持久化标题，
/// 否则从首条用户消息生成回退标题。
fn session_title_base(store: &SessionStore, session_id: &str) -> String {
    if let Ok(Some(title)) = store.read_session_title(session_id) {
        let title = title.trim();
        if !title.is_empty() {
            return title.to_string();
        }
    }
    if let Ok(Some(prompt)) = store.first_user_prompt(session_id) {
        let summary = generate_session_summary(&prompt);
        if !summary.is_empty() {
            return summary;
        }
    }
    String::new()
}

/// 把 dst session 的标题改成带 fork 标记的版本，深度基于源 session 标题递增。
fn apply_fork_title(store: &SessionStore, src_id: &str, dst_id: &str) -> std::io::Result<()> {
    let base = session_title_base(store, src_id);
    let (depth, inner) = parse_fork_marker(&base);
    let new_title = format_fork_title(depth + 1, inner);
    store.write_session_title_with_origin(dst_id, &new_title, SessionTitleOrigin::Model)
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
        history::{
            Message, SessionInfo, SuspendedSessionStore, append_history_messages,
            read_stale_patch_targets_sqlite, write_stale_patch_targets_sqlite,
        },
        types::{AgentContext, AppConfig, FunctionCall, SkillBiasMemory, ToolCall},
    };
    use chrono::{Duration, Local};
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
            pending_skill_continuation: None,
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
            stale_patch_targets: Default::default(),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sessions_new_clears_session_local_runtime_state() {
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
        let source_session_id = app.session_id.clone();
        crate::ai::driver::runtime_ctx::TURN_IDENTITY
            .scope((source_session_id.clone(), 0), async {
                crate::ai::tools::enable_tools::set_explicit_enabled_tool_names(vec![
                    "mcp_feishu_doc_create_from_markdown".to_string(),
                ]);
            })
            .await;

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
        crate::ai::driver::runtime_ctx::TURN_IDENTITY
            .scope((source_session_id, 1), async {
                assert!(crate::ai::tools::enable_tools::explicit_enabled_tool_names().is_empty());
            })
            .await;
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn sessions_use_restores_stale_patch_targets_without_cross_session_leakage() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let mut app = test_app(&root);
        let store = SessionStore::new(app.config.history_file.as_path());
        let target_id = "sess-target";
        let target_path = store.session_history_file(target_id);
        append_history_messages(
            &target_path,
            &[Message {
                role: "user".to_string(),
                content: Value::String("target session".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        let target = PathBuf::from("/tmp/target-session.rs");
        write_stale_patch_targets_sqlite(
            &target_path,
            &rustc_hash::FxHashSet::from_iter([target.clone()]),
        )
        .unwrap();
        app.stale_patch_targets
            .insert(PathBuf::from("/tmp/source-session.rs"));

        try_handle_session_command(&mut app, "/sessions use sess-target").unwrap();

        assert_eq!(
            app.stale_patch_targets,
            rustc_hash::FxHashSet::from_iter([target])
        );

        try_handle_session_command(&mut app, "/sessions new").unwrap();
        assert!(
            app.stale_patch_targets.is_empty(),
            "a brand-new session must not inherit the previous session ledger"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn sessions_use_migrates_legacy_stale_patch_state_from_messages() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let mut app = test_app(&root);
        let store = SessionStore::new(app.config.history_file.as_path());
        let target_id = "legacy-stale";
        let target_path = store.session_history_file(target_id);
        let patch_path = PathBuf::from("/tmp/legacy-stale.rs");
        append_history_messages(
            &target_path,
            &[
                Message {
                    role: "assistant".to_string(),
                    content: Value::String(String::new()),
                    tool_calls: Some(vec![ToolCall {
                        id: "legacy-patch".to_string(),
                        tool_type: "function".to_string(),
                        function: FunctionCall {
                            name: "apply_patch".to_string(),
                            arguments: serde_json::json!({
                                "file_path": patch_path,
                                "patch": "@@\n-old\n+new",
                            })
                            .to_string(),
                        },
                    }]),
                    tool_call_id: None,
                    reasoning_content: None,
                },
                Message {
                    role: "tool".to_string(),
                    content: Value::String(
                        "Error: apply_patch failed: ambiguous patch: hunk context matches 2 locations."
                            .to_string(),
                    ),
                    tool_calls: None,
                    tool_call_id: Some("legacy-patch".to_string()),
                    reasoning_content: None,
                },
            ],
        )
        .unwrap();
        assert_eq!(read_stale_patch_targets_sqlite(&target_path).unwrap(), None);

        try_handle_session_command(&mut app, "/sessions use legacy-stale").unwrap();

        assert!(app.stale_patch_targets.contains(&patch_path));
        assert!(
            read_stale_patch_targets_sqlite(&target_path)
                .unwrap()
                .is_some_and(|targets| targets.contains(&patch_path)),
            "legacy replay must be persisted so later compression cannot erase it"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sessions_branch_also_clears_stale_skill_bias_and_explicit_tools() {
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
        let source_session_id = app.session_id.clone();
        crate::ai::driver::runtime_ctx::TURN_IDENTITY
            .scope((source_session_id.clone(), 0), async {
                crate::ai::tools::enable_tools::set_explicit_enabled_tool_names(vec![
                    "mcp_feishu_doc_create_from_markdown".to_string(),
                ]);
            })
            .await;

        try_handle_session_command(&mut app, "/sessions branch 1").unwrap();

        assert!(app.last_skill_bias.is_none());
        assert!(app.forced_skill.is_none());
        assert!(app.forced_question.is_none());
        crate::ai::driver::runtime_ctx::TURN_IDENTITY
            .scope((source_session_id, 1), async {
                assert!(crate::ai::tools::enable_tools::explicit_enabled_tool_names().is_empty());
            })
            .await;
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
        // 写入一条用户消息，使 session 非空（否则新的空 session 保护会拒绝挂起，
        // 因为主循环退出时会删除空 session，留下悬空绑定）。
        let store = SessionStore::new(app.config.history_file.as_path());
        let path = store.session_history_file(&app.session_id);
        append_history_messages(
            &path,
            &[Message {
                role: "user".to_string(),
                content: Value::String("hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
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
    fn bg_on_empty_session_does_not_write_dangling_binding() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let suspended_root = root.join("suspended");
        unsafe {
            std::env::set_var("RUST_TOOLS_SUSPENDED_SESSIONS_DIR", &suspended_root);
            std::env::set_var("TERM_SESSION_ID", "term-empty");
        }

        let mut app = test_app(&root);
        // 不写入任何用户消息，session 为空。
        try_handle_session_command(&mut app, "/bg").unwrap();

        // 仍请求退出，但不能留下挂起绑定（否则主循环删除空 session 后成为悬空绑定）。
        assert!(app.shutdown.load(Ordering::Relaxed));
        let entries = SuspendedSessionStore::new()
            .peek_entries_for_terminal_key("terminal:term-empty")
            .unwrap();
        assert!(
            entries.is_empty(),
            "empty session must not be suspended, got {entries:?}"
        );

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

    #[test]
    fn sessions_branch_retains_complete_user_turns() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let mut app = test_app(&root);
        let store = SessionStore::new(app.config.history_file.as_path());
        let source = store.session_history_file(&app.session_id);
        let message = |role: &str, content: &str| Message {
            role: role.to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        };
        append_history_messages(
            &source,
            &[
                message("user", "first request"),
                message("assistant", "tool call"),
                message("tool", "tool result"),
                message("assistant", "first answer"),
                message("user", "second request"),
                message("assistant", "second answer"),
            ],
        )
        .unwrap();

        try_handle_session_command(&mut app, "/sessions branch 1 as=turn-one").unwrap();

        let branched = store.read_all_messages("turn-one").unwrap();
        assert_eq!(branched.len(), 4);
        assert_eq!(branched[0].role, "user");
        assert_eq!(branched[1].role, "assistant");
        assert_eq!(branched[2].role, "tool");
        assert_eq!(branched[3].role, "assistant");

        let _ = fs::remove_dir_all(root);
    }

    fn make_session_info(id: &str, days_ago: Option<i64>) -> SessionInfo {
        SessionInfo {
            id: id.to_string(),
            modified_local: days_ago.map(|d| Local::now() - Duration::days(d)),
            history_revision: 0,
            size_bytes: 0,
            first_user_prompt: None,
            summary: None,
        }
    }

    #[test]
    fn select_stale_sessions_filters_by_age_and_keeps_current() {
        let sessions = vec![
            make_session_info("old", Some(40)),
            make_session_info("recent", Some(5)),
            make_session_info("current", Some(40)),
            make_session_info("unknown", None),
        ];
        let cutoff = Local::now() - Duration::days(30);
        let stale = select_stale_sessions(&sessions, "current", cutoff);
        let stale_ids: Vec<&str> = stale.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(stale_ids, vec!["old"]);
    }

    /// 把目标 session 文件的 mtime 回拨到 N 天前，用于模拟“N 天未读”。
    fn set_session_activity_days_ago(path: &std::path::Path, days: i64) {
        let activity_unix_ms = (Local::now() - Duration::days(days)).timestamp_millis();
        rusqlite::Connection::open(path)
            .unwrap()
            .execute(
                "INSERT INTO meta(key, value) VALUES('last_activity_unix_ms', ?1) \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                rusqlite::params![activity_unix_ms.to_string()],
            )
            .unwrap();
        use std::time::Duration as StdDuration;
        let time = std::time::SystemTime::now() - StdDuration::from_secs((days as u64) * 86400);
        let times = std::fs::FileTimes::new().set_modified(time);
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_times(times)
            .unwrap();
    }

    #[test]
    fn prune_selects_and_deletes_only_old_disk_sessions() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let app = test_app(&root);
        let store = SessionStore::new(app.config.history_file.as_path());

        // 三个 session：两个旧（>30 天）、一个新（刚刚写入）。
        for id in ["sess-old-a", "sess-old-b", "sess-fresh"] {
            let path = store.session_history_file(id);
            append_history_messages(
                &path,
                &[Message {
                    role: "user".to_string(),
                    content: Value::String(format!("hi {id}")),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                }],
            )
            .unwrap();
        }
        set_session_activity_days_ago(&store.session_history_file("sess-old-a"), 40);
        set_session_activity_days_ago(&store.session_history_file("sess-old-b"), 35);

        let cutoff = Local::now() - Duration::days(30);
        let sessions = store.list_sessions().unwrap();
        let stale = select_stale_sessions(&sessions, &app.session_id, cutoff);
        let mut stale_ids: Vec<String> = stale.iter().map(|s| s.id.clone()).collect();
        stale_ids.sort();
        assert_eq!(
            stale_ids,
            vec!["sess-old-a".to_string(), "sess-old-b".to_string()]
        );

        // 走与 handler 相同的最终重检 + 加锁删除路径，仅省略交互确认与 PID 扫描。
        for s in &stale {
            assert_eq!(
                store
                    .delete_session_if_unchanged(s, cutoff, || Ok(false))
                    .unwrap(),
                PruneSessionDeleteResult::Deleted
            );
        }
        let mut remaining: Vec<String> = store
            .list_sessions()
            .unwrap()
            .into_iter()
            .map(|s| s.id)
            .collect();
        remaining.sort();
        assert_eq!(remaining, vec!["sess-fresh".to_string()]);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn prune_rechecks_activity_and_session_state_before_deleting() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let app = test_app(&root);
        let store = SessionStore::new(app.config.history_file.as_path());
        let session_id = "sess-racy";
        let path = store.session_history_file(session_id);
        let message = |content: &str| Message {
            role: "user".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        };
        append_history_messages(&path, &[message("old")]).unwrap();
        set_session_activity_days_ago(&path, 40);

        let cutoff = Local::now() - Duration::days(30);
        let sessions = store.list_sessions().unwrap();
        let candidate = sessions.iter().find(|s| s.id == session_id).unwrap();
        assert_eq!(
            store
                .delete_session_if_unchanged(candidate, cutoff, || Ok(true))
                .unwrap(),
            PruneSessionDeleteResult::Active
        );
        assert!(path.exists(), "active session must not be deleted");

        append_history_messages(&path, &[message("new activity")]).unwrap();
        assert_eq!(
            store
                .delete_session_if_unchanged(candidate, cutoff, || Ok(false))
                .unwrap(),
            PruneSessionDeleteResult::NotExpired
        );
        assert!(
            path.exists(),
            "session changed after confirmation must be kept"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn prune_rejects_out_of_range_days_without_panic() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let mut app = test_app(&root);
        let store = SessionStore::new(app.config.history_file.as_path());

        // 落盘两个旧 session，用于验证越界输入被拒后不会误删任何数据。
        for id in ["sess-stale-a", "sess-stale-b"] {
            let path = store.session_history_file(id);
            append_history_messages(
                &path,
                &[Message {
                    role: "user".to_string(),
                    content: Value::String(format!("hi {id}")),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                }],
            )
            .unwrap();
        }
        set_session_activity_days_ago(&store.session_history_file("sess-stale-a"), 40);
        set_session_activity_days_ago(&store.session_history_file("sess-stale-b"), 35);

        // `0`（退化成删除所有非当前 session）、负数、恰好越过上界、以及会让 chrono
        // 日期运算溢出 panic 的超大值，都应被优雅拒绝。
        // `1e8` 天回拨会越过 `DateTime` 下界——若无上界守卫，这一行本身就会 panic。
        let over_bound = (MAX_PRUNE_DAYS + 1).to_string();
        let max_i64 = i64::MAX.to_string();
        for bad in [
            "0",
            "-1",
            "100000000",
            over_bound.as_str(),
            max_i64.as_str(),
        ] {
            let handled = try_handle_session_command(&mut app, &format!("/sessions prune {bad}"))
                .expect("prune handler must not error on out-of-range days");
            assert!(
                handled,
                "prune should consume the command for input '{bad}'"
            );
        }

        // 两个旧 session 仍然存在：越界输入在进入删除逻辑之前就返回了。
        let mut remaining: Vec<String> = store
            .list_sessions()
            .unwrap()
            .into_iter()
            .map(|s| s.id)
            .collect();
        remaining.sort();
        assert_eq!(
            remaining,
            vec!["sess-stale-a".to_string(), "sess-stale-b".to_string()]
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn fork_command_marks_title_and_keeps_original() {
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
                    content: Value::String("帮我实现 /fork 功能".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
                Message {
                    role: "assistant".to_string(),
                    content: Value::String("好的，开始实现。".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
            ],
        )
        .unwrap();
        // 给源 session 写一个模型标题，作为 fork 标记的基底。
        store
            .write_session_title_with_origin(
                &app.session_id,
                "实现fork功能",
                SessionTitleOrigin::Model,
            )
            .unwrap();

        // 第一次 fork：标题加上 [fork] 标记，并切到新 session。
        try_handle_session_command(&mut app, "/fork").unwrap();
        let first_fork_id = app.session_id.clone();
        assert_ne!(first_fork_id, "sess-old");
        assert_eq!(
            store.read_session_title(&first_fork_id).unwrap().unwrap(),
            "[fork] 实现fork功能"
        );

        // 原 session 保留未删除，标题不变。
        assert!(store.session_history_file("sess-old").exists());
        assert_eq!(
            store.read_session_title("sess-old").unwrap().unwrap(),
            "实现fork功能"
        );

        // 在 fork 出来的 session 上再 fork：标记深度递增为 [fork 2]。
        try_handle_session_command(&mut app, "/fork").unwrap();
        assert_ne!(app.session_id, first_fork_id);
        assert_eq!(
            store.read_session_title(&app.session_id).unwrap().unwrap(),
            "[fork 2] 实现fork功能"
        );
        // 两个 fork 分支都保留未删除。
        assert!(store.session_history_file(&first_fork_id).exists());
        assert!(store.session_history_file("sess-old").exists());

        let _ = fs::remove_dir_all(root);
    }
}
