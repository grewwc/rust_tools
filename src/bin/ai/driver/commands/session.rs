use aios_kernel::primitives::{DaemonKind, DaemonState};
use std::io::IsTerminal;
use uuid::Uuid;

use crate::ai::{
    history::{
        PruneSessionDeleteResult, SessionInfo, SessionStore, SessionTitleOrigin,
        SuspendedSessionEntry, SuspendedSessionStore, format_suspended_timestamp_label,
        generate_session_summary,
    },
    types::App,
};

/// Canonical subcommands exposed to help and completion; legacy aliases are kept
/// only for compatibility and no longer advertised.
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

/// Highlight style for sessions marked important via `/mark`. ANSI colors are
/// only emitted when stdout is a terminal; piped output stays plain.
fn marked_session_style() -> (&'static str, &'static str) {
    if std::io::stdout().is_terminal() {
        (crate::ai::theme::ACCENT_MARKED, crate::ai::theme::RESET)
    } else {
        ("", "")
    }
}

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
    app.forced_skills.clear();
    app.forced_skill_source = None;
    app.pending_skill_continuation = None;
    app.forced_question = None;
    app.last_skill_bias = None;
    app.stale_patch_targets.clear();
    app.prune_marks.clear();
}

fn load_stale_patch_targets(
    store: &SessionStore,
    session_id: &str,
    history_file: &std::path::Path,
) -> std::io::Result<rustc_hash::FxHashSet<std::path::PathBuf>> {
    if let Some(targets) = crate::ai::history::read_stale_patch_targets_sqlite(history_file)? {
        return Ok(targets);
    }

    // Compatibility with sessions created before the dedicated meta existed:
    // replay once from surviving structured messages on first load only, then
    // write back to meta. Afterwards the ledger never depends on message shape
    // again, even if history is compressed.
    let messages = store.read_all_messages(session_id)?;
    let targets = crate::ai::driver::turn_runtime::stale_patch_targets_from_messages(&messages);
    if history_file.exists() {
        crate::ai::history::write_stale_patch_targets_sqlite(history_file, &targets)?;
    }
    Ok(targets)
}

/// Restores the persisted runtime state of the current App's session. Startup
/// restore, persona switching, and `/sessions use` must all go through here, so
/// stale-patch state neither pollutes across sessions nor gets lost.
pub(in crate::ai) fn restore_session_local_runtime_state(app: &mut App) -> std::io::Result<()> {
    let store = SessionStore::new(app.config.history_file.as_path());
    app.stale_patch_targets =
        load_stale_patch_targets(&store, &app.session_id, &app.session_history_file)?;
    restore_prune_marks_for_history(app)
}

/// Restores model prune counters from the current `session_history_file`.
/// Sub-agents own an independent history and must not reuse the parent session's
/// in-memory state from `App::clone`; resume must also restore from the child
/// history.
pub(in crate::ai) fn restore_prune_marks_for_history(app: &mut App) -> std::io::Result<()> {
    app.prune_marks.clear();
    let mut prune_marks =
        match crate::ai::history::read_llm_prune_marks_sqlite(&app.session_history_file) {
            Ok(marks) => marks,
            Err(error) => {
                if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
                    eprintln!("[context-prune] failed to restore mark state: {error}");
                }
                return Ok(());
            }
        };
    if !prune_marks.is_empty() {
        // Meta may come from a longer pre-rewind/branch history. Before restoring,
        // keep only ids that still exist in the target session's current projection
        // and are prunable, so old counters do not bind wrongly to deleted or
        // protected results.
        let messages =
            match crate::ai::history::build_message_arr(usize::MAX, &app.session_history_file) {
                Ok(messages) => messages,
                Err(error) => {
                    if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
                        eprintln!("[context-prune] failed to validate restored marks: {error}");
                    }
                    return Ok(());
                }
            };
        let active_ids =
            crate::ai::history::compress::llm_prune::active_prunable_tool_ids(&messages);
        let before = prune_marks.len();
        prune_marks.retain(|id, count| *count > 0 && active_ids.contains(id));
        if prune_marks.len() != before {
            if let Err(error) = crate::ai::history::write_llm_prune_marks_sqlite(
                &app.session_history_file,
                &prune_marks,
            ) && crate::ai::driver::runtime_ctx::terminal_output_enabled()
            {
                eprintln!("[context-prune] failed to reconcile restored marks: {error}");
            }
        }
    }
    app.prune_marks = prune_marks;
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
    // Register the live marker first, then load the target state, so prune cannot
    // delete the target session during the switch window.
    let stale_patch_targets = load_stale_patch_targets(store, &session_id, &history_file)?;
    clear_session_local_runtime_state(app);
    app.session_id = session_id.to_string();
    app.session_history_file = history_file;
    app.stale_patch_targets = stale_patch_targets;
    restore_prune_marks_for_history(app)?;
    app.sync_persona_session_binding();
    Ok(())
}

/// A history rewind replaces the current session's messages in place, so the
/// ledger must be rebuilt and persisted in step: keep neither the pre-rewind meta
/// nor a simple clear that would let the next patch bypass the fresh-read gate.
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

/// `/clear`: clears the screen (terminal display) only; no conversation history
/// or session state is touched.
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

    // Clear screen: ANSI escape - clear the whole screen + move the cursor to
    // the top-left
    use std::io::Write;
    print!("\x1b[2J\x1b[H");
    let _ = std::io::stdout().flush();
    true
}

/// Parses the canonical target selector used by export/archive and ensures a
/// plain ID refers to an existing session.
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

/// Maximum allowed days for `/sessions prune <days>` (~10k years). Far below
/// `chrono`'s `Duration::days` and `DateTime` overflow thresholds, so illegal
/// oversized input is stopped before a panic. The valid range is
/// `1..=MAX_PRUNE_DAYS`: `0` would degrade into "delete all non-current
/// sessions" and is rejected explicitly.
pub(crate) const MAX_PRUNE_DAYS: i64 = 3_650_000;

/// Selects sessions inactive for N days: those whose `modified_local` is earlier
/// than `cutoff` count as expired. The current session is never deleted; sessions
/// without a timestamp cannot be age-ordered and are conservatively skipped.
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
    let top_level_mark = cmd == "mark";
    let top_level_unmark = cmd == "unmark";
    if cmd != "sessions"
        && cmd != "session"
        && cmd != "ss"
        && !top_level_suspend
        && !top_level_close
        && !top_level_fork
        && !top_level_mark
        && !top_level_unmark
    {
        return Ok(false);
    }
    let action = if top_level_suspend {
        "suspend"
    } else if top_level_close {
        "close"
    } else if top_level_fork {
        "fork"
    } else if top_level_mark {
        "mark"
    } else if top_level_unmark {
        "unmark"
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
            println!("  /mark                     mark current session as important (shown in red in `/ss`)");
            println!("  /unmark                   remove the important mark from the current session");
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
            // By default do not recursively stat each session's size (the assets
            // recursion is the only heavy work of `/ss`); only explicit `verbose`
            // (`/ss verbose` or `/ss list verbose`) stats them in parallel.
            let verbose =
                action == "verbose" || parts.next().map(|t| t == "verbose").unwrap_or(false);
            let mut sessions = store.list_sessions()?;
            if verbose {
                // Sizes are stat-ed in parallel on demand: list_sessions reads only
                // metadata, and the recursive assets stat runs here across cores.
                store.attach_session_sizes(&mut sessions)?;
            }
            if sessions.is_empty() {
                println!("No sessions.");
            } else {
                // Compute the max ID length for alignment
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
                    let (style_open, style_close) = if s.marked {
                        marked_session_style()
                    } else {
                        ("", "")
                    };
                    println!(
                        "{style_open}{} {:<width$}  {}  {:>8}  {}{style_close}",
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
            // Show the session summary
            // Read only the current session's preview, avoiding a scan and stat
            // of all sessions.
            if let Ok(Some((summary, modified_local))) = store.read_session_preview(&app.session_id)
            {
                if let Some(summary) = &summary {
                    println!("summary: {}", summary);
                }
                let size = store.session_total_size(&app.session_id).unwrap_or(0);
                println!("size: {}", format_size(size));
                let marked = store.read_session_marked(&app.session_id).unwrap_or(false);
                println!("marked: {}", if marked { "yes" } else { "no" });
                if let Some(t) = modified_local {
                    println!("modified: {}", t.format("%Y-%m-%d %H:%M:%S"));
                }
            }
        }
        "mark" => {
            match store.write_session_marked(&app.session_id, true) {
                Ok(()) => println!("Marked session as important: {}", app.session_id),
                Err(err) => eprintln!("[mark] failed to mark session: {err}"),
            }
        }
        "unmark" => {
            match store.write_session_marked(&app.session_id, false) {
                Ok(()) => println!("Removed important mark from session: {}", app.session_id),
                Err(err) => eprintln!("[unmark] failed to unmark session: {err}"),
            }
        }
        "new" | "create" => {
            let new_id = Uuid::new_v4().to_string();
            // Clear the old session's history cache and explicit-enabled tools
            // before switching, so the next turn carries no cross-session dirty
            // state.
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
            // Show the session summary
            // Read only the target session's preview, avoiding a scan of all
            // sessions.
            if let Ok(Some((summary, _))) = store.read_session_preview(id) {
                if let Some(summary) = &summary {
                    println!("summary: {}", summary);
                }
            }
        }
        "suspend" | "bg" | "detach" => {
            // Stay consistent with the Ctrl+C suspend path
            // (`should_suspend_session_on_sigint`): an id explicitly given via
            // `--session` is always suspended; otherwise, if the session has no
            // user messages yet, the main loop's `cleanup_one_shot` deletes that
            // empty session on exit, and writing a suspension entry here would
            // leave a dangling binding to a deleted session — the next `a` launch
            // would try to restore a nonexistent session.
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
                // Save the current model for restoration
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
            // /close: delete the current session and exit the interactive
            // conversation (the opposite of /suspend's "keep and return to shell"
            // — here the session is destroyed outright). Reuse the store (already
            // constructed above).
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
                // Must first terminate subagents still running for this session;
                // otherwise, after the SQLite file is deleted, live Futures may
                // write again and rebuild derived history.
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
            // Optional as=<id> specifies the session id after import
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
            // Clear the associated history cache and explicit-enabled tools, so
            // the next turn hits no stale cache and carries no meaningless tool
            // list.
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
            // /sessions prune <days>: delete sessions inactive for N days (the
            // current session is never deleted).
            let Some(days_str) = parts.next() else {
                println!("missing days. try: /sessions prune <days>");
                return Ok(true);
            };
            let Ok(days) = days_str.parse::<i64>() else {
                println!("invalid days: '{}'", days_str);
                return Ok(true);
            };
            // Lower bound is 1: `prune 0` puts the cutoff at now, i.e. delete all
            // non-current sessions — contrary to the "clean up N days inactive"
            // semantics and a collateral-damage footgun, so it is rejected
            // outright. The upper bound prevents `Duration::days` / `DateTime`
            // overflow panics; ~10k years suffices for any cleanup.
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
                        // Terminate the session's sub-agents after the final
                        // validation, so Futures do not write back after deletion.
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
            // Parse src=<id> / as=<id>; when src is not given, default to the
            // current session.
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
                    // Write a depth-tagged fork marker title for the forked session
                    // (supports fork of fork).
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
            // Usage: /sessions branch <keep_turns> [src=<id>] [as=<id>]
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

/// Parses the fork marker at the start of a title, returning (fork depth, inner
/// title with the marker removed).
/// Returns (0, original title) when there is no marker or the format mismatches.
/// `[fork]` means depth 1, `[fork N]` means depth N.
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

/// Builds a title with a fork depth marker. Depth 1 renders as `[fork]`, N≥2 as
/// `[fork N]`.
/// Total length is capped at 40 chars, so it is not judged a low-quality title
/// and later overwritten by model regeneration.
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

/// Reads the source session's effective title as the base for the fork marker:
/// prefer the persisted title, otherwise generate a fallback title from the
/// first user message.
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

/// Rewrites the dst session's title to the fork-marked version, with depth
/// incremented from the source session's title.
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

/// Formats a file size into a human-readable form (KB/MB/GB).
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
            read_stale_patch_targets_sqlite, write_llm_prune_marks_sqlite,
            write_stale_patch_targets_sqlite,
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
            },
            session_id: session_id.clone(),
            session_history_file: session_store.session_history_file(&session_id),
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::new(),
            current_model: crate::ai::model_names::all()
                .first()
                .map(|model| crate::ai::model_names::model_handle(model))
                .expect("model registry is empty"),
            current_agent: "build".to_string(),
            current_agent_manifest: None,
            pending_files: None,
            forced_skills: vec!["feishu-upload-md".to_string()],
            forced_skill_source: None,
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
            tool_middlewares: Vec::new(),
            llm_middlewares: Vec::new(),
            hooks: Default::default(),
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
        app.prune_marks.insert("source-call".to_string(), 1);

        try_handle_session_command(&mut app, "/sessions new").unwrap();

        assert!(app.last_skill_bias.is_none());
        assert!(app.forced_skills.is_empty());
        assert!(app.forced_question.is_none());
        assert!(app.attached_image_files.is_empty());
        assert!(app.prune_marks.is_empty());
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
        let mut target_messages = vec![Message {
            role: "user".to_string(),
            content: Value::String("target session".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        for index in 0..5 {
            let id = format!("call_{index}");
            target_messages.push(Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![ToolCall {
                    id: id.clone(),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: "execute_command".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tool_call_id: None,
                reasoning_content: None,
            });
            target_messages.push(Message {
                role: "tool".to_string(),
                content: Value::String("target result\n".repeat(1_000)),
                tool_calls: None,
                tool_call_id: Some(id),
                reasoning_content: None,
            });
        }
        append_history_messages(&target_path, &target_messages).unwrap();
        write_llm_prune_marks_sqlite(
            &target_path,
            &[("call_0".to_string(), 2_u8)].into_iter().collect(),
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
        app.prune_marks.insert("source-call".to_string(), 1);

        try_handle_session_command(&mut app, "/sessions use sess-target").unwrap();

        assert_eq!(
            app.stale_patch_targets,
            rustc_hash::FxHashSet::from_iter([target])
        );
        assert_eq!(app.prune_marks.get("call_0"), Some(&2));
        assert!(!app.prune_marks.contains_key("source-call"));

        try_handle_session_command(&mut app, "/sessions new").unwrap();
        assert!(
            app.stale_patch_targets.is_empty(),
            "a brand-new session must not inherit the previous session ledger"
        );
        assert!(app.prune_marks.is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn restore_session_state_restores_only_active_prune_marks() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let mut app = test_app(&root);
        let store = SessionStore::new(app.config.history_file.as_path());
        let session_id = app.session_id.clone();
        let path = store.session_history_file(&session_id);
        let mut messages = Vec::new();
        for index in 0..5 {
            let id = format!("call_{index}");
            messages.push(Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![ToolCall {
                    id: id.clone(),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: "execute_command".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                tool_call_id: None,
                reasoning_content: None,
            });
            messages.push(Message {
                role: "tool".to_string(),
                content: Value::String("result\n".repeat(1_000)),
                tool_calls: None,
                tool_call_id: Some(id),
                reasoning_content: None,
            });
        }
        append_history_messages(&path, &messages).unwrap();
        let persisted = [
            ("call_0".to_string(), 1_u8),
            ("missing".to_string(), 2_u8),
        // The most recent four groups are protected; restored state must not
        // carry the old counters.
            ("call_4".to_string(), 2_u8),
        ]
        .into_iter()
        .collect();
        write_llm_prune_marks_sqlite(&path, &persisted).unwrap();

        restore_session_local_runtime_state(&mut app).unwrap();

        assert_eq!(app.prune_marks.len(), 1);
        assert_eq!(app.prune_marks.get("call_0"), Some(&1));
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
        assert!(app.forced_skills.is_empty());
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
        // Write one user message to make the session non-empty (otherwise the new
        // empty-session protection rejects the suspension, because the main loop
        // deletes empty sessions on exit, leaving a dangling binding).
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
        // No user message written; the session is empty.
        try_handle_session_command(&mut app, "/bg").unwrap();

        // Still request exit, but must not leave a suspension binding (it would
        // dangle after the main loop deletes the empty session).
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

        // Write test messages
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

        // Export to zip
        let archive_path = root.join("export.zip");
        store
            .export_session_archive(&app.session_id, &archive_path)
            .expect("export should succeed");
        assert!(archive_path.exists(), "archive file should exist");

        // Import as a new session
        let dst_id = "imported-session".to_string();
        let result = store.import_session_archive(&archive_path, &dst_id);
        assert!(result.is_ok(), "import should succeed: {:?}", result.err());

        // Verify the imported messages match the originals
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
            marked: false,
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

    /// Rolls the target session file's mtime back N days, to simulate "unread
    /// for N days".
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

        // Three sessions: two old (>30 days), one new (just written).
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

        // Take the same final recheck + locked-delete path as the handler,
        // omitting only the interactive confirmation and PID scan.
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
    fn mark_and_unmark_persist_and_show_in_session_list() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let root = test_history_root();
        let mut app = test_app(&root);
        let store = SessionStore::new(app.config.history_file.as_path());
        let id = app.session_id.clone();

        // A session is not marked by default.
        assert!(!store.read_session_marked(&id).unwrap());

        assert!(try_handle_session_command(&mut app, "/mark").unwrap());
        assert!(store.read_session_marked(&id).unwrap());
        let listed = store.list_sessions().unwrap();
        assert_eq!(
            listed.iter().find(|s| s.id == id).map(|s| s.marked),
            Some(true)
        );

        assert!(try_handle_session_command(&mut app, "/unmark").unwrap());
        assert!(!store.read_session_marked(&id).unwrap());
        assert_eq!(
            store
                .list_sessions()
                .unwrap()
                .iter()
                .find(|s| s.id == id)
                .map(|s| s.marked),
            Some(false)
        );

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

        // Persist two old sessions, to verify out-of-range input rejection
        // deletes nothing.
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

        // `0` (degenerates into deleting all non-current sessions), negatives,
        // exactly past the upper bound, and huge values that would overflow
        // chrono date arithmetic into a panic must all be rejected gracefully.
        // Rolling back `1e8` days crosses `DateTime`'s lower bound — without the
        // upper-bound guard, this line itself would panic.
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

        // Both old sessions still exist: out-of-range input returned before
        // reaching the deletion logic.
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
        // Write a model title to the source session as the base for the fork
        // marker.
        store
            .write_session_title_with_origin(
                &app.session_id,
                "实现fork功能",
                SessionTitleOrigin::Model,
            )
            .unwrap();

        // First fork: the title gains the [fork] marker and we switch to the new
        // session.
        try_handle_session_command(&mut app, "/fork").unwrap();
        let first_fork_id = app.session_id.clone();
        assert_ne!(first_fork_id, "sess-old");
        assert_eq!(
            store.read_session_title(&first_fork_id).unwrap().unwrap(),
            "[fork] 实现fork功能"
        );

        // The original session is kept, not deleted, with its title unchanged.
        assert!(store.session_history_file("sess-old").exists());
        assert_eq!(
            store.read_session_title("sess-old").unwrap().unwrap(),
            "实现fork功能"
        );

        // Fork again on the forked session: the marker depth increments to
        // [fork 2].
        try_handle_session_command(&mut app, "/fork").unwrap();
        assert_ne!(app.session_id, first_fork_id);
        assert_eq!(
            store.read_session_title(&app.session_id).unwrap().unwrap(),
            "[fork 2] 实现fork功能"
        );
        // Both fork branches are kept, not deleted.
        assert!(store.session_history_file(&first_fork_id).exists());
        assert!(store.session_history_file("sess-old").exists());

        let _ = fs::remove_dir_all(root);
    }
}
