//! Background process dispatch: select ready processes, decode task goals,
//! build questions, clone app context, and spawn each as a tokio task.
//!
//! Extracted from `driver/mod.rs` `run_loop` to establish a clear boundary
//! between background dispatch and the foreground interaction loop
//! (review Finding #1, Phase 2).

use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ai::{
    agents::AgentManifest,
    mcp::SharedMcpClient,
    skills::SkillManifest,
    tools::task_tools::{capped_subagent_manifest, with_task_entry_by_pid},
    types::App,
};

use super::agent_routing::{activate_primary_agent, ensure_runtime_manifests_loaded};
use super::process_context::{
    build_background_process_question, finalize_turn_quota, process_history_path,
    resolve_background_subagent_context,
};
use super::runtime_ctx;
use super::scheduler::{
    DispatchOutcomeTag, classify_process_outcome, decode_background_process_task_goal,
    maybe_emit_scheduler_eval, publish_background_task_failure, record_scheduler_outcome,
    resolve_background_subagent_override, select_background_batch,
    should_publish_subagent_task_result,
};
use super::turn_runtime;
use super::{BgSubagentGuard, TASK_PID, terminate_and_cleanup};

const MAX_SUBAGENT_STATUS_DETAILS: usize = 3;

/// Background subagent history is kept only while the process can still be
/// scheduled. Normal termination, failure, panic, or a `task_cancel` that aborts
/// the future are all cleaned up via Drop. Private memory files and the exclusive
/// cwd scratch directory share history's lifetime and are reclaimed here too,
/// preventing buildup over long runs.
struct BackgroundSubagentHistoryGuard {
    path: Option<PathBuf>,
    memory_path: Option<PathBuf>,
    cwd_dir: Option<PathBuf>,
}

impl BackgroundSubagentHistoryGuard {
    fn new(path: PathBuf) -> Self {
        Self {
            path: Some(path),
            memory_path: None,
            cwd_dir: None,
        }
    }

    /// Register the private memory files and exclusive cwd scratch directory that
    /// are spawned with the task and must be reclaimed with history's lifetime.
    /// Paths are built deterministically, so they can be computed once at
    /// construction.
    fn with_scoped_artifacts(
        mut self,
        memory_path: Option<PathBuf>,
        cwd_dir: Option<PathBuf>,
    ) -> Self {
        self.memory_path = memory_path;
        self.cwd_dir = cwd_dir;
        self
    }

    fn preserve(&mut self) {
        self.path = None;
        self.memory_path = None;
        self.cwd_dir = None;
    }

    fn preserve_memory(&mut self) {
        self.memory_path = None;
    }
}

impl Drop for BackgroundSubagentHistoryGuard {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = crate::ai::history::delete_subagent_history(&path);
        }
        if let Some(memory_path) = self.memory_path.take() {
            let _ = crate::ai::history::delete_subagent_memory(&memory_path);
        }
        if let Some(cwd_dir) = self.cwd_dir.take() {
            let _ = std::fs::remove_dir_all(&cwd_dir);
        }
    }
}

/// The only foreground-facing subagent status display. Background tasks stay
/// silent, refreshing one compact status line only at scheduler safe points, so
/// concurrent prose or multi-line ANSI redraws never disturb the terminal.
pub(super) struct SubagentStatusLine {
    last_line: Option<String>,
    is_tty: bool,
}

/// Status bar fields must stay single-line and carry no terminal control
/// characters. The task description comes from model arguments, so it cannot be
/// written to the foreground TTY directly.
fn sanitize_status_field(value: &str) -> String {
    let mut sanitized = String::with_capacity(value.len());
    let mut pending_space = false;
    for ch in value.chars() {
        if ch.is_control() || ch.is_whitespace() {
            pending_space = !sanitized.is_empty();
            continue;
        }
        if pending_space {
            sanitized.push(' ');
            pending_space = false;
        }
        sanitized.push(ch);
    }
    sanitized
}

impl SubagentStatusLine {
    pub(super) fn new() -> Self {
        Self {
            last_line: None,
            is_tty: std::io::stdout().is_terminal(),
        }
    }

    pub(super) fn refresh(&mut self, app: &App) {
        let store = crate::ai::history::SessionStore::new(app.config.history_file.as_path());
        let current_pid = std::process::id() as i32;
        let mut session_ids =
            crate::ai::driver::session_pid::scan_all_session_pids(store.sessions_root())
                .unwrap_or_default()
                .into_iter()
                .filter(|(_, pid, alive)| *alive && *pid == current_pid)
                .map(|(session_id, _, _)| session_id)
                .collect::<Vec<_>>();
        if !session_ids
            .iter()
            .any(|session_id| session_id == &app.session_id)
        {
            session_ids.push(app.session_id.clone());
        }
        session_ids.sort();
        session_ids.dedup();

        let statuses_by_session = {
            let mut os = app.os.lock().unwrap_or_else(|err| err.into_inner());
            session_ids
                .into_iter()
                .map(|session_id| {
                    let statuses = crate::ai::tools::task_tools::subagent_terminal_statuses(
                        os.as_mut(),
                        &session_id,
                    );
                    (session_id, statuses)
                })
                .collect::<Vec<_>>()
        };

        let mut statuses = Vec::new();
        for (session_id, session_statuses) in statuses_by_session {
            let snapshots = session_statuses
                .iter()
                .map(|status| crate::ai::driver::session_pid::AgentSnapshot {
                    agent_name: status.agent_name.clone(),
                    description: status.description.clone(),
                    state: status.state.clone(),
                    elapsed_secs: status.elapsed_secs,
                    progress: status.progress.clone(),
                })
                .collect::<Vec<_>>();
            let _ = crate::ai::driver::session_pid::write_agent_snapshots(
                store.sessions_root(),
                &session_id,
                &snapshots,
            );
            if session_id == app.session_id {
                statuses = session_statuses;
            }
        }

        if statuses.is_empty() {
            return;
        }

        let line = crate::ai::stream::clamp_line_to_terminal_row_with_reserve(
            &render_subagent_status_line(&statuses),
            0,
        );
        if self.last_line.as_deref() == Some(line.as_str()) {
            return;
        }

        if !self.is_tty {
            return;
        }
        print!("\r\x1b[2K{line}");
        let _ = std::io::stdout().flush();
        self.last_line = Some(line);
    }

    /// Before the foreground resumes streaming output or the input box, finalize
    /// and end the dynamic line so it no longer occupies a cursor row.
    pub(super) fn finish(&mut self) {
        let Some(line) = self.last_line.take() else {
            return;
        };
        if self.is_tty {
            print!("\r\x1b[2K{line}\n");
            let _ = std::io::stdout().flush();
        }
    }
}

fn render_subagent_status_line(
    statuses: &[crate::ai::tools::task_tools::SubagentTerminalStatus],
) -> String {
    let total = statuses.len();
    let active = statuses
        .iter()
        .filter(|status| !is_terminal_subagent_state(&status.state))
        .count();
    let mut parts = vec![if active == total {
        format!("subagents · {total} active")
    } else {
        format!("subagents · {active}/{total} active")
    }];

    parts.extend(
        statuses
            .iter()
            .take(MAX_SUBAGENT_STATUS_DETAILS)
            .map(|status| {
                let mut detail = format!(
                    "{} {} {} ({})",
                    sanitize_status_field(&status.description),
                    format_subagent_elapsed(status.elapsed_secs),
                    sanitize_status_field(&status.state),
                    sanitize_status_field(&status.agent_name)
                );
                if let Some(progress) = status.progress.as_deref() {
                    detail.push_str(": ");
                    detail.push_str(&sanitize_status_field(progress));
                }
                detail
            }),
    );

    if total > MAX_SUBAGENT_STATUS_DETAILS {
        parts.push(format!("+{} more", total - MAX_SUBAGENT_STATUS_DETAILS));
    }

    parts.join(" · ")
}

fn is_terminal_subagent_state(state: &str) -> bool {
    matches!(
        state,
        "completed" | "failed" | "cancelled" | "canceled" | "timeout" | "terminated"
    )
}

fn format_subagent_elapsed(elapsed_secs: u64) -> String {
    if elapsed_secs < 60 {
        return format!("{elapsed_secs}s");
    }
    if elapsed_secs < 60 * 60 {
        return format!("{}m{}s", elapsed_secs / 60, elapsed_secs % 60);
    }
    format!("{}h{}m", elapsed_secs / 3600, (elapsed_secs % 3600) / 60)
}

impl Drop for SubagentStatusLine {
    fn drop(&mut self) {
        // Every early return and error path must first end the unterminated
        // dynamic status line, so the shell prompt or later error text does not
        // append to it.
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BackgroundSubagentHistoryGuard, format_subagent_elapsed, render_subagent_status_line,
        sanitize_status_field,
    };
    use crate::ai::tools::task_tools::SubagentTerminalStatus;

    #[test]
    fn subagent_status_field_is_safe_single_line_text() {
        assert_eq!(
            sanitize_status_field("review\ncode\r\x1b[31m now\tplease"),
            "review code [31m now please"
        );
    }

    #[test]
    fn subagent_status_line_shows_counts_elapsed_and_truncates_details() {
        let statuses = vec![
            SubagentTerminalStatus {
                description: "review stream".to_string(),
                agent_name: "explorer".to_string(),
                state: "running".to_string(),
                elapsed_secs: 38,
                progress: Some("using read_file · src/bin/ai/driver.rs".to_string()),
            },
            SubagentTerminalStatus {
                description: "review driver".to_string(),
                agent_name: "explorer".to_string(),
                state: "waiting".to_string(),
                elapsed_secs: 61,
                progress: None,
            },
            SubagentTerminalStatus {
                description: "compile check".to_string(),
                agent_name: "executor".to_string(),
                state: "completed".to_string(),
                elapsed_secs: 3661,
                progress: None,
            },
            SubagentTerminalStatus {
                description: "extra task".to_string(),
                agent_name: "executor".to_string(),
                state: "running".to_string(),
                elapsed_secs: 5,
                progress: None,
            },
        ];

        assert_eq!(
            render_subagent_status_line(&statuses),
            "subagents · 3/4 active · review stream 38s running (explorer): using read_file · src/bin/ai/driver.rs · review driver 1m1s waiting (explorer) · compile check 1h1m completed (executor) · +1 more"
        );
        assert_eq!(format_subagent_elapsed(59), "59s");
        assert_eq!(format_subagent_elapsed(60), "1m0s");
    }

    #[test]
    fn background_subagent_history_guard_cleans_or_preserves_history() {
        let root = std::env::temp_dir().join(format!(
            "background-subagent-history-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();

        let cleaned = root.join("session.proc-1.sqlite");
        std::fs::write(&cleaned, b"db").unwrap();
        std::fs::write(format!("{}-wal", cleaned.display()), b"wal").unwrap();
        drop(BackgroundSubagentHistoryGuard::new(cleaned.clone()));
        assert!(!cleaned.exists());
        assert!(!std::path::Path::new(&format!("{}-wal", cleaned.display())).exists());

        let preserved = root.join("session.proc-2.sqlite");
        std::fs::write(&preserved, b"db").unwrap();
        let mut guard = BackgroundSubagentHistoryGuard::new(preserved.clone());
        guard.preserve();
        drop(guard);
        assert!(preserved.exists());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn background_subagent_guard_cleans_scoped_memory_and_cwd_on_drop() {
        let root = std::env::temp_dir().join(format!(
            "background-subagent-scoped-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();

        let history = root.join("session.subagent-task_x.sqlite");
        std::fs::write(&history, b"db").unwrap();
        // Private memory: the jsonl body + derived .db and its WAL sidecar.
        let memory = root.join("agent_memory.subagent-task_x.jsonl");
        std::fs::write(&memory, b"[]").unwrap();
        let memory_db = root.join("agent_memory.subagent-task_x.db");
        std::fs::write(&memory_db, b"db").unwrap();
        std::fs::write(format!("{}-wal", memory_db.display()), b"wal").unwrap();
        // Exclusive cwd scratch directory (including contents; needs recursive removal).
        let cwd = root.join("subagent-cwd-task_x");
        std::fs::create_dir_all(cwd.join("nested")).unwrap();
        std::fs::write(cwd.join("scratch.txt"), b"tmp").unwrap();
        // Subagent-scoped assets (plan_state / side_note / working-checkpoint) live in
        // `<stem>.assets` next to the child history; guard Drop must reclaim them too.
        let assets = root.join("session.subagent-task_x.assets");
        std::fs::create_dir_all(assets.join("side_notes")).unwrap();
        std::fs::write(assets.join("plan-state.json"), b"{}").unwrap();

        drop(
            BackgroundSubagentHistoryGuard::new(history.clone())
                .with_scoped_artifacts(Some(memory.clone()), Some(cwd.clone())),
        );

        assert!(!history.exists(), "history db must be cleaned");
        assert!(!memory.exists(), "memory jsonl must be cleaned");
        assert!(!memory_db.exists(), "derived memory .db must be cleaned");
        assert!(
            !std::path::Path::new(&format!("{}-wal", memory_db.display())).exists(),
            "memory .db WAL sidecar must be cleaned"
        );
        assert!(!cwd.exists(), "cwd scratch dir must be recursively cleaned");
        assert!(!assets.exists(), "subagent assets dir must be cleaned");

        // preserve() must also keep memory/cwd/assets (normal-end / resume scenarios:
        // the resumed process still reads its plan-state / side-notes / checkpoint).
        let history2 = root.join("session.subagent-task_y.sqlite");
        let memory2 = root.join("agent_memory.subagent-task_y.jsonl");
        let cwd2 = root.join("subagent-cwd-task_y");
        let assets2 = root.join("session.subagent-task_y.assets");
        std::fs::write(&history2, b"db").unwrap();
        std::fs::write(&memory2, b"[]").unwrap();
        std::fs::create_dir_all(&cwd2).unwrap();
        std::fs::create_dir_all(&assets2).unwrap();
        std::fs::write(assets2.join("plan-state.json"), b"{}").unwrap();
        let mut guard = BackgroundSubagentHistoryGuard::new(history2.clone())
            .with_scoped_artifacts(Some(memory2.clone()), Some(cwd2.clone()));
        guard.preserve();
        drop(guard);
        assert!(history2.exists() && memory2.exists() && cwd2.exists());
        assert!(assets2.exists(), "preserve() must keep subagent assets dir");

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn background_subagent_memory_is_merged_before_cleanup() {
        let root = std::env::temp_dir().join(format!(
            "background-subagent-merge-order-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let history = root.join("session.subagent-task_merge.sqlite");
        let private_memory = root.join("agent_memory.subagent-task_merge.jsonl");
        let main_memory = root.join("agent_memory.jsonl");
        std::fs::write(&history, b"db").unwrap();
        std::fs::write(
            &private_memory,
            serde_json::json!({
                "id": "mem-merge",
                "timestamp": "2026-07-31T00:00:00Z",
                "category": "project_memory",
                "note": "durable subagent conclusion",
                "tags": [],
                "source": "test",
                "priority": 180,
                "owner_pid": 42,
                "owner_pgid": 42,
                "image_path": null
            })
            .to_string()
                + "\n",
        )
        .unwrap();
        let guard = BackgroundSubagentHistoryGuard::new(history.clone())
            .with_scoped_artifacts(Some(private_memory.clone()), None);

        assert_eq!(
            crate::ai::tools::service::memory::merge_subagent_whitelist(
                &private_memory,
                &main_memory
            )
            .unwrap(),
            1
        );
        drop(guard);

        assert!(!private_memory.exists());
        assert!(
            std::fs::read_to_string(&main_memory)
                .unwrap()
                .contains("durable subagent conclusion")
        );
        let _ = std::fs::remove_dir_all(root);
    }
}

/// Dispatch a batch of background processes: select ready processes, decode
/// task goals, build questions, clone app context, and spawn each as a
/// tokio task with proper scope setup.
pub(super) fn dispatch_background_batch(
    app: &mut App,
    mcp_client: &SharedMcpClient,
    skill_manifests: &mut Arc<Vec<SkillManifest>>,
    agent_manifests: &mut Arc<Vec<AgentManifest>>,
    manifests_loaded: &mut bool,
    epoch: u64,
) {
    let background_procs: Vec<aios_kernel::kernel::Process> = {
        let mut os = app.os.lock().unwrap();
        select_background_batch(os.as_mut(), epoch, app.session_id.as_str())
    };
    maybe_emit_scheduler_eval(epoch, app.session_id.as_str());

    if background_procs.is_empty() {
        return;
    }

    ensure_runtime_manifests_loaded(app, skill_manifests, agent_manifests, manifests_loaded);

    let original_history_file = app.session_history_file.clone();

    let mut task_specs: Vec<(
        u64,
        String,
        PathBuf,
        Option<String>,
        Option<String>,
        Option<u64>,
        Option<aios_kernel::primitives::FutexAddr>,
        Option<String>,
        Option<serde_json::Value>,
        Option<crate::ai::models::AutoModelFallbackSpec>,
        usize,
        bool,
        bool,
    )> = Vec::new();
    for proc in &background_procs {
        let pid = proc.pid;
        let task_goal = match decode_background_process_task_goal(&proc.goal) {
            Ok(goal) => goal,
            Err(err) => {
                let (result_channel_id, completion_futex_addr) =
                    with_task_entry_by_pid(pid, |entry| {
                        (
                            Some(entry.result_channel_id),
                            Some(entry.completion_futex_addr),
                        )
                    })
                    .unwrap_or((None, None));
                let mut os = app.os.lock().unwrap();
                publish_background_task_failure(
                    os.as_mut(),
                    pid,
                    result_channel_id,
                    completion_futex_addr,
                    &format!("Corrupted subagent task goal for pid {}: {}", pid, err),
                );
                continue;
            }
        };
        let mailbox_messages: Vec<String> = proc.mailbox.iter().cloned().collect();
        // When the mailbox is non-empty, build_background_process_question goes
        // through format_wakeup_prompt, producing a system scheduling notice
        // (not user input), so it should be persisted marked as an internal_note.
        let is_resume_wakeup = !mailbox_messages.is_empty();
        if !mailbox_messages.is_empty() {
            let mut os = app.os.lock().unwrap();
            if let Some(actual) = os.get_process_mut(pid) {
                actual.mailbox.clear();
            }
        }
        let proc_question = build_background_process_question(
            pid,
            &proc.goal,
            task_goal.as_ref().map(|goal| goal.prompt.as_str()),
            &mailbox_messages,
        );

        let initialize_history = {
            let mut os = app.os.lock().unwrap();
            os.set_current_pid(Some(pid));
            let mut initialize_history = false;
            if let Some(p) = os.get_process_mut(pid) {
                initialize_history = p.history_file.is_none();
                if initialize_history {
                    p.history_file = Some(process_history_path(&original_history_file, pid));
                }
                let _ = os.process_pending_signals();
            }
            initialize_history
        };

        let history_path = process_history_path(&original_history_file, pid);
        task_specs.push((
            pid,
            proc_question,
            history_path,
            task_goal.as_ref().map(|goal| goal.agent_name.clone()),
            task_goal.as_ref().map(|goal| goal.model.clone()),
            task_goal.as_ref().map(|goal| goal.result_channel_id),
            task_goal
                .as_ref()
                .map(|goal| aios_kernel::primitives::FutexAddr(goal.completion_futex_addr)),
            task_goal.as_ref().map(|goal| goal.task_id.clone()),
            task_goal
                .as_ref()
                .and_then(|goal| goal.response_schema.clone()),
            task_goal.as_ref().and_then(|goal| goal.auto_model_fallback),
            task_goal.as_ref().map(|goal| goal.spawn_depth).unwrap_or(0),
            is_resume_wakeup,
            initialize_history,
        ));
    }

    for (
        pid,
        proc_question,
        history_path,
        agent_override,
        model_override,
        result_channel_id,
        completion_futex_addr,
        task_id,
        response_schema,
        auto_model_fallback,
        spawn_depth,
        is_resume_wakeup,
        initialize_history,
    ) in task_specs
    {
        let mut task_app = app.fork_for_subagent();
        // Background tasks need their own streaming/cancel_stream flags:
        // `App::clone` by default shares the same Arc set with the parent App, so
        // concurrent background run_turns would overwrite each other's streaming
        // and cancel each other (clear_stream_cancel resets the shared
        // cancel_stream). A background task's cancel_stream must be shared with
        // its registry entry: the sync command polls it and kills the actual OS
        // process group on timeout/cancel, and Tokio abort alone cannot interrupt
        // an in-flight sync poll. shutdown still stays shared with the parent
        // App: a session-level exit must propagate to background tasks so they
        // can wind down gracefully.
        task_app.streaming = Arc::new(AtomicBool::new(false));
        task_app.cancel_stream = task_id
            .as_deref()
            .and_then(|task_id| {
                crate::ai::tools::task_tools::with_task_entry(task_id, |entry| {
                    entry.cancel_stream.clone()
                })
            })
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        crate::ai::types::clear_stream_cancel(&task_app);
        let task_mcp = mcp_client.clone();
        let task_os = app.os.clone();
        let task_agent = match resolve_background_subagent_override(
            agent_manifests.as_slice(),
            agent_override.as_deref(),
        ) {
            Ok(agent) => agent,
            Err(err) => {
                let mut os = app.os.lock().unwrap();
                publish_background_task_failure(
                    os.as_mut(),
                    pid,
                    result_channel_id,
                    completion_futex_addr,
                    &err,
                );
                continue;
            }
        };
        if let Some(agent) = task_agent {
            let capped_agent = capped_subagent_manifest(agent);
            activate_primary_agent(&mut task_app, &capped_agent);
        }
        let next_model = model_override.unwrap_or_else(|| app.current_model.clone());

        let inherit = task_id
            .as_deref()
            .and_then(|tid| crate::ai::tools::task_tools::with_task_entry(tid, |e| e.inherit))
            .unwrap_or_default();
        let (effective_history, task_skills) = match resolve_background_subagent_context(
            history_path,
            original_history_file.as_path(),
            skill_manifests,
            task_id.as_deref(),
            inherit,
            initialize_history,
        ) {
            Ok(context) => context,
            Err(err) => {
                let mut os = app.os.lock().unwrap();
                publish_background_task_failure(
                    os.as_mut(),
                    pid,
                    result_channel_id,
                    completion_futex_addr,
                    &err,
                );
                continue;
            }
        };
        task_app.session_history_file = effective_history;
        let _ =
            crate::ai::driver::commands::session::restore_prune_marks_for_history(&mut task_app);
        let task_driver_ctx = runtime_ctx::DriverContext::from_app_snapshot(
            &task_app,
            task_mcp.clone(),
            task_skills.clone(),
            agent_manifests.clone(),
        );
        let scope_task_id = task_id.clone().unwrap_or_else(|| format!("pid-{pid}"));
        let phase_slot = crate::ai::tools::task_tools::task_progress_slot(&scope_task_id)
            .unwrap_or_else(runtime_ctx::new_subagent_progress_slot);
        let phase_slot_for_payload = phase_slot.clone();
        let parent_history_for_scopes = original_history_file.clone();

        // Private memory files and the exclusive cwd scratch directory are
        // spawned with the task and reclaimed with history's lifetime (preserve
        // keeps them on normal end; Drop cleans them up on error/abort/panic).
        // Both paths must reuse the same logic used at creation time, so a second
        // divergent definition cannot cause missed or wrong deletions.
        let scoped_memory_path = (!inherit.memory).then(|| {
            runtime_ctx::make_subagent_memory_path(&parent_history_for_scopes, &scope_task_id)
        });
        let scratch_base_for_cwd = parent_history_for_scopes
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        let scoped_cwd_dir = if inherit.cwd {
            None
        } else {
            runtime_ctx::make_subagent_cwd(&scratch_base_for_cwd, &scope_task_id)
        };
        let history_guard =
            BackgroundSubagentHistoryGuard::new(task_app.session_history_file.clone())
                .with_scoped_artifacts(scoped_memory_path.clone(), scoped_cwd_dir.clone());
        let preserve_scoped_artifacts = Arc::new(AtomicBool::new(false));
        let preserve_scoped_artifacts_for_inner = preserve_scoped_artifacts.clone();
        let memory_merge_failed = Arc::new(AtomicBool::new(false));
        let memory_merge_failed_for_inner = memory_merge_failed.clone();
        let private_memory_for_merge = scoped_memory_path.clone();
        let persona_memory_path = app.current_persona_memory_file();
        let main_memory_for_merge = persona_memory_path.clone();

        // Slot used by the sub-agent's `finalize_turn` to publish
        // its final assistant text. Cloned into the result-channel
        // payload below so `task_wait` can surface what the
        // sub-agent actually produced (instead of just "completed
        // with empty output").
        let result_slot_for_payload: runtime_ctx::SubagentResultSlot = std::sync::Arc::new(
            tokio::sync::Mutex::new(runtime_ctx::SubagentResult::default()),
        );
        let result_slot_for_scope = result_slot_for_payload.clone();

        let inner_fut = TASK_PID.scope(Some(pid), async move {
            crate::ai::tools::registry::common::clear_tool_cancel();
            let run = runtime_ctx::IS_RESUME_TURN.scope(
                is_resume_wakeup,
                turn_runtime::run_turn(
                    &mut task_app,
                    &task_mcp,
                    &task_skills,
                    usize::MAX,
                    proc_question,
                    String::new(),
                    next_model,
                    None,
                    false,
                    false,
                ),
            );
            let result = if let Some(spec) = auto_model_fallback {
                runtime_ctx::AUTO_MODEL_FALLBACK.scope(spec, run).await
            } else {
                run.await
            }
            .map_err(|e| format!("{}", e));
            let captured_result = if result_channel_id.is_some() {
                result_slot_for_payload.lock().await.clone()
            } else {
                runtime_ctx::SubagentResult::default()
            };
            let captured_output = captured_result.parent_payload;
            let response_for_validation = captured_result.final_assistant_text;
            let progress = runtime_ctx::subagent_progress_snapshot(&phase_slot_for_payload)
                .unwrap_or_else(|| "working".to_string());
            let memory_merge_error = private_memory_for_merge.and_then(|private_memory| {
                crate::ai::tools::service::memory::merge_subagent_whitelist(
                    &private_memory,
                    &main_memory_for_merge,
                )
                .err()
            });
            if memory_merge_error.is_some() {
                memory_merge_failed_for_inner.store(true, Ordering::Release);
            }
            let response_validation_error = if result.is_ok() && memory_merge_error.is_none() {
                crate::ai::tools::task_tools::validate_subagent_response(
                    response_schema.as_ref(),
                    &response_for_validation,
                )
                .err()
            } else {
                None
            };
            let mut os = task_os.lock().unwrap();
            os.set_current_pid(Some(pid));
            let task_succeeded = result.is_ok()
                && memory_merge_error.is_none()
                && response_validation_error.is_none();
            let publish_task_result = should_publish_subagent_task_result(
                task_succeeded,
                &captured_output,
                os.get_process(pid).map(|proc| &proc.state),
            );
            if publish_task_result && let Some(result_channel_id) = result_channel_id {
                let payload = serde_json::json!({
                    "status": if task_succeeded { "completed" } else { "failed" },
                    "output": captured_output,
                    "error": result
                        .as_ref()
                        .err()
                        .cloned()
                        .or(memory_merge_error)
                        .or(response_validation_error),
                    "progress": progress,
                })
                .to_string();
                let _ = os.channel_send(
                    Some(pid),
                    aios_kernel::primitives::ChannelId(result_channel_id),
                    payload,
                );
                let _ = os.channel_close(
                    Some(pid),
                    aios_kernel::primitives::ChannelId(result_channel_id),
                );
                let _ = os.channel_release_named(
                    aios_kernel::primitives::ChannelId(result_channel_id),
                    "task_result.producer",
                );
            }
            if publish_task_result && let Some(addr) = completion_futex_addr {
                let _ = os.futex_store(addr, 1);
                super::notify_scheduler();
            }
            let preserve_history = match result {
                Ok(_outcome) => {
                    let outcome = classify_process_outcome(&**os, pid);
                    record_scheduler_outcome(os.as_mut(), pid, outcome);
                    os.increment_turns_used_for(pid);
                    let (should_terminate, termination_result) =
                        finalize_turn_quota(os.as_mut(), pid);
                    if should_terminate {
                        terminate_and_cleanup(os.as_mut(), pid, termination_result, true);
                        false
                    } else if os.is_round_robin() {
                        os.set_current_pid(Some(pid));
                        os.requeue_current();
                        true
                    } else {
                        true
                    }
                }
                Err(err) => {
                    record_scheduler_outcome(os.as_mut(), pid, DispatchOutcomeTag::Failed);
                    terminate_and_cleanup(os.as_mut(), pid, format!("Failed: {}", err), true);
                    false
                }
            };
            drop(os);
            preserve_scoped_artifacts_for_inner.store(preserve_history, Ordering::Release);
        });

        type BoxedTaskFuture = std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>;
        let mut wrapped: BoxedTaskFuture = Box::pin(inner_fut);
        wrapped =
            Box::pin(runtime_ctx::PERSONA_MEMORY_PATH.scope(persona_memory_path.clone(), wrapped));
        wrapped = Box::pin(runtime_ctx::SUBAGENT_RESULT_SLOT.scope(result_slot_for_scope, wrapped));
        wrapped = Box::pin(runtime_ctx::SUBAGENT_PHASE.scope(phase_slot, wrapped));
        wrapped = Box::pin(runtime_ctx::SUBAGENT_TASK_ID.scope(scope_task_id.clone(), wrapped));
        if let Some(mem_path) = scoped_memory_path {
            // Subagents have private memory by default: after finalize,
            // whitelisted entries (is_permanent_memory) are merged back into the
            // main memory file so long-term assets can be shared across tasks,
            // while ordinary task_events stay in the private file and never
            // pollute main memory.
            wrapped = Box::pin(runtime_ctx::SUBAGENT_MEMORY_PATH.scope(mem_path, wrapped));
        }
        if let Some(scratch) = scoped_cwd_dir {
            wrapped = Box::pin(runtime_ctx::SUBAGENT_CWD.scope(scratch, wrapped));
        }
        // Set the subagent nesting depth, used by `task_spawn` / `task` to
        // detect recursive fan-out inside subagents.
        wrapped = Box::pin(runtime_ctx::SUBAGENT_DEPTH.scope(spawn_depth, wrapped));
        // Background tasks only hand final results to task_wait/task_status
        // aggregation; subagents are forbidden from contending for the foreground
        // terminal directly, avoiding concurrent streaming output and ANSI cursor
        // control corrupting each other.
        wrapped = Box::pin(runtime_ctx::SUPPRESS_TERMINAL_OUTPUT.scope(true, wrapped));

        // Count the in-flight background subagent: the guard is moved into the
        // task with the spawned future, and Drop auto-decrements on task end
        // (normal/error/panic), so the input box is never gated permanently.
        let inflight_guard = BgSubagentGuard::new();
        let guarded_fut = async move {
            let _inflight_guard = inflight_guard;
            let mut history_guard = history_guard;
            wrapped.await;
            if memory_merge_failed.load(Ordering::Acquire) {
                history_guard.preserve_memory();
            }
            if preserve_scoped_artifacts.load(Ordering::Acquire) {
                history_guard.preserve();
            }
        };
        let handle = tokio::spawn(runtime_ctx::DRIVER_CTX.scope(task_driver_ctx, guarded_fut));
        crate::ai::tools::task_tools::set_task_abort_handle(&scope_task_id, handle.abort_handle());
    }
}
