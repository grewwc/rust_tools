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
    tools::task_tools::{self, capped_subagent_manifest, with_task_entry_by_pid},
    types::App,
};

use super::agent_routing::{activate_primary_agent, ensure_runtime_manifests_loaded};
use super::process_context::{
    build_background_process_question, finalize_turn_quota_outcome, process_history_path,
    resolve_background_subagent_context,
};
use super::runtime_ctx;
use super::scheduler::{
    DispatchOutcomeTag, SchedulerDecision, classify_process_outcome,
    decode_background_process_task_goal, maybe_emit_scheduler_eval,
    publish_background_task_failure, record_scheduler_outcome,
    resolve_background_subagent_override, select_background_batch,
    should_publish_subagent_task_result,
};
use super::subagent_launch::{ExternalSubagentLaunch, resolve_external_subagent_cwd};
use super::turn_runtime;
use super::{BgSubagentGuard, TASK_PID, terminate_and_cleanup};

struct BackgroundTurnTarget {
    pid: u64,
    task_id: Option<String>,
    result_channel_id: Option<u64>,
    completion_futex_addr: Option<aios_kernel::primitives::FutexAddr>,
    backend_id: task_tools::BackendId,
}

enum BackgroundTaskState {
    Completed,
    Failed(String),
    Blocked(String),
    /// Empty output can be a cooperative suspension rather than a final response.
    Incomplete(Option<String>),
}

struct BackgroundTaskOutcome {
    state: BackgroundTaskState,
    output: String,
    progress: String,
    backend_session: Option<String>,
    retain_recovery: bool,
}

impl BackgroundTaskOutcome {
    fn status(&self) -> &'static str {
        match self.state {
            BackgroundTaskState::Completed => "completed",
            BackgroundTaskState::Failed(_) => "failed",
            BackgroundTaskState::Blocked(_) => "blocked",
            BackgroundTaskState::Incomplete(_) => "running",
        }
    }

    fn error(&self) -> Option<&str> {
        match &self.state {
            BackgroundTaskState::Failed(error) | BackgroundTaskState::Blocked(error) => Some(error),
            BackgroundTaskState::Completed | BackgroundTaskState::Incomplete(_) => None,
        }
    }
}

fn resolve_background_task_outcome(
    outcome: task_tools::Outcome,
    memory_merge_error: Option<String>,
    response_schema: Option<&serde_json::Value>,
) -> BackgroundTaskOutcome {
    let retain_recovery = memory_merge_error.is_some();
    let mut output = outcome.parent_payload;
    let state = if let Some(error) = outcome.execution_error.or(memory_merge_error) {
        BackgroundTaskState::Failed(error)
    } else {
        match task_tools::parse_subagent_handoff(&outcome.validation_body) {
            Err(error) => BackgroundTaskState::Failed(error),
            Ok(Some(handoff)) => {
                if let Some(action) = handoff.suggested_action {
                    output = action;
                }
                BackgroundTaskState::Blocked(format!(
                    "{}: {}",
                    handoff.kind.as_str(),
                    handoff.reason,
                ))
            }
            Ok(None) => {
                let validation_error = task_tools::validate_subagent_response(
                    response_schema,
                    &outcome.validation_body,
                )
                .err();
                if output.trim().is_empty() && outcome.validation_body.trim().is_empty() {
                    BackgroundTaskState::Incomplete(validation_error)
                } else if let Some(error) = validation_error {
                    BackgroundTaskState::Failed(error)
                } else {
                    BackgroundTaskState::Completed
                }
            }
        }
    };
    BackgroundTaskOutcome {
        state,
        output,
        progress: outcome.progress,
        backend_session: outcome.backend_session,
        retain_recovery,
    }
}

struct BackgroundTurnRetention {
    preserve_history: bool,
    delivery_failed: bool,
}

async fn supervise_background_worker(
    worker: impl std::future::Future<Output = task_tools::Outcome>,
    events: &task_tools::Events,
    default_progress: &str,
) -> (task_tools::Outcome, bool) {
    use futures_util::FutureExt;

    match std::panic::AssertUnwindSafe(worker).catch_unwind().await {
        Ok(outcome) => (outcome, false),
        Err(panic) => {
            let detail = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic payload");
            (
                task_tools::Outcome::failed(
                    format!("Background worker panicked: {detail}"),
                    events,
                    default_progress,
                ),
                true,
            )
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BackgroundFinalizationOrigin {
    Turn,
    Preparation,
}

struct BackgroundTerminalPlan {
    outcome: BackgroundTaskOutcome,
    origin: BackgroundFinalizationOrigin,
    cleanup_history: Option<PathBuf>,
    publication_guard: task_tools::TaskPublicationGuard,
}

enum BackgroundFinalization {
    AlreadyTerminal,
    Suspended,
    Terminal(BackgroundTerminalPlan),
}

/// Reserve terminal ownership and account usage using only in-memory kernel operations.
/// A publication guard keeps collectors pending until lock-free persistence and IPC complete.
fn reserve_background_finalization(
    os: &mut (dyn aios_kernel::kernel::Kernel + Send),
    target: &BackgroundTurnTarget,
    mut outcome: BackgroundTaskOutcome,
    origin: BackgroundFinalizationOrigin,
) -> BackgroundFinalization {
    let pid = target.pid;
    let Some(process) = os.get_process(pid) else {
        return BackgroundFinalization::AlreadyTerminal;
    };
    if process.state == aios_kernel::kernel::ProcessState::Terminated {
        return BackgroundFinalization::AlreadyTerminal;
    }
    let suspended = !should_publish_subagent_task_result(true, "", Some(&process.state));
    if !suspended && let BackgroundTaskState::Incomplete(validation_error) = &mut outcome.state {
        outcome.state = validation_error
            .take()
            .map(BackgroundTaskState::Failed)
            .unwrap_or(BackgroundTaskState::Completed);
    }
    let publication_guard = if matches!(outcome.state, BackgroundTaskState::Incomplete(_)) {
        None
    } else if let Some(guard) = task_tools::TaskPublicationGuard::reserve(os, pid) {
        Some(guard)
    } else {
        return BackgroundFinalization::AlreadyTerminal;
    };
    os.set_current_pid(Some(pid));
    let quota = (origin == BackgroundFinalizationOrigin::Turn)
        .then(|| finalize_turn_quota_outcome(os, pid));
    if matches!(outcome.state, BackgroundTaskState::Completed)
        && let Some(quota) = &quota
        && matches!(
            quota.verdict,
            aios_kernel::primitives::RlimitVerdict::Exceeded { .. }
        )
        && (quota.should_terminate || target.result_channel_id.is_some())
    {
        outcome.state = BackgroundTaskState::Failed(super::format_rlimit_termination_result(
            quota.verdict.clone(),
        ));
    }
    let (dispatch_outcome, termination_result) = match &outcome.state {
        BackgroundTaskState::Failed(error) => {
            (DispatchOutcomeTag::Failed, Some(format!("Failed: {error}")))
        }
        BackgroundTaskState::Blocked(error) => (
            DispatchOutcomeTag::Blocked,
            Some(format!("Blocked: {error}")),
        ),
        BackgroundTaskState::Completed => (
            DispatchOutcomeTag::Advanced,
            quota.as_ref().and_then(|quota| {
                (quota.should_terminate || target.result_channel_id.is_some())
                    .then(|| quota.termination_result.clone())
            }),
        ),
        BackgroundTaskState::Incomplete(_) => (classify_process_outcome(os, pid), None),
    };
    record_scheduler_outcome(os, pid, dispatch_outcome);
    let Some(termination_result) = termination_result else {
        if os.is_round_robin() {
            os.set_current_pid(Some(pid));
            os.requeue_current();
        }
        return BackgroundFinalization::Suspended;
    };
    let history = os
        .get_process_mut(pid)
        .and_then(|process| process.history_file.take());
    let cleanup_history = (origin == BackgroundFinalizationOrigin::Preparation)
        .then_some(history)
        .flatten();
    // Detachment guarantees the legacy kernel cleanup has no external file to delete.
    // Converting the process first also lets cancellation recognize that this result already owns it.
    terminate_and_cleanup(os, pid, termination_result, true);
    BackgroundFinalization::Terminal(BackgroundTerminalPlan {
        outcome,
        origin,
        cleanup_history,
        publication_guard: publication_guard.expect("a terminal outcome reserves publication"),
    })
}

/// Serialize and durably persist before taking the kernel lock for IPC.
fn prepare_background_publication(
    target: &BackgroundTurnTarget,
    plan: &BackgroundTerminalPlan,
) -> Option<Result<task_tools::PreparedTaskPublication, String>> {
    let channel = target.result_channel_id?;
    let outcome = &plan.outcome;
    let payload = match plan.origin {
        BackgroundFinalizationOrigin::Preparation => serde_json::json!({
            "status": outcome.status(),
            "output": outcome.output,
            "error": outcome.error(),
        }),
        BackgroundFinalizationOrigin::Turn => {
            let backend_session = match target.task_id.as_deref() {
                Some(task_id) => task_tools::backend_session::record_session(
                    target.backend_id,
                    task_id,
                    outcome.backend_session.clone(),
                ),
                None => outcome.backend_session.clone(),
            };
            serde_json::json!({
                "status": outcome.status(),
                "output": outcome.output,
                "error": outcome.error(),
                "progress": outcome.progress,
                "backend_session": backend_session,
            })
        }
    }
    .to_string();
    Some(task_tools::PreparedTaskPublication::prepare(
        target.task_id.as_deref(),
        target.pid,
        channel,
        payload,
    ))
}

fn commit_background_publication(
    os: &mut dyn aios_kernel::kernel::Kernel,
    target: &BackgroundTurnTarget,
    publication: &mut Option<Result<task_tools::PreparedTaskPublication, String>>,
) {
    if let Some(Ok(publication)) = publication {
        publication.commit_ipc(os);
    }
    if let Some(addr) = target.completion_futex_addr {
        let _ = os.futex_store(addr, 1);
        super::notify_scheduler();
    }
}

/// Warning persistence and external history deletion both run without the kernel guard.
fn finish_background_publication(
    plan: BackgroundTerminalPlan,
    publication: Option<Result<task_tools::PreparedTaskPublication, String>>,
    before_cleanup: impl FnOnce(),
) -> BackgroundTurnRetention {
    let delivery =
        publication.map(|publication| publication.and_then(|publication| publication.finish()));
    let delivery_failed = matches!(delivery, Some(Err(_)));
    if let Some(Err(error)) = delivery {
        runtime_ctx::publish_subagent_phase("result delivery failed; recovery retained");
        eprintln!("[Task result delivery] {error}");
    }
    let preserve_history = plan.outcome.retain_recovery || delivery_failed;
    if !preserve_history && let Some(history_path) = &plan.cleanup_history {
        before_cleanup();
        let _ = std::fs::remove_file(history_path);
    }
    drop(plan.publication_guard);
    BackgroundTurnRetention {
        preserve_history,
        delivery_failed,
    }
}

fn finalize_background_turn_shared(
    shared_os: &aios_kernel::SharedKernel,
    target: &BackgroundTurnTarget,
    outcome: BackgroundTaskOutcome,
) -> BackgroundTurnRetention {
    finalize_background_turn_with_io(shared_os, target, outcome, || {}, || {})
}

fn finalize_background_turn_with_io(
    shared_os: &aios_kernel::SharedKernel,
    target: &BackgroundTurnTarget,
    outcome: BackgroundTaskOutcome,
    before_publication: impl FnOnce(),
    before_cleanup: impl FnOnce(),
) -> BackgroundTurnRetention {
    finalize_background_outcome_with_io(
        shared_os,
        target,
        outcome,
        BackgroundFinalizationOrigin::Turn,
        before_publication,
        before_cleanup,
    )
}

fn finalize_background_outcome_with_io(
    shared_os: &aios_kernel::SharedKernel,
    target: &BackgroundTurnTarget,
    outcome: BackgroundTaskOutcome,
    origin: BackgroundFinalizationOrigin,
    before_publication: impl FnOnce(),
    before_cleanup: impl FnOnce(),
) -> BackgroundTurnRetention {
    let finalization = {
        let mut os = shared_os.lock().unwrap();
        reserve_background_finalization(os.as_mut(), target, outcome, origin)
    };
    let plan = match finalization {
        BackgroundFinalization::AlreadyTerminal => {
            return BackgroundTurnRetention {
                preserve_history: false,
                delivery_failed: false,
            };
        }
        BackgroundFinalization::Suspended => {
            return BackgroundTurnRetention {
                preserve_history: true,
                delivery_failed: false,
            };
        }
        BackgroundFinalization::Terminal(plan) => plan,
    };
    before_publication();
    let mut publication = prepare_background_publication(target, &plan);
    {
        let mut os = shared_os.lock().unwrap();
        commit_background_publication(os.as_mut(), target, &mut publication);
    }
    finish_background_publication(plan, publication, before_cleanup)
}

pub(super) fn publish_background_preparation_failure_with_io(
    shared_os: &aios_kernel::SharedKernel,
    pid: u64,
    result_channel_id: Option<u64>,
    completion_futex_addr: Option<aios_kernel::primitives::FutexAddr>,
    error: &str,
    before_publication: impl FnOnce(),
    before_cleanup: impl FnOnce(),
) {
    let goal = shared_os
        .lock()
        .unwrap()
        .get_process(pid)
        .map(|process| process.goal.clone());
    let task_id = goal
        .as_deref()
        .and_then(task_tools::decode_os_task_goal)
        .map(|goal| goal.task_id)
        .or_else(|| {
            result_channel_id.and_then(|channel| task_tools::task_id_for_result(pid, channel))
        });
    let backend_id = task_id
        .as_deref()
        .and_then(|task_id| task_tools::with_task_entry(task_id, |entry| entry.backend))
        .flatten()
        .unwrap_or_default();
    let target = BackgroundTurnTarget {
        pid,
        task_id,
        result_channel_id,
        completion_futex_addr,
        backend_id,
    };
    let outcome = BackgroundTaskOutcome {
        state: BackgroundTaskState::Failed(error.to_string()),
        output: String::new(),
        progress: String::new(),
        backend_session: None,
        retain_recovery: false,
    };
    finalize_background_outcome_with_io(
        shared_os,
        &target,
        outcome,
        BackgroundFinalizationOrigin::Preparation,
        before_publication,
        before_cleanup,
    );
}

/// The existing behavior tests use a caller-owned LocalOS without a shared mutex.
/// Run the same production stages, retaining their narrow IPC/quota assertions.
#[cfg(test)]
fn finalize_background_turn(
    os: &mut (dyn aios_kernel::kernel::Kernel + Send),
    target: &BackgroundTurnTarget,
    outcome: BackgroundTaskOutcome,
) -> BackgroundTurnRetention {
    let plan = match reserve_background_finalization(
        os,
        target,
        outcome,
        BackgroundFinalizationOrigin::Turn,
    ) {
        BackgroundFinalization::AlreadyTerminal => {
            return BackgroundTurnRetention {
                preserve_history: false,
                delivery_failed: false,
            };
        }
        BackgroundFinalization::Suspended => {
            return BackgroundTurnRetention {
                preserve_history: true,
                delivery_failed: false,
            };
        }
        BackgroundFinalization::Terminal(plan) => plan,
    };
    let mut publication = prepare_background_publication(target, &plan);
    commit_background_publication(os, target, &mut publication);
    finish_background_publication(plan, publication, || {})
}

/// Reclaim child history and private artifacts when no resumable process or recovery
/// source remains. Suspensions, memory-merge failures, publication warnings, and caught
/// worker panics explicitly retain them; normal terminal cleanup runs through Drop.
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
        self.release_memory_index();
        self.memory_path = None;
        self.cwd_dir = None;
    }

    fn preserve_memory(&mut self) {
        self.release_memory_index();
        self.memory_path = None;
    }

    /// Preserving the private memory file keeps it on disk for later evidence
    /// reads, but the derived SQLite index holds a database connection in this
    /// process. Drop the cached index so a finished task does not pin a handle
    /// for the rest of the parent's lifetime; a later access reopens it lazily.
    fn release_memory_index(&self) {
        if let Some(memory_path) = self.memory_path.as_deref() {
            crate::ai::tools::storage::memory_store::remove_memory_index_entry(memory_path);
        }
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
/// silent, refreshing one compact status block only at scheduler safe points,
/// so it cannot interleave with foreground output.
pub(super) struct SubagentStatusLine {
    last_line: Option<String>,
    last_columns: Option<usize>,
    cursor_queries_disabled: bool,
    last_serve_key: Option<Vec<serde_json::Value>>,
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
        let is_tty = std::io::stdout().is_terminal();
        Self {
            last_line: None,
            last_columns: None,
            cursor_queries_disabled: false,
            last_serve_key: None,
            is_tty,
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

        // Snapshot the task registry before taking the kernel lock. The registry is never held
        // while the kernel is locked, so the status line cannot form the reverse of the
        // registry -> kernel order used by the task tools.
        let snapshots_by_session = session_ids
            .into_iter()
            .map(|session_id| {
                let snapshots =
                    crate::ai::tools::task_tools::subagent_status_snapshots(&session_id);
                (session_id, snapshots)
            })
            .collect::<Vec<_>>();
        let statuses_by_session = {
            let mut os = app.os.lock().unwrap_or_else(|err| err.into_inner());
            snapshots_by_session
                .into_iter()
                .map(|(session_id, snapshots)| {
                    let statuses = crate::ai::tools::task_tools::subagent_terminal_statuses(
                        os.as_mut(),
                        snapshots,
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

        let status_line = render_subagent_status_line(&statuses);
        if crate::ai::background::serve_live_streaming() {
            self.publish_status_if_changed(&statuses, &status_line, |status_line| {
                crate::ai::background::publish_serve_frame(
                    crate::ai::background::ServeLiveKind::SubagentProgress,
                    status_line,
                );
            });
        }

        // The side-note composer (Ctrl+G) owns the footer row / DECSTBM scroll
        // region while open: the in-place redraw is deferred (see
        // `redraw_status_line`) so its bare `\r\x1b[2K` cannot displace the
        // output cursor or erase the composer draft. Serve frames above still
        // publish. This deferral applies to every scheduler-tick refresh in the
        // run loop — both the task_wait PARKED wait window and the
        // background-subagent wait window.
        self.redraw_status_line(
            &status_line,
            crate::ai::stream::side_note_input::side_note_composer_active(),
        );
    }

    fn publish_status_if_changed(
        &mut self,
        statuses: &[crate::ai::tools::task_tools::SubagentTerminalStatus],
        status_line: &str,
        publish: impl FnOnce(&str),
    ) {
        let key = agent_team_subagent_status_semantic_key(statuses);
        if self.last_serve_key.as_ref() == Some(&key) {
            return;
        }
        publish(status_line);
        self.last_serve_key = Some(key);
    }

    /// In-place TTY redraw of the status line, gated on the side-note composer.
    /// While `composer_active` the composer owns the footer row / DECSTBM scroll
    /// region, so the write is deferred (see `status_line_redraw_deferred`):
    /// its row erase would displace the output cursor and erase the
    /// composer draft, and the deferred line is not recorded.
    fn redraw_status_line(&mut self, status_line: &str, composer_active: bool) {
        if composer_active || !self.is_tty {
            return;
        }
        // The raw side-note listener must yield before a CPR query. Acquire
        // input before stdout so its shutdown cannot wait on our output lock.
        let _stdin_owner = crate::commonw::prompt::acquire_foreground_stdin();
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        if crate::ai::stream::side_note_input::side_note_composer_active() {
            return;
        }
        let cols = self.emulator_columns(&mut out);
        let line = clamp_status_block_at_width(status_line, cols);
        if status_line_redraw_deferred(
            composer_active,
            self.last_line
                .as_deref()
                .filter(|_| self.last_columns == Some(cols)),
            &line,
            self.is_tty,
        ) {
            return;
        }
        if redraw_status_block(&mut out, self.last_line.as_deref(), &line, true).is_ok() {
            self.last_line = Some(line);
            self.last_columns = Some(cols);
        }
    }

    fn emulator_columns(&mut self, out: &mut impl Write) -> usize {
        if !self.cursor_queries_disabled && std::io::stdin().is_terminal() {
            match crate::ai::prompt::SingleRowFrame::measure_columns(out, || {
                crossterm::cursor::position().map(|(column, _)| column)
            }) {
                Ok(cols) => return usize::from(cols),
                // Unsupported terminals must not pay a query timeout on every
                // scheduler refresh. Auto-wrap stays disabled on that path.
                Err(_) => self.cursor_queries_disabled = true,
            }
        }
        crate::ai::stream::raw_terminal_cols()
    }

    /// Before the foreground resumes streaming output or the input box, finalize
    /// and end the dynamic line so it no longer occupies a cursor row.
    pub(super) fn finish(&mut self) {
        self.last_serve_key = None;
        self.finish_status_line(crate::ai::stream::side_note_input::side_note_composer_active());
    }

    /// Finalize the dynamic status line, deferred while the side-note composer
    /// owns the footer row / DECSTBM scroll region: never finalize a dynamic
    /// line into the transcript while the composer is open. The status itself
    /// is not lost (serve frames and tool results carry it); only the transient
    /// TTY redraw is deferred. Keep `last_line` so a later refresh/finish after
    /// the composer closes still sees and finalizes it.
    fn finish_status_line(&mut self, composer_active: bool) {
        if composer_active {
            return;
        }
        let Some(line) = self.last_line.take() else {
            return;
        };
        if self.is_tty {
            let _stdin_owner = crate::commonw::prompt::acquire_foreground_stdin();
            let stdout = std::io::stdout();
            let mut out = stdout.lock();
            if crate::ai::stream::side_note_input::side_note_composer_active() {
                self.last_line = Some(line);
                return;
            }
            // Hard line feeds restore independent transcript rows before the
            // cursor leaves this temporary group. Otherwise its padded edge
            // links would become ordinary reflowable transcript content.
            let cols = self.emulator_columns(&mut out);
            let final_line = clamp_status_block_at_width(&line, cols);
            let _ = redraw_status_block(&mut out, Some(&line), &final_line, false);
            let _ = writeln!(out);
            let _ = out.flush();
        }
        self.last_columns = None;
    }
}

fn subagent_status_semantic_key(
    statuses: &[crate::ai::tools::task_tools::SubagentTerminalStatus],
) -> Vec<serde_json::Value> {
    statuses
        .iter()
        .map(|status| {
            let progress = status.progress.as_deref().unwrap_or_default();
            let snapshot = status.progress_snapshot.as_ref();
            let plan = match snapshot {
                Some(snapshot) => snapshot
                    .checkpoint_summary
                    .as_deref()
                    .map(sanitize_status_field)
                    .filter(|summary| !summary.is_empty()),
                None => progress.split_once(" · plan: ").map(|(_, plan)| {
                    let summary = plan
                        .rsplit_once(" · elapsed ")
                        .map_or(plan, |(summary, _)| summary);
                    sanitize_status_field(summary)
                }),
            };
            let (phase, latest) = match snapshot {
                Some(snapshot) => (
                    sanitize_status_field(&snapshot.phase),
                    subagent_activity_progress(status).map(|activity| activity.latest),
                ),
                None => (
                    sanitize_status_field(
                        &progress
                            .split(" · ")
                            .filter(|field| {
                                !field.starts_with("elapsed ")
                                    && !field.starts_with("last activity ")
                                    && !field.starts_with("updates ")
                                    && *field != "checkpoint due"
                            })
                            .collect::<Vec<_>>()
                            .join(" · "),
                    ),
                    None,
                ),
            };
            serde_json::json!({
                "task": [status.description, status.agent_name, status.backend,
                    status.model, status.reasoning_effort],
                "state": status.state,
                "phase": phase,
                "latest": latest,
                "plan": plan,
                "pending_side_notes": progress.split(" · ")
                    .find(|field| field.starts_with("pending side-notes: "))
                    .map(sanitize_status_field),
                "checkpoint_due": snapshot.map_or_else(
                    || progress.ends_with(" · checkpoint due"),
                    |snapshot| snapshot.checkpoint_due,
                ),
            })
        })
        .collect()
}

fn agent_team_subagent_status_semantic_key(
    statuses: &[crate::ai::tools::task_tools::SubagentTerminalStatus],
) -> Vec<serde_json::Value> {
    let active_statuses = sorted_active_subagent_statuses(statuses);
    let mut key = vec![serde_json::json!({
        "total": statuses.len(),
        "active": active_statuses.len(),
    })];
    key.extend(active_statuses.into_iter().map(|status| {
        serde_json::json!({
            "task_id": status.task_id,
            "status": subagent_status_semantic_key(std::slice::from_ref(status)),
            "updates": subagent_activity_progress(status).and_then(|activity| activity.updates),
        })
    }));
    key
}

/// Preserve spawn order across refreshes; task id breaks the unlikely timestamp
/// tie. Progress event arrival, adding a task, or removing a task therefore
/// cannot reorder the remaining entries.
fn sorted_active_subagent_statuses(
    statuses: &[crate::ai::tools::task_tools::SubagentTerminalStatus],
) -> Vec<&crate::ai::tools::task_tools::SubagentTerminalStatus> {
    let mut active_statuses = statuses
        .iter()
        .filter(|status| !is_terminal_subagent_state(&status.state))
        .collect::<Vec<_>>();
    active_statuses.sort_by(|a, b| {
        a.started_at
            .cmp(&b.started_at)
            .then_with(|| a.task_id.cmp(&b.task_id))
    });
    active_statuses
}

fn render_agent_team_subagent_status_block_at_width(
    statuses: &[crate::ai::tools::task_tools::SubagentTerminalStatus],
    width: usize,
) -> String {
    let active_statuses = sorted_active_subagent_statuses(statuses);
    let count = if active_statuses.len() == statuses.len() {
        format!("subagents · {} active", active_statuses.len())
    } else {
        format!(
            "subagents · {}/{} active",
            active_statuses.len(),
            statuses.len()
        )
    };
    let mut lines = vec![truncate_status_field_to_width(&count, width)];
    let last_index = active_statuses.len().saturating_sub(1);
    for (index, status) in active_statuses.into_iter().enumerate() {
        let is_last = index == last_index;
        lines.push(render_agent_team_subagent_header(status, is_last, width));
        if let Some(detail) = render_agent_team_subagent_detail(status, is_last, width) {
            lines.push(detail);
        }
    }
    lines.join("\n")
}

fn render_agent_team_subagent_header(
    status: &crate::ai::tools::task_tools::SubagentTerminalStatus,
    is_last: bool,
    width: usize,
) -> String {
    let prefix = if is_last { "└─ " } else { "├─ " };
    let mut identity = String::new();
    if let Some(backend) = status.backend.as_deref() {
        identity.push_str(&format!("[backend={}] ", sanitize_status_field(backend)));
    }
    identity.push_str(&sanitize_status_field(&status.agent_name));
    if let Some(model) = status.model.as_deref() {
        identity.push_str(&format!(" ({})", sanitize_status_field(model)));
    }
    if let Some(effort) = status.reasoning_effort.as_deref() {
        identity.push_str(&format!(" (effort: {})", sanitize_status_field(effort)));
    }
    let state = format!(
        " · {} {}",
        sanitize_status_field(&status.state),
        format_subagent_elapsed(status.elapsed_secs)
    );
    let task_prefix = " · task: ";
    let content_width = width
        .saturating_sub(display_width(prefix))
        .saturating_sub(display_width(&state))
        .saturating_sub(display_width(task_prefix));
    let task = sanitize_status_field(&status.description);
    let task_budget = (content_width / 2).min(display_width(&task));
    let identity_budget = content_width.saturating_sub(task_budget);
    let identity = truncate_status_field_to_width(&identity, identity_budget);
    let task = truncate_status_field_to_width(
        &task,
        content_width.saturating_sub(display_width(&identity)),
    );
    truncate_status_field_to_width(
        &format!("{prefix}{identity}{state}{task_prefix}{task}"),
        width,
    )
}

fn render_agent_team_subagent_detail(
    status: &crate::ai::tools::task_tools::SubagentTerminalStatus,
    is_last: bool,
    width: usize,
) -> Option<String> {
    let prefix = if is_last { "   ↳ " } else { "│  ↳ " };
    if let Some(activity) = subagent_activity_progress(status) {
        let phase_prefix = "phase: ";
        let updates = activity
            .updates
            .map(|updates| format!(" · updates {updates}"))
            .unwrap_or_default();
        let latest_prefix = format!("{updates} · latest: ");
        let age = sanitize_status_field(&activity.age);
        let age_suffix = if age.ends_with("ago") {
            format!(" · {age}")
        } else {
            format!(" · {age} ago")
        };
        let content_width = width
            .saturating_sub(display_width(prefix))
            .saturating_sub(display_width(&phase_prefix))
            .saturating_sub(display_width(&latest_prefix))
            .saturating_sub(display_width(&age_suffix));
        let phase_budget = content_width / 3;
        let latest_budget = content_width.saturating_sub(phase_budget);
        let phase = truncate_status_field_to_width(&activity.phase, phase_budget);
        let latest = truncate_status_field_to_width(&activity.latest, latest_budget);
        return Some(render_status_line_with_prefix(
            prefix,
            &format!("{phase_prefix}{phase}{latest_prefix}{latest}{age_suffix}"),
            width,
        ));
    }

    let progress = status
        .progress
        .as_deref()
        .map(status_progress_text)
        .or_else(|| {
            status
                .progress_snapshot
                .as_ref()
                .map(|snapshot| status_progress_text(&snapshot.display_status_line()))
        })?;
    if progress.trim().is_empty() {
        return None;
    }
    Some(render_status_line_with_prefix(
        prefix,
        &format!("progress: {progress}"),
        width,
    ))
}

/// Tree indentation is trusted static text. Keep it outside field sanitization,
/// which intentionally folds whitespace and would otherwise collapse or remove
/// the spaces that align detail arrows across entries.
fn render_status_line_with_prefix(prefix: &str, content: &str, width: usize) -> String {
    let prefix_width = display_width(prefix);
    if prefix_width > width {
        let mut rendered = String::new();
        let mut rendered_width = 0;
        for ch in prefix.chars() {
            let char_width = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
            if rendered_width + char_width > width {
                break;
            }
            rendered.push(ch);
            rendered_width += char_width;
        }
        return rendered;
    }
    let content = truncate_status_field_to_width(content, width - prefix_width);
    format!("{prefix}{content}")
}

fn render_subagent_status_line(
    statuses: &[crate::ai::tools::task_tools::SubagentTerminalStatus],
) -> String {
    render_subagent_status_line_at_width(statuses, crate::ai::stream::raw_terminal_cols())
}

fn render_subagent_status_line_at_width(
    statuses: &[crate::ai::tools::task_tools::SubagentTerminalStatus],
    width: usize,
) -> String {
    render_agent_team_subagent_status_block_at_width(statuses, width)
}

struct ActivityProgress {
    latest: String,
    age: String,
    phase: String,
    updates: Option<u64>,
}

/// Project activity from the shared snapshot, independent of progress text order.
fn subagent_activity_progress(
    status: &crate::ai::tools::task_tools::SubagentTerminalStatus,
) -> Option<ActivityProgress> {
    let snapshot = status.progress_snapshot.as_ref()?;
    let latest = snapshot
        .timeline
        .iter()
        .rev()
        .find(|event| event.kind == runtime_ctx::SubagentProgressEventKind::Activity)?;
    let latest = sanitize_status_field(&latest.summary);
    let phase = sanitize_status_field(&snapshot.phase);
    if latest.is_empty() || latest == phase {
        return None;
    }
    Some(ActivityProgress {
        latest,
        age: runtime_ctx::format_progress_duration(snapshot.stale_for),
        phase,
        updates: (snapshot.sequence > 1).then_some(snapshot.sequence),
    })
}

fn display_width(value: &str) -> usize {
    unicode_width::UnicodeWidthStr::width(value)
}

fn truncate_status_field_to_width(value: &str, max_width: usize) -> String {
    truncate_status_row_to_width(&sanitize_status_field(value), max_width)
}

fn truncate_status_row_to_width(value: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    if display_width(&value) <= max_width {
        return value.to_string();
    }
    let budget = max_width.saturating_sub(1);
    let mut width = 0;
    let mut truncated = String::new();
    for ch in value.chars() {
        let char_width = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + char_width > budget {
            break;
        }
        truncated.push(ch);
        width += char_width;
    }
    truncated.push('…');
    // Presentation selectors can widen a character beyond its individual width.
    while display_width(&truncated) > max_width {
        truncated.pop();
        truncated.pop();
        truncated.push('…');
    }
    truncated
}

fn clamp_status_block(status: &str) -> String {
    clamp_status_block_at_width(status, crate::ai::stream::raw_terminal_cols())
}

fn clamp_status_block_at_width(status: &str, cols: usize) -> String {
    let width = cols.saturating_sub(2).max(1);
    status
        .lines()
        .map(|line| truncate_status_row_to_width(line, width))
        .collect::<Vec<_>>()
        .join("\n")
}

fn redraw_status_block(
    out: &mut impl Write,
    previous: Option<&str>,
    status: &str,
    preserve_group: bool,
) -> std::io::Result<()> {
    let mut frame = crate::ai::prompt::SingleRowFrame::begin(out)?;
    let previous_rows = previous.map_or(1, |line| line.lines().count().max(1));
    let current_rows = status.lines().count().max(1);
    // ECH preserves isWrapped; full-line EL would sever the cursor group.
    // Its saturated count erases the actual emulator width, including when
    // the PTY still reports the width of the preceding frame.
    write!(frame.output(), "\r\x1b[65535X")?;
    for _ in 1..previous_rows {
        write!(frame.output(), "\x1b[1A\r\x1b[65535X")?;
    }
    if current_rows > previous_rows {
        if previous_rows > 1 {
            write!(frame.output(), "\x1b[{}B", previous_rows - 1)?;
        }
        // Reserve new rows before visible ink is written. These hard LFs
        // cannot reflow a partially painted status into the transcript.
        for _ in previous_rows..current_rows {
            write!(frame.output(), "\r\n")?;
        }
        write!(frame.output(), "\x1b[{}A\r", current_rows - 1)?;
    } else if current_rows < previous_rows {
        // Detach only the released rows, whose cells have already been erased.
        write!(frame.output(), "\x1b[{}B", current_rows)?;
        for row in current_rows..previous_rows {
            write!(frame.output(), "\r\x1b[2K")?;
            if row + 1 < previous_rows {
                write!(frame.output(), "\x1b[1B")?;
            }
        }
        write!(frame.output(), "\x1b[{}A\r", previous_rows - 1)?;
    }
    if preserve_group && previous.is_none_or(|_| current_rows != previous_rows) {
        frame.prepare_cursor_group(current_rows)?;
    }
    let mut lines = status.lines();
    if let Some(first) = lines.next() {
        write!(frame.output(), "{first}")?;
    }
    for line in lines {
        if !preserve_group {
            write!(frame.output(), "\r\n")?;
        } else {
            // LF resets xterm's isWrapped flag. Reuse rows with cursor motion
            // so a resize partway through painting cannot split the group.
            write!(frame.output(), "\r\x1b[1B")?;
        }
        write!(frame.output(), "{line}")?;
    }
    write!(frame.output(), "\r")?;
    frame.finish()
}

fn is_terminal_subagent_state(state: &str) -> bool {
    matches!(
        state,
        "completed" | "failed" | "cancelled" | "canceled" | "timeout" | "terminated"
    )
}

/// The outer status line already prints the task's elapsed time, so drop the
/// duplicate `elapsed` segment from the progress snapshot while keeping the
/// more useful last-activity age.
fn status_progress_text(progress: &str) -> String {
    let Some((head, tail)) = progress.split_once(" · elapsed ") else {
        return progress.to_string();
    };
    match tail.split_once(" · last activity ") {
        Some((_, last)) => format!("{head} · last {last}"),
        None => head.to_string(),
    }
}

/// Coexistence rule with the side-note composer (Ctrl+G): while the composer
/// owns the footer row / DECSTBM scroll region, the in-place status-line
/// redraw must be deferred — its bare `\r\x1b[2K` would displace the output
/// cursor and erase the composer draft. The deferred line is also not
/// recorded, so a later `finish` has nothing stale to finalize; the next
/// refresh after the composer closes redraws it. Unchanged lines and non-TTY
/// streams defer for the same reason (nothing new to draw). This applies to
/// every scheduler-tick refresh in the run loop, i.e. both the task_wait
/// PARKED wait window and the background-subagent wait window.
fn status_line_redraw_deferred(
    composer_active: bool,
    last_line: Option<&str>,
    line: &str,
    is_tty: bool,
) -> bool {
    composer_active || last_line == Some(line) || !is_tty
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
        BackgroundDispatchSpec, BackgroundSubagentHistoryGuard, BackgroundTurnTarget,
        SubagentStatusLine, agent_team_subagent_status_semantic_key, clamp_status_block,
        finalize_background_turn, format_subagent_elapsed, redraw_status_block,
        render_agent_team_subagent_status_block_at_width, render_subagent_status_line_at_width,
        resolve_background_task_outcome, sanitize_status_field, select_background_batch_with_log,
        sorted_active_subagent_statuses, status_line_redraw_deferred, supervise_background_worker,
    };
    use crate::ai::driver::runtime_ctx::{
        SubagentProgressSnapshot, new_subagent_progress_slot, publish_subagent_activity_to_slot,
        publish_subagent_checkpoint_to_slot, publish_subagent_phase_to_slot,
        subagent_progress_state_snapshot,
    };
    use crate::ai::tools::task_tools::SubagentTerminalStatus;
    use aios_kernel::kernel::{KernelInternal, ProcessState, Syscall};
    use aios_kernel::local::LocalOS;
    use aios_kernel::primitives::{
        ChannelId, FutexOps, IpcOps, IpcRecvResult, ResourceLimit, ResourceUsageDelta, RlimitOps,
    };

    fn background_turn_fixture() -> (LocalOS, BackgroundTurnTarget) {
        let mut os = LocalOS::new();
        let root = os.begin_foreground("root".to_string(), "goal".to_string(), 10, 8, None);
        let pid = os
            .spawn(
                Some(root),
                "child".to_string(),
                "goal".to_string(),
                20,
                8,
                None,
                None,
            )
            .unwrap();
        assert_eq!(os.pop_ready().unwrap().pid, pid);
        let channel = os.channel_create(Some(root), 1, "turn result".to_string());
        let futex = os.futex_create(0, "turn completion".to_string());
        (
            os,
            BackgroundTurnTarget {
                pid,
                task_id: None,
                result_channel_id: Some(channel.0),
                completion_futex_addr: Some(futex),
                backend_id: crate::ai::tools::task_tools::BackendId::KERNEL,
            },
        )
    }

    fn background_turn_output(body: &str) -> crate::ai::tools::task_tools::Outcome {
        crate::ai::tools::task_tools::Outcome {
            parent_payload: format!("Evidence retained for the parent\n{body}"),
            validation_body: body.to_string(),
            progress: "finished test turn".to_string(),
            backend_session: Some("test-backend-session".to_string()),
            ..Default::default()
        }
    }

    fn background_turn_payload(
        os: &mut LocalOS,
        target: &BackgroundTurnTarget,
    ) -> serde_json::Value {
        let IpcRecvResult::Message(payload) = os
            .channel_try_recv(None, ChannelId(target.result_channel_id.unwrap()))
            .unwrap()
        else {
            panic!("the background turn must publish its terminal result");
        };
        serde_json::from_str(&payload).unwrap()
    }

    struct PublicationFixture {
        root: std::path::PathBuf,
        history_file: std::path::PathBuf,
        child_history: std::path::PathBuf,
        shared: aios_kernel::SharedKernel,
        target: BackgroundTurnTarget,
        context: std::sync::Arc<crate::ai::driver::runtime_ctx::DriverContext>,
        owner_pid: u64,
        session_id: String,
    }

    impl PublicationFixture {
        fn new() -> Self {
            use crate::ai::tools::task_tools::{AsyncTaskEntry, InheritOptions};
            use aios_kernel::primitives::ChannelOwnerTag;

            aios_kernel::kernel::register_current_pid_provider(super::super::current_task_pid);
            let root = std::env::temp_dir().join(format!(
                "runtime-lock-publication-{}",
                uuid::Uuid::new_v4().simple()
            ));
            std::fs::create_dir_all(&root).unwrap();
            let history_file = root.join("history.sqlite");
            let child_history = root.join("child.sqlite");
            std::fs::write(&child_history, "child recovery fixture").unwrap();
            let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
            let session_id = format!("lock-session-{}", uuid::Uuid::new_v4().simple());
            let mut app = super::super::tests::test_app("build");
            app.session_id = session_id.clone();
            app.config.history_file = history_file.clone();
            app.session_history_file = history_file.clone();
            let shared = app.os.clone();
            let (owner_pid, pid, channel, futex) = {
                let mut os = shared.lock().unwrap();
                let owner =
                    os.begin_foreground("parent".to_string(), "goal".to_string(), 10, 8, None);
                let pid = os
                    .spawn(
                        Some(owner),
                        "child".to_string(),
                        "goal".to_string(),
                        20,
                        8,
                        None,
                        None,
                    )
                    .unwrap();
                assert_eq!(os.pop_ready().unwrap().pid, pid);
                os.get_process_mut(pid).unwrap().history_file = Some(child_history.clone());
                let channel = os.channel_create_tagged_with_holders(
                    Some(owner),
                    1,
                    "task result".to_string(),
                    ChannelOwnerTag::TaskResult,
                    vec![
                        "task_result.producer".to_string(),
                        "task_result.consumer".to_string(),
                    ],
                );
                let futex = os.futex_create(0, "task completion".to_string());
                os.set_current_pid(Some(owner));
                (owner, pid, channel, futex)
            };
            crate::ai::tools::task_tools::insert_task_entry_for_test(
                task_id.clone(),
                AsyncTaskEntry {
                    session_id: session_id.clone(),
                    evidence_history_file: Some(history_file.clone()),
                    owner_pid,
                    pid,
                    result_channel_id: channel.0,
                    completion_futex_addr: futex,
                    description: "real task publication".to_string(),
                    agent_name: "build".to_string(),
                    backend: None,
                    model: "test-model".to_string(),
                    is_model_auto_selected: false,
                    auto_model_fallback: None,
                    selection_explanation: "fixture".to_string(),
                    inherit: InheritOptions::default(),
                    abort_handle: None,
                    cancel_stream: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                    started_at: std::time::Instant::now(),
                    last_progress_notification_at: None,
                    last_progress_persisted_at: None,
                },
            );
            let context = crate::ai::driver::runtime_ctx::DriverContext::from_app_snapshot(
                &app,
                std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new())),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
            );
            Self {
                root,
                history_file,
                child_history,
                shared,
                target: BackgroundTurnTarget {
                    pid,
                    task_id: Some(task_id),
                    result_channel_id: Some(channel.0),
                    completion_futex_addr: Some(futex),
                    backend_id: crate::ai::tools::task_tools::BackendId::KERNEL,
                },
                context,
                owner_pid,
                session_id,
            }
        }

        fn scope<T>(&self, run: impl FnOnce() -> T) -> T {
            super::super::TASK_PID.sync_scope(Some(self.owner_pid), || {
                crate::ai::driver::runtime_ctx::DRIVER_CTX.sync_scope(self.context.clone(), || {
                    crate::ai::driver::runtime_ctx::TURN_IDENTITY
                        .sync_scope((self.session_id.clone(), 0usize), run)
                })
            })
        }

        fn published(&self) -> Option<(String, String, bool)> {
            crate::ai::history::read_available_task_evidence_status_payload(
                &self.history_file,
                &self.session_id,
                self.target.task_id.as_deref().unwrap(),
            )
            .unwrap()
        }

        fn payload(&self) -> serde_json::Value {
            let os = self.shared.lock().unwrap();
            let IpcRecvResult::Message(payload) = os
                .channel_peek(None, ChannelId(self.target.result_channel_id.unwrap()))
                .unwrap()
            else {
                panic!("terminal IPC payload missing")
            };
            serde_json::from_str(&payload).unwrap()
        }
    }

    impl Drop for PublicationFixture {
        fn drop(&mut self) {
            crate::ai::tools::task_tools::remove_task_entry(
                self.target.task_id.as_deref().unwrap(),
            );
            for backend in [
                crate::ai::tools::task_tools::BackendId::KERNEL,
                crate::ai::tools::task_tools::BackendId::TRAECLI,
            ] {
                crate::ai::tools::task_tools::backend_session::forget_session(
                    backend,
                    self.target.task_id.as_deref().unwrap(),
                );
            }
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn runtime_lock_publication_real_task_persists_outside_kernel() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let fixture = PublicationFixture::new();
        let outcome = resolve_background_task_outcome(
            background_turn_output("validated final body"),
            None,
            None,
        );
        let mut called = false;
        let retention = super::finalize_background_turn_with_io(
            &fixture.shared,
            &fixture.target,
            outcome,
            || {
                assert!(
                    fixture.shared.try_lock().is_ok(),
                    "JSON and ledger publication must run without the kernel guard"
                );
                called = true;
            },
            || {},
        );
        assert!(called);
        assert!(!retention.delivery_failed);
        assert_eq!(fixture.payload()["status"], "completed");
        let (status, body, delivered) = fixture.published().unwrap();
        assert_eq!(status, "completed");
        assert!(body.contains("validated final body"));
        assert!(!delivered);
        let os = fixture.shared.lock().unwrap();
        assert_eq!(os.rusage_get(fixture.target.pid).unwrap().turns, 1);
        assert_eq!(
            os.get_process(fixture.target.pid)
                .unwrap()
                .result
                .as_deref(),
            Some("Completed")
        );
    }

    #[test]
    fn runtime_lock_publication_prepare_failure_cleans_after_persistence() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let fixture = PublicationFixture::new();
        fixture
            .shared
            .lock()
            .unwrap()
            .get_process_mut(fixture.target.pid)
            .unwrap()
            .goal = "AIOS_SUBAGENT_TASK:{corrupt".to_string();
        let mut cleaned = false;
        super::super::scheduler::publish_background_task_failure_with_io(
            &fixture.shared,
            fixture.target.pid,
            fixture.target.result_channel_id,
            fixture.target.completion_futex_addr,
            "preparation failed",
            || {
                assert!(
                    fixture.shared.try_lock().is_ok(),
                    "prepare-failure publication must run outside the kernel lock"
                );
            },
            || {
                assert!(
                    fixture.shared.try_lock().is_ok(),
                    "history deletion must run outside the kernel lock"
                );
                assert!(
                    fixture.child_history.exists(),
                    "cleanup must happen after terminal persistence"
                );
                assert_eq!(fixture.published().unwrap().0, "failed");
                assert_eq!(fixture.payload()["error"], "preparation failed");
                assert!(
                    fixture
                        .shared
                        .lock()
                        .unwrap()
                        .get_process(fixture.target.pid)
                        .unwrap()
                        .history_file
                        .is_none()
                );
                cleaned = true;
            },
        );
        assert!(cleaned);
        assert!(!fixture.child_history.exists());
        assert_eq!(
            fixture
                .shared
                .lock()
                .unwrap()
                .rusage_get(fixture.target.pid)
                .unwrap()
                .turns,
            0
        );
    }

    #[test]
    fn runtime_lock_publication_completion_claim_survives_cancel_and_collection() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let fixture = PublicationFixture::new();
        let outcome = resolve_background_task_outcome(
            background_turn_output("winning completion"),
            None,
            None,
        );
        super::finalize_background_turn_with_io(
            &fixture.shared,
            &fixture.target,
            outcome,
            || {
                assert!(fixture.shared.try_lock().is_ok());
                let context = fixture.context.clone();
                let task_id = fixture.target.task_id.clone().unwrap();
                let session_id = fixture.session_id.clone();
                let owner = fixture.owner_pid;
                let cancelled = std::thread::spawn(move || {
                    super::super::TASK_PID.sync_scope(Some(owner), || {
                        crate::ai::driver::runtime_ctx::DRIVER_CTX.sync_scope(context, || {
                            crate::ai::driver::runtime_ctx::TURN_IDENTITY.sync_scope(
                                (session_id, 0usize),
                                || {
                                    crate::ai::tools::task_tools::execute_task_cancel(
                                        &serde_json::json!({"task_ids":[task_id]}),
                                    )
                                    .unwrap()
                                },
                            )
                        })
                    })
                })
                .join()
                .unwrap();
                assert!(cancelled.contains("already finished"), "{cancelled}");
                let pending = fixture.scope(|| {
                    crate::ai::tools::task_tools::execute_task_status(
                        &serde_json::json!({"task_ids":[fixture.target.task_id.clone().unwrap()]}),
                    )
                    .unwrap()
                });
                assert!(
                    !pending.contains("terminated without publishing"),
                    "{pending}"
                );
                assert!(
                    crate::ai::tools::task_tools::with_task_entry(
                        fixture.target.task_id.as_deref().unwrap(),
                        |_| ()
                    )
                    .is_some()
                );
            },
            || {},
        );
        assert_eq!(fixture.payload()["status"], "completed");
        assert_eq!(fixture.published().unwrap().0, "completed");
        let collected = fixture.scope(|| crate::ai::tools::task_tools::execute_task_wait(&serde_json::json!({"task_ids":[fixture.target.task_id.clone().unwrap()], "timeout_secs":1})).unwrap());
        assert!(collected.contains("winning completion"), "{collected}");
    }

    #[test]
    fn runtime_lock_publication_cancel_wins_before_late_completion() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let fixture = PublicationFixture::new();
        fixture.scope(|| crate::ai::tools::task_tools::execute_task_cancel(&serde_json::json!({"task_ids":[fixture.target.task_id.clone().unwrap()], "reason":"cancel wins"})).unwrap());
        assert!(
            fixture.child_history.exists(),
            "uncollected cancellation retains its history"
        );
        let original = fixture
            .shared
            .lock()
            .unwrap()
            .get_process(fixture.target.pid)
            .unwrap()
            .result
            .clone();
        let outcome =
            resolve_background_task_outcome(background_turn_output("late completion"), None, None);
        super::finalize_background_turn_with_io(
            &fixture.shared,
            &fixture.target,
            outcome,
            || panic!("late completion must not publish"),
            || panic!("late completion must not delete artifacts"),
        );
        assert_eq!(fixture.payload()["status"], "cancelled");
        assert_eq!(fixture.published().unwrap().0, "cancelled");
        let os = fixture.shared.lock().unwrap();
        assert_eq!(os.get_process(fixture.target.pid).unwrap().result, original);
        assert_eq!(os.rusage_get(fixture.target.pid).unwrap().turns, 0);
        drop(os);
        let collected = fixture.scope(|| crate::ai::tools::task_tools::execute_task_wait(&serde_json::json!({"task_ids":[fixture.target.task_id.clone().unwrap()], "timeout_secs":1})).unwrap());
        assert!(collected.contains("cancel wins"));
        assert!(
            !fixture.child_history.exists(),
            "collection drains its external cleanup"
        );
    }

    #[test]
    fn runtime_lock_publication_cancel_and_timeout_persist_warnings_outside_kernel() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        for status in ["cancelled", "timeout"] {
            let fixture = PublicationFixture::new();
            let task_id = fixture.target.task_id.as_deref().unwrap();
            let mut entry =
                crate::ai::tools::task_tools::with_task_entry(task_id, Clone::clone).unwrap();
            entry.backend = Some(crate::ai::tools::task_tools::BackendId::TRAECLI);
            crate::ai::tools::task_tools::insert_task_entry_for_test(task_id.to_string(), entry);
            let session = "cccccccc-1111-2222-3333-444444444444";
            let slot = crate::ai::tools::task_tools::backend_session::track_session(
                crate::ai::tools::task_tools::BackendId::TRAECLI,
                task_id,
            );
            *slot.lock().unwrap() = Some(session.to_string());
            fixture
                .shared
                .lock()
                .unwrap()
                .channel_close(None, ChannelId(fixture.target.result_channel_id.unwrap()))
                .unwrap();
            fixture.scope(|| {
                crate::ai::tools::task_tools::write_terminal_task_result_with_io(
                    &fixture.shared,
                    fixture.target.task_id.as_deref().unwrap(),
                    fixture.target.pid,
                    fixture.target.result_channel_id.unwrap(),
                    fixture.target.completion_futex_addr.unwrap(),
                    status,
                    "terminal fixture reason",
                    || {
                        assert!(
                            fixture.shared.try_lock().is_ok(),
                            "cancellation/timeout persistence must run outside kernel lock"
                        )
                    },
                    || {
                        assert!(
                            fixture.shared.try_lock().is_ok(),
                            "delivery warning persistence must run outside kernel lock"
                        );
                        assert_eq!(fixture.published().unwrap().0, status);
                    },
                )
                .unwrap()
            });
            let (stored_status, body, delivered) = fixture.published().unwrap();
            assert_eq!(stored_status, status);
            assert!(body.contains("terminal fixture reason"));
            assert!(body.contains("IPC publication failed"));
            assert!(body.contains(session));
            assert!(!delivered);
            {
                let os = fixture.shared.lock().unwrap();
                assert_eq!(
                    os.get_process(fixture.target.pid).unwrap().state,
                    ProcessState::Terminated
                );
                assert_eq!(
                    os.futex_load(fixture.target.completion_futex_addr.unwrap()),
                    Some(1)
                );
            }
            let collected = fixture.scope(|| {
                crate::ai::tools::task_tools::execute_task_wait(
                    &serde_json::json!({"task_ids":[task_id], "timeout_secs":1}),
                )
                .unwrap()
            });
            assert!(collected.contains("terminal fixture reason"));
            assert!(collected.contains(session));
            assert!(fixture.published().unwrap().2);
        }
    }

    #[test]
    fn background_turn_success_charges_once_and_preserves_payload() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let (mut os, target) = background_turn_fixture();
        os.rlimit_set(target.pid, ResourceLimit::from_legacy(1))
            .unwrap();
        let body = r#"{"count":1}"#;
        let schema = serde_json::json!({"type":"object", "required":["count"], "properties":{"count":{"type":"integer"}}});
        let outcome =
            resolve_background_task_outcome(background_turn_output(body), None, Some(&schema));
        let retention = finalize_background_turn(&mut os, &target, outcome);
        let payload = background_turn_payload(&mut os, &target);
        assert_eq!(payload["status"], "completed");
        assert!(payload["error"].is_null());
        assert_eq!(
            payload["output"],
            format!("Evidence retained for the parent\n{body}")
        );
        assert_eq!(payload["backend_session"], "test-backend-session");
        assert_eq!(os.rusage_get(target.pid).unwrap().turns, 1);
        assert_eq!(os.get_process(target.pid).unwrap().turns_used, 1);
        assert_eq!(
            os.get_process(target.pid).unwrap().result.as_deref(),
            Some("Completed")
        );
        assert_eq!(
            os.get_process(target.pid).unwrap().state,
            ProcessState::Terminated
        );
        assert_eq!(
            os.futex_load(target.completion_futex_addr.unwrap()),
            Some(1)
        );
        assert!(!retention.preserve_history && !retention.delivery_failed);
    }

    #[test]
    fn background_turn_execution_failure_charges_once() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let (mut os, target) = background_turn_fixture();
        let mut outcome = background_turn_output("partial work");
        outcome.execution_error = Some("worker execution failed".to_string());
        let outcome = resolve_background_task_outcome(
            outcome,
            Some("secondary memory error".to_string()),
            None,
        );
        finalize_background_turn(&mut os, &target, outcome);
        let payload = background_turn_payload(&mut os, &target);
        assert_eq!(payload["status"], "failed");
        assert_eq!(payload["error"], "worker execution failed");
        assert!(payload["output"].as_str().unwrap().contains("partial work"));
        assert_eq!(
            os.get_process(target.pid).unwrap().result.as_deref(),
            Some("Failed: worker execution failed")
        );
        assert_eq!(os.rusage_get(target.pid).unwrap().turns, 1);
    }

    #[test]
    fn background_turn_malformed_handoff_is_failed_everywhere() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        for body in [
            "<subagent_handoff>{}",
            "<subagent_handoff>invalid JSON</subagent_handoff>",
            r#"<subagent_handoff>{"kind":"permission_required","reason":" "}</subagent_handoff>"#,
        ] {
            let (mut os, target) = background_turn_fixture();
            let outcome = resolve_background_task_outcome(background_turn_output(body), None, None);
            finalize_background_turn(&mut os, &target, outcome);
            let payload = background_turn_payload(&mut os, &target);
            assert_eq!(payload["status"], "failed", "{payload}");
            let error = payload["error"].as_str().unwrap();
            assert_eq!(
                os.get_process(target.pid).unwrap().result.as_deref(),
                Some(format!("Failed: {error}").as_str())
            );
        }
    }

    #[test]
    fn background_turn_schema_failure_is_failed_everywhere() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        for body in ["not JSON", r#"{"count":"wrong"}"#] {
            let (mut os, target) = background_turn_fixture();
            let schema = serde_json::json!({"type":"object", "required":["count"], "properties":{"count":{"type":"integer"}}});
            let outcome =
                resolve_background_task_outcome(background_turn_output(body), None, Some(&schema));
            finalize_background_turn(&mut os, &target, outcome);
            let payload = background_turn_payload(&mut os, &target);
            assert_eq!(payload["status"], "failed");
            let error = payload["error"].as_str().unwrap();
            assert_eq!(
                os.get_process(target.pid).unwrap().result.as_deref(),
                Some(format!("Failed: {error}").as_str())
            );
        }
    }

    #[test]
    fn background_turn_memory_failure_is_failed_and_retains_recovery() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let (mut os, target) = background_turn_fixture();
        let outcome = resolve_background_task_outcome(
            background_turn_output("valid final response"),
            Some("private memory merge failed".to_string()),
            None,
        );
        let retention = finalize_background_turn(&mut os, &target, outcome);
        let payload = background_turn_payload(&mut os, &target);
        assert_eq!(payload["status"], "failed");
        assert_eq!(payload["error"], "private memory merge failed");
        assert_eq!(
            os.get_process(target.pid).unwrap().result.as_deref(),
            Some("Failed: private memory merge failed")
        );
        assert!(
            retention.preserve_history,
            "memory merge failure retains child recovery artifacts"
        );
    }

    #[test]
    fn background_turn_valid_handoff_bypasses_schema_and_is_blocked_everywhere() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        for kind in [
            "authorization_required",
            "permission_required",
            "user_input_required",
        ] {
            let (mut os, target) = background_turn_fixture();
            let body = format!(
                "<subagent_handoff>{{\"kind\":\"{kind}\",\"reason\":\"approval needed\",\"suggested_action\":\"approve this operation\"}}</subagent_handoff>"
            );
            let schema = serde_json::json!({"type":"object", "required":["count"]});
            let outcome =
                resolve_background_task_outcome(background_turn_output(&body), None, Some(&schema));
            finalize_background_turn(&mut os, &target, outcome);
            let payload = background_turn_payload(&mut os, &target);
            assert_eq!(payload["status"], "blocked");
            assert_eq!(payload["output"], "approve this operation");
            assert_eq!(payload["error"], format!("{kind}: approval needed"));
            assert_eq!(
                os.get_process(target.pid).unwrap().result.as_deref(),
                Some(format!("Blocked: {kind}: approval needed").as_str())
            );
        }
    }

    #[test]
    fn background_turn_quota_failure_is_decided_before_publication() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let (mut os, target) = background_turn_fixture();
        os.rlimit_set(target.pid, ResourceLimit::from_legacy(1))
            .unwrap();
        os.rusage_charge(
            target.pid,
            ResourceUsageDelta {
                turns: 1,
                ..Default::default()
            },
        );
        let outcome =
            resolve_background_task_outcome(background_turn_output("final answer"), None, None);
        finalize_background_turn(&mut os, &target, outcome);
        let payload = background_turn_payload(&mut os, &target);
        assert_eq!(payload["status"], "failed");
        assert!(
            payload["error"]
                .as_str()
                .unwrap()
                .contains("Resource limit exceeded")
        );
        assert_eq!(os.rusage_get(target.pid).unwrap().turns, 2);
        assert!(
            os.get_process(target.pid)
                .unwrap()
                .result
                .as_ref()
                .unwrap()
                .contains("Resource limit exceeded")
        );
    }

    #[test]
    fn background_turn_suspension_keeps_result_open_and_charges_once() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        for state in [
            ProcessState::Stopped,
            ProcessState::Sleeping { until_tick: 10 },
            ProcessState::Waiting {
                reason: aios_kernel::kernel::WaitReason::ProcessExit { on_pid: 99 },
            },
        ] {
            for schema in [None, Some(serde_json::json!({"type":"object"}))] {
                let (mut os, target) = background_turn_fixture();
                os.get_process_mut(target.pid).unwrap().state = state.clone();
                let outcome = resolve_background_task_outcome(
                    crate::ai::tools::task_tools::Outcome::default(),
                    None,
                    schema.as_ref(),
                );
                let retention = finalize_background_turn(&mut os, &target, outcome);
                assert_eq!(
                    os.channel_peek(None, ChannelId(target.result_channel_id.unwrap()))
                        .unwrap(),
                    IpcRecvResult::Empty
                );
                assert_eq!(
                    os.futex_load(target.completion_futex_addr.unwrap()),
                    Some(0)
                );
                assert_eq!(os.get_process(target.pid).unwrap().state, state);
                assert_eq!(os.rusage_get(target.pid).unwrap().turns, 1);
                assert!(retention.preserve_history);
            }
        }
    }

    #[test]
    fn background_turn_preserves_existing_terminal_result() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let (mut os, target) = background_turn_fixture();
        let channel = ChannelId(target.result_channel_id.unwrap());
        os.channel_send(
            Some(target.pid),
            channel,
            serde_json::json!({"status":"cancelled", "error":"user cancelled"}).to_string(),
        )
        .unwrap();
        os.channel_close(None, channel).unwrap();
        os.terminate_current("cancelled: user cancelled".to_string());
        let outcome = resolve_background_task_outcome(
            background_turn_output("late successful response"),
            None,
            None,
        );
        finalize_background_turn(&mut os, &target, outcome);
        assert_eq!(
            background_turn_payload(&mut os, &target)["status"],
            "cancelled"
        );
        assert_eq!(
            os.get_process(target.pid).unwrap().result.as_deref(),
            Some("cancelled: user cancelled")
        );
        assert_eq!(os.rusage_get(target.pid).unwrap().turns, 0);
    }

    #[tokio::test]
    async fn background_turn_worker_panic_publishes_one_failure_with_recovery() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let (mut os, target) = background_turn_fixture();
        let phase = new_subagent_progress_slot();
        publish_subagent_checkpoint_to_slot(
            &phase,
            "saved checkpoint",
            "verified work before panic",
        );
        let events = crate::ai::tools::task_tools::Events {
            phase_slot: Some(phase),
            session_slot: Some(std::sync::Arc::new(std::sync::Mutex::new(Some(
                "worker-session".to_string(),
            )))),
        };
        let (outcome, panicked) = supervise_background_worker(
            async { panic!("injected worker panic") },
            &events,
            "working",
        )
        .await;
        assert!(panicked);
        let mut outcome = resolve_background_task_outcome(outcome, None, None);
        outcome.retain_recovery |= panicked;
        let retention = finalize_background_turn(&mut os, &target, outcome);
        let payload = background_turn_payload(&mut os, &target);
        assert_eq!(payload["status"], "failed");
        assert!(
            payload["error"]
                .as_str()
                .unwrap()
                .contains("injected worker panic")
        );
        assert!(
            payload["output"]
                .as_str()
                .unwrap()
                .contains("verified work before panic")
        );
        assert_eq!(payload["backend_session"], "worker-session");
        assert_eq!(os.rusage_get(target.pid).unwrap().turns, 1);
        assert_eq!(
            os.get_process(target.pid).unwrap().state,
            ProcessState::Terminated
        );
        assert!(retention.preserve_history);
        let late =
            resolve_background_task_outcome(background_turn_output("late response"), None, None);
        finalize_background_turn(&mut os, &target, late);
        assert_eq!(os.rusage_get(target.pid).unwrap().turns, 1);
        assert_eq!(
            os.channel_peek(None, ChannelId(target.result_channel_id.unwrap()))
                .unwrap(),
            IpcRecvResult::Closed
        );
    }

    #[tokio::test]
    async fn background_turn_worker_supervision_keeps_abort_semantics() {
        let events = crate::ai::tools::task_tools::Events::default();
        let (started, ready) = tokio::sync::oneshot::channel();
        let worker = tokio::spawn(async move {
            supervise_background_worker(
                async move {
                    started.send(()).unwrap();
                    std::future::pending().await
                },
                &events,
                "working",
            )
            .await
        });
        ready.await.unwrap();
        worker.abort();
        assert!(worker.await.unwrap_err().is_cancelled());
    }

    #[test]
    fn background_turn_suspended_over_quota_warns_and_stays_pending() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let (mut os, target) = background_turn_fixture();
        os.rlimit_set(target.pid, ResourceLimit::from_legacy(1))
            .unwrap();
        os.rusage_charge(
            target.pid,
            ResourceUsageDelta {
                turns: 1,
                ..Default::default()
            },
        );
        os.get_process_mut(target.pid).unwrap().state = ProcessState::Stopped;
        let schema = serde_json::json!({"type":"object"});
        let outcome = resolve_background_task_outcome(
            crate::ai::tools::task_tools::Outcome::default(),
            None,
            Some(&schema),
        );
        finalize_background_turn(&mut os, &target, outcome);
        assert_eq!(os.rusage_get(target.pid).unwrap().turns, 2);
        let process = os.get_process(target.pid).unwrap();
        assert_eq!(process.state, ProcessState::Stopped);
        assert!(
            process
                .mailbox
                .iter()
                .any(|message| message.contains("[rlimit-warning]")
                    && message.contains("used=2, limit=1"))
        );
        assert_eq!(
            os.channel_peek(None, ChannelId(target.result_channel_id.unwrap()))
                .unwrap(),
            IpcRecvResult::Empty
        );
        assert_eq!(
            os.futex_load(target.completion_futex_addr.unwrap()),
            Some(0)
        );
    }

    #[test]
    fn background_turn_empty_final_response_still_validates_schema() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let (mut os, target) = background_turn_fixture();
        let schema = serde_json::json!({"type":"object"});
        let outcome = resolve_background_task_outcome(
            crate::ai::tools::task_tools::Outcome::default(),
            None,
            Some(&schema),
        );
        finalize_background_turn(&mut os, &target, outcome);
        let payload = background_turn_payload(&mut os, &target);
        assert_eq!(payload["status"], "failed");
        assert!(
            payload["error"]
                .as_str()
                .unwrap()
                .contains("not valid JSON")
        );
        assert_eq!(
            os.get_process(target.pid).unwrap().state,
            ProcessState::Terminated
        );
    }

    #[test]
    fn background_turn_final_response_terminates_a_suspended_task() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let (mut os, target) = background_turn_fixture();
        os.get_process_mut(target.pid).unwrap().state = ProcessState::Stopped;
        let outcome =
            resolve_background_task_outcome(background_turn_output("final response"), None, None);
        finalize_background_turn(&mut os, &target, outcome);
        assert_eq!(
            background_turn_payload(&mut os, &target)["status"],
            "completed"
        );
        assert_eq!(
            os.get_process(target.pid).unwrap().state,
            ProcessState::Terminated
        );
        assert_eq!(
            os.get_process(target.pid).unwrap().result.as_deref(),
            Some("Completed")
        );
    }

    fn activity_snapshot(phase: &str, latest: &str, sequence: u64) -> SubagentProgressSnapshot {
        let slot = new_subagent_progress_slot();
        publish_subagent_activity_to_slot(&slot, phase, latest);
        let mut snapshot = subagent_progress_state_snapshot(&slot).unwrap();
        snapshot.sequence = sequence;
        snapshot.elapsed = std::time::Duration::from_secs(8);
        snapshot.stale_for = std::time::Duration::from_secs(2);
        snapshot
    }

    #[test]
    fn scheduler_dispatch_log_sink_runs_after_kernel_unlock() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        crate::ai::driver::scheduler::reset_scheduler_test_state();
        let os = crate::ai::driver::new_local_kernel();
        let expected_pids = {
            let mut kernel = os.lock().unwrap();
            (0..2)
                .map(|index| {
                    kernel
                        .spawn(
                            None,
                            format!("fixture-{index}"),
                            "goal".to_string(),
                            10,
                            10,
                            None,
                            None,
                        )
                        .unwrap()
                })
                .collect::<Vec<_>>()
        };
        let session_id = "scheduler-dispatch-unlocked-sink";
        let mut sink_calls = 0;
        let selected = select_background_batch_with_log(&os, 1, |decision| {
            assert!(
                os.try_lock().is_ok(),
                "the log sink must run after the kernel guard is dropped"
            );
            sink_calls += 1;
            decision.emit(session_id);
        });
        assert_eq!(sink_calls, 1);
        assert_eq!(
            selected.iter().map(|proc| proc.pid).collect::<Vec<_>>(),
            expected_pids
        );
        let logs = crate::ai::driver::decision_log::get_decision_log_store()
            .recent_by_session(session_id, 2);
        assert_eq!(logs.len(), 1);
        let decision = &logs[0];
        assert_eq!(decision.session_id, session_id);
        assert_eq!(decision.turn_id, 1);
        assert_eq!(
            decision.decision_type,
            crate::ai::driver::decision_log::DecisionType::SchedulerDispatch
        );
        assert_eq!(decision.context, "scheduler_epoch=1");
        assert_eq!(
            decision.chosen_option,
            format!("selected={expected_pids:?}")
        );
        assert_eq!(decision.reasoning, "eligible=2 deferred=0");
        assert_eq!(
            decision.alternatives_considered,
            expected_pids
                .iter()
                .map(|pid| format!("pid={pid} score=1015 cooldown_until=0 fail_streak=0"))
                .collect::<Vec<_>>()
        );
        assert!(decision.confidence.is_none());
        assert!(decision.execution_time_ms.is_none());
        let outcome = decision.outcome.as_ref().unwrap();
        assert!(outcome.success);
        assert_eq!(outcome.message, "scheduler decision accepted");
        assert!(outcome.user_feedback.is_none());
        crate::ai::driver::scheduler::reset_scheduler_test_state();
    }

    #[test]
    fn dispatch_launch_background_spec_preserves_task_goal_inputs() {
        use crate::ai::tools::task_tools::{BackendId, OsTaskGoal};

        let goal: OsTaskGoal = serde_json::from_value(serde_json::json!({
            "task_id": "task-launch",
            "result_channel_id": 71,
            "completion_futex_addr": 83,
            "description": "stored task description",
            "prompt": "stored task prompt",
            "agent_name": "launch-agent",
            "model": "kernel-launch-model",
            "selection_explanation": "stored selection",
            "auto_model_fallback": {"require_thinking": true, "target_tier": "Heavy"},
            "spawn_depth": 2,
            "parent_agent_team_mode": true,
            "response_schema": {"type": "string"},
            "backend": "traecli",
            "backend_model": "external-launch-model",
            "backend_continue_from": "previous-task",
            "active_skill_names": ["second-skill", "first-skill"],
            "image_files": ["/task/second.png", "/task/first.png"],
        }))
        .unwrap();
        let history_path = std::path::PathBuf::from("child-history.sqlite");
        let reasoning_effort = Some("high".to_string());
        let spec = BackgroundDispatchSpec::from_task_goal(
            97,
            "prepared wakeup question".to_string(),
            history_path.clone(),
            Some(&goal),
            true,
            false,
            &reasoning_effort,
        );

        assert_eq!(spec.pid, 97);
        assert_eq!(spec.proc_question, "prepared wakeup question");
        assert_eq!(spec.history_path, history_path);
        assert_eq!(
            spec.agent_override.as_deref(),
            Some(goal.agent_name.as_str())
        );
        assert_eq!(spec.model_override.as_deref(), Some(goal.model.as_str()));
        assert_eq!(spec.result_channel_id, Some(goal.result_channel_id));
        assert_eq!(
            spec.completion_futex_addr.map(|addr| addr.0),
            Some(goal.completion_futex_addr)
        );
        assert_eq!(spec.task_id.as_deref(), Some(goal.task_id.as_str()));
        assert_eq!(spec.response_schema, goal.response_schema);
        let fallback = spec.auto_model_fallback.unwrap();
        assert_eq!(
            fallback.require_thinking,
            goal.auto_model_fallback.unwrap().require_thinking
        );
        assert_eq!(
            fallback.target_tier,
            goal.auto_model_fallback.unwrap().target_tier
        );
        assert_eq!(spec.spawn_depth, goal.spawn_depth);
        assert_eq!(spec.parent_agent_team_mode, goal.parent_agent_team_mode);
        assert!(spec.is_resume_wakeup);
        assert!(!spec.initialize_history);
        assert_eq!(spec.backend_id, BackendId::TRAECLI);
        assert_eq!(spec.backend_model, goal.backend_model);
        assert_eq!(spec.backend_continue_from, goal.backend_continue_from);
        assert_eq!(spec.active_skill_names, goal.active_skill_names);
        assert_eq!(spec.backend_reasoning_effort, reasoning_effort);
        assert_eq!(spec.image_files, goal.image_files);
    }

    #[test]
    fn dispatch_launch_background_spec_preserves_non_task_defaults() {
        for reasoning_effort in [None, Some("medium".to_string())] {
            for (is_resume_wakeup, initialize_history) in
                [(false, false), (true, false), (false, true), (true, true)]
            {
                let spec = BackgroundDispatchSpec::from_task_goal(
                    19,
                    "non-task question".to_string(),
                    std::path::PathBuf::from("non-task-history.sqlite"),
                    None,
                    is_resume_wakeup,
                    initialize_history,
                    &reasoning_effort,
                );

                assert_eq!(spec.pid, 19);
                assert_eq!(spec.proc_question, "non-task question");
                assert_eq!(
                    spec.history_path,
                    std::path::Path::new("non-task-history.sqlite")
                );
                assert!(spec.agent_override.is_none());
                assert!(spec.model_override.is_none());
                assert!(spec.result_channel_id.is_none());
                assert!(spec.completion_futex_addr.is_none());
                assert!(spec.task_id.is_none());
                assert!(spec.response_schema.is_none());
                assert!(spec.auto_model_fallback.is_none());
                assert_eq!(spec.spawn_depth, 0);
                assert!(!spec.parent_agent_team_mode);
                assert_eq!(spec.is_resume_wakeup, is_resume_wakeup);
                assert_eq!(spec.initialize_history, initialize_history);
                assert_eq!(
                    spec.backend_id,
                    crate::ai::tools::task_tools::BackendId::KERNEL
                );
                assert!(spec.backend_model.is_none());
                assert!(spec.backend_continue_from.is_none());
                assert!(spec.active_skill_names.is_empty());
                assert_eq!(spec.backend_reasoning_effort, reasoning_effort);
                assert!(spec.image_files.is_empty());
            }
        }
    }

    #[test]
    fn background_decoded_kernel_goal_ignores_legacy_continuation() {
        use crate::ai::tools::task_tools::{
            BackendId, ContinueContext, OsTaskGoal, decode_os_task_goal, encode_os_task_goal,
        };

        let mut goal: OsTaskGoal = serde_json::from_value(serde_json::json!({
            "task_id": "task_legacy_resume",
            "result_channel_id": 1,
            "completion_futex_addr": 2,
            "description": "persisted kernel continuation",
            "prompt": "Original kernel task prompt.",
            "agent_name": "navigator",
            "model": "kernel-model",
            "selection_explanation": "persisted fixture",
            "backend_continue_from": "nonempty-legacy-resume-reference",
        }))
        .unwrap();

        for stored_backend in [
            None,
            Some("kernel"),
            Some("unknown-backend"),
            Some("trae-cli"),
            Some("coco"),
        ] {
            goal.backend = stored_backend.map(str::to_string);
            let encoded = encode_os_task_goal(&goal).unwrap();
            let decoded = decode_os_task_goal(&encoded).expect("legacy combination must decode");
            assert!(
                super::decode_background_process_task_goal(&encoded)
                    .unwrap()
                    .is_some()
            );
            let backend_id = BackendId::from_stored(decoded.backend.as_deref());
            assert_eq!(backend_id, BackendId::KERNEL);
            let backend = backend_id.resolve();
            let context = ContinueContext {
                continue_from: decoded.backend_continue_from.as_deref(),
                parent_history: std::path::Path::new("unused-parent.sqlite"),
                prompt: &decoded.prompt,
            };
            // Normal kernel continuation remains rejected; decoded background goals ignore it.
            assert_eq!(
                backend.continuation(&context).unwrap_err(),
                "'backend_continue_from' needs backend 'traecli'"
            );
            let continuation =
                super::prepare_background_subagent_continuation(backend.as_ref(), &context)
                    .expect("background kernel must ignore the legacy resume field");
            let (resume_session, prompt, side_note_files) = continuation.into_parts();
            assert!(resume_session.is_none());
            assert_eq!(prompt, decoded.prompt);
            assert!(side_note_files.is_empty());
        }
    }

    #[test]
    fn subagent_status_field_is_safe_single_line_text() {
        assert_eq!(
            sanitize_status_field("review\ncode\r\x1b[31m now\tplease"),
            "review code [31m now please"
        );
    }

    #[test]
    fn live_status_dedup_ignores_elapsed_and_activity_age() {
        for activity in [None, Some("command: rg -n foo src")] {
            let slot = new_subagent_progress_slot();
            if let Some(activity) = activity {
                publish_subagent_activity_to_slot(&slot, "running command", activity);
            } else {
                publish_subagent_phase_to_slot(&slot, "calling model");
            }
            publish_subagent_checkpoint_to_slot(&slot, "inspect task path", "Saved plan");
            let mut snapshot = subagent_progress_state_snapshot(&slot).unwrap();
            snapshot.elapsed = std::time::Duration::from_secs(5);
            snapshot.stale_for = std::time::Duration::from_secs(1);
            let mut status = SubagentTerminalStatus {
                task_id: "task-focus".to_string(),
                started_at: std::time::Instant::now(),
                description: "inspect".to_string(),
                agent_name: "explorer".to_string(),
                backend: Some("traecli".to_string()),
                model: Some("gpt-5.5".to_string()),
                reasoning_effort: None,
                state: "running".to_string(),
                elapsed_secs: 5,
                progress: Some(snapshot.display_status_line()),
                progress_snapshot: Some(snapshot.clone()),
            };
            let mut line = SubagentStatusLine::new();
            let mut published = Vec::new();
            let first = render_subagent_status_line_at_width(&[status.clone()], 120);
            line.publish_status_if_changed(&[status.clone()], &first, |payload| {
                published.push(payload.to_string());
            });
            line.publish_status_if_changed(&[status.clone()], &first, |payload| {
                published.push(payload.to_string());
            });
            assert_eq!(published, vec![first.clone()]);

            status.elapsed_secs = 8;
            snapshot.elapsed = std::time::Duration::from_secs(8);
            snapshot.stale_for = std::time::Duration::from_secs(4);
            snapshot.checkpoint_summary = Some("inspect  task\tpath".to_string());
            status.progress = Some(snapshot.display_status_line());
            status.progress_snapshot = Some(snapshot);
            let later = render_subagent_status_line_at_width(&[status.clone()], 120);
            assert_ne!(first, later);
            line.publish_status_if_changed(&[status], &later, |payload| {
                published.push(payload.to_string());
            });
            assert_eq!(published, vec![first]);
        }
    }

    #[test]
    fn live_status_dedup_publishes_semantic_changes_and_resets_on_finish() {
        let mut status = SubagentTerminalStatus {
            task_id: "task-focus".to_string(),
            started_at: std::time::Instant::now(),
            description: "inspect".to_string(),
            agent_name: "explorer".to_string(),
            backend: Some("traecli".to_string()),
            model: Some("gpt-5.5".to_string()),
            reasoning_effort: None,
            state: "running".to_string(),
            elapsed_secs: 5,
            progress: None,
            progress_snapshot: None,
        };
        let slot = new_subagent_progress_slot();
        let mut line = SubagentStatusLine::new();
        line.is_tty = false;
        let mut published = 0;
        for change in 0..7 {
            match change {
                0 => publish_subagent_activity_to_slot(
                    &slot,
                    "running command",
                    "command: rg -n foo src",
                ),
                1 => publish_subagent_phase_to_slot(&slot, "processing output"),
                2 => publish_subagent_activity_to_slot(
                    &slot,
                    "processing output",
                    "completed: rg -n foo src · exit 0",
                ),
                3 => publish_subagent_checkpoint_to_slot(&slot, "inspect task path", "Saved plan"),
                4 => publish_subagent_checkpoint_to_slot(&slot, "verify task path", "Updated plan"),
                // A new event with the same checkpoint still advances visible updates.
                5 => {
                    publish_subagent_checkpoint_to_slot(&slot, "verify task path", "Verified plan")
                }
                _ => {}
            }
            let mut snapshot = subagent_progress_state_snapshot(&slot).unwrap();
            snapshot.checkpoint_due = change == 6;
            status.progress = Some(snapshot.display_status_line());
            status.progress_snapshot = Some(snapshot);
            let rendered = render_subagent_status_line_at_width(&[status.clone()], 120);
            line.publish_status_if_changed(&[status.clone()], &rendered, |_| published += 1);
        }
        assert_eq!(published, 7);

        status.description = "inspect another task".to_string();
        let rendered = render_subagent_status_line_at_width(&[status.clone()], 120);
        line.publish_status_if_changed(&[status.clone()], &rendered, |_| published += 1);
        assert_eq!(published, 8);
        line.finish();
        assert!(line.last_serve_key.is_none());
        line.publish_status_if_changed(&[status], &rendered, |_| published += 1);
        assert_eq!(published, 9);
    }

    #[test]
    fn subagent_status_line_shows_counts_elapsed_and_truncates_details() {
        let started_at = std::time::Instant::now();
        let statuses = [
            SubagentTerminalStatus {
                task_id: "task-review-stream".to_string(),
                started_at,
                description: "review stream".to_string(),
                agent_name: "explorer".to_string(),
                backend: None,
                model: None,
                reasoning_effort: None,
                state: "running".to_string(),
                elapsed_secs: 38,
                progress: Some(format!(
                    "using read_file · plan: {}",
                    "inspect task path ".repeat(80)
                )),
                progress_snapshot: None,
            },
            SubagentTerminalStatus {
                task_id: "task-review-driver".to_string(),
                started_at: started_at + std::time::Duration::from_secs(1),
                description: "review driver".to_string(),
                agent_name: "explorer".to_string(),
                backend: None,
                model: None,
                reasoning_effort: None,
                state: "waiting".to_string(),
                elapsed_secs: 61,
                progress: Some("waiting for task_wait PARKED".to_string()),
                progress_snapshot: None,
            },
            SubagentTerminalStatus {
                task_id: "task-compile-check".to_string(),
                started_at: started_at + std::time::Duration::from_secs(2),
                description: "compile check".to_string(),
                agent_name: "executor".to_string(),
                backend: None,
                model: None,
                reasoning_effort: None,
                state: "completed".to_string(),
                elapsed_secs: 3661,
                progress: None,
                progress_snapshot: None,
            },
            SubagentTerminalStatus {
                task_id: "task-extra".to_string(),
                started_at: started_at + std::time::Duration::from_secs(3),
                description: "extra task".to_string(),
                agent_name: "executor".to_string(),
                backend: None,
                model: None,
                reasoning_effort: None,
                state: "running".to_string(),
                elapsed_secs: 5,
                progress: None,
                progress_snapshot: None,
            },
            SubagentTerminalStatus {
                task_id: "task-last".to_string(),
                started_at: started_at + std::time::Duration::from_secs(4),
                description: "last task".to_string(),
                agent_name: "executor".to_string(),
                backend: None,
                model: None,
                reasoning_effort: None,
                state: "running".to_string(),
                elapsed_secs: 2,
                progress: None,
                progress_snapshot: None,
            },
        ];

        for width in [80, 120] {
            let block = render_subagent_status_line_at_width(&statuses, width);
            assert_eq!(block.lines().next(), Some("subagents · 4/5 active"));
            assert_eq!(block.lines().count(), 7, "{block}");
            for task in ["review stream", "review driver", "extra task", "last task"] {
                assert!(block.contains(&format!("task: {task}")), "{block}");
            }
            assert!(block.contains("running 38s"), "{block}");
            assert!(block.contains("waiting 1m1s"), "{block}");
            assert!(
                block.contains("progress: using read_file · plan:"),
                "{block}"
            );
            assert!(
                block.contains("progress: waiting for task_wait PARKED"),
                "{block}"
            );
            assert!(block.contains('…'), "{block}");
            assert!(block.lines().last().unwrap().starts_with("└─ "), "{block}");
            assert!(!block.contains("compile check"), "{block}");
            assert!(!block.contains(" more"), "{block}");
            assert!(
                block
                    .lines()
                    .all(|line| super::display_width(line) <= width)
            );
        }
        assert_eq!(format_subagent_elapsed(59), "59s");
        assert_eq!(format_subagent_elapsed(60), "1m0s");
        assert_eq!(format_subagent_elapsed(3661), "1h1m");
    }

    #[test]
    fn subagent_status_line_labels_external_backends_only() {
        let kernel = SubagentTerminalStatus {
            task_id: "task-kernel".to_string(),
            started_at: std::time::Instant::now(),
            description: "inspect".to_string(),
            agent_name: "explorer".to_string(),
            backend: None,
            model: None,
            reasoning_effort: None,
            state: "running".to_string(),
            elapsed_secs: 5,
            progress_snapshot: None,
            progress: None,
        };
        assert_eq!(
            render_subagent_status_line_at_width(&[kernel], 120),
            "subagents · 1 active\n└─ explorer · running 5s · task: inspect"
        );

        let mut external = SubagentTerminalStatus {
            task_id: "task-external".to_string(),
            started_at: std::time::Instant::now(),
            description: "inspect".to_string(),
            agent_name: "explorer".to_string(),
            backend: Some("traecli".to_string()),
            model: Some("gpt-5.5".to_string()),
            reasoning_effort: None,
            state: "running".to_string(),
            elapsed_secs: 5,
            progress_snapshot: None,
            progress: Some("thinking".to_string()),
        };
        // Short task labels leave room for the optional backend metadata.
        assert_eq!(
            render_subagent_status_line_at_width(&[external.clone()], 120),
            "subagents · 1 active\n└─ [backend=traecli] explorer (gpt-5.5) · running 5s · task: inspect\n   ↳ progress: thinking"
        );
        external.reasoning_effort = Some("xhigh".to_string());
        assert_eq!(
            render_subagent_status_line_at_width(&[external], 120),
            "subagents · 1 active\n└─ [backend=traecli] explorer (gpt-5.5) (effort: xhigh) · running 5s · task: inspect\n   ↳ progress: thinking"
        );
    }

    #[test]
    fn subagent_status_block_is_three_width_bounded_lines_with_live_activity() {
        let status = SubagentTerminalStatus {
            task_id: "task-live".to_string(),
            started_at: std::time::Instant::now(),
            description:
                "更新任务派发说明，同时保留每一个已有的内核兼容行为".to_string(),
            agent_name: "build-and-validation-worker".to_string(),
            backend: Some("traecli-with-a-very-long-backend-name".to_string()),
            model: Some("gpt-5.5".to_string()),
            reasoning_effort: Some("xhigh".to_string()),
            state: "running".to_string(),
            elapsed_secs: 598,
            progress_snapshot: Some(activity_snapshot(
                "processing output",
                "completed: rg -n \"backend\" src/bin/ai/driver/background_dispatch.rs",
                60,
            )),
            progress: Some(
                "completed: rg -n \"backend\" src/bin/ai/driver/background_dispatch.rs · last activity 2s ago · processing output · updates 60 · plan: preserve compatibility · elapsed 9m58s"
                    .to_string(),
            ),
        };
        let block = render_subagent_status_line_at_width(&[status.clone()], 120);
        let lines = block.lines().collect::<Vec<_>>();

        assert_eq!(lines.len(), 3, "{block}");
        assert_eq!(lines[0], "subagents · 1 active", "{block}");
        assert!(lines[1].starts_with("└─ [backend="), "{block}");
        assert!(lines[1].contains("running 9m58s"), "{block}");
        assert!(lines[1].contains("task: "), "{block}");
        assert!(
            lines[2].starts_with("   ↳ phase: processing output"),
            "{block}"
        );
        assert!(lines[2].contains("updates 60"), "{block}");
        assert!(lines[2].contains("latest: completed:"), "{block}");
        assert!(lines[2].ends_with(" · 2s ago"), "{block}");
        assert!(block.contains('…'), "{block}");
        for line in lines {
            assert!(
                unicode_width::UnicodeWidthStr::width(line) <= 120,
                "line exceeds width: {line:?}"
            );
        }

        for width in [72, 80] {
            let narrow = render_subagent_status_line_at_width(&[status.clone()], width);
            assert_eq!(narrow.lines().count(), 3, "{narrow}");
            assert!(narrow.contains("running 9m58s"), "{narrow}");
            assert!(narrow.contains("updates 60"), "{narrow}");
            assert!(narrow.contains("latest: completed:"), "{narrow}");
            assert!(
                narrow
                    .lines()
                    .all(|line| super::display_width(line) <= width)
            );
        }

        // Text can be absent, reordered, or stale without changing structured activity.
        for progress in [
            None,
            Some("elapsed 9m58s · invalid text · last activity 99s ago".to_string()),
        ] {
            let mut changed = status.clone();
            changed.progress = progress;
            assert_eq!(
                render_subagent_status_line_at_width(&[changed.clone()], 120),
                block
            );
            assert_eq!(
                agent_team_subagent_status_semantic_key(&[changed]),
                agent_team_subagent_status_semantic_key(&[status.clone()])
            );
        }

        let mut long_fields = status.clone();
        long_fields.backend = Some("backend❤\u{fe0f}".repeat(24));
        long_fields.model = Some("model❤\u{fe0f}".repeat(24));
        long_fields.reasoning_effort = Some("effort❤\u{fe0f}".repeat(24));
        long_fields.agent_name = "worker❤\u{fe0f}".repeat(24);
        long_fields.state = "running❤\u{fe0f}".repeat(24);
        long_fields.description = "任务❤\u{fe0f}".repeat(24);
        long_fields.progress_snapshot = Some(activity_snapshot(
            &"phase❤\u{fe0f}".repeat(24),
            &"activity❤\u{fe0f}".repeat(24),
            u64::MAX,
        ));
        let mut no_activity = long_fields.clone();
        no_activity.progress = None;
        no_activity.progress_snapshot = None;
        let mut kernel_activity = status.clone();
        kernel_activity.backend = None;
        kernel_activity.model = None;
        kernel_activity.reasoning_effort = None;
        for sample in [status, long_fields, no_activity, kernel_activity] {
            for width in 0..=128 {
                let block = render_subagent_status_line_at_width(&[sample.clone()], width);
                for line in block.split('\n') {
                    assert!(
                        unicode_width::UnicodeWidthStr::width(line) <= width,
                        "line exceeds width {width}: {line:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn agent_team_status_block_shows_all_active_subagents_stably() {
        let spawned = std::time::Instant::now();
        let status = |task_id: &str, label: &str, spawn_offset_secs: u64| SubagentTerminalStatus {
            task_id: task_id.to_string(),
            started_at: spawned + std::time::Duration::from_secs(spawn_offset_secs),
            description: format!("{label} task"),
            agent_name: format!("worker-{label}"),
            backend: Some("traecli".to_string()),
            model: Some("test-model".to_string()),
            reasoning_effort: Some("max".to_string()),
            state: "running".to_string(),
            elapsed_secs: 8,
            progress_snapshot: Some(activity_snapshot(
                &format!("processing {label}"),
                &format!("completed: {label} command · exit 0"),
                7,
            )),
            progress: None,
        };
        let statuses = vec![
            status("task-m", "gamma", 2),
            status("task-z", "alpha", 0),
            status("task-a", "beta", 1),
        ];

        let block = render_agent_team_subagent_status_block_at_width(&statuses, 160);
        let lines = block.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 7, "{block}");
        assert_eq!(clamp_status_block(&block).lines().count(), 7, "{block}");
        assert_eq!(lines[0], "subagents · 3 active");
        for (offset, label) in ["alpha", "beta", "gamma"].into_iter().enumerate() {
            let header = lines[1 + offset * 2];
            let detail = lines[2 + offset * 2];
            assert!(header.contains(label), "{block}");
            assert!(detail.contains(&format!("processing {label}")), "{block}");
            assert!(detail.contains(&format!("{label} command")), "{block}");
            for other in ["alpha", "beta", "gamma"] {
                if other != label {
                    assert!(!header.contains(other), "{block}");
                    assert!(!detail.contains(other), "{block}");
                }
            }
        }
        assert!(lines[1].starts_with("├─ "), "{block}");
        assert!(lines[5].starts_with("└─ "), "{block}");
        assert!(lines[2].starts_with("│  ↳ "), "{block}");
        assert!(lines[4].starts_with("│  ↳ "), "{block}");
        assert!(lines[6].starts_with("   ↳ "), "{block}");
        for detail in [lines[2], lines[4], lines[6]] {
            assert_eq!(
                detail.chars().position(|ch| ch == '↳'),
                Some(3),
                "detail arrow is misaligned: {detail:?}"
            );
        }
        assert!(!block.contains(" more"), "{block}");

        let mut reordered = statuses.clone();
        reordered.reverse();
        let ordered_ids = |items: &[SubagentTerminalStatus]| {
            sorted_active_subagent_statuses(items)
                .into_iter()
                .map(|status| status.task_id.clone())
                .collect::<Vec<_>>()
        };
        let expected_order = ["task-z", "task-a", "task-m"].map(str::to_string);
        assert_eq!(ordered_ids(&statuses), expected_order);
        assert_eq!(ordered_ids(&reordered), expected_order);
        let mut tied = reordered.clone();
        for status in &mut tied {
            status.started_at = spawned;
        }
        assert_eq!(
            ordered_ids(&tied),
            ["task-a", "task-m", "task-z"].map(str::to_string)
        );
        let semantic_key = agent_team_subagent_status_semantic_key(&statuses);
        assert_eq!(
            semantic_key,
            agent_team_subagent_status_semantic_key(&reordered)
        );
        for task_id in ["task-z", "task-a", "task-m"] {
            let mut updated = reordered.clone();
            let progress = updated
                .iter_mut()
                .find(|status| status.task_id == task_id)
                .unwrap()
                .progress_snapshot
                .as_mut()
                .unwrap();
            progress.sequence = 8;
            assert_ne!(
                semantic_key,
                agent_team_subagent_status_semantic_key(&updated),
                "an update from {task_id} must refresh the frame"
            );
        }

        let changed = vec![
            status("task-d", "delta", 3),
            status("task-m", "gamma", 2),
            status("task-z", "alpha", 0),
        ];
        assert_eq!(
            ordered_ids(&changed),
            ["task-z", "task-m", "task-d"].map(str::to_string)
        );
        let narrow = render_subagent_status_line_at_width(&changed, 72);
        assert_eq!(narrow.lines().count(), 7, "{narrow}");
        for line in narrow.lines() {
            assert!(
                unicode_width::UnicodeWidthStr::width(line) <= 72,
                "line exceeds width: {line:?}"
            );
        }
    }

    #[test]
    fn kernel_status_without_activity_keeps_phase_and_saved_plan() {
        let slot = new_subagent_progress_slot();
        publish_subagent_checkpoint_to_slot(&slot, "inspect task path", "Saved task plan");
        for phase in [
            "streaming response",
            "using read_file",
            "using exec_command",
            "calling model",
        ] {
            publish_subagent_phase_to_slot(&slot, phase);
            let mut snapshot = subagent_progress_state_snapshot(&slot).unwrap();
            snapshot.elapsed = std::time::Duration::from_secs(65);
            snapshot.stale_for = std::time::Duration::from_secs(3);
            let status = SubagentTerminalStatus {
                task_id: "task-kernel".to_string(),
                started_at: std::time::Instant::now(),
                description: "inspect scheduler".to_string(),
                agent_name: "explorer".to_string(),
                backend: None,
                model: None,
                reasoning_effort: None,
                state: "running".to_string(),
                elapsed_secs: 65,
                progress: None,
                progress_snapshot: Some(snapshot),
            };
            let block = render_subagent_status_line_at_width(&[status.clone()], 120);
            assert_eq!(
                block,
                format!(
                    "subagents · 1 active\n└─ explorer · running 1m5s · task: inspect scheduler\n   ↳ progress: {phase} · plan: inspect task path · last 3s ago"
                )
            );
            assert!(!block.contains("latest:"), "{block}");
            assert!(!block.contains("[backend="), "{block}");
            let narrow = render_subagent_status_line_at_width(&[status], 80);
            assert!(narrow.contains(&format!("progress: {phase}")), "{narrow}");
            assert!(narrow.contains("plan: inspect task path"), "{narrow}");
            assert!(narrow.lines().all(|line| super::display_width(line) <= 80));
        }
    }

    /// Real production status output for an external PTY/emulator driver.
    /// Set RUST_TOOLS_STATUS_FIXTURE=1, send r to refresh, and q to finish.
    /// OSC 777 status;frame carries the sampled width and rendered rows.
    #[test]
    #[ignore]
    fn subagent_status_pty_fixture() -> std::io::Result<()> {
        use std::io::{IsTerminal, Read, Write};

        if std::env::var("RUST_TOOLS_STATUS_FIXTURE").as_deref() != Ok("1") {
            return Ok(());
        }
        if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
            return Err(std::io::Error::other("status fixture requires a PTY"));
        }
        struct RestoreTerminal;
        impl Drop for RestoreTerminal {
            fn drop(&mut self) {
                let _ = crossterm::terminal::disable_raw_mode();
            }
        }
        crossterm::terminal::enable_raw_mode()?;
        let _restore = RestoreTerminal;
        {
            let mut out = std::io::stdout().lock();
            for index in 0..96 {
                write!(
                    out,
                    "TRANSCRIPT{index:02}: unchanged response text abcdefghijklmnopqrstuvwxyz0123456789\r\n"
                )?;
            }
            write!(out, "BODY_END\r\n")?;
            out.flush()?;
        }
        let mut status = SubagentTerminalStatus {
            task_id: "status-fixture".to_string(),
            started_at: std::time::Instant::now(),
            description: "Audit false compile-pass claim and independently verify every resize frame".to_string(),
            agent_name: "build".to_string(),
            backend: Some("traecli".to_string()),
            model: Some("fixture-model".to_string()),
            reasoning_effort: Some("max".to_string()),
            state: "running".to_string(),
            elapsed_secs: 123,
            progress_snapshot: Some(activity_snapshot(
                "processing output",
                "completed: git diff --check and verify the terminal rendering path",
                91,
            )),
            progress: Some("completed: git diff --check and verify the terminal rendering path · last activity 2s ago · processing output · updates 91 · elapsed 2m3s".to_string()),
        };
        let mut display = SubagentStatusLine::new();
        let mut sequence = 0;
        loop {
            sequence += 1;
            let cols = crate::ai::stream::raw_terminal_cols();
            let block = render_subagent_status_line_at_width(&[status.clone()], cols);
            display.redraw_status_line(&block, false);
            let metadata = serde_json::json!({
                "sequence": sequence,
                "cols": cols,
                "lines": display.last_line,
            });
            let mut out = std::io::stdout().lock();
            write!(out, "\x1b]777;status;frame;{metadata}\x07")?;
            out.flush()?;
            drop(out);
            let mut key = [0];
            std::io::stdin().read_exact(&mut key)?;
            if key[0] == b'q' {
                display.finish();
                return Ok(());
            }
            status.elapsed_secs += 1;
        }
    }

    #[test]
    fn status_block_direct_ellipsis_reserves_the_link_columns() {
        let block = "subagents · 1 active\n└─ task: Audit false compile-pass claim\n   ↳ latest: completed validation";
        let narrow = super::clamp_status_block_at_width(block, 12);
        assert_eq!(narrow.lines().count(), 3);
        assert!(narrow.lines().all(|line| line.ends_with('…')));
        assert!(narrow.lines().all(|line| super::display_width(line) <= 10));
        assert!(narrow.lines().nth(1).unwrap().starts_with("└─ "));
        assert!(narrow.lines().nth(2).unwrap().starts_with("   ↳ "));
    }

    #[test]
    fn status_block_refresh_keeps_existing_rows_in_the_cursor_group() {
        let mut bytes = Vec::new();
        redraw_status_block(
            &mut bytes,
            Some("count\nheader\nprogress"),
            "new count\nnew header\nnew progress",
            true,
        )
        .unwrap();
        // Hard LF would sever the group while the first row already has ink,
        // allowing xterm to reflow it during the rest of the frame.
        assert!(!bytes.contains(&b'\n'));
        assert_eq!(
            bytes
                .windows(4)
                .filter(|bytes| *bytes == b"\x1b[1B")
                .count(),
            2
        );
    }

    #[test]
    fn status_block_redraw_clears_every_previous_row() {
        let mut output = Vec::new();
        redraw_status_block(
            &mut output,
            Some("old one\nold two\nold three"),
            "new one\nnew two",
            false,
        )
        .unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "\x1b[?2026h\x1b[?7l\r\x1b[65535X\x1b[1A\r\x1b[65535X\x1b[1A\r\x1b[65535X\x1b[2B\r\x1b[2K\x1b[2A\rnew one\r\nnew two\r\x1b[?7h\x1b[?2026l"
        );
    }

    #[test]
    fn status_line_redraw_defers_while_side_note_composer_is_open() {
        // The coexistence rule: an OPEN Ctrl+G side-note composer owns the
        // footer row / DECSTBM scroll region, so the in-place redraw must be
        // deferred (and the line not recorded) on every scheduler tick — both
        // in the task_wait PARKED wait window and during a turn.
        assert!(status_line_redraw_deferred(
            true,
            None,
            "subagents · 1 active",
            true
        ));
        assert!(status_line_redraw_deferred(
            true,
            Some("subagents · 1 active"),
            "subagents · 1 active",
            true
        ));
        // Same-line dedup and non-TTY streams also defer the redraw.
        assert!(status_line_redraw_deferred(
            false,
            Some("line"),
            "line",
            true
        ));
        assert!(status_line_redraw_deferred(false, None, "line", false));
        // Live path: a new line on a TTY with no composer open is drawn.
        assert!(!status_line_redraw_deferred(
            false,
            None,
            "subagents · 1 active",
            true
        ));
    }

    #[test]
    fn status_line_redraw_and_finish_skip_writes_while_composer_is_open() {
        let status = SubagentTerminalStatus {
            task_id: "task-parked".to_string(),
            started_at: std::time::Instant::now(),
            description: "wait for a task".to_string(),
            agent_name: "explorer".to_string(),
            backend: None,
            model: None,
            reasoning_effort: None,
            state: "waiting".to_string(),
            elapsed_secs: 5,
            progress: Some("task_wait PARKED".to_string()),
            progress_snapshot: None,
        };
        let cols = crate::ai::stream::raw_terminal_cols();
        let block = render_subagent_status_line_at_width(&[status.clone()], cols);
        let expected = clamp_status_block(&block);
        let mut line = SubagentStatusLine::new();
        line.is_tty = true; // Simulate a live TTY; cargo test captures stdout.

        // Refresh ticks while the composer is open: no record, no write.
        line.redraw_status_line(&block, true);
        assert_eq!(line.last_line, None);
        let two_tasks = render_subagent_status_line_at_width(&[status.clone(), status], cols);
        line.redraw_status_line(&two_tasks, true);
        assert_eq!(line.last_line, None);

        line.redraw_status_line(&block, false);
        assert_eq!(line.last_line.as_deref(), Some(expected.as_str()));
        assert_eq!(line.last_columns, Some(cols));

        // A changed width invalidates dedup even when the visible text is unchanged.
        line.last_columns = Some(cols + 1);
        line.redraw_status_line(&block, false);
        assert_eq!(line.last_columns, Some(cols));

        // Foreground resumption defers finishing until the composer closes.
        line.finish_status_line(true);
        assert_eq!(line.last_line.as_deref(), Some(expected.as_str()));
        line.finish_status_line(false);
        assert_eq!(line.last_line, None);
        assert_eq!(line.last_columns, None);
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

/// Persisted kernel goals ignore external resume fields, as the original dispatch did.
fn prepare_background_subagent_continuation(
    backend: &dyn task_tools::Backend,
    context: &task_tools::ContinueContext<'_>,
) -> Result<task_tools::Continuation, String> {
    match backend.execution_mode() {
        task_tools::ExecutionMode::KernelDriver => {
            Ok(task_tools::Continuation::fresh(context.prompt))
        }
        task_tools::ExecutionMode::External => backend.continuation(context),
    }
}

/// Inputs prepared for one background process before any task in the batch is spawned.
struct BackgroundDispatchSpec {
    pid: u64,
    proc_question: String,
    history_path: PathBuf,
    agent_override: Option<String>,
    model_override: Option<String>,
    result_channel_id: Option<u64>,
    completion_futex_addr: Option<aios_kernel::primitives::FutexAddr>,
    task_id: Option<String>,
    response_schema: Option<serde_json::Value>,
    auto_model_fallback: Option<crate::ai::models::AutoModelFallbackSpec>,
    spawn_depth: usize,
    parent_agent_team_mode: bool,
    is_resume_wakeup: bool,
    initialize_history: bool,
    backend_id: task_tools::BackendId,
    backend_model: Option<String>,
    backend_continue_from: Option<String>,
    active_skill_names: Vec<String>,
    backend_reasoning_effort: Option<String>,
    image_files: Vec<String>,
}

impl BackgroundDispatchSpec {
    fn from_task_goal(
        pid: u64,
        proc_question: String,
        history_path: PathBuf,
        task_goal: Option<&task_tools::OsTaskGoal>,
        is_resume_wakeup: bool,
        initialize_history: bool,
        backend_reasoning_effort: &Option<String>,
    ) -> Self {
        Self {
            pid,
            proc_question,
            history_path,
            agent_override: task_goal.map(|goal| goal.agent_name.clone()),
            model_override: task_goal.map(|goal| goal.model.clone()),
            result_channel_id: task_goal.map(|goal| goal.result_channel_id),
            completion_futex_addr: task_goal
                .map(|goal| aios_kernel::primitives::FutexAddr(goal.completion_futex_addr)),
            task_id: task_goal.map(|goal| goal.task_id.clone()),
            response_schema: task_goal.and_then(|goal| goal.response_schema.clone()),
            auto_model_fallback: task_goal.and_then(|goal| goal.auto_model_fallback),
            spawn_depth: task_goal.map(|goal| goal.spawn_depth).unwrap_or(0),
            parent_agent_team_mode: task_goal.is_some_and(|goal| goal.parent_agent_team_mode),
            is_resume_wakeup,
            initialize_history,
            backend_id: task_tools::BackendId::from_stored(
                task_goal.and_then(|goal| goal.backend.as_deref()),
            ),
            backend_model: task_goal.and_then(|goal| goal.backend_model.clone()),
            backend_continue_from: task_goal.and_then(|goal| goal.backend_continue_from.clone()),
            active_skill_names: task_goal
                .map(|goal| goal.active_skill_names.clone())
                .unwrap_or_default(),
            backend_reasoning_effort: backend_reasoning_effort.clone(),
            image_files: task_goal
                .map(|goal| goal.image_files.clone())
                .unwrap_or_default(),
        }
    }
}

fn select_background_batch_with_log(
    shared_os: &aios_kernel::kernel::SharedKernel,
    epoch: u64,
    emit_decision: impl FnOnce(SchedulerDecision),
) -> Vec<aios_kernel::kernel::Process> {
    let (background_procs, decision) = {
        let mut os = shared_os.lock().unwrap();
        select_background_batch(os.as_mut(), epoch)
    };
    if let Some(decision) = decision {
        emit_decision(decision);
    }
    background_procs
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
    let background_procs = select_background_batch_with_log(&app.os, epoch, |decision| {
        decision.emit(app.session_id.as_str());
    });
    maybe_emit_scheduler_eval(epoch, app.session_id.as_str());

    if background_procs.is_empty() {
        return;
    }

    ensure_runtime_manifests_loaded(app, skill_manifests, agent_manifests, manifests_loaded);

    let original_history_file = app.session_history_file.clone();

    let mut task_specs: Vec<BackgroundDispatchSpec> = Vec::new();
    let backend_reasoning_effort =
        crate::ai::request::resolve_reasoning_effort(app, &app.current_model)
            .map(|effort| effort.as_str().to_string());
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
                publish_background_task_failure(
                    &app.os,
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
        task_specs.push(BackgroundDispatchSpec::from_task_goal(
            pid,
            proc_question,
            history_path,
            task_goal.as_ref(),
            is_resume_wakeup,
            initialize_history,
            &backend_reasoning_effort,
        ));
    }

    for BackgroundDispatchSpec {
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
        parent_agent_team_mode,
        is_resume_wakeup,
        initialize_history,
        backend_id,
        backend_model,
        backend_continue_from,
        active_skill_names,
        backend_reasoning_effort,
        image_files,
    } in task_specs
    {
        let backend = backend_id.resolve();
        let execution_mode = backend.execution_mode();
        // Preparation froze the worker model; attribution maps only the live
        // parent's effort here. Result wording stays in the task record.
        let identity = backend.attribution(&task_tools::ModelContext {
            kernel_model: model_override.as_deref().unwrap_or(&app.current_model),
            backend_model: backend_model.as_deref(),
            session_worker_model: None,
            kernel_reasoning_effort: None,
            backend_reasoning_effort: backend_reasoning_effort.as_deref(),
            selection_explanation: "",
        });
        let backend_model = if backend.descriptor().supports_worker_model {
            Some(identity.model)
        } else {
            backend_model
        };
        let backend_reasoning_effort = identity.reasoning_effort;
        if execution_mode == task_tools::ExecutionMode::External
            && let Some(task_id) = task_id.as_deref()
        {
            crate::ai::tools::task_tools::set_task_reasoning_effort(
                task_id,
                backend_reasoning_effort.clone(),
            );
        }
        let mut task_app = app.fork_for_subagent();
        // A resumed task keeps its original images in its first message; the
        // foreground may have moved on to a different image. Only the initial
        // dispatch attaches this task's own files, so a later wakeup cannot
        // silently swap in the live turn's attachments.
        task_app.attached_image_files = if initialize_history {
            image_files.clone()
        } else {
            Vec::new()
        };
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
                publish_background_task_failure(
                    &app.os,
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
                publish_background_task_failure(
                    &app.os,
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
        let phase_slot_for_backend = phase_slot.clone();
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
        // Explicit inheritance: snapshot the parent's effective cwd now and
        // carry it into the child via `SUBAGENT_CWD`, instead of relying on
        // the ambient process cwd being right at every tool call inside the
        // subagent. Resolution-only: never registered with the history guard,
        // so the guard cannot delete the parent's working directory.
        let inherited_cwd_dir = if inherit.cwd {
            runtime_ctx::effective_cwd().ok()
        } else {
            None
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

        // External inputs are frozen before the driver's callback owns the app.
        let backend_cwd = resolve_external_subagent_cwd(
            inherit.cwd,
            inherited_cwd_dir.as_ref(),
            scoped_cwd_dir.as_ref(),
        );
        let backend_cancel = task_app.cancel_stream.clone();
        let backend_workspace_allowed = inherit.cwd;
        let backend_parent_history = original_history_file.clone();

        let backend_session_slot: task_tools::backend_session::SessionSlot =
            Arc::new(std::sync::Mutex::new(None));
        let default_progress = backend.descriptor().text.default_progress;
        let inner_fut = TASK_PID.scope(Some(pid), async move {
            crate::ai::tools::registry::common::clear_tool_cancel();
            // External resolution reads deferred notes without consuming them;
            // the backend commits the queues only after a successful process spawn.
            let continuation = prepare_background_subagent_continuation(
                backend.as_ref(),
                &task_tools::ContinueContext {
                    continue_from: backend_continue_from.as_deref(),
                    parent_history: &backend_parent_history,
                    prompt: &proc_question,
                },
            );
            let events = task_tools::Events {
                phase_slot: Some(phase_slot_for_backend),
                session_slot: Some(backend_session_slot.clone()),
            };
            let memory_merge_error_slot = Arc::new(std::sync::Mutex::new(None::<String>));
            let memory_merge_error_for_run = memory_merge_error_slot.clone();
            let events_for_supervision = events.clone();
            let (outcome, worker_panicked) = supervise_background_worker(
                async {
                    match continuation {
                Err(error) => task_tools::Outcome {
                    execution_error: Some(error),
                    progress: default_progress.to_string(),
                    ..task_tools::Outcome::default()
                },
                Ok(continuation) => {
                    if backend.descriptor().supports_resume
                        && let Some(task_id) = task_id.as_deref()
                    {
                        task_tools::backend_session::track_session_slot(
                            backend_id,
                            task_id,
                            backend_session_slot.clone(),
                        );
                    }
                    let (resume_session, prompt, side_note_files) = continuation.into_parts();
                    let context = match execution_mode {
                        task_tools::ExecutionMode::External => {
                            task_tools::RunContext::External(
                                ExternalSubagentLaunch {
                                    prompt,
                                    cwd: backend_cwd,
                                    workspace_allowed: backend_workspace_allowed,
                                    model: backend_model,
                                    skill_manifests: task_skills.as_slice(),
                                    active_skill_names: &active_skill_names,
                                    resume_session,
                                    side_note_files,
                                    reasoning_effort: backend_reasoning_effort,
                                    image_files: image_files.clone(),
                                    phase_slot: events.phase_slot.clone(),
                                }
                                .into_spec(),
                            )
                        }
                        task_tools::ExecutionMode::KernelDriver => {
                            task_tools::RunContext::kernel(move |events, _cancel| {
                                Box::pin(async move {
                                    let run = runtime_ctx::IS_RESUME_TURN.scope(
                                        is_resume_wakeup,
                                        turn_runtime::run_turn(
                                            &mut task_app,
                                            &task_mcp,
                                            &task_skills,
                                            usize::MAX,
                                            prompt,
                                            String::new(),
                                            next_model,
                                            None,
                                            None,
                                            false,
                                            false,
                                        ),
                                    );
                                    let run = turn_runtime::SUBAGENT_PARENT_AGENT_TEAM_MODE
                                        .scope(parent_agent_team_mode, run);
                                    let result = if let Some(spec) = auto_model_fallback {
                                        runtime_ctx::AUTO_MODEL_FALLBACK.scope(spec, run).await
                                    } else {
                                        run.await
                                    }
                                    .map(|_| ())
                                    .map_err(|error| format!("{error}"));
                                    let captured = if result_channel_id.is_some() {
                                        result_slot_for_payload.lock().await.clone()
                                    } else {
                                        runtime_ctx::SubagentResult::default()
                                    };
                                    let progress = events
                                        .phase_slot
                                        .as_ref()
                                        .and_then(runtime_ctx::subagent_progress_snapshot)
                                        .unwrap_or_else(|| default_progress.to_string());
                                    let progress_snapshot = events.phase_slot.as_ref().and_then(
                                        runtime_ctx::subagent_progress_state_snapshot,
                                    );
                                    let memory_merge_error = private_memory_for_merge.and_then(
                                        |private_memory| {
                                            crate::ai::tools::service::memory::merge_subagent_whitelist(
                                                &private_memory,
                                                &main_memory_for_merge,
                                            )
                                            .err()
                                        },
                                    );
                                    if memory_merge_error.is_some() {
                                        memory_merge_failed_for_inner.store(true, Ordering::Release);
                                    }
                                    *memory_merge_error_for_run
                                        .lock()
                                        .unwrap_or_else(|poison| poison.into_inner()) = memory_merge_error;
                                    task_tools::Outcome {
                                        execution_error: result.err(),
                                        parent_payload: captured.parent_payload,
                                        validation_body: captured.final_assistant_text,
                                        progress,
                                        progress_snapshot,
                                        ..task_tools::Outcome::default()
                                    }
                                })
                            })
                        }
                    };
                    backend.run(context, events, backend_cancel).await
                }
            }
                },
                &events_for_supervision,
                default_progress,
            )
            .await;
            // Keep the existing external failure payload and default progress;
            // sessions are independently promoted from the captured slot.
            let mut outcome = outcome;
            if outcome.execution_error.is_some()
                && execution_mode == task_tools::ExecutionMode::External
            {
                outcome.parent_payload.clear();
                outcome.validation_body.clear();
                outcome.progress = default_progress.to_string();
            }
            if let Some(session) = &outcome.backend_session {
                *backend_session_slot
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner()) = Some(session.clone());
            }
            let memory_merge_error = memory_merge_error_slot
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .take();
            let mut outcome = resolve_background_task_outcome(
                outcome,
                memory_merge_error,
                response_schema.as_ref(),
            );
            outcome.retain_recovery |= worker_panicked;
            let target = BackgroundTurnTarget {
                pid,
                task_id,
                result_channel_id,
                completion_futex_addr,
                backend_id,
            };
            let retention = finalize_background_turn_shared(&task_os, &target, outcome);
            preserve_scoped_artifacts_for_inner.store(
                retention.preserve_history || retention.delivery_failed,
                Ordering::Release,
            );
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
        if let Some(dir) = inherited_cwd_dir {
            wrapped = Box::pin(runtime_ctx::SUBAGENT_CWD.scope(dir, wrapped));
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
