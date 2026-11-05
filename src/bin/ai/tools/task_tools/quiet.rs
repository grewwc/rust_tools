use super::*;
use crate::ai::driver::runtime_ctx::{self, SubagentProgressSnapshot};
use rust_tools::commonw::FastMap;

const TASK_WAIT_LINE_MAX_CHARS: usize = 300;
const CHECKPOINT_SUMMARY_MAX_CHARS: usize = 160;
const TASK_WAIT_DEADLINE_PREFIX: &str = "[task-wait] still running:";
type StatusCursor = FastMap<String, (String, String, u64, u64)>;

#[derive(Clone, Copy)]
pub(super) enum TaskWaitReturnReason {
    Parked,
    ResultsReady,
    BudgetElapsed,
    Interrupted,
    Woke,
}

impl TaskWaitReturnReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Parked => "parked",
            Self::ResultsReady => "results_ready",
            Self::BudgetElapsed => "budget_elapsed",
            Self::Interrupted => "interrupted",
            Self::Woke => "woke",
        }
    }
}

/// Consult the current agent's active skills, never the manifest catalog or session history.
pub(crate) fn is_agent_team_mode() -> bool {
    runtime_ctx::try_current().is_some_and(|context| {
        context.active_skill_names.read().is_ok_and(|names| {
            names
                .iter()
                .any(|name| name.eq_ignore_ascii_case("agent-team"))
        })
    })
}

/// Only quiet task-wait deadlines emit this marker; terminal events retain their existing path.
pub(crate) fn is_agent_team_wait_deadline_message(message: &str) -> bool {
    message.starts_with(TASK_WAIT_DEADLINE_PREFIX)
}

#[derive(Clone)]
struct TaskDigest {
    task_id: String,
    status: String,
    phase: String,
    idle_secs: u64,
    revision: u64,
    sequence: u64,
    checkpoint_summary: Option<String>,
}

impl TaskDigest {
    fn new(
        task_id: &str,
        status: String,
        progress: Option<&SubagentProgressSnapshot>,
        elapsed: Duration,
        pending_side_note: Option<String>,
    ) -> Self {
        let mut digest = Self::from_progress(task_id, status, progress, elapsed);
        if let Some(pending) = pending_side_note {
            if digest.phase == "-" {
                digest.phase = pending;
            } else {
                digest.phase.push_str(" · ");
                digest.phase.push_str(&pending);
            }
        }
        digest
    }

    fn from_progress(
        task_id: &str,
        status: String,
        progress: Option<&SubagentProgressSnapshot>,
        elapsed: Duration,
    ) -> Self {
        Self {
            task_id: task_id.to_string(),
            status,
            phase: progress.map_or_else(|| "-".to_string(), |p| compact_line(&p.phase, 80)),
            idle_secs: progress.map_or(elapsed, |p| p.stale_for).as_secs(),
            revision: progress.map_or(0, |p| p.checkpoint_revision),
            sequence: progress.map_or(0, |p| p.sequence),
            checkpoint_summary: progress
                .and_then(|p| p.checkpoint_summary.as_deref())
                .map(|summary| compact_line(summary, CHECKPOINT_SUMMARY_MAX_CHARS)),
        }
    }

    fn status_line(&self) -> String {
        let mut line = format!(
            "{} status={} phase={} idle={}s rev={}",
            self.task_id, self.status, self.phase, self.idle_secs, self.revision
        );
        if let Some(summary) = &self.checkpoint_summary
            && !summary.is_empty()
        {
            line.push_str(" checkpoint=");
            line.push_str(summary);
        }
        line
    }
}

fn compact_line(text: &str, max_chars: usize) -> String {
    let sanitized: String = text
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect();
    let normalized = sanitized.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = normalized.chars();
    let mut line: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() && max_chars > 0 {
        line.pop();
        line.push('…');
    }
    line
}

fn format_status_digest(tasks: &[TaskDigest], cursor: &mut StatusCursor) -> String {
    let mut running = 0;
    let mut done = 0;
    let mut failed = 0;
    for task in tasks {
        match TaskTerminalStatus::parse(&task.status) {
            Some(TaskTerminalStatus::Completed) => done += 1,
            Some(_) => failed += 1,
            None => running += 1,
        }
    }
    let mut lines = vec![format!(
        "[task-status] running={running} done={done} failed={failed}"
    )];
    cursor.retain(|task_id, _| tasks.iter().any(|task| &task.task_id == task_id));
    for task in tasks {
        let revision = (
            task.status.clone(),
            task.phase.clone(),
            task.revision,
            task.sequence,
        );
        if cursor.get(&task.task_id) != Some(&revision) {
            lines.push(task.status_line());
        }
        cursor.insert(task.task_id.clone(), revision);
    }
    lines.join("\n")
}

pub(super) fn execute_agent_team_task_status(
    tracked: &[(String, AsyncTaskEntry)],
) -> Result<String, String> {
    let mut tasks = Vec::new();
    let mut completed_outputs = Vec::new();
    let mut finished_ids = Vec::new();
    let kernel_result = with_os_kernel(|os| {
        for (task_id, entry) in tracked {
            let state = task_state_string(os, entry.result_channel_id, entry.pid)?;
            // Snapshot before collection releases the progress slot. This read does not consume
            // checkpoint excerpts, which remain available through the existing evidence path.
            let progress = task_progress_slot(task_id)
                .and_then(|slot| runtime_ctx::subagent_progress_state_snapshot(&slot));
            let mut task = TaskDigest::new(
                task_id,
                state,
                progress.as_ref(),
                entry.started_at.elapsed(),
                pending_task_side_note_progress(task_id),
            );
            if let Some(collected) = drain_task_result(
                os,
                task_id,
                entry,
                "subagent result collected by task_status",
            )? {
                task.status = collected.result.status;
                completed_outputs.push(collected.rendered);
                finished_ids.push(task_id.clone());
            }
            tasks.push(task);
        }
        Ok(())
    });
    for task_id in &finished_ids {
        remove_task_entry(task_id);
    }
    if let Err(mut error) = kernel_result {
        if !completed_outputs.is_empty() {
            error.push_str("\n\n[Task results collected before the error]\n");
            error.push_str(&completed_outputs.join("\n\n---\n\n"));
        }
        return Err(error);
    }

    // Advance the cursor only after every fallible collection succeeds. Terminal result bodies
    // always bypass the digest/cursor and retain the existing durable collection format.
    let mut output = if let Some(context) = runtime_ctx::try_current() {
        format_status_digest(
            &tasks,
            &mut lock_unpoisoned(&context.agent_team_task_status_cursor),
        )
    } else {
        format_status_digest(&tasks, &mut StatusCursor::default())
    };
    if !completed_outputs.is_empty() {
        output.push_str("\n\n");
        output.push_str(&completed_outputs.join("\n\n---\n\n"));
    }
    Ok(output)
}

fn format_wait_digest(
    tasks: &[TaskDigest],
    reason: TaskWaitReturnReason,
    waited: Duration,
) -> String {
    let header = format!(
        "{TASK_WAIT_DEADLINE_PREFIX} reason={} waited={}s",
        reason.as_str(),
        waited.as_secs()
    );
    let pending = tasks
        .iter()
        .map(|task| {
            format!(
                "{}(idle={}s,phase={})",
                task.task_id, task.idle_secs, task.phase
            )
        })
        .collect::<Vec<_>>();
    let full = format!("{header} {}", pending.join(" "));
    if full.chars().count() <= TASK_WAIT_LINE_MAX_CHARS {
        return full;
    }

    let omission_hint = |omitted| {
        format!(" (+{omitted} omitted; task_status for IDs, task_evidence_read for detail)")
    };
    // Reserve the return metadata and recovery hint before adding any whole task rows.
    let mut output = header;
    let mut shown = 0;
    for task in &pending {
        let remaining = pending.len() - shown - 1;
        let hint_len = if remaining > 0 {
            omission_hint(remaining).chars().count()
        } else {
            0
        };
        if output.chars().count() + 1 + task.chars().count() + hint_len > TASK_WAIT_LINE_MAX_CHARS {
            break;
        }
        output.push(' ');
        output.push_str(task);
        shown += 1;
    }
    if shown < pending.len() {
        output.push_str(&omission_hint(pending.len() - shown));
    }
    output
}

pub(super) fn format_agent_team_task_wait(
    task_ids: &[String],
    reason: TaskWaitReturnReason,
    waited: Duration,
) -> String {
    let tasks = task_ids
        .iter()
        .filter_map(|task_id| {
            // Read progress and side notes before the registry callback: side-note lookup
            // re-enters the task registry, whose mutex is not reentrant.
            let progress = task_progress_slot(task_id)
                .and_then(|slot| runtime_ctx::subagent_progress_state_snapshot(&slot));
            let pending_side_note = pending_task_side_note_progress(task_id);
            with_task_entry(task_id, |entry| {
                TaskDigest::new(
                    task_id,
                    "running".to_string(),
                    progress.as_ref(),
                    entry.started_at.elapsed(),
                    pending_side_note,
                )
            })
        })
        .collect::<Vec<_>>();
    format_wait_digest(&tasks, reason, waited)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(task_id: &str, status: &str, revision: u64) -> TaskDigest {
        TaskDigest {
            task_id: task_id.to_string(),
            status: status.to_string(),
            phase: "read".to_string(),
            idle_secs: 12,
            revision,
            sequence: revision,
            checkpoint_summary: Some("saved finding".to_string()),
        }
    }

    fn waited_secs(output: &str) -> u64 {
        output
            .split_whitespace()
            .find_map(|token| token.strip_prefix("waited="))
            .and_then(|value| value.strip_suffix('s'))
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| panic!("missing wait duration: {output}"))
    }

    #[test]
    fn agent_team_quiet_wait_reason_and_elapsed_preserve_task_idle() {
        for (reason, token) in [
            (TaskWaitReturnReason::Parked, "parked"),
            (TaskWaitReturnReason::BudgetElapsed, "budget_elapsed"),
            (TaskWaitReturnReason::ResultsReady, "results_ready"),
            (TaskWaitReturnReason::Interrupted, "interrupted"),
            (TaskWaitReturnReason::Woke, "woke"),
        ] {
            let output = format_wait_digest(
                &[task("task-a", "running", 0)],
                reason,
                Duration::from_secs(17),
            );
            assert_eq!(
                output,
                format!(
                    "[task-wait] still running: reason={token} waited=17s task-a(idle=12s,phase=read)"
                )
            );
        }
    }

    #[test]
    fn agent_team_quiet_status_empty_registry() {
        let mut cursor = StatusCursor::default();
        cursor.insert(
            "old".to_string(),
            ("running".to_string(), "read".to_string(), 1, 1),
        );
        assert_eq!(
            format_status_digest(&[], &mut cursor),
            "[task-status] running=0 done=0 failed=0"
        );
        assert!(cursor.is_empty());
    }

    #[test]
    fn agent_team_quiet_status_changed_and_unchanged_revisions() {
        let mut cursor = StatusCursor::default();
        let mut tasks = vec![
            task("task-a", "running", 1),
            task("task-b", "completed", 2),
            task("task-c", "failed", 3),
        ];
        let first = format_status_digest(&tasks, &mut cursor);
        assert!(first.starts_with("[task-status] running=1 done=1 failed=1\n"));
        assert!(
            first.contains(
                "task-a status=running phase=read idle=12s rev=1 checkpoint=saved finding"
            )
        );
        assert_eq!(first.lines().count(), 4);
        tasks[0].idle_secs = 40;
        assert_eq!(format_status_digest(&tasks, &mut cursor).lines().count(), 1);
        tasks[0].revision += 1;
        let changed = format_status_digest(&tasks, &mut cursor);
        assert_eq!(changed.lines().count(), 2);
        assert!(changed.contains("task-a status=running phase=read idle=40s rev=2"));
        tasks[0].sequence += 1;
        assert_eq!(format_status_digest(&tasks, &mut cursor).lines().count(), 2);
        tasks[0].status = "completed".to_string();
        assert!(format_status_digest(&tasks, &mut cursor).contains("task-a status=completed"));
    }

    #[test]
    fn agent_team_quiet_status_and_timeout_truncate_unicode_to_one_line() {
        let progress = SubagentProgressSnapshot {
            phase: "read\nfiles".to_string(),
            checkpoint_summary: Some("界".repeat(161)),
            checkpoint_excerpt: Some("full checkpoint must not appear".to_string()),
            checkpoint_revision: 1,
            elapsed: Duration::from_secs(20),
            stale_for: Duration::from_secs(12),
            checkpoint_due: false,
            sequence: 1,
            timeline: Vec::new(),
        };
        let row = TaskDigest::from_progress(
            "task-a",
            "running".to_string(),
            Some(&progress),
            Duration::ZERO,
        );
        assert_eq!(
            row.checkpoint_summary.as_ref().unwrap().chars().count(),
            160
        );
        assert!(row.checkpoint_summary.as_ref().unwrap().ends_with('…'));
        assert_eq!(row.status_line().lines().count(), 1);
        assert!(!row.status_line().contains("full checkpoint"));
        let timeout = format_wait_digest(
            &vec![row; 20],
            TaskWaitReturnReason::BudgetElapsed,
            Duration::from_secs(30),
        );
        assert!(timeout.chars().count() <= TASK_WAIT_LINE_MAX_CHARS);
        assert_eq!(timeout.lines().count(), 1);
        assert!(timeout.contains("omitted"));
        assert!(timeout.contains("task_status"));
        assert!(timeout.contains("task_evidence_read"));
        assert!(timeout.contains("reason=budget_elapsed waited=30s"));
        assert!(timeout.contains("idle=12s,phase=read files"));
        assert!(!timeout.contains("TASK_WAIT_TIMEOUT"));
        assert!(!timeout.contains("goal"));
        assert!(!timeout.contains("task_wait"));
    }

    #[test]
    fn agent_team_quiet_wait_truncates_at_whole_task_ids_with_recovery() {
        let tasks = (0..8)
            .map(|index| task(&format!("task-{index}-{}", "x".repeat(48)), "running", 0))
            .collect::<Vec<_>>();
        for (reason, token) in [
            (TaskWaitReturnReason::Parked, "parked"),
            (TaskWaitReturnReason::BudgetElapsed, "budget_elapsed"),
            (TaskWaitReturnReason::ResultsReady, "results_ready"),
            (TaskWaitReturnReason::Interrupted, "interrupted"),
            (TaskWaitReturnReason::Woke, "woke"),
        ] {
            let waited = Duration::MAX;
            let header = format!(
                "{TASK_WAIT_DEADLINE_PREFIX} reason={token} waited={}s",
                waited.as_secs()
            );
            let output = format_wait_digest(&tasks, reason, waited);
            assert!(output.chars().count() <= TASK_WAIT_LINE_MAX_CHARS);
            assert_eq!(output.lines().count(), 1);
            let (shown, hint) = output
                .strip_prefix(&header)
                .unwrap()
                .split_once(" (+")
                .unwrap();
            let shown_ids = shown
                .split_whitespace()
                .map(|row| row.split_once("(idle=").unwrap().0)
                .collect::<Vec<_>>();
            assert!(!shown_ids.is_empty());
            assert!(shown_ids.len() < tasks.len());
            for (id, task) in shown_ids.iter().zip(&tasks) {
                assert_eq!(*id, task.task_id);
            }
            assert!(hint.starts_with(&format!("{} omitted;", tasks.len() - shown_ids.len())));
            assert!(hint.contains("task_status"));
            assert!(hint.contains("task_evidence_read"));

            let oversized = task(&format!("task-{}", "x".repeat(300)), "running", 0);
            let output = format_wait_digest(&[oversized], reason, waited);
            assert!(output.chars().count() <= TASK_WAIT_LINE_MAX_CHARS);
            assert!(output.starts_with(&format!("{header} (+1 omitted;")));
            assert!(output.contains("task_status"));
            assert!(output.contains("task_evidence_read"));
            assert!(!output.contains("task-xxx"));
        }
    }

    #[test]
    fn agent_team_quiet_mode_defaults_false_and_reads_only_active_skills() {
        let _env_guard = lock_unpoisoned(&crate::ai::test_support::ENV_LOCK);
        assert!(!is_agent_team_mode());
        let context = runtime_ctx::DriverContext::new(
            super::super::tests::test_app_with_model("test-model".to_string()),
            Arc::new(Mutex::new(crate::ai::mcp::McpClient::new())),
            Arc::new(Vec::new()),
            Arc::new(Vec::new()),
        );
        runtime_ctx::DRIVER_CTX.sync_scope(context.clone(), || {
            assert!(!is_agent_team_mode());
            runtime_ctx::set_active_skill_names(vec!["AGENT-TEAM".to_string()]);
            assert!(is_agent_team_mode());
            runtime_ctx::set_active_skill_names(vec!["other".to_string()]);
            assert!(!is_agent_team_mode());
        });
        context
            .active_skill_names
            .write()
            .unwrap()
            .push("agent-team".to_string());
        let _fallback = runtime_ctx::DriverCtxThreadFallback::install(Some(context));
        assert!(is_agent_team_mode());
    }

    struct RuntimeFixture {
        context: Arc<runtime_ctx::DriverContext>,
        owner_pid: u64,
        root: PathBuf,
        task_ids: Vec<String>,
    }

    impl RuntimeFixture {
        fn new() -> Self {
            let mut app = super::super::tests::test_app_with_model("test-model".to_string());
            app.session_id = format!("agent-team-quiet-{}", Uuid::new_v4().simple());
            let root = runtime_ctx::temp_dir().unwrap().join(&app.session_id);
            app.config.history_file = root.join("history.sqlite");
            app.session_history_file = app.config.history_file.clone();
            let owner_pid = app.os.lock().unwrap().begin_foreground(
                "owner".to_string(),
                "original goal must not repeat".to_string(),
                10,
                8,
                None,
            );
            let context = runtime_ctx::DriverContext::new(
                app,
                Arc::new(Mutex::new(crate::ai::mcp::McpClient::new())),
                Arc::new(Vec::new()),
                Arc::new(Vec::new()),
            );
            context
                .active_skill_names
                .write()
                .unwrap()
                .push("agent-team".to_string());
            Self {
                context,
                owner_pid,
                root,
                task_ids: Vec::new(),
            }
        }

        fn spawn(&mut self) -> String {
            let task_id = format!("task-{}", Uuid::new_v4().simple());
            let (pid, channel, futex) = {
                let mut os = self.context.app_proto.os.lock().unwrap();
                let pid = os
                    .spawn(
                        Some(self.owner_pid),
                        "child".to_string(),
                        "child goal".to_string(),
                        20,
                        4,
                        None,
                        None,
                    )
                    .unwrap();
                let channel = os.channel_create(Some(pid), 1, "quiet-result".to_string());
                let futex = os.futex_create(0, "quiet-complete".to_string());
                (pid, channel.raw(), futex)
            };
            insert_task_entry_for_test(
                task_id.clone(),
                AsyncTaskEntry {
                    session_id: self.context.app_proto.session_id.clone(),
                    owner_pid: self.owner_pid,
                    pid,
                    result_channel_id: channel,
                    completion_futex_addr: futex,
                    description: "quiet test".to_string(),
                    agent_name: "build".to_string(),
                    backend: None,
                    model: "test-model".to_string(),
                    is_model_auto_selected: false,
                    auto_model_fallback: None,
                    selection_explanation: "test selection".to_string(),
                    inherit: InheritOptions::default(),
                    abort_handle: None,
                    cancel_stream: Arc::new(AtomicBool::new(false)),
                    started_at: Instant::now(),
                    last_progress_notification_at: None,
                    last_progress_persisted_at: None,
                },
            );
            lock_unpoisoned(&TASK_PROGRESS_REGISTRY)
                .insert(task_id.clone(), runtime_ctx::new_subagent_progress_slot());
            self.task_ids.push(task_id.clone());
            task_id
        }

        fn scope<T>(&self, action: impl FnOnce() -> T) -> T {
            self.context
                .app_proto
                .os
                .lock()
                .unwrap()
                .set_current_pid(Some(self.owner_pid));
            runtime_ctx::TURN_IDENTITY
                .sync_scope((self.context.app_proto.session_id.clone(), 0), || {
                    runtime_ctx::DRIVER_CTX.sync_scope(self.context.clone(), action)
                })
        }

        fn scope_with_global_os<T>(&self, action: impl FnOnce() -> T) -> T {
            struct RestoreGlobalOs(Option<SharedKernel>);

            impl Drop for RestoreGlobalOs {
                fn drop(&mut self) {
                    *lock_unpoisoned(&GLOBAL_OS) = self.0.take();
                }
            }

            let _restore = RestoreGlobalOs(
                lock_unpoisoned(&GLOBAL_OS).replace(self.context.app_proto.os.clone()),
            );
            self.scope(action)
        }

        fn publish(&self, task_id: &str, status: &str, output: &str, error: Option<&str>) {
            let entry = with_task_entry(task_id, Clone::clone).unwrap();
            self.context
                .app_proto
                .os
                .lock()
                .unwrap()
                .channel_send(
                    Some(entry.pid),
                    ChannelId(entry.result_channel_id),
                    serde_json::json!({
                        "status": status, "output": output, "error": error,
                    })
                    .to_string(),
                )
                .unwrap();
        }
    }

    impl Drop for RuntimeFixture {
        fn drop(&mut self) {
            for task_id in &self.task_ids {
                remove_task_entry(task_id);
                lock_unpoisoned(&TASK_PROGRESS_REGISTRY).remove(task_id);
            }
            lock_unpoisoned(&TASK_WAIT_STATES)
                .retain(|key, _| key.session_id != self.context.app_proto.session_id);
            lock_unpoisoned(&TASK_WAIT_NOOP_COUNTS)
                .retain(|key, _| key.session_id != self.context.app_proto.session_id);
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn agent_team_quiet_wait_formats_pending_side_note_without_registry_recursion() {
        let _env_guard = lock_unpoisoned(&crate::ai::test_support::ENV_LOCK);
        let mut fixture = RuntimeFixture::new();
        let task_id = fixture.spawn();
        let mut entry = with_task_entry(&task_id, Clone::clone).unwrap();
        entry.backend = Some(SubagentBackend::TraeCli);
        insert_task_entry_for_test(task_id.clone(), entry);
        let history_file = &fixture.context.app_proto.session_history_file;
        crate::ai::driver::side_note::push_side_note(
            history_file,
            "Keep the parser behavior.",
            "lead",
            Some(&task_id),
        )
        .unwrap();

        let output = fixture.scope(|| {
            format_agent_team_task_wait(
                &[task_id.clone()],
                TaskWaitReturnReason::Parked,
                Duration::ZERO,
            )
        });
        assert!(output.starts_with(&format!(
            "{TASK_WAIT_DEADLINE_PREFIX} reason=parked waited=0s {task_id}(idle="
        )));
        assert!(
            output.contains(
                "phase=starting · pending side-notes: 1 (queued for backend_continue_from)"
            )
        );
        assert_eq!(output.lines().count(), 1);
        assert!(output.chars().count() <= TASK_WAIT_LINE_MAX_CHARS);
        assert!(!output.contains("Keep the parser behavior."));
        assert_eq!(
            crate::ai::driver::side_note::pending_side_note_count(history_file, Some(&task_id)),
            1
        );
    }

    #[test]
    fn agent_team_quiet_wait_non_suspending_return_reports_interrupted_or_woke() {
        use crate::ai::tools::registry::common;

        let _env_guard = lock_unpoisoned(&crate::ai::test_support::ENV_LOCK);
        for (agent_team_mode, cancelled, reason, label) in [
            (true, true, "interrupted", "INTERRUPTED"),
            (false, true, "interrupted", "INTERRUPTED"),
            (true, false, "woke", "WOKE"),
            (false, false, "woke", "WOKE"),
        ] {
            let mut fixture = RuntimeFixture::new();
            if !agent_team_mode {
                fixture.context.active_skill_names.write().unwrap().clear();
            }
            let task_id = fixture.spawn();
            let entry = with_task_entry(&task_id, Clone::clone).unwrap();
            let args = serde_json::json!({
                "task_ids": [task_id.clone()], "timeout_secs": 60, "wait_policy": "all"
            });
            let key = task_wait_key(
                &fixture.context.app_proto.session_id,
                fixture.owner_pid,
                &WaitPolicy::All,
                &[task_id.clone()],
            );
            let before_wait = Instant::now();
            let output = fixture
                .scope_with_global_os(|| {
                    common::clear_tool_cancel();
                    let state = load_or_create_task_wait_state(&key, 60, agent_team_mode);
                    assert!(!state.expired);
                    if cancelled {
                        assert!(common::try_request_tool_cancel());
                        assert!(common::is_tool_cancel_requested());
                    } else {
                        // A completion source may be ready before its terminal payload is available.
                        with_os_kernel(|os| {
                            let _ = os.futex_store(entry.completion_futex_addr, 1);
                            Ok(())
                        })
                        .unwrap();
                        assert!(!common::is_tool_cancel_requested());
                    }
                    let result = execute_task_wait(&args);
                    assert!(Instant::now() < state.deadline);
                    with_os_kernel(|os| {
                        assert_eq!(os.current_process_id(), Some(fixture.owner_pid));
                        assert!(is_task_pending(os, entry.pid)?);
                        assert!(
                            os.channel_meta(ChannelId(entry.result_channel_id))
                                .is_some()
                        );
                        Ok(())
                    })
                    .unwrap();
                    common::clear_tool_cancel();
                    result
                })
                .unwrap();
            let after_wait = Instant::now();
            assert!(!output.contains("reason=budget_elapsed"), "{output}");
            assert!(!output.contains("[task_wait BUDGET ELAPSED]"), "{output}");
            assert!(output.contains(&task_id));
            assert!(with_task_entry(&task_id, |_| ()).is_some());
            if agent_team_mode {
                assert!(output.starts_with(TASK_WAIT_DEADLINE_PREFIX));
                assert!(output.contains(&format!("reason={reason}")), "{output}");
                assert!(
                    waited_secs(&output)
                        <= after_wait.saturating_duration_since(before_wait).as_secs()
                );
                assert!(output.contains("(idle="));
                assert!(output.contains("phase=starting"));
                assert!(output.chars().count() <= TASK_WAIT_LINE_MAX_CHARS);
            } else {
                assert!(
                    output.starts_with(&format!("[task_wait {label}]")),
                    "{output}"
                );
                assert!(output.contains("wait budget has not elapsed"), "{output}");
            }
        }
    }

    #[test]
    fn agent_team_quiet_wait_honors_requested_timeout_and_wakes_with_one_line() {
        let _env_guard = lock_unpoisoned(&crate::ai::test_support::ENV_LOCK);
        let mut fixture = RuntimeFixture::new();
        let task_id = fixture.spawn();
        let args = serde_json::json!({
            "task_ids": [task_id.clone()], "timeout_secs": 300, "wait_policy": "all"
        });
        record_last_spawn_batch(vec![task_id.clone()]);
        let before_wait = Instant::now();
        let parked = fixture.scope(|| execute_task_wait(&args)).unwrap();
        let after_wait = Instant::now();
        assert!(is_agent_team_wait_deadline_message(&parked));
        assert!(parked.contains("reason=parked"));
        assert!(
            waited_secs(&parked) <= after_wait.saturating_duration_since(before_wait).as_secs()
        );
        assert!(parked.contains("(idle="));
        assert_eq!(parked.lines().count(), 1);
        assert!(parked.chars().count() <= TASK_WAIT_LINE_MAX_CHARS);
        let key = task_wait_key(
            &fixture.context.app_proto.session_id,
            fixture.owner_pid,
            &WaitPolicy::All,
            &[task_id.clone()],
        );
        {
            let mut states = lock_unpoisoned(&TASK_WAIT_STATES);
            let state = states.get_mut(&key).unwrap();
            assert!(state.agent_team_mode);
            assert_eq!(state.timeout_secs, 300);
            assert!(state.deadline >= before_wait + Duration::from_secs(300));
            assert!(state.deadline <= after_wait + Duration::from_secs(300));
            state.deadline = Instant::now() - Duration::from_secs(1);
        }
        fixture.scope(wake_expired_task_waits);
        {
            let os = fixture.context.app_proto.os.lock().unwrap();
            let owner = os.get_process(fixture.owner_pid).unwrap();
            assert!(matches!(owner.state, ProcessState::Ready));
            assert_eq!(owner.mailbox.len(), 1);
            let notice = owner.mailbox.front().unwrap();
            assert!(notice.contains("reason=budget_elapsed"));
            assert!(waited_secs(notice) >= 300);
            assert_eq!(notice.lines().count(), 1);
            assert!(notice.chars().count() <= TASK_WAIT_LINE_MAX_CHARS);
            assert!(!notice.contains("TASK_WAIT_TIMEOUT"));
            assert!(!notice.contains("goal"));
            assert!(!notice.contains("task_wait"));
        }
        let before_elapsed = Instant::now();
        let elapsed = fixture.scope(|| execute_task_wait(&args)).unwrap();
        let after_elapsed = Instant::now();
        assert!(is_agent_team_wait_deadline_message(&elapsed));
        assert!(elapsed.contains("reason=budget_elapsed"));
        assert!(
            waited_secs(&elapsed)
                <= after_elapsed
                    .saturating_duration_since(before_elapsed)
                    .as_secs()
        );
        assert!(elapsed.contains("(idle="));
        assert_eq!(elapsed.lines().count(), 1);
        assert!(with_task_entry(&task_id, |_| ()).is_some());
        reset_last_spawn_batch_for_test();
    }

    #[test]
    fn agent_team_quiet_wait_short_and_default_timeouts_are_not_extended() {
        let _env_guard = lock_unpoisoned(&crate::ai::test_support::ENV_LOCK);
        for (requested, expected) in [(Some(1), 1), (None, DEFAULT_TASK_WAIT_TIMEOUT_SECS)] {
            let mut fixture = RuntimeFixture::new();
            let task_id = fixture.spawn();
            let mut args = serde_json::json!({
                "task_ids": [task_id.clone()], "wait_policy": "all"
            });
            if let Some(timeout_secs) = requested {
                args["timeout_secs"] = serde_json::json!(timeout_secs);
            }
            let before_wait = Instant::now();
            let parked = fixture.scope(|| execute_task_wait(&args)).unwrap();
            let after_wait = Instant::now();
            let budget = Duration::from_secs(expected);
            assert!(after_wait.duration_since(before_wait) < budget);
            assert!(is_agent_team_wait_deadline_message(&parked));
            assert_eq!(parked.lines().count(), 1);
            let key = task_wait_key(
                &fixture.context.app_proto.session_id,
                fixture.owner_pid,
                &WaitPolicy::All,
                &[task_id],
            );
            let states = lock_unpoisoned(&TASK_WAIT_STATES);
            let state = states.get(&key).unwrap();
            assert!(state.agent_team_mode);
            assert_eq!(state.timeout_secs, expected);
            assert!(state.deadline >= before_wait + budget);
            assert!(state.deadline <= after_wait + budget);
        }
    }

    #[test]
    fn agent_team_quiet_wait_delivers_terminal_result_and_error_before_deadline() {
        let _env_guard = lock_unpoisoned(&crate::ai::test_support::ENV_LOCK);
        let mut fixture = RuntimeFixture::new();
        let done = fixture.spawn();
        let pending = fixture.spawn();
        let args = serde_json::json!({ "task_ids": [done.clone(), pending.clone()] });
        fixture.scope(|| execute_task_wait(&args)).unwrap();
        let output = format!("terminal output\n{}", "界".repeat(1000));
        let error = format!("terminal error {}", "x".repeat(1000));
        fixture.publish(&done, "failed", &output, Some(&error));
        assert!(matches!(
            fixture
                .context
                .app_proto
                .os
                .lock()
                .unwrap()
                .get_process(fixture.owner_pid)
                .unwrap()
                .state,
            ProcessState::Ready
        ));
        let collected = fixture.scope(|| execute_task_wait(&args)).unwrap();
        assert_eq!(collected.matches(&output).count(), 1);
        assert_eq!(collected.matches(&error).count(), 1);
        assert!(collected.contains("FAILED"));
        assert!(with_task_entry(&done, |_| ()).is_none());
        assert!(with_task_entry(&pending, |_| ()).is_some());
        let ledger = fixture
            .scope(|| {
                read_persisted_task_evidence_status(&fixture.context.app_proto.session_id, &done)
            })
            .unwrap()
            .unwrap();
        assert_eq!(ledger.0, "failed");
        assert_eq!(ledger.1, collected.split_once("\n\n---\n\n").unwrap().0);
    }

    #[test]
    fn agent_team_quiet_wait_all_returns_partial_terminal_results_without_parking() {
        let _env_guard = lock_unpoisoned(&crate::ai::test_support::ENV_LOCK);
        for (budget_elapsed, reason) in [(false, "results_ready"), (true, "budget_elapsed")] {
            let mut fixture = RuntimeFixture::new();
            let done = fixture.spawn();
            let pending = fixture.spawn();
            let args = serde_json::json!({ "task_ids": [done.clone(), pending.clone()], "wait_policy": "all" });
            fixture.scope(|| execute_task_wait(&args)).unwrap();
            if budget_elapsed {
                let key = task_wait_key(
                    &fixture.context.app_proto.session_id,
                    fixture.owner_pid,
                    &WaitPolicy::All,
                    &[done.clone(), pending.clone()],
                );
                lock_unpoisoned(&TASK_WAIT_STATES)
                    .get_mut(&key)
                    .unwrap()
                    .deadline = Instant::now() - Duration::from_secs(1);
            }
            let output = format!("complete output\n{}", "界".repeat(1000));
            fixture.publish(&done, "completed", &output, None);
            let before_wait = Instant::now();
            let collected = fixture.scope(|| execute_task_wait(&args)).unwrap();
            let after_wait = Instant::now();
            assert_eq!(collected.matches(&output).count(), 1);
            assert!(collected.contains("COMPLETED"));
            let (result, notice) = collected.rsplit_once("\n\n---\n\n").unwrap();
            assert!(result.contains(&output));
            assert!(
                collected.find(&output).unwrap()
                    < collected.find(TASK_WAIT_DEADLINE_PREFIX).unwrap()
            );
            assert!(notice.contains(&pending));
            assert!(is_agent_team_wait_deadline_message(notice));
            assert!(notice.contains(&format!("reason={reason}")));
            assert!(
                waited_secs(notice) <= after_wait.saturating_duration_since(before_wait).as_secs()
            );
            assert!(notice.contains("(idle="));
            assert!(notice.contains("phase=starting"));
            assert_eq!(notice.lines().count(), 1);
            assert!(notice.chars().count() <= TASK_WAIT_LINE_MAX_CHARS);
            assert!(matches!(
                fixture
                    .context
                    .app_proto
                    .os
                    .lock()
                    .unwrap()
                    .get_process(fixture.owner_pid)
                    .unwrap()
                    .state,
                ProcessState::Ready
            ));
        }
    }

    #[test]
    fn agent_team_quiet_status_collects_complete_and_failed_payloads_once() {
        let _env_guard = lock_unpoisoned(&crate::ai::test_support::ENV_LOCK);
        let mut fixture = RuntimeFixture::new();
        let done = fixture.spawn();
        let failed = fixture.spawn();
        let first = fixture
            .scope(|| execute_task_status(&serde_json::json!({})))
            .unwrap();
        assert_eq!(first.lines().count(), 3);
        assert_eq!(
            fixture
                .scope(|| execute_task_status(&serde_json::json!({})))
                .unwrap(),
            "[task-status] running=2 done=0 failed=0"
        );
        let output = format!("complete output {}", "界".repeat(1000));
        let error = format!("failed output {}", "x".repeat(1000));
        fixture.publish(&done, "completed", &output, None);
        fixture.publish(&failed, "failed", "failure", Some(&error));
        let collected = fixture
            .scope(|| execute_task_status(&serde_json::json!({})))
            .unwrap();
        assert!(collected.starts_with("[task-status] running=0 done=1 failed=1\n"));
        assert_eq!(collected.matches(&output).count(), 1);
        assert_eq!(collected.matches(&error).count(), 1);
        assert_eq!(
            fixture
                .scope(|| execute_task_status(&serde_json::json!({})))
                .unwrap(),
            "[task-status] running=0 done=0 failed=0"
        );
        assert!(!collected.contains("Progress notes:"));
        assert!(!collected.contains("Completed task results below"));
    }

    #[test]
    fn agent_team_quiet_wait_propagates_collection_errors() {
        let _env_guard = lock_unpoisoned(&crate::ai::test_support::ENV_LOCK);
        let mut fixture = RuntimeFixture::new();
        let task_id = fixture.spawn();
        let entry = with_task_entry(&task_id, Clone::clone).unwrap();
        fixture
            .context
            .app_proto
            .os
            .lock()
            .unwrap()
            .channel_send(
                Some(entry.pid),
                ChannelId(entry.result_channel_id),
                "invalid json".to_string(),
            )
            .unwrap();
        let error = fixture
            .scope(|| execute_task_wait(&serde_json::json!({ "task_ids": [task_id] })))
            .unwrap_err();
        assert!(
            error.starts_with("Failed to decode stored task result from channel"),
            "{error}"
        );
    }

    #[test]
    fn agent_team_quiet_status_preserves_completed_payload_on_later_collection_error() {
        let _env_guard = lock_unpoisoned(&crate::ai::test_support::ENV_LOCK);
        let mut fixture = RuntimeFixture::new();
        let done = fixture.spawn();
        let corrupt = fixture.spawn();
        fixture
            .scope(|| execute_task_status(&serde_json::json!({})))
            .unwrap();
        let cursor_before = lock_unpoisoned(&fixture.context.agent_team_task_status_cursor).clone();
        let output = format!("completed evidence\n{}", "界".repeat(1_000));
        fixture.publish(&done, "completed", &output, None);
        let corrupt_entry = with_task_entry(&corrupt, Clone::clone).unwrap();
        fixture
            .context
            .app_proto
            .os
            .lock()
            .unwrap()
            .channel_send(
                Some(corrupt_entry.pid),
                ChannelId(corrupt_entry.result_channel_id),
                "invalid json".to_string(),
            )
            .unwrap();
        let tracked = vec![
            (done.clone(), with_task_entry(&done, Clone::clone).unwrap()),
            (corrupt.clone(), corrupt_entry),
        ];
        let error = fixture
            .scope(|| execute_agent_team_task_status(&tracked))
            .unwrap_err();
        assert!(error.starts_with("Failed to decode stored task result from channel"));
        assert!(error.contains(&done));
        assert_eq!(error.matches(&output).count(), 1);
        assert!(with_task_entry(&done, |_| ()).is_none());
        assert!(with_task_entry(&corrupt, |_| ()).is_some());
        assert_eq!(
            *lock_unpoisoned(&fixture.context.agent_team_task_status_cursor),
            cursor_before
        );
        let ledger = fixture
            .scope(|| {
                read_persisted_task_evidence_status(&fixture.context.app_proto.session_id, &done)
            })
            .unwrap()
            .unwrap();
        assert_eq!(ledger.0, "completed");
        assert_eq!(
            error
                .split_once("[Task results collected before the error]\n")
                .unwrap()
                .1,
            ledger.1
        );
    }
}
