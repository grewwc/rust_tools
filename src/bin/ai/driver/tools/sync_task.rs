// =============================================================================
// Synchronous `task` tool interception
// =============================================================================
// The synchronous `task` tool is intercepted by the driver and executed
// inside the active turn's runtime, instead of being routed through the
// kernel scheduler like `task_spawn`. This lets the calling agent block
// on a single sub-agent without forking a subprocess and without relying
// on the outer driver loop to make progress (which it cannot, because the
// outer driver loop is currently awaiting this tool call).
//
// Execution model:
//   1. Read `DRIVER_CTX` to obtain a snapshot of the parent runtime
//      (`app_proto`, `mcp_client`, skill / agent manifests).
//   2. Run pre-flight (subagent + model selection, inherit parsing) via
//      `task_tools::prepare_subagent_task`.
//   3. Build a `task_app` cloned from `app_proto`, applying the inherit
//      flags. Activate the chosen subagent on the clone.
//   4. `tokio::spawn` `run_turn` for the sub-agent, wrapped in a fresh
//      `DRIVER_CTX` scope so nested sub-agents inherit the same context
//      bridge.
//   5. Block on a `oneshot::Receiver` via `Handle::current().block_on` to
//      surface the sub-agent's terminal status to the caller.
// =============================================================================

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::ai::{
    agents,
    driver::{runtime_ctx, turn_runtime},
    history,
    tools::task_tools,
    types::ToolResult,
};

use super::super::runtime_ctx::DriverContext;

/// Hard upper bound on how long a synchronous `task` tool call may block
/// the parent agent. Keeps a runaway sub-agent from wedging the foreground
/// turn forever. Subagents are leaf tasks with a separate iteration cap;
/// ten minutes gives complex sub-tasks (multi-step research, cross-module
/// refactors, audits) enough budget to return useful partial evidence without
/// wedging the parent turn for an interactive session.
const SYNC_TASK_HARD_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const TIMEOUT_RECOVERY_MAX_CHARS: usize = 24_000;
const TIMEOUT_RECOVERY_TAIL_MESSAGES: usize = 40;

struct SyncSubagentHistoryGuard {
    path: PathBuf,
    memory_path: Option<PathBuf>,
    cwd_dir: Option<PathBuf>,
    /// Set before a hard timeout: on Drop, preserve the subagent history (rename instead of
    /// delete) so the parent can extract pre-timeout evidence.
    preserve_on_drop: Arc<AtomicBool>,
}

impl SyncSubagentHistoryGuard {
    fn new(path: PathBuf, preserve_on_drop: Arc<AtomicBool>) -> Self {
        Self {
            path,
            memory_path: None,
            cwd_dir: None,
            preserve_on_drop,
        }
    }

    fn with_scoped_artifacts(
        mut self,
        memory_path: Option<PathBuf>,
        cwd_dir: Option<PathBuf>,
    ) -> Self {
        self.memory_path = memory_path;
        self.cwd_dir = cwd_dir;
        self
    }

    fn preserve_memory(&mut self) {
        self.memory_path = None;
    }
}

impl Drop for SyncSubagentHistoryGuard {
    fn drop(&mut self) {
        if self.preserve_on_drop.load(Ordering::Acquire) {
            // Hard timeout: preserve the history the subagent has already written (rename instead
            // of delete) for the parent to extract pre-timeout evidence.
            match history::preserve_subagent_history(&self.path) {
                Some(preserved) => {
                    eprintln!(
                        "[Warning] preserved sync subagent history at {}",
                        preserved.display()
                    );
                }
                None => {
                    // History file missing (the subagent has not written anything yet); clean up
                    // as before.
                    if let Err(error) = history::delete_subagent_history(&self.path) {
                        eprintln!(
                            "[Warning] failed to clean up sync subagent history {}: {error}",
                            self.path.display()
                        );
                    }
                }
            }
        } else if let Err(error) = history::delete_subagent_history(&self.path) {
            eprintln!(
                "[Warning] failed to clean up sync subagent history {}: {error}",
                self.path.display()
            );
        }
        if let Some(memory_path) = self.memory_path.take()
            && let Err(error) = history::delete_subagent_memory(&memory_path)
        {
            eprintln!(
                "[Warning] failed to clean up sync subagent memory {}: {error}",
                memory_path.display()
            );
        }
        if let Some(cwd_dir) = self.cwd_dir.take()
            && let Err(error) = std::fs::remove_dir_all(&cwd_dir)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!(
                "[Warning] failed to clean up sync subagent cwd {}: {error}",
                cwd_dir.display()
            );
        }
    }
}

/// Refresh interval for the subagent "running" heartbeat. A synchronous sub-agent does not own
/// the terminal itself; the foreground wait loop uses this single-line heartbeat to show
/// progress until the task completes / is cancelled / times out.
const SUBAGENT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);

type BoxedSubagentFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

fn suppress_subagent_terminal_output(wrapped: BoxedSubagentFuture) -> BoxedSubagentFuture {
    Box::pin(runtime_ctx::SUPPRESS_TERMINAL_OUTPUT.scope(true, wrapped))
}

/// Runs a synchronous sub-agent requested by the model through the `task` tool. Ordinary tool
/// calls may use the full hard-timeout budget (currently 10 minutes); only explicit driver
/// commands may ask the sub-agent to wrap up before the hard timeout.
pub(super) fn execute_sync_task(tool_call_id: &str, args: &Value) -> Result<ToolResult, String> {
    execute_sync_task_with_hard_timeout(tool_call_id, args, SYNC_TASK_HARD_TIMEOUT)
}

/// Runs a synchronous sub-agent initiated by the driver itself, using the caller-chosen hard
/// timeout.
///
/// This entry point stays crate-private so model tool arguments cannot inflate the foreground
/// wait time into an unbounded value.
pub(super) fn execute_sync_task_with_hard_timeout(
    tool_call_id: &str,
    args: &Value,
    hard_timeout: Duration,
) -> Result<ToolResult, String> {
    execute_sync_task_with_pre_timeout_wrap_up(tool_call_id, args, hard_timeout, None)
}

/// Same as a normal synchronous `task`, but allows explicit commands to reserve some time before
/// the hard timeout so the sub-agent can stop expanding its investigation and wrap up based on
/// the evidence at hand. This parameter is not exposed to model tool calls.
pub(super) fn execute_sync_task_with_pre_timeout_wrap_up(
    tool_call_id: &str,
    args: &Value,
    hard_timeout: Duration,
    wrap_up_lead_time: Option<Duration>,
) -> Result<ToolResult, String> {
    // Recursion depth guard: prevents a mode:all heavy agent from delegating through synchronous
    // `task` into unbounded nesting. Consistent with the check in `spawn_subagent_kernel_task`.
    let parent_depth = runtime_ctx::current_subagent_depth();
    let child_depth = parent_depth + 1;
    if child_depth > task_tools::MAX_SUBAGENT_SPAWN_DEPTH {
        return Err(format!(
            "Subagent nesting depth {} exceeds maximum {}. \
             The current agent is already a nested subagent; further delegation \
             would risk unbounded recursion. Execute the work directly instead.",
            child_depth,
            task_tools::MAX_SUBAGENT_SPAWN_DEPTH,
        ));
    }
    let prepared = task_tools::prepare_subagent_task(args)?;
    let ctx = runtime_ctx::try_current().ok_or_else(|| {
        "task tool requires an active driver turn (DRIVER_CTX is not set)".to_string()
    })?;

    let mut task_app = ctx.app_proto.clone();
    // Driver-internal dispatch options (the task tool schema seen by model calls has neither
    // field, so they never match):
    // - `image_files`: attaches images directly to the sub-agent's first user message so a VL
    //   model sees them in the first round, avoiding the redundant read_file-then-re-attach
    //   base64 round trip;
    // - `reasoning_effort`: lowers the reasoning level for simple sub-tasks such as pure
    //   transcription (e.g. minimal for image parsing).
    if let Some(images) = args.get("image_files").and_then(|v| v.as_array()) {
        let resolved = images
            .iter()
            .filter_map(|v| v.as_str())
            // build_content reads with fs::read on the raw path without resolving cwd; resolve
            // relative paths to absolute under effective_cwd to match the original read_file
            // resolution semantics, so a CWD mismatch cannot leave images unreadable.
            .map(|s| {
                let p = std::path::Path::new(s);
                if p.is_absolute() {
                    s.to_string()
                } else {
                    crate::ai::driver::runtime_ctx::effective_cwd()
                        .unwrap_or_default()
                        .join(p)
                        .to_string_lossy()
                        .into_owned()
                }
            })
            .collect::<Vec<_>>();
        // Persisted history stores image attachments as opaque session-asset keys
        // (`build_reference_content` rejects paths outside the session assets
        // directory). The sub-agent path skips `finalize_question`, which snapshots
        // @-references for the foreground, so capture the caller-provided files here.
        let assets_dir = {
            let store = history::SessionStore::new(task_app.config.history_file.as_path());
            store.session_assets_dir(&task_app.session_id)
        };
        task_app.attached_image_files = crate::ai::driver::input::snapshot_image_attachments(
            resolved,
            assets_dir.as_path(),
        )
        .map_err(|error| format!("failed to snapshot task images: {error}"))?;
    }
    if let Some(level) = args
        .get("reasoning_effort")
        .and_then(|v| v.as_str())
        .and_then(crate::ai::provider::ReasoningEffort::parse)
    {
        task_app.cli.reasoning_effort_override = Some(Some(level));
    }
    // Key: the sub-agent no longer shares shutdown/streaming/cancel_stream flags with the parent.
    // Sharing would let a single Ctrl+C aimed at the sub-agent flip the global shutdown and take
    // down the main agent too (worst when the sub-agent is stuck in the silent prepare phase with
    // streaming=false). Give it a fresh set of private flags: targeted cancel only flips the
    // sub-agent's own cancel, and the parent survives intact.
    let subagent_shutdown = Arc::new(AtomicBool::new(false));
    let subagent_streaming = Arc::new(AtomicBool::new(false));
    let subagent_cancel = Arc::new(AtomicBool::new(false));
    task_app.shutdown = subagent_shutdown.clone();
    task_app.streaming = subagent_streaming.clone();
    task_app.cancel_stream = subagent_cancel.clone();

    let parent_history_path = ctx.app_proto.session_history_file.clone();
    let task_id = uuid::Uuid::new_v4().simple().to_string();
    let private_memory_path = (!prepared.inherit.memory)
        .then(|| runtime_ctx::make_subagent_memory_path(&parent_history_path, &task_id));
    let scratch_base = parent_history_path
        .parent()
        .map(|path| path.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    let parent_history = task_app.session_history_file.clone();
    let child_history = subagent_history_path(&parent_history, &task_id);
    crate::ai::history::prepare_subagent_history(
        &parent_history,
        &child_history,
        prepared.inherit.history,
        true,
    )
    .map_err(|err| format!("准备同步子代理历史失败：{err}"))?;
    let private_cwd_dir = if prepared.inherit.cwd {
        None
    } else {
        runtime_ctx::make_subagent_cwd(&scratch_base, &task_id)
    };
    // Set by the parent before a hard timeout; on guard Drop the sub-agent history is preserved
    // (renamed, not deleted).
    let preserve_history_on_timeout = Arc::new(AtomicBool::new(false));
    let history_cleanup =
        SyncSubagentHistoryGuard::new(child_history.clone(), preserve_history_on_timeout.clone())
            .with_scoped_artifacts(private_memory_path.clone(), private_cwd_dir.clone());
    // Whether or not history is inherited, the sub-agent only writes its own history file and
    // must never write back to the parent's canonical history.
    task_app.session_history_file = child_history.clone();
    let _ = crate::ai::driver::commands::session::restore_prune_marks_for_history(&mut task_app);

    if let Some(agent) =
        agents::find_agent_by_name(ctx.agent_manifests.as_ref(), &prepared.agent_name)
    {
        if agent.disabled {
            return Err(format!(
                "Selected subagent '{}' is disabled.",
                prepared.agent_name
            ));
        }
        let capped_agent = task_tools::capped_subagent_manifest(agent);
        super::super::activate_primary_agent(&mut task_app, &capped_agent);
    }

    let task_skill_manifests = if prepared.inherit.skills {
        ctx.skill_manifests.clone()
    } else {
        std::sync::Arc::new(Vec::new())
    };

    let task_mcp = ctx.mcp_client.clone();
    let task_agent_manifests = ctx.agent_manifests.clone();
    let log_description = prepared.description.clone();
    let log_agent_name = prepared.agent_name.clone();
    let log_model = prepared.model.clone();
    let log_selection_explanation = prepared.selection_explanation.clone();

    println!(
        "\n[Task] Launching subagent '{}' with model '{}' inherit={} for: {}",
        prepared.agent_name,
        prepared.model,
        prepared.inherit.describe(),
        prepared.description,
    );
    println!("{}", prepared.selection_explanation);

    let started = Instant::now();
    let (tx, rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

    let subagent_app = task_app;
    let task_skill_manifests_for_spawn = task_skill_manifests.clone();
    let task_mcp_for_spawn = task_mcp.clone();
    let task_agent_manifests_for_spawn = task_agent_manifests.clone();
    let prompt = prepared.prompt.clone();
    let response_schema = prepared.response_schema.clone();
    let model = prepared.model.clone();
    let auto_model_fallback = prepared.auto_model_fallback;

    let spawn_driver_ctx = DriverContext::new(
        subagent_app.clone(),
        task_mcp_for_spawn.clone(),
        task_skill_manifests_for_spawn.clone(),
        task_agent_manifests_for_spawn.clone(),
    );

    // The wait loop listens to the **sub-agent's own** shutdown/cancel flags (not the parent's).
    // The first Ctrl+C goes through ForegroundSubagentGuard to flip `subagent_cancel` in a
    // targeted way, waking the wait loop and cancelling the sub-agent while the parent is
    // unaffected.
    let wait_shutdown = subagent_shutdown.clone();
    let wait_cancel = subagent_cancel.clone();
    // Slot used by the sub-agent's `finalize_turn` to publish its final
    // assistant text. Created here, scoped via `SUBAGENT_RESULT_SLOT` over
    // the spawned future, and read once the sub-agent returns.
    let result_slot: runtime_ctx::SubagentResultSlot = Arc::new(tokio::sync::Mutex::new(
        runtime_ctx::SubagentResult::default(),
    ));
    let result_slot_for_scope = result_slot.clone();
    // Slot the sub-agent writes its current execution phase into; the wait
    // loop reads it to annotate the heartbeat line ("… · calling model").
    let phase_slot = runtime_ctx::new_subagent_progress_slot();
    let phase_slot_for_scope = phase_slot.clone();
    let wrap_up_signal = wrap_up_lead_time.map(|_| Arc::new(AtomicBool::new(false)));
    let wrap_up_signal_for_scope = wrap_up_signal.clone();

    let inner_fut = async move {
        let mut subagent_app = subagent_app;
        crate::ai::tools::registry::common::clear_tool_cancel();
        let run = turn_runtime::run_turn(
            &mut subagent_app,
            &task_mcp_for_spawn,
            task_skill_manifests_for_spawn.as_slice(),
            usize::MAX,
            prompt,
            String::new(),
            model,
            None,
            false,
            false,
        );
        let result = if let Some(spec) = auto_model_fallback {
            runtime_ctx::AUTO_MODEL_FALLBACK.scope(spec, run).await
        } else {
            run.await
        }
        .map(|_outcome| ())
        .map_err(|e| format!("{}", e));
        let _ = tx.send(result);
    };

    let mut wrapped: BoxedSubagentFuture = Box::pin(inner_fut);
    let persona_memory_path = spawn_driver_ctx.app_proto.current_persona_memory_file();

    // Always install the result slot scope so `finalize_turn` can publish
    // the answer back to us regardless of inherit settings. Also install the
    // phase slot so the sub-agent's `run_turn` can report its current phase.
    wrapped =
        Box::pin(runtime_ctx::PERSONA_MEMORY_PATH.scope(persona_memory_path.clone(), wrapped));
    wrapped = Box::pin(runtime_ctx::SUBAGENT_PHASE.scope(phase_slot_for_scope, wrapped));
    if let Some(wrap_up_signal) = wrap_up_signal_for_scope {
        wrapped = Box::pin(runtime_ctx::SUBAGENT_WRAP_UP_SIGNAL.scope(wrap_up_signal, wrapped));
    }
    wrapped = Box::pin(runtime_ctx::SUBAGENT_RESULT_SLOT.scope(result_slot_for_scope, wrapped));
    wrapped = Box::pin(runtime_ctx::SUBAGENT_DEPTH.scope(child_depth, wrapped));

    let mut memory_merge = None;
    if let Some(mem_path) = private_memory_path {
        // Sub-agent memory is private by default: merge whitelisted entries back into the main file
        let main_path = persona_memory_path;
        let private_for_merge = mem_path.clone();
        wrapped = Box::pin(runtime_ctx::SUBAGENT_MEMORY_PATH.scope(mem_path, wrapped));
        memory_merge = Some((private_for_merge, main_path));
    }

    if let Some(scratch) = private_cwd_dir {
        wrapped = Box::pin(runtime_ctx::SUBAGENT_CWD.scope(scratch, wrapped));
    }

    wrapped = suppress_subagent_terminal_output(wrapped);

    let memory_merge_error = Arc::new(std::sync::Mutex::new(None));
    let memory_merge_error_for_task = memory_merge_error.clone();
    let guarded = async move {
        let mut history_cleanup = history_cleanup;
        wrapped.await;
        if let Some((private_memory, main_memory)) = memory_merge {
            match crate::ai::tools::service::memory::merge_subagent_whitelist(
                &private_memory,
                &main_memory,
            ) {
                Ok(_) => {}
                Err(error) => {
                    history_cleanup.preserve_memory();
                    *memory_merge_error_for_task
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner()) = Some(error);
                }
            }
        }
    };
    let subagent_handle = tokio::spawn(runtime_ctx::DRIVER_CTX.scope(spawn_driver_ctx, guarded));

    // Register the sub-agent's private cancel flag in the foreground sub-agent registry: on
    // Ctrl+C the SIGINT handler preferentially cancels the top-of-stack sub-agent (by flipping
    // this flag) instead of shutting down the main agent. The guard unregisters automatically
    // when this function returns, so stale entries never leak.
    let _foreground_guard =
        crate::ai::driver::signal::ForegroundSubagentGuard::register(subagent_cancel.clone());

    // Wait for the sub-agent, driven only by three events (no more 50ms polling):
    //   1. the sub-agent oneshot returns;
    //   2. the sub-agent's cancel/shutdown wakes us via REQUEST_INTERRUPT_NOTIFY;
    //   3. the hard timeout expires.
    //
    // The atomic flag is only a condition check, not a wake-up mechanism; every normal path that
    // writes cancel/shutdown must call signal_request_interrupt()/request_shutdown() to send Notify.
    let join_result: Result<Result<(), String>, String> = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(wait_for_sync_task_completion_with_wrap_up(
            rx,
            wait_shutdown,
            wait_cancel,
            phase_slot.clone(),
            started,
            hard_timeout,
            wrap_up_lead_time,
            wrap_up_signal,
        ))
    });
    if join_result.is_err() {
        subagent_cancel.store(true, Ordering::Release);
        // Hard timeout: set the flag first so guard Drop preserves the history (rename instead
        // of delete), then abort.
        preserve_history_on_timeout.store(true, Ordering::Release);
        subagent_handle.abort();
    }
    // `abort` only sends a cancel request; the DB, which the task may still be writing to, must
    // not be deleted until the task has actually exited.
    let _ =
        tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(subagent_handle));

    let duration = started.elapsed();
    let elapsed_secs = duration.as_secs_f64();

    // Hard timeout: the guard has already renamed the sub-agent history aside; here we extract
    // the pre-timeout work product and publish it to the result slot, so 10 minutes of work is
    // not lost (previously the timeout path only returned an empty result).
    if let Err(timeout_error) = &join_result {
        let timeout_phase = phase_slot
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone();
        publish_timeout_evidence(
            &child_history,
            &result_slot,
            timeout_error,
            &format!("{timeout_phase:?}"),
        );
    }

    let captured_result = result_slot
        .try_lock()
        .ok()
        .map(|guard| guard.clone())
        .unwrap_or_default();
    let captured_output = captured_result.parent_payload;
    let response_for_validation = captured_result.final_assistant_text;
    let memory_merge_error = memory_merge_error
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .take();

    let (mut status, mut error) = match (join_result, memory_merge_error) {
        (Ok(Ok(())), Some(error)) => ("FAILED", Some(error)),
        (Ok(Ok(())), None) => ("COMPLETED", None),
        (Ok(Err(err)), _) => ("FAILED", Some(err)),
        (Err(err), _) => (subagent_wait_error_status(&err), Some(err)),
    };
    if status == "COMPLETED"
        && let Err(validation_error) = task_tools::validate_subagent_response(
            response_schema.as_ref(),
            &response_for_validation,
        )
    {
        status = "FAILED";
        error = Some(validation_error);
    }
    let rendered = format_subagent_output(
        status,
        &log_description,
        &log_agent_name,
        &log_model,
        elapsed_secs,
        &log_selection_explanation,
        &captured_output,
        error.as_deref(),
    );
    let content = format!("[task_id={task_id}]\n{rendered}");
    history::record_delivered_task_evidence(
        ctx.app_proto.config.history_file.as_path(),
        &ctx.app_proto.session_id,
        history::DeliveredTaskEvidence {
            task_id: &task_id,
            description: &log_description,
            agent_name: &log_agent_name,
            model: &log_model,
            status: &status.to_ascii_lowercase(),
            payload: &content,
        },
    )
    .map_err(|error| format!("failed to persist synchronous task evidence: {error}"))?;
    Ok(ToolResult {
        tool_call_id: tool_call_id.to_string(),
        content,
    })
}

async fn wait_for_sync_task_completion(
    rx: tokio::sync::oneshot::Receiver<Result<(), String>>,
    parent_shutdown: Arc<AtomicBool>,
    parent_cancel: Arc<AtomicBool>,
    phase_slot: runtime_ctx::SubagentPhaseSlot,
    started: Instant,
    hard_timeout: Duration,
) -> Result<Result<(), String>, String> {
    wait_for_sync_task_completion_with_wrap_up(
        rx,
        parent_shutdown,
        parent_cancel,
        phase_slot,
        started,
        hard_timeout,
        None,
        None,
    )
    .await
}

async fn wait_for_sync_task_completion_with_wrap_up(
    mut rx: tokio::sync::oneshot::Receiver<Result<(), String>>,
    parent_shutdown: Arc<AtomicBool>,
    parent_cancel: Arc<AtomicBool>,
    phase_slot: runtime_ctx::SubagentPhaseSlot,
    started: Instant,
    hard_timeout: Duration,
    wrap_up_lead_time: Option<Duration>,
    wrap_up_signal: Option<runtime_ctx::SubagentWrapUpSignal>,
) -> Result<Result<(), String>, String> {
    // The heartbeat is shown only on an interactive TTY: it uses `\r` + line clearing for
    // single-line in-place refresh, and those control sequences would pollute output when piped
    // or redirected, so non-TTY is disabled entirely.
    let show_heartbeat = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let wait_for_result = async {
        let mut wrap_up_deadline = wrap_up_lead_time.zip(wrap_up_signal).map(|(lead, signal)| {
            (
                Box::pin(tokio::time::sleep(hard_timeout.saturating_sub(lead))),
                signal,
            )
        });
        let interrupt_notify = crate::ai::driver::signal::request_interrupt_notify();
        let mut heartbeat = tokio::time::interval(SUBAGENT_HEARTBEAT_INTERVAL);
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // The first tick of the interval is ready immediately; consume it first so the first
        // heartbeat appears one interval later, avoiding a flicker when the subagent emits its
        // first packet quickly.
        heartbeat.tick().await;
        let mut heartbeat_visible = false;
        loop {
            if parent_shutdown.load(Ordering::Relaxed) {
                clear_heartbeat_line(show_heartbeat, &mut heartbeat_visible);
                return Err("subagent task aborted: parent shutdown requested".to_string());
            }
            if parent_cancel.load(Ordering::Relaxed) {
                clear_heartbeat_line(show_heartbeat, &mut heartbeat_visible);
                return Err("subagent task aborted: stream cancel requested".to_string());
            }

            // Register the Notify future before re-checking the atomic, so a signal arriving
            // between the check and the registration cannot be lost.
            let notified = interrupt_notify.notified();
            if parent_shutdown.load(Ordering::Relaxed) || parent_cancel.load(Ordering::Relaxed) {
                continue;
            }

            tokio::select! {
                biased;
                res = &mut rx => {
                    clear_heartbeat_line(show_heartbeat, &mut heartbeat_visible);
                    return match res {
                        Ok(inner) => Ok(inner),
                        Err(e) => Err(format!(
                            "subagent task channel closed before result: {e}"
                        )),
                    };
                }
                _ = async {
                    match wrap_up_deadline.as_mut() {
                        Some((deadline, _)) => deadline.as_mut().await,
                        None => std::future::pending::<()>().await,
                    }
                } => {
                    let (_, signal) = wrap_up_deadline
                        .take()
                        .expect("wrap-up deadline must exist when it fires");
                    signal.store(true, Ordering::Release);
                }
                _ = notified => {
                    continue;
                }
                _ = heartbeat.tick(), if show_heartbeat => {
                    let phase = runtime_ctx::subagent_progress_snapshot(&phase_slot)
                        .unwrap_or_default();
                    print_heartbeat_line(started.elapsed(), &phase);
                    heartbeat_visible = true;
                }
            }
        }
    };

    match tokio::time::timeout(hard_timeout, wait_for_result).await {
        Ok(result) => result,
        Err(_) => Err(format!(
            "subagent task exceeded hard timeout of {}s",
            hard_timeout.as_secs()
        )),
    }
}

/// After a hard timeout, extracts the work product the sub-agent already wrote to its history
/// and publishes it to the result slot. The parent then sees a pre-timeout evidence excerpt and
/// the preserved file path in the failure result instead of an empty one (the previous timeout
/// path discarded all 10 minutes of work).
fn publish_timeout_evidence(
    child_history: &Path,
    result_slot: &runtime_ctx::SubagentResultSlot,
    timeout_error: &str,
    phase: &str,
) {
    let preserved = history::preserved_subagent_history_path(child_history);
    // Read the recent messages (including tool output) as recoverable evidence; even a read
    // failure must still publish a structured diagnosis so the hard timeout never degrades into
    // an empty result with no context.
    let (excerpt, extraction_error) = if preserved.exists() {
        match history::build_message_arr(TIMEOUT_RECOVERY_TAIL_MESSAGES, &preserved) {
            Ok(messages) => (
                history::messages_to_markdown_capped(
                    &messages,
                    &preserved.to_string_lossy(),
                    TIMEOUT_RECOVERY_MAX_CHARS,
                ),
                None,
            ),
            Err(error) => (String::new(), Some(error.to_string())),
        }
    } else {
        (
            String::new(),
            Some("preserved child history was not found".to_string()),
        )
    };
    let payload = format_timeout_recovery_payload(
        timeout_error,
        phase,
        &preserved,
        excerpt.trim(),
        extraction_error.as_deref(),
    );
    if let Ok(mut guard) = result_slot.try_lock() {
        if guard.parent_payload.trim().is_empty() && guard.final_assistant_text.trim().is_empty() {
            guard.parent_payload = payload;
        }
    }
}

fn format_timeout_recovery_payload(
    timeout_error: &str,
    phase: &str,
    preserved: &Path,
    excerpt: &str,
    extraction_error: Option<&str>,
) -> String {
    let status = if timeout_error.contains("exceeded hard timeout") {
        "timed_out"
    } else {
        "interrupted"
    };
    let evidence = if excerpt.is_empty() {
        "未恢复到非空消息；请结合下方诊断和保留路径继续排查。"
    } else {
        excerpt
    };
    let extraction = extraction_error
        .map(|error| format!("\nhistory_extraction_error: {error}"))
        .unwrap_or_default();
    format!(
        "SUBAGENT_TIMEOUT_RECOVERY_V1\nstatus: {status}\nerror: {timeout_error}\nlast_phase: {phase}\npreserved_child_history: {}{extraction}\n\n## 中断前已完成的工作（恢复节选）\n\n{evidence}\n\n以上是阶段性证据而非完整审计结论；后续应从这些证据继续，而不是从零重跑。",
        preserved.display(),
    )
}

/// Builds a subagent heartbeat that occupies at most one physical terminal line. Long paths,
/// plan and other phase details must be truncated to the current terminal width, otherwise the
/// terminal auto-wraps and the next `\r` can only overwrite the last line, so stale state
/// accumulates over time.
fn render_heartbeat_line(elapsed: Duration, phase: &str) -> String {
    let secs = elapsed.as_secs();
    let phase = phase.trim();
    let line = if phase.is_empty() {
        format!("⏳ subagent running… {secs}s (Ctrl+C to cancel)")
    } else {
        format!("⏳ subagent running… {secs}s · {phase} (Ctrl+C to cancel)")
    };
    crate::ai::stream::clamp_line_to_terminal_row_with_reserve(&line, 0)
}

/// Refreshes a single subagent heartbeat line in place (no newline). Uses `\r` to return to the
/// line start plus `\x1b[2K` to clear the whole line, so repeated heartbeats occupy the same
/// line; rendered dim so it does not distract.
fn print_heartbeat_line(elapsed: Duration, phase: &str) {
    use std::io::Write;
    let line = render_heartbeat_line(elapsed, phase);
    print!("\r\x1b[2K\x1b[2m{line}\x1b[0m");
    let _ = std::io::stdout().flush();
}

/// Clears the current heartbeat line (if any). Called when the subagent starts emitting output,
/// when the task ends, or when it is cancelled, ensuring the heartbeat never lingers or sticks
/// to the same line as later real output.
fn clear_heartbeat_line(show_heartbeat: bool, heartbeat_visible: &mut bool) {
    if show_heartbeat && *heartbeat_visible {
        use std::io::Write;
        print!("\r\x1b[2K");
        let _ = std::io::stdout().flush();
        *heartbeat_visible = false;
    }
}

/// Build the textual representation returned to the parent agent. Always
/// includes the captured sub-agent output when available, so the parent
/// actually sees what the sub-agent produced instead of just a status
/// header.
fn format_subagent_output(
    status: &str,
    description: &str,
    agent: &str,
    model: &str,
    elapsed_secs: f64,
    selection_explanation: &str,
    captured_output: &str,
    error: Option<&str>,
) -> String {
    let mut parts = vec![format!(
        "[Task: {} via {} @ {}] {} after {:.1}s",
        description, agent, model, status, elapsed_secs
    )];
    if !selection_explanation.is_empty() {
        parts.push(selection_explanation.to_string());
    }
    if let Some(err) = error
        && !err.trim().is_empty()
    {
        parts.push(format!("Error: {}", err));
    }
    let trimmed_output = captured_output.trim();
    if !trimmed_output.is_empty() {
        parts.push(trimmed_output.to_string());
    } else {
        parts.push("(subagent did not produce any final assistant text)".to_string());
    }
    parts.push(task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER.to_string());
    parts.join("\n")
}

fn subagent_wait_error_status(err: &str) -> &'static str {
    let err = err.to_ascii_lowercase();
    if err.contains("hard timeout") {
        "TIMED_OUT"
    } else if err.contains("aborted") || err.contains("cancel") {
        "CANCELLED"
    } else {
        "FAILED"
    }
}

fn subagent_history_path(base: &std::path::Path, task_id: &str) -> PathBuf {
    crate::ai::driver::process_context::history_path_with_suffix(
        base,
        &format!(".subagent-{task_id}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sync_task_wait_signals_wrap_up_before_hard_timeout() {
        let (tx, rx) = tokio::sync::oneshot::channel::<Result<(), String>>();
        let signal: runtime_ctx::SubagentWrapUpSignal = Arc::new(AtomicBool::new(false));
        let signal_for_wait = signal.clone();
        let phase_slot = runtime_ctx::new_subagent_progress_slot();
        let waiter = tokio::spawn(wait_for_sync_task_completion_with_wrap_up(
            rx,
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
            phase_slot,
            Instant::now(),
            Duration::from_millis(800),
            Some(Duration::from_millis(600)),
            Some(signal_for_wait),
        ));

        tokio::time::timeout(Duration::from_millis(600), async {
            while !signal.load(Ordering::Acquire) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("wrap-up signal should arrive before the hard timeout");
        tx.send(Ok(()))
            .expect("waiting child should still accept its completion");
        assert_eq!(waiter.await.expect("waiter should not panic"), Ok(Ok(())));
    }

    fn flags() -> (Arc<AtomicBool>, Arc<AtomicBool>) {
        (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
        )
    }

    fn phase() -> runtime_ctx::SubagentPhaseSlot {
        runtime_ctx::new_subagent_progress_slot()
    }

    #[test]
    fn sync_task_heartbeat_is_safe_and_clamped_to_one_terminal_row() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        unsafe { std::env::set_var("COLUMNS", "48") };

        let slot = runtime_ctx::new_subagent_progress_slot();
        let phase = runtime_ctx::SUBAGENT_PHASE.sync_scope(slot.clone(), || {
            runtime_ctx::publish_subagent_phase(
                "using read_file · \x1b[2J/Users/example/a/very/long/path.rs\x07",
            );
            runtime_ctx::subagent_progress_snapshot(&slot).expect("progress should be published")
        });
        let line = render_heartbeat_line(Duration::from_secs(66), &phase);

        assert!(!phase.chars().any(char::is_control));
        assert!(!line.chars().any(char::is_control));
        assert!(
            line.ends_with('…'),
            "long heartbeat should be truncated: {line}"
        );
        assert!(unicode_width::UnicodeWidthStr::width(line.as_str()) <= 48);
        unsafe { std::env::remove_var("COLUMNS") };
    }

    #[tokio::test]
    async fn sync_task_subagent_future_suppresses_terminal_output() {
        assert!(runtime_ctx::terminal_output_enabled());
        let (tx, rx) = tokio::sync::oneshot::channel();
        let fut: BoxedSubagentFuture = Box::pin(async move {
            let _ = tx.send(runtime_ctx::terminal_output_enabled());
        });

        suppress_subagent_terminal_output(fut).await;

        assert!(
            !rx.await
                .expect("subagent future should report terminal state")
        );
        assert!(runtime_ctx::terminal_output_enabled());
    }

    #[tokio::test]
    async fn sync_task_wait_returns_subagent_result() {
        let (shutdown, cancel) = flags();
        let (tx, rx) = tokio::sync::oneshot::channel();
        tx.send(Ok(())).unwrap();

        let result = wait_for_sync_task_completion(
            rx,
            shutdown,
            cancel,
            phase(),
            Instant::now(),
            Duration::from_secs(1),
        )
        .await;

        assert_eq!(result, Ok(Ok(())));
    }

    #[tokio::test]
    async fn sync_task_wait_wakes_on_cancel_notify() {
        let _signal_guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        crate::ai::driver::signal::clear_request_interrupt();
        let (shutdown, cancel) = flags();
        let (_tx, rx) = tokio::sync::oneshot::channel();
        let cancel_for_trigger = cancel.clone();

        let waiter = tokio::spawn(wait_for_sync_task_completion(
            rx,
            shutdown,
            cancel,
            phase(),
            Instant::now(),
            Duration::from_secs(5),
        ));
        tokio::time::sleep(Duration::from_millis(20)).await;
        cancel_for_trigger.store(true, Ordering::Relaxed);
        crate::ai::driver::signal::signal_request_interrupt();

        let result = tokio::time::timeout(Duration::from_millis(200), waiter)
            .await
            .expect("sync task wait should wake from Notify")
            .expect("wait task should not panic");
        assert_eq!(
            result,
            Err("subagent task aborted: stream cancel requested".to_string())
        );
        crate::ai::driver::signal::clear_request_interrupt();
    }

    #[tokio::test]
    async fn sync_task_wait_respects_hard_timeout() {
        let (shutdown, cancel) = flags();
        let (_tx, rx) = tokio::sync::oneshot::channel();

        let result = wait_for_sync_task_completion(
            rx,
            shutdown,
            cancel,
            phase(),
            Instant::now(),
            Duration::from_millis(10),
        )
        .await;

        assert_eq!(
            result,
            Err("subagent task exceeded hard timeout of 0s".to_string())
        );
    }

    #[test]
    fn sync_task_formats_timeout_as_parent_visible_output() {
        let timeout_error = format!(
            "subagent task exceeded hard timeout of {}s",
            SYNC_TASK_HARD_TIMEOUT.as_secs()
        );
        let output = format_subagent_output(
            subagent_wait_error_status(&timeout_error),
            "verify behavior",
            "build",
            "qwen3.7-max",
            SYNC_TASK_HARD_TIMEOUT.as_secs_f64(),
            "model_reason=auto-selected",
            "",
            Some(&timeout_error),
        );

        assert!(output.contains("TIMED_OUT"));
        assert!(output.contains(&format!("Error: {timeout_error}")));
        assert!(output.contains("(subagent did not produce any final assistant text)"));
        assert!(output.contains(task_tools::SUBAGENT_PARENT_SUMMARY_REMINDER));
    }

    #[test]
    fn ordinary_sync_task_hard_timeout_remains_ten_minutes() {
        assert_eq!(SYNC_TASK_HARD_TIMEOUT, Duration::from_secs(10 * 60));
    }

    #[test]
    fn subagent_history_path_preserves_sqlite_extension() {
        let got = subagent_history_path(std::path::Path::new("/tmp/session.sqlite"), "abc123");

        assert_eq!(
            got,
            std::path::PathBuf::from("/tmp/session.subagent-abc123.sqlite")
        );
    }

    #[test]
    fn subagent_history_path_preserves_text_extension() {
        let got = subagent_history_path(std::path::Path::new("/tmp/session.txt"), "abc123");

        assert_eq!(
            got,
            std::path::PathBuf::from("/tmp/session.subagent-abc123.txt")
        );
    }

    #[test]
    fn inherited_sync_subagent_history_is_forked_from_parent() {
        let root =
            std::env::temp_dir().join(format!("ai-sync-history-fork-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let parent_path = root.join("session.sqlite");
        let child_path = subagent_history_path(&parent_path, "abc123");
        let parent = rusqlite::Connection::open(&parent_path).unwrap();
        parent
            .execute("CREATE TABLE evidence(value TEXT)", [])
            .unwrap();
        parent
            .execute("INSERT INTO evidence VALUES ('parent')", [])
            .unwrap();
        drop(parent);

        crate::ai::history::prepare_subagent_history(&parent_path, &child_path, true, true)
            .unwrap();

        let child = rusqlite::Connection::open(&child_path).unwrap();
        assert_eq!(
            child
                .query_row("SELECT value FROM evidence", [], |row| row
                    .get::<_, String>(0))
                .unwrap(),
            "parent"
        );
        child
            .execute("INSERT INTO evidence VALUES ('child')", [])
            .unwrap();
        drop(child);

        let parent = rusqlite::Connection::open(&parent_path).unwrap();
        assert_eq!(
            parent
                .query_row("SELECT COUNT(*) FROM evidence", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            1
        );
        drop(parent);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn sync_subagent_text_history_uses_text_backend() {
        let root =
            std::env::temp_dir().join(format!("ai-sync-text-history-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let parent_path = root.join("session.txt");
        let inherited_path = subagent_history_path(&parent_path, "inherited");
        let empty_path = subagent_history_path(&parent_path, "empty");
        std::fs::write(&parent_path, "user:parent evidence\n").unwrap();

        crate::ai::history::prepare_subagent_history(&parent_path, &inherited_path, true, true)
            .unwrap();
        crate::ai::history::prepare_subagent_history(&parent_path, &empty_path, false, true)
            .unwrap();

        assert_eq!(
            std::fs::read_to_string(&inherited_path).unwrap(),
            "user:parent evidence\n"
        );
        assert_eq!(std::fs::read_to_string(&empty_path).unwrap(), "");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn sync_subagent_history_guard_removes_database_sidecars_and_lock() {
        let root =
            std::env::temp_dir().join(format!("ai-sync-history-cleanup-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let child_path = root.join("session.subagent-child.sqlite");
        std::fs::write(&child_path, b"db").unwrap();
        std::fs::write(format!("{}-wal", child_path.display()), b"wal").unwrap();
        std::fs::write(format!("{}-shm", child_path.display()), b"shm").unwrap();
        let lock_path = child_path.with_file_name(format!(
            ".{}.state.lock",
            child_path.file_name().unwrap().to_string_lossy()
        ));
        std::fs::write(&lock_path, b"").unwrap();
        let memory_path = root.join("agent_memory.subagent-child.jsonl");
        let memory_db_path = root.join("agent_memory.subagent-child.db");
        std::fs::write(&memory_path, b"[]").unwrap();
        std::fs::write(&memory_db_path, b"db").unwrap();
        std::fs::write(format!("{}-wal", memory_db_path.display()), b"wal").unwrap();
        let cwd_path = root.join("subagent-cwd-child");
        std::fs::create_dir_all(cwd_path.join("nested")).unwrap();
        std::fs::write(cwd_path.join("scratch.txt"), b"scratch").unwrap();

        drop(
            SyncSubagentHistoryGuard::new(child_path.clone(), Arc::new(AtomicBool::new(false)))
                .with_scoped_artifacts(Some(memory_path.clone()), Some(cwd_path.clone())),
        );

        assert!(!child_path.exists());
        assert!(!std::path::Path::new(&format!("{}-wal", child_path.display())).exists());
        assert!(!std::path::Path::new(&format!("{}-shm", child_path.display())).exists());
        assert!(!lock_path.exists());
        assert!(!memory_path.exists());
        assert!(!memory_db_path.exists());
        assert!(!std::path::Path::new(&format!("{}-wal", memory_db_path.display())).exists());
        assert!(!cwd_path.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn sync_subagent_memory_is_merged_before_cleanup() {
        let root = std::env::temp_dir().join(format!(
            "ai-sync-memory-merge-order-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let child_path = root.join("session.subagent-child.sqlite");
        let private_memory = root.join("agent_memory.subagent-child.jsonl");
        let main_memory = root.join("agent_memory.jsonl");
        std::fs::write(&child_path, b"db").unwrap();
        std::fs::write(
            &private_memory,
            serde_json::json!({
                "id": "mem-sync-merge",
                "timestamp": "2026-07-31T00:00:00Z",
                "category": "project_memory",
                "note": "durable synchronous conclusion",
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
        let guard = SyncSubagentHistoryGuard::new(child_path, Arc::new(AtomicBool::new(false)))
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
                .contains("durable synchronous conclusion")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn timeout_recovery_payload_keeps_partial_audit_work() {
        let payload = format_timeout_recovery_payload(
            "audit subagent exceeded hard timeout (600s)",
            "CallingTool",
            Path::new("/tmp/audit-timeout.sqlite"),
            "AUDIT_CHECKPOINT: checked src/a.rs; finding at src/a.rs:42",
            None,
        );

        assert!(payload.contains("SUBAGENT_TIMEOUT_RECOVERY_V1"));
        assert!(payload.contains("last_phase: CallingTool"));
        assert!(payload.contains("src/a.rs:42"));
        assert!(payload.contains("/tmp/audit-timeout.sqlite"));
        assert!(payload.contains("不是从零重跑"));
    }

    #[test]
    fn timeout_recovery_payload_exposes_missing_history_diagnostic() {
        let payload = format_timeout_recovery_payload(
            "audit subagent exceeded hard timeout (600s)",
            "WaitingForModel",
            Path::new("/tmp/missing.sqlite"),
            "",
            Some("preserved child history was not found"),
        );

        assert!(payload.contains("history_extraction_error"));
        assert!(payload.contains("preserved child history was not found"));
        assert!(payload.contains("未恢复到非空消息"));
    }

    #[test]
    fn inherited_sync_subagent_history_fork_failure_is_returned() {
        let root =
            std::env::temp_dir().join(format!("sync-task-fork-error-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let child = root.join("child.sqlite");

        let error =
            crate::ai::history::prepare_subagent_history(&root, &child, true, true).unwrap_err();

        assert!(!error.to_string().is_empty());
        assert!(!child.exists());
        let _ = std::fs::remove_dir_all(root);
    }
}
