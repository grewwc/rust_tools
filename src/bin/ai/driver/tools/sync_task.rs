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
/// turn forever. Subagents are leaf tasks with a separate iteration cap; ten
/// minutes is enough to return useful partial evidence without wedging the
/// parent turn for an interactive session.
const SYNC_TASK_HARD_TIMEOUT: Duration = Duration::from_secs(600);
const TIMEOUT_RECOVERY_MAX_CHARS: usize = 24_000;
const TIMEOUT_RECOVERY_TAIL_MESSAGES: usize = 40;

struct SyncSubagentHistoryGuard {
    path: PathBuf,
    memory_path: Option<PathBuf>,
    cwd_dir: Option<PathBuf>,
    /// 硬超时前置位：Drop 时保留子代理历史（改名而非删除），供父代理提取超时前证据。
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
            // 硬超时：保留子代理已写入的历史（改名而非删除），供父代理提取超时前证据。
            match history::preserve_subagent_history(&self.path) {
                Some(preserved) => {
                    eprintln!(
                        "[Warning] preserved sync subagent history at {}",
                        preserved.display()
                    );
                }
                None => {
                    // 历史文件不存在（子代理尚未写入任何内容），按原逻辑清理。
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

/// 子代理"运行中"心跳的刷新间隔。同步子 agent 自身不直接拥有 terminal；
/// 前台等待循环用这条单行 heartbeat 展示进度，直到任务完成/取消/超时。
const SUBAGENT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);

type BoxedSubagentFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

fn suppress_subagent_terminal_output(wrapped: BoxedSubagentFuture) -> BoxedSubagentFuture {
    Box::pin(runtime_ctx::SUPPRESS_TERMINAL_OUTPUT.scope(true, wrapped))
}

/// 执行模型通过 `task` 工具发起的同步子代理。普通工具调用可使用完整五分钟预算；
/// 仅 driver 的显式命令可选择在硬超时前请求子代理收口。
pub(super) fn execute_sync_task(tool_call_id: &str, args: &Value) -> Result<ToolResult, String> {
    execute_sync_task_with_hard_timeout(tool_call_id, args, SYNC_TASK_HARD_TIMEOUT)
}

/// 执行 driver 自己发起的同步子代理，并采用调用方明确选择的硬超时。
///
/// 该入口保持 crate 私有，避免模型工具参数把前台等待时间放大为无界值。
pub(super) fn execute_sync_task_with_hard_timeout(
    tool_call_id: &str,
    args: &Value,
    hard_timeout: Duration,
) -> Result<ToolResult, String> {
    execute_sync_task_with_pre_timeout_wrap_up(tool_call_id, args, hard_timeout, None)
}

/// 与普通同步 `task` 相同，但允许显式命令在硬超时前预留一段时间，请子代理
/// 停止扩展调查并基于现有证据收口。该参数不暴露给模型工具调用。
pub(super) fn execute_sync_task_with_pre_timeout_wrap_up(
    tool_call_id: &str,
    args: &Value,
    hard_timeout: Duration,
    wrap_up_lead_time: Option<Duration>,
) -> Result<ToolResult, String> {
    // 递归深度守卫：防止 mode:all 的 heavy agent 通过同步 `task`
    // 无限嵌套委派。与 `spawn_subagent_kernel_task` 中的检查保持一致。
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
    // 驱动内部分派可选参数（模型调用的 task 工具 schema 不含这两个字段，不会命中）：
    // - `image_files`：把图片直接附加到子代理首条 user 消息，让 VL 模型第一轮就看到
    //   图，省掉「先 read_file、再在下一轮重复附加 base64」的冗余往返；
    // - `reasoning_effort`：压低纯转录等简单子任务的思考档位（如图片解析用 minimal）。
    if let Some(images) = args.get("image_files").and_then(|v| v.as_array()) {
        task_app.attached_image_files = images
            .iter()
            .filter_map(|v| v.as_str())
            // build_content 用原始路径 fs::read，不解析 cwd；把相对路径按 effective_cwd
            // 转绝对，与原 read_file 流程的解析语义保持一致，避免 CWD 错位打不开图。
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
            .collect();
    }
    if let Some(level) = args
        .get("reasoning_effort")
        .and_then(|v| v.as_str())
        .and_then(crate::ai::provider::ReasoningEffort::parse)
    {
        task_app.cli.reasoning_effort_override = Some(Some(level));
    }
    // 关键：子 agent 不再与父 agent 共享 shutdown/streaming/cancel_stream 标志。
    // 共享会让一次针对子 agent 的 Ctrl+C 误置全局 shutdown、连带关掉主 agent
    // （子 agent 卡在静默 prepare 阶段、streaming=false 时尤甚）。给它一组全新的
    // 私有标志：定向取消只翻子 agent 自己的 cancel，父 agent 安然存活。
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
    // 硬超时前由父进程置位，guard Drop 时保留子代理历史（改名而非删除）。
    let preserve_history_on_timeout = Arc::new(AtomicBool::new(false));
    let history_cleanup =
        SyncSubagentHistoryGuard::new(child_history.clone(), preserve_history_on_timeout.clone())
            .with_scoped_artifacts(private_memory_path.clone(), private_cwd_dir.clone());
    // 无论是否继承，子代理都只写自己的历史文件，绝不能写回父 canonical history。
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

    // 等待循环监听 **子 agent 自己** 的 shutdown/cancel 标志（而非父 agent 的）。
    // 第一次 Ctrl+C 经 ForegroundSubagentGuard 定向翻 `subagent_cancel`，唤醒
    // 等待循环、把子 agent 取消掉，父 agent 不受影响。
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
        // sub-agent 默认私有 memory：merge 白名单条目回主文件
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

    // 把子 agent 的私有 cancel 标志登记到前台子 agent 注册表：Ctrl+C 时
    // SIGINT 处理器会优先定向取消栈顶子 agent（翻这个标志），而不是关掉主 agent。
    // guard 在本函数返回时自动注销，绝不泄漏陈旧条目。
    let _foreground_guard =
        crate::ai::driver::signal::ForegroundSubagentGuard::register(subagent_cancel.clone());

    // 等待 sub-agent：只由三个事件驱动，不再 50ms 轮询。
    //   1. sub-agent oneshot 返回；
    //   2. 子 agent 的 cancel/shutdown 通过 REQUEST_INTERRUPT_NOTIFY 唤醒；
    //   3. hard timeout 到期。
    //
    // atomic flag 只作为条件判断，不作为唤醒机制；正常写入 cancel/shutdown 的入口
    // 必须调用 signal_request_interrupt()/request_shutdown() 发送 Notify。
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
        // 硬超时：先置位让 guard Drop 保留历史（改名而非删除），再 abort。
        preserve_history_on_timeout.store(true, Ordering::Release);
        subagent_handle.abort();
    }
    // `abort` 只发出取消请求；必须等任务真正退出后才能删除仍可能被它写入的 DB。
    let _ =
        tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(subagent_handle));

    let duration = started.elapsed();
    let elapsed_secs = duration.as_secs_f64();

    // 硬超时：guard 已把子代理历史改名保留，这里提取超时前的工作产物发布到 result slot，
    // 避免 15 分钟工作全部丢失（此前超时只返回空结果）。
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
    // 心跳只在交互式 TTY 下显示：它用 `\r` + 清行做单行原地刷新，管道/重定向
    // 场景下这些控制序列会污染输出，所以非 TTY 直接关闭。
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
        // interval 的第一次 tick 立即就绪，先吃掉它，让首个心跳延后一个间隔出现，
        // 避免 subagent 很快就出首包时还闪一下心跳。
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

            // 先注册 Notify future，再复查 atomic，避免 signal 在检查和
            // 注册之间发生时丢唤醒。
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

/// 硬超时后把子代理已写入历史的工作产物提取出来，发布到 result slot。
/// 父代理随后在失败结果里能看到超时前的证据节选与保留文件路径，而不是空结果
/// （此前超时路径把 15 分钟的工作产物全部丢弃）。
fn publish_timeout_evidence(
    child_history: &Path,
    result_slot: &runtime_ctx::SubagentResultSlot,
    timeout_error: &str,
    phase: &str,
) {
    let preserved = history::preserved_subagent_history_path(child_history);
    // 读取最近消息（含工具输出）作为可恢复证据；读取失败也必须发布结构化诊断，
    // 不能让硬超时退化成没有任何上下文的空结果。
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

/// 构造最多占一个终端物理行的 subagent 心跳。长路径、计划等阶段详情必须按当前
/// 终端宽度截断，否则终端自动折行后，下一次 `\r` 只能覆盖最后一行，旧状态会逐次累积。
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

/// 原地刷新一行 subagent 运行心跳（不换行）。用 `\r` 回到行首 + `\x1b[2K`
/// 清整行，保证多次心跳只占同一行；暗色显示以免喧宾夺主。
fn print_heartbeat_line(elapsed: Duration, phase: &str) {
    use std::io::Write;
    let line = render_heartbeat_line(elapsed, phase);
    print!("\r\x1b[2K\x1b[2m{line}\x1b[0m");
    let _ = std::io::stdout().flush();
}

/// 清除当前心跳行（如果有）。在 subagent 开始输出 / 任务结束 / 被取消时调用，
/// 确保心跳不残留、也不会和后续真实输出粘在同一行。
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
        drop(child);
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
            "audit subagent exceeded hard timeout (900s)",
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
            "audit subagent exceeded hard timeout (900s)",
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
