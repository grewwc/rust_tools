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
use std::path::PathBuf;
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
    tools::task_tools,
    types::ToolResult,
};

use super::super::runtime_ctx::DriverContext;

/// Hard upper bound on how long a synchronous `task` tool call may block
/// the parent agent. Keeps a runaway sub-agent from wedging the foreground
/// turn forever. Subagents are leaf tasks with a separate iteration cap; five
/// minutes is enough to return useful partial evidence without wedging the
/// parent turn for an interactive session.
const SYNC_TASK_HARD_TIMEOUT: Duration = Duration::from_secs(300);

/// 子代理"运行中"心跳的刷新间隔。同步子 agent 自身不直接拥有 terminal；
/// 前台等待循环用这条单行 heartbeat 展示进度，直到任务完成/取消/超时。
const SUBAGENT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);

type BoxedSubagentFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

fn suppress_subagent_terminal_output(wrapped: BoxedSubagentFuture) -> BoxedSubagentFuture {
    Box::pin(runtime_ctx::SUPPRESS_TERMINAL_OUTPUT.scope(true, wrapped))
}

pub(super) fn execute_sync_task(tool_call_id: &str, args: &Value) -> Result<ToolResult, String> {
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

    if !prepared.inherit.history {
        task_app.session_history_file =
            subagent_history_path(&task_app.session_history_file, &task_id);
    }

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
    let result_slot: runtime_ctx::SubagentResultSlot = Arc::new(tokio::sync::Mutex::new(None));
    let result_slot_for_scope = result_slot.clone();
    // Slot the sub-agent writes its current execution phase into; the wait
    // loop reads it to annotate the heartbeat line ("… · calling model").
    let phase_slot: runtime_ctx::SubagentPhaseSlot = Arc::new(std::sync::Mutex::new(String::new()));
    let phase_slot_for_scope = phase_slot.clone();

    let inherit = prepared.inherit;

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
    wrapped = Box::pin(runtime_ctx::SUBAGENT_RESULT_SLOT.scope(result_slot_for_scope, wrapped));
    wrapped = Box::pin(runtime_ctx::SUBAGENT_DEPTH.scope(child_depth, wrapped));

    if !inherit.memory {
        let mem_path = runtime_ctx::make_subagent_memory_path(&parent_history_path, &task_id);
        // sub-agent 默认私有 memory：merge 白名单条目回主文件
        let main_path = persona_memory_path;
        let private_for_merge = mem_path.clone();
        wrapped = Box::pin(runtime_ctx::SUBAGENT_MEMORY_PATH.scope(mem_path, wrapped));
        let inner = wrapped;
        wrapped = Box::pin(async move {
            inner.await;
            let _ = crate::ai::tools::service::memory::merge_subagent_whitelist(
                &private_for_merge,
                &main_path,
            );
        });
    }

    if !inherit.cwd {
        let scratch_base = parent_history_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        if let Some(scratch) = runtime_ctx::make_subagent_cwd(&scratch_base, &task_id) {
            wrapped = Box::pin(runtime_ctx::SUBAGENT_CWD.scope(scratch, wrapped));
        }
    }

    wrapped = suppress_subagent_terminal_output(wrapped);

    let subagent_handle = tokio::spawn(runtime_ctx::DRIVER_CTX.scope(spawn_driver_ctx, wrapped));

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
        tokio::runtime::Handle::current().block_on(wait_for_sync_task_completion(
            rx,
            wait_shutdown,
            wait_cancel,
            phase_slot,
            started,
            SYNC_TASK_HARD_TIMEOUT,
        ))
    });
    if join_result.is_err() {
        subagent_handle.abort();
    }

    let duration = started.elapsed();
    let elapsed_secs = duration.as_secs_f64();

    let captured_output = result_slot
        .try_lock()
        .ok()
        .and_then(|guard| guard.clone())
        .unwrap_or_default();

    let (status, error) = match join_result {
        Ok(Ok(())) => ("COMPLETED", None),
        Ok(Err(err)) => ("FAILED", Some(err)),
        Err(err) => (subagent_wait_error_status(&err), Some(err)),
    };
    Ok(ToolResult {
        tool_call_id: tool_call_id.to_string(),
        content: format_subagent_output(
            status,
            &log_description,
            &log_agent_name,
            &log_model,
            elapsed_secs,
            &log_selection_explanation,
            &captured_output,
            error.as_deref(),
        ),
    })
}

async fn wait_for_sync_task_completion(
    mut rx: tokio::sync::oneshot::Receiver<Result<(), String>>,
    parent_shutdown: Arc<AtomicBool>,
    parent_cancel: Arc<AtomicBool>,
    phase_slot: runtime_ctx::SubagentPhaseSlot,
    started: Instant,
    hard_timeout: Duration,
) -> Result<Result<(), String>, String> {
    // 心跳只在交互式 TTY 下显示：它用 `\r` + 清行做单行原地刷新，管道/重定向
    // 场景下这些控制序列会污染输出，所以非 TTY 直接关闭。
    let show_heartbeat = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let wait_for_result = async {
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
                _ = notified => {
                    continue;
                }
                _ = heartbeat.tick(), if show_heartbeat => {
                    let phase = phase_slot
                        .lock()
                        .ok()
                        .map(|guard| guard.clone())
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

/// 原地刷新一行 subagent 运行心跳（不换行）。用 `\r` 回到行首 + `\x1b[2K`
/// 清整行，保证多次心跳只占同一行；暗色显示以免喧宾夺主。`phase` 非空时
/// 追加当前执行阶段（如 "calling model"），让用户看到子 agent 正在做什么。
fn print_heartbeat_line(elapsed: Duration, phase: &str) {
    use std::io::Write;
    let secs = elapsed.as_secs();
    let phase = phase.trim();
    if phase.is_empty() {
        print!("\r\x1b[2K\x1b[2m⏳ subagent running… {secs}s (Ctrl+C to cancel)\x1b[0m");
    } else {
        print!("\r\x1b[2K\x1b[2m⏳ subagent running… {secs}s · {phase} (Ctrl+C to cancel)\x1b[0m");
    }
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

    fn flags() -> (Arc<AtomicBool>, Arc<AtomicBool>) {
        (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
        )
    }

    fn phase() -> runtime_ctx::SubagentPhaseSlot {
        Arc::new(std::sync::Mutex::new(String::new()))
    }

    #[tokio::test]
    async fn sync_task_subagent_future_suppresses_terminal_output() {
        assert!(runtime_ctx::terminal_output_enabled());
        let (tx, rx) = tokio::sync::oneshot::channel();
        let fut: BoxedSubagentFuture = Box::pin(async move {
            let _ = tx.send(runtime_ctx::terminal_output_enabled());
        });

        suppress_subagent_terminal_output(fut).await;

        assert!(!rx.await.expect("subagent future should report terminal state"));
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
    fn subagent_history_path_preserves_sqlite_extension() {
        let got = subagent_history_path(
            std::path::Path::new("/tmp/session.sqlite"),
            "abc123",
        );

        assert_eq!(got, std::path::PathBuf::from("/tmp/session.subagent-abc123.sqlite"));
    }
}
