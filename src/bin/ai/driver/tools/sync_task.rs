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
    Arc, Mutex,
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
/// turn forever. 10 minutes is generous for a single subagent invocation
/// while still being shorter than typical interactive patience.
const SYNC_TASK_HARD_TIMEOUT: Duration = Duration::from_secs(600);

/// 子代理"运行中"心跳的刷新间隔。仅在 subagent 尚未产出任何流式输出
/// 的等待窗口里使用（首个 token 到达前），用于消除"看似卡死"的死寂感。
const SUBAGENT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);

pub(super) fn execute_sync_task(tool_call_id: &str, args: &Value) -> Result<ToolResult, String> {
    let prepared = task_tools::prepare_subagent_task(args)?;
    let ctx = runtime_ctx::try_current().ok_or_else(|| {
        "task tool requires an active driver turn (DRIVER_CTX is not set)".to_string()
    })?;

    let mut task_app = ctx.app_proto.clone();
    crate::ai::types::clear_stream_cancel(&task_app);

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
        super::super::activate_primary_agent(&mut task_app, agent);
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

    let spawn_driver_ctx = DriverContext::new(
        subagent_app.clone(),
        task_mcp_for_spawn.clone(),
        task_skill_manifests_for_spawn.clone(),
        task_agent_manifests_for_spawn.clone(),
    );

    // Capture the parent's shutdown / cancel flags so the wait loop below
    // can react to ctrl-c instead of pinning a tokio worker thread for the
    // entire sub-agent run.
    let parent_shutdown = ctx.app_proto.shutdown.clone();
    let parent_cancel = ctx.app_proto.cancel_stream.clone();
    // `streaming` 是 `Arc<AtomicBool>`，clone 时共享同一个标志位；subagent 的
    // `run_turn` 一旦开始流式响应就会把它置 true（见 iteration.rs 的
    // `StreamingFlagGuard`）。父 agent 此刻阻塞在工具执行里、自身不流式，所以
    // 这个共享标志可以精确地当作"subagent 首个输出已到达"的信号，用来在恰当
    // 时机关闭等待心跳，避免心跳与 subagent 正文交错。
    let parent_streaming = ctx.app_proto.streaming.clone();
    // Slot used by the sub-agent's `finalize_turn` to publish its final
    // assistant text. Created here, scoped via `SUBAGENT_RESULT_SLOT` over
    // the spawned future, and read once the sub-agent returns.
    let result_slot: runtime_ctx::SubagentResultSlot = Arc::new(Mutex::new(None));
    let result_slot_for_scope = result_slot.clone();

    let inherit = prepared.inherit;

    let inner_fut = async move {
        let mut subagent_app = subagent_app;
        crate::ai::tools::registry::common::clear_tool_cancel();
        let result = turn_runtime::run_turn(
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
        )
        .await
        .map(|_outcome| ())
        .map_err(|e| format!("{}", e));
        let _ = tx.send(result);
    };

    type BoxedSubagentFuture = Pin<Box<dyn Future<Output = ()> + Send>>;
    let mut wrapped: BoxedSubagentFuture = Box::pin(inner_fut);

    // Always install the result slot scope so `finalize_turn` can publish
    // the answer back to us regardless of inherit settings.
    wrapped = Box::pin(runtime_ctx::SUBAGENT_RESULT_SLOT.scope(result_slot_for_scope, wrapped));

    if !inherit.memory {
        let mem_path = runtime_ctx::make_subagent_memory_path(&parent_history_path, &task_id);
        // sub-agent 默认私有 memory：merge 白名单条目回主文件
        let main_path = crate::ai::tools::storage::memory_store::MemoryStore::from_env_or_config()
            .path()
            .to_path_buf();
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

    tokio::spawn(runtime_ctx::DRIVER_CTX.scope(spawn_driver_ctx, wrapped));

    // 等待 sub-agent：只由三个事件驱动，不再 50ms 轮询。
    //   1. sub-agent oneshot 返回；
    //   2. 父 agent 的 cancel/shutdown 通过 REQUEST_INTERRUPT_NOTIFY 唤醒；
    //   3. hard timeout 到期。
    //
    // atomic flag 只作为条件判断，不作为唤醒机制；正常写入 cancel/shutdown 的入口
    // 必须调用 signal_request_interrupt()/request_shutdown() 发送 Notify。
    let join_result: Result<Result<(), String>, String> = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(wait_for_sync_task_completion(
            rx,
            parent_shutdown,
            parent_cancel,
            parent_streaming,
            started,
            SYNC_TASK_HARD_TIMEOUT,
        ))
    });

    let duration = started.elapsed();
    let elapsed_secs = duration.as_secs_f64();

    let captured_output = result_slot
        .lock()
        .ok()
        .and_then(|guard| guard.clone())
        .unwrap_or_default();

    match join_result {
        Ok(Ok(())) => Ok(ToolResult {
            tool_call_id: tool_call_id.to_string(),
            content: format_subagent_output(
                "COMPLETED",
                &log_description,
                &log_agent_name,
                &log_model,
                elapsed_secs,
                &log_selection_explanation,
                &captured_output,
                None,
            ),
        }),
        Ok(Err(err)) => Err(format_subagent_output(
            "FAILED",
            &log_description,
            &log_agent_name,
            &log_model,
            elapsed_secs,
            &log_selection_explanation,
            &captured_output,
            Some(&err),
        )),
        Err(err) => Err(format_subagent_output(
            "INTERNAL_ERROR",
            &log_description,
            &log_agent_name,
            &log_model,
            elapsed_secs,
            &log_selection_explanation,
            &captured_output,
            Some(&err),
        )),
    }
}

async fn wait_for_sync_task_completion(
    mut rx: tokio::sync::oneshot::Receiver<Result<(), String>>,
    parent_shutdown: Arc<AtomicBool>,
    parent_cancel: Arc<AtomicBool>,
    streaming_flag: Arc<AtomicBool>,
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
        // 一旦 subagent 开始流式输出就永久停止心跳：之后它会持续打印正文/工具
        // 调用，用户能直接看到活动，心跳只会和正文交错添乱。
        let mut subagent_started = false;
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
                _ = heartbeat.tick(), if show_heartbeat && !subagent_started => {
                    if streaming_flag.load(Ordering::Relaxed) {
                        // subagent 已开始产出：清掉心跳行并永久停用心跳。
                        subagent_started = true;
                        clear_heartbeat_line(show_heartbeat, &mut heartbeat_visible);
                        continue;
                    }
                    print_heartbeat_line(started.elapsed());
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
/// 清整行，保证多次心跳只占同一行；暗色显示以免喧宾夺主。
fn print_heartbeat_line(elapsed: Duration) {
    use std::io::Write;
    let secs = elapsed.as_secs();
    print!("\r\x1b[2K\x1b[2m⏳ subagent running… {secs}s (Ctrl+C to cancel)\x1b[0m");
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
    parts.join("\n")
}

fn subagent_history_path(base: &std::path::Path, task_id: &str) -> PathBuf {
    let file_name = base
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| format!("{name}.subagent-{task_id}"))
        .unwrap_or_else(|| format!("session.subagent-{task_id}"));
    base.with_file_name(file_name)
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

    #[tokio::test]
    async fn sync_task_wait_returns_subagent_result() {
        let (shutdown, cancel) = flags();
        let streaming = Arc::new(AtomicBool::new(false));
        let (tx, rx) = tokio::sync::oneshot::channel();
        tx.send(Ok(())).unwrap();

        let result = wait_for_sync_task_completion(
            rx,
            shutdown,
            cancel,
            streaming,
            Instant::now(),
            Duration::from_secs(1),
        )
        .await;

        assert_eq!(result, Ok(Ok(())));
    }

    #[tokio::test]
    async fn sync_task_wait_wakes_on_cancel_notify() {
        let _signal_guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
        crate::ai::driver::signal::clear_request_interrupt();
        let (shutdown, cancel) = flags();
        let streaming = Arc::new(AtomicBool::new(false));
        let (_tx, rx) = tokio::sync::oneshot::channel();
        let cancel_for_trigger = cancel.clone();

        let waiter = tokio::spawn(wait_for_sync_task_completion(
            rx,
            shutdown,
            cancel,
            streaming,
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
        let streaming = Arc::new(AtomicBool::new(false));
        let (_tx, rx) = tokio::sync::oneshot::channel();

        let result = wait_for_sync_task_completion(
            rx,
            shutdown,
            cancel,
            streaming,
            Instant::now(),
            Duration::from_millis(10),
        )
        .await;

        assert_eq!(
            result,
            Err("subagent task exceeded hard timeout of 0s".to_string())
        );
    }
}
