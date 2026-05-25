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
use std::sync::{Arc, Mutex};
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
/// 兜底周期，覆盖外部代码直接 `.store(true)` 但未触发 `REQUEST_INTERRUPT_NOTIFY`
/// 的边缘情况。绝大多数场景由 oneshot 完成或 Notify 即时唤醒，无需依赖此兜底。
/// 旧实现是 50ms 紧凑轮询；改成 500ms 后空闲 wake-up 频次降低 10×，
/// 同时保留对未发 Notify 的 atomic 写入的最坏 500ms 反应延迟。
const SYNC_TASK_FALLBACK_POLL_INTERVAL: Duration = Duration::from_millis(500);

pub(super) fn execute_sync_task(
    tool_call_id: &str,
    args: &Value,
) -> Result<ToolResult, String> {
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

    // Slot used by the sub-agent's `finalize_turn` to publish its final
    // assistant text. Created here, scoped via `SUBAGENT_RESULT_SLOT` over
    // the spawned future, and read once the sub-agent returns.
    let result_slot: runtime_ctx::SubagentResultSlot =
        Arc::new(Mutex::new(None));
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
    wrapped = Box::pin(
        runtime_ctx::SUBAGENT_RESULT_SLOT.scope(result_slot_for_scope, wrapped),
    );

    if !inherit.memory {
        let mem_path =
            runtime_ctx::make_subagent_memory_path(&parent_history_path, &task_id);
        // sub-agent 默认私有 memory：merge 白名单条目回主文件
        let main_path = crate::ai::tools::storage::memory_store::
            MemoryStore::from_env_or_config().path().to_path_buf();
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

    // Wait for the sub-agent in a loop that also honours shutdown,
    // cancel_stream and a hard timeout. 之前是 50ms 紧凑轮询，改成
    // `tokio::select!` 监听三路事件：
    //   1) sub-agent 的 oneshot::Receiver 完成；
    //   2) 信号模块的 REQUEST_INTERRUPT_NOTIFY（ctrl-c → cancel_stream/shutdown 同时发 notify）；
    //   3) 兜底 `tokio::time::sleep(SYNC_TASK_FALLBACK_POLL_INTERVAL)`，覆盖
    //      外部代码直接 `.store(true)` 但未触发 Notify 的极少数场景，并兜底 hard timeout。
    // 这样空闲时不再每 50ms 醒来一次，硬中断能立刻感知。
    let join_result: Result<Result<(), String>, String> =
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                let mut rx = rx;
                let interrupt_notify =
                    crate::ai::driver::signal::request_interrupt_notify();
                loop {
                    if parent_shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                        return Err("subagent task aborted: parent shutdown requested".to_string());
                    }
                    if parent_cancel.load(std::sync::atomic::Ordering::Relaxed) {
                        return Err("subagent task aborted: stream cancel requested".to_string());
                    }
                    let elapsed = started.elapsed();
                    if elapsed >= SYNC_TASK_HARD_TIMEOUT {
                        return Err(format!(
                            "subagent task exceeded hard timeout of {}s",
                            SYNC_TASK_HARD_TIMEOUT.as_secs()
                        ));
                    }
                    let remaining = SYNC_TASK_HARD_TIMEOUT - elapsed;
                    let fallback = SYNC_TASK_FALLBACK_POLL_INTERVAL.min(remaining);
                    // 注册 notified 之前再做一次 atomic 检查，避免 Notify 与
                    // store(true) 之间的 race。
                    let notified = interrupt_notify.notified();
                    if parent_shutdown.load(std::sync::atomic::Ordering::Relaxed)
                        || parent_cancel.load(std::sync::atomic::Ordering::Relaxed)
                    {
                        continue;
                    }
                    tokio::select! {
                        biased;
                        res = &mut rx => {
                            return match res {
                                Ok(inner) => Ok(inner),
                                Err(e) => Err(format!(
                                    "subagent task channel closed before result: {e}"
                                )),
                            };
                        }
                        _ = notified => { continue; }
                        _ = tokio::time::sleep(fallback) => { continue; }
                    }
                }
            })
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
        parts.push(
            "(subagent did not produce any final assistant text)".to_string(),
        );
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
