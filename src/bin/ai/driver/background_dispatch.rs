//! Background process dispatch: select ready processes, decode task goals,
//! build questions, clone app context, and spawn each as a tokio task.
//!
//! Extracted from `driver/mod.rs` `run_loop` to establish a clear boundary
//! between background dispatch and the foreground interaction loop
//! (review Finding #1, Phase 2).

use std::path::PathBuf;
use std::sync::Arc;

use crate::ai::{
    agents::AgentManifest, mcp::SharedMcpClient, skills::SkillManifest,
    tools::task_tools::with_task_entry_by_pid, types::App,
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

    use colored::Colorize;
    for proc in &background_procs {
        println!(
            "\n{} Process {} ({})",
            "[OS Dispatch]".bright_blue().bold(),
            proc.pid,
            proc.name
        );
    }

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
        Option<crate::ai::models::AutoModelFallbackSpec>,
        usize,
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
        // mailbox 非空时 build_background_process_question 走 format_wakeup_prompt，
        // 生成的是系统调度通知（非用户输入），持久化时应标记为 internal_note。
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

        {
            let mut os = app.os.lock().unwrap();
            os.set_current_pid(Some(pid));
            if let Some(p) = os.get_process_mut(pid) {
                if p.history_file.is_none() {
                    p.history_file = Some(process_history_path(&original_history_file, pid));
                }
                let _ = os.process_pending_signals();
            }
        }

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
            task_goal.as_ref().and_then(|goal| goal.auto_model_fallback),
            task_goal.as_ref().map(|goal| goal.spawn_depth).unwrap_or(0),
            is_resume_wakeup,
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
        auto_model_fallback,
        spawn_depth,
        is_resume_wakeup,
    ) in task_specs
    {
        let mut task_app = app.clone();
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
            activate_primary_agent(&mut task_app, agent);
        }
        let next_model = model_override.unwrap_or_else(|| app.current_model.clone());

        let inherit = task_id
            .as_deref()
            .and_then(|tid| crate::ai::tools::task_tools::with_task_entry(tid, |e| e.inherit))
            .unwrap_or_default();
        let (effective_history, task_skills) = resolve_background_subagent_context(
            history_path,
            original_history_file.as_path(),
            skill_manifests,
            task_id.as_deref(),
            inherit,
        );
        task_app.session_history_file = effective_history;
        let task_driver_ctx = runtime_ctx::DriverContext::new(
            task_app.clone(),
            task_mcp.clone(),
            task_skills.clone(),
            agent_manifests.clone(),
        );
        let scope_task_id = task_id.clone().unwrap_or_else(|| format!("pid-{pid}"));
        let parent_history_for_scopes = original_history_file.clone();

        // Slot used by the sub-agent's `finalize_turn` to publish
        // its final assistant text. Cloned into the result-channel
        // payload below so `task_wait` can surface what the
        // sub-agent actually produced (instead of just "completed
        // with empty output").
        let result_slot_for_payload: runtime_ctx::SubagentResultSlot =
            std::sync::Arc::new(tokio::sync::Mutex::new(None));
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
            let captured_output = if result_channel_id.is_some() {
                result_slot_for_payload
                    .lock()
                    .await
                    .clone()
                    .unwrap_or_default()
            } else {
                String::new()
            };
            let mut os = task_os.lock().unwrap();
            os.set_current_pid(Some(pid));
            let publish_task_result = should_publish_subagent_task_result(
                result.is_ok(),
                &captured_output,
                os.get_process(pid).map(|proc| &proc.state),
            );
            if publish_task_result && let Some(result_channel_id) = result_channel_id {
                let payload = serde_json::json!({
                    "status": if result.is_ok() { "completed" } else { "failed" },
                    "output": captured_output,
                    "error": result.as_ref().err().cloned(),
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
            }
            match result {
                Ok(_outcome) => {
                    let outcome = classify_process_outcome(&**os, pid);
                    record_scheduler_outcome(os.as_mut(), pid, outcome);
                    os.increment_turns_used_for(pid);
                    let (should_terminate, termination_result) =
                        finalize_turn_quota(os.as_mut(), pid);
                    if should_terminate {
                        terminate_and_cleanup(os.as_mut(), pid, termination_result, true);
                    } else if os.is_round_robin() {
                        os.set_current_pid(Some(pid));
                        os.requeue_current();
                    }
                }
                Err(err) => {
                    record_scheduler_outcome(os.as_mut(), pid, DispatchOutcomeTag::Failed);
                    terminate_and_cleanup(os.as_mut(), pid, format!("Failed: {}", err), true);
                }
            }
        });

        type BoxedTaskFuture = std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>;
        let mut wrapped: BoxedTaskFuture = Box::pin(inner_fut);
        let persona_memory_path = app.current_persona_memory_file();
        wrapped =
            Box::pin(runtime_ctx::PERSONA_MEMORY_PATH.scope(persona_memory_path.clone(), wrapped));
        wrapped = Box::pin(runtime_ctx::SUBAGENT_RESULT_SLOT.scope(result_slot_for_scope, wrapped));
        if !inherit.memory {
            let mem_path =
                runtime_ctx::make_subagent_memory_path(&parent_history_for_scopes, &scope_task_id);
            // sub-agent 默认私有 memory：finalize 后把白名单条目
            // (is_permanent_memory) 合并回主 memory 文件，让 long-term
            // assets 能跨 task 共享，但普通 task_event 留在私有文件，
            // 不污染主记忆。
            let main_path = persona_memory_path;
            let private_for_merge = mem_path.clone();
            wrapped = Box::pin(runtime_ctx::SUBAGENT_MEMORY_PATH.scope(mem_path, wrapped));
            // 这里包一层 outer future：sub-agent run 完成后 merge。
            // merge_subagent_whitelist 内部用 for_tests_with_path
            // 直接绑定 main_path，绕过 SUBAGENT_MEMORY_PATH override，
            // 避免白名单条目又被写回私有文件（=死循环）。
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
            let scratch_base = parent_history_for_scopes
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."));
            if let Some(scratch) = runtime_ctx::make_subagent_cwd(&scratch_base, &scope_task_id) {
                wrapped = Box::pin(runtime_ctx::SUBAGENT_CWD.scope(scratch, wrapped));
            }
        }
        // 设置子代理嵌套深度，供 `task_spawn` / `task` 在子代理内部
        // 检测递归扇出时使用。
        wrapped = Box::pin(runtime_ctx::SUBAGENT_DEPTH.scope(spawn_depth, wrapped));

        // 计入在途后台子 agent：guard 随 spawned future 一同 move 进任务，
        // 任务结束（正常 / 错误 / panic）时 Drop 自动 dec，避免输入框被永久门控。
        let inflight_guard = BgSubagentGuard::new();
        let guarded_fut = async move {
            let _guard = inflight_guard;
            wrapped.await
        };
        let handle = tokio::spawn(runtime_ctx::DRIVER_CTX.scope(task_driver_ctx, guarded_fut));
        crate::ai::tools::task_tools::set_task_abort_handle(&scope_task_id, handle.abort_handle());
    }
}
