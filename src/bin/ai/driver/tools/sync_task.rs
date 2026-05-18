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
use std::time::Instant;

use serde_json::Value;

use crate::ai::{
    agents,
    driver::{runtime_ctx, turn_runtime},
    tools::task_tools,
    types::ToolResult,
};

use super::super::runtime_ctx::DriverContext;

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

    if !inherit.memory {
        let mem_path =
            runtime_ctx::make_subagent_memory_path(&parent_history_path, &task_id);
        wrapped = Box::pin(runtime_ctx::SUBAGENT_MEMORY_PATH.scope(mem_path, wrapped));
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

    let join_result: Result<Result<(), String>, String> =
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                rx.await
                    .map_err(|e| format!("subagent task channel closed before result: {e}"))
            })
        });

    let duration = started.elapsed();
    let elapsed_secs = duration.as_secs_f64();

    match join_result {
        Ok(Ok(())) => Ok(ToolResult {
            tool_call_id: tool_call_id.to_string(),
            content: format!(
                "[Task: {} via {} @ {}] COMPLETED after {:.1}s\n{}",
                log_description,
                log_agent_name,
                log_model,
                elapsed_secs,
                log_selection_explanation,
            ),
        }),
        Ok(Err(err)) => Err(format!(
            "[Task: {} via {} @ {}] FAILED after {:.1}s\n{}\n{}",
            log_description,
            log_agent_name,
            log_model,
            elapsed_secs,
            log_selection_explanation,
            err,
        )),
        Err(err) => Err(format!(
            "[Task: {} via {} @ {}] internal failure after {:.1}s: {}",
            log_description, log_agent_name, log_model, elapsed_secs, err,
        )),
    }
}

fn subagent_history_path(base: &std::path::Path, task_id: &str) -> PathBuf {
    let file_name = base
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| format!("{name}.subagent-{task_id}"))
        .unwrap_or_else(|| format!("session.subagent-{task_id}"));
    base.with_file_name(file_name)
}
