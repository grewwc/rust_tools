use super::*;

pub(super) fn next_task_id() -> String {
    format!("task_{}", Uuid::new_v4().simple())
}

pub(crate) fn encode_os_task_goal(goal: &OsTaskGoal) -> Result<String, String> {
    serde_json::to_string(goal)
        .map(|payload| format!("{TASK_GOAL_PREFIX}{payload}"))
        .map_err(|err| format!("Failed to encode task goal: {err}"))
}

pub(crate) fn is_encoded_task_goal(goal: &str) -> bool {
    goal.starts_with(TASK_GOAL_PREFIX)
}

pub(crate) fn decode_os_task_goal(goal: &str) -> Option<OsTaskGoal> {
    let payload = goal.strip_prefix(TASK_GOAL_PREFIX)?;
    serde_json::from_str(payload).ok()
}

/// Runs a mutable operation on the AIOS kernel.
///
/// Preferred path: take the `SharedKernel` held by the current turn from the `DRIVER_CTX`
/// task-local, so high-frequency paths such as `task_wait` / `task_spawn` reuse the Arc the turn
/// scope already holds, avoiding the extra lock and indirection of the `GLOBAL_OS` global static.
///
/// Fallback path: when the caller is not inside a `DRIVER_CTX` scope (e.g. early driver startup or
/// a unit test invoking the tool from a synchronous context), fall back to `GLOBAL_OS` for backward
/// compatibility.
pub(super) fn with_os_kernel<T>(
    f: impl FnOnce(&mut dyn Kernel) -> Result<T, String>,
) -> Result<T, String> {
    let shared: SharedKernel = match crate::ai::driver::runtime_ctx::try_current() {
        Some(ctx) => ctx.app_proto.os.clone(),
        None => {
            let guard = GLOBAL_OS
                .lock()
                .map_err(|e| format!("Failed to lock AIOS kernel handle: {e}"))?;
            guard
                .as_ref()
                .cloned()
                .ok_or("AIOS kernel is not initialized.".to_string())?
        }
    };
    let mut kernel = shared
        .lock()
        .map_err(|e| format!("Failed to lock AIOS kernel: {e}"))?;
    f(kernel.as_mut())
}

pub(super) fn current_task_owner_pid() -> Result<u64, String> {
    with_os_kernel(|os| {
        os.current_process_id()
            .ok_or("task orchestration requires an active AIOS process context.".to_string())
    })
}

pub(super) fn current_task_owner_pid_opt() -> Option<u64> {
    with_os_kernel(|os| Ok(os.current_process_id()))
        .ok()
        .flatten()
}

pub(super) fn active_foreground_owner_pid(os: &mut dyn Kernel) -> Option<u64> {
    if let Some(pid) = os.current_process_id()
        && os.get_process(pid).is_some_and(|proc| proc.is_foreground)
    {
        return Some(pid);
    }
    os.list_processes()
        .into_iter()
        .find(|proc| proc.is_foreground && !matches!(proc.state, ProcessState::Terminated))
        .map(|proc| proc.pid)
}

pub(super) fn task_entry_owned_by(
    entry: &AsyncTaskEntry,
    session_id: &str,
    owner_pid: u64,
) -> bool {
    entry.session_id == session_id && entry.owner_pid == owner_pid
}

pub(super) fn task_wait_key(
    session_id: &str,
    owner_pid: u64,
    wait_policy: &WaitPolicy,
    task_ids: &[String],
) -> TaskWaitKey {
    let mut normalized = task_ids.to_vec();
    normalized.sort();
    normalized.dedup();
    TaskWaitKey {
        session_id: session_id.to_string(),
        owner_pid,
        wait_policy: wait_policy.into(),
        task_ids: normalized,
    }
}
inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task",
        description: "",

        execute: execute_task,
    }
});

// `task` / `task_spawn` / `task_spawn_batch` / `task_wait` / `task_status`
// may all carry the only visible result of a subagent: the spawn-family arguments (subagent
// prompt / response schema) and return values (the task_id list) are required inputs for the
// later wait/status/integrate calls; once a result is lossy-compressed or LLM-pruned, the main
// agent can lose its grounding on already-finished subtasks. Lossy compression and pruning are
// uniformly banned here; oversized content goes to an overflow stub + file_path instead of being
// reduced to an unrecoverable summary.
inventory::submit!(ToolHistoryPolicyRegistration {
    name: "task",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Never,
        counts_toward_precision_inline_budget: false,
    },
});

pub(crate) fn execute_task(_args: &Value) -> Result<String, String> {
    Err("task is handled by the runtime".to_string())
}

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_spawn",
        description: "",

        execute: execute_task_spawn,
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "task_spawn_batch",
        description: "",

        execute: execute_task_spawn_batch,
    }
});

inventory::submit!(ToolHistoryPolicyRegistration {
    name: "task_spawn",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Never,
        counts_toward_precision_inline_budget: false,
    },
});

inventory::submit!(ToolHistoryPolicyRegistration {
    name: "task_spawn_batch",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::Never,
        counts_toward_precision_inline_budget: false,
    },
});

/// Pre-flight subagent task spec produced from a `task` / `task_spawn` tool
/// call before the kernel actually spawns the new process.
#[derive(Clone)]
pub(crate) struct PreparedSubagentTask {
    pub(crate) description: String,
    pub(crate) prompt: String,
    pub(crate) response_schema: Option<Value>,
    pub(crate) agent_name: String,
    pub(crate) model: String,
    pub(crate) is_model_auto_selected: bool,
    pub(crate) auto_model_fallback: Option<models::AutoModelFallbackSpec>,
    pub(crate) selection_explanation: String,
    pub(crate) inherit: InheritOptions,
}

pub(in crate::ai) fn capped_subagent_manifest(agent: &AgentManifest) -> AgentManifest {
    let mut capped = agent.clone();
    let max_steps = agent
        .max_steps
        .unwrap_or(SUBAGENT_MAX_ITERATIONS)
        .min(SUBAGENT_MAX_ITERATIONS_HARD_CAP)
        .max(1);
    capped.max_steps = Some(max_steps);
    capped
}

pub(super) fn wrap_subagent_prompt(
    description: &str,
    prompt: &str,
    response_schema: Option<&Value>,
) -> String {
    let response_contract = response_schema
        .map(|schema| {
            let schema =
                serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());
            format!(
                "Required response contract:\n\
                 - Return exactly one JSON value matching the schema below.\n\
                 - Do not wrap the JSON in Markdown fences or add prose before or after it.\n\
                 <response_schema>\n{schema}\n</response_schema>\n\n"
            )
        })
        .unwrap_or_default();
    format!(
        "Subagent task: {}\n\n\
         Runtime constraints:\n\
         - Treat this as a bounded leaf task for the parent agent. Do not expand scope beyond the task.\n\
         - Reuse observed evidence and avoid equivalent read/search/list/command variants unless omitted text is needed; prefer one targeted broad call over many small ones.\n\
         - Ground factual claims in observed evidence. For review or diagnosis, trace the relevant path and check likely counter-evidence before reporting a finding.\n\
         - If evidence is incomplete, return a concise partial result separating confirmed conclusions, unresolved hypotheses, missing evidence, and the next verification step.\n\n\
         {}Parent task prompt:\n{}",
        description.trim(),
        response_contract,
        prompt.trim()
    )
}

pub(super) fn parse_response_schema(args: &Value) -> Result<Option<Value>, String> {
    let Some(schema) = args.get("response_schema") else {
        return Ok(None);
    };
    if schema.is_null() {
        return Ok(None);
    }
    if !schema.is_object() {
        return Err("'response_schema' must be a JSON Schema object".to_string());
    }
    jsonschema::validator_for(schema)
        .map_err(|error| format!("Invalid 'response_schema': {error}"))?;
    Ok(Some(schema.clone()))
}

pub(crate) fn validate_subagent_response(
    response_schema: Option<&Value>,
    output: &str,
) -> Result<(), String> {
    let Some(schema) = response_schema else {
        return Ok(());
    };
    let instance: Value = serde_json::from_str(output.trim()).map_err(|error| {
        format!("Subagent response is not valid JSON required by response_schema: {error}")
    })?;
    let validator = jsonschema::validator_for(schema)
        .map_err(|error| format!("Invalid response_schema: {error}"))?;
    validator
        .validate(&instance)
        .map_err(|error| format!("Subagent response did not match response_schema: {error}"))
}

/// Parse and validate a `task` / `task_spawn` tool call payload, run subagent
/// auto-selection, and resolve the model. Used both by the async `task_spawn`
/// path and by the synchronous `task` interception in the driver.
pub(crate) fn prepare_subagent_task(args: &Value) -> Result<PreparedSubagentTask, String> {
    let description = args["description"]
        .as_str()
        .ok_or("Missing 'description' parameter")?;
    let prompt = args["prompt"]
        .as_str()
        .ok_or("Missing 'prompt' parameter")?;
    let agent = args["agent"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let model_override = args["model"]
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty());

    if description.trim().is_empty() {
        return Err("description cannot be empty".to_string());
    }
    if prompt.trim().is_empty() {
        return Err("prompt cannot be empty".to_string());
    }

    let inherit = InheritOptions::from_value(&args["inherit"])?;
    let response_schema = parse_response_schema(args)?;

    // Prefer the agent_manifests cached in DRIVER_CTX, so each task_spawn does not re-read the
    // disk. When not inside a DRIVER_CTX scope (rare, e.g. unit tests), fall back to
    // load_all_agents().
    let cached = crate::ai::driver::runtime_ctx::try_current();
    let owned_fallback;
    let all_agents: &[AgentManifest] = if let Some(ref ctx) = cached {
        ctx.agent_manifests.as_slice()
    } else {
        owned_fallback = agents::load_all_agents();
        &owned_fallback
    };
    let selected = select_subagent(all_agents, agent, description, prompt)?;
    let (selected_model, is_model_auto_selected, auto_model_fallback, inherited_parent_model) =
        if let Some(model_override) = model_override {
            (models::determine_model(model_override), false, None, false)
        } else {
            let parent_model = cached
                .as_ref()
                .map(|ctx| ctx.app_proto.current_model.as_str());
            let choice = models::choose_model_for_subagent(
                parent_model,
                selected.agent,
                description,
                prompt,
            );
            (
                choice.model,
                choice.is_auto_selected,
                choice.fallback,
                !choice.is_auto_selected,
            )
        };
    let selection_explanation = build_selection_explanation(
        &selected,
        &selected_model,
        model_override,
        inherited_parent_model,
    );

    Ok(PreparedSubagentTask {
        description: description.to_string(),
        prompt: wrap_subagent_prompt(description, prompt, response_schema.as_ref()),
        response_schema,
        agent_name: selected.agent.name.clone(),
        model: selected_model,
        is_model_auto_selected,
        auto_model_fallback,
        selection_explanation,
        inherit,
    })
}

pub(crate) struct SpawnedSubagentTask {
    pub(crate) task_id: String,
    pub(crate) pid: u64,
    pub(crate) result_channel_id: u64,
    pub(crate) completion_futex_addr: FutexAddr,
}

/// Spawn a subagent kernel process and register it in `TASK_REGISTRY`. The
/// returned handle exposes the IPC channel + futex that the caller can wait
/// on. Used by both `task_spawn` (async) and the synchronous `task` runtime
/// interception path.
pub(crate) fn spawn_subagent_kernel_task(
    prepared: &PreparedSubagentTask,
) -> Result<SpawnedSubagentTask, String> {
    spawn_subagent_kernel_task_attempt(prepared, None)
}

pub(super) fn spawn_subagent_kernel_task_attempt(
    prepared: &PreparedSubagentTask,
    retry_of: Option<&str>,
) -> Result<SpawnedSubagentTask, String> {
    let parent_depth = crate::ai::driver::runtime_ctx::current_subagent_depth();
    let child_depth = parent_depth + 1;
    if child_depth > MAX_SUBAGENT_SPAWN_DEPTH {
        return Err(format!(
            "Subagent nesting depth {} exceeds maximum {}. \
             The current agent is already a nested subagent; further delegation \
             would risk unbounded recursion. Execute the work directly instead.",
            child_depth, MAX_SUBAGENT_SPAWN_DEPTH,
        ));
    }
    {
        let registry = TASK_REGISTRY.lock().unwrap();
        if registry.len() >= MAX_TASK_REGISTRY_SIZE {
            return Err(format!(
                "Subagent task registry is full ({MAX_TASK_REGISTRY_SIZE}). \
                 Collect and integrate existing task results before spawning another task."
            ));
        }
    }
    let task_id = next_task_id();
    let (owner_pid, pid, result_channel_id, completion_futex_addr) = with_os_kernel(|os| {
        let parent_pid = os
            .current_process_id()
            .ok_or("subagent task requires an active AIOS process context.".to_string())?;
        let result_channel = os.channel_create_tagged_with_holders(
            Some(parent_pid),
            1,
            format!("task_result:{task_id}"),
            ChannelOwnerTag::TaskResult,
            vec![
                "task_result.producer".to_string(),
                "task_result.consumer".to_string(),
            ],
        );
        let completion_futex = os.futex_create(0, format!("task_completion:{task_id}"));
        let process_goal = encode_os_task_goal(&OsTaskGoal {
            task_id: task_id.clone(),
            result_channel_id: result_channel.raw(),
            completion_futex_addr: completion_futex.raw(),
            description: prepared.description.clone(),
            prompt: prepared.prompt.clone(),
            agent_name: prepared.agent_name.clone(),
            model: prepared.model.clone(),
            is_model_auto_selected: prepared.is_model_auto_selected,
            auto_model_fallback: prepared.auto_model_fallback,
            selection_explanation: prepared.selection_explanation.clone(),
            spawn_depth: child_depth,
            response_schema: prepared.response_schema.clone(),
        })?;
        let pid = os.spawn(
            Some(parent_pid),
            prepared.agent_name.clone(),
            process_goal,
            DEFAULT_TASK_PRIORITY,
            DEFAULT_TASK_QUOTA_TURNS,
            None,
            None,
        )?;
        Ok((parent_pid, pid, result_channel.raw(), completion_futex))
    })?;

    {
        let mut registry = TASK_REGISTRY.lock().unwrap();
        registry.insert(
            task_id.clone(),
            AsyncTaskEntry {
                session_id: crate::ai::driver::runtime_ctx::current_session_id_or_empty(),
                result_observed: false,
                owner_pid,
                pid,
                result_channel_id,
                completion_futex_addr,
                description: prepared.description.clone(),
                agent_name: prepared.agent_name.clone(),
                model: prepared.model.clone(),
                is_model_auto_selected: prepared.is_model_auto_selected,
                auto_model_fallback: prepared.auto_model_fallback,
                selection_explanation: prepared.selection_explanation.clone(),
                inherit: prepared.inherit,
                started_at: Instant::now(),
                last_progress_notification_at: None,
                last_progress_persisted_at: None,
                abort_handle: None,
                cancel_stream: Arc::new(AtomicBool::new(false)),
            },
        );
    }
    TASK_PROGRESS_REGISTRY.lock().unwrap().insert(
        task_id.clone(),
        crate::ai::driver::runtime_ctx::new_subagent_progress_slot(),
    );
    register_retry_spec(
        &task_id,
        crate::ai::driver::runtime_ctx::current_session_id_or_empty(),
        owner_pid,
        prepared.clone(),
        retry_of.unwrap_or(&task_id).to_string(),
    );
    crate::ai::driver::notify_scheduler_after(SUBAGENT_WALL_CLOCK_TIMEOUT);

    Ok(SpawnedSubagentTask {
        task_id,
        pid,
        result_channel_id,
        completion_futex_addr,
    })
}
pub(crate) fn execute_task_spawn(args: &Value) -> Result<String, String> {
    ensure_top_level_task_orchestration("task_spawn")?;
    let prepared = prepare_subagent_task(args)?;
    let spawned = spawn_subagent_kernel_task(&prepared)?;
    record_last_spawn_batch(vec![spawned.task_id.clone()]);
    record_subagent_spawn_audit(&spawned.task_id, &prepared);

    Ok(format!(
        "Task spawned: task_id={}, pid={}, agent={}, model={}, inherit={}\nContinue independent parent-side work now. Do not call task_wait immediately unless the parent is blocked on this result; use task_status for a non-blocking snapshot.",
        spawned.task_id,
        spawned.pid,
        prepared.agent_name,
        prepared.model,
        prepared.inherit.describe()
    ))
}

pub(crate) fn execute_task_spawn_batch(args: &Value) -> Result<String, String> {
    ensure_top_level_task_orchestration("task_spawn_batch")?;
    let tasks = args
        .get("tasks")
        .and_then(Value::as_array)
        .ok_or_else(|| "task_spawn_batch requires a 'tasks' array".to_string())?;
    if tasks.is_empty() {
        return Err("task_spawn_batch requires at least one task".to_string());
    }
    if tasks.len() > MAX_SUBAGENT_SPAWN_BATCH_SIZE {
        return Err(format!(
            "task_spawn_batch accepts at most {MAX_SUBAGENT_SPAWN_BATCH_SIZE} tasks per call"
        ));
    }

    // Complete the whole batch preflight first, so earlier children are not already started when a
    // later entry has invalid arguments.
    let prepared = tasks
        .iter()
        .enumerate()
        .map(|(index, task)| {
            prepare_subagent_task(task)
                .map_err(|error| format!("task_spawn_batch tasks[{index}]: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut spawned_count = 0usize;
    let mut entries = Vec::with_capacity(prepared.len());
    let mut spawned_ids = Vec::with_capacity(prepared.len());
    for (index, task) in prepared.iter().enumerate() {
        match spawn_subagent_kernel_task(task) {
            Ok(spawned) => {
                spawned_count += 1;
                spawned_ids.push(spawned.task_id.clone());
                record_subagent_spawn_audit(&spawned.task_id, task);
                entries.push(serde_json::json!({
                    "index": index,
                    "status": "spawned",
                    "task_id": spawned.task_id,
                    "pid": spawned.pid,
                    "agent": task.agent_name,
                    "model": task.model,
                    "inherit": task.inherit.describe(),
                }));
            }
            Err(error) => entries.push(serde_json::json!({
                "index": index,
                "status": "failed",
                "error": error,
                "agent": task.agent_name,
                "model": task.model,
                "inherit": task.inherit.describe(),
            })),
        }
    }
    if !spawned_ids.is_empty() {
        record_last_spawn_batch(spawned_ids);
    }

    Ok(serde_json::to_string_pretty(&serde_json::json!({
        "spawned": spawned_count,
        "failed": entries.len() - spawned_count,
        "tasks": entries,
        "next": "Continue independent parent-side work; use task_status for snapshots and task_wait only when blocked on results."
    }))
    .expect("serializing task_spawn_batch result cannot fail"))
}
