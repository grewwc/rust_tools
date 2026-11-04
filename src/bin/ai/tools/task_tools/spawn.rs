use super::*;
use crate::ai::driver::turn_runtime::is_evidence_gated_audit_agent;

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

// `task` / `task_spawn` / `task_wait` / `task_status`
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
        prune: ToolPrunePolicy::AfterIntegration,
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

inventory::submit!(ToolHistoryPolicyRegistration {
    name: "task_spawn",
    policy: ToolHistoryPolicy {
        lossy_compress: ToolLossyCompressPolicy::Never,
        prune: ToolPrunePolicy::AfterIntegration,
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
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) is_model_auto_selected: bool,
    pub(crate) auto_model_fallback: Option<models::AutoModelFallbackSpec>,
    pub(crate) selection_explanation: String,
    pub(crate) inherit: InheritOptions,
    pub(crate) backend: SubagentBackend,
    pub(crate) show_backend_launch_tag: bool,
    /// Frozen execution model for TraeCLI; ignored by the kernel runner.
    pub(crate) backend_model: Option<String>,
    /// Optional resume target for external backends (a previous task_id or a
    /// raw backend session id); resolved at dispatch time, when its queued
    /// side-notes are prepended to the follow-up prompt. Ignored by kernel.
    pub(crate) backend_continue_from: Option<String>,
    /// Original images attached to the parent turn. The kernel path attaches
    /// them to the child's first message; external backends receive them as
    /// image arguments and absolute paths in the prompt.
    pub(crate) image_files: Vec<String>,
}

impl PreparedSubagentTask {
    pub(super) fn execution_model(&self) -> &str {
        if self.backend.is_traecli() {
            self.backend_model
                .as_deref()
                .expect("TraeCLI execution model is resolved during task preparation")
        } else {
            &self.model
        }
    }
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
    let response_contract = match response_schema {
        Some(schema) => {
            let schema =
                serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());
            format!(
                include_str!("prompts/subagent_response_schema.md"),
                schema = schema
            )
        }
        // Without an explicit schema the parent still needs a stable, auditable
        // shape (evidence + unresolved gaps), not free prose. This is guidance
        // only: no default JSON enforcement, which would break agents that
        // legitimately answer in prose or with an envelope.
        None => include_str!("prompts/subagent_default_response_contract.md").to_string(),
    };
    format!(
        include_str!("prompts/subagent_wrap.md"),
        task_description = description.trim(),
        response_contract = response_contract,
        parent_prompt = prompt.trim()
    )
}

const SUBAGENT_HANDOFF_PREFIX: &str = "<subagent_handoff>";
const SUBAGENT_HANDOFF_SUFFIX: &str = "</subagent_handoff>";

pub(crate) fn parse_subagent_handoff(output: &str) -> Result<Option<SubagentHandoff>, String> {
    let trimmed = output.trim();
    if !trimmed.starts_with(SUBAGENT_HANDOFF_PREFIX) {
        return Ok(None);
    }
    if !trimmed.ends_with(SUBAGENT_HANDOFF_SUFFIX) {
        return Err("subagent handoff is missing its closing tag".to_string());
    }
    let body =
        &trimmed[SUBAGENT_HANDOFF_PREFIX.len()..trimmed.len() - SUBAGENT_HANDOFF_SUFFIX.len()];
    let mut handoff: SubagentHandoff = serde_json::from_str(body)
        .map_err(|error| format!("subagent handoff is not valid JSON: {error}"))?;
    handoff.reason = handoff.reason.trim().to_string();
    if handoff.reason.is_empty() {
        return Err("subagent handoff reason must not be empty".to_string());
    }
    if handoff.reason.chars().count() > 2_000 {
        return Err("subagent handoff reason exceeds 2000 characters".to_string());
    }
    handoff.suggested_action = handoff
        .suggested_action
        .map(|action| action.trim().to_string())
        .filter(|action| !action.is_empty());
    if handoff
        .suggested_action
        .as_ref()
        .is_some_and(|action| action.chars().count() > 2_000)
    {
        return Err("subagent handoff suggested_action exceeds 2000 characters".to_string());
    }
    Ok(Some(handoff))
}

/// External backends cannot see the parent's message content. Carry the
/// original image paths in the task prompt so the child can attach them on the
/// first request (TraeCLI `-i`) and re-open them on any later turn.
fn append_image_files_hint(mut prompt: String, image_files: &[String]) -> String {
    if image_files.is_empty() {
        return prompt;
    }
    prompt.push_str("\n\n## Original images\n");
    prompt.push_str("The parent turn attached these original image files:\n");
    for path in image_files {
        prompt.push_str("- ");
        prompt.push_str(path);
        prompt.push('\n');
    }
    prompt.push_str(
        "The first request includes these images. Use your image viewer on a path \
         (`read_file` / `view_image`) whenever you need the original again.",
    );
    prompt
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
    let backend_model = parse_backend_model(args.get("backend_model"))?;
    let backend_continue_from = parse_backend_continue_from(args.get("backend_continue_from"))?;
    let explicit_backend = SubagentBackend::parse_optional(args.get("backend"))?;
    let cached = crate::ai::driver::runtime_ctx::try_current();
    let session_backend = cached
        .as_ref()
        .map(|ctx| ctx.app_proto.session_default_subagent_backend)
        .unwrap_or_default();
    let backend = explicit_backend.unwrap_or_else(|| {
        if backend_continue_from.is_some() {
            SubagentBackend::TraeCli
        } else {
            session_backend
        }
    });
    // The worker model belongs to the external backend, never to the main
    // agent's model registry. Per-call choices override the session pick.
    let backend_model = backend_model.or_else(|| {
        if backend.is_traecli() {
            cached
                .as_ref()
                .and_then(|ctx| ctx.app_proto.session_default_subagent_model.clone())
        } else {
            None
        }
    });
    let backend_model = if backend.is_traecli() {
        Some(traecli_model_for(backend_model.as_deref()))
    } else {
        backend_model
    };
    let show_backend_launch_tag = explicit_backend.is_some()
        || backend.is_traecli()
        || matches!(session_backend, SubagentBackend::TraeCli);
    if backend_continue_from.is_some() && matches!(explicit_backend, Some(SubagentBackend::Kernel))
    {
        return Err("'backend_continue_from' needs backend 'traecli'".to_string());
    }

    // Prefer the agent_manifests cached in DRIVER_CTX, so each task_spawn does not re-read the
    // disk. When not inside a DRIVER_CTX scope (rare, e.g. unit tests), fall back to
    // load_all_agents().
    // Original images belong to the parent turn, not to the agent manifest.
    // Snapshot them here so the background dispatch loop and external backend
    // keep using this task's images after the foreground turn has moved on.
    let image_files = cached
        .as_ref()
        .map(|ctx| ctx.app_proto.attached_image_files.clone())
        .unwrap_or_default();
    let owned_fallback;
    let all_agents: &[AgentManifest] = if let Some(ref ctx) = cached {
        ctx.agent_manifests.as_slice()
    } else {
        owned_fallback = agents::load_all_agents();
        &owned_fallback
    };
    let selected = select_subagent(all_agents, agent, description, prompt)?;
    if response_schema.is_some() && is_evidence_gated_audit_agent(&selected.agent.name) {
        // The review agents publish a machine-checked `<audit_report>` envelope as their
        // final response; a `response_schema` contract on the same body contradicts it.
        return Err(format!(
            "agent `{}` publishes a machine-checked `<audit_report>` envelope; call it without \
             `response_schema` and read the envelope from the returned result",
            selected.agent.name,
        ));
    }
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
    let reasoning_effort = if backend.is_traecli() {
        None
    } else {
        Some(
            crate::ai::request::reasoning_effort_display_label_for_subagent(
                &selected.agent.name,
                &selected_model,
            )
            .to_string(),
        )
    };

    let prompt = wrap_subagent_prompt(description, prompt, response_schema.as_ref());
    let prompt = if backend.is_traecli() {
        append_image_files_hint(prompt, &image_files)
    } else {
        prompt
    };

    Ok(PreparedSubagentTask {
        description: description.to_string(),
        prompt,
        response_schema,
        agent_name: selected.agent.name.clone(),
        model: selected_model,
        reasoning_effort,
        is_model_auto_selected,
        auto_model_fallback,
        selection_explanation,
        inherit,
        backend,
        show_backend_launch_tag,
        backend_model,
        backend_continue_from,
        image_files,
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

/// IPC objects created before `os.spawn` succeeds. Goal encoding or spawn can fail, and no
/// collector ever learns about such a task, so the guard releases the channel and futex on the
/// error path instead of leaking them.
struct PendingTaskIpc {
    result_channel: ChannelId,
    completion_futex: FutexAddr,
    committed: bool,
}

impl Drop for PendingTaskIpc {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        let _ = with_os_kernel(|os| {
            let _ = os.channel_close(None, self.result_channel);
            let _ = os.channel_release_named(self.result_channel, "task_result.consumer");
            let _ = os.channel_release_named(self.result_channel, "task_result.producer");
            let _ = os.channel_destroy(None, self.result_channel);
            let _ = os.futex_destroy(self.completion_futex);
            Ok(())
        });
    }
}

pub(super) fn spawn_subagent_kernel_task_attempt(
    prepared: &PreparedSubagentTask,
    retry_of: Option<&str>,
) -> Result<SpawnedSubagentTask, String> {
    let parent_depth = crate::ai::driver::runtime_ctx::current_subagent_depth();
    let parent_agent_team_mode = is_agent_team_mode();
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
        let registry = lock_task_registry();
        if registry.len() >= MAX_TASK_REGISTRY_SIZE {
            return Err(format!(
                "Subagent task registry is full ({MAX_TASK_REGISTRY_SIZE}). \
                 Collect and integrate existing task results before spawning another task."
            ));
        }
    }
    if prepared.backend.is_traecli() && !traecli_binary_available() {
        return Err(format!(
            "traecli backend requested but binary '{}' was not found; install traecli or set TRAECLI_BIN",
            traecli_binary()
        ));
    }
    let task_id = next_task_id();
    let mut pending_ipc = None;
    let spawned = with_os_kernel(|os| {
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
        pending_ipc = Some(PendingTaskIpc {
            result_channel,
            completion_futex,
            committed: false,
        });
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
            parent_agent_team_mode,
            response_schema: prepared.response_schema.clone(),
            backend: prepared
                .show_backend_launch_tag
                .then(|| prepared.backend.as_str().to_string()),
            backend_model: prepared.backend_model.clone(),
            backend_continue_from: prepared.backend_continue_from.clone(),
            // Snapshot now: background dispatch may run after this turn ends,
            // when the live driver context no longer carries the guard state.
            // Kernel tasks never read it, so their payload keeps the old shape.
            active_skill_names: if prepared.backend.is_traecli() {
                crate::ai::driver::runtime_ctx::active_skill_names()
            } else {
                Vec::new()
            },
            image_files: prepared.image_files.clone(),
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
    if let Some(ipc) = pending_ipc.as_mut() {
        ipc.committed = true;
    }
    let (owner_pid, pid, result_channel_id, completion_futex_addr) = spawned;

    {
        let mut registry = lock_task_registry();
        registry.insert(
            task_id.clone(),
            AsyncTaskEntry {
                session_id: crate::ai::driver::runtime_ctx::current_session_id_or_empty(),
                owner_pid,
                pid,
                result_channel_id,
                completion_futex_addr,
                description: prepared.description.clone(),
                agent_name: prepared.agent_name.clone(),
                backend: prepared.backend.is_traecli().then_some(prepared.backend),
                // External backends run on their own model, so records and the
                // status line report that instead of the unused kernel-side
                // selection. Kernel entries keep the provider model.
                model: prepared.execution_model().to_string(),
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
    if !prepared.backend.is_traecli() {
        set_task_reasoning_effort(&task_id, prepared.reasoning_effort.clone());
    }
    lock_unpoisoned(&TASK_PROGRESS_REGISTRY).insert(
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
    let tasks = args
        .get("tasks")
        .and_then(Value::as_array)
        .ok_or_else(|| "task_spawn requires a 'tasks' array".to_string())?;
    if tasks.is_empty() {
        return Err("task_spawn requires at least one task".to_string());
    }
    if tasks.len() > MAX_SUBAGENT_SPAWN_BATCH_SIZE {
        return Err(format!(
            "task_spawn accepts at most {MAX_SUBAGENT_SPAWN_BATCH_SIZE} tasks per call"
        ));
    }

    // Complete the whole batch preflight first, so earlier children are not already started when a
    // later entry has invalid arguments.
    let prepared = tasks
        .iter()
        .enumerate()
        .map(|(index, task)| {
            prepare_subagent_task(task)
                .map_err(|error| format!("task_spawn tasks[{index}]: {error}"))
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
                let mut entry = serde_json::json!({
                    "index": index,
                    "status": "spawned",
                    "task_id": spawned.task_id,
                    "pid": spawned.pid,
                    "agent": task.agent_name,
                    "model": task.execution_model(),
                    "inherit": task.inherit.describe(),
                });
                if let Some(effort) = &task.reasoning_effort {
                    entry["reasoning_effort"] = serde_json::json!(effort);
                }
                if task.show_backend_launch_tag {
                    entry["backend"] = serde_json::json!(task.backend.as_str());
                }
                if let Some(progress) = current_task_progress(&spawned.task_id) {
                    entry["progress"] = serde_json::json!(progress);
                }
                entries.push(entry);
            }
            Err(error) => {
                let mut entry = serde_json::json!({
                    "index": index,
                    "status": "failed",
                    "error": error,
                    "agent": task.agent_name,
                    "model": task.execution_model(),
                    "inherit": task.inherit.describe(),
                });
                if let Some(effort) = &task.reasoning_effort {
                    entry["reasoning_effort"] = serde_json::json!(effort);
                }
                if task.show_backend_launch_tag {
                    entry["backend"] = serde_json::json!(task.backend.as_str());
                }
                entries.push(entry);
            }
        }
    }
    if !spawned_ids.is_empty() {
        record_last_spawn_batch(spawned_ids);
    }
    if spawned_count == 0 {
        let details = entries
            .iter()
            .filter_map(|entry| entry.get("error").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(format!(
            "task_spawn failed to spawn all {} task(s): {details}",
            entries.len()
        ));
    }

    Ok(serde_json::to_string_pretty(&serde_json::json!({
        "spawned": spawned_count,
        "failed": entries.len() - spawned_count,
        "tasks": entries,
        "next": "Use task_status for non-blocking progress (phase, updates, last activity). Do not duplicate a running branch; use task_wait only when blocked on a result, then task_integrate."
    }))
    .expect("serializing task_spawn result cannot fail"))
}
