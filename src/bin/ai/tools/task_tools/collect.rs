use super::*;

/// After a successful spawn, writes a placeholder audit record (delivered_at=0); when the result
/// is delivered, `record_delivered_task_evidence` overwrites the status fields for the same
/// task_id, so the ledger can always answer "whether the current agent called a subagent, when,
/// and with which agent/model". The audit write is best-effort: failure does not block the spawn
/// itself.
pub(super) fn record_subagent_spawn_audit(task_id: &str, prepared: &PreparedSubagentTask) {
    let Some(context) = crate::ai::driver::runtime_ctx::try_current() else {
        return;
    };
    let _ = crate::ai::history::record_task_spawn_audit(
        context.app_proto.config.history_file.as_path(),
        &context.app_proto.session_id,
        task_id,
        &prepared.description,
        &prepared.agent_name,
        prepared.execution_model(),
    );
}

/// Persist a terminal status for a task whose result is not collected.
///
/// Cancellation, wall-clock reaping, and turn wrap-up know the terminal state before
/// `task_wait` runs. Recording it keeps `task_audit` accurate when the turn ends first;
/// the ledger row remains undelivered, so it is not treated as pending evidence.
pub(super) fn record_uncollected_terminal_status(session_id: &str, task_id: &str, status: &str) {
    if session_id.is_empty() {
        return;
    }
    let Some(context) = crate::ai::driver::runtime_ctx::try_current() else {
        return;
    };
    let _ = crate::ai::history::record_uncollected_task_terminal_status(
        context.app_proto.config.history_file.as_path(),
        session_id,
        task_id,
        status,
    );
}

pub(super) fn persist_rendered_task_evidence(
    task_id: &str,
    entry: &AsyncTaskEntry,
    status: &str,
    rendered: &str,
) -> Result<(), String> {
    let Some(history_file) = task_evidence_history_file(entry) else {
        return Ok(());
    };
    crate::ai::history::record_delivered_task_evidence(
        &history_file,
        &entry.session_id,
        crate::ai::history::DeliveredTaskEvidence {
            task_id,
            description: &entry.description,
            agent_name: &entry.agent_name,
            model: &entry.model,
            status,
            payload: rendered,
        },
    )
    .map_err(|error| format!("failed to persist task evidence for {task_id}: {error}"))
}

fn task_evidence_history_file(entry: &AsyncTaskEntry) -> Option<PathBuf> {
    entry
        .evidence_history_file
        .clone()
        .or_else(|| {
            crate::ai::driver::runtime_ctx::try_current()
                .map(|context| context.app_proto.config.history_file.clone())
        })
        .filter(|path| !path.as_os_str().is_empty() && !entry.session_id.is_empty())
}

/// Every terminal producer commits the full result before attempting IPC. Publication failures
/// retain a recovery payload in the parent ledger and return a visible error to the driver.
pub(crate) fn publish_task_result(
    os: &mut dyn Kernel,
    task_id: Option<&str>,
    pid: u64,
    result_channel_id: u64,
    payload: String,
) -> Result<(), String> {
    let task = task_id.and_then(|task_id| {
        with_task_entry(task_id, |entry| entry.clone())
            .filter(|entry| entry.pid == pid && entry.result_channel_id == result_channel_id)
            .map(|entry| (task_id.to_string(), entry))
    });
    let Some((task_id, entry)) = task else {
        let channel = ChannelId(result_channel_id);
        os.channel_send(Some(pid), channel, payload)?;
        os.channel_close(None, channel)?;
        if os.channel_meta(channel).is_some_and(|metadata| {
            metadata
                .ref_holders
                .iter()
                .any(|holder| holder == "task_result.producer")
        }) {
            os.channel_release_named(channel, "task_result.producer")?;
        }
        return Ok(());
    };
    let result: StoredTaskResult = serde_json::from_str(&payload).map_err(|error| {
        format!("cannot publish task {task_id}: invalid terminal result: {error}")
    })?;
    let history_file = task_evidence_history_file(&entry)
        .filter(|path| !path.as_os_str().is_empty() && !entry.session_id.is_empty());
    let persistence = history_file.as_ref().map(|history_file| {
        let rendered = format_task_result_with_id(&task_id, &entry, result.clone());
        crate::ai::history::record_published_task_evidence(
            history_file,
            &entry.session_id,
            crate::ai::history::DeliveredTaskEvidence {
                task_id: &task_id,
                description: &entry.description,
                agent_name: &entry.agent_name,
                model: &entry.model,
                status: &result.status,
                payload: &rendered,
            },
        )
    });
    if matches!(&persistence, Some(Ok(false))) {
        // A terminal publication already owns this slot, including one collected earlier.
        return Ok(());
    }
    let mut errors = Vec::new();
    if let Some(Err(error)) = &persistence {
        errors.push(format!("durable publication failed: {error}"));
    }
    let sent = match os.channel_send(Some(pid), ChannelId(result_channel_id), payload) {
        Ok(()) => true,
        Err(error) => {
            let error = error.trim_end_matches('.');
            let saved = matches!(&persistence, Some(Ok(true)));
            let recovery = if saved {
                "The full result is saved in the parent ledger; task_wait/task_status can collect it."
            } else {
                "The result could not be saved in the parent ledger; retain the child history for recovery."
            };
            let warning = format!(
                "[Task result delivery warning] IPC publication failed: {error}. {recovery}"
            );
            if saved
                && let Some(history_file) = &history_file
                && let Err(error) = crate::ai::history::record_task_publication_warning(
                    history_file,
                    &entry.session_id,
                    &task_id,
                    &warning,
                )
            {
                errors.push(format!("failed to save IPC delivery warning: {error}"));
            }
            errors.push(warning);
            false
        }
    };
    if (sent || matches!(&persistence, Some(Ok(true))))
        && os.channel_meta(ChannelId(result_channel_id)).is_some()
    {
        // Closing is a host lifecycle operation; the producer is not the consumer owner.
        if let Err(error) = os.channel_close(None, ChannelId(result_channel_id)) {
            errors.push(format!("failed to close result channel: {error}"));
        }
        if let Err(error) =
            os.channel_release_named(ChannelId(result_channel_id), "task_result.producer")
        {
            errors.push(format!("failed to release result producer: {error}"));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!("task_id={task_id}: {}", errors.join("; ")))
    }
}

/// Reads the persisted evidence state (status + payload) for the given task_id in the current
/// session. Returns None when there is no driver context or nothing has been persisted (a lenient
/// semantic consistent with the persisting side).
pub(super) fn read_persisted_task_evidence_status(
    session_id: &str,
    task_id: &str,
) -> Result<Option<(String, String)>, String> {
    let Some(context) = crate::ai::driver::runtime_ctx::try_current() else {
        return Ok(None);
    };
    crate::ai::history::read_task_evidence_status_payload(
        context.app_proto.config.history_file.as_path(),
        session_id,
        task_id,
    )
    .map_err(|error| format!("failed to read task evidence for {task_id}: {error}"))
}

/// One terminal result drained from a subagent's result channel.
pub(super) struct DrainedTaskResult {
    pub(super) result: StoredTaskResult,
    pub(super) rendered: String,
}

/// Drain a terminal result: persist evidence first, consume the channel, then release the IPC
/// resources and terminate the kernel process. Returns `Ok(None)` while the task is still running.
///
/// The registry entry is deliberately left untouched. Callers remove it after the kernel closure
/// returns, including on the error path, so a failure while collecting a later task in the same
/// batch cannot strand an entry whose channel has already been destroyed.
pub(super) fn drain_task_result(
    os: &mut dyn Kernel,
    task_id: &str,
    entry: &AsyncTaskEntry,
    cleanup_reason: &str,
) -> Result<Option<DrainedTaskResult>, String> {
    let Some(result) = read_task_result(os, entry.result_channel_id, false)? else {
        if let Some(history_file) = task_evidence_history_file(entry)
            && let Some((status, rendered, delivered)) =
                crate::ai::history::read_available_task_evidence_status_payload(
                    &history_file,
                    &entry.session_id,
                    task_id,
                )
                .map_err(|error| {
                    format!("failed to read available task evidence for {task_id}: {error}")
                })?
        {
            // Durable terminal truth takes precedence over an AIOS process that has not been
            // torn down yet, including a collection interrupted after its ledger commit.
            if !delivered {
                persist_rendered_task_evidence(task_id, entry, &status, &rendered)?;
            }
            mark_task_retry_status(task_id, &status);
            lock_unpoisoned(&TASK_PROGRESS_REGISTRY).take(&task_id.to_string());
            cleanup_collected_task(os, entry, cleanup_reason);
            return Ok(Some(DrainedTaskResult {
                result: StoredTaskResult {
                    status,
                    output: rendered.clone(),
                    error: None,
                    progress: None,
                    backend_session: None,
                },
                rendered,
            }));
        }
        if is_task_pending(os, entry.pid)? {
            return Ok(None);
        }
        let result = StoredTaskResult {
            status: TaskTerminalStatus::Failed.as_str().to_string(),
            output: String::new(),
            error: Some(format!(
                "Subagent process pid={} terminated without publishing any output.",
                entry.pid
            )),
            progress: current_task_progress(task_id),
            backend_session: backend_session::record_session(
                task_backend(entry.backend).descriptor().id,
                task_id,
                None,
            ),
        };
        let rendered = format_task_result_with_id(task_id, entry, result.clone());
        persist_rendered_task_evidence(task_id, entry, &result.status, &rendered)?;
        mark_task_retry_status(task_id, &result.status);
        lock_unpoisoned(&TASK_PROGRESS_REGISTRY).take(&task_id.to_string());
        cleanup_collected_task(os, entry, cleanup_reason);
        return Ok(Some(DrainedTaskResult { result, rendered }));
    };

    let rendered = format_task_result_with_id(task_id, entry, result.clone());
    persist_rendered_task_evidence(task_id, entry, &result.status, &rendered)?;
    if read_task_result(os, entry.result_channel_id, true)?.is_none() {
        return Err(format!(
            "task result for {task_id} disappeared after durable persistence"
        ));
    }
    mark_task_retry_status(task_id, &result.status);
    lock_unpoisoned(&TASK_PROGRESS_REGISTRY).take(&task_id.to_string());
    cleanup_collected_task(os, entry, cleanup_reason);
    Ok(Some(DrainedTaskResult { result, rendered }))
}

pub(super) enum OwnedTaskPoll {
    Pending {
        state: String,
    },
    Terminal {
        result: StoredTaskResult,
        rendered: String,
    },
}

/// Non-blocking collection entry point used by the Team/Graph orchestrator.
///
/// Follows the same truth path as `task_status`: persist evidence first, then
/// consume the channel, and finally clean up IPC and the registry. This way the
/// graph executor does not build a second subagent result protocol and cannot
/// bypass the durable evidence that context rebuild depends on.
pub(super) fn poll_owned_task_result(task_id: &str) -> Result<OwnedTaskPoll, String> {
    let current_session_id = crate::ai::driver::runtime_ctx::current_session_id_or_empty();
    let current_owner_pid = current_task_owner_pid()?;
    let entry = {
        let registry = lock_task_registry();
        registry.get_ref(&task_id.to_string()).cloned()
    };
    let Some(entry) = entry else {
        // Entry missing: in the normal flow an entry is only removed after the
        // task is polled to Terminal and its evidence is persisted. If this
        // session already persisted evidence for the task (the only path for the
        // H1 regression "polled but checkpoint not saved"), return an idempotent
        // Terminal so the graph/team checkpoint can self-heal instead of hanging
        // forever; without evidence this is still a genuine error.
        if let Some((status, payload)) =
            read_persisted_task_evidence_status(&current_session_id, task_id)?
        {
            return Ok(OwnedTaskPoll::Terminal {
                result: StoredTaskResult {
                    status,
                    output: payload.clone(),
                    error: None,
                    progress: None,
                    backend_session: None,
                },
                rendered: payload,
            });
        }
        return Err(format!("Unknown graph-managed task_id: {task_id}"));
    };
    if !task_entry_owned_by(&entry, &current_session_id, current_owner_pid) {
        return Err(format!(
            "Task {task_id} is not owned by the current process/session"
        ));
    }

    let poll = with_os_kernel(|os| {
        let collected = drain_task_result(
            os,
            task_id,
            &entry,
            "subagent result collected by agent graph",
        )?;
        Ok(match collected {
            Some(collected) => OwnedTaskPoll::Terminal {
                result: collected.result,
                rendered: collected.rendered,
            },
            None => OwnedTaskPoll::Pending {
                state: task_state_string(os, entry.result_channel_id, entry.pid)?,
            },
        })
    });

    if matches!(&poll, Ok(OwnedTaskPoll::Terminal { .. })) {
        remove_task_entry(task_id);
    }
    poll
}

/// After collecting a terminal result, release all IPC and terminate and reap
/// the corresponding kernel process.
///
/// The result payload may wake the parent agent before the driver finishes
/// updating the final process state; therefore the collector cannot rely on
/// `drop_terminated` alone. Whether the process is still Ready/Running or
/// already Terminated, terminal collection closes out the one-shot subagent
/// task so the process table does not grow indefinitely.
pub(super) fn cleanup_collected_task(os: &mut dyn Kernel, entry: &AsyncTaskEntry, reason: &str) {
    let channel_id = ChannelId(entry.result_channel_id);
    let _ = os.channel_close(None, channel_id);
    let _ = os.channel_release_named(channel_id, "task_result.consumer");
    let _ = os.channel_release_named(channel_id, "task_result.producer");
    let _ = os.channel_destroy(None, channel_id);
    let _ = os.futex_destroy(entry.completion_futex_addr);

    if os.get_process(entry.pid).is_none() {
        return;
    }

    // Defend against tests or a corrupted registry registering the foreground
    // owner itself as a task pid; collecting a subagent result must never
    // terminate a still-running parent process. If it has already terminated,
    // the cleanup below still runs normally and drops it.
    if entry.pid == entry.owner_pid
        && !matches!(
            os.get_process(entry.pid).map(|process| &process.state),
            Some(ProcessState::Terminated)
        )
    {
        return;
    }

    // kill_process uses the current pid for parent/child permission checks. The
    // normal path keeps the owner as current; when the session owner is gone we
    // fall back to the child terminating itself, so deleting a session never
    // leaves an orphan behind.
    let collector_pid = if os.get_process(entry.owner_pid).is_some() {
        entry.owner_pid
    } else {
        entry.pid
    };
    os.set_current_pid(Some(collector_pid));
    let _ = os.cleanup_process_resources(entry.pid);
    if !matches!(
        os.get_process(entry.pid).map(|process| &process.state),
        Some(ProcessState::Terminated)
    ) {
        let _ = os.kill_process(entry.pid, reason.to_string());
    }
    let _ = os.drop_terminated(entry.pid);

    if entry.owner_pid != entry.pid && os.get_process(entry.owner_pid).is_some() {
        os.set_current_pid(Some(entry.owner_pid));
    } else {
        os.set_current_pid(None);
    }
}

pub(super) fn subagent_document_text(agent: &AgentManifest) -> String {
    let mut parts = vec![agent.name.clone(), agent.description.clone()];
    if !agent.prompt.trim().is_empty() {
        parts.push(agent.prompt.chars().take(1500).collect());
    }
    parts.join("\n")
}

/// Extract the set of 2-4 character n-grams from text (lowercased, whitespace
/// collapsed and normalized).
/// Used only for set-similarity scoring; carries no term-frequency /
/// inverse-document-frequency weights.
pub(super) fn char_ngram_set_from_text(input: &str) -> FxHashSet<String> {
    let mut normalized = String::with_capacity(input.len());
    let mut prev_space = false;
    for ch in input.to_lowercase().chars() {
        if ch.is_whitespace() {
            if !prev_space {
                normalized.push(' ');
            }
            prev_space = true;
        } else {
            normalized.push(ch);
            prev_space = false;
        }
    }
    let normalized = normalized.trim();
    if normalized.is_empty() {
        return FxHashSet::default();
    }
    let chars: Vec<char> = format!("^{normalized}$").chars().collect();
    let mut set = FxHashSet::default();
    for n in 2..=4 {
        if chars.len() < n {
            continue;
        }
        for window in chars.windows(n) {
            let token: String = window.iter().collect();
            if token.trim().is_empty() {
                continue;
            }
            set.insert(token);
        }
    }
    set
}

/// Subagent auto-selection scoring: Jaccard overlap of two prebuilt normalized-
/// text character n-gram sets (the task-text query set vs. one candidate's
/// document set). Callers cache these sets once per selection round and reuse
/// them across comparisons; scoring is pure integer counting plus a single
/// division, so reusing the same set values yields bit-identical scores.
pub(super) fn char_ngram_jaccard(query: &FxHashSet<String>, doc: &FxHashSet<String>) -> f64 {
    if query.is_empty() || doc.is_empty() {
        return 0.0;
    }
    let intersection = query.intersection(doc).count();
    let union = query.len() + doc.len() - intersection;
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

#[derive(Debug)]
pub(super) struct SelectedSubagent<'a> {
    pub(super) agent: &'a AgentManifest,
    pub(super) auto_selected: bool,
    pub(super) score: i32,
}

/// An explicit agent wins. Otherwise inherit the parent's agent type, subject to
/// the same availability and auto-selection restrictions as any other candidate.
/// An ineligible parent requires an explicit choice; only a missing parent identity
/// falls back to lexical selection, so delegation cannot silently change agent type.
pub(super) fn select_subagent_for_parent<'a>(
    all_agents: &'a [AgentManifest],
    requested_agent: Option<&str>,
    parent_agent: Option<&str>,
    description: &str,
    prompt: &str,
) -> Result<SelectedSubagent<'a>, String> {
    if requested_agent.is_some() {
        return select_subagent(all_agents, requested_agent, description, prompt);
    }
    let Some(parent_name) = parent_agent.map(str::trim).filter(|name| !name.is_empty()) else {
        return select_subagent(all_agents, None, description, prompt);
    };
    let agent = all_agents
        .iter()
        .find(|agent| agent.name.eq_ignore_ascii_case(parent_name))
        .ok_or_else(|| {
            format!(
                "Cannot auto-select parent agent '{parent_name}': it is unavailable. \
                 Specify an available subagent explicitly in the agent field."
            )
        })?;
    let ineligible_reason = if agent.disabled {
        Some("it is disabled")
    } else if agent.hidden {
        Some("it is hidden")
    } else if !agent.is_subagent() {
        Some("it is not a subagent")
    } else if !agent.auto_select {
        Some("it sets auto_select: false")
    } else {
        None
    };
    if let Some(reason) = ineligible_reason {
        return Err(format!(
            "Cannot auto-select parent agent '{}': {reason}. \
             Specify an available subagent explicitly in the agent field.",
            agent.name
        ));
    }
    Ok(SelectedSubagent {
        agent,
        auto_selected: true,
        score: 0,
    })
}

/// Select an explicit subagent or score candidates when no parent identity is
/// available. Candidates with `auto_select: false` never compete on similarity.
pub(super) fn select_subagent<'a>(
    all_agents: &'a [AgentManifest],
    requested_agent: Option<&str>,
    description: &str,
    prompt: &str,
) -> Result<SelectedSubagent<'a>, String> {
    let subagents = agents::get_subagents(all_agents);
    if subagents.is_empty() {
        return Err(
            "No subagents are available. Add at least one agent with mode: subagent or all."
                .to_string(),
        );
    }

    if let Some(requested) = requested_agent {
        if let Some(agent) = subagents
            .iter()
            .copied()
            .find(|agent| agent.name.eq_ignore_ascii_case(requested))
        {
            return Ok(SelectedSubagent {
                agent,
                auto_selected: false,
                score: 0,
            });
        }

        if let Some(agent) = agents::find_agent_by_name(all_agents, requested) {
            return Err(format!(
                "Agent '{}' exists but is not a subagent. Use a subagent or omit the agent field for auto-selection.",
                agent.name
            ));
        }

        let available = subagents
            .iter()
            .map(|agent| agent.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "Unknown subagent '{}'. Available subagents: {}",
            requested, available
        ));
    }

    // Contract-bound agents opt out of auto-selection: their prompt fixes the
    // output shape (e.g. a machine-checked <audit_report>) and forbids edits, so
    // a lexical score could otherwise hand them work they may not perform. They
    // remain reachable through an explicit `agent` name handled above.
    let subagents: Vec<&AgentManifest> = subagents
        .into_iter()
        .filter(|agent| agent.auto_select)
        .collect();
    if subagents.is_empty() {
        return Err(
            "No subagent accepts auto-selection: every subagent sets auto_select: false. \
             Name one in the agent field, or mark one agent auto_select: true."
                .to_string(),
        );
    }

    let task_text = format!("{description}\n{prompt}");

    // Build the query n-gram set once plus one document set per candidate,
    // then reuse the cached sets in every pairwise comparison instead of
    // rebuilding them inside each scorer call. Each score is the same formula
    // over the same set values as before, so results stay bit-identical.
    let query_ngrams = char_ngram_set_from_text(&task_text);
    let doc_ngrams: Vec<FxHashSet<String>> = subagents
        .iter()
        .map(|agent| char_ngram_set_from_text(&subagent_document_text(agent)))
        .collect();

    subagents
        .into_iter()
        .enumerate()
        .max_by(|(index_a, agent_a), (index_b, agent_b)| {
            char_ngram_jaccard(&query_ngrams, &doc_ngrams[*index_a])
                .total_cmp(&char_ngram_jaccard(&query_ngrams, &doc_ngrams[*index_b]))
                .then_with(|| agent_b.name.cmp(&agent_a.name))
        })
        .map(|(index, agent)| {
            // Same Jaccard value over the same cached sets as any comparison,
            // so the reported score matches what pre-refactor code reported.
            let score = char_ngram_jaccard(&query_ngrams, &doc_ngrams[index]);
            SelectedSubagent {
                agent,
                auto_selected: true,
                score: (score * 100.0) as i32,
            }
        })
        .ok_or_else(|| "No subagents are available.".to_string())
}

pub(super) fn format_agent_model_tier(agent: &AgentManifest) -> &'static str {
    match agent.model_tier {
        Some(AgentModelTier::Light) => "light",
        Some(AgentModelTier::Standard) | None => "standard",
        Some(AgentModelTier::Heavy) => "heavy",
    }
}

pub(super) fn format_quality_tier(tier: crate::ai::provider::ModelQualityTier) -> &'static str {
    match tier {
        crate::ai::provider::ModelQualityTier::Basic => "basic",
        crate::ai::provider::ModelQualityTier::Standard => "standard",
        crate::ai::provider::ModelQualityTier::Strong => "strong",
        crate::ai::provider::ModelQualityTier::Flagship => "flagship",
    }
}

#[cfg(test)]
pub(super) fn build_selection_explanation(
    selected: &SelectedSubagent<'_>,
    selected_model: &str,
    model_override: Option<&str>,
    inherited_parent_model: bool,
) -> String {
    build_selection_explanation_for_parent(
        selected,
        selected_model,
        model_override,
        inherited_parent_model,
        None,
    )
}

pub(super) fn build_selection_explanation_for_parent(
    selected: &SelectedSubagent<'_>,
    selected_model: &str,
    model_override: Option<&str>,
    inherited_parent_model: bool,
    parent_agent: Option<&str>,
) -> String {
    let agent_reason = if selected.auto_selected {
        match parent_agent.map(str::trim).filter(|name| !name.is_empty()) {
            Some(parent) => format!("agent_reason=inherited parent agent type '{parent}'"),
            None => format!(
                "agent_reason=auto-selected as the best available subagent (parent agent unavailable; score={})",
                selected.score
            ),
        }
    } else {
        "agent_reason=explicit agent override".to_string()
    };

    let model_reason = if model_override
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .is_some()
    {
        "model_reason=explicit model override".to_string()
    } else if inherited_parent_model {
        "model_reason=inherited parent agent current model".to_string()
    } else {
        format!(
            "model_reason=auto-selected for agent_tier={} using {} platform via {} adapter and {} quality_tier",
            format_agent_model_tier(selected.agent),
            models::model_platform_label(selected_model),
            crate::ai::model_names::adapter_slug(models::model_adapter(selected_model)),
            format_quality_tier(models::model_quality_tier(selected_model))
        )
    };

    format!("{agent_reason}\n{model_reason}")
}

#[cfg(test)]
mod parent_selection_tests {
    use super::*;

    const DESCRIPTION: &str = "Quick fix with minimal reasoning";
    const PROMPT: &str = "Quick fix with minimal reasoning and fast execution.";

    fn manifests() -> Vec<AgentManifest> {
        serde_json::from_value(serde_json::json!([
            {"name": "build", "description": "Default agent for all tasks", "mode": "all"},
            {"name": "sharp", "description": DESCRIPTION, "prompt": PROMPT, "mode": "all"},
            {"name": "audit", "description": "Contract-bound reviewer", "mode": "all", "auto_select": false}
        ]))
        .unwrap()
    }

    #[test]
    fn auto_selection_inherits_build_despite_sharp_matching_the_task_better() {
        let all_agents = manifests();
        let scored = select_subagent(&all_agents, None, DESCRIPTION, PROMPT).unwrap();
        assert_eq!(scored.agent.name, "sharp");

        let selected =
            select_subagent_for_parent(&all_agents, None, Some("build"), DESCRIPTION, PROMPT)
                .unwrap();
        assert_eq!(selected.agent.name, "build");
        assert!(selected.auto_selected);
        let explanation = build_selection_explanation_for_parent(
            &selected,
            "parent-model",
            None,
            true,
            Some("build"),
        );
        assert!(explanation.contains("agent_reason=inherited parent agent type 'build'"));
        assert!(explanation.contains("model_reason=inherited parent agent current model"));
        assert!(!explanation.contains("score="));
    }

    #[test]
    fn auto_selection_respects_explicit_sharp_and_contract_bound_agents() {
        let all_agents = manifests();
        for requested in ["SHARP", "audit"] {
            for parent in [Some("build"), Some("missing"), None] {
                let selected = select_subagent_for_parent(
                    &all_agents,
                    Some(requested),
                    parent,
                    DESCRIPTION,
                    PROMPT,
                )
                .unwrap();
                assert!(selected.agent.name.eq_ignore_ascii_case(requested));
                assert!(!selected.auto_selected);
                let explanation = build_selection_explanation_for_parent(
                    &selected,
                    "chosen-model",
                    Some("chosen-model"),
                    false,
                    parent,
                );
                assert!(explanation.contains("agent_reason=explicit agent override"));
                assert!(explanation.contains("model_reason=explicit model override"));
            }
        }
    }

    #[test]
    fn auto_selection_rejects_ineligible_parent_instead_of_picking_sharp() {
        for (field, value, reason) in [
            ("disabled", serde_json::json!(true), "disabled"),
            ("hidden", serde_json::json!(true), "hidden"),
            ("mode", serde_json::json!("primary"), "not a subagent"),
            (
                "auto_select",
                serde_json::json!(false),
                "auto_select: false",
            ),
        ] {
            let mut all_agents = manifests();
            let mut parent = serde_json::to_value(&all_agents[0]).unwrap();
            parent[field] = value;
            all_agents[0] = serde_json::from_value(parent).unwrap();
            let error =
                select_subagent_for_parent(&all_agents, None, Some("build"), DESCRIPTION, PROMPT)
                    .unwrap_err();
            assert!(error.contains("parent agent 'build'"), "{error}");
            assert!(error.contains(reason), "{error}");
            assert!(error.contains("explicitly in the agent field"), "{error}");
        }
        let all_agents = manifests();
        for (parent, reason) in [("missing", "unavailable"), ("audit", "auto_select: false")] {
            let error =
                select_subagent_for_parent(&all_agents, None, Some(parent), DESCRIPTION, PROMPT)
                    .unwrap_err();
            assert!(error.contains(reason), "{error}");
        }
    }

    #[test]
    fn auto_selection_sharp_parent_keeps_its_own_identity() {
        let all_agents = manifests();
        let selected = select_subagent_for_parent(
            &all_agents,
            None,
            Some("SHARP"),
            "Build",
            "Build a project",
        )
        .unwrap();
        assert_eq!(selected.agent.name, "sharp");
        assert!(selected.auto_selected);
    }

    #[test]
    fn auto_selection_missing_parent_identity_keeps_documented_scoring_fallback() {
        let all_agents = manifests();
        for parent in [None, Some(""), Some("  ")] {
            let selected =
                select_subagent_for_parent(&all_agents, None, parent, DESCRIPTION, PROMPT).unwrap();
            assert_eq!(selected.agent.name, "sharp");
            let explanation = build_selection_explanation_for_parent(
                &selected,
                "parent-model",
                None,
                true,
                parent,
            );
            assert!(explanation.contains("parent agent unavailable; score="));
        }
    }
}
