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
        &prepared.model,
    );
}

pub(super) fn persist_rendered_task_evidence(
    task_id: &str,
    entry: &AsyncTaskEntry,
    status: &str,
    rendered: &str,
) -> Result<(), String> {
    let Some(context) = crate::ai::driver::runtime_ctx::try_current() else {
        return Ok(());
    };
    crate::ai::history::record_delivered_task_evidence(
        context.app_proto.config.history_file.as_path(),
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

pub(super) fn collect_missing_task_result(
    task_id: &str,
    entry: &AsyncTaskEntry,
) -> Result<String, String> {
    let result = StoredTaskResult {
        status: "failed".to_string(),
        output: String::new(),
        error: Some(format!(
            "Subagent process pid={} terminated without publishing any output.",
            entry.pid
        )),
        progress: current_task_progress(task_id),
    };
    let rendered = format_task_result_with_id(task_id, entry, result.clone());
    persist_rendered_task_evidence(task_id, entry, &result.status, &rendered)?;
    mark_task_retry_status(task_id, &result.status);
    TASK_PROGRESS_REGISTRY
        .lock()
        .unwrap()
        .take(&task_id.to_string());
    Ok(rendered)
}

pub(super) fn collect_ready_task_result(
    os: &mut dyn Kernel,
    task_id: &str,
    entry: &AsyncTaskEntry,
) -> Result<Option<String>, String> {
    let Some(result) = read_task_result(os, entry.result_channel_id, false)? else {
        return Ok(None);
    };
    let rendered = format_task_result_with_id(task_id, entry, result.clone());
    persist_rendered_task_evidence(task_id, entry, &result.status, &rendered)?;
    let consumed = read_task_result(os, entry.result_channel_id, true)?;
    if consumed.is_none() {
        return Err(format!(
            "task result for {task_id} disappeared after durable persistence"
        ));
    }
    mark_task_retry_status(task_id, &result.status);
    TASK_PROGRESS_REGISTRY
        .lock()
        .unwrap()
        .take(&task_id.to_string());
    Ok(Some(rendered))
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
        let registry = TASK_REGISTRY.lock().unwrap();
        let entry = match registry.get_ref(&task_id.to_string()) {
            Some(entry) => entry,
            None => {
                // Entry missing: in the normal flow an entry is only removed
                // after the task is polled to Terminal and its evidence is
                // persisted. If this session already persisted evidence for the
                // task (the only path for the H1 regression "polled but
                // checkpoint not saved"), return an idempotent Terminal so the
                // graph/team checkpoint can self-heal instead of hanging
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
                        },
                        rendered: payload,
                    });
                }
                return Err(format!("Unknown graph-managed task_id: {task_id}"));
            }
        };
        if !task_entry_owned_by(entry, &current_session_id, current_owner_pid) {
            return Err(format!(
                "Task {task_id} is not owned by the current process/session"
            ));
        }
        entry.clone()
    };

    let poll = with_os_kernel(|os| {
        if let Some(result) = read_task_result(os, entry.result_channel_id, false)? {
            let rendered = format_task_result_with_id(task_id, &entry, result.clone());
            persist_rendered_task_evidence(task_id, &entry, &result.status, &rendered)?;
            if read_task_result(os, entry.result_channel_id, true)?.is_none() {
                return Err(format!(
                    "task result for {task_id} disappeared after durable persistence"
                ));
            }
            mark_task_retry_status(task_id, &result.status);
            TASK_PROGRESS_REGISTRY
                .lock()
                .unwrap()
                .take(&task_id.to_string());
            cleanup_collected_task(os, &entry, "subagent result collected by agent graph");
            return Ok(OwnedTaskPoll::Terminal { result, rendered });
        }

        if is_task_pending(os, entry.pid)? {
            return Ok(OwnedTaskPoll::Pending {
                state: task_state_string(os, entry.result_channel_id, entry.pid)?,
            });
        }

        let result = StoredTaskResult {
            status: "failed".to_string(),
            output: String::new(),
            error: Some(format!(
                "Subagent process pid={} terminated without publishing any output.",
                entry.pid
            )),
            progress: current_task_progress(task_id),
        };
        let rendered = format_task_result_with_id(task_id, &entry, result.clone());
        persist_rendered_task_evidence(task_id, &entry, &result.status, &rendered)?;
        mark_task_retry_status(task_id, &result.status);
        TASK_PROGRESS_REGISTRY
            .lock()
            .unwrap()
            .take(&task_id.to_string());
        cleanup_collected_task(
            os,
            &entry,
            "graph-managed subagent terminated without output",
        );
        Ok(OwnedTaskPoll::Terminal { result, rendered })
    })?;

    if matches!(poll, OwnedTaskPoll::Terminal { .. }) {
        TASK_REGISTRY.lock().unwrap().remove(&task_id.to_string());
    }
    Ok(poll)
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

pub(super) fn build_selection_explanation(
    selected: &SelectedSubagent<'_>,
    selected_model: &str,
    model_override: Option<&str>,
    inherited_parent_model: bool,
) -> String {
    let agent_reason = if selected.auto_selected {
        format!(
            "agent_reason=auto-selected as the best available subagent (score={})",
            selected.score
        )
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
