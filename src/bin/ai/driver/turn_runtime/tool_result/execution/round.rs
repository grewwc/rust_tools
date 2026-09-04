//! Tool-call round orchestration: read-only suppression, duplicate-call
//! uniquification, and round-level execution.

use super::*;

#[crate::ai::agent_hang_span(
    "pre-fix",
    "C",
    "turn_runtime::run_turn:execute_tool_calls",
    "[DEBUG] executing tool calls",
    "[DEBUG] executed tool calls",
    {
        "iteration": _iteration,
        "tool_calls": tool_calls
            .iter()
            .map(|tool| tool.function.name.clone())
            .collect::<Vec<_>>(),
    },
    {
        "iteration": _iteration,
        "tool_result_count": __agent_hang_result
            .as_ref()
            .map(|v| v.tool_results.len())
            .unwrap_or(0),
        "cached_hits": __agent_hang_result
            .as_ref()
            .map(|v| v.cached_hits.clone())
            .unwrap_or_default(),
        "ok": __agent_hang_result.is_ok(),
        "elapsed_ms": __agent_hang_elapsed_ms,
    }
)]
pub(in crate::ai::driver::turn_runtime) fn execute_tool_calls_for_round(
    session_id: &str,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    tool_calls: &[ToolCall],
    allowed_tool_names: &rust_tools::commonw::FastSet<String>,
    observer: Option<&mut dyn tools::ToolExecutionObserver>,
    _iteration: usize,
) -> Result<ExecuteToolCallsResult, Box<dyn std::error::Error>> {
    tools::execute_tool_calls(
        session_id,
        mcp_client,
        shared_mcp_client,
        tool_calls,
        Some(allowed_tool_names),
        observer,
    )
}

pub(super) fn build_first_use_tool_guidance_messages_with(
    exec_result: &ExecuteToolCallsResult,
    prior_turn_messages: &[Message],
    guidance_for: impl Fn(&str) -> Option<String>,
) -> Vec<Message> {
    let previously_called = |tool_name: &str| {
        prior_turn_messages
            .iter()
            .filter(|message| message.role == "assistant")
            .flat_map(|message| message.tool_calls.iter().flatten())
            .any(|tool_call| tool_call.function.name == tool_name)
    };
    let mut guided_this_round = FastSet::default();
    let mut messages = Vec::new();
    for (tool_call, _) in exec_result
        .executed_tool_calls
        .iter()
        .zip(exec_result.tool_results.iter())
    {
        let tool_name = tool_call.function.name.as_str();
        if previously_called(tool_name) || !guided_this_round.insert(tool_name.to_string()) {
            continue;
        }
        let Some(guidance) = guidance_for(tool_name) else {
            continue;
        };
        messages.push(Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: serde_json::Value::String(format!(
                "[tool_first_use_guidance name={tool_name}]\n{guidance}"
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }
    messages
}

fn build_first_use_tool_guidance_messages(
    exec_result: &ExecuteToolCallsResult,
    prior_turn_messages: &[Message],
) -> Vec<Message> {
    build_first_use_tool_guidance_messages_with(exec_result, prior_turn_messages, |tool_name| {
        crate::ai::tools::tool_first_use_guidance(tool_name).map(str::to_owned)
    })
}

pub(in crate::ai::driver::turn_runtime) fn handle_tool_call_round(
    app: &mut App,
    source_model: &str,
    // Since Step 5, real dispatch is handled by RoundToolExecutorAdapter, which locks
    // shared_mcp_client to obtain `&McpClient`; this parameter is kept for compatibility
    // with existing callers. In production it is a routing_snapshot() value — routing
    // through the shared `cached_server_prefixes` is equivalent to the real client, and
    // real MCP execution goes through shared_mcp_client.
    _mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    tool_call_execution: &ToolCallExecution,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    iteration: usize,
    rejection_reason: Option<ToolCallRejectionReason>,
    suppressed_read_only_results: &HashMap<String, String>,
    turn_had_tool_error: &mut bool,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let remaining_meta = parse_prune_meta_and_update_marks(
        app,
        messages,
        &tool_call_execution.stream_result.hidden_meta,
    );
    let mut exec_result = if let Some(reason) = rejection_reason {
        reject_tool_calls(&tool_call_execution.stream_result.tool_calls, reason)
    } else {
        // Step 5: build the per-round ToolExecutor chain with real dispatch as the inner
        // adapter; an empty middleware chain is the identity — zero behavior change.
        let adapter = RoundToolExecutorAdapter {
            session_id: app.session_id.clone(),
            shared_mcp_client: shared_mcp_client.clone(),
            allowed_tool_names: tool_call_execution.allowed_tool_names.clone(),
            suppressed_read_only_results: suppressed_read_only_results.clone(),
            iteration,
        };
        let executor = build_tool_executor_chain(app.tool_middlewares.clone(), Box::new(adapter));
        // The port `execute` is async; this path is synchronous driving (including test
        // threads without a tokio runtime), so futures_executor::block_on blocks the
        // current thread (independent executor, usable in any context).
        let output = futures_executor::block_on(
            executor.execute(app, tool_call_execution.stream_result.tool_calls.clone()),
        )
        // The port error is `Box<dyn Error + Send + Sync>` while this function returns
        // `Box<dyn Error>` (Sized constraint); map to `io::Error` first then propagate
        // with `?`; no prefix is added so middleware/dispatch context is preserved.
        .map_err(|e| std::io::Error::other(e.to_string()))?;
        let ToolExecOutput {
            tool_results,
            assistant_messages,
            executed_tool_calls,
            cached_hits,
            execution_outcomes,
            had_error,
        } = output;
        // Assistant messages injected by middleware (always empty with the current empty
        // chain): the field is kept; mounting is left for future middleware capability.
        let _unwired_assistant_messages = assistant_messages;
        ExecuteToolCallsResult {
            executed_tool_calls,
            tool_results,
            cached_hits,
            execution_outcomes,
            had_error,
        }
    };
    let persisted_tool_call_ids =
        crate::ai::history::read_tool_message_ids_sqlite(&app.session_history_file)
            .unwrap_or_default();
    uniquify_tool_call_occurrences(messages, &persisted_tool_call_ids, &mut exec_result);
    *turn_had_tool_error |= exec_result.had_error;
    // The apply_patch stale-target ledger must be updated after results settle and before
    // the next round's guard check. messages is not a reliable truth source: history
    // compression folds failed groups into internal_note; so the live state lives on App
    // and is mirrored into the current session's SQLite meta. The `apply_patch retry
    // blocked` text produced when the guard rejects a call is neither success nor
    // mismatch, so it has no effect on the ledger.
    update_stale_patch_targets(
        &mut app.stale_patch_targets,
        &exec_result.executed_tool_calls,
        &exec_result.tool_results,
    );
    // Capture turn-local first-use guidance before appending this round's
    // assistant tool calls to canonical history; after that append, those calls
    // would be indistinguishable from calls made in earlier rounds.
    let first_use_guidance_messages =
        build_first_use_tool_guidance_messages(&exec_result, turn_messages);
    // Write the ledger before messages hit disk: if the process crashes between the two
    // writes, leaving a conservative fresh-read requirement is safe; losing the mismatch
    // state instead would let a stale patch through after session recovery.
    // Ordinary one-off temp sessions are deleted on exit; no separate SQLite is created
    // for them.
    let ephemeral_one_shot = one_shot_mode && app.cli.session.is_none();
    if !ephemeral_one_shot
        && let Err(error) = crate::ai::history::write_stale_patch_targets_sqlite(
            &app.session_history_file,
            &app.stale_patch_targets,
        )
    {
        eprintln!("[Warning] failed to persist stale patch targets: {error}");
    }
    append_cached_tool_results_note(&exec_result, messages, turn_messages);
    append_tool_result_messages_for_model(
        app,
        source_model,
        &tool_call_execution.stream_result.assistant_text,
        &tool_call_execution.stream_result.reasoning_text,
        &tool_call_execution.stream_result.reasoning_items,
        &exec_result,
        messages,
        turn_messages,
    );
    for message in first_use_guidance_messages {
        append_message_pair(messages, turn_messages, message);
    }
    record_hidden_self_note(app, turn_messages, &remaining_meta);
    record_tool_inspection_artifacts(messages, turn_messages);

    let history_ready = persist_pending_turn_messages_for_model(
        app,
        source_model,
        one_shot_mode,
        turn_messages,
        persisted_turn_messages,
    );
    if history_ready {
        let outcomes = exec_result
            .execution_outcomes
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        if let Err(error) = crate::ai::history::append_tool_execution_outcomes_sqlite(
            &app.session_history_file,
            &outcomes,
        ) {
            // If the bypass-state write fails, degrade safely to “do not fold” so the
            // original tool result is unaffected.
            eprintln!("[Warning] failed to persist structured tool outcomes: {error}");
        }
    }

    Ok(terminal_dedupe_candidate_from_assistant_text(
        &tool_call_execution.stream_result.assistant_text,
    ))
}

/// Terminal dedup candidates must align with the actually visible body: the digest is
/// extra image-understanding content shown only to the model, never in the terminal, so
/// candidates strip it too before comparing or falling back to rendering.
pub(in crate::ai::driver::turn_runtime) fn terminal_dedupe_candidate_from_assistant_text(
    assistant_text: &str,
) -> Option<String> {
    let visible_text = crate::ai::request::strip_digest_blocks(assistant_text.trim());
    (!visible_text.is_empty()).then(|| visible_text.to_string())
}

pub(in crate::ai::driver::turn_runtime) fn execute_tool_calls_with_suppressed_read_only_calls(
    session_id: &str,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    tool_calls: &[ToolCall],
    allowed_tool_names: &rust_tools::commonw::FastSet<String>,
    observer: Option<&mut dyn tools::ToolExecutionObserver>,
    iteration: usize,
    suppressed_results: &HashMap<String, String>,
) -> Result<ExecuteToolCallsResult, Box<dyn std::error::Error>> {
    if suppressed_results.is_empty() {
        return execute_tool_calls_for_round(
            session_id,
            mcp_client,
            shared_mcp_client,
            tool_calls,
            allowed_tool_names,
            observer,
            iteration,
        );
    }

    let executable = tool_calls
        .iter()
        .filter(|tool_call| !suppressed_results.contains_key(&tool_call.id))
        .cloned()
        .collect::<Vec<_>>();
    let executed = if executable.is_empty() {
        ExecuteToolCallsResult {
            executed_tool_calls: Vec::new(),
            tool_results: Vec::new(),
            cached_hits: Vec::new(),
            execution_outcomes: Vec::new(),
            had_error: false,
        }
    } else {
        execute_tool_calls_for_round(
            session_id,
            mcp_client,
            shared_mcp_client,
            &executable,
            allowed_tool_names,
            observer,
            iteration,
        )?
    };

    let executed_had_error = executed.had_error;
    let mut executed = executed
        .executed_tool_calls
        .into_iter()
        .zip(executed.tool_results)
        .zip(executed.cached_hits)
        .zip(executed.execution_outcomes)
        .map(|(((call, result), cached), outcome)| (call, result, cached, outcome));
    let mut tool_results = Vec::with_capacity(tool_calls.len());
    let mut cached_hits = Vec::with_capacity(tool_calls.len());
    let mut execution_outcomes = Vec::with_capacity(tool_calls.len());
    for tool_call in tool_calls {
        if let Some(content) = suppressed_results.get(&tool_call.id) {
            tool_results.push(crate::ai::types::ToolResult {
                tool_call_id: tool_call.id.clone(),
                content: content.clone(),
            });
            // A dedup result is only a short anchor pointing at the original call in the
            // current context, not a real cached body.
            cached_hits.push(false);
            execution_outcomes.push(None);
            continue;
        }
        let Some((executed_call, result, cached, outcome)) = executed.next() else {
            tool_results.push(crate::ai::types::ToolResult {
                tool_call_id: tool_call.id.clone(),
                content: "Error: tool execution returned no result for this call.".to_string(),
            });
            cached_hits.push(false);
            execution_outcomes.push(None);
            continue;
        };
        debug_assert_eq!(executed_call.id, tool_call.id);
        tool_results.push(result);
        cached_hits.push(cached);
        execution_outcomes.push(outcome);
    }

    Ok(ExecuteToolCallsResult {
        executed_tool_calls: tool_calls.to_vec(),
        tool_results,
        cached_hits,
        execution_outcomes,
        had_error: executed_had_error
            || suppressed_results
                .values()
                .any(|content| content.trim_start().starts_with("Error:")),
    })
}

/// `tool_call_id` only guarantees association within a single model response; some
/// providers/fallbacks reuse ids across rounds. Before writing history, any colliding
/// assistant/tool/outcome triple is rewritten to a fresh occurrence ID so later
/// compression and structured outcomes always associate with one real call.
pub(in crate::ai::driver::turn_runtime) fn uniquify_tool_call_occurrences(
    messages: &[Message],
    persisted_tool_call_ids: &[String],
    result: &mut ExecuteToolCallsResult,
) {
    let mut used = messages
        .iter()
        .flat_map(|message| message.tool_calls.iter().flatten())
        .map(|call| call.id.clone())
        .collect::<HashSet<_>>();
    used.extend(
        messages
            .iter()
            .filter_map(|message| message.tool_call_id.clone()),
    );
    // The context budget may have pruned earlier calls from live messages; the full
    // persisted history must also join collision detection so a new occurrence never
    // collides with an old message that is no longer in live context.
    used.extend(persisted_tool_call_ids.iter().cloned());

    for index in 0..result.executed_tool_calls.len() {
        let original = result.executed_tool_calls[index].id.clone();
        let occurrence_id = if used.insert(original.clone()) {
            original
        } else {
            loop {
                let candidate = format!("call_{}", uuid::Uuid::new_v4().simple());
                if used.insert(candidate.clone()) {
                    break candidate;
                }
            }
        };
        result.executed_tool_calls[index].id = occurrence_id.clone();
        if let Some(tool_result) = result.tool_results.get_mut(index) {
            tool_result.tool_call_id = occurrence_id.clone();
        }
        if let Some(Some(outcome)) = result.execution_outcomes.get_mut(index) {
            outcome.tool_call_id = occurrence_id;
        }
    }
}
