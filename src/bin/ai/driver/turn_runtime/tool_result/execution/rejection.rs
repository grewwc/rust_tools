//! Pre-execution gating of tool calls: scoped-instruction preflight
//! targets, rejection reasons, and deduplication of repeated read-only and
//! knowledge-search calls.

use super::*;

#[derive(Clone, Copy)]
pub(in crate::ai::driver::turn_runtime) enum ToolCallRejectionReason {
    NoToolHandoff,
    PatchRetryNeedsFreshRead,
    ScopedInstructionsNeedReload,
}

#[cfg(test)]
pub(in crate::ai::driver::turn_runtime) fn mutation_needs_scoped_instruction_preflight(
    messages: &[Message],
    tool_calls: &[ToolCall],
) -> bool {
    !mutation_scoped_instruction_preflight_targets(messages, tool_calls).is_empty()
}

pub(in crate::ai::driver::turn_runtime) fn mutation_scoped_instruction_preflight_targets(
    messages: &[Message],
    tool_calls: &[ToolCall],
) -> Vec<PathBuf> {
    let targets = crate::ai::driver::turn_runtime::iteration::project_instruction_target_paths_from_tool_calls(
        tool_calls, false,
    );
    if targets.is_empty() {
        return Vec::new();
    }
    let system_prompt = messages
        .first()
        .and_then(|message| message.content.as_str())
        .unwrap_or_default();
    if crate::ai::driver::skill_runtime::scoped_project_instructions_missing(
        system_prompt,
        &targets,
    ) {
        targets
    } else {
        Vec::new()
    }
}

pub(in crate::ai::driver::turn_runtime) fn reject_tool_calls(
    tool_calls: &[ToolCall],
    reason: ToolCallRejectionReason,
) -> ExecuteToolCallsResult {
    ExecuteToolCallsResult {
        executed_tool_calls: tool_calls.to_vec(),
        tool_results: tool_calls
            .iter()
            .map(|tool_call| crate::ai::types::ToolResult {
                tool_call_id: tool_call.id.clone(),
                content: rejected_tool_call_message(&tool_call.function.name, reason),
            })
            .collect(),
        cached_hits: vec![false; tool_calls.len()],
        execution_outcomes: Vec::new(),
        had_error: true,
    }
}

pub(in crate::ai::driver::turn_runtime) fn rejected_tool_call_message(
    tool_name: &str,
    reason: ToolCallRejectionReason,
) -> String {
    match reason {
        ToolCallRejectionReason::NoToolHandoff => format!(
            "Error: tool calls are disabled in no-tool handoff mode for this turn. \
Do not call '{tool_name}' again; instead summarize confirmed facts, answer what you can, and explain the remaining work / blockers / next steps."
        ),
        ToolCallRejectionReason::PatchRetryNeedsFreshRead => format!(
            "Error: apply_patch retry blocked. The previous patch for this file failed with `ambiguous patch`, so the matched text was not unique. \
Do NOT retry patches in this batch — doing so will only fail again. Required recovery steps: (1) call `read_file` on the SAME target path with use_line_numbers=false to get the current raw file content (no line-number prefixes, so you can copy exact text into the patch); (2) copy context lines DIRECTLY from that fresh output, including function names or distinctive surrounding lines to ensure each hunk matches exactly ONE location; (3) call `apply_patch` only in a LATER tool round after you have successfully read the file."
        ),
        ToolCallRejectionReason::ScopedInstructionsNeedReload => format!(
            "Error: '{tool_name}' was paused before execution because target-scoped project instructions were not loaded yet. \
No file was changed. The runtime will add the applicable instruction documents on the next model step. Review those rules, then retry the mutation in a later tool round; do not repeat it in this batch."
        ),
    }
}

pub(in crate::ai::driver::turn_runtime) fn duplicate_read_only_suppressions(
    messages: &[Message],
    turn_messages: &[Message],
    tool_calls: &[ToolCall],
) -> HashMap<String, String> {
    // If the current batch contains any call that cannot be proven read-only, the
    // order between reads and state changes cannot be guaranteed; in that case we
    // must really execute, not reuse old results.
    if tool_calls.iter().any(read_only_replay_invalidating_call) {
        return HashMap::new();
    }

    let mut call_signatures = HashMap::new();
    let mut invalidating_call_ids = HashSet::new();
    let mut completed = HashMap::new();
    // Build anchors from the canonical text of the current turn, then require that the
    // same text still exists verbatim in the request context. This way compression/dedup/
    // overflow stubs and the suppression itself never become new anchors.
    for message in turn_messages {
        // Synthetic user messages (image followups, etc.) are not real round boundaries
        // and must not reset the dedup state.
        if message.role == "user" && !is_runtime_synthetic_user_message(message) {
            call_signatures.clear();
            invalidating_call_ids.clear();
            completed.clear();
            continue;
        }
        if let Some(previous_calls) = &message.tool_calls {
            for tool_call in previous_calls {
                if let Some(signature) = read_only_tool_signature(tool_call) {
                    call_signatures.insert(tool_call.id.as_str(), signature);
                } else if read_only_replay_invalidating_call(tool_call) {
                    invalidating_call_ids.insert(tool_call.id.as_str());
                }
            }
        }
        if message.role == "tool"
            && let Some(call_id) = message.tool_call_id.as_deref()
        {
            // Failure does not imply no side effects: a shell command may write a file
            // before exiting non-zero. Any unregistered call that returns conservatively
            // invalidates old snapshots.
            if invalidating_call_ids.contains(call_id) {
                completed.clear();
                continue;
            }
            if let Some(signature) = call_signatures.get(call_id)
                && tool_result_completed_successfully(&message.content)
                && tool_result_is_available_verbatim(messages, call_id, &message.content)
            {
                // Keep only the original call anchor, do not copy the old body; the
                // original result is already in the current request context.
                completed.insert(signature.clone(), call_id.to_string());
            }
        }
    }

    tool_calls
        .iter()
        .filter_map(|tool_call| {
            let signature = read_only_tool_signature(tool_call)?;
            completed.get(&signature).map(|previous_call_id| {
                (
                    tool_call.id.clone(),
                    duplicate_read_only_suppression_message(
                        &tool_call.function.name,
                        previous_call_id,
                    ),
                )
            })
        })
        .collect()
}

pub(in crate::ai::driver::turn_runtime) fn read_only_replay_invalidating_call(
    tool_call: &ToolCall,
) -> bool {
    read_only_tool_signature(tool_call).is_none()
}

pub(in crate::ai::driver::turn_runtime) const DUPLICATE_READ_ONLY_SUPPRESSION_PREFIX: &str =
    "Duplicate read-only call to '";

pub(in crate::ai::driver::turn_runtime) fn duplicate_read_only_suppression_message(
    tool_name: &str,
    previous_call_id: &str,
) -> String {
    format!(
        "Duplicate read-only call to '{tool_name}' suppressed: identical successful call '{previous_call_id}' is already present in the current request context. Reuse that earlier result; execute again only after relevant state changes or with different arguments."
    )
}

#[cfg(test)]
pub(in crate::ai::driver::turn_runtime) fn duplicate_read_only_call_ids(
    messages: &[Message],
    tool_calls: &[ToolCall],
) -> HashSet<String> {
    duplicate_read_only_suppressions(messages, messages, tool_calls)
        .into_keys()
        .collect()
}

#[cfg(test)]
pub(in crate::ai::driver::turn_runtime) fn duplicate_read_only_call_ids_with_context(
    messages: &[Message],
    turn_messages: &[Message],
    tool_calls: &[ToolCall],
) -> HashSet<String> {
    duplicate_read_only_suppressions(messages, turn_messages, tool_calls)
        .into_keys()
        .collect()
}

pub(in crate::ai::driver::turn_runtime) fn tool_result_is_available_verbatim(
    messages: &[Message],
    call_id: &str,
    canonical_content: &serde_json::Value,
) -> bool {
    messages.iter().any(|message| {
        message.role == "tool"
            && message.tool_call_id.as_deref() == Some(call_id)
            && message.content == *canonical_content
    })
}

pub(in crate::ai::driver::turn_runtime) fn tool_result_completed_successfully(
    content: &serde_json::Value,
) -> bool {
    let text = content.as_str().unwrap_or_default().trim_start();
    !text.starts_with("Error:")
        && !text.starts_with("Exit code:")
        && !text.starts_with(DUPLICATE_READ_ONLY_SUPPRESSION_PREFIX)
}

pub(in crate::ai::driver::turn_runtime) fn read_only_tool_signature(
    tool_call: &ToolCall,
) -> Option<String> {
    if !crate::ai::tools::tool_allows_same_turn_replay(&tool_call.function.name) {
        return None;
    }

    let mut args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
        .unwrap_or_else(|_| serde_json::Value::String(tool_call.function.arguments.clone()));
    // P3: execute_command is only replayable within the same turn when the command can be
    // proven read-only — results of mutating commands (cargo test, git commit, etc.) must
    // not be treated as reusable evidence, or state changes would be masked.
    // The read-only check includes cargo-verify subcommands (needed for evidence-fingerprint
    // normalization); but for same-turn replay, build-verification output contains volatile
    // progress/duration lines and the latest state must be observed, so exclude them here.
    if tool_call.function.name == "execute_command" {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if !crate::ai::driver::turn_runtime::checkpoint::execute_command_is_read_only(command)
            || crate::ai::driver::turn_runtime::checkpoint::command_is_cargo_verify(command)
        {
            return None;
        }
    }
    // P3: normalize read_file paths so `./x` and `x` count as the same read (aligned with
    // the P1-1 evidence fingerprints).
    if tool_call.function.name == "read_file" {
        if let Some(obj) = args.as_object_mut() {
            for key in ["file_path", "path", "filePath"] {
                if let Some(value) = obj.get_mut(key) {
                    if let Some(path) = value.as_str() {
                        *value = serde_json::Value::String(
                            crate::ai::driver::turn_runtime::progress::normalize_rescan_path(path),
                        );
                    }
                }
            }
        }
    }
    let args_json = serde_json::to_string(&args).unwrap_or_else(|_| args.to_string());
    Some(format!("{}\n{}", tool_call.function.name, args_json))
}

/// `knowledge_search` is reusable read-only fact within one user turn. The generic
/// duplicate protection only compares whole batches; here we suppress re-searches per
/// single semantic signature, so other valid tools in the same batch are not rejected
/// as collateral. Any knowledge write invalidates old searches and allows searching again.
pub(in crate::ai::driver::turn_runtime) fn duplicate_knowledge_search_call_ids(
    messages: &[Message],
    tool_calls: &[ToolCall],
) -> HashSet<String> {
    if tool_calls.iter().any(knowledge_store_mutated) {
        return HashSet::new();
    }

    let mut result_by_id: HashMap<&str, &str> = HashMap::new();
    for message in messages {
        if message.role != "tool" {
            continue;
        }
        if let (Some(id), Some(content)) =
            (message.tool_call_id.as_deref(), message.content.as_str())
        {
            result_by_id.insert(id, content);
        }
    }

    let mut completed_searches = HashSet::new();
    for message in messages.iter().rev() {
        // Synthetic user messages (evidence handoffs, etc.) are not real round boundaries
        // and must not cut the reverse scan.
        if message.role == "user" && !is_runtime_synthetic_user_message(message) {
            break;
        }
        let Some(previous_calls) = message.tool_calls.as_ref() else {
            continue;
        };
        if previous_calls.iter().any(knowledge_store_mutated) {
            break;
        }
        for previous in previous_calls {
            let Some(signature) = knowledge_search_signature(previous) else {
                continue;
            };
            let Some(result) = result_by_id.get(previous.id.as_str()).copied() else {
                continue;
            };
            if !result.trim_start().starts_with("Error:") {
                completed_searches.insert(signature);
            }
        }
    }

    let mut duplicate_ids = HashSet::new();
    for tool_call in tool_calls {
        let Some(signature) = knowledge_search_signature(tool_call) else {
            continue;
        };
        if !completed_searches.insert(signature) {
            duplicate_ids.insert(tool_call.id.clone());
        }
    }
    duplicate_ids
}

pub(in crate::ai::driver::turn_runtime) fn knowledge_search_signature(
    tool_call: &ToolCall,
) -> Option<String> {
    if tool_call.function.name != "knowledge_search" {
        return None;
    }
    let args = serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments).ok()?;
    let query = args.get("query")?.as_str()?.trim();
    if query.is_empty() {
        return None;
    }
    let category = args
        .get("category")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|category| !category.is_empty())
        .unwrap_or("");
    let limit = args
        .get("limit")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(10);
    Some(format!(
        "{}\n{}\n{limit}",
        query.to_lowercase(),
        category.to_lowercase()
    ))
}

pub(in crate::ai::driver::turn_runtime) fn knowledge_store_mutated(tool_call: &ToolCall) -> bool {
    match tool_call.function.name.as_str() {
        "knowledge_save" | "knowledge_forget" => true,
        "knowledge_consolidate" => {
            serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
                .ok()
                .is_some_and(|args| {
                    args.get("action").and_then(serde_json::Value::as_str) == Some("execute")
                })
        }
        _ => false,
    }
}

pub(in crate::ai::driver::turn_runtime) fn duplicate_knowledge_search_message() -> String {
    "Error: this knowledge_search was already completed with the same query in the current user turn. Reuse its result; search again only after knowledge changes or with a materially different query.".to_string()
}
