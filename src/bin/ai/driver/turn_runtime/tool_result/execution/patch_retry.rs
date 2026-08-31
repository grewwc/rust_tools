//! Apply-patch retry policy: stale-target detection and the
//! fresh-read requirement for failed patches.

use super::*;

pub(in crate::ai::driver::turn_runtime) fn extract_apply_patch_target_paths_from_patch(patch: &str) -> Vec<PathBuf> {
    crate::ai::tools::apply_patch_target_paths_from_patch(patch)
        .into_iter()
        .map(|path| FileStore::new(path).path().to_path_buf())
        .collect()
}

/// An `apply_patch` ambiguity means the patch does not match uniquely, so the model must
/// re-read the target file; tweaking the old patch further only fails again. This consults
/// the [`App::stale_patch_targets`] runtime ledger (maintained by
/// [`update_stale_patch_targets`] after each round's tool results settle): a target stays
/// in the ledger after a failure until one successful `read_file` / `write_file` /
/// `apply_patch` removes it and allows patching again.
///
/// Why not scan `messages` anymore: history compression folds failed apply_patch groups
/// into `internal_note` stubs (dropping the `role=tool` result and `assistant.tool_calls`),
/// so the old message-scanning implementation lost stale state and could not block retries.
/// The ledger is a truth source immune to compression.
pub(in crate::ai::driver::turn_runtime) fn patch_retry_requires_fresh_read(
    stale_patch_targets: &rustc_hash::FxHashSet<PathBuf>,
    tool_calls: &[ToolCall],
) -> bool {
    if stale_patch_targets.is_empty() {
        return false;
    }
    tool_calls.iter().any(|tool_call| {
        tool_call.function.name == "apply_patch"
            && patch_target_paths(tool_call)
                .into_iter()
                .any(|path| stale_patch_targets.contains(&path))
    })
}

/// Incrementally maintain the [`App::stale_patch_targets`] ledger from the tool calls
/// actually executed this round and their results.
///
/// Rules (equivalent to the old message scan, but the state lives in an in-memory ledger
/// unaffected by history compression):
/// - `apply_patch` success (`Successfully patched`) → remove the target paths from the ledger;
/// - `apply_patch` failure with `ambiguous patch` → record only the actually failed target paths;
/// - `read_file` not starting with `Error:` → remove the target paths (truth has been re-read);
/// - `write_file` success (`Successfully wrote to`) → remove the target paths.
///
/// Only calls that have a corresponding result are processed; paths are normalized through
/// [`patch_target_paths`] / [`file_tool_target_path`] so relative-path / `~` / absolute-path
/// spelling differences cannot bypass the gate.
pub(in crate::ai::driver::turn_runtime) fn update_stale_patch_targets(
    stale_patch_targets: &mut rustc_hash::FxHashSet<PathBuf>,
    executed_tool_calls: &[ToolCall],
    tool_results: &[crate::ai::types::ToolResult],
) {
    let result_by_id: HashMap<&str, &str> = tool_results
        .iter()
        .map(|result| (result.tool_call_id.as_str(), result.content.as_str()))
        .collect();
    for tool_call in executed_tool_calls {
        let Some(result) = result_by_id.get(tool_call.id.as_str()).copied() else {
            continue;
        };
        match tool_call.function.name.as_str() {
            "apply_patch" => {
                let paths = patch_target_paths(tool_call);
                if paths.is_empty() {
                    continue;
                }
                if result.trim_start().starts_with("Successfully patched") {
                    for path in paths {
                        stale_patch_targets.remove(&path);
                    }
                } else {
                    stale_patch_targets
                        .extend(patch_failure_stale_targets(tool_call, result, &paths));
                }
            }
            "read_file" => {
                let Some(path) = file_tool_target_path(tool_call) else {
                    continue;
                };
                if !result.trim_start().starts_with("Error:") {
                    stale_patch_targets.remove(&path);
                }
            }
            "write_file" => {
                let Some(path) = file_tool_target_path(tool_call) else {
                    continue;
                };
                if result.trim_start().starts_with("Successfully wrote to") {
                    stale_patch_targets.remove(&path);
                }
            }
            _ => {}
        }
    }
}

/// Rebuild the stale-patch ledger from structured tool messages still retained in an
/// old session.
///
/// New sessions restore directly from the SQLite meta; this only serves old stores that
/// predate the meta upgrade, and it writes back immediately after the first load so later
/// history compression never drops the tool-call pairings needed for the rebuild.
pub(in crate::ai::driver) fn stale_patch_targets_from_messages(
    messages: &[Message],
) -> rustc_hash::FxHashSet<PathBuf> {
    let mut tool_calls = Vec::new();
    let mut tool_results = Vec::new();
    for message in messages {
        if let Some(calls) = &message.tool_calls {
            tool_calls.extend(calls.iter().cloned());
        }
        if message.role == "tool"
            && let (Some(tool_call_id), Some(content)) =
                (message.tool_call_id.as_deref(), message.content.as_str())
        {
            tool_results.push(crate::ai::types::ToolResult {
                tool_call_id: tool_call_id.to_string(),
                content: content.to_string(),
            });
        }
    }

    let mut stale_patch_targets = rustc_hash::FxHashSet::default();
    update_stale_patch_targets(&mut stale_patch_targets, &tool_calls, &tool_results);
    stale_patch_targets
}

pub(in crate::ai::driver::turn_runtime) fn patch_failure_diagnostic(result: &str) -> &str {
    result
        .split_once(crate::ai::tools::PATCH_TEXT_BLOCK_START)
        .map_or(result, |(before, _)| before)
}

pub(in crate::ai::driver::turn_runtime) fn direct_patch_failure_is_ambiguous(diagnostic: &str) -> bool {
    diagnostic
        .trim_start()
        .strip_prefix("Error: apply_patch failed: ")
        .unwrap_or(diagnostic.trim_start())
        .starts_with("ambiguous patch:")
}

pub(in crate::ai::driver::turn_runtime) fn patch_failure_stale_targets(
    tool_call: &ToolCall,
    result: &str,
    targets: &[PathBuf],
) -> Vec<PathBuf> {
    let diagnostic = patch_failure_diagnostic(result);
    let failed_targets: Vec<PathBuf> = targets
        .iter()
        .filter(|path| {
            diagnostic.contains(&format!(
                "failed while preparing patch for {}: ambiguous patch:",
                path.display()
            ))
        })
        .cloned()
        .collect();
    if !failed_targets.is_empty() {
        failed_targets
    } else if direct_patch_failure_is_ambiguous(diagnostic) {
        patch_target_paths(tool_call)
    } else {
        Vec::new()
    }
}

pub(in crate::ai::driver::turn_runtime) fn patch_target_paths(tool_call: &ToolCall) -> Vec<PathBuf> {
    let Ok(args) = serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments) else {
        return Vec::new();
    };
    if let Some(target) = args
        .get("file_path")
        .or_else(|| args.get("path"))
        .and_then(serde_json::Value::as_str)
    {
        return vec![FileStore::new(PathBuf::from(target)).path().to_path_buf()];
    }
    args.get("patch")
        .and_then(serde_json::Value::as_str)
        .map(extract_apply_patch_target_paths_from_patch)
        .unwrap_or_default()
}

pub(in crate::ai::driver::turn_runtime) fn file_tool_target_path(tool_call: &ToolCall) -> Option<PathBuf> {
    let args = serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments).ok()?;
    let target = args
        .get("file_path")
        .or_else(|| args.get("path"))
        .and_then(serde_json::Value::as_str)?;
    Some(FileStore::new(PathBuf::from(target)).path().to_path_buf())
}
