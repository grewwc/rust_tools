use crate::ai::tools::os_tools::GLOBAL_OS;
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use aios_kernel::kernel::WaitPolicy;
use chrono::{DateTime, Duration, Local, Utc};
use rust_tools::commonw::FastSet;
use rust_tools::cw::SkipMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::{LazyLock, Mutex};
use std::thread;
use std::time::{Duration as StdDuration, Instant, UNIX_EPOCH};

use crate::ai::{
    driver::print::{
        echo_tool_args, echo_tool_output, format_tool_status_cached, format_tool_status_completed,
        format_tool_status_deferred, format_tool_status_failed, format_tool_status_running,
        format_tool_status_skipped,
    },
    mcp::{McpClient, SharedMcpClient},
    tools as builtin_tools,
    tools::task_tools::epoll_wait_many,
    types::{ToolCall, ToolResult},
};
use crate::commonw::prompt::prompt_yes_or_no_interruptible;

mod async_pipe;
mod barrier;
mod oauth;
mod sync_task;

#[allow(unused_imports)]
use async_pipe::*;

static TOOL_FAILURES: LazyLock<Mutex<SkipMap<String, usize>>> =
    LazyLock::new(|| Mutex::new(SkipMap::default()));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolFailureKind {
    Argument,
    Permission,
    Canceled,
    Transient,
    Permanent,
}

#[derive(Debug, Clone)]
pub(super) enum ToolRoute {
    Builtin,
    Mcp {
        server_name: String,
        tool_name: String,
    },
}

#[derive(Debug, Clone)]
struct PreparedToolCall {
    route: ToolRoute,
    args: Value,
}

pub(super) struct ExecuteToolCallsResult {
    pub(super) executed_tool_calls: Vec<ToolCall>,
    pub(super) tool_results: Vec<ToolResult>,
    pub(super) cached_hits: Vec<bool>,
    /// 本轮是否有任何工具执行失败（`RunOneResult.ok == false`）。
    /// 结构化信号，供下游 reflection/evolution 判定 turn 质量时使用，
    /// 替代旧版扫描 assistant 答案文本找 "error"/"failed" 的脆弱做法。
    pub(super) had_error: bool,
}

pub(super) struct RunOneResult {
    pub(super) tool_result: ToolResult,
    pub(super) ok: bool,
    pub(super) executed: bool,
    pub(super) cached: bool,
}

pub(super) trait ToolExecutionObserver {
    fn on_tool_started(&mut self, _tool_call: &ToolCall) {}

    fn on_tool_stream(&mut self, _tool_call: &ToolCall, _chunk: &[u8]) {}

    fn on_tool_finished(&mut self, _tool_call: &ToolCall, _run_result: &RunOneResult) {}
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolCachePayload {
    tool_name: String,
    args: Value,
    result: String,
    #[serde(default)]
    file_fingerprints: Vec<CachedFileFingerprint>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CachedFileFingerprint {
    path: String,
    size: u64,
    modified_ms: Option<u64>,
}

const TOOL_CACHE_RECENT_LIMIT: usize = 400;
const TOOL_CACHE_MAX_RESULT_CHARS: usize = 12_000;
const TOOL_CACHE_TTL_MINUTES: i64 = 30;

fn route_tool_call(mcp_client: &McpClient, tool_name: &str) -> ToolRoute {
    if let Some((server_name, tool_name)) = mcp_client.parse_tool_name_for_known_server(tool_name) {
        ToolRoute::Mcp {
            server_name,
            tool_name,
        }
    } else {
        ToolRoute::Builtin
    }
}

fn parse_tool_args(tool_call: &ToolCall) -> Result<Value, ToolResult> {
    let raw_args = tool_call.function.arguments.trim();
    if raw_args.is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str(raw_args).map_err(|err| ToolResult {
        tool_call_id: tool_call.id.clone(),
        content: format!("Error: failed to parse arguments: {}", err),
    })
}

fn prepare_tool_call(
    mcp_client: &McpClient,
    tool_call: &ToolCall,
    allowed_tool_names: Option<&FastSet<String>>,
) -> Result<PreparedToolCall, ToolResult> {
    if let Some(allowed_tool_names) = allowed_tool_names
        && !allowed_tool_names.contains(&tool_call.function.name)
    {
        return Err(ToolResult {
            tool_call_id: tool_call.id.clone(),
            content: format!(
                "Error: tool '{}' is not available in this turn's tool schema.",
                tool_call.function.name
            ),
        });
    }
    Ok(PreparedToolCall {
        route: route_tool_call(mcp_client, &tool_call.function.name),
        args: parse_tool_args(tool_call)?,
    })
}

fn requires_user_confirmation_for_tool(_tool_name: &str) -> bool {
    false
}

fn confirm_tool_execution(tool_call: &ToolCall, args: &Value) -> Result<(), RunOneResult> {
    if !requires_user_confirmation_for_tool(&tool_call.function.name) {
        return Ok(());
    }

    let confirm =
        prompt_yes_or_no_interruptible(&format!("Confirm tool execution:{} (y/n): ", args));
    if confirm == Some(true) {
        return Ok(());
    }

    println!("canceled by user.");
    Err(RunOneResult {
        tool_result: ToolResult {
            tool_call_id: tool_call.id.clone(),
            content: if confirm.is_none() {
                format!(
                    "Error: {} canceled by user (Ctrl+C)",
                    tool_call.function.name
                )
            } else {
                format!("Error: {} canceled by user", tool_call.function.name)
            },
        },
        ok: false,
        executed: false,
        cached: false,
    })
}

#[derive(Clone, Copy)]
struct ToolAlternative {
    name: &'static str,
    description: &'static str,
}

fn tool_visible_in_current_turn(
    available_tool_names: Option<&FastSet<String>>,
    tool_name: &str,
) -> bool {
    available_tool_names.is_some_and(|names| names.contains(tool_name))
}

fn format_tool_alternatives(alternatives: &[ToolAlternative]) -> String {
    alternatives
        .iter()
        .map(|tool| format!("`{}` ({})", tool.name, tool.description))
        .collect::<Vec<_>>()
        .join(", ")
}

fn equivalent_tools(
    tool_name: &str,
    available_tool_names: Option<&FastSet<String>>,
) -> Option<String> {
    let candidates: &[ToolAlternative] = match tool_name {
        "read_file" => &[
            ToolAlternative {
                name: "code_search",
                description: "locate the relevant region first",
            },
        ],
        "code_search" => &[
            ToolAlternative {
                name: "text_grep",
                description: "regex over file contents",
            },
            ToolAlternative {
                name: "search_files",
                description: "filename or glob match",
            },
            ToolAlternative {
                name: "find_path",
                description: "path-suffix or glob match",
            },
        ],
        "text_grep" => &[
            ToolAlternative {
                name: "code_search",
                description: "semantic or structural search",
            },
            ToolAlternative {
                name: "search_files",
                description: "filename or glob match",
            },
            ToolAlternative {
                name: "find_path",
                description: "path-only discovery",
            },
        ],
        "find_path" => &[
            ToolAlternative {
                name: "search_files",
                description: "filename or glob match",
            },
            ToolAlternative {
                name: "text_grep",
                description: "content search",
            },
            ToolAlternative {
                name: "code_search",
                description: "semantic or structural search",
            },
        ],
        "search_files" => &[
            ToolAlternative {
                name: "find_path",
                description: "path-suffix or glob match",
            },
            ToolAlternative {
                name: "text_grep",
                description: "content search",
            },
            ToolAlternative {
                name: "code_search",
                description: "semantic or structural search",
            },
        ],
        _ => &[],
    };

    let visible = candidates
        .iter()
        .copied()
        .filter(|tool| tool_visible_in_current_turn(available_tool_names, tool.name))
        .collect::<Vec<_>>();
    (!visible.is_empty()).then(|| format_tool_alternatives(&visible))
}

fn remediation_hint(
    tool_name: &str,
    err: &str,
    available_tool_names: Option<&FastSet<String>>,
) -> Option<String> {
    let err_lower = err.to_lowercase();

    if tool_name == "apply_patch"
        && (err_lower.contains("no hunks found")
            || err_lower.contains("invalid hunk")
            || err_lower.contains("context mismatch")
            || err_lower.contains("ambiguous patch")
            || err_lower.contains("missing file_path")
            || err_lower.contains("missing patch")
            || err_lower.contains("patch target mismatch"))
    {
        let read_file_hint = if tool_visible_in_current_turn(available_tool_names, "read_file") {
            "build hunk context from raw file text only — do not copy line numbers, truncation notices, or the Symbol outline block into the patch"
        } else {
            "build hunk context from the exact current file text only"
        };
        let write_file_hint = if tool_visible_in_current_turn(available_tool_names, "write_file") {
            " If you are replacing the whole file, use `write_file` instead."
        } else {
            ""
        };
        return Some(format!(
            "Suggestion: `apply_patch` accepts either raw unified-diff hunks starting with `@@`, or a single-file `*** Begin Patch` envelope. Use `file_path` (or the compatibility alias `path`) for the target file, and {}.{}",
            read_file_hint, write_file_hint
        ));
    }

    if tool_name == "mcp_feishu_docs_get_text_by_url" && err_lower.contains("unsupported url") {
        let discovery_hint =
            if tool_visible_in_current_turn(available_tool_names, "mcp_feishu_docs_search") {
                " Use `mcp_feishu_docs_search` to find the document first,"
            } else {
                ""
            };
        return Some(format!(
            "Suggestion: this tool only works for supported Feishu/Lark docs URLs. Do not retry with the same URL.{} or ask the user for a direct Feishu docs/wiki/sheet URL.",
            discovery_hint
        ));
    }

    if err_lower.contains("failed to parse arguments") || err_lower.contains("invalid type") {
        return Some(
            "Suggestion: fix the tool arguments to match the declared JSON schema before retrying."
                .to_string(),
        );
    }

    if err_lower.contains("no such file") || err_lower.contains("not found") {
        // 文件类工具在 "not found" 时优先建议先用搜索类工具确认目标
        if let Some(fallback) = equivalent_tools(tool_name, available_tool_names) {
            return Some(format!(
                "Suggestion: verify the path or identifier first. Equivalent tools you can try \
                 instead of retrying with the same args: {}.",
                fallback
            ));
        }
        return Some(
            "Suggestion: verify the path or identifier first, or use a search/list tool to discover the correct target before retrying.".to_string(),
        );
    }

    if err_lower.contains("timeout") || err_lower.contains("timed out") {
        return Some(
            "Suggestion: retry once with a narrower query or a smaller scope. If it still fails, switch to another tool or ask the user.".to_string(),
        );
    }

    // 通用 fallback：如果工具名在等价表里，提示可改用的备选工具
    if tool_name == "execute_command" {
        let mut fallback = Vec::new();
        if tool_visible_in_current_turn(available_tool_names, "read_file") {
            fallback.push("read files (whole or precise line ranges) with `read_file`");
        }
        if tool_visible_in_current_turn(available_tool_names, "list_directory") {
            fallback.push("inspect directories with `list_directory`");
        }
        if !fallback.is_empty() {
            return Some(format!(
                "Suggestion: if this failure is intrinsic (not a transient I/O error), break the command into smaller pieces or {} instead of running shell just to inspect state.",
                fallback.join(", ")
            ));
        }
    }

    if let Some(fallback) = equivalent_tools(tool_name, available_tool_names) {
        return Some(format!(
            "Suggestion: if this failure is intrinsic (not a transient I/O error), \
             try an equivalent tool instead of repeating: {}.",
            fallback
        ));
    }

    None
}

fn format_tool_error(
    tool_call: &ToolCall,
    err: &str,
    available_tool_names: Option<&FastSet<String>>,
) -> ToolResult {
    ToolResult {
        tool_call_id: tool_call.id.clone(),
        content: if let Some(hint) =
            remediation_hint(&tool_call.function.name, err, available_tool_names)
        {
            format!(
                "Error: {} failed: {}\n{}",
                tool_call.function.name, err, hint
            )
        } else {
            format!("Error: {} failed: {}", tool_call.function.name, err)
        },
    }
}

fn execute_prepared_tool_call(
    session_id: &str,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    tool_call: &ToolCall,
    prepared: &PreparedToolCall,
    allowed_tool_names: Option<&FastSet<String>>,
    observer: &mut Option<&mut dyn ToolExecutionObserver>,
) -> Result<ToolResult, String> {
    match &prepared.route {
        ToolRoute::Builtin => {
            if tool_call.function.name == "task" {
                sync_task::execute_sync_task(&tool_call.id, &prepared.args).map(|tr| tr)
            } else if tool_call.function.name == "tool_spawn" {
                execute_tool_spawn(
                    session_id,
                    mcp_client,
                    shared_mcp_client,
                    &tool_call.id,
                    &prepared.args,
                    allowed_tool_names,
                )
            } else if tool_call.function.name == "tool_wait" {
                execute_tool_wait(session_id, &tool_call.id, &prepared.args)
            } else if tool_call.function.name == "tool_status" {
                execute_tool_status(session_id, &tool_call.id, &prepared.args)
            } else if tool_call.function.name == "tool_cancel" {
                execute_tool_cancel(session_id, &tool_call.id, &prepared.args)
            } else {
                execute_prepared_builtin_tool_call(tool_call, prepared, |chunk| {
                    if let Some(observer) = observer.as_deref_mut() {
                        observer.on_tool_stream(tool_call, chunk);
                    }
                })
            }
        }
        ToolRoute::Mcp {
            server_name,
            tool_name,
        } => {
            // `mcp_client` 是 orchestrator 传入的 routing_snapshot（servers 为空，
            // 仅用于路由/schema）。实际执行必须走共享的真实客户端，否则 call_tool
            // 会在空的 servers map 里找不到连接而报 "Server not found"。
            let guard = shared_mcp_client
                .lock()
                .map_err(|_| "Shared MCP client poisoned".to_string())?;
            oauth::execute_mcp_tool_call(&guard, tool_call, server_name, tool_name, &prepared.args)
        }
    }
}

fn execute_prepared_builtin_tool_call<F>(
    tool_call: &ToolCall,
    prepared: &PreparedToolCall,
    mut on_chunk: F,
) -> Result<ToolResult, String>
where
    F: FnMut(&[u8]),
{
    builtin_tools::execute_tool_call_with_args_streaming(
        &tool_call.id,
        &tool_call.function.name,
        &prepared.args,
        &mut on_chunk,
    )
}

fn validate_spawnable_tool(target_tool_name: &str, route: &ToolRoute) -> Result<(), String> {
    if let ToolRoute::Mcp { tool_name, .. } = route {
        if tool_name.starts_with("oauth_") {
            return Err("OAuth helper MCP tools cannot be spawned asynchronously.".to_string());
        }
        if matches!(target_tool_name, "tool_spawn" | "tool_wait" | "tool_status") {
            return Err(format!(
                "Tool '{}' cannot be spawned recursively.",
                target_tool_name
            ));
        }
        return Ok(());
    }

    let Some(spec) = crate::ai::tools::registry::common::get_tool_spec(target_tool_name) else {
        return Err(format!("Unknown tool: {}", target_tool_name));
    };

    if spec.async_policy != crate::ai::tools::registry::common::ToolAsyncPolicy::Spawnable {
        return Err(format!(
            "Tool '{}' is not marked as spawnable for async execution.",
            target_tool_name
        ));
    }

    if matches!(target_tool_name, "tool_spawn" | "tool_wait" | "tool_status") {
        return Err(format!(
            "Tool '{}' cannot be spawned recursively.",
            target_tool_name
        ));
    }

    Ok(())
}

fn execute_tool_spawn(
    session_id: &str,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    tool_call_id: &str,
    args: &Value,
    allowed_tool_names: Option<&FastSet<String>>,
) -> Result<ToolResult, String> {
    let target_tool_name = args
        .get("tool_name")
        .and_then(Value::as_str)
        .ok_or("Missing 'tool_name' parameter")?;
    let target_args = args
        .get("arguments")
        .cloned()
        .ok_or("Missing 'arguments' parameter")?;

    if let Some(allowed_tool_names) = allowed_tool_names
        && !allowed_tool_names.contains(target_tool_name)
    {
        return Err(format!(
            "tool '{}' is not available in this turn's tool schema.",
            target_tool_name
        ));
    }

    let async_task_id = next_async_tool_id();
    let synthetic_tool_call = ToolCall {
        id: format!("async-call-{}", async_task_id),
        tool_type: "function".to_string(),
        function: crate::ai::types::FunctionCall {
            name: target_tool_name.to_string(),
            arguments: serde_json::to_string(&target_args)
                .map_err(|e| format!("Failed to serialize arguments: {}", e))?,
        },
    };
    let prepared = PreparedToolCall {
        route: route_tool_call(mcp_client, target_tool_name),
        args: target_args,
    };

    validate_spawnable_tool(target_tool_name, &prepared.route)?;
    let result_channel_id = create_async_tool_channel(&async_task_id);
    let completion_futex_addr = create_async_tool_completion_futex(&async_task_id);
    let started_at = Instant::now();

    if let Some(tool_result) =
        load_cached_tool_result(session_id, &synthetic_tool_call, &prepared.args)
    {
        let run_result = RunOneResult {
            tool_result,
            ok: true,
            executed: false,
            cached: true,
        };
        let mut registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
        registry.insert(
            async_task_id.clone(),
            AsyncToolEntry {
                result_channel_id,
                completion_futex_addr,
                session_id: session_id.to_string(),
                tool_name: target_tool_name.to_string(),
                started_at,
                state: AsyncToolState::Completed((prepared.route.clone(), run_result)),
            },
        );
        signal_async_tool_completion(completion_futex_addr);
        if let Some(entry) = registry.get_ref(&async_task_id) {
            let started = async_tool_pipe_message_from_started(&async_task_id, entry, 0);
            send_async_tool_pipe_message(entry, &started);
            persist_async_tool_snapshot(&async_task_id, entry);
        }
        prune_completed_async_tools(&mut registry);
        return Ok(ToolResult {
            tool_call_id: tool_call_id.to_string(),
            content: json!({
                "task_id": async_task_id,
                "tool_name": target_tool_name,
                "status": "completed",
                "cached": true,
            })
            .to_string(),
        });
    }

    let session_id = session_id.to_string();
    let session_id_for_registry = session_id.clone();
    let tool_name = target_tool_name.to_string();
    let prepared_for_thread = prepared.clone();
    let tool_call_for_thread = synthetic_tool_call.clone();
    let route_for_registry = prepared.route.clone();
    let shared_mcp_client_for_thread = shared_mcp_client.clone();
    let async_task_id_for_thread = async_task_id.clone();
    let tool_name_for_thread = tool_name.clone();
    let result_channel_id_for_thread = result_channel_id;
    let completion_futex_addr_for_thread = completion_futex_addr;
    let started_at_for_thread = started_at;
    let available_tool_names_for_thread = allowed_tool_names.cloned();

    // Insert the Running registry entry BEFORE spawning the worker thread.
    // If the worker completes quickly, it will try to update the entry to
    // Completed — the entry must already exist or the completion is lost and
    // the task is stuck in Running forever.
    {
        let mut registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
        registry.insert(
            async_task_id.clone(),
            AsyncToolEntry {
                result_channel_id,
                completion_futex_addr,
                session_id: session_id_for_registry.clone(),
                tool_name: tool_name.clone(),
                started_at,
                state: AsyncToolState::Running,
            },
        );
        prune_completed_async_tools(&mut registry);
    }

    thread::spawn(move || {
        let mut pipe_observer = AsyncToolPipeObserver {
            task_id: async_task_id_for_thread.clone(),
            session_id: session_id.clone(),
            tool_name: tool_name_for_thread.clone(),
            result_channel_id: result_channel_id_for_thread,
            completion_futex_addr: completion_futex_addr_for_thread,
            started_at: started_at_for_thread,
            next_seq: 0,
        };
        pipe_observer.on_tool_started(&tool_call_for_thread);
        let result = execute_with_safe_retry(
            &prepared_for_thread.route,
            &tool_call_for_thread.function.name,
            || match &prepared_for_thread.route {
                ToolRoute::Builtin => execute_prepared_builtin_tool_call(
                    &tool_call_for_thread,
                    &prepared_for_thread,
                    |chunk| {
                        pipe_observer.on_tool_stream(&tool_call_for_thread, chunk);
                    },
                ),
                ToolRoute::Mcp { .. } => {
                    let guard = shared_mcp_client_for_thread
                        .lock()
                        .map_err(|_| "Shared MCP client poisoned".to_string());
                    match guard {
                        Ok(mc) => oauth::execute_mcp_tool_call(
                            &mc,
                            &tool_call_for_thread,
                            match &prepared_for_thread.route {
                                ToolRoute::Mcp { server_name, .. } => server_name,
                                ToolRoute::Builtin => unreachable!(),
                            },
                            match &prepared_for_thread.route {
                                ToolRoute::Mcp { tool_name, .. } => tool_name,
                                ToolRoute::Builtin => unreachable!(),
                            },
                            &prepared_for_thread.args,
                        )
                        .map_err(|e| e.to_string()),
                        Err(err) => Err(err),
                    }
                }
            },
        );
        let run_result = finalize_execution_result(
            &session_id,
            &tool_call_for_thread,
            &prepared_for_thread,
            result,
            available_tool_names_for_thread.as_ref(),
            true,
            false,
        );
        pipe_observer.on_tool_finished(&tool_call_for_thread, &run_result);
        signal_async_tool_completion(completion_futex_addr_for_thread);

        if let Ok(mut registry) = ASYNC_TOOL_REGISTRY.lock() {
            if let Some(entry) = registry.get_mut(&async_task_id_for_thread)
                && matches!(entry.state, AsyncToolState::Running)
            {
                entry.state = AsyncToolState::Completed((route_for_registry, run_result));
            }
        }
    });

    Ok(ToolResult {
        tool_call_id: tool_call_id.to_string(),
        content: json!({
            "task_id": async_task_id,
            "tool_name": target_tool_name,
            "status": "running",
            "cached": false,
        })
        .to_string(),
    })
}

fn execute_tool_status(
    session_id: &str,
    tool_call_id: &str,
    args: &Value,
) -> Result<ToolResult, String> {
    let filter_task_ids = args.get("task_ids").and_then(Value::as_array).map(|items| {
        items
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect::<Vec<_>>()
    });

    let mut results = Vec::new();
    {
        let mut registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
        let task_ids: Vec<String> = if let Some(ids) = filter_task_ids {
            ids
        } else {
            registry
                .iter()
                .filter(|(_, entry)| entry.session_id == session_id)
                .map(|(task_id, _)| task_id.clone())
                .collect()
        };

        for task_id in task_ids {
            let Some(entry) = registry.get_mut(&task_id) else {
                continue;
            };
            let pipe_messages = peek_async_tool_pipe_messages(entry);
            let agg = aggregate_async_tool_pipe_messages(&pipe_messages);
            if let Some(snapshot) = load_async_tool_snapshot(entry) {
                if snapshot.session_id != session_id {
                    continue;
                }
                let mut value = async_tool_result_json(
                    &snapshot.task_id,
                    &snapshot.tool_name,
                    &snapshot.status,
                    snapshot.ok,
                    snapshot.cached,
                    snapshot.executed,
                    snapshot.reason.as_deref(),
                    snapshot.elapsed_secs,
                    &agg,
                );
                if let Some(obj) = value.as_object_mut()
                    && let Some(content) = snapshot.content
                {
                    obj.insert("content".to_string(), json!(content));
                }
                results.push(value);
                continue;
            }
            if entry.session_id != session_id {
                continue;
            }
            match &entry.state {
                AsyncToolState::Running => results.push(async_tool_result_json(
                    &task_id,
                    &entry.tool_name,
                    "running",
                    None,
                    None,
                    None,
                    None,
                    entry.started_at.elapsed().as_secs_f64(),
                    &agg,
                )),
                AsyncToolState::Completed((_route, run_result)) => {
                    results.push(async_tool_result_json(
                        &task_id,
                        &entry.tool_name,
                        if run_result.ok { "completed" } else { "failed" },
                        Some(run_result.ok),
                        Some(run_result.cached),
                        Some(run_result.executed),
                        None,
                        entry.started_at.elapsed().as_secs_f64(),
                        &agg,
                    ))
                }
                AsyncToolState::Canceled { reason } => results.push(async_tool_result_json(
                    &task_id,
                    &entry.tool_name,
                    "canceled",
                    Some(false),
                    Some(false),
                    Some(false),
                    Some(reason),
                    entry.started_at.elapsed().as_secs_f64(),
                    &agg,
                )),
            }
        }
    }

    Ok(ToolResult {
        tool_call_id: tool_call_id.to_string(),
        content: json!({ "results": results }).to_string(),
    })
}

fn execute_tool_cancel(
    session_id: &str,
    tool_call_id: &str,
    args: &Value,
) -> Result<ToolResult, String> {
    let task_ids = args
        .get("task_ids")
        .and_then(Value::as_array)
        .ok_or("Missing 'task_ids' array parameter")?
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect::<Vec<_>>();
    if task_ids.is_empty() {
        return Err("task_ids array cannot be empty".to_string());
    }

    let reason = args
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or("canceled by model")
        .to_string();

    let mut results = Vec::new();
    let mut registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
    for task_id in task_ids {
        let Some(entry) = registry.get_mut(&task_id) else {
            return Err(format!("Unknown task_id: {}", task_id));
        };
        if entry.session_id != session_id {
            return Err(format!("Task {} does not belong to this session.", task_id));
        }

        match &entry.state {
            AsyncToolState::Running => {
                entry.state = AsyncToolState::Canceled {
                    reason: reason.clone(),
                };
                persist_async_tool_snapshot(&task_id, entry);
                signal_async_tool_completion(entry.completion_futex_addr);
                results.push(json!({
                    "task_id": task_id,
                    "tool_name": entry.tool_name,
                    "status": "canceled",
                    "reason": reason,
                }));
            }
            AsyncToolState::Completed((_route, run_result)) => {
                results.push(json!({
                    "task_id": task_id,
                    "tool_name": entry.tool_name,
                    "status": if run_result.ok { "completed" } else { "failed" },
                    "reason": "task already finished",
                }));
            }
            AsyncToolState::Canceled { reason } => {
                results.push(json!({
                    "task_id": task_id,
                    "tool_name": entry.tool_name,
                    "status": "canceled",
                    "reason": reason,
                }));
            }
        }
    }
    prune_completed_async_tools(&mut registry);

    Ok(ToolResult {
        tool_call_id: tool_call_id.to_string(),
        content: json!({ "results": results }).to_string(),
    })
}

fn execute_tool_wait(
    session_id: &str,
    tool_call_id: &str,
    args: &Value,
) -> Result<ToolResult, String> {
    let task_ids = args
        .get("task_ids")
        .and_then(Value::as_array)
        .ok_or("Missing 'task_ids' array parameter")?
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect::<Vec<_>>();
    if task_ids.is_empty() {
        return Err("task_ids array cannot be empty".to_string());
    }

    let wait_policy = parse_wait_policy(args)?;
    let timeout_ticks = args.get("timeout_ticks").and_then(Value::as_u64);
    let (initial_terminal, initial_pending) = collect_async_task_snapshot(session_id, &task_ids)?;
    let satisfied = match wait_policy {
        WaitPolicy::Any => !initial_terminal.is_empty(),
        WaitPolicy::All => initial_pending.is_empty(),
    };
    if satisfied {
        if initial_pending.is_empty() {
            let registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
            for task_id in &task_ids {
                if let Some(entry) = registry.get_ref(task_id) {
                    delete_async_tool_snapshot(entry);
                }
            }
        }
        let all_done = initial_pending.is_empty();
        let completed_for_results = initial_terminal.clone();
        let pending_for_results = initial_pending.clone();
        let suggested_next_actions = if all_done {
            vec!["continue_reasoning"]
        } else if !initial_terminal.is_empty() {
            vec![
                "continue_reasoning_with_partial_results",
                "use_tool_status",
                "use_tool_cancel",
            ]
        } else {
            vec!["use_tool_status", "continue_reasoning", "use_tool_cancel"]
        };
        return Ok(ToolResult {
            tool_call_id: tool_call_id.to_string(),
            content: json!({
                "all_done": all_done,
                "completed": initial_terminal,
                "pending": initial_pending,
                "results": {
                    "completed": completed_for_results,
                    "pending": pending_for_results,
                },
                "wait_policy": match wait_policy { WaitPolicy::Any => "any", WaitPolicy::All => "all" },
                "suggested_next_actions": suggested_next_actions,
            })
            .to_string(),
        });
    }

    if let Ok(guard) = GLOBAL_OS.lock()
        && let Some(os) = guard.as_ref()
        && let Ok(mut os) = os.lock()
        && os.current_process_id().is_some()
    {
        let wait_sources = lookup_wait_sources(os.as_mut(), session_id, &task_ids)?;
        // 工具层的 wait_policy（all/any）已在上方 collect_async_task_snapshot 的
        // `satisfied` 判定里生效，并会在被唤醒后的下一次调用里重新评估。底层 park
        // **必须** 用 WaitPolicy::Any：wait_sources 里含一个用于中断的 cancel futex，
        // 它在正常路径下永远不会就绪；若按 All 等待，epoll_wait_many 的
        // `pending_sources.is_empty()` 判定永远不成立 —— 即使所有真实任务都已完成，
        // 进程也不会被唤醒（timeout_ticks 缺省时甚至永久挂起）。这与 execute_task_wait
        // 的 park 策略保持一致。
        let wait = epoll_wait_many(
            os.as_mut(),
            &format!("tool_wait:{}:{}", session_id, task_ids.join(",")),
            &wait_sources,
            WaitPolicy::Any,
            timeout_ticks,
        )?;
        if wait.suspended {
            return Ok(ToolResult {
                tool_call_id: tool_call_id.to_string(),
                content: json!({
                    "status": "suspended",
                    "wait_policy": match wait_policy { WaitPolicy::Any => "any", WaitPolicy::All => "all" },
                    "task_ids": task_ids,
                    "event_ids": wait.event_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
                    "timeout_tick": wait.timeout_tick,
                    "message": "Current process suspended via epoll_wait_many. Yield control now; after wake-up, inspect mailbox and use tool_status or tool_wait again if needed."
                })
                .to_string(),
            });
        }
        // Condition already satisfied; fall through to collect results.
    } else {
        // No OS context — use the inline polling path below.
        let already_satisfied = false;
        let _ = already_satisfied;
    }

    // This point is reached in two cases:
    //   1. OS context available but wait condition was already satisfied (no suspension needed).
    //   2. No OS context — fall through to poll-based wait.
    // In case 1 all tasks are done so the loop below exits immediately.
    let max_wait_ms = args
        .get("max_wait_ms")
        .and_then(Value::as_u64)
        .or_else(|| {
            args.get("timeout_secs")
                .and_then(Value::as_u64)
                .map(|secs| secs.saturating_mul(1000))
        })
        .unwrap_or(1500);
    let deadline = Instant::now() + StdDuration::from_millis(max_wait_ms);
    while Instant::now() < deadline {
        let registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
        let mut has_terminal = false;
        let mut has_running = false;
        for task_id in &task_ids {
            let Some(entry) = registry.get_ref(task_id) else {
                return Err(format!("Unknown task_id: {}", task_id));
            };
            if entry.session_id != session_id {
                return Err(format!("Task {} does not belong to this session.", task_id));
            }
            match entry.state {
                AsyncToolState::Running => has_running = true,
                AsyncToolState::Completed(_) | AsyncToolState::Canceled { .. } => {
                    has_terminal = true
                }
            }
        }
        drop(registry);
        if has_terminal || !has_running {
            break;
        }
        thread::sleep(StdDuration::from_millis(50));
    }

    let (terminal_results, pending) = collect_async_task_snapshot(session_id, &task_ids)?;
    if pending.is_empty() {
        let registry = ASYNC_TOOL_REGISTRY.lock().unwrap();
        for task_id in &task_ids {
            if let Some(entry) = registry.get_ref(task_id) {
                delete_async_tool_snapshot(entry);
            }
        }
    }

    let all_done = pending.is_empty();
    let completed_for_results = terminal_results.clone();
    let pending_for_results = pending.clone();
    let suggested_next_actions = if all_done {
        vec!["continue_reasoning"]
    } else if !terminal_results.is_empty() {
        vec![
            "continue_reasoning_with_partial_results",
            "use_tool_status",
            "use_tool_cancel",
        ]
    } else {
        vec!["use_tool_status", "continue_reasoning", "use_tool_cancel"]
    };

    Ok(ToolResult {
        tool_call_id: tool_call_id.to_string(),
        content: json!({
            "all_done": all_done,
            "wait_window_ms": max_wait_ms,
            "completed": terminal_results,
            "pending": pending,
            "results": {
                "completed": completed_for_results,
                "pending": pending_for_results,
            },
            "suggested_next_actions": suggested_next_actions,
        })
        .to_string(),
    })
}

fn record_tool_failure(tool_name: &str) {
    if let Ok(mut map) = TOOL_FAILURES.lock() {
        let counter = map.entry(tool_name.to_string()).or_insert(0);
        *counter = counter.saturating_add(1).min(100);
    }
}

fn classify_tool_error(err: &str) -> ToolFailureKind {
    let lower = err.to_ascii_lowercase();
    if lower.contains("failed to parse arguments")
        || lower.contains("invalid type")
        || lower.contains("missing '")
        || lower.contains("missing parameter")
    {
        return ToolFailureKind::Argument;
    }
    if lower.contains("permission denied")
        || lower.contains("not in the allowed whitelist")
        || lower.contains("not available in this turn's tool schema")
        || lower.contains("kernel tool-call quota")
        || lower.contains("forbidden")
    {
        return ToolFailureKind::Permission;
    }
    if lower.contains("canceled by user") || lower.contains("cancelled by user") {
        return ToolFailureKind::Canceled;
    }
    if lower.contains("timeout")
        || lower.contains("timed out")
        || lower.contains("temporar")
        || lower.contains("connection reset")
        || lower.contains("connection refused")
        || lower.contains("broken pipe")
        || lower.contains("eof")
        || lower.contains("dns")
        || lower.contains("unavailable")
        || lower.contains("rate limit")
    {
        return ToolFailureKind::Transient;
    }
    ToolFailureKind::Permanent
}

fn should_retry_once(route: &ToolRoute, tool_name: &str, err: &str) -> bool {
    if classify_tool_error(err) != ToolFailureKind::Transient {
        return false;
    }
    // 仅对本地只读工具做一次重试，避免副作用工具重复执行。
    matches!(route, ToolRoute::Builtin)
        && is_cacheable_tool_name(tool_name)
        && tool_name != "execute_command"
}

fn execute_with_safe_retry<F>(
    route: &ToolRoute,
    tool_name: &str,
    mut exec: F,
) -> Result<ToolResult, String>
where
    F: FnMut() -> Result<ToolResult, String>,
{
    let mut result = exec();
    if let Err(err) = result.as_ref() {
        if should_retry_once(route, tool_name, err) {
            print!(
                "\r\x1b[2K{} (transient error; one safe retry)\n",
                crate::ai::driver::print::format_tool_status(
                    "Retry",
                    tool_name,
                    crate::ai::theme::ACCENT_WARN
                )
            );
            result = exec();
        }
    }
    result
}

fn finalize_execution_result(
    session_id: &str,
    tool_call: &ToolCall,
    prepared: &PreparedToolCall,
    result: Result<ToolResult, String>,
    available_tool_names: Option<&FastSet<String>>,
    executed: bool,
    cached: bool,
) -> RunOneResult {
    let failure_kind = result.as_ref().err().map(|err| classify_tool_error(err));
    let run_result = match result {
        Ok(tool_result) => {
            if executed && !cached {
                store_tool_cache_result(session_id, tool_call, &prepared.args, &tool_result);
            }
            RunOneResult {
                tool_result,
                ok: true,
                executed,
                cached,
            }
        }
        Err(err) => RunOneResult {
            tool_result: format_tool_error(tool_call, &err, available_tool_names),
            ok: false,
            executed,
            cached,
        },
    };
    if run_result.executed && !run_result.ok {
        // 仅统计会反映到"工具可靠性"的失败，避免把参数错误/用户取消
        // 错误地当作工具本身不稳定，导致路由/惩罚劣化。
        if matches!(
            failure_kind,
            Some(ToolFailureKind::Transient | ToolFailureKind::Permanent)
        ) {
            record_tool_failure(&tool_call.function.name);
        }
    }
    run_result
}

fn print_run_status(tool_call: &ToolCall, run_result: &RunOneResult) {
    let name = &tool_call.function.name;
    if run_result.cached {
        println!("{}", format_tool_status_cached(name));
    } else if !run_result.executed {
        println!("{}", format_tool_status_skipped(name));
    } else if run_result.ok {
        // 已执行的工具：用 \r 回到行首覆盖 running 状态，保持同一行
        print!("\r\x1b[2K{}\n", format_tool_status_completed(name));
    } else {
        print!("\r\x1b[2K{}\n", format_tool_status_failed(name));
    }

    // 部分工具的输出对用户有较高可见性价值，额外把其内容回显到终端。
    // 具体哪些工具回显由工具自身提交的 `ToolDisplayConfig` 控制，
    // 这里不感知具体工具名，便于后续扩展。
    if run_result.ok || run_result.cached {
        echo_tool_args(name, &tool_call.function.arguments);
    }
    if run_result.ok {
        echo_tool_output(name, &run_result.tool_result.content);
    }
}

fn reserve_current_process_tool_call_budget(tool_call: &ToolCall) -> Result<(), RunOneResult> {
    use aios_kernel::primitives::{ResourceUsageDelta, RlimitDim, RlimitVerdict};

    let Ok(guard) = GLOBAL_OS.lock() else {
        return Ok(());
    };
    let Some(os_arc) = guard.as_ref() else {
        return Ok(());
    };
    let Ok(mut os) = os_arc.lock() else {
        return Ok(());
    };
    let Some(pid) = os.current_process_id() else {
        return Ok(());
    };

    match os.rlimit_check(
        pid,
        &ResourceUsageDelta {
            tool_calls: 1,
            ..Default::default()
        },
    ) {
        RlimitVerdict::Exceeded {
            dimension: RlimitDim::ToolCalls,
            used,
            limit,
        } => Err(RunOneResult {
            tool_result: ToolResult {
                tool_call_id: tool_call.id.clone(),
                content: format!(
                    "Error: tool '{}' would exceed the kernel tool-call quota (used={} limit={}).",
                    tool_call.function.name, used, limit
                ),
            },
            ok: false,
            executed: false,
            cached: false,
        }),
        _ => {
            os.increment_tool_calls_used_for(pid);
            Ok(())
        }
    }
}

fn run_one(
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    session_id: &str,
    tool_call: &ToolCall,
    allowed_tool_names: Option<&FastSet<String>>,
    observer: &mut Option<&mut dyn ToolExecutionObserver>,
) -> (ToolRoute, RunOneResult) {
    let prepared = match prepare_tool_call(mcp_client, tool_call, allowed_tool_names) {
        Ok(prepared) => prepared,
        Err(tool_result) => {
            return (
                route_tool_call(mcp_client, &tool_call.function.name),
                RunOneResult {
                    tool_result,
                    ok: false,
                    executed: true,
                    cached: false,
                },
            );
        }
    };

    if let Err(result) = confirm_tool_execution(tool_call, &prepared.args) {
        return (prepared.route, result);
    }

    if let Ok(guard) = GLOBAL_OS.lock() {
        if let Some(os_arc) = guard.as_ref() {
            if let Ok(os) = os_arc.lock() {
                if let Some(current_pid) = os.current_process_id() {
                    if let Some(proc) = os.get_process(current_pid) {
                        if !proc.allowed_tools.is_empty()
                            && !proc.allowed_tools.contains(&tool_call.function.name)
                        {
                            let content = format!(
                                "Error: tool '{}' is not in the allowed whitelist for this process.",
                                tool_call.function.name
                            );
                            return (
                                prepared.route,
                                RunOneResult {
                                    tool_result: ToolResult {
                                        tool_call_id: tool_call.id.clone(),
                                        content,
                                    },
                                    ok: false,
                                    executed: false,
                                    cached: false,
                                },
                            );
                        }
                    }
                }
            }
        }
    }

    if let Some(tool_result) = load_cached_tool_result(session_id, tool_call, &prepared.args) {
        return (
            prepared.route,
            RunOneResult {
                tool_result,
                ok: true,
                executed: false,
                cached: true,
            },
        );
    }

    // 不换行，以便完成状态用 \r 覆盖在同一行
    print!("{}", format_tool_status_running(&tool_call.function.name));
    let _ = std::io::stdout().flush();

    if let Err(run_result) = reserve_current_process_tool_call_budget(tool_call) {
        return (prepared.route, run_result);
    }

    if let Some(observer) = observer.as_deref_mut() {
        observer.on_tool_started(tool_call);
    }

    let result = execute_with_safe_retry(&prepared.route, &tool_call.function.name, || {
        execute_prepared_tool_call(
            session_id,
            mcp_client,
            shared_mcp_client,
            tool_call,
            &prepared,
            allowed_tool_names,
            observer,
        )
    });
    let run_result = finalize_execution_result(
        session_id,
        tool_call,
        &prepared,
        result,
        allowed_tool_names,
        true,
        false,
    );

    (prepared.route, run_result)
}

pub(super) fn execute_tool_calls(
    session_id: &str,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    tool_calls: &[ToolCall],
    allowed_tool_names: Option<&FastSet<String>>,
    observer: Option<&mut dyn ToolExecutionObserver>,
) -> Result<ExecuteToolCallsResult, Box<dyn Error>> {
    if tokio::runtime::Handle::try_current().is_ok() {
        return tokio::task::block_in_place(|| {
            execute_tool_calls_inner(
                session_id,
                mcp_client,
                shared_mcp_client,
                tool_calls,
                allowed_tool_names,
                observer,
            )
        });
    }
    execute_tool_calls_inner(
        session_id,
        mcp_client,
        shared_mcp_client,
        tool_calls,
        allowed_tool_names,
        observer,
    )
}

fn execute_tool_calls_inner(
    session_id: &str,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    tool_calls: &[ToolCall],
    allowed_tool_names: Option<&FastSet<String>>,
    mut observer: Option<&mut dyn ToolExecutionObserver>,
) -> Result<ExecuteToolCallsResult, Box<dyn Error>> {
    let mut executed_tool_calls = Vec::with_capacity(tool_calls.len());
    let mut tool_results = Vec::with_capacity(tool_calls.len());
    let mut cached_hits = Vec::with_capacity(tool_calls.len());
    let mut had_error = false;

    let mut idx = 0usize;
    while idx < tool_calls.len() {
        if crate::ai::tools::registry::common::is_tool_cancel_requested() {
            for deferred in &tool_calls[idx..] {
                println!("{}", format_tool_status_deferred(&deferred.function.name));
            }
            break;
        }

        // 当模型在一轮里批量发出多个只读、无副作用、且永不触发 barrier 的工具
        // 调用（如同时 read_file 多个文件）时，把这些连续调用并行执行以降低延迟。
        // 任何带副作用 / 需要 barrier / 流式输出的工具都走原有的顺序路径。
        let batch_len = parallel_safe_batch_len(mcp_client, &tool_calls[idx..]);
        if batch_len >= 2 {
            let batch = &tool_calls[idx..idx + batch_len];
            for tool_call in batch.iter() {
                crate::ai::driver::hooks::run_lifecycle_hook(
                    crate::ai::driver::hooks::HookEvent::BeforeTool,
                    Some(&tool_call.function.name),
                    None,
                );
            }
            let batch_results = run_parallel_readonly_batch(
                mcp_client,
                shared_mcp_client,
                session_id,
                batch,
                allowed_tool_names,
                &mut observer,
            );
            for (tool_call, (_route, run_result)) in batch.iter().zip(batch_results.into_iter()) {
                executed_tool_calls.push(tool_call.clone());
                cached_hits.push(run_result.cached);
                notify_tool_finished(&mut observer, tool_call, &run_result);
                print_run_status(tool_call, &run_result);
                crate::ai::driver::hooks::run_lifecycle_hook(
                    crate::ai::driver::hooks::HookEvent::AfterTool,
                    Some(&tool_call.function.name),
                    Some(run_result.ok),
                );
                tool_results.push(run_result.tool_result);
                had_error |= !run_result.ok;
            }
            idx += batch_len;
            continue;
        }

        let tool_call = &tool_calls[idx];
        let is_last = idx + 1 >= tool_calls.len();
        crate::ai::driver::hooks::run_lifecycle_hook(
            crate::ai::driver::hooks::HookEvent::BeforeTool,
            Some(&tool_call.function.name),
            None,
        );
        let (route, run_result) = run_one(
            mcp_client,
            shared_mcp_client,
            session_id,
            tool_call,
            allowed_tool_names,
            &mut observer,
        );
        let should_barrier = barrier::should_barrier_after(
            &route,
            tool_call,
            run_result.ok,
            &run_result.tool_result.content,
        );

        executed_tool_calls.push(tool_call.clone());
        cached_hits.push(run_result.cached);
        notify_tool_finished(&mut observer, tool_call, &run_result);
        print_run_status(tool_call, &run_result);
        crate::ai::driver::hooks::run_lifecycle_hook(
            crate::ai::driver::hooks::HookEvent::AfterTool,
            Some(&tool_call.function.name),
            Some(run_result.ok),
        );
        tool_results.push(run_result.tool_result);
        had_error |= !run_result.ok;

        if should_barrier && !is_last {
            for deferred in &tool_calls[idx + 1..] {
                println!("{}", format_tool_status_deferred(&deferred.function.name));
            }
            break;
        }

        if crate::ai::tools::registry::common::is_tool_cancel_requested() {
            for deferred in &tool_calls[idx + 1..] {
                println!("{}", format_tool_status_deferred(&deferred.function.name));
            }
            break;
        }
        idx += 1;
    }

    Ok(ExecuteToolCallsResult {
        executed_tool_calls,
        tool_results,
        cached_hits,
        had_error,
    })
}

/// 上限：单批并行只读工具的并发度，避免模型一次发起几十个调用时打满线程。
const PARALLEL_READONLY_MAX_CONCURRENCY: usize = 8;

/// 判断一个工具调用是否可安全并行执行：必须是 builtin 路由、只读（命中
/// `is_cacheable_tool_name` 的复用白名单且不在 mutating 列表）、且永不触发
/// barrier。MCP 工具（始终 barrier）、写类工具、命令执行、子 agent / 异步任务
/// 工具都会被排除，因此并行批次与顺序执行在语义上完全等价，只是更快。
fn is_parallel_safe_tool_call(mcp_client: &McpClient, tool_call: &ToolCall) -> bool {
    let name = &tool_call.function.name;
    if !is_cacheable_tool_name(name) {
        return false;
    }
    let route = route_tool_call(mcp_client, name);
    if !matches!(route, ToolRoute::Builtin) {
        return false;
    }
    barrier::rule_is_never(&route, name)
}

/// 返回从切片头部开始、连续可并行执行的工具数量（上限受并发度约束）。
fn parallel_safe_batch_len(mcp_client: &McpClient, tool_calls: &[ToolCall]) -> usize {
    let mut len = 0usize;
    for tool_call in tool_calls {
        if len >= PARALLEL_READONLY_MAX_CONCURRENCY {
            break;
        }
        if !is_parallel_safe_tool_call(mcp_client, tool_call) {
            break;
        }
        len += 1;
    }
    len
}

/// 并行执行一批只读工具，结果按输入顺序返回。每个线程使用独立的、无 observer
/// 的 `run_one`（只读工具不产生流式输出），共享的 `mcp_client` / `session_id`
/// 均为不可变引用，安全跨 `thread::scope` 线程共享。observer 的 started/finished
/// 回调仍由调用方按顺序触发，以保持原有契约。
fn run_parallel_readonly_batch(
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    session_id: &str,
    batch: &[ToolCall],
    allowed_tool_names: Option<&FastSet<String>>,
    observer: &mut Option<&mut dyn ToolExecutionObserver>,
) -> Vec<(ToolRoute, RunOneResult)> {
    // 在并发执行前，按顺序触发 on_tool_started，保持观察者看到的启动顺序稳定。
    if let Some(observer) = observer.as_deref_mut() {
        for tool_call in batch {
            observer.on_tool_started(tool_call);
        }
    }

    std::thread::scope(|scope| {
        let handles: Vec<_> = batch
            .iter()
            .map(|tool_call| {
                scope.spawn(move || {
                    let mut no_observer: Option<&mut dyn ToolExecutionObserver> = None;
                    run_one(
                        mcp_client,
                        shared_mcp_client,
                        session_id,
                        tool_call,
                        allowed_tool_names,
                        &mut no_observer,
                    )
                })
            })
            .collect();
        handles
            .into_iter()
            .zip(batch.iter())
            .map(|(h, tool_call)| {
                h.join().unwrap_or_else(|_| {
                    (
                        ToolRoute::Builtin,
                        RunOneResult {
                            tool_result: ToolResult {
                                tool_call_id: tool_call.id.clone(),
                                content: "Error: parallel tool execution thread panicked"
                                    .to_string(),
                            },
                            ok: false,
                            executed: true,
                            cached: false,
                        },
                    )
                })
            })
            .collect()
    })
}

fn notify_tool_finished(
    observer: &mut Option<&mut dyn ToolExecutionObserver>,
    tool_call: &ToolCall,
    run_result: &RunOneResult,
) {
    if let Some(observer) = observer.as_deref_mut() {
        observer.on_tool_finished(tool_call, run_result);
    }
}

fn load_cached_tool_result(
    session_id: &str,
    tool_call: &ToolCall,
    args: &Value,
) -> Option<ToolResult> {
    if !is_cacheable_tool_name(&tool_call.function.name) {
        return None;
    }
    let source = format!("session:{session_id}");
    let cache_key = build_tool_cache_key(&tool_call.function.name, args);
    let store = MemoryStore::from_env_or_config();
    let entries = store.recent(TOOL_CACHE_RECENT_LIMIT).ok()?;
    for entry in entries {
        if entry.category != "tool_cache" {
            continue;
        }
        if !is_tool_cache_entry_fresh(&entry) {
            continue;
        }
        if entry.source.as_deref() != Some(source.as_str()) {
            continue;
        }
        if entry.tags.first().map(String::as_str) != Some(tool_call.function.name.as_str()) {
            continue;
        }
        if entry.tags.get(1).map(String::as_str) != Some(cache_key.as_str()) {
            continue;
        }
        let payload = serde_json::from_str::<ToolCachePayload>(&entry.note).ok()?;
        if payload.tool_name != tool_call.function.name || payload.args != *args {
            continue;
        }
        if !tool_cache_validation_matches(&payload) {
            continue;
        }
        return Some(ToolResult {
            tool_call_id: tool_call.id.clone(),
            content: payload.result,
        });
    }
    None
}

fn is_tool_cache_entry_fresh(entry: &AgentMemoryEntry) -> bool {
    let Ok(timestamp) = DateTime::parse_from_rfc3339(&entry.timestamp) else {
        return false;
    };
    let timestamp = timestamp.with_timezone(&Utc);
    Utc::now().signed_duration_since(timestamp) <= Duration::minutes(TOOL_CACHE_TTL_MINUTES)
}

fn store_tool_cache_result(
    session_id: &str,
    tool_call: &ToolCall,
    args: &Value,
    tool_result: &ToolResult,
) {
    if !is_cacheable_tool_name(&tool_call.function.name) {
        return;
    }
    if tool_result.content.trim().is_empty() || tool_result.content.starts_with("Error:") {
        return;
    }
    let payload = ToolCachePayload {
        tool_name: tool_call.function.name.clone(),
        args: args.clone(),
        result: truncate_chars(&tool_result.content, TOOL_CACHE_MAX_RESULT_CHARS),
        file_fingerprints: collect_tool_cache_file_fingerprints(&tool_call.function.name, args),
    };
    let Ok(note) = serde_json::to_string(&payload) else {
        return;
    };
    let cache_key = build_tool_cache_key(&tool_call.function.name, args);
    let entry = AgentMemoryEntry {
        id: None,
        timestamp: Local::now().to_rfc3339(),
        category: "tool_cache".to_string(),
        note,
        tags: vec![tool_call.function.name.clone(), cache_key],
        source: Some(format!("session:{session_id}")),
        priority: Some(80),
        owner_pid: None,
        owner_pgid: None,
        image_path: None,
    };
    let store = MemoryStore::from_env_or_config();
    let _ = store.append(&entry);
    store.maintain_after_append();
}

fn is_cacheable_tool_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    let mutating = [
        "create",
        "delete",
        "remove",
        "update",
        "write",
        "save",
        "append",
        "insert",
        "rename",
        "move",
        "install",
        "run",
        "execute",
        "oauth",
        "open_browser",
        "report_event",
        "memory",
        "kill_terminal",
        "edit",
        "apply_patch",
    ];
    if mutating.iter().any(|needle| lower.contains(needle)) {
        return false;
    }
    let reusable = [
        "search", "find", "read", "get", "list", "view", "fetch", "export",
    ];
    reusable.iter().any(|needle| lower.contains(needle))
}

fn build_tool_cache_key(name: &str, args: &Value) -> String {
    use sha2::{Digest, Sha256};
    let args_json = serde_json::to_string(args).unwrap_or_else(|_| args.to_string());
    let digest = Sha256::digest(format!("{name}\n{args_json}").as_bytes());
    let mut s = String::with_capacity(32);
    for b in &digest[..16] {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn tool_cache_validation_matches(payload: &ToolCachePayload) -> bool {
    let current = collect_tool_cache_file_fingerprints(&payload.tool_name, &payload.args);
    current == payload.file_fingerprints
}

fn collect_tool_cache_file_fingerprints(
    tool_name: &str,
    args: &Value,
) -> Vec<CachedFileFingerprint> {
    let path = match tool_name {
        "read_file" | "read_file_lines" => args
            .get("file_path")
            .or_else(|| args.get("path"))
            .and_then(Value::as_str),
        _ => None,
    };
    path.and_then(cached_file_fingerprint_for_path)
        .into_iter()
        .collect()
}

fn cached_file_fingerprint_for_path(path: &str) -> Option<CachedFileFingerprint> {
    let meta = fs::metadata(path).ok()?;
    if !meta.is_file() {
        return None;
    }
    let modified_ms = meta
        .modified()
        .ok()
        .and_then(|mtime| mtime.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis().min(u64::MAX as u128) as u64);
    Some(CachedFileFingerprint {
        path: Path::new(path).display().to_string(),
        size: meta.len(),
        modified_ms,
    })
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    if max_chars == 0 || s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests;

pub(super) fn penalty_for_skill_tools(skill: &crate::ai::skills::SkillManifest) -> f64 {
    if skill.tools.is_empty() {
        return 0.0;
    }
    let tools = &skill.tools;
    let Ok(map) = TOOL_FAILURES.lock() else {
        return 0.0;
    };
    let mut score = 0.0f64;
    for t in tools {
        if let Some(c) = map.get_ref(t) {
            score += (*c as f64).min(10.0);
        }
    }
    score
}
