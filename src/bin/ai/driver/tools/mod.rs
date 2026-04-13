use chrono::{DateTime, Duration, Local, Utc};
use colored::Colorize;
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use serde_json::{Value, json};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::thread;
use rust_tools::commonw::FastMap;
use std::sync::{LazyLock, Mutex};

use crate::ai::{
    mcp::McpClient,
    tools as builtin_tools,
    types::{ToolCall, ToolResult},
};
use crate::commonw::prompt::prompt_yes_or_no_interruptible;

mod barrier;
mod oauth;

static TOOL_FAILURES: LazyLock<Mutex<FastMap<String, usize>>> =
    LazyLock::new(|| Mutex::new(FastMap::default()));

#[derive(Debug, Clone)]
enum ToolRoute {
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
) -> Result<PreparedToolCall, ToolResult> {
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

fn remediation_hint(tool_name: &str, err: &str) -> Option<String> {
    let err_lower = err.to_lowercase();

    if tool_name == "mcp_feishu_docs_get_text_by_url" && err_lower.contains("unsupported url") {
        return Some(
            "Suggestion: this tool only works for supported Feishu/Lark docs URLs. Do not retry with the same URL. Use mcp_feishu_docs_search to find the document first, or ask the user for a direct Feishu docs/wiki/sheet URL.".to_string(),
        );
    }

    if err_lower.contains("failed to parse arguments") || err_lower.contains("invalid type") {
        return Some(
            "Suggestion: fix the tool arguments to match the declared JSON schema before retrying."
                .to_string(),
        );
    }

    if err_lower.contains("no such file") || err_lower.contains("not found") {
        return Some(
            "Suggestion: verify the path or identifier first, or use a search/list tool to discover the correct target before retrying.".to_string(),
        );
    }

    if err_lower.contains("timeout") || err_lower.contains("timed out") {
        return Some(
            "Suggestion: retry once with a narrower query or a smaller scope. If it still fails, switch to another tool or ask the user.".to_string(),
        );
    }

    None
}

fn format_tool_error(tool_call: &ToolCall, err: &str) -> ToolResult {
    ToolResult {
        tool_call_id: tool_call.id.clone(),
        content: if let Some(hint) = remediation_hint(&tool_call.function.name, err) {
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
    mcp_client: &McpClient,
    tool_call: &ToolCall,
    prepared: &PreparedToolCall,
    observer: &mut Option<&mut dyn ToolExecutionObserver>,
) -> Result<ToolResult, String> {
    match &prepared.route {
        ToolRoute::Builtin => {
            if tool_call.function.name == "execute_command" {
                builtin_tools::command_tools::execute_command_streaming(&prepared.args, |chunk| {
                    if let Some(observer) = observer.as_deref_mut() {
                        observer.on_tool_stream(tool_call, chunk);
                    }
                })
                .map(|content| ToolResult {
                    tool_call_id: tool_call.id.clone(),
                    content,
                })
            } else {
                builtin_tools::execute_tool_call_with_args(
                    &tool_call.id,
                    &tool_call.function.name,
                    &prepared.args,
                )
                .map_err(|e| e.to_string())
            }
        }
        ToolRoute::Mcp {
            server_name,
            tool_name,
        } => oauth::execute_mcp_tool_call(
            mcp_client,
            tool_call,
            server_name,
            tool_name,
            &prepared.args,
        ),
    }
}

fn execute_prepared_builtin_tool_call(
    tool_call: &ToolCall,
    prepared: &PreparedToolCall,
) -> Result<ToolResult, String> {
    builtin_tools::execute_tool_call_with_args(
        &tool_call.id,
        &tool_call.function.name,
        &prepared.args,
    )
    .map_err(|e| e.to_string())
}

fn record_tool_failure(tool_name: &str) {
    if let Ok(mut map) = TOOL_FAILURES.lock() {
        let counter = map.entry(tool_name.to_string()).or_insert(0);
        *counter = counter.saturating_add(1).min(100);
    }
}

fn finalize_execution_result(
    session_id: &str,
    tool_call: &ToolCall,
    prepared: &PreparedToolCall,
    result: Result<ToolResult, String>,
    executed: bool,
    cached: bool,
) -> RunOneResult {
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
            tool_result: format_tool_error(tool_call, &err),
            ok: false,
            executed,
            cached,
        },
    };
    if run_result.executed && !run_result.ok {
        record_tool_failure(&tool_call.function.name);
    }
    run_result
}

fn is_parallel_task_tool_call(tool_call: &ToolCall, prepared: &PreparedToolCall) -> bool {
    matches!(prepared.route, ToolRoute::Builtin) && tool_call.function.name == "task"
}

fn print_run_status(tool_call: &ToolCall, run_result: &RunOneResult) {
    let name = &tool_call.function.name;
    if run_result.cached {
        println!("\n[Cached] {}", name.bright_blue());
    } else if !run_result.executed {
        println!("\n[Skipped] {}", name.yellow());
    } else if run_result.ok {
        println!("\n[Completed] {}", name.green());
    } else {
        println!("\n[Failed] {}", name.red());
    }
}

fn parallel_task_batch_len(mcp_client: &McpClient, tool_calls: &[ToolCall], start: usize) -> usize {
    let mut len = 0usize;
    for tool_call in &tool_calls[start..] {
        let Ok(prepared) = prepare_tool_call(mcp_client, tool_call) else {
            break;
        };
        if !is_parallel_task_tool_call(tool_call, &prepared) {
            break;
        }
        len += 1;
    }
    len
}

fn execute_parallel_task_batch(
    session_id: &str,
    mcp_client: &McpClient,
    tool_calls: &[ToolCall],
) -> Vec<(ToolRoute, RunOneResult)> {
    let mut ordered_results: Vec<Option<(ToolRoute, RunOneResult)>> =
        std::iter::repeat_with(|| None).take(tool_calls.len()).collect();
    let mut pending = Vec::new();

    for (idx, tool_call) in tool_calls.iter().enumerate() {
        let prepared = match prepare_tool_call(mcp_client, tool_call) {
            Ok(prepared) => prepared,
            Err(tool_result) => {
                ordered_results[idx] = Some((
                    route_tool_call(mcp_client, &tool_call.function.name),
                    RunOneResult {
                        tool_result,
                        ok: false,
                        executed: true,
                        cached: false,
                    },
                ));
                continue;
            }
        };

        if let Err(result) = confirm_tool_execution(tool_call, &prepared.args) {
            ordered_results[idx] = Some((prepared.route, result));
            continue;
        }

        if let Some(tool_result) = load_cached_tool_result(session_id, tool_call, &prepared.args) {
            ordered_results[idx] = Some((
                prepared.route,
                RunOneResult {
                    tool_result,
                    ok: true,
                    executed: false,
                    cached: true,
                },
            ));
            continue;
        }

        pending.push((idx, tool_call.clone(), prepared));
    }

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(pending.len());
        for (idx, tool_call, prepared) in pending {
            handles.push(scope.spawn(move || {
                let result = execute_prepared_builtin_tool_call(&tool_call, &prepared);
                (idx, tool_call, prepared, result)
            }));
        }

        for handle in handles {
            match handle.join() {
                Ok((idx, tool_call, prepared, result)) => {
                    let run_result = finalize_execution_result(
                        session_id,
                        &tool_call,
                        &prepared,
                        result,
                        true,
                        false,
                    );
                    ordered_results[idx] = Some((prepared.route, run_result));
                }
                Err(_) => {
                    // This path should be unreachable in normal operation, but keep the
                    // batch resilient if one worker panics.
                }
            }
        }
    });

    ordered_results
        .into_iter()
        .enumerate()
        .map(|(idx, item)| {
            item.unwrap_or_else(|| {
                let tool_call = &tool_calls[idx];
                (
                    route_tool_call(mcp_client, &tool_call.function.name),
                    RunOneResult {
                        tool_result: format_tool_error(
                            tool_call,
                            "parallel task worker panicked before producing a result",
                        ),
                        ok: false,
                        executed: true,
                        cached: false,
                    },
                )
            })
        })
        .collect()
}

fn run_one(
    mcp_client: &McpClient,
    session_id: &str,
    tool_call: &ToolCall,
    observer: &mut Option<&mut dyn ToolExecutionObserver>,
) -> (ToolRoute, RunOneResult) {
    let prepared = match prepare_tool_call(mcp_client, tool_call) {
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

    println!("\n[Running] {}", tool_call.function.name.cyan());

    if let Some(observer) = observer.as_deref_mut() {
        observer.on_tool_started(tool_call);
    }

    let result = execute_prepared_tool_call(mcp_client, tool_call, &prepared, observer);
    let run_result = finalize_execution_result(session_id, tool_call, &prepared, result, true, false);

    (prepared.route, run_result)
}

pub(super) fn execute_tool_calls(
    session_id: &str,
    mcp_client: &McpClient,
    tool_calls: &[ToolCall],
    observer: Option<&mut dyn ToolExecutionObserver>,
) -> Result<ExecuteToolCallsResult, Box<dyn Error>> {
    if tokio::runtime::Handle::try_current().is_ok() {
        return tokio::task::block_in_place(|| {
            execute_tool_calls_inner(session_id, mcp_client, tool_calls, observer)
        });
    }
    execute_tool_calls_inner(session_id, mcp_client, tool_calls, observer)
}

fn execute_tool_calls_inner(
    session_id: &str,
    mcp_client: &McpClient,
    tool_calls: &[ToolCall],
    mut observer: Option<&mut dyn ToolExecutionObserver>,
) -> Result<ExecuteToolCallsResult, Box<dyn Error>> {
    let mut executed_tool_calls = Vec::with_capacity(tool_calls.len());
    let mut tool_results = Vec::with_capacity(tool_calls.len());
    let mut cached_hits = Vec::with_capacity(tool_calls.len());

    let mut idx = 0usize;
    while idx < tool_calls.len() {
        if crate::ai::tools::registry::common::is_tool_cancel_requested() {
            for deferred in &tool_calls[idx..] {
                println!("\n[Deferred] {}", deferred.function.name.yellow());
            }
            break;
        }

        let batch_len = parallel_task_batch_len(mcp_client, tool_calls, idx);
        if batch_len > 1 {
            let batch = &tool_calls[idx..idx + batch_len];
            let batch_results = execute_parallel_task_batch(session_id, mcp_client, batch);
            for (tool_call, (route, run_result)) in batch.iter().zip(batch_results.into_iter()) {
                executed_tool_calls.push(tool_call.clone());
                cached_hits.push(run_result.cached);
                let should_barrier = barrier::should_barrier_after(
                    &route,
                    tool_call,
                    run_result.ok,
                    &run_result.tool_result.content,
                );
                notify_tool_finished(&mut observer, tool_call, &run_result);
                print_run_status(tool_call, &run_result);
                tool_results.push(run_result.tool_result);
                if should_barrier {
                    for deferred in &tool_calls[idx + batch_len..] {
                        println!("\n[Deferred] {}", deferred.function.name.yellow());
                    }
                    return Ok(ExecuteToolCallsResult {
                        executed_tool_calls,
                        tool_results,
                        cached_hits,
                    });
                }
                if crate::ai::tools::registry::common::is_tool_cancel_requested() {
                    for deferred in &tool_calls[idx + batch_len..] {
                        println!("\n[Deferred] {}", deferred.function.name.yellow());
                    }
                    return Ok(ExecuteToolCallsResult {
                        executed_tool_calls,
                        tool_results,
                        cached_hits,
                    });
                }
            }
            idx += batch_len;
            continue;
        }

        let tool_call = &tool_calls[idx];
        let is_last = idx + 1 >= tool_calls.len();
        let (route, run_result) = run_one(mcp_client, session_id, tool_call, &mut observer);
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
        tool_results.push(run_result.tool_result);

        if should_barrier && !is_last {
            for deferred in &tool_calls[idx + 1..] {
                println!("\n[Deferred] {}", deferred.function.name.yellow());
            }
            break;
        }

        if crate::ai::tools::registry::common::is_tool_cancel_requested() {
            for deferred in &tool_calls[idx + 1..] {
                println!("\n[Deferred] {}", deferred.function.name.yellow());
            }
            break;
        }
        idx += 1;
    }

    Ok(ExecuteToolCallsResult {
        executed_tool_calls,
        tool_results,
        cached_hits,
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

fn load_cached_tool_result(session_id: &str, tool_call: &ToolCall, args: &Value) -> Option<ToolResult> {
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

fn store_tool_cache_result(session_id: &str, tool_call: &ToolCall, args: &Value, tool_result: &ToolResult) {
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
    };
    let store = MemoryStore::from_env_or_config();
    let _ = store.append(&entry);
    store.maintain_after_append();
}

fn is_cacheable_tool_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    let mutating = [
        "create", "delete", "remove", "update", "write", "save", "append", "insert",
        "rename", "move", "install", "run", "execute", "oauth", "open_browser",
        "report_event", "memory", "kill_terminal", "edit", "apply_patch",
    ];
    if mutating.iter().any(|needle| lower.contains(needle)) {
        return false;
    }
    let reusable = ["search", "read", "get", "list", "view", "fetch", "export"];
    reusable.iter().any(|needle| lower.contains(needle))
}

fn build_tool_cache_key(name: &str, args: &Value) -> String {
    let args_json = serde_json::to_string(args).unwrap_or_else(|_| args.to_string());
    format!("{:x}", md5::compute(format!("{name}\n{args_json}")))
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
mod tests {
    use super::{
        build_tool_cache_key, is_cacheable_tool_name, is_tool_cache_entry_fresh,
        parallel_task_batch_len, TOOL_CACHE_TTL_MINUTES,
    };
    use crate::ai::mcp::McpClient;
    use crate::ai::types::{FunctionCall, ToolCall};
    use crate::ai::tools::storage::memory_store::AgentMemoryEntry;
    use chrono::{Duration, Utc};
    use serde_json::json;

    #[test]
    fn cacheable_tool_name_prefers_read_only_tools() {
        assert!(is_cacheable_tool_name("read_file"));
        assert!(is_cacheable_tool_name("grep_search"));
        assert!(!is_cacheable_tool_name("create_file"));
        assert!(!is_cacheable_tool_name("execute_command"));
    }

    #[test]
    fn tool_cache_key_is_stable_for_same_args() {
        let key1 = build_tool_cache_key("read_file", &json!({"path":"a","start":1}));
        let key2 = build_tool_cache_key("read_file", &json!({"path":"a","start":1}));
        let key3 = build_tool_cache_key("read_file", &json!({"path":"a","start":2}));
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn tool_cache_entry_obeys_ttl() {
        let fresh = AgentMemoryEntry {
            id: None,
            timestamp: Utc::now().to_rfc3339(),
            category: "tool_cache".to_string(),
            note: "{}".to_string(),
            tags: Vec::new(),
            source: None,
            priority: Some(80),
        };
        let stale = AgentMemoryEntry {
            timestamp: (Utc::now() - Duration::minutes(TOOL_CACHE_TTL_MINUTES + 1)).to_rfc3339(),
            ..fresh.clone()
        };
        assert!(is_tool_cache_entry_fresh(&fresh));
        assert!(!is_tool_cache_entry_fresh(&stale));
    }

    fn tool_call(name: &str) -> ToolCall {
        ToolCall {
            id: format!("call-{name}"),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: "{}".to_string(),
            },
        }
    }

    #[test]
    fn parallel_task_batch_len_only_groups_contiguous_task_calls() {
        let client = McpClient::new();
        let tool_calls = vec![tool_call("task"), tool_call("task"), tool_call("read_file"), tool_call("task")];

        assert_eq!(parallel_task_batch_len(&client, &tool_calls, 0), 2);
        assert_eq!(parallel_task_batch_len(&client, &tool_calls, 2), 0);
        assert_eq!(parallel_task_batch_len(&client, &tool_calls, 3), 1);
    }
}

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
        if let Some(c) = map.get(t) {
            score += (*c as f64).min(10.0);
        }
    }
    score
}
