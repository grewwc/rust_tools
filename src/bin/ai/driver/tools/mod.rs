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
use std::time::{Duration as StdDuration, UNIX_EPOCH};

use crate::ai::{
    driver::print::{
        echo_tool_args, echo_tool_output, format_file_tool_target, format_tool_status_cached,
        format_tool_status_completed, format_tool_status_deferred, format_tool_status_failed,
        format_tool_status_running, format_tool_status_skipped,
        format_tool_status_with_file_target,
    },
    history::ToolExecutionOutcome,
    mcp::{McpClient, SharedMcpClient},
    ports::ToolExecOutput,
    tools as builtin_tools,
    tools::os_tools::GLOBAL_OS,
    tools::storage::memory_store::{AgentMemoryEntry, MemoryStore},
    types::{ToolCall, ToolResult},
};
use crate::commonw::prompt::prompt_yes_or_no_interruptible;

mod barrier;
mod oauth;
mod sync_task;

/// 供 driver 内部的显式命令复用同步 `task` 路径，保持子代理隔离、取消和证据持久化语义。
pub(in crate::ai::driver) fn execute_direct_subagent_task(
    tool_call_id: &str,
    args: &Value,
    hard_timeout: StdDuration,
    wrap_up_lead_time: Option<StdDuration>,
) -> Result<ToolResult, String> {
    sync_task::execute_sync_task_with_pre_timeout_wrap_up(
        tool_call_id,
        args,
        hard_timeout,
        wrap_up_lead_time,
    )
}

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
    /// 每个真实执行调用的结构化状态与环境签名；与展示正文解耦并单独持久化。
    pub(super) execution_outcomes: Vec<Option<ToolExecutionOutcome>>,
    /// 本轮是否有任何工具执行失败（`RunOneResult.ok == false`）。
    /// 结构化信号，供下游 reflection/evolution 判定 turn 质量时使用，
    /// 替代旧版扫描 assistant 答案文本找 "error"/"failed" 的脆弱做法。
    pub(super) had_error: bool,
}

impl ExecuteToolCallsResult {
    /// 转换为端口输出：透传真实派发的全部字段，供 ToolExecutor 链无损失消费
    /// （`assistant_messages` 恒为空，由中间件链在需要时填充）。
    pub(super) fn into_tool_exec_output(self) -> ToolExecOutput {
        let ExecuteToolCallsResult {
            executed_tool_calls,
            tool_results,
            cached_hits,
            execution_outcomes,
            had_error,
        } = self;
        ToolExecOutput {
            tool_results,
            assistant_messages: Vec::new(),
            executed_tool_calls,
            cached_hits,
            execution_outcomes,
            had_error,
        }
    }
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
const TOOL_CACHE_READ_FILE_TOOL: &str = "read_file";

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
    let args = parse_tool_args(tool_call)?;
    if let Some(stub_error) = overflow_stub_argument_error(tool_call, &args) {
        return Err(stub_error);
    }
    Ok(PreparedToolCall {
        route: route_tool_call(mcp_client, &tool_call.function.name),
        args,
    })
}

/// Detect tool arguments that are actually a context-overflow pointer stub
/// (`{"_context_overflow_truncated": ..., "archive_file_path": ..., "preview": ...}`)
/// rather than real parameters. The projection compressor archives oversized
/// *completed* calls in this shape, and a model that has such a stub in its
/// visible context sometimes re-emits the stub keys verbatim as a fresh call
/// (observed with apply_patch `patch` and write_file `content`). Each tool then
/// reports a misleading "parameter missing" error that invites an identical
/// retry (apply_patch retried the same stub shape three times in a row,
/// write_file twice), instead of converging.
/// Rejecting at the central dispatch point gives one unambiguous repair
/// instruction regardless of which tool is targeted.
fn overflow_stub_argument_error(tool_call: &ToolCall, args: &Value) -> Option<ToolResult> {
    let Some(map) = args.as_object() else {
        return None;
    };
    let marker_is_stub = map
        .get("_context_overflow_truncated")
        .is_some_and(|value| match value {
            // Runtime emits a JSON bool; a transcribing model often sends the
            // stringified forms instead. Accept all shapes.
            Value::Bool(true) => true,
            Value::String(text) => {
                matches!(text.trim().to_ascii_lowercase().as_str(), "true" | "1")
            }
            Value::Number(number) => number.as_u64() == Some(1),
            _ => false,
        });
    // Fallback shape: no marker but the full archive-pointer key set.
    let pointer_shape = ["original_chars", "archive_file_path", "preview"]
        .iter()
        .all(|key| map.contains_key(*key));
    if !(marker_is_stub || pointer_shape) {
        return None;
    }
    let archive_path = map
        .get("archive_file_path")
        .and_then(Value::as_str)
        .unwrap_or("<unknown>");
    Some(ToolResult {
        tool_call_id: tool_call.id.clone(),
        content: format!(
            "Error: the arguments for '{}' are a context-overflow pointer stub \
             (keys _context_overflow_truncated/original_chars/archive_file_path/preview), \
             not real parameters. The archived original text is at: {archive_path}. \
             Do NOT resend these stub keys. Regenerate the real arguments for this call \
             (re-derive the payload from the current file state, or if the payload exceeds \
             a tool's inline limit, write it to a temp file with write_file(temp=true) and \
             pass its path via the tool's file parameter, e.g. apply_patch `patch_file`).",
            tool_call.function.name
        ),
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

fn tool_visible_in_current_turn(
    available_tool_names: Option<&FastSet<String>>,
    tool_name: &str,
) -> bool {
    available_tool_names.is_some_and(|names| names.contains(tool_name))
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
        // 根据具体错误类型给出差异化建议，避免模型收到泛化提示后仍然反复犯同类错误
        let specific_hint = if err_lower.contains("context mismatch") {
            "Hunk context does not match the file. Re-read the file with `read_file` first, then build hunk context from the EXACT raw text (no line numbers, no truncation notices)."
        } else if err_lower.contains("ambiguous patch") {
            "Multiple locations match. Add more unique surrounding context lines (2–4 extra) so the tool can pin the exact location."
        } else if err_lower.contains("no hunks found") || err_lower.contains("invalid hunk") {
            "Patch could not be parsed. Use raw unified-diff format starting with `@@ -old_start,old_count +new_start,new_count @@` or the `*** Begin Patch` / `*** Update File:` envelope."
        } else if err_lower.contains("missing file_path") {
            "Provide `file_path` as a parameter, or wrap the patch in a `*** Begin Patch` / `*** Update File: <path>` envelope."
        } else if err_lower.contains("missing patch") {
            "`patch` must be a string, not a JSON object or array. Pass the patch text as a string value."
        } else if err_lower.contains("patch target mismatch") {
            "The `file_path` arg does not match the target in the envelope. Use consistent file paths, or omit `file_path` and let the envelope specify the target."
        } else {
            "Use raw unified-diff hunks or the `*** Begin Patch` envelope format. Re-read the file before building hunk context."
        };
        let write_file_hint = if tool_visible_in_current_turn(available_tool_names, "write_file") {
            " If replacing the whole file, use `write_file` instead."
        } else {
            ""
        };
        return Some(format!("Suggestion: {}", specific_hint) + write_file_hint);
    }

    // Note: mcp_feishu_docs_search has been removed; users must provide direct Feishu URLs.

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

    if tool_name == "execute_command" {
        let mut fallback = Vec::new();
        if tool_visible_in_current_turn(available_tool_names, "read_file") {
            fallback.push("read files (whole or precise line ranges) with `read_file`");
        }
        if tool_visible_in_current_turn(available_tool_names, "tree") {
            fallback.push("inspect directory layout with `tree`");
        }
        if !fallback.is_empty() {
            return Some(format!(
                "Suggestion: if this failure is intrinsic (not a transient I/O error), break the command into smaller pieces or {} instead of running shell just to inspect state.",
                fallback.join(", ")
            ));
        }
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
    shared_mcp_client: &SharedMcpClient,
    tool_call: &ToolCall,
    prepared: &PreparedToolCall,
    observer: &mut Option<&mut dyn ToolExecutionObserver>,
) -> Result<ToolResult, String> {
    match &prepared.route {
        ToolRoute::Builtin => {
            if tool_call.function.name == "task" {
                sync_task::execute_sync_task(&tool_call.id, &prepared.args).map(|tr| tr)
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
    if !crate::ai::driver::runtime_ctx::terminal_output_enabled() {
        return;
    }
    let name = &tool_call.function.name;
    let inline_file_target = matches!(name.as_str(), "read_file" | "write_file" | "apply_patch")
        .then(|| format_file_tool_target(name, &tool_call.function.arguments))
        .flatten();
    let with_file_target = |status_line: String| {
        if let Some(target) = inline_file_target.as_deref() {
            format_tool_status_with_file_target(status_line, target)
        } else {
            status_line
        }
    };

    if run_result.cached {
        println!("{}", with_file_target(format_tool_status_cached(name)));
    } else if !run_result.executed {
        println!("{}", with_file_target(format_tool_status_skipped(name)));
    } else if run_result.ok {
        // 已执行的工具：用 \r 回到行首覆盖 running 状态，保持同一行
        print!(
            "\r\x1b[2K{}\n",
            with_file_target(format_tool_status_completed(name))
        );
    } else {
        print!(
            "\r\x1b[2K{}\n",
            with_file_target(format_tool_status_failed(name))
        );
    }

    // 部分工具的输出对用户有较高可见性价值，额外把其内容回显到终端。
    // 具体哪些工具回显由工具自身提交的 `ToolDisplayConfig` 控制，
    // 这里不感知具体工具名，便于后续扩展。
    if run_result.ok || run_result.cached {
        echo_tool_args(name, &tool_call.function.arguments);
    }
    if run_result.ok {
        echo_tool_output(
            name,
            &run_result.tool_result.content,
            &tool_call.function.arguments,
        );
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

fn subagent_tool_phase(tool_name: &str, args: &Value) -> String {
    let target = args
        .get("file_path")
        .or_else(|| args.get("path"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty());
    match target {
        Some(target) => format!("using {tool_name} · {target}"),
        None => format!("using {tool_name}"),
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

    // 实时 side-note：lead-agent → subagent（或前景）。在工具分发层直接处理，无需 MCP 往返。
    if tool_call.function.name == "send_side_note" {
        let res = crate::ai::tools::service::side_note::handle_send_side_note(
            &tool_call.id,
            &prepared.args,
        );
        let run_result = finalize_execution_result(
            session_id,
            tool_call,
            &prepared,
            res,
            allowed_tool_names,
            true,
            false,
        );
        return (prepared.route, run_result);
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

    let progress = subagent_tool_phase(&tool_call.function.name, &prepared.args);
    crate::ai::driver::runtime_ctx::publish_subagent_phase(&progress);

    if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
        // 不换行，以便完成状态用 \r 覆盖在同一行
        print!("{}", format_tool_status_running(&tool_call.function.name));
        let _ = std::io::stdout().flush();
    }

    if let Err(run_result) = reserve_current_process_tool_call_budget(tool_call) {
        return (prepared.route, run_result);
    }

    if let Some(observer) = observer.as_deref_mut() {
        observer.on_tool_started(tool_call);
    }

    let result = execute_with_safe_retry(&prepared.route, &tool_call.function.name, || {
        execute_prepared_tool_call(shared_mcp_client, tool_call, &prepared, observer)
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
    let mut execution_outcomes = Vec::with_capacity(tool_calls.len());
    let mut had_error = false;
    let execution_cwd = crate::ai::driver::runtime_ctx::effective_cwd()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_default();

    let mut idx = 0usize;
    while idx < tool_calls.len() {
        if crate::ai::tools::registry::common::is_tool_cancel_requested() {
            if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
                for deferred in &tool_calls[idx..] {
                    println!("{}", format_tool_status_deferred(&deferred.function.name));
                }
            }
            break;
        }

        // 当模型在一轮里批量发出多个只读、无副作用、且永不触发 barrier 的工具
        // 调用时，把这些连续调用并行执行以降低延迟。
        //
        // 例外：`read_file` 这类源码 grounding 工具必须强制
        // 串行。它的本地执行本身很快，但返回体积大、进入上下文的成本高；若同轮
        // 批量并行，极易放大证据面、冲击 history 预算，并诱导模型继续“读更多”
        // 而不是基于已有证据收敛。任何带副作用 / 需要 barrier / 流式输出的工具
        // 也继续走原有顺序路径。
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
            for (tool_call, (route, run_result)) in batch.iter().zip(batch_results.into_iter()) {
                executed_tool_calls.push(tool_call.clone());
                cached_hits.push(run_result.cached);
                execution_outcomes.push(Some(tool_execution_outcome(
                    session_id,
                    &execution_cwd,
                    &route,
                    tool_call,
                    run_result.ok,
                )));
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
        execution_outcomes.push(Some(tool_execution_outcome(
            session_id,
            &execution_cwd,
            &route,
            tool_call,
            run_result.ok,
        )));
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
            if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
                for deferred in &tool_calls[idx + 1..] {
                    println!("{}", format_tool_status_deferred(&deferred.function.name));
                }
            }
            break;
        }

        if crate::ai::tools::registry::common::is_tool_cancel_requested() {
            if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
                for deferred in &tool_calls[idx + 1..] {
                    println!("{}", format_tool_status_deferred(&deferred.function.name));
                }
            }
            break;
        }
        idx += 1;
    }

    Ok(ExecuteToolCallsResult {
        executed_tool_calls,
        tool_results,
        cached_hits,
        execution_outcomes,
        had_error,
    })
}

fn tool_execution_outcome(
    session_id: &str,
    cwd: &str,
    route: &ToolRoute,
    tool_call: &ToolCall,
    succeeded: bool,
) -> ToolExecutionOutcome {
    fn canonicalize_json(value: Value) -> Value {
        match value {
            Value::Array(values) => {
                Value::Array(values.into_iter().map(canonicalize_json).collect())
            }
            Value::Object(values) => {
                let mut entries = values.into_iter().collect::<Vec<_>>();
                entries.sort_by(|left, right| left.0.cmp(&right.0));
                Value::Object(
                    entries
                        .into_iter()
                        .map(|(key, value)| (key, canonicalize_json(value)))
                        .collect(),
                )
            }
            scalar => scalar,
        }
    }

    let arguments = serde_json::from_str::<Value>(&tool_call.function.arguments)
        .map(canonicalize_json)
        .map(|value| json!({ "json": value }))
        .unwrap_or_else(|_| json!({ "raw": tool_call.function.arguments }));
    let route = match route {
        ToolRoute::Builtin => json!({ "kind": "builtin" }),
        ToolRoute::Mcp {
            server_name,
            tool_name,
        } => json!({
            "kind": "mcp",
            "server": server_name,
            "tool": tool_name,
        }),
    };
    let execution_signature = json!({
        "version": 1,
        "session": session_id,
        "cwd": cwd,
        "route": route,
        "tool": tool_call.function.name,
        "arguments": arguments,
    })
    .to_string();

    ToolExecutionOutcome {
        tool_call_id: tool_call.id.clone(),
        execution_signature,
        succeeded,
    }
}

/// 上限：单批并行只读工具的并发度，避免模型一次发起几十个调用时打满线程。
const PARALLEL_READONLY_MAX_CONCURRENCY: usize = 8;

/// 判断一个工具调用是否可安全并行执行：必须是 builtin 路由、只读（命中
/// `is_cacheable_tool_name` 的复用白名单且不在 mutating 列表）、且永不触发
/// barrier。
///
/// `read_file` 属于高精度 grounding 入口：虽然技术上无副作
/// 用，但它返回的大块证据进入上下文的成本远高于执行成本，必须串行以压缩证据面，
/// 帮助模型沿“定位 -> 阅读 -> 判断 -> 修改”的收敛路径推进。
///
/// MCP 工具（始终 barrier）、写类工具、命令执行、子 agent / 异步任务工具都会
/// 被排除，因此并行批次与顺序执行在语义上完全等价，只是更快。
fn is_parallel_safe_tool_call(mcp_client: &McpClient, tool_call: &ToolCall) -> bool {
    let name = &tool_call.function.name;
    // 源码阅读工具必须强制串行：执行本身便宜，但并行批量返回会迅速把
    // 大量精确证据塞进上下文，放大 evidence flooding 风险并干扰后续收敛。
    if matches!(name.as_str(), "read_file" | "read_file_lines") {
        return false;
    }
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

    // 批次线程是 `std::thread::scope` 创建的原始 OS 线程，tokio task-local
    // `DRIVER_CTX` 在这些线程上不可见。先在 driver 任务内捕获上下文，再在每条
    // 批次线程安装回退，否则依赖会话上下文的只读工具（如 search_overflow 解析
    // session assets 目录）在批量并行时会硬失败。
    let batch_ctx = crate::ai::driver::runtime_ctx::try_current();
    std::thread::scope(|scope| {
        let handles: Vec<_> = batch
            .iter()
            .map(|tool_call| {
                let ctx = batch_ctx.clone();
                scope.spawn(move || {
                    let _ctx_fallback = ctx.as_ref().map(|driver_ctx| {
                        crate::ai::driver::runtime_ctx::DriverCtxThreadFallback::install(Some(
                            std::sync::Arc::clone(driver_ctx),
                        ))
                    });
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
    if !should_store_or_load_tool_cache(&tool_call.function.name, args) {
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
    if !should_store_or_load_tool_cache(&tool_call.function.name, args) {
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

fn should_store_or_load_tool_cache(tool_name: &str, args: &Value) -> bool {
    if !is_cacheable_tool_name(tool_name) {
        return false;
    }
    // 目前只有 read_file 缓存具备可校验的环境指纹。搜索/目录/MCP 读取类工具虽然是
    // 只读，但其结果会随外部状态变化；没有对应 fingerprint 时不落盘/不命中缓存，
    // 避免把陈旧检索结果直接回放给模型。
    tool_name == TOOL_CACHE_READ_FILE_TOOL
        && !collect_tool_cache_file_fingerprints(tool_name, args).is_empty()
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
        TOOL_CACHE_READ_FILE_TOOL => args
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
