use crate::ai::{
    driver::tools::{self, ExecuteToolCallsResult},
    history::{Message, ROLE_INTERNAL_NOTE},
    mcp::{McpClient, SharedMcpClient},
    stream::clamp_line_to_terminal_row_with_reserve,
    tools::{storage::file_store::FileStore, task_tools},
    types::{App, ToolCall},
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    io::Write,
    path::PathBuf,
};

use super::super::persistence::persist_pending_turn_messages;
use super::super::{
    MAX_TOOL_RESULT_LINE_TRIM_CHARS, TOOL_OVERFLOW_PREVIEW_CHARS,
    iteration::no_tool_handoff_note,
    max_tool_result_inline_chars,
    types::{IterationExecution, PreparedToolResult, ToolCallExecution, TurnLoopStep},
};
use super::{
    messaging::{
        append_cached_tool_results_note, append_message_pair, append_tool_result_messages,
        parse_prune_meta_and_update_marks, record_final_stream_response, record_hidden_self_note,
        record_tool_inspection_artifacts,
    },
    overflow::{build_model_overflow_stub, summarize_large_tool_output, write_tool_overflow_file},
    preview::{build_terminal_preview, tail_chars},
};
use crate::ai::driver::print::{
    format_tool_output_line, format_tool_output_prefix, print_tool_command_line,
    print_tool_note_line, sanitize_for_terminal,
};
use crate::ai::theme::{ACCENT_MUTED, ACCENT_RULE, RESET};

/// 适合"中段按行裁剪"的非精确概览工具。
///
/// read_file(_lines) 的每一行都可能是
/// agent 后续判断需要引用的精确证据，不能做有损中段抽样；这些工具只允许在
/// 超过 inline 上限后 offload 到 session 文件，并在模型上下文里保留 path + stub。
fn supports_line_trim(tool_name: &str) -> bool {
    matches!(tool_name, "tree" | "ast_outline")
}

/// 把"中等大"（介于 MAX_TOOL_RESULT_LINE_TRIM_CHARS 和 MAX_TOOL_RESULT_INLINE_CHARS 之间）
/// 的结构化输出折叠为：头 N 行 + 命中关键词的若干行 + 尾 M 行 + 中段标注。
/// 不写盘、不破坏整体语义，只是把"中段冗余"压掉。
fn line_trim_middle(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    if total_lines <= 80 {
        return content.to_string();
    }

    let head_lines = 40usize;
    let tail_lines = 20usize;

    let mut head = Vec::with_capacity(head_lines);
    for line in lines.iter().take(head_lines) {
        head.push(*line);
    }
    let tail_start = total_lines.saturating_sub(tail_lines);
    let mut tail = Vec::with_capacity(tail_lines);
    if tail_start > head_lines {
        for line in lines.iter().skip(tail_start) {
            tail.push(*line);
        }
    }

    // 在中段（head_lines..tail_start）按关键字采样 8 行
    let mut key_lines: Vec<(usize, &str)> = Vec::new();
    if tail_start > head_lines {
        for (i, line) in lines.iter().enumerate().take(tail_start).skip(head_lines) {
            let lower = line.to_ascii_lowercase();
            let important = lower.contains("error")
                || lower.contains("fail")
                || lower.contains("panic")
                || lower.contains("warn")
                || lower.contains("todo")
                || lower.contains("fixme")
                || lower.contains("//!")
                || lower.contains("///")
                || lower.starts_with("fn ")
                || lower.starts_with("pub fn ")
                || lower.starts_with("impl ")
                || lower.starts_with("struct ")
                || lower.starts_with("trait ")
                || lower.starts_with("enum ")
                || lower.starts_with("#[")
                || lower.contains(": error")
                || lower.contains(": warning");
            if important {
                key_lines.push((i, *line));
                if key_lines.len() >= 8 {
                    break;
                }
            }
        }
    }

    let omitted_count = total_lines.saturating_sub(head_lines + tail.len());
    let mut out = String::with_capacity(content.len() / 2);
    for line in &head {
        out.push_str(line);
        out.push('\n');
    }
    if !key_lines.is_empty() {
        out.push_str(&format!(
            "\n... [middle trimmed: {} lines folded; key-match samples below]\n",
            omitted_count.saturating_sub(key_lines.len())
        ));
        for (idx, line) in &key_lines {
            out.push_str(&format!("L{idx}: {line}\n"));
        }
        out.push_str("...\n");
    } else {
        out.push_str(&format!(
            "\n... [middle trimmed: {} lines folded]\n",
            omitted_count
        ));
    }
    for line in &tail {
        out.push_str(line);
        out.push('\n');
    }
    out
}

pub(in crate::ai::driver::turn_runtime) fn prepare_tool_result(
    app: &App,
    tool_name: &str,
    content: &str,
) -> PreparedToolResult {
    let inline_limit = max_tool_result_inline_chars(&app.current_model);
    let char_count = content.chars().count();
    if char_count <= MAX_TOOL_RESULT_LINE_TRIM_CHARS {
        return PreparedToolResult {
            content_for_model: content.to_string(),
            content_for_terminal: build_terminal_preview(tool_name, content),
        };
    }

    if char_count <= inline_limit && supports_line_trim(tool_name) {
        let trimmed = line_trim_middle(content);
        // 复用 trimmed 的字节长度做廉价短路：trimmed 是从 content 里挑选若干行
        // 拼接出来的（可能改动；保留 ASCII / UTF-8 不变），如果字节更短就一定是
        // 字符更短，不必再做完整 chars().count() 双扫描。
        if trimmed.len() < content.len() && trimmed.chars().count() < char_count {
            return PreparedToolResult {
                content_for_model: trimmed,
                content_for_terminal: build_terminal_preview(tool_name, content),
            };
        }
    }

    if char_count <= inline_limit {
        return PreparedToolResult {
            content_for_model: content.to_string(),
            content_for_terminal: build_terminal_preview(tool_name, content),
        };
    }

    let summary = summarize_large_tool_output(content);
    let path = write_tool_overflow_file(app, tool_name, &summary.body).ok();
    let content_for_model = build_model_overflow_stub(path.as_ref(), &summary);
    let content_for_terminal = if let Some(path) = path {
        format!(
            "{}\n[Saved full output to {}]\n",
            build_terminal_preview(
                tool_name,
                &tail_chars(&summary.body, TOOL_OVERFLOW_PREVIEW_CHARS)
            ),
            path.display()
        )
    } else {
        build_terminal_preview(
            tool_name,
            &tail_chars(&summary.body, TOOL_OVERFLOW_PREVIEW_CHARS),
        )
    };

    PreparedToolResult {
        content_for_model,
        content_for_terminal,
    }
}

/// 当前轮刚产出的 tool result 需要先以 raw content 进入 messages，
/// 让“最近 N 条工具结果保留原文”的保护从入口就成立，而不是先在这里被
/// stub/summary 弱化，再指望后面的 `KEEP_RECENT_TOOL_MESSAGES` 兜底。
///
/// 终端侧仍沿用原有 preview / overflow 文件逻辑，避免把超大结果整块刷到屏幕。
pub(in crate::ai::driver::turn_runtime) fn prepare_recent_tool_result(
    app: &App,
    tool_name: &str,
    content: &str,
) -> PreparedToolResult {
    let content_for_terminal = prepare_tool_result(app, tool_name, content).content_for_terminal;
    PreparedToolResult {
        content_for_model: content.to_string(),
        content_for_terminal,
    }
}

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
fn execute_tool_calls_for_round(
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

#[derive(Clone, Copy)]
enum ToolCallRejectionReason {
    NoToolHandoff,
    PatchRetryNeedsFreshRead,
}

fn reject_tool_calls(
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
        had_error: true,
    }
}

fn rejected_tool_call_message(tool_name: &str, reason: ToolCallRejectionReason) -> String {
    match reason {
        ToolCallRejectionReason::NoToolHandoff => format!(
            "Error: tool calls are disabled in no-tool handoff mode for this turn. \
Do not call '{tool_name}' again; instead summarize confirmed facts, answer what you can, and explain the remaining work / blockers / next steps."
        ),
        ToolCallRejectionReason::PatchRetryNeedsFreshRead => format!(
            "Error: apply_patch retry blocked. The previous patch for this file failed with `context mismatch` or `ambiguous patch`, which means the file content you are working from is stale or the context is not unique. \
Do NOT retry patches in this batch — doing so will only fail again. Required recovery steps: (1) call `read_file` on the SAME target path to get the current truth state; (2) copy context lines DIRECTLY from that fresh output, including function names or distinctive surrounding lines to ensure each hunk matches exactly ONE location; (3) do NOT copy the leading line-number + tab prefix that read_file prints (each line is rendered as a right-aligned line number followed by a TAB, e.g. `    42\\t<code>`) — copy only the code after the tab; (4) call `apply_patch` only in a LATER tool round after you have successfully read the file."
        ),
    }
}

fn duplicate_read_only_call_ids(messages: &[Message], tool_calls: &[ToolCall]) -> HashSet<String> {
    let mut call_signatures = HashMap::new();
    let mut completed = HashSet::new();
    for message in messages {
        if message.role == "user" {
            call_signatures.clear();
            completed.clear();
            continue;
        }
        if let Some(previous_calls) = &message.tool_calls {
            for tool_call in previous_calls {
                if let Some(signature) = read_only_tool_signature(tool_call) {
                    call_signatures.insert(tool_call.id.as_str(), signature);
                }
            }
        }
        if message.role == "tool"
            && let Some(call_id) = message.tool_call_id.as_deref()
            && let Some(signature) = call_signatures.get(call_id)
            && tool_result_completed_successfully(&message.content)
        {
            completed.insert(signature.clone());
        }
    }

    tool_calls
        .iter()
        .filter_map(|tool_call| {
            let signature = read_only_tool_signature(tool_call)?;
            completed.contains(&signature).then(|| tool_call.id.clone())
        })
        .collect()
}

fn tool_result_completed_successfully(content: &serde_json::Value) -> bool {
    let text = content.as_str().unwrap_or_default().trim_start();
    !text.starts_with("Error:") && !text.starts_with("Exit code:")
}

fn read_only_tool_signature(tool_call: &ToolCall) -> Option<String> {
    if !repeat_guarded_read_only_tool_name(&tool_call.function.name) {
        return None;
    }

    let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
        .unwrap_or_else(|_| serde_json::Value::String(tool_call.function.arguments.clone()));
    let args_json = serde_json::to_string(&args).unwrap_or_else(|_| args.to_string());
    Some(format!("{}\n{}", tool_call.function.name, args_json))
}

fn repeat_guarded_read_only_tool_name(name: &str) -> bool {
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

/// `knowledge_search` 在一个 user turn 内是可复用的只读事实。通用重复保护只会
/// 比较整批调用；这里按单条语义签名抑制重搜，因此同批的其它有效工具不会被连带
/// 拒绝。任何知识写入都会使旧搜索失效，随后允许再次搜索。
fn duplicate_knowledge_search_call_ids(
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
        if message.role == "user" {
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

fn knowledge_search_signature(tool_call: &ToolCall) -> Option<String> {
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

fn knowledge_store_mutated(tool_call: &ToolCall) -> bool {
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

fn duplicate_knowledge_search_message() -> String {
    "Error: this knowledge_search was already completed with the same query in the current user turn. Reuse its result; search again only after knowledge changes or with a materially different query.".to_string()
}

fn duplicate_read_only_message(tool_name: &str) -> String {
    if tool_name == "knowledge_search" {
        return duplicate_knowledge_search_message();
    }
    format!(
        "Error: this read-only call to '{tool_name}' already completed successfully in the current user turn. \
Reuse its earlier result; only retry after the underlying data changes or with arguments that request different information."
    )
}

fn extract_apply_patch_target_paths_from_patch(patch: &str) -> Vec<PathBuf> {
    patch
        .lines()
        .filter_map(|line| {
            let line = line.trim_start();
            [
                "*** Update File: ",
                "*** Add File: ",
                "*** Replace in line: ",
            ]
            .iter()
            .find_map(|prefix| line.strip_prefix(prefix))
            .map(|path| {
                FileStore::new(PathBuf::from(path.trim()))
                    .path()
                    .to_path_buf()
            })
        })
        .collect()
}

/// `apply_patch` 的 context mismatch / ambiguity 说明模型当前持有的文件事实已过期，
/// 继续微调旧 patch 只会重复失败。这里以实际工具消息为准：目标文件在失败后必须有
/// 一次成功的 `read_file`，才允许再次 patch。路径统一经 FileStore 归一化，避免
/// 相对路径、`~` 和绝对路径写法不同而绕过门控。
fn patch_retry_requires_fresh_read(messages: &[Message], tool_calls: &[ToolCall]) -> bool {
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

    let mut stale_patch_targets = HashSet::new();
    for message in messages {
        let Some(previous_calls) = message.tool_calls.as_ref() else {
            continue;
        };
        for tool_call in previous_calls {
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
                    } else if patch_failure_requires_fresh_read(result) {
                        stale_patch_targets.extend(paths);
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

    tool_calls.iter().any(|tool_call| {
        tool_call.function.name == "apply_patch"
            && patch_target_paths(tool_call)
                .into_iter()
                .any(|path| stale_patch_targets.contains(&path))
    })
}

fn patch_failure_requires_fresh_read(result: &str) -> bool {
    let result = result.to_ascii_lowercase();
    result.contains("context mismatch") || result.contains("ambiguous patch")
}

fn patch_target_paths(tool_call: &ToolCall) -> Vec<PathBuf> {
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

fn file_tool_target_path(tool_call: &ToolCall) -> Option<PathBuf> {
    let args = serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments).ok()?;
    let target = args
        .get("file_path")
        .or_else(|| args.get("path"))
        .and_then(serde_json::Value::as_str)?;
    Some(FileStore::new(PathBuf::from(target)).path().to_path_buf())
}

/// 前台同步工具执行（尤其是 `execute_command` 的流式输出）也属于“当前 turn 的可中断
/// 输出阶段”。若这里不抬起 `app.streaming`，Ctrl+C 会被 SIGINT 处理器误判成
/// `Shutdown`，直接退出主进程，而不是取消当前工具轮次。
struct ToolExecutionStreamingGuard {
    flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl ToolExecutionStreamingGuard {
    fn new(flag: &std::sync::Arc<std::sync::atomic::AtomicBool>) -> Self {
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
        Self {
            flag: std::sync::Arc::clone(flag),
        }
    }
}

impl Drop for ToolExecutionStreamingGuard {
    fn drop(&mut self) {
        self.flag.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

struct TerminalToolObserver<'a> {
    app: &'a App,
    active_stream_tool_call_id: Option<String>,
    pending_utf8: Vec<u8>,
    visual_output_probe: String,
    visual_output_line: String,
    visual_output_detected: bool,
    at_line_start: bool,
    streamed_any_output: bool,
    // 流式输出折叠状态
    allow_inline_fold_updates: bool,
    fold_total_lines: usize,
    tty_fold: TtyToolOutputFoldState,
}

// 典型终端二维码约 30–50 行；保留 64 行能完整展示扫码登录等一次性视觉输出，
// 同时仍为构建日志等无界流式输出提供确定上限。
const TOOL_OUTPUT_FOLD_MAX_VISIBLE: usize = 64;
// 常规命令日志不应出现在终端；只有连续的 block-glyph 网格才按视觉输出展示。
// 这个上限既覆盖常见终端二维码，又避免长时间普通日志无限占用探测缓冲区。
const VISUAL_OUTPUT_PROBE_MAX_BYTES: usize = 16 * 1024;
const VISUAL_OUTPUT_MIN_CONSECUTIVE_GRID_ROWS: usize = 3;
const VISUAL_OUTPUT_MIN_BLOCK_GLYPHS_PER_ROW: usize = 8;

/// 判断一行是否像由 Unicode block glyph 绘制的终端视觉输出（例如二维码）。
/// 不根据命令名做白名单，避免把某个 CLI 的行为硬编码进通用执行器。
fn is_terminal_visual_grid_line(line: &str) -> bool {
    line.chars()
        .filter(|ch| {
            matches!(
                ch,
                '█' | '▀' | '▄' | '▌' | '▐' | '▖' | '▗' | '▘' | '▝' | '▚' | '▞' | '■'
            )
        })
        .count()
        >= VISUAL_OUTPUT_MIN_BLOCK_GLYPHS_PER_ROW
}

/// 至少连续三行 block-glyph 网格才视作视觉输出，防止进度条或普通文本误触发。
fn contains_terminal_visual_grid(text: &str) -> bool {
    let mut consecutive_rows = 0;
    for line in text.lines() {
        if is_terminal_visual_grid_line(line) {
            consecutive_rows += 1;
            if consecutive_rows >= VISUAL_OUTPUT_MIN_CONSECUTIVE_GRID_ROWS {
                return true;
            }
        } else {
            consecutive_rows = 0;
        }
    }
    false
}

fn trim_visual_output_probe(probe: &mut String) {
    if probe.len() <= VISUAL_OUTPUT_PROBE_MAX_BYTES {
        return;
    }

    let excess = probe.len() - VISUAL_OUTPUT_PROBE_MAX_BYTES;
    let trim_at = probe
        .char_indices()
        .find_map(|(offset, _)| (offset >= excess).then_some(offset))
        .unwrap_or(probe.len());
    probe.drain(..trim_at);
}

#[derive(Debug, Default)]
struct TtyToolOutputFoldState {
    recent_lines: VecDeque<String>,
    current_line: String,
    total_lines: usize,
    window_rows: usize,
}

impl TtyToolOutputFoldState {
    fn reset(&mut self) {
        self.recent_lines.clear();
        self.current_line.clear();
        self.total_lines = 0;
        self.window_rows = 0;
    }

    fn push_text(&mut self, text: &str) -> std::io::Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        for ch in text.chars() {
            if ch == '\n' {
                self.total_lines += 1;
                self.recent_lines
                    .push_back(std::mem::take(&mut self.current_line));
                while self.recent_lines.len() > TOOL_OUTPUT_FOLD_MAX_VISIBLE {
                    self.recent_lines.pop_front();
                }
            } else {
                self.current_line.push(ch);
            }
        }
        self.redraw()
    }

    fn finish(&mut self) -> std::io::Result<()> {
        self.redraw()
    }

    fn redraw(&mut self) -> std::io::Result<()> {
        let mut out = std::io::stdout();
        if self.window_rows > 0 {
            write!(out, "\x1b[{}A\r\x1b[0J", self.window_rows)?;
        }

        let (window, window_rows) = render_tty_tool_output_fold_window(self);
        if !window.is_empty() {
            out.write_all(window.as_bytes())?;
            out.flush()?;
        }
        self.window_rows = window_rows;
        Ok(())
    }
}

fn tty_tool_output_hidden_count(fold: &TtyToolOutputFoldState) -> usize {
    let current_line = usize::from(!fold.current_line.is_empty());
    fold.total_lines
        .saturating_add(current_line)
        .saturating_sub(TOOL_OUTPUT_FOLD_MAX_VISIBLE)
}

fn tty_tool_output_visible_lines(fold: &TtyToolOutputFoldState) -> Vec<&str> {
    let current_line = usize::from(!fold.current_line.is_empty());
    let visible_completed = TOOL_OUTPUT_FOLD_MAX_VISIBLE.saturating_sub(current_line);
    let completed_skip = fold.recent_lines.len().saturating_sub(visible_completed);
    let mut visible = fold
        .recent_lines
        .iter()
        .skip(completed_skip)
        .map(String::as_str)
        .collect::<Vec<_>>();
    if current_line > 0 {
        visible.push(fold.current_line.as_str());
    }
    visible
}

fn render_tty_tool_output_fold_window(fold: &TtyToolOutputFoldState) -> (String, usize) {
    let hidden_count = tty_tool_output_hidden_count(fold);
    let visible_lines = tty_tool_output_visible_lines(fold);
    if hidden_count == 0 && visible_lines.is_empty() {
        return (String::new(), 0);
    }

    let mut out = String::new();
    // 每条行都被 clamp 成「最多占一个物理行」，窗口物理行数恒等于逻辑行数，
    // cursor-up 擦除精确，不再因超长/宽字符输出行的自动折行让擦除行数算少而残留。
    let mut rows = 0usize;

    if hidden_count > 0 {
        let marker = format!(
            "  {ACCENT_RULE}│{RESET} {ACCENT_MUTED}{}{RESET}",
            clamp_tool_output_body(&format!("··· {hidden_count} lines folded ···"))
        );
        rows += 1;
        out.push_str(&marker);
        out.push('\n');
    }

    for line in visible_lines {
        let rendered = format_tool_output_line(&clamp_tool_output_body(line));
        rows += 1;
        out.push_str(&rendered);
        out.push('\n');
    }

    (out, rows)
}

/// 工具输出折叠行统一带 `  │ ` 前缀（4 列），正文按终端列宽减 4 clamp 成单物理行。
fn clamp_tool_output_body(body: &str) -> String {
    const PREFIX_COLS: usize = 4;
    clamp_line_to_terminal_row_with_reserve(body, PREFIX_COLS)
}

impl<'a> TerminalToolObserver<'a> {
    fn new(app: &'a App) -> Self {
        Self {
            app,
            active_stream_tool_call_id: None,
            pending_utf8: Vec::new(),
            visual_output_probe: String::new(),
            visual_output_line: String::new(),
            visual_output_detected: false,
            at_line_start: true,
            streamed_any_output: false,
            fold_total_lines: 0,
            // `\r` / `CSI 2K` 这类原地刷新只适合真实 TTY。IDE Chat / pipe /
            // 日志采集场景不会解释 ANSI 光标控制，原样输出后就会泄漏成 `[2K`。
            allow_inline_fold_updates: std::io::IsTerminal::is_terminal(&std::io::stdout()),
            tty_fold: TtyToolOutputFoldState::default(),
        }
    }

    fn reset_stream_state(&mut self) {
        self.active_stream_tool_call_id = None;
        self.pending_utf8.clear();
        self.visual_output_probe.clear();
        self.visual_output_line.clear();
        self.visual_output_detected = false;
        self.at_line_start = true;
        self.streamed_any_output = false;
        self.fold_total_lines = 0;
        self.tty_fold.reset();
    }

    fn start_stream_output(&mut self, tool_call: &ToolCall) {
        if self.active_stream_tool_call_id.as_deref() == Some(tool_call.id.as_str()) {
            return;
        }
        self.reset_stream_state();
        self.active_stream_tool_call_id = Some(tool_call.id.clone());
        let label = if tool_call.function.name == "execute_command" {
            "streaming command output"
        } else {
            "streaming tool output"
        };
        print_tool_note_line("output", label);
    }

    fn push_stream_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.streamed_any_output = true;
        // 工具输出被禁用时仍记录已收到流，避免完成时误报“无输出”，但不可绕过
        // runtime_ctx 的终端输出开关直接写 stdout。
        if !crate::ai::driver::runtime_ctx::terminal_output_enabled() {
            return;
        }

        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        let sanitized = sanitize_for_terminal(&normalized);
        if sanitized.is_empty() {
            return;
        }

        if !self.visual_output_detected {
            self.visual_output_probe.push_str(&sanitized);
            if !contains_terminal_visual_grid(&self.visual_output_probe) {
                trim_visual_output_probe(&mut self.visual_output_probe);
                return;
            }

            self.visual_output_detected = true;
            let visual_output = std::mem::take(&mut self.visual_output_probe);
            self.push_visual_output_text(&visual_output);
            return;
        }

        self.push_visual_output_text(&sanitized);
    }

    /// 已确认存在视觉网格后，仍只展示构成网格的行；后续普通日志保持隐藏。
    fn push_visual_output_text(&mut self, text: &str) {
        self.visual_output_line.push_str(text);
        while let Some(newline_at) = self.visual_output_line.find('\n') {
            let line = self.visual_output_line[..=newline_at].to_string();
            self.visual_output_line.drain(..=newline_at);
            if is_terminal_visual_grid_line(&line) {
                self.render_visual_output_text(&line);
            }
        }

        // 非换行的普通日志不能无限堆积；二维码行会在换行到达后再做判定。
        if self.visual_output_line.len() > VISUAL_OUTPUT_PROBE_MAX_BYTES {
            self.visual_output_line.clear();
        }
    }

    fn flush_visual_output_line(&mut self) {
        if self.visual_output_line.is_empty() {
            return;
        }

        let line = std::mem::take(&mut self.visual_output_line);
        if is_terminal_visual_grid_line(&line) {
            // 补齐换行，避免紧随其后的完成状态与最后一行视觉输出粘连。
            self.render_visual_output_text(&format!("{line}\n"));
        }
    }

    fn render_visual_output_text(&mut self, text: &str) {
        if self.allow_inline_fold_updates {
            let _ = self.tty_fold.push_text(text);
            let _ = std::io::stdout().flush();
            return;
        }

        for ch in text.chars() {
            if ch == '\n' {
                self.fold_total_lines += 1;
                if self.fold_total_lines <= TOOL_OUTPUT_FOLD_MAX_VISIBLE {
                    print!("{RESET}\n");
                    self.at_line_start = true;
                } else if self.fold_total_lines == TOOL_OUTPUT_FOLD_MAX_VISIBLE + 1 {
                    print!("{RESET}\n");
                    self.at_line_start = true;
                    println!(
                        "  {ACCENT_RULE}│{RESET} {ACCENT_MUTED}··· streaming output folded until completion ···{RESET}"
                    );
                }
            } else if self.fold_total_lines < TOOL_OUTPUT_FOLD_MAX_VISIBLE {
                if self.at_line_start {
                    print!("{}", format_tool_output_prefix());
                    self.at_line_start = false;
                }
                print!("{ch}");
            }
        }
        let _ = std::io::stdout().flush();
    }

    fn push_stream_text_for_tool(&mut self, tool_call: &ToolCall, text: &str) {
        if text.is_empty() {
            return;
        }
        self.start_stream_output(tool_call);
        self.push_stream_text(text);
    }

    fn flush_pending_utf8(&mut self) {
        if self.pending_utf8.is_empty() {
            return;
        }
        let text = String::from_utf8_lossy(&self.pending_utf8).into_owned();
        self.pending_utf8.clear();
        self.push_stream_text(&text);
    }

    fn finish_stream_output(&mut self, newline: bool) {
        self.flush_pending_utf8();
        self.flush_visual_output_line();
        if !crate::ai::driver::runtime_ctx::terminal_output_enabled() {
            return;
        }
        if !self.visual_output_detected {
            return;
        }
        if self.allow_inline_fold_updates {
            let _ = self.tty_fold.finish();
            return;
        }
        if self.fold_total_lines > TOOL_OUTPUT_FOLD_MAX_VISIBLE {
            let folded = self.fold_total_lines - TOOL_OUTPUT_FOLD_MAX_VISIBLE;
            println!(
                "  {ACCENT_RULE}│{RESET} {ACCENT_MUTED}··· {folded} lines folded ···{RESET}"
            );
            self.at_line_start = true;
        } else if !self.at_line_start {
            if newline {
                print!("{RESET}\n");
                self.at_line_start = true;
            } else {
                print!("{RESET}");
            }
            let _ = std::io::stdout().flush();
        }
    }

    fn print_prepared_tool_result(&mut self, prepared: &PreparedToolResult) {
        // 终端不再打印工具输出内容，只保留状态行。
        let _ = prepared;
    }

    fn print_captured_command_output(&mut self, prepared: &PreparedToolResult) {
        // 终端不再打印工具输出内容，只保留状态行。
        let _ = prepared;
    }
}

/// 把 `execute_command` 等命令类工具的 arguments 渲染成单行可读的命令文本，
/// 用于工具开始时在终端打印「输入」。多行命令折叠为单行，过长则截断。
/// 解析失败（缺 `command` 字段或非法 JSON）时返回 None。
fn format_command_input(arguments: &str) -> Option<String> {
    let args: serde_json::Value = serde_json::from_str(arguments).ok()?;
    let command = args.get("command")?.as_str()?;
    // 折叠换行，避免一条命令在终端占多行打乱状态行布局
    let mut line = command.replace('\n', " ⏎ ").replace('\r', "");
    const MAX_CHARS: usize = 200;
    if line.chars().count() > MAX_CHARS {
        let kept: String = line.chars().take(MAX_CHARS.saturating_sub(1)).collect();
        line = format!("{kept}…");
    }
    if let Some(cwd) = args.get("cwd").and_then(serde_json::Value::as_str) {
        if !cwd.is_empty() {
            line.push_str(&format!("  (cwd: {cwd})"));
        }
    }
    if args.get("pty").and_then(serde_json::Value::as_bool) == Some(true) {
        line.push_str("  (PTY)");
    }
    Some(line)
}

impl tools::ToolExecutionObserver for TerminalToolObserver<'_> {
    fn on_tool_started(&mut self, tool_call: &ToolCall) {
        if matches!(
            tool_call.function.name.as_str(),
            "execute_command" | "run_command" | "shell" | "bash"
        ) {
            if let Some(line) = format_command_input(&tool_call.function.arguments) {
                print_tool_command_line(&line);
            }
        }
    }

    fn on_tool_stream(&mut self, tool_call: &ToolCall, chunk: &[u8]) {
        self.pending_utf8.extend_from_slice(chunk);
        loop {
            match std::str::from_utf8(&self.pending_utf8) {
                Ok(text) => {
                    let text = text.to_string();
                    self.pending_utf8.clear();
                    self.push_stream_text_for_tool(tool_call, &text);
                    break;
                }
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    if valid_up_to == 0 {
                        if err.error_len().is_some() {
                            self.flush_pending_utf8();
                        }
                        break;
                    }

                    let text =
                        String::from_utf8_lossy(&self.pending_utf8[..valid_up_to]).into_owned();
                    self.pending_utf8.drain(..valid_up_to);
                    self.push_stream_text_for_tool(tool_call, &text);

                    if err.error_len().is_some() {
                        self.flush_pending_utf8();
                    }
                }
            }
        }
    }

    fn on_tool_finished(&mut self, tool_call: &ToolCall, run_result: &tools::RunOneResult) {
        let streamed_output = self.active_stream_tool_call_id.as_deref()
            == Some(tool_call.id.as_str())
            && self.streamed_any_output;
        if streamed_output {
            let is_failure = if tool_call.function.name == "execute_command" {
                run_result.tool_result.content.starts_with("Exit code:")
            } else {
                !run_result.ok
            };
            self.finish_stream_output(is_failure);

            if is_failure {
                if let Some(exit_line) = run_result.tool_result.content.lines().next() {
                    print_tool_note_line("error", exit_line);
                }
            } else if tool_call.function.name == "execute_command" {
                print_tool_note_line("result", "command completed");
            } else {
                print_tool_note_line("result", "tool completed");
            }

            self.reset_stream_state();
            return;
        }

        let prepared = prepare_recent_tool_result(
            self.app,
            &tool_call.function.name,
            &run_result.tool_result.content,
        );
        self.print_prepared_tool_result(&prepared);
    }
}

fn handle_tool_call_round(
    app: &mut App,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    tool_call_execution: &ToolCallExecution,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    iteration: usize,
    rejection_reason: Option<ToolCallRejectionReason>,
    suppressed_read_only_call_ids: &HashSet<String>,
    turn_had_tool_error: &mut bool,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let remaining_meta = parse_prune_meta_and_update_marks(
        app,
        messages,
        &tool_call_execution.stream_result.hidden_meta,
    );
    let exec_result = if let Some(reason) = rejection_reason {
        reject_tool_calls(&tool_call_execution.stream_result.tool_calls, reason)
    } else {
        let mut observer = TerminalToolObserver::new(app);
        let _streaming_guard = ToolExecutionStreamingGuard::new(&app.streaming);
        execute_tool_calls_with_suppressed_read_only_calls(
            &app.session_id,
            mcp_client,
            shared_mcp_client,
            &tool_call_execution.stream_result.tool_calls,
            &tool_call_execution.allowed_tool_names,
            Some(&mut observer),
            iteration,
            suppressed_read_only_call_ids,
        )?
    };
    *turn_had_tool_error |= exec_result.had_error;
    append_cached_tool_results_note(&exec_result, messages, turn_messages);
    append_tool_result_messages(
        app,
        &tool_call_execution.stream_result.assistant_text,
        &tool_call_execution.stream_result.reasoning_text,
        &tool_call_execution.stream_result.reasoning_items,
        &exec_result,
        messages,
        turn_messages,
    );
    record_hidden_self_note(app, turn_messages, &remaining_meta);
    record_tool_inspection_artifacts(messages, turn_messages);

    persist_pending_turn_messages(app, one_shot_mode, turn_messages, persisted_turn_messages);

    Ok(None)
}

fn execute_tool_calls_with_suppressed_read_only_calls(
    session_id: &str,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    tool_calls: &[ToolCall],
    allowed_tool_names: &rust_tools::commonw::FastSet<String>,
    observer: Option<&mut dyn tools::ToolExecutionObserver>,
    iteration: usize,
    suppressed_call_ids: &HashSet<String>,
) -> Result<ExecuteToolCallsResult, Box<dyn std::error::Error>> {
    if suppressed_call_ids.is_empty() {
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
        .filter(|tool_call| !suppressed_call_ids.contains(&tool_call.id))
        .cloned()
        .collect::<Vec<_>>();
    let executed = if executable.is_empty() {
        ExecuteToolCallsResult {
            executed_tool_calls: Vec::new(),
            tool_results: Vec::new(),
            cached_hits: Vec::new(),
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

    let mut result_by_id = HashMap::new();
    for (index, result) in executed.tool_results.into_iter().enumerate() {
        let cached = executed.cached_hits.get(index).copied().unwrap_or(false);
        result_by_id.insert(result.tool_call_id.clone(), (result, cached));
    }
    let mut tool_results = Vec::with_capacity(tool_calls.len());
    let mut cached_hits = Vec::with_capacity(tool_calls.len());
    for tool_call in tool_calls {
        if suppressed_call_ids.contains(&tool_call.id) {
            tool_results.push(crate::ai::types::ToolResult {
                tool_call_id: tool_call.id.clone(),
                content: duplicate_read_only_message(&tool_call.function.name),
            });
            cached_hits.push(false);
            continue;
        }
        let Some((result, cached)) = result_by_id.remove(&tool_call.id) else {
            tool_results.push(crate::ai::types::ToolResult {
                tool_call_id: tool_call.id.clone(),
                content: "Error: tool execution returned no result for this call.".to_string(),
            });
            cached_hits.push(false);
            continue;
        };
        tool_results.push(result);
        cached_hits.push(cached);
    }

    Ok(ExecuteToolCallsResult {
        executed_tool_calls: tool_calls.to_vec(),
        tool_results,
        cached_hits,
        had_error: executed.had_error || !suppressed_call_ids.is_empty(),
    })
}

const PENDING_SUBAGENT_TASKS_FOLLOWUP_PREFIX: &str = "tool_followup:pending_subagent_tasks\n";

fn clear_pending_subagent_tasks_followup(messages: &mut Vec<Message>) {
    messages.retain(|message| {
        !(message.role == ROLE_INTERNAL_NOTE
            && matches!(
                &message.content,
                serde_json::Value::String(text)
                    if text.starts_with(PENDING_SUBAGENT_TASKS_FOLLOWUP_PREFIX)
            ))
    });
}

fn clear_no_tool_handoff_note(messages: &mut Vec<Message>) {
    let note = no_tool_handoff_note();
    messages.retain(|message| {
        !(message.role == ROLE_INTERNAL_NOTE
            && matches!(&message.content, serde_json::Value::String(text) if text == note))
    });
}

fn reopen_turn_for_outstanding_subagent_tasks(
    messages: &mut Vec<Message>,
    session_id: &str,
) -> bool {
    let outstanding_anchor = match task_tools::build_outstanding_task_anchor(session_id) {
        Ok(Some(note)) => note,
        Ok(None) => return false,
        Err(err) => {
            let _ = writeln!(
                std::io::stderr(),
                "  [task-anchor] failed to inspect outstanding subagent tasks: {err}"
            );
            return false;
        }
    };

    clear_pending_subagent_tasks_followup(messages);
    clear_no_tool_handoff_note(messages);

    let mut note = String::from(PENDING_SUBAGENT_TASKS_FOLLOWUP_PREFIX);
    note.push_str(
        "The previous assistant response tried to finish the turn while spawned subagent tasks were still outstanding.\n",
    );
    note.push_str("This is not a final answer. Continue the current turn now.\n");
    note.push_str(
        "Temporarily lift no-tool handoff if it was active, but only so you can collect or inspect the outstanding subagent results.\n",
    );
    note.push_str(
        "Immediate next step: call `task_wait` or `task_status` for the outstanding task_ids below. Do not answer the user until every listed task has been handled.\n\n",
    );
    note.push_str(&outstanding_anchor);
    messages.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: serde_json::Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    true
}

const TRUNCATION_RETRY_NOTE_PREFIX: &str = "tool_followup:output_truncated\n";
const DEGENERATE_REPETITION_RETRY_NOTE_PREFIX: &str = "tool_followup:degenerate_repetition\n";
const DEGENERATE_REPETITION_FINISH_REASON: &str = "degenerate_repetition";

/// 在检测到本轮响应被截断后，把已产出的可见文本（若有）作为部分进展保留，并追加
/// 一条收缩重写提示，指导模型下一轮缩小单次输出规模后重发被截断的操作。
///
/// 幂等：同一条提示不会重复注入，避免连续截断时堆叠多份相同 note。
fn append_truncation_retry_note(
    stream_result: &crate::ai::types::StreamResult,
    messages: &mut Vec<Message>,
    consecutive_truncations: usize,
) {
    use serde_json::Value;

    let degenerate_repetition = stream_result
        .finish_reason_value
        .as_deref()
        .is_some_and(|reason| reason == DEGENERATE_REPETITION_FINISH_REASON);

    // 保留模型已输出的可见文本作为"部分进展"，让重试时不至于完全丢失上下文。
    // 截断场景下这段文本往往是半截的意图说明，仅作参考，不当作最终回答。
    //
    // 仅写入内存 messages（本 turn 内可见），不写入 turn_messages 持久化轨道。
    // 原因：partial text 是半截文本，不是有效的对话记录。连续截断时会累积
    // 多条大体积 partial text，持久化后污染历史文件，下个 turn 加载时占据
    // 大量字符预算，导致 compress_messages_for_context 压缩/丢弃正常对话历史，
    // 表现为"历史清空"。与 truncation note 保持一致：过程性内容不持久化。
    let partial = stream_result.assistant_text.trim();
    if !partial.is_empty() {
        messages.push(Message {
            role: "assistant".to_string(),
            content: Value::String(partial.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }

    // 移除上一轮的截断/重复退化提示（如有），替换为携带最新计数的新提示。
    // 早期版本是幂等的——只注入一次后跳过。但连续截断时模型得不到
    // "已再次截断"的反馈，只看到和上次一样的上下文，大概率产出相似
    // 长度的内容再被截断，陷入盲循环。改为每次更新计数，让模型感知
    // 到严重程度在递增。
    messages.retain(|message| {
        !(message.role == ROLE_INTERNAL_NOTE
            && message.content.as_str().is_some_and(|content| {
                content.starts_with(TRUNCATION_RETRY_NOTE_PREFIX)
                    || content.starts_with(DEGENERATE_REPETITION_RETRY_NOTE_PREFIX)
            }))
    });

    if degenerate_repetition {
        let note = format!(
            "{}上一轮推理流出现了连续重复片段，运行时已提前终止该次生成，避免继续消耗 token。\n\
             不要续写或复述那段推理。请从最近的工具结果重新判断当前状态：\n\
             - 不要重试已被策略拒绝的同一命令；改用当前可用的专用工具；\n\
             - 只执行完成任务所需的下一步，避免重复搜索或重复解释；\n\
             - 若已有足够证据，直接给出结论。",
            DEGENERATE_REPETITION_RETRY_NOTE_PREFIX
        );
        messages.push(Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(note),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
        return;
    }

    let mut note = String::from(TRUNCATION_RETRY_NOTE_PREFIX);
    if consecutive_truncations > 1 {
        note.push_str(&format!(
            "（已连续第 {} 次被截断，上次的收缩幅度不够，请进一步大幅缩小单次输出规模）\n",
            consecutive_truncations
        ));
    }
    note.push_str("上一轮响应在生成中途被截断（疑似撞到输出长度上限），未能完成。\n");
    note.push_str("这不是最终回答。请继续当前任务，并显著缩小单次输出规模：\n");
    note.push_str(
        "- 若在写文件：把大文件拆成多次调用（先创建骨架，再分块 append/edit），单次 write 控制在几百行以内；\n",
    );
    note.push_str("- 优先用小步、多次的工具调用，而不是一次性产出超大内容；\n");
    note.push_str("- 只重发被截断的那个操作，不要重复已经成功完成的步骤。");
    // 过程性纠偏提示：仅在本 turn 内下发给 LLM，不写入 turn_messages 持久化轨道。
    // 该提示只在"刚发生截断的下一轮"有意义；若持久化会在后续每个 turn 反复重放，
    // 让模型永久性地畏手畏脚、输出规模受限——正是"一次变蠢后持续变蠢"的根因之一。
    messages.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

fn extract_image_paths_from_file_read_tool_calls(tool_calls: &[ToolCall]) -> Vec<String> {
    let mut out = Vec::new();
    for tool_call in tool_calls {
        if !matches!(
            tool_call.function.name.as_str(),
            "read_file"
        ) {
            continue;
        }
        let Ok(args) = serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
        else {
            continue;
        };
        let Some(path) = args
            .get("file_path")
            .or_else(|| args.get("path"))
            .and_then(serde_json::Value::as_str)
        else {
            continue;
        };
        if crate::ai::files::is_image_path(path) && !out.iter().any(|existing| existing == path) {
            out.push(path.to_string());
        }
    }
    out
}

fn append_auto_image_followup_message(
    app: &App,
    question: &str,
    shared_mcp_client: &SharedMcpClient,
    image_paths: &[String],
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
) -> Result<(), Box<dyn std::error::Error>> {
    if image_paths.is_empty() {
        return Ok(());
    }

    let question = if question.trim().is_empty() {
        "Analyze the requested image file.".to_string()
    } else {
        question.to_string()
    };

    let content = if crate::ai::models::supports_image_input(&app.current_model) {
        crate::ai::request::build_content(&app.current_model, &question, image_paths)?
    } else if let Some(ocr) =
        crate::ai::driver::model::ocr_images_for_attached_input(shared_mcp_client, image_paths)?
    {
        let prompt = if ocr.has_usable_text() {
            format!(
                "{}\n\n[Auto OCR From Image File Read via {}]\n{}",
                question, ocr.tool_name, ocr.content
            )
        } else {
            format!(
                "{}\n\n[Image file read was auto-upgraded to attachment semantics, but OCR did not produce usable text.]",
                question
            )
        };
        serde_json::Value::String(prompt)
    } else {
        serde_json::Value::String(format!(
            "{}\n\n[Image file read was auto-upgraded to attachment semantics, but no OCR tool was available for this text-only model.]",
            question
        ))
    };

    append_message_pair(
        messages,
        turn_messages,
        Message {
            role: "user".to_string(),
            content,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    );
    Ok(())
}

pub(in crate::ai::driver::turn_runtime) fn handle_iteration_execution(
    app: &mut App,
    question: &str,
    mcp_client: &McpClient,
    shared_mcp_client: &SharedMcpClient,
    execution: IterationExecution,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    final_assistant_text: &mut String,
    final_assistant_recorded: &mut bool,
    force_final_response: &mut bool,
    terminal_dedupe_candidate: &mut Option<String>,
    _no_active_skill: bool,
    iteration: usize,
    max_iterations: usize,
    consecutive_truncations: usize,
    turn_had_tool_error: &mut bool,
) -> Result<TurnLoopStep, Box<dyn std::error::Error>> {
    match execution {
        IterationExecution::Exit(outcome) => Ok(TurnLoopStep::Return(outcome)),
        IterationExecution::RequestFailed(text) => {
            *final_assistant_text = text;
            Ok(TurnLoopStep::Break)
        }
        IterationExecution::EmptyResponse => {
            // 模型返回空响应（无文本、无工具调用、无思考内容），自动重试
            let _ = writeln!(std::io::stderr(), "  ⚠ 模型返回空响应，自动重试…");
            Ok(TurnLoopStep::Continue)
        }
        IterationExecution::Truncated(stream_result) => {
            if stream_result.stream_error {
                // 流读取错误（服务端不稳定）导致的截断：不注入收缩提示，
                // 不保留 partial text（流中断时的 partial 不可靠），
                // 简单重试即可。日志已在 orchestrator 层打印。
                Ok(TurnLoopStep::Continue)
            } else {
                let degenerate_repetition = stream_result
                    .finish_reason_value
                    .as_deref()
                    .is_some_and(|reason| reason == DEGENERATE_REPETITION_FINISH_REASON);
                if degenerate_repetition {
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ⚠ 检测到模型推理内容持续重复，已中止本次生成并纠偏重试…"
                    );
                } else {
                    // 真截断：模型撞输出上限或工具 JSON 半截。保留已产出的
                    // 可见文本作为上下文，并注入一条收缩重写提示后自动重试。
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ⚠ 模型响应被截断（疑似输出上限），提示收缩后自动重试…"
                    );
                }
                // 打印截断诊断信息，便于排查截断原因。
                let partial = stream_result.assistant_text.trim();
                let reasoning = stream_result.reasoning_text.trim();
                // 截断原因诊断
                if stream_result.truncated_by_length && partial.is_empty() {
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ├─ 截断原因: finish_reason=length，无可见文本输出"
                    );
                    if !reasoning.is_empty() {
                        let _ = writeln!(
                            std::io::stderr(),
                            "  ├─ reasoning 已产出 {} 字符（可能 reasoning 耗尽了 token 预算）",
                            stream_result.reasoning_text.len()
                        );
                        let r_snippet = if reasoning.len() > 600 {
                            let rchar_count = reasoning.chars().count();
                            let head: String = reasoning.chars().take(300).collect();
                            let tail: String = reasoning.chars().skip(rchar_count - 300).collect();
                            format!("{}…[共 {} 字符]…{}", head, rchar_count, tail)
                        } else {
                            reasoning.to_string()
                        };
                        let _ = writeln!(
                            std::io::stderr(),
                            "  ├─ reasoning 片段:\n{}\n  ├─ （结束）",
                            r_snippet
                        );
                    } else {
                        let _ = writeln!(
                            std::io::stderr(),
                            "  ├─ 无 reasoning 输出（模型可能刚开始就被掐断）"
                        );
                    }
                } else if !partial.is_empty() {
                    let snippet = if partial.chars().count() > 600 {
                        let char_count = partial.chars().count();
                        let head: String = partial.chars().take(300).collect();
                        let tail: String = partial.chars().skip(char_count - 300).collect();
                        format!("{}…[截断，共 {} 字符]…{}", head, char_count, tail)
                    } else {
                        partial.to_string()
                    };
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ├─ 已产出的部分文本（{} 字符）:\n{}\n  ├─ （结束）",
                        partial.len(),
                        snippet
                    );
                    if !reasoning.is_empty() {
                        let _ = writeln!(
                            std::io::stderr(),
                            "  ├─ reasoning 内容已产出 {} 字符",
                            stream_result.reasoning_text.len()
                        );
                    }
                } else {
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ├─ 无可见文本、无 reasoning（可能 tool call JSON 被截断丢弃）"
                    );
                }
                // 打印服务端返回的 finish_reason 原始值和 usage 统计，
                // 用于排查"reasoning token 耗尽预算导致零输出截断"等根因。
                if let Some(ref reason) = stream_result.finish_reason_value {
                    let _ = writeln!(std::io::stderr(), "  ├─ finish_reason = {:?}", reason);
                }
                if stream_result.usage_prompt_tokens > 0
                    || stream_result.usage_completion_tokens > 0
                    || stream_result.usage_reasoning_tokens > 0
                {
                    let _ = writeln!(
                        std::io::stderr(),
                        "  ├─ usage: prompt={}, completion={} (reasoning={})",
                        stream_result.usage_prompt_tokens,
                        stream_result.usage_completion_tokens,
                        stream_result.usage_reasoning_tokens,
                    );
                } else {
                    let _ = writeln!(std::io::stderr(), "  ├─ usage: 服务端未返回 token 统计");
                }
                let _ = writeln!(std::io::stderr(), "  └─ （诊断结束）");
                append_truncation_retry_note(&stream_result, messages, consecutive_truncations);
                Ok(TurnLoopStep::Continue)
            }
        }
        IterationExecution::FinalResponse(stream_result) => {
            // 收尾 veto：仍有未收口的 subagent task 时，打回一轮强制收集结果。
            // 但必须尊重迭代硬上限——否则子任务永不到终态且模型拒绝 task_wait 时
            // 会无限活锁，而且每轮重置 force_final_response 还会反复顶掉 orchestrator
            // 的安全刹车（tool-loop / progress-budget / iteration-limit hard-stop）。
            // 到达硬上限后放行收尾，让 max_iterations 保持为权威天花板。
            if iteration < max_iterations
                && reopen_turn_for_outstanding_subagent_tasks(messages, &app.session_id)
            {
                *force_final_response = false;
                return Ok(TurnLoopStep::Continue);
            }
            let reasoning_only_completion = stream_result.assistant_text.trim().is_empty()
                && !stream_result.reasoning_text.trim().is_empty()
                && stream_result.tool_calls.is_empty();
            if reasoning_only_completion {
                if *force_final_response {
                    *final_assistant_text =
                        "[模型只返回了思考内容，没有给出最终回答，请重试或切换模型]".to_string();
                    return Ok(TurnLoopStep::Break);
                }
                *force_final_response = true;
                return Ok(TurnLoopStep::Continue);
            }
            let was_truncated_by_length = stream_result.truncated_by_length;
            record_final_stream_response(
                app,
                stream_result,
                messages,
                turn_messages,
                final_assistant_text,
                final_assistant_recorded,
            );
            // finish_reason=length 但有可见文本：按 Completed 接受，但注入一条轻量
            // 提示让模型知道输出可能不完整。不触发重试（避免推理模型 reasoning
            // 占满预算时无意义循环），只在下轮请求里提醒模型自行检查/补全。
            if was_truncated_by_length {
                let note = "self_note:output_length_warning\n\
                            上一轮响应触发了输出长度上限（finish_reason=length）。\n\
                            已保留可见文本作为本轮回答。若你判断内容可能不完整（如文件写入中途被截断），\n\
                            请在下一步主动检查并补全；若内容已完整则忽略此提示。";
                messages.push(Message {
                    role: ROLE_INTERNAL_NOTE.to_string(),
                    content: serde_json::Value::String(note.to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            // 收尾 veto 仅在 iteration < max_iterations 时打回；到达硬上限后
            // reopen 被跳过，模型最终回答可能完全忽略未回收的子任务。此处把
            // 未回收子任务状态附进最终回答做可见性兜底（不再打回，避免活锁）。
            if iteration >= max_iterations {
                if let Ok(Some(notice)) =
                    task_tools::build_abandoned_tasks_notice(&app.session_id, max_iterations)
                {
                    final_assistant_text.push_str("\n\n");
                    final_assistant_text.push_str(&notice);
                }
            }
            Ok(TurnLoopStep::Break)
        }
        IterationExecution::ToolCall(tool_call_execution) => {
            let patch_retry_needs_fresh_read = !*force_final_response
                && patch_retry_requires_fresh_read(
                    messages,
                    &tool_call_execution.stream_result.tool_calls,
                );
            let rejection_reason = if *force_final_response {
                Some(ToolCallRejectionReason::NoToolHandoff)
            } else if patch_retry_needs_fresh_read {
                Some(ToolCallRejectionReason::PatchRetryNeedsFreshRead)
            } else {
                None
            };
            let suppressed_read_only_call_ids = if rejection_reason.is_none() {
                let mut call_ids = duplicate_read_only_call_ids(
                    messages,
                    &tool_call_execution.stream_result.tool_calls,
                );
                call_ids.extend(duplicate_knowledge_search_call_ids(
                    messages,
                    &tool_call_execution.stream_result.tool_calls,
                ));
                call_ids
            } else {
                HashSet::new()
            };
            let image_read_paths = if rejection_reason.is_none() {
                extract_image_paths_from_file_read_tool_calls(
                    &tool_call_execution.stream_result.tool_calls,
                )
            } else {
                Vec::new()
            };
            *terminal_dedupe_candidate = handle_tool_call_round(
                app,
                mcp_client,
                shared_mcp_client,
                &tool_call_execution,
                messages,
                turn_messages,
                one_shot_mode,
                persisted_turn_messages,
                iteration,
                rejection_reason,
                &suppressed_read_only_call_ids,
                turn_had_tool_error,
            )?;
            append_auto_image_followup_message(
                app,
                question,
                shared_mcp_client,
                &image_read_paths,
                messages,
                turn_messages,
            )?;

            crate::ai::driver::input::clear_stdin_buffer();

            {
                let mut os = app.os.lock().unwrap();
                if os.consume_yield_requested() {
                    return Ok(TurnLoopStep::Return(
                        crate::ai::driver::turn_runtime::types::TurnOutcome::Continue,
                    ));
                }
            }

            if iteration >= max_iterations {
                if *force_final_response {
                    let mut text = format!(
                        "Agent reached the tool iteration limit ({max_iterations}) without producing a final answer."
                    );
                    // 到达硬上限放行收尾：把仍未回收的子任务状态附进最终回答做
                    // 可见性兜底。此处不再打回模型（避免子任务永不到终态时无限
                    // 活锁），仅确保未回收结果不被静默抛弃。
                    if let Ok(Some(notice)) =
                        task_tools::build_abandoned_tasks_notice(&app.session_id, max_iterations)
                    {
                        text.push_str("\n\n");
                        text.push_str(&notice);
                    }
                    *final_assistant_text = text;
                    return Ok(TurnLoopStep::Break);
                }
                *force_final_response = true;
            } else {
                // AIOS: kernel is the authoritative source for tool-call quota.
                // 当前 usage 已经超限、或下一次 tool call 会超限，都应该切到
                // force-final，但 tool-call 配额本身不该阻断“无工具的最终回答”。
                use aios_kernel::primitives::{ResourceUsageDelta, RlimitDim, RlimitVerdict};
                let os = app.os.lock().unwrap();
                if let Some(pid) = os.current_process_id() {
                    let current_verdict = os.rlimit_check(pid, &Default::default());
                    let next_tool_verdict = os.rlimit_check(
                        pid,
                        &ResourceUsageDelta {
                            tool_calls: 1,
                            ..Default::default()
                        },
                    );
                    drop(os);
                    if let RlimitVerdict::Exceeded {
                        dimension,
                        used,
                        limit,
                    } = current_verdict
                    {
                        match dimension {
                            RlimitDim::Turns => {
                                if *force_final_response {
                                    *final_assistant_text = format!(
                                        "Agent exceeded kernel rlimit ({:?}: used={} limit={}).",
                                        dimension, used, limit
                                    );
                                    return Ok(TurnLoopStep::Break);
                                }
                                *force_final_response = true;
                            }
                            RlimitDim::ToolCalls => {
                                *force_final_response = true;
                            }
                            _ => {}
                        }
                    }
                    if matches!(
                        next_tool_verdict,
                        RlimitVerdict::Exceeded {
                            dimension: RlimitDim::ToolCalls,
                            ..
                        }
                    ) {
                        *force_final_response = true;
                    }
                }
            }

            Ok(TurnLoopStep::Continue)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{
        cli::ParsedCli,
        driver::signal,
        types::{
            AgentContext, App, AppConfig, FunctionCall, FunctionDefinition, ToolDefinition,
            ToolResult,
        },
    };
    use aios_kernel::primitives::ResourceLimit;
    use rust_tools::cw::SkipMap;
    use serde_json::Value;
    use std::path::PathBuf;
    use std::sync::{Arc, atomic::AtomicBool};
    use std::time::{Duration, Instant};

    fn test_app_with_tools(tool_names: &[&str]) -> App {
        App {
            cli: ParsedCli::default(),
            config: AppConfig {
                api_key: String::new(),
                base_history_file: PathBuf::new(),
                history_file: PathBuf::new(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 0,
                history_keep_last: 0,
                history_summary_max_chars: 0,
                intent_model: None,
                agent_route_model_path: PathBuf::new(),
                skill_match_model_path: PathBuf::new(),
            },
            session_id: "test".to_string(),
            session_history_file: PathBuf::new(),
            active_persona: crate::ai::persona::default_persona(),
            client: reqwest::Client::builder().build().unwrap(),
            current_model: String::new(),
            current_agent: "build".to_string(),
            current_agent_manifest: None,
            pending_files: None,
            forced_skill: None,
            forced_question: None,
            attached_image_files: Vec::new(),
            shutdown: Arc::new(AtomicBool::new(false)),
            streaming: Arc::new(AtomicBool::new(false)),
            cancel_stream: Arc::new(AtomicBool::new(false)),
            ignore_next_prompt_interrupt: false,
            prompt_editor: None,
            agent_context: Some(AgentContext {
                tools: tool_names
                    .iter()
                    .map(|name| ToolDefinition {
                        tool_type: "function".to_string(),
                        function: FunctionDefinition {
                            name: (*name).to_string(),
                            description: String::new(),
                            parameters: serde_json::json!({}),
                        },
                    })
                    .collect(),
                mcp_servers: SkipMap::default(),
                max_iterations: 16,
            }),
            last_skill_bias: None,
            os: crate::ai::driver::new_local_kernel(),
            agent_reload_counter: None,
            observers: vec![Box::new(
                crate::ai::driver::thinking::ThinkingOrchestrator::new(),
            )],
            last_known_prompt_tokens: None,
            last_known_cached_prompt_tokens: None,
            goal_mode: None,
            last_turn_had_tool_calls: false,
            last_turn_interrupted: false,
            prune_marks: Default::default(),
            turn_reasoning_items: Default::default(),
        }
    }

    fn test_tool_call(id: &str, name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    fn assistant_tool_call_message(tool_call: ToolCall) -> Message {
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![tool_call]),
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn tool_result_message(id: &str, content: &str) -> Message {
        Message {
            role: "tool".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: Some(id.to_string()),
            reasoning_content: None,
        }
    }

    #[test]
    fn duplicate_read_only_call_ids_span_intervening_tool_calls() {
        let args = serde_json::json!({ "file_path": "/tmp/demo.txt", "offset": 1 });
        let previous = test_tool_call("call_previous", "read_file", args.clone());
        let current = test_tool_call("call_current", "read_file", args);
        let messages = vec![
            assistant_tool_call_message(previous),
            tool_result_message("call_previous", "previous result"),
            assistant_tool_call_message(test_tool_call(
                "call_other",
                "list_directory",
                serde_json::json!({ "path": "/tmp" }),
            )),
            tool_result_message("call_other", "other.rs"),
        ];

        assert_eq!(
            duplicate_read_only_call_ids(&messages, &[current]),
            HashSet::from(["call_current".to_string()])
        );
    }

    #[test]
    fn duplicate_read_only_call_ids_do_not_cross_user_boundary() {
        let args = serde_json::json!({ "file_path": "/tmp/demo.txt" });
        let previous = test_tool_call("call_previous", "read_file", args.clone());
        let current = test_tool_call("call_current", "read_file", args);
        let messages = vec![
            assistant_tool_call_message(previous),
            tool_result_message("call_previous", "previous result"),
            Message {
                role: "user".to_string(),
                content: Value::String("read it again".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        assert!(duplicate_read_only_call_ids(&messages, &[current]).is_empty());
    }

    #[test]
    fn repeated_mutating_tool_request_is_not_suppressed() {
        let args = serde_json::json!({ "command": "cargo check" });
        let previous = test_tool_call("call_previous", "execute_command", args.clone());
        let current = test_tool_call("call_current", "execute_command", args);
        let messages = vec![
            assistant_tool_call_message(previous),
            tool_result_message("call_previous", "previous result"),
        ];

        assert!(duplicate_read_only_call_ids(&messages, &[current]).is_empty());
    }

    #[test]
    fn failed_read_only_call_is_not_suppressed() {
        let args = serde_json::json!({ "file_path": "/tmp/demo.txt" });
        let previous = test_tool_call("call_previous", "read_file", args.clone());
        let current = test_tool_call("call_current", "read_file", args);
        let messages = vec![
            assistant_tool_call_message(previous),
            tool_result_message("call_previous", "Error: file temporarily unavailable"),
        ];

        assert!(duplicate_read_only_call_ids(&messages, &[current]).is_empty());
    }

    #[test]
    fn duplicate_knowledge_search_is_suppressed_inside_mixed_tool_batch() {
        let previous = test_tool_call(
            "call_search_previous",
            "knowledge_search",
            serde_json::json!({ "query": "durable preference" }),
        );
        let messages = vec![
            assistant_tool_call_message(previous),
            tool_result_message("call_search_previous", "1. matching preference"),
        ];
        let current = vec![
            test_tool_call(
                "call_command",
                "execute_command",
                serde_json::json!({ "command": "pwd" }),
            ),
            test_tool_call(
                "call_search_retry",
                "knowledge_search",
                serde_json::json!({
                    "query": "  DURABLE PREFERENCE ",
                    "category": "",
                    "limit": 10
                }),
            ),
        ];

        let suppressed = duplicate_knowledge_search_call_ids(&messages, &current);
        assert_eq!(suppressed, HashSet::from(["call_search_retry".to_string()]));
    }

    #[test]
    fn knowledge_change_allows_the_same_search_again() {
        let messages = vec![
            assistant_tool_call_message(test_tool_call(
                "call_search_previous",
                "knowledge_search",
                serde_json::json!({ "query": "durable preference" }),
            )),
            tool_result_message("call_search_previous", "1. matching preference"),
            assistant_tool_call_message(test_tool_call(
                "call_save",
                "knowledge_save",
                serde_json::json!({ "content": "new durable preference" }),
            )),
            tool_result_message("call_save", "Saved to knowledge"),
        ];
        let current = test_tool_call(
            "call_search_retry",
            "knowledge_search",
            serde_json::json!({ "query": "durable preference" }),
        );

        assert!(duplicate_knowledge_search_call_ids(&messages, &[current]).is_empty());
    }

    #[test]
    fn failed_knowledge_search_does_not_block_retry() {
        let previous = test_tool_call(
            "call_search_previous",
            "knowledge_search",
            serde_json::json!({ "query": "durable preference" }),
        );
        let messages = vec![
            assistant_tool_call_message(previous),
            tool_result_message(
                "call_search_previous",
                "Error: knowledge database unavailable",
            ),
        ];
        let current = test_tool_call(
            "call_search_retry",
            "knowledge_search",
            serde_json::json!({ "query": "durable preference" }),
        );

        assert!(duplicate_knowledge_search_call_ids(&messages, &[current]).is_empty());
    }

    #[test]
    fn patch_retry_requires_fresh_read_after_context_mismatch() {
        let path = "/tmp/patch-target.rs";
        let messages = vec![
            assistant_tool_call_message(test_tool_call(
                "call_failed_patch",
                "apply_patch",
                serde_json::json!({ "file_path": path, "patch": "@@\n-old\n+new" }),
            )),
            tool_result_message(
                "call_failed_patch",
                "Error: apply_patch failed: context mismatch: patch hunk could not be located.",
            ),
        ];
        let retry = test_tool_call(
            "call_retry",
            "apply_patch",
            serde_json::json!({ "path": path, "patch": "@@\n-old\n+newer" }),
        );

        assert!(patch_retry_requires_fresh_read(&messages, &[retry]));
    }

    #[test]
    fn patch_retry_is_released_by_successful_read_of_same_target() {
        let path = "/tmp/patch-target.rs";
        let messages = vec![
            assistant_tool_call_message(test_tool_call(
                "call_failed_patch",
                "apply_patch",
                serde_json::json!({ "file_path": path, "patch": "@@\n-old\n+new" }),
            )),
            tool_result_message(
                "call_failed_patch",
                "Error: apply_patch failed: ambiguous patch: hunk context matches 2 locations.",
            ),
            assistant_tool_call_message(test_tool_call(
                "call_fresh_read",
                "read_file",
                serde_json::json!({ "path": path }),
            )),
            tool_result_message("call_fresh_read", "fn current() {}\n"),
        ];
        let retry = test_tool_call(
            "call_retry",
            "apply_patch",
            serde_json::json!({ "file_path": path, "patch": "@@\n-old\n+newer" }),
        );

        assert!(!patch_retry_requires_fresh_read(&messages, &[retry]));
    }

    #[test]
    fn patch_retry_is_not_released_by_read_of_another_target() {
        let patch_path = "/tmp/patch-target.rs";
        let messages = vec![
            assistant_tool_call_message(test_tool_call(
                "call_failed_patch",
                "apply_patch",
                serde_json::json!({
                    "patch": format!(
                        "*** Begin Patch\n*** Update File: {patch_path}\n@@\n-old\n+new\n*** End Patch"
                    )
                }),
            )),
            tool_result_message(
                "call_failed_patch",
                "Error: apply_patch failed: context mismatch: patch hunk could not be located.",
            ),
            assistant_tool_call_message(test_tool_call(
                "call_other_read",
                "read_file",
                serde_json::json!({ "file_path": "/tmp/another-target.rs" }),
            )),
            tool_result_message("call_other_read", "unrelated current content\n"),
        ];
        let retry = test_tool_call(
            "call_retry",
            "apply_patch",
            serde_json::json!({ "file_path": patch_path, "patch": "@@\n-old\n+newer" }),
        );

        assert!(patch_retry_requires_fresh_read(&messages, &[retry]));
    }

    #[test]
    fn patch_retry_multi_file_failure_blocks_all_targets_until_each_is_re_read() {
        let a = "/tmp/patch-a.rs";
        let b = "/tmp/patch-b.rs";
        let messages = vec![
            assistant_tool_call_message(test_tool_call(
                "call_failed_patch",
                "apply_patch",
                serde_json::json!({
                    "patch": format!(
                        "*** Begin Patch\n*** Update File: {a}\n@@\n-old_a\n+new_a\n*** Update File: {b}\n@@\n-old_b\n+new_b\n*** End Patch"
                    )
                }),
            )),
            tool_result_message(
                "call_failed_patch",
                &format!(
                    "Error: apply_patch failed: failed while preparing patch for {b}: context mismatch: patch hunk could not be located."
                ),
            ),
            assistant_tool_call_message(test_tool_call(
                "call_read_a",
                "read_file",
                serde_json::json!({ "file_path": a }),
            )),
            tool_result_message("call_read_a", "fn current_a() {}\n"),
        ];
        let retry = test_tool_call(
            "call_retry_b",
            "apply_patch",
            serde_json::json!({ "file_path": b, "patch": "@@\n-old_b\n+newer_b" }),
        );

        assert!(patch_retry_requires_fresh_read(&messages, &[retry]));
    }

    #[test]
    fn patch_retry_multi_file_failure_is_released_after_all_targets_are_re_read() {
        let a = "/tmp/patch-a.rs";
        let b = "/tmp/patch-b.rs";
        let messages = vec![
            assistant_tool_call_message(test_tool_call(
                "call_failed_patch",
                "apply_patch",
                serde_json::json!({
                    "patch": format!(
                        "*** Begin Patch\n*** Update File: {a}\n@@\n-old_a\n+new_a\n*** Update File: {b}\n@@\n-old_b\n+new_b\n*** End Patch"
                    )
                }),
            )),
            tool_result_message(
                "call_failed_patch",
                &format!(
                    "Error: apply_patch failed: failed while preparing patch for {b}: ambiguous patch: hunk context matches 2 locations."
                ),
            ),
            assistant_tool_call_message(test_tool_call(
                "call_read_a",
                "read_file",
                serde_json::json!({ "file_path": a }),
            )),
            tool_result_message("call_read_a", "fn current_a() {}\n"),
            assistant_tool_call_message(test_tool_call(
                "call_read_b",
                "read_file",
                serde_json::json!({ "path": b }),
            )),
            tool_result_message("call_read_b", "1| fn current_b() {}\n"),
        ];
        let retry = test_tool_call(
            "call_retry_b",
            "apply_patch",
            serde_json::json!({ "file_path": b, "patch": "@@\n-old_b\n+newer_b" }),
        );

        assert!(!patch_retry_requires_fresh_read(&messages, &[retry]));
    }

    #[test]
    fn duplicate_read_only_tool_call_is_suppressed_without_forcing_final_response() {
        let mut app = test_app_with_tools(&["read_file"]);
        let shared_mcp_client = Arc::new(std::sync::Mutex::new(McpClient::new()));
        let current_call = test_tool_call(
            "call_current",
            "read_file",
            serde_json::json!({ "file_path": "/tmp/demo.txt" }),
        );
        let mut messages = vec![
            assistant_tool_call_message(test_tool_call(
                "call_previous",
                "read_file",
                serde_json::json!({ "file_path": "/tmp/demo.txt" }),
            )),
            tool_result_message("call_previous", "previous result"),
        ];
        let mut turn_messages = Vec::new();
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut terminal_dedupe_candidate = None;
        let consecutive_truncations = 0;
        let mut force_final_response = false;
        let mut persisted_turn_messages = 0;
        let mut turn_had_tool_error = false;

        let step = handle_iteration_execution(
            &mut app,
            "read the file",
            &shared_mcp_client.lock().unwrap(),
            &shared_mcp_client,
            IterationExecution::ToolCall(ToolCallExecution {
                stream_result: crate::ai::types::StreamResult {
                    tool_calls: vec![current_call],
                    ..Default::default()
                },
                allowed_tool_names: rust_tools::commonw::FastSet::from_iter([
                    "read_file".to_string()
                ]),
            }),
            &mut messages,
            &mut turn_messages,
            false,
            &mut persisted_turn_messages,
            &mut final_assistant_text,
            &mut final_assistant_recorded,
            &mut force_final_response,
            &mut terminal_dedupe_candidate,
            false,
            1,
            16,
            consecutive_truncations,
            &mut turn_had_tool_error,
        )
        .unwrap();

        assert!(matches!(step, TurnLoopStep::Continue));
        assert!(!force_final_response);
        assert!(turn_had_tool_error);
        let rejected_tool_result = messages
            .iter()
            .rev()
            .find(|message| message.role == "tool")
            .expect("rejection should append a tool result");
        assert!(
            rejected_tool_result
                .content
                .as_str()
                .unwrap_or_default()
                .contains("already completed successfully in the current user turn")
        );
        assert!(
            rejected_tool_result
                .content
                .as_str()
                .unwrap_or_default()
                .contains("underlying data changes")
        );
    }

    #[test]
    fn patch_retry_without_fresh_read_is_rejected() {
        let mut app = test_app_with_tools(&["apply_patch", "read_file"]);
        let shared_mcp_client = Arc::new(std::sync::Mutex::new(McpClient::new()));
        let path = "/tmp/patch-target.rs";
        let current_call = test_tool_call(
            "call_retry",
            "apply_patch",
            serde_json::json!({ "file_path": path, "patch": "@@\n-old\n+new" }),
        );
        let mut messages = vec![
            assistant_tool_call_message(test_tool_call(
                "call_failed_patch",
                "apply_patch",
                serde_json::json!({ "file_path": path, "patch": "@@\n-old\n+new" }),
            )),
            tool_result_message(
                "call_failed_patch",
                "Error: apply_patch failed: context mismatch: patch hunk could not be located.",
            ),
        ];
        let mut turn_messages = Vec::new();
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut terminal_dedupe_candidate = None;
        let consecutive_truncations = 0;
        let mut force_final_response = false;
        let mut persisted_turn_messages = 0;
        let mut turn_had_tool_error = false;

        let step = handle_iteration_execution(
            &mut app,
            "update the file",
            &shared_mcp_client.lock().unwrap(),
            &shared_mcp_client,
            IterationExecution::ToolCall(ToolCallExecution {
                stream_result: crate::ai::types::StreamResult {
                    tool_calls: vec![current_call],
                    ..Default::default()
                },
                allowed_tool_names: rust_tools::commonw::FastSet::from_iter([
                    "apply_patch".to_string(),
                    "read_file".to_string(),
                ]),
            }),
            &mut messages,
            &mut turn_messages,
            false,
            &mut persisted_turn_messages,
            &mut final_assistant_text,
            &mut final_assistant_recorded,
            &mut force_final_response,
            &mut terminal_dedupe_candidate,
            false,
            1,
            16,
            consecutive_truncations,
            &mut turn_had_tool_error,
        )
        .unwrap();

        assert!(matches!(step, TurnLoopStep::Continue));
        assert!(turn_had_tool_error);
        let rejected_tool_result = messages
            .iter()
            .rev()
            .find(|message| message.role == "tool")
            .expect("rejection should append a tool result");
        assert!(
            rejected_tool_result
                .content
                .as_str()
                .unwrap_or_default()
                .contains("apply_patch retry blocked")
        );
    }

    #[test]
    fn tool_call_round_persists_hidden_context_checkpoint() {
        let session_root =
            std::env::temp_dir().join(format!("ai-tool-round-checkpoint-{}", uuid::Uuid::new_v4()));
        let history_file = session_root.join("history.sqlite");
        let mut app = test_app_with_tools(&["read_file"]);
        app.session_history_file = history_file.clone();
        app.session_id = "checkpoint-test".to_string();

        let shared_mcp_client = Arc::new(std::sync::Mutex::new(McpClient::new()));
        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut terminal_dedupe_candidate = None;
        let mut force_final_response = false;
        let mut persisted_turn_messages = 0;
        let mut turn_had_tool_error = false;

        let step = handle_iteration_execution(
            &mut app,
            "read the file and continue",
            &shared_mcp_client.lock().unwrap(),
            &shared_mcp_client,
            IterationExecution::ToolCall(ToolCallExecution {
                stream_result: crate::ai::types::StreamResult {
                    assistant_text: "先读文件。".to_string(),
                    hidden_meta: "<meta:self_note>\n<context_checkpoint>\nsummary: 已确认根因\n证据：src/lib.rs:42。\n</context_checkpoint>\n</meta:self_note>".to_string(),
                    tool_calls: vec![test_tool_call(
                        "call_read",
                        "read_file",
                        serde_json::json!({ "file_path": "Cargo.toml" }),
                    )],
                    ..Default::default()
                },
                allowed_tool_names: rust_tools::commonw::FastSet::from_iter(["read_file".to_string()]),
            }),
            &mut messages,
            &mut turn_messages,
            false,
            &mut persisted_turn_messages,
            &mut final_assistant_text,
            &mut final_assistant_recorded,
            &mut force_final_response,
            &mut terminal_dedupe_candidate,
            false,
            1,
            16,
            0,
            &mut turn_had_tool_error,
        )
        .unwrap();

        assert!(matches!(step, TurnLoopStep::Continue));
        let checkpoint_marker = turn_messages
            .iter()
            .find_map(|message| {
                (message.role == ROLE_INTERNAL_NOTE)
                    .then(|| message.content.as_str())
                    .flatten()
                    .filter(|content| content.starts_with("[context_checkpoint path="))
            })
            .expect("tool-call hidden checkpoint should be persisted");
        let marker_path = checkpoint_marker
            .strip_prefix("[context_checkpoint path=")
            .and_then(|rest| rest.split(']').next())
            .expect("marker should include checkpoint path");
        assert!(
            std::path::Path::new(marker_path).is_file(),
            "checkpoint file should exist: {marker_path}"
        );

        let _ = std::fs::remove_dir_all(session_root.join("history.sessions"));
    }

    #[test]
    fn tool_call_round_no_longer_requests_terminal_dedupe() {
        let exec_result = ExecuteToolCallsResult {
            executed_tool_calls: vec![ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "execute_command".to_string(),
                    arguments: "{\"command\":\"seq 3\"}".to_string(),
                },
            }],
            tool_results: vec![ToolResult {
                tool_call_id: "call_1".to_string(),
                content: "1\n2\n3\n".to_string(),
            }],
            cached_hits: vec![false],
            had_error: false,
        };

        assert_eq!(exec_result.executed_tool_calls.len(), 1);
        assert_eq!(exec_result.tool_results.len(), 1);
    }

    #[test]
    fn extract_image_paths_from_file_read_tool_calls_collects_image_reads() {
        let tool_calls = vec![
            ToolCall {
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: r#"{"file_path":"/tmp/shot.png"}"#.to_string(),
                },
            },
            ToolCall {
                id: "call_2".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: r#"{"file_path":"/tmp/notes.txt"}"#.to_string(),
                },
            },
        ];
        assert_eq!(
            extract_image_paths_from_file_read_tool_calls(&tool_calls),
            vec!["/tmp/shot.png".to_string()]
        );
    }

    #[test]
    fn tty_tool_output_fold_window_keeps_latest_visible_lines() {
        // 断言正文/标记原样存在；置宽 COLUMNS 以免与 COLUMNS=12 的 clamp 用例并发时
        // 读到泄漏的窄列宽而被截断。
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        unsafe {
            std::env::set_var("COLUMNS", "200");
        }

        let mut fold = TtyToolOutputFoldState::default();
        fold.total_lines = TOOL_OUTPUT_FOLD_MAX_VISIBLE;
        for idx in 1..=TOOL_OUTPUT_FOLD_MAX_VISIBLE {
            fold.recent_lines.push_back(format!("line-{idx}"));
        }
        fold.current_line = format!("line-{}", TOOL_OUTPUT_FOLD_MAX_VISIBLE + 1);

        let expected_owned = (2..=TOOL_OUTPUT_FOLD_MAX_VISIBLE + 1)
            .map(|idx| format!("line-{idx}"))
            .collect::<Vec<_>>();
        assert_eq!(tty_tool_output_hidden_count(&fold), 1);
        assert_eq!(
            tty_tool_output_visible_lines(&fold),
            expected_owned
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );

        let (window, _) = render_tty_tool_output_fold_window(&fold);
        assert_eq!(window.matches("lines folded").count(), 1);
        assert!(!window.contains("line-1"));
        assert!(window.contains("line-2"));
        assert!(window.contains(&format!("line-{}", TOOL_OUTPUT_FOLD_MAX_VISIBLE + 1)));

        unsafe {
            std::env::remove_var("COLUMNS");
        }
    }

    #[test]
    fn tty_tool_output_fold_window_preserves_mock_qr_output() {
        // 模拟扫码登录命令输出：二维码通常为 30–50 行，不能被通用日志折叠策略截断。
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        unsafe {
            std::env::set_var("COLUMNS", "200");
        }

        let mock_qr = (0..41)
            .map(|row| format!("mock-qr-{row:02} ██  ██  ██  ██"))
            .collect::<Vec<_>>();
        let mut fold = TtyToolOutputFoldState::default();
        fold.total_lines = mock_qr.len();
        fold.recent_lines.extend(mock_qr.iter().cloned());

        let (window, rows) = render_tty_tool_output_fold_window(&fold);
        assert_eq!(tty_tool_output_hidden_count(&fold), 0);
        assert_eq!(rows, mock_qr.len());
        assert!(!window.contains("lines folded"));
        for row in &mock_qr {
            assert!(window.contains(row), "missing QR row: {row}");
        }

        unsafe {
            std::env::remove_var("COLUMNS");
        }
    }

    #[test]
    fn terminal_visual_grid_detection_requires_a_block_glyph_grid() {
        // 普通命令输出（如 git diff）即使有很多行，也不能被渲染到终端。
        let git_diff = "diff --git a/file.rs b/file.rs\n@@ -1,3 +1,4 @@\n-old line\n+new line\n";
        assert!(!contains_terminal_visual_grid(git_diff));

        let mock_qr = (0..VISUAL_OUTPUT_MIN_CONSECUTIVE_GRID_ROWS)
            .map(|row| format!("mock-qr-{row:02} ██  ██  ██  ██"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(contains_terminal_visual_grid(&mock_qr));
    }

    #[test]
    fn command_input_marks_pseudo_terminal_mode() {
        let pty = format_command_input(
            r#"{"command":"login --qr","pty":true,"cwd":"/tmp"}"#,
        )
        .expect("valid command arguments");
        assert_eq!(pty, "login --qr  (cwd: /tmp)  (PTY)");

        let piped = format_command_input(r#"{"command":"git diff","pty":false}"#)
            .expect("valid command arguments");
        assert_eq!(piped, "git diff");
    }

    #[test]
    fn tty_tool_output_fold_window_clamps_each_line_to_single_row() {
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        unsafe {
            std::env::set_var("COLUMNS", "12");
        }

        let mut fold = TtyToolOutputFoldState::default();
        fold.total_lines = TOOL_OUTPUT_FOLD_MAX_VISIBLE;
        fold.recent_lines
            .push_back("12345678901234567890".to_string());
        for idx in 0..(TOOL_OUTPUT_FOLD_MAX_VISIBLE - 2) {
            fold.recent_lines.push_back(format!("pad-{idx}"));
        }
        fold.recent_lines.push_back("abcdef".to_string());
        fold.current_line = "ghijklmnopqrst".to_string();

        let (window, rows) = render_tty_tool_output_fold_window(&fold);
        let visible_lines = tty_tool_output_visible_lines(&fold);

        // 每条渲染行被 clamp 成单物理行：窗口物理行数 == 1 折叠标记 + 可见逻辑行数。
        assert_eq!(rows, 1 + visible_lines.len());
        // 每条渲染行（去掉 `  │ ` 前缀与 ANSI 后）不超过终端列宽（12），cursor-up 精确。
        for line in window.lines() {
            let visible = crate::ai::driver::print::sanitize_for_terminal(line);
            assert!(
                unicode_width::UnicodeWidthStr::width(visible.as_str()) <= 12,
                "line exceeds terminal width: {visible:?}"
            );
        }
        assert!(!window.contains("12345678901234567890"));
        assert!(window.contains("abcdef"));
        // 超宽行被截断为省略号结尾，不再原样残留导致 cursor-up 少算行数。
        assert!(window.contains('…'));

        unsafe {
            std::env::remove_var("COLUMNS");
        }
    }

    #[test]
    fn reasoning_only_final_response_retries_once_with_forced_final() {
        let mut app = test_app_with_tools(&["read_file"]);
        let mcp = crate::ai::mcp::McpClient::new();
        let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(mcp));
        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        let mut persisted_turn_messages = 0usize;
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut force_final_response = false;
        let mut terminal_dedupe_candidate = None;

        let step = handle_iteration_execution(
            &mut app,
            "compare two yaml files",
            &shared_mcp.lock().unwrap(),
            &shared_mcp,
            IterationExecution::FinalResponse(crate::ai::types::StreamResult {
                outcome: crate::ai::types::StreamOutcome::Completed,
                tool_calls: Vec::new(),
                assistant_text: String::new(),
                hidden_meta: String::new(),
                reasoning_text: "I should read both files first.".to_string(),
                reasoning_items: Vec::new(),
                skip_response_drain: true,
                truncated_by_length: false,
                stream_error: false,
                finish_reason_value: None,
                usage_prompt_tokens: 0,
                usage_cached_prompt_tokens: 0,
                usage_completion_tokens: 0,
                usage_reasoning_tokens: 0,
            }),
            &mut messages,
            &mut turn_messages,
            false,
            &mut persisted_turn_messages,
            &mut final_assistant_text,
            &mut final_assistant_recorded,
            &mut force_final_response,
            &mut terminal_dedupe_candidate,
            true,
            1,
            16,
            0,
            &mut false,
        )
        .unwrap();

        assert!(matches!(step, TurnLoopStep::Continue));
        assert!(force_final_response);
        assert!(final_assistant_text.is_empty());
        assert!(!final_assistant_recorded);
        assert!(messages.is_empty());
        assert!(turn_messages.is_empty());
    }

    #[test]
    fn reasoning_only_final_response_stops_after_forced_retry() {
        let mut app = test_app_with_tools(&["read_file"]);
        let mcp = crate::ai::mcp::McpClient::new();
        let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(mcp));
        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        let mut persisted_turn_messages = 0usize;
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut force_final_response = true;
        let mut terminal_dedupe_candidate = None;

        let step = handle_iteration_execution(
            &mut app,
            "compare two yaml files",
            &shared_mcp.lock().unwrap(),
            &shared_mcp,
            IterationExecution::FinalResponse(crate::ai::types::StreamResult {
                outcome: crate::ai::types::StreamOutcome::Completed,
                tool_calls: Vec::new(),
                assistant_text: String::new(),
                hidden_meta: String::new(),
                reasoning_text: "I should read both files first.".to_string(),
                reasoning_items: Vec::new(),
                skip_response_drain: true,
                truncated_by_length: false,
                stream_error: false,
                finish_reason_value: None,
                usage_prompt_tokens: 0,
                usage_cached_prompt_tokens: 0,
                usage_completion_tokens: 0,
                usage_reasoning_tokens: 0,
            }),
            &mut messages,
            &mut turn_messages,
            false,
            &mut persisted_turn_messages,
            &mut final_assistant_text,
            &mut final_assistant_recorded,
            &mut force_final_response,
            &mut terminal_dedupe_candidate,
            true,
            2,
            16,
            0,
            &mut false,
        )
        .unwrap();

        assert!(matches!(step, TurnLoopStep::Break));
        assert_eq!(
            final_assistant_text,
            "[模型只返回了思考内容，没有给出最终回答，请重试或切换模型]"
        );
        assert!(!final_assistant_recorded);
        assert!(messages.is_empty());
        assert!(turn_messages.is_empty());
    }

    #[test]
    fn final_response_with_outstanding_subagent_task_reopens_turn_and_clears_no_tool_handoff() {
        let _env_guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut app = test_app_with_tools(&["task_wait", "task_status"]);
        app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
        crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());

        let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
        let (pid, result_channel_id) = {
            let mut os = app.os.lock().unwrap();
            let pid = os.begin_foreground(
                "child".to_string(),
                "goal".to_string(),
                10,
                usize::MAX,
                None,
            );
            let channel = os.channel_create(Some(pid), 1, "task-result".to_string());
            (pid, channel.raw())
        };
        crate::ai::tools::task_tools::insert_task_entry_for_test(
            task_id.clone(),
            crate::ai::tools::task_tools::AsyncTaskEntry {
                session_id: app.session_id.clone(),
                result_observed: false,
                owner_pid: pid,
                pid,
                result_channel_id,
                completion_futex_addr: aios_kernel::primitives::FutexAddr(1),
                description: "inspect parser".to_string(),
                agent_name: "build".to_string(),
                model: "qwen3.7-max".to_string(),
                is_model_auto_selected: false,
                auto_model_fallback: None,
                selection_explanation: "explicit override".to_string(),
                inherit: crate::ai::tools::task_tools::InheritOptions::default(),
                abort_handle: None,
                started_at: Instant::now(),
            },
        );

        let shared_mcp =
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
        let mut messages = vec![Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(no_tool_handoff_note().to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let mut turn_messages = Vec::new();
        let mut persisted_turn_messages = 0usize;
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut force_final_response = true;
        let mut terminal_dedupe_candidate = None;

        let step = handle_iteration_execution(
            &mut app,
            "wrap up",
            &shared_mcp.lock().unwrap(),
            &shared_mcp,
            IterationExecution::FinalResponse(crate::ai::types::StreamResult {
                outcome: crate::ai::types::StreamOutcome::Completed,
                tool_calls: Vec::new(),
                assistant_text: "done".to_string(),
                hidden_meta: String::new(),
                reasoning_text: String::new(),
                reasoning_items: Vec::new(),
                skip_response_drain: true,
                truncated_by_length: false,
                stream_error: false,
                finish_reason_value: None,
                usage_prompt_tokens: 0,
                usage_cached_prompt_tokens: 0,
                usage_completion_tokens: 0,
                usage_reasoning_tokens: 0,
            }),
            &mut messages,
            &mut turn_messages,
            false,
            &mut persisted_turn_messages,
            &mut final_assistant_text,
            &mut final_assistant_recorded,
            &mut force_final_response,
            &mut terminal_dedupe_candidate,
            true,
            2,
            16,
            0,
            &mut false,
        )
        .unwrap();

        assert!(matches!(step, TurnLoopStep::Continue));
        assert!(!force_final_response);
        assert!(final_assistant_text.is_empty());
        assert!(!final_assistant_recorded);
        assert!(turn_messages.is_empty());
        let joined = messages
            .iter()
            .map(|message| message.content.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains(PENDING_SUBAGENT_TASKS_FOLLOWUP_PREFIX.trim_end()));
        assert!(joined.contains(&task_id));
        assert!(joined.contains("Immediate next step: call `task_wait` or `task_status`"));
        assert!(!joined.contains(no_tool_handoff_note()));

        let _ = crate::ai::tools::task_tools::remove_task_entry(&task_id);
        if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
            *guard = None;
        }
    }

    #[test]
    fn final_response_at_iteration_ceiling_finishes_despite_outstanding_task() {
        // 迭代硬上限是权威天花板：即使还有未收口的 subagent task，也不能无限
        // 打回收尾（否则子任务永不到终态时会活锁，并反复顶掉安全刹车）。
        let _env_guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut app = test_app_with_tools(&["task_wait", "task_status"]);
        app.session_id = format!("test-session-{}", uuid::Uuid::new_v4().simple());
        crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());

        let task_id = format!("task_{}", uuid::Uuid::new_v4().simple());
        let (pid, result_channel_id) = {
            let mut os = app.os.lock().unwrap();
            let pid = os.begin_foreground(
                "child".to_string(),
                "goal".to_string(),
                10,
                usize::MAX,
                None,
            );
            let channel = os.channel_create(Some(pid), 1, "task-result".to_string());
            (pid, channel.raw())
        };
        crate::ai::tools::task_tools::insert_task_entry_for_test(
            task_id.clone(),
            crate::ai::tools::task_tools::AsyncTaskEntry {
                session_id: app.session_id.clone(),
                result_observed: false,
                owner_pid: pid,
                pid,
                result_channel_id,
                completion_futex_addr: aios_kernel::primitives::FutexAddr(1),
                description: "inspect parser".to_string(),
                agent_name: "build".to_string(),
                model: "qwen3.7-max".to_string(),
                is_model_auto_selected: false,
                auto_model_fallback: None,
                selection_explanation: "explicit override".to_string(),
                inherit: crate::ai::tools::task_tools::InheritOptions::default(),
                abort_handle: None,
                started_at: Instant::now(),
            },
        );

        let shared_mcp =
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        let mut persisted_turn_messages = 0usize;
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut force_final_response = true;
        let mut terminal_dedupe_candidate = None;

        let max_iterations = 16;
        let step = handle_iteration_execution(
            &mut app,
            "wrap up",
            &shared_mcp.lock().unwrap(),
            &shared_mcp,
            IterationExecution::FinalResponse(crate::ai::types::StreamResult {
                outcome: crate::ai::types::StreamOutcome::Completed,
                tool_calls: Vec::new(),
                assistant_text: "done".to_string(),
                hidden_meta: String::new(),
                reasoning_text: String::new(),
                reasoning_items: Vec::new(),
                skip_response_drain: true,
                truncated_by_length: false,
                stream_error: false,
                finish_reason_value: None,
                usage_prompt_tokens: 0,
                usage_cached_prompt_tokens: 0,
                usage_completion_tokens: 0,
                usage_reasoning_tokens: 0,
            }),
            &mut messages,
            &mut turn_messages,
            false,
            &mut persisted_turn_messages,
            &mut final_assistant_text,
            &mut final_assistant_recorded,
            &mut force_final_response,
            &mut terminal_dedupe_candidate,
            true,
            max_iterations,
            max_iterations,
            0,
            &mut false,
        )
        .unwrap();

        // 到达硬上限：不再打回，允许收尾。
        assert!(matches!(step, TurnLoopStep::Break));
        assert!(final_assistant_text.starts_with("done\n\n"));
        assert!(final_assistant_text.contains("1 spawned subagent task(s) were still outstanding"));
        assert!(final_assistant_text.contains(&task_id));
        assert!(final_assistant_text.contains("Required follow-up: re-run this turn"));
        let joined = messages
            .iter()
            .map(|message| message.content.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!joined.contains(PENDING_SUBAGENT_TASKS_FOLLOWUP_PREFIX.trim_end()));

        let _ = crate::ai::tools::task_tools::remove_task_entry(&task_id);
        if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
            *guard = None;
        }
    }

    #[test]
    fn truncated_response_retries_and_injects_shrink_note() {
        let mut app = test_app_with_tools(&["write_file"]);
        let mcp = crate::ai::mcp::McpClient::new();
        let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(mcp));
        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        let mut persisted_turn_messages = 0usize;
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut force_final_response = false;
        let mut terminal_dedupe_candidate = None;

        let step = handle_iteration_execution(
            &mut app,
            "write a big script",
            &shared_mcp.lock().unwrap(),
            &shared_mcp,
            IterationExecution::Truncated(crate::ai::types::StreamResult {
                outcome: crate::ai::types::StreamOutcome::Truncated,
                tool_calls: Vec::new(),
                assistant_text: "现在让我来编写一个综合脚本".to_string(),
                hidden_meta: String::new(),
                reasoning_text: String::new(),
                reasoning_items: Vec::new(),
                skip_response_drain: true,
                truncated_by_length: false,
                stream_error: false,
                finish_reason_value: None,
                usage_prompt_tokens: 0,
                usage_cached_prompt_tokens: 0,
                usage_completion_tokens: 0,
                usage_reasoning_tokens: 0,
            }),
            &mut messages,
            &mut turn_messages,
            false,
            &mut persisted_turn_messages,
            &mut final_assistant_text,
            &mut final_assistant_recorded,
            &mut force_final_response,
            &mut terminal_dedupe_candidate,
            true,
            1,
            16,
            1,
            &mut false,
        )
        .unwrap();

        // 截断应自动重试（Continue），不得静默完成。
        assert!(matches!(step, TurnLoopStep::Continue));
        assert!(final_assistant_text.is_empty());
        assert!(!final_assistant_recorded);
        // 部分可见文本被保留为 assistant 上下文。
        assert!(
            messages.iter().any(|m| m.role == "assistant"
                && m.content.as_str() == Some("现在让我来编写一个综合脚本"))
        );
        // partial text 不得写入 turn_messages 持久化轨道——连续截断时多条
        // 大体积半截文本会污染历史文件，导致下个 turn 正常历史被压缩丢弃。
        assert!(
            !turn_messages.iter().any(|m| m.role == "assistant"
                && m.content.as_str() == Some("现在让我来编写一个综合脚本")),
            "partial text must not leak into turn_messages (persistence track)"
        );
        // 注入了一条收缩重写提示。
        assert!(messages.iter().any(|m| {
            m.role == ROLE_INTERNAL_NOTE
                && m.content
                    .as_str()
                    .is_some_and(|c| c.starts_with(TRUNCATION_RETRY_NOTE_PREFIX))
        }));
    }

    #[test]
    fn truncation_retry_note_replaces_with_updated_count() {
        let mut app = test_app_with_tools(&["write_file"]);
        let mcp = crate::ai::mcp::McpClient::new();
        let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(mcp));
        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        let mut persisted_turn_messages = 0usize;
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut force_final_response = false;
        let mut terminal_dedupe_candidate = None;

        for consecutive in 1..=2 {
            handle_iteration_execution(
                &mut app,
                "write a big script",
                &shared_mcp.lock().unwrap(),
                &shared_mcp,
                IterationExecution::Truncated(crate::ai::types::StreamResult {
                    outcome: crate::ai::types::StreamOutcome::Truncated,
                    tool_calls: Vec::new(),
                    assistant_text: String::new(),
                    hidden_meta: String::new(),
                    reasoning_text: String::new(),
                    reasoning_items: Vec::new(),
                    skip_response_drain: true,
                    truncated_by_length: false,
                    stream_error: false,
                    finish_reason_value: None,
                    usage_prompt_tokens: 0,
                    usage_cached_prompt_tokens: 0,
                    usage_completion_tokens: 0,
                    usage_reasoning_tokens: 0,
                }),
                &mut messages,
                &mut turn_messages,
                false,
                &mut persisted_turn_messages,
                &mut final_assistant_text,
                &mut final_assistant_recorded,
                &mut force_final_response,
                &mut terminal_dedupe_candidate,
                true,
                1,
                16,
                consecutive,
                &mut false,
            )
            .unwrap();
        }

        let note_count = messages
            .iter()
            .filter(|m| {
                m.role == ROLE_INTERNAL_NOTE
                    && m.content
                        .as_str()
                        .is_some_and(|c| c.starts_with(TRUNCATION_RETRY_NOTE_PREFIX))
            })
            .count();
        // 旧 note 被移除、新 note 被注入，始终只有 1 条（而非堆叠 2 条）。
        assert_eq!(note_count, 1, "重复截断应替换旧 note 而非堆叠");
        // 第 2 次截断的 note 应携带计数 "2"，让模型感知严重程度递增。
        let note = messages.iter().find(|m| {
            m.role == ROLE_INTERNAL_NOTE
                && m.content
                    .as_str()
                    .is_some_and(|c| c.starts_with(TRUNCATION_RETRY_NOTE_PREFIX))
        });
        assert!(
            note.and_then(|m| m.content.as_str())
                .is_some_and(|c| c.contains("第 2 次")),
            "第 2 次截断的 note 应包含计数"
        );
    }

    #[test]
    fn stream_error_truncation_skips_shrink_note_and_partial_text() {
        let mut app = test_app_with_tools(&["write_file"]);
        let mcp = crate::ai::mcp::McpClient::new();
        let shared_mcp = std::sync::Arc::new(std::sync::Mutex::new(mcp));
        let mut messages = vec![Message {
            role: "user".to_string(),
            content: Value::String("write a big script".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let mut turn_messages = Vec::new();
        let mut persisted_turn_messages = 0;
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut force_final_response = false;
        let mut terminal_dedupe_candidate: Option<String> = None;

        let step = handle_iteration_execution(
            &mut app,
            "write a big script",
            &shared_mcp.lock().unwrap(),
            &shared_mcp,
            IterationExecution::Truncated(crate::ai::types::StreamResult {
                outcome: crate::ai::types::StreamOutcome::Truncated,
                tool_calls: Vec::new(),
                assistant_text: "partial content from broken stream".to_string(),
                hidden_meta: String::new(),
                reasoning_text: String::new(),
                reasoning_items: Vec::new(),
                skip_response_drain: true,
                truncated_by_length: false,
                stream_error: true,
                finish_reason_value: None,
                usage_prompt_tokens: 0,
                usage_cached_prompt_tokens: 0,
                usage_completion_tokens: 0,
                usage_reasoning_tokens: 0,
            }),
            &mut messages,
            &mut turn_messages,
            false,
            &mut persisted_turn_messages,
            &mut final_assistant_text,
            &mut final_assistant_recorded,
            &mut force_final_response,
            &mut terminal_dedupe_candidate,
            true,
            1,
            16,
            1,
            &mut false,
        )
        .unwrap();

        // 应该继续重试
        assert!(matches!(step, TurnLoopStep::Continue));
        // 不应注入收缩提示——流错误和输出大小无关
        let has_shrink_note = messages.iter().any(|m| {
            m.role == ROLE_INTERNAL_NOTE
                && m.content
                    .as_str()
                    .is_some_and(|c| c.starts_with(TRUNCATION_RETRY_NOTE_PREFIX))
        });
        assert!(!has_shrink_note, "stream_error 截断不应注入收缩提示");
        // 不应保留 partial text——流中断时的 partial 不可靠
        let has_partial = messages.iter().any(|m| {
            m.role == "assistant"
                && m.content
                    .as_str()
                    .is_some_and(|c| c.contains("partial content from broken stream"))
        });
        assert!(!has_partial, "stream_error 截断不应保留 partial text");
    }

    #[test]
    fn forced_final_hallucinated_tool_call_is_rejected_without_consuming_quota() {
        let _env_guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut app = test_app_with_tools(&["read_file"]);
        let pid = {
            let mut os = app.os.lock().unwrap();
            let pid =
                os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
            let mut lim = ResourceLimit::unlimited();
            lim.max_tool_calls = 64;
            os.rlimit_set(pid, lim).unwrap();
            pid
        };
        crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());

        let path = std::env::temp_dir().join(format!("forced-final-{}.txt", pid));
        std::fs::write(&path, "hello").unwrap();

        let shared_mcp =
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        let mut persisted_turn_messages = 0usize;
        let mut final_assistant_text = String::new();
        let mut final_assistant_recorded = false;
        let mut force_final_response = true;
        let mut terminal_dedupe_candidate = None;

        let step = handle_iteration_execution(
            &mut app,
            "summarize findings",
            &shared_mcp.lock().unwrap(),
            &shared_mcp,
            IterationExecution::ToolCall(ToolCallExecution {
                stream_result: crate::ai::types::StreamResult {
                    outcome: crate::ai::types::StreamOutcome::ToolCall,
                    tool_calls: vec![ToolCall {
                        id: "call_1".to_string(),
                        tool_type: "function".to_string(),
                        function: FunctionCall {
                            name: "read_file".to_string(),
                            arguments: format!(r#"{{"file_path":"{}"}}"#, path.to_string_lossy()),
                        },
                    }],
                    assistant_text: String::new(),
                    hidden_meta: String::new(),
                    reasoning_text: String::new(),
                    reasoning_items: Vec::new(),
                    skip_response_drain: true,
                    truncated_by_length: false,
                    stream_error: false,
                    finish_reason_value: None,
                    usage_prompt_tokens: 0,
                    usage_cached_prompt_tokens: 0,
                    usage_completion_tokens: 0,
                    usage_reasoning_tokens: 0,
                },
                allowed_tool_names: ["read_file".to_string()].into_iter().collect(),
            }),
            &mut messages,
            &mut turn_messages,
            false,
            &mut persisted_turn_messages,
            &mut final_assistant_text,
            &mut final_assistant_recorded,
            &mut force_final_response,
            &mut terminal_dedupe_candidate,
            true,
            3,
            16,
            0,
            &mut false,
        )
        .unwrap();

        assert!(matches!(step, TurnLoopStep::Continue));
        assert!(force_final_response);
        assert!(final_assistant_text.is_empty());
        assert!(!final_assistant_recorded);
        {
            let os = app.os.lock().unwrap();
            assert_eq!(os.rusage_get(pid).unwrap().tool_calls, 0);
        }
        let joined = turn_messages
            .iter()
            .map(|msg| msg.content.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(joined.contains("disabled in no-tool handoff mode"));
        assert!(!joined.contains("exceeded kernel rlimit"));

        let _ = std::fs::remove_file(&path);
        if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
            *guard = None;
        }
    }

    #[test]
    fn auto_image_followup_uses_multimodal_message_for_vl_model() {
        let mut app = test_app_with_tools(&[]);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("tool-followup-{}.png", uuid::Uuid::new_v4()));
        std::fs::write(&path, b"fake").unwrap();
        app.current_model = crate::ai::model_names::all()
            .iter()
            .find(|m| m.is_vl)
            .map(|m| m.name.clone())
            .expect("models.json must contain at least one VL model");

        let mut messages = Vec::new();
        let mut turn_messages = Vec::new();
        let shared_mcp =
            std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
        append_auto_image_followup_message(
            &app,
            "describe the file",
            &shared_mcp,
            &[path.to_string_lossy().to_string()],
            &mut messages,
            &mut turn_messages,
        )
        .unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert!(messages[0].content.is_array());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ctrl_c_during_foreground_tool_round_cancels_without_shutdown() {
        let _env_guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        signal::clear_request_interrupt();

        let app = test_app_with_tools(&["execute_command"]);
        {
            let mut os = app.os.lock().unwrap();
            let _ = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
        }
        crate::ai::tools::os_tools::init_os_tools_globals(app.os.clone());

        let streaming = app.streaming.clone();
        let shutdown = app.shutdown.clone();
        let cancel_stream = app.cancel_stream.clone();
        let started_marker = std::env::temp_dir().join(format!(
            "a_ctrl_c_foreground_tool_{}_{}",
            std::process::id(),
            Instant::now().elapsed().as_nanos()
        ));
        let command_marker = started_marker.to_string_lossy().replace('\'', "'\\''");

        let handle = std::thread::spawn(move || {
            let mut app = app;
            let mcp = crate::ai::mcp::McpClient::new();
            let shared_mcp =
                std::sync::Arc::new(std::sync::Mutex::new(crate::ai::mcp::McpClient::new()));
            let mut messages = Vec::new();
            let mut turn_messages = Vec::new();
            let mut persisted_turn_messages = 0usize;
            let mut turn_had_tool_error = false;
            let start = Instant::now();
            let result = handle_tool_call_round(
                &mut app,
                &mcp,
                &shared_mcp,
                &ToolCallExecution {
                    stream_result: crate::ai::types::StreamResult {
                        outcome: crate::ai::types::StreamOutcome::ToolCall,
                        tool_calls: vec![ToolCall {
                            id: "call_1".to_string(),
                            tool_type: "function".to_string(),
                            function: FunctionCall {
                                name: "execute_command".to_string(),
                                arguments: serde_json::json!({
                                    "command": format!("touch '{command_marker}'; sleep 2"),
                                })
                                .to_string(),
                            },
                        }],
                        assistant_text: String::new(),
                        hidden_meta: String::new(),
                        reasoning_text: String::new(),
                        reasoning_items: Vec::new(),
                        skip_response_drain: true,
                        truncated_by_length: false,
                        stream_error: false,
                        finish_reason_value: None,
                        usage_prompt_tokens: 0,
                        usage_cached_prompt_tokens: 0,
                        usage_completion_tokens: 0,
                        usage_reasoning_tokens: 0,
                    },
                    allowed_tool_names: ["execute_command".to_string()].into_iter().collect(),
                },
                &mut messages,
                &mut turn_messages,
                true,
                &mut persisted_turn_messages,
                1,
                None,
                &HashSet::new(),
                &mut turn_had_tool_error,
            );
            (
                result.map(|_| ()).map_err(|err| err.to_string()),
                start.elapsed(),
                app,
            )
        });

        let wait_started = Instant::now();
        while !started_marker.exists() && wait_started.elapsed() < Duration::from_secs(1) {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            started_marker.exists(),
            "foreground tool command never started"
        );

        signal::handle_sigint(
            shutdown.as_ref(),
            streaming.as_ref(),
            cancel_stream.as_ref(),
        );

        let (result, elapsed, returned_app) = handle.join().unwrap();
        let _ = std::fs::remove_file(&started_marker);

        returned_app
            .cancel_stream
            .store(false, std::sync::atomic::Ordering::Relaxed);
        crate::ai::tools::registry::common::clear_tool_cancel();
        signal::clear_request_interrupt();
        if let Ok(mut guard) = crate::ai::tools::os_tools::GLOBAL_OS.lock() {
            *guard = None;
        }

        assert!(result.is_ok());
        assert!(
            elapsed < Duration::from_secs(1),
            "tool round did not stop promptly after Ctrl+C: {elapsed:?}"
        );
        assert!(
            !shutdown.load(std::sync::atomic::Ordering::Relaxed),
            "Ctrl+C during foreground tool round should not request shutdown"
        );
    }
}
