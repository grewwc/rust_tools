//! 工具组折叠（tool group folding）。
//!
//! 把消息序列中较早的 `assistant(tool_calls) + 配套 tool` 组整组折叠成
//! 单条 `internal_note` stub，逐字保留最近的工具组。

use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;
use std::path::Path;

use crate::ai::types::ToolCall;

use super::super::types::{Message, ROLE_INTERNAL_NOTE, is_system_like_role, retained_turn_start};
use super::text_utils::truncate_to_chars;
use super::tool_overflow::{
    build_tool_overflow_recall_lines, is_non_compressible_tool, is_preserved_user_or_image_stub,
    preserve_noncompressible_tool_result_for_fold,
};
use super::{
    COMPRESSED_TOOL_EVIDENCE_MARKER, is_context_checkpoint_marker,
    keep_recent_user_turns_when_trimming, normalize_whitespace, value_to_string,
};

pub(super) fn first_tool_call_group(messages: &[Message]) -> Option<Vec<usize>> {
    let assistant_idx = messages.iter().position(|m| {
        if m.role != "assistant" {
            return false;
        }
        let Some(tool_calls) = &m.tool_calls else {
            return false;
        };
        if tool_calls.is_empty() {
            return false;
        }
        !tool_calls
            .iter()
            .any(|tc| is_non_compressible_tool(&tc.function.name))
    })?;
    let tool_call_ids: Vec<String> = messages[assistant_idx]
        .tool_calls
        .as_ref()
        .unwrap()
        .iter()
        .map(|tc| tc.id.clone())
        .collect();
    let mut group = vec![assistant_idx];
    // 仅收集紧跟 assistant 之后、连续出现的 tool 消息，遇到任何非 tool（含
    // 下一个 assistant/user/system/internal_note）即停。这样可以防止把
    // 不同 turn 中残留的同 id stub（来自 dedup_repeated_tool_results 替换）
    // 一并卷入同一 group，导致跨 turn 整组被折叠/删除。
    let mut i = assistant_idx + 1;
    while i < messages.len() && messages[i].role == "tool" {
        if let Some(ref id) = messages[i].tool_call_id {
            if tool_call_ids.contains(id) {
                group.push(i);
            } else {
                // 同位置出现了不属于本 assistant 的 tool 消息（理论上不应该
                // 发生，但若发生，停止扫描以避免破坏 OpenAI 配对协议）。
                break;
            }
        } else {
            break;
        }
        i += 1;
    }
    Some(group)
}

pub(super) fn first_trim_candidate(messages: &[Message], budget: usize) -> Option<usize> {
    let keep_recent_user_turns = keep_recent_user_turns_when_trimming(messages, budget);
    let protected_tail_start = retained_turn_start(messages, keep_recent_user_turns);

    // 跳过头部所有 system-like（system / internal_note）消息：它们承载 agent
    // 指令、工具列表、历史摘要等关键上下文，不能被裁掉。
    // 旧实现只跳过以"对话摘要/历史摘要"前缀开头的条目，会把普通 system prompt
    // 当成可裁削目标，触发"上下文压缩后回复戛然而止"。
    // 同时把最近 N 轮 user 起始的整段尾部窗口都设为保护区，避免把多阶段任务
    // 的上一个子目标与当前子目标切开。
    let mut index = 0usize;
    while index < messages.len() && is_system_like_role(&messages[index].role) {
        index += 1;
    }

    while index < protected_tail_start {
        let message = &messages[index];

        // checkpoint 的正文已落盘，短标记是回读正文的唯一入口。它不应因位于
        // 早期对话中而被兜底裁剪掉。
        if is_context_checkpoint_marker(message) {
            index += 1;
            continue;
        }

        // 已外溢到会话文件的 user/image 占位 stub 不删：它只是一个指向归档
        // 文件的指针（原文零压缩保存在磁盘上），删掉会让模型彻底失去线索。
        // 普通的 user / 含图片消息仍可被裁剪：大体量内容已由 proactive spill
        // 提前搬到文件，剩下的小体量旧 user 轮次应进入「丢弃 + 摘要」路径，
        // 否则 30+ 轮纯文本对话永远无法收敛进预算，且早期目标会被静默丢失。
        if is_preserved_user_or_image_stub(&value_to_string(&message.content)) {
            index += 1;
            continue;
        }

        // tool 不单删：否则可能破坏 assistant(tool_calls) ↔ tool 的配对关系。
        if message.role == "tool" {
            index += 1;
            continue;
        }

        // 带 tool_calls 的 assistant 不单删：保持协议配对一致性。
        if message.role == "assistant"
            && message
                .tool_calls
                .as_ref()
                .map(|calls| !calls.is_empty())
                .unwrap_or(false)
        {
            index += 1;
            continue;
        }

        return Some(index);
    }

    None
}

/// 渐进式卸载：把一个 (assistant tool_calls + 配套 tool 结果) 整组折叠成单条
/// `internal_note`。不可有损压缩的结果先写入会话 asset，再在 stub 中保留回读路径；
/// 普通结果保留压缩后的关键结论，避免后续轮次重复劳动。
pub(super) fn fold_tool_call_group_to_stub(
    messages: &[Message],
    group: &[usize],
    overflow_dir: Option<&Path>,
) -> Option<Message> {
    if group.is_empty() {
        return None;
    }
    let assistant_idx = group[0];
    let assistant = messages.get(assistant_idx)?;
    let tool_calls = assistant.tool_calls.as_ref()?;
    if tool_calls.is_empty() {
        return None;
    }

    let mut lines = Vec::with_capacity(tool_calls.len() + 6);
    lines.push(format!(
        "compressed_tool_round: {} tool calls (folded for context budget)",
        tool_calls.len()
    ));
    lines.push(COMPRESSED_TOOL_EVIDENCE_MARKER.to_string());

    // checkpoint 优先取 assistant 正文；但 tool-call 轮的 content 往往为空
    // （模型只发 tool_calls、无叙述）。此时回退到 reasoning_content——那是模型
    // 发起这批调用前的思考，作为压缩后的决策锚点远比固定的 <empty> 有价值，
    // 能避免模型在压缩后失忆、从同一轮重启取证。
    let assistant_content = value_to_string(&assistant.content);
    let assistant_text = normalize_whitespace(assistant_content.trim());
    let checkpoint = if !assistant_text.is_empty() {
        assistant_text
    } else {
        assistant
            .reasoning_content
            .as_deref()
            .map(|reasoning| normalize_whitespace(reasoning.trim()))
            .filter(|reasoning| !reasoning.is_empty())
            .unwrap_or_default()
    };
    if checkpoint.is_empty() {
        lines.push(
            "assistant_checkpoint: <empty; no persisted decision before these tool calls>"
                .to_string(),
        );
    } else {
        lines.push(format!(
            "assistant_checkpoint: {}",
            truncate_to_chars(&checkpoint, 720)
        ));
    }
    lines.push("evidence:".to_string());

    for tc in tool_calls.iter().take(8) {
        let result_text = group
            .iter()
            .skip(1)
            .find_map(|idx| {
                let m = messages.get(*idx)?;
                if m.tool_call_id.as_deref() == Some(tc.id.as_str()) {
                    Some(value_to_string(&m.content))
                } else {
                    None
                }
            })
            .unwrap_or_default();
        let recall = tool_result_recall_text(tc, &result_text, overflow_dir)?;
        let invocation = tool_call_invocation_recall(tc);
        let target = tool_call_target_recall(tc);
        lines.push(format!(
            "- {}{}{} => {}",
            tc.function.name, target, invocation, recall
        ));
    }
    if tool_calls.len() > 8 {
        lines.push(format!(
            "- ... ({} more tools omitted)",
            tool_calls.len() - 8
        ));
    }
    if tool_calls
        .iter()
        .any(|tool_call| is_non_compressible_tool(&tool_call.function.name))
    {
        lines.push("compression_decision: reuse the evidence above before repeating the same read/search/list/command action; only re-run or re-read if exact omitted text is required or the underlying target changed.".to_string());
    }

    Some(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(lines.join("\n")),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    })
}

fn parsed_tool_args(tool_call: &ToolCall) -> Option<Value> {
    serde_json::from_str::<Value>(&tool_call.function.arguments).ok()
}

fn arg_string(args: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| args.get(*key).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn arg_u64(args: &Value, key: &str) -> Option<u64> {
    args.get(key)
        .and_then(|value| value.as_u64().or_else(|| value.as_str()?.parse().ok()))
}

fn tool_call_target_recall(tool_call: &ToolCall) -> String {
    let Some(args) = parsed_tool_args(tool_call) else {
        return String::new();
    };
    let mut fields = Vec::new();
    match tool_call.function.name.as_str() {
        "read_file" | "read_file_lines" => {
            if let Some(path) = arg_string(&args, &["file_path", "path", "filePath"]) {
                fields.push(format!(
                    "file: {}",
                    truncate_to_chars(&normalize_whitespace(&path), 240)
                ));
            }
            if let Some(offset) = arg_u64(&args, "offset") {
                if let Some(limit) = arg_u64(&args, "limit") {
                    fields.push(format!(
                        "range: lines={}..{}",
                        offset,
                        offset.saturating_add(limit.saturating_sub(1))
                    ));
                } else {
                    fields.push(format!("range: offset={offset}"));
                }
            } else if let Some(limit) = arg_u64(&args, "limit") {
                fields.push(format!("range: first {limit} lines"));
            }
        }
        "find_path" => {
            if let Some(pattern) = arg_string(&args, &["pattern", "query"]) {
                fields.push(format!(
                    "pattern: {}",
                    truncate_to_chars(&normalize_whitespace(&pattern), 240)
                ));
            }
            if let Some(path) = arg_string(&args, &["path"]) {
                fields.push(format!(
                    "path: {}",
                    truncate_to_chars(&normalize_whitespace(&path), 160)
                ));
            }
        }
        "code_search" => {
            if let Some(operation) = arg_string(&args, &["operation"]) {
                fields.push(format!(
                    "operation: {}",
                    truncate_to_chars(&normalize_whitespace(&operation), 80)
                ));
            }
            if let Some(query) = arg_string(&args, &["query"]) {
                fields.push(format!(
                    "query: {}",
                    truncate_to_chars(&normalize_whitespace(&query), 240)
                ));
            }
            if let Some(path) = arg_string(&args, &["path"]) {
                fields.push(format!(
                    "path: {}",
                    truncate_to_chars(&normalize_whitespace(&path), 160)
                ));
            }
            if let Some(file_pattern) = arg_string(&args, &["file_pattern", "filePattern"]) {
                fields.push(format!(
                    "file_pattern: {}",
                    truncate_to_chars(&normalize_whitespace(&file_pattern), 160)
                ));
            }
        }
        "list_directory" => {
            if let Some(path) = arg_string(&args, &["path"]) {
                fields.push(format!(
                    "path: {}",
                    truncate_to_chars(&normalize_whitespace(&path), 160)
                ));
            }
        }
        "write_file" | "create_file" | "edit_file" | "delete_path" => {
            if let Some(path) = arg_string(&args, &["file_path", "path", "filePath"]) {
                fields.push(format!(
                    "file: {}",
                    truncate_to_chars(&normalize_whitespace(&path), 240)
                ));
            }
        }
        _ => {}
    }

    if fields.is_empty() {
        String::new()
    } else {
        format!(" [{}]", fields.join("; "))
    }
}

/// `execute_command` 的结果本身无法说明它回答的是哪个问题。工具组折叠会移除
/// 原始 tool_call，因此必须把命令和 cwd 留在召回文本中；否则多个「成功但无输出」
/// 的 git log 会退化成无法区分的记录，模型会把它当作尚未执行而重新开始排查。
fn tool_call_invocation_recall(tool_call: &ToolCall) -> String {
    if tool_call.function.name != "execute_command" {
        return String::new();
    }

    let args = serde_json::from_str::<Value>(&tool_call.function.arguments).ok();
    let command = args
        .as_ref()
        .and_then(|args| args.get("command"))
        .and_then(Value::as_str)
        .map(|command| truncate_to_chars(&normalize_whitespace(command), 720));
    let cwd = args
        .as_ref()
        .and_then(|args| args.get("cwd"))
        .and_then(Value::as_str)
        .filter(|cwd| !cwd.trim().is_empty())
        .map(|cwd| truncate_to_chars(&normalize_whitespace(cwd), 240));

    let mut fields = Vec::with_capacity(2);
    if let Some(command) = command.filter(|command| !command.is_empty()) {
        fields.push(format!("command: {command}"));
    }
    if let Some(cwd) = cwd.filter(|cwd| !cwd.is_empty()) {
        fields.push(format!("cwd: {cwd}"));
    }
    if fields.is_empty() {
        let args = truncate_to_chars(&normalize_whitespace(&tool_call.function.arguments), 240);
        if !args.is_empty() {
            fields.push(format!("arguments: {args}"));
        }
    }

    if fields.is_empty() {
        String::new()
    } else {
        format!(" [{}]", fields.join("; "))
    }
}

/// 为工具组折叠生成结果召回文本。高精度结果在移除原始消息前必须先归档；若归档
/// 失败则返回 `None`，调用方保留整组原文，不能将唯一证据降级为首句。
fn tool_result_recall_text(
    tool_call: &ToolCall,
    result_text: &str,
    overflow_dir: Option<&Path>,
) -> Option<String> {
    let tool_name = tool_call.function.name.as_str();
    if !is_non_compressible_tool(tool_name) || result_text.trim().is_empty() {
        return Some(tool_result_recall_one_liner(result_text));
    }

    let already_archived = result_text
        .lines()
        .map(str::trim)
        .any(|line| line.starts_with("- file_path:") || line.starts_with("file_path:"));
    let preserved =
        preserve_noncompressible_tool_result_for_fold(overflow_dir, tool_name, result_text)?;
    let path = preserved
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("- file_path:") || line.starts_with("file_path:"))?;

    if tool_name == "execute_command" && !already_archived {
        let recall = command_result_recall(result_text);
        let recall_lower = recall.to_ascii_lowercase();
        let has_error_signal = recall_lower.contains("error")
            || recall_lower.contains("failed")
            || recall_lower.contains("panic")
            || recall_lower.contains("blocked")
            || recall_lower.contains("aborting")
            || recall_lower.contains("could not compile");
        if has_error_signal {
            return Some(append_original_recall_lines(
                format!(
                    "{}\n  {path}\n  - 命令输出包含错误信号；如需完整诊断日志，可用 read_file 读取以上文件。",
                    recall,
                ),
                tool_call,
                &preserved,
            ));
        }
        return Some(append_original_recall_lines(
            format!("{}\n  {path}", recall),
            tool_call,
            &preserved,
        ));
    }

    Some(append_original_recall_lines(
        truncate_to_chars(&normalize_whitespace(path), 240),
        tool_call,
        &preserved,
    ))
}

/// 工具组折叠是第二级压缩：不能把一级 stub 中的 `original_*` 调用锚点再次丢掉。
///
/// 优先从仍在场的 ToolCall 参数重建；旧格式或参数解析失败时，再保留已有 stub 中的
/// 锚点。这样历史只剩内部归档路径时，模型仍知道原始文件、命令或检索是什么。
fn append_original_recall_lines(
    mut recall: String,
    tool_call: &ToolCall,
    preserved: &str,
) -> String {
    let mut seen = FxHashSet::default();
    let from_call =
        build_tool_overflow_recall_lines(&tool_call.function.name, &tool_call.function.arguments);
    let from_preserved = preserved.lines().filter_map(|line| {
        let line = line.trim();
        line.starts_with("- original_").then_some(line)
    });

    for line in from_call.iter().map(String::as_str).chain(from_preserved) {
        if seen.insert(line.to_string()) {
            recall.push_str("\n  ");
            recall.push_str(line);
        }
    }
    recall
}

/// 命令输出被折叠时至少保留退出状态、关键诊断与尾部结论。完整日志仍由调用方
/// 归档到 `file_path`，这里的职责只是让模型能在不重跑命令的情况下判断下一步。
fn command_result_recall(result_text: &str) -> String {
    const MAX_SIGNALS: usize = 5;
    const MAX_CHARS: usize = 720;

    let lines = result_text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return "command produced no output".to_string();
    }

    let mut signals = Vec::with_capacity(MAX_SIGNALS + 2);
    push_command_signal(&mut signals, lines[0]);
    for line in &lines {
        let lower = line.to_ascii_lowercase();
        let diagnostic = lower.contains("error")
            || lower.contains("failed")
            || lower.contains("panic")
            || lower.contains("test result:")
            || lower.contains("failures:")
            || lower.contains("could not compile")
            || lower.contains("aborting due to");
        if diagnostic {
            push_command_signal(&mut signals, line);
            if signals.len() >= MAX_SIGNALS {
                break;
            }
        }
    }
    if signals.len() < MAX_SIGNALS {
        push_command_signal(&mut signals, lines[lines.len() - 1]);
    }

    truncate_to_chars(&signals.join(" | "), MAX_CHARS)
}

fn push_command_signal(signals: &mut Vec<String>, line: &str) {
    let line = truncate_to_chars(&normalize_whitespace(line), 220);
    if !line.is_empty() && !signals.iter().any(|existing| existing == &line) {
        signals.push(line);
    }
}

/// 单轮内折叠：LLM 摘要兜底时，尾窗（当前 user 轮）内部保留逐字的最近工具组数。
/// 更早的同轮工具组折叠成单行 stub，解决"臃肿全堆在一个 user 轮内、无跨轮边界
/// 可摘要"时压缩器空转的问题。
pub(super) const MID_TURN_LLM_SUMMARY_KEEP_RECENT_TOOL_GROUPS: usize = 4;

/// 返回最近 `keep_recent_groups` 个完整工具组中所有 tool 结果的消息下标。
///
/// 一个 assistant(tool_calls) 可能并行发起任意数量的调用。保护窗口必须按这个
/// 原子组计算，不能按 tool 消息条数切开；否则一个批次会出现部分结果仍在上下文、
/// 部分结果已被 offload/dedup 的状态，迫使模型重新读取缺失文件。
pub(super) fn recent_tool_group_message_indices(
    messages: &[Message],
    keep_recent_groups: usize,
) -> FxHashSet<usize> {
    recent_tool_result_groups(messages, keep_recent_groups)
        .into_iter()
        .flatten()
        .collect()
}

/// 返回最近 `keep_recent_groups` 个完整工具组的 tool 结果下标，并保留组边界。
pub(super) fn recent_tool_result_groups(
    messages: &[Message],
    keep_recent_groups: usize,
) -> Vec<Vec<usize>> {
    if keep_recent_groups == 0 {
        return Vec::new();
    }

    let mut groups = Vec::<Vec<usize>>::new();
    for (anchor, assistant) in messages.iter().enumerate() {
        if assistant.role != "assistant" {
            continue;
        }
        let Some(calls) = assistant
            .tool_calls
            .as_ref()
            .filter(|calls| !calls.is_empty())
        else {
            continue;
        };
        let call_ids: FxHashSet<&str> = calls.iter().map(|call| call.id.as_str()).collect();
        let mut result_indices = Vec::new();
        for (idx, message) in messages.iter().enumerate().skip(anchor + 1) {
            if message.role == "assistant" && message.tool_calls.is_some() {
                break;
            }
            if message.role == "tool"
                && message
                    .tool_call_id
                    .as_deref()
                    .is_some_and(|id| call_ids.contains(id))
            {
                result_indices.push(idx);
            }
        }
        if !result_indices.is_empty() {
            groups.push(result_indices);
        }
    }

    groups.into_iter().rev().take(keep_recent_groups).collect()
}

/// 为折叠 stub 生成"召回锚点"单行：优先提取已外溢 tool 结果里的 `file_path:`
/// 指针（模型据此可重新 read_file），否则退回结果首个非空行。
/// 保证折叠早期 precision 工具组时仍留下可召回的线索，避免失忆式重复检索。
fn tool_result_recall_one_liner(result_text: &str) -> String {
    if let Some(path_line) = result_text
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("- file_path:") || line.starts_with("file_path:"))
    {
        return truncate_to_chars(&normalize_whitespace(path_line), 200);
    }
    let first_meaningful = result_text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("");
    truncate_to_chars(&normalize_whitespace(first_meaningful), 160)
}

/// 从工具调用的 JSON `arguments` 里取 `file_path`（或兼容的 `path`）。
/// `apply_patch` 还可能把目标写在 `patch` 正文 `*** Update File: <path>` /
/// `*** Add File: <path>` / `*** Replace in line: <path>` 信封里，做兜底解析。
fn extract_file_path_args(arguments: &str) -> Vec<String> {
    let Ok(v) = serde_json::from_str::<Value>(arguments) else {
        return Vec::new();
    };
    if let Some(p) = v.get("file_path").and_then(|x| x.as_str()) {
        return vec![p.to_string()];
    }
    if let Some(p) = v.get("path").and_then(|x| x.as_str()) {
        return vec![p.to_string()];
    }
    if let Some(patch) = v.get("patch").and_then(|x| x.as_str()) {
        return patch
            .lines()
            .filter_map(|line| {
                let trimmed = line.trim_start();
                trimmed
                    .strip_prefix("*** Update File:")
                    .or_else(|| trimmed.strip_prefix("*** Add File:"))
                    .or_else(|| trimmed.strip_prefix("*** Replace in line:"))
                    .map(|rest| rest.trim().to_string())
            })
            .collect();
    }
    Vec::new()
}

/// 扫描整条消息流，找出"最近一次 `apply_patch` 失败、且之后同路径没有成功"
/// 的目标文件路径集合。
///
/// 折叠器遇到 `read_file` 这些路径的工具组时跳过折叠，让模型在重试 patch 时
/// 仍能看到原始文件内容以构造精确 context。判定：按消息顺序记录每个路径最近
/// 一次 `apply_patch` 的结果；结果 content 以 `Successfully patched` 开头视为成功，
/// 否则视为失败。只保留最终状态为"失败"的路径——一旦后续同路径 patch 成功即
/// 自动解除。当失败的 `apply_patch` 调用本身被压缩移出历史后，该路径也随之
/// 消失，保护范围天然有界（通常 1–3 个文件）。
fn collect_pending_patch_paths(messages: &[Message]) -> FxHashSet<String> {
    // tool_call_id → tool 结果文本
    let mut result_by_id: FxHashMap<String, String> = FxHashMap::default();
    for m in messages {
        if m.role == "tool" {
            if let Some(id) = m.tool_call_id.clone() {
                result_by_id.insert(id, value_to_string(&m.content));
            }
        }
    }
    // path → 最近一次 apply_patch 是否成功
    let mut last_state: FxHashMap<String, bool> = FxHashMap::default();
    for m in messages {
        if m.role != "assistant" {
            continue;
        }
        let Some(tcs) = m.tool_calls.as_ref() else {
            continue;
        };
        for tc in tcs {
            if tc.function.name != "apply_patch" {
                continue;
            }
            let paths = extract_file_path_args(&tc.function.arguments);
            if paths.is_empty() {
                continue;
            }
            let result = result_by_id.get(&tc.id).map(String::as_str).unwrap_or("");
            let succeeded = result.trim_start().starts_with("Successfully patched");
            for path in paths {
                last_state.insert(path, succeeded);
            }
        }
    }
    last_state
        .into_iter()
        .filter(|(_, ok)| !*ok)
        .map(|(p, _)| p)
        .collect()
}

/// 本工具组是否含 `read_file` / `read_file_lines` 调用，且其 `file_path` 正是某个尚未成功的
/// `apply_patch` 目标。若是，折叠器应跳过折叠以保留模型重试 patch 所需的
/// 原始文件内容。
fn group_reads_pending_patch_target(
    messages: &[Message],
    group: &[usize],
    pending_paths: &FxHashSet<String>,
) -> bool {
    let Some(assistant) = messages.get(group[0]) else {
        return false;
    };
    let Some(tcs) = assistant.tool_calls.as_ref() else {
        return false;
    };
    tcs.iter().any(|tc| {
        matches!(tc.function.name.as_str(), "read_file" | "read_file_lines")
            && extract_file_path_args(&tc.function.arguments)
                .into_iter()
                .any(|p| pending_paths.contains(&p))
    })
}

/// 把消息序列中较早的 `assistant(tool_calls) + 配套 tool` 组整组折叠成单条
/// `internal_note` stub，逐字保留最近 `keep_recent_groups` 个工具组以及所有
/// 非工具组消息（user / system / internal_note / 纯文本 assistant）。
///
/// 关键不变量：折叠时把 assistant 及其全部 tool 响应**整组一起**替换成一条
/// stub，因此不会出现"留下 assistant.tool_calls 却丢掉配套 tool 响应"的配对
/// 断裂（OpenAI 协议要求二者成对）。含 `is_non_compressible_tool` 的组也会被
/// 折叠，但 stub 内保留其 `file_path:` 召回锚点。
///
/// 返回 `(folded_messages, folded_group_count)`。
pub(super) fn fold_early_tool_groups(
    messages: &[Message],
    keep_recent_groups: usize,
    overflow_dir: Option<&Path>,
) -> (Vec<Message>, usize) {
    // 定位所有 assistant(tool_calls) 起始位置，作为工具组锚点。
    let group_anchors: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(idx, m)| {
            let has_calls = m
                .tool_calls
                .as_ref()
                .map(|calls| !calls.is_empty())
                .unwrap_or(false);
            (m.role == "assistant" && has_calls).then_some(idx)
        })
        .collect();
    if group_anchors.len() <= keep_recent_groups {
        return (messages.to_vec(), 0);
    }
    // 最近 keep_recent_groups 个工具组逐字保留；更早的折叠。
    // keep_recent_groups=0 时折叠全部工具组（fold_before_anchor 取消息末尾），
    // 避免 group_anchors[len - 0] 越界 panic。
    let fold_before_anchor = if keep_recent_groups == 0 {
        messages.len()
    } else {
        group_anchors[group_anchors.len() - keep_recent_groups]
    };

    // 方向二：pending-patch 定向保留。扫描历史中"最近一次 apply_patch 失败、
    // 且之后同路径没有成功"的目标文件路径；折叠器遇到读这些路径的 read_file
    // 组时跳过折叠，让模型在重试 patch 时仍能看到原始文件内容以构造精确
    // context，避免"内容被 offload → patch context mismatch → 重读 → 被判
    // 无进展 → 硬停"的死结。保护范围天然有界：patch 成功或失败的 apply_patch
    // 调用本身被压缩移出历史后，该路径自动解除。
    let pending_patch_paths = collect_pending_patch_paths(messages);

    let mut out: Vec<Message> = Vec::with_capacity(messages.len());
    let mut folded_groups = 0usize;
    let mut idx = 0usize;
    while idx < messages.len() {
        let message = &messages[idx];
        let has_calls = message
            .tool_calls
            .as_ref()
            .map(|calls| !calls.is_empty())
            .unwrap_or(false);
        // 只有位于折叠区（早于最近保留窗口）的 assistant(tool_calls) 组才折叠。
        if idx < fold_before_anchor && message.role == "assistant" && has_calls {
            let tool_call_ids: FxHashSet<&str> = message
                .tool_calls
                .as_ref()
                .unwrap()
                .iter()
                .map(|tc| tc.id.as_str())
                .collect();
            // 收集紧随其后、属于本 assistant 的连续 tool 响应，构成完整组。
            let mut group = vec![idx];
            let mut cursor = idx + 1;
            while cursor < messages.len() && messages[cursor].role == "tool" {
                match messages[cursor].tool_call_id.as_deref() {
                    Some(id) if tool_call_ids.contains(id) => group.push(cursor),
                    _ => break,
                }
                cursor += 1;
            }
            if let Some(stub) = fold_tool_call_group_to_stub(messages, &group, overflow_dir) {
                // pending-patch 目标路径的 read_file 组跳过折叠，逐字保留。
                if group_reads_pending_patch_target(messages, &group, &pending_patch_paths) {
                    for &gi in &group {
                        out.push(messages[gi].clone());
                    }
                    idx = cursor;
                    continue;
                }
                out.push(stub);
                folded_groups += 1;
                idx = cursor;
                continue;
            }
        }
        out.push(message.clone());
        idx += 1;
    }
    (out, folded_groups)
}
