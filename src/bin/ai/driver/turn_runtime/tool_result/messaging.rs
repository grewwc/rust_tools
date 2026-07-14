use std::borrow::Cow;

use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;

use crate::ai::{
    code_discovery_policy::{
        CodeDiscoveryRecord, classify_finding, confidence_label, kind_label, persistence_limit,
        priority_for_confidence, render_record, should_persist,
    },
    driver::tools::ExecuteToolCallsResult,
    history::{Message, ROLE_INTERNAL_NOTE, is_system_like_role},
    types::App,
    types::ToolCall,
};

use super::super::types::PreparedToolResult;
use super::execution::prepare_recent_tool_result;

const CODE_INSPECTION_MEMORY_PREFIX: &str = "Current code-inspection working memory:";
const CODE_DISCOVERY_PREFIX: &str = "code_discovery:";
const CODE_DISCOVERY_CATEGORY: &str = "code_discovery";

#[derive(Debug, Clone)]
struct RepoInspectionFinding {
    tool_name: String,
    rendered: String,
    highlight: String,
}

pub(super) fn append_message_pair(
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    message: Message,
) {
    messages.push(message.clone());
    turn_messages.push(message);
}

pub(super) fn record_hidden_self_note(
    app: &App,
    turn_messages: &mut Vec<Message>,
    hidden_meta: &str,
) {
    let hidden_meta = hidden_meta.trim();
    if hidden_meta.is_empty() {
        return;
    }

    let record = Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(format!("self_note:\n{hidden_meta}")),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    };
    turn_messages.push(record);

    let entry = crate::ai::tools::storage::memory_store::AgentMemoryEntry {
        id: None,
        timestamp: chrono::Local::now().to_rfc3339(),
        category: "self_note".to_string(),
        note: hidden_meta.to_string(),
        tags: vec!["agent".to_string(), "policy".to_string()],
        source: Some(format!("session:{}", app.session_id)),
        priority: Some(255),
        owner_pid: None,
        owner_pgid: None,
        image_path: None,
    };
    let store = crate::ai::tools::storage::memory_store::MemoryStore::from_env_or_config();
    let _ = store.append(&entry);
    store.maintain_after_append();
}

pub(super) fn parse_prune_meta_and_update_marks(
    app: &mut App,
    messages: &[Message],
    hidden_meta: &str,
) -> String {
    let (prune_ids, remaining_meta) =
        crate::ai::history::compress::llm_prune::parse_prune_from_hidden_meta(hidden_meta);
    let active_tool_ids =
        crate::ai::history::compress::llm_prune::active_prunable_tool_ids(messages);
    crate::ai::history::compress::llm_prune::update_prune_marks(
        &mut app.prune_marks,
        &prune_ids,
        &active_tool_ids,
    );
    remaining_meta
}

pub(super) fn append_cached_tool_results_note(
    exec_result: &ExecuteToolCallsResult,
    messages: &mut Vec<Message>,
    _turn_messages: &mut Vec<Message>,
) {
    if !exec_result.cached_hits.iter().any(|hit| *hit) {
        return;
    }

    let cached_names = exec_result
        .executed_tool_calls
        .iter()
        .zip(exec_result.cached_hits.iter())
        .filter_map(|(tool_call, cached)| cached.then_some(tool_call.function.name.as_str()))
        .collect::<Vec<_>>()
        .join(", ");
    let cache_note = Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(format!(
            "Context note: reused cached tool results from the current session for identical calls within the recent TTL. Treat these results as already verified context unless the user asks to refresh. Tools: {cached_names}"
        )),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    };
    // 过程性提示：仅在本 turn 内下发给 LLM，不写入 turn_messages 持久化轨道。
    // 缓存命中是"仅当前回合有效"的上下文，若持久化会在后续每个 turn 反复重放，
    // 且没有跨 turn 去重，属于单调膨胀的历史污染。
    messages.push(cache_note);
}

pub(super) fn print_tool_result_preview(_tool_name: &str, prepared: &PreparedToolResult) {
    // 终端不再打印工具输出内容，只保留工具调用状态行。
    let _ = prepared;
}

/// 智能截断到最近一处句子边界（中英文兼容），并附 `…[truncated: N chars omitted]`
/// 显式标记，避免后续 agent 误以为这是完整 narration。
/// 在目标 cap 附近的 [cap*0.6, cap] 区间里找句号/换行；如果找不到则退化为按字符 cap 切。
fn smart_truncate_to_sentence(text: &str, cap_chars: usize) -> String {
    let total = text.chars().count();
    if total <= cap_chars {
        return text.to_string();
    }
    // 句子边界候选字符（中英文都覆盖）
    const BOUNDARY_CHARS: &[char] = &['。', '！', '？', '\n', '.', '!', '?'];
    // 在 [cap*0.6, cap] 区间里找最靠右的边界——窗口宽到 40% 才能在
    // 较短 narration 里大概率命中至少一个 sentence break。
    let lower = cap_chars * 6 / 10;
    let mut last_boundary: Option<usize> = None;
    let chars: Vec<char> = text.chars().take(cap_chars).collect();
    for (i, ch) in chars.iter().enumerate() {
        if i >= lower && BOUNDARY_CHARS.contains(ch) {
            last_boundary = Some(i + 1); // 含边界字符本身
        }
    }
    let cut = last_boundary.unwrap_or(cap_chars);
    let mut out: String = text.chars().take(cut).collect();
    let omitted = total - cut;
    out.push_str(&format!("…[truncated: {omitted} chars omitted]"));
    out
}

pub(super) fn append_tool_result_messages(
    app: &mut App,
    stream_assistant_text: &str,
    stream_reasoning_text: &str,
    exec_result: &ExecuteToolCallsResult,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
) {
    // 截断 tool-call 前的 assistant narration：纯叙述对后续轮次价值有限，
    // 多轮叠加后会大幅膨胀上下文，对最终回答几乎没有帮助。
    // 智能截断：优先回退到最近一处句子边界（。！？.!?\n），保证不腰斩半句话；
    // 仍然把 cap 控制在 800 字附近，且加 `…[truncated]` 显式标记，
    // 避免后续 agent 误以为这是完整 narration。
    const TOOL_CALL_NARRATION_MAX_CHARS: usize = 800;
    let narration = if stream_assistant_text.chars().count() > TOOL_CALL_NARRATION_MAX_CHARS {
        smart_truncate_to_sentence(stream_assistant_text, TOOL_CALL_NARRATION_MAX_CHARS)
    } else {
        stream_assistant_text.to_string()
    };
    let assistant_msg = Message {
        role: "assistant".to_string(),
        content: Value::String(narration),
        tool_calls: Some(exec_result.executed_tool_calls.clone()),
        tool_call_id: None,
        reasoning_content: (!stream_reasoning_text.is_empty())
            .then(|| stream_reasoning_text.to_string()),
    };
    append_message_pair(messages, turn_messages, assistant_msg);

    for (tool_call, result) in exec_result
        .executed_tool_calls
        .iter()
        .zip(exec_result.tool_results.iter())
    {
        let prepared = prepare_recent_tool_result(app, &tool_call.function.name, &result.content);
        for obs in app.observers.iter_mut() {
            if obs.is_poisoned() {
                continue;
            }
            let ctx = crate::ai::driver::observer::ToolResultContext {
                tool_name: tool_call.function.name.clone(),
                result_content: result.content.as_str(),
                success: {
                    let content_lower = result.content.to_lowercase();
                    let is_execution_tool = tool_call.function.name == "execute_command"
                        || tool_call.function.name == "run_command"
                        || tool_call.function.name == "shell"
                        || tool_call.function.name == "bash";
                    if is_execution_tool {
                        !content_lower.contains("error:")
                            && !content_lower.contains("exit code")
                            && !content_lower.contains("command not found")
                            && !content_lower.contains("permission denied")
                    } else {
                        !content_lower.starts_with("error:")
                            && !content_lower.starts_with("failed:")
                    }
                },
            };
            let obs_name = obs.name().to_string();
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                obs.on_tool_result(&ctx);
            }))
            .is_err()
            {
                eprintln!(
                    "[Warning] observer '{}' panicked in on_tool_result; disabling for rest of conversation.",
                    obs_name
                );
                obs.mark_poisoned();
            }
        }
        let tool_message = Message {
            role: "tool".to_string(),
            content: Value::String(prepared.content_for_model),
            tool_calls: None,
            tool_call_id: Some(result.tool_call_id.clone()),
            reasoning_content: None,
        };
        append_message_pair(messages, turn_messages, tool_message);
    }
}

/// 在一次工具调用轮结束后，基于本轮累计的 `turn_messages` 生成两类记账：
/// (1) code-inspection working memory（写进 `messages`，喂给模型）；
/// (2) 持久化的 code discoveries（写进 `messages`/`turn_messages` 并落库）。
/// 两者都依赖对 repo-inspection 工具输出的同一次扫描——这里只扫一次并复用，
/// 避免在长 turn 里对全量 `turn_messages` 做 O(rounds²) 的重复扫描与 content 克隆。
pub(super) fn record_tool_inspection_artifacts(
    app: &App,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    allowed_tool_names: &rust_tools::commonw::FastSet<String>,
) {
    let findings = collect_repo_inspection_findings(turn_messages);
    append_code_inspection_working_memory(messages, turn_messages, &findings, allowed_tool_names);
    record_persistent_code_discoveries(app, messages, turn_messages, &findings);
}

fn append_code_inspection_working_memory(
    messages: &mut Vec<Message>,
    turn_messages: &[Message],
    findings: &[RepoInspectionFinding],
    allowed_tool_names: &rust_tools::commonw::FastSet<String>,
) {
    let Some(note) =
        build_code_inspection_working_memory(turn_messages, findings, allowed_tool_names)
    else {
        return;
    };

    // In-place 替换：working memory 是"当前轮工具调用的总结"，
    // 每轮都会重新生成。把旧的同前缀 note 原地替换，避免多轮叠加堆出 N 条
    // 大体相同、但都被持久化的 internal_note。
    let mut found_prior = false;
    for message in messages.iter_mut() {
        if !is_system_like_role(&message.role) {
            continue;
        }
        if let Value::String(content) = &message.content {
            if content.starts_with(CODE_INSPECTION_MEMORY_PREFIX) {
                // 完全相同则什么都不做（保持 prompt cache 命中）
                if content == &note {
                    return;
                }
                message.content = Value::String(note.clone());
                found_prior = true;
                break;
            }
        }
    }
    if found_prior {
        return;
    }

    messages.push(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(note),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

fn record_persistent_code_discoveries(
    app: &App,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    findings: &[RepoInspectionFinding],
) {
    let discoveries = build_persistent_code_discoveries(findings);
    if discoveries.is_empty() {
        return;
    }

    let body = discoveries
        .iter()
        .map(render_record)
        .collect::<Vec<_>>()
        .join("\n");

    // 渐进式合并：把新 discoveries 合并到已有的 code_discovery internal_note，
    // 避免多轮 push 出 N 条 code_discovery 记录堆在 messages 里。
    // 行级去重保留首次出现，新发现追加到尾部。
    let merged_into_existing = merge_into_existing_code_discovery(messages, &body)
        || merge_into_existing_code_discovery(turn_messages, &body);
    if !merged_into_existing {
        let record = Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(format!("{CODE_DISCOVERY_PREFIX}\n{body}")),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        };
        append_message_pair(messages, turn_messages, record);
    }

    let store = crate::ai::tools::storage::memory_store::MemoryStore::from_env_or_config();
    let session_source = format!("session:{}", app.session_id);
    for discovery in &discoveries {
        let entry = crate::ai::tools::storage::memory_store::AgentMemoryEntry {
            id: None,
            timestamp: chrono::Local::now().to_rfc3339(),
            category: CODE_DISCOVERY_CATEGORY.to_string(),
            note: discovery.finding.clone(),
            tags: vec![
                "code".to_string(),
                "debug".to_string(),
                "session".to_string(),
                format!("kind:{}", kind_label(discovery.kind)),
                format!("confidence:{}", confidence_label(discovery.confidence)),
            ],
            source: Some(session_source.clone()),
            priority: Some(priority_for_confidence(discovery.confidence)),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        let _ = store.append(&entry);
    }
    store.maintain_after_append();
}

/// 把 new_body 行级追加到已有 code_discovery internal_note 末尾（去重保留首次）。
/// 命中已有 note 时 in-place 替换其内容并返回 true；否则返回 false（调用方走 push 路径）。
fn merge_into_existing_code_discovery(messages: &mut [Message], new_body: &str) -> bool {
    for message in messages.iter_mut().rev() {
        if message.role != ROLE_INTERNAL_NOTE {
            continue;
        }
        let Value::String(content) = &message.content else {
            continue;
        };
        if !content.starts_with(CODE_DISCOVERY_PREFIX) {
            continue;
        }
        let existing_body = content[CODE_DISCOVERY_PREFIX.len()..]
            .trim_start()
            .to_string();
        let mut seen: FxHashSet<String> = existing_body
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        let mut merged = existing_body.clone();
        let mut appended_any = false;
        for line in new_body.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if seen.insert(trimmed.to_string()) {
                if !merged.is_empty() && !merged.ends_with('\n') {
                    merged.push('\n');
                }
                merged.push_str(trimmed);
                appended_any = true;
            }
        }
        if !appended_any {
            // 完全是已记录过的行：什么都不做，返回 true 表示无需新增 message
            return true;
        }
        message.content = Value::String(format!("{CODE_DISCOVERY_PREFIX}\n{merged}"));
        return true;
    }
    false
}

pub(super) fn record_final_stream_response(
    app: &mut App,
    stream_result: crate::ai::types::StreamResult,
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    final_assistant_text: &mut String,
    final_assistant_recorded: &mut bool,
) {
    // 解析模型在 hidden_meta 中的 prune 标记，更新连续裁剪计数表，并剥离
    // prune 行，避免把裁剪协议持久化成普通 self_note。
    let remaining_meta =
        parse_prune_meta_and_update_marks(app, messages, &stream_result.hidden_meta);
    let assistant_msg = Message {
        role: "assistant".to_string(),
        content: Value::String(stream_result.assistant_text.clone()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: (!stream_result.reasoning_text.is_empty())
            .then(|| stream_result.reasoning_text.clone()),
    };
    append_message_pair(messages, turn_messages, assistant_msg);
    *final_assistant_text = stream_result.assistant_text;
    *final_assistant_recorded = true;
    record_hidden_self_note(app, turn_messages, &remaining_meta);
}

fn build_code_inspection_working_memory(
    turn_messages: &[Message],
    findings: &[RepoInspectionFinding],
    allowed_tool_names: &rust_tools::commonw::FastSet<String>,
) -> Option<String> {
    let exact_calls = collect_completed_repo_inspection_calls(turn_messages);

    let mut raw_repo_tool_count = 0usize;
    let mut code_search_count = 0usize;
    let mut write_tool_count = 0usize;
    for message in turn_messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            let tool_name = tool_call.function.name.as_str();
            if !is_repo_inspection_tool(tool_name) {
                continue;
            }
            if is_raw_repo_tool(tool_name) {
                raw_repo_tool_count += 1;
            }
            if is_write_tool(tool_name) {
                write_tool_count += 1;
            }
            if tool_name == "code_search" {
                code_search_count += 1;
            }
        }
    }

    if findings.is_empty()
        && exact_calls.is_empty()
        && raw_repo_tool_count < 2
        && write_tool_count == 0
    {
        return None;
    }

    let mut note = String::from(CODE_INSPECTION_MEMORY_PREFIX);
    note.push('\n');
    if !exact_calls.is_empty() {
        note.push_str("Completed exact tool calls in this turn; do not repeat identical args unless the file/data changed or the previous result was unusable:\n");
        for call in exact_calls.iter().rev().take(8).rev() {
            note.push_str("- ");
            note.push_str(call);
            note.push('\n');
        }
    }
    for finding in findings.iter().rev().take(6).rev() {
        note.push_str(&finding.rendered);
        note.push('\n');
    }
    note.push_str(
        "Treat these findings as already-known context. Avoid re-running the same reads unless you need verification.\n",
    );
    let can_use_code_search = allowed_tool_names.contains("code_search");
    if can_use_code_search && raw_repo_tool_count >= 1 && code_search_count == 0 {
        note.push_str(
            "Code-navigation correction: you have started raw inspection without `code_search`. Before another raw read, use `code_search` first to locate the relevant file/symbol/definition, then read only the specific region you need.\n",
        );
    } else if can_use_code_search && raw_repo_tool_count >= 2 && code_search_count <= 1 {
        note.push_str(
            "Code-navigation correction: too many raw reads/searches. Prefer one `code_search` hop plus one targeted local read instead of another `read_file_lines` or `find_path`.\n",
        );
    }
    Some(truncate_note(&note, 1800))
}

fn collect_completed_repo_inspection_calls(turn_messages: &[Message]) -> Vec<String> {
    let completed_ids = turn_messages
        .iter()
        .filter_map(|message| {
            (message.role == "tool")
                .then(|| message.tool_call_id.clone())
                .flatten()
        })
        .collect::<Vec<_>>();
    if completed_ids.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut seen = FxHashSet::default();
    for message in turn_messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            if !completed_ids.iter().any(|id| id == &tool_call.id) {
                continue;
            }
            let tool_name = tool_call.function.name.as_str();
            if !is_repo_inspection_tool(tool_name) {
                continue;
            }
            let args = normalized_tool_arguments(&tool_call.function.arguments);
            let rendered = format!("{tool_name}({args})");
            if seen.insert(rendered.clone()) {
                out.push(rendered);
            }
        }
    }
    out
}

fn normalized_tool_arguments(raw: &str) -> String {
    serde_json::from_str::<Value>(raw)
        .map(|value| value.to_string())
        .unwrap_or_else(|_| raw.trim().to_string())
}

fn build_persistent_code_discoveries(
    findings: &[RepoInspectionFinding],
) -> Vec<CodeDiscoveryRecord> {
    findings
        .iter()
        .filter_map(|finding| classify_code_discovery(finding))
        .rev()
        .take(persistence_limit())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn collect_repo_inspection_findings(turn_messages: &[Message]) -> Vec<RepoInspectionFinding> {
    let tool_outputs = turn_messages
        .iter()
        .filter_map(|message| {
            message.tool_call_id.as_deref().map(|id| {
                let content = match &message.content {
                    Value::String(content) => Cow::Borrowed(content.as_str()),
                    other => Cow::Owned(other.to_string()),
                };
                (id, content)
            })
        })
        .collect::<FxHashMap<_, _>>();

    let mut findings = Vec::new();
    let mut seen = FxHashSet::default();

    for message in turn_messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            let tool_name = tool_call.function.name.as_str();
            if !is_repo_inspection_tool(tool_name) {
                continue;
            }

            let tool_call_id = tool_call.id.as_str();
            let Some(content) = tool_outputs.get(tool_call_id) else {
                continue;
            };
            let scope = describe_tool_call(tool_call);
            let highlight = summarize_tool_result(tool_name, content);
            if highlight.is_empty() {
                continue;
            }
            let line = format!("- {}{} => {}", tool_name, scope, highlight);
            if seen.insert(line.clone()) {
                findings.push(RepoInspectionFinding {
                    tool_name: tool_name.to_string(),
                    rendered: line,
                    highlight,
                });
            }
        }
    }
    findings
}

fn is_repo_inspection_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "code_search"
            | "read_file"
            // 兼容旧会话历史里残留的 read_file_lines（已并入 read_file）。
            | "read_file_lines"
            | "search_files"
            | "find_path"
            | "list_directory"
            | "apply_patch"
            | "write_file"
    )
}

fn is_raw_repo_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "read_file" | "read_file_lines" | "search_files" | "find_path" | "list_directory"
    )
}

fn is_write_tool(tool_name: &str) -> bool {
    matches!(tool_name, "apply_patch" | "write_file")
}

fn persistent_code_discovery_already_present(messages: &[Message], body: &str) -> bool {
    messages.iter().any(|message| match &message.content {
        Value::String(content) => {
            content.starts_with(CODE_DISCOVERY_PREFIX)
                && content[CODE_DISCOVERY_PREFIX.len()..].trim_start() == body
        }
        _ => false,
    })
}

fn classify_code_discovery(finding: &RepoInspectionFinding) -> Option<CodeDiscoveryRecord> {
    let record = classify_finding(&finding.tool_name, &finding.highlight, &finding.rendered)?;
    should_persist(record.confidence).then_some(record)
}

fn describe_tool_call(tool_call: &ToolCall) -> String {
    let Ok(args) = serde_json::from_str::<Value>(&tool_call.function.arguments) else {
        return String::new();
    };
    match tool_call.function.name.as_str() {
        "code_search" => {
            let operation = args
                .get("operation")
                .and_then(|v| v.as_str())
                .unwrap_or("search");
            let mut parts = vec![format!("operation={operation}")];
            for key in ["query", "symbol", "file_path", "path", "intent"] {
                if let Some(value) = args.get(key).and_then(|v| v.as_str())
                    && !value.is_empty()
                {
                    parts.push(format!("{key}={}", truncate_inline(value, 48)));
                }
            }
            format!("({})", parts.join(", "))
        }
        "read_file" | "read_file_lines" => {
            let path = args
                .get("file_path")
                .or_else(|| args.get("path"))
                .and_then(|v| v.as_str())
                .map(|v| truncate_inline(v, 64))
                .unwrap_or_else(|| "?".to_string());
            let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(1);
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(0);
            if limit > 0 {
                format!(
                    "(file={}, lines={}..{})",
                    path,
                    offset,
                    offset + limit.saturating_sub(1)
                )
            } else {
                format!("(file={path})")
            }
        }
        "find_path" | "search_files" => {
            let query = args
                .get("query")
                .or_else(|| args.get("pattern"))
                .and_then(|v| v.as_str())
                .map(|v| truncate_inline(v, 48))
                .unwrap_or_else(|| "?".to_string());
            format!("(query={query})")
        }
        "list_directory" => args
            .get("path")
            .and_then(|v| v.as_str())
            .map(|path| format!("(path={})", truncate_inline(path, 64)))
            .unwrap_or_default(),
        "apply_patch" => {
            let path = args
                .get("file_path")
                .or_else(|| args.get("path"))
                .and_then(|v| v.as_str())
                .map(|v| truncate_inline(v, 64))
                .unwrap_or_else(|| "?".to_string());
            format!("(file={})", path)
        }
        "write_file" => {
            let path = args
                .get("file_path")
                .or_else(|| args.get("path"))
                .and_then(|v| v.as_str())
                .map(|v| truncate_inline(v, 64))
                .unwrap_or_else(|| "?".to_string());
            format!("(file={})", path)
        }
        _ => String::new(),
    }
}

fn summarize_tool_result(tool_name: &str, content: &str) -> String {
    if is_write_tool(tool_name) {
        let first_line = content.lines().next().unwrap_or("").trim();
        if first_line.is_empty() {
            return "OK".to_string();
        }
        return truncate_inline(first_line, 120);
    }

    let mut lines = content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| !line.starts_with("code_search route="))
        .filter(|line| !line.starts_with("- tail_preview:"))
        .collect::<Vec<_>>();
    if let Some(summary_line) = lines
        .iter()
        .find(|line| line.starts_with("- summary:") || line.starts_with("summary:"))
    {
        return truncate_inline(summary_line, 160);
    }

    if let Some(error_like) = lines.iter().find(|line| {
        let lower = line.to_ascii_lowercase();
        lower.contains("error")
            || lower.contains("failed")
            || lower.contains("panic")
            || lower.contains("missing")
    }) {
        return truncate_inline(error_like, 160);
    }

    if tool_name == "code_search" {
        lines.retain(|line| {
            !line.starts_with("No exact symbol")
                && !line.starts_with("No exact matches")
                && !line.starts_with("No files matched")
        });
    }

    lines
        .into_iter()
        .next()
        .map(|line| truncate_inline(line, 160))
        .unwrap_or_default()
}

fn truncate_inline(value: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (idx, ch) in value.chars().enumerate() {
        if idx >= max_chars {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

fn truncate_note(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    truncate_inline(value, max_chars)
}

#[cfg(test)]
mod tests {
    use super::smart_truncate_to_sentence;
    use super::{
        RepoInspectionFinding, build_code_inspection_working_memory,
        build_persistent_code_discoveries, classify_code_discovery,
        collect_repo_inspection_findings, is_repo_inspection_tool,
        persistent_code_discovery_already_present,
    };
    use crate::ai::code_discovery_policy::{CodeDiscoveryConfidence, CodeDiscoveryKind};
    use crate::ai::history::Message;
    use crate::ai::types::{FunctionCall, ToolCall};
    use serde_json::Value;

    fn tool_call(id: &str, name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    fn available_tool_names(names: &[&str]) -> rust_tools::commonw::FastSet<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    #[test]
    fn repo_inspection_tools_include_code_search() {
        assert!(is_repo_inspection_tool("code_search"));
    }

    #[test]
    fn smart_truncate_falls_back_to_sentence_boundary() {
        // cap=80, lower=80*6/7≈68. 句号位置：22 / 49 / 71 / 后续。
        let text = "Step one finished ok. Step two searched repo. Step three is running. \
             Step four extends well beyond the cap.";
        let out = smart_truncate_to_sentence(text, 80);
        assert!(out.contains("[truncated: "), "expected marker, got: {out}");
        // 应切到位置 71（"is running."）之后，包含 "is running."；
        // 不应留下半句 "Step four"
        assert!(
            out.contains("is running."),
            "expected boundary fallback to '. ', got: {out}"
        );
        assert!(
            !out.contains("Step four"),
            "should not include next sentence, got: {out}"
        );
    }

    #[test]
    fn smart_truncate_falls_back_to_char_cap_when_no_boundary() {
        // 整段没有任何句号/换行：必须退化为按字符 cap 切并加显式标记。
        let text = "x".repeat(200);
        let out = smart_truncate_to_sentence(&text, 50);
        assert!(out.starts_with(&"x".repeat(50)));
        assert!(out.contains("[truncated:"));
    }

    #[test]
    fn smart_truncate_returns_input_when_under_cap() {
        let text = "short text";
        assert_eq!(smart_truncate_to_sentence(text, 100), text);
    }

    #[test]
    fn working_memory_note_includes_findings_and_correction() {
        let turn_messages = vec![
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![
                    tool_call(
                        "1",
                        "read_file_lines",
                        serde_json::json!({"file_path":"src/lib.rs","offset":10,"limit":20}),
                    ),
                    tool_call(
                        "2",
                        "find_path",
                        serde_json::json!({"pattern":"panic!","path":"src"}),
                    ),
                    tool_call(
                        "3",
                        "read_file",
                        serde_json::json!({"file_path":"src/main.rs","offset":1,"limit":40}),
                    ),
                ]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("    10\tfn load_config() {".to_string()),
                tool_calls: None,
                tool_call_id: Some("1".to_string()),
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("src/main.rs:42: panic!(\"boom\")".to_string()),
                tool_calls: None,
                tool_call_id: Some("2".to_string()),
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("     1\tmod main;".to_string()),
                tool_calls: None,
                tool_call_id: Some("3".to_string()),
                reasoning_content: None,
            },
        ];

        let findings = collect_repo_inspection_findings(&turn_messages);
        let note = build_code_inspection_working_memory(
            &turn_messages,
            &findings,
            &available_tool_names(&["code_search", "read_file", "read_file_lines", "find_path"]),
        )
        .expect("note");
        assert!(note.contains("Current code-inspection working memory"));
        assert!(note.contains("Completed exact tool calls in this turn"));
        assert!(note.contains("read_file_lines("));
        assert!(note.contains("\"file_path\":\"src/lib.rs\""));
        assert!(note.contains("\"offset\":10"));
        assert!(note.contains("read_file("));
        assert!(note.contains("\"file_path\":\"src/main.rs\""));
        assert!(note.contains("read_file_lines(file=src/lib.rs, lines=10..29)"));
        assert!(note.contains("find_path(query=panic!)"));
        assert!(note.contains("Code-navigation correction"));
        assert!(
            note.contains("use `code_search` first")
                || note.contains("Before another raw read, use `code_search` first")
        );
    }

    #[test]
    fn working_memory_note_uses_code_search_without_correction_when_present() {
        let turn_messages = vec![
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![tool_call(
                    "1",
                    "code_search",
                    serde_json::json!({"operation":"text_search","query":"load_config"}),
                )]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String(
                    "code_search route=content operation=text_search\nsrc/lib.rs:10: fn load_config() {"
                        .to_string(),
                ),
                tool_calls: None,
                tool_call_id: Some("1".to_string()),
                reasoning_content: None,
            },
        ];

        let findings = collect_repo_inspection_findings(&turn_messages);
        let note = build_code_inspection_working_memory(
            &turn_messages,
            &findings,
            &available_tool_names(&["code_search", "read_file", "find_path"]),
        )
        .expect("note");
        assert!(note.contains("code_search(operation=text_search, query=load_config)"));
        assert!(note.contains("fn load_config()"));
        assert!(!note.contains("Code-navigation correction"));
    }

    #[test]
    fn working_memory_note_hides_code_search_when_unavailable() {
        let turn_messages = vec![
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![
                    tool_call(
                        "1",
                        "read_file_lines",
                        serde_json::json!({"file_path":"src/lib.rs","offset":10,"limit":20}),
                    ),
                    tool_call(
                        "2",
                        "find_path",
                        serde_json::json!({"pattern":"panic!","path":"src"}),
                    ),
                ]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("    10\tfn load_config() {".to_string()),
                tool_calls: None,
                tool_call_id: Some("1".to_string()),
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("src/main.rs:42: panic!(\"boom\")".to_string()),
                tool_calls: None,
                tool_call_id: Some("2".to_string()),
                reasoning_content: None,
            },
        ];

        let findings = collect_repo_inspection_findings(&turn_messages);
        let note = build_code_inspection_working_memory(
            &turn_messages,
            &findings,
            &available_tool_names(&["read_file", "read_file_lines", "find_path"]),
        )
        .expect("note");
        assert!(!note.contains("Code-navigation correction"));
        assert!(!note.contains("`code_search`"));
    }

    #[test]
    fn persistent_code_discoveries_keep_only_high_value_findings() {
        let turn_messages = vec![
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![
                    tool_call(
                        "1",
                        "read_file_lines",
                        serde_json::json!({"file_path":"src/lib.rs","offset":10,"limit":20}),
                    ),
                    tool_call("2", "list_directory", serde_json::json!({"path":"src"})),
                ]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("    10\tfn load_config() {".to_string()),
                tool_calls: None,
                tool_call_id: Some("1".to_string()),
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("main.rs\nlib.rs".to_string()),
                tool_calls: None,
                tool_call_id: Some("2".to_string()),
                reasoning_content: None,
            },
        ];

        let findings = collect_repo_inspection_findings(&turn_messages);
        let discoveries = build_persistent_code_discoveries(&findings);
        assert_eq!(discoveries.len(), 1);
        assert!(discoveries[0].finding.contains("fn load_config()"));
    }

    #[test]
    fn duplicate_persistent_discovery_is_detected() {
        let messages = vec![Message {
            role: "system".to_string(),
            content: Value::String(
                "code_discovery:\n- read_file_lines(file=src/lib.rs, lines=10..29) => fn load_config() {"
                    .to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        assert!(persistent_code_discovery_already_present(
            &messages,
            "- read_file_lines(file=src/lib.rs, lines=10..29) => fn load_config() {"
        ));
    }

    #[test]
    fn classify_code_discovery_marks_root_cause() {
        let finding = RepoInspectionFinding {
            tool_name: "read_file_lines".to_string(),
            rendered:
                "- read_file_lines(file=src/main.rs, lines=40..50) => root cause: config cache is empty due to missing APP_ENV"
                    .to_string(),
            highlight: "root cause: config cache is empty due to missing APP_ENV".to_string(),
        };

        let record = classify_code_discovery(&finding).expect("record");
        assert_eq!(record.kind, CodeDiscoveryKind::RootCause);
        assert_eq!(record.confidence, CodeDiscoveryConfidence::High);
    }

    #[test]
    fn classify_code_discovery_marks_entry_point() {
        let finding = RepoInspectionFinding {
            tool_name: "read_file_lines".to_string(),
            rendered:
                "- read_file_lines(file=src/main.rs, lines=1..20) => fn main() calls app::run() as the entry point"
                    .to_string(),
            highlight: "fn main() calls app::run() as the entry point".to_string(),
        };

        let record = classify_code_discovery(&finding).expect("record");
        assert_eq!(record.kind, CodeDiscoveryKind::EntryPoint);
        assert_eq!(record.confidence, CodeDiscoveryConfidence::High);
    }

    #[test]
    fn classify_code_discovery_marks_call_chain() {
        let finding = RepoInspectionFinding {
            tool_name: "code_search".to_string(),
            rendered:
                "- code_search(operation=structural, intent=find_calls, query=load_config) => call chain: main -> bootstrap -> load_config"
                    .to_string(),
            highlight: "call chain: main -> bootstrap -> load_config".to_string(),
        };

        let record = classify_code_discovery(&finding).expect("record");
        assert_eq!(record.kind, CodeDiscoveryKind::CallChain);
        assert_eq!(record.confidence, CodeDiscoveryConfidence::Medium);
    }
}
