use std::{borrow::Cow, fs, io::Write, path::Path};

use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;

use crate::ai::{
    driver::tools::ExecuteToolCallsResult,
    history::{Message, ROLE_INTERNAL_NOTE, SessionStore, is_system_like_role},
    tools::tool_history_policy,
    types::App,
    types::ToolCall,
};

use super::super::types::PreparedToolResult;
use super::execution::{prepare_recent_tool_result, prepare_tool_result};

const CODE_INSPECTION_MEMORY_PREFIX: &str = "Current code-inspection working memory:";
const CONTEXT_CHECKPOINT_OPEN: &str = "<context_checkpoint>";
const CONTEXT_CHECKPOINT_CLOSE: &str = "</context_checkpoint>";
const CONTEXT_CHECKPOINT_SUMMARY_MAX_CHARS: usize = 240;

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

    let (checkpoints, hidden_meta) = extract_context_checkpoints(hidden_meta);
    for (index, checkpoint) in checkpoints.into_iter().enumerate() {
        let summary = truncate_checkpoint_summary(&checkpoint.summary);
        let marker = match save_context_checkpoint(app, index, &summary, &checkpoint.body) {
            Ok(path) => format!("[context_checkpoint path={}] {}", path.display(), summary),
            Err(error) => {
                eprintln!("failed to save context checkpoint: {error}");
                turn_messages.push(Message {
                    role: ROLE_INTERNAL_NOTE.to_string(),
                    content: Value::String(format!(
                        "self_note:\n[context_checkpoint save_failed] {summary}\n{}",
                        checkpoint.body
                    )),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
                format!(
                    "[context_checkpoint save_failed] {} (正文已作为 self_note 保留)",
                    summary
                )
            }
        };
        turn_messages.push(Message {
            role: ROLE_INTERNAL_NOTE.to_string(),
            content: Value::String(marker),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }

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

#[derive(Debug, PartialEq, Eq)]
struct ContextCheckpoint {
    summary: String,
    body: String,
}

/// 从隐藏元信息中摘出模型主动保存的长期结论。格式刻意保持纯文本，避免模型为
/// 生成 checkpoint 还要学习 JSON 转义规则：首行是 `summary: ...`，其余为正文。
fn extract_context_checkpoints(hidden_meta: &str) -> (Vec<ContextCheckpoint>, String) {
    let mut checkpoints = Vec::new();
    let mut remaining = String::new();
    let mut cursor = hidden_meta;

    while let Some(start) = cursor.find(CONTEXT_CHECKPOINT_OPEN) {
        remaining.push_str(&cursor[..start]);
        let after_open = &cursor[start + CONTEXT_CHECKPOINT_OPEN.len()..];
        let Some(end) = after_open.find(CONTEXT_CHECKPOINT_CLOSE) else {
            // 标签不完整时不要吞掉模型笔记，按普通 self_note 保存。
            remaining.push_str(&cursor[start..]);
            return (checkpoints, remaining);
        };
        let raw = after_open[..end].trim();
        let (summary, body) = raw.split_once('\n').unwrap_or((raw, ""));
        let summary = summary
            .trim()
            .strip_prefix("summary:")
            .unwrap_or(summary.trim())
            .trim();
        let body = body.trim();
        if !summary.is_empty() && !body.is_empty() {
            checkpoints.push(ContextCheckpoint {
                summary: summary.to_string(),
                body: body.to_string(),
            });
        } else {
            // 不完整 checkpoint 同样退化为普通笔记，避免悄悄丢失结论。
            remaining.push_str(
                &cursor[start
                    ..start + CONTEXT_CHECKPOINT_OPEN.len() + end + CONTEXT_CHECKPOINT_CLOSE.len()],
            );
        }
        cursor = &after_open[end + CONTEXT_CHECKPOINT_CLOSE.len()..];
    }
    remaining.push_str(cursor);
    (checkpoints, remaining)
}

fn truncate_checkpoint_summary(summary: &str) -> String {
    let mut out = String::new();
    for ch in summary.chars().filter(|ch| !ch.is_control()) {
        if out.chars().count() == CONTEXT_CHECKPOINT_SUMMARY_MAX_CHARS {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out.trim().to_string()
}

fn save_context_checkpoint(
    app: &App,
    index: usize,
    summary: &str,
    body: &str,
) -> std::io::Result<std::path::PathBuf> {
    let assets_dir = SessionStore::new(&app.session_history_file)
        .session_assets_dir(&app.session_id)
        .join("context-checkpoints");
    save_context_checkpoint_in_dir(&assets_dir, index, summary, body)
}

/// 先把 checkpoint 写入同目录临时文件并 fsync，再原子改名。
/// marker 只会在此函数成功后写入 history，因此模型不会拿到半写入或不存在的正文路径。
fn save_context_checkpoint_in_dir(
    assets_dir: &Path,
    index: usize,
    summary: &str,
    body: &str,
) -> std::io::Result<std::path::PathBuf> {
    fs::create_dir_all(&assets_dir)?;
    let timestamp = chrono::Local::now().format("%Y%m%dT%H%M%S%3f");
    let file_name = format!(
        "context-checkpoint-{timestamp}-{index}-{}.md",
        uuid::Uuid::new_v4()
    );
    let path = assets_dir.join(&file_name);
    let temporary_path = assets_dir.join(format!(".{file_name}.tmp"));
    let contents = format!("# Context checkpoint\n\n摘要：{summary}\n\n---\n\n{body}\n");

    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary_path)?;
        file.write_all(contents.as_bytes())?;
        file.sync_all()?;
        fs::rename(&temporary_path, &path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result?;
    Ok(path)
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

fn prepare_tool_results_for_history(
    app: &App,
    exec_result: &ExecuteToolCallsResult,
) -> Vec<PreparedToolResult> {
    let mut prepared = exec_result
        .executed_tool_calls
        .iter()
        .zip(exec_result.tool_results.iter())
        .map(|(tool_call, result)| {
            prepare_recent_tool_result(app, &tool_call.function.name, &result.content)
        })
        .collect::<Vec<_>>();

    let inline_budget = super::super::max_tool_result_inline_chars(&app.current_model) / 2;
    let mut precision_indices = exec_result
        .executed_tool_calls
        .iter()
        .enumerate()
        .filter_map(|(idx, tool_call)| {
            tool_history_policy(&tool_call.function.name)
                .counts_toward_precision_inline_budget()
                .then_some(idx)
        })
        .collect::<Vec<_>>();

    let mut total_chars = precision_indices
        .iter()
        .map(|idx| prepared[*idx].content_for_model.chars().count())
        .sum::<usize>();

    if total_chars <= inline_budget {
        return prepared;
    }

    precision_indices.sort_unstable_by_key(|idx| {
        std::cmp::Reverse(prepared[*idx].content_for_model.chars().count())
    });

    for idx in precision_indices {
        if total_chars <= inline_budget {
            break;
        }
        let content = &exec_result.tool_results[idx].content;
        let offloaded = prepare_tool_result(
            app,
            &exec_result.executed_tool_calls[idx].function.name,
            content,
        );
        if offloaded.content_for_model == content.as_str() {
            continue;
        }
        let previous_len = prepared[idx].content_for_model.chars().count();
        prepared[idx] = offloaded;
        total_chars = total_chars.saturating_sub(previous_len);
        total_chars = total_chars.saturating_add(prepared[idx].content_for_model.chars().count());
    }

    prepared
}

pub(super) fn append_tool_result_messages(
    app: &mut App,
    stream_assistant_text: &str,
    stream_reasoning_text: &str,
    stream_reasoning_items: &[serde_json::Value],
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
    messages.push(assistant_msg.clone());
    turn_messages
        .push(crate::ai::history::compress::sanitize_message_for_persisted_history(&assistant_msg));

    // 侧信道：把本轮捕获的 reasoning items 挂到「首个 tool_call id」上，供
    // Responses 协议回放时在对应 function_call 前原样 splice。仅内存态、不落盘。
    // 空时不写入——拿不到 encrypted_content 自动退化为不回放，零 regression。
    if !stream_reasoning_items.is_empty() {
        if let Some(first_call_id) = exec_result
            .executed_tool_calls
            .first()
            .map(|call| call.id.clone())
        {
            app.turn_reasoning_items
                .insert(first_call_id, stream_reasoning_items.to_vec());
        }
    }

    let prepared_results = prepare_tool_results_for_history(app, exec_result);
    for ((tool_call, result), prepared) in exec_result
        .executed_tool_calls
        .iter()
        .zip(exec_result.tool_results.iter())
        .zip(prepared_results.into_iter())
    {
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

/// 在一次工具调用轮结束后，基于本轮累计的 `turn_messages` 生成
/// code-inspection working memory，避免在长 turn 里重复扫描工具结果。
pub(super) fn record_tool_inspection_artifacts(
    messages: &mut Vec<Message>,
    turn_messages: &mut Vec<Message>,
    allowed_tool_names: &rust_tools::commonw::FastSet<String>,
) {
    let findings = collect_repo_inspection_findings(turn_messages);
    append_code_inspection_working_memory(messages, turn_messages, &findings, allowed_tool_names);
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
    messages.push(assistant_msg.clone());
    turn_messages
        .push(crate::ai::history::compress::sanitize_message_for_persisted_history(&assistant_msg));
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
            "Code-navigation correction: too many raw reads/searches. Prefer one `code_search` hop plus one targeted local read instead of another `read_file` or `find_path`.\n",
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
            | "find_path"
            | "list_directory"
            | "apply_patch"
            | "write_file"
    )
}

fn is_raw_repo_tool(tool_name: &str) -> bool {
    matches!(tool_name, "read_file" | "find_path" | "list_directory")
}

fn is_write_tool(tool_name: &str) -> bool {
    matches!(tool_name, "apply_patch" | "write_file")
}

fn infer_apply_patch_targets(args: &Value) -> Vec<String> {
    if let Some(target) = args
        .get("file_path")
        .or_else(|| args.get("path"))
        .and_then(|v| v.as_str())
    {
        return vec![target.to_string()];
    }
    args.get("patch")
        .and_then(|v| v.as_str())
        .map(extract_apply_patch_targets_from_patch)
        .unwrap_or_default()
}

fn extract_apply_patch_targets_from_patch(patch: &str) -> Vec<String> {
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
            .map(|path| path.trim().to_string())
        })
        .collect()
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
        "read_file" => {
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
        "find_path" => {
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
            let targets = infer_apply_patch_targets(&args);
            match targets.as_slice() {
                [] => "(file=?)".to_string(),
                [path] => format!("(file={})", truncate_inline(path, 64)),
                _ => format!(
                    "(files={}, targets={})",
                    targets.len(),
                    truncate_inline(&targets.join(", "), 96)
                ),
            }
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
    use super::{
        prepare_tool_results_for_history, RepoInspectionFinding, build_code_inspection_working_memory,
        collect_repo_inspection_findings, describe_tool_call, is_repo_inspection_tool,
    };
    use super::{
        extract_context_checkpoints, save_context_checkpoint_in_dir, smart_truncate_to_sentence,
        truncate_checkpoint_summary,
    };
    use std::sync::{Arc, atomic::AtomicBool};

    use crate::ai::driver::tools::ExecuteToolCallsResult;
    use crate::ai::{
        cli::ParsedCli,
        history::{Message, SessionStore},
        types::{App, AppConfig, FunctionCall, ToolCall, ToolResult},
    };
    use serde_json::Value;
    use std::path::PathBuf;

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

    fn test_app(history_file: PathBuf) -> App {
        let mut app = App {
            cli: ParsedCli::default(),
            config: AppConfig {
                api_key: String::new(),
                base_history_file: history_file.clone(),
                history_file: history_file.clone(),
                endpoint: String::new(),
                vl_default_model: String::new(),
                history_max_chars: 24_000,
                history_keep_last: 256,
                history_summary_max_chars: 4_000,
                intent_model: None,
                agent_route_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/agent_route/agent_route_model.json"),
                skill_match_model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("src/bin/ai/config/skill_match/skill_match_model.json"),
            },
            session_id: "test".to_string(),
            session_history_file: history_file.clone(),
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
            agent_context: None,
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
        };
        let store = SessionStore::new(history_file.as_path());
        store.ensure_root_dir().unwrap();
        app.session_history_file = store.session_history_file(&app.session_id);
        std::fs::write(&app.session_history_file, b"test").unwrap();
        app
    }

    fn exec_result_from_calls_and_contents(calls: &[ToolCall], contents: &[String]) -> ExecuteToolCallsResult {
        ExecuteToolCallsResult {
            executed_tool_calls: calls.to_vec(),
            tool_results: calls
                .iter()
                .zip(contents.iter())
                .map(|(call, content)| ToolResult {
                    tool_call_id: call.id.clone(),
                    content: content.clone(),
                })
                .collect(),
            cached_hits: vec![false; calls.len()],
            had_error: false,
        }
    }

    #[test]
    fn repo_inspection_tools_include_code_search() {
        assert!(is_repo_inspection_tool("code_search"));
    }

    #[test]
    fn describe_tool_call_infers_apply_patch_target_from_envelope() {
        let call = tool_call(
            "1",
            "apply_patch",
            serde_json::json!({
                "patch": "*** Begin Patch\n*** Update File: src/main.rs\n@@\n-old\n+new\n*** End Patch\n"
            }),
        );

        assert_eq!(describe_tool_call(&call), "(file=src/main.rs)");
    }

    #[test]
    fn describe_tool_call_shows_multi_file_apply_patch_targets() {
        let call = tool_call(
            "1",
            "apply_patch",
            serde_json::json!({
                "patch": "*** Begin Patch\n*** Update File: src/a.rs\n@@\n-old_a\n+new_a\n*** Add File: src/b.rs\n+hello\n*** End Patch\n"
            }),
        );

        let described = describe_tool_call(&call);
        assert!(described.contains("files=2"), "described: {described}");
        assert!(described.contains("src/a.rs"), "described: {described}");
        assert!(described.contains("src/b.rs"), "described: {described}");
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
    fn prepare_tool_results_for_history_spills_oversized_precision_batch() {
        let history_file = std::env::temp_dir().join(format!(
            "ai-batch-precision-spill-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file.clone());
        let per_result = "x".repeat(20_000);
        let calls = (0..4)
            .map(|i| {
                tool_call(
                    &format!("rf-{i}"),
                    "read_file",
                    serde_json::json!({
                        "file_path": format!("src/file_{i}.rs"),
                        "offset": 1,
                        "limit": 400
                    }),
                )
            })
            .collect::<Vec<_>>();
        let contents = vec![per_result; 4];
        let exec_result = exec_result_from_calls_and_contents(&calls, &contents);

        let prepared = prepare_tool_results_for_history(&app, &exec_result);
        let spilled = prepared
            .iter()
            .filter(|result| result.content_for_model.contains("Output too large; full result saved"))
            .count();
        assert!(spilled >= 2, "expected oversized precision batch to spill, got: {spilled}");

        let store = crate::ai::history::SessionStore::new(history_file.as_path());
        let _ = store.delete_session(&app.session_id);
    }

    #[test]
    fn prepare_tool_results_for_history_keeps_small_precision_batch_raw() {
        let history_file = std::env::temp_dir().join(format!(
            "ai-batch-precision-raw-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let app = test_app(history_file.clone());
        let calls = (0..2)
            .map(|i| {
                tool_call(
                    &format!("rf-small-{i}"),
                    "read_file",
                    serde_json::json!({
                        "file_path": format!("src/small_{i}.rs"),
                        "offset": 1,
                        "limit": 120
                    }),
                )
            })
            .collect::<Vec<_>>();
        let contents = vec!["x".repeat(6_000), "y".repeat(6_000)];
        let exec_result = exec_result_from_calls_and_contents(&calls, &contents);

        let prepared = prepare_tool_results_for_history(&app, &exec_result);
        assert!(prepared
            .iter()
            .all(|result| !result.content_for_model.contains("Output too large; full result saved")));
        assert_eq!(prepared[0].content_for_model, contents[0]);
        assert_eq!(prepared[1].content_for_model, contents[1]);

        let store = crate::ai::history::SessionStore::new(history_file.as_path());
        let _ = store.delete_session(&app.session_id);
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
                        "read_file",
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
            &available_tool_names(&["code_search", "read_file", "find_path"]),
        )
        .expect("note");
        assert!(note.contains("Current code-inspection working memory"));
        assert!(note.contains("Completed exact tool calls in this turn"));
        assert!(note.contains("read_file("));
        assert!(note.contains("\"file_path\":\"src/lib.rs\""));
        assert!(note.contains("\"offset\":10"));
        assert!(note.contains("read_file("));
        assert!(note.contains("\"file_path\":\"src/main.rs\""));
        assert!(note.contains("read_file(file=src/lib.rs, lines=10..29)"));
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
                        "read_file",
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
            &available_tool_names(&["read_file", "find_path"]),
        )
        .expect("note");
        assert!(!note.contains("Code-navigation correction"));
        assert!(!note.contains("`code_search`"));
    }

    #[test]
    fn context_checkpoint_is_extracted_and_ordinary_note_is_preserved() {
        let input = "before\n<context_checkpoint>\nsummary: 已确认根因\n证据：src/lib.rs:42。\n</context_checkpoint>\nafter";
        let (checkpoints, remainder) = extract_context_checkpoints(input);

        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0].summary, "已确认根因");
        assert_eq!(checkpoints[0].body, "证据：src/lib.rs:42。");
        assert_eq!(remainder, "before\n\nafter");
    }

    #[test]
    fn incomplete_context_checkpoint_remains_a_self_note() {
        let input = "<context_checkpoint>\nsummary: 只有摘要\n</context_checkpoint>";
        let (checkpoints, remainder) = extract_context_checkpoints(input);

        assert!(checkpoints.is_empty());
        assert_eq!(remainder, input);
    }

    #[test]
    fn context_checkpoint_summary_is_short_and_single_line() {
        let summary = format!("abc\ndef{}", "x".repeat(300));
        let short = truncate_checkpoint_summary(&summary);

        assert!(!short.contains('\n'));
        assert!(short.ends_with('…'));
        assert!(short.chars().count() <= 241);
    }

    #[test]
    fn context_checkpoint_write_is_unique_complete_and_leaves_no_temporary_file() {
        let directory = std::env::temp_dir().join(format!(
            "ai-context-checkpoint-test-{}",
            uuid::Uuid::new_v4()
        ));

        let first = save_context_checkpoint_in_dir(&directory, 0, "first", "first body")
            .expect("first checkpoint should save");
        let second = save_context_checkpoint_in_dir(&directory, 0, "second", "second body")
            .expect("second checkpoint should save");

        assert_ne!(first, second);
        assert_eq!(
            std::fs::read_to_string(&first).expect("first checkpoint should be readable"),
            "# Context checkpoint\n\n摘要：first\n\n---\n\nfirst body\n"
        );
        assert_eq!(
            std::fs::read_to_string(&second).expect("second checkpoint should be readable"),
            "# Context checkpoint\n\n摘要：second\n\n---\n\nsecond body\n"
        );
        let entries = std::fs::read_dir(&directory)
            .expect("checkpoint directory should be readable")
            .collect::<Result<Vec<_>, _>>()
            .expect("checkpoint directory entries should be readable");
        assert_eq!(entries.len(), 2);
        assert!(
            entries
                .iter()
                .all(|entry| { !entry.file_name().to_string_lossy().contains(".tmp") })
        );

        std::fs::remove_dir_all(&directory)
            .expect("temporary checkpoint directory should clean up");
    }
}
