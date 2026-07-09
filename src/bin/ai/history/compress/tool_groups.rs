//! 工具组折叠（tool group folding）。
//!
//! 把消息序列中较早的 `assistant(tool_calls) + 配套 tool` 组整组折叠成
//! 单条 `internal_note` stub，逐字保留最近的工具组。

use rustc_hash::FxHashSet;
use serde_json::Value;

use super::super::types::{Message, ROLE_INTERNAL_NOTE, is_system_like_role, retained_turn_start};
use super::text_utils::truncate_to_chars;
use super::tool_overflow::{is_non_compressible_tool, is_preserved_user_or_image_stub};
use super::{keep_recent_user_turns_when_trimming, normalize_whitespace, value_to_string};

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

pub(super) fn first_trim_candidate(messages: &[Message]) -> Option<usize> {
    let keep_recent_user_turns = keep_recent_user_turns_when_trimming(messages);
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
/// `internal_note`，保留"工具列表 + 每个工具结果首句"，便于后续轮次知道
/// 之前发生过什么、避免重复劳动；同时大幅压缩 token 占用。
pub(super) fn fold_tool_call_group_to_stub(
    messages: &[Message],
    group: &[usize],
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

    let mut lines = Vec::with_capacity(tool_calls.len() + 1);
    lines.push(format!(
        "compressed_tool_round: {} tool calls (folded for context budget)",
        tool_calls.len()
    ));

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
        let one_liner = tool_result_recall_one_liner(&result_text);
        lines.push(format!("- {} => {}", tc.function.name, one_liner));
    }
    if tool_calls.len() > 8 {
        lines.push(format!(
            "- ... ({} more tools omitted)",
            tool_calls.len() - 8
        ));
    }

    Some(Message {
        role: ROLE_INTERNAL_NOTE.to_string(),
        content: Value::String(lines.join("\n")),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    })
}

/// 单轮内折叠：LLM 摘要兜底时，尾窗（当前 user 轮）内部保留逐字的最近工具组数。
/// 更早的同轮工具组折叠成单行 stub，解决"臃肿全堆在一个 user 轮内、无跨轮边界
/// 可摘要"时压缩器空转的问题。
pub(super) const MID_TURN_LLM_SUMMARY_KEEP_RECENT_TOOL_GROUPS: usize = 4;

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
            if let Some(stub) = fold_tool_call_group_to_stub(messages, &group) {
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
