//! 工具组折叠（tool group folding）。
//!
//! 把消息序列中较早的 `assistant(tool_calls) + 配套 tool` 组整组折叠成
//! 单条 `internal_note` stub，逐字保留最近的工具组。

use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;

use super::super::types::{Message, ROLE_INTERNAL_NOTE, is_system_like_role, retained_turn_start};
use super::text_utils::truncate_to_chars;
use super::tool_overflow::{is_non_compressible_tool, is_preserved_user_or_image_stub};
use super::{
    is_context_checkpoint_marker, keep_recent_user_turns_when_trimming, normalize_whitespace,
    value_to_string,
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

/// 与 dedup/offload/prune 一致的「最近工具结果保护窗口」在折叠语境下的锚点。
///
/// 其它有损路径都按最近 `KEEP_RECENT_TOOL_MESSAGES` 条 `tool` 消息保护尾窗，而
/// 折叠按「工具组」计窗口。返回「最早那条受保护 tool 消息所属工具组」的锚点
/// 下标：折叠边界夹到该锚点之前，即可保证折叠永不越过统一的近端保护窗口。
/// 若受保护的 tool 消息不足或找不到归属组，返回 `None`（不额外收紧）。
fn recent_tool_message_protection_anchor(
    messages: &[Message],
    group_anchors: &[usize],
) -> Option<usize> {
    if group_anchors.is_empty() {
        return None;
    }
    let tool_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(idx, m)| (m.role == "tool").then_some(idx))
        .collect();
    if tool_indices.len() <= super::KEEP_RECENT_TOOL_MESSAGES {
        // 全部 tool 消息都在保护窗口内：折叠边界夹到最早的工具组锚点之前，
        // 等价于「不折叠任何拥有 tool 结果的组」。
        return group_anchors.first().copied();
    }
    let protect_from = tool_indices.len() - super::KEEP_RECENT_TOOL_MESSAGES;
    let earliest_protected_tool_idx = tool_indices[protect_from];
    // 最早受保护 tool 消息所属的工具组 = 不晚于它的最后一个 assistant(tool_calls) 锚点。
    group_anchors
        .iter()
        .rev()
        .find(|&&anchor| anchor <= earliest_protected_tool_idx)
        .copied()
        .or_else(|| group_anchors.first().copied())
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
    let mut fold_before_anchor = if keep_recent_groups == 0 {
        messages.len()
    } else {
        group_anchors[group_anchors.len() - keep_recent_groups]
    };

    // 与 dedup/offload/prune 统一的「最近工具结果保护窗口」下界：这些有损路径
    // 都按最近 KEEP_RECENT_TOOL_MESSAGES 条 tool 消息保护尾窗，而折叠按「工具组」
    // 数计窗口，二者单位不一致会让折叠把其它路径承诺保留的近端 tool 结果偷偷
    // 折成 stub（近端结果被弱化 → 模型误以为要重跑同一工具）。这里把折叠边界
    // 再夹到「拥有最早受保护 tool 消息的那个工具组锚点」之前，保证折叠永不越过
    // 统一的近端保护窗口。
    if let Some(protected_anchor) = recent_tool_message_protection_anchor(messages, &group_anchors)
    {
        fold_before_anchor = fold_before_anchor.min(protected_anchor);
    }

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
            if let Some(stub) = fold_tool_call_group_to_stub(messages, &group) {
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
