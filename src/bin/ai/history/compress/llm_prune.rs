//! LLM 引导的上下文裁剪（模型标记 → 延迟裁剪）。
//!
//! 这是现有压缩逻辑的**补充**模块，不修改任何已有的压缩代码。
//!
//! ## 原理
//!
//! 每次调用模型时，在 system prompt 中追加一段简短提示，要求模型在
//! 响应中使用 `<meta:self_note>prune:tool_call_id1,tool_call_id2</meta:self_note>`
//! 标记当前不再需要的低价值 tool 消息（通常是过期的普通工具结果）。
//!
//! - user / system / assistant / internal_note 等角色即
//!   使被标记也永远不会被裁剪（见 `is_protected_role`）。
//! - 仅 `role == "tool"` 且拥有 `tool_call_id` 的消息才可能被裁剪。
//! - 工具自身通过 `ToolHistoryPolicyRegistration` 声明 `prune: Never` 的结果
//!   （如 `plan`）永远不会被裁剪；`read_file` / 检索类 / `execute_command` 结果
//!   虽「不可有损压缩」，但**允许**在过时后被 LLM 裁剪（两个维度正交）。
//! - 连续被标记 **PRUNE_THRESHOLD** 次后，消息内容被替换为简短占位符
//!   （保留消息结构、不删除，避免破坏 tool_call ↔ tool_response 配对）。
//! - 如果某条消息在某轮未被标记，其计数重置为 0（"连续"语义）。
//!
//! ## 安全保证
//!
//! 1. 不删除任何消息，不改变 messages 数组长度或顺序。
//! 2. 不修改现有的 `compress/mod.rs` / `context_budget.rs` 逻辑。
//! 3. 只替换 tool 消息的 content 字段为一个短占位符。
//! 4. 最近 `KEEP_RECENT_TOOL_MESSAGES` 条 tool 结果始终保护，避免误裁剪当前轮所需结果。

use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;

use crate::ai::history::types::Message;

use super::tool_overflow::build_tool_call_name_index;

/// 判断某工具结果是否被其注册策略标记为「永不 LLM 裁剪」。
/// 查询工具自身声明的 [`ToolHistoryPolicy`]（见各工具注册文件），
/// 而非硬编码工具名。默认（未注册）允许裁剪；只有显式声明
/// `prune: Never` 的工具（如 `plan`）返回 true。
fn is_prune_protected_tool(tool_name: &str) -> bool {
    !crate::ai::tools::registry::common::tool_history_policy(tool_name).allows_prune()
}

/// 连续被标记多少次后才裁剪。
pub(crate) const PRUNE_THRESHOLD: u8 = 3;

/// 历史消息少于该值时不注入裁剪提示（太短无需裁剪）。
pub(crate) const PRUNE_PROMPT_MIN_MESSAGES: usize = 20;

/// 系统提示中注入的裁剪协议说明。
/// 保持简短，避免占用过多 token。
pub(crate) const PRUNE_PROTOCOL_PROMPT: &str = "\n## Context Management Protocol\n\
When your context holds outdated tool results, actively reclaim space by marking them.\n\
Include a hidden self-note listing the tool_call_ids to prune:\n\
`<meta:self_note>prune:call_abc,call_xyz</meta:self_note>`\n\
Mark any tool result that is now superseded or no longer needed — including old file\n\
reads and code/search results whose content you have already used, that you have since\n\
re-read, or that describe code you have already edited.\n\
Rules:\n\
- Never mark user messages, system instructions, assistant messages, plans, or the most recent tool results.\n\
- Marking is advisory and reversible: the system keeps a result until you mark it on several consecutive turns, and always protects recent results and plans.\n\
- Put the `prune:` directive on its own line; if you also write a normal self_note, keep it in the same hidden note.";

/// 判断该角色的消息是否受保护（永不被裁剪）。
fn is_protected_role(role: &str) -> bool {
    !matches!(role, "tool")
    // tool 本身不受保护，其他所有角色都受保护
}

/// 同上，更清晰的写法。
fn is_prunable_message(msg: &Message) -> bool {
    msg.role == "tool" && msg.tool_call_id.is_some()
}

/// 从模型响应的 hidden_meta 中解析 prune 标记。
///
/// hidden_meta 可能包含多行，其中以 `prune:` 开头的行是裁剪指令，
/// 其余行是常规 self_note 内容（由调用方处理）。
///
/// 返回 `(prune_ids, remaining_meta)`:
/// - `prune_ids`: 被标记的 tool_call_id 列表
/// - `remaining_meta`: 移除了 prune 行后的剩余 hidden_meta（给 self_note 用）
pub(crate) fn parse_prune_from_hidden_meta(hidden_meta: &str) -> (Vec<String>, String) {
    let mut prune_ids = Vec::new();
    let mut remaining_lines = Vec::new();
    let mut saw_prune = false;

    for line in hidden_meta.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("prune:") {
            saw_prune = true;
            // 解析逗号分隔的 tool_call_id 列表
            for id in rest.split(',') {
                let id = id.trim();
                if !id.is_empty() {
                    prune_ids.push(id.to_string());
                }
            }
        } else if !trimmed.is_empty() {
            remaining_lines.push(line.to_string());
        }
    }

    let remaining = if saw_prune {
        remaining_lines.join("\n")
    } else {
        hidden_meta.to_string()
    };
    (prune_ids, remaining)
}

/// 更新裁剪计数。
///
/// - `current_marks`: 当前会话的裁剪计数表（tool_call_id → 连续计数）
/// - `prune_ids`: 本轮模型标记的 tool_call_id 列表
/// - `active_tool_ids`: 本轮 messages 中实际存在的 tool_call_id 集合
///
/// 逻辑：
/// 1. 对每个被标记的 id，计数 +1
/// 2. 对未被标记但存在于 active_tool_ids 中的 id，计数重置为 0
/// 3. 移除计数为 0 的条目
pub(crate) fn update_prune_marks(
    current_marks: &mut FxHashMap<String, u8>,
    prune_ids: &[String],
    active_prunable_tool_ids: &FxHashSet<String>,
) {
    let marked_ids = prune_ids
        .iter()
        .filter(|id| active_prunable_tool_ids.contains(*id))
        .cloned()
        .collect::<FxHashSet<_>>();

    // 重置未被标记的 tool 的计数
    for id in active_prunable_tool_ids {
        if !marked_ids.contains(id) {
            current_marks.remove(id);
        }
    }

    // 增加被标记的 tool 的计数
    for id in marked_ids {
        let count = current_marks.entry(id).or_insert(0);
        *count = count.saturating_add(1);
    }

    // 清理计数为 0、已不在当前上下文、或已被保护策略排除的条目。
    current_marks.retain(|id, v| *v > 0 && active_prunable_tool_ids.contains(id));
}

/// 收集当前上下文中允许被 LLM 引导裁剪的 tool_call_id。
///
/// 保护策略：
/// - 最近完整工具组的结果保留全文。
/// - 工具注册策略声明 `prune: Never` 的结果（如 `plan`）永不裁剪。
///   注意 `read_file` / 检索类虽「不可有损压缩」但**允许**裁剪。
pub(crate) fn active_prunable_tool_ids(messages: &[Message]) -> FxHashSet<String> {
    let protected_ids = protected_tool_call_ids(messages);
    messages
        .iter()
        .filter_map(|message| {
            if !is_prunable_message(message) {
                return None;
            }
            let id = message.tool_call_id.as_ref()?;
            (!protected_ids.contains(id)).then(|| id.clone())
        })
        .collect()
}

fn protected_tool_call_ids(messages: &[Message]) -> FxHashSet<String> {
    let id_to_tool_name = build_tool_call_name_index(messages);
    let protected_indices = super::tool_groups::recent_tool_group_message_indices(
        messages,
        super::KEEP_RECENT_TOOL_GROUPS,
    );

    let mut protected = FxHashSet::default();
    for (idx, message) in messages.iter().enumerate() {
        if message.role != "tool" {
            continue;
        }
        let Some(tool_call_id) = message.tool_call_id.as_ref() else {
            continue;
        };
        if protected_indices.contains(&idx) {
            protected.insert(tool_call_id.clone());
            continue;
        }
        if id_to_tool_name
            .get(tool_call_id)
            .is_some_and(|name| is_prune_protected_tool(name))
        {
            protected.insert(tool_call_id.clone());
        }
    }
    protected
}

/// 单次 `apply_pruning` 的裁剪统计，供调用方打印终端简讯。
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct PruneReport {
    /// 本次被替换为占位符的 tool 结果条数。
    pub(crate) pruned_count: usize,
    /// 净释放的字符数（原内容长度减去占位符长度之和）。
    pub(crate) freed_chars: usize,
    /// 涉及的工具名（去重、按首次出现顺序）。
    pub(crate) tools: Vec<String>,
}

/// 对 messages 数组应用裁剪。
///
/// 将计数 >= PRUNE_THRESHOLD 的 tool 消息内容替换为占位符。
/// 不删除消息、不改变数组长度。
/// 受 `protected_tool_call_ids` 保护的消息（最近完整工具组、以及注册策略声明
/// `prune: Never` 的工具，如 `plan`）
/// 永不被裁剪，避免误裁剪当前轮所需结果或任务路线图锚点。
///
/// 返回本次裁剪的统计报告（供调用方打印终端简讯）。
pub(crate) fn apply_pruning(
    messages: &mut [Message],
    prune_marks: &FxHashMap<String, u8>,
) -> PruneReport {
    let mut report = PruneReport::default();
    if prune_marks.is_empty() {
        return report;
    }

    let id_to_tool_name = build_tool_call_name_index(messages);
    let protected_ids = protected_tool_call_ids(messages);

    for msg in messages.iter_mut() {
        if !is_prunable_message(msg) {
            continue;
        }

        let Some(ref tool_call_id) = msg.tool_call_id else {
            continue;
        };

        if protected_ids.contains(tool_call_id) {
            continue;
        }

        if let Some(&count) = prune_marks.get(tool_call_id) {
            if count >= PRUNE_THRESHOLD {
                let freed = msg.content.as_str().map(|s| s.chars().count()).unwrap_or(0);
                if let Some(name) = id_to_tool_name.get(tool_call_id) {
                    if !report.tools.contains(name) {
                        report.tools.push(name.clone());
                    }
                }
                // 替换内容为占位符，保留消息结构
                let placeholder =
                    format!("[pruned: tool result marked as outdated {} times]", count);
                report.freed_chars += freed.saturating_sub(placeholder.chars().count());
                msg.content = Value::String(placeholder);
                report.pruned_count += 1;
            }
        }
    }

    report
}

/// 判断当前历史长度是否值得注入裁剪提示。
pub(crate) fn should_inject_prune_prompt(message_count: usize) -> bool {
    message_count >= PRUNE_PROMPT_MIN_MESSAGES
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{FunctionCall, ToolCall};
    use serde_json::Value;

    fn make_tool_message(tool_call_id: &str, content: &str) -> Message {
        Message {
            role: "tool".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
            reasoning_content: None,
        }
    }

    fn make_user_message(content: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn make_assistant_message(content: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn make_assistant_tool_call(tool_call_id: &str, tool_name: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: tool_call_id.to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: tool_name.to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn test_is_protected_role() {
        assert!(!is_protected_role("tool"));
        assert!(is_protected_role("user"));
        assert!(is_protected_role("system"));
        assert!(is_protected_role("assistant"));
        assert!(is_protected_role("internal_note"));
    }

    #[test]
    fn test_is_prunable_message() {
        let tool_msg = make_tool_message("call_1", "result");
        assert!(is_prunable_message(&tool_msg));

        let user_msg = make_user_message("hello");
        assert!(!is_prunable_message(&user_msg));

        let assistant_msg = make_assistant_message("hi");
        assert!(!is_prunable_message(&assistant_msg));

        // tool 消息但没有 tool_call_id
        let tool_no_id = Message {
            role: "tool".to_string(),
            content: Value::String("result".to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        };
        assert!(!is_prunable_message(&tool_no_id));
    }

    #[test]
    fn test_parse_prune_from_hidden_meta() {
        let hidden_meta = "prune:call_abc,call_xyz\nDo: be concise\nAvoid: verbosity";
        let (ids, remaining) = parse_prune_from_hidden_meta(hidden_meta);

        assert_eq!(ids, vec!["call_abc", "call_xyz"]);
        assert!(remaining.contains("Do: be concise"));
        assert!(remaining.contains("Avoid: verbosity"));
        assert!(!remaining.contains("prune:"));
    }

    #[test]
    fn test_parse_prune_only() {
        let hidden_meta = "prune:call_1,call_2";
        let (ids, remaining) = parse_prune_from_hidden_meta(hidden_meta);

        assert_eq!(ids.len(), 2);
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_no_prune() {
        let hidden_meta = "Do: be focused\nAvoid: tangents";
        let (ids, remaining) = parse_prune_from_hidden_meta(hidden_meta);

        assert!(ids.is_empty());
        assert_eq!(remaining, "Do: be focused\nAvoid: tangents");
    }

    #[test]
    fn test_parse_empty() {
        let (ids, remaining) = parse_prune_from_hidden_meta("");
        assert!(ids.is_empty());
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_update_prune_marks_increment() {
        let mut marks = FxHashMap::default();
        let active: FxHashSet<String> = ["call_1", "call_2", "call_3"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        // 第一轮标记 call_1, call_2
        update_prune_marks(
            &mut marks,
            &["call_1".to_string(), "call_2".to_string()],
            &active,
        );
        assert_eq!(marks.get("call_1"), Some(&1));
        assert_eq!(marks.get("call_2"), Some(&1));
        assert!(!marks.contains_key("call_3"));

        // 第二轮标记 call_1, call_2
        update_prune_marks(
            &mut marks,
            &["call_1".to_string(), "call_2".to_string()],
            &active,
        );
        assert_eq!(marks.get("call_1"), Some(&2));
        assert_eq!(marks.get("call_2"), Some(&2));

        // 第三轮只标记 call_1，call_2 计数重置
        update_prune_marks(&mut marks, &["call_1".to_string()], &active);
        assert_eq!(marks.get("call_1"), Some(&3));
        assert!(!marks.contains_key("call_2")); // 重置后被清理
    }

    #[test]
    fn test_update_prune_marks_empty_round_resets_active_marks() {
        let mut marks = FxHashMap::default();
        marks.insert("call_1".to_string(), 2);
        marks.insert("stale".to_string(), 2);
        let active: FxHashSet<String> =
            ["call_1", "call_2"].iter().map(|s| s.to_string()).collect();

        update_prune_marks(&mut marks, &[], &active);

        assert!(marks.is_empty());
    }

    #[test]
    fn test_update_prune_marks_deduplicates_single_round_marks() {
        let mut marks = FxHashMap::default();
        let active: FxHashSet<String> = ["call_1"].iter().map(|s| s.to_string()).collect();

        update_prune_marks(
            &mut marks,
            &[
                "call_1".to_string(),
                "call_1".to_string(),
                "missing".to_string(),
            ],
            &active,
        );

        assert_eq!(marks.get("call_1"), Some(&1));
        assert!(!marks.contains_key("missing"));
    }

    #[test]
    fn test_apply_pruning_replaces_content() {
        let mut marks = FxHashMap::default();
        marks.insert("call_old".to_string(), PRUNE_THRESHOLD);
        marks.insert("call_keep".to_string(), 1);

        let mut messages = vec![
            make_tool_message(
                "call_old",
                "very long outdated result that should be pruned",
            ),
            make_tool_message("call_keep", "still relevant result"),
            make_tool_message("call_recent_1", "current turn result 1"),
            make_tool_message("call_recent_2", "current turn result 2"),
            make_tool_message("call_recent_3", "current turn result 3"),
            make_tool_message("call_recent_4", "current turn result 4"),
            make_tool_message("call_recent_5", "current turn result 5"),
            make_tool_message("call_recent_6", "current turn result 6"),
            make_user_message("what about this?"),
        ];

        let pruned = apply_pruning(&mut messages, &marks);

        assert_eq!(pruned.pruned_count, 1);
        // call_old 内容被替换
        assert!(messages[0].content.as_str().unwrap().contains("[pruned"));
        // call_keep 内容不变（计数 < threshold）
        assert_eq!(
            messages[1].content.as_str().unwrap(),
            "still relevant result"
        );
        // 最近 tool 窗口内容不变
        assert_eq!(
            messages[2].content.as_str().unwrap(),
            "current turn result 1"
        );
        // user 消息不变
        assert_eq!(messages[8].content.as_str().unwrap(), "what about this?");
    }

    #[test]
    fn test_apply_pruning_protects_recent_tool_groups() {
        let mut marks = FxHashMap::default();
        marks.insert("call_last".to_string(), PRUNE_THRESHOLD);

        let mut messages = vec![
            make_assistant_tool_call("call_prev", "execute_command"),
            make_tool_message("call_prev", "old result"),
            make_assistant_tool_call("call_last", "execute_command"),
            make_tool_message("call_last", "most recent result"),
        ];

        let pruned = apply_pruning(&mut messages, &marks);

        // call_last 所在的最近完整工具组受保护，不被裁剪。
        assert_eq!(pruned.pruned_count, 0);
        assert_eq!(messages[3].content.as_str().unwrap(), "most recent result");
    }

    #[test]
    fn test_apply_pruning_empty_marks() {
        let mut messages = vec![make_tool_message("call_1", "result")];

        let pruned = apply_pruning(&mut messages, &FxHashMap::default());
        assert_eq!(pruned.pruned_count, 0);
        assert_eq!(messages[0].content.as_str().unwrap(), "result");
    }

    #[test]
    fn test_apply_pruning_never_touches_user_or_assistant() {
        let mut marks = FxHashMap::default();
        // 即使 user/assistant 有对应的 "tool_call_id"，也不会被裁剪
        marks.insert("call_1".to_string(), PRUNE_THRESHOLD);
        marks.insert("call_2".to_string(), PRUNE_THRESHOLD);

        let mut messages = vec![
            make_user_message("important user question"),
            make_assistant_message("important assistant response"),
            make_assistant_tool_call("call_1", "execute_command"),
            make_tool_message("call_1", "outdated tool result"),
            make_assistant_tool_call("call_2", "execute_command"),
            make_tool_message("call_2", "current tool result"),
            make_assistant_tool_call("call_3", "execute_command"),
            make_tool_message("call_3", "recent tool result 3"),
            make_assistant_tool_call("call_4", "execute_command"),
            make_tool_message("call_4", "recent tool result 4"),
            make_assistant_tool_call("call_5", "execute_command"),
            make_tool_message("call_5", "recent tool result 5"),
        ];

        let pruned = apply_pruning(&mut messages, &marks);

        assert_eq!(pruned.pruned_count, 1);
        assert_eq!(
            messages[0].content.as_str().unwrap(),
            "important user question"
        );
        assert_eq!(
            messages[1].content.as_str().unwrap(),
            "important assistant response"
        );
        assert!(messages[3].content.as_str().unwrap().contains("[pruned"));
    }

    #[test]
    fn test_active_prunable_tool_ids_excludes_recent_groups_and_non_compressible_tools() {
        let messages = vec![
            make_assistant_tool_call("call_plan", "plan"),
            make_tool_message("call_plan", "task plan"),
            make_assistant_tool_call("call_old", "execute_command"),
            make_tool_message("call_old", "old command output"),
            make_assistant_tool_call("call_recent_1", "execute_command"),
            make_tool_message("call_recent_1", "recent 1"),
            make_assistant_tool_call("call_recent_2", "execute_command"),
            make_tool_message("call_recent_2", "recent 2"),
            make_assistant_tool_call("call_recent_3", "execute_command"),
            make_tool_message("call_recent_3", "recent 3"),
            make_assistant_tool_call("call_recent_4", "execute_command"),
            make_tool_message("call_recent_4", "recent 4"),
        ];

        let ids = active_prunable_tool_ids(&messages);

        assert_eq!(ids.len(), 1);
        assert!(ids.contains("call_old"));
        assert!(!ids.contains("call_plan"));
        assert!(!ids.contains("call_recent_1"));
    }

    /// 解耦不变量：`read_file` 声明 `lossy_compress: Never` 但 `prune: Allow`，
    /// 因此虽然「不可有损压缩」，其过时旧结果仍可被 LLM 裁剪。而 `plan`
    /// 声明 `prune: Never`，永不进入裁剪候选。
    #[test]
    fn test_active_prunable_allows_read_file_but_protects_plan() {
        let messages = vec![
            make_assistant_tool_call("call_plan", "plan"),
            make_tool_message("call_plan", "task plan"),
            make_assistant_tool_call("call_read", "read_file"),
            make_tool_message("call_read", "old file contents already used"),
            make_assistant_tool_call("call_recent_1", "execute_command"),
            make_tool_message("call_recent_1", "recent 1"),
            make_assistant_tool_call("call_recent_2", "execute_command"),
            make_tool_message("call_recent_2", "recent 2"),
            make_assistant_tool_call("call_recent_3", "execute_command"),
            make_tool_message("call_recent_3", "recent 3"),
            make_assistant_tool_call("call_recent_4", "execute_command"),
            make_tool_message("call_recent_4", "recent 4"),
            make_assistant_tool_call("call_recent_5", "execute_command"),
            make_tool_message("call_recent_5", "recent 5"),
            make_assistant_tool_call("call_recent_6", "execute_command"),
            make_tool_message("call_recent_6", "recent 6"),
        ];

        let ids = active_prunable_tool_ids(&messages);

        // read_file 现在允许裁剪（旧行为下会被排除）。
        assert!(ids.contains("call_read"));
        // plan 仍受注册策略保护，永不裁剪。
        assert!(!ids.contains("call_plan"));
    }

    #[test]
    fn test_apply_pruning_protects_non_compressible_tools() {
        let mut marks = FxHashMap::default();
        marks.insert("call_plan".to_string(), PRUNE_THRESHOLD);
        marks.insert("call_old".to_string(), PRUNE_THRESHOLD);

        let mut messages = vec![
            make_assistant_tool_call("call_plan", "plan"),
            make_tool_message("call_plan", "task plan"),
            make_assistant_tool_call("call_old", "execute_command"),
            make_tool_message("call_old", "old command output"),
            make_assistant_tool_call("call_recent_1", "execute_command"),
            make_tool_message("call_recent_1", "recent 1"),
            make_assistant_tool_call("call_recent_2", "execute_command"),
            make_tool_message("call_recent_2", "recent 2"),
            make_assistant_tool_call("call_recent_3", "execute_command"),
            make_tool_message("call_recent_3", "recent 3"),
            make_assistant_tool_call("call_recent_4", "execute_command"),
            make_tool_message("call_recent_4", "recent 4"),
            make_assistant_tool_call("call_recent_5", "execute_command"),
            make_tool_message("call_recent_5", "recent 5"),
            make_assistant_tool_call("call_recent_6", "execute_command"),
            make_tool_message("call_recent_6", "recent 6"),
        ];

        let pruned = apply_pruning(&mut messages, &marks);

        assert_eq!(pruned.pruned_count, 1);
        assert_eq!(messages[1].content.as_str().unwrap(), "task plan");
        assert!(messages[3].content.as_str().unwrap().contains("[pruned"));
    }

    #[test]
    fn test_should_inject_prune_prompt() {
        assert!(!should_inject_prune_prompt(0));
        assert!(!should_inject_prune_prompt(PRUNE_PROMPT_MIN_MESSAGES - 1));
        assert!(should_inject_prune_prompt(PRUNE_PROMPT_MIN_MESSAGES));
        assert!(should_inject_prune_prompt(100));
    }

    #[test]
    fn test_prune_threshold_is_reasonable() {
        // 确保阈值不是 0 或 1（太激进），也不超过 10（太保守）
        assert!(PRUNE_THRESHOLD >= 2);
        assert!(PRUNE_THRESHOLD <= 10);
    }

    #[test]
    fn test_message_count_after_pruning_unchanged() {
        let mut marks = FxHashMap::default();
        marks.insert("call_1".to_string(), PRUNE_THRESHOLD);
        marks.insert("call_2".to_string(), PRUNE_THRESHOLD);
        marks.insert("call_3".to_string(), PRUNE_THRESHOLD);

        let mut messages = vec![
            make_tool_message("call_1", "result 1"),
            make_tool_message("call_2", "result 2"),
            make_tool_message("call_3", "result 3"),
            make_tool_message("call_4", "old unmarked result"),
            make_tool_message("call_5", "recent result 5"),
            make_tool_message("call_6", "recent result 6"),
            make_tool_message("call_7", "recent result 7"),
            make_tool_message("call_8", "recent result 8"),
            make_tool_message("call_9", "recent result 9"),
            make_tool_message("call_10", "recent result 10"),
        ];

        let len_before = messages.len();
        let pruned = apply_pruning(&mut messages, &marks);
        let len_after = messages.len();

        assert_eq!(len_before, len_after);
        assert_eq!(pruned.pruned_count, 3); // 最近 6 条 tool 受保护
    }
}
