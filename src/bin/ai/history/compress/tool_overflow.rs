//! 工具溢出处理与持久化摘要构建。
//!
//! - `prepare_tool_messages_structured`：结构化裁剪 tool 消息
//! - `build_persisted_summary_text` / `build_persisted_summary_text_with_app`：构建持久化摘要
//! - `write_preserved_tool_overflow_file` 等：将溢出内容写入归档文件
//! - `structured_tool_output_summary`：工具结果结构化摘要
//! - `is_non_compressible_tool` / `is_preserved_user_or_image_stub`：工具分类判断

use std::path::{Path, PathBuf};

use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;

use crate::ai::{request, tools::tool_history_policy, types::App};

use super::super::types::{Message, ROLE_INTERNAL_NOTE, is_system_like_role, retained_turn_start};
use super::text_utils::{keep_ends_by_chars, summarize_text, truncate_to_chars};
use super::tool_groups::{recent_tool_group_message_indices, recent_tool_result_groups};
use super::{
    COMPRESSED_TOOL_EVIDENCE_MARKER, IMAGE_OVERFLOW_SPILL_MIN_CHARS, KEEP_RECENT_TOOL_GROUPS,
    PRESERVED_CONTENT_STUB_PREFIX, PRESERVED_IMAGE_OVERFLOW_DIR, PRESERVED_TOOL_OVERFLOW_DIR,
    PRESERVED_USER_OVERFLOW_DIR, USER_OVERFLOW_SPILL_MIN_CHARS, automatic_summary_body,
    dedup_adjacent, keep_recent_user_turns_when_trimming, message_contains_image,
    normalize_whitespace, redact_images_except_last, strip_nested_prior_summary_prefixes,
    tool_message_indices, value_to_string,
};

const PRESERVED_TOOL_OVERFLOW_STUB_PREFIX: &str = "[[PRESERVED_TOOL_OVERFLOW_STUB_V1]]";
const LEGACY_PRESERVED_TOOL_OVERFLOW_STUB_PREFIX: &str =
    "Output preserved for non-compressible tool `";

pub(super) async fn build_persisted_summary_text_with_app(
    app: &App,
    messages: &[Message],
    max_chars: usize,
) -> String {
    let mut prepared = messages.to_vec();
    prepare_tool_messages_structured(
        &mut prepared,
        360,
        KEEP_RECENT_TOOL_GROUPS,
        None,
        &FxHashSet::default(),
    );
    redact_images_except_last(&mut prepared, 0);
    dedup_adjacent(&mut prepared);
    normalize_internal_notes_for_summary_model(&mut prepared);

    if let Some(summary) = request::summarize_history_via_model(app, &prepared, max_chars).await {
        let summary = normalize_whitespace(&summary);
        if !summary.is_empty() {
            return summary;
        }
    }

    build_persisted_summary_text(messages, max_chars)
}

pub(super) fn normalize_internal_notes_for_summary_model(messages: &mut Vec<Message>) {
    let mut out = Vec::with_capacity(messages.len());
    let mut seen_auto_summary = false;

    for mut message in messages.drain(..) {
        if message.role == ROLE_INTERNAL_NOTE {
            let text = value_to_string(&message.content);
            if let Some(body) = automatic_summary_body(&text) {
                if seen_auto_summary {
                    continue;
                }
                let body = strip_nested_prior_summary_prefixes(body);
                if !body.is_empty() {
                    message.content = Value::String(format!(
                        "已有历史摘要（供本次压缩吸收，不要逐字复制）：\n{}",
                        summarize_text(&body, 2_000)
                    ));
                    out.push(message);
                    seen_auto_summary = true;
                }
                continue;
            }

            if text.trim_start().contains(COMPRESSED_TOOL_EVIDENCE_MARKER) {
                out.push(message);
                continue;
            }

            // 普通 internal_note 多为过程性提示、cache/loop 状态或 self_note 的
            // inline 副本。它们不应被当成长期历史事实交给摘要模型反复吸收。
            continue;
        }
        out.push(message);
    }

    *messages = out;
}

pub(super) fn prepare_tool_messages_structured(
    messages: &mut [Message],
    max_chars_per_msg: usize,
    keep_recent_groups: usize,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &FxHashSet<String>,
) {
    let id_to_tool_name = build_tool_call_name_index(messages);
    let id_to_tool_args = build_tool_call_arguments_index(messages);
    let indices = tool_message_indices(messages);
    let protected_indices = recent_tool_group_message_indices(messages, keep_recent_groups);
    for &idx in &indices {
        let message = &mut messages[idx];
        let text = value_to_string(&message.content);
        if text.trim().is_empty() {
            continue;
        }
        // 已外溢的 precision 工具结果是稳定指针，不能再把 stub 当成原始结果
        // 外溢一次。否则每轮压缩都会写出 `stub -> stub` 新文件，既泄漏磁盘，
        // 也让模型必须沿多层指针才能回到原始证据。
        if is_preserved_tool_overflow_stub(&text) {
            continue;
        }

        let tool_name = message
            .tool_call_id
            .as_deref()
            .and_then(|id| id_to_tool_name.get(id))
            .map(|s| s.as_str());
        if let Some(name) = tool_name
            && is_non_compressible_tool(name)
        {
            // 最近完整工具组不外溢：刚读到的文件/检索结果必须在下一轮请求里
            // 完整可见，否则模型看到的是「已卸载，请重读」stub，会立刻再发一次
            // 同样的 read_file——在会话超软阈值、每轮都跑压缩时表现为无限重读。
            // 只有保护尾窗之外的旧 precision 结果才零压缩外溢到磁盘。
            let is_explicitly_protected = message
                .tool_call_id
                .as_ref()
                .is_some_and(|id| protected_tool_call_ids.contains(id));
            if !is_explicitly_protected
                && !protected_indices.contains(&idx)
                && text.chars().count() > max_chars_per_msg
                && let Some(path) = overflow_dir
                    .and_then(|dir| write_preserved_tool_overflow_file(dir, name, &text))
            {
                let recall_lines = message
                    .tool_call_id
                    .as_deref()
                    .and_then(|id| id_to_tool_args.get(id))
                    .map(|args| build_tool_overflow_recall_lines(name, args))
                    .unwrap_or_default();
                message.content = Value::String(build_preserved_tool_overflow_stub(
                    &path,
                    name,
                    &text,
                    &recall_lines,
                ));
            }
            continue;
        }

        if protected_indices.contains(&idx) {
            // 最近完整工具组的普通工具结果仍保留全文，避免误伤近端上下文。
            continue;
        }

        let summary = structured_tool_output_summary(&text, max_chars_per_msg);
        if !summary.is_empty() && summary != text {
            message.content = Value::String(summary);
        }
    }
}

/// 最新并行批次可能单独超过上下文窗口。此时仍按完整组判定，但对注册为高精度
/// grounding 的结果设置 inline 上限：超过预算的结果零压缩外溢并保留可召回 stub。
/// `task` / `task_wait` 等聚合结果没有注册该标志，不会挤占 read_file 等证据的预算。
pub(super) fn enforce_protected_precision_group_budget(
    messages: &mut [Message],
    keep_recent_groups: usize,
    inline_budget: usize,
    overflow_dir: Option<&Path>,
    protected_tool_call_ids: &FxHashSet<String>,
) {
    let Some(overflow_dir) = overflow_dir else {
        return;
    };
    let id_to_tool_name = build_tool_call_name_index(messages);
    let id_to_tool_args = build_tool_call_arguments_index(messages);

    for group in recent_tool_result_groups(messages, keep_recent_groups) {
        let mut precision_results: Vec<(usize, String)> = group
            .into_iter()
            .filter_map(|idx| {
                let tool_name = messages[idx]
                    .tool_call_id
                    .as_deref()
                    .and_then(|id| id_to_tool_name.get(id))?;
                tool_history_policy(tool_name)
                    .counts_toward_precision_inline_budget()
                    .then(|| (idx, tool_name.clone()))
            })
            .collect();

        let mut total_chars = precision_results
            .iter()
            .map(|(idx, _)| value_to_string(&messages[*idx].content).chars().count())
            .sum::<usize>();
        precision_results.sort_unstable_by_key(|(idx, _)| {
            std::cmp::Reverse(value_to_string(&messages[*idx].content).chars().count())
        });

        // 优先外溢最大的结果，以最少的 stub 腾出足够空间；其余同组证据仍完整可见。
        for (idx, tool_name) in precision_results {
            if total_chars <= inline_budget {
                break;
            }
            let text = value_to_string(&messages[idx].content);
            if text.trim().is_empty() || is_preserved_tool_overflow_stub(&text) {
                continue;
            }
            if messages[idx]
                .tool_call_id
                .as_ref()
                .is_some_and(|id| protected_tool_call_ids.contains(id))
            {
                continue;
            }
            let text_len = text.chars().count();
            if let Some(path) = write_preserved_tool_overflow_file(overflow_dir, &tool_name, &text)
            {
                let recall_lines = messages[idx]
                    .tool_call_id
                    .as_deref()
                    .and_then(|id| id_to_tool_args.get(id))
                    .map(|args| build_tool_overflow_recall_lines(&tool_name, args))
                    .unwrap_or_default();
                messages[idx].content = Value::String(build_preserved_tool_overflow_stub(
                    &path,
                    &tool_name,
                    &text,
                    &recall_lines,
                ));
                total_chars = total_chars.saturating_sub(text_len);
            }
        }
    }
}

pub(super) fn build_tool_call_name_index(messages: &[Message]) -> FxHashMap<String, String> {
    let mut out = FxHashMap::default();
    for message in messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            out.insert(tool_call.id.clone(), tool_call.function.name.clone());
        }
    }
    out
}

fn build_tool_call_arguments_index(messages: &[Message]) -> FxHashMap<String, String> {
    let mut out = FxHashMap::default();
    for message in messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            out.insert(tool_call.id.clone(), tool_call.function.arguments.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{FunctionCall, ToolCall};

    fn assistant_call(id: &str, name: &str) -> Message {
        assistant_call_args(id, name, "{}")
    }

    fn assistant_call_args(id: &str, name: &str, arguments: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![ToolCall {
                id: id.to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: arguments.to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn tool_result(id: &str, content: &str) -> Message {
        Message {
            role: "tool".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: Some(id.to_string()),
            reasoning_content: None,
        }
    }

    #[test]
    fn preserved_tool_overflow_stub_is_not_spilled_again() {
        let overflow_dir =
            std::env::temp_dir().join(format!("ai-tool-overflow-stub-{}", uuid::Uuid::new_v4()));
        let mut messages = vec![
            assistant_call("old", "read_file"),
            tool_result("old", &"x".repeat(1_000)),
            assistant_call("recent", "read_file"),
            tool_result("recent", "recent result"),
        ];

        prepare_tool_messages_structured(
            &mut messages,
            80,
            1,
            Some(&overflow_dir),
            &FxHashSet::default(),
        );
        let first_stub = value_to_string(&messages[1].content);
        assert!(is_preserved_tool_overflow_stub(&first_stub));
        let overflow_path = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
        assert_eq!(std::fs::read_dir(&overflow_path).unwrap().count(), 1);

        prepare_tool_messages_structured(
            &mut messages,
            80,
            1,
            Some(&overflow_dir),
            &FxHashSet::default(),
        );
        assert_eq!(value_to_string(&messages[1].content), first_stub);
        assert_eq!(std::fs::read_dir(&overflow_path).unwrap().count(), 1);

        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn preserved_read_file_overflow_stub_keeps_original_target_anchor() {
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-tool-overflow-read-anchor-{}",
            uuid::Uuid::new_v4()
        ));
        let mut messages = vec![
            assistant_call_args(
                "old",
                "read_file",
                r#"{"file_path":"src/lib.rs","offset":120,"limit":40}"#,
            ),
            tool_result("old", &"x".repeat(1_000)),
            assistant_call("recent", "read_file"),
            tool_result("recent", "recent result"),
        ];

        prepare_tool_messages_structured(
            &mut messages,
            80,
            1,
            Some(&overflow_dir),
            &FxHashSet::default(),
        );
        let stub = value_to_string(&messages[1].content);
        assert!(
            stub.contains("- original_file_path: src/lib.rs"),
            "stub: {stub}"
        );
        assert!(
            stub.contains("- original_range: lines=120..159"),
            "stub: {stub}"
        );
        assert!(
            stub.contains("优先继续读取 `original_file_path` 指向的原始文件"),
            "stub: {stub}"
        );

        let anchor = collapse_overflow_stub_to_anchor(&stub).expect("stub should collapse");
        assert!(
            anchor.contains("- original_file_path: src/lib.rs"),
            "anchor: {anchor}"
        );
        assert!(
            anchor.contains("优先读取 `original_file_path`"),
            "anchor: {anchor}"
        );

        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn preserved_execute_command_overflow_stub_keeps_original_command_anchor() {
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-tool-overflow-command-anchor-{}",
            uuid::Uuid::new_v4()
        ));
        let mut messages = vec![
            assistant_call_args(
                "old",
                "execute_command",
                r#"{"command":"git log --stat","cwd":"/repo"}"#,
            ),
            tool_result("old", &"x".repeat(1_000)),
            assistant_call("recent", "read_file"),
            tool_result("recent", "recent result"),
        ];

        prepare_tool_messages_structured(
            &mut messages,
            80,
            1,
            Some(&overflow_dir),
            &FxHashSet::default(),
        );
        let stub = value_to_string(&messages[1].content);
        assert!(
            stub.contains("- original_command: git log --stat"),
            "stub: {stub}"
        );
        assert!(stub.contains("- original_cwd: /repo"), "stub: {stub}");
        assert!(
            stub.contains("优先根据 `original_command` / `original_cwd` 继续判断"),
            "stub: {stub}"
        );

        let anchor = collapse_overflow_stub_to_anchor(&stub).expect("stub should collapse");
        assert!(
            anchor.contains("- original_command: git log --stat"),
            "anchor: {anchor}"
        );

        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    #[test]
    fn legacy_tool_overflow_stub_is_recognized() {
        let legacy = "Output preserved for non-compressible tool `read_file`.\n\
            - file_path: /tmp/result.txt\n\
            - use read_file to inspect exact content.\n\
            Preview (for recall; not exhaustive):";
        assert!(is_preserved_tool_overflow_stub(legacy));
    }

    #[test]
    fn protected_precision_budget_excludes_aggregated_task_results() {
        let overflow_dir = std::env::temp_dir().join(format!(
            "ai-precision-group-budget-{}",
            uuid::Uuid::new_v4()
        ));
        let mut call = assistant_call("read", "read_file");
        call.tool_calls.as_mut().unwrap().push(ToolCall {
            id: "task".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "task_wait".to_string(),
                arguments: "{}".to_string(),
            },
        });
        let mut messages = vec![
            call,
            tool_result("read", &"r".repeat(1_000)),
            tool_result("task", &"t".repeat(10_000)),
        ];

        enforce_protected_precision_group_budget(
            &mut messages,
            1,
            200,
            Some(&overflow_dir),
            &FxHashSet::default(),
        );

        assert!(is_preserved_tool_overflow_stub(&value_to_string(
            &messages[1].content
        )));
        assert_eq!(value_to_string(&messages[2].content).len(), 10_000);
        let _ = std::fs::remove_dir_all(overflow_dir);
    }

    fn user_msg(text: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: Value::String(text.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    /// 构造一条「首次外溢」形态的 stub（含多行 Preview 正文），用于折叠测试。
    fn overflow_stub_with_preview(file_path: &str, tool_name: &str) -> String {
        let full = (0..40)
            .map(|i| format!("line {i}: some content"))
            .collect::<Vec<_>>()
            .join("\n");
        build_preserved_tool_overflow_stub(Path::new(file_path), tool_name, &full, &[])
    }

    #[test]
    fn collapse_overflow_stub_to_anchor_drops_preview_keeps_file_path() {
        let stub = overflow_stub_with_preview("/tmp/session/read-abc.txt", "read_file");
        // 前置条件：首次 stub 确实带 Preview 正文。
        assert!(stub.contains("Preview ("));

        let anchor = collapse_overflow_stub_to_anchor(&stub).expect("should collapse");
        // 预览正文被丢弃。
        assert!(!anchor.contains("Preview ("));
        // file_path 保留。
        assert!(anchor.contains("- file_path: /tmp/session/read-abc.txt"));
        // 工具名保留（新格式用 "Output preserved for tool"）。
        assert!(anchor.contains("Output preserved for tool `read_file`"));
        // read_file 类型的归档有"通常无需重新读取"提示（而非旧的诱导式 "use read_file"）。
        assert!(anchor.contains("之前读取文件的结果归档"));
        // 仍是合法 stub（前缀不变），后续压缩链继续按 stub 豁免识别。
        assert!(is_preserved_tool_overflow_stub(&anchor));
        // 体量骤降。
        assert!(anchor.len() < stub.len());
    }

    #[test]
    fn age_out_overflow_stub_previews_is_idempotent() {
        let stub = overflow_stub_with_preview("/tmp/session/read-xyz.txt", "read_file");
        // 两条 user 轮，让 stub 落在保护尾窗之外（retained_turn_start 之前）。
        let mut messages = vec![
            user_msg("q1"),
            assistant_call("s", "read_file"),
            tool_result("s", "placeholder"),
            user_msg("q2"),
            user_msg("q3"),
        ];
        messages[2].content = Value::String(stub);

        age_out_overflow_stub_previews(&mut messages, 1);
        let after_first = value_to_string(&messages[2].content);
        assert!(!after_first.contains("Preview ("));

        // 再跑一次：已是锚点形态，内容不得再变（防 stub->stub 抖动）。
        age_out_overflow_stub_previews(&mut messages, 1);
        assert_eq!(value_to_string(&messages[2].content), after_first);
    }

    #[test]
    fn age_out_overflow_stub_previews_respects_protected_tail() {
        // 早期 stub（尾窗外）与近端 stub（尾窗内）各一条。
        let early = overflow_stub_with_preview("/tmp/session/early.txt", "read_file");
        let recent = overflow_stub_with_preview("/tmp/session/recent.txt", "read_file");
        let mut messages = vec![
            user_msg("q1"),
            assistant_call("early", "read_file"),
            tool_result("early", "placeholder"),
            user_msg("q2"),
            assistant_call("recent", "read_file"),
            tool_result("recent", "placeholder"),
        ];
        messages[2].content = Value::String(early);
        messages[5].content = Value::String(recent.clone());

        // 保护最近 1 个 user 轮（q2 起）：早期 stub 折叠，尾窗内 recent 保留完整预览。
        age_out_overflow_stub_previews(&mut messages, 1);
        assert!(!value_to_string(&messages[2].content).contains("Preview ("));
        assert_eq!(value_to_string(&messages[5].content), recent);
        assert!(value_to_string(&messages[5].content).contains("Preview ("));
    }
}

/// 「读取/检索」类工具的输出零压缩（不行裁剪、不去重折叠、不整组删除），
/// 超阈值时只做"零压缩外溢到会话文件 + 留指针 stub"。这类输出复现代价高，
/// 一旦被压掉模型就会反复重跑同一次检索（典型失忆/原地打转症状）。
///
/// 现在改为查询工具自身声明的历史保留策略
/// （`ToolHistoryPolicyRegistration`，见各工具注册文件），而非在此硬编码
/// 工具名列表。默认未注册的工具允许有损压缩；只有显式声明
/// `lossy_compress: Never` 的工具（`read_file` / 检索类 / `execute_command`）
/// 返回 true。`plan` 不再禁止有损压缩：最新一版由最近工具组保护窗口完整保留，
/// 旧版可摘要压缩以释放上下文。注意：这与「是否允许 LLM 裁剪」是正交维度——见 `llm_prune.rs`。
pub(super) fn is_non_compressible_tool(tool_name: &str) -> bool {
    !crate::ai::tools::registry::common::tool_history_policy(tool_name).allows_lossy_compress()
}

/// 将尚未外溢的高精度工具结果写入会话 asset，并返回带 `file_path` 的稳定 stub。
///
/// 工具组折叠会移除原始 `tool` 消息；对不可有损压缩的结果，必须先走这条路径，
/// 否则折叠 note 只剩一行首句，完整诊断将没有任何可回读的真相来源。已有 stub
/// 直接复用，避免同一结果在多轮压缩中反复写出副本。调用方可额外传入
/// `original_*` 锚点，让 stub 自身就明确区分「原始目标」与「内部归档文件」。
pub(super) fn preserve_noncompressible_tool_result_for_fold(
    overflow_dir: Option<&Path>,
    tool_name: &str,
    content: &str,
    recall_lines: &[String],
) -> Option<String> {
    if is_preserved_tool_overflow_stub(content) {
        return Some(content.to_string());
    }
    let path =
        overflow_dir.and_then(|dir| write_preserved_tool_overflow_file(dir, tool_name, content))?;
    Some(build_preserved_tool_overflow_stub(
        &path,
        tool_name,
        content,
        recall_lines,
    ))
}

fn write_preserved_tool_overflow_file(
    overflow_dir: &Path,
    tool_name: &str,
    content: &str,
) -> Option<PathBuf> {
    let dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    std::fs::create_dir_all(&dir).ok()?;
    let safe_tool = tool_name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    let file_name = format!(
        "{}-{}-{}.txt",
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ"),
        safe_tool,
        uuid::Uuid::new_v4().simple()
    );
    let path = dir.join(file_name);
    std::fs::write(&path, content).ok()?;
    Some(path)
}

fn build_preserved_tool_overflow_stub(
    path: &Path,
    tool_name: &str,
    full_content: &str,
    recall_lines: &[String],
) -> String {
    // 仍把全文外溢到磁盘以控制上下文体积，但在 stub 内保留 head+tail 预览，
    // 让后续 turn 拥有"召回锚点"——模型据此判断是否真的需要重新 read_file，
    // 避免早期读到的代码被搬走后出现"失忆/反复重读"。
    // 提示文案保持中性：明确告知"仅在需要完整内容时才读取"，防止 LLM 看到
    // file_path 就无条件重读导致外溢→重读→再外溢的无限循环。
    let preview = build_overflow_content_preview(full_content);
    let tool_hint = preserved_tool_overflow_hint(tool_name, recall_lines);
    let mut out = format!(
        "{PRESERVED_TOOL_OVERFLOW_STUB_PREFIX}\n\
         Output preserved for tool `{tool_name}`. Full result saved to session asset:\n\
         - file_path: {}",
        path.display(),
    );
    for line in recall_lines {
        out.push('\n');
        out.push_str(line);
    }
    out.push('\n');
    out.push_str(tool_hint);
    out.push('\n');
    out.push_str(&preview);
    out
}

fn preserved_tool_overflow_hint(tool_name: &str, recall_lines: &[String]) -> &'static str {
    let has_original_file_path = recall_lines
        .iter()
        .any(|line| line.starts_with("- original_file_path: "));
    let has_original_command = recall_lines
        .iter()
        .any(|line| line.starts_with("- original_command: "));
    match tool_name {
        "read_file" if has_original_file_path => {
            "（这是之前读取文件的结果归档；优先继续读取 `original_file_path` 指向的原始文件，而不是本归档文件。仅当你确实需要这次已外溢结果的完整原文时才读取 `file_path`。）"
        }
        "read_file" => {
            "（这是之前读取文件的结果归档；除非你需要重新查看该文件的完整内容且预览不足，否则不要重新读取。）"
        }
        "execute_command" if has_original_command => {
            "（这是之前命令输出的归档；优先根据 `original_command` / `original_cwd` 继续判断，不要把 `file_path` 当作源码文件去读。仅在需要完整日志时才读取。）"
        }
        _ => "（如需查看完整输出，可用 read_file 读取此文件；若预览已足够回答问题则忽略。）",
    }
}

pub(super) fn build_tool_overflow_recall_lines(tool_name: &str, arguments: &str) -> Vec<String> {
    let Ok(args) = serde_json::from_str::<Value>(arguments) else {
        return Vec::new();
    };

    match tool_name {
        "read_file" => {
            let mut lines = Vec::with_capacity(2);
            if let Some(path) = value_string_from_keys(&args, &["file_path", "path", "filePath"]) {
                lines.push(format!(
                    "- original_file_path: {}",
                    truncate_to_chars(&normalize_whitespace(&path), 240)
                ));
            }

            if let Some((label, range)) = read_file_range_summary(&args) {
                lines.push(format!("- original_range: {label}={range}"));
            }
            lines
        }
        "execute_command" | "run_command" | "shell" | "bash" => {
            let mut lines = Vec::with_capacity(2);
            if let Some(command) = value_string_from_keys(&args, &["command"]) {
                lines.push(format!(
                    "- original_command: {}",
                    truncate_to_chars(&normalize_whitespace(&command), 720)
                ));
            }
            if let Some(cwd) = value_string_from_keys(&args, &["cwd"]) {
                let cwd = normalize_whitespace(&cwd);
                if !cwd.is_empty() {
                    lines.push(format!("- original_cwd: {}", truncate_to_chars(&cwd, 240)));
                }
            }
            lines
        }
        _ => Vec::new(),
    }
}

fn value_string_from_keys(args: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| args.get(*key).and_then(Value::as_str))
        .map(|value| value.to_string())
}

fn read_file_range_summary(args: &Value) -> Option<(&'static str, String)> {
    let start_line = args.get("startLine").and_then(Value::as_u64);
    let end_line = args.get("endLine").and_then(Value::as_u64);
    if let (Some(start_line), Some(end_line)) = (start_line, end_line) {
        return Some(("lines", format!("{start_line}..{end_line}")));
    }

    let offset = args.get("offset").and_then(Value::as_u64);
    let limit = args.get("limit").and_then(Value::as_u64);
    match (offset, limit) {
        (Some(offset), Some(limit)) if limit > 0 => Some((
            "lines",
            format!("{offset}..{}", offset + limit.saturating_sub(1)),
        )),
        (Some(offset), _) => Some(("offset", offset.to_string())),
        _ => None,
    }
}

fn is_preserved_tool_overflow_stub(text: &str) -> bool {
    let text = text.trim_start();
    if text.starts_with(PRESERVED_TOOL_OVERFLOW_STUB_PREFIX) {
        return text.contains("\n- file_path: ");
    }
    // Legacy formats (older sessions):
    // - "Output preserved for non-compressible tool `..."  (pre-refactor)
    // - "Output preserved for tool `..."                   (new format)
    if (text.starts_with(LEGACY_PRESERVED_TOOL_OVERFLOW_STUB_PREFIX)
        || text.starts_with("Output preserved for tool `"))
        && text.contains("\n- file_path: ")
    {
        return true;
    }
    false
}

/// 把一条已外溢的 tool overflow stub 的 head+tail 预览体收敛为「单行召回锚点」
/// （仅保留 `file_path:` 指针 + 回读提示，丢弃 `Preview (...)` 及其后所有行）。
///
/// 老 stub 的预览在长会话里单调累积（真实案例：800 条 × ~1KB ≈ 849KB），而
/// `file_path` 才是模型精确回读的唯一必要信息——预览只是「首次召回锚点」，一旦
/// 该 stub 已远离当前工作焦点，预览正文的边际价值趋近于 0。收敛后每条从 ~1KB
/// 降到 ~200 字符，召回能力零损失（仍可 read_file 回读原文）。
///
/// 解析失败（无法定位 file_path 或工具名）返回 `None`，保持原文不动，绝不破坏。
/// 对「已经是锚点形态」（不含 `Preview (` 段）的 stub 亦返回 `None`，保证幂等、
/// 不产生 stub→stub 抖动。
fn collapse_overflow_stub_to_anchor(text: &str) -> Option<String> {
    if !is_preserved_tool_overflow_stub(text) {
        return None;
    }
    // 已是锚点形态（无预览段）：幂等，返回 None 表示无需改写。
    if !text.contains("Preview (") {
        return None;
    }
    // 支持新旧两种格式解析 tool_name
    let tool_name = text
        .split_once("non-compressible tool `")
        .and_then(|(_, rest)| rest.split_once('`'))
        .map(|(name, _)| name.to_string())
        .or_else(|| {
            text.split_once("Output preserved for tool `")
                .and_then(|(_, rest)| rest.split_once('`'))
                .map(|(name, _)| name.to_string())
        })?;
    let file_path = text
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
        .map(str::trim)
        .filter(|p| !p.is_empty())?;
    let recall_lines = text
        .lines()
        .map(str::trim_start)
        .filter(|line| line.starts_with("- original_"))
        .map(str::to_string)
        .collect::<Vec<_>>();
    let tool_hint = if tool_name == "read_file" {
        if recall_lines
            .iter()
            .any(|line| line.starts_with("- original_file_path: "))
        {
            "（这是之前读取文件的结果归档；通常无需回读本归档。若必须追原文，优先读取 `original_file_path`。）"
        } else {
            "（这是之前读取文件的结果归档；通常无需重新读取。）"
        }
    } else if tool_name == "execute_command"
        && recall_lines
            .iter()
            .any(|line| line.starts_with("- original_command: "))
    {
        "（这是之前命令输出的归档；通常无需回读本归档，优先依据 `original_command` / `original_cwd` 继续判断。）"
    } else {
        "（如需完整输出可用 read_file 读取。）"
    };
    let mut out = format!(
        "{PRESERVED_TOOL_OVERFLOW_STUB_PREFIX}\n\
         Output preserved for tool `{tool_name}`. Full result saved to session asset:\n\
         - file_path: {file_path}"
    );
    for line in &recall_lines {
        out.push('\n');
        out.push_str(line);
    }
    out.push('\n');
    out.push_str(tool_hint);
    Some(out)
}

/// 将「保护尾窗之外」的 overflow stub 预览体老化折叠为单行锚点。仅作用于已外溢
/// 的 tool stub（`is_preserved_tool_overflow_stub`），不碰原始 tool 结果；尾窗内
/// stub 保留完整 head+tail 预览（当前工作焦点仍需要它的召回上下文）。
///
/// 与预算驱动的组折叠互补：即便某条 stub 所在的组因近端保护未被
/// `fold_early_tool_groups` 折叠，其预览正文也会随对话推进老化收敛，防止历史里
/// 上百条早期 read_file 预览单调累积。
pub(super) fn age_out_overflow_stub_previews(
    messages: &mut [Message],
    keep_recent_user_turns: usize,
) {
    let protected_tail_start = retained_turn_start(messages, keep_recent_user_turns);
    for message in messages.iter_mut().take(protected_tail_start) {
        if message.role != "tool" {
            continue;
        }
        let Value::String(text) = &message.content else {
            continue;
        };
        if let Some(anchor) = collapse_overflow_stub_to_anchor(text) {
            message.content = Value::String(anchor);
        }
    }
}

/// 为外溢内容生成 head+tail 预览。短内容直接全量保留；长内容保留前后各若干行，
/// 中间用占位行折叠，并标注省略的行数。
fn build_overflow_content_preview(content: &str) -> String {
    const HEAD_LINES: usize = 8;
    const TAIL_LINES: usize = 4;
    const MAX_LINE_CHARS: usize = 200;

    let truncate_line = |line: &str| -> String {
        if line.chars().count() > MAX_LINE_CHARS {
            let kept: String = line.chars().take(MAX_LINE_CHARS).collect();
            format!("{kept} …")
        } else {
            line.to_string()
        }
    };

    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let mut out = String::from("Preview (for recall; not exhaustive):\n");
    if total <= HEAD_LINES + TAIL_LINES {
        for line in &lines {
            out.push_str(&truncate_line(line));
            out.push('\n');
        }
    } else {
        for line in &lines[..HEAD_LINES] {
            out.push_str(&truncate_line(line));
            out.push('\n');
        }
        out.push_str(&format!(
            "... [{} line(s) omitted; read the file above for full content] ...\n",
            total - HEAD_LINES - TAIL_LINES
        ));
        for line in &lines[total - TAIL_LINES..] {
            out.push_str(&truncate_line(line));
            out.push('\n');
        }
    }
    out.trim_end().to_string()
}

pub(super) fn is_preserved_user_or_image_stub(text: &str) -> bool {
    parse_preserved_message_stub(text).is_some()
}

fn parse_preserved_message_stub(text: &str) -> Option<(String, String)> {
    if let Some(payload) = text.strip_prefix(PRESERVED_CONTENT_STUB_PREFIX) {
        let value = serde_json::from_str::<Value>(payload).ok()?;
        let kind = value.get("kind")?.as_str()?.to_string();
        let file_path = value.get("file_path")?.as_str()?.to_string();
        return ((kind == "user" || kind == "image") && !file_path.is_empty())
            .then_some((kind, file_path));
    }

    let kind = if text.starts_with("较早的用户图片内容已归档") {
        "image"
    } else if text.starts_with("较早的用户") {
        "user"
    } else {
        return None;
    };
    let file_path = text
        .lines()
        .find_map(|line| line.strip_prefix("归档文件: "))?
        .trim();
    (!file_path.is_empty()).then(|| (kind.to_string(), file_path.to_string()))
}

/// 将内部归档协议转换成模型可理解的上下文说明，同时兼容已经落盘的旧 JSON stub。
pub(in crate::ai) fn normalize_preserved_message_stubs_for_model(messages: &mut [Message]) {
    for message in messages {
        let Value::String(text) = &message.content else {
            continue;
        };
        let Some((kind, file_path)) = parse_preserved_message_stub(text) else {
            continue;
        };
        message.content = Value::String(build_preserved_message_overflow_stub(
            Path::new(&file_path),
            &kind,
        ));
    }
}

fn first_preserved_content_spill_candidate(messages: &[Message], budget: usize) -> Option<usize> {
    let keep_recent_user_turns = keep_recent_user_turns_when_trimming(messages, budget);
    let protected_tail_start = retained_turn_start(messages, keep_recent_user_turns);
    for (idx, message) in messages.iter().enumerate() {
        if idx >= protected_tail_start {
            break;
        }
        if is_system_like_role(&message.role) || message.role == "tool" {
            continue;
        }
        if message.role == "assistant"
            && message
                .tool_calls
                .as_ref()
                .map(|calls| !calls.is_empty())
                .unwrap_or(false)
        {
            continue;
        }

        let text = value_to_string(&message.content);
        if is_preserved_user_or_image_stub(&text) {
            continue;
        }

        // value_to_string 会把图片 base64 折叠成 "[图片]"，无法反映真实体量。
        // 对图片消息改用原始 content 的序列化长度判断是否需要外溢，与「把大图
        // 搬到会话临时文件」的意图一致；普通文本消息仍按 value_to_string 计费。
        let char_count = if message_contains_image(&message.content) {
            message.content.to_string().chars().count()
        } else {
            text.chars().count()
        };
        if message_contains_image(&message.content) && char_count >= IMAGE_OVERFLOW_SPILL_MIN_CHARS
        {
            return Some(idx);
        }
        if message.role == "user" && char_count >= USER_OVERFLOW_SPILL_MIN_CHARS {
            return Some(idx);
        }
    }
    None
}

fn write_preserved_message_overflow_file(
    overflow_dir: &Path,
    message: &Message,
    kind: &str,
) -> Option<PathBuf> {
    let subdir = if kind == "image" {
        PRESERVED_IMAGE_OVERFLOW_DIR
    } else {
        PRESERVED_USER_OVERFLOW_DIR
    };
    let dir = overflow_dir.join(subdir);
    std::fs::create_dir_all(&dir).ok()?;
    let file_name = format!(
        "{}-{}-{}.json",
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ"),
        kind,
        uuid::Uuid::new_v4().simple()
    );
    let path = dir.join(file_name);

    let mut payload = serde_json::Map::new();
    payload.insert("role".to_string(), Value::String(message.role.clone()));
    payload.insert("kind".to_string(), Value::String(kind.to_string()));
    payload.insert("content".to_string(), message.content.clone());
    if let Some(tool_calls) = &message.tool_calls {
        payload.insert(
            "tool_calls".to_string(),
            serde_json::to_value(tool_calls).ok()?,
        );
    }
    if let Some(tool_call_id) = &message.tool_call_id {
        payload.insert(
            "tool_call_id".to_string(),
            Value::String(tool_call_id.clone()),
        );
    }

    let serialized = serde_json::to_string_pretty(&Value::Object(payload)).ok()?;
    std::fs::write(&path, serialized).ok()?;
    Some(path)
}

fn build_preserved_message_overflow_stub(path: &Path, kind: &str) -> String {
    let content_kind = if kind == "image" { "图片" } else { "文本" };
    format!(
        "较早的用户{content_kind}内容已归档，原文未丢失。\n归档文件: {}\n这是一条上下文归档提示，不是用户的新请求。仅当当前任务确实依赖原文时，才使用 read_file 读取该文件。",
        path.display()
    )
}

pub(super) fn try_spill_preserved_message_to_stub(
    messages: &mut [Message],
    overflow_dir: &Path,
    budget: usize,
) -> bool {
    let Some(idx) = first_preserved_content_spill_candidate(messages, budget) else {
        return false;
    };
    let kind = if message_contains_image(&messages[idx].content) {
        "image"
    } else {
        "user"
    };
    let Some(path) = write_preserved_message_overflow_file(overflow_dir, &messages[idx], kind)
    else {
        return false;
    };
    messages[idx].content = Value::String(build_preserved_message_overflow_stub(&path, kind));
    true
}

/// 主动把体量过大的旧 user / 图片消息（保护尾窗之前的）搬到会话临时文件，
/// 原地替换为紧凑 stub。原文零压缩保存在磁盘上，但不再占用每轮请求的 payload。
///
/// 这与预算驱动的循环内 spill 互补：自从图片在预算里只按 [`IMAGE_BUDGET_CHARS`]
/// 名义计费后，单张大图本身不再触发 `messages_total_chars > max_chars`，于是
/// 循环内的 spill 永远不会被调用。这里改为「无论是否超预算，只要旧消息原始
/// 体量超过阈值就外溢」，既保证大图/大段用户原文被零压缩归档，又避免它们污染
/// 后续每一轮请求。最新一轮（保护尾窗内）的 user/图片永不外溢。
pub(super) fn spill_oversized_preserved_messages(
    messages: &mut [Message],
    overflow_dir: &Path,
    budget: usize,
) {
    while try_spill_preserved_message_to_stub(messages, overflow_dir, budget) {}
}

fn structured_tool_output_summary(text: &str, max_chars: usize) -> String {
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return String::new();
    }
    if lines.len() <= 8 {
        let mut out = Vec::new();
        let mut used = 0usize;
        for line in lines
            .into_iter()
            .map(tool_line_signature)
            .filter(|line| !line.is_empty())
        {
            let extra = if out.is_empty() { 0 } else { 1 };
            if used + extra + line.chars().count() > max_chars {
                break;
            }
            used += extra + line.chars().count();
            out.push(line);
        }
        return out.join("\n");
    }

    let mut sections = Vec::new();
    push_section_with_budget(
        &mut sections,
        format!("tool_output_lines: {}", lines.len()),
        max_chars,
    );

    let key_signals = lines
        .iter()
        .filter(|line| is_important_tool_line(line))
        .map(|line| tool_line_signature(line))
        .filter(|line| !line.is_empty())
        .fold(Vec::new(), |mut acc: Vec<String>, line| {
            push_unique_limited_global(&mut acc, line, 4);
            acc
        });
    if !key_signals.is_empty() {
        push_section_with_budget(
            &mut sections,
            format!("key_signals: {}", key_signals.join(" || ")),
            max_chars,
        );
    }

    let path_hints = lines
        .iter()
        .flat_map(|line| extract_path_like_tokens(line))
        .fold(Vec::new(), |mut acc: Vec<String>, token| {
            push_unique_limited_global(&mut acc, token, 4);
            acc
        });
    if !path_hints.is_empty() {
        push_section_with_budget(
            &mut sections,
            format!("paths: {}", path_hints.join(", ")),
            max_chars,
        );
    }

    let chunk_size = (lines.len() / 3).max(1);
    let mut chunk_summaries = Vec::new();
    for (chunk_index, chunk) in lines.chunks(chunk_size).take(3).enumerate() {
        let chunk_summary = summarize_tool_chunk(chunk_index + 1, chunk);
        if !chunk_summary.is_empty() {
            chunk_summaries.push(chunk_summary);
        }
    }
    if !chunk_summaries.is_empty() {
        push_section_with_budget(
            &mut sections,
            format!("chunks:\n- {}", chunk_summaries.join("\n- ")),
            max_chars,
        );
    }

    sections.join("\n")
}

fn push_section_with_budget(target: &mut Vec<String>, section: String, max_chars: usize) {
    if section.is_empty() {
        return;
    }
    let current = if target.is_empty() {
        0
    } else {
        target.join("\n").chars().count() + 1
    };
    if current + section.chars().count() <= max_chars {
        target.push(section);
        return;
    }
    if target.is_empty() {
        target.push(summarize_text(&section, max_chars));
    }
}

fn summarize_tool_chunk(chunk_index: usize, chunk: &[&str]) -> String {
    if chunk.is_empty() {
        return String::new();
    }
    let mut picks: Vec<String> = Vec::new();
    let first = tool_line_signature(chunk[0]);
    if !first.is_empty() {
        push_unique_limited_global(&mut picks, first, 4);
    }
    for line in chunk
        .iter()
        .filter(|line| is_important_tool_line(line))
        .take(2)
    {
        let sig = tool_line_signature(line);
        if !sig.is_empty() {
            push_unique_limited_global(&mut picks, sig, 4);
        }
    }
    if let Some(last) = chunk.last() {
        let last = tool_line_signature(last);
        if !last.is_empty() {
            push_unique_limited_global(&mut picks, last, 4);
        }
    }
    if picks.is_empty() {
        return String::new();
    }
    format!("chunk_{chunk_index}: {}", picks.join(" | "))
}

pub(super) fn tool_line_signature(line: &str) -> String {
    let normalized = normalize_whitespace(line);
    if normalized.is_empty() {
        return String::new();
    }
    let words = normalized.split_whitespace().collect::<Vec<_>>();
    if words.len() <= 18 {
        return normalized;
    }

    let head = words.iter().take(12).copied().collect::<Vec<_>>().join(" ");
    let mut notable_tail = Vec::new();
    for word in words.iter().rev() {
        let token = word.trim_matches(|ch: char| {
            ch.is_whitespace()
                || matches!(
                    ch,
                    ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\''
                )
        });
        if token.is_empty() {
            continue;
        }
        let looks_notable = token.contains('/')
            || token.contains('.')
            || token.chars().any(|ch| ch.is_ascii_digit())
            || looks_like_error_code(token);
        if looks_notable {
            push_unique_limited_global(&mut notable_tail, token.to_string(), 4);
        }
    }
    notable_tail.reverse();
    if notable_tail.is_empty() {
        return head;
    }
    format!("{head} | {}", notable_tail.join(" "))
}

fn is_important_tool_line(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.contains("error")
        || lower.contains("failed")
        || lower.contains("panic")
        || lower.contains("exception")
        || lower.contains("timeout")
        || lower.contains("not found")
        || lower.contains("traceback")
        || lower.contains("exit code")
        || lower.contains("warning")
        || lower.contains("completed")
        || lower.contains("success")
}

fn extract_path_like_tokens(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in line.split_whitespace() {
        let token = raw.trim_matches(|ch: char| {
            ch.is_whitespace()
                || matches!(
                    ch,
                    ',' | ';' | ':' | '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\''
                )
        });
        if token.len() > 160 || token.is_empty() {
            continue;
        }
        if token.starts_with("http://") || token.starts_with("https://") {
            continue;
        }
        let looks_like_path = token.contains('/')
            || [
                ".rs", ".tsx", ".ts", ".jsx", ".js", ".py", ".go", ".java", ".kt", ".swift", ".c",
                ".cc", ".cpp", ".h", ".hpp", ".toml", ".yaml", ".yml", ".json",
            ]
            .iter()
            .any(|suffix| token.ends_with(suffix));
        if looks_like_path {
            push_unique_limited_global(&mut out, token.to_string(), 8);
        }
    }
    out
}

fn looks_like_error_code(token: &str) -> bool {
    let bytes = token.as_bytes();
    bytes.len() == 5 && bytes[0] == b'E' && bytes[1..].iter().all(|byte| byte.is_ascii_digit())
}

fn push_unique_limited_global(target: &mut Vec<String>, value: String, max_items: usize) {
    if value.is_empty() || target.iter().any(|item| item == &value) || target.len() >= max_items {
        return;
    }
    target.push(value);
}

pub(super) fn build_persisted_summary_text(messages: &[Message], max_chars: usize) -> String {
    #[derive(Default, Clone)]
    struct TurnSummary {
        topic_key: String,
        topic_label: String,
        user: String,
        user_key: String,
        assistant_final: String,
        tool_names: Vec<String>,
        tool_highlights: Vec<String>,
        count: usize,
    }

    fn normalize_semantic_key(s: &str) -> String {
        let mut out = String::new();
        for ch in s.chars() {
            let is_cjk = ('\u{4E00}'..='\u{9FFF}').contains(&ch);
            if is_cjk || ch.is_ascii_alphanumeric() {
                out.push(ch.to_ascii_lowercase());
                continue;
            }
            if ch.is_whitespace() {
                out.push(' ');
            }
        }
        normalize_whitespace(&out)
    }

    fn extract_topic_from_text(text: &str) -> Option<(String, String)> {
        fn trim_punct(s: &str) -> &str {
            s.trim_matches(|ch: char| {
                ch.is_whitespace()
                    || matches!(
                        ch,
                        ',' | '.'
                            | ';'
                            | ':'
                            | '!'
                            | '?'
                            | '('
                            | ')'
                            | '['
                            | ']'
                            | '{'
                            | '}'
                            | '<'
                            | '>'
                            | '"'
                            | '\''
                            | '`'
                    )
            })
        }

        fn candidate_file_token(token: &str) -> Option<&str> {
            let token = trim_punct(token);
            if token.is_empty() || token.len() > 96 {
                return None;
            }
            if token.starts_with("http://") || token.starts_with("https://") {
                return None;
            }
            let token = token.split('#').next().unwrap_or(token);
            let token = token.split('?').next().unwrap_or(token);
            let token = token.split_once(':').map(|(a, _)| a).unwrap_or(token);
            let suffixes = [
                ".rs", ".tsx", ".ts", ".jsx", ".js", ".py", ".go", ".java", ".kt", ".swift", ".c",
                ".cc", ".cpp", ".h", ".hpp", ".toml", ".yaml", ".yml", ".json",
            ];
            if suffixes.iter().any(|suf| token.ends_with(suf)) {
                return Some(token);
            }
            None
        }

        fn basename(path: &str) -> &str {
            path.rsplit('/').next().unwrap_or(path)
        }

        fn find_error_code(text: &str) -> Option<String> {
            let bytes = text.as_bytes();
            let mut i = 0usize;
            while i + 5 <= bytes.len() {
                if bytes[i] == b'E'
                    && bytes[i + 1].is_ascii_digit()
                    && bytes[i + 2].is_ascii_digit()
                    && bytes[i + 3].is_ascii_digit()
                    && bytes[i + 4].is_ascii_digit()
                {
                    let code = &text[i..i + 5];
                    return Some(code.to_string());
                }
                i += 1;
            }
            None
        }

        if let Some(code) = find_error_code(text) {
            return Some((code.to_ascii_lowercase(), code));
        }

        for raw in text.split_whitespace() {
            if let Some(token) = candidate_file_token(raw) {
                let label = basename(token).to_string();
                return Some((token.to_ascii_lowercase(), label));
            }
            let token = trim_punct(raw);
            if token.contains('/')
                && token.len() <= 96
                && token.chars().any(|c| c == '.')
                && !token.starts_with("http://")
                && !token.starts_with("https://")
            {
                let label = basename(token).to_string();
                return Some((token.to_ascii_lowercase(), label));
            }
        }

        None
    }

    fn push_unique_limited(target: &mut Vec<String>, value: String, max_items: usize) {
        if value.is_empty() || target.iter().any(|item| item == &value) || target.len() >= max_items
        {
            return;
        }
        target.push(value);
    }

    fn tool_highlight(text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        let lowered = text.to_ascii_lowercase();
        let important = lowered.contains("error")
            || lowered.contains("failed")
            || lowered.contains("panic")
            || lowered.contains("exception")
            || lowered.contains("[error]");
        if important {
            return extract_important_lines(text, 120);
        }
        summarize_text(&normalize_whitespace(text), 80)
    }

    fn extract_important_lines(text: &str, target_chars: usize) -> String {
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.is_empty() {
            return String::new();
        }
        let mut selected: Vec<&str> = Vec::new();
        let mut chars = 0usize;
        for line in &lines {
            let lowered = line.to_ascii_lowercase();
            let is_key = lowered.contains("error")
                || lowered.contains("failed")
                || lowered.contains("panic")
                || lowered.contains("exception")
                || lowered.contains("not found")
                || lowered.contains("timeout");
            if is_key || selected.is_empty() {
                if chars + line.trim().chars().count() + 2 > target_chars {
                    if selected.is_empty() {
                        let trimmed = line.trim();
                        selected.push(trimmed);
                    }
                    break;
                }
                selected.push(line.trim());
                chars += line.trim().chars().count() + 2;
            }
        }
        let result = selected.join("; ");
        if result.chars().count() <= target_chars {
            return result;
        }
        keep_ends_by_chars(&result, target_chars)
    }

    fn finalize_turn(turns: &mut Vec<TurnSummary>, current: &mut TurnSummary) {
        if current.user.trim().is_empty()
            && current.assistant_final.trim().is_empty()
            && current.tool_names.is_empty()
            && current.tool_highlights.is_empty()
        {
            *current = TurnSummary::default();
            return;
        }
        if current.count == 0 {
            current.count = 1;
        }
        turns.push(current.clone());
        *current = TurnSummary::default();
    }

    fn merge_turns(mut turns: Vec<TurnSummary>) -> Vec<TurnSummary> {
        let mut out: Vec<TurnSummary> = Vec::with_capacity(turns.len());
        for turn in turns.drain(..) {
            if let Some(last) = out.last_mut()
                && !turn.user_key.is_empty()
                && last.user_key == turn.user_key
            {
                last.count = last.count.saturating_add(turn.count.max(1));
                if last.topic_label.is_empty() && !turn.topic_label.is_empty() {
                    last.topic_label = turn.topic_label;
                    last.topic_key = turn.topic_key;
                }
                if !turn.assistant_final.is_empty()
                    && turn.assistant_final != last.assistant_final
                    && last.assistant_final.chars().count() < 200
                {
                    if last.assistant_final.is_empty() {
                        last.assistant_final = turn.assistant_final;
                    } else {
                        last.assistant_final = summarize_text(
                            &format!("{} / {}", last.assistant_final, turn.assistant_final),
                            250,
                        );
                    }
                }
                for name in turn.tool_names {
                    push_unique_limited(&mut last.tool_names, name, 6);
                }
                for h in turn.tool_highlights {
                    push_unique_limited(&mut last.tool_highlights, h, 3);
                }
                continue;
            }
            out.push(turn);
        }
        out
    }

    fn render_line(turn: &TurnSummary) -> String {
        let mut line = String::new();
        if turn.count > 1 {
            line.push_str(&format!("重复×{} ", turn.count));
        }
        if !turn.topic_label.is_empty() {
            line.push_str("主题: ");
            line.push_str(&turn.topic_label);
            line.push_str(" | ");
        }
        if !turn.user.is_empty() {
            line.push_str("用户: ");
            line.push_str(&turn.user);
        }
        if !turn.assistant_final.is_empty() {
            if !line.is_empty() {
                line.push_str(" | ");
            }
            line.push_str("结论: ");
            line.push_str(&turn.assistant_final);
        }
        if !turn.tool_names.is_empty() {
            if !line.is_empty() {
                line.push_str(" | ");
            }
            line.push_str("工具: ");
            line.push_str(&turn.tool_names.join(", "));
        }
        if !turn.tool_highlights.is_empty() {
            if !line.is_empty() {
                line.push_str(" | ");
            }
            line.push_str("关键: ");
            line.push_str(&turn.tool_highlights.join("；"));
        }
        line
    }

    fn render_known_tool_line(turn: &TurnSummary) -> Option<String> {
        if turn.tool_names.is_empty() {
            return None;
        }
        let mut line = String::new();
        line.push_str("- ");
        line.push_str(&turn.tool_names.join(", "));
        if !turn.topic_label.is_empty() {
            line.push_str(" @ ");
            line.push_str(&turn.topic_label);
        }
        let conclusion = if !turn.tool_highlights.is_empty() {
            turn.tool_highlights.join("；")
        } else {
            turn.assistant_final.clone()
        };
        if !conclusion.is_empty() {
            line.push_str(" => ");
            line.push_str(&conclusion);
        }
        Some(line)
    }

    fn push_line_with_budget(lines: &mut Vec<String>, mut line: String, max_chars: usize) -> bool {
        let line_chars = line.chars().count();
        if lines.is_empty() {
            if line_chars > max_chars {
                lines.push(summarize_text(&line, max_chars));
                return true;
            }
            lines.push(line);
            return true;
        }
        let current_len = lines.join("\n").chars().count();
        let remaining = max_chars.saturating_sub(current_len + 1);
        if remaining < 30 {
            return false;
        }
        if line_chars > remaining {
            line = summarize_text(&line, remaining);
        }
        if line.chars().count() <= remaining {
            lines.push(line);
            true
        } else {
            false
        }
    }

    let mut initial_goal = String::new();
    let mut pre_summary_lines: Vec<String> = Vec::new();
    let mut turns: Vec<TurnSummary> = Vec::new();
    let mut current = TurnSummary::default();

    for message in messages {
        let text = normalize_whitespace(&value_to_string(&message.content));
        match message.role.as_str() {
            role if role == ROLE_INTERNAL_NOTE => {
                if let Some(body) = automatic_summary_body(&text) {
                    let normalized =
                        summarize_text(&strip_nested_prior_summary_prefixes(body), 400);
                    if !normalized.is_empty() {
                        push_unique_limited(
                            &mut pre_summary_lines,
                            format!("- 更早摘要: {normalized}"),
                            3,
                        );
                    }
                }
            }
            role if is_system_like_role(role) => {}
            "user" => {
                finalize_turn(&mut turns, &mut current);
                if initial_goal.is_empty() {
                    initial_goal = summarize_text(&text, 240);
                }
                current.user = summarize_text(&text, 200);
                current.user_key = truncate_to_chars(&normalize_semantic_key(&text), 160);
                if let Some((k, label)) = extract_topic_from_text(&text) {
                    current.topic_key = k;
                    current.topic_label = label;
                }
                if current.count == 0 {
                    current.count = 1;
                }
            }
            "assistant" => {
                if !text.is_empty() {
                    current.assistant_final = summarize_text(&text, 250);
                    if current.topic_key.is_empty() {
                        if let Some((k, label)) = extract_topic_from_text(&text) {
                            current.topic_key = k;
                            current.topic_label = label;
                        }
                    }
                }
                if let Some(tool_calls) = &message.tool_calls {
                    for tool_call in tool_calls {
                        push_unique_limited(
                            &mut current.tool_names,
                            tool_call.function.name.clone(),
                            6,
                        );
                        if current.topic_key.is_empty() {
                            current.topic_key = tool_call.function.name.to_ascii_lowercase();
                            current.topic_label = tool_call.function.name.clone();
                        }
                    }
                }
            }
            "tool" => {
                let h = tool_highlight(&text);
                if !h.is_empty() {
                    push_unique_limited(&mut current.tool_highlights, h.clone(), 3);
                    if current.topic_key.is_empty() {
                        if let Some((k, label)) = extract_topic_from_text(&h) {
                            current.topic_key = k;
                            current.topic_label = label;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    finalize_turn(&mut turns, &mut current);

    let recent_count = turns.len().min(3);
    let recent_turns: Vec<TurnSummary> = turns
        .iter()
        .rev()
        .take(recent_count)
        .rev()
        .cloned()
        .collect();

    let pending_tasks: Vec<String> = turns
        .iter()
        .rev()
        .take(2)
        .filter(|t| !t.user.is_empty() && t.assistant_final.is_empty())
        .map(|t| t.user.clone())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    let merged = merge_turns(turns);
    let mut known_tool_lines: Vec<String> = Vec::new();
    for t in &merged {
        if let Some(line) = render_known_tool_line(t)
            && !known_tool_lines.iter().any(|existing| existing == &line)
            && known_tool_lines.len() < 10
        {
            known_tool_lines.push(line);
        }
    }
    let reserved_tool_chars = if known_tool_lines.is_empty() {
        0
    } else {
        let tool_blob = format!("已知工具结论:\n{}", known_tool_lines.join("\n"));
        tool_blob.chars().count().min(max_chars / 3)
    };
    let body_budget = max_chars
        .saturating_sub(reserved_tool_chars)
        .max(max_chars / 2);
    let mut lines: Vec<String> = Vec::new();
    if !initial_goal.is_empty()
        && !push_line_with_budget(&mut lines, format!("初始目标: {initial_goal}"), body_budget)
    {
        return summarize_text(&lines.join("\n"), max_chars);
    }
    for s in pre_summary_lines.into_iter().take(3) {
        if !push_line_with_budget(&mut lines, s, body_budget) {
            return summarize_text(&lines.join("\n"), max_chars);
        }
    }
    for t in &merged {
        if !push_line_with_budget(&mut lines, format!("- {}", render_line(t)), body_budget) {
            break;
        }
    }

    if !known_tool_lines.is_empty() {
        let _ = push_line_with_budget(&mut lines, "已知工具结论:".to_string(), max_chars);
        for line in known_tool_lines {
            if !push_line_with_budget(&mut lines, line, max_chars) {
                break;
            }
        }
    }

    if !recent_turns.is_empty() {
        let _ = push_line_with_budget(&mut lines, String::new(), max_chars);
        let _ = push_line_with_budget(&mut lines, "当前工作:".to_string(), max_chars);
        for t in &recent_turns {
            let mut parts = Vec::new();
            if !t.topic_label.is_empty() {
                parts.push(format!("主题: {}", t.topic_label));
            }
            if !t.user.is_empty() {
                parts.push(format!("用户: {}", t.user));
            }
            if !t.assistant_final.is_empty() {
                parts.push(format!("助手: {}", t.assistant_final));
            }
            if !t.tool_names.is_empty() {
                parts.push(format!("工具: {}", t.tool_names.join(", ")));
            }
            if !t.tool_highlights.is_empty() {
                parts.push(format!("关键: {}", t.tool_highlights.join("；")));
            }
            let line = format!("- {}", parts.join(" | "));
            if !push_line_with_budget(&mut lines, summarize_text(&line, 600), max_chars) {
                break;
            }
        }
    }

    if !pending_tasks.is_empty() {
        let _ = push_line_with_budget(&mut lines, String::new(), max_chars);
        let _ = push_line_with_budget(&mut lines, "待办任务:".to_string(), max_chars);
        for task in &pending_tasks {
            if !push_line_with_budget(
                &mut lines,
                format!("- {}", summarize_text(task, 300)),
                max_chars,
            ) {
                break;
            }
        }
    }

    summarize_text(&lines.join("\n"), max_chars)
}
