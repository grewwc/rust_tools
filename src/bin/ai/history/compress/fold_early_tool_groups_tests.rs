use super::*;
use crate::ai::types::{FunctionCall, ToolCall};
use rustc_hash::FxHashSet;

fn msg(role: &str, content: &str) -> Message {
    Message {
        role: role.to_string(),
        content: Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn assistant_call(id: &str, name: &str) -> Message {
    Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: "{}".to_string(),
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

/// 构造：system + user + N 个 (assistant tool_calls + tool 结果) 组，全部在
/// 同一个 user 轮内（只有一条 user 消息）——正是"臃肿全堆在当前轮"的场景。
fn single_turn_with_groups(n: usize, tool_result_chars: usize) -> Vec<Message> {
    let mut messages = vec![msg("system", "system prompt"), msg("user", "干活")];
    for i in 0..n {
        let id = format!("call-{i}");
        messages.push(assistant_call(&id, "text_grep"));
        messages.push(tool_result(&id, &"x".repeat(tool_result_chars)));
    }
    messages
}

fn assert_tool_pairs_consistent(messages: &[Message]) {
    let mut assistant_ids: FxHashSet<String> = FxHashSet::default();
    for m in messages {
        if m.role == "assistant"
            && let Some(calls) = &m.tool_calls
        {
            for c in calls {
                assistant_ids.insert(c.id.clone());
            }
        }
    }
    let mut tool_ids: FxHashSet<String> = FxHashSet::default();
    for m in messages {
        if m.role == "tool"
            && let Some(id) = &m.tool_call_id
        {
            tool_ids.insert(id.clone());
        }
    }
    assert_eq!(
        assistant_ids, tool_ids,
        "every assistant.tool_calls id must have a paired tool message and vice versa"
    );
}

fn archive_file_path_from_text(text: &str) -> String {
    text.lines()
        .find_map(|line| line.trim().strip_prefix("- archive_file_path: "))
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .expect("text should contain archive_file_path")
        .to_string()
}

#[test]
fn folds_early_groups_in_a_single_bloated_turn() {
    let messages = single_turn_with_groups(10, 2_000);
    let before = messages_total_chars(&messages);

    let (folded, folded_groups) = fold_early_tool_groups(&messages, 4, None, &FxHashSet::default());

    // 10 组各 1 条 tool 结果 → 10 条 tool 消息。虽然 keep_recent_groups=4，但
    // 最近完整 4 组逐字保留，最早 6 组折叠。
    assert_eq!(folded_groups, 6);
    let after = messages_total_chars(&folded);
    assert!(
        after < before,
        "folding must reduce size: {after} !< {before}"
    );
    assert_tool_pairs_consistent(&folded);
}

#[test]
fn preserves_user_message_verbatim() {
    let messages = single_turn_with_groups(8, 1_500);
    let (folded, _) = fold_early_tool_groups(&messages, 4, None, &FxHashSet::default());

    let user = folded
        .iter()
        .find(|m| m.role == "user")
        .expect("user message must survive");
    assert_eq!(value_to_string(&user.content), "干活");
}

#[test]
fn keeps_recent_groups_verbatim() {
    let messages = single_turn_with_groups(8, 1_500);
    let (folded, _) = fold_early_tool_groups(&messages, 4, None, &FxHashSet::default());

    // 8 组各 1 条 tool 结果。按完整组保护最近 4 组，最早 4 组折叠为 stub。
    let full_tool_results = folded
        .iter()
        .filter(|m| m.role == "tool" && value_to_string(&m.content) == "x".repeat(1_500))
        .count();
    assert_eq!(full_tool_results, 4);
}

#[test]
fn no_op_when_group_count_within_keep_window() {
    let messages = single_turn_with_groups(3, 1_000);
    let (folded, folded_groups) = fold_early_tool_groups(&messages, 4, None, &FxHashSet::default());

    assert_eq!(folded_groups, 0);
    assert_eq!(folded.len(), messages.len());
}

/// 组原子性不变量：即使调用方要求最激进的 `keep_recent_groups=0`，折叠也必须
/// 保留最近完整工具组，而不是按扁平 tool 消息数从并行批次中间切开。否则模型会
/// 看到同批调用的一半结果，误以为另一半需要重跑。
#[test]
fn fold_never_crosses_recent_tool_message_protection_window() {
    let messages = single_turn_with_groups(10, 1_200);

    // keep_recent_groups=0 表面上要折叠全部 10 组。
    let (folded, folded_groups) = fold_early_tool_groups(&messages, 0, None, &FxHashSet::default());

    // 每组 1 条 tool 结果；调用方要求保留 0 组，因此 10 组都可折叠。
    assert_eq!(folded_groups, 10);
    let full_tool_results = folded
        .iter()
        .filter(|m| m.role == "tool" && value_to_string(&m.content) == "x".repeat(1_200))
        .count();
    assert_eq!(
        full_tool_results, 0,
        "调用方要求保留 0 组时，不应留下任何原始 tool 结果"
    );
    assert_tool_pairs_consistent(&folded);
}

#[test]
fn stub_preserves_file_path_recall_anchor() {
    let mut messages = vec![msg("system", "s"), msg("user", "干活")];
    // 早期一组：read_file 结果已外溢，含 file_path 指针，必须在 stub 中保留。
    messages.push(assistant_call("call-old", "read_file"));
    messages.push(tool_result(
            "call-old",
            "Output preserved for non-compressible tool `read_file`.\n- file_path: /tmp/session/xyz.txt\n- use read_file to inspect exact content.",
        ));
    // 追加足够多的近端组把上面那组挤进折叠区。
    for i in 0..6 {
        let id = format!("call-{i}");
        messages.push(assistant_call(&id, "text_grep"));
        messages.push(tool_result(&id, "recent"));
    }

    let (folded, folded_groups) = fold_early_tool_groups(&messages, 4, None, &FxHashSet::default());
    assert!(folded_groups >= 1);
    let stub_text: String = folded
        .iter()
        .filter(|m| m.role == ROLE_INTERNAL_NOTE)
        .map(|m| value_to_string(&m.content))
        .collect();
    assert!(
        stub_text.contains("/tmp/session/xyz.txt"),
        "folded stub must retain the file_path recall anchor, got: {stub_text}"
    );
}

#[test]
fn folded_read_file_group_keeps_preview_and_original_target_anchor() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-read-preview-fold-{}", uuid::Uuid::new_v4()));
    let mut messages = vec![msg("system", "s"), msg("user", "排查 prompt")];
    messages.push(assistant_call_args(
        "read-prompt",
        "read_file",
        r#"{"file_path":"src/bin/ai/prompt.rs","offset":1,"limit":220}"#,
    ));
    messages.push(tool_result(
        "read-prompt",
        "1\tuse std::{\n\
         2\t    fs,\n\
         3\t    io::{self, BufRead},\n\
         4\t    path::{Path, PathBuf},\n\
         5\t};\n\
         110\tpub(super) fn read_multi_line(&mut self) -> io::Result<Option<String>> {\n\
         111\t    use std::io::IsTerminal;\n\
         112\t    if !io::stdout().is_terminal() || !io::stdin().is_terminal() {\n\
         113\t        return self.read_multi_line_no_tty();\n\
         114\t    }\n\
         115\t    self.read_multi_line_tui()\n\
         116\t}\n",
    ));
    for i in 0..4 {
        let id = format!("later-{i}");
        messages.push(assistant_call(&id, "text_grep"));
        messages.push(tool_result(&id, "later"));
    }

    let (folded, folded_groups) = fold_early_tool_groups(
        &messages,
        4,
        Some(overflow_dir.as_path()),
        &FxHashSet::default(),
    );
    assert_eq!(folded_groups, 1);
    let stub = folded
        .iter()
        .find(|message| {
            message.role == ROLE_INTERNAL_NOTE
                && value_to_string(&message.content).contains(COMPRESSED_TOOL_EVIDENCE_MARKER)
        })
        .map(|message| value_to_string(&message.content))
        .expect("folded read_file group should become evidence note");

    assert!(stub.contains("preview:"), "{stub}");
    assert!(
        stub.contains("pub(super) fn read_multi_line(&mut self) -> io::Result<Option<String>>"),
        "{stub}"
    );
    assert!(
        stub.contains("- original_file_path: src/bin/ai/prompt.rs"),
        "{stub}"
    );
    assert!(stub.contains("- original_range: lines=1..220"), "{stub}");
    assert!(stub.contains("- archive_file_path:"), "{stub}");
    assert!(
        !stub.contains("=> - file_path: "),
        "folded recall should not surface archive file as primary file_path anchor: {stub}"
    );

    let _ = std::fs::remove_dir_all(overflow_dir);
}

/// 一级外溢 stub 已包含（或可由 tool call 重建）原始调用参数时，二级工具组折叠
/// 也必须保留它们。否则 history 只剩不可辨识的内部归档路径，模型会把它当源码
/// 回读，导致「压缩产物不存在 / 源码消失」的错误判断。
#[test]
fn folded_archived_precision_tools_keep_original_invocation_anchors() {
    let mut messages = vec![msg("system", "s"), msg("user", "排查问题")];
    let cases = [
        (
            "read",
            "read_file",
            r#"{"file_path":"src/bin/ai/driver/turn_runtime/orchestrator.rs","offset":120,"limit":40}"#,
        ),
        (
            "command",
            "execute_command",
            r#"{"command":"git status --short","cwd":"/repo"}"#,
        ),
        ("list", "tree", r#"{"path":"src/bin/ai"}"#),
    ];
    for (id, name, arguments) in cases {
        messages.push(assistant_call_args(id, name, arguments));
        messages.push(tool_result(
            id,
            &format!(
                "[[PRESERVED_TOOL_OVERFLOW_STUB_V1]]\n\
                 Output preserved for tool `{name}`. Full result saved to session asset:\n\
                 - file_path: /tmp/session/{id}.txt"
            ),
        ));
    }
    for i in 0..4 {
        let id = format!("later-{i}");
        messages.push(assistant_call(&id, "text_grep"));
        messages.push(tool_result(&id, "recent"));
    }

    let (folded, folded_groups) = fold_early_tool_groups(&messages, 4, None, &FxHashSet::default());
    assert_eq!(folded_groups, 3);
    let folded_text = folded
        .iter()
        .filter(|message| message.role == ROLE_INTERNAL_NOTE)
        .map(|message| value_to_string(&message.content))
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        folded_text
            .contains("- original_file_path: src/bin/ai/driver/turn_runtime/orchestrator.rs"),
        "{folded_text}"
    );
    assert!(
        folded_text.contains("- original_range: lines=120..159"),
        "{folded_text}"
    );
    assert!(
        folded_text.contains("- original_command: git status --short"),
        "{folded_text}"
    );
    assert!(
        folded_text.contains("- original_cwd: /repo"),
        "{folded_text}"
    );
    assert!(
        folded_text.contains("- original_path: src/bin/ai"),
        "{folded_text}"
    );
}

/// 同一 user turn 内工具组过多时，`cargo test` 一类命令可能离开最近组保护窗。
/// 折叠后仍必须能看到失败结论和关键报错，并通过 `file_path` 读取完整日志。
#[test]
fn folded_command_failure_keeps_diagnostics_and_full_output_pointer() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-command-fold-{}", uuid::Uuid::new_v4()));
    let command_output = "Exit code: 101\n\
        Checking rust_tools v0.1.0 (/repo)\n\
        error[E0425]: cannot find value `missing` in this scope\n\
        error: could not compile `rust_tools` (bin \"a\") due to 1 previous error\n\
        test result: FAILED. 0 passed; 1 failed";
    let mut messages = vec![msg("system", "s"), msg("user", "修复编译失败")];
    messages.push(assistant_call("command", "execute_command"));
    messages.push(tool_result("command", command_output));
    // 将命令组推出最近 4 组保护窗，模拟一轮内大量 read/search 后触发 LLM 摘要。
    for i in 0..4 {
        let id = format!("later-{i}");
        messages.push(assistant_call(&id, "text_grep"));
        messages.push(tool_result(&id, "later"));
    }

    let (folded, folded_groups) = fold_early_tool_groups(
        &messages,
        4,
        Some(overflow_dir.as_path()),
        &FxHashSet::default(),
    );
    assert_eq!(folded_groups, 1);
    let stub = folded
        .iter()
        .find(|message| {
            message.role == ROLE_INTERNAL_NOTE
                && value_to_string(&message.content).contains("execute_command")
        })
        .map(|message| value_to_string(&message.content))
        .expect("command group should be folded into a recall stub");
    assert!(stub.contains("Exit code: 101"), "{stub}");
    assert!(stub.contains("error[E0425]"), "{stub}");
    assert!(stub.contains("could not compile"), "{stub}");
    let path = stub
        .lines()
        .find_map(|line| line.trim().strip_prefix("- file_path: "))
        .expect("folded command must retain a full-output file path");
    assert_eq!(
        std::fs::read_to_string(path).expect("archived command output should be readable"),
        command_output
    );

    let _ = std::fs::remove_dir_all(overflow_dir);
}

fn assistant_call_with_reasoning(id: &str, name: &str, reasoning: &str) -> Message {
    let mut m = assistant_call(id, name);
    m.reasoning_content = Some(reasoning.to_string());
    m
}

fn assistant_plain_with_reasoning(reasoning: &str) -> Message {
    Message {
        role: "assistant".to_string(),
        content: Value::String("答复".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: Some(reasoning.to_string()),
    }
}

/// 跨轮滑窗：带 tool_calls 的 assistant reasoning 只保留最近
/// `KEEP_RECENT_TOOL_CALL_REASONING` 条，更早的置 None；纯回答 reasoning 只留最近一条。
#[test]
fn keeps_only_recent_tool_call_reasoning_across_turns() {
    assert_eq!(KEEP_RECENT_TOOL_CALL_REASONING, 3);

    let mut messages = vec![
        msg("system", "s"),
        msg("user", "干活"),
        // 早期纯回答 reasoning：非最近一条，应被丢弃。
        assistant_plain_with_reasoning("early-plain"),
    ];
    // 5 组带 tool_calls 的 reasoning：rank 0/1 应丢弃，rank 2/3/4 保留。
    for i in 0..5 {
        let id = format!("call-{i}");
        messages.push(assistant_call_with_reasoning(
            &id,
            "text_grep",
            &format!("tc-{i}"),
        ));
        messages.push(tool_result(&id, "r"));
    }
    // 最近一条纯回答 reasoning：应保留。
    messages.push(assistant_plain_with_reasoning("final-plain"));

    keep_only_recent_reasoning_content(&mut messages);

    // 用 tool_call id 定位 tool-call reasoning。
    let tc_reasoning = |id: &str| -> Option<String> {
        messages
            .iter()
            .find(|m| {
                m.tool_calls
                    .as_ref()
                    .map(|calls| calls.iter().any(|c| c.id == id))
                    .unwrap_or(false)
            })
            .and_then(|m| m.reasoning_content.clone())
    };
    assert_eq!(
        tc_reasoning("call-0"),
        None,
        "rank 0 tool-call reasoning must be dropped"
    );
    assert_eq!(
        tc_reasoning("call-1"),
        None,
        "rank 1 tool-call reasoning must be dropped"
    );
    assert_eq!(tc_reasoning("call-2").as_deref(), Some("tc-2"));
    assert_eq!(tc_reasoning("call-3").as_deref(), Some("tc-3"));
    assert_eq!(tc_reasoning("call-4").as_deref(), Some("tc-4"));

    // 纯回答 reasoning：只保留最近一条（final-plain），早期一条置 None。
    let plain_reasonings: Vec<Option<String>> = messages
        .iter()
        .filter(|m| m.role == "assistant" && m.tool_calls.is_none())
        .map(|m| m.reasoning_content.clone())
        .collect();
    assert_eq!(
        plain_reasonings,
        vec![None, Some("final-plain".to_string())]
    );
}

#[test]
fn exact_replay_reasoning_survives_recent_reasoning_window() {
    let mut messages = Vec::new();
    for i in 0..5 {
        let original = assistant_call_with_reasoning(
            &format!("glm-call-{i}"),
            "read_file",
            &format!("glm-reasoning-{i}"),
        );
        messages.push(sanitize_message_for_persisted_history_for_model(
            "glm-5.2-opencode",
            &original,
        ));
    }

    keep_only_recent_reasoning_content(&mut messages);

    assert!(
        messages
            .iter()
            .all(|message| message.reasoning_content.is_some()),
        "exact replay 模型的跨工具 continuation state 不能被通用三轮窗口裁掉"
    );
}

#[test]
fn path_c_preserves_exact_replay_reasoning_verbatim() {
    let model = "glm-5.2-opencode";
    let raw_reasoning = "reasoning-state-".repeat(64);
    let assistant = sanitize_message_for_persisted_history_for_model(
        model,
        &assistant_call_with_reasoning("glm-call", "read_file", &raw_reasoning),
    );
    let encoded = assistant
        .reasoning_content
        .clone()
        .expect("exact-replay reasoning should be encoded");
    assert!(encoded.starts_with(PERSISTED_REASONING_REPLAY_PREFIX));

    // Path C 的单字段上限不能把 marker+payload 当普通 reasoning 截断。
    let mut per_field_capped = vec![assistant.clone(), tool_result("glm-call", "ok")];
    assert!(emergency_cap_messages_to_fit(
        &mut per_field_capped,
        usize::MAX,
        160,
        None,
        &FxHashSet::default(),
    ));
    assert_eq!(
        per_field_capped[0].reasoning_content.as_deref(),
        Some(encoded.as_str())
    );

    let mut directly_capped = assistant.clone();
    assert!(!truncate_mutable_field(
        &mut directly_capped,
        MutableMessageField::Reasoning,
        encoded.chars().count(),
        None,
    ));
    assert_eq!(
        directly_capped.reasoning_content.as_deref(),
        Some(encoded.as_str())
    );

    // 若 exact replay 本身已超过总预算，宁可报告未达标，也不能制造不可解码的伪状态。
    let mut aggregate_capped = vec![assistant, tool_result("glm-call", "ok")];
    assert!(!emergency_cap_messages_to_fit(
        &mut aggregate_capped,
        160,
        usize::MAX,
        None,
        &FxHashSet::default(),
    ));
    assert_eq!(
        aggregate_capped[0].reasoning_content.as_deref(),
        Some(encoded.as_str())
    );
    assert_eq!(
        decode_reasoning_replay_for_model(
            model,
            aggregate_capped[0]
                .reasoning_content
                .as_deref()
                .expect("replay marker should remain present"),
        )
        .as_deref(),
        Some(raw_reasoning.as_str())
    );
}

#[test]
fn truncating_mutable_content_archives_original_field() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-truncate-field-{}", uuid::Uuid::new_v4()));
    let original = format!("prefix-{}-suffix", "x".repeat(2_000));
    let mut message = msg("assistant", &original);

    assert!(truncate_mutable_field(
        &mut message,
        MutableMessageField::Content,
        1_600,
        Some(overflow_dir.as_path()),
    ));

    let truncated = value_to_string(&message.content);
    assert!(
        truncated.contains("[context-overflow-truncated] full original archived at:"),
        "{truncated}"
    );
    assert!(truncated.contains("head+tail preview"), "{truncated}");

    let archived = std::fs::read_to_string(overflow_dir.join("overflow-history.md"))
        .expect("truncated original field should be archived");
    assert!(archived.contains("- field: content"), "{archived}");
    assert!(archived.contains(&original), "{archived}");
    assert!(archived.contains("raw_message_json"), "{archived}");

    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn hard_budget_truncation_converges_with_archive_pointer_overhead() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-truncate-converges-{}", uuid::Uuid::new_v4()));
    let mut messages = vec![msg("assistant", &"x".repeat(4_000))];

    assert!(truncate_mutable_messages_to_fit(
        &mut messages,
        700,
        Some(overflow_dir.as_path()),
        &FxHashSet::default(),
    ));
    assert!(messages_total_chars(&messages) <= 700);

    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn hard_budget_truncation_falls_back_when_archive_write_fails() {
    let blocked_dir =
        std::env::temp_dir().join(format!("ai-truncate-blocked-{}", uuid::Uuid::new_v4()));
    std::fs::write(&blocked_dir, "not a directory").expect("create blocking file");
    let overflow_dir = blocked_dir.join("session-assets");
    let mut messages = vec![msg("assistant", &"x".repeat(4_000))];

    assert!(truncate_mutable_messages_to_fit(
        &mut messages,
        600,
        Some(overflow_dir.as_path()),
        &FxHashSet::default(),
    ));
    assert!(messages_total_chars(&messages) <= 600);
    assert!(
        !value_to_string(&messages[0].content).contains("full original archived at:"),
        "failed archive must not leave a dangling pointer"
    );

    let _ = std::fs::remove_file(blocked_dir);
}

#[test]
fn summary_shrink_preserves_system_order_when_archive_write_fails() {
    let blocked_dir =
        std::env::temp_dir().join(format!("ai-summary-blocked-{}", uuid::Uuid::new_v4()));
    std::fs::write(&blocked_dir, "not a directory").expect("create blocking file");
    let overflow_dir = blocked_dir.join("session-assets");
    let messages = vec![
        msg("system", "system prompt"),
        msg("assistant", &"old answer ".repeat(600)),
        msg("user", "recent request"),
    ];

    let shrunk = shrink_messages_to_fit_with_summary(
        messages,
        500,
        200,
        Some(overflow_dir.as_path()),
        &FxHashSet::default(),
    );

    assert_eq!(shrunk.len(), 3);
    assert_eq!(shrunk[0].role, "system");
    assert_eq!(shrunk[1].role, "assistant");
    assert_eq!(shrunk[2].role, "user");
    assert_eq!(value_to_string(&shrunk[0].content), "system prompt");
    assert_eq!(value_to_string(&shrunk[2].content), "recent request");

    let _ = std::fs::remove_file(blocked_dir);
}

#[test]
fn persisted_summary_absorbs_prior_summary_without_nested_prefix() {
    let messages = vec![
        msg(
            ROLE_INTERNAL_NOTE,
            "历史摘要（自动压缩，以下为更早对话的简短语义）：\n- 更早摘要: 初始目标: 修复压缩\n- 已知结论: 保留路径",
        ),
        msg("user", "继续排查 compress.rs"),
        msg("assistant", "发现摘要递归污染"),
    ];

    let summary = build_persisted_summary_text(&messages, 2_000);

    assert!(summary.contains("初始目标: 修复压缩"), "{summary}");
    assert!(
        !summary.contains("更早摘要: - 更早摘要:"),
        "summary should not recursively wrap prior summaries: {summary}"
    );
}

#[test]
fn summary_model_input_drops_ephemeral_internal_notes() {
    let mut messages = vec![
        msg("user", "修复问题"),
        msg(ROLE_INTERNAL_NOTE, "self_note:\n一次性观察"),
        msg(ROLE_INTERNAL_NOTE, "tool_followup:output_truncated"),
        msg(
            ROLE_INTERNAL_NOTE,
            &format!(
                "compressed_tool_round: 1 tool calls (folded for context budget)\n{}\nevidence:\n- read_file [file: src/lib.rs] => 已读过",
                COMPRESSED_TOOL_EVIDENCE_MARKER
            ),
        ),
        msg(
            ROLE_INTERNAL_NOTE,
            "对话摘要（自动压缩，以下为早期对话要点）：\n初始目标: 保留",
        ),
        msg(
            ROLE_INTERNAL_NOTE,
            "历史摘要（自动压缩，以下为更早对话的简短语义）：\n初始目标: 应去重",
        ),
    ];

    normalize_internal_notes_for_summary_model(&mut messages);

    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].role, "user");
    let note = value_to_string(&messages[1].content);
    assert!(
        note.contains(COMPRESSED_TOOL_EVIDENCE_MARKER),
        "compressed tool evidence should survive summary input normalization: {note}"
    );
    assert!(note.contains("src/lib.rs"), "{note}");
    let note = value_to_string(&messages[2].content);
    assert!(note.contains("已有历史摘要"), "{note}");
    assert!(note.contains("初始目标: 保留"), "{note}");
    assert!(!note.contains("self_note"), "{note}");
    assert!(!note.contains("tool_followup"), "{note}");
    assert!(!note.contains("应去重"), "{note}");
}

fn assistant_call_args(id: &str, name: &str, arguments: &str) -> Message {
    let mut m = assistant_call(id, name);
    if let Some(calls) = &mut m.tool_calls {
        calls[0].function.arguments = arguments.to_string();
    }
    m
}

fn assistant_call_args_with_content(
    id: &str,
    name: &str,
    arguments: &str,
    content: &str,
) -> Message {
    let mut m = assistant_call_args(id, name, arguments);
    m.content = Value::String(content.to_string());
    m
}

#[test]
fn folded_tool_group_keeps_assistant_checkpoint_and_evidence_targets() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-evidence-fold-{}", uuid::Uuid::new_v4()));
    let mut messages = vec![msg("system", "s"), msg("user", "分析历史")];
    messages.push(assistant_call_args_with_content(
        "read-history",
        "read_file",
        r#"{"file_path":"0341-history.json","offset":1,"limit":120}"#,
        "已确认文件存在，下一步只统计 role 分布。",
    ));
    messages.push(tool_result(
        "read-history",
        "     1\t[\n     2\t{\"role\":\"user\"}\n... [truncated: showing lines 1-120 of 6393]",
    ));
    for i in 0..4 {
        let id = format!("later-{i}");
        messages.push(assistant_call(&id, "text_grep"));
        messages.push(tool_result(&id, "later"));
    }

    let (folded, folded_groups) = fold_early_tool_groups(
        &messages,
        4,
        Some(overflow_dir.as_path()),
        &FxHashSet::default(),
    );
    assert_eq!(folded_groups, 1);
    let stub = folded
        .iter()
        .find(|message| {
            message.role == ROLE_INTERNAL_NOTE
                && value_to_string(&message.content).contains(COMPRESSED_TOOL_EVIDENCE_MARKER)
        })
        .map(|message| value_to_string(&message.content))
        .expect("folded read group should become evidence note");

    assert!(
        stub.contains("assistant_checkpoint: 已确认文件存在，下一步只统计 role 分布。"),
        "{stub}"
    );
    assert!(stub.contains("evidence:"), "{stub}");
    assert!(
        stub.contains("read_file [file: 0341-history.json; range: lines=1..120]"),
        "{stub}"
    );
    assert!(
        stub.contains("compression_decision: reuse the evidence above"),
        "{stub}"
    );

    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn folded_tool_group_points_to_raw_group_archive() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-raw-group-archive-{}", uuid::Uuid::new_v4()));
    let mut messages = vec![msg("system", "s"), msg("user", "查找 needle")];
    messages.push(assistant_call_args_with_content(
        "grep-raw",
        "text_grep",
        r#"{"pattern":"needle","path":"src/bin/ai"}"#,
        "准备按 needle 搜索目标模块。",
    ));
    messages.push(tool_result(
        "grep-raw",
        "src/bin/ai/example.rs:42: unique raw grep result",
    ));
    for i in 0..4 {
        let id = format!("later-raw-{i}");
        messages.push(assistant_call(&id, "text_grep"));
        messages.push(tool_result(&id, "later"));
    }

    let (folded, folded_groups) = fold_early_tool_groups(
        &messages,
        4,
        Some(overflow_dir.as_path()),
        &FxHashSet::default(),
    );
    assert_eq!(folded_groups, 1);
    let stub = folded
        .iter()
        .find(|message| {
            message.role == ROLE_INTERNAL_NOTE
                && value_to_string(&message.content)
                    .contains("- archive_scope: folded_tool_group_raw_messages")
        })
        .map(|message| value_to_string(&message.content))
        .expect("folded group should expose an archive pointer");

    assert!(
        stub.contains("- archive_scope: folded_tool_group_raw_messages"),
        "{stub}"
    );
    let archive_path = archive_file_path_from_text(&stub);
    let archived = std::fs::read_to_string(&archive_path)
        .expect("folded group raw archive should be readable");
    assert!(archived.contains("grep-raw"), "{archived}");
    assert!(archived.contains("text_grep"), "{archived}");
    assert!(
        archived.contains("准备按 needle 搜索目标模块。"),
        "{archived}"
    );
    assert!(archived.contains("unique raw grep result"), "{archived}");
    assert!(archived.contains("raw_message_json"), "{archived}");

    let _ = std::fs::remove_dir_all(overflow_dir);
}

/// tool-call 轮的 assistant.content 常为空（模型只发 tool_calls、无叙述）。此时只能
/// 从结构化 tool_calls 重建活动摘要，不能把隐藏 reasoning 提升为后续模型可见的事实。
#[test]
fn folded_tool_group_ignores_reasoning_when_content_empty() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-evidence-fold-reason-{}", uuid::Uuid::new_v4()));
    let mut messages = vec![msg("system", "s"), msg("user", "分析历史")];
    let mut call = assistant_call_args(
        "read-history",
        "read_file",
        r#"{"file_path":"0341-history.json","offset":1,"limit":120}"#,
    );
    // content 留空（tool-call 轮的典型形态），仅提供 reasoning。
    call.content = Value::String(String::new());
    call.reasoning_content = Some(
        "已确认该文件是 6393 行的会话历史；下一步只统计 role 分布，不再整文件回读。".to_string(),
    );
    messages.push(call);
    messages.push(tool_result(
        "read-history",
        "     1\t[\n     2\t{\"role\":\"user\"}\n... [truncated: showing lines 1-120 of 6393]",
    ));
    for i in 0..4 {
        let id = format!("later-{i}");
        messages.push(assistant_call(&id, "text_grep"));
        messages.push(tool_result(&id, "later"));
    }

    let (folded, folded_groups) = fold_early_tool_groups(
        &messages,
        4,
        Some(overflow_dir.as_path()),
        &FxHashSet::default(),
    );
    assert_eq!(folded_groups, 1);
    let stub = folded
        .iter()
        .find(|message| {
            message.role == ROLE_INTERNAL_NOTE
                && value_to_string(&message.content).contains(COMPRESSED_TOOL_EVIDENCE_MARKER)
        })
        .map(|message| value_to_string(&message.content))
        .expect("folded read group should become evidence note");

    assert!(stub.contains("assistant_checkpoint: no assistant narration was persisted; reconstructed completed tool activity: read_file [file: 0341-history.json; range: lines=1..120]"), "{stub}");
    assert!(
        !stub.contains("已确认该文件是 6393 行的会话历史"),
        "hidden reasoning must not be promoted into compressed context: {stub}"
    );

    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn folded_tool_group_reconstructs_checkpoint_when_tool_call_has_no_text() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-evidence-fold-reconstructed-{}",
        uuid::Uuid::new_v4()
    ));
    let mut messages = vec![msg("system", "s"), msg("user", "分析历史")];
    let mut call = assistant_call_args(
        "read-history",
        "read_file",
        r#"{"file_path":"0341-history.json","offset":1,"limit":120}"#,
    );
    call.content = Value::Null;
    call.reasoning_content = None;
    call.tool_calls
        .as_mut()
        .expect("assistant call should contain tool calls")
        .push(ToolCall {
            id: "opaque-call".to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: "opaque_tool".to_string(),
                arguments: r#"{"payload":"DO_NOT_SURFACE_RAW_ARGUMENTS"}"#.to_string(),
            },
        });
    messages.push(call);
    messages.push(tool_result(
        "read-history",
        "     1\t[\n     2\t{\"role\":\"user\"}\n... [truncated: showing lines 1-120 of 6393]",
    ));
    messages.push(tool_result("opaque-call", "opaque result"));
    for i in 0..4 {
        let id = format!("later-{i}");
        messages.push(assistant_call(&id, "text_grep"));
        messages.push(tool_result(&id, "later"));
    }

    let (folded, folded_groups) = fold_early_tool_groups(
        &messages,
        4,
        Some(overflow_dir.as_path()),
        &FxHashSet::default(),
    );
    assert_eq!(folded_groups, 1);
    let stub = folded
        .iter()
        .find(|message| {
            message.role == ROLE_INTERNAL_NOTE
                && value_to_string(&message.content).contains(COMPRESSED_TOOL_EVIDENCE_MARKER)
        })
        .map(|message| value_to_string(&message.content))
        .expect("folded read group should become evidence note");

    assert!(
        stub.contains("assistant_checkpoint: no assistant narration was persisted; reconstructed completed tool activity: read_file [file: 0341-history.json; range: lines=1..120]; opaque_tool"),
        "{stub}"
    );
    assert!(
        !stub.contains("DO_NOT_SURFACE_RAW_ARGUMENTS"),
        "generic tool arguments must not be copied into the checkpoint: {stub}"
    );

    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn current_turn_precision_results_stay_raw_during_mid_turn_compress() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-current-turn-precision-{}",
        uuid::Uuid::new_v4()
    ));
    let content = (1..=120)
        .map(|line| format!("{line:>6}\tlet value_{line} = {line};\n"))
        .collect::<String>();
    let mut messages = vec![
        msg("system", "system prompt"),
        msg("user", "当前轮读取后应直接修复"),
    ];
    messages.push(assistant_call_args(
        "read-1",
        "read_file",
        r#"{"file_path":"src/lib.rs","offset":1,"limit":120}"#,
    ));
    messages.push(tool_result("read-1", &content));
    messages.push(assistant_call_args(
        "read-2",
        "read_file",
        r#"{"file_path":"src/lib.rs","offset":1,"limit":120}"#,
    ));
    messages.push(tool_result("read-2", &content));

    let (compressed, _, _) = mid_turn_compress(messages, 2_000, Some(&overflow_dir));
    let results = compressed
        .iter()
        .filter(|message| message.role == "tool")
        .map(|message| value_to_string(&message.content))
        .collect::<Vec<_>>();

    assert_eq!(results, vec![content.clone(), content]);
    assert!(
        results.iter().all(|text| {
            !text.contains("[deduped:") && !text.contains("Output preserved for tool")
        }),
        "current-turn precision outputs must stay raw: {results:?}"
    );

    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn current_turn_task_wait_is_protected_from_path_c_lossy_truncation() {
    let messages = vec![
        msg("system", "system prompt"),
        msg("user", "等待并采用子任务结论"),
        assistant_call("wait-1", "task_wait"),
        tool_result("wait-1", &"aggregated conclusion\n".repeat(600)),
    ];

    let precision_ids = current_turn_precision_tool_call_ids(&messages);
    let lossless_ids = current_turn_lossless_tool_call_ids(&messages);

    assert!(
        !precision_ids.contains("wait-1"),
        "task_wait must not consume the precision inline budget"
    );
    assert!(
        lossless_ids.contains("wait-1"),
        "Path C must preserve task_wait instead of truncating its conclusion"
    );
}

#[test]
fn leading_compressed_tool_evidence_notes_are_not_immortal_prefix_context() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-leading-tool-evidence-{}", uuid::Uuid::new_v4()));
    let compressed_note = format!(
        "compressed_tool_round: 1 tool calls (folded for context budget)\n{}\nassistant_checkpoint: <empty; no persisted decision before these tool calls>\nevidence:\n- read_file [file: src/lib.rs] => {}",
        COMPRESSED_TOOL_EVIDENCE_MARKER,
        "x".repeat(1_200)
    );
    let mut messages = vec![
        msg("system", "system prompt"),
        msg(
            ROLE_INTERNAL_NOTE,
            "历史摘要（自动压缩，以下为更早对话的简短语义）：\n初始目标: 修复上下文压缩",
        ),
    ];
    for _ in 0..8 {
        messages.push(msg(ROLE_INTERNAL_NOTE, &compressed_note));
    }
    for i in 0..4 {
        messages.push(msg("user", &format!("旧问题 {i}")));
        messages.push(msg("assistant", "旧回答"));
    }
    messages.push(msg("user", "当前问题必须保留"));

    let trim_idx = first_trim_candidate(&messages, 2_000)
        .expect("leading compressed evidence should be eligible for trimming");
    assert_eq!(
        value_to_string(&messages[trim_idx].content),
        compressed_note,
        "summary/system prefix should stay protected, but compressed tool evidence should not"
    );

    let (compressed, before, after) = mid_turn_compress(messages, 2_000, Some(&overflow_dir));
    assert!(after < before, "compression should make progress");
    assert!(
        compressed.iter().any(|message| message.role == "user"
            && message.content.as_str() == Some("当前问题必须保留")),
        "latest user message must remain protected"
    );
    let remaining_evidence = compressed
        .iter()
        .filter(|message| is_compressed_tool_evidence_note(message))
        .count();
    assert!(
        remaining_evidence < 8,
        "stale compressed evidence notes should not remain an immortal prefix"
    );
    let archived = std::fs::read_to_string(overflow_dir.join("overflow-history.md"))
        .expect("trimmed compressed evidence should be archived before removal");
    assert!(archived.contains("compressed_tool_round"), "{archived}");
    assert!(
        archived.contains(COMPRESSED_TOOL_EVIDENCE_MARKER),
        "{archived}"
    );

    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn compressed_tool_evidence_has_independent_inline_budget() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-bounded-tool-evidence-{}", uuid::Uuid::new_v4()));
    let mut messages = vec![msg("system", "system prompt"), msg("user", "分析长工具链")];
    for index in 0..24 {
        messages.push(msg(
            ROLE_INTERNAL_NOTE,
            &format!(
                "compressed_tool_round: 1 tool calls (folded for context budget)\n{}\nassistant_checkpoint: checkpoint-{index:02}\nevidence:\n- read_file => {}",
                COMPRESSED_TOOL_EVIDENCE_MARKER,
                "x".repeat(900)
            ),
        ));
    }
    assert!(compressed_tool_evidence_exceeds_inline_budget(&messages));

    // 整体远低于全局 100K 预算，仍应由工具证据自己的 12K 上限主动收敛。
    let compressed =
        compress_messages_for_context(messages, 100_000, 256, 8_000, Some(overflow_dir.clone()));
    let evidence = compressed
        .iter()
        .filter(|message| is_compressed_tool_evidence_note(message))
        .map(|message| value_to_string(&message.content))
        .collect::<Vec<_>>();
    let inline_chars = compressed
        .iter()
        .filter(|message| is_compressed_tool_evidence_note(message))
        .map(message_billable_chars)
        .sum::<usize>();

    assert!(inline_chars <= MAX_COMPRESSED_TOOL_EVIDENCE_INLINE_CHARS);
    assert!(evidence.iter().any(|text| text.contains("checkpoint-23")));
    assert!(!evidence.iter().any(|text| text.contains("checkpoint-00")));
    assert_eq!(
        compressed
            .iter()
            .filter(|message| is_archive_note_message(message))
            .count(),
        1
    );
    let archived = std::fs::read_to_string(overflow_dir.join("overflow-history.md"))
        .expect("older evidence should be archived before removal");
    assert!(archived.contains("checkpoint-00"), "{archived}");
    assert!(!archived.contains("checkpoint-23"), "{archived}");

    let _ = std::fs::remove_dir_all(overflow_dir);
}

/// 压缩后的命令组必须保留调用参数。仅保留「成功但无输出」不足以说明已经查过
/// 哪个 author/date/cwd 组合，模型会把它当成未执行过的调查而从同一条 git log 重启。
#[test]
fn folded_command_keeps_invocation_for_empty_success() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-command-invocation-{}", uuid::Uuid::new_v4()));
    let command = r#"git log --all --author="wangwenchao.129" --since="2026-07-22 00:00" --until="2026-07-23 00:00""#;
    let mut messages = vec![msg("system", "s"), msg("user", "审查今天的改动")];
    messages.push(assistant_call_args(
        "git-log",
        "execute_command",
        r#"{"command":"git log --all --author=\"wangwenchao.129\" --since=\"2026-07-22 00:00\" --until=\"2026-07-23 00:00\"","cwd":"/data01/AeolusLLM"}"#,
    ));
    messages.push(tool_result(
        "git-log",
        "(command succeeded with exit code 0 and produced no output)",
    ));
    for i in 0..4 {
        let id = format!("later-{i}");
        messages.push(assistant_call(&id, "text_grep"));
        messages.push(tool_result(&id, "later"));
    }

    let (folded, folded_groups) = fold_early_tool_groups(
        &messages,
        4,
        Some(overflow_dir.as_path()),
        &FxHashSet::default(),
    );
    assert_eq!(folded_groups, 1);
    let stub = folded
        .iter()
        .find(|message| {
            message.role == ROLE_INTERNAL_NOTE
                && value_to_string(&message.content).contains("execute_command")
        })
        .map(|message| value_to_string(&message.content))
        .expect("command group should be folded into a recall stub");
    assert!(stub.contains(&format!("command: {command}")), "{stub}");
    assert!(stub.contains("cwd: /data01/AeolusLLM"), "{stub}");
    assert!(stub.contains(COMPRESSED_TOOL_EVIDENCE_MARKER), "{stub}");
    assert!(
        stub.contains("command succeeded with exit code 0 and produced no output"),
        "{stub}"
    );
    assert!(
        stub.contains("compression_decision: reuse the evidence above before repeating the same read/search/list/command action"),
        "{stub}"
    );

    let _ = std::fs::remove_dir_all(overflow_dir);
}

fn assistant_call_args_multi(id: &str, calls: &[(&str, &str)]) -> Message {
    Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(
            calls
                .iter()
                .map(|(name, args)| ToolCall {
                    id: id.to_string(),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: name.to_string(),
                        arguments: args.to_string(),
                    },
                })
                .collect(),
        ),
        tool_call_id: None,
        reasoning_content: None,
    }
}

/// apply_patch 失败后，该路径最近的 read_file 结果不得被折叠——否则模型
/// 会因拿不到精确 context 再次 patch 失败、陷入"重读→再失败"循环。
#[test]
fn preserves_read_file_for_pending_patch_path() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-pending-patch-fold-{}",
        uuid::Uuid::new_v4()
    ));
    let mut messages = vec![msg("system", "s"), msg("user", "改代码")];
    // 早期：read_file 读 /a.rs（将被 apply_patch 引用）。
    messages.push(assistant_call_args(
        "call-rf",
        "read_file",
        r#"{"file_path":"/a.rs"}"#,
    ));
    messages.push(tool_result(
        "call-rf",
        "PENDING_READ_SENTINEL\nOutput preserved for non-compressible tool `read_file`.\n- file_path: /a.rs\n- use read_file to inspect exact content.",
    ));
    // apply_patch 针对 /a.rs 失败（pending）。
    messages.push(assistant_call_args(
        "call-ap",
        "apply_patch",
        r#"{"file_path":"/a.rs","patch":"@@ @@\n-x\n+y\n"}"#,
    ));
    messages.push(tool_result(
        "call-ap",
        "Error: apply_patch failed: context mismatch: patch hunk could not be located.",
    ));
    // 追加足够多近端组把上面挤进折叠区。
    for i in 0..6 {
        let id = format!("call-{i}");
        messages.push(assistant_call(&id, "text_grep"));
        messages.push(tool_result(&id, "recent"));
    }

    let (folded, folded_groups) = fold_early_tool_groups(
        &messages,
        4,
        Some(overflow_dir.as_path()),
        &FxHashSet::default(),
    );
    assert!(folded_groups >= 1, "应至少折叠 apply_patch/grep 组");
    // read_file 组必须逐字保留（不是 ROLE_INTERNAL_NOTE stub）。
    let rf = folded
        .iter()
        .find(|m| {
            m.role == "assistant"
                && m.tool_calls
                    .as_ref()
                    .and_then(|cs| cs.first())
                    .map(|c| c.function.name == "read_file")
                    .unwrap_or(false)
        })
        .expect("pending-patch 路径的 read_file 组不应被折叠");
    let _ = rf; // 仅断言其存在且 role 仍为 assistant
    assert_tool_pairs_consistent(&folded);
    let archive = std::fs::read_to_string(overflow_dir.join(OVERFLOW_HISTORY_FILENAME))
        .expect("other folded groups should create overflow archive");
    assert!(
        !archive.contains("PENDING_READ_SENTINEL"),
        "pending-patch read group must not be archived while it remains inline"
    );
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn preserves_read_file_for_each_pending_path_from_multi_file_patch() {
    let mut messages = vec![msg("system", "s"), msg("user", "批量改代码")];
    messages.push(assistant_call_args(
        "call-rf-a",
        "read_file",
        r#"{"file_path":"/a.rs"}"#,
    ));
    messages.push(tool_result(
        "call-rf-a",
        "Output preserved for non-compressible tool `read_file`.\n- file_path: /a.rs\n- use read_file to inspect exact content.",
    ));
    messages.push(assistant_call_args(
        "call-rf-b",
        "read_file",
        r#"{"file_path":"/b.rs"}"#,
    ));
    messages.push(tool_result(
        "call-rf-b",
        "Output preserved for non-compressible tool `read_file`.\n- file_path: /b.rs\n- use read_file to inspect exact content.",
    ));
    messages.push(assistant_call_args(
        "call-ap",
        "apply_patch",
        r#"{"patch":"*** Begin Patch\n*** Update File: /a.rs\n@@\n-old_a\n+new_a\n*** Update File: /b.rs\n@@\n-old_b\n+new_b\n*** End Patch"}"#,
    ));
    messages.push(tool_result(
        "call-ap",
        "Error: apply_patch failed: failed while preparing patch for /b.rs: context mismatch: patch hunk could not be located.",
    ));
    for i in 0..6 {
        let id = format!("call-multi-{i}");
        messages.push(assistant_call(&id, "text_grep"));
        messages.push(tool_result(&id, "recent"));
    }

    let (folded, _) = fold_early_tool_groups(&messages, 4, None, &FxHashSet::default());
    for target in ["/a.rs", "/b.rs"] {
        let preserved = folded.iter().any(|m| {
            m.role == "assistant"
                && m.tool_calls
                    .as_ref()
                    .and_then(|cs| cs.first())
                    .map(|c| {
                        c.function.name == "read_file"
                            && c.function.arguments.contains(&format!("\"{target}\""))
                    })
                    .unwrap_or(false)
        });
        assert!(
            preserved,
            "pending multi-file patch path {target} should be preserved"
        );
    }
    assert_tool_pairs_consistent(&folded);
}

#[test]
fn removes_only_byte_identical_overlap_from_an_aged_read_file_result() {
    let mut messages = vec![
        assistant_call_args(
            "older",
            "read_file",
            r#"{"file_path":"/a.rs","offset":1,"limit":3}"#,
        ),
        tool_result("older", "     1\tone\n     2\ttwo\n     3\tthree"),
        assistant_call_args(
            "later",
            "read_file",
            r#"{"file_path":"/a.rs","offset":2,"limit":3}"#,
        ),
        tool_result("later", "     2\ttwo\n     3\tthree\n     4\tfour"),
    ];
    let signatures = rustc_hash::FxHashMap::from_iter([
        (
            "older".to_string(),
            (
                "read_file".to_string(),
                r#"{"file_path":"/a.rs","offset":1,"limit":3}"#.to_string(),
            ),
        ),
        (
            "later".to_string(),
            (
                "read_file".to_string(),
                r#"{"file_path":"/a.rs","offset":2,"limit":3}"#.to_string(),
            ),
        ),
    ]);

    dedup_overlapping_read_file_results(
        &mut messages,
        &signatures,
        &FxHashSet::default(),
        &FxHashSet::default(),
    );

    let earlier = value_to_string(&messages[1].content);
    assert!(earlier.contains("overlap dedup: 2"), "{earlier}");
    assert!(earlier.contains("1\tone"), "{earlier}");
    assert!(!earlier.contains("2\ttwo"), "{earlier}");
    assert_eq!(
        value_to_string(&messages[3].content),
        "     2\ttwo\n     3\tthree\n     4\tfour"
    );
}

#[test]
fn retains_overlap_when_the_file_changed_between_reads() {
    let mut messages = vec![
        assistant_call_args("older", "read_file", r#"{"file_path":"/a.rs"}"#),
        tool_result("older", "     1\tone\n     2\tbefore"),
        assistant_call_args("later", "read_file", r#"{"file_path":"/a.rs"}"#),
        tool_result("later", "     2\tafter\n     3\tthree"),
    ];
    let signatures = rustc_hash::FxHashMap::from_iter([
        (
            "older".to_string(),
            (
                "read_file".to_string(),
                r#"{"file_path":"/a.rs"}"#.to_string(),
            ),
        ),
        (
            "later".to_string(),
            (
                "read_file".to_string(),
                r#"{"file_path":"/a.rs"}"#.to_string(),
            ),
        ),
    ]);

    dedup_overlapping_read_file_results(
        &mut messages,
        &signatures,
        &FxHashSet::default(),
        &FxHashSet::default(),
    );

    assert_eq!(
        value_to_string(&messages[1].content),
        "     1\tone\n     2\tbefore"
    );
}

/// 渐进折叠窗口序列从最大保护窗口起、递进收紧但**绝不到 0**：最近一次工具交互
/// 始终至少保留 1 组逐字，避免最近的结构化工具上下文也被折叠成 stub。
#[test]
fn progressive_fold_windows_never_reach_zero() {
    let windows = progressive_fold_windows();
    assert_eq!(
        *windows.first().unwrap(),
        KEEP_RECENT_TOOL_GROUPS,
        "sequence must start at the max protection window"
    );
    assert_eq!(
        *windows.last().unwrap(),
        MIN_KEEP_RECENT_TOOL_GROUPS,
        "sequence must end at the minimum protection window, not 0"
    );
    assert!(
        windows.iter().all(|&w| w >= MIN_KEEP_RECENT_TOOL_GROUPS),
        "no window may drop below the minimum (was {windows:?})"
    );
    assert!(
        !windows.contains(&0),
        "window 0 folds the most recent tool interaction into a stub (was {windows:?})"
    );
    // 严格递减，保证每一步真正放宽折叠范围、不空转。
    assert!(
        windows.windows(2).all(|pair| pair[0] > pair[1]),
        "windows must be strictly decreasing (was {windows:?})"
    );
}

#[test]
fn context_compaction_state_is_model_visible_and_idempotent() {
    let mut messages = vec![
        msg("system", "system"),
        msg("user", "finish the investigation"),
    ];

    upsert_context_compaction_state(&mut messages);
    upsert_context_compaction_state(&mut messages);

    let notes = messages
        .iter()
        .filter(|message| is_context_compaction_state(message))
        .collect::<Vec<_>>();
    assert_eq!(
        notes.len(),
        1,
        "repeated compaction must update one state note"
    );
    let user_index = messages
        .iter()
        .position(|message| message.role == "user")
        .expect("test setup has user message");
    let note_index = messages
        .iter()
        .position(is_context_compaction_state)
        .expect("compaction state note should be inserted");
    assert!(
        note_index > user_index,
        "compaction state note must stay after the latest user message"
    );

    let content = value_to_string(&notes[0].content);
    assert!(content.contains("passed the runtime budget guard"));
    assert!(content.contains("does not mean the model context is full"));
    assert!(content.contains("original_file_path"));
}
