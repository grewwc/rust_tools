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

#[test]
fn folds_early_groups_in_a_single_bloated_turn() {
    let messages = single_turn_with_groups(10, 2_000);
    let before = messages_total_chars(&messages);

    let (folded, folded_groups) = fold_early_tool_groups(&messages, 4, None);

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
    let (folded, _) = fold_early_tool_groups(&messages, 4, None);

    let user = folded
        .iter()
        .find(|m| m.role == "user")
        .expect("user message must survive");
    assert_eq!(value_to_string(&user.content), "干活");
}

#[test]
fn keeps_recent_groups_verbatim() {
    let messages = single_turn_with_groups(8, 1_500);
    let (folded, _) = fold_early_tool_groups(&messages, 4, None);

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
    let (folded, folded_groups) = fold_early_tool_groups(&messages, 4, None);

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
    let (folded, folded_groups) = fold_early_tool_groups(&messages, 0, None);

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

    let (folded, folded_groups) = fold_early_tool_groups(&messages, 4, None);
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
        (
            "search",
            "code_search",
            r#"{"operation":"text_search","query":"task_wait","path":"src/bin/ai"}"#,
        ),
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

    let (folded, folded_groups) = fold_early_tool_groups(&messages, 4, None);
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
        folded_text.contains("- original_operation: text_search"),
        "{folded_text}"
    );
    assert!(
        folded_text.contains("- original_query: task_wait"),
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

    let (folded, folded_groups) =
        fold_early_tool_groups(&messages, 4, Some(overflow_dir.as_path()));
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
            "对话摘要（自动压缩，以下为早期对话要点）：\n初始目标: 保留",
        ),
        msg(
            ROLE_INTERNAL_NOTE,
            "历史摘要（自动压缩，以下为更早对话的简短语义）：\n初始目标: 应去重",
        ),
    ];

    normalize_internal_notes_for_summary_model(&mut messages);

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "user");
    let note = value_to_string(&messages[1].content);
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

    let (folded, folded_groups) =
        fold_early_tool_groups(&messages, 4, Some(overflow_dir.as_path()));
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
    assert!(
        stub.contains("command succeeded with exit code 0 and produced no output"),
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
    let mut messages = vec![msg("system", "s"), msg("user", "改代码")];
    // 早期：read_file 读 /a.rs（将被 apply_patch 引用）。
    messages.push(assistant_call_args(
        "call-rf",
        "read_file",
        r#"{"file_path":"/a.rs"}"#,
    ));
    messages.push(tool_result(
        "call-rf",
        "Output preserved for non-compressible tool `read_file`.\n- file_path: /a.rs\n- use read_file to inspect exact content.",
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

    let (folded, folded_groups) = fold_early_tool_groups(&messages, 4, None);
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

    let (folded, _) = fold_early_tool_groups(&messages, 4, None);
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

    dedup_overlapping_read_file_results(&mut messages, &signatures, &FxHashSet::default());

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

    dedup_overlapping_read_file_results(&mut messages, &signatures, &FxHashSet::default());

    assert_eq!(
        value_to_string(&messages[1].content),
        "     1\tone\n     2\tbefore"
    );
}
