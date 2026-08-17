//! user/image 外溢 stub 合并与批量裁剪回归测试（问题 1-4 修复验证）。
//!
//! 覆盖：
//! - `merge_old_user_overflow_stubs`：保护尾窗之外的旧 stub 折叠为单条目录指针，
//!   占位开销从 O(N) 收敛到 O(1)；最近轮次的 stub 保持逐条指针（问题 1）。
//! - `trim_removable_messages_batch`：user 候选跳过而非 break，其后的可裁候选仍
//!   被批量移除，user 原文永不删除（问题 2 + 3）。
//! - 批量裁剪不再把 internal note 追加进 overflow-history.md（问题 4）。

use super::*;

fn msg(role: &str, content: &str) -> Message {
    Message {
        role: role.to_string(),
        content: Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn user_stub(dir: &str, name: &str) -> Message {
    let path = format!("{dir}/preserved_user_overflow/{name}.json");
    msg(
        "user",
        &format!(
            "较早的用户文本内容已归档，原文未丢失。\n归档文件: {path}\n\
             这是一条上下文归档提示，不是用户的新请求。仅当当前任务确实依赖原文时，\
             才使用 read_file 读取该文件。"
        ),
    )
}

fn note(text: &str) -> Message {
    msg(ROLE_INTERNAL_NOTE, text)
}

/// 构造一条旧版 role=user 的合并指针（模拟修复前落盘的快照）。
fn old_merged_stub(count: usize, dir: &str) -> Message {
    msg(
        "user",
        &format!(
            "较早的用户内容已归档（共 {count} 条，原文零压缩保存在会话归档目录）。\n\
             归档目录: {dir}\n\
             这是一条上下文归档提示，不是用户的新请求。"
        ),
    )
}

/// 问题 1：合并目录指针不能冒充 user，也不能吞掉后续摘要仍需依赖的
/// 最近 2/3 个真实 user 边界。
#[test]
fn merge_old_user_overflow_stubs_collapses_old_stubs_keeps_recent() {
    let dir = "/sessions/test-session/assets";
    let mut messages = vec![
        msg("system", "sys"),
        user_stub(dir, "20260101T000000Z-user-1"),
        msg("assistant", "a1"),
        user_stub(dir, "20260102T000000Z-user-2"),
        msg("assistant", "a2"),
        user_stub(dir, "20260103T000000Z-user-3"),
        msg("assistant", "a3"),
        user_stub(dir, "20260104T000000Z-user-4"),
        msg("assistant", "a4"),
        user_stub(dir, "20260105T000000Z-user-5"),
        msg("assistant", "a5"),
        user_stub(dir, "20260106T000000Z-user-6"),
        msg("assistant", "a6"),
        msg("user", "最近的用户轮次"),
    ];
    merge_old_user_overflow_stubs(&mut messages, 1);

    // 前 4 条 stub → 1 条 internal_note；最近 3 个真实 user 边界仍在。
    assert_eq!(messages.len(), 11, "{messages:?}");
    let merged = &messages[1];
    assert_eq!(merged.role, ROLE_INTERNAL_NOTE);
    let text = value_to_string(&merged.content);
    assert!(
        text.starts_with("较早的用户内容已归档（共 4 条"),
        "合并指针应携带条数: {text}"
    );
    assert!(
        text.contains("归档目录: /sessions/test-session/assets/preserved_user_overflow"),
        "{text}"
    );
    // 合并指针仍是受保护的 stub：后续 first_trim_candidate / truncate 不会误删。
    assert!(is_preserved_user_or_image_stub(&text));
    let user_messages: Vec<_> = messages
        .iter()
        .filter(|message| message.role == "user")
        .collect();
    assert_eq!(user_messages.len(), 3, "{messages:?}");
    assert!(value_to_string(&user_messages[0].content).contains("user-5"));
    assert!(value_to_string(&user_messages[1].content).contains("user-6"));
    assert_eq!(value_to_string(&user_messages[2].content), "最近的用户轮次");

    // mid-turn 摘要保护最近 2 轮时仍能得到非零 split，旧前缀可被真正摘要。
    let summary_split = retained_turn_start(&messages, 2);
    assert!(
        summary_split > 0,
        "合并后不应退化为只有一个 user 边界: {messages:?}"
    );
    assert!(value_to_string(&messages[summary_split].content).contains("user-6"));
}

/// 审计回归：合并指针进入下一次请求时会经过模型规范化；所有目录和
/// tree -> read_file 的可执行召回协议必须完整保留，不能退化成对目录 read_file。
#[test]
fn merged_stub_survives_request_normalization_with_all_directories() {
    let mut messages = vec![
        msg("system", "sys"),
        user_stub("/sessions/session-a/assets", "20260101T000000Z-user-1"),
        user_stub("/sessions/session-a/assets", "20260102T000000Z-user-2"),
        user_stub("/sessions/session-b/assets", "20260103T000000Z-user-3"),
        user_stub("/sessions/session-b/assets", "20260104T000000Z-user-4"),
        user_stub("/sessions/session-b/assets", "20260105T000000Z-user-5"),
        user_stub("/sessions/session-b/assets", "20260106T000000Z-user-6"),
        msg("user", "最近轮次"),
    ];
    merge_old_user_overflow_stubs(&mut messages, 1);

    // 模拟修复前已经落入上下文快照的首版格式：它把目录误标成「归档文件」。
    // 新版本必须兼容恢复，不能只修复新生成的指针。
    let merged_idx = messages
        .iter()
        .position(|message| {
            value_to_string(&message.content).starts_with("较早的用户内容已归档（共 4 条")
        })
        .unwrap();
    let legacy = value_to_string(&messages[merged_idx].content).replace("归档目录: ", "归档文件: ");
    messages[merged_idx].content = Value::String(legacy);

    // 模拟合并后的上下文进入下一次真实请求入口。
    let normalized = compress_messages_for_context(messages, 100_000, 0, 0, None, None);
    let merged = normalized
        .iter()
        .map(|message| value_to_string(&message.content))
        .find(|text| text.starts_with("较早的用户内容已归档（共 4 条"))
        .expect("合并指针应在请求规范化后继续存在");

    for dir in [
        "/sessions/session-a/assets/preserved_user_overflow",
        "/sessions/session-b/assets/preserved_user_overflow",
    ] {
        assert!(merged.contains(&format!("归档目录: {dir}")), "{merged}");
    }
    assert!(merged.contains("使用 tree 列出上述归档目录"), "{merged}");
    assert!(merged.contains("使用 read_file 读取具体文件"), "{merged}");
    assert!(merged.contains("不要对目录直接调用 read_file"), "{merged}");
    assert!(is_preserved_user_or_image_stub(&merged));
}

/// 已有合并指针必须吸收后续老化的单文件 stub，始终收敛为一个指针，
/// 否则长会话会变成「每 4 条新增一个合并指针」的线性膨胀。
#[test]
fn repeated_merge_absorbs_new_stubs_into_one_pointer() {
    let dir = "/sessions/test-session/assets";
    let mut messages = vec![
        user_stub(dir, "20260101T000000Z-user-1"),
        user_stub(dir, "20260102T000000Z-user-2"),
        user_stub(dir, "20260103T000000Z-user-3"),
        user_stub(dir, "20260104T000000Z-user-4"),
        user_stub(dir, "20260105T000000Z-user-5"),
        user_stub(dir, "20260106T000000Z-user-6"),
        msg("user", "最近轮次"),
    ];
    merge_old_user_overflow_stubs(&mut messages, 1);
    for ordinal in 7..=10 {
        messages.insert(
            messages.len() - 1,
            user_stub(dir, &format!("202601{ordinal:02}T000000Z-user-{ordinal}")),
        );
    }
    merge_old_user_overflow_stubs(&mut messages, 1);

    // 一个聚合指针 + 两个结构边界 stub + 最新 user，消息数仍是 O(1)。
    assert_eq!(messages.len(), 4, "{messages:?}");
    let merged = value_to_string(&messages[0].content);
    assert!(
        merged.starts_with("较早的用户内容已归档（共 8 条"),
        "{merged}"
    );
    assert_eq!(merged.matches("归档目录: ").count(), 1, "{merged}");
    assert_eq!(
        messages
            .iter()
            .filter(|message| message.role == "user")
            .count(),
        3
    );
}

/// 问题 1：stub 数量不足阈值时不合并（保留逐条指针以便精确召回）。
#[test]
fn merge_old_user_overflow_stubs_keeps_below_threshold() {
    let dir = "/sessions/test-session/assets";
    let mut messages = vec![
        user_stub(dir, "20260101T000000Z-user-1"),
        user_stub(dir, "20260102T000000Z-user-2"),
        user_stub(dir, "20260103T000000Z-user-3"),
        msg("user", "最近"),
    ];
    merge_old_user_overflow_stubs(&mut messages, 1);
    assert_eq!(messages.len(), 4);
    // 3 条 stub 原样保留（不足阈值不合并），最近轮次不受影响。
    assert!(
        messages[..3]
            .iter()
            .all(|m| is_preserved_user_or_image_stub(&value_to_string(&m.content))),
        "{messages:?}"
    );
    assert_eq!(messages[3].role, "user");
    assert_eq!(value_to_string(&messages[3].content), "最近");
}

/// 问题 2 + 3：batch 遇到 user 候选时跳过继续而不是 break——
/// 其后的可裁候选仍被批量移除；user 原文永不删除。
#[test]
fn batch_skips_user_candidates_instead_of_breaking() {
    let mut messages = vec![
        msg("user", "q1"),
        msg("assistant", "a1"),
        msg("user", "q2（旧轮次，此路径不可删）"),
        msg("assistant", &"x".repeat(1000)),
        msg("assistant", &"y".repeat(1000)),
        msg("user", "最近轮次 q3"),
        msg("assistant", "a3"),
    ];
    let removed = trim_removable_messages_batch(&mut messages, 40, None);
    assert!(removed);
    // 两个可裁的 assistant 长文本被移除；user 全部保留（含旧轮次 q2）。
    assert_eq!(messages.len(), 4, "{messages:?}");
    assert!(
        messages.iter().all(|m| {
            m.role == "user"
                || value_to_string(&m.content).contains("a1")
                || value_to_string(&m.content).contains("a3")
        }),
        "{messages:?}"
    );
    assert!(
        messages
            .iter()
            .any(|m| m.role == "user" && value_to_string(&m.content).contains("q2")),
        "user 候选绝不能被移除: {messages:?}"
    );
}

/// internal_note 被裁剪时必须有可搜索的逐字归档，同时不能重复 append 到
/// overflow-history.md 造成长期膨胀。
#[test]
fn batch_archives_internal_notes_deduplicated() {
    let dir = std::env::temp_dir().join(format!(
        "a_compress_note_archive_test_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // 足够多的 user 轮次让 keep_recent 降到 1（否则尾部窗口保护全部消息，
    // batch 不会裁剪——与修复无关的既有语义）。
    let mut messages = vec![
        note("compressed_tool_round: 1 tool calls (folded for context budget)"),
        msg("assistant", &"可裁的正文消息".repeat(50)),
        msg("user", &"旧轮次 1".repeat(20)),
        msg("user", &"旧轮次 2".repeat(20)),
        msg("user", &"旧轮次 3".repeat(20)),
        msg("user", "最近"),
    ];
    let removed = trim_removable_messages_batch(&mut messages, 40, Some(&dir));
    assert!(removed, "note 与 assistant 都应被批量移除");
    assert_eq!(messages.len(), 5, "{messages:?}");
    assert!(
        messages.iter().any(|m| {
            m.role == ROLE_INTERNAL_NOTE
                && value_to_string(&m.content).contains("internal-note-overflow")
        }),
        "{messages:?}"
    );

    let archive_file = dir.join("overflow-history.md");
    let archived = std::fs::read_to_string(&archive_file).unwrap_or_default();
    assert!(
        archived.contains("可裁的正文消息"),
        "正文消息必须归档: {archived}"
    );
    assert!(
        !archived.contains("compressed_tool_round"),
        "internal note 不应重复归档: {archived}"
    );
    let note_dir = dir.join("internal-note-overflow");
    let files: Vec<_> = std::fs::read_dir(&note_dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(files.len(), 1, "{files:?}");
    let note_archive = std::fs::read_to_string(&files[0]).unwrap();
    assert!(
        note_archive.contains("compressed_tool_round"),
        "{note_archive}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn batch_uses_low_budget_three_turn_tail_before_crossing_48k() {
    // 构造总字符 > 48K 但最近三轮尾窗 ≤ 40K 的场景：
    // 预算 40K ≤ 48K 时应保护 3 轮而非 2 轮。
    // 每条 assistant 用唯一前缀标记，避免 any(len==N) 被其他消息满足。
    let mut messages = vec![
        msg("user", &"x".repeat(20_000)),
        msg("assistant", &"o".repeat(20_000)),
        msg("user", "third-recent-user"),
        msg("assistant", &format!("third-asst-{}", "t".repeat(9_990))),
        msg("user", "second-recent-user"),
        msg("assistant", &format!("second-asst-{}", "s".repeat(9_989))),
        msg("user", "recent-user"),
        msg("assistant", &format!("recent-asst-{}", "r".repeat(9_991))),
    ];

    assert!(messages_total_chars(&messages) > KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS);
    assert!(trim_removable_messages_batch(&mut messages, 40_000, None));
    // 第三近用户轮次不得在跨过 48K 前被裁掉
    assert!(
        messages
            .iter()
            .any(|m| value_to_string(&m.content).contains("third-recent-user")),
        "第三近用户轮次不得在跨过 48K 前被裁掉: {messages:?}"
    );
    // 第三近 assistant 正文必须完整保留（旧逻辑只保 2 轮时会删掉它）
    assert!(
        messages
            .iter()
            .any(|m| value_to_string(&m.content).contains("third-asst-")),
        "第三近 assistant 正文应完整保留: {messages:?}"
    );
    assert!(
        messages
            .iter()
            .any(|m| value_to_string(&m.content).contains("second-asst-")),
        "第二近 assistant 正文应完整保留: {messages:?}"
    );
    assert!(
        messages
            .iter()
            .any(|m| value_to_string(&m.content).contains("recent-asst-")),
        "最近 assistant 正文应完整保留: {messages:?}"
    );
}

#[test]
fn merge_old_user_overflow_stubs_migrates_role_of_single_old_merged_stub() {
    // 旧版快照可能只有一条 role=user 的合并指针、无新增单条 stub。
    // 即使不触发重新折叠，角色也必须迁移为 internal_note。
    let mut messages = vec![
        old_merged_stub(4, "/tmp/assets/overflow"),
        msg("assistant", "old-resp"),
        msg("user", "turn-2"),
        msg("assistant", "resp-2"),
        msg("user", "turn-3"),
        msg("assistant", "resp-3"),
        msg("user", "recent"),
        msg("assistant", "recent-resp"),
    ];
    merge_old_user_overflow_stubs(&mut messages, 1);
    assert_eq!(
        messages[0].role, "internal_note",
        "旧版 role=user 合并指针应迁移为 internal_note: {messages:?}"
    );
}

#[test]
fn normalize_preserved_message_stubs_migrates_old_user_role_merged_stub() {
    // normalize 是发送给模型前的最后一道关卡，也必须兜底迁移角色。
    let mut messages = vec![old_merged_stub(4, "/tmp/assets/overflow")];
    normalize_preserved_message_stubs_for_model(&mut messages);
    assert_eq!(
        messages[0].role, "internal_note",
        "normalize 应把旧版 role=user 合并指针迁移为 internal_note: {messages:?}"
    );
}
