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
            "[preserved-message-pointer-v2]\ncontent_kind: user\narchive_file_path: {path}\n\
             Earlier user content was archived verbatim. This is context recovery metadata, not a new user request."
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
    merge_old_user_overflow_stubs(&mut messages, 1, None);

    // Four old stubs become one internal note; the three recent user boundaries remain.
    assert_eq!(messages.len(), 11, "{messages:?}");
    let merged = &messages[1];
    assert_eq!(merged.role, ROLE_INTERNAL_NOTE);
    let text = value_to_string(&merged.content);
    assert!(
        text.starts_with("[preserved-messages-pointer-v2]\narchived_message_count: 4"),
        "合并指针应携带条数: {text}"
    );
    assert!(
        text.contains(
            "archive_directory_path: /sessions/test-session/assets/preserved_user_overflow"
        ),
        "{text}"
    );
    // The merged pointer remains protected from later trimming.
    assert!(is_preserved_user_or_image_stub(&text));
    let user_messages: Vec<_> = messages
        .iter()
        .filter(|message| message.role == "user")
        .collect();
    assert_eq!(user_messages.len(), 3, "{messages:?}");
    assert!(value_to_string(&user_messages[0].content).contains("user-5"));
    assert!(value_to_string(&user_messages[1].content).contains("user-6"));
    assert_eq!(value_to_string(&user_messages[2].content), "最近的用户轮次");

    // Protecting two recent turns still leaves a non-zero summarization split.
    let summary_split = retained_turn_start(&messages, 2);
    assert!(
        summary_split > 0,
        "合并后不应退化为只有一个 user 边界: {messages:?}"
    );
    assert!(value_to_string(&messages[summary_split].content).contains("user-6"));
}

/// Normalization preserves every archive directory and the tree/read_file
/// recovery protocol without attempting to read a directory as a file.
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
    merge_old_user_overflow_stubs(&mut messages, 1, None);

    // 模拟修复前已经落入上下文快照的首版格式：它把目录误标成「归档文件」。
    // 新版本必须兼容恢复，不能只修复新生成的指针。
    let merged_idx = messages
        .iter()
        .position(|message| {
            value_to_string(&message.content).starts_with("[preserved-messages-pointer-v2]")
        })
        .unwrap();
    let legacy = value_to_string(&messages[merged_idx].content)
        .replace("archive_directory_path: ", "归档文件: ");
    messages[merged_idx].content = Value::String(legacy);

    // 模拟合并后的上下文进入下一次真实请求入口。
    let normalized = compress_messages_for_context(messages, 100_000, 0, 0, None, None);
    let merged = normalized
        .iter()
        .map(|message| value_to_string(&message.content))
        .find(|text| text.starts_with("[preserved-messages-pointer-v2]"))
        .expect("合并指针应在请求规范化后继续存在");

    for dir in [
        "/sessions/session-a/assets/preserved_user_overflow",
        "/sessions/session-b/assets/preserved_user_overflow",
    ] {
        assert!(
            merged.contains(&format!("archive_directory_path: {dir}")),
            "{merged}"
        );
    }
    assert!(
        merged.contains("use tree on each archive directory"),
        "{merged}"
    );
    assert!(
        merged.contains("read_file on an individual JSON file"),
        "{merged}"
    );
    assert!(
        merged.contains("never call read_file on a directory"),
        "{merged}"
    );
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
    merge_old_user_overflow_stubs(&mut messages, 1, None);
    for ordinal in 7..=10 {
        messages.insert(
            messages.len() - 1,
            user_stub(dir, &format!("202601{ordinal:02}T000000Z-user-{ordinal}")),
        );
    }
    merge_old_user_overflow_stubs(&mut messages, 1, None);

    // 一个聚合指针 + 两个结构边界 stub + 最新 user，消息数仍是 O(1)。
    assert_eq!(messages.len(), 4, "{messages:?}");
    let merged = value_to_string(&messages[0].content);
    assert!(
        merged.starts_with("[preserved-messages-pointer-v2]\narchived_message_count: 8"),
        "{merged}"
    );
    assert_eq!(
        merged.matches("archive_directory_path: ").count(),
        1,
        "{merged}"
    );
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
    merge_old_user_overflow_stubs(&mut messages, 1, None);
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
    merge_old_user_overflow_stubs(&mut messages, 1, None);
    assert_eq!(
        messages[0].role, "internal_note",
        "旧版 role=user 合并指针应迁移为 internal_note: {messages:?}"
    );
}

#[test]
fn normalize_preserved_message_stubs_migrates_old_user_role_merged_stub() {
    // Normalization is the final boundary before model serialization and must
    // migrate both the role and the legacy wire format.
    let mut messages = vec![old_merged_stub(4, "/tmp/assets/overflow")];
    normalize_preserved_message_stubs_for_model(&mut messages);
    assert_eq!(
        messages[0].role, "internal_note",
        "normalize 应把旧版 role=user 合并指针迁移为 internal_note: {messages:?}"
    );
    let text = value_to_string(&messages[0].content);
    assert!(
        text.starts_with("[preserved-messages-pointer-v2]"),
        "{text}"
    );
    assert!(text.contains("archive_directory_path: /tmp/assets/overflow"));
}

#[test]
fn normalize_preserved_message_stubs_migrates_legacy_single_pointer() {
    let mut messages = vec![msg(
        "user",
        "较早的用户图片内容已归档，原文未丢失。\n归档文件: /tmp/assets/image.json",
    )];

    normalize_preserved_message_stubs_for_model(&mut messages);

    let text = value_to_string(&messages[0].content);
    assert!(text.starts_with("[preserved-message-pointer-v2]"), "{text}");
    assert!(text.contains("content_kind: image"), "{text}");
    assert!(
        text.contains("archive_file_path: /tmp/assets/image.json"),
        "{text}"
    );
}

/// 回归：生产路径第二轮压缩（summary_max_chars == 0）不得丢弃
/// prepare_turn 已生成的早期对话摘要 note / 归档 note。
#[test]
fn second_pass_zero_summary_budget_keeps_early_summary_and_archive_notes() {
    let messages = vec![
        note("历史摘要（自动压缩）: 早期对话的压缩表示……"),
        note("长期记忆归档\n较早的内部上下文注记已逐字归档。"),
        msg("user", "u1"),
        msg("assistant", "a1"),
        msg("user", "u2"),
        msg("assistant", "a2"),
        msg("user", "最近的用户轮次"),
    ];
    // keep_last=1：仅最近一个真实 user 轮次留在尾窗，摘要/归档 note 落入 older 段。
    // summary_max_chars=0：不重建摘要，旧摘要 note 必须像 checkpoint 一样保留，
    // 否则 orchestrator 的第二轮压缩会静默丢掉 prepare_turn 生成的摘要。
    let out = compress_messages_for_context(messages, 1_000_000, 1, 0, None, None);
    let texts: Vec<String> = out.iter().map(|m| value_to_string(&m.content)).collect();
    assert!(
        texts.iter().any(|t| t.starts_with("历史摘要（自动压缩")),
        "第二轮压缩不得丢弃早期摘要 note: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.starts_with("长期记忆归档")),
        "第二轮压缩不得丢弃归档 note: {texts:?}"
    );
    assert!(
        out.iter()
            .any(|m| m.role == "user" && value_to_string(&m.content) == "最近的用户轮次"),
        "最近的用户轮次必须保留: {texts:?}"
    );
}

/// A spilled old user message turns into archive metadata; the reminder it was
/// sent with must not be replayed onto the stub.
#[test]
fn spilled_user_message_does_not_replay_its_request_reminder() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-spill-reminder-{}", uuid::Uuid::new_v4()));
    let mut old_user = msg("user", &"long user input ".repeat(USER_OVERFLOW_SPILL_MIN_CHARS));
    crate::ai::history::attach_request_reminder(&mut old_user, "<system-reminder>x</system-reminder>");
    let mut messages = vec![
        msg("system", "sys"),
        old_user,
        msg("assistant", "a1"),
        msg("user", "u2"),
        msg("assistant", "a2"),
        msg("user", "u3"),
        msg("assistant", "a3"),
        msg("user", "最近的用户轮次"),
    ];
    assert!(try_spill_preserved_message_to_stub(&mut messages, &overflow_dir, 1_000));
    assert!(is_preserved_user_or_image_stub(&value_to_string(&messages[1].content)));
    assert_eq!(messages[1].reasoning_content, None);
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// New alias-form stubs merge exactly like legacy absolute ones: the merged
/// pointer keeps an absolute directory (the model passes it to `tree`), while
/// an unresolvable alias (no overflow dir this round) stays single instead of
/// merging with a wrong directory.
#[test]
fn merge_old_user_overflow_stubs_resolves_session_asset_aliases() {
    let root = std::env::temp_dir().join(format!("ai-merge-alias-{}", uuid::Uuid::new_v4()));
    let assets = root.join("session.assets");
    std::fs::create_dir_all(assets.join(PRESERVED_USER_OVERFLOW_DIR)).unwrap();
    let alias_stub = |name: &str| {
        msg(
            "user",
            &format!(
                "[preserved-message-pointer-v2]\ncontent_kind: user\narchive_file_path: session-asset:{}/{name}\nEarlier user content was archived verbatim.",
                PRESERVED_USER_OVERFLOW_DIR
            ),
        )
    };
    let fixture = || {
        let mut messages = vec![msg("system", "sys")];
        for index in 1..=6 {
            messages.push(alias_stub(&format!("user-{index}.json")));
            messages.push(msg("assistant", &format!("a{index}")));
        }
        messages.push(msg("user", "recent turn"));
        messages
    };
    let mut messages = fixture();
    merge_old_user_overflow_stubs(&mut messages, 1, Some(&assets));
    let merged = messages
        .iter()
        .find(|message| {
            value_to_string(&message.content).starts_with("[preserved-messages-pointer-v2]")
        })
        .expect("alias stubs must merge with an overflow dir");
    assert_eq!(merged.role, ROLE_INTERNAL_NOTE);
    let expected_dir = assets.join(PRESERVED_USER_OVERFLOW_DIR);
    assert!(
        value_to_string(&merged.content).contains(&format!(
            "archive_directory_path: {}",
            expected_dir.display()
        )),
        "merged pointer must carry the resolved absolute dir: {}",
        value_to_string(&merged.content)
    );

    // Without an overflow dir the aliases cannot resolve: they stay single
    // (reachable) instead of merging with a bogus directory.
    let mut messages = fixture();
    merge_old_user_overflow_stubs(&mut messages, 1, None);
    assert!(
        messages
            .iter()
            .all(|message| !value_to_string(&message.content)
                .starts_with("[preserved-messages-pointer-v2]")),
        "unresolvable aliases must stay single: {messages:?}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// Content-addressed spill naming: re-spilling identical content reuses one
/// file instead of orphaning a timestamp+uuid file per round.
#[test]
fn spill_reuses_one_file_for_identical_content() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-spill-dedup-{}", uuid::Uuid::new_v4()));
    let body = "long user input ".repeat(USER_OVERFLOW_SPILL_MIN_CHARS);
    let mut messages = vec![
        msg("system", "sys"),
        msg("user", &body),
        msg("assistant", "a1"),
        msg("user", &body),
        msg("assistant", "a2"),
        msg("user", "u3"),
        msg("assistant", "a3"),
        msg("user", "u4"),
        msg("assistant", "a4"),
        msg("user", "recent turn"),
    ];
    assert!(try_spill_preserved_message_to_stub(
        &mut messages,
        &overflow_dir,
        1_000
    ));
    assert!(try_spill_preserved_message_to_stub(
        &mut messages,
        &overflow_dir,
        1_000
    ));
    let first = value_to_string(&messages[1].content);
    let second = value_to_string(&messages[3].content);
    assert!(is_preserved_user_or_image_stub(&first));
    assert!(is_preserved_user_or_image_stub(&second));
    assert_eq!(
        first, second,
        "identical content must spill to the same alias"
    );
    let entries: Vec<_> = std::fs::read_dir(overflow_dir.join(PRESERVED_USER_OVERFLOW_DIR))
        .unwrap()
        .collect();
    assert_eq!(entries.len(), 1, "identical spills must share one file");
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// Old sessions carry absolute-path single stubs (e.g. in persisted
/// snapshots); normalization rebuilds them into session-relative aliases
/// without needing the session directory.
#[test]
fn normalize_rebuilds_legacy_absolute_spill_stub_into_alias() {
    let absolute = format!(
        "/sessions/old/assets/{}/20260101T000000Z-user-abc.json",
        PRESERVED_USER_OVERFLOW_DIR
    );
    let mut messages = vec![msg(
        "user",
        &format!(
            "[preserved-message-pointer-v2]\ncontent_kind: user\narchive_file_path: {absolute}\nEarlier user content was archived verbatim."
        ),
    )];
    normalize_preserved_message_stubs_for_model(&mut messages);
    let text = value_to_string(&messages[0].content);
    assert!(
        text.contains("archive_file_path: session-asset:user-overflow-preserved/20260101T000000Z-user-abc.json"),
        "legacy absolute stub must rebuild into an alias: {text}"
    );
    assert!(!text.contains("/sessions/old/assets"), "{text}");
    assert_eq!(messages[0].role, "user");
}
