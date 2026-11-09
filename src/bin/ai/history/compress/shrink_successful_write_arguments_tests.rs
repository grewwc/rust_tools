use super::truncate::build_context_overflow_tool_arguments_stub;
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

fn assistant_call_args(id: &str, name: &str, args: &str) -> Message {
    Message {
        role: "assistant".to_string(),
        content: Value::String(String::new()),
        tool_calls: Some(vec![ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: args.to_string(),
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

fn big_write_args() -> String {
    format!(
        r#"{{"file_path": "/tmp/out.txt", "content": "{}"}}"#,
        "x".repeat(5000)
    )
}

fn temp_overflow_dir() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("ai-shrink-write-{}", uuid::Uuid::new_v4()))
}

/// A successful write outside the recent window becomes a metadata-only stub;
/// its original arguments are archived verbatim.
#[test]
fn shrink_successful_write_outside_window() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let big_args = big_write_args();
    messages.push(assistant_call_args("call-w", "write_file", &big_args));
    messages.push(tool_result("call-w", "Successfully wrote to /tmp/out.txt"));
    // 占满保护窗口（最近 4 组），让 write_file 组滑出窗口。
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);

    let stub = &messages[3].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments;
    let archive_path = overflow_dir.join(OVERFLOW_HISTORY_FILENAME);
    let parsed: Value = serde_json::from_str(stub).expect("stub must remain valid JSON");
    assert_eq!(
        parsed,
        serde_json::json!({
            "_context_overflow_truncated": true,
            "original_chars": big_args.chars().count(),
            "archive_file_path": archive_path.to_string_lossy(),
        }),
        "stub must contain only recovery metadata, without preview or payload keys"
    );
    assert!(is_context_overflow_truncated_tool_arguments(stub));
    assert_eq!(messages[3].tool_calls.as_ref().unwrap()[0].id, "call-w");
    assert_eq!(messages[4].tool_call_id.as_deref(), Some("call-w"));
    assert_eq!(
        messages[4].content.as_str(),
        Some("Successfully wrote to /tmp/out.txt")
    );
    // The original arguments are archived without altering their paired result.
    assert!(
        archive_path.is_file(),
        "originals archived to overflow-history.md"
    );
    let archived = std::fs::read_to_string(&archive_path).unwrap();
    assert!(
        archived.contains(&big_args),
        "archive keeps the complete original arguments"
    );
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// 保护窗口内（最近 KEEP_RECENT_TOOL_GROUPS 组）的成功 write_file：arguments 原样保留，
/// 模型可能仍在引用刚写入的正文构造后续编辑。
#[test]
fn keep_write_arguments_inside_recent_window() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let big_args = big_write_args();
    messages.push(assistant_call_args("call-w", "write_file", &big_args));
    messages.push(tool_result("call-w", "Successfully wrote to /tmp/out.txt"));

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);

    let args = &messages[2].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments;
    assert_eq!(args, &big_args, "recent window keeps full arguments");
    assert!(!overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists());
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// 失败结果（Error: 开头）：保留完整 arguments，模型需要依据原文修复/重试。
#[test]
fn failed_write_keeps_arguments() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let big_args = big_write_args();
    messages.push(assistant_call_args("call-w", "write_file", &big_args));
    messages.push(tool_result(
        "call-w",
        "Error: write_file failed: permission denied",
    ));
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);

    let args = &messages[2].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments;
    assert_eq!(args, &big_args, "failed write keeps full arguments");
    assert!(!overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists());
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// 窗口外成功 apply_patch：patch 正文同样被替换为 stub。
#[test]
fn shrink_successful_apply_patch_outside_window() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let big_args = format!(
        r#"{{"patch": "*** Begin Patch\n*** Update File: /tmp/a.rs\n@@\n-{}\n+{}\n*** End Patch"}}"#,
        "old line".repeat(200),
        "new line".repeat(200)
    );
    messages.push(assistant_call_args("call-p", "apply_patch", &big_args));
    messages.push(tool_result(
        "call-p",
        "Successfully patched /tmp/a.rs; +1 -1 (1 lines)",
    ));
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);

    let stub = &messages[3].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments;
    assert!(stub.contains("\"_context_overflow_truncated\""));
    assert!(overflow_dir.join(OVERFLOW_HISTORY_FILENAME).is_file());
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// 短 arguments（≤160 字符）不触发：没有可释放的空间，保持原样。
#[test]
fn short_arguments_unchanged() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let small_args = r#"{"file_path": "/tmp/a.txt", "content": "hi"}"#.to_string();
    messages.push(assistant_call_args("call-s", "write_file", &small_args));
    messages.push(tool_result("call-s", "Successfully wrote to /tmp/a.txt"));
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);

    let args = &messages[2].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments;
    assert_eq!(args, &small_args);
    assert!(!overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists());
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// 幂等：已替换为 stub 的调用不再重复归档/替换。
#[test]
fn idempotent_when_already_replaced() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let big_args = big_write_args();
    messages.push(assistant_call_args("call-w", "write_file", &big_args));
    messages.push(tool_result("call-w", "Successfully wrote to /tmp/out.txt"));
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);
    let stub_after_first = messages[3].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments
        .clone();
    let archive_after_first = std::fs::read_to_string(overflow_dir.join(OVERFLOW_HISTORY_FILENAME))
        .unwrap()
        .len();

    // 第二次调用：stub 不变，归档文件不再增长。
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);
    let stub_after_second = &messages[3].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments;
    assert_eq!(stub_after_second, &stub_after_first);
    let archive_after_second =
        std::fs::read_to_string(overflow_dir.join(OVERFLOW_HISTORY_FILENAME))
            .unwrap()
            .len();
    assert_eq!(
        archive_after_second, archive_after_first,
        "no duplicate archive"
    );
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// 当前轮保护 id（protected_tool_call_ids）：即使窗口外也保留完整 arguments。
#[test]
fn protected_call_ids_never_shrunk() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let big_args = big_write_args();
    messages.push(assistant_call_args("call-w", "write_file", &big_args));
    messages.push(tool_result("call-w", "Successfully wrote to /tmp/out.txt"));
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let mut protected = FxHashSet::default();
    protected.insert("call-w".to_string());
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);

    let args = &messages[2].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments;
    assert_eq!(args, &big_args);
    assert!(!overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists());
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// 非写入工具（read_file）即使 arguments 很大也不处理：只针对 write_file/apply_patch。
#[test]
fn non_write_tools_ignored() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let big_args = format!(r#"{{"file_path": "/tmp/a.rs", "limit": 1000}}"#);
    messages.push(assistant_call_args("call-r", "read_file", &big_args));
    messages.push(tool_result("call-r", "some content"));
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);

    let args = &messages[2].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments;
    assert_eq!(args, &big_args);
    assert!(!overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists());
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// Without an overflow directory, the metadata-only stub has no archive path
/// and remains terminal rather than being re-archived as the original payload.
#[test]
fn no_overflow_dir_still_works() {
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    let big_args = big_write_args();
    messages.push(assistant_call_args("call-w", "write_file", &big_args));
    messages.push(tool_result("call-w", "Successfully wrote to /tmp/out.txt"));
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, None, &protected);

    let stub = &messages[2].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments;
    assert_eq!(
        serde_json::from_str::<Value>(stub).unwrap(),
        serde_json::json!({
            "_context_overflow_truncated": true,
            "original_chars": big_args.chars().count(),
            "archive_file_path": Value::Null,
        })
    );
    assert!(is_context_overflow_truncated_tool_arguments(stub));
    let first = stub.clone();
    assert!(!truncate_mutable_field(
        &mut messages[2],
        MutableMessageField::ToolArguments(0),
        usize::MAX,
        None,
        FieldArchivePolicy::BestEffort,
    ));
    assert_eq!(
        messages[2].tool_calls.as_ref().unwrap()[0]
            .function
            .arguments,
        first
    );
}

#[test]
fn tool_argument_pointer_keeps_original_chars_and_drops_legacy_preview() {
    let legacy = serde_json::json!({
        "_context_overflow_truncated": true,
        "original_chars": 5_000,
        "archive_file_path": "/tmp/archive.md",
        "preview": "{\"file_path\":\"/tmp/out.txt\",\"content\":\"partial payload\"}",
    })
    .to_string();
    let pointer = build_context_overflow_tool_arguments_pointer(&legacy, 160)
        .expect("legacy preview should minimize to a metadata-only pointer");
    assert_eq!(
        serde_json::from_str::<Value>(&pointer).unwrap(),
        serde_json::json!({
            "_context_overflow_truncated": true,
            "original_chars": 5_000,
            "archive_file_path": "/tmp/archive.md",
        })
    );
    assert!(is_context_overflow_truncated_tool_arguments(&pointer));
    assert!(build_context_overflow_tool_arguments_pointer(&pointer, usize::MAX).is_none());
}

/// 回归：中等大小（约 161..=240 字符）成功 write_file 参数——替换成 stub 不会比原文
/// 更短——既不替换也不归档。修复前这里会「先归档、后判定失败」：归档文件被写入但
/// 消息未改，下一轮压缩同一参数再次成为候选，导致每轮重复归档、溢出文件无界增长。
#[test]
fn medium_arguments_not_shrunk_and_not_archived() {
    let overflow_dir = temp_overflow_dir();
    let mut messages = vec![msg("system", "s"), msg("user", "do")];
    // 约 200 字符：超过触发阈值（160），但 stub 骨架（含归档路径）已接近原文长度，
    // 替换不可能严格变短。
    let medium_args = format!(
        r#"{{"file_path": "/tmp/med.txt", "content": "{}"}}"#,
        "m".repeat(160)
    );
    assert!(
        medium_args.chars().count() > 160 && medium_args.chars().count() <= 240,
        "medium_args chars = {}",
        medium_args.chars().count()
    );
    messages.push(assistant_call_args("call-m", "write_file", &medium_args));
    messages.push(tool_result("call-m", "Successfully wrote to /tmp/med.txt"));
    for i in 0..4 {
        let id = format!("call-{i}");
        messages.push(assistant_call_args(&id, "text_grep", r#"{"pattern": "x"}"#));
        messages.push(tool_result(&id, "no match"));
    }

    let protected = FxHashSet::default();
    shrink_successful_write_arguments(&mut messages, Some(&overflow_dir), &protected);

    // 替换失败：arguments 原样保留（用 id 定位，避免依赖 note 插入位置）。
    let write_idx = messages
        .iter()
        .position(|m| {
            m.tool_calls
                .as_ref()
                .is_some_and(|calls| calls[0].id == "call-m")
        })
        .unwrap();
    let args = &messages[write_idx].tool_calls.as_ref().unwrap()[0]
        .function
        .arguments;
    assert_eq!(
        args, &medium_args,
        "medium args stay intact when stub cannot shrink"
    );
    // 关键断言：没有写入任何归档条目（修复前这里会写一份永远用不上的重复原文）。
    assert!(
        !overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists(),
        "no archive write when truncation is impossible"
    );
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

/// 长归档路径吃光全部 target 预算时，ToolArguments stub 的 preview 会退化为空串。
/// 修复前这种 `"preview": ""` stub 会被写回历史，模型随后把
/// `_context_overflow_truncated` / `archive_file_path` / `preview` 当成真实参数名
/// 回发，形成死循环（见会话 ab41bc6d）。修复后应直接放弃截断、保留原 arguments，
/// 且不落盘归档。
#[test]
fn empty_preview_stub_rejected_when_path_eats_budget() {
    // 构造超长 overflow 目录名，使归档路径长度逼近/超过 target(160)。
    let long_segment = "d".repeat(140);
    let overflow_dir = std::env::temp_dir().join(long_segment);
    let original_args = format!(
        r#"{{"file_path":"/tmp/x.txt","content":"{}"}}"#,
        "c".repeat(400)
    );
    let mut message = assistant_call_args("call-x", "write_file", &original_args);
    // original≈460, target=160：reduce_by=300。
    let reduced = truncate_mutable_field(
        &mut message,
        MutableMessageField::ToolArguments(0),
        300,
        Some(&overflow_dir),
        FieldArchivePolicy::BestEffort,
    );
    assert!(
        !reduced,
        "must reject truncation when preview budget is exhausted"
    );
    let args = &message.tool_calls.as_ref().unwrap()[0].function.arguments;
    assert_eq!(args, &original_args, "arguments must be left intact");
    assert!(
        !overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists(),
        "no archive write when truncation is rejected"
    );
}

#[test]
fn metadata_only_tool_argument_stub_rejects_long_archive_path() {
    let target = 160;
    let original_args = big_write_args();
    let original_chars = original_args.chars().count();
    let overflow_dir = temp_overflow_dir().join("d".repeat(180));
    let archive_path = overflow_dir.join(OVERFLOW_HISTORY_FILENAME);
    let archive_path = archive_path.to_string_lossy();
    assert!(archive_path.chars().count() > target);
    assert!(
        build_context_overflow_tool_arguments_stub(original_chars, Some(&archive_path), target)
            .is_none(),
        "the final metadata builder must reject an oversized archive locator"
    );

    let mut message = assistant_call_args("call-long-path", "write_file", &original_args);
    assert!(!truncate_mutable_field(
        &mut message,
        MutableMessageField::ToolArguments(0),
        original_chars - target,
        Some(&overflow_dir),
        FieldArchivePolicy::BestEffort,
    ));
    assert_eq!(
        message.tool_calls.as_ref().unwrap()[0].function.arguments,
        original_args
    );
    assert!(!overflow_dir.join(OVERFLOW_HISTORY_FILENAME).exists());

    let no_path = build_context_overflow_tool_arguments_stub(original_chars, None, target)
        .expect("the existing no-path shape should still fit");
    assert_eq!(
        serde_json::from_str::<Value>(&no_path).unwrap(),
        serde_json::json!({
            "_context_overflow_truncated": true,
            "original_chars": original_chars,
            "archive_file_path": Value::Null,
        })
    );
    assert!(is_context_overflow_truncated_tool_arguments(&no_path));
    let pointer_chars = no_path.chars().count();
    assert!(pointer_chars <= target && pointer_chars < original_chars);
    assert!(
        build_context_overflow_tool_arguments_stub(original_chars, None, pointer_chars).is_some()
    );
    assert!(
        build_context_overflow_tool_arguments_stub(original_chars, None, pointer_chars - 1)
            .is_none()
    );
    assert!(build_context_overflow_tool_arguments_stub(1, None, target).is_none());
}
