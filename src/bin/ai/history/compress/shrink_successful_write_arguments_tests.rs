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

/// 窗口外成功 write_file：大 content 被替换为 `_context_overflow_truncated` 指针 stub，
/// 原文零压缩归档到 overflow 目录。
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
    assert!(
        stub.contains("\"_context_overflow_truncated\""),
        "stub should mark truncation: {stub}"
    );
    assert!(
        stub.contains("file_path"),
        "stub preview keeps file_path anchor"
    );
    assert!(
        stub.contains("archive_file_path"),
        "stub keeps archive pointer"
    );
    // 原文归档：overflow 目录下出现归档文件，且不含窗口内组。
    let archive_path = overflow_dir.join(OVERFLOW_HISTORY_FILENAME);
    assert!(
        archive_path.is_file(),
        "originals archived to overflow-history.md"
    );
    let archived = std::fs::read_to_string(&archive_path).unwrap();
    assert!(
        archived.contains("\"content\""),
        "archived keeps original args"
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

/// 无归档目录（overflow_dir = None）时：不崩溃，stub 仍生效（内联截断预览）。
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
    assert!(stub.contains("\"_context_overflow_truncated\""));
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
