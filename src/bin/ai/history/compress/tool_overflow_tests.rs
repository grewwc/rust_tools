use super::tool_overflow::is_preserved_tool_overflow_stub;
use super::*;
use crate::ai::types::{FunctionCall, ToolCall};
use rustc_hash::FxHashSet;

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
        None,
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
        None,
        &FxHashSet::default(),
    );
    assert_eq!(value_to_string(&messages[1].content), first_stub);
    assert_eq!(std::fs::read_dir(&overflow_path).unwrap().count(), 1);

    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn pruned_stable_archive_is_content_addressed() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-pruned-stable-content-addressed-{}",
        uuid::Uuid::new_v4()
    ));

    // Same (tool, id, content) must map to one idempotent archive file.
    let first = write_preserved_tool_overflow_file_stable(
        &overflow_dir,
        "call-1",
        "read_file",
        "result body",
    )
    .unwrap();
    let second = write_preserved_tool_overflow_file_stable(
        &overflow_dir,
        "call-1",
        "read_file",
        "result body",
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(std::fs::read_to_string(&first).unwrap(), "result body");

    // Same call id with different content must NOT reuse the old file:
    // a replayed tool call would otherwise read back stale bytes.
    let replayed = write_preserved_tool_overflow_file_stable(
        &overflow_dir,
        "call-1",
        "read_file",
        "replayed body",
    )
    .unwrap();
    assert_ne!(first, replayed);
    assert_eq!(std::fs::read_to_string(&replayed).unwrap(), "replayed body");
    let overflow_path = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    assert_eq!(std::fs::read_dir(&overflow_path).unwrap().count(), 2);

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
        None,
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
        stub.contains("Archived snapshot of an earlier read"),
        "stub: {stub}"
    );

    let anchor = collapse_overflow_stub_to_anchor(&stub).expect("stub should collapse");
    assert!(
        anchor.contains("- original_file_path: src/lib.rs"),
        "anchor: {anchor}"
    );
    assert!(
        anchor.contains("Archived snapshot of an earlier read"),
        "anchor: {anchor}"
    );

    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn preserved_stub_preview_includes_line_numbered_key_lines() {
    // read_file output carries a `{:>6}\t` line-number prefix: key_lines
    // should parse the prefix and use real line numbers for L tags (rather
    // than failing every match because of the prefix), so long files remain
    // locatable by line number after being spilled.
    let content = "\
 1\tuse std::fmt;\n\
 2\t\n\
 3\tpub fn main() {\n\
 4\t    let x = 1;\n\
 5\t}\n\
 6\tfn helper() {}\n\
 7\t//! crate docs\n\
 8\tstruct Foo;\n";
    let preview = build_overflow_content_preview(content);
    assert!(preview.contains("- key_lines (5):"), "preview: {preview}");
    assert!(preview.contains("L1: use std::fmt;"), "preview: {preview}");
    assert!(preview.contains("L3: pub fn main()"), "preview: {preview}");
    assert!(preview.contains("L6: fn helper()"), "preview: {preview}");
    assert!(preview.contains("L8: struct Foo;"), "preview: {preview}");
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
        None,
        &FxHashSet::default(),
    );
    let stub = value_to_string(&messages[1].content);
    assert!(
        stub.contains("- original_command: git log --stat"),
        "stub: {stub}"
    );
    assert!(stub.contains("- original_cwd: /repo"), "stub: {stub}");
    assert!(
        stub.contains("Continue from `original_command` / `original_cwd`"),
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
        None,
        &FxHashSet::default(),
        false,
    );

    assert!(is_preserved_tool_overflow_stub(&value_to_string(
        &messages[1].content
    )));
    assert_eq!(value_to_string(&messages[2].content).len(), 10_000);
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn precision_budget_never_expands_small_results_into_larger_stubs() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-precision-group-budget-{}",
        uuid::Uuid::new_v4()
    ));
    let mut call = assistant_call("small", "read_file");
    call.tool_calls.as_mut().unwrap().push(ToolCall {
        id: "big".to_string(),
        tool_type: "function".to_string(),
        function: FunctionCall {
            name: "read_file".to_string(),
            arguments: "{}".to_string(),
        },
    });
    let mut messages = vec![
        call,
        tool_result("small", &"s".repeat(100)),
        tool_result("big", &"b".repeat(10_000)),
    ];

    enforce_protected_precision_group_budget(
        &mut messages,
        1,
        200,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
        false,
    );

    // A small result (100 chars) bloats when swapped for a stub with a long
    // path: the original text must stay inline.
    assert_eq!(value_to_string(&messages[1].content), "s".repeat(100));
    // The large result is spilled into a stub, and the stub is strictly
    // shorter than the original.
    let stub = value_to_string(&messages[2].content);
    assert!(is_preserved_tool_overflow_stub(&stub), "{stub}");
    assert!(stub.chars().count() < 10_000, "{stub}");
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn enforce_group_budget_reuses_reread_archive_asset_instead_of_rearchiving() {
    // When a read_file result that read back an archived asset (no longer
    // protected after crossing turns) enters group spilling again, the
    // existing archive file must be reused (the stub points at the same
    // file) instead of minting a randomly named new one — otherwise "spill
    // → read-back → spill again" generates a new archive on every
    // read-back, forming an unbounded chain where the model never gets
    // stable content.
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-precision-group-reuse-{}", uuid::Uuid::new_v4()));
    // 1) Generate an archive asset via Path C
    let mut messages = vec![
        assistant_call("spill", "read_file"),
        // Leave ample headroom in the payload: the fingerprint adds a fixed
        // ~16-byte overhead to the stub, and if the original and stub sizes
        // were extremely close the anti-bloat guard (stub>=original) could
        // flip, making this test about reuse semantics rather than byte
        // coincidence.
        tool_result("spill", &"x".repeat(4_000)),
        assistant_call("recent", "read_file"),
        tool_result("recent", "recent result"),
    ];
    let mut protected = FxHashSet::default();
    protected.insert("spill".to_string());
    let stub1 =
        spill_protected_precision_to_fit(&mut messages, 80, Some(&overflow_dir), None, &protected);
    assert!(stub1 > 0);
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    let archive_path = std::fs::read_dir(&archive_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let raw = std::fs::read_to_string(&archive_path).unwrap();

    // 2) The model reads the archive back: the result (1000 chars) exceeds
    // the group inline budget → triggers the enforce spill
    let read_args = serde_json::json!({ "file_path": archive_path.to_string_lossy() });
    let mut messages = vec![
        assistant_call_args("re-read", "read_file", &read_args.to_string()),
        tool_result("re-read", &raw),
    ];
    enforce_protected_precision_group_budget(
        &mut messages,
        1,
        120,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
        false,
    );

    // 3) The existing asset is reused: the directory still has exactly 1
    // file, and the stub points at the same archive_path
    assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
    let stub_text = value_to_string(&messages[1].content);
    assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
    assert!(
        stub_text.contains(archive_path.to_str().unwrap()),
        "{stub_text}"
    );
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn cap_oversized_reuses_reread_archive_asset_instead_of_rearchiving() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-cap-reread-reuse-{}", uuid::Uuid::new_v4()));
    // 1) First let the cap itself write an archive asset
    let mut messages = vec![
        assistant_call("first", "read_file"),
        tool_result("first", &"y".repeat(70_000)),
    ];
    let capped =
        cap_oversized_tool_results_for_context(&mut messages, 64_000, Some(&overflow_dir), None);
    assert!(capped > 0);
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    let archive_path = std::fs::read_dir(&archive_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let raw = std::fs::read_to_string(&archive_path).unwrap();

    // 2) The model reads the archive back (body 70k > the 64k hard cap) →
    // the existing file is reused, no new file written
    let read_args = serde_json::json!({ "file_path": archive_path.to_string_lossy() });
    let mut messages = vec![
        assistant_call_args("re-read", "read_file", &read_args.to_string()),
        tool_result("re-read", &raw),
    ];
    let capped =
        cap_oversized_tool_results_for_context(&mut messages, 64_000, Some(&overflow_dir), None);
    assert_eq!(capped, 1);
    assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
    let stub_text = value_to_string(&messages[1].content);
    assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
    assert!(
        stub_text.contains(archive_path.to_str().unwrap()),
        "{stub_text}"
    );
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn prepare_structured_reuses_reread_archive_asset_instead_of_rearchiving() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-prepare-reread-reuse-{}", uuid::Uuid::new_v4()));
    // 1) First let prepare write an archive asset (an old read_file result,
    // over the 480 threshold and outside the tail window)
    let mut messages = vec![
        assistant_call("first", "read_file"),
        tool_result("first", &"z".repeat(2_000)),
    ];
    prepare_tool_messages_structured(
        &mut messages,
        480,
        0,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    let archive_path = std::fs::read_dir(&archive_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let raw = std::fs::read_to_string(&archive_path).unwrap();

    // 2) The model reads the archive back and the result enters prepare
    // again (unprotected, outside the tail window) → the existing file is
    // reused
    let read_args = serde_json::json!({ "file_path": archive_path.to_string_lossy() });
    let mut messages = vec![
        assistant_call_args("re-read", "read_file", &read_args.to_string()),
        tool_result("re-read", &raw),
    ];
    prepare_tool_messages_structured(
        &mut messages,
        480,
        0,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
    let stub_text = value_to_string(&messages[1].content);
    assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
    assert!(
        stub_text.contains(archive_path.to_str().unwrap()),
        "{stub_text}"
    );
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn prepare_structured_spill_is_deterministic_across_reprojections() {
    // P1 regression: when the same canonical tool result is spilled in two
    // independent projections, it must map to the same deterministic
    // archive file rather than minting a randomly named new copy every
    // round (the old behavior caused unbounded bloat within one session:
    // 368 files for only 211 unique contents). Using **different**
    // overflow_dirs for the two projections would hide idempotence, so the
    // same dir is reused to simulate round-by-round compaction of one
    // session.
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-prepare-deterministic-spill-{}",
        uuid::Uuid::new_v4()
    ));
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    // A single very long line: after line truncation the preview makes the
    // stub significantly smaller than the original → it does spill (the
    // anti-bloat guard does not trigger).
    let big = "b".repeat(4_000);
    let build = || {
        vec![
            assistant_call("spill", "read_file"),
            tool_result("spill", &big),
            assistant_call("recent", "read_file"),
            tool_result("recent", "recent result"),
        ]
    };

    let mut first = build();
    prepare_tool_messages_structured(
        &mut first,
        480,
        1,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    let first_stub = value_to_string(&first[1].content);
    assert!(is_preserved_tool_overflow_stub(&first_stub), "{first_stub}");
    assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);

    // Second projection: the same canonical result (same tool_call_id +
    // same body) is compacted again. Deterministic naming → the existing
    // file is hit, no new copy is added, and the stub text stays stable
    // across rounds.
    let mut second = build();
    prepare_tool_messages_structured(
        &mut second,
        480,
        1,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );
    let second_stub = value_to_string(&second[1].content);
    assert_eq!(
        second_stub, first_stub,
        "重投影后 stub 文本必须稳定（prompt cache 不断裂）"
    );
    assert_eq!(
        std::fs::read_dir(&archive_dir).unwrap().count(),
        1,
        "同一结果重复外溢不得铸造新归档文件"
    );

    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn prepare_structured_keeps_small_multiline_result_inline() {
    // P2 regression: a few-hundred-byte multi-line grep result (over
    // max_chars_per_msg but still small) becomes larger when swapped for a
    // stub with a full head/tail preview. The anti-bloat guard must keep
    // the original inline and write no archive file — otherwise the model
    // sees "evicted, please re-read" and reads back repeatedly (session
    // 9f4d0fae's "read results kept being archived as stubs" was exactly
    // this path: a 673-char grep swapped for a larger stub).
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-prepare-small-inline-{}", uuid::Uuid::new_v4()));
    // 20 lines × ~30 chars ≈ 600 chars: over max_chars_per_msg=480, but the
    // preview contains the whole body verbatim, so the stub cannot be
    // smaller than the original.
    let grep_like = (0..20)
        .map(|i| format!("src/bin/ai/mod.rs:{i}: use crate::ai::x;"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(grep_like.chars().count() > 480);
    let mut messages = vec![
        assistant_call("grep", "execute_command"),
        tool_result("grep", &grep_like),
        assistant_call("recent", "read_file"),
        tool_result("recent", "recent result"),
    ];

    prepare_tool_messages_structured(
        &mut messages,
        480,
        1,
        Some(&overflow_dir),
        None,
        &FxHashSet::default(),
    );

    let content = value_to_string(&messages[1].content);
    assert_eq!(content, grep_like, "小的多行精确结果必须保留原文内联");
    assert!(
        !is_preserved_tool_overflow_stub(&content),
        "不应被换成 stub"
    );
    // No archive file is written (on bloat the new file is deleted; here it
    // should never have been written at all).
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    let archived = std::fs::read_dir(&archive_dir)
        .map(|d| d.count())
        .unwrap_or(0);
    assert_eq!(archived, 0, "膨胀结果不得留下归档文件");

    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn cap_reuses_execute_command_cat_archive_instead_of_rearchiving() {
    let overflow_dir =
        std::env::temp_dir().join(format!("ai-cap-cat-reuse-{}", uuid::Uuid::new_v4()));
    // 1) First let the cap write an execute_command archive asset
    let mut messages = vec![
        assistant_call("run", "execute_command"),
        tool_result("run", &"log line\n".repeat(30_000)),
    ];
    let capped =
        cap_oversized_tool_results_for_context(&mut messages, 64_000, Some(&overflow_dir), None);
    assert!(capped > 0);
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    let archive_path = std::fs::read_dir(&archive_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let raw = std::fs::read_to_string(&archive_path).unwrap();

    // 2) The model reads the archive body back with `cat <archive>` (over
    // the hard cap) → the existing file is reused
    let run_args = serde_json::json!({
        "command": format!("cat {}", archive_path.to_string_lossy()),
        "pty": false,
    });
    let mut messages = vec![
        assistant_call_args("re-cat", "execute_command", &run_args.to_string()),
        tool_result("re-cat", &raw),
    ];
    let capped =
        cap_oversized_tool_results_for_context(&mut messages, 64_000, Some(&overflow_dir), None);
    assert_eq!(capped, 1);
    assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
    let stub_text = value_to_string(&messages[1].content);
    assert!(is_preserved_tool_overflow_stub(&stub_text), "{stub_text}");
    assert!(
        stub_text.contains(archive_path.to_str().unwrap()),
        "{stub_text}"
    );
    let _ = std::fs::remove_dir_all(&overflow_dir);
}

#[test]
fn path_c_spills_all_protected_precision_groups_without_recent_group_cap() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-global-precision-budget-{}",
        uuid::Uuid::new_v4()
    ));
    let mut messages = Vec::new();
    let mut protected = FxHashSet::default();
    for index in 0..8 {
        let id = format!("read-{index}");
        protected.insert(id.clone());
        messages.push(assistant_call(&id, "read_file"));
        messages.push(tool_result(&id, &"line of exact evidence\n".repeat(600)));
    }

    let spilled =
        spill_protected_precision_to_fit(&mut messages, 0, Some(&overflow_dir), None, &protected);

    // Covers the second half of Path C: when still over budget after the
    // spill, the emergency cap kicks in. Every preserved stub must first
    // shrink into a non-truncatable minimal pointer and must not be run
    // through generic head/tail truncation again.
    assert!(super::messages_total_chars(&messages) > 4_000);
    super::emergency_cap_messages_to_fit(
        &mut messages,
        4_000,
        160,
        Some(&overflow_dir),
        &protected,
    );

    assert_eq!(spilled, 8);
    let stubs = messages
        .iter()
        .filter_map(|message| {
            let content = value_to_string(&message.content);
            is_preserved_tool_overflow_stub(&content).then_some(content)
        })
        .collect::<Vec<_>>();
    assert_eq!(stubs.len(), 8);
    for stub in stubs {
        let file_path = stub
            .lines()
            .find_map(|line| line.strip_prefix("- file_path: "))
            .expect("minimal overflow stub must retain file_path");
        assert!(Path::new(file_path).is_file());
        assert!(!stub.contains("Preview ("));
    }
    assert!(super::messages_total_chars(&messages) <= 4_000);
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn path_c_reuses_reread_session_asset_instead_of_rearchiving_it() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let test_root =
        std::env::temp_dir().join(format!("ai-reread-session-asset-{}", uuid::Uuid::new_v4()));
    let effective_cwd = test_root.join("workspace");
    let overflow_dir = effective_cwd.join("session-assets");
    let archive_dir = overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR);
    std::fs::create_dir_all(&archive_dir).unwrap();
    let archive_path = archive_dir.join("prior-read.txt");
    let content = "previously preserved evidence\n".repeat(800);
    std::fs::write(&archive_path, &content).unwrap();
    let relative_archive_path = archive_path
        .strip_prefix(&effective_cwd)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let arguments = serde_json::json!({
        "file_path": relative_archive_path,
        "offset": 1,
        "limit": 10_000,
    })
    .to_string();
    let mut protected = FxHashSet::default();
    protected.insert("reread".to_string());
    let mut messages = vec![
        assistant_call_args("reread", "read_file", &arguments),
        tool_result("reread", &content),
    ];

    let spilled = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(effective_cwd, || {
        spill_protected_precision_to_fit(&mut messages, 0, Some(&overflow_dir), None, &protected)
    });

    assert_eq!(spilled, 1);
    let stub = value_to_string(&messages[1].content);
    let file_path = stub
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
        .expect("reused stub must retain the existing archive pointer");
    assert_eq!(Path::new(file_path), archive_path.canonicalize().unwrap());
    assert!(stub.contains("- original_range: lines=1..10000"));
    assert_eq!(std::fs::read_dir(&archive_dir).unwrap().count(), 1);
    assert_eq!(std::fs::read_to_string(&archive_path).unwrap(), content);
    let _ = std::fs::remove_dir_all(test_root);
}

#[test]
fn path_c_snapshots_mutable_session_temp_asset_instead_of_reusing_it() {
    let _guard = crate::ai::test_support::ENV_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let test_root =
        std::env::temp_dir().join(format!("ai-reread-session-temp-{}", uuid::Uuid::new_v4()));
    let effective_cwd = test_root.join("workspace");
    let overflow_dir = effective_cwd.join("session-assets");
    let temp_dir = overflow_dir.join("tmp");
    std::fs::create_dir_all(&temp_dir).unwrap();
    let temp_path = temp_dir.join("mutable.txt");
    let content = "temporary evidence before mutation\n".repeat(800);
    std::fs::write(&temp_path, &content).unwrap();
    let relative_temp_path = temp_path
        .strip_prefix(&effective_cwd)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let arguments = serde_json::json!({
        "file_path": relative_temp_path,
        "offset": 1,
        "limit": 10_000,
    })
    .to_string();
    let mut protected = FxHashSet::default();
    protected.insert("reread".to_string());
    let mut messages = vec![
        assistant_call_args("reread", "read_file", &arguments),
        tool_result("reread", &content),
    ];

    let spilled = crate::ai::driver::runtime_ctx::SUBAGENT_CWD.sync_scope(effective_cwd, || {
        spill_protected_precision_to_fit(&mut messages, 0, Some(&overflow_dir), None, &protected)
    });

    assert_eq!(spilled, 1);
    let stub = value_to_string(&messages[1].content);
    let snapshot_path = stub
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
        .map(PathBuf::from)
        .expect("mutable session file must be snapshotted into an overflow archive");
    assert_ne!(snapshot_path, temp_path.canonicalize().unwrap());
    assert!(snapshot_path.starts_with(overflow_dir.join(PRESERVED_TOOL_OVERFLOW_DIR)));
    assert_eq!(std::fs::read_to_string(&snapshot_path).unwrap(), content);

    std::fs::write(&temp_path, "temporary evidence after mutation\n").unwrap();
    assert_eq!(std::fs::read_to_string(&snapshot_path).unwrap(), content);
    let _ = std::fs::remove_dir_all(test_root);
}

#[test]
fn path_c_spills_aggregated_task_wait_result_losslessly() {
    // task_wait forbids lossy compression but occupies no inline budget;
    // Path C's global fallback must spill it losslessly with a file pointer
    // left behind, rather than excluding it from candidates and letting
    // later lossy truncation lose the aggregate truth.
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-global-precision-taskwait-{}",
        uuid::Uuid::new_v4()
    ));
    let mut protected = FxHashSet::default();
    protected.insert("wait".to_string());
    let mut messages = vec![
        assistant_call("wait", "task_wait"),
        tool_result("wait", &"aggregated subagent conclusion\n".repeat(600)),
    ];

    let spilled =
        spill_protected_precision_to_fit(&mut messages, 0, Some(&overflow_dir), None, &protected);

    assert_eq!(spilled, 1, "task_wait 大结果应被 Path C 无损外溢");
    let stub = value_to_string(&messages[1].content);
    assert!(
        is_preserved_tool_overflow_stub(&stub),
        "外溢后应替换为 overflow stub"
    );
    let file_path = stub
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("- file_path: "))
        .expect("overflow stub 必须保留可召回的 file_path 指针");
    assert!(Path::new(file_path.trim()).is_file(), "外溢原文必须落盘");
    let _ = std::fs::remove_dir_all(overflow_dir);
}

#[test]
fn path_c_does_not_expand_short_protected_results_into_stubs() {
    let overflow_dir = std::env::temp_dir().join(format!(
        "ai-global-precision-short-{}",
        uuid::Uuid::new_v4()
    ));
    let mut protected = FxHashSet::default();
    protected.insert("read-short".to_string());
    let mut messages = vec![
        assistant_call("read-short", "read_file"),
        tool_result("read-short", "ok"),
    ];
    let before = super::messages_total_chars(&messages);

    let spilled =
        spill_protected_precision_to_fit(&mut messages, 0, Some(&overflow_dir), None, &protected);

    assert_eq!(spilled, 0);
    assert_eq!(value_to_string(&messages[1].content), "ok");
    assert_eq!(super::messages_total_chars(&messages), before);
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

/// Builds a stub in its "first spill" shape (with a multi-line Preview
/// body), for fold testing.
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
    // Precondition: the first-spill stub really does carry a Preview body.
    assert!(stub.contains("Preview ("));

    let anchor = collapse_overflow_stub_to_anchor(&stub).expect("should collapse");
    // The preview body is discarded.
    assert!(!anchor.contains("Preview ("));
    // The file_path is kept.
    assert!(anchor.contains("- file_path: /tmp/session/read-abc.txt"));
    // The tool name is kept (the new format uses "Output preserved for
    // tool").
    assert!(anchor.contains("Output preserved for tool `read_file`"));
    // read_file-type archives carry the "usually no need to re-read" notice
    // (instead of the old leading "use read_file").
    assert!(anchor.contains("Archived snapshot of an earlier read"));
    // Still a valid stub (prefix unchanged); the downstream compaction
    // chain keeps recognizing it via the stub exemption.
    assert!(is_preserved_tool_overflow_stub(&anchor));
    // The size drops sharply.
    assert!(anchor.len() < stub.len());
}

#[test]
fn preserved_stub_carries_fingerprint_line() {
    let full = "Compiling rust_tools v0.1.0 (/repo)\n\
                warning: unused variable `root_idx`\n\
                error[E0308]: mismatched types in sched_ctx\n";
    let stub =
        build_preserved_tool_overflow_stub(Path::new("/tmp/fp.txt"), "execute_command", full, &[]);
    assert!(is_preserved_tool_overflow_stub(&stub));

    // Deterministic in content: same bytes -> byte-identical stub text.
    let stub_again =
        build_preserved_tool_overflow_stub(Path::new("/tmp/fp.txt"), "execute_command", full, &[]);
    assert_eq!(stub, stub_again);

    let fp_line = stub
        .lines()
        .find_map(|l| l.trim_start().strip_prefix("- fingerprint: "))
        .expect("fingerprint line present on fresh stub");
    // sha= segment: exactly 12 hex chars.
    let sha = fp_line
        .split("sha=")
        .nth(1)
        .unwrap()
        .split(' ')
        .next()
        .unwrap();
    assert_eq!(sha.len(), 12);
    assert!(sha.chars().all(|c| c.is_ascii_hexdigit()));
    // Keyword casing is preserved verbatim so tokens stay greppable in the archive.
    assert!(
        fp_line.contains("keys="),
        "keys= segment present: {fp_line}"
    );
    assert!(
        fp_line.contains("rust_tools") || fp_line.contains("root_idx"),
        "keywords: {fp_line}"
    );
}

#[test]
fn collapse_and_minimize_carry_fingerprint_through() {
    let full = "alpha beta gamma\nE0308 mismatched_types hit\n".repeat(30);
    let stub = build_preserved_tool_overflow_stub(
        Path::new("/tmp/carry.txt"),
        "execute_command",
        full.as_str(),
        &[],
    );

    let anchor = collapse_overflow_stub_to_anchor(&stub).expect("collapse");
    assert!(!anchor.contains("Preview ("));
    assert!(
        anchor.contains("- fingerprint: "),
        "anchor carries fingerprint: {anchor}"
    );

    let pointer = minimize_overflow_stub_to_pointer(&stub).expect("minimize");
    assert!(pointer.contains("- file_path: /tmp/carry.txt"));
    assert!(
        pointer.contains("- fingerprint: "),
        "pointer keeps retrieval signal"
    );
    assert!(is_preserved_tool_overflow_stub(&pointer));

    // Legacy stubs (pre-fingerprint) minimize cleanly without fabricated fields.
    let legacy = format!(
        "{PRESERVED_TOOL_OVERFLOW_STUB_PREFIX}\nOutput preserved for tool `x`.\n- file_path: /tmp/legacy.txt"
    );
    let minimized = minimize_overflow_stub_to_pointer(&legacy).unwrap();
    assert!(!minimized.contains("fingerprint"));
}

#[test]
fn fingerprint_skips_degenerate_gist() {
    // A single repeated character carries no recall signal; the gist segment is
    // omitted entirely so aged stubs of degenerate outputs stay minimal.
    let fp = stub_fingerprint_line(&"x".repeat(1_000)).expect("non-empty content has fingerprint");
    assert!(fp.contains("sha="), "{fp}");
    assert!(!fp.contains("gist="), "no gist for degenerate body: {fp}");

    // Real signal still gets a gist.
    let fp2 = stub_fingerprint_line("warning: sched_ctx drifted from kernel root\n");
    assert!(fp2.unwrap().contains("gist=\""), "informative line kept");
}

#[test]
fn fingerprint_keywords_dedup_case_insensitively_and_stay_deterministic() {
    // Mixed-case repeats of the same token must collapse to one keyword,
    // keeping the first-seen casing; the set-based dedup must not perturb
    // ordering vs. the previous linear scan.
    let content = "sched_ctx SCHED_CTX Sched_Ctx root_idx ROOT_IDX payloadxyz\n";
    let keys = extract_fingerprint_keywords(content);
    let sched = keys
        .iter()
        .filter(|k| k.eq_ignore_ascii_case("sched_ctx"))
        .count();
    assert_eq!(
        sched, 1,
        "case-insensitive dedup collapses repeats: {keys:?}"
    );
    assert!(
        keys.contains(&"sched_ctx".to_string()),
        "first casing kept: {keys:?}"
    );
    assert!(keys.len() <= FINGERPRINT_KEY_COUNT);

    // Fully deterministic across calls (no RNG / hash-order leakage into output).
    assert_eq!(keys, extract_fingerprint_keywords(content));
}

#[test]
fn age_out_overflow_stub_previews_is_idempotent() {
    let stub = overflow_stub_with_preview("/tmp/session/read-xyz.txt", "read_file");
    // Two user turns place the stub outside the protected tail window
    // (before retained_turn_start).
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

    // Run again: it is already in anchor shape and the content must not
    // change (prevents stub->stub churn).
    age_out_overflow_stub_previews(&mut messages, 1);
    assert_eq!(value_to_string(&messages[2].content), after_first);
}

#[test]
fn age_out_overflow_stub_previews_respects_protected_tail() {
    // One early stub (outside the tail window) and one recent stub (inside
    // the tail window).
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

    // Protect the most recent user turn (from q2 on): the early stub folds,
    // while the recent one inside the tail window keeps its full preview.
    age_out_overflow_stub_previews(&mut messages, 1);
    assert!(!value_to_string(&messages[2].content).contains("Preview ("));
    assert_eq!(value_to_string(&messages[5].content), recent);
    assert!(value_to_string(&messages[5].content).contains("Preview ("));
}
