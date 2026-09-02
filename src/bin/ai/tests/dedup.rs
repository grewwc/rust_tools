//! Byte-identical repeated read_file dedup tests.

use serde_json::Value;

use super::super::history::{Message, compress_messages_for_context};
use super::*;

#[test]
fn compression_collapses_byte_identical_repeated_read_file_but_keeps_changed_versions() {
    // Regression test: break the "repeated full re-read" amnesia loop.
    // When the same file is read repeatedly with **byte-identical content**, only one full copy should be kept,
    // with the redundant duplicates folded into back-reference stubs (lossless). Versions whose content truly changes (the file was edited)
    // must each be kept in full — identical signatures must never wrongly fold them into stubs and lose the real differences.
    let identical = format!("// agent_adapter.py\n{}", "A".repeat(6_000));
    let changed_v1 = format!("// controller.py v1\n{}", "B".repeat(6_000));
    let changed_v2 = format!("// controller.py v2 EDITED\n{}", "C".repeat(6_000));

    let mut messages = vec![Message {
        role: "system".to_string(),
        content: Value::String("system prompt".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    // 6 reads of the same file with identical content (simulating repeated full re-reads in an amnesia loop).
    for i in 0..6 {
        let (a, t) = read_file_call_pair(
            &format!("call_same_{i}"),
            "/repo/agent_adapter.py",
            &identical,
        );
        messages.push(a);
        messages.push(t);
    }
    // The same file gets edited, producing two distinct versions, each read once.
    let (a1, t1) = read_file_call_pair("call_ctrl_1", "/repo/controller.py", &changed_v1);
    messages.push(a1);
    messages.push(t1);
    let (a2, t2) = read_file_call_pair("call_ctrl_2", "/repo/controller.py", &changed_v2);
    messages.push(a2);
    messages.push(t2);

    // Append enough "near-end" tool messages to push all reads above out of the KEEP_RECENT protected window,
    // making sure dedup actually applies to them. Every message's content/path is unique so they are not folded by dedup themselves.
    for i in 0..8 {
        let (a, t) = read_file_call_pair(
            &format!("call_pad_{i}"),
            &format!("/repo/pad_{i}.py"),
            &format!("padding-{i}"),
        );
        messages.push(a);
        messages.push(t);
    }

    // overflow_dir=None: isolate the dedup behavior so the single retained full copy is not further offloaded
    // to a disk stub (the 480-char offload threshold is an orthogonal path covered by dedicated tests).
    let compressed = compress_messages_for_context(messages, 200_000, 256, 400, None, None);

    let full_identical = compressed
        .iter()
        .filter(|m| m.content.as_str() == Some(identical.as_str()))
        .count();
    assert_eq!(
        full_identical, 1,
        "byte-identical repeated read_file must collapse to exactly one full copy"
    );

    let dedup_stubs: Vec<&str> = compressed
        .iter()
        .filter_map(|m| m.content.as_str())
        .filter(|s| s.contains("byte-identical") && s.contains("No need to re-read"))
        .collect();
    assert_eq!(
        dedup_stubs.len(),
        5,
        "the other five identical reads must become re-read-suppressing dedup stubs"
    );
    for (i, stub) in dedup_stubs.iter().enumerate() {
        let call_id = format!("call_same_{i}");
        assert!(stub.contains("- original_tool_call_id: "), "{stub}");
        assert!(
            stub.contains("- canonical_tool_call_id: call_same_5"),
            "{stub}"
        );
        assert!(
            stub.contains(&format!("- original_tool_call_id: {call_id}")),
            "{stub}"
        );
        assert!(stub.contains("- canonical_message_index: "), "{stub}");
        assert!(
            stub.contains(r#""filePath":"/repo/agent_adapter.py""#),
            "{stub}"
        );
        assert!(
            stub.contains("- original_target: file=/repo/agent_adapter.py"),
            "{stub}"
        );
        assert!(stub.contains("- preview: // agent_adapter.py"), "{stub}");
    }

    // Both distinct versions must be kept in full, never folded just because signatures match.
    assert!(
        compressed
            .iter()
            .any(|m| m.content.as_str() == Some(changed_v1.as_str())),
        "changed file version 1 must be preserved verbatim"
    );
    assert!(
        compressed
            .iter()
            .any(|m| m.content.as_str() == Some(changed_v2.as_str())),
        "changed file version 2 must be preserved verbatim"
    );
}

#[test]
fn dedup_skips_byte_identical_overflow_archived_stubs() {
    // Regression test (real case session c0ad15e6, msg 471/472):
    // when a tool result's content **is itself already an overflow archive stub** (`[context-overflow-truncated]`
    // or `[[PRESERVED_TOOL_OVERFLOW_STUB_V1]]`), it is not a "full result", just a recall pointer to the
    // original text on disk. Byte-identical dedup registers the first seen in reverse order as canonical; when a copy is
    // byte-identical to canonical ⇒ canonical is likewise a truncated stub. Folding it into "reuse the canonical full
    // result" would be a false claim, steering the model into a back-reference chain where the next hop is still a stub and the original text is never reached.
    // Correct behavior: skip folding; each stub keeps its own file_path recall pointer.
    let archived_stub = format!(
        "[context-overflow-truncated] full original archived at: \
         /sess.assets/tool-overflow-compressed/20260803T041154Z-read_file-abc.txt\n\
         head+tail preview:\n{}",
        "X".repeat(4_000)
    );

    let mut messages = vec![Message {
        role: "system".to_string(),
        content: Value::String("system prompt".to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }];

    // 5 reads of the same file whose results are all already **the same truncated stub** (simulating repeated replays after overflow).
    for i in 0..5 {
        let (a, t) = read_file_call_pair(
            &format!("call_stub_{i}"),
            "/repo/task_tools.rs",
            &archived_stub,
        );
        messages.push(a);
        messages.push(t);
    }

    // Append near-end tool messages with unique content to push the reads above out of the KEEP_RECENT protected window,
    // making sure dedup actually applies to them.
    for i in 0..8 {
        let (a, t) = read_file_call_pair(
            &format!("call_pad_{i}"),
            &format!("/repo/pad_{i}.py"),
            &format!("padding-{i}"),
        );
        messages.push(a);
        messages.push(t);
    }

    let compressed = compress_messages_for_context(messages, 200_000, 256, 400, None, None);

    // Key assertion: no byte-identical dedup stub of the form "reuse the canonical full result" may ever
    // be produced — that would falsely present an already-truncated stub as reusable full text.
    let lying_dedup_stubs = compressed
        .iter()
        .filter_map(|m| m.content.as_str())
        .filter(|s| s.contains("byte-identical") && s.contains("No need to re-read"))
        .count();
    assert_eq!(
        lying_dedup_stubs, 0,
        "overflow-archived stubs must not be dedup-collapsed into a 'reuse canonical full result' claim"
    );

    // All 5 truncated stubs must be preserved verbatim (each with its file_path recall pointer), not one missing.
    let preserved_archive_stubs = compressed
        .iter()
        .filter_map(|m| m.content.as_str())
        .filter(|s| s.trim_start().starts_with("[context-overflow-truncated]"))
        .count();
    assert_eq!(
        preserved_archive_stubs, 5,
        "all five overflow-archived stubs must be preserved so each keeps its own recall pointer"
    );
}
