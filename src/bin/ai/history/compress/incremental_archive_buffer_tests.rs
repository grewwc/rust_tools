//! Byte-for-byte parity with the original per-message JSONL serialization.

use super::*;

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "incremental-source-buffer-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn incremental_memory_archive_buffer_matches_legacy_jsonl() {
    let dir = TestDir::new();
    let mut input: Vec<Message> = [
        serde_json::json!({
            "role": "user",
            "content": "alpha\nβ\r\t\\\"\u{0000}",
            "reasoning_content": "runtime-origin:synthetic-user:v1",
        }),
        serde_json::json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "inspect β"}],
            "tool_calls": [{
                "id": "read-1", "type": "function", "function": {
                    "name": "read_file",
                    "arguments": "{\"path\":\"β.rs\",\"line\":1}",
                }
            }],
            "reasoning_content": "think\nagain",
        }),
        serde_json::json!({
            "role": "tool", "content": {"z": [null, true, 1], "a": "output"},
            "tool_call_id": "read-1",
        }),
        serde_json::json!({"role": "assistant", "content": null, "tool_calls": []}),
        serde_json::json!({"role": "internal_note", "content": "archive_file_path: prior.jsonl"}),
    ]
    .into_iter()
    .map(|value| serde_json::from_value(value).unwrap())
    .collect();
    input.push(input[0].clone());

    for source in input[..input.len() - 1]
        .iter()
        .map(std::slice::from_ref)
        .chain(std::iter::once(input.as_slice()))
    {
        // Preserve the previous implementation as the independent byte oracle.
        let expected: String = source
            .iter()
            .map(|message| format!("{}\n", serde_json::to_string(message).unwrap()))
            .collect();
        let plan = plan_incremental_summary(
            source,
            "Current work:\n- preserve all source fields",
            4_000,
            Some(&dir.0),
        )
        .unwrap();
        assert_eq!(plan.archive.content(), expected);
        let text = value_to_string(&plan.message().content);
        let record: Value = serde_json::from_str(
            text.strip_prefix(INCREMENTAL_SUMMARY_PREFIX)
                .unwrap()
                .trim(),
        )
        .unwrap();
        let digest = content_sha256_hex(expected.as_bytes());
        assert_eq!(
            record["source"],
            serde_json::json!({
                "archive_file_path": stub_archive_display_path(&plan.archive_path),
                "sha256": digest,
                "source_model": Value::Null,
                "start_line": 1,
                "end_line": source.len(),
                "encoding": "one_raw_message_json_per_line",
            })
        );
        assert_eq!(record["schema"], 1);
        assert_eq!(record["provenance"], "assistant_derived_unverified");
        assert!(!plan.archive_path.exists(), "planning must remain pure");
        assert!(plan.commit());
        let actual = std::fs::read_to_string(&plan.archive_path).unwrap();
        assert_eq!(actual, expected);
        let restored: Vec<Message> = actual
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(restored, source);
        assert!(plan.commit(), "exact sources must remain reusable");
        assert_eq!(
            std::fs::read_to_string(&plan.archive_path).unwrap(),
            expected
        );
    }
}
