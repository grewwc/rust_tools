//! Byte-for-byte parity with the unchanged single-field archive path.

use super::*;
use crate::ai::types::{FunctionCall, ToolCall};

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("ai-field-archive-batch-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const HEADER: &str = "# Overflow History Archive\n\nThe content below was moved out of the context window; use the read_file tool to read this file back when earlier context is needed.\n\n---\n\n";
const SEPARATOR: &str = "\n---\n\n";
const RAW_MESSAGE: &str = r#"{
  "role": "assistant",
  "content": "alpha\nβ",
  "tool_calls": [
    {
      "id": "call-a",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"path\":\"alpha.rs\"}"
      }
    },
    {
      "id": "call-b",
      "type": "function",
      "function": {
        "name": "run",
        "arguments": "{\"command\":\"printf z\"}"
      }
    }
  ],
  "reasoning_content": "think \"twice\""
}"#;
const DIGESTS: [&str; 4] = [
    "9fa962a4a71347623a811b658834a195bb6e696fa81e3d2d032d02d8d29aabfc",
    "f71db6af3d66ad1e04e1e67cd4cedaf07058ea1c892eed51786b32b2ad89a00b",
    "b35d389b696572887d4636e4b99120f851c7ebcff173f9a57a3049661e7b3369",
    "f692238215cf594ad6e4436bba990144695f546973779756dfed2368f87e8dba",
];
const FIELDS: [MutableMessageField; 4] = [
    MutableMessageField::Content,
    MutableMessageField::Reasoning,
    MutableMessageField::ToolArguments(0),
    MutableMessageField::ToolArguments(1),
];

fn fixture() -> Message {
    Message {
        role: "assistant".to_string(),
        content: Value::String("alpha\nβ".to_string()),
        tool_calls: Some(vec![
            ToolCall {
                id: "call-a".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "read_file".to_string(),
                    arguments: r#"{"path":"alpha.rs"}"#.to_string(),
                },
            },
            ToolCall {
                id: "call-b".to_string(),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: "run".to_string(),
                    arguments: r#"{"command":"printf z"}"#.to_string(),
                },
            },
        ]),
        tool_call_id: None,
        reasoning_content: Some("think \"twice\"".to_string()),
    }
}

fn expected_payload(label: &str, original: &str, raw_message: &str) -> String {
    format!(
        "## Truncated field original text\n\n### Field original text\n\n- role: assistant\n- field: {label}\n\nBegin original text\n{original}\nEnd original text\n\nraw_message_json:\n```json\n{raw_message}\n```\n\n"
    )
}

fn payloads() -> [String; 4] {
    [
        expected_payload("content", "alpha\nβ", RAW_MESSAGE),
        expected_payload("reasoning_content", "think \"twice\"", RAW_MESSAGE),
        expected_payload(
            "tool_calls[0].function.arguments",
            r#"{"path":"alpha.rs"}"#,
            RAW_MESSAGE,
        ),
        expected_payload(
            "tool_calls[1].function.arguments",
            r#"{"command":"printf z"}"#,
            RAW_MESSAGE,
        ),
    ]
}

fn expected_body(prior: Option<&str>, payloads: &[String]) -> Vec<u8> {
    let mut body = prior.unwrap_or(HEADER).to_string();
    for (index, payload) in payloads.iter().enumerate() {
        if prior.is_some() || index > 0 {
            body.push_str(SEPARATOR);
        }
        body.push_str(payload);
    }
    body.into_bytes()
}

fn expected_index(prior: &[u8], digests: &[&str]) -> Vec<u8> {
    let mut index = prior.to_vec();
    for digest in digests {
        index.extend_from_slice(digest.as_bytes());
        index.push(b'\n');
    }
    index
}

fn body_path(dir: &Path) -> PathBuf {
    dir.join(OVERFLOW_HISTORY_FILENAME)
}

fn index_path(dir: &Path) -> PathBuf {
    dir.join(format!("{OVERFLOW_HISTORY_FILENAME}.fingerprints"))
}

fn snapshot(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().to_str().unwrap().to_string(),
                std::fs::read(entry.path()).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn expected_files(body: Vec<u8>, index: Vec<u8>) -> Vec<(String, Vec<u8>)> {
    vec![
        (OVERFLOW_HISTORY_FILENAME.to_string(), body),
        (format!("{OVERFLOW_HISTORY_FILENAME}.fingerprints"), index),
    ]
}

fn sequential_archive(dir: &Path, fields: &[MutableMessageField]) {
    for field in fields {
        assert_eq!(
            archive_truncated_field_to_overflow(&fixture(), *field, Some(dir)),
            Some(body_path(dir).to_string_lossy().into_owned())
        );
    }
}

fn batch_archive(
    batch: &mut TruncatedFieldArchiveBatch,
    dir: &Path,
    fields: &[MutableMessageField],
) {
    for field in fields {
        assert_eq!(
            batch.archive_field(&fixture(), *field),
            Some(body_path(dir).to_string_lossy().into_owned())
        );
    }
}

#[test]
fn field_archive_batch_matches_sequential_bytes_and_index() {
    for seeded in [false, true] {
        let legacy = TestDir::new();
        let batched = TestDir::new();
        let prior_body = seeded.then_some("# Earlier archive\n");
        let prior_index = if seeded {
            b"\n legacy-fingerprint \n".as_slice()
        } else {
            b"".as_slice()
        };
        if let Some(body) = prior_body {
            for dir in [&legacy.0, &batched.0] {
                std::fs::write(body_path(dir), body).unwrap();
                std::fs::write(index_path(dir), prior_index).unwrap();
            }
        }
        sequential_archive(&legacy.0, &FIELDS);
        let mut batch = TruncatedFieldArchiveBatch::new(Some(&batched.0));
        batch_archive(&mut batch, &batched.0, &FIELDS);
        let expected = expected_files(
            expected_body(prior_body, &payloads()),
            expected_index(prior_index, &DIGESTS),
        );
        assert_eq!(snapshot(&legacy.0), expected);
        assert_eq!(snapshot(&batched.0), expected);
        assert_eq!(batch.fingerprint_index_loads(), 1);
    }
}

#[test]
fn field_archive_batch_deduplicates_repeated_payloads() {
    let legacy = TestDir::new();
    let batched = TestDir::new();
    let fields = [
        FIELDS[0], FIELDS[0], FIELDS[1], FIELDS[0], FIELDS[2], FIELDS[3], FIELDS[0],
    ];
    sequential_archive(&legacy.0, &fields);
    let mut batch = TruncatedFieldArchiveBatch::new(Some(&batched.0));
    batch_archive(&mut batch, &batched.0, &fields);
    let expected = expected_files(
        expected_body(None, &payloads()),
        expected_index(b"", &DIGESTS),
    );
    assert_eq!(snapshot(&legacy.0), expected);
    assert_eq!(snapshot(&batched.0), expected);
    assert_eq!(batch.fingerprint_index_loads(), 1);

    let mut replay = TruncatedFieldArchiveBatch::new(Some(&batched.0));
    batch_archive(&mut replay, &batched.0, &fields);
    assert_eq!(snapshot(&batched.0), expected);
    assert_eq!(replay.fingerprint_index_loads(), 1);
}

fn budget_messages() -> Vec<Message> {
    ['A', 'B', 'C', 'D']
        .into_iter()
        .map(|letter| Message {
            role: "assistant".to_string(),
            content: Value::String(letter.to_string().repeat(1_800)),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        })
        .collect()
}

#[test]
fn field_archive_batch_matches_sequential_budget_truncation() {
    let dir = TestDir::new();
    let mut legacy = budget_messages();
    let protected = Default::default();
    assert!(truncate_mutable_messages_to_fit_with_archive(
        &mut legacy,
        1_000,
        Some(&dir.0),
        &protected,
        true,
        None,
    ));
    let payloads = ['A', 'B', 'C', 'D'].map(|letter| {
        let original = letter.to_string().repeat(1_800);
        let raw = format!("{{\n  \"role\": \"assistant\",\n  \"content\": \"{original}\"\n}}");
        expected_payload("content", &original, &raw)
    });
    let digests = payloads
        .iter()
        .map(|payload| content_sha256_hex(payload.as_bytes()))
        .collect::<Vec<_>>();
    let expected = expected_files(
        expected_body(None, &payloads),
        expected_index(b"", &digests.iter().map(String::as_str).collect::<Vec<_>>()),
    );
    assert_eq!(snapshot(&dir.0), expected);
    std::fs::remove_file(body_path(&dir.0)).unwrap();
    std::fs::remove_file(index_path(&dir.0)).unwrap();

    let mut batched = budget_messages();
    let mut batch = TruncatedFieldArchiveBatch::new(Some(&dir.0));
    assert!(truncate_mutable_messages_to_fit_with_archive(
        &mut batched,
        1_000,
        Some(&dir.0),
        &protected,
        true,
        Some(&mut batch),
    ));
    assert_eq!(batched, legacy);
    assert_eq!(snapshot(&dir.0), expected);
    assert_eq!(batch.fingerprint_index_loads(), 1);
}

#[test]
fn field_archive_batch_keeps_best_effort_preview_on_body_failure() {
    let dir = TestDir::new();
    std::fs::create_dir(body_path(&dir.0)).unwrap();
    let protected = Default::default();
    let mut legacy = budget_messages();
    let legacy_result = truncate_mutable_messages_to_fit_with_archive(
        &mut legacy,
        1_000,
        Some(&dir.0),
        &protected,
        true,
        None,
    );
    let mut batched = budget_messages();
    let mut batch = TruncatedFieldArchiveBatch::new(Some(&dir.0));
    let batch_result = truncate_mutable_messages_to_fit_with_archive(
        &mut batched,
        1_000,
        Some(&dir.0),
        &protected,
        true,
        Some(&mut batch),
    );
    assert!(legacy_result);
    assert_eq!(batch_result, legacy_result);
    assert_eq!(batched, legacy);
    assert_eq!(batched.len(), 4);
    for message in batched {
        assert_eq!(
            message.content,
            Value::String(
                "[context-overflow-truncated] full original was not archived; inline preview omitted to meet context budget."
                    .to_string()
            )
        );
    }
    assert!(body_path(&dir.0).is_dir());
    assert!(!index_path(&dir.0).exists());

    // A gentler cut keeps an actual inline preview after the same IO failure.
    let mut legacy_preview = budget_messages().remove(0);
    let mut batch_preview = legacy_preview.clone();
    for (message, archive) in [
        (&mut legacy_preview, None),
        (&mut batch_preview, Some(&mut batch)),
    ] {
        assert_eq!(
            truncate_mutable_field_with_policy(
                message,
                MutableMessageField::Content,
                600,
                Some(&dir.0),
                FieldArchivePolicy::BestEffort,
                false,
                archive,
            ),
            FieldTruncateOutcome::Truncated
        );
    }
    assert_eq!(batch_preview, legacy_preview);
    let preview = batch_preview.content.as_str().unwrap();
    assert!(preview.starts_with("[context-overflow-truncated] head+tail preview:\nAAAA"));
    assert!(preview.ends_with("AAAA"));
    assert!(!index_path(&dir.0).exists());
}

#[test]
fn field_archive_batch_preserves_unwritable_index_fallback() {
    let legacy = TestDir::new();
    let batched = TestDir::new();
    for dir in [&legacy.0, &batched.0] {
        std::fs::create_dir(index_path(dir)).unwrap();
    }
    let fields = [FIELDS[0], FIELDS[0], FIELDS[1], FIELDS[0]];
    sequential_archive(&legacy.0, &fields);
    let mut batch = TruncatedFieldArchiveBatch::new(Some(&batched.0));
    batch_archive(&mut batch, &batched.0, &fields);
    let payloads = payloads();
    let expected = expected_body(
        None,
        &[
            payloads[0].clone(),
            payloads[0].clone(),
            payloads[1].clone(),
            payloads[0].clone(),
        ],
    );
    assert_eq!(std::fs::read(body_path(&legacy.0)).unwrap(), expected);
    assert_eq!(std::fs::read(body_path(&batched.0)).unwrap(), expected);
    assert!(index_path(&legacy.0).is_dir());
    assert!(index_path(&batched.0).is_dir());
    assert_eq!(batch.fingerprint_index_loads(), fields.len());
}

#[test]
fn field_archive_batch_preserves_invalid_utf8_index_fallback() {
    let legacy = TestDir::new();
    let batched = TestDir::new();
    for dir in [&legacy.0, &batched.0] {
        std::fs::write(index_path(dir), [0xff, b'\n']).unwrap();
    }
    let fields = [FIELDS[0], FIELDS[0], FIELDS[1]];
    sequential_archive(&legacy.0, &fields);
    let mut batch = TruncatedFieldArchiveBatch::new(Some(&batched.0));
    batch_archive(&mut batch, &batched.0, &fields);
    let payloads = payloads();
    let expected = expected_files(
        expected_body(
            None,
            &[
                payloads[0].clone(),
                payloads[0].clone(),
                payloads[1].clone(),
            ],
        ),
        expected_index(&[0xff, b'\n'], &[DIGESTS[0], DIGESTS[0], DIGESTS[1]]),
    );
    assert_eq!(snapshot(&legacy.0), expected);
    assert_eq!(snapshot(&batched.0), expected);
    assert_eq!(batch.fingerprint_index_loads(), fields.len());
}

#[test]
fn field_archive_batch_preserves_unterminated_index_entry() {
    let legacy = TestDir::new();
    let batched = TestDir::new();
    for dir in [&legacy.0, &batched.0] {
        std::fs::write(body_path(dir), "# Earlier archive\n").unwrap();
        std::fs::write(index_path(dir), DIGESTS[3]).unwrap();
    }
    let fields = [FIELDS[0], FIELDS[3], FIELDS[0]];
    sequential_archive(&legacy.0, &fields);
    let mut batch = TruncatedFieldArchiveBatch::new(Some(&batched.0));
    batch_archive(&mut batch, &batched.0, &fields);
    let payloads = payloads();
    let expected = expected_files(
        expected_body(
            Some("# Earlier archive\n"),
            &[
                payloads[0].clone(),
                payloads[3].clone(),
                payloads[0].clone(),
            ],
        ),
        expected_index(DIGESTS[3].as_bytes(), &[DIGESTS[0], DIGESTS[3], DIGESTS[0]]),
    );
    assert_eq!(snapshot(&legacy.0), expected);
    assert_eq!(snapshot(&batched.0), expected);
    assert_eq!(batch.fingerprint_index_loads(), 2);
}

#[test]
fn field_archive_batch_heals_missing_or_empty_body() {
    for empty in [false, true] {
        let legacy = TestDir::new();
        let batched = TestDir::new();
        sequential_archive(&legacy.0, &FIELDS[..1]);
        let mut batch = TruncatedFieldArchiveBatch::new(Some(&batched.0));
        batch_archive(&mut batch, &batched.0, &FIELDS[..1]);
        for dir in [&legacy.0, &batched.0] {
            if empty {
                std::fs::write(body_path(dir), b"").unwrap();
            } else {
                std::fs::remove_file(body_path(dir)).unwrap();
            }
        }
        sequential_archive(&legacy.0, &FIELDS[..1]);
        batch_archive(&mut batch, &batched.0, &FIELDS[..1]);
        let expected = expected_files(
            expected_body(empty.then_some(""), &payloads()[..1]),
            expected_index(b"", &DIGESTS[..1]),
        );
        assert_eq!(snapshot(&legacy.0), expected);
        assert_eq!(snapshot(&batched.0), expected);
        assert_eq!(batch.fingerprint_index_loads(), 1);
    }
}
