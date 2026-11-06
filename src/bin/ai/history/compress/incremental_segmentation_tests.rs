//! End-to-end segmentation evidence through the real summary-input renderer.

use std::sync::Mutex;

use super::summary_input::build_summary_history_input;
use super::*;

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "incremental-segmentation-{}",
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

fn message(role: &str, text: impl Into<String>) -> Message {
    Message {
        role: role.to_string(),
        content: Value::String(text.into()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn side_note(text: &str, ts: u64) -> Message {
    crate::ai::driver::side_note::side_notes_to_messages(vec![
        crate::ai::driver::side_note::SideNote {
            from: "upstream".to_string(),
            content: text.to_string(),
            ts,
            target: None,
        },
    ])
    .remove(0)
}

#[derive(Default)]
struct Summarizer {
    calls: Mutex<Vec<Vec<Message>>>,
    failure: Option<(usize, Option<&'static str>)>,
    padding: usize,
}

impl HistoryMessageSummarizer for Summarizer {
    fn summarize_history_messages<'a>(
        &'a self,
        messages: &'a [Message],
        _max_chars: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send + 'a>> {
        Box::pin(async move {
            let mut calls = self.calls.lock().unwrap();
            let index = calls.len();
            calls.push(messages.to_vec());
            drop(calls);
            if let Some((failure_index, response)) = self.failure
                && index == failure_index
            {
                return response.map(str::to_string);
            }
            let opening: String = value_to_string(&messages[0].content)
                .chars()
                .take(120)
                .collect();
            Some(format!(
                "Main request:\n- CHUNK_{index}: {opening}{}",
                " detail".repeat(self.padding)
            ))
        })
    }
}

fn record(plan: &IncrementalSummaryPlan) -> Value {
    let text = value_to_string(&plan.message().content);
    serde_json::from_str(
        text.strip_prefix(INCREMENTAL_SUMMARY_PREFIX)
            .unwrap()
            .trim(),
    )
    .unwrap()
}

fn verify_archive(plan: &IncrementalSummaryPlan, original: &[Message]) {
    // The previous serializer is an independent byte oracle, including newlines
    // and fields not rendered into the auxiliary model request.
    let expected: String = original
        .iter()
        .map(|message| format!("{}\n", serde_json::to_string(message).unwrap()))
        .collect();
    assert_eq!(plan.archive.content(), expected);
    let record = record(plan);
    assert_eq!(record["source"]["start_line"], 1);
    assert_eq!(record["source"]["end_line"], original.len());
    assert_eq!(
        record["source"]["sha256"],
        content_sha256_hex(expected.as_bytes())
    );
    assert_eq!(
        record["source"]["archive_file_path"],
        stub_archive_display_path(&plan.archive_path)
    );
    assert!(
        !plan.archive_path.exists(),
        "the candidate must remain pure"
    );
    assert!(plan.commit());
    let actual = std::fs::read_to_string(&plan.archive_path).unwrap();
    assert_eq!(actual, expected);
    let restored: Vec<Message> = actual
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(restored, original);
}

fn verify_transcripts(calls: &[Vec<Message>]) {
    assert!(!calls.is_empty());
    for call in calls {
        let transcript = build_summary_history_input(call, SUMMARY_TRANSCRIPT_MAX_CHARS);
        assert!(!transcript.is_empty(), "a bounded segment must be accepted");
        assert!(transcript.chars().count() <= SUMMARY_TRANSCRIPT_MAX_CHARS);
        assert!(
            !transcript.contains(SUMMARY_MESSAGE_SEPARATOR),
            "a segment must not sample away chronological rows"
        );
        for note in call.iter().filter(|message| is_side_note_message(message)) {
            assert!(transcript.contains(note.content.as_str().unwrap()));
            assert!(transcript.contains(&format!(
                "### SIDE-NOTE\n\n{}",
                note.content.as_str().unwrap()
            )));
            assert!(is_runtime_synthetic_user_message(note));
            assert!(!crate::ai::history::is_user_input_message(note));
        }
    }
}

#[tokio::test]
async fn incremental_memory_segments_thousand_turns_with_exact_source_ranges() {
    let dir = TestDir::new();
    let old = plan_incremental_summary(
        &[message("user", "prior evidence")],
        "Main request:\n- PRIOR_SUMMARY",
        4_000,
        Some(&dir.0),
    )
    .unwrap();
    assert!(old.commit());
    let prior = old.message().clone();
    let early_note = side_note(
        &format!("EARLY_GUIDANCE {} FINAL_CONDITION", "中".repeat(7_000)),
        42,
    );
    let mut input = vec![
        message("system", "runtime policy"),
        prior.clone(),
        message("user", "ROUND_0000: remember the opening request"),
        early_note.clone(),
        message("assistant", "ANSWER_0000"),
    ];
    for turn in 1..1_000 {
        input.push(message(
            "user",
            format!("ROUND_{turn:04}: {}", "request detail ".repeat(10)),
        ));
        input.push(message(
            "assistant",
            format!("ANSWER_{turn:04}: {}", "response detail ".repeat(10)),
        ));
    }
    let original = input.clone();
    let summarizer = Summarizer::default();
    let plan = plan_incremental_summary_with_summarizer(&summarizer, &input, 8_000, Some(&dir.0))
        .await
        .unwrap();
    assert_eq!(input, original);
    assert_eq!(input[1], prior);
    let calls = summarizer.calls.lock().unwrap();
    assert!(calls.len() > 1);
    verify_transcripts(&calls);
    let supplied: Vec<Message> = calls.iter().flatten().cloned().collect();
    assert_eq!(supplied, summary_delta_messages(&original));
    assert!(supplied.contains(&early_note));
    assert!(!supplied.iter().any(is_summary_message));
    let record = record(&plan);
    assert_eq!(record["summary_input"]["segments"], calls.len());
    assert_eq!(record["summary_input"]["max_segment_chars"], 22_000);
    let entries = record["entries"].as_array().unwrap();
    assert_eq!(entries.len(), calls.len());
    for (index, (entry, call)) in entries.iter().zip(calls.iter()).enumerate() {
        let start = original
            .iter()
            .position(|message| message == &call[0])
            .unwrap()
            + 1;
        let end = original
            .iter()
            .position(|message| message == call.last().unwrap())
            .unwrap()
            + 1;
        assert_eq!(entry["segment"], index + 1);
        assert_eq!(entry["source_range"]["start_line"], start);
        assert_eq!(entry["source_range"]["end_line"], end);
        assert_eq!(entry["status"], "derived_unverified");
    }
    assert!(entries[0]["text"].as_str().unwrap().contains("ROUND_0000"));
    assert!(value_to_string(&plan.message().content).chars().count() <= 8_000);
    verify_archive(&plan, &original);
}

#[tokio::test]
async fn incremental_memory_segments_oversized_side_note_without_losing_unicode_body() {
    let dir = TestDir::new();
    let body = format!(
        "OPENING_CONSTRAINT\n{}\nMIDDLE_CONDITION\n{}\nFINAL_CONCLUSION",
        "界".repeat(30_000),
        "🦀".repeat(30_000)
    );
    let note = side_note(&body, 43);
    let header = note.content.as_str().unwrap().split_once("\n\n").unwrap().0;
    let input = vec![
        message("user", "opening request"),
        note.clone(),
        message("assistant", "continuation"),
    ];
    let summarizer = Summarizer::default();
    let plan = plan_incremental_summary_with_summarizer(&summarizer, &input, 8_000, Some(&dir.0))
        .await
        .unwrap();
    let calls = summarizer.calls.lock().unwrap();
    verify_transcripts(&calls);
    let fragments: Vec<&Message> = calls
        .iter()
        .flatten()
        .filter(|message| is_side_note_message(message))
        .collect();
    assert!(fragments.len() > 1);
    let mut restored_body = String::new();
    for fragment in fragments {
        let text = fragment.content.as_str().unwrap();
        assert!(text.starts_with(&format!("{header}\n\n")));
        assert!(text.ends_with("FINAL_CONCLUSION"));
        assert_eq!(fragment.role, note.role);
        assert_eq!(fragment.reasoning_content, note.reasoning_content);
        let (_, continuation) = text.split_once("\n\n").unwrap();
        let (notice, remainder) = continuation.split_once('\n').unwrap();
        assert!(notice.contains("source line 2"));
        let (piece, _) = remainder
            .rsplit_once("\n\n[Side-note ending repeated for context]\n")
            .unwrap();
        restored_body.push_str(piece);
    }
    assert_eq!(restored_body, body);
    let record = record(&plan);
    assert_eq!(record["summary_input"]["partial_side_note"], true);
    assert_eq!(
        record["reuse_guard"],
        "partial_draft_recover_source_before_using_judgments"
    );
    for entry in record["entries"].as_array().unwrap() {
        let index = entry["segment"].as_u64().unwrap() as usize - 1;
        if calls[index].iter().any(is_side_note_message) {
            assert!(entry["source_range"]["start_line"].as_u64().unwrap() <= 2);
            assert!(entry["source_range"]["end_line"].as_u64().unwrap() >= 2);
        }
    }
    verify_archive(&plan, &input);
}

#[tokio::test]
async fn incremental_memory_segment_failure_preserves_old_summary_and_source_span() {
    let dir = TestDir::new();
    let old = plan_incremental_summary(
        &[message("user", "old source")],
        "Main request:\n- OLD_SUMMARY",
        4_000,
        Some(&dir.0),
    )
    .unwrap();
    assert!(old.commit());
    let prior = old.message().clone();
    let input = vec![
        prior.clone(),
        side_note(&"a".repeat(18_000), 1),
        side_note(&"b".repeat(18_000), 2),
        side_note(&"c".repeat(18_000), 3),
    ];
    let original = input.clone();
    let old_bytes = std::fs::read(&old.archive_path).unwrap();
    // Cover transport failure, empty output, and a draft with no usable entries.
    for response in [None, Some(" \n\t"), Some("Main request:\n- none")] {
        let summarizer = Summarizer {
            failure: Some((1, response)),
            ..Default::default()
        };
        assert!(
            plan_incremental_summary_with_summarizer(&summarizer, &input, 4_000, Some(&dir.0))
                .await
                .is_none()
        );
        assert_eq!(summarizer.calls.lock().unwrap().len(), 2);
        assert_eq!(input, original);
        assert_eq!(input[0], prior);
        assert_eq!(std::fs::read(&old.archive_path).unwrap(), old_bytes);
        assert_eq!(
            std::fs::read_dir(dir.0.join(SOURCE_DIR)).unwrap().count(),
            1
        );
    }
}

#[tokio::test]
async fn incremental_memory_segment_output_cap_keeps_opening_and_archives_omitted_entries() {
    let dir = TestDir::new();
    let input: Vec<Message> = (0..18)
        .map(|index| {
            side_note(
                &format!("ORIGINAL_{index} {} END_{index}", "x".repeat(18_000)),
                index,
            )
        })
        .collect();
    let summarizer = Summarizer {
        padding: 60,
        ..Default::default()
    };
    let plan = plan_incremental_summary_with_summarizer(&summarizer, &input, 4_000, Some(&dir.0))
        .await
        .unwrap();
    let calls = summarizer.calls.lock().unwrap();
    assert_eq!(calls.len(), input.len());
    verify_transcripts(&calls);
    let record = record(&plan);
    let entries = record["entries"].as_array().unwrap();
    assert!(entries.len() < calls.len());
    assert_eq!(entries.first().unwrap()["segment"], 1);
    assert_eq!(entries.last().unwrap()["segment"], 18);
    assert_eq!(entries.first().unwrap()["source_range"]["start_line"], 1);
    assert_eq!(entries.last().unwrap()["source_range"]["end_line"], 18);
    assert_eq!(record["omitted_entries"], true);
    assert_eq!(
        record["reuse_guard"],
        "partial_draft_recover_source_before_using_judgments"
    );
    assert!(value_to_string(&plan.message().content).chars().count() <= 4_000);
    verify_archive(&plan, &input);
}

#[tokio::test]
async fn incremental_memory_segment_refuses_oversized_identity_before_model_calls() {
    let dir = TestDir::new();
    let author = "x".repeat(SUMMARY_TRANSCRIPT_MAX_CHARS);
    let note = crate::ai::history::runtime_synthetic_user_message(Value::String(format!(
        "[side-note from={author} ts=44] Live guidance (side-note) from upstream agent `{author}`\n\nKeep the conclusion"
    )));
    assert!(is_side_note_message(&note));
    let input = vec![note];
    let summarizer = Summarizer::default();
    assert!(
        plan_incremental_summary_with_summarizer(&summarizer, &input, 4_000, Some(&dir.0))
            .await
            .is_none()
    );
    assert!(summarizer.calls.lock().unwrap().is_empty());
    assert!(!dir.0.join(SOURCE_DIR).exists());
    assert!(is_side_note_message(&input[0]));
}

#[tokio::test]
async fn incremental_memory_segment_keeps_tool_metadata_and_archives_all_excerpts() {
    let dir = TestDir::new();
    let call: Message = serde_json::from_value(serde_json::json!({
        "role": "assistant", "content": "inspect tool evidence",
        "tool_calls": [{"id": "read-1", "type": "function", "function": {
            "name": "read_file", "arguments": "argument 界".repeat(20_000)
        }}],
        "reasoning_content": "original reasoning 界".repeat(5_000),
    }))
    .unwrap();
    let mut result = message("tool", "RAW_EVIDENCE 界".repeat(20_000));
    result.tool_call_id = Some("read-1".to_string());
    let input = vec![message("user", "inspect"), call, result];
    let summarizer = Summarizer::default();
    let plan = plan_incremental_summary_with_summarizer(&summarizer, &input, 8_000, Some(&dir.0))
        .await
        .unwrap();
    let calls = summarizer.calls.lock().unwrap();
    verify_transcripts(&calls);
    let transcript = build_summary_history_input(&calls[0], SUMMARY_TRANSCRIPT_MAX_CHARS);
    assert!(transcript.contains("id=read-1"));
    assert!(transcript.contains("name=read_file"));
    assert!(transcript.contains("Tool result for call id: read-1"));
    assert!(transcript.contains("[message excerpt truncated]"));
    verify_archive(&plan, &input);
}
