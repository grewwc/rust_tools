use super::*;

fn overflow_test_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ai-overflow-sink-{tag}-{}", uuid::Uuid::new_v4()))
}

fn user_message(content: &str) -> Message {
    Message {
        role: "user".to_string(),
        content: Value::String(content.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

const PAYLOAD_MARK: &str = "overflow-sink-staleness-marker-7f3d";

#[test]
fn flush_rearchives_when_archive_body_missing_but_index_stale() {
    let dir = overflow_test_dir("stale-index");
    std::fs::create_dir_all(&dir).unwrap();

    let mut sink = OverflowSink::new(&dir);
    sink.push_messages(&[user_message(PAYLOAD_MARK)]);
    assert!(sink.flush(), "first flush must append and index");
    let body_path = sink.file_path().to_path_buf();
    let sidecar_path = PathBuf::from(format!("{}.fingerprints", body_path.to_string_lossy()));
    assert!(body_path.exists(), "archive body must exist after flush");
    assert!(sidecar_path.exists(), "index must exist after flush");
    // One append embeds the marker in both the formatted section and the
    // raw_message_json block, so assert against this exact baseline instead
    // of counting occurrences.
    let original_body = std::fs::read_to_string(&body_path).unwrap();

    // Simulate external cleanup: drop the archive body but keep the index.
    std::fs::remove_file(&body_path).unwrap();

    // Same instance replays its buffer: the stale hit must heal (clear set,
    // delete sidecar) and re-append instead of reporting a false success.
    assert!(sink.flush(), "stale-index flush must re-archive");
    let healed = std::fs::read_to_string(&body_path).unwrap();
    assert_eq!(healed, original_body, "re-archive must be byte-identical");
    assert!(
        std::fs::read_to_string(&sidecar_path)
            .unwrap()
            .lines()
            .any(|line| !line.trim().is_empty()),
        "sidecar must be rebuilt with at least one digest"
    );

    // A fresh sink trusts the rebuilt index: identical payload must dedupe.
    let mut sink2 = OverflowSink::new(&dir);
    sink2.push_messages(&[user_message(PAYLOAD_MARK)]);
    assert!(sink2.flush(), "dedup path must still report success");
    let final_body = std::fs::read_to_string(&body_path).unwrap();
    assert_eq!(final_body, original_body);
}

#[test]
fn flush_rearchives_when_archive_truncated_to_empty_but_index_stale() {
    // Deletion is not the only external hazard: an in-place truncate leaves the
    // archive file present but empty while the sidecar survives. A guard that
    // only checks existence would still trust the stale index and skip the
    // write, and callers drop the payload from the projection on that false
    // success — silently losing evidence readers were promised at file_path().
    let dir = overflow_test_dir("empty-body");
    std::fs::create_dir_all(&dir).unwrap();

    let mut sink = OverflowSink::new(&dir);
    sink.push_messages(&[user_message(PAYLOAD_MARK)]);
    assert!(sink.flush(), "first flush must append and index");
    let body_path = sink.file_path().to_path_buf();

    // Simulate external truncation: file still exists but is now empty.
    std::fs::write(&body_path, b"").unwrap();
    assert!(body_path.exists(), "truncated archive still exists");
    assert_eq!(std::fs::metadata(&body_path).unwrap().len(), 0);

    // A fresh compression pass reprojects the same payload (each pass builds a
    // new OverflowSink), so the heal must not depend on replaying a live buffer.
    let mut sink2 = OverflowSink::new(&dir);
    sink2.push_messages(&[user_message(PAYLOAD_MARK)]);
    assert!(
        sink2.flush(),
        "stale-index flush over an empty body must re-archive"
    );
    let healed = std::fs::read_to_string(&body_path).unwrap();
    assert!(
        healed.contains(PAYLOAD_MARK),
        "payload must be re-archived, not skipped on a stale index over an empty file"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
