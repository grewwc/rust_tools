//! Byte-level differential coverage for the conservative prefix reuse path.

use std::sync::Mutex;

use super::prefix_reuse::{PrefixReuseCache, PrefixReuseKey};
use super::*;
use crate::ai::types::{FunctionCall, ToolCall};

const MAX_CHARS: usize = 100_000;
const KEEP_LAST: usize = 4;
const SUMMARY_CHARS: usize = 4_000;

fn message(role: &str, text: &str) -> Message {
    Message {
        role: role.to_owned(),
        content: Value::String(text.to_owned()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn assistant_call(id: &str) -> Message {
    let mut message = message("assistant", "Inspect the implementation.");
    message.tool_calls = Some(vec![ToolCall {
        id: id.to_owned(),
        tool_type: "function".to_owned(),
        function: FunctionCall {
            name: "read_file".to_owned(),
            arguments: r#"{"file_path":"sample.rs"}"#.to_owned(),
        },
    }]);
    message
}

fn tool_result(id: &str) -> Message {
    let mut message = message("tool", "1\tfn example() {}\n2\t// retained evidence");
    message.tool_call_id = Some(id.to_owned());
    message
}

fn raw_prefix() -> Vec<Message> {
    let summary = message(
        ROLE_INTERNAL_NOTE,
        "[mid-turn-summary] Prior implementation decision.",
    );
    let archive = message(
        ROLE_INTERNAL_NOTE,
        &format!("{ARCHIVE_NOTE_PREFIX}\n{ARCHIVE_FILE_PATH_PREFIX}/tmp/prior-history.md\n"),
    );
    let mut call = assistant_call("prior-read");
    call.reasoning_content = Some(encode_reasoning_replay_state(
        "fixture-model",
        "Exact replay 文本.",
    ));
    let mut result = tool_result("prior-read");
    result.content = Value::String("Retained evidence 文本.\n".repeat(400));
    vec![
        message("system", "Stable policy."),
        summary.clone(),
        archive.clone(),
        summary,
        archive,
        message("user", "Earlier request."),
        call,
        result,
        message("assistant", "Earlier task completed."),
    ]
}

fn cached_compress(
    messages: Vec<Message>,
    max_chars: usize,
    keep_last: usize,
    summary_chars: usize,
    overflow_dir: Option<&Path>,
    cache: &Mutex<PrefixReuseCache>,
) -> ContextCompressionOutcome {
    compress_messages_with_prefix_cache(
        messages,
        max_chars,
        keep_last,
        summary_chars,
        overflow_dir.map(Path::to_path_buf),
        None,
        cache,
    )
}

fn assert_same_projection(actual: &ContextCompressionOutcome, full: &ContextCompressionOutcome) {
    assert_eq!(
        serde_json::to_vec(&actual.messages).unwrap(),
        serde_json::to_vec(&full.messages).unwrap(),
        "model-visible projection bytes differ"
    );
    assert_eq!(actual.status, full.status);
    assert_eq!(actual.before_chars, full.before_chars);
    assert_eq!(actual.after_chars, full.after_chars);
    assert_eq!(actual.max_chars, full.max_chars);
}

#[test]
fn prefix_reuse_projection_matches_full_pipeline() {
    let cache = Mutex::new(PrefixReuseCache::default());
    let raw = raw_prefix();
    let raw_chars = messages_total_chars(&raw);
    let prefix = cached_compress(raw, MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, &cache);
    assert_eq!(prefix.status, ContextCompressionStatus::Complete);
    assert!(
        prefix.after_chars < raw_chars,
        "seed must contain an actually compressed prefix"
    );
    let key = PrefixReuseKey::new(MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, None);
    let mut probe = PrefixReuseCache::from_outcome(key, &prefix);

    let mut input = prefix.messages;
    input.push(message(
        "user",
        "Continue with the next task.\n逐字节保留 UTF-8 内容。",
    ));
    let full = compress_messages_for_context_full(
        input.clone(),
        MAX_CHARS,
        KEEP_LAST,
        SUMMARY_CHARS,
        None,
        None,
    );
    let before = messages_total_chars(&input);
    let pointers = input
        .iter()
        .map(|message| message.content.as_str().unwrap().as_ptr())
        .collect::<Vec<_>>();
    // A separate local certificate proves the actual call below must be a hit.
    assert_eq!(
        probe.try_reuse(
            &input,
            &PrefixReuseKey::new(MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, None)
        ),
        Some(before)
    );
    let actual = cached_compress(input, MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, &cache);
    assert_same_projection(&actual, &full);
    assert_eq!(actual.before_chars, before);
    assert_eq!(actual.after_chars, before);
    for (message, pointer) in actual.messages.iter().zip(pointers) {
        assert_eq!(message.content.as_str().unwrap().as_ptr(), pointer);
    }

    // A hit advances the cache to the newest output without rerunning admission.
    let next = actual.messages;
    let key = PrefixReuseKey::new(MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, None);
    assert!(
        cache
            .lock()
            .unwrap()
            .try_reuse(&next[..next.len() - 1], &key)
            .is_none()
    );
    let mut next = next;
    next.push(message("assistant", "The follow-up is ready."));
    let full = compress_messages_for_context_full(
        next.clone(),
        MAX_CHARS,
        KEEP_LAST,
        SUMMARY_CHARS,
        None,
        None,
    );
    let actual = cached_compress(next, MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, &cache);
    assert_same_projection(&actual, &full);
}

#[test]
fn prefix_reuse_projection_matches_full_pipeline_without_keep_last() {
    let cache = Mutex::new(PrefixReuseCache::default());
    let prefix = cached_compress(raw_prefix(), MAX_CHARS, 0, SUMMARY_CHARS, None, &cache);
    let key = PrefixReuseKey::new(MAX_CHARS, 0, SUMMARY_CHARS, None, None);
    let mut probe = PrefixReuseCache::from_outcome(key, &prefix);
    let mut input = prefix.messages;
    input.push(message("user", "A new request with keep_last disabled."));
    assert!(
        probe
            .try_reuse(
                &input,
                &PrefixReuseKey::new(MAX_CHARS, 0, SUMMARY_CHARS, None, None)
            )
            .is_some()
    );
    let full =
        compress_messages_for_context_full(input.clone(), MAX_CHARS, 0, SUMMARY_CHARS, None, None);
    let actual = cached_compress(input, MAX_CHARS, 0, SUMMARY_CHARS, None, &cache);
    assert_same_projection(&actual, &full);
}

#[test]
fn prefix_reuse_projection_matches_full_pipeline_tail_fallbacks() {
    let cases = vec![
        ("tool result", vec![tool_result("new-result")]),
        ("tool call", vec![assistant_call("new-call")]),
        ("empty tool protocol", {
            let mut message = message("assistant", "No calls yet.");
            message.tool_calls = Some(Vec::new());
            vec![message]
        }),
        (
            "summary note",
            vec![message(
                ROLE_INTERNAL_NOTE,
                "[mid-turn-summary] A new summary.",
            )],
        ),
        (
            "archive note",
            vec![message(ROLE_INTERNAL_NOTE, ARCHIVE_NOTE_PREFIX)],
        ),
        (
            "self note",
            vec![message(ROLE_INTERNAL_NOTE, "self_note: a new memory")],
        ),
        (
            "quoted summary",
            vec![message("user", "[incremental-memory-v1] quoted note")],
        ),
        ("image", {
            let mut image = message("user", "");
            image.content = serde_json::json!([{"type":"image_url", "image_url":{"url":"data:image/png;base64,AA=="}}]);
            vec![image]
        }),
        (
            "keep_last boundary",
            (0..KEEP_LAST)
                .map(|index| message("user", &format!("Request {index}")))
                .collect(),
        ),
        ("over budget", vec![message("user", &"x".repeat(MAX_CHARS))]),
        (
            "trim boundary",
            (0..3)
                .map(|index| message("user", &format!("Request {index}")))
                .collect(),
        ),
    ];
    for (name, tail) in cases {
        let cache = Mutex::new(PrefixReuseCache::default());
        let prefix = cached_compress(
            raw_prefix(),
            MAX_CHARS,
            KEEP_LAST,
            SUMMARY_CHARS,
            None,
            &cache,
        );
        let mut input = prefix.messages;
        input.extend(tail);
        let key = PrefixReuseKey::new(MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, None);
        assert!(
            cache.lock().unwrap().try_reuse(&input, &key).is_none(),
            "unexpected hit: {name}"
        );
        let full = compress_messages_for_context_full(
            input.clone(),
            MAX_CHARS,
            KEEP_LAST,
            SUMMARY_CHARS,
            None,
            None,
        );
        let actual = cached_compress(input, MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, &cache);
        assert_same_projection(&actual, &full);
    }
}

#[test]
fn prefix_reuse_projection_matches_full_pipeline_guidance_and_failures() {
    let cache = Mutex::new(PrefixReuseCache::default());
    let guidance = message(
        ROLE_INTERNAL_NOTE,
        "[tool_first_use_guidance name=read_file]\nKeep exact evidence.",
    );
    let prefix = cached_compress(
        vec![message("user", "Current request."), guidance],
        MAX_CHARS,
        KEEP_LAST,
        SUMMARY_CHARS,
        None,
        &cache,
    );
    let mut input = prefix.messages;
    input.push(message("user", "New request expires prior guidance."));
    let key = PrefixReuseKey::new(MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, None);
    assert!(cache.lock().unwrap().try_reuse(&input, &key).is_none());
    let full = compress_messages_for_context_full(
        input.clone(),
        MAX_CHARS,
        KEEP_LAST,
        SUMMARY_CHARS,
        None,
        None,
    );
    assert_eq!(full.status, ContextCompressionStatus::Complete);
    let actual = cached_compress(input, MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, &cache);
    assert_same_projection(&actual, &full);

    // Disabled and archive/increment failures always use the original statuses.
    let dir = std::env::temp_dir().join(format!("ai-prefix-reuse-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let invalid_sink = dir.join("not-a-directory");
    std::fs::write(&invalid_sink, b"occupied").unwrap();
    let failures = [
        (0, KEEP_LAST, None, ContextCompressionStatus::Disabled),
        (
            MAX_CHARS,
            1,
            None,
            ContextCompressionStatus::MissingArchiveSink,
        ),
        (
            MAX_CHARS,
            1,
            Some(invalid_sink.as_path()),
            ContextCompressionStatus::ArchiveCommitFailed,
        ),
    ];
    for (max, keep, sink, status) in failures {
        let cache = Mutex::new(PrefixReuseCache::default());
        let prefix = cached_compress(
            vec![message("user", "Previous request.")],
            max,
            keep,
            SUMMARY_CHARS,
            sink,
            &cache,
        );
        let mut input = prefix.messages;
        input.push(message("user", "Next request."));
        let key = PrefixReuseKey::new(max, keep, SUMMARY_CHARS, sink, None);
        assert!(cache.lock().unwrap().try_reuse(&input, &key).is_none());
        let full = compress_messages_for_context_full(
            input.clone(),
            max,
            keep,
            SUMMARY_CHARS,
            sink.map(Path::to_path_buf),
            None,
        );
        assert_eq!(full.status, status);
        let actual = cached_compress(input, max, keep, SUMMARY_CHARS, sink, &cache);
        assert_same_projection(&actual, &full);
        assert!(
            cache
                .lock()
                .unwrap()
                .try_reuse(&actual.messages, &key)
                .is_none()
        );
    }

    let input = vec![
        message(
            ROLE_INTERNAL_NOTE,
            &format!("{INCREMENTAL_SUMMARY_PREFIX}\n{}", "a".repeat(4_100)),
        ),
        message(
            ROLE_INTERNAL_NOTE,
            &format!("{INCREMENTAL_SUMMARY_PREFIX}\n{}", "b".repeat(4_100)),
        ),
        message("user", "Retain the unarchived increments."),
    ];
    let cache = Mutex::new(PrefixReuseCache::default());
    let full = compress_messages_for_context_full(
        input.clone(),
        MAX_CHARS,
        KEEP_LAST,
        SUMMARY_CHARS,
        None,
        None,
    );
    assert_eq!(full.status, ContextCompressionStatus::MissingArchiveSink);
    let actual = cached_compress(input, MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, &cache);
    assert_same_projection(&actual, &full);
    assert!(
        cache
            .lock()
            .unwrap()
            .try_reuse(
                &actual.messages,
                &PrefixReuseKey::new(MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, None)
            )
            .is_none()
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn prefix_reuse_projection_matches_full_pipeline_key_and_admission_fallbacks() {
    let cache = Mutex::new(PrefixReuseCache::default());
    let prefix = cached_compress(
        raw_prefix(),
        MAX_CHARS,
        KEEP_LAST,
        SUMMARY_CHARS,
        None,
        &cache,
    );
    let different_keys = [
        PrefixReuseKey::new(MAX_CHARS - 1, KEEP_LAST, SUMMARY_CHARS, None, None),
        PrefixReuseKey::new(MAX_CHARS, KEEP_LAST - 1, SUMMARY_CHARS, None, None),
        PrefixReuseKey::new(MAX_CHARS, KEEP_LAST, SUMMARY_CHARS - 1, None, None),
        PrefixReuseKey::new(
            MAX_CHARS,
            KEEP_LAST,
            SUMMARY_CHARS,
            Some(Path::new("/tmp/another-session")),
            None,
        ),
        PrefixReuseKey::new(
            MAX_CHARS,
            KEEP_LAST,
            SUMMARY_CHARS,
            None,
            Some(Path::new("/tmp/another-cwd")),
        ),
    ];
    for key in different_keys {
        assert!(
            cache
                .lock()
                .unwrap()
                .try_reuse(&prefix.messages, &key)
                .is_none()
        );
    }
    let mut changed = prefix.messages.clone();
    changed[0].reasoning_content = Some("different metadata".to_owned());
    let key = PrefixReuseKey::new(MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, None);
    assert!(cache.lock().unwrap().try_reuse(&changed, &key).is_none());

    // The newest oversized increment can survive a Complete full pass. Its
    // independent window is still not certified as a no-op for reuse.
    let cache = Mutex::new(PrefixReuseCache::default());
    let input = vec![
        message(
            ROLE_INTERNAL_NOTE,
            &format!(
                "{INCREMENTAL_SUMMARY_PREFIX}\n{}",
                "a".repeat(MAX_INCREMENTAL_SUMMARY_INLINE_CHARS + 1)
            ),
        ),
        message("user", "Keep the newest increment."),
    ];
    let prefix = cached_compress(input, MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, &cache);
    assert_eq!(prefix.status, ContextCompressionStatus::Complete);
    assert!(
        cache
            .lock()
            .unwrap()
            .try_reuse(&prefix.messages, &key)
            .is_none()
    );

    let cache = Mutex::new(PrefixReuseCache::default());
    let input = vec![
        message(
            ROLE_INTERNAL_NOTE,
            &format!(
                "{COMPRESSED_TOOL_EVIDENCE_MARKER}\n{}",
                "a".repeat(MAX_COMPRESSED_TOOL_EVIDENCE_INLINE_CHARS + 1)
            ),
        ),
        message("user", "Keep the newest tool evidence."),
    ];
    let prefix = cached_compress(input, MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, &cache);
    assert_eq!(prefix.status, ContextCompressionStatus::Complete);
    assert!(
        cache
            .lock()
            .unwrap()
            .try_reuse(&prefix.messages, &key)
            .is_none()
    );

    // Crossing the existing char threshold can shrink protection even when no
    // new user arrives and keep_last remains satisfied.
    let cache = Mutex::new(PrefixReuseCache::default());
    let input = vec![
        message("user", "First."),
        message(
            "assistant",
            &"p".repeat(KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS - 100),
        ),
        message("user", "Second."),
        message("user", "Third."),
    ];
    let prefix = cached_compress(input, MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, &cache);
    let mut input = prefix.messages;
    input.push(message("assistant", &"q".repeat(200)));
    assert!(cache.lock().unwrap().try_reuse(&input, &key).is_none());
    let full = compress_messages_for_context_full(
        input.clone(),
        MAX_CHARS,
        KEEP_LAST,
        SUMMARY_CHARS,
        None,
        None,
    );
    let actual = cached_compress(input, MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, &cache);
    assert_same_projection(&actual, &full);
}
