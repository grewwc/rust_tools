//! Request/snapshot integration for durable identity-projection certificates.

use serde_json::json;

use super::*;
use crate::ai::test_support::ENV_LOCK;
use crate::ai::types::{FunctionCall, ToolCall};

#[derive(Clone)]
struct RequestConfig {
    count: usize,
    max_chars: usize,
    keep_last: usize,
    summary_chars: usize,
    overflow_dir: Option<PathBuf>,
    cwd: Option<PathBuf>,
}

impl Default for RequestConfig {
    fn default() -> Self {
        Self {
            count: 0,
            max_chars: 100_000,
            keep_last: 4,
            summary_chars: 4_000,
            overflow_dir: None,
            cwd: None,
        }
    }
}

impl RequestConfig {
    fn keep_last(&self) -> usize {
        if self.count == 0 {
            self.keep_last
        } else {
            self.count
        }
    }

    fn policy(&self) -> Option<compress::PrefixCertificatePolicy> {
        compress::PrefixCertificatePolicy::new(
            self.max_chars,
            self.keep_last(),
            self.summary_chars,
            self.overflow_dir.as_deref(),
            self.cwd.as_deref(),
        )
    }

    fn fingerprint(&self) -> String {
        context_projection_fingerprint(
            self.max_chars,
            self.keep_last,
            self.summary_chars,
            self.overflow_dir.as_deref(),
        )
    }

    fn build(&self, path: &Path) -> (ContextCompressionOutcome, ContextSnapshotBasis) {
        let (outcome, basis) = build_context_history_with_snapshot_basis(
            self.count,
            path,
            self.max_chars,
            self.keep_last,
            self.summary_chars,
            self.overflow_dir.clone(),
            self.cwd.as_deref(),
        )
        .unwrap();
        (
            outcome,
            basis.expect("SQLite request must return a snapshot basis"),
        )
    }

    fn certificate(&self, messages: &[Message]) -> Option<String> {
        compress::prefix_certificate_json_from_cache(messages, &self.policy()?)
    }

    fn full(&self, mut messages: Vec<Message>) -> ContextCompressionOutcome {
        clear_b1_cache();
        let before_chars = messages_total_chars_pub(&messages);
        compress::cap_raw_tool_results_for_context(
            &mut messages,
            self.overflow_dir.as_deref(),
            self.cwd.as_deref(),
        );
        let mut outcome = if self.max_chars == 0 {
            let messages = if self.count >= messages.len() {
                messages
            } else {
                messages[messages.len() - self.count..].to_vec()
            };
            ContextCompressionOutcome::new(
                messages,
                before_chars,
                self.max_chars,
                ContextCompressionStatus::Disabled,
            )
        } else {
            // An empty B1 cache forces the unchanged full compression pipeline.
            compress_messages_for_context_with_outcome(
                messages,
                self.max_chars,
                self.keep_last(),
                self.summary_chars,
                self.overflow_dir.clone(),
                self.cwd.as_deref(),
            )
        };
        outcome.before_chars = before_chars;
        outcome
    }
}

struct Fixture {
    dir: PathBuf,
    path: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir =
            std::env::temp_dir().join(format!("b2-prefix-integration-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        Self {
            path: dir.join("history.sqlite"),
            dir,
        }
    }

    fn read(&self, config: &RequestConfig) -> sqlite::ContextHistory {
        sqlite::read_context_history_sqlite(&self.path, &config.fingerprint()).unwrap()
    }

    fn snapshot(&self, input: &[Message], config: &RequestConfig) -> Vec<Message> {
        append_history_messages(&self.path, input).unwrap();
        self.restart();
        let (outcome, basis) = config.build(&self.path);
        assert_eq!(outcome.status, ContextCompressionStatus::Complete);
        assert!(
            config.certificate(&outcome.messages).is_some(),
            "fixture must be admitted by B1"
        );
        assert!(
            persist_context_snapshot_from_basis(&self.path, &outcome.messages, &basis).unwrap()
        );
        let stored = self.read(config);
        let meta = stored.snapshot_meta.as_ref().unwrap();
        assert_eq!(stored.messages, outcome.messages);
        assert_eq!(meta.message_count, outcome.messages.len());
        assert!(compress::validate_certificate_json(
            meta.compression_state
                .as_deref()
                .expect("snapshot must carry B2 evidence"),
            &stored.messages,
            &[],
            &config.policy().unwrap(),
        ));
        outcome.messages
    }

    fn restart(&self) {
        clear_b1_cache();
        invalidate_context_history_cache_for(&self.path);
    }

    fn replace_state(&self, prefix: &[Message], config: &RequestConfig, state: Option<&str>) {
        let context = self.read(config);
        assert!(
            sqlite::write_context_snapshot_sqlite_if_stamp(
                &self.path,
                prefix,
                context.source_message_id,
                &config.fingerprint(),
                context.snapshot_stamp,
                state,
                Duration::from_secs(1),
            )
            .unwrap()
            .is_some()
        );
        invalidate_context_history_cache_for(&self.path);
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        invalidate_context_history_cache_for(&self.path);
        sqlite::close_cached_write_conn(&self.path);
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn message(role: &str, text: &str) -> Message {
    Message {
        role: role.to_owned(),
        content: json!(text),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn plain_prefix() -> Vec<Message> {
    vec![
        message("system", "Stable policy."),
        message("user", "Earlier request."),
        message("assistant", "Earlier task completed.\nUTF-8 文本."),
    ]
}

fn clear_b1_cache() {
    // Empty input cannot be admitted, so the public compressor clears CACHE
    // without exposing process-cache internals to the history integration tests.
    let empty =
        compress_messages_for_context_with_outcome(Vec::new(), 100_000, 4, 4_000, None, None);
    assert!(empty.messages.is_empty());
}

fn assert_equal_outcomes(actual: &ContextCompressionOutcome, full: &ContextCompressionOutcome) {
    assert_eq!(
        serde_json::to_vec(&actual.messages).unwrap(),
        serde_json::to_vec(&full.messages).unwrap(),
    );
    assert_eq!(actual.status, full.status);
    assert_eq!(actual.before_chars, full.before_chars);
    assert_eq!(actual.after_chars, full.after_chars);
    assert_eq!(actual.max_chars, full.max_chars);
}

fn assert_full_fallback(
    fixture: &Fixture,
    config: &RequestConfig,
) -> (ContextCompressionOutcome, ContextSnapshotBasis) {
    let context = fixture.read(config);
    assert!(
        try_reuse_context_snapshot_prefix(
            &context.messages,
            context.snapshot_meta.as_ref(),
            config.policy().as_ref(),
        )
        .is_none()
    );
    fixture.restart();
    let (actual, basis) = config.build(&fixture.path);
    let admitted_by_request = config.certificate(&actual.messages).is_some();
    let full = config.full(context.messages);
    assert_equal_outcomes(&actual, &full);
    if config.certificate(&full.messages).is_some() {
        assert!(
            admitted_by_request,
            "a certifiable full fallback must populate B1 after the simulated restart"
        );
    }
    (actual, basis)
}

#[test]
fn prefix_cert_request_reuses_snapshot_after_restart_byte_for_byte() {
    let _guard = ENV_LOCK.lock().unwrap();
    let fixture = Fixture::new();
    let config = RequestConfig::default();
    let prefix = fixture.snapshot(&plain_prefix(), &config);
    let tail = vec![
        message(
            "user",
            "Continue with the next task.\n逐字节保留 UTF-8 内容。",
        ),
        message("assistant", "The follow-up is ready."),
    ];
    append_history_messages(&fixture.path, &tail).unwrap();
    let context = fixture.read(&config);
    let meta = context.snapshot_meta.as_ref().unwrap();
    assert!(meta.guidance_unchanged);
    assert_eq!(meta.message_count, prefix.len());
    assert_eq!(&context.messages[meta.message_count..], tail);
    fixture.restart();
    assert!(config.certificate(&prefix).is_none());
    let (actual, basis) = config.build(&fixture.path);
    assert!(
        config.certificate(&actual.messages).is_none(),
        "B2 reuse must succeed independently of B1 admission"
    );
    assert_eq!(
        basis.snapshot_revision,
        context.snapshot_stamp.snapshot_revision
    );
    assert_eq!(
        actual.before_chars,
        messages_total_chars_pub(&context.messages)
    );
    let full = config.full(context.messages);
    assert!(config.certificate(&full.messages).is_some());
    assert_equal_outcomes(&actual, &full);
}

#[test]
fn prefix_cert_request_guidance_change_runs_full_pipeline() {
    let _guard = ENV_LOCK.lock().unwrap();
    let fixture = Fixture::new();
    let config = RequestConfig::default();
    let guidance = message(
        ROLE_INTERNAL_NOTE,
        "[tool_first_use_guidance name=read_file]\nRead serially.",
    );
    let mut input = plain_prefix();
    input.insert(2, guidance.clone());
    let prefix = fixture.snapshot(&input, &config);
    let mut unchanged_meta = fixture.read(&config).snapshot_meta.unwrap();
    unchanged_meta.guidance_unchanged = false;
    assert!(
        try_reuse_context_snapshot_prefix(
            &prefix,
            Some(&unchanged_meta),
            config.policy().as_ref(),
        )
        .is_none()
    );
    append_history_messages(&fixture.path, &[message("user", "New real user turn.")]).unwrap();
    let context = fixture.read(&config);
    assert!(!context.snapshot_meta.as_ref().unwrap().guidance_unchanged);
    assert!(!context.messages.contains(&guidance));
    let (actual, _) = assert_full_fallback(&fixture, &config);
    assert!(!actual.messages.contains(&guidance));
}

#[test]
fn prefix_cert_request_new_user_boundary_runs_full_pipeline() {
    let _guard = ENV_LOCK.lock().unwrap();
    let fixture = Fixture::new();
    let config = RequestConfig::default();
    let mut input = vec![message("system", "Stable policy.")];
    for turn in 1..=3 {
        input.push(message("user", &format!("Request {turn}.")));
        input.push(message("assistant", &format!("Answer {turn}.")));
    }
    let prefix = fixture.snapshot(&input, &config);
    let certificate: compress::PrefixCertificate =
        serde_json::from_str(&config.certificate(&prefix).unwrap()).unwrap();
    assert_eq!(certificate.trim_start, 0);
    append_history_messages(&fixture.path, &[message("user", "Request 4.")]).unwrap();
    let context = fixture.read(&config);
    assert!(context.snapshot_meta.as_ref().unwrap().guidance_unchanged);
    assert_eq!(user_input_turn_count(&context.messages), config.keep_last);
    assert!(types::retained_turn_start(&context.messages, 3) > certificate.trim_start);
    let _ = assert_full_fallback(&fixture, &config);
}

#[test]
fn prefix_cert_request_policy_change_runs_full_pipeline() {
    let _guard = ENV_LOCK.lock().unwrap();
    let fixture = Fixture::new();
    let config = RequestConfig::default();
    fixture.snapshot(&plain_prefix(), &config);
    let mut changes = vec![config.clone(); 6];
    changes[0].max_chars += 1;
    changes[1].keep_last += 1;
    changes[2].summary_chars += 1;
    changes[3].overflow_dir = Some(fixture.dir.join("assets"));
    changes[4].count = 3;
    changes[5].cwd = Some(fixture.dir.clone());
    for (index, changed) in changes.iter().enumerate() {
        let context = fixture.read(changed);
        assert_eq!(
            context.snapshot_meta.is_some(),
            index >= 4,
            "count/cwd changes must be rejected by B2 even when the snapshot fingerprint matches"
        );
        let _ = assert_full_fallback(&fixture, changed);
    }

    // A request cached before a cwd change must not return its old B2 policy.
    fixture.restart();
    let (reused, _) = config.build(&fixture.path);
    assert!(config.certificate(&reused.messages).is_none());
    clear_b1_cache();
    let changed = &changes[5];
    let (actual, basis) = changed.build(&fixture.path);
    assert_eq!(basis.compression_policy, changed.policy());
    assert!(changed.certificate(&actual.messages).is_some());
    let full = changed.full(fixture.read(changed).messages);
    assert_equal_outcomes(&actual, &full);
}

#[test]
fn prefix_cert_request_corrupt_or_missing_certificate_runs_full_pipeline() {
    let _guard = ENV_LOCK.lock().unwrap();
    let fixture = Fixture::new();
    let config = RequestConfig::default();
    let prefix = fixture.snapshot(&plain_prefix(), &config);
    let mut changed_digest: serde_json::Value =
        serde_json::from_str(&config.certificate(&prefix).unwrap()).unwrap();
    changed_digest["prefix_sha256"] = json!("corrupt digest");
    let changed_digest = changed_digest.to_string();
    for state in [None, Some("{"), Some("null"), Some(changed_digest.as_str())] {
        fixture.replace_state(&prefix, &config, state);
        let _ = assert_full_fallback(&fixture, &config);
    }
}

#[test]
fn prefix_cert_snapshot_write_certifies_only_actual_rows() {
    let _guard = ENV_LOCK.lock().unwrap();
    let fixture = Fixture::new();
    let config = RequestConfig::default();
    let prefix = fixture.snapshot(&plain_prefix(), &config);
    let context = fixture.read(&config);
    let basis = ContextSnapshotBasis::from_context(&context, config.fingerprint(), config.policy());
    let mut actual_rows = prefix.clone();
    actual_rows.last_mut().unwrap().content = json!("Different rows selected for the snapshot.");
    assert!(persist_context_snapshot_from_basis(&fixture.path, &actual_rows, &basis).unwrap());
    let written = fixture.read(&config);
    assert_eq!(written.messages, actual_rows);
    assert!(
        written
            .snapshot_meta
            .as_ref()
            .unwrap()
            .compression_state
            .is_none()
    );
    assert!(
        !persist_context_snapshot_from_basis(&fixture.path, &prefix, &basis).unwrap(),
        "the same basis must not overwrite a newer physical snapshot"
    );
    let after_stale_write = fixture.read(&config);
    assert_eq!(after_stale_write.snapshot_stamp, written.snapshot_stamp);
    assert_eq!(after_stale_write.messages, actual_rows);
    let _ = assert_full_fallback(&fixture, &config);
}

#[test]
fn prefix_cert_snapshot_write_uses_rejected_policy_physical_stamp() {
    let _guard = ENV_LOCK.lock().unwrap();
    let fixture = Fixture::new();
    let config = RequestConfig::default();
    fixture.snapshot(&plain_prefix(), &config);
    let mut changed = config.clone();
    changed.max_chars += 1;
    let rejected = fixture.read(&changed);
    assert!(rejected.snapshot_meta.is_none());
    assert!(rejected.snapshot_stamp.snapshot_revision > 0);
    let (actual, basis) = assert_full_fallback(&fixture, &changed);
    assert_eq!(
        basis.snapshot_revision,
        rejected.snapshot_stamp.snapshot_revision
    );
    assert_eq!(
        basis.canonical_generation,
        rejected.snapshot_stamp.canonical_generation
    );
    assert!(persist_context_snapshot_from_basis(&fixture.path, &actual.messages, &basis).unwrap());
    let written = fixture.read(&changed);
    assert_eq!(
        written.snapshot_stamp.snapshot_revision,
        rejected.snapshot_stamp.snapshot_revision + 1,
    );
    assert!(written.snapshot_meta.unwrap().compression_state.is_some());
}

#[test]
fn prefix_cert_request_nonplain_tail_and_budget_overflow_run_full_pipeline() {
    let _guard = ENV_LOCK.lock().unwrap();
    let config = RequestConfig::default();
    let mut reasoning = message("assistant", "Continuation with provider replay.");
    reasoning.tool_calls = Some(vec![ToolCall {
        id: "read-tail".into(),
        tool_type: "function".into(),
        function: FunctionCall {
            name: "read_file".into(),
            arguments: r#"{"file_path":"a.rs"}"#.into(),
        },
    }]);
    reasoning.reasoning_content = Some(compress::encode_reasoning_replay_state(
        "fixture-model",
        "Retained reasoning.",
    ));
    let mut structured = message("user", "");
    structured.content = json!([{"type": "text", "text": "Structured continuation."}]);
    let oversized = message("assistant", &"x".repeat(config.max_chars + 1));
    for tail in [reasoning, structured, oversized] {
        let fixture = Fixture::new();
        fixture.snapshot(&plain_prefix(), &config);
        append_history_messages(&fixture.path, &[tail]).unwrap();
        let _ = assert_full_fallback(&fixture, &config);
    }
}

#[tokio::test]
async fn prefix_cert_post_turn_snapshot_persists_certificate_and_reads_hit() {
    let _guard = ENV_LOCK.lock().unwrap();
    let mut fixture = Fixture::new();
    let mut app = crate::ai::tests::test_app_with_cancel_stream(Default::default());
    app.config.history_file = fixture.dir.join("history.jsonl");
    app.session_id = "b2-post-turn".into();
    let store = SessionStore::new(&app.config.history_file);
    fixture.path = store.session_history_file(&app.session_id);
    app.session_history_file = fixture.path.clone();
    let note = message(ROLE_INTERNAL_NOTE, "[mid-turn-summary] Prior decision.");
    let mut input = plain_prefix();
    input.insert(1, note.clone());
    input.insert(1, note);
    let mut expected = input.clone();
    expected.remove(1);
    let config = RequestConfig {
        max_chars: messages_total_chars_pub(&expected) + 1,
        overflow_dir: Some(store.session_assets_dir(&app.session_id)),
        ..RequestConfig::default()
    };
    app.config.history_max_chars = config.max_chars;
    app.config.history_keep_last = config.keep_last;
    app.config.history_summary_max_chars = config.summary_chars;
    append_history_messages(&fixture.path, &input).unwrap();
    fixture.restart();
    assert!(messages_total_chars_pub(&fixture.read(&config).messages) > config.max_chars);
    compact_session_history_with_app(&app, None).await.unwrap();
    let context = fixture.read(&config);
    let meta = context
        .snapshot_meta
        .as_ref()
        .expect("post-turn must write a snapshot");
    assert!(meta.compression_state.is_some());
    assert!(messages_total_chars_pub(&context.messages) < messages_total_chars_pub(&input));
    fixture.restart();
    let (actual, _) = config.build(&fixture.path);
    assert!(config.certificate(&actual.messages).is_none());
    let full = config.full(context.messages);
    assert!(config.certificate(&full.messages).is_some());
    assert_equal_outcomes(&actual, &full);
    assert_eq!(
        serde_json::to_vec(&build_message_arr(usize::MAX, &fixture.path).unwrap()).unwrap(),
        serde_json::to_vec(&input).unwrap(),
        "post-turn compression must preserve canonical history bytes"
    );
}

#[test]
fn prefix_cert_request_raw_tool_cap_rejects_certificate_and_preserves_before_chars() {
    let _guard = ENV_LOCK.lock().unwrap();
    let fixture = Fixture::new();
    let config = RequestConfig {
        max_chars: usize::MAX,
        overflow_dir: Some(fixture.dir.join("assets")),
        ..RequestConfig::default()
    };
    let mut call = message("assistant", "Read retained evidence.");
    call.tool_calls = Some(vec![ToolCall {
        id: "read-raw".into(),
        tool_type: "function".into(),
        function: FunctionCall {
            name: "read_file".into(),
            arguments: r#"{"file_path":"a.rs"}"#.into(),
        },
    }]);
    let mut result = message("tool", &"retained row\n".repeat(6_000));
    result.tool_call_id = Some("read-raw".into());
    let input = vec![
        message("system", "Stable policy."),
        message("user", "Read the evidence."),
        call,
        result,
        message("assistant", "Evidence retained."),
    ];
    append_history_messages(&fixture.path, &input).unwrap();
    fixture.restart();
    // Certify a full-pipeline identity before the request's independent raw cap.
    let outcome = compress_messages_for_context_with_outcome(
        input,
        config.max_chars,
        config.keep_last(),
        config.summary_chars,
        config.overflow_dir.clone(),
        None,
    );
    assert!(config.certificate(&outcome.messages).is_some());
    let context = fixture.read(&config);
    let basis = ContextSnapshotBasis::from_context(&context, config.fingerprint(), config.policy());
    assert!(persist_context_snapshot_from_basis(&fixture.path, &outcome.messages, &basis).unwrap());
    let context = fixture.read(&config);
    assert!(
        try_reuse_context_snapshot_prefix(
            &context.messages,
            context.snapshot_meta.as_ref(),
            config.policy().as_ref(),
        )
        .is_some()
    );
    let mut capped = context.messages.clone();
    assert!(
        compress::cap_raw_tool_results_for_context(
            &mut capped,
            config.overflow_dir.as_deref(),
            None,
        ) > 0
    );
    assert!(
        try_reuse_context_snapshot_prefix(
            &capped,
            context.snapshot_meta.as_ref(),
            config.policy().as_ref(),
        )
        .is_none()
    );
    fixture.restart();
    let (actual, _) = config.build(&fixture.path);
    assert_eq!(
        actual.before_chars,
        messages_total_chars_pub(&context.messages)
    );
    assert!(actual.after_chars < actual.before_chars);
    let full = config.full(context.messages);
    assert_equal_outcomes(&actual, &full);
}
