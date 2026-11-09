//! Snapshot-certificate rejection and byte-level differential coverage.

use serde_json::json;

use super::prefix_reuse::{
    CACHE, CACHE_MAX_BYTES, CACHE_MAX_MESSAGES, PrefixReuseCache, PrefixReuseKey, projection_bytes,
    retention_boundaries,
};
use super::*;
use crate::ai::test_support::ENV_LOCK;
use crate::ai::types::{FunctionCall, ToolCall};

const MAX_CHARS: usize = 100_000;
const KEEP_LAST: usize = 4;
const SUMMARY_CHARS: usize = 4_000;

fn message(role: &str, text: &str) -> Message {
    Message {
        role: role.to_owned(),
        content: json!(text),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn policy() -> PrefixCertificatePolicy {
    PrefixCertificatePolicy::new(MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, None).unwrap()
}

fn plain_prefix() -> Vec<Message> {
    vec![
        message("system", "Stable policy."),
        message("user", "Earlier request."),
        message("assistant", "Earlier task completed."),
    ]
}

fn compressed_prefix_input() -> Vec<Message> {
    let summary = message(ROLE_INTERNAL_NOTE, "[mid-turn-summary] Prior decision.");
    let mut call = message("assistant", "Inspect the implementation.");
    call.tool_calls = Some(vec![ToolCall {
        id: "read-1".to_owned(),
        tool_type: "function".to_owned(),
        function: FunctionCall {
            name: "read_file".to_owned(),
            arguments: r#"{"file_path":"a.rs"}"#.to_owned(),
        },
    }]);
    call.reasoning_content = Some(encode_reasoning_replay_state("fixture-model", "first"));
    let mut result = message("tool", "Exact retained evidence 文本.\n");
    result.tool_call_id = Some("read-1".to_owned());
    vec![
        message("system", "Stable policy."),
        summary.clone(),
        summary,
        message("user", "Earlier request."),
        call,
        result,
        message("assistant", "Earlier task completed."),
    ]
}

fn certify(
    input: Vec<Message>,
    policy: &PrefixCertificatePolicy,
) -> (Vec<Message>, PrefixCertificate, String) {
    *CACHE.lock().unwrap() = PrefixReuseCache::default();
    let outcome = compress_messages_for_context_with_outcome(
        input,
        policy.max_chars,
        policy.keep_last,
        policy.summary_max_chars,
        policy.overflow_dir.as_ref().map(PathBuf::from),
        policy.cwd.as_deref().map(Path::new),
    );
    assert_eq!(outcome.status, ContextCompressionStatus::Complete);
    let certificate = prefix_certificate_from_cache(&outcome.messages, policy)
        .expect("fixture must pass all B1 admission and reuse gates");
    let json = prefix_certificate_json_from_cache(&outcome.messages, policy).unwrap();
    (outcome.messages, certificate, json)
}

#[test]
fn prefix_cert_round_trip_reuses_prefix_and_tail_without_advancing_certificate() {
    let _guard = ENV_LOCK.lock().unwrap();
    let policy = policy();
    let (prefix, certificate, json) = certify(plain_prefix(), &policy);
    let decoded: PrefixCertificate = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, certificate);
    assert_eq!(certificate.prefix_message_count, prefix.len());
    assert_eq!(certificate.prefix_sha256.len(), 64);
    assert_eq!(certificate.chars, messages_total_chars(&prefix));
    assert_eq!(certificate.bytes, projection_bytes(&prefix).unwrap());
    assert_eq!(certificate.real_user_turns, 1);
    assert_eq!(certificate.older_recent_split, Some(0));
    let tail = vec![
        message("user", "Continue with UTF-8 文本."),
        message("assistant", "The next task is ready."),
    ];
    assert!(decoded.validate(&prefix, &tail, &policy));
    assert!(validate_certificate_json(&json, &prefix, &tail, &policy));
    assert_eq!(serde_json::to_string(&decoded).unwrap(), json);
    assert_eq!(
        prefix_certificate_from_cache(&prefix, &policy),
        Some(certificate)
    );
    let mut extended = prefix;
    extended.extend(tail);
    assert!(prefix_certificate_from_cache(&extended, &policy).is_none());
}

#[test]
fn prefix_cert_rejects_malformed_version_policy_digest_count_and_metrics() {
    let _guard = ENV_LOCK.lock().unwrap();
    let policy = policy();
    let (prefix, certificate, json) = certify(plain_prefix(), &policy);
    for invalid in ["{", "null", "[]", "{}"] {
        assert!(!validate_certificate_json(invalid, &prefix, &[], &policy));
    }
    let changes = [
        ("proof_version", json!(PROOF_VERSION + 1)),
        ("prefix_sha256", json!("untrusted digest")),
        ("prefix_message_count", json!(prefix.len() + 1)),
        ("chars", json!(certificate.chars + 1)),
        ("bytes", json!(certificate.bytes + 1)),
        ("real_user_turns", json!(certificate.real_user_turns + 1)),
        ("trim_start", json!(certificate.trim_start + 1)),
        ("merge_start", json!(certificate.merge_start + 1)),
        ("older_recent_split", json!(1)),
    ];
    for (field, value) in changes {
        let mut changed = serde_json::to_value(&certificate).unwrap();
        changed[field] = value;
        assert!(
            !validate_certificate_json(&changed.to_string(), &prefix, &[], &policy),
            "unexpected acceptance of changed {field}"
        );
    }
    let mut changed = serde_json::to_value(&certificate).unwrap();
    changed["policy"]["max_chars"] = json!(MAX_CHARS + 1);
    assert!(!validate_certificate_json(
        &changed.to_string(),
        &prefix,
        &[],
        &policy
    ));
    changed = serde_json::to_value(&certificate).unwrap();
    changed["unknown_proof"] = json!(true);
    assert!(!validate_certificate_json(
        &changed.to_string(),
        &prefix,
        &[],
        &policy
    ));
    changed = serde_json::to_value(&certificate).unwrap();
    changed.as_object_mut().unwrap().remove("chars");
    assert!(!validate_certificate_json(
        &changed.to_string(),
        &prefix,
        &[],
        &policy
    ));
    assert!(!validate_certificate_json(
        &json,
        &prefix[..prefix.len() - 1],
        &[],
        &policy
    ));

    let mut different_policies = vec![policy.clone(); 5];
    different_policies[0].max_chars -= 1;
    different_policies[1].keep_last -= 1;
    different_policies[2].summary_max_chars -= 1;
    different_policies[3].overflow_dir = Some("/tmp/different-assets".to_owned());
    different_policies[4].cwd = Some("/tmp/different-cwd".to_owned());
    for changed in different_policies {
        assert!(!validate_certificate_json(&json, &prefix, &[], &changed));
        assert!(prefix_certificate_from_cache(&prefix, &changed).is_none());
    }
}

#[test]
fn prefix_cert_digest_binds_every_message_field() {
    let _guard = ENV_LOCK.lock().unwrap();
    let policy = policy();
    let (prefix, certificate, _) = certify(compressed_prefix_input(), &policy);
    let call_index = prefix
        .iter()
        .position(|message| message.tool_calls.is_some())
        .unwrap();
    let result_index = prefix
        .iter()
        .position(|message| message.role == "tool")
        .unwrap();
    let mut variants = vec![prefix.clone(); 5];
    variants[0][0].content = json!("Changed policy.");
    variants[1][call_index].role = "developer".to_owned();
    variants[2][call_index].tool_calls.as_mut().unwrap()[0]
        .function
        .arguments = r#"{"file_path":"b.rs"}"#.to_owned();
    variants[3][result_index].tool_call_id = Some("read-2".to_owned());
    variants[4][call_index].reasoning_content =
        Some(encode_reasoning_replay_state("fixture-model", "other"));
    for changed_prefix in variants {
        // Repair every other metric so the original digest must reject the
        // changed envelope even when its content and retention still fit.
        let mut changed_certificate = certificate.clone();
        changed_certificate.chars = messages_total_chars(&changed_prefix);
        changed_certificate.bytes = projection_bytes(&changed_prefix).unwrap();
        changed_certificate.real_user_turns = user_input_turn_count(&changed_prefix);
        (
            changed_certificate.trim_start,
            changed_certificate.merge_start,
        ) = retention_boundaries(&changed_prefix, changed_certificate.chars);
        assert!(!changed_certificate.validate(&changed_prefix, &[], &policy));
    }
}

#[test]
fn prefix_cert_rejects_unproved_tail_payloads() {
    let _guard = ENV_LOCK.lock().unwrap();
    let policy = policy();
    let (prefix, _, json) = certify(plain_prefix(), &policy);
    let mut call = message("assistant", "Inspect a new file.");
    call.tool_calls = Some(vec![ToolCall {
        id: "new-call".to_owned(),
        tool_type: "function".to_owned(),
        function: FunctionCall {
            name: "read_file".to_owned(),
            arguments: "{}".to_owned(),
        },
    }]);
    let mut empty_calls = message("assistant", "No calls yet.");
    empty_calls.tool_calls = Some(Vec::new());
    let mut call_id = message("assistant", "Protocol sidecar.");
    call_id.tool_call_id = Some("new-call".to_owned());
    let mut reasoning = message("assistant", "Visible response.");
    reasoning.reasoning_content = Some("Reasoning sidecar.".to_owned());
    let mut image = message("user", "");
    image.content = json!([{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]);
    let mut structured_text = message("user", "");
    structured_text.content = json!([{"type":"text","text":"Structured content."}]);
    let cases = [
        ("tool result", message("tool", "New evidence.")),
        ("tool call", call),
        ("empty tool calls", empty_calls),
        ("tool call id", call_id),
        ("reasoning sidecar", reasoning),
        (
            "summary",
            message(ROLE_INTERNAL_NOTE, "[mid-turn-summary] New note."),
        ),
        (
            "quoted summary",
            message("user", "[incremental-memory-v1] New note."),
        ),
        (
            "internal note",
            message(ROLE_INTERNAL_NOTE, "Runtime policy."),
        ),
        ("quoted archive", message("assistant", ARCHIVE_NOTE_PREFIX)),
        (
            "quoted self note",
            message("assistant", "self_note: new memory"),
        ),
        ("image", image),
        ("structured text", structured_text),
        (
            "inline image",
            message("user", "data:image/png;base64,AA=="),
        ),
        (
            "preserved stub",
            message(
                "user",
                &format!(
                    "{PRESERVED_CONTENT_STUB_PREFIX}{{\"kind\":\"user\",\"file_path\":\"/tmp/user.md\"}}"
                ),
            ),
        ),
        (
            "tool evidence note",
            message("assistant", COMPRESSED_TOOL_EVIDENCE_MARKER),
        ),
        (
            "synthetic user",
            crate::ai::history::runtime_synthetic_user_message(json!("Task finished.")),
        ),
    ];
    for (name, message) in cases {
        assert!(
            !validate_certificate_json(&json, &prefix, &[message], &policy),
            "unexpected acceptance of {name}"
        );
    }
}

#[test]
fn prefix_cert_rejects_total_budget_overflow() {
    let _guard = ENV_LOCK.lock().unwrap();
    let mut policy = policy();
    policy.max_chars = messages_total_chars(&plain_prefix()) + 5;
    let (prefix, _, json) = certify(plain_prefix(), &policy);
    assert!(validate_certificate_json(
        &json,
        &prefix,
        &[message("assistant", "12345")],
        &policy
    ));
    assert!(!validate_certificate_json(
        &json,
        &prefix,
        &[message("assistant", "123456")],
        &policy
    ));
}

#[test]
fn prefix_cert_rejects_b1_byte_and_message_caps() {
    let _guard = ENV_LOCK.lock().unwrap();
    let mut policy = policy();
    policy.max_chars = CACHE_MAX_BYTES * 2;
    let (prefix, _, json) = certify(plain_prefix(), &policy);
    let tail = vec![message("assistant", &"x".repeat(CACHE_MAX_BYTES))];
    assert!(messages_total_chars(&tail) + messages_total_chars(&prefix) < policy.max_chars);
    assert!(!validate_certificate_json(&json, &prefix, &tail, &policy));
    let mut tail = vec![message("assistant", ""); CACHE_MAX_MESSAGES - prefix.len()];
    assert!(validate_certificate_json(&json, &prefix, &tail, &policy));
    tail.push(message("assistant", ""));
    assert!(!validate_certificate_json(&json, &prefix, &tail, &policy));
}

#[test]
fn prefix_cert_rejects_keep_last_crossing_before_retention_moves() {
    let _guard = ENV_LOCK.lock().unwrap();
    let mut policy = policy();
    policy.keep_last = 1;
    let (prefix, certificate, json) = certify(plain_prefix(), &policy);
    let tail = vec![message("user", "Another real request.")];
    let mut merged = prefix.clone();
    merged.extend(tail.clone());
    assert_eq!(
        retention_boundaries(&merged, messages_total_chars(&merged)),
        (certificate.trim_start, certificate.merge_start)
    );
    assert_eq!(user_input_turn_count(&merged), policy.keep_last + 1);
    assert!(!validate_certificate_json(&json, &prefix, &tail, &policy));
}

#[test]
fn prefix_cert_rejects_trim_and_merge_boundary_movement() {
    let _guard = ENV_LOCK.lock().unwrap();
    let policy = policy();
    let (prefix, certificate, json) = certify(
        vec![
            message("user", "First."),
            message(
                "assistant",
                &"p".repeat(KEEP_THREE_RECENT_USER_TURNS_MAX_CHARS - 100),
            ),
            message("user", "Second."),
            message("user", "Third."),
        ],
        &policy,
    );
    let tail = vec![message("assistant", &"q".repeat(200))];
    let mut merged = prefix.clone();
    merged.extend(tail.clone());
    let moved = retention_boundaries(&merged, messages_total_chars(&merged));
    assert_ne!(moved.0, certificate.trim_start);
    assert_eq!(moved.1, certificate.merge_start);
    assert!(!validate_certificate_json(&json, &prefix, &tail, &policy));

    let (prefix, certificate, json) = certify(merged, &policy);
    let tail = vec![message("user", "Fourth.")];
    let mut merged = prefix.clone();
    merged.extend(tail.clone());
    let moved = retention_boundaries(&merged, messages_total_chars(&merged));
    assert_ne!(moved.0, certificate.trim_start);
    assert_ne!(moved.1, certificate.merge_start);
    assert!(user_input_turn_count(&merged) <= policy.keep_last);
    assert!(!validate_certificate_json(&json, &prefix, &tail, &policy));
}

#[test]
fn prefix_cert_rejects_new_real_user_expiring_old_guidance() {
    let _guard = ENV_LOCK.lock().unwrap();
    let policy = policy();
    let (prefix, certificate, json) = certify(
        vec![
            message("user", "Current request."),
            message(
                ROLE_INTERNAL_NOTE,
                "[tool_first_use_guidance name=read_file]\nKeep exact evidence.",
            ),
        ],
        &policy,
    );
    assert!(validate_certificate_json(
        &json,
        &prefix,
        &[message("assistant", "Still the current turn.")],
        &policy
    ));
    let tail = vec![message("user", "A new request.")];
    let mut merged = prefix.clone();
    merged.extend(tail.clone());
    assert_eq!(
        retention_boundaries(&merged, messages_total_chars(&merged)),
        (certificate.trim_start, certificate.merge_start)
    );
    assert!(!validate_certificate_json(&json, &prefix, &tail, &policy));
    *CACHE.lock().unwrap() = PrefixReuseCache::from_outcome(
        PrefixReuseKey::new(MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, None),
        &ContextCompressionOutcome::new(
            merged.clone(),
            messages_total_chars(&merged),
            MAX_CHARS,
            ContextCompressionStatus::Complete,
        ),
    );
    assert!(prefix_certificate_from_cache(&merged, &policy).is_none());
}

#[test]
fn prefix_cert_keep_last_disabled_serializes_explicit_null_split() {
    let _guard = ENV_LOCK.lock().unwrap();
    let mut policy = policy();
    policy.keep_last = 0;
    let (prefix, certificate, json) = certify(plain_prefix(), &policy);
    assert_eq!(certificate.older_recent_split, None);
    let serialized: Value = serde_json::from_str(&json).unwrap();
    assert_eq!(serialized["older_recent_split"], Value::Null);
    assert_eq!(serialized["policy"]["overflow_dir"], Value::Null);
    assert_eq!(serialized["policy"]["cwd"], Value::Null);
    assert!(validate_certificate_json(
        &json,
        &prefix,
        &[message("user", "Continue.")],
        &policy
    ));
}

#[test]
fn prefix_cert_only_generates_from_exact_admitted_cache_entry() {
    let _guard = ENV_LOCK.lock().unwrap();
    let policy = policy();
    *CACHE.lock().unwrap() = PrefixReuseCache::default();
    assert!(prefix_certificate_from_cache(&plain_prefix(), &policy).is_none());
    let (prefix, _, _) = certify(plain_prefix(), &policy);
    let mut changed = prefix.clone();
    changed[0].reasoning_content = Some("Different metadata.".to_owned());
    assert!(prefix_certificate_from_cache(&changed, &policy).is_none());
    changed = prefix.clone();
    changed.push(message("assistant", "A valid but not cached tail."));
    assert!(prefix_certificate_from_cache(&changed, &policy).is_none());
    assert!(prefix_certificate_from_cache(&prefix[..prefix.len() - 1], &policy).is_none());

    let unproved = [
        vec![
            message(
                ROLE_INTERNAL_NOTE,
                &format!(
                    "{INCREMENTAL_SUMMARY_PREFIX}\n{}",
                    "s".repeat(MAX_INCREMENTAL_SUMMARY_INLINE_CHARS + 1)
                ),
            ),
            message("user", "Keep the increment."),
        ],
        vec![
            message(
                ROLE_INTERNAL_NOTE,
                &format!(
                    "{COMPRESSED_TOOL_EVIDENCE_MARKER}\n{}",
                    "e".repeat(MAX_COMPRESSED_TOOL_EVIDENCE_INLINE_CHARS + 1)
                ),
            ),
            message("user", "Keep the evidence."),
        ],
        vec![message(ROLE_INTERNAL_NOTE, "self_note: retained"); MAX_SELF_NOTES_IN_MESSAGES + 1],
    ];
    for messages in unproved {
        let outcome = ContextCompressionOutcome::new(
            messages.clone(),
            messages_total_chars(&messages),
            MAX_CHARS,
            ContextCompressionStatus::Complete,
        );
        *CACHE.lock().unwrap() = PrefixReuseCache::from_outcome(
            PrefixReuseKey::new(MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, None),
            &outcome,
        );
        assert!(prefix_certificate_from_cache(&messages, &policy).is_none());
    }
    for status in [
        ContextCompressionStatus::Disabled,
        ContextCompressionStatus::MissingArchiveSink,
        ContextCompressionStatus::ArchiveCommitFailed,
    ] {
        let outcome = ContextCompressionOutcome::new(
            prefix.clone(),
            messages_total_chars(&prefix),
            MAX_CHARS,
            status,
        );
        *CACHE.lock().unwrap() = PrefixReuseCache::from_outcome(
            PrefixReuseKey::new(MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, None),
            &outcome,
        );
        assert!(prefix_certificate_from_cache(&prefix, &policy).is_none());
    }
}

#[test]
fn prefix_cert_policy_preserves_lexical_path_key_and_memory_estimate() {
    let _guard = ENV_LOCK.lock().unwrap();
    let policy = PrefixCertificatePolicy::new(
        MAX_CHARS,
        KEEP_LAST,
        SUMMARY_CHARS,
        Some(Path::new("/tmp/./prefix-cert/../assets")),
        Some(Path::new("/tmp/./cwd")),
    )
    .unwrap();
    assert_eq!(
        policy.overflow_dir.as_deref(),
        Some("/tmp/prefix-cert/../assets")
    );
    assert_eq!(policy.cwd.as_deref(), Some("/tmp/cwd"));
    let (prefix, certificate, json) = certify(plain_prefix(), &policy);
    assert_eq!(
        certificate.bytes,
        projection_bytes(&prefix).unwrap()
            + Path::new(policy.overflow_dir.as_deref().unwrap())
                .as_os_str()
                .len()
            + Path::new(policy.cwd.as_deref().unwrap()).as_os_str().len()
    );
    assert!(validate_certificate_json(
        &json,
        &prefix,
        &[message("assistant", "Continue.")],
        &policy
    ));

    // Equivalent lexical spelling must not invent a smaller B1 byte estimate.
    let outcome = ContextCompressionOutcome::new(
        prefix.clone(),
        messages_total_chars(&prefix),
        MAX_CHARS,
        ContextCompressionStatus::Complete,
    );
    *CACHE.lock().unwrap() = PrefixReuseCache::from_outcome(
        PrefixReuseKey::new(
            MAX_CHARS,
            KEEP_LAST,
            SUMMARY_CHARS,
            Some(Path::new("/tmp/./prefix-cert/../assets")),
            Some(Path::new("/tmp/./cwd")),
        ),
        &outcome,
    );
    assert!(prefix_certificate_from_cache(&prefix, &policy).is_none());
}

#[cfg(unix)]
#[test]
fn prefix_cert_rejects_non_utf8_policy_paths() {
    use std::os::unix::ffi::OsStringExt;
    let path = PathBuf::from(std::ffi::OsString::from_vec(vec![b'/', b't', 0xff]));
    assert!(
        PrefixCertificatePolicy::new(MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, Some(&path), None)
            .is_none()
    );
    assert!(
        PrefixCertificatePolicy::new(MAX_CHARS, KEEP_LAST, SUMMARY_CHARS, None, Some(&path))
            .is_none()
    );
}

#[test]
fn prefix_cert_hit_matches_forced_full_pipeline_byte_for_byte() {
    let _guard = ENV_LOCK.lock().unwrap();
    for keep_last in [KEEP_LAST, 0] {
        let mut policy = policy();
        policy.keep_last = keep_last;
        let raw = compressed_prefix_input();
        let raw_chars = messages_total_chars(&raw);
        let (prefix, certificate, json) = certify(raw, &policy);
        assert!(
            certificate.chars < raw_chars,
            "fixture must actually compact a prefix"
        );
        // Drop the process-local proof to demonstrate that the JSON alone drives
        // the reuse branch after a restart.
        *CACHE.lock().unwrap() = PrefixReuseCache::default();
        let tail = vec![
            message(
                "user",
                "Continue with the next task.\n逐字节保留 UTF-8 内容。",
            ),
            message("assistant", "The follow-up is ready."),
        ];
        let actual = try_reuse_certificate_json(&json, &prefix, &tail, &policy)
            .expect("certificate reuse branch must be a hit");
        let mut input = prefix.clone();
        input.extend(tail);
        let full = compress_messages_for_context_full(
            input.clone(),
            policy.max_chars,
            policy.keep_last,
            policy.summary_max_chars,
            None,
            None,
        );
        assert_eq!(
            serde_json::to_vec(&actual.messages).unwrap(),
            serde_json::to_vec(&full.messages).unwrap()
        );
        assert_eq!(actual.status, full.status);
        assert_eq!(actual.before_chars, full.before_chars);
        assert_eq!(actual.after_chars, full.after_chars);
        assert_eq!(actual.max_chars, full.max_chars);
        assert_eq!(actual.before_chars, messages_total_chars(&input));
        assert!(prefix_certificate_from_cache(&prefix, &policy).is_none());
    }
}
