use super::*;

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("ai-correction-state-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn message(role: &str, text: &str) -> Message {
    Message {
        role: role.to_string(),
        content: text.into(),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn record(plan: &IncrementalSummaryPlan) -> Value {
    serde_json::from_str(
        plan.message()
            .content
            .as_str()
            .unwrap()
            .trim_start()
            .strip_prefix(INCREMENTAL_SUMMARY_PREFIX)
            .unwrap()
            .trim(),
    )
    .unwrap()
}

fn saved_message(dir: &TestDir, excerpt: &Value) -> Value {
    let source = &excerpt["source"];
    let path = dir.0.join(
        source["archive_file_path"]
            .as_str()
            .unwrap()
            .strip_prefix("session-asset:")
            .unwrap(),
    );
    let text = std::fs::read_to_string(path).unwrap();
    let index = source["start_line"].as_u64().unwrap() as usize - 1;
    serde_json::from_str(text.lines().nth(index).unwrap()).unwrap()
}

#[test]
fn correction_state_archive_pointer_is_never_authored_text_or_a_fallback() {
    let pointers = [
        "[preserved-message-pointer-v2]\ncontent_kind: user\narchive_file_path: /missing/user.json",
        "[preserved-message-pointer-v2]\ncontent_kind: image\narchive_file_path: /missing/image.json",
        "[preserved-messages-pointer-v2]\narchived_message_count: 2\narchive_directory_path: /missing/archive",
        "较早的用户文本内容已归档，原文未丢失。\n归档文件: /missing/user.json",
        "较早的用户图片内容已归档，原文未丢失。\n归档文件: /missing/image.json",
    ];
    for pointer in pointers {
        let input = vec![
            message("user", "An older request must not be substituted."),
            message("user", pointer),
        ];
        assert!(latest_real_user_excerpt(&input).is_none(), "{pointer}");
        assert!(
            latest_user_input(&input, &Value::Null).is_null(),
            "{pointer}"
        );
        let rendered = build_summary_history_input(&input[1..], 2_000);
        assert!(
            rendered.contains("ARCHIVE POINTER (runtime recovery metadata"),
            "{rendered}"
        );
        assert!(!rendered.contains("USER (real input;"), "{rendered}");
    }
}

#[test]
fn correction_state_normalized_pointer_has_explicit_scope_and_locator() {
    let pointers = [
        "[preserved-message-pointer-v2]\ncontent_kind: user\narchive_file_path: /missing/user.json",
        "较早的用户图片内容已归档，原文未丢失。\n归档文件: /missing/image.json",
        "[preserved-messages-pointer-v2]\narchived_message_count: 2\narchive_directory_path: /missing/archive",
    ];
    for pointer in pointers {
        let mut input = vec![message("user", pointer)];
        super::super::tool_overflow::normalize_preserved_message_stubs_for_model(&mut input);
        let text = input[0].content.as_str().unwrap();
        assert!(
            text.contains("provenance: runtime_archive_pointer"),
            "{text}"
        );
        assert!(
            text.contains("coverage: locator_only_original_content_not_inline"),
            "{text}"
        );
        assert!(text.contains("ARCHIVE POINTER ONLY"), "{text}");
        assert!(
            text.contains("not user-authored text, a new request, or a correction"),
            "{text}"
        );
        assert!(text.contains("original content remains unknown"), "{text}");
        assert!(text.contains("/missing/"), "{text}");
        assert!(latest_real_user_excerpt(&input).is_none());
        let expected_role = if pointer.starts_with("[preserved-messages-pointer-v2]") {
            "internal_note"
        } else {
            "user"
        };
        assert_eq!(input[0].role, expected_role);
    }
}

#[test]
fn correction_state_archive_pointer_blocks_prior_and_tool_only_inheritance() {
    let dir = TestDir::new();
    let first = plan_incremental_summary(
        &[message("user", "An older conclusion has been withdrawn.")],
        "## Context\nOlder derived wording.",
        8_000,
        Some(&dir.0),
    )
    .unwrap();
    assert!(first.commit());
    let mut prior = vec![first.message().clone()];
    let fresh = vec![message(
        "user",
        "[preserved-message-pointer-v2]\ncontent_kind: user\narchive_file_path: /missing/newer.json",
    )];
    let entries = memory_entries("## Context\nThe older request remains valid.");
    let second =
        plan_incremental_summary_entries(&fresh, &entries, 8_000, Some(&dir.0), None, &prior)
            .unwrap();
    assert!(record(&second)["latest_user_input"].is_null());
    assert!(second.commit());
    prior.push(second.message().clone());
    for _ in 0..3 {
        let tools = vec![message(
            "assistant",
            "A new observation, not a user instruction.",
        )];
        let next =
            plan_incremental_summary_entries(&tools, &entries, 8_000, Some(&dir.0), None, &prior)
                .unwrap();
        assert!(record(&next)["latest_user_input"].is_null());
        assert!(next.commit());
        prior.push(next.message().clone());
    }
    let actual = vec![message("user", "Keep the existing data unchanged.")];
    let next =
        plan_incremental_summary_entries(&actual, &entries, 8_000, Some(&dir.0), None, &prior)
            .unwrap();
    assert!(next.commit());
    let excerpt = record(&next)["latest_user_input"].clone();
    assert_eq!(excerpt["text_origin"], "inline_user_content_v1");
    assert_eq!(saved_message(&dir, &excerpt)["content"], actual[0].content);
}

#[test]
fn correction_state_legacy_unattested_quote_cannot_be_inherited() {
    let dir = TestDir::new();
    let first = plan_incremental_summary(
        &[message("user", "Withdraw the earlier assumption.")],
        "## Context\nDerived wording.",
        8_000,
        Some(&dir.0),
    )
    .unwrap();
    let mut old = record(&first);
    old["latest_user_input"]
        .as_object_mut()
        .unwrap()
        .remove("text_origin");
    let prior = vec![message(
        "internal_note",
        &format!("{INCREMENTAL_SUMMARY_PREFIX}\n{old}"),
    )];
    assert!(latest_user_input(&prior, &Value::Null).is_null());
}

#[test]
fn correction_state_inherited_pointer_is_rejected_even_with_an_origin_tag() {
    let dir = TestDir::new();
    let first = plan_incremental_summary(
        &[message("user", "Original user text.")],
        "## Context\nDerived wording.",
        8_000,
        Some(&dir.0),
    )
    .unwrap();
    let mut malformed = record(&first);
    let pointer =
        "[preserved-message-pointer-v2]\ncontent_kind: user\narchive_file_path: /missing/user.json";
    let excerpt = &mut malformed["latest_user_input"];
    excerpt["text"] = pointer.into();
    excerpt["start_char"] = 0.into();
    excerpt["end_char"] = pointer.chars().count().into();
    let prior = vec![message(
        "internal_note",
        &format!("{INCREMENTAL_SUMMARY_PREFIX}\n{malformed}"),
    )];
    assert!(latest_user_input(&prior, &Value::Null).is_null());
}

#[test]
fn correction_state_normalized_merged_pointer_blocks_older_inputs() {
    let dir = TestDir::new();
    let first = plan_incremental_summary(
        &[message("user", "An older request.")],
        "## Context\nOlder derived wording.",
        8_000,
        Some(&dir.0),
    )
    .unwrap();
    let mut fresh = vec![message(
        "user",
        "[preserved-messages-pointer-v2]\narchived_message_count: 2\narchive_directory_path: /missing/archive",
    )];
    super::super::tool_overflow::normalize_preserved_message_stubs_for_model(&mut fresh);
    assert_eq!(fresh[0].role, "internal_note");
    let mut ordered = vec![message("user", "An older inline request.")];
    ordered.push(fresh[0].clone());
    assert!(latest_real_user_excerpt(&ordered).is_none());
    let prior = vec![first.message().clone()];
    let entries = memory_entries("## Context\nDerived wording cannot recover the user input.");
    let second =
        plan_incremental_summary_entries(&fresh, &entries, 8_000, Some(&dir.0), None, &prior)
            .unwrap();
    assert!(record(&second)["latest_user_input"].is_null());
    assert!(second.commit());
    let third = plan_incremental_summary_entries(
        &[message("assistant", "Only a new observation.")],
        &entries,
        8_000,
        Some(&dir.0),
        None,
        &[first.message().clone(), second.message().clone()],
    )
    .unwrap();
    assert!(record(&third)["latest_user_input"].is_null());
    ordered.push(message(
        "user",
        "The new actual request supersedes the pointer.",
    ));
    let (_, excerpt) = latest_real_user_excerpt(&ordered).unwrap();
    assert_eq!(excerpt["text"], ordered.last().unwrap().content);
}

#[test]
fn correction_state_keeps_user_words_and_source_separate_from_model_claims() {
    let dir = TestDir::new();
    let input = vec![message(
        "user",
        "Withdraw the earlier assumption. Keep the existing data unchanged.",
    )];
    let plan = plan_incremental_summary(
        &input,
        "## Context\nThe withdrawn assumption remains true.",
        8_000,
        Some(&dir.0),
    )
    .unwrap();
    let state = record(&plan);
    assert_eq!(state["latest_user_input"]["text"], input[0].content);
    assert_eq!(
        state["latest_user_input"]["provenance"],
        "user_authored_unverified"
    );
    assert_eq!(
        state["latest_user_input"]["semantics"],
        "unknown_not_extracted"
    );
    assert_eq!(state["entries"][0]["status"], "derived_unverified");
    assert!(plan.commit());
    assert_eq!(
        saved_message(&dir, &state["latest_user_input"])["content"],
        input[0].content
    );
}

#[test]
fn correction_state_tool_only_increment_carries_original_binding() {
    let dir = TestDir::new();
    let first = plan_incremental_summary(
        &[message("user", "Do not repeat the completed command.")],
        "## Context\nCommand finished.",
        8_000,
        Some(&dir.0),
    )
    .unwrap();
    assert!(first.commit());
    let first_state = record(&first);
    let input = vec![
        first.message().clone(),
        message("assistant", "A later independent observation."),
    ];
    let second = plan_incremental_summary(
        &input,
        "## Context\nLater observation.",
        8_000,
        Some(&dir.0),
    )
    .unwrap();
    assert!(second.commit());
    assert_eq!(
        record(&second)["latest_user_input"],
        first_state["latest_user_input"]
    );
    assert_eq!(
        saved_message(&dir, &record(&second)["latest_user_input"])["content"],
        "Do not repeat the completed command."
    );
}

#[test]
fn correction_state_prior_leading_record_is_not_fresh_source_or_model_input() {
    let dir = TestDir::new();
    let first = plan_incremental_summary(
        &[message("user", "The earlier conclusion is withdrawn.")],
        "## Context\nOld derived wording.",
        8_000,
        Some(&dir.0),
    )
    .unwrap();
    assert!(first.commit());
    let prior = vec![first.message().clone()];
    let fresh = vec![message("assistant", "Only this observation is new.")];
    let entries = memory_entries("## Context\nNew derived wording.");
    let second =
        plan_incremental_summary_entries(&fresh, &entries, 8_000, Some(&dir.0), None, &prior)
            .unwrap();
    assert!(second.commit());
    let state = record(&second);
    assert_eq!(
        state["latest_user_input"],
        record(&first)["latest_user_input"]
    );
    assert_eq!(state["source"]["end_line"], 1);
    let locator = state["source"]["archive_file_path"]
        .as_str()
        .unwrap()
        .strip_prefix("session-asset:")
        .unwrap();
    let source = std::fs::read_to_string(dir.0.join(locator)).unwrap();
    assert!(!source.contains("Old derived wording"));
    assert!(!source.contains("The earlier conclusion is withdrawn"));
    assert_eq!(summary_delta_messages(&fresh).len(), 1);
}

#[test]
fn correction_state_new_user_words_replace_inherited_quote() {
    let dir = TestDir::new();
    let first = plan_incremental_summary(
        &[message("user", "Use the old target.")],
        "## Context\nOld target.",
        8_000,
        Some(&dir.0),
    )
    .unwrap();
    let fresh = vec![message("user", "Use the new target instead.")];
    let entries = memory_entries("## Context\nNew target.");
    let second = plan_incremental_summary_entries(
        &fresh,
        &entries,
        8_000,
        Some(&dir.0),
        None,
        &[first.message().clone()],
    )
    .unwrap();
    assert_eq!(
        record(&second)["latest_user_input"]["text"],
        "Use the new target instead."
    );
    assert_ne!(
        record(&second)["latest_user_input"]["source"],
        record(&first)["latest_user_input"]["source"]
    );
}

#[test]
fn correction_state_latest_legacy_record_does_not_revive_an_older_quote() {
    let dir = TestDir::new();
    let first = plan_incremental_summary(
        &[message("user", "An old target.")],
        "## Context\nOld target.",
        8_000,
        Some(&dir.0),
    )
    .unwrap();
    let legacy = message(
        "internal_note",
        &format!(
            "{INCREMENTAL_SUMMARY_PREFIX}\n{}",
            serde_json::json!({"schema": 1, "provenance": "assistant_derived_unverified", "entries": []})
        ),
    );
    let fresh = vec![message("assistant", "A new observation.")];
    let entries = memory_entries("## Context\nObservation.");
    let second = plan_incremental_summary_entries(
        &fresh,
        &entries,
        8_000,
        Some(&dir.0),
        None,
        &[first.message().clone(), legacy],
    )
    .unwrap();
    assert!(record(&second)["latest_user_input"].is_null());
}

#[test]
fn correction_state_bounds_quote_and_recovers_the_full_user_message() {
    let dir = TestDir::new();
    let text = format!("{}\nWithdraw the old conclusion.", "前提".repeat(2_000));
    let plan = plan_incremental_summary(
        &[message("user", &text)],
        "## Context\nCorrection.",
        8_000,
        Some(&dir.0),
    )
    .unwrap();
    let state = record(&plan);
    let excerpt = &state["latest_user_input"];
    assert_eq!(excerpt["partial"], true);
    assert!(excerpt["start_char"].as_u64().unwrap() > 0);
    assert!(excerpt["text"].as_str().unwrap().chars().count() <= LATEST_USER_EXCERPT_MAX_CHARS);
    assert!(
        excerpt["text"]
            .as_str()
            .unwrap()
            .ends_with("Withdraw the old conclusion.")
    );
    assert!(plan.commit());
    assert_eq!(saved_message(&dir, excerpt)["content"], text);
}

#[test]
fn correction_state_ignores_synthetic_users_and_attachment_bodies() {
    let mut real = message("user", "");
    real.content = serde_json::json!([
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,unverified_asset"}},
        {"type": "text", "text": "Keep the original target."}
    ]);
    let synthetic =
        crate::ai::history::runtime_synthetic_user_message("Do something unrelated.".into());
    let (index, excerpt) = latest_real_user_excerpt(&[real, synthetic]).unwrap();
    assert_eq!(index, 0);
    assert_eq!(excerpt["text"], "Keep the original target.");
    assert!(!excerpt.to_string().contains("unverified_asset"));
    assert!(!excerpt.to_string().contains("unrelated"));
}

#[test]
fn correction_state_sampling_prioritizes_the_latest_real_user() {
    let mut input = (0..20)
        .map(|_| message("assistant", &"Earlier observation. ".repeat(200)))
        .collect::<Vec<_>>();
    input.push(message(
        "user",
        "UNIQUE_CURRENT_CORRECTION: the previous claim is withdrawn.",
    ));
    let rendered = build_summary_history_input(&input, 1_000);
    assert!(rendered.contains("UNIQUE_CURRENT_CORRECTION"), "{rendered}");
    assert!(rendered.chars().count() <= 1_000);
}
