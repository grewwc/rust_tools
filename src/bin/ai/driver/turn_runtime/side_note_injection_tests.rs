use super::*;
use crate::ai::driver::model::{OcrExtraction, OcrImageSummary};
use crate::ai::driver::side_note::{current_target_id, push_side_note, side_note_file};
use crate::ai::driver::tests::test_app;
use crate::ai::{models, request};

fn unexpected_image_parse(_: &[String]) -> Result<Option<OcrExtraction>, String> {
    panic!("plain text and native images must not invoke image parsing")
}

fn canonical_user(text: &str) -> history::Message {
    history::Message {
        role: "user".to_string(),
        content: serde_json::json!(text),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

#[test]
fn side_note_injection_commits_canonical_order_without_duplicate_append() {
    let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let dir = std::env::temp_dir().join(format!(
        "side-note-canonical-order-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let mut app = crate::ai::driver::tests::test_app("foreground");
    app.session_history_file = dir.join("fixture.sqlite");
    let target = current_target_id();
    push_side_note(
        &app.session_history_file,
        "new constraint",
        "user",
        target.as_deref(),
    )
    .unwrap();
    let mut turn_messages = vec![
        canonical_user("original request"),
        history::Message {
            role: "assistant".to_string(),
            ..canonical_user("preceding assistant text")
        },
    ];
    let mut messages = turn_messages.clone();
    let mut persisted = 0;
    drain_side_notes_and_inject(
        &app,
        &app.current_model,
        unexpected_image_parse,
        &mut messages,
        &mut turn_messages,
        &mut persisted,
    )
    .unwrap();
    assert_eq!(persisted, 3);
    assert_eq!(
        history::read_all_messages_sqlite(&app.session_history_file).unwrap(),
        turn_messages
    );
    assert_eq!(messages, turn_messages);
    assert!(history::is_side_note_message(&turn_messages[2]));
    assert_eq!(
        std::fs::metadata(side_note_file(&app.session_history_file, target.as_deref()))
            .unwrap()
            .len(),
        0
    );
    assert!(persist_pending_turn_messages(
        &app,
        false,
        &turn_messages,
        &mut persisted
    ));
    assert_eq!(
        history::read_all_messages_sqlite(&app.session_history_file)
            .unwrap()
            .len(),
        3
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn side_note_injection_append_failure_keeps_queue_and_memory_order() {
    let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let dir = std::env::temp_dir().join(format!(
        "side-note-canonical-failure-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let mut app = crate::ai::driver::tests::test_app("foreground");
    app.session_history_file = dir.join("blocked.sqlite");
    std::fs::create_dir_all(&app.session_history_file).unwrap();
    let target = current_target_id();
    let queue = push_side_note(
        &app.session_history_file,
        "must retry",
        "user",
        target.as_deref(),
    )
    .unwrap();
    let original_queue = std::fs::read_to_string(&queue).unwrap();
    let original_messages = vec![canonical_user("original request")];
    let mut messages = original_messages.clone();
    let mut turn_messages = original_messages.clone();
    let mut persisted = 0;
    assert!(
        drain_side_notes_and_inject(
            &app,
            &app.current_model,
            unexpected_image_parse,
            &mut messages,
            &mut turn_messages,
            &mut persisted
        )
        .is_err()
    );
    assert_eq!(messages, original_messages);
    assert_eq!(turn_messages, original_messages);
    assert_eq!(persisted, 0);
    assert_eq!(std::fs::read_to_string(queue).unwrap(), original_queue);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn side_note_injection_images_persist_references_and_send_immutable_pixels() {
    let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let dir = std::env::temp_dir().join(format!("side-note-images-{}", uuid::Uuid::new_v4()));
    let mut app = test_app("foreground");
    app.session_history_file = dir.join("fixture.sqlite");
    let model = crate::ai::model_names::all()
        .into_iter()
        .find(|model| models::supports_image_input(&model.key))
        .unwrap()
        .key
        .clone();
    let assets = crate::ai::driver::side_note::assets_dir_for_history(&app.session_history_file);
    std::fs::create_dir_all(&assets).unwrap();
    let source = assets.join("shot.png");
    std::fs::write(&source, b"original screenshot").unwrap();
    let target = current_target_id();
    push_side_note(
        &app.session_history_file,
        "inspect [[image:shot.png]]",
        "user",
        target.as_deref(),
    )
    .unwrap();
    let mut messages = Vec::new();
    let mut canonical = Vec::new();
    let mut persisted = 0;
    drain_side_notes_and_inject(
        &app,
        &model,
        unexpected_image_parse,
        &mut messages,
        &mut canonical,
        &mut persisted,
    )
    .unwrap();
    assert_eq!(persisted, 1);
    assert!(history::is_side_note_message(&canonical[0]));
    assert!(
        canonical[0]
            .content
            .to_string()
            .contains("\"type\":\"reference\"")
    );
    assert!(!canonical[0].content.to_string().contains("data:image/"));
    assert!(
        messages[0]
            .content
            .to_string()
            .contains("data:image/png;base64,")
    );
    assert_eq!(
        history::read_all_messages_sqlite(&app.session_history_file).unwrap(),
        canonical
    );
    std::fs::write(source, b"replacement screenshot").unwrap();
    let mut replayed = canonical[0].content.clone();
    request::materialize_references(&mut replayed, Some(&assets));
    assert_eq!(replayed, messages[0].content);
    drain_side_notes_and_inject(
        &app,
        &model,
        unexpected_image_parse,
        &mut messages,
        &mut canonical,
        &mut persisted,
    )
    .unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(canonical.len(), 1);
    assert!(!crate::ai::driver::side_note::has_pending_side_notes(
        &app.session_history_file,
        target.as_deref()
    ));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn side_note_injection_missing_image_retains_queue_and_memory() {
    let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let dir =
        std::env::temp_dir().join(format!("side-note-missing-image-{}", uuid::Uuid::new_v4()));
    let mut app = test_app("foreground");
    app.session_history_file = dir.join("fixture.sqlite");
    let model = crate::ai::model_names::all()
        .into_iter()
        .find(|model| models::supports_image_input(&model.key))
        .unwrap()
        .key
        .clone();
    let target = current_target_id();
    let queue = push_side_note(
        &app.session_history_file,
        "[[image:missing.png]]",
        "user",
        target.as_deref(),
    )
    .unwrap();
    let original = std::fs::read_to_string(&queue).unwrap();
    let mut messages = vec![canonical_user("original")];
    let mut canonical = messages.clone();
    let mut persisted = 0;
    assert!(
        drain_side_notes_and_inject(
            &app,
            &model,
            unexpected_image_parse,
            &mut messages,
            &mut canonical,
            &mut persisted
        )
        .is_err()
    );
    assert_eq!(messages, vec![canonical_user("original")]);
    assert_eq!(canonical, messages);
    assert_eq!(persisted, 0);
    assert_eq!(std::fs::read_to_string(queue).unwrap(), original);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn side_note_injection_text_only_model_receives_parsed_images_and_preserves_new_notes() {
    let _guard = crate::ai::test_support::ENV_LOCK.lock().unwrap();
    let dir = std::env::temp_dir().join(format!("side-note-parsed-image-{}", uuid::Uuid::new_v4()));
    let mut app = test_app("foreground");
    app.session_history_file = dir.join("fixture.sqlite");
    let model = crate::ai::model_names::all()
        .into_iter()
        .find(|model| !models::supports_image_input(&model.key))
        .unwrap()
        .key
        .clone();
    let target = current_target_id();
    let queue = push_side_note(
        &app.session_history_file,
        "[[image:shot.png]]",
        "user",
        target.as_deref(),
    )
    .unwrap();
    let original = std::fs::read_to_string(&queue).unwrap();
    let mut messages = Vec::new();
    let mut canonical = Vec::new();
    let mut persisted = 0;
    let assets = crate::ai::driver::side_note::assets_dir_for_history(&app.session_history_file);
    std::fs::write(assets.join("shot.png"), b"screenshot bytes").unwrap();
    let mut parses = 0;
    drain_side_notes_and_inject(
        &app,
        &model,
        |images| {
            parses += 1;
            assert_eq!(images.len(), 1);
            assert_eq!(std::fs::read(&images[0]).unwrap(), b"screenshot bytes");
            // Appending here proves image parsing owns no queue lock, and its
            // acknowledgement must retain the newer note for the next request.
            push_side_note(
                &app.session_history_file,
                "arrived during parsing",
                "user",
                target.as_deref(),
            )
            .unwrap();
            Ok(Some(OcrExtraction {
                tool_name: "fixture image parser".to_string(),
                content: "visible screenshot text".to_string(),
                images: vec![OcrImageSummary {
                    file_name: "shot.png".to_string(),
                    extracted_chars: 23,
                    error: None,
                }],
            }))
        },
        &mut messages,
        &mut canonical,
        &mut persisted,
    )
    .unwrap();
    assert_eq!(parses, 1);
    assert_eq!(persisted, 1);
    assert_eq!(messages, canonical);
    assert!(
        messages[0]
            .content
            .as_str()
            .unwrap()
            .contains("[Attached Image Content via fixture image parser]\nvisible screenshot text")
    );
    assert!(!messages[0].content.as_str().unwrap().contains("[[image:"));
    assert!(history::is_side_note_message(&canonical[0]));
    let remaining = std::fs::read_to_string(&queue).unwrap();
    assert_ne!(remaining, original);
    assert!(remaining.contains("arrived during parsing"));
    assert!(!remaining.contains("shot.png"));
    drain_side_notes_and_inject(
        &app,
        &model,
        unexpected_image_parse,
        &mut messages,
        &mut canonical,
        &mut persisted,
    )
    .unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(
        history::read_all_messages_sqlite(&app.session_history_file).unwrap(),
        canonical
    );
    assert!(std::fs::read_to_string(queue).unwrap().is_empty());
    let _ = std::fs::remove_dir_all(dir);
}
