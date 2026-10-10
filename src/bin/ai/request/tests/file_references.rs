use super::common::{first_openai_vl_model_name, test_attachment_assets_dir, write_image_snapshot};
use super::super::*;
use crate::ai::tools::os_tools::{GLOBAL_OS, init_os_tools_globals};
use crate::ai::{cli::ParsedCli, types::AppConfig};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, atomic::AtomicBool};

#[test]
fn build_reference_content_keeps_image_boundary_for_vl_model() {
    let Some(model) = first_openai_vl_model_name() else {
        eprintln!(
            "[test] skipping build_reference_content_keeps_image_boundary_for_vl_model: \
                 no OpenAi+VL model present in model registry"
        );
        return;
    };

    let assets_dir = test_attachment_assets_dir();
    let path = write_image_snapshot(&assets_dir, "sample.png", b"fake");

    let value = build_reference_content(&model, "describe", &[path], "", &assets_dir).unwrap();

    let parts = value.as_array().unwrap();
    assert_eq!(parts.len(), 2, "one reference part + one text part");
    let first = &parts[0];
    assert_eq!(first.get("type").and_then(Value::as_str), Some("reference"));
    assert_eq!(first.get("kind").and_then(Value::as_str), Some("image"));
    assert_eq!(
        first.get("name").and_then(Value::as_str),
        Some("sample.png")
    );
    let asset = first.get("asset").and_then(Value::as_str).unwrap();
    assert!(asset.starts_with("attachments/"));
    assert!(asset.ends_with("/sample.png"));
    // The persisted form must never contain base64 image data.
    let serialized = value.to_string();
    assert!(!serialized.contains("base64,"));
    assert_eq!(parts[1].get("type").and_then(Value::as_str), Some("text"));
    assert_eq!(
        parts[1].get("text").and_then(Value::as_str),
        Some("describe")
    );
}

#[test]
fn build_reference_content_degrades_to_string_without_files() {
    let assets_dir = test_attachment_assets_dir();
    let value = build_reference_content("any-model", "hello", &[], "", &assets_dir).unwrap();
    assert_eq!(value, Value::String("hello".to_string()));
}

#[test]
fn materialize_references_roundtrips_reference_back_to_inline_image() {
    let Some(model) = first_openai_vl_model_name() else {
        eprintln!(
            "[test] skipping materialize_references_roundtrips_reference_back_to_inline_image: \
                 no OpenAi+VL model present in model registry"
        );
        return;
    };
    let assets_dir = test_attachment_assets_dir();
    let path = write_image_snapshot(&assets_dir, "sample.png", b"fake");

    let mut content =
        build_reference_content(&model, "describe", &[path], "", &assets_dir).unwrap();
    assert!(materialize_references(&mut content, Some(&assets_dir)));

    let parts = content.as_array().unwrap();
    let first = &parts[0];
    assert_eq!(first.get("type").and_then(Value::as_str), Some("image_url"));
    assert!(
        first
            .get("image_url")
            .and_then(|v| v.get("url"))
            .and_then(Value::as_str)
            .map(|s| s.starts_with("data:image/png;base64,"))
            .unwrap_or(false)
    );
    assert_eq!(
        parts[1].get("text").and_then(Value::as_str),
        Some("describe")
    );
}

#[test]
fn materialize_references_is_idempotent_for_legacy_content() {
    // Plain strings and already-inline image_url arrays are legacy persisted
    // forms; materialization must pass them through untouched.
    let mut plain = Value::String("no image".to_string());
    assert!(!materialize_references(&mut plain, None));
    assert_eq!(plain, Value::String("no image".to_string()));

    let mut legacy = serde_json::json!([
        { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } },
        { "type": "text", "text": "old" }
    ]);
    assert!(!materialize_references(&mut legacy, None));
    assert_eq!(
        legacy[0].get("type").and_then(Value::as_str),
        Some("image_url")
    );
}

#[test]
fn materialize_references_degrades_missing_file_to_text_marker() {
    let assets_dir = test_attachment_assets_dir();
    let mut content = serde_json::json!([
        { "type": "reference", "kind": "image", "name": "gone.png", "asset": "attachments/missing/gone.png" },
        { "type": "text", "text": "describe" }
    ]);
    assert!(materialize_references(&mut content, Some(&assets_dir)));
    let parts = content.as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].get("type").and_then(Value::as_str), Some("text"));
    assert!(
        parts[0]
            .get("text")
            .and_then(Value::as_str)
            .map(|t| t.contains("引用图片快照不可用"))
            .unwrap_or(false)
    );
    assert_eq!(
        parts[1].get("text").and_then(Value::as_str),
        Some("describe")
    );
}

#[test]
fn build_reference_content_keeps_text_file_boundary_for_any_model() {
    let assets_dir = test_attachment_assets_dir();
    let attachments = "[Attached text file: /tmp/source.rs]\nfn run() {}\n[/Attached text file]";
    let value =
        build_reference_content("non-vl-model", "review this", &[], attachments, &assets_dir)
            .unwrap();

    let parts = value.as_array().unwrap();
    assert_eq!(parts.len(), 2, "one file reference part + one text part");
    let first = &parts[0];
    assert_eq!(first.get("type").and_then(Value::as_str), Some("reference"));
    assert_eq!(first.get("kind").and_then(Value::as_str), Some("file"));
    assert_eq!(
        first.get("name").and_then(Value::as_str),
        Some("attached files")
    );
    assert_eq!(first.get("text").and_then(Value::as_str), Some(attachments));
    assert_eq!(parts[1].get("type").and_then(Value::as_str), Some("text"));
    assert_eq!(
        parts[1].get("text").and_then(Value::as_str),
        Some("review this")
    );
}

#[test]
fn file_reference_snapshot_counts_toward_request_budget() {
    let assets_dir = test_attachment_assets_dir();
    let attachments = "x".repeat(12_000);
    let content =
        build_reference_content("non-vl-model", "review", &[], &attachments, &assets_dir).unwrap();
    let message = Message {
        role: "user".to_string(),
        content,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    };
    assert!(crate::ai::history::message_billable_chars(&message) >= attachments.len());
}

#[test]
fn materialize_references_restores_text_file_inline_content() {
    let assets_dir = test_attachment_assets_dir();
    let attachments = "[Attached text file: /tmp/source.rs]\nfn run() {}\n[/Attached text file]";
    let mut content =
        build_reference_content("non-vl-model", "review this", &[], attachments, &assets_dir)
            .unwrap();
    assert!(materialize_references(&mut content, None));

    let parts = content.as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].get("type").and_then(Value::as_str), Some("text"));
    let text = parts[0].get("text").and_then(Value::as_str).unwrap();
    assert_eq!(text, attachments);
    assert_eq!(
        parts[1].get("text").and_then(Value::as_str),
        Some("review this")
    );
}

#[test]
fn materialize_references_degrades_missing_text_file_to_marker() {
    let mut content = serde_json::json!([
        { "type": "reference", "kind": "file", "name": "gone.txt" },
        { "type": "text", "text": "review" }
    ]);
    assert!(materialize_references(&mut content, None));
    let parts = content.as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].get("type").and_then(Value::as_str), Some("text"));
    assert!(
        parts[0]
            .get("text")
            .and_then(Value::as_str)
            .map(|t| t.contains("引用文件快照不可用"))
            .unwrap_or(false)
    );
    assert_eq!(parts[1].get("text").and_then(Value::as_str), Some("review"));
}

#[test]
fn build_reference_content_supports_multiple_references() {
    let Some(model) = first_openai_vl_model_name() else {
        eprintln!(
            "[test] skipping build_reference_content_supports_multiple_references: \
                 no OpenAi+VL model present in model registry"
        );
        return;
    };
    let assets_dir = test_attachment_assets_dir();
    let img_paths = vec![
        write_image_snapshot(&assets_dir, "one.png", b"fake"),
        write_image_snapshot(&assets_dir, "two.png", b"fake"),
    ];
    let attachments =
        "[Attached text file: one.txt]\ncontent one\n\n[Attached text file: two.txt]\ncontent two";

    let mut content =
        build_reference_content(&model, "describe", &img_paths, attachments, &assets_dir).unwrap();
    let parts = content.as_array().unwrap();
    // Two image references + one immutable text snapshot + user text.
    assert_eq!(parts.len(), 4);
    assert_eq!(parts[0].get("kind").and_then(Value::as_str), Some("image"));
    assert_eq!(parts[1].get("kind").and_then(Value::as_str), Some("image"));
    assert_eq!(parts[2].get("kind").and_then(Value::as_str), Some("file"));
    assert_eq!(parts[3].get("type").and_then(Value::as_str), Some("text"));
    let serialized = content.to_string();
    assert!(!serialized.contains("base64,"));
    assert!(serialized.contains("content one"));

    // Materializing restores images and replays the captured file snapshot.
    assert!(materialize_references(&mut content, Some(&assets_dir)));
    let parts = content.as_array().unwrap();
    assert_eq!(parts.len(), 4);
    assert_eq!(
        parts[0].get("type").and_then(Value::as_str),
        Some("image_url")
    );
    assert_eq!(
        parts[1].get("type").and_then(Value::as_str),
        Some("image_url")
    );
    let texts: Vec<&str> = parts
        .iter()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect();
    assert!(texts.iter().any(|t| t.contains("content one")));
    assert!(texts.iter().any(|t| t.contains("content two")));
    assert!(texts.iter().any(|t| t.contains("describe")));
}

#[test]
fn materialize_references_degrades_unknown_kind_to_marker() {
    // A future reference kind must never leak as raw JSON onto the provider
    // wire (OpenAI-style content parts reject an unknown `type`); degrade it to
    // an explicit text marker instead.
    let mut content = serde_json::json!([
        { "type": "reference", "kind": "audio", "name": "clip.m4a", "path": "/tmp/clip.m4a" },
        { "type": "text", "text": "describe" }
    ]);
    assert!(materialize_references(&mut content, None));
    let parts = content.as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].get("type").and_then(Value::as_str), Some("text"));
    assert!(
        parts[0]
            .get("text")
            .and_then(Value::as_str)
            .map(|t| t.contains("引用内容类型未知") && t.contains("audio"))
            .unwrap_or(false)
    );
    assert_eq!(
        parts[1].get("text").and_then(Value::as_str),
        Some("describe")
    );
}
