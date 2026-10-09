//! Cross-turn image-digest backfill: entries reported as pending are fetched, persisted
//! immediately, and become digest text on the following load.

use super::{DigestSource, MAX_IMAGE_DIGEST_BACKFILLS_PER_TURN, backfill_pending_image_digests};
use crate::ai::{history, model_names, request};

/// One image attachment captured in the session assets directory, in the canonical
/// `reference` content form a finished turn persists for it.
fn digest_image_asset(
    assets: &std::path::Path,
    file_name: &str,
    question: &str,
) -> (std::path::PathBuf, serde_json::Value) {
    let path = assets.join(file_name);
    std::fs::write(&path, b"\x89PNG\r\n\x1a\n").expect("write asset");
    let content = request::build_reference_content(
        &model_key_with_vl(true),
        question,
        &[path.to_string_lossy().into_owned()],
        "",
        assets,
    )
    .expect("canonical reference content");
    (path, content)
}

fn model_key_with_vl(is_vl: bool) -> String {
    model_names::all()
        .into_iter()
        .find(|model| model.is_vl == is_vl)
        .map(|model| model.key.clone())
        .expect("model registry has a model with this capability")
}

fn digest_message(role: &str, content: serde_json::Value) -> history::Message {
    history::Message {
        role: role.to_string(),
        content,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

fn image_message_fingerprint(message: &history::Message) -> String {
    request::last_image_user_message_fingerprint(std::slice::from_ref(message))
        .expect("image message has a fingerprint")
}

/// Entries whose digest never reached history metadata are retried: each fetched digest is
/// written immediately so the next turn's load replaces that image, an entry whose fetch
/// fails stays pending, and a model that cannot see images triggers no work at all.
#[tokio::test]
async fn pending_image_digests_are_persisted_and_replaced_on_the_next_load() {
    let root = std::env::temp_dir().join(format!("image_digest_backfill_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let assets = root.join("assets");
    std::fs::create_dir_all(&assets).expect("create assets dir");
    let history_file = root.join("history.sqlite");

    let (path_a, content_a) = digest_image_asset(&assets, "shot-a.png", "第一张图是什么？");
    let (_path_b, content_b) = digest_image_asset(&assets, "shot-b.png", "第二张图是什么？");
    history::append_history_messages(
        &history_file,
        &[
            digest_message("user", content_a.clone()),
            digest_message("assistant", serde_json::Value::String("看到了".into())),
            digest_message("user", content_b.clone()),
        ],
    )
    .expect("store canonical messages");

    // The canonical content must survive the real store round trip unchanged: digest records
    // are keyed by a hash of that content, so any rewrite would break the key computed at
    // turn end and silently disable digest replacement.
    let mut loaded = history::read_all_messages_sqlite(&history_file).expect("load");
    let loaded_a = loaded
        .iter()
        .find(|m| m.role == "user" && m.content == content_a)
        .expect("first image message round-tripped");
    let loaded_b = loaded
        .iter()
        .find(|m| m.role == "user" && m.content == content_b)
        .expect("second image message round-tripped");
    let fingerprint_a = image_message_fingerprint(loaded_a);
    let fingerprint_b = image_message_fingerprint(loaded_b);
    assert_eq!(
        fingerprint_a,
        image_message_fingerprint(&digest_message("user", content_a.clone())),
        "reloaded history must hash to the key the writing turn computed"
    );
    let asset_a = path_a.canonicalize().expect("asset path");

    // Nothing is digested yet: both images are reported for retry and neither is swapped.
    let pending = request::replace_old_images_with_persisted_digests(
        &history_file,
        &mut loaded,
        Some(assets.as_path()),
    )
    .expect("load pass");
    assert_eq!(pending.replaced, 0);
    assert_eq!(pending.pending.len(), 2);
    assert_eq!(pending.pending[0].fingerprint, fingerprint_a);
    assert_eq!(
        pending.pending[0].image_paths,
        vec![asset_a.to_string_lossy().into_owned()]
    );
    assert_eq!(pending.pending[1].fingerprint, fingerprint_b);

    // One fetch succeeds and one fails: only the fetched digest may reach the database, and
    // it must be written immediately rather than on some later turn.
    let stored = backfill_pending_image_digests(
        &history_file,
        &pending.pending,
        &model_key_with_vl(true),
        MAX_IMAGE_DIGEST_BACKFILLS_PER_TURN,
        DigestSource::Canned {
            values: &[Some("第一张是一只猫".to_string()), None],
        },
    )
    .await;
    assert_eq!(stored, 1, "only the digest that was fetched is persisted");

    // Next turn's load: the repaired image is replaced by its digest (keeping the original
    // path readable) while the failed one keeps first-send semantics.
    let mut next = history::read_all_messages_sqlite(&history_file).expect("reload");
    let outcome = request::replace_old_images_with_persisted_digests(
        &history_file,
        &mut next,
        Some(assets.as_path()),
    )
    .expect("load pass");
    assert_eq!(outcome.replaced, 1);
    assert_eq!(outcome.pending.len(), 1);
    assert_eq!(outcome.pending[0].fingerprint, fingerprint_b);
    let replaced = next
        .iter()
        .find(|m| m.role == "user" && m.content.to_string().contains("第一张是一只猫"))
        .expect("digest text replaced the image");
    assert!(
        replaced
            .content
            .to_string()
            .contains(asset_a.to_string_lossy().as_ref()),
        "the digest must keep the original image path readable"
    );
    assert!(
        next.iter()
            .any(|m| m.role == "user" && request::content_has_image(&m.content)),
        "the image whose digest failed is still sent in full"
    );

    // A model that cannot see images triggers no request and writes nothing.
    let stored = backfill_pending_image_digests(
        &history_file,
        &outcome.pending,
        &model_key_with_vl(false),
        MAX_IMAGE_DIGEST_BACKFILLS_PER_TURN,
        DigestSource::Canned {
            values: &[Some("不该被写入".to_string())],
        },
    )
    .await;
    assert_eq!(stored, 0);
    let mut after = history::read_all_messages_sqlite(&history_file).expect("reload");
    let after = request::replace_old_images_with_persisted_digests(
        &history_file,
        &mut after,
        Some(assets.as_path()),
    )
    .expect("load pass");
    assert_eq!(
        after.pending.len(),
        1,
        "no digest was invented without a VL model"
    );

    let _ = std::fs::remove_dir_all(&root);
}
