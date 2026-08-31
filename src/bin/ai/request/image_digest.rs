//! Vision-model (VL) image digest protocol.
//!
//! Background: within a single turn's tool-call loop, `messages` is one persistent
//! array carried through the whole turn, so base64 images inlined in user messages
//! get replayed verbatim on every request. For servers like Doubao/Ark that bill
//! TPM by the actual multimodal wire payload, consecutive tool calls keep counting
//! the same large image into the 60s window and trigger 429 rate limiting (the
//! local budget underestimates at a nominal 1024 chars, so the preflight gate
//! cannot catch it).
//!
//! Approach: in the first round, send the full original image to the model and ask
//! it to emit a fixed-format "image digest"; in later rounds, replace the image
//! parts in the request projection's `messages` with that text digest (plus the
//! original image path), so base64 is never replayed again. The canonical
//! `turn_messages` (persisted history) always keeps the original image; only the
//! request projection is rewritten — honoring the History-truthfulness invariant.
//!
//! The digest is obtained via two paths:
//! 1. Piggyback: parse the digest out of the first tool-call response's
//!    `assistant_text` / `reasoning_content` (the model emits it as instructed).
//! 2. Fallback: if the first round did not yield one, send one dedicated
//!    tool-disabled request before entering the next round to force obtaining
//!    the digest (see [`describe_image_for_digest`]).

use std::{
    hash::{Hash, Hasher},
    io,
    path::Path,
    time::Duration,
};

use rustc_hash::FxHasher;
use serde_json::{Value, json};

use super::builder::{build_content, build_request_body};
use super::{
    api_key_for_request_model, apply_request_auth, endpoint_for_request_model,
    extract_router_content,
};
use crate::ai::{history::Message, types::App};

/// Begin sentinel of the digest body. The model must emit this line verbatim;
/// the agent locates the digest by it.
pub(crate) const DIGEST_BEGIN: &str = "<<<IMAGE_DIGEST>>>";
/// End sentinel of the digest body.
pub(crate) const DIGEST_END: &str = "<<<END_IMAGE_DIGEST>>>";
/// Fixed prefix marker of the "image handling protocol" instruction injected
/// into the request's user message.
/// The replacement phase uses it to identify and remove that instruction text
/// part (its job is done).
const INSTRUCTION_TAG: &str = "[图片处理协议]";
/// Response-header timeout (seconds) for the fallback request. A lenient bound,
/// consistent with the history-summary aux requests.
const DIGEST_REQUEST_HEADER_TIMEOUT_SECS: u64 = 60;
/// Response-body read timeout (seconds) for the fallback request.
const DIGEST_REQUEST_BODY_TIMEOUT_SECS: u64 = 30;

/// The image handling protocol instruction injected into the user message in the
/// first round. Fixed runtime-owned text containing no user content. It tells the
/// model: the original image is sent only this round and only this digest remains
/// afterwards, so all visual information needed for the task must go into the digest.
pub(crate) fn digest_instruction() -> String {
    format!(
        "{INSTRUCTION_TAG} 本轮会附带原始图片，但为控制 Token，后续轮次不会再发送原图。\
请在本轮回答里，用下面的固定格式输出一段“图片摘要”，把完成任务所需的全部视觉信息\
（界面结构、可见文字、代码、数值、颜色、布局与位置关系等）写清楚——后续你将只能\
依赖这段摘要，看不到原图：\n\
{DIGEST_BEGIN}\n（在这里写图片摘要）\n{DIGEST_END}\n\
摘要只是中间步骤，绝不能作为本轮回复的结尾：输出完摘要后，你必须继续回答用户本轮\
提出的实际问题（例如提取图片中的链接、总结界面内容等），并在必要时照常调用工具；\
最终以对用户问题的完整答复结束本轮。你可以在输出摘要的同时并行作答或调用工具。\
原图路径由系统记录，确有需要时可用文件工具重新读取原图。"
    )
}

/// Parses the digest body out of model-returned text (between the two sentinels;
/// counts as a hit only if non-empty after trimming).
pub(crate) fn parse_digest(text: &str) -> Option<String> {
    let begin = text.find(DIGEST_BEGIN)? + DIGEST_BEGIN.len();
    let rest = &text[begin..];
    let end = rest.find(DIGEST_END)?;
    let body = rest[..end].trim();
    (!body.is_empty()).then(|| body.to_string())
}

/// Strips all complete digest spans (including both sentinels) from the full text,
/// **for terminal display only**; model-visible text / history must keep the
/// original. Matches non-greedily like `parse_digest`; with BEGIN but no END,
/// only BEGIN is stripped and the body is kept as is, so model narration is not
/// silently swallowed.
pub(crate) fn strip_digest_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        let Some(begin) = rest.find(DIGEST_BEGIN) else {
            out.push_str(rest);
            break;
        };
        let after_begin = &rest[begin + DIGEST_BEGIN.len()..];
        let Some(end) = after_begin.find(DIGEST_END) else {
            out.push_str(&rest[..begin]);
            out.push_str(after_begin);
            break;
        };
        out.push_str(&rest[..begin]);
        rest = &after_begin[end + DIGEST_END.len()..];
    }
    out
}

/// Whether a raw model response is "digest-only": it contains a parseable digest
/// and nothing visible outside the digest blocks. The terminal strips the digest,
/// so ending the turn on such a response looks like the model was interrupted and
/// answered nothing. The orchestrator treats it as intermediate and continues the
/// loop, so the digest replaces the image and the model gets to answer the user's
/// actual question. Callers must pass the full raw response (assistant text plus
/// reasoning text): both are parsed for the digest, and the reasoning text alone
/// may carry the sentinels when the visible text is empty.
pub(crate) fn is_digest_only_response(text: &str) -> bool {
    let trimmed = text.trim();
    !trimmed.is_empty()
        && parse_digest(text).is_some()
        && strip_digest_blocks(text).trim().is_empty()
}

/// Whether a content part is an image: an inline `image_url` part (request
/// form) or a persisted `reference` part with `kind == "image"` (the long-term
/// history form from `build_reference_content`). Treating both as images keeps
/// digest swapping and image detection consistent across the two
/// representations.
fn is_image_part(part: &Value) -> bool {
    part.get("type").and_then(Value::as_str) == Some("image_url")
        || part.get("image_url").is_some()
        || (part.get("type").and_then(Value::as_str) == Some("reference")
            && part.get("kind").and_then(Value::as_str) == Some("image"))
}

/// Whether a message's content (multimodal array) still contains an image part.
pub(crate) fn content_has_image(content: &Value) -> bool {
    content
        .as_array()
        .map(|parts| parts.iter().any(is_image_part))
        .unwrap_or(false)
}

/// Whether a text part is our injected "image handling protocol" instruction.
fn is_instruction_part(part: &Value) -> bool {
    part.get("text")
        .and_then(Value::as_str)
        .map(|t| t.contains(INSTRUCTION_TAG))
        .unwrap_or(false)
}

/// Builds the digest text body that replaces an image: an explicit note that the
/// image has been converted to text + the original image path + the digest itself.
fn build_digest_text(digest: &str, image_paths: &[String]) -> String {
    let paths = if image_paths.is_empty() {
        "（未记录）".to_string()
    } else {
        image_paths.join(", ")
    };
    format!(
        "[图片已转为文字摘要以控制 Token 消耗；后续轮次不再重复发送原图。\
如需精确像素/视觉细节，可用文件工具读取原图。]\n原图路径: {paths}\n图片摘要:\n{digest}"
    )
}

/// Replaces the image parts in a user message's content within the request
/// projection with a text digest.
///
/// - Drops all image_url parts and the injected instruction part;
/// - Keeps the remaining parts (context reminder, the user's actual question, etc.);
/// - Appends one digest text part (including the original image path).
///
/// Only acts when content is an array and really contains images; returns whether
/// a replacement happened. Callers must use this only on request-projection
/// `messages`, never on the canonical `turn_messages`.
pub(crate) fn swap_images_with_digest(
    content: &mut Value,
    digest: &str,
    image_paths: &[String],
) -> bool {
    let Value::Array(parts) = content else {
        return false;
    };
    if !parts.iter().any(is_image_part) {
        return false;
    }
    let mut kept: Vec<Value> = Vec::with_capacity(parts.len());
    for part in parts.drain(..) {
        if is_image_part(&part) || is_instruction_part(&part) {
            continue;
        }
        kept.push(part);
    }
    kept.push(json!({
        "type": "text",
        "text": build_digest_text(digest, image_paths),
    }));
    *content = Value::Array(kept);
    true
}

/// Cross-turn image digest: computes a stable fingerprint (content hash) of an
/// image-bearing user message's content, used as the message key for digest
/// records in history metadata. The same message has identical content when
/// persisted and loaded, so the fingerprint is stable across processes and
/// reliably maps a digest back to its original message.
pub(crate) fn image_message_fingerprint(content: &Value) -> String {
    let mut hasher = FxHasher::default();
    content.to_string().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Cross-turn image digest: takes the fingerprint of the last image-bearing user
/// message's content in `messages` (the one whose digest should be persisted).
pub(crate) fn last_image_user_message_fingerprint(messages: &[Message]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user" && content_has_image(&m.content))
        .map(|m| image_message_fingerprint(&m.content))
}

/// Cross-turn image digest: after loading history, replaces old image messages
/// that have a persisted digest with the digest text.
/// Only the request projection (history just loaded in the prepare phase)
/// changes; the canonical database is untouched; messages without a persisted
/// digest keep their original image (first-send semantics).
pub(crate) fn replace_old_images_with_persisted_digests(
    history_file: &Path,
    messages: &mut [Message],
) -> io::Result<usize> {
    if !messages
        .iter()
        .any(|m| m.role == "user" && content_has_image(&m.content))
    {
        return Ok(0);
    }
    let mut replaced = 0;
    for m in messages
        .iter_mut()
        .filter(|m| m.role == "user" && content_has_image(&m.content))
    {
        let fp = image_message_fingerprint(&m.content);
        if let Some((digest, paths)) =
            crate::ai::history::read_image_digest_sqlite(history_file, &fp)?
        {
            if swap_images_with_digest(&mut m.content, &digest, &paths) {
                replaced += 1;
            }
        }
    }
    Ok(replaced)
}

/// Fallback: sends one dedicated, tool-disabled, one-shot VL request to force
/// obtaining the image digest.
///
/// Called only when the piggyback path failed to parse a digest from the
/// first-round response. Must use a VL model that can see images (the caller
/// passes this turn's actual model), not `control_model_for_aux_tasks` (the aux
/// model may be a text-only intent model that cannot see images).
///
/// Returns the digest text: prefer the body between the sentinels; if the model
/// did not wrap it in the expected format, the whole response is the digest (this
/// is a dedicated "digest only" call). Any failure returns `None`, and the caller
/// decides whether to keep the original image.
pub(crate) async fn describe_image_for_digest(
    app: &App,
    model: &str,
    image_files: &[String],
) -> Option<String> {
    if image_files.is_empty() {
        return None;
    }
    // build_content returns a multimodal array only when the model supports image
    // input; otherwise it degrades to a plain string, meaning this model cannot
    // see images and a fallback digest is moot.
    let user_content = build_content(model, "请按上面的协议为图片生成摘要。", image_files).ok()?;
    if !user_content.is_array() {
        return None;
    }
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(digest_instruction()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: user_content,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    // Tools disabled, non-streaming (consistent with summarize_history_via_model:
    // the two falses are stream / enable_thinking, and all of
    // tools/tool_choice/reasoning are None).
    let mut request_body = build_request_body(
        model, &messages, false, false, None, None, None, None, None, None, None,
    );
    let endpoint = endpoint_for_request_model(app, model);
    let api_key = api_key_for_request_model(app, model);
    let http_body = super::protocol::build_http_body_for_request(model, &endpoint, &mut request_body);

    let send_future = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
        .header("Content-Type", "application/json")
        .body(http_body)
        .send();
    let response = match tokio::time::timeout(
        Duration::from_secs(DIGEST_REQUEST_HEADER_TIMEOUT_SECS),
        send_future,
    )
    .await
    {
        Ok(r) => r.ok()?,
        Err(_) => {
            super::emit_request_diagnostic(format_args!(
                "[image-digest] timeout waiting for response headers, keeping original image"
            ));
            return None;
        }
    };
    if !response.status().is_success() {
        return None;
    }
    let text = match tokio::time::timeout(
        Duration::from_secs(DIGEST_REQUEST_BODY_TIMEOUT_SECS),
        response.text(),
    )
    .await
    {
        Ok(r) => r.ok()?,
        Err(_) => {
            super::emit_request_diagnostic(format_args!(
                "[image-digest] timeout reading response body, keeping original image"
            ));
            return None;
        }
    };
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = extract_router_content(&v)?;
    // Dedicated call: prefer the body between the sentinels; if the model did not
    // wrap them, the whole response is the digest.
    parse_digest(&content).or_else(|| {
        let trimmed = content.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_digest_extracts_body_between_sentinels() {
        let text = format!(
            "some narration\n{DIGEST_BEGIN}\n  界面顶部有一个搜索框  \n{DIGEST_END}\ntrailing"
        );
        assert_eq!(parse_digest(&text).as_deref(), Some("界面顶部有一个搜索框"));
    }

    #[test]
    fn parse_digest_reads_from_reasoning_like_text() {
        // Reproduce a failed session: the assistant's visible text is empty and
        // the digest landed in reasoning.
        let reasoning = format!("我先看图。{DIGEST_BEGIN}\ncode: fn main() {{}}\n{DIGEST_END}");
        assert_eq!(
            parse_digest(&reasoning).as_deref(),
            Some("code: fn main() {}")
        );
    }

    #[test]
    fn parse_digest_none_when_missing_or_empty() {
        assert_eq!(parse_digest("no sentinels here"), None);
        // Begin sentinel only, no end sentinel -> None.
        assert_eq!(parse_digest(&format!("{DIGEST_BEGIN} dangling")), None);
        // Empty between the sentinels -> None.
        assert_eq!(
            parse_digest(&format!("{DIGEST_BEGIN}   {DIGEST_END}")),
            None
        );
    }

    #[test]
    fn strip_digest_blocks_removes_complete_regions_only() {
        // The complete span is stripped (including sentinels); surrounding
        // narration is kept.
        assert_eq!(
            strip_digest_blocks(&format!("前言{DIGEST_BEGIN} 摘要内容 {DIGEST_END}后语")),
            "前言后语"
        );
        // No span -> returned unchanged.
        let plain = "普通文本";
        assert_eq!(strip_digest_blocks(plain), plain);
        // BEGIN without END -> only the control sentinel is hidden; the body is
        // still kept.
        let dangling = format!("{DIGEST_BEGIN} 未闭合");
        assert_eq!(strip_digest_blocks(&dangling), " 未闭合");
        // All spans are stripped.
        assert_eq!(
            strip_digest_blocks(&format!(
                "a{DIGEST_BEGIN}1{DIGEST_END}b{DIGEST_BEGIN}2{DIGEST_END}c"
            )),
            "abc"
        );
        // Consistent with parse_digest semantics: non-greedy.
        assert_eq!(
            strip_digest_blocks(&format!(
                "{DIGEST_BEGIN}1{DIGEST_END}mid{DIGEST_BEGIN}2{DIGEST_END}"
            )),
            "mid"
        );
    }

    #[test]
    fn digest_only_detection() {
        // A response that is entirely the digest block counts as digest-only.
        assert!(is_digest_only_response(&format!(
            "{DIGEST_BEGIN}\n摘要内容\n{DIGEST_END}"
        )));
        // Trailing whitespace outside an otherwise complete digest still counts
        // as digest-only.
        assert!(is_digest_only_response(&format!(
            "{DIGEST_BEGIN}摘要{DIGEST_END}\n\n"
        )));
        // Narration outside the digest means the user sees real content -> not
        // digest-only.
        assert!(!is_digest_only_response(&format!(
            "我来看看。{DIGEST_BEGIN}摘要{DIGEST_END}答案是 X"
        )));
        // No sentinels at all -> not digest-only.
        assert!(!is_digest_only_response("普通回复"));
        // BEGIN without END keeps the body visible -> not digest-only.
        assert!(!is_digest_only_response(&format!("{DIGEST_BEGIN} 未闭合")));
        // Empty / blank input is not a digest-only response.
        assert!(!is_digest_only_response(""));
        assert!(!is_digest_only_response("   "));
    }

    #[test]
    fn content_has_image_detects_both_shapes() {
        let with_type = Value::Array(vec![json!({
            "type": "image_url",
            "image_url": { "url": "data:image/png;base64,AAAA" }
        })]);
        assert!(content_has_image(&with_type));

        let text_only = Value::Array(vec![json!({ "type": "text", "text": "hi" })]);
        assert!(!content_has_image(&text_only));

        // Plain-string content contains no images.
        assert!(!content_has_image(&Value::String("hi".to_string())));
    }

    #[test]
    fn swap_replaces_image_keeps_question_drops_instruction() {
        let mut content = Value::Array(vec![
            json!({ "type": "text", "text": "上下文提醒" }),
            json!({
                "type": "image_url",
                "image_url": { "url": "data:image/png;base64,AAAA" }
            }),
            json!({ "type": "text", "text": "帮我看看这个界面" }),
            json!({ "type": "text", "text": digest_instruction() }),
        ]);
        let paths = vec!["/tmp/shot.png".to_string()];
        let swapped = swap_images_with_digest(&mut content, "顶部是标题栏", &paths);
        assert!(swapped);

        let parts = content.as_array().unwrap();
        // Both the image part and the instruction part are removed.
        assert!(!parts.iter().any(is_image_part));
        assert!(!parts.iter().any(is_instruction_part));
        // The original context reminder and the user's question body are kept.
        assert!(
            parts
                .iter()
                .any(|p| p.get("text").and_then(Value::as_str) == Some("上下文提醒"))
        );
        assert!(
            parts
                .iter()
                .any(|p| p.get("text").and_then(Value::as_str) == Some("帮我看看这个界面"))
        );
        // The appended digest text contains the original image path and the
        // digest body.
        let digest_part = parts
            .last()
            .and_then(|p| p.get("text"))
            .and_then(Value::as_str)
            .unwrap();
        assert!(digest_part.contains("/tmp/shot.png"));
        assert!(digest_part.contains("顶部是标题栏"));
    }

    #[test]
    fn swap_noop_when_no_image() {
        let mut content = Value::Array(vec![json!({ "type": "text", "text": "只有文字" })]);
        assert!(!swap_images_with_digest(&mut content, "x", &[]));
        // Plain strings are likewise untouched.
        let mut plain = Value::String("hello".to_string());
        assert!(!swap_images_with_digest(&mut plain, "x", &[]));
        assert_eq!(plain, Value::String("hello".to_string()));
    }

    #[test]
    fn swap_is_idempotent_second_call_noop() {
        let mut content = Value::Array(vec![
            json!({
                "type": "image_url",
                "image_url": { "url": "data:image/png;base64,AAAA" }
            }),
            json!({ "type": "text", "text": "问题" }),
        ]);
        assert!(swap_images_with_digest(&mut content, "摘要", &[]));
        // No images left; the second replacement should be a no-op (avoid
        // appending the digest twice).
        assert!(!swap_images_with_digest(&mut content, "摘要", &[]));
    }

    fn test_msg(role: &str, content: Value) -> Message {
        Message {
            role: role.to_string(),
            content,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn image_content(url: &str, question: &str) -> Value {
        Value::Array(vec![
            json!({ "type": "image_url", "image_url": { "url": url } }),
            json!({ "type": "text", "text": question }),
        ])
    }

    #[test]
    fn image_message_fingerprint_stable_and_distinct() {
        let a = image_content("data:image/png;base64,AAAA", "问题");
        let b = image_content("data:image/png;base64,BBBB", "问题");
        // The fingerprint is stable for identical content (reproducible across
        // processes; used as the history metadata key).
        assert_eq!(image_message_fingerprint(&a), image_message_fingerprint(&a));
        // Different image content yields different fingerprints, avoiding digest
        // mismatches.
        assert_ne!(image_message_fingerprint(&a), image_message_fingerprint(&b));
    }

    #[test]
    fn last_image_user_message_fingerprint_targets_last_image_user() {
        let first = image_content("data:image/png;base64,AAAA", "q1");
        let second = image_content("data:image/png;base64,BBBB", "q2");
        let msgs = vec![
            test_msg("user", first.clone()),
            test_msg("assistant", Value::String("a1".into())),
            test_msg("user", second.clone()),
        ];
        // Takes the last image-bearing user message (the image of the current turn).
        assert_eq!(
            last_image_user_message_fingerprint(&msgs).as_deref(),
            Some(image_message_fingerprint(&second).as_str())
        );
        // No images -> None.
        assert_eq!(
            last_image_user_message_fingerprint(&[test_msg("user", Value::String("无图".into()))]),
            None
        );
    }

    #[test]
    fn persisted_digest_replaces_old_image_on_load() {
        let db = std::env::temp_dir().join(format!(
            "image_digest_roundtrip_{}.sqlite",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&db);
        let content = image_content("data:image/png;base64,AAAA", "上一轮问题");
        let mut msgs = vec![
            test_msg("assistant", Value::String("上一轮回复".into())),
            test_msg("user", content.clone()),
            test_msg("user", Value::String("本轮新问题".into())),
        ];
        let fp = image_message_fingerprint(&content);
        crate::ai::history::upsert_image_digest_sqlite(
            &db,
            &fp,
            "顶部是标题栏，左侧是导航",
            &["/tmp/shot.png".to_string()],
        )
        .expect("upsert digest");
        // Replacement on load: old images with a persisted digest are swapped for
        // the digest text.
        let replaced =
            replace_old_images_with_persisted_digests(&db, &mut msgs).expect("replace");
        assert_eq!(replaced, 1);
        assert!(!content_has_image(&msgs[1].content));
        let joined = msgs[1].content.to_string();
        assert!(joined.contains("顶部是标题栏，左侧是导航"));
        assert!(joined.contains("/tmp/shot.png"));
        // Plain user messages without images are unaffected.
        assert_eq!(msgs[2].content, Value::String("本轮新问题".into()));

        // Image messages without a persisted digest keep the original image
        // (first-send semantics).
        let mut fresh = vec![test_msg(
            "user",
            image_content("data:image/png;base64,BBBB", "另一张"),
        )];
        let replaced_none =
            replace_old_images_with_persisted_digests(&db, &mut fresh).expect("replace");
        assert_eq!(replaced_none, 0);
        assert!(content_has_image(&fresh[0].content));

        let _ = std::fs::remove_file(&db);
        let _ = std::fs::remove_file(db.with_file_name(format!(
            ".{}.state.lock",
            db.file_name().unwrap().to_string_lossy()
        )));
    }
}
