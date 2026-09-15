//! Request body construction and token budget estimation.
//!
//! Construction logic extracted from request/mod.rs:
//! - Image/text content construction (multimodal support)
//! - Request body assembly (model/messages/tools/thinking/stream/max_tokens, etc.)
//! - Prompt token estimation and max_tokens clamping (keep prompt + output within the window)

use std::{
    fs,
    path::{Component, Path, PathBuf},
};

use base64::Engine as _;
use serde_json::{Value, json};

use super::RequestBody;
use super::reasoning::resolve_reasoning_wire_controls;
use crate::ai::{
    files,
    history::Message,
    models,
    provider::{adapter_for, compatible_wire_shapes},
    request_protocol::RequestProtocolDialect,
};

/// Builds message content: returns a string for text-only models or no images; otherwise,
/// returns an `[{image_url}, {text}]` array for multimodal models, inlining images as base64 data URIs.
pub(crate) fn build_content(
    model: &str,
    question: &str,
    image_files: &[String],
) -> Result<Value, Box<dyn std::error::Error>> {
    if !models::supports_image_input(model) || image_files.is_empty() {
        return Ok(Value::String(question.to_string()));
    }

    let mut parts = Vec::new();
    for file in image_files {
        let bytes = fs::read(file)?;
        let mime = files::image_mime_type(file);
        let image = base64::engine::general_purpose::STANDARD.encode(bytes);
        parts.push(json!({
            "type": "image_url",
            "image_url": {
                "url": format!("data:{mime};base64,{image}")
            },
        }));
    }
    parts.push(json!({
        "type": "text",
        "text": question,
    }));
    Ok(Value::Array(parts))
}

/// Builds the PERSISTED form of a user message's content, keeping the write-time
/// provenance boundary between the user's own words and referenced/attached
/// artifacts: each image becomes a `reference` part (kind=image, name + path)
/// instead of inline base64, and each text-extractable attachment (text file /
/// PDF) becomes a `reference` part (kind=file, name + path) instead of inline
/// content. Long-term history keeps this form so any later reader (another
/// session debugging this one, /history rendering, compression summaries) can
/// tell real user content apart from references instead of mistaking inline
/// attachment data for the user's own words. `materialize_references` restores
/// the inline form for requests.
pub(crate) fn build_reference_content(
    model: &str,
    question: &str,
    image_files: &[String],
    attachments_text: &str,
    attachment_assets_dir: &Path,
) -> Result<Value, Box<dyn std::error::Error>> {
    let supports_images = models::supports_image_input(model);
    if attachments_text.trim().is_empty() && (!supports_images || image_files.is_empty()) {
        return Ok(Value::String(question.to_string()));
    }

    let mut parts = Vec::new();
    // Non-VL models never receive image parts (their image context arrives via
    // OCR text in the question), so their persisted form skips image references
    // but still keeps text-file references.
    if supports_images {
        for file in image_files {
            let name = std::path::Path::new(file)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| file.clone());
            let asset = attachment_asset_key(file, attachment_assets_dir)?;
            parts.push(json!({
                "type": "reference",
                "kind": "image",
                "name": name,
                "asset": asset,
            }));
        }
    }
    if !attachments_text.trim().is_empty() {
        parts.push(json!({
            "type": "reference",
            "kind": "file",
            "name": "attached files",
            "text": attachments_text,
        }));
    }
    parts.push(json!({
        "type": "text",
        "text": question,
    }));
    Ok(Value::Array(parts))
}

/// Converts a captured image's absolute path into an opaque key relative to the
/// current session assets directory. Persisting this key, rather than a source
/// path, keeps references stable across working directories and session forks.
fn attachment_asset_key(
    path: &str,
    attachment_assets_dir: &Path,
) -> Result<String, Box<dyn std::error::Error>> {
    let root = attachment_assets_dir.canonicalize()?;
    let path = Path::new(path).canonicalize()?;
    let relative = path.strip_prefix(&root).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "image attachment must be captured inside the session assets directory",
        )
    })?;
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "image attachment asset key is invalid",
        )
        .into());
    }
    Ok(relative.to_string_lossy().into_owned())
}

/// Materializes persisted reference parts for a request. Image references read
/// an immutable session asset, and file references replay their write-time text
/// snapshot. Legacy path references deliberately become a marker instead of
/// rereading a mutable source file. Returns whether any part changed.
pub(crate) fn materialize_references(
    content: &mut Value,
    attachment_assets_dir: Option<&Path>,
) -> bool {
    let Value::Array(parts) = content else {
        return false;
    };
    let mut changed = false;
    let mut out: Vec<Value> = Vec::with_capacity(parts.len());
    for part in parts.drain(..) {
        let is_ref = part.get("type").and_then(Value::as_str) == Some("reference");
        if !is_ref {
            out.push(part);
            continue;
        }
        changed = true;
        let kind = part.get("kind").and_then(Value::as_str).unwrap_or("");
        let name = part
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("attachment")
            .to_string();
        match kind {
            "image" => {
                let asset = part.get("asset").and_then(Value::as_str);
                match resolve_attachment_asset(attachment_assets_dir, asset).and_then(fs::read) {
                    Ok(bytes) => {
                        let mime = files::image_mime_type(&name);
                        let image = base64::engine::general_purpose::STANDARD.encode(bytes);
                        out.push(json!({
                            "type": "image_url",
                            "image_url": {
                                "url": format!("data:{mime};base64,{image}")
                            },
                        }));
                    }
                    Err(_) => {
                        out.push(json!({
                            "type": "text",
                            "text": format!("[引用图片快照不可用: {name}]"),
                        }));
                    }
                }
            }
            "file" => {
                let text = part
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("[引用文件快照不可用: attached files]");
                out.push(json!({ "type": "text", "text": text }));
            }
            // Forward-compat guard: only image/file exist today (see
            // build_reference_content), but a reference of any future kind must
            // never be forwarded verbatim — OpenAI-style content parts reject an
            // unknown `type`, which would fail the whole request.
            _ => out.push(json!({
                "type": "text",
                "text": format!("[引用内容类型未知: {kind} ({name})]"),
            })),
        }
    }
    *content = Value::Array(out);
    changed
}

/// Resolves an opaque session-relative asset key without allowing references to
/// escape the active session's assets directory.
fn resolve_attachment_asset(root: Option<&Path>, asset: Option<&str>) -> std::io::Result<PathBuf> {
    let root = root.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "attachment assets are unavailable",
        )
    })?;
    let asset = asset.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "attachment asset key is missing",
        )
    })?;
    let relative = Path::new(asset);
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "attachment asset key is invalid",
        ));
    }
    let root = root.canonicalize()?;
    let asset = root.join(relative).canonicalize()?;
    if asset.strip_prefix(&root).is_err() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "attachment asset escapes the session assets directory",
        ));
    }
    Ok(asset)
}

/// Conservative character-to-token conversion for mixed Chinese, English, and code: about 2 chars/token.
/// Intentionally overestimates prompt usage, preferring to clamp the output cap early rather
/// than let the prompt and output together exceed the context window.
const CHARS_PER_TOKEN_CONSERVATIVE: usize = 2;
/// Minimum output tokens retained after clamping: leaves room for visible output even when
/// the prompt nears the window, avoiding immediate truncation from tiny or zero max_tokens.
pub(super) const MIN_OUTPUT_TOKENS_FLOOR: u32 = 1_024;
/// Safety margin for hidden provider overhead (template tokens, role separators, reasoning reserves, etc.).
const CONTEXT_WINDOW_SAFETY_MARGIN_TOKENS: usize = 2_048;

/// Conservatively estimates prompt tokens in messages, tending to overestimate. Without
/// server usage feedback, approximates from the character count at about 2 chars/token.
fn estimate_prompt_tokens(messages: &[Message]) -> usize {
    let chars: usize = messages
        .iter()
        .map(super::super::history::message_billable_chars)
        .sum();
    chars.div_ceil(CHARS_PER_TOKEN_CONSERVATIVE)
}

/// Character count of the compact JSON text `serde_json::to_string` produces
/// for `value`, computed without materializing that text. Every branch mirrors
/// serde_json's compact formatter exactly:
/// - strings: 2 quotes + per-char contributions (`\"`, `\\`, `\b`, `\t`, `\n`,
///   `\f`, `\r` are two chars each, other control chars become `\u00xx`
///   (6 chars), everything else is emitted verbatim = 1 char per scalar);
/// - containers: delimiters plus single-char commas/colons;
/// - numbers: delegated to `serde_json::to_string` itself (tiny transient
///   allocation per numeric leaf only), so integer/float formatting — the one
///   formatter whose output is non-trivial to replicate — is identical by
///   construction regardless of serde_json feature flags.
/// The token-estimation caller only consumes this count, so replacing the old
/// full-schema `to_string` keeps the estimated number provably identical
/// (guarded by the parity tests at the bottom of this file) while avoiding a
/// potentially multi-megabyte string allocation per request.
fn compact_json_char_len(value: &Value) -> usize {
    match value {
        Value::Null => 4,
        Value::Bool(true) => 4,
        Value::Bool(false) => 5,
        Value::Number(_) => serde_json::to_string(value)
            .map(|encoded| encoded.chars().count())
            .unwrap_or(0),
        Value::String(text) => json_string_char_len(text),
        Value::Array(items) => {
            2 + items.iter().map(compact_json_char_len).sum::<usize>()
                + items.len().saturating_sub(1)
        }
        Value::Object(map) => {
            2 + map
                .iter()
                .map(|(key, item)| json_string_char_len(key) + 1 + compact_json_char_len(item))
                .sum::<usize>()
                + map.len().saturating_sub(1)
        }
    }
}

fn json_string_char_len(text: &str) -> usize {
    2 + text
        .chars()
        .map(|ch| match ch {
            '"' | '\\' | '\u{8}' | '\t' | '\n' | '\u{c}' | '\r' => 2,
            ch if (ch as u32) < 0x20 => 6,
            _ => 1,
        })
        .sum::<usize>()
}

/// Estimates prompt tokens for tool schemas. Definitions (name/description/JSON Schema) are
/// sent and counted on every request and can be large with many tools/MCP. Uses the same
/// conservative conversion on serialized characters. `None` / empty tool sets contribute 0.
fn estimate_tools_tokens(tools: Option<&Value>) -> usize {
    let Some(tools) = tools else {
        return 0;
    };
    compact_json_char_len(tools).div_ceil(CHARS_PER_TOKEN_CONSERVATIVE)
}

/// Estimates input tokens for this request: messages + tool schemas.
/// The request transport's TPM preflight gate and max_tokens clamp share this path so
/// context-window and rate budgets do not diverge because of different estimates.
pub(super) fn estimate_request_prompt_tokens(messages: &[Message], tools: Option<&Value>) -> usize {
    estimate_prompt_tokens(messages) + estimate_tools_tokens(tools)
}

/// Clamp the output cap to the estimated remaining context window. Any supplied
/// prompt count must already describe this request, not an earlier response.
/// The output floor is a last-resort allowance, not proof that input fits.
pub(crate) fn clamp_max_tokens_for_prompt(
    model: &str,
    messages: &[Message],
    tools: Option<&Value>,
    model_max: u32,
    current_prompt_tokens: Option<u64>,
) -> u32 {
    clamp_with_estimated_prompt(
        model,
        current_prompt_tokens
            .map(|tokens| usize::try_from(tokens).unwrap_or(usize::MAX))
            .unwrap_or_else(|| estimate_request_prompt_tokens(messages, tools)),
        model_max,
    )
}

/// Reuse a current-request estimate without walking the history again.
pub(super) fn clamp_with_estimated_prompt(model: &str, est_prompt: usize, model_max: u32) -> u32 {
    let window = models::context_window_tokens(model);
    let remaining = window
        .saturating_sub(est_prompt)
        .saturating_sub(CONTEXT_WINDOW_SAFETY_MARGIN_TOKENS);
    let remaining = u32::try_from(remaining).unwrap_or(u32::MAX);
    model_max.min(remaining).max(MIN_OUTPUT_TOKENS_FLOOR)
}

/// Assembles the HTTP request body (`RequestBody`), adding thinking / reasoning /
/// search / stream_options / max_tokens per model capabilities and clamping max_tokens to fit the window.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_request_body<'a>(
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
    enable_thinking: bool,
    enable_search: Option<bool>,
    tools: Option<Value>,
    tool_choice: Option<Value>,
    reasoning_effort: Option<&'a str>,
    max_tokens_override: Option<u32>,
    current_prompt_tokens: Option<u64>,
    reasoning_items: Option<&'a rustc_hash::FxHashMap<String, Vec<Value>>>,
) -> RequestBody<'a> {
    let adapter_kind = models::model_adapter(model);
    let endpoint = models::endpoint_for_model(model, "");
    let adapter = adapter_for(adapter_kind, &endpoint);
    let request_model = models::request_model_name(model);
    let (thinking, reasoning_effort, reasoning) =
        resolve_reasoning_wire_controls(model, &endpoint, enable_thinking, reasoning_effort);
    // Responses maps search to the built-in web_search tool; for Chat Completions,
    // the provider dialect still decides whether to retain `enable_search`.
    let request_protocol = models::request_protocol_dialect(model, &endpoint);
    let enable_search = if request_protocol == RequestProtocolDialect::Responses {
        enable_search
    } else if adapter_kind == crate::ai::provider::ApiProvider::Compatible {
        let (es, _, _) = compatible_wire_shapes(&endpoint, enable_search, None);
        es
    } else {
        adapter.enable_search_field(enable_search)
    };
    // Explicitly request streaming usage: some adapters (DashScope compatible-mode) omit
    // usage by default and require stream_options.include_usage for token accounting.
    let stream_options = stream.then(|| json!({ "include_usage": true }));
    // Send max_tokens only when the model declares max_output_tokens; clamp to the remaining
    // context window so prompt + requested output cap cannot overflow it (the cause of GLM's
    // repeated truncation/retry loop with long contexts). Models without max_output_tokens
    // still omit this field, preserving their wire behavior.
    // Compute the raw estimate once. Transport may calibrate it against a
    // compatible request snapshot, but keeps this uncalibrated baseline for
    // the next response's growth calculation.
    let est_prompt = estimate_request_prompt_tokens(messages, tools.as_ref());
    let max_tokens = models::max_output_tokens(model).map(|model_max| {
        let current = current_prompt_tokens
            .map(|tokens| usize::try_from(tokens).unwrap_or(usize::MAX))
            .unwrap_or(est_prompt);
        clamp_with_estimated_prompt(model, current, model_max)
    });
    // Adapt to zero-output truncation: after completion=0 + finish_reason=length, the
    // orchestrator lowers max_tokens_override. Replace the clamp result with that value
    // so the next request avoids the server's empty-response rejection of oversized max_tokens.
    let max_tokens = match (max_tokens, max_tokens_override) {
        (Some(_), Some(override_val)) => Some(override_val),
        (mt, _) => mt,
    };
    RequestBody {
        model: request_model,
        messages,
        stream,
        thinking,
        enable_search,
        tools,
        tool_choice,
        reasoning_effort,
        reasoning,
        stream_options,
        max_tokens,
        reasoning_items,
        reasoning_encrypted_replay: models::reasoning_encrypted_replay_enabled(model),
        estimated_prompt_tokens: est_prompt,
    }
}

#[cfg(test)]
mod compact_json_char_len_tests {
    use super::{compact_json_char_len, json_string_char_len};
    use serde_json::{Map, Value, json};

    /// Ground truth: the walker must agree with `serde_json::to_string(..)`
    /// char count for every value. This is the parity guard that lets
    /// `estimate_tools_tokens` skip materializing the encoded schema string.
    fn assert_parity(value: &Value) {
        let expected = serde_json::to_string(value)
            .expect("test value must serialize")
            .chars()
            .count();
        assert_eq!(
            compact_json_char_len(value),
            expected,
            "char-len parity failed for {value}"
        );
    }

    #[test]
    fn scalars_and_escaping_match_compact_formatter() {
        assert_parity(&Value::Null);
        assert_parity(&Value::Bool(true));
        assert_parity(&Value::Bool(false));

        // Every ASCII control char plus quote/backslash escaping; serde_json
        // escapes controls < 0x20 and emits everything else verbatim.
        for byte in 0u32..0x20 {
            let ch = char::from_u32(byte).expect("ASCII control char");
            assert_eq!(
                json_string_char_len(&ch.to_string()),
                2 + {
                    match ch {
                        '\u{8}' | '\t' | '\n' | '\u{c}' | '\r' => 2,
                        _ => 6,
                    }
                }
            );
            assert_parity(&json!({ "k": ch.to_string() }));
        }
        for text in ["\"", "\\", "a\"b\\c", "\u{7f}", "中文", "🙂", "line\nbreak"] {
            assert_parity(&json!({ text: text }));
        }
    }

    #[test]
    fn numbers_match_serde_json_formatting_including_floats() {
        for value in [
            json!(0),
            json!(-1),
            json!(i64::MIN),
            json!(i64::MAX),
            json!(u64::MAX),
            json!(0.0),
            json!(-0.0),
            json!(0.3),
            json!(1.5),
            json!(1e100),
            json!(5e-324),
            json!(f64::MIN),
            json!(f64::MAX),
            json!(f64::INFINITY),
            json!(f64::NEG_INFINITY),
            json!(f64::NAN),
            Value::Number(serde_json::Number::from_f64(9007199254740993.0).unwrap()),
        ] {
            assert_parity(&value);
        }
    }

    #[test]
    fn containers_and_empty_shapes_match() {
        for value in [
            json!([]),
            json!([[]]),
            json!({}),
            json!([{}, {}]),
            json!({"a": []}),
            json!({"a": {"b": [1, [2, {"c": null}]]}}),
            json!([true, false, null, -2.5, "s", {"k": [1]}]),
        ] {
            assert_parity(&value);
        }
    }

    #[test]
    fn randomized_values_stay_in_parity_with_serde_json() {
        // Deterministic xorshift PRNG so failures are reproducible.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        fn build(rng: &mut impl FnMut() -> u64, depth: u32) -> Value {
            let pick = (rng()) % 100;
            if depth == 0 || pick < 30 {
                return match (rng()) % 6 {
                    0 => Value::Null,
                    1 => Value::Bool(rng() % 2 == 0),
                    2 => Value::from((rng() as i64).wrapping_mul(1_000_003)),
                    3 => Value::from(f64::from_bits(rng())),
                    4 => Value::from(rng()),
                    _ => {
                        // Sample control chars / quotes / non-ASCII into strings.
                        let unit = (rng() % 0x2FFF) as u32;
                        let ch = match char::from_u32(unit) {
                            Some(ch) if ch != '\u{0}' => ch,
                            _ => 'x',
                        };
                        Value::String(ch.to_string())
                    }
                };
            }
            if pick < 65 {
                let len = (rng() % 4) as usize;
                return Value::Array((0..len).map(|_| build(rng, depth - 1)).collect());
            }
            let len = (rng() % 4) as usize;
            let mut map = Map::new();
            for i in 0..len {
                let key_char = char::from_u32(0x20 + (rng() % 0x40) as u32).unwrap_or('k');
                map.insert(format!("k{key_char}{i}"), build(rng, depth - 1));
            }
            Value::Object(map)
        }

        for _ in 0..2_000 {
            let value = build(&mut next, 5);
            assert_parity(&value);
        }
    }
}
