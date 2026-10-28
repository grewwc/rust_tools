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
use super::reasoning::resolve_reasoning_wire_controls_from_parts;
use crate::ai::{
    files,
    history::{
        Message,
        compress::{self, BillableCounts},
    },
    model_names::ReasoningEffortWire,
    models,
    provider::{ApiProvider, adapter_for, compatible_wire_shapes},
    request_protocol::RequestProtocolDialect,
    types::ToolDefinition,
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
pub(super) fn resolve_attachment_asset(
    root: Option<&Path>,
    asset: Option<&str>,
) -> std::io::Result<PathBuf> {
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

/// Minimum output tokens retained after clamping: leaves room for visible output even when
/// the prompt nears the window, avoiding immediate truncation from tiny or zero max_tokens.
pub(super) const MIN_OUTPUT_TOKENS_FLOOR: u32 = 1_024;
/// Safety margin for hidden provider overhead (template tokens, role separators, reasoning reserves, etc.).
pub(super) const CONTEXT_WINDOW_SAFETY_MARGIN_TOKENS: usize = 2_048;

/// Prompt-token estimate of messages over the same fields the char budget
/// bills, weighted by script rather than a flat chars-per-token ratio (see
/// [`BillableCounts::tokens`]). Counts are summed before rounding so many
/// short messages do not accumulate per-message ceil error.
pub(super) fn message_prompt_counts(messages: &[Message]) -> BillableCounts {
    messages
        .iter()
        .map(compress::message_billable_counts)
        .sum::<BillableCounts>()
}

fn estimate_prompt_tokens(messages: &[Message]) -> usize {
    message_prompt_counts(messages).tokens()
}

/// Encrypted reasoning replay spliced in front of Responses function calls.
/// Only items keyed by a projected message's first tool call reach the wire
/// (see `protocol::responses_input`); entries for compressed-away rounds are
/// not billed. `encrypted_content` is base64 of an opaque payload: following
/// codex (`context_manager/history.rs`), decoded bytes ~= 3/4 of the encoded
/// length, minus a fixed envelope, at ~4 bytes per token. Every other field
/// (summary text, ids, type) is charged as ordinary compact JSON.
fn estimate_reasoning_items_tokens(
    messages: &[Message],
    items: Option<&rustc_hash::FxHashMap<String, Vec<Value>>>,
) -> usize {
    estimate_reasoning_items_projected(messages, items, 4, BillableCounts::tokens)
}

/// Pessimistic twin of [`estimate_reasoning_items_tokens`]: opaque encrypted
/// bytes at 2 chars/token, text counts via [`BillableCounts::tokens_upper`].
fn estimate_reasoning_items_tokens_upper(
    messages: &[Message],
    items: Option<&rustc_hash::FxHashMap<String, Vec<Value>>>,
) -> usize {
    estimate_reasoning_items_projected(messages, items, 2, BillableCounts::tokens_upper)
}

/// Shared replay walk behind both reasoning estimates. The lower estimate
/// passes the historical divisor/projection, so its arithmetic is unchanged.
fn estimate_reasoning_items_projected(
    messages: &[Message],
    items: Option<&rustc_hash::FxHashMap<String, Vec<Value>>>,
    encrypted_divisor: usize,
    project: fn(BillableCounts) -> usize,
) -> usize {
    const ENCRYPTED_ENVELOPE_BYTES: usize = 650;
    let Some(items) = items.filter(|items| !items.is_empty()) else {
        return 0;
    };
    let mut counts = BillableCounts::default();
    let mut encrypted_tokens = 0usize;
    let replayed = messages
        .iter()
        .filter_map(|message| message.tool_calls.as_ref()?.first())
        .filter_map(|call| items.get(&call.id))
        .flatten();
    for item in replayed {
        let Value::Object(map) = item else {
            counts += compact_json_counts(item);
            continue;
        };
        counts += BillableCounts::ascii(2 + map.len().saturating_sub(1));
        for (key, value) in map {
            counts += BillableCounts::json_string(key) + BillableCounts::ascii(1);
            match (key.as_str(), value) {
                ("encrypted_content", Value::String(encoded)) => {
                    let decoded = encoded.len().saturating_mul(3) / 4;
                    encrypted_tokens = encrypted_tokens.saturating_add(
                        decoded
                            .saturating_sub(ENCRYPTED_ENVELOPE_BYTES)
                            .div_ceil(encrypted_divisor),
                    );
                }
                _ => counts += compact_json_counts(value),
            }
        }
    }
    encrypted_tokens.saturating_add(project(counts))
}

/// Script mix of the compact JSON text `serde_json::to_string` produces for
/// `value`, computed without materializing that text. Every branch mirrors
/// serde_json's compact formatter exactly:
/// - strings: see [`BillableCounts::json_string`];
/// - containers: delimiters plus single-char commas/colons;
/// - numbers: delegated to `serde_json::to_string` itself (tiny transient
///   allocation per numeric leaf only), so integer/float formatting — the one
///   formatter whose output is non-trivial to replicate — is identical by
///   construction regardless of serde_json feature flags.
/// `chars()` of the result is pinned against serde by the parity tests at the
/// bottom of this file, which therefore also pin what `tokens()` bills, while
/// avoiding a potentially multi-megabyte string allocation per request.
fn compact_json_counts(value: &Value) -> BillableCounts {
    match value {
        Value::Null | Value::Bool(true) => BillableCounts::ascii(4),
        Value::Bool(false) => BillableCounts::ascii(5),
        Value::Number(_) => serde_json::to_string(value)
            .map(|encoded| BillableCounts::ascii(encoded.len()))
            .unwrap_or_default(),
        Value::String(text) => BillableCounts::json_string(text),
        Value::Array(items) => {
            items.iter().map(compact_json_counts).sum::<BillableCounts>()
                + BillableCounts::ascii(2 + items.len().saturating_sub(1))
        }
        Value::Object(map) => {
            map.iter()
                .map(|(key, item)| {
                    BillableCounts::json_string(key)
                        + BillableCounts::ascii(1)
                        + compact_json_counts(item)
                })
                .sum::<BillableCounts>()
                + BillableCounts::ascii(2 + map.len().saturating_sub(1))
        }
    }
}

#[cfg(test)]
fn compact_json_char_len(value: &Value) -> usize {
    compact_json_counts(value).chars()
}

#[cfg(test)]
fn json_string_char_len(text: &str) -> usize {
    BillableCounts::json_string(text).chars()
}

/// Estimates prompt tokens for tool schemas. Definitions (name/description/JSON Schema) are
/// sent and counted on every request and can be large with many tools/MCP. Uses the same
/// weights on the serialized characters. `None` / empty tool sets contribute 0.
fn estimate_tools_tokens(tools: Option<&Value>) -> usize {
    tools.map_or(0, |tools| compact_json_counts(tools).tokens())
}

/// Pessimistic twin of [`estimate_tools_tokens`]: identical counts projected
/// with [`BillableCounts::tokens_upper`]. Upper-bound input only; budgets and
/// clamps keep the lower estimate.
fn estimate_tools_tokens_upper(tools: Option<&Value>) -> usize {
    tools.map_or(0, |tools| compact_json_counts(tools).tokens_upper())
}

/// Compact-JSON counts of `serde_json::to_value(defs)` computed directly from
/// the definitions, without building the intermediate `Value`. Mirrors serde's
/// struct layout field-for-field (`preserve_order` keeps declaration order):
/// each tool is `{"type":..,"function":{"name":..,"description":..,"parameters":..}}`.
/// The parity test below pins it against the real serialization so the preview
/// path can skip the per-request deep clone of the whole tool schema.
#[cfg(test)]
pub(super) fn tools_definitions_compact_json_char_len(defs: &[ToolDefinition]) -> usize {
    tools_definitions_compact_json_counts(defs).chars()
}

fn tools_definitions_compact_json_counts(defs: &[ToolDefinition]) -> BillableCounts {
    let key = |name: &str| BillableCounts::json_string(name) + BillableCounts::ascii(1);
    defs.iter()
        .map(|def| {
            // `{"name":..,"description":..,"parameters":..}`: braces + 2 commas.
            let function = BillableCounts::ascii(4)
                + key("name")
                + BillableCounts::json_string(&def.function.name)
                + key("description")
                + BillableCounts::json_string(&def.function.description)
                + key("parameters")
                + compact_json_counts(&def.function.parameters);
            // `{"type":..,"function":..}`: braces + 1 comma.
            BillableCounts::ascii(3)
                + key("type")
                + BillableCounts::json_string(&def.tool_type)
                + key("function")
                + function
        })
        .sum::<BillableCounts>()
        + BillableCounts::ascii(2 + defs.len().saturating_sub(1))
}

/// Token estimate for tool schemas without serializing them: `Some` carries
/// the definitions the wire body would send (non-empty by construction, matching
/// `agent_tools_for_request` which yields `None` for empty tool sets), `None`
/// contributes 0 — mirroring `estimate_tools_tokens`' `None`-vs-`Some` split.
pub(super) fn estimate_tools_tokens_from_definitions(defs: Option<&[ToolDefinition]>) -> usize {
    defs.map_or(0, |defs| tools_definitions_compact_json_counts(defs).tokens())
}

/// Pessimistic twin of [`estimate_tools_tokens_from_definitions`] for the
/// preview path. Same counts, [`BillableCounts::tokens_upper`] projection.
pub(super) fn estimate_tools_tokens_upper_from_definitions(
    defs: Option<&[ToolDefinition]>,
) -> usize {
    defs.map_or(0, |defs| {
        tools_definitions_compact_json_counts(defs).tokens_upper()
    })
}

/// Preview-path prompt estimate: messages, tool schemas, and reasoning replay,
/// none materialized as JSON. Matches `estimate_request_prompt_tokens` on the
/// equivalent wire body (parity-tested) while skipping the schema deep clone.
#[cfg(test)]
pub(super) fn estimate_request_prompt_tokens_from_definitions(
    messages: &[Message],
    defs: Option<&[ToolDefinition]>,
    reasoning_items: Option<&rustc_hash::FxHashMap<String, Vec<Value>>>,
) -> usize {
    estimate_request_prompt_from_definitions(messages, defs, reasoning_items).0
}

/// [`estimate_request_prompt_tokens_from_definitions`] plus the message
/// counts it walked, so the preview budget derives its chars-per-token ratio
/// without a second history scan.
pub(super) fn estimate_request_prompt_from_definitions(
    messages: &[Message],
    defs: Option<&[ToolDefinition]>,
    reasoning_items: Option<&rustc_hash::FxHashMap<String, Vec<Value>>>,
) -> (usize, BillableCounts) {
    let counts = message_prompt_counts(messages);
    let tokens = counts.tokens()
        + estimate_tools_tokens_from_definitions(defs)
        + estimate_reasoning_items_tokens(messages, reasoning_items);
    (tokens, counts)
}

/// Pessimistic twin of [`estimate_request_prompt_from_definitions`] for the
/// preview path. The counts walk is shared in spirit but projected with
/// `tokens_upper`, so this needs its own pass over the same inputs.
pub(super) fn estimate_request_prompt_upper_from_definitions(
    messages: &[Message],
    defs: Option<&[ToolDefinition]>,
    reasoning_items: Option<&rustc_hash::FxHashMap<String, Vec<Value>>>,
) -> usize {
    message_prompt_counts(messages).tokens_upper()
        + estimate_tools_tokens_upper_from_definitions(defs)
        + estimate_reasoning_items_tokens_upper(messages, reasoning_items)
}

/// Estimates input tokens for this request: messages + tool schemas + any
/// reasoning replay the wire body splices in.
/// The request transport's TPM preflight gate and max_tokens clamp share this path so
/// context-window and rate budgets do not diverge because of different estimates.
pub(super) fn estimate_request_prompt_tokens(
    messages: &[Message],
    tools: Option<&Value>,
    reasoning_items: Option<&rustc_hash::FxHashMap<String, Vec<Value>>>,
) -> usize {
    estimate_request_prompt(messages, tools, reasoning_items).0
}

fn estimate_request_prompt(
    messages: &[Message],
    tools: Option<&Value>,
    reasoning_items: Option<&rustc_hash::FxHashMap<String, Vec<Value>>>,
) -> (usize, BillableCounts) {
    let counts = message_prompt_counts(messages);
    let tokens = counts.tokens()
        + estimate_tools_tokens(tools)
        + estimate_reasoning_items_tokens(messages, reasoning_items);
    (tokens, counts)
}

/// Pessimistic twin of [`estimate_request_prompt`]: same inputs, every
/// component projected with its `_upper` variant. Stored on the wire body
/// next to the lower estimate (via `PromptTokenFeedback`) for the water
/// meter's display interval.
pub(super) fn estimate_request_prompt_upper(
    messages: &[Message],
    tools: Option<&Value>,
    reasoning_items: Option<&rustc_hash::FxHashMap<String, Vec<Value>>>,
) -> usize {
    message_prompt_counts(messages).tokens_upper()
        + estimate_tools_tokens_upper(tools)
        + estimate_reasoning_items_tokens_upper(messages, reasoning_items)
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
            .unwrap_or_else(|| estimate_request_prompt_tokens(messages, tools, None)),
        model_max,
    )
}

/// Reuse a current-request estimate without walking the history again.
pub(super) fn clamp_with_estimated_prompt(model: &str, est_prompt: usize, model_max: u32) -> u32 {
    clamp_with_estimated_prompt_from_parts(models::context_window_tokens(model), est_prompt, model_max)
}

/// Registry-free core of [`clamp_with_estimated_prompt`] over the resolved
/// context window.
pub(super) fn clamp_with_estimated_prompt_from_parts(
    context_window: usize,
    est_prompt: usize,
    model_max: u32,
) -> u32 {
    let window = context_window;
    let remaining = window
        .saturating_sub(est_prompt)
        .saturating_sub(CONTEXT_WINDOW_SAFETY_MARGIN_TOKENS);
    let remaining = u32::try_from(remaining).unwrap_or(u32::MAX);
    model_max.min(remaining).max(MIN_OUTPUT_TOKENS_FLOOR)
}

/// Pre-resolved registry inputs for request-body assembly: every `models::*`
/// lookup `build_request_body` needs, resolved once from the registry key.
/// Behavior tests construct this literally with synthetic values so wire
/// assertions never depend on registry keys (which churn on renames).
pub(super) struct RequestModelInputs {
    pub request_model: String,
    pub endpoint: String,
    pub adapter: ApiProvider,
    pub effort_wire: Option<ReasoningEffortWire>,
    pub dialect: RequestProtocolDialect,
    pub max_output: Option<u32>,
    pub encrypted_replay: bool,
    pub context_window: usize,
}

impl RequestModelInputs {
    pub(super) fn resolve(model: &str) -> Self {
        let endpoint = models::endpoint_for_model(model, "");
        Self::resolve_with_endpoint(model, &endpoint)
    }

    /// Resolve with a caller-supplied endpoint (e.g. the app-level override
    /// from `endpoint_for_request_model`): the dialect is re-resolved against
    /// it because undeclared models infer protocol from endpoint shape.
    pub(super) fn resolve_with_endpoint(model: &str, endpoint: &str) -> Self {
        Self {
            request_model: models::request_model_name(model),
            adapter: models::model_adapter(model),
            effort_wire: models::reasoning_effort_wire(model),
            dialect: models::request_protocol_dialect(model, endpoint),
            max_output: models::max_output_tokens(model),
            encrypted_replay: models::reasoning_encrypted_replay_enabled(model),
            context_window: models::context_window_tokens(model),
            endpoint: endpoint.to_string(),
        }
    }
}

/// Resolve the wire `enable_search` field shared by body assembly and the
/// tool-schema-free preview estimate. Responses keeps the caller flag, plain
/// OpenAI-compatible endpoints split by endpoint shape, every other adapter
/// decides via its own `enable_search_field`. Extracted so both paths cannot
/// diverge on search-gated fingerprints.
pub(super) fn resolve_wire_enable_search(
    model: &str,
    endpoint: &str,
    enable_search: Option<bool>,
) -> Option<bool> {
    let adapter_kind = models::model_adapter(model);
    resolve_wire_enable_search_from_parts(
        adapter_kind,
        models::request_protocol_dialect(model, endpoint),
        endpoint,
        enable_search,
    )
}

/// Registry-free core of [`resolve_wire_enable_search`] over pre-resolved
/// inputs.
pub(super) fn resolve_wire_enable_search_from_parts(
    adapter_kind: ApiProvider,
    dialect: RequestProtocolDialect,
    endpoint: &str,
    enable_search: Option<bool>,
) -> Option<bool> {
    if dialect == RequestProtocolDialect::Responses {
        return enable_search;
    }
    if adapter_kind == ApiProvider::Compatible {
        let (enable_search, _, _) = compatible_wire_shapes(endpoint, enable_search, None);
        return enable_search;
    }
    adapter_for(adapter_kind, endpoint).enable_search_field(enable_search)
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
    build_request_body_from_parts(
        &RequestModelInputs::resolve(model),
        messages,
        stream,
        enable_thinking,
        enable_search,
        tools,
        tool_choice,
        reasoning_effort,
        max_tokens_override,
        current_prompt_tokens,
        reasoning_items,
    )
}

/// Registry-free core of [`build_request_body`]: identical assembly over
/// pre-resolved [`RequestModelInputs`]. Behavior tests construct the inputs
/// literally; production resolves them once via [`RequestModelInputs::resolve`].
#[allow(clippy::too_many_arguments)]
pub(super) fn build_request_body_from_parts<'a>(
    inputs: &RequestModelInputs,
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
    let endpoint = &inputs.endpoint;
    let request_model = inputs.request_model.clone();
    let (thinking, reasoning_effort, reasoning) =
        resolve_reasoning_wire_controls_from_parts(
            inputs.adapter,
            &request_model,
            endpoint,
            inputs.effort_wire,
            enable_thinking,
            reasoning_effort,
        );
    // Responses maps search to the built-in web_search tool; for Chat Completions,
    // the provider dialect still decides whether to retain `enable_search`.
    let enable_search = resolve_wire_enable_search_from_parts(
        inputs.adapter,
        inputs.dialect,
        endpoint,
        enable_search,
    );
    // Explicitly request streaming usage: some adapters (DashScope compatible-mode) omit
    // usage by default and require stream_options.include_usage for token accounting.
    let stream_options = stream.then(|| json!({ "include_usage": true }));
    // Send max_tokens only when the model declares max_output_tokens; clamp to the remaining
    // context window so prompt + requested output cap cannot overflow it (the cause of GLM's
    // repeated truncation/retry loop with long contexts). Models without max_output_tokens
    // still omit this field, preserving their wire behavior.
    // Compute the raw estimate once. Transport may calibrate it against a
    // compatible request snapshot, but keeps this uncalibrated baseline for
    // the next response's growth calculation. Reasoning replay is included
    // whenever the side channel carries it; a cross-dialect fallback that
    // drops it on the wire only overestimates.
    let (est_prompt, message_counts) =
        estimate_request_prompt(messages, tools.as_ref(), reasoning_items);
    let max_tokens = inputs.max_output.map(|model_max| {
        let current = current_prompt_tokens
            .map(|tokens| usize::try_from(tokens).unwrap_or(usize::MAX))
            .unwrap_or(est_prompt);
        clamp_with_estimated_prompt_from_parts(inputs.context_window, current, model_max)
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
        tool_errors: None,
        reasoning_encrypted_replay: inputs.encrypted_replay,
        estimated_prompt_tokens: est_prompt,
        message_counts,
        prompt_cache_key: None,
        explicit_cache_breakpoints: false,
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

    #[test]
    fn tool_definitions_estimate_matches_serialized_schema() {
        use super::{
            estimate_tools_tokens_from_definitions, tools_definitions_compact_json_char_len,
        };
        use crate::ai::types::{FunctionDefinition, ToolDefinition};

        fn tool(name: &str, description: &str, parameters: Value) -> ToolDefinition {
            ToolDefinition {
                tool_type: "function".to_owned(),
                function: FunctionDefinition {
                    name: name.to_owned(),
                    description: description.to_owned(),
                    parameters,
                },
            }
        }

        fn assert_defs_parity(defs: &[ToolDefinition]) {
            let serialized = serde_json::to_value(defs).expect("tool defs must serialize");
            assert_eq!(
                tools_definitions_compact_json_char_len(defs),
                compact_json_char_len(&serialized),
                "defs char-len parity failed"
            );
            // Byte-level ground truth, not just walker-vs-walker.
            assert_eq!(
                tools_definitions_compact_json_char_len(defs),
                serde_json::to_string(&serialized)
                    .expect("tool defs must serialize")
                    .chars()
                    .count(),
                "defs char-len must match compact JSON text"
            );
            assert_eq!(
                estimate_tools_tokens_from_definitions(Some(defs)),
                super::estimate_tools_tokens(Some(&serialized)),
                "defs token parity failed"
            );
        }

        assert_defs_parity(&[]);
        assert_defs_parity(&[tool("read_file", "read a file", json!({"type": "object"}))]);
        assert_defs_parity(&[
            tool(
                "a\"b\\c\n",
                "desc with \"quotes\", \\backslash\\, \u{1}control, 中文, 🙂",
                json!({"type": "object", "properties": {"p": {"type": "string"}}}),
            ),
            tool("", "", Value::Null),
            tool(
                "mcp__server__long_tool_name___with_many_chars",
                &"x".repeat(5_000),
                json!([1, "two", {"three": [true, null, 1.5]}]),
            ),
        ]);
        assert_eq!(estimate_tools_tokens_from_definitions(None), 0);
    }

    #[test]
    fn token_estimate_weights_scripts_and_bills_only_projected_reasoning_replay() {
        use super::{estimate_request_prompt_tokens, estimate_tools_tokens};
        use crate::ai::history::Message;
        use crate::ai::types::{FunctionCall, ToolCall};

        // ASCII JSON at 3 chars/token, CJK at 1 token per scalar (+2 quotes).
        assert_eq!(estimate_tools_tokens(Some(&json!("abcd"))), 2);
        assert_eq!(estimate_tools_tokens(Some(&json!("中文"))), 3);

        let call = |id: &str| ToolCall {
            id: id.to_owned(),
            tool_type: "function".to_owned(),
            function: FunctionCall {
                name: "read_file".to_owned(),
                arguments: "{}".to_owned(),
            },
        };
        let messages = vec![Message {
            role: "assistant".to_owned(),
            content: Value::String(String::new()),
            tool_calls: Some(vec![call("live"), call("second")]),
            tool_call_id: None,
            reasoning_content: None,
        }];
        let base = estimate_request_prompt_tokens(&messages, None, None);
        let encrypted = "A".repeat(4_000);
        let item = json!({"type": "reasoning", "encrypted_content": encrypted});
        let mut items = rustc_hash::FxHashMap::default();
        // Keyed by a non-first call or by a compressed-away round: not on the wire.
        items.insert("second".to_owned(), vec![item.clone()]);
        items.insert("gone".to_owned(), vec![item.clone()]);
        assert_eq!(
            estimate_request_prompt_tokens(&messages, None, Some(&items)),
            base
        );
        items.insert("live".to_owned(), vec![item]);
        // 4000 base64 chars -> 3000 bytes - 650 envelope -> 588 tokens, plus
        // `{"type":"reasoning","encrypted_content":}` as ASCII JSON (14 tokens).
        assert_eq!(
            estimate_request_prompt_tokens(&messages, None, Some(&items)),
            base + 588 + 14
        );
    }
}
