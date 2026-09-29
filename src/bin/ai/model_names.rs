use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use crate::commonw::utils::expanduser;
use rust_tools::cw::SkipMap;
use rust_tools::cw::SkipSet;

use super::provider::{ApiProvider, ModelQualityTier, ReasoningEffort};
use super::request_protocol::RequestProtocolDialect;

/// Default `true` for opt-in capability flags. Matches the `models.rs` fallback
/// used when a registry entry omits the field, so minimal user configs keep
/// working instead of failing to parse.
fn default_true() -> bool {
    true
}

/// Parse a human-friendly token count: plain numbers (`131072`), grouped
/// digits (`1_048_576`, `1,048,576`), or `k`/`M` magnitudes (`128k` = 131072,
/// `1M` = 1048576; `k` = 1024, `M` = 1024*1024, matching how providers quote
/// context/output limits). Anything else is rejected so typos fail fast
/// instead of silently becoming a default.
fn parse_token_count(text: &str) -> Result<u64, String> {
    let trimmed = text.trim();
    let (digits, multiplier) = if let Some(rest) = trimmed
        .strip_suffix('k')
        .or_else(|| trimmed.strip_suffix('K'))
    {
        (rest, 1024u64)
    } else if let Some(rest) = trimmed
        .strip_suffix('m')
        .or_else(|| trimmed.strip_suffix('M'))
    {
        (rest, 1024 * 1024u64)
    } else {
        (trimmed, 1u64)
    };
    let cleaned: String = digits
        .chars()
        .filter(|ch| *ch != '_' && *ch != ',')
        .collect();
    let base: u64 = cleaned
        .trim()
        .parse()
        .map_err(|_| format!("invalid token count '{text}'"))?;
    base.checked_mul(multiplier)
        .ok_or_else(|| format!("invalid token count '{text}'"))
}

/// Parse an optional token count from a JSON number or a human-friendly string
/// (see `parse_token_count`). Missing/null stays `None`; any other shape
/// is a hard parse error so typos still fail fast instead of silently
/// becoming a default.
fn deserialize_optional_usize_string_or_number<'de, D>(
    deserializer: D,
) -> Result<Option<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let raw = Option::<serde_json::Value>::deserialize(deserializer)?;
    match raw {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom(format!("invalid token count {n}"))),
        Some(serde_json::Value::String(s)) => {
            let value = parse_token_count(&s).map_err(serde::de::Error::custom)?;
            usize::try_from(value)
                .map(Some)
                .map_err(|_| serde::de::Error::custom(format!("invalid token count '{s}'")))
        }
        Some(other) => Err(serde::de::Error::custom(format!(
            "invalid token count {other}"
        ))),
    }
}

/// Same tolerance as the `usize` variant, for the per-response output cap.
fn deserialize_optional_u32_string_or_number<'de, D>(
    deserializer: D,
) -> Result<Option<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let raw = Option::<serde_json::Value>::deserialize(deserializer)?;
    match raw {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom(format!("invalid token count {n}"))),
        Some(serde_json::Value::String(s)) => {
            let value = parse_token_count(&s).map_err(serde::de::Error::custom)?;
            u32::try_from(value)
                .map(Some)
                .map_err(|_| serde::de::Error::custom(format!("invalid token count '{s}'")))
        }
        Some(other) => Err(serde::de::Error::custom(format!(
            "invalid token count {other}"
        ))),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffortWire {
    TopLevel,
    Nested,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDef {
    pub key: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub name: String,
    /// Request wire-profile adapter. Determines request body / streaming parse / auth candidate chain behavior.
    /// New configs should use `adapter`; `provider` is kept as a backward-compatible alias.
    #[serde(default, alias = "provider")]
    pub adapter: ApiProvider,
    /// Platform identifier, used only for display names, selector suffixes, logs, and config semantics.
    /// Defaults to the `adapter` slug when unset (e.g. `compatible` / `openai`).
    #[serde(default)]
    pub platform: Option<String>,
    #[serde(default, alias = "base_url")]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub api_key_config_key: Option<String>,
    /// Optional: specify the API key literal directly (takes precedence over api_key_config_key).
    /// For cases that should not go through configw lookup (e.g. temporary tests, custom endpoints).
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub quality_tier: ModelQualityTier,
    /// Defaults to false, matching the `models::is_vl_model` fallback for unknown models.
    #[serde(default)]
    pub is_vl: bool,
    /// Provider-native web search. Enable only when the request protocol has a clear wire mapping:
    /// DashScope Chat Completions uses `enable_search`, OpenAI Responses uses the built-in
    /// `web_search` tool; client-side tools are not part of this field.
    /// Defaults to true, matching the `models::search_enabled` fallback for unknown models.
    #[serde(default = "default_true")]
    pub search_enabled: bool,
    /// Defaults to true, matching the `models::tools_enabled` fallback for unknown models.
    #[serde(default = "default_true")]
    pub tools_default_enabled: bool,
    /// Whether `cache_control: {"type":"ephemeral"}` can be injected onto message content blocks
    /// to enable explicit prompt caching.
    #[serde(default, alias = "supports_explicit_prompt_cache")]
    pub explicit_prompt_cache: bool,
    #[serde(default)]
    pub enable_thinking: bool,
    /// Total token budget for one request: prompt + reserved output combined.
    /// Copy the provider's "context" number here (e.g. `1048576` for a 1M model;
    /// `"1M"` / `"128k"` spellings are accepted). Rough scale: ~2 characters per
    /// token on Chinese-heavy text, so 1M tokens hold about 2MB of text.
    /// Used for the driver's dynamic compression budget estimation and for
    /// clamping `max_output_tokens` to what still fits (see
    /// `clamp_max_tokens_for_prompt`): when unset, a conservative per-tier
    /// fallback applies (flagship 256k / strong 200k / standard 128k / basic
    /// 100k), which over-compresses history and squeezes the output cap on
    /// large-context tasks. Always declare this together with
    /// `max_output_tokens`, with a window strictly larger than the cap.
    #[serde(
        default,
        alias = "context_window",
        alias = "max_context_tokens",
        deserialize_with = "deserialize_optional_usize_string_or_number"
    )]
    pub context_window_tokens: Option<usize>,
    /// Maximum tokens for a single response (visible text + hidden reasoning
    /// combined), sent as the request's `max_tokens`. Copy the provider's
    /// "output" number here (e.g. `131072`; `"128k"` accepted).
    /// Many OpenAI-compatible providers apply a conservative completion cap when the client does not specify one,
    /// truncating large `write_file` payloads / long documents mid-generation. Declaring a value close to the
    /// model's real limit mitigates truncation; when unset (None), `max_tokens` is not sent, preserving historical behavior.
    /// Must stay strictly below `context_window_tokens`: declare both together,
    /// otherwise the prompt clamp squeezes the cap to the 1024 floor on long
    /// tasks and every response truncates.
    #[serde(
        default,
        alias = "max_tokens",
        alias = "max_completion_tokens",
        deserialize_with = "deserialize_optional_u32_string_or_number"
    )]
    pub max_output_tokens: Option<u32>,
    /// Optional: request-layer prompt-token-per-minute preflight budget. Only when the model registry
    /// explicitly sets this field does the request layer wait before sending; when unset (or 0) the TPM preflight is
    /// skipped entirely, so a wrong default never penalizes different providers / keys.
    #[serde(default)]
    pub request_tpm_limit: Option<u64>,
    /// Optional: HTTP protocol dialect for requests. Most models default to
    /// `chat_completions`; only a few models (e.g. modelhub GPT-5.x) explicitly use
    /// `responses`. When unset, a compatible inference is made from the endpoint shape for smooth upgrades of historical configs.
    #[serde(default)]
    pub request_protocol: Option<RequestProtocolDialect>,
    /// Optional: only effective for the `responses` protocol. When enabled, requests carry
    /// `include: ["reasoning.encrypted_content"]` to request encrypted reasoning items, and replay the server-returned `reasoning` output
    /// item verbatim into the input on subsequent tool-call rounds within the same turn,
    /// letting the model retain the previous hop's reasoning context across a multi-step tool chain. Off by default: undeclared
    /// models behave unchanged, and when a gateway does not pass through `encrypted_content` the replay degrades to no replay automatically.
    #[serde(default)]
    pub reasoning_encrypted_replay: bool,
    /// Optional: whether a Chat Completions model requires subsequent tool-call requests to replay the assistant's
    /// `reasoning_content` verbatim. This differs from protocols that only require the field to exist (it may be an empty string);
    /// off by default so unrelated models neither accumulate nor leak hidden reasoning text.
    #[serde(default)]
    pub reasoning_content_replay: bool,
    /// Optional: whether the model's reasoning chain is inlined in the `content` channel instead of a separate
    /// `reasoning_content` field. A few reasoner gateways (observed on volcano ark `/coding`
    /// endpoints with deepseek-v4 / glm models) use a "pre-filled `<think>`" chat template: the reasoning text
    /// is written directly into `content` and the whole block ends with a lone dangling `</think>`, never producing
    /// `reasoning_content`. When enabled, the streaming layer uses `</think>` in the content channel to split the leaked
    /// reasoning chain back into reasoning, keeping the chain of thought out of the visible body (otherwise the "final
    /// answer is printed twice"). Off by default: undeclared model behavior is completely unchanged.
    #[serde(default)]
    pub reasoning_in_content: bool,
    /// Subagent model selection priority (higher wins). Within a tier, models are sorted by this value descending.
    /// Defaults to 0; users can override it in ~/.config/rust_tools/models/ (or the legacy single-file
    /// ~/.config/rust_tools/models.json) to adjust preference without recompiling.
    #[serde(default)]
    pub subagent_priority: i32,

    /// Optional: override the adapter's reasoning-effort wire shape. Different model families behind the same gateway may use
    /// different fields (e.g. DashScope DeepSeek / GLM use the top-level `reasoning_effort`;
    /// undeclared Alibaba models do not send unconfirmed reasoning-effort fields).
    #[serde(default)]
    pub reasoning_effort_wire: Option<ReasoningEffortWire>,

    /// Optional: default reasoning effort tier. The concrete wire shape prefers the model-level override above, and otherwise comes
    /// from the provider adapter. CLI / `/model effort` command overrides take precedence over this.
    ///
    /// Values accepted in the model registry (models/ directory), case-insensitive:
    /// - `"auto"` / `"none"` / `"off"` or the field omitted: equivalent to `None`, no `reasoning_effort` in the request
    ///   (compatible with historical behavior);
    /// - `"minimal"` / `"low"` / `"medium"` / `"high"` / `"xhigh"` / `"max"`: the corresponding tiers.
    #[serde(
        default,
        alias = "reasoning_effort",
        rename = "default_reasoning_effort",
        deserialize_with = "deserialize_default_reasoning_effort"
    )]
    pub reasoning_effort: Option<ReasoningEffort>,

    /// The model's `reasoning_effort` conflicts with the `tools` parameter in the request body.
    /// Some gateways (e.g. bytedance modelhub) reject requests carrying both on `/v1/chat/completions`, returning
    /// 400 "Function tools with reasoning_effort are not
    /// supported". When true, the request layer automatically omits `reasoning_effort` whenever tools is non-empty;
    /// requests without tools still carry the field as usual,
    /// preserving thinking capability.
    #[serde(default)]
    pub reasoning_effort_conflicts_with_tools: bool,
    /// Ignored annotation slot so JSON files can carry reviewer notes
    /// (`"_comment": "..."`) without tripping `deny_unknown_fields`.
    /// Never read at runtime.
    #[serde(default)]
    pub _comment: Option<serde_json::Value>,
}

/// Deserialize a reasoning effort tier from a string; accepts literals such as `auto` / `none` / `off` as
/// "unset" semantics, equivalent to omitting the field.
fn deserialize_default_reasoning_effort<'de, D>(
    deserializer: D,
) -> Result<Option<ReasoningEffort>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let raw = Option::<String>::deserialize(deserializer)?;
    let Some(value) = raw else {
        return Ok(None);
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "auto" | "none" | "off" | "default" | "null" => Ok(None),
        _ => match ReasoningEffort::parse(trimmed) {
            Some(level) => Ok(Some(level)),
            None => Err(serde::de::Error::custom(format!(
                "unknown default_reasoning_effort '{}': expected auto/minimal/low/medium/high/xhigh/max/off",
                trimmed
            ))),
        },
    }
}

static USER_MODELS: LazyLock<Vec<ModelDef>> = LazyLock::new(load_user_models);
static BUILTIN_MODELS: LazyLock<Vec<ModelDef>> = LazyLock::new(load_builtin_models);
static USER_BY_KEY: LazyLock<SkipMap<String, usize>> = LazyLock::new(build_user_key_index);
static BUILTIN_BY_KEY: LazyLock<SkipMap<String, usize>> = LazyLock::new(build_builtin_key_index);
static USER_BY_NAME: LazyLock<SkipMap<String, usize>> = LazyLock::new(build_user_name_index);
static BUILTIN_BY_NAME: LazyLock<SkipMap<String, usize>> = LazyLock::new(build_builtin_name_index);

fn lookup_key(value: &str) -> String {
    let mut normalized = String::new();
    let mut pending_dash = false;
    for ch in value.trim().to_ascii_lowercase().chars() {
        if ch.is_whitespace() {
            if !normalized.is_empty() {
                pending_dash = true;
            }
            continue;
        }
        if pending_dash && !normalized.ends_with('-') {
            normalized.push('-');
        }
        pending_dash = false;
        normalized.push(ch);
    }
    normalized.trim_matches('-').to_string()
}

pub fn adapter_slug(adapter: ApiProvider) -> &'static str {
    match adapter {
        ApiProvider::Alibaba => "alibaba",
        ApiProvider::Compatible => "compatible",
        ApiProvider::OpenAi => "openai",
        ApiProvider::OpenCode => "opencode",
    }
}

pub fn platform_slug(model: &ModelDef) -> String {
    model
        .platform
        .as_deref()
        .map(str::trim)
        .filter(|platform| !platform.is_empty())
        .map(lookup_key)
        .filter(|platform| !platform.is_empty())
        .unwrap_or_else(|| adapter_slug(model.adapter).to_string())
}

pub fn platform_label(model: &ModelDef) -> String {
    platform_slug(model)
}

pub fn model_handle(model: &ModelDef) -> String {
    // If name is in encrypted form (enc: prefix), use the key as the display name
    // to avoid showing garbled enc:xxx-<platform> in the completion panel.
    let is_encrypted = model.name.starts_with("enc:");
    let name = if is_encrypted {
        String::new()
    } else {
        lookup_key(&model.name)
    };
    if name.is_empty() {
        return lookup_key(&model.key);
    }
    format!("{}-{}", name, platform_slug(model))
}

pub fn legacy_adapter_handle(model: &ModelDef) -> Option<String> {
    let is_encrypted = model.name.starts_with("enc:");
    if is_encrypted {
        return None;
    }
    let name = lookup_key(&model.name);
    if name.is_empty() {
        return None;
    }
    let legacy = format!("{}-{}", name, adapter_slug(model.adapter));
    if legacy.eq_ignore_ascii_case(&model_handle(model)) {
        None
    } else {
        Some(legacy)
    }
}

/// User model registry directory (new format, recommended): `~/.config/rust_tools/models/`,
/// following the same convention as the built-in `models/` directory (one JSON file per model).
fn user_config_dir() -> PathBuf {
    let home = expanduser("~/.config/rust_tools/models");
    match home {
        std::borrow::Cow::Owned(s) => PathBuf::from(s),
        std::borrow::Cow::Borrowed(s) => PathBuf::from(s),
    }
}

/// Legacy format compatibility: single-file `~/.config/rust_tools/models.json` user overrides.
/// The new directory format takes precedence when present; the legacy file is only a fallback.
fn legacy_user_config_path() -> PathBuf {
    let home = expanduser("~/.config/rust_tools/models.json");
    match home {
        std::borrow::Cow::Owned(s) => PathBuf::from(s),
        std::borrow::Cow::Borrowed(s) => PathBuf::from(s),
    }
}

fn builtin_config_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

/// Parse a single model file: supports both a single object `{...}` and an object array `[{...}]`.
/// Returns `None` on read/parse/validation failure (error already printed).
fn load_models_from_file(path: &Path) -> Option<Vec<ModelDef>> {
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("[model_names] failed to read {}: {}", path.display(), e);
            return None;
        }
    };
    // Parse as an array first (for merged files), then as a single object on failure.
    if let Ok(models) = serde_json::from_str::<Vec<ModelDef>>(&content) {
        for model in &models {
            if let Err(reason) = validate_model_def(model, path) {
                eprintln!("[model_names] invalid model in {}: {}", path.display(), reason);
                return None;
            }
        }
        return Some(models);
    }
    match serde_json::from_str::<ModelDef>(&content) {
        Ok(model) => {
            if let Err(reason) = validate_model_def(&model, path) {
                eprintln!("[model_names] invalid model in {}: {}", path.display(), reason);
                return None;
            }
            Some(vec![model])
        }
        Err(e) => {
            eprintln!("[model_names] failed to parse {}: {}", path.display(), e);
            None
        }
    }
}

/// Load-time validation: hard errors for values that cannot work (the caller
/// fails the file so misconfigurations surface at startup, not mid-task),
/// warnings only for merely suspicious ones. In particular the
/// silent-truncation pairing (`max_output_tokens` must stay below the
/// effective window, see the `clamp_max_tokens_for_prompt` guard) is rejected
/// here instead of truncating every long response.
fn validate_model_def(model: &ModelDef, path: &Path) -> Result<(), String> {
    if model.key.trim().is_empty() {
        return Err(
            "model with empty `key` will never resolve; check for a typo".to_string(),
        );
    }
    if model.name.trim().is_empty() {
        eprintln!(
            "[model_names] {}: model '{}' has an empty `name`; request routing falls back to the key",
            path.display(),
            model.key
        );
    }
    if let Some(window) = model.context_window_tokens {
        if window == 0 {
            return Err(format!(
                "model '{}' declares context_window_tokens 0; omit the field for the tier fallback or declare the provider's context number",
                model.key
            ));
        } else if window < 8_192 {
            eprintln!(
                "[model_names] {}: model '{}' declares a suspiciously small window {window}; prompt clamping may squeeze max_tokens to the floor",
                path.display(),
                model.key
            );
        }
    }
    match model.max_output_tokens {
        Some(0) => {
            return Err(format!(
                "model '{}' declares max_output_tokens 0; omit the field to send no cap or declare the provider's output number",
                model.key
            ));
        }
        Some(out) => {
            if let Some(window) = model.context_window_tokens.filter(|w| *w > 0)
                && (out as usize) >= window
            {
                return Err(format!(
                    "model '{}' declares max_output_tokens {out} >= context window {window}; the cap is unreachable and long generations truncate early. Declare a window strictly larger than the cap",
                    model.key
                ));
            }
        }
        None => {}
    }
    if let Some(endpoint) = model.endpoint.as_deref()
        && endpoint.trim().is_empty()
    {
        eprintln!(
            "[model_names] {}: model '{}' declares an empty `endpoint`; it is treated as unset",
            path.display(),
            model.key
        );
    }
    Ok(())
}

/// Read all `*.json` model files in the directory (sorted by filename for a deterministic load order).
/// With `strict` true, any file read/parse failure returns `None` (used by the built-in registry to
/// exit immediately instead of silently degrading); with false, bad files are skipped and loading continues (user registry).
fn load_models_from_dir(dir: &Path, strict: bool) -> Option<Vec<ModelDef>> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!("[model_names] failed to read {}: {}", dir.display(), e);
            return None;
        }
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
        })
        .collect();
    paths.sort();
    let mut models = Vec::new();
    for path in paths {
        match load_models_from_file(&path) {
            Some(file_models) => models.extend(file_models),
            None if strict => return None,
            None => {}
        }
    }
    // Duplicate keys in one directory shadow each other in the key index (last
    // file wins). Reject so a copy-paste `key` does not silently hide a model.
    // User-overrides-builtin shadowing across registries is intentional and
    // not affected here.
    let mut keys: Vec<String> = models.iter().map(|m| lookup_key(&m.key)).collect();
    keys.sort();
    for pair in keys.windows(2) {
        if !pair[0].is_empty() && pair[0] == pair[1] {
            eprintln!(
                "[model_names] duplicate model key '{}' in {}; fix the copy-paste `key`",
                pair[0],
                dir.display()
            );
            return None;
        }
    }
    Some(models)
}

fn load_user_models() -> Vec<ModelDef> {
    // New-format directory takes precedence; fall back to the legacy single-file override when absent.
    // Fail fast like the builtin registry: a broken user file exits with the
    // file-specific reason already printed, instead of silently skipping it.
    let dir = user_config_dir();
    if dir.is_dir() {
        return load_models_from_dir(&dir, true).unwrap_or_else(|| {
            eprintln!(
                "[model_names] failed to load user models from {}",
                dir.display()
            );
            std::process::exit(1)
        });
    }
    let legacy = legacy_user_config_path();
    if legacy.exists() {
        load_models_from_file(&legacy).unwrap_or_else(|| {
            eprintln!(
                "[model_names] failed to load user models from {}",
                legacy.display()
            );
            std::process::exit(1)
        })
    } else {
        Vec::new()
    }
}

fn load_builtin_models() -> Vec<ModelDef> {
    let dir = builtin_config_dir();
    if !dir.is_dir() {
        eprintln!(
            "[model_names] builtin models dir not found at {}",
            dir.display()
        );
        std::process::exit(1);
    }
    load_models_from_dir(&dir, true).unwrap_or_else(|| {
        eprintln!(
            "[model_names] failed to load builtin models from {}",
            dir.display()
        );
        std::process::exit(1)
    })
}

fn build_user_key_index() -> SkipMap<String, usize> {
    let mut index = SkipMap::default();
    for (i, m) in USER_MODELS.iter().enumerate() {
        insert_model_key_aliases(&mut index, m, i);
    }
    index
}

fn build_builtin_key_index() -> SkipMap<String, usize> {
    let mut index = SkipMap::default();
    for (i, m) in BUILTIN_MODELS.iter().enumerate() {
        insert_model_key_aliases(&mut index, m, i);
    }
    index
}

fn insert_key_alias(index: &mut SkipMap<String, usize>, alias: &str, i: usize) {
    let key = lookup_key(alias);
    if !key.is_empty() {
        index.insert(key, i);
    }
}

fn insert_model_key_aliases(index: &mut SkipMap<String, usize>, model: &ModelDef, i: usize) {
    insert_key_alias(index, &model_handle(model), i);
    if let Some(legacy) = legacy_adapter_handle(model) {
        insert_key_alias(index, &legacy, i);
    }
    insert_key_alias(index, &model.key, i);
    for alias in &model.aliases {
        insert_key_alias(index, alias, i);
    }
}

fn build_user_name_index() -> SkipMap<String, usize> {
    let mut index = SkipMap::default();
    for (i, m) in USER_MODELS.iter().enumerate() {
        let key = lookup_key(&m.name);
        if !key.is_empty() && !index.contains_key(&key) {
            index.insert(key, i);
        }
    }
    index
}

fn build_builtin_name_index() -> SkipMap<String, usize> {
    let mut index = SkipMap::default();
    for (i, m) in BUILTIN_MODELS.iter().enumerate() {
        let key = lookup_key(&m.name);
        if !key.is_empty() && !index.contains_key(&key) {
            index.insert(key, i);
        }
    }
    index
}

pub fn all() -> Vec<&'static ModelDef> {
    let mut seen = SkipSet::new(16);
    let mut result = Vec::new();

    for m in USER_MODELS.iter() {
        let key = lookup_key(&model_handle(m));
        if seen.insert(key) {
            result.push(m);
        }
    }

    for m in BUILTIN_MODELS.iter() {
        let key = lookup_key(&model_handle(m));
        if seen.insert(key) {
            result.push(m);
        }
    }

    result
}

pub fn find_by_name(name: &str) -> Option<&'static ModelDef> {
    let name_lower = lookup_key(name);

    if let Some(&i) = USER_BY_NAME.get_ref(&name_lower) {
        return Some(&USER_MODELS[i]);
    }

    BUILTIN_BY_NAME
        .get_ref(&name_lower)
        .map(|&i| &BUILTIN_MODELS[i])
}

pub fn find_by_key(key: &str) -> Option<&'static ModelDef> {
    let key_lower = lookup_key(key);

    if let Some(&i) = USER_BY_KEY.get_ref(&key_lower) {
        return Some(&USER_MODELS[i]);
    }

    BUILTIN_BY_KEY
        .get_ref(&key_lower)
        .map(|&i| &BUILTIN_MODELS[i])
}

pub fn find_by_identifier(identifier: &str) -> Option<&'static ModelDef> {
    let trimmed = identifier.trim();
    if trimmed.is_empty() {
        return None;
    }
    find_by_key(trimmed).or_else(|| find_by_name(trimmed))
}
