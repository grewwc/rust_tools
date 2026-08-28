//! Inline tool-call recovery and validation.
//!
//! Handles tool calls that appear in non-standard formats in model output:
//! - `InlineToolCallParser` / `INLINE_PARSERS`: list of registered parsers
//! - `normalize_inline_tool_call_markup`: normalizes namespace-prefixed XML tags
//! - `recover_inline_tool_calls`: recovers tool calls from plain text
//! - `recover_json_tool_calls` / `recover_hermes_xml_tool_calls` / `recover_anthropic_xml_tool_calls`
//!   / `recover_bare_xml_tool_calls`: per-format parsers
//! - `strip_inline_tool_call_wrappers`: removes tool-call wrapper tags
//! - `normalize_tool_call_arguments` / `find_json_object_end`: argument validation
//! - `collect_valid_tool_calls` / `ensure_tool_calls_section_open`: tool-call collection and rendering

use std::sync::LazyLock;

use regex::Regex;

use super::runtime::{
    clear_waiting_hint, finalize_thinking_fold, format_end_thinking_line, write_stream_content,
};
use super::state::{StreamMarkers, StreamProcessingState, ToolCallBuilder};
use crate::ai::types::{App, ToolCall};

type InlineToolCallParser = fn(&str) -> Option<Vec<ToolCall>>;

const INLINE_PARSERS: &[InlineToolCallParser] = &[
    recover_hermes_xml_tool_calls,
    recover_anthropic_xml_tool_calls,
    recover_bare_xml_tool_calls,
    recover_json_tool_calls,
];

/// Normalizes namespace-prefixed XML tags in model output into standard XML tags, e.g.
/// `<|DSML|invoke name="x">` / `<｜｜DSML｜｜invoke name="x">` → `<invoke name="x">`，
/// `</|DSML|invoke>` / `</｜｜DSML｜｜invoke>` → `</invoke>`。
/// Also handles the space-separated form some models emit: `<｜｜DSML ｜ invoke ...>`.
/// This way the Hermes / Anthropic parsers need no per-`<|PREFIX|>` protocol adaptation.
pub(super) fn normalize_inline_tool_call_markup(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains("<|")
        && !text.contains("</|")
        && !text.contains("<｜")
        && !text.contains("</｜")
    {
        return std::borrow::Cow::Borrowed(text);
    }
    static OPEN_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r#"<(?:\|\s*([^>|]+?)\s*\|\s*([^\s>|｜]+)([^＞>]*)|｜\s*｜?\s*([^｜＞>]+?)\s*｜\s*｜?\s*([^\s｜＞>]+)([^＞>]*))[＞>]"#,
        )
            .expect("valid open-tag regex")
    });
    static CLOSE_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r#"</(?:\|\s*([^>|]+?)\s*\|\s*([^\s>|｜]+)|｜\s*｜?\s*([^｜＞>]+?)\s*｜\s*｜?\s*([^\s｜＞>]+))\s*[＞>]"#,
        )
            .expect("valid close-tag regex")
    });
    let s = OPEN_RE.replace_all(text, |caps: &regex::Captures<'_>| {
        let local = caps
            .get(2)
            .or_else(|| caps.get(5))
            .map(|m| m.as_str())
            .unwrap_or("");
        let tail = caps
            .get(3)
            .or_else(|| caps.get(6))
            .map(|m| m.as_str())
            .unwrap_or("");
        format!("<{local}{tail}>")
    });
    std::borrow::Cow::Owned(
        CLOSE_RE
            .replace_all(&s, |caps: &regex::Captures<'_>| {
                let local = caps
                    .get(2)
                    .or_else(|| caps.get(4))
                    .map(|m| m.as_str())
                    .unwrap_or("");
                format!("</{local}>")
            })
            .into_owned(),
    )
}

/// Stateful namespace-tag normalizer for streaming scenarios.
///
/// `normalize_inline_tool_call_markup` is a stateless whole-text normalizer; it cannot
/// handle a marker split across two delta chunk boundaries (e.g. chunk A receives
/// `<｜｜DS`, chunk B receives `ML｜｜tool_calls>`). In that case the raw marker leaks
/// into the body text and subsequent parsing fails.
///
/// This struct caches a trailing prefix that may be "half a marker" and normalizes it
/// together once the next chunk completes it.
#[derive(Default)]
pub(super) struct InlineMarkupNormalizer {
    pending: String,
}

impl InlineMarkupNormalizer {
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Feeds in one chunk and returns normalized text that is safe to emit. If the tail
    /// may be a truncated marker prefix, it is stashed in `pending` until the next
    /// `push` completes it.
    pub(super) fn push(&mut self, chunk: &str) -> String {
        if chunk.is_empty() {
            return String::new();
        }
        self.pending.push_str(chunk);
        // Find the last `<`: everything after it may be an unclosed (namespace) tag.
        // If that segment could be a split marker, keep all of it for the next chunk.
        let hold_from = self
            .pending
            .rfind('<')
            .filter(|&pos| Self::maybe_partial_marker(&self.pending[pos..]));
        let emit = match hold_from {
            Some(pos) => {
                let tail = self.pending.split_off(pos);
                let head = std::mem::replace(&mut self.pending, tail);
                head
            }
            None => std::mem::take(&mut self.pending),
        };
        normalize_inline_tool_call_markup(&emit).into_owned()
    }

    /// Flushes the remaining cache at end of stream (either a complete marker missing
    /// its trailing `>`, or plain text).
    pub(super) fn flush(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let rest = std::mem::take(&mut self.pending);
        normalize_inline_tool_call_markup(&rest).into_owned()
    }

    /// Determines whether a fragment starting at some `<` may be a namespace marker
    /// that has not been fully received yet — i.e. `<|...` / `<｜｜...` / `</|...` /
    /// `</｜｜...` with no closing `>` yet.
    /// If the fragment already contains `>`, the tag is complete and need not be kept.
    fn maybe_partial_marker(frag: &str) -> bool {
        if frag.contains('>') {
            return false;
        }
        // Walk the prefix character by character to check whether it could still grow
        // into the start of a namespace marker.
        // Allowed starts: `<`, `</`, followed by an ASCII `|` or a full-width `｜`.
        const HALF: char = '|';
        const FULL: char = '｜';
        let mut chars = frag.chars();
        debug_assert_eq!(chars.next(), Some('<'));
        let mut rest = chars.as_str();
        if let Some(after) = rest.strip_prefix('/') {
            rest = after;
        }
        // Empty (stopped right at `<` or `</`) also counts as a possible marker prefix.
        if rest.is_empty() {
            return true;
        }
        // Allow ASCII/full-width vertical bars inside the marker prefix; as long as no
        // closing `>` has appeared, treat it as a possibly partial marker.
        rest.starts_with(HALF) || rest.starts_with(FULL)
    }
}

/// Attempts to parse a whole assistant text back into one or more tool_calls.
/// Via the parser registry plus upfront XML namespace normalization, it uniformly
/// handles the inline tool call forms produced by different models (Hermes XML,
/// Anthropic XML, JSON, `<|PREFIX|>` wrappers).
/// Returns as soon as any parser succeeds; if all fail, the text is treated as a
/// plain answer.
pub(super) fn recover_inline_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let normalized = normalize_inline_tool_call_markup(text);
    for parser in INLINE_PARSERS {
        if let Some(calls) = parser(&normalized) {
            return Some(calls);
        }
    }
    None
}

/// Recognizes JSON-form tool calls in assistant text (single object or array).
fn recover_json_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let stripped = strip_inline_tool_call_wrappers(text);
    let trimmed = stripped.trim();
    if trimmed.is_empty() {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(trimmed).ok()?;
    let raw_calls: Vec<&serde_json::Value> = match &value {
        serde_json::Value::Object(_) => vec![&value],
        serde_json::Value::Array(items) if !items.is_empty() => items.iter().collect(),
        _ => return None,
    };

    let mut out = Vec::with_capacity(raw_calls.len());
    for (idx, raw) in raw_calls.into_iter().enumerate() {
        let obj = raw.as_object()?;
        // Handles both the OpenAI style {"function": {"name", "arguments"}, "id"} and
        // the simplified style {"name", "arguments"}.
        let (name, arguments_value, id) = if let Some(func) = obj.get("function") {
            let func_obj = func.as_object()?;
            let name = func_obj.get("name")?.as_str()?.to_string();
            let args = func_obj
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
            let id = obj
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            (name, args, id)
        } else {
            let name = obj.get("name")?.as_str()?.to_string();
            let args = obj
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
            let id = obj
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            (name, args, id)
        };
        if name.trim().is_empty() {
            return None;
        }
        let arguments = match arguments_value {
            serde_json::Value::String(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    "{}".to_string()
                } else {
                    // Validate that the inner string is really JSON, so arbitrary strings
                    // are not passed through as args.
                    serde_json::from_str::<serde_json::Value>(trimmed).ok()?;
                    trimmed.to_string()
                }
            }
            other => other.to_string(),
        };
        out.push(ToolCall {
            id: id.unwrap_or_else(|| format!("inline_{idx}")),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall { name, arguments },
        });
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Parses Hermes / Qwen style XML tool calls. Supports:
///   - multiple `<function=NAME> ... </function>` blocks (parallel tool calls)
///   - JSON body: `<function=read_file>{"path":"/x"}</function>`
///   - parameter-tag body: `<function=read_file><parameter=path>/x</parameter></function>`
///   - an optional outer `<tool_call>...</tool_call>` wrapper
/// Returns as soon as any `<function=...>` block parses; returns None if all fail.
fn recover_hermes_xml_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let mut out: Vec<ToolCall> = Vec::new();
    let mut rest = text;
    let mut idx = 0usize;
    while let Some(open_rel) = rest.find("<function=") {
        let after_open = &rest[open_rel + "<function=".len()..];
        // Function name runs up to the first '>'.
        let Some(name_end) = after_open.find('>') else {
            break;
        };
        let name = after_open[..name_end].trim().to_string();
        let body_start = name_end + 1;
        // Body runs up to the matching </function>; if the closing tag is missing,
        // take the rest.
        let body_region = &after_open[body_start..];
        let (body, consumed_to) = match body_region.find("</function>") {
            Some(close_rel) => (
                &body_region[..close_rel],
                body_start + close_rel + "</function>".len(),
            ),
            None => (body_region, body_region.len() + body_start),
        };
        if !name.is_empty() {
            if let Some(arguments) = parse_hermes_function_body(body) {
                out.push(ToolCall {
                    id: format!("inline_xml_{idx}"),
                    tool_type: "function".to_string(),
                    function: crate::ai::types::FunctionCall { name, arguments },
                });
                idx += 1;
            }
        }
        // Advance past the end of this block and keep scanning for subsequent
        // parallel blocks.
        let advance = open_rel + "<function=".len() + consumed_to;
        if advance >= rest.len() {
            break;
        }
        rest = &rest[advance..];
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Parses Anthropic / Claude style XML tool calls. Supports:
///   - multiple `<invoke name="NAME"> ... </invoke>` blocks (parallel tool calls)
///   - parameters as a set of `<parameter name="key">value</parameter>` tags
///   - an optional outer `<function_calls>` / `<tool_calls>` wrapper
///   - tags may carry a namespace prefix (e.g. `antml:invoke`)
/// Returns as soon as any `<invoke ...>` block parses; returns None if all fail.
fn recover_anthropic_xml_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let mut out: Vec<ToolCall> = Vec::new();
    let mut rest = text;
    let mut idx = 0usize;
    while let Some(open_rel) = rest.find("<invoke") {
        let after_tag = &rest[open_rel..];
        // Locate the '>' of this invoke's opening tag.
        let Some(open_gt) = after_tag.find('>') else {
            break;
        };
        let open_tag = &after_tag[..=open_gt];
        let name = parse_anthropic_xml_name_attr(open_tag);
        let body_start = open_rel + open_gt + 1;
        let body_region = &rest[body_start..];
        let (body, consumed_to) = match body_region.find("</invoke>") {
            Some(close_rel) => (
                &body_region[..close_rel],
                body_start + close_rel + "</invoke>".len(),
            ),
            None => (body_region, rest.len()),
        };
        if !name.trim().is_empty() {
            let arguments = parse_anthropic_invoke_body(body);
            out.push(ToolCall {
                id: format!("inline_anthropic_{idx}"),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall { name, arguments },
            });
            idx += 1;
        }
        if consumed_to >= rest.len() {
            break;
        }
        rest = &rest[consumed_to..];
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Parses bare tool-name XML: `<execute_command>pwd</execute_command>`.
/// Only takes effect for registered tool names on the allowlist, so ordinary
/// HTML/XML tags are not mistaken for tool calls.
/// Unlike Hermes / Anthropic, the tag name itself is the tool name; the body may be
/// JSON arguments, or raw text for tools like `execute_command` that have exactly
/// one required string parameter.
fn recover_bare_xml_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let stripped = strip_inline_tool_call_wrappers(text);
    let mut rest = stripped.trim();
    if rest.is_empty() {
        return None;
    }

    let mut out = Vec::new();
    let mut idx = 0usize;
    while !rest.is_empty() {
        let Some(open_end) = rest.find('>') else {
            return None;
        };
        let Some(name) = parse_bare_xml_open_tag(&rest[..=open_end]) else {
            return None;
        };
        let body_start = open_end + 1;
        let close_tag = format!("</{name}>");
        let Some(close_rel) = rest[body_start..].find(&close_tag) else {
            return None;
        };
        let body_end = body_start + close_rel;
        let arguments = parse_bare_xml_tool_body(&name, &rest[body_start..body_end])?;
        out.push(ToolCall {
            id: format!("inline_bare_xml_{idx}"),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall { name, arguments },
        });
        idx += 1;
        rest = rest[body_end + close_tag.len()..].trim_start();
    }

    if out.is_empty() { None } else { Some(out) }
}

/// Parses the `<parameter name="key">value</parameter>` tags in an `<invoke>` body
/// into a JSON arguments string; returns `{}` when there are no parameters.
fn parse_anthropic_invoke_body(body: &str) -> String {
    let mut map = serde_json::Map::new();
    let mut rest = body;
    while let Some(open_rel) = rest.find("<parameter") {
        let after_tag = &rest[open_rel..];
        let Some(open_gt) = after_tag.find('>') else {
            break;
        };
        let open_tag = &after_tag[..=open_gt];
        let key = parse_anthropic_xml_name_attr(open_tag);
        let force_string = parse_anthropic_xml_bool_attr(open_tag, "string");
        let value_region = &after_tag[open_gt + 1..];
        let (raw_value, consumed_in_after) = match value_region.find("</parameter>") {
            Some(close_rel) => (
                &value_region[..close_rel],
                open_gt + 1 + close_rel + "</parameter>".len(),
            ),
            None => break,
        };
        let raw_value = raw_value.trim();
        if !key.trim().is_empty() {
            let value = if force_string {
                serde_json::Value::String(raw_value.to_string())
            } else {
                serde_json::from_str::<serde_json::Value>(raw_value)
                    .unwrap_or_else(|_| serde_json::Value::String(raw_value.to_string()))
            };
            map.insert(key, value);
        }
        let advance = open_rel + consumed_in_after;
        if advance >= rest.len() {
            break;
        }
        rest = &rest[advance..];
    }
    if map.is_empty() {
        "{}".to_string()
    } else {
        serde_json::Value::Object(map).to_string()
    }
}

/// Extracts the `name` attribute value from an `<invoke name="x">` /
/// `<parameter name="y">` opening tag; supports double or single quotes.
fn parse_anthropic_xml_name_attr(open_tag: &str) -> String {
    super::splitter::parse_xml_attr_value(open_tag, "name")
        .unwrap_or_default()
        .to_string()
}

/// Parses bool attributes such as `string="true"` from an opening tag's attribute
/// string (DSML protocol).
/// Values are case-insensitive; a missing or non-"true" value returns false.
fn parse_anthropic_xml_bool_attr(open_tag: &str, name: &str) -> bool {
    super::splitter::parse_xml_attr_value(open_tag, name)
        .is_some_and(|value| value.eq_ignore_ascii_case("true"))
}

/// Parses a bare XML opening tag; the tag name must itself be a registered tool
/// name, and the tag must carry no attributes.
pub(super) fn parse_bare_xml_open_tag(tag: &str) -> Option<String> {
    let inner = tag.trim();
    if !inner.starts_with('<') || !inner.ends_with('>') {
        return None;
    }
    let inner = inner[1..inner.len() - 1].trim();
    if inner.is_empty() || inner.starts_with('/') || inner.ends_with('/') {
        return None;
    }
    if inner.contains(char::is_whitespace) {
        return None;
    }
    crate::ai::tools::registry::common::is_registered_tool_name(inner).then(|| inner.to_string())
}

/// Parses a bare XML tool body. Prefers JSON objects / Hermes parameter tags;
/// if the body is only raw text, it safely degrades only for tools with exactly
/// one required string parameter, e.g.
/// `<execute_command>pwd</execute_command>` → `{"command":"pwd"}`.
pub(super) fn parse_bare_xml_tool_body(tool_name: &str, body: &str) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Some("{}".to_string());
    }
    if let Some(args) = parse_hermes_function_body(trimmed) {
        return Some(args);
    }

    let key = single_required_string_argument_key(tool_name)?;
    Some(serde_json::json!({ key: trimmed }).to_string())
}

fn single_required_string_argument_key(tool_name: &str) -> Option<String> {
    let schema = crate::ai::tools::registry::tool_metadata::tool_parameters(tool_name);
    let schema = schema.as_object()?;
    let required = schema.get("required")?.as_array()?;
    let props = schema.get("properties")?.as_object()?;
    // Find all required parameters whose type is string.
    // If and only if there is exactly one required string parameter, the raw text
    // body maps to it; other required non-string parameters (e.g. execute_command's
    // pty: bool) are filled from runtime defaults.
    let mut string_keys = Vec::new();
    for item in required {
        let key = item.as_str()?;
        if let Some(prop) = props.get(key).and_then(|v| v.as_object()) {
            if prop.get("type").and_then(|v| v.as_str()) == Some("string") {
                string_keys.push(key);
            }
        }
    }
    if string_keys.len() == 1 {
        Some(string_keys[0].to_string())
    } else {
        None
    }
}

/// Parses the body of a single `<function=...>` into a JSON arguments string.
/// The body may be a JSON object directly, or a set of
/// `<parameter=key>value</parameter>` tags.
pub(super) fn parse_hermes_function_body(body: &str) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        // A parameterless tool call (e.g. `<function=list_dir></function>`) is legal;
        // return an empty object.
        return Some("{}".to_string());
    }
    // Form 1: the body is itself a JSON object.
    if trimmed.starts_with('{') {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if value.is_object() {
                return Some(value.to_string());
            }
        }
    }
    // Form 2: a set of <parameter=key>value</parameter> tags.
    if trimmed.contains("<parameter=") {
        let mut map = serde_json::Map::new();
        let mut rest = trimmed;
        while let Some(open_rel) = rest.find("<parameter=") {
            let after_open = &rest[open_rel + "<parameter=".len()..];
            let Some(key_end) = after_open.find('>') else {
                break;
            };
            let key = after_open[..key_end].trim().to_string();
            let value_region = &after_open[key_end + 1..];
            let Some(close_rel) = value_region.find("</parameter>") else {
                break;
            };
            let raw_value = value_region[..close_rel].trim();
            // Try to parse the value as a JSON scalar/structure (number, bool, object,
            // array); otherwise treat it as a string.
            let value = serde_json::from_str::<serde_json::Value>(raw_value)
                .unwrap_or_else(|_| serde_json::Value::String(raw_value.to_string()));
            if !key.is_empty() {
                map.insert(key, value);
            }
            rest = &value_region[close_rel + "</parameter>".len()..];
        }
        if !map.is_empty() {
            return Some(serde_json::Value::Object(map).to_string());
        }
    }
    None
}

/// Strips common wrapper forms emitted by models: ```json ... ```, ``` ... ```,
/// `<tool_call> ... </tool_call>`、`<|tool_call_begin|> ... <|tool_call_end|>`。
/// Strips one layer only when the whole input is wrapped like this; otherwise
/// returns it unchanged.
fn strip_inline_tool_call_wrappers(text: &str) -> String {
    let mut s = text.trim().to_string();
    // markdown fenced code block
    if let Some(rest) = s.strip_prefix("```") {
        if let Some(end) = rest.rfind("```") {
            let inner = &rest[..end];
            // Remove a possible language tag on the first line (json / JSON)
            let inner_trimmed = inner.trim_start();
            let inner_no_lang = inner_trimmed
                .strip_prefix("json")
                .or_else(|| inner_trimmed.strip_prefix("JSON"))
                .unwrap_or(inner_trimmed);
            s = inner_no_lang.trim().to_string();
        }
    }
    // <tool_call>...</tool_call>
    if let Some(rest) = s.strip_prefix("<tool_call>") {
        if let Some(end) = rest.rfind("</tool_call>") {
            s = rest[..end].trim().to_string();
        }
    }
    // <|tool_call_begin|>...<|tool_call_end|>
    if let Some(rest) = s.strip_prefix("<|tool_call_begin|>") {
        if let Some(end) = rest.rfind("<|tool_call_end|>") {
            s = rest[..end].trim().to_string();
        }
    }
    s
}

pub(super) fn normalize_tool_call_arguments(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Some("{}".to_string());
    }
    // Standard path: the whole input is valid JSON.
    if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
        return Some(trimmed.to_string());
    }
    // Some providers (qwen3.7 etc.) mix XML parameter tags into
    // delta.tool_calls.arguments (e.g.
    // `{"k":"v"}</parameter><parameter-langs>...</parameter></function>`).
    // Try extracting the arguments with the Hermes body parser.
    if trimmed.contains("<parameter=") || trimmed.contains("</parameter>") {
        if let Some(args) = parse_hermes_function_body(trimmed) {
            return Some(args);
        }
    }
    // Try to slice a JSON object prefix: starting at '{', find the last matching '}'.
    if trimmed.starts_with('{') {
        if let Some(end) = find_json_object_end(trimmed) {
            let candidate = &trimmed[..=end];
            if serde_json::from_str::<serde_json::Value>(candidate).is_ok() {
                return Some(candidate.to_string());
            }
        }
    }
    None
}

/// Starting from the `{` at the beginning of the string, tracks brace nesting depth
/// while skipping string literals, and returns the index of the matching `}`.
fn find_json_object_end(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Collects the valid tool calls for this turn. Returns
/// `(tool call list, whether anything was dropped)`: a tool call whose arguments
/// JSON is incomplete (typically a large `write_file` truncated by the output
/// limit) and cannot be repaired is dropped with `dropped=true`, letting callers
/// distinguish "truncated" from "no tool calls this turn".
pub(super) fn collect_valid_tool_calls(
    builders: &mut rust_tools::cw::SkipMap<usize, ToolCallBuilder>,
) -> (Vec<ToolCall>, bool) {
    let mut dropped = false;
    let tool_calls = builders
        .drain()
        .filter_map(|(_, mut builder)| {
            let Some(arguments) = normalize_tool_call_arguments(&builder.arguments) else {
                dropped = true;
                // Print the truncated arguments fragment to ease debugging "why was it
                // truncated".
                // arguments can be large (large-file write_file); show only the first
                // and last 300 chars.
                let raw = &builder.arguments;
                let char_count = raw.chars().count();
                let snippet = if char_count > 600 {
                    // Slice at char boundaries to avoid panicking by cutting a
                    // multi-byte UTF-8 character (e.g. Chinese) in the middle.
                    let head: String = raw.chars().take(300).collect();
                    let tail: String = raw
                        .chars()
                        .rev()
                        .take(300)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect();
                    format!("{}…[截断，共 {} 字符]…{}", head, char_count, tail)
                } else {
                    raw.to_string()
                };
                if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
                    eprintln!(
                        "[Warning] dropping malformed tool call '{}' due to incomplete JSON arguments\n\
                         └─ 截断的 arguments 片段:\n{}",
                        builder.function_name, snippet
                    );
                }
                return None;
            };
            builder.arguments = arguments;
            Some(builder.build())
        })
        .collect();
    (tool_calls, dropped)
}

pub(super) fn ensure_tool_calls_section_open(
    _app: &mut App,
    markers: &StreamMarkers,
    state: &mut StreamProcessingState,
) {
    if state.render.printed_tool_calls_header {
        return;
    }

    if !crate::ai::driver::runtime_ctx::terminal_output_enabled() {
        state.content.thinking_open = false;
        state.render.printed_tool_calls_header = true;
        return;
    }

    let _ = clear_waiting_hint(state);

    if state.content.thinking_open {
        // If fold mode is active, render the fold ending first
        if state.render.thinking_fold.active {
            let _ = finalize_thinking_fold(state);
        } else {
            let _ = write_stream_content(
                &format_end_thinking_line(markers, &state.render.markdown),
                &mut state.render.markdown,
                false,
            );
        }
        state.content.thinking_open = false;
    }
    let _ = state.render.markdown.flush_pending();
    state.render.printed_tool_calls_header = true;
}
