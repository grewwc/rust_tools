//! 流式响应解析类型与辅助函数。
//!
//! 包含 OpenAI-compatible 流式响应的反序列化结构（`StreamChunk` / `StreamUsage`
//! / `StreamChoice` 等）、serde 辅助函数以及推理片段合并逻辑。
//! 这些类型被 `request` 内部各子模块以及 `stream` 渲染层共享。

use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::ai::history::Message;
use rustc_hash::FxHashMap;

#[derive(Debug, Serialize)]
pub(crate) struct RequestBody<'a> {
    pub(crate) model: String,
    pub(crate) messages: &'a [Message],
    pub(crate) stream: bool,
    /// 思考开关的线缆字段，由 `thinking` 方言模块决定具体 key 与形状
    /// （`enable_thinking: bool` / `thinking: {"type":...}` / 或空）。
    /// 核心层只持有 provider 无关的 Map，wire 编码完全归属方言。
    #[serde(flatten)]
    pub(crate) thinking: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) enable_search: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_choice: Option<Value>,
    /// OpenAI / OpenRouter / OpenCode 兼容协议的推理强度顶层字段。
    /// DashScope compatible provider 使用下方的嵌套 `reasoning.effort`。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning: Option<Value>,
    /// 流式请求显式索取 usage 统计：`{ "include_usage": true }`。
    /// 部分 provider（已知 DashScope compatible-mode）流式下默认不返回 `usage`，
    /// 必须显式声明，否则 token 用量无法统计、`/usage` 会漏计。非流式时为 None。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stream_options: Option<Value>,
    /// 单次响应最大输出 token 数（来自 models.json 的 `max_output_tokens`）。
    /// 缺省不下发，沿用 provider 默认补全上限；显式指定可缓解长输出被截断。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_tokens: Option<u32>,
    /// 当前 turn 捕获的 Responses `reasoning` output items 侧信道（key = 带
    /// tool_calls 的 assistant 消息首个 tool_call id）。仅 Responses 方言读取，
    /// 用于在对应 function_call 前原样回放 encrypted reasoning。`#[serde(skip)]`：
    /// 不参与 chat-completions 序列化，也绝不落盘。
    #[serde(skip)]
    pub(crate) reasoning_items: Option<&'a FxHashMap<String, Vec<Value>>>,
    /// 模型能力位（来自 models.json `reasoning_encrypted_replay`）：开启后
    /// Responses 请求带 `include: ["reasoning.encrypted_content"]` 索取加密推理项。
    /// 在 builder 阶段用原始 model key 解析，避免 protocol 层用（可能加密的）
    /// request model name 反查失败。`#[serde(skip)]`：非 chat-completions 字段。
    #[serde(skip)]
    pub(crate) reasoning_encrypted_replay: bool,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct StreamChunk {
    #[serde(default, deserialize_with = "vec_or_default")]
    pub(crate) choices: Vec<StreamChoice>,
    /// OpenAI-compatible usage block. Only present on the final chunk
    /// (and only if `stream_options: { include_usage: true }` was requested,
    /// though many providers include it unconditionally).
    #[serde(default)]
    pub(crate) usage: Option<StreamUsage>,
    /// Some providers echo the model name on every chunk.
    #[serde(default, deserialize_with = "string_or_default")]
    pub(crate) model: String,
}

/// OpenAI-compatible `usage` object. We intentionally keep it permissive
/// (all optional / default=0) so that varied providers do not break parsing.
///
/// Field-name compatibility: providers diverge on naming. Besides the canonical
/// OpenAI `prompt_tokens` / `completion_tokens`, we accept the Anthropic-style
/// `input_tokens` / `output_tokens` (and a few common spellings) via serde
/// `alias`, so that non-OpenAI-shaped responses are not silently counted as 0.
/// After deserialization, always call [`StreamUsage::normalized`] to fill in
/// components that the provider omitted but can be derived from `total_tokens`.
#[derive(Debug, Default, Clone, Deserialize)]
pub(crate) struct StreamUsage {
    #[serde(default, alias = "input_tokens", alias = "prompt_token_count")]
    pub(crate) prompt_tokens: u64,
    #[serde(default, alias = "output_tokens", alias = "completion_token_count")]
    pub(crate) completion_tokens: u64,
    #[serde(default, alias = "total_token_count")]
    pub(crate) total_tokens: u64,
    /// OpenAI newer format: prompt_tokens_details.cached_tokens
    #[serde(default, alias = "input_tokens_details")]
    pub(crate) prompt_tokens_details: Option<StreamPromptTokensDetails>,
    /// OpenAI reasoning models / qwen thinking mode report the reasoning slice
    /// here. Most providers already fold this into `completion_tokens`, so we
    /// only use it as a floor (see [`StreamUsage::normalized`]) to avoid
    /// double-counting while still recovering output tokens when a provider
    /// reports reasoning separately and leaves `completion_tokens` at 0.
    #[serde(default, alias = "output_tokens_details")]
    pub(crate) completion_tokens_details: Option<StreamCompletionTokensDetails>,
}

impl StreamUsage {
    /// Backfill omitted token components from whatever the provider did report,
    /// so downstream accounting does not under-count.
    ///
    /// Rules (all conservative, never inflate beyond reported totals):
    /// - If exactly one of prompt/completion is missing but `total_tokens`
    ///   covers the other, derive the missing component as the remainder.
    /// - If `completion_tokens` is 0 but a separate `reasoning_tokens` slice is
    ///   present, treat reasoning tokens as the output floor.
    /// - Keep `total_tokens` consistent (>= prompt + completion) for display.
    pub(crate) fn normalized(mut self) -> Self {
        // Reasoning-only providers: recover output tokens from the details slice.
        if self.completion_tokens == 0 {
            if let Some(reasoning) = self
                .completion_tokens_details
                .as_ref()
                .map(|d| d.reasoning_tokens)
                .filter(|&r| r > 0)
            {
                self.completion_tokens = reasoning;
            }
        }

        // Derive a missing component from the total when the total is larger
        // than the single component we do have.
        if self.total_tokens > 0 {
            if self.completion_tokens == 0 && self.prompt_tokens > 0 {
                self.completion_tokens = self.total_tokens.saturating_sub(self.prompt_tokens);
            } else if self.prompt_tokens == 0 && self.completion_tokens > 0 {
                self.prompt_tokens = self.total_tokens.saturating_sub(self.completion_tokens);
            }
        }

        // Keep total at least the sum of the parts for honest display.
        let sum = self.prompt_tokens.saturating_add(self.completion_tokens);
        if self.total_tokens < sum {
            self.total_tokens = sum;
        }
        self
    }
}

#[derive(Debug, Default, Clone, Deserialize)]
pub(crate) struct StreamPromptTokensDetails {
    #[serde(default)]
    pub(crate) cached_tokens: u64,
}

#[derive(Debug, Default, Clone, Deserialize)]
pub(crate) struct StreamCompletionTokensDetails {
    #[serde(default, alias = "thinking_tokens")]
    pub(crate) reasoning_tokens: u64,
}

impl StreamChunk {
    pub(crate) fn merge_reasoning(&mut self) {
        for choice in &mut self.choices {
            if choice.delta.reasoning_content.is_empty()
                && !choice.message.reasoning_content.is_empty()
            {
                choice.delta.reasoning_content = choice.message.reasoning_content.clone();
            }
            if choice.delta.content.is_empty() && !choice.message.content.is_empty() {
                choice.delta.content = std::mem::take(&mut choice.message.content);
            }
            if choice.delta.tool_calls.is_empty() && !choice.message.tool_calls.is_empty() {
                choice.delta.tool_calls = std::mem::take(&mut choice.message.tool_calls);
            }
            if choice.delta.reasoning_content.is_empty() && !choice.reasoning_content.is_empty() {
                choice.delta.reasoning_content = choice.reasoning_content.clone();
            }
            if choice.delta.reasoning_details.is_empty() && !choice.reasoning_details.is_empty() {
                choice.delta.reasoning_details = choice.reasoning_details.clone();
            }
            choice.delta.reasoning_content = merge_reasoning_fragments(
                &choice.delta.reasoning_details,
                &choice.delta.reasoning_content,
            );
            choice.delta.reasoning_details.clear();
        }
    }
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct StreamChoice {
    #[serde(default)]
    pub(crate) delta: StreamDelta,
    /// 少数 OpenAI-compatible 网关会把快照块放在 `message` 而不是 `delta`。
    /// 统一折叠进 delta，后续流处理层不需要认识这种 wire 差异。
    #[serde(default)]
    pub(crate) message: StreamDelta,
    #[serde(
        default,
        alias = "reasoning",
        alias = "reasoning_text",
        deserialize_with = "reasoning_content_string_or_default"
    )]
    pub(crate) reasoning_content: String,
    #[serde(default, deserialize_with = "reasoning_details_string_or_default")]
    pub(crate) reasoning_details: String,
    #[serde(default)]
    pub(crate) finish_reason: Option<String>,
}

#[derive(Debug, Default)]
pub(crate) struct StreamDelta {
    pub(crate) content: String,
    pub(crate) reasoning_content: String,
    pub(crate) reasoning_details: String,
    pub(crate) tool_calls: Vec<StreamToolCall>,
}

impl<'de> Deserialize<'de> for StreamDelta {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Option::<Value>::deserialize(deserializer)?;
        Ok(value
            .as_ref()
            .map(stream_delta_from_value)
            .unwrap_or_default())
    }
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct StreamToolCall {
    #[serde(default)]
    pub(crate) index: usize,
    #[serde(default, deserialize_with = "string_or_default")]
    pub(crate) id: String,
    #[serde(rename = "type", default, deserialize_with = "string_or_default")]
    pub(crate) tool_type: String,
    #[serde(default)]
    pub(crate) function: StreamFunctionCall,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct StreamFunctionCall {
    #[serde(default, deserialize_with = "string_or_default")]
    pub(crate) name: String,
    #[serde(default, deserialize_with = "string_or_default")]
    pub(crate) arguments: String,
}

fn vec_or_default<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    Option::<Vec<T>>::deserialize(deserializer).map(|opt| opt.unwrap_or_default())
}

fn string_or_default<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value
        .as_ref()
        .map(json_value_to_string_lossy)
        .unwrap_or_default())
}

fn reasoning_details_string_or_default<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value
        .as_ref()
        .map(extract_reasoning_details_text)
        .unwrap_or_default())
}

fn reasoning_content_string_or_default<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value
        .as_ref()
        .map(extract_reasoning_details_text)
        .unwrap_or_default())
}

pub(super) fn extract_displayable_text(value: &Value) -> String {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .map(extract_displayable_text)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(""),
        Value::Object(map) => extract_text_from_object(map, &["text", "content", "delta"]),
    }
}

fn stream_delta_from_value(value: &Value) -> StreamDelta {
    let Some(map) = value.as_object() else {
        return StreamDelta {
            content: extract_displayable_text(value),
            ..Default::default()
        };
    };

    let (content, reasoning_from_content) = map
        .get("content")
        .map(extract_structured_stream_content_channels)
        .unwrap_or_default();
    let reasoning_content = [
        "reasoning_content",
        "reasoning",
        "reasoning_text",
        "summary_text",
        "summary",
    ]
    .iter()
    .filter_map(|key| map.get(*key))
    .map(extract_reasoning_details_text)
    .find(|text| !text.is_empty())
    .unwrap_or_default();
    let reasoning_content = merge_reasoning_fragments(&reasoning_content, &reasoning_from_content);

    let reasoning_details = map
        .get("reasoning_details")
        .map(extract_reasoning_details_text)
        .unwrap_or_default();
    let tool_calls = map
        .get("tool_calls")
        .and_then(|value| serde_json::from_value::<Vec<StreamToolCall>>(value.clone()).ok())
        .unwrap_or_default();

    StreamDelta {
        content,
        reasoning_content,
        reasoning_details,
        tool_calls,
    }
}

fn extract_structured_stream_content_channels(value: &Value) -> (String, String) {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => (String::new(), String::new()),
        Value::String(s) => (s.clone(), String::new()),
        Value::Array(items) => {
            let mut content = String::new();
            let mut reasoning = String::new();
            for item in items {
                let (item_content, item_reasoning) =
                    extract_structured_stream_content_channels(item);
                content.push_str(&item_content);
                reasoning.push_str(&item_reasoning);
            }
            (content, reasoning)
        }
        Value::Object(map) => {
            let item_type = map
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase();
            if matches!(
                item_type.as_str(),
                "summary_text" | "reasoning" | "reasoning_text" | "reasoning_content"
            ) {
                return (String::new(), extract_reasoning_details_text(value));
            }
            for key in ["text", "content", "delta", "refusal"] {
                if let Some(inner) = map.get(key) {
                    let (content, _) = extract_structured_stream_content_channels(inner);
                    if !content.is_empty() {
                        return (content, String::new());
                    }
                }
            }
            (String::new(), String::new())
        }
    }
}

fn extract_reasoning_details_text(value: &Value) -> String {
    match value {
        Value::Array(items) => items
            .iter()
            .map(extract_reasoning_details_text)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(""),
        Value::Object(map) => extract_text_from_object(
            map,
            &[
                "text",
                "content",
                "delta",
                "summary_text",
                "reasoning_text",
                "reasoning",
                "summary",
            ],
        ),
        Value::String(s) => s.clone(),
        _ => String::new(),
    }
}

fn extract_text_from_object(
    map: &serde_json::Map<String, Value>,
    preferred_keys: &[&str],
) -> String {
    for key in preferred_keys {
        if let Some(inner) = map.get(*key) {
            let extracted = match *key {
                "reasoning" | "summary" => extract_reasoning_details_text(inner),
                _ => extract_displayable_text(inner),
            };
            if !extracted.is_empty() {
                return extracted;
            }
        }
    }
    String::new()
}

pub(crate) fn merge_reasoning_fragments(details: &str, content: &str) -> String {
    if content.is_empty() {
        return details.to_string();
    }
    if details.is_empty() {
        return content.to_string();
    }
    if content.contains(details) {
        return content.to_string();
    }
    if details.contains(content) {
        return details.to_string();
    }

    let overlap = longest_suffix_prefix_overlap(details, content);
    if overlap > 0 {
        return format!("{}{}", details, &content[overlap..]);
    }

    let content_stripped = content.trim_start();
    let stripped_overlap = longest_suffix_prefix_overlap(details, content_stripped);
    if stripped_overlap > 0 {
        return format!("{}{}", details, &content_stripped[stripped_overlap..]);
    }

    if looks_like_continuation(details, content_stripped) {
        return format!("{details}{content}");
    }

    content.to_string()
}

fn looks_like_continuation(prefix: &str, continuation: &str) -> bool {
    let first = match continuation.chars().next() {
        Some(c) => c,
        None => return false,
    };

    if is_continuation_punctuation(first) {
        return true;
    }

    if first == '\'' {
        let rest = &continuation[first.len_utf8()..];
        if rest.starts_with('s')
            || rest.starts_with('t')
            || rest.starts_with("re ")
            || rest.starts_with("ve ")
            || rest.starts_with("ll ")
            || rest.starts_with("d ")
            || rest.starts_with("m ")
        {
            return true;
        }
    }

    if continuation.starts_with("n't") {
        let prefix_stripped = prefix.trim_end();
        if prefix_stripped.ends_with("is")
            || prefix_stripped.ends_with("was")
            || prefix_stripped.ends_with("were")
            || prefix_stripped.ends_with("would")
            || prefix_stripped.ends_with("could")
            || prefix_stripped.ends_with("should")
            || prefix_stripped.ends_with("do")
            || prefix_stripped.ends_with("does")
            || prefix_stripped.ends_with("did")
            || prefix_stripped.ends_with("has")
            || prefix_stripped.ends_with("have")
            || prefix_stripped.ends_with("had")
            || prefix_stripped.ends_with("ca")
        {
            return true;
        }
    }

    false
}

fn is_continuation_punctuation(ch: char) -> bool {
    matches!(
        ch,
        ',' | '.'
            | ';'
            | ':'
            | '!'
            | '?'
            | ')'
            | ']'
            | '}'
            | '，'
            | '。'
            | '；'
            | '：'
            | '！'
            | '？'
            | '）'
            | '】'
            | '》'
            | '…'
    )
}

fn longest_suffix_prefix_overlap(left: &str, right: &str) -> usize {
    let mut candidates = right.char_indices().map(|(idx, _)| idx).collect::<Vec<_>>();
    candidates.push(right.len());
    candidates.reverse();

    for overlap in candidates {
        if overlap == 0 || overlap > left.len() {
            continue;
        }
        if left.ends_with(&right[..overlap]) {
            return overlap;
        }
    }
    0
}

fn json_value_to_string_lossy(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Array(items) => items
            .iter()
            .map(json_value_to_string_lossy)
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(""),
        Value::Object(map) => {
            for key in [
                "text",
                "content",
                "value",
                "reasoning_content",
                "reasoning",
                "arguments",
            ] {
                if let Some(inner) = map.get(key) {
                    let extracted = json_value_to_string_lossy(inner);
                    if !extracted.is_empty() {
                        return extracted;
                    }
                }
            }
            serde_json::to_string(value).unwrap_or_default()
        }
    }
}
