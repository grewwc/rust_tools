//! 内联工具调用恢复与验证。
//!
//! 处理模型输出中非标准格式的工具调用：
//! - `InlineToolCallParser` / `INLINE_PARSERS`：注册的解析器列表
//! - `normalize_inline_tool_call_markup`：归一化命名空间前缀 XML 标签
//! - `recover_inline_tool_calls`：从纯文本中恢复工具调用
//! - `recover_json_tool_calls` / `recover_hermes_xml_tool_calls` / `recover_anthropic_xml_tool_calls`
//!   / `recover_bare_xml_tool_calls`：各格式解析器
//! - `strip_inline_tool_call_wrappers`：移除工具调用包装标签
//! - `normalize_tool_call_arguments` / `find_json_object_end`：参数验证
//! - `collect_valid_tool_calls` / `ensure_tool_calls_section_open`：工具调用收集与渲染

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

/// 把模型输出里的命名空间前缀 XML 标签归一化为标准 XML 标签，例如
/// `<|DSML|invoke name="x">` / `<｜｜DSML｜｜invoke name="x">` → `<invoke name="x">`，
/// `</|DSML|invoke>` / `</｜｜DSML｜｜invoke>` → `</invoke>`。
/// 这样 Hermes / Anthropic 解析器无需为每个 `<|PREFIX|>` 协议单独适配。
pub(super) fn normalize_inline_tool_call_markup(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains("<|") && !text.contains("<｜｜") && !text.contains("</｜｜") {
        return std::borrow::Cow::Borrowed(text);
    }
    static OPEN_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"<(?:\|([^>|]+)\|([^\s>]+)|｜｜([^＞>]+)｜｜([^\s>]+))"#)
            .expect("valid open-tag regex")
    });
    static CLOSE_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"</(?:\|([^>|]+)\|([^\s>]+)|｜｜([^＞>]+)｜｜([^\s>]+))>"#)
            .expect("valid close-tag regex")
    });
    let s = OPEN_RE.replace_all(text, |caps: &regex::Captures<'_>| {
        let local = caps
            .get(2)
            .or_else(|| caps.get(4))
            .map(|m| m.as_str())
            .unwrap_or("");
        format!("<{local}")
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

/// 尝试把整段 assistant 文本反向识别为一个/多个 tool_call。
/// 通过 parser 注册表 + 前置 XML 命名空间归一化，统一处理不同模型产出的
/// inline tool call 形态（Hermes XML、Anthropic XML、JSON、`<|PREFIX|>` 包装）。
/// 任一 parser 成功即返回；全部失败则视为普通文本回答。
pub(super) fn recover_inline_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let normalized = normalize_inline_tool_call_markup(text);
    for parser in INLINE_PARSERS {
        if let Some(calls) = parser(&normalized) {
            return Some(calls);
        }
    }
    None
}

/// 从 assistant 文本里识别 JSON 形态的工具调用（单个对象或数组）。
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
        // 兼容 OpenAI 风格 {"function": {"name", "arguments"}, "id"} 与
        // 简化风格 {"name", "arguments"}。
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
                    // 校验内层字符串确实是 JSON，避免把任意字符串当 args 透传。
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

/// 解析 Hermes / Qwen 风格的 XML tool call。支持：
///   - 多个 `<function=NAME> ... </function>` 块（并行工具调用）
///   - body 为 JSON：`<function=read_file>{"path":"/x"}</function>`
///   - body 为 parameter 标签：`<function=read_file><parameter=path>/x</parameter></function>`
///   - 外层可有可无 `<tool_call>...</tool_call>` 包裹
/// 任意一个 `<function=...>` 块解析成功即返回；全部失败返回 None。
fn recover_hermes_xml_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let mut out: Vec<ToolCall> = Vec::new();
    let mut rest = text;
    let mut idx = 0usize;
    while let Some(open_rel) = rest.find("<function=") {
        let after_open = &rest[open_rel + "<function=".len()..];
        // 函数名到第一个 '>' 为止。
        let Some(name_end) = after_open.find('>') else {
            break;
        };
        let name = after_open[..name_end].trim().to_string();
        let body_start = name_end + 1;
        // body 到配套 </function> 为止；缺失闭合标签时取剩余全部。
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
        // 前进到本块结束之后，继续扫描后续并行块。
        let advance = open_rel + "<function=".len() + consumed_to;
        if advance >= rest.len() {
            break;
        }
        rest = &rest[advance..];
    }
    if out.is_empty() { None } else { Some(out) }
}

/// 解析 Anthropic / Claude 风格的 XML tool call。支持：
///   - 多个 `<invoke name="NAME"> ... </invoke>` 块（并行工具调用）
///   - 参数为 `<parameter name="key">value</parameter>` 标签集合
///   - 外层可有可无 `<function_calls>` / `<tool_calls>` 包裹
///   - 标签可带命名空间前缀（如 `antml:invoke`）
/// 任意一个 `<invoke ...>` 块解析成功即返回；全部失败返回 None。
fn recover_anthropic_xml_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let mut out: Vec<ToolCall> = Vec::new();
    let mut rest = text;
    let mut idx = 0usize;
    while let Some(open_rel) = rest.find("<invoke") {
        let after_tag = &rest[open_rel..];
        // 定位本 invoke 开标签的 '>'。
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

/// 解析裸工具名 XML：`<execute_command>pwd</execute_command>`。
/// 仅对白名单里的已注册工具名生效，避免把普通 HTML/XML 标签误当工具调用。
/// 与 Hermes / Anthropic 不同，这里标签名本身就是工具名，body 既可能是 JSON
/// arguments，也可能是像 `execute_command` 这样只有一个必填字符串参数的原始文本。
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

/// 把 `<invoke>` body 里的 `<parameter name="key">value</parameter>` 解析为 JSON
/// arguments 字符串；无参数返回 `{}`。
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
            let value = serde_json::from_str::<serde_json::Value>(raw_value)
                .unwrap_or_else(|_| serde_json::Value::String(raw_value.to_string()));
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

/// 从 `<invoke name="x">` / `<parameter name="y">` 开标签里抽取 `name` 属性值，
/// 支持双引号或单引号。
fn parse_anthropic_xml_name_attr(open_tag: &str) -> String {
    let Some(pos) = open_tag.find("name") else {
        return String::new();
    };
    let after = open_tag[pos + "name".len()..].trim_start();
    let after = after.strip_prefix('=').unwrap_or(after).trim_start();
    let (quote, body) = if let Some(b) = after.strip_prefix('"') {
        ('"', b)
    } else if let Some(b) = after.strip_prefix('\'') {
        ('\'', b)
    } else {
        return String::new();
    };
    match body.find(quote) {
        Some(end) => body[..end].to_string(),
        None => String::new(),
    }
}

/// 解析裸 XML 开标签，要求标签名本身是已注册工具名，且不带属性。
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

/// 解析裸 XML 工具体。优先接受 JSON object / Hermes parameter 标签；
/// 若 body 只是原始文本，则仅对“恰好一个必填 string 参数”的工具做安全降级，
/// 例如 `<execute_command>pwd</execute_command>` → `{"command":"pwd"}`。
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
    let spec = crate::ai::tools::registry::common::get_tool_spec(tool_name)?;
    let schema = (spec.parameters)();
    let schema = schema.as_object()?;
    let required = schema.get("required")?.as_array()?;
    if required.len() != 1 {
        return None;
    }
    let key = required.first()?.as_str()?;
    let props = schema.get("properties")?.as_object()?;
    let prop = props.get(key)?.as_object()?;
    match prop.get("type")?.as_str() {
        Some("string") => Some(key.to_string()),
        _ => None,
    }
}

/// 把单个 `<function=...>` 的 body 解析为 JSON arguments 字符串。
/// body 既可能直接是 JSON 对象，也可能是若干 `<parameter=key>value</parameter>`。
pub(super) fn parse_hermes_function_body(body: &str) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        // 无参数工具调用（如 `<function=list_dir></function>`）合法，返回空对象。
        return Some("{}".to_string());
    }
    // 形态 1：body 本身就是 JSON 对象。
    if trimmed.starts_with('{') {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if value.is_object() {
                return Some(value.to_string());
            }
        }
    }
    // 形态 2：<parameter=key>value</parameter> 标签集合。
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
            // 尝试把值解析成 JSON 标量/结构（数字、bool、对象、数组）；否则当字符串。
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

/// 剥掉模型常见的包裹形态：```json ... ```、``` ... ```、
/// `<tool_call> ... </tool_call>`、`<|tool_call_begin|> ... <|tool_call_end|>`。
/// 仅当输入整体被这些包裹时才剥离一层；否则原样返回。
fn strip_inline_tool_call_wrappers(text: &str) -> String {
    let mut s = text.trim().to_string();
    // markdown fenced code block
    if let Some(rest) = s.strip_prefix("```") {
        if let Some(end) = rest.rfind("```") {
            let inner = &rest[..end];
            // 去掉首行可能的语言标签（json / JSON）
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
    // 标准路径：整体就是合法 JSON。
    if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
        return Some(trimmed.to_string());
    }
    // 部分 provider（qwen3.7 等）在 delta.tool_calls.arguments 中混入 XML
    // parameter 标签（如 `{"k":"v"}</parameter><parameter-langs>...</parameter></function>`）。
    // 尝试用 Hermes body 解析器提取参数。
    if trimmed.contains("<parameter=") || trimmed.contains("</parameter>") {
        if let Some(args) = parse_hermes_function_body(trimmed) {
            return Some(args);
        }
    }
    // 尝试截取 JSON 对象前缀：从 '{' 开始找到最后一个配对的 '}'。
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

/// 从字符串开头的 `{` 开始，追踪大括号嵌套深度，跳过字符串字面量，返回配对 `}` 的索引。
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

/// 收集本轮有效工具调用。返回 `(工具调用列表, 是否发生过丢弃)`：当某个工具调用
/// 的 arguments JSON 不完整（典型：大文件 `write_file` 撞输出上限被截断）而无法
/// 修复时会被丢弃并返回 `dropped=true`，供上层区分"截断"与"正常无工具调用"。
pub(super) fn collect_valid_tool_calls(
    builders: &mut rust_tools::cw::SkipMap<usize, ToolCallBuilder>,
) -> (Vec<ToolCall>, bool) {
    let mut dropped = false;
    let tool_calls = builders
        .drain()
        .filter_map(|(_, mut builder)| {
            let Some(arguments) = normalize_tool_call_arguments(&builder.arguments) else {
                dropped = true;
                // 打印被截断的 arguments 片段，便于排查"为什么被截断"。
                // arguments 可能很大（大文件 write_file），只显示头尾各 300 字符。
                let raw = &builder.arguments;
                let char_count = raw.chars().count();
                let snippet = if char_count > 600 {
                    // 按字符边界截取，避免在多字节 UTF-8 字符（如中文）中间切片导致 panic。
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
                eprintln!(
                    "[Warning] dropping malformed tool call '{}' due to incomplete JSON arguments\n\
                     └─ 截断的 arguments 片段:\n{}",
                    builder.function_name, snippet
                );
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

    let _ = clear_waiting_hint(state);

    if state.content.thinking_open {
        // 如果折叠模式活跃，先执行折叠结束渲染
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
