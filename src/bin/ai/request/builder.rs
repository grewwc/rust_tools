//! 请求体构建 + token 预算估算。
//!
//! 从 request/mod.rs 提取的构建逻辑：
//! - 图片/文本 content 构建（多模态支持）
//! - 请求体组装（model/messages/tools/thinking/stream/max_tokens 等）
//! - prompt token 估算与 max_tokens 钳制（防止 prompt+输出撞爆窗口）

use std::fs;

use base64::Engine as _;
use serde_json::{json, Value};

use crate::ai::{
    files,
    history::Message,
    models,
    provider::adapter_for,
};
use super::reasoning::resolve_reasoning_wire_controls;
use super::RequestBody;

/// 构建消息 content：纯文本模型或无图片时返回字符串；多模态模型返回
/// `[{image_url}, {text}]` 数组。图片以 base64 data URI 内联。
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

/// 中英文 + 代码混合语料下的保守「字符 -> token」换算：每 token 约 2 个字符。
/// 偏保守（高估 prompt 占用），宁可提前钳小输出上限，也不让 prompt + 输出一起
/// 撞爆窗口。
const CHARS_PER_TOKEN_CONSERVATIVE: usize = 2;
/// 钳制后为输出保留的最小 token 数：即便 prompt 已接近窗口，也保证有一段可见
/// 输出空间，避免下发一个过小甚至为 0 的 max_tokens 反而立刻截断。
pub(super) const MIN_OUTPUT_TOKENS_FLOOR: u32 = 1_024;
/// 为 provider 的隐性开销（模板 token、role 分隔符、思考链预留等）预留的安全余量。
const CONTEXT_WINDOW_SAFETY_MARGIN_TOKENS: usize = 2_048;

/// 估算 messages 的 prompt token 数（保守：高估）。无服务端 usage 反馈时以字符
/// 数近似（每 token ~2 字符）。
fn estimate_prompt_tokens(messages: &[Message]) -> usize {
    let chars: usize = messages
        .iter()
        .map(|m| super::super::history::value_to_string(&m.content).chars().count())
        .sum();
    chars.div_ceil(CHARS_PER_TOKEN_CONSERVATIVE)
}

/// 估算工具 schema 的 prompt token 数。工具定义（name/description/JSON Schema）
/// 会随每次请求发送并计入 prompt 占用；启用大量工具/MCP 时体积可观。以序列化后
/// 的字符数按同一保守换算折算。`None` / 空工具集贡献 0。
fn estimate_tools_tokens(tools: Option<&Value>) -> usize {
    let Some(tools) = tools else {
        return 0;
    };
    let chars = serde_json::to_string(tools)
        .map(|s| s.chars().count())
        .unwrap_or(0);
    chars.div_ceil(CHARS_PER_TOKEN_CONSERVATIVE)
}

/// 按「剩余上下文窗口」钳制单次请求的输出上限：
/// `min(model_max, window - est_prompt - safety_margin)`，并 floor 到
/// [`MIN_OUTPUT_TOKENS_FLOOR`]。这样即使模型声明了很大的 max_output_tokens，
/// 在高占用 prompt 下也不会 prompt + 输出一起超过 token 窗口。
pub(crate) fn clamp_max_tokens_for_prompt(
    model: &str,
    messages: &[Message],
    tools: Option<&Value>,
    model_max: u32,
    known_prompt_tokens: Option<u64>,
) -> u32 {
    let window = models::context_window_tokens(model);
    // 本轮实际消息量的字符估算（保守：每 token ~2 字符）。工具 schema 也会随请求
    // 一起发送、占用 prompt 窗口，故把其序列化长度折算进 prompt token--启用大量
    // 工具/MCP 时不计入会显著高估可用输出预算，导致 prompt+输出撞爆窗口。
    let est_prompt = estimate_prompt_tokens(messages) + estimate_tools_tokens(tools);
    // 优先使用服务端返回的实际 prompt_tokens，比字符估算精确得多。但该值来自
    // *上一轮* 请求：若这一轮刚发生历史压缩，prompt 骤降，而回填的 known 仍是
    // 压缩前的高值--直接用它会把 remaining 误算成接近 0，clamp 触底到
    // MIN_OUTPUT_TOKENS_FLOOR(1024)。always-thinking 模型(GLM)拿 1024 预算会被
    // reasoning 全部吃光 -> completion=0 可见文本 -> 截断重试死循环。
    // 因此对 known 设上界：正常轮 known ≈ est + tools(~1.2x est)，压缩轮 known 会
    // 远超 est(数倍)。超过 2×est 判定为陈旧，回退到本轮字符估算。
    let est_prompt = match known_prompt_tokens.map(|p| p as usize) {
        Some(known) if est_prompt > 0 => known.min(est_prompt.saturating_mul(2)),
        Some(known) => known,
        None => est_prompt,
    };
    let remaining = window
        .saturating_sub(est_prompt)
        .saturating_sub(CONTEXT_WINDOW_SAFETY_MARGIN_TOKENS);
    let remaining = u32::try_from(remaining).unwrap_or(u32::MAX);
    model_max.min(remaining).max(MIN_OUTPUT_TOKENS_FLOOR)
}

/// 组装 HTTP 请求体（`RequestBody`）。按模型能力注入 thinking / reasoning /
/// search / stream_options / max_tokens 等字段，并钳制 max_tokens 防止超窗口。
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
    known_prompt_tokens: Option<u64>,
) -> RequestBody<'a> {
    let adapter_kind = models::model_adapter(model);
    let endpoint = models::endpoint_for_model(model, "");
    let adapter = adapter_for(adapter_kind, &endpoint);
    let request_model = models::request_model_name(model);
    let (thinking, reasoning_effort, reasoning) =
        resolve_reasoning_wire_controls(model, &endpoint, enable_thinking, reasoning_effort);
    // 流式请求显式索取 usage：部分 adapter（DashScope compatible-mode）流式下
    // 默认不返回 usage，必须声明 stream_options.include_usage 才能统计 token。
    let stream_options = stream.then(|| json!({ "include_usage": true }));
    // 仅在模型声明了 max_output_tokens 时下发 max_tokens；并按「剩余上下文窗口」
    // 钳制，避免 prompt + 请求的输出上限一起挤爆 token 窗口（GLM 长上下文下反复
    // 截断 -> 重试死循环的根因）。未声明 max_output_tokens 的模型保持不下发字段，
    // wire 行为不变。
    let max_tokens = models::max_output_tokens(model).map(|model_max| {
        clamp_max_tokens_for_prompt(
            model,
            messages,
            tools.as_ref(),
            model_max,
            known_prompt_tokens,
        )
    });
    // 零输出截断自适应：当上一轮检测到 completion=0 + finish_reason=length 时，
    // orchestrator 会把 max_tokens_override 设为更小的值。此处用该值替换 clamp 结果，
    // 让下一轮请求发送更小的 max_tokens，绕过服务端对超大 max_tokens 的空响应拒绝。
    let max_tokens = match (max_tokens, max_tokens_override) {
        (Some(_), Some(override_val)) => Some(override_val),
        (mt, _) => mt,
    };
    RequestBody {
        model: request_model,
        messages,
        stream,
        thinking,
        enable_search: adapter.enable_search_field(enable_search),
        tools,
        tool_choice,
        reasoning_effort,
        reasoning,
        stream_options,
        max_tokens,
    }
}
