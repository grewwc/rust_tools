//! 视觉模型（VL）图片摘要（digest）协议。
//!
//! 背景：单个 turn 内的工具调用循环里，`messages` 是一条贯穿始终的持久数组，
//! 用户消息里内联的 base64 图片会在每一轮请求里被原样重放。对 Doubao/Ark 这类
//! 按实际 multimodal wire payload 计费 TPM 的服务端，连续 tool-call 会把同一张
//! 大图反复计入 60s 窗口，触发 429 限流（本地预算按名义 1024 字符低估，预检门
//! 拦不住）。
//!
//! 方案：第一轮把原图完整发给模型，同时要求它输出一段固定格式的“图片摘要”；
//! 之后的轮次把请求投影 `messages` 里的图片 part 替换成这段文字摘要（附原图
//! 路径），从此不再重放 base64。canonical `turn_messages`（落盘历史）永远保留
//! 原始图片，只有请求投影被改写——遵守 History-truthfulness 不可变式。
//!
//! digest 的获取有两条路径：
//! 1. 搭车：从第一轮 tool-call 响应的 `assistant_text` / `reasoning_content`
//!    里 parse 出摘要（模型按指令附带输出）。
//! 2. 兜底：若第一轮没拿到，进入第二轮前发一次专门的、禁用工具的一次性请求，
//!    强制拿到摘要（见 [`describe_image_for_digest`]）。

use std::time::Duration;

use serde_json::{Value, json};

use super::builder::{build_content, build_request_body};
use super::{
    api_key_for_request_model, apply_request_auth, endpoint_for_request_model,
    extract_router_content,
};
use crate::ai::{history::Message, types::App};

/// digest 正文起始哨兵。模型必须原样输出这一行，agent 据此定位摘要。
pub(crate) const DIGEST_BEGIN: &str = "<<<IMAGE_DIGEST>>>";
/// digest 正文结束哨兵。
pub(crate) const DIGEST_END: &str = "<<<END_IMAGE_DIGEST>>>";
/// 注入到请求用户消息里的“图片处理协议”指令的固定前缀标记。
/// 替换阶段据此识别并移除这条指令 text part（它已完成使命）。
const INSTRUCTION_TAG: &str = "[图片处理协议]";
/// 兜底请求的响应头超时（秒）。与历史摘要辅助请求保持一致的宽松兜底。
const DIGEST_REQUEST_HEADER_TIMEOUT_SECS: u64 = 60;
/// 兜底请求的响应体读取超时（秒）。
const DIGEST_REQUEST_BODY_TIMEOUT_SECS: u64 = 30;

/// 第一轮注入到用户消息里的图片处理协议指令。固定 runtime-owned 文本，不含任何
/// 用户内容。告知模型：原图只发这一轮，之后只剩这段摘要，故必须把完成任务所需
/// 的全部视觉信息写进摘要。
pub(crate) fn digest_instruction() -> String {
    format!(
        "{INSTRUCTION_TAG} 本轮会附带原始图片，但为控制 Token，后续轮次不会再发送原图。\
请在本轮回答里，用下面的固定格式输出一段“图片摘要”，把完成任务所需的全部视觉信息\
（界面结构、可见文字、代码、数值、颜色、布局与位置关系等）写清楚——后续你将只能\
依赖这段摘要，看不到原图：\n\
{DIGEST_BEGIN}\n（在这里写图片摘要）\n{DIGEST_END}\n\
你仍可以在本轮正常调用工具；摘要与工具调用可以同时输出。原图路径由系统记录，\
确有需要时可用文件工具重新读取原图。"
    )
}

/// 从模型返回文本里 parse 出 digest 正文（两个哨兵之间，trim 后非空才算命中）。
pub(crate) fn parse_digest(text: &str) -> Option<String> {
    let begin = text.find(DIGEST_BEGIN)? + DIGEST_BEGIN.len();
    let rest = &text[begin..];
    let end = rest.find(DIGEST_END)?;
    let body = rest[..end].trim();
    (!body.is_empty()).then(|| body.to_string())
}

/// 判断一个 content part 是否是图片（image_url）。
fn is_image_part(part: &Value) -> bool {
    part.get("type").and_then(Value::as_str) == Some("image_url") || part.get("image_url").is_some()
}

/// 判断某条消息 content（多模态数组）里是否仍含图片 part。
pub(crate) fn content_has_image(content: &Value) -> bool {
    content
        .as_array()
        .map(|parts| parts.iter().any(is_image_part))
        .unwrap_or(false)
}

/// 判断一个 text part 是否是我们注入的“图片处理协议”指令。
fn is_instruction_part(part: &Value) -> bool {
    part.get("text")
        .and_then(Value::as_str)
        .map(|t| t.contains(INSTRUCTION_TAG))
        .unwrap_or(false)
}

/// 构造替换图片的摘要 text 正文：显式声明图片已转文字 + 原图路径 + 摘要本体。
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

/// 把请求投影里某条用户消息 content 中的图片 part 替换为一段文字摘要。
///
/// - 丢弃所有 image_url part 与注入的指令 part；
/// - 保留其余 part（context reminder、用户问题正文等）；
/// - 追加一段 digest 文本 part（含原图路径）。
///
/// 仅在 content 是数组且确含图片时才动，返回是否发生替换。调用方只应对请求投影
/// `messages` 使用，绝不能改 canonical `turn_messages`。
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

/// 兜底：发一次专门的、禁用工具的一次性 VL 请求，强制拿到图片摘要。
///
/// 只有在“搭车”未能从第一轮响应里 parse 出 digest 时才调用。必须用能看图的
/// VL 模型（调用方传入本 turn 的实际模型），不能走 `control_model_for_aux_tasks`
/// （辅助模型可能是纯文本 intent 模型，看不到图）。
///
/// 返回摘要文本：优先取哨兵之间的正文；若模型没按格式包裹，则整段响应即摘要
/// （这是一次“只要摘要”的专用调用）。任何失败都返回 `None`，调用方据此决定
/// 是否保留原图。
pub(crate) async fn describe_image_for_digest(
    app: &App,
    model: &str,
    image_files: &[String],
) -> Option<String> {
    if image_files.is_empty() {
        return None;
    }
    // build_content 只有在模型支持图片输入时才返回多模态数组；否则退化为纯字符串，
    // 说明这个模型看不到图，兜底摘要无从谈起。
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

    // 禁用工具、非流式（与 summarize_history_via_model 一致：两个 false 分别是
    // stream / enable_thinking，其余 tools/tool_choice/reasoning 全 None）。
    let request_body = build_request_body(
        model, &messages, false, false, None, None, None, None, None, None, None,
    );
    let endpoint = endpoint_for_request_model(app, model);
    let api_key = api_key_for_request_model(app, model);
    let http_body = super::protocol::build_http_body_for_request(model, &endpoint, &request_body);

    let send_future = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
        .header("Content-Type", "application/json")
        .json(&http_body)
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
    // 专用调用：优先取哨兵正文；模型若没包裹哨兵，则整段响应即摘要。
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
        // 复现失败 session：assistant 可见文本为空，摘要落在 reasoning 里。
        let reasoning = format!("我先看图。{DIGEST_BEGIN}\ncode: fn main() {{}}\n{DIGEST_END}");
        assert_eq!(
            parse_digest(&reasoning).as_deref(),
            Some("code: fn main() {}")
        );
    }

    #[test]
    fn parse_digest_none_when_missing_or_empty() {
        assert_eq!(parse_digest("no sentinels here"), None);
        // 只有起始哨兵、没有结束哨兵 -> None。
        assert_eq!(parse_digest(&format!("{DIGEST_BEGIN} dangling")), None);
        // 哨兵之间为空 -> None。
        assert_eq!(
            parse_digest(&format!("{DIGEST_BEGIN}   {DIGEST_END}")),
            None
        );
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

        // 纯字符串 content 不含图片。
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
        // 图片 part 与指令 part 都被移除。
        assert!(!parts.iter().any(is_image_part));
        assert!(!parts.iter().any(is_instruction_part));
        // 原上下文提醒与用户问题正文保留。
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
        // 追加的 digest text 含原图路径与摘要本体。
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
        // 纯字符串同样不动。
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
        // 已无图片，第二次替换应为 no-op（避免重复追加 digest）。
        assert!(!swap_images_with_digest(&mut content, "摘要", &[]));
    }
}
