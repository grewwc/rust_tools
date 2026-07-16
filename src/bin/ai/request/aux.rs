//! 辅助 LLM 任务：历史摘要、会话标题生成、token 用量落账。
//!
//! 这些函数共享主链路的 `build_request_body` / auth 基础设施，但不参与主 turn
//! 流式，因此独立为子模块，降低 `mod.rs` 的认知负担。

use std::time::Duration;

use serde_json::Value;

use super::builder::build_request_body;
use super::types::StreamUsage;
use super::{
    api_key_for_request_model, apply_request_auth, control_model_for_aux_tasks,
    endpoint_for_request_model, extract_router_content,
};
use crate::ai::{
    history::{Message, messages_to_markdown},
    types::App,
};

/// 会话标题请求的超时（秒）。后台辅助任务，用宽松超时避免阻塞主流程。
pub(super) const SESSION_TITLE_REQUEST_TIMEOUT_SECS: u64 = 90;
pub(super) const SESSION_TITLE_BODY_TIMEOUT_SECS: u64 = 45;

/// 用 LLM 将较早的对话历史压缩成摘要文本，供 context-budget 压缩器使用。
///
/// 三段式截断（head 12k + middle keypoints 4k + tail 6k），比 head+tail
/// 二段式多保留中段的 error/fix/decision 行，避免摘要器漏掉关键改动。
pub(crate) async fn summarize_history_via_model(
    app: &App,
    messages: &[Message],
    max_chars: usize,
) -> Option<String> {
    if messages.is_empty() || max_chars == 0 {
        return None;
    }

    let transcript = messages_to_markdown(messages, &app.session_id);
    // 三段式截断：head 12k + middle 关键命中 4k + tail 6k，总计 22k 字符。
    // 比原先 head 16k + tail 6k 多保留中段的 error/fix/decision 行，避免
    // 摘要器只看见"开头任务陈述 + 末尾收尾"而漏掉中段关键改动。
    let transcript = if transcript.chars().count() > 24_000 {
        let head: String = transcript.chars().take(12_000).collect();
        let tail: String = transcript
            .chars()
            .rev()
            .take(6_000)
            .collect::<String>()
            .chars()
            .rev()
            .collect();

        // 中段关键行抽取：从 head 之后、tail 之前的中间部分挑选 error/fail/panic/
        // fix/diff/apply_patch/decision 等关键标记行，控制在 4k 字符内。
        let total_chars = transcript.chars().count();
        let mid_start_chars = 12_000usize;
        let mid_end_chars = total_chars.saturating_sub(6_000);
        let middle_segment: String = if mid_end_chars > mid_start_chars {
            transcript
                .chars()
                .skip(mid_start_chars)
                .take(mid_end_chars - mid_start_chars)
                .collect()
        } else {
            String::new()
        };
        let mut keypoints = String::new();
        let mut keypoint_chars = 0usize;
        const MID_KEYPOINTS_BUDGET: usize = 4_000;
        for line in middle_segment.lines() {
            let lower = line.to_lowercase();
            let is_key = lower.contains("error")
                || lower.contains("fail")
                || lower.contains("panic")
                || lower.contains("fix")
                || lower.contains("diff")
                || lower.contains("apply_patch")
                || lower.contains("write_file")
                || lower.contains("decision")
                || lower.contains("conclusion")
                || lower.contains("结论")
                || lower.contains("修复")
                || lower.contains("错误");
            if !is_key {
                continue;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let chunk_len = trimmed.chars().count() + 1;
            if keypoint_chars + chunk_len > MID_KEYPOINTS_BUDGET {
                break;
            }
            keypoints.push_str(trimmed);
            keypoints.push('\n');
            keypoint_chars += chunk_len;
        }

        if keypoints.trim().is_empty() {
            format!("{head}\n\n[... older transcript omitted for summary budget ...]\n\n{tail}")
        } else {
            format!(
                "{head}\n\n[... middle segment compressed; keypoints below ...]\n{keypoints}\n[... end of middle keypoints ...]\n\n{tail}"
            )
        }
    } else {
        transcript
    };

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(format!(
                "你是一个软件开发对话历史压缩器。你的任务是把较早对话压缩成后续 coding agent 能继续工作的摘要。\n\
输出要求：\n\
- 只输出纯文本，不要 markdown 代码块，不要解释。\n\
- 必须保留：用户明确要求、文件路径/函数名/工具名、关键报错、修复结论、当前工作、未完成任务。\n\
- 优先保留事实和决定，删除寒暄、重复确认、冗长日志。\n\
- 使用下面这些标题，并且每个标题下用 `- ` 开头的短行：\n\
主要请求:\n关键上下文:\n错误与修复:\n当前工作:\n待办任务:\n已知工具结论:\n\
- 如果某项没有内容，写 `- 无`。\n\
- 总长度尽量控制在 {} 个字符以内。",
                max_chars
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(format!("请压缩下面的较早对话：\n\n{}", transcript)),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let control_model = control_model_for_aux_tasks(app);
    let request_body = build_request_body(
        &control_model,
        &messages,
        false,
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let endpoint = endpoint_for_request_model(app, &control_model);
    let api_key = api_key_for_request_model(app, &control_model);
    // 历史摘要是 turn 收尾的后台辅助请求（任务边界压缩会在每次答案交付后触发）。
    // 主 client 只有 connect_timeout、没有整体 timeout，若摘要模型接受连接后迟迟
    // 不返回响应头，这里的裸 .send()/.text() 会永久阻塞、CPU 0，表现为"答案已输出
    // 但迟迟不回到提示符"的卡死。用显式超时兜底，超时即放弃摘要（保持原始历史）。
    let send_future = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send();
    let response = match tokio::time::timeout(Duration::from_secs(60), send_future).await {
        Ok(r) => r.ok()?,
        Err(_) => {
            eprintln!("[summary] timeout (60s) waiting for response headers, skipping");
            return None;
        }
    };
    if !response.status().is_success() {
        return None;
    }
    let text = match tokio::time::timeout(Duration::from_secs(30), response.text()).await {
        Ok(r) => r.ok()?,
        Err(_) => {
            eprintln!("[summary] timeout (30s) reading response body, skipping");
            return None;
        }
    };
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = extract_router_content(&v)?;
    let trimmed = content.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// 用 LLM 为当前对话生成一个简短的概括性标题（不超过 20 字）。
/// 供 session 列表和输入框顶部展示使用。
pub(crate) async fn generate_session_title_via_model(
    app: &App,
    messages: &[crate::ai::history::Message],
) -> Option<String> {
    use crate::ai::history::{is_system_like_role, value_to_string};

    if messages.is_empty() {
        return None;
    }

    // 只取最近的对话内容用于生成标题（最多 8000 字符）
    let dialog: Vec<String> = messages
        .iter()
        .filter(|m| !is_system_like_role(&m.role))
        .map(|m| {
            let role = match m.role.as_str() {
                "user" => "用户",
                "assistant" => "助手",
                "tool" => "工具",
                _ => m.role.as_str(),
            };
            // 去掉图片内容，只保留文本，避免 LLM 看到 base64 数据生成无意义的标题
            let text_only =
                super::normalize::normalize_message_content_for_text_only_model(&m.content);
            let content = value_to_string(&text_only);
            format!("{role}: {content}")
        })
        .collect();

    if dialog.is_empty() {
        return None;
    }

    let mut transcript = dialog.join("\n");
    if transcript.chars().count() > 8000 {
        transcript = transcript.chars().take(8000).collect();
    }

    let system_prompt = "你是一个对话标题生成器。根据下面的对话内容，生成一个不超过20个字的简短标题，概括对话的核心主题。\n\
要求：\n\
- 只输出标题本身，不要引号，不要解释，不要前缀。\n\
- 标题要具体、有信息量，不要太笼统。\n\
- 优先用名词短语或动宾短语。\n\
- 如果是编程相关，提到关键技术或文件名。";

    let user_prompt = format!("对话内容：\n\n{transcript}\n\n请生成标题：");

    let title_messages = vec![
        crate::ai::history::Message {
            role: "system".to_string(),
            content: serde_json::Value::String(system_prompt.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        crate::ai::history::Message {
            role: "user".to_string(),
            content: serde_json::Value::String(user_prompt),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let control_model = control_model_for_aux_tasks(app);
    let request_body = build_request_body(
        &control_model,
        &title_messages,
        false,
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let endpoint = endpoint_for_request_model(app, &control_model);
    let api_key = api_key_for_request_model(app, &control_model);

    let send_future = apply_request_auth(app.client.post(&endpoint), &endpoint, &api_key)
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send();

    let response = match tokio::time::timeout(
        std::time::Duration::from_secs(SESSION_TITLE_REQUEST_TIMEOUT_SECS),
        send_future,
    )
    .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(_)) => {
            // eprintln!("[session-title] request error: {e}");
            return None;
        }
        Err(_) => {
            // eprintln!(
            // "[session-title] timeout ({}s) sending request, skipping",
            // SESSION_TITLE_REQUEST_TIMEOUT_SECS
            // );
            return None;
        }
    };

    let status = response.status();
    if !status.is_success() {
        // eprintln!("[session-title] HTTP {status}, skipping");
        return None;
    }

    let text = match tokio::time::timeout(
        std::time::Duration::from_secs(SESSION_TITLE_BODY_TIMEOUT_SECS),
        response.text(),
    )
    .await
    {
        Ok(Ok(t)) => t,
        Ok(Err(_)) => {
            // eprintln!("[session-title] body read error: {e}");
            return None;
        }
        Err(_) => {
            // eprintln!(
            // "[session-title] timeout ({}s) reading body, skipping",
            // SESSION_TITLE_BODY_TIMEOUT_SECS
            // );
            return None;
        }
    };

    let v: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => {
            // eprintln!("[session-title] JSON parse error: {e}");
            return None;
        }
    };
    let content = match extract_router_content(&v) {
        Some(c) => c,
        None => {
            // eprintln!("[session-title] extract_router_content returned None");
            return None;
        }
    };
    let trimmed = content.trim().to_string();

    // 清理：去掉引号、去掉换行、截断到 30 字符
    let cleaned = trimmed
        .trim_matches(|c: char| {
            c == '"' || c == '「' || c == '」' || c == '\'' || c.is_whitespace()
        })
        .lines()
        .next()
        .unwrap_or("")
        .to_string();

    if cleaned.is_empty() {
        return None;
    }

    // 截断到 30 字符（中文一个字算一个 char）
    let result: String = if cleaned.chars().count() > 30 {
        cleaned.chars().take(30).collect()
    } else {
        cleaned
    };

    Some(result)
}

/// AIOS bridge: take a parsed OpenAI-compatible `StreamUsage` (plus the
/// requested model name and latency) and hand it to the kernel's LLM device
/// for accounting. This is the single chokepoint where agent-land meets
/// `/dev/llm`; every LLM call site must route through here instead of
/// dropping usage on the floor.
///
/// The kernel takes care of:
///   - converting prompt/completion tokens to cost_micros (via `llm_price`)
///   - calling `rusage_charge` so rlimit enforcement stays authoritative
///   - emitting a `trace_event("llm.account", ...)` for observability
pub(crate) fn charge_llm_usage_to_kernel(
    app: &App,
    requested_model: &str,
    usage: &StreamUsage,
    latency_ms: u64,
) -> Option<aios_kernel::primitives::LlmAccountOutcome> {
    charge_llm_usage_via_kernel(&app.os, requested_model, usage, latency_ms)
}

/// 与 [`charge_llm_usage_to_kernel`] 等价，但直接接受一个 `SharedKernel`。
/// 供没有 `App` 句柄的调用方（如后台 reflection 的 `background_call`）使用--
/// `GLOBAL_OS` 与 `App.os` 共享同一把 `Arc<Mutex<Kernel>>`，落账语义一致。
pub(crate) fn charge_llm_usage_via_kernel(
    os: &aios_kernel::kernel::SharedKernel,
    requested_model: &str,
    usage: &StreamUsage,
    latency_ms: u64,
) -> Option<aios_kernel::primitives::LlmAccountOutcome> {
    // Fast path: a zero-usage report is noise.
    if usage.prompt_tokens == 0 && usage.completion_tokens == 0 {
        return None;
    }
    let cached = usage
        .prompt_tokens_details
        .as_ref()
        .map(|d| d.cached_tokens)
        .unwrap_or(0);
    let report = aios_kernel::primitives::LlmUsageReport {
        model: requested_model.to_string(),
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        cached_prompt_tokens: cached,
        latency_ms,
    };
    // 在内核里落账（计费 + rusage + trace + 追加审计账本），同时拿出本次需要
    // drain 落库的增量记录。SQLite I/O 放到 guard 释放之后，避免持内核锁做磁盘写。
    let (outcome, drained, head) = {
        let mut guard = match os.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let pid = guard.current_process_id()?;
        let outcome = guard.llm_account(pid, report);
        let cursor = crate::ai::tools::storage::token_usage_store::drain_cursor();
        let drained = guard.llm_usage_drain_since(cursor);
        let head = guard.llm_usage_head_seq();
        (outcome, drained, head)
    };
    // best-effort 落库到独立的 token 用量统计表，失败不影响主流程。
    crate::ai::tools::storage::token_usage_store::persist_drained(&drained, head);
    Some(outcome)
}
