//! 辅助 LLM 任务：历史摘要、会话标题生成、token 用量落账。
//!
//! 这些函数共享主链路的 `build_request_body` / auth 基础设施，但不参与主 turn
//! 流式，因此独立为子模块，降低 `mod.rs` 的认知负担。

use std::time::{Duration, Instant};

use serde_json::Value;

use super::builder::build_request_body;
use super::types::StreamUsage;
use super::{
    apply_request_auth, control_model_for_aux_tasks, control_model_for_aux_tasks_from,
    endpoint_for_request_model, extract_router_content,
};
#[cfg(test)]
use crate::ai::history::compress::summary_input::render_summary_message;
use crate::ai::history::compress::summary_input::{
    SUMMARY_TRANSCRIPT_MAX_CHARS, build_summary_history_input,
};
use crate::ai::{
    history::{
        HistoryMessageSummarizer, Message, compress::is_preserved_user_or_image_stub,
        is_runtime_synthetic_user_message,
    },
    models,
    provider::adapter_for,
    types::{App, AppConfig},
};

/// 会话标题请求的超时（秒）。后台辅助任务，用宽松超时避免阻塞主流程。
pub(super) const SESSION_TITLE_REQUEST_TIMEOUT_SECS: u64 = 90;
pub(super) const SESSION_TITLE_BODY_TIMEOUT_SECS: u64 = 45;
const HISTORY_SUMMARY_USER_PREFIX: &str = "Please compress the earlier conversation below:\n\n";

/// Caller-provided runtime pieces an auxiliary model request needs.
///
/// Deliberately narrower than `App`: the serve daemon generates a session
/// title once the turn child has exited, where no `App` exists but the config,
/// an HTTP client, and the session id do.
pub(crate) struct AuxRequestContext<'a> {
    pub(crate) config: &'a AppConfig,
    pub(crate) client: &'a reqwest::Client,
    pub(crate) session_id: &'a str,
}

/// 辅助请求（标题/摘要）的 API key 轮换候选列表。
///
/// 与主请求链路（transport.rs）一致：primary key 解析后，再经
/// adapter.collect_api_keys 收集 provider 专属命名 key（如 opencode.api_key_xxx）。
/// 此前辅助请求只用 primary key，命名 key 配置下会回退到全局 api_key，该 key 对
/// 网关失效时 401 静默失败 → session 无标题/无摘要（f319d490 / 9833f002）。
fn aux_request_key_candidates(model: &str, endpoint: &str, global_fallback: &str) -> Vec<String> {
    let primary = models::api_key_for_model(model, global_fallback);
    adapter_for(models::model_adapter(model), endpoint).collect_api_keys(&primary)
}

/// Send an auxiliary LLM request (POST chat/completions) with key rotation until one succeeds.
///
/// Each key gets its own timeout so one hung key cannot stall a background
/// auxiliary task. `overall_deadline` optionally bounds the whole rotation so
/// callers cannot multiply their latency budget by the number of keys. Returns
/// the 2xx response body; when every key fails it returns the last error as
/// (kind, message), with the same kind classification as
/// [`record_title_failure`].
pub(super) async fn send_aux_chat_request_with_key_rotation(
    ctx: &AuxRequestContext<'_>,
    model: &str,
    endpoint: &str,
    http_body: Vec<u8>,
    header_timeout: Duration,
    body_timeout: Duration,
    overall_deadline: Option<Instant>,
) -> Result<String, (String, String)> {
    let keys = aux_request_key_candidates(model, endpoint, &ctx.config.api_key);
    send_aux_chat_request_for_keys(
        ctx,
        endpoint,
        http_body,
        &keys,
        header_timeout,
        body_timeout,
        overall_deadline,
    )
    .await
}

/// Send one auxiliary request with only the highest-priority configured key.
///
/// Automatic history compression uses this path because a failed attempt must
/// preserve the current projection rather than fan the same summary out across
/// fallback keys. Other auxiliary tasks retain key rotation through
/// [`send_aux_chat_request_with_key_rotation`].
async fn send_aux_chat_request_once(
    ctx: &AuxRequestContext<'_>,
    model: &str,
    endpoint: &str,
    http_body: Vec<u8>,
    header_timeout: Duration,
    body_timeout: Duration,
) -> Result<String, (String, String)> {
    let keys = aux_request_key_candidates(model, endpoint, &ctx.config.api_key);
    let Some(api_key) = keys.first() else {
        return Err((
            "request_error".to_string(),
            "no API key candidate available".to_string(),
        ));
    };
    send_aux_chat_request_for_keys(
        ctx,
        endpoint,
        http_body,
        std::slice::from_ref(api_key),
        header_timeout,
        body_timeout,
        None,
    )
    .await
}

async fn send_aux_chat_request_for_keys(
    ctx: &AuxRequestContext<'_>,
    endpoint: &str,
    http_body: Vec<u8>,
    keys: &[String],
    header_timeout: Duration,
    body_timeout: Duration,
    overall_deadline: Option<Instant>,
) -> Result<String, (String, String)> {
    let mut last_err: (String, String) = ("http_error".to_string(), "unknown".to_string());

    for (idx, api_key) in keys.iter().enumerate() {
        if overall_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            last_err = (
                "request_timeout".to_string(),
                "auxiliary request deadline exceeded".to_string(),
            );
            break;
        }
        if idx > 0 {
            super::emit_request_diagnostic(format_args!(
                "[aux] key #{} failed, trying next key #{} ({} remaining)",
                idx - 1,
                idx,
                keys.len() - idx
            ));
        }

        let send_future =
            apply_request_auth(ctx.client.post(endpoint), endpoint, api_key, ctx.session_id)
                .header("Content-Type", "application/json")
                .body(http_body.clone())
                .send();

        let Some(header_limit) = phase_timeout(header_timeout, overall_deadline) else {
            last_err = (
                "request_timeout".to_string(),
                "auxiliary request deadline exceeded".to_string(),
            );
            break;
        };
        let response = match tokio::time::timeout(header_limit, send_future).await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                last_err = ("request_error".to_string(), e.to_string());
                continue;
            }
            Err(_) => {
                last_err = (
                    "request_timeout".to_string(),
                    format!("{}s", header_limit.as_secs()),
                );
                continue;
            }
        };

        let status = response.status();
        if !status.is_success() {
            last_err = ("http_error".to_string(), format!("HTTP {status}"));
            continue;
        }

        let Some(body_limit) = phase_timeout(body_timeout, overall_deadline) else {
            last_err = (
                "request_timeout".to_string(),
                "auxiliary request deadline exceeded".to_string(),
            );
            break;
        };
        let text = match tokio::time::timeout(body_limit, response.text()).await {
            Ok(Ok(t)) => t,
            Ok(Err(e)) => {
                last_err = ("body_read_error".to_string(), e.to_string());
                continue;
            }
            Err(_) => {
                last_err = (
                    "body_timeout".to_string(),
                    format!("{}s", body_limit.as_secs()),
                );
                continue;
            }
        };
        return Ok(text);
    }
    Err(last_err)
}

fn phase_timeout(phase: Duration, overall_deadline: Option<Instant>) -> Option<Duration> {
    match overall_deadline {
        None => Some(phase),
        Some(deadline) => {
            let remaining = deadline.saturating_duration_since(Instant::now());
            (!remaining.is_zero()).then(|| phase.min(remaining))
        }
    }
}

fn is_exported_message_heading(line: &str) -> bool {
    ["### 👤 ", "### 🤖 ", "### ⚙️ ", "### 🔧 ", "### 📝 "]
        .iter()
        .any(|prefix| line.starts_with(prefix))
}

/// 从被截断的中段提取关键行时保留消息角色，避免把助手旧结论误当成工具事实。
fn extract_middle_keypoints(
    middle_segment: &str,
    initial_heading: Option<&str>,
    char_budget: usize,
) -> String {
    let mut current_heading = initial_heading;
    let mut emitted_heading: Option<&str> = None;
    let mut keypoints = String::new();
    let mut keypoint_chars = 0usize;

    for line in middle_segment.lines() {
        if is_exported_message_heading(line) {
            current_heading = Some(line);
            continue;
        }

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
        let trimmed = line.trim();
        if !is_key || trimmed.is_empty() {
            continue;
        }

        let heading_chars = if current_heading != emitted_heading {
            current_heading.map_or(0, |heading| heading.chars().count() + 1)
        } else {
            0
        };
        let chunk_chars = heading_chars + trimmed.chars().count() + 1;
        if keypoint_chars + chunk_chars > char_budget {
            break;
        }

        if current_heading != emitted_heading
            && let Some(heading) = current_heading
        {
            keypoints.push_str(heading);
            keypoints.push('\n');
            emitted_heading = Some(heading);
        }
        keypoints.push_str(trimmed);
        keypoints.push('\n');
        keypoint_chars += chunk_chars;
    }

    keypoints
}

#[cfg(test)]
#[path = "tests/side_note_summary.rs"]
mod side_note_summary_tests;

#[cfg(test)]
mod summary_prompt_tests {
    use super::*;

    #[test]
    fn summary_prompt_configuration_preserves_default_and_literal_preferences() {
        let default = format!(include_str!("prompts/history_compressor.md"), 4_000);
        assert_eq!(history_summary_prompt(4_000, ""), default);
        assert_eq!(history_summary_prompt(4_000, " \n\t"), default);
        let configured = history_summary_prompt(4_000, "Keep {file:line} citations.");
        assert!(configured.starts_with(&default));
        assert!(configured.ends_with("Keep {file:line} citations."));
        assert!(
            configured
                .contains("subject to the provenance, uncertainty, and output-limit rules above")
        );
    }

    #[test]
    fn phase_timeout_clamps_to_deadline() {
        assert_eq!(
            phase_timeout(Duration::from_secs(15), None),
            Some(Duration::from_secs(15))
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        let limit = phase_timeout(Duration::from_secs(15), Some(deadline)).expect("time remains");
        assert!(limit <= Duration::from_secs(10));
        assert!(limit > Duration::ZERO);
    }
}

fn history_summary_prompt(max_chars: usize, additional: &str) -> String {
    let mut prompt = format!(include_str!("prompts/history_compressor.md"), max_chars);
    let additional = additional.trim();
    if !additional.is_empty() {
        prompt.push_str("\n\nAdditional summary preferences (subject to the provenance, uncertainty, and output-limit rules above):\n");
        prompt.push_str(additional);
    }
    prompt
}

struct HistoryCompressionProgress {
    local: Option<super::TransientStatusLine>,
    remote: bool,
}

impl HistoryCompressionProgress {
    fn start() -> Self {
        let remote = crate::ai::background::serve_live_streaming();
        if remote {
            crate::ai::background::publish_serve_frame(
                crate::ai::background::ServeLiveKind::CompressionStart,
                "",
            );
        }
        let mut local = (!remote).then(super::TransientStatusLine::new).flatten();
        if let Some(line) = local.as_mut() {
            line.update("Compressing history with the language model…");
        }
        Self { local, remote }
    }
}

impl Drop for HistoryCompressionProgress {
    fn drop(&mut self) {
        if let Some(line) = self.local.as_mut() {
            line.clear();
        }
        if self.remote {
            crate::ai::background::publish_serve_frame(
                crate::ai::background::ServeLiveKind::CompressionDone,
                "",
            );
        }
    }
}

/// Summarize older dialogue for context compression. Priority sampling preserves
/// recent requests, errors and decisions without letting one large message consume
/// the entire input budget.
pub(crate) async fn summarize_history_via_model(
    app: &App,
    messages: &[Message],
    max_chars: usize,
) -> Option<String> {
    if messages.is_empty() || max_chars == 0 {
        return None;
    }

    // Present priority-selected messages in their original chronological order.
    let transcript = build_summary_history_input(messages, SUMMARY_TRANSCRIPT_MAX_CHARS);
    if transcript.is_empty() {
        return None;
    }
    let cfg = crate::commonw::configw::get_all_config();
    let additional = cfg.get(
        crate::ai::config_schema::AiConfig::HISTORY_SUMMARY_PROMPT,
        "",
    );

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(history_summary_prompt(max_chars, &additional)),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(format!("{HISTORY_SUMMARY_USER_PREFIX}{transcript}")),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let control_model = control_model_for_aux_tasks(app);
    let mut request_body = build_request_body(
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
    let http_body =
        super::protocol::build_http_body_for_request(&control_model, &endpoint, &mut request_body);
    // A summary attempt is one bounded model request. On any failure the caller
    // preserves the original projection; it must not retry the same summary
    // through fallback keys and multiply compression latency or cost.
    let ctx = AuxRequestContext {
        config: &app.config,
        client: &app.client,
        session_id: &app.session_id,
    };
    let _progress = HistoryCompressionProgress::start();
    let text = match send_aux_chat_request_once(
        &ctx,
        &control_model,
        &endpoint,
        http_body,
        Duration::from_secs(60),
        Duration::from_secs(30),
    )
    .await
    {
        Ok(text) => text,
        Err((kind, msg)) => {
            super::emit_request_diagnostic(format_args!(
                "[summary] request failed ({kind}: {msg}), skipping"
            ));
            return None;
        }
    };
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = extract_router_content(&v)?;
    let trimmed = content.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

impl HistoryMessageSummarizer for App {
    fn summarize_history_messages<'a>(
        &'a self,
        messages: &'a [Message],
        max_chars: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send + 'a>> {
        Box::pin(summarize_history_via_model(self, messages, max_chars))
    }
}

fn session_title_text_content(content: &Value) -> String {
    let text = match content {
        Value::Array(parts) => parts
            .iter()
            .filter(|part| part.get("type").and_then(Value::as_str) != Some("image_url"))
            .filter_map(|part| {
                let text = super::types::extract_displayable_text(part);
                let text = text.trim();
                (!text.is_empty()).then(|| text.to_string())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::String(text) => text.trim().to_string(),
        other => crate::ai::history::value_to_string(other)
            .trim()
            .to_string(),
    };
    // The terminal's startup-resume selection prompt can be echoed into the
    // first user message; strip that UI noise so the title model summarizes
    // the real request instead of the prompt itself.
    crate::ai::history::strip_terminal_prompt_echo(&text)
}

/// Archive pointers are recovery metadata, not session-title material.
fn is_preserved_content_message(text: &str) -> bool {
    is_preserved_user_or_image_stub(text.trim_start())
}

fn session_title_dialog_lines(messages: &[crate::ai::history::Message]) -> Vec<String> {
    messages
        .iter()
        // 工具结果常常很长、且不等于用户想解决的问题。只给标题模型用户意图和
        // 最终回答，避免工具输出抢占有限的标题上下文窗口。
        .filter(|message| matches!(message.role.as_str(), "user" | "assistant"))
        // 子代理证据交接等运行时合成的 user 消息不是真实轮次（AGENTS.md 不变式
        // 12），若当成「用户: ...」喂给标题模型，多 skill/子代理会话的转录会被
        // 交接内容污染，导致模型回退到「帮我/请…」式低质量标题而被过滤。
        .filter(|message| !is_runtime_synthetic_user_message(message))
        .filter(|message| {
            !message
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty())
        })
        .filter_map(|message| {
            let content = session_title_text_content(&message.content);
            if content.is_empty() {
                return None;
            }
            if is_preserved_content_message(&content) {
                return None;
            }

            let role = match message.role.as_str() {
                "user" => "user",
                "assistant" => "assistant",
                _ => return None,
            };
            Some(format!("{role}: {content}"))
        })
        .collect()
}

/// 把会话标题生成的静默失败写入决策日志，替代被注释掉的 eprintln。
/// 标题任务在后台 spawn 中执行，eprintln 不可见；决策日志是唯一可观测渠道。
/// Record a failed title request in the decision log, so a transport failure is
/// distinguishable from "model answered but the quality filter rejected it".
fn record_title_failure(ctx: &AuxRequestContext<'_>, reason: &str, detail: &str) {
    crate::ai::driver::decision_log::log_session_title_failure(
        crate::ai::driver::decision_log::get_decision_log_store(),
        ctx.session_id,
        crate::ai::driver::runtime_ctx::current_turn_id_or_zero(),
        reason,
        detail,
    );
}

const SESSION_TITLE_TRANSCRIPT_MAX_CHARS: usize = 8_000;
const SESSION_TITLE_TRANSCRIPT_HEAD_CHARS: usize = 2_400;

/// 长对话保留开头的用户意图和结尾的最新结论，而不是只截取开头。
fn compact_session_title_transcript(dialog: &[String]) -> String {
    let transcript = dialog.join("\n");
    if transcript.chars().count() <= SESSION_TITLE_TRANSCRIPT_MAX_CHARS {
        return transcript;
    }

    let head: String = transcript
        .chars()
        .take(SESSION_TITLE_TRANSCRIPT_HEAD_CHARS)
        .collect();
    let tail_len = SESSION_TITLE_TRANSCRIPT_MAX_CHARS - SESSION_TITLE_TRANSCRIPT_HEAD_CHARS - 3;
    let mut tail: Vec<char> = transcript.chars().rev().take(tail_len).collect();
    tail.reverse();
    format!("{head}\n…\n{}", tail.into_iter().collect::<String>())
}

#[cfg(test)]
mod session_title_tests {
    use super::*;
    use crate::ai::history::Message;

    #[test]
    fn middle_keypoints_keep_role_provenance() {
        let middle =
            "Conclusion: guessed cause\n\n---\n\n### 🔧 TOOL\n\nError: direct diagnostic\n";

        let keypoints = extract_middle_keypoints(middle, Some("### 🤖 ASSISTANT"), 1_000);

        assert!(keypoints.contains("### 🤖 ASSISTANT\nConclusion: guessed cause"));
        assert!(keypoints.contains("### 🔧 TOOL\nError: direct diagnostic"));
    }

    #[test]
    fn summary_input_keeps_later_user_request_after_huge_message() {
        let messages = vec![
            Message {
                role: "tool".to_string(),
                content: Value::String("early-noise".repeat(20_000)),
                tool_calls: None,
                tool_call_id: Some("call-huge".to_string()),
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: Value::String("LATEST_EXPLICIT_REQUEST".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        let transcript = build_summary_history_input(&messages, 8_000);

        assert!(
            transcript.contains("LATEST_EXPLICIT_REQUEST"),
            "{transcript}"
        );
        assert!(transcript.chars().count() <= 8_000);
    }

    #[test]
    fn summary_message_keeps_tool_identity_when_arguments_are_huge() {
        let message: Message = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": "tool dispatch",
            "tool_calls": [{
                "id": "call-important",
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "arguments": "x".repeat(100_000)
                }
            }],
            "tool_call_id": null,
            "reasoning_content": null
        }))
        .unwrap();

        let rendered = render_summary_message(&message, 2_000);

        assert!(rendered.contains("id=call-important"), "{rendered}");
        assert!(rendered.contains("name=execute_command"), "{rendered}");
        assert!(rendered.chars().count() <= 2_000);
    }

    #[test]
    fn title_transcript_keeps_text_and_omits_images() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: Value::Array(vec![
                serde_json::json!({
                    "type": "image_url",
                    "image_url": { "url": "data:image/png;base64,abc" }
                }),
                serde_json::json!({ "type": "text", "text": "优化图片请求的 session title" }),
            ]),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        let dialog = session_title_dialog_lines(&messages);

        assert_eq!(dialog, vec!["user: 优化图片请求的 session title"]);
        assert!(!dialog[0].contains("image_url"));
        assert!(!dialog[0].contains("base64"));
    }

    #[test]
    fn title_transcript_skips_image_only_messages() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: serde_json::json!([{
                "type": "image_url",
                "image_url": { "url": "data:image/png;base64,abc" }
            }]),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        assert!(session_title_dialog_lines(&messages).is_empty());
    }

    #[test]
    fn title_transcript_skips_preserved_content_stubs() {
        let messages = [
            r#"[[PRESERVED_CONTENT_STUB_V1]]{"kind":"image","file_path":"/tmp/x"}"#,
            "[preserved-message-pointer-v2]\ncontent_kind: image\narchive_file_path: /tmp/x",
            "较早的用户图片内容已归档，原文未丢失。\n归档文件: /tmp/x",
        ]
        .into_iter()
        .map(|content| Message {
            role: "user".to_string(),
            content: Value::String(content.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        })
        .collect::<Vec<_>>();

        assert!(session_title_dialog_lines(&messages).is_empty());
    }

    #[test]
    fn title_transcript_skips_runtime_synthetic_user_messages() {
        use crate::ai::history::runtime_synthetic_user_message;
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Value::String("修复 session title".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            // agent-team 等多子代理场景：子代理证据交接被写成运行时合成的 user 消息，
            // 不应以「用户: …」的形式混入标题转录。
            runtime_synthetic_user_message(Value::String(
                "[Subagent evidence] 子代理交接内容...".to_string(),
            )),
        ];

        let dialog = session_title_dialog_lines(&messages);

        assert_eq!(dialog, vec!["user: 修复 session title"]);
    }

    #[test]
    fn title_transcript_ignores_tool_output_and_keeps_final_answer() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: Value::String("修复 session title".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("无关且很长的工具输出".repeat(100)),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".to_string(),
                content: Value::String("已改为在完整回复后生成标题".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        let dialog = session_title_dialog_lines(&messages);

        assert_eq!(
            dialog,
            vec![
                "user: 修复 session title",
                "assistant: 已改为在完整回复后生成标题"
            ]
        );
    }

    #[test]
    fn long_title_transcript_keeps_initial_intent_and_final_conclusion() {
        let dialog = vec![
            format!("user: 任务意图{}", "a".repeat(3_000)),
            format!("assistant: 最终结论{}", "b".repeat(6_000)),
        ];

        let transcript = compact_session_title_transcript(&dialog);

        assert!(transcript.starts_with("user: 任务意图"));
        assert!(transcript.ends_with('b'));
        assert!(transcript.contains("\n…\n"));
        assert!(transcript.chars().count() <= SESSION_TITLE_TRANSCRIPT_MAX_CHARS);
    }

    #[test]
    fn title_transcript_strips_terminal_resume_echo() {
        // The terminal's resume-selection prompt echoed into the first user
        // message must not reach the title model (it produced the abstract
        // "单会话仍要确认" title from exactly this capture shape).
        let messages = vec![Message {
            role: "user".to_string(),
            content: Value::String(
                "修复一个 \"/bg\" 相关的问题。为什么一个terminal，只挂了一个session，还要确认？\n\
[resume] 当前 terminal 有 1 个挂起 session：\n\
  1. 6ebee611-40ca-4c15-9361-81d2de4e8c62  persona=default  modified=2026-09-17 15:03  suspended=2026-09-17 15:03\n\
选择要恢复的 session [回车=恢复，n=新 session]: "
                    .to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

        let dialog = session_title_dialog_lines(&messages);

        assert_eq!(
            dialog,
            vec![
                "user: 修复一个 \"/bg\" 相关的问题。为什么一个terminal，只挂了一个session，还要确认？"
            ]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn summary_request_uses_only_the_preferred_provider_key() {
        // Regression: with a named provider key configured (`opencode.api_key_*`), auxiliary
        // requests (title/summary) must offer the same key candidates as the main request.
        // Using only the primary key fell back to the global api_key, which fails against the
        // gateway with a silent 401 (sessions ended up without titles/summaries).
        // The model is taken from the registry so entry renames cannot break the test.
        let _guard = crate::ai::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let old_configw_path = std::env::var_os("CONFIGW_PATH");
        let dir = std::env::temp_dir().join(format!("configw_aux_key_test_{}", std::process::id()));
        let path = dir.join("configW");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            &path,
            "api_key = \"global-invalid\"\nopencode.api_key_grewwc = \"named-valid\"\n",
        )
        .unwrap();
        unsafe { std::env::set_var("CONFIGW_PATH", &path) };
        crate::commonw::configw::refresh();

        let model = crate::ai::model_names::all()
            .iter()
            .find(|m| {
                m.adapter == crate::ai::provider::ApiProvider::OpenCode
                    && !m.name.starts_with("enc:")
            })
            .copied()
            .expect("registry must contain an OpenCode entry with a plaintext name");

        let keys = aux_request_key_candidates(
            &model.key,
            crate::ai::provider::OPENCODE_DEFAULT_ENDPOINT,
            "global-invalid",
        );
        assert_eq!(keys, vec!["named-valid", "global-invalid"]);

        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4_096];
            socket.read(&mut request).await.unwrap();
            socket
                .write_all(
                    b"HTTP/1.1 400 Bad Request\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
                )
                .await
                .unwrap();
            if let Ok(Ok((mut retry, _))) =
                tokio::time::timeout(Duration::from_millis(250), listener.accept()).await
            {
                retry.read(&mut request).await.unwrap();
                retry
                    .write_all(
                        b"HTTP/1.1 400 Bad Request\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
                    )
                    .await
                    .unwrap();
                true
            } else {
                false
            }
        });
        let config = AppConfig {
            api_key: "global-invalid".to_string(),
            base_history_file: Default::default(),
            history_file: Default::default(),
            endpoint: String::new(),
            vl_default_model: String::new(),
            history_max_chars: 0,
            history_keep_last: 0,
            history_summary_max_chars: 0,
            intent_model: None,
        };
        let client = reqwest::Client::new();
        let ctx = AuxRequestContext {
            config: &config,
            client: &client,
            session_id: "summary-single-request-test",
        };
        let result = send_aux_chat_request_once(
            &ctx,
            &model.key,
            &endpoint,
            b"{}".to_vec(),
            Duration::from_secs(2),
            Duration::from_secs(2),
        )
        .await;
        assert!(result.is_err());
        assert!(
            !server.await.unwrap(),
            "summary failure must not retry with the fallback API key"
        );

        // Restore the previous CONFIGW_PATH (if any) so later tests in this process are unaffected.
        match old_configw_path {
            Some(old) => unsafe { std::env::set_var("CONFIGW_PATH", old) },
            None => unsafe { std::env::remove_var("CONFIGW_PATH") },
        }
        crate::commonw::configw::refresh();
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// Generate a short summarizing title (<= 20 chars) for the conversation with the LLM.
/// Feeds the session list and the prompt's top line.
pub(crate) async fn generate_session_title_via_model(
    app: &App,
    messages: &[crate::ai::history::Message],
) -> Option<String> {
    let ctx = AuxRequestContext {
        config: &app.config,
        client: &app.client,
        session_id: &app.session_id,
    };
    generate_session_title_with_context(&ctx, &app.current_model, messages).await
}

/// [`generate_session_title_via_model`] for callers that hold an
/// [`AuxRequestContext`] instead of an `App` (the serve daemon generates the
/// title once the turn child has exited; see
/// `driver::turn_runtime::generate_session_title_outside_turn`).
pub(crate) async fn generate_session_title_with_context(
    ctx: &AuxRequestContext<'_>,
    current_model: &str,
    messages: &[crate::ai::history::Message],
) -> Option<String> {
    if messages.is_empty() {
        return None;
    }

    // Only the user's intent and the assistant's final answer feed the title.
    // Image parts stay out of it so unrelated UI text inside a screenshot
    // cannot skew the title; an image-only request relies on the text the user
    // typed alongside it.
    let dialog = session_title_dialog_lines(messages);

    if dialog.is_empty() {
        return None;
    }

    let transcript = compact_session_title_transcript(&dialog);

    let system_prompt = include_str!("prompts/session_title.md");

    let user_prompt = format!("Conversation:\n\n{transcript}\n\nGenerate the title:");

    let control_model = control_model_for_aux_tasks_from(ctx.config, current_model);
    let title_model = control_model;
    let user_content = Value::String(user_prompt);

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
            content: user_content,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];

    let mut request_body = build_request_body(
        &title_model,
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
    let endpoint = models::endpoint_for_model(&title_model, &ctx.config.endpoint);
    let http_body =
        super::protocol::build_http_body_for_request(&title_model, &endpoint, &mut request_body);

    // Rotate keys exactly like the main request path (`collect_api_keys`):
    // under a named-key configuration (`opencode.api_key_xxx`) using only the
    // primary key silently fails with a gateway 401 and the session title
    // stays on the fallback forever. The last key's error is what gets logged.
    let text = match send_aux_chat_request_with_key_rotation(
        ctx,
        &title_model,
        &endpoint,
        http_body,
        Duration::from_secs(SESSION_TITLE_REQUEST_TIMEOUT_SECS),
        Duration::from_secs(SESSION_TITLE_BODY_TIMEOUT_SECS),
        None,
    )
    .await
    {
        Ok(text) => text,
        Err((kind, msg)) => {
            record_title_failure(ctx, &kind, &msg);
            return None;
        }
    };

    let v: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            record_title_failure(ctx, "json_parse_error", &e.to_string());
            return None;
        }
    };
    let content = match extract_router_content(&v) {
        Some(c) => c,
        None => {
            record_title_failure(ctx, "extract_router_content_none", "");
            return None;
        }
    };
    // Strip the `<think>...</think>` chain of thought before line splitting and
    // cleaning. A thinking-mode model returns the chain together with the
    // answer, so without stripping first the `.lines().next()` below would cut
    // at the `<think>` first line and persist a reasoning fragment as a valid
    // title. This has to happen before the line split.
    let content = crate::ai::history::strip_think_tags(&content);
    let trimmed = content.trim().to_string();

    // Clean: drop quotes, keep the first line, then cap the length below.
    let cleaned = trimmed
        .trim_matches(|c: char| {
            c == '"' || c == '「' || c == '」' || c == '\'' || c.is_whitespace()
        })
        .lines()
        .next()
        .unwrap_or("")
        .to_string();

    if cleaned.is_empty() {
        record_title_failure(ctx, "empty_title_after_cleanup", &trimmed);
        return None;
    }

    // Cap at 30 characters (a CJK glyph counts as one char).
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

/// Equivalent to [`charge_llm_usage_to_kernel`], but takes a `SharedKernel`
/// directly. Intended for callers without an `App` handle (e.g. background
/// reflection's `background_call`): `GLOBAL_OS` and `App.os` share the same
/// `Arc<Mutex<Kernel>>`, so the accounting semantics are identical.
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
    let reasoning = usage
        .completion_tokens_details
        .as_ref()
        .map(|d| d.reasoning_tokens)
        .unwrap_or(0);
    let report = aios_kernel::primitives::LlmUsageReport {
        model: requested_model.to_string(),
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        reasoning_tokens: reasoning,
        cached_prompt_tokens: cached,
        latency_ms,
    };
    // Serialize the whole "drain -> persist -> advance cursor" sequence: the cursor is
    // read and the ledger drained under the kernel lock, but it only advances after a
    // successful commit in `persist_drained`. Without this guard, concurrent LLM calls
    // (e.g. in-process subagents) could both read the same cursor and double-insert the
    // same records. A concurrent caller blocks here until we commit, then reads the
    // already-advanced cursor and only fetches new records.
    let _drain_guard = crate::ai::tools::storage::token_usage_store::drain_lock();
    // Settle the call in the kernel (billing + rusage + trace + appending to the audit
    // ledger) and fetch the incremental records to drain for this call. SQLite I/O
    // happens after the guard is released, so we never hold the kernel lock for disk writes.
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
    // Best-effort persistence into the dedicated token-usage stats table;
    // failures never block the main flow.
    crate::ai::tools::storage::token_usage_store::persist_drained(&drained, head);
    Some(outcome)
}
