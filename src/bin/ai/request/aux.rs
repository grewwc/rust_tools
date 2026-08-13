//! 辅助 LLM 任务：历史摘要、会话标题生成、token 用量落账。
//!
//! 这些函数共享主链路的 `build_request_body` / auth 基础设施，但不参与主 turn
//! 流式，因此独立为子模块，降低 `mod.rs` 的认知负担。

use std::time::Duration;

use serde_json::Value;

use super::builder::build_request_body;
use super::types::StreamUsage;
use super::{
    apply_request_auth, control_model_for_aux_tasks, endpoint_for_request_model,
    extract_router_content,
};
use crate::ai::{
    history::{Message, is_runtime_synthetic_user_message, messages_to_markdown},
    models, provider::adapter_for,
    types::App,
};

/// 会话标题请求的超时（秒）。后台辅助任务，用宽松超时避免阻塞主流程。
pub(super) const SESSION_TITLE_REQUEST_TIMEOUT_SECS: u64 = 90;
pub(super) const SESSION_TITLE_BODY_TIMEOUT_SECS: u64 = 45;

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

/// 发送辅助 LLM 请求（POST chat/completions），key 轮换直到成功。
///
/// 对每个 key 单独套用超时，避免单个 key 挂起拖慢后台辅助任务。
/// 返回 2xx 响应体文本；全部 key 失败时返回最后一个错误的 (kind, message)，
/// kind 与 record_title_failure 的错误分类一致。
async fn send_aux_chat_request_with_key_rotation(
    app: &App,
    model: &str,
    endpoint: &str,
    http_body: Vec<u8>,
    header_timeout: Duration,
    body_timeout: Duration,
) -> Result<String, (String, String)> {
    let keys = aux_request_key_candidates(model, endpoint, &app.config.api_key);
    let mut last_err: (String, String) = ("http_error".to_string(), "unknown".to_string());

    for (idx, api_key) in keys.iter().enumerate() {
        if idx > 0 {
            super::emit_request_diagnostic(format_args!(
                "[aux] key #{} failed, trying next key #{} ({} remaining)",
                idx - 1,
                idx,
                keys.len() - idx
            ));
        }

        let send_future = apply_request_auth(app.client.post(endpoint), endpoint, api_key)
            .header("Content-Type", "application/json")
            .body(http_body.clone())
            .send();

        let response = match tokio::time::timeout(header_timeout, send_future).await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                last_err = ("request_error".to_string(), e.to_string());
                continue;
            }
            Err(_) => {
                last_err = (
                    "request_timeout".to_string(),
                    format!("{}s", header_timeout.as_secs()),
                );
                continue;
            }
        };

        let status = response.status();
        if !status.is_success() {
            last_err = ("http_error".to_string(), format!("HTTP {status}"));
            continue;
        }

        let text = match tokio::time::timeout(body_timeout, response.text()).await {
            Ok(Ok(t)) => t,
            Ok(Err(e)) => {
                last_err = ("body_read_error".to_string(), e.to_string());
                continue;
            }
            Err(_) => {
                last_err = (
                    "body_timeout".to_string(),
                    format!("{}s", body_timeout.as_secs()),
                );
                continue;
            }
        };
        return Ok(text);
    }
    Err(last_err)
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
        const MID_KEYPOINTS_BUDGET: usize = 4_000;
        let initial_heading = head
            .lines()
            .rev()
            .find(|line| is_exported_message_heading(line));
        let keypoints =
            extract_middle_keypoints(&middle_segment, initial_heading, MID_KEYPOINTS_BUDGET);

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
                "You are a software development conversation-history compressor. Your task is to compress the earlier conversation into a summary that a later coding agent can keep working from.\n\
Output requirements:\n\
- Output plain text only; no markdown code blocks, no explanations.\n\
- Must retain: explicit user requests, file paths / function names / tool names, key errors, current work, unfinished tasks, and re-readable source paths or tool invocations.\n\
- Strictly distinguish three categories: verified facts directly supported by tool/source evidence, assistant judgments raised earlier but not yet verified, and open questions still to be confirmed. Never rewrite a statement from the assistant into a fact or a fix conclusion just because it came from the assistant.\n\
- Only tool/source evidence directly visible in the input can support \"verified\"; an assistant's older conclusion or the paths it cites are only read-back locators and cannot be upgraded to facts by citation alone.\n\
- Verified facts should carry a source (file path, command, or tool name) when possible; when there is no source, mark it \"source not retained\". Conflicting evidence and uncertainty must be preserved; do not make determinacy rulings on the model's behalf.\n\
- Prioritize user decisions and sourced facts; drop small talk, repeated confirmations, and verbose logs.\n\
- Use the headings below, with short lines starting with `- ` under each:\n\
Main request:\nUser decisions:\nVerified facts and sources:\nUnverified assistant judgments:\nConflicts and unknowns:\nCurrent work:\nPending tasks:\n\
- If a section has no content, write `- none`.\n\
- Keep the total length within about {} characters.",
                max_chars
            )),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: Value::String(format!("Please compress the earlier conversation below:\n\n{}", transcript)),
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
    let http_body =
        super::protocol::build_http_body_for_request(&control_model, &endpoint, &request_body);
    // 历史摘要是 turn 收尾的后台辅助请求（任务边界压缩会在每次答案交付后触发）。
    // 主 client 只有 connect_timeout、没有整体 timeout，若摘要模型接受连接后迟迟
    // 不返回响应头，这里的裸 .send()/.text() 会永久阻塞、CPU 0，表现为"答案已输出
    // 但迟迟不回到提示符"的卡死。用显式超时兜底，超时即放弃摘要（保持原始历史）。
    // key 按 collect_api_keys 轮换（与主请求链路一致）：命名 key 配置下仅用
    // primary 会对网关 401 静默失败 → 无摘要。全部 key 失败时放弃摘要。
    let text = match send_aux_chat_request_with_key_rotation(
        app,
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

fn session_title_text_content(content: &Value) -> String {
    match content {
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
    }
}

/// 归档提示仅用于上下文恢复，不能成为会话标题的素材。
fn is_preserved_content_message(text: &str) -> bool {
    let text = text.trim_start();
    text.starts_with("[[PRESERVED_CONTENT_STUB_V1]]")
        || text.starts_with("较早的用户图片内容已归档")
        || text.starts_with("较早的用户文本内容已归档")
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
                "user" => "用户",
                "assistant" => "助手",
                _ => return None,
            };
            Some(format!("{role}: {content}"))
        })
        .collect()
}

/// 把会话标题生成的静默失败写入决策日志，替代被注释掉的 eprintln。
/// 标题任务在后台 spawn 中执行，eprintln 不可见；决策日志是唯一可观测渠道。
fn record_title_failure(app: &App, reason: &str, detail: &str) {
    crate::ai::driver::decision_log::log_session_title_failure(
        crate::ai::driver::decision_log::get_decision_log_store(),
        &app.session_id,
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
        let middle = "Conclusion: guessed cause\n\n---\n\n### 🔧 TOOL\n\nError: direct diagnostic\n";

        let keypoints = extract_middle_keypoints(middle, Some("### 🤖 ASSISTANT"), 1_000);

        assert!(keypoints.contains("### 🤖 ASSISTANT\nConclusion: guessed cause"));
        assert!(keypoints.contains("### 🔧 TOOL\nError: direct diagnostic"));
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

        assert_eq!(dialog, vec!["用户: 优化图片请求的 session title"]);
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
        let messages = vec![Message {
            role: "user".to_string(),
            content: Value::String(
                r#"[[PRESERVED_CONTENT_STUB_V1]]{"kind":"image","file_path":"/tmp/x"}"#.to_string(),
            ),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];

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

        assert_eq!(dialog, vec!["用户: 修复 session title"]);
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
                "用户: 修复 session title",
                "助手: 已改为在完整回复后生成标题"
            ]
        );
    }

    #[test]
    fn long_title_transcript_keeps_initial_intent_and_final_conclusion() {
        let dialog = vec![
            format!("用户: 任务意图{}", "a".repeat(3_000)),
            format!("助手: 最终结论{}", "b".repeat(6_000)),
        ];

        let transcript = compact_session_title_transcript(&dialog);

        assert!(transcript.starts_with("用户: 任务意图"));
        assert!(transcript.ends_with('b'));
        assert!(transcript.contains("\n…\n"));
        assert!(transcript.chars().count() <= SESSION_TITLE_TRANSCRIPT_MAX_CHARS);
    }

    #[test]
    fn aux_request_key_candidates_prefers_named_provider_key() {
        // 回归：命名 key（opencode.api_key_xxx）配置下，辅助请求（标题/摘要）的
        // key 候选必须与主请求一致优先使用命名 key；此前只用 primary key 会回退
        // 到全局 api_key（对网关失效时 401 静默失败 → session 无标题/无摘要）。
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

        let keys = aux_request_key_candidates(
            "deepseek-v4-flash-opencode",
            crate::ai::provider::OPENCODE_DEFAULT_ENDPOINT,
            "global-invalid",
        );
        assert_eq!(keys, vec!["named-valid", "global-invalid"]);

        // 清理：恢复原有 CONFIGW_PATH（若有），避免影响同进程后续测试
        match old_configw_path {
            Some(old) => unsafe { std::env::set_var("CONFIGW_PATH", old) },
            None => unsafe { std::env::remove_var("CONFIGW_PATH") },
        }
        crate::commonw::configw::refresh();
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// 用 LLM 为当前对话生成一个简短的概括性标题（不超过 20 字）。
/// 供 session 列表和输入框顶部展示使用。
pub(crate) async fn generate_session_title_via_model(
    app: &App,
    messages: &[crate::ai::history::Message],
) -> Option<String> {
    if messages.is_empty() {
        return None;
    }

    // 只取用户意图和助手最终回答用于生成标题。图片内容不参与标题生成，避免模型被截图
    // 里的无关 UI 文案干扰；图片请求依赖用户同时输入的文字来概括主题。
    let dialog = session_title_dialog_lines(messages);

    if dialog.is_empty() {
        return None;
    }

    let transcript = compact_session_title_transcript(&dialog);

    let system_prompt = "你是一个对话标题生成器。根据下面的对话内容，生成一个不超过20个字的简短标题，概括对话的核心主题。\n\
要求：\n\
- 只输出标题本身，不要引号，不要解释，不要前缀。\n\
- 标题要具体、有信息量，不要太笼统。\n\
- 如果对话附带图片，基于用户同时输入的文字概括主题，不要只复述‘看截图’、‘图片问题’等泛化表述。\n\
- 优先用名词短语或动宾短语。\n\
- 如果是编程相关，提到关键技术或文件名。";

    let user_prompt = format!("对话内容：\n\n{transcript}\n\n请生成标题：");

    let control_model = control_model_for_aux_tasks(app);
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

    let request_body = build_request_body(
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
    let endpoint = endpoint_for_request_model(app, &title_model);
    let http_body =
        super::protocol::build_http_body_for_request(&title_model, &endpoint, &request_body);

    // key 按 collect_api_keys 轮换（与主请求链路一致）：命名 key
    // （opencode.api_key_xxx）配置下仅用 primary key 会对网关 401 静默失败，
    // 导致 session 标题长期停留在 fallback。失败时记录最后一个 key 的错误。
    let text = match send_aux_chat_request_with_key_rotation(
        app,
        &title_model,
        &endpoint,
        http_body,
        Duration::from_secs(SESSION_TITLE_REQUEST_TIMEOUT_SECS),
        Duration::from_secs(SESSION_TITLE_BODY_TIMEOUT_SECS),
    )
    .await
    {
        Ok(text) => text,
        Err((kind, msg)) => {
            record_title_failure(app, &kind, &msg);
            return None;
        }
    };

    let v: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            record_title_failure(app, "json_parse_error", &e.to_string());
            return None;
        }
    };
    let content = match extract_router_content(&v) {
        Some(c) => c,
        None => {
            record_title_failure(app, "extract_router_content_none", "");
            return None;
        }
    };
    // 先剥离 `<think>...</think>` 思维链，再做行拆分/清洗。thinking 模式模型会
    // 把思维链连同答案一起返回，若不先剥离，下面的 `.lines().next()` 会截到
    // `<think>` 首行，把思维链碎片当成合格标题写库。必须在 line 拆分之前完成。
    let content = crate::ai::history::strip_think_tags(&content);
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
        record_title_failure(app, "empty_title_after_cleanup", &trimmed);
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
