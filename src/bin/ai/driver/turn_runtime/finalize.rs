use crate::ai::{
    driver::print::{format_empty_state, print_assistant_banner_with_app_and_skill},
    history::{
        Message, SessionTitle, SessionTitleOrigin, compact_session_history_at_boundary_with_app,
        compact_session_history_with_app, generate_session_summary, is_low_quality_session_title,
        normalize_generated_session_title, value_to_string,
    },
    types::App,
};
use rust_tools::commonw::FastSet;
use serde_json::Value;
use std::sync::{LazyLock, Mutex};

use super::{TurnOutcome, persistence::persist_pending_turn_messages_for_model};

const SUBAGENT_TOOL_EVIDENCE_MAX_CALLS: usize = 8;
const SUBAGENT_TOOL_EVIDENCE_MAX_CHARS_PER_RESULT: usize = 700;
const SUBAGENT_TOOL_EVIDENCE_MAX_BLOCK_CHARS: usize = 4_000;
const MODEL_CONTEXT_ECHO_PREFIX: &str = "[Model-authored note from an earlier turn;";
static SESSION_TITLE_IN_FLIGHT: LazyLock<Mutex<FastSet<String>>> =
    LazyLock::new(|| Mutex::new(FastSet::default()));
/// 与标题任务同构的 in-flight 去重：避免同一 session 同时跑多次持久化压缩
///（前台收尾派发的后台压缩与下一轮 prepare 的防御性压缩可能重叠）。
static SESSION_COMPACTION_IN_FLIGHT: LazyLock<Mutex<FastSet<String>>> =
    LazyLock::new(|| Mutex::new(FastSet::default()));

/// 标题任务必须从基础 history 文件推导 sessions 根目录；不能传入当前 session
/// 数据库的父目录，否则会错误地拼接出嵌套的 `.sessions` 路径。
fn session_title_store(history_file: &std::path::Path) -> crate::ai::history::SessionStore {
    crate::ai::history::SessionStore::new(history_file)
}

fn ensure_final_assistant_recorded(
    final_assistant_text: &str,
    final_assistant_recorded: bool,
    turn_messages: &mut Vec<Message>,
) {
    if final_assistant_recorded {
        return;
    }

    turn_messages.push(Message {
        role: "assistant".to_string(),
        content: Value::String(final_assistant_text.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
}

/// 模型偶尔会把 request projection 中的 internal_note 原样回显。仅供模型消费的
/// runtime note 必须留在 canonical history，但不能作为用户可见回答打印到 terminal。
/// 已流式输出的回答只补画 runtime 显式标记为用户可见的追加提示。
fn terminal_final_text_to_render(
    final_assistant_text: &str,
    final_assistant_recorded: bool,
    user_visible_suffix: Option<&str>,
) -> Option<String> {
    // 已写入 canonical history 的最终模型响应此前已经由 stream runtime 实时输出；
    // finalize 只补画未经过流式响应的本地 fallback，避免整段正文在终端重复一次。
    if final_assistant_recorded {
        return user_visible_suffix
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string);
    }
    let terminal_text = final_assistant_text
        .split_once("\n\n[Runtime warning]")
        .map_or(final_assistant_text, |(visible, _)| visible);
    let trimmed = terminal_text.trim_start();
    if trimmed.starts_with(MODEL_CONTEXT_ECHO_PREFIX)
        || trimmed.starts_with("self_note:")
        || trimmed.starts_with("[Runtime warning]")
    {
        return None;
    }
    Some(terminal_text.to_string())
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut out = value
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    out.push('…');
    out
}

fn normalized_tool_args(raw: &str) -> String {
    serde_json::from_str::<Value>(raw)
        .map(|value| value.to_string())
        .unwrap_or_else(|_| raw.trim().to_string())
}

fn collect_subagent_tool_evidence(turn_messages: &[Message]) -> Vec<String> {
    use rustc_hash::FxHashMap;

    let mut id_to_call: FxHashMap<String, (String, String)> = FxHashMap::default();
    for message in turn_messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            id_to_call.insert(
                tool_call.id.clone(),
                (
                    tool_call.function.name.clone(),
                    normalized_tool_args(&tool_call.function.arguments),
                ),
            );
        }
    }

    let mut evidence = Vec::new();
    for message in turn_messages {
        if message.role != "tool" {
            continue;
        }
        let Some(tool_call_id) = message.tool_call_id.as_ref() else {
            continue;
        };
        let Some((tool_name, args)) = id_to_call.get(tool_call_id) else {
            continue;
        };
        let content = value_to_string(&message.content);
        let content = content.trim();
        if content.is_empty() {
            continue;
        }
        evidence.push(format!(
            "- {}({}) => {}",
            tool_name,
            args,
            truncate_chars(content, SUBAGENT_TOOL_EVIDENCE_MAX_CHARS_PER_RESULT)
        ));
        if evidence.len() >= SUBAGENT_TOOL_EVIDENCE_MAX_CALLS {
            break;
        }
    }
    evidence
}

fn format_subagent_result_for_parent(
    final_assistant_text: &str,
    turn_messages: &[Message],
) -> String {
    let final_assistant_text = final_assistant_text.trim();
    let evidence = collect_subagent_tool_evidence(turn_messages);
    if evidence.is_empty() {
        return final_assistant_text.to_string();
    }

    let mut evidence_block = String::from("[Subagent tool evidence]\n");
    evidence_block.push_str(
        "The subagent used these tool results while producing the answer; treat them as already observed context.\n",
    );
    evidence_block.push_str(&evidence.join("\n"));

    let mut output = truncate_chars(&evidence_block, SUBAGENT_TOOL_EVIDENCE_MAX_BLOCK_CHARS);
    if final_assistant_text.is_empty() {
        return output;
    }
    output.push_str("\n\n");
    output.push_str("[Subagent final answer]\n");
    output.push_str(final_assistant_text);
    output
}

fn subagent_result_payload_for_parent(
    final_assistant_text: &str,
    turn_messages: &[Message],
) -> Option<String> {
    let output = format_subagent_result_for_parent(final_assistant_text, turn_messages);
    (!output.trim().is_empty()).then_some(output)
}

pub(in crate::ai::driver::turn_runtime) fn should_generate_session_title_in_background(
    one_shot_mode: bool,
    should_quit: bool,
) -> bool {
    !one_shot_mode && !should_quit
}

fn should_compact_session_history_in_background(
    one_shot_mode: bool,
    should_quit: bool,
    is_subagent: bool,
) -> bool {
    // 子代理 history 由调用方的生命周期 guard 管理；压缩若脱离当前 future，
    // guard 可能先删除临时 SQLite，后台任务随后访问已不存在的文件。
    !is_subagent && !one_shot_mode && !should_quit
}

fn mark_session_title_generation_started(session_id: &str) -> bool {
    let mut in_flight = SESSION_TITLE_IN_FLIGHT
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    in_flight.insert(session_id.to_string())
}

fn mark_session_title_generation_finished(session_id: &str) {
    let mut in_flight = SESSION_TITLE_IN_FLIGHT
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    in_flight.remove(session_id);
}

/// 返回 true 表示成功抢到压缩 in-flight 槽位；false 表示该 session 已有
/// 压缩任务在跑，调用方应直接跳过，避免重复压缩同一批历史。
pub(in crate::ai::driver::turn_runtime) fn mark_session_compaction_started(
    session_id: &str,
) -> bool {
    let mut in_flight = SESSION_COMPACTION_IN_FLIGHT
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    in_flight.insert(session_id.to_string())
}

pub(in crate::ai::driver::turn_runtime) fn mark_session_compaction_finished(session_id: &str) {
    let mut in_flight = SESSION_COMPACTION_IN_FLIGHT
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    in_flight.remove(session_id);
}

/// 交互式收尾把持久化压缩派发到后台：压缩结果只是 context snapshot 的写回缓存，
/// 下一轮 `build_context_history` 始终从 canonical 层重新压缩，故延迟落盘不影响
/// 正确性。前台 turn 因此不必在答案交付后再干等一次 CPU 压缩 + SQLite 写事务。
///
/// `at_boundary` 对应本轮是否再调用过工具（无工具调用 = "答案已交付"，用更激进
/// 阈值）。返回后 in-flight 槽位由 spawned task 负责清理。
fn spawn_background_compaction(app: &App, at_boundary: bool) {
    if !mark_session_compaction_started(&app.session_id) {
        // 已有压缩在跑（例如上一轮的后台任务尚未完成）；本轮跳过即可，
        // 待其完成后下一轮 prepare 会基于最新 canonical 再判定。
        return;
    }
    let task_app = app.clone();
    let session_id = task_app.session_id.clone();
    // 后台任务不拥有前台终端：压缩的告警/日志不得抢占前台输出光标。
    tokio::spawn(
        crate::ai::driver::runtime_ctx::SUPPRESS_TERMINAL_OUTPUT.scope(true, async move {
            let compact_result = if at_boundary {
                compact_session_history_at_boundary_with_app(&task_app).await
            } else {
                compact_session_history_with_app(&task_app).await
            };
            if let Err(err) = compact_result {
                eprintln!("[Warning] Failed to compact persisted history: {}", err);
            }
            mark_session_compaction_finished(&session_id);
        }),
    );
}

/// 收尾阶段的持久化压缩派发。前台交互式 turn 走后台（不阻塞前台回到 prompt）；
/// one-shot、即将退出的进程及子代理走前台 `.await`，确保 snapshot 在所属 history
/// 的生命周期结束前落盘。
async fn dispatch_finalize_compaction(app: &App, at_boundary: bool, run_in_background: bool) {
    if run_in_background {
        spawn_background_compaction(app, at_boundary);
        return;
    }
    let compact_result = if at_boundary {
        compact_session_history_at_boundary_with_app(app).await
    } else {
        compact_session_history_with_app(app).await
    };
    if let Err(err) = compact_result
        && crate::ai::driver::runtime_ctx::terminal_output_enabled()
    {
        eprintln!("[Warning] Failed to compact persisted history: {}", err);
    }
}

/// 标题任务可以在当前 user message 尚未落盘时启动；把这条已提交输入补进快照，
/// 让辅助模型无需等待主请求或首轮 assistant response。
fn session_title_messages(
    mut persisted_messages: Vec<Message>,
    pending_user_input: Option<&str>,
) -> Vec<Message> {
    let Some(user_input) = pending_user_input
        .map(str::trim)
        .filter(|input| !input.is_empty())
    else {
        return persisted_messages;
    };

    persisted_messages.push(Message {
        role: "user".to_string(),
        content: Value::String(user_input.to_string()),
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    });
    persisted_messages
}

fn has_session_title_source(messages: &[Message]) -> bool {
    messages.iter().any(|message| {
        message.role == "user" && !value_to_string(&message.content).trim().is_empty()
    })
}

/// fallback 与旧版无来源标题的匹配依据必须保持和落盘时一致。
fn fallback_session_title(messages: &[Message]) -> String {
    messages
        .iter()
        .filter(|message| message.role == "user")
        .map(|message| value_to_string(&message.content))
        .map(|text| normalize_generated_session_title(&generate_session_summary(&text)))
        .find(|title| !title.is_empty())
        .unwrap_or_default()
}

fn should_generate_model_session_title(
    existing: Option<&SessionTitle>,
    fallback_title: &str,
) -> bool {
    let Some(existing) = existing else {
        return true;
    };
    if is_low_quality_session_title(&existing.text) {
        return true;
    }

    match existing.origin {
        SessionTitleOrigin::Fallback => true,
        // 没有来源标记的旧标题只有在与旧 fallback 完全一致时才升级，避免误覆盖
        // 已经由模型生成的历史好标题。
        SessionTitleOrigin::Legacy => {
            !fallback_title.is_empty()
                && normalize_generated_session_title(&existing.text) == fallback_title
        }
        SessionTitleOrigin::Model => false,
    }
}

fn should_write_fallback_session_title(
    existing: Option<&SessionTitle>,
    fallback_title: &str,
) -> bool {
    match existing {
        None => true,
        Some(existing) if is_low_quality_session_title(&existing.text) => true,
        // 迁移旧 fallback 时补上来源标记；之后可以可靠地继续尝试模型升级。
        Some(existing) => {
            existing.origin == SessionTitleOrigin::Legacy
                && normalize_generated_session_title(&existing.text) == fallback_title
        }
    }
}

pub(super) async fn finalize_turn(
    app: &mut App,
    _next_model: &str,
    response_source_model: &str,
    question: &str,
    final_assistant_text: &str,
    final_assistant_recorded: bool,
    user_visible_final_suffix: Option<&str>,
    active_skill_name: Option<&str>,
    turn_messages: &mut Vec<Message>,
    one_shot_mode: bool,
    persisted_turn_messages: &mut usize,
    should_quit: bool,
    _had_tool_error: bool,
) -> Result<TurnOutcome, Box<dyn std::error::Error>> {
    if let Some(subagent_output_for_parent) =
        subagent_result_payload_for_parent(final_assistant_text, turn_messages)
    {
        // 尽早发布给父 agent：即便本轮没有最终 assistant 正文，只要留下了可复用的
        // subagent 证据（如 read_file 结果），父 agent 也必须感知。
        // 这样同步 `task` 与异步 `task_wait` 都能拿到同一份父侧 payload。
        crate::ai::driver::runtime_ctx::publish_subagent_result(
            &subagent_output_for_parent,
            final_assistant_text,
        )
        .await;
    }

    if !final_assistant_text.trim().is_empty() {
        ensure_final_assistant_recorded(
            final_assistant_text,
            final_assistant_recorded,
            turn_messages,
        );
        if crate::ai::driver::runtime_ctx::terminal_output_enabled()
            && let Some(visible_text) =
                terminal_final_text_to_render(
                    final_assistant_text,
                    final_assistant_recorded,
                    user_visible_final_suffix,
                )
        {
            print_assistant_banner_with_app_and_skill(Some(app), active_skill_name);
            // digest 是给模型看的附加图片理解内容，最终回显同样剥离
            let visible_text = crate::ai::request::strip_digest_blocks(&visible_text);
            crate::ai::stream::render_markdown_block(&visible_text)?;
        }
        persist_pending_turn_messages_for_model(
            app,
            response_source_model,
            one_shot_mode,
            turn_messages,
            persisted_turn_messages,
        );
        // 任务边界判定：当前 turn 没有再调工具，意味着 agent 已经把答案交付，
        // 这是一个自然的"任务完成"切点；用更激进的阈值（160 turns）触发摘要，
        // 避免对话一直堆到硬上限（200 turns）才被动压缩。
        let had_tool_calls = turn_messages
            .iter()
            .any(|m| m.role == "tool" || m.tool_calls.as_ref().map_or(false, |c| !c.is_empty()));
        // goal 模式下，run_loop 通过此标志判定目标是否完成：
        // 一轮结束时没有调用任何工具 = agent 已交付最终结果。
        app.last_turn_had_tool_calls = had_tool_calls;
        // 持久化压缩：前台交互式 turn 派发到后台，避免答案交付后再等待 CPU 压缩
        // 与 SQLite 写事务（快照只是写回缓存，下一轮从 canonical 重算）。
        // one-shot、即将退出及子代理 turn 走当前 future，避免所属 history 先结束生命周期。
        // `at_boundary` = 本轮没有再调工具（答案已交付），用更激进的阈值。
        dispatch_finalize_compaction(
            app,
            !had_tool_calls,
            should_compact_session_history_in_background(
                one_shot_mode,
                should_quit,
                crate::ai::driver::runtime_ctx::has_subagent_result_slot(),
            ),
        )
        .await;
        // 尝试为当前对话生成 LLM 概括性标题（如果尚未生成且已有足够上下文）。
        // 交互式前台 turn 不应在这里等待一个后台质量任务。
        maybe_generate_session_title(
            app,
            should_generate_session_title_in_background(one_shot_mode, should_quit),
        )
        .await;
        // println!();

        let mut first_observer_emitted = false;
        let mut poisoned: Vec<String> = Vec::new();
        for obs in app.observers.iter_mut() {
            if obs.is_poisoned() {
                continue;
            }
            let ctx = crate::ai::driver::observer::FinalizeContext {
                question: question.to_string(),
                final_text: final_assistant_text.to_string(),
                had_tool_calls,
            };
            let obs_name = obs.name().to_string();
            let output = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                obs.on_finalize(&ctx)
            })) {
                Ok(o) => o,
                Err(_) => {
                    if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
                        eprintln!(
                            "[Warning] observer '{}' panicked in on_finalize; disabling for rest of conversation.",
                            obs_name
                        );
                    }
                    obs.mark_poisoned();
                    poisoned.push(obs_name);
                    continue;
                }
            };
            if output.display_lines.is_empty() {
                continue;
            }
            if first_observer_emitted && crate::ai::driver::runtime_ctx::terminal_output_enabled() {
                println!("---");
            }
            first_observer_emitted = true;
            if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
                for line in &output.display_lines {
                    println!("{}", line);
                }
            }
        }
        let _ = poisoned;
    } else {
        if crate::ai::driver::runtime_ctx::terminal_output_enabled() {
            println!("{}", format_empty_state("no response"));
        }
        app.last_turn_had_tool_calls = false;
    }

    Ok(if should_quit {
        TurnOutcome::Quit
    } else {
        TurnOutcome::Continue
    })
}

/// 在 turn 结束后尝试用 LLM 生成 session 概括性标题。
/// 条件：至少有 1 个 user turn，且没有已生成标题或现有标题质量过低。
pub(super) async fn maybe_generate_session_title(app: &App, run_in_background: bool) {
    let store = session_title_store(&app.config.history_file);
    // 已有模型标题的恢复 session 不需要为标题检查解码整段 canonical history。
    if store.has_generated_title(&app.session_id) {
        return;
    }
    if !store
        .read_all_messages(&app.session_id)
        .is_ok_and(|messages| has_session_title_source(&messages))
    {
        // 新 session 启动时没有历史，不要占住 in-flight 标记；否则用户立即提交
        // 首条输入时，真正需要的标题任务可能被这个空任务挡掉。
        return;
    }
    if !mark_session_title_generation_started(&app.session_id) {
        return;
    }

    if run_in_background {
        let task_app = app.clone();
        let session_id = task_app.session_id.clone();
        tokio::spawn(async move {
            generate_session_title_if_missing(&task_app, None).await;
            mark_session_title_generation_finished(&session_id);
        });
        return;
    }

    generate_session_title_if_missing(app, None).await;
    mark_session_title_generation_finished(&app.session_id);
}

/// 用户提交输入后立即派发标题生成，不等待该 turn 的历史落盘或模型响应。
pub(super) async fn maybe_generate_session_title_for_input(app: &App, user_input: &str) {
    if user_input.trim().is_empty() || !mark_session_title_generation_started(&app.session_id) {
        return;
    }

    let task_app = app.clone();
    let session_id = task_app.session_id.clone();
    let user_input = user_input.to_string();
    tokio::spawn(async move {
        generate_session_title_if_missing(&task_app, Some(&user_input)).await;
        mark_session_title_generation_finished(&session_id);
    });
}

async fn generate_session_title_if_missing(app: &App, pending_user_input: Option<&str>) {
    // SessionStore 接收基础 history 文件，并据此推导 `<stem>.sessions/`。
    // 传入当前 session sqlite 的父目录会额外拼出一层 `.sessions`，导致标题任务
    // 读取不到当前会话的消息。
    let store = session_title_store(&app.config.history_file);

    let persisted_messages = match store.read_all_messages(&app.session_id) {
        Ok(messages) => messages,
        // 首条输入触发时 session sqlite 可能还没由主 turn 创建；pending input 本身
        // 已足够生成标题，因此把“尚无历史文件”视为空历史，而不是错过整个首轮。
        Err(_) if pending_user_input.is_some() => Vec::new(),
        Err(_) => return,
    };
    let all_messages = session_title_messages(persisted_messages, pending_user_input);
    if !has_session_title_source(&all_messages) {
        return;
    }

    let fallback_title = fallback_session_title(&all_messages);
    let existing = store
        .read_session_title_with_origin(&app.session_id)
        .ok()
        .flatten();
    if !should_generate_model_session_title(existing.as_ref(), &fallback_title) {
        return;
    }

    let generated_title = crate::ai::request::generate_session_title_via_model(app, &all_messages)
        .await
        .map(|title| normalize_generated_session_title(&title))
        .filter(|title| !title.is_empty() && !is_low_quality_session_title(title));

    // 网络请求期间，另一个进程可能已写入标题；重新读取后只在仍可升级时覆盖。
    let current = store
        .read_session_title_with_origin(&app.session_id)
        .ok()
        .flatten();
    if let Some(title) = generated_title {
        if should_generate_model_session_title(current.as_ref(), &fallback_title) {
            if store
                .write_session_title_with_origin(&app.session_id, &title, SessionTitleOrigin::Model)
                .is_ok()
            {
                crate::ai::prompt::notify_session_title_updated(&app.session_id, &title);
            }
        }
        return;
    }

    // 失败时只补写 fallback，绝不覆盖已有的合格标题；下一次完成回合仍会重试模型。
    if !fallback_title.is_empty()
        && should_write_fallback_session_title(current.as_ref(), &fallback_title)
    {
        if store
            .write_session_title_with_origin(
                &app.session_id,
                &fallback_title,
                SessionTitleOrigin::Fallback,
            )
            .is_ok()
        {
            crate::ai::prompt::notify_session_title_updated(&app.session_id, &fallback_title);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{FunctionCall, ToolCall};

    fn tool_call(id: &str, name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    #[test]
    fn subagent_parent_result_includes_tool_evidence() {
        let turn_messages = vec![
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![tool_call(
                    "call-1",
                    "read_file",
                    serde_json::json!({"file_path":"src/lib.rs","offset":10,"limit":20}),
                )]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("    10\tfn load_config() {".to_string()),
                tool_calls: None,
                tool_call_id: Some("call-1".to_string()),
                reasoning_content: None,
            },
        ];

        let output = format_subagent_result_for_parent("done", &turn_messages);
        assert!(output.starts_with("[Subagent tool evidence]"));
        assert!(output.contains("[Subagent tool evidence]"));
        assert!(output.contains("read_file("));
        assert!(output.contains("\"file_path\":\"src/lib.rs\""));
        assert!(output.contains("fn load_config()"));
        assert!(output.contains("[Subagent final answer]\ndone"));
    }

    #[test]
    fn subagent_parent_result_without_tools_is_plain_final_text() {
        let output = format_subagent_result_for_parent("done", &[]);
        assert_eq!(output, "done");
    }

    #[test]
    fn terminal_final_text_only_renders_unstreamed_visible_fallbacks() {
        assert_eq!(
            terminal_final_text_to_render("实时正文", true, None),
            None,
            "streamed model text must not be redrawn during finalize"
        );
        assert_eq!(
            terminal_final_text_to_render(
                "[Model-authored note from an earlier turn; this is not authoritative evidence.]\nself_note:completion_evidence_required",
                false,
                None,
            ),
            None
        );
        assert_eq!(
            terminal_final_text_to_render("self_note:completion_evidence_required", false, None),
            None
        );
        assert_eq!(
            terminal_final_text_to_render(
                "[Runtime warning] Completion is unverified.",
                false,
                None,
            ),
            None
        );
        assert_eq!(
            terminal_final_text_to_render("本地 fallback。", false, None),
            Some("本地 fallback。".to_string())
        );
        assert_eq!(
            terminal_final_text_to_render("完成修复。", false, None),
            Some("完成修复。".to_string())
        );
        assert_eq!(
            terminal_final_text_to_render(
                "完成修复。\n\n[Runtime warning] Completion is unverified.",
                false,
                None,
            ),
            Some("完成修复。".to_string())
        );
        assert_eq!(
            terminal_final_text_to_render(
                "已流式输出。\n\n[Runtime warning] Completion is unverified.",
                true,
                Some("[Runtime warning] Completion is unverified."),
            ),
            Some("[Runtime warning] Completion is unverified.".to_string())
        );
    }

    #[test]
    fn subagent_result_payload_for_parent_uses_tool_evidence_without_final_text() {
        let turn_messages = vec![
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![tool_call(
                    "call-1",
                    "read_file",
                    serde_json::json!({"file_path":"src/lib.rs"}),
                )]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("pub mod ai;".to_string()),
                tool_calls: None,
                tool_call_id: Some("call-1".to_string()),
                reasoning_content: None,
            },
        ];

        let output = subagent_result_payload_for_parent("", &turn_messages)
            .expect("tool evidence should still publish to parent");
        assert!(output.starts_with("[Subagent tool evidence]"));
        assert!(output.contains("read_file("));
        assert!(output.contains("pub mod ai;"));
        assert!(!output.contains("[Subagent final answer]"));
    }

    #[test]
    fn subagent_result_payload_for_parent_is_none_when_completely_empty() {
        assert!(subagent_result_payload_for_parent("", &[]).is_none());
    }

    #[test]
    fn subagent_parent_result_keeps_evidence_after_long_final_text() {
        let turn_messages = vec![
            Message {
                role: "assistant".to_string(),
                content: Value::String(String::new()),
                tool_calls: Some(vec![tool_call(
                    "call-1",
                    "read_file",
                    serde_json::json!({"file_path":"src/lib.rs"}),
                )]),
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".to_string(),
                content: Value::String("pub mod ai;".to_string()),
                tool_calls: None,
                tool_call_id: Some("call-1".to_string()),
                reasoning_content: None,
            },
        ];

        let long_final = "x".repeat(SUBAGENT_TOOL_EVIDENCE_MAX_BLOCK_CHARS + 100);
        let output = format_subagent_result_for_parent(long_final.as_str(), &turn_messages);
        assert!(output.starts_with("[Subagent tool evidence]"));
        assert!(output.contains("[Subagent tool evidence]"));
        assert!(output.contains("read_file("));
        assert!(output.contains("pub mod ai;"));
        assert!(output.contains("[Subagent final answer]\n"));
        assert!(output.ends_with(&long_final));
    }

    #[test]
    fn session_title_fallback_and_legacy_titles_remain_eligible_for_model_upgrade() {
        let fallback = "修复 session title";
        let fallback_title = SessionTitle {
            text: fallback.to_string(),
            origin: SessionTitleOrigin::Fallback,
        };
        let legacy_fallback_title = SessionTitle {
            text: fallback.to_string(),
            origin: SessionTitleOrigin::Legacy,
        };
        let model_title = SessionTitle {
            text: "完成标题生成修复".to_string(),
            origin: SessionTitleOrigin::Model,
        };

        assert!(should_generate_model_session_title(
            Some(&fallback_title),
            fallback
        ));
        assert!(should_generate_model_session_title(
            Some(&legacy_fallback_title),
            fallback
        ));
        assert!(!should_generate_model_session_title(
            Some(&model_title),
            fallback
        ));
    }

    #[test]
    fn session_title_generation_uses_submitted_user_input_before_persistence() {
        let messages = session_title_messages(Vec::new(), Some("  修复标题显示延迟  "));

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(value_to_string(&messages[0].content), "修复标题显示延迟");
        assert_eq!(fallback_session_title(&messages), "修复标题显示延迟");
        assert!(has_session_title_source(&messages));
        assert!(!has_session_title_source(&[]));
    }

    #[test]
    fn session_title_background_policy_only_backgrounds_live_interactive_turns() {
        assert!(should_generate_session_title_in_background(false, false));
        assert!(!should_generate_session_title_in_background(true, false));
        assert!(!should_generate_session_title_in_background(false, true));
        assert!(!should_generate_session_title_in_background(true, true));
    }

    #[test]
    fn persisted_history_compaction_stays_within_subagent_lifetime() {
        assert!(should_compact_session_history_in_background(
            false, false, false
        ));
        assert!(!should_compact_session_history_in_background(
            false, false, true
        ));
        assert!(!should_compact_session_history_in_background(
            true, false, false
        ));
        assert!(!should_compact_session_history_in_background(
            false, true, false
        ));
    }

    #[test]
    fn session_title_store_reads_active_session_from_base_history_file() {
        let store = session_title_store(std::path::Path::new("/tmp/a.history.sqlite"));

        assert_eq!(
            store.session_history_file("current-session"),
            std::path::Path::new("/tmp/a.history.sessions/current-session.sqlite")
        );
    }
}
