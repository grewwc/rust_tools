use serde_json::Value;

use crate::ai::{
    history::Message,
    request::{self, build_content},
    types::App,
};
use crate::commonw::configw;

pub(super) fn reflection_filtered(question: &str, answer: &str, turn_messages: &Vec<Message>) -> bool {
    let cfg = configw::get_all_config();
    let enabled = !cfg
        .get_opt("ai.reflection.filter.enable")
        .unwrap_or_else(|| "false".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if !enabled {
        return false;
    }
    let min_q = cfg
        .get_opt("ai.reflection.filter.min_question_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let min_a = cfg
        .get_opt("ai.reflection.filter.min_answer_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(80);
    let require_tool = cfg
        .get_opt("ai.reflection.filter.require_tool_or_long")
        .unwrap_or_else(|| "false".to_string())
        .trim()
        .eq_ignore_ascii_case("true");
    let q = question.trim();
    let a = answer.trim();
    if q.chars().count() < min_q && a.chars().count() < min_a {
        return true;
    }
    if require_tool && !turn_has_tool(turn_messages) && a.chars().count() < min_a {
        return true;
    }
    false
}

pub(super) fn critic_filtered(question: &str, draft: &str) -> bool {
    let cfg = configw::get_all_config();
    let min_q = cfg
        .get_opt("ai.critic_revise.filter.min_question_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let min_a = cfg
        .get_opt("ai.critic_revise.filter.min_answer_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(120);
    let q = question.trim();
    let a = draft.trim();
    q.chars().count() < min_q && a.chars().count() < min_a
}

pub(super) async fn model_should_reflect(
    app: &mut App,
    model: &str,
    question: &str,
    answer: &str,
    had_tool: bool,
) -> Option<bool> {
    use tokio::time::{Duration, timeout};
    let cfg = configw::get_all_config();
    let to_ms = cfg
        .get_opt("ai.reflection.model_gate.timeout_ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(2000);
    let system = "You are a binary classifier that decides whether to capture a short 'experience note' for future turns.\nReturn STRICT JSON ONLY with the shape: {\"reflect\": true|false}.\nRules:\n- reflect=true when Q/A contains non-trivial reasoning, code, multi-step instructions, tool usage outcomes, errors/diagnosis, or decisions that should guide future behavior.\n- reflect=false for greetings, acknowledgements, trivial answers, or very short single-turn exchanges with no actionable guidance.\nDo not include explanations or extra text.";
    let user = format!(
        "question:\n{}\n\nanswer:\n{}\n\nhad_tool:\n{}",
        question.trim(),
        answer.trim(),
        if had_tool { "true" } else { "false" }
    );
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(system.to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".to_string(),
            content: build_content(model, &user, &[])
                .unwrap_or(Value::String(user)),
            tool_calls: None,
            tool_call_id: None,
        },
    ];
    let fut = request::do_request_messages(app, model, &messages, false);
    let resp = match timeout(Duration::from_millis(to_ms), fut).await {
        Ok(Ok(r)) => r,
        _ => return None,
    };
    let text = resp.text().await.ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = super::background::extract_content(&v).unwrap_or_default();
    parse_reflect_flag(&content)
}

pub(super) fn parse_reflect_flag(s: &str) -> Option<bool> {
    let trimmed = s.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return v.get("reflect").and_then(|b| b.as_bool());
    }
    let l = trimmed.find('{')?;
    let r = trimmed.rfind('}')?;
    if r < l {
        return None;
    }
    let sub = &trimmed[l..=r];
    serde_json::from_str::<Value>(sub)
        .ok()
        .and_then(|v| v.get("reflect").and_then(|b| b.as_bool()))
}

pub(super) fn turn_uses_repo_inspection_tools(messages: &Vec<Message>) -> bool {
    const REPO_INSPECTION_TOOLS: &[&str] = &[
        "code_search",
        "read_file",
        "read_file_lines",
        "list_directory",
        "search_files",
        "grep_search",
        "execute_command",
    ];
    messages.iter().any(|message| {
        message
            .tool_calls
            .as_ref()
            .map(|calls| {
                calls.iter().any(|call| {
                    REPO_INSPECTION_TOOLS
                        .iter()
                        .any(|name| call.function.name == *name)
                })
            })
            .unwrap_or(false)
    })
}

pub(super) fn answer_looks_unstable_for_writeback(answer: &str) -> bool {
    let lower = answer.to_lowercase();
    if answer.chars().count() < 40 {
        return true;
    }
    [
        "[本轮请求失败",
        "i'm sorry",
        "不确定",
        "可能",
        "猜测",
        "大概",
        "无法确认",
        "need to verify",
        "might be",
    ]
    .iter()
    .any(|needle| lower.contains(&needle.to_lowercase()))
}

pub(super) fn turn_has_tool(messages: &Vec<Message>) -> bool {
    for m in messages {
        if m.role == "tool" {
            return true;
        }
        if let Some(calls) = m.tool_calls.as_ref()
            && !calls.is_empty()
        {
            return true;
        }
    }
    false
}

pub(super) async fn model_should_revise(
    app: &mut App,
    model: &str,
    question: &str,
    draft: &str,
) -> Option<bool> {
    let system = "You decide if the DRAFT answer should be CRITIC→REVISE refined.\nReturn STRICT JSON ONLY: {\"revise\": true|false}.\nRules:\n- true ONLY for software engineering tasks: code writing/review/debug/refactor, tool execution results, build/test errors, patch proposals.\n- false for general knowledge, Q&A like weather/news/sports/finance, travel, generic suggestions, or casual chat.\n- false when the answer is short and already sufficient without code/steps.\nNo extra text.";
    let user = format!("QUESTION:\n{}\n\nDRAFT:\n{}", question.trim(), draft.trim());
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: Value::String(system.to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".to_string(),
            content: build_content(model, &user, &[])
                .unwrap_or(Value::String(user)),
            tool_calls: None,
            tool_call_id: None,
        },
    ];
    let resp = request::do_request_messages(app, model, &messages, false)
        .await
        .ok()?;
    let text = resp.text().await.ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let content = super::background::extract_content(&v).unwrap_or_default();
    if let Ok(v2) = serde_json::from_str::<Value>(content.trim()) {
        return v2.get("revise").and_then(|b| b.as_bool());
    }
    None
}

pub(super) fn reflection_filtered_bg(question: &str, answer: &str, had_tool: bool) -> bool {
    let cfg = configw::get_all_config();
    let enabled = !cfg
        .get_opt("ai.reflection.filter.enable")
        .unwrap_or_else(|| "false".to_string())
        .trim()
        .eq_ignore_ascii_case("false");
    if !enabled {
        return false;
    }
    let min_q = cfg
        .get_opt("ai.reflection.filter.min_question_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let min_a = cfg
        .get_opt("ai.reflection.filter.min_answer_chars")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(80);
    let require_tool = cfg
        .get_opt("ai.reflection.filter.require_tool_or_long")
        .unwrap_or_else(|| "false".to_string())
        .trim()
        .eq_ignore_ascii_case("true");
    let q = question.trim();
    let a = answer.trim();
    if q.chars().count() < min_q && a.chars().count() < min_a {
        return true;
    }
    if require_tool && !had_tool && a.chars().count() < min_a {
        return true;
    }
    false
}
