use serde_json::Value;

use crate::ai::{
    history::Message,
    request::{self, build_content},
    types::App,
};
use crate::commonw::configw;

pub(super) fn reflection_filtered(
    question: &str,
    answer: &str,
    turn_messages: &Vec<Message>,
) -> bool {
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
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: build_content(model, &user, &[]).unwrap_or(Value::String(user)),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
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

/// 解析 LLM 返回的 reflect 标志。
///
/// 鲁棒性顺序：
/// 1. 整段就是合法 JSON —— 直接读 `reflect`
/// 2. 否则扫描所有"括号深度配平"的候选段，逐个尝试解析
///    （处理 `Some text {"a":1} more {"reflect": true}` 这种混合输出）
/// 3. 仍未找到则返回 None
pub(super) fn parse_reflect_flag(s: &str) -> Option<bool> {
    let trimmed = s.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed)
        && let Some(b) = v.get("reflect").and_then(|b| b.as_bool())
    {
        return Some(b);
    }
    for candidate in iter_balanced_json_objects(trimmed) {
        if let Ok(v) = serde_json::from_str::<Value>(&candidate)
            && let Some(b) = v.get("reflect").and_then(|b| b.as_bool())
        {
            return Some(b);
        }
    }
    None
}

/// 从字符串中按 `{` / `}` 深度配平规则提取所有顶层 JSON 对象候选段。
///
/// 该实现简单稳健：忽略字符串字面量内部的 `{` `}` 转义并不重要，
/// 因为 serde_json 解析失败的候选段会被自动跳过。
fn iter_balanced_json_objects(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            // 寻找深度归零的位置
            let start = i;
            let mut depth = 0i32;
            let mut in_str = false;
            let mut escape = false;
            let mut j = i;
            while j < bytes.len() {
                let c = bytes[j];
                if in_str {
                    if escape {
                        escape = false;
                    } else if c == b'\\' {
                        escape = true;
                    } else if c == b'"' {
                        in_str = false;
                    }
                } else {
                    match c {
                        b'"' => in_str = true,
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            if depth == 0 {
                                // [start..=j] 是一段配平的对象
                                if let Ok(slice) = std::str::from_utf8(&bytes[start..=j]) {
                                    out.push(slice.to_string());
                                }
                                i = j + 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                j += 1;
            }
            if depth != 0 {
                // 未配平：不再前进会无限循环
                break;
            }
            continue;
        }
        i += 1;
    }
    out
}

pub(super) fn turn_uses_repo_inspection_tools(messages: &Vec<Message>) -> bool {
    const REPO_INSPECTION_TOOLS: &[&str] = &[
        "code_search",
        "read_file",
        "read_file_lines",
        "list_directory",
        "find_path",
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

/// 判断答案是否"不稳定"，不应作为长期经验/记忆回写。
///
/// 设计目标：宁可放过 false negative，也不要把日常技术答复（含"可能"/"大概"
/// 这类高频副词）误判为不稳定。
///
/// 命中条件（任一即视为不稳定）：
/// 1. 系统级失败标记（错误回退、空答复）
/// 2. 模型显式宣称"无法回答 / 不知道 / 我不确定"——必须是"否定 + 自指"的明确句式
/// 3. 答案极短（< 24 字符），且未呈现"实质内容"信号（代码块、列表、引用、数字、英文实词）
pub(super) fn answer_looks_unstable_for_writeback(answer: &str) -> bool {
    let trimmed = answer.trim();
    if trimmed.is_empty() {
        return true;
    }
    let lower = trimmed.to_lowercase();

    // 1) 系统级失败标记
    if lower.starts_with("[本轮请求失败") || lower.starts_with("[turn failed") {
        return true;
    }

    // 2) 模型自指否定 —— 必须是完整短语，避免误杀含"可能/大概"的正常回答
    const SELF_NEGATION_PHRASES: &[&str] = &[
        "i don't know",
        "i do not know",
        "i'm not sure",
        "i am not sure",
        "i cannot answer",
        "i can't answer",
        "i'm sorry, i ",
        "我不知道",
        "我无法回答",
        "我无法确认",
        "我不能确定",
        "我没法回答",
        "无法给出明确",
    ];
    if SELF_NEGATION_PHRASES.iter().any(|p| lower.contains(p)) {
        return true;
    }

    // 3) 极短答复且无实质内容
    if trimmed.chars().count() < 24 && !looks_substantive(trimmed) {
        return true;
    }

    false
}

/// 是否包含"实质内容"信号（即便很短也很可能是有用的回答）。
fn looks_substantive(s: &str) -> bool {
    // 代码块 / 行内代码 / 命令
    if s.contains("```") || s.contains('`') {
        return true;
    }
    // 列表 / 引用 / 链接 / 路径
    for marker in ["- ", "* ", "1.", "http://", "https://", "/", "::", "->"] {
        if s.contains(marker) {
            return true;
        }
    }
    // 含数字（如 PR 号、行号、版本号）
    if s.chars().any(|c| c.is_ascii_digit()) {
        return true;
    }
    // 含 Unicode 实词字符（>= 2 个连续字母——含 ASCII / CJK / 假名 etc.）。
    // `char::is_alphabetic` 基于 Unicode 属性，CJK 字符也会被识别。
    let mut run = 0usize;
    for c in s.chars() {
        if c.is_alphabetic() {
            run += 1;
            if run >= 2 {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
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
            reasoning_content: None,
        },
        Message {
            role: "user".to_string(),
            content: build_content(model, &user, &[]).unwrap_or(Value::String(user)),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_reflect_flag_strict_json() {
        assert_eq!(parse_reflect_flag(r#"{"reflect": true}"#), Some(true));
        assert_eq!(parse_reflect_flag(r#"{"reflect": false}"#), Some(false));
    }

    #[test]
    fn parse_reflect_flag_with_surrounding_text() {
        let s = r#"Here is my analysis: {"context": "stuff"} and the answer is {"reflect": true}"#;
        assert_eq!(parse_reflect_flag(s), Some(true));
    }

    #[test]
    fn parse_reflect_flag_handles_nested_braces() {
        let s = r#"prefix {"data": {"nested": 1}} {"reflect": false} suffix"#;
        assert_eq!(parse_reflect_flag(s), Some(false));
    }

    #[test]
    fn parse_reflect_flag_returns_none_when_missing() {
        assert_eq!(parse_reflect_flag("no json here"), None);
        assert_eq!(parse_reflect_flag(r#"{"other": 1}"#), None);
    }

    #[test]
    fn answer_unstable_keeps_normal_chinese_answer() {
        // 含"可能/大概/猜测"的常规技术回答不再被误判
        assert!(!answer_looks_unstable_for_writeback(
            "可能的优化方案如下：先看 N+1 查询，然后看缓存策略。"
        ));
        assert!(!answer_looks_unstable_for_writeback(
            "不需要猜测，可以直接查看 git log 验证。"
        ));
        assert!(!answer_looks_unstable_for_writeback(
            "大概率可以编译通过，建议跑一遍测试。"
        ));
    }

    #[test]
    fn answer_unstable_flags_self_negation() {
        assert!(answer_looks_unstable_for_writeback(
            "我不知道这个错误的具体原因，建议查看完整日志。"
        ));
        assert!(answer_looks_unstable_for_writeback(
            "I don't know the answer to this question."
        ));
    }

    #[test]
    fn answer_unstable_flags_system_failure() {
        assert!(answer_looks_unstable_for_writeback(
            "[本轮请求失败] timeout"
        ));
    }

    #[test]
    fn answer_unstable_short_no_substance() {
        assert!(answer_looks_unstable_for_writeback("。。。"));
        assert!(answer_looks_unstable_for_writeback(""));
        // 短但有实质内容（代码、数字）就不算 unstable
        assert!(!answer_looks_unstable_for_writeback("`Vec<u8>`"));
        assert!(!answer_looks_unstable_for_writeback("PR #1234"));
        assert!(!answer_looks_unstable_for_writeback("好的，已修复 bug"));
    }
}
