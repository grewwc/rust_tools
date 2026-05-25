use serde_json::Value;

use super::{compress::value_to_string, types::Message};

/// 默认硬上限：避免在极长 session 下构造 100MB+ 的中间字符串再截断。
/// 调用方（如 summarize_history_via_model）即使只想要 12K 也会先在内存里
/// 拼整个 markdown，因此这里加边界式累计是必要的兜底。
const MESSAGES_TO_MARKDOWN_HARD_CAP: usize = 256 * 1024;

pub(in crate::ai) fn messages_to_markdown(messages: &[Message], session_id: &str) -> String {
    messages_to_markdown_capped(messages, session_id, MESSAGES_TO_MARKDOWN_HARD_CAP)
}

/// 显式带字符预算的导出版本。预算用尽时立即终止追加，并在末尾标注被裁掉的消息数量。
/// 单测 / sessions export 等需要更大预算的场景可直接调用本函数。
pub(in crate::ai) fn messages_to_markdown_capped(
    messages: &[Message],
    session_id: &str,
    char_budget: usize,
) -> String {
    let mut md = String::new();
    md.push_str(&format!("# Session: {}\n\n", session_id));
    md.push_str(&format!("**Total messages:** {}\n\n", messages.len()));
    md.push_str("---\n\n");

    let mut truncated_at: Option<usize> = None;
    for (i, msg) in messages.iter().enumerate() {
        if md.len() >= char_budget {
            truncated_at = Some(i);
            break;
        }
        let role_emoji = match msg.role.as_str() {
            "user" => "👤",
            "assistant" => "🤖",
            "system" => "⚙️",
            "tool" => "🔧",
            _ => "📝",
        };

        md.push_str(&format!(
            "### {} {}\n\n",
            role_emoji,
            msg.role.to_uppercase()
        ));

        let content_str = value_to_string(&msg.content);
        if !content_str.is_empty() {
            md.push_str(&content_str);
            md.push_str("\n\n");
        }

        if let Some(ref tool_calls) = msg.tool_calls {
            md.push_str("**Tool Calls:**\n");
            for tc in tool_calls {
                md.push_str(&format!("- `{}`", tc.function.name));
                if !tc.function.arguments.trim().is_empty() {
                    if let Ok(args_val) = serde_json::from_str::<Value>(&tc.function.arguments) {
                        md.push_str(&format!("({})", args_val));
                    } else {
                        md.push_str(&format!("({})", tc.function.arguments));
                    }
                }
                md.push('\n');
            }
            md.push('\n');
        }

        if let Some(ref tool_call_id) = msg.tool_call_id {
            md.push_str(&format!("**Tool Call ID:** `{}`\n\n", tool_call_id));
        }

        if i < messages.len() - 1 {
            md.push_str("---\n\n");
        }
    }

    if let Some(stop_at) = truncated_at {
        let omitted = messages.len() - stop_at;
        md.push_str(&format!(
            "\n---\n\n_[truncated: {} message(s) omitted to fit {} char budget]_\n",
            omitted, char_budget
        ));
    }

    md
}
