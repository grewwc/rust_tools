use serde_json::Value;

use super::{compress::value_to_string, types::Message};

pub(in crate::ai) fn messages_to_markdown(messages: &[Message], session_id: &str) -> String {
    let mut md = String::new();
    md.push_str(&format!("# Session: {}\n\n", session_id));
    md.push_str(&format!("**Total messages:** {}\n\n", messages.len()));
    md.push_str("---\n\n");

    for (i, msg) in messages.iter().enumerate() {
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

    md
}
