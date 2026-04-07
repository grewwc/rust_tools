use std::{fs::File, io::Write, path::Path};

use chrono::Local;

use crate::ai::{
    history::{value_to_string, Message, SessionStore},
    types::{App, ToolCall},
};

pub fn try_handle_share_command(
    app: &mut App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let normalized = if let Some(rest) = trimmed.strip_prefix('/') {
        rest
    } else if let Some(rest) = trimmed.strip_prefix(':') {
        rest
    } else {
        return Ok(false);
    };
    if normalized != "share" && normalized != "s" {
        return Ok(false);
    }

    let default_path = format!("./session-{}.md", app.session_id);
    let output_path_str = normalized
        .split_whitespace()
        .nth(1)
        .unwrap_or(&default_path);
    let output_path = Path::new(output_path_str);

    let store = SessionStore::new(app.config.history_file.as_path());
    let messages = store.read_all_messages(&app.session_id)?;

    if messages.is_empty() {
        println!("No messages to share in current session.");
        return Ok(true);
    }

    let markdown = build_share_markdown(
        &messages,
        &app.session_id,
        &app.current_model,
        &app.current_agent,
    );

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = File::create(output_path)?;
    file.write_all(markdown.as_bytes())?;

    println!("Session shared to '{}'", output_path.display());
    Ok(true)
}

fn build_share_markdown(
    messages: &[Message],
    session_id: &str,
    model: &str,
    agent: &str,
) -> String {
    let mut md = String::new();

    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S %Z");

    md.push_str("# Session Share\n\n");
    md.push_str("| | |\n");
    md.push_str("|---|---|\n");
    md.push_str(&format!("| **Session ID** | `{}` |\n", session_id));
    md.push_str(&format!("| **Timestamp** | {} |\n", timestamp));
    md.push_str(&format!("| **Model** | {} |\n", model));
    md.push_str(&format!("| **Agent** | {} |\n", agent));
    md.push_str(&format!("| **Messages** | {} |\n", messages.len()));
    md.push_str("\n---\n\n");

    let first_user = messages.iter().find(|m| m.role == "user");
    let last_assistant = messages.iter().rev().find(|m| m.role == "assistant");

    if first_user.is_some() || last_assistant.is_some() {
        md.push_str("## Summary\n\n");
        if let Some(msg) = first_user {
            let content = value_to_string(&msg.content);
            let preview = truncate(&content, 200);
            md.push_str(&format!("**First prompt:** {}\n\n", preview));
        }
        if let Some(msg) = last_assistant {
            let content = value_to_string(&msg.content);
            let preview = truncate(&content, 200);
            md.push_str(&format!("**Last response:** {}\n\n", preview));
        }
        md.push_str("---\n\n");
    }

    md.push_str("## Conversation\n\n");

    let mut turn_index = 0;
    let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
    let mut pending_tool_results: Vec<(String, String)> = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "user" => {
                flush_tool_block(&mut md, &mut pending_tool_calls, &mut pending_tool_results);
                turn_index += 1;
                let content = value_to_string(&msg.content);
                md.push_str(&format!("### Turn {} — User\n\n", turn_index));
                md.push_str(&format!("{}\n\n", content));
            }
            "assistant" => {
                flush_tool_block(&mut md, &mut pending_tool_calls, &mut pending_tool_results);
                let content = value_to_string(&msg.content);
                if !content.is_empty() {
                    md.push_str(&format!("### Turn {} — Assistant\n\n", turn_index));
                    md.push_str(&format!("{}\n\n", content));
                }
                if let Some(ref tool_calls) = msg.tool_calls {
                    pending_tool_calls = tool_calls.clone();
                }
            }
            "tool" => {
                let content = value_to_string(&msg.content);
                if let Some(ref tool_call_id) = msg.tool_call_id {
                    pending_tool_results.push((tool_call_id.clone(), content));
                }
            }
            "system" => {
                let content = value_to_string(&msg.content);
                if content.contains("自动压缩") {
                    continue;
                }
                if !content.is_empty() {
                    md.push_str("### System\n\n");
                    md.push_str(&format!("{}\n\n", content));
                }
            }
            _ => {}
        }
    }

    flush_tool_block(&mut md, &mut pending_tool_calls, &mut pending_tool_results);

    md
}

fn flush_tool_block(
    md: &mut String,
    tool_calls: &mut Vec<ToolCall>,
    tool_results: &mut Vec<(String, String)>,
) {
    if tool_calls.is_empty() {
        return;
    }

    md.push_str("<details>\n<summary><strong>Tool Calls</strong></summary>\n\n");

    for tc in tool_calls.iter() {
        md.push_str(&format!("#### `{}`\n\n", tc.function.name));
        if !tc.function.arguments.trim().is_empty() {
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(&tc.function.arguments) {
                md.push_str(&format!("```json\n{}\n```\n\n", args));
            } else {
                md.push_str(&format!("```\n{}\n```\n\n", tc.function.arguments));
            }
        }

        let result = tool_results
            .iter()
            .find(|(id, _)| id == &tc.id)
            .map(|(_, content)| content.as_str())
            .unwrap_or("*no result*");

        let result_preview = truncate(result, 600);
        md.push_str(&format!("**Result:**\n\n```\n{}\n```\n\n", result_preview));
    }

    md.push_str("</details>\n\n");

    tool_calls.clear();
    tool_results.clear();
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let end = s
        .char_indices()
        .nth(max_chars)
        .map(|(idx, _)| idx)
        .unwrap_or_else(|| s.len());
    let mut out = s[..end].to_string();
    out.push_str("...");
    out
}
