// =============================================================================
// /memo interactive command: save a model conclusion or given text into the
// memo-type knowledge base
// =============================================================================
// An interactive counterpart of `a -n <text>`:
//   /memo              -- save the previous assistant turn's body text as a memo
//   /memo <text>       -- have the model refine the given text and save it as a memo
// =============================================================================

use super::status_line::{clear_status, print_status, show_status};
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use crate::ai::types::App;

/// Return `Ok(true)` when the input is a `/memo` command and run the save asynchronously.
pub(crate) async fn try_handle_memo_command(
    app: &mut App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let rest = if let Some(r) = trimmed.strip_prefix("/memo") {
        r
    } else if let Some(r) = trimmed.strip_prefix(":memo") {
        r
    } else {
        return Ok(false);
    };

    // Only accept an exact `memo` or `memo ` prefix to avoid matching things like `/memorial`.
    if !rest.is_empty() && !rest.starts_with(' ') && !rest.starts_with('\t') {
        return Ok(false);
    }

    let arg = rest.trim().to_string();
    execute_memo_save(app, arg).await?;
    Ok(true)
}

async fn execute_memo_save(app: &mut App, arg: String) -> Result<(), Box<dyn std::error::Error>> {
    let store = MemoryStore::from_env_or_config();

    // 1) Determine the raw text to save
    let raw_text = if !arg.is_empty() {
        arg
    } else {
        // No argument: take the previous assistant turn's conclusion body
        match last_assistant_conclusion(app)? {
            Some(text) => text,
            None => {
                show_status(
                    "[memo] 未找到上一轮的模型结论，请先进行一次对话，或使用 /memo <text> 手动指定内容。",
                );
                return Ok(());
            }
        }
    };

    if raw_text.trim().is_empty() {
        show_status("[memo] 内容为空，已取消保存。");
        return Ok(());
    }

    // 2) Have the model tidy up the content so it is more suitable as a knowledge-base memo
    print_status("[memo] 正在整理内容...");
    let model = crate::ai::models::initial_model(&app.cli);
    let messages = vec![
        serde_json::json!({
            "role": "system",
            "content": "你是一个笔记整理助手。请把用户输入的内容理解、整理、改写为一条清晰、结构化、便于日后检索的笔记。\
                        保留所有关键信息和事实，去除口语化冗余，必要时用简洁的要点组织。直接输出整理后的笔记正文，不要添加任何解释或前后缀。用中文回答。",
        }),
        serde_json::json!({
            "role": "user",
            "content": raw_text,
        }),
    ];
    let note_content =
        match crate::ai::request::do_request_json(app, &model, &messages, false, false).await {
            Ok(response) => crate::ai::request::extract_response_text(&response)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or(raw_text.clone()),
            Err(err) => {
                clear_status();
                show_status(&format!("[memo] 整理失败，保存原始输入: {}", err));
                raw_text.clone()
            }
        };

    clear_status();

    // 3) Save to the memo category in the knowledge base
    let now = chrono::Local::now().to_rfc3339();
    let entry = AgentMemoryEntry {
        id: Some(format!("mem_{}", uuid::Uuid::new_v4().simple())),
        timestamp: now,
        category: "memo".to_string(),
        note: note_content.clone(),
        tags: vec![],
        source: Some("interactive_memo".to_string()),
        priority: Some(150),
        owner_pid: None,
        owner_pgid: None,
        image_path: None,
    };

    match store.append(&entry) {
        Ok(()) => {
            let preview: String = note_content.chars().take(80).collect();
            let suffix = if note_content.chars().count() > 80 {
                "..."
            } else {
                ""
            };
            show_status(&format!("[memo] 已保存: {}{}", preview, suffix));
        }
        Err(err) => {
            show_status(&format!("[memo] 保存失败: {}", err));
            return Err(err.into());
        }
    }

    Ok(())
}

/// Find the body text of the most recent assistant message without tool_calls in session history.
fn last_assistant_conclusion(app: &App) -> Result<Option<String>, Box<dyn std::error::Error>> {
    use crate::ai::history;

    let history_file = &app.session_history_file;
    let messages = history::build_message_arr(usize::MAX, history_file)?;

    Ok(messages.iter().rev().find_map(|message| {
        if message.role != "assistant" {
            return None;
        }
        let has_tool_calls = message
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty());
        if has_tool_calls {
            return None;
        }
        let text = searchable_content(&message.content);
        if text.trim().is_empty() {
            return None;
        }
        Some(text)
    }))
}

/// Extract plain text from a historical message's content field.
/// content may be a string or an array of content blocks (multimodal).
fn searchable_content(content: &serde_json::Value) -> String {
    if let Some(s) = content.as_str() {
        return s.to_string();
    }
    if let Some(arr) = content.as_array() {
        let mut out = String::new();
        for item in arr {
            if let Some(ty) = item.get("type").and_then(|v| v.as_str()) {
                if ty == "text" {
                    if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                        if !out.is_empty() {
                            out.push('\n');
                        }
                        out.push_str(t);
                    }
                }
            }
        }
        return out;
    }
    String::new()
}
