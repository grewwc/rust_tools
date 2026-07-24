// =============================================================================
// /memo 交互命令 —— 把模型结论或指定文本保存到 memo 类型知识库
// =============================================================================
// 类似 `a -n <text>` 的交互式版本：
//   /memo              —— 把上一轮 assistant 的正文结论保存为 memo
//   /memo <text>       —— 把指定文本经模型整理后保存为 memo
// =============================================================================

use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
use crate::ai::types::App;

/// 判断输入是否为 `/memo` 命令；若是则异步执行保存，返回 `Ok(true)`。
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

    // 只接受精确的 `memo` 或 `memo ` 开头，避免误匹配类似 `/memorial` 这种。
    if !rest.is_empty() && !rest.starts_with(' ') && !rest.starts_with('\t') {
        return Ok(false);
    }

    let arg = rest.trim().to_string();
    execute_memo_save(app, arg).await?;
    Ok(true)
}

async fn execute_memo_save(
    app: &mut App,
    arg: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let store = MemoryStore::from_env_or_config();

    // 1) 确定要保存的原始文本
    let raw_text = if !arg.is_empty() {
        arg
    } else {
        // 无参数：取上一轮 assistant 的结论正文
        match last_assistant_conclusion(app)? {
            Some(text) => text,
            None => {
                eprintln!("[memo] 未找到上一轮的模型结论，请先进行一次对话，或使用 `/memo <text>` 手动指定内容。");
                return Ok(());
            }
        }
    };

    if raw_text.trim().is_empty() {
        eprintln!("[memo] 内容为空，已取消保存。");
        return Ok(());
    }

    // 2) 调用模型整理内容，使其更适合作为知识库 memo
    println!("[memo] 正在整理内容...");
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
    let note_content = match crate::ai::request::do_request_json(app, &model, &messages, false, false).await {
        Ok(response) => crate::ai::request::extract_response_text(&response)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or(raw_text.clone()),
        Err(err) => {
            eprintln!("[memo] 整理失败，保存原始输入: {}", err);
            raw_text.clone()
        }
    };

    // 3) 保存到知识库 memo 类别
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
            println!("[memo] 已保存到知识库 [memo]：");
            let preview: String = note_content.chars().take(200).collect();
            println!("  {}", preview);
            if note_content.chars().count() > 200 {
                println!("  ...");
            }
        }
        Err(err) => {
            eprintln!("[memo] 保存失败: {}", err);
            return Err(err.into());
        }
    }

    Ok(())
}

/// 从会话历史中找到最近一条不含 tool_calls 的 assistant 消息正文。
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

/// 从历史消息的 content 字段中提取纯文本。
/// content 可能是字符串，也可能是内容块数组（多模态）。
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
