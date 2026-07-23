// =============================================================================
// Note/Memo Search + Knowledge Consolidation CLI Subsystem
// =============================================================================
// 从 driver/mod.rs 抽离的笔记/备忘录搜索 + 知识整理 CLI 子功能。
// =============================================================================

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use rustc_hash::FxHashSet;

use crate::ai::cli::ParsedCli;
use crate::ai::types::{App, clear_stream_cancel};

use super::signal::ForegroundTurnGuard;

/// Read recent history entries from the session file.
/// Used by auto-routing to understand conversation context.
pub(super) fn read_recent_history(app: &App) -> Vec<crate::ai::history::Message> {
    use crate::ai::history::{build_message_arr, read_recent_messages_sqlite};

    let is_sqlite_history = matches!(
        app.session_history_file
            .extension()
            .and_then(|ext| ext.to_str()),
        Some("sqlite") | Some("db")
    );

    if is_sqlite_history {
        return read_recent_messages_sqlite(app.session_history_file.as_path(), 10)
            .unwrap_or_default();
    }

    build_message_arr(10, &app.session_history_file)
        .map(|entries| entries.into_iter().rev().collect())
        .unwrap_or_default()
}

pub(super) fn note_search_interactive_mode(cli: &ParsedCli) -> bool {
    cli.note_search && cli.interactive
}

/// 如果剪贴板有图片，使用视觉模型理解内容；
/// 否则使用 `-n` 后面提供的文本；若也没有文本，则进入多行输入框让用户输入。
pub(super) async fn handle_note_save(app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
    use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
    use arboard::Clipboard;
    use image::buffer::ConvertBuffer;
    use image::{ImageBuffer, Rgb, Rgba};
    use std::fs;

    let store = MemoryStore::from_env_or_config();
    // -n 是字符串 flag，只会捕获其后的第一个 token（如 `a -n aeolus 线上日志路径：...`
    // 只会把 "aeolus" 当作 note 值），其余 token 落到位置参数里。这里把位置参数拼接回来，
    // 避免内容被截断、导致后续检索不到完整笔记。
    let provided_text = {
        let mut parts: Vec<String> = Vec::new();
        if let Some(text) = app.cli.note.clone() {
            let text = text.trim();
            if !text.is_empty() {
                parts.push(text.to_string());
            }
        }
        let extra = app.cli.args.join(" ");
        let extra = extra.trim();
        if !extra.is_empty() {
            parts.push(extra.to_string());
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" "))
        }
    };

    // 图片持久化目录：与 memory 文件同目录下的 note_images/。
    // 之前的实现把截图写进 /tmp 然后立即删除、并存 image_path: None，
    // 导致图片彻底丢失、memo 永远无法再引用原图。改成持久化保存。
    let images_dir = store
        .path()
        .parent()
        .map(|parent| parent.join("note_images"))
        .unwrap_or_else(|| PathBuf::from("note_images"));

    // 尝试从剪贴板获取图片并持久化保存
    let clipboard_image_path: Option<String> = match Clipboard::new() {
        Ok(mut clipboard) => {
            if let Ok(image) = clipboard.get_image() {
                let data = image.bytes;
                if !data.is_empty() {
                    let image_buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(
                        image.width as u32,
                        image.height as u32,
                        data.to_vec(),
                    );
                    if let Some(buf) = image_buf {
                        let rgb_buf: ImageBuffer<Rgb<u8>, Vec<u8>> = buf.convert();
                        if let Err(err) = fs::create_dir_all(&images_dir) {
                            eprintln!("[note] Failed to create image dir: {}", err);
                            None
                        } else {
                            let file_name = format!(
                                "note_{}_{}.png",
                                chrono::Local::now().format("%Y%m%d_%H%M%S"),
                                std::process::id()
                            );
                            let save_path = images_dir.join(file_name);
                            if rgb_buf.save(&save_path).is_ok() {
                                Some(save_path.to_string_lossy().into_owned())
                            } else {
                                None
                            }
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        }
        Err(_) => None,
    };

    let note_content = if let Some(image_path) = &clipboard_image_path {
        // 有图片，调用视觉模型理解内容
        println!("[note] Detected image in clipboard, analyzing...");

        let model = crate::ai::models::default_vl_model();

        // 构建包含图片的消息
        let content = crate::ai::request::build_content(
            &model,
            "请详细描述这张图片的内容，包括关键信息、文字、数据等。用中文回答。",
            &[image_path.clone()],
        )?;

        let messages = vec![serde_json::json!({
            "role": "user",
            "content": content,
        })];

        // 调用模型
        match crate::ai::request::do_request_json(app, &model, &messages, false, false).await {
            Ok(response) => crate::ai::request::extract_response_text(&response)
                .unwrap_or_else(|| "无法获取模型响应".to_string()),
            Err(err) => {
                eprintln!("[note] Failed to analyze image: {}", err);
                let _ = fs::remove_file(image_path);
                return Err(err);
            }
        }
    } else {
        // 没有图片：取得原始文本（来自 -n 后面的文本，或多行输入框），
        // 统一先交给模型理解、整理后再保存，避免直接堆原文。
        let raw = if let Some(text) = provided_text.filter(|t| !t.trim().is_empty()) {
            text
        } else {
            // 既没有图片也没有文本：进入多行输入框，让用户手动输入要保存的内容。
            println!("[note] 剪贴板没有图片，请输入要保存的内容（多行；提交后保存，留空取消）：");
            let input = match app.prompt_editor.as_mut() {
                Some(editor) => editor.read_multi_line().ok().flatten(),
                None => None,
            };
            match input {
                Some(s) if !s.trim().is_empty() => s,
                _ => {
                    eprintln!("[note] 未输入任何内容，已取消");
                    return Err("no content to save".into());
                }
            }
        };

        // 调用模型理解并整理用户输入，使其更适合作为知识库 memo。
        println!("[note] 正在整理内容...");
        let model = crate::ai::models::initial_model(&app.cli);
        let messages = vec![
            serde_json::json!({
                "role": "system",
                "content": "你是一个笔记整理助手。请把用户输入的内容理解、整理、改写为一条清晰、结构化、便于日后检索的笔记。\
                            保留所有关键信息和事实，去除口语化冗余，必要时用简洁的要点组织。直接输出整理后的笔记正文，不要添加任何解释或前后缀。用中文回答。",
            }),
            serde_json::json!({
                "role": "user",
                "content": raw,
            }),
        ];
        match crate::ai::request::do_request_json(app, &model, &messages, false, false).await {
            Ok(response) => crate::ai::request::extract_response_text(&response)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or(raw),
            Err(err) => {
                // 整理失败时退回保存原始输入，避免丢失用户内容。
                eprintln!("[note] 整理失败，保存原始输入: {}", err);
                raw
            }
        }
    };

    // 保存到知识库（图片已持久化，路径写入 image_path 以便后续引用）
    let now = chrono::Local::now().to_rfc3339();
    let entry = AgentMemoryEntry {
        id: Some(format!("mem_{}", uuid::Uuid::new_v4().simple())),
        timestamp: now,
        category: "memo".to_string(),
        note: note_content.clone(),
        tags: vec![],
        source: Some("cli_note".to_string()),
        priority: Some(150),
        owner_pid: None,
        owner_pgid: None,
        image_path: clipboard_image_path.clone(),
    };

    match store.append(&entry) {
        Ok(()) => {
            if let Some(image_path) = &clipboard_image_path {
                println!(
                    "[note] Image content saved to knowledge base [memo] (image: {}):",
                    image_path
                );
            } else {
                println!("[note] Saved to knowledge base [memo]:");
            }
            println!("  {}", note_content.chars().take(200).collect::<String>());
            if note_content.chars().count() > 200 {
                println!("  ...");
            }
        }
        Err(err) => {
            eprintln!("[note] Failed to save: {}", err);
            return Err(err.into());
        }
    }
    Ok(())
}

/// 一个轻量的终端 "Searching..." 动画提示。
///
/// 在 stderr 上用回车 `\r` 原地刷新一帧帧 spinner，`stop()` / drop 时清除当前行，
/// 不会污染随后的正式输出（正式结果走 stdout）。仅在 stderr 为 TTY 时启用，
/// 管道 / 重定向场景自动静默，避免写入垃圾字符。
struct SearchSpinner {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl SearchSpinner {
    fn start(label: &str) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        // 非 TTY（被管道/重定向）时不画动画，返回一个空 spinner。
        if !std::io::IsTerminal::is_terminal(&std::io::stderr()) {
            return Self { stop, handle: None };
        }
        let label = label.to_string();
        let stop_cloned = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            use std::io::Write as _;
            const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let mut i = 0usize;
            while !stop_cloned.load(Ordering::Relaxed) {
                let mut err = std::io::stderr();
                let _ = write!(err, "\r{} {}...", FRAMES[i % FRAMES.len()], label);
                let _ = err.flush();
                i += 1;
                std::thread::sleep(Duration::from_millis(80));
            }
            // 清除当前行（足够覆盖 "<frame> <label>..."）。
            let mut err = std::io::stderr();
            let _ = write!(err, "\r{}\r", " ".repeat(label.len() + 8));
            let _ = err.flush();
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }

    fn stop(self) {
        // 显式消费，触发 Drop。
        drop(self);
    }
}

impl Drop for SearchSpinner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

const NOTE_SEARCH_QUERY_HISTORY_MAX_MESSAGES: usize = 4;
const NOTE_SEARCH_QUERY_HISTORY_MAX_CHARS: usize = 200;

fn truncate_note_search_excerpt(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }
    let mut out = trimmed.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}

fn build_note_search_retrieval_query(
    question: &str,
    recent_history: &[crate::ai::history::Message],
) -> String {
    let question = question.trim();
    if question.is_empty() {
        return String::new();
    }

    let snippets = recent_history
        .iter()
        .filter(|message| matches!(message.role.as_str(), "user" | "assistant"))
        .filter_map(|message| {
            let content = crate::ai::history::value_to_string(&message.content);
            let content =
                truncate_note_search_excerpt(&content, NOTE_SEARCH_QUERY_HISTORY_MAX_CHARS);
            if content.is_empty() {
                return None;
            }
            let role = if message.role == "user" {
                "用户"
            } else {
                "助手"
            };
            Some(format!("{role}: {content}"))
        })
        .take(NOTE_SEARCH_QUERY_HISTORY_MAX_MESSAGES)
        .collect::<Vec<_>>();
    let mut snippets = snippets;
    snippets.reverse();

    if snippets.is_empty() {
        return question.to_string();
    }

    format!(
        "当前问题：{question}\n最近对话上下文：\n{}",
        snippets.join("\n")
    )
}

fn build_note_search_chat_history(
    app: &App,
    history_count: usize,
) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
    let overflow_dir = {
        let store = crate::ai::history::SessionStore::new(app.config.history_file.as_path());
        Some(store.session_assets_dir(&app.session_id))
    };
    let history = crate::ai::history::build_context_history(
        history_count,
        &app.session_history_file,
        app.config.history_max_chars,
        app.config.history_keep_last,
        app.config.history_summary_max_chars,
        overflow_dir,
    )?;

    Ok(history
        .into_iter()
        .filter(|message| matches!(message.role.as_str(), "user" | "assistant"))
        .filter_map(|message| {
            let content = crate::ai::history::value_to_string(&message.content);
            let content = content.trim().to_string();
            if content.is_empty() {
                return None;
            }
            Some(serde_json::json!({
                "role": message.role,
                "content": content,
            }))
        })
        .collect())
}

fn select_note_search_candidates<'a>(
    candidates: &'a [crate::ai::tools::service::memory::ScoredMemo],
) -> Vec<&'a crate::ai::tools::service::memory::ScoredMemo> {
    if candidates
        .first()
        .is_some_and(|candidate| candidate.semantic)
    {
        let top = candidates[0].score.max(1e-6);
        let threshold = top * 0.6;
        candidates
            .iter()
            .enumerate()
            .filter(|(index, candidate)| *index == 0 || candidate.score >= threshold)
            .take(8)
            .map(|(_, candidate)| candidate)
            .collect()
    } else {
        candidates.iter().collect()
    }
}

async fn answer_memo_search(
    app: &App,
    question: &str,
    history_count: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let question = question.trim().to_string();
    if question.is_empty() {
        eprintln!("[note-search] 用法: a -ns <查询内容>");
        return Err("note-search requires a query".into());
    }

    // 安装远程 embedding provider（若已配置）。必须在任何 embedder::is_ready()
    // 调用之前执行——GLOBAL_PROVIDER 是 OnceLock，首次读取即定型。
    // 未配置 / 配置不全时此调用无副作用，检索退回 BM25/lexical。
    crate::ai::knowledge::indexing::embedder::warm_up();

    // 检索 + 模型总结都可能耗时，给一个 "Searching..." 动画提示（输出前自动清除）。
    let _spinner = SearchSpinner::start("Searching memo");
    let retrieval_query = if note_search_interactive_mode(&app.cli) {
        build_note_search_retrieval_query(&question, &read_recent_history(app))
    } else {
        question.clone()
    };

    // `a -n` 保存的用户笔记固定为 memo；notebook 检索不混入其他内部知识类别。
    let candidates = match crate::ai::tools::service::memory::search_memo_candidates_scored(
        &retrieval_query,
        20,
        true,
    ) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("[note-search] 检索失败: {}", err);
            return Err(err.into());
        }
    };
    if candidates.is_empty() {
        return Ok(format!("没有在知识库中找到与「{}」相关的内容。", question));
    }

    // 按语义分数收紧喂给 LLM 的条数，进一步防止大知识库撑爆上下文。
    // 仅在本次确实用了语义打分（embedding 可用）时收紧——此时分数可比较、
    // 排序可信；否则保持全部 20 条交给 LLM（与历史行为一致，不丢候选）。
    // 收紧策略：保留 top1 锚点；其余条目要求语义分数 >= top1 的 60% 才纳入，
    // 至多 8 条。这样只砍掉明显不相关的长尾，不影响真正相关的笔记。
    let selected = select_note_search_candidates(&candidates);

    // 把检索到的条目作为上下文，让模型基于这些内容回答用户的问题。
    let mut context = String::new();
    for (idx, candidate) in selected.iter().enumerate() {
        context.push_str(&format!("[{}] {}\n", idx + 1, candidate.entry.note));
    }

    let mut messages = vec![serde_json::json!({
        "role": "system",
        "content": "你处于 notebook 检索问答模式。下面会给出当前问题，以及本轮从用户 notebook（memo）里检索到的若干条笔记。\
                    每一轮都必须优先依据本轮检索结果回答。最近对话仅用于理解省略、代词和追问；如果最近对话与本轮检索结果冲突，以本轮检索结果为准。\
                    如果检索结果里没有足够信息回答，就直接说明。用中文回答，使用 Markdown 格式。",
    })];
    if note_search_interactive_mode(&app.cli) {
        messages.extend(build_note_search_chat_history(app, history_count)?);
    }
    messages.push(serde_json::json!({
        "role": "user",
        "content": format!("当前问题：{}\n\n本轮 notebook 检索结果：\n{}", question, context),
    }));

    match crate::ai::request::do_request_json(app, &app.current_model, &messages, false, true).await
    {
        Ok(response) => {
            let answer = crate::ai::request::extract_response_text(&response)
                .unwrap_or_default()
                .trim()
                .to_string();
            if answer.is_empty() {
                // 模型无输出时退回展示已选中的原始条目（复用上面的检索结果，不重复检索）。
                Ok(selected
                    .iter()
                    .enumerate()
                    .map(|(i, candidate)| format!("{}. {}", i + 1, candidate.entry.note))
                    .collect::<Vec<_>>()
                    .join("\n\n"))
            } else {
                Ok(answer)
            }
        }
        Err(err) => {
            eprintln!("[note-search] 总结失败: {}", err);
            Err(err.into())
        }
    }
}

fn persist_note_search_turn(app: &App, question: &str, answer: &str) {
    let question = question.trim();
    let answer = answer.trim();
    if question.is_empty() || answer.is_empty() {
        return;
    }

    let messages = vec![
        crate::ai::history::Message {
            role: "user".to_string(),
            content: serde_json::Value::String(question.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
        crate::ai::history::Message {
            role: "assistant".to_string(),
            content: serde_json::Value::String(answer.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        },
    ];
    if let Err(err) = crate::ai::history::append_history_messages_uncompacted(
        &app.session_history_file,
        &messages,
    ) {
        eprintln!("[Warning] Failed to save notebook search history: {}", err);
    }
}

pub(super) async fn handle_note_search_interactive_turn(
    app: &App,
    question: &str,
    history_count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    clear_stream_cancel(app);
    crate::ai::tools::registry::common::clear_tool_cancel();
    let _guard = ForegroundTurnGuard::enter();
    let answer = answer_memo_search(app, question, history_count).await?;
    crate::ai::stream::render_markdown_block(&answer).ok();
    persist_note_search_turn(app, question, &answer);
    Ok(())
}

/// 处理 --note-search / -ns：从知识库中检索 memo 类条目，再用模型根据检索到的
/// 内容总结、回答用户的问题（而不是直接堆砌原始条目）。
pub(super) async fn handle_memo_search(app: &App) -> Result<(), Box<dyn std::error::Error>> {
    let query = app.cli.args.join(" ");
    let answer = answer_memo_search(app, &query, 0).await?;
    crate::ai::stream::render_markdown_block(&answer).ok();
    Ok(())
}

// 处理 consolidate 计划里的 merge 项：只为有效 merge（ids 非空且 merged_content 非空）
// 生成新条目，并把对应源 IDs 自动并入删除集合，避免"旧条目 + 合并条目"并存。
fn build_consolidation_merge_entries(
    merge_plan: &[&serde_json::Value],
) -> (
    FxHashSet<String>,
    usize,
    Vec<crate::ai::tools::storage::memory_store::AgentMemoryEntry>,
) {
    let mut merge_delete_ids = FxHashSet::default();
    let mut merged_count = 0usize;
    let mut new_entries = Vec::new();

    for item in merge_plan {
        let ids: Vec<&str> = item["ids"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        let content = item["merged_content"].as_str().unwrap_or("").trim();
        if ids.is_empty() || content.is_empty() {
            continue;
        }

        merged_count += ids.len();
        merge_delete_ids.extend(ids.iter().map(|id| (*id).to_string()));
        new_entries.push(crate::ai::tools::storage::memory_store::AgentMemoryEntry {
            id: Some(crate::ai::tools::service::memory::next_memory_id()),
            timestamp: chrono::Local::now().to_rfc3339(),
            category: "user_memory".into(),
            note: content.to_string(),
            tags: vec!["consolidated".into()],
            source: None,
            priority: Some(150),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        });
    }

    (merge_delete_ids, merged_count, new_entries)
}

/// 处理 --consolidate-knowledge：读取全部知识条目 → 模型分析 → 执行整理。
///
/// **优化策略**（避免 60s 超时）：
/// 1. 只分析优先级 < 200 的条目（≥200 受保护）
/// 2. 按时间倒序取**最近 15 条**（之前 30 条还是太多）
/// 3. 每条内容截断到**40 字**（之前 80 字）
/// 4. 用 JSON 数组格式（比文本格式更省 token）
/// 5. 英文 system prompt（模型响应更快）
pub(super) async fn handle_consolidate_knowledge(
    app: &App,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
    use serde_json::Value;

    let store = MemoryStore::from_env_or_config();
    let all_entries = store.all().map_err(|e| format!("读取失败：{}", e))?;

    if all_entries.is_empty() {
        println!("📭 知识库为空，无需整理。");
        return Ok(());
    }

    // 过滤：优先级 < 200 的才分析；排除 memo（用户记事本，不应被自动删除/合并）；按时间倒序；取最近 15 条
    let mut candidates: Vec<&AgentMemoryEntry> = all_entries
        .iter()
        .filter(|e| e.priority.unwrap_or(100) < 200 && e.category != "memo")
        .collect();
    candidates.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    candidates.truncate(15);

    if candidates.is_empty() {
        println!("📭 没有可整理的条目（全部优先级 ≥ 200，已保护）。");
        return Ok(());
    }

    // 构建紧凑的 JSON 数组（比文本格式更省 token）
    let mut entries_json = Vec::new();
    for entry in &candidates {
        let id = entry.id.as_deref().unwrap_or("unknown");
        let prio = entry.priority.unwrap_or(100);
        let ts_short: String = entry.timestamp.chars().take(10).collect();
        let preview: String = if entry.note.chars().count() > 40 {
            entry.note.chars().take(40).collect::<String>() + "…"
        } else {
            entry.note.clone()
        };
        entries_json.push(serde_json::json!({
            "id": id,
            "cat": entry.category,
            "pri": prio,
            "tags": entry.tags,
            "date": ts_short,
            "src": entry.source.as_deref().unwrap_or(""),
            "text": preview,
        }));
    }

    let sys = "You are a knowledge curator. Analyze entries and suggest deletions/merges.\n\
        Return ONLY valid JSON:\n\
        {\"reasoning\":\"1-sentence summary\",\"delete_ids\":[\"id1\",\"id2\"],\"merge_plan\":[{\"ids\":[\"id1\",\"id2\"],\"merged_content\":\"...\"}]}\n\
        Rules: delete duplicates/obsolete; merge related; keep useful. Priority>=200 already filtered out.";

    let prompt = format!(
        "Analyze these {} entries:\n{}",
        candidates.len(),
        serde_json::to_string(&entries_json).unwrap()
    );
    let messages = vec![
        serde_json::json!({"role": "system", "content": sys}),
        serde_json::json!({"role": "user", "content": prompt}),
    ];

    // 知识整理用主模型（用户的默认对话模型）。走流式链路：响应头立即返回、
    // 数据按 chunk 增量到达，避免非流式"等整段 body 生成完"被 60s 超时撑爆。
    let model = crate::ai::models::initial_model(&app.cli);
    let spinner = SearchSpinner::start("整理知识库");
    let raw = match crate::ai::request::do_request_text_streaming(app, &model, &messages).await {
        Ok(text) => {
            spinner.stop();
            text
        }
        Err(err) => {
            spinner.stop();
            eprintln!("[consolidate] Request failed: {}", err);
            return Err(err);
        }
    };

    let raw = raw.trim();
    let cleaned = raw
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    if cleaned.is_empty() || raw.is_empty() {
        println!("⚠  Empty response. No changes.");
        return Ok(());
    }

    let plan: Value = match serde_json::from_str(cleaned) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[consolidate] JSON parse error: {}", e);
            eprintln!(
                "[consolidate] Raw: {}",
                raw.chars().take(200).collect::<String>()
            );
            return Ok(());
        }
    };

    if let Some(reasoning) = plan["reasoning"].as_str() {
        println!("\n🔍 {}\n", reasoning);
    }

    let delete_ids: Vec<&str> = plan["delete_ids"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let merge_plan: Vec<&Value> = plan["merge_plan"]
        .as_array()
        .map(|a| a.iter().collect())
        .unwrap_or_default();
    let (merge_delete_ids, merged_count, new_entries) =
        build_consolidation_merge_entries(&merge_plan);

    let mut delete_id_set: FxHashSet<String> =
        delete_ids.iter().map(|id| (*id).to_string()).collect();
    delete_id_set.extend(merge_delete_ids);

    if delete_id_set.is_empty() && new_entries.is_empty() {
        println!("✅ Already well-organized. Nothing to change.");
        return Ok(());
    }

    let delete_refs: Vec<&str> = delete_id_set.iter().map(String::as_str).collect();
    match store.apply_batch_update(&delete_refs, &new_entries) {
        Ok(report) => {
            if !delete_refs.is_empty() {
                println!("🗑  Deleted {} entries", report.deleted);
            }
            if !new_entries.is_empty() {
                println!(
                    "💾 Merged {} entries into {} new",
                    merged_count, report.appended
                );
            }
        }
        Err(e) => {
            eprintln!("  Consolidation error: {}", e);
        }
    }

    println!("\n✨ Done.");
    Ok(())
}

/// 处理 --note-delete / -nd <一段话>：用模型在知识库中匹配最相关的 memo 条目，
/// 找到对应 id，删除前请用户确认。
pub(super) async fn handle_note_delete(
    app: &mut App,
    query: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    // 拼接查询：flag 的值 + 其余位置参数；都为空时进入多行输入框。
    let mut query = query.trim().to_string();
    if !app.cli.args.is_empty() {
        let extra = app.cli.args.join(" ");
        if !query.is_empty() {
            query.push(' ');
        }
        query.push_str(extra.trim());
    }
    let query = query.trim().to_string();
    let query = if query.is_empty() {
        println!("[note-delete] 请描述你想删除的内容（多行；提交后开始匹配，留空取消）：");
        let input = match app.prompt_editor.as_mut() {
            Some(editor) => editor.read_multi_line().ok().flatten(),
            None => None,
        };
        match input {
            Some(s) if !s.trim().is_empty() => s.trim().to_string(),
            _ => {
                eprintln!("[note-delete] 未输入任何内容，已取消");
                return Ok(());
            }
        }
    } else {
        query
    };

    // 检索候选条目。
    let candidates = match crate::ai::tools::service::memory::search_memo_candidates(
        &query,
        10,
        false,
    ) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("[note-delete] 检索失败: {}", err);
            return Err(err.into());
        }
    };
    if candidates.is_empty() {
        println!(
            "[note-delete] 没有找到与「{}」相关的可删除 memo 条目。",
            query
        );
        return Ok(());
    }

    // 让模型从候选中挑选最匹配的一条（返回其序号，或 NONE）。
    let mut listing = String::new();
    for (idx, e) in candidates.iter().enumerate() {
        let note_preview: String = e.note.chars().take(300).collect();
        listing.push_str(&format!("{}. {}\n", idx + 1, note_preview));
    }

    let model = crate::ai::models::initial_model(&app.cli);
    let messages = vec![
        serde_json::json!({
            "role": "system",
            "content": "你是一个知识库删除助手。用户会给出一段描述，以及若干条带编号的候选笔记。\
                        请判断哪些条目符合用户想删除的内容——可能是一条，也可能是多条。\
                        只输出这些条目的编号，用英文逗号分隔（如 1 或 1,3,4）。\
                        如果没有任何一条明显匹配，只输出 NONE。不要输出任何解释或多余字符。",
        }),
        serde_json::json!({
            "role": "user",
            "content": format!("用户描述：{}\n\n候选条目：\n{}", query, listing),
        }),
    ];

    let chosen =
        match crate::ai::request::do_request_json(app, &model, &messages, false, false).await {
            Ok(response) => crate::ai::request::extract_response_text(&response)
                .map(|s| s.trim().to_string())
                .unwrap_or_default(),
            Err(err) => {
                eprintln!("[note-delete] 模型匹配失败: {}", err);
                String::new()
            }
        };

    // 解析模型返回的若干编号（支持逗号 / 空格 / 顿号等分隔），去重并保持升序。
    let mut chosen_indices: Vec<usize> = Vec::new();
    {
        let mut num = String::new();
        let flush = |num: &mut String, out: &mut Vec<usize>| {
            if let Ok(n) = num.parse::<usize>() {
                if n >= 1 && n <= candidates.len() {
                    let idx = n - 1;
                    if !out.contains(&idx) {
                        out.push(idx);
                    }
                }
            }
            num.clear();
        };
        for c in chosen.chars() {
            if c.is_ascii_digit() {
                num.push(c);
            } else {
                flush(&mut num, &mut chosen_indices);
            }
        }
        flush(&mut num, &mut chosen_indices);
    }
    chosen_indices.sort_unstable();

    if chosen_indices.is_empty() {
        println!(
            "[note-delete] 模型未能从候选中确定要删除的条目，已取消。可换个更具体的描述重试。"
        );
        return Ok(());
    }

    let targets: Vec<&crate::ai::tools::storage::memory_store::AgentMemoryEntry> =
        chosen_indices.iter().map(|&i| &candidates[i]).collect();

    // 删除前确认 + 精选。列出条目后，用户可以：
    //   - 直接回车 / y / all / a：删除全部列出条目
    //   - 输入编号（如 1,3）：只删除指定编号
    //   - n / 回车以外的取消词：取消
    println!("\n[note-delete] 匹配到以下 {} 条条目：", targets.len());
    for (n, target) in targets.iter().enumerate() {
        println!("  [{}]", n + 1);
        if let Some(id) = target.id.as_deref().filter(|s| !s.is_empty()) {
            println!("    id: {}", id);
        }
        println!("    时间: {}", target.timestamp);
        println!(
            "    内容: {}",
            target.note.chars().take(500).collect::<String>()
        );
    }
    print!("\n请输入要删除的编号（如 1,3；输入 all 删除全部，直接回车=全部，n=取消）: ");
    std::io::stdout().flush().ok();

    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer).ok();
    let answer = answer.trim().to_lowercase();

    // 解析用户选择，得到最终要删除的 targets 子集。
    let selected: Vec<&crate::ai::tools::storage::memory_store::AgentMemoryEntry> = if answer
        .is_empty()
        || answer == "y"
        || answer == "yes"
        || answer == "all"
        || answer == "a"
    {
        targets.clone()
    } else if answer == "n" || answer == "no" || answer == "q" || answer == "cancel" {
        println!("[note-delete] 已取消，未删除任何内容。");
        return Ok(());
    } else {
        // 解析编号列表（针对上面列出的 1..=targets.len()）。
        let mut picks: Vec<usize> = Vec::new();
        let mut num = String::new();
        let flush = |num: &mut String, out: &mut Vec<usize>| {
            if let Ok(n) = num.parse::<usize>() {
                if n >= 1 && n <= targets.len() {
                    let idx = n - 1;
                    if !out.contains(&idx) {
                        out.push(idx);
                    }
                }
            }
            num.clear();
        };
        for c in answer.chars() {
            if c.is_ascii_digit() {
                num.push(c);
            } else {
                flush(&mut num, &mut picks);
            }
        }
        flush(&mut num, &mut picks);
        picks.sort_unstable();
        if picks.is_empty() {
            println!("[note-delete] 未识别到有效编号，已取消，未删除任何内容。");
            return Ok(());
        }
        picks.into_iter().map(|i| targets[i]).collect()
    };

    let mut deleted = 0usize;
    let mut failed = 0usize;
    for target in &selected {
        match crate::ai::tools::service::memory::delete_memo_entry(target) {
            Ok(_) => deleted += 1,
            Err(err) => {
                failed += 1;
                eprintln!(
                    "[note-delete] 删除失败 (时间 {}): {}",
                    target.timestamp, err
                );
            }
        }
    }
    println!(
        "[note-delete] 完成：已删除 {} 条，失败 {} 条。",
        deleted, failed
    );
    if failed > 0 && deleted == 0 {
        return Err("all deletions failed".into());
    }
    Ok(())
}

/// 处理 --note-edit / -ne <一段话>：用模型在知识库中匹配相关 memo 条目，
/// 匹配到多条时让用户选定一条，在编辑器中预填原文改写后保存（保留 id、更新时间戳）。
pub(super) async fn handle_note_edit(
    app: &mut App,
    query: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    // 状态行着色：与黑底白字的 note 正文区分开。
    const NE: &str = "\x1b[1;36m[note-edit]\x1b[0m"; // 青色加粗标签
    const FIELD: &str = "\x1b[2m"; // 字段名（id/时间/内容）暗灰
    const HINT: &str = "\x1b[1;32m"; // 操作提示绿色加粗
    const IDX: &str = "\x1b[1;33m"; // 候选编号黄色加粗
    const RST: &str = "\x1b[0m";

    // 拼接查询：flag 的值 + 其余位置参数；都为空时进入多行输入框。
    let mut query = query.trim().to_string();
    if !app.cli.args.is_empty() {
        let extra = app.cli.args.join(" ");
        if !query.is_empty() {
            query.push(' ');
        }
        query.push_str(extra.trim());
    }
    let query = query.trim().to_string();
    let query = if query.is_empty() {
        println!("{NE} 请描述你想修改的内容（多行；提交后开始匹配，留空取消）：");
        let input = match app.prompt_editor.as_mut() {
            Some(editor) => editor.read_multi_line().ok().flatten(),
            None => None,
        };
        match input {
            Some(s) if !s.trim().is_empty() => s.trim().to_string(),
            _ => {
                eprintln!("{NE} 未输入任何内容，已取消");
                return Ok(());
            }
        }
    } else {
        query
    };

    // 检索 + 模型匹配都可能耗时，给一个状态条动画（输出前自动清除），与 -ns 一致。
    let spinner = SearchSpinner::start("匹配知识库条目");

    // 检索候选条目。
    let candidates = match crate::ai::tools::service::memory::search_memo_candidates(
        &query,
        10,
        false,
    ) {
        Ok(c) => c,
        Err(err) => {
            spinner.stop();
            eprintln!("{NE} 检索失败: {}", err);
            return Err(err.into());
        }
    };
    if candidates.is_empty() {
        spinner.stop();
        println!("{NE} 没有找到与「{}」相关的可修改 memo 条目。", query);
        return Ok(());
    }

    // 让模型从候选中挑选匹配的条目（可能多条），返回编号。
    let mut listing = String::new();
    for (idx, e) in candidates.iter().enumerate() {
        let note_preview: String = e.note.chars().take(300).collect();
        listing.push_str(&format!("{}. {}\n", idx + 1, note_preview));
    }

    let model = crate::ai::models::initial_model(&app.cli);
    let messages = vec![
        serde_json::json!({
            "role": "system",
            "content": "你是一个知识库编辑助手。用户会给出一段描述，以及若干条带编号的候选笔记。\
                        请判断哪些条目符合用户想修改的内容——可能是一条，也可能是多条。\
                        只输出这些条目的编号，用英文逗号分隔（如 1 或 1,3,4）。\
                        如果没有任何一条明显匹配，只输出 NONE。不要输出任何解释或多余字符。",
        }),
        serde_json::json!({
            "role": "user",
            "content": format!("用户描述：{}\n\n候选条目：\n{}", query, listing),
        }),
    ];

    let mut matched_err: Option<String> = None;
    let chosen =
        match crate::ai::request::do_request_json(app, &model, &messages, false, false).await {
            Ok(response) => crate::ai::request::extract_response_text(&response)
                .map(|s| s.trim().to_string())
                .unwrap_or_default(),
            Err(err) => {
                matched_err = Some(format!("{}", err));
                String::new()
            }
        };
    spinner.stop();
    if let Some(err) = matched_err {
        eprintln!("{NE} 模型匹配失败: {}", err);
    }

    // 解析模型返回的编号集合。
    let parse_indices = |s: &str, max: usize| -> Vec<usize> {
        let mut out: Vec<usize> = Vec::new();
        let mut num = String::new();
        let flush = |num: &mut String, out: &mut Vec<usize>| {
            if let Ok(n) = num.parse::<usize>() {
                if n >= 1 && n <= max {
                    let idx = n - 1;
                    if !out.contains(&idx) {
                        out.push(idx);
                    }
                }
            }
            num.clear();
        };
        for c in s.chars() {
            if c.is_ascii_digit() {
                num.push(c);
            } else {
                flush(&mut num, &mut out);
            }
        }
        flush(&mut num, &mut out);
        out.sort_unstable();
        out
    };

    let mut matched = parse_indices(&chosen, candidates.len());
    if matched.is_empty() {
        println!("{NE} 模型未能从候选中确定要修改的条目，已取消。可换个更具体的描述重试。");
        return Ok(());
    }

    // 匹配到多条：列出后让用户选定恰好一条来编辑（编辑是针对单条内容的）。
    let target_idx = if matched.len() == 1 {
        matched[0]
    } else {
        println!("\n{NE} 匹配到以下 {IDX}{}{RST} 条条目：", matched.len());
        for (n, &ci) in matched.iter().enumerate() {
            let e = &candidates[ci];
            println!("  {IDX}[{}]{RST}", n + 1);
            if let Some(id) = e.id.as_deref().filter(|s| !s.is_empty()) {
                println!("    {FIELD}id:{RST} {}", id);
            }
            println!("    {FIELD}时间:{RST} {}", e.timestamp);
            println!(
                "    {FIELD}内容:{RST} {}",
                e.note.chars().take(500).collect::<String>()
            );
        }
        print!("\n{HINT}请输入要修改的编号（只能选一条；n=取消）:{RST} ");
        std::io::stdout().flush().ok();
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer).ok();
        let answer = answer.trim().to_lowercase();
        if answer == "n" || answer == "no" || answer == "q" || answer == "cancel" {
            println!("{NE} 已取消，未修改任何内容。");
            return Ok(());
        }
        let picks = parse_indices(&answer, matched.len());
        match picks.first() {
            Some(&p) => matched.remove(p),
            None => {
                println!("{NE} 未识别到有效编号，已取消，未修改任何内容。");
                return Ok(());
            }
        }
    };

    let target = candidates[target_idx].clone();

    // 在编辑器中预填原文，让用户改写。
    println!("\n{NE} 将打开编辑器修改以下条目（原文已预填；留空或不改动即取消）：");
    if let Some(id) = target.id.as_deref().filter(|s| !s.is_empty()) {
        println!("    {FIELD}id:{RST} {}", id);
    }
    println!("    {FIELD}时间:{RST} {}", target.timestamp);

    let new_note = match app.prompt_editor.as_mut() {
        Some(editor) => {
            editor.set_prefill(target.note.clone());
            editor.read_multi_line().ok().flatten()
        }
        None => None,
    };
    let new_note = match new_note {
        Some(s) if !s.trim().is_empty() => s.trim().to_string(),
        _ => {
            println!("{NE} 未输入新内容，已取消。");
            return Ok(());
        }
    };
    if new_note == target.note.trim() {
        println!("{NE} 内容未变化，已取消。");
        return Ok(());
    }

    // 在保存前用 LLM 整理用户改写后的内容：只做格式/表达上的润色，
    // 严格禁止改变语义。整理失败则回退到用户编辑的原文，不阻塞保存。
    let final_note = {
        let spinner = SearchSpinner::start("整理修改内容");
        let mut tidy_err: Option<String> = None;
        let tidy_messages = vec![
            serde_json::json!({
                "role": "system",
                "content": "你是一个知识库整理助手。用户会给出一段刚刚在编辑器里改写完的笔记内容。\
                            请帮用户整理这段内容，使其更清晰、更易读。\
                            \n严格约束：\n\
                            1. 绝对不要改变内容的语义、事实或意图，只能调整格式、排版、标点和表达方式；\n\
                            2. 不要增删任何实质性信息；\n\
                            3. 保留原文的语言（中文保持中文，英文保持英文）；\n\
                            4. 只输出整理后的正文，不要输出任何解释、前后缀或 markdown 代码块标记。",
            }),
            serde_json::json!({
                "role": "user",
                "content": new_note.clone(),
            }),
        ];
        let result =
            match crate::ai::request::do_request_json(app, &model, &tidy_messages, false, false)
                .await
            {
                Ok(response) => crate::ai::request::extract_response_text(&response)
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty()),
                Err(err) => {
                    tidy_err = Some(format!("{}", err));
                    None
                }
            };
        spinner.stop();
        if let Some(err) = tidy_err {
            eprintln!("{NE} 模型整理失败（将保存原文）: {}", err);
        }
        match result {
            Some(tidied) if tidied != new_note => {
                println!("{NE} 已整理修改内容（语义未变）：");
                println!(
                    "  {FIELD}整理后:{RST} {}",
                    tidied.chars().take(500).collect::<String>()
                );
                tidied
            }
            _ => new_note,
        }
    };

    match crate::ai::tools::service::memory::update_memo_entry(&target, &final_note) {
        Ok(_) => {
            println!("{NE} 已更新该条目。");
            Ok(())
        }
        Err(err) => {
            eprintln!("{NE} 更新失败: {}", err);
            Err(err.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::history::Message;

    #[test]
    fn note_search_followup_query_includes_recent_history() {
        let history = vec![
            Message {
                role: "assistant".to_string(),
                content: serde_json::Value::String(
                    "第一条讲的是 trait object 和 dyn 的区别。".to_string(),
                ),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            Message {
                role: "user".to_string(),
                content: serde_json::Value::String("帮我找 trait object 的笔记".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
        ];

        let query = build_note_search_retrieval_query("再展开第一条", &history);
        assert!(query.contains("当前问题：再展开第一条"));
        assert!(query.contains("用户: 帮我找 trait object 的笔记"));
        assert!(query.contains("助手: 第一条讲的是 trait object 和 dyn 的区别。"));

        let user_pos = query
            .find("用户: 帮我找 trait object 的笔记")
            .expect("user context should be present");
        let assistant_pos = query
            .find("助手: 第一条讲的是 trait object 和 dyn 的区别。")
            .expect("assistant context should be present");
        assert!(user_pos < assistant_pos);
    }

    #[test]
    fn consolidation_merge_plan_auto_deletes_valid_merge_ids() {
        let merge_plan = [
            serde_json::json!({
                "ids": ["id_a", "id_b"],
                "merged_content": "合并后的内容"
            }),
            serde_json::json!({
                "ids": ["ignored"],
                "merged_content": ""
            }),
        ];
        let merge_plan_refs: Vec<&serde_json::Value> = merge_plan.iter().collect();

        let (delete_ids, merged_count, new_entries) =
            build_consolidation_merge_entries(&merge_plan_refs);

        assert_eq!(merged_count, 2);
        assert_eq!(delete_ids.len(), 2);
        assert!(delete_ids.contains("id_a"));
        assert!(delete_ids.contains("id_b"));
        assert!(!delete_ids.contains("ignored"));
        assert_eq!(new_entries.len(), 1);
        assert!(
            new_entries[0]
                .id
                .as_deref()
                .is_some_and(|id| id.starts_with("mem_"))
        );
        assert_eq!(new_entries[0].note, "合并后的内容");
        assert_eq!(new_entries[0].tags, vec!["consolidated".to_string()]);
    }
}
