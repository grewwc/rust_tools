// =============================================================================
// Note/Memo Search + Knowledge Consolidation CLI Subsystem
// =============================================================================
// Note/memo search + knowledge consolidation CLI subfeature extracted from driver/mod.rs.
// =============================================================================

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ai::cli::ParsedCli;
use crate::ai::tools::storage::memory_store::{AgentMemoryEntry, MemoryStore};
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
    if !cli.note_search {
        return false;
    }
    if cli.interactive {
        return true;
    }
    // `a -ns` with no substantive query content (whitespace does not count)
    // automatically enters interactive mode, equivalent to `a -ns -i`; only a query
    // with content stays a one-shot single-round retrieval.
    !cli.args.iter().any(|arg| !arg.trim().is_empty())
}

/// If the clipboard holds an image, a vision model interprets the content;
/// otherwise use the text following `-n`; if there is no text either, open the
/// multi-line input box for the user to type it.
pub(super) async fn handle_note_save(app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
    use arboard::Clipboard;
    use image::buffer::ConvertBuffer;
    use image::{ImageBuffer, Rgb, Rgba};
    use std::fs;

    let store = MemoryStore::from_env_or_config();
    // -n is a string flag and only captures the first token after it (e.g. in
    // `a -n aeolus prod log path: ...` only "aeolus" becomes the note value), and the
    // remaining tokens fall into positional args. The positional args are joined back
    // here so the content is not truncated and later retrieval can find the full note.
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

    // Image persistence dir: note_images/ next to the memory file. The previous
    // implementation wrote screenshots to /tmp, deleted them immediately, and stored
    // image_path: None, losing the image for good so the memo could never reference
    // the original again. Switched to persistent storage.
    let images_dir = store
        .path()
        .parent()
        .map(|parent| parent.join("note_images"))
        .unwrap_or_else(|| PathBuf::from("note_images"));

    // Try to fetch an image from the clipboard and persist it
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
        // Image present: ask the vision model to interpret the content
        println!("[note] Detected image in clipboard, analyzing...");

        let model = crate::ai::models::determine_vl_model(&app.current_model);

        // Build the message that includes the image
        let content = crate::ai::request::build_content(
            &model,
            "请详细描述这张图片的内容，包括关键信息、文字、数据等。用中文回答。",
            &[image_path.clone()],
        )?;

        let messages = vec![serde_json::json!({
            "role": "user",
            "content": content,
        })];

        // Call the model
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
        // No image: get the raw text (from the text after -n or the multi-line input
        // box), and always have the model interpret and organize it before saving
        // instead of dumping the raw text directly.
        let raw = if let Some(text) = provided_text.filter(|t| !t.trim().is_empty()) {
            text
        } else {
            // Neither image nor text: open the multi-line input box so the user can
            // enter the content to save.
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

        // Ask the model to interpret and organize the user input so it works better
        // as a knowledge-base memo.
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
                // On organization failure fall back to saving the raw input so user
                // content is not lost.
                eprintln!("[note] 整理失败，保存原始输入: {}", err);
                raw
            }
        }
    };

    // Save to the knowledge base (the image is already persisted; its path is
    // written to image_path for later reference)
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

/// A lightweight terminal "Searching..." spinner.
///
/// Refreshes the spinner in place on stderr using carriage returns `\r`; on
/// `stop()` / drop it clears the current line so the following formal output
/// (formal results go to stdout) is not polluted. Enabled only when stderr is a
/// TTY; silently disabled for pipes/redirects to avoid writing garbage characters.
struct SearchSpinner {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl SearchSpinner {
    fn start(label: &str) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        // With a non-TTY (piped/redirected) stderr, draw no animation and return an
        // inert spinner.
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
            // Clear the current line (wide enough to cover "<frame> <label>...").
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
        // Consume explicitly to trigger Drop.
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
        crate::ai::driver::runtime_ctx::effective_cwd().ok().as_deref(),
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
    // Lexical-only scoring: no comparable semantic threshold anymore, so keep
    // every candidate for the LLM (matches the historical no-embedding path).
    candidates.iter().collect()
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

    // Retrieval plus model summarization can both take a while; show a
    // "Searching..." spinner (cleared automatically before output).
    let _spinner = SearchSpinner::start("Searching memo");
    let retrieval_query = if note_search_interactive_mode(&app.cli) {
        build_note_search_retrieval_query(&question, &read_recent_history(app))
    } else {
        question.clone()
    };

    // Notes saved by `a -n` are always memos; notebook retrieval never mixes in
    // other internal knowledge categories.
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

    // The semantic tightening was removed along with the embedding pipeline:
    // lexical scores have no comparable threshold, so every candidate is kept for
    // the LLM (consistent with the historical no-embedding path; no candidate lost).
    let selected = select_note_search_candidates(&candidates);

    // Feed the retrieved entries as context so the model answers the user's
    // question from them.
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
                // When the model produced no output, fall back to listing the selected
                // raw entries (reusing the retrieval above; no second search).
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
    if let Err(err) =
        crate::ai::history::append_history_messages(&app.session_history_file, &messages)
    {
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

/// Handle --note-search / -ns: retrieve memo entries from the knowledge base,
/// then have the model summarize them and answer the user's question (rather
/// than dumping the raw entries directly).
pub(super) async fn handle_memo_search(app: &App) -> Result<(), Box<dyn std::error::Error>> {
    let query = app.cli.args.join(" ");
    let answer = answer_memo_search(app, &query, 0).await?;
    crate::ai::stream::render_markdown_block(&answer).ok();
    Ok(())
}

#[derive(Clone, Copy)]
enum ConsolidationScope {
    Memo,
    OtherKnowledge,
}

impl ConsolidationScope {
    fn label(self) -> &'static str {
        match self {
            Self::Memo => "memo 笔记",
            Self::OtherKnowledge => "非 memo 知识",
        }
    }

    fn matches(self, entry: &AgentMemoryEntry) -> bool {
        match self {
            // Memos with images keep their original entries, so merging cannot
            // lose the attachment association.
            Self::Memo => entry.category == "memo" && entry.image_path.is_none(),
            Self::OtherKnowledge => entry.category != "memo",
        }
    }

    fn merged_category(self) -> &'static str {
        match self {
            Self::Memo => "memo",
            // Keep the original consolidate merge target for non-memo entries.
            Self::OtherKnowledge => "user_memory",
        }
    }

    fn merged_source(self) -> Option<&'static str> {
        match self {
            Self::Memo => Some("cli_note"),
            Self::OtherKnowledge => None,
        }
    }

    fn curator_rule(self) -> &'static str {
        match self {
            Self::Memo => {
                "Retain every concrete fact, troubleshooting step, command, identifier, and database detail when merging memos."
            }
            Self::OtherKnowledge => "Keep useful knowledge and its intent intact when merging.",
        }
    }
}

fn consolidation_candidates(
    all_entries: &[AgentMemoryEntry],
    scope: ConsolidationScope,
) -> Vec<&AgentMemoryEntry> {
    let mut candidates: Vec<&AgentMemoryEntry> = all_entries
        .iter()
        .filter(|entry| entry.priority.unwrap_or(100) < 200)
        // Legacy entries without an ID cannot be precisely replaced by
        // apply_batch_update; the model gets no plan over them.
        .filter(|entry| entry.id.as_deref().is_some_and(|id| !id.trim().is_empty()))
        .filter(|entry| scope.matches(entry))
        .collect();
    candidates.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    candidates.truncate(15);
    candidates
}

// Process merge items in a consolidate plan: generate new entries only for valid
// merges (non-empty ids and non-empty merged_content), and fold the source IDs
// into the delete set automatically, avoiding "old entry + merged entry"
// coexistence.
//
// originals provides an id -> full-entry lookup for **lossless merging**: the
// model only sees truncated previews, and a merged_content rewritten from memory
// loses detail; user memos are non-recoverable note data, so a new entry must
// equal model summary + the full text of every source entry. Consolidation never
// compresses or drops content.
fn build_consolidation_merge_entries(
    category: &str,
    source: Option<&str>,
    merge_plan: &[&serde_json::Value],
    eligible_ids: &FxHashSet<&str>,
    originals: &FxHashMap<&str, &AgentMemoryEntry>,
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
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default()
            .into_iter()
            .filter(|id| eligible_ids.contains(id) && !merge_delete_ids.contains(*id))
            .collect();
        let content = item["merged_content"].as_str().unwrap_or("").trim();
        // A single rewritten entry is not a merge; at least two actionable entries
        // are required to be replaced by one merged entry.
        if ids.len() < 2 || content.is_empty() {
            continue;
        }

        // Lossless merge: model summary first, then the full text of every source
        // entry, so no detail is lost.
        let mut note = content.to_string();
        let original_texts: Vec<&str> = ids
            .iter()
            .filter_map(|id| originals.get(*id))
            .map(|entry| entry.note.trim())
            .filter(|text| !text.is_empty())
            .collect();
        if !original_texts.is_empty() {
            note.push_str("\n\n--- 原文保留（合并来源）---\n");
            note.push_str(&original_texts.join("\n\n"));
        }

        merged_count += ids.len();
        merge_delete_ids.extend(ids.iter().map(|id| (*id).to_string()));
        new_entries.push(crate::ai::tools::storage::memory_store::AgentMemoryEntry {
            id: Some(crate::ai::tools::service::memory::next_memory_id()),
            timestamp: chrono::Local::now().to_rfc3339(),
            category: category.into(),
            note,
            tags: vec!["consolidated".into()],
            source: source.map(str::to_string),
            priority: Some(150),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        });
    }

    (merge_delete_ids, merged_count, new_entries)
}

async fn consolidate_scope(
    app: &App,
    store: &MemoryStore,
    scope: ConsolidationScope,
    candidates: &[&AgentMemoryEntry],
) -> Result<(), Box<dyn std::error::Error>> {
    use serde_json::Value;

    let mut entries_json = Vec::new();
    for entry in candidates {
        let id = entry.id.as_deref().unwrap_or("unknown");
        let prio = entry.priority.unwrap_or(100);
        let ts_short: String = entry.timestamp.chars().take(10).collect();
        // The preview is what the model uses for merge/delete decisions. 500 chars
        // covers the vast majority of memos; overlong entries are explicitly marked
        // as truncated. Merges still store the full original text (see
        // build_consolidation_merge_entries), so preview truncation loses nothing.
        let note_len = entry.note.chars().count();
        let preview: String = if note_len > 500 {
            entry.note.chars().take(500).collect::<String>()
                + &format!("…[truncated, total {} chars]", note_len)
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

    // id -> full entry: merges concatenate the source memo's full text into the
    // new entry so consolidation loses no content.
    let originals: FxHashMap<&str, &AgentMemoryEntry> = candidates
        .iter()
        .filter_map(|entry| entry.id.as_deref().map(|id| (id, *entry)))
        .collect();

    let sys = format!(
        "You are a knowledge curator. Analyze only the current scope: {}. \
         Every listed entry belongs to this scope; never combine it with another scope.\n\
         Return ONLY valid JSON:\n\
         {{\"reasoning\":\"1-sentence summary\",\"delete_ids\":[\"id1\",\"id2\"],\"merge_plan\":[{{\"ids\":[\"id1\",\"id2\"],\"merged_content\":\"...\"}}]}}\n\
         Rules: use only listed IDs; delete only exact duplicates or obsolete entries; merge only related entries; keep useful entries. {} Priority>=200 entries are already excluded.",
        scope.label(),
        scope.curator_rule(),
    );
    let prompt = format!(
        "Analyze these {} {} entries:\n{}",
        candidates.len(),
        scope.label(),
        serde_json::to_string(&entries_json).unwrap()
    );
    let messages = vec![
        serde_json::json!({"role": "system", "content": sys}),
        serde_json::json!({"role": "user", "content": prompt}),
    ];

    let model = crate::ai::models::initial_model(&app.cli);
    let spinner = SearchSpinner::start(&format!("整理{}", scope.label()));
    let raw = match crate::ai::request::do_request_text_streaming(app, &model, &messages).await {
        Ok(text) => {
            spinner.stop();
            text
        }
        Err(err) => {
            spinner.stop();
            eprintln!("[consolidate:{}] Request failed: {}", scope.label(), err);
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
        println!("⚠ [{}] Empty response. No changes.", scope.label());
        return Ok(());
    }

    let plan: Value = match serde_json::from_str(cleaned) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("[consolidate:{}] JSON parse error: {}", scope.label(), err);
            eprintln!(
                "[consolidate:{}] Raw: {}",
                scope.label(),
                raw.chars().take(200).collect::<String>()
            );
            return Ok(());
        }
    };
    if let Some(reasoning) = plan["reasoning"].as_str() {
        println!("\n🔍 [{}] {}\n", scope.label(), reasoning);
    }

    let eligible_ids: FxHashSet<&str> = candidates
        .iter()
        .filter_map(|entry| entry.id.as_deref())
        .collect();
    let requested_delete_ids: Vec<&str> = plan["delete_ids"]
        .as_array()
        .map(|entries| entries.iter().filter_map(|value| value.as_str()).collect())
        .unwrap_or_default();
    let merge_plan: Vec<&Value> = plan["merge_plan"]
        .as_array()
        .map(|entries| entries.iter().collect())
        .unwrap_or_default();
    let (merge_delete_ids, merged_count, new_entries) = build_consolidation_merge_entries(
        scope.merged_category(),
        scope.merged_source(),
        &merge_plan,
        &eligible_ids,
        &originals,
    );

    // Model output can only affect the current batch's candidate IDs, so memos
    // and other categories can never delete or merge each other.
    let mut delete_id_set: FxHashSet<String> = requested_delete_ids
        .into_iter()
        .filter(|id| eligible_ids.contains(id))
        .map(str::to_string)
        .collect();
    delete_id_set.extend(merge_delete_ids);

    if delete_id_set.is_empty() && new_entries.is_empty() {
        println!(
            "✅ [{}] Already well-organized. Nothing to change.",
            scope.label()
        );
        return Ok(());
    }

    let delete_refs: Vec<&str> = delete_id_set.iter().map(String::as_str).collect();
    match store.apply_batch_update(&delete_refs, &new_entries) {
        Ok(report) => {
            if !delete_refs.is_empty() {
                println!("🗑  [{}] Deleted {} entries", scope.label(), report.deleted);
            }
            if !new_entries.is_empty() {
                println!(
                    "💾 [{}] Merged {} entries into {} new",
                    scope.label(),
                    merged_count,
                    report.appended
                );
            }
        }
        Err(err) => eprintln!("[consolidate:{}] Error: {}", scope.label(), err),
    }

    Ok(())
}

/// Handle --consolidate-knowledge: read all knowledge entries → model analysis →
/// run the consolidation.
///
/// **Optimization strategy** (avoid timeouts / control tokens):
/// 1. Only entries with priority < 200 are analyzed (>= 200 is protected)
/// 2. memos and other categories each take the **most recent 15** by timestamp
///    descending, never mixed in one pass
/// 3. Each preview is truncated to **500 chars**; merged entries still get the
///    full source memo text on write (lossless)
/// 4. JSON format is used (cheaper in tokens than free text)
/// 5. English system prompt (models respond faster)
///
/// Read scope: main file plus all rotated archives (all_with_archives).
/// Historical memos/knowledge moved into archives by rotation also enter
/// consolidation; the corresponding deletes land in the archive file itself via
/// apply_batch_update.
pub(super) async fn handle_consolidate_knowledge(
    app: &App,
) -> Result<(), Box<dyn std::error::Error>> {
    let store = MemoryStore::from_env_or_config();
    let all_entries = store
        .all_with_archives()
        .map_err(|e| format!("读取失败：{}", e))?;

    if all_entries.is_empty() {
        println!("📭 知识库为空，无需整理。");
        return Ok(());
    }

    let mut scopes_with_candidates = 0usize;
    // Memos run first; even if the later non-memo model request fails, the user's
    // notes were already processed independently.
    for scope in [ConsolidationScope::Memo, ConsolidationScope::OtherKnowledge] {
        let candidates = consolidation_candidates(&all_entries, scope);
        if candidates.is_empty() {
            continue;
        }
        scopes_with_candidates += 1;
        println!("📚 整理 {}（{} 条）", scope.label(), candidates.len());
        consolidate_scope(app, &store, scope, &candidates).await?;
    }

    if scopes_with_candidates == 0 {
        println!("📭 没有可整理的条目（仅处理有 ID、优先级 < 200 的文本记录）。");
        return Ok(());
    }

    println!("\n✨ Done.");
    Ok(())
}

/// Handle --note-delete / -nd <text>: have the model match the most relevant memo
/// entries in the knowledge base, resolve their ids, and ask the user to confirm
/// before deleting.
pub(super) async fn handle_note_delete(
    app: &mut App,
    query: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    // Join the query: flag value + remaining positional args; if both are empty,
    // open the multi-line input box.
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

    // Retrieve candidate entries.
    let candidates =
        match crate::ai::tools::service::memory::search_memo_candidates(&query, 10, false) {
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

    // Have the model pick the best-matching entries from the candidates (returns
    // their indices, or NONE).
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

    // Parse the indices returned by the model (supports comma / space /
    // enumeration-comma separators), dedupe, and keep ascending order.
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

    // Confirmation + selection before deleting. After listing, the user can:
    //   - press Enter directly / y / all / a: delete all listed entries
    //   - enter indices (e.g. 1,3): delete only those entries
    //   - n / any other cancel word: cancel
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

    // Parse the user's choice into the final subset of targets to delete.
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
        // Parse the index list (against the 1..=targets.len() listing above).
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

/// Handle --note-edit / -ne <text>: have the model match related memo entries in
/// the knowledge base; with several matches the user picks exactly one, the
/// original text is prefilled in the editor, and the rewrite is saved (id kept,
/// timestamp updated).
pub(super) async fn handle_note_edit(
    app: &mut App,
    query: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    // Status-line coloring: keeps the note body (white on black) visually distinct.
    const NE: &str = "\x1b[1;36m[note-edit]\x1b[0m"; // bold cyan label
    const FIELD: &str = "\x1b[2m"; // field names (id/time/content) in dim gray
    const HINT: &str = "\x1b[1;32m"; // action hints in bold green
    const IDX: &str = "\x1b[1;33m"; // candidate indices in bold yellow
    const RST: &str = "\x1b[0m";

    // Join the query: flag value + remaining positional args; if both are empty,
    // open the multi-line input box.
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

    // Retrieval plus model matching can both take a while; show a spinner (cleared
    // automatically before output), same as -ns.
    let spinner = SearchSpinner::start("匹配知识库条目");

    // Retrieve candidate entries.
    let candidates =
        match crate::ai::tools::service::memory::search_memo_candidates(&query, 10, false) {
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

    // Have the model pick the matching entries from the candidates (possibly
    // several), returning their indices.
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

    // Parse the set of indices returned by the model.
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

    // Several matches: list them and let the user pick exactly one to edit
    // (editing operates on a single entry).
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

    // Prefill the original text in the editor for the user to rewrite.
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

    // Before saving, use the LLM to tidy the user's rewritten content: polish
    // formatting/wording only, strictly forbidden from changing semantics. On
    // tidy failure fall back to the user's edited text; saving is never blocked.
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
    fn note_search_mode_detection() {
        // `a -ns xxx`: with query content → one-shot single-round retrieval.
        let one_shot = crate::ai::cli::parse_cli_args(
            vec!["a".to_string(), "-ns".to_string(), "trait object".to_string()].into_iter(),
        );
        assert!(one_shot.note_search);
        assert!(!note_search_interactive_mode(&one_shot));

        // `a -ns`: no substantive content → automatically interactive (equivalent
        // to `a -ns -i`).
        let auto = crate::ai::cli::parse_cli_args(vec!["a".to_string(), "-ns".to_string()].into_iter());
        assert!(auto.note_search);
        assert!(note_search_interactive_mode(&auto));

        // Blank/empty strings do not count as content.
        let blank = crate::ai::cli::parse_cli_args(
            vec!["a".to_string(), "-ns".to_string(), "   ".to_string()].into_iter(),
        );
        assert!(note_search_interactive_mode(&blank));

        // `a -ns -i`: explicit interactive mode, original semantics kept.
        let explicit = crate::ai::cli::parse_cli_args(
            vec!["a".to_string(), "-ns".to_string(), "-i".to_string()].into_iter(),
        );
        assert!(note_search_interactive_mode(&explicit));

        // Without -ns: even with no content this is not notebook-search
        // interactive mode.
        let no_ns = crate::ai::cli::parse_cli_args(vec!["a".to_string()].into_iter());
        assert!(!note_search_interactive_mode(&no_ns));
    }

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

        let eligible_ids = FxHashSet::from_iter(["id_a", "id_b", "ignored"].into_iter());
        // Source entry text: must be preserved losslessly after merging, not just
        // the model summary.
        let entry_a = AgentMemoryEntry {
            id: Some("id_a".to_string()),
            timestamp: "2026-07-24T00:00:00Z".to_string(),
            category: "memo".to_string(),
            note: "完整原文A：详细的排障步骤与具体命令".to_string(),
            tags: Vec::new(),
            source: Some("cli_note".to_string()),
            priority: Some(150),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        let entry_b = AgentMemoryEntry {
            id: Some("id_b".to_string()),
            timestamp: "2026-07-24T00:00:00Z".to_string(),
            category: "memo".to_string(),
            note: "完整原文B：关键标识符与复现步骤".to_string(),
            tags: Vec::new(),
            source: Some("cli_note".to_string()),
            priority: Some(150),
            owner_pid: None,
            owner_pgid: None,
            image_path: None,
        };
        let originals = FxHashMap::from_iter([
            ("id_a", &entry_a),
            ("id_b", &entry_b),
        ]);
        let (delete_ids, merged_count, new_entries) = build_consolidation_merge_entries(
            "memo",
            Some("test"),
            &merge_plan_refs,
            &eligible_ids,
            &originals,
        );

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
        // Lossless merge: summary + the full text of every source entry.
        assert_eq!(
            new_entries[0].note,
            "合并后的内容\n\n--- 原文保留（合并来源）---\n完整原文A：详细的排障步骤与具体命令\n\n完整原文B：关键标识符与复现步骤"
        );
        assert_eq!(new_entries[0].tags, vec!["consolidated".to_string()]);
    }

    #[test]
    fn memo_consolidation_isolated_from_other_categories() {
        let entry = |id: &str, category: &str, image_path: Option<&str>| AgentMemoryEntry {
            id: Some(id.to_string()),
            timestamp: "2026-07-24T00:00:00Z".to_string(),
            category: category.to_string(),
            note: format!("{id} 内容"),
            tags: Vec::new(),
            source: Some("cli_note".to_string()),
            priority: Some(150),
            owner_pid: None,
            owner_pgid: None,
            image_path: image_path.map(str::to_string),
        };
        let entries = vec![
            entry("memo_a", "memo", None),
            entry("memo_b", "memo", None),
            entry("memo_image", "memo", Some("/tmp/note.png")),
            entry("knowledge_a", "user_memory", None),
        ];

        let candidates = consolidation_candidates(&entries, ConsolidationScope::Memo);
        assert_eq!(candidates.len(), 2);
        assert!(candidates.iter().all(|entry| entry.category == "memo"));
        assert!(candidates.iter().all(|entry| entry.image_path.is_none()));

        let eligible_ids: FxHashSet<&str> = candidates
            .iter()
            .filter_map(|entry| entry.id.as_deref())
            .collect();
        let merge_plan = [serde_json::json!({
            "ids": ["memo_a", "memo_b", "knowledge_a"],
            "merged_content": "合并后的 memo"
        })];
        let merge_plan_refs: Vec<&serde_json::Value> = merge_plan.iter().collect();
        let originals: FxHashMap<&str, &AgentMemoryEntry> = candidates
            .iter()
            .filter_map(|entry| entry.id.as_deref().map(|id| (id, *entry)))
            .collect();
        let (delete_ids, merged_count, new_entries) = build_consolidation_merge_entries(
            "memo",
            Some("cli_note"),
            &merge_plan_refs,
            &eligible_ids,
            &originals,
        );

        assert_eq!(merged_count, 2);
        assert!(delete_ids.contains("memo_a"));
        assert!(delete_ids.contains("memo_b"));
        assert!(!delete_ids.contains("knowledge_a"));
        assert_eq!(new_entries.len(), 1);
        assert_eq!(new_entries[0].category, "memo");
        assert_eq!(new_entries[0].source.as_deref(), Some("cli_note"));
        // Lossless merge: the source memos' full text must be preserved in the new
        // entry.
        assert!(new_entries[0].note.contains("合并后的 memo"));
        assert!(new_entries[0].note.contains("memo_a 内容"));
        assert!(new_entries[0].note.contains("memo_b 内容"));
    }
}
