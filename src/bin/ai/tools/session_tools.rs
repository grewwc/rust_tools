//! Read-only cross-session inspection tools: `session_list`, `session_read`,
//! `session_search`.
//!
//! These let a session (agent) inspect the context of other sessions in the
//! same session store without ever writing to them: canonical messages go
//! through the read-only sqlite open (no migration, no journal switch), asset
//! search is a plain bounded `fs` walk, and session ids are validated before
//! they touch any path. No lock, cache, or derived file is created, so the
//! "session lifecycle is user-owned" invariant stays intact. Members of the
//! `session` group are lazy (loaded via `enable_tools`), keeping them out of
//! every turn's resident schema.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::ai::history::{SessionInfo, SessionStore, read_all_messages_sqlite_read_only};
use crate::ai::tools::common::{ToolRegistration, ToolSpec};

const MAX_OUTPUT_CHARS: usize = 24_000;
const MAX_MESSAGE_CHARS: usize = 600;
const MAX_ASSET_FILE_BYTES: u64 = 2 * 1024 * 1024;
const MAX_ASSET_FILES: usize = 200;
const MAX_ASSET_DEPTH: usize = 3;
const SNIPPET_WINDOW: usize = 120;
const MAX_CHECKPOINTS_SHOWN: usize = 20;

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "session_list",
        description: "",

        execute: execute_session_list,
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "session_read",
        description: "",

        execute: execute_session_read,
    }
});

inventory::submit!(ToolRegistration {
    spec: ToolSpec {
        name: "session_search",
        description: "",

        execute: execute_session_search,
    }
});

/// The session store of the running driver. Fails closed when no driver
/// context is installed (one-shot calls / unit tests).
fn current_store() -> Result<SessionStore, String> {
    match crate::ai::driver::runtime_ctx::try_current() {
        Some(ctx) => Ok(SessionStore::new(&ctx.app_proto.config.history_file)),
        None => Err(
            "session tools need an active driver context (not available in one-shot/test calls)"
                .to_string(),
        ),
    }
}

/// Validate a session id and resolve it to its canonical sqlite file, failing
/// with a clear error when the session does not exist.
fn resolve_session_file(store: &SessionStore, session_id: &str) -> Result<PathBuf, String> {
    SessionStore::validate_session_id(session_id).map_err(|e| e.to_string())?;
    let file = store.session_history_file(session_id);
    if !file.is_file() {
        return Err(format!(
            "session '{session_id}' does not exist (no history found for this id)"
        ));
    }
    Ok(file)
}

fn content_to_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Render one message as a single line. Assistant messages whose body is empty
/// but carry tool calls show the calls instead of an empty line.
fn render_message(
    role: &str,
    content: &Value,
    tool_calls: Option<&Vec<crate::ai::types::ToolCall>>,
) -> String {
    let mut body = content_to_text(content);
    if body.trim().is_empty() {
        if let Some(calls) = tool_calls {
            let rendered: Vec<String> = calls
                .iter()
                .map(|c| {
                    format!(
                        "{}({})",
                        c.function.name,
                        truncate(&c.function.arguments, SNIPPET_WINDOW)
                    )
                })
                .collect();
            body = format!("tool_call: {}", rendered.join(", "));
        } else {
            body = "(empty)".to_string();
        }
    }
    format!("[{role}] {}", truncate(&body, MAX_MESSAGE_CHARS))
}

/// Truncate at a char boundary, keeping the removed length visible.
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…(+{} chars)", &s[..end], s[end..].chars().count())
}

/// A ~`SNIPPET_WINDOW`-char window centered on the first match of `needle`.
fn snippet_around(text: &str, needle: &str) -> String {
    let lower = text.to_lowercase();
    match lower.find(needle) {
        Some(pos) => {
            let start = pos.saturating_sub(SNIPPET_WINDOW / 3);
            let end = (pos + needle.len() + SNIPPET_WINDOW).min(text.len());
            let start = (0..=start).rev().find(|&i| text.is_char_boundary(i)).unwrap_or(0);
            let end = (end..=text.len()).find(|&i| text.is_char_boundary(i)).unwrap_or(text.len());
            let mut out = String::new();
            if start > 0 {
                out.push('…');
            }
            out.push_str(&text[start..end]);
            if end < text.len() {
                out.push('…');
            }
            out
        }
        None => truncate(text, SNIPPET_WINDOW),
    }
}

fn truncate_output(out: &str) -> String {
    if out.len() <= MAX_OUTPUT_CHARS {
        return out.to_string();
    }
    let mut end = MAX_OUTPUT_CHARS;
    while !out.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}…\n-- output truncated at {} chars",
        &out[..end],
        MAX_OUTPUT_CHARS
    )
}

/// Extensions that are never worth text-searching.
fn is_textish(path: &Path) -> bool {
    const SKIP: &[&str] = &[
        "png", "jpg", "jpeg", "gif", "webp", "ico", "pdf", "zip", "sqlite", "sqlite-wal",
        "sqlite-shm", "bin", "wav", "mp3", "mp4", "mov", "ttf", "woff", "woff2", "lock", "pid",
        "state",
    ];
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
    {
        Some(ext) => !SKIP.contains(&ext.as_str()),
        None => true,
    }
}

/// Walk `<session>.assets/` (depth- and size-bounded) and collect text files
/// containing `needle`, with per-file hit counts for ranking.
fn collect_asset_hits(root: &Path, dir: &Path, needle: &str, depth: usize, hits: &mut Vec<(String, usize, String)>) {
    if depth > MAX_ASSET_DEPTH || hits.len() >= MAX_ASSET_FILES {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        // Never follow symlinks: a link inside the assets dir could point
        // outside it (or at a device that never returns EOF).
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        if file_type.is_symlink() {
            continue;
        }
        let path = entry.path();
        let meta = match entry.metadata() {
            Ok(meta) => meta,
            Err(_) => continue,
        };
        if meta.is_dir() {
            collect_asset_hits(root, &path, needle, depth + 1, hits);
        } else if meta.len() <= MAX_ASSET_FILE_BYTES && is_textish(&path) {
            let mut buf = Vec::new();
            if std::fs::File::open(&path)
                .and_then(|mut f| f.read_to_end(&mut buf))
                .is_err()
            {
                continue;
            }
            let text = String::from_utf8_lossy(&buf);
            let count = text.to_lowercase().matches(needle).count();
            if count > 0 {
                let rel = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .to_string();
                hits.push((rel, count, snippet_around(&text, needle)));
            }
        }
    }
}

/// `session_list`: enumerate sessions of the store, newest activity first.
fn execute_session_list(args: &Value) -> Result<String, String> {
    let limit = args["limit"].as_u64().unwrap_or(20).min(100) as usize;
    let query = args["query"].as_str().map(|q| q.to_lowercase());
    let store = current_store()?;
    let mut sessions: Vec<SessionInfo> = store.list_sessions().map_err(|e| e.to_string())?;
    sessions.sort_by(|a, b| b.modified_local.cmp(&a.modified_local));

    let mut out = String::new();
    let mut shown = 0usize;
    for session in sessions {
        if let Some(q) = &query {
            let title = session.summary.as_deref().unwrap_or("");
            let prompt = session.first_user_prompt.as_deref().unwrap_or("");
            if !session.id.to_lowercase().contains(q)
                && !title.to_lowercase().contains(q)
                && !prompt.to_lowercase().contains(q)
            {
                continue;
            }
        }
        if shown >= limit {
            break;
        }
        let title = session.summary.as_deref().unwrap_or("(untitled)");
        let modified = session
            .modified_local
            .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "-".to_string());
        let marked = if session.marked { " [marked]" } else { "" };
        let prompt = session
            .first_user_prompt
            .as_deref()
            .unwrap_or("")
            .chars()
            .take(80)
            .collect::<String>();
        let prompt = if prompt.is_empty() {
            String::new()
        } else {
            format!(" | {}", prompt.replace('\n', " "))
        };
        out.push_str(&format!(
            "{:>3}. {} | {} | {}{}{}\n",
            shown + 1,
            session.id,
            modified,
            title,
            marked,
            prompt
        ));
        shown += 1;
    }
    if shown == 0 {
        return Ok("No sessions found.".to_string());
    }
    out.push_str(&format!("-- {} session(s) listed (limit {limit})", shown));
    Ok(truncate_output(&out))
}

/// `session_read`: view one session's context. `mode=digest` (default) shows a
/// header (title / prompt / size / checkpoints) plus the most recent messages;
/// `mode=raw` renders messages only.
fn execute_session_read(args: &Value) -> Result<String, String> {
    let session_id = args["session_id"]
        .as_str()
        .ok_or("Missing 'session_id' parameter")?;
    let mode = args["mode"].as_str().unwrap_or("digest");
    if mode != "digest" && mode != "raw" {
        return Err(format!("'mode' must be 'digest' or 'raw', got '{mode}'"));
    }
    let default_limit = if mode == "digest" { 20 } else { 60 };
    let limit = args["limit"].as_u64().unwrap_or(default_limit).min(200) as usize;

    let store = current_store()?;
    let file = resolve_session_file(&store, session_id)?;
    let messages = read_all_messages_sqlite_read_only(&file).map_err(|e| e.to_string())?;

    let mut out = String::new();
    if mode == "digest" {
        let info = store
            .list_sessions()
            .map_err(|e| e.to_string())?
            .into_iter()
            .find(|s| s.id == session_id);
        if let Some(info) = info {
            out.push_str(&format!("session: {}\n", info.id));
            out.push_str(&format!(
                "title: {}\n",
                info.summary.as_deref().unwrap_or("(untitled)")
            ));
            if let Some(modified) = info.modified_local {
                out.push_str(&format!("last activity: {}\n", modified.format("%Y-%m-%d %H:%M")));
            }
            if info.marked {
                out.push_str("marked: yes\n");
            }
            if let Some(prompt) = info.first_user_prompt.as_deref() {
                out.push_str(&format!(
                    "first prompt: {}\n",
                    truncate(&prompt.replace('\n', " "), 240)
                ));
            }
            if info.size_bytes > 0 {
                out.push_str(&format!("store size: {} bytes\n", info.size_bytes));
            }
        }
        // Checkpoints are listed by direct directory read (never via
        // CheckpointStore, whose lock/recovery writes to the session).
        let ckpt_dir = store.checkpoints_dir(session_id);
        let mut checkpoints: Vec<(String, String, u64)> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&ckpt_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if let Ok(meta) = entry.metadata() {
                    let modified = meta
                        .modified()
                        .ok()
                        .and_then(|t| {
                            chrono::DateTime::<chrono::Local>::from(t)
                                .format("%Y-%m-%d %H:%M")
                                .to_string()
                                .into()
                        })
                        .unwrap_or_else(|| "-".to_string());
                    checkpoints.push((name, modified, meta.len()));
                }
            }
        }
        checkpoints.sort_by(|a, b| b.1.cmp(&a.1));
        if checkpoints.is_empty() {
            out.push_str("checkpoints: none\n");
        } else {
            out.push_str(&format!("checkpoints ({}):\n", checkpoints.len()));
            for (name, modified, size) in checkpoints.iter().take(MAX_CHECKPOINTS_SHOWN) {
                out.push_str(&format!("  {name} | {modified} | {size} bytes\n"));
            }
        }
        out.push('\n');
    }

    let start = messages.len().saturating_sub(limit);
    for message in &messages[start..] {
        out.push_str(&render_message(
            &message.role,
            &message.content,
            message.tool_calls.as_ref(),
        ));
        out.push('\n');
    }
    out.push_str(&format!(
        "-- {} of {} messages shown",
        messages.len() - start,
        messages.len()
    ));
    Ok(truncate_output(&out))
}

/// `session_search`: substring search over one session's canonical messages
/// (newest first) and its text assets; each result shows a snippet around the
/// match.
fn execute_session_search(args: &Value) -> Result<String, String> {
    let session_id = args["session_id"]
        .as_str()
        .ok_or("Missing 'session_id' parameter")?;
    let query = args["query"].as_str().ok_or("Missing 'query' parameter")?;
    let max_results = args["max_results"].as_u64().unwrap_or(20).min(100) as usize;
    let needle = query.to_lowercase();
    if needle.is_empty() {
        return Err("'query' must not be empty".to_string());
    }

    let store = current_store()?;
    let file = resolve_session_file(&store, session_id)?;
    let mut out = String::new();
    let mut count = 0usize;

    let messages = read_all_messages_sqlite_read_only(&file).map_err(|e| e.to_string())?;
    for (index, message) in messages.iter().enumerate().rev() {
        if count >= max_results {
            break;
        }
        let text = content_to_text(&message.content);
        if text.to_lowercase().contains(&needle) {
            out.push_str(&format!(
                "canonical #{index} [{}]: {}\n",
                message.role,
                snippet_around(&text, &needle)
            ));
            count += 1;
        }
    }

    let assets = store.session_assets_dir(session_id);
    let mut asset_hits: Vec<(String, usize, String)> = Vec::new();
    collect_asset_hits(&assets, &assets, &needle, 0, &mut asset_hits);
    asset_hits.sort_by(|a, b| b.1.cmp(&a.1));
    for (path, hits, snippet) in asset_hits {
        if count >= max_results {
            break;
        }
        out.push_str(&format!("assets {path} ({hits} hit(s)): {snippet}\n"));
        count += 1;
    }

    if count == 0 {
        return Ok(format!(
            "No matches for '{query}' in session '{session_id}' (canonical messages + text assets)."
        ));
    }
    out.push_str(&format!("-- {count} match(es)"));
    Ok(truncate_output(&out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_keeps_char_boundaries_and_reports_removed_length() {
        let short = "hello";
        assert_eq!(truncate(short, 10), "hello");
        let long = "a".repeat(1000);
        assert_eq!(truncate(&long, 100).len(), 100 + "…(+900 chars)".len());
        // Multi-byte content must not panic or split a char.
        let emoji = "😀".repeat(500);
        let cut = truncate(&emoji, 10);
        assert!(cut.is_char_boundary(cut.len()) && cut.contains('…'));
    }

    #[test]
    fn snippet_around_centers_on_match() {
        // Long enough that the window must be cut on both sides.
        let text = format!(
            "{}dddd{}",
            "a".repeat(SNIPPET_WINDOW * 3),
            "b".repeat(SNIPPET_WINDOW * 3)
        );
        let snippet = snippet_around(&text, "dddd");
        assert!(snippet.contains("dddd"));
        assert!(snippet.starts_with('…'), "long text must be cut on the left");
        assert!(snippet.ends_with('…'), "long text must be cut on the right");
    }

    #[test]
    fn is_textish_skips_binary_extensions() {
        assert!(is_textish(Path::new("foo.txt")));
        assert!(is_textish(Path::new("overflow-history.md")));
        assert!(!is_textish(Path::new("session.sqlite")));
        assert!(!is_textish(Path::new("image.png")));
        assert!(!is_textish(Path::new("a.sqlite-wal")));
    }

    #[test]
    fn render_message_shows_tool_calls_when_body_empty() {
        let content = Value::String(String::new());
        let rendered = render_message("assistant", &content, None);
        assert!(rendered.contains("(empty)"));
        let calls = vec![crate::ai::types::ToolCall {
            id: "1".to_string(),
            tool_type: "function".to_string(),
            function: crate::ai::types::FunctionCall {
                name: "read_file".to_string(),
                arguments: "{\"file\":\"a.rs\"}".to_string(),
            },
        }];
        let rendered = render_message("assistant", &content, Some(&calls));
        assert!(rendered.contains("tool_call: read_file("));
    }

    #[test]
    fn truncate_reports_removed_chars_not_bytes() {
        let emoji = "😀".repeat(500); // 4 bytes per char
        let cut = truncate(&emoji, 10);
        let kept = cut.split("…(+").next().unwrap();
        let reported: usize = cut
            .rsplit("(+")
            .next()
            .unwrap()
            .split(" chars)")
            .next()
            .unwrap()
            .parse()
            .unwrap();
        // The label must count chars, not bytes (bytes would be 4x here).
        assert_eq!(reported, emoji.chars().count() - kept.chars().count());
        assert_ne!(reported, emoji.len() - kept.len());
    }

    #[test]
    fn resolve_session_file_validates_ids_and_existence() {
        let root = std::env::temp_dir().join(format!(
            "session_tools_test_{}_resolve",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        // SessionStore derives the sessions root as `<stem>.sessions/` next to
        // the history file (sessions_root_from_history_file).
        let store = SessionStore::new(&root.join("history-file"));
        std::fs::create_dir_all(store.sessions_root()).unwrap();
        std::fs::write(store.session_history_file("abcd1234"), b"").unwrap();

        // Traversal and malformed ids are rejected before any path is touched.
        for bad in ["", "../evil", "a/b", "a\\b", "..", "中文", "a b", "x".repeat(129).as_str()] {
            assert!(
                resolve_session_file(&store, bad).is_err(),
                "id {bad:?} should be rejected"
            );
        }
        // Valid but absent id -> clear not-found error.
        let err = resolve_session_file(&store, "zzzz9999").unwrap_err();
        assert!(err.contains("does not exist"), "unexpected error: {err}");
        // Valid and present id resolves to the canonical sqlite path.
        let file = resolve_session_file(&store, "abcd1234").unwrap();
        assert_eq!(file.file_name().unwrap().to_string_lossy(), "abcd1234.sqlite");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn collect_asset_hits_skips_symlinks() {
        let root = std::env::temp_dir().join(format!(
            "session_tools_test_{}_symlink",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root.join("assets")).unwrap();

        // Real file inside the assets dir: must be found.
        std::fs::write(root.join("assets").join("real.txt"), "needle inside the session").unwrap();
        // Files and dirs outside the assets dir: must stay invisible.
        let outside = root.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.txt"), "needle secret outside").unwrap();
        std::os::unix::fs::symlink(
            &outside.join("secret.txt"),
            root.join("assets").join("link.txt"),
        )
        .unwrap();
        std::os::unix::fs::symlink(&outside, root.join("assets").join("linkdir")).unwrap();

        let mut hits = Vec::new();
        collect_asset_hits(
            &root.join("assets"),
            &root.join("assets"),
            "needle",
            0,
            &mut hits,
        );
        assert_eq!(hits.len(), 1, "only the real file may match; hits={hits:?}");
        assert!(hits[0].0.contains("real.txt"));

        let _ = std::fs::remove_dir_all(&root);
    }
}
