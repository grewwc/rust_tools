mod blob;
mod checkpoint;
mod compress;
mod markdown;
mod sessions;
mod sqlite;
mod suspended;
mod types;

use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::SystemTime;

use crate::ai::types::App;
#[allow(unused_imports)]
pub(in crate::ai) use blob::{
    append_history, append_history_messages, append_history_messages_uncompacted,
    build_message_arr, delete_history_artifacts, replace_history_messages,
};
#[allow(unused_imports)]
pub(in crate::ai) use checkpoint::{CheckpointInfo, CheckpointStore};
#[allow(unused_imports)]
pub(in crate::ai) use compress::compress_messages_for_context;
#[allow(unused_imports)]
pub(in crate::ai) use compress::value_to_string;
#[allow(unused_imports)]
pub(in crate::ai) use compress::{
    messages_total_chars_pub, mid_turn_compress, mid_turn_llm_summarize,
};
#[allow(unused_imports)]
pub(in crate::ai) use markdown::messages_to_markdown;
pub(in crate::ai) use sessions::generate_session_summary;
#[allow(unused_imports)]
pub(in crate::ai) use sessions::{SessionInfo, SessionStore};
#[allow(unused_imports)]
pub(in crate::ai) use sqlite::read_recent_messages_sqlite;
#[allow(unused_imports)]
pub(in crate::ai) use sqlite::read_recent_turn_window_sqlite;
#[allow(unused_imports)]
pub(in crate::ai) use suspended::{
    SuspendedSessionEntry, SuspendedSessionStore, format_suspended_timestamp_label,
};
#[allow(unused_imports)]
pub(in crate::ai) use types::{COLON, MAX_HISTORY_TURNS, Message, NEWLINE};

pub(in crate::ai) const ROLE_SYSTEM: &str = types::ROLE_SYSTEM;
pub(in crate::ai) const ROLE_INTERNAL_NOTE: &str = types::ROLE_INTERNAL_NOTE;

const CONTEXT_HISTORY_CACHE_LIMIT: usize = 8;

static CONTEXT_HISTORY_CACHE: LazyLock<Mutex<Vec<ContextHistoryCacheEntry>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

#[derive(Clone, PartialEq, Eq)]
struct ContextHistoryCacheKey {
    history_file: PathBuf,
    history_count: usize,
    history_max_chars: usize,
    history_keep_last: usize,
    history_summary_max_chars: usize,
    overflow_dir: Option<PathBuf>,
    file_len: Option<u64>,
    modified_unix_ms: Option<u128>,
    /// SQLite `PRAGMA data_version`：每次 DB 被修改后递增。
    /// WAL 模式下主文件 len/mtime 可能长时间不变，单独依赖文件元数据
    /// 会让 cache 错误命中已删/已改的历史；data_version 是 SQLite
    /// 内建的强失效信号。
    sqlite_data_version: Option<i64>,
}

struct ContextHistoryCacheEntry {
    key: ContextHistoryCacheKey,
    value: Arc<Vec<Message>>,
}

pub(in crate::ai) fn is_internal_note_role(role: &str) -> bool {
    types::is_internal_note_role(role)
}

pub(in crate::ai) fn is_system_like_role(role: &str) -> bool {
    types::is_system_like_role(role)
}

pub(in crate::ai) fn build_context_history(
    history_count: usize,
    history_file: &Path,
    history_max_chars: usize,
    history_keep_last: usize,
    history_summary_max_chars: usize,
    overflow_dir: Option<PathBuf>,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let cache_key = context_history_cache_key(
        history_file,
        history_count,
        history_max_chars,
        history_keep_last,
        history_summary_max_chars,
        overflow_dir.as_deref(),
    );
    if let Some(cached) = try_get_cached_context_history(&cache_key) {
        return Ok(cached);
    }

    if history_max_chars > 0
        && blob::is_sqlite_path(history_file)
        && let Some(fast) = try_build_context_history_sqlite_fastpath(
            history_file,
            history_count,
            history_max_chars,
            history_keep_last,
            history_summary_max_chars,
            overflow_dir.as_deref(),
        )?
    {
        store_cached_context_history(cache_key, fast.clone());
        return Ok(fast);
    }

    let history = build_message_arr(usize::MAX, history_file)?;
    let out = if history_max_chars == 0 {
        if history_count >= history.len() {
            history
        } else {
            history[history.len() - history_count..].to_vec()
        }
    } else {
        let keep_last = if history_count == 0 {
            history_keep_last
        } else {
            history_count
        };
        compress_messages_for_context(
            history,
            history_max_chars,
            keep_last,
            history_summary_max_chars,
            overflow_dir,
        )
    };
    store_cached_context_history(cache_key, out.clone());
    Ok(out)
}

fn try_build_context_history_sqlite_fastpath(
    history_file: &Path,
    history_count: usize,
    history_max_chars: usize,
    history_keep_last: usize,
    history_summary_max_chars: usize,
    overflow_dir: Option<&std::path::Path>,
) -> Result<Option<Vec<Message>>, Box<dyn std::error::Error>> {
    let keep_last = if history_count == 0 {
        history_keep_last
    } else {
        history_count
    };
    let recent = sqlite::read_recent_turn_window_sqlite(history_file, keep_last)?;
    if recent.messages.is_empty() {
        return Ok(Some(Vec::new()));
    }
    if !recent.has_older_messages {
        return Ok(Some(compress_messages_for_context(
            recent.messages,
            history_max_chars,
            keep_last,
            history_summary_max_chars,
            overflow_dir.map(|p| p.to_path_buf()),
        )));
    }

    let Some(start_message_id) = recent.start_message_id else {
        return Ok(None);
    };
    let Some(summary) =
        sqlite::read_latest_history_summary_before_id_sqlite(history_file, start_message_id)?
    else {
        return Ok(None);
    };

    let mut messages = Vec::with_capacity(recent.messages.len() + 1);
    messages.push(summary);
    messages.extend(recent.messages);
    Ok(Some(compress_messages_for_context(
        messages,
        history_max_chars,
        keep_last,
        history_summary_max_chars,
        overflow_dir.map(|p| p.to_path_buf()),
    )))
}

fn context_history_cache_key(
    history_file: &Path,
    history_count: usize,
    history_max_chars: usize,
    history_keep_last: usize,
    history_summary_max_chars: usize,
    overflow_dir: Option<&Path>,
) -> ContextHistoryCacheKey {
    let metadata = std::fs::metadata(history_file).ok();
    let file_len = metadata.as_ref().map(|m| m.len());
    let modified_unix_ms = metadata
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(system_time_millis);
    let sqlite_data_version = if blob::is_sqlite_path(history_file) {
        sqlite::read_data_version(history_file)
    } else {
        None
    };
    ContextHistoryCacheKey {
        history_file: history_file.to_path_buf(),
        history_count,
        history_max_chars,
        history_keep_last,
        history_summary_max_chars,
        overflow_dir: overflow_dir.map(Path::to_path_buf),
        file_len,
        modified_unix_ms,
        sqlite_data_version,
    }
}

fn system_time_millis(value: SystemTime) -> Option<u128> {
    value
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis())
}

fn try_get_cached_context_history(key: &ContextHistoryCacheKey) -> Option<Vec<Message>> {
    let cache = CONTEXT_HISTORY_CACHE.lock().ok()?;
    cache
        .iter()
        .find(|entry| &entry.key == key)
        .map(|entry| (*entry.value).clone())
}

fn store_cached_context_history(key: ContextHistoryCacheKey, value: Vec<Message>) {
    let Ok(mut cache) = CONTEXT_HISTORY_CACHE.lock() else {
        return;
    };
    cache.retain(|entry| entry.key != key);
    cache.insert(
        0,
        ContextHistoryCacheEntry {
            key,
            value: Arc::new(value),
        },
    );
    if cache.len() > CONTEXT_HISTORY_CACHE_LIMIT {
        cache.truncate(CONTEXT_HISTORY_CACHE_LIMIT);
    }
}

/// 清除指定 history_file 的所有 context 缓存条目。
/// session 切换 / clear-history / delete 时调用，避免下个 turn 命中
/// 已经被删/被替换的旧历史。
pub(in crate::ai) fn invalidate_context_history_cache_for(history_file: &std::path::Path) {
    let Ok(mut cache) = CONTEXT_HISTORY_CACHE.lock() else {
        return;
    };
    cache.retain(|entry| entry.key.history_file != history_file);
}

/// 全量清空 context history 缓存。极端场景（如清理任务、单测）使用。
#[allow(dead_code)]
pub(in crate::ai) fn clear_context_history_cache() {
    if let Ok(mut cache) = CONTEXT_HISTORY_CACHE.lock() {
        cache.clear();
    }
}

pub(in crate::ai) async fn compact_session_history_with_app(
    app: &App,
) -> Result<(), Box<dyn std::error::Error>> {
    compact_session_history_with_app_inner(app, false).await
}

/// 任务边界触发的压缩：阈值更激进（160 vs 200），适合 turn 收尾且 agent 没有
/// 再调工具的"答案已交付"时刻调用。
pub(in crate::ai) async fn compact_session_history_at_boundary_with_app(
    app: &App,
) -> Result<(), Box<dyn std::error::Error>> {
    compact_session_history_with_app_inner(app, true).await
}

async fn compact_session_history_with_app_inner(
    app: &App,
    at_boundary: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let history_file = &app.session_history_file;
    // === 增量游标式快路径 ===
    // 99% 的 turn 收尾时，user_turns 远低于压缩阈值，根本不需要全量读。
    // 这里先做廉价的 COUNT，未达阈值直接返回，避免反序列化几十 MB 的 tool 输出。
    if blob::is_sqlite_path(history_file) {
        let user_turns = sqlite::count_user_turns_sqlite(history_file.as_path())?;
        let threshold = if at_boundary {
            crate::ai::history::compress::persisted_history_keep_recent_turns()
        } else {
            MAX_HISTORY_TURNS
        };
        if user_turns <= threshold {
            return Ok(());
        }
    }

    let messages = if blob::is_sqlite_path(history_file) {
        sqlite::build_message_arr_sqlite(usize::MAX, history_file.as_path())?
    } else {
        let history = match std::fs::read_to_string(history_file) {
            Ok(history) => history,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(err) => return Err(err.into()),
        };
        blob::parse_history_blob(&history)
    };

    let compacted = if at_boundary {
        compress::compact_persisted_history_at_boundary_with_app(app, messages.clone()).await
    } else {
        compress::compact_persisted_history_with_app(app, messages.clone()).await
    };
    if compacted == messages {
        return Ok(());
    }

    if blob::is_sqlite_path(history_file) {
        sqlite::replace_all_messages_sqlite(history_file.as_path(), &compacted)?;
    } else {
        std::fs::write(
            history_file,
            blob::serialize_history_messages_for_storage(&compacted),
        )?;
    }
    Ok(())
}
