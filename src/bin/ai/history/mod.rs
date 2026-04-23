mod blob;
mod compress;
mod markdown;
mod sessions;
mod sqlite;
mod types;

use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::SystemTime;

#[allow(unused_imports)]
pub(in crate::ai) use blob::{
    append_history, append_history_messages, append_history_messages_uncompacted, build_message_arr,
    delete_history_artifacts,
};
#[allow(unused_imports)]
pub(in crate::ai) use compress::compress_messages_for_context;
#[allow(unused_imports)]
pub(in crate::ai) use compress::value_to_string;
#[allow(unused_imports)]
pub(in crate::ai) use markdown::messages_to_markdown;
#[allow(unused_imports)]
pub(in crate::ai) use sessions::{SessionInfo, SessionStore};
#[allow(unused_imports)]
pub(in crate::ai) use sqlite::read_recent_turn_window_sqlite;
#[allow(unused_imports)]
pub(in crate::ai) use types::{COLON, MAX_HISTORY_TURNS, Message, NEWLINE};
use crate::ai::types::App;

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
    file_len: Option<u64>,
    modified_unix_ms: Option<u128>,
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
    history_file: &PathBuf,
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
    history_file: &PathBuf,
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
    let recent = sqlite::read_recent_turn_window_sqlite(history_file.as_path(), keep_last)?;
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
        sqlite::read_latest_history_summary_before_id_sqlite(history_file.as_path(), start_message_id)?
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
    history_file: &PathBuf,
    history_count: usize,
    history_max_chars: usize,
    history_keep_last: usize,
    history_summary_max_chars: usize,
) -> ContextHistoryCacheKey {
    let metadata = std::fs::metadata(history_file).ok();
    let file_len = metadata.as_ref().map(|m| m.len());
    let modified_unix_ms = metadata
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(system_time_millis);
    ContextHistoryCacheKey {
        history_file: history_file.clone(),
        history_count,
        history_max_chars,
        history_keep_last,
        history_summary_max_chars,
        file_len,
        modified_unix_ms,
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
    cache.iter().find(|entry| &entry.key == key).map(|entry| (*entry.value).clone())
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

pub(in crate::ai) async fn compact_session_history_with_app(
    app: &App,
) -> Result<(), Box<dyn std::error::Error>> {
    let history_file = &app.session_history_file;
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

    let compacted = compress::compact_persisted_history_with_app(app, messages.clone()).await;
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
