mod archive;
mod blob;
mod checkpoint;
pub(crate) mod compress;
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
    is_summary_note_text, message_billable_chars, messages_total_chars_pub, mid_turn_compress,
    mid_turn_llm_summarize,
};
#[allow(unused_imports)]
pub(in crate::ai) use markdown::messages_to_markdown;
pub(in crate::ai) use sessions::generate_session_summary;
#[allow(unused_imports)]
pub(in crate::ai) use sessions::strip_think_tags;
#[allow(unused_imports)]
pub(in crate::ai) use sessions::{SessionInfo, SessionStore};
#[allow(unused_imports)]
pub(in crate::ai) use sqlite::{
    append_tool_execution_outcomes_sqlite, read_recent_messages_sqlite,
    read_stale_patch_targets_sqlite, read_tool_execution_outcomes_sqlite,
    read_tool_message_ids_sqlite, write_stale_patch_targets_sqlite,
};
#[allow(unused_imports)]
pub(in crate::ai) use sqlite::read_recent_turn_window_sqlite;
#[allow(unused_imports)]
pub(in crate::ai) use suspended::{
    SuspendedSessionEntry, SuspendedSessionStore, format_suspended_timestamp_label,
};
#[allow(unused_imports)]
pub(in crate::ai) use types::{
    COLON, MAX_HISTORY_TURNS, Message, NEWLINE, ToolExecutionOutcome,
};

pub(in crate::ai) const ROLE_SYSTEM: &str = types::ROLE_SYSTEM;
pub(in crate::ai) const ROLE_INTERNAL_NOTE: &str = types::ROLE_INTERNAL_NOTE;

pub(in crate::ai) fn normalize_generated_session_title(title: &str) -> String {
    sessions::normalize_generated_session_title(title)
}

pub(in crate::ai) fn is_low_quality_session_title(title: &str) -> bool {
    sessions::is_low_quality_session_title(title)
}

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
    /// history DB 的写入版本号（`meta.history_revision`）：每次写事务内递增。
    /// WAL 模式下主文件 len/mtime 可能长时间不变，单独依赖文件元数据会让 cache
    /// 错误命中已删/已改的历史。该版本号是**跨连接**可见的强失效信号，
    /// 取代不可靠的 `PRAGMA data_version`（后者是连接局部值，新连接读到的初值
    /// 不随外部写入而变）。
    history_revision: Option<i64>,
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

/// `/history` 人工查看入口需要展示完整会话，而不是只展示压缩后留在主历史库里的
/// inline 消息。归档仅在查看时展开，不进入模型上下文，也不参与 rewind 写回。
pub(in crate::ai) fn build_message_arr_for_history_view(
    history_file: &Path,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let messages = build_message_arr(usize::MAX, history_file)?;
    Ok(archive::expand_overflow_archives(messages))
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

    let checkpoint_markers =
        sqlite::read_context_checkpoint_markers_before_id_sqlite(history_file, start_message_id)?;
    let mut messages = Vec::with_capacity(recent.messages.len() + checkpoint_markers.len() + 1);
    messages.push(summary);
    messages.extend(checkpoint_markers);
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
    let history_revision = if blob::is_sqlite_path(history_file) {
        sqlite::read_history_revision(history_file)
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
        history_revision,
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
    // 常规情况下 user_turns 远低于压缩阈值，根本不需要全量读。但不能只看
    // turn 数：单个 read_file/命令输出就可能把短会话撑到数 MB，若不在这里落盘
    // 压缩，下轮又会从原始历史重新做一次昂贵的请求期压缩。
    if blob::is_sqlite_path(history_file) {
        let user_turns = sqlite::count_user_turns_sqlite(history_file.as_path())?;
        let exceeds_context_budget = app.config.history_max_chars > 0
            && sqlite::total_message_chars_sqlite(history_file.as_path())?
                > app.config.history_max_chars;
        let exceeds_tool_evidence_budget = sqlite::compressed_tool_evidence_chars_sqlite(
            history_file.as_path(),
        )? > compress::compressed_tool_evidence_inline_chars_limit();
        let threshold = if at_boundary {
            crate::ai::history::compress::persisted_history_keep_recent_turns()
        } else {
            MAX_HISTORY_TURNS
        };
        if user_turns <= threshold
            && !exceeds_context_budget
            && !exceeds_tool_evidence_budget
        {
            return Ok(());
        }
    }

    let messages = if blob::is_sqlite_path(history_file) {
        // 用**原始** raw read，而非 build_message_arr_sqlite（后者会先跑启发式
        // compact_persisted_history）。app-aware 压缩必须看到未经启发式压缩的原文，
        // 语义与非-sqlite 分支的 parse_history_blob 对齐；随后压缩函数内部照常
        // sanitize_persisted_history_messages。
        sqlite::read_all_messages_sqlite(history_file.as_path())?
    } else {
        let history = match std::fs::read_to_string(history_file) {
            Ok(history) => history,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(err) => return Err(err.into()),
        };
        blob::parse_history_blob(&history)
    };

    let original_chars = messages_total_chars_pub(&messages);
    let exceeds_context_budget =
        app.config.history_max_chars > 0 && original_chars > app.config.history_max_chars;
    let exceeds_tool_evidence_budget =
        compress::compressed_tool_evidence_exceeds_inline_budget(&messages);
    let compacted = if exceeds_context_budget || exceeds_tool_evidence_budget {
        // 与下一轮 `build_context_history` 使用完全相同的压缩策略，并把结果写回
        // history。原始的大块内容会进入 session assets，stub 保留可精确读回的
        // 路径和预览；因此降低重复压缩成本不会损失可召回的信息。
        let store = SessionStore::new(app.config.history_file.as_path());
        compress::compress_messages_for_context(
            messages.clone(),
            app.config.history_max_chars,
            app.config.history_keep_last,
            app.config.history_summary_max_chars,
            Some(store.session_assets_dir(&app.session_id)),
        )
    } else if at_boundary {
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
    let reason = if exceeds_context_budget {
        "context-budget"
    } else if exceeds_tool_evidence_budget {
        "tool-evidence-budget"
    } else {
        "turn-count"
    };
    eprintln!(
        "[history] persisted {reason} compaction: {original_chars} -> {} chars",
        messages_total_chars_pub(&compacted)
    );
    Ok(())
}
