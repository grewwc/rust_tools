mod archive;
mod blob;
mod checkpoint;
pub(crate) mod compress;
mod markdown;
mod sessions;
mod sqlite;
mod suspended;
mod types;

use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::SystemTime;

use crate::ai::types::App;
#[allow(unused_imports)]
pub(in crate::ai) use blob::{
    append_history, append_history_messages, append_history_messages_for_model, build_message_arr,
    delete_history_artifacts, replace_history_messages, truncate_history_messages,
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
pub(in crate::ai) use sessions::{SessionInfo, SessionStore, SessionTitle, SessionTitleOrigin};
#[allow(unused_imports)]
pub(in crate::ai) use sqlite::fork_history_for_subagent;

/// 为子代理准备独立的历史文件。首次派发按需 fork 父历史；resume 只复用既有
/// child 文件，绝不能再次用父快照覆盖子代理已经产生的证据。
pub(in crate::ai) fn prepare_subagent_history(
    parent: &Path,
    child: &Path,
    inherit_history: bool,
    initialize: bool,
) -> io::Result<()> {
    if initialize {
        let parent_is_sqlite = blob::is_sqlite_path(parent);
        let child_is_sqlite = blob::is_sqlite_path(child);
        if parent_is_sqlite != child_is_sqlite {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "父子代理历史后端不一致：{} -> {}",
                    parent.display(),
                    child.display()
                ),
            ));
        }
        return match (parent_is_sqlite, inherit_history) {
            (true, true) => fork_history_for_subagent(parent, child),
            (true, false) => sqlite::reset_history_for_subagent(child),
            (false, _) => publish_text_subagent_history(parent, child, inherit_history),
        };
    }
    if child.is_file() {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("子代理历史文件不存在，无法 resume：{}", child.display()),
        ))
    }
}

/// 文本历史后端不能交给 SQLite Online Backup。先写同目录临时文件再 rename，
/// 避免子代理观察到半份父历史；父文件尚未创建时继承语义等同于空历史。
fn publish_text_subagent_history(
    parent: &Path,
    child: &Path,
    inherit_history: bool,
) -> io::Result<()> {
    if let Some(dir) = child.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file_name = child
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("history.txt");
    let temporary = child.with_file_name(format!(
        ".{file_name}.prepare-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    let result = (|| {
        if inherit_history && parent.is_file() {
            std::fs::copy(parent, &temporary)?;
        } else {
            std::fs::write(&temporary, b"")?;
        }
        std::fs::rename(&temporary, child)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

/// 同步子代理的 history 仅在任务执行期间存在。任务已停止后清除主文件、SQLite
/// sidecar 与跨进程 state lock；文本后端复用同一清理入口也不会产生额外副作用。
pub(in crate::ai) fn delete_subagent_history(path: &Path) -> io::Result<()> {
    let history_result = blob::delete_history_artifacts(path);
    let lock_result = sqlite::delete_session_state_lock(path);
    history_result.and(lock_result)
}

#[cfg(test)]
pub(in crate::ai) use sqlite::read_context_history_sqlite;
#[allow(unused_imports)]
pub(in crate::ai) use sqlite::read_recent_turn_window_sqlite;
#[allow(unused_imports)]
pub(in crate::ai) use sqlite::{
    append_tool_execution_outcomes_sqlite, read_recent_messages_sqlite,
    read_stale_patch_targets_sqlite, read_tool_execution_outcomes_sqlite,
    read_tool_message_ids_sqlite, write_stale_patch_targets_sqlite,
};
#[allow(unused_imports)]
pub(in crate::ai) use suspended::{
    SuspendedSessionEntry, SuspendedSessionStore, format_suspended_timestamp_label,
};
#[allow(unused_imports)]
pub(in crate::ai) use types::{COLON, MAX_HISTORY_TURNS, Message, NEWLINE, ToolExecutionOutcome};

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
    let projection_fingerprint = context_projection_fingerprint(
        history_max_chars,
        history_keep_last,
        history_summary_max_chars,
        overflow_dir.as_deref(),
    );
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

    // SQLite 会话把 canonical history 与可替换的 context snapshot 分层保存。
    // 请求只读取 snapshot + watermark 之后的原始增量；人工历史始终读取 canonical 表。
    let mut history = if blob::is_sqlite_path(history_file) {
        sqlite::read_context_history_sqlite(history_file, &projection_fingerprint)?.messages
    } else {
        build_message_arr(usize::MAX, history_file)?
    };
    // canonical 层刻意保留 raw 工具结果；请求层必须重新执行同一物理上限，
    // 防止 snapshot 水位之后的 SQLite tail 绕过 current-turn 投影。
    compress::cap_raw_tool_results_for_context(&mut history, overflow_dir.as_deref());
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

/// 快照只对生成它的投影策略有效。策略配置变化时回退到 canonical messages
/// 重建，避免继续沿用旧预算或旧压缩算法产生的有损上下文。
fn context_projection_fingerprint(
    history_max_chars: usize,
    history_keep_last: usize,
    history_summary_max_chars: usize,
    overflow_dir: Option<&Path>,
) -> String {
    const PROJECTION_VERSION: u8 = 1;
    let overflow_dir = overflow_dir
        .map(|path| path.to_string_lossy())
        .unwrap_or_default();
    format!(
        "v{PROJECTION_VERSION}|max={history_max_chars}|keep={history_keep_last}|summary={history_summary_max_chars}|assets={overflow_dir}"
    )
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

/// 原子预留 session 的下一个全局 turn 序号。
///
/// 当前 session store 统一使用 SQLite；拒绝旧文本路径，避免悄悄退回会在
/// 重启或多进程场景重复编号的进程内计数。
pub(in crate::ai) fn reserve_turn_index(history_file: &Path) -> io::Result<usize> {
    if !blob::is_sqlite_path(history_file) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "persistent turn sequence requires a SQLite session",
        ));
    }
    sqlite::reserve_turn_index_sqlite(history_file)
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
    let store = SessionStore::new(app.config.history_file.as_path());
    let overflow_dir = store.session_assets_dir(&app.session_id);
    let projection_fingerprint = context_projection_fingerprint(
        app.config.history_max_chars,
        app.config.history_keep_last,
        app.config.history_summary_max_chars,
        Some(&overflow_dir),
    );
    let (messages, sqlite_source) = if blob::is_sqlite_path(history_file) {
        let context =
            sqlite::read_context_history_sqlite(history_file.as_path(), &projection_fingerprint)?;
        if context.snapshot_is_current {
            return Ok(());
        }
        let source = Some((context.source_message_id, context.canonical_generation));
        (context.messages, source)
    } else {
        let history = match std::fs::read_to_string(history_file) {
            Ok(history) => history,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(err) => return Err(err.into()),
        };
        (blob::parse_history_blob(&history), None)
    };
    if messages.is_empty() {
        return Ok(());
    }

    let original_chars = messages_total_chars_pub(&messages);
    let user_turns = messages
        .iter()
        .filter(|message| message.role == "user")
        .count();
    let exceeds_context_budget =
        app.config.history_max_chars > 0 && original_chars > app.config.history_max_chars;
    let exceeds_tool_evidence_budget =
        compress::compressed_tool_evidence_exceeds_inline_budget(&messages);
    let threshold = if at_boundary {
        compress::persisted_history_keep_recent_turns()
    } else {
        MAX_HISTORY_TURNS
    };
    if user_turns <= threshold && !exceeds_context_budget && !exceeds_tool_evidence_budget {
        return Ok(());
    }

    let compacted = if exceeds_context_budget || exceeds_tool_evidence_budget {
        // 与下一轮 `build_context_history` 使用完全相同的压缩策略，并把结果写回
        // context snapshot。原始消息只存在 canonical 层，不会被压缩覆盖。
        compress::compress_messages_for_context(
            messages.clone(),
            app.config.history_max_chars,
            app.config.history_keep_last,
            app.config.history_summary_max_chars,
            Some(overflow_dir),
        )
    } else if at_boundary {
        compress::compact_persisted_history_at_boundary_with_app(app, messages.clone()).await
    } else {
        compress::compact_persisted_history_with_app(app, messages.clone()).await
    };

    if let Some((source_message_id, history_generation)) = sqlite_source {
        sqlite::write_context_snapshot_sqlite(
            history_file.as_path(),
            &compacted,
            source_message_id,
            history_generation,
            &projection_fingerprint,
        )?;
    } else if compacted != messages {
        std::fs::write(
            history_file,
            blob::serialize_history_messages_for_storage(&compacted),
        )?;
    } else {
        return Ok(());
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
