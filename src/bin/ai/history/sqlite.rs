use std::{
    cell::RefCell,
    fs,
    fs::OpenOptions,
    io,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::fd::AsRawFd;

use rusqlite::{
    Connection, Error as RusqliteError, ErrorCode, OpenFlags, OptionalExtension,
    TransactionBehavior, params,
};
use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;

use crate::ai::types::ToolCall;

use super::{
    blob,
    compress::{self, COMPRESSED_TOOL_EVIDENCE_MARKER, is_summary_note_text, value_to_string},
    types::{Message, ROLE_INTERNAL_NOTE, SkillActivationEvent, ToolExecutionOutcome},
};

const STALE_PATCH_TARGETS_META_KEY: &str = "stale_patch_targets_v1";
const LLM_PRUNE_MARKS_META_KEY: &str = "llm_prune_marks_v1";
const LAST_ACTIVITY_META_KEY: &str = "last_activity_unix_ms";

static SESSION_STATE_LOCKS: LazyLock<Mutex<FxHashMap<PathBuf, Arc<Mutex<()>>>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

// 每个工作线程缓存一个最近打开的 history 连接，避免每次 read/write 都
// `Connection::open` + PRAGMA 初始化，并让 `prepare_cached` 的语句缓存
// 真正跨调用复用。
//
// 安全性：
// - `thread_local` 保证连接只在所属线程上被访问，`Connection` 不是 `Sync`
//   也因此不会被并发使用。
// - 读取路径本身不持有 per-session `Mutex`（并发 writer 仍会通过各自连接写入），
//   但每次读取都新开一个 `conn.transaction()` 并在同一调用内 commit：WAL 的
//   per-transaction 快照因此始终读到最新已提交状态，复用连接不会读到陈旧数据。
// - `(dev, ino, len)` 指纹只负责一件事：在复用前校验底层文件未被 rename/replace
//   整体替换。文件替换操作（rollback/compact/reset）会改变 inode，此时丢弃旧
//   连接、重新打开，语义与每次新开连接完全一致。
thread_local! {
    static CACHED_HISTORY_CONN: RefCell<Option<CachedHistoryConn>> = const { RefCell::new(None) };
}

struct CachedHistoryConn {
    path: PathBuf,
    fingerprint: FileFingerprint,
    conn: Connection,
}

#[derive(PartialEq, Eq)]
struct FileFingerprint {
    dev: u64,
    ino: u64,
    len: u64,
}

#[cfg(unix)]
fn file_fingerprint(metadata: &fs::Metadata) -> FileFingerprint {
    use std::os::unix::fs::MetadataExt;
    FileFingerprint {
        dev: metadata.dev(),
        ino: metadata.ino(),
        len: metadata.len(),
    }
}

#[cfg(not(unix))]
fn file_fingerprint(metadata: &fs::Metadata) -> FileFingerprint {
    FileFingerprint {
        dev: 0,
        ino: 0,
        len: metadata.len(),
    }
}

/// history_revision 的进程内缓存：指纹 (主库 len/mtime, WAL sidecar len/mtime) 失效。
///
/// `read_history_revision` 每次调用都 `Connection::open` + SQL 查询（每轮上下文
/// 缓存校验 + session 列表刷新会调用多次），而 revision 只在消息写入时才变化。
/// 这里用文件元数据指纹缓存结果：本进程/外部进程写入都会改变主库或 WAL sidecar
/// 的 len/mtime，指纹失配即重查，语义与无缓存一致。
///
/// **WAL 关键**：history DB 启用 WAL 模式，提交只写 `-wal` sidecar，主库
/// len/mtime 在 checkpoint 前可以完全不变。若指纹仅覆盖主库，有存活连接时
/// 缓存会持续返回旧 revision，破坏 `history_revision` 作为跨连接信号的语义。
/// 因此指纹同时覆盖主库和 `-wal` sidecar，任一变化即失效。
static HISTORY_REVISION_CACHE: LazyLock<
    Mutex<FxHashMap<PathBuf, (((u64, Option<SystemTime>), (u64, Option<SystemTime>)), i64)>>,
> = LazyLock::new(|| Mutex::new(FxHashMap::default()));

fn history_file_fingerprint(path: &Path) -> ((u64, Option<SystemTime>), (u64, Option<SystemTime>)) {
    let main = std::fs::metadata(path)
        .map(|m| (m.len(), m.modified().ok()))
        .unwrap_or((0, None));
    // WAL 模式下提交先落 `-wal` sidecar，主库 mtime 在 checkpoint 前不变；
    // 将 sidecar 元数据纳入指纹，确保未 checkpoint 的写入也能触发缓存失效。
    let wal_path = PathBuf::from(format!("{}-wal", path.display()));
    let wal = std::fs::metadata(&wal_path)
        .map(|m| (m.len(), m.modified().ok()))
        .unwrap_or((0, None));
    (main, wal)
}

fn session_state_lock_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("history.sqlite");
    path.with_file_name(format!(".{file_name}.state.lock"))
}

pub(super) fn delete_session_state_lock(path: &Path) -> io::Result<()> {
    match fs::remove_file(session_state_lock_path(path)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// 回收 [`SESSION_STATE_LOCKS`] 中某路径的 per-path 锁条目。
///
/// 子代理历史文件按 pid / task_id 唯一，若只删磁盘 `.state.lock` 文件而不回收
/// 进程内 map 条目，长跑主会话派生大量子代理后 map 会无界增长直到进程退出。
/// 这里在子代理生命周期结束（`delete_subagent_history`）时清理条目；仅当
/// `Arc::strong_count == 1`（无其他线程正持有该锁的克隆）时移除，避免摘除一把
/// 正在被 `with_session_state_lock` 使用的锁，从而破坏该路径的互斥语义。
pub(super) fn remove_session_state_lock_entry(path: &Path) {
    let mut locks = SESSION_STATE_LOCKS.lock().unwrap_or_else(|poison| {
        warn_session_lock_poison(path, "history lock registry");
        poison.into_inner()
    });
    if let Some(existing) = locks.get(path)
        && Arc::strong_count(existing) == 1
    {
        locks.remove(path);
    }
}

/// 将会替换整个 live SQLite 文件的 rollback 与常规 canonical writer 串行化。
/// 进程内 mutex 避免同进程线程间 `flock` 语义差异，文件锁负责跨进程互斥。
pub(super) fn with_session_state_lock<T>(
    path: &Path,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    let lock = {
        let mut locks = SESSION_STATE_LOCKS.lock().unwrap_or_else(|poison| {
            warn_session_lock_poison(path, "history lock registry");
            poison.into_inner()
        });
        locks
            .entry(path.to_path_buf())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    };
    let _guard = lock.lock().unwrap_or_else(|poison| {
        warn_session_lock_poison(path, "per-session history state lock");
        poison.into_inner()
    });

    let lock_path = session_state_lock_path(path);
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(lock_path)?;
    #[cfg(unix)]
    unsafe {
        if libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    let result = operation();
    #[cfg(unix)]
    unsafe {
        let _ = libc::flock(lock_file.as_raw_fd(), libc::LOCK_UN);
    }
    result
}

/// 锁中毒意味着此前持锁线程在持有进程内 history 锁时 panic。
/// 磁盘上的跨进程 `flock` 与 SQLite 事务仍是真实的安全边界，因此这里
/// 继续恢复执行（保持原有语义），但不再静默：打印一次性告警，便于排查
/// 导致 panic 的上游缺陷。
fn warn_session_lock_poison(path: &Path, which: &str) {
    eprintln!(
        "[Warning] {} was poisoned (a previous holder panicked). \
         Recovering, but the earlier panic should be investigated. path={}",
        which,
        path.display()
    );
}

fn session_state_lock_timeout(path: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::WouldBlock,
        format!("timed out acquiring history state lock: {}", path.display()),
    )
}

fn wait_for_session_state_lock(path: &Path, deadline: Instant) -> io::Result<()> {
    if Instant::now() >= deadline {
        return Err(session_state_lock_timeout(path));
    }
    std::thread::sleep(
        deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_millis(10)),
    );
    Ok(())
}

/// 与 [`with_session_state_lock`] 相同，但把进程内 mutex 与跨进程 flock
/// 一并限制在调用方给定的截止时间内，供短超时重试路径使用。
fn with_session_state_lock_until<T>(
    path: &Path,
    deadline: Instant,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    let lock = loop {
        match SESSION_STATE_LOCKS.try_lock() {
            Ok(mut locks) => {
                break locks
                    .entry(path.to_path_buf())
                    .or_insert_with(|| Arc::new(Mutex::new(())))
                    .clone();
            }
            Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                warn_session_lock_poison(path, "history lock registry");
                let mut locks = poisoned.into_inner();
                break locks
                    .entry(path.to_path_buf())
                    .or_insert_with(|| Arc::new(Mutex::new(())))
                    .clone();
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                wait_for_session_state_lock(path, deadline)?;
            }
        }
    };
    let _guard = loop {
        match lock.try_lock() {
            Ok(guard) => break guard,
            Err(std::sync::TryLockError::Poisoned(poisoned)) => {
                warn_session_lock_poison(path, "per-session history state lock");
                break poisoned.into_inner();
            }
            Err(std::sync::TryLockError::WouldBlock) => {
                wait_for_session_state_lock(path, deadline)?;
            }
        }
    };

    let lock_path = session_state_lock_path(path);
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(lock_path)?;
    #[cfg(unix)]
    loop {
        // SAFETY: `lock_file` 在整个临界区内保持打开，drop 时内核自动释放 flock。
        let status = unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if status == 0 {
            break;
        }
        let error = io::Error::last_os_error();
        if !matches!(
            error.raw_os_error(),
            Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN
        ) {
            return Err(error);
        }
        wait_for_session_state_lock(path, deadline)?;
    }

    operation()
}

/// 模型实际请求所消费的可重建上下文。`messages` 永远是唯一的原始会话记录；
/// 这里的消息只是一次压缩快照，加上 `source_message_id` 之后的新原始消息即可
/// 重建当前上下文。`canonical_generation` 用于拒绝并发 rewind/clear 后的陈旧快照。
pub(in crate::ai) struct ContextHistory {
    pub(in crate::ai) messages: Vec<Message>,
    pub(in crate::ai) source_message_id: i64,
    pub(in crate::ai) canonical_generation: i64,
    pub(in crate::ai) snapshot_is_current: bool,
}

pub(in crate::ai) struct RecentTurnWindow {
    pub(in crate::ai) messages: Vec<Message>,
    pub(in crate::ai) start_message_id: Option<i64>,
    pub(in crate::ai) has_older_messages: bool,
}

/// `/ss` 列表展示所需的轻量元数据。
pub(in crate::ai) struct SessionListMetadata {
    pub(in crate::ai) first_user_prompt: Option<String>,
    pub(in crate::ai) session_title: Option<String>,
    pub(in crate::ai) last_activity_unix_ms: Option<i64>,
    pub(in crate::ai) history_revision: i64,
}

fn open_history_db(path: &Path) -> Result<Connection, io::Error> {
    open_history_db_with_busy_timeout(path, Duration::from_secs(5))
}

fn open_history_db_with_busy_timeout(
    path: &Path,
    busy_timeout: Duration,
) -> Result<Connection, io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    // 本函数是同步的，且会被 async 调用者（compaction 的读/写路径）经 `?` 复用。
    // 因此这里绝不做同步 sleep 重试 —— 那会阻塞 tokio worker。瞬时 I/O 失败
    // （如并发首开 WAL 时的 SQLITE_IOERR_FSTAT）统一以 `WouldBlock` 返回，交给
    // async 调用点的非阻塞重试循环（`tokio::time::sleep`）处理；纯同步调用点则
    // 依赖 SQLite 自身的 busy_timeout。非瞬时错误按原语义直接上抛。
    try_open_history_db(path, busy_timeout)
}

fn try_open_history_db(path: &Path, busy_timeout: Duration) -> Result<Connection, io::Error> {
    let conn = fresh_connection(path, busy_timeout)?;
    Ok(conn)
}

fn fresh_connection(path: &Path, busy_timeout: Duration) -> Result<Connection, io::Error> {
    let conn = Connection::open(path).map_err(|error| sqlite_error(path, "open", error))?;
    conn.busy_timeout(busy_timeout)
        .map_err(|error| sqlite_error(path, "set busy_timeout", error))?;
    conn.pragma_update(None, "journal_mode", "WAL")
        .map_err(|error| sqlite_error(path, "set journal_mode=WAL", error))?;
    Ok(conn)
}

/// 在缓存的 history 连接上执行一个只读操作。连接按线程缓存，并在底层文件
/// inode/大小变化（rollback/compact/reset 等替换操作）时自动重连，语义与
/// 每次新开连接一致，但省掉了每轮 turn 的 open + PRAGMA 与语句重编译开销。
///
/// 注意：返回的借用数据不能逃逸出闭包（`Connection` 不是 `Sync`，且连接
/// 归 thread_local 所有）。仅用于热路径只读查询。
fn with_cached_read_conn<T>(
    path: &Path,
    busy_timeout: Duration,
    op: impl FnOnce(&mut Connection) -> io::Result<T>,
) -> Result<T, io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let metadata = fs::metadata(path)
        .map_err(|error| io::Error::other(format!("metadata {}: {error}", path.display())))?;
    let fingerprint = file_fingerprint(&metadata);
    let path_owned = path.to_path_buf();

    CACHED_HISTORY_CONN.with(|slot| {
        let mut slot = slot.borrow_mut();
        let need_reconnect = match slot.as_ref() {
            Some(cached) => cached.path != path_owned || cached.fingerprint != fingerprint,
            None => true,
        };
        if need_reconnect {
            let conn = fresh_connection(&path_owned, busy_timeout)?;
            *slot = Some(CachedHistoryConn {
                path: path_owned,
                fingerprint,
                conn,
            });
        }
        let cached = slot.as_mut().expect("cached connection just initialized");
        op(&mut cached.conn)
    })
}

fn sqlite_error_kind(error: &RusqliteError) -> io::ErrorKind {
    match error {
        RusqliteError::SqliteFailure(inner, _)
            if matches!(
                inner.code,
                ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked | ErrorCode::SystemIoFailure
            ) =>
        {
            io::ErrorKind::WouldBlock
        }
        _ => io::ErrorKind::Other,
    }
}

fn sqlite_error(path: &Path, operation: &str, error: RusqliteError) -> io::Error {
    let kind = sqlite_error_kind(&error);
    let detail = match &error {
        RusqliteError::SqliteFailure(inner, message) => {
            let message = message
                .as_deref()
                .map(|message| format!("; message={message}"))
                .unwrap_or_default();
            format!(
                "SQLite {operation} failed for {}: {error}; code={:?}; extended_code={}{}",
                path.display(),
                inner.code,
                inner.extended_code,
                message
            )
        }
        _ => format!("SQLite {operation} failed for {}: {error}", path.display()),
    };
    io::Error::new(kind, detail)
}

/// 会话列表只读元数据，不能因枚举而创建目录、初始化数据库或切换 journal mode。
fn open_history_db_read_only(path: &Path) -> Result<Connection, io::Error> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|error| io::Error::other(error.to_string()))?;
    conn.busy_timeout(Duration::from_secs(5))
        .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(conn)
}

fn init_history_schema(conn: &Connection) -> Result<(), io::Error> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_calls TEXT,
            tool_call_id TEXT,
            reasoning_content TEXT,
            source_model TEXT,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
        CREATE TABLE IF NOT EXISTS context_messages (
            position INTEGER PRIMARY KEY,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            tool_calls TEXT,
            tool_call_id TEXT,
            reasoning_content TEXT
        );
        CREATE TABLE IF NOT EXISTS context_snapshot (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            source_message_id INTEGER NOT NULL,
            source_generation INTEGER NOT NULL,
            projection_fingerprint TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        CREATE TABLE IF NOT EXISTS tool_execution_outcomes (
            tool_call_id TEXT PRIMARY KEY,
            execution_signature TEXT NOT NULL,
            succeeded INTEGER NOT NULL,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        CREATE TABLE IF NOT EXISTS skill_activation_events (
            id INTEGER PRIMARY KEY,
            requested_skill TEXT NOT NULL,
            injected_skill TEXT,
            source TEXT NOT NULL,
            outcome TEXT NOT NULL,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );",
    )
    .map_err(|error| io::Error::new(sqlite_error_kind(&error), error.to_string()))?;
    add_column_if_missing(conn, "messages", "tool_calls", "TEXT")?;
    add_column_if_missing(conn, "messages", "tool_call_id", "TEXT")?;
    add_column_if_missing(conn, "messages", "reasoning_content", "TEXT")?;
    add_column_if_missing(conn, "messages", "source_model", "TEXT")?;
    // 旧快照无法证明符合当前投影策略，空 fingerprint 会让读取路径安全地忽略它。
    add_column_if_missing(
        conn,
        "context_snapshot",
        "projection_fingerprint",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    Ok(())
}

fn add_column_if_missing(
    conn: &Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> Result<(), io::Error> {
    let sql = format!("ALTER TABLE {table} ADD COLUMN {column} {definition}");
    match conn.execute(&sql, []) {
        Ok(_) => Ok(()),
        Err(err) if err.to_string().contains("duplicate column name") => Ok(()),
        Err(error) => Err(io::Error::new(sqlite_error_kind(&error), error.to_string())),
    }
}

/// 读取 history DB 的写入版本号（存于 meta 表 key='history_revision'）。
/// 每次消息写入/删除/替换都会在同一事务内 `bump_history_revision` 递增该值，
/// 因此它是一个**跨连接**单调递增的全局信号，可靠地反映"库内容是否变化"。
///
/// 不能用 `PRAGMA data_version` 代替：它是**连接局部**的比较值——每个新开的
/// `Connection` 只把它当作"自本连接打开以来是否被其他连接改过"的基准，新连接
/// 读到的初值不随外部写入而变（实测新连接恒返回 2），因此无法作为跨连接缓存
/// 失效依据。缺失（老库尚未写入过 revision）时返回 0，与"从未修改"一致。
pub(in crate::ai) fn read_history_revision(path: &Path) -> Option<i64> {
    // 进程内缓存：指纹 (主库 len/mtime, WAL sidecar len/mtime) 未变即复用上次
    // 结果，避免每次调用都新开连接 + SQL 查询（每轮上下文缓存校验 / session
    // 列表刷新高频调用）。
    let fingerprint = history_file_fingerprint(path);
    if let Ok(cache) = HISTORY_REVISION_CACHE.lock() {
        if let Some((cached_fp, rev)) = cache.get(path) {
            if *cached_fp == fingerprint {
                return Some(*rev);
            }
        }
    }
    let conn = Connection::open(path).ok()?;
    // meta 表可能尚未创建（全新库）或尚未写入过 revision：两种情况都视为 0
    // （"从未修改"），保证返回值稳定可比。只有连接本身打不开才返回 None。
    let value: Option<i64> = conn
        .query_row(
            "SELECT CAST(value AS INTEGER) FROM meta WHERE key='history_revision' LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .unwrap_or(None);
    let revision = value.unwrap_or(0);
    if let Ok(mut cache) = HISTORY_REVISION_CACHE.lock() {
        cache.insert(path.to_path_buf(), (fingerprint, revision));
    }
    Some(revision)
}

/// 清理指定 history 路径的 revision 缓存条目。
/// 在 history 文件删除或改名时调用，避免缓存随 session/sub-agent 唯一路径
/// 无限增长。
pub(in crate::ai) fn remove_history_revision_cache_entry(path: &Path) {
    if let Ok(mut cache) = HISTORY_REVISION_CACHE.lock() {
        cache.remove(path);
    }
}

#[cfg(test)]
pub(in crate::ai) fn history_revision_cache_contains(path: &Path) -> bool {
    HISTORY_REVISION_CACHE
        .lock()
        .is_ok_and(|cache| cache.contains_key(path))
}

/// 记录最近一次消息写入时间；同一毫秒内连续写入也保持单调递增。
fn touch_session_activity(conn: &Connection) -> io::Result<()> {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64;
    let previous = read_i64_meta_from_conn(conn, LAST_ACTIVITY_META_KEY).unwrap_or(0);
    let next = now_ms.max(previous.saturating_add(1));
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value, created_at)
         VALUES (?1, ?2, unixepoch())",
        params![LAST_ACTIVITY_META_KEY, next.to_string()],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(())
}

/// 在写事务内递增历史版本，并同步刷新逻辑活动时间。
fn bump_history_revision(conn: &Connection) -> io::Result<()> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('history_revision', '1')
         ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1",
        [],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    touch_session_activity(conn)
}

fn history_generation(conn: &Connection) -> io::Result<i64> {
    conn.query_row(
        "SELECT COALESCE(
            (SELECT CAST(value AS INTEGER) FROM meta WHERE key='history_generation'),
            0
         )",
        [],
        |row| row.get(0),
    )
    .map_err(|error| io::Error::other(error.to_string()))
}

/// 任何会重写 canonical messages 的操作都必须使派生快照失效。
fn invalidate_context_snapshot(conn: &Connection) -> io::Result<()> {
    conn.execute("DELETE FROM context_messages", [])
        .map_err(|error| io::Error::other(error.to_string()))?;
    conn.execute("DELETE FROM context_snapshot", [])
        .map_err(|error| io::Error::other(error.to_string()))?;
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('history_generation', '1')
         ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1",
        [],
    )
    .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(())
}

/// 原子预留一个 session 全局 turn 序号。
///
/// 序号写入 SQLite 元数据而不是保存在进程内存中，因此重启和多个进程并发恢复
/// 同一 session 时也不会重复。旧 session 首次分配时从已持久化的 user turn 数
/// 继续编号，兼容此前 `turn_index` 的语义。
pub(in crate::ai) fn reserve_turn_index_sqlite(path: &Path) -> io::Result<usize> {
    with_session_state_lock(path, || reserve_turn_index_sqlite_unlocked(path))
}

fn reserve_turn_index_sqlite_unlocked(path: &Path) -> io::Result<usize> {
    let mut conn = open_history_db(path)?;
    init_history_schema(&conn)?;
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|e| io::Error::other(e.to_string()))?;

    let current = tx
        .query_row("SELECT value FROM meta WHERE key = 'turn_seq'", [], |row| {
            row.get::<_, String>(0)
        })
        .optional()
        .map_err(|e| io::Error::other(e.to_string()))?
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or_else(|| {
            tx.query_row(
                "SELECT COUNT(*) FROM messages WHERE role = 'user'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0)
        })
        .max(0);
    let next = current
        .checked_add(1)
        .ok_or_else(|| io::Error::other("turn sequence overflow"))?;
    tx.execute(
        "INSERT INTO meta(key, value) VALUES('turn_seq', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![next.to_string()],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    touch_session_activity(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))?;
    usize::try_from(current).map_err(io::Error::other)
}

/// 读取回滚前 live 库的三个单调计数器：`history_generation`、
/// `history_revision`、`turn_seq`。回滚会用 `backup_sqlite` 把 checkpoint 库
/// 整库覆盖到 live 路径，这会把这三个计数器一起还原成 checkpoint 时刻的旧值，
/// 破坏"跨回滚单调递增"不变量。本函数在覆盖前读取 live 值，供
/// `rebase_metadata_after_rollback` 在覆盖后把它们抬高到 live 之上。
/// 库不存在或 meta 行缺失时对应返回 0，与"从未修改"基准一致。
pub(in crate::ai) struct LiveRollbackMetadata {
    pub(in crate::ai) generation: i64,
    pub(in crate::ai) revision: i64,
    pub(in crate::ai) turn_seq: i64,
}

impl LiveRollbackMetadata {
    fn zero() -> Self {
        Self {
            generation: 0,
            revision: 0,
            turn_seq: 0,
        }
    }
}

pub(in crate::ai) fn read_live_rollback_metadata(path: &Path) -> io::Result<LiveRollbackMetadata> {
    // `Connection::open` 会创建文件，这里只需读取已存在 live 库，缺失时按 0 基准返回。
    if !path.exists() {
        return Ok(LiveRollbackMetadata::zero());
    }
    let conn = Connection::open(path).map_err(|e| io::Error::other(e.to_string()))?;
    conn.busy_timeout(Duration::from_secs(5))
        .map_err(|e| io::Error::other(e.to_string()))?;
    let meta_exists = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'",
            [],
            |_| Ok(()),
        )
        .optional()
        .map_err(|e| io::Error::other(e.to_string()))?
        .is_some();
    if !meta_exists {
        return Ok(LiveRollbackMetadata::zero());
    }
    let read = |key: &str| -> io::Result<i64> {
        let value = conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1 LIMIT 1",
                params![key],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(|e| io::Error::other(e.to_string()))?;
        value.map_or(Ok(0), |value| {
            value.parse::<i64>().map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid SQLite meta value for {key}: {error}"),
                )
            })
        })
    };
    Ok(LiveRollbackMetadata {
        generation: read("history_generation")?,
        revision: read("history_revision")?,
        turn_seq: read("turn_seq")?,
    })
}

/// 在 `backup_sqlite` 用 checkpoint 覆盖 live 库之后调用：清空派生快照
/// （canonical 已回退到 checkpoint，旧快照不再匹配），并把三个单调计数器抬高到
/// 回滚前 live 值之上，保证：
/// - `history_generation` 严格大于 live 值 -> 任何仍持有旧 generation 的
///   stale 快照写入会被 fencing 拒绝（`write_context_snapshot_sqlite` 的
///   generation 比对返回 false）。
/// - `turn_seq` 不低于 live 值 -> 回滚后新分配的 turn 序号不会复用已用过的序号。
/// - `history_revision` 严格大于 live 值 -> 跨连接文件缓存能观测到变化并重载。
/// 全程在一个 Immediate 事务内提交，避免覆盖后留下"已回滚但计数器未抬升"的
/// 不一致窗口。
pub(in crate::ai) fn rebase_metadata_after_rollback(
    path: &Path,
    live_generation: i64,
    live_revision: i64,
    live_turn_seq: i64,
) -> io::Result<()> {
    let mut conn = open_history_db(path)?;
    init_history_schema(&conn)?;
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|e| io::Error::other(e.to_string()))?;
    // canonical 已回退，旧派生快照（可能由 checkpoint 时刻的 generation 标记）
    // 不再匹配，清空以强制下次读取从 canonical 重算。
    tx.execute("DELETE FROM context_messages", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    tx.execute("DELETE FROM context_snapshot", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    // 各计数器取 MAX(checkpoint 值, live 值) 再 +1（generation/revision 需严格
    // 递增；turn_seq 取 live 值即可，因为 live 值是"下一个待分配序号"）。
    let bump_gen = live_generation.saturating_add(1).max(0);
    let bump_rev = live_revision.saturating_add(1).max(0);
    let bump_turn = live_turn_seq.max(0);
    tx.execute(
        "INSERT INTO meta (key, value) VALUES ('history_generation', ?1)
         ON CONFLICT(key) DO UPDATE SET value = MAX(CAST(value AS INTEGER), CAST(?1 AS INTEGER))",
        params![bump_gen.to_string()],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    tx.execute(
        "INSERT INTO meta (key, value) VALUES ('turn_seq', ?1)
         ON CONFLICT(key) DO UPDATE SET value = MAX(CAST(value AS INTEGER), CAST(?1 AS INTEGER))",
        params![bump_turn.to_string()],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    tx.execute(
        "INSERT INTO meta (key, value) VALUES ('history_revision', ?1)
         ON CONFLICT(key) DO UPDATE SET value = MAX(CAST(value AS INTEGER), CAST(?1 AS INTEGER))",
        params![bump_rev.to_string()],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    touch_session_activity(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))
}

/// 在 session 状态锁内准备一个已 rebase 的临时库，再通过 Online Backup 的原子
/// rename 发布到 live 路径。崩溃发生在最终发布前时 live 库保持不变；发布后则元数据
/// 已完整抬升，不存在“库已回滚但计数器仍是旧值”的中间状态。
pub(in crate::ai) fn restore_sqlite_after_rollback(
    checkpoint: &Path,
    live_path: &Path,
) -> io::Result<()> {
    with_session_state_lock(live_path, || {
        let live = read_live_rollback_metadata(live_path)?;
        let parent = live_path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let working = parent.join(format!(".rollback-rebased-{}.sqlite", uuid::Uuid::new_v4()));

        let result = (|| {
            backup_sqlite(checkpoint, &working)?;
            rebase_metadata_after_rollback(
                &working,
                live.generation,
                live.revision,
                live.turn_seq,
            )?;
            // 第二次 backup 会把 working 的 WAL 一并物化到最终临时主库，随后以
            // 单次 rename 发布，避免只移动主文件而遗漏 rebase 事务。
            backup_sqlite(&working, live_path)
        })();
        let _ = fs::remove_file(&working);
        let _ = remove_sqlite_sidecars(&working);
        result
    })
}

/// 用 SQLite Online Backup API 创建一致快照，并以原子替换的方式写入目标。
/// 直接复制 WAL 主文件会遗漏尚未 checkpoint 的页；backup API 会从 source 的同一
/// SQLite 快照读取主库和 WAL。主库替换成功后会移除旧侧车文件，避免旧 WAL/SHM
/// 与新主库混用。
pub(in crate::ai) fn backup_sqlite(source: &Path, target: &Path) -> io::Result<()> {
    if !source.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("SQLite source does not exist: {}", source.display()),
        ));
    }
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("history.sqlite");
    let temporary = parent.join(format!(".{file_name}.backup-{}.tmp", uuid::Uuid::new_v4()));

    let result = (|| {
        let source_conn = Connection::open(source).map_err(|e| io::Error::other(e.to_string()))?;
        source_conn
            .backup(rusqlite::DatabaseName::Main, &temporary, None)
            .map_err(|e| io::Error::other(e.to_string()))?;
        drop(source_conn);

        fs::rename(&temporary, target)?;
        remove_sqlite_sidecars(target)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
        let _ = remove_sqlite_sidecars(&temporary);
    }
    result
}

/// 为子 agent 复制父会话的历史库到独立的 per-process 文件。
///
/// `inherit.history` 语义应为「子 agent 可读父会话上下文，但写入不回灌父库」。
/// 直接复用父库的 canonical 文件会让子 agent 的内部 prompt/tool trace 污染父会话，
/// 并在多个并发子 agent 间交错写入同一 session。这里用 SQLite Online Backup 做一次
/// 一致性快照拷贝，子 agent 后续只读写自己的 fork 文件，父库保持隔离。
///
/// 父会话尚无历史文件（首次会话）时发布一个全新的空库，不能复用残留 child 库。
pub(in crate::ai) fn fork_history_for_subagent(parent: &Path, child: &Path) -> io::Result<()> {
    with_session_state_lock(child, || match fs::metadata(parent) {
        Ok(_metadata) => {
            // 确保子文件父目录存在，否则 backup_sqlite 的临时文件会创建失败。
            if let Some(dir) = child.parent() {
                fs::create_dir_all(dir)?;
            }
            backup_sqlite(parent, child)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            // 父会话尚无历史文件，子 agent 从空白历史开始。
            reset_history_for_subagent_unlocked(child)
        }
        Err(error) => Err(error),
    })
}

/// 首次派发无历史继承的子代理时，发布一个全新的空库，不能复用同 pid 的残留库。
pub(in crate::ai) fn reset_history_for_subagent(child: &Path) -> io::Result<()> {
    with_session_state_lock(child, || reset_history_for_subagent_unlocked(child))
}

fn reset_history_for_subagent_unlocked(child: &Path) -> io::Result<()> {
    let temporary = child.with_extension(format!(
        "sqlite.empty-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    if let Some(parent) = temporary.parent() {
        fs::create_dir_all(parent)?;
    }
    let result = (|| {
        let conn = open_history_db(&temporary)?;
        init_history_schema(&conn)?;
        drop(conn);
        backup_sqlite(&temporary, child)
    })();
    let _ = fs::remove_file(&temporary);
    let _ = remove_sqlite_sidecars(&temporary);
    result
}

fn remove_sqlite_sidecars(path: &Path) -> io::Result<()> {
    for suffix in ["-wal", "-shm", "-journal"] {
        let sidecar = PathBuf::from(format!("{}{}", path.display(), suffix));
        match fs::remove_file(sidecar) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// 廉价查询当前 history DB 中 role='user' 的消息数。
/// 用于 boundary compact 在 hot path 上"先 count 再决定是否全量读"，
/// 避免每个 turn 收尾都把几万条消息（含大块 tool 输出）反序列化一遍。
pub(in crate::ai) fn count_user_turns_sqlite(path: &Path) -> io::Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    let conn = open_history_db(path)?;
    // schema 可能尚未创建（全新 session），messages 表不存在时直接返 0。
    let table_exists: bool = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'",
            [],
            |_| Ok(true),
        )
        .optional()
        .map_err(|e| io::Error::other(e.to_string()))?
        .unwrap_or(false);
    if !table_exists {
        return Ok(0);
    }
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(1) FROM messages WHERE role = 'user'",
            [],
            |row| row.get(0),
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(count.max(0) as usize)
}

/// 廉价统计持久化消息的有效载荷大小，用于在 user turn 数尚少、但工具输出已经
/// 很大时仍能触发历史落盘压缩。不能以 sqlite 文件大小判断：WAL/空闲页不会在
/// 替换消息后立刻回收，会导致每轮都误判为超限。
pub(in crate::ai) fn total_message_chars_sqlite(path: &Path) -> io::Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    let conn = open_history_db(path)?;
    let table_exists: bool = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'",
            [],
            |_| Ok(true),
        )
        .optional()
        .map_err(|e| io::Error::other(e.to_string()))?
        .unwrap_or(false);
    if !table_exists {
        return Ok(0);
    }
    let total: i64 = conn
        .query_row(
            "SELECT COALESCE(SUM(length(content) + COALESCE(length(tool_calls), 0) + COALESCE(length(reasoning_content), 0)), 0) FROM messages",
            [],
            |row| row.get(0),
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(total.max(0) as usize)
}

/// 廉价统计已经折叠成 internal_note 的旧工具证据体积。它有独立于全局 history
/// 预算的内联上限，避免少量 user turn 下逐条证据在达到总预算前持续累积。
pub(in crate::ai) fn compressed_tool_evidence_chars_sqlite(path: &Path) -> io::Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    let conn = open_history_db(path)?;
    let table_exists: bool = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'",
            [],
            |_| Ok(true),
        )
        .optional()
        .map_err(|e| io::Error::other(e.to_string()))?
        .unwrap_or(false);
    if !table_exists {
        return Ok(0);
    }
    let total: i64 = conn
        .query_row(
            "SELECT COALESCE(SUM(length(content)), 0)
             FROM messages
             WHERE role = ?1 AND instr(content, ?2) > 0",
            params![ROLE_INTERNAL_NOTE, COMPRESSED_TOOL_EVIDENCE_MARKER],
            |row| row.get(0),
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(total.max(0) as usize)
}

/// 持久化每个真实工具调用的结构化成败与执行签名。工具结果正文仍只保存在
/// `messages`，因此请求投影可折叠已解决错误，而人工历史仍保留原始诊断。
pub(in crate::ai) fn append_tool_execution_outcomes_sqlite(
    path: &Path,
    outcomes: &[ToolExecutionOutcome],
) -> io::Result<()> {
    if outcomes.is_empty() || !blob::is_sqlite_path(path) {
        return Ok(());
    }
    with_session_state_lock(path, || {
        let mut conn = open_history_db(path)?;
        init_history_schema(&conn)?;
        let tx = conn
            .transaction()
            .map_err(|error| io::Error::other(error.to_string()))?;
        {
            let mut statement = tx
                .prepare(
                    "INSERT INTO tool_execution_outcomes
                        (tool_call_id, execution_signature, succeeded)
                     VALUES (?1, ?2, ?3)
                     ON CONFLICT(tool_call_id) DO NOTHING",
                )
                .map_err(|error| io::Error::other(error.to_string()))?;
            for outcome in outcomes {
                statement
                    .execute(params![
                        outcome.tool_call_id,
                        outcome.execution_signature,
                        outcome.succeeded
                    ])
                    .map_err(|error| io::Error::other(error.to_string()))?;
            }
        }
        bump_history_revision(&tx)?;
        tx.commit()
            .map_err(|error| io::Error::other(error.to_string()))
    })
}

/// 读取请求投影所需的结构化工具结果。老会话没有旁路表时安全退化为空集合，
/// 不对历史正文做任何基于自然语言的成败猜测。
pub(in crate::ai) fn read_tool_execution_outcomes_sqlite(
    path: &Path,
) -> io::Result<Vec<ToolExecutionOutcome>> {
    if !blob::is_sqlite_path(path) || !path.exists() {
        return Ok(Vec::new());
    }
    let conn = open_history_db(path)?;
    let table_exists = conn
        .query_row(
            "SELECT 1 FROM sqlite_master
             WHERE type='table' AND name='tool_execution_outcomes'",
            [],
            |_| Ok(true),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?
        .unwrap_or(false);
    if !table_exists {
        return Ok(Vec::new());
    }
    let mut statement = conn
        .prepare(
            "SELECT tool_call_id, execution_signature, succeeded
             FROM tool_execution_outcomes ORDER BY created_at ASC, rowid ASC",
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
    let rows = statement
        .query_map([], |row| {
            Ok(ToolExecutionOutcome {
                tool_call_id: row.get(0)?,
                execution_signature: row.get(1)?,
                succeeded: row.get::<_, i64>(2)? != 0,
            })
        })
        .map_err(|error| io::Error::other(error.to_string()))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| io::Error::other(error.to_string()))
}

/// 持久化显式 skill 选择的实际注入结果。原始记录是诊断旁路，不会污染 canonical
/// messages；运行时可从成功记录导出有界的历史事实。
pub(in crate::ai) fn append_skill_activation_event_sqlite(
    path: &Path,
    event: &SkillActivationEvent,
) -> io::Result<()> {
    if !blob::is_sqlite_path(path) {
        return Ok(());
    }
    with_session_state_lock(path, || {
        let mut conn = open_history_db(path)?;
        init_history_schema(&conn)?;
        let tx = conn
            .transaction()
            .map_err(|error| io::Error::other(error.to_string()))?;
        tx.execute(
            "INSERT INTO skill_activation_events
                (requested_skill, injected_skill, source, outcome)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                event.requested_skill,
                event.injected_skill,
                event.source,
                event.outcome,
            ],
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
        bump_history_revision(&tx)?;
        tx.commit()
            .map_err(|error| io::Error::other(error.to_string()))
    })
}

/// 读取 session 内的显式 skill 注入审计记录。旧会话没有该旁路表时安全退化为空。
pub(in crate::ai) fn read_skill_activation_events_sqlite(
    path: &Path,
) -> io::Result<Vec<SkillActivationEvent>> {
    if !blob::is_sqlite_path(path) || !path.exists() {
        return Ok(Vec::new());
    }
    let conn = open_history_db(path)?;
    let table_exists = conn
        .query_row(
            "SELECT 1 FROM sqlite_master
             WHERE type='table' AND name='skill_activation_events'",
            [],
            |_| Ok(true),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?
        .unwrap_or(false);
    if !table_exists {
        return Ok(Vec::new());
    }
    let mut statement = conn
        .prepare(
            "SELECT requested_skill, injected_skill, source, outcome
             FROM skill_activation_events ORDER BY created_at ASC, rowid ASC",
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
    let rows = statement
        .query_map([], |row| {
            Ok(SkillActivationEvent {
                requested_skill: row.get(0)?,
                injected_skill: row.get(1)?,
                source: row.get(2)?,
                outcome: row.get(3)?,
            })
        })
        .map_err(|error| io::Error::other(error.to_string()))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| io::Error::other(error.to_string()))
}

/// 读取持久化 tool 消息使用过的关联 ID。live context 可能已裁掉较早消息，
/// 生成新 occurrence ID 时仍须避开完整历史中的这些 ID。
pub(in crate::ai) fn read_tool_message_ids_sqlite(path: &Path) -> io::Result<Vec<String>> {
    if !blob::is_sqlite_path(path) || !path.exists() {
        return Ok(Vec::new());
    }
    let conn = open_history_db(path)?;
    let mut statement = conn
        .prepare(
            "SELECT DISTINCT tool_call_id FROM messages
             WHERE role = 'tool' AND tool_call_id IS NOT NULL",
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
    let rows = statement
        .query_map([], |row| row.get(0))
        .map_err(|error| io::Error::other(error.to_string()))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| io::Error::other(error.to_string()))
}

/// 读取当前 session 的 stale-patch 账本。`None` 表示旧数据库尚未写入过该状态，
/// 调用方应从仍可见的结构化消息回放一次并写回；`Some(empty)` 则表示已知为空，
/// 不能再次扫描可能含旧失败记录的历史。
pub(in crate::ai) fn read_stale_patch_targets_sqlite(
    path: &Path,
) -> io::Result<Option<FxHashSet<PathBuf>>> {
    if !blob::is_sqlite_path(path) || !path.exists() {
        return Ok(None);
    }
    let conn = open_history_db(path)?;
    let table_exists = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'",
            [],
            |_| Ok(true),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?
        .unwrap_or(false);
    if !table_exists {
        return Ok(None);
    }
    let raw: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key=?1 LIMIT 1",
            params![STALE_PATCH_TARGETS_META_KEY],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?;
    raw.map(|raw| {
        serde_json::from_str::<Vec<PathBuf>>(&raw)
            .map(|paths| paths.into_iter().collect())
            .map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid stale patch target metadata: {error}"),
                )
            })
    })
    .transpose()
}

/// 原子替换当前 session 的 stale-patch 账本。空集合也显式写成 `[]`，用于区分
/// “已知为空”与“旧数据库尚未初始化”；该运行时元数据不改变模型历史，故不递增
/// `history_revision`。
pub(in crate::ai) fn write_stale_patch_targets_sqlite(
    path: &Path,
    targets: &FxHashSet<PathBuf>,
) -> io::Result<()> {
    if !blob::is_sqlite_path(path) {
        return Ok(());
    }
    let mut paths = targets.iter().cloned().collect::<Vec<_>>();
    paths.sort();
    let encoded =
        serde_json::to_string(&paths).map_err(|error| io::Error::other(error.to_string()))?;
    with_session_state_lock(path, || {
        let conn = open_history_db(path)?;
        init_history_schema(&conn)?;
        conn.execute(
            "INSERT INTO meta (key, value, created_at) VALUES (?1, ?2, unixepoch())
             ON CONFLICT(key) DO UPDATE SET value=excluded.value, created_at=excluded.created_at",
            params![STALE_PATCH_TARGETS_META_KEY, encoded],
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
        Ok(())
    })
}

/// 读取当前 session 的模型引导裁剪计数。缺失或非 SQLite history 安全退化为空。
pub(in crate::ai) fn read_llm_prune_marks_sqlite(path: &Path) -> io::Result<FxHashMap<String, u8>> {
    if !blob::is_sqlite_path(path) || !path.exists() {
        return Ok(FxHashMap::default());
    }
    let conn = open_history_db(path)?;
    let table_exists = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'",
            [],
            |_| Ok(true),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?
        .unwrap_or(false);
    if !table_exists {
        return Ok(FxHashMap::default());
    }
    let raw: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key=?1 LIMIT 1",
            params![LLM_PRUNE_MARKS_META_KEY],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?;
    let Some(raw) = raw else {
        return Ok(FxHashMap::default());
    };
    let entries = serde_json::from_str::<Vec<(String, u8)>>(&raw).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid LLM prune mark metadata: {error}"),
        )
    })?;
    Ok(entries
        .into_iter()
        .filter(|(id, count)| !id.trim().is_empty() && *count > 0)
        .take(1_024)
        .collect())
}

/// 原子替换当前 session 的模型引导裁剪计数。该旁路状态不改变 canonical
/// messages，因此不递增 history_revision；空表直接删除 meta，避免遗留空状态。
pub(in crate::ai) fn write_llm_prune_marks_sqlite(
    path: &Path,
    marks: &FxHashMap<String, u8>,
) -> io::Result<()> {
    if !blob::is_sqlite_path(path) {
        return Ok(());
    }
    let mut entries = marks
        .iter()
        .filter(|(id, count)| !id.trim().is_empty() && **count > 0)
        .map(|(id, count)| (id.clone(), *count))
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let encoded =
        serde_json::to_string(&entries).map_err(|error| io::Error::other(error.to_string()))?;
    with_session_state_lock(path, || {
        let conn = open_history_db(path)?;
        init_history_schema(&conn)?;
        if entries.is_empty() {
            conn.execute(
                "DELETE FROM meta WHERE key=?1",
                params![LLM_PRUNE_MARKS_META_KEY],
            )
            .map_err(|error| io::Error::other(error.to_string()))?;
        } else {
            conn.execute(
                "INSERT INTO meta (key, value, created_at) VALUES (?1, ?2, unixepoch())
                 ON CONFLICT(key) DO UPDATE SET value=excluded.value, created_at=excluded.created_at",
                params![LLM_PRUNE_MARKS_META_KEY, encoded],
            )
            .map_err(|error| io::Error::other(error.to_string()))?;
        }
        Ok(())
    })
}

fn clear_llm_prune_marks_meta(conn: &Connection) -> io::Result<()> {
    conn.execute(
        "DELETE FROM meta WHERE key=?1",
        params![LLM_PRUNE_MARKS_META_KEY],
    )
    .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(())
}

/// outcome 只属于同一 `tool_call_id` 的 tool 消息。历史替换、压缩或分支截断后
/// 立即清掉失去消息所有者的旁路记录，避免已删除 occurrence 的状态污染保留历史。
fn prune_orphan_tool_execution_outcomes(conn: &Connection) -> io::Result<()> {
    conn.execute(
        "DELETE FROM tool_execution_outcomes
         WHERE tool_call_id NOT IN (
             SELECT DISTINCT tool_call_id FROM messages
             WHERE role = 'tool' AND tool_call_id IS NOT NULL
         )",
        [],
    )
    .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(())
}

/// 旧历史可能在 occurrence ID 修复前复用过 `tool_call_id`。一旦后续替换或
/// 截断只保留其中一条，仅按当前消息计数就无法知道 outcome 原本属于哪一次，
/// 因此必须在改变消息集合前永久丢弃这些歧义旁路状态。
fn drop_ambiguous_tool_execution_outcomes(conn: &Connection) -> io::Result<()> {
    conn.execute(
        "DELETE FROM tool_execution_outcomes
         WHERE tool_call_id IN (
             SELECT tool_call_id FROM messages
             WHERE role = 'tool' AND tool_call_id IS NOT NULL
             GROUP BY tool_call_id HAVING COUNT(1) > 1
         )",
        [],
    )
    .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(())
}

pub(in crate::ai) fn append_history_sqlite(path: &Path, entries: Vec<Message>) -> io::Result<()> {
    append_history_sqlite_for_model(path, entries, None)
}

/// 只向 canonical history 追加原始消息。模型来源作为旁路元数据保存，绝不改写
/// `Message` 本身；后续仅在构造可重建 context view 时生成协议专属投影。
pub(in crate::ai) fn append_history_sqlite_for_model(
    path: &Path,
    entries: Vec<Message>,
    source_model: Option<&str>,
) -> io::Result<()> {
    with_session_state_lock(path, || {
        append_history_sqlite_for_model_unlocked(path, entries, source_model)
    })
}

fn append_history_sqlite_for_model_unlocked(
    path: &Path,
    entries: Vec<Message>,
    source_model: Option<&str>,
) -> io::Result<()> {
    let mut conn = open_history_db(path)?;
    init_history_schema(&conn)?;
    if entries.is_empty() {
        return Ok(());
    }
    let first_user_in_blob = entries
        .iter()
        .find(|message| message.role == "user")
        .map(|message| value_to_string(&message.content));
    let tx = conn
        .transaction()
        .map_err(|e| io::Error::other(e.to_string()))?;
    {
        let existing_first: Option<String> = tx
            .query_row(
                "SELECT value FROM meta WHERE key='first_user_prompt' LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()
            .unwrap_or(None);
        if existing_first.is_none() {
            let first_existing_user: Option<String> = tx
                .query_row(
                    "SELECT content FROM messages WHERE role='user' ORDER BY id ASC LIMIT 1",
                    [],
                    |row| row.get(0),
                )
                .optional()
                .unwrap_or(None);
            let first_user_prompt = first_existing_user.or(first_user_in_blob.clone());
            if let Some(v) = first_user_prompt.as_deref() {
                let _ = tx.execute(
                    "INSERT OR IGNORE INTO meta (key, value) VALUES ('first_user_prompt', ?1)",
                    params![v],
                );
            }
        }
        insert_messages(&tx, entries, source_model)?;
    }
    bump_history_revision(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))
}

pub(in crate::ai) fn replace_all_messages_sqlite(
    path: &Path,
    messages: &[Message],
) -> io::Result<()> {
    with_session_state_lock(path, || {
        replace_all_messages_sqlite_unlocked(path, messages)
    })
}

/// 唤醒笔记去重（方案1）：同一进程、同一批 task_ids 的 TASK_WAIT_TIMEOUT "仍在等待"
/// 唤醒笔记只保留最新一条。调用方在准备追加一条内省笔记时调用本函数：
/// 删除历史尾部 `WAKE_NOTE_DEDUP_SCAN` 条消息内所有与待追加笔记身份相同的旧等待笔记
/// （由调用方随后把最新一条追加到尾部）；非"仍在等待"唤醒笔记或未命中时返回 `Ok(false)`。
pub(in crate::ai) fn coalesce_repeated_wait_wake_notes_sqlite(
    path: &Path,
    note: &Message,
) -> io::Result<bool> {
    with_session_state_lock(path, || {
        coalesce_repeated_wait_wake_notes_sqlite_unlocked(path, note)
    })
}

fn coalesce_repeated_wait_wake_notes_sqlite_unlocked(
    path: &Path,
    note: &Message,
) -> io::Result<bool> {
    // fast path：非"仍在等待"唤醒笔记时不做任何 IO
    if note.role != super::types::ROLE_INTERNAL_NOTE {
        return Ok(false);
    }
    let Some(text) = note.content.as_str() else {
        return Ok(false);
    };
    let Some(identity) = super::types::parse_still_waiting_wake_identity(text) else {
        return Ok(false);
    };

    let mut conn = open_history_db(path)?;
    init_history_schema(&conn)?;
    let mut stmt = conn
        .prepare(
            // 与 blob 后端语义一致：窗口是历史尾部 WAKE_NOTE_DEDUP_SCAN 条消息（不限角色），
            // 再对其中 internal_note 行做身份匹配 —— LIMIT 先于角色过滤生效。
            "SELECT id, content
             FROM (SELECT id, content, role FROM messages ORDER BY id DESC LIMIT ?1)
             WHERE role = ?2
             ORDER BY id ASC",
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
    let rows = stmt
        .query_map(
            params![
                super::types::WAKE_NOTE_DEDUP_SCAN as i64,
                super::types::ROLE_INTERNAL_NOTE
            ],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
    let mut to_delete = Vec::<i64>::new();
    for row in rows {
        let (id, content_json) = row.map_err(|e| io::Error::other(e.to_string()))?;
        let content = decode_message_content(&content_json);
        let Some(content_text) = content.as_str() else {
            continue;
        };
        if super::types::parse_still_waiting_wake_identity(content_text).as_ref()
            == Some(&identity)
        {
            to_delete.push(id);
        }
    }
    if to_delete.is_empty() {
        return Ok(false);
    }

    drop(stmt);
    let tx = conn
        .transaction()
        .map_err(|e| io::Error::other(e.to_string()))?;
    for id in &to_delete {
        tx.execute("DELETE FROM messages WHERE id=?1", params![id])
            .map_err(|e| io::Error::other(e.to_string()))?;
    }
    bump_history_revision(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))?;
    Ok(true)
}

fn replace_all_messages_sqlite_unlocked(path: &Path, messages: &[Message]) -> io::Result<()> {
    let mut conn = open_history_db(path)?;
    init_history_schema(&conn)?;
    let tx = conn
        .transaction()
        .map_err(|e| io::Error::other(e.to_string()))?;
    drop_ambiguous_tool_execution_outcomes(&tx)?;
    tx.execute("DELETE FROM messages", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    invalidate_context_snapshot(&tx)?;
    tx.execute("DELETE FROM meta WHERE key='first_user_prompt'", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    insert_messages(&tx, messages.to_vec(), None)?;
    prune_orphan_tool_execution_outcomes(&tx)?;
    refresh_first_user_prompt_meta(&tx, messages)?;
    bump_history_revision(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))
}

fn insert_messages(
    conn: &Connection,
    messages: Vec<Message>,
    source_model: Option<&str>,
) -> io::Result<()> {
    let mut stmt = conn
        .prepare(
            "INSERT INTO messages
                (role, content, tool_calls, tool_call_id, reasoning_content, source_model)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
    for message in messages {
        let content =
            serde_json::to_string(&message.content).map_err(|e| io::Error::other(e.to_string()))?;
        let tool_calls = message
            .tool_calls
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|e| io::Error::other(e.to_string()))?;
        stmt.execute(params![
            message.role,
            content,
            tool_calls,
            message.tool_call_id,
            message.reasoning_content,
            source_model,
        ])
        .map_err(|e| io::Error::other(e.to_string()))?;
    }
    Ok(())
}

fn refresh_first_user_prompt_meta(conn: &Connection, messages: &[Message]) -> io::Result<()> {
    let Some(first_user_prompt) = messages
        .iter()
        .find(|message| message.role == "user")
        .map(|message| value_to_string(&message.content))
    else {
        return Ok(());
    };
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('first_user_prompt', ?1)",
        params![first_user_prompt],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(())
}

fn read_messages_with_sql(
    conn: &Connection,
    sql: &str,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map([], |row| {
        let role: String = row.get(0)?;
        let content: String = row.get(1)?;
        let tool_calls: Option<String> = row.get(2)?;
        let tool_call_id: Option<String> = row.get(3)?;
        let reasoning_content: Option<String> = row.get(4)?;
        Ok((role, content, tool_calls, tool_call_id, reasoning_content))
    })?;
    let mut messages = Vec::new();
    for row in rows {
        let (role, content, tool_calls, tool_call_id, reasoning_content) = row?;
        messages.push(Message {
            role,
            content: decode_message_content(&content),
            tool_calls: decode_tool_calls(tool_calls.as_deref()),
            tool_call_id,
            reasoning_content,
        });
    }
    Ok(messages)
}

fn read_projected_canonical_messages_after_id(
    conn: &Connection,
    after_id: i64,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let mut stmt = conn.prepare(
        "SELECT role, content, tool_calls, tool_call_id, reasoning_content, source_model
         FROM messages
         WHERE id > ?1
         ORDER BY id ASC",
    )?;
    let rows = stmt.query_map(params![after_id], |row| {
        let role: String = row.get(0)?;
        let content: String = row.get(1)?;
        let tool_calls: Option<String> = row.get(2)?;
        let tool_call_id: Option<String> = row.get(3)?;
        let reasoning_content: Option<String> = row.get(4)?;
        let source_model: Option<String> = row.get(5)?;
        Ok((
            Message {
                role,
                content: decode_message_content(&content),
                tool_calls: decode_tool_calls(tool_calls.as_deref()),
                tool_call_id,
                reasoning_content,
            },
            source_model,
        ))
    })?;

    let mut messages = Vec::new();
    for row in rows {
        let (message, source_model) = row?;
        messages.push(match source_model.as_deref() {
            Some(model) => {
                compress::sanitize_message_for_persisted_history_for_model(model, &message)
            }
            None => compress::sanitize_message_for_persisted_history(&message),
        });
    }
    Ok(messages)
}

/// 返回模型上下文层：最近一次可替换压缩快照，加上快照水位之后的原始消息投影。
/// 读取在同一个 SQLite 快照事务中完成，因此 `source_message_id` 精确描述返回值已经
/// 消费到的 canonical 水位；并发追加会自然成为下一次读取的 tail，不会被快照吞掉。
pub(in crate::ai) fn read_context_history_sqlite(
    path: &Path,
    projection_fingerprint: &str,
) -> io::Result<ContextHistory> {
    with_cached_read_conn(path, Duration::from_secs(2), |conn| {
        read_context_history_on_conn(conn, projection_fingerprint)
    })
}

fn read_context_history_on_conn(
    conn: &mut Connection,
    projection_fingerprint: &str,
) -> io::Result<ContextHistory> {
    init_history_schema(conn)?;
    let tx = conn
        .transaction()
        .map_err(|error| io::Error::other(error.to_string()))?;
    let canonical_generation = history_generation(&tx)?;
    let latest_message_id = tx
        .query_row("SELECT COALESCE(MAX(id), 0) FROM messages", [], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(|error| io::Error::other(error.to_string()))?;
    let snapshot = tx
        .query_row(
            "SELECT source_message_id, source_generation, projection_fingerprint
             FROM context_snapshot WHERE singleton = 1",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?
        .filter(|(_, generation, fingerprint)| {
            *generation == canonical_generation && fingerprint == projection_fingerprint
        });

    let (mut messages, after_id, has_snapshot) = if let Some((source_message_id, _, _)) = snapshot {
        let messages = read_messages_with_sql(
            &tx,
            "SELECT role, content, tool_calls, tool_call_id, reasoning_content
                 FROM context_messages ORDER BY position ASC",
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
        (messages, source_message_id, true)
    } else {
        (Vec::new(), 0, false)
    };
    messages.extend(
        read_projected_canonical_messages_after_id(&tx, after_id)
            .map_err(|error| io::Error::other(error.to_string()))?,
    );
    tx.commit()
        .map_err(|error| io::Error::other(error.to_string()))?;

    Ok(ContextHistory {
        messages,
        source_message_id: latest_message_id,
        canonical_generation,
        snapshot_is_current: has_snapshot && after_id == latest_message_id,
    })
}

/// 原子替换可重建的上下文快照。若读取快照后发生 rewind/clear 等 canonical 改写，
/// generation 会变化，此时拒绝陈旧结果；普通并发 append 不改变 generation，且其
/// message id 大于传入水位，后续读取会把它作为 tail 合并回来。
pub(in crate::ai) fn write_context_snapshot_sqlite(
    path: &Path,
    messages: &[Message],
    source_message_id: i64,
    canonical_generation: i64,
    projection_fingerprint: &str,
) -> io::Result<bool> {
    write_context_snapshot_sqlite_with_busy_timeout(
        path,
        messages,
        source_message_id,
        canonical_generation,
        projection_fingerprint,
        Duration::from_secs(5),
    )
}

pub(in crate::ai) fn write_context_snapshot_sqlite_with_busy_timeout(
    path: &Path,
    messages: &[Message],
    source_message_id: i64,
    canonical_generation: i64,
    projection_fingerprint: &str,
    busy_timeout: Duration,
) -> io::Result<bool> {
    let deadline = Instant::now() + busy_timeout;
    with_session_state_lock_until(path, deadline, || {
        let remaining = deadline
            .saturating_duration_since(Instant::now())
            .max(Duration::from_millis(1));
        let mut conn = open_history_db_with_busy_timeout(path, remaining)?;
        init_history_schema(&conn)?;
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| sqlite_error(path, "begin context snapshot transaction", error))?;
        if history_generation(&tx)? != canonical_generation {
            return Ok(false);
        }

        tx.execute("DELETE FROM context_messages", [])
            .map_err(|error| sqlite_error(path, "clear context_messages", error))?;
        insert_context_messages(&tx, path, messages)?;
        tx.execute(
            "INSERT INTO context_snapshot
                (singleton, source_message_id, source_generation, projection_fingerprint)
             VALUES (1, ?1, ?2, ?3)
             ON CONFLICT(singleton) DO UPDATE SET
                source_message_id = excluded.source_message_id,
                source_generation = excluded.source_generation,
                projection_fingerprint = excluded.projection_fingerprint",
            params![
                source_message_id,
                canonical_generation,
                projection_fingerprint
            ],
        )
        .map_err(|error| sqlite_error(path, "upsert context_snapshot", error))?;
        bump_history_revision(&tx)?;
        tx.commit()
            .map_err(|error| sqlite_error(path, "commit context snapshot transaction", error))?;
        Ok(true)
    })
}

fn insert_context_messages(conn: &Connection, path: &Path, messages: &[Message]) -> io::Result<()> {
    let mut stmt = conn
        .prepare(
            "INSERT INTO context_messages
                (position, role, content, tool_calls, tool_call_id, reasoning_content)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )
        .map_err(|error| sqlite_error(path, "prepare context_messages insert", error))?;
    for (position, message) in messages.iter().enumerate() {
        let content = serde_json::to_string(&message.content)
            .map_err(|error| io::Error::other(error.to_string()))?;
        let tool_calls = message
            .tool_calls
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| io::Error::other(error.to_string()))?;
        stmt.execute(params![
            position as i64,
            message.role,
            content,
            tool_calls,
            message.tool_call_id,
            message.reasoning_content,
        ])
        .map_err(|error| {
            sqlite_error(
                path,
                &format!("insert context_messages row {position}"),
                error,
            )
        })?;
    }
    Ok(())
}

pub(in crate::ai) fn build_message_arr_sqlite(
    history_count: usize,
    history_file: &Path,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let conn = match open_history_db(history_file) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.into()),
    };
    init_history_schema(&conn)?;

    let messages = read_messages_with_sql(
        &conn,
        "SELECT role, content, tool_calls, tool_call_id, reasoning_content
         FROM messages
         ORDER BY id ASC",
    )?;
    if history_count >= messages.len() {
        return Ok(messages);
    }
    Ok(messages[messages.len() - history_count..].to_vec())
}

pub(in crate::ai) fn read_recent_messages_sqlite(
    history_file: &Path,
    limit: usize,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let conn = match open_history_db(history_file) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.into()),
    };
    init_history_schema(&conn)?;

    let mut stmt = conn.prepare(
        "SELECT role, content, tool_calls, tool_call_id, reasoning_content
         FROM messages
         ORDER BY id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], |row| {
        let role: String = row.get(0)?;
        let content: String = row.get(1)?;
        let tool_calls: Option<String> = row.get(2)?;
        let tool_call_id: Option<String> = row.get(3)?;
        let reasoning_content: Option<String> = row.get(4)?;
        Ok(Message {
            role,
            content: decode_message_content(&content),
            tool_calls: decode_tool_calls(tool_calls.as_deref()),
            tool_call_id,
            reasoning_content,
        })
    })?;

    let mut messages = Vec::new();
    for row in rows {
        messages.push(row?);
    }
    Ok(messages)
}

pub(in crate::ai) fn read_recent_turn_window_sqlite(
    history_file: &Path,
    keep_last_user_turns: usize,
) -> Result<RecentTurnWindow, Box<dyn std::error::Error>> {
    let conn = match open_history_db(history_file) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            return Ok(RecentTurnWindow {
                messages: Vec::new(),
                start_message_id: None,
                has_older_messages: false,
            });
        }
        Err(err) => return Err(err.into()),
    };
    init_history_schema(&conn)?;

    if keep_last_user_turns == 0 {
        let messages = read_messages_with_sql(
            &conn,
            "SELECT role, content, tool_calls, tool_call_id, reasoning_content
             FROM messages
             ORDER BY id ASC",
        )?;
        return Ok(RecentTurnWindow {
            messages,
            start_message_id: None,
            has_older_messages: false,
        });
    }

    let threshold_user_id: Option<i64> = conn
        .query_row(
            "SELECT id FROM messages
             WHERE role='user'
             ORDER BY id DESC
             LIMIT 1 OFFSET ?1",
            params![keep_last_user_turns.saturating_sub(1) as i64],
            |row| row.get(0),
        )
        .optional()?;

    let Some(start_message_id) = threshold_user_id else {
        let messages = read_messages_with_sql(
            &conn,
            "SELECT role, content, tool_calls, tool_call_id, reasoning_content
             FROM messages
             ORDER BY id ASC",
        )?;
        return Ok(RecentTurnWindow {
            messages,
            start_message_id: None,
            has_older_messages: false,
        });
    };

    let has_older_messages = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM messages WHERE id < ?1 LIMIT 1)",
            params![start_message_id],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0)
        != 0;

    let messages = read_messages_since_id(&conn, start_message_id)?;
    Ok(RecentTurnWindow {
        messages,
        start_message_id: Some(start_message_id),
        has_older_messages,
    })
}

pub(in crate::ai) fn read_latest_history_summary_before_id_sqlite(
    history_file: &Path,
    before_message_id: i64,
) -> Result<Option<Message>, Box<dyn std::error::Error>> {
    let conn = match open_history_db(history_file) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    init_history_schema(&conn)?;

    let mut stmt = conn.prepare(
        "SELECT role, content, tool_calls, tool_call_id, reasoning_content
         FROM messages
         WHERE id < ?1 AND role = ?2
         ORDER BY id DESC
         LIMIT 8",
    )?;
    let rows = stmt.query_map(params![before_message_id, ROLE_INTERNAL_NOTE], |row| {
        let role: String = row.get(0)?;
        let content: String = row.get(1)?;
        let tool_calls: Option<String> = row.get(2)?;
        let tool_call_id: Option<String> = row.get(3)?;
        let reasoning_content: Option<String> = row.get(4)?;
        Ok(Message {
            role,
            content: decode_message_content(&content),
            tool_calls: decode_tool_calls(tool_calls.as_deref()),
            tool_call_id,
            reasoning_content,
        })
    })?;

    for row in rows {
        let message = row?;
        let Some(text) = message.content.as_str() else {
            continue;
        };
        // 摘要前缀识别统一走 compress::is_summary_note_text（唯一真源）。
        // 此前这里硬编码 3 种前缀、漏掉 `长期记忆摘要（压缩保留）`，导致 fastpath
        // 找不到 overflow 路径产生的摘要接续点、每轮回退到全量慢路径重新压缩。
        if is_summary_note_text(text) {
            return Ok(Some(message));
        }
    }
    Ok(None)
}

/// 读取滑动窗口之前最近的 context checkpoint markers。它们是正文 asset 的唯一
/// 索引，不能因为 SQLite fast path 只加载 recent turns 而从请求上下文静默消失。
/// 请求正规化层仍会将最终投影限制为最近 8 条。
pub(in crate::ai) fn read_context_checkpoint_markers_before_id_sqlite(
    history_file: &Path,
    before_message_id: i64,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let conn = match open_history_db(history_file) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.into()),
    };
    init_history_schema(&conn)?;

    let mut stmt = conn.prepare(
        "SELECT role, content, tool_calls, tool_call_id, reasoning_content
         FROM messages
         WHERE id < ?1
           AND role = ?2
           AND instr(content, '[context_checkpoint') > 0
         ORDER BY id DESC
         LIMIT 8",
    )?;
    let rows = stmt.query_map(params![before_message_id, ROLE_INTERNAL_NOTE], |row| {
        let role: String = row.get(0)?;
        let content: String = row.get(1)?;
        let tool_calls: Option<String> = row.get(2)?;
        let tool_call_id: Option<String> = row.get(3)?;
        let reasoning_content: Option<String> = row.get(4)?;
        Ok(Message {
            role,
            content: decode_message_content(&content),
            tool_calls: decode_tool_calls(tool_calls.as_deref()),
            tool_call_id,
            reasoning_content,
        })
    })?;

    let mut markers = Vec::new();
    for row in rows {
        let message = row?;
        if message
            .content
            .as_str()
            .is_some_and(|text| text.trim_start().starts_with("[context_checkpoint "))
        {
            markers.push(message);
        }
    }
    markers.reverse();
    Ok(markers)
}

/// 把 history 中 context checkpoint marker 的 assets 路径重定位到新 session。
/// fork 时传入源 assets 目录做精确前缀替换；归档导入时源路径未知，会仅接受
/// `context-checkpoints/<file>` 的受控相对尾部，避免把普通文本或任意绝对路径改写。
pub(in crate::ai) fn remap_context_checkpoint_paths_sqlite(
    history_file: &Path,
    source_assets: Option<&Path>,
    target_assets: &Path,
) -> io::Result<usize> {
    with_session_state_lock(history_file, || {
        remap_context_checkpoint_paths_sqlite_unlocked(history_file, source_assets, target_assets)
    })
}

fn remap_context_checkpoint_paths_sqlite_unlocked(
    history_file: &Path,
    source_assets: Option<&Path>,
    target_assets: &Path,
) -> io::Result<usize> {
    let mut conn = open_history_db(history_file)?;
    init_history_schema(&conn)?;
    let tx = conn
        .transaction()
        .map_err(|e| io::Error::other(e.to_string()))?;
    let rows = {
        let mut stmt = tx
            .prepare(
                "SELECT id, content
                 FROM messages
                 WHERE role = ?1
                   AND instr(content, '[context_checkpoint path=') > 0",
            )
            .map_err(|e| io::Error::other(e.to_string()))?;
        let rows = stmt
            .query_map([ROLE_INTERNAL_NOTE], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| io::Error::other(e.to_string()))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| io::Error::other(e.to_string()))?
    };

    let mut remapped = 0usize;
    for (id, encoded_content) in rows {
        let content = decode_message_content(&encoded_content);
        let Some(text) = content.as_str() else {
            continue;
        };
        let Some(remapped_text) =
            remap_context_checkpoint_marker(text, source_assets, target_assets)
        else {
            continue;
        };
        let encoded = serde_json::to_string(&Value::String(remapped_text))
            .map_err(|e| io::Error::other(e.to_string()))?;
        tx.execute(
            "UPDATE messages SET content = ?1 WHERE id = ?2",
            params![encoded, id],
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
        remapped += 1;
    }
    if remapped > 0 {
        invalidate_context_snapshot(&tx)?;
        bump_history_revision(&tx)?;
    }
    tx.commit().map_err(|e| io::Error::other(e.to_string()))?;
    Ok(remapped)
}

fn remap_context_checkpoint_marker(
    text: &str,
    source_assets: Option<&Path>,
    target_assets: &Path,
) -> Option<String> {
    const PREFIX: &str = "[context_checkpoint path=";
    let leading_len = text.len().checked_sub(text.trim_start().len())?;
    let (leading, trimmed) = text.split_at(leading_len);
    let rest = trimmed.strip_prefix(PREFIX)?;
    let closing = rest.find(']')?;
    let recorded = Path::new(&rest[..closing]);
    let relative = source_assets
        .and_then(|source| recorded.strip_prefix(source).ok())
        .and_then(checked_context_checkpoint_relative)
        .or_else(|| checked_context_checkpoint_relative(recorded))?;
    let remapped = target_assets.join(relative);
    Some(format!(
        "{leading}{PREFIX}{}{}",
        remapped.display(),
        &rest[closing..]
    ))
}

fn checked_context_checkpoint_relative(path: &Path) -> Option<PathBuf> {
    let mut found_checkpoint_dir = false;
    let mut relative = PathBuf::new();
    let mut has_file = false;
    for component in path.components() {
        if !found_checkpoint_dir {
            if let std::path::Component::Normal(part) = component
                && part == "context-checkpoints"
            {
                relative.push(part);
                found_checkpoint_dir = true;
            }
            continue;
        };
        match component {
            std::path::Component::Normal(part) => {
                relative.push(part);
                has_file = true;
            }
            _ => return None,
        }
    }
    (found_checkpoint_dir && has_file).then_some(relative)
}

pub(in crate::ai) fn clear_session_history_sqlite(path: &Path) -> io::Result<()> {
    with_session_state_lock(path, || clear_session_history_sqlite_unlocked(path))
}

fn clear_session_history_sqlite_unlocked(path: &Path) -> io::Result<()> {
    let mut conn = match open_history_db(path) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err),
    };
    init_history_schema(&conn)?;
    // 事务包裹：DELETE messages / DELETE meta / bump revision 必须原子提交，
    // 否则中途崩溃会留下"messages 已清空但 revision 未变"的不一致状态，
    // 导致 context 缓存误判为未变化而继续供应旧历史。
    let tx = conn
        .transaction()
        .map_err(|e| io::Error::other(e.to_string()))?;
    tx.execute("DELETE FROM messages", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    invalidate_context_snapshot(&tx)?;
    tx.execute("DELETE FROM tool_execution_outcomes", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    tx.execute("DELETE FROM skill_activation_events", [])
        .map_err(|e| io::Error::other(e.to_string()))?;
    // 保留 history_revision 行：它是缓存失效计数器，须跨 clear **单调递增**。
    // history_generation 是快照并发写的 fencing token，clear 后也必须单调递增；
    // turn_seq 同样是 session 级身份，清空上下文不能让旧序号被复用。
    // 若连同它一起删掉，bump 会从 1 重新开始，版本号回退后可能与早期缓存
    // 条目的 revision 撞车，反而让已失效的旧历史被误命中。
    tx.execute(
        "DELETE FROM meta
         WHERE key NOT IN ('history_revision', 'history_generation', 'turn_seq')",
        [],
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    bump_history_revision(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))
}

/// 把 messages 表保留到前 `keep` 条（按 id 升序）。用于 session branch：
/// 复制完整 sqlite 后再回滚到指定消息数。`keep == 0` 等价于 clear。
pub(in crate::ai) fn truncate_messages_sqlite(path: &Path, keep: usize) -> io::Result<()> {
    with_session_state_lock(path, || truncate_messages_sqlite_unlocked(path, keep))
}

fn truncate_messages_sqlite_unlocked(path: &Path, keep: usize) -> io::Result<()> {
    let mut conn = match open_history_db(path) {
        Ok(c) => c,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err),
    };
    init_history_schema(&conn)?;
    // 事务包裹：DELETE + bump revision 原子提交，避免中途崩溃留下
    // "已删消息但 revision 未变"的不一致状态（context 缓存供应错误的空结果）。
    let tx = conn
        .transaction()
        .map_err(|e| io::Error::other(e.to_string()))?;
    drop_ambiguous_tool_execution_outcomes(&tx)?;
    if keep == 0 {
        tx.execute("DELETE FROM messages", [])
            .map_err(|e| io::Error::other(e.to_string()))?;
        invalidate_context_snapshot(&tx)?;
        tx.execute("DELETE FROM tool_execution_outcomes", [])
            .map_err(|e| io::Error::other(e.to_string()))?;
        tx.execute("DELETE FROM skill_activation_events", [])
            .map_err(|e| io::Error::other(e.to_string()))?;
        clear_llm_prune_marks_meta(&tx)?;
        bump_history_revision(&tx)?;
        return tx.commit().map_err(|e| io::Error::other(e.to_string()));
    }
    // 取前 `keep` 条的最大 id，删掉其后的所有行。
    let cutoff: Option<i64> = tx
        .query_row(
            "SELECT id FROM messages ORDER BY id ASC LIMIT 1 OFFSET ?1",
            params![(keep as i64) - 1],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| io::Error::other(e.to_string()))?;
    if let Some(cutoff_id) = cutoff {
        tx.execute("DELETE FROM messages WHERE id > ?1", params![cutoff_id])
            .map_err(|e| io::Error::other(e.to_string()))?;
        invalidate_context_snapshot(&tx)?;
        clear_llm_prune_marks_meta(&tx)?;
    }
    prune_orphan_tool_execution_outcomes(&tx)?;
    bump_history_revision(&tx)?;
    tx.commit().map_err(|e| io::Error::other(e.to_string()))
}

/// 把 messages 表保留到前 `keep_turns` 个完整用户 turn。
///
/// 用户 turn 从 `role='user'` 开始，到下一条用户消息前结束；按下一条用户消息
/// 截断可让 assistant tool call 与随后的 tool result 留在同一侧。
pub(in crate::ai) fn truncate_messages_to_user_turns_sqlite(
    path: &Path,
    keep_turns: usize,
) -> io::Result<()> {
    if keep_turns == 0 {
        return truncate_messages_sqlite(path, 0);
    }

    with_session_state_lock(path, || {
        truncate_messages_to_user_turns_sqlite_unlocked(path, keep_turns)
    })
}

fn truncate_messages_to_user_turns_sqlite_unlocked(
    path: &Path,
    keep_turns: usize,
) -> io::Result<()> {
    let mut conn = match open_history_db(path) {
        Ok(connection) => connection,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    init_history_schema(&conn)?;
    let tx = conn
        .transaction()
        .map_err(|error| io::Error::other(error.to_string()))?;
    drop_ambiguous_tool_execution_outcomes(&tx)?;
    let next_turn_start: Option<i64> = tx
        .query_row(
            "SELECT id FROM messages WHERE role = 'user' ORDER BY id ASC LIMIT 1 OFFSET ?1",
            params![keep_turns as i64],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| io::Error::other(error.to_string()))?;
    if let Some(next_turn_start) = next_turn_start {
        tx.execute(
            "DELETE FROM messages WHERE id >= ?1",
            params![next_turn_start],
        )
        .map_err(|error| io::Error::other(error.to_string()))?;
        invalidate_context_snapshot(&tx)?;
        clear_llm_prune_marks_meta(&tx)?;
    }
    prune_orphan_tool_execution_outcomes(&tx)?;
    bump_history_revision(&tx)?;
    tx.commit()
        .map_err(|error| io::Error::other(error.to_string()))
}

pub(in crate::ai) fn read_first_user_prompt_sqlite(path: &Path) -> io::Result<Option<String>> {
    let conn = open_history_db(path)?;
    read_first_user_prompt_from_conn(&conn)
}

fn read_first_user_prompt_from_conn(conn: &Connection) -> io::Result<Option<String>> {
    let meta: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key='first_user_prompt' LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .unwrap_or(None);
    if meta
        .as_deref()
        .is_some_and(|prompt| !super::sessions::is_preserved_content_message(prompt))
    {
        return Ok(meta);
    }

    // 缓存的首条消息可能是图片/文本归档协议。继续向后查找第一条真实用户请求，
    // 避免过滤内部协议后把已有会话错误显示成 `new session`。
    let mut stmt = conn
        .prepare("SELECT content FROM messages WHERE role='user' ORDER BY id ASC")
        .map_err(|e| io::Error::other(e.to_string()))?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| io::Error::other(e.to_string()))?;
    let mut prompts = Vec::with_capacity(3);
    for raw in rows {
        let raw = raw.map_err(|e| io::Error::other(e.to_string()))?;
        let prompt = value_to_string(&decode_message_content(&raw));
        if !super::sessions::is_preserved_content_message(&prompt) {
            prompts.push(prompt);
            if prompts.len() == 3 {
                break;
            }
        }
    }
    Ok((!prompts.is_empty()).then(|| prompts.join("\n---\n")))
}

/// 读取 session 标题（存储在 meta 表中，key='session_title'）。
pub(in crate::ai) fn read_session_title_sqlite(path: &Path) -> io::Result<Option<String>> {
    let conn = open_history_db(path)?;
    Ok(read_session_title_from_conn(&conn))
}

fn read_session_title_from_conn(conn: &Connection) -> Option<String> {
    let title: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key='session_title' LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .unwrap_or(None);
    title.filter(|title| !title.trim().is_empty())
}

fn read_i64_meta_from_conn(conn: &Connection, key: &str) -> Option<i64> {
    conn.query_row(
        "SELECT value FROM meta WHERE key=?1 LIMIT 1",
        params![key],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .ok()
    .flatten()
    .and_then(|value| value.parse::<i64>().ok())
}

/// 旧 session 尚未写入显式活动时间时，以最后一条 canonical message 的创建时间
/// 作为活动时间。`messages.created_at` 使用 Unix 秒，列表接口统一返回毫秒。
fn read_latest_message_activity_unix_ms(conn: &Connection) -> Option<i64> {
    conn.query_row("SELECT MAX(created_at) FROM messages", [], |row| {
        row.get::<_, Option<i64>>(0)
    })
    .ok()
    .flatten()
    .and_then(|seconds| seconds.checked_mul(1_000))
}

/// 单次只读连接读取 `/ss` 列表的标题、首条用户请求与活动时间。
///
/// 两项元数据沿用列表层原本的容错语义：单项查询失败不影响另一项，也不会让
/// 一个损坏或旧格式的 session 阻断整个列表。
pub(in crate::ai) fn read_session_list_metadata_sqlite(
    path: &Path,
) -> io::Result<SessionListMetadata> {
    let conn = open_history_db_read_only(path)?;
    Ok(SessionListMetadata {
        first_user_prompt: read_first_user_prompt_from_conn(&conn).unwrap_or(None),
        session_title: read_session_title_from_conn(&conn),
        last_activity_unix_ms: read_i64_meta_from_conn(&conn, LAST_ACTIVITY_META_KEY)
            .or_else(|| read_latest_message_activity_unix_ms(&conn)),
        history_revision: read_i64_meta_from_conn(&conn, "history_revision").unwrap_or(0),
    })
}

/// 读取 session 标题来源（`model` / `fallback`）；缺失时调用方按旧数据处理。
pub(in crate::ai) fn read_session_title_origin_sqlite(path: &Path) -> io::Result<Option<String>> {
    let conn = open_history_db(path)?;
    let origin: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key='session_title_origin' LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()
        .unwrap_or(None);
    Ok(origin.filter(|value| !value.trim().is_empty()))
}

/// 原子写入 session 标题及其来源，避免 fallback 被误认为模型标题而永久跳过升级。
pub(in crate::ai) fn write_session_title_sqlite(
    path: &Path,
    title: &str,
    origin: &str,
) -> io::Result<()> {
    with_session_state_lock(path, || {
        let mut conn = open_history_db(path)?;
        init_history_schema(&conn)?;
        let tx = conn
            .transaction()
            .map_err(|e| io::Error::other(e.to_string()))?;
        tx.execute(
            "INSERT OR REPLACE INTO meta (key, value, created_at) VALUES ('session_title', ?1, unixepoch())",
            rusqlite::params![title],
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
        tx.execute(
            "INSERT OR REPLACE INTO meta (key, value, created_at) VALUES ('session_title_origin', ?1, unixepoch())",
            rusqlite::params![origin],
        )
        .map_err(|e| io::Error::other(e.to_string()))?;
        touch_session_activity(&tx)?;
        tx.commit().map_err(|e| io::Error::other(e.to_string()))
    })
}

fn decode_message_content(content: &str) -> Value {
    serde_json::from_str(content).unwrap_or_else(|_| Value::String(content.to_string()))
}

fn decode_tool_calls(tool_calls: Option<&str>) -> Option<Vec<ToolCall>> {
    tool_calls.and_then(|raw| serde_json::from_str(raw).ok())
}

pub(in crate::ai) fn read_all_messages_sqlite(path: &Path) -> io::Result<Vec<Message>> {
    let conn = open_history_db(path)?;

    read_messages_with_sql(
        &conn,
        "SELECT role, content, tool_calls, tool_call_id, reasoning_content
         FROM messages
         ORDER BY id ASC",
    )
    .map_err(|e| io::Error::other(e.to_string()))
}

fn read_messages_since_id(
    conn: &Connection,
    start_message_id: i64,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let mut stmt = conn.prepare(
        "SELECT role, content, tool_calls, tool_call_id, reasoning_content
         FROM messages
         WHERE id >= ?1
         ORDER BY id ASC",
    )?;
    let rows = stmt.query_map(params![start_message_id], |row| {
        let role: String = row.get(0)?;
        let content: String = row.get(1)?;
        let tool_calls: Option<String> = row.get(2)?;
        let tool_call_id: Option<String> = row.get(3)?;
        let reasoning_content: Option<String> = row.get(4)?;
        Ok((role, content, tool_calls, tool_call_id, reasoning_content))
    })?;
    let mut messages = Vec::new();
    for row in rows {
        let (role, content, tool_calls, tool_call_id, reasoning_content) = row?;
        messages.push(Message {
            role,
            content: decode_message_content(&content),
            tool_calls: decode_tool_calls(tool_calls.as_deref()),
            tool_call_id,
            reasoning_content,
        });
    }
    Ok(messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn sqlite_error_classifies_transient_failures_as_retryable() {
        // BUSY/LOCKED 与 SQLITE_IOERR 系统 I/O 失败（如并发首开 WAL 时的
        // FSTAT/SHMOPEN）都属瞬时，必须归为 WouldBlock 供上层短退避重试。
        for result_code in [
            rusqlite::ffi::SQLITE_BUSY,
            rusqlite::ffi::SQLITE_LOCKED,
            rusqlite::ffi::SQLITE_IOERR,
            rusqlite::ffi::SQLITE_IOERR_FSTAT,
        ] {
            let error =
                rusqlite::Error::SqliteFailure(rusqlite::ffi::Error::new(result_code), None);
            assert_eq!(
                sqlite_error_kind(&error),
                io::ErrorKind::WouldBlock,
                "schema and transaction paths must share retry classification"
            );
            let io_error = sqlite_error(Path::new("history.db"), "test operation", error);
            assert_eq!(io_error.kind(), io::ErrorKind::WouldBlock);
        }

        // 非瞬时失败（只读文件系统）不得重试。
        let error = rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_READONLY),
            None,
        );
        let io_error = sqlite_error(Path::new("history.db"), "test operation", error);
        assert_eq!(io_error.kind(), io::ErrorKind::Other);
    }

    #[test]
    fn context_snapshot_busy_timeout_includes_session_state_lock_wait() {
        let dir = std::env::temp_dir().join(format!(
            "snapshot_state_lock_timeout_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        let holder_path = path.clone();
        let (locked_tx, locked_rx) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            with_session_state_lock(&holder_path, || {
                locked_tx.send(()).unwrap();
                std::thread::sleep(Duration::from_millis(250));
                Ok(())
            })
            .unwrap();
        });
        locked_rx.recv().unwrap();

        let started = Instant::now();
        let error = write_context_snapshot_sqlite_with_busy_timeout(
            &path,
            &[],
            0,
            0,
            "fingerprint",
            Duration::from_millis(30),
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
        assert!(started.elapsed() < Duration::from_millis(200));

        holder.join().unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }

    fn msg(role: &str, text: &str) -> Message {
        Message {
            role: role.to_string(),
            content: Value::String(text.to_string()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn tool_msg(id: &str, text: &str) -> Message {
        let mut message = msg("tool", text);
        message.tool_call_id = Some(id.to_string());
        message
    }

    fn outcome(id: &str) -> ToolExecutionOutcome {
        ToolExecutionOutcome {
            tool_call_id: id.to_string(),
            execution_signature: format!("signature-{id}"),
            succeeded: true,
        }
    }

    #[test]
    fn turn_sequence_is_atomic_and_survives_history_clear() {
        let dir = std::env::temp_dir().join(format!(
            "turn_sequence_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        append_history_sqlite(&path, vec![msg("user", "first"), msg("user", "second")]).unwrap();

        // 旧 session 第一次升级时从已落盘的 user turn 数继续编号。
        assert_eq!(reserve_turn_index_sqlite(&path).unwrap(), 2);
        assert_eq!(reserve_turn_index_sqlite(&path).unwrap(), 3);

        clear_session_history_sqlite(&path).unwrap();
        assert_eq!(reserve_turn_index_sqlite(&path).unwrap(), 4);

        // 每个线程都建立独立连接，覆盖多 runtime / 多进程相同的事务竞争路径。
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let path = path.clone();
                std::thread::spawn(move || reserve_turn_index_sqlite(&path).unwrap())
            })
            .collect();
        let mut indexes: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        indexes.sort_unstable();
        assert_eq!(indexes, (5..13).collect::<Vec<_>>());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rollback_metadata_read_error_does_not_replace_live_history() {
        let dir = std::env::temp_dir().join(format!(
            "rollback_metadata_error_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let live = dir.join("live.sqlite");
        let checkpoint = dir.join("checkpoint.sqlite");
        append_history_sqlite(&live, vec![msg("user", "live message")]).unwrap();
        append_history_sqlite(&checkpoint, vec![msg("user", "checkpoint message")]).unwrap();
        let conn = Connection::open(&live).unwrap();
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('turn_seq', 'invalid')
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [],
        )
        .unwrap();
        drop(conn);

        let error = restore_sqlite_after_rollback(&checkpoint, &live).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        let messages = read_all_messages_sqlite(&live).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(value_to_string(&messages[0].content), "live message");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P1 回归：`read_history_revision` 必须**跨连接**观察到写入递增。
    /// 每次写路径都开新连接读版本号，模拟 build_context_history 的缓存失效判定。
    /// 旧实现用连接局部的 `PRAGMA data_version`，新连接恒返回固定值，无法失效缓存。
    #[test]
    fn history_revision_increments_across_fresh_connections() {
        let dir = std::env::temp_dir().join(format!(
            "hist_rev_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");

        // 全新库尚未写入过 revision：视为 0（"从未修改"）。
        assert_eq!(read_history_revision(&path), Some(0));

        append_history_sqlite(&path, vec![msg("user", "hi")]).unwrap();
        let r1 = read_history_revision(&path).unwrap();
        assert!(r1 > 0, "append should bump revision, got {r1}");

        append_history_sqlite(&path, vec![msg("assistant", "hello")]).unwrap();
        let r2 = read_history_revision(&path).unwrap();
        assert!(r2 > r1, "second append should bump again: {r1} -> {r2}");

        replace_all_messages_sqlite(&path, &[msg("user", "reset")]).unwrap();
        let r3 = read_history_revision(&path).unwrap();
        assert!(r3 > r2, "replace should bump: {r2} -> {r3}");

        truncate_messages_sqlite(&path, 0).unwrap();
        let r4 = read_history_revision(&path).unwrap();
        assert!(r4 > r3, "truncate should bump: {r3} -> {r4}");

        // clear 会 DELETE meta 后再 bump，结构与其它写路径不同，需单独覆盖。
        append_history_sqlite(&path, vec![msg("user", "again")]).unwrap();
        let r5 = read_history_revision(&path).unwrap();
        clear_session_history_sqlite(&path).unwrap();
        let r6 = read_history_revision(&path).unwrap();
        assert!(
            r6 > r5,
            "clear should bump even after wiping meta: {r5} -> {r6}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn history_revision_cache_invalidates_on_wal_write_with_live_connection() {
        // 验证：有存活连接时 WAL 写入不 checkpoint 主库，但 `-wal` sidecar 变化
        // 必须使 revision 缓存失效，否则缓存返回旧值。
        let dir = std::env::temp_dir().join(format!(
            "hist_rev_wal_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");

        append_history_sqlite(&path, vec![msg("user", "first")]).unwrap();
        let r1 = read_history_revision(&path).unwrap();
        assert!(r1 > 0);

        // 保持连接存活，阻止 WAL checkpoint 回写主库
        let guard = rusqlite::Connection::open(&path).unwrap();

        // 短生命连接写入：WAL 增长但主库 mtime 可能不变
        append_history_sqlite(&path, vec![msg("user", "second")]).unwrap();

        // 缓存必须失效（WAL sidecar 元数据变化），返回新 revision
        let r2 = read_history_revision(&path).unwrap();
        assert!(
            r2 > r1,
            "revision must increment even with live connection: {r1} -> {r2}"
        );

        drop(guard);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn llm_prune_marks_round_trip_and_clear_on_history_truncate() {
        let dir = std::env::temp_dir().join(format!(
            "llm_prune_marks_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.sqlite");
        append_history_sqlite(&path, vec![msg("user", "seed")]).unwrap();

        let marks = [("call_b".to_string(), 2_u8), ("call_a".to_string(), 1_u8)]
            .into_iter()
            .collect::<FxHashMap<_, _>>();
        write_llm_prune_marks_sqlite(&path, &marks).unwrap();
        assert_eq!(read_llm_prune_marks_sqlite(&path).unwrap(), marks);

        truncate_messages_sqlite(&path, 0).unwrap();
        assert!(read_llm_prune_marks_sqlite(&path).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_patch_targets_survive_history_replacement_and_clear_with_session() {
        let dir = std::env::temp_dir().join(format!(
            "stale_patch_meta_test_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.sqlite");
        append_history_sqlite(&path, vec![msg("user", "before compression")]).unwrap();

        let targets =
            FxHashSet::from_iter([PathBuf::from("/tmp/a.rs"), PathBuf::from("/tmp/b.rs")]);
        write_stale_patch_targets_sqlite(&path, &targets).unwrap();
        assert_eq!(
            read_stale_patch_targets_sqlite(&path).unwrap(),
            Some(targets.clone())
        );

        // replace_all_messages 是持久化 history 压缩/改写路径；账本不能随消息形态丢失。
        replace_all_messages_sqlite(&path, &[msg("user", "after compression")]).unwrap();
        assert_eq!(
            read_stale_patch_targets_sqlite(&path).unwrap(),
            Some(targets)
        );

        // 显式空集合也要与“旧库尚无 meta”区分，防止恢复时误走 legacy 回放。
        write_stale_patch_targets_sqlite(&path, &FxHashSet::default()).unwrap();
        assert_eq!(
            read_stale_patch_targets_sqlite(&path).unwrap(),
            Some(FxHashSet::default())
        );

        clear_session_history_sqlite(&path).unwrap();
        assert_eq!(read_stale_patch_targets_sqlite(&path).unwrap(), None);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn first_user_prompt_skips_preserved_content_notices() {
        let dir = std::env::temp_dir().join(format!(
            "first_prompt_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");

        append_history_sqlite(
            &path,
            vec![
                msg("user", "较早的用户图片内容已归档，原文未丢失。"),
                msg("user", r#"[[PRESERVED_CONTENT_STUB_V1]]{"kind":"image"}"#),
                msg("user", "较早的用户图片内容已归档，归档文件: /tmp/2"),
                msg("user", "较早的用户图片内容已归档，归档文件: /tmp/3"),
                msg("user", "较早的用户图片内容已归档，归档文件: /tmp/4"),
                msg("user", "这是实际用户请求"),
                msg("user", "这是后续用户请求"),
            ],
        )
        .unwrap();

        assert_eq!(
            read_first_user_prompt_sqlite(&path).unwrap().as_deref(),
            Some("这是实际用户请求\n---\n这是后续用户请求")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_list_metadata_reads_fields_and_legacy_activity_without_creating_database() {
        let dir = std::env::temp_dir().join(format!(
            "session_list_metadata_test_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let missing_path = dir.join("missing.db");
        assert!(read_session_list_metadata_sqlite(&missing_path).is_err());
        assert!(!missing_path.exists());

        let path = dir.join("history.db");
        append_history_sqlite(&path, vec![msg("user", "first prompt")]).unwrap();
        write_session_title_sqlite(&path, "generated title", "model").unwrap();

        // 模拟尚未写入 last_activity_unix_ms 的旧 session。列表必须使用消息时间，
        // 不能依赖会被只读连接创建/刷新的 SQLite -shm 文件时间。
        let conn = open_history_db(&path).unwrap();
        conn.execute("UPDATE messages SET created_at = ?1", [1_700_000_000_i64])
            .unwrap();
        conn.execute("DELETE FROM meta WHERE key = 'last_activity_unix_ms'", [])
            .unwrap();
        drop(conn);

        let metadata = read_session_list_metadata_sqlite(&path).unwrap();
        assert_eq!(metadata.first_user_prompt.as_deref(), Some("first prompt"));
        assert_eq!(metadata.session_title.as_deref(), Some("generated title"));
        assert_eq!(metadata.last_activity_unix_ms, Some(1_700_000_000_000));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn structured_tool_outcomes_follow_message_lifecycle() {
        let dir = std::env::temp_dir().join(format!(
            "tool_outcome_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        let original_messages = vec![tool_msg("call-1", "first"), tool_msg("call-2", "second")];
        append_history_sqlite(&path, original_messages.clone()).unwrap();
        let expected = vec![outcome("call-1"), outcome("call-2")];
        append_tool_execution_outcomes_sqlite(&path, &expected).unwrap();

        assert_eq!(
            read_tool_execution_outcomes_sqlite(&path).unwrap(),
            expected
        );
        assert_eq!(read_all_messages_sqlite(&path).unwrap(), original_messages);

        replace_all_messages_sqlite(&path, &[tool_msg("call-2", "second")]).unwrap();
        assert_eq!(
            read_tool_execution_outcomes_sqlite(&path).unwrap(),
            vec![outcome("call-2")]
        );

        append_history_sqlite(&path, vec![tool_msg("call-3", "third")]).unwrap();
        append_tool_execution_outcomes_sqlite(&path, &[outcome("call-3")]).unwrap();
        truncate_messages_sqlite(&path, 1).unwrap();
        assert_eq!(
            read_tool_execution_outcomes_sqlite(&path).unwrap(),
            vec![outcome("call-2")]
        );

        clear_session_history_sqlite(&path).unwrap();
        assert!(
            read_tool_execution_outcomes_sqlite(&path)
                .unwrap()
                .is_empty()
        );

        // 旧历史可能已经复用过同一 ID；改变消息集合前必须永久丢弃其歧义 outcome，
        // 否则删除较新的 occurrence 后会把它的状态错误绑定到保留的旧消息。
        append_history_sqlite(
            &path,
            vec![
                tool_msg("legacy-reused", "older"),
                tool_msg("legacy-reused", "newer"),
            ],
        )
        .unwrap();
        append_tool_execution_outcomes_sqlite(&path, &[outcome("legacy-reused")]).unwrap();
        replace_all_messages_sqlite(&path, &[tool_msg("legacy-reused", "older")]).unwrap();
        assert!(
            read_tool_execution_outcomes_sqlite(&path)
                .unwrap()
                .is_empty()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn skill_activation_events_preserve_injection_audit_without_messages() {
        let dir = std::env::temp_dir().join(format!(
            "skill_activation_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        let event = SkillActivationEvent {
            requested_skill: "bytedcli".to_string(),
            injected_skill: Some("bytedcli".to_string()),
            source: "/skills-inline".to_string(),
            outcome: "injected".to_string(),
        };

        append_skill_activation_event_sqlite(&path, &event).unwrap();

        assert_eq!(
            read_skill_activation_events_sqlite(&path).unwrap(),
            vec![event]
        );
        assert!(read_all_messages_sqlite(&path).unwrap().is_empty());

        clear_session_history_sqlite(&path).unwrap();
        assert!(
            read_skill_activation_events_sqlite(&path)
                .unwrap()
                .is_empty()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn structured_tool_outcomes_ignore_non_sqlite_history_files() {
        let dir = std::env::temp_dir().join(format!(
            "tool_outcome_text_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.txt");
        std::fs::write(&path, "plain text history\n").unwrap();

        append_tool_execution_outcomes_sqlite(&path, &[outcome("call-1")]).unwrap();

        assert!(
            read_tool_execution_outcomes_sqlite(&path)
                .unwrap()
                .is_empty()
        );
        assert!(read_tool_message_ids_sqlite(&path).unwrap().is_empty());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "plain text history\n"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncating_by_user_turn_prunes_removed_tool_outcomes() {
        let dir = std::env::temp_dir().join(format!(
            "tool_outcome_turn_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        append_history_sqlite(
            &path,
            vec![
                msg("user", "first turn"),
                tool_msg("call-1", "first result"),
                msg("user", "second turn"),
                tool_msg("call-2", "second result"),
            ],
        )
        .unwrap();
        append_tool_execution_outcomes_sqlite(&path, &[outcome("call-1"), outcome("call-2")])
            .unwrap();

        truncate_messages_to_user_turns_sqlite(&path, 1).unwrap();

        assert_eq!(
            read_tool_execution_outcomes_sqlite(&path).unwrap(),
            vec![outcome("call-1")]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn canonical_messages_are_unchanged_by_context_snapshots() {
        const PROJECTION: &str = "test-policy-a";
        let dir = std::env::temp_dir().join(format!(
            "context_snapshot_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        let assistant = Message {
            role: "assistant".to_string(),
            content: Value::String("准备读取文件".to_string()),
            tool_calls: Some(vec![ToolCall {
                id: "call-1".to_string(),
                tool_type: "function".to_string(),
                function: crate::ai::types::FunctionCall {
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            reasoning_content: Some("provider 原始 continuation state".to_string()),
        };
        let original = vec![msg("user", "读取文件"), assistant.clone()];
        append_history_sqlite_for_model(&path, original.clone(), Some("glm-5.2-opencode")).unwrap();

        let source = read_context_history_sqlite(&path, PROJECTION).unwrap();
        assert_eq!(read_all_messages_sqlite(&path).unwrap(), original);
        assert_ne!(
            source.messages[1].reasoning_content,
            assistant.reasoning_content
        );

        let snapshot = vec![msg(ROLE_INTERNAL_NOTE, "[History summary]\n已读取文件")];
        write_context_snapshot_sqlite(
            &path,
            &snapshot,
            source.source_message_id,
            source.canonical_generation,
            PROJECTION,
        )
        .unwrap();
        assert_eq!(read_all_messages_sqlite(&path).unwrap(), original);

        let tail = msg("user", "继续分析");
        append_history_sqlite_for_model(&path, vec![tail.clone()], Some("gpt-5.5")).unwrap();
        let projected = read_context_history_sqlite(&path, PROJECTION).unwrap();
        assert_eq!(projected.messages, vec![snapshot[0].clone(), tail.clone()]);
        assert!(!projected.snapshot_is_current);

        let mut expected_canonical = original;
        expected_canonical.push(tail);
        assert_eq!(read_all_messages_sqlite(&path).unwrap(), expected_canonical);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_context_snapshot_cannot_resurrect_truncated_history() {
        const PROJECTION: &str = "test-policy-a";
        let dir = std::env::temp_dir().join(format!(
            "stale_context_snapshot_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        append_history_sqlite(&path, vec![msg("user", "保留"), msg("assistant", "删除")]).unwrap();
        let stale_source = read_context_history_sqlite(&path, PROJECTION).unwrap();

        truncate_messages_sqlite(&path, 1).unwrap();
        let written = write_context_snapshot_sqlite(
            &path,
            &[msg(ROLE_INTERNAL_NOTE, "包含已删除内容的旧快照")],
            stale_source.source_message_id,
            stale_source.canonical_generation,
            PROJECTION,
        )
        .unwrap();
        assert!(!written);
        assert_eq!(
            read_context_history_sqlite(&path, PROJECTION)
                .unwrap()
                .messages,
            vec![msg("user", "保留")]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn context_snapshot_is_ignored_when_projection_policy_changes() {
        let dir = std::env::temp_dir().join(format!(
            "context_snapshot_policy_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        let canonical = vec![msg("user", "必须从 canonical 重建")];
        append_history_sqlite(&path, canonical.clone()).unwrap();

        let source = read_context_history_sqlite(&path, "policy-a").unwrap();
        write_context_snapshot_sqlite(
            &path,
            &[msg(ROLE_INTERNAL_NOTE, "旧策略摘要")],
            source.source_message_id,
            source.canonical_generation,
            "policy-a",
        )
        .unwrap();
        assert!(
            read_context_history_sqlite(&path, "policy-a")
                .unwrap()
                .snapshot_is_current
        );

        let rebuilt = read_context_history_sqlite(&path, "policy-b").unwrap();
        assert_eq!(rebuilt.messages, canonical);
        assert!(!rebuilt.snapshot_is_current);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_context_snapshot_cannot_resurrect_cleared_history() {
        const PROJECTION: &str = "test-policy-a";
        let dir = std::env::temp_dir().join(format!(
            "stale_context_snapshot_clear_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        append_history_sqlite(&path, vec![msg("user", "已经清除")]).unwrap();
        let stale_source = read_context_history_sqlite(&path, PROJECTION).unwrap();

        clear_session_history_sqlite(&path).unwrap();
        let current = msg("user", "清除后的新会话");
        append_history_sqlite(&path, vec![current.clone()]).unwrap();
        let written = write_context_snapshot_sqlite(
            &path,
            &[msg(ROLE_INTERNAL_NOTE, "包含已清除内容的旧快照")],
            stale_source.source_message_id,
            stale_source.canonical_generation,
            PROJECTION,
        )
        .unwrap();

        assert!(!written);
        assert_eq!(
            read_context_history_sqlite(&path, PROJECTION)
                .unwrap()
                .messages,
            vec![current]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rollback_state_lock_serializes_every_public_side_state_writer() {
        let dir = std::env::temp_dir().join(format!(
            "history_side_writer_lock_test_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        append_history_sqlite(&path, vec![msg("user", "seed")]).unwrap();

        type Writer = Box<dyn FnOnce(&Path) -> io::Result<()> + Send>;
        let writers: Vec<(&str, Writer)> = vec![
            (
                "tool outcomes",
                Box::new(|path| append_tool_execution_outcomes_sqlite(path, &[outcome("call-1")])),
            ),
            (
                "stale patch targets",
                Box::new(|path| {
                    let mut targets = FxHashSet::default();
                    targets.insert(PathBuf::from("src/main.rs"));
                    write_stale_patch_targets_sqlite(path, &targets)
                }),
            ),
            (
                "context snapshot",
                Box::new(|path| {
                    let source = read_context_history_sqlite(path, "lock-test")?;
                    write_context_snapshot_sqlite(
                        path,
                        &[msg(ROLE_INTERNAL_NOTE, "snapshot")],
                        source.source_message_id,
                        source.canonical_generation,
                        "lock-test",
                    )
                    .map(|_| ())
                }),
            ),
            (
                "session title",
                Box::new(|path| write_session_title_sqlite(path, "title", "test")),
            ),
        ];

        for (name, writer) in writers {
            let (lock_held_tx, lock_held_rx) = std::sync::mpsc::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let lock_path = path.clone();
            let lock_thread = std::thread::spawn(move || {
                with_session_state_lock(&lock_path, || {
                    lock_held_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(())
                })
            });
            lock_held_rx.recv().unwrap();

            let (done_tx, done_rx) = std::sync::mpsc::channel();
            let writer_path = path.clone();
            let writer_thread = std::thread::spawn(move || {
                let result = writer(&writer_path);
                done_tx.send(()).unwrap();
                result
            });
            assert!(
                done_rx
                    .recv_timeout(std::time::Duration::from_millis(100))
                    .is_err(),
                "{name} bypassed the rollback state lock"
            );

            release_tx.send(()).unwrap();
            lock_thread.join().unwrap().unwrap();
            writer_thread.join().unwrap().unwrap();
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn wake_note_text(pid: u64, ids: &[&str], checkpoint: &str) -> String {
        format!(
            "[Process {pid} Woke Up] Original goal: test goal\nNew mailbox messages:\n[TASK_WAIT_TIMEOUT]\nWall-clock task_wait budget elapsed after 30s. Re-call `task_wait` with the same task_ids to collect any ready results and receive the budget-elapsed status. task_ids=[{}]\nProgress: {checkpoint}\n\nWake-up handling rules:\n- rule\n\nResume execution based on the goal and these messages.",
            ids.join(", ")
        )
    }

    #[test]
    fn wait_wake_notes_coalesce_keeps_latest_in_sqlite_history() {
        let dir = std::env::temp_dir().join(format!(
            "wait_wake_coalesce_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");

        append_history_sqlite(
            &path,
            vec![
                msg("user", "goal"),
                msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a", "task_b"], "checkpoint-1")),
                msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a", "task_b"], "checkpoint-2")),
                msg(ROLE_INTERNAL_NOTE, &wake_note_text(7, &["task_x"], "checkpoint-3")),
            ],
        )
        .unwrap();

        let latest =
            msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a", "task_b"], "checkpoint-4"));
        assert!(coalesce_repeated_wait_wake_notes_sqlite(&path, &latest).unwrap());
        // 调用方随后把最新一条追加到尾部。
        append_history_sqlite(&path, vec![latest]).unwrap();

        let notes: Vec<_> = read_all_messages_sqlite(&path)
            .unwrap()
            .into_iter()
            .filter(|m| m.role == ROLE_INTERNAL_NOTE)
            .collect();
        assert_eq!(notes.len(), 2);
        assert!(notes[0].content.as_str().unwrap().contains("checkpoint-3"));
        assert!(notes[1].content.as_str().unwrap().contains("checkpoint-4"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wait_wake_coalesce_sqlite_window_is_last_total_messages() {
        let dir = std::env::temp_dir().join(format!(
            "sqlite_wake_dedup_window_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");

        // 与 blob 后端一致的窗口语义：扫描历史尾部 WAKE_NOTE_DEDUP_SCAN 条消息（不限角色），
        // 而非“最近 WAKE_NOTE_DEDUP_SCAN 条 internal_note”。
        // 旧等待笔记在第 1 条，其后跟 WAKE_NOTE_DEDUP_SCAN+1 条 user 消息，故其在窗口外，
        // 不应被删除（若按 internal_note 窗口扫描则会被命中删除）。
        let mut history = vec![msg(
            ROLE_INTERNAL_NOTE,
            &wake_note_text(6, &["task_a"], "checkpoint-old"),
        )];
        history.extend(
            (0..crate::ai::history::types::WAKE_NOTE_DEDUP_SCAN as usize + 1)
                .map(|i| msg("user", &format!("filler {i}"))),
        );
        append_history_sqlite(&path, history).unwrap();

        let latest =
            msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a"], "checkpoint-new"));
        assert!(!coalesce_repeated_wait_wake_notes_sqlite(&path, &latest).unwrap());

        let notes: Vec<_> = read_all_messages_sqlite(&path)
            .unwrap()
            .into_iter()
            .filter(|m| m.role == ROLE_INTERNAL_NOTE)
            .collect();
        // 窗口外的旧等待笔记保留，未被误删。
        assert_eq!(notes.len(), 1);
        assert!(notes[0].content.as_str().unwrap().contains("checkpoint-old"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wait_wake_coalesce_sqlite_is_noop_when_nothing_matches() {
        let dir = std::env::temp_dir().join(format!(
            "wait_wake_coalesce_noop_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.db");
        append_history_sqlite(
            &path,
            vec![
                msg("user", "goal"),
                msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a"], "checkpoint-1")),
            ],
        )
        .unwrap();

        // 同一 pid 但不同 task 集合：身份不同，不去重。
        let other = msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_z"], "checkpoint-x"));
        assert!(!coalesce_repeated_wait_wake_notes_sqlite(&path, &other).unwrap());

        // 非 internal_note 消息：fast path 不做任何 IO。
        assert!(!coalesce_repeated_wait_wake_notes_sqlite(&path, &msg("user", "hello")).unwrap());

        // 真实结果唤醒（parse 为 None）：不去重。
        let result_wake = msg(
            ROLE_INTERNAL_NOTE,
            "[Process 6 Woke Up] Original goal: g\nNew mailbox messages:\n[EVENT_WAKE]\nready\n\nWake-up handling rules:\n- rule\n\nResume execution based on the goal and these messages.",
        );
        assert!(!coalesce_repeated_wait_wake_notes_sqlite(&path, &result_wake).unwrap());

        // 数据库不存在：best-effort 返回 false，不报错。
        // 注意：必须用有效 wait note 才能越过 fast path，真正走到 open_history_db 分支。
        let missing = dir.join("missing.db");
        let wait_note = msg(ROLE_INTERNAL_NOTE, &wake_note_text(6, &["task_a"], "c"));
        assert!(!coalesce_repeated_wait_wake_notes_sqlite(&missing, &wait_note).unwrap());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
