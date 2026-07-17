//! 会话检查点（checkpoint）/ 回滚。
//!
//! 每个 session 的对话历史落在一个独立的 SQLite WAL 文件里。检查点通过
//! SQLite Online Backup API 创建一致快照，并同时保存 session assets，避免主库
//! 复制遗漏 WAL 页或 context checkpoint 正文。
//!
//! context-history 缓存以文件 len/mtime 与 `meta.history_revision` 作为失效键，
//! 复制文件会改变这些值，从而自动让缓存失效，回滚后能读到正确历史。

#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex, RwLock},
};

use chrono::{DateTime, Local};
#[cfg(unix)]
use libc::{LOCK_EX, LOCK_SH, LOCK_UN, flock};
use rust_tools::commonw::FastMap;

use crate::ai::history::SessionStore;

use super::{sessions::copy_dir_recursively, sqlite::backup_sqlite};

const MAX_CHECKPOINTS_PER_SESSION: usize = 20;
const MAX_CHECKPOINT_STORAGE_BYTES: u64 = 512 * 1024 * 1024;
const SQLITE_SIDECAR_SUFFIXES: [&str; 3] = ["-wal", "-shm", "-journal"];
const GENERATION_SQLITE_FILE: &str = "history.sqlite";
const GENERATION_ASSETS_DIR: &str = "assets";
const GENERATION_MANIFEST_FILE: &str = "checkpoint.manifest";
const GENERATION_STAGE_MARKER: &str = ".stage-";
const GENERATION_PREVIOUS_MARKER: &str = ".previous-";
const LIVE_ROLLBACK_PREFIX: &str = ".live-rollback-";
const LIVE_ROLLBACK_MANIFEST: &str = "rollback.manifest";

/// 进程内互斥与 `flock` 共同覆盖同一个 session 的 checkpoint 生命周期。
/// 每个 session 各自持有 Mutex，避免不同 session 的快照操作互相串行化。
static CHECKPOINT_SESSION_LOCKS: LazyLock<Mutex<FastMap<PathBuf, Arc<Mutex<()>>>>> =
    LazyLock::new(|| Mutex::new(FastMap::default()));

/// checkpoint 根目录的共享/排他 gate。普通操作持共享锁，`clear-all` 持排他锁，
/// 因此后者不会和一个刚创建的 session checkpoint 交错。
static CHECKPOINT_ROOT_LOCKS: LazyLock<Mutex<FastMap<PathBuf, Arc<RwLock<()>>>>> =
    LazyLock::new(|| Mutex::new(FastMap::default()));

#[derive(Debug, Clone)]
pub(in crate::ai) struct CheckpointInfo {
    pub(in crate::ai) name: String,
    pub(in crate::ai) modified_local: Option<DateTime<Local>>,
    pub(in crate::ai) size_bytes: u64,
}

#[derive(Debug, Clone)]
pub(in crate::ai) struct CheckpointStore {
    /// 当前 session 的 live sqlite 文件。
    session_file: PathBuf,
    /// 当前 session 的 assets 目录。
    session_assets: PathBuf,
    /// 该 session 的检查点目录。
    dir: PathBuf,
}

#[derive(Debug, Clone)]
enum CheckpointSource {
    Generation { sqlite: PathBuf, assets: PathBuf },
    Legacy { sqlite: PathBuf },
}

impl CheckpointStore {
    pub(in crate::ai) fn new(history_file: &Path, session_id: &str) -> Self {
        let store = SessionStore::new(history_file);
        Self::from_session_paths(
            store.session_history_file(session_id),
            store.session_assets_dir(session_id),
            store.checkpoints_dir(session_id),
        )
    }

    pub(super) fn from_session_paths(
        session_file: PathBuf,
        session_assets: PathBuf,
        dir: PathBuf,
    ) -> Self {
        Self {
            session_file,
            session_assets,
            dir,
        }
    }

    fn checkpoint_path(&self, name: &str) -> PathBuf {
        self.generation_dir(name).join(GENERATION_SQLITE_FILE)
    }

    fn checkpoint_assets_path(&self, name: &str) -> PathBuf {
        self.generation_dir(name).join(GENERATION_ASSETS_DIR)
    }

    fn generation_dir(&self, name: &str) -> PathBuf {
        self.dir.join(sanitize_name(name))
    }

    fn legacy_checkpoint_path(&self, name: &str) -> PathBuf {
        self.dir.join(format!("{}.sqlite", sanitize_name(name)))
    }

    fn legacy_checkpoint_assets_path(&self, name: &str) -> PathBuf {
        self.dir.join(format!("{}.assets", sanitize_name(name)))
    }

    /// 保存当前 session 历史为名为 `name` 的检查点。返回检查点文件路径。
    pub(in crate::ai) fn save(&self, name: &str) -> io::Result<PathBuf> {
        let normalized_name = sanitize_name(name);
        self.with_locked(|store| {
            if !store.session_file.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "current session has no history yet; nothing to checkpoint",
                ));
            }

            let checkpoints = store.list_unlocked()?;
            let existing_size = store.checkpoint_storage_size_for_name(&normalized_name)?;
            let staged = store.stage_generation(&normalized_name)?;
            let staged_size = directory_storage_size(&staged)?;
            let result = validate_checkpoint_budget(
                checkpoints.len(),
                store
                    .checkpoint_source_unlocked(&normalized_name)?
                    .is_some(),
                checkpoints.iter().fold(0u64, |total, checkpoint| {
                    total.saturating_add(checkpoint.size_bytes)
                }),
                existing_size,
                staged_size,
                MAX_CHECKPOINTS_PER_SESSION,
                MAX_CHECKPOINT_STORAGE_BYTES,
            )
            .and_then(|()| store.publish_generation(&normalized_name, &staged));
            if result.is_err() {
                let _ = fs::remove_dir_all(&staged);
            }
            result.map(|()| store.checkpoint_path(&normalized_name))
        })
    }

    /// 列出该 session 的全部检查点（按修改时间从新到旧，大小含 assets）。
    pub(in crate::ai) fn list(&self) -> io::Result<Vec<CheckpointInfo>> {
        self.with_locked(|store| store.list_unlocked())
    }

    /// 把名为 `name` 的检查点回滚到当前 session（覆盖 live 历史）。
    pub(in crate::ai) fn rollback(&self, name: &str) -> io::Result<()> {
        let normalized_name = sanitize_name(name);
        self.with_locked(|store| {
            let Some(source) = store.checkpoint_source_unlocked(&normalized_name)? else {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("checkpoint '{normalized_name}' not found"),
                ));
            };
            match source {
                // 旧格式没有 assets 快照，保留既有兼容语义：仅恢复 SQLite。
                CheckpointSource::Legacy { sqlite } => {
                    if let Some(parent) = store.session_file.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    backup_sqlite(&sqlite, &store.session_file)
                }
                CheckpointSource::Generation { sqlite, assets } => {
                    let transaction = store.stage_live_rollback(&sqlite, &assets)?;
                    store.complete_live_rollback(&transaction)
                }
            }
        })
    }

    /// 删除名为 `name` 的检查点。返回是否确有文件被删除。
    pub(in crate::ai) fn delete(&self, name: &str) -> io::Result<bool> {
        let normalized_name = sanitize_name(name);
        self.with_locked(|store| {
            let generation = store.generation_dir(&normalized_name);
            let mut deleted = false;
            match fs::remove_dir_all(generation) {
                Ok(()) => deleted = true,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
            Ok(store.remove_legacy_checkpoint(&normalized_name)? || deleted)
        })
    }

    /// 在启动期和每次 checkpoint 操作前恢复未完成的 live rollback。
    pub(super) fn recover(&self) -> io::Result<()> {
        self.with_locked(|_| Ok(()))
    }

    fn with_locked<T>(&self, operation: impl FnOnce(&Self) -> io::Result<T>) -> io::Result<T> {
        with_checkpoint_lock(&self.dir, || {
            self.recover_live_rollbacks_unlocked()?;
            self.recover_incomplete_generations_unlocked()?;
            operation(self)
        })
    }

    fn list_unlocked(&self) -> io::Result<Vec<CheckpointInfo>> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(v) => v,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(err),
        };
        let mut out = Vec::new();
        let mut legacy_entries = Vec::new();
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let metadata = match entry.metadata() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if metadata.is_dir() && generation_is_ready(&path) {
                let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                    continue;
                };
                out.push(CheckpointInfo {
                    name: name.to_string(),
                    modified_local: metadata.modified().ok().map(DateTime::<Local>::from),
                    size_bytes: directory_storage_size(&path)?,
                });
            } else if path.extension().and_then(|s| s.to_str()) == Some("sqlite") {
                legacy_entries.push((path, metadata));
            }
        }
        for (path, metadata) in legacy_entries {
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if out.iter().any(|checkpoint| checkpoint.name == stem) {
                continue;
            }
            out.push(CheckpointInfo {
                name: stem.to_string(),
                modified_local: metadata.modified().ok().map(DateTime::<Local>::from),
                size_bytes: checkpoint_storage_size(
                    &path,
                    &self.legacy_checkpoint_assets_path(stem),
                )?,
            });
        }
        out.sort_by(|a, b| b.modified_local.cmp(&a.modified_local));
        Ok(out)
    }

    fn checkpoint_source_unlocked(&self, name: &str) -> io::Result<Option<CheckpointSource>> {
        let generation = self.generation_dir(name);
        if generation_is_ready(&generation) {
            return Ok(Some(CheckpointSource::Generation {
                sqlite: generation.join(GENERATION_SQLITE_FILE),
                assets: generation.join(GENERATION_ASSETS_DIR),
            }));
        }
        let legacy = self.legacy_checkpoint_path(name);
        if legacy.exists() {
            return Ok(Some(CheckpointSource::Legacy { sqlite: legacy }));
        }
        Ok(None)
    }

    fn checkpoint_storage_size_for_name(&self, name: &str) -> io::Result<u64> {
        let generation = self.generation_dir(name);
        let mut size = if generation_is_ready(&generation) {
            directory_storage_size(&generation)?
        } else {
            0
        };
        size = size.saturating_add(checkpoint_storage_size(
            &self.legacy_checkpoint_path(name),
            &self.legacy_checkpoint_assets_path(name),
        )?);
        Ok(size)
    }

    fn stage_generation(&self, name: &str) -> io::Result<PathBuf> {
        fs::create_dir_all(&self.dir)?;
        let staged = self.dir.join(format!(
            ".{name}{GENERATION_STAGE_MARKER}{}",
            uuid::Uuid::new_v4()
        ));
        let result = (|| {
            fs::create_dir_all(&staged)?;
            backup_sqlite(&self.session_file, &staged.join(GENERATION_SQLITE_FILE))?;
            copy_directory_snapshot(&self.session_assets, &staged.join(GENERATION_ASSETS_DIR))?;
            fs::write(staged.join(GENERATION_MANIFEST_FILE), b"checkpoint-v1\n")?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&staged);
        }
        result.map(|()| staged)
    }

    fn publish_generation(&self, name: &str, staged: &Path) -> io::Result<()> {
        let destination = self.generation_dir(name);
        let previous = self.dir.join(format!(
            ".{name}{GENERATION_PREVIOUS_MARKER}{}",
            uuid::Uuid::new_v4()
        ));
        let had_previous = destination.exists();
        if had_previous {
            fs::rename(&destination, &previous)?;
        }
        if let Err(error) = fs::rename(staged, &destination) {
            if had_previous {
                let _ = fs::rename(&previous, &destination);
            }
            return Err(error);
        }
        if had_previous {
            let _ = fs::remove_dir_all(previous);
        }
        // 新 generation 已成为唯一真相；旧布局仅作兼容读取，后续清理失败不会破坏
        // 已发布的 checkpoint，下次持锁操作会再次回收。
        let _ = self.remove_legacy_checkpoint(name);
        Ok(())
    }

    fn remove_legacy_checkpoint(&self, name: &str) -> io::Result<bool> {
        let path = self.legacy_checkpoint_path(name);
        let assets = self.legacy_checkpoint_assets_path(name);
        let mut deleted = false;
        match fs::remove_file(&path) {
            Ok(()) => deleted = true,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }
        for suffix in SQLITE_SIDECAR_SUFFIXES {
            let sidecar = PathBuf::from(format!("{}{}", path.display(), suffix));
            match fs::remove_file(sidecar) {
                Ok(()) => deleted = true,
                Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                Err(err) => return Err(err),
            }
        }
        match fs::remove_dir_all(&assets) {
            Ok(()) => deleted = true,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }
        Ok(deleted)
    }

    fn stage_live_rollback(&self, sqlite: &Path, assets: &Path) -> io::Result<PathBuf> {
        fs::create_dir_all(&self.dir)?;
        let transaction = self
            .dir
            .join(format!("{LIVE_ROLLBACK_PREFIX}{}", uuid::Uuid::new_v4()));
        let result = (|| {
            let new_state = transaction.join("new");
            fs::create_dir_all(&new_state)?;
            backup_sqlite(sqlite, &new_state.join(GENERATION_SQLITE_FILE))?;
            copy_directory_snapshot(assets, &new_state.join(GENERATION_ASSETS_DIR))?;
            fs::write(transaction.join(LIVE_ROLLBACK_MANIFEST), b"commit\n")?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&transaction);
        }
        result.map(|()| transaction)
    }

    fn complete_live_rollback(&self, transaction: &Path) -> io::Result<()> {
        let new_state = transaction.join("new");
        let sqlite = new_state.join(GENERATION_SQLITE_FILE);
        let assets = new_state.join(GENERATION_ASSETS_DIR);
        if !transaction.join(LIVE_ROLLBACK_MANIFEST).is_file()
            || !sqlite.is_file()
            || !assets.is_dir()
        {
            return Err(io::Error::other(format!(
                "incomplete checkpoint rollback transaction: {}",
                transaction.display()
            )));
        }
        if let Some(parent) = self.session_file.parent() {
            fs::create_dir_all(parent)?;
        }
        backup_sqlite(&sqlite, &self.session_file)?;
        replace_live_assets_from_transaction(transaction, &assets, &self.session_assets)?;
        fs::remove_dir_all(transaction)?;
        Ok(())
    }

    fn recover_live_rollbacks_unlocked(&self) -> io::Result<()> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if !entry.file_type()?.is_dir() || !name.starts_with(LIVE_ROLLBACK_PREFIX) {
                continue;
            }
            if path.join(LIVE_ROLLBACK_MANIFEST).is_file() {
                self.complete_live_rollback(&path)?;
            } else {
                fs::remove_dir_all(path)?;
            }
        }
        Ok(())
    }

    fn recover_incomplete_generations_unlocked(&self) -> io::Result<()> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries.collect::<Result<Vec<_>, _>>()?,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        let mut staged = Vec::new();
        let mut previous = Vec::new();
        for entry in &entries {
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if let Some(base) = staged_generation_base(name, GENERATION_STAGE_MARKER) {
                staged.push((base.to_string(), entry.path()));
            } else if let Some(base) = staged_generation_base(name, GENERATION_PREVIOUS_MARKER) {
                previous.push((base.to_string(), entry.path()));
            }
        }

        for (name, stage) in staged {
            if !generation_is_ready(&stage) {
                fs::remove_dir_all(stage)?;
                continue;
            }
            let destination = self.generation_dir(&name);
            if destination.exists() {
                fs::remove_dir_all(stage)?;
                continue;
            }
            fs::rename(&stage, &destination)?;
            if let Some(index) = previous
                .iter()
                .position(|(previous_name, _)| previous_name == &name)
            {
                let (_, old) = previous.swap_remove(index);
                let _ = fs::remove_dir_all(old);
            }
        }

        for (name, old) in previous {
            let destination = self.generation_dir(&name);
            if destination.exists() {
                fs::remove_dir_all(old)?;
            } else {
                fs::rename(old, destination)?;
            }
        }

        // generation 已发布时，遗留的旧 `.sqlite` / `.assets` 不再是可恢复状态，
        // 必须回收并避免其绕开配额统计。
        for entry in entries {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("sqlite") {
                continue;
            }
            let Some(name) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };
            if generation_is_ready(&self.generation_dir(name)) {
                self.remove_legacy_checkpoint(name)?;
            }
        }
        Ok(())
    }
}

/// 对同一 session 的 checkpoint 事务持有进程内与跨进程排他锁。
pub(super) fn with_checkpoint_lock<T>(
    checkpoint_dir: &Path,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    let lock_dir = checkpoint_lock_dir(checkpoint_dir);
    with_checkpoint_root_shared_lock(&lock_dir, || {
        with_session_checkpoint_lock(&lock_dir, checkpoint_dir, operation)
    })
}

/// 清空全部 session 时独占 checkpoint 根目录，避免与保存、回滚、fork 或归档交错。
pub(super) fn with_checkpoint_root_exclusive_lock<T>(
    checkpoints_root: &Path,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    with_checkpoint_root_exclusive_lock_impl(
        &checkpoint_lock_dir_from_root(checkpoints_root),
        operation,
    )
}

/// 锁文件放在 checkpoint 根目录的同级，清理 snapshot 目录时不会 unlink 正在持有的
/// lock inode。
fn checkpoint_lock_dir(checkpoint_dir: &Path) -> PathBuf {
    let checkpoints_root = checkpoint_dir.parent().unwrap_or_else(|| Path::new("."));
    checkpoint_lock_dir_from_root(checkpoints_root)
}

fn checkpoint_lock_dir_from_root(checkpoints_root: &Path) -> PathBuf {
    checkpoints_root
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".checkpoint-locks")
}

fn with_session_checkpoint_lock<T>(
    lock_dir: &Path,
    checkpoint_dir: &Path,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    fs::create_dir_all(lock_dir)?;
    let name = checkpoint_dir
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("session");
    let lock_path = lock_dir.join(format!("{name}.lock"));
    let local_lock = {
        let mut locks = CHECKPOINT_SESSION_LOCKS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Arc::clone(
            locks
                .entry(lock_path.clone())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        )
    };
    let guard = local_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let file = fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(lock_path)?;

    #[cfg(unix)]
    unsafe {
        if flock(file.as_raw_fd(), LOCK_EX) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    let result = operation();
    #[cfg(unix)]
    unsafe {
        let _ = flock(file.as_raw_fd(), LOCK_UN);
    }
    drop(file);
    drop(guard);
    result
}

/// 同时访问多个 session 时按规范化目录顺序取得锁，避免两个方向相反的 fork
/// 互相等待。重复目录会去重，因此退化为同一 session 时也不会重入 Mutex。
pub(super) fn with_checkpoint_locks<T>(
    checkpoint_dirs: &[&Path],
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    let mut dirs = checkpoint_dirs
        .iter()
        .map(|dir| (*dir).to_path_buf())
        .collect::<Vec<_>>();
    dirs.sort();
    dirs.dedup();
    let Some(first) = dirs.first() else {
        return operation();
    };
    let lock_dir = checkpoint_lock_dir(first);
    if dirs
        .iter()
        .any(|checkpoint_dir| checkpoint_lock_dir(checkpoint_dir) != lock_dir)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "checkpoint sessions must share one root directory",
        ));
    }
    with_checkpoint_root_shared_lock(&lock_dir, || {
        with_checkpoint_locks_inner(&dirs, &lock_dir, &mut Some(operation))
    })
}

fn with_checkpoint_locks_inner<T, F>(
    checkpoint_dirs: &[PathBuf],
    lock_dir: &Path,
    operation: &mut Option<F>,
) -> io::Result<T>
where
    F: FnOnce() -> io::Result<T>,
{
    let Some((checkpoint_dir, remaining)) = checkpoint_dirs.split_first() else {
        return operation
            .take()
            .expect("checkpoint lock operation must run once")();
    };
    with_session_checkpoint_lock(lock_dir, checkpoint_dir, || {
        with_checkpoint_locks_inner(remaining, lock_dir, operation)
    })
}

fn with_checkpoint_root_shared_lock<T>(
    lock_dir: &Path,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    let lock_path = lock_dir.join(".all.lock");
    let local_lock = checkpoint_root_local_lock(&lock_path);
    let guard = local_lock
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let result = with_checkpoint_root_file_lock(&lock_path, false, operation);
    drop(guard);
    result
}

fn with_checkpoint_root_exclusive_lock_impl<T>(
    lock_dir: &Path,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    let lock_path = lock_dir.join(".all.lock");
    let local_lock = checkpoint_root_local_lock(&lock_path);
    let guard = local_lock
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let result = with_checkpoint_root_file_lock(&lock_path, true, operation);
    drop(guard);
    result
}

fn checkpoint_root_local_lock(lock_path: &Path) -> Arc<RwLock<()>> {
    let mut locks = CHECKPOINT_ROOT_LOCKS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    Arc::clone(
        locks
            .entry(lock_path.to_path_buf())
            .or_insert_with(|| Arc::new(RwLock::new(()))),
    )
}

fn with_checkpoint_root_file_lock<T>(
    lock_path: &Path,
    exclusive: bool,
    operation: impl FnOnce() -> io::Result<T>,
) -> io::Result<T> {
    let parent = lock_path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file = fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(lock_path)?;
    #[cfg(unix)]
    unsafe {
        let lock_mode = if exclusive { LOCK_EX } else { LOCK_SH };
        if flock(file.as_raw_fd(), lock_mode) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    let result = operation();
    #[cfg(unix)]
    unsafe {
        let _ = flock(file.as_raw_fd(), LOCK_UN);
    }
    result
}

fn validate_checkpoint_budget(
    current_count: usize,
    replacing_existing: bool,
    current_size: u64,
    replaced_size: u64,
    snapshot_size: u64,
    max_count: usize,
    max_size: u64,
) -> io::Result<()> {
    if !replacing_existing && current_count >= max_count {
        return Err(io::Error::new(
            io::ErrorKind::StorageFull,
            format!(
                "checkpoint limit reached ({max_count} per session); delete an old checkpoint or save with an existing name to replace it"
            ),
        ));
    }

    let projected_size = current_size
        .saturating_sub(replaced_size)
        .saturating_add(snapshot_size);
    if projected_size > max_size {
        return Err(io::Error::new(
            io::ErrorKind::StorageFull,
            format!(
                "checkpoint storage limit reached ({} MiB per session; projected {} bytes); delete old checkpoints before saving",
                max_size / (1024 * 1024),
                projected_size,
            ),
        ));
    }
    Ok(())
}

fn checkpoint_storage_size(sqlite: &Path, assets: &Path) -> io::Result<u64> {
    Ok(sqlite_storage_size(sqlite)?.saturating_add(directory_storage_size(assets)?))
}

fn sqlite_storage_size(path: &Path) -> io::Result<u64> {
    let mut size = match fs::metadata(path) {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => 0,
        Err(error) => return Err(error),
    };
    for suffix in SQLITE_SIDECAR_SUFFIXES {
        let sidecar = PathBuf::from(format!("{}{}", path.display(), suffix));
        match fs::metadata(sidecar) {
            Ok(metadata) => size = size.saturating_add(metadata.len()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(size)
}

fn directory_storage_size(path: &Path) -> io::Result<u64> {
    let entries = match fs::read_dir(path) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    let mut size = 0u64;
    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            size = size.saturating_add(directory_storage_size(&entry.path())?);
        } else {
            size = size.saturating_add(entry.metadata()?.len());
        }
    }
    Ok(size)
}

fn generation_is_ready(path: &Path) -> bool {
    path.join(GENERATION_MANIFEST_FILE).is_file()
        && path.join(GENERATION_SQLITE_FILE).is_file()
        && path.join(GENERATION_ASSETS_DIR).is_dir()
}

fn staged_generation_base<'a>(name: &'a str, marker: &str) -> Option<&'a str> {
    name.strip_prefix('.')
        .and_then(|trimmed| trimmed.split_once(marker).map(|(base, _)| base))
        .filter(|base| !base.is_empty())
}

fn copy_directory_snapshot(source: &Path, destination: &Path) -> io::Result<()> {
    if destination.exists() {
        fs::remove_dir_all(destination)?;
    }
    fs::create_dir_all(destination)?;
    if source.is_dir() {
        copy_dir_recursively(source, destination)?;
    }
    Ok(())
}

/// 从事务内不可变快照替换 live assets。旧目录与发布副本都在事务目录中，
/// 因此异常退出时恢复器仍可依据 `new/` 完成同一个 rollback。
fn replace_live_assets_from_transaction(
    transaction: &Path,
    source: &Path,
    destination: &Path,
) -> io::Result<()> {
    let staged = transaction.join("assets-publish");
    let previous = transaction.join("assets-previous");
    copy_directory_snapshot(source, &staged)?;
    if previous.exists() {
        fs::remove_dir_all(&previous)?;
    }
    if destination.exists() {
        fs::rename(destination, &previous)?;
    }
    if let Err(error) = fs::rename(&staged, destination) {
        if previous.exists() && !destination.exists() {
            let _ = fs::rename(&previous, destination);
        }
        return Err(error);
    }
    Ok(())
}

/// 把检查点名规整为安全的文件名（字母数字、`-`、`_`，其余转 `_`）。
fn sanitize_name(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else if ch.is_whitespace() || ch == '.' || ch == '/' {
            out.push('_');
        }
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        "checkpoint".to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::history::{Message, append_history_messages, build_message_arr};
    use serde_json::Value;
    use std::sync::{Arc, Barrier};

    fn temp_history_file() -> PathBuf {
        let mut p = std::env::temp_dir();
        let unique = format!(
            "ai_ckpt_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        p.push(unique);
        p.push("history.sqlite");
        p
    }

    #[test]
    fn sanitize_name_handles_unsafe_chars() {
        assert_eq!(sanitize_name("my checkpoint"), "my_checkpoint");
        assert_eq!(sanitize_name("../../etc/passwd"), "etc_passwd");
        assert_eq!(sanitize_name(""), "checkpoint");
        assert_eq!(sanitize_name("v1.2"), "v1_2");
    }

    #[test]
    fn checkpoint_budget_rejects_over_limit_without_replacing_existing_entry() {
        let count_error = validate_checkpoint_budget(2, false, 10, 0, 1, 2, 100).unwrap_err();
        assert_eq!(count_error.kind(), io::ErrorKind::StorageFull);

        validate_checkpoint_budget(2, true, 10, 5, 6, 2, 100).unwrap();

        let storage_error = validate_checkpoint_budget(1, false, 90, 10, 21, 2, 100).unwrap_err();
        assert_eq!(storage_error.kind(), io::ErrorKind::StorageFull);
    }

    #[test]
    fn checkpoint_save_recovers_complete_staged_generation() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-stage");
        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("staged history".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        let staged = store.dir.join(".stable.stage-interrupted");
        fs::create_dir_all(staged.join(GENERATION_ASSETS_DIR)).unwrap();
        backup_sqlite(&store.session_file, &staged.join(GENERATION_SQLITE_FILE)).unwrap();
        fs::write(staged.join(GENERATION_MANIFEST_FILE), b"checkpoint-v1\n").unwrap();

        let listed = store.list().unwrap();

        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "stable");
        assert!(!staged.exists());
        assert!(store.checkpoint_path("stable").exists());
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn checkpoint_list_reclaims_incomplete_staged_generation() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-stage-cleanup");
        let staged = store.dir.join(".orphan.stage-interrupted");
        fs::create_dir_all(&staged).unwrap();
        fs::write(staged.join("partial-assets.bin"), vec![0u8; 64 * 1024]).unwrap();

        assert!(store.list().unwrap().is_empty());
        assert!(!staged.exists());
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn concurrent_checkpoint_saves_respect_session_limit() {
        let history_file = temp_history_file();
        let store = Arc::new(CheckpointStore::new(&history_file, "sess-concurrent"));
        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("source".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();

        let barrier = Arc::new(Barrier::new(MAX_CHECKPOINTS_PER_SESSION + 1));
        let mut tasks = Vec::new();
        for index in 0..=MAX_CHECKPOINTS_PER_SESSION {
            let store = Arc::clone(&store);
            let barrier = Arc::clone(&barrier);
            tasks.push(std::thread::spawn(move || {
                barrier.wait();
                store.save(&format!("checkpoint-{index}")).is_ok()
            }));
        }
        let saved = tasks
            .into_iter()
            .filter_map(|task| task.join().ok())
            .filter(|saved| *saved)
            .count();

        assert_eq!(saved, MAX_CHECKPOINTS_PER_SESSION);
        assert_eq!(store.list().unwrap().len(), MAX_CHECKPOINTS_PER_SESSION);
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn save_list_rollback_delete_roundtrip() {
        let history_file = temp_history_file();
        let session_id = "sess-abc";
        let store = CheckpointStore::new(&history_file, session_id);

        // 还没有 session 文件时 save 应失败。
        assert!(store.save("c1").is_err());

        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("v1".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();

        // 保存检查点 c1。
        let ckpt = store.save("c1").unwrap();
        assert!(ckpt.exists());

        // 修改 live 历史为 "v2"。
        append_history_messages(
            &store.session_file,
            &[Message {
                role: "assistant".to_string(),
                content: Value::String("v2".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();

        // list 能看到 c1。
        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "c1");

        // 回滚到 c1，live 历史应恢复为只有 "v1"。
        store.rollback("c1").unwrap();
        let restored = build_message_arr(10, &store.session_file).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].content, Value::String("v1".to_string()));

        // 回滚不存在的检查点应报错。
        assert!(store.rollback("nope").is_err());

        // 删除 c1。
        assert!(store.delete("c1").unwrap());
        assert!(!store.delete("c1").unwrap());
        assert!(store.list().unwrap().is_empty());

        // 清理。
        if let Some(grandparent) = history_file.parent() {
            let _ = fs::remove_dir_all(grandparent);
        }
    }

    #[test]
    fn checkpoint_restores_wal_history_and_context_assets() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-assets");
        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("before checkpoint".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        let checkpoint_asset = store
            .session_assets
            .join("context-checkpoints")
            .join("durable.md");
        fs::create_dir_all(checkpoint_asset.parent().unwrap()).unwrap();
        fs::write(&checkpoint_asset, "before asset").unwrap();

        store.save("stable").unwrap();
        assert!(store.checkpoint_assets_path("stable").exists());
        let listed = store.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert!(
            listed[0].size_bytes
                >= fs::metadata(store.checkpoint_path("stable")).unwrap().len()
                    + "before asset".len() as u64
        );

        append_history_messages(
            &store.session_file,
            &[Message {
                role: "assistant".to_string(),
                content: Value::String("after checkpoint".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        fs::write(&checkpoint_asset, "after asset").unwrap();

        store.rollback("stable").unwrap();
        let restored = build_message_arr(10, &store.session_file).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(
            restored[0].content,
            Value::String("before checkpoint".to_string())
        );
        assert_eq!(
            fs::read_to_string(checkpoint_asset).unwrap(),
            "before asset"
        );

        assert!(store.delete("stable").unwrap());
        assert!(!store.checkpoint_assets_path("stable").exists());
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn legacy_sqlite_only_checkpoint_keeps_current_assets() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-legacy");
        let legacy_path = store.legacy_checkpoint_path("legacy");
        append_history_messages(
            &legacy_path,
            &[Message {
                role: "user".to_string(),
                content: Value::String("legacy history".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        let live_asset = store
            .session_assets
            .join("context-checkpoints")
            .join("live.md");
        fs::create_dir_all(live_asset.parent().unwrap()).unwrap();
        fs::write(&live_asset, "current asset").unwrap();

        store.rollback("legacy").unwrap();

        assert_eq!(
            build_message_arr(10, &store.session_file).unwrap()[0].content,
            Value::String("legacy history".to_string())
        );
        assert_eq!(fs::read_to_string(live_asset).unwrap(), "current asset");
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    #[test]
    fn startup_recovery_completes_interrupted_live_rollback() {
        let history_file = temp_history_file();
        let store = CheckpointStore::new(&history_file, "sess-rollback-recovery");
        append_history_messages(
            &store.session_file,
            &[Message {
                role: "user".to_string(),
                content: Value::String("before".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        let live_asset = store
            .session_assets
            .join("context-checkpoints")
            .join("state.md");
        fs::create_dir_all(live_asset.parent().unwrap()).unwrap();
        fs::write(&live_asset, "before asset").unwrap();
        store.save("stable").unwrap();

        append_history_messages(
            &store.session_file,
            &[Message {
                role: "assistant".to_string(),
                content: Value::String("after".to_string()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
        )
        .unwrap();
        fs::write(&live_asset, "after asset").unwrap();

        let transaction = store
            .stage_live_rollback(
                &store.checkpoint_path("stable"),
                &store.checkpoint_assets_path("stable"),
            )
            .unwrap();
        assert!(transaction.exists());

        store.recover().unwrap();

        let restored = build_message_arr(10, &store.session_file).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].content, Value::String("before".to_string()));
        assert_eq!(fs::read_to_string(live_asset).unwrap(), "before asset");
        assert!(!transaction.exists());
        if let Some(root) = history_file.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }
}
