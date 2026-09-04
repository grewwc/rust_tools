//! Session PID registration: every `a` process writes a `<session_id>.<pid>.pid`
//! file under the sessions directory at startup and removes it on exit.
//!
//! The `/proc` command discovers all running sessions by scanning these files
//! (foreground + `a -bg` background), instead of relying only on the cwd `*.pid`
//! files (written only by `-bg`).
//!
//! Design notes:
//! - A Drop guard cleans up the file on normal exit / panic.
//! - Even if the process is SIGKILLed (Drop never runs), `/proc` also cleans
//!   up leftover files via a PID-liveness probe.
//! - File content is the decimal PID text only, matching the cwd PID file
//!   format of `a -bg`.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

/// A compact, cross-process view of one live subagent.
///
/// The owning `a` process writes these next to its session PID marker so a
/// separate `a /proc` invocation can inspect subagents without reaching into
/// another process's in-memory task registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(in crate::ai) struct AgentSnapshot {
    pub(in crate::ai) agent_name: String,
    pub(in crate::ai) description: String,
    pub(in crate::ai) state: String,
    pub(in crate::ai) elapsed_secs: u64,
    pub(in crate::ai) progress: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AgentSnapshotFile {
    process_start_token: String,
    snapshots: Vec<AgentSnapshot>,
}

static OWN_PROCESS_START_TOKEN: OnceLock<Option<String>> = OnceLock::new();

const MAX_AGENT_SNAPSHOTS: usize = 32;
const MAX_AGENT_NAME_CHARS: usize = 64;
const MAX_AGENT_STATE_CHARS: usize = 32;
const MAX_AGENT_DETAIL_CHARS: usize = 512;
const MAX_AGENT_SNAPSHOT_BYTES: u64 = 64 * 1024;

fn truncate_snapshot_field(value: &str, limit: usize) -> String {
    let mut truncated = value.chars().take(limit).collect::<String>();
    if value.chars().nth(limit).is_some() {
        truncated.push('…');
    }
    truncated
}

fn bounded_agent_snapshots(snapshots: &[AgentSnapshot]) -> Vec<AgentSnapshot> {
    snapshots
        .iter()
        .take(MAX_AGENT_SNAPSHOTS)
        .map(|snapshot| AgentSnapshot {
            agent_name: truncate_snapshot_field(&snapshot.agent_name, MAX_AGENT_NAME_CHARS),
            description: truncate_snapshot_field(&snapshot.description, MAX_AGENT_DETAIL_CHARS),
            state: truncate_snapshot_field(&snapshot.state, MAX_AGENT_STATE_CHARS),
            elapsed_secs: snapshot.elapsed_secs,
            progress: snapshot
                .progress
                .as_deref()
                .map(|progress| truncate_snapshot_field(progress, MAX_AGENT_DETAIL_CHARS)),
        })
        .collect()
}

fn agent_snapshots_are_bounded(snapshots: &[AgentSnapshot]) -> bool {
    snapshots.len() <= MAX_AGENT_SNAPSHOTS
        && snapshots.iter().all(|snapshot| {
            snapshot.agent_name.chars().count() <= MAX_AGENT_NAME_CHARS + 1
                && snapshot.description.chars().count() <= MAX_AGENT_DETAIL_CHARS + 1
                && snapshot.state.chars().count() <= MAX_AGENT_STATE_CHARS + 1
                && snapshot
                    .progress
                    .as_ref()
                    .is_none_or(|progress| progress.chars().count() <= MAX_AGENT_DETAIL_CHARS + 1)
        })
}

fn read_bounded_snapshot_file(path: &std::path::Path) -> Option<AgentSnapshotFile> {
    let file = fs::File::open(path).ok()?;
    let metadata = file.metadata().ok()?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_AGENT_SNAPSHOT_BYTES {
        return None;
    }

    let mut content = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_AGENT_SNAPSHOT_BYTES + 1)
        .read_to_end(&mut content)
        .ok()?;
    if content.len() as u64 > MAX_AGENT_SNAPSHOT_BYTES {
        return None;
    }

    let snapshot_file = serde_json::from_slice::<AgentSnapshotFile>(&content).ok()?;
    agent_snapshots_are_bounded(&snapshot_file.snapshots).then_some(snapshot_file)
}

fn process_start_token(pid: i32) -> Option<String> {
    let output = Command::new("ps")
        .args(["-o", "lstart=", "-p", &pid.to_string()])
        .env("LC_ALL", "C")
        .env("LANG", "C")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let token = String::from_utf8(output.stdout).ok()?;
    let token = token.trim();
    (!token.is_empty()).then(|| token.to_string())
}

fn own_process_start_token() -> Option<String> {
    OWN_PROCESS_START_TOKEN
        .get_or_init(|| process_start_token(std::process::id() as i32))
        .clone()
}

fn agent_snapshot_path_for_pid_file(pid_file: &std::path::Path) -> PathBuf {
    pid_file.with_extension("agents.json")
}

fn safe_session_id(session_id: &str) -> String {
    session_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect()
}

fn agent_snapshot_path(sessions_root: &std::path::Path, session_id: &str, pid: u32) -> PathBuf {
    let safe_id = safe_session_id(session_id);
    sessions_root.join(format!("{safe_id}.{pid}.agents.json"))
}

fn snapshot_directories(sessions_root: &std::path::Path) -> Vec<PathBuf> {
    let base = resolve_sessions_base(sessions_root);
    let mut directories = vec![sessions_root.to_path_buf()];
    if base != sessions_root {
        directories.push(base.clone());
    }
    if let Ok(entries) = fs::read_dir(&base) {
        directories.extend(entries.flatten().map(|entry| entry.path()).filter(|path| {
            path.is_dir() && path.extension().and_then(|suffix| suffix.to_str()) == Some("sessions")
        }));
    }
    directories.sort();
    directories.dedup();
    directories
}

fn agent_snapshot_paths(
    sessions_root: &std::path::Path,
    session_id: &str,
    pid: u32,
) -> Vec<PathBuf> {
    let safe_id = safe_session_id(session_id);
    let marker_name = format!("{safe_id}.{pid}.pid");
    let mut paths = vec![agent_snapshot_path(sessions_root, session_id, pid)];
    for directory in snapshot_directories(sessions_root) {
        if directory.join(&marker_name).is_file() {
            paths.push(agent_snapshot_path(&directory, session_id, pid));
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

/// Publish the live subagent view for this process. An empty snapshot clears
/// the sidecar so completed and collected tasks do not appear as active.
pub(in crate::ai) fn clear_agent_snapshots(
    sessions_root: &std::path::Path,
    session_id: &str,
) -> io::Result<()> {
    let mut error = None;
    for path in agent_snapshot_paths(sessions_root, session_id, std::process::id()) {
        if let Err(err) = fs::remove_file(path) {
            if err.kind() != io::ErrorKind::NotFound {
                error = Some(err);
            }
        }
    }
    error.map_or(Ok(()), Err)
}

pub(in crate::ai) fn write_agent_snapshots(
    sessions_root: &std::path::Path,
    session_id: &str,
    snapshots: &[AgentSnapshot],
) -> io::Result<()> {
    let paths = agent_snapshot_paths(sessions_root, session_id, std::process::id());
    if snapshots.is_empty() {
        return clear_agent_snapshots(sessions_root, session_id);
    }

    let process_start_token = own_process_start_token().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::Other,
            "could not determine the current process start time",
        )
    })?;
    let serialized = serde_json::to_vec(&AgentSnapshotFile {
        process_start_token,
        snapshots: bounded_agent_snapshots(snapshots),
    })
    .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    if serialized.len() as u64 > MAX_AGENT_SNAPSHOT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "agent snapshot exceeds the size limit",
        ));
    }
    for path in paths {
        let temporary = path.with_extension("agents.json.tmp");
        fs::write(&temporary, &serialized)?;
        fs::rename(temporary, path)?;
    }
    Ok(())
}

/// Read the snapshots published by an active session PID. The sessions base
/// may contain persona-specific `*.sessions` directories, so search each
/// directory that `scan_all_session_pids` covers.
pub(in crate::ai) fn read_agent_snapshots(
    sessions_root: &std::path::Path,
    session_id: &str,
    pid: i32,
) -> Vec<AgentSnapshot> {
    let Ok(pid) = u32::try_from(pid) else {
        return Vec::new();
    };
    for directory in snapshot_directories(sessions_root) {
        let path = agent_snapshot_path(&directory, session_id, pid);
        let Some(snapshot_file) = read_bounded_snapshot_file(&path) else {
            continue;
        };
        if process_start_token(pid as i32).as_deref()
            == Some(snapshot_file.process_start_token.as_str())
        {
            return snapshot_file.snapshots;
        }
    }
    Vec::new()
}

/// Writes and manages the `<session_id>.<pid>.pid` file under the sessions
/// directory. Writes the PID on creation and removes the file on Drop.
pub(in crate::ai) struct SessionPidGuard {
    path: Option<PathBuf>,
    sessions_root: Option<PathBuf>,
}

impl SessionPidGuard {
    /// Writes `<session_id>.<pid>.pid` under `sessions_root`, containing the current
    /// process PID. On write failure only prints a warning; never blocks startup.
    pub(in crate::ai) fn register(sessions_root: &std::path::Path, session_id: &str) -> Self {
        match mark_session_pid(sessions_root, session_id, false) {
            Ok(path) => Self {
                path: Some(path),
                sessions_root: Some(sessions_root.to_path_buf()),
            },
            Err((path, err)) => {
                eprintln!(
                    "[Warning] 无法写入 session PID 文件 ({}): {err}",
                    path.display()
                );
                Self {
                    path: None,
                    sessions_root: None,
                }
            }
        }
    }
}

/// Additionally registers the current process as an active session.
///
/// Session switches retain old markers until the process exits so prune cannot
/// delete the newly selected session during the transition. `SessionPidGuard`
/// removes every marker for its PID when the owning process exits.
pub(in crate::ai) fn mark_session_pid(
    sessions_root: &std::path::Path,
    session_id: &str,
    require_existing: bool,
) -> Result<PathBuf, (PathBuf, io::Error)> {
    let pid = std::process::id();
    let safe_id: String = session_id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    let path = sessions_root.join(format!("{safe_id}.{pid}.pid"));
    if safe_id.is_empty() {
        return Err((
            path,
            io::Error::new(io::ErrorKind::InvalidInput, "session id 为空"),
        ));
    }
    crate::ai::history::with_sessions_lifecycle_lock(sessions_root, || {
        let history_path = sessions_root.join(format!("{safe_id}.sqlite"));
        if require_existing && !history_path.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("session '{}' no longer exists", session_id),
            ));
        }
        fs::write(&path, pid.to_string())?;
        let _ = fs::remove_file(agent_snapshot_path_for_pid_file(&path));
        Ok(())
    })
    .map(|()| path.clone())
    .map_err(|err| (path, err))
}

impl Drop for SessionPidGuard {
    fn drop(&mut self) {
        if let Some(ref sessions_root) = self.sessions_root {
            cleanup_current_process_markers(sessions_root);
        } else if let Some(ref path) = self.path {
            let _ = fs::remove_file(path);
            let _ = fs::remove_file(agent_snapshot_path_for_pid_file(path));
        }
    }
}

fn cleanup_current_process_markers(sessions_root: &std::path::Path) {
    let base = resolve_sessions_base(sessions_root);
    let mut directories = vec![sessions_root.to_path_buf()];
    if base != sessions_root {
        directories.push(base.clone());
    }
    if let Ok(entries) = fs::read_dir(&base) {
        directories.extend(entries.flatten().map(|entry| entry.path()).filter(|path| {
            path.is_dir() && path.extension().and_then(|suffix| suffix.to_str()) == Some("sessions")
        }));
    }

    let pid_suffix = format!(".{}.pid", std::process::id());
    for directory in directories {
        let Ok(entries) = fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(&pid_suffix))
            {
                let _ = fs::remove_file(&path);
                let _ = fs::remove_file(agent_snapshot_path_for_pid_file(&path));
            }
        }
    }
}

/// 判断哪个目录是"sessions 基目录"——即包含所有 `.sqlite` 文件和
/// `*.sessions/` 子目录的公共父目录。
///
/// 不同 persona / config 的 `a` 进程可能使用不同的 `history_file`，
/// 导致 `sessions_root` 不同（可能是顶层 `~/.xxx.sessions/`，也可能是
/// 其下的 `~/.xxx.sessions/persona.sessions/` 子目录）。
/// 基目录是包含 `*.sessions` 子目录的那一层。
fn resolve_sessions_base(sessions_root: &std::path::Path) -> std::path::PathBuf {
    // 如果 sessions_root 自身包含 *.sessions 子目录，它就是基目录。
    if dir_has_sessions_subdirs(sessions_root) {
        return sessions_root.to_path_buf();
    }
    // 否则检查父目录。
    if let Some(parent) = sessions_root.parent() {
        if dir_has_sessions_subdirs(parent) || dir_has_sqlite_files(parent) {
            return parent.to_path_buf();
        }
    }
    sessions_root.to_path_buf()
}

fn dir_has_sessions_subdirs(dir: &std::path::Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        if entry.path().extension().and_then(|s| s.to_str()) == Some("sessions") {
            return true;
        }
    }
    false
}

fn dir_has_sqlite_files(dir: &std::path::Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    entries
        .flatten()
        .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("sqlite"))
}

/// 扫描 sessions 目录下的所有 `*.pid` 文件，返回 (session_id, pid, alive) 列表。
/// 同时兼容旧版 `<session_id>.pid` 和新版 `<session_id>.<pid>.pid`；会自动清理
/// PID 已死的残留文件。
pub(in crate::ai) fn scan_session_pids(
    sessions_root: &std::path::Path,
) -> io::Result<Vec<(String, i32, bool)>> {
    let entries = match fs::read_dir(sessions_root) {
        Ok(v) => v,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };
    let mut found = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("pid") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let pid: i32 = match content.trim().parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let alive = pid_is_alive(pid);
        if !alive {
            // Clean up leftover PID files
            let _ = fs::remove_file(&path);
            let _ = fs::remove_file(agent_snapshot_path_for_pid_file(&path));
        }
        let session_id = stem
            .rsplit_once('.')
            .filter(|(_, marker_pid)| marker_pid.parse::<i32>().ok() == Some(pid))
            .map_or(stem, |(session_id, _)| session_id);
        found.push((session_id.to_string(), pid, alive));
    }
    // 按 session_id 排序保证输出稳定
    found.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(found)
}

/// 扫描 sessions 基目录及其所有 `*.sessions/` 子目录，汇总所有活跃 PID 文件。
///
/// 不同 persona / config 的 `a` 进程可能使用不同的 `history_file`，
/// 导致 PID 文件分布在不同目录。此函数自动定位基目录并递归扫描，
/// 确保所有活跃 session 都被发现。
pub(in crate::ai) fn scan_all_session_pids(
    sessions_root: &std::path::Path,
) -> io::Result<Vec<(String, i32, bool)>> {
    let base = resolve_sessions_base(sessions_root);
    let mut all = scan_session_pids(&base)?;

    // 扫描基目录下的所有 *.sessions 子目录
    if let Ok(entries) = fs::read_dir(&base) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("sessions") {
                continue;
            }
            if path.is_dir() {
                let sub = scan_session_pids(&path)?;
                all.extend(sub);
            }
        }
    }

    // 去重：同一个 session_id / PID 可能在多个目录出现，保留一份即可；
    // 同一 session 的不同进程必须全部保留。
    all.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    all.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    Ok(all)
}

/// 通过 `lsof` 发现给定进程正在使用的 `.sqlite` 文件，作为 PID 文件机制的兜底。
///
/// 旧版本 `a` 启动的 session 不会写 PID 文件，但如果该 session 正在读写
/// history（SQLite 连接打开中），`lsof` 能抓到。对于空闲等待输入的旧版本
/// session 可能漏报。
///
/// 与旧实现的关键区别：不再用 `lsof +D <sessions_root>` 递归扫描整个 sessions
/// 目录树（条目多时极慢），而是直接查询给定 PID 的文件描述符表
/// （`lsof -p pids -Fpcn`），只看这些进程打开了哪些 `.sqlite`。
///
/// `pids` 应为"未通过 PID 文件登记的 `a` 进程"列表。返回 (session_id, pid)，
/// 已按 session_id 去重。
pub(in crate::ai) fn discover_lsof_sessions(
    _sessions_root: &std::path::Path,
    pids: &[i32],
) -> Vec<(String, i32)> {
    if pids.is_empty() {
        return Vec::new();
    }
    // lsof -p 接受逗号分隔的 PID 列表
    let joined = pids
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let output = match crate::fork_guard::output(
        std::process::Command::new("lsof")
            .arg("-p")
            .arg(&joined)
            .arg("-Fpcn"),
    ) {
        Ok(o) if o.status.success() || !o.stdout.is_empty() => o,
        _ => return Vec::new(),
    };
    let text = String::from_utf8_lossy(&output.stdout);
    let mut found: std::collections::BTreeMap<String, i32> = std::collections::BTreeMap::new();

    // lsof -F 输出格式：每条记录由多个单字母前缀行组成。
    // p=PID, c=command, n=file path。我们关注 n 行中含 .sqlite 的。
    let mut current_pid: Option<i32> = None;
    for line in text.lines() {
        match line.chars().next() {
            Some('p') => {
                current_pid = line[1..].trim().parse().ok();
            }
            Some('n') if line.contains(".sqlite") => {
                // 从文件路径提取 session_id：取文件名去掉 .sqlite 后缀
                // 也匹配 .sqlite-wal / .sqlite-shm
                if let Some(pid) = current_pid {
                    let path = std::path::Path::new(&line[1..]);
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        // .sqlite-wal 的 file_stem 是 "abc.sqlite-wal"，需要再去掉 -wal
                        // 但 .sqlite 的 file_stem 是 "abc"
                        let sid = stem
                            .strip_suffix("-wal")
                            .or_else(|| stem.strip_suffix("-shm"))
                            .unwrap_or(stem);
                        found.entry(sid.to_string()).or_insert(pid);
                    }
                }
            }
            _ => {}
        }
    }
    found.into_iter().collect()
}

/// 检测进程是否有控制终端（tty）。
/// 前台交互式 session 有 tty（如 ttys001），`a -bg` daemon 没有（显示为 `??`）。
///
/// 一次性批量查询多个进程，把 N 次 `ps` 的 fork/exec 降为单次。这是 `/proc`
/// 输出循环里的主要性能瓶颈点：旧实现对每个活跃 session 各跑一次 `ps`。
pub(in crate::ai) fn tty_map_for_pids(pids: &[i32]) -> FxHashMap<i32, bool> {
    let mut map = FxHashMap::default();
    if pids.is_empty() {
        return map;
    }
    // macOS `ps -p` 接受逗号分隔的 PID 列表，一次返回所有进程的 tty。
    let joined = pids
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let output = crate::fork_guard::output(
        std::process::Command::new("ps")
            .arg("-o")
            .arg("pid=,tt=")
            .arg("-p")
            .arg(&joined),
    );
    let Ok(o) = output else {
        return map;
    };
    let text = String::from_utf8_lossy(&o.stdout);
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let Some(pid) = it.next().and_then(|s| s.parse::<i32>().ok()) else {
            continue;
        };
        let tt = it.next().unwrap_or("");
        map.insert(pid, !tt.is_empty() && tt != "??");
    }
    map
}

/// 列出所有正在运行的 `a` 进程的 PID（通过 `pgrep -x a`，回退 `ps`）。
pub(in crate::ai) fn list_a_pids() -> Vec<i32> {
    // pgrep -x a：精确匹配进程名为 "a" 的进程
    let output = crate::fork_guard::output(std::process::Command::new("pgrep").arg("-x").arg("a"));
    match output {
        Ok(o) => {
            let text = String::from_utf8_lossy(&o.stdout);
            text.lines().filter_map(|l| l.trim().parse().ok()).collect()
        }
        Err(_) => {
            // pgrep 不可用时，回退到 ps：输出 pid + comm，过滤 comm=="a"
            let alt = crate::fork_guard::output(
                std::process::Command::new("ps")
                    .arg("-eo")
                    .arg("pid=,comm="),
            );
            match alt {
                Ok(o) => {
                    let text = String::from_utf8_lossy(&o.stdout);
                    text.lines()
                        .filter_map(|l| {
                            let mut it = l.split_whitespace();
                            let pid: i32 = it.next()?.parse().ok()?;
                            let comm = it.next()?;
                            (comm == "a").then_some(pid)
                        })
                        .collect()
                }
                Err(_) => Vec::new(),
            }
        }
    }
}

/// 探测 PID 是否仍然存活（Unix：kill(pid, 0) 不返回 ESRCH 即存活）。
#[cfg(unix)]
fn pid_is_alive(pid: i32) -> bool {
    let ret = unsafe { libc::kill(pid, 0) };
    if ret == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

#[cfg(not(unix))]
fn pid_is_alive(pid: i32) -> bool {
    let _ = pid;
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_writes_and_removes_pid_file() {
        let dir =
            std::env::temp_dir().join(format!("rust-tools-pid-guard-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let sid = "test-session-001";

        {
            let _guard = SessionPidGuard::register(&dir, sid);
            let pid_path = dir.join(format!("{sid}.{}.pid", std::process::id()));
            assert!(
                pid_path.exists(),
                "PID file should exist while guard is alive"
            );
            let content = fs::read_to_string(&pid_path).unwrap();
            let pid: i32 = content.trim().parse().unwrap();
            assert_eq!(pid as u32, std::process::id());
        }

        // Drop 后文件应被删除
        let pid_path = dir.join(format!("{sid}.{}.pid", std::process::id()));
        assert!(!pid_path.exists(), "PID file should be removed after drop");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn agent_snapshot_fields_are_bounded_before_persistence() {
        let snapshots = (0..MAX_AGENT_SNAPSHOTS + 1)
            .map(|_| AgentSnapshot {
                agent_name: "n".repeat(MAX_AGENT_NAME_CHARS + 1),
                description: "d".repeat(MAX_AGENT_DETAIL_CHARS + 1),
                state: "s".repeat(MAX_AGENT_STATE_CHARS + 1),
                elapsed_secs: 0,
                progress: Some("p".repeat(MAX_AGENT_DETAIL_CHARS + 1)),
            })
            .collect::<Vec<_>>();

        let bounded = bounded_agent_snapshots(&snapshots);
        assert_eq!(bounded.len(), MAX_AGENT_SNAPSHOTS);
        assert_eq!(
            bounded[0].agent_name.chars().count(),
            MAX_AGENT_NAME_CHARS + 1
        );
        assert_eq!(
            bounded[0].description.chars().count(),
            MAX_AGENT_DETAIL_CHARS + 1
        );
        assert_eq!(bounded[0].state.chars().count(), MAX_AGENT_STATE_CHARS + 1);
        assert_eq!(
            bounded[0].progress.as_ref().unwrap().chars().count(),
            MAX_AGENT_DETAIL_CHARS + 1
        );
    }

    #[test]
    fn agent_snapshots_round_trip_and_clear() {
        let dir = std::env::temp_dir().join(format!(
            "rust-tools-agent-snapshots-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        let sid = "test-session-001";
        let snapshots = vec![AgentSnapshot {
            agent_name: "researcher".to_string(),
            description: "Inspect /proc output".to_string(),
            state: "running".to_string(),
            elapsed_secs: 42,
            progress: Some("reading source".to_string()),
        }];

        write_agent_snapshots(&dir, sid, &snapshots).unwrap();
        assert_eq!(
            read_agent_snapshots(&dir, sid, std::process::id() as i32),
            snapshots
        );

        clear_agent_snapshots(&dir, sid).unwrap();
        assert!(read_agent_snapshots(&dir, sid, std::process::id() as i32).is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn agent_snapshots_follow_session_markers_across_persona_roots() {
        let base = std::env::temp_dir().join(format!(
            "rust-tools-agent-snapshot-personas-{}",
            uuid::Uuid::new_v4()
        ));
        let default_root = base.join("default.sessions");
        let persona_root = base.join("persona.sessions");
        fs::create_dir_all(&default_root).unwrap();
        fs::create_dir_all(&persona_root).unwrap();
        let sid = "session-default";
        let snapshots = vec![AgentSnapshot {
            agent_name: "researcher".to_string(),
            description: "Continue work after a persona switch".to_string(),
            state: "running".to_string(),
            elapsed_secs: 8,
            progress: None,
        }];

        mark_session_pid(&default_root, sid, false).unwrap();
        write_agent_snapshots(&persona_root, sid, &snapshots).unwrap();
        assert_eq!(
            read_agent_snapshots(&default_root, sid, std::process::id() as i32),
            snapshots
        );

        clear_agent_snapshots(&persona_root, sid).unwrap();
        assert!(read_agent_snapshots(&default_root, sid, std::process::id() as i32).is_empty());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn scan_finds_registered_pids() {
        let dir =
            std::env::temp_dir().join(format!("rust-tools-pid-scan-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();

        let _g1 = SessionPidGuard::register(&dir, "session-a");
        let _g2 = SessionPidGuard::register(&dir, "session-b");

        let results = scan_session_pids(&dir).unwrap();
        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|(id, _, _)| id.as_str()).collect();
        assert!(ids.contains(&"session-a"));
        assert!(ids.contains(&"session-b"));

        // 当前进程的 PID 应标记为存活
        for (_, pid, alive) in &results {
            assert!(*alive, "own PID should be alive");
            assert_eq!(*pid as u32, std::process::id());
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn agent_snapshot_guard_removes_markers_for_sessions_registered_after_startup() {
        let dir = std::env::temp_dir().join(format!(
            "rust-tools-pid-switch-cleanup-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        let pid = std::process::id();
        let snapshots = vec![AgentSnapshot {
            agent_name: "researcher".to_string(),
            description: "Inspect cleanup".to_string(),
            state: "running".to_string(),
            elapsed_secs: 1,
            progress: None,
        }];

        {
            let _guard = SessionPidGuard::register(&dir, "session-first");
            write_agent_snapshots(&dir, "session-first", &snapshots).unwrap();
            mark_session_pid(&dir, "session-second", false).unwrap();
            write_agent_snapshots(&dir, "session-second", &snapshots).unwrap();
        }

        for session_id in ["session-first", "session-second"] {
            assert!(!dir.join(format!("{session_id}.{pid}.pid")).exists());
            assert!(!dir.join(format!("{session_id}.{pid}.agents.json")).exists());
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn per_process_markers_do_not_overwrite_each_other() {
        let dir = std::env::temp_dir().join(format!(
            "rust-tools-pid-multi-process-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        let sid = "shared-session";
        let mut child =
            crate::fork_guard::spawn(std::process::Command::new("sleep").arg("30")).unwrap();
        let child_pid = i32::try_from(child.id()).unwrap();
        let child_path = dir.join(format!("{sid}.{child_pid}.pid"));
        fs::write(&child_path, child_pid.to_string()).unwrap();

        {
            let _guard = SessionPidGuard::register(&dir, sid);
            let results = scan_session_pids(&dir).unwrap();
            assert_eq!(
                results
                    .iter()
                    .filter(|(session_id, _, alive)| session_id == sid && *alive)
                    .count(),
                2,
                "同一 session 的两个活动进程都必须被发现"
            );
        }

        assert!(
            child_path.exists(),
            "一个进程退出时不能删除另一个进程的 PID 标记"
        );
        child.kill().unwrap();
        child.wait().unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan_accepts_legacy_session_pid_file() {
        let dir =
            std::env::temp_dir().join(format!("rust-tools-pid-legacy-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let sid = "legacy-session";
        fs::write(
            dir.join(format!("{sid}.pid")),
            std::process::id().to_string(),
        )
        .unwrap();

        let results = scan_session_pids(&dir).unwrap();
        assert!(results.iter().any(|(session_id, pid, alive)| {
            session_id == sid && *pid as u32 == std::process::id() && *alive
        }));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan_all_finds_pids_across_subdirectories() {
        // 模拟真实布局：base/ 下有 .pid 文件，base/persona.sessions/ 下也有 .pid 文件
        let base =
            std::env::temp_dir().join(format!("rust-tools-pid-scanall-{}", uuid::Uuid::new_v4()));
        let sub = base.join("persona-x.sessions");
        fs::create_dir_all(&sub).unwrap();

        // 在 base 写一个 PID 文件
        let _g1 = SessionPidGuard::register(&base, "session-top");
        // 在子目录写一个 PID 文件
        let _g2 = SessionPidGuard::register(&sub, "session-sub");

        // 从子目录视角调用 scan_all_session_pids，应发现两个目录的 PID 文件
        let results = scan_all_session_pids(&sub).unwrap();
        let ids: Vec<&str> = results.iter().map(|(id, _, _)| id.as_str()).collect();
        assert!(
            ids.contains(&"session-top"),
            "should find PID in parent dir"
        );
        assert!(ids.contains(&"session-sub"), "should find PID in sub dir");

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn resolve_sessions_base_finds_parent() {
        let base =
            std::env::temp_dir().join(format!("rust-tools-pid-base-{}", uuid::Uuid::new_v4()));
        let sub = base.join("persona.sessions");
        fs::create_dir_all(&sub).unwrap();
        // 在 base 放一个 .sqlite 文件，让 dir_has_sqlite_files 返回 true
        fs::write(base.join("dummy.sqlite"), "").unwrap();

        let resolved = resolve_sessions_base(&sub);
        assert_eq!(
            resolved, base,
            "should resolve to parent when sub has no .sessions subdirs"
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn tty_map_batches_multiple_pids_in_one_call() {
        // 空输入直接返回空映射，不应启动 ps。
        assert!(tty_map_for_pids(&[]).is_empty());

        // 当前进程自身一定存活：ps 应返回它。是否拥有 tty 取决于运行环境
        // （cargo test 下通常无控制终端 -> false），故只断言键存在，不断言值。
        let own = std::process::id() as i32;
        let map = tty_map_for_pids(&[own]);
        assert!(map.contains_key(&own), "ps 应当返回当前进程自身的 pid");

        // 逗号分隔的多 PID 仍只发一次 ps，全部存活 PID 都应出现。
        let map = tty_map_for_pids(&[own, own]);
        assert!(map.contains_key(&own));
    }
}
