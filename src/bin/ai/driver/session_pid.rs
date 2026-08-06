//! Session PID 注册：每个 `a` 进程启动时在 sessions 目录下写入
//! `<session_id>.<pid>.pid` 文件，退出时自动删除。
//!
//! `/proc` 命令通过扫描这些文件来发现所有正在运行的 session
//! （前台 + `a -bg` 后台），而不是仅依赖 cwd 下的 `*.pid`（只有 `-bg` 才写）。
//!
//! 设计要点：
//! - 使用 Drop guard 确保正常退出 / panic 时文件被清理。
//! - 即使进程被 SIGKILL 杀死（Drop 不会执行），`/proc` 也会通过
//!   PID 存活探测清理残留文件。
//! - 文件内容仅为 PID 的十进制文本，与 `a -bg` 的 cwd PID 文件格式一致。

use rustc_hash::FxHashMap;
use std::fs;
use std::io;
use std::path::PathBuf;

/// 写入并管理 sessions 目录下的 `<session_id>.<pid>.pid` 文件。
/// 创建时写入 PID，Drop 时删除文件。
pub(in crate::ai) struct SessionPidGuard {
    path: Option<PathBuf>,
}

impl SessionPidGuard {
    /// 在 `sessions_root` 目录下写入 `<session_id>.<pid>.pid`，内容为当前进程 PID。
    /// 如果写入失败只打印警告，不阻断启动。
    pub(in crate::ai) fn register(sessions_root: &std::path::Path, session_id: &str) -> Self {
        match mark_session_pid(sessions_root, session_id, false) {
            Ok(path) => Self { path: Some(path) },
            Err((path, err)) => {
                eprintln!(
                    "[Warning] 无法写入 session PID 文件 ({}): {err}",
                    path.display()
                );
                Self { path: None }
            }
        }
    }
}

/// 为当前进程额外登记一个活跃 session。
///
/// session 切换后保留旧标记是安全的：进程退出后扫描器会清理失效 PID；
/// 更重要的是，prune 不会误删当前已经切换到的新 session。
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
        fs::write(&path, pid.to_string())
    })
    .map(|()| path.clone())
    .map_err(|err| (path, err))
}

impl Drop for SessionPidGuard {
    fn drop(&mut self) {
        if let Some(ref path) = self.path {
            let _ = fs::remove_file(path);
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
            // 清理残留的 PID 文件
            let _ = fs::remove_file(&path);
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
    let output = match std::process::Command::new("lsof")
        .arg("-p")
        .arg(&joined)
        .arg("-Fpcn")
        .output()
    {
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
    let output = std::process::Command::new("ps")
        .arg("-o")
        .arg("pid=,tt=")
        .arg("-p")
        .arg(&joined)
        .output();
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
    let output = std::process::Command::new("pgrep")
        .arg("-x")
        .arg("a")
        .output();
    match output {
        Ok(o) => {
            let text = String::from_utf8_lossy(&o.stdout);
            text.lines().filter_map(|l| l.trim().parse().ok()).collect()
        }
        Err(_) => {
            // pgrep 不可用时，回退到 ps：输出 pid + comm，过滤 comm=="a"
            let alt = std::process::Command::new("ps")
                .arg("-eo")
                .arg("pid=,comm=")
                .output();
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
    fn per_process_markers_do_not_overwrite_each_other() {
        let dir = std::env::temp_dir().join(format!(
            "rust-tools-pid-multi-process-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        let sid = "shared-session";
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .unwrap();
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
