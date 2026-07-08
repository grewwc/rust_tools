//! Session PID 注册：每个 `a` 进程启动时在 sessions 目录下写入
//! `<session_id>.pid` 文件，退出时自动删除。
//!
//! `/proc` 命令通过扫描这些文件来发现所有正在运行的 session
//! （前台 + `a -bg` 后台），而不是仅依赖 cwd 下的 `*.pid`（只有 `-bg` 才写）。
//!
//! 设计要点：
//! - 使用 Drop guard 确保正常退出 / panic 时文件被清理。
//! - 即使进程被 SIGKILL 杀死（Drop 不会执行），`/proc` 也会通过
//!   PID 存活探测清理残留文件。
//! - 文件内容仅为 PID 的十进制文本，与 `a -bg` 的 cwd PID 文件格式一致。

use std::fs;
use std::io;
use std::path::PathBuf;

/// 写入并管理 sessions 目录下的 `<session_id>.pid` 文件。
/// 创建时写入 PID，Drop 时删除文件。
pub(in crate::ai) struct SessionPidGuard {
    path: Option<PathBuf>,
}

impl SessionPidGuard {
    /// 在 `sessions_root` 目录下写入 `<session_id>.pid`，内容为当前进程 PID。
    /// 如果写入失败只打印警告，不阻断启动。
    pub(in crate::ai) fn register(sessions_root: &std::path::Path, session_id: &str) -> Self {
        let safe_id: String = session_id
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        if safe_id.is_empty() {
            return Self { path: None };
        }
        let path = sessions_root.join(format!("{safe_id}.pid"));
        let pid = std::process::id();
        match fs::write(&path, pid.to_string()) {
            Ok(()) => Self { path: Some(path) },
            Err(err) => {
                eprintln!("[Warning] 无法写入 session PID 文件 ({}): {err}", path.display());
                Self { path: None }
            }
        }
    }
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
    entries.flatten()
        .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("sqlite"))
}

/// 扫描 sessions 目录下的所有 `*.pid` 文件，返回 (session_id, pid, alive) 列表。
/// 会自动清理 PID 已死的残留文件。
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
        found.push((stem.to_string(), pid, alive));
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

    // 去重：同一个 session_id 可能在多个目录出现，保留 alive 的
    all.sort_by(|a, b| a.0.cmp(&b.0));
    all.dedup_by(|a, b| a.0 == b.0);
    Ok(all)
}

/// 通过 `lsof` 扫描 sessions 目录，发现正在使用 `.sqlite` 文件的进程。
///
/// 这是 PID 文件机制的**兜底**：旧版本 `a` 启动的 session 不会写 PID 文件，
/// 但如果该 session 正在读写 history（SQLite 连接打开中），`lsof` 能抓到。
/// 对于空闲等待输入的旧版本 session，此方法可能漏报。
///
/// 返回 (session_id, pid) 列表，已按 session_id 去重。
pub(in crate::ai) fn discover_lsof_sessions(
    sessions_root: &std::path::Path,
) -> Vec<(String, i32)> {
    // 扫描基目录（覆盖所有 persona / config 的 sessions 目录）
    let base = resolve_sessions_base(sessions_root);
    discover_lsof_in_dir(&base)
}

/// 对单个目录运行 `lsof +D`，解析输出。
fn discover_lsof_in_dir(dir: &std::path::Path) -> Vec<(String, i32)> {
    let output = match std::process::Command::new("lsof")
        .arg("+D")
        .arg(dir)
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
pub(in crate::ai) fn process_has_tty(pid: i32) -> bool {
    let output = std::process::Command::new("ps")
        .arg("-o")
        .arg("tt=")
        .arg("-p")
        .arg(pid.to_string())
        .output();
    match output {
        Ok(o) => {
            let tt = String::from_utf8_lossy(&o.stdout).trim().to_string();
            !tt.is_empty() && tt != "??"
        }
        Err(_) => false,
    }
}

/// 统计当前正在运行的 `a` 进程数量（通过 `pgrep -x a`）。
/// 用于在 `/proc` 输出中提示"有 N 个 a 进程在跑，但只识别出 M 个 session"。
pub(in crate::ai) fn count_a_processes() -> usize {
    // pgrep -x a：精确匹配进程名为 "a" 的进程
    let output = std::process::Command::new("pgrep")
        .arg("-x")
        .arg("a")
        .output();
    match output {
        Ok(o) => {
            let text = String::from_utf8_lossy(&o.stdout);
            text.lines().filter(|l| !l.trim().is_empty()).count()
        }
        Err(_) => {
            // pgrep 不可用时，回退到 ps
            let alt = std::process::Command::new("ps")
                .arg("-eo")
                .arg("comm=")
                .output();
            match alt {
                Ok(o) => {
                    let text = String::from_utf8_lossy(&o.stdout);
                    text.lines().map(|l| l.trim()).filter(|l| *l == "a").count()
                }
                Err(_) => 0,
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
        let dir = std::env::temp_dir().join(format!("rust-tools-pid-guard-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let sid = "test-session-001";

        {
            let _guard = SessionPidGuard::register(&dir, sid);
            let pid_path = dir.join(format!("{sid}.pid"));
            assert!(pid_path.exists(), "PID file should exist while guard is alive");
            let content = fs::read_to_string(&pid_path).unwrap();
            let pid: i32 = content.trim().parse().unwrap();
            assert_eq!(pid as u32, std::process::id());
        }

        // Drop 后文件应被删除
        let pid_path = dir.join(format!("{sid}.pid"));
        assert!(!pid_path.exists(), "PID file should be removed after drop");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn scan_finds_registered_pids() {
        let dir = std::env::temp_dir().join(format!("rust-tools-pid-scan-{}", uuid::Uuid::new_v4()));
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
    fn scan_all_finds_pids_across_subdirectories() {
        // 模拟真实布局：base/ 下有 .pid 文件，base/persona.sessions/ 下也有 .pid 文件
        let base = std::env::temp_dir().join(format!(
            "rust-tools-pid-scanall-{}",
            uuid::Uuid::new_v4()
        ));
        let sub = base.join("persona-x.sessions");
        fs::create_dir_all(&sub).unwrap();

        // 在 base 写一个 PID 文件
        let _g1 = SessionPidGuard::register(&base, "session-top");
        // 在子目录写一个 PID 文件
        let _g2 = SessionPidGuard::register(&sub, "session-sub");

        // 从子目录视角调用 scan_all_session_pids，应发现两个目录的 PID 文件
        let results = scan_all_session_pids(&sub).unwrap();
        let ids: Vec<&str> = results.iter().map(|(id, _, _)| id.as_str()).collect();
        assert!(ids.contains(&"session-top"), "should find PID in parent dir");
        assert!(ids.contains(&"session-sub"), "should find PID in sub dir");

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn resolve_sessions_base_finds_parent() {
        let base = std::env::temp_dir().join(format!(
            "rust-tools-pid-base-{}",
            uuid::Uuid::new_v4()
        ));
        let sub = base.join("persona.sessions");
        fs::create_dir_all(&sub).unwrap();
        // 在 base 放一个 .sqlite 文件，让 dir_has_sqlite_files 返回 true
        fs::write(base.join("dummy.sqlite"), "").unwrap();

        let resolved = resolve_sessions_base(&sub);
        assert_eq!(resolved, base, "should resolve to parent when sub has no .sessions subdirs");

        let _ = fs::remove_dir_all(&base);
    }
}
