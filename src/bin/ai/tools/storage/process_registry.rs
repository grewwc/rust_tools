// =============================================================================
// 会话级后台进程组注册表（进程内、内存态）
// =============================================================================
// agent 通过 `execute_command` 执行的命令若用 `&` 派生了常驻服务（典型：
// `python app.py &` 启动 Flask），前台命令返回后该服务会成为孤儿进程继续运
// 行。本注册表按 session_id 记录这些残留进程组的 pgid，在会话结束时统一
// `killpg` 清理，避免遗留后台进程。
//
// 关键设计：**只存在内存中，绝不持久化到磁盘。** pgid 仅在当前进程存活期间
// 有意义；一旦本进程退出，同一数值可能被操作系统复用给无关进程，此时按旧
// pgid `killpg` 会误杀。因此不能沿用 `temp_registry` 的文件持久化范式。
//
// 同理也不能用 `runtime_ctx::temp_dir()` 作为键：注册发生在 turn 内
// （DRIVER_CTX 存在），清理发生在 turn 外（会话结束），二者解析出的路径不同。
// 用 session_id 字符串作键可规避该错位。
// =============================================================================

use std::sync::{LazyLock, Mutex};

use rust_tools::commonw::{FastMap, FastSet};

/// session_id -> 该会话登记的后台进程组 pgid 集合。
static REGISTRY: LazyLock<Mutex<FastMap<String, FastSet<u32>>>> =
    LazyLock::new(|| Mutex::new(FastMap::default()));

/// 登记一个后台进程组。`session_id` 为空时忽略（无归属，交由进程退出兜底）。
pub(crate) fn register(session_id: &str, pgid: u32) {
    if session_id.is_empty() || pgid == 0 {
        return;
    }
    let mut guard = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    guard
        .entry(session_id.to_string())
        .or_default()
        .insert(pgid);
}

/// 清理指定会话登记的所有后台进程组：先 `SIGTERM` 给存活组一次优雅退出的机会，
/// 短暂等待后对仍存活的组补 `SIGKILL`。返回实际发出终止信号的进程组数量。
pub(crate) fn kill_session(session_id: &str) -> usize {
    if session_id.is_empty() {
        return 0;
    }
    let pgids: Vec<u32> = {
        let mut guard = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
        match guard.remove(session_id) {
            Some(set) => set.into_iter().collect(),
            None => return 0,
        }
    };
    kill_pgids(&pgids)
}

#[cfg(unix)]
fn kill_pgids(pgids: &[u32]) -> usize {
    let mut signaled = 0usize;
    let mut alive: Vec<i32> = Vec::new();
    for &pgid in pgids {
        let pg = -(pgid as libc::pid_t);
        // 仅对仍存活的进程组计数并发送 SIGTERM。
        if unsafe { libc::kill(pg, 0) } == 0 {
            unsafe {
                let _ = libc::kill(pg, libc::SIGTERM);
            }
            alive.push(pg);
            signaled += 1;
        }
    }
    if alive.is_empty() {
        return 0;
    }
    // 给 SIGTERM 一点时间生效，随后对仍未退出的组补 SIGKILL。
    std::thread::sleep(std::time::Duration::from_millis(200));
    for pg in alive {
        if unsafe { libc::kill(pg, 0) } == 0 {
            unsafe {
                let _ = libc::kill(pg, libc::SIGKILL);
            }
        }
    }
    signaled
}

#[cfg(not(unix))]
fn kill_pgids(_pgids: &[u32]) -> usize {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_ignores_empty_session_and_zero_pgid() {
        // 不应 panic，且不产生条目。
        register("", 123);
        register("sess", 0);
        assert_eq!(kill_session("sess"), 0);
    }

    #[test]
    fn kill_unknown_session_is_noop() {
        assert_eq!(kill_session("does-not-exist"), 0);
    }

    #[test]
    fn kill_session_consumes_entries() {
        // 用一个不太可能存活的高位 pgid：kill(-pgid, 0) 预期返回 ESRCH，
        // 因此 signaled 计数为 0，但条目仍应被移除（第二次调用返回 0）。
        let sid = format!("proc_reg_test_{}", uuid::Uuid::new_v4());
        register(&sid, 0x7FFF_FFF0);
        let _ = kill_session(&sid);
        assert_eq!(kill_session(&sid), 0);
    }
}
