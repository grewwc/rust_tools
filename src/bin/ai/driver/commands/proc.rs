//! `/proc` 命令：展示当前正在运行的 session。
//!
//! 灵感来自 Unix 的 `/proc` 文件系统——把所有"活着"的 session 汇总在一张表里。
//!
//! 数据来源（三重探测，逐层兜底）：
//! 1. **PID 文件**（sessions 目录下的 `<id>.pid`）：新版本 `a` 启动时自动写入，
//!    退出时自动删除。最可靠。
//! 2. **`lsof` 扫描**：对 sessions 目录做 `lsof +D`，发现正在读写 `.sqlite` 文件的
//!    进程。兜底旧版本 `a` 启动的 session（它们不写 PID 文件），但对空闲等待
//!    输入的旧版本 session 可能漏报。
//! 3. **`pgrep` 计数**：统计名为 `a` 的进程总数，用于提示
//!    "有 N 个 a 进程在跑，但只识别出 M 个 session"。
//!
//! 注意：通过 `/bg`、`/suspend` 等挂起的 session **不算**活跃——它们的进程已退出，
//! 只是保存了状态供后续恢复。使用 `/sessions list` 查看所有已保存的 session。

use std::process;

use crate::ai::{
    driver::session_pid,
    history::SessionStore,
    types::App,
};

/// 从 SessionStore 查找指定 session 的摘要信息。
fn lookup_session_info(
    store: &SessionStore,
    session_id: &str,
) -> Option<(Option<String>, Option<String>)> {
    store
        .list_sessions()
        .ok()?
        .into_iter()
        .find(|s| s.id == session_id)
        .map(|s| {
            let summary = s.summary.filter(|v| !v.is_empty());
            let modified = s
                .modified_local
                .map(|t| t.format("%Y-%m-%d %H:%M").to_string());
            (summary, modified)
        })
}

/// 合并后的活跃 session 记录。
struct ActiveSession {
    session_id: String,
    pid: i32,
    source: &'static str, // "pid-file" / "lsof"
}

pub fn try_handle_proc_command(
    app: &App,
    input: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let normalized = if let Some(rest) = trimmed.strip_prefix('/') {
        rest
    } else if let Some(rest) = trimmed.strip_prefix(':') {
        rest
    } else {
        return Ok(false);
    };
    let mut parts = normalized.split_whitespace();
    let Some(cmd) = parts.next() else {
        return Ok(false);
    };
    if cmd != "proc" {
        return Ok(false);
    }
    let action = parts.next().unwrap_or("list");
    if matches!(action, "help" | "h") {
        print_proc_help();
        return Ok(true);
    }
    if !matches!(action, "list" | "ls" | "") {
        println!("Unknown /proc subcommand: {action}");
        println!("Run /proc help for usage.");
        return Ok(true);
    }

    let store = SessionStore::new(app.config.history_file.as_path());
    let _ = store.ensure_root_dir();
    let current_pid = process::id() as i32;
    let sessions_root = store.sessions_root();

    // ---- 收集活跃 session（三重探测）----
    let mut by_sid: std::collections::BTreeMap<String, ActiveSession> = std::collections::BTreeMap::new();

    // 1) PID 文件（扫描基目录及所有 *.sessions 子目录）
    for (sid, pid, alive) in session_pid::scan_all_session_pids(sessions_root)? {
        if alive {
            by_sid.entry(sid.clone()).or_insert(ActiveSession {
                session_id: sid,
                pid,
                source: "pid-file",
            });
        }
    }

    // 2) lsof 兜底
    for (sid, pid) in session_pid::discover_lsof_sessions(sessions_root) {
        by_sid.entry(sid.clone()).or_insert(ActiveSession {
            session_id: sid,
            pid,
            source: "lsof",
        });
    }

    // 排除当前进程自身——`a /proc` 是一次性查询，不是真正的活跃 session。
    let sessions: Vec<&ActiveSession> = by_sid
        .values()
        .filter(|s| s.pid != current_pid)
        .collect();
    let identified = sessions.len();

    // 3) pgrep 计数（减去自身）
    let total_a = session_pid::count_a_processes().saturating_sub(1);

    if sessions.is_empty() {
        println!("No active sessions identified.");
        if total_a > 0 {
            println!(
                "  (but {total_a} `a` process(es) detected via pgrep — possibly started with an older version)"
            );
        }
        return Ok(true);
    }

    println!("Active sessions ({identified}):");
    println!();

    for s in &sessions {
        let tag = if session_pid::process_has_tty(s.pid) {
            "interactive"
        } else {
            "background"
        };

        let (summary, modified) =
            lookup_session_info(&store, &s.session_id).unwrap_or((None, None));

        println!("  [{tag:<11}]  pid={:<8}  session={}", s.pid, s.session_id);
        if let Some(m) = &modified {
            println!("                modified: {m}");
        }
        println!(
            "                summary : {}",
            summary.as_deref().unwrap_or("-")
        );
        if s.source == "lsof" {
            println!("                source  : lsof (no pid-file, possibly older version)");
        }
        println!();
    }

    // 提示未识别的进程
    if total_a > identified {
        let diff = total_a - identified;
        println!(
            "Note: {diff} additional `a` process(es) running but session not identified"
        );
        println!("      (likely started with an older version without pid-file support)");
        println!();
    }

    Ok(true)
}

fn print_proc_help() {
    println!("/proc commands:");
    println!();
    println!("  /proc                     show running sessions (interactive + background)");
    println!("  /proc list                same as /proc");
    println!("  /proc help                show this help message");
    println!();
    println!("Note: sessions suspended via /bg or /suspend are NOT shown here —");
    println!("      their processes have exited. Use /sessions list to see all saved sessions.");
    println!();
}
