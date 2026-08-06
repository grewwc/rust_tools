//! `/proc` 命令：展示当前正在运行的 session。
//!
//! 灵感来自 Unix 的 `/proc` 文件系统--把所有"活着"的 session 汇总在一张表里。
//!
//! 数据来源（三重探测，逐层兜底）：
//! 1. **PID 文件**（sessions 目录下的 `<id>.pid`）：新版本 `a` 启动时自动写入，
//!    退出时自动删除。最可靠。
//! 2. **`lsof` 扫描**：仅对未通过 PID 文件登记的 `a` 进程查询其打开的 `.sqlite`
//!    文件（`lsof -p`），而非递归扫描整个 sessions 目录。兜底旧版本 `a` 启动的
//!    session（它们不写 PID 文件），但对空闲等待输入的旧版本 session 可能漏报。
//! 3. **`pgrep` 计数**：统计名为 `a` 的进程总数，用于提示
//!    "有 N 个 a 进程在跑，但只识别出 M 个 session"。
//!
//! 注意：通过 `/bg`、`/suspend` 等挂起的 session **不算**活跃--它们的进程已退出，
//! 只是保存了状态供后续恢复。使用 `/sessions list` 查看所有已保存的 session。

use std::collections::{BTreeMap, BTreeSet};
use std::process;

use crate::ai::{driver::session_pid, history::SessionStore, types::App};

/// 合并后的活跃 session 记录。
struct ActiveSession {
    session_id: String,
    pid: i32,
    source: &'static str, // "pid-file" / "lsof"
}

pub fn try_handle_proc_command(app: &App, input: &str) -> Result<bool, Box<dyn std::error::Error>> {
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
    let mut by_sid: BTreeMap<String, ActiveSession> = BTreeMap::new();
    let mut registered_pids: BTreeSet<i32> = BTreeSet::new();

    // 1) PID 文件（扫描基目录及所有 *.sessions 子目录）
    for (sid, pid, alive) in session_pid::scan_all_session_pids(sessions_root)? {
        if alive {
            registered_pids.insert(pid);
            by_sid.entry(sid.clone()).or_insert(ActiveSession {
                session_id: sid,
                pid,
                source: "pid-file",
            });
        }
    }

    // 2) lsof 兜底（仅对未登记的 a 进程查询其打开的 .sqlite，避免 lsof +D 递归扫描）
    let a_pids = session_pid::list_a_pids();
    let total_a = a_pids.len().saturating_sub(1); // 减去自身
    let unregistered: Vec<i32> = a_pids
        .into_iter()
        .filter(|&p| p != current_pid && !registered_pids.contains(&p))
        .collect();
    for (sid, pid) in session_pid::discover_lsof_sessions(sessions_root, &unregistered) {
        by_sid.entry(sid.clone()).or_insert(ActiveSession {
            session_id: sid,
            pid,
            source: "lsof",
        });
    }

    // 排除当前进程自身--`a /proc` 是一次性查询，不是真正的活跃 session。
    let sessions: Vec<&ActiveSession> = by_sid.values().filter(|s| s.pid != current_pid).collect();
    let identified = sessions.len();

    if sessions.is_empty() {
        println!("No active sessions identified.");
        if total_a > 0 {
            println!(
                "  (but {total_a} `a` process(es) detected via pgrep - possibly started with an older version)"
            );
        }
        return Ok(true);
    }

    println!("Active sessions ({identified}):");
    println!();

    // 批量查询所有活跃 session 的 tty：单次 `ps` 取代逐进程 fork/exec。
    let active_pids: Vec<i32> = sessions.iter().map(|s| s.pid).collect();
    let tty_map = session_pid::tty_map_for_pids(&active_pids);

    // 仅读取活跃 session 的预览（summary + modified）。不要用 list_sessions()：
    // 它会打开全部已保存 session 的 .sqlite，并递归统计每个 session 的
    // assets/checkpoints 目录大小，而 /proc 只需要少数活跃 session 的文本预览。
    let previews: BTreeMap<String, (Option<String>, Option<String>)> = sessions
        .iter()
        .map(|s| {
            let (summary, modified) = store
                .read_session_preview(&s.session_id)
                .ok()
                .flatten()
                .unwrap_or((None, None));
            let modified = modified.map(|t| t.format("%Y-%m-%d %H:%M").to_string());
            (s.session_id.clone(), (summary, modified))
        })
        .collect();

    for s in &sessions {
        let tag = if *tty_map.get(&s.pid).unwrap_or(&false) {
            "interactive"
        } else {
            "background"
        };

        let (summary, modified) = previews.get(&s.session_id).cloned().unwrap_or((None, None));

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
        println!("Note: {diff} additional `a` process(es) running but session not identified");
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
    println!("Note: sessions suspended via /bg or /suspend are NOT shown here -");
    println!("      their processes have exited. Use /sessions list to see all saved sessions.");
    println!();
}
