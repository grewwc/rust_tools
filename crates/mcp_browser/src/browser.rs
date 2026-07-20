//! 浏览器会话生命周期：懒启动一个受控 Chrome、复用单个 Page、
//! 后台轮询 CDP Handler，并在退出时干净关闭。
//!
//! 不变量：
//! - **Handler 必须被持续轮询**，否则所有 CDP 调用（goto/click/...）都不会推进。
//!   `launch()` 里 `tokio::spawn` 一个 `while handler.next().await` 循环解决。
//! - 复用**单个** Page，登录态 / 多步流程靠它保持。
//! - `launch` 是新开一个受控 Chrome（独立临时 profile），不劫持用户已开窗口；
//!   若要 attach 用户手动开的 `--remote-debugging` 实例，用 MCP_BROWSER_WS_URL。

use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::page::Page;
use futures_util::StreamExt;
use std::path::{Path, PathBuf};
use tokio::task::JoinHandle;

/// 默认 macOS Chrome 可执行路径；可用 MCP_BROWSER_CHROME 覆盖。
const DEFAULT_CHROME: &str = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

/// 自动生成的临时 profile 目录前缀：`<temp>/mcp_browser-profile-<pid>`。
/// GC 与 launch 共用此常量，避免命名漂移。
const TEMP_PROFILE_PREFIX: &str = "mcp_browser-profile-";

/// 启动时垃圾回收：删除属于**已死进程**的残留临时 profile 目录。
///
/// 为何需要：宿主 `a` 在一次性任务结束时通常直接 kill 本子进程（SIGKILL 不可
/// 捕获），`shutdown()` 的即时清理来不及跑，临时 profile 会残留。信号处理无法
/// 可靠兜底，故改由“下一个进程启动时”扫描回收：对每个 `mcp_browser-profile-<pid>`
/// 目录，用 `kill(pid, 0)` 探测该 pid 是否存活，已死则删除。自愈、无需信号处理。
///
/// 只回收自动生成的 `mcp_browser-profile-*`；用户经 MCP_BROWSER_USER_DATA_DIR
/// 指定的目录不在此列，天然不受影响。
pub fn gc_stale_profiles() {
    let base = std::env::temp_dir();
    let Ok(entries) = std::fs::read_dir(&base) else {
        return;
    };
    let me = std::process::id();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(pid_str) = name.strip_prefix(TEMP_PROFILE_PREFIX) else {
            continue;
        };
        let Ok(pid) = pid_str.parse::<u32>() else {
            continue;
        };
        // 跳过自己（理论上此刻还没建目录，但防御性判断），只删已死进程的目录。
        if pid == me || process_alive(pid) {
            continue;
        }
        let _ = std::fs::remove_dir_all(entry.path());
    }
}

/// 用 `kill(pid, 0)` 探测进程是否存活：不发送信号，仅做权限/存在性检查。
/// 返回 `true` 表示进程存在（或存在但无权限，此时保守视为存活、不删其目录）。
fn process_alive(pid: u32) -> bool {
    // SAFETY: kill(2) with signal 0 只做存在性检查，不改动任何进程状态。
    let ret = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if ret == 0 {
        return true;
    }
    // errno == EPERM：进程存在但我们无权限 → 保守当作存活。ESRCH 才是真的没了。
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// 清理 profile 目录里的 Chrome 单例锁（best-effort）。
///
/// `SingletonLock`/`SingletonSocket`/`SingletonCookie` 是 Chrome 防多开的锁；
/// 若上一个受控 Chrome 被非正常杀死（如 MCP 客户端超时 kill 子进程），锁会残留，
/// 导致下次启动报 `Failed to create ... SingletonLock: File exists (17)` 而中止。
/// 它们都是 symlink，`remove_file` 删除链接本身、不跟随。
fn purge_singleton_locks(dir: &Path) {
    for name in ["SingletonLock", "SingletonSocket", "SingletonCookie"] {
        let _ = std::fs::remove_file(dir.join(name));
    }
}

/// 一个存活的浏览器会话：受控 Browser + 复用的单个 Page + Handler 轮询任务。
pub struct BrowserSession {
    pub browser: Browser,
    pub page: Page,
    handler_task: JoinHandle<()>,
    /// 自动生成的临时 profile 目录，shutdown 时清理；用户显式指定或 attach 模式为 None。
    temp_profile_dir: Option<PathBuf>,
}

impl BrowserSession {
    /// 懒启动一个受控 Chrome 并打开一个空白页。
    ///
    /// 环境变量：
    /// - `MCP_BROWSER_WS_URL`：若设置，改为 attach 已有实例（`Browser::connect`）。
    /// - `MCP_BROWSER_CHROME`：Chrome 可执行路径（默认见 DEFAULT_CHROME）。
    /// - `MCP_BROWSER_HEADLESS`：`0`（默认）有头，利于登录/交互；`1` 无头。
    /// - `MCP_BROWSER_USER_DATA_DIR`：显式 profile 目录（持久化登录态）；不清理、
    ///   多进程共用会冲突。未设时每进程用一个唯一临时目录，退出时清理。
    pub async fn launch() -> Result<Self, String> {
        let (browser, mut handler, temp_profile_dir) =
            if let Ok(ws) = std::env::var("MCP_BROWSER_WS_URL") {
                let (browser, handler) = Browser::connect(ws).await.map_err(|e| {
                    format!("failed to connect to browser at MCP_BROWSER_WS_URL: {e}")
                })?;
                (browser, handler, None)
            } else {
                let chrome = std::env::var("MCP_BROWSER_CHROME")
                    .unwrap_or_else(|_| DEFAULT_CHROME.to_string());
                let headless = std::env::var("MCP_BROWSER_HEADLESS")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);

                // profile 目录：显式指定则复用（不清理，供持久化登录态）；
                // 否则每进程生成一个唯一临时目录，退出时删除，避免多实例撞
                // 同一固定目录的 SingletonLock（chromiumoxide 默认行为的坑）。
                let (data_dir, temp) = match std::env::var("MCP_BROWSER_USER_DATA_DIR") {
                    Ok(d) if !d.trim().is_empty() => (PathBuf::from(d), None),
                    _ => {
                        let dir = std::env::temp_dir()
                            .join(format!("{TEMP_PROFILE_PREFIX}{}", std::process::id()));
                        (dir.clone(), Some(dir))
                    }
                };
                // 无论新旧目录，先清可能残留的单例锁再启动。
                let _ = std::fs::create_dir_all(&data_dir);
                purge_singleton_locks(&data_dir);

                let mut builder = BrowserConfig::builder()
                    .chrome_executable(&chrome)
                    .user_data_dir(&data_dir);
                builder = if headless {
                    builder.new_headless_mode()
                } else {
                    builder.with_head()
                };
                let config = builder
                    .build()
                    .map_err(|e| format!("failed to build browser config: {e}"))?;
                let (browser, handler) = Browser::launch(config)
                    .await
                    .map_err(|e| format!("failed to launch Chrome at '{chrome}': {e}"))?;
                (browser, handler, temp)
            };

        // Handler 必须持续轮询，否则 CDP 调用不会推进。
        let handler_task = tokio::spawn(async move { while handler.next().await.is_some() {} });

        let page = browser
            .new_page("about:blank")
            .await
            .map_err(|e| format!("failed to open initial page: {e}"))?;

        Ok(BrowserSession {
            browser,
            page,
            handler_task,
            temp_profile_dir,
        })
    }

    /// 关闭浏览器并中止 Handler 轮询任务。best-effort。
    pub async fn shutdown(mut self) {
        let _ = self.browser.close().await;
        self.handler_task.abort();
        // 仅清理自动生成的临时 profile；用户显式指定的目录保留（持久化登录态）。
        if let Some(dir) = self.temp_profile_dir.take() {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

/// 若尚无会话则懒启动一个，然后返回可变引用。
pub async fn ensure_session(
    session: &mut Option<BrowserSession>,
) -> Result<&mut BrowserSession, String> {
    if session.is_none() {
        *session = Some(BrowserSession::launch().await?);
    }
    Ok(session.as_mut().expect("session just initialized"))
}
