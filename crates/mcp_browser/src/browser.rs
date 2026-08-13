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
    /// 待人工操作标记：检测到人机校验后置位（分类），wait_for_human 确认解决、
    /// 检测为空或 navigate 到新页面后清除。置位期间改动类工具（click/type/press_key）
    /// 会在输出前加 [HUMAN_ACTION_PENDING] 提醒，让模型停下来把操作交给用户。
    pub pending_human: Option<String>,
}

/// 会话顶层抽象：受控 Chrome（CDP）vs. 驱动用户已打开的 Chrome（AppleScript）。
pub enum Session {
    Cdp(BrowserSession),
    AppleScript(crate::applescript::ApplescriptSession),
}

/// 驱动模式选择，由 `MCP_BROWSER_DRIVER` 环境变量决定。
///
/// - 默认 `applescript`：复用用户已打开的 Chrome（新开标签页，绝不退出用户浏览器）。
/// - `cdp`：启动一个受控的新 Chrome 实例（历史行为）。
/// - 设置 `MCP_BROWSER_WS_URL` 时恒为 `cdp`（显式 attach 已有调试端口实例）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverMode {
    AppleScript,
    Cdp,
}

impl DriverMode {
    pub fn from_env() -> Self {
        if std::env::var("MCP_BROWSER_WS_URL").ok().is_some() {
            return DriverMode::Cdp;
        }
        match std::env::var("MCP_BROWSER_DRIVER").ok().as_deref() {
            Some("cdp") | Some("CDP") => DriverMode::Cdp,
            _ if cfg!(target_os = "macos") => DriverMode::AppleScript,
            // 非 macOS 平台没有 osascript，默认退到受控实例
            // （配合 MCP_BROWSER_WS_URL 可 attach 用户自启的调试端口实例）。
            _ => DriverMode::Cdp,
        }
    }

    pub fn is_applescript(&self) -> bool {
        matches!(self, DriverMode::AppleScript)
    }
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
                let ws = resolve_ws_url(&ws).await.map_err(|e| {
                    format!("failed to resolve MCP_BROWSER_WS_URL ({ws}): {e}")
                })?;
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
            pending_human: None,
        })
    }

    /// 关闭会话并中止 Handler 轮询任务。best-effort。
    ///
    /// 关键保护：attach 模式（MCP_BROWSER_WS_URL，也就是用户自己的 Chrome）
    /// 绝不对浏览器整体 close（那会退出用户的浏览器），只关我们创建的会话
    /// 标签页；只有自启的受控实例才整体 close。
    pub async fn shutdown(mut self) {
        if self.temp_profile_dir.is_some() {
            let _ = self.browser.close().await;
        } else {
            let _ = self.page.close().await;
        }
        self.handler_task.abort();
        // 仅清理自动生成的临时 profile；用户显式指定的目录保留（持久化登录态）。
        if let Some(dir) = self.temp_profile_dir.take() {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

/// 把 MCP_BROWSER_WS_URL 统一成 chromiumoxide 需要的 ws:// 地址：
/// - `ws://` / `wss://` 原样返回；
/// - `http://host:port` 或裸 `host:port` 则自动请求 `/json/version` 取
///   `webSocketDebuggerUrl`，省去手工复制一长串 ws 地址（Windows/Linux 上
///   配合 `--remote-debugging-port` 使用，等价于"复用用户已开的 Chrome"）。
async fn resolve_ws_url(input: &str) -> Result<String, String> {
    let input = input.trim();
    if input.starts_with("ws://") || input.starts_with("wss://") {
        return Ok(input.to_string());
    }
    let (host, port) = {
        let base = input
            .strip_prefix("http://")
            .or_else(|| input.strip_prefix("https://"))
            .unwrap_or(input);
        // 只取 "host:port" 部分，忽略用户可能贴上的 /devtools/... 路径。
        let hp = base.split('/').next().unwrap_or(base);
        match hp.strip_prefix('[').and_then(|h| h.strip_suffix(']')).unwrap_or(hp).rsplit_once(':') {
            Some((h, p)) if !h.is_empty() && !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()) => {
                (h.to_string(), p.to_string())
            }
            _ => {
                return Err(format!(
                    "无法解析主机端口（应为 host:port 或 http://host:port）: {input}"
                ))
            }
        }
    };
    let resp = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        http_get(&host, &port, "/json/version"),
    )
    .await
    .map_err(|_| {
        format!(
            "连接 {host}:{port} 超时——Chrome 是否以 --remote-debugging-port={port} 启动？"
        )
    })??;
    let (status, body) = resp;
    if !status.starts_with("200") {
        return Err(format!("{host}:{port}/json/version 返回 {status}"));
    }
    let v: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("解析调试端点响应失败: {e}"))?;
    let ws = v["webSocketDebuggerUrl"]
        .as_str()
        .ok_or_else(|| "调试端点响应里没有 webSocketDebuggerUrl".to_string())?;
    Ok(ws.to_string())
}

/// 极简 HTTP GET（只面向 127.0.0.1 这类本地调试端点，零新增依赖）。
async fn http_get(host: &str, port: &str, path: &str) -> Result<(String, String), String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let port: u16 = port.parse().map_err(|_| format!("端口不合法: {port}"))?;
    let mut s = tokio::net::TcpStream::connect((host, port))
        .await
        .map_err(|e| format!("连接 {host}:{port} 失败: {e}"))?;
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\nUser-Agent: mcp_browser\r\n\r\n"
    );
    s.write_all(req.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).await.map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&buf).to_string();
    let status = text
        .split_whitespace()
        .nth(1)
        .unwrap_or("")
        .to_string();
    let body = text.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
    Ok((status, body))
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
