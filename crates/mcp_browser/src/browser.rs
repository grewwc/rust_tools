//! Browser session lifecycle: lazily launch a controlled Chrome, reuse a single Page,
//! poll the CDP Handler in the background, and shut down cleanly on exit.
//!
//! Invariants:
//! - **The Handler must be polled continuously**, otherwise no CDP call (goto/click/...)
//!   makes progress. `launch()` solves this with a `tokio::spawn`'d
//!   `while handler.next().await` loop.
//! - Reuse a **single** Page; login state / multi-step flows rely on it.
//! - `launch` starts a new controlled Chrome (independent temp profile) and does not
//!   hijack the user's open windows; to attach a user-started `--remote-debugging`
//!   instance, use MCP_BROWSER_WS_URL.

use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::page::Page;
use futures_util::StreamExt;
use std::path::{Path, PathBuf};
use tokio::task::JoinHandle;

/// Default macOS Chrome executable path; overridable via MCP_BROWSER_CHROME.
const DEFAULT_CHROME: &str = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

/// Prefix for auto-generated temp profile dirs: `<temp>/mcp_browser-profile-<pid>`.
/// Shared by GC and launch to avoid naming drift.
const TEMP_PROFILE_PREFIX: &str = "mcp_browser-profile-";

/// Startup garbage collection: remove leftover temp profile dirs belonging to **dead
/// processes**.
///
/// Why: the host `a` usually kills this subprocess outright when a one-shot task ends
/// (SIGKILL cannot be caught), so `shutdown()`'s immediate cleanup never runs and temp
/// profiles linger. Signal handling cannot reliably cover this, so cleanup is instead
/// done by a scan on the next process startup: for each `mcp_browser-profile-<pid>`
/// dir, probe whether the pid is still alive with `kill(pid, 0)` and delete it if dead.
/// Self-healing, no signal handling needed.
///
/// Only auto-generated `mcp_browser-profile-*` dirs are reclaimed; dirs the user sets
/// via MCP_BROWSER_USER_DATA_DIR are not in this set and are naturally unaffected.
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
        // Skip ourselves (in theory no dir is created yet, but defensively), and only delete dirs of dead processes.
        if pid == me || process_alive(pid) {
            continue;
        }
        let _ = std::fs::remove_dir_all(entry.path());
    }
}

/// Probe whether a process is alive with `kill(pid, 0)`: sends no signal, only a
/// permission/existence check. Returns `true` when the process exists (or exists but
/// is inaccessible -- conservatively treated as alive so its dir is not deleted).
fn process_alive(pid: u32) -> bool {
    // SAFETY: kill(2) with signal 0 only performs an existence check and does not alter any process state.
    let ret = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if ret == 0 {
        return true;
    }
    // errno == EPERM: the process exists but we lack permission -> conservatively treat as alive. Only ESRCH means it is really gone.
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Clean up Chrome singleton locks in the profile dir (best-effort).
///
/// `SingletonLock`/`SingletonSocket`/`SingletonCookie` are Chrome's multi-instance
/// locks; if the previous controlled Chrome was killed abnormally (e.g. the MCP client
/// killed the subprocess on timeout), the locks linger and the next launch aborts with
/// `Failed to create ... SingletonLock: File exists (17)`. They are symlinks, and
/// `remove_file` removes the link itself without following it.
fn purge_singleton_locks(dir: &Path) {
    for name in ["SingletonLock", "SingletonSocket", "SingletonCookie"] {
        let _ = std::fs::remove_file(dir.join(name));
    }
}

/// A live browser session: controlled Browser + a single reused Page + the Handler polling task.
pub struct BrowserSession {
    pub browser: Browser,
    pub page: Page,
    handler_task: JoinHandle<()>,
    /// Auto-generated temp profile dir, cleaned up on shutdown; None when the user
    /// explicitly set one or in attach mode.
    temp_profile_dir: Option<PathBuf>,
    /// Pending-human-action marker: set (with its category) when a human verification
    /// is detected, cleared once wait_for_human confirms resolution, detection is empty,
    /// or navigate reaches a new page. While set, mutating tools (click/type/press_key)
    /// prepend [HUMAN_ACTION_PENDING] to their output so the model stops and hands the
    /// action to the user.
    pub pending_human: Option<String>,
}

/// Top-level session abstraction: controlled Chrome (CDP) vs. driving the user's already-open Chrome (AppleScript).
pub enum Session {
    Cdp(BrowserSession),
    AppleScript(crate::applescript::ApplescriptSession),
}

/// Driver mode selection, decided by the `MCP_BROWSER_DRIVER` environment variable.
///
/// - Default `applescript`: reuse the user's already-open Chrome (new tab, never quits the user's browser).
/// - `cdp`: launch a new controlled Chrome instance (historical behavior).
/// - When `MCP_BROWSER_WS_URL` is set, always `cdp` (explicitly attach an existing debugging-port instance).
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
            // Non-macOS platforms have no osascript, so default to a controlled instance
            // (combined with MCP_BROWSER_WS_URL it can attach a user-started debugging-port instance).
            _ => DriverMode::Cdp,
        }
    }

    pub fn is_applescript(&self) -> bool {
        matches!(self, DriverMode::AppleScript)
    }
}

impl BrowserSession {
    /// Lazily launch a controlled Chrome and open a blank page.
    ///
    /// Environment variables:
    /// - `MCP_BROWSER_WS_URL`: if set, attach an existing instance instead (`Browser::connect`).
    /// - `MCP_BROWSER_CHROME`: Chrome executable path (default see DEFAULT_CHROME).
    /// - `MCP_BROWSER_HEADLESS`: `0` (default) headed, good for login/interaction; `1` headless.
    /// - `MCP_BROWSER_USER_DATA_DIR`: explicit profile dir (persists login state); not
    ///   cleaned up, and sharing it across processes causes conflicts. When unset, each
    ///   process gets a unique temp dir that is removed on exit.
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

                // Profile dir: if explicitly set, reuse it (no cleanup, for persisting
                // login state); otherwise generate a unique temp dir per process that is
                // deleted on exit, avoiding multiple instances colliding on the same
                // fixed dir's SingletonLock (a chromiumoxide default-behavior trap).
                let (data_dir, temp) = match std::env::var("MCP_BROWSER_USER_DATA_DIR") {
                    Ok(d) if !d.trim().is_empty() => (PathBuf::from(d), None),
                    _ => {
                        let dir = std::env::temp_dir()
                            .join(format!("{TEMP_PROFILE_PREFIX}{}", std::process::id()));
                        (dir.clone(), Some(dir))
                    }
                };
                // Whether the dir is new or old, clear any leftover singleton locks before launching.
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

        // The Handler must be polled continuously, otherwise CDP calls make no progress.
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

    /// Close the session and stop the Handler polling task. Best-effort.
    ///
    /// Key protection: in attach mode (MCP_BROWSER_WS_URL, i.e. the user's own Chrome)
    /// never `close` the whole browser (that would quit the user's browser); only close
    /// the session tab we created. Only a self-launched controlled instance is closed whole.
    pub async fn shutdown(mut self) {
        if self.temp_profile_dir.is_some() {
            let _ = self.browser.close().await;
        } else {
            let _ = self.page.close().await;
        }
        self.handler_task.abort();
        // Only clean up the auto-generated temp profile; dirs the user explicitly set are kept (persisted login state).
        if let Some(dir) = self.temp_profile_dir.take() {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

/// Normalize MCP_BROWSER_WS_URL into the ws:// address chromiumoxide needs:
/// - `ws://` / `wss://` returned as-is;
/// - `http://host:port` or bare `host:port` automatically fetches `/json/version` for
///   `webSocketDebuggerUrl`, avoiding manual copying of a long ws address (used with
///   `--remote-debugging-port` on Windows/Linux, equivalent to "reusing the user's
///   already-open Chrome").
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
        // Take only the "host:port" part, ignoring any /devtools/... path the user may have pasted.
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

/// Minimal HTTP GET (only for local debugging endpoints like 127.0.0.1; zero new dependencies).
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

/// Lazily launch a session if none exists, then return a mutable reference to it.
pub async fn ensure_session(
    session: &mut Option<BrowserSession>,
) -> Result<&mut BrowserSession, String> {
    if session.is_none() {
        *session = Some(BrowserSession::launch().await?);
    }
    Ok(session.as_mut().expect("session just initialized"))
}
